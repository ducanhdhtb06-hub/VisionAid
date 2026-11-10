import concurrent.futures
import json
import os
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from loi_do_sau.truc_tiep.goi_tin import dong_goi
from loi_do_sau.truc_tiep.may_chu import pcm_sang_wav, tao_app

MA = 'ma-bi-mat-123'
CAU = 'Dừng lại. Xe máy, 2 mét'
T0 = 1_000_000.0


def jpeg(i=0):
    img = np.full((360, 640, 3), (i * 40) % 255, np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


class EngineGia:
    """Nói CAU ở lần gọi thứ `noi_o`; chậm `cham` giây mỗi lần; ném lỗi ở lần gọi `loi_o`."""
    def __init__(self, noi_o=3, cham=0.0, loi_o=None):
        self.goi, self.noi_o, self.cham, self.loi_o = [], noi_o, cham, loi_o
        self.noi_lai = 0

    def step(self, img, t):
        self.goi.append(t)
        time.sleep(self.cham)
        if len(self.goi) == self.loi_o:
            raise RuntimeError('CUDA out of memory (giả)')
        if len(self.goi) == self.noi_o:
            return CAU, pcm_sang_wav(np.zeros(1600, np.int16))
        return None, None

    def cho_noi_lai(self):
        self.noi_lai += 1


def ket_noi(tmp_path, engine):
    return TestClient(tao_app(engine, MA, str(tmp_path)))


def gui(ws, i):
    ws.send_bytes(dong_goi(1_000_000.0 + i * 100, i, jpeg(i)))


def test_moi_khung_chay_tren_cung_mot_luong_voi_lam_nong(tmp_path):
    # đo 2026-11-06: lần chạy đầu trên một luồng mới mất ~4,3 s ở mọi model (CUDA có trạng thái theo luồng), luồng đã chạy
    # chỉ 130 ms. Bể luồng có thể đổi luồng bất kỳ lúc nào, ví dụ khi iPhone nối lại: hình đứng hơn 4 s giữa lúc đang đi.
    class EngineGhiLuong(EngineGia):
        def __init__(self):
            super().__init__()
            self.luong = set()

        def step(self, img, t):
            self.luong.add(threading.get_ident())
            return None, None

    eng = EngineGhiLuong()
    luong = concurrent.futures.ThreadPoolExecutor(1)
    da_nong = luong.submit(threading.get_ident).result()
    with TestClient(tao_app(eng, MA, str(tmp_path), luong=luong)) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        for i in range(1, 7):
            gui(ws, i)
            ws.receive_json()
    assert eng.luong == {da_nong}


def test_sai_ma_bi_tu_choi(tmp_path):
    eng = EngineGia()
    with ket_noi(tmp_path, eng) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect('/ws?k=sai') as ws:
                ws.send_bytes(dong_goi(1.0, 1, jpeg()))
                ws.receive_json()
    assert eng.goi == []


def test_tra_cau_va_wav(tmp_path):
    with ket_noi(tmp_path, EngineGia(noi_o=3)) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        tin = []
        for i in (1, 2, 3):
            gui(ws, i)
            tin.append(ws.receive_json())
        assert [m['id'] for m in tin] == [1, 2, 3]
        assert tin[0]['cau'] is None and tin[2]['cau'] == CAU
        assert tin[2]['t_chup'] == 1_000_300.0
        assert ws.receive_bytes()[:4] == b'RIFF'


def test_chi_xu_ly_khung_moi_nhat(tmp_path):
    eng = EngineGia(noi_o=99, cham=0.3)
    with ket_noi(tmp_path, eng) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        for i in range(1, 6):
            gui(ws, i)
        ids = [ws.receive_json()['id'], ws.receive_json()['id']]
    assert ids == [1, 5]
    assert len(eng.goi) == 2
    log = json.load(open(tmp_path / 'log.json'))
    bo = {k['id']: k['bo_qua'] for k in log['khung']}
    assert bo == {1: False, 2: True, 3: True, 4: True, 5: False}


def test_bao_bo_cho_noi_lai(tmp_path):
    eng = EngineGia()
    with ket_noi(tmp_path, eng) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        ws.send_text(json.dumps({'phan_hoi': 'da_phat', 'id': 3, 'tre_ms': 400}))
        ws.send_text(json.dumps({'phan_hoi': 'bo', 'id': 3, 'tre_ms': 1800}))
        # bị cắt / bị thay: đã có câu mới hơn thế chỗ -> KHÔNG nói lại (nói lại gây vòng lặp "Dừng lại" mỗi 0,1 s)
        ws.send_text(json.dumps({'phan_hoi': 'bi_cat', 'id': 4, 'tre_ms': 300}))
        ws.send_text(json.dumps({'phan_hoi': 'thay', 'id': 5, 'tre_ms': 300}))
        gui(ws, 1)
        ws.receive_json()          # đồng bộ: mọi tin trước đó đã được xử lý
    assert eng.noi_lai == 1
    log = json.load(open(tmp_path / 'log.json'))
    assert [p['phan_hoi'] for p in log['phan_hoi']] == ['da_phat', 'bo', 'bi_cat', 'thay']


def test_engine_loi_khong_chet(tmp_path):
    with ket_noi(tmp_path, EngineGia(noi_o=99, loi_o=2)) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        gui(ws, 1)
        assert 'loi' not in ws.receive_json()
        gui(ws, 2)
        m = ws.receive_json()
        assert m['id'] == 2 and 'CUDA' in m['loi']
        gui(ws, 3)
        assert ws.receive_json()['id'] == 3


def test_ghi_phien(tmp_path):
    with ket_noi(tmp_path, EngineGia(noi_o=2)) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        for i in (1, 2, 3, 4):
            gui(ws, i)
            ws.receive_json()
            if i == 2:
                ws.receive_bytes()
    cap = cv2.VideoCapture(str(tmp_path / 'khung.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (640, 360)
    log = json.load(open(tmp_path / 'log.json'))
    assert set(log) >= {'bat_dau', 'khung', 'phan_hoi'}
    k2 = log['khung'][1]
    assert set(k2) >= {'id', 't_chup', 't_nhan', 'ms_xu_ly', 'cau', 'bo_qua'}
    assert k2['cau'] == CAU
    assert os.path.exists(tmp_path / 'cau' / '2.wav')


def test_noi_lai_thoi_gian_lien_tuc(tmp_path):
    # 4G rớt rồi nối lại: thời gian đưa cho engine phải tiếp tục tăng (Speaker/tracker giữ mốc cũ)
    eng = EngineGia(noi_o=99)
    with ket_noi(tmp_path, eng) as c:
        with c.websocket_connect(f'/ws?k={MA}') as ws:
            ws.send_bytes(dong_goi(T0, 1, jpeg()))
            ws.receive_json()
            ws.send_bytes(dong_goi(T0 + 100_000, 2, jpeg()))
            ws.receive_json()
        with c.websocket_connect(f'/ws?k={MA}') as ws:
            ws.send_bytes(dong_goi(T0 + 160_000, 3, jpeg()))
            ws.receive_json()
    assert eng.goi == pytest.approx([0.0, 100.0, 160.0])


def test_noi_lai_khong_mat_ban_ghi(tmp_path):
    with ket_noi(tmp_path, EngineGia(noi_o=99)) as c:
        for lan in range(2):
            with c.websocket_connect(f'/ws?k={MA}') as ws:
                for i in range(3):
                    ws.send_bytes(dong_goi(T0 + (lan * 3 + i) * 100, lan * 3 + i + 1, jpeg(i)))
                    ws.receive_json()
    cap = cv2.VideoCapture(str(tmp_path / 'khung.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    log = json.load(open(tmp_path / 'log.json'))
    assert [k['id'] for k in log['khung']] == [1, 2, 3, 4, 5, 6]


def test_ket_noi_moi_dong_ket_noi_cu(tmp_path):
    with ket_noi(tmp_path, EngineGia(noi_o=99)) as c:
        cu = c.websocket_connect(f'/ws?k={MA}').__enter__()
        cu.send_bytes(dong_goi(T0, 1, jpeg()))
        cu.receive_json()
        with c.websocket_connect(f'/ws?k={MA}') as moi:
            moi.send_bytes(dong_goi(T0 + 100, 2, jpeg()))
            assert moi.receive_json()['id'] == 2
            with pytest.raises(WebSocketDisconnect):
                cu.receive_json()
        cu.__exit__(None, None, None)    # phía test đóng nốt (uvicorn thật tự cắt TCP)


def test_sai_ma_dong_1008(tmp_path):
    # trình duyệt chỉ thấy mã 1008 nếu máy chủ accept rồi mới đóng (đóng trước accept -> 403/1006, trang thử lại mãi)
    with ket_noi(tmp_path, EngineGia()) as c, c.websocket_connect('/ws?k=sai') as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1008


def test_tac_vu_chet_thi_dong_ket_noi(tmp_path, monkeypatch):
    # lỗi bất ngờ ngoài engine (ví dụ đầy đĩa khi ghi wav) không được để phiên "sống" mà im lặng
    from loi_do_sau.truc_tiep import may_chu

    def hong(*a, **k):
        raise OSError('No space left on device')
    monkeypatch.setattr(may_chu.Phien, 'xong', hong)
    with ket_noi(tmp_path, EngineGia()) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        ws.send_bytes(dong_goi(T0, 1, jpeg()))
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1011


def test_cho_noi_lai_ap_dung_o_khung_sau():
    # cho_noi_lai chạy ở luồng sự kiện, step ở luồng khác: chỉ đặt cờ, step sau mới xóa trí nhớ Speaker
    from types import SimpleNamespace
    from loi_do_sau.truc_tiep.may_chu import EngineThat
    sp = SimpleNamespace(last='cau cu', busy_until=99.0, events=[])
    thay = []
    nv = SimpleNamespace(speaker=sp, step=lambda img, t: thay.append((sp.last, sp.busy_until)) or None)
    e = EngineThat.__new__(EngineThat)
    e.nv, e.lam_xau = nv, None
    e.cho_noi_lai()
    assert sp.last == 'cau cu'                     # chưa đụng tới Speaker
    e.step(np.zeros((360, 640, 3), np.uint8), 1.0)
    assert thay == [(None, -1.0)]                  # step thấy Speaker đã được xóa


def _so_khung_khi_doc_duoc(f, cho_s=3.0):
    # bắt tay đóng của TestClient không đợi khối finally phía máy chủ: chờ tới khi file đọc được
    het = time.time() + cho_s
    while time.time() < het:
        cap = cv2.VideoCapture(str(f))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if n:
            return n
        time.sleep(0.05)
    return 0


def test_ngat_ket_noi_la_xem_duoc_video(tmp_path):
    # đo thật 25/10: bấm Dừng trên iPhone xong khung.mp4 vẫn hỏng ("moov atom not found") vì chỉ đóng khi tắt máy chủ
    with ket_noi(tmp_path, EngineGia(noi_o=99)) as c:
        with c.websocket_connect(f'/ws?k={MA}') as ws:
            for i in range(1, 4):
                ws.send_bytes(dong_goi(T0 + i * 100, i, jpeg(i)))
                ws.receive_json()
        assert _so_khung_khi_doc_duoc(tmp_path / 'khung.mp4') == 3      # máy chủ VẪN chạy
        with c.websocket_connect(f'/ws?k={MA}') as ws:                  # nối lại: video nối tiếp, không ghi đè
            for i in range(4, 6):
                ws.send_bytes(dong_goi(T0 + i * 100, i, jpeg(i)))
                ws.receive_json()
        het = time.time() + 3
        while _so_khung_khi_doc_duoc(tmp_path / 'khung.mp4') != 5 and time.time() < het:
            time.sleep(0.05)
        assert _so_khung_khi_doc_duoc(tmp_path / 'khung.mp4') == 5


def test_lam_xau_nhan_khoang_cach_thoi_gian_that():
    # nhòe giả lập phải theo thời gian thật giữa hai khung (4G: ~0,5 s), không mặc định 0,1 s
    from types import SimpleNamespace
    from loi_do_sau.truc_tiep.may_chu import EngineThat
    nhan = []
    e = EngineThat.__new__(EngineThat)
    e.nv = SimpleNamespace(speaker=SimpleNamespace(last=None, busy_until=0, events=[]), step=lambda img, t: None)
    e.lam_xau = lambda img, dt=0.1: nhan.append(round(dt, 3)) or img
    e.can_noi_lai = False
    for t in (10.0, 10.5, 10.6):
        e.step(np.zeros((360, 640, 3), np.uint8), t)
    assert nhan == [0.1, 0.5, 0.1]


# ---------- Thử module dẫn theo vạch bằng iPhone (kế hoạch 2026-10-25-dan-theo-vach-sau-ra-soat, Task 1) ----------

def test_tao_app_kich_thuoc_4_3(tmp_path):
    """--dan-vach: ảnh vào engine là 640x480 (giống FRAMESIZE_VGA của firmware), không phải 640x360."""
    eng = EngineGia(noi_o=99)
    nhan = []
    eng.step = lambda img, t: (nhan.append(img.shape[:2]), (None, None))[1]
    with TestClient(tao_app(eng, MA, str(tmp_path), kich_thuoc=(640, 480))) as c, \
            c.websocket_connect(f'/ws?k={MA}') as ws:
        gui(ws, 1)
        ws.receive_json()
    assert nhan == [(480, 640)]


def _can_giong_doc():
    """Bỏ qua test cần file giọng đọc (voice/, sinh bằng tools/gen_vi_voice.py, không đưa lên git) khi máy chưa có."""
    from loi_do_sau.truc_tiep.may_chu import wav_cua_cau
    if wav_cua_cau('Dịch sang phải') is None:
        pytest.skip('chưa có giọng đọc voice/ (tools/gen_vi_voice.py)')


def test_wav_cua_cau_co_va_khong():
    _can_giong_doc()
    from loi_do_sau.truc_tiep.may_chu import wav_cua_cau
    assert wav_cua_cau('Dịch sang phải')[:4] == b'RIFF'
    assert wav_cua_cau('Câu này không có trong bảng giọng nói') is None


def test_engine_dan_vach_bat_cau_va_dong_ho_ao():
    """Engine chạy đúng đường hệ thống: bắt câu process_frame phát (play_voice_text) và cho nó thấy giờ ảo = t."""
    _can_giong_doc()
    from loi_do_sau.truc_tiep.may_chu import EngineDanVach
    import workflow_blindpath as wb  # pyright: ignore[reportMissingImports]

    class NavGia:
        dan_vach = None

        def __init__(self):
            self.gio = []

        def process_frame(self, img):
            self.gio.append(wb.time.time())
            wb.play_voice_text('Dịch sang phải')

    nav = NavGia()
    eng = EngineDanVach(navigator=nav)
    img = np.zeros((480, 640, 3), np.uint8)
    eng.step(img, 0.0)
    cau, wav = eng.step(img, 2.5)
    assert cau == 'Dịch sang phải' and wav[:4] == b'RIFF'
    assert nav.gio[1] - nav.gio[0] == pytest.approx(2.5)


def test_engine_dan_vach_bao_khi_thieu_wav(capsys):
    """Câu không có wav thì iPhone chỉ hiện chữ, người đeo không nghe: máy chủ phải in cảnh báo để biết."""
    from loi_do_sau.truc_tiep.may_chu import EngineDanVach
    import workflow_blindpath as wb  # pyright: ignore[reportMissingImports]

    class NavGia:
        dan_vach = None

        def process_frame(self, img):
            wb.play_voice_text('Câu lạ không có giọng')

    cau, wav = EngineDanVach(navigator=NavGia()).step(np.zeros((480, 640, 3), np.uint8), 0.0)
    assert cau == 'Câu lạ không có giọng' and wav is None
    assert 'không có wav' in capsys.readouterr().out


def test_engine_dan_vach_lam_nong_khong_de_lai_dau_vet():
    """Không làm nóng thì khung đầu mất 5,9 s, câu đầu của buổi thử tới muộn. Nhưng làm nóng không được để lại câu nói
    hay trạng thái dẫn đường: câu đầu của buổi thử vẫn phải nói đúng."""
    from loi_do_sau.truc_tiep.may_chu import EngineDanVach
    import workflow_blindpath as wb  # pyright: ignore[reportMissingImports]

    class NeGia:
        H, W = 360, 480

        def __init__(self):
            self.goi = []

        def dvh(self, img, co=None):
            self.goi.append(co)

    class NavGia:
        dan_vach = None

        def __init__(self):
            self.ne_vat_can, self.khung, self.lan_reset = NeGia(), [], 0

        def process_frame(self, img):
            self.khung.append(img.shape)
            wb.play_voice_text('Chưa thấy vạch dẫn đường.')

        def reset(self):
            self.lan_reset += 1

    nav = NavGia()
    eng = EngineDanVach(navigator=nav)
    eng.lam_nong()
    assert nav.khung == [(480, 640, 3)] and nav.ne_vat_can.goi == [(360, 480)] and nav.lan_reset == 1
    assert eng.da_noi == []
    nav.process_frame = lambda img: wb.play_voice_text('Dịch sang phải')
    assert eng.step(np.zeros((480, 640, 3), np.uint8), 0.0)[0] == 'Dịch sang phải'


def test_may_chu_bo_qua_tin_lenh(tmp_path):
    # trang iPhone gửi {"lenh": ...} cho cầu nối; máy chủ thử (may_chu) phải bỏ qua, phiên vẫn chạy
    with ket_noi(tmp_path, EngineGia(noi_o=2)) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        ws.send_text(json.dumps({'lenh': 'bat_dau'}))
        gui(ws, 0)
        gui(ws, 1)
        assert ws.receive_json()['id'] in (0, 1)


def test_ma_qr_mo_app(tmp_path):
    """Chế độ --dan-vach / --noi-thang: "/" là app VisionAid, "/thu" là trang thử cũ."""
    c = ket_noi(tmp_path, EngineGia())
    assert 'VisionAid' in c.get('/').text
    assert 'Thử kính' in c.get('/thu').text


# ---------- nút "Hỏi" của app: câu hỏi chữ + khung mới nhất -> Gemini -> câu trả lời + WAV ----------
class HoiGia:
    """Thay Gemini: ghi lại (ảnh, câu hỏi), chờ `cham` giây rồi trả lời; ném lỗi nếu `loi`."""
    def __init__(self, cham=0.0, loi=None, khong_tieng=False):
        self.goi, self.cham, self.loi, self.nong, self.khong_tieng = [], cham, loi, 0, khong_tieng

    async def tra_loi(self, jpg, cau):
        import asyncio
        self.goi.append((jpg, cau))
        await asyncio.sleep(self.cham)
        if self.loi:
            raise self.loi
        if self.khong_tieng:
            return 'Phía trước có một xe máy đỗ bên phải.', None
        return 'Phía trước có một xe máy đỗ bên phải.', pcm_sang_wav(np.zeros(2400, np.int16), 24000)

    async def lam_nong(self):
        self.nong += 1


def _hoi(ws, cau):
    ws.send_text(json.dumps({'lenh': 'hoi', 'cau': cau}))


def _doc_den_hoi(ws, loai, toi_da=20):
    """Đọc tin JSON (bỏ qua câu dẫn đường + WAV của nó) tới khi gặp tin hỏi đáp `loai`."""
    for _ in range(toi_da):
        m = ws.receive()
        if m.get('text'):
            j = json.loads(m['text'])
            if j.get('hoi') == loai:
                return j
    raise AssertionError(f'không thấy hoi={loai}')


def test_hoi_gui_khung_moi_nhat_va_tra_loi_kem_wav(tmp_path):
    hoi = HoiGia()
    with TestClient(tao_app(EngineGia(noi_o=99), MA, str(tmp_path), hoi=hoi)) as c, \
            c.websocket_connect(f'/ws?k={MA}') as ws:
        for i in (1, 2):
            gui(ws, i)
            ws.receive_json()
        _hoi(ws, 'Phía trước có gì?')
        assert _doc_den_hoi(ws, 'dang_hoi')['cau_hoi'] == 'Phía trước có gì?'
        tl = _doc_den_hoi(ws, 'tra_loi')
        assert tl['cau'] == 'Phía trước có một xe máy đỗ bên phải.'
        # t_chup theo đồng hồ iPhone của khung mới nhất: trang tính độ trễ đúng, không coi câu trả lời là câu cũ
        assert tl['t_chup'] == 1_000_200.0 and tl['id'] == 2
        assert ws.receive_bytes()[:4] == b'RIFF'
    assert hoi.goi == [(jpeg(2), 'Phía trước có gì?')]   # đúng ảnh iPhone gửi, không nén lại
    assert hoi.nong == 1
    dong = [json.loads(x) for x in open(tmp_path / 'hoi.jsonl', encoding='utf-8')]
    assert dong[0]['cau_hoi'] == 'Phía trước có gì?' and dong[0]['tra_loi'].startswith('Phía trước')


def test_tra_loi_khong_co_tieng_thi_gui_chu_de_dien_thoai_tu_doc(tmp_path):
    # Gemini đọc lỗi (quá tải / mất mạng) mà đã có câu trả lời (vd. model trên máy): đừng báo "Không hỏi được"
    hoi = HoiGia(khong_tieng=True)
    with TestClient(tao_app(EngineGia(noi_o=99), MA, str(tmp_path), hoi=hoi)) as c, \
            c.websocket_connect(f'/ws?k={MA}') as ws:
        gui(ws, 1)
        ws.receive_json()
        _hoi(ws, 'Phía trước có gì?')
        tl = _doc_den_hoi(ws, 'tra_loi')
        assert tl['cau'] == 'Phía trước có một xe máy đỗ bên phải.' and tl['doc_tai_cho'] is True
        gui(ws, 2)
        assert ws.receive().get('text')            # không có WAV đi kèm: tin kế là trả lời khung, không phải tiếng
    dong = [json.loads(x) for x in open(tmp_path / 'hoi.jsonl', encoding='utf-8')]
    assert dong[0]['tra_loi'].startswith('Phía trước') and 'loi' not in dong[0]


def test_hoi_rong_hoac_chua_co_hinh_thi_bao_loi(tmp_path):
    hoi = HoiGia()
    with TestClient(tao_app(EngineGia(noi_o=99), MA, str(tmp_path), hoi=hoi)) as c, \
            c.websocket_connect(f'/ws?k={MA}') as ws:
        _hoi(ws, 'Phía trước có gì?')
        assert 'hình' in _doc_den_hoi(ws, 'loi')['loi_hoi']
        gui(ws, 1)
        ws.receive_json()
        _hoi(ws, '   ')
        assert _doc_den_hoi(ws, 'loi')['loi_hoi']
    assert hoi.goi == []


def test_hoi_chong_cau_truoc_bi_tu_choi(tmp_path):
    hoi = HoiGia(cham=0.5)
    with TestClient(tao_app(EngineGia(noi_o=99), MA, str(tmp_path), hoi=hoi)) as c, \
            c.websocket_connect(f'/ws?k={MA}') as ws:
        gui(ws, 1)
        ws.receive_json()
        _hoi(ws, 'Câu một là gì?')
        _doc_den_hoi(ws, 'dang_hoi')
        _hoi(ws, 'Câu hai là gì?')
        assert 'trước' in _doc_den_hoi(ws, 'loi')['loi_hoi']
        _doc_den_hoi(ws, 'tra_loi')
    assert [c for _, c in hoi.goi] == ['Câu một là gì?']


def test_hoi_loi_api_bao_ve_trang_khong_lam_chet_dan_duong(tmp_path):
    hoi = HoiGia(loi=RuntimeError('429 hết lượt'))
    with TestClient(tao_app(EngineGia(noi_o=3), MA, str(tmp_path), hoi=hoi)) as c, \
            c.websocket_connect(f'/ws?k={MA}') as ws:
        gui(ws, 1)
        ws.receive_json()
        _hoi(ws, 'Phía trước có gì?')
        assert _doc_den_hoi(ws, 'loi')['loi_hoi']
        for i in (2, 3):                       # dẫn đường vẫn chạy sau lỗi hỏi
            gui(ws, i)
        ids = []
        for _ in range(6):
            m = ws.receive()
            if m.get('text') and 'id' in json.loads(m['text']):
                ids.append(json.loads(m['text'])['id'])
            if 3 in ids:
                break
        assert 3 in ids


def test_khong_co_hoi_thi_bao_che_do_nay_chua_hoi_duoc(tmp_path):
    with ket_noi(tmp_path, EngineGia(noi_o=99)) as c, c.websocket_connect(f'/ws?k={MA}') as ws:
        gui(ws, 1)
        ws.receive_json()
        _hoi(ws, 'Phía trước có gì?')
        assert _doc_den_hoi(ws, 'loi')['loi_hoi']
