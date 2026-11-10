import math
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from loi_do_sau.truc_tiep import chay, chung_chi, may_chu
from loi_do_sau.truc_tiep.chay import tim_url

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(GOC, 'OpenAIglasses_for_Navigation', 'tools'))
from esp32_ao import phat_lai  # noqa: E402
from esp32_ao.ho_so_anh import HoSoAnh, doc_ho_so  # noqa: E402


class _DungTaiDay(Exception):
    """Dừng chay.py ngay trước bước tốn GPU và 30 giây nạp model."""


def _env_app_main(monkeypatch, tmp_path, *doi_so):
    """Chạy chay.py thật tới lúc bật app_main, trả env tiến trình con nhận được."""
    env = {}

    def popen_gia(lenh, **kw):
        env.update(kw['env'])
        raise _DungTaiDay

    monkeypatch.setattr(chay, '__file__', str(tmp_path / 'chay.py'))   # thư mục phiên nằm trong tmp_path
    monkeypatch.setattr(phat_lai, 'may_chu_dang_chay', lambda *a: False)
    monkeypatch.setattr(phat_lai.subprocess, 'Popen', popen_gia)
    monkeypatch.setattr(sys, 'argv', ['chay.py', '--cau-noi', '--khong-ham', *doi_so])
    with pytest.raises(_DungTaiDay):
        chay.main()
    return env


def _goc_that_cua_anh(ten_ho_so):
    """Đo góc nhìn ngang của ảnh HoSoAnh tạo từ khung 640x360 của trang: mỗi điểm vào mang toạ độ x của nó,
    độ dốc giữa ảnh ra = số điểm ảnh gốc trên một điểm ảnh ra. k1 = 0: méo chỉ dồn mép, không đổi tiêu cự ở tâm."""
    cfg = dict(doc_ho_so(ten_ho_so), k1=0.0)
    x = np.tile(np.arange(640, dtype=np.float32), (360, 1))
    ra = HoSoAnh(cfg)._ong_kinh(np.dstack([x, x, x]))
    doc = (ra[240, 420, 0] - ra[240, 220, 0]) / 200
    nguon = cfg.get('hfov_nguon', 69.0)
    return 2 * math.degrees(math.atan(doc * math.tan(math.radians(nguon / 2))))


@pytest.mark.parametrize('ten', ['ov2640_xiao', 'ov3660_xiao', 'khong'])
def test_cau_noi_bao_may_chu_dung_goc_cua_anh_gui_di(monkeypatch, tmp_path, ten):
    # lỗi 2026-11-06: ảnh đã cắt còn 56° mà dan_theo_vach vẫn tính theo 70° -> tiêu cự sai 24%, khoảng cách tới chỗ
    # rẽ / hết vạch sai theo
    monkeypatch.delenv('AIGLASS_HFOV', raising=False)
    env = _env_app_main(monkeypatch, tmp_path, '--ho-so', ten)
    assert 'AIGLASS_HFOV' in env
    assert float(env['AIGLASS_HFOV']) == pytest.approx(_goc_that_cua_anh(ten), abs=0.5)


@pytest.mark.parametrize('doi_so, env_san, muon', [
    (('--hfov', '63'), None, 63.0),     # góc đo được thắng góc ước lượng trong hồ sơ
    ((), '60', 60.0),                   # góc đặt sẵn trong shell không bị ghi đè
    (('--hfov', '63'), '60', 63.0),     # đặt thẳng trên dòng lệnh là ý định rõ nhất
])
def test_cau_noi_uu_tien_goc_nguoi_dung_dat(monkeypatch, tmp_path, doi_so, env_san, muon):
    monkeypatch.delenv('AIGLASS_HFOV', raising=False)
    if env_san:
        monkeypatch.setenv('AIGLASS_HFOV', env_san)
    assert float(_env_app_main(monkeypatch, tmp_path, *doi_so)['AIGLASS_HFOV']) == muon


@pytest.mark.parametrize('doi_so, muon', [((), 55.0), (('--hfov', '60'), 60.0)])
def test_dan_vach_giu_nguyen_goc_nhin(monkeypatch, tmp_path, doi_so, muon):
    def engine_gia(_lam_xau):
        raise _DungTaiDay

    monkeypatch.setenv('AIGLASS_HFOV', 'chua_dat')   # main() ghi thẳng os.environ: để monkeypatch trả lại sau test
    monkeypatch.setattr(chay, '__file__', str(tmp_path / 'chay.py'))
    monkeypatch.setattr(may_chu, 'EngineDanVach', engine_gia)
    monkeypatch.setattr(sys, 'argv', ['chay.py', '--dan-vach', '--khong-ham', *doi_so])
    with pytest.raises(_DungTaiDay):
        chay.main()
    assert float(os.environ['AIGLASS_HFOV']) == muon


def test_tim_url_trong_log_cloudflared():
    dong = '2026-10-25T00:10:01Z INF |  https://quiet-river-lamp-1234.trycloudflare.com                      |'
    assert tim_url(dong) == 'https://quiet-river-lamp-1234.trycloudflare.com'


def test_bo_qua_dong_khac():
    assert tim_url('2026-10-25T00:10:01Z INF Requesting new quick Tunnel on trycloudflare.com...') is None
    assert tim_url('INF +----------------------------+') is None


def test_bo_qua_dia_chi_api_trong_dong_loi():
    # lỗi thật 2026-10-27: Cloudflare lỗi tạm, dòng lỗi chứa api.trycloudflare.com -> bị lấy nhầm làm link cho iPhone
    dong = ('ERR Error unmarshaling QuickTunnel response: failed to request quick Tunnel: '
            'Post "https://api.trycloudflare.com/tunnel": EOF')
    assert tim_url(dong) is None


class _EngineGia:
    def __init__(self, ghi):
        self.ghi = ghi

    def lam_nong(self):
        self.ghi['thu_tu'].append('lam_nong')
        self.ghi['luong_lam_nong'] = threading.get_ident()


def _chay_noi_thang(monkeypatch, tmp_path, *doi_so, bat_may_nhin=None):
    """main() thật tới lúc uvicorn nhận tham số. Thay phần nặng hoặc đụng ra ngoài: model, server, cổng, khóa ngủ,
    llama-server (bat_may_nhin: hàm thay thế, mặc định coi như không có model trên máy)."""
    import khoi_dong_dan_duong  # pyright: ignore[reportMissingImports]
    ghi = {'thu_tu': []}
    if bat_may_nhin is not None:
        ghi['bat_may_nhin'] = bat_may_nhin
    monkeypatch.setattr(chay, '__file__', str(tmp_path / 'chay.py'))
    monkeypatch.setattr(may_chu, 'EngineDanVach', lambda _lam_xau: _EngineGia(ghi))
    monkeypatch.setattr(khoi_dong_dan_duong, 'dong_bang_bo_nho', lambda: ghi['thu_tu'].append('dong_bang'))
    monkeypatch.setattr(chung_chi, 'THU_MUC', str(tmp_path / 'cc'))
    monkeypatch.setattr(chung_chi, 'ip_may', lambda: ['127.0.0.1'])
    monkeypatch.setattr(chung_chi, 'phat_chung_chi', lambda ca, ip: ghi.update(ca=ca, ip=ip))
    monkeypatch.setattr(chay, 'giu_may_thuc', lambda: ghi['thu_tu'].append('giu_thuc'))
    monkeypatch.setattr(chay, 'mo_ham', lambda *a: pytest.fail('không được mở đường hầm Cloudflare'))
    monkeypatch.setattr('uvicorn.run', lambda _app, **kw: ghi.update(uvicorn=kw))
    monkeypatch.setattr(chay, 'bat_may_nhin', ghi.get('bat_may_nhin', lambda thu_muc: (None, False)))
    tao_app_that = may_chu.tao_app

    def tao_app_ghi(*a, **k):
        ghi['luong_khung'] = k['luong'].submit(threading.get_ident).result()
        ghi['hoi'] = k.get('hoi')
        return tao_app_that(*a, **k)

    monkeypatch.setattr(may_chu, 'tao_app', tao_app_ghi)
    monkeypatch.setattr(sys, 'argv', ['chay.py', *doi_so])
    chay.main()
    return ghi


def test_noi_thang_phuc_vu_https_tren_mang_noi_bo(monkeypatch, tmp_path, capsys):
    # Safari chỉ mở camera trên https; iPhone nối vào laptop qua mạng nên không được chỉ nghe 127.0.0.1
    ghi = _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang')
    kw = ghi['uvicorn']
    assert kw['host'] == '0.0.0.0'
    assert os.path.exists(kw['ssl_certfile']) and os.path.exists(kw['ssl_keyfile'])
    ra = capsys.readouterr().out
    # mã QR mở app trên GitHub Pages, mang theo địa chỉ laptop; ảnh 4:3 đúng đường dẫn vạch
    assert 'github.io/VisionAid/?may=127.0.0.1:8765&k=' in ra and 'khung=43' in ra
    assert 'http://127.0.0.1:8766/' in ra and ghi['ip'] == '127.0.0.1'


def test_noi_thang_trang_may_tinh_nhu_cu(monkeypatch, tmp_path, capsys):
    _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang', '--trang-may-tinh')
    ra = capsys.readouterr().out
    assert 'https://127.0.0.1:8765/?k=' in ra and 'khung=43' in ra and 'github.io' not in ra


def test_noi_thang_lam_nong_roi_moi_dong_bang_bo_nho(monkeypatch, tmp_path):
    # đóng băng trước khi làm nóng thì đối tượng tạo lúc làm nóng vẫn bị GC quét, hình vẫn đứng
    ghi = _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang')
    assert ghi['thu_tu'].index('lam_nong') < ghi['thu_tu'].index('dong_bang')
    assert 'giu_thuc' in ghi['thu_tu']


def test_noi_thang_lam_nong_tren_dung_luong_se_xu_ly_khung(monkeypatch, tmp_path):
    # làm nóng ở luồng khác thì khung thật đầu tiên vẫn mất ~4,3 s
    ghi = _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang')
    assert ghi['luong_lam_nong'] == ghi['luong_khung']


def test_noi_thang_dung_duoc_voi_cau_noi(monkeypatch, tmp_path, capsys):
    # so bản gốc với bản mới trên cùng đường mạng: qua đường hầm hình kẹt tới 15 s (đo 2026-11-10), lời dẫn tới trễ
    # nên khó biết sai do thuật toán hay do mạng. (--cau-noi vẫn chậm hơn ~1,5 s vì chờ hết câu mới có tiếng.)
    from loi_do_sau.truc_tiep import cau_noi
    ghi = {'thu_tu': []}
    monkeypatch.setattr(chay, '__file__', str(tmp_path / 'chay.py'))
    monkeypatch.setattr(phat_lai, 'may_chu_dang_chay', lambda *a: False)
    monkeypatch.setattr(phat_lai, 'bat_may_chu', lambda *a, **k: 'app_main')
    monkeypatch.setattr(phat_lai, 'tat_may_chu', lambda p: ghi.update(tat=p))
    monkeypatch.setattr(cau_noi, 'tao_app_cau_noi', lambda *a, **k: 'app')
    monkeypatch.setattr(chung_chi, 'THU_MUC', str(tmp_path / 'cc'))
    monkeypatch.setattr(chung_chi, 'ip_may', lambda: ['127.0.0.1'])
    monkeypatch.setattr(chung_chi, 'phat_chung_chi', lambda ca, ip: ghi.update(ip=ip))
    monkeypatch.setattr(chay, 'giu_may_thuc', lambda: ghi['thu_tu'].append('giu_thuc'))
    monkeypatch.setattr(chay, 'mo_ham', lambda *a: pytest.fail('không được mở đường hầm Cloudflare'))
    monkeypatch.setattr(chay.signal, 'signal', lambda *a: None)   # đừng để pytest bỏ qua Ctrl+C sau test
    monkeypatch.setattr('uvicorn.run', lambda _app, **kw: ghi.update(uvicorn=kw))
    monkeypatch.setattr(sys, 'argv', ['chay.py', '--noi-thang', '--cau-noi', '--goc-viet', '--trang-may-tinh'])
    chay.main()
    kw = ghi['uvicorn']
    assert kw['host'] == '0.0.0.0' and kw['port'] == 8765
    assert os.path.exists(kw['ssl_certfile']) and os.path.exists(kw['ssl_keyfile'])
    ra = capsys.readouterr().out
    assert 'https://127.0.0.1:8765/?k=' in ra and 'github.io' not in ra and 'http://127.0.0.1:8766/' in ra
    assert 'giu_thuc' in ghi['thu_tu'] and ghi['tat'] == 'app_main'


def _co_khoa_ngu():
    ra = subprocess.run(['systemd-inhibit', '--list', '--no-pager'], capture_output=True, text=True).stdout
    return 'glassai' in ra


def _cho(dieu_kien, giay=5.0):
    het = time.time() + giay
    while time.time() < het:
        if dieu_kien():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(shutil.which('systemd-inhibit') is None, reason='máy không có systemd-inhibit')
def test_khoa_ngu_tu_nha_khi_chuong_trinh_bi_giet():
    # SIGTERM làm chay.py chết mà không qua finally (uvicorn phát lại tín hiệu): khóa kẹt lại thì laptop không bao giờ ngủ
    ma = ('import sys, time\n'
          f'sys.path.insert(0, {GOC!r})\n'
          'from loi_do_sau.truc_tiep.chay import giu_may_thuc\n'
          'khoa = giu_may_thuc()\n'
          'print("ok", flush=True)\n'
          'time.sleep(60)\n')
    p = subprocess.Popen([sys.executable, '-c', ma], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout is not None and p.stdout.readline().strip() == 'ok'
        assert _cho(_co_khoa_ngu)
    finally:
        p.kill()
        p.wait()
    assert _cho(lambda: not _co_khoa_ngu())


def test_dia_chi_qr_mo_app_pages_kem_may_tinh():
    """Mã QR mở app trên GitHub Pages (địa chỉ cố định) và mang theo địa chỉ máy tính; --trang-may-tinh như cũ."""
    from loi_do_sau.truc_tiep.chay import PAGES, dia_chi_qr
    assert dia_chi_qr('https://ab-cd.trycloudflare.com', 'M', '&khung=43&fps=8') == \
        PAGES + '?may=ab-cd.trycloudflare.com&k=M&khung=43&fps=8'
    assert dia_chi_qr('https://192.168.1.5:8765', 'M', '', PAGES) == PAGES + '?may=192.168.1.5:8765&k=M'
    assert dia_chi_qr('https://ab-cd.trycloudflare.com', 'M', '&fps=10', None) == \
        'https://ab-cd.trycloudflare.com/?k=M&fps=10'



# ---------- model nhìn trên máy (llama-server + Qwen3-VL-2B) cho nút Hỏi ----------
DS_GPU = """ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: AMD Radeon 780M Graphics (RADV PHOENIX) (8154 MiB, 8011 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, 2806 MiB free)
"""


def test_chon_gpu_nvidia_theo_ten():
    # GPU AMD tích hợp lấy RAM chung: chạy ở đó RAM máy còn 0,75 GB, dễ làm máy chủ dẫn đường bị giết (đo 2026-11-10)
    assert chay.chon_gpu(DS_GPU) == 'Vulkan1'
    assert chay.chon_gpu('Available devices:\n  Vulkan0: AMD Radeon 780M Graphics (8154 MiB)\n') is None


def _tep_may_nhin(monkeypatch, tmp_path):
    exe, thu_muc = tmp_path / 'llama-server', tmp_path / 'qwen'
    thu_muc.mkdir()
    exe.write_text('')
    for ten in (chay.QWEN_VL_MODEL, chay.QWEN_VL_MMPROJ):
        (thu_muc / ten).write_text('')
    monkeypatch.setattr(chay, 'LLAMA_SERVER', str(exe))
    monkeypatch.setattr(chay, 'QWEN_VL_DIR', str(thu_muc))
    monkeypatch.setattr(chay, '_liet_ke_gpu', lambda exe: DS_GPU)
    monkeypatch.setattr(chay, '_cong_dang_mo', lambda cong: False)
    monkeypatch.delenv('AIGLASS_MAY_NHIN', raising=False)
    return exe, thu_muc


def test_bat_may_nhin_dung_model_gpu_va_cong(monkeypatch, tmp_path):
    exe, thu_muc = _tep_may_nhin(monkeypatch, tmp_path)
    goi = {}

    def popen(lenh, **kw):
        goi['lenh'] = lenh
        return 'tien_trinh'
    monkeypatch.setattr(chay.subprocess, 'Popen', popen)
    p, co = chay.bat_may_nhin(str(tmp_path))
    assert (p, co) == ('tien_trinh', True)
    l = goi['lenh']
    assert l[0] == str(exe) and l[l.index('-m') + 1] == str(thu_muc / chay.QWEN_VL_MODEL)
    assert l[l.index('--mmproj') + 1] == str(thu_muc / chay.QWEN_VL_MMPROJ)
    assert l[l.index('--device') + 1] == 'Vulkan1'
    assert l[l.index('--host') + 1] == '127.0.0.1' and l[l.index('--port') + 1] == str(chay.CONG_MAY_NHIN)


def test_bat_may_nhin_thieu_tep_thi_bo_qua(monkeypatch, tmp_path, capsys):
    _tep_may_nhin(monkeypatch, tmp_path)
    monkeypatch.setattr(chay, 'QWEN_VL_DIR', str(tmp_path / 'khong_co'))
    monkeypatch.setattr(chay.subprocess, 'Popen', lambda *a, **k: pytest.fail('không được bật'))
    assert chay.bat_may_nhin(str(tmp_path)) == (None, False)
    assert 'model trên máy' in capsys.readouterr().out


def test_bat_may_nhin_tat_bang_bien_moi_truong(monkeypatch, tmp_path):
    _tep_may_nhin(monkeypatch, tmp_path)
    monkeypatch.setenv('AIGLASS_MAY_NHIN', '0')
    monkeypatch.setattr(chay.subprocess, 'Popen', lambda *a, **k: pytest.fail('không được bật'))
    assert chay.bat_may_nhin(str(tmp_path)) == (None, False)


def test_bat_may_nhin_da_chay_san_thi_dung_lai(monkeypatch, tmp_path):
    _tep_may_nhin(monkeypatch, tmp_path)
    monkeypatch.setattr(chay, '_cong_dang_mo', lambda cong: True)
    monkeypatch.setattr(chay.subprocess, 'Popen', lambda *a, **k: pytest.fail('không bật bản thứ hai'))
    assert chay.bat_may_nhin(str(tmp_path)) == (None, True)


def test_co_model_tren_may_thi_nut_hoi_chi_dung_may(monkeypatch, tmp_path):
    from loi_do_sau.truc_tiep import hoi_dap
    monkeypatch.setattr(hoi_dap, 'co_khoa', lambda: True)
    ghi = _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang', bat_may_nhin=lambda thu_muc: (None, True))
    assert isinstance(ghi['hoi'], hoi_dap.HoiTrenMay)


def test_hoi_gemini_thi_dung_gemini_du_co_model_tren_may(monkeypatch, tmp_path):
    from loi_do_sau.truc_tiep import hoi_dap
    monkeypatch.setattr(hoi_dap, 'co_khoa', lambda: True)
    ghi = _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang', '--hoi-gemini',
                          bat_may_nhin=lambda thu_muc: (None, True))
    assert type(ghi['hoi']) is hoi_dap.HoiGemini


def test_khong_model_tren_may_thi_dung_gemini(monkeypatch, tmp_path):
    from loi_do_sau.truc_tiep import hoi_dap
    monkeypatch.setattr(hoi_dap, 'co_khoa', lambda: True)
    assert type(_chay_noi_thang(monkeypatch, tmp_path, '--noi-thang')['hoi']) is hoi_dap.HoiGemini


def test_khong_khoa_gemini_nhung_co_model_tren_may_van_bat_nut_hoi(monkeypatch, tmp_path):
    from loi_do_sau.truc_tiep import hoi_dap

    class _Tt:
        def __init__(self):
            self.tat = False

        def terminate(self):
            self.tat = True
    tt = _Tt()
    monkeypatch.setattr(hoi_dap, 'co_khoa', lambda: False)
    ghi = _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang', bat_may_nhin=lambda thu_muc: (tt, True))
    assert ghi['hoi'] is not None
    assert tt.tat                                       # tắt máy chủ thì tắt luôn llama-server mình bật


def test_khong_khoa_khong_model_tren_may_thi_tat_nut_hoi(monkeypatch, tmp_path):
    from loi_do_sau.truc_tiep import hoi_dap
    monkeypatch.setattr(hoi_dap, 'co_khoa', lambda: False)
    assert _chay_noi_thang(monkeypatch, tmp_path, '--noi-thang')['hoi'] is None
