"""
Máy chủ thử kính trực tiếp: iPhone kẹp mũ gửi khung JPEG qua một WebSocket, máy chạy module né vật cản và gửi
câu nói (wav) ngược lại. Mỗi phiên ghi khung.mp4 (mọi khung đã nhận, theo thứ tự), cau/<id>.wav và log.json.

Thiết kế: docs/superpowers/specs/2026-10-24-thu-kinh-truc-tiep-design.md
"""
import asyncio
import concurrent.futures
import contextlib
import datetime
import io
import json
import secrets
import os
import shutil
import subprocess
import time
import wave
from typing import Any, Protocol

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from loi_do_sau.truc_tiep.goi_tin import mo_goi

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
DU_AN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     'OpenAIglasses_for_Navigation')
W, H = 640, 360
SR = 16000


def _them_du_an():
    import sys
    if DU_AN not in sys.path:
        sys.path.insert(0, DU_AN)


_them_du_an()
_da_gop_giong = False


def wav_cua_cau(text: str) -> bytes | None:
    """File wav của câu nói, tra như audio_player.play_voice_text (NFC, thử thêm/bỏ dấu chấm cuối). None nếu không có."""
    global _da_gop_giong
    import unicodedata
    import audio_player  # pyright: ignore[reportMissingImports]
    if not _da_gop_giong:
        audio_player._merge_voice_map()
        _da_gop_giong = True
    t = unicodedata.normalize('NFC', (text or '').strip())
    for k in (t, t + '.', t.rstrip('.!?')):
        p = audio_player.AUDIO_MAP.get(k)       # câu ghép của module né vật cản được đăng ký lúc chạy
        if p and os.path.exists(p):
            with open(p, 'rb') as f:
                return f.read()
    return None


class Engine(Protocol):
    def step(self, img_bgr: np.ndarray, t: float) -> tuple[str | None, bytes | None]: ...

    def cho_noi_lai(self) -> None: ...


def pcm_sang_wav(pcm: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


class EngineThat:
    """Bọc NeVatCanDoSau (OpenAIglasses_for_Navigation/ne_vat_can_do_sau.py). lam_xau: hàm làm ảnh giống camera kính."""
    def __init__(self, lam_xau=None):
        import sys
        du_an = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'OpenAIglasses_for_Navigation')
        sys.path.insert(0, du_an)
        from ne_vat_can_do_sau import NeVatCanDoSau  # pyright: ignore[reportMissingImports]
        self.nv = NeVatCanDoSau()
        self.lam_xau = lam_xau
        self.can_noi_lai = False

    def step(self, img_bgr, t):
        if getattr(self, 'can_noi_lai', False):   # áp dụng ở luồng của step, không đụng Speaker từ luồng khác
            self.can_noi_lai = False
            self.nv.speaker.last = None
            self.nv.speaker.busy_until = -1.0
        if self.lam_xau:
            # khung đầu: chưa có khung trước, dùng 0,1 s (bộ giả lập cũng chưa có gì để làm nhòe)
            dt = t - getattr(self, 't_truoc', t - 0.1)
            self.t_truoc = t
            img_bgr = self.lam_xau(img_bgr, dt=dt)
        cau = self.nv.step(img_bgr, t)
        if not cau:
            return None, None
        return cau, pcm_sang_wav(self.nv.speaker.events[-1][1])

    def cho_noi_lai(self):
        self.can_noi_lai = True


class EngineDanVach:
    """Chạy đúng đường hệ thống dẫn theo vạch: BlindPathNavigator.process_frame (yolo-seg + dan_theo_vach + module
    né vật cản độ sâu với cờ "đang theo vạch"). Đồng hồ của workflow_blindpath và ne_vat_can_do_sau là đồng hồ ảo
    = giờ chụp trên iPhone, nên các luật chờ (1 s, 1,5 s, 4 s...) đúng cả khi phát lại phiên nhanh hơn thời gian thật."""
    GOC_GIO = 1000.0

    def __init__(self, lam_xau=None, navigator=None):
        import types
        import time as _time
        import workflow_blindpath as wb  # pyright: ignore[reportMissingImports]
        self.gio = self.GOC_GIO
        self.dong_ho = types.SimpleNamespace(**{k: getattr(_time, k) for k in dir(_time) if not k.startswith('_')})
        self.dong_ho.time = lambda: self.gio
        wb.time = self.dong_ho
        self.da_noi: list[str] = []
        wb.play_voice_text = self.da_noi.append       # workflow_blindpath import play_voice_text theo tên
        self.nav = navigator if navigator is not None else self._tao_nav(wb)
        self.lam_xau = lam_xau
        self.can_noi_lai = False
        self.cau_cuoi = None

    def _tao_nav(self, wb):
        import torch
        from ultralytics import YOLO
        import ne_vat_can_do_sau  # pyright: ignore[reportMissingImports]
        ne_vat_can_do_sau.time = self.dong_ho         # trước khi tạo navigator: NeVatCanDoSau lấy mốc t0 lúc khởi tạo
        seg = YOLO(os.path.join(DU_AN, 'model', 'yolo-seg.pt'))
        if torch.cuda.is_available():
            seg.to('cuda')
        return wb.BlindPathNavigator(seg, None)       # vật cản: module độ sâu (mặc định), không cần YOLOE cũ

    def lam_nong(self, cao=480, rong=640):
        """Gọi trước khi nhận khung (khung đầu không làm nóng mất 5,9 s, đo 2026-11-06); xong trả dẫn đường về như mới."""
        from khoi_dong_dan_duong import lam_nong_phan_vung  # pyright: ignore[reportMissingImports]
        self.nav.process_frame(np.full((cao, rong, 3), 128, np.uint8))
        lam_nong_phan_vung(self.nav, cao, rong)
        self.nav.reset()
        self.da_noi.clear()

    def step(self, img_bgr, t):
        if self.can_noi_lai:                          # áp dụng ở luồng của step
            self.can_noi_lai = False
            dv = getattr(self.nav, 'dan_vach', None)
            if dv is not None and self.cau_cuoi:
                dv.chua_noi(self.cau_cuoi)
        if self.lam_xau:
            dt = t - getattr(self, 't_truoc', t - 0.1)
            self.t_truoc = t
            img_bgr = self.lam_xau(img_bgr, dt=dt)
        self.gio = self.GOC_GIO + t
        n0 = len(self.da_noi)
        self.nav.process_frame(img_bgr)
        moi = self.da_noi[n0:]
        if not moi:
            return None, None
        self.cau_cuoi = moi[-1]
        wav = wav_cua_cau(self.cau_cuoi)
        if wav is None:                               # iPhone sẽ chỉ hiện chữ: người đeo không nghe thấy
            print(f'[THU KINH] CẢNH BÁO: câu không có wav (người đeo không nghe): {self.cau_cuoi!r}', flush=True)
        return self.cau_cuoi, wav

    def cho_noi_lai(self):
        self.can_noi_lai = True


class Phien:
    """Ghi một phiên thử: khung.mp4, cau/<id>.wav, log.jsonl (mỗi sự kiện một dòng, ghi ngay: máy tắt đột ngột
    vẫn còn) và log.json (bản đầy đủ, ghi khi kết thúc/ngắt kết nối).
    Video ghi theo đoạn (doan/<n>.mp4, mỗi lần kết nối một đoạn); ngắt kết nối thì đóng đoạn và ghép lại thành
    khung.mp4 -> bấm Dừng trên iPhone là xem được ngay, không phải tắt máy chủ."""
    def __init__(self, thu_muc):
        self.thu_muc = thu_muc
        os.makedirs(os.path.join(thu_muc, 'cau'), exist_ok=True)
        os.makedirs(os.path.join(thu_muc, 'doan'), exist_ok=True)
        self.writer = None
        self.doan: list[str] = []
        self.log: dict[str, Any] = dict(bat_dau=datetime.datetime.now().isoformat(timespec='seconds'), khung=[], phan_hoi=[])
        self.theo_id = {}
        self.jsonl = open(os.path.join(thu_muc, 'log.jsonl'), 'a', encoding='utf-8')
        self._dong({'loai': 'bat_dau', 'bat_dau': self.log['bat_dau']})

    def _dong(self, d):
        self.jsonl.write(json.dumps(d, ensure_ascii=False) + '\n')
        self.jsonl.flush()

    def nhan(self, id_khung, t_chup, img):
        if self.writer is None:
            self.doan.append(os.path.join(self.thu_muc, 'doan', f'{len(self.doan) + 1:04d}.mp4'))
            self.writer = cv2.VideoWriter(self.doan[-1], cv2.VideoWriter.fourcc(*'mp4v'), 10, (img.shape[1], img.shape[0]))
        self.writer.write(img)
        k = dict(id=id_khung, t_chup=t_chup, t_nhan=time.time() * 1000, ms_xu_ly=None, cau=None, bo_qua=True)
        self.log['khung'].append(k)
        self.theo_id[id_khung] = k
        self._dong(dict(k, loai='nhan'))

    def xong(self, id_khung, ms_xu_ly, cau=None, wav=None, loi=None):
        k = self.theo_id[id_khung]
        k.update(ms_xu_ly=ms_xu_ly, cau=cau, bo_qua=False)
        if loi:
            k['loi'] = loi
        if wav:
            with open(os.path.join(self.thu_muc, 'cau', f'{id_khung}.wav'), 'wb') as f:
                f.write(wav)
        self._dong(dict(loai='xong', id=id_khung, ms_xu_ly=ms_xu_ly, cau=cau, loi=loi))

    def phan_hoi(self, ph):
        ph = dict(ph, t_nhan=time.time() * 1000)
        self.log['phan_hoi'].append(ph)
        self._dong(dict(ph, loai='phan_hoi'))

    def ghi_log(self):
        tmp = os.path.join(self.thu_muc, 'log.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.log, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.thu_muc, 'log.json'))

    def dong_doan(self) -> list[str]:
        """Đóng đoạn đang ghi (nhanh, gọi ở luồng sự kiện); trả về các đoạn đã đóng để ghép."""
        if self.writer is None:
            return []
        self.writer.release()
        self.writer = None
        return list(self.doan)

    def dong_video(self):
        self.ghep(self.dong_doan())

    def ghep(self, doan: list[str]):
        """Ghép các đoạn thành khung.mp4 (ffmpeg concat, chép nguyên luồng, không nén lại). Chậm: chạy ở luồng phụ."""
        if not doan:
            return
        out = os.path.join(self.thu_muc, 'khung.mp4')
        if len(doan) == 1:
            shutil.copyfile(doan[0], out + '.tmp')
        else:
            ds = os.path.join(self.thu_muc, 'doan', f'ds_{len(doan)}.txt')
            with open(ds, 'w') as f:
                f.writelines(f"file '{os.path.abspath(d)}'\n" for d in doan)
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', ds,
                            '-c', 'copy', '-f', 'mp4', out + '.tmp'], check=True)
        os.replace(out + '.tmp', out)

    def dong(self):
        self.dong_video()
        self.ghi_log()
        self.jsonl.close()


def tao_app(engine: Engine, ma: str, thu_muc_phien: str, kich_thuoc=(W, H), luong=None, hoi=None) -> FastAPI:
    """kich_thuoc (rộng, cao): mọi khung được đưa về cỡ này trước khi ghi và xử lý (dẫn vạch: 640x480 như ESP32).
    hoi: đối tượng có `async tra_loi(jpg, câu) -> (chữ, wav)` và `async lam_nong()` (hoi_dap.HoiGemini) cho nút
    "Hỏi" của app; None: trả lời "chế độ này chưa hỏi được"."""
    rong, cao = kich_thuoc
    # engine luôn chạy trên MỘT luồng, cùng luồng với làm nóng: lần chạy đầu trên luồng mới mất ~4,3 s ở mọi model (CUDA có
    # trạng thái theo luồng, đo 2026-11-06), bể luồng mặc định có thể đổi luồng giữa chừng, ví dụ khi iPhone nối lại
    luong = luong or concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='engine')
    # Một phiên = một app: 4G rớt rồi nối lại vẫn dùng chung mốc thời gian (Speaker/tracker của engine giữ mốc cũ)
    # và chung bản ghi (nối lại không được ghi đè video/log). Khóa: engine không chạy hai khung cùng lúc.
    chung: dict[str, Any] = dict(phien=None, t0=None, ws=None)
    khoa = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def vong_doi(_app):
        yield
        if chung['phien'] is not None:
            chung['phien'].dong()

    app = FastAPI(lifespan=vong_doi)
    if os.path.isdir(STATIC):
        app.mount('/static', StaticFiles(directory=STATIC), name='static')

    @app.get('/')
    async def trang():                # app VisionAid (mã QR mở vào đây)
        return FileResponse(os.path.join(STATIC, 'app.html'))

    @app.get('/thu')
    async def trang_thu():            # trang thử cũ cho người phát triển (cùng lõi kinh_loi.js)
        return FileResponse(os.path.join(STATIC, 'thu_kinh.html'))

    @app.websocket('/ws')
    async def ws_kinh(ws: WebSocket):
        await ws.accept()               # accept rồi mới đóng: trình duyệt nhận được mã 1008 ("Sai mã")
        if not secrets.compare_digest(ws.query_params.get('k', ''), ma):
            await ws.close(code=1008)
            return
        cu, chung['ws'] = chung['ws'], ws
        if cu is not None:              # kết nối cũ (nửa sống sau khi rớt mạng): đóng, chỉ giữ kết nối mới nhất
            with contextlib.suppress(Exception):
                await cu.close(code=1012)
        if chung['phien'] is None:
            chung['phien'] = Phien(thu_muc_phien)
        phien = chung['phien']
        cho: dict[str, tuple[int, float, np.ndarray] | None] = {'khung': None}             # ô "khung mới nhất": khung mới thay khung cũ chưa xử lý
        co_khung = asyncio.Event()
        # JSON có câu và WAV đi liền nhau thành cặp (trang ghép WAV với JSON ngay trước nó): câu dẫn đường và câu trả
        # lời chạy ở hai tác vụ, không khóa thì cặp này chen vào giữa cặp kia -> đọc nhầm tiếng cho câu
        khoa_gui = asyncio.Lock()
        moi_nhat: dict[str, Any] = dict(id=None, t_chup=None, jpg=None)   # khung iPhone mới nhất, cho câu hỏi
        dang_hoi: dict[str, Any] = dict(tac_vu=None)
        if hoi is not None:
            asyncio.create_task(hoi.lam_nong())   # lần gọi Gemini đầu chậm 4,5–9,4 s (đo 30/10): làm trước khi hỏi

        async def gui_cap(tin, wav=None):
            async with khoa_gui:
                await ws.send_json(tin)
                if wav:
                    await ws.send_bytes(wav)

        async def tra_loi(cau_hoi):
            bd = time.perf_counter()
            ghi = dict(t=time.time(), cau_hoi=cau_hoi, id_khung=moi_nhat['id'])
            try:
                cau, wav = await hoi.tra_loi(moi_nhat['jpg'], cau_hoi)
                ms = int((time.perf_counter() - bd) * 1000)
                ghi.update(tra_loi=cau, ms=ms)
                if not cau:
                    raise RuntimeError('không có câu trả lời')
                if not wav:                 # Gemini đọc lỗi: điện thoại tự đọc chữ bằng giọng tiếng Việt có sẵn
                    ghi.update(doc_tai_cho=True)
                    await gui_cap(dict(hoi='tra_loi', doc_tai_cho=True, cau=cau, ms_xu_ly=ms))
                    return
                # t_chup của khung MỚI NHẤT lúc trả (đồng hồ iPhone): trang tính trễ đúng; câu trả lời không bị coi cũ
                await gui_cap(dict(hoi='tra_loi', id=moi_nhat['id'], t_chup=moi_nhat['t_chup'], ms_xu_ly=ms,
                                   cau=cau), wav)
            except Exception as e:  # noqa: BLE001 – lỗi API (hết lượt, mất mạng) không được làm chết dẫn đường
                ghi.update(loi=repr(e))
                print(f'[THU KINH] hỏi lỗi: {e!r}', flush=True)
                await gui_cap(dict(hoi='loi', loi_hoi='Không hỏi được, thử lại sau.'))
            finally:
                dang_hoi['tac_vu'] = None
                with open(os.path.join(thu_muc_phien, 'hoi.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(ghi, ensure_ascii=False) + '\n')

        async def nhan_hoi(cau_hoi):
            cau_hoi = (cau_hoi or '').strip()
            if hoi is None:
                loi = 'Chế độ này chưa hỏi được.'
            elif not cau_hoi:
                loi = 'Chưa nghe rõ câu hỏi.'
            elif moi_nhat['jpg'] is None:
                loi = 'Chưa có hình từ camera.'
            elif dang_hoi['tac_vu'] is not None:
                loi = 'Đang trả lời câu trước.'
            else:
                await gui_cap(dict(hoi='dang_hoi', cau_hoi=cau_hoi))
                dang_hoi['tac_vu'] = asyncio.create_task(tra_loi(cau_hoi))
                return
            await gui_cap(dict(hoi='loi', loi_hoi=loi))

        async def xu_ly():
            try:
                await _xu_ly()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 – lỗi ngoài engine (đĩa đầy...): đóng để iPhone báo động, không im
                print(f'[THU KINH] tác vụ xử lý chết: {e!r}', flush=True)
                with contextlib.suppress(Exception):
                    await ws.close(code=1011)

        async def _xu_ly():
            while True:
                await co_khung.wait()
                co_khung.clear()
                khung = cho['khung']
                if khung is None:
                    continue
                id_khung, t_chup, img = khung
                cho['khung'] = None
                t = (t_chup - chung['t0']) / 1000.0
                bd = time.perf_counter()
                tin = dict(id=id_khung, t_chup=t_chup)
                try:
                    async with khoa:
                        cau, wav = await asyncio.get_running_loop().run_in_executor(luong, engine.step, img, t)
                except Exception as e:  # noqa: BLE001 – lỗi một khung không được giết cả phiên
                    ms = int((time.perf_counter() - bd) * 1000)
                    phien.xong(id_khung, ms, loi=repr(e))
                    await gui_cap(dict(tin, ms_xu_ly=ms, cau=None, loi=repr(e)))
                    continue
                ms = int((time.perf_counter() - bd) * 1000)
                phien.xong(id_khung, ms, cau, wav)
                await gui_cap(dict(tin, ms_xu_ly=ms, cau=cau), wav)

        tac_vu = asyncio.create_task(xu_ly())
        try:
            while True:
                msg = await ws.receive()
                if msg['type'] == 'websocket.disconnect':
                    break
                if msg.get('bytes') is not None:
                    try:
                        t_chup, id_khung, jpg = mo_goi(msg['bytes'])
                    except ValueError:
                        continue
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if img.shape[:2] != (cao, rong):
                        img = cv2.resize(img, (rong, cao), interpolation=cv2.INTER_AREA)
                    if chung['t0'] is None:
                        chung['t0'] = t_chup
                    phien.nhan(id_khung, t_chup, img)
                    moi_nhat.update(id=id_khung, t_chup=t_chup, jpg=jpg)
                    cho['khung'] = (id_khung, t_chup, img)
                    co_khung.set()
                elif msg.get('text'):
                    try:
                        ph = json.loads(msg['text'])
                    except json.JSONDecodeError:
                        continue
                    if ph.get('lenh') == 'hoi':
                        await nhan_hoi(ph.get('cau'))
                    elif 'phan_hoi' in ph:
                        phien.phan_hoi(ph)
                        # chỉ câu bị bỏ vì tới quá trễ mới cần nói lại; câu bị cắt / bị thay đã có câu mới hơn
                        # thế chỗ (nói lại thì "Dừng lại" cắt nhau thành vòng lặp mỗi khung)
                        if ph['phan_hoi'] == 'bo':
                            engine.cho_noi_lai()
        except WebSocketDisconnect:
            pass
        finally:
            tac_vu.cancel()
            if dang_hoi['tac_vu'] is not None:
                dang_hoi['tac_vu'].cancel()
            phien.ghi_log()
            if chung['ws'] is ws:
                chung['ws'] = None
                # kết nối cuối đã đi: đóng video cho xem được ngay (bị kết nối mới thay chỗ thì để nó ghi tiếp)
                await run_in_threadpool(phien.ghep, phien.dong_doan())

    return app
