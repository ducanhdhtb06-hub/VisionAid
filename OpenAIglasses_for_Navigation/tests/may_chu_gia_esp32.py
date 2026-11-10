"""Máy chủ giả có các endpoint giống app_main (cho test ESP32 ảo, không cần GPU/model).

Bắt chước đúng hành vi app_main.py ở những chỗ ESP32 ảo phụ thuộc:
- /ws/camera: đã có camera thì đóng TRƯỚC accept (app_main: ws.close(code=1013) -> trình khách thấy bắt tay bị từ chối);
- /ws_audio: "PROMPT:<câu>" -> "OK:PROMPT_ACCEPTED", "START" -> "OK:STARTED";
- /stream.wav: StreamingResponse (HTTP/1.1 chunked), đầu WAV 8 kHz kích thước không biết trước (audio_stream);
- /ws/viewer: gửi JPEG; UDP: nhận gói IMU.
Kịch bản (dict): tu_choi_camera (số lần từ chối), dong_camera_sau_s, restart_sau_s, loa [(chờ_s, pcm)...],
dong_loa_lan_dau, cau [(chờ_s, câu)] + nhat_ky (ghi nhật ký câu như AIGLASS_NHAT_KY_CAU).
"""
import asyncio
import contextlib
import json
import socket
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from audio_stream import _wav_header_unknown_size


def _cong_trong(kieu=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kieu) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MayChuGia:
    def __init__(self, kich_ban=None):
        self.kb = dict(kich_ban or {})
        self.cong, self.cong_udp = _cong_trong(), _cong_trong(socket.SOCK_DGRAM)
        self.khung: list[tuple[float, bytes]] = []
        self.tin_mic: list[str] = []
        self.mic_nhi_phan: list[int] = []
        self.imu: list[dict] = []
        self.su_kien: list[tuple[float, str]] = []
        self.camera_ket_noi = 0
        self.loa_ket_noi = 0
        self.prompt_khi_co_camera: list[bool] = []   # app_main chỉ bật dẫn đường được khi camera đã nối
        self.t_loa_bat_dau = None
        self._tu_choi = int(self.kb.get("tu_choi_camera", 0))
        self._camera = None
        self._da_dong_camera = False
        self._dung = False
        self.app = self._tao_app()

    def _ghi(self, ten):
        self.su_kien.append((time.time(), ten))

    def _tao_app(self):
        app = FastAPI()
        mc = self

        @app.websocket("/ws/camera")
        async def camera(ws: WebSocket):
            if mc._camera is not None or mc._tu_choi > 0:
                mc._tu_choi = max(0, mc._tu_choi - 1)
                await ws.close(code=1013)
                return
            await ws.accept()
            mc._camera = ws
            mc.camera_ket_noi += 1
            mc._ghi("cam_noi")
            dong_sau = mc.kb.get("dong_camera_sau_s")
            if dong_sau is not None and not mc._da_dong_camera:
                mc._da_dong_camera = True

                async def _dong():
                    await asyncio.sleep(dong_sau)
                    with contextlib.suppress(Exception):
                        await ws.close()
                asyncio.create_task(_dong())
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if msg.get("bytes") is not None:
                        mc.khung.append((time.time(), msg["bytes"]))
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                mc._camera = None
                mc._ghi("cam_ngat")

        @app.websocket("/ws_audio")
        async def am_thanh(ws: WebSocket):
            await ws.accept()
            restart = mc.kb.get("restart_sau_s")
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if msg.get("text") is not None:
                        s = msg["text"]
                        mc.tin_mic.append(s)
                        if s.startswith("PROMPT:"):
                            mc.prompt_khi_co_camera.append(mc._camera is not None)
                            await ws.send_text("OK:PROMPT_ACCEPTED")
                        elif s.strip().upper() == "START":
                            await ws.send_text("OK:STARTED")
                            if mc.kb.get("restart_moi_s") and not getattr(ws, "_dang_restart", False):
                                ws._dang_restart = True     # như khóa ASR giả: lỗi -> RESTART mãi

                                async def _restart_mai():
                                    with contextlib.suppress(Exception):
                                        while not mc._dung:
                                            await asyncio.sleep(mc.kb["restart_moi_s"])
                                            await ws.send_text("RESTART")
                                asyncio.create_task(_restart_mai())
                            if restart is not None:
                                restart = None

                                async def _restart():
                                    await asyncio.sleep(mc.kb["restart_sau_s"])
                                    with contextlib.suppress(Exception):
                                        await ws.send_text("RESTART")
                                asyncio.create_task(_restart())
                    elif msg.get("bytes") is not None:
                        mc.mic_nhi_phan.append(len(msg["bytes"]))
            except (WebSocketDisconnect, RuntimeError):
                pass

        @app.get("/stream.wav")
        async def loa():
            mc.loa_ket_noi += 1
            lan = mc.loa_ket_noi

            async def gen():
                yield _wav_header_unknown_size(8000, 1, 2)
                if lan == 1:
                    mc.t_loa_bat_dau = time.time()
                    for cho, pcm in mc.kb.get("loa", []):
                        await asyncio.sleep(cho)
                        yield pcm
                    if mc.kb.get("dong_loa_lan_dau"):
                        return
                while not mc._dung:
                    await asyncio.sleep(0.05)
            return StreamingResponse(gen(), media_type="audio/wav")

        @app.websocket("/ws/viewer")
        async def xem(ws: WebSocket):
            await ws.accept()
            ok, jpg = cv2.imencode(".jpg", np.full((480, 640, 3), 90, np.uint8))
            try:
                while not mc._dung:
                    await ws.send_bytes(jpg.tobytes())
                    await asyncio.sleep(0.1)
            except Exception:
                pass

        return app

    def _nghe_udp(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", self.cong_udp))
        s.settimeout(0.1)
        while not self._dung:
            try:
                d, _ = s.recvfrom(4096)
                self.imu.append(json.loads(d.decode()))
            except socket.timeout:
                continue
        s.close()

    def _ghi_nhat_ky_cau(self):
        bat_dau = time.time()
        for cho, cau in self.kb.get("cau", []):
            bat_dau += cho
            while time.time() < bat_dau and not self._dung:
                time.sleep(0.01)
            with open(self.kb["nhat_ky"], "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(t=time.time(), cau=cau, phat=True, ly_do=""), ensure_ascii=False) + "\n")

    def __enter__(self):
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=self.cong, log_level="error", ws="websockets")
        self._server = uvicorn.Server(cfg)
        self._luong = threading.Thread(target=self._server.run, daemon=True)
        self._luong.start()
        threading.Thread(target=self._nghe_udp, daemon=True).start()
        if self.kb.get("cau"):
            threading.Thread(target=self._ghi_nhat_ky_cau, daemon=True).start()
        han = time.time() + 10
        while not self._server.started and time.time() < han:
            time.sleep(0.02)
        return self

    def __exit__(self, *a):
        self._dung = True
        self._server.should_exit = True
        self._luong.join(timeout=5)


def chay_may_chu_gia(kich_ban=None) -> MayChuGia:
    return MayChuGia(kich_ban)
