"""Các kênh của ESP32 ảo, bắt chước đúng firmware compile/compile.ino.

- KenhAnh  : WebSocket /ws/camera, mỗi tin nhị phân là một JPEG; mất kết nối/bị từ chối thì thử lại mỗi 1 s
             (loop() của firmware); bíp khi mất kết nối sau khi đã từng nối, tối đa 10 s/lần (bipMatKetNoi).
- KenhMic  : WebSocket /ws_audio. "lenh": chỉ gửi "PROMPT:<câu>" (bật/tắt dẫn đường không qua ASR);
             "firmware": gửi "START" rồi 640 byte (20 ms PCM16 16 kHz) im lặng mỗi 20 ms, "RESTART" -> gửi lại "START".
- KenhImu  : UDP 50 Hz, JSON {"ts","temp_c","accel":{x,y,z},"gyro":{x,y,z}} như taskImuLoop.
- KenhLoa  : HTTP GET /stream.wav (đọc được chunked như makeBodyReader), đọc khối RIFF/fmt/data, ghi PCM theo giờ phát.
- KenhXem  : WebSocket /ws/viewer, lưu ảnh máy chủ vẽ (để xuất video).
Mọi thời điểm là time.time() trên cùng máy với máy chủ.
"""
import asyncio
import contextlib
import json
import os
import struct
import time
import wave

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

THU_LAI_S = 1.0          # firmware: "[WS-CAM] retry in 1s"
BIP_CACH_S = 10.0        # bipMatKetNoi tự giới hạn 10 s/lần
KHOI_MIC = 640           # BYTES_PER_CHUNK = 16000 * 20 / 1000 * 2
CHO_NOI_LAI_LOA_S = 0.3  # taskHttpPlay: cli.stop(); delay(300)


def _bat_ws(url):
    return connect(url, max_size=None, open_timeout=3, ping_interval=None, close_timeout=1)


class KenhAnh:
    def __init__(self, host: str, cong: int = 8081, thu_lai_s: float = THU_LAI_S):
        self.url = f"ws://{host}:{cong}/ws/camera"
        self.thu_lai_s = thu_lai_s
        self.ws = None
        self.lenh_nhan: list[tuple[float, str]] = []
        self.su_kien: list[tuple[float, str]] = []
        self._da_noi = False
        self._t_bip = -1e9
        self._cam_den = 0.0      # không nối lại trước thời điểm này (mô phỏng rớt mạng)
        self._dung = False
        self._task = None

    def _ghi(self, ten):
        self.su_kien.append((time.time(), ten))

    def _mat(self):
        if self.ws is None:
            return
        self.ws = None
        self._ghi("mat_ket_noi")
        t = time.time()
        if self._da_noi and t - self._t_bip >= BIP_CACH_S:
            self._t_bip = t
            self._ghi("bip")

    async def _vong(self):
        while not self._dung:
            if self.ws is not None or time.time() < self._cam_den:
                await asyncio.sleep(0.02)
                continue
            try:
                ws = await _bat_ws(self.url)
            except InvalidStatus:
                self._ghi("bi_tu_choi_1013")
                await asyncio.sleep(self.thu_lai_s)
                continue
            except (OSError, asyncio.TimeoutError, WebSocketException):
                await asyncio.sleep(self.thu_lai_s)
                continue
            self.ws, self._da_noi = ws, True
            self._ghi("ket_noi")
            try:
                async for m in ws:
                    if isinstance(m, str):
                        self.lenh_nhan.append((time.time(), m))
            except ConnectionClosed:
                pass
            if self.ws is ws:
                self._mat()
                if not self._dung:
                    await asyncio.sleep(self.thu_lai_s)

    async def ket_noi(self, cho_s: float = 10.0) -> bool:
        """Bật vòng tự nối lại (như loop() của firmware) và chờ tới khi nối được."""
        if self._task is None:
            self._task = asyncio.create_task(self._vong())
        han = time.time() + cho_s
        while self.ws is None and time.time() < han:
            await asyncio.sleep(0.02)
        return self.ws is not None

    async def gui(self, jpeg: bytes) -> bool:
        ws = self.ws
        if ws is None:
            return False
        try:
            await ws.send(jpeg)
            return True
        except ConnectionClosed:
            return False

    async def ngat(self, den_t: float = 0.0):
        """Mất kết nối như rớt WiFi: đóng thật (máy chủ thấy ngắt), không nối lại trước den_t."""
        self._cam_den = den_t
        ws = self.ws
        if ws is not None:
            self._mat()
            with contextlib.suppress(Exception):
                await ws.close()

    async def dong(self):
        """Kết thúc phiên (không tính là mất kết nối, không bíp)."""
        self._dung = True
        ws, self.ws = self.ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task


class KenhMic:
    def __init__(self, host: str, cong: int = 8081, che_do: str = "lenh"):
        assert che_do in ("tat", "lenh", "firmware")
        self.url = f"ws://{host}:{cong}/ws_audio"
        self.che_do = che_do
        self.ws = None
        self.dem_start = 0
        self.tin_nhan: list[tuple[float, str]] = []
        self._ok = asyncio.Event()
        self._tasks = []

    async def ket_noi(self, cho_s: float = 10.0) -> bool:
        if self.che_do == "tat":
            return True
        han = time.time() + cho_s
        while self.ws is None and time.time() < han:
            try:
                self.ws = await _bat_ws(self.url)
            except (OSError, asyncio.TimeoutError, WebSocketException):
                await asyncio.sleep(THU_LAI_S)
        if self.ws is None:
            return False
        self._tasks.append(asyncio.create_task(self._nghe()))
        if self.che_do == "firmware":
            await self._start()
            self._tasks.append(asyncio.create_task(self._im_lang()))
        return True

    async def _start(self):
        await self.ws.send("START")
        self.dem_start += 1

    async def _nghe(self):
        try:
            async for m in self.ws:
                if not isinstance(m, str):
                    continue
                self.tin_nhan.append((time.time(), m))
                if m.startswith("OK:PROMPT_ACCEPTED"):
                    self._ok.set()
                elif m.strip() == "RESTART" and self.che_do == "firmware":
                    await self._start()
        except ConnectionClosed:
            pass

    async def _im_lang(self):
        loop = asyncio.get_running_loop()
        t0, k, khoi = loop.time(), 0, bytes(KHOI_MIC)
        try:
            while True:
                k += 1
                await asyncio.sleep(max(0.0, t0 + k * 0.02 - loop.time()))
                await self.ws.send(khoi)
        except (ConnectionClosed, asyncio.CancelledError):
            pass

    async def lenh(self, cau: str, cho_s: float = 3.0) -> bool:
        """Gửi "PROMPT:<câu>" (app_main gọi start_ai_with_text_custom) và chờ OK:PROMPT_ACCEPTED."""
        if self.ws is None:
            return False
        self._ok.clear()
        await self.ws.send("PROMPT:" + cau)
        try:
            await asyncio.wait_for(self._ok.wait(), cho_s)
            return True
        except asyncio.TimeoutError:
            return False

    async def dong(self):
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.ws.close()
            self.ws = None


class KenhImu:
    def __init__(self, host: str, cong: int = 12345, hz: float = 50.0, seed: int = 0):
        self.dia_chi = (host, cong)
        self.hz = hz
        self.rng = np.random.default_rng(seed)
        self._dung = False

    def _dung_yen(self, t):
        r = self.rng.normal
        return dict(accel=dict(x=r(0, 0.02), y=r(0, 0.02), z=1.0 + r(0, 0.02)),
                    gyro=dict(x=r(0, 0.5), y=r(0, 0.5), z=r(0, 0.5)))

    async def chay(self, nguon=None):
        """nguon(t) -> {"accel":{x,y,z},"gyro":{x,y,z}}; mặc định đứng yên kèm nhiễu."""
        loop = asyncio.get_running_loop()
        tr, _ = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, remote_addr=self.dia_chi)
        t0, k = loop.time(), 0
        try:
            while not self._dung:
                d = (nguon or self._dung_yen)(loop.time() - t0)
                a, g = d["accel"], d["gyro"]
                goi = {"ts": int(time.time() * 1000) & 0xFFFFFFFF, "temp_c": 35.0,
                       "accel": {k2: round(float(a[k2]), 3) for k2 in "xyz"},
                       "gyro": {k2: round(float(g[k2]), 3) for k2 in "xyz"}}
                tr.sendto(json.dumps(goi).encode())
                k += 1
                await asyncio.sleep(max(0.0, t0 + k / self.hz - loop.time()))
        finally:
            tr.close()

    async def dong(self):
        self._dung = True


class _HetLuong(Exception):
    pass


class KenhLoa:
    """Đọc /stream.wav như taskHttpPlay. Giờ phát của mỗi cửa sổ 20 ms = max(giờ byte tới, giờ phát cửa sổ trước + 20 ms)
    (loa phát tuần tự). Đoạn tiếng: bắt đầu khi RMS vượt ngưỡng >= 2 cửa sổ liền, kết thúc sau >= 300 ms im lặng
    (hoặc 300 ms không có byte nào: máy chủ chỉ gửi khi có tiếng)."""

    def __init__(self, host: str, cong: int, wav_ra: str, nguong_rms: float = 300.0):
        self.host, self.cong, self.wav_ra = host, cong, wav_ra
        self.nguong = nguong_rms
        self.tan_so = None
        self.doan_tieng: list[tuple[float, float]] = []
        self.mat_tieng: list[tuple[float, float]] = []
        self._manh: list[tuple[float, bytes]] = []    # (giờ phát, pcm)
        self._t_phat = 0.0
        self._du = b""
        self._to_lien = 0
        self._dang_noi = None        # giờ bắt đầu đoạn đang nói
        self._t_to_cuoi = 0.0        # giờ kết thúc cửa sổ to gần nhất
        self._dung = False
        self.t0 = None

    def _cua_so(self, t_toi, pcm):
        n = int(self.tan_so * 0.02) * 2
        self._du += pcm
        while len(self._du) >= n:
            khoi, self._du = self._du[:n], self._du[n:]
            t = max(t_toi, self._t_phat)
            self._t_phat = t + 0.02
            self._manh.append((t, khoi))
            x = np.frombuffer(khoi, "<i2").astype(np.float32)
            to = float(np.sqrt(np.mean(x * x))) > self.nguong
            if to:
                self._to_lien += 1
                if self._dang_noi is None and self._to_lien >= 2:
                    self._dang_noi = t - 0.02
                self._t_to_cuoi = t + 0.02
            else:
                self._to_lien = 0
                if self._dang_noi is not None and t + 0.02 - self._t_to_cuoi >= 0.3:
                    self._ket_doan()

    def _ket_doan(self):
        if self._dang_noi is not None:
            self.doan_tieng.append((self._dang_noi, self._t_to_cuoi))
            self._dang_noi = None

    def _kiem_im(self):
        if self._dang_noi is not None and time.time() - max(self._t_to_cuoi, self._t_phat) >= 0.3:
            self._ket_doan()

    async def _doc_dong(self, r):
        return (await r.readline()).decode("latin1").rstrip("\r\n")

    async def _mot_lan(self):
        r, w = await asyncio.open_connection(self.host, self.cong)
        try:
            w.write(f"GET /stream.wav HTTP/1.1\r\nHost: {self.host}:{self.cong}\r\nConnection: keep-alive\r\n\r\n".encode())
            await w.drain()
            chunked, dong = False, await self._doc_dong(r)
            if not dong:
                raise _HetLuong
            while True:
                dong = await self._doc_dong(r)
                u = dong.lower()
                if u.startswith("transfer-encoding:") and "chunked" in u:
                    chunked = True
                if dong == "":
                    break
            con_lai = [0]

            async def doc(n):           # makeBodyReader: bỏ lớp chunked
                ra = b""
                while len(ra) < n:
                    if chunked and con_lai[0] == 0:
                        kich = await self._doc_dong(r)
                        if kich == "":
                            kich = await self._doc_dong(r)
                        con_lai[0] = int(kich.split(";")[0], 16)
                        if con_lai[0] == 0:
                            raise _HetLuong
                    can = n - len(ra) if not chunked else min(n - len(ra), con_lai[0])
                    b = await r.read(can)
                    if not b:
                        raise _HetLuong
                    ra += b
                    if chunked:
                        con_lai[0] -= len(b)
                return ra

            dau = await doc(12)
            if dau[:4] != b"RIFF" or dau[8:12] != b"WAVE":
                raise _HetLuong
            while True:
                ma, co = struct.unpack("<4sI", await doc(8))
                if ma == b"fmt ":
                    fmt = await doc(co)
                    self.tan_so = struct.unpack_from("<I", fmt, 4)[0]
                elif ma == b"data":
                    break
                else:
                    await doc(co)
            if self.mat_tieng and self.mat_tieng[-1][1] is None:
                self.mat_tieng[-1] = (self.mat_tieng[-1][0], time.time())
            while not self._dung:
                try:
                    b = await asyncio.wait_for(doc(640), 0.1)
                except asyncio.TimeoutError:
                    self._kiem_im()
                    continue
                self._cua_so(time.time(), b)
        finally:
            w.close()

    async def chay(self):
        self.t0 = time.time()
        while not self._dung:
            try:
                await self._mot_lan()
            except (_HetLuong, OSError, asyncio.IncompleteReadError, ValueError):
                pass
            if self._dung:
                break
            self._ket_doan()
            self.mat_tieng.append((time.time(), None))
            await asyncio.sleep(CHO_NOI_LAI_LOA_S)
        self._ket_doan()
        self.mat_tieng = [(a, b) for a, b in self.mat_tieng if b is not None]
        self.ghi_wav()

    def pcm_trong(self, dau: float, cuoi: float) -> bytes:
        """PCM các cửa sổ có giờ phát trong [dau, cuoi] (cầu nối cắt từng câu gửi về iPhone)."""
        return b"".join(p for t, p in self._manh if dau - 1e-6 <= t < cuoi + 1e-6)

    def ghi_wav(self):
        """WAV theo giờ phát tính từ t0 (chèn im lặng chỗ không có tiếng) để khớp giờ khi xuất video."""
        sr = self.tan_so or 8000
        out = bytearray()
        for t, pcm in self._manh:
            vi_tri = int(round((t - self.t0) * sr)) * 2
            if vi_tri > len(out):
                out += bytes(vi_tri - len(out))
            out += pcm
        os.makedirs(os.path.dirname(os.path.abspath(self.wav_ra)), exist_ok=True)
        with wave.open(self.wav_ra, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(out))

    async def dong(self):
        self._dung = True


class KenhXem:
    def __init__(self, host: str, cong: int, thu_muc: str):
        self.url = f"ws://{host}:{cong}/ws/viewer"
        self.thu_muc = thu_muc
        self._dung = False

    async def chay(self):
        os.makedirs(os.path.join(self.thu_muc, "xem"), exist_ok=True)
        n = 0
        with open(os.path.join(self.thu_muc, "xem.jsonl"), "a", encoding="utf-8") as f:
            while not self._dung:
                try:
                    async with _bat_ws(self.url) as ws:
                        while not self._dung:
                            try:
                                m = await asyncio.wait_for(ws.recv(), 0.2)
                            except asyncio.TimeoutError:
                                continue
                            if isinstance(m, bytes):
                                ten = f"xem/{n:06d}.jpg"
                                with open(os.path.join(self.thu_muc, ten), "wb") as g:
                                    g.write(m)
                                f.write(json.dumps(dict(t=time.time(), file=ten)) + "\n")
                                n += 1
                except (OSError, asyncio.TimeoutError, WebSocketException):
                    await asyncio.sleep(THU_LAI_S)

    async def dong(self):
        self._dung = True
