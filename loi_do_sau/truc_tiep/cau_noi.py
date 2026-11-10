"""Cầu nối chế độ B: iPhone kẹp mũ làm kính -> ESP32 ảo -> app_main THẬT.

iPhone mở trang thu_kinh.html (giao thức điện thoại như may_chu: gói [t_chụp][id][JPEG], trả JSON + WAV). Cầu nối
biến mỗi khung thành khung "như ESP32" (HoSoAnh), đóng vai ESP32 với app_main (KenhAnh /ws/camera, KenhLoa
/stream.wav, KenhMic "PROMPT:" để bật/tắt dẫn đường, IMU, xem) và cắt từng câu từ /stream.wav gửi về tai nghe iPhone.
iPhone rớt mạng (không bấm Dừng) -> đóng kênh ảnh ngay để app_main phản ứng như kính thật rớt (báo "Mất hình").
Ghi thư mục phiên cùng dạng phat_lai (xuat_video.py dùng được).
"""
import asyncio
import contextlib
import datetime
import io
import json
import os
import secrets
import sys
import time
import wave

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(GOC, "OpenAIglasses_for_Navigation", "tools"))
from esp32_ao import danh_gia  # noqa: E402
from esp32_ao.giao_thuc import KenhAnh, KenhImu, KenhLoa, KenhMic, KenhXem  # noqa: E402
from esp32_ao.ho_so_anh import HoSoAnh, doc_ho_so  # noqa: E402
from loi_do_sau.truc_tiep.goi_tin import mo_goi  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CAU_BAT, CAU_TAT = "bắt đầu dẫn đường", "dừng dẫn đường"


def _wav(pcm: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def _cau_luc(nhat_ky, t):
    """Câu phat=True gần nhất có giờ <= t (trong 2 s) — luồng /stream.wav chỉ có tiếng, chữ lấy từ nhật ký."""
    tim = [c for c in danh_gia._jsonl(nhat_ky) if c.get("phat") and 0 <= t - c["t"] <= 2.0] if nhat_ky else []
    return tim[-1]["cau"] if tim else "(không rõ câu)"


def tao_app_cau_noi(ma: str, thu_muc_phien: str, ho_so: str, host_may_chu: str = "127.0.0.1",
                    cong_may_chu: int = 8081, cong_udp: int = 12345, nhat_ky_may_chu: str | None = None,
                    cau_bat: str = CAU_BAT, cau_tat: str = CAU_TAT) -> FastAPI:
    app = FastAPI()
    cfg = doc_ho_so(ho_so)
    h, c = host_may_chu, cong_may_chu

    @app.get("/")
    async def trang():                # app VisionAid (mã QR mở vào đây)
        return FileResponse(os.path.join(STATIC, "app.html"))

    @app.get("/thu")
    async def trang_thu():            # trang thử cũ cho người phát triển (cùng lõi kinh_loi.js)
        return FileResponse(os.path.join(STATIC, "thu_kinh.html"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")   # âm báo trang tải lúc bấm bắt đầu

    hien = {"ws": None, "anh": None}    # kết nối iPhone đang giữ camera của app_main

    @app.websocket("/ws")
    async def ws_kinh(ws: WebSocket):
        await ws.accept()               # accept rồi mới đóng: trình duyệt nhận được mã 1008 ("Sai mã")
        if not secrets.compare_digest(ws.query_params.get("k", ""), ma):
            await ws.close(code=1008)
            return
        # kết nối mới nhất thắng (lỗi thật 2026-10-28: 2 tab -> kết nối cũ giữ camera, kết nối mới bị app_main từ chối
        # 1013 mãi, 0 khung tới máy chủ): đóng kết nối cũ và nhả camera của nó ngay (không chờ trình duyệt cũ trả lời)
        cu, anh_cu = hien["ws"], hien["anh"]
        hien["ws"], hien["anh"] = ws, None
        if cu is not None:
            with contextlib.suppress(Exception):
                await cu.close(code=1012)
        if anh_cu is not None:
            await anh_cu.dong()
        thu_muc = os.path.join(thu_muc_phien, datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        os.makedirs(thu_muc, exist_ok=True)
        anh, loa = KenhAnh(h, c), KenhLoa(h, c, os.path.join(thu_muc, "loa.wav"))
        if hien["ws"] is ws:
            hien["anh"] = anh
        xem, imu, mic = KenhXem(h, c, thu_muc), KenhImu(h, cong_udp), KenhMic(h, c, "lenh")
        nen = [asyncio.create_task(x.chay()) for x in (loa, xem, imu)]
        nen.append(asyncio.create_task(anh.ket_noi(cho_s=30)))   # như bo thật: nối camera ngay lúc khởi động
        await mic.ket_noi(cho_s=10)
        hs = HoSoAnh(cfg)
        o = {"khung": None, "t_dau": None, "id": None, "t_chup": None}
        co = asyncio.Event()
        f_khung = open(os.path.join(thu_muc, "khung.jsonl"), "w", encoding="utf-8")
        f_ph = open(os.path.join(thu_muc, "phan_hoi.jsonl"), "w", encoding="utf-8")

        def ghi(id_, t_dt, t_den, t_gui, bo):
            f_khung.write(json.dumps(dict(id=id_, t_video=(t_dt - o["t_dau"]) / 1000.0, t_chup=t_den, t_gui=t_gui,
                                          bo=bo)) + "\n")

        async def gui_anh():           # chỉ giữ khung mới nhất khi đang bận (như GRAB_LATEST)
            while True:
                await co.wait()
                co.clear()
                k, o["khung"] = o["khung"], None
                if k is None:
                    continue
                t_den, t_dt, id_, jpeg_vao = k
                bgr = cv2.imdecode(np.frombuffer(jpeg_vao, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if anh.ws is None:
                    await anh.ket_noi(cho_s=5)
                ok = await anh.gui(hs(bgr, t_den))
                ghi(id_, t_dt, t_den, time.time() if ok else None, not ok)
                o["id"], o["t_chup"] = id_, t_dt
                # thu_kinh.html chỉ gửi khung tiếp khi khung trước được xác nhận (không có: chờ 2 s, báo mất kết nối)
                await ws.send_json(dict(id=id_, t_chup=t_dt, ms_xu_ly=round((time.time() - t_den) * 1000), cau=None,
                                        **({} if ok else dict(loi="máy chủ chưa nhận hình"))))

        async def chuyen_tieng():      # mỗi đoạn tiếng của loa -> một câu (JSON) + WAV về iPhone
            da = 0
            while True:
                await asyncio.sleep(0.05)
                while da < len(loa.doan_tieng):
                    dau, cuoi = loa.doan_tieng[da]
                    da += 1
                    await ws.send_json(dict(id=o["id"], t_chup=o["t_chup"], ms_xu_ly=0,
                                            cau=_cau_luc(nhat_ky_may_chu, dau)))
                    await ws.send_bytes(_wav(loa.pcm_trong(dau, cuoi), loa.tan_so or 8000))

        viec = [asyncio.create_task(gui_anh()), asyncio.create_task(chuyen_tieng())]
        t_bat_dau = time.time()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    t_dt, id_, jpeg_vao = mo_goi(msg["bytes"])
                    if o["t_dau"] is None:
                        o["t_dau"] = t_dt
                    if o["khung"] is not None:
                        cu = o["khung"]
                        ghi(cu[2], cu[1], cu[0], None, True)
                    o["khung"] = (time.time(), t_dt, id_, jpeg_vao)
                    co.set()
                elif msg.get("text"):
                    try:
                        tin = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    if tin.get("lenh") == "bat_dau":
                        await anh.ket_noi(cho_s=10)      # app_main chỉ tạo bộ điều phối khi camera đã nối
                        await mic.lenh(cau_bat)
                    elif tin.get("lenh") == "dung":
                        await mic.lenh(cau_tat)
                        await anh.dong()
                    elif tin.get("lenh") == "hoi":   # nút Hỏi chỉ nối ở chế độ --dan-vach / --noi-thang (may_chu)
                        await ws.send_json(dict(hoi="loi", loi_hoi="Chế độ này chưa hỏi được."))
                    elif "phan_hoi" in tin:
                        f_ph.write(json.dumps(dict(tin, t=time.time()), ensure_ascii=False) + "\n")
        finally:
            for v in viec:
                v.cancel()
            await anh.dong()           # iPhone rớt mà không bấm Dừng: kính "mất kết nối" với app_main ngay
            if hien["ws"] is ws:
                hien["ws"], hien["anh"] = None, None
            await mic.dong()
            for x in (loa, xem, imu):
                await x.dong()
            await asyncio.gather(*nen, *viec, return_exceptions=True)
            f_khung.close()
            f_ph.close()
            with open(os.path.join(thu_muc, "loa_doan.json"), "w") as g:
                json.dump(loa.doan_tieng, g)
            with open(os.path.join(thu_muc, "su_kien.json"), "w", encoding="utf-8") as g:
                json.dump(dict(anh=anh.su_kien, mic=mic.tin_nhan, t0_loa=loa.t0), g, ensure_ascii=False)
            cau = [x for x in danh_gia._jsonl(nhat_ky_may_chu) if x["t"] >= t_bat_dau - 1] if nhat_ky_may_chu else []
            with open(os.path.join(thu_muc, "cau.jsonl"), "w", encoding="utf-8") as g:
                g.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in cau)

    return app
