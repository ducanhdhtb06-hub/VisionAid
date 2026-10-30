# -*- coding: utf-8 -*-
"""Thử module quan sát (quan_sat.py) với API thật, không cần kính.

  python tools/thu_quan_sat.py anh.jpg
  python tools/thu_quan_sat.py video.MOV --giay 98.6 --hoi "Cái gì chắn đường?"
  python tools/thu_quan_sat.py video.MOV --giay 10 20 30 --doc      # nhiều mốc, có tạo tiếng (.wav)

Làm nóng trước (như máy chủ lúc khởi động), in câu trả lời và thời gian (hỏi -> tiếng đầu); --doc lưu tiếng đọc vào test_out/quan_sat/ để nghe.
Cần GEMINI_API_KEY trong .env (chỉ thử nhánh Gemini; nhánh Qwen chạy qua app_main).
"""
import argparse
import asyncio
import os
import sys
import time
import wave

import cv2
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quan_sat as Q  # noqa: E402


def anh_jpeg(duong, giay):
    """Ảnh (jpg/png) hoặc một khung của video tại `giay` -> JPEG 640 px như kính gửi."""
    if giay is None:
        img = cv2.imread(duong)
    else:
        cap = cv2.VideoCapture(duong)
        cap.set(cv2.CAP_PROP_POS_MSEC, giay * 1000)
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {duong} {'' if giay is None else f'@{giay}s'}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (640, round(h * 640 / w)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nguon", help="ảnh hoặc video")
    ap.add_argument("--giay", type=float, nargs="*", help="mốc giây trong video (bỏ trống nếu là ảnh)")
    ap.add_argument("--hoi", default="Phía trước có gì?")
    ap.add_argument("--doc", action="store_true", help="tạo tiếng đọc bằng Gemini TTS, lưu .wav")
    a = ap.parse_args()
    load_dotenv()
    os.makedirs("test_out/quan_sat", exist_ok=True)
    print(f"model nhìn: {Q.GEMINI_MODEL} | model đọc: {Q.GEMINI_TTS_MODEL} | hỏi: {a.hoi}")
    await Q.lam_nong()
    for g in (a.giay or [None]):
        t = time.time()
        cau = await Q.mo_ta_gemini(anh_jpeg(a.nguon, g), a.hoi)
        t_cau = time.time() - t
        nhan = os.path.basename(a.nguon) + ("" if g is None else f" @{g:g}s")
        print(f"\n[{nhan}] {t_cau:.1f}s: {cau}")
        if a.doc and cau:
            pcm, t_dau = b"", None
            async for manh in Q.doc_gemini_dan(cau):
                t_dau = t_dau or time.time() - t
                pcm += manh
            ra = f"test_out/quan_sat/{os.path.splitext(os.path.basename(a.nguon))[0]}_{g or 0:g}.wav"
            with wave.open(ra, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(pcm)
            print(f"   hỏi -> tiếng đầu {t_dau:.1f}s | {len(pcm) / 48000:.1f}s tiếng: {ra}")

if __name__ == "__main__":
    asyncio.run(main())
