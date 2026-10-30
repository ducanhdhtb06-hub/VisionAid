# -*- coding: utf-8 -*-
"""Thử module quan sát (quan_sat.py) với API thật, không cần kính.

  python tools/thu_quan_sat.py anh.jpg
  python tools/thu_quan_sat.py video.MOV                            # tự nghe câu hỏi trong video (PhoWhisper trên máy)
  python tools/thu_quan_sat.py video.MOV --giay 98.6 --hoi "Cái gì chắn đường?"   # tự đặt câu hỏi và mốc giây
  python tools/thu_quan_sat.py video.MOV --giay 10 20 30 --doc      # nhiều mốc, có tạo tiếng (.wav)

Không có --hoi: công cụ tìm các đoạn có người nói trong video, chép thành chữ bằng PhoWhisper (chạy trên máy, như kính
thật nghe câu hỏi), câu nào là câu quan sát ("phía trước có gì", "chữ này là gì"...) thì lấy khung hình ngay sau đó để hỏi.

Làm nóng trước (như máy chủ lúc khởi động), in câu trả lời và thời gian (hỏi -> tiếng đầu); --doc lưu tiếng đọc vào test_out/quan_sat/ để nghe.
Cần GEMINI_API_KEY trong .env (chỉ thử nhánh Gemini; nhánh Qwen chạy qua app_main).
"""
import argparse
import asyncio
import os
import subprocess
import sys
import time
import wave

import cv2
import numpy as np
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


def nghe_cau_hoi(video):
    """Âm thanh của video -> [(bắt đầu, kết thúc, chữ)] cho từng đoạn có người nói (PhoWhisper chạy trên máy)."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    doan = Q.cat_doan_noi(a, 16000)
    if not doan:
        return []
    import torch
    from transformers import pipeline
    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=os.getenv("AIGLASS_ASR_MODEL", "vinai/PhoWhisper-small"),
                   device=0 if cuda else -1, dtype=torch.float16 if cuda else torch.float32)
    kq = []
    for t0, t1 in doan:
        x = a[int(max(t0 - 0.2, 0) * 16000):int((t1 + 0.2) * 16000)]
        chu = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "vi"})["text"].strip()
        kq.append((t0, t1, chu))
    return kq


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nguon", help="ảnh hoặc video")
    ap.add_argument("--giay", type=float, nargs="*", help="mốc giây trong video (bỏ trống nếu là ảnh)")
    ap.add_argument("--hoi", default=None, help="câu hỏi; bỏ trống = tự nghe câu hỏi trong video")
    ap.add_argument("--doc", action="store_true", help="tạo tiếng đọc bằng Gemini TTS, lưu .wav")
    a = ap.parse_args()
    load_dotenv()
    os.makedirs("test_out/quan_sat", exist_ok=True)
    print(f"model nhìn: {Q.GEMINI_MODEL} | model đọc: {Q.GEMINI_TTS_MODEL}")
    viec = [(g, a.hoi) for g in (a.giay or [None])] if a.hoi else []
    if not a.hoi:
        for t0, t1, chu in nghe_cau_hoi(a.nguon):
            la = Q.la_cau_quan_sat(chu)
            print(f"nghe được [{t0:.1f}–{t1:.1f}s]: \"{chu}\"  ->  {'câu quan sát' if la else 'bỏ qua (không phải câu quan sát)'}")
            if la:
                viec.append((t1 + 0.3, chu))                 # khung hình ngay sau khi người dùng hỏi xong
        if not viec:
            raise SystemExit("Không nghe thấy câu quan sát nào trong video. Dùng --hoi và --giay để tự đặt.")
    await Q.lam_nong()
    for g, hoi in viec:
        t = time.time()
        cau = await Q.mo_ta_gemini(anh_jpeg(a.nguon, g), hoi)
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
