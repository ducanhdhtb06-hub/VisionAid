#!/usr/bin/env python3
"""Xuất video chia đôi có tiếng từ một thư mục phiên ESP32 ảo (chế độ A hoặc cầu nối iPhone).

Trái: video quay từ ngoài (--video-trai, lệch giờ --lech-trai) hoặc khung nguồn (--video-nguon) theo khung.jsonl.
Phải: ảnh máy chủ vẽ (xem/). Dải dưới: câu kính nói (chữ có dấu, Pillow + DejaVuSans), hiện 2,5 s từ lúc loa bắt đầu.
Tiếng: loa.wav đặt đúng giờ (su_kien.json "t0_loa"). --loa-nho: lọc 300–3400 Hz như loa nhỏ MAX98357A và trộn
tiếng gốc của video nguồn (tiếng ồn đường) để nghe thử câu có rõ không.
Chạy: ../.venv/bin/python tools/esp32_ao/xuat_video.py <thư mục phiên> --ra ra.mp4 --video-nguon ../my/IMG_7358.MOV
"""
import argparse
import bisect
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS = 10
CAO_DAI = 80
HIEN_S = 2.5


def _jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(d) for d in f if d.strip()]


def _cat_43(img):
    h, w = img.shape[:2]
    rong = min(w, int(round(h * 4 / 3)))
    x0 = (w - rong) // 2
    return cv2.resize(img[:, x0:x0 + rong], (640, 480), interpolation=cv2.INTER_AREA)


def _co_tieng(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", video], capture_output=True, text=True)
    return bool(r.stdout.strip())


class _DocTuan:
    """Đọc video tuần tự theo giờ tăng dần (nhanh hơn tua từng khung)."""

    def __init__(self, video, lech=0.0):
        self.cap = cv2.VideoCapture(video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.lech, self.i, self.khung = lech, -1, None

    def luc(self, t):
        can = int((t + self.lech) * self.fps)
        while self.i < can:
            ok, img = self.cap.read()
            if not ok:
                break
            self.i += 1
            self.khung = img
        return self.khung


def xuat(thu_muc: str, ra_mp4: str, video_trai: str | None = None, lech_trai_s: float = 0.0,
         video_nguon: str | None = None, loa_nho: bool = False) -> str:
    if not os.path.exists(FONT):
        raise FileNotFoundError(f"thiếu font tiếng Việt {FONT} (sudo apt install fonts-dejavu-core)")
    khung = sorted((k for k in _jsonl(os.path.join(thu_muc, "khung.jsonl")) if not k.get("bo")), key=lambda k: k["t_chup"])
    if not khung:
        raise ValueError("thư mục phiên không có khung nào")
    T = khung[0]["t_chup"]
    dai = khung[-1]["t_chup"] - T + 1.0 / FPS
    t_k = [k["t_chup"] for k in khung]
    xem = _jsonl(os.path.join(thu_muc, "xem.jsonl"))
    t_x = [x["t"] for x in xem]
    doan = sorted(a for a, _ in json.load(open(os.path.join(thu_muc, "loa_doan.json"))))
    phu_de = []
    for c in _jsonl(os.path.join(thu_muc, "cau.jsonl")):
        if not c.get("phat"):
            continue
        i = bisect.bisect_left(doan, c["t"])
        dau = doan[i] if i < len(doan) and doan[i] - c["t"] <= 2.0 else c["t"]
        phu_de.append((dau, c["cau"]))
    font = ImageFont.truetype(FONT, 38)
    nguon = _DocTuan(video_nguon) if video_nguon else None
    trai = _DocTuan(video_trai, lech_trai_s) if video_trai else None
    with tempfile.TemporaryDirectory() as tmp:
        hinh = os.path.join(tmp, "hinh.mp4")
        vw = cv2.VideoWriter(hinh, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 480 + CAO_DAI))
        for n in range(int(round(dai * FPS))):
            t = T + n / FPS
            khung_hinh = np.zeros((480 + CAO_DAI, 1280, 3), np.uint8)
            if trai is not None:
                img = trai.luc(t - T)
            elif nguon is not None:
                k = khung[max(0, bisect.bisect_right(t_k, t) - 1)]
                img = nguon.luc(k["t_video"])
            else:
                img = None
            if img is not None:
                khung_hinh[:480, :640] = _cat_43(img)
            j = bisect.bisect_right(t_x, t) - 1
            if j >= 0:
                anh = cv2.imread(os.path.join(thu_muc, xem[j]["file"]))
                if anh is not None:
                    khung_hinh[:480, 640:] = cv2.resize(anh, (640, 480))
            chu = [c for d, c in phu_de if d <= t < d + HIEN_S]
            if chu:
                pil = Image.fromarray(khung_hinh[480:, :, ::-1].copy())
                ImageDraw.Draw(pil).text((20, 18), chu[-1], font=font, fill=(255, 255, 255))
                khung_hinh[480:] = np.array(pil)[:, :, ::-1]
            vw.write(khung_hinh)
        vw.release()
        su_kien = json.load(open(os.path.join(thu_muc, "su_kien.json"), encoding="utf-8"))
        lech = T - su_kien.get("t0_loa", T)          # giây: loa.wav bắt đầu trước video bao lâu
        lenh = ["ffmpeg", "-v", "error", "-y", "-i", hinh]
        if lech >= 0:
            lenh += ["-ss", f"{lech:.3f}", "-i", os.path.join(thu_muc, "loa.wav")]
        else:
            lenh += ["-itsoffset", f"{-lech:.3f}", "-i", os.path.join(thu_muc, "loa.wav")]
        bo_loc = "[1:a]aresample=16000"
        if loa_nho:
            bo_loc += ",highpass=f=300,highpass=f=300,lowpass=f=3400"
        if loa_nho and video_nguon and _co_tieng(video_nguon):
            k0 = khung[0]["t_video"]
            lenh += ["-ss", f"{k0:.3f}", "-i", video_nguon]
            bo_loc += "[k];[2:a]aresample=16000,pan=mono|c0=c0[n];[k][n]amix=inputs=2:normalize=0[a]"
        else:
            bo_loc += "[a]"
        lenh += ["-filter_complex", bo_loc, "-map", "0:v", "-map", "[a]", "-t", f"{dai:.3f}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", ra_mp4]
        subprocess.run(lenh, check=True)
    return ra_mp4


def main():
    ap = argparse.ArgumentParser(description="Xuất video chia đôi có tiếng từ thư mục phiên ESP32 ảo")
    ap.add_argument("thu_muc")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--video-nguon", default=None)
    ap.add_argument("--video-trai", default=None, help="video quay người đi từ bên ngoài (tùy chọn)")
    ap.add_argument("--lech-trai", type=float, default=0.0, help="giây lệch của video bên trái")
    ap.add_argument("--loa-nho", action="store_true")
    a = ap.parse_args()
    print(xuat(a.thu_muc, a.ra, a.video_trai, a.lech_trai, a.video_nguon, a.loa_nho))


if __name__ == "__main__":
    main()
