#!/usr/bin/env python3
"""
Biến video quay bằng điện thoại thành giống camera kính rẻ (XIAO ESP32S3 Sense / ESP32-CAM, cảm biến OV2640)
để thử module né vật cản trước khi có kính thật:
  - 10 khung/giây, ảnh 640x360, nén JPEG mạnh (ESP32 gửi JPEG chất lượng thấp cho đỡ tốn WiFi)
  - nhòe khi đầu chuyển động nhanh (OV2640 phơi sáng lâu), nhiễu hạt, màu nhạt hơn
  - thỉnh thoảng rơi khung (WiFi chập chờn): lặp lại khung trước

Chạy (đứng ở glassai/, đã bật .venv):
  python loi_do_sau/gia_lap_kinh.py my/IMG_7354.MOV                 # -> my/IMG_7354_kinh.mp4
  python loi_do_sau/gia_lap_kinh.py my/IMG_7354.MOV --jpeg 25 --roi 0.1
Rồi chạy ket_hop_3_model.py trên file _kinh.mp4 như video thường.
"""
import argparse
import os

import cv2
import numpy as np


class LamXau:
    """Làm một khung 640x360 giống camera kính rẻ: nhòe theo chuyển động, màu nhạt, nhiễu hạt, nén JPEG.
    Giữ trạng thái giữa các khung (khung trước để đo chuyển động). Không làm rơi khung (máy chủ trực tiếp có rơi thật)."""
    def __init__(self, jpeg=35, nhieu=6.0, seed=0):
        self.jpeg, self.nhieu = jpeg, nhieu
        self.rng = np.random.default_rng(seed)
        self.prev_gray = None

    def __call__(self, img, dt=0.1):
        """dt: số giây từ khung trước. Nhòe = tốc độ quay đầu x thời gian phơi sáng, nên quy độ dịch về mỗi 0,1 s:
        qua 4G khung cách nhau ~0,5 s, độ dịch gấp 5 lần nhưng ảnh không nhòe hơn."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            # nhòe theo hướng chuyển động: đo dịch chuyển toàn ảnh giữa 2 khung
            (dx, dy), _ = cv2.phaseCorrelate(np.float32(self.prev_gray), np.float32(gray))
            k = int(min(15, np.hypot(dx, dy) * 0.6 * 0.1 / max(dt, 0.1)))
            if k >= 2:
                ker = np.zeros((k, k), np.float32)
                ang = np.arctan2(dy, dx)
                c = (k - 1) / 2
                cv2.line(ker, (int(c - np.cos(ang) * c), int(c - np.sin(ang) * c)),
                         (int(c + np.cos(ang) * c), int(c + np.sin(ang) * c)), 1, 1)
                img = cv2.filter2D(img, -1, ker / max(1, ker.sum()))
        self.prev_gray = gray
        # màu nhạt, tương phản thấp hơn điện thoại; nhiễu hạt
        img = cv2.addWeighted(img, 0.85, np.full_like(img, 128), 0.15, 0)
        img = np.clip(img.astype(np.float32) + self.rng.normal(0, self.nhieu, img.shape), 0, 255).astype(np.uint8)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--fps', type=float, default=10, help='khung/giây camera kính gửi lên')
    ap.add_argument('--jpeg', type=int, default=35, help='chất lượng JPEG (ESP32 thường 10–20 theo thang ngược, ~30–40 ở đây)')
    ap.add_argument('--nhieu', type=float, default=6, help='độ lệch chuẩn nhiễu hạt (0–255)')
    ap.add_argument('--roi', type=float, default=0.05, help='tỉ lệ khung bị rơi do WiFi')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(vfps / args.fps))
    out = args.out or os.path.splitext(args.video)[0] + '_kinh.mp4'
    W, H = 640, 360
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), vfps / step, (W, H))
    lx = LamXau(args.jpeg, args.nhieu)
    rng = lx.rng                             # dùng chung bộ số ngẫu nhiên: giữ đúng kết quả như trước khi tách hàm
    last = None
    i = n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        img = lx(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA))
        if last is not None and rng.random() < args.roi:
            img = last                       # rơi khung: máy chủ nhận lại khung cũ
        last = img
        wr.write(img)
        n += 1
    wr.release()
    print(f'Đã ghi {out}: {n} khung, {vfps / step:.0f} khung/giây')


if __name__ == '__main__':
    main()
