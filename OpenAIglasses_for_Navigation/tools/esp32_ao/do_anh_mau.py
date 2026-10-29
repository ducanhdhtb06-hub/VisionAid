#!/usr/bin/env python3
"""Đo ảnh mẫu chụp bằng bo ESP32 thật: bảng lượng tử JPEG, kiểu lấy mẫu màu, dung lượng, và thống kê ảnh
(nhiễu theo độ sáng, độ nét, tương phản, bão hòa) để hiệu chỉnh hồ sơ ảnh của ESP32 ảo.
Chạy: ../.venv/bin/python tools/esp32_ao/do_anh_mau.py test_out/esp32_mau/ov2640
"""
import argparse
import collections
import io
import json
import os

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

LAY_MAU = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def doc_jpeg(nguon) -> dict:
    du_lieu = nguon if isinstance(nguon, (bytes, bytearray)) else open(nguon, "rb").read()
    im = Image.open(io.BytesIO(du_lieu))
    bang = [list(im.quantization[k]) for k in sorted(im.quantization)]
    return dict(rong=im.width, cao=im.height, bang_luong_tu=bang,
                lay_mau=LAY_MAU.get(JpegImagePlugin.get_sampling(im), "khac"), so_byte=len(du_lieu))


def thong_ke(bgr: np.ndarray) -> dict:
    """nhieu: [(mức sáng giữa khoảng 32, sigma kênh G trên các khối 8x8 phẳng nhất)]; do_net: trung bình 5 % độ lớn
    gradient lớn nhất; tuong_phan: độ lệch chuẩn độ sáng; bao_hoa: trung bình S (HSV, 0–255); sang_tb."""
    g = bgr[:, :, 1].astype(np.float32)
    h, w = g.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    khoi = g[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    tb, sd = khoi.mean(1), khoi.std(1)
    nhieu = []
    for b in range(8):
        chon = sd[(tb >= 32 * b) & (tb < 32 * b + 32)]
        if len(chon) >= 5:
            phang = chon[chon <= np.percentile(chon, 50)]
            nhieu.append((32 * b + 16, float(np.median(phang))))
    xam = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(xam, cv2.CV_32F, 1, 0), cv2.Sobel(xam, cv2.CV_32F, 0, 1))
    nguong = np.percentile(mag, 95)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return dict(nhieu=nhieu, do_net=float(mag[mag >= nguong].mean()), tuong_phan=float(xam.std()),
                bao_hoa=float(hsv[:, :, 1].mean()), sang_tb=float(xam.mean()))


def do_thu_muc(thu_muc: str) -> dict:
    tung_anh = []
    for ten in sorted(os.listdir(thu_muc)):
        if not ten.lower().endswith((".jpg", ".jpeg")):
            continue
        p = os.path.join(thu_muc, ten)
        info = doc_jpeg(p)
        bgr = cv2.imdecode(np.frombuffer(open(p, "rb").read(), np.uint8), cv2.IMREAD_COLOR)
        info.update(thong_ke(bgr), file=ten)
        tung_anh.append(info)
    if not tung_anh:
        raise ValueError(f"không có ảnh JPEG trong {thu_muc}")
    bang = collections.Counter(json.dumps(a["bang_luong_tu"]) for a in tung_anh).most_common()
    tong = dict(so_anh=len(tung_anh),
                bang_luong_tu=json.loads(bang[0][0]), so_bang_khac_nhau=len(bang),
                lay_mau=collections.Counter(a["lay_mau"] for a in tung_anh).most_common(1)[0][0],
                kich_thuoc=collections.Counter(f"{a['rong']}x{a['cao']}" for a in tung_anh).most_common(),
                so_byte_trung_vi=float(np.median([a["so_byte"] for a in tung_anh])),
                do_net=float(np.median([a["do_net"] for a in tung_anh])),
                tuong_phan=float(np.median([a["tuong_phan"] for a in tung_anh])),
                bao_hoa=float(np.median([a["bao_hoa"] for a in tung_anh])),
                tung_anh=tung_anh)
    with open(os.path.join(thu_muc, "thong_ke.json"), "w", encoding="utf-8") as f:
        json.dump(tong, f, ensure_ascii=False, indent=1)
    return tong


def main():
    ap = argparse.ArgumentParser(description="Đo ảnh mẫu bo ESP32 thật (bảng lượng tử, lấy mẫu màu, thống kê)")
    ap.add_argument("thu_muc")
    t = do_thu_muc(ap.parse_args().thu_muc)
    print(f"{t['so_anh']} ảnh | lấy mẫu {t['lay_mau']} | kích thước {t['kich_thuoc']} | "
          f"{t['so_bang_khac_nhau']} bộ bảng lượng tử | dung lượng trung vị {t['so_byte_trung_vi']:.0f} B | "
          f"độ nét {t['do_net']:.0f} | tương phản {t['tuong_phan']:.1f} | bão hòa {t['bao_hoa']:.0f}")


if __name__ == "__main__":
    main()
