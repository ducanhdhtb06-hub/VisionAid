#!/usr/bin/env python3
"""Hiệu chỉnh hồ sơ ảnh ESP32 cho khớp ảnh mẫu chụp bằng bo thật (dùng trong ngày kiểm chứng).

Đo thống kê ảnh mẫu (do_anh_mau) -> dò lưới tham số hồ sơ trên khung video iPhone (cảnh tương tự) sao cho thống kê
ảnh mô phỏng gần ảnh mẫu nhất. Hai cấu hình cách nhau < 5 % thì chọn cái NHIỄU/MỜ hơn (tránh lạc quan).
Chạy: ../.venv/bin/python tools/esp32_ao/hieu_chinh.py --mau test_out/esp32_mau/ov2640 --ho-so ov2640_xiao \
        --video ../my/IMG_7354.MOV ../my/IMG_7356.MOV ../my/IMG_7358.MOV --ghi
"""
import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esp32_ao.do_anh_mau import do_thu_muc, thong_ke  # noqa: E402
from esp32_ao.ho_so_anh import THU_MUC_HO_SO, HoSoAnh, doc_ho_so  # noqa: E402

CHI_SO = ("do_net", "tuong_phan", "bao_hoa", "nhieu_tb")
XAU_HON = ("nhieu_doc", "nhieu_photon", "do_net_sigma", "t_phoi_sang_toi_da_ms", "t_phoi_sang_ms", "t_doc_anh_ms")


def thong_ke_tap(anh: list) -> dict:
    tk = [thong_ke(a) for a in anh]
    nhieu = [s for t in tk for _, s in t["nhieu"]]
    return dict(do_net=float(np.median([t["do_net"] for t in tk])),
                tuong_phan=float(np.median([t["tuong_phan"] for t in tk])),
                bao_hoa=float(np.median([t["bao_hoa"] for t in tk])),
                nhieu_tb=float(np.median(nhieu)) if nhieu else 0.0)


def khoang_cach(a: dict, b: dict) -> float:
    """Tổng chênh lệch tương đối của các chỉ số (0 = giống hệt)."""
    if "nhieu_tb" not in a:
        a = dict(a, nhieu_tb=float(np.median([s for _, s in a["nhieu"]])) if a.get("nhieu") else 0.0)
    if "nhieu_tb" not in b:
        b = dict(b, nhieu_tb=float(np.median([s for _, s in b["nhieu"]])) if b.get("nhieu") else 0.0)
    return float(sum(abs(a[k] - b[k]) / max(1e-6, abs(b[k]), 1.0) for k in CHI_SO))


def _mo_phong(khung, cfg):
    return thong_ke_tap([cv2.imdecode(np.frombuffer(HoSoAnh(cfg, seed=9)(k, 0.0), np.uint8), 1) for k in khung])


def khop(khung_nguon: list, tk_mau: dict, cfg: dict, luoi: dict) -> tuple[dict, float]:
    khoa = list(luoi)
    ung_vien = []
    for gia in itertools.product(*(luoi[k] for k in khoa)):
        c = dict(cfg, **dict(zip(khoa, gia)))
        ung_vien.append((khoang_cach(_mo_phong(khung_nguon, c), tk_mau), c))
    tot = min(d for d, _ in ung_vien)
    gan = [(d, c) for d, c in ung_vien if d <= tot * 1.05 + 1e-9]
    d, c = max(gan, key=lambda dc: tuple(dc[1].get(k, 0) or 0 for k in XAU_HON))   # hòa -> chọn xấu hơn
    return c, d


def bao_cao(tk_mau: dict, truoc: dict, sau: dict, nguon: str) -> str:
    dong = ["| chỉ số | ảnh mẫu | mô phỏng trước | mô phỏng sau |", "|---|---|---|---|"]
    for k in CHI_SO:
        dong.append(f"| {k} | {tk_mau[k]:.2f} | {truoc[k]:.2f} | {sau[k]:.2f} |")
    return f"Nguồn ảnh mẫu: {nguon}\n\n" + "\n".join(dong) + "\n"


def _khung_video(videos, so=30):
    ra = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, n - 1, max(1, so // len(videos))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, img = cap.read()
            if ok:
                ra.append(img)
    return ra


def main():
    ap = argparse.ArgumentParser(description="Hiệu chỉnh hồ sơ ảnh ESP32 theo ảnh mẫu bo thật")
    ap.add_argument("--mau", required=True, help="thư mục ảnh mẫu JPEG chụp bằng bo thật")
    ap.add_argument("--ho-so", required=True)
    ap.add_argument("--video", nargs="+", required=True, help="video iPhone (dev) cảnh tương tự")
    ap.add_argument("--ghi", action="store_true", help="ghi tham số mới + bảng lượng tử vào hồ sơ")
    a = ap.parse_args()
    tong = do_thu_muc(a.mau)
    cfg = doc_ho_so(a.ho_so)
    if tong["so_anh"] >= 1:
        cfg.update(bang_luong_tu=tong["bang_luong_tu"], lay_mau=tong["lay_mau"],
                   bang_luong_tu_nguon=f"chép từ {tong['so_anh']} ảnh mẫu ({a.mau})", lay_mau_nguon="đọc từ ảnh mẫu")
    tk_mau = dict(do_net=tong["do_net"], tuong_phan=tong["tuong_phan"], bao_hoa=tong["bao_hoa"],
                  nhieu_tb=float(np.median([s for t in tong["tung_anh"] for _, s in t["nhieu"]] or [0.0])))
    khung = _khung_video(a.video)
    truoc = _mo_phong(khung, cfg)
    if tong["so_anh"] >= 3:
        luoi = dict(nhieu_doc=[1.0, 2.0, 4.0], nhieu_photon=[0.3, 0.6, 1.0], do_net_sigma=[0.4, 0.8, 1.5],
                    tuong_phan=[1.0, 1.1, 1.25], bao_hoa=[1.0, 1.15, 1.3])
        cfg, _ = khop(khung, tk_mau, cfg, luoi)
        for k in luoi:
            cfg[f"{k}_nguon"] = f"hiệu chỉnh từ {tong['so_anh']} ảnh mẫu"
    else:
        print("Ít hơn 3 ảnh mẫu: chỉ chép bảng lượng tử, giữ các tham số ước lượng")
    sau = _mo_phong(khung, cfg)
    bc = bao_cao(tk_mau, truoc, sau, a.mau)
    with open(os.path.join(a.mau, "bao_cao_hieu_chinh.md"), "w", encoding="utf-8") as f:
        f.write(bc)
    print(bc)
    if a.ghi:
        with open(os.path.join(THU_MUC_HO_SO, f"{a.ho_so}.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
