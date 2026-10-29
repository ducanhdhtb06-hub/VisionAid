#!/usr/bin/env python3
"""Quét TỪNG YẾU TỐ quanh cấu hình xấu-hợp-lý (chế độ A chạy thời gian thật: đủ ma trận 72 cấu hình ~17 giờ,
từng yếu tố 9 cấu hình ~2 giờ trên 4 video dev). Chạy lần lượt, chạy tiếp được sau khi bị ngắt (bỏ qua cấu hình
đã có bang_so.json). Hàng HỎNG: có lệnh sai chiều hoặc trễ p95 > 1500 ms. KHÔNG dùng video giữ riêng.
Chạy: ../.venv/bin/python tools/esp32_ao/quet.py khiem.mp4 ../my/IMG_7354.MOV ../my/IMG_7356.MOV ../my/IMG_7358.MOV \
        --ra test_out/esp32_ao/quet --nhan test_cham_diem/nhan_vach_dev.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esp32_ao import phat_lai  # noqa: E402

GOC = dict(ho_so="ov2640_xiao", hfov_ong_kinh=56, hfov_may_chu=56, jpeg=17, phoi_sang="co_dinh", mang="kem", fps_gui=8)
TRE_HONG_MS = 1500


def cau_hinh_quet() -> list[dict]:
    doi = [("goc", {}),
           ("hfov60", dict(hfov_ong_kinh=60, hfov_may_chu=60)), ("hfov66", dict(hfov_ong_kinh=66, hfov_may_chu=66)),
           ("lech_hfov", dict(hfov_may_chu=70)), ("ov3660", dict(ho_so="ov3660_xiao")), ("jpeg12", dict(jpeg=12)),
           ("phoi_tu_dong", dict(phoi_sang="tu_dong")), ("mang_tot", dict(mang="tot")), ("mang_rot", dict(mang="rot"))]
    return [dict(GOC, ten=ten, **d) for ten, d in doi]


def chay_quet(videos: list[str], ra: str, nhan: str | None) -> list[dict]:
    ket_qua = []
    for ch in cau_hinh_quet():
        for v in videos:
            ten_v = os.path.splitext(os.path.basename(v))[0]
            thu_muc = os.path.join(ra, ch["ten"], ten_v)
            p = os.path.join(thu_muc, "bang_so.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    bs = json.load(f)
            else:
                bs = phat_lai.chay(v, thu_muc, ho_so=ch["ho_so"], mang=ch["mang"], fps_gui=ch["fps_gui"],
                                     hfov_may_chu=ch["hfov_may_chu"], nhan=nhan, ghi_xem=False,
                                     ghi_de=dict(hfov_ong_kinh=ch["hfov_ong_kinh"], jpeg=ch["jpeg"],
                                                 phoi_sang=ch["phoi_sang"]))
            ket_qua.append(dict(ten=ch["ten"], video=ten_v, bang=bs))
            with open(os.path.join(ra, "bang.md"), "w", encoding="utf-8") as f:
                f.write(bang(ket_qua))
    return ket_qua


def _so(x, dang="{:.0f}"):
    return "—" if x is None else dang.format(x)


def bang(ket_qua: list[dict]) -> str:
    dong = ["| cấu hình | video | sai chiều | câu/phút | trễ p50/p95 ms | bỏ khung % | mất hình đúng/nhầm | |",
            "|---|---|---|---|---|---|---|---|"]
    theo_ten: dict[str, list] = {}
    for k in ket_qua:
        theo_ten.setdefault(k["ten"], []).append(k)
    for ten, ds in theo_ten.items():
        for k in ds:
            b = k["bang"]
            hong = (b.get("sai_chieu") or 0) > 0 or (b.get("tre_ms_p95") or 0) > TRE_HONG_MS
            dong.append(f"| {ten} | {k['video']} | {_so(b.get('sai_chieu'))} | {_so(b.get('cau_phut'), '{:.1f}')} | "
                        f"{_so(b.get('tre_ms_p50'))}/{_so(b.get('tre_ms_p95'))} | {_so(b.get('khung_bo_pct'), '{:.1f}')} | "
                        f"{b.get('mat_hinh_dung', 0)}/{b.get('mat_hinh_nham', 0)} | {'HỎNG' if hong else ''} |")
    return "\n".join(dong) + "\n"


def main():
    ap = argparse.ArgumentParser(description="ESP32 ảo — quét từng yếu tố (chỉ video dev)")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--nhan", default=None)
    a = ap.parse_args()
    print(bang(chay_quet(a.videos, a.ra, a.nhan)))


if __name__ == "__main__":
    main()
