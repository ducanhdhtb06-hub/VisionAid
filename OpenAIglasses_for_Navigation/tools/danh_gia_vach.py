#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đánh giá offline phần dẫn theo gạch dẫn đường: cách cũ (độ nghiêng ở chân ảnh) vs mới (dan_theo_vach.py).

Chạy (đứng ở OpenAIglasses_for_Navigation/):
    ../.venv/bin/python tools/danh_gia_vach.py khiem.mp4 ../my/IMG_7354.MOV --out test_out/vach \
        --nhan test_cham_diem/nhan_vach_dev.json

In cho mỗi video: % khung có vạch, % khung đo tin cậy, câu/phút của cách mới, danh sách câu theo thời gian,
và (nếu có --nhan) kiểm tra theo nhãn mắt:
  - số đo: dấu "lệch" của cách mới so với nhãn trái/phải, |lệch| trên khung nhãn "giữa";
  - lệnh SAI CHIỀU: câu dịch/rẽ của cách mới trong ±1 s quanh khung nhãn trái/phải mà chỉ ngược phía vạch;
    lệnh của cách cũ ở đúng khung nhãn (không lọc câu) để so.
Lưu ảnh mẫu có vẽ (mask, hai mép, điểm tụ, câu) mỗi --mau giây vào --out/<video>/.
Video giữ riêng (IMG_7357, IMG_7359, IMG_7355) chỉ chạy sau khi xong cả 3 giai đoạn.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GOC)
from dan_theo_vach import DanTheoVach  # noqa: E402

HUONG = {"Dịch sang phải": 1, "Rẽ phải một chút": 1, "Dịch sang trái": -1, "Rẽ trái một chút": -1}


def lenh_cu(m):
    """Logic cũ (_get_pixel_domain_features + _generate_navigation_guidance), không làm mượt, không lọc câu."""
    h, w = m.shape
    d = []
    for y in range(h - 1, int(h * 0.3), -5):
        xs = np.flatnonzero(m[y])
        if xs.size > 10:
            d.append([y, (xs[0] + xs[-1]) / 2, xs[-1] - xs[0]])
    if len(d) < 20:
        return ""
    d = np.array(d)
    c = np.polyfit(d[:, 0], d[:, 1], 2, w=d[:, 2])
    goc = np.arctan(2 * c[0] * h + c[1])
    lech = (np.poly1d(c)(h * 0.4) - w / 2) / w
    if goc > np.deg2rad(10):
        return "Rẽ trái"
    if goc < -np.deg2rad(10):
        return "Rẽ phải"
    if abs(lech) > 0.15:
        return "Dịch phải" if lech > 0 else "Dịch trái"
    return ""


HUONG_CU = {"Rẽ trái": -1, "Dịch trái": -1, "Rẽ phải": 1, "Dịch phải": 1}


def mask_vach(model, img):
    h, w = img.shape[:2]
    r = model.predict(img, conf=0.25, verbose=False)[0]
    m = np.zeros((h, w), np.uint8)
    if r.masks is None:
        return None
    for p, c in zip(r.masks.xy, r.boxes.cls.cpu().numpy()):
        if int(c) == 1 and len(p) >= 3:
            cv2.fillPoly(m, [p.astype(np.int32)], 1)
    return m if m.any() else None


def ve(img, m, do, cau):
    o = img.copy()
    H, W = o.shape[:2]
    if m is not None:
        o[m > 0] = (0.55 * o[m > 0] + [0, 110, 110]).astype(np.uint8)
    if do is not None and do.tin_cay:
        y0 = int(np.clip(do.y_vp, 0, H - 1))
        for a in (do.mep_trai, do.mep_phai):
            cv2.line(o, (int(np.polyval(a, H - 1)), H - 1), (int(np.polyval(a, y0)), y0), (0, 0, 255), 2)
        cv2.circle(o, (int(np.clip(do.x_vp, 0, W - 1)), y0), 6, (255, 0, 255), -1)
        txt = f"lech {do.lech:+.2f}W  cheo {do.goc:+.0f}do"
    else:
        txt = "khong chac: " + (do.ly_do if do else "khong thay vach")
    for i, s in enumerate([txt, cau]):
        if s:
            s = s.encode("ascii", "ignore").decode() if i == 0 else s
            cv2.putText(o, s, (8, 24 + 26 * i), 0, 0.6, (0, 0, 0), 4)
            cv2.putText(o, s, (8, 24 + 26 * i), 0, 0.6, (255, 255, 255), 1)
    return o


def chay_video(path, model, fps, out_dir, mau):
    ten = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    buoc = vfps / fps
    dv = DanTheoVach()
    kq = dict(video=ten, khung=0, co_vach=0, tin_cay=0, cau=[], do={}, cu={})
    os.makedirs(os.path.join(out_dir, ten), exist_ok=True)
    i, ke_tiep, lan_luu = 0, 0.0, -1e9
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i + 1e-6 >= ke_tiep:
            ke_tiep += buoc
            _, img = cap.retrieve()
            img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
            t = round(i / vfps, 2)
            m = mask_vach(model, img)
            cau, do = dv.buoc(m, t)
            kq["khung"] += 1
            kq["co_vach"] += m is not None
            kq["tin_cay"] += bool(do is not None and do.tin_cay)
            kq["do"][t] = None if do is None or not do.tin_cay else (do.lech, do.goc)
            kq["cu"][t] = lenh_cu(m) if m is not None else ""
            if cau:
                kq["cau"].append((t, cau))
            if t - lan_luu >= mau or cau:
                cv2.imwrite(os.path.join(out_dir, ten, f"{t:07.2f}.jpg"), ve(img, m, do, cau))
                lan_luu = t
        i += 1
    kq["giay"] = i / vfps
    return kq


def chay_cache(path_npz, ten):
    """Như chay_video nhưng đọc mask đã lưu (np.savez: t, m = packbits 640x360 hoặc rỗng) - không vẽ ảnh."""
    z = np.load(path_npz, allow_pickle=True)
    dv = DanTheoVach()
    kq = dict(video=ten, khung=0, co_vach=0, tin_cay=0, cau=[], do={}, cu={})
    for t, b in zip(z["t"], z["m"]):
        t = round(float(t), 2)
        m = np.unpackbits(b)[:640 * 360].reshape(360, 640) if b.size else None
        cau, do = dv.buoc(m, t)
        kq["khung"] += 1
        kq["co_vach"] += m is not None
        kq["tin_cay"] += bool(do is not None and do.tin_cay)
        kq["do"][t] = None if do is None or not do.tin_cay else (do.lech, do.goc)
        kq["cu"][t] = lenh_cu(m) if m is not None else ""
        if cau:
            kq["cau"].append((t, cau))
    kq["giay"] = float(z["t"][-1]) if len(z["t"]) else 0.0
    return kq


def gan_nhat(bang, t, cua_so):
    ts = [x for x in bang if abs(x - t) <= cua_so]
    return min(ts, key=lambda x: abs(x - t)) if ts else None


def cham(kq, nhan):
    ten = kq["video"].replace("_kinh", "")
    nh = [n for n in nhan if n["video"] == ten]
    dung_dau = sai_dau = khong_do = 0
    giua_lech = []
    sai_moi = sai_cu = lenh_cu_ben = 0
    thua_moi = 0
    for n in nh:
        tk = gan_nhat(kq["do"], n["t"], 0.15)
        do = kq["do"].get(tk) if tk is not None else None
        ben = {"trai": -1, "phai": 1}.get(n["vach"])
        if n["vach"] == "khong_ro":
            continue
        if do is None:
            khong_do += 1
        elif ben:
            dung_dau += np.sign(do[0]) == ben
            sai_dau += np.sign(do[0]) == -ben
        else:
            giua_lech.append(abs(do[0]))
        for t, c in kq["cau"]:
            if abs(t - n["t"]) <= 1.0 and c in HUONG:
                if ben and HUONG[c] == -ben:
                    sai_moi += 1
                if not ben:
                    thua_moi += 1
        c_cu = kq["cu"].get(tk, "") if tk is not None else ""
        if c_cu and ben:
            lenh_cu_ben += 1
            sai_cu += HUONG_CU[c_cu] == -ben
    return dict(nhan=len(nh), dung_dau=int(dung_dau), sai_dau=int(sai_dau), khong_do=khong_do,
                giua_lech_tv=float(np.median(giua_lech)) if giua_lech else None,
                giua_lech_max=float(np.max(giua_lech)) if giua_lech else None,
                sai_chieu_moi=sai_moi, thua_moi=thua_moi, sai_chieu_cu=f"{sai_cu}/{lenh_cu_ben}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--out", default="test_out/vach")
    ap.add_argument("--nhan")
    ap.add_argument("--mau", type=float, default=5.0)
    ap.add_argument("--cache", help="thư mục mask đã lưu <tên video>.npz: chạy lại vài giây, không vẽ ảnh")
    a = ap.parse_args()
    model = None
    if not a.cache:
        from ultralytics import YOLO
        model = YOLO(os.path.join(GOC, "model", "yolo-seg.pt"))
    nhan = json.load(open(a.nhan, encoding="utf-8"))["khung"] if a.nhan else None
    tong = []
    for v in a.videos:
        ten = os.path.splitext(os.path.basename(v))[0]
        if a.cache:
            kq = chay_cache(os.path.join(a.cache, ten + ".npz"), ten)
            os.makedirs(os.path.join(a.out, ten), exist_ok=True)
        else:
            kq = chay_video(v, model, a.fps, a.out, a.mau)
        phut = kq["giay"] / 60
        print(f"\n=== {kq['video']}: {kq['giay']:.0f} s, {kq['khung']} khung | có vạch {100 * kq['co_vach'] / kq['khung']:.0f}% "
              f"| đo tin cậy {100 * kq['tin_cay'] / kq['khung']:.0f}% (trên khung có vạch "
              f"{100 * kq['tin_cay'] / max(1, kq['co_vach']):.0f}%) | câu mới {len(kq['cau'])} = {len(kq['cau']) / phut:.1f}/phút")
        cu_doi = sum(1 for a_, b_ in zip(list(kq["cu"].values()), list(kq["cu"].values())[1:]) if b_ and b_ != a_)
        print(f"    cách cũ (chưa lọc câu): lệnh đổi {cu_doi} lần = {cu_doi / phut:.1f}/phút")
        print("    câu mới:", "; ".join(f"{t:.1f}s {c}" for t, c in kq["cau"]) or "(không)")
        if nhan:
            ch = cham(kq, nhan)
            print(f"    theo nhãn: {ch}")
            tong.append(ch)
        json.dump({k: v for k, v in kq.items() if k in ("video", "giay", "khung", "co_vach", "tin_cay", "cau")},
                  open(os.path.join(a.out, kq["video"], "kq.json"), "w"), ensure_ascii=False, indent=1)
    if tong:
        print("\n=== TỔNG theo nhãn:",
              {k: sum(c[k] for c in tong) for k in ("nhan", "dung_dau", "sai_dau", "khong_do", "sai_chieu_moi", "thua_moi")},
              "| sai chiều cũ:", [c["sai_chieu_cu"] for c in tong])


if __name__ == "__main__":
    main()
