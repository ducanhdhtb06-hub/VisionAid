"""Bảng số của một thư mục phiên ESP32 ảo (khung.jsonl, cau.jsonl, loa_doan.json, su_kien.json).

- tre: câu phat=True -> đầu đoạn tiếng đầu tiên trong loa_doan tại/sau giờ câu (<= 2 s) trừ t_chup của khung gửi
  gần nhất trước câu (camera chụp -> loa bắt đầu nói).
- giờ video của câu = t_video của khung gửi gần nhất trước câu.
- sai_chieu/thua: cùng luật tools/danh_gia_vach.cham (nhãn trong ±1,0 s, câu ngược phía nhãn = sai chiều).
- mat_hinh: câu "Mất hình…" 1,5–3,5 s sau lúc mất kết nối thật = đúng, còn lại = nhầm.
"""
import bisect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from danh_gia_vach import HUONG  # noqa: E402

CAU_MAT_HINH = "Mất hình ảnh từ kính, hãy dừng lại."


def _jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(d) for d in f if d.strip()]


def _json(p, mac_dinh):
    if not os.path.exists(p):
        return mac_dinh
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tinh_bang_so(thu_muc: str, nhan: list | None = None, ten_video: str | None = None) -> dict:
    khung = sorted(_jsonl(os.path.join(thu_muc, "khung.jsonl")), key=lambda k: k["t_chup"])
    cau = [c for c in _jsonl(os.path.join(thu_muc, "cau.jsonl")) if c.get("phat")]
    doan = sorted(_json(os.path.join(thu_muc, "loa_doan.json"), []))
    su_kien = _json(os.path.join(thu_muc, "su_kien.json"), {})
    gui = sorted((k for k in khung if not k.get("bo")), key=lambda k: k["t_gui"])
    t_gui = [k["t_gui"] for k in gui]

    def khung_truoc(t):
        i = bisect.bisect_right(t_gui, t) - 1
        return gui[i] if i >= 0 else None

    giay = (khung[-1]["t_video"] - khung[0]["t_video"]) if len(khung) > 1 else 0.0
    dau_doan = [a for a, _ in doan]
    tre = []
    theo_video = []
    for c in cau:
        k = khung_truoc(c["t"])
        if k is None:
            continue
        theo_video.append((k["t_video"], c["cau"]))
        i = bisect.bisect_left(dau_doan, c["t"])
        if i < len(dau_doan) and dau_doan[i] - c["t"] <= 2.0:
            tre.append((dau_doan[i] - k["t_chup"]) * 1000.0)

    sai = thua = None
    if nhan is not None:
        sai = thua = 0
        for n in nhan:
            if ten_video is not None and n["video"] != ten_video:
                continue
            if n["vach"] == "khong_ro":
                continue
            ben = {"trai": -1, "phai": 1}.get(n["vach"])
            for tv, c in theo_video:
                if abs(tv - n["t"]) <= 1.0 and c in HUONG:
                    if ben and HUONG[c] == -ben:
                        sai += 1
                    if not ben:
                        thua += 1

    mat = [t for t, ten in su_kien.get("anh", []) if ten == "mat_ket_noi"]
    dung = nham = 0
    for c in cau:
        if c["cau"] != CAU_MAT_HINH:
            continue
        if any(1.5 <= c["t"] - t <= 3.5 for t in mat):
            dung += 1
        else:
            nham += 1

    bo = sum(1 for k in khung if k.get("bo"))
    return dict(giay=round(giay, 1), cau_phat=len(cau),
                cau_phut=round(len(cau) / (giay / 60.0), 1) if giay > 0 else None,
                sai_chieu=sai, thua=thua,
                tre_ms_p50=float(np.percentile(tre, 50)) if tre else None,
                tre_ms_p95=float(np.percentile(tre, 95)) if tre else None,
                khung_gui=len(gui), khung_bo_pct=round(100.0 * bo / len(khung), 1) if khung else 0.0,
                mat_hinh_dung=dung, mat_hinh_nham=nham)
