"""Quét câu kính đã phát tìm cặp câu MÂU THUẪN / THỪA (người dùng 28/10: "vừa bảo đi sang trái lại bảo dừng lại").

Loại câu: "vong" (bộ điều khiển vật chắn: "… Đi sang trái." / "… dò đường vòng."), "dung" (có "dừng lại"),
"dich" ("Dịch sang …"), "vat" (câu vật cản kiểu cũ "Xe máy, phía trước"). Cảnh báo nguy hiểm luôn báo (đang tới,
lòng đường, cầu thang, ngang đầu, hố, mất hình) không tính.
Cặp trong CUA_SO giây (câu sau tính từ câu trước):
  vong_roi_dung  bảo đi vòng rồi bảo dừng          dung_roi_di    bảo dừng rồi bảo đi vòng (cùng vật) / dịch
  vong_roi_dich  đang đi vòng lại bị kéo "dịch"     vong_doi_ben   đổi bên đi vòng (hai câu đều nói bên)
  thua           2 câu về cùng một vật cách nhau <= THUA_S giây
  vong_ve_phia_xe  "X, bên trái, đang tới" rồi bảo "Đi sang trái" (về phía xe đang lao tới)
Chạy: ../.venv/bin/python tools/esp32_ao/mau_thuan.py <thư mục phiên> [...]   (đứng ở OpenAIglasses_for_Navigation/)
"""
import json
import os
import re
import sys

CUA_SO = 5.0
THUA_S = 1.5
NGUY = ("đang tới", "lòng đường", "cầu thang", "ngang đầu", "hố", "mất hình", "không nhìn rõ")


def _ben(cau):
    m = re.search(r"(?:Đi|Dịch) sang (trái|phải)", cau)
    return m.group(1) if m else None


def _ten(cau):
    """Tên vật trong câu ("Dừng lại. Xe máy" / "Xe máy, phía trước" / "Xe máy chắn vạch …"), chữ thường."""
    s = cau.split(". ", 1)[1] if cau.startswith(("Dừng lại. ", "Dịch sang")) and ". " in cau else cau
    return re.split(r",| chắn", s, maxsplit=1)[0].strip().lower()


def _cung_vat(c1, c2):
    a, b = _ten(c1), _ten(c2)
    return a == b or "vật cản" in (a, b)


def loai_cau(cau):
    c = cau.lower()
    if any(k in c for k in NGUY) or c.strip() == "dừng lại.":   # "Dừng lại." riêng: ngay sau "X, đang tới"
        return None
    if "đi sang" in c or "dò đường vòng" in c:
        return "vong"
    if "dừng lại" in c:
        return "dung"
    if c.startswith("dịch sang"):
        return "dich"
    if re.match(r"^[^,.]+, (phía trước|bên trái|bên phải)", cau):
        return "vat"
    return None


def _cap(a, b):
    (ca, la), (cb, lb) = a, b
    if la == "vong" and lb == "dung":
        return "vong_roi_dung"
    if la == "dung" and (lb == "dich" or (lb == "vong" and _cung_vat(ca, cb))):   # bảo vòng vật KHÁC: không mâu thuẫn
        return "dung_roi_di"
    if la == "vong" and lb == "dich":
        return "vong_roi_dich"
    if la == "vong" and lb == "vong" and None not in (_ben(ca), _ben(cb)) and _ben(ca) != _ben(cb):
        return "vong_doi_ben"                     # "dò đường vòng" rồi "Đi sang trái": biết thêm bên trống, không đổi bên
    return None


def _ve_phia_xe(cau):
    """Câu "X, bên trái/phải, đang tới" rồi trong CUA_SO giây bảo "Đi sang" đúng bên đó: đi về phía xe đang lao tới."""
    kq = []
    for i, (t1, c1) in enumerate(cau):
        m = re.search(r"bên (trái|phải), đang tới", c1)
        if not m:
            continue
        for t2, c2 in cau[i + 1:]:
            if t2 - t1 > CUA_SO:
                break
            if loai_cau(c2) == "vong" and _ben(c2) == m.group(1):
                kq.append(dict(t1=round(t1, 1), c1=c1, t2=round(t2, 1), c2=c2, loai="vong_ve_phia_xe"))
    return kq


def quet(cau):
    """cau: [(giây, câu đã phát)] theo thời gian -> [{t1, c1, t2, c2, loai}]."""
    xe = _ve_phia_xe(cau)
    ds = [(t, c, loai_cau(c)) for t, c in cau]
    ds = [x for x in ds if x[2]]
    kq = []
    for i, (t1, c1, l1) in enumerate(ds):
        for t2, c2, l2 in ds[i + 1:]:
            if t2 - t1 > CUA_SO:
                break
            loai = _cap((c1, l1), (c2, l2))
            if loai is None and t2 - t1 <= THUA_S and _ten(c1) == _ten(c2) and l1 != "dich" and l2 != "dich":
                loai = "thua"
            if loai:
                kq.append(dict(t1=round(t1, 1), c1=c1, t2=round(t2, 1), c2=c2, loai=loai))
    return xe + kq


def cau_cua_phien(thu_muc):
    """Câu đã phát của một phiên ESP32 ảo, giờ tính theo video."""
    with open(os.path.join(thu_muc, "khung.jsonl"), encoding="utf-8") as f:
        k = json.loads(f.readline())
    t0 = k["t_chup"] - k["t_video"]
    with open(os.path.join(thu_muc, "cau.jsonl"), encoding="utf-8") as f:
        return [(c["t"] - t0, c["cau"]) for c in map(json.loads, f) if c.get("phat")]


if __name__ == "__main__":
    tong = 0
    for tm in sys.argv[1:]:
        kq = quet(cau_cua_phien(tm))
        tong += len(kq)
        print(f"== {tm}: {len(kq)} cặp")
        for k in kq:
            print(f"   {k['loai']:14s} {k['t1']:7.1f} {k['c1']!r} -> {k['t2']:7.1f} {k['c2']!r}")
    print("TỔNG", tong)
