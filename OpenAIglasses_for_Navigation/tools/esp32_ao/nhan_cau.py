#!/usr/bin/env python3
"""Nhận câu từ TIẾNG loa (loa.wav + loa_doan.json của một phiên ESP32 ảo) bằng so dạng sóng với các WAV câu nói.

ĐỘ TIN CẬY THẤP — KHÔNG DÙNG ĐỂ CHẤM ĐIỂM (đo 2026-10-27): bản gốc Việt hóa chỉ nhận được 10/64 đoạn (điểm ~0,6:
tiếng qua luồng phát bị nén/biến đổi); bản mới 3/21 (câu ghép từ nhiều mảnh, không có mẫu nguyên câu).
Thay bằng boc_ghi_cau.py (ghi câu ngay lúc phát, chính xác). Giữ lại làm tham khảo.

Dùng khi máy chủ không ghi nhật ký câu (code gốc Trung Quốc không có AIGLASS_NHAT_KY_CAU): chữ phụ đề lấy từ đúng cái
người đeo kính nghe được. Mẫu được lấy mẫu lại 8 kHz bằng audioop.ratecv như audio_player, nên dạng sóng gần như trùng
khớp -> tương quan chuẩn hóa ~1 với câu đúng, phân biệt được "Rẽ trái"/"Rẽ phải". Một đoạn tiếng có thể chứa nhiều câu
nối nhau: nhận lặp (câu khớp nhất, xóa phần đó, tìm tiếp).
Chạy: ../.venv/bin/python tools/esp32_ao/nhan_cau.py <thư mục phiên> --ban-do voice/map.vi.json [--ra cau.jsonl]
"""
import argparse
import audioop
import json
import os
import wave

import numpy as np

NGUONG = 0.6          # tương quan chuẩn hóa tối thiểu để nhận là một câu


def _doc_8k(duong: str) -> np.ndarray:
    with wave.open(duong) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        b = w.readframes(w.getnframes())
    if ch == 2:
        b = audioop.tomono(b, sw, 1, 0)
    if sw != 2:
        b = audioop.lin2lin(b, sw, 2)
    if sr != 8000:
        b, _ = audioop.ratecv(b, 2, 1, sr, 8000, None)
    return np.frombuffer(b, "<i2").astype(np.float32)


def doc_mau(ban_do: str) -> list[tuple[str, np.ndarray]]:
    with open(ban_do, encoding="utf-8") as f:
        m = json.load(f)
    goc = os.path.dirname(os.path.abspath(ban_do))
    ra = []
    for cau, v in m.items():
        p = os.path.join(goc, v["files"][0])
        if os.path.isfile(p):
            x = _doc_8k(p)
            nz = np.nonzero(np.abs(x) > 300)[0]          # bỏ im lặng đầu/cuối của file mẫu
            if len(nz):
                ra.append((cau, x[nz[0]:nz[-1] + 1]))
    return ra


def _ncc(tin: np.ndarray, mau: np.ndarray) -> tuple[float, int]:
    """Tương quan chuẩn hóa lớn nhất của mẫu trượt trên tín hiệu (FFT) và vị trí."""
    n = len(mau)
    if len(tin) < n:
        tin = np.pad(tin, (0, n - len(tin)))
    L = 1 << int(np.ceil(np.log2(len(tin) + n)))
    c = np.fft.irfft(np.fft.rfft(tin, L) * np.conj(np.fft.rfft(mau, L)), L)[:len(tin) - n + 1]
    cs = np.concatenate(([0.0], np.cumsum(tin.astype(np.float64) ** 2)))   # năng lượng trượt O(N) (convolve O(N·n): chậm)
    nl = np.maximum(cs[n:] - cs[:-n], 0.0)
    r = c / (np.sqrt(nl) * np.linalg.norm(mau) + 1e-9)
    r[nl < n * 100.0 ** 2] = 0.0         # gần im lặng (RMS < 100, gồm đoạn đã xóa): chia ~0 cho điểm ảo rất lớn -> lặp mãi
    i = int(np.argmax(r))
    return float(r[i]), i


def nhan(thu_muc: str, mau: list[tuple[str, np.ndarray]]) -> list[dict]:
    with open(os.path.join(thu_muc, "su_kien.json"), encoding="utf-8") as f:
        t0 = json.load(f)["t0_loa"]
    with open(os.path.join(thu_muc, "loa_doan.json"), encoding="utf-8") as f:
        doan = json.load(f)
    loa = _doc_8k(os.path.join(thu_muc, "loa.wav"))
    ra = []
    for a, b in doan:
        i0, i1 = max(0, int((a - t0 - 0.15) * 8000)), int((b - t0 + 0.15) * 8000)
        con = loa[i0:i1].copy()
        while np.count_nonzero(np.abs(con) > 300) > 0.3 * 8000:
            tot = max(((c,) + _ncc(con, x) for c, x in mau if len(x) <= len(con) + 800), key=lambda z: z[1],
                      default=None)
            if tot is None or tot[1] < NGUONG:
                break
            cau, diem, vt = tot
            ra.append(dict(t=t0 + (i0 + vt) / 8000.0, cau=cau, phat=True, ly_do="nhan_tu_tieng", diem=round(diem, 3)))
            con[vt:vt + len(dict(mau)[cau])] = 0
    ra.sort(key=lambda d: d["t"])
    return ra


def main():
    ap = argparse.ArgumentParser(description="Nhận câu từ tiếng loa của phiên ESP32 ảo")
    ap.add_argument("thu_muc")
    ap.add_argument("--ban-do", required=True, help="map.*.json: câu -> files (WAV)")
    ap.add_argument("--ra", default=None, help="file jsonl ra (mặc định <thu_muc>/cau_tu_tieng.jsonl)")
    a = ap.parse_args()
    ds = nhan(a.thu_muc, doc_mau(a.ban_do))
    p = a.ra or os.path.join(a.thu_muc, "cau_tu_tieng.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in ds)
    print(f"{len(ds)} câu -> {p}")


if __name__ == "__main__":
    main()
