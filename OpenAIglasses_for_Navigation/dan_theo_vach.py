# dan_theo_vach.py
# -*- coding: utf-8 -*-
"""
Dẫn theo gạch dẫn đường (vạch cho người khiếm thị) - đo bằng điểm tụ + bề rộng vạch.
Thiết kế: docs/superpowers/specs/2026-10-25-dan-theo-vach-design.md

Hình học (camera lỗ kim, tiêu cự f, tâm ảnh cx): dải gạch thẳng bề rộng W, tâm cách người d (dương = bên phải),
hướng dải lệch psi so với hướng camera.
  - điểm tụ:  x_vp = cx + f*tan(psi)            -> góc đi chéo, chỉ cần f (từ góc nhìn ngang)
  - (x_tâm - x_vp) / bề_rộng = d / W            -> độ lệch tính bằng "bề rộng vạch",
    không cần chiều cao hay độ chúc camera (camera chúc theta thêm sai số -h*tan(theta)*tan(psi)/W,
    ~0,13W ở theta=10°, psi=12°).
Không phụ thuộc model: đầu vào là mask vạch (lớp blind_path của yolo-seg).
"""
import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

PHAN_GAN = 0.6          # chỉ dùng 60% dưới của đoạn vạch nhìn thấy (tránh khúc rẽ phía xa)
HANG_HEP_PX = 6         # hàng hẹp hơn thế bỏ qua
BUOC_HANG = 2
# Cổng "không chắc" (spec 6.2)
HANG_TOI_THIEU = 15     # số hàng dùng được để khớp mép
NGAN_TI_LE = 0.12       # đoạn vạch nhìn thấy tối thiểu / chiều cao ảnh
SAI_SO_MEP = 0.15       # sai số khớp mép / bề rộng vạch: lớn hơn = mép không thẳng (vạch vỡ, cong)
MEP_GIU_TOI_THIEU = 0.7 # đường thẳng tốt nhất của một mép được < 70% số hàng ủng hộ = mép bị vật che
DUNG_SAI_MEP = 0.08     # hàng "ủng hộ" một đường mép nếu cách nó <= 8% bề rộng vạch (tối thiểu 2 px)
NHANH_LAN = 2.0         # bề rộng thật > 2 lần bề rộng suy từ hai mép = có nhánh rẽ / ngã ba
LECH_CHAN_TROI_DO = 6.0 # điểm tụ lệch khỏi chân trời > 6° (đầu cúi/ngẩng tức thời hiếm hơn thế)
LY_DO_CHAN_TROI = "điểm tụ xa chân trời"


@dataclass
class DoVach:
    lech: float           # độ lệch tâm vạch so với người, đơn vị "bề rộng vạch" (dương = vạch bên phải)
    goc: float            # góc đi chéo (độ, dương = vạch chạy chếch sang phải)
    x_vp: float
    y_vp: float
    tin_cay: bool
    ly_do: str            # lý do không chắc ("" nếu tin cậy)
    mep_trai: tuple       # hệ số x = a*y + b
    mep_phai: tuple
    y_dinh: float = float("nan")   # hàng trên cùng (đầu xa) của dải gạch nhìn thấy
    dinh_cham_mep: bool = False    # đầu xa chạm mép trái/phải ảnh (vạch đi ra khỏi khung, không phải hết)
    sap_het: bool = False          # do DanTheoVach gán: đầu xa của vạch dừng trong Z_HET mét
    re: str = ""                   # do DanTheoVach gán: "L+1"/"L-1" vạch rẽ phải/trái, "T" ngã rẽ (đang tiến lại gần)


def _khop_mep(y, x, dung_sai):
    """Khớp x = a*y + b theo đồng thuận: chọn đường qua 2 hàng được nhiều hàng nằm trong dung_sai nhất
    (thử các cặp trong 12 hàng rải đều, tất định), rồi bình phương tối thiểu trên các hàng đó.
    Trả (hệ số, sai số trung vị, tỉ lệ hàng ủng hộ). Mép bị vật che -> tỉ lệ ủng hộ thấp."""
    chon = np.unique(np.linspace(0, len(y) - 1, 12).astype(int))
    tot, a_tot = -1, None
    for i in chon:
        for j in chon:
            if j <= i or y[i] == y[j]:
                continue
            a = np.array([(x[j] - x[i]) / (y[j] - y[i]), 0.0])
            a[1] = x[i] - a[0] * y[i]
            n = int((np.abs(np.polyval(a, y) - x) <= dung_sai).sum())
            if n > tot:
                tot, a_tot = n, a
    giu = np.abs(np.polyval(a_tot, y) - x) <= dung_sai
    a = np.polyfit(y[giu], x[giu], 1)
    giu = np.abs(np.polyval(a, y) - x) <= dung_sai
    sai = float(np.median(np.abs(np.polyval(a, y[giu]) - x[giu]))) if giu.any() else float("inf")
    return a, sai, float(giu.mean())


def _chon_dai_gach(mask):
    """Dải gạch người đang theo = mảnh GẦN CHÂN nhất (đáy thấp nhất) trong các mảnh đủ lớn (>= 20% mảnh to nhất).
    Không lấy mảnh to nhất: một đốm nhận nhầm to ở xa (biển hiệu, tường) sẽ che mất vạch thật dưới chân."""
    m = np.asarray(mask) > 0
    if m.ndim == 3:
        m = m.any(axis=2)             # mask nhiều kênh: vạch ở kênh nào cũng nhận
    m = m.astype(np.uint8)
    if not m.any():
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    dt = st[1:, cv2.CC_STAT_AREA]
    day = st[1:, cv2.CC_STAT_TOP] + st[1:, cv2.CC_STAT_HEIGHT]
    ung = np.flatnonzero(dt >= 0.2 * dt.max())
    k = 1 + int(max(ung, key=lambda i: (day[i], dt[i])))
    return lab == k


def _khong_chac(ly_do):
    nan = float("nan")
    return DoVach(nan, nan, nan, nan, False, ly_do, (nan, nan), (nan, nan))


def do_vach(mask, hfov_deg=70.0, y_chan_troi=None):
    """Đo độ lệch và góc đi chéo từ mask vạch. None nếu không có vạch.
    y_chan_troi: hàng chân trời ước lượng (BoChanTroi); None -> chưa biết, lần đo chưa được tin (chỉ để học).
    Lần đo không qua cổng có tin_cay=False và ly_do (các cổng: spec mục 6.2)."""
    if mask is None:
        return None
    m = _chon_dai_gach(mask)
    if m is None:
        return None
    H, W = m.shape
    hang = np.where(m.any(axis=1))[0]
    y_bot, y_top = int(hang.max()), int(hang.min())
    if y_bot - y_top < NGAN_TI_LE * H:
        return _khong_chac("đoạn vạch ngắn")
    y_lim = y_bot - PHAN_GAN * (y_bot - y_top)
    Y, L, R, RONG_THAT = [], [], [], []
    for y in range(y_bot, int(y_lim), -BUOC_HANG):
        xs = np.flatnonzero(m[y])
        if xs.size < HANG_HEP_PX:
            continue
        RONG_THAT.append((y, xs[-1] - xs[0]))
        if xs[0] <= 1 or xs[-1] >= W - 2:      # chạm mép ảnh: không biết mép thật
            continue
        Y.append(y)
        L.append(xs[0])
        R.append(xs[-1])
    if len(Y) < HANG_TOI_THIEU:
        if len(RONG_THAT) >= HANG_TOI_THIEU:
            return _khong_chac("vạch rộng hơn khung ảnh")   # vẫn THẤY vạch (vd. tấm gạch cảnh báo), không phải mất
        return _khong_chac("đoạn vạch ngắn")
    Y, L, R = (np.asarray(v, float) for v in (Y, L, R))
    dung_sai = max(2.0, DUNG_SAI_MEP * float(np.median(R - L)))
    aL, eL, gL = _khop_mep(Y, L, dung_sai)
    aR, eR, gR = _khop_mep(Y, R, dung_sai)
    # đi lên trên ảnh (y giảm) thì vạch phải hẹp lại: aR[0] - aL[0] > 0
    if aR[0] - aL[0] < 1e-3:
        return _khong_chac("hai mép không hội tụ, không có điểm tụ")
    y_vp = (aR[1] - aL[1]) / (aL[0] - aR[0])
    x_vp = np.polyval(aL, y_vp)
    yb = Y.max()
    xl, xr = np.polyval(aL, yb), np.polyval(aR, yb)
    rong = xr - xl
    if rong <= 0 or y_vp >= Y.min():
        return _khong_chac("điểm tụ không nằm phía trên vạch")
    f = (W / 2) / np.tan(np.radians(hfov_deg / 2))
    # camera chúc theta (suy từ chân trời): x_vp - cx = f*tan(psi)/cos(theta)
    cos_chuc = np.cos(np.arctan((H / 2 - y_chan_troi) / f)) if y_chan_troi is not None else 1.0
    goc = float(np.degrees(np.arctan((x_vp - W / 2) * cos_chuc / f)))
    lech = float(((xl + xr) / 2 - x_vp) / rong)
    ly_do = []
    if (eL + eR) / 2 / rong > SAI_SO_MEP:
        ly_do.append("mép không thẳng")
    if min(gL, gR) < MEP_GIU_TOI_THIEU:
        ly_do.append("mép bị che")
    nhanh = sum(1 for y, r in RONG_THAT if r > NHANH_LAN * (np.polyval(aR, y) - np.polyval(aL, y)))
    if nhanh >= 3:
        ly_do.append("có nhánh rẽ")
    if y_chan_troi is None:
        # chưa biết chân trời: chưa tin, nhưng lần đo vẫn được dùng để học chân trời (camera gắn ngang/ngẩng
        # thì chân trời nằm dưới giữa ảnh - không được tự khóa)
        ly_do.append(LY_DO_CHAN_TROI)
    elif abs(np.degrees(np.arctan((H / 2 - y_vp) / f) - np.arctan((H / 2 - y_chan_troi) / f))) > LECH_CHAN_TROI_DO:
        ly_do.append(LY_DO_CHAN_TROI)
    xs_dinh = np.flatnonzero(m[y_top:y_top + 4].any(axis=0))
    cham = bool(xs_dinh.size and (xs_dinh[0] <= 1 or xs_dinh[-1] >= W - 2))
    return DoVach(lech, goc, float(x_vp), float(y_vp), not ly_do, "; ".join(ly_do), tuple(aL), tuple(aR),
                  float(y_top), cham)


class BoChanTroi:
    """Hàng chân trời = trung vị y điểm tụ (lần đo qua các cổng khác) trong cua_so_s giây gần nhất."""

    def __init__(self, cua_so_s=5.0, toi_thieu=5, phu_s=1.0):
        # >= 5 mẫu trải >= 1 s (không phải số mẫu cố định lớn): ESP32 3 khung/giây, nửa khung bị che vẫn học được
        self.cua_so_s, self.toi_thieu, self.phu_s = cua_so_s, toi_thieu, phu_s
        self.mau = deque()

    def cap_nhat(self, y_vp, t):
        self.mau.append((t, y_vp))
        while self.mau and self.mau[0][0] < t - self.cua_so_s:
            self.mau.popleft()

    def gia_tri(self, t=None):
        while t is not None and self.mau and self.mau[0][0] < t - self.cua_so_s:
            self.mau.popleft()            # không có mẫu mới thì chân trời cũ cũng hết hạn
        if len(self.mau) < self.toi_thieu or self.mau[-1][0] - self.mau[0][0] < self.phu_s:
            return None
        return float(np.median([y for _, y in self.mau]))


# ---------- Vạch bị chắn (spec 7.1) ----------
CHAN_TI_LE = 0.3     # dải chân vật phủ >= 30% bề rộng vạch (~12 cm của 40 cm) = gậy/chân sẽ vướng
CHAN_XA_M = 5.0      # chỉ xét vật trong 5 m (khoảng cách do module né vật cản độ sâu đo)
T_GIU_DO = 1.5       # dùng lần đo tin cậy gần nhất trong 1,5 s (xe đè lên mép làm lần đo hiện tại không chắc)
T_SAU_CHAN = 4.0     # còn im thêm 4 s sau khi hết thấy vật trên vạch (người đang vòng qua bằng gậy)


def hanh_lang(do, H, W):
    """Mask bool (H x W) của dải gạch theo hai mép đã khớp, từ chân ảnh tới điểm tụ."""
    ys = np.arange(H, dtype=float)
    xl, xr = np.polyval(do.mep_trai, ys), np.polyval(do.mep_phai, ys)
    xx = np.arange(W)[None, :]
    m = (xx >= xl[:, None]) & (xx <= xr[:, None])
    m[:int(np.clip(np.ceil(do.y_vp), 0, H))] = False
    return m


def vat_tren_vach(vat, do, H, W, xa_m=CHAN_XA_M):
    """Vật (dict có 'near' bool, 'dist' m - từ NeVatCanDoSau.cand) đứng trên dải gạch trong xa_m mét:
    dải chân (10% hàng thấp nhất của vùng near) phủ >= CHAN_TI_LE bề rộng vạch ở cùng hàng."""
    ra = []
    for o in vat or []:
        if o.get("dist", 1e9) > xa_m:
            continue
        near = np.asarray(o["near"]) > 0
        if near.shape != (H, W):
            near = cv2.resize(near.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        hang = np.flatnonzero(near.any(axis=1))
        if not hang.size:
            continue
        y_bot, y_top = hang.max(), hang.min()
        tot = 0.0
        for y in hang[hang >= y_bot - max(1.0, 0.1 * (y_bot - y_top))]:
            xl, xr = np.polyval(do.mep_trai, y), np.polyval(do.mep_phai, y)
            if y <= do.y_vp or xr - xl < 2:
                continue
            xs = np.flatnonzero(near[y])
            tot = max(tot, ((xs >= xl) & (xs <= xr)).sum() / (xr - xl))
        if tot >= CHAN_TI_LE:
            ra.append(o)
    return ra


# ---------- Vật chắn hay cạnh vạch, theo 3D (spec 2026-10-27-vong-vat-can-tren-vach, mục 4.1) ----------
CAO_VAT = (0.15, 2.0)  # phần vật ở độ cao này chạm người (gậy quét sát đất không dò được tay lái, gương xe)
DIEM_CHAN = 15       # >= số điểm vật nằm trong lối đi -> chắn


def _vat_tren_mat_dat(o, do, geo, hw_vach, xa_m):
    """Điểm vật (cao CAO_VAT, Z <= xa_m) và hai mép vạch trên MẶT ĐẤT tại khoảng cách của từng điểm:
    (xs, x_mép_trái, x_mép_phải), hoặc None nếu thiếu lần đo / hình học / không khớp được mép (vật che mép)."""
    if do is None or geo is None:
        return None
    X, Z, h = geo["X"], geo["Z"], geo["height"]
    Hg, Wg = X.shape
    Hv, Wv = hw_vach
    sx, sy = Wg / Wv, Hg / Hv
    y0 = max(float(do.y_vp), 0.0)
    mep = {}
    for ten, he_so in (("trai", do.mep_trai), ("phai", do.mep_phai)):
        diem = []
        for yv in np.linspace(y0 + 0.1 * (Hv - y0), Hv - 1, 12):
            xg, yg = int(round(np.polyval(he_so, yv) * sx)), int(round(yv * sy))
            if 0 <= xg < Wg and 0 <= yg < Hg and np.isfinite(Z[yg, xg]) and 0 < Z[yg, xg] < 30 \
                    and abs(h[yg, xg]) < 0.1:            # điểm mép phải nằm trên mặt đất (không bị vật che)
                diem.append((Z[yg, xg], X[yg, xg]))
        if len(diem) < 4:
            return None
        z, x = np.array(diem).T
        mep[ten] = np.polyfit(z, x, 1)                    # X_mép(Z): mép vạch là đường thẳng trên mặt đất
    m = np.asarray(o["mask"]) > 0
    if m.shape != (Hg, Wg):
        m = cv2.resize(m.astype(np.uint8), (Wg, Hg), interpolation=cv2.INTER_NEAREST) > 0
    m &= np.isfinite(Z) & np.isfinite(X) & (h > CAO_VAT[0]) & (h < CAO_VAT[1]) & (Z <= xa_m)
    zs, xs = Z[m], X[m]
    return xs, np.polyval(mep["trai"], zs), np.polyval(mep["phai"], zs)


def phan_loai_vat(o, do, geo, hw_vach, xa_m=CHAN_XA_M):
    """Vật o (dict 'mask' cùng cỡ geo) so với dải vạch giữa hai mép của `do`, tính trên MẶT ĐẤT (X, Z của geo - hình
    học module vật cản), không so điểm ảnh ở các hàng khác nhau.
    "chan": có >= DIEM_CHAN điểm vật cao CAO_VAT, Z <= xa_m đè lên dải vạch (kể cả phần nhô ra như tay lái);
    "khong_chac": thiếu lần đo vạch / hình học / không khớp được mép (vật che mép vạch) -> người gọi phải BÁO, không
    được im; "canh": dựng được dải vạch và vật nằm ngoài (kể cả xe đỗ sát cạnh vạch: người dùng 28/10 chỉ quan tâm vật
    trong vùng vạch; dev đo xe đỗ cạnh vạch lấn 1–20 cm ra ngoài mép, xe đè vạch thật lấn 30–90 cm)."""
    d = _vat_tren_mat_dat(o, do, geo, hw_vach, xa_m)
    if d is None:
        return "khong_chac"
    xs, xt, xp = d
    return "chan" if ((xs >= xt) & (xs <= xp)).sum() >= DIEM_CHAN else "canh"


LECH_NUA = 0.8       # >= 80% phần vật đè vạch nằm ở một nửa -> vật ở nửa đó; còn lại là "giữa"


def vi_tri_tren_vach(o, do, geo, hw_vach, xa_m=CHAN_XA_M):
    """Phần vật đè lên vạch nằm ở nửa nào: "trai" | "phai" | "giua", None nếu không đè lên vạch / không đo được.
    Để câu nói rõ "Xe máy chắn vạch bên phải. Đi sang trái." (người dùng 28/10: "vòng bên trái" nghe thành xe ở trái)."""
    d = _vat_tren_mat_dat(o, do, geo, hw_vach, xa_m)
    if d is None:
        return None
    xs, xt, xp = d
    trong = (xs >= xt) & (xs <= xp)
    if trong.sum() < DIEM_CHAN:
        return None
    phai = float(np.mean(xs[trong] > (xt[trong] + xp[trong]) / 2))
    return "phai" if phai >= LECH_NUA else "trai" if phai <= 1 - LECH_NUA else "giua"


# ---------- Hết vạch (spec 7.2) ----------
Z_HET = 5.0          # báo khi đầu xa của vạch dừng trong 5 m (đi 0,7 m/s: còn ~3–7 s)
H_CAM = float(os.getenv("AIGLASS_NE_HCAM", "1.5"))   # chiều cao mắt/camera (m), chỉ dùng để quy ra 5 m
T_HET = 1.0          # giữ ổn định 1 s
T_SAU_HET = 10.0     # sau khi đã báo hết vạch, vạch biến mất trong 10 s là đúng dự kiến: không báo "mất"
TIEN_GAN_M = 0.3     # đầu vạch thật đứng yên trên mặt đất: người bước tới thì nó gần lại >= 0,3 m trong ~1 s.
                     # Ranh giới do vạch mờ / ảnh kém ở xa thì đi theo người (khoảng cách gần như không đổi).
TIEP_TI_LE = 0.2     # trên đường kéo dài phía sau đầu vạch còn >= 20% là mask vạch (vd. rãnh nước cắt ngang) -> chưa hết


def khoang_cach_dau_vach(do, y_chan_troi, W, hfov_deg):
    """Khoảng cách (m) tới đầu xa của vạch: h / tan(góc dưới chân trời). None nếu không tính được."""
    if do is None or y_chan_troi is None or not do.y_dinh > y_chan_troi:
        return None
    f = (W / 2) / np.tan(np.radians(hfov_deg / 2))
    return float(H_CAM * f / (do.y_dinh - y_chan_troi))


def dau_vach_gan(do, y_chan_troi, H, W, hfov_deg, vat=None, mask=None):
    """Đầu xa của dải gạch nằm trong Z_HET m, trong ảnh (không chạm mép), không có vật đè lên chỗ đó,
    và phía sau nó (theo đường kéo dài) không còn mảnh vạch nào."""
    if do is None or not do.tin_cay or y_chan_troi is None or do.dinh_cham_mep:
        return False
    z = khoang_cach_dau_vach(do, y_chan_troi, W, hfov_deg)
    if z is None or z > Z_HET:
        return False
    if mask is not None:
        y_xa0, y_xa1 = int(np.ceil(max(do.y_vp, y_chan_troi))) + 3, int(do.y_dinh) - 6
        if y_xa1 > y_xa0:
            hl_xa = hanh_lang(do, H, W)[y_xa0:y_xa1]
            tiep = ((np.asarray(mask)[y_xa0:y_xa1] > 0) & hl_xa).sum()
            if hl_xa.sum() and tiep >= TIEP_TI_LE * hl_xa.sum():
                return False                              # vạch còn tiếp phía sau (bị rãnh/vết nứt cắt ngang)
    y0, y1 = int(max(0, do.y_dinh - 20)), int(min(H, do.y_dinh + 10))
    hl = hanh_lang(do, H, W)[y0:y1]
    for o in vat or []:
        near = np.asarray(o["near"]) > 0
        if near.shape != (H, W):
            near = cv2.resize(near.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if (near[y0:y1] & hl).any():
            return False                                  # vật che đầu vạch: bị chắn, không phải hết
    return True


# ---------- Vạch rẽ chữ L / ngã rẽ (spec 7.3) ----------
Z_RE = 4.0           # báo chỗ rẽ trong 4 m
NHANH_RONG = 1.5     # một hàng có mask lấn ra ngoài hành lang thẳng >= 1,5 bề rộng vạch ở một bên = nhánh ngang
NHANH_HANG = 3       # nhánh phải dày >= 3 hàng liên tiếp (nhánh 40 cm ở 6 m vẫn ~6 hàng ở ảnh 360 dòng)
T_RE = 1.0
T_QUEN_RE = 3.0      # 3 s không thấy chỗ rẽ nữa thì chỗ rẽ sau được báo lại


HANG_GAN = 25        # đoạn vạch gần chân dùng làm hành lang thẳng khi tìm chỗ rẽ (25 hàng ~ 0,3–0,5 m mặt đất)


def hanh_lang_gan(comp, H, W):
    """Hai mép của đoạn vạch GẦN CHÂN nhất (HANG_GAN hàng dưới cùng không chạm mép ảnh), khớp đồng thuận.
    Không cần phép đo chung tin cậy: chỗ rẽ thật hay lộn xộn ở phía xa (mép bị che, điểm tụ nhảy) nhưng đoạn
    dưới chân vẫn thẳng. Trả DoVach chỉ có mep_trai/mep_phai/x_vp/y_vp, hoặc None."""
    Y, L, R = [], [], []
    for y in np.flatnonzero(comp.any(axis=1))[::-1]:
        xs = np.flatnonzero(comp[y])
        if xs.size < HANG_HEP_PX or xs[0] <= 1 or xs[-1] >= W - 2:
            continue
        Y.append(y)
        L.append(xs[0])
        R.append(xs[-1])
        if len(Y) >= HANG_GAN:
            break
    if len(Y) < 12:
        return None
    Y, L, R = (np.asarray(v, float) for v in (Y, L, R))
    dung_sai = max(2.0, DUNG_SAI_MEP * float(np.median(R - L)))
    aL, _, _ = _khop_mep(Y, L, dung_sai)
    aR, _, _ = _khop_mep(Y, R, dung_sai)
    if aR[0] - aL[0] < 1e-3:
        return None
    y_vp = (aR[1] - aL[1]) / (aL[0] - aR[0])
    if y_vp >= Y.min():
        return None
    nan = float("nan")
    return DoVach(nan, nan, float(np.polyval(aL, y_vp)), float(y_vp), True, "", tuple(aL), tuple(aR))


def tim_re(comp, do, y_chan_troi, H, W, hfov_deg):
    """comp: mask bool của mảnh dải gạch dưới chân (gồm cả nhánh); do: lần đo tin cậy (hai mép đã khớp).
    Trả dict(kieu="L"|"T", ben=+1 phải/-1 trái, z=m tới chỗ rẽ) cho nhánh ngang GẦN nhất, hoặc None."""
    if do is None or y_chan_troi is None or comp is None:
        return None
    f = (W / 2) / np.tan(np.radians(hfov_deg / 2))
    hang = np.flatnonzero(comp.any(axis=1))
    if not hang.size:
        return None
    y_lo = int(max(np.ceil(max(do.y_vp, y_chan_troi)) + 2, hang.min()))
    chay = {1: 0, -1: 0}
    for y in range(H - 1, y_lo - 1, -1):
        xl, xr = np.polyval(do.mep_trai, y), np.polyval(do.mep_phai, y)
        w = xr - xl
        xs = np.flatnonzero(comp[y])
        if w < 3 or not xs.size:
            chay = {1: 0, -1: 0}
            continue
        for ben, lan in ((1, xs[-1] - xr), (-1, xl - xs[0])):
            chay[ben] = chay[ben] + 1 if lan >= NHANH_RONG * w else 0
        co = [b for b in (1, -1) if chay[b] >= NHANH_HANG]
        if co:
            y_b = y + NHANH_HANG // 2
            if y_b <= y_chan_troi:
                return None
            z = float(H_CAM * f / (y_b - y_chan_troi))
            # vạch chính còn đi thẳng tiếp sau chỗ rẽ? (>= 20% hành lang phía xa là mask vạch)
            y_xa1 = y - 2
            y_xa0 = int(np.ceil(max(do.y_vp, y_chan_troi))) + 3
            tiep = False
            if y_xa1 > y_xa0:
                hl = hanh_lang(do, H, W)[y_xa0:y_xa1]
                tiep = hl.sum() > 0 and (comp[y_xa0:y_xa1] & hl).sum() >= TIEP_TI_LE * hl.sum()
            kieu = "T" if (tiep or len(co) == 2) else "L"
            return dict(kieu=kieu, ben=co[0] if len(co) == 1 else 0, z=z)
    return None


# ---------- Luật nói (spec 6.3) ----------
LECH_BAT, LECH_NHA = 0.5, 0.3     # bề rộng vạch: |lech| > 0,5 = tâm người đã ra tới mép vạch (không phụ thuộc vạch rộng/hẹp); nhả 0,3 = đã vào trong, cách mép >= 0,2
GOC_BAT, GOC_NHA = 15.0, 8.0      # độ: đi chéo 15° qua 2 m lệch ~0,5 m (~1,3 bề rộng vạch)
GOC_VE = 5.0                      # đang lệch mà hướng về phía vạch >= 5° -> người đang tự sửa, im
T_LECH, T_GOC = 1.0, 1.5          # giây phải giữ ổn định (góc lâu hơn: quay đầu nhìn cũng đổi góc)
TI_LE = 0.7                       # >= 70% khung tin cậy trong cửa sổ thỏa điều kiện
NHAC_LAI, NHAC_MAX = 4.0, 16.0    # cùng một đợt lệch: nhắc lại sau 4 s, rồi 8 s, rồi tối đa mỗi 16 s
T_MAT, T_TIM_LAI = 3.0, 1.0
KHUNG_HO = 1.0                    # hai khung cách nhau > 1 s = rơi khung / nhảy cóc: không nối cửa sổ qua chỗ hở
                                  # (ESP32 3–8 khung/giây, khoảng cách khung có thể rung 0,2–0,6 s)
PHU_TOI_THIEU = 0.7               # khung tin cậy phải trải >= 70% cửa sổ (vài khung chớp nhoáng không đủ)
# "Rẽ" khi quay đầu: camera trên đầu thấy quay đầu y hệt quay thân. Thân đi chéo psi thì vị trí so với vạch TRÔI
# v·sin(psi) (0,7 m/s, 15° -> 18 cm/s); chỉ quay đầu thì không trôi. AIGLASS_RE: "cu" = không xét trôi,
# "troi" = chỉ nói Rẽ khi độ lệch trôi đúng chiều góc chéo (độ dốc / sai số chuẩn >= 2, không cần mét), "tat" = không Rẽ.
TROI_T = 2.0                      # độ dốc tính trên mọi khung tin cậy trong 2 s gần nhất

CAU = {
    ("lech", 1): "Dịch sang phải", ("lech", -1): "Dịch sang trái",
    ("goc", 1): "Rẽ phải một chút", ("goc", -1): "Rẽ trái một chút",
    "mat": "Mất đường đi, đang tìm lại.", "tim_lai": "Đã quay lại đường dẫn.",
    "het": "Sắp hết vạch dẫn đường.",
    "L+1": "Vạch rẽ phải phía trước.", "L-1": "Vạch rẽ trái phía trước.", "T": "Phía trước có ngã rẽ vạch.",
    "thay": "Đã thấy vạch dẫn đường.", "chua_thay": "Chưa thấy vạch dẫn đường.",
}
# Người khiếm thị không phân biệt được "im vì đi đúng" với "im vì kính chưa thấy gì": báo một lần khi bắt được vạch
# lần đầu, và nhắc (4 -> 8 -> 16 s) nếu từ lúc bắt đầu đã T_CHUA_THAY giây mà chưa từng theo được vạch.
T_CHUA_THAY = 5.0
# Thử iPhone 2026-10-26 (trong nhà, không có vạch): nhắc 4 -> 8 s xen câu vật cản nghe dồn dập -> nhắc 15 s rồi 30 s
NHAC_CHUA_THAY, NHAC_CHUA_THAY_MAX = 15.0, 30.0


class LuatNoi:
    """Biến chuỗi lần đo theo thời gian thành câu nói thưa: đi đúng thì im, lệch/chéo ổn định mới nói."""

    def __init__(self, che_do_re=None, thong_bao=None):
        self.che_do_re = os.getenv("AIGLASS_RE", "troi") if che_do_re is None else che_do_re
        # AIGLASS_THONG_BAO_VACH=0: tắt "Đã thấy / Chưa thấy vạch" (người dùng đã quen kính)
        self.thong_bao = os.getenv("AIGLASS_THONG_BAO_VACH", "1") != "0" if thong_bao is None else thong_bao
        self.ls = deque()                 # (t, DoVach|None) trong 2 s gần nhất
        self.lech_dang = 0                # lệnh đang bật: +1 phải, -1 trái, 0 không
        self.goc_dang = 0
        self.da_noi = {}                  # câu -> lần nói gần nhất
        self.so_lan = {}                  # câu -> số lần đã nói trong đợt lệch/chéo hiện tại (giãn nhắc lại)
        self.t_tc = -1e9                  # lần cuối có số đo tin cậy
        self.dang_theo = False            # đã từng theo vạch ổn định
        self.da_bao_mat = False
        self.t_thay_vach = None
        self.t_chan = -1e9                # lần cuối thấy vật trên vạch
        self.da_bao_het, self.t_het = False, -1e9
        self.da_bao_re, self.t_re = False, -1e9
        self.da_tung_theo = False         # đã từng theo vạch ổn định trong phiên này
        self.can_bao_thay = False         # vừa bắt được vạch lần đầu: chờ báo "Đã thấy vạch"
        self.t_bat_dau = None

    def _on_dinh(self, t, T, dk, dung=lambda d: d.tin_cay):
        """Trong [t-T, t]: đã quan sát liên tục đủ T giây (có khung ở mốc t-T trở về trước, không hở > KHUNG_HO),
        >= 2 khung tin cậy trải >= 70% cửa sổ, >= 70% khung tin cậy thỏa dk."""
        moc = [i for i, (ti, _) in enumerate(self.ls) if ti <= t - T + 1e-6]
        if not moc:
            return False                  # mới bắt đầu: chưa đủ quan sát
        ts = [ti for ti, _ in list(self.ls)[moc[-1]:]]
        if any(b - a > KHUNG_HO for a, b in zip(ts, ts[1:])):
            return False                  # thời gian nhảy cóc (rơi khung): dữ liệu cũ không còn đúng
        tin = [(ti, d) for ti, d in self.ls if ti >= t - T - 1e-6 and d is not None and dung(d)]
        if len(tin) < 2 or tin[0][0] > t - PHU_TOI_THIEU * T + 1e-6:
            return False                  # số đo tin cậy chỉ chớp nhoáng (ví dụ giữa lúc quay đầu, ảnh nhòe)
        return sum(1 for _, d in tin if dk(d)) >= TI_LE * len(tin)

    def _dang_troi(self, t, dau):
        """Độ lệch đang trôi theo chiều của góc chéo dấu `dau` (góc dương = vạch chạy chếch phải = người đi lệch
        sang trái so với vạch -> lệch tăng): hồi quy lệch theo thời gian, độ dốc cùng dấu và >= 2 sai số chuẩn."""
        diem = [(ti, d.lech) for ti, d in self.ls if ti >= t - TROI_T - 1e-6 and d is not None and d.tin_cay]
        if len(diem) < 4:
            return False
        x, y = np.array(diem).T
        x = x - x.mean()
        sxx = float((x * x).sum())
        if sxx <= 0:
            return False
        doc = float((x * (y - y.mean())).sum() / sxx)
        if doc * dau <= 0:
            return False
        du = y - y.mean() - doc * x
        se = float(np.sqrt((du * du).sum() / max(1, len(diem) - 2) / sxx))
        return se == 0 or abs(doc) / se >= 2.0

    def _noi(self, cau, t, nhac=NHAC_LAI, nhac_max=NHAC_MAX):
        n = self.so_lan.get(cau, 0)
        cho = min(nhac * 2 ** max(0, n - 1), nhac_max)
        if t - self.da_noi.get(cau, -1e9) < cho:
            return ""
        self.da_noi[cau] = t
        self.so_lan[cau] = n + 1
        return cau

    def huy(self, cau):
        """Câu vừa trả về KHÔNG được phát (process_frame chặn vì đang đọc câu khác): coi như chưa nói, thử lại."""
        self.da_noi.pop(cau, None)
        self.so_lan[cau] = max(0, self.so_lan.get(cau, 0) - 1)
        if cau == CAU["mat"]:
            self.da_bao_mat = False
        elif cau == CAU["het"]:
            self.da_bao_het, self.t_het = False, -1e9
        elif cau in (CAU["L+1"], CAU["L-1"], CAU["T"]):
            self.da_bao_re = False
        elif cau == CAU["tim_lai"]:
            self.da_bao_mat = True
        elif cau == CAU["thay"]:
            self.can_bao_thay = True

    def _nha(self, loai):
        """Kết thúc đợt lệch/chéo: lần sau bắt đầu lại nhịp nhắc 4 s."""
        if loai == "lech":
            dang, self.lech_dang = self.lech_dang, 0
        else:
            dang, self.goc_dang = self.goc_dang, 0
        if dang:
            self.so_lan.pop(CAU[(loai, dang)], None)

    def buoc(self, do, t, co_vat_chan=False):
        if self.t_bat_dau is None:
            self.t_bat_dau = t
        cau = self._buoc_chinh(do, t, co_vat_chan)
        if not self.thong_bao:
            return cau
        if self.can_bao_thay:
            self.can_bao_thay = False
            if not cau:                   # câu chỉ hướng / sự kiện cùng lúc đã đủ cho biết kính thấy vạch
                return self._noi(CAU["thay"], t)
        if not cau and not self.da_tung_theo and t - self.t_bat_dau >= T_CHUA_THAY:
            return self._noi(CAU["chua_thay"], t, NHAC_CHUA_THAY, NHAC_CHUA_THAY_MAX)
        return cau

    def _buoc_chinh(self, do, t, co_vat_chan):
        self.ls.append((t, do))
        while self.ls and self.ls[0][0] < t - 2.0:
            self.ls.popleft()
        if co_vat_chan:
            self.t_chan = t
        if t - self.t_chan < T_SAU_CHAN:
            # có vật trên vạch: người đang vòng qua bằng gậy -> không kéo về vạch, không báo "mất vạch"
            self.t_thay_vach = t
            self._nha("lech")
            self._nha("goc")
            return ""
        tin_cay = do is not None and do.tin_cay
        if t - self.t_tc > KHUNG_HO:
            # quãng dài không có số đo tin cậy: lệnh đang bật không còn căn cứ, phải xét ổn định lại từ đầu
            self._nha("lech")
            self._nha("goc")
        if tin_cay:
            self.t_tc = t
        if do is not None and "ngắn" not in do.ly_do:
            self.t_thay_vach = t          # vạch còn đó (kể cả khi đo không chắc vì bị che / ngã ba)

        # mất vạch / tìm lại: "mất" = không THẤY vạch >= 3 s, không phải "đo không chắc"
        if self.da_bao_mat:
            if tin_cay and self._on_dinh(t, T_TIM_LAI, lambda d: True):
                self.da_bao_mat = False
                return self._noi(CAU["tim_lai"], t)
            return ""
        if self.dang_theo and self.t_thay_vach is not None and t - self.t_thay_vach >= T_MAT:
            if t - self.t_het < T_SAU_HET:
                self.dang_theo = False    # đã báo sắp hết vạch: vạch biến mất là đúng dự kiến, không báo "mất"
                return ""
            self.da_bao_mat = True
            self._nha("lech")
            self._nha("goc")
            return self._noi(CAU["mat"], t)
        # vạch rẽ / ngã rẽ phía trước (một lần cho mỗi chỗ rẽ). Khung chỉ "không chắc" vì có nhánh rẽ vẫn được
        # tính - chính cái nhánh là bằng chứng của chỗ rẽ (lệnh dịch/rẽ thì vẫn không dùng các khung đó)
        if do is not None and do.re:
            self.t_re = t
        elif self.da_bao_re and t - self.t_re > T_QUEN_RE:
            self.da_bao_re = False
        if do is not None and do.re and not self.da_bao_re:
            for k in ("L+1", "L-1", "T"):
                if self._on_dinh(t, T_RE, lambda d, k=k: d.re == k, dung=lambda d: d.tin_cay or d.re):
                    self.da_bao_re = True
                    return self._noi(CAU[k], t)
        if not tin_cay:
            return ""
        if not self.dang_theo and self._on_dinh(t, T_TIM_LAI, lambda d: True):
            self.dang_theo = True
            if not self.da_tung_theo:
                self.da_tung_theo, self.can_bao_thay = True, True

        # sắp hết vạch (một lần cho mỗi đoạn vạch; thấy vạch kéo dài lại 1,5 s thì cho báo lần sau)
        if self.da_bao_het and self._on_dinh(t, 1.5, lambda d: not d.sap_het):
            self.da_bao_het = False
        if not self.da_bao_het and self._on_dinh(t, T_HET, lambda d: d.sap_het):
            self.da_bao_het, self.t_het = True, t
            self._nha("lech")
            self._nha("goc")
            return self._noi(CAU["het"], t)

        # lệch (xét trước)
        if self.lech_dang and self.lech_dang * do.lech < LECH_NHA:
            self._nha("lech")             # nhả theo DẤU: vạch đã sang hẳn bên kia cũng là nhả (không lặp lệnh sai chiều)
        if not self.lech_dang:
            for dau in (1, -1):
                if self._on_dinh(t, T_LECH, lambda d, s=dau: s * d.lech > LECH_BAT):
                    self.lech_dang = dau
        if self.lech_dang:
            self._nha("goc")
            if -self.lech_dang * do.goc >= GOC_VE:      # đang hướng về phía vạch: tự sửa rồi
                return ""
            return self._noi(CAU[("lech", self.lech_dang)], t)

        # đi chéo
        if self.goc_dang and self.goc_dang * do.goc < GOC_NHA:
            self._nha("goc")
        if not self.goc_dang and self.che_do_re != "tat":
            for dau in (1, -1):
                if self._on_dinh(t, T_GOC, lambda d, s=dau: s * d.goc > GOC_BAT and abs(d.lech) <= LECH_BAT) \
                        and (self.che_do_re != "troi" or self._dang_troi(t, dau)):
                    self.goc_dang = dau
        if self.goc_dang:
            return self._noi(CAU[("goc", self.goc_dang)], t)
        return ""


class DanTheoVach:
    """Mặt tiền cho workflow_blindpath: mỗi khung gọi buoc(mask vạch, thời gian) -> (câu, DoVach|None)."""

    def __init__(self, hfov_deg=None, bao_het=None):
        self.hfov = float(os.getenv("AIGLASS_HFOV", "70")) if hfov_deg is None else hfov_deg
        # "Sắp hết vạch": TẮT mặc định (AIGLASS_HET_VACH=1 để bật) - video dev chưa có chỗ hết vạch thật để kiểm,
        # mà xe đỗ đè vạch trông y như hết vạch nếu module né vật cản bỏ sót xe (spec 7.2)
        self.bao_het = os.getenv("AIGLASS_HET_VACH", "0") == "1" if bao_het is None else bao_het
        self.chan_troi = BoChanTroi()
        self.luat = LuatNoi()
        self.do_gan, self.t_do_gan = None, -1e9      # lần đo tin cậy gần nhất (cho hành lang vạch)
        self.t_thay = -1e9
        self.hw = (360, 640)
        self.vat_chan = []
        self.giu_im = False                          # bộ điều khiển vật chắn đang xử lý: tắt lệnh dẫn vạch thường
        self.ls_dau = deque()                        # (t, khoảng cách đầu vạch) của các khung "đầu vạch gần"
        self.ls_re = deque()                         # (t, khoảng cách, loại) của các khung thấy chỗ rẽ trong Z_RE

    def chua_noi(self, cau):
        """process_frame không phát được câu này: để luật nói thử lại."""
        self.luat.huy(cau)

    def da_tren_vach(self, t):
        """Đang đứng trên vạch: đã theo vạch ổn định và lần đo tin cậy gần nhất (<= 1,5 s) lệch <= 0,75 bề rộng."""
        return (self.luat.dang_theo and self.do_gan is not None and t - self.t_do_gan <= T_GIU_DO
                and abs(self.do_gan.lech) <= LECH_BAT)

    def _re(self, t, mask):
        """Chỗ rẽ trong Z_RE m VÀ đang tiến lại gần >= TIEN_GAN_M trong ~1 s (lọc ảo giác ảnh đứng yên)."""
        comp = _chon_dai_gach(mask)
        base = hanh_lang_gan(comp, *self.hw) if comp is not None else None
        info = tim_re(comp, base, self.chan_troi.gia_tri(t), *self.hw, self.hfov)
        if info is None:
            self.ls_re.clear()
            return ""
        k = "T" if info["kieu"] == "T" else f"L{info['ben']:+d}"
        self.ls_re.append((t, info["z"], k))       # giữ lịch sử cả khi còn xa (bằng chứng tiến lại gần từ 6 m vẫn đúng)
        while self.ls_re and self.ls_re[0][0] < t - 2.0:
            self.ls_re.popleft()
        cu = [zi for ti, zi, ki in self.ls_re if ti <= t - 0.7 and ki == k]
        return k if info["z"] <= Z_RE and cu and cu[-1] - info["z"] >= TIEN_GAN_M else ""

    def _sap_het(self, do, t, vat, mask):
        """Đầu vạch gần (dau_vach_gan) VÀ đang tiến lại gần >= TIEN_GAN_M trong ~1 s (người đang bước tới)."""
        y_h = self.chan_troi.gia_tri(t)
        if not dau_vach_gan(do, y_h, *self.hw, self.hfov, vat, mask):
            self.ls_dau.clear()
            return False
        z = khoang_cach_dau_vach(do, y_h, self.hw[1], self.hfov)
        self.ls_dau.append((t, z))
        while self.ls_dau and self.ls_dau[0][0] < t - 2.0:
            self.ls_dau.popleft()
        cu = [zi for ti, zi in self.ls_dau if ti <= t - 0.7]
        return bool(cu) and cu[-1] - z >= TIEN_GAN_M

    def dang_theo_vach(self, t, giay=T_GIU_DO):
        """Vạch được thấy trong `giay` giây gần nhất (để bản né vật cản không bảo "dịch" ra khỏi vạch)."""
        return t - self.t_thay <= giay

    def buoc(self, mask, t, vat=None):
        """vat: danh sách vật của module né vật cản (NeVatCanDoSau.cand) để biết vạch có bị chắn không."""
        if mask is not None:
            self.hw = np.asarray(mask).shape[:2]
        do = do_vach(mask, self.hfov, self.chan_troi.gia_tri(t))
        # chân trời học từ mọi lần đo qua các cổng khác (kể cả lần chỉ trượt cổng chân trời), để khi người
        # cúi/ngẩng hẳn thì trung vị dịch theo sau ~nửa cửa sổ, không bị khóa ở góc chúc cũ
        if do is not None and (do.tin_cay or do.ly_do == LY_DO_CHAN_TROI):
            self.chan_troi.cap_nhat(do.y_vp, t)
        if do is not None and do.tin_cay:
            self.do_gan, self.t_do_gan = do, t
        if do is not None and "ngắn" not in do.ly_do:
            self.t_thay = t
        goc = self.do_gan if t - self.t_do_gan <= T_GIU_DO else None
        self.vat_chan = vat_tren_vach(vat, goc, *self.hw) if (vat and goc is not None) else []
        if do is not None and "ngắn" not in do.ly_do and mask is not None:
            # chỗ rẽ: dùng hành lang từ đoạn vạch gần chân của chính khung này (không cần đo chung tin cậy)
            do.re = self._re(t, mask)
        if do is not None and do.tin_cay:
            if self.bao_het and not do.re:
                do.sap_het = self._sap_het(do, t, vat, mask)
        self.do_cuoi = do                            # lần đo gần nhất (để gỡ lỗi / giả lập đọc lại)
        return self.luat.buoc(do, t, co_vat_chan=bool(self.vat_chan) or self.giu_im), do

    @staticmethod
    def ve(do, viz, H, W):
        """Thêm phần tử vẽ (định dạng _draw_visualizations): hai mép, điểm tụ, bảng lệch/góc/lý do."""
        if do is None:
            viz.append({"type": "data_panel", "data": {"Vạch": "không thấy"}, "position": (25, H - 75)})
            return
        if do.tin_cay:
            y_xa = int(np.clip(do.y_vp, 0, H - 1))
            for a in (do.mep_trai, do.mep_phai):
                viz.append({"type": "polyline", "width": 2, "color": "rgba(0, 255, 255, 1.0)",
                            "points": [[int(np.polyval(a, H - 1)), H - 1], [int(np.polyval(a, y_xa)), y_xa]]})
            viz.append({"type": "circle", "radius": 6, "color": "rgba(255, 0, 255, 1.0)",
                        "center": [int(np.clip(do.x_vp, 0, W - 1)), y_xa]})
            data = {"Lệch": f"{do.lech:+.2f} vạch", "Chéo": f"{do.goc:+.0f}°"}
        else:
            data = {"Vạch": "không chắc", "Lý do": do.ly_do}
        viz.append({"type": "data_panel", "data": data, "position": (25, H - 75)})
