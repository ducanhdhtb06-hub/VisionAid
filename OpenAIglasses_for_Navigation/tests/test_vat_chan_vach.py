"""Vật chắn trên vạch — giai đoạn 1 (spec docs/superpowers/specs/2026-10-27-vong-vat-can-tren-vach-design.md).
Chạy: ../.venv/bin/python -m pytest tests/test_vat_chan_vach.py -q  (đứng ở OpenAIglasses_for_Navigation/)
"""
import numpy as np
import pytest

from dan_theo_vach import DoVach, phan_loai_vat, vi_tri_tren_vach

HG, WG = 360, 640                       # cỡ hình học của module vật cản
F = (WG / 2) / np.tan(np.radians(35))   # góc nhìn 70°
H_CAM, Y_H = 1.5, 180.0                 # camera cao 1,5 m, chân trời giữa ảnh


def mat_dat():
    """Hình học mặt đất phẳng: Z = F·h/(y − y_h), X = (x − W/2)·Z/F, cao 0."""
    yy, xx = np.mgrid[0:HG, 0:WG].astype(float)
    Z = np.where(yy > Y_H + 0.5, F * H_CAM / np.maximum(yy - Y_H, 1e-6), np.inf)
    X = (xx - WG / 2) * Z / F
    return dict(X=X, Z=Z, height=np.zeros((HG, WG)))


def do_vach(nua_rong=0.5, hw=(HG, WG)):
    """Vạch thẳng trước mặt, mép ở X = ±nua_rong (m); mép theo cỡ ảnh hw: x = a·y + b."""
    sy, sx = hw[0] / HG, hw[1] / WG
    mep = []
    for X in (-nua_rong, nua_rong):
        a, b = X / H_CAM, WG / 2 - X * Y_H / H_CAM        # theo cỡ geo
        mep.append((a * sx / sy, b * sx))                 # đổi sang cỡ hw
    return DoVach(lech=0.0, goc=0.0, x_vp=WG / 2 * sx, y_vp=Y_H * sy, tin_cay=True, ly_do="",
                  mep_trai=mep[0], mep_phai=mep[1])


def dat_vat(geo, x0, x1, z, cao=(0.2, 1.0), y=(150, 230), x_px=(300, 340)):
    """Đặt một vật (khối điểm) có X trong [x0, x1] m, ở khoảng cách z m, cao trong `cao` m."""
    m = np.zeros((HG, WG), bool)
    m[y[0]:y[1], x_px[0]:x_px[1]] = True
    n = m.sum()
    geo["X"][m] = np.linspace(x0, x1, n)
    geo["Z"][m] = z
    geo["height"][m] = np.linspace(cao[0], cao[1], n)
    return dict(mask=m, near=m, dist=z, name="motorcycle")


def test_vat_tren_vach_la_chan():
    g = mat_dat()
    o = dat_vat(g, -0.3, 0.3, 3.0)
    assert phan_loai_vat(o, do_vach(), g, (HG, WG)) == "chan"


def test_vat_canh_vach_la_canh():
    g = mat_dat()
    o = dat_vat(g, 1.0, 1.6, 3.0)                       # cách mép vạch 0,5 m (> 0,2 m nới vai)
    assert phan_loai_vat(o, do_vach(), g, (HG, WG)) == "canh"


def _xe_co_tay_lai(x_tay_lai):
    # bánh xe ngoài vạch (X 0,9–1,5 m), tay lái/gương nhô vào tới X = x_tay_lai ở độ cao 1 m
    g = mat_dat()
    o = dat_vat(g, 0.9, 1.5, 3.0, cao=(0.2, 0.6))
    tay_lai = np.zeros((HG, WG), bool)
    tay_lai[140:150, 280:300] = True
    g["X"][tay_lai], g["Z"][tay_lai], g["height"][tay_lai] = x_tay_lai, 3.0, 1.0
    o["mask"] = o["mask"] | tay_lai
    return o, g


def test_phan_nho_vao_vach_la_chan():
    o, g = _xe_co_tay_lai(0.4)                          # tay lái đè lên vạch (mép phải X 0,5 m)
    assert phan_loai_vat(o, do_vach(), g, (HG, WG)) == "chan"


def test_chi_cham_le_vai_la_canh():
    # xe đỗ CẠNH vạch, chỉ lấn vào phần lề 0,2 m nới cho vai (chạy 7 video 2026-10-28: dev đo lấn 1–20 cm, vd. IMG_7356
    # 150 s, khiem 73 s; xe đè vạch thật lấn 30–90 cm). Người dùng (28/10): chỉ quan tâm vật trong vùng vạch -> im.
    o, g = _xe_co_tay_lai(0.6)
    assert phan_loai_vat(o, do_vach(), g, (HG, WG)) == "canh"


def test_khong_co_do_vach_thi_khong_chac():
    g = mat_dat()
    o = dat_vat(g, 1.0, 1.6, 3.0)
    assert phan_loai_vat(o, None, g, (HG, WG)) == "khong_chac"
    assert phan_loai_vat(o, do_vach(), None, (HG, WG)) == "khong_chac"


def test_mep_vach_bi_vat_che_het_thi_khong_chac():
    # vật phủ kín cả hai mép vạch ở mọi hàng dưới chân trời -> không dựng được lối đi 3D
    g = mat_dat()
    o = dat_vat(g, -0.3, 0.3, 3.0)
    g["height"][int(Y_H) + 1:, :] = 0.8
    assert phan_loai_vat(o, do_vach(), g, (HG, WG)) == "khong_chac"


def test_vat_xa_hon_5m_khong_chan():
    g = mat_dat()
    o = dat_vat(g, -0.3, 0.3, 7.0)
    assert phan_loai_vat(o, do_vach(), g, (HG, WG)) == "canh"


def test_mep_vach_o_co_anh_khac_van_dung():
    # mask vạch 480x640 (ESP32 VGA), hình học 360x640 (module vật cản)
    g = mat_dat()
    o = dat_vat(g, 1.0, 1.6, 3.0)
    assert phan_loai_vat(o, do_vach(hw=(480, 640)), g, (480, 640)) == "canh"
    o2 = dat_vat(g, -0.3, 0.3, 3.0)
    assert phan_loai_vat(o2, do_vach(hw=(480, 640)), g, (480, 640)) == "chan"


# ---------- Task 3: bộ điều khiển BoVatChan ----------
from vat_chan_vach import BoVatChan  # noqa: E402

XE = "Xe máy chắn vạch. Đi sang trái."


def vat(d, ben=("trai", "phai"), ten="xe máy"):
    return dict(dist=d, ten=ten, ben=list(ben))


def chay(bo, buoc_ds, t0=0.0, dt=0.2):
    """buoc_ds: danh sách (tren_vach, vat, lech, thay_vach) theo từng khung; trả (t, câu) có câu."""
    ra, t = [], t0
    for tv, v, lech, thay in buoc_ds:
        c = bo.buoc(t, tv, v, lech, thay)
        if c:
            ra.append((round(t, 2), c))
        t += dt
    return ra, t


def test_bao_khi_3m_va_chot_ben():
    bo = BoVatChan()
    ds = [(True, vat(5.0 - 0.25 * i), 0.0, True) for i in range(10)]           # 5,0 -> 2,75 m
    ds += [(True, vat(2.5, ben=("phai",)), 0.0, True)] * 5                       # sát vật: chỉ còn thấy bên phải
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE]
    assert ra[0][0] == pytest.approx(1.6)                                         # khung đầu tiên <= 3,0 m
    assert bo.dang_xu_ly and bo.ben_chot == "trai"


def test_khong_bao_khi_khong_tren_vach():
    bo = BoVatChan()
    ra, _ = chay(bo, [(False, vat(2.5), 0.0, True)] * 10)
    assert ra == [] and not bo.dang_xu_ly


def test_chan_het():
    # người dùng 28/10: "vạch bị chắn sao lại bắt họ dừng lại, phải bảo họ vòng ra" -> không chắc bên nào: dò gậy vòng
    bo = BoVatChan()
    ra, t = chay(bo, [(True, vat(2.8, ben=()), 0.0, True)] * 3)
    assert [c for _, c in ra] == ["Xe máy chắn vạch. Dùng gậy dò đường vòng."]
    assert bo.dang_xu_ly
    ra, _ = chay(bo, [(True, None, 0.0, True)] * 10, t0=t)                       # vật đi mất 2 s
    assert ra == [] and not bo.dang_xu_ly


def test_ve_vach_sau_khi_roi():
    bo = BoVatChan()
    ds = [(True, vat(2.8), 0.0, True)]                                            # báo
    ds += [(False, vat(2.0), 1.2, True)] * 5                                     # bước sang trái, còn thấy vật
    ds += [(False, None, 1.2, True)] * 10                                        # đi ngang vật (khuất) 2 s
    ds += [(True, None, 0.1, True)] * 3                                          # về vạch
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE, "Đã về vạch."]
    assert not bo.dang_xu_ly


def test_khong_ve_vach_khi_chua_roi():
    # ngay sau câu báo người còn đứng trên vạch (chưa bước) -> không được nói "Đã về vạch"
    bo = BoVatChan()
    ra, _ = chay(bo, [(True, vat(2.8), 0.0, True)] * 3 + [(True, None, 0.0, True)] * 15)
    assert [c for _, c in ra] == [XE]


def test_vat_di_mat_thi_im_va_xong():
    bo = BoVatChan()
    chay(bo, [(True, vat(2.8), 0.0, True)] + [(True, None, 0.0, True)] * 10)
    assert not bo.dang_xu_ly                                                      # vật hết chắn >= 1,5 s


def test_nhac_15s_mot_lan():
    bo = BoVatChan()
    ds = [(True, vat(2.8), 0.0, True)] + [(False, None, 0.9, True)] * 100        # rời vạch 20 s chưa về
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE, "Vạch ở bên phải."]
    assert ra[1][0] == pytest.approx(0.2 + 15.0, abs=0.21)


def test_nhac_khong_thay_vach():
    bo = BoVatChan()
    ra, _ = chay(bo, [(True, vat(2.8), 0.0, True)] + [(False, None, None, False)] * 90)
    assert [c for _, c in ra] == [XE, "Chưa thấy lại vạch, hãy dùng gậy."]


def test_chua_noi_thi_bao_lai():
    bo = BoVatChan()
    assert bo.buoc(0.0, True, vat(2.8), 0.0, True) == XE
    bo.chua_noi()                                                                 # thua câu khác, không phát được
    assert not bo.dang_xu_ly
    assert bo.buoc(0.2, True, vat(2.7), 0.0, True) == XE


def test_giu_ben_da_chon_tu_xa():
    bo = BoVatChan()
    ds = [(True, vat(4.5, ben=("phai", "trai")), 0.0, True)] * 3                 # từ xa: bên phải tốt hơn
    ds += [(True, vat(2.9, ben=("trai", "phai")), 0.0, True)]                    # gần: thứ tự đổi, phải vẫn đi được
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == ["Xe máy chắn vạch. Đi sang phải."]


def test_quay_ve_som_truoc_vat_thi_bao_lai():
    bo = BoVatChan()
    ds = [(True, vat(2.8), 0.0, True)] + [(False, vat(2.4), 1.0, True)] * 12    # rời vạch thật 2,4 s
    ds += [(True, vat(1.8), 0.0, True)] * 3                                      # về vạch khi vật còn trước mặt
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE, XE]


def test_ten_vat_khong_ro():
    bo = BoVatChan()
    assert bo.buoc(0.0, True, vat(2.8, ten=None), 0.0, True) == "Vật cản chắn vạch. Đi sang trái."


# ---------- Task 4: xếp bên vòng ----------
from vat_chan_vach import xep_ben  # noqa: E402


def test_xep_ben_bo_ben_duong():
    free = {"dich_trai": (True, 0.8), "dich_phai": (True, 0.6)}
    assert xep_ben(free, {"dich_trai": 0.0, "dich_phai": 0.5}) == ["trai"]      # bên phải phần lớn là lòng đường


def test_xep_ben_uu_tien_xa_duong():
    free = {"dich_trai": (True, 0.6), "dich_phai": (True, 0.9)}
    assert xep_ben(free, {"dich_trai": 0.2, "dich_phai": 0.0}) == ["phai", "trai"]   # xa đường trước, dịch xa hơn chút


def test_xep_ben_khong_biet_duong_thi_dich_it():
    free = {"dich_trai": (True, 0.9), "dich_phai": (True, 0.6)}
    assert xep_ben(free, {}) == ["phai", "trai"]


def test_xep_ben_khong_ben_nao():
    assert xep_ben({"dich_trai": (False, 0.7), "dich_phai": (False, 0.7)}, {}) == []
    assert xep_ben({}, {}) == []


def test_vat_moi_khac_ben_khi_dang_xu_ly_thi_bao_lai():
    # lỗi thật (IMG_7358 6,9–9,5 s, chạy 2026-10-27): báo vật A xong, vật B khác trên vạch tiến tới 3 m mà kính im
    bo = BoVatChan()
    ds = [(True, vat(1.4, ten="ghế"), 0.0, True)]                                # báo A (vòng trái)
    ds += [(True, vat(4.9 - 0.2 * i, ben=("phai",)), 0.0, True) for i in range(12)]  # B: chỉ vòng phải được
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == ["Ghế chắn vạch. Đi sang trái.", "Xe máy chắn vạch. Đi sang phải."]


def test_day_xe_cung_ben_chi_noi_mot_cau():
    # chạy 7 video (2026-10-28): dãy xe đỗ trên vạch, mỗi xe một câu "… vòng bên trái" (IMG_7354: 17 câu mới).
    # Đang đi vòng bên trái mà xe sau vẫn vòng trái được -> lời dặn cũ vẫn đúng, im; về vạch nói "Đã về vạch" một lần.
    bo = BoVatChan()
    ds = [(True, vat(2.8), 0.0, True)]                                            # xe 1: báo
    ds += [(False, vat(2.0), 1.0, True)] * 12                                    # đi vòng 2,4 s
    ds += [(False, vat(4.5 - 0.2 * i), 1.0, True) for i in range(10)]            # xe 2 cùng hàng, vẫn vòng trái được
    ds += [(False, None, 1.0, True)] * 10                                        # qua hết dãy
    ds += [(True, None, 0.0, True)] * 3                                          # về vạch
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE, "Đã về vạch."]


# ---------- Sửa sau chạy 7 video (2026-10-28) ----------
from vat_chan_vach import OnDinh, ten_noi  # noqa: E402


def test_on_dinh_can_thay_lien_tuc_1s():
    # vết lạ / người đi ngang chỉ hiện vài khung (IMG_7354 303 s "Xe đạp chắn vạch" trên nền trống)
    od = OnDinh()
    kq = [od.cap_nhat(0.2 * i, vat(3.0 - 0.1 * i)) for i in range(7)]             # 0 -> 1,2 s
    assert kq[:5] == [False] * 5 and kq[5] and kq[6]                              # đủ 1 s ở t = 1,0


def test_on_dinh_mat_lau_hoac_vat_khac_thi_dem_lai():
    od = OnDinh()
    for i in range(6):
        od.cap_nhat(0.2 * i, vat(3.0))
    assert od.cap_nhat(1.2, None) is True                                         # sót 1 khung: chưa tính là mất
    assert od.cap_nhat(3.0, vat(2.9)) is False                                    # mất 1,8 s -> đếm lại
    for i in range(6):
        od.cap_nhat(5.0 + 0.2 * i, vat(2.5))
    assert od.cap_nhat(6.2, vat(4.0)) is False                                    # xa hơn > 1 m: vật khác


def test_ten_noi_chi_ten_tin_duoc():
    assert ten_noi("parked motorcycle", "xe máy") == "xe máy"
    assert ten_noi("bench", "ghế dài") == "ghế dài"
    assert ten_noi("truck", "xe tải") is None                                     # tấm quây gọi nhầm "Xe tải" (7359)
    assert ten_noi("cat", "mèo") is None                                          # túi nilon gọi nhầm "Mèo" (7354)
    assert ten_noi(None, None) is None


def _ne_khong_model():
    from ne_vat_can_do_sau import NeVatCanDoSau
    ne = NeVatCanDoSau.__new__(NeVatCanDoSau)                                     # không nạp model
    ne.yo = []
    return ne


def test_dung_yen_nguoi_dong_vat_va_chua_ro_toc_do():
    # người đi tới bị coi là vật đứng chắn (7359 67 s "Vạch bị chắn"; 7356 114 s người đi cùng chiều)
    ne = _ne_khong_model()
    xe_do = dict(src="yoloe", name="motorcycle", vok=True, v=0.0, vx=0.0)
    assert ne.dung_yen(xe_do)
    assert not ne.dung_yen(dict(xe_do, name="person"))                            # người: có thể đi bất cứ lúc nào
    assert not ne.dung_yen(dict(xe_do, name="dog"))
    assert not ne.dung_yen(dict(xe_do, vok=False))                                # chưa đo được tốc độ
    assert not ne.dung_yen(dict(xe_do, v=2.5))


def test_vat_moi_sau_chan_het():
    bo = BoVatChan()
    ds = [(True, vat(1.7, ben=()), 0.0, True)]                                   # A: không bên nào -> chắn hết
    ds += [(True, vat(4.0 - 0.3 * i), 0.0, True) for i in range(6)]              # B mới: 4,0 -> 2,5 m
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == ["Xe máy chắn vạch. Dùng gậy dò đường vòng.", XE]


def test_cung_vat_nhieu_khung_khong_coi_la_moi():
    bo = BoVatChan()
    ds = [(True, vat(d), 0.0, True) for d in (2.9, 3.1, 2.8, 3.0, 2.7, 2.9, 2.6)]  # đo dao động ±0,3 m
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE]


def test_ben_di_duoc_cho_xe_do_yoloe():
    # lỗi thật (IMG_7358, 2026-10-27): xe đỗ do YOLOE thấy (v ~ 0) -> plan_avoid coi là "đi cùng chiều", không xét hai
    # bên -> không chọn được bên nào, kính nói "Vạch bị chắn" dù bên trái trống
    from ne_vat_can_do_sau import NeVatCanDoSau
    ne = NeVatCanDoSau.__new__(NeVatCanDoSau)                       # không nạp model
    ne.W, ne.H, ne.F = WG, HG, F
    ne.geo = mat_dat()
    o = dat_vat(ne.geo, 0.5, 1.5, 3.0)
    o.update(src="yoloe", name="motorcycle", vok=True, v=0.0, vx=0.0, xl=0.5, xr=1.5, high=False)
    assert ne.ben_di_duoc(o) == ["trai"]                            # phải: cần dịch 2,0 m > 1,5 m tối đa


def test_xe_do_lien_nhau_cung_cau_khong_noi_lap():
    # IMG_7358 2,0 s và 2,62 s: hai xe đỗ liền nhau -> cùng một câu nói lại sau 0,6 s (thừa)
    bo = BoVatChan()
    ds = [(True, vat(1.7), 0.0, True)] * 2 + [(True, vat(2.9), 0.0, True)] * 2   # xe thứ hai ngay sau
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE]
    assert bo.dang_xu_ly


def test_xe_do_lien_nhau_khac_ben_van_noi():
    bo = BoVatChan()
    ds = [(True, vat(1.7), 0.0, True)] * 2 + [(True, vat(2.9, ben=("phai",)), 0.0, True)] * 2
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE, "Xe máy chắn vạch. Đi sang phải."]


def test_chap_chon_tren_vach_khong_tinh_la_roi():
    # quét 4 video dev (2026-10-27): người quay không đi vòng, phép đo "trên vạch" chập chờn < 2 s -> 6 câu
    # "Đã về vạch." thừa ở IMG_7354. Đi vòng thật qua một xe mất >= 2 s ngoài vạch.
    bo = BoVatChan()
    ds = [(True, vat(2.8), 0.0, True)] + [(False, vat(2.5), 0.9, True)] * 4        # ra ngoài 0,8 s
    ds += [(True, None, 0.0, True)] * 12                                         # lại trên vạch, vật hết chắn
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE]


# ---------- Câu rõ vị trí xe và hướng đi (người dùng 28/10: "xe chắn bên phải mà bảo bên trái") ----------
from vat_chan_vach import cau_bao  # noqa: E402


def test_vi_tri_vat_tren_vach():
    g = mat_dat()
    assert vi_tri_tren_vach(dat_vat(g, 0.1, 0.45, 3.0), do_vach(), g, (HG, WG)) == "phai"
    g = mat_dat()
    assert vi_tri_tren_vach(dat_vat(g, -0.45, -0.1, 3.0), do_vach(), g, (HG, WG)) == "trai"
    g = mat_dat()
    assert vi_tri_tren_vach(dat_vat(g, -0.3, 0.3, 3.0), do_vach(), g, (HG, WG)) == "giua"
    g = mat_dat()
    assert vi_tri_tren_vach(dat_vat(g, 1.0, 1.6, 3.0), do_vach(), g, (HG, WG)) is None   # không đè lên vạch
    assert vi_tri_tren_vach(dat_vat(g, 0.1, 0.45, 3.0), None, g, (HG, WG)) is None


def test_cau_noi_ro_vi_tri_va_huong_di():
    # "vòng bên trái" bị nghe thành "xe ở bên trái": nói rõ xe ở đâu, rồi đi sang đâu
    assert cau_bao("xe máy", "trai", "phai") == "Xe máy chắn vạch bên phải. Đi sang trái."
    assert cau_bao("xe máy", "phai", "trai") == "Xe máy chắn vạch bên trái. Đi sang phải."
    assert cau_bao("ghế", "trai", "giua") == "Ghế chắn giữa vạch. Đi sang trái."
    assert cau_bao(None, "phai", None) == "Vật cản chắn vạch. Đi sang phải."


def test_bo_dieu_khien_noi_vi_tri():
    bo = BoVatChan()
    v = dict(vat(2.8), vi_tri="phai")
    assert bo.buoc(0.0, True, v, 0.0, True) == "Xe máy chắn vạch bên phải. Đi sang trái."


def test_ghep_cau_tu_manh_giong(monkeypatch):
    # câu mới phải ghép đúng mảnh WAV (thiếu mảnh -> người khiếm thị không nghe được)
    import ne_vat_can_do_sau as N
    ne = N.NeVatCanDoSau.__new__(N.NeVatCanDoSau)
    dung = []

    class Loa:
        def clip(self, k):
            dung.append(k)
            return np.zeros(10, np.int16)
    ne.speaker = Loa()
    monkeypatch.setattr(N, "_dang_ky_cau_noi", lambda cau, pcm: None)
    ne.dang_ky_cau("Xe máy chắn vạch bên phải. Đi sang trái.", "motorcycle")
    assert dung == ["ten_xe_may", "chan_vach_ben_phai", "di_sang_trai"]
    dung.clear()
    ne.dang_ky_cau("Vật cản chắn giữa vạch. Đi sang phải.", None)
    assert dung == ["ten_vat_can", "chan_giua_vach", "di_sang_phai"]
    dung.clear()
    ne.dang_ky_cau("Ghế chắn vạch bên trái. Đi sang phải.", "chair")
    assert dung == ["ten_ghe", "chan_vach_ben_trai", "di_sang_phai"]


# ---------- Không đọc số mét (người dùng 28/10: "bỏ kiểu cách mấy mét, nói ra nó sai") ----------
def test_cau_canh_bao_khong_co_so_met():
    # kính đo khoảng cách sai (~10% gần hơn thật, đo COLMAP 2026-10-27): nói số mét sai còn hại hơn không nói
    from ne_vat_can_do_sau import cau_vat
    assert cau_vat("dung_lai", "xe máy", "phía trước", False) == "Dừng lại. Xe máy"
    assert cau_vat(None, "sắp xuống lòng đường", "phía trước", False) == "Sắp xuống lòng đường, phía trước"
    assert cau_vat(None, "xe máy", "bên trái", True) == "Xe máy, bên trái, đang tới"


def test_loa_khong_ghep_manh_so_met():
    from ne_vat_can_do_sau import K               # module vật cản tự thêm đường dẫn thu_nghiem_do_sau
    sp = K.Speaker(noi_khoang_cach=False)
    dung = []
    sp.clip = lambda k: (dung.append(k), np.zeros(10, np.int16))[1]
    o = dict(dist=2.3, high=False, side="phía trước")
    assert sp.update(10.0, o, "motorcycle", act="dung_lai")
    assert dung and not any(k.startswith("cach_") for k in dung), dung



# ---------- Đợt sửa dứt điểm 28/10: chọn bên theo cả quãng tiến lại gần; không bảo dừng với xe đỗ ----------
def test_chon_ben_theo_ca_quang_tien_lai():
    # lỗi thật IMG_7358 ~9 s (chẩn đoán 28/10): bên trái lúc trống lúc "vướng" theo từng khung; đúng khung 3 m "vướng"
    # -> "Vạch bị chắn". Chọn bên theo số đông các khung từ 5 m tới 3 m.
    bo = BoVatChan()
    ds = [(True, vat(5.0 - 0.25 * i, ben=("trai",) if i % 3 else ()), 0.0, True) for i in range(8)]   # trống 5/8
    ds += [(True, vat(2.9, ben=()), 0.0, True)]                                    # khung 3 m: "vướng"
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == [XE]


def test_ben_it_khi_trong_thi_khong_chon():
    bo = BoVatChan()
    ds = [(True, vat(5.0 - 0.25 * i, ben=("trai",) if i % 4 == 0 else ()), 0.0, True) for i in range(9)]   # trống 3/9
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == ["Xe máy chắn vạch. Dùng gậy dò đường vòng."]


def test_khong_ben_nao_thi_huong_xa_long_duong():
    # không bên nào chắc trống, nhưng bên phải là lòng đường -> đi sang trái (phía nhà), dò gậy; KHÔNG BAO GIỜ sang đường
    bo = BoVatChan()
    v = dict(vat(2.8, ben=()), duong={"trai": 0.0, "phai": 0.8}, vi_tri="giua")
    ra, _ = chay(bo, [(True, v, 0.0, True)] * 2)
    assert [c for _, c in ra] == ["Xe máy chắn giữa vạch. Đi sang trái, dò gậy."]
    assert bo.dang_xu_ly and bo.ben_chot == "trai"


def test_ca_hai_ben_la_duong_thi_do_gay():
    bo = BoVatChan()
    v = dict(vat(2.8, ben=()), duong={"trai": 0.7, "phai": 0.8})
    assert bo.buoc(0.0, True, v, 0.0, True) == "Xe máy chắn vạch. Dùng gậy dò đường vòng."


def test_do_gay_roi_ve_vach():
    bo = BoVatChan()
    v = dict(vat(2.8, ben=()), duong={"trai": 0.0, "phai": 0.8})
    ds = [(True, v, 0.0, True)] + [(False, None, 1.0, True)] * 15 + [(True, None, 0.0, True)] * 3
    ra, _ = chay(bo, ds)
    assert [c for _, c in ra] == ["Xe máy chắn vạch. Đi sang trái, dò gậy.", "Đã về vạch."]


def test_ghep_cau_do_gay(monkeypatch):
    import ne_vat_can_do_sau as N
    ne = N.NeVatCanDoSau.__new__(N.NeVatCanDoSau)
    dung = []

    class Loa:
        def clip(self, k):
            dung.append(k)
            return np.zeros(10, np.int16)
    ne.speaker = Loa()
    monkeypatch.setattr(N, "_dang_ky_cau_noi", lambda cau, pcm: None)
    ne.dang_ky_cau("Xe máy chắn giữa vạch. Đi sang trái, dò gậy.", "motorcycle")
    assert dung == ["ten_xe_may", "chan_giua_vach", "di_sang_trai", "do_gay"]
    dung.clear()
    ne.dang_ky_cau("Vật cản chắn vạch. Dùng gậy dò đường vòng.", None)
    assert dung == ["ten_vat_can", "chan_vach", "dung_gay_do_vong"]


def test_xe_dang_toi_noi_ro_dang_toi():
    # lỗi thật IMG_7354 353 s (28/10): "…Đi sang trái." rồi "Dừng lại. Xe máy" — câu sau là xe KHÁC đang chạy tới 7 m,
    # thiếu chữ "đang tới" nên nghe như kính tự mâu thuẫn
    from ne_vat_can_do_sau import cau_vat, K
    assert cau_vat("dung_lai", "xe máy", "phía trước", True) == "Dừng lại. Xe máy, phía trước, đang tới"
    sp = K.Speaker(noi_khoang_cach=False)
    dung = []
    sp.clip = lambda k: (dung.append(k), np.zeros(10, np.int16))[1]
    assert sp.update(10.0, dict(dist=7.0, high=False, side="phía trước"), "motorcycle", act="dung_lai", toi=True)
    assert dung == ["dung_lai", "ten_xe_may", "phia_truoc", "dang_toi"]


def test_che_do_chi_bao_tin_khong_ra_lenh():
    # xe đỗ ngay trước mặt mà bộ điều khiển chưa nhận: chỉ báo "Xe máy, phía trước", không "Dừng lại" / "Dịch"
    from ne_vat_can_do_sau import lenh_sau_loc
    assert lenh_sau_loc("tin", "dung_lai") is None
    assert lenh_sau_loc("tin", "dich_trai") is None
    assert lenh_sau_loc(True, "dung_lai") == "dung_lai"


def test_mep_via_he_2_trong_3_khung():
    # mép vỉa hè sát ngưỡng chỉ thấy 2/3 khung liền tùy pha khung gửi -> bỏ sót "Sắp xuống lòng đường" (28/10)
    from collections import deque
    from ne_vat_can_do_sau import du_khung
    assert du_khung(deque([True, False, True], maxlen=3))
    assert du_khung(deque([True, True, True], maxlen=3))
    assert not du_khung(deque([False, False, True], maxlen=3))
    assert not du_khung(deque([True, True], maxlen=3))                             # chưa đủ cửa sổ


# ---------- Xe tới gần che mép vạch (người dùng 29/10, demo IMG_7354 285–355 s: "bảo đi vòng rồi lại 'Xe máy, phía
# trước'"; "phải nói là xe máy lấn vạch đi vòng sang trái") ----------
from vat_chan_vach import NhoVatChan, T_HET, VAT_MOI  # noqa: E402


def test_nho_vat_chan_khi_xe_che_mep_vach():
    # IMG_7354 299,9 s / 341,3 s: tới gần, chính chiếc xe che mép vạch -> phan_loai_vat "khong_chac"; xe vẫn ở đó
    n = NhoVatChan()
    n.cap_nhat(10.0, 3.4, "giua")
    assert n.loai(10.3, "khong_chac", 3.1) == "chan"
    assert n.loai(10.3, "chan", 3.1) == "chan"
    assert n.loai(10.3, "canh", 3.1) == "canh"                                   # đo lại được vạch, xe nằm ngoài
    assert n.loai(10.3, "khong_chac", 3.4 + VAT_MOI + 0.1) == "khong_chac"       # xa hơn > 1 m: vật khác
    assert n.loai(10.0 + T_HET, "khong_chac", 3.0) == "khong_chac"               # mất >= 1,5 s: quên
    assert NhoVatChan().loai(10.0, "khong_chac", 3.0) == "khong_chac"            # chưa từng thấy chắn


def test_nho_vat_chan_giu_vi_tri_lan_do_duoc():
    n = NhoVatChan()
    assert n.cap_nhat(10.0, 3.4, "phai") == "phai"
    assert n.cap_nhat(10.3, 3.0, None) == "phai"                                 # không đo được: vị trí lần đo gần nhất
    assert n.cap_nhat(12.0, 4.0, None) is None                                   # vật mới sau khi đã quên


def test_mat_do_vach_thoang_qua_van_bao_vong():
    # IMG_7354 312 s: xe che vạch, mất lần đo vạch đúng lúc xe tới 3 m -> "không trên vạch" -> không bảo vòng, module
    # vật cản nói "Xe máy, phía trước". Ra ngoài vạch < T_ROI_MIN chưa là rời vạch (như khi đã báo).
    bo = BoVatChan()
    _, t = chay(bo, [(True, vat(3.6 - 0.2 * i), 0.0, True) for i in range(3)])  # 3,6 -> 3,2 m trên vạch
    ra, _ = chay(bo, [(False, vat(2.9), None, False)] * 2, t0=t)                # mất đo vạch 0,4 s
    assert [c for _, c in ra] == [XE]


def test_roi_vach_lau_thi_khong_bao_vong():
    bo = BoVatChan()
    chay(bo, [(True, None, 0.0, True)])
    ra, _ = chay(bo, [(False, vat(2.9), 1.2, True)] * 5, t0=2.0)                # ra ngoài vạch từ 2 s trước
    assert ra == [] and not bo.dang_xu_ly
    assert bo.tren_gan(2.0, False) is False and bo.tren_gan(2.0, True) is True


# ---------- "Đang tới" nhầm với xe đỗ (IMG_7358 99,3 s, 29/10: "Đi sang trái." rồi "Dừng lại. Xe máy, phía trước,
# đang tới" về xe đỗ) ----------
def test_toc_do_gan_xe_do_bi_ghep_nham():
    # hai xe đỗ liền nhau bị ghép thành một vết (3,8 m -> 2,0 m trong 0,3 s) -> trung vị tốc độ 2,07 m/s >= V_TOI.
    # Nửa giây gần nhất chỉ tiến lại bằng tốc độ người đi (0,8 m/s): không phải đang tới.
    from ne_vat_can_do_sau import toc_do_gan
    gia = [(0.0, 4.19), (0.54, 3.79), (0.82, 1.99), (1.09, 1.76), (1.38, 1.55)]
    assert toc_do_gan(gia) == pytest.approx(0.79, abs=0.02)
    that = [(0.0, 9.62), (0.24, 7.26), (0.49, 5.66), (0.73, 3.81), (1.04, 2.3)]     # IMG_7358 76,8 s: xe chạy tới thật
    assert toc_do_gan(that) == pytest.approx(6.1, abs=0.1)
    assert toc_do_gan([(0.0, 3.0), (0.3, 2.9)]) is None                               # chưa đủ nửa giây
    assert toc_do_gan(None) is None


def test_dang_toi_can_ca_nua_giay_gan_nhat():
    from ne_vat_can_do_sau import dang_toi
    assert dang_toi(dict(v=6.0, v_gan=6.1))
    assert not dang_toi(dict(v=2.06, v_gan=0.79))                                    # chỉ trung vị cao do ghép nhầm
    assert dang_toi(dict(v=2.5, v_gan=None))                                         # không biết: như cũ
    assert not dang_toi(dict(v=1.0, v_gan=3.0))


def test_dung_yen_khi_toc_do_cao_do_ghep_nham():
    ne = _ne_khong_model()
    xe = dict(src="yoloe", name="motorcycle", vok=True, v=2.06, v_gan=0.79, vx=0.0)
    assert ne.dung_yen(xe)
    assert not ne.dung_yen(dict(xe, v_gan=3.0))                                       # đang tới thật


def test_nho_vat_chan_tiep():
    n = NhoVatChan()
    assert not n.tiep(10.0, 3.0)
    n.cap_nhat(10.0, 2.3, "phai")
    assert n.tiep(10.3, 2.1) and not n.tiep(10.3, 3.4) and not n.tiep(11.6, 2.0)


def test_co_the_dang_do():
    # IMG_7358 98,5–99,8 s (29/10): đi ngang xe đỗ, YOLOE mất dấu lập vết mới ("chưa đo được tốc độ"); IMG_7354 26,3 s:
    # người đi chéo -> tốc độ ngang thoáng qua. Vật nối tiếp xe đỗ vừa thấy vẫn là xe đỗ, trừ người/con vật/xe đang tới.
    ne = _ne_khong_model()
    moi = dict(src="yoloe", name="motorcycle", vok=False, v=0.0, vx=0.0)
    assert ne.co_the_dang_do(moi)
    assert ne.co_the_dang_do(dict(moi, vok=True, vx=1.2))
    assert not ne.co_the_dang_do(dict(moi, name="person"))
    assert not ne.co_the_dang_do(dict(moi, toi=True))
    assert not ne.co_the_dang_do(dict(moi, nhanh=True))


def test_xe_vua_bao_dang_toi_thi_chi_noi_dung_lai():
    # IMG_7358 88,1/88,3 s và 104,5/105,2 s (29/10): "Xe máy, bên trái, đang tới" rồi ngay sau "Dừng lại. Xe máy, bên
    # trái, đang tới" về cùng chiếc xe -> lặp. Lệnh dừng vẫn phải nói (gấp), nhưng chỉ "Dừng lại."
    from ne_vat_can_do_sau import cau_vat, K
    sp = K.Speaker(im_xa=True, noi_khoang_cach=False)
    dung = []
    sp.clip = lambda k: (dung.append(k), np.zeros(10, np.int16))[1]
    xe = dict(dist=6.5, high=False, side="bên trái")
    assert sp.update(10.0, xe, "motorcycle", act=None, toi=True) and not sp.ngan
    dung.clear()
    assert sp.update(10.3, dict(xe, dist=6.0), "motorcycle", act="dung_lai", toi=True) and sp.ngan
    assert dung == ["dung_lai"]
    assert cau_vat("dung_lai", "xe máy", "bên trái", True, ngan=True) == "Dừng lại."
    dung.clear()                                                                     # xe khác / lâu sau: câu đầy đủ
    assert sp.update(20.0, dict(xe, dist=6.0), "motorcycle", act="dung_lai", toi=True) and not sp.ngan
    assert dung == ["dung_lai", "ten_xe_may", "ben_trai", "dang_toi"]


def test_khong_bao_di_ve_phia_xe_dang_chan():
    # IMG_7354 218,8 s (29/10): "Xe máy chắn vạch bên phải. Đi sang phải, dò gậy." — xe ở nửa phải mà bảo sang phải
    # (phía xa lòng đường) -> nghe như mâu thuẫn, và lối trống thật là nửa bên kia. Không chắc bên nào: dò gậy vòng.
    bo = BoVatChan()
    v = dict(vat(2.8, ben=()), duong={"trai": 0.8, "phai": 0.0}, vi_tri="phai")
    assert bo.buoc(0.0, True, v, 0.0, True) == "Xe máy chắn vạch bên phải. Dùng gậy dò đường vòng."
    bo = BoVatChan()                                                                # xe ở nửa trái: sang phải vẫn được
    assert bo.buoc(0.0, True, dict(v, vi_tri="trai"), 0.0, True) == "Xe máy chắn vạch bên trái. Đi sang phải, dò gậy."


def test_kich_thuoc_phai_to_ra_khi_dang_toi():
    # IMG_7354 299,7 s (29/10): xe đạp ĐỖ ở 11 m, độ sâu nhảy 11,3 -> 7,5 m trong 0,8 s -> "Xe đạp, phía trước, đang tới"
    # ngay sau "Đi sang trái.". Xe tới gần thật thì trên ảnh to ra theo tỉ lệ khoảng cách; đây gần như không to ra.
    from ne_vat_can_do_sau import kich_thuoc_hop_ly
    assert not kich_thuoc_hop_ly([(0.0, 11.34, 40, False), (0.4, 9.7, 41, False), (0.82, 7.52, 42, False)])
    assert kich_thuoc_hop_ly([(0.0, 9.19, 40, False), (0.5, 6.53, 55, False)])          # xe chạy tới thật: to ra 1,4 lần
    assert kich_thuoc_hop_ly([(0.0, 5.0, 80, False), (0.6, 3.0, 90, True)])              # chạm mép ảnh: không biết
    assert kich_thuoc_hop_ly([(0.0, 5.0, 80, False), (0.3, 3.0, 82, False)])             # chưa đủ nửa giây: không biết


def test_dang_toi_can_kich_thuoc_hop_ly():
    from ne_vat_can_do_sau import dang_toi
    assert not dang_toi(dict(v=3.5, v_gan=4.7, to_ra=False))
    assert dang_toi(dict(v=3.5, v_gan=4.7, to_ra=True))


def test_xe_dang_toi_lai_gan_nhanh_van_la_cung_xe():
    # IMG_7358 88,0/88,3/89,0 s (29/10): xe lao tới 6,5 -> 2 m trong 1 s; "cùng vật" chỉ nhận lệch < 2 m -> coi là xe
    # mới -> nói lại cả câu lần thứ ba. Xe đang tới cùng tên, gần lại, trong 3 s: vẫn là xe đó.
    from ne_vat_can_do_sau import K
    sp = K.Speaker(im_xa=True, noi_khoang_cach=False)
    sp.clip = lambda k: np.zeros(10, np.int16)
    xe = dict(dist=6.5, high=False, side="bên trái")
    assert sp.update(10.0, xe, "motorcycle", act=None, toi=True)
    assert sp.update(10.3, dict(xe, dist=6.0), "motorcycle", act="dung_lai", toi=True) and sp.ngan
    assert sp.update(11.0, dict(xe, dist=2.0), "motorcycle", act="dung_lai", toi=True) is None


def test_nguoi_dang_toi_khong_dung_kiem_tra_xe_do():
    # IMG_7354 45 s (29/10): người đi tới thật bị mất "Người, bên trái, đang tới" vì tốc độ nửa giây gần nhất dao động
    # quanh ngưỡng (đi bộ ~2,3 m/s tương đối). Kiểm tra ghép nhầm/độ sâu nhảy là cho XE đỗ; người không bao giờ "đỗ".
    from ne_vat_can_do_sau import dang_toi
    assert dang_toi(dict(name="person", v=2.3, v_gan=0.9, to_ra=False))
    assert not dang_toi(dict(name="motorcycle", v=2.3, v_gan=0.9, to_ra=True))


def test_theo_vach_khong_do_duoc_van_bao_vong():
    # đang theo vạch (thấy vạch trong 5 s) mà mất lần đo vì xe che: chưa có bằng chứng đã rời vạch -> vẫn bảo vòng
    bo = BoVatChan()
    assert bo.buoc(10.0, False, vat(2.8), None, False, theo_vach=True) == XE
    assert BoVatChan().buoc(10.0, False, vat(2.8), None, False) == ""


def test_dang_dung_tren_vach_thi_khong_o_long_duong():
    # IMG_7354 65,6 s (29/10, lượt cuối): đo được rõ người đứng trên vạch mà phần nhận lòng đường báo "Đang ở lòng đường,
    # chú ý xe". Vạch dẫn đường lát trên vỉa hè: đứng trên vạch thì không ở lòng đường. Bước xuống đường khi đi vòng thì
    # không còn trên vạch -> vẫn báo.
    from ne_vat_can_do_sau import o_long_duong
    assert o_long_duong(True, tren_vach=False)
    assert not o_long_duong(True, tren_vach=True)
    assert not o_long_duong(False, tren_vach=False)


def test_vua_bao_vat_thi_lenh_sau_chi_noi_lenh():
    # IMG_7355 137,2/138,5 s (29/10): "Xe máy, phía trước" rồi 1,3 s sau "Dừng lại. Xe máy" về cùng chiếc xe -> lặp.
    # Cùng vật vừa nói < 3 s mà giờ có lệnh: chỉ nói lệnh.
    from ne_vat_can_do_sau import K
    sp = K.Speaker(im_xa=True, noi_khoang_cach=False)
    dung = []
    sp.clip = lambda k: (dung.append(k), np.zeros(10, np.int16))[1]
    xe = dict(dist=2.8, high=False, side="phía trước")
    assert sp.update(10.0, xe, "motorcycle") and not sp.ngan
    dung.clear()
    assert sp.update(11.3, dict(xe, dist=2.4), "motorcycle", act="dung_lai") and sp.ngan
    assert dung == ["dung_lai"]
