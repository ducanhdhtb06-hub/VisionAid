"""Test module dẫn theo gạch dẫn đường (dan_theo_vach.py).

Mask tổng hợp được vẽ bằng phép chiếu camera lỗ kim thật (camera cao h, chúc xuống pitch),
nên kiểm được đúng khẳng định của thiết kế: đo lệch/góc KHÔNG cần biết h hay pitch.
Chạy: ../.venv/bin/python -m pytest tests/test_dan_theo_vach.py -q  (đứng ở OpenAIglasses_for_Navigation/)
"""
import numpy as np
import cv2
import pytest

import dan_theo_vach as V

H, WI = 360, 640


def tieu_cu(hfov=70.0, w=WI):
    return (w / 2) / np.tan(np.radians(hfov / 2))


def chieu(X, Zg, h, pitch_deg, f, H=H, W=WI):
    """Điểm mặt đất (X sang phải, Zg về phía trước, mét) -> pixel (u, v)."""
    th = np.radians(pitch_deg)
    zc = Zg * np.cos(th) + h * np.sin(th)
    yc = h * np.cos(th) - Zg * np.sin(th)
    return W / 2 + f * X / zc, H / 2 + f * yc / zc


def ve_vach(d_m, psi_deg, W_m=0.4, h_cam=1.5, pitch_deg=10, hfov=70, z_min=0.3, z_max=12.0, che=None):
    """Mask (H x WI, uint8 0/1) của dải gạch thẳng: tâm cách người d_m (dương = bên phải),
    hướng lệch psi_deg so với hướng camera (dương = dải chạy chếch sang phải)."""
    f = tieu_cu(hfov)
    psi = np.radians(psi_deg)
    z = np.geomspace(z_min, z_max, 300)
    tam = d_m + z * np.tan(psi)
    nua = W_m / 2 / np.cos(psi)
    ut, vt = chieu(tam - nua, z, h_cam, pitch_deg, f)
    up, vp = chieu(tam + nua, z, h_cam, pitch_deg, f)
    poly = np.concatenate([np.stack([ut, vt], 1), np.stack([up, vp], 1)[::-1]])
    m = np.zeros((H, WI), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    if che is not None:
        x0, y0, x1, y1 = che
        m[y0:y1, x0:x1] = 0
    return m


# ---------- Task 1: đo lệch và góc ----------

def do_(m, pitch_deg=10):
    """do_vach với chân trời thật của camera giả lập (module tự học chân trời khi chạy - xem DanTheoVach)."""
    return V.do_vach(m, y_chan_troi=H / 2 - tieu_cu() * np.tan(np.radians(pitch_deg)))


def test_tam_vach_thang_truoc_mat():
    do = do_(ve_vach(0, 0))
    assert do.tin_cay
    assert do.lech == pytest.approx(0, abs=0.1)
    assert do.goc == pytest.approx(0, abs=1.5)


@pytest.mark.parametrize("d", [-0.4, 0.3])
@pytest.mark.parametrize("psi", [-12, 0, 12])
def test_lech_khong_phu_thuoc_goc(d, psi):
    # Giới hạn ghi trong thiết kế: so với khoảng cách vuông góc d·cos(psi)/W, camera chúc theta làm lệch thêm
    # tối đa h·tan(theta)·tan|psi|/W (~0,14W ở theta=10°, psi=12°, h=1,5 m, W=0,4 m)
    do = do_(ve_vach(d, psi))
    assert do.tin_cay
    bien = 1.5 * np.tan(np.radians(10)) * np.tan(np.radians(abs(psi))) / 0.4
    assert abs(do.lech - d * np.cos(np.radians(psi)) / 0.4) <= bien + 0.05
    assert do.goc == pytest.approx(psi, abs=1.5)


@pytest.mark.parametrize("pitch", [5, 10, 20])
@pytest.mark.parametrize("h", [1.3, 1.6])
def test_khong_can_hieu_chinh(pitch, h):
    do = do_(ve_vach(0.3, 0, h_cam=h, pitch_deg=pitch), pitch)
    assert do.tin_cay
    assert do.lech == pytest.approx(0.75, abs=0.05)
    do = do_(ve_vach(0.0, 8, h_cam=h, pitch_deg=pitch), pitch)
    assert do.goc == pytest.approx(8, abs=1.5)


def test_dau_lech():
    assert do_(ve_vach(0.4, 0)).lech > 0      # vạch bên phải người
    assert do_(ve_vach(-0.4, 0)).lech < 0
    assert do_(ve_vach(0, 10)).goc > 0        # vạch chạy chếch sang phải
    assert do_(ve_vach(0, -10)).goc < 0


def test_mask_rong_tra_none():
    assert V.do_vach(np.zeros((H, WI), np.uint8)) is None
    assert V.do_vach(None) is None


def test_mask_255_va_bool():
    m = ve_vach(0.25, 6)
    a = V.do_vach(m).lech
    assert V.do_vach(m * 255).lech == pytest.approx(a, abs=1e-6)
    assert V.do_vach(m.astype(bool)).lech == pytest.approx(a, abs=1e-6)


# ---------- Task 2: cổng "không chắc" và đường chân trời ----------

def _khong_chac(do, tu_khoa):
    assert do is None or (do.tin_cay is False and tu_khoa in do.ly_do), do


def test_ngan_qua():
    _khong_chac(V.do_vach(ve_vach(0, 0, z_max=2.8)), "ngắn")


def test_mot_mep_bi_che():
    # vật đè lên nửa phải dải gạch trên ~40% đoạn gần
    _khong_chac(V.do_vach(ve_vach(0, 0, che=(330, 240, 420, 302))), "che")


def test_nga_ba():
    m = ve_vach(0, 0)
    m[int(0.70 * H):int(0.76 * H), 320:560] = 1      # nhánh rẽ ngang sang phải
    _khong_chac(V.do_vach(m), "nhánh")


def test_mep_song_song():
    m = np.zeros((H, WI), np.uint8)
    m[:, 300:340] = 1
    _khong_chac(V.do_vach(m), "điểm tụ")


def test_chan_troi_lech():
    f = tieu_cu()
    y_h = H / 2 - f * np.tan(np.radians(10))           # chân trời thật khi chúc 10°
    assert V.do_vach(ve_vach(0, 0), y_chan_troi=y_h).tin_cay
    _khong_chac(V.do_vach(ve_vach(0, 0), y_chan_troi=y_h + f * np.tan(np.radians(10))), "chân trời")


def test_moi_hang_cham_mep_anh():
    # đứng ngay trên vạch rất rộng / ống kính hẹp: hàng nào cũng chạm mép ảnh -> không biết mép thật
    m = np.zeros((H, WI), np.uint8)
    m[150:, :] = 1
    _khong_chac(V.do_vach(m), "rộng hơn khung")      # vẫn thấy vạch, chỉ không đo được (không tính là mất)


def test_bo_chan_troi():
    b = V.BoChanTroi(cua_so_s=5.0, toi_thieu=10)
    for i in range(9):
        b.cap_nhat(100 + (i % 3) - 1, t=i * 0.1)
    assert b.gia_tri() is None
    b.cap_nhat(100, t=1.0)
    b.cap_nhat(101, t=1.1)
    assert b.gia_tri() == pytest.approx(100, abs=1)
    for i in range(11):                                  # mẫu mới phải trải >= 1 s
        b.cap_nhat(200, t=10 + i * 0.1)                  # mẫu cũ (> 5 s) bị bỏ
    assert b.gia_tri() == pytest.approx(200)


# ---------- Task 3: luật nói ----------

def dv(lech, goc, tin=True):
    return V.DoVach(lech, goc, 320.0, 100.0, tin, "" if tin else "giả", (0.0, 0.0), (0.0, 0.0))


def chay(doan, fps=10, t0=0.0, luat=None):
    """doan: list (số giây, DoVach|None). Trả (luat, list (t, câu) không rỗng)."""
    luat = luat or V.LuatNoi()
    ra, t = [], t0
    for giay, do in doan:
        for _ in range(int(round(giay * fps))):
            cau = luat.buoc(do, t)
            if cau:
                ra.append((round(t, 2), cau))
            t += 1.0 / fps
    return luat, ra


def test_dung_thi_im():
    assert chay([(10, dv(0.1, 2))])[1] == []


def test_lech_phai_sau_1s():
    _, ra = chay([(10, dv(1.2, 0))])
    assert ra[0][1] == "Dịch sang phải"
    assert 1.0 <= ra[0][0] <= 1.3
    lan = [t for t, c in ra if c == "Dịch sang phải"]
    assert all(b - a >= 4.0 - 1e-9 for a, b in zip(lan, lan[1:]))
    assert chay([(3, dv(-1.2, 0))])[1][0][1] == "Dịch sang trái"


def test_dang_tu_quay_ve_thi_im():
    assert chay([(10, dv(1.2, -8))])[1] == []


def test_cheo():
    # cơ chế bật/nhả của luật góc; lệch hằng (không trôi) nên chạy ở chế độ "cu" (chế độ "troi" coi là quay đầu)
    cu = lambda: V.LuatNoi(che_do_re="cu")
    _, ra = chay([(3, dv(0.2, 20))], luat=cu())
    assert ra[0][1] == "Rẽ phải một chút" and ra[0][0] >= 1.5
    luat, _ = chay([(2, dv(0.2, 20))], luat=cu())
    _, ra = chay([(5, dv(0.2, 12))], t0=2.0, luat=luat)          # 12° > 8°: chưa nhả, vẫn nhắc sau 4 s
    assert [c for _, c in ra] == ["Rẽ phải một chút"]
    _, ra = chay([(1, dv(0.2, 5)), (1.2, dv(0.2, 12))], t0=7.0, luat=luat)   # 5° < 8°: nhả, 12° không bật lại
    assert ra == []
    assert chay([(3, dv(0.2, -20))], luat=cu())[1][0][1] == "Rẽ trái một chút"


def test_dich_truoc_re():
    _, ra = chay([(3, dv(1.2, 20))])
    assert [c for _, c in ra] == ["Dịch sang phải"]


def test_khung_khong_chac_khong_tinh():
    doan = []
    for _ in range(15):
        doan += [(0.1, dv(1.2, 0)), (0.1, dv(0.0, 0, tin=False))]
    _, ra = chay(doan)
    assert ra and ra[0][1] == "Dịch sang phải"


def test_mat_va_tim_lai():
    _, ra = chay([(2, dv(0.1, 0)), (3.5, None), (1.2, dv(0.1, 0))])
    cau = [c for _, c in ra]
    assert cau == ["Mất đường đi, đang tìm lại.", "Đã quay lại đường dẫn."]
    assert 4.85 <= ra[0][0] <= 5.2          # khung tin cậy cuối ở t=1,9 + 3 s


def test_nhay_coc_thoi_gian():
    luat, ra = chay([(0.5, dv(1.2, 0))])
    assert ra == []
    assert luat.buoc(dv(1.2, 0), 5.5) == ""


# ---------- Sửa sau đánh giá video dev (2026-10-25) ----------

def _dv_ly_do(ly_do):
    return V.DoVach(float("nan"), float("nan"), float("nan"), float("nan"), False, ly_do, (0.0, 0.0), (0.0, 0.0))


def test_khong_chac_nhung_van_thay_vach_thi_khong_bao_mat():
    # vạch vẫn thấy nhưng bị che / ngã ba: im lặng, KHÔNG nói "mất đường đi"
    _, ra = chay([(2, dv(0.1, 0)), (6, _dv_ly_do("mép bị che"))])
    assert ra == []


def test_doan_vach_ngan_tinh_la_mat():
    _, ra = chay([(2, dv(0.1, 0)), (3.5, _dv_ly_do("đoạn vạch ngắn"))])
    assert [c for _, c in ra] == ["Mất đường đi, đang tìm lại."]


def test_bo_chan_troi_het_han_khi_khong_cap_nhat():
    b = V.BoChanTroi(cua_so_s=5.0, toi_thieu=10)
    for i in range(12):
        b.cap_nhat(100, t=i * 0.1)
    assert b.gia_tri(t=1.2) == pytest.approx(100)
    assert b.gia_tri(t=7.0) is None             # 5 s không có mẫu mới: không dùng chân trời cũ


def test_chan_troi_theo_kip_khi_doi_goc_chuc():
    # người cúi đầu thêm 10°: chân trời dịch ~86 px (> 6°); sau vài giây phải đo tin cậy lại
    dv_ = V.DanTheoVach()
    t = 0.0
    for _ in range(30):
        dv_.buoc(ve_vach(0, 0, pitch_deg=10), t)
        t += 0.1
    tin = []
    for _ in range(60):
        _, do = dv_.buoc(ve_vach(0, 0, pitch_deg=20), t)
        tin.append(do.tin_cay)
        t += 0.1
    assert all(tin[-20:])                        # 4–6 s sau khi đổi: tin cậy lại


def test_camera_ngang_hoi_ngang_len_van_hoc_duoc_chan_troi():
    # camera kính gắn hơi ngẩng (-4°): chân trời nằm DƯỚI giữa ảnh; không được im lặng mãi
    dv_ = V.DanTheoVach()
    tin = []
    for i in range(40):
        _, do = dv_.buoc(ve_vach(0, 0, pitch_deg=-4, z_max=25), i * 0.1)
        tin.append(do is not None and do.tin_cay)
    assert all(tin[-10:])


# ---------- GĐ2a: vạch bị chắn ----------

def _vat(x0, y0, x1, y1, dist=3.0):
    m = np.zeros((H, WI), bool)
    m[y0:y1, x0:x1] = True
    return dict(near=m, mask=m, dist=dist, name="motorcycle")


def test_hanh_lang_phu_dung_dai_gach():
    m = ve_vach(0.2, 5)
    do = do_(m)
    hl = V.hanh_lang(do, H, WI)
    vung = slice(int(do.y_vp) + 20, H)                     # bỏ sát điểm tụ (vạch chỉ còn vài px)
    trung = (hl[vung] & (m[vung] > 0)).sum() / max(1, (m[vung] > 0).sum())
    thua = (hl[vung] & (m[vung] == 0)).sum() / max(1, hl[vung].sum())
    assert trung > 0.9 and thua < 0.15


def test_vat_tren_vach():
    do = do_(ve_vach(0, 0))
    # tâm vạch ở x=320; vật đứng giữa vạch, chân ở hàng ~300
    assert len(V.vat_tren_vach([_vat(300, 240, 380, 300)], do, H, WI)) == 1
    # vật bên cạnh vạch (bên phải hẳn)
    assert V.vat_tren_vach([_vat(420, 240, 520, 300)], do, H, WI) == []
    # trên vạch nhưng còn xa > 5 m
    assert V.vat_tren_vach([_vat(300, 240, 380, 300, dist=7.0)], do, H, WI) == []
    # chỉ chạm mép vạch (< 30% bề rộng)
    x_phai = int(np.polyval(do.mep_phai, 299))
    assert V.vat_tren_vach([_vat(x_phai - 3, 240, x_phai + 60, 300)], do, H, WI) == []


def test_co_vat_chan_thi_khong_keo_ve_vach():
    luat = V.LuatNoi()
    t, ra = 0.0, []
    for _ in range(30):                                     # 3 s đang vòng qua vật (lệch phải khỏi vạch)
        c = luat.buoc(dv(1.2, 0), t, co_vat_chan=True)
        ra += [c] if c else []
        t += 0.1
    for _ in range(30):                                     # vật hết thấy: vẫn im thêm 4 s
        c = luat.buoc(dv(1.2, 0), t)
        ra += [c] if c else []
        t += 0.1
    assert ra == []
    for _ in range(20):                                     # sau 4 s: được nhắc lại như thường
        c = luat.buoc(dv(1.2, 0), t)
        ra += [c] if c else []
        t += 0.1
    assert ra == ["Dịch sang phải"]


def test_co_vat_chan_thi_khong_bao_mat_vach():
    luat = V.LuatNoi()
    _, ra = chay([(2, dv(0.1, 0))], luat=luat)
    t = 2.0
    for _ in range(50):                                     # vật che kín vạch 5 s
        assert luat.buoc(None, t, co_vat_chan=True) == ""
        t += 0.1


def test_dan_theo_vach_dung_lan_do_tin_cay_truoc_khi_vat_che_mep():
    dv_ = V.DanTheoVach()
    t = 0.0
    for _ in range(20):
        dv_.buoc(ve_vach(0, 0), t)
        t += 0.1
    assert dv_.dang_theo_vach(t)
    che = ve_vach(0, 0, che=(330, 240, 420, 302))           # xe đè nửa phải vạch -> lần đo hiện tại không chắc
    _, do = dv_.buoc(che, t, vat=[_vat(310, 230, 420, 300)])
    assert not do.tin_cay
    assert len(dv_.vat_chan) == 1


# ---------- Sửa theo rà soát độc lập (2026-10-25) ----------

def test_dao_dau_nha_lenh_lech():
    # [Nghiêm trọng #1] bước quá tay: vạch chuyển hẳn sang bên kia -> không được lặp lệnh cũ (sai chiều)
    _, ra = chay([(2, dv(1.2, 0)), (8, dv(-0.42, 0))], fps=3)
    assert ra and ra[0][1] == "Dịch sang phải"
    assert [c for t, c in ra if t >= 2.0] == []


def test_dao_dau_nha_lenh_goc():
    _, ra = chay([(2.5, dv(0.2, 20)), (8, dv(0.2, -12))], fps=3, luat=V.LuatNoi(che_do_re="cu"))
    assert ra and ra[0][1] == "Rẽ phải một chút"
    assert [c for t, c in ra if t >= 2.5] == []


def test_qua_quang_khong_chac_phai_xet_lai():
    # sau quãng không chắc dài, lệnh cũ không được giữ nguyên nhờ "trễ"
    _, ra = chay([(2, dv(1.2, 0)), (1.5, dv(0, 0, tin=False)), (6, dv(0.42, 0))])
    assert [c for t, c in ra if t >= 2.0] == []


def test_on_dinh_can_khung_tin_cay_phu_du_thoi_gian():
    # [Quan trọng #2] 0,4 s số đo giữa quãng không chắc (quay đầu, ảnh nhòe) không đủ để bảo rẽ
    _, ra = chay([(3, dv(0.2, 2)), (1.3, dv(0, 0, tin=False)), (0.4, dv(0.2, 20)), (2, dv(0, 0, tin=False))])
    assert ra == []


def test_tim_lai_can_du_thoi_gian_tin_cay():
    _, ra = chay([(2, dv(0.1, 0)), (3.5, None), (0.3, dv(0.1, 0)), (2, None)])
    assert [c for _, c in ra] == ["Mất đường đi, đang tìm lại."]


def test_fps_thap_va_rung_khung_van_noi_duoc():
    # [Quan trọng #3] ESP32 ~2,5 khung/giây, khoảng cách khung 0,2–0,6 s, một nửa khung không thấy vạch
    dv_ = V.DanTheoVach()
    rng = np.random.default_rng(0)
    m, t, ra = ve_vach(0.5, 0), 0.0, []
    for i in range(60):
        cau, _ = dv_.buoc(m if i % 2 == 0 else None, t)
        if cau:
            ra.append((t, cau))
        t += rng.uniform(0.2, 0.6)
    assert ra and ra[0][1] == "Dịch sang phải" and ra[0][0] < 10


def test_nhac_lai_gian_dan():
    # [Quan trọng #7] đứng/đi cạnh vạch lâu: 4 -> 8 -> 16 s, không nhắc 15 lần/phút
    _, ra = chay([(40, dv(1.2, 0))])
    ts = [t for t, _ in ra]
    assert len(ts) == 4
    assert ts[1] - ts[0] >= 4 - 1e-6 and ts[2] - ts[1] >= 8 - 1e-6 and ts[3] - ts[2] >= 16 - 1e-6


def test_dot_lech_moi_bat_dau_lai_4s():
    _, ra = chay([(6, dv(1.2, 0)), (2, dv(0, 0)), (6, dv(1.2, 0))])
    ts = [t for t, _ in ra]
    assert len(ts) == 4 and ts[3] - ts[2] == pytest.approx(4.0, abs=0.15)


def test_huy_cau_bi_nuot_thi_noi_lai():
    # [Quan trọng #4] process_frame chặn câu (đang đọc câu khác) -> luật nói phải thử lại, không coi là đã nói
    luat, ra = chay([(1.5, dv(1.2, 0))])
    assert ra[0][1] == "Dịch sang phải"
    luat.huy("Dịch sang phải")
    assert luat.buoc(dv(1.2, 0), 1.5) == "Dịch sang phải"
    luat, _ = chay([(2, dv(0.1, 0)), (3.2, None)])
    luat.huy("Mất đường đi, đang tìm lại.")
    assert luat.buoc(None, 5.3) == "Mất đường đi, đang tìm lại."


# ---------- GĐ2b: hết vạch ----------

def _chay_dv(masks, fps=10, vat=None, bao_het=True):
    dv_, t, ra = V.DanTheoVach(bao_het=bao_het), 0.0, []
    for m in masks:
        cau, _ = dv_.buoc(m, t, vat=vat)
        if cau:
            ra.append((round(t, 2), cau))
        t += 1.0 / fps
    return dv_, ra


def test_do_vach_biet_dau_xa():
    do = do_(ve_vach(0, 0, z_max=4.0))
    assert do.y_dinh > 250 and not do.dinh_cham_mep          # đầu vạch ~4 m nằm trong ảnh


def test_het_vach_trong_5m():
    # 2 s vạch dài để học chân trời, rồi tới đoạn vạch chỉ còn 4 m
    # người bước 0,8 m/s tới đầu vạch (đầu vạch phải tiến lại gần - xem test_bien_mo_di_theo_nguoi_khong_phai_het)
    _, ra = _chay_dv([ve_vach(0, 0)] * 20 + [ve_vach(0, 0, z_max=5.5 - 0.08 * i) for i in range(30)])
    assert [c for _, c in ra] == ["Sắp hết vạch dẫn đường."]
    assert 2.6 <= ra[0][0] <= 4.6


def test_vach_dai_khong_bao_het():
    assert _chay_dv([ve_vach(0, 0)] * 60)[1] == []


def test_vat_che_dau_vach_khong_phai_het():
    m = np.zeros((H, WI), bool)
    m[230:275, 290:360] = True                               # vật đứng ngay đầu vạch (4 m)
    vat = [dict(near=m, mask=m, dist=4.0, name="motorcycle")]
    _, ra = _chay_dv([ve_vach(0, 0)] * 20 + [ve_vach(0, 0, z_max=5.5 - 0.08 * i) for i in range(30)], vat=vat)
    assert all(c != "Sắp hết vạch dẫn đường." for _, c in ra)


def test_dau_vach_cham_mep_anh_khong_phai_het():
    # đi chéo nhiều: vạch chạy ra khỏi mép ảnh, không phải hết
    _, ra = _chay_dv([ve_vach(0, 0)] * 20 + [ve_vach(0.3, 40)] * 20)
    assert all(c != "Sắp hết vạch dẫn đường." for _, c in ra)


def test_da_bao_het_thi_khong_bao_mat():
    _, ra = _chay_dv([ve_vach(0, 0)] * 20 + [ve_vach(0, 0, z_max=5.5 - 0.08 * i) for i in range(30)] + [None] * 50)
    assert [c for _, c in ra] == ["Sắp hết vạch dẫn đường."]


def test_ranh_cat_ngang_vach_khong_phai_het():
    # rãnh thoát nước cắt ngang làm mask tách 2 mảnh; mảnh xa vẫn nằm trên đường kéo dài -> vạch còn tiếp
    m = ve_vach(0, 0)
    m[262:268, :] = 0
    _, ra = _chay_dv([ve_vach(0, 0)] * 20 + [m] * 30)
    assert all(c != "Sắp hết vạch dẫn đường." for _, c in ra)


def test_bien_mo_di_theo_nguoi_khong_phai_het():
    # vạch mờ phía xa: ranh giới mask đứng yên trong ảnh khi người bước tới (đầu vạch thật thì tiến lại gần)
    _, ra = _chay_dv([ve_vach(0, 0)] * 20 + [ve_vach(0, 0, z_max=4.0)] * 40)
    assert all(c != "Sắp hết vạch dẫn đường." for _, c in ra)


def test_het_vach_that_tien_lai_gan():
    # người đi 0,8 m/s về phía đầu vạch thật: đầu vạch 5,5 m -> 3 m
    seq = [ve_vach(0, 0)] * 20 + [ve_vach(0, 0, z_max=5.5 - 0.08 * i) for i in range(30)]
    _, ra = _chay_dv(seq)
    assert [c for _, c in ra] == ["Sắp hết vạch dẫn đường."]


def test_het_vach_tat_mac_dinh(monkeypatch):
    monkeypatch.delenv("AIGLASS_HET_VACH", raising=False)
    assert V.DanTheoVach().bao_het is False
    seq = [ve_vach(0, 0)] * 20 + [ve_vach(0, 0, z_max=5.5 - 0.08 * i) for i in range(30)]
    _, ra = _chay_dv(seq, bao_het=None)
    assert ra == []


# ---------- Lỗi nhỏ của rà soát (làm theo yêu cầu "hết công suất") ----------

def test_chon_dai_gan_chan_khong_phai_dai_to_nhat():
    m = ve_vach(0, 0)
    m[:60, 0:400] = 1                                       # đốm nhận nhầm rất to ở xa (biển hiệu, tường…)
    do = do_(m)
    assert do is not None and do.tin_cay and do.lech == pytest.approx(0, abs=0.1)


@pytest.mark.parametrize("pitch", [10, 30])
def test_goc_tinh_ca_do_chuc(pitch):
    do = do_(ve_vach(0, 15, pitch_deg=pitch), pitch)
    assert do.goc == pytest.approx(15, abs=1.0)


def test_cong_chan_troi_tinh_bang_goc():
    # chúc 30°: lệch 5° so với chân trời vẫn tin cậy, lệch 7° thì không (cổng ±6° theo GÓC, không theo pixel)
    f = tieu_cu()
    y_h = H / 2 - f * np.tan(np.radians(30))
    m = ve_vach(0, 0, pitch_deg=30)
    goc_vp = np.degrees(np.arctan((H / 2 - V.do_vach(m, y_chan_troi=y_h).y_vp) / f))
    y5 = H / 2 - f * np.tan(np.radians(goc_vp - 5))
    y7 = H / 2 - f * np.tan(np.radians(goc_vp - 7))
    assert V.do_vach(m, y_chan_troi=y5).tin_cay
    assert "chân trời" in V.do_vach(m, y_chan_troi=y7).ly_do


def test_mask_nhieu_kenh_vach_o_kenh_bat_ky():
    m = ve_vach(0, 0)
    m3 = np.zeros((H, WI, 3), np.uint8)
    m3[..., 2] = m
    assert V.do_vach(m3) is not None


def test_dung_tren_tam_gach_rong_khong_bao_mat():
    luat_dv = V.DanTheoVach()
    t = 0.0
    for _ in range(25):
        luat_dv.buoc(ve_vach(0, 0), t)
        t += 0.1
    rong = np.zeros((H, WI), np.uint8)
    rong[150:, :] = 1                                       # tấm gạch cảnh báo phủ hết bề ngang
    for _ in range(50):
        cau, do = luat_dv.buoc(rong, t)
        assert cau == "" and "ngắn" not in do.ly_do
        t += 0.1


# ---------- GĐ3: vạch rẽ chữ L / ngã rẽ ----------

def ve_vach_re(huong=1, z_re=4.0, thang_tiep=False, W_m=0.4, h_cam=1.5, pitch_deg=10, hfov=70):
    """Dải gạch thẳng trước mặt tới z_re (m) rồi rẽ vuông góc sang phải (huong=1) / trái (-1) dài 3 m;
    thang_tiep=True: dải chính vẫn đi thẳng tiếp (ngã rẽ chữ T)."""
    m = ve_vach(0, 0, W_m=W_m, h_cam=h_cam, pitch_deg=pitch_deg, hfov=hfov,
                z_max=12.0 if thang_tiep else z_re + W_m)
    f = tieu_cu(hfov)
    xs = np.linspace(0, 3.0, 100) * huong
    u1, v1 = chieu(xs, np.full_like(xs, z_re), h_cam, pitch_deg, f)
    u2, v2 = chieu(xs, np.full_like(xs, z_re + W_m), h_cam, pitch_deg, f)
    poly = np.concatenate([np.stack([u1, v1], 1), np.stack([u2, v2], 1)[::-1]])
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m


def _di_toi(ham, z0=6.0, toc=0.08, n=45, **kw):
    """Người đi 0,8 m/s về phía chỗ rẽ: 2 s vạch thẳng (học chân trời) rồi chỗ rẽ từ z0 m tiến lại gần."""
    return [ve_vach(0, 0)] * 20 + [ham(z_re=z0 - toc * i, **kw) for i in range(n)]


@pytest.mark.parametrize("huong,cau", [(1, "Vạch rẽ phải phía trước."), (-1, "Vạch rẽ trái phía trước.")])
def test_vach_re_chu_L(huong, cau):
    _, ra = _chay_dv(_di_toi(ve_vach_re, huong=huong))
    assert [c for _, c in ra] == [cau]


def test_nga_re_chu_T():
    _, ra = _chay_dv(_di_toi(ve_vach_re, huong=1, thang_tiep=True))
    assert [c for _, c in ra] == ["Phía trước có ngã rẽ vạch."]


def test_vach_thang_khong_bao_re():
    _, ra = _chay_dv([ve_vach(0, 0)] * 60 + [ve_vach(0.2, 6)] * 30)
    assert ra == []


def test_re_dung_yen_khong_bao():
    # chỗ rẽ không tiến lại gần (người đứng yên / ảo giác ảnh): không báo
    _, ra = _chay_dv([ve_vach(0, 0)] * 20 + [ve_vach_re(huong=1, z_re=3.5)] * 30)
    assert ra == []


def test_vach_re_khi_do_chung_khong_tin_cay():
    # chỗ rẽ thật thường lộn xộn (mép phía xa bị che, điểm tụ nhảy): phép đo chung không tin cậy, nhưng đoạn vạch
    # gần chân vẫn thẳng -> vẫn phải thấy chỗ rẽ
    def ham(z_re, **kw):
        m = ve_vach_re(huong=-1, z_re=z_re)
        m[300:330, 335:420] = 0                              # vật che mép phải đoạn giữa -> đo chung "mép bị che"
        return m
    seq = [ve_vach(0, 0)] * 20 + [ham(z_re=6.0 - 0.08 * i) for i in range(45)]
    dv_, t, ra = V.DanTheoVach(), 0.0, []
    for m in seq:
        cau, _ = dv_.buoc(m, t)
        if cau:
            ra.append(cau)
        t += 0.1
    assert ra == ["Vạch rẽ trái phía trước."]


# ---------- Kế hoạch sau rà soát, Task 4: 3 cách xử lý "Rẽ" (cu / troi / tat) ----------
# Thân đi chéo thì vị trí so với vạch TRÔI dần (v·sinψ); chỉ quay đầu thì không trôi. "troi": chỉ nói Rẽ khi
# độ lệch đang trôi đúng chiều với góc chéo.

def _chay_ham(f_lech, f_goc, giay, che_do, fps=10):
    luat = V.LuatNoi(che_do_re=che_do)
    ra, n = [], int(giay * fps)
    for k in range(n):
        t = k / fps
        cau = luat.buoc(dv(f_lech(t), f_goc(t)), t)
        if cau:
            ra.append((round(t, 2), cau))
    return ra


def _co_re(ra):
    return [c for _, c in ra if c.startswith("Rẽ")]


def test_troi_quay_dau_khong_re():
    # đầu quay sang trái 20° (vạch trông chạy chếch sang phải), thân vẫn đi thẳng: lệch không đổi
    assert _co_re(_chay_ham(lambda t: 0.1, lambda t: 20.0, 4, "troi")) == []


def test_troi_di_cheo_that_co_re():
    # thân quay trái 18°: người trôi sang trái so với vạch -> vạch dịch dần sang phải (lệch tăng 0,15 bề rộng/giây)
    ra = _chay_ham(lambda t: 0.15 * t, lambda t: 18.0, 3, "troi")
    assert "Rẽ phải một chút" in _co_re(ra)


def test_tat_khong_bao_gio_re():
    assert _co_re(_chay_ham(lambda t: 0.15 * t, lambda t: 18.0, 3, "tat")) == []


def test_cu_giu_nguyen_hanh_vi():
    # cách cũ: góc chéo ổn định 1,5 s là nói, không xét trôi (kể cả khi chỉ quay đầu)
    assert "Rẽ phải một chút" in _co_re(_chay_ham(lambda t: 0.1, lambda t: 20.0, 4, "cu"))


def test_che_do_re_doc_tu_bien_moi_truong(monkeypatch):
    monkeypatch.setenv("AIGLASS_RE", "tat")
    assert V.LuatNoi().che_do_re == "tat"


# ---------- Kế hoạch sau rà soát, Task 6: báo "Đã thấy vạch" / "Chưa thấy vạch" ----------
# Người khiếm thị không phân biệt được "im vì đi đúng" với "im vì kính chưa thấy gì".
THAY, CHUA = "Đã thấy vạch dẫn đường.", "Chưa thấy vạch dẫn đường."


@pytest.fixture(autouse=False)
def bat_thong_bao(monkeypatch):
    monkeypatch.setenv("AIGLASS_THONG_BAO_VACH", "1")


def test_bao_thay_vach_mot_lan(bat_thong_bao):
    _, ra = chay([(2, dv(0.1, 0)), (4, None), (3, dv(0.1, 0))])
    cau = [c for _, c in ra]
    assert cau.count(THAY) == 1 and cau[0] == THAY
    assert cau.index("Mất đường đi, đang tìm lại.") < cau.index("Đã quay lại đường dẫn.")


def test_lenh_dich_luc_bat_vach_thay_cho_cau_thay(bat_thong_bao):
    # vừa bắt được vạch mà đã lệch: câu "Dịch" đủ cho biết kính thấy vạch, không nói thêm "Đã thấy"
    _, ra = chay([(5, dv(1.2, 0))])
    assert ra[0][1] == "Dịch sang phải" and THAY not in [c for _, c in ra]


def test_chua_thay_vach_sau_5s(bat_thong_bao):
    # thử iPhone 2026-10-26 (trong nhà, không có vạch): nhắc 4 -> 8 s xen giữa câu vật cản nghe dồn dập ->
    # nhắc thưa: 15 s rồi tối đa mỗi 30 s
    _, ra = chay([(60, None)])
    lan = [t for t, c in ra if c == CHUA]
    assert len(lan) == 3 and lan[0] == pytest.approx(5.0, abs=0.15)
    # mỗi lần có thể trễ 1 khung vì bước 0,1 s + làm tròn số thực
    assert lan[1] - lan[0] == pytest.approx(15.0, abs=0.15) and lan[2] - lan[1] == pytest.approx(30.0, abs=0.15)
    assert [c for _, c in ra if c != CHUA] == []


def test_huy_cau_thay_thi_noi_lai(bat_thong_bao):
    luat, ra = chay([(2, dv(0.1, 0))])
    assert [c for _, c in ra] == [THAY]
    luat.huy(THAY)
    _, ra2 = chay([(0.3, dv(0.1, 0))], t0=2.0, luat=luat)
    assert [c for _, c in ra2] == [THAY]


def test_tat_thong_bao_bang_bien_moi_truong():
    # conftest đặt AIGLASS_THONG_BAO_VACH=0
    _, ra = chay([(2, dv(0.1, 0)), (10, None)])
    assert THAY not in [c for _, c in ra]
    _, ra = chay([(25, None)])
    assert CHUA not in [c for _, c in ra]


# ---------- Kế hoạch sau rà soát, Task 5: ngưỡng "Dịch" theo mép vạch (0,5 bề rộng) ----------

def test_lech_0_6_thi_dich():
    # tâm người đã ra ngoài mép vạch (0,6 bề rộng > 0,5): phải nhắc
    _, ra = chay([(1.5, dv(0.6, 0))], fps=8)
    assert "Dịch sang phải" in [c for _, c in ra]


def test_lac_lu_vach_hep_khong_noi():
    # vạch hẹp 0,3 m: ngưỡng 0,5 bề rộng chỉ là 15 cm; lắc ngang ±8 cm khi bước (0,9 Hz) không được sinh câu nào
    dvach = V.DanTheoVach(hfov_deg=70.0)
    ra = []
    for k in range(200):
        t = k / 10
        cau, _ = dvach.buoc(ve_vach(0.08 * np.sin(2 * np.pi * 0.9 * t), 0, W_m=0.3, z_max=25), t)
        if cau:
            ra.append((t, cau))
    assert ra == []


def test_quay_dau_vach_hep_khong_dich():
    # vạch hẹp 0,3 m, đứng giữa vạch, camera chúc 10°, quay đầu sang phải 30° trong 3 s: sai số đo lệch
    # h·tanθ·sinψ/W ≈ 0,44 bề rộng (gần ngưỡng) nhưng cùng lúc góc chéo ngược dấu -> luật "đang tự quay về" phải chặn
    dvach = V.DanTheoVach(hfov_deg=70.0)
    ra = []
    for k in range(50):
        t = k / 10
        psi = -30.0 if t >= 2.0 else 0.0
        cau, _ = dvach.buoc(ve_vach(0.0, psi, W_m=0.3, z_max=25), t)
        if cau:
            ra.append((t, cau))
    assert [c for _, c in ra if c.startswith("Dịch")] == []


def test_che_do_re_mac_dinh_la_troi(monkeypatch):
    # Task 4: mặc định "troi" — giữ lệnh Rẽ khi đi chéo thật (kịch bản kính ảo 8/8 như "cu", "tat" chỉ 1/8)
    monkeypatch.delenv("AIGLASS_RE", raising=False)
    assert V.LuatNoi().che_do_re == "troi"
