"""Ghép dan_theo_vach vào BlindPathNavigator (workflow_blindpath.py).
Chạy: ../.venv/bin/python -m pytest tests/test_tich_hop_vach.py -q  (đứng ở OpenAIglasses_for_Navigation/)"""
import numpy as np
import pytest

from test_dan_theo_vach import ve_vach, H, WI


@pytest.fixture
def wb(monkeypatch):
    monkeypatch.setenv("AIGLASS_OBS_ENGINE", "cu")          # không nạp model độ sâu
    import workflow_blindpath
    return workflow_blindpath


def _nav(wb, monkeypatch, gio):
    monkeypatch.setattr(wb.time, "time", lambda: gio[0])
    return wb.BlindPathNavigator()


def _chay(nav, gio, mask, giay, fps=10):
    img = np.zeros((H, WI, 3), np.uint8)
    ra = []
    for _ in range(int(giay * fps)):
        ra.append(nav._handle_navigating(mask, img, [], H, WI, None))
        gio[0] += 1.0 / fps
    return ra


def test_mac_dinh_bat_va_vao_thang_navigating(wb, monkeypatch):
    nav = _nav(wb, monkeypatch, [0.0])
    assert nav.dan_vach is not None
    assert nav.current_state == "NAVIGATING"
    nav.reset()
    assert nav.current_state == "NAVIGATING"


def test_tat_bang_bien_moi_truong(wb, monkeypatch):
    monkeypatch.setenv("AIGLASS_DAN_VACH", "cu")
    nav = _nav(wb, monkeypatch, [0.0])
    assert nav.dan_vach is None
    assert nav.current_state == "ONBOARDING"


def test_navigating_lech_phai(wb, monkeypatch):
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    assert "Dịch sang phải" in _chay(nav, gio, ve_vach(0.5, 0), 3)    # khởi động: học chân trời ~1 s + ổn định ~1 s
    nav = _nav(wb, monkeypatch, gio)
    ra = _chay(nav, gio, ve_vach(0, 0), 3)
    assert set(ra) == {""}


def test_khong_vao_maneuvering_turn(wb, monkeypatch):
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    m = ve_vach(0, 0)
    m[int(0.30 * H):int(0.36 * H), 320:600] = 1              # vạch rẽ phải chữ L phía xa
    _chay(nav, gio, m, 3)
    assert nav.current_state == "NAVIGATING"


def test_giu_cau_vat_can_cua_module_do_sau(wb, monkeypatch):
    # process_frame đặt câu của module né vật cản (bản 28) TRƯỚC khi máy trạng thái chạy
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.pending_obstacle_voice = "Phía trước có vật cản, dừng lại."
    _chay(nav, gio, ve_vach(0, 0), 0.1)
    assert nav.pending_obstacle_voice == "Phía trước có vật cản, dừng lại."


def test_mat_vach_qua_process_frame(wb, monkeypatch):
    # yolo-seg không thấy vạch -> _detect_path_and_crosswalk trả None; vẫn phải báo mất vạch
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    da_noi = []
    monkeypatch.setattr(wb, "play_voice_text", lambda s: da_noi.append(s))
    hien_tai = {"m": ve_vach(0, 0)}
    monkeypatch.setattr(nav, "_detect_path_and_crosswalk", lambda img: (hien_tai["m"], None))
    img = np.zeros((H, WI, 3), np.uint8)
    for i in range(55):
        if i == 20:
            hien_tai["m"] = None
        nav.process_frame(img)
        gio[0] += 0.1
    assert "Mất đường đi, đang tìm lại." in da_noi


class _NeGia:
    """Thay module né vật cản độ sâu: ghi lại cờ được truyền vào, trả danh sách vật cố định."""
    def __init__(self, cand):
        self.cand, self.co = cand, []

    def step(self, image, t=None, dang_qua_duong=False, chi_bao=None):
        self.co.append(dang_qua_duong)
        return None


def test_dang_theo_vach_thi_ne_vat_can_chi_dung(wb, monkeypatch):
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = _NeGia([])
    monkeypatch.setattr(nav, "_detect_path_and_crosswalk", lambda img: (ve_vach(0, 0), None))
    img = np.zeros((H, WI, 3), np.uint8)
    for _ in range(10):
        nav.process_frame(img)
        gio[0] += 0.1
    assert nav.ne_vat_can.co[0] is False                     # khung đầu: chưa theo vạch
    assert nav.ne_vat_can.co[-1] is True                     # đang theo vạch: không bao giờ bảo "dịch"


def test_vat_tren_vach_den_duoc_module_vach(wb, monkeypatch):
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    m = np.zeros((H, WI), bool)
    m[240:300, 300:380] = True
    nav.ne_vat_can = _NeGia([dict(near=m, mask=m, dist=3.0, name="motorcycle")])
    monkeypatch.setattr(nav, "_detect_path_and_crosswalk", lambda img: (ve_vach(0, 0), None))
    img = np.zeros((H, WI, 3), np.uint8)
    for _ in range(20):                                     # chân trời cần ~1 s để học
        nav.process_frame(img)
        gio[0] += 0.1
    assert len(nav.dan_vach.vat_chan) == 1


def _chay_pf(nav, gio, masks, wb, monkeypatch, moi_khung=None):
    da_noi = []
    monkeypatch.setattr(wb, "play_voice_text", lambda s: da_noi.append((round(gio[0], 2), s)))
    hien = {"m": None}
    monkeypatch.setattr(nav, "_detect_path_and_crosswalk", lambda img: (hien["m"], None))
    img = np.zeros((H, WI, 3), np.uint8)
    kq = []
    for m in masks:
        hien["m"] = m
        if moi_khung:
            moi_khung(da_noi)
        kq.append(nav.process_frame(img))
        gio[0] += 0.1
    return da_noi, kq


class _NeNoi(_NeGia):
    def __init__(self):
        super().__init__([])
        self.noi = None

    def step(self, image, t=None, dang_qua_duong=False, chi_bao=None):
        c, self.noi = self.noi, None
        return c


def test_cau_vat_can_khong_bi_chan_boi_gioi_han_1_2s(wb, monkeypatch):
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = _NeNoi()
    da = {"ok": False}

    def moi_khung(da_noi):
        if not da["ok"] and any(c == "Dịch sang phải" for _, c in da_noi):
            nav.ne_vat_can.noi, da["ok"] = "Dừng lại. Xe máy, 3 mét", True
    da_noi, _ = _chay_pf(nav, gio, [ve_vach(0.5, 0)] * 30, wb, monkeypatch, moi_khung)
    assert "Dừng lại. Xe máy, 3 mét" in [c for _, c in da_noi]


def test_cau_dan_vach_bi_nuot_duoc_noi_lai_ngay(wb, monkeypatch):
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)

    def moi_khung(da_noi):
        if gio[0] < 1.6:
            nav.last_any_speech_time = gio[0]          # đang đọc câu khác suốt 1,6 s đầu
    da_noi, _ = _chay_pf(nav, gio, [ve_vach(0.5, 0)] * 40, wb, monkeypatch, moi_khung)
    lan = [t for t, c in da_noi if c == "Dịch sang phải"]
    assert lan and lan[0] < 3.0


def test_guidance_text_tra_ve_dung_cau_da_phat(wb, monkeypatch):
    # app_main phát lại res.guidance_text: phải đúng câu process_frame đã phát (không phát câu đã bị chặn)
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)

    def moi_khung(da_noi):
        if gio[0] < 1.6:
            nav.last_any_speech_time = gio[0]
    da_noi, kq = _chay_pf(nav, gio, [ve_vach(0.5, 0)] * 40, wb, monkeypatch, moi_khung)
    tra = [r.guidance_text for r in kq if r.guidance_text]
    assert tra == [c for _, c in da_noi]


def test_state_bao_onboarding_khi_chua_tren_vach(wb, monkeypatch):
    # [Quan trọng #5] orchestrator dùng state=="NAVIGATING" để nói "Đúng hướng" sau khi qua đường
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    _, kq = _chay_pf(nav, gio, [None] * 10, wb, monkeypatch)
    assert kq[-1].state_info["state"] == "ONBOARDING"
    _, kq = _chay_pf(nav, gio, [ve_vach(0, 0)] * 25, wb, monkeypatch)
    assert kq[-1].state_info["state"] == "NAVIGATING"
    _, kq = _chay_pf(nav, gio, [ve_vach(0.6, 0)] * 20, wb, monkeypatch)
    assert kq[-1].state_info["state"] == "ONBOARDING"


def test_khong_co_model_thi_khong_bịa_vach(wb, monkeypatch):
    # [Quan trọng #6] model lỗi/không có: không được trả dải giả (kính sẽ im như thể đang đi đúng)
    nav = _nav(wb, monkeypatch, [0.0])
    assert nav._detect_path_and_crosswalk(np.zeros((H, WI, 3), np.uint8))[0] is None


# ---------- Vật chắn trên vạch, giai đoạn 1 (spec 2026-10-27-vong-vat-can-tren-vach) ----------
from test_dan_theo_vach import chieu, tieu_cu  # noqa: E402

CAU_NE = "Dừng lại. Xe máy, 3 mét"


def geo_mat_dat(h=1.5, pitch_deg=10, hfov=70):
    """Hình học mặt đất (X, Z, cao) đúng phép chiếu của ve_vach (ngược của chieu)."""
    f, th = tieu_cu(hfov), np.radians(pitch_deg)
    v, u = np.mgrid[0:H, 0:WI].astype(float)
    a, b = (u - WI / 2) / f, (v - H / 2) / f
    mau = b * np.cos(th) + np.sin(th)
    Z = np.where(mau > 1e-6, h * (np.cos(th) - b * np.sin(th)) / np.maximum(mau, 1e-6), np.inf)
    zc = Z * np.cos(th) + h * np.sin(th)
    return dict(X=a * zc, Z=Z, height=np.zeros((H, WI)))


def dat_vat_3d(geo, x0, x1, z, cao=(0.2, 1.0)):
    """Vật đứng ở khoảng cách z, X trong [x0, x1]: vùng ảnh từ chân vật lên 40 hàng."""
    f = tieu_cu()
    u0, vb = chieu(x0, z, 1.5, 10, f)
    u1, _ = chieu(x1, z, 1.5, 10, f)
    m = np.zeros((H, WI), bool)
    m[int(vb) - 40:int(vb), int(u0):int(u1) + 1] = True
    n = m.sum()
    geo["X"][m], geo["Z"][m], geo["height"][m] = np.linspace(x0, x1, n), z, np.linspace(cao[0], cao[1], n)
    return dict(mask=m, near=m, dist=z, name="motorcycle", src="depth", xl=x0, xr=x1, side="phía trước", high=False)


class _NeVatChan:
    """Module vật cản giả có hình học 3D: nói CAU_NE nếu còn vật được chi_bao chấp nhận."""
    def __init__(self):
        self.cand, self.geo, self.yo, self.chi_bao, self.dang_ky = [], geo_mat_dat(), [], [], []

    def step(self, image, t=None, dang_qua_duong=False, chi_bao=None):
        self.chi_bao.append(chi_bao)
        con = [o for o in self.cand if chi_bao is None or chi_bao(o)]
        return CAU_NE if con else None

    @staticmethod
    def dung_yen(o):
        return not o.get("toi") and not o.get("chua_ro") and not o.get("lech")

    @staticmethod
    def co_the_dang_do(o):
        return not o.get("toi")

    def ten_vat(self, o):
        return "motorcycle", "xe máy"

    def ben_di_duoc(self, o):
        return ["trai", "phai"]

    def dang_ky_cau(self, cau, ten_en=None):
        self.dang_ky.append(cau)
        return cau


def _chay_vat(wb, monkeypatch, vat_fn, giay=6.0):
    """3 s đầu: chỉ có vạch (học chân trời, vào trạng thái theo vạch); sau đó đặt vật (vat_fn(geo) -> dict)."""
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeVatChan()
    def moi_khung(da_noi):
        if gio[0] >= 3.0 and not ne.cand:
            ne.cand = [vat_fn(ne.geo)]
    da_noi, _ = _chay_pf(nav, gio, [ve_vach(0, 0)] * int(giay * 10), wb, monkeypatch, moi_khung)
    return nav, ne, [(t, c) for t, c in da_noi if t >= 3.0]


def test_vat_canh_vach_im_khi_tren_vach(wb, monkeypatch):
    nav, ne, sau = _chay_vat(wb, monkeypatch, lambda g: dat_vat_3d(g, 1.0, 1.5, 2.8))
    assert CAU_NE not in [c for _, c in sau]                  # vật cạnh vạch: im
    assert not any("chắn vạch" in c for _, c in sau)


def test_vat_chan_bao_vong(wb, monkeypatch):
    nav, ne, sau = _chay_vat(wb, monkeypatch, lambda g: dat_vat_3d(g, -0.15, 0.15, 2.8))
    cau = [c for _, c in sau]
    assert "Xe máy chắn giữa vạch. Đi sang trái." in cau
    i = cau.index("Xe máy chắn giữa vạch. Đi sang trái.")
    assert sau[i][0] == pytest.approx(4.0, abs=0.15)           # thấy liên tục 1 s mới giao cho bộ điều khiển
    assert CAU_NE not in cau[i:]                              # từ đó module vật cản không nói trùng vật này
    assert nav.dan_vach.giu_im and nav.bo_vat_chan.dang_xu_ly


def test_tat_bang_bien_moi_truong_vat_chan(wb, monkeypatch):
    monkeypatch.setenv("AIGLASS_VAT_CHAN_VACH", "0")
    nav, ne, sau = _chay_vat(wb, monkeypatch, lambda g: dat_vat_3d(g, 1.0, 1.5, 2.8))
    assert nav.bo_vat_chan is None
    assert CAU_NE in [c for _, c in sau]                      # hành vi cũ: báo mọi vật trong lối đi


def test_vat_chan_thoang_qua_thi_im(wb, monkeypatch):
    # vật chắn đứng yên chỉ thấy 0,5 s (vết lạ): chưa đủ 1 s -> bộ điều khiển không nói; đợt sửa 28/10: xe đỗ trên vạch
    # chỉ bộ điều khiển được nói -> im hẳn (trước đây module vật cản báo chen vào -> 2 câu về cùng một xe)
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeVatChan()
    def moi_khung(da_noi):
        ne.cand = [dat_vat_3d(ne.geo, -0.15, 0.15, 2.8)] if 3.0 <= gio[0] < 3.5 else []
    da_noi, _ = _chay_pf(nav, gio, [ve_vach(0, 0)] * 60, wb, monkeypatch, moi_khung)
    cau = [c for t, c in da_noi if t >= 3.0]
    assert not any("chắn vạch" in c for c in cau)
    assert CAU_NE not in cau


def test_dang_di_vong_chi_bao_nguy_hiem(wb, monkeypatch):
    # người dùng (28/10): trên vạch chỉ quan tâm vật trong vùng vạch; lúc đang đi vòng (ngoài vạch) im vật khác,
    # trừ nguy hiểm luôn báo (xe lao tới, mép đường, cầu thang, hố, vật ngang đầu)
    gio = [10.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeVatChan()
    nav.bo_vat_chan.trang_thai = "da_bao"
    monkeypatch.setattr(nav.dan_vach, "da_tren_vach", lambda t: False)
    cb = nav._chi_bao_vat_chan(gio[0])
    vat = dat_vat_3d(ne.geo, 1.0, 1.5, 2.8)
    assert cb is not None and cb(vat) is False
    assert cb(dict(vat, toi=True)) is True
    assert cb(dict(vat, src="curb")) is True
    assert cb(dict(vat, high=True)) is True


def _cb(wb, monkeypatch, tren=True, theo=True, loai="chan", tt="binh_thuong", on_dinh=False):
    """Bộ lọc câu khi đang theo vạch, với phân loại vật / trạng thái giả."""
    import dan_theo_vach
    gio = [10.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeVatChan()
    nav.bo_vat_chan.trang_thai, nav.vat_chan_on_dinh = tt, on_dinh
    monkeypatch.setattr(nav.dan_vach, "da_tren_vach", lambda t: tren)
    monkeypatch.setattr(nav.dan_vach, "dang_theo_vach", lambda t, *a: theo)
    monkeypatch.setattr(dan_theo_vach, "phan_loai_vat", lambda *a, **k: loai)
    return nav._chi_bao_vat_chan(gio[0]), ne


def test_vat_tren_vach_con_xa_thi_im(wb, monkeypatch):
    # chạy IMG_7354 (28/10): "Xe máy, phía trước, 5 mét" rồi mới "Xe máy chắn vạch, vòng bên trái" ở 3 m -> thừa
    cb, ne = _cb(wb, monkeypatch, loai="chan")
    assert cb(dat_vat_3d(ne.geo, -0.15, 0.15, 4.6)) is False
    # gần mà bộ điều khiển chưa nhận (chưa đủ 1 s): xe đỗ trên vạch chỉ bộ điều khiển được nói (đợt sửa 28/10: câu
    # "Xe máy, phía trước" rồi ngay sau "…Đi sang trái" về cùng một xe)
    assert cb(dat_vat_3d(ne.geo, -0.15, 0.15, 2.5)) is False


def test_da_bao_vong_thi_im_vat_khong_chac(wb, monkeypatch):
    # chạy IMG_7354 (28/10): đã bảo vòng, tới gần xe che mép vạch -> "Dừng lại. Xe máy, 2 mét" ngược lời bảo vòng
    cb, ne = _cb(wb, monkeypatch, loai="khong_chac", tt="da_bao", on_dinh=True)
    assert cb(dat_vat_3d(ne.geo, -0.15, 0.15, 2.1)) is False
    assert cb(dict(dat_vat_3d(ne.geo, -0.15, 0.15, 2.1), toi=True)) is True


def test_mat_vach_chi_bao_vat_ngay_truoc_mat(wb, monkeypatch):
    # đang theo vạch nhưng lần đo vạch tạm mất (xe che): xe đỗ ngay trước mặt là xe chắn -> bộ điều khiển bảo vòng
    # (người dùng 29/10: "làm y hệt demo trong mọi trường hợp"; trước đây chỉ "Xe máy, phía trước"); vật bên cạnh / xa im
    cb, ne = _cb(wb, monkeypatch, tren=False, loai="khong_chac")
    gan = dat_vat_3d(ne.geo, -0.15, 0.15, 2.2)
    assert cb(gan) is False
    assert cb(dict(gan, dist=5.0)) is False
    assert cb(dict(gan, side="bên phải")) is False


def test_khong_theo_vach_thi_nhu_cu(wb, monkeypatch):
    cb, _ = _cb(wb, monkeypatch, tren=False, theo=False)
    assert cb is None



def test_tren_vach_xe_do_khong_ro_vach_chi_bao_tin(wb, monkeypatch):
    # xe che mép vạch (không đo được) ngay trước mặt: bộ điều khiển bảo vòng, module vật cản im
    cb, ne = _cb(wb, monkeypatch, loai="khong_chac")
    assert cb(dat_vat_3d(ne.geo, -0.15, 0.15, 2.2)) is False
    assert cb(dict(dat_vat_3d(ne.geo, -0.15, 0.15, 2.2), side="bên phải")) is False


def test_mat_vach_vai_giay_van_tinh_dang_theo(wb, monkeypatch):
    # lỗi thật IMG_7354 252 s: "Vạch bị chắn" rồi "Dịch sang trái. Xe máy" — xe che vạch > 1,5 s, kính quay về chế độ cũ.
    # Vạch thấy trong 5 s gần nhất vẫn là đang theo vạch.
    import dan_theo_vach
    gio = [10.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = _NeVatChan()
    monkeypatch.setattr(nav.dan_vach, "da_tren_vach", lambda t: False)
    nav.dan_vach.t_thay = 7.0                                        # thấy vạch 3 s trước
    monkeypatch.setattr(dan_theo_vach, "phan_loai_vat", lambda *a, **k: "khong_chac")
    assert nav._chi_bao_vat_chan(gio[0]) is not None
    nav.dan_vach.t_thay = 4.0                                        # 6 s trước: đã rời hẳn -> như cũ
    assert nav._chi_bao_vat_chan(gio[0]) is None


# ---------- Xe tới gần che mép vạch (người dùng 29/10, demo IMG_7354 285–355 s) ----------
def _chay_xe_toi(wb, monkeypatch, loai_fn, tren_fn=None, giay=9.0, v=0.7, them_fn=None):
    """3 s đầu chỉ có vạch; sau đó xe đỗ giữa vạch tiến lại từ 4,6 m (người đi v m/s), qua mặt thì khuất (< 0,3 m).
    loai_fn(dist, t): kết quả phan_loai_vat giả; tren_fn(t): False = lần đo vạch mất (da_tren_vach False)."""
    import dan_theo_vach
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeVatChan()
    goc = dat_vat_3d(ne.geo, -0.15, 0.15, 2.8)
    monkeypatch.setattr(dan_theo_vach, "phan_loai_vat", lambda o, *a, **k: loai_fn(o["dist"], gio[0]))
    if tren_fn is not None:                                 # mất lần đo vạch: không trên vạch, không có lần đo
        that, do_that = nav.dan_vach.da_tren_vach, nav._do_vach_moi
        monkeypatch.setattr(nav.dan_vach, "da_tren_vach", lambda t: tren_fn(t) and that(t))
        monkeypatch.setattr(nav, "_do_vach_moi", lambda t: do_that(t) if tren_fn(t) else None)

    def moi_khung(da_noi):
        d = 4.6 - v * (gio[0] - 3.0)
        ne.cand = [dict(goc, dist=d, **(them_fn(d) if them_fn else {}))] if gio[0] >= 3.0 and d > 0.3 else []
    da_noi, _ = _chay_pf(nav, gio, [ve_vach(0, 0)] * int(giay * 10), wb, monkeypatch, moi_khung)
    return nav, [c for t, c in da_noi if t >= 3.0]


def test_xe_che_mep_vach_truoc_3m_van_bao_vong(wb, monkeypatch):
    # IMG_7354 341 s: xe "chan" ở 3,4–3,7 m, tới 3,3 m chính xe che mép vạch ("khong_chac") -> chưa kịp bảo vòng thì
    # module vật cản nói "Xe máy, phía trước". Người dùng: "phải nói là xe máy lấn vạch, đi vòng sang trái".
    _, cau = _chay_xe_toi(wb, monkeypatch, lambda d, t: "chan" if d > 3.3 else "khong_chac")
    assert sum("Đi sang trái" in c for c in cau) == 1
    assert CAU_NE not in cau


def test_da_bao_vong_toi_gan_xe_che_vach_khong_noi_lai(wb, monkeypatch):
    # IMG_7354 299–302 s: "Xe máy chắn giữa vạch. Đi sang trái." rồi tới gần chính xe đó che mép vạch -> bộ điều khiển
    # tưởng xe đã đi (hết chắn 1,5 s) -> "Xe máy, phía trước" về chính chiếc xe vừa bảo vòng.
    _, cau = _chay_xe_toi(wb, monkeypatch, lambda d, t: "chan" if d > 2.5 else "khong_chac", giay=10.0)
    assert sum("Đi sang trái" in c for c in cau) == 1
    assert CAU_NE not in cau


def test_mat_do_vach_dung_luc_xe_toi_3m_van_bao_vong(wb, monkeypatch):
    # IMG_7354 312 s: dãy xe che vạch -> mất lần đo vạch ("không trên vạch") đúng lúc xe tới 3 m -> chỉ "Xe máy, phía
    # trước". Ra ngoài vạch < 2 s chưa là rời vạch; xe vừa thấy chắn vẫn là xe chắn.
    _, cau = _chay_xe_toi(wb, monkeypatch, lambda d, t: "chan" if t < 5.0 else "khong_chac",
                          tren_fn=lambda t: t < 5.0)
    assert sum("Đi sang trái" in c for c in cau) == 1
    assert CAU_NE not in cau


def test_di_ngang_xe_do_mat_dau_van_la_xe_chan(wb, monkeypatch):
    # IMG_7358 98,5–99,8 s: đi ngang xe đỗ, YOLOE lập vết mới (chưa đo được tốc độ = "không đứng yên") -> bộ điều khiển
    # tưởng xe đã đi, module vật cản được nói lại về chính xe đó. Xe vừa thấy đỗ ở đúng chỗ đó vẫn là xe đỗ.
    nav, cau = _chay_xe_toi(wb, monkeypatch, lambda d, t: "chan" if d > 2.5 else "khong_chac", giay=10.0,
                            them_fn=lambda d: dict(chua_ro=d <= 2.5))
    assert sum("Đi sang trái" in c for c in cau) == 1
    assert CAU_NE not in cau
    assert nav.bo_vat_chan.dang_xu_ly


def test_xe_do_vua_thay_lech_ngang_thoang_qua_van_la_xe_do(wb, monkeypatch):
    # IMG_7354 26,3 s (29/10): người đi chéo -> xe đỗ có tốc độ ngang > 0,8 m/s trong 2 khung = "không đứng yên" ->
    # "Dừng lại. Xe máy" rồi 0,5 s sau "Xe máy chắn vạch bên phải. Đi sang trái.". Xe vừa thấy đỗ đúng chỗ đó vẫn là xe đỗ.
    _, cau = _chay_xe_toi(wb, monkeypatch, lambda d, t: "chan", them_fn=lambda d: dict(lech=2.6 < d < 3.1))
    assert sum("Đi sang trái" in c for c in cau) == 1
    assert CAU_NE not in cau


def test_do_duoc_vach_ma_xe_khong_phan_loai_duoc_chi_bao_tin(wb, monkeypatch):
    # khiem 57,9 s (29/10): đo được vạch nhưng người nhìn chéo sang trái vạch ("Rẽ phải một chút"), xe đỗ "phía trước"
    # camera không nằm trên vạch -> "Ô tô chắn vạch" sai. Đo được vạch thì "phía trước" camera không phải lối đi trên
    # vạch: chỉ báo thông tin như cũ.
    _, cau = _chay_xe_toi(wb, monkeypatch, lambda d, t: "khong_chac")
    assert not any("chắn" in c for c in cau)


def test_xe_che_kin_vach_mat_do_vach_lau_van_bao_vong(wb, monkeypatch):
    # như trên nhưng mất cả lần đo vạch > 2 s (IMG_7354 325,7 s: không trên vạch, không có lần đo): vẫn đang theo vạch
    _, cau = _chay_xe_toi(wb, monkeypatch, lambda d, t: "khong_chac", tren_fn=lambda t: t < 3.5)
    assert sum("Đi sang trái" in c for c in cau) == 1
    assert CAU_NE not in cau


def test_bao_module_vat_can_nguoi_dang_tren_vach(wb, monkeypatch):
    # module vật cản cần biết người đang đứng trên vạch (để không báo "Đang ở lòng đường" khi đang trên vạch)
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeVatChan()
    _chay_pf(nav, gio, [ve_vach(0, 0)] * 40, wb, monkeypatch)
    assert ne.tren_vach is True
    monkeypatch.setattr(nav.dan_vach, "da_tren_vach", lambda t: False)
    _chay_pf(nav, gio, [ve_vach(0, 0)] * 2, wb, monkeypatch)
    assert ne.tren_vach is False


def test_con_xe_dang_toi_thi_chua_bao_vong(wb, monkeypatch):
    # IMG_7355 147,1–147,8 s (29/10): "Xe máy, bên trái, đang tới" -> "Dừng lại." -> 0,2 s sau "Vật cản chắn vạch. Đi
    # sang trái." (về phía xe đang lao tới). Còn xe đang tới trong tầm nhìn thì người phải dừng: chưa bảo đi vòng.
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeVatChan()
    xe_toi = dict(dat_vat_3d(ne.geo, -2.0, -1.5, 4.0), toi=True, side="bên trái")

    def moi_khung(da_noi):
        if gio[0] >= 3.0:
            ne.cand = [dat_vat_3d(ne.geo, -0.15, 0.15, 2.8)] + ([xe_toi] if gio[0] < 6.0 else [])
    da_noi, _ = _chay_pf(nav, gio, [ve_vach(0, 0)] * 80, wb, monkeypatch, moi_khung)
    vong = [t for t, c in da_noi if "Đi sang" in c]
    assert vong and min(vong) >= 6.0                       # chỉ bảo vòng sau khi xe đang tới đã qua


class _NeXeToi(_NeVatChan):
    """Như _NeVatChan nhưng có xe đang tới thì nói câu "đang tới" (như module vật cản thật)."""
    def step(self, image, t=None, dang_qua_duong=False, chi_bao=None):
        self.chi_bao.append(chi_bao)
        return "Xe máy, bên trái, đang tới" if any(o.get("toi") for o in self.cand) else None


def test_vua_bao_xe_dang_toi_thi_chua_bao_vong(wb, monkeypatch):
    # IMG_7357 67,6 -> 68,6 s (29/10): "Xe máy, bên trái, đang tới" rồi 1 s sau "Vật cản chắn vạch. Đi sang trái." — xe
    # giảm tốc nên hết cờ "đang tới" nhưng vẫn đang đi ngang bên trái. Vừa báo xe đang tới (< 3 s): chưa bảo vòng.
    gio = [0.0]
    nav = _nav(wb, monkeypatch, gio)
    nav.ne_vat_can = ne = _NeXeToi()
    xe_toi = dict(dat_vat_3d(ne.geo, -2.0, -1.5, 4.0), toi=True, side="bên trái")

    def moi_khung(da_noi):
        if gio[0] >= 3.0:
            ne.cand = [dat_vat_3d(ne.geo, -0.15, 0.15, 2.8)] + ([xe_toi] if gio[0] < 3.5 else [])
    da_noi, _ = _chay_pf(nav, gio, [ve_vach(0, 0)] * 90, wb, monkeypatch, moi_khung)
    canh = [t for t, c in da_noi if "đang tới" in c]
    vong = [t for t, c in da_noi if "Đi sang" in c]
    assert canh and vong and min(vong) >= max(canh) + 3.0 - 0.15
