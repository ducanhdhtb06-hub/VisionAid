"""Test khoi_dong_dan_duong.py: tạo + làm nóng bộ dẫn đường lúc máy chủ khởi động.
Lỗi gốc (đo 2026-10-26): kính nối lần đầu -> BlindPathNavigator tạo trên vòng sự kiện 4,2 s (máy chủ đứng),
khung đầu 5 s (model khởi động) -> câu đầu sau ~15 s.
Chạy: ../.venv/bin/python -m pytest tests/test_khoi_dong_dan_duong.py -q  (đứng ở OpenAIglasses_for_Navigation/)
"""
import gc
from types import SimpleNamespace

import numpy as np

from khoi_dong_dan_duong import dong_bang_bo_nho, lam_nong, tao_bo_dan_duong


class MasterGia:
    def __init__(self, loi=False, blind=None):
        self.trang_thai = "IDLE"
        self.khung = []
        self.loi = loi
        self.blind = blind

    def start_blind_path_navigation(self):
        self.trang_thai = "BLINDPATH_NAV"

    def process_frame(self, bgr):
        if self.loi:
            raise RuntimeError("model hỏng")
        self.khung.append(bgr)

    def reset(self):
        self.trang_thai = "IDLE"

    def get_state(self):
        return self.trang_thai


def test_lam_nong_goi_process_frame_mot_lan():
    m = MasterGia()
    assert lam_nong(m) is not None
    assert len(m.khung) == 1
    assert m.khung[0].shape == (480, 640, 3) and m.khung[0].dtype == np.uint8


def test_lam_nong_tra_ve_idle():
    m = MasterGia()
    lam_nong(m)
    assert m.get_state() == "IDLE"


def test_lam_nong_loi_khong_nem():
    m = MasterGia(loi=True)
    assert lam_nong(m) is None
    assert m.get_state() == "IDLE"


class NeGia:
    H, W = 360, 480

    def __init__(self):
        self.goi = []

    def dvh(self, img, co=None):
        self.goi.append((img.shape, co))


def test_lam_nong_chay_ca_phan_vung_long_duong():
    # khung xám không có mặt đất nên module vật cản bỏ qua UperNet: lần chạy đầu của nó (1,3 s) rơi vào khung thật đầu
    # tiên, hình đứng ngay lúc bật dẫn đường (đo 2026-11-06)
    ne = NeGia()
    lam_nong(MasterGia(blind=SimpleNamespace(ne_vat_can=ne)))
    assert ne.goi == [((480, 640, 3), (360, 480))]


def test_lam_nong_khi_tat_phan_vung_van_chay():
    assert lam_nong(MasterGia(blind=SimpleNamespace(ne_vat_can=SimpleNamespace(dvh=None)))) is not None


def test_dong_bang_loai_doi_tuong_cu_khoi_viec_quet_cua_gc():
    # đo 2026-11-06: GC thế hệ 2 quét ~784.000 đối tượng tạo lúc nạp model, mất 256–333 ms và treo cả máy chủ, cứ ~40 s
    # một lần (hình đứng ~0,5 s). Đóng băng xong, GC chỉ còn 11–27 ms.
    class Vat:
        pass

    cu = Vat()
    try:
        assert dong_bang_bo_nho() > 0
        assert not any(o is cu for o in gc.get_objects())
        moi = Vat()
        assert any(o is moi for o in gc.get_objects())     # đối tượng tạo sau vẫn được dọn bình thường
    finally:
        gc.unfreeze()


def test_tao_bo_dan_duong_khong_model_tra_none():
    assert tao_bo_dan_duong(None, None) is None
