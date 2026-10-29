"""Quét câu mâu thuẫn trong câu kính đã phát (tools/esp32_ao/mau_thuan.py).
Người dùng 28/10: "vừa bảo đi sang trái lại bảo dừng lại" — đo bằng máy, không soi bằng mắt.
Chạy: ../.venv/bin/python -m pytest tests/test_esp32_ao_mau_thuan.py -q  (đứng ở OpenAIglasses_for_Navigation/)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from esp32_ao.mau_thuan import quet  # noqa: E402


def _loai(kq):
    return [k["loai"] for k in kq]


def test_vong_roi_dung_la_mau_thuan():
    # IMG_7354 (28/10): "Xe máy chắn giữa vạch. Đi sang trái." rồi 0,25 s sau "Dừng lại. Xe máy"
    kq = quet([(68.61, "Xe máy chắn giữa vạch. Đi sang trái."), (68.86, "Dừng lại. Xe máy")])
    assert _loai(kq) == ["vong_roi_dung"]


def test_bi_chan_roi_dich_la_mau_thuan():
    # IMG_7354 (28/10): "Vạch bị chắn, dừng lại, hãy dùng gậy." rồi 2 s sau "Dịch sang trái. Xe máy, 2 mét"
    kq = quet([(251.8, "Vạch bị chắn, dừng lại, hãy dùng gậy."), (253.8, "Dịch sang trái. Xe máy, 2 mét")])
    assert _loai(kq) == ["dung_roi_di"]


def test_vong_roi_dich_nguoc_lai():
    kq = quet([(10.0, "Xe máy chắn vạch bên phải. Đi sang trái."), (12.0, "Dịch sang phải")])
    assert _loai(kq) == ["vong_roi_dich"]


def test_hai_cau_ve_mot_vat_dinh_lien_la_thua():
    # IMG_7354 (28/10): "Xe máy, phía trước" rồi 0,2 s sau "Dừng lại. Xe máy"
    kq = quet([(27.4, "Xe máy, phía trước"), (27.6, "Dừng lại. Xe máy")])
    assert _loai(kq) == ["thua"]


def test_khong_tinh_canh_bao_nguy_hiem_va_cau_xa_nhau():
    assert quet([(10.0, "Xe máy chắn vạch bên phải. Đi sang trái."), (10.5, "Xe máy, bên trái, đang tới")]) == []
    assert quet([(10.0, "Xe máy chắn vạch bên phải. Đi sang trái."), (10.5, "Sắp xuống lòng đường, phía trước")]) == []
    assert quet([(10.0, "Xe máy chắn vạch bên phải. Đi sang trái."), (16.0, "Dừng lại. Xe máy")]) == []
    assert quet([(10.0, "Xe máy chắn vạch bên phải. Đi sang trái."), (14.0, "Đã về vạch.")]) == []


def test_dung_lai_ngan_sau_xe_dang_toi_la_nguy_hiem():
    # "Dừng lại." đứng riêng chỉ nói ngay sau "X, đang tới" (Speaker.ngan, 29/10) -> cảnh báo nguy hiểm, không tính
    cau = [(10.0, "Xe máy chắn vạch bên phải. Đi sang trái."), (11.0, "Xe máy, bên trái, đang tới"), (11.3, "Dừng lại.")]
    assert quet(cau) == []


def test_khac_vat_hoac_chua_ro_ben_khong_tinh():
    # IMG_7358 bản cuối (29/10): "Dừng lại. Người" rồi 4 s sau bảo vòng XE MÁY (vật khác); "Dùng gậy dò đường vòng."
    # rồi "Đi sang trái." (biết thêm bên trống, không phải đổi bên) -> không phải mâu thuẫn
    cau = [(6.2, "Dừng lại. Người"), (10.3, "Xe máy chắn vạch bên phải. Đi sang trái."),
           (98.2, "Xe máy chắn vạch bên phải. Dùng gậy dò đường vòng."), (100.2, "Xe máy chắn vạch bên phải. Đi sang trái.")]
    assert quet(cau) == []
    assert _loai(quet([(1.0, "Dừng lại. Xe máy"), (1.5, "Xe máy chắn vạch bên phải. Đi sang trái.")])) == ["dung_roi_di"]
    assert _loai(quet([(1.0, "Xe máy chắn vạch bên phải. Đi sang trái."),
                       (2.0, "Xe máy chắn vạch bên trái. Đi sang phải.")])) == ["vong_doi_ben"]


def test_bao_vong_ve_phia_xe_dang_toi():
    # IMG_7355 147 s (29/10): "Xe máy, bên trái, đang tới" rồi "… Đi sang trái." -> đi về phía xe đang lao tới
    cau = [(147.1, "Xe máy, bên trái, đang tới"), (147.6, "Dừng lại."), (147.8, "Vật cản chắn vạch. Đi sang trái.")]
    assert _loai(quet(cau)) == ["vong_ve_phia_xe"]
    assert quet([(1.0, "Xe máy, bên trái, đang tới"), (2.0, "Xe máy chắn vạch bên trái. Đi sang phải.")]) == []
