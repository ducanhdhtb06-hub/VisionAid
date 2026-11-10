import pytest

from loi_do_sau.truc_tiep.goi_tin import dong_goi, mo_goi


def test_dong_mo_khop():
    b = dong_goi(1729780000123.5, 42, b'\xff\xd8abc')
    assert mo_goi(b) == (1729780000123.5, 42, b'\xff\xd8abc')


def test_goi_ngan_bao_loi():
    with pytest.raises(ValueError):
        mo_goi(b'\x00' * 11)
