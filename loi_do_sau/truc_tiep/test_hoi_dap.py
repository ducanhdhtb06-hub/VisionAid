import asyncio

from loi_do_sau.truc_tiep import hoi_dap
from loi_do_sau.truc_tiep.hoi_dap import HoiTrenMay


class QsMay:
    def __init__(self, cau='Phía trước là vỉa hè trống.'):
        self.cau, self.goi = cau, []

    async def mo_ta_tren_may(self, jpeg, cau_hoi):
        self.goi.append(('may', jpeg, cau_hoi))
        return self.cau

    async def lam_nong_tren_may(self):
        self.goi.append(('nong_may',))


def test_hoi_tren_may_tra_chu_de_dien_thoai_doc():
    qs = QsMay()
    h = HoiTrenMay(qs)
    assert asyncio.run(h.tra_loi(b'JPG', 'Có gì?')) == ('Phía trước là vỉa hè trống.', None)
    asyncio.run(h.lam_nong())
    assert qs.goi == [('may', b'JPG', 'Có gì?'), ('nong_may',)]


def test_khong_con_gemini():
    # 2026-11-11: nút Hỏi chỉ dùng mô hình trên máy
    assert not hasattr(hoi_dap, 'HoiGemini') and not hasattr(hoi_dap, 'co_khoa')
    assert 'gemini' not in open(hoi_dap.__file__, encoding='utf-8').read().lower().replace('đã gỡ gemini', '')
