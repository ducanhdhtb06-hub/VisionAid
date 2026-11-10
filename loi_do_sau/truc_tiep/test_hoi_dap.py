import asyncio
import io
import wave

import numpy as np

from loi_do_sau.truc_tiep.hoi_dap import HoiGemini, pcm24_sang_wav


class QsGia:
    def __init__(self, cau='Phía trước là vỉa hè trống.'):
        self.cau, self.goi = cau, []

    async def mo_ta_gemini(self, jpeg, cau_hoi):
        self.goi.append(('nhin', jpeg, cau_hoi))
        return self.cau

    async def doc_gemini(self, cau):
        self.goi.append(('doc', cau))
        return np.zeros(4800, np.int16).tobytes()

    async def lam_nong(self):
        self.goi.append(('nong',))


def test_tra_loi_gui_anh_cau_hoi_va_doc_cau_tra_loi():
    qs = QsGia()
    cau, wav = asyncio.run(HoiGemini(qs).tra_loi(b'JPG', 'Phía trước có gì?'))
    assert cau == 'Phía trước là vỉa hè trống.'
    assert qs.goi == [('nhin', b'JPG', 'Phía trước có gì?'), ('doc', cau)]
    with wave.open(io.BytesIO(wav)) as w:            # đúng 24 kHz: sai tần số thì giọng đọc nhanh/chậm, méo tiếng
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (24000, 1, 2, 4800)


def test_khong_co_cau_tra_loi_thi_khong_goi_doc():
    qs = QsGia(cau='')
    assert asyncio.run(HoiGemini(qs).tra_loi(b'JPG', 'Có gì?')) == ('', None)
    assert [g[0] for g in qs.goi] == ['nhin']


def test_wav_24k():
    assert pcm24_sang_wav(b'\x00\x00' * 10)[:4] == b'RIFF'


class QsDocLoi(QsGia):
    """Nhìn được (vd. model trên máy) nhưng đọc thành tiếng lỗi / treo (Gemini TTS quá tải hay mất mạng)."""
    def __init__(self, loi=None, treo=False):
        super().__init__()
        self.loi, self.treo = loi, treo

    async def doc_gemini(self, cau):
        self.goi.append(('doc', cau))
        if self.treo:
            await asyncio.sleep(60)
        raise self.loi


def test_doc_loi_van_tra_chu_de_dien_thoai_tu_doc():
    # 2026-11-10: Gemini quá tải cả đọc lẫn nhìn; model trên máy trả lời được thì đừng bỏ câu trả lời
    cau, wav = asyncio.run(HoiGemini(QsDocLoi(loi=RuntimeError('503'))).tra_loi(b'JPG', 'Có gì?'))
    assert cau == 'Phía trước là vỉa hè trống.' and wav is None


def test_doc_treo_thi_thoi_cho_va_tra_chu(monkeypatch):
    import loi_do_sau.truc_tiep.hoi_dap as hd
    monkeypatch.setattr(hd, 'CHO_DOC', 0.05)
    cau, wav = asyncio.run(HoiGemini(QsDocLoi(treo=True)).tra_loi(b'JPG', 'Có gì?'))
    assert cau and wav is None


class QsMay(QsGia):
    async def mo_ta_tren_may(self, jpeg, cau_hoi):
        self.goi.append(('may', jpeg, cau_hoi))
        return self.cau

    async def lam_nong_tren_may(self):
        self.goi.append(('nong_may',))


def test_hoi_tren_may_khong_goi_gemini_va_de_dien_thoai_doc():
    from loi_do_sau.truc_tiep.hoi_dap import HoiTrenMay
    qs = QsMay()
    h = HoiTrenMay(qs)
    assert asyncio.run(h.tra_loi(b'JPG', 'Có gì?')) == ('Phía trước là vỉa hè trống.', None)
    asyncio.run(h.lam_nong())
    assert qs.goi == [('may', b'JPG', 'Có gì?'), ('nong_may',)]      # không nhìn / đọc / làm nóng bằng Gemini
