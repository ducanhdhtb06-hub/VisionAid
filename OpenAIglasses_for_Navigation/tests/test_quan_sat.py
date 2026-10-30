# -*- coding: utf-8 -*-
"""Module quan sát nhờ API (quan_sat.py): chọn nhà cung cấp, gửi ảnh + câu hỏi, đọc câu trả lời bằng Gemini TTS."""
import asyncio
import base64

import numpy as np
import pytest

import quan_sat as Q


def test_chon_nha_cung_cap(monkeypatch):
    monkeypatch.delenv("AIGLASS_QUAN_SAT_API", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert Q.nha_cung_cap() == "qwen"                     # mặc định như cũ
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert Q.nha_cung_cap() == "gemini"                   # có khóa Gemini thì dùng Gemini
    monkeypatch.setenv("AIGLASS_QUAN_SAT_API", "qwen")
    assert Q.nha_cung_cap() == "qwen"                     # đặt rõ thì theo biến
    monkeypatch.setenv("AIGLASS_QUAN_SAT_API", "abc")
    with pytest.raises(ValueError):
        Q.nha_cung_cap()


def test_la_cau_quan_sat():
    assert Q.la_cau_quan_sat("Phía trước có gì vậy?")
    assert Q.la_cau_quan_sat("Phía trước tôi là gì?")       # IMG_2143 (30/10): câu người dùng thật nói
    assert Q.la_cau_quan_sat("Đọc giúp tôi hạn dùng")
    assert Q.la_cau_quan_sat("Cái áo này màu gì?")
    assert Q.la_cau_quan_sat("mô tả xung quanh giúp tôi")
    assert not Q.la_cau_quan_sat("bắt đầu dẫn đường")


def test_noi_dung_gui_kem_anh_va_chi_dan():
    nd = Q.noi_dung(b"\xff\xd8abc", "Phía trước có gì?")
    assert nd[0]["role"] == "system" and "tiếng Việt" in nd[0]["content"]
    phan = nd[1]["content"]
    assert phan[0]["type"] == "image_url"
    assert phan[0]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8abc").decode()
    assert phan[1] == {"type": "text", "text": "Phía trước có gì?"}
    assert Q.noi_dung(None, "chào")[1]["content"] == [{"type": "text", "text": "chào"}]    # chưa có ảnh: chỉ chữ


class _Tra:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


def test_mo_ta_gemini_goi_cong_tuong_thich_openai(monkeypatch):
    goi = {}

    class _Client:
        def __init__(self, **kw):
            goi["kw"] = kw
            self.chat = type("Chat", (), {"completions": self})()

        async def create(self, **kw):
            goi["create"] = kw
            return _Tra("  Phía trước có một xe máy đỗ bên phải.  ")
    monkeypatch.setenv("GEMINI_API_KEY", "khoa")
    monkeypatch.setattr(Q, "AsyncOpenAI", _Client)
    monkeypatch.setattr(Q, "_oai", {})
    cau = asyncio.run(Q.mo_ta_gemini(b"jpg", "Phía trước có gì?"))
    assert cau == "Phía trước có một xe máy đỗ bên phải."
    assert goi["kw"] == dict(api_key="khoa", base_url=Q.GEMINI_BASE_URL, timeout=Q.HET_GIO)
    assert goi["create"]["model"] == Q.GEMINI_MODEL and goi["create"]["messages"][1]["content"][0]["type"] == "image_url"


def test_thieu_khoa_gemini_bao_loi_ro(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        asyncio.run(Q.mo_ta_gemini(b"jpg", "?"))


def test_doc_gemini_tra_pcm_24k(monkeypatch):
    pcm = b"\x01\x00\x02\x00"
    goi = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"inlineData": {
                "mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm).decode()}}]}}]}

    class _Http:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            goi.update(url=url, headers=headers, json=json)
            return _Resp()
    monkeypatch.setenv("GEMINI_API_KEY", "khoa")
    monkeypatch.setattr(Q.httpx, "AsyncClient", _Http)
    monkeypatch.setattr(Q, "_http", None)
    assert asyncio.run(Q.doc_gemini("Xin chào")) == pcm
    assert Q.GEMINI_TTS_MODEL in goi["url"] and goi["headers"]["x-goog-api-key"] == "khoa"
    assert goi["json"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert goi["json"]["contents"][0]["parts"][0]["text"] == "Xin chào"     # không kèm lời dặn (bị đọc thành tiếng)


def test_chi_dan_ngan_va_nhac_bac_truoc():
    assert "tối đa hai câu" in Q.CHI_DAN and "bậc" in Q.CHI_DAN


def test_tieng_tit_ngan_8k():
    pcm = Q.tieng_tit()
    assert len(pcm) == 2 * int(8000 * 0.15) and max(abs(int.from_bytes(pcm[i:i + 2], "little", signed=True))
                                                   for i in range(0, len(pcm), 2)) > 1000


class _Sse:
    """Giả luồng SSE của streamGenerateContent: mỗi dòng data: chứa một mẩu PCM."""
    def __init__(self, manh, loi=None):
        self.manh, self.loi = manh, loi

    def raise_for_status(self):
        if self.loi:
            raise self.loi

    async def aiter_lines(self):
        import json
        for m in self.manh:
            yield "data: " + json.dumps({"candidates": [{"content": {"parts": [{"inlineData": {
                "mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(m).decode()}}]}}]})
            yield ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def test_doc_dan_tra_tung_manh_va_goi_lai_mot_lan(monkeypatch):
    goi = []

    class _Http:
        def stream(self, method, url, headers=None, json=None):
            goi.append(url)
            if len(goi) == 1:                                 # lần đầu lỗi mạng -> gọi lại một lần
                raise Q.httpx.ReadTimeout("chậm")
            return _Sse([b"\x01\x00", b"\x02\x00"])
    monkeypatch.setenv("GEMINI_API_KEY", "khoa")
    monkeypatch.setattr(Q, "_http", _Http())

    async def lay():
        return [m async for m in Q.doc_gemini_dan("Xin chào")]
    assert asyncio.run(lay()) == [b"\x01\x00", b"\x02\x00"]
    assert len(goi) == 2 and "streamGenerateContent" in goi[0] and "alt=sse" in goi[0]


def test_doc_dan_loi_ca_hai_lan_thi_bao_loi(monkeypatch):
    class _Http:
        def stream(self, *a, **k):
            raise Q.httpx.ReadTimeout("chậm")
    monkeypatch.setenv("GEMINI_API_KEY", "khoa")
    monkeypatch.setattr(Q, "_http", _Http())

    async def lay():
        return [m async for m in Q.doc_gemini_dan("Xin chào")]
    with pytest.raises(Q.httpx.ReadTimeout):
        asyncio.run(lay())


def _tieng(giay, doan, sr=16000):
    """Nền ồn nhỏ + các đoạn "nói" (ồn to hơn) đặt ở [(bắt đầu, kết thúc)] giây."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, int(giay * sr)).astype(np.float32)
    for t0, t1 in doan:
        i0, i1 = int(t0 * sr), int(t1 * sr)
        a[i0:i1] += rng.normal(0, 0.15, i1 - i0).astype(np.float32)
    return a


def test_cat_doan_noi_tim_dung_cac_doan():
    a = _tieng(10, [(1.0, 2.0), (5.0, 5.8)])
    d = Q.cat_doan_noi(a, 16000)
    assert len(d) == 2
    assert abs(d[0][0] - 1.0) < 0.15 and abs(d[0][1] - 2.0) < 0.15
    assert abs(d[1][0] - 5.0) < 0.15 and abs(d[1][1] - 5.8) < 0.15


def test_cat_doan_noi_noi_cac_quang_nghi_ngan_va_bo_tieng_dong_ngan():
    a = _tieng(10, [(1.0, 1.6), (1.9, 2.5), (6.0, 6.15)])     # 2 nửa cách 0,3 s = một câu; tiếng động 0,15 s bị bỏ
    d = Q.cat_doan_noi(a, 16000)
    assert len(d) == 1 and abs(d[0][0] - 1.0) < 0.15 and abs(d[0][1] - 2.5) < 0.15


def test_cat_doan_noi_im_lang_thi_khong_co_doan():
    assert Q.cat_doan_noi(_tieng(5, []), 16000) == []


def test_lich_phu_de_tinh_moc_tit_va_tieng_dau():
    from tools import thu_quan_sat as T
    l = T.lich_phu_de(dict(t0=1.1, t1=2.4, t_asr=0.2, t_dau=4.2, giay_tieng=8.3))
    assert (l["tit"], l["noi"], l["het"]) == (2.6, 6.8, 15.1)        # nghe xong 2,4 + 0,2 -> tít; +4,2 -> tiếng đầu
    assert l["hoi"] == (1.1, 6.8) and l["nghe"] == (2.6, 6.8)


def test_lich_phu_de_cau_hoi_go_tay_khong_co_buoc_nghe():
    from tools import thu_quan_sat as T
    l = T.lich_phu_de(dict(t0=None, t1=30.0, t_asr=0.0, t_dau=3.0, giay_tieng=5.0))
    assert l["tit"] == 30.0 and l["noi"] == 33.0 and l["hoi"][0] == 28.5       # nhãn "Bạn hỏi" hiện 1,5 s trước khung


def _het_luot_oai():
    import openai
    return openai.RateLimitError("quota", response=Q.httpx.Response(429, request=Q.httpx.Request("POST", "http://x")), body=None)


def test_ds_khoa_doc_nhieu_khoa_khong_trung(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "b")
    monkeypatch.setenv("GEMINI_API_KEYS", "c, a ,d")
    assert Q._ds_khoa() == ["c", "a", "d", "b"]


def test_het_luot_thi_chuyen_sang_khoa_du_phong_va_nho(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "b")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(Q, "_oai", {})
    monkeypatch.setattr(Q, "_het_luot", {})
    dung = []

    class _Client:
        def __init__(self, api_key, **kw):
            self.key = api_key
            self.chat = type("Chat", (), {"completions": self})()

        async def create(self, **kw):
            dung.append(self.key)
            if self.key == "a":
                raise _het_luot_oai()
            return _Tra("ok " + self.key)
    monkeypatch.setattr(Q, "AsyncOpenAI", _Client)
    assert asyncio.run(Q.mo_ta_gemini(b"jpg", "?")) == "ok b"
    assert dung == ["a", "b"]                                    # a hết lượt -> b
    dung.clear()
    assert asyncio.run(Q.mo_ta_gemini(b"jpg", "?")) == "ok b"
    assert dung == ["b"]                                         # nhớ a vừa hết lượt, không thử lại ngay


def test_het_luot_ca_hai_khoa_thi_bao_loi(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "b")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(Q, "_oai", {})
    monkeypatch.setattr(Q, "_het_luot", {})

    class _Client:
        def __init__(self, api_key, **kw):
            self.chat = type("Chat", (), {"completions": self})()

        async def create(self, **kw):
            raise _het_luot_oai()
    monkeypatch.setattr(Q, "AsyncOpenAI", _Client)
    import openai
    with pytest.raises(openai.RateLimitError):
        asyncio.run(Q.mo_ta_gemini(b"jpg", "?"))


def test_doc_dan_het_luot_thi_chuyen_khoa_du_phong(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "b")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(Q, "_het_luot", {})
    dung = []

    class _Http:
        def stream(self, method, url, headers=None, json=None):
            dung.append(headers["x-goog-api-key"])
            if headers["x-goog-api-key"] == "a":
                return _Sse([], loi=Q.httpx.HTTPStatusError("429", request=Q.httpx.Request("POST", url),
                                                           response=Q.httpx.Response(429)))
            return _Sse([b"\x01\x00"])
    monkeypatch.setattr(Q, "_http", _Http())

    async def lay():
        return [m async for m in Q.doc_gemini_dan("Xin chào")]
    assert asyncio.run(lay()) == [b"\x01\x00"] and dung == ["a", "b"]
