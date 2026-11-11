# -*- coding: utf-8 -*-
"""Module quan sát (quan_sat.py): ảnh + câu hỏi -> mô hình nhìn chạy trên máy (Qwen3-VL-2B qua llama-server)."""
import asyncio
import base64

import numpy as np
import pytest

import quan_sat as Q


def test_chi_con_mo_hinh_tren_may():
    # 2026-11-11: đã gỡ Gemini (nhìn + đọc) và chọn nhà cung cấp; chỉ dùng mô hình trên máy
    for ten in ("mo_ta_gemini", "doc_gemini", "doc_gemini_dan", "nha_cung_cap", "GEMINI_MODEL", "lam_nong"):
        assert not hasattr(Q, ten), ten
    assert "gemini" not in open(Q.__file__, encoding="utf-8").read().lower().replace("đã gỡ gemini", "")


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


def test_chi_dan_ngan_va_nhac_bac_truoc():
    assert "tối đa hai câu" in Q.CHI_DAN and "bậc" in Q.CHI_DAN


def test_tieng_tit_ngan_8k():
    pcm = Q.tieng_tit()
    assert len(pcm) == 2 * int(8000 * 0.15) and max(abs(int.from_bytes(pcm[i:i + 2], "little", signed=True))
                                                   for i in range(0, len(pcm), 2)) > 1000


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


def test_lich_phu_de_tinh_moc_tit_va_cau_tra_loi():
    from tools import thu_quan_sat as T
    l = T.lich_phu_de(dict(t0=1.1, t1=2.4, t_asr=0.2, t_tra_loi=0.5))
    assert (l["tit"], l["noi"]) == (2.6, 3.1)               # nghe xong 2,4 + chép chữ 0,2 -> tít; +0,5 -> câu trả lời
    assert l["hoi"] == (1.1, 3.1) and l["nghe"] == (2.6, 3.1)


def test_lich_phu_de_cau_hoi_go_tay_khong_co_buoc_nghe():
    from tools import thu_quan_sat as T
    l = T.lich_phu_de(dict(t0=None, t1=30.0, t_asr=0.0, t_tra_loi=0.4))
    assert l["tit"] == 30.0 and l["noi"] == 30.4 and l["hoi"][0] == 28.5       # nhãn "Bạn hỏi" hiện 1,5 s trước khung


class _Tra:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


def _client_may(monkeypatch, goi, loi=None):
    """AsyncOpenAI giả: ghi [(model, base_url, kw)]; `loi` là hàm trả ngoại lệ để ném ở mỗi lần gọi (None: trả lời)."""
    class _Client:
        def __init__(self, **kw):
            self.base_url = kw.get("base_url")
            self.chat = type("Chat", (), {"completions": self})()

        async def create(self, **kw):
            goi.append((kw["model"], self.base_url, kw))
            if loi:
                raise loi()
            return _Tra("ok " + kw["model"])
    monkeypatch.setattr(Q, "AsyncOpenAI", _Client)
    monkeypatch.setattr(Q, "MAY_NHIN_URL", "http://127.0.0.1:8090/v1")


def test_mo_ta_tren_may_gui_anh_chi_dan_toi_model_tren_may(monkeypatch):
    goi = []
    _client_may(monkeypatch, goi)
    assert asyncio.run(Q.mo_ta_tren_may(b"jpg", "Phía trước có gì?")) == "ok " + Q.MAY_NHIN
    assert [(m, u) for m, u, _ in goi] == [(Q.MAY_NHIN, "http://127.0.0.1:8090/v1")]
    kw = goi[0][2]
    assert kw["max_tokens"] == Q.MAY_NHIN_TOKEN
    assert kw["messages"][1]["content"][0]["type"] == "image_url"


def _loi_ket_noi():
    import httpx
    import openai
    return openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8090"))


def test_model_tren_may_chua_bat_thi_goi_lai_mot_lan_roi_bao_loi(monkeypatch):
    import openai
    goi = []
    _client_may(monkeypatch, goi, loi=_loi_ket_noi)
    with pytest.raises(openai.APIConnectionError):
        asyncio.run(Q.mo_ta_tren_may(b"jpg", "?"))
    assert len(goi) == 2                                     # llama-server vừa bật còn nạp model: gọi lại một lần


def test_lam_nong_tren_may_bang_anh_va_khong_nem_loi(monkeypatch):
    # lần hỏi ảnh đầu tiên của llama.cpp Vulkan mất 16 s (dịch shader, đo 2026-11-10): làm lúc bật máy chủ
    goi = []
    _client_may(monkeypatch, goi)
    asyncio.run(Q.lam_nong_tren_may())
    assert [m for m, _, _ in goi] == [Q.MAY_NHIN] and goi[0][2]["messages"][1]["content"][0]["type"] == "image_url"
    goi2 = []
    _client_may(monkeypatch, goi2, loi=_loi_ket_noi)
    asyncio.run(Q.lam_nong_tren_may())                       # chưa bật llama-server: chỉ in lỗi, không ném
