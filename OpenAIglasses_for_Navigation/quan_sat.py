# -*- coding: utf-8 -*-
"""Quan sát nhờ API: gửi ảnh camera + câu hỏi của người dùng lên AI thị giác, đọc câu trả lời.

Nhà cung cấp chọn bằng biến môi trường AIGLASS_QUAN_SAT_API:
  qwen   — Qwen-Omni (DashScope) như cũ: trả cả chữ lẫn giọng (omni_client.stream_chat)
  gemini — Gemini qua cổng tương thích OpenAI (chữ), rồi Gemini TTS đọc thành PCM16 24 kHz
Không đặt: có GEMINI_API_KEY thì dùng Gemini, không thì Qwen.
Lưu ý: ảnh camera (có thể có người đi đường) được gửi ra máy chủ của nhà cung cấp.
"""
import base64
import json
import os

import httpx
import numpy as np
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{kieu}"
# đo 30/10 (IMG_2143): 3.1-flash-lite 1,8 s; 3.5-flash-lite 2,0 s; 3.5-flash 50 s; 3.8-flash không trả lời
GEMINI_MODEL = os.getenv("AIGLASS_GEMINI_MODEL", "gemini-3.1-flash-lite")
# đo 30/10: 3.8-flash-lite-tts 3,7 s cả câu, phát dần thì mẩu đầu 1,0 s; 3.1-flash-tts 5,1 s; 2.5-flash-tts 5,8 s
GEMINI_TTS_MODEL = os.getenv("AIGLASS_GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_GIONG = os.getenv("AIGLASS_GEMINI_GIONG", "Kore")
HET_GIO = 20.0                                   # giây; quá thì báo lỗi thay vì treo

CHI_DAN = (
    "Bạn là trợ lý trên kính cho người khiếm thị, nhìn qua camera trên kính (ảnh kèm theo là những gì phía trước "
    "người dùng). Luôn trả lời bằng tiếng Việt, ngắn gọn, tối đa hai câu, vì người dùng nghe qua loa. "
    "Nói trước những gì có thể làm vấp hoặc va: bậc thềm, cầu thang, mép, vật dưới chân, vật chắn đường; có những "
    "thứ đó thì không nói đường thông thoáng. Dùng 'bên trái', 'bên phải', 'phía trước'. "
    "Nếu người dùng hỏi về vật đang cầm, đọc chữ trên vật (tên, hạn dùng, mệnh giá tiền); chữ không rõ thì bảo "
    "xoay vật hoặc đưa lại gần. Không đoán số mét. Thấy không rõ thì nói là không rõ, không bịa."
)

KW_QUAN_SAT = ["phía trước", "trước mặt", "xung quanh", "quan sát", "mô tả", "đây là đâu", "đang ở đâu",
               "nhìn xem", "nhìn giúp", "xem giúp", "giúp tôi xem", "là gì", "có gì", "đọc giúp", "đọc chữ",
               "màu gì", "hạn dùng", "bao nhiêu tiền"]

_oai = None                                      # client dùng chung: giữ kết ghep, khỏi bắt tay TLS mỗi lần hỏi
_http = None


def nha_cung_cap() -> str:
    ten = os.getenv("AIGLASS_QUAN_SAT_API", "").strip().lower()
    if not ten:
        return "gemini" if os.getenv("GEMINI_API_KEY") else "qwen"
    if ten not in ("gemini", "qwen"):
        raise ValueError(f"AIGLASS_QUAN_SAT_API={ten!r}: chỉ nhận 'gemini' hoặc 'qwen'")
    return ten


def la_cau_quan_sat(cau: str) -> bool:
    c = cau.lower()
    return any(k in c for k in KW_QUAN_SAT)


def noi_dung(jpeg, cau_hoi: str) -> list:
    """Tin nhắn kiểu OpenAI: chỉ dẫn hệ thống + (ảnh nếu có) + câu hỏi."""
    phan = []
    if jpeg:
        phan.append({"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}})
    phan.append({"type": "text", "text": cau_hoi})
    return [{"role": "system", "content": CHI_DAN}, {"role": "user", "content": phan}]


def _khoa_gemini() -> str:
    khoa = os.getenv("GEMINI_API_KEY", "")
    if not khoa:
        raise RuntimeError("Chưa đặt GEMINI_API_KEY (trong .env)")
    return khoa


def _client_oai():
    global _oai
    if _oai is None:
        _oai = AsyncOpenAI(api_key=_khoa_gemini(), base_url=GEMINI_BASE_URL, timeout=HET_GIO)
    return _oai


def _client_http():
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=HET_GIO)
    return _http


async def mo_ta_gemini(jpeg, cau_hoi: str) -> str:
    """Gửi ảnh + câu hỏi lên Gemini, trả câu trả lời (chữ). Lỗi mạng: gọi lại một lần."""
    for lan in range(2):
        try:
            tra = await _client_oai().chat.completions.create(model=GEMINI_MODEL, messages=noi_dung(jpeg, cau_hoi))
            return (tra.choices[0].message.content or "").strip()
        except (APITimeoutError, APIConnectionError):
            if lan:
                raise


def _than_tts(cau: str) -> dict:
    # chỉ gửi đúng câu cần đọc: thêm lời dặn ("Đọc bằng tiếng Việt…") thì model đọc luôn cả lời dặn (PhoWhisper
    # chép lại tiếng 30/10 thấy rõ)
    return {"contents": [{"parts": [{"text": cau}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_GIONG}}}}}


async def doc_gemini(cau: str) -> bytes:
    """Gemini TTS: chữ -> PCM16 mono 24 kHz (chờ đủ cả câu)."""
    r = await _client_http().post(GEMINI_TTS_URL.format(model=GEMINI_TTS_MODEL, kieu="generateContent"),
                                  headers={"x-goog-api-key": _khoa_gemini()}, json=_than_tts(cau))
    r.raise_for_status()
    phan = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
    return base64.b64decode(phan["data"])


async def doc_gemini_dan(cau: str):
    """Gemini TTS phát dần: sinh từng mẩu PCM16 24 kHz ngay khi tới (mẩu đầu ~1 s thay vì ~3,7 s chờ cả câu).
    Lỗi mạng trước khi có mẩu nào: gọi lại một lần."""
    url = GEMINI_TTS_URL.format(model=GEMINI_TTS_MODEL, kieu="streamGenerateContent") + "?alt=sse"
    for lan in range(2):
        da_co = False
        try:
            async with _client_http().stream("POST", url, headers={"x-goog-api-key": _khoa_gemini()},
                                             json=_than_tts(cau)) as r:
                r.raise_for_status()
                async for dong in r.aiter_lines():
                    if not dong.startswith("data:"):
                        continue
                    for p in json.loads(dong[5:])["candidates"][0]["content"].get("parts", []):
                        if "inlineData" in p:
                            da_co = True
                            yield base64.b64decode(p["inlineData"]["data"])
            return
        except (httpx.TimeoutException, httpx.NetworkError):
            if lan or da_co:
                raise


def tieng_tit(giay: float = 0.15, hz: float = 880.0) -> bytes:
    """Tiếng "tít" PCM16 mono 8 kHz: phát ngay khi nhận câu hỏi để người dùng biết kính đang xử lý."""
    n = int(8000 * giay)
    mau = (np.sin(2 * np.pi * hz * np.arange(n) / 8000) * 6000 * np.hanning(n)).astype("<i2")
    return mau.tobytes()


async def lam_nong():
    """Gọi trước một yêu cầu nhỏ lúc bật máy chủ: lần gọi đầu model nhìn chậm hẳn (đo 30/10: 4,5–9,4 s).
    Không làm nóng model đọc: gói miễn phí giới hạn số lần gọi TTS mỗi phút (429 khi gọi dồn, 30/10), để dành lượt
    cho câu hỏi thật."""
    try:
        await _client_oai().chat.completions.create(model=GEMINI_MODEL, messages=[{"role": "user", "content": "Chào"}],
                                                    max_tokens=5)
        print("[QUAN_SAT] Đã làm nóng Gemini", flush=True)
    except Exception as e:
        print(f"[QUAN_SAT] Làm nóng Gemini lỗi (bỏ qua): {e}", flush=True)


def cat_doan_noi(a, sr: int = 16000, khung: float = 0.05, nghi_toi_da: float = 0.5, dai_toi_thieu: float = 0.4):
    """Tìm các đoạn có người nói trong âm thanh mono `a` (float, [-1, 1]): [(bắt đầu giây, kết thúc giây)].

    Ngưỡng theo mức ồn nền của chính đoạn thu (trung vị năng lượng) nên chạy được trên video quay ngoài đường lẫn
    trong nhà. Quãng nghỉ ngắn hơn `nghi_toi_da` được ghep lại (một câu); đoạn ngắn hơn `dai_toi_thieu` (tiếng động) bỏ."""
    n = int(sr * khung)
    if len(a) < n:
        return []
    nang = np.sqrt((a[:len(a) // n * n].reshape(-1, n) ** 2).mean(axis=1))
    nguong = max(3.0 * float(np.median(nang)), 0.2 * float(np.percentile(nang, 99)), 0.02)
    noi = nang > nguong
    doan, dau = [], None
    for i, v in enumerate(noi):
        if v and dau is None:
            dau = i
        elif not v and dau is not None:
            doan.append([dau * khung, i * khung])
            dau = None
    if dau is not None:
        doan.append([dau * khung, len(noi) * khung])
    ghep = []
    for d in doan:
        if ghep and d[0] - ghep[-1][1] <= nghi_toi_da:
            ghep[-1][1] = d[1]
        else:
            ghep.append(d)
    return [(t0, t1) for t0, t1 in ghep if t1 - t0 >= dai_toi_thieu]
