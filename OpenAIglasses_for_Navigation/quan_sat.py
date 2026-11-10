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
import time

import httpx
import numpy as np
from openai import (NOT_GIVEN, APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError,
                    NotFoundError, RateLimitError)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{kieu}"
# đo 30/10 (IMG_2143): 3.1-flash-lite 1,8 s; 3.5-flash-lite 2,0 s; 3.5-flash 50 s; 3.8-flash không trả lời
GEMINI_MODEL = os.getenv("AIGLASS_GEMINI_MODEL", "gemini-3.1-flash-lite")
# đo 30/10: 3.8-flash-lite-tts 3,7 s cả câu, phát dần thì mẩu đầu 1,0 s; 3.1-flash-tts 5,1 s; 2.5-flash-tts 5,8 s
GEMINI_TTS_MODEL = os.getenv("AIGLASS_GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_GIONG = os.getenv("AIGLASS_GEMINI_GIONG", "Kore")
HET_GIO = 20.0                                   # giây; quá thì báo lỗi thay vì treo


def _ds_env(ten: str, mac_dinh: str) -> list:
    return [m.strip() for m in os.getenv(ten, mac_dinh).split(",") if m.strip()]


# Model dự phòng (cách nhau dấu phẩy): model chính quá tải (503), lỗi máy chủ, bị gỡ (404), hết lượt hay chậm quá
# CHO_MODEL giây thì hỏi model kế. Đo 2026-11-10 lúc 3.1-flash-lite quá tải: nó 19,6 s, 3.5-flash-lite 1,9 s;
# 3-flash-preview 3,8–4,0 s; 2.5-flash(-lite) đã bị gỡ (404); flash-latest, 3.7-flash cũng 503.
# Đọc: 3.8-flash-lite-tts 2,2 s, 3.1-flash-tts-preview 6,4 s, 2.5-flash-preview-tts 12,8 s, 3.1-flash-tts 404.
GEMINI_MODEL_DU_PHONG = _ds_env("AIGLASS_GEMINI_MODEL_DU_PHONG", "gemini-3.5-flash-lite,gemini-3-flash-preview")
GEMINI_TTS_DU_PHONG = _ds_env("AIGLASS_GEMINI_TTS_DU_PHONG", "gemini-3.1-flash-tts-preview,gemini-2.5-flash-preview-tts")
CHO_MODEL = float(os.getenv("AIGLASS_GEMINI_CHO", "6"))          # giây chờ một model nhìn khi còn model khác để hỏi
CHO_MODEL_DOC = float(os.getenv("AIGLASS_GEMINI_CHO_DOC", "8"))  # như trên cho model đọc (câu dài 3,7 s lúc bình thường)
NGHI_QUA_TAI = 120.0                             # giây; model vừa lỗi thì các câu hỏi sau đi thẳng sang model dự phòng

# Model nhìn chạy TRÊN MÁY: llama.cpp llama-server (MIT) + Qwen3-VL-2B-Instruct Q4_K_M (Apache-2.0), cổng tương thích
# OpenAI. Dự phòng CUỐI khi mọi model Gemini lỗi / chậm / mất mạng / không có khóa. Đo 2026-11-10 trên RTX 4060 (Vulkan):
# 0,25–0,63 s một câu (Gemini lúc quá tải 2–20 s), tả kém Gemini hơn chút. Rỗng = tắt. chay.py tự bật llama-server.
MAY_NHIN_URL = os.getenv("AIGLASS_MAY_NHIN_URL", "http://127.0.0.1:8090/v1")
MAY_NHIN = "tren-may"                            # tên trong danh sách model (llama-server bỏ qua tên model)
MAY_NHIN_TOKEN = 150                             # chặn câu dài lan man của model nhỏ (chỉ dẫn đã dặn tối đa hai câu)

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

_oai = {}                                        # client dùng chung theo từng khóa: giữ kết nối, khỏi bắt tay TLS mỗi lần hỏi
_http = None
_het_luot = {}                                   # khóa -> thời điểm (monotonic) được thử lại sau khi báo hết lượt (429)
THU_LAI_SAU = 900.0                              # giây; khóa hết lượt trong ngày: khóa kia gánh, sau đó mới thử lại
_qua_tai = {}                                    # model -> thời điểm (monotonic) hết nghỉ sau khi lỗi / chậm quá


def _thu_tu_model(chinh: str, du_phong: list) -> list:
    """Model chính rồi các model dự phòng; model đang nghỉ (vừa lỗi) xếp cuối, vẫn thử khi các model khác cũng lỗi."""
    ds = list(dict.fromkeys([chinh, *du_phong]))
    now = time.monotonic()
    return sorted(ds, key=lambda m: _qua_tai.get(m, 0.0) > now)


def _cho_model_nghi(model: str, ly_do) -> None:
    _qua_tai[model] = time.monotonic() + NGHI_QUA_TAI
    print(f"[QUAN_SAT] Model {model} không dùng được ({ly_do}), hỏi model dự phòng", flush=True)


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


def _ds_khoa() -> list:
    """Các khóa Gemini theo thứ tự ưu tiên: GEMINI_API_KEYS (cách nhau dấu phẩy), GEMINI_API_KEY, GEMINI_API_KEY_2, _3."""
    ks = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
    for ten in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        k = os.getenv(ten, "").strip()
        if k and k not in ks:
            ks.append(k)
    return ks


def _khoa_gemini() -> str:
    """Khóa đang dùng: khóa đầu tiên chưa báo hết lượt; hết cả thì khóa sắp được thử lại sớm nhất."""
    ks = _ds_khoa()
    if not ks:
        raise RuntimeError("Chưa đặt GEMINI_API_KEY (trong .env)")
    now = time.monotonic()
    con = [k for k in ks if _het_luot.get(k, 0.0) <= now]
    return con[0] if con else min(ks, key=lambda k: _het_luot.get(k, 0.0))


def _het_luot_khoa(khoa: str) -> bool:
    """Ghi nhận khóa vừa hết lượt; True nếu còn khóa khác chưa hết để thử."""
    _het_luot[khoa] = time.monotonic() + THU_LAI_SAU
    print("[QUAN_SAT] Khóa Gemini hết lượt" + (", chuyển khóa dự phòng" if len(_ds_khoa()) > 1 else ""), flush=True)
    return any(_het_luot.get(k, 0.0) <= time.monotonic() for k in _ds_khoa())


def _client_oai(khoa: str):
    if khoa not in _oai:
        # max_retries=0: thư viện tự thử lại 503 có chờ lùi -> một câu hỏi kéo 30–37 s (đo 2026-11-10). Thử lại /
        # chuyển model do mo_ta_gemini tự lo.
        _oai[khoa] = AsyncOpenAI(api_key=khoa, base_url=GEMINI_BASE_URL, timeout=HET_GIO, max_retries=0)
    return _oai[khoa]


def _client_http():
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=HET_GIO)
    return _http


async def mo_ta_gemini(jpeg, cau_hoi: str) -> str:
    """Gửi ảnh + câu hỏi lên Gemini, trả câu trả lời (chữ).
    Model chính quá tải / lỗi máy chủ / bị gỡ / chậm quá CHO_MODEL giây: hỏi model dự phòng. Hết lượt: chuyển khóa dự
    phòng, hết cả khóa thì sang model kế (lượt tính theo model). Model cuối lỗi mạng / máy chủ: gọi lại một lần.
    Sau cùng là model trên máy (MAY_NHIN_URL); không có khóa Gemini thì hỏi thẳng model trên máy."""
    if not _ds_khoa() and MAY_NHIN_URL:
        ds = [MAY_NHIN]
    else:
        ds = _thu_tu_model(GEMINI_MODEL, GEMINI_MODEL_DU_PHONG + ([MAY_NHIN] if MAY_NHIN_URL else []))
    i, goi_lai = 0, 0
    while True:
        model, cuoi = ds[i], i == len(ds) - 1
        if model == MAY_NHIN:
            khoa, toi_da = None, MAY_NHIN_TOKEN
            client = AsyncOpenAI(api_key="khong-can", base_url=MAY_NHIN_URL, timeout=HET_GIO, max_retries=0)
        else:
            khoa, toi_da = _khoa_gemini(), NOT_GIVEN
            client = _client_oai(khoa)
        try:
            tra = await client.chat.completions.create(
                model=model, messages=noi_dung(jpeg, cau_hoi), timeout=HET_GIO if cuoi else CHO_MODEL,
                max_tokens=toi_da)
            _qua_tai.pop(model, None)
            return (tra.choices[0].message.content or "").strip()
        except RateLimitError:
            if khoa and _het_luot_khoa(khoa):
                continue
            if cuoi:
                raise
            i += 1
        except NotFoundError:
            if cuoi:
                raise
            _cho_model_nghi(model, "404, model không còn")
            i += 1
        except (APITimeoutError, APIConnectionError, InternalServerError) as e:
            if not cuoi:
                _cho_model_nghi(model, "chậm quá" if isinstance(e, APITimeoutError) else type(e).__name__)
                i += 1
                continue
            goi_lai += 1
            if goi_lai > 1:
                raise


def _than_tts(cau: str) -> dict:
    # chỉ gửi đúng câu cần đọc: thêm lời dặn ("Đọc bằng tiếng Việt…") thì model đọc luôn cả lời dặn (PhoWhisper
    # chép lại tiếng 30/10 thấy rõ)
    return {"contents": [{"parts": [{"text": cau}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_GIONG}}}}}


def _doi_model_doc(ma: int) -> bool:
    """Mã HTTP của model đọc mà model kế có thể làm được: quá tải / lỗi máy chủ, model bị gỡ."""
    return ma >= 500 or ma == 404


async def doc_gemini(cau: str) -> bytes:
    """Gemini TTS: chữ -> PCM16 mono 24 kHz (chờ đủ cả câu). Model chính lỗi / chậm quá CHO_MODEL_DOC giây: đọc bằng
    model dự phòng; hết lượt: chuyển khóa, hết cả khóa thì sang model kế; model cuối lỗi mạng: gọi lại một lần."""
    ds = _thu_tu_model(GEMINI_TTS_MODEL, GEMINI_TTS_DU_PHONG)
    i, goi_lai = 0, 0
    while True:
        model, cuoi = ds[i], i == len(ds) - 1
        khoa = _khoa_gemini()
        try:
            r = await _client_http().post(GEMINI_TTS_URL.format(model=model, kieu="generateContent"),
                                          headers={"x-goog-api-key": khoa}, json=_than_tts(cau),
                                          timeout=HET_GIO if cuoi else CHO_MODEL_DOC)
        except (httpx.TimeoutException, httpx.NetworkError):
            if not cuoi:
                _cho_model_nghi(model, "chậm quá / lỗi mạng")
                i += 1
                continue
            goi_lai += 1
            if goi_lai > 1:
                raise
            continue
        if r.status_code == 429:
            if _het_luot_khoa(khoa):
                continue
            if not cuoi:
                i += 1
                continue
        elif _doi_model_doc(r.status_code) and not cuoi:
            _cho_model_nghi(model, r.status_code)
            i += 1
            continue
        r.raise_for_status()
        _qua_tai.pop(model, None)
        phan = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        return base64.b64decode(phan["data"])


async def doc_gemini_dan(cau: str):
    """Gemini TTS phát dần: sinh từng mẩu PCM16 24 kHz ngay khi tới (mẩu đầu ~1 s thay vì ~3,7 s chờ cả câu).
    Trước khi có mẩu nào: model chính lỗi / chậm thì đọc bằng model dự phòng; hết lượt (429): chuyển khóa dự phòng,
    hết cả khóa thì sang model kế; model cuối lỗi mạng: gọi lại một lần. Đã phát mẩu nào thì lỗi là báo lỗi luôn."""
    ds = _thu_tu_model(GEMINI_TTS_MODEL, GEMINI_TTS_DU_PHONG)
    i, lan = 0, 0
    while True:
        model, cuoi = ds[i], i == len(ds) - 1
        url = GEMINI_TTS_URL.format(model=model, kieu="streamGenerateContent") + "?alt=sse"
        da_co = False
        khoa = _khoa_gemini()
        try:
            async with _client_http().stream("POST", url, headers={"x-goog-api-key": khoa}, json=_than_tts(cau),
                                             timeout=HET_GIO if cuoi else CHO_MODEL_DOC) as r:
                r.raise_for_status()
                async for dong in r.aiter_lines():
                    if not dong.startswith("data:"):
                        continue
                    for p in json.loads(dong[5:])["candidates"][0]["content"].get("parts", []):
                        if "inlineData" in p:
                            da_co = True
                            yield base64.b64decode(p["inlineData"]["data"])
            _qua_tai.pop(model, None)
            return
        except httpx.HTTPStatusError as e:
            ma = e.response.status_code
            if da_co:
                raise
            if ma == 429:
                if _het_luot_khoa(khoa):
                    continue
                if not cuoi:
                    i += 1
                    continue
            elif _doi_model_doc(ma) and not cuoi:
                _cho_model_nghi(model, ma)
                i += 1
                continue
            raise
        except (httpx.TimeoutException, httpx.NetworkError):
            if da_co:
                raise
            if not cuoi:
                _cho_model_nghi(model, "chậm quá / lỗi mạng")
                i += 1
                continue
            lan += 1
            if lan > 1:
                raise


def tieng_tit(giay: float = 0.15, hz: float = 880.0) -> bytes:
    """Tiếng "tít" PCM16 mono 8 kHz: phát ngay khi nhận câu hỏi để người dùng biết kính đang xử lý."""
    n = int(8000 * giay)
    mau = (np.sin(2 * np.pi * hz * np.arange(n) / 8000) * 6000 * np.hanning(n)).astype("<i2")
    return mau.tobytes()


async def lam_nong():
    """Gọi trước một yêu cầu nhỏ lúc bật máy chủ: lần gọi đầu model nhìn chậm hẳn (đo 30/10: 4,5–9,4 s).
    Không làm nóng model đọc: gói miễn phí giới hạn số lần gọi TTS mỗi phút (429 khi gọi dồn, 30/10), để dành lượt
    cho câu hỏi thật. Model chính lỗi thì cho nó nghỉ và làm nóng model dự phòng: câu hỏi đầu đi thẳng sang đó.
    Model trên máy được làm nóng bằng một ảnh xám: lần hỏi ảnh đầu của llama.cpp Vulkan mất 16 s (dịch shader)."""
    ds = _thu_tu_model(GEMINI_MODEL, GEMINI_MODEL_DU_PHONG) if _ds_khoa() else []
    for i, model in enumerate(ds):
        try:
            await _client_oai(_khoa_gemini()).chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Chào"}], max_tokens=5,
                timeout=HET_GIO if i == len(ds) - 1 else CHO_MODEL)
            print(f"[QUAN_SAT] Đã làm nóng Gemini ({model})", flush=True)
            break
        except Exception as e:
            print(f"[QUAN_SAT] Làm nóng Gemini {model} lỗi (bỏ qua): {str(e)[:160]}", flush=True)
            if i < len(ds) - 1:
                _cho_model_nghi(model, "lỗi lúc làm nóng")
    if MAY_NHIN_URL:
        await lam_nong_tren_may()


def _client_may(timeout: float = HET_GIO):
    return AsyncOpenAI(api_key="khong-can", base_url=MAY_NHIN_URL, timeout=timeout, max_retries=0)


async def mo_ta_tren_may(jpeg, cau_hoi: str) -> str:
    """Chỉ hỏi model trên máy (nút Hỏi mặc định từ 2026-11-10): không gửi ảnh ra ngoài, không chờ Google.
    Lỗi kết nối / chậm: gọi lại một lần (llama-server vừa bật còn đang nạp model)."""
    lan = 0
    while True:
        try:
            tra = await _client_may().chat.completions.create(
                model=MAY_NHIN, messages=noi_dung(jpeg, cau_hoi), max_tokens=MAY_NHIN_TOKEN)
            return (tra.choices[0].message.content or "").strip()
        except (APITimeoutError, APIConnectionError, InternalServerError):
            lan += 1
            if lan > 1:
                raise


async def lam_nong_tren_may():
    """Một câu hỏi kèm ảnh xám: lần hỏi ảnh đầu của llama.cpp Vulkan mất 16 s (dịch shader)."""
    try:
        await _client_may(60).chat.completions.create(model=MAY_NHIN, max_tokens=5,
                                                      messages=noi_dung(_anh_xam(), "Chào"))
        print("[QUAN_SAT] Đã làm nóng model nhìn trên máy", flush=True)
    except Exception as e:
        print(f"[QUAN_SAT] Làm nóng model trên máy lỗi (bỏ qua, có thể chưa bật): {str(e)[:120]}", flush=True)


def _anh_xam(rong: int = 640, cao: int = 480) -> bytes:
    """JPEG xám đúng cỡ khung kính: shader của llama.cpp dịch theo cỡ ảnh."""
    import cv2
    return cv2.imencode(".jpg", np.full((cao, rong, 3), 128, np.uint8))[1].tobytes()


def cat_doan_noi(a, sr: int = 16000, khung: float = 0.05, nghi_toi_da: float = 0.5, dai_toi_thieu: float = 0.4):
    """Tìm các đoạn có người nói trong âm thanh mono `a` (float, [-1, 1]): [(bắt đầu giây, kết thúc giây)].

    Ngưỡng theo mức ồn nền của chính đoạn thu (trung vị năng lượng) nên chạy được trên video quay ngoài đường lẫn
    trong nhà. Quãng nghỉ ngắn hơn `nghi_toi_da` được nối lại (một câu); đoạn ngắn hơn `dai_toi_thieu` (tiếng động) bỏ."""
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
