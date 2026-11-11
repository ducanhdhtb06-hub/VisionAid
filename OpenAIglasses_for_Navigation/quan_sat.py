# -*- coding: utf-8 -*-
"""Quan sát môi trường: ảnh camera + câu hỏi của người dùng -> mô hình nhìn chạy TRÊN MÁY -> câu trả lời (chữ).

Mô hình duy nhất: Qwen3-VL-2B-Instruct GGUF Q4_K_M (Apache-2.0) chạy bằng llama.cpp llama-server (MIT), cổng tương thích
OpenAI ở AIGLASS_MAY_NHIN_URL (mặc định http://127.0.0.1:8090/v1; máy chủ app chay.py tự bật). Ảnh không rời máy, không
cần khóa API, không trả phí theo lượt. Đo 2026-11-10 trên RTX 4060 (Vulkan): 0,25–0,63 s một câu.
Câu trả lời là chữ: điện thoại (app VisionAid) tự đọc bằng giọng tiếng Việt có sẵn.
2026-11-11: đã gỡ Gemini (nhìn + đọc) và Qwen-Omni; chỉ dùng mô hình trên máy.
"""
import base64
import os

import numpy as np
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError

MAY_NHIN_URL = os.getenv("AIGLASS_MAY_NHIN_URL", "http://127.0.0.1:8090/v1")
MAY_NHIN = "tren-may"                            # tên model gửi đi (llama-server bỏ qua tên model)
MAY_NHIN_TOKEN = 150                             # chặn câu dài lan man của model nhỏ (chỉ dẫn đã dặn tối đa hai câu)
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


def _client_may(timeout: float = HET_GIO):
    return AsyncOpenAI(api_key="khong-can", base_url=MAY_NHIN_URL, timeout=timeout, max_retries=0)


async def mo_ta_tren_may(jpeg, cau_hoi: str) -> str:
    """Hỏi model nhìn trên máy: ảnh + câu hỏi -> câu trả lời (chữ). Ảnh không gửi ra ngoài.
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


def tieng_tit(giay: float = 0.15, hz: float = 880.0) -> bytes:
    """Tiếng "tít" PCM16 mono 8 kHz: phát ngay khi nhận câu hỏi để người dùng biết kính đang xử lý."""
    n = int(8000 * giay)
    mau = (np.sin(2 * np.pi * hz * np.arange(n) / 8000) * 6000 * np.hanning(n)).astype("<i2")
    return mau.tobytes()


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
