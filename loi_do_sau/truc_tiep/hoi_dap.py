"""Nút "Hỏi" của app VisionAid: câu hỏi (chữ, iPhone tự nhận giọng) + khung camera mới nhất -> Gemini -> câu trả
lời (chữ) + giọng đọc (WAV). Dùng lại quan_sat.py của dự án (cùng model, chỉ dẫn, khóa dự phòng như module quan sát).

Ảnh camera và câu hỏi được gửi tới máy chủ Google. Khóa: GEMINI_API_KEY (và _2, _3) trong OpenAIglasses_for_Navigation/.env.
"""
import asyncio
import io
import os
import wave

DU_AN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     'OpenAIglasses_for_Navigation')
SR_GEMINI = 24000          # Gemini TTS trả PCM16 mono 24 kHz
# giây chờ Gemini đọc thành tiếng; quá (hoặc lỗi) thì trả câu chữ để điện thoại tự đọc bằng giọng tiếng Việt có sẵn:
# lúc Gemini quá tải / mất mạng, câu trả lời của model trên máy vẫn tới tai người dùng (2026-11-10)
CHO_DOC = 8.0     # bằng CHO_MODEL_DOC: model đọc chính chậm quá thì điện thoại đọc luôn, khỏi chờ model đọc dự phòng


def pcm24_sang_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR_GEMINI)
        w.writeframes(pcm)
    return buf.getvalue()


def co_khoa() -> bool:
    nap_env()
    return any(os.getenv(k, '').strip() for k in ('GEMINI_API_KEYS', 'GEMINI_API_KEY', 'GEMINI_API_KEY_2',
                                                  'GEMINI_API_KEY_3'))


def nap_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(DU_AN, '.env'))   # không ghi đè biến đã đặt sẵn


class HoiGemini:
    def __init__(self, quan_sat=None):
        nap_env()
        if quan_sat is None:
            import quan_sat                         # pyright: ignore[reportMissingImports]  (DU_AN đã ở sys.path)
        self.qs = quan_sat

    async def tra_loi(self, jpg: bytes, cau_hoi: str) -> tuple[str, bytes | None]:
        cau = await self.qs.mo_ta_gemini(jpg, cau_hoi)
        if not cau:
            return '', None
        # chờ đủ cả câu rồi gửi một WAV: trang ghép một câu với một WAV (đọc dần sẽ nhanh hơn ~2,7 s, làm sau)
        try:
            return cau, pcm24_sang_wav(await asyncio.wait_for(self.qs.doc_gemini(cau), CHO_DOC))
        except Exception as e:  # noqa: BLE001 – đọc lỗi không được làm mất câu trả lời
            print(f'[HOI] Gemini đọc lỗi ({e!r}): điện thoại tự đọc câu trả lời', flush=True)
            return cau, None

    async def lam_nong(self):
        await self.qs.lam_nong()                    # tự bắt lỗi và in ra, không ném


class HoiTrenMay(HoiGemini):
    """Nút Hỏi chỉ dùng model nhìn trên máy (llama-server + Qwen3-VL-2B), không gọi Gemini: trả (chữ, None) để điện
    thoại tự đọc ngay. Đo 2026-11-10: Gemini lúc quá tải làm một câu hỏi mất 12,8–13,7 s; máy 0,3–0,6 s.
    Ảnh camera không rời máy tính. Kém chính xác hơn Gemini (model 2B)."""

    async def tra_loi(self, jpg: bytes, cau_hoi: str) -> tuple[str, bytes | None]:
        return await self.qs.mo_ta_tren_may(jpg, cau_hoi), None

    async def lam_nong(self):
        await self.qs.lam_nong_tren_may()           # tự bắt lỗi và in ra, không ném
