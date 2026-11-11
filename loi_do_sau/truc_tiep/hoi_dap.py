"""Nút "Hỏi" của app VisionAid: câu hỏi (chữ, điện thoại tự nhận giọng) + khung camera mới nhất -> mô hình nhìn chạy
trên máy (Qwen3-VL-2B qua llama-server, quan_sat.mo_ta_tren_may) -> câu trả lời (chữ). Điện thoại tự đọc câu trả lời
bằng giọng tiếng Việt có sẵn. Ảnh camera không rời máy tính, không cần khóa API.
2026-11-11: đã gỡ Gemini; chỉ dùng mô hình trên máy.
"""


class HoiTrenMay:
    """Đo 2026-11-10 (RTX 4060): 0,3–0,6 s một câu. Kém chi tiết hơn các mô hình lớn trên mạng (model 2B)."""

    def __init__(self, quan_sat=None):
        if quan_sat is None:
            import quan_sat                         # pyright: ignore[reportMissingImports]  (dự án đã ở sys.path)
        self.qs = quan_sat

    async def tra_loi(self, jpg: bytes, cau_hoi: str) -> tuple[str, bytes | None]:
        """(chữ, None): không có WAV, điện thoại đọc chữ ngay."""
        return await self.qs.mo_ta_tren_may(jpg, cau_hoi), None

    async def lam_nong(self):
        await self.qs.lam_nong_tren_may()           # tự bắt lỗi và in ra, không ném
