# uu_tien_cau.py
# -*- coding: utf-8 -*-
"""
Câu khẩn ngắt câu thường đang đọc.

Lỗi gốc (ESP32 ảo, test_out/esp32_ao/bao_cao_nhip.md "Phần trễ còn lại"): máy chủ ra câu sau 0,12–0,24 s, nhưng
"Dừng lại…" phải chờ câu thông tin đang đọc xong (tới 1,8 s). BoNgat: câu khẩn đến lúc đang phát câu THƯỜNG thì
cắt câu đó; câu khẩn không cắt câu khẩn (hai cảnh báo liên tiếp, cắt thì không nghe trọn câu nào); câu thường
không cắt gì. Thử iPhone 2026-10-26: câu bị cắt dính liền câu khẩn nghe như chèn nhau -> lay_vua_cat() cho
audio_player biết để chèn tiếng bíp trước câu khẩn.
"""
import threading
import unicodedata

TU_KHAN = ("dừng lại", "đang tới", "chắn vạch", "chắn giữa vạch", "bị chắn")   # "chắn vạch": báo vật chắn trên vạch (spec 2026-10-27)


def la_cau_khan(text: str) -> bool:
    t = unicodedata.normalize("NFC", text or "").lower()
    return any(k in t for k in TU_KHAN)


class BoNgat:
    def __init__(self):
        self._khoa = threading.Lock()
        self._dang_phat = False
        self._khan = False
        self._ngat = False
        self._vua_cat = False

    def bat_dau(self, khan: bool):
        with self._khoa:
            self._dang_phat, self._khan, self._ngat = True, khan, False

    def xong(self):
        with self._khoa:
            self._vua_cat = self._ngat
            self._dang_phat = self._ngat = False

    def lay_vua_cat(self) -> bool:
        """True (một lần) nếu câu vừa xong là bị cắt ngang."""
        with self._khoa:
            v, self._vua_cat = self._vua_cat, False
            return v

    def co_cau_moi(self, khan: bool) -> bool:
        with self._khoa:
            if khan and self._dang_phat and not self._khan:
                self._ngat = True
            return self._ngat

    def nen_dung(self) -> bool:
        return self._ngat
