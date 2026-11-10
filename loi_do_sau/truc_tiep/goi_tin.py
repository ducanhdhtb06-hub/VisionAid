"""Gói tin điện thoại -> máy: [float64 LE: thời điểm chụp (ms, Date.now())][uint32 LE: id khung][JPEG]."""
import struct

_DAU = struct.Struct('<dI')


def dong_goi(t_chup: float, id_khung: int, jpeg: bytes) -> bytes:
    return _DAU.pack(t_chup, id_khung) + jpeg


def mo_goi(b: bytes) -> tuple[float, int, bytes]:
    if len(b) < _DAU.size:
        raise ValueError(f'gói tin quá ngắn: {len(b)} byte')
    t_chup, id_khung = _DAU.unpack_from(b)
    return t_chup, id_khung, b[_DAU.size:]
