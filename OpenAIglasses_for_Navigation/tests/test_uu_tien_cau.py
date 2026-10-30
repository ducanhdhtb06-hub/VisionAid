"""Test uu_tien_cau.py + broadcast_pcm16_realtime(dung=...): câu khẩn ngắt câu thường đang đọc.
Lỗi gốc (ESP32 ảo, test_out/esp32_ao/bao_cao_nhip.md "Phần trễ còn lại"): "Dừng lại…" chờ câu thường đọc xong tới 1,8 s.
Chạy: ../.venv/bin/python -m pytest tests/test_uu_tien_cau.py -q  (đứng ở OpenAIglasses_for_Navigation/)
"""
import asyncio

import pytest

import audio_stream
from uu_tien_cau import BoNgat, la_cau_khan


@pytest.mark.parametrize("cau", ["Dừng lại. Xe máy, 2 mét", "Xe máy, bên trái, đang tới",
                                 "Mất hình ảnh từ kính, hãy dừng lại.", "Phía trước có người, dừng lại.",
                                 "Xe máy chắn vạch bên phải. Đi sang trái.", "Xe máy chắn giữa vạch. Đi sang trái.",
                                 "Vạch bị chắn, dừng lại, hãy dùng gậy."])
def test_la_cau_khan(cau):
    assert la_cau_khan(cau)


@pytest.mark.parametrize("cau", ["Dịch sang phải", "Được rồi, hãy dừng dịch ngang.", "Đã dừng dẫn đường.",
                                 "Chưa thấy vạch dẫn đường.", "", "Đã về vạch.", "Vạch ở bên phải."])
def test_khong_khan(cau):
    assert not la_cau_khan(cau)


def test_khan_ngat_thuong():
    b = BoNgat()
    b.bat_dau(khan=False)
    assert b.co_cau_moi(khan=True) is True
    assert b.nen_dung()


def test_khan_khong_ngat_khan():
    b = BoNgat()
    b.bat_dau(khan=True)
    assert b.co_cau_moi(khan=True) is False
    assert not b.nen_dung()


def test_thuong_khong_ngat():
    b = BoNgat()
    b.bat_dau(khan=False)
    assert b.co_cau_moi(khan=False) is False
    b.bat_dau(khan=True)
    assert b.co_cau_moi(khan=False) is False
    assert not b.nen_dung()


def test_ngat_xong_phat_tiep():
    b = BoNgat()
    b.bat_dau(khan=False)
    b.co_cau_moi(khan=True)
    b.xong()
    b.bat_dau(khan=True)
    assert not b.nen_dung()


def test_vua_cat_thi_bao_mot_lan():
    # thử iPhone 2026-10-26: câu bị cắt dính liền câu khẩn, nghe như hai câu chèn nhau -> câu khẩn sau khi cắt
    # phải có tiếng bíp mở đầu
    b = BoNgat()
    b.bat_dau(khan=False)
    b.co_cau_moi(khan=True)
    b.xong()
    assert b.lay_vua_cat() is True
    assert b.lay_vua_cat() is False


def test_xong_binh_thuong_khong_bao_cat():
    b = BoNgat()
    b.bat_dau(khan=False)
    b.xong()
    assert b.lay_vua_cat() is False


def test_khong_phat_gi_thi_khong_ngat():
    b = BoNgat()
    assert b.co_cau_moi(khan=True) is False
    b.bat_dau(khan=False)
    b.xong()
    assert b.co_cau_moi(khan=True) is False


def test_broadcast_dung_giua_chung(monkeypatch):
    async def f():
        q = asyncio.Queue(maxsize=1000)
        sc = audio_stream.StreamClient(q=q, abort_event=asyncio.Event())
        monkeypatch.setattr(audio_stream, "stream_clients", {sc})
        dem = {"n": 0}

        def dung():
            dem["n"] += 1
            return dem["n"] > 5
        await audio_stream.broadcast_pcm16_realtime(b"\x00" * 16000, dung=dung)   # 1 s ở 8 kHz 16 bit
        return dem["n"], q.qsize()
    so_lan, con_lai = asyncio.run(f())
    assert so_lan <= 7
    assert con_lai == 0


# ---- nối vào audio_player: luồng phát thật, broadcast giả ghi lại từng mảnh 20 ms ----
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import audio_player as AP  # noqa: E402

THUONG, KHAN = "Dịch sang phải", "Dừng lại. Xe máy, 2 mét"


class _Manh(list):
    goi: list


@pytest.fixture
def loa(monkeypatch):
    manh = _Manh()
    manh.goi = []                          # pcm của từng lần gọi broadcast (một câu)

    async def phat_gia(pcm, dung=None):
        manh.goi.append(pcm)
        for i in range(0, len(pcm), 320):
            if dung is not None and dung():
                break
            manh.append((time.time(), pcm[i + 320 - 1] if i + 320 <= len(pcm) else pcm[-1]))
            await asyncio.sleep(0.02)

    monkeypatch.setattr(AP, "broadcast_pcm16_realtime", phat_gia)
    monkeypatch.setattr(AP, "AUDIO_MAP", {THUONG: "a", KHAN: "b", "Dịch sang trái": "c"})
    monkeypatch.setattr(AP, "_audio_cache", {"a": b"\x11" * 16000, "b": b"\x22" * 4800,   # 1 s, 0,3 s
                                             "c": b"\x33" * 4800})
    monkeypatch.setattr(AP, "_audio_queue", queue.PriorityQueue(maxsize=10))
    monkeypatch.setattr(AP, "_initialized", True)
    monkeypatch.setattr(AP, "_bo_ngat", BoNgat())
    threading.Thread(target=AP._audio_worker, daemon=True).start()
    yield manh
    try:                                   # xả câu còn chờ rồi mới gửi None (None không so được với tuple)
        while True:
            AP._audio_queue.get_nowait()
    except queue.Empty:
        pass
    AP._audio_queue.put(None)
    time.sleep(0.05)


def _bat_dau(manh, ma):
    return next((t for t, b in manh if b == ma), None)


def test_audio_player_khan_cat_cau_thuong(loa, monkeypatch):
    monkeypatch.setattr(AP, "_NGAT_CAU", True)
    AP.play_audio_threadsafe(THUONG)
    time.sleep(0.25)
    t_goi = time.time()
    AP.play_audio_threadsafe(KHAN)
    time.sleep(0.8)
    t_khan = _bat_dau(loa, 0x22)
    assert t_khan is not None and t_khan - t_goi < 0.35     # cắt <= 20 ms + bíp 0,2 s rồi mới tới câu khẩn
    assert sum(1 for _, b in loa if b == 0x11) < 25        # câu thường (50 mảnh) bị cắt


def test_audio_player_tat_ngat_thi_cho(loa, monkeypatch):
    monkeypatch.setattr(AP, "_NGAT_CAU", False)
    AP.play_audio_threadsafe(THUONG)
    time.sleep(0.25)
    t_goi = time.time()
    AP.play_audio_threadsafe(KHAN)
    time.sleep(1.6)
    assert _bat_dau(loa, 0x22) - t_goi > 0.5


def test_cau_khan_sau_khi_cat_co_bip(loa, monkeypatch):
    monkeypatch.setattr(AP, "_NGAT_CAU", True)
    AP.play_audio_threadsafe(THUONG)
    time.sleep(0.25)
    AP.play_audio_threadsafe(KHAN)
    time.sleep(0.8)
    pcm = loa.goi[1]
    dau = pcm[:pcm.index(b"\x22")]        # phần trước nội dung câu khẩn
    assert any(x not in (0,) for x in dau), "phải có tiếng bíp trước câu khẩn"


def test_hai_cau_lien_nhau_co_quang_nghi(loa, monkeypatch):
    monkeypatch.setattr(AP, "_NGAT_CAU", True)
    AP.play_audio_threadsafe(THUONG)
    time.sleep(0.05)
    AP.play_audio_threadsafe("Dịch sang trái")           # câu thường xếp hàng sau câu thường
    time.sleep(2.6)
    assert len(loa.goi) == 2
    # >= 0,25 s im lặng trước câu thứ hai (8 kHz, 16 bit). Quãng nghỉ là 0,3 s TÍNH CẢ thời gian đã im (lead_ms = 300 - idle),
    # nên khi máy bận, luồng phát trễ vài ms thì phần im nối thêm ngắn đúng chừng đó: dung sai 50 ms (đo 30/10: lỗi 16 byte)
    assert loa.goi[1][:4000] == bytes(4000)
