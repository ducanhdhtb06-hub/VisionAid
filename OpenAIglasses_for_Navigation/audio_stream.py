# audio_stream.py
# -*- coding: utf-8 -*-
import asyncio
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Any, Dict
from fastapi import Request
from fastapi.responses import StreamingResponse

# ===== Tham số cơ bản của luồng WAV xuống =====
STREAM_SR = 8000  # Đổi sang 8kHz, ESP32 hỗ trợ
STREAM_CH = 1
STREAM_SW = 2
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000  # 320B (8kHz)

# ===== Công tắc tổng của tác vụ phát AI =====
current_ai_task: Optional[asyncio.Task] = None

async def cancel_current_ai():
    """Hủy tác vụ giọng nói của mô hình lớn hiện tại và chờ nó thoát."""
    global current_ai_task
    task = current_ai_task
    current_ai_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

def is_playing_now() -> bool:
    t = current_ai_task
    return (t is not None) and (not t.done())

# ===== Quản lý kết nối /stream.wav =====
@dataclass(frozen=True)
class StreamClient:
    q: asyncio.Queue
    abort_event: asyncio.Event

stream_clients: "Set[StreamClient]" = set()
STREAM_QUEUE_MAX = 96  # Bộ đệm nhỏ, tránh dồn ứ

def _wav_header_unknown_size(sr=16000, ch=1, sw=2) -> bytes:
    import struct
    byte_rate = sr * ch * sw
    block_align = ch * sw
    data_size = 0x7FFFFFF0
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, sw * 8,
        b"data", data_size
    )

async def hard_reset_audio(reason: str = ""):
    """
    **Dọn sạch một chạm**: bỏ mọi kết nối của trình phát (đặt abort_event) + hủy tác vụ AI hiện tại.
    Như vậy âm thanh cũ không còn nơi nào để đi, cũng không còn tác vụ nào tiếp tục sinh ra.
    """
    # 1) Ngắt mọi kết nối HTTP đang phát
    for sc in list(stream_clients):
        try:
            sc.abort_event.set()
        except Exception:
            pass
    stream_clients.clear()

    # 2) Hủy tác vụ AI hiện tại
    await cancel_current_ai()

    # 3) Ghi nhật ký
    if reason:
        print(f"[HARD-RESET] {reason}")

async def broadcast_pcm16_realtime(pcm16: bytes, dung=None):
    """Gửi pcm16 tới mọi kết nối còn sống theo nhịp 20ms; hàng đợi đầy thì bỏ phần đuôi để giữ thời gian thực.
    dung(): trả True thì cắt câu ngay (câu khẩn đến, uu_tien_cau.BoNgat) và xả tiếng còn chờ trong hàng đợi kết nối."""
    # [Mới] Ghi âm (ghi toàn bộ trước khi phân phối, tránh bị cắt mảnh)
    try:
        import sync_recorder
        sync_recorder.record_audio(pcm16, text="[Hội thoại Omni]")
    except Exception:
        pass  # Lỗi thầm lặng, không ảnh hưởng việc phát
    
    loop = asyncio.get_event_loop()
    next_tick = loop.time()
    off = 0
    while off < len(pcm16):
        if dung is not None and dung():
            for sc in list(stream_clients):
                try:
                    while True:
                        sc.q.get_nowait()
                except Exception:
                    pass
            break
        take = min(BYTES_PER_20MS_16K, len(pcm16) - off)
        piece = pcm16[off:off + take]

        dead: List[StreamClient] = []
        for sc in list(stream_clients):
            if sc.abort_event.is_set():
                dead.append(sc)
                continue
            try:
                if sc.q.full():
                    try: sc.q.get_nowait()
                    except Exception: pass
                sc.q.put_nowait(piece)
            except Exception:
                dead.append(sc)
        for sc in dead:
            try: stream_clients.discard(sc)
            except Exception: pass

        next_tick += 0.020
        now = loop.time()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
        off += take

# ===== Bộ đăng ký route FastAPI =====
def register_stream_route(app):
    @app.get("/stream.wav")
    async def stream_wav(_: Request):
        # —— Ép chỉ một kết nối (hoặc rất ít), ngắt hết các kết nối cũ trước ——
        for sc in list(stream_clients):
            try: sc.abort_event.set()
            except Exception: pass
        stream_clients.clear()

        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        sc = StreamClient(q=q, abort_event=abort_event)
        stream_clients.add(sc)

        async def gen():
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        break
                    if chunk:
                        yield chunk
            finally:
                stream_clients.discard(sc)
        return StreamingResponse(gen(), media_type="audio/wav")