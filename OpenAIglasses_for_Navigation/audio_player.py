# audio_player.py
# Xử lý phát các file âm thanh ghi sẵn, xuất qua loa ESP32

import os, unicodedata
import wave
import json
import asyncio
import threading
import queue
import numpy as np
import time
from audio_stream import broadcast_pcm16_realtime
from uu_tien_cau import BoNgat, la_cau_khan
from audio_compressor import compressed_audio_cache, AudioCompressor

# Nhập bộ ghi (tránh nhập vòng, nhập động khi cần)
_recorder_imported = False
_sync_recorder = None

def _get_recorder():
    """Nhập bộ ghi theo kiểu trì hoãn"""
    global _recorder_imported, _sync_recorder
    if not _recorder_imported:
        try:
            import sync_recorder as sr
            _sync_recorder = sr
            _recorder_imported = True
        except Exception as e:
            print(f"[AUDIO] Không nhập được bộ ghi: {e}")
            _recorder_imported = True  # Đánh dấu đã thử, tránh lặp lại
    return _sync_recorder

# Tương thích âm thanh mẫu của dự án cũ (giữ lại)
AUDIO_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")

# Mới: thư mục voice và bảng ánh xạ
# Dùng thư mục voice cạnh script, tránh vấn đề thư mục làm việc
VOICE_DIR = os.getenv("VOICE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice"))
VOICE_MAP_FILE = os.path.join(VOICE_DIR, "map.vi.json")

# Ánh xạ file âm thanh (sẽ gộp với ánh xạ voice)
AUDIO_MAP = {
    "Phát hiện vật thể": os.path.join(AUDIO_BASE_DIR, "音频1.wav"),
    "Lên": os.path.join(AUDIO_BASE_DIR, "音频2.wav"),
    "Xuống": os.path.join(AUDIO_BASE_DIR, "音频3.wav"),
    "Sang trái": os.path.join(AUDIO_BASE_DIR, "音频4.wav"),
    "Sang phải": os.path.join(AUDIO_BASE_DIR, "音频5.wav"),
    "OK": os.path.join(AUDIO_BASE_DIR, "音频6.wav"),
    "Tiến lên": os.path.join(AUDIO_BASE_DIR, "音频7.wav"),
    "Lùi lại": os.path.join(AUDIO_BASE_DIR, "音频8.wav"),
    "Đã lấy được vật": os.path.join(AUDIO_BASE_DIR, "音频9.wav"),
}

# Bộ đệm âm thanh, tránh đọc lặp
_audio_cache = {}

# Hàng đợi phát âm thanh và luồng xử lý - dùng hàng đợi ưu tiên
_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0  # Bộ đếm ưu tiên tăng dần
_worker_thread = None
_worker_loop = None
_is_playing = False  # Đánh dấu có đang phát âm thanh hay không
_bo_ngat = BoNgat()  # câu khẩn ("Dừng lại…", "…đang tới") cắt câu thường đang đọc; AIGLASS_NGAT_CAU=0 tắt
_NGAT_CAU = os.getenv("AIGLASS_NGAT_CAU", "1") != "0"
# Thử iPhone 2026-10-26: câu nối đuôi nhau không quãng nghỉ, câu bị cắt dính liền câu khẩn -> nghe như chèn nhau.
# Giữa hai câu: im lặng >= 0,3 s. Câu khẩn ngay sau khi cắt: bíp 880 Hz 0,12 s + im 0,08 s thay cho quãng nghỉ.
QUANG_NGHI_MS = 300
_BIP = (np.sin(2 * np.pi * 880 * np.arange(960) / 8000) * 6000 * np.hanning(960)).astype("<i2").tobytes() \
    + b"\x00" * 1280
_playing_lock = threading.Lock()  # Khóa phát
_initialized = False
_last_play_ts = 0.0  # Ghi lại thời điểm kết thúc lần phát trước, dùng để quyết định độ dài im lặng khởi động

def load_wav_file(filepath):
    """Nạp file WAV và trả về dữ liệu PCM (tự động chuyển sang 8kHz)"""
    if filepath in _audio_cache:
        return _audio_cache[filepath]
    
    # Dùng bộ đệm nén
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        compressed_data = compressed_audio_cache.load_and_compress(filepath)
        if compressed_data:
            # Lưu dữ liệu đã nén
            _audio_cache[filepath] = compressed_data
            return compressed_data
    
    # Cách nạp gốc (không nén)
    try:
        with wave.open(filepath, 'rb') as wav:
            # Kiểm tra định dạng âm thanh
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            
            if channels != 1:
                print(f"[AUDIO] Cảnh báo: {filepath} không phải đơn kênh, chỉ dùng kênh đầu tiên")
            if sampwidth != 2:
                print(f"[AUDIO] Cảnh báo: {filepath} không phải âm thanh 16 bit")
            
            # Đọc toàn bộ khung
            frames = wav.readframes(wav.getnframes())
            
            # Nếu là stereo thì chỉ lấy kênh trái
            if channels == 2:
                import audioop
                frames = audioop.tomono(frames, sampwidth, 1, 0)
            
            # Thống nhất chuyển sang 8kHz (dùng ratecv để giữ nguyên cao độ và tốc độ)
            if framerate != 8000:
                import audioop
                frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                print(f"[AUDIO] Lấy mẫu lại: {filepath} {framerate}Hz -> 8000Hz")
            
            _audio_cache[filepath] = frames
            return frames
            
    except Exception as e:
        print(f"[AUDIO] Nạp file âm thanh thất bại {filepath}: {e}")
        return None

def _merge_voice_map():
    """Đọc voice/map.vi.json và gộp vào AUDIO_MAP"""
    try:
        if not os.path.exists(VOICE_MAP_FILE):
            print(f"[AUDIO] Không tìm thấy file ánh xạ: {VOICE_MAP_FILE}")
            return
        with open(VOICE_MAP_FILE, "r", encoding="utf-8") as f:
            m = json.load(f)
        added = 0
        for text, info in (m or {}).items():
            files = (info or {}).get("files") or []
            if not files:
                continue
            fname = files[0]
            fpath = os.path.join(VOICE_DIR, fname)
            if os.path.exists(fpath):
                AUDIO_MAP[unicodedata.normalize("NFC", text)] = fpath
                added += 1
            else:
                print(f"[AUDIO] Thiếu file ánh xạ: {fpath}")
        print(f"[AUDIO] Đã gộp ánh xạ voice {added} mục")
    except Exception as e:
        print(f"[AUDIO] Đọc ánh xạ voice thất bại: {e}")

def preload_all_audio():
    """Nạp sẵn mọi file âm thanh vào bộ nhớ"""
    print("[AUDIO] Bắt đầu nạp sẵn các file âm thanh...")
    loaded_count = 0
    
    # [Tạm tắt đổi tốc độ] vì cần sửa cơ chế bộ đệm
    # Danh sách âm thanh cần tăng tốc (liên quan đến vạch qua đường)
    # speedup_keywords = ["vạch qua đường", "khung hình"]
    # speedup_factor = 1.3  # tăng tốc 30%
    
    for audio_key, filepath in AUDIO_MAP.items():
        if os.path.exists(filepath):
            # [Sửa lỗi] Tạm thời nạp với tốc độ mặc định
            # need_speedup = any(keyword in audio_key for keyword in speedup_keywords)
            # speed = speedup_factor if need_speedup else 1.0
            
            data = load_wav_file(filepath)  # Dùng tham số mặc định
            if data:
                loaded_count += 1
                # if need_speedup:
                #     print(f"[AUDIO] Nạp (tăng tốc {speedup_factor}x): {audio_key}")
        else:
            # Giảm bớt đầu ra gây nhiễu
            pass
    print(f"[AUDIO] Nạp sẵn xong, tổng cộng đã nạp {loaded_count} file âm thanh")

def _audio_worker():
    """Luồng xử lý phát âm thanh"""
    global _worker_loop
    
    # Thử đặt mức ưu tiên luồng (đặc thù Windows)
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            # Đặt luồng ở mức ưu tiên cao
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                1  # THREAD_PRIORITY_ABOVE_NORMAL
            )
            print("[AUDIO] Đặt luồng âm thanh ở mức ưu tiên cao")
    except Exception as e:
        print(f"[AUDIO] Đặt mức ưu tiên luồng thất bại: {e}")
    
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)
    
    async def process_queue():
        while True:
            try:
                # Lấy dữ liệu từ hàng đợi ưu tiên
                priority_data = await asyncio.get_event_loop().run_in_executor(None, _audio_queue.get, True)
                if priority_data is None:
                    break
                # Tách mức ưu tiên và dữ liệu âm thanh thực
                if isinstance(priority_data, tuple) and len(priority_data) == 2:
                    _, audio_data = priority_data
                else:
                    audio_data = priority_data
                khan = False
                if isinstance(audio_data, tuple):
                    audio_data, khan = audio_data
                await _broadcast_audio_optimized(audio_data, khan)
            except Exception as e:
                print(f"[AUDIO] Lỗi luồng xử lý: {e}")
    
    _worker_loop.run_until_complete(process_queue())

async def _broadcast_audio_optimized(pcm_data: bytes, khan: bool = False):
    """Phát âm thanh tối ưu: mỗi lần gọi do lớp dưới gửi theo nhịp 20ms, bỏ nhịp lặp và sleep ở tầng Python"""
    global _last_play_ts, _is_playing
    try:
        # Đặt cờ đang phát
        with _playing_lock:
            _is_playing = True
        _bo_ngat.bat_dau(khan)
        # Lúc này pcm_data đã là dữ liệu PCM 16 bit đã giải nén (8kHz)
        now = time.monotonic()
        idle_sec = now - (_last_play_ts or now)
        vua_cat = _bo_ngat.lay_vua_cat()
        # Lần đầu hoặc sau thời gian nhàn rỗi dài thì khởi động bằng im lặng dài hơn; câu nối tiếp câu khác: đủ
        # quãng nghỉ QUANG_NGHI_MS (tính cả lúc đã im); ngay sau khi bị cắt: bíp thay quãng nghỉ
        if idle_sec > 3.0:
            lead_ms = 160
        elif vua_cat:
            lead_ms = 60
        else:
            lead_ms = max(60, QUANG_NGHI_MS - int(idle_sec * 1000))
        tail_ms = 40

        lead_silence = b'\x00' * (lead_ms * 8000 * 2 // 1000)  # 8k * 2B
        if vua_cat:
            lead_silence += _BIP
        tail_silence = b'\x00' * (tail_ms * 8000 * 2 // 1000)

        # Dữ liệu âm thanh đầy đủ (gồm cả im lặng)
        full_audio = lead_silence + pcm_data + tail_silence
        
        # Lưu ý: việc ghi âm được thực hiện thống nhất trong broadcast_pcm16_realtime, tránh ghi trùng

        # Mỗi lần gọi giao cho lớp dưới điều nhịp (nhịp 20ms được cài trong broadcast_pcm16_realtime)
        await broadcast_pcm16_realtime(full_audio, dung=_bo_ngat.nen_dung)

        _last_play_ts = time.monotonic()
    except Exception as e:
        print(f"[AUDIO] Phát âm thanh thất bại: {e}")
    finally:
        # Xóa cờ đang phát
        _bo_ngat.xong()
        with _playing_lock:
            _is_playing = False

def initialize_audio_system():
    """Khởi tạo hệ thống âm thanh"""
    global _initialized, _worker_thread, _last_play_ts
    
    if _initialized:
        return
    
    # Gộp ánh xạ voice trước rồi mới nạp sẵn
    _merge_voice_map()
    preload_all_audio()
    
    _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
    _worker_thread.start()
    _initialized = True
    _last_play_ts = 0.0
    
    # Hiển thị thống kê nén
    if os.getenv("AIGLASS_COMPRESS_AUDIO", "1") == "1":
        stats = compressed_audio_cache.get_compression_stats()
        print(f"[AUDIO] Thống kê nén âm thanh:")
        print(f"  - Số file: {stats['files_cached']}")
        print(f"  - Kích thước gốc: {stats['total_original_size'] / 1024:.1f} KB")
        print(f"  - Sau khi nén: {stats['total_compressed_size'] / 1024:.1f} KB")
        print(f"  - Tỷ lệ nén: {stats['compression_ratio']:.1%}")
        print(f"  - Tiết kiệm: {stats['bytes_saved'] / 1024:.1f} KB")
    
    print("[AUDIO] Khởi tạo hệ thống âm thanh xong (nạp sẵn + luồng xử lý)")

def play_audio_threadsafe(audio_key):
    """Hàm phát âm thanh an toàn đa luồng"""
    global _audio_queue, _audio_priority
    
    if not _initialized:
        initialize_audio_system()
    
    if audio_key not in AUDIO_MAP:
        print(f"[AUDIO] Khóa âm thanh không xác định: {audio_key}")
        return
    
    filepath = AUDIO_MAP[audio_key]
    pcm_data = _audio_cache.get(filepath)
    if pcm_data is None:
        print(f"[AUDIO] Âm thanh chưa có trong bộ đệm: {audio_key}")
        return
    
    # Nếu là dữ liệu nén thì giải nén trước
    if pcm_data and len(pcm_data) > 5 and pcm_data[0] in [0x01, 0x02]:
        pcm_data = compressed_audio_cache.decompress(pcm_data)
        if not pcm_data:
            print(f"[AUDIO] Giải nén thất bại: {audio_key}")
            return
    
    khan = _NGAT_CAU and la_cau_khan(audio_key)
    if khan and _bo_ngat.co_cau_moi(True):
        # câu khẩn: cắt câu thường đang đọc (luồng phát dừng ở nhịp 20 ms kế tiếp) và bỏ câu đang chờ
        print(f"[AUDIO] Câu khẩn, cắt câu đang đọc: {audio_key}")
        try:
            while True:
                _audio_queue.get_nowait()
        except queue.Empty:
            pass

    # [Tối ưu] Chiến lược thông báo thời gian thực: giữ hàng đợi ở mức tối thiểu, tránh độ trễ dồn ứ
    queue_size = _audio_queue.qsize()
    
    # Kiểm tra có đang phát hay không
    with _playing_lock:
        currently_playing = _is_playing
    
    # Chiến lược thời gian thực: chỉ cho phép tồn 1 mục, vượt quá thì xóa ngay
    if queue_size > 0 and not currently_playing:
        # Khi chưa phát thì xóa ngay, phát giọng nói mới nhất
        print(f"[AUDIO] Xóa hàng đợi (hiện có {queue_size} mục), phát giọng nói mới nhất")
        _audio_queue = queue.PriorityQueue(maxsize=10)
    elif queue_size > 1 and currently_playing:
        # Khi đang phát mà tồn đọng >1 mục thì xóa (giữ tính thời gian thực)
        print(f"[AUDIO] Hàng đợi tồn đọng ({queue_size} mục), xóa để giữ thời gian thực")
        _audio_queue = queue.PriorityQueue(maxsize=10)
    try:
        # Dùng hàng đợi ưu tiên, đảm bảo âm thanh phát đúng thứ tự
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, (pcm_data, khan)))
        if queue_size >= 1:
            print(f"[AUDIO] Kích thước hàng đợi phát hiện tại: {queue_size + 1}")
    except queue.Full:
        # Hàng đợi phát đầy thì bỏ, giữ tính thời gian thực
        print(f"[AUDIO] Hàng đợi đầy, bỏ: {audio_key}")
        pass

# Giới hạn tần suất giọng nói toàn cục
_last_voice_time = 0
_last_voice_text = ""
_voice_cooldown = 1.0  # Cùng một giọng nói phải cách nhau ít nhất 1 giây

# Định nghĩa ưu tiên giọng nói
VOICE_PRIORITY = {
    'obstacle': 100,     # Vật cản - ưu tiên cao nhất
    'direction': 50,     # Rẽ/dịch ngang - ưu tiên trung bình
    'straight': 10,      # Giữ hướng đi thẳng - ưu tiên thấp nhất
    'other': 30          # Khác - ưu tiên mặc định
}

# Mới: phát trực tiếp theo nội dung nhắc bằng tiếng Trung (có chuẩn hóa nhẹ và hạ cấp)
def _phat_cau(text: str) -> str:
    """
    Nhận câu nhắc (tiếng Việt), tự động khớp ánh xạ voice và phát.
    - Thử nguyên văn
    - Thử thêm/bỏ dấu câu cuối câu (. ! ?)
    - Nếu chứa "Phía trước có… chú ý tránh" mà không khớp, hạ cấp về "Phía trước có vật cản, chú ý tránh."
    """
    global _last_voice_time, _last_voice_text
    
    if not text:
        return "rong"
    if not _initialized:
        initialize_audio_system()
    
    # Giới hạn toàn cục: cùng một văn bản không phát lặp trong thời gian ngắn
    current_time = time.time()
    if text == _last_voice_text and current_time - _last_voice_time < _voice_cooldown:
        return "trung_lap"  # Bỏ qua thầm lặng

    candidates = []
    t = unicodedata.normalize("NFC", text.strip())
    candidates.append(t)
    # Thử thêm dấu chấm cuối câu
    if t[-1:] not in ("。", "！", "!", "？", "?", "."):
        candidates.append(t + ".")
    else:
        # Thử bỏ dấu câu
        t2 = t.rstrip("。.!！?？")
        if t2 and t2 != t:
            candidates.append(t2)

    # Lần lượt thử khớp
    for ck in candidates:
        if ck in AUDIO_MAP:
            play_audio_threadsafe(ck)
            _last_voice_text = text
            _last_voice_time = current_time
            return ""

    # Hạ cấp cho câu kiểu "Phía trước có… chú ý tránh"
    if ("Phía trước có" in t) and ("chú ý tránh" in t):
        fallback = "Phía trước có vật cản, chú ý tránh."
        if fallback in AUDIO_MAP:
            play_audio_threadsafe(fallback)
            _last_voice_text = text
            _last_voice_time = current_time
            return ""

    # Thử các biến thể thường gặp cho các mục kiểu "Hãy dịch/chỉnh nhẹ/xoay sang…"
    base = t.rstrip("。.!！?？")
    if base in AUDIO_MAP:
        play_audio_threadsafe(base)
        _last_voice_text = text
        _last_voice_time = current_time
        return ""
    if base + "." in AUDIO_MAP:
        play_audio_threadsafe(base + ".")
        _last_voice_text = text
        _last_voice_time = current_time
        return ""

    # Nếu không khớp thì ghi nhật ký (tiện gỡ lỗi)
    print(f"[AUDIO] Không tìm thấy giọng nói khớp: {text}")
    return "khong_co_wav"


def play_voice_text(text: str):
    """Phát câu nhắc (xem _phat_cau). AIGLASS_NHAT_KY_CAU=<file>: ghi mỗi lần gọi một dòng JSON
    {t, cau, phat, ly_do} - chỉ dùng khi thử (ESP32 ảo chấm điểm: luồng /stream.wav chỉ có tiếng, không có chữ)."""
    ly_do = _phat_cau(text)
    duong_dan = os.getenv("AIGLASS_NHAT_KY_CAU")
    if duong_dan and ly_do != "rong":
        try:
            with open(duong_dan, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(t=time.time(), cau=text, phat=ly_do == "", ly_do=ly_do), ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[AUDIO] Không ghi được nhật ký câu: {e}")

# Tương thích giao diện cũ
play_audio_on_esp32 = play_audio_threadsafe