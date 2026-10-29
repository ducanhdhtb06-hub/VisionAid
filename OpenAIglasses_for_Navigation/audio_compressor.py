# audio_compressor.py
# -*- coding: utf-8 -*-
"""
Công cụ nén âm thanh - dùng để giảm chiếm dụng băng thông mạng
Hỗ trợ nén PCM 16kHz 16bit sang định dạng nhỏ hơn
"""
import os
import wave
import struct
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioCompressor:
    """Bộ nén âm thanh - hỗ trợ nhiều thuật toán nén"""
    
    @staticmethod
    def pcm16_to_ulaw(pcm_data: bytes) -> bytes:
        """
        Chuyển PCM 16 bit sang μ-law 8 bit
        Tỷ lệ nén: 50% (16bit -> 8bit)
        """
        # Phân tích PCM 16 bit
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Nén μ-law
        ulaw_data = bytearray()
        for sample in samples:
            ulaw_byte = AudioCompressor._linear_to_ulaw(sample)
            ulaw_data.append(ulaw_byte)
        
        return bytes(ulaw_data)
    
    @staticmethod
    def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
        """
        Chuyển μ-law 8 bit ngược lại PCM 16 bit
        """
        pcm_samples = []
        for ulaw_byte in ulaw_data:
            pcm_sample = AudioCompressor._ulaw_to_linear(ulaw_byte)
            pcm_samples.append(pcm_sample)
        
        return np.array(pcm_samples, dtype=np.int16).tobytes()
    
    @staticmethod
    def _linear_to_ulaw(sample: int) -> int:
        """
        PCM tuyến tính 16 bit sang μ-law
        """
        # Bảng mã hóa μ-law
        ULAW_MAX = 0x1FFF
        ULAW_BIAS = 0x84
        
        # Giới hạn phạm vi
        sample = max(-32768, min(32767, sample))
        
        # Lấy bit dấu
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample
        
        # Thêm độ lệch
        sample = sample + ULAW_BIAS
        
        # Giới hạn giá trị lớn nhất
        if sample > ULAW_MAX:
            sample = ULAW_MAX
        
        # Tìm số mũ và phần định trị
        exponent = 7
        for exp in range(7, -1, -1):
            if sample & (0x4000 >> exp):
                exponent = exp
                break
        
        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulawbyte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        
        return ulawbyte
    
    @staticmethod
    def _ulaw_to_linear(ulawbyte: int) -> int:
        """
        μ-law sang PCM tuyến tính 16 bit
        """
        ULAW_BIAS = 0x84
        
        ulawbyte = ~ulawbyte & 0xFF
        sign = ulawbyte & 0x80
        exponent = (ulawbyte >> 4) & 0x07
        mantissa = ulawbyte & 0x0F
        
        sample = ((mantissa << 3) + ULAW_BIAS) << exponent
        
        if sign:
            sample = -sample
            
        return sample
    
    @staticmethod
    def pcm16_to_adpcm(pcm_data: bytes) -> bytes:
        """
        Chuyển PCM 16 bit sang ADPCM 4 bit
        Tỷ lệ nén: 75% (16bit -> 4bit)
        Giữ chất lượng giọng nói khá tốt
        """
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Bảng bước IMA ADPCM
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]
        
        # Bảng điều chỉnh chỉ số
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]
        
        # Khởi tạo
        adpcm_data = bytearray()
        predicted = 0
        step_index = 0
        
        # Cứ hai mẫu đóng gói thành một byte
        for i in range(0, len(samples), 2):
            byte = 0
            
            for j in range(2):
                if i + j < len(samples):
                    sample = samples[i + j]
                    
                    # Tính chênh lệch
                    diff = sample - predicted
                    
                    # Lượng tử hóa
                    step = step_table[step_index]
                    adpcm_sample = 0
                    
                    if diff < 0:
                        adpcm_sample = 8
                        diff = -diff
                    
                    if diff >= step:
                        adpcm_sample |= 4
                        diff -= step
                        
                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 2
                        diff -= step
                        
                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 1
                    
                    # Cập nhật giá trị dự đoán
                    step = step_table[step_index]
                    diff = 0
                    if adpcm_sample & 4:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 2:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 1:
                        diff += step
                    step >>= 1
                    diff += step
                    
                    if adpcm_sample & 8:
                        predicted -= diff
                    else:
                        predicted += diff
                    
                    # Giới hạn phạm vi giá trị dự đoán
                    if predicted > 32767:
                        predicted = 32767
                    elif predicted < -32768:
                        predicted = -32768
                    
                    # Cập nhật chỉ số bước
                    step_index += index_table[adpcm_sample & 7]
                    if step_index < 0:
                        step_index = 0
                    elif step_index > 88:
                        step_index = 88
                    
                    # Đóng gói vào byte
                    if j == 0:
                        byte = adpcm_sample
                    else:
                        byte |= (adpcm_sample << 4)
            
            adpcm_data.append(byte)
        
        # Thêm thông tin đầu: giá trị dự đoán ban đầu và chỉ số bước
        header = struct.pack('<hB', predicted, step_index)
        return header + bytes(adpcm_data)
    
    @staticmethod
    def adpcm_to_pcm16(adpcm_data: bytes) -> bytes:
        """
        Chuyển ADPCM 4 bit ngược lại PCM 16 bit
        """
        if len(adpcm_data) < 3:
            return b''
        
        # Đọc phần đầu
        predicted, step_index = struct.unpack('<hB', adpcm_data[:3])
        adpcm_bytes = adpcm_data[3:]
        
        # Bảng bước IMA ADPCM
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]
        
        # Bảng điều chỉnh chỉ số
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]
        
        pcm_samples = []
        
        for byte in adpcm_bytes:
            # Giải mã hai mẫu 4 bit
            for shift in [0, 4]:
                adpcm_sample = (byte >> shift) & 0x0F
                
                # Tính chênh lệch
                step = step_table[step_index]
                diff = 0
                
                if adpcm_sample & 4:
                    diff += step
                step >>= 1
                if adpcm_sample & 2:
                    diff += step
                step >>= 1
                if adpcm_sample & 1:
                    diff += step
                step >>= 1
                diff += step
                
                if adpcm_sample & 8:
                    predicted -= diff
                else:
                    predicted += diff
                
                # Giới hạn phạm vi
                if predicted > 32767:
                    predicted = 32767
                elif predicted < -32768:
                    predicted = -32768
                
                pcm_samples.append(predicted)
                
                # Cập nhật chỉ số bước
                step_index += index_table[adpcm_sample & 7]
                if step_index < 0:
                    step_index = 0
                elif step_index > 88:
                    step_index = 88
        
        return np.array(pcm_samples, dtype=np.int16).tobytes()
    
    @staticmethod
    def downsample_pcm16(pcm_data: bytes, from_rate: int = 16000, to_rate: int = 8000) -> bytes:
        """
        Giảm tần số lấy mẫu (tùy chọn)
        16kHz -> 8kHz có thể giảm thêm 50% lượng dữ liệu
        """
        if from_rate == to_rate:
            return pcm_data
            
        # Phân tích dữ liệu PCM
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Giảm tần số lấy mẫu đơn giản (cứ cách một mẫu lấy một mẫu)
        if from_rate == 16000 and to_rate == 8000:
            downsampled = samples[::2]
        else:
            # Lấy mẫu lại phức tạp hơn cần scipy
            ratio = to_rate / from_rate
            new_length = int(len(samples) * ratio)
            downsampled = np.interp(
                np.linspace(0, len(samples) - 1, new_length),
                np.arange(len(samples)),
                samples
            ).astype(np.int16)
        
        return downsampled.tobytes()


class CompressedAudioCache:
    """Bộ đệm âm thanh nén"""
    
    def __init__(self, compression_type: str = "adpcm", use_downsample: bool = False):
        """
        compression_type: "none", "ulaw", "adpcm"
        """
        self.compression_type = compression_type
        self.use_downsample = use_downsample
        self._cache = {}  # {filepath: compressed_data}
        self._original_sizes = {}  # {filepath: original_size}
        
    def load_and_compress(self, filepath: str) -> Optional[bytes]:
        """Nạp và nén file âm thanh (thống nhất chuyển sang 8kHz)"""
        if filepath in self._cache:
            return self._cache[filepath]
        
        try:
            with wave.open(filepath, 'rb') as wav:
                # Kiểm tra định dạng
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                
                if channels != 1:
                    logger.warning(f"{filepath} không phải đơn kênh")
                if sampwidth != 2:
                    logger.warning(f"{filepath} không phải âm thanh 16 bit")
                
                # Đọc toàn bộ dữ liệu
                frames = wav.readframes(wav.getnframes())
                
                # Nếu là stereo thì chuyển sang mono
                if channels == 2:
                    import audioop
                    frames = audioop.tomono(frames, sampwidth, 1, 0)
                
                # [Sửa đổi] Luôn chuyển về 8kHz (dùng ratecv để giữ nguyên cao độ và tốc độ)
                if framerate != 8000:
                    import audioop
                    frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                    framerate = 8000
                
                # Ghi lại kích thước gốc (kích thước sau khi chuyển đổi)
                self._original_sizes[filepath] = len(frames)
                
                # Nén
                if self.compression_type == "ulaw":
                    compressed = AudioCompressor.pcm16_to_ulaw(frames)
                    # Thêm thông tin đầu đơn giản (1 byte nhận dạng + 4 byte độ dài gốc)
                    header = struct.pack('!BI', 0x01, len(frames))  # 0x01 biểu thị μ-law
                    compressed = header + compressed
                elif self.compression_type == "adpcm":
                    compressed = AudioCompressor.pcm16_to_adpcm(frames)
                    # Thêm thông tin đầu đơn giản (1 byte nhận dạng + 4 byte độ dài gốc)
                    header = struct.pack('!BI', 0x02, len(frames))  # 0x02 biểu thị ADPCM
                    compressed = header + compressed
                else:
                    compressed = frames
                
                self._cache[filepath] = compressed
                
                # In tỷ lệ nén
                compression_ratio = len(compressed) / self._original_sizes[filepath]
                logger.info(f"[Nén] {os.path.basename(filepath)}: "
                          f"{self._original_sizes[filepath]} -> {len(compressed)} bytes "
                          f"({compression_ratio:.1%})")
                
                return compressed
                
        except Exception as e:
            logger.error(f"Nén âm thanh thất bại {filepath}: {e}")
            return None
    
    def decompress(self, compressed_data: bytes) -> Optional[bytes]:
        """Giải nén dữ liệu âm thanh"""
        if not compressed_data or len(compressed_data) < 5:
            return compressed_data
        
        try:
            # Kiểm tra phần đầu
            compression_type = compressed_data[0]
            if compression_type == 0x01:  # Nhận dạng μ-law
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                ulaw_data = compressed_data[header_size:]
                
                # Giải nén μ-law
                pcm_data = AudioCompressor.ulaw_to_pcm16(ulaw_data)
                
                return pcm_data
            elif compression_type == 0x02:  # Nhận dạng ADPCM
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                adpcm_data = compressed_data[header_size:]
                
                # Giải nén ADPCM
                pcm_data = AudioCompressor.adpcm_to_pcm16(adpcm_data)
                
                return pcm_data
            else:
                # Dữ liệu chưa nén
                return compressed_data
                
        except Exception as e:
            logger.error(f"Giải nén âm thanh thất bại: {e}")
            return compressed_data
    
    def get_compression_stats(self) -> dict:
        """Lấy thông tin thống kê nén"""
        total_original = sum(self._original_sizes.values())
        total_compressed = sum(len(data) for data in self._cache.values())
        
        return {
            "files_cached": len(self._cache),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "compression_ratio": total_compressed / total_original if total_original > 0 else 0,
            "bytes_saved": total_original - total_compressed
        }


# Thể hiện bộ đệm âm thanh nén toàn cục
# Mặc định dùng nén ADPCM, chất lượng âm thanh tốt hơn, tỷ lệ nén cũng khá (75%)
# Có thể đặt qua biến môi trường AIGLASS_COMPRESS_TYPE: none, ulaw, adpcm
import os
compression_type = os.getenv("AIGLASS_COMPRESS_TYPE", "adpcm").lower()
if compression_type not in ["none", "ulaw", "adpcm"]:
    compression_type = "adpcm"
compressed_audio_cache = CompressedAudioCache(compression_type=compression_type, use_downsample=False) 