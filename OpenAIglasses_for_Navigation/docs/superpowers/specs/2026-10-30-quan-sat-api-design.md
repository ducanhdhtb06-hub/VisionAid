# Quan sát nhờ API (2026-10-30)

**Mục đích:** người dùng hỏi bằng giọng nói ("Phía trước có gì?", "Mô tả xung quanh"), kính gửi khung hình mới nhất
kèm câu hỏi lên AI thị giác và đọc câu trả lời. Giống chế độ trò chuyện có sẵn, nhưng đổi được nhà cung cấp.

**Cách chọn nhà cung cấp** (`quan_sat.py`):

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `AIGLASS_QUAN_SAT_API` | `gemini` / `qwen` | không đặt: có `GEMINI_API_KEY` thì Gemini, không thì Qwen |
| `GEMINI_API_KEY` | khóa Google AI Studio | đặt trong `.env`, không commit |
| `AIGLASS_GEMINI_MODEL` | `gemini-3.1-flash-lite` | model nhìn ảnh (cổng tương thích OpenAI) |
| `AIGLASS_GEMINI_TTS_MODEL` | `gemini-3.8-flash-lite-tts` | model đọc, phát dần (PCM16 mono 24 kHz) |
| `AIGLASS_GEMINI_GIONG` | `Kore` | giọng đọc dựng sẵn của Gemini |

- **Qwen:** như cũ, Qwen-Omni trả cả chữ lẫn giọng (`omni_client.stream_chat`).
- **Gemini:** "tít" ngay khi nhận câu hỏi -> ảnh + câu hỏi -> chữ -> Gemini TTS **phát dần** (`streamGenerateContent`)
  -> hạ 8 kHz như luồng Omni. Client dùng chung (giữ kết nối); lỗi mạng thì gọi lại một lần; lúc bật máy chủ làm
  nóng model nhìn (không làm nóng model đọc, để dành lượt TTS của gói miễn phí).
- Câu quan sát (`KW_QUAN_SAT`) được phép cả khi đang dẫn đường; như mọi lượt hội thoại, dẫn đường tạm dừng rồi khôi
  phục khi trả lời xong.
- Chỉ dẫn hệ thống: tiếng Việt, tối đa ba câu, nói vật nguy hiểm trước, dùng trái/phải/phía trước, không đoán số mét,
  không bịa.

**Giới hạn:** nhận giọng nói (ASR) vẫn dùng DashScope, nên vẫn cần `DASHSCOPE_API_KEY`. Ảnh camera (có thể có người đi
đường) được gửi ra máy chủ nhà cung cấp; cần nói rõ trong điều khoản sản phẩm. Đã thử với khóa thật (30/10): ảnh xe máy đỗ lấn vạch -> trả lời đúng, tổng ~5 s từ lúc hỏi tới lúc bắt đầu đọc.

**Đo tốc độ (30/10, IMG_2143 hành lang, "Phía trước tôi là gì?"):**

| Cách | Hỏi -> tiếng đầu |
|---|---|
| 3.5-flash-lite + TTS chờ cả câu (bản đầu) | ~7,4 s |
| Gemini Live (native audio, một bước) | 3,8–4,9 s |
| **3.1-flash-lite + TTS phát dần (hiện tại)** | **2,8 s** |

Model nhìn: 3.1-flash-lite 1,8 s; 3.5-flash-lite 2,0 s; 3.5-flash 50 s; 3.8-flash không trả lời; 2.5-flash(-lite) không
còn mở cho khóa mới. Model đọc: 3.8-flash-lite-tts 3,7 s cả câu / 1,0 s mẩu đầu; 3.1-flash-tts 5,1 s; 2.5-flash-tts 5,8 s.
Gói miễn phí giới hạn số lần gọi TTS mỗi phút: gọi dồn bị 429.
