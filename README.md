# glassai — Dẫn theo vạch dẫn đường cho người khiếm thị

Phần **dẫn theo gạch dẫn đường (vạch xúc giác)** của kính AI glassai: camera trên kính nhìn vạch và vật cản phía trước,
nói hướng đi bằng tiếng Việt.

> **Bản thử nghiệm — CHƯA an toàn cho người dùng thật.** Mới kiểm trên video quay ở vỉa hè Hà Nội (phát lại như kính
> ESP32), chưa thử với người khiếm thị đi thật. Luôn dùng kèm gậy.

## Kính làm gì

| Tình huống | Kính nói |
|---|---|
| Tìm thấy vạch / lệch khỏi vạch | "Đã thấy vạch dẫn đường." / "Dịch sang trái" / "Dịch sang phải" |
| Vật đỗ đè lên vạch | "Xe máy chắn vạch bên phải. Đi sang trái." / "Xe máy chắn giữa vạch. Đi sang trái, dò gậy." / "… Dùng gậy dò đường vòng." |
| Đi vòng xong, về lại vạch | "Đã về vạch." |
| Xe đang chạy tới | "Xe máy, bên trái, đang tới" → "Dừng lại." |
| Mép vỉa hè, cầu thang, vật ngang đầu | "Sắp xuống lòng đường, phía trước", "Cầu thang lên, phía trước", "Dừng lại. Vật cản ngang đầu" |

Nguyên tắc: **một giọng nói, ít lời** — khi đang theo vạch chỉ quan tâm vật trong vùng vạch (vật cạnh vạch thì im),
không đọc số mét, không bao giờ bảo đi về phía lòng đường hay về phía xe đang lao tới; 5 loại nguy hiểm luôn báo.

## Cấu trúc

```
OpenAIglasses_for_Navigation/
  dan_theo_vach.py        đo vạch (hai mép, độ lệch, chéo), luật nói dịch trái/phải, phân loại vật chắn/cạnh vạch (3D)
  vat_chan_vach.py        bộ điều khiển vật chắn trên vạch: chọn bên đi vòng, câu nói, "Đã về vạch", nhớ xe chắn
  ne_vat_can_do_sau.py    né vật cản bằng độ sâu (Depth Anything V2) + nhận diện (D-FINE), xe đang tới, mép vỉa hè
  uu_tien_cau.py          câu gấp ("Dừng lại", "đang tới", "chắn vạch") được chen ngang câu đang đọc
  workflow_blindpath.py   ghép tất cả vào luồng dẫn đường (BlindPathNavigator)
  audio_*.py, obstacle_detector_client.py, crosswalk_awareness.py   module hỗ trợ từ dự án gốc (để chạy được test)
  tools/esp32_ao/         "kính ESP32 ảo": phát lại video như camera kính thật (56°, mạng chậm), quét câu mâu thuẫn
  tests/                  222 test (pytest)
  docs/superpowers/       thiết kế (specs) và kế hoạch (plans) từng giai đoạn
thu_nghiem_do_sau/
  ket_hop_3_model.py      lõi độ sâu + nhận diện + theo dõi vật + bộ nói (Speaker)
  tao_giong.py, giong/    tạo và các mảnh giọng nói tiếng Việt (.wav) dùng để ghép câu
```

## Cài và chạy test

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd OpenAIglasses_for_Navigation
python -m pytest tests -q          # 222 test, không cần GPU / model
```

Chạy thật trên camera cần thêm phần còn lại của dự án gốc
[OpenAIglasses_for_Navigation](https://github.com/AI-FanGe/OpenAIglasses_for_Navigation) (máy chủ `app_main.py`,
model phân đoạn vạch từ ModelScope `archifancy/AIGlasses_for_navigation`). Model độ sâu / nhận diện / lòng đường
(Hugging Face) tự tải lần đầu chạy. Các công tắc chính (biến môi trường):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `AIGLASS_DAN_VACH` | bật | dẫn theo vạch kiểu mới (`0` = luật gốc) |
| `AIGLASS_VAT_CHAN_VACH` | bật | bộ điều khiển vật chắn trên vạch (`0` = như cũ) |
| `AIGLASS_HFOV` | 70 | góc nhìn ngang camera (kính OV2640: 56) |
| `AIGLASS_NOI_KHOANG_CACH` | 0 | `1` = đọc số mét |

Không có khóa API nào trong mã; nếu dùng phần chat của dự án gốc, đặt khóa trong `.env` (đã nằm trong `.gitignore`).

## Giấy phép

Mã ở đây theo **MIT** (xem `LICENSE`; phần lấy từ dự án gốc giữ bản quyền AI-FanGe).
