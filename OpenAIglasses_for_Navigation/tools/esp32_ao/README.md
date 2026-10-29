# Kính ESP32 ảo

Giả làm bo **XIAO ESP32S3 Sense** đúng giao thức firmware (`compile/compile.ino`), để chạy **`app_main.py` thật** khi chưa có bo.

- **Giống gần 100%:** đường truyền (ảnh `/ws/camera`, mic `/ws_audio`, IMU UDP 12345, loa `/stream.wav`) và nhịp gửi.
- **Gần đúng:** hình ảnh. Hồ sơ OV2640/OV3660 hiện còn **ước lượng**; ảnh từ bo thật sẽ dùng để chỉnh lại.

Thiết kế: `docs/superpowers/specs/2026-10-26-kinh-esp32-ao-design.md`. Kế hoạch: `docs/superpowers/plans/2026-10-26-kinh-esp32-ao.md`.

Mọi lệnh chạy từ `OpenAIglasses_for_Navigation/`. Lệnh nào bật `app_main` thì chỉ chạy khi **không có việc GPU nào khác** (`kinh_ao`, `danh_gia_vach`, `analyze_guidance`…): chạy song song làm kết quả hỏng. Khóa API thật **không cần**: ESP32 ảo tự đặt khóa giả trong môi trường của tiến trình con, không ghi `.env`.

## Chế độ A: phát lại video (chấm điểm)
```bash
../.venv/bin/python tools/esp32_ao/phat_lai.py ../my/IMG_7358.MOV --ra test_out/esp32_ao/thu \
    --ho-so ov2640_xiao --mang tot --fps-gui 8 --nhan test_cham_diem/nhan_vach_dev.json
```
- Chạy **đúng thời gian thật** (video 2 phút chạy mất 2 phút), vì máy chủ dùng đồng hồ thật.
- Mỗi lượt tự bật `app_main` rồi tự tắt. Nếu cổng 8081 đang có máy chủ khác, lệnh dừng với mã 2 và không đụng vào máy chủ đó.
- Tùy chọn:
  - `--mang tot|kem|rot`: mạng tốt (4 Mbit/s), kém (chập chờn, kẹt), hoặc rớt 3 s (đổi bằng `--t-rot`, `--dai-rot`).
  - `--fps-gui`: giới hạn khung/giây như `SET:FPS`; 0 = không giới hạn, như firmware gốc.
  - `--mic lenh|firmware|tat`: kiểu kênh mic.
  - `--ghi-de khoa=gia_tri`: ghi đè một tham số của hồ sơ ảnh.
  - `--hfov-may-chu`: góc nhìn đặt cho máy chủ (`AIGLASS_HFOV`).
- Kết quả: `bang_so.json` (sai chiều, câu/phút, độ trễ camera → loa p50/p95, % khung bị bỏ, báo "Mất hình" đúng/nhầm), kèm `khung.jsonl`, `cau.jsonl`, `loa.wav`, `xem/`.

## Xuất video chia đôi có tiếng
```bash
../.venv/bin/python tools/esp32_ao/xuat_video.py test_out/esp32_ao/thu --ra demo.mp4 --video-nguon ../my/IMG_7358.MOV
```
- Bên trái: cảnh nguồn, hoặc video quay người đi từ bên ngoài (`--video-trai`, `--lech-trai`).
- Bên phải: ảnh máy chủ vẽ. Dưới cùng: câu kính nói.
- `--loa-nho`: nghe thử qua bộ lọc loa nhỏ, trộn tiếng ồn đường.

## Chế độ B: iPhone làm kính (quay demo)
```bash
cd .. && .venv/bin/python thu_nghiem_do_sau/truc_tiep/chay.py --cau-noi --ho-so ov2640_xiao
```
1. Lệnh tự bật `app_main`, mở đường hầm và in mã QR. Quét mã bằng iPhone.
2. iPhone **kẹp mũ, cầm ngang**, đeo tai nghe. Bấm **Bắt đầu**: kính bắt đầu dẫn đường. Bấm **Dừng**: tắt.
3. Ảnh iPhone được biến thành "ảnh ESP32" trước khi vào máy chủ thật.
4. Nếu mất mạng iPhone, máy chủ phản ứng như kính thật bị rớt.
5. Xuất video: lệnh `xuat_video.py` như trên, với `<thư mục phiên con>` trong `thu_nghiem_do_sau/truc_tiep/phien/cau_noi_*`.

**Khi đi thật: luôn mở mắt**, có người đi kèm. Không bịt mắt ngoài đường.

### Cách quay video nguồn bằng iPhone
- Kẹp lên vành mũ sát trán, cầm ngang, chúc xuống khoảng 10°.
- Ống kính 1x, không dùng 0.5x. Tắt Action mode và Cinematic.
- Nếu được, dùng app cho chỉnh **màn trập nhanh** (1/500 s), để ảnh nguồn ít nhòe sẵn. Hồ sơ ảnh sẽ tự thêm nhòe đúng kiểu ESP32.
- Video mới chỉ dùng **làm bài kiểm tra**, không dùng để chỉnh ngưỡng.

## Quét những gì chưa biết (chạy qua đêm, chỉ video dev)
```bash
../.venv/bin/python tools/esp32_ao/quet.py khiem.mp4 ../my/IMG_7354.MOV ../my/IMG_7356.MOV ../my/IMG_7358.MOV \
    --ra test_out/esp32_ao/quet --nhan test_cham_diem/nhan_vach_dev.json
```
- Chạy 9 cấu hình, mỗi cấu hình đổi đúng một yếu tố so với cấu hình gốc: góc nhìn 60°/66°, lệch góc nhìn với máy chủ (ống 56° nhưng máy chủ đặt 70°), OV3660, JPEG 12, phơi sáng tự động, mạng tốt, mạng rớt.
- Kết quả: `bang.md`. Hàng đánh dấu **HỎNG** khi có lệnh sai chiều hoặc trễ p95 > 1,5 s. Bị ngắt giữa chừng thì chạy lại lệnh, nó làm tiếp phần còn thiếu.

## Ngày kiểm chứng (khi có bo thật, khoảng 1 ngày)
1. **Loại camera:** mở Serial Monitor (115200), xem log khởi động. Bo mới có thể gắn **OV3660**: phải lật dọc ảnh (`set_vflip`).
2. **Lật ảnh:** mở `/ws/viewer`.
   - Giơ **tay phải**: phải hiện bên **phải** màn hình. Không đúng thì đổi `set_hmirror`.
   - Trời phải ở **trên**. Không đúng thì đổi `set_vflip`.
   - Lật sai = mọi lệnh Dịch/Rẽ **ngược chiều**.
3. **Góc nhìn:** đặt kính cách tường 1 m, đo bề rộng tường thấy trong ảnh, rồi tính `HFOV = 2·atan(bề_rộng/2)`. Chạy máy chủ với `AIGLASS_HFOV=<số đo>`, và ghi `hfov_ong_kinh` vào hồ sơ.
4. **Ảnh mẫu:** chụp **≥ 3 ảnh** VGA bằng bo (nắng, bóng râm, chiều tối), chép vào `test_out/esp32_mau/ov2640/` (hoặc `ov3660/`). Rồi chạy:
   ```bash
   ../.venv/bin/python tools/esp32_ao/do_anh_mau.py test_out/esp32_mau/ov2640
   ../.venv/bin/python tools/esp32_ao/hieu_chinh.py --mau test_out/esp32_mau/ov2640 --ho-so ov2640_xiao \
       --video ../my/IMG_7354.MOV ../my/IMG_7356.MOV ../my/IMG_7358.MOV --ghi
   ```
   Chỉ cần 1 ảnh là chép được **bảng nén JPEG thật**. Từ 3 ảnh trở lên mới hiệu chỉnh được nhiễu, độ nét, tương phản.
5. **Nhịp và độ trễ thật:** đo số khung/giây bo gửi, độ trễ camera → loa, tầm WiFi, độ nóng sau 30 phút. So với bảng của chế độ A.
6. Chạy lại `quet.py` với hồ sơ đã chỉnh, rồi so với lần trước.

## Tình trạng và giới hạn (cập nhật 2026-10-26)
- Hồ sơ ảnh OV2640/OV3660 là **ước lượng** (chưa có ảnh từ bo thật). Bảng nén JPEG đang tạm dùng bảng chuẩn ở chất lượng 35.
- Mạng "tốt" giả định 4 Mbit/s. Ngoài đời có thể khác.
- Phát hiện từ ESP32 ảo (chi tiết trong sổ `.superpowers/sdd/2026-10-26-kinh-esp32-ao/progress.md`):
  1. Từ lúc máy chủ quyết định nói đến lúc loa phát mất khoảng 0,5–1,1 s.
  2. Khung dồn trong bộ đệm mạng khi gửi nhanh hơn máy chủ xử lý (khoảng 6 khung/s), nên máy chủ xử lý **khung cũ**. Trễ camera → loa đo được: p50 khoảng 1,1–1,2 s; p95 4,6 s nếu không giới hạn fps.
  3. Mỗi câu bị gọi phát 2 lần (lần 2 bị chống lặp chặn, vô hại).
