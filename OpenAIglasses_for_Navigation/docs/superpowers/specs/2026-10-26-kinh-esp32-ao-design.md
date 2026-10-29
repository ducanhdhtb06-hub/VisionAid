# Thiết kế: Kính ESP32 ảo — chạy module chỉ đường "như đeo kính thật" khi chưa có bo

Ngày: 2026-10-26 · Trạng thái: **đã duyệt, đã làm Task 1–9 + 11 (Task 10 quét đang chạy)** — xem tools/esp32_ao/README.md · Phạm vi: chỉ module chỉ đường (dẫn theo vạch)

## 1. Ý định và lý do

Người dùng cần **một bản MVP để gọi vốn**, nhưng **chưa mua được bo XIAO ESP32S3 Sense**. Cần một cách chạy module chỉ đường giống đeo kính thật nhất có thể, để:

1. **Đo và kiểm chứng** mọi thay đổi của module chỉ đường trước khi có phần cứng, bằng số liệu thuyết phục được.
2. **Quay video demo** cho nhà đầu tư.
3. Khi có bo thì **cắm vào là chạy**, ít bất ngờ, vì máy chủ đã quen với đúng loại dữ liệu bo sẽ gửi.

Sự thật cần giữ: **không thể giống 100% khi chưa có bo.** Mục tiêu là giống gần 100% ở đường truyền và nhịp thời gian, giống gần đúng có kiểm chứng ở hình ảnh, và ghi rõ phần nào chưa biết.

## 2. Mục tiêu (đo được)

- **G1. Đường truyền.** "ESP32 ảo" nói đúng giao thức firmware (`compile/compile.ino` là chuẩn, xem 7.1), cắm vào **`app_main.py` thật** không cần sửa máy chủ, trừ nhật ký câu nói chỉ dùng khi thử (7.5).
- **G2. Hình ảnh.** Hồ sơ ảnh cho **OV2640 và OV3660** (bo bán ra có thể gắn một trong hai). Bảng nén JPEG lấy **y hệt** từ ảnh mẫu chụp bằng bo thật. Dung lượng khung, nhiễu theo độ sáng, độ nét, độ tương phản và độ bão hòa khớp ảnh mẫu trong ngưỡng do người làm đặt sau khi có ảnh mẫu, kèm bảng so.
- **G3. Nhịp thời gian.** Số khung/giây, chỉ giữ khung mới nhất, trễ, WiFi chập chờn, mất kết nối: bắt chước đúng hành vi firmware. Đo được **độ trễ từ lúc chụp khung đến lúc câu bắt đầu phát** (p50/p95).
- **G4. Chế độ A (phát lại video).** Chạy lặp lại được: cùng `seed` cho cùng kết quả. Chạy hàng loạt qua đêm, xuất bảng số.
- **G5. Chế độ B (iPhone làm kính, đi thật).** Dùng **cùng hồ sơ** với A. Xuất **video demo chia đôi màn hình có tiếng**.
- **G6. Ngày kiểm chứng khi có bo.** Có danh sách việc và công cụ so ảnh bo thật với hồ sơ, để chỉnh hồ sơ cho khớp.

## 3. Không làm

- Không giả lập chip ESP32 ở mức lệnh máy (QEMU…): máy chủ chỉ thấy giao thức, nên giả lập giao thức là đủ.
- Không mô phỏng pin, nhiệt độ, sóng vô tuyến chi tiết.
- Không làm qua đường, đèn giao thông, tìm đồ, trò chuyện AI.
- Không thay model nhận vạch (việc riêng, sau này, vì giấy phép).
- Không sửa firmware. Nếu thấy firmware lệch với máy chủ thì ghi lại để làm riêng.

## 4. Điều không được để xảy ra

- **Hồ sơ ảnh đẹp hơn thật** → số liệu lạc quan → ra đường thật thì hỏng. Khi chưa chắc, **chọn phía xấu hơn**.
- **Chỉnh luật dẫn đường cho khớp hồ sơ trên chính video dev** (overfit). Video giữ riêng không dùng để chỉnh.
- **Đi bịt mắt ngoài đường** để "thử như người mù". Đi thật luôn mở mắt. Thử phản ứng theo lệnh chỉ ở khu vực kín, có người đi kèm.
- Tài liệu gọi vốn nói "giống 100%" hoặc "an toàn".

## 5. Ràng buộc

- Chưa có bo. Cảm biến có thể là OV2640 hoặc OV3660; góc nhìn ống kính chưa biết (ước lượng 55–66° theo chiều ngang).
- Máy chủ: laptop RTX 4060, khoảng 170 ms/khung (đo ở Task 1). Chạy GPU **lần lượt**, không chạy song song (kết quả hỏng — Task 4).
- Không commit `.env` hay API key. Khi chạy thử `app_main` thì đặt `DASHSCOPE_API_KEY` giả ngay trong lệnh.
- Ảnh mẫu tải từ mạng chỉ để đo: ghi nguồn và giấy phép, không phân phối lại.
- Hiện module chỉ đường dùng `yolo-seg.pt` (AGPL): dùng nội bộ, chưa phát hành.

## 6. Các cách đã cân nhắc

| Cách | Được | Mất |
|---|---|---|
| **A. ESP32 ảo phát lại video đã quay** | Lặp lại được, chấm điểm được, chạy qua đêm, kiểm lại mọi lần sửa | Người trong video không phản ứng theo lệnh |
| **B. iPhone làm kính, đi thật** | Người thật, mạng thật, trễ thật; ra video demo | Không lặp lại được; phải ra đường; bên dưới vẫn là camera iPhone |
| **C. Kết hợp: một hồ sơ ESP32 dùng chung cho A và B** | Có cả số liệu (A) lẫn video demo (B); làm hồ sơ một lần | Nhiều việc hơn làm riêng A hoặc B |

**Người dùng chọn C** (2026-10-26). Thứ tự: hồ sơ + A trước (để đo và kiểm chứng), B sau (để quay demo), ngày kiểm chứng khi có bo.

## 7. Thiết kế đã chốt

### 7.1 Giao thức firmware phải bắt chước (đọc từ `compile/compile.ino`)

| Đường | Firmware làm gì | ESP32 ảo phải làm |
|---|---|---|
| Ảnh lên | WebSocket client tới `ws://HOST:8081/ws/camera`, gửi **nhị phân**: mỗi tin là một JPEG 640×480. 2 bộ đệm, `CAMERA_GRAB_LATEST`, hàng đợi đầy thì bỏ khung cũ nhất. `g_target_fps` (0 = không giới hạn). Mất kết nối thì thử lại mỗi 1 s. Nhận lệnh chữ `SET:FRAMESIZE=…`, `SET:QUALITY=…`, `SET:FPS=…`, `SNAP:HQ` (trả `SNAP:BEGIN`, khung, `SNAP:END`) | Như vậy. Hiện máy chủ không gửi lệnh `SET:`/`SNAP:`, nhưng vẫn nhận và ghi log nếu có |
| Tiếng lên (mic) | WebSocket tới `/ws_audio`. Kết nối xong gửi chữ `START`, rồi gửi nhị phân PCM16 16 kHz mono, mỗi tin 20 ms (640 byte). Nhận `RESTART` thì xóa hàng đợi và gửi lại `START` | Như vậy. Mặc định gửi im lặng; có thể phát một file tiếng (ví dụ câu lệnh thu sẵn) |
| IMU | UDP tới cổng 12345, 50 Hz, JSON `{"ts":…,"temp_c":…,"accel":{"x","y","z"},"gyro":{"x","y","z"}}` | Gửi IMU giả: đứng yên kèm nhiễu, hoặc suy ra từ chuyển động đầu của kính ảo (`kinh_ao`) |
| Tiếng xuống (loa) | HTTP `GET /stream.wav` (giữ kết nối), đọc đầu WAV, khởi động lại I2S theo tần số trong đầu (máy chủ phát 8 kHz) | Như vậy. Ghi lại WAV kèm thời điểm, và phát ra loa hoặc tai nghe qua bộ lọc "loa nhỏ" (7.4) |
| Mất kết nối | Bíp sau khi `wsCam` mất (Task 2b, tối đa 10 s/lần) | Ghi sự kiện "kính sẽ bíp" để kiểm cùng câu "Mất hình…" của máy chủ |

### 7.2 Hồ sơ ảnh (ống kính + cảm biến + nén), đi theo thứ tự

1. **Ống kính:** cắt khung nguồn về **góc nhìn ngang** của hồ sơ, tỉ lệ 4:3, méo thùng nhẹ (hệ số k1). Góc nhìn là tham số, vì chưa biết.
2. **Phơi sáng:** hai kiểu.
   - **"Cố định"** như firmware gốc (`aec_value 40`, tự tăng độ nhạy): chỗ tối thì tối và nhiễu.
   - **"Tự động"**: sáng đều nhưng nhòe hơn khi lắc đầu.
3. **Nhòe do chuyển động:** theo tốc độ xoay đầu đo từ luồng quang học giữa hai khung, nhân với thời gian phơi sáng của hồ sơ. Video nguồn nên quay với **màn trập nhanh** để ít nhòe sẵn (7.6).
4. **Màn trập cuộn (rolling shutter):** dịch từng hàng ảnh theo chuyển động, với thời gian đọc ảnh của cảm biến.
5. **Cảm biến:** nhiễu phụ thuộc độ sáng (nhiễu photon + nhiễu đọc), dải sáng hẹp (cắt vùng sáng và vùng tối), màu và độ tương phản theo cài đặt firmware (`contrast 1`, `saturation 1`).
6. **Nén JPEG:** bảng lượng tử và kiểu lấy mẫu màu **sao chép từ ảnh mẫu thật** (đọc trong đầu file JPEG; dự kiến 4:2:2 nhưng phải kiểm). Hai mức chất lượng 17 (gốc) và 12. Nếu ảnh mẫu không đúng mức 17 thì dùng mức gần nhất và ghi "ước lượng". Thời gian đọc ảnh của màn trập cuộn (bước 4) lấy từ tài liệu cảm biến nếu có, không có thì ước lượng và ghi rõ.

Mỗi hồ sơ là một file tham số (ví dụ `ov2640_xiao.json`, `ov3660_xiao.json`). Mỗi tham số ghi **nguồn**: đo từ ảnh mẫu nào, hay ước lượng.

**Hiệu chỉnh:**
- Tìm ảnh mẫu chụp bằng bo thật trên mạng (repo GitHub, issue, bộ dữ liệu ESP32-CAM; ưu tiên XIAO ESP32S3 Sense ở VGA, cả OV2640 lẫn OV3660).
- Đo các chỉ số trên ảnh mẫu, rồi chỉnh tham số cho ảnh iPhone đã qua hồ sơ, chụp cảnh tương tự, có chỉ số khớp.
- Xuất bảng so. Chỗ nào không có dữ liệu, ví dụ bóng râm, thì đặt phía xấu hơn và ghi "ước lượng".

Bộ `LamXau` hiện có (`thu_nghiem_do_sau/gia_lap_kinh.py`) chưa hiệu chỉnh theo bo thật. Hồ sơ mới **thay** nó trong đường ESP32 ảo, giữ `LamXau` cho các công cụ cũ.

### 7.3 Nhịp thời gian và mạng

- **Nhịp chụp:** cảm biến chụp ở tốc độ của nó, `GRAB_LATEST` giữ khung mới nhất, gửi theo `g_target_fps` (thử 0 và 8).
- **Hồ sơ mạng:**
  - "tốt": trễ thấp, không rớt.
  - "kém": trễ dao động, rớt gói theo đợt, băng thông giới hạn.
  - "rớt": mất hẳn 3 s rồi có lại.
- Ghi log mọi khung (lúc chụp, lúc gửi, có bị bỏ không) và mọi câu (lúc máy chủ phát, lúc tới loa). Từ đó tính độ trễ.

### 7.4 Tiếng ra loa

WAV 8 kHz đi qua bộ lọc "loa nhỏ MAX98357A" (cắt trầm, cắt cao), trộn tiếng ồn đường lấy từ chính tiếng của video đã quay, để người dùng nghe thử câu có rõ không. Chỉ dùng để nghe; không chấm điểm bằng nó.

### 7.5 Bật chế độ dẫn đường khi thử

Firmware không có nút; người dùng ra lệnh bằng giọng ("bắt đầu dẫn đường") qua mic → ASR DashScope. Lần thử trước (Task 2a) gọi thẳng `start_ai_with_text_custom` trong cùng tiến trình.

**Cập nhật khi viết kế hoạch (2026-10-26):** `app_main` **đã có sẵn** lệnh chữ `PROMPT:<câu>` trên `/ws_audio` (gọi `start_ai_with_text_custom`, trả `OK:PROMPT_ACCEPTED`). ESP32 ảo và cầu nối dùng lệnh này: gửi `PROMPT:bắt đầu dẫn đường` để bật, `PROMPT:dừng dẫn đường` để tắt. **Không thêm cổng lệnh nào vào máy chủ.**

Thay đổi máy chủ duy nhất là một **nhật ký câu nói** chỉ dùng khi thử: biến môi trường `AIGLASS_NHAT_KY_CAU=<file>`, mặc định tắt, ghi mỗi lần `play_voice_text` được gọi (giờ, câu, có phát hay bị chặn). Lý do: luồng `/stream.wav` chỉ chứa tiếng, không có chữ, nên không chấm được "sai chiều" nếu không biết câu nào được nói.

(Thiết kế cũ: thêm một cổng lệnh chỉ dùng khi thử, chỉ bật khi đặt biến môi trường. Không cần nữa.) Hai lối khác vẫn giữ để tham khảo:
- Phát file tiếng qua `/ws_audio` để đi đường ASR thật. Cần khóa DashScope thật và mạng; dùng cho demo nếu người dùng muốn (câu hỏi mở 1).
- Vào thẳng chế độ dẫn đường ngay khi khởi động.

### 7.6 Hai chế độ chạy

- **A — phát lại:** đầu vào là video iPhone (hoặc khung từ `kinh_ao` để thêm động tác đầu và thân), qua hồ sơ ảnh, nhịp và mạng, rồi vào `app_main` thật. Đầu ra:
  - log JSON;
  - bảng số (sai chiều, câu/phút, % đo tin cậy, độ trễ p50/p95, % thời gian mất vạch, số lần báo "Mất hình" đúng/nhầm);
  - video chia đôi có tiếng.
- **B — iPhone làm kính, đi thật:**
  - iPhone kẹp mũ mở trang web (dựa trên `thu_kinh.html` đã có), gửi khung **chất lượng cao** tới một **cầu nối** trên laptop.
  - Cầu nối áp **cùng hồ sơ** rồi đóng vai ESP32 ảo với `app_main` thật, đồng thời chuyển tiếng `/stream.wav` về tai nghe iPhone. Nhờ vậy `app_main` nhận đúng thứ bo thật sẽ gửi.
  - Mạng ngoài đường: hotspot (laptop phát, hoặc iPhone phát). Ghi lại phiên để xuất video demo.
- **Cách quay video nguồn:** iPhone kẹp mũ, ống 1x, 4:3 hoặc 16:9, **màn trập nhanh** (dùng app quay cho chỉnh tốc độ màn trập), tắt chống rung mạnh.

### 7.7 Quét những gì chưa biết

Chạy chế độ A với ma trận nhỏ, ra bảng "module chịu được đến đâu" và đánh dấu tổ hợp làm module hỏng:
- góc nhìn {55°, 60°, 66°}
- cảm biến {OV2640, OV3660}
- JPEG {12, 17}
- phơi sáng {cố định, tự động}
- mạng {tốt, kém, rớt}

Mặc định chọn tổ hợp **xấu nhất có lý** để báo số.

### 7.8 Ngày kiểm chứng khi có bo (khoảng 1 ngày)

1. Xem log khởi động: loại cảm biến. Kiểm lật ảnh: tay phải hiện bên phải, trời ở trên.
2. Đo góc nhìn ngang (tường cách 1 m), đặt `AIGLASS_HFOV`.
3. Chụp bộ ảnh mẫu cùng cảnh với iPhone (nắng, bóng râm, chiều tối), chạy so với hồ sơ, chỉnh hồ sơ.
4. Đo nhịp gửi thật, trễ đầu-cuối thật, tầm WiFi, độ nóng sau 30 phút.
5. Chạy lại bảng số chế độ A bằng hồ sơ đã chỉnh, so với trước.

## 8. Để người làm tự quyết

Tên file và hàm, cấu trúc thư mục (đề xuất `tools/esp32_ao/`), thư viện mạng, cách vẽ video chia đôi, các ngưỡng so ảnh cụ thể (đặt sau khi có ảnh mẫu), mô hình rớt gói cụ thể.

## 9. Câu hỏi đã trả lời / còn mở

1. **Đã chốt (2026-10-26): KHÔNG ra lệnh bằng giọng trong MVP.** Bật dẫn đường bằng **nút** trên trang iPhone (chế độ B) và bằng cổng lệnh khi thử (chế độ A). Lý do: ASR hiện tại (`paraformer-realtime-v2`) chưa kiểm có nghe tiếng Việt không, cần khóa thật và 4G; lỗi ở bước này làm hỏng cả demo. Ra lệnh bằng giọng là việc sau (thử Whisper chạy trên laptop).
2. **Còn mở:** mốc gặp nhà đầu tư. Mặc định (quyết định của người viết): làm theo thứ tự 7.6 (A → B), phần quét 7.7 chạy qua đêm sau B. Nếu mốc gần, đưa B và video demo lên trước, ngay sau giai đoạn đường truyền.

## 10. Ghi chú của người viết (quyết định của mình, người dùng kiểm lại)

- A trước, B sau: vì A cho số đo để hiệu chỉnh hồ sơ; B dùng lại hồ sơ đó.
- Hồ sơ "phía xấu hơn" khi thiếu dữ liệu: để tránh lạc quan.
- Cầu nối cho B thay vì sửa `app_main`: để `app_main` nhận đúng thứ bo thật gửi.
- Cổng lệnh khi thử mặc định tắt: không để lọt vào bản chạy thật.
