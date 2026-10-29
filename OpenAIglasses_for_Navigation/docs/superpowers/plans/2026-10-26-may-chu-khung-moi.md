# Máy chủ: khung mới nhất + phục hồi khi rớt kết nối — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Sửa 3 lỗi máy chủ mà ESP32 ảo tìm ra (`test_out/esp32_ao/bao_cao_nhip.md`, `quet/bang.md`):
1. xử lý khung cũ, trễ ~1,2 s;
2. rớt WiFi thì im lặng luôn;
3. lệnh bật dẫn đường đến trước khi kính kết nối thì bị mất.

**Architecture:** Logic thuần nằm trong module mới `ket_noi_kinh.py` (test được, không cần model): `KhungMoiNhat` và `PhucHoiDanDuong`. `app_main.ws_camera_esp` sửa như sau:
- (a) đọc khung liên tục, `process_frame` chạy trong luồng phụ (một khung một lúc), luôn lấy khung mới nhất;
- (b) kết nối camera mới **thay** kết nối cũ, không từ chối 1013;
- (c) mất kết nối lúc đang dẫn đường thì nhớ lại; nối lại trong 60 s thì tự bật lại dẫn đường và nói "Đã có lại hình ảnh, tiếp tục dẫn đường.";
- (d) lệnh "bắt đầu dẫn đường" khi chưa có bộ điều phối thì ghi nhớ 60 s, bật ngay khi kính nối.

**Spec:** người dùng duyệt trong hội thoại 2026-10-26 ("oke tiếp tục cho tôi"). Mục tiêu đo:
- trễ p50 < 0,6 s, p95 < 1,2 s (mạng tốt, fps 0);
- lượt "rớt" có "Mất hình" + "Đã có lại hình ảnh…", và câu/phút sau khi nối lại như bình thường.

## Global Constraints
- Không đổi logic dẫn vạch (`dan_theo_vach.py`, `workflow_blindpath.py`) hay vật cản.
- Mặc định hành vi khác chỉ ở 4 điểm trên. Cửa sổ phục hồi `AIGLASS_PHUC_HOI_S` (mặc định 60).
- `orchestrator.process_frame` chỉ một luồng gọi tại một thời điểm.
- Test: `../.venv/bin/python -m pytest tests/ -q` đứng ở `OpenAIglasses_for_Navigation/`. Đo bằng `tools/esp32_ao` (lần lượt, không song song GPU).
- Không commit khi chưa được phép.

## Review Focus
1. Kính rớt kiểu "nửa sống" (TCP không đóng): kết nối mới phải vào được ngay. → Task 2: kiểm thật bằng ESP32 ảo (lượt rớt, bị 1013 trước đây).
2. Người dùng tự nói "dừng dẫn đường" rồi kính rớt: **không** được tự bật lại. → Task 1 `test_da_dung_thi_khong_phuc_hoi`.
3. Lệnh chờ quá 60 s không được tự bật. → Task 1 `test_lenh_cho_het_han`.
4. Lỗi trong luồng xử lý một khung không được giết vòng đọc. → Task 2: bọc try như cũ, kiểm qua log.

### Task 1: `ket_noi_kinh.py` (logic thuần)
**Files:** Create `ket_noi_kinh.py` · Test `tests/test_ket_noi_kinh.py`

**Interfaces:**
- `KhungMoiNhat`:
  - `dat(du_lieu)`: đè khung cũ, tăng `bo`;
  - `async lay() -> bytes`: chờ tới khi có khung, trả khung mới nhất;
  - `dong()`: `lay()` trả `None`.
- `PhucHoiDanDuong(cua_so_s=60.0)`:
  - `mat_ket_noi(t, dang_dan: bool)`;
  - `nguoi_dung_dung()`: xóa trạng thái nhớ;
  - `noi_lai(t) -> bool`: True nếu phải tự bật lại; chỉ một lần cho mỗi lần mất;
  - `ghi_lenh_cho(t)`;
  - `lay_lenh_cho(t) -> bool`: True nếu có lệnh chờ chưa hết hạn; lấy xong thì xóa.

- [ ] Test hỏng:
  - `test_khung_moi_nhat_de_khung_cu`, `test_lay_cho_den_khi_co_khung`, `test_dong_thi_lay_tra_none`;
  - `test_noi_lai_trong_60s_thi_phuc_hoi`, `test_noi_lai_sau_60s_thi_khong`, `test_khong_dang_dan_thi_khong`;
  - `test_da_dung_thi_khong_phuc_hoi`, `test_phuc_hoi_mot_lan`;
  - `test_lenh_cho_trong_han`, `test_lenh_cho_het_han`.
- [ ] Đỏ → cài đặt → xanh, chạy cả bộ test → `task-done`.

### Task 2: Nối vào app_main + câu mới
**Files:** Modify `app_main.py` (`ws_camera_esp`, nhánh `KW_NAV_START`/`KW_NAV_STOP` của `start_ai_with_text_custom`) · `voice/map.vi.json` + WAV "Đã có lại hình ảnh, tiếp tục dẫn đường." (`tools/gen_vi_voice.py --only`)

- [ ] Kết nối mới thay kết nối cũ: đóng cũ mã 1012, bỏ chặn 1013.
- [ ] Đọc khung liên tục vào `KhungMoiNhat`. Ghi hình/`last_frames`/`bridge_io` vẫn cho mọi khung. Tác vụ xử lý: `lay()` → giải mã → `run_in_threadpool(orchestrator.process_frame, bgr)` → phát câu + viewer như cũ.
- [ ] `finally`: nếu trạng thái đang dẫn đường thì `phuc_hoi.mat_ket_noi(t, True)` trước khi `reset()`. Kết nối mới: nếu `phuc_hoi.noi_lai(t)` → `orchestrator.start_blind_path_navigation()` + nói câu mới; nếu `lay_lenh_cho(t)` → bật dẫn đường.
- [ ] Nhánh lệnh: `KW_NAV_STOP` → `phuc_hoi.nguoi_dung_dung()`; `KW_NAV_START` khi chưa có bộ điều phối → `phuc_hoi.ghi_lenh_cho(t)`.
- [ ] Toàn bộ test xanh. Kiểm thật: `phat_lai` 30 s → không lỗi, có câu.

### Task 3: Đo trước/sau bằng ESP32 ảo
- [ ] Lặp lại lượt a, b, d (`test_out/esp32_ao/t8`) với code mới → `test_out/esp32_ao/t8_sau/`. Quét lại cấu hình `goc`, `mang_tot`, `mang_rot` trên 4 video dev → so bảng.
- [ ] Ghi `bao_cao_nhip.md` (bảng trước/sau), `TIEN_DO_DAN_VACH.md`, xuất 1 video demo lượt rớt mạng.
