# Câu khẩn cấp ngắt câu thường + nạp sẵn bộ dẫn đường — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Sửa 2 việc còn lại sau kế hoạch `2026-10-26-may-chu-khung-moi` (`test_out/esp32_ao/bao_cao_nhip.md`, "Phần trễ còn lại"):
1. Câu khẩn cấp ("Dừng lại…", "…đang tới", "Mất hình…") phải chờ câu thường đang đọc xong, có khi tới 1,8 s.
2. Lần đầu kính kết nối: máy chủ đứng 4,2 s để tạo BlindPathNavigator (trên vòng sự kiện), và khung đầu mất 5 s để model khởi động.

**Architecture:**
- (1) Module thuần `uu_tien_cau.py`: `la_cau_khan(text) -> bool`.
  - `audio_player` nhớ câu đang phát có khẩn không.
  - Câu khẩn đến khi đang phát câu THƯỜNG: xóa hàng đợi, đặt cờ ngắt; `broadcast_pcm16_realtime(pcm, dung=...)` dừng ở nhịp 20 ms kế tiếp và bỏ phần tiếng còn trong hàng đợi của từng kết nối.
  - Câu khẩn không ngắt câu khẩn; câu thường không ngắt gì.
- (2) Module `khoi_dong_dan_duong.py`:
  - `tao_bo_dan_duong(seg, obs)` trả `(blind, cross, master)`;
  - `lam_nong(master)` chạy một khung xám qua `process_frame` ở chế độ dẫn đường, rồi `reset()`.
  - `app_main` gọi hai hàm này trong hook startup (trước khi nhận kết nối); `ws_camera_esp` chỉ tạo khi còn `None` (dự phòng).

**Spec:** người dùng duyệt trong hội thoại 2026-10-26 ("oke làm tiếp việc 1 và 2 cho tôi").

## Global Constraints
- Không đổi nội dung câu, logic dẫn vạch hay vật cản; chỉ đổi thứ tự/cắt khi phát.
- Từ khóa khẩn (so khớp chữ thường, NFC): `"dừng lại"`, `"đang tới"`. "Mất hình ảnh từ kính, hãy dừng lại." khớp `"dừng lại"`.
- Tắt được: `AIGLASS_NGAT_CAU=0` thì phát như cũ.
- Test: `../.venv/bin/python -m pytest tests/ -q` (đứng ở `OpenAIglasses_for_Navigation/`). Đo bằng `tools/esp32_ao`, lần lượt, không chạy GPU song song. Không commit khi chưa được phép.

## Review Focus
1. Hai câu khẩn liên tiếp ("Dừng lại. Xe máy, 5 mét" rồi "…2 mét"): câu sau KHÔNG cắt câu trước, chỉ thay câu đang chờ. → Task 1 `test_khan_khong_ngat_khan`.
2. Ngắt xong phải phát được câu sau: cờ ngắt phải được xóa trước khi phát câu mới. → Task 1 `test_ngat_xong_phat_tiep`.
3. Câu thường đến khi câu khẩn đang phát: không cắt. → Task 1 `test_thuong_khong_ngat`.
4. Làm nóng lỗi (không có model, ném ngoại lệ): máy chủ vẫn khởi động, chỉ in cảnh báo. → Task 2 `test_lam_nong_loi_khong_nem`.
5. Sau làm nóng, trạng thái về IDLE, không còn câu nào bị phát. → Task 2 `test_lam_nong_tra_ve_idle`.

### Task 1: Câu khẩn ngắt câu thường
**Files:** Create `uu_tien_cau.py` · Modify `audio_player.py` (`play_audio_threadsafe`, `_broadcast_audio_optimized`, `_phat_cau`), `audio_stream.py` (`broadcast_pcm16_realtime`) · Test `tests/test_uu_tien_cau.py`

**Interfaces:**
- `uu_tien_cau.la_cau_khan(text: str) -> bool`
- `uu_tien_cau.BoNgat`: trạng thái thuần, gọi từ luồng âm thanh.
  - `bat_dau(khan: bool)`: bắt đầu phát một câu, xóa cờ ngắt.
  - `xong()`
  - `co_cau_moi(khan: bool) -> bool`: True nếu phải ngắt câu đang phát, và khi đó đặt cờ.
  - `nen_dung() -> bool`
- `audio_stream.broadcast_pcm16_realtime(pcm16: bytes, dung=None)`: `dung` là hàm không đối số. Khi trả True thì dừng và xả hàng đợi của từng kết nối.

- [ ] Test hỏng:
  - `test_la_cau_khan`: "Dừng lại. Xe máy, 2 mét", "Xe máy, bên trái, đang tới", "Mất hình ảnh từ kính, hãy dừng lại." → True; "Dịch sang phải", "Được rồi, hãy dừng dịch ngang.", "Đã dừng dẫn đường." → False.
  - `test_khan_ngat_thuong`, `test_khan_khong_ngat_khan`, `test_thuong_khong_ngat`, `test_ngat_xong_phat_tiep`, `test_khong_phat_gi_thi_khong_ngat`.
  - `test_broadcast_dung_giua_chung`: pcm 1 s, `dung` trả True sau 5 lần gọi → số mảnh gửi ≤ 6, hàng đợi kết nối rỗng.
- [ ] Đỏ → cài → xanh. Nối vào `audio_player` (đọc `AIGLASS_NGAT_CAU`), cả bộ test xanh → `task-done`.

### Task 2: Nạp sẵn + làm nóng bộ dẫn đường lúc khởi động
**Files:** Create `khoi_dong_dan_duong.py` · Modify `app_main.py` (hook startup mới; `ws_camera_esp` dùng đối tượng đã có) · Test `tests/test_khoi_dong_dan_duong.py`

**Interfaces:**
- `tao_bo_dan_duong(seg_model, obstacle_detector) -> tuple | None`: `None` khi `seg_model` là `None`.
- `lam_nong(master, cao=480, rong=640) -> float | None`: số giây khung làm nóng; `None` khi lỗi. Không ném ngoại lệ; luôn `reset()` master.

- [ ] Test hỏng, dùng master giả:
  - `test_lam_nong_goi_process_frame_mot_lan` (khung 480×640×3 uint8);
  - `test_lam_nong_tra_ve_idle`;
  - `test_lam_nong_loi_khong_nem`;
  - `test_tao_bo_dan_duong_khong_model_tra_none`.
- [ ] Cài, nối startup: chạy trong `run_in_threadpool`, gán global `blind_path_navigator`, `cross_street_navigator`, `orchestrator`.
- [ ] Kiểm thật: máy chủ lên, ESP32 ảo nối. Mốc kết nối → khung xem đầu < 2 s (trước 14–15 s).

### Task 3: Đo trước/sau
- [ ] Lặp lượt a, b, d trên IMG_7358 → `test_out/esp32_ao/t8_sau2/`; quét `goc`, `mang_tot`, `mang_rot` trên 4 video dev → `quet_sau2/`.
- [ ] Tách trễ "chụp → ra câu" và "chờ loa" cho câu khẩn (`scratchpad/tach_tre.py`). Mục tiêu: câu khẩn chờ loa p95 < 0,3 s; sai chiều 0.
- [ ] Ghi `bao_cao_nhip.md`, `TIEN_DO_DAN_VACH.md`.
