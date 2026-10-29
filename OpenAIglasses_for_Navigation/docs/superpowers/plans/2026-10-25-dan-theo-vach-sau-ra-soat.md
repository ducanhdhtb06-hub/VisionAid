# Dẫn theo vạch — sửa sau rà soát 2026-10-25: kế hoạch thực hiện

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho thử trực tiếp bằng iPhone, sửa 4 điểm yếu + lỗi "Rẽ" khi quay đầu tìm ra khi rà soát (kính im khi mất hình, nhắc "dịch" quá muộn với vạch rộng, kính ảo chấm sai vì giả định vạch 0,4 m, module vật cản nói quá nhiều khi đang theo vạch), rồi đo lại.

**Architecture:** Giữ nguyên kiến trúc `dan_theo_vach.py` (đo điểm tụ + bề rộng vạch, `LuatNoi`). Thêm một lớp nhỏ thuần Python `canh_bao_ket_noi.py` (dễ test) để `app_main.py` báo khi ngừng nhận ảnh; firmware tự bíp khi mất kết nối (máy chủ không phát tiếng được khi mất mạng). Module vật cản nhận thêm một bộ lọc `chi_bao` do `DanTheoVach` cung cấp.

**Tech Stack:** Python 3.12 (`../.venv`), numpy, OpenCV, pytest; edge-tts (giọng HoaiMy) qua `tools/gen_vi_voice.py`; Arduino ESP32-S3 (`compile/compile.ino`, không có arduino-cli trên máy này).

**Spec:** `docs/superpowers/specs/2026-10-25-dan-theo-vach-design.md` (mục 6.3 luật nói, 7.1 vạch bị chắn, 9 kính ảo). Số liệu rà soát: bề rộng vạch đo được 0,97–1,00 m (IMG_7354/7356/7358), ~0,6 m (khiem, kém tin cậy); kính ảo cũ: lệch 0,5 m chỉ 3/12 lần có lệnh; 1 câu "Rẽ phải" sai do quay đầu 30° (IMG_7358 106 s); khi chạy cả hệ thống 71–91% câu nói là của module vật cản.

## Global Constraints

- Mọi chạy lệnh đứng ở `OpenAIglasses_for_Navigation/`, Python là `../.venv/bin/python`.
- Câu nói mới phải có WAV và có khóa trong `voice/map.vi.json` (khớp chính xác chuỗi). Sinh bằng `../.venv/bin/python tools/gen_vi_voice.py --engine edge-tts --only "<câu>"`.
- Mọi ngưỡng mới phải ghi **lý do vật lý** vào spec. Không chỉnh ngưỡng theo điểm trên video dev.
- Video giữ riêng (`../my/IMG_7357.MOV`, `IMG_7359.MOV`, `IMG_7355.MOV` 0–65 s) chạy **đúng một lần** ở Task 8, sau khi sửa xong; không chỉnh code theo kết quả đó.
- Giữ được hành vi cũ: `AIGLASS_DAN_VACH=cu` vẫn chạy đường cũ, không đụng đường cũ.
- Repo chưa commit việt hóa: bước "Commit" chỉ làm khi người dùng cho phép; nếu không thì bỏ qua.
- Không ghi `.env`/API key.

## Review Focus

1. **Vạch hẹp (0,3 m) với ngưỡng mới 0,5 bề rộng** = chỉ 15 cm: người lắc lư bình thường không được sinh "Dịch". → test `test_lac_lu_vach_hep_khong_noi` ở Task 5.
2. **Quay đầu 30° trên vạch hẹp**: sai số đo lệch `h·tanθ·sinψ/W` ≈ 0,44 ở W = 0,3 m, θ = 10°, gần ngưỡng 0,5; phải được `GOC_VE` chặn. → test `test_quay_dau_vach_hep_khong_dich` ở Task 5.
3. **Ảnh có lại sau khi mất hình**: không báo lặp; không báo khi không ở chế độ dẫn đường; vẫn báo được dù orchestrator đã bị đặt lại về IDLE lúc ESP32 rớt. → test ở Task 2a.
4. **Hành lang vạch cũ hơn 1,5 s hoặc chưa có** khi lọc vật cản: phải BÁO (an toàn hơn im). → test `test_khong_co_hanh_lang_thi_van_bao` ở Task 7.
5. **Xe đang lao tới bên cạnh vạch**: vẫn báo dù không nằm trên vạch. → test `test_xe_dang_toi_van_bao` ở Task 7.

## Việc của người dùng (song song, không chặn các task)

- Quay bằng iPhone kẹp mũ theo `../HUONG_DAN_QUAY_VIDEO_VACH.md` (cố tình lệch, chéo, quay đầu), hoặc thử trực tiếp bằng công cụ Task 1.
- Đặt mua bo XIAO ESP32S3 Sense (bo firmware đang viết cho).
- **Danh sách kiểm trước lần đầu đeo kính thật:** (1) lật gương: vật bên phải phải hiện bên phải trên `/ws/viewer` (sai = mọi lệnh trái/phải bị đảo); (2) góc nhìn ngang ống kính → `AIGLASS_HFOV`; (3) độ chúc camera trên kính; (4) số khung/giây thực qua WiFi.

---

### Task 1: Công cụ thử trực tiếp bằng iPhone chạy module dẫn vạch

**Files:**
- Modify: `../thu_nghiem_do_sau/truc_tiep/may_chu.py` (thêm `EngineDanVach`, `wav_cua_cau`; `tao_app` nhận `kich_thuoc`)
- Modify: `../thu_nghiem_do_sau/truc_tiep/chay.py` (cờ `--dan-vach`, `--hfov`)
- Modify: `../thu_nghiem_do_sau/truc_tiep/static/thu_kinh.html` (tham số URL `khung=43` cắt giữa 4:3 640×480, `fps`)
- Test: `../thu_nghiem_do_sau/truc_tiep/test_may_chu.py`

**Interfaces:**
- Produces: `EngineDanVach(lam_xau=None)` cùng giao thức `Engine` (`step(img_bgr, t) -> (câu|None, wav|None)`, `cho_noi_lai()`); chạy đúng đường hệ thống `BlindPathNavigator.process_frame` (yolo-seg + `dan_theo_vach` + module vật cản độ sâu với cờ "đang theo vạch"), đồng hồ ảo = `t` (giờ chụp trên iPhone) để luật chờ đúng cả khi phát lại. `wav_cua_cau(text) -> bytes | None` (tra `audio_player.AUDIO_MAP` như `play_voice_text`). `tao_app(engine, ma, thu_muc_phien, kich_thuoc=(640, 360))`.

Lý do: chưa có kính ESP32 → iPhone kẹp mũ là camera thật duy nhất có chuyển động đầu thật; cắt giữa 4:3 từ video 16:9 góc ~70° cho góc nhìn ngang ~55° (gần camera kính) → `AIGLASS_HFOV=55` mặc định khi `--dan-vach`. ESP32 gửi 3–8 khung/giây → `fps=8`.

- [ ] **Step 1: Test hỏng**: `test_tao_app_kich_thuoc_4_3` (gửi JPEG 640×360 với `kich_thuoc=(640, 480)` → engine nhận ảnh `shape[:2] == (480, 640)`); `test_wav_cua_cau_co_va_khong` (`"Dịch sang phải"` → bytes bắt đầu `b"RIFF"`; câu lạ → None); `test_engine_dan_vach_bat_cau_va_dong_ho_ao` (engine dựng với navigator giả có `process_frame` gọi `workflow_blindpath.play_voice_text("Dịch sang phải")` và ghi lại `workflow_blindpath.time.time()` → `step(img, 2.5)` trả `("Dịch sang phải", wav)` và đồng hồ đọc được = gốc + 2,5).
- [ ] **Step 2:** `cd .. && .venv/bin/python -m pytest thu_nghiem_do_sau/truc_tiep -q` → 3 test mới FAIL.
- [ ] **Step 3:** cài đặt; HTML: `khung=43` → canvas 640×480, `drawImage` cắt giữa (sx = (vw − vh·4/3)/2); `fps` → chu kỳ `guiKhung`; `chay.py --dan-vach` in mã QR có `&khung=43&fps=8`, đặt `AIGLASS_HFOV` (mặc định 55).
- [ ] **Step 4:** test cả thư mục PASS; chạy thật `chay.py --dan-vach --khong-ham`, đẩy 20 khung từ `../my/IMG_7356.MOV` qua WebSocket (script nhỏ) → log phiên có khung đã xử lý, không lỗi.
- [ ] **Step 5:** Commit (nếu được phép).

### Task 2a: Máy chủ báo khi ngừng nhận ảnh từ kính

**Files:**
- Create: `canh_bao_ket_noi.py`
- Modify: `app_main.py` (chỗ nhận khung `/ws/camera` ~dòng 948; thêm một tác vụ asyncio lúc khởi động)
- Create WAV + khóa map: "Mất hình ảnh từ kính, hãy dừng lại."
- Test: `tests/test_canh_bao_ket_noi.py`

**Interfaces:**
- Produces: `class CanhBaoMatHinh(nguong_s=2.0)`, `cap_nhat_khung(t: float, dang_dan: bool) -> None`, `kiem_tra(t: float) -> str` (trả câu một lần mỗi đợt mất hình, "" nếu không). `CAU_MAT_HINH = "Mất hình ảnh từ kính, hãy dừng lại."`

Lý do 2 s: đi 0,7 m/s thì 2 s là 1,4 m không được dẫn; ESP32 gửi ≥ 3 khung/giây nên 2 s không có khung là bất thường. `dang_dan` được **chốt** từ khung cuối (khi ESP32 rớt, `app_main` đặt lại orchestrator về IDLE, nếu đọc trạng thái lúc kiểm tra thì sẽ không bao giờ báo).

- [ ] **Step 1: Test hỏng**: `test_bao_sau_2s` (khung ở t = 0..1 với `dang_dan=True`, `kiem_tra(3.1)` → câu, `kiem_tra(3.5)` → ""); `test_khong_bao_khi_khong_dan` (`dang_dan=False` → ""); `test_co_lai_roi_mat_lai_thi_bao_lai`; `test_chua_tung_co_khung_thi_im`.
- [ ] **Step 2:** chạy → FAIL (chưa có module).
- [ ] **Step 3:** cài lớp; trong `app_main.py` gọi `cap_nhat_khung(time.time(), orchestrator.get_state() not in ("IDLE", "CHAT"))` mỗi khung nhận được, và một vòng `asyncio` mỗi 0,5 s gọi `kiem_tra` → `play_voice_text(câu)` khi khác "". Sinh WAV.
- [ ] **Step 4:** `pytest tests/ -q` PASS. Kiểm thật: chạy server theo skill `aiglass-run-verify`, gửi ảnh bằng client WebSocket thử 5 s rồi ngắt → log có `[Thông báo giọng nói]`/câu "Mất hình ảnh…" sau ~2 s.
- [ ] **Step 5:** Commit (nếu được phép).

### Task 2b: Firmware tự bíp khi mất kết nối

**Files:**
- Modify: `compile/compile.ino` (hàm mới `bipMatKetNoi()`; gọi trong `wsCam.onEvent` nhánh `ConnectionClosed` ~dòng 908 và khi luồng `/stream.wav` kết nối lại thất bại liên tiếp ≥ 3 lần ở nhãn `reconnect:` ~dòng 670)

**Interfaces:**
- Produces: `void bipMatKetNoi()`: 3 tiếng 1 kHz × 150 ms, cách 150 ms, ghi qua `i2sOut` (cùng định dạng stereo 32-bit đang dùng ở dòng ~715); không bíp lại trong 10 s.

Lý do: mất WiFi thì máy chủ không gửi được tiếng xuống kính, nên chỉ có kính tự báo được.

- [ ] **Step 1:** viết hàm + 2 chỗ gọi; dùng cờ `volatile` để không ghi I2S cùng lúc với tác vụ phát WAV (chỉ bíp khi tác vụ phát không đang ghi).
- [ ] **Step 2:** Không biên dịch được trên máy này (không có arduino-cli) → ghi rõ trong báo cáo: **người dùng nạp bằng Arduino IDE và thử**: tắt WiFi router khi kính đang chạy → nghe 3 tiếng bíp trong ≤ 3 s.
- [ ] **Step 3:** Commit (nếu được phép).

### Task 3: Kính ảo dùng bề rộng vạch đo được và ảnh 4:3

**Files:**
- Modify: `tools/kinh_ao.py` (`lich_kich_ban`, `chay`, `main`, `CameraAo` mặc định `h_ao`)
- Test: `tests/test_kinh_ao.py`

**Interfaces:**
- Produces: `uoc_luong_be_rong(ds_do, H, W, hfov, h_cam) -> float | None` (trung vị bề rộng vạch, mét, từ danh sách `(DoVach, y_chan_troi)` tin cậy có `|goc| < 5`; None nếu < 20 mẫu); `lich_kich_ban(t)` trả `dx` tính bằng **bề rộng vạch** (±0,8), `chay` nhân với bề rộng đo được; `chay(..., be_rong=None)` và cờ `--be-rong` để ép giá trị.
- Kết quả `cham` có thêm khóa `be_rong_m`.

- [ ] **Step 1: Test hỏng**
  - `test_uoc_luong_be_rong_mask_tong_hop`: vẽ vạch W = 1,0 m, h = 1,5, pitch 9° bằng `ve_vach` (copy hàm chiếu từ `tests/test_dan_theo_vach.py` hoặc import), cho 30 khung qua `DanTheoVach` → `abs(uoc_luong_be_rong(...) - 1.0) < 0.08`.
  - `test_uoc_luong_be_rong_it_mau`: 5 mẫu → `None`.
  - `test_camera_ao_mac_dinh_4_3`: `CameraAo(960, 540).ve(...)` trả ảnh `shape[:2] == (480, 640)`.
  - `test_kich_ban_cheo_co_troi`: trong pha chéo (psi = −20°, đi 0,7 m/s), `dx` thay đổi đúng `0.7·sin(psi)·dt` mỗi bước (người đi chéo thật thì trôi; xoay thân tại chỗ = quay đầu).
  - `test_kich_ban_dx_theo_be_rong`: `lich_kich_ban(12.0)[0] == pytest.approx(-0.8)`, `lich_kich_ban(20.0)[0] == pytest.approx(0.8)`.
- [ ] **Step 2:** `../.venv/bin/python -m pytest tests/test_kinh_ao.py -q` → 5 test mới FAIL.
- [ ] **Step 3: Cài đặt**
  - Công thức (camera chúc θ suy từ chân trời, `v` = hàng chân ảnh − H/2, `w_px` = bề rộng giữa hai mép đã khớp ở hàng đó): `W_m = w_px·h_cam / (v·cosθ + f·sinθ)`, `θ = atan((H/2 − y_chan_troi)/f)`.
  - `chay`: lượt đọc trước 10 s đầu (từ `bat_dau`, 5 khung/giây) qua `DanTheoVach` riêng để có bề rộng; < 20 mẫu thì dùng 0,6 và in cảnh báo. Đáp án: `dx_tong = -lech0 * be_rong + dx_m`. Mặc định camera kính ảo 640×480 (giống `FRAMESIZE_VGA` của firmware); `VideoWriter` theo kích thước ảnh thật.
  - Hàm `sai_chieu` giữ ngưỡng 0,15 m / 5°.
- [ ] **Step 4:** chạy lại cả `tests/test_kinh_ao.py` → PASS (sửa các test cũ đang giả định 360 dòng, ghi lý do: firmware gửi VGA 4:3).
- [ ] **Step 5:** chạy `../.venv/bin/python tools/kinh_ao.py ../my/IMG_7356.MOV --che-do kich_ban --giay 60` → in `be_rong_m` trong khoảng 0,9–1,1.
- [ ] **Step 6:** Commit (nếu được phép).

### Task 4: So 3 cách xử lý "Rẽ" trên vòng kín, chọn cách đơn giản nhất

**Files:**
- Modify: `dan_theo_vach.py` (`LuatNoi` nhận `che_do_re`: `"cu"` | `"troi"` | `"tat"`; mặc định theo kết quả)
- Modify: `tools/kinh_ao.py` (cờ `--che-do-re`, truyền qua env `AIGLASS_RE`)
- Test: `tests/test_dan_theo_vach.py`

**Interfaces:**
- Produces: `AIGLASS_RE` ∈ {`cu`, `troi`, `tat`}. `troi`: chỉ nói "Rẽ …" khi độ dốc hồi quy của `lech` theo thời gian trong cửa sổ `T_GOC` cùng dấu với `goc` và |độ dốc|/sai số chuẩn ≥ 2 (không cần mét, không cần hiệu chỉnh). `tat`: không bao giờ nói "Rẽ".

Lý do vật lý: thân đi chéo ψ thì vị trí so với vạch trôi v·sinψ (0,7 m/s, 15° → 18 cm/s); chỉ quay đầu thì không trôi (sai số đo do chúc camera là hằng số khi giữ đầu). IMU trên kính đo đầu chứ không đo thân nên không thay được tín hiệu này.

- [ ] **Step 1: Test hỏng**: `test_troi_quay_dau_khong_re` (goc = +20° giữ 3 s, lech hằng 0,1 → không "Rẽ"); `test_troi_di_cheo_that_co_re` (goc = +18°, lech tăng 0,15/s từ 0 → có "Rẽ phải một chút"); `test_tat_khong_bao_gio_re`; `test_cu_giu_nguyen_hanh_vi` (chuỗi cũ → giống trước).
- [ ] **Step 2:** chạy → FAIL. **Step 3:** cài đặt.
- [ ] **Step 4:** `pytest tests/ -q` PASS. Chạy `kinh_ao.py --che-do vong_kin --giay 300` cho 3 video dev × 3 chế độ (+ `--khong-kinh`), seed cố định → bảng: % thời gian trên vạch, lệch lớn nhất, câu/phút, số câu sai chiều. Chạy `kich_ban` (đã có trôi khi chéo từ Task 3) × 3 chế độ → "Rẽ" khi chỉ quay đầu.
- [ ] **Step 5:** chọn mặc định = chế độ đơn giản nhất (thứ tự ưu tiên `tat` < `troi` < `cu`) mà % trên vạch không kém chế độ tốt nhất quá 2 điểm và sai chiều = 0; ghi `Ruling:` + bảng vào spec mục 6.3. Commit (nếu được phép).

### Task 5: Ngưỡng "Dịch" theo mép vạch

**Files:**
- Modify: `dan_theo_vach.py:352` (`LECH_BAT`, `LECH_NHA`)
- Modify: `docs/superpowers/specs/2026-10-25-dan-theo-vach-design.md` mục 6.3 (dòng "Lệch khỏi vạch")
- Test: `tests/test_dan_theo_vach.py`

**Interfaces:**
- Produces: `LECH_BAT = 0.5`, `LECH_NHA = 0.3` (đơn vị bề rộng vạch). `da_tren_vach` và nhánh "đi chéo" dùng chung `LECH_BAT` nên tự đổi theo.

Lý do vật lý (ghi vào spec): `|lech| > 0,5` nghĩa là tâm người đã ra tới mép vạch, nửa bàn chân ở ngoài; không phụ thuộc vạch rộng hay hẹp. Ngưỡng cũ 0,75 với vạch 1 m nghĩa là người đã ra ngoài mép 25 cm mới được nhắc. Nhả ở 0,3: tâm người đã vào trong vạch, cách mép ≥ 0,2 bề rộng. Sai số do chúc camera (≤ 0,14 bề rộng ở θ = 10°, |ψ| = 12°, W = 0,4 m) vẫn nhỏ hơn khoảng cách 0,5 − 0,3.

- [ ] **Step 1: Test hỏng**
  - `test_lech_0_6_thi_dich`: chuỗi `DoVach` tin cậy lech = +0,6, goc = 0 trong 1,2 s ở 8 khung/giây → có "Dịch sang phải".
  - `test_lac_lu_vach_hep_khong_noi`: vạch W = 0,3 m vẽ bằng `ve_vach`, người lắc ngang ±0,08 m hình sin 0,9 Hz trong 20 s, `psi` = 0 → `DanTheoVach.buoc` không trả câu nào.
  - `test_quay_dau_vach_hep_khong_dich`: W = 0,3 m, d = 0, pitch 10°, `psi` = −30° (quay đầu phải) giữ 3 s → không có "Dịch sang …" (được phép có "Rẽ …", đã biết, chờ IMU).
- [ ] **Step 2:** chạy 3 test → test 1 FAIL, 2 và 3 ghi lại kết quả (nếu đã PASS thì vẫn giữ làm test chốt).
- [ ] **Step 3:** đổi hai hằng số; cập nhật mục 6.3 của spec cùng lý do ở trên.
- [ ] **Step 4:** `../.venv/bin/python -m pytest tests/ -q` → tất cả PASS. Test cũ nào hỏng vì dùng số 0,75 thì sửa giá trị trong test, không sửa luật.
- [ ] **Step 5:** Commit (nếu được phép).

### Task 6: Báo "đã thấy vạch" / "chưa thấy vạch"

**Files:**
- Modify: `dan_theo_vach.py` (`CAU`, `LuatNoi.__init__`, `LuatNoi.buoc`, `LuatNoi.huy`)
- Create WAV: `voice/vi/…` + khóa trong `voice/map.vi.json` cho "Đã thấy vạch dẫn đường." và "Chưa thấy vạch dẫn đường."
- Test: `tests/test_dan_theo_vach.py`

**Interfaces:**
- Produces: `CAU["thay"] = "Đã thấy vạch dẫn đường."`, `CAU["chua_thay"] = "Chưa thấy vạch dẫn đường."`, `T_CHUA_THAY = 5.0`.

Hành vi: lần đầu `dang_theo` thành True trong một phiên `LuatNoi` → nói "Đã thấy vạch dẫn đường." đúng một lần (các lần sau dùng "Đã quay lại đường dẫn." như cũ). Nếu từ lúc bắt đầu (khung đầu tiên đưa vào `buoc`) mà ≥ 5 s chưa từng `dang_theo` → "Chưa thấy vạch dẫn đường.", nhắc lại theo nhịp `_noi` (4 → 8 → 16 s). Lý do: người khiếm thị không phân biệt được "im vì đi đúng" với "im vì kính chưa thấy gì".

- [ ] **Step 1: Test hỏng**: `test_bao_thay_vach_mot_lan` (2 s tin cậy → đúng 1 câu "Đã thấy…"; mất 4 s rồi có lại → "Mất đường đi…" rồi "Đã quay lại…", không có "Đã thấy…" lần 2); `test_chua_thay_vach_sau_5s` (toàn `None` 25 s → "Chưa thấy…" ở ~5 s, ~9 s, ~17 s; không câu nào trước 5 s); `test_huy_cau_thay_thi_noi_lai` (`huy("Đã thấy vạch dẫn đường.")` → khung sau nói lại).
- [ ] **Step 2:** chạy → FAIL.
- [ ] **Step 3:** cài đặt; sinh 2 WAV bằng `tools/gen_vi_voice.py --engine edge-tts --only …`.
- [ ] **Step 4:** `pytest tests/ -q` PASS; `grep -c "Đã thấy vạch dẫn đường.\|Chưa thấy vạch dẫn đường." voice/map.vi.json` → 2.
- [ ] **Step 5:** Commit (nếu được phép).

### Task 7: Đo câu vật cản trên/ngoài vạch, rồi mới quyết bộ lọc

**Bước quyết định trước (Step 0):** chạy cả hệ thống đồng hồ ảo trên 4 video dev, với mỗi câu vật cản ghi vật đó có nằm trên hành lang vạch (`vat_tren_vach`) và người có đang đứng trên vạch (`da_tren_vach`) không. Nếu < 30% câu là vật ngoài vạch khi người đứng trên vạch → `Ruling:` bỏ task (lợi ít), ghi số vào spec. Ngược lại làm tiếp các bước dưới, với sửa: bộ lọc chỉ áp khi `da_tren_vach(t)` (KHÔNG phải `dang_theo_vach`) — người lệch khỏi vạch vẫn thấy vạch, vật ngay trước mặt họ không được tắt tiếng.

**Files:**
- Modify: `ne_vat_can_do_sau.py:111` (`step` thêm tham số `chi_bao=None`; lọc ngay sau khi chọn `o = cand[0]`)
- Modify: `dan_theo_vach.py` (`DanTheoVach.vat_can_bao`)
- Modify: `workflow_blindpath.py:499-501` (truyền `chi_bao` khi `da_tren_vach`)
- Test: `tests/test_dan_theo_vach.py`, `tests/test_tich_hop_vach.py`

**Interfaces:**
- Produces: `DanTheoVach.vat_can_bao(o: dict, t: float) -> bool`: True nếu không có lần đo tin cậy trong `T_GIU_DO` (1,5 s) **hoặc** `vat_tren_vach([o], do_gan, *hw)` khác rỗng. `NeVatCanDoSau.step(frame, t=None, dang_qua_duong=False, chi_bao=None)`: nếu `chi_bao` có và `chi_bao(o)` là False và `o` không `toi` (đang tới) và `o.get('src')` không thuộc `('stairs', 'curb', 'drop')` → xử lý như không có vật (`self.speaker.update(t, None, None)`, trả None). Câu "Kính không nhìn rõ…", "Đang ở lòng đường…" không bị lọc.

Lý do: khi theo vạch, vật bên cạnh vạch không chặn đường đi; gậy và vạch đã giữ người trong hành lang. Xe đang lao tới và bậc/mép đường vẫn nguy hiểm dù ở đâu.

- [ ] **Step 1: Test hỏng**
  - `test_vat_canh_vach_khong_bao` (đơn vị, `vat_can_bao` với `near` nằm ngoài hành lang → False).
  - `test_vat_tren_vach_bao` (→ True).
  - `test_khong_co_hanh_lang_thi_van_bao` (`do_gan` cũ 2 s → True).
  - `test_xe_dang_toi_van_bao` (ghép: `step` với `chi_bao=lambda o: False`, vật `toi=True` → vẫn có câu "đang tới"; dùng cách giả `det`/độ sâu như `tests/test_tich_hop_vach.py` đang làm).
  - `test_workflow_truyen_chi_bao_khi_theo_vach` (ghép: khi `da_tren_vach` True thì `step` nhận `chi_bao` khác None; thấy vạch nhưng lệch > `LECH_BAT` thì None).
- [ ] **Step 2:** chạy → FAIL.
- [ ] **Step 3:** cài đặt theo Interfaces.
- [ ] **Step 4:** `pytest tests/ -q` PASS.
- [ ] **Step 5:** Commit (nếu được phép).

### Task 8: Đo lại và báo cáo (verification-before-completion)

**Files:**
- Modify: `docs/superpowers/specs/2026-10-25-dan-theo-vach-design.md` (bảng kết quả mới), `../TIEN_DO_DAN_VACH.md`, `../NHAT_KY_PHIEN_LAM_VIEC.md`

- [ ] **Step 1:** `../.venv/bin/python -m pytest tests/ -q` → in số test, tất cả PASS.
- [ ] **Step 2:** kính ảo cho 3 video dev: `../.venv/bin/python tools/kinh_ao.py ../my/IMG_735{4,6,8}.MOV --che-do kich_ban --giay 120 --json test_out/kinh_ao/<tên>.json` → ghi: `be_rong_m`, `sai_chieu`, số động tác lệch trái/phải/chéo có lệnh đúng (trước: lệch 3/12, chéo 11/12, sai chiều 1).
- [ ] **Step 3:** `tools/danh_gia_vach.py` trên 4 video dev với `--nhan test_cham_diem/nhan_vach_dev.json` → câu sai chiều theo nhãn (phải 0), câu/phút của module vạch (trước 1–7).
- [ ] **Step 4:** chạy cả hệ thống đồng hồ ảo như `test_out/he_thong/` trên 4 video dev → tổng câu/phút và % câu của module vật cản (trước 8–12,5 câu/phút, vật cản 71–91%).
- [ ] **Step 4b:** chạy `tools/danh_gia_vach.py` và cả hệ thống trên 3 video giữ riêng (`../my/IMG_7357.MOV`, `IMG_7359.MOV`, `IMG_7355.MOV` 0–65 s) **một lần**; nếu người dùng đã quay video mới bằng iPhone kẹp mũ (Task 1) thì chạy cả những video đó. Không sửa code theo kết quả này.
- [ ] **Step 5:** báo cáo trung thực cả chỗ xấu đi (ví dụ câu "Dịch" tăng do ngưỡng thấp hơn), nêu rõ số nào là trên video dev đã xem (nguy cơ lạc quan) và việc còn chờ người dùng: nạp firmware Task 5, video đi lệch có chủ ý, góc nhìn ống kính thật, kiểm lật gương.
