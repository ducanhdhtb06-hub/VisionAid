# Dẫn theo gạch dẫn đường — Giai đoạn 1 — Kế hoạch thực hiện

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay phần ra lệnh dẫn theo vạch (đang bảo rẽ ngược chiều 98% khi lệch) bằng đo điểm tụ + bề rộng vạch, có cổng "không chắc" và luật nói thưa.

**Architecture:** Module thuần NumPy/OpenCV `dan_theo_vach.py` (đo → đường chân trời → luật nói) không phụ thuộc model; `workflow_blindpath.py` gọi nó trong trạng thái NAVIGATING khi `AIGLASS_DAN_VACH` khác `cu`. Đánh giá offline bằng `tools/danh_gia_vach.py` trên video dev.

**Tech Stack:** Python 3.12, NumPy, OpenCV, pytest 9 (venv `glassai/.venv`), Ultralytics chỉ trong công cụ đánh giá (model vạch có sẵn).

**Spec:** `docs/superpowers/specs/2026-10-25-dan-theo-vach-design.md`

## Global Constraints

- Mọi lệnh chạy từ `OpenAIglasses_for_Navigation/` với `../.venv/bin/python`.
- Câu nói chỉ dùng chuỗi đã có trong `voice/map.vi.json`: `"Dịch sang trái"`, `"Dịch sang phải"`, `"Rẽ trái một chút"`, `"Rẽ phải một chút"`, `"Mất đường đi, đang tìm lại."`, `"Đã quay lại đường dẫn."`.
- Ngưỡng lấy nguyên từ spec 6.2–6.3; không chỉnh theo điểm video. Đổi ngưỡng phải ghi lý do vật lý vào spec.
- `AIGLASS_DAN_VACH=cu` phải cho lại đúng hành vi cũ.
- Không commit trừ khi người dùng yêu cầu (quy tắc phiên); thay bước commit bằng chạy toàn bộ test.
- Video giữ riêng (`IMG_7357`, `IMG_7359`, `IMG_7355`) không được chạy trong kế hoạch này.

## Review Focus

1. Mask rỗng / None / chỉ vài đốm nhiễu → không lỗi, `tin_cay=False`.
2. Dải gạch rộng hơn ảnh ở chân (đứng ngay trên vạch, góc rộng) → các hàng chạm mép bị bỏ; nếu hết hàng → không chắc, không lỗi.
3. Hai mép gần song song (điểm tụ ở vô cực) → không chia cho 0, `tin_cay=False`.
4. Thời gian nhảy cóc (rơi khung, t tăng 5 s) → không phát lệnh cũ dựa trên cửa sổ đã hết hạn.
5. Mask giá trị 0/255 hoặc bool thay vì 0/1 → kết quả như nhau.

---

## Cấu trúc file

- Create `dan_theo_vach.py` — đo (`do_vach`), `BoChanTroi`, `LuatNoi`, mặt tiền `DanTheoVach`.
- Create `tests/test_dan_theo_vach.py` — mask tổng hợp bằng phép chiếu camera + test đo, cổng, luật nói.
- Create `tests/test_tich_hop_vach.py` — `BlindPathNavigator` dùng module mới.
- Modify `workflow_blindpath.py` — `__init__` (khởi tạo + bỏ ONBOARDING), `_handle_navigating` (rẽ nhánh), `reset`.
- Create `tools/danh_gia_vach.py` — đánh giá offline cũ vs mới trên video, xuất câu/phút, % tin cậy, ảnh mẫu, chấm sai chiều theo nhãn.
- Create `test_cham_diem/nhan_vach_dev.json` — nhãn mắt (Claude chấm) vị trí vạch so với người trên khung mẫu dev.

### Task 1: Đo lệch và góc từ mask

**Files:** Create `dan_theo_vach.py`, `tests/test_dan_theo_vach.py`

**Interfaces:**
- Produces: `@dataclass DoVach(lech: float, goc: float, x_vp: float, y_vp: float, tin_cay: bool, ly_do: str, mep_trai: tuple[float,float], mep_phai: tuple[float,float])` (mép = hệ số `x = a*y + b`); `do_vach(mask: np.ndarray, hfov_deg: float = 70.0, y_chan_troi: float | None = None) -> DoVach | None` (None khi không có vạch).
- Test helper (trong file test): `ve_vach(d_m, psi_deg, W_m=0.4, h_cam=1.5, pitch_deg=10, hfov=70, H=360, Wimg=640, z_max=12, che=None) -> np.ndarray` chiếu dải gạch mặt đất qua camera lỗ kim (chúc `pitch_deg`) rồi tô đa giác.

- [ ] **Step 1: Viết test**
  - `test_tam_vach_thang_truoc_mat`: `ve_vach(0, 0)` → `lech≈0 (±0.1)`, `goc≈0 (±1.5)`, `tin_cay`.
  - `test_lech_khong_phu_thuoc_goc` (parametrize `d∈{-0.4,0.3}`, `psi∈{-12,0,12}`): `lech ≈ d/0.4 (±0.1)`, `goc ≈ psi (±1.5)`.
  - `test_khong_can_hieu_chinh` (parametrize `pitch∈{5,10,20}`, `h_cam∈{1.3,1.6}`): `ve_vach(0.3, 8, ...)` → `lech≈0.75 (±0.1)`, `goc≈8 (±1.5)`.
  - `test_dau_lech`: vạch bên phải (`d=+0.4`) → `lech>0`; `psi=+10` → `goc>0`.
  - `test_mask_rong_tra_none`: mask toàn 0 → `None`; mask None → `None`.
  - `test_mask_255_va_bool` : `ve_vach(...)*255` và `.astype(bool)` cho cùng `lech` (±1e-6).
- [ ] **Step 2:** `../.venv/bin/python -m pytest tests/test_dan_theo_vach.py -q` → FAIL (ImportError).
- [ ] **Step 3:** Viết `DoVach`, `do_vach` theo spec 6.2 bước 1–5 (thành phần lớn nhất, 60% dưới, bỏ hàng chạm mép/hẹp < 6 px, khớp mép lặp 3 lần ngưỡng `max(2, 2.5×trung vị)`, giao mép → điểm tụ; `lech=(tâm_chân − x_vp)/rộng_chân`, `goc=deg(atan((x_vp−cx)/f))`, `f=(W/2)/tan(hfov/2)`). Cổng để Task 2 (tạm `tin_cay=True` nếu đủ hàng).
- [ ] **Step 4:** chạy lại → PASS.

### Task 2: Cổng "không chắc" và đường chân trời

**Files:** Modify `dan_theo_vach.py`, `tests/test_dan_theo_vach.py`

**Interfaces:**
- Consumes: `do_vach`, `DoVach`.
- Produces: `class BoChanTroi(cua_so_s=5.0, toi_thieu=10)` với `cap_nhat(y_vp: float, t: float) -> None`, `gia_tri() -> float | None`. `do_vach(..., y_chan_troi=...)` áp cổng chân trời ±6° (đổi sang px: `f*tan(6°)`); khi `y_chan_troi is None` thì đòi `y_vp < H/2`.

- [ ] **Step 1: Viết test** (mỗi test kiểm `tin_cay is False` và `ly_do` chứa từ khóa):
  - `test_ngan_qua` (`z_max=1.5`) → `"ngắn"`.
  - `test_mot_mep_bi_che` (`che=` hình chữ nhật đè 40% chiều cao mép phải) → `"che"`.
  - `test_nga_ba` (mask vạch thẳng + một nhánh ngang ở 70% chiều cao) → `"nhánh"`.
  - `test_mep_song_song` (hai mép song song dọc ảnh) → `"điểm tụ"`, không exception.
  - `test_chan_troi_lech` (`y_chan_troi` = giá trị đúng + `f*tan(10°)`) → `"chân trời"`; với giá trị đúng → `tin_cay`.
  - `test_vach_rong_hon_anh`: `ve_vach(0,0,W_m=3.0)` (mọi hàng chân chạm mép) → không exception, `tin_cay False` hoặc None.
  - `test_bo_chan_troi`: 12 mẫu quanh 100 trong 5 s → `gia_tri()≈100`; mẫu cũ hơn 5 s bị bỏ; < 10 mẫu → None.
- [ ] **Step 2:** chạy → FAIL.
- [ ] **Step 3:** Thêm 6 cổng theo bảng spec 6.2 (ngưỡng: 15 hàng, 12% H, sai số/bề rộng 0,15, mép giữ < 70%, điểm tụ phải trên chân vạch, ±6°, bề rộng thật > 2× suy ra). `BoChanTroi` lưu `deque[(t, y)]`, trả trung vị.
- [ ] **Step 4:** chạy toàn file → PASS.

### Task 3: Luật nói

**Files:** Modify `dan_theo_vach.py`, `tests/test_dan_theo_vach.py`

**Interfaces:**
- Consumes: `DoVach`.
- Produces: `class LuatNoi` với `buoc(do: DoVach | None, t: float) -> str` (chuỗi Global Constraints hoặc `""`). Hằng: `LECH_BAT=0.75, LECH_NHA=0.4, GOC_BAT=15, GOC_NHA=8, GOC_VE=5, T_LECH=1.0, T_GOC=1.5, TI_LE=0.7, NHAC_LAI=4.0, T_MAT=3.0, T_TIM_LAI=1.0`.

- [ ] **Step 1: Viết test** (helper `chay(luat, chuoi_do, fps=10)` trả list `(t, câu)` không rỗng):
  - `test_dung_thi_im`: 10 s `lech=0.1, goc=2` → không câu nào.
  - `test_lech_phai_sau_1s`: `lech=1.2, goc=0` → câu đầu `"Dịch sang phải"` ở `t∈[1.0,1.3]`; lặp lại không sớm hơn 4 s; `lech=-1.2` → `"Dịch sang trái"`.
  - `test_dang_tu_quay_ve_thi_im`: `lech=1.2, goc=-8` → im.
  - `test_cheo`: `lech=0.2, goc=20` → `"Rẽ phải một chút"` ở `t≥1.5`; `goc=12` sau khi đã bật vẫn giữ (trễ), `goc=5` → nhả.
  - `test_dich_truoc_re`: `lech=1.2, goc=20` (cùng dấu, đang đi xa) → `"Dịch sang phải"`.
  - `test_khung_khong_chac_khong_tinh`: xen kẽ `tin_cay=False` vẫn ra câu khi các khung tin cậy thỏa ≥ 70%.
  - `test_mat_va_tim_lai`: 2 s tốt, 3,5 s None → `"Mất đường đi, đang tìm lại."` đúng 1 lần; rồi 1,2 s tốt → `"Đã quay lại đường dẫn."`.
  - `test_nhay_coc_thoi_gian`: 0,5 s `lech=1.2` rồi khung kế tiếp t+5 s `lech=1.2` → không nói ngay ở khung đó.
- [ ] **Step 2:** chạy → FAIL.
- [ ] **Step 3:** `LuatNoi` giữ `deque[(t, DoVach)]` ≤ 2 s; "ổn định T" = trong `[t−T, t]` có ≥ 3 khung tin cậy, ≥ 70% thỏa, và khung tin cậy cũ nhất ≥ `t−T+0.1` *(để nhảy cóc không tính)*; trạng thái bật/nhả có trễ; nhắc lại cùng câu ≥ 4 s.
- [ ] **Step 4:** chạy → PASS.

### Task 4: Ghép vào `workflow_blindpath.py`

**Files:** Modify `dan_theo_vach.py` (mặt tiền), `workflow_blindpath.py`; Create `tests/test_tich_hop_vach.py`

**Interfaces:**
- Produces: `class DanTheoVach(hfov_deg=float(os.getenv("AIGLASS_HFOV", 70)))` với `buoc(mask, t) -> tuple[str, DoVach | None]` và `ve(do, frame_visualizations, H, W) -> None` (thêm `polyline` hai mép, `circle` điểm tụ, `data_panel` lệch/góc/lý do theo định dạng `_draw_visualizations` đang dùng).
- `BlindPathNavigator.dan_vach: DanTheoVach | None`.

- [ ] **Step 1: Viết test** (`AIGLASS_OBS_ENGINE=cu` để không nạp model độ sâu; tạo `BlindPathNavigator()` không model; `monkeypatch` `workflow_blindpath.time.time`):
  - `test_mac_dinh_bat_va_vao_thang_navigating`: `nav.dan_vach is not None`, `nav.current_state == "NAVIGATING"`.
  - `test_tat_bang_bien_moi_truong`: `AIGLASS_DAN_VACH=cu` → `dan_vach is None`, `current_state == "ONBOARDING"`.
  - `test_navigating_lech_phai`: gọi `_handle_navigating(ve_vach(0.5,0), ...)` 15 lần cách 0,1 s → có lần trả `"Dịch sang phải"`; với `ve_vach(0,0)` → luôn `""` và không bao giờ `"Trích xuất đặc trưng đường đi thất bại"`.
  - `test_khong_vao_maneuvering_turn`: mask chữ L 30 khung → `current_state` vẫn `"NAVIGATING"`.
- [ ] **Step 2:** chạy → FAIL.
- [ ] **Step 3:** Trong `__init__` (sau khối `ne_vat_can`): tạo `dan_vach` trừ khi `AIGLASS_DAN_VACH` ∈ {`cu`,`old`}; nếu có thì `current_state = STATE_NAVIGATING`; `reset()` cũng vậy và gọi lại `DanTheoVach()`. Trong `_handle_navigating`: nếu `self.dan_vach`: chạy khối vật cản như cũ, rồi `cau, do = self.dan_vach.buoc(mask, time.time())`, `self.dan_vach.ve(...)`, trả `cau`; bỏ qua phát hiện khúc cua và `_get_pixel_domain_features`.
- [ ] **Step 4:** chạy cả `tests/` → PASS; `python -c "import workflow_blindpath"` không lỗi.

### Task 5: Đánh giá offline trên video dev

**Files:** Create `tools/danh_gia_vach.py`, `test_cham_diem/nhan_vach_dev.json`

**Interfaces:**
- Consumes: `DanTheoVach`, `do_vach`; model `model/yolo-seg.pt` (lớp 1).
- Produces: CLI `danh_gia_vach.py VIDEO... --fps 10 --out DIR [--nhan FILE] [--mau 2.0]` in mỗi video: số khung, % có vạch, % tin cậy, câu/phút **mới** và **cũ** (logic cũ sao chép như spike: góc tiếp tuyến bậc 2 ngưỡng 10°, lệch ở 0,4H ngưỡng 0,15, rẽ trước dịch, không lọc câu), danh sách câu mới theo thời gian; lưu ảnh mẫu mỗi `--mau` giây (có vẽ) vào `DIR/<video>/`; nếu có `--nhan`: đếm câu "Dịch/Rẽ" mới và cũ **sai chiều** so với nhãn khung gần nhất (≤ 1 s).
- Nhãn JSON: `{"mo_ta": ..., "nguoi_cham": "Claude (nhìn ảnh gốc, chưa xem đầu ra module mới)", "khung": [{"video", "t", "vach": "trai|giua|phai|khong_ro"}]}` — `trai` nghĩa là vạch ở bên trái người đi (đúng câu phải là "dịch/rẽ trái" hoặc im).

- [ ] **Step 1:** Trích ~60 khung ngẫu nhiên (seed cố định) từ 4 video dev **gốc, chưa vẽ gì**; Claude xem và chấm `vach` → `nhan_vach_dev.json`, trước khi xem đầu ra module mới cho các khung đó.
- [ ] **Step 2:** Viết công cụ; chạy trên `khiem.mp4`, `../my/IMG_7354.MOV`, `../my/IMG_7356.MOV`, `../my/IMG_7358.MOV` và bản `../loi_do_sau/video_kinh/{khiem,IMG_7354,IMG_7356,IMG_7358}_kinh.mp4`.
- [ ] **Step 3:** Đạt khi: 0 câu mới sai chiều theo nhãn; câu/phút mới ≪ 31; ghi số liệu thật (kể cả chỗ kém) vào spec mục 6.5 và `NHAT_KY_PHIEN_LAM_VIEC.md`.
- [ ] **Step 4:** Xuất 1–2 video có vẽ + tiếng cho người dùng xem vào `../video_ket_qua/vach_gd1/` bằng `tools/test_navigation.py ... --save` (đường đi thật của hệ thống).
