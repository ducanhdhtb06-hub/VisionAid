# Vật chắn trên vạch — Giai đoạn 1 — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Khi người đang ở trên vạch: im với vật cạnh vạch (khi chắc chắn), báo vật chắn ở ~3 m kèm bên vòng (không phía
lòng đường), im để gậy dẫn, xác nhận "Đã về vạch".

**Architecture:** Hai mảnh logic thuần (test không cần model): `dan_theo_vach.phan_loai_vat` (chắn / cạnh / không chắc,
theo lối đi 3D trên vạch) và module mới `vat_chan_vach.BoVatChan` (các bước báo → im → về vạch). Module vật cản lộ ra
hình học 3D khung hiện tại, nhận bộ lọc `chi_bao`, cho biết bên đi được và ghép câu mới. `workflow_blindpath` nối
lại: lọc câu module vật cản, chạy bộ điều khiển sau dẫn vạch, tắt lệnh dẫn vạch khi bộ điều khiển đang xử lý.

**Spec:** `docs/superpowers/specs/2026-10-27-vong-vat-can-tren-vach-design.md`

## Global Constraints
- Chỉ module dẫn theo vạch, chỉ khi người ở trên vạch (`DanTheoVach.da_tren_vach`); ngoài vạch như cũ.
- Lối đi trên vạch = dải vạch nới mỗi bên 0,2 m; vật chắn = phần vật cao 0,15–2 m trong lối đi, trong 5 m.
- Nói khi vật ≤ 3,0 m. Câu: "{Tên} chắn vạch, vòng bên trái." / "… vòng bên phải." / "Vạch bị chắn, dừng lại, hãy
  dùng gậy." / "Đã về vạch." / "Vạch ở bên phải." / "Vạch ở bên trái." / "Chưa thấy lại vạch, hãy dùng gậy."
- Hết chắn = không thấy vật chắn ≥ 1,5 s; nhắc thông tin sau 15 s rời vạch chưa về (1 lần).
- Luôn báo như cũ: `toi`, `nhanh`, curb, stairs, drop, hole, `high`.
- Công tắc `AIGLASS_VAT_CHAN_VACH` (mặc định "1"; "0" = hành vi cũ).
- Test: `../.venv/bin/python -m pytest tests/ -q` (đứng ở `OpenAIglasses_for_Navigation/`). Không commit khi chưa được
  phép. Không dùng video giữ riêng (IMG_7355/7357/7359).

## Review Focus
1. Xe che mép vạch → lần đo vạch không tin cậy → phải "không chắc" (báo), không im. → Task 2 `test_khong_co_do_vach_thi_khong_chac`.
2. Tay lái nhô vào lối đi, chân ngoài vạch → "chắn". → Task 2 `test_phan_nho_vao_loi_di_la_chan`.
3. Ngay sau câu báo người còn đứng trên vạch → không nói "Đã về vạch". → Task 3 `test_khong_ve_vach_khi_chua_roi`.
4. Câu bộ điều khiển thua câu khác (không được phát) → nói lại khung sau. → Task 3 `test_chua_noi_thi_bao_lai`, Task 5.
5. Bên lòng đường không bao giờ được chọn. → Task 4 `test_xep_ben_bo_ben_duong`.

### Task 1: Câu nói mới + từ khóa khẩn
**Files:** `../thu_nghiem_do_sau/tao_giong.py` (MANH), sinh WAV vào `../thu_nghiem_do_sau/giong/`; `uu_tien_cau.py`
(`TU_KHAN` thêm "chắn vạch", "bị chắn"); test `tests/test_uu_tien_cau.py`.
- [ ] Test: `la_cau_khan("Xe máy chắn vạch, vòng bên trái.")`, `la_cau_khan("Vạch bị chắn, dừng lại, hãy dùng gậy.")`
  → True; `la_cau_khan("Đã về vạch.")` → False. Đỏ → sửa → xanh.
- [ ] Thêm mảnh: `chan_vach` "chắn vạch", `vong_ben_trai` "vòng bên trái", `vong_ben_phai` "vòng bên phải",
  `vach_bi_chan` "Vạch bị chắn, dừng lại, hãy dùng gậy", `da_ve_vach` "Đã về vạch", `vach_ben_phai` "Vạch ở bên phải",
  `vach_ben_trai` "Vạch ở bên trái", `chua_thay_lai_vach` "Chưa thấy lại vạch, hãy dùng gậy". Chạy `tao_giong.py`
  (chỉ sinh file còn thiếu); kiểm 8 file tồn tại.

### Task 2: `dan_theo_vach.phan_loai_vat`
**Files:** `dan_theo_vach.py`; test `tests/test_vat_chan_vach.py`.
**Interfaces:** `phan_loai_vat(o: dict, do: DoVach|None, geo: dict|None, hw_vach: (H,W), le_m=0.2, xa_m=5.0) -> str`
trả `"chan" | "canh" | "khong_chac"`. `o["mask"]` cùng cỡ với `geo["X"]` (Hg, Wg); mép vạch của `do` theo cỡ `hw_vach`
→ quy đổi điểm ảnh sang cỡ geo; lấy X, Z của các điểm mép ở ≥ 4 hàng dưới điểm tụ, khớp X_trái(Z), X_phải(Z) tuyến tính;
điểm vật cao 0,15–2 m, Z ≤ xa_m, X trong [X_trái − le, X_phải + le] ≥ 20 điểm → "chan"; không khớp được mép / thiếu
`do` / `geo` → "khong_chac"; còn lại "canh".
- [ ] Test hỏng: `test_vat_tren_vach_la_chan`, `test_vat_canh_vach_la_canh`, `test_phan_nho_vao_loi_di_la_chan`,
  `test_khong_co_do_vach_thi_khong_chac`, `test_vat_xa_hon_5m_khong_chan`. (geo giả: mặt đất phẳng, camera cao 1,5 m.)
- [ ] Cài → xanh → cả bộ test.

### Task 3: Bộ điều khiển `vat_chan_vach.BoVatChan`
**Files:** tạo `vat_chan_vach.py`; test `tests/test_vat_chan_vach.py`.
**Interfaces:** `BoVatChan.buoc(t, tren_vach: bool, vat: dict|None, lech: float|None, thay_vach: bool) -> str` — `vat`
= vật chắn gần nhất {dist, ten (tiếng Việt hoặc None), ben: list["trai"|"phai"] theo thứ tự ưu tiên}; trả câu ("" nếu
không). `.dang_xu_ly -> bool` (ĐÃ BÁO / CHẮN HẾT). `.chua_noi()`: câu vừa trả không phát được → về trạng thái trước để
báo lại khung sau.
- [ ] Test hỏng: `test_bao_khi_3m_va_chot_ben`, `test_khong_bao_khi_khong_tren_vach`, `test_chan_het`,
  `test_ve_vach_sau_khi_roi`, `test_khong_ve_vach_khi_chua_roi`, `test_vat_di_mat_thi_im`, `test_nhac_15s_mot_lan`,
  `test_nhac_khong_thay_vach`, `test_chua_noi_thi_bao_lai`, `test_giu_ben_da_chon_tu_xa`.
- [ ] Cài → xanh.

### Task 4: Module vật cản lộ thông tin + lọc + ghép câu
**Files:** `ne_vat_can_do_sau.py`; test `tests/test_vat_chan_vach.py`.
**Interfaces:** `step(..., chi_bao=None)`: `self.cand` giữ đủ (cho module vạch); nếu `chi_bao` thì chỉ xét vật
`chi_bao(o)` True khi chọn câu. `self.geo` = hình học khung hiện tại (None nếu không có). `ben_di_duoc(o) -> list[str]`
(dùng `plan_avoid` với `prefer=None`, xếp bằng `xep_ben`). Hàm thuần `xep_ben(free: dict, ti_le_duong: dict) -> list`:
bỏ bên không đi được, bỏ bên có tỉ lệ lòng đường ≥ 0,3, ưu tiên tỉ lệ lòng đường thấp rồi dịch ít. `ghep_cau(keys:
list[str], text: str) -> str`: ghép mảnh WAV (khoảng nghỉ như Speaker) và đăng ký vào AUDIO_MAP; trả `text`.
- [ ] Test hỏng (hàm thuần): `test_xep_ben_bo_ben_duong`, `test_xep_ben_uu_tien_xa_duong`, `test_xep_ben_khong_ben_nao`.
- [ ] Cài → xanh.

### Task 5: Nối vào `workflow_blindpath` + công tắc
**Files:** `workflow_blindpath.py`, `dan_theo_vach.py` (`DanTheoVach.giu_im`: True thì `co_vat_chan`), test
`tests/test_tich_hop_vach.py` (sửa giả `_NeGia.step` nhận `**kw`).
- [ ] `chi_bao(o)`: True nếu loại luôn báo; nếu đang trên vạch (hoặc bộ điều khiển đang xử lý): False khi
  `phan_loai_vat` là "chan" hoặc "canh"; True khi "khong_chac". Ngoài vạch và bộ điều khiển rảnh: None (như cũ).
- [ ] Sau khi có câu dẫn vạch: lấy vật chắn đứng yên gần nhất, gọi `BoVatChan.buoc`; câu ra đưa vào ứng viên nguồn
  `obstacle` (ưu tiên 100), thay câu module vật cản trừ khi câu đó là "đang tới"; không phát được → `chua_noi()`.
  `dan_vach.giu_im = bo.dang_xu_ly`.
- [ ] Test hỏng: `test_vat_canh_vach_im_khi_tren_vach`, `test_vat_chan_bao_vong`, `test_tat_bang_bien_moi_truong_vat_chan`.
- [ ] Cài → xanh → cả bộ test.

### Task 6: Chạy thật trên video dev + ESP32 ảo
- [ ] `phat_lai` các đoạn có vật đè vạch (IMG_7358 0–30 s, IMG_7354, IMG_7356 30–60 s): xuất video, xem bằng mắt câu
  đúng lúc/đúng bên. Quét cấu hình `goc` 4 video dev: sai chiều = 0, câu/phút không tăng so với `quet_sau2/goc`.
- [ ] Ghi kết quả vào `TIEN_DO_DAN_VACH.md`.
