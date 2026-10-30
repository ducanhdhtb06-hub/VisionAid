# Dẫn theo gạch dẫn đường (vạch cho người khiếm thị) — thiết kế

Ngày: 2026-10-25. Trạng thái: người dùng đã cho làm tiếp mà không cần hỏi thêm; mọi quyết định không do người dùng nói đều ghi là **[Claude quyết]**.

## 1. Mục đích và lý do

Kính AI cho người khiếm thị Việt Nam (khởi nghiệp). Người dùng muốn tạm gác né vật cản tổng quát (khó ra MVP) để làm trước **MVP dẫn đi dọc gạch dẫn đường** (dải gạch vàng có gờ trên vỉa hè). Có dải gạch nghĩa là đã biết đường đi, nên bài toán hẹp và đo được.

Người dùng giữ **dạng kính** (camera trên đầu), không dùng điện thoại đeo ngực.

## 2. Hiện trạng (đo 2026-10-23 và 2026-10-25)

- Nhận vạch: `model/yolo-seg.pt` (lớp 1 = `blind_path`) thấy vạch ở 81–100% thời gian trên 6 video `my/` và 84% trên `khiem.mp4`; xem 69 khung bằng mắt, không thấy khung tô nhầm.
- Ra lệnh (`workflow_blindpath.py`, `_generate_navigation_guidance`) sai nặng:
  - "rẽ" lấy độ nghiêng dải gạch trên ảnh ở chân ảnh → thực chất là độ lệch ngang (tương quan hạng +0,85 trên 4 video dev). Khi người lệch > 0,5 bề rộng vạch, **98% (354/361) lệnh rẽ là ngược chiều**.
  - "dịch" đo ở 40% chiều cao ảnh (gần điểm tụ) và xét sau "rẽ" → gần như không nói.
  - Phát hiện khúc cua ảo 18% thời gian; đọc cả thông báo lỗi nội bộ; ~31 câu/phút.

## 3. Mục tiêu / không làm / thất bại dù chạy

**Mục tiêu MVP (4 việc):** tìm vạch; giữ người trên vạch; báo sự kiện (vạch bị chắn, vạch rẽ, hết vạch); tìm lại vạch.

**Không làm trong MVP:** dẫn vòng qua vật cản (chỉ báo "dừng"), qua đường, đèn giao thông, bậc xuống/hố.

**Coi là thất bại dù code chạy:** có câu chỉ hướng sai chiều; nói nhiều đến mức người dùng tắt kính; vạch bị chắn mà im lặng.

## 4. Ràng buộc

- Không overfit 9 video: ngưỡng phải có lý do vật lý/hình học; video giữ riêng không dùng để chỉnh.
- Câu nói phải có sẵn WAV trong `voice/map.vi.json` (khớp chính xác chuỗi), câu mới thì sinh bằng `tools/gen_vi_voice.py`.
- Camera trên đầu: hướng nhìn (yaw) của đầu ≠ hướng đi. Độ lệch ngang không bị ảnh hưởng bởi việc quay đầu; góc đi chéo thì có.
- `yolo-seg.pt` là AGPL (train bằng Ultralytics) → dùng tạm cho MVP thử nghiệm.
- Giữ được hành vi cũ để so: biến môi trường `AIGLASS_DAN_VACH=cu`.

## 5. Các cách đã cân nhắc

| Cách | Mô tả | Đánh đổi | |
|---|---|---|---|
| **A. Điểm tụ + bề rộng vạch** | Hai mép dải gạch → điểm tụ → góc đi chéo; (tâm − điểm tụ)/bề rộng ở cùng hàng → độ lệch tính bằng "bề rộng vạch" | Không cần hiệu chỉnh camera; cần đoạn vạch thẳng đủ dài, phải có cổng "không chắc" | **Chọn** |
| B. Chiếu xuống mặt đất | Đổi ảnh sang bản đồ mặt đất bằng độ chúc + chiều cao camera | Cho số mét thật nhưng phụ thuộc hiệu chỉnh (bản 26 tự hiệu chỉnh thất bại); trên kính độ chúc đổi liên tục | Để sau, cho câu "2 m nữa rẽ" |
| C. Vá ngưỡng | Đảo thứ tự dịch/rẽ, nâng ngưỡng | Vẫn trộn hai đại lượng; ngưỡng dò theo video = overfit | Loại |

Bản thử (spike) cách A trên 1.814 khung dev: tương quan góc–lệch +0,85 → −0,13; ổn định (đổi trung vị 0,5° và 0,03W mỗi 0,5 s); hỏng khi vật che mép vạch và ở ngã ba/chữ L.

## 6. Thiết kế giai đoạn 1 (cốt lõi MVP)

### 6.1 Hình học (vì sao không cần hiệu chỉnh)

Camera lỗ kim, tiêu cự `f` (pixel), tâm ảnh `cx`. Dải gạch thẳng, bề rộng `W`, tâm dải cách người `d` theo phương ngang (dương = vạch ở bên phải), hướng dải lệch `ψ` so với trục camera. Tại độ sâu `Z`:

- `x_tâm(Z) = cx + f·tanψ + f·d/Z`, `bề_rộng(Z) = f·W/Z`
- Điểm tụ: `x_vp = cx + f·tanψ`
- `(x_tâm − x_vp) / bề_rộng = d / W` — **không phụ thuộc ψ, Z, độ cao hay độ chúc camera**.
- `ψ = atan((x_vp − cx)/f)` — chỉ cần `f` (từ góc nhìn ngang, `AIGLASS_HFOV`, mặc định 70°).
- Chính xác hơn, khi camera chúc xuống góc θ: số đo lệch = `(d·cosψ − h·tanθ·tanψ)/W` (lệch tại một điểm sau chân người h·tanθ). Sai số ≤ ~0,14W ở θ = 10°, |ψ| = 12°, h = 1,5 m, W = 0,4 m — nhỏ hơn ngưỡng 0,5W (trước 0,75W); được test kiểm đúng giới hạn này.

### 6.2 Module đo `dan_theo_vach.py` (mới, trong `OpenAIglasses_for_Navigation/`)

`do_vach(mask, hfov_deg) -> DoVach | None` với `DoVach(lech, goc, x_vp, y_vp, tin_cay, ly_do)`:

1. Lấy thành phần liên thông lớn nhất của mask vạch.
2. Chỉ dùng phần gần (60% dưới của đoạn vạch nhìn thấy) để tránh khúc rẽ phía xa **[Claude quyết, từ spike]**.
3. Mỗi hàng: mép trái/phải; bỏ hàng chạm mép ảnh hoặc quá hẹp.
4. Khớp đường thẳng cho từng mép theo **đồng thuận**: đường qua 2 hàng được nhiều hàng nằm trong dung sai (8% bề rộng vạch, tối thiểu 2 px) nhất, rồi bình phương tối thiểu trên các hàng đó. *(Sửa khi làm Task 2: cách "loại điểm lạc theo trung vị" ban đầu khớp ra đường thỏa hiệp khi ~50% mép bị che, cổng "che" không bắt được.)*
5. Giao hai mép → điểm tụ; `lech`, `goc` theo 6.1.

**Cổng "không chắc"** (trả `tin_cay=False` và lý do) — mỗi cổng có lý do vật lý:

| Cổng | Lý do |
|---|---|
| < 15 hàng dùng được, hoặc đoạn vạch < 12% chiều cao ảnh | quá ngắn để khớp đường thẳng |
| sai số khớp mép / bề rộng > 0,15 | mép không thẳng (vạch vỡ, cong) |
| một mép giữ < 70% số hàng sau khi loại điểm lạc | mép bị vật che (xe đè lên vạch) |
| điểm tụ không nằm phía trên chân vạch | hai mép không hội tụ về phía xa → không phải đoạn vạch thẳng đi ra xa |
| điểm tụ lệch khỏi đường chân trời ước lượng quá ±6° (đổi sang pixel bằng `f`) | mọi đường thẳng trên mặt đất hội tụ về chân trời; đầu cúi/ngẩng tức thời hiếm khi > 6° so với trung bình vài giây |
| bề rộng thật ở một hàng > 2× bề rộng suy từ hai mép | có nhánh rẽ / ngã ba |

**Đường chân trời** (`BoChanTroi`): trung vị `y_vp` trong ~5 s gần nhất; mẫu quá 5 s hết hạn kể cả khi không có mẫu mới. Cần ≥ 5 mẫu trải ≥ 1 s (để ESP32 3 khung/giây, nửa khung bị che vẫn học được); chưa đủ thì lần đo chưa được tin nhưng vẫn dùng để học (camera gắn ngang/ngẩng thì chân trời nằm dưới giữa ảnh). Học từ mọi lần đo **qua các cổng khác** (kể cả lần chỉ trượt cổng chân trời) — *(sửa sau đánh giá video dev: bản đầu chỉ học từ lần đo tin cậy và chỉ xóa mẫu cũ khi có mẫu mới, nên khi người cúi/ngẩng > 6° thì chân trời bị khóa vĩnh viễn ở góc cũ; "điểm tụ xa chân trời" là lý do loại khung nhiều nhất).*

### 6.3 Luật nói `LuatNoi` (trong cùng module)

Đầu vào mỗi khung: `DoVach` (hoặc None) + thời gian. Đầu ra: câu (chuỗi có trong `map.vi.json`) hoặc "".

| Tình huống | Điều kiện | Câu | Lý do ngưỡng |
|---|---|---|---|
| Lệch khỏi vạch | `|lech| > 0,5` giữ ổn định ≥ 1 s; nhả khi `< 0,3` (Task 5, 2026-10-26; trước: 0,75 / 0,4) | "Dịch sang phải" / "Dịch sang trái" (theo dấu `lech`) | `|d| > 0,5W`: tâm người đã ra tới mép vạch, nửa bàn chân ở ngoài — không phụ thuộc vạch rộng hay hẹp (ngưỡng cũ 0,75 với vạch 1 m = đã ra ngoài mép 25 cm mới nhắc). Nhả 0,3: tâm người đã vào trong, cách mép ≥ 0,2W; sai số do chúc camera (≤ 0,14W) vẫn nhỏ hơn khoảng 0,5 − 0,3. Kịch bản kính ảo 3 video: pha lệch nhắc đúng 5/12 → 9/12, sai chiều 0 → 0, pha chéo 8/12 → 7/12 |
| Đang tự quay về | đang lệch nhưng `ψ` hướng về phía vạch (dấu `ψ` ngược dấu `lech`, `|ψ| ≥ 5°`) | im | người đã tự sửa, không nói chồng |
| Đi chéo | `|ψ| > 15°` giữ ≥ 1,5 s, `|lech| ≤ LECH_BAT` (0,5); nhả khi `< 8°` | "Rẽ phải một chút" / "Rẽ trái một chút" (theo dấu `ψ`) | 15° đi 2 m lệch ~0,5 m ≈ 1,3 bề rộng vạch; đòi lâu hơn vì quay đầu nhìn cũng làm đổi `ψ` |
| Đúng | còn lại | im | đi đúng thì im |
| Mất vạch | đang theo vạch, ≥ 3 s **không thấy vạch** (không có mask, hoặc đoạn vạch quá ngắn) | "Mất đường đi, đang tìm lại." (một lần) | đo "không chắc" (bị che, ngã ba) mà vạch vẫn còn thì im, không báo mất *(sửa sau đánh giá: bản đầu báo mất 8 lần/6 phút ở IMG_7354 khi vạch vẫn thấy rõ)* |
| Tìm lại | sau "mất vạch", có lại ≥ 1 s tin cậy | "Đã quay lại đường dẫn." | |

- "Ổn định ≥ T" = đã quan sát liên tục T giây (không hở > 1 s giữa hai khung), có ≥ 2 khung tin cậy **trải ≥ 70% cửa sổ**, và ≥ 70% khung tin cậy thỏa điều kiện *(sửa sau rà soát độc lập: bản đầu chỉ cần 3 khung tin cậy bất kỳ → 0,3 s số đo giữa lúc quay đầu đủ để bảo rẽ)*.
- **Nhả lệnh theo dấu** (`dấu_lệnh × lệch < 0,4`, tương tự góc) và xóa lệnh đang bật khi > 1 s không có số đo tin cậy *(sửa sau rà soát: bản đầu nhả theo trị tuyệt đối nên khi người bước quá tay sang hẳn bên kia vạch, kính lặp lệnh cũ = SAI CHIỀU)*.
- Cùng một đợt lệch/chéo: nhắc lại sau 4 s, rồi 8 s, rồi tối đa mỗi 16 s; đợt mới bắt đầu lại từ 4 s **[Claude quyết]**.
- Dịch xét trước rẽ.

#### 6.3b Sửa sau rà soát (2026-10-25 tối, kế hoạch `plans/2026-10-25-dan-theo-vach-sau-ra-soat.md`)

- **Rẽ khi quay đầu (Task 4, 2026-10-26): mặc định `AIGLASS_RE=troi`.** Chỉ nói "Rẽ…" khi độ lệch đang trôi đúng chiều góc chéo (độ dốc hồi quy 2 s ≥ 2 sai số chuẩn). Đo: kịch bản kính ảo 3 video — sai chiều 0 mọi chế độ; lệnh đúng ở pha chéo cu 8/12, troi 8/12, tat 1/12 (tat loại); video thật IMG_7358 cu = troi. Vòng kín không phân biệt được các chế độ (người ảo không có hành vi dò lại khi mất vạch) — chưa có bằng chứng % trên vạch.

- **"Đã thấy vạch dẫn đường."** một lần khi bắt được vạch lần đầu trong phiên (câu "Dịch/Rẽ" phát cùng lúc thì thay cho nó); **"Chưa thấy vạch dẫn đường."** nếu 5 s từ lúc bắt đầu chưa từng theo được vạch, nhắc 4 → 8 → 16 s. Lý do: người khiếm thị không phân biệt được "im vì đi đúng" với "im vì kính chưa thấy gì". `AIGLASS_THONG_BAO_VACH=0` tắt.
- **Mất hình từ kính:** máy chủ nói "Mất hình ảnh từ kính, hãy dừng lại." khi đang dẫn mà 2 s không có khung (mốc tính cả lúc xử lý xong một khung, vì `process_frame` chạy trên vòng sự kiện); firmware bíp 3 tiếng khi mất WebSocket camera (tối đa mỗi 10 s). Lý do: 2 s ở 0,7 m/s = 1,4 m không được dẫn; mất WiFi thì máy chủ không gửi tiếng được.
- **`AIGLASS_RE`** (`cu` mặc định / `troi` / `tat`): cách xử lý "Rẽ" khi camera trên đầu không phân biệt được quay đầu với quay thân. `troi`: chỉ nói Rẽ khi độ lệch đang trôi đúng chiều góc chéo (thân đi chéo ψ thì trôi v·sinψ; quay đầu thì không). Chọn mặc định theo thí nghiệm kính ảo (Task 4).
- Bề rộng vạch thật ở `my/` ≈ 0,97–1,00 m (đo hình học, h = 1,5 m), xác nhận bằng mắt (~2 viên lát vỉa hè), không phải 0,4 m như giả định cũ của kính ảo.
- **Rủi ro nhận dạng (phát hiện trên kính ảo):** ảnh làm xấu kiểu kính khiến yolo-seg bỏ sót vạch thật và tô nhầm dải chân tường (thẳng, hội tụ đúng chân trời → qua mọi cổng hình học) → số đo lệch sai dấu. Không cổng hình học nào loại hết được; cách gốc là tự train model trên ảnh camera kính (có mẫu âm: chân tường, mép vỉa hè).

### 6.4 Ghép vào dự án

- `BlindPathNavigator.__init__`: `self.dan_vach = DanTheoVach()` trừ khi `AIGLASS_DAN_VACH=cu`.
- Khi bật: bỏ bước ONBOARDING (vào thẳng NAVIGATING, luật "dịch" đã lo việc lên vạch); trong `_handle_navigating` gọi `self.dan_vach.buoc(mask, t)` thay cho phát hiện khúc cua + `_generate_navigation_guidance`; không bao giờ trả "Trích xuất đặc trưng đường đi thất bại". Vật cản (bản 28) giữ nguyên đường đi riêng (ưu tiên 100).
- Luồng giới hạn câu trong `process_frame` *(sửa sau rà soát)*: câu vật cản miễn giới hạn 1,2 s (module độ sâu chỉ trả câu một lần mỗi sự kiện); câu dẫn vạch không qua lọc trùng cũ, nếu bị chặn thì báo lại `dan_vach.chua_noi` để thử khung sau; `guidance_text` trả về đúng câu đã phát (app_main phát lại nó); `state_info["state"]` = "NAVIGATING" chỉ khi đang đứng trên vạch (orchestrator dùng sau khi qua đường); model YOLO lỗi/không có → `None`, không bịa dải giả.
- Vẽ lên ảnh: hai mép, điểm tụ, `lech`, `goc`, lý do không chắc (để gỡ lỗi khi xem video).

### 6.5 Kiểm chứng

1. **Test đơn vị trên mask tổng hợp** (vẽ dải gạch bằng phép chiếu camera với `d`, `ψ`, độ chúc, độ cao đã biết): đo lại được `d/W` (sai ≤ 0,1) và `ψ` (sai ≤ 1,5°) với nhiều độ chúc/độ cao khác nhau — chứng minh không phụ thuộc hiệu chỉnh. Thêm mask bị che một mép, mask ngã ba, mask quá ngắn → `tin_cay=False`.
2. **Test luật nói**: chuỗi đo giả lập theo thời gian → đúng câu, đúng lúc, im khi đúng, không lặp trong 4 s, quay về thì im.
3. **Đánh giá offline trên video dev** (`khiem`, `IMG_7354`, `IMG_7356`, `IMG_7358`, cả bản gốc lẫn bản giả lập kính `loi_do_sau/video_kinh/*_kinh.mp4`): câu/phút, % khung tin cậy, và **số câu sai chiều** so với nhãn mắt (Claude tự chấm mẫu khung: vạch ở trái/giữa/phải người đi — ghi rõ là nhãn của Claude).
4. **Video giữ riêng** (`IMG_7357`, `IMG_7359`, `IMG_7355` 0–65 s, video đi lệch quay sau): chỉ chạy sau khi xong cả 3 giai đoạn.

**Đạt giai đoạn 1 khi:** 0 câu sai chiều trên nhãn mẫu; câu/phút giảm mạnh so với 31; test xanh.

**Kết quả đo (2026-10-25, video dev, 10 khung/giây, nhãn mắt 60 khung do Claude chấm — chỉ 6 khung vạch lệch rõ trái/phải vì người quay gần như luôn đi trên vạch):**

| | khiem | IMG_7354 | IMG_7356 | IMG_7358 |
|---|---|---|---|---|
| Có vạch (yolo-seg) gốc / giả lập kính | 87% / 80% | 97% / 84% | 100% / 90% | 100% / 93% |
| Đo tin cậy trên khung có vạch, gốc / kính | 81% / 80% | 80% / 78% | 79% / 82% | 78% / 79% |
| Câu/phút mới, gốc / kính | 7,0 / 7,3 | 3,4 / 2,7 | 1,1 / 0,7 | 1,8 / 2,3 |
| Lệnh cũ đổi/phút (chưa lọc), gốc | 26,5 | 42,1 | 55,7 | 30,6 |

- Câu sai chiều theo nhãn: **mới 0** (gốc và kính); **cũ 5/5** ở bản gốc, 2/2 ở bản kính.
- Dấu độ lệch ở khung nhãn trái/phải: 4 đúng / 0 sai (gốc), 3 / 0 (kính). Khung nhãn "giữa": |lệch| trung vị 0,07–0,25, lớn nhất 0,98 (một khung IMG_7354).
- Kiểm thêm bằng mắt 12 câu mới (kể cả ngoài khung nhãn): 12/12 đúng phía.
- `khiem` 155–187 s nhắc "Dịch sang phải" mỗi 4 s: đúng — người quay đi bên trái dải gạch cả đoạn.
- Hạn chế: mẫu lệch ít (6 khung) → chưa đủ để khẳng định "không bao giờ sai chiều"; cần video cố tình đi lệch/chéo. Số đo trên 4 video dev (đã xem khi sửa) có nguy cơ lạc quan; video giữ riêng chưa chạy.

## 7. Giai đoạn 2 và 3 (phác thảo, thiết kế chi tiết sau khi giai đoạn 1 đạt)

- **GĐ2 — vạch bị chắn / hết vạch:** dùng đầu ra bản 28 (D-FINE + độ sâu), chỉ xét vật nằm trên hành lang dải gạch phía trước (nối dài hai mép đã khớp) trong ~5 m → "Phía trước có vật cản, dừng lại." (đã có WAV). Hết vạch: vạch còn thấy nhưng đầu xa của nó dừng lại gần (đoạn nhìn thấy ngắn dần nhiều giây) → câu mới cần sinh WAV.

### 7.1 Thiết kế GĐ2a — vạch bị chắn (2026-10-25, [Claude quyết], người dùng cho tự quyết)

- **Không thêm câu mới, không để 2 module cùng nói về 1 vật.** Bản 28 đã gọi tên + khoảng cách ("Dừng lại. Xe máy, 3 mét"); GĐ2a chỉ đổi hành vi của nó khi đang theo vạch.
- **Đang theo vạch** (vạch thấy trong 1,5 s gần nhất): truyền cờ "chỉ dừng" vào `NeVatCanDoSau.step` (như `dang_qua_duong`) → bản 28 **không bao giờ bảo "dịch"** (MVP: kính không dẫn vòng, người dùng tự vòng bằng gậy).
- **Hành lang vạch**: đa giác giữa hai mép của lần đo tin cậy gần nhất (≤ 1,5 s; mép bị vật che thì lần đo hiện tại không tin cậy, dùng lần trước), từ chân ảnh tới điểm tụ.
- **Vật trên vạch**: vật của bản 28 (`cand`, `dist` ≤ 5 m) có **dải chân** (10% hàng thấp nhất của vùng `near`) phủ ≥ 30% bề rộng hành lang tại cùng hàng. Lý do: vật chạm đất trên ≥ 1/3 bề rộng vạch (~12 cm của 40 cm) là gậy/chân sẽ vướng.
- **Khi có vật trên vạch** (và 4 s sau khi hết thấy): `LuatNoi` **tắt "dịch/rẽ" và "mất vạch"** — người đang vòng qua vật bằng gậy, kéo họ về vạch lúc này là chỉ vào vật. Khi lại tin cậy ổn định 1 s sau khi đã báo "mất vạch" → "Đã quay lại đường dẫn." như cũ.
- **Hết vạch** (GĐ2b): xem 7.2.
- Kiểm: test đơn vị (hành lang, dải chân, tạm tắt luật nói), test ghép (cờ chỉ dừng truyền vào step); video: IMG_7358 ~10 s (xe phủ bạt đè vạch), IMG_7354 các đoạn xe máy đè vạch — kiểm bằng mắt không có câu "dịch về vạch" lúc vật còn trên vạch.
### 7.2 GĐ2b — sắp hết vạch (2026-10-25, TẮT mặc định, `AIGLASS_HET_VACH=1` để bật)

- Báo "Sắp hết vạch dẫn đường." (WAV mới, edge-tts HoaiMy) một lần khi, ổn định 1 s: đo tin cậy; đầu xa của dải gạch trong 5 m (khoảng cách = h / tan(góc dưới chân trời), h = `AIGLASS_NE_HCAM`, mặc định 1,5 m); đầu vạch không chạm mép ảnh; không có vật (module né vật cản) đè lên đầu vạch; trên đường kéo dài phía sau đầu vạch không còn ≥ 20% mask vạch (rãnh nước/vết nứt cắt ngang); **đầu vạch tiến lại gần ≥ 0,3 m trong ~1 s** (đầu vạch thật đứng yên trên đất; ranh giới do vạch mờ/ảnh kém đi theo người).
- Đã báo hết thì vạch biến mất trong 10 s sau đó không báo "Mất đường đi".
- Vì sao tắt: video dev không có chỗ hết vạch thật; 7 lần báo còn lại trên dev chủ yếu là xe đỗ đè vạch (chỉ loại được khi module né vật cản thấy xe). Bật khi đã kiểm trên video có động tác 10 của `HUONG_DAN_QUAY_VIDEO_VACH.md` và chạy cả hệ thống.

### 7.3 GĐ3 — vạch rẽ chữ L / ngã rẽ (2026-10-25, bật mặc định)

- Hành lang thẳng lấy từ **đoạn vạch gần chân** (25 hàng dưới cùng của mảnh vạch dưới chân, khớp đồng thuận) — không cần phép đo chung tin cậy, vì ở chỗ rẽ thật phía xa thường lộn xộn (mép bị che, điểm tụ nhảy) nhưng đoạn dưới chân vẫn thẳng.
- **Nhánh ngang**: ≥ 3 hàng liên tiếp có mask lấn ra ngoài hành lang ≥ 1,5 bề rộng vạch ở một bên. Vạch chính còn ≥ 20% hành lang phía xa sau nhánh → ngã rẽ ("Phía trước có ngã rẽ vạch."), không còn → chữ L ("Vạch rẽ trái/phải phía trước."); nhánh hai bên → ngã rẽ.
- Báo khi chỗ rẽ ≤ 4 m, ổn định 1 s (khung "có nhánh rẽ" cũng được tính), **tiến lại gần ≥ 0,3 m trong ~1 s** (lịch sử giữ từ xa); một lần mỗi chỗ rẽ, quên sau 3 s không thấy.
- Dev (chưa có đáp án độc lập): 2 lần báo, IMG_7358 61,7 s trùng ngã ba thật; chỗ chữ L IMG_7356 ~40 s chưa bắt được (ảnh lộn xộn). Chờ nhãn của tác tử chấm mù.

## 9. Giả lập kính (`tools/kinh_ao.py`, 2026-10-25)

- **Hình học chính xác**: xoay camera (quay đầu, cúi/ngẩng, nghiêng, quay thân) = homography thuần xoay; dịch ngang (bước lệch khỏi vạch) = homography mặt phẳng đất (dải gạch nằm trên đất nên đúng tuyệt đối; vật đứng bị méo nhẹ). Pixel nhìn đất trong 40 m dùng homography đất, còn lại dùng phép xoay. 8 test kiểm: bước lệch 0,3 m đo ra đúng −0,75W; quay thân 10° đo ra −10°; quay đầu 15° cho ảnh y hệt quay thân 15°.
- Camera kính ảo: góc nhìn 56° (hẹp hơn điện thoại để còn lề khi xoay/dịch), 640×360, đầu lắc theo bước (±1,5° cúi 1,8 Hz, ±2° quay, ±1° nghiêng), làm xấu ảnh như ESP32 (`gia_lap_kinh.LamXau`: nhòe, nhiễu, JPEG 35), 8 khung/giây có rung.
- Chạy qua **đúng hệ thống** (`NavigationMaster` → `BlindPathNavigator.process_frame`, đồng hồ ảo).
- **Đáp án**: lệch/chéo của người quay video nguồn (đo trên ảnh nguồn, trung vị 1 s) + phần giả lập. Sai chiều = lệnh dịch/rẽ ngược phía cần đi khi lệch > 0,15 m / chéo > 5°.
- `kich_ban`: 10 động tác của `HUONG_DAN_QUAY_VIDEO_VACH.md` lặp mỗi 60 s. `vong_kin`: người mù ảo (lệch hướng ±0,5°/s + nhiễu 2°/√s, phản xạ 1 s, bước 0,4 m / xoay 10°, thỉnh thoảng quay đầu 20–30°), so có kính / không kính.
- Giới hạn: không mô phỏng được vật đứng thay đổi theo góc nhìn, ống kính góc rộng méo, ảnh thật của cảm biến OV2640/OV3660; người mù ảo là mô hình đơn giản.

- **GĐ3 — vạch rẽ chữ L / ngã ba (phác thảo cũ):** dùng cổng "nhánh" của 6.2 + hình dạng mask phía xa. Chưa chắc chiều → "Vạch đổi hướng phía trước, đi chậm lại" (câu mới).

## 8. Rủi ro

- Quay đầu trên kính làm `ψ` đổi mà hướng đi không đổi → giảm bằng cách đòi ổn định 1,5 s và ngưỡng 15°; sau này dùng IMU có sẵn trên bo (ICM42688).
- Ống kính góc rộng (ESP32-P4/OV5647 sau này) làm đường thẳng cong → cần hiệu chỉnh méo **một lần cho mỗi loại ống kính**.
- Vỉa hè VN bị chắn nhiều → kính hay im; đo tỉ lệ khung tin cậy để biết.
- Ảnh ESP32 xấu → chưa biết yolo-seg còn thấy vạch không; thử bằng video giả lập kính, sau đó bằng kính thật.

## 9. Để lại cho người làm

Tên hàm/biến cụ thể, cách vẽ gỡ lỗi, cấu trúc test. Các ngưỡng ở 6.2–6.3 là giá trị ban đầu có lý do vật lý; **không** chỉnh theo điểm trên video dev — nếu phải đổi, ghi lý do vật lý vào đây.
