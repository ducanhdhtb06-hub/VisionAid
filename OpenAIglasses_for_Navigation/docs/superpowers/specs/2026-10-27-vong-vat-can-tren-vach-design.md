# Module dẫn theo vạch: vật chắn trên vạch — Thiết kế

Ngày: 2026-10-27. **Phạm vi: chỉ module dẫn theo vạch (`dan_theo_vach.py`), chỉ khi người đang ở trên vạch.**
Ngoài vạch và các module khác (qua đường, đèn, tìm đồ, app iPhone, hiệu chỉnh khoảng cách) giữ nguyên.
Người dùng chọn cách làm "bộ điều khiển có các bước", duyệt hướng ("oke tiến hành") với 3 sửa đổi Claude đề xuất
(mục 5.3, 4.1, 6). Mục ghi **[Claude quyết]** là quyết định của Claude, người dùng đọc lại để sửa.

**Giai đoạn (người dùng duyệt 2026-10-27, "tiến hành"):** làm **giai đoạn 1** = toàn bộ tài liệu này (báo + chọn bên,
im đoạn giữa để gậy dẫn, "Đã về vạch", lọc vật cạnh vạch). Để sau, chỉ làm khi thử thật cho thấy cần:
- GĐ2 "Đi thẳng." khi người đã lệch đủ ra khỏi vật (camera còn thấy vật — đo được);
- GĐ3 "Đã qua, dịch sang … về vạch." dựa trên **quãng đường đã đi** (vật khuất khỏi khung khi người còn cách đầu xe
  ~0,85 m, nên "vạch phía trước đã thông" chưa đủ) — đo bằng mặt đất trôi trong ảnh hoặc IMU trên kính thật, kiểm sai
  ≤ 15% so với quãng đường của cảnh 3D (COLMAP) mới bật.

## 1. Mục đích

Người khiếm thị bám theo vạch. Khi đang ở trên vạch, kính chỉ nói về vật **chắn trên vạch**: báo sớm, chỉ **bên còn
trống để vòng** (không bao giờ phía lòng đường), để người dùng gậy đi vòng, rồi **xác nhận khi đã về vạch**. Vật đứng
cạnh vạch thì im. Thay hành vi hiện tại (spec `2026-10-25-dan-theo-vach-design.md` mục 7.1: chỉ "Dừng lại. X, n mét",
không chỉ bên, không xác nhận).

## 2. Mục tiêu / không làm / coi là thất bại

**Mục tiêu:** im với vật cạnh vạch (khi chắc chắn); báo vật chắn ở ~3 m kèm bên vòng; xác nhận "Đã về vạch".

**Không làm:** ra lệnh "quay lại vạch" theo thời gian (kính không thấy lúc người đi ngang vật — mục 3); IMU / đếm bước;
hiệu chỉnh khoảng cách (đo 2026-10-27: kính đoán gần hơn thật ~10% khi điều kiện tốt — chấp nhận); hành lang cong theo
vạch (vật được tìm trong lối đi thẳng 1,2 m trước mặt như hiện tại); mọi thứ khi người không ở trên vạch.

**Coi là thất bại dù code chạy:** im với vật thật sự chắn lối (kể cả phần nhô ra như tay lái, gương xe); chỉ bên vòng là
phía lòng đường; im khi xe đang lao tới / sắp xuống lòng đường / cầu thang; nói nhiều hơn hiện tại; câu chỉ hướng sai
chiều.

## 3. Ràng buộc vật lý

- Camera kính hẹp (~56° ngang, ~43° dọc), mù gần chân: nhìn thẳng thì thấy đất từ ~3,8 m (cúi 10°: 2,4 m); vật cao ~1 m
  lệch 0,5 m ra khỏi khung khi còn ~1 m. → **Chọn bên khi còn xa và chốt lúc nói.** → **Kính không biết lúc nào người
  đã đi qua hết vật** → không ra lệnh quay về; gậy dẫn đoạn này, kính chỉ xác nhận khi thấy người đã về vạch.
- Câu nói phải ngắn: người đi ~0,6–0,8 m/s, câu 2,5 s = đi thêm 1,5–2 m trong lúc nghe.

## 4. Nói về vật nào (khi người đang ở trên vạch: `DanTheoVach.da_tren_vach(t)`)

### 4.1 Vật chắn hay vật cạnh vạch [sửa đổi 2]
- **Lối đi trên vạch** = dải vạch (hai mép của lần đo tin cậy gần nhất, ≤ 1,5 s) **nới mỗi bên 0,2 m** cho vai người.
  [Claude quyết: 0,2 m]
- Vật **chắn** nếu **bất kỳ phần nào** của nó cao 0,15–2 m (không chỉ chân vật) nằm trong lối đi đó, trong 5 m. Xét
  theo vị trí 3D trên mặt đất của từng điểm vật (module vật cản đã có độ sâu + mặt đất), không so điểm ảnh ở hàng khác
  nhau. Lý do: tay lái, gương xe nhô ra ngang hông/ngực trong khi bánh xe ngoài vạch; gậy quét sát đất không chạm.
- Vật **cạnh vạch** (im) chỉ khi **chắc chắn**: có lần đo vạch tin cậy ≤ 1,5 s **và** dựng được lối đi 3D **và** mọi
  phần của vật nằm ngoài lối đi. Thiếu bất kỳ điều nào → coi như chắn / báo như hiện tại (xe đè vạch hay che mất mép
  vạch → lần đo "không chắc" → không được im).

### 4.2 Luôn báo như hiện tại (không lọc)
Xe/người đang lao tới; mép lòng đường, "Đang ở lòng đường…"; cầu thang (và hố/bậc nếu bật); vật treo ngang đầu;
"Kính không nhìn rõ…".

## 5. Bộ điều khiển vật chắn (logic thuần, test được không cần model)

### 5.1 Các bước
```
BÌNH THƯỜNG --(vật chắn đứng yên, ≤ 3 m, người trên vạch)--> ĐÃ BÁO (bên đã chốt)  | không bên nào --> CHẮN HẾT
ĐÃ BÁO --(người đã rời vạch rồi đứng lại trên vạch, vật không còn chắn phía trước ≥ 1,5 s)--> BÌNH THƯỜNG: "Đã về vạch."
ĐÃ BÁO --(người không rời vạch, vật hết chắn ≥ 1,5 s — vật đã đi / thấy nhầm)--> BÌNH THƯỜNG (im)
ĐÃ BÁO --(15 s sau khi rời vạch chưa về)--> nói 1 lần: thấy vạch ở một bên -> "Vạch ở bên phải." / "… trái.";
                                             không thấy vạch -> "Chưa thấy lại vạch, hãy dùng gậy."  --> BÌNH THƯỜNG
CHẮN HẾT --(vật hết chắn ≥ 1,5 s, hoặc 15 s)--> BÌNH THƯỜNG
bất kỳ --(vật chắn MỚI, khác vật cũ)--> làm lại từ đầu
```

### 5.2 Báo và chọn bên [sửa đổi 3: câu ngắn, không số mét]
- Theo dõi vật chắn gần nhất từ khi thấy (≤ 5 m); mỗi khung hỏi module vật cản bên nào đi được (đã có `plan_avoid`:
  dải rộng 1 người cạnh vật, không có gì nhô lên tới sau vật 1,5 m, trong khung, **không phải phần lớn là lòng đường**).
  Giữ bên đã chọn nếu vẫn đi được.
- Khi vật ≤ 3,0 m (ước lượng) nói một câu:
  - "{Tên} chắn vạch, vòng bên trái." / "… vòng bên phải."
  - không bên nào: "Vạch bị chắn, dừng lại, hãy dùng gậy."
  Bên được **chốt** lúc nói.
- Hai bên đều được: bên xa lòng đường hơn; không biết lòng đường thì bên dịch ít hơn. [Claude quyết]
- Câu báo là câu khẩn (cắt câu thường, có bíp): thêm "chắn vạch" và "bị chắn" vào từ khóa khẩn. [Claude quyết]

### 5.3 Sau khi báo [sửa đổi 1: không ra lệnh quay về theo giờ]
- Kính **im** về vạch: tắt lệnh dẫn vạch thường ("Dịch sang…", "Rẽ…", "Mất đường đi…") và module vật cản không bảo
  "dịch" (vẫn báo vật khác chắn lối đi thẳng: "Dừng lại. X, n mét", và mục 4.2).
- Xác nhận "Đã về vạch." khi người **đã rời vạch rồi đứng lại trên vạch** (tránh nói ngay sau câu báo lúc người chưa
  bước) và vật không còn chắn phía trước.
- Nhắc thông tin (không ra lệnh bước) đúng 1 lần sau 15 s rời vạch chưa về (mục 5.1). [Claude quyết: 15 s]
- Thay cơ chế im cố định 4 s hiện có (`T_SAU_CHAN`) khi bộ điều khiển chạy.

## 6. Câu nói mới (WAV, ghép mảnh như module vật cản đang làm)
"chắn vạch" · "vòng bên trái" · "vòng bên phải" · "Vạch bị chắn, dừng lại, hãy dùng gậy." · "Đã về vạch." ·
"Vạch ở bên phải." · "Vạch ở bên trái." · "Chưa thấy lại vạch, hãy dùng gậy."
Giọng edge-tts (HoaiMy) như hiện có — sẽ thay ở bản sau.

## 7. Cách làm

| Cách | Mô tả | Quyết định |
|---|---|---|
| 1. Sửa nhẹ | lọc theo vạch, nói "vòng bên…", về vạch bằng im cố định 4 s | Không: 4 s là đoán |
| **2. Bộ điều khiển có các bước** | mục 5, chuyển bước theo điều kiện thật, logic thuần có test | **Chọn** |
| 3. Thêm IMU đếm bước | biết người đã đi qua vật bao xa | Để sau: chờ kính thật |

Nối vào code (người làm tự chọn chi tiết): module vật cản cần lộ ra thông tin 3D của khung (để xét lối đi trên vạch) và
bên đi được cho một vật bất kỳ, và nhận bộ lọc "vật này có được nói không"; `workflow_blindpath.py` gọi bộ điều khiển
sau module vật cản và dẫn vạch, quyết định câu cuối. Công tắc biến môi trường (mặc định bật; tắt = hành vi 7.1 cũ).

## 8. Kiểm chứng
- **Test tự động (viết trước):** mọi lối chuyển mục 5.1; chốt bên khi nói; không chọn bên lòng đường; tay lái nhô vào
  lối đi (chân ngoài vạch) → coi là chắn; vạch đo "không chắc" → không im; vật cạnh vạch chắc chắn → im; xe đang tới vẫn
  báo; không "Đã về vạch" khi người chưa rời vạch; nhắc 15 s đúng 1 lần; người không ở trên vạch → như cũ.
- **Video dev + ESP32 ảo** (không dùng video giữ riêng): các đoạn vật đè vạch (IMG_7358 ~10 s; IMG_7354 các đoạn xe máy
  đè vạch; IMG_7356 ~40 s) — xem bằng mắt câu đúng lúc/đúng bên; sai chiều = 0; câu/phút không tăng.
- **Bắt buộc thử đi thật bằng iPhone** (người trong video không làm theo lệnh).

## 9. Cập nhật sau thử trên video (28/10, theo yêu cầu người dùng)
- Chỉ quan tâm vật trong vùng vạch: vật ngoài dải vạch (kể cả xe đỗ chỉ chạm lề vai) im; vật chắn > 3 m im.
- Câu nói rõ vị trí rồi hướng đi: "Xe máy chắn vạch bên phải. Đi sang trái." / "… chắn giữa vạch …". Không đọc số mét.
- Không còn "Vạch bị chắn, dừng lại": không bên nào chắc trống -> "… Đi sang <phía xa lòng đường>, dò gậy." hoặc
  "… Dùng gậy dò đường vòng."; không bao giờ hướng sang lòng đường. Bên đi vòng chọn theo số đông khung 5 m -> 3 m.
- Một người nói: khi đang theo vạch (thấy vạch trong 5 s), xe đỗ chỉ bộ điều khiển được nói; module vật cản chỉ báo
  thông tin ("Xe máy, phía trước") hoặc nguy hiểm luôn báo (xe đang tới nói rõ "đang tới", mép đường 2/3 khung, ...).
- Kiểm chứng: `tools/esp32_ao/mau_thuan.py` quét câu mâu thuẫn: 7 video 18 cặp -> 0.

## 10. Xe tới gần che mép vạch (29/10, theo yêu cầu người dùng)
Người dùng xem demo IMG_7354 285–355 s: "bảo đi vòng rồi lại 'Xe máy, phía trước'", "phải nói là xe máy lấn vạch, đi
vòng sang trái"; "làm y hệt cái demo trong tất cả trường hợp". Nhật ký tạm (phát lại 270–360 s) chỉ ra một nguyên nhân:
tới gần (~3 m), chính chiếc xe che mép vạch -> phân loại "khong_chac" (và có lúc mất cả lần đo vạch) ->
- đã bảo vòng: bộ điều khiển tưởng xe đã đi (hết chắn 1,5 s) -> module vật cản nói "Xe máy, phía trước" về chính xe đó;
- chưa bảo vòng (xe "chan" ở 3,4–3,7 m): tới 3 m thì "khong_chac" hoặc "không trên vạch" -> chỉ câu thông tin.
Sửa (lý do vật lý, không thêm ngưỡng): xe không vì bị nhìn gần mà hết chắn — vật "khong_chac" nối tiếp vật chắn vừa
thấy (< T_HET, không xa hơn quá VAT_MOI) vẫn là chắn (`NhoVatChan`, giữ cả vị trí đo được gần nhất); mất đo vạch
< T_ROI_MIN chưa là rời vạch (`BoVatChan.tren_gan`, cùng định nghĩa rời vạch khi đã báo). "canh" (đo lại được vạch, xe
nằm ngoài) giữ nguyên.
Hai lỗi cùng họ thấy khi thử IMG_7358 (29/10):
- "Đang tới" nhầm với xe đỗ (99,3 s, ngay sau "Đi sang trái."): hai xe đỗ liền nhau bị ghép thành một vết (khoảng cách
  nhảy 3,8 -> 2,0 m) -> trung vị tốc độ vọt lên 2,07 m/s. "Đang tới" giờ cần cả nửa giây gần nhất tiến lại >= V_TOI
  (`ne_vat_can_do_sau.dang_toi`, cùng 0,5 s tối thiểu của bộ theo dõi); xe chạy tới thật (76,6 / 88,1 / 104,8 s) vẫn báo.
- Đi ngang xe đỗ, YOLOE mất dấu lập vết mới ("chưa đo được tốc độ" = không đứng yên) -> bộ điều khiển tưởng xe đã đi.
  Vết mới nối tiếp xe đỗ vừa thấy (`NhoVatChan.tiep`) (kể cả tốc độ ngang thoáng qua khi người đi chéo, IMG_7354 26,3 s) vẫn là xe đỗ (`NeVatCanDoSau.co_the_dang_do`; người, con vật,
  xe đang tới thì không).
Thêm (29/10, cùng mục tiêu "mọi trường hợp giống demo"):
- Xe vừa được báo "đang tới" < 3 s mà giờ cần lệnh dừng: chỉ nói "Dừng lại." (`Speaker.ngan`), không lặp cả câu.
- Không bao giờ bảo đi về đúng nửa vạch xe đang đè ("Xe máy chắn vạch bên phải. Đi sang phải" — IMG_7354 218,8 s):
  -> "Dùng gậy dò đường vòng."
- "Đang tới" cần kích thước trên ảnh to ra hợp với độ sâu (`kich_thuoc_hop_ly`: máy ảnh lỗ kim, cao_ảnh x khoảng cách
  ~ không đổi; to ra < nửa phần độ sâu báo = độ sâu nhảy). Lỗi thật IMG_7354 299,7 s: xe đạp đỗ 11 m báo "đang tới".
- Người/con vật không dùng hai phép kiểm tra "đang tới" của xe đỗ (người không bao giờ "đỗ"; IMG_7354 45 s mất
  "Người, bên trái, đang tới" vì tốc độ nửa giây dao động quanh ngưỡng).
- Xe che kín vạch (không bao giờ phân loại được): đang theo vạch (thấy vạch trong 5 s), vật đứng yên giữa lối đi
  (lệch trục <= 0,2 m = "phía trước") trong 5 m là vật chắn; mất đo vạch mà chưa có bằng chứng đã rời vạch
  (`BoVatChan.buoc(theo_vach=True)`) vẫn được bảo vòng. Đã đo được là người ở ngoài vạch: vẫn chỉ báo thông tin.
  (IMG_7354 325,7 s, IMG_7358 10 s, khiem 131 s: trước đây chỉ "Xe máy, phía trước".)
  Chỉ khi MẤT đo vạch: đo được vạch thì đã biết vật có trên vạch không; "phía trước" camera không phải lối trên vạch
  khi người nhìn chéo (khiem 57,9 s: "Ô tô chắn vạch" sai).
- Đo được rõ người đứng trên vạch thì không báo "Đang ở lòng đường" (vạch lát trên vỉa hè; IMG_7354 65,6 s báo nhầm).
  Bước xuống đường khi đi vòng thì không còn trên vạch: vẫn báo (`ne_vat_can_do_sau.o_long_duong`).
- Còn xe đang tới trong tầm nhìn thì bộ điều khiển chưa bảo vòng (người phải dừng trước; IMG_7355 147 s: "…đang tới"
  -> "Dừng lại." -> 0,2 s sau "Đi sang trái." về phía xe). Bộ quét thêm loại `vong_ve_phia_xe`.
  Vừa báo xe đang tới < 3 s (khoảng "cùng một xe" của Speaker) cũng chưa bảo vòng: xe giảm tốc hết cờ "đang tới" nhưng
  vẫn đang đi ngang (IMG_7357 67,6 -> 68,6 s).
- Cùng vật vừa được nói < 3 s mà giờ có lệnh: chỉ nói lệnh (`Speaker.ngan`, mở rộng từ "đang tới" sang câu thông tin;
  IMG_7355 137 s "Xe máy, phía trước" rồi "Dừng lại. Xe máy").
- Vật đứng yên giữa lối (lệch trục <= 0,2 m) trong 5 m, không phân loại được: là vật chắn khi MẤT đo vạch hoặc khi đo
  được người đang ĐỨNG TRÊN vạch (khiem 131 s: "Ô tô, phía trước" rồi 1 s sau mới bảo vòng). Người đo được ở ngoài
  vạch (nhìn chéo, khiem 57,9 s) -> vẫn chỉ báo thông tin.
