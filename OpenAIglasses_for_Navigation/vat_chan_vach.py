# vat_chan_vach.py
# -*- coding: utf-8 -*-
"""
Vật chắn trên vạch — giai đoạn 1 (spec docs/superpowers/specs/2026-10-27-vong-vat-can-tren-vach-design.md, mục 5).

Chỉ chạy khi người đang ở trên vạch. Các bước:
  BÌNH THƯỜNG --(vật chắn đứng yên <= 3 m)--> ĐÃ BÁO "Xe máy chắn vạch bên phải. Đi sang trái."
    bên đi vòng chọn theo SỐ ĐÔNG các khung từ 5 m tới 3 m (một khung nhiễu không quyết định; lỗi thật IMG_7358 9 s).
    không bên nào chắc trống: KHÔNG bảo dừng (người dùng 28/10: "phải bảo họ vòng ra") ->
      biết một bên là lòng đường: "Xe máy chắn giữa vạch. Đi sang trái, dò gậy." (phía xa lòng đường)
      không biết: "Xe máy chắn giữa vạch. Dùng gậy dò đường vòng."   — không bao giờ hướng sang lòng đường.
  ĐÃ BÁO: kính im (camera hẹp không thấy lúc người đi ngang vật -> gậy dẫn đoạn này, kính không ra lệnh quay về).
    rời vạch rồi đứng lại trên vạch, vật hết chắn >= 1,5 s -> "Đã về vạch."
    không rời vạch, vật hết chắn >= 1,5 s (vật đã đi / thấy nhầm) -> xong, im
    về vạch khi vật vẫn chắn <= 3 m trước mặt -> báo lại câu cũ
    rời vạch 15 s chưa về -> 1 câu thông tin: "Vạch ở bên phải/trái." hoặc "Chưa thấy lại vạch, hãy dùng gậy."
Logic thuần (không cần model); workflow_blindpath đưa vào vật chắn gần nhất và trạng thái vạch mỗi khung.
"""
import copy

D_NOI = 3.0        # nói khi vật chắn còn <= 3 m (ước lượng; kính đoán gần hơn thật ~10%): ~4–5 s đi tới vật
D_THEO = 5.0       # bắt đầu chọn và giữ bên từ 5 m (lúc camera còn thấy hai bên vật)
T_HET = 1.5        # không thấy vật chắn >= 1,5 s = hết chắn
T_NHAC = 15.0      # rời vạch 15 s chưa về -> nhắc thông tin một lần
T_KET = 20.0       # ĐÃ BÁO mà người không rời vạch 20 s -> xong
T_ROI_MIN = 2.0    # ra ngoài vạch liên tục >= 2 s mới là "đã rời" (đi vòng 1 xe mất vài giây; ngắn hơn = đo chập chờn)
T_LAP = 5.0        # vật mới cần đúng câu vừa nói trong 5 s (xe đỗ liền nhau) -> không nói lại
VAT_MOI = 1.0      # vật chắn gần nhất xa hơn lần trước > 1 m = vật KHÁC (cùng vật chỉ gần lại khi người bước tới)
T_ON_DINH = 1.0    # thấy vật chắn liên tục >= 1 s mới giao cho bộ điều khiển (vết lạ / người đi ngang chỉ vài khung)

CAU_VE_VACH = "Đã về vạch."
CAU_CHUA_THAY = "Chưa thấy lại vạch, hãy dùng gậy."


NGUONG_DUONG = 0.3   # dải sẽ đi vòng có >= 30% là lòng đường -> không chọn (như plan_avoid bản 17)
TI_LE_TRONG = 0.5    # bên được thấy trống trong >= nửa số khung từ 5 m tới 3 m mới chọn; lòng đường: >= nửa số khung


def xep_ben(free, ti_le_duong):
    """Thứ tự bên vòng: free = {'dich_trai'|'dich_phai': (đi được, mét phải dịch)} (từ plan_avoid),
    ti_le_duong = {..: tỉ lệ lòng đường trên dải sẽ đi} (thiếu = không biết). Bỏ bên không đi được / phần lớn lòng
    đường; ưu tiên bên ít lòng đường hơn, rồi bên dịch ít hơn. Trả ["trai"|"phai", ...]."""
    ung = []
    for act, (ok, dich) in free.items():
        duong = ti_le_duong.get(act)
        if ok and (duong is None or duong < NGUONG_DUONG):
            ung.append((round(duong or 0.0, 2), dich, "trai" if act == "dich_trai" else "phai"))
    return [b for _, _, b in sorted(ung)]


# tên đồ vật thường đỗ/đặt trên vỉa hè mà model gọi khá đúng; tên khác (xe tải, xe buýt, con vật, ...) -> "Vật cản"
# (chạy 7 video 2026-10-28: tấm quây -> "Xe tải", túi nilon -> "Mèo")
TEN_TIN = {'parked motorcycle', 'motorcycle', 'scooter', 'bicycle', 'car', 'stroller', 'plastic stool', 'stool',
           'chair', 'table', 'bench', 'pole', 'bollard', 'potted plant', 'basket', 'bucket', 'box', 'traffic cone',
           'trash can', 'signboard', 'tree trunk', 'fire hydrant'}


def ten_noi(ten_en, ten_vi):
    """Tên tiếng Việt để nói, hoặc None (câu dùng "Vật cản") nếu tên không đáng tin."""
    return ten_vi if ten_en in TEN_TIN else None


class OnDinh:
    """Vật chắn gần nhất đã thấy liên tục >= T_ON_DINH giây chưa (mất >= T_HET giây hoặc vật khác thì đếm lại)."""
    def __init__(self):
        self.t_dau = None
        self.t_thay = -1e9
        self.d_cuoi = None

    def cap_nhat(self, t, vat):
        if vat is None:
            return self.t_dau is not None and t - self.t_thay < T_HET and t - self.t_dau >= T_ON_DINH
        if self.t_dau is None or t - self.t_thay >= T_HET or vat["dist"] > self.d_cuoi + VAT_MOI:
            self.t_dau = t
        self.t_thay, self.d_cuoi = t, vat["dist"]
        return t - self.t_dau >= T_ON_DINH - 1e-6


class NhoVatChan:
    """Nhớ vật chắn vừa thấy đè lên vạch. Tới gần, chính vật che mép vạch nên phan_loai_vat ra "khong_chac" — vật không
    vì thế mà hết chắn (người dùng 29/10, IMG_7354 299–302 s / 341 s: bộ điều khiển tưởng xe đã đi hoặc chưa kịp bảo
    vòng -> module vật cản nói "Xe máy, phía trước" về chính chiếc xe đó).
    Vật "khong_chac" nối tiếp vật chắn vừa thấy (< T_HET s, không xa hơn quá VAT_MOI) -> vẫn "chan"; "canh" giữ nguyên."""
    def __init__(self):
        self.t, self.d, self.vi_tri = -1e9, None, None

    def tiep(self, t, dist):
        """Vật ở khoảng cách dist nối tiếp vật chắn vừa thấy (đi ngang xe đỗ, YOLOE mất dấu lập vết mới: vẫn xe đó)."""
        return self.d is not None and t - self.t < T_HET and dist <= self.d + VAT_MOI

    def loai(self, t, loai, dist):
        return "chan" if loai == "khong_chac" and self.tiep(t, dist) else loai

    def cap_nhat(self, t, dist, vi_tri):
        """Ghi vật chắn đã chọn ở khung này; trả vị trí để nói (lần đo được gần nhất của cùng vật nếu giờ không đo được)."""
        if self.d is None or t - self.t >= T_HET:
            self.vi_tri = None
        self.t, self.d = t, dist
        self.vi_tri = vi_tri or self.vi_tri
        return self.vi_tri


def cau_bao(ten, ben, vi_tri=None, kieu="ben"):
    """Nói rõ vật ở đâu rồi mới tới hướng đi (người dùng 28/10: "vòng bên trái" nghe thành "xe ở bên trái"):
    kieu "ben": "Xe máy chắn vạch bên phải. Đi sang trái." / "do_gay": "… Đi sang trái, dò gậy." (không chắc bên đó
    trống, chỉ biết bên kia là lòng đường) / "do_vong": "… Dùng gậy dò đường vòng." (không biết bên nào)."""
    ten = ten or "vật cản"
    cho = {"phai": "chắn vạch bên phải", "trai": "chắn vạch bên trái", "giua": "chắn giữa vạch"}.get(vi_tri, "chắn vạch")
    dau = f"{ten[0].upper() + ten[1:]} {cho}."
    if kieu == "do_vong":
        return f"{dau} Dùng gậy dò đường vòng."
    huong = f"Đi sang {'trái' if ben == 'trai' else 'phải'}"
    return f"{dau} {huong}, dò gậy." if kieu == "do_gay" else f"{dau} {huong}."


class BoVatChan:
    def __init__(self):
        self.trang_thai = "binh_thuong"
        self.ben_giu = None        # bên đang nghiêng về khi còn xa (giữ nếu vẫn đi được)
        self.ben_chot = None       # bên đã nói (None: "dò gậy đường vòng")
        self.kieu = "ben"          # kiểu câu đã nói (cau_bao)
        self.ten = self.vi_tri = None
        self.t_bao = self.t_roi = None
        self.t_thay = -1e9         # lần cuối thấy vật chắn
        self.d_cuoi = None         # khoảng cách lần cuối thấy vật chắn (nhận ra vật mới)
        self.cau_cuoi, self.t_cau_cuoi = None, -1e9
        self.t_tren = -1e9         # lần cuối đứng trên vạch
        self._phieu_moi()
        self._truoc = None

    @property
    def dang_xu_ly(self):
        return self.trang_thai == "da_bao"

    def tren_gan(self, t, tren_vach):
        """Trên vạch, hoặc mới ra ngoài < T_ROI_MIN (lần đo chập chờn / xe che vạch làm mất lần đo: chưa là rời vạch;
        lỗi thật IMG_7354 312 s: mất đo vạch đúng lúc xe tới 3 m -> không bảo vòng)."""
        return tren_vach or t - self.t_tren < T_ROI_MIN

    def chua_noi(self):
        """Câu vừa trả không được phát (thua câu khác): quay về trạng thái trước để khung sau nói lại."""
        if self._truoc is not None:
            self.__dict__.update(self._truoc)
            self._truoc = None

    def _phieu_moi(self):
        # mỗi bên: [số khung trống, số khung] và [số khung là lòng đường, số khung biết lòng đường] từ 5 m tới 3 m
        self.phieu = {"trai": [0, 0], "phai": [0, 0]}
        self.phieu_duong = {"trai": [0, 0], "phai": [0, 0]}

    def _xong(self):
        self.trang_thai, self.ben_giu, self.ben_chot, self.t_bao, self.t_roi = "binh_thuong", None, None, None, None
        self._phieu_moi()

    def _noi(self, t, cau):
        """Không nói lại đúng câu vừa nói trong T_LAP giây (xe đỗ liền nhau: cùng bên vòng)."""
        lap = cau == self.cau_cuoi and t - self.t_cau_cuoi < T_LAP
        self.cau_cuoi, self.t_cau_cuoi = cau, t
        return "" if lap else cau

    def _bo_phieu(self, vat):
        ben = vat.get("ben") or []
        duong = vat.get("duong") or {}
        for b in ("trai", "phai"):
            self.phieu[b][0] += b in ben
            self.phieu[b][1] += 1
            if duong.get(b) is not None:
                self.phieu_duong[b][0] += duong[b] >= NGUONG_DUONG
                self.phieu_duong[b][1] += 1
        trong = {b: n / tong for b, (n, tong) in self.phieu.items() if tong and n / tong >= TI_LE_TRONG}
        if self.ben_giu not in trong:                 # giữ bên đã nghiêng về từ xa nếu vẫn đủ phiếu
            thu_tu = {b: i for i, b in enumerate(ben)}
            self.ben_giu = min(trong, key=lambda b: (-trong[b], thu_tu.get(b, 2))) if trong else None

    def _ben_xa_duong(self):
        """Bên đối diện bên lòng đường (đa số khung), nếu đúng MỘT bên là lòng đường; còn lại None."""
        la_duong = [b for b, (n, tong) in self.phieu_duong.items() if tong and n / tong >= TI_LE_TRONG]
        if len(la_duong) != 1:
            return None
        return "phai" if la_duong[0] == "trai" else "trai"

    def _cau(self):
        return cau_bao(self.ten, self.ben_chot, self.vi_tri, self.kieu)

    def buoc(self, t, tren_vach, vat, lech, thay_vach, theo_vach=False):
        """vat: vật chắn gần nhất {dist, ten, ben: ["trai"|"phai", ...] theo thứ tự ưu tiên, vi_tri: nửa vạch bị đè
        "trai"|"phai"|"giua"|None, duong: {"trai"|"phai": tỉ lệ lòng đường dải đi vòng | None}} hoặc None.
        lech: độ lệch vạch (dương = vạch bên phải) hoặc None; thay_vach: có lần đo vạch tin cậy gần đây.
        theo_vach: đang theo vạch mà mất lần đo (xe che vạch) - chưa có bằng chứng đã rời vạch: được bắt đầu bảo vòng."""
        self._truoc = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "_truoc"}
        tren_gan = self.tren_gan(t, tren_vach)
        if tren_vach:
            self.t_tren = t
        if vat is not None:
            if self.d_cuoi is not None and vat["dist"] > self.d_cuoi + VAT_MOI:
                # vật chắn MỚI (vật cũ đã khuất / đã qua). Đang đi vòng mà bên đã nói vẫn vòng được (dãy xe đỗ cùng
                # hàng) -> lời dặn cũ vẫn đúng, im. Khác bên -> làm lại từ đầu, chọn lại bên
                # (lỗi thật IMG_7358 6,9–9,5 s: đang xử lý vật cũ nên im khi xe máy khác tiến tới 3 m)
                if self.trang_thai == "da_bao" and self.ben_chot in (vat.get("ben") or []):
                    self.t_bao, self.ten = t, vat.get("ten") or self.ten
                    self.vi_tri = vat.get("vi_tri") or self.vi_tri
                else:
                    self._xong()
            self.t_thay, self.d_cuoi = t, vat["dist"]
        het_chan = t - self.t_thay >= T_HET
        if het_chan:
            self.d_cuoi = None
        if self.trang_thai == "binh_thuong":
            if vat is None:
                if het_chan:
                    self.ben_giu = None
                    self._phieu_moi()
                return ""
            if not (tren_gan or theo_vach):
                return ""
            if vat["dist"] <= D_THEO:
                self._bo_phieu(vat)
            if vat["dist"] > D_NOI:
                return ""
            self.t_bao, self.t_roi, self.ten, self.vi_tri = t, None, vat.get("ten"), vat.get("vi_tri")
            self.trang_thai = "da_bao"
            # không bao giờ bảo đi về đúng nửa vạch xe đang đè ("Xe máy chắn vạch bên phải. Đi sang phải" nghe như mâu
            # thuẫn; lỗi thật IMG_7354 218,8 s, 29/10)
            if self.ben_giu is not None and self.ben_giu != self.vi_tri:
                self.ben_chot, self.kieu = self.ben_giu, "ben"
            else:                                      # không bên nào chắc trống: dò gậy vòng, không bảo dừng
                self.ben_chot = self._ben_xa_duong()
                if self.ben_chot == self.vi_tri:
                    self.ben_chot = None
                self.kieu = "do_gay" if self.ben_chot else "do_vong"
            return self._noi(t, self._cau())
        # ĐÃ BÁO
        if not tren_vach:
            if self.t_roi is None:
                self.t_roi = t
            if t - self.t_roi >= T_NHAC:
                self._xong()
                if thay_vach and lech is not None:
                    return "Vạch ở bên phải." if lech > 0 else "Vạch ở bên trái."
                return CAU_CHUA_THAY
            return ""
        if self.t_roi is not None and t - self.t_roi < T_ROI_MIN:
            self.t_roi = None                            # ra ngoài chưa đủ 2 s: đo chập chờn, chưa tính là rời vạch
        if self.t_roi is not None:                       # đã rời vạch, giờ đứng lại trên vạch
            if het_chan:
                self._xong()
                return CAU_VE_VACH
            if vat is not None and vat["dist"] <= D_NOI:  # về vạch sớm, vật vẫn chắn trước mặt -> báo lại
                self.t_roi, self.ten = None, vat.get("ten") or self.ten
                self.vi_tri = vat.get("vi_tri") or self.vi_tri
                return self._cau()
            return ""
        if het_chan or t - self.t_bao >= T_KET:          # chưa rời vạch mà vật đã hết chắn / quá lâu
            self._xong()
        return ""
