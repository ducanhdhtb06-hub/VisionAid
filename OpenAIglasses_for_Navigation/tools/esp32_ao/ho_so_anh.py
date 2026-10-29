"""Hồ sơ ảnh ESP32: biến khung video nguồn (iPhone) thành JPEG 640x480 "như camera OV2640/OV3660 trên kính".

Thứ tự (spec 7.2): ống kính (cắt theo góc nhìn ngang + méo thùng k1) -> phơi sáng (cố định như firmware gốc
aec_value 40 + tự tăng độ nhạy, hoặc tự động có trần) -> nhòe theo chuyển động × thời gian phơi sáng ->
màn trập cuộn (hàng dưới chụp muộn hơn t_doc_anh) -> cảm biến (dải sáng, tương phản, bão hòa, nhiễu photon + nhiễu
đọc nhân độ nhạy) -> JPEG với bảng lượng tử + lấy mẫu màu chép từ ảnh mẫu bo thật.
Hồ sơ: tools/esp32_ao/ho_so/<tên>.json; mỗi khóa có "<khóa>_nguon" ghi đo từ đâu hay "ước lượng".
"""
import io
import json
import math
import os

import cv2
import numpy as np
from PIL import Image

THU_MUC_HO_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ho_so")
LAY_MAU = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
MUC_TIEU = 0.45          # độ sáng trung bình phơi sáng tự động nhắm tới


def doc_ho_so(ten: str) -> dict:
    with open(os.path.join(THU_MUC_HO_SO, f"{ten}.json"), encoding="utf-8") as f:
        return json.load(f)


class HoSoAnh:
    def __init__(self, cfg: dict, seed: int = 0):
        self.c = dict(cfg)
        self.rng = np.random.default_rng(seed)
        self._truoc = None          # (t, ảnh xám nhỏ) để đo chuyển động
        self._ban_do_meo = None

    # ---------- ống kính ----------
    def _ong_kinh(self, bgr):
        h, w = bgr.shape[:2]
        if h > w:
            raise ValueError("video dọc — kính quay ngang, hãy dùng video quay ngang")
        hfov = self.c.get("hfov_ong_kinh")
        if hfov is None:
            wc = min(w, h * 4 / 3)
        else:
            nguon = self.c.get("hfov_nguon", 69.0)
            if hfov > nguon:
                raise ValueError(f"góc nhìn ống kính {hfov}° lớn hơn góc nhìn video nguồn {nguon}°")
            wc = w * math.tan(math.radians(hfov / 2)) / math.tan(math.radians(nguon / 2))
        hc = wc * 3 / 4
        x0 = int(round((w - wc) / 2))
        wc_i = int(round(wc))
        if hc <= h:
            y0 = int(round((h - hc) / 2))
            img = cv2.resize(bgr[y0:y0 + int(round(hc)), x0:x0 + wc_i], (640, 480), interpolation=cv2.INTER_AREA)
        else:                       # nguồn 16:9 thiếu chiều dọc: giữ đúng tỉ lệ ngang, đệm trên/dưới
            cao = int(round(h * 640 / wc))
            img = cv2.resize(bgr[:, x0:x0 + wc_i], (640, cao), interpolation=cv2.INTER_AREA)
            tren = (480 - cao) // 2
            img = cv2.copyMakeBorder(img, tren, 480 - cao - tren, 0, 0, cv2.BORDER_REPLICATE)
        k1 = self.c.get("k1", 0.0) or 0.0
        if k1:
            if self._ban_do_meo is None:
                yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
                xn, yn = (xx - 320) / 320, (yy - 240) / 320
                r2 = xn * xn + yn * yn
                self._ban_do_meo = (xn * (1 + k1 * r2) * 320 + 320, yn * (1 + k1 * r2) * 320 + 240)
            img = cv2.remap(img, *self._ban_do_meo, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return img

    # ---------- phơi sáng ----------
    def _phoi_sang(self, L):
        """Trả (thời gian phơi sáng s, độ nhạy G, tỉ lệ tín hiệu so với phơi sáng chuẩn)."""
        kieu = self.c.get("phoi_sang", "co_dinh")
        t_cd = self.c.get("t_phoi_sang_ms", 4.0) / 1000.0
        g_max = self.c.get("do_nhay_toi_da", 2.0)
        k = MUC_TIEU / max(float(L.mean()), 1e-3)
        if kieu == "khong":
            return t_cd, 1.0, 1.0
        if kieu == "co_dinh":
            return t_cd, float(np.clip(k, 1.0, g_max)), 1.0
        t_max = self.c.get("t_phoi_sang_toi_da_ms", 30.0) / 1000.0
        t_e = float(np.clip(t_cd * k, t_cd * 0.1, t_max))
        return t_e, float(np.clip(k / (t_e / t_cd), 1.0, g_max)), t_e / t_cd

    # ---------- chuyển động ----------
    def _van_toc(self, img, t):
        xam = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        v = (0.0, 0.0)
        if self._truoc is not None and t > self._truoc[0]:
            (dx, dy), _ = cv2.phaseCorrelate(self._truoc[1], xam)
            dt = t - self._truoc[0]
            v = (dx / dt, dy / dt)
        self._truoc = (t, xam)
        return v

    @staticmethod
    def _nhoe(img, vx, vy, t_e):
        dai = math.hypot(vx, vy) * t_e
        k = int(min(41, round(dai)))
        if k < 2:
            return img
        ker = np.zeros((k, k), np.float32)
        c, goc = (k - 1) / 2, math.atan2(vy, vx)
        cv2.line(ker, (int(round(c - math.cos(goc) * c)), int(round(c - math.sin(goc) * c))),
                 (int(round(c + math.cos(goc) * c)), int(round(c + math.sin(goc) * c))), 1.0, 1)
        return cv2.filter2D(img, -1, ker / max(1.0, ker.sum()))

    def _man_trap_cuon(self, img, vx):
        t_doc = self.c.get("t_doc_anh_ms", 0.0) / 1000.0
        if not t_doc or abs(vx) * t_doc < 0.5:
            return img
        yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
        lech = vx * (yy / 480.0 - 0.5) * t_doc            # hàng dưới chụp muộn hơn: vật đã đi thêm
        return cv2.remap(img, xx - lech, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---------- cảm biến + nén ----------
    def _cam_bien(self, img_f, G, ti_le):
        c = self.c
        S_truoc = img_f * ti_le                          # tín hiệu trước khuếch đại (0–1)
        S = S_truoc * G
        toi, sang = c.get("diem_cat_toi", 0.0), c.get("diem_cat_sang", 1.0)
        S = np.clip((S - toi) / max(1e-6, sang - toi), 0.0, 1.0)
        tp, bh = c.get("tuong_phan", 1.0), c.get("bao_hoa", 1.0)
        if tp != 1.0:
            S = np.clip(0.5 + (S - 0.5) * tp, 0.0, 1.0)
        if bh != 1.0:
            xam = S.mean(axis=2, keepdims=True)
            S = np.clip(xam + (S - xam) * bh, 0.0, 1.0)
        dn = S * 255.0
        np_, nd = c.get("nhieu_photon", 0.0), c.get("nhieu_doc", 0.0)
        if np_ or nd:
            sigma = np.sqrt((np_ * np.sqrt(np.clip(S_truoc, 0, None) * 255.0) * G) ** 2 + (nd * G) ** 2)
            dn = dn + self.rng.standard_normal(dn.shape).astype(np.float32) * sigma
        return np.clip(dn, 0, 255).astype(np.uint8)

    def _nen(self, img):
        bang = self.c.get("bang_luong_tu")
        if self.c.get("jpeg", 17) == 12 and bang is not None:
            bang = self.c.get("bang_luong_tu_q12") or [[max(1, round(q * 12 / 17)) for q in t] for t in bang]
        if bang is None:
            return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.c.get("chat_luong", 80))])[1].tobytes()
        buf = io.BytesIO()
        Image.fromarray(img[:, :, ::-1]).save(buf, "JPEG", qtables=bang,
                                             subsampling=LAY_MAU[self.c.get("lay_mau", "4:2:2")])
        return buf.getvalue()

    def __call__(self, bgr: np.ndarray, t: float) -> bytes:
        img = self._ong_kinh(bgr)
        vx, vy = self._van_toc(img, t)
        L = img.astype(np.float32) / 255.0
        t_e, G, ti_le = self._phoi_sang(L)
        img = self._nhoe(img, vx, vy, t_e)
        img = self._man_trap_cuon(img, vx)
        sigma = self.c.get("do_net_sigma", 0.0)
        if sigma:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        if self.c.get("phoi_sang", "co_dinh") != "khong":
            img = self._cam_bien(img.astype(np.float32) / 255.0, G, ti_le)
        return self._nen(img)
