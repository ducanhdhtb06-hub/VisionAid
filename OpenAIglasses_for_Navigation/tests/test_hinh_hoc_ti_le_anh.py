"""Module vật cản đo đúng vị trí vật trên mọi tỉ lệ ảnh và ống kính.
Lỗi thật 2026-11-06: ảnh kính 640x480 bị ép về 640x360 mà công thức dùng một tiêu cự cho hai trục -> lệch ngang đọc lớn
hơn thật 1,35 lần (đo trên 695 cặp vật, 3 video dev), hành lang ±0,6 m thực chất chỉ còn ±0,45 m.
Cảnh giả tính tay từ máy ảnh lỗ kim. Chỉ thay model ở ranh giới: bộ nhận diện giả tìm khối trắng trên đúng ảnh module
đưa vào, nên phần thu nhỏ ảnh vẫn chạy thật."""
import math

import cv2
import numpy as np
import pytest

import ne_vat_can_do_sau as N

K = N.K
H_CAM = 1.5


class _MayAnh:
    """Máy ảnh lỗ kim cao H_CAM, chúc xuống `chuc_do`, ảnh rong x cao, góc nhìn ngang hfov (độ)."""
    def __init__(self, rong, cao, hfov, chuc_do=0.0):
        self.w, self.h = rong, cao
        self.f = (rong / 2) / math.tan(math.radians(hfov / 2))
        self.th = math.radians(chuc_do)
        self.rng = np.random.default_rng(0)

    def chieu(self, x, z, cao_vat=0.0):
        """(cột, hàng) của điểm lệch x m, tới trước z m, cao cao_vat m so với mặt đất."""
        d = H_CAM - cao_vat
        zc = z * math.cos(self.th) + d * math.sin(self.th)
        yc = d * math.cos(self.th) - z * math.sin(self.th)
        return self.w / 2 + self.f * x / zc, self.h / 2 + self.f * yc / zc

    def y_chan_troi(self):
        return self.h / 2 - self.f * math.tan(self.th)

    def khung(self, x_trai, x_phai, z):
        """Mặt đất nhiễu xám + vật trắng cao 1 m, đứng ở khoảng cách z, chiếm từ x_trai tới x_phai."""
        anh = self.rng.integers(60, 160, (self.h, self.w, 3), dtype=np.uint8)
        cot_trai, hang_chan = self.chieu(x_trai, z)
        cot_phai, _ = self.chieu(x_phai, z)
        _, hang_dinh = self.chieu(x_trai, z, 1.0)
        cv2.rectangle(anh, (round(cot_trai), round(hang_dinh)), (round(cot_phai), round(hang_chan)), (255, 255, 255), -1)
        return anh


class _ModelGia:
    def to(self, _):
        return self

    def eval(self):
        return self

    def half(self):
        return self


class _NhanDienGia:
    """Thay D-FINE: một xe máy ở khối trắng trên ảnh được đưa vào, trả đủ các khoá như DFineDet.detect.
    co=(cao, rộng): trả toạ độ theo cỡ đó, như target_sizes của D-FINE. rong_toi_thieu, cao_toi_thieu: khối nhỏ hơn mức
    này (tính trên ảnh được đưa vào) thì không thấy, như vật nhỏ trên ảnh ít chi tiết."""
    rong_toi_thieu = cao_toi_thieu = 0

    def __init__(self, _repo):
        pass

    def detect(self, img, co=None):
        h, w = img.shape[:2]
        H, W = co or (h, w)
        ys, xs = np.nonzero(img.min(axis=2) > 200)
        if not len(ys) or xs.max() - xs.min() + 1 < self.rong_toi_thieu or ys.max() - ys.min() + 1 < self.cao_toi_thieu:
            return []
        x0, x1 = round(float(xs.min()) * W / w), round(float(xs.max()) * W / w)
        y0, y1 = round(float(ys.min()) * H / h), round(float(ys.max()) * H / h)
        mask = np.zeros((H, W), np.uint8)
        mask[y0:y1 + 1, x0:x1 + 1] = 255
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        return [dict(name='motorcycle', mask=mask, area=area, area_ratio=area / (H * W), center_x=(x0 + x1) / 2,
                     center_y=(y0 + y1) / 2, bottom_y_ratio=y1 / H)]


class _DuongGia:
    """Thay UperNet: cả khung là lòng đường nếu ảnh được đưa vào đủ chi tiết; mặt nạ theo cỡ co, như target_sizes."""
    def __call__(self, img, co=None):
        H, W = co or img.shape[:2]
        return np.full((H, W), img.shape[1] >= 640), np.zeros((H, W), bool)


class _LoaGia:
    """Test này xét hình học, không xét câu nói (Speaker thật có test riêng)."""
    events = []

    def say(self, *a, **k):
        return False

    def update(self, *a, **k):
        return None


@pytest.fixture
def tao_ne(monkeypatch):
    monkeypatch.delenv('AIGLASS_HFOV', raising=False)
    monkeypatch.setattr(K, 'H_CAM', H_CAM)
    monkeypatch.setattr(K, 'DFineDet', _NhanDienGia)
    monkeypatch.setattr('transformers.AutoModelForDepthEstimation.from_pretrained', lambda *a, **k: _ModelGia())
    monkeypatch.setattr(N, '_dang_ky_cau_noi', lambda *a: None)

    def tao(cam, hfov_khai=None, duong='0'):
        monkeypatch.setenv('AIGLASS_NE_DUONG', duong)
        if hfov_khai:
            monkeypatch.setenv('AIGLASS_HFOV', hfov_khai)
        ne = N.NeVatCanDoSau()

        def do_sau_mat_dat(img):
            """Mặt đất phẳng: độ sâu nghịch đảo tỉ lệ với số hàng dưới chân trời, trên đúng ảnh được đưa vào."""
            H, W = img.shape[:2]
            hang = np.arange(H, dtype=np.float32) - cam.y_chan_troi() * H / cam.h
            return np.tile(np.maximum(hang, 0.0)[:, None], (1, W)) + 1e-3

        ne._do_sau = do_sau_mat_dat
        ne.speaker = _LoaGia()
        return ne
    return tao


@pytest.mark.parametrize('rong, cao, hfov, khai', [
    (640, 360, 70.0, None),      # video điện thoại ở cỡ xử lý
    (1920, 1080, 70.0, None),    # video điện thoại gốc
    (640, 360, 60.0, '60'),      # 16:9, ống kính khác, có khai góc nhìn
    (640, 480, 55.4, None),      # 4:3 chưa khai: coi là phần cắt 4:3 của camera điện thoại
    (640, 480, 56.0, '56'),      # kính, ống kính ước lượng
    (640, 480, 66.0, '66'),      # kính, ống kính rộng nhất có thể của OV2640
])
@pytest.mark.parametrize('chuc_do', [0.0, 9.0])
def test_do_dung_vi_tri_vat(tao_ne, rong, cao, hfov, khai, chuc_do):
    cam = _MayAnh(rong, cao, hfov, chuc_do)
    ne = tao_ne(cam, khai)
    ne.step(cam.khung(0.2, 0.8, 5.0), t=0.0)
    o = K.yoloe_obstacles(ne.yo, ne.geo, ne.F, ne.H, ne.W)[0]
    assert (o['xl'] + o['xr']) / 2 == pytest.approx(0.5, rel=0.05)
    assert o['dist'] == pytest.approx(5.0, rel=0.05)


@pytest.mark.parametrize('mep_gan, phai_bao', [(0.5, True), (0.8, False)])
def test_xe_do_chi_duoc_bao_khi_lan_vao_loi_di(tao_ne, mep_gan, phai_bao):
    # hành lang ±0,6 m: xe đỗ cách giữa lối 0,5 m chắn vai người đi; cách 0,8 m thì không
    cam = _MayAnh(640, 480, 56.0, 9.0)
    ne = tao_ne(cam, '56')
    for i in range(4):
        ne.step(cam.khung(mep_gan, mep_gan + 0.6, 4.0), t=0.1 * i)
    assert bool(ne.cand) is phai_bao


def test_vat_nho_khong_mat_vi_thu_anh_truoc_khi_nhan_dien(tao_ne, monkeypatch):
    # đo 3 video dev: thu về rộng 480 trước D-FINE mất 9,6% vật (mạng tự phóng về 640x640, chi tiết đã mất không lấy lại
    # được). Hình học tính ở 480 nhưng mạng phải nhận ảnh gốc đúng tỉ lệ. Vật ~14x74 điểm ở ảnh gốc; thu về 480: ~10x55;
    # ép 640x360 như cũ: ~14x55 (đo: khớp ảnh đủ phân giải 74% so với 85% của ảnh gốc).
    monkeypatch.setattr(_NhanDienGia, 'rong_toi_thieu', 12)
    monkeypatch.setattr(_NhanDienGia, 'cao_toi_thieu', 65)
    cam = _MayAnh(640, 480, 56.0, 9.0)
    ne = tao_ne(cam, '56')
    ne.step(cam.khung(0.1, 0.29, 8.0), t=0.0)
    ds = K.yoloe_obstacles(ne.yo, ne.geo, ne.F, ne.H, ne.W)
    assert len(ds) == 1
    assert (ds[0]['xl'] + ds[0]['xr']) / 2 == pytest.approx(0.195, abs=0.03)
    assert ds[0]['dist'] == pytest.approx(8.0, rel=0.05)


def test_mat_na_long_duong_khop_co_hinh_hoc(tao_ne, monkeypatch):
    # workflow_blindpath bắt mọi lỗi của module rồi lặng lẽ về luật cũ: mặt nạ lệch cỡ hình học sẽ không ai thấy
    monkeypatch.setattr(K, 'DuongVaHe', _DuongGia)
    cam = _MayAnh(640, 480, 56.0, 9.0)
    ne = tao_ne(cam, '56', duong='1')
    ne.step(cam.khung(0.2, 0.8, 5.0), t=0.0)
    assert ne.dang_o_duong is True


def test_toc_do_ngang_dung_tren_anh_kinh(tao_ne):
    # dung_yen() và plan_avoid() dùng ngưỡng 0,8 m/s: tốc độ ngang sai làm xe đỗ / xe cắt ngang bị phân loại nhầm
    cam = _MayAnh(640, 480, 56.0)
    ne = tao_ne(cam, '56')
    for i in range(10):
        x = -0.5 + 0.1 * i                       # 1 m/s sang phải, 0,1 s mỗi khung
        ne.step(cam.khung(x - 0.3, x + 0.3, 4.0), t=0.1 * i)
    assert ne.cand[0]['vx'] == pytest.approx(1.0, rel=0.1)
