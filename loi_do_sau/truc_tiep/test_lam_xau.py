import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from loi_do_sau.gia_lap_kinh import LamXau

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lam_xau_giu_kich_thuoc_va_doi_anh():
    img = np.random.default_rng(1).integers(0, 255, (360, 640, 3), np.uint8)
    lx = LamXau()
    a = lx(img)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.abs(a.astype(int) - img.astype(int)).mean() > 2


def test_chay_lai_cho_anh_y_het(tmp_path):
    # cham_diem.py so số giữa các bản trên video _kinh: cùng video phải ra đúng cùng ảnh, kể cả nhiễu và khung rơi
    # ngẫu nhiên. --roi cao để chắc chắn có khung rơi trong 3 giây.
    src = os.path.join(GOC, 'loi_do_sau/video/vid_b.mp4')
    if not os.path.exists(src):
        pytest.skip('chưa có video mẫu vid_b.mp4 (video không đưa lên git)')
    clip = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-t', '3', '-i', src, '-an', str(clip)], check=True)
    lenh = [sys.executable, os.path.join(GOC, 'loi_do_sau/gia_lap_kinh.py'), str(clip), '--roi', '0.3', '--out']
    for ten in ('lan1.mp4', 'lan2.mp4'):
        subprocess.run(lenh + [str(tmp_path / ten)], check=True, capture_output=True)
    a, b = cv2.VideoCapture(str(tmp_path / 'lan1.mp4')), cv2.VideoCapture(str(tmp_path / 'lan2.mp4'))
    n = 0
    while True:
        oa, fa = a.read()
        ob, fb = b.read()
        assert oa == ob
        if not oa:
            break
        assert np.array_equal(fa, fb)
        n += 1
    assert n > 20


def _do_net(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def test_nhoe_theo_toc_do_khong_theo_khoang_cach_khung():
    # đo thật 25/10: qua 4G hai khung cách nhau ~0,5 s -> cùng tốc độ quay đầu nhưng dịch gấp 5 lần; nhòe thật chỉ
    # phụ thuộc tốc độ x thời gian phơi sáng. Dịch 25 px sau 0,5 s (50 px/s) phải nhòe như dịch 5 px sau 0,1 s.
    rng = np.random.default_rng(2)
    nen = cv2.resize(rng.integers(0, 255, (90, 200, 3), np.uint8), (800, 360), interpolation=cv2.INTER_NEAREST)
    a, b_nhanh, b_cham = nen[:, :640], nen[:, 5:645], nen[:, 25:665]
    lx1, lx5 = LamXau(nhieu=0, seed=0), LamXau(nhieu=0, seed=0)
    lx1(a); lx5(a)
    net_01 = _do_net(lx1(b_nhanh, dt=0.1))
    net_05 = _do_net(lx5(b_cham, dt=0.5))
    lx_cu = LamXau(nhieu=0, seed=0)
    lx_cu(a)
    net_05_cu = _do_net(lx_cu(b_cham))   # không truyền dt: coi như 0,1 s (giữ kết quả cũ) -> nhòe hơn
    assert abs(net_05 - net_01) / net_01 < 0.25
    assert net_05_cu < 0.7 * net_05
