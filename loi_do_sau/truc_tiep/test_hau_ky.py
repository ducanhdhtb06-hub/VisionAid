import json
import os
import subprocess

import cv2
import numpy as np
import pytest

from loi_do_sau.truc_tiep.hau_ky import doi_chi_ghi, ghep_tieng, thong_ke
from loi_do_sau.truc_tiep.may_chu import pcm_sang_wav

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T0 = 1_000_000.0


def tao_phien(d, so_khung=20, bo_qua=(), phan_hoi=(), cach_ms=100):
    os.makedirs(d / 'cau', exist_ok=True)
    wr = cv2.VideoWriter(str(d / 'khung.mp4'), cv2.VideoWriter.fourcc(*'mp4v'), 10, (640, 360))
    for i in range(so_khung):
        wr.write(np.full((360, 640, 3), i * 10 % 255, np.uint8))
    wr.release()
    khung = [dict(id=i, t_chup=T0 + (i - 1) * cach_ms, t_nhan=0, ms_xu_ly=70, cau=None, bo_qua=i in bo_qua)
             for i in range(1, so_khung + 1)]
    json.dump(dict(bat_dau='2026-10-25T00:00:00', khung=khung, phan_hoi=list(phan_hoi)), open(d / 'log.json', 'w'))


def test_thong_ke(tmp_path):
    tao_phien(tmp_path, so_khung=10, bo_qua=(4, 7), phan_hoi=[
        dict(phan_hoi='da_phat', id=3, tre_ms=400), dict(phan_hoi='da_phat', id=5, tre_ms=600),
        dict(phan_hoi='bo', id=8, tre_ms=1800), dict(phan_hoi='bi_cat', id=9, tre_ms=300)])
    tk = thong_ke(str(tmp_path))
    assert tk['so_khung'] == 10 and tk['ty_le_bo_qua'] == 0.2
    assert (tk['da_phat'], tk['bo'], tk['bi_cat'], tk['thay']) == (2, 1, 1, 0)
    assert tk['tre_trung_vi'] == 500 and tk['tre_p90'] == 600


def test_ghep_tieng_dung_thoi_diem_nghe(tmp_path):
    # câu của khung 11 (chụp ở giây 1,0) phát sau 500 ms -> tiếng phải nằm ở ~1,5 s, trước đó im
    tao_phien(tmp_path, so_khung=30, phan_hoi=[dict(phan_hoi='da_phat', id=11, tre_ms=500),
                                               dict(phan_hoi='bo', id=21, tre_ms=1900)])
    tone = (np.sin(np.arange(8000) / 16000 * 2 * np.pi * 440) * 20000).astype(np.int16)
    for i in (11, 21):
        open(tmp_path / 'cau' / f'{i}.wav', 'wb').write(pcm_sang_wav(tone))
    out = ghep_tieng(str(tmp_path))
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', out, '-f', 's16le', '-ac', '1', '-ar', '16000', '-'],
                         capture_output=True, check=True).stdout
    a = np.abs(np.frombuffer(raw, np.int16).astype(float))
    muc = lambda t0, t1: a[int(t0 * 16000):int(t1 * 16000)].mean()  # noqa: E731
    assert muc(0.2, 1.4) < 200            # im trước lúc phát
    assert muc(1.55, 1.95) > 5000         # câu đã phát
    assert muc(2.2, 2.9) < 200            # câu bị bỏ (khung 21, ~3,0 s) không có trong tiếng người dùng nghe


def test_doi_chi_ghi(tmp_path):
    if not os.path.exists(os.path.join(GOC, 'loi_do_sau/video/vid_b.mp4')):
        pytest.skip('chưa có video mẫu vid_b.mp4 (video không đưa lên git)')
    src = tmp_path / 'kinh.mov'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-t', '2', '-i', os.path.join(GOC, 'loi_do_sau/video/vid_b.mp4'),
                    '-an', str(src)], check=True)
    out = doi_chi_ghi(str(src))
    cap = cv2.VideoCapture(out)
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (640, 360)
    assert round(cap.get(cv2.CAP_PROP_FPS)) == 10
    assert 18 <= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 21


def test_ghep_tieng_khi_4g_cham(tmp_path):
    # 4G chậm: chỉ ~4 khung/giây (cách 250 ms). Video phải dài đúng thời gian thật và tiếng đúng lúc nghe.
    tao_phien(tmp_path, so_khung=24, cach_ms=250, phan_hoi=[dict(phan_hoi='da_phat', id=17, tre_ms=500)])
    tone = (np.sin(np.arange(8000) / 16000 * 2 * np.pi * 440) * 20000).astype(np.int16)
    open(tmp_path / 'cau' / '17.wav', 'wb').write(pcm_sang_wav(tone))
    out = ghep_tieng(str(tmp_path))
    cap = cv2.VideoCapture(out)
    dai = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    assert dai == pytest.approx(6.0, abs=0.3)            # 24 khung x 250 ms
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', out, '-f', 's16le', '-ac', '1', '-ar', '16000', '-'],
                         capture_output=True, check=True).stdout
    a = np.abs(np.frombuffer(raw, np.int16).astype(float))
    muc = lambda t0, t1: a[int(t0 * 16000):int(t1 * 16000)].mean()  # noqa: E731
    assert muc(0.5, 4.3) < 200 and muc(4.55, 4.95) > 5000  # khung 17 chụp ở 4,0 s + 500 ms
    # khung đang hiện lúc 4,6 s phải là khung 19 (chụp 4,5 s, màu 18*10), không phải khung thứ 46 (không có)
    cap.set(cv2.CAP_PROP_POS_MSEC, 4600)
    ok, fr = cap.read()
    assert ok and abs(int(fr[180, 320, 0]) - 180) < 12


def test_doc_phien_bi_tat_ngang(tmp_path):
    # máy tính tắt đột ngột: không có log.json, chỉ có log.jsonl ghi dần -> hậu kỳ vẫn đọc được
    from loi_do_sau.truc_tiep.may_chu import Phien
    p = Phien(str(tmp_path))
    img = np.zeros((360, 640, 3), np.uint8)
    p.nhan(1, T0, img)
    p.xong(1, 70)
    p.nhan(2, T0 + 100, img)
    p.xong(2, 72, 'Xe máy, bên trái, 3 mét', pcm_sang_wav(np.zeros(800, np.int16)))
    p.phan_hoi(dict(phan_hoi='da_phat', id=2, tre_ms=150))
    p.writer.release()                       # mô phỏng: video đã có khung, nhưng dong() không kịp chạy
    assert not os.path.exists(tmp_path / 'log.json')
    tk = thong_ke(str(tmp_path))
    assert tk['so_khung'] == 2 and tk['da_phat'] == 1 and tk['tre_trung_vi'] == 150
