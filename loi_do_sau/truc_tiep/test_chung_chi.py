"""Chứng chỉ https cho iPhone nối thẳng laptop: Safari chỉ mở camera trên https, wss cần chứng chỉ được tin hẳn.
Chạy (đứng ở glassai/): .venv/bin/python -m pytest loi_do_sau/truc_tiep/test_chung_chi.py -q"""
import datetime
import os
import socket
import ssl
import stat
import threading
import urllib.error
import urllib.request

import pytest
from cryptography import x509

from loi_do_sau.truc_tiep.chung_chi import ca_goc, cap_chung_chi_may, chon_ip, loc_noi_bo, phat_chung_chi


def _bat_tay(ca, crt, key, ip_tren_chung_chi):
    """Bắt tay TLS thật qua 127.0.0.1; client tin đúng CA và đòi chứng chỉ khớp ip_tren_chung_chi."""
    sv = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sv.load_cert_chain(crt, key)
    nghe = socket.create_server(('127.0.0.1', 0))
    cong = nghe.getsockname()[1]

    def phuc_vu():
        c, _ = nghe.accept()
        try:
            with sv.wrap_socket(c, server_side=True) as s:
                s.recv(1)
        except (ssl.SSLError, OSError):
            pass

    th = threading.Thread(target=phuc_vu, daemon=True)
    th.start()
    kh = ssl.create_default_context(cafile=ca)
    try:
        with socket.create_connection(('127.0.0.1', cong), timeout=5) as c:
            with kh.wrap_socket(c, server_hostname=ip_tren_chung_chi) as s:
                s.send(b'x')
    finally:
        th.join(5)
        nghe.close()


def test_iphone_tin_chung_chi_may_qua_ca_rieng(tmp_path):
    ca, _ = ca_goc(str(tmp_path))
    crt, key = cap_chung_chi_may(str(tmp_path), ['172.20.10.2', '127.0.0.1'])
    _bat_tay(ca, crt, key, '172.20.10.2')


def test_ip_khong_co_trong_chung_chi_bi_tu_choi(tmp_path):
    ca, _ = ca_goc(str(tmp_path))
    crt, key = cap_chung_chi_may(str(tmp_path), ['172.20.10.2'])
    with pytest.raises(ssl.SSLCertVerificationError):
        _bat_tay(ca, crt, key, '172.20.10.3')


def test_lo_khoa_ca_van_khong_gia_mao_duoc_dia_chi_cong_khai(tmp_path):
    # CA này nằm trên iPhone riêng của người dùng: khóa bị lộ cũng chỉ giả được IP nội bộ, không giả được trang thật
    ca, _ = ca_goc(str(tmp_path))
    crt, key = cap_chung_chi_may(str(tmp_path), ['8.8.8.8'])
    with pytest.raises(ssl.SSLCertVerificationError, match='subtree|constraint|violation'):
        _bat_tay(ca, crt, key, '8.8.8.8')


def test_ca_giu_nguyen_giua_cac_lan_chay_khoa_chi_chu_may_doc_duoc(tmp_path):
    # đổi CA là người dùng phải cài lại trên iPhone
    crt1, key = ca_goc(str(tmp_path))
    with open(crt1, 'rb') as f:
        dau = f.read()
    crt2, _ = ca_goc(str(tmp_path))
    with open(crt2, 'rb') as f:
        assert f.read() == dau
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600


def test_chung_chi_may_dung_dieu_kien_cua_apple(tmp_path):
    # iOS từ chối chứng chỉ máy chủ thiếu SAN, thiếu serverAuth, hoặc hạn quá 825 ngày
    ca_goc(str(tmp_path))
    crt, _ = cap_chung_chi_may(str(tmp_path), ['172.20.10.2'])
    with open(crt, 'rb') as f:
        c = x509.load_pem_x509_certificate(f.read())
    assert c.not_valid_after_utc - c.not_valid_before_utc <= datetime.timedelta(days=825)
    eku = c.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert x509.oid.ExtendedKeyUsageOID.SERVER_AUTH in eku


def test_cap_lai_theo_ip_moi_ma_khong_doi_ca(tmp_path):
    # điểm phát iPhone có thể cấp IP khác cho laptop giữa hai lần chạy
    ca, _ = ca_goc(str(tmp_path))
    cap_chung_chi_may(str(tmp_path), ['172.20.10.2'])
    crt, key = cap_chung_chi_may(str(tmp_path), ['172.20.10.5'])
    _bat_tay(ca, crt, key, '172.20.10.5')


@pytest.mark.parametrize('ds, muon', [
    (['192.168.55.106', '172.20.10.3'], '172.20.10.3'),    # có điểm phát iPhone thì mã QR phải trỏ vào đó
    (['192.168.55.106'], '192.168.55.106'),
])
def test_chon_ip_uu_tien_diem_phat_iphone(ds, muon):
    assert chon_ip(ds) == muon


def test_ip_cong_khai_khong_lam_hong_ca_chung_chi(tmp_path):
    # một IP ngoài dải CA trong chứng chỉ là iPhone từ chối cả chứng chỉ, kể cả khi đang nối bằng IP nội bộ
    ds = loc_noi_bo(['8.8.8.8', '172.20.10.2', '192.168.1.5'])
    assert ds == ['172.20.10.2', '192.168.1.5']
    ca, _ = ca_goc(str(tmp_path))
    crt, key = cap_chung_chi_may(str(tmp_path), ds)
    _bat_tay(ca, crt, key, '172.20.10.2')


def test_chon_ip_khong_co_mang_bao_ro():
    with pytest.raises(RuntimeError, match='WiFi'):
        chon_ip([])


def test_phat_chung_chi_qua_http_khong_lo_khoa(tmp_path):
    # iPhone chưa tin CA nên phải tải qua http thường: chỉ được phát phần công khai
    crt, key = ca_goc(str(tmp_path))
    sv = phat_chung_chi(crt, '127.0.0.1', cong=0)
    cong = sv.server_address[1]
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{cong}/glassai.crt') as r:
            than = r.read()
            assert r.headers['Content-Type'] == 'application/x-x509-ca-cert'
        with open(crt, 'rb') as f:
            assert than == f.read()
        with urllib.request.urlopen(f'http://127.0.0.1:{cong}/') as r:
            trang = r.read().decode()
        assert 'href="/glassai.crt"' in trang
        with open(key, 'rb') as f:
            assert f.read() not in than
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f'http://127.0.0.1:{cong}/ca.key')
    finally:
        sv.shutdown()
        sv.server_close()
