"""Chứng chỉ https để iPhone nối thẳng laptop qua điểm phát WiFi (chay.py --noi-thang), không qua Cloudflare.

Safari chỉ mở camera trên https, và wss cần chứng chỉ được tin hẳn (bỏ qua cảnh báo trên trang là chưa đủ), nên phải cài
một CA riêng lên iPhone, một lần. CA chỉ ký được cho IP nội bộ (name constraints): khóa có lộ cũng không giả được trang
web thật trên iPhone. Khóa nằm ở ~/.config/glassai/chung_chi/, không nằm trong repo.
"""
import datetime
import http.server
import ipaddress
import os
import re
import subprocess
import threading

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

THU_MUC = os.path.expanduser('~/.config/glassai/chung_chi')
CONG_CHUNG_CHI = 8766
DIEM_PHAT_IPHONE = ipaddress.ip_network('172.20.10.0/28')     # dải IP cố định của Điểm truy cập cá nhân trên iPhone
MANG_NOI_BO = [ipaddress.ip_network(n) for n in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '127.0.0.0/8')]
TRANG = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Cài chứng chỉ</title>
<style>body{font:20px/1.5 -apple-system,sans-serif;padding:20px;max-width:640px;margin:auto}
a{display:block;padding:18px;text-align:center;background:#0a7;color:#fff;border-radius:14px;text-decoration:none;
font-weight:700}li{margin:12px 0}</style></head><body><h2>Cài chứng chỉ (một lần)</h2>
<a href="/glassai.crt">Tải chứng chỉ</a><ol>
<li>Bấm nút trên rồi chọn <b>Cho phép</b>.</li>
<li>Mở <b>Cài đặt</b>, bấm <b>Đã tải về hồ sơ</b> ở đầu màn hình, rồi <b>Cài đặt</b>.</li>
<li>Vào <b>Cài đặt chung &gt; Giới thiệu &gt; Cài đặt tin cậy chứng nhận</b>, bật <b>GlassAI</b>.</li>
<li>Quay lại quét mã QR thứ hai trên laptop.</li></ol>
<p>Thử xong muốn gỡ: Cài đặt chung &gt; Quản lý VPN &amp; thiết bị &gt; GlassAI &gt; Xóa hồ sơ.</p></body></html>"""


def _ghi_khoa(duong_dan, khoa):
    du_lieu = khoa.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
    fd = os.open(duong_dan, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(du_lieu)


def _doc_ca(crt, key):
    with open(crt, 'rb') as f:
        chung_chi = x509.load_pem_x509_certificate(f.read())
    with open(key, 'rb') as f:
        khoa = serialization.load_pem_private_key(f.read(), password=None)
    assert isinstance(khoa, ec.EllipticCurvePrivateKey)
    return chung_chi, khoa


def ca_goc(thu_muc=THU_MUC):
    """(đường dẫn ca.crt, ca.key); tạo nếu chưa có, có rồi thì giữ nguyên: đổi CA là phải cài lại trên iPhone."""
    crt, key = os.path.join(thu_muc, 'ca.crt'), os.path.join(thu_muc, 'ca.key')
    if os.path.exists(crt) and os.path.exists(key):
        return crt, key
    os.makedirs(thu_muc, mode=0o700, exist_ok=True)
    khoa = ec.generate_private_key(ec.SECP256R1())
    ten = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'GlassAI (chi mang noi bo)')])
    nay = datetime.datetime.now(datetime.timezone.utc)
    chung_chi = (
        x509.CertificateBuilder().subject_name(ten).issuer_name(ten).public_key(khoa.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(nay - datetime.timedelta(days=1)).not_valid_after(nay + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
                                     encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.NameConstraints(permitted_subtrees=[x509.IPAddress(n) for n in MANG_NOI_BO],
                                            excluded_subtrees=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(khoa.public_key()), critical=False)
        .sign(khoa, hashes.SHA256()))
    _ghi_khoa(key, khoa)
    with open(crt, 'wb') as f:
        f.write(chung_chi.public_bytes(serialization.Encoding.PEM))
    return crt, key


def cap_chung_chi_may(thu_muc, ips):
    """(may.crt, may.key) ký bằng CA cho đúng các IP này. Cấp lại mỗi lần chạy vì IP laptop có thể đổi."""
    ca, khoa_ca = _doc_ca(*ca_goc(thu_muc))
    khoa = ec.generate_private_key(ec.SECP256R1())
    nay = datetime.datetime.now(datetime.timezone.utc)
    chung_chi = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'GlassAI laptop')]))
        .issuer_name(ca.subject).public_key(khoa.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(nay - datetime.timedelta(days=1)).not_valid_after(nay + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(i)) for i in ips]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(khoa_ca.public_key()), critical=False)
        .sign(khoa_ca, hashes.SHA256()))
    crt, key = os.path.join(thu_muc, 'may.crt'), os.path.join(thu_muc, 'may.key')
    _ghi_khoa(key, khoa)
    with open(crt, 'wb') as f:
        f.write(chung_chi.public_bytes(serialization.Encoding.PEM))
    return crt, key


def loc_noi_bo(ds):
    """Chỉ giữ IP trong dải CA được ký: một IP công khai trong chứng chỉ làm iPhone từ chối cả chứng chỉ."""
    return [i for i in ds if any(ipaddress.ip_address(i) in n for n in MANG_NOI_BO)]


def ip_may():
    """Mọi IPv4 nội bộ của laptop, không tính 127.0.0.1."""
    ra = subprocess.run(['ip', '-4', '-o', 'addr', 'show', 'scope', 'global'], capture_output=True, text=True).stdout
    return loc_noi_bo(re.findall(r'inet (\d+\.\d+\.\d+\.\d+)/', ra))


def chon_ip(ds):
    """IP in trong mã QR: ưu tiên điểm phát iPhone, vì laptop có thể đang nối thêm mạng khác."""
    if not ds:
        raise RuntimeError('Laptop chưa nối mạng WiFi nào: bật Điểm truy cập cá nhân trên iPhone rồi cho laptop nối vào.')
    return next((i for i in ds if ipaddress.ip_address(i) in DIEM_PHAT_IPHONE), ds[0])


def phat_chung_chi(ca_crt, ip, cong=CONG_CHUNG_CHI):
    """Phát phần công khai của CA qua http thường (iPhone chưa tin CA nên chưa vào https được). Trả server đang chạy."""
    with open(ca_crt, 'rb') as f:
        than = f.read()

    class Xu(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            noi_dung = {'/': (TRANG.encode(), 'text/html; charset=utf-8'),
                        '/glassai.crt': (than, 'application/x-x509-ca-cert')}.get(self.path)
            if noi_dung is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', noi_dung[1])
            self.send_header('Content-Length', str(len(noi_dung[0])))
            self.end_headers()
            self.wfile.write(noi_dung[0])

        def log_message(self, format, *args):
            pass

    sv = http.server.ThreadingHTTPServer((ip, cong), Xu)
    threading.Thread(target=sv.serve_forever, daemon=True).start()
    return sv
