#!/usr/bin/env python3
"""
Chạy thử kính trực tiếp: nạp module né vật cản, mở máy chủ ở 127.0.0.1:8765, mở đường hầm https (Cloudflare
Quick Tunnel) và in mã QR để iPhone quét. Ctrl+C để dừng; video + log của phiên nằm ở truc_tiep/phien/<giờ>/.

Chạy (đứng ở glassai/):
  .venv/bin/python loi_do_sau/truc_tiep/chay.py              # ảnh thô từ điện thoại
  .venv/bin/python loi_do_sau/truc_tiep/chay.py --gia-lap    # làm ảnh giống camera kính rẻ trước khi xử lý
                                                                     # (khung.mp4 vẫn lưu ảnh THÔ; chấm lại offline thì chạy gia_lap_kinh.py trước)
  .venv/bin/python loi_do_sau/truc_tiep/chay.py --khong-ham  # chỉ trên máy này (thử bằng Chrome)
  .venv/bin/python loi_do_sau/truc_tiep/chay.py --dan-vach   # chạy cả hệ thống DẪN THEO VẠCH (iPhone kẹp mũ
                                                                     # thay kính): ảnh 4:3 640x480, 8 khung/giây
  .venv/bin/python loi_do_sau/truc_tiep/chay.py --noi-thang  # như --dan-vach nhưng iPhone nối thẳng laptop qua
                                                                     # điểm phát WiFi, không qua Internet: đi thử ngoài
                                                                     # đường, laptop trong ba lô (nối WiFi trước khi chạy)
"""
import argparse
import concurrent.futures
import datetime
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if GOC not in sys.path:
    sys.path.insert(0, GOC)

CONG = 8765
HFOV_DAN_VACH = 55.0   # phần 4:3 cắt từ camera iPhone (~70° ở 16:9 -> ~55°)
_URL = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')


def tim_url(dong: str) -> str | None:
    """Lấy địa chỉ https của đường hầm từ một dòng log cloudflared (None nếu dòng đó không có)."""
    m = _URL.search(dong)
    if not m or m.group(0) == "https://api.trycloudflare.com":   # địa chỉ API trong dòng lỗi, không phải đường hầm
        return None
    return m.group(0)


def hfov_ho_so(cfg: dict) -> float:
    """Góc nhìn ngang (độ) của ảnh mà hồ sơ tạo từ khung 16:9 của trang (cách cắt: HoSoAnh._ong_kinh)."""
    hfov = cfg.get('hfov_ong_kinh')
    if hfov is None:        # hồ sơ "khong": chỉ cắt giữa 4:3, còn 3/4 bề ngang
        hfov = 2 * math.degrees(math.atan(0.75 * math.tan(math.radians(cfg.get('hfov_nguon', 69.0) / 2))))
    return hfov


# App VisionAid trên GitHub Pages: địa chỉ cố định, cài ra màn hình chính một lần dùng mãi. Mã QR mở app này và
# mang theo địa chỉ máy tính (may=) + mã (k=). --trang-may-tinh: mở trang do chính máy tính phục vụ (như trước).
PAGES = 'https://ducanhdhtb06-hub.github.io/VisionAid/'


def dia_chi_qr(goc: str, ma: str, them: str = '', pages: str | None = PAGES) -> str:
    """Địa chỉ trong mã QR. goc = địa chỉ https của máy tính (đường hầm hoặc https://IP:cổng)."""
    if pages:
        return f'{pages}?may={goc.split("://", 1)[1].rstrip("/")}&k={ma}{them}'
    return f'{goc.rstrip("/")}/?k={ma}{them}'


def in_qr(url):
    import qrcode
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.print_ascii(invert=True)


def mo_ham(ma, them='', cong=CONG, pages=PAGES):
    exe = shutil.which('cloudflared') or os.path.expanduser('~/.local/bin/cloudflared')
    # --protocol http2: mạng ở đây bóp QUIC/UDP (đo 2026-10-27 qua đường hầm: QUIC 0,8 khung/s, trễ ~3 s;
    # HTTP/2 7,3 khung/s, trễ ~0,2 s)
    p = subprocess.Popen([exe, 'tunnel', '--no-autoupdate', '--protocol', 'http2', '--url', f'http://127.0.0.1:{cong}'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def doc():
        assert p.stderr is not None
        for dong in p.stderr:
            url = tim_url(dong)
            if url:
                day_du = dia_chi_qr(url, ma, them, pages)
                print(f'\n=== Mở trên iPhone (quét mã QR bằng Camera) ===\n{day_du}\n', flush=True)
                in_qr(day_du)
                print('Đường hầm cần 10–30 giây để chạy sau khi hiện địa chỉ.', flush=True)
    threading.Thread(target=doc, daemon=True).start()
    return p


def mo_noi_thang(ma, them, cong, pages=PAGES):
    """iPhone nối thẳng laptop, không qua Internet. In hai mã QR; trả tham số https cho uvicorn."""
    from loi_do_sau.truc_tiep import chung_chi as cc
    ds = cc.ip_may()
    ip = cc.chon_ip(ds)
    ca, _ = cc.ca_goc(cc.THU_MUC)
    crt, key = cc.cap_chung_chi_may(cc.THU_MUC, ds)
    cc.phat_chung_chi(ca, ip)
    url_cc, url = f'http://{ip}:{cc.CONG_CHUNG_CHI}/', dia_chi_qr(f'https://{ip}:{cong}', ma, them, pages)
    print(f'\n=== Lần đầu dùng iPhone này: quét mã này để cài chứng chỉ ===\n{url_cc}\n', flush=True)
    in_qr(url_cc)
    print(f'\n=== Mở trên iPhone (quét mã QR bằng Camera) ===\n{url}\n', flush=True)
    in_qr(url)
    print('Gập nắp laptop được, máy không ngủ. Máy nóng khi chạy: để ba lô thoáng, đừng nhét kín.', flush=True)
    return dict(ssl_certfile=crt, ssl_keyfile=key)


# Model nhìn TRÊN MÁY cho nút Hỏi (dự phòng cuối khi Gemini lỗi / mất mạng, quan_sat.MAY_NHIN_URL): llama.cpp
# llama-server bản Vulkan (MIT) + Qwen3-VL-2B-Instruct GGUF Q4_K_M (Apache-2.0), tải 2026-11-10, đã kiểm SHA-256.
LLAMA_SERVER = os.getenv('AIGLASS_LLAMA_SERVER',
                         os.path.expanduser('~/.local/opt/llama.cpp-b12243/llama-b12243/llama-server'))
QWEN_VL_DIR = os.getenv('AIGLASS_QWEN_VL_DIR', os.path.expanduser('~/.local/share/glassai/qwen3vl'))
QWEN_VL_MODEL = 'Qwen3VL-2B-Instruct-Q4_K_M.gguf'
QWEN_VL_MMPROJ = 'mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf'
CONG_MAY_NHIN = 8090   # khớp quan_sat.MAY_NHIN_URL mặc định
_GPU = re.compile(r'^\s*(Vulkan\d+):\s*(.+)$', re.M)


def chon_gpu(ds: str) -> str | None:
    """GPU NVIDIA trong danh sách `llama-server --list-devices`. Không chọn GPU tích hợp: nó lấy RAM chung, chạy ở đó
    RAM máy còn 0,75 GB (đo 2026-11-10), dễ làm máy chủ dẫn đường bị giết vì hết RAM. Không có thì None."""
    return next((ten for ten, mo_ta in _GPU.findall(ds) if 'NVIDIA' in mo_ta.upper()), None)


def _liet_ke_gpu(exe: str) -> str:
    return subprocess.run([exe, '--list-devices'], capture_output=True, text=True, timeout=60).stdout


def _cong_dang_mo(cong: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', cong)) == 0


def bat_may_nhin(thu_muc: str):
    """Bật llama-server cho nút Hỏi. Trả (Popen hoặc None, model trên máy có dùng được không).
    AIGLASS_MAY_NHIN=0: tắt. Đang chạy sẵn (cổng mở): dùng lại, không bật bản thứ hai. Thiếu tệp: bỏ qua, chỉ Gemini.
    Bật SAU khi model dẫn đường đã nạp: llama lấy phần bộ nhớ GPU còn lại (~2,2 GB), dẫn đường không thiếu chỗ."""
    if os.getenv('AIGLASS_MAY_NHIN', '1') == '0':
        return None, False
    model, mmproj = os.path.join(QWEN_VL_DIR, QWEN_VL_MODEL), os.path.join(QWEN_VL_DIR, QWEN_VL_MMPROJ)
    if not all(os.path.exists(f) for f in (LLAMA_SERVER, model, mmproj)):
        print(f'Chưa có model trên máy (llama-server + Qwen3-VL-2B ở {QWEN_VL_DIR}): nút Hỏi chỉ dùng Gemini.',
              flush=True)
        return None, False
    if _cong_dang_mo(CONG_MAY_NHIN):
        print(f'Model trên máy: dùng llama-server đang chạy ở cổng {CONG_MAY_NHIN}.', flush=True)
        return None, True
    lenh = [LLAMA_SERVER, '-m', model, '--mmproj', mmproj, '-ngl', '99', '-c', '4096',
            '--host', '127.0.0.1', '--port', str(CONG_MAY_NHIN)]
    gpu = os.getenv('AIGLASS_MAY_NHIN_GPU') or chon_gpu(_liet_ke_gpu(LLAMA_SERVER))
    if gpu:
        lenh[5:5] = ['--device', gpu]
    log = open(os.path.join(thu_muc, 'llama_server.log'), 'w')
    p = subprocess.Popen(lenh, stdout=log, stderr=subprocess.STDOUT)
    print(f'Model trên máy: bật llama-server (Qwen3-VL-2B, {gpu or "GPU mặc định"}, cổng {CONG_MAY_NHIN}).', flush=True)
    return p, True


def giu_may_thuc():
    """Giữ laptop không ngủ khi gập nắp, suốt lúc chạy. Trả Popen: phải giữ biến này, mất nó là khóa nhả.
    Khóa nằm ở tiến trình con đọc đường ống: chay.py chết kiểu gì (SIGTERM không qua finally) ống cũng đóng, khóa tự nhả;
    khóa kẹt lại thì laptop không bao giờ ngủ nữa."""
    exe = shutil.which('systemd-inhibit')
    if exe is None:
        print('CẢNH BÁO: không có systemd-inhibit, gập nắp laptop sẽ ngủ và ngắt kết nối.', flush=True)
        return None
    return subprocess.Popen([exe, '--what=handle-lid-switch:sleep:idle', '--who=glassai', '--why=Đang thử kính',
                             '--mode=block', 'cat'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gia-lap', action='store_true', help='làm ảnh giống camera kính rẻ (gia_lap_kinh.LamXau)')
    ap.add_argument('--khong-ham', action='store_true', help='không mở đường hầm, chỉ http://127.0.0.1')
    ap.add_argument('--cong', type=int, default=CONG, help='cổng máy chủ (mặc định 8765)')
    ap.add_argument('--dan-vach', action='store_true', help='chạy hệ thống dẫn theo vạch thay vì chỉ module né vật cản')
    ap.add_argument('--noi-thang', action='store_true',
                    help='như --dan-vach, iPhone nối thẳng laptop qua điểm phát WiFi, không đường hầm (https nội bộ)')
    ap.add_argument('--hfov', type=float, default=None,
                    help='góc nhìn ngang (độ) báo cho máy chủ. --dan-vach: mặc định 55 (phần 4:3 cắt từ camera '
                         'iPhone). --cau-noi: mặc định AIGLASS_HFOV nếu đã đặt, không thì theo hồ sơ ảnh')
    ap.add_argument('--cau-noi', action='store_true',
                    help='chế độ B: iPhone làm kính -> ESP32 ảo -> app_main THẬT (tự bật app_main, cổng 8081)')
    ap.add_argument('--ho-so', default='ov2640_xiao', help='--cau-noi: hồ sơ ảnh ESP32 (ov2640_xiao/ov3660_xiao/khong)')
    ap.add_argument('--goc-viet', action='store_true',
                    help='--cau-noi: chạy bản GỐC ĐÃ VIỆT HÓA (OpenAIglasses_goc_viet: logic y hệt gốc, chữ tiếng Việt)')
    ap.add_argument('--goc-trung-quoc', action='store_true',
                    help='--cau-noi: chạy CODE GỐC Trung Quốc y hệt (bak-zh) nói tiếng Việt, để so với bản mới')
    ap.add_argument('--hoi-gemini', action='store_true',
                    help='nút Hỏi dùng Gemini (ảnh gửi tới Google) thay vì model nhìn trên máy; mặc định: có model '
                         'trên máy thì chỉ dùng nó (nhanh, ảnh không rời máy), không có thì dùng Gemini')
    ap.add_argument('--trang-may-tinh', action='store_true',
                    help='mã QR mở trang do máy tính phục vụ thay vì app VisionAid trên GitHub Pages')
    args = ap.parse_args()
    if args.cau_noi:   # --noi-thang dùng được: so bản gốc với bản mới trên cùng đường mạng (vẫn chậm hơn ~1,5 s vì
        return chay_cau_noi(args)   # đường cầu nối chờ hết câu mới có tiếng)
    if args.noi_thang:
        args.dan_vach = True

    import uvicorn
    from loi_do_sau.truc_tiep.may_chu import EngineDanVach, EngineThat, tao_app

    ma = secrets.token_urlsafe(12)
    pages = None if args.trang_may_tinh else PAGES
    thu_muc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phien',
                           datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(thu_muc, exist_ok=True)
    lam_xau = None
    if args.gia_lap:
        from loi_do_sau.gia_lap_kinh import LamXau
        lam_xau = LamXau()
    print('Đang nạp model (khoảng 20–40 giây)...', flush=True)
    them, kich_thuoc = '', (640, 360)
    luong = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='engine')
    if args.dan_vach:
        os.environ['AIGLASS_HFOV'] = str(args.hfov or HFOV_DAN_VACH)   # trước khi tạo DanTheoVach (đọc env lúc khởi tạo)
        engine = EngineDanVach(lam_xau)
        luong.submit(engine.lam_nong).result()     # đúng luồng sẽ xử lý khung (xem tao_app)
        them, kich_thuoc = '&khung=43&fps=8', (640, 480)
    else:
        engine = EngineThat(lam_xau)
    from khoi_dong_dan_duong import dong_bang_bo_nho  # pyright: ignore[reportMissingImports]
    dong_bang_bo_nho()
    from loi_do_sau.truc_tiep.hoi_dap import HoiGemini, HoiTrenMay, co_khoa
    hoi = None
    may_nhin, co_may_nhin = bat_may_nhin(thu_muc)
    if co_may_nhin and not args.hoi_gemini:
        hoi = HoiTrenMay()
        print('Nút "Hỏi": bật, chỉ model nhìn trên máy (Qwen3-VL-2B), điện thoại tự đọc. Ảnh không rời máy tính. '
              '--hoi-gemini để dùng Gemini.', flush=True)
    elif co_khoa():
        hoi = HoiGemini()
        print('Nút "Hỏi": bật (Gemini' + (', dự phòng cuối: model trên máy' if co_may_nhin else '') +
              '). Ảnh camera và câu hỏi được gửi tới Google.', flush=True)
    elif co_may_nhin:
        hoi = HoiGemini()
        print('Nút "Hỏi": bật, chỉ model trên máy (chưa có GEMINI_API_KEY). Ảnh không rời máy tính.', flush=True)
    else:
        print('Nút "Hỏi": TẮT — chưa có GEMINI_API_KEY trong OpenAIglasses_for_Navigation/.env', flush=True)
    app = tao_app(engine, ma, thu_muc, kich_thuoc, luong=luong, hoi=hoi)
    ham = thuc = None
    ket_noi: dict = dict(host='127.0.0.1')
    if args.noi_thang:
        ket_noi = dict(host='0.0.0.0', **mo_noi_thang(ma, them, args.cong, pages))
        thuc = giu_may_thuc()
    elif args.khong_ham:
        print(f'\nMở: http://127.0.0.1:{args.cong}/?k={ma}{them}\n', flush=True)
    else:
        ham = mo_ham(ma, them, args.cong, pages)
    try:
        # iPhone còn mở trang thì uvicorn chờ kết nối đóng mãi khi Ctrl+C -> app_main mồ côi giữ cổng (lỗi 2026-10-28)
        uvicorn.run(app, port=args.cong, log_level='warning', timeout_graceful_shutdown=3, **ket_noi)
    finally:
        if ham:
            ham.terminate()
        if thuc:
            thuc.terminate()
        if may_nhin:
            may_nhin.terminate()
        print(f'\nPhiên đã lưu ở: {thu_muc}\nHậu kỳ: .venv/bin/python loi_do_sau/truc_tiep/hau_ky.py {thu_muc}')


def chay_cau_noi(args):
    import uvicorn
    sys.path.insert(0, os.path.join(GOC, 'OpenAIglasses_for_Navigation', 'tools'))
    from esp32_ao.ho_so_anh import doc_ho_so
    from esp32_ao.phat_lai import bat_may_chu, may_chu_dang_chay, tat_may_chu
    from loi_do_sau.truc_tiep.cau_noi import tao_app_cau_noi

    if may_chu_dang_chay('127.0.0.1', 8081):
        print('Cổng 8081 đang có máy chủ khác — tắt nó trước (không phải do cầu nối bật).', flush=True)
        raise SystemExit(2)
    ma = secrets.token_urlsafe(12)
    thu_muc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phien',
                           ('goc_viet_' if args.goc_viet else 'goc_zh_' if args.goc_trung_quoc else 'cau_noi_') + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(thu_muc, exist_ok=True)
    nhat_ky = os.path.join(thu_muc, 'nhat_ky_may_chu.jsonl')
    print('Đang bật app_main (nạp model 30–60 giây)...', flush=True)
    if args.goc_viet:
        ban = os.path.join(GOC, 'OpenAIglasses_goc_viet')
        print(f'BẢN GỐC ĐÃ VIỆT HÓA (logic y hệt gốc): {ban}', flush=True)
        p = bat_may_chu(nhat_ky, {}, log=os.path.join(thu_muc, 'app_main.log'), thu_muc=ban, boc_ghi_cau=True)
        app = tao_app_cau_noi(ma, thu_muc, args.ho_so, nhat_ky_may_chu=nhat_ky)
    elif args.goc_trung_quoc:
        from loi_do_sau.truc_tiep.goc_trung_quoc import (CAU_BAT_ZH, CAU_TAT_ZH, tao_ban_chay,
                                                                 tao_giong_viet)
        ban = tao_ban_chay(os.path.join(GOC, 'goc_trung_quoc_chay'))
        env = {'VOICE_DIR': tao_giong_viet(os.path.join(ban, 'giong_viet'))}
        print(f'CODE GỐC Trung Quốc (chỉ đổi giọng sang tiếng Việt): {ban}', flush=True)
        p = bat_may_chu(nhat_ky, env, log=os.path.join(thu_muc, 'app_main.log'), thu_muc=ban)
        app = tao_app_cau_noi(ma, thu_muc, args.ho_so, nhat_ky_may_chu=nhat_ky, cau_bat=CAU_BAT_ZH, cau_tat=CAU_TAT_ZH)
    else:
        # dan_theo_vach và module né vật cản cùng đọc biến này (hai bản gốc ở trên thì không)
        hfov = args.hfov or os.environ.get('AIGLASS_HFOV') or hfov_ho_so(doc_ho_so(args.ho_so))
        print(f'Góc nhìn báo cho máy chủ: {float(hfov):.1f}°', flush=True)
        p = bat_may_chu(nhat_ky, {'AIGLASS_HFOV': hfov}, log=os.path.join(thu_muc, 'app_main.log'))
        app = tao_app_cau_noi(ma, thu_muc, args.ho_so, nhat_ky_may_chu=nhat_ky)
    them = '&fps=10'
    ham = thuc = None
    ket_noi: dict = dict(host='127.0.0.1')
    pages = None if args.trang_may_tinh else PAGES
    try:
        if args.noi_thang:
            ket_noi = dict(host='0.0.0.0', **mo_noi_thang(ma, them, args.cong, pages))
            thuc = giu_may_thuc()
        elif args.khong_ham:
            print(f'\nMở: http://127.0.0.1:{args.cong}/?k={ma}{them}\n', flush=True)
        else:
            ham = mo_ham(ma, them, args.cong, pages)
        # iPhone còn mở trang thì uvicorn chờ kết nối đóng mãi khi Ctrl+C -> app_main mồ côi giữ cổng (lỗi 2026-10-28)
        uvicorn.run(app, port=args.cong, log_level='warning', timeout_graceful_shutdown=3, **ket_noi)
    finally:
        # uvicorn phát lại Ctrl+C sau khi tắt -> KeyboardInterrupt cắt ngang lúc chờ app_main tắt, app_main mồ côi giữ
        # cổng 8081/12345 (lỗi thật 2026-10-28). Đang dọn thì bỏ qua Ctrl+C / TERM tiếp theo.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if ham:
            ham.terminate()
        if thuc:
            thuc.terminate()
        tat_may_chu(p)
        print(f'\nPhiên đã lưu ở: {thu_muc}\nXuất video: .venv/bin/python OpenAIglasses_for_Navigation/tools/esp32_ao/'
              f'xuat_video.py {thu_muc}/<phiên con> --ra demo.mp4')


if __name__ == '__main__':
    main()
