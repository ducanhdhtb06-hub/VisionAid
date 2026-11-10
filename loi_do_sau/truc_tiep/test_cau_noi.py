"""Test cầu nối iPhone -> ESP32 ảo -> app_main (cau_noi.py), dùng máy chủ giả có endpoint như app_main.
Chạy (đứng ở glassai/): .venv/bin/python -m pytest loi_do_sau/truc_tiep/test_cau_noi.py -q
"""
import concurrent.futures
import json
import math
import os
import struct
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(GOC, "OpenAIglasses_for_Navigation"), os.path.join(GOC, "OpenAIglasses_for_Navigation", "tests"),
          os.path.join(GOC, "OpenAIglasses_for_Navigation", "tools")):
    sys.path.insert(0, p)
from esp32_ao.do_anh_mau import doc_jpeg  # noqa: E402
from esp32_ao.ho_so_anh import doc_ho_so  # noqa: E402
from may_chu_gia_esp32 import chay_may_chu_gia  # noqa: E402
from loi_do_sau.truc_tiep.cau_noi import tao_app_cau_noi  # noqa: E402
from loi_do_sau.truc_tiep.goi_tin import dong_goi  # noqa: E402

MA = "ma-cau-noi"


def _jpeg(rong=1280, cao=720):
    img = cv2.resize(np.random.default_rng(0).integers(0, 255, (18, 32, 3), np.uint8), (rong, cao))
    return cv2.imencode(".jpg", img)[1].tobytes()


def _tieng(giay, sr=8000):
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr))) for i in range(int(giay * sr)))


def _app(mc, tmp_path, nhat_ky=None):
    return TestClient(tao_app_cau_noi(MA, str(tmp_path / "phien"), "ov2640_xiao", cong_may_chu=mc.cong,
                                      cong_udp=mc.cong_udp, nhat_ky_may_chu=nhat_ky))


def _cho(dk, giay=5.0):
    han = time.time() + giay
    while time.time() < han:
        if dk():
            return True
        time.sleep(0.05)
    return False


def _nhan(ws, giay=5.0):
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(ws.receive).result(timeout=giay)


def test_sai_ma_dong_1008(tmp_path):
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c:
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/ws?k=sai") as ws:
                ws.receive_json()
        assert e.value.code == 1008


def test_khung_iphone_thanh_khung_esp32(tmp_path):
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c, c.websocket_connect(f"/ws?k={MA}") as ws:
        ws.send_bytes(dong_goi(1_000_000.0, 1, _jpeg()))
        assert _cho(lambda: len(mc.khung) > 0)
        d = doc_jpeg(mc.khung[0][1])
    assert (d["rong"], d["cao"]) == (640, 480)
    assert d["bang_luong_tu"] == doc_ho_so("ov2640_xiao")["bang_luong_tu"]


def test_bat_dau_gui_prompt(tmp_path):
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c, c.websocket_connect(f"/ws?k={MA}") as ws:
        ws.send_text(json.dumps({"lenh": "bat_dau"}))
        assert _cho(lambda: "PROMPT:bắt đầu dẫn đường" in mc.tin_mic)


def test_cau_lenh_tieng_trung_cho_code_goc(tmp_path):
    # code gốc Trung Quốc (goc_trung_quoc.py) chỉ hiểu "开始导航" / "停止导航"
    with chay_may_chu_gia() as mc, TestClient(tao_app_cau_noi(
            MA, str(tmp_path / "phien"), "ov2640_xiao", cong_may_chu=mc.cong, cong_udp=mc.cong_udp,
            cau_bat="开始导航", cau_tat="停止导航")) as c, c.websocket_connect(f"/ws?k={MA}") as ws:
        ws.send_text(json.dumps({"lenh": "bat_dau"}))
        assert _cho(lambda: "PROMPT:开始导航" in mc.tin_mic)
        ws.send_text(json.dumps({"lenh": "dung"}))
        assert _cho(lambda: "PROMPT:停止导航" in mc.tin_mic)


def test_bat_dau_truoc_khung_dau_van_co_camera(tmp_path):
    # app_main tạo bộ điều phối khi camera nối: lệnh bật đến trước camera thì bị bỏ qua -> cầu nối phải nối camera trước
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c, c.websocket_connect(f"/ws?k={MA}") as ws:
        ws.send_text(json.dumps({"lenh": "bat_dau"}))
        assert _cho(lambda: len(mc.prompt_khi_co_camera) == 1)
        assert mc.prompt_khi_co_camera == [True]


def test_dung_gui_prompt_dung_va_dong_camera(tmp_path):
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c, c.websocket_connect(f"/ws?k={MA}") as ws:
        ws.send_bytes(dong_goi(1_000_000.0, 1, _jpeg()))
        assert _cho(lambda: mc.camera_ket_noi == 1)
        ws.send_text(json.dumps({"lenh": "dung"}))
        assert _cho(lambda: "PROMPT:dừng dẫn đường" in mc.tin_mic)
        assert _cho(lambda: "cam_ngat" in [s for _, s in mc.su_kien], 1.5)


def test_cau_tu_may_chu_toi_iphone(tmp_path):
    nk = tmp_path / "nk.jsonl"
    kb = dict(loa=[(1.0, _tieng(0.4)), (0.4, bytes(8000))], cau=[(1.0, "Dịch sang phải")], nhat_ky=str(nk))
    with chay_may_chu_gia(kb) as mc, _app(mc, tmp_path, str(nk)) as c, c.websocket_connect(f"/ws?k={MA}") as ws:
        ws.send_bytes(dong_goi(1_000_000.0, 7, _jpeg()))
        tin = _nhan(ws)
        while tin.get("text") and json.loads(tin["text"])["cau"] is None:   # bỏ tin xác nhận khung
            tin = _nhan(ws)
        assert json.loads(tin["text"])["cau"] == "Dịch sang phải"
        wav = _nhan(ws)
        assert wav["bytes"][:4] == b"RIFF"


def test_iphone_rot_mang_thi_dong_camera(tmp_path):
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c:
        with c.websocket_connect(f"/ws?k={MA}") as ws:
            ws.send_bytes(dong_goi(1_000_000.0, 1, _jpeg()))
            assert _cho(lambda: mc.camera_ket_noi == 1)
        assert _cho(lambda: "cam_ngat" in [s for _, s in mc.su_kien], 1.5)
        assert "PROMPT:dừng dẫn đường" not in mc.tin_mic


def test_trang_tai_duoc_am_bao(tmp_path):
    """thu_kinh.html tải /static/<tên>.wav lúc bấm bắt đầu; thiếu -> Safari "Decoding failed", trang dừng (lỗi thật
    lần thử iPhone đầu, 2026-10-26)."""
    import re
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "kinh_loi.js"),
                encoding="utf-8").read()
    ten = re.search(r"\[([^\]]*)\]\.map\(napAm\)", html).group(1)
    ten = re.findall(r"'([a-z_]+)'", ten)
    assert ten
    tc = TestClient(tao_app_cau_noi(MA, str(tmp_path / "phien"), "ov2640_xiao"))
    for t in ten:
        r = tc.get(f"/static/{t}.wav")
        assert r.status_code == 200, t
        assert r.content[:4] == b"RIFF", t


def test_moi_khung_duoc_xac_nhan(tmp_path):
    """thu_kinh.html chỉ gửi khung tiếp khi có JSON xác nhận khung trước (không có thì chờ 2 s + báo mất kết nối
    sau 1 s). Lỗi thật lần thử iPhone 2026-10-26: cầu nối chỉ gửi JSON khi có câu -> 1 khung / 2 s, "Mất hình" liên tục."""
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c, c.websocket_connect(f"/ws?k={MA}") as ws:
        for i in range(1, 4):
            t0 = time.time()
            ws.send_bytes(dong_goi(1_000_000.0 + i * 100, i, _jpeg()))
            j = _nhan(ws, 2.0)
            assert j["type"] == "websocket.send" and j.get("text"), j
            tin = json.loads(j["text"])
            assert tin["id"] == i and tin["t_chup"] == 1_000_000.0 + i * 100 and not tin.get("loi")
            assert time.time() - t0 < 1.0


def _ma_dong(ws, giay=3.0):
    """Mã đóng kết nối mà máy chủ gửi (None nếu sau `giay` giây vẫn chưa đóng); không treo test."""
    ex = concurrent.futures.ThreadPoolExecutor(1)
    try:
        han = time.time() + giay
        while time.time() < han:
            try:
                m = ex.submit(ws.receive).result(timeout=max(0.05, han - time.time()))
            except concurrent.futures.TimeoutError:
                return None
            if m["type"] == "websocket.close":
                return m.get("code")
        return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def test_ket_noi_moi_thay_ket_noi_cu(tmp_path):
    """Lỗi thật lần thử iPhone 2026-10-28: iPhone mở 2 kết nối cách nhau 1 s (2 tab / quét QR + bấm link). Kết nối cũ
    giữ camera của app_main, kết nối mới bị app_main từ chối 1013 mỗi giây -> 0 khung tới máy chủ, "mất kết nối".
    Kết nối mới nhất thắng: đóng kết nối cũ (1012), nhả camera, kết nối mới nối camera và gửi được hình."""
    with chay_may_chu_gia() as mc, _app(mc, tmp_path) as c:
        with c.websocket_connect(f"/ws?k={MA}") as ws1:
            ws1.send_bytes(dong_goi(1_000_000.0, 1, _jpeg()))
            assert _cho(lambda: mc.camera_ket_noi == 1)
            with c.websocket_connect(f"/ws?k={MA}") as ws2:
                assert _ma_dong(ws1) == 1012
                ws2.send_bytes(dong_goi(1_000_100.0, 1, _jpeg()))
                tin = json.loads(_nhan(ws2, 5.0)["text"])
                assert tin["id"] == 1 and not tin.get("loi"), tin
                assert mc.camera_ket_noi == 2


def test_trang_bi_thay_thi_khong_noi_lai(tmp_path):
    """Lỗi thật 2026-10-28: kết nối mới đóng kết nối cũ (1012) nhưng trang cũ tự nối lại sau 2 s và lại đá trang mới
    -> 2 tab đá nhau 188 lần / 3 phút, chỉ 5 khung tới máy chủ. Trang bị thay (1012) phải dừng hẳn, không nối lại;
    mất mạng thường (1006) vẫn nối lại. Chạy hàm moKetNoi() thật của trang bằng node, WebSocket/setTimeout giả."""
    import re
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("không có node")
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "kinh_loi.js"),
                encoding="utf-8").read()
    ham = re.search(r"function moKetNoi\(\) \{.*?\n\}\n", html, re.S).group(0)
    js = """
let ws, chay = true, cheDo = 'truc_tiep', saiMa = false, idDaXong = 0, idKhung = 0, ketNoiLuc = 0, phanHoiCuoi = 0;
const MA = 'x', MAY = '', location = { protocol: 'https:', host: 'h' }, hen = [], trang = [];
let soLanMo = 0, daDung = false;
class WebSocket { constructor() { soLanMo++; } send() {} }
function setTimeout(f) { hen.push(f); }
function trangThai(t) { trang.push(t); }
function nhanJson() {} function nhanWav() {}
function dung() { daDung = true; chay = false; }
""" + ham + """
function thu(code) {
  soLanMo = 0; hen.length = 0; daDung = false; chay = true;
  moKetNoi(); ws.onclose({ code }); hen.forEach(f => f());
  return { noiLai: soLanMo > 1, daDung, trang: trang[trang.length - 1] };
}
console.log(JSON.stringify({ thay: thu(1012), rot: thu(1006) }));
"""
    kq = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, check=True).stdout)
    assert not kq["thay"]["noiLai"] and kq["thay"]["daDung"], kq
    assert "khác" in kq["thay"]["trang"], kq
    assert kq["rot"]["noiLai"] and not kq["rot"]["daDung"], kq


def test_nguong_bao_mat_ket_noi_chiu_duoc_rung_cua_duong_ham():
    """Đo qua đường hầm Cloudflare 2026-10-28 (giả iPhone y như trang, 3 phút): trễ p50 285 ms, p95 665 ms, có lúc khựng
    tới 2,7 s. Ngưỡng 1 s -> báo động "MẤT KẾT NỐI" 13 lần / 3 phút (người dùng: "cứ mất kết nối"); 2,5 s -> 1 lần
    (khựng thật 2,7 s). Ngưỡng phải >= 2,5 s."""
    import re
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "kinh_loi.js"),
                encoding="utf-8").read()
    assert int(re.search(r"const MAT_KET_NOI_MS = (\d+)", html).group(1)) >= 2500


@pytest.mark.parametrize("trang", ["thu_kinh.html", "app.html"])
def test_trang_co_du_phan_tu_cho_loi(trang):
    """kinh_loi.js dùng chung cho trang thử và app VisionAid: trang thiếu một id mà lõi gọi $('...') thì lỗi lúc bấm
    Bắt đầu (null.textContent) — chỉ lộ ra trên điện thoại."""
    import re
    thu_muc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    js = open(os.path.join(thu_muc, "kinh_loi.js"), encoding="utf-8").read()
    html = open(os.path.join(thu_muc, trang), encoding="utf-8").read()
    assert re.search(r'<script src="/?static/kinh_loi\.js"></script>', html)
    can = set(re.findall(r"\$\('([a-z]+)'\)", js))
    co = set(re.findall(r'id="([a-z]+)"', html))
    assert can and can <= co, can - co
    assert 'name="cd" value="truc_tiep"' in html and 'name="cd" value="chi_ghi"' in html


def test_ma_qr_mo_app_va_trang_thu_van_con(tmp_path):
    """Mã QR trỏ "/" -> app VisionAid; trang thử cũ ở "/thu"; manifest + biểu tượng tải được (thiếu -> không cài
    được ra màn hình chính)."""
    tc = TestClient(tao_app_cau_noi(MA, str(tmp_path / "phien"), "ov2640_xiao"))
    assert "VisionAid" in tc.get("/").text
    assert "Thử kính" in tc.get("/thu").text
    man = tc.get("/static/manifest.json").json()
    for ic in man["icons"]:
        r = tc.get("/static/" + ic["src"])   # đường dẫn tương đối theo manifest (chạy cả trên Pages)
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG", ic
    assert tc.get("/static/icon-180.png").status_code == 200
