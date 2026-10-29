#!/usr/bin/env python3
"""Chế độ A của ESP32 ảo: phát lại video vào app_main THẬT như bo XIAO ESP32S3 Sense đang gửi.

Một lượt: (tự bật app_main làm tiến trình con, DASHSCOPE_API_KEY giả chỉ trong env con, nhật ký câu nói bật)
-> nối kênh ảnh/loa/xem/IMU/mic -> "PROMPT:bắt đầu dẫn đường" -> gửi khung theo THỜI GIAN THẬT (máy chủ dùng
đồng hồ thật) -> hết video chờ cho_cuoi_s -> "PROMPT:dừng dẫn đường" -> ghi thư mục phiên + bang_so.json.

Chạy (đứng ở OpenAIglasses_for_Navigation/, không có việc GPU nào khác đang chạy):
  ../.venv/bin/python tools/esp32_ao/phat_lai.py ../my/IMG_7358.MOV --giay 30 --ra test_out/esp32_ao/thu
"""
import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import cv2
import httpx

GOC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esp32_ao import danh_gia  # noqa: E402
from esp32_ao.giao_thuc import KenhAnh, KenhImu, KenhLoa, KenhMic, KenhXem  # noqa: E402
from esp32_ao.ho_so_anh import HoSoAnh, doc_ho_so  # noqa: E402
from esp32_ao.nhip import CamBien, GioiHanFps, GrabLatest, MangGia  # noqa: E402

CAU_BAT = "bắt đầu dẫn đường"
CAU_TAT = "dừng dẫn đường"


def may_chu_dang_chay(host: str, cong: int) -> bool:
    try:
        with socket.create_connection((host, cong), timeout=0.5):
            return True
    except OSError:
        return False


def bat_may_chu(nhat_ky: str, env_them: dict, log: str | None = None, cho_s: float = 120.0,
                thu_muc: str = GOC, boc_ghi_cau: bool = False) -> subprocess.Popen:
    """Bật app_main.py (cổng 8081 cố định trong app_main). Khóa API giả chỉ nằm trong env tiến trình con.
    thu_muc: thư mục chứa app_main.py (mặc định bản hiện tại; goc_trung_quoc.py dùng bản gốc)."""
    env = os.environ.copy()
    env.setdefault("DASHSCOPE_API_KEY", "gia_de_thu")
    env["AIGLASS_NHAT_KY_CAU"] = nhat_ky
    env.setdefault("AIGLASS_TAT_GHI", "1")             # không ghi video/âm thanh mỗi lượt (đầy đĩa)
    env.update({k: str(v) for k, v in env_them.items()})
    out = open(log, "w") if log else subprocess.DEVNULL
    lenh = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "boc_ghi_cau.py")] if boc_ghi_cau \
        else [sys.executable, "app_main.py"]       # code gốc không ghi câu: bọc từ ngoài (boc_ghi_cau.py)
    p = subprocess.Popen(lenh, cwd=thu_muc, env=env, stdout=out, stderr=subprocess.STDOUT)
    han = time.time() + cho_s
    while time.time() < han:
        if p.poll() is not None:
            raise RuntimeError(f"app_main thoát sớm (mã {p.returncode}), xem log: {log}")
        try:
            if httpx.get("http://127.0.0.1:8081/api/health", timeout=1.0).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    tat_may_chu(p)
    raise RuntimeError(f"app_main không sẵn sàng sau {cho_s:.0f} s, xem log: {log}")


def tat_may_chu(p: subprocess.Popen):
    p.terminate()
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(5)


def bien_anh_khong(bgr, chat_luong: int = 80) -> bytes:
    """Hồ sơ "khong": cắt giữa 4:3 -> 640x480 -> JPEG OpenCV (mặc định 80; ghi_de chat_luong) — không làm xấu; để so."""
    h, w = bgr.shape[:2]
    if h > w:
        raise ValueError("video dọc — kính quay ngang, hãy dùng video quay ngang")
    rong = min(w, int(round(h * 4 / 3)))
    x0 = (w - rong) // 2
    img = cv2.resize(bgr[:, x0:x0 + rong], (640, 480), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, chat_luong])[1].tobytes()


def _mo_video(video, bat_dau_s, giay):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"không mở được video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if h > w:
        raise ValueError("video dọc — kính quay ngang, hãy dùng video quay ngang")
    if bat_dau_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, bat_dau_s * 1000)
    so = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - bat_dau_s * fps)
    if giay:
        so = min(so, int(giay * fps))
    return cap, fps, so


async def _phien(cap, fps, so_khung, ra, bien_anh, host, cong, cong_udp, mic, seed, cho_cuoi_s, bat_dau_s,
                 cam, mang_gia, fps_gui, ghi_xem=True, cau_bat=CAU_BAT, cau_tat=CAU_TAT):
    anh = KenhAnh(host, cong)
    loa = KenhLoa(host, cong, os.path.join(ra, "loa.wav"))
    xem = KenhXem(host, cong, ra)
    imu = KenhImu(host, cong_udp, seed=seed)
    kmic = KenhMic(host, cong, che_do=mic)
    nen = [asyncio.create_task(x.chay()) for x in ((loa, xem, imu) if ghi_xem else (loa, imu))]
    if not await anh.ket_noi(cho_s=30):
        raise RuntimeError("không nối được /ws/camera")
    await kmic.ket_noi(cho_s=30)          # lần nối camera đầu máy chủ nạp model ~15 s, vòng sự kiện đứng
    if mic != "tat" and not await kmic.lenh(cau_bat):
        print("[ESP32 ẢO] Cảnh báo: máy chủ không xác nhận lệnh bắt đầu dẫn đường", flush=True)
    t_bat_dau = time.time()
    hang = GrabLatest()
    gioi_han = GioiHanFps(fps_gui)
    xong = asyncio.Event()
    with open(os.path.join(ra, "khung.jsonl"), "w", encoding="utf-8") as f:
        def ghi(id_, t_video, t_chup, t_gui, bo):
            f.write(json.dumps(dict(id=id_, t_video=t_video, t_chup=t_chup, t_gui=t_gui, bo=bo)) + "\n")

        async def chup():          # taskCamCapture: chụp theo nhịp cảm biến, GRAB_LATEST giữ khung mới nhất
            i, bgr, dem = -1, None, 0
            for tc in cam.thoi_diem_chup(0.0, so_khung / fps):
                can = int(tc * fps)
                while i < can:
                    ok, moi = cap.read()
                    if not ok:
                        break
                    i, bgr = i + 1, moi
                if bgr is None or i < can:
                    break
                cho = t_bat_dau + tc - time.time()
                if cho > 0:
                    await asyncio.sleep(cho)
                cu = hang.dat(t_bat_dau + tc, (dem, bat_dau_s + i / fps, bgr))
                if cu is not None:
                    ghi(cu[1][0], cu[1][1], cu[0], None, True)
                dem += 1
            xong.set()

        async def gui():           # taskCamSend: lấy khung mới nhất, giới hạn fps, sendBinary bị chặn theo mạng
            t_gui_truoc = -1e9
            while not (xong.is_set() and hang._o is None):
                o = hang.lay()
                if o is None:
                    await asyncio.sleep(0.002)
                    continue
                t_chup, (id_, t_video, bgr) = o
                cho = gioi_han.cho(t_gui_truoc, time.time())
                if cho > 0:
                    await asyncio.sleep(cho)
                t_gui_truoc = time.time()
                tuong_doi = t_gui_truoc - t_bat_dau
                if mang_gia.dang_rot(tuong_doi):
                    if anh.ws is not None:
                        await anh.ngat(den_t=t_bat_dau + mang_gia.t_rot + mang_gia.dai_rot)
                    ghi(id_, t_video, t_chup, None, True)
                    continue
                jpeg = bien_anh(bgr, t_chup)
                await asyncio.sleep(mang_gia.tre_gui(tuong_doi, len(jpeg)))
                ok = await anh.gui(jpeg)
                ghi(id_, t_video, t_chup, time.time() if ok else None, not ok)

        await asyncio.gather(chup(), gui())
    if mic != "tat":                                   # như người dùng bấm Dừng rồi mới tháo kính
        await kmic.lenh(cau_tat)
    await asyncio.sleep(cho_cuoi_s)                    # chờ câu cuối phát xong
    await anh.dong()
    await kmic.dong()
    for x in (loa, xem, imu):
        await x.dong()
    await asyncio.gather(*nen, return_exceptions=True)
    with open(os.path.join(ra, "loa_doan.json"), "w") as g:
        json.dump(loa.doan_tieng, g)
    with open(os.path.join(ra, "su_kien.json"), "w", encoding="utf-8") as g:
        json.dump(dict(anh=anh.su_kien, lenh_anh=anh.lenh_nhan, mic=kmic.tin_nhan, mat_tieng=loa.mat_tieng,
                       dem_start=kmic.dem_start, t0_loa=loa.t0), g, ensure_ascii=False)
    return t_bat_dau, time.time()


def chay(video, thu_muc_ra, ho_so="khong", ghi_de=None, mang="tot", fps_gui=0.0, mic="lenh", hfov_may_chu=None,
         hfov_nguon=69.0, bat_dau_s=0.0, giay=None, seed=0, tu_bat_may_chu=True, host="127.0.0.1", cong=8081,
         nhan=None, cong_udp=12345, nhat_ky_may_chu=None, cho_cuoi_s=3.0, ten_video=None, fps_cam=None,
         t_rot=20.0, dai_rot=3.0, ghi_xem=True, cau_bat=CAU_BAT, cau_tat=CAU_TAT, thu_muc_may_chu=GOC,
         env_may_chu=None, boc_ghi_cau=False) -> dict:
    cap, fps, so_khung = _mo_video(video, bat_dau_s, giay)
    if tu_bat_may_chu and may_chu_dang_chay(host, cong):
        print(f"[ESP32 ẢO] Cổng {cong} đang có máy chủ khác — tắt nó (không phải do ESP32 ảo bật) hoặc dùng "
              f"--may-chu-co-san", flush=True)
        raise SystemExit(2)
    os.makedirs(thu_muc_ra, exist_ok=True)
    if ho_so == "khong":
        chat_luong = int((ghi_de or {}).get("chat_luong", 80))
        bien_anh = lambda bgr, t: bien_anh_khong(bgr, chat_luong)  # noqa: E731
    else:
        cfg = doc_ho_so(ho_so)
        cfg.update(hfov_nguon=hfov_nguon, **(ghi_de or {}))
        bien_anh = HoSoAnh(cfg, seed=seed)
    p = None
    if tu_bat_may_chu:
        nhat_ky_may_chu = os.path.abspath(os.path.join(thu_muc_ra, "nhat_ky_may_chu.jsonl"))
        env = {"AIGLASS_HFOV": hfov_may_chu} if hfov_may_chu else {}
        env.update(env_may_chu or {})
        p = bat_may_chu(nhat_ky_may_chu, env, log=os.path.join(thu_muc_ra, "app_main.log"), thu_muc=thu_muc_may_chu,
                        boc_ghi_cau=boc_ghi_cau)
    try:
        cam = CamBien(fps_cam or fps, rung_ms=0.0 if fps_cam is None else 5.0, seed=seed)
        mang_gia = MangGia(mang, seed=seed, t_rot=t_rot, dai_rot=dai_rot)
        t0, t1 = asyncio.run(_phien(cap, fps, so_khung, thu_muc_ra, bien_anh, host, cong, cong_udp, mic, seed,
                                    cho_cuoi_s, bat_dau_s, cam, mang_gia, fps_gui, ghi_xem, cau_bat, cau_tat))
    finally:
        cap.release()
        if p is not None:
            tat_may_chu(p)
    cau = [c for c in danh_gia._jsonl(nhat_ky_may_chu) if t0 - 1.0 <= c["t"] <= t1] if nhat_ky_may_chu else []
    with open(os.path.join(thu_muc_ra, "cau.jsonl"), "w", encoding="utf-8") as g:
        g.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cau)
    if isinstance(nhan, str):
        with open(nhan, encoding="utf-8") as g:
            nhan = json.load(g)["khung"]
    ten_video = ten_video or os.path.splitext(os.path.basename(video))[0]
    bang = danh_gia.tinh_bang_so(thu_muc_ra, nhan=nhan, ten_video=ten_video)
    with open(os.path.join(thu_muc_ra, "bang_so.json"), "w", encoding="utf-8") as g:
        json.dump(bang, g, ensure_ascii=False, indent=1)
    return bang


def main():
    ap = argparse.ArgumentParser(description="ESP32 ảo — chế độ A: phát lại video vào app_main thật")
    ap.add_argument("video")
    ap.add_argument("--ra", required=True, help="thư mục phiên")
    ap.add_argument("--ho-so", default="khong")
    ap.add_argument("--ghi-de", action="append", default=[], help="khoa=gia_tri ghi đè hồ sơ ảnh (lặp được)")
    ap.add_argument("--mang", default="tot", choices=["tot", "kem", "rot"])
    ap.add_argument("--fps-gui", type=float, default=0.0)
    ap.add_argument("--fps-cam", type=float, default=None, help="nhịp cảm biến (mặc định = fps video nguồn)")
    ap.add_argument("--t-rot", type=float, default=20.0)
    ap.add_argument("--dai-rot", type=float, default=3.0)
    ap.add_argument("--mic", default="lenh", choices=["tat", "lenh", "firmware"])
    ap.add_argument("--hfov-may-chu", type=float, default=None)
    ap.add_argument("--hfov-nguon", type=float, default=69.0)
    ap.add_argument("--bat-dau", type=float, default=0.0)
    ap.add_argument("--giay", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goc-viet", action="store_true",
                    help="chạy bản GỐC ĐÃ VIỆT HÓA (glassai/OpenAIglasses_goc_viet: logic y hệt gốc, chữ tiếng Việt)")
    ap.add_argument("--goc-trung-quoc", action="store_true",
                    help="chạy CODE GỐC Trung Quốc y hệt (bak-zh) nói tiếng Việt (thu_nghiem_do_sau/truc_tiep/goc_trung_quoc.py)")
    ap.add_argument("--nhan", default=None, help="file nhãn (test_cham_diem/nhan_vach_dev.json)")
    ap.add_argument("--may-chu-co-san", action="store_true", help="dùng app_main đang chạy (không có nhật ký câu)")
    a = ap.parse_args()
    ghi_de = {}
    for kv in a.ghi_de:
        k, v = kv.split("=", 1)
        try:
            ghi_de[k] = json.loads(v)
        except json.JSONDecodeError:
            ghi_de[k] = v
    goc = {}
    if a.goc_viet:
        goc = dict(thu_muc_may_chu=os.path.join(os.path.dirname(GOC), "OpenAIglasses_goc_viet"), boc_ghi_cau=True)
    elif a.goc_trung_quoc:
        sys.path.insert(0, os.path.dirname(GOC))
        from thu_nghiem_do_sau.truc_tiep import goc_trung_quoc as G
        ban = G.tao_ban_chay(os.path.join(os.path.dirname(GOC), "goc_trung_quoc_chay"))
        goc = dict(cau_bat=G.CAU_BAT_ZH, cau_tat=G.CAU_TAT_ZH, thu_muc_may_chu=ban,
                   env_may_chu={"VOICE_DIR": G.tao_giong_viet(os.path.join(ban, "giong_viet"))})
    bang = chay(a.video, a.ra, ho_so=a.ho_so, ghi_de=ghi_de or None, mang=a.mang, fps_gui=a.fps_gui, mic=a.mic,
                hfov_may_chu=a.hfov_may_chu, hfov_nguon=a.hfov_nguon, bat_dau_s=a.bat_dau, giay=a.giay,
                seed=a.seed, tu_bat_may_chu=not a.may_chu_co_san, nhan=a.nhan, fps_cam=a.fps_cam,
                t_rot=a.t_rot, dai_rot=a.dai_rot, **goc)
    print(json.dumps(bang, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
