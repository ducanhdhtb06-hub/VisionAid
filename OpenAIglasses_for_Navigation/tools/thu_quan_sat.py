# -*- coding: utf-8 -*-
"""Thử module quan sát (quan_sat.py) với API thật, không cần kính.

  python tools/thu_quan_sat.py anh.jpg
  python tools/thu_quan_sat.py video.MOV                            # tự nghe câu hỏi trong video (PhoWhisper trên máy)
  python tools/thu_quan_sat.py video.MOV --giay 98.6 --hoi "Cái gì chắn đường?"   # tự đặt câu hỏi và mốc giây
  python tools/thu_quan_sat.py video.MOV --giay 10 20 30 --doc      # nhiều mốc, có tạo tiếng (.wav)
  python tools/thu_quan_sat.py video.MOV --xuat ket_qua.mp4         # xuất video có phụ đề + tiếng kính trả lời

Không có --hoi: công cụ tìm các đoạn có người nói trong video, chép thành chữ bằng PhoWhisper (chạy trên máy, như kính
thật nghe câu hỏi), câu nào là câu quan sát ("phía trước có gì", "chữ này là gì"...) thì lấy khung hình ngay sau đó để hỏi.

Làm nóng trước (như máy chủ lúc khởi động), in câu trả lời và thời gian (hỏi -> tiếng đầu); --doc lưu tiếng đọc vào test_out/quan_sat/ để nghe.
--xuat (tự bật --doc): ghép video gốc + phụ đề "Bạn hỏi" / "Kính nghe được" + tiếng "tít" + giọng kính trả lời. Mốc thời gian
là số đo thật của lượt chạy: nghe xong -> chép chữ (PhoWhisper) -> nhìn ảnh -> tiếng đầu tiên.
Cần GEMINI_API_KEY trong .env (chỉ thử nhánh Gemini; nhánh Qwen chạy qua app_main).
"""
import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import wave

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quan_sat as Q  # noqa: E402


def anh_jpeg(duong, giay):
    """Ảnh (jpg/png) hoặc một khung của video tại `giay` -> JPEG 640 px như kính gửi."""
    if giay is None:
        img = cv2.imread(duong)
    else:
        cap = cv2.VideoCapture(duong)
        cap.set(cv2.CAP_PROP_POS_MSEC, giay * 1000)
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {duong} {'' if giay is None else f'@{giay}s'}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (640, round(h * 640 / w)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def nghe_cau_hoi(video):
    """Âm thanh của video -> [(bắt đầu, kết thúc, chữ, giây chép chữ)] cho từng đoạn có người nói (PhoWhisper trên máy)."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    doan = Q.cat_doan_noi(a, 16000)
    if not doan:
        return []
    import torch
    from transformers import pipeline
    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=os.getenv("AIGLASS_ASR_MODEL", "vinai/PhoWhisper-small"),
                   device=0 if cuda else -1, dtype=torch.float16 if cuda else torch.float32)
    kq = []
    for t0, t1 in doan:
        x = a[int(max(t0 - 0.2, 0) * 16000):int((t1 + 0.2) * 16000)]
        t = time.time()
        chu = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "vi"})["text"].strip()
        kq.append((t0, t1, chu, time.time() - t))
    return kq


def lich_phu_de(m):
    """Mốc thời gian (giây trên video) của một lượt hỏi-đáp từ các số đo thật:
    m = {t0: bắt đầu nói (None nếu câu hỏi gõ tay), t1: nói xong / mốc khung hình, t_asr: giây chép chữ,
         t_dau: giây từ lúc có chữ tới tiếng đầu tiên (nhìn ảnh + đọc), giay_tieng: độ dài tiếng trả lời}."""
    tit = round(m["t1"] + m["t_asr"], 2)                           # nghe xong + chép chữ xong: kêu "tít"
    noi = round(tit + m["t_dau"], 2)                               # tiếng đầu tiên của kính
    t0 = m["t0"] if m["t0"] is not None else max(m["t1"] - 1.5, 0.0)
    return dict(tit=tit, noi=noi, het=round(noi + m["giay_tieng"], 2), hoi=(round(t0, 2), noi), nghe=(tit, noi))


def _font():
    return subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans:bold"], capture_output=True, text=True).stdout


def xuat_video(nguon, muc, ra):
    """Ghép video gốc + phụ đề + "tít" + giọng kính trả lời -> `ra` (mp4 1280x720).
    muc = [{hoi, chu_nghe (None nếu gõ tay), cau, pcm (PCM16 24 kHz), **lich_phu_de}]."""
    font = _font()
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", nguon]))
    end = round(max(dur, max(m["het"] for m in muc)) + 1.2, 2)
    co_am = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", nguon],
                                capture_output=True, text=True).stdout.strip())
    tmp = tempfile.mkdtemp(prefix="quan_sat_")

    def viet(ten, nd):
        f = os.path.join(tmp, ten)
        open(f, "w", encoding="utf-8").write(nd)
        return f

    def wav(ten, pcm, sr):
        f = os.path.join(tmp, ten)
        with wave.open(f, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        return f
    box = "box=1:boxcolor=black@0.65:boxborderw=10"
    vf = [f"scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,fps=30,"
          f"tpad=stop_mode=clone:stop_duration={round(end, 1)},trim=0:{end},setpts=PTS-STARTPTS"]
    for i, m in enumerate(muc):
        hoi_t, nghe_t = m["hoi"], m["nghe"]
        sau = muc[i + 1]["hoi"][0] if i + 1 < len(muc) else end            # phụ đề trả lời hiện tới lúc câu hỏi kế tiếp
        tl = "\n".join(["Kính: " + d if k == 0 else d for k, d in enumerate(textwrap.wrap(m["cau"], 62))])
        vf.append(f"drawtext=fontfile={font}:textfile={viet(f'hoi{i}.txt', 'Bạn hỏi: ' + chr(34) + m['hoi_chu'] + chr(34))}:"
                  f"x=(w-tw)/2:y=h-130:fontsize=32:fontcolor=white:{box}:enable='between(t,{hoi_t[0]},{hoi_t[1]})'")
        if m["chu_nghe"] is not None:
            vf.append(f"drawtext=fontfile={font}:textfile={viet(f'nghe{i}.txt', 'Kính tự nghe được: ' + chr(34) + m['chu_nghe'] + chr(34))}:"
                      f"x=(w-tw)/2:y=h-75:fontsize=22:fontcolor=yellow:{box}:enable='between(t,{nghe_t[0]},{nghe_t[1]})'")
        vf.append(f"drawtext=fontfile={font}:textfile={viet(f'tit{i}.txt', 'tít… kính đang nhìn ảnh (' + str(round(m['noi'] - m['tit'], 1)).replace('.', ',') + ' giây)')}:"
                  f"x=(w-tw)/2:y=40:fontsize=28:fontcolor=yellow:{box}:enable='between(t,{nghe_t[0]},{nghe_t[1]})'")
        vf.append(f"drawtext=fontfile={font}:textfile={viet(f'tl{i}.txt', tl)}:x=(w-tw)/2:y=h-60-th:fontsize=28:fontcolor=white:"
                  f"line_spacing=10:{box}:enable='between(t,{m['noi']},{sau})'")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", nguon]
    am = [f"[0:a]volume=2.2,apad,atrim=0:{end}[a0]"] if co_am else [f"anullsrc=r=48000:cl=mono,atrim=0:{end}[a0]"]
    ten_am = ["[a0]"]
    n = 1
    for i, m in enumerate(muc):
        cmd += ["-i", wav(f"tit{i}.wav", Q.tieng_tit(), 8000), "-i", wav(f"tl{i}.wav", m["pcm"], 24000)]
        for f, t, v in ((n, m["tit"], 1.5), (n + 1, m["noi"], 1.6)):
            ms = int(t * 1000)
            am.append(f"[{f}:a]aresample=48000,volume={v},adelay={ms}|{ms},apad,atrim=0:{end}[a{f}]")
            ten_am.append(f"[a{f}]")
        n += 2
    fc = f"[0:v]{','.join(vf)},format=yuv420p[v];" + ";".join(am) + f";{''.join(ten_am)}amix=inputs={len(ten_am)}:normalize=0[a]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "25", "-preset", "veryfast",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "128k", ra]
    subprocess.run(cmd, check=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nguon", help="ảnh hoặc video")
    ap.add_argument("--giay", type=float, nargs="*", help="mốc giây trong video (bỏ trống nếu là ảnh)")
    ap.add_argument("--hoi", default=None, help="câu hỏi; bỏ trống = tự nghe câu hỏi trong video")
    ap.add_argument("--doc", action="store_true", help="tạo tiếng đọc bằng Gemini TTS, lưu .wav")
    ap.add_argument("--xuat", metavar="KET_QUA.mp4", help="xuất video có phụ đề và tiếng kính trả lời (tự bật --doc)")
    a = ap.parse_args()
    if a.xuat:
        a.doc = True
        if a.nguon.lower().endswith((".jpg", ".jpeg", ".png")):
            raise SystemExit("--xuat cần video, không phải ảnh.")
    load_dotenv()
    os.makedirs("test_out/quan_sat", exist_ok=True)
    print(f"model nhìn: {Q.GEMINI_MODEL} | model đọc: {Q.GEMINI_TTS_MODEL}")
    viec = [dict(g=g, hoi=a.hoi, t0=None, t1=g or 0.0, t_asr=0.0, chu_nghe=None) for g in (a.giay or [None])] if a.hoi else []
    if not a.hoi:
        for t0, t1, chu, t_asr in nghe_cau_hoi(a.nguon):
            la = Q.la_cau_quan_sat(chu)
            print(f"nghe được [{t0:.1f}–{t1:.1f}s]: \"{chu}\"  ->  {'câu quan sát' if la else 'bỏ qua (không phải câu quan sát)'}")
            if la:                                           # khung hình ngay sau khi người dùng hỏi xong
                viec.append(dict(g=t1 + 0.3, hoi=chu, t0=t0, t1=t1, t_asr=t_asr, chu_nghe=chu))
        if not viec:
            raise SystemExit("Không nghe thấy câu quan sát nào trong video. Dùng --hoi và --giay để tự đặt.")
    await Q.lam_nong()
    muc = []
    for v in viec:
        g, hoi = v["g"], v["hoi"]
        t = time.time()
        cau = await Q.mo_ta_gemini(anh_jpeg(a.nguon, g), hoi)
        t_cau = time.time() - t
        nhan = os.path.basename(a.nguon) + ("" if g is None else f" @{g:g}s")
        print(f"\n[{nhan}] {t_cau:.1f}s: {cau}")
        if a.doc and cau:
            pcm, t_dau = b"", None
            async for manh in Q.doc_gemini_dan(cau):
                t_dau = t_dau or time.time() - t
                pcm += manh
            ra = f"test_out/quan_sat/{os.path.splitext(os.path.basename(a.nguon))[0]}_{g or 0:g}.wav"
            with wave.open(ra, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(pcm)
            print(f"   hỏi -> tiếng đầu {t_dau:.1f}s | {len(pcm) / 48000:.1f}s tiếng: {ra}")
            if a.xuat and pcm:
                muc.append({**v, "hoi_chu": hoi, "cau": cau, "pcm": pcm,
                            **lich_phu_de(dict(t0=v["t0"], t1=v["t1"], t_asr=v["t_asr"], t_dau=t_dau, giay_tieng=len(pcm) / 48000))})
    if a.xuat:
        if not muc:
            raise SystemExit("Không có câu trả lời nào có tiếng để ghép video (hết lượt đọc của Gemini?).")
        xuat_video(a.nguon, muc, a.xuat)
        print(f"\nĐã xuất video: {a.xuat}")

if __name__ == "__main__":
    asyncio.run(main())
