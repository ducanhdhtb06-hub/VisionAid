#!/usr/bin/env python3
"""
Hậu kỳ một phiên thử kính trực tiếp (thư mục do chay.py tạo):
  - co_tieng.mp4: khung máy đã nhận + đúng những câu người dùng THỰC SỰ nghe (phản hồi 'da_phat'), đúng lúc phát;
  - thống kê: số khung, % khung bỏ qua (mạng dồn), độ trễ chụp -> bắt đầu nói (đo trên iPhone), câu phát/bỏ/cắt/thay.
File "chỉ ghi" từ iPhone: --chi-ghi đổi sang mp4 640x360, 10 khung/giây để chạy ket_hop_3_model.py.

Chạy (đứng ở glassai/):
  .venv/bin/python loi_do_sau/truc_tiep/hau_ky.py loi_do_sau/truc_tiep/phien/<giờ>
  .venv/bin/python loi_do_sau/truc_tiep/hau_ky.py --chi-ghi ~/Downloads/kinh_....mp4
"""
import argparse
import json
import os
import subprocess
import tempfile
import wave

import cv2
import numpy as np

SR = 16000


def _doc_log(thu_muc):
    """log.json nếu có; máy tắt ngang (không có log.json) thì dựng lại từ log.jsonl ghi dần."""
    p = os.path.join(thu_muc, 'log.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    log, theo_id = dict(bat_dau=None, khung=[], phan_hoi=[]), {}
    with open(os.path.join(thu_muc, 'log.jsonl'), encoding='utf-8') as f:
        for dong in f:
            try:
                d = json.loads(dong)
            except json.JSONDecodeError:          # dòng cuối có thể bị cắt dở khi máy tắt
                continue
            loai = d.pop('loai', None)
            if loai == 'bat_dau':
                log['bat_dau'] = d['bat_dau']
            elif loai == 'nhan':
                theo_id[d['id']] = d
                log['khung'].append(d)
            elif loai == 'xong' and d['id'] in theo_id:
                theo_id[d['id']].update(ms_xu_ly=d['ms_xu_ly'], cau=d['cau'], bo_qua=False)
            elif loai == 'phan_hoi':
                log['phan_hoi'].append(d)
    return log


def thong_ke(thu_muc: str) -> dict:
    log = _doc_log(thu_muc)
    khung, ph = log['khung'], log['phan_hoi']
    dem = {k: sum(p['phan_hoi'] == k for p in ph) for k in ('da_phat', 'bo', 'bi_cat', 'thay')}
    tre = [p['tre_ms'] for p in ph if p['phan_hoi'] == 'da_phat']
    return dict(so_khung=len(khung), ty_le_bo_qua=round(sum(k['bo_qua'] for k in khung) / max(1, len(khung)), 3),
                tre_trung_vi=float(np.median(tre)) if tre else None,
                tre_p90=float(np.percentile(tre, 90, method='higher')) if tre else None, **dem)


def _dung_video_theo_gio_chup(thu_muc, log, out):
    """khung.mp4 ghi mỗi khung đã nhận đúng 1 lần (4G chậm thì < 10 khung/giây thật). Dựng lại 10 khung/giây theo
    giờ chụp: mỗi mốc 0,1 s hiện khung mới nhất đã chụp tới lúc đó -> video dài đúng thời gian thật, khớp tiếng."""
    t0 = log['khung'][0]['t_chup']
    moc = [(k['t_chup'] - t0) / 1000 for k in log['khung']]
    cap = cv2.VideoCapture(os.path.join(thu_muc, 'khung.mp4'))
    wr = cv2.VideoWriter(out, cv2.VideoWriter.fourcc(*'mp4v'), 10, (640, 360))
    hien, i = None, 0
    for n in range(int(moc[-1] * 10) + 1):
        while i < len(moc) and moc[i] <= n / 10 + 1e-6:
            ok, fr = cap.read()
            if not ok:
                break
            hien, i = fr, i + 1
        if hien is not None:
            wr.write(hien)
    wr.release()
    return moc[-1]


def ghep_tieng(thu_muc: str) -> str:
    """Trục thời gian = giờ chụp (đồng hồ iPhone) của khung đầu; tiếng đặt ở giờ chụp + độ trễ iPhone đo."""
    log = _doc_log(thu_muc)
    t_chup = {k['id']: k['t_chup'] for k in log['khung']}
    t0 = log['khung'][0]['t_chup'] if log['khung'] else 0.0
    with tempfile.TemporaryDirectory() as td:
        hinh = os.path.join(td, 'hinh.mp4')
        dai = _dung_video_theo_gio_chup(thu_muc, log, hinh) + 0.1
        return _ghep(thu_muc, log, t_chup, t0, dai, hinh, td)


def _ghep(thu_muc, log, t_chup, t0, dai, hinh, td):
    track = np.zeros(int(dai * SR), np.int32)
    for p in log['phan_hoi']:
        wav = os.path.join(thu_muc, 'cau', f"{p['id']}.wav")
        if p['phan_hoi'] != 'da_phat' or p['id'] not in t_chup or not os.path.exists(wav):
            continue
        with wave.open(wav) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        i = int(((t_chup[p['id']] - t0) + p['tre_ms']) / 1000 * SR)
        track[i:i + len(a)] += a[:max(0, len(track) - i)]
    track = np.clip(track, -32768, 32767).astype(np.int16)
    out = os.path.join(thu_muc, 'co_tieng.mp4')
    tw = os.path.join(td, 'tieng.wav')
    with wave.open(tw, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(track.tobytes())
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', hinh, '-i', tw,
                    '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '64k',
                    '-shortest', out], check=True)
    return out


def doi_chi_ghi(file: str) -> str:
    out = os.path.splitext(file)[0] + '_10fps.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', file, '-vf', 'scale=640:360,fps=10', '-an',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('duong_dan', help='thư mục phiên, hoặc file video khi dùng --chi-ghi')
    ap.add_argument('--chi-ghi', action='store_true', help='đổi file "chỉ ghi" của iPhone sang mp4 10 khung/giây')
    args = ap.parse_args()
    if args.chi_ghi:
        out = doi_chi_ghi(args.duong_dan)
        print(f'Đã đổi: {out}\nChạy: .venv/bin/python loi_do_sau/ket_hop_3_model.py {out}')
        return
    tk = thong_ke(args.duong_dan)
    out = ghep_tieng(args.duong_dan)
    print(f"Khung đã nhận: {tk['so_khung']} | bỏ qua do mạng dồn: {tk['ty_le_bo_qua']:.0%}")
    print(f"Độ trễ chụp -> bắt đầu nói (iPhone đo): trung vị {tk['tre_trung_vi']} ms, p90 {tk['tre_p90']} ms")
    print(f"Câu: phát {tk['da_phat']} | bỏ vì trễ {tk['bo']} | bị 'Dừng lại' cắt {tk['bi_cat']} | bị câu mới thay {tk['thay']}")
    print(f'Video có tiếng (đúng những gì người dùng nghe): {out}')
    print(f"Chấm lại offline: .venv/bin/python loi_do_sau/ket_hop_3_model.py {os.path.join(args.duong_dan, 'khung.mp4')}")


if __name__ == '__main__':
    main()
