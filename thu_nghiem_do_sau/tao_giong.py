#!/usr/bin/env python3
"""
Tạo các mảnh giọng nói ngắn cho bản thử nghiệm (edge-tts, cần internet), lưu ở thu_nghiem_do_sau/giong/.
Câu cảnh báo được ghép lúc chạy: <tên vật> + <bên nào> + <khoảng cách>, ví dụ "Xe máy" + "bên phải" + "2 mét".
Không đụng tới voice/ của dự án.

Chạy (đứng ở glassai/, đã bật .venv):
  python thu_nghiem_do_sau/tao_giong.py
"""
import asyncio
import os
import subprocess
import tempfile
import time

import edge_tts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'giong')
VOICE = 'vi-VN-HoaiMyNeural'

# tên file -> câu đọc
MANH = {
    # tên vật
    'ten_xe_may': 'Xe máy', 'ten_o_to': 'Ô tô', 'ten_nguoi': 'Người', 'ten_xe_dap': 'Xe đạp',
    'ten_xe_buyt': 'Xe buýt', 'ten_xe_tai': 'Xe tải', 'ten_xe_day': 'Xe đẩy', 'ten_cho': 'Chó',
    'ten_meo': 'Mèo', 'ten_ga_chim': 'Gà', 'ten_dong_vat': 'Con vật', 'ten_ghe': 'Ghế', 'ten_ban': 'Bàn',
    'ten_ghe_dai': 'Ghế dài', 'ten_cot': 'Cột', 'ten_chau_cay': 'Chậu cây', 'ten_gio': 'Giỏ', 'ten_xo': 'Xô',
    'ten_thung': 'Thùng', 'ten_coc': 'Cọc chắn', 'ten_thung_rac': 'Thùng rác', 'ten_ma_no_canh': 'Ma nơ canh',
    'ten_tru_nuoc': 'Trụ nước', 'ten_da': 'Đá', 'ten_goc_cay': 'Gốc cây', 'ten_bien': 'Biển',
    'ten_vat_can': 'Vật cản', 'ten_vat_can_ngang_dau': 'Vật cản ngang đầu',
    # bên nào
    'ben_trai': 'bên trái', 'ben_phai': 'bên phải', 'phia_truoc': 'phía trước',
    # khoảng cách
    'cach_1': 'một mét', 'cach_2': 'hai mét', 'cach_3': 'ba mét', 'cach_4': 'bốn mét', 'cach_5': 'năm mét',
    # lệnh tránh (bản 4)
    'dich_trai': 'Dịch sang trái', 'dich_phai': 'Dịch sang phải', 'dung_lai': 'Dừng lại!', 'dang_toi': 'đang tới', 'ten_cau_thang': 'Cầu thang lên', 'ten_mep_via_he': 'Sắp xuống lòng đường', 'dang_o_duong': 'Đang ở lòng đường, chú ý xe', 'len_via_he': 'Đã lên vỉa hè', 'den_do': 'Đèn đỏ, dừng lại chờ', 'den_xanh': 'Đèn xanh',
    # [Bản 26–27] kính không nhìn rõ; địa hình đi xuống
    'khong_nhin_ro': 'Kính không nhìn rõ, hãy dùng gậy', 'nhin_ro_lai': 'Kính đã nhìn rõ lại',
    'ten_bac_xuong': 'Bậc xuống', 'ten_ho': 'Hố',
    # thử kính trực tiếp: điện thoại tự phát khi mất kết nối / trang bị tắt
    'mat_ket_noi': 'Mất kết nối, hãy dùng gậy', 'trang_bi_tat': 'Trang bị tắt, hãy dùng gậy',
    'da_ket_noi': 'Đã kết nối',
    # vật chắn trên vạch (spec 2026-10-27-vong-vat-can-tren-vach)
    'chan_vach': 'chắn vạch', 'vong_ben_trai': 'vòng bên trái', 'vong_ben_phai': 'vòng bên phải',
    'chan_vach_ben_phai': 'chắn vạch bên phải', 'chan_vach_ben_trai': 'chắn vạch bên trái', 'chan_giua_vach': 'chắn giữa vạch',
    'di_sang_trai': 'Đi sang trái', 'di_sang_phai': 'Đi sang phải',
    'do_gay': 'dò gậy', 'dung_gay_do_vong': 'Dùng gậy dò đường vòng',
    'vach_bi_chan': 'Vạch bị chắn, dừng lại, hãy dùng gậy', 'da_ve_vach': 'Đã về vạch',
    'vach_ben_phai': 'Vạch ở bên phải', 'vach_ben_trai': 'Vạch ở bên trái',
    'chua_thay_lai_vach': 'Chưa thấy lại vạch, hãy dùng gậy',
}


async def synth(text, path):
    await edge_tts.Communicate(text, VOICE, rate='+15%').save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = skip = fail = 0
    for key, text in MANH.items():
        dst = os.path.join(OUT, key + '.wav')
        if os.path.exists(dst):
            skip += 1
            continue
        with tempfile.TemporaryDirectory() as td:
            mp3 = os.path.join(td, 'a.mp3')
            for attempt in range(3):
                try:
                    asyncio.run(synth(text, mp3))
                    # mono 16 kHz, cắt bớt khoảng lặng đầu/cuối cho câu ghép gọn
                    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3, '-af',
                                    'silenceremove=start_periods=1:start_threshold=-45dB,areverse,'
                                    'silenceremove=start_periods=1:start_threshold=-45dB,areverse',
                                    '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', dst], check=True)
                    ok += 1
                    break
                except Exception as e:  # noqa: BLE001
                    time.sleep(1 + attempt)
                    if attempt == 2:
                        fail += 1
                        print('LỖI', key, e)
    print(f"Tạo {ok}, đã có {skip}, lỗi {fail} -> {OUT}")


if __name__ == '__main__':
    main()
