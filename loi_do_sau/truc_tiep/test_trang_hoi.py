"""Luật phát tiếng của trang khi có câu trả lời nút Hỏi — chạy hàm thật nhanWav/phat của kinh_loi.js bằng node."""
import json
import os
import re
import shutil
import subprocess

import pytest

JS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'kinh_loi.js'), encoding='utf-8').read()


def _ham(ten):
    return re.search(r'(?:async )?function ' + ten + r'\(.*?\n\}\n', JS, re.S).group(0)


def _chay(kich_ban):
    node = shutil.which('node')
    if node is None:
        pytest.skip('không có node')
    js = """
const TRE_BO_MS = 1500; let jsonCho = null, dangPhat = null, cauCho = null, now = 100000;
let dangNoiTaiCho = false, luotTaiCho = 0, dangIm = false;
const speechSynthesis = { cancel() { daPhat.push('HUY_TAI_CHO'); } };
const Date = { now: () => now }; const phanHoi = [], daPhat = [];
const $ = () => ({ set textContent(v) {} });
const ac = { decodeAudioData: async b => b };
let nguonCuoi = null;
function phatBuffer(buf, onXong) { const n = { buf, onended: onXong, stop() { this.dung = true; } }; nguonCuoi = n;
  daPhat.push(buf); return n; }
function guiPhanHoi(ph, id) { phanHoi.push([ph, id]); }
function xong() { const n = dangPhat.nguon; n.onended && n.onended(); }
async function toi(j, buf) { jsonCho = j; await nhanWav(buf); }
""" + _ham('nhanWav') + _ham('phat') + _ham('catTaiCho') + _ham('xongTaiCho') + _ham('batDauIm') + \
        _ham('hetIm') + kich_ban + "\nchay().then(() => console.log(JSON.stringify({ daPhat, phanHoi })));"
    return json.loads(subprocess.run([node, '-e', js], capture_output=True, text=True, check=True).stdout)


def test_cau_tra_loi_tre_van_duoc_doc():
    kq = _chay("""async function chay() {
      await toi({ id: 1, t_chup: now - 4000, cau: 'Phía trước có ghế.', hoi: 'tra_loi' }, 'TL');
      await toi({ id: 2, t_chup: now - 4000, cau: 'Vạch ở bên phải.' }, 'CU'); }""")
    assert kq['daPhat'] == ['TL']                       # câu trả lời đọc dù tới sau 4 s
    assert ['bo', 2] in kq['phanHoi']                   # câu dẫn đường cũ vẫn bỏ như trước


def test_cau_tra_loi_dang_cho_khong_bi_cau_thuong_thay_cho():
    kq = _chay("""async function chay() {
      await toi({ id: 1, t_chup: now, cau: 'Vạch ở bên phải.' }, 'V1');
      await toi({ id: 2, t_chup: now, cau: 'Phía trước có ghế.', hoi: 'tra_loi' }, 'TL');
      await toi({ id: 3, t_chup: now, cau: 'Đã thấy vạch dẫn đường.' }, 'V3');
      now += 3000; xong(); }""")
    assert kq['daPhat'] == ['V1', 'TL']                 # trả lời phát sau câu đang nói, dù đã chờ 3 s
    assert ['thay', 3] in kq['phanHoi']


def test_dung_lai_van_cat_ngang_cau_tra_loi():
    kq = _chay("""async function chay() {
      await toi({ id: 1, t_chup: now, cau: 'Phía trước có ghế.', hoi: 'tra_loi' }, 'TL');
      await toi({ id: 2, t_chup: now, cau: 'Dừng lại. Xe máy, phía trước, đang tới' }, 'DUNG'); }""")
    assert kq['daPhat'] == ['TL', 'DUNG'] and ['bi_cat', 1] in kq['phanHoi']


# ---------- câu trả lời điện thoại tự đọc (Gemini đọc lỗi): giữ luật một người nói ----------
def test_dien_thoai_dang_tu_doc_thi_cau_thuong_cho_doc_xong():
    kq = _chay("""async function chay() {
      dangNoiTaiCho = true; luotTaiCho = 1;
      await toi({ id: 1, t_chup: now, cau: 'Vạch ở bên phải.' }, 'V1');
      daPhat.push('|'); xongTaiCho(1); }""")
    assert kq['daPhat'] == ['|', 'V1']                  # không nói chồng lên câu trả lời; đọc xong mới phát


def test_dung_lai_cat_ngang_dien_thoai_dang_tu_doc():
    kq = _chay("""async function chay() {
      dangNoiTaiCho = true; luotTaiCho = 1;
      await toi({ id: 1, t_chup: now, cau: 'Dừng lại. Xe máy, phía trước, đang tới' }, 'DUNG'); }""")
    assert kq['daPhat'] == ['HUY_TAI_CHO', 'DUNG']


def test_bao_xong_cua_luot_doc_cu_khong_lam_xong_luot_moi():
    # lưới an toàn hẹn giờ của câu trả lời trước tới muộn: không được thả câu dẫn đường chen vào câu đang đọc
    kq = _chay("""async function chay() {
      dangNoiTaiCho = true; luotTaiCho = 2;
      await toi({ id: 1, t_chup: now, cau: 'Vạch ở bên phải.' }, 'V1');
      xongTaiCho(1); }""")
    assert kq['daPhat'] == []


def _chay_tai_cho(kich_ban):
    """Chạy docTraLoiTaiCho / moKhoaGiong thật với speechSynthesis giả, trả các tin trang báo về máy chủ."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('không có node')
    js = """
const TRE_BO_MS = 1500; let dangPhat = null, cauCho = null, dangNoiTaiCho = false, luotTaiCho = 0;
const Date = { now: () => 0 }; const bao = [], noi = [];
const $ = () => ({ set textContent(v) {} });
const ws = { readyState: 1, send: s => bao.push(JSON.parse(s)) };
let cuoi = null;
class SpeechSynthesisUtterance { constructor(t) { this.text = t; cuoi = this; } }
const speechSynthesis = { speak(u) { noi.push(u.text); }, cancel() {},
  getVoices: () => [{ name: 'Linh', lang: 'vi-VN' }, { name: 'Samantha', lang: 'en-US' }] };
function setTimeout() {}
function phat() {} function guiPhanHoi() {}
""" + _ham('catTaiCho') + _ham('xongTaiCho') + _ham('baoTaiCho') + _ham('moKhoaGiong') + _ham('docTraLoiTaiCho') + \
        kich_ban + "\nconsole.log(JSON.stringify({ bao, noi }));"
    return json.loads(subprocess.run([node, '-e', js], capture_output=True, text=True, check=True).stdout)


def test_doc_tai_cho_bao_ve_may_chu_de_biet_vi_sao_im():
    # 2026-11-10 iPhone thật không ra tiếng câu trả lời: phải biết iOS có bắt đầu đọc không, có giọng tiếng Việt không
    kq = _chay_tai_cho("docTraLoiTaiCho('Phía trước có ghế.'); cuoi.onstart(); cuoi.onend();")
    su_kien = [b['su_kien'] for b in kq['bao']]
    assert su_kien == ['goi', 'bat_dau', 'xong'] and all(b['phan_hoi'] == 'tai_cho' for b in kq['bao'])
    assert kq['bao'][0]['giong_viet'] == ['Linh']
    assert kq['noi'] == ['Phía trước có ghế.']


def test_doc_tai_cho_loi_thi_bao_ma_loi():
    kq = _chay_tai_cho("docTraLoiTaiCho('Có.'); cuoi.onerror({ error: 'not-allowed' });")
    assert kq['bao'][-1]['su_kien'] == 'loi' and kq['bao'][-1]['chi_tiet'] == 'not-allowed'


def test_mo_khoa_giong_doc_mot_cau_rong():
    # iOS chỉ cho trang đọc sau khi đã đọc một lần trong lúc người dùng chạm (nút Bắt đầu / thả nút Hỏi)
    kq = _chay_tai_cho("moKhoaGiong();")
    assert kq['noi'] == ['']


# ---------- đang giữ nút Hỏi: kính im (tiếng kính lọt vào micro làm câu hỏi bị nghe sai) ----------
def test_bam_giu_hoi_cat_cau_dang_noi():
    kq = _chay("""async function chay() {
      await toi({ id: 1, t_chup: now, cau: 'Vạch ở bên phải.' }, 'V1');
      batDauIm(); }""")
    assert kq['daPhat'] == ['V1', 'HUY_TAI_CHO'] and ['bi_cat', 1] in kq['phanHoi']


def test_dang_giu_hoi_khong_phat_ca_dung_lai_tha_tay_moi_phat_neu_con_moi():
    kq = _chay("""async function chay() {
      batDauIm();
      await toi({ id: 1, t_chup: now, cau: 'Vạch ở bên phải.' }, 'V1');
      await toi({ id: 2, t_chup: now, cau: 'Dừng lại. Xe máy, phía trước, đang tới' }, 'DUNG');
      daPhat.push('|'); now += 500; hetIm(); }""")
    assert kq['daPhat'] == ['HUY_TAI_CHO', '|', 'DUNG']   # im suốt lúc giữ; thả tay: câu mới nhất, còn mới -> phát
    assert ['thay', 1] in kq['phanHoi']


def test_tha_tay_cau_giu_lai_da_cu_thi_bo_de_may_chu_noi_lai():
    kq = _chay("""async function chay() {
      batDauIm();
      await toi({ id: 1, t_chup: now, cau: 'Vạch ở bên phải.' }, 'V1');
      now += 4000; hetIm(); }""")
    assert 'V1' not in kq['daPhat'] and ['bo', 1] in kq['phanHoi']
