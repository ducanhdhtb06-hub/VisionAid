# ne_vat_can_do_sau.py
# -*- coding: utf-8 -*-
"""
Module né vật cản mới (THỬ NGHIỆM; mặc định bật trong workflow_blindpath, AIGLASS_OBS_ENGINE=cu để tắt).

Ghép: Depth Anything V2 Small (độ sâu, Apache-2.0) + D-FINE (nhận diện, Apache-2.0)
+ UperNet-ConvNeXt-tiny (lòng đường/vỉa hè, MIT)
-> câu nói kiểu "Dịch sang trái. Xe máy", "Xe máy, bên trái, đang tới", "Cầu thang lên, phía trước",
   "Sắp xuống lòng đường, phía trước"; đang ở lòng đường thì không bao giờ bảo "dịch".
Không đọc số mét (đo khoảng cách còn sai ~10%; người dùng 28/10). AIGLASS_NOI_KHOANG_CACH=1: đọc lại như cũ.
Ảnh tối / camera bị che >= 1,5 s -> "Kính không nhìn rõ, hãy dùng gậy" (không im lặng khi kính mù).
AIGLASS_NE_BAC_XUONG=1: thử báo bậc xuống / hố (chưa đủ dữ liệu kiểm tra, mặc định tắt).
Không dùng Ultralytics (AGPL).

Logic lõi nằm ở thu_nghiem_do_sau/ket_hop_3_model.py (một nguồn duy nhất, đã chấm điểm trên 9 video);
module này chỉ chạy logic đó cho từng khung hình thời gian thực và đăng ký câu nói ghép vào audio_player.
Đường dẫn thư mục lõi: biến môi trường AIGLASS_NE_DIR (mặc định ../thu_nghiem_do_sau).

CHƯA kiểm tra trên kính thật (ESP32); số liệu chỉ từ video điện thoại. CHƯA an toàn cho người dùng thật.
"""
import collections
import hashlib
import logging
import os
import sys
import time
import wave

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_NE_DIR = os.getenv("AIGLASS_NE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "thu_nghiem_do_sau"))
if _NE_DIR not in sys.path:
    sys.path.insert(0, _NE_DIR)
import ket_hop_3_model as K  # noqa: E402

K.H_CAM = float(os.getenv("AIGLASS_NE_HCAM", K.H_CAM))  # chiều cao camera kính (m), lấy từ hieu_chinh_khoang_cach.py

_GHEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice", "vi", "_ne_ghep")


def _dang_ky_cau_noi(text, pcm16):
    """Ghi câu ghép ra file wav (16 kHz) và đăng ký vào audio_player.AUDIO_MAP để play_voice_text phát được."""
    try:
        import audio_player
    except Exception as e:  # chạy độc lập (không có hệ thống âm thanh dự án)
        logger.debug(f"[NE] Không nạp được audio_player: {e}")
        return
    os.makedirs(_GHEP_DIR, exist_ok=True)
    path = os.path.join(_GHEP_DIR, hashlib.md5(text.encode("utf-8")).hexdigest()[:16] + ".wav")
    if not os.path.exists(path):
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(K.SR)
            w.writeframes(pcm16.tobytes())
    import unicodedata
    key = unicodedata.normalize("NFC", text)
    if key not in audio_player.AUDIO_MAP:
        audio_player.AUDIO_MAP[key] = path
        audio_player.load_wav_file(path)


def cau_vat(act, ten, ben, toi, d=None, ngan=False):
    """Chữ của câu module vật cản (khớp mảnh âm thanh Speaker ghép): d=None -> không đọc số mét; ngan -> chỉ lệnh
    (Speaker.ngan: xe vừa được báo "đang tới")."""
    if ngan:
        return f"{K.ACT_VI[act]}."
    met = f", {d} mét" if d is not None else ""
    if act and toi:      # xe đang lao tới: nói rõ "đang tới" (lỗi thật IMG_7354 353 s: "Dừng lại. Xe máy" về xe KHÁC
        return f"{K.ACT_VI[act]}. {ten.capitalize()}, {ben}, đang tới"   # nghe như mâu thuẫn câu "Đi sang trái")
    if act:
        return f"{K.ACT_VI[act]}. {ten.capitalize()}{met}"
    if toi:
        return f"{ten.capitalize()}, {ben}, đang tới"
    return f"{ten.capitalize()}, {ben}{met}"


def lenh_sau_loc(muc, act):
    """muc = kết quả bộ lọc chi_bao của vật sẽ nói: "tin" = chỉ báo thông tin (xe đỗ trước mặt khi đang theo vạch mà
    bộ điều khiển vật chắn chưa nhận) -> không ra lệnh "Dừng lại" / "Dịch" (một người nói, đợt sửa 28/10)."""
    return None if muc == "tin" else act


def du_khung(streak, can=2):
    """Đủ cửa sổ và thấy trong >= can khung. Mép vỉa hè sát ngưỡng chỉ thấy 2/3 khung liền tùy pha khung gửi
    -> bỏ sót "Sắp xuống lòng đường" (chạy lặp 28/10); 2/3 vẫn lọc được ảo giác 1 khung."""
    return len(streak) == streak.maxlen and sum(streak) >= can


def o_long_duong(dang_o_duong, tren_vach):
    """Đang ở lòng đường, trừ khi đo được rõ người đứng trên vạch dẫn đường (vạch lát trên vỉa hè; IMG_7354 65,6 s,
    29/10: đứng trên vạch mà báo "Đang ở lòng đường"). Bước xuống đường khi đi vòng thì không còn trên vạch: vẫn báo."""
    return bool(dang_o_duong) and not tren_vach


def toc_do_gan(hist, giay=0.5):
    """Tốc độ tiến lại (m/s) trong `giay` giây gần nhất của một vết [(t, khoảng cách)], None nếu vết chưa dài đủ."""
    if not hist:
        return None
    t1, z1 = hist[-1]
    cu = [(t, z) for t, z in hist if t1 - t >= giay]
    if not cu:
        return None
    t0, z0 = cu[-1]
    return (z0 - z1) / (t1 - t0)


def chieu_cao_anh(mask):
    """(số hàng ảnh vật chiếm, có chạm mép trên/dưới ảnh không) - chạm mép thì chiều cao không còn đúng tỉ lệ."""
    hang = np.flatnonzero(np.asarray(mask).any(axis=1))
    if not hang.size:
        return 0, True
    return int(hang[-1] - hang[0] + 1), bool(hang[0] <= 1 or hang[-1] >= np.asarray(mask).shape[0] - 2)


def kich_thuoc_hop_ly(vet, giay=0.5):
    """vet [(t, khoảng cách, chiều cao ảnh, chạm mép)]: vật tới gần thật thì trên ảnh to ra theo tỉ lệ khoảng cách
    (máy ảnh lỗ kim: cao_ảnh x khoảng cách ~ không đổi). Trong `giay` giây gần nhất độ sâu báo gần lại g lần mà ảnh to ra
    chưa được nửa phần đó -> độ sâu nhảy (vật mỏng/xa đo sai), không phải đang tới. Không biết (vết ngắn, chạm mép) -> True."""
    if not vet:
        return True
    t1, z1, h1, c1 = vet[-1]
    cu = [x for x in vet if t1 - x[0] >= giay]
    if not cu or c1:
        return True
    _, z0, h0, c0 = cu[-1]
    if c0 or h0 <= 0 or z1 <= 0 or z0 <= z1:
        return True
    return h1 / h0 - 1.0 >= 0.5 * (z0 / z1 - 1.0)


def dang_toi(o):
    """Xe đang tiến lại >= V_TOI: trung vị cả vết VÀ nửa giây gần nhất (khi biết). Hai xe đỗ liền nhau bị ghép thành
    một vết (khoảng cách nhảy 3,8 -> 2,0 m) làm trung vị vọt lên dù nửa giây gần nhất chỉ tiến lại bằng tốc độ người
    đi (lỗi thật IMG_7358 99,3 s, 29/10: "Đi sang trái." rồi "Dừng lại. Xe máy, phía trước, đang tới" về xe đỗ)."""
    if o.get('name') in NeVatCanDoSau.DONG_VAT:     # người/con vật không bao giờ "đỗ": như cũ (IMG_7354 45 s, 29/10)
        return o.get('v', 0.0) >= K.V_TOI
    vg = o.get('v_gan')
    return o.get('v', 0.0) >= K.V_TOI and (vg is None or vg >= K.V_TOI) and o.get('to_ra', True)


class NeVatCanDoSau:
    """Gọi step(ảnh BGR) cho mỗi khung (nên ~10 khung/giây); trả về câu cần nói (str) hoặc None.
    Câu trả về đã được đăng ký vào AUDIO_MAP, chỉ cần đưa vào luồng giọng nói vật cản (ưu tiên 100)."""
    W, H = 640, 360

    def __init__(self, detector=None, lenh_xa=3.0, persist=3):
        from transformers import AutoModelForDepthEstimation
        self.dev = K.DEV if torch.cuda.is_available() else "cpu"
        self.depth = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf").to(self.dev).eval()
        if self.dev != "cpu":
            self.depth = self.depth.half()
        self.det = K.DFineDet(detector or os.getenv("AIGLASS_NE_DETECTOR", "ustc-community/dfine-small-obj2coco"))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        self.F = (self.W / 2) / np.tan(np.radians(K.HFOV_DEG / 2))
        self.lenh_xa = lenh_xa
        self.streak = collections.deque(maxlen=persist)
        self.stairs_streak = collections.deque(maxlen=3)
        self.curb_streak = collections.deque(maxlen=3)
        self.drop_streak = collections.deque(maxlen=3)
        self.bac_xuong = os.getenv('AIGLASS_NE_BAC_XUONG', '0') == '1'   # bậc xuống/hố: đang thử, mặc định tắt
        self.km = K.KinhMu()
        # phân vùng lòng đường / vỉa hè (MIT): mép vỉa hè, biết đang ở lòng đường, không bảo dịch ra đường
        self.dvh = None if os.getenv('AIGLASS_NE_DUONG', '1') == '0' else K.DuongVaHe()
        self.dang_o_duong = False
        self.ttd = K.TrangThaiDuong()
        self.tracker = K.Tracker()
        self.noi_met = os.getenv('AIGLASS_NOI_KHOANG_CACH', '0') == '1'
        self.speaker = K.Speaker(im_xa=True, noi_khoang_cach=self.noi_met)
        self.last_act, self.last_act_t, self.last_act_d = None, -99.0, 99.0
        self.t0 = time.time()
        self.cand = []          # vật đang được báo (để vẽ)
        self.geo, self.yo = None, []   # hình học 3D + vật nhận diện của khung gần nhất (module vạch dùng: vật chắn)
        logger.info(f"[NE] Module né vật cản độ sâu sẵn sàng ({self.dev}), lõi: {_NE_DIR}")

    def _do_sau(self, img):
        rgb = torch.from_numpy(cv2.cvtColor(cv2.resize(img, (K.DEPTH_SIZE, K.DEPTH_SIZE)),
                                            cv2.COLOR_BGR2RGB)).to(self.dev)
        x = (rgb.permute(2, 0, 1)[None].float() / 255 - self.mean) / self.std
        if self.dev != "cpu":
            x = x.half()
        with torch.inference_mode():
            d = self.depth(pixel_values=x).predicted_depth[0].float()
        return cv2.resize(d.cpu().numpy(), (self.W, self.H))

    def step(self, frame_bgr, t=None, dang_qua_duong=False, chi_bao=None):
        """dang_qua_duong=True (đang trên vạch/lòng đường): không bao giờ bảo \"dịch\", chỉ \"Dừng lại\"/\"đang tới\".
        chi_bao(o) -> bool: chỉ nói về vật được chấp nhận (module vạch: vật cạnh vạch im, vật chắn do bộ điều khiển
        vật chắn nói); self.cand vẫn giữ đủ mọi vật."""
        t = (time.time() - self.t0) if t is None else t
        W, H, F = self.W, self.H, self.F
        img = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
        D = self._do_sau(img)
        yo = self.det.detect(img)
        anh_xau = K.KinhMu.anh_xau(img)          # tối / camera bị che: không dùng khung này để báo gì
        ground = None if anh_xau else K.fit_ground(D, H, W)
        cau_mu = self.km.update(t, anh_xau or ground is None)
        if cau_mu:
            if self.speaker.say(t, cau_mu, chen=cau_mu == 'khong_nhin_ro'):
                if cau_mu == 'khong_nhin_ro':
                    self.km.t_bao = t
                text = {'khong_nhin_ro': 'Kính không nhìn rõ, hãy dùng gậy', 'nhin_ro_lai': 'Kính đã nhìn rõ lại'}[cau_mu]
                _dang_ky_cau_noi(text, self.speaker.events[-1][1])
                del self.speaker.events[:-1]
                return text
            if cau_mu == 'nhin_ro_lai':
                self.km.mu = True               # đang đọc câu khác: thử lại khung sau
        obs, _, geo, _ = K.find_obstacles(D, H, W, ground, F) if ground else ([], None, None, [])
        self.geo, self.yo = geo, yo
        self.streak.append(bool(obs))
        cand = obs if len(self.streak) == self.streak.maxlen and all(self.streak) else []
        dyn = K.yoloe_obstacles(yo, geo, F, H, W) if geo else []
        self.tracker.update(t, dyn, W)
        song = {tr['id'] for tr in self.tracker.tracks}
        self.vet_co = {k: v for k, v in getattr(self, 'vet_co', {}).items() if k in song}   # (t, z, cao ảnh, chạm mép)
        vet = {tr['id']: tr['hist'] for tr in self.tracker.tracks}
        for o in dyn:
            o['v_gan'] = toc_do_gan(vet.get(o.get('tid')))
            co = self.vet_co.setdefault(o.get('tid'), collections.deque(maxlen=12))
            co.append((t, o['dist']) + chieu_cao_anh(o['mask']))
            o['to_ra'] = kich_thuoc_hop_ly(co)
            if o['name'] == 'person' and o.get('vok') and o['v'] >= K.V_XE:
                o['name'] = 'motorcycle'
            toi = dang_toi(o)
            nhanh = toi and o['v'] >= K.V_NHANH and o['gap'] <= K.HANH_LANG + K.LE_DONG
            le = K.LE_TOI if toi else (0.0 if o['name'] in K.XE_CO else K.LE_DONG)   # xe đỗ: không nới hành lang
            if o['gap'] > K.HANH_LANG + le or \
                    o['dist'] > (K.NGUONG_NHANH if o['v'] >= K.V_NHANH else K.NGUONG_M):
                continue
            if o['n'] < 2 and not toi:
                continue
            o['nhanh'], o['toi'] = nhanh, toi
            cand = [c for c in cand if (c['near'] & o['mask']).sum() < 0.3 * max(1, c['near'].sum())] + [o]
        dang_o_duong = False
        if geo and self.dvh is not None:
            road, side = self.dvh(img)
            geo['road'] = road
            dang_o_duong, mep = K.road_state(road, side, geo, H, W)
            self.curb_streak.append(mep is not None)
            if mep is not None and du_khung(self.curb_streak):
                lane_m = (np.abs(geo['X']) < 0.5) & (geo['Z'] >= mep - 0.25) & (geo['Z'] < mep + 0.5)
                cand.append(dict(mask=lane_m, near=lane_m, dist=mep, side='phía trước', high=False, name='curb',
                                 gap=0.0, xl=-0.5, xr=0.5, src='curb'))
        dang_o_duong = o_long_duong(dang_o_duong, getattr(self, 'tren_vach', False))
        self.dang_o_duong = dang_o_duong
        dang_qua_duong = dang_qua_duong or dang_o_duong
        if self.dvh is not None and geo:
            cau_tt = self.ttd.update(t, dang_o_duong)
            if cau_tt and self.speaker.say(t, cau_tt):
                self.ttd.da_bao(t)
                text = {'dang_o_duong': 'Đang ở lòng đường, chú ý xe', 'len_via_he': 'Đã lên vỉa hè'}[cau_tt]
                _dang_ky_cau_noi(text, self.speaker.events[-1][1])
                del self.speaker.events[:-1]
                return text
        st = K.find_stairs(geo, H, W, [o['mask'] for o in yo]) if geo else None
        self.stairs_streak.append(st is not None)
        if st and all(self.stairs_streak) and len(self.stairs_streak) == 3 and st['dist'] <= K.NGUONG_M:
            cand = [c for c in cand if c.get('src') == 'yoloe' or
                    (c['near'] & st['mask']).sum() < 0.3 * max(1, c['near'].sum())]
            cand.append(st)
        if self.bac_xuong and geo:
            dr = K.find_drop(geo, H, W, [o['mask'] for o in yo], geo.get('road'))
            self.drop_streak.append(dr is not None)
            if dr and all(self.drop_streak) and len(self.drop_streak) == 3 and dr['dist'] <= K.NGUONG_M:
                cand = [c for c in cand if c.get('src') == 'yoloe' or
                        (c['near'] & dr['mask']).sum() < 0.3 * max(1, c['near'].sum())]
                cand.append(dr)
        cand.sort(key=lambda c: c['dist'])
        self.cand = cand
        if chi_bao is not None:
            cand = [c for c in cand if chi_bao(c)]
        if not cand:
            self.speaker.update(t, None, None)
            if t - self.last_act_t > 1.5:
                self.last_act = None
            return None
        o = cand[0]
        nm_en = o['name'] if o.get('src') in ('yoloe', 'stairs', 'curb', 'drop') else K.name_of(o['near'], yo)
        nm = K.TEN_VI.get(nm_en, nm_en) if nm_en else None
        nm_txt = nm or ('vật cản ngang đầu' if o['high'] else 'vật cản')
        act, _ = K.plan_avoid(o, geo, F, W, prefer=self.last_act)
        if self.lenh_xa and act in ('dich_trai', 'dich_phai') and o['dist'] > self.lenh_xa:
            act = None
        if dang_qua_duong and act in ('dich_trai', 'dich_phai'):
            act = 'dung_lai' if o['dist'] < 2.5 or o.get('toi') else None   # giữa dòng xe: không dịch
        cung_vat = abs(o['dist'] - self.last_act_d) < 1.2
        if act is None and self.last_act and t - self.last_act_t < 1.5 and cung_vat:
            act = self.last_act
        elif act == 'dung_lai' and self.last_act in ('dich_trai', 'dich_phai') and t - self.last_act_t < 3.0 \
                and cung_vat and not o.get('nhanh'):
            act = self.last_act
        else:
            self.last_act_t, self.last_act_d = t, o['dist']
        act = lenh_sau_loc(chi_bao(o) if chi_bao is not None else True, act)
        self.last_act = act
        n_ev = len(self.speaker.events)
        if o.get('src') == 'yoloe' and not o.get('vok') and not act:
            return None     # người/xe mới thấy < 0,5 s: chờ biết đang tới hay đứng yên rồi mới nói 1 câu
        d = self.speaker.update(t, o, nm_en, act, toi=o.get('toi', False))
        if not d:
            return None
        text = cau_vat(act, nm_txt, o['side'], o.get('toi', False), d if self.noi_met else None,
                       ngan=self.speaker.ngan)
        _dang_ky_cau_noi(text, self.speaker.events[n_ev][1])
        del self.speaker.events[:-1]          # không giữ âm thanh cũ trong bộ nhớ
        return text

    # ---------- cho bộ điều khiển vật chắn trên vạch (vat_chan_vach.py) ----------
    DONG_VAT = ('person', 'dog', 'cat', 'bird', 'animal')   # người / con vật: có thể đi bất cứ lúc nào

    def dung_yen(self, o):
        """Vật đứng yên (đỗ, đặt trên vạch) - không phải xe/người đang chạy tới, không phải địa hình.
        Người, con vật: không bao giờ coi là đứng yên (chạy 7 video 2026-10-28: người đi tới bị báo "Vạch bị chắn").
        Vật YOLOE chưa đo được tốc độ: chưa biết, không coi là đứng yên."""
        if o.get('toi') or o.get('nhanh') or o.get('src') in ('curb', 'stairs', 'drop', 'hole'):
            return False
        if self.ten_vat(o)[0] in self.DONG_VAT:
            return False
        if o.get('src') == 'yoloe':
            return bool(o.get('vok')) and not dang_toi(o) and abs(o.get('vx', 0.0)) < 0.8
        return True

    def co_the_dang_do(self, o):
        """Có thể là xe/vật đỗ: không phải người/con vật, không đang tới. Bộ điều khiển vật chắn coi là xe đỗ nếu nó nối
        tiếp xe đỗ vừa thấy (NhoVatChan.tiep): YOLOE mất dấu lập vết mới chưa rõ tốc độ (IMG_7358 98,5–99,8 s), người đi
        chéo làm tốc độ ngang thoáng qua > 0,8 m/s (IMG_7354 26,3 s)."""
        return not (o.get('toi') or o.get('nhanh')) and self.ten_vat(o)[0] not in self.DONG_VAT

    def ten_vat(self, o):
        """(tên tiếng Anh hoặc None, tên tiếng Việt hoặc None) như câu nói hiện có."""
        en = o.get('name') if o.get('src') in ('yoloe', 'stairs', 'curb', 'drop') else K.name_of(o['near'], self.yo)
        return en, (K.TEN_VI.get(en, None) if en else None)

    def ben_di_duoc(self, o):
        """Thứ tự bên vòng cho vật o (vat_chan_vach.xep_ben): plan_avoid (dải rộng 1 người, không có gì nhô lên,
        trong khung, không phải phần lớn lòng đường) + tỉ lệ lòng đường từng bên."""
        from vat_chan_vach import xep_ben
        geo = self.geo
        if geo is None or 'xl' not in o or 'xr' not in o:
            return []
        # chỉ dùng phần kiểm KHOẢNG TRỐNG của plan_avoid: bỏ các nhánh chính sách cho xe chạy / chưa rõ / "đi cùng chiều"
        # (xe ĐỖ do YOLOE thấy có v ~ 0 rơi vào nhánh "đi cùng chiều" -> không xét hai bên; lỗi thật IMG_7358)
        _, free = K.plan_avoid(dict(o, src='do_sau', nhanh=False, toi=False, gap=0.0), geo, self.F, self.W,
                               prefer=None)
        duong = {}
        if geo.get('road') is not None:
            Z, X = geo['Z'], geo['X']
            for act, c in (('dich_trai', o['xl'] - K.NUA_NGUOI - K.DEM), ('dich_phai', o['xr'] + K.NUA_NGUOI + K.DEM)):
                st = (np.abs(X - c) < K.NUA_NGUOI) & (Z > 0.5) & (Z < o['dist'] + 1.0)
                if st.sum() >= 50:
                    duong[act] = float(geo['road'][st].mean())
        self.ti_le_duong = {'trai': duong.get('dich_trai'), 'phai': duong.get('dich_phai')}   # bộ điều khiển: bên xa đường
        return xep_ben(free or {}, duong)

    def dang_ky_cau(self, cau, ten_en=None):
        """Ghép câu của bộ điều khiển vật chắn từ mảnh WAV (thu_nghiem_do_sau/giong) và đăng ký để phát được."""
        rieng = {'Vạch bị chắn, dừng lại, hãy dùng gậy.': ['vach_bi_chan'], 'Đã về vạch.': ['da_ve_vach'],
                 'Vạch ở bên phải.': ['vach_ben_phai'], 'Vạch ở bên trái.': ['vach_ben_trai'],
                 'Chưa thấy lại vạch, hãy dùng gậy.': ['chua_thay_lai_vach']}
        cho = ('chan_vach_ben_phai' if 'chắn vạch bên phải' in cau else 'chan_vach_ben_trai' if 'chắn vạch bên trái'
               in cau else 'chan_giua_vach' if 'chắn giữa vạch' in cau else 'chan_vach')
        if 'dò đường vòng' in cau:
            sau = ['dung_gay_do_vong']
        else:
            sau = ['di_sang_trai' if 'Đi sang trái' in cau else 'di_sang_phai'] + (['do_gay'] if 'dò gậy' in cau else [])
        keys = rieng.get(cau) or [K.TEN_KEY.get(ten_en) or 'ten_vat_can', cho] + sau
        gap = np.zeros(int(0.08 * K.SR), np.int16)
        nghi = np.zeros(int(0.25 * K.SR), np.int16)   # dấu chấm giữa "… bên phải." và "Đi sang trái."
        phan = []
        for i, k in enumerate(keys):
            phan += [self.speaker.clip(k), nghi if (i == 1 and k.startswith('chan_') and len(keys) > 2) else gap]
        _dang_ky_cau_noi(cau, np.concatenate(phan[:-1]))
        return cau

    def ve(self, vis):
        """Tô vật đang báo lên ảnh vis (cùng tỉ lệ khung gốc) để xem trên /ws/viewer."""
        if not self.cand:
            return vis
        h, w = vis.shape[:2]
        for k, o in enumerate(self.cand[:3]):
            m = cv2.resize(o['mask'].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            col = np.array((255, 0, 255) if o.get('src') == 'yoloe' else (0, 0, 255))
            vis[m] = (0.6 * vis[m] + 0.4 * col).astype(np.uint8)
        return vis
