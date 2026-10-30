#!/usr/bin/env python3
"""
THỬ NGHIỆM ghép 3 model cho module né vật cản (không sửa code dự án):
  1. Độ sâu (Depth Anything V2 Small, Apache-2.0)  -> NGƯỜI GÁC: có thứ gì nhô lên, chắn hành lang, ở gần không?
  2. YOLOE (danh sách từ khóa VN)                   -> GỌI TÊN vật mà độ sâu tìm thấy
  3. yolo-seg của tác giả                           -> vẽ gạch dẫn đường / vạch qua đường (để tham khảo)
So sánh luôn với CÁCH CŨ (YOLOE + luật "chân ở 1/4 dưới ảnh hoặc > 12% ảnh").

Ý tưởng tìm vật cản từ độ sâu:
- Độ sâu của Depth Anything là "nghịch đảo khoảng cách" tương đối (lớn = gần), sai khác một tỉ lệ + độ lệch.
- Với mặt đất phẳng, giá trị này tăng TUYẾN TÍNH theo hàng ảnh y: D_đất(y) = a*y + b  ->  khớp đường thẳng ở vùng dưới-giữa.
- Từ a, b + chiều cao camera (H_CAM) + tiêu cự ước lượng (FOV) -> đổi ra mét: Z = F*H_CAM*a / D.
- Điểm ảnh là VẬT CẢN nếu: gần hơn mặt đất ở cùng hàng (nhô lên), nằm trong hành lang rộng ~vai,
  cao 0.15–2.0 m so với mặt đất, cách < NGUONG_M mét. Gom thành khối, lấy khối gần nhất.

[Bản 3, 2026-10-24] YOLOE thành NGƯỜI GÁC THỨ HAI cho người/xe/con vật (không qua luật lọc tường):
- khoảng cách lấy từ CHÂN vật chạm đất (hình học mặt đất), lệch ngang lấy theo MÉP GẦN lối đi nhất;
- theo dõi từng vật qua các khung -> vật LAO TỚI nhanh thì nhìn xa hơn (8 m) và báo ngay;
- chạy lại cách bản 2 (chỉ độ sâu quyết định): thêm --khong-gop.

Chạy (đứng ở glassai/, đã bật .venv):
  python loi_do_sau/ket_hop_3_model.py loi_do_sau/video/vid_a.mp4
Kết quả: loi_do_sau/ket_qua/<tên>_ket_hop.mp4 + số liệu in ra màn hình.
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import wave
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForDepthEstimation

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), 'OpenAIglasses_for_Navigation')
DEV = 'cuda'

# Tham số (ước lượng, chỉnh được)
H_CAM = 1.5          # chiều cao camera so với mặt đất (m); [Bản 7] --hieu-chinh: tự ước lượng (thử, chưa tốt hơn)
HFOV_DEG = 70.0      # góc nhìn ngang của camera điện thoại (1x) ở khung 16:9
HANH_LANG = 0.6      # nửa bề rộng hành lang (m) -> rộng 1.2 m
NGUONG_M = 5.0       # cảnh báo khi vật trong hành lang gần hơn (m); camera điện thoại nhìn thẳng nên mép ảnh đã ~3 m
CAO_MIN, CAO_MAX = 0.15, 2.0   # vật nhô cao khỏi mặt đất trong khoảng này (m)
DEPTH_SIZE = 392     # cỡ ảnh đưa vào model độ sâu
# [Bản 3] vật YOLOE được quyền tự báo (người, xe, con vật: hay di chuyển, hay bị luật lọc tường bỏ mất)
DONG = {'person', 'motorcycle', 'parked motorcycle', 'scooter', 'bicycle', 'car', 'bus', 'truck',
        'stroller', 'dog', 'cat', 'animal'}
LE_DONG = 0.5        # nới hành lang thêm mỗi bên cho vật động (m) -> người đi sát vai vẫn được báo (bản 9: 0,2 -> 0,5)
NGUONG_NHANH = 12.0  # vật đang lao tới nhanh: báo từ xa hơn (m). [Bản 26] 8 -> 12: xe máy 6 m/s còn ~2 s để tránh
V_NHANH = 3.0        # tốc độ tiến lại (m/s) coi là "lao tới"; người đi bộ đã tự tạo ~1,2–1,5 m/s với vật đứng yên
# [Bản 6] vật ĐANG TỚI (người/xe đi ngược chiều): tiến lại nhanh hơn hẳn tốc độ đi bộ của mình
V_TOI = 2.0          # m/s
LE_TOI = 0.9         # nới hành lang mỗi bên cho vật đang tới (m): xe đi lệch 1–1,5 m vẫn có thể quệt vào
V_XE = 3.5           # 'người' tiến lại nhanh hơn mức này -> gọi là xe máy (người lái che mất xe)
XE_CO = {'motorcycle', 'parked motorcycle', 'scooter', 'bicycle', 'car', 'bus', 'truck'}
V_CUNG_CHIEU = 0.6   # vật động gần như không lại gần (đi cùng chiều phía trước) -> chỉ báo, không bảo dừng/dịch

VN_CLASSES = ['person', 'parked motorcycle', 'motorcycle', 'bicycle', 'car', 'bus', 'truck', 'scooter',
              'stroller', 'dog', 'cat', 'bird', 'animal', 'plastic stool', 'stool', 'chair', 'table', 'bench',
              'pole', 'bollard', 'potted plant', 'basket', 'bucket', 'box', 'traffic cone', 'trash can',
              'mannequin', 'fire hydrant', 'stone', 'tree trunk', 'signboard']
TEN_VI = {'person': 'người', 'parked motorcycle': 'xe máy', 'motorcycle': 'xe máy', 'scooter': 'xe máy',
          'bicycle': 'xe đạp', 'car': 'ô tô', 'bus': 'xe buýt', 'truck': 'xe tải', 'stroller': 'xe đẩy',
          'dog': 'chó', 'cat': 'mèo', 'bird': 'gà/chim', 'animal': 'động vật', 'plastic stool': 'ghế',
          'stool': 'ghế', 'chair': 'ghế', 'table': 'bàn', 'bench': 'ghế dài', 'pole': 'cột', 'bollard': 'cột',
          'potted plant': 'chậu cây', 'basket': 'giỏ', 'bucket': 'xô', 'box': 'thùng', 'traffic cone': 'cọc',
          'trash can': 'thùng rác', 'mannequin': 'ma-nơ-canh', 'fire hydrant': 'trụ nước', 'stone': 'đá',
          'tree trunk': 'gốc cây', 'signboard': 'biển'}

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


class DFineDet:
    """[Bản 8] Thay YOLOE (AGPL) bằng D-FINE (Apache-2.0, dùng thương mại được), nhận diện 80 lớp COCO.
    Không có mặt nạ -> dùng khung chữ nhật làm mặt nạ. Trả về cùng dạng dict như ObstacleDetectorClient.detect."""
    COCO2VN = {'person': 'person', 'bicycle': 'bicycle', 'car': 'car', 'motorcycle': 'motorcycle', 'bus': 'bus',
               'truck': 'truck', 'dog': 'dog', 'cat': 'cat', 'bird': 'bird', 'bench': 'bench', 'chair': 'chair',
               'potted plant': 'potted plant', 'fire hydrant': 'fire hydrant', 'dining table': 'table',
               'suitcase': 'box', 'umbrella': 'signboard', 'stop sign': 'signboard', 'cow': 'animal',
               'horse': 'animal', 'sheep': 'animal',
               # [Bản 24] model obj2coco dùng tên VOC: thiếu 3 dòng này thì MỌI xe máy/chậu cây/bàn đều bị bỏ (bản 8–23)
               'motorbike': 'motorcycle', 'pottedplant': 'potted plant', 'diningtable': 'table'}

    def __init__(self, repo, nguong=0.4):
        from transformers import AutoImageProcessor, DFineForObjectDetection
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = DFineForObjectDetection.from_pretrained(repo).to(DEV).eval().half()
        self.nguong = nguong

    def detect(self, img):
        H, W = img.shape[:2]
        inp = self.proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors='pt').to(DEV)
        inp['pixel_values'] = inp['pixel_values'].half()
        with torch.inference_mode():
            out = self.model(**inp)
        r = self.proc.post_process_object_detection(out, target_sizes=[(H, W)], threshold=self.nguong)[0]
        boxes = []
        for lab, b in zip(r['labels'].tolist(), r['boxes'].float().cpu().numpy()):
            name = self.COCO2VN.get(self.model.config.id2label[int(lab)])
            if name:
                boxes.append([name, *[float(v) for v in b]])
        # [Bản 11] người ngồi trên xe máy/xe đạp -> gộp vào xe (gọi "Xe máy", không phải "Người")
        xe = [bx for bx in boxes if bx[0] in ('motorcycle', 'bicycle')]
        gop = []
        for bx in boxes:
            if bx[0] != 'person':
                continue
            for v in xe:
                ix = max(0.0, min(bx[3], v[3]) - max(bx[1], v[1]))
                # chân người nằm trong khung xe và hai khung chồng ngang > 50% bề rộng người
                if ix > 0.5 * (bx[3] - bx[1]) and v[2] < bx[4] <= v[4] + 0.1 * (v[4] - v[2]):
                    v[1], v[2], v[3], v[4] = min(v[1], bx[1]), min(v[2], bx[2]), max(v[3], bx[3]), max(v[4], bx[4])
                    gop.append(id(bx))
                    break
        res = []
        for name, *b in boxes:
            if name == 'person' and any(id(bx) in gop for bx in boxes if bx[1:] == b and bx[0] == 'person'):
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((H, W), np.uint8)
            mask[y0:y1 + 1, x0:x1 + 1] = 255
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            if area / (H * W) > 0.7:
                continue
            res.append(dict(name=name, mask=mask, area=area, area_ratio=area / (H * W), center_x=(x0 + x1) / 2,
                            center_y=(y0 + y1) / 2, bottom_y_ratio=y1 / H))
        return res


def fit_ground(D, H, W):
    """Khớp D_đất(y)=a*y+b từ vùng dưới-giữa ảnh (thường là mặt đất). Trả về (a, b) hoặc None."""
    x0, x1 = int(W * 0.3), int(W * 0.7)
    ys = np.arange(int(H * 0.6), H - 2, 3)
    vals = np.array([np.percentile(D[y, x0:x1], 30) for y in ys])  # phân vị thấp: bỏ bớt vật nhô lên
    best, best_n = None, 0
    rng = np.random.default_rng(0)
    for _ in range(60):  # RANSAC đơn giản
        i, j = rng.choice(len(ys), 2, replace=False)
        if ys[i] == ys[j]:
            continue
        a = (vals[j] - vals[i]) / (ys[j] - ys[i])
        b = vals[i] - a * ys[i]
        if a <= 0:
            continue
        res = np.abs(vals - (a * ys + b)) / (np.abs(vals) + 1e-6)
        n = (res < 0.05).sum()
        if n > best_n:
            best, best_n = (a, b), n
    if best is None or best_n < len(ys) * 0.4:
        return None
    m = np.abs(vals - (best[0] * ys + best[1])) / (np.abs(vals) + 1e-6) < 0.05
    a, b = np.polyfit(ys[m], vals[m], 1)
    return (a, b) if a > 0 else None


GOC_CHUC = True     # [Bản 28] tính góc chúc camera (kính chúc xuống 15–20°); False = công thức cũ (camera nhìn ngang)


def goc_chuc(y_h, H, F):
    """Góc chúc θ (rad) suy từ hàng chân trời: chân trời nằm trên tâm ảnh F·tanθ điểm ảnh."""
    return float(np.arctan((H / 2 - y_h) / F)) if GOC_CHUC else 0.0


def cham_dat(yb, y_h, H, F):
    """Điểm chạm đất ở hàng ảnh yb -> (khoảng cách tới trước theo mặt đất, độ sâu dọc trục camera) (m).
    Camera cao H_CAM chúc θ: tia qua hàng yb nghiêng θ + atan((yb - cy)/F) dưới phương ngang."""
    th = goc_chuc(y_h, H, F)
    v = yb - H / 2 if GOC_CHUC else yb - y_h
    roi = v * np.cos(th) + F * np.sin(th)          # > 0: tia đi xuống
    zc = F * H_CAM / max(roi, 1e-6)
    return zc * (np.cos(th) - v / F * np.sin(th)), zc


def find_obstacles(D, H, W, ground, F):
    """Trả về (danh sách khối vật cản, ảnh mặt nạ hành lang, hình học, khối bị loại kèm lý do)."""
    a, b = ground
    y_h = -b / a                                  # hàng chân trời
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = goc_chuc(y_h, H, F)
    # [Bản 28] mặt đất: 1/Zc = cosθ·(y − y_h)/(F·H) và D tỉ lệ 1/Zc -> Zc = F·H·a/(D·cosθ) (θ = 0: đúng công thức cũ)
    Zc = F * H_CAM * a / (np.maximum(D, 1e-6) * np.cos(th))   # độ sâu dọc trục camera (m)
    v = yy - H / 2 if GOC_CHUC else yy - y_h
    Z = Zc * (np.cos(th) - v / F * np.sin(th))    # khoảng cách tới trước theo mặt đất (m)
    X = (xx - W / 2) * Zc / F                     # lệch ngang (m)
    height = H_CAM - Zc * (v * np.cos(th) + F * np.sin(th)) / F   # độ cao so với mặt đất (m)
    corridor = (np.abs(X) < HANH_LANG) & (Z < NGUONG_M) & (Z > 0.3)
    mask = corridor & (height > CAO_MIN) & (height < CAO_MAX)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    obs, rejected = [], []
    for k in range(1, n):
        m = lab == k
        if stats[k, cv2.CC_STAT_AREA] < H * W * 0.003:
            rejected.append((m, 'qua_nho'))
            continue
        z = float(np.percentile(Z[m], 10))
        z90 = float(np.percentile(Z[m], 90))
        xm = float(np.median(X[m]))
        # [Sửa lỗi 1] Tường, cửa cuốn, xe đỗ song song chạy DỌC hướng đi: nằm ở mép hành lang,
        # vị trí ngang gần như không đổi (mặt phẳng song song đường đi) và kéo dài về phía trước -> bỏ qua
        xs = float(np.std(X[m]))
        if abs(xm) > 0.3 and xs < 0.12 and z90 - z > 0.8:
            rejected.append((m, 'loc_tuong'))
            continue
        hmin = float(np.percentile(height[m], 5))
        side = 'bên trái' if xm < -0.2 else ('bên phải' if xm > 0.2 else 'phía trước')
        near = m & (Z <= z + 0.7)  # phần gần nhất của khối, dùng để gọi tên
        obs.append(dict(mask=m, near=near, dist=z, side=side, high=hmin > 1.2))
    obs.sort(key=lambda o: o['dist'])
    # hình hành lang trên mặt đất để vẽ
    lane = np.zeros((H, W), np.uint8)
    pts = []
    for zz in (0.8, NGUONG_M):
        goc = np.arctan(H_CAM / zz) - th           # góc của tia tới điểm đất cách zz so với trục camera
        y = H / 2 + F * np.tan(goc) if GOC_CHUC else y_h + F * H_CAM / zz
        zc = np.hypot(zz, H_CAM) * np.cos(goc)
        for xs in (-HANH_LANG, HANH_LANG):
            pts.append((W / 2 + xs * F / zc, y))
    poly = np.array([pts[0], pts[1], pts[3], pts[2]], np.int32)
    cv2.fillPoly(lane, [poly], 1)
    for o in obs:
        o['xl'] = float(np.percentile(X[o['near']], 5))
        o['xr'] = float(np.percentile(X[o['near']], 95))
    return obs, lane, dict(Z=Z, Zc=Zc, X=X, height=height, y_h=y_h, theta=th), rejected


def yoloe_obstacles(yo, geo, F, H, W):
    """[Bản 3] Người/xe/con vật YOLOE thấy -> tính khoảng cách từ CHÂN chạm đất và lệch ngang theo MÉP GẦN
    lối đi nhất. Trả về mọi vật động cách < NGUONG_NHANH (lọc hành lang/khoảng cách để sau, khi biết tốc độ)."""
    out = []
    for o in yo:
        if o['name'] not in DONG:
            continue
        m = o['mask'] > 0
        ys, xs = np.nonzero(m)
        if len(ys) < 50:
            continue
        yb = ys.max()
        if yb >= H - 3 or yb <= geo['y_h'] + 2:
            # chân bị cắt ở mép ảnh (vật rất gần) hoặc ở trên chân trời -> lấy theo độ sâu của phần gần nhất
            z = float(np.percentile(geo['Z'][m], 10))
            zc = float(np.percentile(geo['Zc'][m], 10))
        else:
            z, zc = cham_dat(yb, geo['y_h'], H, F)
        if not 0.3 < z < NGUONG_NHANH:
            continue
        low = ys >= yb - max(3, (yb - ys.min()) * 0.3)    # 30% dưới của vật (chân, bánh xe)
        xl = (np.percentile(xs[low], 5) - W / 2) * zc / F
        xr = (np.percentile(xs[low], 95) - W / 2) * zc / F
        gap = max(xl, -xr, 0.0)                            # khoảng cách ngang từ giữa lối tới mép gần nhất
        if xr < -0.2:
            side = 'bên trái'
        elif xl > 0.2:
            side = 'bên phải'
        else:
            side = 'phía trước'
        out.append(dict(mask=m, near=m, dist=z, side=side, high=False, name=o['name'], gap=gap, xl=xl, xr=xr,
                        cx=float(xs.mean()), src='yoloe'))
    return out


NUA_NGUOI = 0.35     # nửa bề ngang người đi (m)
DEM = 0.15           # khoảng đệm khi đi qua vật (m)
DICH_MAX = 1.5       # dịch xa hơn mức này coi như không đi được


def plan_avoid(o, geo, F, W, prefer=None):
    """[Bản 4] Chọn lệnh tránh cho vật o: 'dich_trai' / 'dich_phai' / 'dung_lai' / None (chỉ cảnh báo).
    Bên trái đi được nếu dải rộng 1 người, đặt ngay cạnh mép trái của vật, không có gì nhô lên
    (tính tới hơi sau vật 1,5 m) và vẫn nằm trong khung hình; tương tự bên phải."""
    if o.get('nhanh'):
        # [Bản 27] vật lao tới: "Dừng lại" chỉ khi sắp chạm (< 2 s) hoặc đã gần (< 4 m); còn xa thì chỉ báo
        # "đang tới" (tốc độ đo ở 8–12 m nhiễu, ra lệnh dừng quá sớm gây hoảng)
        sap_cham = o['dist'] < 4.0 or o['dist'] / max(o.get('v', 0.0), 0.1) < 2.0
        return ('dung_lai' if sap_cham else None), {}
    if o.get('name') == 'curb':
        return None, {}                       # mép vỉa hè: chỉ báo, người dùng tự dò bằng gậy
    if o.get('name') in ('stairs', 'drop', 'hole'):
        return None, {}                       # cầu thang: chỉ báo, người dùng tự dò bậc bằng gậy
    if o.get('src') == 'yoloe' and o.get('name') in XE_CO and o.get('vok') and \
            (o['v'] >= V_TOI or abs(o.get('vx', 0.0)) >= 0.8):
        # [Bản 15] xe ĐANG CHẠY (tiến lại hoặc cắt ngang, ví dụ khi qua đường): không bao giờ bảo dịch
        return ('dung_lai' if o['dist'] < 4.0 else None), {}
    if o.get('src') == 'yoloe' and o.get('name') in XE_CO and not o.get('vok'):
        return None, {}                       # chưa rõ xe chạy hay đỗ (< 0,5 s theo dõi): chưa bảo dịch
    if o.get('gap', 0) > HANH_LANG:          # vật động chỉ sát vai, không chắn lối -> chỉ cảnh báo
        return None, {}
    if o.get('src') == 'yoloe' and o.get('vok') and o['v'] < V_CUNG_CHIEU:
        return None, {}                       # [Bản 6] đi cùng chiều phía trước, không lại gần -> chỉ báo
    Z, X, h = geo['Z'], geo['X'], geo['height']
    occ = (h > CAO_MIN) & (h < CAO_MAX) & (Z > 0.3) & (Z < o['dist'] + 1.5) & ~o['mask']
    lim = W * 0.003
    free = {}
    for act, c in (('dich_trai', o['xl'] - NUA_NGUOI - DEM), ('dich_phai', o['xr'] + NUA_NGUOI + DEM)):
        px = W / 2 + c * F / max(o['dist'], 0.5)
        ok = abs(c) <= DICH_MAX and NUA_NGUOI * F / o['dist'] < px < W - NUA_NGUOI * F / o['dist']
        if ok:
            ok = (occ & (np.abs(X - c) < NUA_NGUOI)).sum() < lim
        if ok and geo.get('road') is not None:
            # [Bản 17] dải sẽ dịch tới mà phần lớn là lòng đường -> không đi được
            st = (np.abs(X - c) < NUA_NGUOI) & (Z > 0.5) & (Z < o['dist'] + 1.0)
            ok = st.sum() < 50 or geo['road'][st].mean() < 0.3
        free[act] = (ok, abs(c))
    ok_sides = [a for a in free if free[a][0]]
    if prefer in ok_sides:                    # giữ lệnh cũ nếu vẫn đi được -> không đổi trái/phải liên tục
        return prefer, free
    if ok_sides:
        return min(ok_sides, key=lambda a: free[a][1]), free
    return ('dung_lai' if o['dist'] < 2.5 else None), free


def _rising(prof):
    """Chỉ số nấc bắt đầu chuỗi >= 4 nấc (0,5 m/nấc) dâng lên đều, dốc 0,25–1,2 m/m; None nếu không có."""
    for i in range(len(prof) - 3):
        seg = prof[i:i + 4]
        if np.isnan(seg).any() or seg[0] > 0.12 or seg[1] < 0.06:
            continue
        if (np.diff(seg) > 0.03).all() and 0.25 <= (seg[3] - seg[0]) / 1.5 <= 1.2:
            return i
    return None


def find_drop(geo, H, W, yo_masks=(), road=None):
    """[Bản 26] Bậc ĐI XUỐNG / HỐ trên lối đi: theo khoảng cách dọc lối (bước 0,5 m), độ cao mặt nền (trung vị) đang
    phẳng (> -0,05 m) thì tụt hẳn xuống (< -0,10 m, tụt > 0,08 m trong 0,5 m) và giữ thấp ít nhất 1 m.
    Tụt trên bề ngang rộng (>= 3/4 dải trong ±1 m) -> 'drop' (bậc xuống); chỉ ở dải giữa rồi lại phẳng -> 'hole'.
    Bỏ chỗ là lòng đường (đã có câu "Sắp xuống lòng đường") và chỗ trùng người/xe đã nhận diện."""
    Z, X, h = geo['Z'], geo['X'], geo['height']
    bins = np.arange(0.5, 6.0, 0.5)
    busy = np.zeros((H, W), bool)
    for m in yo_masks:
        busy |= m > 0
    if road is not None:
        busy |= road > 0
    hits = {}
    for x0 in (-1.0, -0.5, 0.0, 0.5):
        strip = (X >= x0) & (X < x0 + 0.5) & (Z > 0.5) & (Z < 6.0) & ~busy
        prof = []
        for z0 in bins:
            m = strip & (Z >= z0) & (Z < z0 + 0.5)
            prof.append(float(np.median(h[m])) if m.sum() > 15 else np.nan)
        prof = np.array(prof)
        for k in range(1, len(prof) - 1):
            if np.isnan(prof[k - 1:k + 2]).any():
                continue
            if prof[k - 1] > -0.05 and prof[k] < -0.10 and prof[k + 1] < -0.10 and prof[k] - prof[k - 1] < -0.08:
                lai_phang = bool(k + 3 < len(prof) and not np.isnan(prof[k + 3]) and prof[k + 3] > -0.05)
                hits[x0] = (k, lai_phang)
                break
    if len(hits) >= 3:
        kind = 'drop'
    elif hits and set(hits) <= {-0.5, 0.0} and all(lp for _, lp in hits.values()):
        kind = 'hole'
    else:
        return None
    z = float(bins[int(np.median([k for k, _ in hits.values()]))])
    mask = (np.abs(X) < 1.0) & (Z >= z) & (Z < z + 1.0) & (h < -0.08) & ~busy
    if mask.sum() < H * W * 0.002:
        return None
    return dict(mask=mask, near=mask, dist=z, side='phía trước', high=False, name=kind,
                gap=0.0, xl=-0.5, xr=0.5, src='drop')


def find_stairs(geo, H, W, yo_masks=()):
    """[Bản 13] Cầu thang ĐI LÊN trên lối đi: theo khoảng cách dọc lối (bước 0,5 m), độ cao mặt nền (trung vị)
    dâng lên đều >= 4 nấc với độ dốc 0,25–1,2 m/m (cầu thang ~30°), khác vật thẳng đứng (dốc rất lớn).
    Phải dâng trên BỀ NGANG RỘNG (>= 3/4 dải 0,5 m trong ±1 m) và không trùng xe/người đã nhận diện
    (đuôi xe máy, ô tô đỗ cũng "dâng lên" nhưng hẹp). Trả về dict vật cản kiểu 'stairs' hoặc None."""
    Z, X, h = geo['Z'], geo['X'], geo['height']
    bins = np.arange(0.5, 8.0, 0.5)
    busy = np.zeros((H, W), bool)
    for m in yo_masks:
        busy |= m > 0
    starts = []
    for x0 in (-1.0, -0.5, 0.0, 0.5):
        strip = (X >= x0) & (X < x0 + 0.5) & (Z > 0.5) & (Z < 8.0) & ~busy
        prof = []
        for z0 in bins:
            m = strip & (Z >= z0) & (Z < z0 + 0.5)
            prof.append(float(np.median(h[m])) if m.sum() > 15 else np.nan)
        k = _rising(np.array(prof))
        if k is not None:
            starts.append(k)
    if len(starts) < 3:
        return None
    z = float(bins[int(np.median(starts))])
    mask = (np.abs(X) < 1.0) & (Z >= z) & (Z < z + 2.0) & (h > 0.04) & ~busy
    if mask.sum() < H * W * 0.003:
        return None
    return dict(mask=mask, near=mask, dist=z + 0.25, side='phía trước', high=False, name='stairs',
                gap=0.0, xl=-0.5, xr=0.5, src='stairs')


class DuongVaHe:
    """[Bản 17] Phân vùng LÒNG ĐƯỜNG / VỈA HÈ bằng UperNet-ConvNeXt-tiny (ADE20K, giấy phép MIT), ~45 ms/khung.
    Trả về (mặt nạ lòng đường, mặt nạ vỉa hè) cùng cỡ ảnh."""
    def __init__(self, repo='openmmlab/upernet-convnext-tiny'):
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = UperNetForSemanticSegmentation.from_pretrained(repo).to(DEV).eval().half()
        lab = self.model.config.id2label
        self.road = [k for k, v in lab.items() if v == 'road']
        self.side = [k for k, v in lab.items() if v == 'sidewalk']

    def __call__(self, img):
        H, W = img.shape[:2]
        inp = self.proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors='pt').to(DEV)
        inp['pixel_values'] = inp['pixel_values'].half()
        with torch.inference_mode():
            out = self.model(**inp)
        seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy()
        return np.isin(seg, self.road), np.isin(seg, self.side)


def road_state(road, side, geo, H, W):
    """[Bản 17] Từ phân vùng + hình học mặt đất:
    - dang_o_duong: dải gần nhất nhìn thấy (giữa lối) phần lớn là lòng đường;
    - mep: khoảng cách (m) tới chỗ vỉa hè chuyển sang lòng đường phía trước (trong 5 m), hoặc None.
    Dùng các dải 0,25 m dọc lối đi rộng 1 m."""
    Z, X = geo['Z'], geo['X']
    lane = (np.abs(X) < 0.5) & (Z > 0.6)
    def frac(z0, z1):
        m = lane & (Z >= z0) & (Z < z1)
        n = m.sum()
        if n < 40:
            return None, None
        return road[m].mean(), side[m].mean()
    # camera nhìn thẳng: mép dưới ảnh có thể đã cách chân 2–3 m -> dùng dải GẦN NHẤT nhìn thấy được
    zl = Z[lane]
    if zl.size < 100:
        return False, None
    zmin = float(np.percentile(zl, 2))
    r0, s0 = frac(zmin, zmin + 0.7)
    dang_o_duong = r0 is not None and r0 > 0.6
    mep = None
    if r0 is not None and s0 > 0.6:
        for z0 in np.arange(zmin + 0.5, 5.0, 0.25):
            r, s_ = frac(z0, z0 + 0.5)
            if r is not None and r > 0.6:
                mep = float(z0 + 0.25)
                break
    return dang_o_duong, mep


class TrangThaiDuong:
    """[Bản 21–22] Theo dõi vào/ra lòng đường. Đổi trạng thái khi giữ ổn định >= 0,8 s; hai lần báo cách nhau >= 6 s;
    "Đã lên vỉa hè" chỉ báo nếu đã ở lòng đường >= 2 s. Câu chưa nói được (đang đọc câu khác) giữ tối đa 3 s.
    Trả về 'dang_o_duong' / 'len_via_he' / None."""
    def __init__(self):
        self.o_duong = False
        self.t_doi = None        # lúc bắt đầu thấy trạng thái mới
        self.t_vao = -99.0       # lúc vào lòng đường
        self.t_bao = -99.0       # lần báo gần nhất
        self.cho = None
        self.da_noi = None       # câu trạng thái đã nói gần nhất

    def update(self, t, o_duong):
        if o_duong != self.o_duong:
            if self.t_doi is None:
                self.t_doi = t
            if t - self.t_doi >= 0.8:
                self.o_duong, self.t_doi = o_duong, None
                if o_duong:
                    self.t_vao = t
                    key = 'dang_o_duong'
                else:
                    key = 'len_via_he' if t - self.t_vao >= 2.0 and self.da_noi == 'dang_o_duong' else None
                if key and key != self.da_noi and t - self.t_bao >= 6.0:
                    self.cho = (key, t)
                elif key is None or self.cho and self.cho[0] != key:
                    self.cho = None
        else:
            self.t_doi = None
        if self.cho and t - self.cho[1] > 3.0:
            self.cho = None
        if self.cho:
            return self.cho[0]
        return None

    def da_bao(self, t):
        self.da_noi = self.cho[0] if self.cho else self.da_noi
        self.cho, self.t_bao = None, t


class KinhMu:
    """[Bản 26] Kính không nhìn rõ (tối, camera bị che/mờ, không tìm được mặt đất) liên tục >= 1,5 s -> nói
    "Kính không nhìn rõ, hãy dùng gậy" (nhắc lại tối đa mỗi 15 s); nhìn rõ lại >= 2 s -> "Kính đã nhìn rõ lại".
    Im lặng lúc kính mù nguy hiểm: người dùng tưởng phía trước trống."""
    def __init__(self):
        self.t_xau = self.t_tot = None
        self.mu = False
        self.t_bao = -99.0

    @staticmethod
    def anh_xau(img):
        """Ảnh quá tối hoặc gần như đồng màu (camera bị che, mờ): không dùng để báo gì cả."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return g.mean() < 30 or g.std() < 12

    def update(self, t, xau):
        if xau:
            self.t_tot = None
            self.t_xau = t if self.t_xau is None else self.t_xau
            if t - self.t_xau >= 1.5 and t - self.t_bao >= 15.0:
                self.mu = True
                return 'khong_nhin_ro'
        else:
            self.t_xau = None
            self.t_tot = t if self.t_tot is None else self.t_tot
            if self.mu and t - self.t_tot >= 2.0:
                self.mu = False
                return 'nhin_ro_lai'
        return None


class HieuChinh:
    """[Bản 7] Tự hiệu chỉnh chiều cao camera từ người đứng trong khung hình (người lớn ~1,65 m):
    H = 1,65 * (hàng chân - hàng chân trời) / chiều cao người (điểm ảnh) -- không phụ thuộc góc nhìn.
    Lấy trung vị các lần đo gần đây; chỉ dùng khi đủ mẫu, kẹp trong 0,9–1,8 m."""
    def __init__(self, macdinh):
        self.mau = collections.deque(maxlen=150)
        self.macdinh = macdinh

    def them(self, yo, y_h, H, W):
        for o in yo:
            if o['name'] != 'person':
                continue
            ys, xs = np.nonzero(o['mask'])
            if len(ys) < 200:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            hp = y1 - y0
            if y0 < 4 or y1 > H - 4 or x0 < 4 or x1 > W - 4 or hp < 40 or y1 - y_h < 10 or (x1 - x0) / hp > 0.6:
                continue                               # bị cắt ở mép, quá nhỏ, ngồi/cúi
            self.mau.append(1.65 * (y1 - y_h) / hp)

    def gia_tri(self):
        if len(self.mau) < 15:
            return self.macdinh
        return float(np.clip(np.median(self.mau), 0.9, 1.8))


class Tracker:
    """[Bản 3] Theo dõi vật động qua các khung: ghép theo tên nhóm + vị trí ngang, đo tốc độ tiến lại (m/s)."""
    NHOM = {'parked motorcycle': 'motorcycle', 'scooter': 'motorcycle', 'cat': 'animal', 'dog': 'animal'}

    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, t, objs, W):
        for tr in self.tracks:
            tr['matched'] = False
        for o in objs:
            g = self.NHOM.get(o['name'], o['name'])
            best, best_c = None, 1e9
            for tr in self.tracks:
                if tr['matched'] or tr['g'] != g:
                    continue
                c = abs(tr['cx'] - o['cx']) / W + abs(tr['z'] - o['dist']) / 3
                if c < best_c and abs(tr['cx'] - o['cx']) < W * 0.2 and abs(tr['z'] - o['dist']) < 2.5:
                    best, best_c = tr, c
            if best is None:
                best = dict(id=self.next_id, g=g, hist=collections.deque(maxlen=12), n=0)
                self.next_id += 1
                self.tracks.append(best)
            best.update(cx=o['cx'], z=o['dist'], seen=t, matched=True)
            best['hist'].append((t, o['dist']))
            best.setdefault('hx', collections.deque(maxlen=12)).append((t, (o['cx'] - W / 2) * o['dist'] / (W / 2 / np.tan(np.radians(HFOV_DEG / 2)))))
            best['n'] += 1
            h = best['hist']
            # [Bản 6] cần >= 5 lần thấy trong >= 0,5 s; lấy trung vị độ dốc từng cặp (bền với khung đo sai)
            o['v'], o['vok'] = 0.0, False
            if len(h) >= 5 and h[-1][0] - h[0][0] >= 0.5:
                sl = [(h[i][1] - h[j][1]) / (h[j][0] - h[i][0]) for i in range(len(h))
                      for j in range(i + 1, len(h)) if h[j][0] - h[i][0] > 0.2]
                if sl:
                    o['v'], o['vok'] = float(np.median(sl)), True   # dương = đang tiến lại
            # [Bản 15] tốc độ ngang (m/s): xe cắt ngang trước mặt
            hx = best['hx']
            o['vx'] = 0.0
            if len(hx) >= 5 and hx[-1][0] - hx[0][0] >= 0.5:
                o['vx'] = float(np.polyfit([p[0] for p in hx], [p[1] for p in hx], 1)[0])
            o['n'] = best['n']
            o['tid'] = best['id']
        self.tracks = [tr for tr in self.tracks if t - tr['seen'] < 1.0]


def name_of(near_mask, yoloe_objs):
    """[Sửa lỗi 2] Gọi tên theo PHẦN GẦN NHẤT của khối (tránh lấy tên vật đứng phía sau).
    Trả về tên tiếng Anh của YOLOE hoặc None."""
    best, best_r = None, 0.0
    area = max(1, near_mask.sum())
    for o in yoloe_objs:
        r = np.logical_and(near_mask, o['mask'] > 0).sum() / area
        if r > best_r:
            best, best_r = o['name'], r
    return best if best_r > 0.3 else None


# ---------- Giọng nói ----------
GIONG = os.path.join(HERE, 'giong')
SR = 16000
TEN_KEY = {'person': 'ten_nguoi', 'parked motorcycle': 'ten_xe_may', 'motorcycle': 'ten_xe_may',
           'scooter': 'ten_xe_may', 'bicycle': 'ten_xe_dap', 'car': 'ten_o_to', 'bus': 'ten_xe_buyt',
           'truck': 'ten_xe_tai', 'stroller': 'ten_xe_day', 'dog': 'ten_cho', 'cat': 'ten_meo',
           'bird': 'ten_ga_chim', 'animal': 'ten_dong_vat', 'plastic stool': 'ten_ghe', 'stool': 'ten_ghe',
           'chair': 'ten_ghe', 'table': 'ten_ban', 'bench': 'ten_ghe_dai', 'pole': 'ten_cot', 'bollard': 'ten_cot',
           'potted plant': 'ten_chau_cay', 'basket': 'ten_gio', 'bucket': 'ten_xo', 'box': 'ten_thung',
           'traffic cone': 'ten_coc', 'trash can': 'ten_thung_rac', 'mannequin': 'ten_ma_no_canh',
           'fire hydrant': 'ten_tru_nuoc', 'stone': 'ten_da', 'tree trunk': 'ten_goc_cay', 'signboard': 'ten_bien'}
TEN_KEY['stairs'] = 'ten_cau_thang'
TEN_KEY['curb'] = 'ten_mep_via_he'
TEN_KEY['drop'] = 'ten_bac_xuong'
TEN_KEY['hole'] = 'ten_ho'
TEN_VI['drop'] = 'bậc xuống'
TEN_VI['hole'] = 'hố'
TEN_VI['curb'] = 'sắp xuống lòng đường'
TEN_VI['stairs'] = 'cầu thang lên'
ACT_VI = {'dich_trai': 'Dịch sang trái', 'dich_phai': 'Dịch sang phải', 'dung_lai': 'Dừng lại'}
SIDE_KEY = {'bên trái': 'ben_trai', 'bên phải': 'ben_phai', 'phía trước': 'phia_truoc'}


def load_clip(key):
    with wave.open(os.path.join(GIONG, key + '.wav')) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


class Speaker:
    """Luật nói [Bản 6] - mỗi vật nói ít lần:
    - vật MỚI (tên khác, hoặc cùng tên nhưng ở chỗ khác hẳn) -> nói ngay;
    - cùng vật: chỉ nói lại khi có LỆNH MỚI (dịch/dừng), khi vật bắt đầu ĐANG TỚI,
      khi vừa vào vùng rất gần (<= 2 m), hoặc đã im >= 8 s;
    - đổi bên hay đổi "Vật cản" <-> tên cụ thể của cùng vật -> không nói lại;
    - đang nói dở thì không chen; mất vật > 1,5 s thì quên."""
    def __init__(self, im_xa=False, noi_khoang_cach=True):
        self.im_xa = im_xa
        self.noi_khoang_cach = noi_khoang_cach   # False: không đọc "x mét" (đo khoảng cách còn sai; người dùng 28/10)
        self.cache = {}
        self.events = []          # (thời điểm giây, mảng âm thanh)
        self.busy_until = -1.0
        self.last = None          # dict(n=tên, act=lệnh, toi=, d=khoảng cách nói, z=khoảng cách thật)
        self.last_t, self.last_seen = -99.0, -99.0
        self.ngan = False         # câu vừa trả chỉ là lệnh ("Dừng lại."): xe vừa được báo "đang tới"

    def clip(self, key):
        if key not in self.cache:
            self.cache[key] = load_clip(key)
        return self.cache[key]

    def update(self, t, ob, name_en, act=None, toi=False):
        self.ngan = False
        if ob is None:
            if t - self.last_seen > 1.5:
                self.last = None
            return None
        self.last_seen = t
        nkey = TEN_KEY.get(name_en) or ('ten_vat_can_ngang_dau' if ob['high'] else 'ten_vat_can')
        skey = SIDE_KEY[ob['side']]
        d = int(min(5, max(1, round(ob['dist']))))
        khan = act == 'dung_lai' and (self.last is None or self.last['act'] != 'dung_lai')
        if t < self.busy_until and not khan:      # [Bản 15] "Dừng lại" mới được chen ngang câu đang đọc
            return None
        if self.im_xa and not act and not toi and skey != 'phia_truoc' and d > 2:
            return None                           # [Bản 10/13] vật bên cạnh, chưa sát (> 2 m), không cần làm gì -> im
        gap_t = t - self.last_t
        dia_hinh = nkey in ('ten_mep_via_he', 'ten_cau_thang', 'ten_bac_xuong', 'ten_ho')   # [Bản 19] mép vỉa hè / cầu thang: luôn được nói
        if self.im_xa and act != 'dung_lai' and not toi and not dia_hinh and gap_t < 3.0:
            return None                           # [Bản 13] câu không gấp: cách câu trước ít nhất 3 s
        L = self.last
        same = L is not None and t - self.last_t < 8.0 and (dia_hinh == (L['n'] in ('ten_mep_via_he', 'ten_cau_thang', 'ten_bac_xuong', 'ten_ho'))) and (
            ((nkey == L['n'] or 'ten_vat_can' in (nkey, L['n'])) and abs(ob['dist'] - L['z']) < 2.0)
            # YOLOE đổi tên liên tục cho cùng một đống vật (xe máy/xe tải/người) ở đúng chỗ cũ
            or (skey == L['s'] and abs(ob['dist'] - L['z']) < 1.0 and t - self.last_t < 4.0)
            # xe đang tới lại gần nhanh (6,5 -> 2 m trong 1 s): vẫn là xe đó (IMG_7358 89,0 s, 29/10: nói lại lần ba)
            or (toi and L['toi'] and nkey == L['n'] and ob['dist'] <= L['z'] + 0.5 and t - self.last_t < 3.0))
        if same:
            if act and act != L['act']:
                # có lệnh mới; xe vừa được báo "đang tới" < 3 s: chỉ nói lệnh (IMG_7358 88 s, 29/10: "Xe máy, bên
                # trái, đang tới" rồi ngay sau "Dừng lại. Xe máy, bên trái, đang tới" -> lặp)
                # cũng vậy khi vật vừa được báo thông tin (IMG_7355 137 s: "Xe máy, phía trước" rồi "Dừng lại. Xe máy")
                self.ngan = bool(toi == L['toi'] and t - self.last_t < 3.0)
            elif toi and not L['toi'] and not act:
                pass                              # vật bắt đầu đi về phía mình
            elif d <= 2 < L['d'] and t - self.last_t >= 1.5:
                pass                              # vừa vào vùng rất gần
            else:
                L['z'], L['n'], L['s'] = ob['dist'], nkey, skey   # cập nhật để còn bám theo vật
                return None
        gap = np.zeros(int(0.08 * SR), np.int16)
        met = [gap, self.clip(f'cach_{d}')] if self.noi_khoang_cach else []
        if self.ngan:
            audio = self.clip(act)
        elif act and toi and not self.noi_khoang_cach:   # "Dừng lại. Xe máy, phía trước, đang tới" (rõ là xe KHÁC đang tới)
            audio = np.concatenate([self.clip(act), gap, gap, self.clip(nkey), gap, self.clip(skey), gap,
                                    self.clip('dang_toi')])
        elif act:   # nói lệnh trước: "Dịch sang phải. Xe máy, 3 mét"
            audio = np.concatenate([self.clip(act), gap, gap, self.clip(nkey)] + met)
        elif toi:  # "Xe máy, bên trái, đang tới"
            audio = np.concatenate([self.clip(nkey), gap, self.clip(skey), gap, self.clip('dang_toi')])
        else:
            audio = np.concatenate([self.clip(nkey), gap, self.clip(skey)] + met)
        self.events.append((t, audio))
        self.busy_until = t + len(audio) / SR + 0.3
        self.last = dict(n=nkey, s=skey, act=act, toi=toi, d=d, z=ob['dist'])
        self.last_t = t
        return d

    def say(self, t, key, chen=False):
        """[Bản 21] Câu trạng thái đứng riêng (vào/ra lòng đường): nói ngay nếu không đang đọc câu khác.
        chen=True: câu an toàn được chen ngang (kính mù)."""
        if t < self.busy_until and not chen:
            return False
        audio = self.clip(key)
        self.events.append((t, audio))
        self.busy_until = t + len(audio) / SR + 0.3
        self.last_t = t
        return True

    def write_track(self, path, duration):
        track = np.zeros(int((duration + 5) * SR), np.int32)
        for t, a in self.events:
            i = int(t * SR)
            track[i:i + len(a)] += a[:max(0, len(track) - i)]
        track = np.clip(track[:int(duration * SR)], -32768, 32767).astype(np.int16)
        with wave.open(path, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(track.tobytes())


def mux(video, wav):
    tmp = video + '.tmp.mp4'
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video, '-i', wav, '-c:v', 'libx264',
                        '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '64k', '-shortest', tmp])
    os.remove(wav)
    if r.returncode == 0:
        os.replace(tmp, video)
        return True
    return False


def old_rule(yoloe_objs):
    near = [o for o in yoloe_objs if o['bottom_y_ratio'] > 0.75 or o['area_ratio'] > 0.12]
    if not near:
        return None
    return TEN_VI.get(max(near, key=lambda o: o['area_ratio'])['name'], 'vật cản')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--fps', type=float, default=10, help='số khung xử lý mỗi giây video')
    ap.add_argument('--persist', type=int, default=3, help='vật phải có liên tục N khung mới báo')
    ap.add_argument('--nguong', type=float, default=NGUONG_M, help='cảnh báo khi vật gần hơn (m)')
    ap.add_argument('--hcam', type=float, default=H_CAM, help='chiều cao camera (m)')
    ap.add_argument('--show', action='store_true', help='hiện cửa sổ xem trực tiếp (q để dừng, space tạm dừng)')
    ap.add_argument('--sound', action='store_true', help='phát giọng nói ra loa ngay lúc chạy')
    ap.add_argument('--save', action='store_true', help='(mặc định đã luôn lưu video có tiếng; giữ cho quen tay)')
    ap.add_argument('--khong-gop', action='store_true', help='chạy như bản 2: chỉ độ sâu quyết định báo')
    ap.add_argument('--yoloe', default='model/yoloe-11l-seg.pt', help='file YOLOE (tính từ OpenAIglasses_for_Navigation/)')
    ap.add_argument('--co-seg', action='store_true', help='vẽ thêm gạch dẫn đường bằng yolo-seg (AGPL, chậm hơn; chỉ để tham khảo)')
    ap.add_argument('--khong-tranh', action='store_true', help='chỉ cảnh báo, không nói lệnh dịch trái/phải/dừng')
    ap.add_argument('--lenh-xa', type=float, default=3,
                    help='chỉ nói "dịch trái/phải" khi vật gần hơn N mét (xa hơn thì chỉ báo thông tin); 0 = mọi khoảng cách')
    ap.add_argument('--detector', default='dfine:ustc-community/dfine-small-obj2coco',
                    help="'yoloe' (AGPL) hoặc 'dfine:<repo HF>' (Apache-2.0), ví dụ dfine:ustc-community/dfine-small-obj2coco")
    ap.add_argument('--hieu-chinh', action='store_true',
                    help='tự ước lượng chiều cao camera từ người trong ảnh (bản 7: dao động mạnh, báo nhầm tăng -> tắt mặc định)')
    ap.add_argument('--le-dong', type=float, default=LE_DONG, help='nới hành lang mỗi bên cho người/xe (m)')
    ap.add_argument('--noi-ca-xa', dest='im_xa', action='store_false',
                    help='nói cả câu thông tin về vật bên cạnh còn xa > 3 m (mặc định: im, chỉ nói lệnh, "đang tới", vật phía trước)')
    ap.add_argument('--khong-duong', action='store_true', help='tắt phân vùng lòng đường/vỉa hè (mép vỉa hè, không dịch ra đường)')
    ap.add_argument('--khong-goc-chuc', action='store_true', help='[Bản 28] công thức cũ: coi camera nhìn ngang')
    ap.add_argument('--bac-xuong', action='store_true', help='[Bản 26] báo bậc xuống/hố (đang thử; mặc định chỉ ghi log)')
    ap.add_argument('--khong-cau-thang', action='store_true', help='tắt phát hiện cầu thang đi lên')
    ap.add_argument('--tu', type=float, default=0, help='chỉ chạy từ giây này (soi một đoạn)')
    ap.add_argument('--den', type=float, default=1e9, help='chỉ chạy đến giây này')
    ap.add_argument('--hau-to', default='', help='thêm vào tên file kết quả, ví dụ _v3')
    args = ap.parse_args()
    globals().update(NGUONG_M=args.nguong, H_CAM=args.hcam, LE_DONG=args.le_dong, GOC_CHUC=not args.khong_goc_chuc)

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(HERE, 'ket_qua')
    os.makedirs(out_dir, exist_ok=True)
    cwd = os.getcwd()
    sys.path.insert(0, PROJECT)
    os.chdir(PROJECT)
    # chỉ nạp Ultralytics (AGPL) khi thật sự dùng yolo-seg hoặc YOLOE
    seg = None
    if args.co_seg:
        from ultralytics import YOLO
        seg = YOLO('model/yolo-seg.pt')
        seg.to(DEV)
    if args.detector.startswith('dfine:'):
        det = DFineDet(args.detector[6:])
    else:
        from obstacle_detector_client import ObstacleDetectorClient
        det = ObstacleDetectorClient(args.yoloe)
        det.WHITELIST_CLASSES = VN_CLASSES
        det.whitelist_embeddings = det.model.get_text_pe(VN_CLASSES)
    os.chdir(cwd)
    depth = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf').to(DEV).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    cap = cv2.VideoCapture(os.path.join(cwd, args.video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(vfps / args.fps))
    W, H = 640, 360
    F = (W / 2) / np.tan(np.radians(HFOV_DEG / 2))
    out_video = os.path.join(out_dir, f'{name}_co_tieng{args.hau_to}.mp4')
    tracker = Tracker()
    stairs_streak = collections.deque(maxlen=3)
    drop_streak = collections.deque(maxlen=3)
    curb_streak = collections.deque(maxlen=3)
    ttd = TrangThaiDuong()
    km = KinhMu()
    dvh = None if args.khong_duong else DuongVaHe()
    hc = HieuChinh(args.hcam)
    last_act, last_act_t, last_act_d = None, -99.0, 99.0
    log = []          # mỗi khung: thời điểm, cảnh báo mới/cũ, chẩn đoán vật động độ sâu bỏ sót
    t_yo = 0.0
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), vfps / step, (W, H))
    speaker = Speaker(im_xa=args.im_xa)
    spoken_log = []
    said_text, said_until = '', -1.0
    font = ImageFont.truetype(FONT, 15)
    stats = collections.Counter()
    names = collections.Counter()
    t_dep = t_all = 0.0
    streak = collections.deque(maxlen=args.persist)
    i = n = 0
    player = None
    if args.sound:
        if shutil.which('paplay'):
            player = ['paplay']
        elif shutil.which('aplay'):
            player = ['aplay', '-q', '-f', 'S16_LE', '-r', '16000', '-c', '1']
        else:
            print("Không có paplay/aplay, sẽ không phát tiếng trực tiếp (video lưu vẫn có tiếng).")
    t_start = time.perf_counter()
    while True:
        if (i + 1) / vfps < args.tu - 2:      # chạy trước 2 s cho tracker kịp có tốc độ
            if not cap.grab():
                break
            i += 1
            continue
        if (i + 1) / vfps > args.den:
            break
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        n += 1
        img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        # 1) độ sâu
        rgb = torch.from_numpy(cv2.cvtColor(cv2.resize(img, (DEPTH_SIZE, DEPTH_SIZE)), cv2.COLOR_BGR2RGB)).to(DEV)
        x = ((rgb.permute(2, 0, 1)[None].float() / 255 - mean) / std).half()
        with torch.inference_mode():
            d = depth(pixel_values=x).predicted_depth[0].float()
        torch.cuda.synchronize()
        t_dep += time.perf_counter() - t0
        D = cv2.resize(d.cpu().numpy(), (W, H))
        # 2) YOLOE + 3) yolo-seg
        t1 = time.perf_counter()
        yo = det.detect(img)
        torch.cuda.synchronize()
        t_yo += time.perf_counter() - t1
        r = seg.predict(img, verbose=False, conf=0.3)[0] if seg else None
        t_all += time.perf_counter() - t0

        anh_xau = KinhMu.anh_xau(img)
        ground = None if anh_xau else fit_ground(D, H, W)
        if ground and args.hieu_chinh:
            hc.them(yo, -ground[1] / ground[0], H, W)
            globals()['H_CAM'] = hc.gia_tri()
        obs, lane, geo, rejected = find_obstacles(D, H, W, ground, F) if ground else ([], None, None, [])
        streak.append(bool(obs))
        warn_new = None
        t_vid = i / vfps
        cand = obs if len(streak) == args.persist and all(streak) else []
        dyn_all = yoloe_obstacles(yo, geo, F, H, W) if geo else []
        tracker.update(t_vid, dyn_all, W)
        for o in dyn_all:
            # [Bản 12] xe máy đi thẳng tới bị người lái che -> chỉ thấy "người"; người đi bộ ngược chiều
            # tiến lại tối đa ~2,6 m/s, nhanh hơn hẳn thì gần như chắc là người đi xe
            if o['name'] == 'person' and o.get('vok') and o['v'] >= V_XE:
                o['name'] = 'motorcycle'
        chan_doan = []
        for o in dyn_all:
            # vật động trong hành lang (nới LE_DONG), gần < NGUONG_M; lao tới nhanh thì < NGUONG_NHANH
            toi = o['v'] >= V_TOI
            nhanh = o['v'] >= V_NHANH and o['gap'] <= HANH_LANG + LE_DONG
            # [Bản 25] xe không lao tới (thường là xe đỗ) không tự bước ngang vào lối như người -> không nới hành lang
            le = LE_TOI if toi else (0.0 if o['name'] in XE_CO else LE_DONG)
            if o['gap'] > HANH_LANG + le or \
                    o['dist'] > (NGUONG_NHANH if o['v'] >= V_NHANH else NGUONG_M):
                continue
            # chẩn đoán: độ sâu có thấy vật này không, nếu không thì bị loại ở bước nào
            ly_do = 'do_sau_thay'
            if not any((ob['mask'] & o['mask']).sum() > 0.2 * o['mask'].sum() for ob in obs):
                ly_do = 'do_sau_khong_thay'
                for rm, why in rejected:
                    if (rm & o['mask']).sum() > 0.2 * o['mask'].sum():
                        ly_do = why
                        break
            chan_doan.append(dict(ten=o['name'], z=round(o['dist'], 1), v=round(o['v'], 1), ly_do=ly_do))
            if args.khong_gop or (o['n'] < 2 and not toi):
                continue
            o['nhanh'], o['toi'] = nhanh, toi
            cand = [c for c in cand if (c['near'] & o['mask']).sum() < 0.3 * max(1, c['near'].sum())] + [o]
        dang_o_duong = False
        if geo and dvh is not None:
            road, side = dvh(img)
            geo['road'] = road
            dang_o_duong, mep = road_state(road, side, geo, H, W)
            curb_streak.append(mep is not None)
            if mep is not None and len(curb_streak) == 3 and all(curb_streak):
                lane_m = (np.abs(geo['X']) < 0.5) & (geo['Z'] >= mep - 0.25) & (geo['Z'] < mep + 0.5)
                cand.append(dict(mask=lane_m, near=lane_m, dist=mep, side='phía trước', high=False, name='curb',
                                 gap=0.0, xl=-0.5, xr=0.5, src='curb'))
        st = find_stairs(geo, H, W, [o['mask'] for o in yo]) if geo and not args.khong_cau_thang else None
        stairs_streak.append(st is not None)
        if st and len(stairs_streak) == 3 and all(stairs_streak) and st['dist'] <= NGUONG_M:
            # cầu thang thay cho các khối độ sâu không tên chồng lên nó
            cand = [c for c in cand if c.get('src') == 'yoloe' or (c['near'] & st['mask']).sum() < 0.3 * max(1, c['near'].sum())]
            cand.append(st)
        dr = find_drop(geo, H, W, [o['mask'] for o in yo], geo.get('road')) if geo else None
        drop_streak.append(dr is not None)
        xuong = None
        if dr and len(drop_streak) == 3 and all(drop_streak) and dr['dist'] <= NGUONG_M:
            xuong = (dr['name'], dr['dist'])
            if args.bac_xuong:
                cand = [c for c in cand if c.get('src') == 'yoloe' or (c['near'] & dr['mask']).sum() < 0.3 * max(1, c['near'].sum())]
                cand.append(dr)
        cand.sort(key=lambda c: c['dist'])
        if cand:
            o = cand[0]
            nm_en = o['name'] if o.get('src') in ('yoloe', 'stairs', 'curb', 'drop') else name_of(o['near'], yo)
            nm = TEN_VI.get(nm_en, nm_en) if nm_en else None
            nm_txt = nm or ('vật cản ngang đầu' if o['high'] else 'vật cản')
            warn_new = f"{nm_txt}, {o['side']}, {o['dist']:.1f} m" + (' (đang tới)' if o.get('toi') else '')
            stats['moi_bao'] += 1
            stats['moi_co_ten' if nm else 'moi_khong_ten'] += 1
            names[nm or 'không tên'] += 1
            act = None
            if not args.khong_tranh:
                act, _ = plan_avoid(o, geo, F, W, prefer=last_act)
                if args.lenh_xa and act in ('dich_trai', 'dich_phai') and o['dist'] > args.lenh_xa:
                    act = None                # [Bản 5] ở xa: chỉ báo thông tin, để người dùng tự quyết
                if dang_o_duong and act in ('dich_trai', 'dich_phai'):
                    act = 'dung_lai' if o['dist'] < 2.5 or o.get('toi') else None   # [Bản 17] giữa lòng đường: không dịch
                cung_vat = abs(o['dist'] - last_act_d) < 1.2   # [Bản 6] chỉ giữ lệnh khi vẫn là vật đó
                if act is None and last_act and t_vid - last_act_t < 1.5 and cung_vat:
                    act = last_act            # giữ lệnh vừa nói thêm 1,5 s, tránh nhảy có lệnh <-> không lệnh
                elif act == 'dung_lai' and last_act in ('dich_trai', 'dich_phai') and t_vid - last_act_t < 3.0 \
                        and cung_vat and not o.get('nhanh'):
                    act = last_act            # vật đã sát (bên trống ra khỏi khung hình): nhớ bên trống lúc còn xa
                else:
                    last_act_t, last_act_d = t_vid, o['dist']
                last_act = act
                if act:
                    warn_new = f"{ACT_VI[act].upper()} | " + warn_new
            n_ev = len(speaker.events)
            if o.get('src') == 'yoloe' and not o.get('vok') and not act:
                d = None    # [Bản 20] người/xe mới thấy < 0,5 s: chờ biết đang tới hay đứng yên rồi mới nói 1 câu
            else:
                d = speaker.update(t_vid, o, nm_en, act, toi=o.get('toi', False))
            if d and args.sound and player:
                # phát ngay câu vừa quyết định nói (PCM 16 kHz mono), không chặn vòng lặp
                pcm = speaker.events[n_ev][1].tobytes()
                pr = subprocess.Popen(player + ['--rate=16000', '--channels=1', '--format=s16le', '--raw']
                                      if player[0] == 'paplay' else player, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pr.stdin.write(pcm)
                pr.stdin.close()
            if d:
                said_text = (f"{ACT_VI[act]}. {nm_txt.capitalize()}, {d} mét" if act
                             else f"{nm_txt.capitalize()}, {o['side']}, đang tới" if o.get('toi')
                             else f"{nm_txt.capitalize()}, {o['side']}, {d} mét")
                said_until = t_vid + 2.0
                spoken_log.append((t_vid, said_text))
        else:
            speaker.update(t_vid, None, None)
            if t_vid - last_act_t > 1.5:
                last_act = None
        if dvh is not None and geo:          # [Bản 26] không thấy mặt đất/ảnh xấu: giữ nguyên trạng thái lòng đường
            cau_tt = ttd.update(t_vid, dang_o_duong)
            if cau_tt and speaker.say(t_vid, cau_tt):
                ttd.da_bao(t_vid)
                said_text, said_until = {'dang_o_duong': 'Đang ở lòng đường, chú ý xe', 'len_via_he': 'Đã lên vỉa hè'}[cau_tt], t_vid + 2.0
                spoken_log.append((t_vid, said_text))
        cau_mu = km.update(t_vid, anh_xau or ground is None)
        if cau_mu and speaker.say(t_vid, cau_mu, chen=cau_mu == 'khong_nhin_ro'):
            if cau_mu == 'khong_nhin_ro':
                km.t_bao = t_vid
            said_text, said_until = {'khong_nhin_ro': 'Kính không nhìn rõ, hãy dùng gậy',
                                     'nhin_ro_lai': 'Kính đã nhìn rõ lại'}[cau_mu], t_vid + 2.5
            spoken_log.append((t_vid, said_text))
        elif cau_mu == 'nhin_ro_lai':
            km.mu = True        # chưa nói được (đang đọc câu khác) -> thử lại khung sau
        warn_old = old_rule(yo)
        log.append(dict(t=round(t_vid, 2), moi=warn_new, cu=warn_old, chan_doan=chan_doan, o_duong=bool(dang_o_duong), mu=km.mu, xuong=xuong,
                        tho=[(o['name'], round(o['dist'], 1), round(o['v'], 1), bool(o.get('vok')), round(o['gap'], 2))
                             for o in dyn_all]))
        stats['cu_bao'] += bool(warn_old)
        stats['ca_hai'] += bool(warn_old) and bool(warn_new)
        stats['khong_dat'] += ground is None

        # vẽ
        vis = img.copy()
        if r is not None and r.masks is not None:
            for c, m in zip(r.boxes.cls, r.masks.data):
                mm = cv2.resize((m.cpu().numpy() > 0.5).astype(np.uint8), (W, H)) > 0
                col = np.array([0, 200, 0]) if int(c) == 1 else np.array([255, 120, 0])
                vis[mm] = (0.6 * vis[mm] + 0.4 * col).astype(np.uint8)
        if lane is not None:
            cnts, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 255, 0), 1)
        for k, o in enumerate(obs + [c for c in cand if c.get('src') == 'yoloe']):
            col = (255, 0, 255) if o.get('src') == 'yoloe' else ((0, 0, 255) if k == 0 else (0, 165, 255))
            vis[o['mask']] = (0.5 * vis[o['mask']] + 0.5 * np.array(col)).astype(np.uint8)
        dn = cv2.normalize(D, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        small = cv2.resize(cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO), (160, 90))
        vis[0:90, W - 160:W] = small
        pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        dr.rectangle([0, H - 46, W, H], fill=(0, 0, 0))
        dr.text((6, H - 44), f"MỚI (độ sâu): {warn_new or '—'}", font=font, fill=(255, 80, 80) if warn_new else (200, 200, 200))
        dr.text((6, H - 23), f"CŨ (YOLOE):   {warn_old or '—'}", font=font, fill=(255, 220, 0) if warn_old else (200, 200, 200))
        dr.text((6, 4), f"{i / vfps:5.1f}s", font=font, fill=(255, 255, 255))
        if t_vid < said_until:
            big = ImageFont.truetype(FONT, 20)
            dr.rectangle([0, 28, W - 165, 58], fill=(180, 0, 0))
            dr.text((8, 31), f"Kính nói: {said_text}", font=big, fill=(255, 255, 255))
        frame_out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        writer.write(frame_out)
        if args.show:
            cv2.imshow('Thu nghiem: do sau + YOLOE + yolo-seg', frame_out)
            # chạy theo nhịp video nếu máy xử lý nhanh hơn
            wait = max(1, int((n / (vfps / step) - (time.perf_counter() - t_start)) * 1000))
            k = cv2.waitKey(wait) & 0xFF
            if k == ord(' '):
                while (cv2.waitKey(50) & 0xFF) not in (ord(' '), ord('q')):
                    pass
            if k == ord('q'):
                print("Đã dừng theo yêu cầu (q). Video được lưu tới đoạn này.")
                break
        if n % 10 == 0:
            cv2.imwrite(os.path.join(out_dir, f'{name}_{args.nguong:g}m_{int(i / vfps):03d}s.jpg'), cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    wav = out_video + '.wav'
    speaker.write_track(wav, n / (vfps / step))
    has_audio = shutil.which('ffmpeg') and mux(out_video, wav)
    print(f"\n===== {name}: {n} khung ({args.fps:g} khung/giây video) =====")
    print(f"Độ sâu: {t_dep / n * 1000:.1f} ms/khung | nhận diện ({args.detector}): {t_yo / n * 1000:.1f} ms/khung | "
          f"tất cả model: {t_all / n * 1000:.1f} ms/khung (~{n / t_all:.1f} khung/giây)")
    with open(os.path.join(out_dir, f'{name}_log{args.hau_to}.json'), 'w') as f:
        json.dump(dict(video=name, khong_gop=args.khong_gop, yoloe=args.yoloe, ms_khung=t_all / n * 1000,
                       khung=log), f, ensure_ascii=False)
    cd = collections.Counter(c['ly_do'] for fr in log for c in fr['chan_doan'])
    print("Vật động trong hành lang -- độ sâu thấy/không thấy (số lượt khung):", dict(cd))
    print(f"Không khớp được mặt đất: {stats['khong_dat']} khung | chiều cao camera dùng cuối cùng: {H_CAM:.2f} m "
          f"({len(hc.mau)} mẫu người)")
    print(f"CÁCH MỚI cảnh báo: {stats['moi_bao']} khung ({stats['moi_bao'] / n * 100:.0f}%) "
          f"| có tên {stats['moi_co_ten']}, không tên {stats['moi_khong_ten']}")
    print(f"CÁCH CŨ cảnh báo:  {stats['cu_bao']} khung ({stats['cu_bao'] / n * 100:.0f}%)")
    print(f"Cả hai cùng báo:   {stats['ca_hai']} khung")
    print("Tên vật cách mới hay báo:", dict(names.most_common(10)))
    dur = n / (vfps / step)
    print(f"\nKính NÓI {len(spoken_log)} câu trong {dur:.0f}s (~{len(spoken_log) / dur * 60:.1f} câu/phút):")
    for t, txt in spoken_log:
        print(f"  {t:6.1f}s  🔊 {txt}")
    print(f"\nVideo {'CÓ TIẾNG' if has_audio else '(không ghép được tiếng)'}: {out_video}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
