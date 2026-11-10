# khoi_dong_dan_duong.py
# -*- coding: utf-8 -*-
"""
Tạo + làm nóng bộ dẫn đường lúc máy chủ khởi động (trước đây tạo khi kính nối lần đầu).

Lỗi gốc (đo 2026-10-26, app_main + IMG_7358): BlindPathNavigator tạo trên vòng sự kiện mất 4,2 s -> máy chủ đứng
(mic/lệnh không vào được); khung đầu qua process_frame mất 5 s (khởi động CUDA/model), khung sau 0,12 s.
"""
import gc
import time

import numpy as np


def dong_bang_bo_nho() -> int:
    """Gọi sau khi nạp xong model. GC thế hệ 2 quét mọi đối tượng trong tiến trình và giữ khóa GIL: ~784.000 đối tượng
    của torch/transformers làm nó mất 256–333 ms, máy chủ đứng hình ~0,5 s cứ ~40 s một lần (đo 2026-11-06). Các đối
    tượng này sống suốt phiên, không cần quét: đóng băng thì GC chỉ còn 11–27 ms. Trả số đối tượng đã đóng băng."""
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()


def tao_bo_dan_duong(seg_model, obstacle_detector):
    """(blind_path_navigator, cross_street_navigator, orchestrator) hoặc None khi chưa có model phân đoạn."""
    if seg_model is None:
        return None
    from navigation_master import NavigationMaster
    from workflow_blindpath import BlindPathNavigator
    from workflow_crossstreet import CrossStreetNavigator
    bp = BlindPathNavigator(seg_model, obstacle_detector)
    cs = CrossStreetNavigator(seg_model=seg_model, coco_model=None, obs_model=None)
    return bp, cs, NavigationMaster(bp, cs)


def lam_nong_phan_vung(nav, cao: int = 480, rong: int = 640):
    """Khung xám không có mặt đất nên module vật cản bỏ qua UperNet: gọi riêng, nếu không lần chạy đầu của nó (1,3 s)
    rơi vào khung thật đầu tiên, hình đứng ngay lúc bật dẫn đường (đo 2026-11-06). Gọi sau khi khung xám đã qua module."""
    ne = getattr(nav, "ne_vat_can", None)
    if ne is not None and getattr(ne, "dvh", None) is not None:
        ne.dvh(np.full((cao, rong, 3), 128, np.uint8), co=(ne.H, ne.W))


def lam_nong(master, cao: int = 480, rong: int = 640):
    """Chạy một khung xám qua process_frame ở chế độ dẫn đường rồi reset về IDLE. Trả số giây, None khi lỗi."""
    t = time.time()
    try:
        master.start_blind_path_navigation()
        master.process_frame(np.full((cao, rong, 3), 128, np.uint8))
        lam_nong_phan_vung(getattr(master, "blind", None), cao, rong)
        return time.time() - t
    except Exception as e:
        print(f"[KHỞI ĐỘNG] Làm nóng bộ dẫn đường lỗi (bỏ qua): {e}")
        return None
    finally:
        master.reset()
