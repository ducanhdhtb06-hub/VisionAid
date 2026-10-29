# app/cloud/obstacle_detector_client.py (file mới)
import logging
import os
import cv2
import numpy as np
import torch
from threading import Semaphore
from contextlib import contextmanager
from ultralytics import YOLOE
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --- Cấu hình GPU/CPU & AMP (chuyển từ workflow blindpath sang, giữ nhất quán) ---
DEVICE = os.getenv("AIGLASS_DEVICE", "cuda:0")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    logger.warning(f"AIGLASS_DEVICE={DEVICE} nhưng không phát hiện CUDA, sẽ quay về CPU")
    DEVICE = "cpu"
IS_CUDA = DEVICE.startswith("cuda")

AMP_POLICY = os.getenv("AIGLASS_AMP", "bf16").lower()
if AMP_POLICY not in ("bf16", "fp16", "off"):
    AMP_POLICY = "bf16"
AMP_DTYPE = torch.bfloat16 if AMP_POLICY == "bf16" else (torch.float16 if AMP_POLICY == "fp16" else None)

# --- Giới hạn đồng thời GPU (chuyển từ workflow blindpath sang, giữ nhất quán) ---
GPU_SLOTS = int(os.getenv("AIGLASS_GPU_SLOTS", "2"))
_gpu_slots = Semaphore(GPU_SLOTS)

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


@contextmanager
def gpu_infer_slot():
    """Quản lý thống nhất giới hạn đồng thời GPU + inference_mode + AMP autocast"""
    with _gpu_slots:
        if IS_CUDA and AMP_POLICY != "off":
            # Giao diện mới: torch.amp.autocast(device_type='cuda', dtype=...)
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
                yield
        else:
            with torch.inference_mode():
                yield


class ObstacleDetectorClient:
    def __init__(self, model_path: str = 'models/yoloe-11l-seg.pt'):
        self.model = None
        self.whitelist_embeddings = None
        self.WHITELIST_CLASSES = [
            'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'animal', 'scooter', 'stroller', 'dog',
            'pole', 'post', 'column', 'pillar', 'stanchion', 'bollard', 'utility pole',
            'telegraph pole', 'light pole', 'street pole', 'signpost', 'support post',
            'vertical post', 'bench', 'chair', 'potted plant', 'hydrant', 'cone', 'stone', 'box'
        ]
        try:
            logger.info("Đang nạp model vật cản YOLOE...")
            self.model = YOLOE(model_path)
            self.model.to(DEVICE)
            self.model.fuse()
            logger.info(f"Nạp model vật cản YOLOE thành công, thiết bị dùng: {DEVICE}")

            logger.info("Đang tính trước đặc trưng văn bản danh sách trắng cho YOLOE...")
            if IS_CUDA and AMP_DTYPE is not None:
                with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    self.whitelist_embeddings = self.model.get_text_pe(self.WHITELIST_CLASSES)
            else:
                self.whitelist_embeddings = self.model.get_text_pe(self.WHITELIST_CLASSES)
            logger.info("Đã tính xong đặc trưng YOLOE.")
        except Exception as e:
            logger.error(f"Nạp model YOLOE hoặc tính đặc trưng thất bại: {e}", exc_info=True)
            raise
    def tensor_to_numpy_mask(mask_tensor):
        """Chuyển an toàn các loại tensor thành mặt nạ numpy"""
        # Xử lý các kiểu dữ liệu khác nhau
        if mask_tensor.dtype in (torch.bfloat16, torch.float16):
            mask_tensor = mask_tensor.float()
        
        # Chuyển sang numpy
        mask = mask_tensor.cpu().numpy()
        
        # Đảm bảo là mặt nạ nhị phân
        if mask.max() <= 1.0:
            mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask = mask.astype(np.uint8)
        
        return mask 
    def detect(self, image: np.ndarray, path_mask: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Dùng danh sách trắng làm gợi ý để tìm vật cản.
        Nếu có path_mask thì lọc không gian liên quan đến đường đi.
        Nếu path_mask là None thì phát hiện toàn cục.
        """
        if self.model is None:
            return []

        H, W = image.shape[:2]
        try:
            self.model.set_classes(self.WHITELIST_CLASSES, self.whitelist_embeddings)
        except Exception as e:
            logger.error(f"Đặt gợi ý YOLOE thất bại: {e}")
            return []

        conf_thr = float(os.getenv("AIGLASS_OBS_CONF", "0.25"))
        with gpu_infer_slot():
            results = self.model.predict(image, verbose=False, conf=conf_thr)

        if not (results and results[0].masks):
            return []

        # --- Lọc và hậu xử lý (logic nhất quán với workflow blindpath) ---
        final_obstacles = []
        num_masks = len(results[0].masks.data)
        num_boxes = len(results[0].boxes.cls) if getattr(results[0].boxes, "cls", None) is not None else 0

        for i, mask_tensor in enumerate(results[0].masks.data):
            if i >= num_boxes: continue

            # [Sửa lỗi] Xử lý mặt nạ kiểu BFloat16
            # Chuyển sang float32 trước, tránh việc numpy không hỗ trợ BFloat16
            if mask_tensor.dtype == torch.bfloat16:
                mask_tensor = mask_tensor.float()
            
            # Chuyển sang mảng numpy
            mask = mask_tensor.cpu().numpy()
            
            # Xử lý mặt nạ xác suất (giá trị 0-1) hoặc mặt nạ nhị phân
            if mask.max() <= 1.0:
                # Mặt nạ xác suất, cần nhị phân hóa
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                # Đã là mặt nạ nhị phân
                mask = mask.astype(np.uint8)
            
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            area = np.sum(mask > 0)

            # Lọc theo kích thước: vật thể quá lớn (như cả mặt đất) thường là nhận diện nhầm
            if (area / (H * W)) > 0.7: continue

            # Lọc theo không gian: nếu có path_mask thì chỉ giữ vật cản nằm trên đường đi
            if path_mask is not None:
                intersection_area = np.sum(cv2.bitwise_and(mask, path_mask) > 0)
                # Phải chồng lấp đủ với đường đi
                if intersection_area < 100 or (intersection_area / area) < 0.01:
                    continue

            cls_id = int(results[0].boxes.cls[i])
            class_names_map = results[0].names
            class_name = "Unknown"
            if isinstance(class_names_map, dict):
                # Nếu là dict, dùng phương thức .get()
                class_name = class_names_map.get(cls_id, "Unknown")
            elif isinstance(class_names_map, list) and 0 <= cls_id < len(class_names_map):
                # Nếu là list, lấy an toàn qua chỉ số
                class_name = class_names_map[cls_id]

            # Tính các chỉ số khoảng cách
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0: continue

            final_obstacles.append({
                'name': class_name.strip(),
                'mask': mask,
                'area': area,
                'area_ratio': area / (H * W),
                'center_x': np.mean(x_coords),
                'center_y': np.mean(y_coords),
                'bottom_y_ratio': np.max(y_coords) / H
            })

        return final_obstacles