# workflow_blindpath.py
# -*- coding: utf-8 -*-
"""
Workflow dẫn đường đường dẫn - bản sạch
Đã bỏ mọi phụ thuộc Redis, Celery, có thể tích hợp trực tiếp vào mọi ứng dụng Python
"""
import os
import time
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import torch  # Thêm dòng này
from obstacle_detector_client import ObstacleDetectorClient
from audio_player import play_voice_text  # Mới
from crosswalk_awareness import CrosswalkAwarenessMonitor, split_combined_voice  # Nhận biết vạch qua đường
# Thử nhập Pillow để hiển thị chữ Trung
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageDraw, ImageFont = None, None, None

logger = logging.getLogger(__name__)

# ========== Định nghĩa hằng số trạng thái ==========
T_THEO_LAU = 5.0   # thấy vạch trong 5 s gần nhất: vẫn đang theo vạch (xe đè che mất vạch vài giây khi tới gần)
STATE_ONBOARDING = "ONBOARDING"
STATE_NAVIGATING = "NAVIGATING"
STATE_MANEUVERING_TURN = "MANEUVERING_TURN"
STATE_AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
STATE_LOCKING_ON = "LOCKING_ON"

# Các bước con của ONBOARDING
ONBOARDING_STEP_ROTATION = "ROTATION"
ONBOARDING_STEP_TRANSLATION = "TRANSLATION"

# Các bước con khi rẽ
MANEUVER_STEP_1_ISSUE_COMMAND = "ISSUE_COMMAND"
MANEUVER_STEP_2_WAIT_FOR_SHIFT = "WAIT_FOR_SHIFT"
MANEUVER_STEP_3_ALIGN_ON_NEW_PATH = "ALIGN_ON_NEW_PATH"

# Định nghĩa màu (định dạng BGR)
VIS_COLORS = {
    "blind_path": (0, 255, 0),      # Xanh lá
    "obstacle": (0, 0, 255),        # Đỏ
    "crosswalk": (0, 165, 255),     # Cam
    "centerline": (0, 255, 255),     # Vàng
    "target_point": (255, 0, 0),     # Xanh dương
    "turn_point": (128, 0, 128),     # Tím
    "pulse_effect": (100, 100, 255)  # Đỏ nhạt
}

# Ánh xạ tên vật cản
_OBSTACLE_NAME_CN = {
    'person': 'người',
    'bicycle': 'xe đạp',
    'car': 'xe',
    'motorcycle': 'xe máy',
    'bus': 'xe buýt',
    'truck': 'xe tải',
    'animal': 'động vật',
    'scooter': 'xe điện',
    'stroller': 'xe đẩy em bé',
    'dog': 'chó',
}

# Danh sách tên lớp động
DYNAMIC_CLASS_NAMES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'animal', 'dog'}

@dataclass
class ProcessingResult:
    """Lớp dữ liệu kết quả xử lý"""
    guidance_text: str  # Văn bản hướng dẫn bằng giọng nói
    visualizations: List[Dict[str, Any]]  # Danh sách phần tử trực quan
    annotated_image: Optional[np.ndarray] = None  # Ảnh đã chú thích
    state_info: Dict[str, Any] = None  # Thông tin trạng thái
    
    def __post_init__(self):
        if self.state_info is None:
            self.state_info = {}


class BlindPathNavigator:
    """Bộ dẫn đường đường dẫn - phiên bản không phụ thuộc bên ngoài"""
    
    def __init__(self, yolo_model=None, obstacle_detector=None):
        """
        Khởi tạo bộ dẫn đường
        :param yolo_model: model phân đoạn YOLO (tùy chọn)
        :param obstacle_detector: bộ phát hiện vật cản (tùy chọn)
        """
        self.yolo_model = yolo_model
        self.obstacle_detector = obstacle_detector
        
        # Biến trạng thái
        self.current_state = STATE_ONBOARDING
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None
        

        # Tham số theo dõi optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Tham số phát hiện đặc trưng
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.05,
            minDistance=10,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04
        )
        
        # Bộ đệm điểm theo dõi optical flow
        self.flow_points = {}  # {mask_type: points}
        self.flow_grace = {}   # {mask_type: grace_count}
        self.FLOW_GRACE_MAX = 3  # [Sửa đổi] Giảm từ 8 khung xuống 3 khung, xóa nhanh phần optical flow còn sót
        
        # Bộ đệm làm mượt đường tâm
        self.centerline_history = []  # Dữ liệu đường tâm lịch sử
        self.centerline_history_max = 5  # Giữ 5 khung gần nhất để làm mượt
        
        # Bộ đệm làm mượt hệ số đa thức
        self.poly_coeffs_history = []  # Hệ số đa thức lịch sử
        self.poly_coeffs_history_max = 8  # Giữ hệ số của 8 khung gần nhất để làm mượt

        # Bộ theo dõi phát hiện khúc cua
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
        
        # Thời gian hồi sau khi rẽ
        self.turn_cooldown_frames = 0
        self.TURN_COOLDOWN_DURATION = 50
        
        # Liên quan đến tránh vật cản
        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None
        
        # Theo dõi vạch qua đường
        self.crosswalk_tracker = {
            'stage': 'not_detected',
            'consecutive_frames': 0,
            'last_area_ratio': 0.0,
            'last_bottom_y_ratio': 0.0,
            'last_center_x_ratio': 0.5,
            'position_announced': False,
            'alignment_status': 'not_aligned',
            'last_seen_frame': 0,
            'last_angle': 0.0
        }
        
        # Bộ đếm khung
        self.frame_counter = 0
        
        # Cấu hình nhắc đi thẳng - hỗ trợ biến môi trường
        self.guide_interval = float(os.getenv("AIGLASS_STRAIGHT_INTERVAL", "4.0"))  # Khoảng cách thông báo (giây)
        self.last_guide_time = 0.0
        self.straight_continuous_mode = os.getenv("AIGLASS_STRAIGHT_CONTINUOUS", "1") == "1"  # Chế độ thông báo liên tục
        self.straight_repeat_limit = int(os.getenv("AIGLASS_STRAIGHT_LIMIT", "2"))  # Số lần tối đa ở chế độ giới hạn
        self.straight_repeat_count = 0
        
        # [Mới] Cấu hình thông báo liên tục lệnh phương hướng
        self.direction_interval = float(os.getenv("AIGLASS_DIRECTION_INTERVAL", "3.0"))  # Khoảng cách lệnh phương hướng (giây)
        self.last_direction_time = 0.0
        self.last_direction_message = ""
        
        # In thông tin cấu hình
        logger.info(f"[BlindPath] Cấu hình thông báo đi thẳng: khoảng cách={self.guide_interval}giây, "
                   f"chế độ liên tục={self.straight_continuous_mode}, "
                   f"giới hạn số lần={self.straight_repeat_limit}")
        logger.info(f"[BlindPath] Cấu hình thông báo phương hướng: khoảng cách={self.direction_interval}giây")

        # Biến bộ đệm
        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_any_speech_time = 0
        
        # Cờ trạng thái chuẩn bị tới vạch qua đường
        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0
        
        # Giọng nói vật cản đang chờ thông báo
        self.pending_obstacle_voice = None
        
        # Nhận diện đèn giao thông
        self.traffic_light_detector = None
        self.init_traffic_light_detector()
        self.traffic_light_history = deque(maxlen=8)  # Dùng để bỏ phiếu đa số
        self.last_traffic_light_state = "unknown"
        self.green_light_announced = False
        
        # Cài đặt ngưỡng
        self.CLASS_CONF_THRESHOLDS = {
            1: 0.20,  # blind_path
            0: 0.30   # crosswalk
        }
        
        # Ngưỡng dẫn đường
        # Ngưỡng dẫn đường
        self.ONBOARDING_ALIGN_THRESHOLD_RATIO = 0.1
        self.VP_FIT_ERROR_THRESHOLD = 8.0

        self.ONBOARDING_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.NAV_ORIENTATION_THRESHOLD_RAD = np.deg2rad(10)
        self.NAV_CENTER_OFFSET_THRESHOLD_RATIO = 0.15
        self.CURVATURE_PROXY_THRESHOLD = 5e-5
        
        # Ngưỡng chuyển sang vạch qua đường
        self.CROSSWALK_SWITCH_AREA_RATIO = 0.22
        self.CROSSWALK_SWITCH_BOTTOM_RATIO = 0.9
        self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES = 10
        
        # Khoảng cách phát hiện vật cản
        # Tham số tối ưu phát hiện vật cản - đọc từ biến môi trường, hỗ trợ tinh chỉnh hiệu năng
        self.OBSTACLE_DETECTION_INTERVAL = int(os.getenv("AIGLASS_OBS_INTERVAL", "15"))  # Mặc định cứ 5 khung phát hiện một lần
        self.OBSTACLE_CACHE_DURATION_FRAMES = int(os.getenv("AIGLASS_OBS_CACHE_FRAMES", "10"))  # Đệm 10 khung

        # Module né vật cản mới (độ sâu + D-FINE, có lệnh né) - MẶC ĐỊNH từ 2026-10-24
        # (bộ chấm 6 video: báo nhầm 4% so với 26% của luật cũ). AIGLASS_OBS_ENGINE=cu để quay lại luật cũ.
        self.ne_vat_can = None
        if os.getenv("AIGLASS_OBS_ENGINE", "depth").lower() not in ("cu", "old"):
            try:
                from ne_vat_can_do_sau import NeVatCanDoSau
                self.ne_vat_can = NeVatCanDoSau()
            except Exception as e:
                logger.error(f"[NE] Không khởi tạo được module né vật cản độ sâu, dùng cách cũ: {e}", exc_info=True)

        # Dẫn theo vạch bằng điểm tụ + bề rộng vạch (dan_theo_vach.py) - MẶC ĐỊNH từ 2026-10-25.
        # Cách cũ bảo rẽ ngược chiều 98% khi người lệch khỏi vạch. AIGLASS_DAN_VACH=cu để quay lại cách cũ.
        self.dan_vach = None
        if os.getenv("AIGLASS_DAN_VACH", "moi").lower() not in ("cu", "old"):
            from dan_theo_vach import DanTheoVach
            self.dan_vach = DanTheoVach()
            self.current_state = STATE_NAVIGATING   # luật "dịch" đã lo việc lên vạch, bỏ bước ONBOARDING

        # Vật chắn trên vạch (spec 2026-10-27-vong-vat-can-tren-vach): đang trên vạch thì im vật cạnh vạch, vật chắn
        # vạch -> "Xe máy chắn vạch bên phải. Đi sang trái.", im lúc đi vòng, "Đã về vạch.". AIGLASS_VAT_CHAN_VACH=0: như cũ.
        self.bo_vat_chan = None
        if self.dan_vach is not None and os.getenv("AIGLASS_VAT_CHAN_VACH", "1") != "0":
            from vat_chan_vach import BoVatChan, OnDinh, NhoVatChan
            self.bo_vat_chan, self.on_dinh_vat_chan, self.vat_chan_on_dinh = BoVatChan(), OnDinh(), False
            self.nho_vat_chan = NhoVatChan()

        # Quản lý thông báo vật cản
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.obstacle_speech_cooldown = 5.0  # Cùng một vật cản không thông báo lặp trong 3 giây
        
        # Tham số ổn định hóa mặt nạ (đã tắt ngoại suy optical flow, các tham số này không còn dùng)
        self.MASK_STAB_MIN_AREA = int(os.getenv("AIGLASS_MASK_MIN_AREA", "1500"))
        self.MASK_STAB_KERNEL = int(os.getenv("AIGLASS_MASK_MORPH", "3"))
        self.MASK_MISS_TTL = 0  # [Sửa thành 0] Tắt ngoại suy optical flow, hoàn toàn thời gian thực
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0
        
        # Tham số theo dõi optical flow
        self.flow_iou_threshold = 0.3  # IoU thấp hơn giá trị này thì khởi tạo lại điểm optical flow
        
        # [Mới] Khoảng cách phát hiện YOLO của đường dẫn
        self.BLINDPATH_DETECTION_INTERVAL = int(os.getenv("AIGLASS_BLINDPATH_INTERVAL", "8"))  # Cứ 2 khung phát hiện một lần
        self.last_blindpath_detection_frame = 0
        self.last_blindpath_mask = None
        self.last_crosswalk_mask = None
        
        # [Mới] Bộ giám sát nhận biết vạch qua đường
        self.crosswalk_monitor = CrosswalkAwarenessMonitor()
        logger.info("[BlindPath] Đã khởi tạo bộ giám sát nhận biết vạch qua đường")
        logger.info(f"[BlindPath] Khoảng cách phát hiện đường dẫn: mỗi {self.BLINDPATH_DETECTION_INTERVAL} khung")
    
    def init_traffic_light_detector(self):
        """Khởi tạo bộ nhận diện đèn giao thông"""
        try:
            # Trước tiên thử dùng model YOLO để nhận diện đèn giao thông
            self.traffic_light_yolo = None
            # Nếu bạn có model đèn giao thông riêng thì nạp ở đây
            # self.traffic_light_yolo = YOLO('path/to/traffic_light_model.pt')
        except Exception as e:
            logger.info(f"Chưa nạp model YOLO đèn giao thông: {e}")
    
    def detect_traffic_light(self, image: np.ndarray) -> str:
        """Phát hiện trạng thái đèn giao thông
        Trả về: 'red', 'green', 'yellow', 'unknown'
        """
        # Chế độ mô phỏng (dùng để kiểm thử)
        if os.getenv("AIGLASS_SIMULATE_TRAFFIC_LIGHT", "0") == "1":
            # Mô phỏng đèn giao thông thay đổi theo số khung
            cycle = (self.frame_counter // 100) % 3
            if cycle == 0:
                return "red"
            elif cycle == 1:
                return "yellow"
            else:
                return "green"
        
        # Nếu có model YOLO thì ưu tiên dùng
        if self.traffic_light_yolo:
            try:
                results = self.traffic_light_yolo.predict(image, verbose=False, conf=0.3)
                # TODO: phân tích kết quả YOLO, xác định màu đèn giao thông
                pass
            except:
                pass
        
        # Dùng nhận diện màu HSV làm phương án dự phòng
        return self._detect_traffic_light_by_color(image)
    
    def _detect_traffic_light_by_color(self, image: np.ndarray) -> str:
        """Phát hiện đèn giao thông dựa trên không gian màu HSV"""
        h, w = image.shape[:2]
        # Phát hiện nửa trên và phần giữa của ảnh (đèn giao thông có thể ở các độ cao khác nhau)
        roi = image[:int(h * 0.7), :]  # Mở rộng phạm vi phát hiện lên 70%
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Ảnh tăng độ sáng dùng để phát hiện (giúp phát hiện đèn giao thông tối hơn)
        hsv_bright = hsv.copy()
        hsv_bright[:, :, 2] = cv2.add(hsv_bright[:, :, 2], 30)  # Tăng độ sáng
        
        # Định nghĩa dải màu (tham số đã tối ưu)
        # Đỏ (hai dải, vì màu đỏ trong HSV vắt qua 0 độ)
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Xanh lá (chỉnh sang dải rộng hơn để thích ứng các loại ánh sáng khác nhau)
        lower_green = np.array([40, 60, 60])
        upper_green = np.array([90, 255, 255])
        
        # Vàng
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # Tạo mặt nạ (phát hiện đồng thời trên ảnh gốc và ảnh tăng sáng)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red1_bright = cv2.inRange(hsv_bright, lower_red1, upper_red1)
        mask_red2_bright = cv2.inRange(hsv_bright, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), 
                                 cv2.bitwise_or(mask_red1_bright, mask_red2_bright))
        
        mask_green = cv2.bitwise_or(cv2.inRange(hsv, lower_green, upper_green),
                                   cv2.inRange(hsv_bright, lower_green, upper_green))
        mask_yellow = cv2.bitwise_or(cv2.inRange(hsv, lower_yellow, upper_yellow),
                                    cv2.inRange(hsv_bright, lower_yellow, upper_yellow))
        
        # Phép toán hình thái học để khử nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        
        # Tính diện tích từng màu
        area_red = cv2.countNonZero(mask_red)
        area_green = cv2.countNonZero(mask_green)
        area_yellow = cv2.countNonZero(mask_yellow)
        
        # Đặt ngưỡng diện tích tối thiểu (giảm ngưỡng để phát hiện nhạy hơn)
        min_area = 30  # Giảm ngưỡng thêm nữa
        
        # Thêm thông tin gỡ lỗi chi tiết hơn
        if hasattr(self, 'frame_counter') and self.frame_counter % 30 == 0:
            logger.info(f"[HSV] Đỏ:{area_red}, Xanh:{area_green}, Vàng:{area_yellow}")
            # Lưu ảnh gỡ lỗi
            if os.getenv("AIGLASS_DEBUG_TRAFFIC_LIGHT", "0") == "1":
                debug_dir = "traffic_light_debug"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_roi.jpg", roi)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_red.jpg", mask_red)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_green.jpg", mask_green)
                cv2.imwrite(f"{debug_dir}/frame_{self.frame_counter}_yellow.jpg", mask_yellow)
        
        # Xác định màu (ưu tiên: xanh > đỏ > vàng)
        if area_green > min_area and area_green > area_red * 0.8:  # Ưu tiên đèn xanh
            return "green"
        elif area_red > min_area and area_red > area_green:
            return "red"
        elif area_yellow > min_area:
            return "yellow"
        else:
            return "unknown"
    
    def _get_voice_priority(self, guidance_text):
        """Lấy mức ưu tiên của lệnh giọng nói
        Ưu tiên: vật cản(100) > rẽ/dịch ngang(50) > giữ đi thẳng(10)
        """
        if not guidance_text:
            return 0
        
        # Thông báo vật cản - ưu tiên cao nhất
        gt = guidance_text.lower()
        obstacle_keywords = ['phía trước có', 'bên trái có', 'bên phải có', 'dừng lại', 'chú ý tránh', 'vật cản']
        for keyword in obstacle_keywords:
            if keyword in gt:
                return 100
        
        # Rẽ và dịch ngang - ưu tiên trung bình
        direction_keywords = ['rẽ trái', 'rẽ phải', 'dịch trái', 'dịch phải', 'sang trái', 'sang phải', 'dịch sang', 'chỉnh nhẹ']
        for keyword in direction_keywords:
            if keyword in gt:
                return 50
        
        # Giữ hướng đi thẳng - ưu tiên thấp nhất
        if 'giữ hướng đi thẳng' in gt or 'tiếp tục tiến lên' in gt or 'đúng hướng,' in gt:
            return 10
        
        # Các lệnh khác - ưu tiên trung bình mặc định
        return 30

    def process_frame(self, image: np.ndarray) -> ProcessingResult:
        """
        Xử lý một khung ảnh
        :param image: ảnh định dạng BGR
        :return: kết quả xử lý
        """
        self.frame_counter += 1
        
        # Cập nhật thời gian hồi
        if self.turn_cooldown_frames > 0:
            self.turn_cooldown_frames -= 1
        
        image_height, image_width = image.shape[:2]
        image_center_x = image_width / 2
        
        # Chuyển sang ảnh xám
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Danh sách phần tử trực quan
        frame_visualizations = []
        guidance_text = ""
        
        # 1. [Sửa thành thời gian thực] Mỗi khung đều chạy phát hiện YOLO, không dùng bộ đệm
        blind_path_mask, crosswalk_mask = self._detect_path_and_crosswalk(image)
        
        # [Gỡ lỗi] Kiểm tra kết quả phát hiện YOLO
        if self.frame_counter % 30 == 0:  # Cứ 30 khung in một lần
            has_blind = blind_path_mask is not None and np.sum(blind_path_mask > 0) > 0
            has_cross = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(f"[Phát hiện YOLO] Frame={self.frame_counter}, đường dẫn={'có' if has_blind else 'không'}, "
                       f"vạch qua đường={'có' if has_cross else 'không'}")
            if has_cross:
                cross_area = np.sum(crosswalk_mask > 0) / crosswalk_mask.size
                logger.info(f"[Phát hiện YOLO] Diện tích gốc của vạch qua đường: {cross_area*100:.2f}%")
        
        # [Sửa đổi] Giữ kết quả phát hiện vạch qua đường, dùng cho nhận biết vạch qua đường
        # crosswalk_mask = None  # không còn ép về None
        
        # 2. [Tắt ổn định hóa mặt nạ và ngoại suy optical flow] Dùng thẳng kết quả phát hiện thời gian thực
        # Không còn dùng ngoại suy optical flow, cập nhật hoàn toàn thời gian thực: có là có, không là không
        # blind_path_mask và crosswalk_mask dùng thẳng kết quả phát hiện ở trên
        crosswalk_mask_before_stabilize = crosswalk_mask
        
        # [Gỡ lỗi] Kiểm tra kết quả sau ổn định hóa
        if self.frame_counter % 30 == 0 and crosswalk_mask_before_stabilize is not None:
            after_stab = crosswalk_mask is not None and np.sum(crosswalk_mask > 0) > 0
            logger.info(f"[Ổn định mặt nạ] Vạch qua đường sau ổn định: {'có' if after_stab else 'không (đã bị lọc)'}")
        
        # [Mới] 3. Phát hiện vật cản suốt quá trình
        # Bất kể ở trạng thái nào đều phát hiện vật cản
        logger.info(f"[Frame {self.frame_counter}] Bắt đầu phát hiện vật cản...")
        
        # Dùng chiến lược bộ đệm, nhưng đảm bảo mọi vật cản đều được trực quan hóa
        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image, blind_path_mask)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
            logger.info(f"[Frame {self.frame_counter}] Đã thực hiện phát hiện vật cản mới, phát hiện được {len(detected_obstacles)} vật cản")
        else:
            if self.frame_counter - self.last_obstacle_detection_frame < self.OBSTACLE_CACHE_DURATION_FRAMES:
                detected_obstacles = self.last_detected_obstacles
                logger.info(f"[Frame {self.frame_counter}] Dùng dữ liệu vật cản trong bộ đệm, tổng cộng {len(detected_obstacles)} vật cản")
            else:
                detected_obstacles = []
                logger.info(f"[Frame {self.frame_counter}] Bộ đệm hết hạn, không có dữ liệu vật cản")
        
        # Thêm trực quan hóa cho tất cả vật cản (không chỉ vật cản ở gần)
        for i, obs in enumerate(detected_obstacles):
            logger.info(f"  Vật cản {i+1}: {obs.get('name', 'unknown')}, "
                    f"bottom_y_ratio={obs.get('bottom_y_ratio', 0):.2f}, "
                    f"area_ratio={obs.get('area_ratio', 0):.3f}, "
                    f"Vị trí=({obs.get('center_x', 0):.0f}, {obs.get('center_y', 0):.0f})")
            self._add_obstacle_visualization(obs, frame_visualizations)
        
        # [Mới] Kiểm tra vật cản ở gần và đặt giọng nói
        if self.ne_vat_can is not None:
            # module mới tự quyết định câu nói (lệnh né / đang tới / cầu thang), chạy mỗi khung
            try:
                dang_qua = crosswalk_mask is not None and np.sum(crosswalk_mask[int(crosswalk_mask.shape[0] * 0.6):] > 0) > 0.05 * crosswalk_mask.size * 0.4
                if self.dan_vach is not None and self.dan_vach.dang_theo_vach(time.time()):
                    dang_qua = True     # đang theo vạch: chỉ báo dừng, không bao giờ bảo "dịch" ra khỏi vạch (spec 7.1)
                # người đang đứng trên vạch (vỉa hè): module vật cản không báo "Đang ở lòng đường"
                self.ne_vat_can.tren_vach = self.dan_vach is not None and self.dan_vach.da_tren_vach(time.time())
                cau = self.ne_vat_can.step(image, dang_qua_duong=dang_qua,
                                           chi_bao=self._chi_bao_vat_chan(time.time()))
                self.pending_obstacle_voice = cau
                if cau:
                    logger.info(f"[NE] {cau}")
            except Exception as e:
                logger.error(f"[NE] Lỗi module né vật cản độ sâu: {e}", exc_info=True)
                self._check_and_set_obstacle_voice(detected_obstacles)
        else:
            self._check_and_set_obstacle_voice(detected_obstacles)
        
        # [Mới] Xử lý nhận biết vạch qua đường
        # Trước hết kiểm tra trạng thái crosswalk_mask
        if crosswalk_mask is not None:
            cross_pixels = np.sum(crosswalk_mask > 0)
            if cross_pixels > 0:
                logger.info(f"[Vạch qua đường] Truyền vào monitor: pixels={cross_pixels}, area={cross_pixels/crosswalk_mask.size*100:.2f}%")
            else:
                logger.info(f"[Vạch qua đường] crosswalk_mask toàn 0, không có vạch qua đường")
        else:
            if self.frame_counter % 30 == 0:
                logger.info(f"[Vạch qua đường] crosswalk_mask là None")
        
        crosswalk_guidance = self.crosswalk_monitor.process_frame(crosswalk_mask, blind_path_mask)
        if crosswalk_guidance:
            logger.info(f"[Nhận biết vạch qua đường] Kết quả phát hiện: area={crosswalk_guidance.get('area', 0):.3f}, "
                       f"should_broadcast={crosswalk_guidance.get('should_broadcast', False)}, "
                       f"voice={crosswalk_guidance.get('voice_text', 'None')}")
        if crosswalk_guidance and crosswalk_guidance['should_broadcast']:
            # Đưa giọng nói vạch qua đường vào danh sách chờ thông báo (qua cơ chế pending)
            if not hasattr(self, 'pending_crosswalk_voice'):
                self.pending_crosswalk_voice = None
            self.pending_crosswalk_voice = crosswalk_guidance
            logger.info(f"[Giọng nói vạch qua đường] Đã đặt giọng nói chờ thông báo: {crosswalk_guidance['voice_text']}, ưu tiên{crosswalk_guidance['priority']}")
        
        # [Mới] Thêm trực quan hóa vạch qua đường
        if crosswalk_mask is not None:
            # Tính dữ liệu trực quan
            total_pixels = crosswalk_mask.size
            crosswalk_pixels = np.sum(crosswalk_mask > 0)
            area_ratio = crosswalk_pixels / total_pixels
            
            y_coords, x_coords = np.where(crosswalk_mask > 0)
            if len(y_coords) > 0:
                center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
                center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]
                has_occlusion = self.crosswalk_monitor._check_occlusion(crosswalk_mask, blind_path_mask)
                
                # Lấy dữ liệu trực quan
                viz_data = self.crosswalk_monitor.get_visualization_data(
                    crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion
                )
                
                # Thêm trực quan hóa mask vạch qua đường
                self._add_mask_visualization(crosswalk_mask, frame_visualizations, 
                                            "crosswalk_mask", viz_data['stage_color'])
                
                # Thêm trực quan hóa thông tin phát hiện vạch qua đường
                self._add_crosswalk_info_visualization(viz_data, image_height, image_width, 
                                                      frame_visualizations)
        
        # [Đã tắt] 4. Cập nhật bộ theo dõi vạch qua đường - dẫn đường đường dẫn không còn chuyển sang vạch qua đường
        # self._update_crosswalk_tracker(crosswalk_mask, image_height, image_width)
        
        # 5. Thêm trực quan hóa đường đi
        # [Khôi phục] Trực quan hóa mask đường dẫn
        self._add_mask_visualization(blind_path_mask, frame_visualizations, "blind_path_mask", "rgba(0, 255, 0, 0.4)")
        # [Trực quan hóa vạch qua đường do crosswalk_monitor xử lý, không thêm ở đây]
        

        # [Đã tắt] 5. Thực hiện logic dẫn đường theo trạng thái - dẫn đường đường dẫn không còn xử lý vạch qua đường
        current_stage = 'not_detected'  # Cố định là không phát hiện vạch qua đường
        # current_stage = self.crosswalk_tracker['stage']  # đã tắt
        
        # Dẫn đường đường dẫn trực tiếp, không kiểm tra trạng thái vạch qua đường
        if False:  # current_stage == 'ready':
            # Kiểm tra đã thông báo nhắc chuẩn bị chưa
            if not hasattr(self, 'crosswalk_ready_announced'):
                self.crosswalk_ready_announced = False
                self.crosswalk_ready_time = 0
            
            current_time = time.time()
            
            # Nhận diện đèn giao thông
            traffic_light_color = self.detect_traffic_light(image)
            self.traffic_light_history.append(traffic_light_color)
            
            # Thông tin gỡ lỗi
            if self.frame_counter % 30 == 0:  # Cứ 30 khung in một lần
                logger.info(f"[Nhận diện đèn giao thông] Màu hiện tại: {traffic_light_color}, lịch sử: {list(self.traffic_light_history)}")
            
            # Bỏ phiếu đa số, lấy trạng thái đèn giao thông ổn định
            if len(self.traffic_light_history) >= 3:
                color_counts = {}
                for color in self.traffic_light_history:
                    color_counts[color] = color_counts.get(color, 0) + 1
                # Lấy màu xuất hiện nhiều nhất
                stable_color = max(color_counts.items(), key=lambda x: x[1])[0]
            else:
                stable_color = "unknown"
            
            # Thêm trực quan hóa trạng thái đèn giao thông
            self._add_traffic_light_visualization(
                stable_color, frame_visualizations, image_height, image_width
            )
            
            # Quyết định thông báo giọng nói
            if not self.crosswalk_ready_announced:
                guidance_text = "Đã căn hướng, chuẩn bị chuyển sang chế độ qua đường."
                self.crosswalk_ready_announced = True
                self.crosswalk_ready_time = current_time
            elif stable_color == "green" and not self.green_light_announced:
                guidance_text = "Đèn xanh đã ổn định, bắt đầu đi qua."
                self.green_light_announced = True
            elif stable_color == "red":
                # Khi đèn đỏ thì nhắc định kỳ
                if current_time - self.crosswalk_ready_time > 5.0:
                    guidance_text = "Đang chờ đèn xanh…"
                    self.crosswalk_ready_time = current_time
                else:
                    guidance_text = ""
            else:
                guidance_text = ""
            
            # Thêm thông tin trạng thái
            frame_visualizations.append({
                "type": "data_panel",
                "data": {
                    "Trạng thái": "Chờ qua đường",
                    "Đèn giao thông": stable_color,
                    "Lịch sử phát hiện": len(self.traffic_light_history)
                },
                "position": (25, image_height - 120)
            })
            
        elif False:  # current_stage == 'approaching':
            guidance_text = self._handle_crosswalk_approaching(
                frame_visualizations, image_height, image_width, image
            )
            
        # elif current_stage in ['far', 'not_detected']:
        else:  # Luôn thực hiện dẫn đường đường dẫn
            # [Đã tắt] Nhắc vạch qua đường
            # if current_stage == 'far' and not self.crosswalk_tracker['position_announced']:
            #     guidance_text = "Phát hiện vạch qua đường ở phía xa, tiếp tục đi thẳng."
            #     self.crosswalk_tracker['position_announced'] = True
            
            if blind_path_mask is None:
                guidance_text = ""
                if self.dan_vach is not None:
                    # khung không thấy vạch vẫn phải đưa vào luật nói (để báo "mất vạch")
                    guidance_text, _ = self.dan_vach.buoc(None, time.time(), vat=self._vat_ne())
                # [Bỏ chữ ở góc trên trái, chuyển sang bảng dữ liệu góc trên phải]
                frame_visualizations.append({
                    "type": "data_panel",
                    "data": {
                        "Trạng thái": "Chờ nhận diện đường dẫn"
                    },
                    "position": (image_width - 180, 20)
                })
            else:
                guidance_text = self._execute_state_machine(
                    blind_path_mask, image, frame_visualizations,
                    image_height, image_width, curr_gray
                )
        
        # 6. Cập nhật bộ đệm
        self.prev_gray = curr_gray
        if blind_path_mask is not None:
            self.prev_blind_path_mask = blind_path_mask.copy()
        if crosswalk_mask is not None:
            self.prev_crosswalk_mask = crosswalk_mask.copy()
        
        # Vật chắn trên vạch: câu của bộ điều khiển thay câu module vật cản (trừ xe đang lao tới)
        if self.pending_obstacle_voice and "đang tới" in self.pending_obstacle_voice:
            self.t_bao_xe_toi = time.time()
        cau_chan = self._buoc_vat_chan(time.time())
        if cau_chan:
            # xe đang lao tới nói trước; còn xe đang tới trong tầm nhìn thì chưa bảo vòng (người phải dừng; IMG_7355
            # 147 s, 29/10: "…đang tới" -> "Dừng lại." -> 0,2 s sau "Đi sang trái." về phía chiếc xe)
            con_xe_toi = any(o.get('toi') or o.get('nhanh') for o in (getattr(self.ne_vat_can, "cand", None) or []))
            # vừa báo xe đang tới < 3 s (cùng khoảng "cùng một xe" của Speaker): xe giảm tốc hết cờ "đang tới" nhưng vẫn
            # đang đi ngang (IMG_7357 67,6 -> 68,6 s: "…bên trái, đang tới" rồi "Đi sang trái.")
            vua_bao = time.time() - getattr(self, "t_bao_xe_toi", -1e9) < 3.0
            if con_xe_toi or vua_bao or (self.pending_obstacle_voice and "đang tới" in self.pending_obstacle_voice):
                self.bo_vat_chan.chua_noi()          # câu vật chắn để khung sau
                cau_chan = ""
            else:
                self.pending_obstacle_voice = cau_chan

        # [Cải tiến] Hệ thống quản lý ưu tiên giọng nói
        current_time = time.time()
        
        # Thu thập mọi lệnh giọng nói có thể
        voice_candidates = []
        
        # 1. Thêm giọng nói dẫn đường chính
        if guidance_text:
            voice_candidates.append({
                'text': guidance_text,
                'priority': self._get_voice_priority(guidance_text),
                'source': 'navigation'
            })
        
        # 2. Kiểm tra có giọng nói vật cản không (kiểm tra độc lập, đảm bảo ưu tiên cao nhất)
        if hasattr(self, 'pending_obstacle_voice'):
            if self.pending_obstacle_voice:
                voice_candidates.append({
                    'text': self.pending_obstacle_voice,
                    'priority': 100,  # Vật cản luôn có ưu tiên cao nhất
                    'source': 'obstacle'
                })
                self.pending_obstacle_voice = None  # Xóa giọng nói vật cản đã xử lý
        
        # [Mới] Kiểm tra có giọng nói vạch qua đường không
        if hasattr(self, 'pending_crosswalk_voice'):
            if self.pending_crosswalk_voice:
                voice_candidates.append({
                    'text': self.pending_crosswalk_voice['voice_text'],
                    'priority': self.pending_crosswalk_voice['priority'],
                    'source': 'crosswalk'
                })
                self.pending_crosswalk_voice = None  # Xóa giọng nói vạch qua đường đã xử lý
        
        # 3. Chọn giọng nói có ưu tiên cao nhất
        if voice_candidates:
            # Sắp xếp theo ưu tiên, lấy cái cao nhất
            voice_candidates.sort(key=lambda x: x['priority'], reverse=True)
            selected_voice = voice_candidates[0]
            final_guidance_text = selected_voice['text']
            
            # Giới hạn thông báo toàn cục (tránh bất kỳ giọng nói nào chồng lên nhau)
            MIN_SPEECH_INTERVAL = 1.2  # Hai giọng nói bất kỳ cách nhau ít nhất 0.8 giây
            # Vật cản không bị giới hạn này (module né vật cản chỉ trả câu MỘT lần cho mỗi sự kiện, chặn là mất hẳn)
            if hasattr(self, 'last_any_speech_time') and selected_voice['source'] != 'obstacle':
                if current_time - self.last_any_speech_time < MIN_SPEECH_INTERVAL:
                    final_guidance_text = ""  # Quá nhanh, bỏ qua lần thông báo này
            
            # Xử lý đặc biệt giới hạn tần suất của "giữ hướng đi thẳng"
            if self.dan_vach is not None and selected_voice['source'] == 'navigation':
                # Dẫn vạch kiểu mới: luật nói (dan_theo_vach.LuatNoi) đã tự giãn nhịp, không lọc thêm ở đây
                if final_guidance_text:
                    self.last_any_speech_time = current_time
            elif final_guidance_text == "Giữ hướng đi thẳng":
                if self.straight_continuous_mode:
                    # Chế độ thông báo liên tục: chỉ kiểm tra khoảng thời gian
                    if current_time - self.last_guide_time >= self.guide_interval:
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
                else:
                    # Chế độ giới hạn gốc
                    if (current_time - self.last_guide_time >= self.guide_interval) and \
                       (self.straight_repeat_count < self.straight_repeat_limit):
                        self.last_guide_time = current_time
                        self.straight_repeat_count += 1
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice['source'] != 'obstacle':
                # [Sửa đổi] Lệnh không phải đi thẳng, không phải vật cản - hỗ trợ thông báo liên tục lệnh phương hướng
                # Xác định có phải lệnh phương hướng không
                direction_keywords = ["rẽ trái", "rẽ phải", "dịch trái", "dịch phải", "sang trái", "sang phải", "dịch sang", "chỉnh nhẹ"]
                is_direction = any(keyword in final_guidance_text.lower() for keyword in direction_keywords)
                
                if is_direction:
                    # Lệnh phương hướng: hỗ trợ thông báo liên tục
                    if final_guidance_text == self.last_direction_message:
                        # Cùng một lệnh phương hướng thì kiểm tra khoảng thời gian
                        if current_time - self.last_direction_time >= self.direction_interval:
                            self.last_direction_time = current_time
                            self.last_any_speech_time = current_time
                            self.straight_repeat_count = 0
                        else:
                            final_guidance_text = ""  # Khoảng thời gian chưa đủ, bỏ qua
                    else:
                        # Lệnh phương hướng mới, thông báo ngay
                        self.last_direction_message = final_guidance_text
                        self.last_direction_time = current_time
                        self.last_any_speech_time = current_time
                        self.straight_repeat_count = 0
                else:
                    # Các lệnh khác: chỉ thông báo một lần
                    if final_guidance_text != self.last_guidance_message:
                        self.last_guidance_message = final_guidance_text
                        self.straight_repeat_count = 0
                        self.last_any_speech_time = current_time
                    else:
                        final_guidance_text = ""
            elif final_guidance_text and selected_voice['source'] == 'obstacle':
                # Giọng nói vật cản luôn được thông báo
                self.last_any_speech_time = current_time
            elif final_guidance_text and selected_voice['source'] == 'crosswalk':
                # Giọng nói vạch qua đường luôn được thông báo (không bị giới hạn kiểm tra lặp)
                self.last_any_speech_time = current_time
                
            # Thông báo giọng nói đã chọn
            if final_guidance_text:
                try:
                    # [Tối ưu] Giọng nói ghép chỉ phát phần đầu, tránh dồn hàng đợi
                    if selected_voice.get('source') == 'crosswalk' and ';' in final_guidance_text:
                        voice_parts = split_combined_voice(final_guidance_text)
                        logger.info(f"[Giọng nói vạch qua đường] Thông báo ghép phát hiện {len(voice_parts)} phần, chỉ phát phần đầu để giữ thời gian thực")
                        # Chỉ phát phần đầu, các phần sau bỏ để giữ tính thời gian thực
                        if voice_parts:
                            play_voice_text(voice_parts[0])
                            logger.info(f"[Thông báo giọng nói] Ưu tiên{selected_voice['priority']}: {voice_parts[0]}")
                    else:
                        play_voice_text(final_guidance_text)
                        logger.info(f"[Thông báo giọng nói] Ưu tiên{selected_voice['priority']}: {final_guidance_text}")
                except Exception as e:
                    logger.error(f"[Thông báo giọng nói] Phát thất bại: {e}")
        else:
            final_guidance_text = ""

        if cau_chan and final_guidance_text != cau_chan:
            self.bo_vat_chan.chua_noi()              # câu vật chắn không được phát: bộ điều khiển báo lại khung sau

        if self.dan_vach is not None:
            # câu dẫn vạch không được phát (đang đọc câu khác / thua câu ưu tiên cao hơn): luật nói thử lại khung sau
            for c in voice_candidates:
                if c['source'] == 'navigation' and c['text'] != final_guidance_text:
                    self.dan_vach.chua_noi(c['text'])
        
        # 7. Tạo ảnh chú thích
        annotated_image = None

        if frame_visualizations:
            annotated_image = self._draw_visualizations(image.copy(), frame_visualizations)
        else:
            annotated_image = image.copy()
        
        # Thêm nút lệnh phía dưới (hiển thị giọng nói thực tế đang thông báo)
        current_instruction = final_guidance_text if final_guidance_text else "Đang chờ..."
        annotated_image = self._draw_command_button(annotated_image, current_instruction)
        
        # 8. Trả về kết quả
        return ProcessingResult(
            # dẫn vạch kiểu mới: trả đúng câu đã phát (app_main phát lại res.guidance_text; không phát câu đã bị chặn)
            guidance_text=final_guidance_text if self.dan_vach is not None else guidance_text,
            visualizations=frame_visualizations,
            annotated_image=annotated_image,
            state_info={
                # orchestrator dùng "NAVIGATING" để biết đã đứng trên vạch (vd. sau khi qua đường)
                "state": self.current_state if self.dan_vach is None else
                         ("NAVIGATING" if self.dan_vach.da_tren_vach(time.time()) else STATE_ONBOARDING),
                "crosswalk_stage": current_stage,
                "frame_count": self.frame_counter
            }
        )
    
    def _detect_path_and_crosswalk(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Phát hiện đường dẫn và vạch qua đường"""
        if self.yolo_model is None and self.dan_vach is not None:
            logger.warning("Chưa nạp model YOLO: không có mask vạch (không bịa dải giả)")
            return None, None
        if self.yolo_model is None:
            # [Mới] Khi không có model thì trả về dữ liệu giả lập để kiểm thử
            logger.warning("Chưa nạp model YOLO, trả về dữ liệu giả lập")
            h, w = image.shape[:2]
            # Tạo mặt nạ đường dẫn giả lập (dải dọc ở giữa)
            blind_path_mask = np.zeros((h, w), dtype=np.uint8)
            # Tạo một dải dọc ở giữa ảnh có độ rộng bằng 20% chiều rộng ảnh
            strip_width = int(w * 0.2)
            strip_left = (w - strip_width) // 2
            blind_path_mask[int(h*0.3):, strip_left:strip_left+strip_width] = 255
            return blind_path_mask, None
        
        blind_path_mask = None
        crosswalk_mask = None
        
        try:
            min_conf = min(self.CLASS_CONF_THRESHOLDS.values())
            results = self.yolo_model.predict(image, verbose=False, conf=min_conf, classes=[0, 1])
            
            if (results and results[0] and results[0].masks is not None and 
                results[0].boxes is not None and len(results[0].masks.data) > 0):
                
                for mask_tensor, conf_tensor, cls_tensor in zip(
                    results[0].masks.data, results[0].boxes.conf, results[0].boxes.cls
                ):
                    class_id = int(cls_tensor.item())
                    confidence = float(conf_tensor.item())
                    threshold = self.CLASS_CONF_THRESHOLDS.get(class_id, 1.0)
                    
                    if confidence >= threshold:
                        current_mask = self._tensor_to_mask(mask_tensor, image.shape[1], image.shape[0])
                        
                        if class_id == 1:  # Đường dẫn
                            if blind_path_mask is None:
                                blind_path_mask = current_mask
                            else:
                                blind_path_mask = cv2.bitwise_or(blind_path_mask, current_mask)
                        elif class_id == 0:  # Vạch qua đường
                            if crosswalk_mask is None:
                                crosswalk_mask = current_mask
                            else:
                                crosswalk_mask = cv2.bitwise_or(crosswalk_mask, current_mask)
        except Exception as e:
            logger.error(f"Phát hiện YOLO thất bại: {e}")
            if self.dan_vach is not None:
                return None, None     # không bịa dải giả: kính phải biết là không thấy vạch (báo "mất vạch")
            # [Mới] Khi phát hiện thất bại cũng trả về dữ liệu giả lập
            h, w = image.shape[:2]
            blind_path_mask = np.zeros((h, w), dtype=np.uint8)
            strip_width = int(w * 0.2)
            strip_left = (w - strip_width) // 2
            blind_path_mask[int(h*0.3):, strip_left:strip_left+strip_width] = 255
        
        return blind_path_mask, crosswalk_mask
    
    def _tensor_to_mask(self, mask_tensor, out_w: int, out_h: int, binarize: bool = True) -> np.ndarray:
        """Chuyển mặt nạ tensor thành mảng numpy"""
        try:
            import torch
            
            if not isinstance(mask_tensor, torch.Tensor):
                arr = np.asarray(mask_tensor)
                if arr.dtype != np.uint8:
                    arr = (arr > 0.5).astype(np.uint8) * 255 if binarize else (arr * 255.0).astype(np.uint8)
                mask_u8 = arr
            else:
                if mask_tensor.dtype in (torch.bfloat16, torch.float16):
                    mask_tensor = mask_tensor.to(torch.float32)
                
                if mask_tensor.ndim > 2:
                    mask_tensor = mask_tensor.squeeze()
                
                if binarize:
                    mask_tensor = (mask_tensor > 0.5).to(torch.uint8).mul_(255)
                    mask_u8 = mask_tensor.cpu().numpy()
                else:
                    mask_u8 = (mask_tensor.mul(255).clamp_(0, 255).to(torch.uint8)).cpu().numpy()
            
            if mask_u8.ndim == 3:
                mask_u8 = mask_u8.squeeze(-1)
            
            if mask_u8.shape[1] != out_w or mask_u8.shape[0] != out_h:
                mask_u8 = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            
            return mask_u8
        except ImportError:
            # Nếu không có torch thì trả về mặt nạ rỗng
            return np.zeros((out_h, out_w), dtype=np.uint8)
    
    def _stabilize_mask(self, prev_gray, curr_gray, raw_mask, prev_stable_mask, mask_type):
        """Ổn định hóa mặt nạ - dùng optical flow Lucas-Kanade"""
        if mask_type == 'blind_path':
            ttl = self.blind_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA
        else:  # crosswalk
            ttl = self.cross_miss_ttl
            min_area = self.MASK_STAB_MIN_AREA
        
        # Gọi phương pháp ổn định hóa optical flow mới
        stable_mask = self._stabilize_seg_mask(
            prev_gray, curr_gray, raw_mask, prev_stable_mask,
            (curr_gray.shape[1], curr_gray.shape[0]) if curr_gray is not None else (640, 480),
            min_area_px=min_area,
            morph_kernel=self.MASK_STAB_KERNEL,
            mask_type=mask_type
        )
        
        if stable_mask is not None:
            # Đặt lại TTL
            if mask_type == 'blind_path':
                self.blind_miss_ttl = self.MASK_MISS_TTL
            else:
                self.cross_miss_ttl = self.MASK_MISS_TTL
            return stable_mask
        else:
            # Giảm TTL
            if mask_type == 'blind_path':
                self.blind_miss_ttl = max(0, self.blind_miss_ttl - 1)
            else:
                self.cross_miss_ttl = max(0, self.cross_miss_ttl - 1)
            return None
    
    def _stabilize_seg_mask(self, prev_gray, curr_gray, curr_mask, prev_stable_mask, 
                          image_wh, min_area_px=1500, morph_kernel=3, iou_high_thr=0.4, mask_type='', 
                          fast_clear=True):
        """Cài đặt ổn định hóa mặt nạ bằng optical flow Lucas-Kanade"""
        W, H = image_wh
        
        def _binarize(mask):
            if mask is None:
                return None
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        
        def _morph_smooth(mask, kernel_size):
            if mask is None:
                return None
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (max(1, kernel_size), max(1, kernel_size)))
            sm = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
            sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, k, iterations=1)
            return sm
        
        curr_mask_b = _binarize(curr_mask)
        prev_mask_b = _binarize(prev_stable_mask)
        
        # Nếu không có dữ liệu lịch sử thì trả về mặt nạ hiện tại
        if prev_mask_b is None or prev_gray is None or curr_gray is None:
            return _morph_smooth(curr_mask_b, morph_kernel) if curr_mask_b is not None else None
        
        # Khung hiện tại có kết quả phát hiện
        if curr_mask_b is not None and np.sum(curr_mask_b > 0) >= min_area_px:
            # Tính IoU với khung trước
            if prev_mask_b is not None:
                inter = np.logical_and(curr_mask_b > 0, prev_mask_b > 0).sum()
                union = np.logical_or(curr_mask_b > 0, prev_mask_b > 0).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0
                
                # IoU đủ cao, nghĩa là phát hiện ổn định, dùng thẳng kết quả hiện tại
                if iou >= iou_high_thr:
                    return _morph_smooth(curr_mask_b, morph_kernel)
                
                # IoU thấp nhưng vẫn có chồng lấp, thực hiện hòa trộn có trọng số
                elif iou > 0.1:
                    # Dùng mặt nạ do optical flow dự đoán
                    flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                    if flow_mask is not None:
                        # Điều chỉnh trọng số động theo IoU
                        # IoU càng thấp càng dựa vào optical flow; IoU càng cao càng dựa vào phát hiện hiện tại
                        w_curr = min(0.9, 0.4 + iou)  # IoU=0.1 thì w_curr=0.5, IoU=0.5 thì w_curr=0.9
                        w_flow = 1.0 - w_curr
                        
                        fused = (w_curr * curr_mask_b.astype(np.float32) + 
                                w_flow * flow_mask.astype(np.float32))
                        fused_bin = (fused >= 128).astype(np.uint8) * 255
                        
                        # Khởi tạo lại điểm optical flow (nếu IoU quá thấp)
                        if iou < self.flow_iou_threshold:
                            self.flow_points['blind_path'] = None
                        
                        return _morph_smooth(fused_bin, morph_kernel)
            
            # Không có lịch sử hoặc IoU quá thấp, dùng phát hiện hiện tại
            return _morph_smooth(curr_mask_b, morph_kernel)
        
        # Khung hiện tại không có kết quả phát hiện, thử dùng ngoại suy optical flow
        else:
            # Lấy TTL tương ứng
            if mask_type == 'blind_path':
                ttl = self.blind_miss_ttl
            else:
                ttl = self.cross_miss_ttl
            
            # [Sửa đổi] Khung hiện tại không có kết quả phát hiện, xóa nhanh
            if fast_clear and ttl <= 1:
                # TTL cạn, trả về None ngay, không dùng optical flow
                return None
                
            if prev_mask_b is not None and np.sum(prev_mask_b > 0) >= min_area_px and ttl > 0:
                # Dùng optical flow dự đoán
                flow_mask = self._predict_mask_with_flow(prev_mask_b, prev_gray, curr_gray)
                if flow_mask is not None and np.sum(flow_mask > 0) >= min_area_px * 0.5:
                    return _morph_smooth(flow_mask, morph_kernel)
            
            # Optical flow thất bại hoặc vượt quá TTL
            return None
    
    def _predict_mask_with_flow(self, prev_mask, prev_gray, curr_gray):
        """Dùng optical flow Lucas-Kanade để dự đoán vị trí mặt nạ (bản cải tiến)"""
        try:
            # Cách 1: thử dùng phương pháp bao lồi (tham khảo yolomedia)
            if hasattr(self, 'flow_points') and 'blind_path' in self.flow_points:
                p0 = self.flow_points['blind_path']
                if p0 is not None and len(p0) >= 5:
                    # Tính optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
                    
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        if len(good_new) >= 5:
                            # Cập nhật điểm optical flow
                            self.flow_points['blind_path'] = good_new.reshape(-1, 1, 2)
                            
                            # Tạo mặt nạ bao lồi
                            hull = cv2.convexHull(good_new.reshape(-1, 1, 2))
                            poly = hull.reshape(-1, 2)
                            
                            if len(poly) >= 3:
                                H, W = curr_gray.shape[:2]
                                flow_mask = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(flow_mask, [poly.astype(np.int32)], 255)
                                return flow_mask
            
            # Cách 2: phương pháp điểm đặc trưng biên (phương pháp gốc, làm dự phòng)
            edge_mask = self._get_edge_mask(prev_mask, offset=10)
            
            # Phát hiện điểm đặc trưng
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=edge_mask, **self.feature_params)
            if p0 is None or len(p0) < 8:
                return None
            
            # Lưu điểm đặc trưng cho lần sau
            self.flow_points['blind_path'] = p0
            
            # Tính optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
            
            if p1 is None or st is None:
                return None
            
            # Chỉ giữ các điểm theo dõi thành công
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) < 5:
                return None
            
            # Ước lượng ma trận biến đổi (dùng RANSAC để tăng độ bền vững)
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            
            if M is None:
                return None
            
            # Áp dụng biến đổi
            H, W = curr_gray.shape[:2]
            flow_mask = cv2.warpAffine(prev_mask, M, (W, H), 
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
            
            return flow_mask
            
        except Exception as e:
            logger.debug(f"Dự đoán optical flow thất bại: {e}")
            return None
            
    
    def _get_edge_mask(self, mask, offset=10):
        """Lấy vùng biên trong của mặt nạ, dùng để phát hiện điểm đặc trưng"""
        if mask is None:
            return None
        
        # Ăn mòn để lấy mặt nạ bên trong
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset*2, offset*2))
        inner = cv2.erode(mask, kernel, iterations=1)
        
        # Biên = gốc - bên trong
        edge = cv2.subtract(mask, inner)
        
        # Giãn nhẹ vùng biên
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.dilate(edge, kernel_small, iterations=1)
        
        return edge

    def _smooth_centerline(self, centerline_data):
        """Làm mượt dữ liệu đường tâm, giảm rung lắc"""
        if centerline_data is None or len(centerline_data) < 5:
            return centerline_data
        
        # Lưu vào lịch sử
        self.centerline_history.append(centerline_data.copy())
        if len(self.centerline_history) > self.centerline_history_max:
            self.centerline_history.pop(0)
        
        # Nếu lịch sử không đủ, trả về dữ liệu khung hiện tại được làm mượt nhẹ
        if len(self.centerline_history) < 3:
            # Làm mượt không gian cho khung hiện tại
            smoothed_data = centerline_data.copy()
            # Dùng trung bình cửa sổ trượt
            window_size = 5
            for i in range(len(smoothed_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(smoothed_data), i + window_size // 2 + 1)
                window = smoothed_data[start_idx:end_idx]
                if len(window) > 0:
                    smoothed_data[i, 1] = np.mean(window[:, 1])  # Làm mượt tọa độ x
                    smoothed_data[i, 2] = np.mean(window[:, 2])  # Làm mượt độ rộng
            return smoothed_data
        
        # Làm mượt theo thời gian: dùng trung bình có trọng số của các khung lịch sử
        smoothed_data = centerline_data.copy()
        
        # Với mỗi tọa độ y tìm dữ liệu tương ứng trong khung lịch sử
        for i, (y, x, width) in enumerate(centerline_data):
            x_values = [x]
            width_values = [width]
            weights = [1.0]  # Khung hiện tại có trọng số cao nhất
            
            # Tìm dữ liệu có tọa độ y gần nhau trong khung lịch sử
            for hist_idx, hist_data in enumerate(self.centerline_history[-3:-1]):  # Dùng 2 khung lịch sử gần nhất
                # Tìm tọa độ y gần nhất
                y_diffs = np.abs(hist_data[:, 0] - y)
                if len(y_diffs) > 0:
                    closest_idx = np.argmin(y_diffs)
                    if y_diffs[closest_idx] < 10:  # Chênh lệch tọa độ y nhỏ hơn 10 điểm ảnh
                        x_values.append(hist_data[closest_idx, 1])
                        width_values.append(hist_data[closest_idx, 2])
                        # Trọng số khung lịch sử giảm dần
                        weights.append(0.5 ** (len(self.centerline_history) - hist_idx - 1))
            
            # Trung bình có trọng số
            if len(x_values) > 1:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                smoothed_data[i, 1] = np.sum(np.array(x_values) * weights)
                smoothed_data[i, 2] = np.sum(np.array(width_values) * weights)
        
        # Làm mượt không gian: lấy trung bình cửa sổ trượt thêm một lần nữa cho kết quả
        window_size = 3
        final_data = smoothed_data.copy()
        for i in range(len(final_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(final_data), i + window_size // 2 + 1)
            window = smoothed_data[start_idx:end_idx]
            if len(window) > 0:
                final_data[i, 1] = np.mean(window[:, 1])
                final_data[i, 2] = np.mean(window[:, 2])
        
        return final_data

    def _estimate_affine(self, prev_gray, curr_gray, mask=None):
        """Dùng optical flow để ước lượng biến đổi affine (phương pháp dự phòng)"""
        try:
            # Trích xuất điểm đặc trưng
            if mask is not None:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
            else:
                p0 = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
            
            if p0 is None or len(p0) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # Tính optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
            
            if p1 is None or st is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # Chỉ giữ các điểm tốt
            good_new = p1[st == 1].reshape(-1, 2)
            good_old = p0[st == 1].reshape(-1, 2)
            
            if len(good_new) < 4:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            # Ước lượng biến đổi affine
            M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
            
            if M is None:
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            
            return M
            
        except Exception as e:
            logger.debug(f"Ước lượng affine thất bại: {e}")
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    def _warp_mask(self, mask, M, output_shape):
        """Áp dụng biến đổi affine"""
        try:
            W, H = output_shape
            warped = cv2.warpAffine(mask, M, (W, H), 
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            return warped
        except:
            return None
    
    def _add_mask_visualization(self, mask, visualizations, viz_type, color, add_outline=True):
        """Thêm trực quan hóa mặt nạ (thêm viền)"""
        if mask is None:
            return
        
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                points = main_contour.squeeze(1)[::5].tolist()
                
                # Thêm phần tô
                visualizations.append({
                    "type": viz_type,
                    "points": points,
                    "color": color
                })
                
                # Thêm viền (đường dẫn không thêm viền)
                if add_outline and viz_type != "blind_path_mask":
                    visualizations.append({
                        "type": "outline",
                        "points": points,
                        "color": "rgba(255, 255, 255, 0.8)",  # Viền trắng
                        "thickness": 3
                    })
        except:
            pass

    
    def _update_crosswalk_tracker(self, crosswalk_mask, image_height, image_width):
        """Cập nhật bộ theo dõi vạch qua đường"""
        if crosswalk_mask is not None:
            self.crosswalk_tracker['consecutive_frames'] += 1
            self.crosswalk_tracker['last_seen_frame'] = self.frame_counter
            
            # Tính các chỉ số quan trọng
            total_area = image_height * image_width
            area_ratio = np.sum(crosswalk_mask > 0) / total_area
            y_coords, x_coords = np.where(crosswalk_mask > 0)
            
            if len(y_coords) > 0:
                bottom_y_ratio = np.max(y_coords) / image_height
                center_x_ratio = np.mean(x_coords) / image_width
                
                self.crosswalk_tracker['last_area_ratio'] = area_ratio
                self.crosswalk_tracker['last_bottom_y_ratio'] = bottom_y_ratio
                self.crosswalk_tracker['last_center_x_ratio'] = center_x_ratio
                
                # Tính góc
                try:
                    contours, _ = cv2.findContours(crosswalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(main_contour)
                        angle = rect[-1]
                        w, h = rect[1]
                        if w < h:
                            angle += 90
                        self.crosswalk_tracker['last_angle'] = angle
                except:
                    self.crosswalk_tracker['last_angle'] = 0.0
                
                # Chuyển trạng thái
                is_ready_to_switch = (
                    area_ratio >= self.CROSSWALK_SWITCH_AREA_RATIO and
                    bottom_y_ratio >= self.CROSSWALK_SWITCH_BOTTOM_RATIO or
                    (self.crosswalk_tracker['consecutive_frames'] >= self.CROSSWALK_SWITCH_CONSECUTIVE_FRAMES 
                     and area_ratio > 0.18)
                )
                
                if is_ready_to_switch and self.crosswalk_tracker['alignment_status'] == 'aligned':
                    if self.crosswalk_tracker['stage'] != 'ready':
                        self.crosswalk_tracker['stage'] = 'ready'
                elif area_ratio > 0.07 or bottom_y_ratio > 0.75:
                    if self.crosswalk_tracker['stage'] in ['far', 'not_detected']:
                        self.crosswalk_tracker['stage'] = 'approaching'
                elif area_ratio > 0.01:
                    if self.crosswalk_tracker['stage'] == 'not_detected':
                        self.crosswalk_tracker['stage'] = 'far'
        else:
            # Phát hiện mất
            if self.frame_counter - self.crosswalk_tracker['last_seen_frame'] > 15:
                self.crosswalk_tracker['stage'] = 'not_detected'
                self.crosswalk_tracker['consecutive_frames'] = 0
                self.crosswalk_tracker['position_announced'] = False
                self.crosswalk_tracker['alignment_status'] = 'not_aligned'
                # Đặt lại cờ trạng thái chuẩn bị
                if hasattr(self, 'crosswalk_ready_announced'):
                    self.crosswalk_ready_announced = False
                    self.crosswalk_ready_time = 0
                if hasattr(self, 'traffic_light_history'):
                    self.traffic_light_history.clear()
                    self.green_light_announced = False
    
    def _handle_crosswalk_approaching(self, frame_visualizations, image_height, image_width, image):
        """Xử lý tình huống tiến gần vạch qua đường"""
        # Phát hiện vật cản
        if self.obstacle_detector and self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            detected_obstacles = self._detect_obstacles(image)
            self.last_detected_obstacles = detected_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
        
        # Thêm trực quan hóa vật cản
        for obs in self.last_detected_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)
        
        # Ưu tiên kiểm tra vật cản ở gần (nâng ngưỡng, chỉ rất gần mới cảnh báo)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75  # Nâng lên 0.75
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12  # Nâng lên 0.12
        near_obstacles = [
            obs for obs in self.last_detected_obstacles
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD)
        ]
        
        # Nếu có vật cản ở gần thì áp dụng cùng logic thông báo
        if near_obstacles:
            main_obstacle = near_obstacles[0]
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()
            
            # Kiểm tra có cần thông báo không (tránh lặp)
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                should_announce = True
                self.last_obstacle_speech_time = current_time
            
            if should_announce:
                return self._speech_for_obstacle(obstacle_name)
        else:
            # Khi không có vật cản thì xóa bản ghi
            self.last_obstacle_speech = ""
        
        # Logic căn chỉnh
        if self.crosswalk_tracker['alignment_status'] == 'not_aligned':
            guidance_text = "Đang tiến gần vạch qua đường, đang giúp bạn căn hướng."
            self.crosswalk_tracker['alignment_status'] = 'aligning'
        else:
            angle = self.crosswalk_tracker['last_angle']
            center_x_ratio = self.crosswalk_tracker['last_center_x_ratio']
            
            ANGLE_ALIGN_THRESHOLD = 15
            POSITION_ALIGN_THRESHOLD = 0.25
            
            if abs(angle) > ANGLE_ALIGN_THRESHOLD:
                guidance_text = "Rẽ phải" if angle < 0 else "Rẽ trái"
            elif abs(center_x_ratio - 0.5) > (POSITION_ALIGN_THRESHOLD / 2):
                guidance_text = "Dịch phải" if center_x_ratio < 0.5 else "Dịch trái"
            else:
                self.crosswalk_tracker['alignment_status'] = 'aligned'
                guidance_text = "Đã căn vạch qua đường, tiếp tục đi."
        
        # Thêm bảng dữ liệu
        data_for_panel = {
            "Trạng thái": "Căn vào vạch qua đường",
            "Hướng dẫn": guidance_text,
            "Góc": f"{self.crosswalk_tracker['last_angle']:.1f}°",
            "Độ lệch": f"{(self.crosswalk_tracker['last_center_x_ratio'] - 0.5):.2f}"
        }
        frame_visualizations.append({
            "type": "data_panel",
            "data": data_for_panel,
            "position": (25, image_height - 75)
        })
        
        return guidance_text
    
    def _execute_state_machine(self, mask, image, frame_visualizations, 
                              image_height, image_width, curr_gray):
        """Thực thi logic máy trạng thái"""
        if self.current_state == STATE_ONBOARDING:
            return self._handle_onboarding(mask, image, frame_visualizations, 
                                         image_height, image_width)
        elif self.current_state == STATE_NAVIGATING:
            return self._handle_navigating(mask, image, frame_visualizations,
                                         image_height, image_width, curr_gray)
        elif self.current_state == STATE_MANEUVERING_TURN:
            return self._handle_maneuvering_turn(mask, image, frame_visualizations,
                                               image_height, image_width)
        elif self.current_state == STATE_LOCKING_ON:
            return self._handle_locking_on(frame_visualizations)
        elif self.current_state == STATE_AVOIDING_OBSTACLE:
            return self._handle_avoiding_obstacle(mask, image, frame_visualizations,
                                                image_height, image_width)
        
        return ""
    
    def _handle_onboarding(self, mask, image, frame_visualizations, image_height, image_width):
        """Xử lý trạng thái lên đường dẫn"""
        image_center_x = image_width / 2
        vp_features = self._get_vanishing_point_features(mask)
        
        if vp_features and vp_features['fit_error'] < self.VP_FIT_ERROR_THRESHOLD:
            # Dùng phương pháp điểm triệt tiêu
            VP, L_center = vp_features["VP"], vp_features["L_center"]
            
            if self.onboarding_step == ONBOARDING_STEP_ROTATION:
                if abs(VP[0] - image_center_x) < (image_width * self.ONBOARDING_ALIGN_THRESHOLD_RATIO):
                    guidance_text = "Đã đúng hướng! Bây giờ hãy chỉnh vị trí."
                    self.onboarding_step = ONBOARDING_STEP_TRANSLATION
                else:
                    guidance_text = "Hãy xoay sang trái." if VP[0] < image_center_x else "Hãy xoay sang phải."
                
                angle_error_px = VP[0] - image_center_x
                self._add_data_panel(frame_visualizations, {
                    "Trạng thái": "Lên đường dẫn (hướng)",
                    "Hướng dẫn": guidance_text,
                    "Góc": f"{angle_error_px:.1f}px",
                    "Độ lệch": "Chờ hiệu chỉnh"
                }, (25, image_height - 75))
                
            elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
                L_center_bottom_x = self._calculate_line_x_at_y(L_center, image_height - 1)
                
                if L_center_bottom_x:
                    center_offset_pixels = L_center_bottom_x - image_center_x
                    center_offset_ratio = abs(center_offset_pixels) / image_width
                    
                    if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                        guidance_text = "Chỉnh xong! Bạn đã ở trên đường dẫn, bắt đầu đi."
                        self.current_state = STATE_NAVIGATING
                    else:
                        guidance_text = "Hãy dịch sang trái." if L_center_bottom_x < image_center_x else "Hãy dịch sang phải."
                    
                    self._add_data_panel(frame_visualizations, {
                        "Trạng thái": "Lên đường dẫn (vị trí)",
                        "Hướng dẫn": guidance_text,
                        "Góc": "Đã căn chỉnh",
                        "Độ lệch": f"{center_offset_ratio * 100:.1f}%"
                    }, (25, image_height - 75))
                else:
                    guidance_text = "Hãy tiến lên để đường dẫn rõ hơn."
        else:
            # Dùng phương pháp miền điểm ảnh
            pixel_features = self._get_pixel_domain_features(mask, image.shape)
            if not pixel_features:
                return ""
            self._add_navigation_info_visualization(pixel_features, image_height, image_width, frame_visualizations)
            guidance_text = self._handle_pixel_domain_onboarding(
                pixel_features, image_height, image_width, frame_visualizations
            )
        
        return guidance_text
    
    def _handle_navigating(self, mask, image, frame_visualizations, 
                          image_height, image_width, curr_gray):
        """Xử lý trạng thái dẫn đường thông thường"""
        image_center_x = image_width / 2

        if self.dan_vach is not None:
            # Vật cản đã được process_frame xử lý (câu ưu tiên 100 + vẽ) trước khi vào đây; KHÔNG đụng
            # pending_obstacle_voice (khối vật cản cũ bên dưới xóa mất câu của module né vật cản độ sâu).
            # Dẫn vạch: đo điểm tụ, không phát hiện khúc cua kiểu cũ.
            cau, do = self.dan_vach.buoc(mask, time.time(), vat=self._vat_ne())
            self.dan_vach.ve(do, frame_visualizations, image_height, image_width)
            return cau

        # Trích xuất đặc trưng đường đi
        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return "Trích xuất đặc trưng đường đi thất bại"
        self._add_navigation_info_visualization(features, image_height, image_width, frame_visualizations)
        
        # Phát hiện khúc cua
        if self.turn_cooldown_frames == 0:
            corner_info = self._detect_sharp_corner(features['centerline_data'])
            if corner_info:
                self._update_turn_tracker(corner_info)
                
                if self.turn_detection_tracker['consecutive_hits'] >= 3:
                    stable_corner_info = self.turn_detection_tracker['corner_info']
                    corner_y = stable_corner_info['corner_point_pixel'][1]
                    turn_trigger_y_threshold = image_height * 0.65
                    
                    if corner_y > turn_trigger_y_threshold:
                        # Kích hoạt rẽ
                        direction_text = 'phải' if self.turn_detection_tracker['direction'] == 'right' else 'trái'
                        self.current_state = STATE_MANEUVERING_TURN
                        self.maneuver_target_info = stable_corner_info
                        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                        self._reset_turn_tracker()
                        # Không còn thông báo "đã tới chỗ rẽ", trả về chuỗi rỗng để logic phía sau xử lý
                        return ""
                    else:
                        # Không còn báo trước khúc cua, tiếp tục dẫn đường thông thường
                        pass
        
        # Ưu tiên 1: phát hiện vật cản (ưu tiên cao nhất)
        obstacles = self._check_obstacles(image, mask, frame_visualizations)
        if obstacles:
            # Lấy vật cản chính
            main_obstacle = obstacles[0]
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()
            
            # Kiểm tra có cần thông báo không (tránh lặp)
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                # Vật cản khác nhau, thông báo ngay
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                # Cùng một vật cản nhưng quá thời gian hồi, thông báo lại
                should_announce = True
                self.last_obstacle_speech_time = current_time
            
            if should_announce:
                # Không vào quy trình tránh vật cản đầy đủ, chỉ cảnh báo
                # Đặt giọng nói vật cản chờ thông báo, thay vì trả về trực tiếp
                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)
            # Nếu không cần thông báo thì tiếp tục dẫn đường thông thường
        else:
            # Không có vật cản, xóa bản ghi
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None
        
        # Ưu tiên 2: dẫn đường thông thường (dịch trái/dịch phải/rẽ trái/rẽ phải > đi thẳng)
        return self._generate_navigation_guidance(
            features, image_height, image_width, frame_visualizations
        )
    
    # ---------- Vật chắn trên vạch (spec 2026-10-27-vong-vat-can-tren-vach) ----------
    def _do_vach_moi(self, t):
        """Lần đo vạch tin cậy trong T_GIU_DO giây gần nhất (None nếu không có)."""
        from dan_theo_vach import T_GIU_DO
        dv = self.dan_vach
        return dv.do_gan if dv is not None and dv.do_gan is not None and t - dv.t_do_gan <= T_GIU_DO else None

    def _co_vat_chan(self):
        return self.bo_vat_chan is not None and hasattr(self.ne_vat_can, "ben_di_duoc")

    def _chi_bao_vat_chan(self, t):
        """Bộ lọc câu của module vật cản (None = như cũ). Người dùng (28/10): khi theo vạch chỉ quan tâm vật trong vùng
        vạch, và MỘT người nói: xe đỗ trên vạch chỉ bộ điều khiển vật chắn được nói (trước đây module vật cản chen
        "Dừng lại. Xe máy" / "Xe máy, phía trước" về cùng chiếc xe -> nghe như mâu thuẫn).
        Luôn báo: xe đang tới, địa hình, vật ngang đầu. Còn lại khi đang theo vạch (thấy vạch trong T_THEO_LAU s):
          - đã báo vòng: im (gậy dẫn đoạn này);
          - vật đứng yên: trên vạch ("chan") -> im (bộ điều khiển nói); không đo được vạch / mất vạch tạm thời -> chỉ
            báo thông tin ("tin": "Xe máy, phía trước", không ra lệnh) nếu ngay phía trước trong D_NOI; ngoài vạch im;
          - vật chưa chắc đứng yên (người, xe chưa rõ tốc độ): trên vạch trong D_NOI -> báo như cũ; mất vạch -> chỉ
            vật phía trước trong D_NOI.
        Xe vừa thấy chắn mà tới gần che mép vạch ("khong_chac") vẫn là xe chắn (NhoVatChan); mất đo vạch < 2 s vẫn là
        trên vạch (BoVatChan.tren_gan) -> bộ điều khiển bảo vòng, không "Xe máy, phía trước" (người dùng 29/10)."""
        if not self._co_vat_chan():
            return None
        dv, bo, nho = self.dan_vach, self.bo_vat_chan, self.nho_vat_chan
        tren = bo.tren_gan(t, dv.da_tren_vach(t))
        # xe đè che mất vạch vài giây vẫn là đang theo vạch (lỗi thật IMG_7354 252 s: "Vạch bị chắn" rồi
        # "Dịch sang trái. Xe máy" khi kính quay về chế độ cũ sau 1,5 s không thấy vạch)
        if not (tren or bo.dang_xu_ly or dv.dang_theo_vach(t, T_THEO_LAU)):
            return None
        from dan_theo_vach import phan_loai_vat
        from vat_chan_vach import D_NOI
        do, ne, hw = self._do_vach_moi(t), self.ne_vat_can, dv.hw

        def truoc_gan(o):
            return o.get('side') == 'phía trước' and o['dist'] <= D_NOI

        def chi_bao(o):
            if o.get('toi') or o.get('nhanh') or o.get('high') or o.get('src') in ('curb', 'stairs', 'drop', 'hole'):
                return True
            if bo.dang_xu_ly:
                return False
            yen = ne.dung_yen(o) or (tren and nho.tiep(t, o['dist']) and ne.co_the_dang_do(o))
            loai = phan_loai_vat(o, do, ne.geo, hw) if tren else "khong_chac"
            if yen and tren and not o.get('high'):
                loai = nho.loai(t, loai, o['dist'])
            if loai == "canh":
                return False
            if yen:
                # xe đỗ ngay trước mặt mà không phân loại được (xe che kín vạch / mất đo vạch): bộ điều khiển bảo vòng
                # (người dùng 29/10, IMG_7354 325,7 s); đo được vạch (biết vật có trên vạch không) thì chỉ báo
                # thông tin (khiem 57,9 s: người nhìn chéo, "phía trước" camera không phải lối trên vạch)
                if loai == "chan" or (truoc_gan(o) and do is None):
                    return False
                return "tin" if truoc_gan(o) else False
            return o['dist'] <= D_NOI if loai == "chan" else truoc_gan(o)
        return chi_bao

    def _buoc_vat_chan(self, t):
        """Chạy bộ điều khiển vật chắn với vật chắn đứng yên gần nhất; trả câu cần nói ("" nếu không)."""
        if not self._co_vat_chan():
            return ""
        from dan_theo_vach import phan_loai_vat, vi_tri_tren_vach
        from vat_chan_vach import ten_noi
        from vat_chan_vach import D_THEO
        ne, dv, do, nho = self.ne_vat_can, self.dan_vach, self._do_vach_moi(t), self.nho_vat_chan
        theo = dv.dang_theo_vach(t, T_THEO_LAU)
        vat = None
        for o in getattr(ne, "cand", None) or []:            # đã xếp theo khoảng cách
            # xe đỗ vừa thấy, đi ngang thì YOLOE lập vết mới / tốc độ ngang thoáng qua: vẫn là xe đỗ đó
            yen = ne.dung_yen(o) or (nho.tiep(t, o['dist']) and ne.co_the_dang_do(o))
            loai = nho.loai(t, phan_loai_vat(o, do, ne.geo, dv.hw), o['dist']) if yen and not o.get('high') else None
            if loai == "khong_chac" and theo and do is None and o.get('side') == 'phía trước' and o['dist'] <= D_THEO:
                loai = "chan"                                # mất đo vạch, giữa lối đi (lệch trục <= 0,2 m): xe che kín vạch
            if loai == "chan":
                en, vi = ne.ten_vat(o)
                vi = ten_noi(en, vi)
                vat = dict(dist=o['dist'], ten=vi, ben=ne.ben_di_duoc(o), ten_en=en if vi else None,
                           vi_tri=nho.cap_nhat(t, o['dist'], vi_tri_tren_vach(o, do, ne.geo, dv.hw)),
                           duong=getattr(ne, "ti_le_duong", None))
                break
        self.vat_chan_on_dinh = self.on_dinh_vat_chan.cap_nhat(t, vat)
        if not (self.vat_chan_on_dinh or self.bo_vat_chan.dang_xu_ly):
            vat = None                                       # chưa thấy liên tục 1 s: chưa giao cho bộ điều khiển
        cau = self.bo_vat_chan.buoc(t, dv.da_tren_vach(t), vat, do.lech if do is not None else None, do is not None,
                                    theo_vach=theo and do is None)
        dv.giu_im = self.bo_vat_chan.dang_xu_ly
        if cau:
            ne.dang_ky_cau(cau, vat.get('ten_en') if vat else None)
        return cau

    def _vat_ne(self):
        """Vật module né vật cản độ sâu đang theo dõi (để module vạch biết vạch có bị chắn không)."""
        return getattr(self.ne_vat_can, "cand", None) if self.ne_vat_can is not None else None

    def _handle_maneuvering_turn(self, mask, image, frame_visualizations,
                                image_height, image_width):
        """Xử lý trạng thái rẽ"""
        features = self._get_pixel_domain_features(mask, image.shape)
        if not features:
            return "Mất đường đi, đang tìm lại."
        self._add_navigation_info_visualization(features, image_height, image_width, frame_visualizations)
        if self.maneuver_step == MANEUVER_STEP_1_ISSUE_COMMAND:
            direction_text = 'phải' if self.maneuver_target_info['direction'] == 'right' else 'trái'
            guidance_text = f"Hãy dịch sang {direction_text}."
            
            poly_func = features['poly_func']
            y_check = image_height * 0.7
            self.maneuver_target_info['old_path_center_x'] = poly_func(y_check)
            
            self.maneuver_step = MANEUVER_STEP_2_WAIT_FOR_SHIFT
            
            self._add_data_panel(frame_visualizations, {
                "Trạng thái": "Xử lý khúc cua",
                "Hướng dẫn": guidance_text,
                "Bước": "Ra lệnh",
                "Hướng": direction_text
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif self.maneuver_step == MANEUVER_STEP_2_WAIT_FOR_SHIFT:
            old_path_x = self.maneuver_target_info.get('old_path_center_x')
            if old_path_x is None:
                self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
                return ""
            
            poly_func = features['poly_func']
            y_check = image_height * 0.7
            current_path_x = poly_func(y_check)
            shift_distance = abs(current_path_x - old_path_x)
            
            centerline_data = features['centerline_data']
            width_at_check_y = self._get_width_at_y(centerline_data, y_check)
            
            if shift_distance > (width_at_check_y * 0.5):
                guidance_text = "Phát hiện đã di chuyển, bắt đầu căn hướng mới."
                self.maneuver_step = MANEUVER_STEP_3_ALIGN_ON_NEW_PATH
            else:
                direction_text = 'phải' if self.maneuver_target_info['direction'] == 'right' else 'trái'
                guidance_text = f"Hãy tiếp tục dịch sang {direction_text}."
            
            self._add_data_panel(frame_visualizations, {
                "Trạng thái": "Xử lý khúc cua",
                "Hướng dẫn": guidance_text,
                "Bước": "Chờ dịch ngang",
                "Lượng dịch": f"{shift_distance:.1f}px"
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif self.maneuver_step == MANEUVER_STEP_3_ALIGN_ON_NEW_PATH:
            poly_func = features['poly_func']
            y_check = image_height * 0.5
            current_path_x_at_center = poly_func(y_check)
            
            pixel_error = current_path_x_at_center - image_width / 2
            center_offset_ratio = abs(pixel_error) / image_width
            
            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "Đã căn hướng theo đường mới, hãy đi thẳng."
                self.current_state = STATE_NAVIGATING
                self.maneuver_target_info = None
                self.turn_cooldown_frames = self.TURN_COOLDOWN_DURATION
            else:
                move_direction = "phải" if pixel_error > 0 else "trái"
                guidance_text = f"Hãy chỉnh nhẹ sang {move_direction}, căn vào đường dẫn."
            
            self._add_data_panel(frame_visualizations, {
                "Trạng thái": "Xử lý khúc cua",
                "Hướng dẫn": guidance_text,
                "Bước": "Căn vào đường mới",
                "Sai số": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
            
            return guidance_text
    
    def _handle_locking_on(self, frame_visualizations):
        """Xử lý trạng thái khóa"""
        if not self.lock_on_data:
            self.current_state = STATE_NAVIGATING
            return ""
        
        main_obstacle = self.lock_on_data['main_obstacle']
        
        # Thêm hiệu ứng xung
        self._add_obstacle_visualization(main_obstacle, frame_visualizations, pulse_effect=True)
        
        # Kiểm tra thời gian
        if time.time() - self.lock_on_data['start_time'] > 0.7:
            self.avoidance_plan = self.lock_on_data['avoidance_plan']
            self.avoidance_step_index = 0
            self.current_state = STATE_AVOIDING_OBSTACLE
            self.lock_on_data = None
        
        return ""
    
    def _handle_avoiding_obstacle(self, mask, image, frame_visualizations,
                                 image_height, image_width):
        """Xử lý trạng thái tránh vật cản"""
        if not self.avoidance_plan or self.avoidance_step_index >= len(self.avoidance_plan):
            self.current_state = STATE_NAVIGATING
            self.avoidance_plan = None
            return "Đã tránh xong, quay lại đường dẫn."
        
        step = self.avoidance_plan[self.avoidance_step_index]
        
        if step['type'] == 'sidestep_clear':
            direction = step['direction']
            
            if self.obstacle_detector:
                final_obstacles = self._detect_obstacles(image, mask)
            else:
                final_obstacles = []
            
            if final_obstacles:
                guidance_text = f"Đường bị chặn, hãy dịch sang {'phải' if direction == 'right' else 'trái'}."
            else:
                guidance_text = "Được rồi, hãy dừng dịch ngang."
                self.avoidance_step_index += 1
            
            self._add_data_panel(frame_visualizations, {
                "Trạng thái": "Đang tránh vật cản",
                "Hướng dẫn": guidance_text,
                "Bước": "Dịch ngang ra ngoài",
                "Hướng": direction
            }, (25, image_height - 75))
            
            return guidance_text
            
        elif step['type'] == 'forward_pass':
            # Xử lý đơn giản hóa, chuyển thẳng sang bước tiếp theo
            self.avoidance_step_index += 1
            return "Đi thẳng vài bước để vượt qua vật cản, rồi nói \"xong\"."
            
        elif step['type'] == 'sidestep_return':
            direction = step['direction']
            features = self._get_pixel_domain_features(mask, image.shape)
            
            if not features:
                return f"Không thấy đường dẫn, hãy dịch nhẹ sang {'phải' if direction == 'right' else 'trái'}."
            
            poly_func = features['poly_func']
            y_target = image_height * 0.5
            x_target = poly_func(y_target)
            
            center_offset_pixels = x_target - image_width / 2
            center_offset_ratio = abs(center_offset_pixels) / image_width
            
            if center_offset_ratio < self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "Đã quay lại đường dẫn."
                self.avoidance_step_index += 1
            else:
                guidance_text = "Dịch sang phải, căn vào đường dẫn" if center_offset_pixels > 0 else "Dịch sang trái, căn vào đường dẫn"
            
            self._add_data_panel(frame_visualizations, {
                "Trạng thái": "Đang tránh vật cản",
                "Hướng dẫn": guidance_text,
                "Bước": "Quay về đường dẫn",
                "Độ lệch": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
            
            return guidance_text
    
    # ========== Phương thức phụ trợ ==========
    
    def _get_vanishing_point_features(self, mask):
        """Trích xuất đặc trưng điểm triệt tiêu"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: 
                return None
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < 5000: 
                return None
            
            rect = cv2.minAreaRect(main_contour)
            center, _, angle = rect
            angle_rad = np.deg2rad(angle)
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], 
                          [np.sin(angle_rad), np.cos(angle_rad)]])
            points_transformed = np.dot(main_contour.squeeze(1) - center, R)
            left_points = main_contour.squeeze(1)[points_transformed[:, 0] < 0]
            right_points = main_contour.squeeze(1)[points_transformed[:, 0] >= 0]
            
            if len(left_points) < 20 or len(right_points) < 20: 
                return None
            
            [vx_l, vy_l, x_l, y_l] = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)
            [vx_r, vy_r, x_r, y_r] = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            a1, b1, c1 = vy_l, -vx_l, vx_l * y_l - vy_l * x_l
            a2, b2, c2 = vy_r, -vx_r, vx_r * y_r - vy_r * x_r
            determinant = a1 * b2 - a2 * b1
            
            if abs(determinant) < 1e-6: 
                return None
            
            vp_x = (b1 * c2 - b2 * c1) / determinant
            vp_y = (a2 * c1 - a1 * c2) / determinant
            L_center = ((vx_l + vx_r) / 2, (vy_l + vy_r) / 2, (x_l + x_r) / 2, (y_l + y_r) / 2)
            
            total_dist = 0
            for pt in left_points: 
                total_dist += abs((pt[0] - x_l) * vy_l - (pt[1] - y_l) * vx_l)
            for pt in right_points: 
                total_dist += abs((pt[0] - x_r) * vx_r - (pt[1] - y_r) * vy_r)
            fit_error = total_dist / (len(left_points) + len(right_points))
            
            return {"VP": (vp_x, vp_y), "L_center": L_center, "fit_error": fit_error}
        except:
            return None
    
    def _get_pixel_domain_features(self, mask, image_shape):
        """Trích xuất đặc trưng miền điểm ảnh"""
        try:
            height, width = image_shape[:2]
            
            centerline_data = []
            for y in range(height - 1, int(height * 0.3), -5):
                row = mask[y, :]
                x_pixels = np.where(row > 0)[0]
                if x_pixels.size > 10:
                    x_min, x_max = x_pixels[0], x_pixels[-1]
                    path_width = x_max - x_min
                    center_x = (x_min + x_max) / 2
                    centerline_data.append([y, center_x, path_width])
            
            if len(centerline_data) < 20: 
                return None
            
            data = np.array(centerline_data)
            
            # Áp dụng làm mượt đường tâm
            data = self._smooth_centerline(data)
            
            # Phát hiện khúc cua gắt
            sharp_turn_index = self._find_sharp_turn(data)
            if sharp_turn_index is not None:
                cutoff_index = int(sharp_turn_index * 0.6)
                if cutoff_index >= 10:
                    data = data[:cutoff_index]
            
            y_coords, x_coords, widths = data[:, 0], data[:, 1], data[:, 2]
            weights = widths
            
            # Khớp đa thức gốc
            coeffs_raw = np.polyfit(y_coords, x_coords, 2, w=weights)
            
            # [Mới] Làm mượt theo thời gian cho hệ số đa thức
            self.poly_coeffs_history.append(coeffs_raw.copy())
            if len(self.poly_coeffs_history) > self.poly_coeffs_history_max:
                self.poly_coeffs_history.pop(0)
            
            # Dùng trung bình động có trọng số theo hàm mũ để làm mượt hệ số
            if len(self.poly_coeffs_history) >= 3:
                # Trọng số: khung gần nhất có trọng số cao hơn
                weights_time = np.array([0.7 ** (len(self.poly_coeffs_history) - i - 1) 
                                        for i in range(len(self.poly_coeffs_history))])
                weights_time = weights_time / np.sum(weights_time)
                
                # Hệ số trung bình có trọng số
                coeffs = np.zeros_like(coeffs_raw)
                for i, hist_coeffs in enumerate(self.poly_coeffs_history):
                    coeffs += hist_coeffs * weights_time[i]
            else:
                coeffs = coeffs_raw
            
            poly_func = np.poly1d(coeffs)
            
            curvature_proxy = abs(coeffs[0])
            tangent_slope = 2 * coeffs[0] * height + coeffs[1]
            tangent_angle_rad = np.arctan(tangent_slope)
            
            return {
                "poly_func": poly_func,
                "curvature_proxy": curvature_proxy,
                "tangent_angle_rad": tangent_angle_rad,
                "centerline_data": np.array(centerline_data)
            }
        except Exception as e:
            logger.warning(f"Pixel domain feature calculation failed: {e}")
            return None
    
    def _find_sharp_turn(self, data):
        """Tìm điểm khúc cua gắt"""
        window_size = 5
        angle_threshold = 30
        
        for i in range(len(data) - 2 * window_size):
            front_window = data[i:i + window_size]
            back_window = data[i + window_size:i + 2 * window_size]
            
            front_dir = [front_window[-1, 1] - front_window[0, 1],
                        front_window[-1, 0] - front_window[0, 0]]
            back_dir = [back_window[-1, 1] - back_window[0, 1],
                       back_window[-1, 0] - back_window[0, 0]]
            
            angle1 = np.arctan2(front_dir[1], front_dir[0])
            angle2 = np.arctan2(back_dir[1], back_dir[0])
            angle_diff = abs(np.degrees(angle2 - angle1))
            
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > angle_threshold:
                return i + window_size
        
        return None
    
    def _detect_sharp_corner(self, centerline_data, angle_threshold_deg=45):
        """Phát hiện khúc cua gắt"""
        try:
            if len(centerline_data) < 15: 
                return None
            points_in_range = np.array(centerline_data)
            num_points = len(points_in_range)
            
            window_size = max(5, int(num_points * 0.15))
            best_turn_info = None
            max_angle_diff = 0
            
            for i in range(0, num_points - 2 * window_size, 2):
                front_segment = points_in_range[i:i + window_size]
                back_segment = points_in_range[i + window_size:i + 2 * window_size]
                
                if len(front_segment) < 3 or len(back_segment) < 3:
                    continue
                
                front_y = front_segment[:, 0]
                front_x = front_segment[:, 1]
                front_coeffs = np.polyfit(front_y, front_x, 1)
                front_slope = front_coeffs[0]
                
                back_y = back_segment[:, 0]
                back_x = back_segment[:, 1]
                back_coeffs = np.polyfit(back_y, back_x, 1)
                back_slope = back_coeffs[0]
                
                front_angle = np.arctan(front_slope)
                back_angle = np.arctan(back_slope)
                
                angle_diff_rad = back_angle - front_angle
                angle_diff_deg = abs(np.degrees(angle_diff_rad))
                
                if angle_diff_deg > max_angle_diff and angle_diff_deg > angle_threshold_deg:
                    max_angle_diff = angle_diff_deg
                    corner_point_idx = i + window_size
                    corner_point = points_in_range[corner_point_idx]
                    
                    direction = "right" if angle_diff_rad > 0 else "left"
                    
                    post_turn_segment = points_in_range[
                        corner_point_idx:min(corner_point_idx + window_size * 2, num_points)]
                    if len(post_turn_segment) > 0:
                        post_turn_center_x = np.mean(post_turn_segment[:, 1])
                    else:
                        post_turn_center_x = corner_point[1]
                    
                    best_turn_info = {
                        "corner_point_pixel": (corner_point[1], corner_point[0]),
                        "turn_angle": max_angle_diff,
                        "direction": direction,
                        "post_turn_center_x": post_turn_center_x,
                        "corner_point_idx": corner_point_idx
                    }
            
            return best_turn_info
        
        except Exception as e:
            logger.warning(f"Corner detection error: {e}")
            return None
    
    def _update_turn_tracker(self, corner_info):
        """Cập nhật bộ theo dõi khúc cua"""
        detected_direction = corner_info['direction']
        
        if detected_direction == self.turn_detection_tracker['direction']:
            self.turn_detection_tracker['consecutive_hits'] += 1
        else:
            self.turn_detection_tracker['direction'] = detected_direction
            self.turn_detection_tracker['consecutive_hits'] = 1
        
        self.turn_detection_tracker['last_seen_frame'] = self.frame_counter
        self.turn_detection_tracker['corner_info'] = corner_info
    
    def _reset_turn_tracker(self):
        """Đặt lại bộ theo dõi khúc cua"""
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
    
    def _calculate_line_x_at_y(self, line_params, y_target):
        """Tính giá trị x của đường thẳng tại tọa độ y cụ thể"""
        vx, vy, x0, y0 = line_params
        if abs(vy) < 1e-6:
            return None
        t = (y_target - y0) / vy
        x = x0 + t * vx
        return x
    
    def _get_width_at_y(self, centerline_data, y_target):
        """Lấy độ rộng đường đi tại tọa độ y cụ thể"""
        ys = centerline_data[:, 0]
        ws = centerline_data[:, 2]
        idx = np.abs(ys - y_target).argmin()
        return ws[idx]
    
    def _detect_obstacles(self, image, path_mask=None):
        """Phát hiện vật cản"""
        logger.info(f"[_detect_obstacles] Bắt đầu thực thi, Frame={self.frame_counter}, obstacle_detector={'đã nạp' if self.obstacle_detector else 'chưa nạp'}")
        
        if self.obstacle_detector is None:
            logger.warning("[_detect_obstacles] Bộ phát hiện vật cản chưa được nạp!")
            return []
        
        # [Mới] In các lớp trong danh sách trắng (chỉ in ở lần gọi đầu tiên)
        if not hasattr(self, '_classes_printed'):
            self._classes_printed = True
            if hasattr(self.obstacle_detector, 'WHITELIST_CLASSES'):
                logger.info("[_detect_obstacles] ===== Các lớp trong danh sách trắng phát hiện vật cản =====")
                for idx, name in enumerate(self.obstacle_detector.WHITELIST_CLASSES):
                    logger.info(f"  - Lớp {idx}: {name}")
                logger.info(f"[_detect_obstacles] Tổng cộng {len(self.obstacle_detector.WHITELIST_CLASSES)} lớp")
        
        try:
            # [Sửa đổi quan trọng] Dùng phương thức detect() của ObstacleDetectorClient
            # Nó tự động xử lý việc đặt gợi ý văn bản của YOLO-E, phát hiện và lọc cơ bản
            logger.info(f"[_detect_obstacles] Gọi ObstacleDetectorClient.detect()... image.shape={image.shape}")
            detected_obstacles = self.obstacle_detector.detect(image, path_mask=path_mask)
            
            logger.info(f"[_detect_obstacles] ObstacleDetectorClient trả về {len(detected_obstacles)} vật thể")
            
            # ObstacleDetectorClient đã trả về dữ liệu đúng định dạng, gồm:
            # - name: tên vật thể
            # - mask: mặt nạ phân đoạn
            # - area: diện tích
            # - area_ratio: tỷ lệ diện tích
            # - center_x, center_y: tọa độ tâm
            # - bottom_y_ratio: tỷ lệ Y của đáy
            
            # Bổ sung một số trường có thể thiếu nhưng code phía sau cần
            H, W = image.shape[:2]
            for i, obj in enumerate(detected_obstacles):
                # Thêm tọa độ khung bao dùng cho trực quan hóa
                if 'mask' in obj and obj['mask'] is not None:
                    y_coords, x_coords = np.where(obj['mask'] > 0)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                        obj['box_coords'] = (x1, y1, x2, y2)
                        
                        # Bổ sung các trường có thể thiếu
                        if 'y_position_ratio' not in obj:
                            obj['y_position_ratio'] = obj.get('center_y', 0) / H
                        if 'label' not in obj:
                            obj['label'] = obj.get('name', 'unknown')
                        if 'center' not in obj:
                            obj['center'] = (obj.get('center_x', 0), obj.get('center_y', 0))
                        # Thêm một độ tin cậy giả (nếu cần)
                        if 'confidence' not in obj:
                            obj['confidence'] = 0.5  # ObstacleDetectorClient đã lọc bỏ các kết quả độ tin cậy thấp
                
                # Xuất nhật ký chi tiết
                logger.info(f"[_detect_obstacles] Vật thể {i+1}/{len(detected_obstacles)}: ")
                logger.info(f"  - Lớp: {obj.get('name', 'unknown')}")
                logger.info(f"  - Diện tích: {obj.get('area', 0)} pixels ({obj.get('area_ratio', 0):.3f} of image)")
                logger.info(f"  - Điểm tâm: ({obj.get('center_x', 0):.1f}, {obj.get('center_y', 0):.1f})")
                logger.info(f"  - bottom_y_ratio: {obj.get('bottom_y_ratio', 0):.3f} (có ở gần không: {'có' if obj.get('bottom_y_ratio', 0) > 0.7 else 'không'})")
                logger.info(f"  - area_ratio: {obj.get('area_ratio', 0):.3f} (có quá lớn không: {'có' if obj.get('area_ratio', 0) > 0.1 else 'không'})")
            
            # [Lưu ý] ObstacleDetectorClient đã thực hiện các bước lọc sau:
            # 1. Lọc theo kích thước (vật thể quá lớn, diện tích vượt 70% sẽ bị lọc)
            # 2. Lọc theo độ tin cậy (theo biến môi trường AIGLASS_OBS_CONF, mặc định 0.25)
            # 3. Nếu có path_mask thì lọc không gian liên quan đến đường đi:
            #    - Yêu cầu giao với đường đi ít nhất 100 điểm ảnh
            #    - Yêu cầu phần giao chiếm ít nhất 1% diện tích vật thể
            # Nên ở đây không cần lọc thêm
            
            logger.info(f"[_detect_obstacles] ===== Kết quả cuối: trả về {len(detected_obstacles)} vật cản =====")
            for idx, obj in enumerate(detected_obstacles):
                logger.info(f"  {idx+1}. {obj.get('name', 'unknown')} - Vị trí:({obj.get('center_x', 0):.0f},{obj.get('center_y', 0):.0f}) "
                        f"bottom_y_ratio:{obj.get('bottom_y_ratio', 0):.2f} area_ratio:{obj.get('area_ratio', 0):.3f}")
            
            return detected_obstacles
            
        except Exception as e:
            logger.error(f"[_detect_obstacles] Phát hiện vật cản thất bại: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _check_and_set_obstacle_voice(self, obstacles):
        """Kiểm tra vật cản và đặt giọng nói chờ thông báo"""
        if not obstacles:
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None
            return
        
        # Lọc vật cản ở gần (nâng ngưỡng, chỉ rất gần mới cảnh báo)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75  # Nâng lên 0.75, đáy vật cản phải nằm dưới 75% khung hình
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12  # Nâng lên 0.12, vật cản phải chiếm từ 12% khung hình trở lên
        
        near_obstacles = []
        for obs in obstacles:
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD):
                near_obstacles.append(obs)
        
        if near_obstacles:
            # Lấy vật cản chính (diện tích lớn nhất)
            main_obstacle = max(near_obstacles, key=lambda x: x.get('area_ratio', 0))
            obstacle_name = main_obstacle.get('name', '')
            current_time = time.time()
            
            # Kiểm tra có cần thông báo không
            should_announce = False
            if obstacle_name != self.last_obstacle_speech:
                # Vật cản khác nhau, thông báo ngay
                should_announce = True
                self.last_obstacle_speech = obstacle_name
                self.last_obstacle_speech_time = current_time
            elif current_time - self.last_obstacle_speech_time > self.obstacle_speech_cooldown:
                # Cùng một vật cản nhưng quá thời gian hồi, thông báo lại
                should_announce = True
                self.last_obstacle_speech_time = current_time
            
            if should_announce:
                self.pending_obstacle_voice = self._speech_for_obstacle(obstacle_name)
        else:
            # Không có vật cản ở gần
            self.last_obstacle_speech = ""
            self.pending_obstacle_voice = None

    def _check_obstacles(self, image, mask, frame_visualizations):
        """Kiểm tra và xử lý vật cản"""
        # Dùng chiến lược bộ đệm
        if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
            final_obstacles = self._detect_obstacles(image, mask)
            # [Mới] Ổn định hóa vật cản, tránh chồng lặp
            if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                final_obstacles = self._stabilize_obstacle_list(
                    final_obstacles, 
                    self.last_detected_obstacles,
                    self.prev_gray,
                    curr_gray,
                    image.shape[:2]
                )
            self.last_detected_obstacles = final_obstacles
            self.last_obstacle_detection_frame = self.frame_counter
        else:
            if self.frame_counter - self.last_obstacle_detection_frame < self.OBSTACLE_CACHE_DURATION_FRAMES:
                final_obstacles = self.last_detected_obstacles
            else:
                final_obstacles = []
        
        # Thêm trực quan hóa
        for obs in final_obstacles:
            self._add_obstacle_visualization(obs, frame_visualizations)
        
        # Lọc vật cản ở gần (nâng ngưỡng, chỉ rất gần mới cảnh báo)
        NEAR_DISTANCE_Y_THRESHOLD = 0.75  # Nâng lên 0.75, đáy vật cản phải nằm dưới 75% khung hình
        NEAR_DISTANCE_AREA_THRESHOLD = 0.12  # Nâng lên 0.12, vật cản phải chiếm từ 12% khung hình trở lên
        
        near_obstacles = [
            obs for obs in final_obstacles
            if (obs.get('bottom_y_ratio', 0) > NEAR_DISTANCE_Y_THRESHOLD or
                obs.get('area_ratio', 0) > NEAR_DISTANCE_AREA_THRESHOLD)
        ]
        
        return near_obstacles
    
    def _plan_avoidance(self, obstacle_info, image_width):
        """Lập kế hoạch đường tránh vật cản"""
        obstacle_center_x = obstacle_info['center_x']
        image_center_x = image_width / 2
        
        if obstacle_center_x < image_center_x:
            turn_direction = 'right'
        else:
            turn_direction = 'left'
        
        plan = [
            {'type': 'sidestep_clear', 'direction': turn_direction},
            {'type': 'forward_pass'},
            {'type': 'sidestep_return', 'direction': 'left' if turn_direction == 'right' else 'right'}
        ]
        return plan
    
    def _generate_navigation_guidance(self, features, image_height, image_width, frame_visualizations):
        """Tạo chỉ dẫn dẫn đường"""
        poly_func = features['poly_func']
        is_curve = features['curvature_proxy'] > self.CURVATURE_PROXY_THRESHOLD
        lookahead_ratio = 0.6 if is_curve else 0.4
        y_target = image_height * lookahead_ratio
        x_target = poly_func(y_target)
        
        # Thêm trực quan hóa đường tâm
        plot_y = np.arange(int(image_height * 0.3), image_height, 5).astype(int)
        plot_x = poly_func(plot_y).astype(int)
        centerline_points = np.vstack((plot_x, plot_y)).T.tolist()
        frame_visualizations.append({
            "type": "polyline",
            "points": centerline_points,
            "color": "yellow",
            "width": 2
        })
        
        # Thêm điểm mục tiêu
        frame_visualizations.append({
            "type": "circle",
            "center": [int(x_target), int(y_target)],
            "radius": 10,
            "color": "red"
        })
        
        # Tính lệnh dẫn đường (ưu tiên: rẽ/dịch ngang > đi thẳng)
        center_offset_pixels = x_target - image_width / 2
        center_offset_ratio = abs(center_offset_pixels) / image_width
        orientation_error_rad = features['tangent_angle_rad']
        
        # Trước tiên kiểm tra có cần rẽ không (rẽ trái/rẽ phải)
        if orientation_error_rad > self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = "Rẽ trái"
        elif orientation_error_rad < -self.NAV_ORIENTATION_THRESHOLD_RAD:
            guidance_text = "Rẽ phải"
        # Sau đó kiểm tra có cần dịch ngang không (dịch trái/dịch phải)
        elif center_offset_ratio > self.NAV_CENTER_OFFSET_THRESHOLD_RATIO:
            guidance_text = "Dịch phải" if center_offset_pixels > 0 else "Dịch trái"
        # Cuối cùng mới là đi thẳng
        else:
            guidance_text = "Giữ hướng đi thẳng"
        
        # Thêm bảng dữ liệu
        self._add_data_panel(frame_visualizations, {
            "Trạng thái": "Dẫn đường thông thường",
            "Hướng dẫn": guidance_text,
            "Hướng đi": f"{np.degrees(orientation_error_rad):.1f}°",
            "Độ lệch": f"{center_offset_ratio * 100:.1f}%"
        }, (25, image_height - 75))
        
        return guidance_text
    
    def _handle_pixel_domain_onboarding(self, pixel_features, image_height, image_width, frame_visualizations):
        """Xử lý hướng dẫn lên đường dẫn ở miền điểm ảnh"""
        image_center_x = image_width / 2
        orientation_error_rad = pixel_features['tangent_angle_rad']
        poly_func = pixel_features['poly_func']
        
        y_bottom = image_height - 1
        x_target_bottom = poly_func(y_bottom)
        center_offset_pixels = x_target_bottom - image_center_x
        center_offset_ratio = abs(center_offset_pixels) / image_width
        
        if self.onboarding_step == ONBOARDING_STEP_ROTATION:
            if abs(orientation_error_rad) < self.ONBOARDING_ORIENTATION_THRESHOLD_RAD:
                guidance_text = "Đã đúng hướng! Bây giờ hãy chỉnh vị trí."
                self.onboarding_step = ONBOARDING_STEP_TRANSLATION
            else:
                guidance_text = "Hãy xoay sang trái." if orientation_error_rad > 0.1 else "Hãy xoay sang phải."
            
            self._add_data_panel(frame_visualizations, {
                "Trạng thái": "Lên đường dẫn (hướng)",
                "Hướng dẫn": guidance_text,
                "Góc": f"{np.degrees(orientation_error_rad):.1f}°",
                "Độ lệch": "Chờ hiệu chỉnh"
            }, (25, image_height - 75))
            self._add_navigation_info_visualization(pixel_features, image_height, image_width, frame_visualizations)
    
            return guidance_text
            
        elif self.onboarding_step == ONBOARDING_STEP_TRANSLATION:
            if center_offset_ratio < self.ONBOARDING_CENTER_OFFSET_THRESHOLD_RATIO:
                guidance_text = "Chỉnh xong! Bạn đã ở trên đường dẫn, bắt đầu đi."
                self.current_state = STATE_NAVIGATING
            else:
                guidance_text = "Hãy dịch sang phải." if center_offset_pixels > 0 else "Hãy dịch sang trái."
            
            self._add_data_panel(frame_visualizations, {
                "Trạng thái": "Lên đường dẫn (vị trí)",
                "Hướng dẫn": guidance_text,
                "Góc": "Đã căn chỉnh",
                "Độ lệch": f"{center_offset_ratio * 100:.1f}%"
            }, (25, image_height - 75))
        
        return guidance_text
    
    def _add_obstacle_visualization(self, obstacle, visualizations, pulse_effect=False):
        """Thêm trực quan hóa vật cản (bản đơn giản: chỉ viền, gần màu đỏ xa màu vàng)"""
        try:
            # Tính mức nguy hiểm của vật cản
            bottom_y_ratio = obstacle.get('bottom_y_ratio', 0)
            area_ratio = obstacle.get('area_ratio', 0)
            
            # Xác định có phải vật cản ở gần không
            is_near = bottom_y_ratio > 0.7 or area_ratio > 0.1  # Vật cản ở gần
            
            # Thêm viền mask (nếu có)
            if 'mask' in obstacle and obstacle['mask'] is not None:
                mask = obstacle['mask']
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Tìm đường viền lớn nhất
                    max_contour = max(contours, key=cv2.contourArea)
                    points = max_contour.squeeze(1)[::5].tolist()
                    
                    # Chọn màu viền theo khoảng cách: gần màu đỏ, xa màu vàng
                    if is_near:
                        outline_color = "rgba(255, 0, 0, 1.0)"  # Đỏ
                        thickness = 3
                    else:
                        outline_color = "rgba(255, 255, 0, 0.8)"  # Vàng
                        thickness = 2
                    
                    # Chỉ thêm viền, không thêm phần tô và chữ
                    visualizations.append({
                        "type": "outline",
                        "points": points,
                        "color": outline_color,
                        "thickness": thickness
                    })
        except Exception as e:
            logger.error(f"[_add_obstacle_visualization] Thêm trực quan hóa vật cản thất bại: {e}")

    def _add_navigation_info_visualization(self, features, image_height, image_width, frame_visualizations):
        """Thêm trực quan hóa thông tin tính toán dẫn đường"""
        if not features:
            return
        
        try:
            # Lấy kết quả tính toán
            poly_func = features.get('poly_func')
            curvature_proxy = features.get('curvature_proxy', 0)
            tangent_angle_rad = features.get('tangent_angle_rad', 0)
            tangent_angle_deg = np.degrees(tangent_angle_rad)
            
            # Vẽ hướng tiếp tuyến
            if poly_func:
                # Tính tiếp tuyến ở đáy khung hình
                y_bottom = image_height - 50
                x_bottom = poly_func(y_bottom)
                
                # Tính điểm cuối của tiếp tuyến
                tangent_length = 100
                dx = tangent_length * np.cos(tangent_angle_rad)
                dy = tangent_length * np.sin(tangent_angle_rad)
                
                # [Mới] Vẽ đường cơ sở nét đứt (thẳng đứng hướng lên)
                baseline_length = 80
                frame_visualizations.append({
                    "type": "dashed_line",
                    "start": [int(x_bottom), int(y_bottom)],
                    "end": [int(x_bottom), int(y_bottom - baseline_length)],
                    "color": "rgba(255, 255, 255, 0.6)",  # Nét đứt trắng
                    "thickness": 2
                })
                
                # Thêm trực quan hóa tiếp tuyến
                frame_visualizations.append({
                    "type": "arrow",
                    "start": [int(x_bottom), int(y_bottom)],
                    "end": [int(x_bottom + dx), int(y_bottom - dy)],  # Chú ý hướng trục Y
                    "color": "rgba(0, 255, 255, 0.8)",  # Xanh lơ
                    "thickness": 3,
                    "tip_length": 0.3
                })
                
                # [Mới] Vẽ cung biểu thị góc kẹp
                arc_radius = 40
                # Góc đường cơ sở là -90 độ (hướng lên), góc tiếp tuyến là tangent_angle_deg
                # Trong OpenCV góc được đo ngược chiều kim đồng hồ từ đường ngang bên phải
                start_angle = -90  # Đường cơ sở (thẳng đứng hướng lên)
                end_angle = -90 + tangent_angle_deg  # Góc tiếp tuyến
                frame_visualizations.append({
                    "type": "angle_arc",
                    "center": [int(x_bottom), int(y_bottom)],
                    "radius": arc_radius,
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "color": "rgba(255, 200, 0, 0.8)",  # Vàng cam
                    "thickness": 2
                })
                
                # Thêm chữ góc (cỡ chữ giảm một nửa)
                frame_visualizations.append({
                    "type": "text_with_bg",
                    "text": f"Góc: {tangent_angle_deg:.1f}°",
                    "position": [int(x_bottom + 10), int(y_bottom - 30)],
                    "font_scale": 0.3,  # Giảm một nửa từ 0.6 xuống 0.3
                    "color": "rgba(255, 255, 255, 1.0)",
                    "bg_color": "rgba(0, 0, 0, 0.7)"
                })
            
            # Thêm thông tin độ cong (cỡ chữ giảm một nửa)
            if curvature_proxy > 0.00001:
                curve_text = "Khúc cua" if curvature_proxy > 0.00005 else "Cua thoải"
                frame_visualizations.append({
                    "type": "text_with_bg",
                    "text": f"{curve_text}: {curvature_proxy:.2e}",
                    "position": [20, 100],
                    "font_scale": 0.25,  # Giảm một nửa từ 0.5 xuống 0.25
                    "color": "rgba(255, 255, 0, 1.0)",
                    "bg_color": "rgba(0, 0, 0, 0.7)"
                })
                
            # Hiển thị các điểm dữ liệu đường tâm
            if 'centerline_data' in features:
                centerline_data = features['centerline_data']
                # Hiển thị độ rộng đường đi ở giữa khung hình
                mid_idx = len(centerline_data) // 2
                if mid_idx < len(centerline_data):
                    y, x, width = centerline_data[mid_idx]
                    # Vẽ đường chỉ độ rộng (đổi thành mũi tên hai chiều)
                    frame_visualizations.append({
                        "type": "double_arrow",  # Loại mũi tên hai chiều mới
                        "start": [int(x - width/2), int(y)],
                        "end": [int(x + width/2), int(y)],
                        "color": "rgba(0, 255, 0, 0.8)",
                        "thickness": 2,
                        "tip_length": 0.15
                    })
                    # Thêm chữ độ rộng (cỡ chữ giảm một nửa)
                    frame_visualizations.append({
                        "type": "text_with_bg",
                        "text": f"Độ rộng: {width:.0f}px",
                        "position": [int(x - 30), int(y - 10)],
                        "font_scale": 0.25,  # Giảm một nửa từ 0.5 xuống 0.25
                        "color": "rgba(255, 255, 255, 1.0)",
                        "bg_color": "rgba(0, 0, 0, 0.7)"
                    })
        except Exception as e:
            logger.error(f"Thêm trực quan hóa thông tin dẫn đường thất bại: {e}")

    def _add_data_panel(self, visualizations, data, position):
        """Thêm bảng dữ liệu"""
        visualizations.append({
            "type": "data_panel",
            "data": data,
            "position": position
        })
    
    def _add_crosswalk_info_visualization(self, viz_data, image_height, image_width, visualizations):
        """Thêm trực quan hóa đẹp mắt cho thông tin phát hiện vạch qua đường"""
        try:
            # 1. Vẽ dấu tâm vạch qua đường (chữ thập lớn)
            center_x = int(viz_data['center_x_ratio'] * image_width)
            center_y = int(viz_data['center_y_ratio'] * image_height)
            
            cross_size = 20 if viz_data['in_arrival'] else 15  # Giảm kích thước
            cross_color = "rgba(255, 100, 0, 1.0)" if viz_data['in_arrival'] else "rgba(0, 200, 255, 0.8)"
            
            # Đường ngang
            visualizations.append({
                "type": "line",
                "start": [center_x - cross_size, center_y],
                "end": [center_x + cross_size, center_y],
                "color": cross_color,
                "thickness": 2  # Làm mảnh hơn
            })
            # Đường dọc
            visualizations.append({
                "type": "line",
                "start": [center_x, center_y - cross_size],
                "end": [center_x, center_y + cross_size],
                "color": cross_color,
                "thickness": 2  # Làm mảnh hơn
            })
            
            # 2. Vẽ mũi tên chỉ về vạch qua đường (từ tâm khung hình tới tâm vạch qua đường)
            screen_center_x = image_width // 2
            screen_center_y = image_height // 2
            
            # Chỉ vẽ mũi tên khi vạch qua đường không nằm ở tâm khung hình
            distance = np.sqrt((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)
            if distance > 80:  # Nâng lên 80 điểm ảnh mới vẽ mũi tên (giảm gây nhiễu)
                visualizations.append({
                    "type": "arrow",
                    "start": [screen_center_x, screen_center_y],
                    "end": [center_x, center_y],
                    "color": "rgba(255, 150, 0, 0.6)",  # Giảm độ trong suốt
                    "thickness": 2,  # Làm mảnh hơn
                    "tip_length": 0.15  # Giảm kích thước mũi tên
                })
            
            # 3. Thêm bảng thông tin (góc trên phải)
            panel_x = image_width - 180
            panel_y = 20
            
            # Chuẩn bị dữ liệu bảng
            panel_data = {
                "Vạch qua đường": viz_data['stage'],
                "Diện tích": f"{viz_data['area_ratio']*100:.1f}%",
                "Phương vị": viz_data['position'],
            }
            
            if viz_data['has_occlusion']:
                panel_data["Trạng thái"] = "Bị che khuất"
            elif viz_data['in_arrival']:
                panel_data["Trạng thái"] = "Có thể qua đường"
            
            visualizations.append({
                "type": "data_panel",
                "data": panel_data,
                "position": (panel_x, panel_y)
            })
            
            # 4. Thêm thanh tiến trình diện tích (trực quan hóa độ lớn diện tích)
            bar_width = 150
            bar_height = 20
            bar_x = image_width - bar_width - 20
            bar_y = panel_y + 90
            
            # Khung nền
            visualizations.append({
                "type": "rectangle",
                "top_left": (bar_x, bar_y),
                "bottom_right": (bar_x + bar_width, bar_y + bar_height),
                "color": "rgba(50, 50, 50, 0.7)",
                "filled": True
            })
            
            # Phần tô tiến trình (0-100%, nhưng tối đa hiển thị tới ngưỡng arrival 0.25 tương ứng 100%)
            progress = min(viz_data['area_ratio'] / 0.25, 1.0)
            fill_width = int(bar_width * progress)
            
            # Chọn màu theo giai đoạn
            if viz_data['in_arrival']:
                fill_color = "rgba(0, 255, 100, 0.8)"  # Xanh lá (có thể qua đường)
            elif viz_data['area_ratio'] >= 0.18:
                fill_color = "rgba(255, 200, 0, 0.8)"  # Vàng (đến gần)
            elif viz_data['area_ratio'] >= 0.08:
                fill_color = "rgba(0, 200, 255, 0.8)"  # Xanh lơ (tiến lại gần)
            else:
                fill_color = "rgba(100, 150, 255, 0.8)"  # Xanh dương (phát hiện)
            
            visualizations.append({
                "type": "rectangle",
                "top_left": (bar_x + 2, bar_y + 2),
                "bottom_right": (bar_x + fill_width - 2, bar_y + bar_height - 2),
                "color": fill_color,
                "filled": True
            })
            
            # Nhãn thanh tiến trình (dùng văn bản tiếng Việt, giảm cỡ chữ)
            visualizations.append({
                "type": "text_with_bg",
                "text": f"Độ gần: {int(progress * 100)}%",
                "position": [bar_x, bar_y - 18],
                "font_scale": 0.25,  # Giảm cỡ chữ
                "color": "rgba(255, 255, 255, 1.0)",
                "bg_color": "rgba(0, 0, 0, 0.7)"
            })
            
        except Exception as e:
            logger.error(f"Thêm trực quan hóa vạch qua đường thất bại: {e}")
    
    def _add_traffic_light_visualization(self, color, visualizations, image_height, image_width):
        """Thêm trực quan hóa trạng thái đèn giao thông"""
        # Vẽ chỉ báo đèn giao thông ở góc trên phải
        x = image_width - 100
        y = 50
        
        # Khung nền
        visualizations.append({
            "type": "rectangle",
            "top_left": (x - 40, y - 40),
            "bottom_right": (x + 40, y + 100),
            "color": "rgba(0, 0, 0, 0.5)",
            "filled": True
        })
        
        # Ba đèn tròn
        colors = {
            "red": [(255, 0, 0), (50, 0, 0), (50, 0, 0)],
            "yellow": [(50, 50, 0), (255, 255, 0), (50, 50, 0)],
            "green": [(0, 50, 0), (0, 50, 0), (0, 255, 0)],
            "unknown": [(50, 50, 50), (50, 50, 50), (50, 50, 50)]
        }
        
        light_colors = colors.get(color, colors["unknown"])
        positions = [y - 20, y + 20, y + 60]
        
        for i, (pos_y, light_color) in enumerate(zip(positions, light_colors)):
            # Vòng ngoài
            visualizations.append({
                "type": "circle",
                "center": [x, pos_y],
                "radius": 18,
                "color": f"rgba(100, 100, 100, 1.0)",
                "thickness": 2
            })
            # Vòng trong (màu của đèn)
            visualizations.append({
                "type": "circle",
                "center": [x, pos_y],
                "radius": 15,
                "color": f"rgba({light_color[0]}, {light_color[1]}, {light_color[2]}, 1.0)",
                "filled": True
            })
        
        # Nhãn
        visualizations.append({
            "type": "text_with_bg",
            "text": f"Đèn tín hiệu: {color}",
            "position": [x - 35, y + 90],
            "font_scale": 0.5,
            "color": "rgba(255, 255, 255, 1.0)",
            "bg_color": "rgba(0, 0, 0, 0.7)"
        })
    
    def _to_cn_obstacle(self, name: str) -> str:
        """Chuyển tên vật cản sang tiếng Việt"""
        try:
            key = (name or '').strip().lower()
            return _OBSTACLE_NAME_CN.get(key, 'vật cản')
        except:
            return 'vật cản'

    def _speech_for_obstacle(self, name: str) -> str:
        k = (name or '').strip().lower()
        if k == 'person': return "Phía trước có người, chú ý tránh."
        if k == 'car': return "Phía trước có xe, chú ý tránh."
        if k == 'bicycle': return "Phía trước có xe đạp, dừng lại."
        if k == 'motorcycle': return "Phía trước có xe máy, dừng lại."
        if k == 'bus': return "Phía trước có xe buýt, dừng lại."
        if k == 'truck': return "Phía trước có xe tải, dừng lại."
        if k == 'scooter': return "Phía trước có xe điện, dừng lại."
        if k == 'stroller': return "Phía trước có xe đẩy em bé, dừng lại."
        if k == 'dog': return "Phía trước có chó, dừng lại."
        if k == 'animal': return "Phía trước có động vật, dừng lại."
        return "Phía trước có vật cản, chú ý tránh."

    def _draw_command_button(self, image, text):
        """Vẽ nút lệnh ở giữa phía dưới (thống nhất với chế độ vạch qua đường)"""
        try:
            H, W = image.shape[:2]
            full_text = f"Lệnh hiện tại: {text if text else '—'}"
            
            # Tham số nút
            font_px = 14
            pad_x, pad_y = 14, 8
            bottom_margin = 28
            
            # Tính kích thước chữ
            if PIL_AVAILABLE:
                try:
                    from PIL import Image as PILImage, ImageDraw, ImageFont
                    # Thử nạp phông chữ tiếng Trung
                    font = None
                    for font_path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
                        if os.path.exists(font_path):
                            try:
                                font = ImageFont.truetype(font_path, font_px)
                                break
                            except:
                                continue
                    if font:
                        bbox = ImageDraw.Draw(PILImage.new('RGB', (1, 1))).textbbox((0, 0), full_text, font=font)
                        tw = max(1, bbox[2] - bbox[0])
                        th = max(1, bbox[3] - bbox[1])
                    else:
                        scale = font_px / 24.0
                        (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                except:
                    scale = font_px / 24.0
                    (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            else:
                scale = font_px / 24.0
                (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            
            # Tính vị trí nút (căn giữa phía dưới)
            bw = tw + pad_x * 2
            bh = th + pad_y * 2
            radius = max(10, bh // 2)
            
            cx = W // 2
            left = max(8, cx - bw // 2)
            top = H - bottom_margin - bh
            right = min(W - 8, left + bw)
            bottom = top + bh
            
            # Vẽ nền bo góc bán trong suốt
            overlay = image.copy()
            bg_color = (26, 32, 41)  # Nền tối
            border_color = (60, 76, 102)  # Viền
            
            # Hình chữ nhật bo góc (giữa + hai hình tròn)
            cv2.rectangle(overlay, (left + radius, top), (right - radius, bottom), bg_color, -1)
            cv2.circle(overlay, (left + radius, (top + bottom) // 2), radius, bg_color, -1)
            cv2.circle(overlay, (right - radius, (top + bottom) // 2), radius, bg_color, -1)
            
            # Trộn bán trong suốt
            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
            
            # Vẽ viền
            cv2.rectangle(image, (left + radius, top), (right - radius, bottom), border_color, 1)
            cv2.circle(image, (left + radius, (top + bottom) // 2), radius, border_color, 1)
            cv2.circle(image, (right - radius, (top + bottom) // 2), radius, border_color, 1)
            
            # Vẽ chữ
            text_x = left + pad_x
            text_y = top + pad_y + th
            
            if PIL_AVAILABLE and 'font' in locals() and font:
                # Dùng PIL vẽ chữ Trung
                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((text_x, top + pad_y), full_text, font=font, fill=(255, 255, 255))
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                # Dùng OpenCV vẽ
                cv2.putText(image, full_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
            
            return image
        except Exception as e:
            logger.error(f"Vẽ nút lệnh thất bại: {e}")
            return image
    
    def _parse_color(self, color_str):
        """Phân tích chuỗi màu, trả về định dạng BGR"""
        try:
            if color_str.startswith('rgba('):
                values = color_str[5:-1].split(',')
                r, g, b = int(values[0]), int(values[1]), int(values[2])
                return (b, g, r)  # OpenCV dùng định dạng BGR
            elif color_str == 'yellow':
                return (0, 255, 255)
            elif color_str == 'red':
                return (0, 0, 255)
            else:
                return (0, 0, 255)  # Mặc định màu đỏ
        except:
            return (0, 0, 255)

    def _draw_data_panel_no_bg(self, image, data, position=(15, 15)):
        """Vẽ bảng dữ liệu (phiên bản không nền đen)"""
        if not PIL_AVAILABLE:
            return image
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")
            
            env_scale = float(os.getenv("AIGLASS_PANEL_SCALE", "0.7"))
            base_font_size = max(10, int(round(14 * env_scale)))
            
            # Thử nhiều phông chữ, đảm bảo hiển thị được chữ Trung
            font = None
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
                "C:/Windows/Fonts/simsun.ttc",    # SimSun
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Vẽ chữ, dùng hiệu ứng viền
            y_offset = position[1]
            for key, value in data.items():
                text = f"{key}: {value}"
                
                # Vẽ viền đen (8 hướng)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((position[0] + dx, y_offset + dy), text, 
                                    font=font, fill=(0, 0, 0, 255))
                
                # Vẽ chữ trắng
                draw.text((position[0], y_offset), text, font=font, fill=(255, 255, 255, 255))
                y_offset += base_font_size + 5
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        except Exception as e:
            logger.warning(f"Vẽ bảng dữ liệu thất bại: {e}")
            return image


    def _draw_visualizations(self, image, viz_elements):
        """Phương pháp vẽ trực quan nâng cao"""
        if not viz_elements:
            return image
        
        # Lấy thời gian hiện tại để làm hiệu ứng hoạt ảnh
        current_time = time.time()
        
        # Tách các loại phần tử khác nhau
        panel_elements = [v for v in viz_elements if v.get("type") == "data_panel"]
        standard_elements = [v for v in viz_elements if v.get("type") != "data_panel"]
        
        # Lượt đầu: vẽ phần tô (hiệu ứng bán trong suốt)
        for element in standard_elements:
            elem_type = element.get("type")
            
            if elem_type in ['blind_path_mask', 'obstacle_mask', 'crosswalk_mask']:
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.5)"))
                    
                    # Xử lý hiệu ứng xung
                    if element.get("effect") == "pulse":
                        pulse_speed = element.get("pulse_speed", 1.0)
                        alpha = 0.3 + 0.3 * np.sin(current_time * pulse_speed * 2 * np.pi)
                    else:
                        alpha = 0.4
                    
                    # Lấy khung bao của đa giác
                    x, y, w, h = cv2.boundingRect(points)
                    
                    # Đảm bảo khung bao nằm trong phạm vi ảnh
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    if w > 0 and h > 0:
                        # Tạo mặt nạ nhị phân, chỉ bằng 1 bên trong đa giác
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        local_points = points - np.array([x, y])
                        cv2.fillPoly(binary_mask, [local_points], 255)
                        
                        # Chỉ trộn màu bên trong đa giác
                        local_region = image[y:y+h, x:x+w].copy()
                        
                        # Tạo lớp phủ màu
                        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                        color_overlay[:] = color
                        
                        # Dùng mặt nạ nhị phân để trộn
                        for c in range(3):
                            local_region[:, :, c] = np.where(
                                binary_mask > 0,
                                (1 - alpha) * local_region[:, :, c] + alpha * color_overlay[:, :, c],
                                local_region[:, :, c]
                            )
                        
                        # Đặt vùng cục bộ đã trộn trở lại ảnh gốc
                        image[y:y+h, x:x+w] = local_region
        
        # Lượt hai: vẽ đường viền và các phần tử khác
        for element in standard_elements:
            elem_type = element.get("type")
            
            # [Mới] Vẽ đường thẳng
            if elem_type == 'line':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                cv2.line(image, start, end, color, thickness)
            
            # Vẽ đường viền/nét
            elif elem_type == 'outline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                    thickness = element.get("thickness", 3)
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
            
            # Vẽ đường gấp khúc
            elif elem_type == 'polyline':
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 0, 1.0)"))
                    thickness = element.get("width", 2)
                    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)
            
            # Vẽ hình tròn
            elif elem_type == 'circle':
                center = tuple(element.get("center", (0, 0)))
                radius = element.get("radius", 10)
                color = self._parse_color(element.get("color", "rgba(255, 0, 0, 1.0)"))
                thickness = element.get("thickness", -1 if element.get("filled", True) else 2)
                cv2.circle(image, center, radius, color, thickness)
            
            # Vẽ hình chữ nhật
            elif elem_type == 'rectangle':
                top_left = tuple(element.get("top_left", (0, 0)))
                bottom_right = tuple(element.get("bottom_right", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 0, 0, 0.5)"))
                thickness = -1 if element.get("filled", True) else 2
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            
            # Vẽ mũi tên
            elif elem_type == 'arrow':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.3)
                cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip_length)
            
            # [Mới] Vẽ mũi tên hai chiều
            elif elem_type == 'double_arrow':
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.15)
                # Vẽ đường ở giữa
                cv2.line(image, start, end, color, thickness)
                # Vẽ mũi tên ở hai đầu
                # Tính vectơ hướng mũi tên
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Vectơ hướng đơn vị
                    ux, uy = dx/length, dy/length
                    # Độ dài mũi tên
                    arrow_len = length * tip_length
                    # Mũi tên đầu trái
                    tip1_x = int(start[0] + arrow_len * ux)
                    tip1_y = int(start[1] + arrow_len * uy)
                    # Vẽ mũi tên đầu trái (chỉ sang trái)
                    angle = np.arctan2(dy, dx)
                    arrow_angle = 30 * np.pi / 180  # Góc mũi tên
                    p1 = (int(start[0] + arrow_len * np.cos(angle - arrow_angle)),
                          int(start[1] + arrow_len * np.sin(angle - arrow_angle)))
                    p2 = (int(start[0] + arrow_len * np.cos(angle + arrow_angle)),
                          int(start[1] + arrow_len * np.sin(angle + arrow_angle)))
                    cv2.line(image, start, p1, color, thickness)
                    cv2.line(image, start, p2, color, thickness)
                    # Mũi tên đầu phải (chỉ sang phải)
                    p3 = (int(end[0] - arrow_len * np.cos(angle - arrow_angle)),
                          int(end[1] - arrow_len * np.sin(angle - arrow_angle)))
                    p4 = (int(end[0] - arrow_len * np.cos(angle + arrow_angle)),
                          int(end[1] - arrow_len * np.sin(angle + arrow_angle)))
                    cv2.line(image, end, p3, color, thickness)
                    cv2.line(image, end, p4, color, thickness)
            
            # [Mới] Vẽ nét đứt
            elif elem_type == 'dashed_line':
                start = np.array(element.get("start", (0, 0)))
                end = np.array(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.6)"))
                thickness = element.get("thickness", 2)
                dash_length = 10
                gap_length = 5
                # Tính tổng độ dài và hướng
                total_vec = end - start
                total_len = np.linalg.norm(total_vec)
                if total_len > 0:
                    unit_vec = total_vec / total_len
                    # Vẽ các đoạn nét đứt
                    current_len = 0
                    while current_len < total_len:
                        seg_start = start + unit_vec * current_len
                        seg_end = start + unit_vec * min(current_len + dash_length, total_len)
                        cv2.line(image, tuple(seg_start.astype(int)), tuple(seg_end.astype(int)), color, thickness)
                        current_len += dash_length + gap_length
            
            # [Mới] Vẽ cung góc
            elif elem_type == 'angle_arc':
                center = tuple(element.get("center", (100, 100)))
                radius = element.get("radius", 40)
                start_angle = element.get("start_angle", -90)
                end_angle = element.get("end_angle", 0)
                color = self._parse_color(element.get("color", "rgba(255, 200, 0, 0.8)"))
                thickness = element.get("thickness", 2)
                # Hàm ellipse của OpenCV: startAngle và endAngle được đo theo chiều kim đồng hồ từ đường ngang bên phải
                # Cần chuyển đổi: góc của chúng ta đo ngược chiều kim đồng hồ từ đường ngang bên phải (chuẩn toán học)
                # OpenCV cần đo theo chiều kim đồng hồ từ đường ngang bên phải
                cv2_start = -end_angle  # Chuyển sang định dạng OpenCV
                cv2_end = -start_angle
                # Đảm bảo phạm vi góc đúng
                if cv2_start > cv2_end:
                    cv2_start, cv2_end = cv2_end, cv2_start
                cv2.ellipse(image, center, (radius, radius), 0, cv2_start, cv2_end, color, thickness)
            
            # [Sửa đổi] Vẽ văn bản có nền (hỗ trợ chữ Trung)
            elif elem_type == 'text_with_bg':
                text = element.get("text", "")
                pos = element.get("position", [10, 30])
                font_scale = element.get("font_scale", 0.6)
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                
                # Dùng hàm vẽ chữ mới hỗ trợ chữ Trung
                image = self._draw_chinese_text(image, text, tuple(pos), 
                                              font_scale=font_scale, 
                                              color=color,
                                              stroke_color=(0, 0, 0),
                                              stroke_width=1)
            
            # Vẽ biểu tượng cảnh báo
            elif elem_type == 'warning_icon':
                pos = element.get("position", (100, 100))
                level = element.get("level", "info")
                text = element.get("text", "")
                flash = element.get("flash", False)
                
                # Chọn màu theo cấp độ
                if level == "danger":
                    icon_color = (0, 0, 255)  # Đỏ
                    text_color = (255, 255, 255)
                elif level == "warning":
                    icon_color = (0, 165, 255)  # Cam
                    text_color = (255, 255, 255)
                else:
                    icon_color = (0, 255, 255)  # Vàng
                    text_color = (0, 0, 0)
                
                # Hiệu ứng nhấp nháy
                if flash:
                    alpha = 0.5 + 0.5 * np.sin(current_time * 4 * np.pi)
                    icon_color = tuple(int(c * alpha) for c in icon_color)
                
                # Vẽ biểu tượng cảnh báo hình tam giác
                triangle = np.array([
                    [pos[0], pos[1] - 20],
                    [pos[0] - 15, pos[1]],
                    [pos[0] + 15, pos[1]]
                ], np.int32)
                cv2.fillPoly(image, [triangle], icon_color)
                cv2.polylines(image, [triangle], True, (255, 255, 255), 2)
                
                # Vẽ dấu chấm than
                cv2.putText(image, "!", (pos[0] - 5, pos[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Vẽ nhãn văn bản (hỗ trợ chữ Trung)
                if text:
                    font_scale = 0.5
                    # Dùng hàm vẽ chữ mới hỗ trợ chữ Trung
                    text_pos = (pos[0] - 20, pos[1] + 20)  # Đơn giản hóa tính vị trí
                    image = self._draw_chinese_text(image, text, text_pos,
                                                  font_scale=font_scale,
                                                  color=text_color,
                                                  stroke_color=(0, 0, 0),
                                                  stroke_width=1)
            
            # Văn bản thông thường
            elif elem_type == 'text':
                text = element.get("text", "")
                pos = tuple(element.get("pos", (10, 30)))
                # Dùng hàm vẽ chữ hỗ trợ chữ Trung
                image = self._draw_chinese_text(image, text, pos,
                                              font_scale=0.7,
                                              color=(255, 255, 255),
                                              stroke_color=(0, 0, 0),
                                              stroke_width=1)
        
        # [Sửa đổi] Vẽ bảng dữ liệu (dùng phiên bản không nền đen)
        if PIL_AVAILABLE:
            for panel in panel_elements:
                image = self._draw_data_panel_no_bg(image, panel["data"], panel["position"])
        else:
            # Nếu không có PIL cũng dùng hiệu ứng viền
            for panel in panel_elements:
                y_offset = panel["position"][1]
                for key, value in panel["data"].items():
                    text = f"{key}: {value}"
                    # Vẽ viền chữ
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(image, text, (panel["position"][0] + dx, y_offset + dy), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    # Vẽ chữ trắng
                    cv2.putText(image, text, (panel["position"][0], y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25
        
        return image


    
    def _draw_chinese_text(self, image, text, position, font_scale=0.6, color=(255, 255, 255), 
                         stroke_color=(0, 0, 0), stroke_width=1):
        """Vẽ chữ Việt, dùng phông chữ hệ thống, chữ trắng viền đen"""
        if not PIL_AVAILABLE:
            # Nếu không có PIL thì quay về cv2.putText (sẽ hiện dấu hỏi)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            return image
        
        try:
            # Chuyển sang ảnh PIL
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Tính cỡ chữ (dựa trên font_scale)
            base_size = 24  # Cỡ chữ cơ bản
            font_size = int(base_size * font_scale / 0.6)
            
            # Thử nạp phông Microsoft YaHei
            font = None
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/msyh.ttf",      # Microsoft YaHei bản cũ
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Chuyển màu BGR của OpenCV sang RGB
            rgb_color = (color[2], color[1], color[0])
            rgb_stroke = (stroke_color[2], stroke_color[1], stroke_color[0])
            
            # Vẽ chữ (có hiệu ứng viền)
            x, y = position
            # Vẽ viền
            draw.text((x, y), text, font=font, fill=rgb_stroke, 
                     stroke_width=stroke_width, stroke_fill=rgb_stroke)
            # Vẽ chữ chính
            draw.text((x, y), text, font=font, fill=rgb_color)
            
            # Chuyển lại định dạng OpenCV
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.warning(f"Vẽ chữ thất bại: {e}")
            # Quay về cv2.putText
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 2)
            return image

    def _draw_data_panel(self, image, data, position=(15, 15)):
        """Vẽ bảng dữ liệu (cần Pillow)"""
        if not PIL_AVAILABLE:
            return image
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")
            
            env_scale = float(os.getenv("AIGLASS_PANEL_SCALE", "0.65"))
            base_font_size = max(8, int(round(16 * env_scale)))
            padding = max(4, int(round(8 * env_scale)))
            
            # Thử nạp phông Microsoft YaHei
            font = None
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
                "C:/Windows/Fonts/msyh.ttf",      # Microsoft YaHei bản cũ
                "C:/Windows/Fonts/simhei.ttf",    # SimHei
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            text_lines = [f"{key}: {value}" for key, value in data.items()]
            text_to_draw = "\n".join(text_lines)
            
            bbox = draw.textbbox(position, text_to_draw, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            bg_rect = [
                (position[0] - padding, position[1] - padding),
                (position[0] + text_w + padding, position[1] + text_h + padding)
            ]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 128))
            draw.text(position, text_to_draw, font=font, fill=(255, 255, 255, 255))
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        except Exception:
            return image
    
    def reset(self):
        """Đặt lại trạng thái bộ dẫn đường"""
        self.current_state = STATE_ONBOARDING
        if self.dan_vach is not None:
            from dan_theo_vach import DanTheoVach
            self.dan_vach = DanTheoVach()
            self.current_state = STATE_NAVIGATING
        if self.bo_vat_chan is not None:
            from vat_chan_vach import BoVatChan, OnDinh, NhoVatChan
            self.bo_vat_chan, self.on_dinh_vat_chan, self.vat_chan_on_dinh = BoVatChan(), OnDinh(), False
            self.nho_vat_chan = NhoVatChan()
        self.onboarding_step = ONBOARDING_STEP_ROTATION
        self.maneuver_step = MANEUVER_STEP_1_ISSUE_COMMAND
        self.maneuver_target_info = None
        self.turn_detection_tracker = {
            'direction': None,
            'consecutive_hits': 0,
            'last_seen_frame': 0,
            'corner_info': None
        }
        self.turn_cooldown_frames = 0
        self.avoidance_plan = None
        self.avoidance_step_index = 0
        self.lock_on_data = None
        
        # Đặt lại phần optical flow và làm mượt
        self.flow_points = {}
        self.flow_grace = {}
        self.centerline_history = []
        self.blind_miss_ttl = 0
        self.cross_miss_ttl = 0
        
        # Đặt lại phần giọng nói
        self.pending_obstacle_voice = None
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        
        # Đặt lại lịch sử hệ số đa thức
        self.poly_coeffs_history = []
        self.crosswalk_tracker = {
            'stage': 'not_detected',
            'consecutive_frames': 0,
            'last_area_ratio': 0.0,
            'last_bottom_y_ratio': 0.0,
            'last_center_x_ratio': 0.5,
            'position_announced': False,
            'alignment_status': 'not_aligned',
            'last_seen_frame': 0,
            'last_angle': 0.0
        }
        self.frame_counter = 0
        self.prev_gray = None
        self.prev_blind_path_mask = None
        self.prev_crosswalk_mask = None
        self.prev_obstacle_cache = []
        self.last_guidance_message = ""
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.last_obstacle_speech = ""
        self.last_obstacle_speech_time = 0
        self.last_any_speech_time = 0
        self.crosswalk_ready_announced = False
        self.crosswalk_ready_time = 0
        self.traffic_light_history.clear()
        self.last_traffic_light_state = "unknown"
        self.green_light_announced = False
    
    def _stabilize_obstacle_list(self, obstacles, prev_obstacles, prev_gray, curr_gray, 
                                image_shape, threshold=0.5):
        """Ổn định kết quả phát hiện vật cản, tránh chồng lặp"""
        if not obstacles or prev_gray is None or curr_gray is None:
            return obstacles
        
        H, W = image_shape
        stabilized = []
        used_prev = set()  # Ghi lại các vật cản lịch sử đã dùng
        
        # Với mỗi vật cản đang được phát hiện
        for curr_obs in obstacles:
            if 'mask' not in curr_obs or curr_obs['mask'] is None:
                stabilized.append(curr_obs)
                continue
                
            curr_mask = curr_obs['mask']
            best_match = None
            best_iou = 0
            best_idx = -1
            
            # Tìm vật cản lịch sử khớp nhất
            if prev_obstacles:
                for idx, prev_obs in enumerate(prev_obstacles):
                    if idx in used_prev or 'mask' not in prev_obs:
                        continue
                    
                    # Dùng optical flow dự đoán vị trí mới của vật cản lịch sử
                    flow_mask = self._predict_mask_with_flow(prev_obs['mask'], prev_gray, curr_gray)
                    if flow_mask is None:
                        flow_mask = prev_obs['mask']
                    
                    # Tính IoU
                    inter = np.logical_and(curr_mask > 0, flow_mask > 0).sum()
                    union = np.logical_or(curr_mask > 0, flow_mask > 0).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    
                    if iou > best_iou and iou > threshold:
                        best_iou = iou
                        best_match = flow_mask
                        best_idx = idx
            
            # Nếu tìm được khớp thì hòa trộn kết quả
            if best_match is not None and best_idx >= 0:
                used_prev.add(best_idx)
                # Hòa trộn phát hiện hiện tại và dự đoán optical flow để tăng độ ổn định
                fused_mask = ((0.8 * curr_mask + 0.2 * best_match) > 128).astype(np.uint8) * 255
                curr_obs['mask'] = fused_mask
                # Cập nhật thuộc tính dẫn xuất
                self._update_obstacle_properties(curr_obs, H, W)
            
            stabilized.append(curr_obs)
        
        return stabilized
  
    def _speech_for_obstacle(self, name: str) -> str:
        k = (name or '').strip().lower()
        if k == 'person': return "Phía trước có người, chú ý tránh."
        if k == 'car': return "Phía trước có xe, chú ý tránh."
        if k == 'bicycle': return "Phía trước có xe đạp, dừng lại."
        if k == 'motorcycle': return "Phía trước có xe máy, dừng lại."
        if k == 'bus': return "Phía trước có xe buýt, dừng lại."
        if k == 'truck': return "Phía trước có xe tải, dừng lại."
        if k == 'scooter': return "Phía trước có xe điện, dừng lại."
        if k == 'stroller': return "Phía trước có xe đẩy em bé, dừng lại."
        if k == 'dog': return "Phía trước có chó, dừng lại."
        if k == 'animal': return "Phía trước có động vật, dừng lại."
        return "Phía trước có vật cản, chú ý tránh."

    def _update_obstacle_properties(self, obs, H, W):
        """Cập nhật các thuộc tính dẫn xuất của vật cản"""
        if 'mask' not in obs or obs['mask'] is None:
            return
        
        mask = obs['mask']
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) > 0:
            obs['area'] = len(y_coords)
            obs['center_x'] = float(np.mean(x_coords))
            obs['center_y'] = float(np.mean(y_coords))
            obs['y_position_ratio'] = obs['center_y'] / H
            obs['area_ratio'] = obs['area'] / (H * W)
            obs['bottom_y_ratio'] = np.max(y_coords) / H
            
            # Cập nhật khung bao
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            obs['box_coords'] = (x1, y1, x2, y2)