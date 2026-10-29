# -*- coding: utf-8 -*-
"""
Bộ giám sát nhận biết vạch qua đường
Phát hiện vạch qua đường dựa trên thay đổi diện tích và nhắc bằng giọng nói
Không liên quan đến chuyển trạng thái, chỉ cung cấp hướng dẫn bằng giọng nói
"""
import time
import numpy as np
from collections import deque
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrosswalkAwarenessMonitor:
    """Bộ giám sát nhận biết vạch qua đường - module nhắc bằng giọng nói thuần túy"""
    
    def __init__(self):
        # Ngưỡng diện tích (mốc cố định)
        self.THRESHOLDS = {
            'discover': 0.01,      # 1% - phát hiện
            'approaching': 0.08,   # 8% - đến gần
            'near': 0.18,          # 18% - rất gần
            'arrival': 0.25,       # 25% - đã tới (có thể qua đường)
        }
        
        # Các ngưỡng đã thông báo (tránh lặp lại)
        self.broadcasted_thresholds = set()
        
        # Lịch sử diện tích
        self.area_history = deque(maxlen=30)  # Lưu 30 khung gần nhất
        
        # Ghi nhận thời gian
        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0
        
        # Cờ trạng thái
        self.in_arrival_state = False  # Có đang ở trạng thái "có thể qua đường" hay không
        self.last_position_zone = None  # Phương vị thông báo lần trước
        
        # Cấu hình khoảng cách thông báo (tham số có thể chỉnh - giá trị càng nhỏ thì thông báo càng thường xuyên)
        # [Chỉnh tham số] Chia mọi khoảng cách cho 1.5, tăng tần suất thông báo lên 1.5 lần
        self.REPEAT_INTERVALS = {
            'approaching': 6.7,   # Giai đoạn đến gần: lặp lại mỗi 6.7 giây (gốc 10 giây ÷ 1.5)
            'near': 3.3,          # Giai đoạn rất gần: lặp lại mỗi 3.3 giây (gốc 5 giây ÷ 1.5)
            'arrival': 5.3,       # Giai đoạn đã tới: lặp lại mỗi 5.3 giây (gốc 8 giây ÷ 1.5)
        }
        # Gợi ý: muốn chỉnh tần suất thì sửa các giá trị này
        # - Giá trị càng nhỏ = thông báo càng thường xuyên
        # - Giá trị càng lớn = thông báo càng thưa
        
        # Ngưỡng xác định không bị che khuất
        self.OCCLUSION_THRESHOLD = 0.30  # Chồng lấp >30% thì coi là bị che khuất
    
    def process_frame(self, crosswalk_mask, blind_path_mask=None) -> Optional[Dict[str, Any]]:
        """
        Xử lý phát hiện vạch qua đường của mỗi khung
        
        Trả về:
        {
            'voice_text': văn bản giọng nói,
            'priority': mức ưu tiên,
            'should_broadcast': có nên thông báo không,
            'area': diện tích hiện tại,
            'position': mô tả phương vị,
            'visualization': thông tin trực quan (dùng để vẽ ở bên ngoài)
        }
        hoặc None (không cần thông báo)
        """
        # Nếu không có vạch qua đường thì đặt lại trạng thái
        if crosswalk_mask is None:
            self._reset_if_needed()
            return None
        
        # 1. Tính diện tích
        total_pixels = crosswalk_mask.size
        crosswalk_pixels = np.sum(crosswalk_mask > 0)
        area_ratio = crosswalk_pixels / total_pixels
        
        # 2. Tính vị trí trung tâm
        y_coords, x_coords = np.where(crosswalk_mask > 0)
        if len(y_coords) == 0:
            return None
        
        center_x_ratio = np.mean(x_coords) / crosswalk_mask.shape[1]
        center_y_ratio = np.mean(y_coords) / crosswalk_mask.shape[0]
        
        # 3. Ghi lịch sử
        current_time = time.time()
        self.area_history.append({
            'area': area_ratio,
            'center_x': center_x_ratio,
            'center_y': center_y_ratio,
            'time': current_time
        })
        
        # 4. Kiểm tra che khuất
        has_occlusion = self._check_occlusion(crosswalk_mask, blind_path_mask)
        
        # 5. Xác định giai đoạn hiện tại và tạo giọng nói
        return self._generate_guidance(area_ratio, center_x_ratio, center_y_ratio, 
                                       has_occlusion, current_time)
    
    def _check_occlusion(self, crosswalk_mask, blind_path_mask) -> bool:
        """Kiểm tra vạch qua đường có bị đường dẫn che khuất không"""
        if blind_path_mask is None:
            return False
        
        crosswalk_area = crosswalk_mask > 0
        blind_path_area = blind_path_mask > 0
        
        # Tính độ chồng lấp
        overlap = np.logical_and(crosswalk_area, blind_path_area)
        overlap_ratio = np.sum(overlap) / max(np.sum(crosswalk_area), 1)
        
        # Chồng lấp vượt ngưỡng thì coi là bị che khuất
        return overlap_ratio > self.OCCLUSION_THRESHOLD
    
    def _get_position_description(self, center_x_ratio) -> str:
        """Lấy mô tả phương vị (chia 3 vùng)"""
        if center_x_ratio < 0.40:
            return "Ở bên trái khung hình"
        elif center_x_ratio < 0.60:
            return "Ở giữa khung hình"
        else:
            return "Ở bên phải khung hình"
    
    def _generate_guidance(self, area_ratio, center_x_ratio, center_y_ratio, 
                          has_occlusion, current_time) -> Optional[Dict[str, Any]]:
        """Tạo giọng nói hướng dẫn"""
        
        # Kiểm tra diện tích có ổn định không (tránh rung lắc)
        if not self._is_area_stable(area_ratio):
            return None
        
        position_desc = self._get_position_description(center_x_ratio)
        
        # Giai đoạn 1: phát hiện (0.01-0.08)
        if area_ratio >= self.THRESHOLDS['discover'] and area_ratio < self.THRESHOLDS['approaching']:
            if self.THRESHOLDS['discover'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['discover'])
                return {
                    'voice_text': f"Phát hiện vạch qua đường ở phía xa;{position_desc}",
                    'priority': 55,  # Nâng lên 55, cao hơn lệnh hướng đường dẫn người khiếm thị (50)
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
        
        # Giai đoạn 2: đến gần (0.08-0.18)
        elif area_ratio >= self.THRESHOLDS['approaching'] and area_ratio < self.THRESHOLDS['near']:
            # Thông báo lần đầu
            if self.THRESHOLDS['approaching'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['approaching'])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Đang tiến gần vạch qua đường;{position_desc}",
                    'priority': 55,  # Nâng lên 55
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
            # Thông báo lặp lại (mỗi 10 giây hoặc khi phương vị đổi)
            elif (current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['approaching'] or
                  position_desc != self.last_position_zone):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Đang tiến gần vạch qua đường;{position_desc}",
                    'priority': 55,  # Nâng lên 55
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
        
        # Giai đoạn 3: rất gần (0.18-0.25)
        elif area_ratio >= self.THRESHOLDS['near'] and area_ratio < self.THRESHOLDS['arrival']:
            # Thông báo lần đầu
            if self.THRESHOLDS['near'] not in self.broadcasted_thresholds:
                self.broadcasted_thresholds.add(self.THRESHOLDS['near'])
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Gần tới vạch qua đường;{position_desc}",
                    'priority': 60,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
            # Thông báo lặp lại (mỗi 5 giây hoặc khi phương vị đổi)
            elif (current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['near'] or
                  position_desc != self.last_position_zone):
                self.last_broadcast_time = current_time
                self.last_position_zone = position_desc
                return {
                    'voice_text': f"Gần tới vạch qua đường;{position_desc}",
                    'priority': 60,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': position_desc
                }
        
        # Giai đoạn 4: đã tới (area ≥ 0.25, không bị che khuất)
        elif area_ratio >= self.THRESHOLDS['arrival']:
            # Phải không bị che khuất mới được nhắc qua đường
            if has_occlusion:
                # Bị che khuất, tạm không nhắc qua đường, giữ ở giai đoạn 3
                logger.info(f"[Vạch qua đường] Diện tích đạt {area_ratio:.2f} nhưng bị che khuất, tạm không nhắc qua đường")
                return None
            
            # Lần đầu tới nơi
            if not self.in_arrival_state:
                self.in_arrival_state = True
                self.arrival_first_broadcast_time = current_time
                self.last_broadcast_time = current_time
                logger.info(f"[Vạch qua đường] Trạng thái đã tới: area={area_ratio:.2f}, không bị che khuất")
                return {
                    'voice_text': "Đã tới vạch qua đường, có thể qua đường",
                    'priority': 80,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': 'Tới nơi'
                }
            # Thông báo lặp lại (mỗi 8 giây)
            elif current_time - self.last_broadcast_time >= self.REPEAT_INTERVALS['arrival']:
                self.last_broadcast_time = current_time
                return {
                    'voice_text': "Đã tới vạch qua đường, có thể qua đường",
                    'priority': 80,
                    'should_broadcast': True,
                    'area': area_ratio,
                    'position': 'Tới nơi'
                }
            # Xử lý quá hạn (sau 30 giây tự động thoát trạng thái đã tới)
            elif current_time - self.arrival_first_broadcast_time > 30.0:
                logger.info("[Vạch qua đường] Trạng thái đã tới quá hạn 30 giây, tự động thoát")
                self.in_arrival_state = False
                return None
        
        # Xử lý hạ cấp: nếu diện tích giảm so với trạng thái đã tới
        if self.in_arrival_state and area_ratio < 0.20:
            logger.info(f"[Vạch qua đường] Diện tích giảm xuống {area_ratio:.2f}, thoát trạng thái đã tới")
            self.in_arrival_state = False
            # Xóa một phần cờ đã thông báo, cho phép thông báo lại
            self.broadcasted_thresholds.discard(self.THRESHOLDS['arrival'])
        
        return None
    
    def _is_area_stable(self, area_ratio, stability_frames=5) -> bool:
        """Kiểm tra diện tích có ổn định không (tránh kích hoạt do rung lắc)"""
        if len(self.area_history) < stability_frames:
            return True  # Giai đoạn đầu, coi là ổn định
        
        recent_areas = [h['area'] for h in list(self.area_history)[-stability_frames:]]
        
        # Kiểm tra N khung gần nhất có đều gần diện tích hiện tại không (±20%)
        for recent_area in recent_areas:
            if abs(recent_area - area_ratio) / max(area_ratio, 0.001) > 0.20:
                return False
        
        return True
    
    def _reset_if_needed(self):
        """Đặt lại trạng thái (khi vạch qua đường biến mất)"""
        if len(self.area_history) > 0:
            logger.info("[Vạch qua đường] Vạch qua đường biến mất, đặt lại trạng thái")
        
        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_position_zone = None
    
    def reset(self):
        """Đặt lại hoàn toàn"""
        self.broadcasted_thresholds.clear()
        self.area_history.clear()
        self.in_arrival_state = False
        self.last_broadcast_time = 0
        self.arrival_first_broadcast_time = 0
        self.last_position_zone = None
        logger.info("[Vạch qua đường] Đã đặt lại bộ giám sát nhận biết")
    
    def is_in_arrival_state(self) -> bool:
        """Có đang ở trạng thái đã tới không (để bên ngoài xác định có tạm dừng giọng nói đường dẫn không)"""
        return self.in_arrival_state
    
    def get_current_area(self) -> float:
        """Lấy diện tích hiện tại"""
        if len(self.area_history) > 0:
            return self.area_history[-1]['area']
        return 0.0
    
    def get_visualization_data(self, crosswalk_mask, area_ratio, center_x_ratio, center_y_ratio, has_occlusion) -> Dict[str, Any]:
        """
        Lấy dữ liệu trực quan
        Trả về dict chứa mọi phần tử trực quan
        """
        if crosswalk_mask is None:
            return {}
        
        # Xác định giai đoạn hiện tại (thống nhất dùng màu cam)
        if area_ratio >= self.THRESHOLDS['arrival']:
            stage = "Tới nơi"
            stage_color = "rgba(255, 165, 0, 0.5)"  # Màu cam
        elif area_ratio >= self.THRESHOLDS['near']:
            stage = "Gần"
            stage_color = "rgba(255, 165, 0, 0.45)"  # Màu cam
        elif area_ratio >= self.THRESHOLDS['approaching']:
            stage = "Tiến lại gần"
            stage_color = "rgba(255, 165, 0, 0.40)"  # Màu cam
        else:
            stage = "Phát hiện"
            stage_color = "rgba(255, 165, 0, 0.35)"  # Màu cam
        
        # Mô tả phương vị
        position = self._get_position_description(center_x_ratio)
        
        return {
            'area_ratio': area_ratio,
            'stage': stage,
            'stage_color': stage_color,
            'position': position.replace("Ở ", "").replace(" khung hình", ""),  # Bỏ tiền tố "ở trong khung hình"
            'center_x_ratio': center_x_ratio,
            'center_y_ratio': center_y_ratio,
            'has_occlusion': has_occlusion,
            'in_arrival': self.in_arrival_state
        }


# Hàm phụ trợ
def split_combined_voice(combined_text: str) -> list:
    """
    Tách giọng nói ghép thành nhiều giọng nói độc lập
    Ví dụ: "Phát hiện vạch qua đường ở phía xa;Ở bên trái khung hình" → ["Phát hiện vạch qua đường ở phía xa", "Ở bên trái khung hình"]
    """
    if ';' in combined_text:
        parts = combined_text.split(';')
        return [p.strip() for p in parts if p.strip()]
    return [combined_text]

