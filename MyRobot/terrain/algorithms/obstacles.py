"""障碍物地形算法。"""

import numpy as np
from .base import TerrainAlgorithm


class DiscreteObstaclesAlgorithm(TerrainAlgorithm):
    """离散障碍物算法。
    
    复现 terrain_utils.discrete_obstacles_terrain 的逻辑。
    在地面（高度=0）上随机放置矩形障碍物。
    """
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成离散障碍物地形。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数，使用 params.discrete_obstacles_height, params.platform_size
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘归零区域（像素数）
            
        Returns:
            np.ndarray: 高度图数组，shape=(rows, cols)，dtype=int16
        """
        length, width = shape
        height_field_raw = np.zeros(shape, dtype=np.int16)
        
        center_x = length // 2
        center_y = width // 2
        
        platform_size_pixels = int(params.platform_size / horizontal_scale / 2)
        
        # 障碍物参数
        max_height = params.discrete_obstacles_height
        rectangle_min_size = 1.0  # 米
        rectangle_max_size = 2.0  # 米
        num_rectangles = 20
        
        rectangle_min_pixels = int(rectangle_min_size / horizontal_scale)
        rectangle_max_pixels = int(rectangle_max_size / horizontal_scale)
        max_height_int = int(max_height / vertical_scale) if max_height > 0 else 1
        
        # 整个地形初始高度为0
        height_field_raw[:, :] = 0
        
        for _ in range(num_rectangles):
            # 随机矩形尺寸
            rect_length = np.random.randint(rectangle_min_pixels, rectangle_max_pixels + 1)
            rect_width = np.random.randint(rectangle_min_pixels, rectangle_max_pixels + 1)
            
            # 随机位置（避开边缘区域）
            max_x = length - rect_length - env_border
            max_y = width - rect_width - env_border
            
            if max_x <= env_border or max_y <= env_border:
                continue
            
            x = np.random.randint(env_border, max_x)
            y = np.random.randint(env_border, max_y)
            
            # 随机高度
            height = np.random.randint(1, max_height_int + 1)
            
            # 检查是否与中心平台重叠
            x_end = x + rect_length
            y_end = y + rect_width
            
            # 如果与平台重叠超过50%，跳过
            overlap_x = max(0, min(x_end, center_x + platform_size_pixels) - max(x, center_x - platform_size_pixels))
            overlap_y = max(0, min(y_end, center_y + platform_size_pixels) - max(y, center_y - platform_size_pixels))
            overlap_area = overlap_x * overlap_y
            rect_area = rect_length * rect_width
            
            if overlap_area > rect_area * 0.5:
                continue
            
            height_field_raw[x:x_end, y:y_end] = height
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw