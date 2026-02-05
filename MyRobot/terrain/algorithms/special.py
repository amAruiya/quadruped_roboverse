"""特殊地形算法（gap, pit, stepping_stones）。"""

import numpy as np
from .base import TerrainAlgorithm


class GapAlgorithm(TerrainAlgorithm):
    """缺口地形算法。
    
    复现 terrain.py 中 gap_terrain 的逻辑。
    在中心周围创建一个环形缺口，边缘归零。
    """
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成缺口地形。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数，使用 params.gap_size, params.platform_size
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘归零区域（像素数）
            
        Returns:
            np.ndarray: 高度图数组，shape=(rows, cols)，dtype=int16
        """
        length, width = shape
        height_field_raw = np.zeros(shape, dtype=np.int16)
        
        gap_size_pixels = int(params.gap_size / horizontal_scale) if params.gap_size > 0 else 0
        platform_size_pixels = int(params.platform_size / horizontal_scale)
        
        center_x = length // 2
        center_y = width // 2
        
        # 整个地形初始高度为0
        height_field_raw[:, :] = 0
        
        if gap_size_pixels > 0:
            # 计算缺口区域
            x1 = (length - platform_size_pixels) // 2
            x2 = x1 + gap_size_pixels
            y1 = (width - platform_size_pixels) // 2
            y2 = y1 + gap_size_pixels
            
            # 大缺口区域（深坑）
            gap_depth_int = int(-1000 / vertical_scale)
            height_field_raw[center_x - x2:center_x + x2, center_y - y2:center_y + y2] = gap_depth_int
            
            # 中心平台（恢复为0）
            height_field_raw[center_x - x1:center_x + x1, center_y - y1:center_y + y1] = 0
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw


class PitAlgorithm(TerrainAlgorithm):
    """坑地形算法。
    
    复现 terrain.py 中 pit_terrain 的逻辑。
    在地面（高度=0）中心挖一个方形坑，边缘归零。
    """
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成坑地形。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数，使用 params.pit_depth, params.platform_size
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘归零区域（像素数）
            
        Returns:
            np.ndarray: 高度图数组，shape=(rows, cols)，dtype=int16
        """
        length, width = shape
        height_field_raw = np.zeros(shape, dtype=np.int16)
        
        # 整个地形初始高度为0
        height_field_raw[:, :] = 0
        
        depth = params.pit_depth
        if depth > 0:
            depth_int = int(depth / vertical_scale)
            platform_size_pixels = int(params.platform_size / horizontal_scale / 2)
            
            x1 = length // 2 - platform_size_pixels
            x2 = length // 2 + platform_size_pixels
            y1 = width // 2 - platform_size_pixels
            y2 = width // 2 + platform_size_pixels
            
            # 中心挖坑
            height_field_raw[x1:x2, y1:y2] = -depth_int
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw


class SteppingStonesAlgorithm(TerrainAlgorithm):
    """踏脚石地形算法。
    
    复现 terrain_utils.stepping_stones_terrain 的逻辑。
    在深坑中放置踏脚石，边缘归零。
    """
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成踏脚石地形。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数，使用 params.stepping_stones_size, params.stone_distance, params.platform_size
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
        stone_size_pixels = int(params.stepping_stones_size / horizontal_scale)
        stone_distance_pixels = int(params.stone_distance / horizontal_scale)
        
        # 先设置整个地形为深坑
        pit_depth_int = int(-1.0 / vertical_scale)
        height_field_raw[:, :] = pit_depth_int
        
        # 中心平台
        height_field_raw[
            center_x - platform_size_pixels:center_x + platform_size_pixels,
            center_y - platform_size_pixels:center_y + platform_size_pixels
        ] = 0
        
        # 放置踏脚石（网格排列）
        stone_spacing = stone_size_pixels + stone_distance_pixels
        
        if stone_spacing > 0:
            for i in range(env_border, length - env_border, stone_spacing):
                for j in range(env_border, width - env_border, stone_spacing):
                    # 检查是否在平台区域附近
                    stone_center_x = i + stone_size_pixels // 2
                    stone_center_y = j + stone_size_pixels // 2
                    
                    if (abs(stone_center_x - center_x) < platform_size_pixels + stone_spacing and
                        abs(stone_center_y - center_y) < platform_size_pixels + stone_spacing):
                        continue
                    
                    # 石头高度：接近0（方便踩踏）
                    max_height = 0.0  # 米
                    height = int(np.random.uniform(-0.05, max_height) / vertical_scale)
                    
                    x_end = min(i + stone_size_pixels, length - env_border)
                    y_end = min(j + stone_size_pixels, width - env_border)
                    
                    height_field_raw[i:x_end, j:y_end] = height
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw