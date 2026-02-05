"""特殊地形算法（gap, pit, stepping_stones）。"""

import numpy as np
from isaacgym import terrain_utils
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

        sub_terrain = terrain_utils.SubTerrain(
            "sub_terrain",
            width=width,
            length=length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale
        )
        
        # RMA: max_height=0.
        terrain_utils.stepping_stones_terrain(
            sub_terrain,
            stone_size=params.stepping_stones_size,
            stone_distance=params.stone_distance,
            max_height=0.,
            platform_size=params.platform_size
        )
        
        height_field_raw = sub_terrain.height_field_raw
        
        # 边缘归零
        if env_border > 0:
            height_field_raw[:env_border, :] = 0
            height_field_raw[-env_border:, :] = 0
            height_field_raw[:, :env_border] = 0
            height_field_raw[:, -env_border:] = 0
            
        return height_field_raw
