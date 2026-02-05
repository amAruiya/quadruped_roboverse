"""障碍物地形算法。"""

import numpy as np
from isaacgym import terrain_utils
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

        sub_terrain = terrain_utils.SubTerrain(
            "sub_terrain",
            width=width,
            length=length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale
        )
        
        # RMA constants
        rectangle_min_size = 1.0
        rectangle_max_size = 2.0
        num_rectangles = 20
        
        terrain_utils.discrete_obstacles_terrain(
            sub_terrain,
            max_height=params.discrete_obstacles_height,
            min_size=rectangle_min_size,
            max_size=rectangle_max_size,
            num_rects=num_rectangles,
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
