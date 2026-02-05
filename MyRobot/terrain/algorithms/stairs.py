"""台阶地形算法。"""

import numpy as np
from isaacgym import terrain_utils
from .base import TerrainAlgorithm


class PyramidStairsAlgorithm(TerrainAlgorithm):
    """金字塔台阶算法。
    
    复现 terrain_utils.pyramid_stairs_terrain 的逻辑。
    从边缘（高度=0）向中心逐级升高/降低。
    """
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成金字塔台阶地形。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数，使用 params.step_height, params.step_depth, params.platform_size
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘归零区域（像素数）
            
        Returns:
            np.ndarray: 高度图数组，shape=(rows, cols)，dtype=int16
        """
        length, width = shape
        
        step_height = params.step_height
        if params.terrain_type == "stairs_down":
            step_height = -step_height

        sub_terrain = terrain_utils.SubTerrain(
            "sub_terrain",
            width=width,
            length=length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale
        )
        
        terrain_utils.pyramid_stairs_terrain(
            sub_terrain,
            step_width=params.step_depth, 
            step_height=step_height,
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
