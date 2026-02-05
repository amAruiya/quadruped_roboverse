"""斜坡地形算法。"""

import numpy as np
from isaacgym import terrain_utils
from .base import TerrainAlgorithm


class PyramidSlopeAlgorithm(TerrainAlgorithm):
    """金字塔斜坡算法。
    
    复现 terrain_utils.pyramid_sloped_terrain 的逻辑。
    从边缘（高度=0）向中心逐渐升高/降低。
    """
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成金字塔斜坡地形。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数，使用 params.slope, params.platform_size
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘归零区域（像素数）
            
        Returns:
            np.ndarray: 高度图数组，shape=(rows, cols)，dtype=int16
        """
        length, width = shape
        # 使用 isaacgym.terrain_utils 生成
        sub_terrain = terrain_utils.SubTerrain(
            "sub_terrain",
            width=width,
            length=length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale
        )
        
        terrain_utils.pyramid_sloped_terrain(
            sub_terrain,
            slope=params.slope,
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


class RoughSlopeAlgorithm(TerrainAlgorithm):
    """粗糙斜坡算法。
    
    组合金字塔斜坡 + 随机噪声。
    """
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成粗糙斜坡地形（斜坡 + 噪声）。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数，使用 params.slope, params.platform_size
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘归零区域（像素数）
            
        Returns:
            np.ndarray: 高度图数组，shape=(rows, cols)，dtype=int16
        """
        length, width = shape
        
        # 1. 生成基础斜坡
        sub_terrain = terrain_utils.SubTerrain(
            "sub_terrain",
            width=width,
            length=length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale
        )
        
        terrain_utils.pyramid_sloped_terrain(
            sub_terrain,
            slope=params.slope,
            platform_size=params.platform_size
        )
        
        # 2. 添加随机噪声
        # RMA参数: min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2
        terrain_utils.random_uniform_terrain(
            sub_terrain,
            min_height=-0.05,
            max_height=0.05,
            step=0.005,
            downsampled_scale=0.2
        )
        
        height_field_raw = sub_terrain.height_field_raw
        
        # 3. 边缘归零
        if env_border > 0:
            height_field_raw[:env_border, :] = 0
            height_field_raw[-env_border:, :] = 0
            height_field_raw[:, :env_border] = 0
            height_field_raw[:, -env_border:] = 0
            
        return height_field_raw
