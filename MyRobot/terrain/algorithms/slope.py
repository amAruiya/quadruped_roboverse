"""斜坡地形算法。"""

import numpy as np
from .base import TerrainAlgorithm
from .basic import RandomUniformNoise


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
        height_field_raw = np.zeros(shape, dtype=np.int16)
        
        center_x = length // 2
        center_y = width // 2
        
        platform_size_pixels = int(params.platform_size / horizontal_scale / 2)
        
        # 从边缘到中心计算高度
        for i in range(length):
            for j in range(width):
                # 边缘区域强制归零
                if i < env_border or i >= length - env_border or \
                   j < env_border or j >= width - env_border:
                    height_field_raw[i, j] = 0
                    continue
                
                # 计算到边缘的距离（归一化 0-1）
                # 从边缘开始，向中心递增
                dist_to_edge_x = min(i - env_border, length - env_border - 1 - i)
                dist_to_edge_y = min(j - env_border, width - env_border - 1 - j)
                
                # 取最小距离（金字塔形状）
                dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)
                
                # 转换为米
                dist_meters = dist_to_edge * horizontal_scale
                
                # 计算高度：从边缘（0）向中心递增
                # 在平台区域限制最大高度
                max_dist = min(center_x - env_border, center_y - env_border) - platform_size_pixels
                if max_dist > 0:
                    dist_meters = min(dist_meters, max_dist * horizontal_scale)
                
                height = dist_meters * params.slope
                height_field_raw[i, j] = int(height / vertical_scale)
        
        return height_field_raw


class RoughSlopeAlgorithm(TerrainAlgorithm):
    """粗糙斜坡算法。
    
    组合金字塔斜坡 + 随机噪声。
    """
    
    def __init__(self):
        self.slope_algo = PyramidSlopeAlgorithm()
    
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
        # 先生成基础斜坡
        height_field_raw = self.slope_algo.generate(
            shape, params, horizontal_scale, vertical_scale, env_border
        )
        
        # 添加随机噪声
        height_field_raw = RandomUniformNoise.add_noise(
            height_field_raw,
            min_height=-0.05,
            max_height=0.05,
            step=0.005,
            downsampled_scale=0.2,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            env_border=env_border
        )
        
        return height_field_raw