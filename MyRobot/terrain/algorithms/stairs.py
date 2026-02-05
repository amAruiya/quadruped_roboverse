"""台阶地形算法。"""

import numpy as np
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
        height_field_raw = np.zeros(shape, dtype=np.int16)
        
        center_x = length // 2
        center_y = width // 2
        
        platform_size_pixels = int(params.platform_size / horizontal_scale / 2)
        step_depth_pixels = int(params.step_depth / horizontal_scale)
        
        # 根据地形类型确定台阶方向
        step_height = params.step_height
        if params.terrain_type == "stairs_down":
            step_height = -step_height
            
        step_height_int = int(step_height / vertical_scale)
        
        for i in range(length):
            for j in range(width):
                # 边缘区域强制归零
                if i < env_border or i >= length - env_border or \
                   j < env_border or j >= width - env_border:
                    height_field_raw[i, j] = 0
                    continue
                
                # 计算到边缘的距离
                dist_to_edge_x = min(i - env_border, length - env_border - 1 - i)
                dist_to_edge_y = min(j - env_border, width - env_border - 1 - j)
                dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)
                
                # 计算台阶数（从边缘开始）
                if step_depth_pixels > 0:
                    num_steps = dist_to_edge // step_depth_pixels
                else:
                    num_steps = 0
                
                # 限制最大台阶数（避免中心过高/过低）
                max_dist = min(center_x - env_border, center_y - env_border) - platform_size_pixels
                if max_dist > 0 and step_depth_pixels > 0:
                    max_steps = max(1, max_dist // step_depth_pixels)
                    num_steps = min(num_steps, max_steps)
                
                height = num_steps * step_height_int
                height_field_raw[i, j] = height
        
        return height_field_raw