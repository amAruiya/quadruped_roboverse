"""基础地形算法（平坦、随机噪声）。"""

import numpy as np
from .base import TerrainAlgorithm


class FlatAlgorithm(TerrainAlgorithm):
    """平坦地形算法。"""
    
    def generate(self, shape, params, horizontal_scale, vertical_scale, env_border=0):
        """生成平坦地形（全零）。"""
        height_field_raw = np.zeros(shape, dtype=np.int16)
        return height_field_raw


class RandomUniformNoise:
    """随机均匀噪声生成器（用于组合其他算法）。"""
    
    @staticmethod
    def add_noise(
        height_field_raw: np.ndarray,
        min_height: float,
        max_height: float,
        step: float,
        downsampled_scale: float,
        horizontal_scale: float,
        vertical_scale: float,
        env_border: int = 0
    ) -> np.ndarray:
        """在现有高度图上添加随机均匀噪声。
        
        复现 terrain_utils.random_uniform_terrain 的逻辑。
        先在低分辨率下生成噪声，再插值到原始分辨率。
        边缘保持原始高度。
        
        Args:
            height_field_raw: 输入高度图数组 (int16)
            min_height: 最小高度扰动（米）
            max_height: 最大高度扰动（米）
            step: 高度离散化步长（米）
            downsampled_scale: 下采样比例
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘保护区域（像素数）
            
        Returns:
            np.ndarray: 添加噪声后的高度图
        """
        length, width = height_field_raw.shape
        
        # 计算下采样尺寸
        downsampled_length = int(length * downsampled_scale)
        downsampled_width = int(width * downsampled_scale)
        
        # 在下采样分辨率下生成离散化的随机高度
        num_steps = int((max_height - min_height) / step) + 1
        height_values = np.linspace(min_height, max_height, num_steps)
        
        random_heights = np.random.choice(
            height_values, 
            size=(downsampled_length, downsampled_width)
        )
        
        # 双线性插值到原始分辨率
        from scipy import interpolate
        
        x_down = np.linspace(0, 1, downsampled_length)
        y_down = np.linspace(0, 1, downsampled_width)
        interp_func = interpolate.RectBivariateSpline(x_down, y_down, random_heights)
        
        x_up = np.linspace(0, 1, length)
        y_up = np.linspace(0, 1, width)
        interpolated_heights = interp_func(x_up, y_up)
        
        # 叠加到原始高度图
        result = height_field_raw.copy()
        result += (interpolated_heights / vertical_scale).astype(np.int16)
        
        # 边缘归零（保持连续性）
        result[:env_border, :] = 0
        result[-env_border:, :] = 0
        result[:, :env_border] = 0
        result[:, -env_border:] = 0
        
        return result