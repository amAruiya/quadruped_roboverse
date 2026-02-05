"""地形算法基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import TerrainParams


class TerrainAlgorithm(ABC):
    """地形算法抽象接口。
    
    每个具体算法负责生成一种地形类型的高度图。
    算法与仿真器无关，只处理纯数学计算。
    """
    
    @abstractmethod
    def generate(
        self, 
        shape: tuple[int, int],  # (rows, cols) in pixels
        params: TerrainParams,
        horizontal_scale: float,
        vertical_scale: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成高度图。
        
        Args:
            shape: 高度图尺寸 (rows, cols) 像素数
            params: 地形参数
            horizontal_scale: 水平分辨率（米/像素）
            vertical_scale: 垂直缩放因子（米/整数值）
            env_border: 边缘归零区域（像素数）
            
        Returns:
            np.ndarray: 高度图数组，shape=(rows, cols)，dtype=int16
        """
        pass