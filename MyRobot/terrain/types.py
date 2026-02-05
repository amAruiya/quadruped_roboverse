"""地形数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch


@dataclass
class HeightField:
    """高度场数据结构。
    
    Attributes:
        heights: 高度数据，shape=(rows, cols)，单位：米
        horizontal_scale: 水平分辨率（米/像素）
        vertical_scale: 垂直缩放因子
        origin: 地形原点坐标 (x, y, z)
    """
    heights: np.ndarray
    horizontal_scale: float
    vertical_scale: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    @property
    def shape(self) -> tuple[int, int]:
        """返回高度图尺寸 (rows, cols)."""
        return self.heights.shape
    
    @property
    def size(self) -> tuple[float, float]:
        """返回实际物理尺寸 (length, width) in meters."""
        rows, cols = self.shape
        return (rows * self.horizontal_scale, cols * self.horizontal_scale)
    
    def to_tensor(self, device: str = "cuda") -> torch.Tensor:
        """转换为 PyTorch Tensor."""
        return torch.from_numpy(self.heights).float().to(device)


@dataclass
class TriMesh:
    """三角网格数据结构。
    
    Attributes:
        vertices: 顶点坐标，shape=(N, 3)
        triangles: 三角形索引，shape=(M, 3)
        origin: 网格原点坐标 (x, y, z)
    """
    vertices: np.ndarray
    triangles: np.ndarray
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TerrainParams:
    """内部地形生成参数。
    
    从 TerrainCfg 解析而来，用于实际生成算法。
    
    Attributes:
        terrain_type: 具体地形类型
        difficulty: 难度等级 (0-1)
        slope: 斜坡坡度
        step_height: 台阶高度（米）
        step_width: 台阶宽度（米）
        step_depth: 台阶深度（米）
        discrete_obstacles_height: 离散障碍物高度
        stepping_stones_size: 踏脚石尺寸
        stone_distance: 踏脚石间距
        gap_size: 缺口尺寸
        pit_depth: 坑深度
        platform_size: 平台尺寸
    """
    terrain_type: Literal["flat", "rough", "slope", "slope_rough", 
                          "stairs_up", "stairs_down", "discrete", 
                          "stepping_stones", "gap", "pit"]
    difficulty: float = 0.5
    slope: float = 0.0
    step_height: float = 0.0
    step_width: float = 0.31
    step_depth: float = 0.26
    discrete_obstacles_height: float = 0.0
    stepping_stones_size: float = 1.0
    stone_distance: float = 0.1
    gap_size: float = 0.0
    pit_depth: float = 0.0
    platform_size: float = 3.0