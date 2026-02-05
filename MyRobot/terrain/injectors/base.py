"""仿真器注入抽象接口。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from MyRobot.configs.task_cfg import TerrainCfg

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


class TerrainInjector(ABC):
    """地形注入适配器接口。
    
    每个仿真器需要实现自己的注入逻辑。
    """
    
    def inject(
        self,
        handler: "BaseSimHandler",
        height_field_raw: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """注入地形到仿真器。
        
        Args:
            handler: 仿真器 handler
            height_field_raw: 高度图数据 (int16)
            config: 地形配置
            **kwargs: 其他参数
        """
        if config.mesh_type == "heightfield":
            self.inject_heightfield(handler, height_field_raw, config, **kwargs)
        elif config.mesh_type == "trimesh":
            vertices, triangles = self._convert_heightfield_to_trimesh(
                height_field_raw, config
            )
            self.inject_trimesh(handler, vertices, triangles, config, **kwargs)
        else:
            raise ValueError(f"Unsupported mesh_type: {config.mesh_type}")
    
    @abstractmethod
    def inject_heightfield(
        self,
        handler: "BaseSimHandler",
        height_field_raw: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """注入高度场。"""
        pass
    
    @abstractmethod
    def inject_trimesh(
        self,
        handler: "BaseSimHandler",
        vertices: np.ndarray,
        triangles: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """注入三角网格。"""
        pass
    
    def _convert_heightfield_to_trimesh(
        self, height_field_raw: np.ndarray, config: TerrainCfg
    ) -> tuple[np.ndarray, np.ndarray]:
        """将高度图转换为三角网格。
        
        优先尝试使用 isaacgym.terrain_utils (支持 slope_threshold 修正垂直墙面).
        如果不可用，使用内部矢量化实现。
        
        Args:
            height_field_raw: 高度图数据 (int16)
            config: 地形配置
            
        Returns:
            (vertices, triangles): 顶点和三角形索引
        """
        # 尝试使用 IsaacGym 的工具函数 (支持垂直面修正)
        try:
            from isaacgym import terrain_utils
            vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
                height_field_raw,
                config.horizontal_scale,
                config.vertical_scale,
                config.slope_threshold
            )
            return vertices, triangles
        except (ImportError, AttributeError):
            pass

        # Fallback: 内部矢量化实现 (无法做垂直面修正，仅标准网格)
        hf = height_field_raw
        rows, cols = hf.shape
        
        # 1. 生成顶点 (Grid Points)
        y_indices, x_indices = np.meshgrid(
            np.arange(cols), np.arange(rows), indexing='xy'
        )
        
        x_grid = x_indices.flatten() * config.horizontal_scale - config.border_size
        y_grid = y_indices.flatten() * config.horizontal_scale - config.border_size
        z_grid = hf.flatten() * config.vertical_scale
        
        vertices = np.column_stack([x_grid, y_grid, z_grid]).astype(np.float32)
        
        # 2. 生成三角形 (Vectorized)
        ridx, cidx = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
        
        # 索引计算: idx = i * cols + j
        v00 = ridx * cols + cidx
        v01 = ridx * cols + cidx + 1
        v10 = (ridx + 1) * cols + cidx
        v11 = (ridx + 1) * cols + cidx + 1
        
        # Triangle 1: Top-Left (00), Bottom-Left (10), Top-Right (01)
        # 这种连线方式是 IsaacGym/PhysX 默认的
        t1 = np.stack([v00, v10, v01], axis=-1)
        
        # Triangle 2: Bottom-Left (10), Bottom-Right (11), Top-Right (01)
        t2 = np.stack([v10, v11, v01], axis=-1)
        
        # 合并并展平
        triangles = np.concatenate([t1.reshape(-1, 3), t2.reshape(-1, 3)], axis=0).astype(np.uint32)
        
        return vertices, triangles
        
        triangles = np.array(triangles, dtype=np.uint32)
        
        return vertices, triangles