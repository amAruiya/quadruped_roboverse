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
        
        复现 terrain_utils.convert_heightfield_to_trimesh 的逻辑。
        
        Args:
            height_field_raw: 高度图数据 (int16)
            config: 地形配置
            
        Returns:
            (vertices, triangles): 顶点和三角形索引
        """
        hf = height_field_raw
        rows, cols = hf.shape
        
        # 生成顶点
        y_indices, x_indices = np.meshgrid(
            np.arange(cols), np.arange(rows), indexing='xy'
        )
        
        x = x_indices.flatten() * config.horizontal_scale - config.border_size
        y = y_indices.flatten() * config.horizontal_scale - config.border_size
        z = hf.flatten() * config.vertical_scale
        
        vertices = np.column_stack([x, y, z]).astype(np.float32)
        
        # 生成三角形
        triangles = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                idx = i * cols + j
                triangles.append([idx, idx + cols, idx + 1])
                triangles.append([idx + 1, idx + cols, idx + cols + 1])
        
        triangles = np.array(triangles, dtype=np.uint32)
        
        return vertices, triangles