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
        
        复现 isaacgym.terrain_utils.convert_heightfield_to_trimesh 的逻辑。
        修正：
        1. 修复原库 numpy float3232 属性错误
        2. 移除对外部库的强制依赖
        3. 支持 slope_threshold 以修正垂直墙面（对楼梯地形至关重要）

        Args:
            height_field_raw: 高度图数据 (int16)
            config: 地形配置
            
        Returns:
            (vertices, triangles): 顶点和三角形索引
        """
        hf = height_field_raw
        num_rows = hf.shape[0]
        num_cols = hf.shape[1]
        
        horizontal_scale = config.horizontal_scale
        vertical_scale = config.vertical_scale
        slope_threshold = config.slope_threshold

        # 生成网格坐标
        y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
        x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
        yy, xx = np.meshgrid(y, x)

        if slope_threshold is not None:
            # 缩放阈值以适应高度场原始单位
            slope_threshold *= horizontal_scale / vertical_scale
            
            move_x = np.zeros((num_rows, num_cols))
            move_y = np.zeros((num_rows, num_cols))
            move_corners = np.zeros((num_rows, num_cols))
            
            # 矢量化计算顶点移动：当坡度过大时，移动顶点以形成垂直面
            move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
            move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
            
            move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
            move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
            
            move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
            move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
            
            xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
            yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

        # 创建网格顶点
        vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
        vertices[:, 0] = xx.flatten() - config.border_size 
        vertices[:, 1] = yy.flatten() - config.border_size
        vertices[:, 2] = hf.flatten() * vertical_scale
        
        # 创建三角形索引 (IsaacGym 拓扑结构)
        # 使用矢量化生成以提高效率
        ridx, cidx = np.meshgrid(np.arange(num_rows - 1), np.arange(num_cols - 1), indexing='ij')
        
        v00 = ridx * num_cols + cidx
        v01 = ridx * num_cols + cidx + 1
        v10 = (ridx + 1) * num_cols + cidx
        v11 = (ridx + 1) * num_cols + cidx + 1
        
        # 构建两个三角形: (v00, v11, v01) 和 (v00, v10, v11)
        # 注意：这里的连接顺序需与 convert_heightfield_to_trimesh 保持一致
        t1 = np.stack([v00, v11, v01], axis=-1)
        t2 = np.stack([v00, v10, v11], axis=-1)
        
        triangles = np.concatenate([t1.reshape(-1, 3), t2.reshape(-1, 3)], axis=0).astype(np.uint32)
        
        return vertices, triangles
        
        triangles = np.array(triangles, dtype=np.uint32)
        
        return vertices, triangles