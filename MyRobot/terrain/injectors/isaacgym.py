"""IsaacGym 地形注入实现。"""

import numpy as np
from loguru import logger
from .base import TerrainInjector
from typing import TYPE_CHECKING
from MyRobot.configs.task_cfg import TerrainCfg

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


class IsaacGymInjector(TerrainInjector):
    """IsaacGym 地形注入器。"""
    
    def inject_heightfield(
        self,
        handler: "BaseSimHandler",
        height_field_raw: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """IsaacGym 高度场注入。"""
        from isaacgym import gymapi
        
        if not hasattr(handler, 'gym') or not hasattr(handler, 'sim'):
            raise RuntimeError("Handler must have 'gym' and 'sim' attributes for IsaacGym")
        
        gym = handler.gym
        sim = handler.sim
        
        tot_rows, tot_cols = height_field_raw.shape
        
        # 配置 heightfield 参数
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = config.horizontal_scale
        hf_params.row_scale = config.horizontal_scale
        hf_params.vertical_scale = config.vertical_scale
        hf_params.nbRows = tot_cols  # IsaacGym 的 nbRows 对应列数
        hf_params.nbColumns = tot_rows  # nbColumns 对应行数
        
        # 设置物理参数
        hf_params.static_friction = config.static_friction
        hf_params.dynamic_friction = config.dynamic_friction
        hf_params.restitution = config.restitution
        
        # 设置 transform（平移边界）
        hf_params.transform.p.x = -config.border_size
        hf_params.transform.p.y = -config.border_size
        hf_params.transform.p.z = 0.0
        
        hf_params.segmentation_id = 0
        
        # 添加到仿真（需要列优先数组）
        gym.add_heightfield(sim, height_field_raw.T, hf_params)
        
        logger.info(
            f"Injected IsaacGym heightfield: {tot_rows}x{tot_cols} samples, "
            f"scale=({config.horizontal_scale}, {config.vertical_scale})"
        )
    
    def inject_trimesh(
        self,
        handler: "BaseSimHandler",
        vertices: np.ndarray,
        triangles: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """IsaacGym 三角网格注入。"""
        from isaacgym import gymapi
        
        if not hasattr(handler, 'gym') or not hasattr(handler, 'sim'):
            raise RuntimeError("Handler must have 'gym' and 'sim' attributes for IsaacGym")
        
        gym = handler.gym
        sim = handler.sim
        
        # 配置 trimesh 参数
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        
        tm_params.static_friction = config.static_friction
        tm_params.dynamic_friction = config.dynamic_friction
        tm_params.restitution = config.restitution
        
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        
        tm_params.segmentation_id = 0
        
        gym.add_triangle_mesh(
            sim,
            vertices.flatten(order='C'),
            triangles.flatten(order='C'),
            tm_params
        )
        
        logger.info(
            f"Injected IsaacGym trimesh: {vertices.shape[0]} vertices, "
            f"{triangles.shape[0]} triangles"
        )