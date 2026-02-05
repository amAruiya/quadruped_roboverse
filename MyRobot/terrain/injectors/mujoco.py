"""MuJoCo 地形注入实现。"""

import numpy as np
from loguru import logger
from .base import TerrainInjector
from typing import TYPE_CHECKING
from MyRobot.configs.task_cfg import TerrainCfg

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


class MuJoCoInjector(TerrainInjector):
    """MuJoCo 地形注入器。"""
    
    def inject_heightfield(
        self,
        handler: "BaseSimHandler",
        height_field_raw: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """MuJoCo 高度场注入。"""
        logger.warning("MuJoCo heightfield injection not implemented yet")
        # TODO: 实现 MuJoCo 高度场注入
    
    def inject_trimesh(
        self,
        handler: "BaseSimHandler",
        vertices: np.ndarray,
        triangles: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """MuJoCo 三角网格注入。"""
        logger.warning("MuJoCo trimesh injection not implemented yet")
        # TODO: 实现 MuJoCo 三角网格注入