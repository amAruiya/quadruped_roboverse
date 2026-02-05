"""Genesis 地形注入实现。"""

import numpy as np
from loguru import logger
from .base import TerrainInjector
from typing import TYPE_CHECKING
from MyRobot.configs.task_cfg import TerrainCfg

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


class GenesisInjector(TerrainInjector):
    """Genesis 地形注入器。"""
    
    def inject_heightfield(
        self,
        handler: "BaseSimHandler",
        height_field_raw: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """Genesis 高度场注入。"""
        logger.warning("Genesis heightfield injection not implemented yet")
        # TODO: 实现 Genesis 高度场注入
    
    def inject_trimesh(
        self,
        handler: "BaseSimHandler",
        vertices: np.ndarray,
        triangles: np.ndarray,
        config: TerrainCfg,
        **kwargs
    ):
        """Genesis 三角网格注入。"""
        logger.warning("Genesis trimesh injection not implemented yet")
        # TODO: 实现 Genesis 三角网格注入