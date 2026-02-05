"""地形生成器（轻量级协调类）。"""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import numpy as np
from loguru import logger

from MyRobot.configs.task_cfg import TerrainCfg
from .config_parser import TerrainConfigParser
from .algorithms import ALGORITHM_REGISTRY
from .injectors import get_injector
from .types import HeightField

if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


class TerrainGenerator:
    """地形生成器。
    
    职责：
    1. 协调配置解析器
    2. 调用算法生成高度图
    3. 管理全局地形网格
    4. 委托仿真器注入给 injector
    
    使用流程：
        1. 初始化：generator = TerrainGenerator(terrain_cfg, simulator_type)
        2. 生成：height_fields = generator.generate()
        3. 注入：generator.bind_handler(handler); generator()
    """
    
    def __init__(
        self,
        cfg: TerrainCfg,
        simulator: Literal["mujoco", "isaacgym", "isaacsim", "genesis"],
    ):
        """初始化地形生成器。
        
        Args:
            cfg: 地形配置
            simulator: 仿真器类型
        """
        self.cfg = cfg
        self.simulator = simulator
        self.handler: "BaseSimHandler | None" = None
        
        # 初始化组件
        self.parser = TerrainConfigParser(cfg)
        self.injector = get_injector(simulator)
        
        # 生成的地形数据
        self.height_fields: list[HeightField] = []
        self.height_field_raw: np.ndarray | None = None
        
        # 环境原点和类型记录
        self.env_origins = np.zeros((self.parser.num_rows, self.parser.num_cols, 3))
        self.terrain_types = np.empty((self.parser.num_rows, self.parser.num_cols), dtype=object)
        
        logger.info(
            f"TerrainGenerator initialized: {self.parser.num_terrains} terrains "
            f"({self.parser.num_rows}x{self.parser.num_cols}), type={cfg.mesh_type}, "
            f"simulator={simulator}"
        )
    
    def generate(self) -> list[HeightField]:
        """生成所有地形高度场。
        
        Returns:
            生成的高度场列表
        """
        self.height_fields = []
        
        # 创建完整高度图（int16 格式，与 IsaacGym 一致）
        self.height_field_raw = np.zeros(
            (self.parser.tot_rows, self.parser.tot_cols), 
            dtype=np.int16
        )
        
        for row in range(self.parser.num_rows):
            for col in range(self.parser.num_cols):
                params = self.parser.get_terrain_params(row, col)
                height_field = self._generate_single_terrain(params, row, col)
                self.height_fields.append(height_field)
        
        logger.info(f"Generated {len(self.height_fields)} terrain height fields")
        return self.height_fields
    
    def _generate_single_terrain(
        self, params, row: int, col: int
    ) -> HeightField:
        """生成单个地形块。
        
        Args:
            params: 地形参数
            row: 行索引
            col: 列索引
            
        Returns:
            HeightField: 高度场数据
        """
        # 记录地形类型
        terrain_type = params.terrain_type
        self.terrain_types[row, col] = terrain_type
        
        # 根据地形类型选择算法
        if terrain_type not in ALGORITHM_REGISTRY:
            logger.warning(f"Unknown terrain type: {terrain_type}, using flat")
            algorithm = ALGORITHM_REGISTRY["flat"]
        else:
            algorithm = ALGORITHM_REGISTRY[terrain_type]
        
        # 生成高度图
        height_field_raw = algorithm.generate(
            shape=(self.parser.length_per_env_pixels, self.parser.width_per_env_pixels),
            params=params,
            horizontal_scale=self.parser.horizontal_scale,
            vertical_scale=self.parser.vertical_scale,
            env_border=self.parser.env_border
        )
        
        # 添加到全局高度图
        self._add_terrain_to_map(height_field_raw, row, col)
        
        # 计算全局原点
        origin_x = row * self.parser.terrain_length
        origin_y = col * self.parser.terrain_width
        origin = (origin_x, origin_y, 0.0)
        
        # 转换为 float32（米为单位）
        heights_float = height_field_raw.astype(np.float32) * self.parser.vertical_scale
        
        return HeightField(
            heights=heights_float,
            horizontal_scale=self.parser.horizontal_scale,
            vertical_scale=self.parser.vertical_scale,
            origin=origin,
        )
    
    def _add_terrain_to_map(self, terrain_hf: np.ndarray, row: int, col: int):
        """将地形块添加到全局高度图，并计算环境原点。
        
        复现 terrain.py 中 add_terrain_to_map 的逻辑。
        """
        # 计算在全局高度图中的位置
        start_x = self.parser.border + row * self.parser.length_per_env_pixels
        end_x = self.parser.border + (row + 1) * self.parser.length_per_env_pixels
        start_y = self.parser.border + col * self.parser.width_per_env_pixels
        end_y = self.parser.border + (col + 1) * self.parser.width_per_env_pixels
        
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain_hf
        
        # 计算环境原点（机器人放置位置）
        env_origin_x = (row + 0.5) * self.parser.terrain_length
        env_origin_y = (col + 0.5) * self.parser.terrain_width
        
        # 在中心区域找最大高度作为 z 原点
        x1 = int((self.parser.terrain_length / 2.0 - 1) / self.parser.horizontal_scale)
        x2 = int((self.parser.terrain_length / 2.0 + 1) / self.parser.horizontal_scale)
        y1 = int((self.parser.terrain_width / 2.0 - 1) / self.parser.horizontal_scale)
        y2 = int((self.parser.terrain_width / 2.0 + 1) / self.parser.horizontal_scale)
        
        env_origin_z = np.max(terrain_hf[x1:x2, y1:y2]) * self.parser.vertical_scale
        self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
    
    # =========================================================================
    # 注入 API
    # =========================================================================
    
    def bind_handler(self, handler: "BaseSimHandler"):
        """绑定仿真 handler。
        
        Args:
            handler: 仿真 handler 实例
        """
        self.handler = handler
        logger.info(f"TerrainGenerator bound to handler: {type(handler).__name__}")
    
    def __call__(self):
        """执行地形注入（作为 setup callback 调用）。"""
        if self.handler is None:
            raise RuntimeError("Handler not bound. Call bind_handler() first.")
        
        # 生成地形（如果还没生成）
        if not self.height_fields or self.height_field_raw is None:
            self.generate()
        
        # 委托给注入器
        self.injector.inject(
            self.handler,
            self.height_field_raw,
            self.cfg
        )
    
    # =========================================================================
    # 兼容性接口（保留原有方法）
    # =========================================================================
    
    def to_trimesh(self, height_field: HeightField):
        """将高度场转换为三角网格（预留接口）。"""
        # 简单实现，委托给 injector 的内部方法
        vertices, triangles = self.injector._convert_heightfield_to_trimesh(
            (height_field.heights / self.parser.vertical_scale).astype(np.int16),
            self.cfg
        )
        
        from .types import TriMesh
        return TriMesh(
            vertices=vertices,
            triangles=triangles,
            origin=height_field.origin,
        )
    
    def convert_heightfield_to_trimesh(self) -> tuple[np.ndarray, np.ndarray]:
        """将全局高度图转换为三角网格。"""
        if self.height_field_raw is None:
            raise RuntimeError("No heightfield generated yet. Call generate() first.")
        
        return self.injector._convert_heightfield_to_trimesh(
            self.height_field_raw, self.cfg
        )