"""地形数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import numpy as np
import torch
from loguru import logger
from MyRobot.configs.task_cfg import TerrainCfg
if TYPE_CHECKING:
    from metasim.sim.base import BaseSimHandler


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
    step_depth: float = 0.26  # 新增
    discrete_obstacles_height: float = 0.0
    stepping_stones_size: float = 1.0
    stone_distance: float = 0.1
    gap_size: float = 0.0
    pit_depth: float = 0.0
    platform_size: float = 3.0


class TerrainGenerator:
    """地形生成器。
    
    使用流程：
        1. 初始化：generator = TerrainGenerator(terrain_cfg, simulator_type)
        2. 生成：height_field = generator.generate()
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
        self.handler: BaseSimHandler | None = None
        
        # 解析配置
        self._parse_config()
        
        # 生成的地形数据
        self.height_fields: list[HeightField] = []
        self.trimeshes: list[TriMesh] = []
        
        # 环境原点（用于机器人放置）
        # self.env_origins: np.ndarray | None = None
        
        logger.info(
            f"TerrainGenerator initialized: {self.num_terrains} terrains "
            f"({self.num_rows}x{self.num_cols}), type={cfg.mesh_type}"
        )
    
    # =========================================================================
    # 1. 配置解析部分
    # =========================================================================
    
    def _parse_config(self):
        """解析配置，转换为内部参数。"""
        cfg = self.cfg
        
        # 网格参数
        self.num_rows = cfg.num_rows
        self.num_cols = cfg.num_cols
        self.num_terrains = self.num_rows * self.num_cols
        
        # 单个地形块尺寸
        self.terrain_length = cfg.terrain_length
        self.terrain_width = cfg.terrain_width
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale
        self.border_size = cfg.border_size
        
        # 计算高度图分辨率（像素数）
        self.width_per_env_pixels = int(self.terrain_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.terrain_length / self.horizontal_scale)
        
        # 边界像素数
        self.border = int(self.border_size / self.horizontal_scale)
        
        self.env_border = int(self.cfg.env_border_size / self.horizontal_scale)
        
        # 总高度图尺寸
        self.tot_rows = self.num_rows * self.length_per_env_pixels + 2 * self.border
        self.tot_cols = self.num_cols * self.width_per_env_pixels + 2 * self.border
        
        # 地形类型比例（累积分布）
        self._parse_terrain_proportions()
        
        # 课程学习参数
        self.curriculum = cfg.curriculum
        self.max_init_level = cfg.max_init_terrain_level
        self.difficulty_scale = cfg.difficulty_scale
        
        # 物理参数
        self.friction = cfg.static_friction
        self.restitution = cfg.restitution
        
        # 环境原点
        self.env_origins = np.zeros((self.num_rows, self.num_cols, 3))
        
        # 地形类型记录（用于调试和可视化）
        self.terrain_types = np.empty((self.num_rows, self.num_cols), dtype=object)
    
    def _parse_terrain_proportions(self):
        """解析地形类型比例，生成累积分布。"""
        if self.cfg.terrain_proportions is None:
            # 默认：全部平坦地形
            self.proportions = [1.0]
            self.terrain_types_list = ["flat"]
            return
        
        proportions = self.cfg.terrain_proportions
        terrain_names = list(proportions.keys())
        terrain_probs = np.array(list(proportions.values()))
        
        # 归一化
        probs_sum = terrain_probs.sum()
        if probs_sum <= 0:
            raise ValueError("Sum of terrain proportions must be positive.")
        terrain_probs /= probs_sum
        
        # 计算累积分布（用于 choice 选择）
        self.proportions = [np.sum(terrain_probs[:i + 1]) for i in range(len(terrain_probs))]
        self.terrain_types_list = terrain_names
    
    def get_terrain_params(self, row: int, col: int) -> TerrainParams:
        """获取指定位置的地形生成参数。
        
        按照 terrain.py 的逻辑：
        - difficulty = row / num_rows (行方向难度递增)
        - choice = col / num_cols (列方向类型变化)
        
        Args:
            row: 地形行索引
            col: 地形列索引
            
        Returns:
            TerrainParams: 地形生成参数
        """
        # 计算难度和选择值
        difficulty = row / max(1, self.num_rows)
        choice = col / max(1, self.num_cols) + 0.001  # 避免边界问题
        
        # 根据 choice 和 proportions 确定地形类型
        return self._create_terrain_params(choice, difficulty)
    
    def _create_terrain_params(
        self, choice: float, difficulty: float
    ) -> TerrainParams:
        """根据选择值和难度创建参数。
        
        复现 terrain.py 中 make_terrain 的参数计算逻辑。
        
        Args:
            choice: 类型选择值 (0-1)
            difficulty: 难度等级 (0-1)
            
        Returns:
            TerrainParams: 地形生成参数
        """
        # 计算各类参数（与 terrain.py 一致）
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        
        # 根据 choice 和累积分布确定地形类型
        props = self.proportions
        
        if len(props) >= 1 and choice < props[0]:
            # 第一类：斜坡（一半向上，一半向下）
            if choice < props[0] / 2:
                slope *= -1
            return TerrainParams(
                terrain_type="slope",
                difficulty=difficulty,
                slope=slope,
                platform_size=3.0,
            )
        
        elif len(props) >= 2 and choice < props[1]:
            # 第二类：带噪声的斜坡
            return TerrainParams(
                terrain_type="slope_rough",
                difficulty=difficulty,
                slope=slope,
                platform_size=3.0,
            )
        
        elif len(props) >= 4 and choice < props[3]:
            # 第三、四类：台阶（上/下）
            if len(props) >= 3 and choice < props[2]:
                step_height *= -1
            return TerrainParams(
                terrain_type="stairs_up" if step_height > 0 else "stairs_down",
                difficulty=difficulty,
                step_height=abs(step_height),
                step_width=self.cfg.step_width,
                step_depth=self.cfg.step_depth,
                platform_size=3.0,
            )
        
        elif len(props) >= 5 and choice < props[4]:
            # 第五类：离散障碍物
            return TerrainParams(
                terrain_type="discrete",
                difficulty=difficulty,
                discrete_obstacles_height=discrete_obstacles_height,
                platform_size=3.0,
            )
        
        elif len(props) >= 6 and choice < props[5]:
            # 第六类：踏脚石
            return TerrainParams(
                terrain_type="stepping_stones",
                difficulty=difficulty,
                stepping_stones_size=stepping_stones_size,
                stone_distance=stone_distance,
                platform_size=4.0,
            )
        
        elif len(props) >= 7 and choice < props[6]:
            # 第七类：缺口
            return TerrainParams(
                terrain_type="gap",
                difficulty=difficulty,
                gap_size=gap_size,
                platform_size=3.0,
            )
        
        else:
            # 第八类：坑
            return TerrainParams(
                terrain_type="pit",
                difficulty=difficulty,
                pit_depth=pit_depth,
                platform_size=4.0,
            )
    
    # =========================================================================
    # 2. 地形生成部分
    # =========================================================================
    
    def generate(self) -> list[HeightField]:
        """生成所有地形高度场。
        
        Returns:
            生成的高度场列表
        """
        self.height_fields = []
        
        # 创建完整高度图（int16 格式，与 IsaacGym 一致）
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                params = self.get_terrain_params(row, col)
                height_field = self._generate_single_terrain(params, row, col)
                self.height_fields.append(height_field)
        
        logger.info(f"Generated {len(self.height_fields)} terrain height fields")
        return self.height_fields
    
    def _generate_single_terrain(
        self, params: TerrainParams, row: int, col: int
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
        
        # 创建地形块的高度图（int16 格式）
        height_field_raw = np.zeros(
            (self.length_per_env_pixels, self.width_per_env_pixels), dtype=np.int16
        )
        
        # 根据地形类型调用对应生成函数
        if terrain_type == "flat":
            pass  # 已经是全零
            
        elif terrain_type == "slope":
            height_field_raw = self._generate_pyramid_sloped_terrain(
                height_field_raw, params.slope, params.platform_size, self.env_border
            )
            
        elif terrain_type == "slope_rough":
            height_field_raw = self._generate_pyramid_sloped_terrain(
                height_field_raw, params.slope, params.platform_size, self.env_border
            )
            height_field_raw = self._generate_random_uniform_terrain(
                height_field_raw, min_height=-0.05, max_height=0.05, 
                step=0.005, downsampled_scale=0.2, env_border=self.env_border
            )
            
        elif terrain_type in ["stairs_up", "stairs_down"]:
            step_height = params.step_height
            if terrain_type == "stairs_down":
                step_height = -step_height
            height_field_raw = self._generate_pyramid_stairs_terrain(
                height_field_raw, params.step_depth, step_height, 
                params.platform_size, self.env_border
            )
            
        elif terrain_type == "discrete":
            height_field_raw = self._generate_discrete_obstacles_terrain(
                height_field_raw, params.discrete_obstacles_height,
                rectangle_min_size=1.0, rectangle_max_size=2.0,
                num_rectangles=20, platform_size=params.platform_size,
                env_border=self.env_border
            )
            
        elif terrain_type == "stepping_stones":
            height_field_raw = self._generate_stepping_stones_terrain(
                height_field_raw, params.stepping_stones_size,
                params.stone_distance, max_height=0.0, 
                platform_size=params.platform_size, env_border=self.env_border
            )
            
        elif terrain_type == "gap":
            height_field_raw = self._generate_gap_terrain(
                height_field_raw, params.gap_size, params.platform_size, self.env_border
            )
            
        elif terrain_type == "pit":
            height_field_raw = self._generate_pit_terrain(
                height_field_raw, params.pit_depth, params.platform_size, self.env_border
            )
        
        # 添加到全局高度图
        self._add_terrain_to_map(height_field_raw, row, col)
        
        # 计算全局原点
        origin_x = row * self.terrain_length
        origin_y = col * self.terrain_width
        origin = (origin_x, origin_y, 0.0)
        
        # 转换为 float32（米为单位）
        heights_float = height_field_raw.astype(np.float32) * self.vertical_scale
        
        return HeightField(
            heights=heights_float,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            origin=origin,
        )
    
    def _add_terrain_to_map(self, terrain_hf: np.ndarray, row: int, col: int):
        """将地形块添加到全局高度图，并计算环境原点。
        
        复现 terrain.py 中 add_terrain_to_map 的逻辑。
        """
        # 计算在全局高度图中的位置
        start_x = self.border + row * self.length_per_env_pixels
        end_x = self.border + (row + 1) * self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = self.border + (col + 1) * self.width_per_env_pixels
        
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain_hf
        
        # 计算环境原点（机器人放置位置）
        env_origin_x = (row + 0.5) * self.terrain_length
        env_origin_y = (col + 0.5) * self.terrain_width
        
        # 在中心区域找最大高度作为 z 原点
        x1 = int((self.terrain_length / 2.0 - 1) / self.horizontal_scale)
        x2 = int((self.terrain_length / 2.0 + 1) / self.horizontal_scale)
        y1 = int((self.terrain_width / 2.0 - 1) / self.horizontal_scale)
        y2 = int((self.terrain_width / 2.0 + 1) / self.horizontal_scale)
        
        env_origin_z = np.max(terrain_hf[x1:x2, y1:y2]) * self.vertical_scale
        self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
    
    # -------------------------------------------------------------------------
    # 地形生成算法（复现 terrain_utils 的核心逻辑）
    # -------------------------------------------------------------------------
    
    def _generate_pyramid_sloped_terrain(
        self, 
        height_field_raw: np.ndarray, 
        slope: float, 
        platform_size: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成金字塔斜坡地形。
        
        复现 terrain_utils.pyramid_sloped_terrain 的逻辑。
        **从边缘（高度=0）向中心逐渐升高/降低**。
        
        Args:
            height_field_raw: 高度图数组 (int16)
            slope: 斜率（正值中心高，负值中心低）
            platform_size: 中心平台尺寸（米）
            env_border: 边缘归零区域（像素数）
        """
        length = height_field_raw.shape[0]
        width = height_field_raw.shape[1]
        
        center_x = length // 2
        center_y = width // 2
        
        platform_size_pixels = int(platform_size / self.horizontal_scale / 2)
        
        # 从边缘到中心计算高度
        for i in range(length):
            for j in range(width):
                # 边缘区域强制归零
                if i < env_border or i >= length - env_border or \
                   j < env_border or j >= width - env_border:
                    height_field_raw[i, j] = 0
                    continue
                
                # 计算到边缘的距离（归一化 0-1）
                # 从边缘开始，向中心递增
                dist_to_edge_x = min(i - env_border, length - env_border - 1 - i)
                dist_to_edge_y = min(j - env_border, width - env_border - 1 - j)
                
                # 取最小距离（金字塔形状）
                dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)
                
                # 转换为米
                dist_meters = dist_to_edge * self.horizontal_scale
                
                # 计算高度：从边缘（0）向中心递增
                # 在平台区域限制最大高度
                max_dist = min(center_x - env_border, center_y - env_border) - platform_size_pixels
                if max_dist > 0:
                    dist_meters = min(dist_meters, max_dist * self.horizontal_scale)
                
                height = dist_meters * slope
                height_field_raw[i, j] = int(height / self.vertical_scale)
        
        return height_field_raw
    
    def _generate_random_uniform_terrain(
        self,
        height_field_raw: np.ndarray,
        min_height: float,
        max_height: float,
        step: float,
        downsampled_scale: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成随机均匀噪声地形。
        
        复现 terrain_utils.random_uniform_terrain 的逻辑。
        先在低分辨率下生成噪声，再插值到原始分辨率。
        **边缘保持原始高度（通常为0）**。
        
        Args:
            height_field_raw: 高度图数组 (int16)
            min_height: 最小高度扰动（米）
            max_height: 最大高度扰动（米）
            step: 高度离散化步长（米）
            downsampled_scale: 下采样比例
            env_border: 边缘保护区域（像素数）
        """
        length = height_field_raw.shape[0]
        width = height_field_raw.shape[1]
        
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
        height_field_raw += (interpolated_heights / self.vertical_scale).astype(np.int16)
        
        # 边缘归零（保持连续性）
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw
    
    def _generate_pyramid_stairs_terrain(
        self,
        height_field_raw: np.ndarray,
        step_depth: float,
        step_height: float,
        platform_size: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成金字塔台阶地形。
        
        复现 terrain_utils.pyramid_stairs_terrain 的逻辑。
        **从边缘（高度=0）向中心逐级升高/降低**。
        
        Args:
            height_field_raw: 高度图数组 (int16)
            step_depth: 台阶深度（水平面宽度，米）
            step_height: 台阶高度（米，正值中心高，负值中心低）
            platform_size: 中心平台尺寸（米）
            env_border: 边缘归零区域（像素数）
        """
        length = height_field_raw.shape[0]
        width = height_field_raw.shape[1]
        
        center_x = length // 2
        center_y = width // 2
        
        platform_size_pixels = int(platform_size / self.horizontal_scale / 2)
        step_depth_pixels = int(step_depth / self.horizontal_scale)
        step_height_int = int(step_height / self.vertical_scale)
        
        for i in range(length):
            for j in range(width):
                # 边缘区域强制归零
                if i < env_border or i >= length - env_border or \
                   j < env_border or j >= width - env_border:
                    height_field_raw[i, j] = 0
                    continue
                
                # 计算到边缘的距离
                dist_to_edge_x = min(i - env_border, length - env_border - 1 - i)
                dist_to_edge_y = min(j - env_border, width - env_border - 1 - j)
                dist_to_edge = min(dist_to_edge_x, dist_to_edge_y)
                
                # 计算台阶数（从边缘开始）
                num_steps = dist_to_edge // step_depth_pixels
                
                # 限制最大台阶数（避免中心过高/过低）
                max_dist = min(center_x - env_border, center_y - env_border) - platform_size_pixels
                max_steps = max(1, max_dist // step_depth_pixels)
                num_steps = min(num_steps, max_steps)
                
                height = num_steps * step_height_int
                height_field_raw[i, j] = height
        
        return height_field_raw
    
    def _generate_discrete_obstacles_terrain(
        self,
        height_field_raw: np.ndarray,
        max_height: float,
        rectangle_min_size: float,
        rectangle_max_size: float,
        num_rectangles: int,
        platform_size: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成离散障碍物地形。
        
        复现 terrain_utils.discrete_obstacles_terrain 的逻辑。
        在**地面（高度=0）**上随机放置矩形障碍物。
        
        Args:
            height_field_raw: 高度图数组 (int16)
            max_height: 障碍物最大高度（米）
            rectangle_min_size: 矩形最小尺寸（米）
            rectangle_max_size: 矩形最大尺寸（米）
            num_rectangles: 矩形数量
            platform_size: 中心平台尺寸（米）
            env_border: 边缘归零区域（像素数）
        """
        length = height_field_raw.shape[0]
        width = height_field_raw.shape[1]
        
        center_x = length // 2
        center_y = width // 2
        
        platform_size_pixels = int(platform_size / self.horizontal_scale / 2)
        
        rectangle_min_pixels = int(rectangle_min_size / self.horizontal_scale)
        rectangle_max_pixels = int(rectangle_max_size / self.horizontal_scale)
        max_height_int = int(max_height / self.vertical_scale)
        
        # 整个地形初始高度为0
        height_field_raw[:, :] = 0
        
        for _ in range(num_rectangles):
            # 随机矩形尺寸
            rect_length = np.random.randint(rectangle_min_pixels, rectangle_max_pixels + 1)
            rect_width = np.random.randint(rectangle_min_pixels, rectangle_max_pixels + 1)
            
            # 随机位置（避开边缘区域）
            max_x = length - rect_length - env_border
            max_y = width - rect_width - env_border
            
            if max_x <= env_border or max_y <= env_border:
                continue
            
            x = np.random.randint(env_border, max_x)
            y = np.random.randint(env_border, max_y)
            
            # 随机高度
            height = np.random.randint(1, max_height_int + 1)
            
            # 检查是否与中心平台重叠
            x_end = x + rect_length
            y_end = y + rect_width
            
            # 如果与平台重叠超过50%，跳过
            overlap_x = max(0, min(x_end, center_x + platform_size_pixels) - max(x, center_x - platform_size_pixels))
            overlap_y = max(0, min(y_end, center_y + platform_size_pixels) - max(y, center_y - platform_size_pixels))
            overlap_area = overlap_x * overlap_y
            rect_area = rect_length * rect_width
            
            if overlap_area > rect_area * 0.5:
                continue
            
            height_field_raw[x:x_end, y:y_end] = height
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw
    
    def _generate_stepping_stones_terrain(
        self,
        height_field_raw: np.ndarray,
        stone_size: float,
        stone_distance: float,
        max_height: float,
        platform_size: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成踏脚石地形。
        
        复现 terrain_utils.stepping_stones_terrain 的逻辑。
        在**深坑**中放置踏脚石，边缘归零。
        
        Args:
            height_field_raw: 高度图数组 (int16)
            stone_size: 石头尺寸（米）
            stone_distance: 石头间距（米）
            max_height: 最大高度变化（米）
            platform_size: 中心平台尺寸（米）
            env_border: 边缘归零区域（像素数）
        """
        length = height_field_raw.shape[0]
        width = height_field_raw.shape[1]
        
        center_x = length // 2
        center_y = width // 2
        
        platform_size_pixels = int(platform_size / self.horizontal_scale / 2)
        stone_size_pixels = int(stone_size / self.horizontal_scale)
        stone_distance_pixels = int(stone_distance / self.horizontal_scale)
        
        # 先设置整个地形为深坑
        pit_depth_int = int(-1.0 / self.vertical_scale)
        height_field_raw[:, :] = pit_depth_int
        
        # 中心平台
        height_field_raw[
            center_x - platform_size_pixels:center_x + platform_size_pixels,
            center_y - platform_size_pixels:center_y + platform_size_pixels
        ] = 0
        
        # 放置踏脚石（网格排列）
        stone_spacing = stone_size_pixels + stone_distance_pixels
        
        for i in range(env_border, length - env_border, stone_spacing):
            for j in range(env_border, width - env_border, stone_spacing):
                # 检查是否在平台区域附近
                stone_center_x = i + stone_size_pixels // 2
                stone_center_y = j + stone_size_pixels // 2
                
                if (abs(stone_center_x - center_x) < platform_size_pixels + stone_spacing and
                    abs(stone_center_y - center_y) < platform_size_pixels + stone_spacing):
                    continue
                
                # 石头高度：接近0（方便踩踏）
                height = int(np.random.uniform(-0.05, max_height) / self.vertical_scale)
                
                x_end = min(i + stone_size_pixels, length - env_border)
                y_end = min(j + stone_size_pixels, width - env_border)
                
                height_field_raw[i:x_end, j:y_end] = height
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw
    
    def _generate_gap_terrain(
        self,
        height_field_raw: np.ndarray,
        gap_size: float,
        platform_size: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成缺口地形。
        
        复现 terrain.py 中 gap_terrain 的逻辑。
        在中心周围创建一个环形缺口，**边缘归零**。
        
        Args:
            height_field_raw: 高度图数组 (int16)
            gap_size: 缺口尺寸（米）
            platform_size: 中心平台尺寸（米）
            env_border: 边缘归零区域（像素数）
        """
        length = height_field_raw.shape[0]
        width = height_field_raw.shape[1]
        
        gap_size_pixels = int(gap_size / self.horizontal_scale)
        platform_size_pixels = int(platform_size / self.horizontal_scale)
        
        center_x = length // 2
        center_y = width // 2
        
        # 整个地形初始高度为0
        height_field_raw[:, :] = 0
        
        # 计算缺口区域
        x1 = (length - platform_size_pixels) // 2
        x2 = x1 + gap_size_pixels
        y1 = (width - platform_size_pixels) // 2
        y2 = y1 + gap_size_pixels
        
        # 大缺口区域（深坑）
        height_field_raw[center_x - x2:center_x + x2, center_y - y2:center_y + y2] = int(-1000 / self.vertical_scale)
        
        # 中心平台（恢复为0）
        height_field_raw[center_x - x1:center_x + x1, center_y - y1:center_y + y1] = 0
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw
    
    def _generate_pit_terrain(
        self,
        height_field_raw: np.ndarray,
        depth: float,
        platform_size: float,
        env_border: int = 0
    ) -> np.ndarray:
        """生成坑地形。
        
        复现 terrain.py 中 pit_terrain 的逻辑。
        在**地面（高度=0）**中心挖一个方形坑，边缘归零。
        
        Args:
            height_field_raw: 高度图数组 (int16)
            depth: 坑深度（米）
            platform_size: 坑的尺寸（米）
            env_border: 边缘归零区域（像素数）
        """
        length = height_field_raw.shape[0]
        width = height_field_raw.shape[1]
        
        # 整个地形初始高度为0
        height_field_raw[:, :] = 0
        
        depth_int = int(depth / self.vertical_scale)
        platform_size_pixels = int(platform_size / self.horizontal_scale / 2)
        
        x1 = length // 2 - platform_size_pixels
        x2 = length // 2 + platform_size_pixels
        y1 = width // 2 - platform_size_pixels
        y2 = width // 2 + platform_size_pixels
        
        # 中心挖坑
        height_field_raw[x1:x2, y1:y2] = -depth_int
        
        # 边缘归零
        height_field_raw[:env_border, :] = 0
        height_field_raw[-env_border:, :] = 0
        height_field_raw[:, :env_border] = 0
        height_field_raw[:, -env_border:] = 0
        
        return height_field_raw
    
    # =========================================================================
    # 3. 格式转换
    # =========================================================================
    
    def to_trimesh(self, height_field: HeightField) -> TriMesh:
        """将高度场转换为三角网格。
        
        Args:
            height_field: 高度场数据
            
        Returns:
            TriMesh: 三角网格数据
        """
        heights = height_field.heights
        rows, cols = heights.shape
        hs = height_field.horizontal_scale
        
        # 生成顶点
        vertices = []
        for i in range(rows):
            for j in range(cols):
                x = i * hs + height_field.origin[0]
                y = j * hs + height_field.origin[1]
                z = heights[i, j] + height_field.origin[2]
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # 生成三角形索引
        triangles = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                idx = i * cols + j
                # 每个方格两个三角形
                triangles.append([idx, idx + cols, idx + 1])
                triangles.append([idx + 1, idx + cols, idx + cols + 1])
        
        triangles = np.array(triangles, dtype=np.int32)
        
        return TriMesh(
            vertices=vertices,
            triangles=triangles,
            origin=height_field.origin,
        )
    
    def convert_heightfield_to_trimesh(self) -> tuple[np.ndarray, np.ndarray]:
        """将全局高度图转换为三角网格。
        
        复现 terrain_utils.convert_heightfield_to_trimesh 的逻辑。
        
        Returns:
            (vertices, triangles): 顶点和三角形索引
        """
        hf = self.height_field_raw
        
        # 按 slope_threshold 进行移动立方体处理（简化版：直接转换）
        rows, cols = hf.shape
        
        # 生成顶点
        y_indices, x_indices = np.meshgrid(
            np.arange(cols), np.arange(rows), indexing='xy'
        )
        
        x = x_indices.flatten() * self.horizontal_scale - self.border_size
        y = y_indices.flatten() * self.horizontal_scale - self.border_size
        z = hf.flatten() * self.vertical_scale
        
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
    
    def to_usd(self, height_field: HeightField, filepath: str):
        """将高度场保存为 USD 文件（预留接口）。
        
        Args:
            height_field: 高度场数据
            filepath: 输出文件路径
        """
        # TODO: 实现 USD 导出
        raise NotImplementedError(
            "USD export requires IsaacSim environment. "
            "Use inject() with bound handler instead."
        )
    
    # =========================================================================
    # 4. 注入 API
    # =========================================================================
    
    def bind_handler(self, handler: BaseSimHandler):
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
        
        # 生成地形
        if not self.height_fields:
            self.generate()
        
        # 根据配置类型注入
        if self.cfg.mesh_type == "heightfield":
            self._inject_heightfield()
        elif self.cfg.mesh_type == "trimesh":
            self._inject_trimesh()
        else:
            logger.warning(f"Unsupported mesh_type: {self.cfg.mesh_type}")
    
    def _inject_heightfield(self):
        """注入高度场地形（仿真器相关实现）。"""
        # 根据仿真器类型调用不同 API
        if self.simulator == "isaacgym":
            self._inject_isaacgym_heightfield()
        elif self.simulator in ["isaacsim", "isaaclab"]:
            self._inject_isaac_heightfield()
        elif self.simulator == "mujoco":
            self._inject_mujoco_heightfield()
        elif self.simulator == "genesis":
            self._inject_genesis_heightfield()
        else:
            raise NotImplementedError(f"Unsupported simulator: {self.simulator}")
    
    def _inject_trimesh(self):
        """注入三角网格地形（仿真器相关实现）。"""
        # 转换为统一的 trimesh
        self.vertices, self.triangles = self.convert_heightfield_to_trimesh()
        
        if self.simulator == "isaacgym":
            self._inject_isaacgym_trimesh()
        elif self.simulator in ["isaacsim", "isaaclab"]:
            self._inject_isaac_trimesh()
        elif self.simulator == "mujoco":
            self._inject_mujoco_trimesh()
        elif self.simulator == "genesis":
            self._inject_genesis_trimesh()
        else:
            raise NotImplementedError(f"Unsupported simulator: {self.simulator}")
    
    # -------------------------------------------------------------------------
    # 仿真器特定实现
    # -------------------------------------------------------------------------
    
    def _inject_isaacgym_heightfield(self):
        """IsaacGym 高度场注入。"""
        from isaacgym import gymapi
        
        if not hasattr(self.handler, 'gym') or not hasattr(self.handler, 'sim'):
            raise RuntimeError("Handler must have 'gym' and 'sim' attributes for IsaacGym")
        
        gym = self.handler.gym
        sim = self.handler.sim
        
        # 配置 heightfield 参数
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.horizontal_scale
        hf_params.row_scale = self.horizontal_scale
        hf_params.vertical_scale = self.vertical_scale
        hf_params.nbRows = self.tot_cols  # IsaacGym 的 nbRows 对应列数
        hf_params.nbColumns = self.tot_rows  # nbColumns 对应行数
        
        # 设置物理参数
        hf_params.static_friction = self.friction
        hf_params.dynamic_friction = self.cfg.dynamic_friction
        hf_params.restitution = self.restitution
        
        # 设置 transform（平移边界）
        hf_params.transform.p.x = -self.border_size
        hf_params.transform.p.y = -self.border_size
        hf_params.transform.p.z = 0.0
        
        hf_params.segmentation_id = 0
        
        # 添加到仿真（需要列优先数组）
        gym.add_heightfield(sim, self.height_field_raw.T, hf_params)
        
        logger.info(
            f"Injected IsaacGym heightfield: {self.tot_rows}x{self.tot_cols} samples, "
            f"scale=({self.horizontal_scale}, {self.vertical_scale})"
        )
    
    def _inject_isaacgym_trimesh(self):
        """IsaacGym 三角网格注入。"""
        from isaacgym import gymapi
        
        if not hasattr(self.handler, 'gym') or not hasattr(self.handler, 'sim'):
            raise RuntimeError("Handler must have 'gym' and 'sim' attributes for IsaacGym")
        
        gym = self.handler.gym
        sim = self.handler.sim
        
        # 配置 trimesh 参数
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.vertices.shape[0]
        tm_params.nb_triangles = self.triangles.shape[0]
        
        tm_params.static_friction = self.friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.restitution
        
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        
        tm_params.segmentation_id = 0
        
        gym.add_triangle_mesh(
            sim,
            self.vertices.flatten(order='C'),
            self.triangles.flatten(order='C'),
            tm_params
        )
        
        logger.info(
            f"Injected IsaacGym trimesh: {self.vertices.shape[0]} vertices, "
            f"{self.triangles.shape[0]} triangles"
        )

    def _inject_isaac_heightfield(self):
        """IsaacSim/Lab 高度场注入。"""
        logger.warning("IsaacSim heightfield injection not implemented yet")
    
    def _inject_mujoco_heightfield(self):
        """MuJoCo 高度场注入。"""
        logger.warning("MuJoCo heightfield注入。")
    
    def _inject_genesis_heightfield(self):
        """Genesis 高度场注入。"""
        logger.warning("Genesis heightfield injection not implemented yet")
    
    def _inject_isaac_trimesh(self):
        """IsaacSim/Lab 三角网格注入。"""
        logger.warning("IsaacSim trimesh injection not implemented yet")
    
    def _inject_mujoco_trimesh(self):
        """MuJoCo 三角网格注入。"""
        logger.warning("MuJoCo trimesh injection not implemented yet")
    
    def _inject_genesis_trimesh(self):
        """Genesis 三角网格注入。"""
        logger.warning("Genesis trimesh injection not implemented yet")