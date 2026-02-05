"""地形配置解析器。"""

import numpy as np
from loguru import logger
from MyRobot.configs.task_cfg import TerrainCfg
from .types import TerrainParams


class TerrainConfigParser:
    """将 TerrainCfg 解析为内部参数。
    
    职责：
    1. 解析网格参数（尺寸、分辨率）
    2. 解析地形类型比例
    3. 根据行列索引返回地形参数
    """
    
    def __init__(self, cfg: TerrainCfg):
        """初始化配置解析器。
        
        Args:
            cfg: 地形配置
        """
        self.cfg = cfg
        self._parse_grid_params()
        self._parse_terrain_proportions()
        
        logger.info(
            f"TerrainConfigParser initialized: {self.num_terrains} terrains "
            f"({self.num_rows}x{self.num_cols})"
        )
    
    def _parse_grid_params(self):
        """解析网格参数。"""
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
        
        # 课程学习参数
        self.curriculum = cfg.curriculum
        self.max_init_level = cfg.max_init_terrain_level
        self.difficulty_scale = cfg.difficulty_scale
        
        # 物理参数
        self.friction = cfg.static_friction
        self.restitution = cfg.restitution
    
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