"""基础运动任务实现。
该模块实现了 BaseLocomotionTask 类，提供四足机器人运动任务的通用框架。
"""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from loguru import logger as log

from metasim.queries import ContactForces
from metasim.queries.base import BaseQueryType
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.types import TensorState
from metasim.utils.state import RobotState, list_state_to_tensor

from MyRobot.configs.task_cfg import BaseTaskCfg
from MyRobot.utils.helper import task_cfg_to_scenario


class BaseLocomotionTask(BaseTaskEnv):
    """四足机器人运动任务基类。

    该类封装了 metasim handler,提供与 example_RMA 兼容的接口,
    同时保持后端无关性。

    生命周期：
        __init__(task_cfg)
            ├─ [Phase 1] 基础配置 + 配置解析
            ├─ [Phase 2] BaseTaskEnv.__init__() → handler.launch()
            ├─ [Phase 3] 获取 handler 信息
            ├─ [Phase 4] 完整缓冲区初始化
            ├─ [Phase 5] 奖励函数 + 回调
            └─ [Phase 6] 手动 reset()

        reset(env_ids)
            ├─ _reset_idx(env_ids)
            ├─ _run_reset_callbacks(env_ids)
            └─ _observation() → obs

        step(action)
            ├─ _pre_physics_step(action)
            ├─ for _ in range(decimation):
            │      ├─ _in_physics_step()
            │      └─ handler.simulate()
            └─ _post_physics_step()
                   ├─ _check_termination()
                   ├─ _compute_reward()
                   └─ _run_post_callbacks()

    Attributes:
        cfg: 任务配置
        handler: metasim 仿真处理器
        num_envs: 并行环境数量
        num_dof: 关节自由度数量
        device: 计算设备
    """

    max_episode_steps = 1000

    def __init__(
        self,
        cfg: BaseTaskCfg,
        device: str | torch.device | None = None,
    ) -> None:
        """初始化任务。

        Args:
            cfg: 任务配置
            device: 计算设备
        """
        # =====================================================================
        # Phase 1: 基础配置 + 配置解析
        # =====================================================================
        self.cfg = cfg
        self.robot_name = cfg.robots
        self.num_envs = cfg.env.num_envs
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 解析配置
        self._parse_cfg()

        # 预初始化地形相关属性（BaseTaskEnv.__init__ 会调用 _get_initial_states，
        # 此时 _init_buffers 尚未执行，需要这些属性存在以走平面模式分支）
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)

        # =====================================================================
        # Phase 2: 构建 scenario 并初始化 handler
        # =====================================================================
        scenario = task_cfg_to_scenario(cfg)
        
        # 直接调用 BaseTaskEnv.__init__，跳过 RLTaskEnv
        BaseTaskEnv.__init__(self, scenario, device)

        # =====================================================================
        # Phase 3: 获取 handler 信息
        # =====================================================================
        self.robot = scenario.robots[0]  # RLTaskEnv 需要
        self.dof_names = self.handler.get_joint_names(self.robot_name, sort=True)
        self.num_dof = len(self.dof_names)
        
        self.body_names = self.handler.get_body_names(self.robot_name, sort=True)
        self.num_bodies = len(self.body_names)

        # =====================================================================
        # Phase 4: 完整缓冲区初始化
        # =====================================================================
        self._init_buffers()

        # =====================================================================
        # Phase 5: 奖励函数 + 回调
        # =====================================================================
        self._prepare_reward_functions()
        self._run_setup_callbacks()

        # =====================================================================
        # Phase 5.5: 地形初始化（trimesh/heightfield 时自动注入）
        #            必须在 _get_initial_states 之前执行，
        #            以便 env_origins 可用于计算机器人出生位置。
        # =====================================================================
        self._setup_terrain()

        # =====================================================================
        # Phase 5.6: 生成初始状态（基于 env_origins 计算正确位置）
        # =====================================================================
        self._initial_states = list_state_to_tensor(
            self.handler, 
            self._get_initial_states(), 
            self.device
        )

        # =====================================================================
        # Phase 6: 手动调用 reset
        # =====================================================================
        log.info("执行首次 reset...")
        self.reset(env_ids=list(range(self.num_envs)))
        self.init_done = True

        # 计算观测/动作空间维度（从 RLTaskEnv 复制）
        states = self.handler.get_states()
        first_obs = self._observation(states)
        self.num_obs = first_obs.shape[-1]
        
        limits = self.robot.joint_limits
        self.joint_names = self.handler.get_joint_names(self.robot.name)
        self._action_low = torch.tensor(
            [limits[j][0] for j in self.joint_names], 
            dtype=torch.float32, 
            device=self.device
        )
        self._action_high = torch.tensor(
            [limits[j][1] for j in self.joint_names], 
            dtype=torch.float32, 
            device=self.device
        )
        self.num_actions = self._action_low.shape[0]

        log.info(
            f"BaseLocomotionTask 初始化完成: {self.num_envs} 环境, "
            f"{self.num_dof} DOF, 观测维度: {self.num_obs}, 动作维度: {self.num_actions}"
        )

    # =========================================================================
    # Phase 1: 配置解析
    # =========================================================================

    def _parse_cfg(self) -> None:
        """解析任务配置。"""
        cfg = self.cfg

        # 时间步长
        self.dt = cfg.sim.dt * cfg.sim.decimation
        self.max_episode_length = int(cfg.env.episode_length_s / self.dt)

        # 观测缩放
        self.obs_scales = cfg.observation.normalization.obs_scales

        # 命令缩放
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
        )

        # 命令范围
        self.command_ranges = cfg.commands.ranges

        # 重力向量（世界坐标系）
        self.gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        
        # 处理地形与地面平面的冲突
        # 如果使用程序化地形 (heightfield/trimesh)，必须禁用默认的无限地面平面，
        # 否则负高度地形 (如 stairs_down) 会被地面平面遮挡
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            if cfg.scene is None:
                cfg.create_ground = False
                log.info(f"Disabled ground plane (create_ground=False) for {cfg.terrain.mesh_type} terrain")
            elif hasattr(cfg.scene, "ground_type") and cfg.scene.ground_type == "plane":
                cfg.scene.ground_type = "none" 
                log.info(f"Disabled default ground plane for {cfg.terrain.mesh_type} terrain")

        log.debug(f"配置解析完成: 时间步长 dt={self.dt:.4f}s, 最大回合长度 ={self.max_episode_length}")

    # =========================================================================
    # Phase 4: 缓冲区初始化
    # =========================================================================

    def _init_buffers(self) -> None:
        """初始化所有缓冲区。
        
        此时 handler 已就绪，num_dof 已知。
        """
        num_envs = self.num_envs
        num_dof = self.num_dof
        device = self.device

        # -----------------------------------------------------------------
        # 基座状态
        # -----------------------------------------------------------------
        self.base_pos = torch.zeros(num_envs, 3, device=device)
        self.base_quat = torch.zeros(num_envs, 4, device=device)
        self.base_quat[:, 3] = 1.0  # w=1 (xyzw 格式)
        self.base_lin_vel = torch.zeros(num_envs, 3, device=device)
        self.base_ang_vel = torch.zeros(num_envs, 3, device=device)

        # 投影重力
        self.projected_gravity = torch.zeros(num_envs, 3, device=device)
        self.projected_gravity[:, 2] = -1.0  # 初始直立

        # 根状态（13 维：pos(3) + rot(4) + lin_vel(3) + ang_vel(3)）
        self.root_states = torch.zeros(num_envs, 13, device=device)
        self.root_states[:, 3] = 1.0  # w=1

        # -----------------------------------------------------------------
        # 关节状态
        # -----------------------------------------------------------------
        self.dof_pos = torch.zeros(num_envs, num_dof, device=device)
        self.dof_vel = torch.zeros(num_envs, num_dof, device=device)
        self.default_dof_pos = self._compute_default_dof_pos()
        
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        # -----------------------------------------------------------------
        # 动作
        # -----------------------------------------------------------------
        self.actions = torch.zeros(num_envs, num_dof, device=device)
        self.last_actions = torch.zeros_like(self.actions)

        # -----------------------------------------------------------------
        # 力矩与 PD 增益
        # -----------------------------------------------------------------
        self.torques = torch.zeros(num_envs, num_dof, device=device)
        self.p_gains = torch.zeros(num_envs, num_dof, device=device)
        self.d_gains = torch.zeros(num_envs, num_dof, device=device)
        self._init_pd_gains()

        # -----------------------------------------------------------------
        # 接触力
        # -----------------------------------------------------------------
        self.contact_forces = torch.zeros(num_envs, self.num_bodies, 3, device=device)

        # -----------------------------------------------------------------
        # 足部
        # -----------------------------------------------------------------
        self.feet_indices = self._get_feet_indices()
        num_feet = len(self.feet_indices)
        
        self.feet_air_time = torch.zeros(num_envs, num_feet, device=device)
        self.last_contacts = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=device)

        # -----------------------------------------------------------------
        # 命令
        # -----------------------------------------------------------------
        self.commands = torch.zeros(num_envs, 3, device=device)

        # -----------------------------------------------------------------
        # 重置与超时标志
        # -----------------------------------------------------------------
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.time_out_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)

        # -----------------------------------------------------------------
        # 地形课程学习
        # -----------------------------------------------------------------
        self.terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.terrain_types = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.terrain_origins = None  # 待 _setup_terrain() 填充 (num_rows, num_cols, 3)
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        self.custom_origins = False  # 是否使用地形系统控制放置位置
        self.max_terrain_level = 0
        self.init_done = False  # 首次 reset 完成前不执行课程学习更新

        # -----------------------------------------------------------------
        # 噪声缩放
        # -----------------------------------------------------------------
        self.noise_scale_vec = self._get_noise_scale_vec()

        # -----------------------------------------------------------------
        # 奖励
        # -----------------------------------------------------------------
        self.rew_buf = torch.zeros(num_envs, device=device)

        # -----------------------------------------------------------------
        # extras（info 字典）
        # -----------------------------------------------------------------
        self.extras = {}

        # -----------------------------------------------------------------
        # 时间计数器
        # -----------------------------------------------------------------
        self.common_step_counter = 0

        log.debug(
            f"缓冲区初始化完成: num_dof={num_dof}, "
            f"num_bodies={self.num_bodies}, num_feet={num_feet}"
        )

    def _compute_default_dof_pos(self) -> torch.Tensor:
        """计算默认关节位置。"""
        default_dof_pos = torch.zeros(self.num_dof, device=self.device)
        if self.cfg.init_state.default_joint_angles is not None:
            for i, name in enumerate(self.dof_names):
                if name in self.cfg.init_state.default_joint_angles:
                    default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
        return default_dof_pos.unsqueeze(0)  # (1, num_dof)

    def _init_pd_gains(self) -> None:
        """初始化 PD 增益。"""
        if self.cfg.control.stiffness is None or self.cfg.control.damping is None:
            return

        for i, name in enumerate(self.dof_names):
            for pattern, kp in self.cfg.control.stiffness.items():
                if pattern in name:
                    self.p_gains[:, i] = kp
                    break
            for pattern, kd in self.cfg.control.damping.items():
                if pattern in name:
                    self.d_gains[:, i] = kd
                    break

    def _get_feet_indices(self) -> torch.Tensor:
        """获取足部 body 索引。"""
        foot_name = self.cfg.asset.foot_name
        feet_indices = []
        for i, name in enumerate(self.body_names):
            if foot_name in name:
                feet_indices.append(i)
        return torch.tensor(feet_indices, dtype=torch.long, device=self.device)

    def _get_noise_scale_vec(self) -> torch.Tensor:
        """构建观测噪声缩放向量。"""
        noise_scales = self.cfg.observation.noise.noise_scales
        noise_level = self.cfg.observation.noise.noise_level

        obs_dim = self._calculate_obs_dim()
        noise_vec = torch.zeros(obs_dim, device=self.device)

        idx = 0
        # base_lin_vel (3)
        noise_vec[idx : idx + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        idx += 3
        # base_ang_vel (3)
        noise_vec[idx : idx + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3
        # projected_gravity (3)
        noise_vec[idx : idx + 3] = noise_scales.gravity * noise_level
        idx += 3
        # commands (3, no noise)
        idx += 3
        # dof_pos (num_dof)
        noise_vec[idx : idx + self.num_dof] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        idx += self.num_dof
        # dof_vel (num_dof)
        noise_vec[idx : idx + self.num_dof] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        idx += self.num_dof

        return noise_vec

    def _calculate_obs_dim(self) -> int:
        """计算观测维度。"""
        return 3 + 3 + 3 + 3 + self.num_dof + self.num_dof + self.num_dof

    # =========================================================================
    # 初始状态生成（覆盖父类）
    # =========================================================================

    def _get_initial_states(self) -> list[dict]:
        """生成每个环境的初始状态。
        
        地形模式 (custom_origins=True):
            pos = env_origins[i] + (random_xy_offset, init_height_offset)
            多个机器人共享同一地形格，通过 ±1m XY 偏移避免重叠。
        平面模式 (custom_origins=False):
            使用配置的固定 init_state.pos，或如果有 handler._env_origin 则基于 env 网格。
        """
        init_height = float(self.cfg.init_state.pos[2])
        init_rot = list(self.cfg.init_state.rot)

        joint_names = sorted(self.cfg.init_state.default_joint_angles.keys())
        default_joint_pos = {
            name: self.cfg.init_state.default_joint_angles[name]
            for name in joint_names
        }

        states = []
        for i in range(self.cfg.env.num_envs):
            if self.custom_origins:
                # 地形模式：位置来自 terrain_origins 查表 + 随机 XY 偏移 + 站立高度
                origin = self.env_origins[i]
                rand_xy = (torch.rand(2) * 2.0 - 1.0)  # [-1, 1] 范围随机偏移
                pos = torch.tensor(
                    [float(origin[0]) + float(rand_xy[0]),
                     float(origin[1]) + float(rand_xy[1]),
                     float(origin[2]) + init_height],
                    dtype=torch.float32,
                )
            else:
                # 平面模式：使用配置的固定位置，或用 handler 的 env 网格
                spacing = getattr(self.cfg.env, 'env_spacing', 3.0)
                handler_env_origins = getattr(self.handler, '_env_origin', None)
                if handler_env_origins is not None and i < len(handler_env_origins):
                    orig = handler_env_origins[i]
                    pos = torch.tensor(
                        [float(orig[0]) + spacing, float(orig[1]) + spacing, init_height],
                        dtype=torch.float32,
                    )
                else:
                    pos = torch.tensor(list(self.cfg.init_state.pos), dtype=torch.float32)

            state = {
                "objects": {},
                "robots": {
                    self.robot_name: {
                        "pos": pos,
                        "rot": torch.tensor(init_rot, dtype=torch.float32),
                        "vel": torch.zeros(3, dtype=torch.float32),
                        "ang_vel": torch.zeros(3, dtype=torch.float32),
                        "dof_pos": default_joint_pos,
                        "dof_vel": {name: 0.0 for name in joint_names},
                    }
                },
            }
            states.append(state)
        return states

    # =========================================================================
    # 奖励函数
    # =========================================================================

    def _prepare_reward_functions(self) -> None:
        """准备奖励函数列表。"""
        self.reward_functions = []
        self.reward_names = []
        self.reward_weights = []

        if not hasattr(self.cfg, "rewards"):
            return

        scales = self.cfg.rewards.scales
        # 使用 __dataclass_fields__ 避免拾取 @configclass 继承的方法 (copy, from_dict 等)
        field_names = list(scales.__dataclass_fields__.keys()) if hasattr(scales, '__dataclass_fields__') else [n for n in dir(scales) if not n.startswith('_')]
        for name in field_names:
            if name.startswith("_"):
                continue
            weight = getattr(scales, name)
            if not isinstance(weight, (int, float)) or weight == 0:
                continue

            func_name = f"_reward_{name}"
            if hasattr(self, func_name):
                self.reward_functions.append(getattr(self, func_name))
                self.reward_names.append(name)
                self.reward_weights.append(weight)
            else:
                log.warning(f"奖励函数 {func_name} 未找到，跳过")

        log.info(f"已注册 {len(self.reward_functions)} 个奖励函数: {self.reward_names}")

    def _compute_reward(self) -> torch.Tensor:
        """计算总奖励。"""
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for func, weight, name in zip(
            self.reward_functions, self.reward_weights, self.reward_names
        ):
            rew = func()
            total_reward += weight * rew

            if "reward_terms" not in self.extras:
                self.extras["reward_terms"] = {}
            self.extras["reward_terms"][name] = rew.mean().item()

        if hasattr(self.cfg, "rewards") and self.cfg.rewards.only_positive_rewards:
            total_reward = torch.clamp(total_reward, min=0.0)

        return total_reward

    # =========================================================================
    # 回调系统
    # =========================================================================

    def _run_setup_callbacks(self) -> None:
        """运行 setup 回调。"""
        if not hasattr(self.cfg, "callbacks"):
            return

        for name, callback_spec in self.cfg.callbacks.setup.items():
            if isinstance(callback_spec, tuple):
                func, kwargs = callback_spec
                func(self, **kwargs)
            else:
                callback_spec(self)
            log.debug(f"执行 setup callback: {name}")

    def _setup_terrain(self) -> None:
        """自动检测配置，Generate 并注入地形（trimesh/heightfield）。
        
        当 cfg.terrain.mesh_type 为 "trimesh" 或 "heightfield" 时自动调用。
        仿照 example_RMA 的 _get_env_origins()，通过 terrain_levels / terrain_types
        将多个 env 映射到地形网格格子上，多个机器人可共享同一个地形格。
        """
        import copy
        import numpy as np

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type not in ["trimesh", "heightfield"]:
            self.terrain_generator = None
            self.custom_origins = False
            return

        self.custom_origins = True
        terrain_cfg = copy.copy(self.cfg.terrain)

        # ------------------------------------------------------------------
        # 1) 生成地形并注入仿真器
        # ------------------------------------------------------------------
        from MyRobot.terrain.generator import TerrainGenerator
        simulator = self.cfg.simulator
        self.terrain_generator = TerrainGenerator(terrain_cfg, simulator)
        self.terrain_generator.bind_handler(self.handler)
        self.terrain_generator()

        # ------------------------------------------------------------------
        # 2) 建立 terrain_origins 查找表 (num_rows, num_cols, 3)
        # ------------------------------------------------------------------
        self.terrain_origins = torch.from_numpy(
            self.terrain_generator.env_origins  # shape (num_rows, num_cols, 3)
        ).float().to(self.device)
        self.max_terrain_level = terrain_cfg.num_rows

        # ------------------------------------------------------------------
        # 3) 将 num_envs 分配到地形网格：terrain_levels (行=难度) + terrain_types (列=类型)
        #    与 example_RMA/envs/base/legged_robot.py _get_env_origins() 一致
        # ------------------------------------------------------------------
        max_init_level = terrain_cfg.max_init_terrain_level
        if not terrain_cfg.curriculum:
            # 非课程学习：初始均匀分布在所有难度行
            max_init_level = terrain_cfg.num_rows - 1

        self.terrain_levels = torch.fmod(
            torch.arange(self.num_envs, device=self.device),
            max_init_level + 1,
        ).long()

        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            max(self.num_envs / terrain_cfg.num_cols, 1),
            rounding_mode="floor",
        ).long().clamp(max=terrain_cfg.num_cols - 1)

        # 通过 [level, type] 索引查表得到每个 env 的世界坐标
        self.env_origins[:] = self.terrain_origins[
            self.terrain_levels, self.terrain_types
        ]

        log.info(
            f"_setup_terrain: {terrain_cfg.num_rows}x{terrain_cfg.num_cols} 格地形已注入, "
            f"格子大小 {terrain_cfg.terrain_length:.1f}x{terrain_cfg.terrain_width:.1f}m, "
            f"simulator={simulator}, curriculum={terrain_cfg.curriculum}, "
            f"max_init_level={max_init_level}"
        )

    def _update_initial_states_with_terrain(self, env_ids: list[int] | None = None) -> None:
        """根据 env_origins 更新指定 env 的机器人初始状态（XY + Z）。
        
        地形课程学习更新 terrain_levels 后调用此函数，
        将机器人重新放置到新的地形格位置。

        Args:
            env_ids: 要更新的环境索引列表，None 表示全部
        """
        if not self.custom_origins or not isinstance(self._initial_states, TensorState):
            return

        init_height_offset = float(self.cfg.init_state.pos[2])
        robot_name = self.robot_name
        root_state = self._initial_states.robots[robot_name].root_state  # (num_envs, 13)

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 更新 XY：地形格中心 + ±1m 随机偏移
        rand_xy = torch.rand(len(env_ids_t), 2, device=self.device) * 2.0 - 1.0
        root_state[env_ids_t, 0] = self.env_origins[env_ids_t, 0] + rand_xy[:, 0]
        root_state[env_ids_t, 1] = self.env_origins[env_ids_t, 1] + rand_xy[:, 1]
        # 更新 Z：地形高度 + 配置高度偏移
        root_state[env_ids_t, 2] = self.env_origins[env_ids_t, 2] + init_height_offset

        log.debug(
            f"_update_initial_states_with_terrain: 已更新 {len(env_ids_t)} 个 env 的 XYZ 坐标"
        )

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        """基于游戏启发的课程学习，调整 env 所在的地形难度等级。
        
        参照 example_RMA/envs/base/legged_robot.py _update_terrain_curriculum()：
        - 行走距离 > terrain_length/4 → 升级到更难地形
        - 行走距离 < 期望距离 × 0.5  → 降级到更简单地形
        - 通关最高难度的机器人随机分配到任意等级

        Args:
            env_ids: 被重置的环境索引
        """
        if not self.init_done or not self.custom_origins:
            return

        # 计算机器人从出生点走出的距离
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )

        terrain_length = self.cfg.terrain.terrain_length

        # 走得足够远 → 升级到更难地形
        move_up = distance > terrain_length / 4.0

        # 走得太近（未达到速度命令期望的一半距离）→ 降级
        move_down = (
            distance < torch.norm(self.commands[env_ids, :2], dim=1)
            * self.max_episode_length * self.dt * 0.5
        ) * ~move_up

        self.terrain_levels[env_ids] += 1 * move_up.long() - 1 * move_down.long()

        # 通关最高等级的机器人随机分配
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )

        # 根据新 level 重新查表更新 env_origins
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def _run_reset_callbacks(self, env_ids: torch.Tensor) -> None:
        """运行 reset 回调。"""
        if not hasattr(self.cfg, "callbacks"):
            return

        for name, callback_spec in self.cfg.callbacks.reset.items():
            if isinstance(callback_spec, tuple):
                func, kwargs = callback_spec
                func(self, env_ids, **kwargs)
            else:
                callback_spec(self, env_ids)

    def _run_terminate_callbacks(self) -> torch.BoolTensor:
        """运行 terminate 回调。"""
        if not hasattr(self.cfg, "callbacks"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        terminate_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_states = self.handler.get_states()

        for name, callback_spec in self.cfg.callbacks.terminate.items():
            if isinstance(callback_spec, tuple):
                func, kwargs = callback_spec
                terminate_buf |= func(self, env_states, **kwargs)
            else:
                terminate_buf |= callback_spec(self, env_states)

        return terminate_buf

    def _run_pre_step_callbacks(self, actions: torch.Tensor) -> torch.Tensor:
        """运行 pre_step 回调。"""
        if not hasattr(self.cfg, "callbacks"):
            return actions

        for name, callback_spec in self.cfg.callbacks.pre_step.items():
            if isinstance(callback_spec, tuple):
                func, kwargs = callback_spec
                actions = func(self, actions, **kwargs)
            else:
                actions = callback_spec(self, actions)

        return actions

    def _run_in_step_callbacks(self, step_idx: int) -> None:
        """运行 in_step 回调。"""
        if not hasattr(self.cfg, "callbacks"):
            return

        for name, callback_spec in self.cfg.callbacks.in_step.items():
            if isinstance(callback_spec, tuple):
                func, kwargs = callback_spec
                func(self, step_idx, **kwargs)
            else:
                callback_spec(self, step_idx)

    def _run_post_step_callbacks(self, env_states: TensorState) -> None:
        """运行 post_step 回调。"""
        if not hasattr(self.cfg, "callbacks"):
            return

        for name, callback_spec in self.cfg.callbacks.post_step.items():
            if isinstance(callback_spec, tuple):
                func, kwargs = callback_spec
                func(self, env_states, **kwargs)
            else:
                callback_spec(self, env_states)

    # =========================================================================
    # 环境 API（覆盖父类）
    # =========================================================================

    def _extra_spec(self) -> dict[str, BaseQueryType]:
        """注册额外的查询类型。"""
        extra_spec = {}
        extra_spec["contact_forces"] = ContactForces()

        if hasattr(self.cfg, "callbacks") and hasattr(self.cfg.callbacks, "query"):
            extra_spec.update(self.cfg.callbacks.query)

        return extra_spec

    def reset(self, states=None, env_ids=None) -> tuple[torch.Tensor, dict]:
        """重置环境。
        
        Args:
            states: 初始状态（可选）
            env_ids: 要重置的环境索引（可选，默认全部）
            
        Returns:
            (obs, info)
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # 重置 episode 计数器
        self._episode_steps[env_ids] = 0

        # 重置环境特定状态（含地形课程学习，会更新 _initial_states 中的位置）
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids_tensor)

        # 设置状态（使用传入的 states 或经课程学习更新后的默认初始状态）
        states_to_set = self._initial_states if states is None else states
        self.handler.set_states(states=states_to_set, env_ids=env_ids)

        # 更新内部状态
        env_states = self.handler.get_states()
        self._update_buffers_from_states(env_states)
        
        # 生成观测
        obs = self._observation(env_states)
        
        return obs, self.extras

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """重置指定环境的内部状态。"""
        if len(env_ids) == 0:
            return

        # 地形课程学习：根据训练表现调整难度等级
        if self.cfg.terrain.curriculum and self.custom_origins:
            self._update_terrain_curriculum(env_ids)
            # 更新这些 env 的初始状态位置（XYZ 都可能改变）
            self._update_initial_states_with_terrain(env_ids.tolist())

        # 重置命令
        self._resample_commands(env_ids)

        # 重置缓冲区
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.time_out_buf[env_ids] = False
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        
        if len(self.feet_indices) > 0:
            self.feet_air_time[env_ids] = 0.0
            self.last_contacts[env_ids] = False

        # 运行 reset 回调
        self._run_reset_callbacks(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """重采样速度命令。"""
        ranges = self.command_ranges

        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            ranges.lin_vel_x[0], ranges.lin_vel_x[1]
        )
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
            ranges.lin_vel_y[0], ranges.lin_vel_y[1]
        )
        self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            ranges.ang_vel_yaw[0], ranges.ang_vel_yaw[1]
        )

        # 小于阈值时置零
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1, keepdim=True) > 0.2
        ).float()

    def step(self, actions: torch.Tensor) -> tuple:
        """执行一步仿真。"""
        clip_actions = self.cfg.observation.normalization.clip_actions
        self.actions = torch.clamp(actions, -clip_actions, clip_actions)
        self.actions = self._run_pre_step_callbacks(self.actions)

        for i in range(self.cfg.sim.decimation):
            self._run_in_step_callbacks(i)
            
            if self.cfg.control.control_type in ["P", "V"]:
                targets = self._compute_targets()
                self.handler.set_dof_targets(targets)
            elif self.cfg.control.control_type == "T":
                torques = self.actions
                self.handler.set_dof_targets(torques)
            else:
                raise ValueError(f"未知控制类型: {self.cfg.control.control_type}")
            
            self.handler.simulate()

        env_states = self.handler.get_states()
        self._post_physics_step(env_states)
        obs = self._observation(env_states)
        reward = self._compute_reward()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.time_out_buf = self.episode_length_buf >= self.max_episode_length

        done_ids = (self.reset_buf | self.time_out_buf).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.reset(env_ids=done_ids.tolist())

        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()

        return obs, reward, self.reset_buf, self.time_out_buf, self.extras

    def _compute_targets(self) -> torch.Tensor:
        """计算关节目标位置/速度（用于 P/V 控制）。"""
        control_type = self.cfg.control.control_type

        if control_type == "P":
            if self.cfg.control.action_offset:
                targets = self.default_dof_pos + self.cfg.control.action_scale * self.actions
            else:
                targets = self.cfg.control.action_scale * self.actions
            return targets

        elif control_type == "V":
            targets = self.dof_pos + self.cfg.control.action_scale * self.actions * self.dt
            return targets

        else:
            raise ValueError(f"_compute_targets 只支持 P/V 控制，当前: {control_type}")
        
    def _post_physics_step(self, env_states: TensorState) -> None:
        """物理步进后的处理。"""
        self._update_buffers_from_states(env_states)

        # 提取四元数（假设 xyzw 格式）
        self.base_quat = self.root_states[:, 3:7]
        
        self.base_lin_vel = self._quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = self._quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = self._quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._update_commands()
        self._check_termination()
        self._run_post_step_callbacks(env_states)

    def _update_buffers_from_states(self, env_states: TensorState) -> None:
        """从仿真状态更新内部缓冲区。"""
        robot_state = env_states.robots[self.robot_name]

        self.root_states = robot_state.root_state
        self.base_pos = self.root_states[:, :3]
        self.dof_pos = robot_state.joint_pos
        self.dof_vel = robot_state.joint_vel

        # 从 robot_state.extra 中读取接触力（IsaacGym 等后端将其存储于此）
        if hasattr(robot_state, "extra") and robot_state.extra is not None:
            contact = robot_state.extra.get("contact_forces", None)
            if contact is not None:
                self.contact_forces = contact

    def _update_commands(self) -> None:
        """更新命令（周期性重采样）。"""
        resample_time_steps = int(self.cfg.commands.resampling_time / self.dt)
        resample_ids = (self.episode_length_buf % resample_time_steps == 0).nonzero(
            as_tuple=False
        ).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

    def _check_termination(self) -> None:
        """检查终止条件。"""
        self.reset_buf = torch.abs(self.projected_gravity[:, 2]) < 0.5
        self.reset_buf |= self._run_terminate_callbacks()

    # =========================================================================
    # 观测
    # =========================================================================

    def _observation(self, env_states: TensorState) -> torch.Tensor:
        """计算观测向量。"""
        obs = torch.cat(
            [
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ],
            dim=-1,
        )

        # 添加噪声
        if self.cfg.observation.noise.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec

        clip_obs = self.cfg.observation.normalization.clip_observations
        obs = torch.clamp(obs, -clip_obs, clip_obs)

        return obs

    @property
    def observation_space(self) -> spaces.Space:
        """观测空间。"""
        return spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.num_obs,))

    @property
    def action_space(self) -> spaces.Space:
        """动作空间。"""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

    # =========================================================================
    # 辅助函数
    # =========================================================================

    @staticmethod
    def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """将向量从世界坐标系旋转到机体坐标系。
        
        假设四元数格式为 [x, y, z, w]
        """
        q_conj = q.clone()
        q_conj[:, :3] *= -1
        return BaseLocomotionTask._quat_apply(q_conj, v)

    @staticmethod
    def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """使用四元数旋转向量。
        
        假设四元数格式为 [x, y, z, w]
        """
        xyz = q[:, :3]
        w = q[:, 3:4]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    # =========================================================================
    # 奖励函数（基础实现，子类可覆盖）
    # =========================================================================

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """线速度追踪奖励。"""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        """角速度追踪奖励。"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_lin_vel_z(self) -> torch.Tensor:
        """z 方向线速度惩罚。"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        """xy 方向角速度惩罚。"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self) -> torch.Tensor:
        """姿态惩罚。"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self) -> torch.Tensor:
        """扭矩惩罚。"""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self) -> torch.Tensor:
        """关节速度惩罚。"""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self) -> torch.Tensor:
        """关节加速度惩罚。"""
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self) -> torch.Tensor:
        """动作变化率惩罚。"""
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_collision(self) -> torch.Tensor:
        """碰撞惩罚（预留）。"""
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_termination(self) -> torch.Tensor:
        """终止惩罚。"""
        return self.reset_buf.float()

    def _reward_dof_pos_limits(self) -> torch.Tensor:
        """关节位置限制惩罚（预留）。"""
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_air_time(self) -> torch.Tensor:
        """足部空中时间奖励（预留）。"""
        return torch.zeros(self.num_envs, device=self.device)