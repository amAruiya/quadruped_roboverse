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

        # =====================================================================
        # Phase 2: 构建 scenario 并初始化 handler
        # =====================================================================
        scenario = task_cfg_to_scenario(cfg)
        
        # 直接调用 BaseTaskEnv.__init__，跳过 RLTaskEnv
        BaseTaskEnv.__init__(self, scenario, device)
        
        # 从 BaseTaskEnv 复制的初始状态转换为 tensor
        self._initial_states = list_state_to_tensor(
            self.handler, 
            self._get_initial_states(), 
            self.device
        )

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
        # Phase 6: 手动调用 reset
        # =====================================================================
        log.info("执行首次 reset...")
        self.reset(env_ids=list(range(self.num_envs)))

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
        """生成每个环境的初始状态。"""
        init_pos = list(self.cfg.init_state.pos)
        init_rot = list(self.cfg.init_state.rot)

        joint_names = sorted(self.cfg.init_state.default_joint_angles.keys())

        default_joint_pos = {
            name: self.cfg.init_state.default_joint_angles[name]
            for name in joint_names
        }

        template_state = {
            "objects": {},
            "robots": {
                self.robot_name: {
                    "pos": torch.tensor(init_pos, dtype=torch.float32),
                    "rot": torch.tensor(init_rot, dtype=torch.float32),
                    "vel": torch.zeros(3, dtype=torch.float32),
                    "ang_vel": torch.zeros(3, dtype=torch.float32),
                    "dof_pos": default_joint_pos,
                    "dof_vel": {name: 0.0 for name in joint_names},
                }
            },
        }

        return [template_state.copy() for _ in range(self.cfg.env.num_envs)]

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
        for name in dir(scales):
            if name.startswith("_"):
                continue
            weight = getattr(scales, name)
            if weight == 0:
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

        # 设置状态（使用传入的 states 或默认初始状态）
        states_to_set = self._initial_states if states is None else states
        self.handler.set_states(states=states_to_set, env_ids=env_ids)

        # 更新内部状态
        env_states = self.handler.get_states()
        self._update_buffers_from_states(env_states)
        
        # 重置环境特定状态
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids_tensor)
        
        # 生成观测
        obs = self._observation(env_states)
        
        return obs, self.extras

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """重置指定环境的内部状态。"""
        if len(env_ids) == 0:
            return

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
        log.debug(f"Step {self.common_step_counter}: 动作序列 actions={self.actions[0].tolist()}")
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
        self.dof_pos = robot_state.joint_pos
        self.dof_vel = robot_state.joint_vel

        if hasattr(env_states, "contact_forces") and env_states.contact_forces is not None:
            self.contact_forces = env_states.contact_forces.get(self.robot_name, self.contact_forces)

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