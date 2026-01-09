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
from metasim.task.rl_task import RLTaskEnv
from metasim.types import TensorState
from metasim.utils.state import RobotState

from MyRobot.configs.task_cfg import BaseTaskCfg
from MyRobot.utils.helper import task_cfg_to_scenario


class BaseLocomotionTask(RLTaskEnv):
    """四足机器人运动任务基类。

    该类封装了 metasim handler,提供与 example_RMA 兼容的接口,
    同时保持后端无关性。

    生命周期：
        __init__(task_cfg)
            ├─ _build_scenario(task_cfg)  # 构建 scenario
            ├─ super().__init__(scenario)     # 创建 handler
            ├─ _parse_cfg(task_cfg)
            ├─ _init_buffers()
            ├─ _prepare_reward_functions()
            └─ _run_setup_callbacks()

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
        """初始化运动任务。

        Args:
            cfg: 任务配置
            device: 计算设备
        """
        self.cfg = cfg
        self._parse_cfg()

        # 设备设置
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 从任务配置构建场景配置
        scenario = task_cfg_to_scenario(cfg)

        # 初始化父类（会调用 handler.launch()）
        super().__init__(scenario, device)

        # 机器人名称
        self.robot_name = self.robot.name if hasattr(self, "robot") else "robot"

        # 获取关节信息
        self.num_dof = len(self.handler.get_joint_names(self.robot_name))
        self.dof_names = self.handler.get_joint_names(self.robot_name, sort=True)

        # 获取刚体信息
        self.body_names = self.handler.get_body_names(self.robot_name, sort=True)
        self.num_bodies = len(self.body_names)

        # 初始化任务特定缓冲区
        self._init_buffers()

        # 准备奖励函数
        self._prepare_reward_functions()

        # 运行 setup 回调
        self._run_setup_callbacks()

        # 标记初始化完成
        self.init_done = True

        log.info(
            f"BaseLocomotionTask 初始化完成: {self.num_envs} 环境, "
            f"{self.num_dof} DOF, 设备: {self.device}"
        )

    # =========================================================================
    # 配置解析
    # =========================================================================

    def _parse_cfg(self) -> None:
        """解析配置，计算派生参数。"""
        self.dt = self.cfg.sim.dt * self.cfg.sim.decimation
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.dt)

        # 观测缩放
        self.obs_scales = self.cfg.observation.normalization.obs_scales
        self.command_ranges = self.cfg.commands.ranges

        log.debug(
            f"配置解析完成: dt={self.dt:.4f}s, "
            f"max_episode_length={self.max_episode_length}"
        )

    # =========================================================================
    # 缓冲区初始化
    # =========================================================================

    def _init_buffers(self) -> None:
        """初始化 PyTorch 缓冲区。"""
        # -------------------- 状态缓冲区 --------------------
        # 根状态: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
        self.root_states = torch.zeros(self.num_envs, 13, device=self.device)
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)  # 机体坐标系
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)  # 机体坐标系

        # 关节状态
        self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        # -------------------- 动作缓冲区 --------------------
        self.actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        # -------------------- 命令缓冲区 --------------------
        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, device=self.device
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
        )

        # -------------------- 时间追踪 --------------------
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.common_step_counter = 0

        # -------------------- 物理常量 --------------------
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(
            self.num_envs, 1
        )
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

        # -------------------- 默认姿态 --------------------
        self.default_dof_pos = self._compute_default_dof_pos()

        # -------------------- PD 增益 --------------------
        self.p_gains = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.d_gains = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self._init_pd_gains()

        # -------------------- 终止/奖励相关 --------------------
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -------------------- 扭矩缓冲区 --------------------
        self.torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.torque_limits = torch.ones(self.num_envs, self.num_dof, device=self.device) * 100.0

        # -------------------- 足部相关 --------------------
        self.feet_indices = self._get_feet_indices()
        if len(self.feet_indices) > 0:
            self.feet_air_time = torch.zeros(
                self.num_envs, len(self.feet_indices), device=self.device
            )
            self.last_contacts = torch.zeros(
                self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
            )
        else:
            self.feet_air_time = torch.zeros(self.num_envs, 0, device=self.device)
            self.last_contacts = torch.zeros(self.num_envs, 0, dtype=torch.bool, device=self.device)

        # -------------------- 接触力 --------------------
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)

        # -------------------- 噪声向量 --------------------
        if self.cfg.observation.noise.add_noise:
            self.noise_scale_vec = self._get_noise_scale_vec()

        # -------------------- 额外信息 --------------------
        self.extras = {}

        log.debug(f"缓冲区初始化完成: {self.num_dof} DOF, {self.num_bodies} bodies")

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
            # 按名称模式匹配
            for pattern, kp in self.cfg.control.stiffness.items():
                if pattern in name:
                    self.p_gains[:, i] = kp
                    break
            for pattern, kd in self.cfg.control.damping.items():
                if pattern in name:
                    self.d_gains[:, i] = kd
                    break

    def _get_feet_indices(self) -> torch.Tensor:
        """获取足部 body 索引。

        子类应覆盖此方法以返回正确的足部索引。
        """
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

        # 计算观测维度
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
        # actions (num_dof, no noise)

        return noise_vec

    def _calculate_obs_dim(self) -> int:
        """计算观测维度。"""
        # 基础观测: lin_vel(3) + ang_vel(3) + gravity(3) + commands(3) +
        #          dof_pos(n) + dof_vel(n) + actions(n)
        return 3 + 3 + 3 + 3 + self.num_dof + self.num_dof + self.num_dof

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

            # 查找对应的奖励函数
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

            # 记录到 extras
            if "reward_terms" not in self.extras:
                self.extras["reward_terms"] = {}
            self.extras["reward_terms"][name] = rew.mean().item()

        # 只保留正奖励（如果配置了）
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

        # 获取当前状态
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

        # 注册接触力查询
        extra_spec["contact_forces"] = ContactForces()

        # 添加配置中的查询
        if hasattr(self.cfg, "callbacks") and hasattr(self.cfg.callbacks, "query"):
            extra_spec.update(self.cfg.callbacks.query)

        return extra_spec

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> tuple[torch.Tensor, dict]:
        """重置环境。

        Args:
            env_ids: 要重置的环境索引

        Returns:
            观测和信息字典
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # 重置选定的环境
        self._reset_idx(env_ids)

        # 获取新状态
        env_states = self.handler.get_states()

        # 更新缓冲区
        self._update_buffers_from_states(env_states)

        # 计算观测
        obs = self._observation(env_states)

        return obs, self.extras

    def _reset_idx(self, env_ids: list[int]) -> None:
        """重置指定环境。"""
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 重置根状态
        self._reset_root_states(env_ids_tensor)

        # 重置关节状态
        self._reset_dof_states(env_ids_tensor)

        # 重采样命令
        self._resample_commands(env_ids_tensor)

        # 重置缓冲区
        self.episode_length_buf[env_ids_tensor] = 0
        self.reset_buf[env_ids_tensor] = False
        self.time_out_buf[env_ids_tensor] = False
        self.last_actions[env_ids_tensor] = 0.0
        self.last_dof_vel[env_ids_tensor] = 0.0
        if len(self.feet_indices) > 0:
            self.feet_air_time[env_ids_tensor] = 0.0
            self.last_contacts[env_ids_tensor] = False

        # 应用状态到仿真器
        self._apply_reset_states(env_ids)

        # 运行 reset 回调
        self._run_reset_callbacks(env_ids_tensor)

    def _reset_root_states(self, env_ids: torch.Tensor) -> None:
        """重置根状态。"""
        init_state = self.cfg.init_state
        self.root_states[env_ids, :3] = torch.tensor(init_state.pos, device=self.device)
        self.root_states[env_ids, 3:7] = torch.tensor(init_state.rot, device=self.device)
        self.root_states[env_ids, 7:10] = torch.tensor(init_state.lin_vel, device=self.device)
        self.root_states[env_ids, 10:13] = torch.tensor(init_state.ang_vel, device=self.device)

    def _reset_dof_states(self, env_ids: torch.Tensor) -> None:
        """重置关节状态。"""
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """重采样速度命令。"""
        ranges = self.command_ranges

        # 线速度 x
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            ranges.lin_vel_x[0], ranges.lin_vel_x[1]
        )
        # 线速度 y
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
            ranges.lin_vel_y[0], ranges.lin_vel_y[1]
        )
        # 角速度 yaw
        self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            ranges.ang_vel_yaw[0], ranges.ang_vel_yaw[1]
        )

        # 小命令置零
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1, keepdim=True) > 0.2
        ).float()

    def _apply_reset_states(self, env_ids: list[int]) -> None:
        """将重置状态应用到仿真器。"""
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 构建状态字典
        robot_state = RobotState(
            root_state=self.root_states[env_ids_tensor],
            joint_pos=self.dof_pos[env_ids_tensor],
            joint_vel=self.dof_vel[env_ids_tensor],
        )

        states = TensorState(robots={self.robot_name: robot_state}, objects={})
        self.handler.set_states(states, env_ids=env_ids, zero_vel=False)

    def step(self, actions: torch.Tensor) -> tuple:
        """执行一步仿真。

        Args:
            actions: 动作张量 (num_envs, num_actions)

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # 裁剪动作
        clip_actions = self.cfg.observation.normalization.clip_actions
        self.actions = torch.clamp(actions, -clip_actions, clip_actions)

        # pre_step 回调
        self.actions = self._run_pre_step_callbacks(self.actions)

        # 执行物理步进
        for i in range(self.cfg.sim.decimation):
            # in_step 回调
            self._run_in_step_callbacks(i)

            # 计算控制目标
            targets = self._compute_targets()

            # 设置关节目标
            self.handler.set_dof_targets({self.robot_name: {"dof_pos_target": targets}})

            # 仿真一步
            self.handler.simulate()

        # 获取新状态
        env_states = self.handler.get_states()

        # 后处理
        self._post_physics_step(env_states)

        # 计算观测
        obs = self._observation(env_states)

        # 计算奖励
        reward = self._compute_reward()

        # 时间推进
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 检查超时
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length

        # 处理重置
        done_ids = (self.reset_buf | self.time_out_buf).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self._reset_idx(done_ids.tolist())

        # 更新动作缓冲区
        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()

        return obs, reward, self.reset_buf, self.time_out_buf, self.extras

    def _compute_targets(self) -> torch.Tensor:
        """计算关节目标位置/扭矩。"""
        control_type = self.cfg.control.control_type

        if control_type == "P":
            # 位置控制
            if self.cfg.control.action_offset:
                targets = self.default_dof_pos + self.cfg.control.action_scale * self.actions
            else:
                targets = self.cfg.control.action_scale * self.actions
            return targets

        elif control_type == "V":
            # 速度控制（转为位置目标）
            targets = self.dof_pos + self.cfg.control.action_scale * self.actions * self.dt
            return targets

        elif control_type == "T":
            # 扭矩控制
            self.torques = self.cfg.control.action_scale * self.actions
            # 通过 PD 计算等效目标位置（或直接返回当前位置）
            return self.dof_pos

        else:
            raise ValueError(f"未知控制类型: {control_type}")

    def _post_physics_step(self, env_states: TensorState) -> None:
        """物理步进后的处理。"""
        # 更新缓冲区
        self._update_buffers_from_states(env_states)

        # 计算机体坐标系下的速度
        self.base_lin_vel = self._quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = self._quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        # 计算投影重力
        self.projected_gravity = self._quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 更新命令（如果需要重采样）
        self._update_commands()

        # 检查终止条件
        self._check_termination()

        # post_step 回调
        self._run_post_step_callbacks(env_states)

    def _update_buffers_from_states(self, env_states: TensorState) -> None:
        """从仿真状态更新内部缓冲区。"""
        robot_state = env_states.robots[self.robot_name]

        # 更新根状态
        self.root_states = robot_state.root_state

        # 更新关节状态
        self.dof_pos = robot_state.joint_pos
        self.dof_vel = robot_state.joint_vel

        # 更新接触力（如果可用）
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
        # 基础终止条件：倾覆
        self.reset_buf = torch.abs(self.projected_gravity[:, 2]) < 0.5

        # 运行 terminate 回调
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
        if self.cfg.observation.noise.add_noise and hasattr(self, "noise_scale_vec"):
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec

        # 裁剪
        clip_obs = self.cfg.observation.normalization.clip_observations
        obs = torch.clamp(obs, -clip_obs, clip_obs)

        return obs

    @property
    def num_obs(self) -> int:
        """观测维度。"""
        return self._calculate_obs_dim()

    @property
    def observation_space(self) -> spaces.Space:
        """观测空间。"""
        return spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.num_obs,))

    @property
    def action_space(self) -> spaces.Space:
        """动作空间。"""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_dof,))

    # =========================================================================
    # 辅助函数
    # =========================================================================

    @staticmethod
    def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """将向量从世界坐标系旋转到机体坐标系。"""
        q_conj = q.clone()
        q_conj[:, :3] *= -1
        return BaseLocomotionTask._quat_apply(q_conj, v)

    @staticmethod
    def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """使用四元数旋转向量。"""
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