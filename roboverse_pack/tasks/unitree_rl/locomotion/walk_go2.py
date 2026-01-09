from __future__ import annotations

import copy
import torch

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse

# 注意：请确保以下配置文件已在对应的路径中定义
try:
    from roboverse_learn.rl.unitree_rl.configs.locomotion.walk_go2 import (
        WalkGo2EnvCfg,
        WalkGo2RslRlTrainCfg,
    )
except ImportError:
    # 如果配置文件尚未创建，此处仅作占位以避免语法检查报错
    # 实际运行时必须提供有效的配置类
    class WalkGo2EnvCfg: pass
    class WalkGo2RslRlTrainCfg: pass

from roboverse_pack.tasks.unitree_rl.base import LeggedRobotTask


@register_task(
    "unitree_rl.walk_go2",
    "go2.walk_go2",
    "walk_go2",
)
class WalkGo2Task(LeggedRobotTask):
    """
    注册的任务包装器，包含场景默认值和配置钩子。
    专为 Unitree Go2 四足机器人适配。
    """

    env_cfg_cls = WalkGo2EnvCfg
    train_cfg_cls = WalkGo2RslRlTrainCfg
    task_name = "walk_go2"

    scenario = ScenarioCfg(
        robots=["go2"],  # 对应 go2_cfg.py 中的 name="go2"
        objects=[],
        cameras=[],
        num_envs=128,    # 默认环境数量
        simulator="isaacgym",
        headless=True,
        env_spacing=2.5, # 环境间距
        decimation=1,    # 控制频率抽取因子
        sim_params=SimParamCfg(
            dt=0.005,    # 物理仿真步长 5ms
            substeps=1,
            num_threads=10,
            solver_type=1,
            num_position_iterations=4,
            num_velocity_iterations=0,
            contact_offset=0.01,
            rest_offset=0.0,
            bounce_threshold_velocity=0.5,
            max_depenetration_velocity=1.0,
            default_buffer_size_multiplier=5,
            replace_cylinder_with_capsule=True,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
        ),
        lights=[
            DomeLightCfg(
                intensity=800.0,
                color=(0.85, 0.9, 1.0),
            )
        ],
    )

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device: str | torch.device | None = None,
        env_cfg: WalkGo2EnvCfg | None = None,
    ) -> None:
        scenario_copy = copy.deepcopy(scenario or type(self).scenario)
        scenario_copy.__post_init__()

        if env_cfg is None:
            # 如果未传入配置，实例化默认的环境配置类
            env_cfg = type(self).env_cfg_cls()

        if device is None:
            device = "cpu" if scenario_copy.simulator == "mujoco" else ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(scenario=scenario_copy, config=env_cfg, device=device)

    def _init_buffers(self):
        """初始化张量缓冲区并定义观测空间的缩放与噪声。"""
        # 观测空间维度计算 (num_obs_single):
        # 3 (commands) + 3 (base_ang_vel) + 3 (projected_gravity) +
        # num_actions (dof_pos) + num_actions (dof_vel) + num_actions (prev_actions) + 2 (gait_phase)
        # 对于 Go2, num_actions 为 12
        self.num_obs_single = 3 + 3 + 3 + self.num_actions * 3 + 2
        
        # 特权观测空间维度 (num_priv_obs_single):
        # 包含真实基座线速度 (base_lin_vel)
        self.num_priv_obs_single = 3 + 3 + 3 + 3 + self.num_actions * 3 + 2
        
        # 重写部分超参数
        self.obs_clip_limit = 100.0
        self.obs_scale = torch.ones(size=(self.num_obs_single,), dtype=torch.float, device=self.device)
        self.priv_obs_scale = torch.ones(size=(self.num_priv_obs_single,), dtype=torch.float, device=self.device)
        self.obs_noise = torch.zeros(size=(self.num_obs_single,), dtype=torch.float, device=self.device)

        ##################### 观测值缩放 (Observation Scale) #####################
        # 索引对应关系取决于 _compute_task_observations 中的拼接顺序
        self.obs_scale[0:2] = 0.2  # 线性速度指令 linear vel commands
        self.obs_scale[2] = 0.25   # 角速度指令 angular vel commands
        self.obs_scale[3:6] = 0.25 # 基座角速度 base angular velocity
        # [6:9] projected_gravity 默认为 1.0
        # [9:9+N] joint position error 默认为 1.0
        self.obs_scale[9 + self.num_actions : 9 + 2 * self.num_actions] = 0.05  # 关节速度 joint velocity

        ##################### 特权观测值缩放 (Privileged Observation Scale) #####################
        self.priv_obs_scale[0:2] = 0.2  # 线性速度指令
        self.priv_obs_scale[2] = 0.25   # 角速度指令
        self.priv_obs_scale[3:6] = 2.0  # 基座线速度 (特权信息)
        self.priv_obs_scale[6:9] = 0.25 # 基座角速度
        # projected_gravity, joint_pos 等后续项
        self.priv_obs_scale[12 + self.num_actions : 12 + 2 * self.num_actions] = 0.05  # 关节速度

        ################### 噪声向量 (Noise Vector) ####################
        # [0:3] -> commands (通常不添加直接噪声，或在指令采样逻辑中处理)
        self.obs_noise[3:6] = 0.2  # base_ang_vel
        self.obs_noise[6:9] = 0.05  # projected_gravity
        self.obs_noise[9 : 9 + self.num_actions] = 0.01 # joint positions
        self.obs_noise[9 + self.num_actions : 9 + 2 * self.num_actions] = 1.5  # joint velocities
        
        return super()._init_buffers()

    def gait_phase(self, period: float = 0.8) -> torch.Tensor:
        """基于回合步数计算全局步态相位 (sin/cos 编码)。"""
        global_phase = (self._episode_steps * self.step_dt) % period / period

        phase = torch.zeros(self.num_envs, 2, device=self.device)
        phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
        phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
        return phase

    def _compute_task_observations(self, env_states: TensorState):
        """计算任务观测值和特权观测值。"""
        robot_state = env_states.robots[self.name]
        base_quat = robot_state.root_state[:, 3:7]
        
        # 将世界坐标系下的速度和重力投影到基座坐标系
        base_lin_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, robot_state.root_state[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        gait_phase = self.gait_phase()

        # 关节位置误差 (当前位置 - 默认位置)
        q = env_states.robots[self.name].joint_pos - self.default_dof_pos
        dq = env_states.robots[self.name].joint_vel

        # 构建普通观测缓冲 (obs_buf)
        obs_buf = torch.cat(
            (
                self.commands_manager.value,  # 3
                base_ang_vel,                 # 3
                projected_gravity,            # 3
                q,                            # num_actions (12)
                dq,                           # num_actions (12)
                self.actions,                 # num_actions (12) - 上一帧动作
                gait_phase,                   # 2
            ),
            dim=-1,
        )

        # 构建特权观测缓冲 (priv_obs_buf) - 包含真实的线速度
        priv_obs_buf = torch.cat(
            (
                self.commands_manager.value,
                base_lin_vel,                 # 3 (Privileged)
                base_ang_vel,
                projected_gravity,
                q,
                dq,
                self.actions,
                gait_phase,
            ),
            dim=-1,
        )

        # 添加观测噪声 (仅对普通观测)
        obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.obs_noise

        # 裁剪范围并应用缩放因子
        obs_buf = obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.obs_scale
        priv_obs_buf = priv_obs_buf.clip(-self.obs_clip_limit, self.obs_clip_limit) * self.priv_obs_scale

        return obs_buf, priv_obs_buf