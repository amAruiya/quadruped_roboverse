"""Leap 四足机器人运动任务配置。

本文件实现了 Leap 机器人的任务配置类，包括: 
- LeapRewardScales: Leap 特有的奖励缩放配置 (覆盖 RewardScales)
- LeapRewardsCfg: Leap 奖励配置 (覆盖 RewardsCfg)
- LeapTaskCfg: Leap 机器人运动任务配置 (覆盖 BaseTaskCfg)
"""

from __future__ import annotations
from metasim.utils import configclass
from MyRobot.configs.task_cfg import (
    AssetCfg,
    CallbacksCfg,
    SimCfg,
    BaseTaskCfg,
    CommandRanges,
    CommandsCfg,
    ControlCfg,
    DomainRandCfg,
    InitStateCfg,
    RewardScales,
    RewardsCfg,
    TerrainCfg,
)
from MyRobot.callbacks.terminate import (
    undesired_contact_termination,
    root_height_below_minimum,
)

@configclass
class LeapRewardScales(RewardScales):
    """Leap 特有的奖励缩放配置。"""

    # 继承基础奖励
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.2
    torques: float = -0.0002
    dof_vel: float = -0.0
    dof_acc: float = -2.5e-7
    action_rate: float = -0.01
    dof_pos_limits: float = -10.0
    feet_air_time: float = 1.0
    collision: float = -1.0
    termination: float = -0.0

    # Leap 特有奖励
    leg_effort_symmetry: float = -0.00002
    hip_pos: float = -1.0
    thigh_pos: float = -0.8
    feet_contact_forces: float = -0.01
    base_height: float = -0.0  # 可选启用
    stand_still: float = -0.0  # 可选启用
    stumble: float = -0.0  # 可选启用

@configclass
class LeapRewardsCfg(RewardsCfg):
    """Leap 奖励配置。"""

    soft_dof_pos_limit: float = 0.9
    base_height_target: float = 0.355
    max_contact_force: float = 100.0
    ema_alpha: float = 0.4  # 用于 leg_effort_symmetry 的 EMA 系数
    scales: LeapRewardScales = LeapRewardScales()


@configclass
class LeapTaskCfg(BaseTaskCfg):
    """Leap 机器人运动任务配置。"""

    sim : SimCfg = SimCfg(
        dt=0.001,
        decimation=4,
    )

    init_state: InitStateCfg = InitStateCfg(
        pos=(0.0, 0.0, 0.365),
        # default_joint_angles 已移除，自动从 RobotCfg.default_joint_positions 读取
    )

    control: ControlCfg = ControlCfg(
        control_type="P",
        # stiffness/damping 已移除，自动从 RobotCfg.actuators 读取
        action_scale=0.25,
        action_offset=True,
    )

    terrain: TerrainCfg = TerrainCfg(
        mesh_type="trimesh",
        curriculum=True,
        # terrain_proportions: [flat, rough, slope, stairs_up, stairs_down, discrete, stepping_stones]
        terrain_proportions=[0.3, 0.3, 0.0, 0.1, 0.1, 0.1, 0.1],
    )

    commands: CommandsCfg = CommandsCfg(
        heading_command=True,
        ranges=CommandRanges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_yaw=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )

    domain_rand: DomainRandCfg = DomainRandCfg(
        randomize_friction=True,
        friction_range=(0.5, 1.75),
        randomize_base_mass=True,
        added_mass_range=(0.0, 2.0),
        randomize_kp_kd=True,
        kp_range=(20.0, 35.0),
        kd_range=(0.4, 1.0),
        push_robots=True,
    )

    rewards: LeapRewardsCfg = LeapRewardsCfg()

    asset: AssetCfg = AssetCfg(
        foot_name="FOOT",
        penalize_contacts_on=["calf"],
        terminate_after_contacts_on=["base", "thigh"],
    )

    callbacks: CallbacksCfg = CallbacksCfg(
        terminate={
            "undesired_contact": (undesired_contact_termination, {
                "contact_body_names": ["base", "thigh"],
                "force_threshold": 1.0,
            }),
            "root_height": (root_height_below_minimum, {
                "min_height": 0.25,
            }),
        },
    )

    robots: str = "leap"  # 将在 task_cfg_to_scenario 中解析为 RobotCfg
    scene: str = None
    simulator: str = "isaacgym"
    headless: bool = False


# 导出配置实例
leap_task_cfg = LeapTaskCfg()