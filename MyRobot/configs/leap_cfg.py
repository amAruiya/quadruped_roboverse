"""Leap 四足机器人运动任务配置。

基于 example_RMA/envs/Leap/Leap_config.py 迁移而来。
"""

from __future__ import annotations
from metasim.utils import configclass
from MyRobot.configs.task_cfg import (
    AssetCfg,
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

@configclass
class LeapTaskCfg(BaseTaskCfg):
    """Leap 机器人运动任务配置。"""

    # =========================================================================
    # 初始状态配置
    # =========================================================================
    init_state: InitStateCfg = InitStateCfg(
        pos=(0.0, 0.0, 0.365),
        default_joint_angles={
            # 左前腿
            "LF_HAA": -0.2,
            "LF_HFE": 0.64,
            "LF_KFE": -1.27,
            # 左后腿
            "LH_HAA": -0.2,
            "LH_HFE": 0.64,
            "LH_KFE": -1.27,
            # 右前腿
            "RF_HAA": 0.2,
            "RF_HFE": 0.64,
            "RF_KFE": -1.27,
            # 右后腿
            "RH_HAA": 0.2,
            "RH_HFE": 0.64,
            "RH_KFE": -1.27,
        },
    )

    # =========================================================================
    # 控制配置
    # =========================================================================
    control: ControlCfg = ControlCfg(
        control_type="P",
        stiffness={
            "HAA": 28.0,
            "HFE": 28.0,
            "KFE": 28.0,
        },
        damping={
            "HAA": 0.8,
            "HFE": 0.8,
            "KFE": 0.8,
        },
        action_scale=0.25,
        action_offset=True,
    )

    # =========================================================================
    # 地形配置
    # =========================================================================
    terrain: TerrainCfg = TerrainCfg(
        mesh_type="trimesh",
        curriculum=True,
        # terrain_proportions: [smooth_slope, rough_slope, stairs_up, stairs_down, discrete, stepping_stones, gaps]
        terrain_proportions=[0.3, 0.3, 0.0, 0.1, 0.1, 0.1, 0.1],
    )

    # =========================================================================
    # 命令配置
    # =========================================================================
    commands: CommandsCfg = CommandsCfg(
        heading_command=True,
        ranges=CommandRanges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_yaw=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )

    # =========================================================================
    # 域随机化配置
    # =========================================================================
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

    # =========================================================================
    # 奖励配置
    # =========================================================================
    rewards: RewardsCfg = RewardsCfg(
        soft_dof_pos_limit=0.9,
        base_height_target=0.355,
        scales=RewardScales(
            # 追踪奖励
            tracking_lin_vel=1.0,
            tracking_ang_vel=0.5,
            # 正则化惩罚
            lin_vel_z=-2.0,
            ang_vel_xy=-0.05,
            orientation=-0.2,
            torques=-0.0002,
            dof_vel=-0.0,
            dof_acc=-2.5e-7,
            action_rate=-0.01,
            # Leap 特定奖励
            dof_pos_limits=-10.0,
            feet_air_time=1.0,
            collision=-1.0,
            termination=-0.0,
        ),
    )

    # =========================================================================
    # 资产配置（任务逻辑相关）
    # =========================================================================
    asset: AssetCfg = AssetCfg(
        foot_name="FOOT",
        penalize_contacts_on=["calf"],
        terminate_after_contacts_on=["base", "thigh"],
    )

    # =========================================================================
    # 场景构建配置
    # =========================================================================
    robots: str = "leap"  # 将在 task_cfg_to_scenario 中解析为 RobotCfg
    scene: str = None
    simulator: str = "isaacgym"
    headless: bool = False


# 导出配置实例
leap_task_cfg = LeapTaskCfg()