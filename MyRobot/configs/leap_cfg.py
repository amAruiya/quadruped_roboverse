"""Leap 四足机器人运动任务配置。

基于 example_RMA/envs/Leap/Leap_config.py 迁移而来。
"""

from __future__ import annotations

from MyRobot.configs.task_cfg import (
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


class LeapTaskCfg(BaseTaskCfg):
    """Leap 机器人运动任务配置。"""

    # =========================================================================
    # 初始状态配置
    # =========================================================================
    class init_state(InitStateCfg):
        pos: tuple[float, float, float] = (0.0, 0.0, 0.365)
        default_joint_angles: dict[str, float] = {
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
        }

    # =========================================================================
    # 控制配置
    # =========================================================================
    class control(ControlCfg):
        control_type: str = "P"
        stiffness: dict[str, float] = {
            "HAA": 28.0,
            "HFE": 28.0,
            "KFE": 28.0,
        }
        damping: dict[str, float] = {
            "HAA": 0.8,
            "HFE": 0.8,
            "KFE": 0.8,
        }
        action_scale: float = 0.25
        action_offset: bool = True

    # =========================================================================
    # 地形配置
    # =========================================================================
    class terrain(TerrainCfg):
        mesh_type: str = "trimesh"
        curriculum: bool = True
        # terrain_proportions: [smooth_slope, rough_slope, stairs_up, stairs_down, discrete, stepping_stones, gaps]
        terrain_proportions: list[float] = [0.3, 0.3, 0.0, 0.1, 0.1, 0.1, 0.1]

    # =========================================================================
    # 命令配置
    # =========================================================================
    class commands(CommandsCfg):
        curriculum: bool = True
        heading_command: bool = True

        class ranges(CommandRanges):
            lin_vel_x: tuple[float, float] = (-1.5, 1.5)
            lin_vel_y: tuple[float, float] = (-1.0, 1.0)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)
            heading: tuple[float, float] = (-3.14, 3.14)

    # =========================================================================
    # 域随机化配置
    # =========================================================================
    class domain_rand(DomainRandCfg):
        randomize_friction: bool = True
        friction_range: tuple[float, float] = (0.5, 1.75)
        randomize_base_mass: bool = True
        added_mass_range: tuple[float, float] = (0.0, 2.0)
        randomize_kp_kd: bool = True
        kp_range: tuple[float, float] = (20.0, 35.0)
        kd_range: tuple[float, float] = (0.4, 1.0)
        push_robots: bool = True

    # =========================================================================
    # 奖励配置
    # =========================================================================
    class rewards(RewardsCfg):
        soft_dof_pos_limit: float = 0.9
        base_height_target: float = 0.355

        class scales(RewardScales):
            # 追踪奖励
            tracking_lin_vel: float = 1.0
            tracking_ang_vel: float = 0.5

            # 正则化惩罚
            lin_vel_z: float = -2.0
            ang_vel_xy: float = -0.05
            orientation: float = -0.2
            torques: float = -0.0002
            dof_vel: float = -0.0
            dof_acc: float = -2.5e-7
            action_rate: float = -0.01

            # Leap 特定奖励
            dof_pos_limits: float = -10.0
            feet_air_time: float = 1.0
            collision: float = -1.0
            termination: float = -0.0

    # =========================================================================
    # 资产配置（任务逻辑相关）
    # =========================================================================
    class asset:
        foot_name: str = "FOOT"
        penalize_contacts_on: list[str] = ["calf"]
        terminate_after_contacts_on: list[str] = ["base", "thigh"]

    # =========================================================================
    # 场景构建配置
    # =========================================================================
    robots: str = "leap"  # 将在 task_cfg_to_scenario 中解析为 RobotCfg
    scene: str = "plane"
    simulator: str = "isaacgym"
    headless: bool = False


# 导出配置实例
leap_task_cfg = LeapTaskCfg()