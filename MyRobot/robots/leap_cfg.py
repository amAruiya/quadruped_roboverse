"""Leap 四足机器人配置。

基于 example_RMA/envs/Leap 和 roboverse_data/robots/Leap/urdf/Leap.urdf 创建。
"""

from __future__ import annotations

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class LeapCfg(RobotCfg):
    """Leap 四足机器人配置。

    Leap 是一款四足机器人,具有 12 个自由度 (4 条腿 × 3 个关节)。
    每条腿包含: HAA (Hip Abduction/Adduction), HFE (Hip Flexion/Extension), KFE (Knee Flexion/Extension)
    """

    # ==================== 基础信息 ====================
    name: str = "leap"
    num_joints: int = 12
    urdf_path: str = "roboverse_data/robots/Leap/urdf/Leap.urdf"
    mjcf_path: str = "roboverse_data/robots/Leap/mjcf/Leap.xml"
    usd_path: str = "roboverse_data/robots/Leap/usd/Leap.usd"

    # ==================== 物理属性 ====================
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    enabled_gravity: bool = True
    fix_base_link: bool = True
    enabled_self_collisions: bool = False

    # ==================== 仿真器特定配置 ====================
    isaacgym_flip_visual_attachments: bool = False  # Leap URDF 已正确对齐
    collapse_fixed_joints: bool = True

    # ==================== 执行器配置 ====================
    # 参考 example_RMA/envs/Leap/Leap_config.py 中的 PD 增益
    # stiffness (kp): 28.0 N⋅m/rad
    # damping (kd): 0.8 N⋅m⋅s/rad
    # torque_limit: 33.5 N⋅m (来自 URDF effort limit)
    # velocity_limit: 21 rad/s (来自 URDF velocity limit)

    actuators: dict[str, BaseActuatorCfg] = {
        # 左前腿 (LF)
        "LF_HAA": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "LF_HFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "LF_KFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        # 左后腿 (LH)
        "LH_HAA": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "LH_HFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "LH_KFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        # 右前腿 (RF)
        "RF_HAA": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "RF_HFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "RF_KFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        # 右后腿 (RH)
        "RH_HAA": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "RH_HFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
        "RH_KFE": BaseActuatorCfg(
            stiffness=28.0,
            damping=0.8,
            torque_limit=33.5,
            velocity_limit=21.0,
        ),
    }

    # ==================== 关节限制 ====================
    # 从 URDF 中提取的关节限制 (单位: 弧度)
    joint_limits: dict[str, tuple[float, float]] = {
        # HAA joints: Hip Abduction/Adduction
        "LF_HAA": (-0.8028514559173915, 0.8028514559173915),
        "LH_HAA": (-0.8028514559173915, 0.8028514559173915),
        "RF_HAA": (-0.8028514559173915, 0.8028514559173915),
        "RH_HAA": (-0.8028514559173915, 0.8028514559173915),
        # HFE joints: Hip Flexion/Extension
        "LF_HFE": (-1.0471975511965976, 4.1887902047863905),
        "LH_HFE": (-1.0471975511965976, 4.1887902047863905),
        "RF_HFE": (-1.0471975511965976, 4.1887902047863905),
        "RH_HFE": (-1.0471975511965976, 4.1887902047863905),
        # KFE joints: Knee Flexion/Extension
        "LF_KFE": (-3.1965336943312392, -0.6162978572970231),
        "LH_KFE": (-3.1965336943312392, -0.6162978572970231),
        "RF_KFE": (-3.1965336943312392, -0.6162978572970231),
        "RH_KFE": (-3.1965336943312392, -0.6162978572970231),
    }

    # ==================== 默认关节位置 ====================
    # 参考 example_RMA/envs/Leap/Leap_config.py 中的 default_joint_angles
    default_joint_positions: dict[str, float] = {
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

    # ==================== 控制类型 ====================
    # 使用力矩控制 (effort control)
    control_type: dict[str, str] = {
        "LF_HAA": "effort",
        "LF_HFE": "effort",
        "LF_KFE": "effort",
        "LH_HAA": "effort",
        "LH_HFE": "effort",
        "LH_KFE": "effort",
        "RF_HAA": "effort",
        "RF_HFE": "effort",
        "RF_KFE": "effort",
        "RH_HAA": "effort",
        "RH_HFE": "effort",
        "RH_KFE": "effort",
    }