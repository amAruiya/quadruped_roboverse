"""Reset 回调函数。

在环境重置时调用。
"""

from __future__ import annotations

import torch
from loguru import logger as log

from metasim.utils.math import quat_from_euler_xyz, quat_mul


def random_root_state(
    task,
    env_ids: torch.Tensor,
    pose_range: list[list] = [[0] * 6, [0] * 6],
    velocity_range: list[list] = [[0] * 6, [0] * 6],
    **kwargs,
) -> None:
    """随机化根状态（位置、姿态、速度）。

    Args:
        task: BaseLocomotionTask 实例
        env_ids: 要重置的环境索引 (N,)
        pose_range: 位姿随机范围 [[x_min, y_min, z_min, roll_min, pitch_min, yaw_min],
                                   [x_max, y_max, z_max, roll_max, pitch_max, yaw_max]]
        velocity_range: 速度随机范围 [[vx_min, vy_min, vz_min, wx_min, wy_min, wz_min],
                                      [vx_max, vy_max, vz_max, wx_max, wy_max, wz_max]]
        **kwargs: 额外参数
    """
    num_resets = len(env_ids)

    # 解析范围
    pose_min = torch.tensor(pose_range[0], device=task.device)
    pose_max = torch.tensor(pose_range[1], device=task.device)
    vel_min = torch.tensor(velocity_range[0], device=task.device)
    vel_max = torch.tensor(velocity_range[1], device=task.device)

    # 随机位置偏移
    pos_offset = torch.rand(num_resets, 3, device=task.device) * (pose_max[:3] - pose_min[:3]) + pose_min[:3]
    task.root_states[env_ids, :3] += pos_offset

    # 随机姿态偏移（欧拉角 -> 四元数）
    euler_offset = (
        torch.rand(num_resets, 3, device=task.device) * (pose_max[3:6] - pose_min[3:6]) + pose_min[3:6]
    )
    quat_offset = quat_from_euler_xyz(euler_offset[:, 0], euler_offset[:, 1], euler_offset[:, 2])

    # 组合原始四元数和偏移
    task.root_states[env_ids, 3:7] = quat_mul(task.root_states[env_ids, 3:7], quat_offset)

    # 随机线速度
    lin_vel = torch.rand(num_resets, 3, device=task.device) * (vel_max[:3] - vel_min[:3]) + vel_min[:3]
    task.root_states[env_ids, 7:10] = lin_vel

    # 随机角速度
    ang_vel = torch.rand(num_resets, 3, device=task.device) * (vel_max[3:6] - vel_min[3:6]) + vel_min[3:6]
    task.root_states[env_ids, 10:13] = ang_vel


def reset_joints_by_scale(
    task,
    env_ids: torch.Tensor,
    position_range: list | tuple = (1.0, 1.0),
    velocity_range: list | tuple = (1.0, 1.0),
    **kwargs,
) -> None:
    """按缩放因子随机化关节状态。

    Args:
        task: BaseLocomotionTask 实例
        env_ids: 要重置的环境索引 (N,)
        position_range: 位置缩放范围 (scale_min, scale_max)，应用于默认关节位置
        velocity_range: 速度缩放范围 (scale_min, scale_max)
        **kwargs: 额外参数
    """
    num_resets = len(env_ids)

    # 随机位置缩放
    pos_scale_min, pos_scale_max = position_range
    pos_scale = (
        torch.rand(num_resets, task.num_dof, device=task.device) * (pos_scale_max - pos_scale_min)
        + pos_scale_min
    )
    task.dof_pos[env_ids] = task.default_dof_pos * pos_scale

    # 裁剪到关节限制（如果配置了）
    if hasattr(task, "dof_pos_limits"):
        task.dof_pos[env_ids] = torch.clamp(
            task.dof_pos[env_ids],
            task.dof_pos_limits[:, 0],
            task.dof_pos_limits[:, 1],
        )

    # 随机速度
    vel_scale_min, vel_scale_max = velocity_range
    vel_scale = (
        torch.rand(num_resets, task.num_dof, device=task.device) * (vel_scale_max - vel_scale_min)
        + vel_scale_min
    )

    # 生成随机速度（基于速度限制）
    if hasattr(task, "dof_vel_limits"):
        max_vel = task.dof_vel_limits
    else:
        max_vel = torch.ones(task.num_dof, device=task.device) * 10.0  # 默认限制

    task.dof_vel[env_ids] = (torch.rand(num_resets, task.num_dof, device=task.device) * 2 - 1) * max_vel * vel_scale