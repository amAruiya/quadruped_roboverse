"""Post-step 回调函数。

在 simulate() 后，计算奖励前调用。
"""

from __future__ import annotations

import torch

from metasim.types import TensorState


def push_by_setting_velocity(
    task,
    env_states: TensorState,
    interval_range_s: tuple | float = 5.0,
    velocity_range: list[list] = [[0, 0, 0], [0, 0, 0]],
    **kwargs,
) -> None:
    """周期性地随机推动机器人（通过设置根速度）。

    Args:
        task: BaseLocomotionTask 实例
        env_states: 当前环境状态
        interval_range_s: 推力间隔时间范围（秒），可以是 float 或 (min, max)
        velocity_range: 速度范围 [[vx_min, vy_min, vz_min], [vx_max, vy_max, vz_max]]
        **kwargs: 额外参数
    """
    # 初始化推力计时器
    if not hasattr(task, "_push_timer"):
        task._push_timer = torch.zeros(task.num_envs, device=task.device)
        task._push_interval = torch.zeros(task.num_envs, device=task.device)

        # 设置初始间隔
        if isinstance(interval_range_s, (list, tuple)):
            interval_min, interval_max = interval_range_s
        else:
            interval_min = interval_max = interval_range_s

        task._push_interval[:] = torch.rand(task.num_envs, device=task.device) * (
            interval_max - interval_min
        ) + interval_min

    # 更新计时器
    task._push_timer += task.dt

    # 找出需要推力的环境
    push_ids = (task._push_timer >= task._push_interval).nonzero(as_tuple=False).flatten()

    if len(push_ids) > 0:
        # 生成随机推力速度
        vel_min = torch.tensor(velocity_range[0], device=task.device)
        vel_max = torch.tensor(velocity_range[1], device=task.device)

        random_vel = torch.rand(len(push_ids), 3, device=task.device) * (vel_max - vel_min) + vel_min

        # 应用推力（叠加到当前速度）
        task.root_states[push_ids, 7:10] += random_vel

        # 重置计时器和重新随机间隔
        task._push_timer[push_ids] = 0.0

        if isinstance(interval_range_s, (list, tuple)):
            interval_min, interval_max = interval_range_s
        else:
            interval_min = interval_max = interval_range_s

        task._push_interval[push_ids] = torch.rand(len(push_ids), device=task.device) * (
            interval_max - interval_min
        ) + interval_min


def terrain_curriculum(
    task,
    env_states: TensorState,
    **kwargs,
) -> None:
    """地形课程学习（预留）。

    Args:
        task: BaseLocomotionTask 实例
        env_states: 当前环境状态
        **kwargs: 额外参数
    """
    # TODO: 实现地形课程学习逻辑
    pass


def command_curriculum(
    task,
    env_states: TensorState,
    **kwargs,
) -> None:
    """命令课程学习（预留）。

    Args:
        task: BaseLocomotionTask 实例
        env_states: 当前环境状态
        **kwargs: 额外参数
    """
    # TODO: 根据性能调整命令范围
    pass