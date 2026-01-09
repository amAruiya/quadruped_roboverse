"""In-step 回调函数。

在 decimation 循环内，每次 simulate() 前调用。
"""

from __future__ import annotations

import torch


def push_robots(
    task,
    step_idx: int,
    push_interval: int = 100,
    push_velocity: tuple[float, float, float] = (1.0, 1.0, 0.0),
    **kwargs,
) -> None:
    """在特定步数施加外部推力（通过设置速度）。

    Args:
        task: BaseLocomotionTask 实例
        step_idx: 当前 decimation 步索引
        push_interval: 推力间隔（步数）
        push_velocity: 推力速度 (vx, vy, vz)
        **kwargs: 额外参数
    """
    if task.common_step_counter % push_interval == 0 and step_idx == 0:
        # 随机选择一部分环境进行推力
        num_push = max(1, task.num_envs // 10)
        push_ids = torch.randint(0, task.num_envs, (num_push,), device=task.device)

        # 设置速度扰动
        push_vel = torch.tensor(push_velocity, device=task.device)
        task.root_states[push_ids, 7:10] += push_vel

def elastic_force(task) -> None:
    # TODO: 实现弹性力逻辑
    pass