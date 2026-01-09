"""Pre-step 回调函数。

在 step() 开始时，动作处理前调用。
"""

from __future__ import annotations

import torch


def action_clip(task, actions: torch.Tensor, clip_range: float = 100.0, **kwargs) -> torch.Tensor:
    """裁剪动作到指定范围。

    Args:
        task: BaseLocomotionTask 实例
        actions: 原始动作 (num_envs, num_actions)
        clip_range: 裁剪范围 [-clip_range, clip_range]
        **kwargs: 额外参数

    Returns:
        裁剪后的动作
    """
    return torch.clamp(actions, -clip_range, clip_range)


def action_smoothing(
    task,
    actions: torch.Tensor,
    alpha: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """对动作进行指数平滑。

    Args:
        task: BaseLocomotionTask 实例
        actions: 当前动作 (num_envs, num_actions)
        alpha: 平滑系数 [0, 1]，0 表示完全使用上一步动作，1 表示完全使用当前动作
        **kwargs: 额外参数

    Returns:
        平滑后的动作
    """
    if not hasattr(task, "last_actions"):
        return actions

    smoothed_actions = alpha * actions + (1 - alpha) * task.last_actions
    return smoothed_actions