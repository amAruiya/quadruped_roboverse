"""Terminate 回调函数。

终止条件检查。
"""

from __future__ import annotations

import torch

from metasim.types import TensorState


def orientation_termination(
    task,
    env_states: TensorState,
    threshold: float = 0.5,
    **kwargs,
) -> torch.BoolTensor:
    """基于姿态的终止条件（倾覆检测）。

    Args:
        task: BaseLocomotionTask 实例
        env_states: 当前环境状态
        threshold: z 轴投影重力阈值，低于此值视为倾覆
        **kwargs: 额外参数

    Returns:
        终止标志 (num_envs,)
    """
    # 检查投影重力的 z 分量
    # projected_gravity[:, 2] 在正常站立时应接近 -1.0
    return torch.abs(task.projected_gravity[:, 2]) < threshold


def undesired_contact_termination(
    task,
    env_states: TensorState,
    contact_body_names: list[str] | None = None,
    force_threshold: float = 1.0,
    **kwargs,
) -> torch.BoolTensor:
    """基于不希望接触的终止条件。

    Args:
        task: BaseLocomotionTask 实例
        env_states: 当前环境状态
        contact_body_names: 不希望接触的 body 名称列表（支持正则表达式）
        force_threshold: 接触力阈值（N）
        **kwargs: 额外参数

    Returns:
        终止标志 (num_envs,)
    """
    if contact_body_names is None:
        # 使用配置中的终止接触列表
        if hasattr(task.cfg, "asset") and task.cfg.asset.terminate_after_contacts_on:
            contact_body_names = task.cfg.asset.terminate_after_contacts_on
        else:
            return torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)

    # 找到对应的 body 索引
    import re

    contact_indices = []
    for pattern in contact_body_names:
        for i, body_name in enumerate(task.body_names):
            if re.search(pattern, body_name):
                contact_indices.append(i)

    if len(contact_indices) == 0:
        return torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)

    contact_indices = torch.tensor(contact_indices, dtype=torch.long, device=task.device)

    # 检查接触力
    # contact_forces: (num_envs, num_bodies, 3)
    if not hasattr(task, "contact_forces") or task.contact_forces is None:
        return torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)

    contact_forces_magnitude = torch.norm(task.contact_forces[:, contact_indices, :], dim=-1)  # (num_envs, n_contacts)
    has_contact = (contact_forces_magnitude > force_threshold).any(dim=1)  # (num_envs,)

    return has_contact


def time_out_termination(
    task,
    env_states: TensorState,
    **kwargs,
) -> torch.BoolTensor:
    """基于时间的终止条件。

    Args:
        task: BaseLocomotionTask 实例
        env_states: 当前环境状态
        **kwargs: 额外参数

    Returns:
        终止标志 (num_envs,)
    """
    return task.episode_length_buf >= task.max_episode_length


def root_height_below_minimum(
    task,
    env_states: TensorState,
    min_height: float = 0.25,
    **kwargs,
) -> torch.BoolTensor:
    """基于根高度的终止条件。

    Args:
        task: BaseLocomotionTask 实例
        env_states: 当前环境状态
        min_height: 最小高度阈值（米）
        **kwargs: 额外参数

    Returns:
        终止标志 (num_envs,)
    """
    return task.base_pos[:, 2] < min_height