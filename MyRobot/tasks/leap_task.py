"""Leap 四足机器人运动任务实现。

基于 example_RMA/envs/Leap/Leap.py 迁移而来，
使用 metasim handler 实现后端无关的仿真。
"""

from __future__ import annotations

import torch
from loguru import logger as log

from MyRobot.configs.leap_cfg import LeapTaskCfg
from MyRobot.tasks.base_task import BaseLocomotionTask


class LeapTask(BaseLocomotionTask):
    """Leap 四足机器人运动任务。

    Leap 是一款 12 自由度四足机器人，每条腿包含：
    - HAA (Hip Abduction/Adduction): 髋关节外展/内收
    - HFE (Hip Flexion/Extension): 髋关节屈/伸
    - KFE (Knee Flexion/Extension): 膝关节屈/伸

    特点：
    - 支持地形课程学习（本阶段暂未启用）
    - 支持域随机化（本阶段暂未启用）
    - 包含 Leap 特有的奖励函数（leg_effort_symmetry, thigh_pos, hip_pos）

    Attributes:
        ema_leg_power: 腿部功率的指数移动平均，用于对称性奖励
    """

    def __init__(
        self,
        cfg: LeapTaskCfg | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """初始化 Leap 任务。

        Args:
            cfg: Leap 任务配置，如果为 None 则使用默认配置
            device: 计算设备
        """
        if cfg is None:
            cfg = LeapTaskCfg()

        super().__init__(cfg, device)

        log.info(f"LeapTask 初始化完成: {self.num_envs} 环境")

    # =========================================================================
    # 缓冲区初始化（扩展）
    # =========================================================================

    def _init_buffers(self) -> None:
        """初始化缓冲区，包括 Leap 特有的缓冲区。"""
        super()._init_buffers()

        # Leap 特有：腿部功率的指数移动平均
        # 用于计算 leg_effort_symmetry 奖励
        self.ema_leg_power = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )

        # 关节分组索引（4 条腿，每条腿 3 个关节）
        # 顺序：LF, LH, RF, RH（与 URDF 中的顺序一致）
        self._init_leg_indices()

        log.debug("Leap 特有缓冲区初始化完成")

    def _init_leg_indices(self) -> None:
        """初始化腿部关节索引。

        Leap 关节命名规则：
        - LF_HAA, LF_HFE, LF_KFE (左前腿)
        - LH_HAA, LH_HFE, LH_KFE (左后腿)
        - RF_HAA, RF_HFE, RF_KFE (右前腿)
        - RH_HAA, RH_HFE, RH_KFE (右后腿)
        """
        # 腿部前缀
        leg_prefixes = ["LF", "LH", "RF", "RH"]
        joint_suffixes = ["HAA", "HFE", "KFE"]

        # 构建每条腿的关节索引
        self.leg_indices = {}
        for leg in leg_prefixes:
            indices = []
            for suffix in joint_suffixes:
                joint_name = f"{leg}_{suffix}"
                if joint_name in self.dof_names:
                    idx = self.dof_names.index(joint_name)
                    indices.append(idx)
            if indices:
                self.leg_indices[leg] = torch.tensor(indices, device=self.device)

        # HAA 关节索引（用于 hip_pos 奖励）
        self.haa_indices = []
        for leg in leg_prefixes:
            joint_name = f"{leg}_HAA"
            if joint_name in self.dof_names:
                self.haa_indices.append(self.dof_names.index(joint_name))
        self.haa_indices = torch.tensor(self.haa_indices, device=self.device)

        # HFE 关节索引（用于 thigh_pos 奖励）
        self.hfe_indices = []
        for leg in leg_prefixes:
            joint_name = f"{leg}_HFE"
            if joint_name in self.dof_names:
                self.hfe_indices.append(self.dof_names.index(joint_name))
        self.hfe_indices = torch.tensor(self.hfe_indices, device=self.device)

        log.debug(f"腿部关节索引: {self.leg_indices}")
        log.debug(f"HAA 索引: {self.haa_indices}, HFE 索引: {self.hfe_indices}")

    # =========================================================================
    # 重置（扩展）
    # =========================================================================

    def _reset_idx(self, env_ids: list[int]) -> None:
        """重置指定环境，包括 Leap 特有的缓冲区。"""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 重置 Leap 特有缓冲区
        self.ema_leg_power[env_ids_tensor] = 0.0

    # =========================================================================
    # Leap 特有奖励函数
    # =========================================================================

    def _reward_leg_effort_symmetry(self) -> torch.Tensor:
        """腿部功率对称性奖励。

        计算四条腿的功率（扭矩 × 关节速度），
        使用指数移动平均平滑，然后计算方差作为惩罚。
        功率越对称（方差越小），奖励越高。

        参考：example_RMA/envs/Leap/Leap.py::_reward_leg_effort_symmetry
        """
        # 计算每个关节的功率
        joint_power = torch.abs(self.torques * self.dof_vel)

        # 重塑为 (num_envs, 4, 3)：4 条腿，每条腿 3 个关节
        num_legs = 4
        joints_per_leg = 3

        if self.num_dof == num_legs * joints_per_leg:
            power_per_leg_grouped = joint_power.view(self.num_envs, num_legs, joints_per_leg)
            # 每条腿的总功率
            current_leg_power = torch.sum(power_per_leg_grouped, dim=2)
        else:
            # 如果关节数不匹配，使用备选方案
            current_leg_power = torch.zeros(self.num_envs, num_legs, device=self.device)
            for i, (leg, indices) in enumerate(self.leg_indices.items()):
                if len(indices) > 0:
                    current_leg_power[:, i] = torch.sum(joint_power[:, indices], dim=1)

        # 指数移动平均
        ema_alpha = getattr(self.cfg.rewards, "ema_alpha", 0.4)
        self.ema_leg_power.mul_(1.0 - ema_alpha).add_(current_leg_power * ema_alpha)

        # 计算 EMA 功率的方差
        variance_of_ema_power = torch.var(self.ema_leg_power, dim=1)

        return variance_of_ema_power

    def _reward_hip_pos(self) -> torch.Tensor:
        """髋关节位置惩罚（HAA 关节）。

        惩罚 HAA 关节偏离默认位置的程度。

        参考：example_RMA/envs/Leap/Leap.py（隐含在 scales 中）
        """
        if len(self.haa_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        haa_pos = self.dof_pos[:, self.haa_indices]
        haa_default = self.default_dof_pos[:, self.haa_indices]

        return torch.sum(torch.square(haa_pos - haa_default), dim=1)

    def _reward_thigh_pos(self) -> torch.Tensor:
        """大腿位置惩罚（HFE 关节）。

        惩罚 HFE 关节偏离默认位置的程度。

        参考：example_RMA/envs/Leap/Leap.py::_reward_thigh_pos
        """
        if len(self.hfe_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        hfe_pos = self.dof_pos[:, self.hfe_indices]
        hfe_default = self.default_dof_pos[:, self.hfe_indices]

        return torch.sum(torch.square(hfe_pos - hfe_default), dim=1)

    def _reward_feet_contact_forces(self) -> torch.Tensor:
        """足部接触力惩罚。

        惩罚过大的足部接触力。

        参考：example_RMA/envs/base/legged_robot.py::_reward_feet_contact_forces
        """
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        # 获取配置的最大接触力
        max_contact_force = getattr(self.cfg.rewards, "max_contact_force", 100.0)

        # 计算足部接触力
        feet_forces = self.contact_forces[:, self.feet_indices, :]
        feet_force_norms = torch.norm(feet_forces, dim=-1)

        # 超过阈值的部分进行惩罚
        exceeded = (feet_force_norms - max_contact_force).clamp(min=0.0)

        return torch.sum(exceeded, dim=1)

    # =========================================================================
    # 覆盖父类的奖励函数（如需特殊处理）
    # =========================================================================

    def _reward_dof_pos_limits(self) -> torch.Tensor:
        """关节位置限制惩罚。

        惩罚接近关节限位的情况。

        参考：example_RMA/envs/base/legged_robot.py::_reward_dof_pos_limits
        """
        # 获取软限位比例
        soft_dof_pos_limit = getattr(self.cfg.rewards, "soft_dof_pos_limit", 0.9)

        # TODO: 从 handler 获取关节限位
        # 目前使用简化实现
        out_of_limits = torch.zeros(self.num_envs, device=self.device)

        return out_of_limits

    def _reward_base_height(self) -> torch.Tensor:
        """基座高度奖励。

        奖励机器人保持在目标高度附近。

        参考：example_RMA/envs/base/legged_robot.py（rewards.base_height_target）
        """
        target_height = getattr(self.cfg.rewards, "base_height_target", 0.355)
        height_error = torch.square(self.base_pos[:, 2] - target_height)

        return height_error

    def _reward_stand_still(self) -> torch.Tensor:
        """静止惩罚。

        当命令速度接近零时，惩罚关节偏离默认位置。

        参考：example_RMA/envs/base/legged_robot.py::_reward_stand_still
        """
        # 检查命令是否接近零
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        standing = (cmd_norm < 0.1).float()

        # 关节偏离惩罚
        dof_deviation = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

        return dof_deviation * standing

    def _reward_stumble(self) -> torch.Tensor:
        """绊倒惩罚。

        惩罚足部撞击垂直表面（水平力远大于垂直力）。

        参考：example_RMA/envs/base/legged_robot.py::_reward_stumble
        """
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        # 获取足部接触力
        feet_forces = self.contact_forces[:, self.feet_indices, :]

        # 水平力
        horizontal_force = torch.norm(feet_forces[:, :, :2], dim=2)
        # 垂直力
        vertical_force = torch.abs(feet_forces[:, :, 2])

        # 如果水平力 > 5 倍垂直力，认为是撞击垂直表面
        stumble = horizontal_force > 5 * vertical_force

        return torch.any(stumble, dim=1).float()


