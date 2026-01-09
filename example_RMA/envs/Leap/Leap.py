from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Leap(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.ema_leg_power = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_leg_effort_symmetry(self):

        joint_power = torch.abs(self.torques * self.dof_vel)
        num_legs = 4
        joints_per_leg = 3
        power_per_leg_grouped = joint_power.view(self.num_envs, num_legs, joints_per_leg)
        current_leg_power = torch.sum(power_per_leg_grouped, dim=2)
        self.ema_leg_power.mul_(1.0 - self.cfg.rewards.ema_alpha).add_(current_leg_power * self.cfg.rewards.ema_alpha)
        variance_of_ema_power = torch.var(self.ema_leg_power, dim=1)
        return variance_of_ema_power
    
    def _reward_thigh_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [1,4,7,10]] - self.default_dof_pos[:, [1,4,7,10]]), dim=1)