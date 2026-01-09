from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.rl_task import RLTaskEnv
from metasim.types import Action, Reward, TensorState
from metasim.utils.state import list_state_to_tensor
from roboverse_learn.rl.unitree_rl.configs.cfg_base import BaseEnvCfg, CallbacksCfg
from MyRobot.configs.base_task_cfg import BaseTaskCfg

class BaseLocomotionTask(RLTaskEnv):
    """Base RLTaskEnv wrapper shared across My Robot."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        config: Any | BaseTaskCfg,
        device: str | torch.device | None = None,
    ) -> None:
        pass
