from __future__ import annotations

try:
    import isaacgym
except ImportError:
    pass

from collections import deque

import numpy as np
import torch

from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler
from loguru import logger as log

try:
    import mujoco
except ImportError:
    pass


class ContactForces(BaseQueryType):
    """Optional query to fetch per-body net contact forces for each robot.

    - For IsaacGym: uses the native net-contact tensor and maps it per-robot in handler indexing order.
    - For IsaacSim: uses contact sensor to get net forces in world frame.
    - For MuJoCo: computes net forces from contact data.
    - For Genesis: uses get_links_net_contact_force() API.
    """

    def __init__(self, history_length: int = 3):
        super().__init__()
        self.history_length = history_length
        self._current_contact_force = None
        self._contact_forces_queue = deque(maxlen=history_length)

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        """Bind the simulator handler and pre-compute per-robot indexing."""
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        
        if self.simulator in ["isaacgym", "mujoco"]:
            self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name)
        elif self.simulator == "isaacsim":
            sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
            self.body_ids_reindex = torch.tensor(
                [self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names],
                dtype=torch.int,
                device=self.handler.device,
            )
        elif self.simulator == "genesis":
            # self.body_ids_reindex = None
            self.body_ids_reindex = self.handler._get_body_ids_reindex(self.robots[0].name)

            log.debug(
                f"Contact force body reindex:\n"
                f"  Shape: {self.body_ids_reindex.shape}\n"
                f"  Values: {self.body_ids_reindex.tolist()}"
            )
        else:
            raise NotImplementedError
        self.initialize()
        self.__call__()

    def initialize(self):
        """Warm-start the queue with `history_length` entries."""
        
        for _ in range(self.history_length):
            if self.simulator == "isaacgym":
                self._current_contact_force = isaacgym.gymtorch.wrap_tensor(
                    self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim)
                )
            elif self.simulator == "isaacsim":
                self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
            elif self.simulator == "mujoco":
                self._current_contact_force = self._get_contact_forces_mujoco()
            elif self.simulator == "genesis":
                self._current_contact_force = self._get_contact_forces_genesis()
            else:
                raise NotImplementedError
            
            self._contact_forces_queue.append(
                self._current_contact_force.clone().view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
            )

    def _get_contact_forces_mujoco(self) -> torch.Tensor:
        """Compute net contact forces on each body.

        Returns:
            torch.Tensor: shape (nbody, 3), contact forces for each body
        """
        nbody = self.handler.physics.model.nbody
        contact_forces = torch.zeros((nbody, 3), device=self.handler.device)

        for i in range(self.handler.physics.data.ncon):
            contact = self.handler.physics.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
            f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

            body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
            body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

            contact_forces[body1] += f_contact
            contact_forces[body2] -= f_contact

        return contact_forces

    def _get_contact_forces_genesis(self) -> torch.Tensor:
        """Get net contact forces for Genesis simulator.
        
        Uses the native get_links_net_contact_force() API which returns
        net force applied on each link due to direct external contacts.
        
        Returns:
            torch.Tensor: shape (n_envs, n_links, 3), contact forces for each link
        """
        # 获取机器人实体
        robot_entity = self.handler.robot_inst
        
        # 调用 Genesis API 获取接触力
        # 返回形状: (n_links, 3) 或 (n_envs, n_links, 3)
        contact_forces = robot_entity.get_links_net_contact_force()
        
        # 确保返回形状为 (n_envs, n_links, 3)
        if contact_forces.ndim == 2:
            # 单环境情况: (n_links, 3) -> (1, n_links, 3)
            contact_forces = contact_forces.unsqueeze(0).expand(self.num_envs, -1, -1)
        
        return contact_forces

    def __call__(self):
        """Fetch the newest net contact forces and update the queue."""
        if self.simulator == "isaacgym":
            self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        elif self.simulator == "isaacsim":
            self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self._current_contact_force = self._get_contact_forces_mujoco()
        elif self.simulator == "genesis":
            self._current_contact_force = self._get_contact_forces_genesis()
        else:
            raise NotImplementedError
        
        self._contact_forces_queue.append(
            self._current_contact_force.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
        )
        return {self.robots[0].name: self}

    @property
    def contact_forces_history(self) -> torch.Tensor:
        """Return stacked history as (num_envs, history_length, num_bodies, 3)."""
        return torch.stack(list(self._contact_forces_queue), dim=1)  # (num_envs, history_length, num_bodies, 3)

    @property
    def contact_forces(self) -> torch.Tensor:
        """Return the latest contact forces snapshot."""
        return self._contact_forces_queue[-1]
