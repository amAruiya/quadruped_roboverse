"""仿真器注入器模块。"""

from .base import TerrainInjector
from .isaacgym import IsaacGymInjector
from .mujoco import MuJoCoInjector
from .genesis import GenesisInjector
from .isaacsim import IsaacSimInjector


def get_injector(simulator: str) -> TerrainInjector:
    """根据仿真器类型返回地形注入器。
    
    Args:
        simulator: 仿真器类型
        
    Returns:
        TerrainInjector: 对应的注入器实例
        
    Raises:
        NotImplementedError: 不支持的仿真器类型
    """
    registry = {
        "isaacgym": IsaacGymInjector(),
        "mujoco": MuJoCoInjector(),
        "genesis": GenesisInjector(),
        "isaacsim": IsaacSimInjector(),
        "isaaclab": IsaacSimInjector(),  # 复用 IsaacSim
    }
    
    if simulator not in registry:
        raise NotImplementedError(f"Unsupported simulator: {simulator}")
    
    return registry[simulator]


__all__ = [
    "TerrainInjector",
    "get_injector",
    "IsaacGymInjector",
    "MuJoCoInjector",
    "GenesisInjector",
    "IsaacSimInjector",
]