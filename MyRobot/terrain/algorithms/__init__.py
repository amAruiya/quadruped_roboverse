"""地形算法模块。"""

from .base import TerrainAlgorithm
from .basic import FlatAlgorithm
from .slope import PyramidSlopeAlgorithm, RoughSlopeAlgorithm
from .stairs import PyramidStairsAlgorithm
from .obstacles import DiscreteObstaclesAlgorithm
from .special import GapAlgorithm, PitAlgorithm, SteppingStonesAlgorithm

# 算法注册表
ALGORITHM_REGISTRY = {
    "flat": FlatAlgorithm(),
    "slope": PyramidSlopeAlgorithm(),
    "slope_rough": RoughSlopeAlgorithm(),
    "stairs_up": PyramidStairsAlgorithm(),
    "stairs_down": PyramidStairsAlgorithm(),
    "discrete": DiscreteObstaclesAlgorithm(),
    "stepping_stones": SteppingStonesAlgorithm(),
    "gap": GapAlgorithm(),
    "pit": PitAlgorithm(),
}

__all__ = [
    "TerrainAlgorithm",
    "ALGORITHM_REGISTRY",
    "FlatAlgorithm",
    "PyramidSlopeAlgorithm",
    "RoughSlopeAlgorithm",
    "PyramidStairsAlgorithm",
    "DiscreteObstaclesAlgorithm",
    "SteppingStonesAlgorithm",
    "GapAlgorithm",
    "PitAlgorithm",
]