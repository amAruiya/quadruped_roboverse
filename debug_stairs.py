
import numpy as np
import sys
import os

# Mock configs and utils
class Params:
    def __init__(self, t_type):
        self.terrain_type = t_type
        self.step_height = 0.2
        self.step_depth = 0.3 # 3 pixels
        self.platform_size = 1.0

# Add curr dir to path to import MyRobot
sys.path.append("/home/ubuntu/RoboVerse")

from MyRobot.terrain.algorithms.stairs import PyramidStairsAlgorithm
from MyRobot.terrain.injectors.base import TerrainInjector
from MyRobot.configs.task_cfg import TerrainCfg

def test_stairs():
    algo = PyramidStairsAlgorithm()
    
    print("--- Testing Stairs Down ---")
    p_down = Params("stairs_down")
    # Simulate config parser output: step_height is absolute
    p_down.step_height = 0.2
    
    hf_down = algo.generate((20, 20), p_down, 0.1, 0.005, 0)
    print("Down Min:", np.min(hf_down), "Max:", np.max(hf_down))
    print("Corner:", hf_down[0,0], "Center:", hf_down[10,10])
    
    print("\n--- Testing Stairs Up ---")
    p_up = Params("stairs_up")
    p_up.step_height = 0.2
    
    hf_up = algo.generate((20, 20), p_up, 0.1, 0.005, 0)
    print("Up Min:", np.min(hf_up), "Max:", np.max(hf_up))
    print("Corner:", hf_up[0,0], "Center:", hf_up[10,10])

test_stairs()
