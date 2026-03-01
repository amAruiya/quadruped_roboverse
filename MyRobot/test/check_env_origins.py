"""检查 IsaacGym env origin 坐标分布。"""
import sys
sys.path.insert(0, '/home/ubuntu/RoboVerse')

from isaacgym import gymapi, gymtorch, gymutil
import torch

from MyRobot.configs.leap_cfg import LeapTaskCfg
from MyRobot.utils.helper import task_cfg_to_scenario
from metasim.sim import get_handler

cfg = LeapTaskCfg()
cfg.env.num_envs = 16

scenario = task_cfg_to_scenario(cfg)
handler = get_handler(scenario)
handler.launch()

print(f"\nenv_spacing = {scenario.env_spacing}")
print(f"num_envs = {handler.num_envs}")
print(f"\n=== handler._env_origin (lower-left corners) ===")
for i, orig in enumerate(handler._env_origin):
    print(f"  env {i:2d}: x={orig[0]:.2f}, y={orig[1]:.2f}, z={orig[2]:.2f}")

# 计算 env 中心
spacing = scenario.env_spacing
print(f"\n=== env centers (origin + spacing = {spacing}) ===")
for i, orig in enumerate(handler._env_origin):
    cx = orig[0] + spacing
    cy = orig[1] + spacing
    print(f"  env {i:2d}: center_x={cx:.2f}, center_y={cy:.2f}")
