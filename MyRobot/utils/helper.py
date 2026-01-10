"""辅助工具函数。

该模块提供任务配置与场景配置之间的转换函数。
"""

from __future__ import annotations

import argparse
from typing import Any

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.setup_util import get_robot

from MyRobot.configs.task_cfg import BaseTaskCfg


def task_cfg_to_scenario(task_cfg: BaseTaskCfg) -> ScenarioCfg:
    """从任务配置构建场景配置。

    Args:
        task_cfg: 任务配置对象

    Returns:
        场景配置对象
    """
    # 解析机器人配置
    if isinstance(task_cfg.robots, str):
        # 字符串格式,使用 get_robot 解析
        robots = [get_robot(task_cfg.robots)]
    elif isinstance(task_cfg.robots, list):
        # 列表格式,逐个解析
        robots = []
        for robot in task_cfg.robots:
            if isinstance(robot, str):
                robots.append(get_robot(robot))
            else:
                robots.append(robot)
    else:
        # 单个 RobotCfg 对象
        robots = [task_cfg.robots]

    scenario = ScenarioCfg(
        # 机器人配置
        robots=robots,
        
        # 物体和场景
        objects=task_cfg.objects,
        scene=task_cfg.scene,
        
        # 相机和灯光
        cameras=task_cfg.cameras,
        lights=task_cfg.lights,
        
        # 仿真参数
        num_envs=task_cfg.env.num_envs,
        env_spacing=task_cfg.env.env_spacing,
        decimation=task_cfg.sim.decimation,
        gravity=task_cfg.sim.gravity,
        
        # 仿真器设置
        simulator=task_cfg.simulator,
        headless=task_cfg.headless,
        
        # 仿真参数详细配置
        sim_params=SimParamCfg(
            dt=task_cfg.sim.dt,
            substeps=getattr(task_cfg.sim, "substeps", 1),
        ),
        
        # 渲染配置
        render=getattr(task_cfg, "render", None),
    )
    
    return scenario


def get_args() -> argparse.Namespace:
    """获取命令行参数。
    包含：
    --task: 任务名称
    --sim: 仿真器类型
    --headless: 无头模式
    --num_envs: 并行环境数量
    --device: 计算设备
    --debug: 调试模式
    
    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(description="MyRobot Task Test")
    
    # 任务相关
    parser.add_argument(
        "--task",
        type=str,
        default="leap",
        help="任务名称"
    )
    
    # 仿真器相关
    parser.add_argument(
        "--sim",
        type=str,
        default="isaacgym",
        choices=["isaacgym", "isaacsim", "mujoco", "genesis"],
        help="仿真器类型"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="无头模式（不显示图形界面）"
    )
    
    # 环境相关
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="并行环境数量，None 表示使用配置文件中的值"
    )
    
    # 设备相关
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (cuda/cpu)，None 表示自动选择"
    )
    
    # 调试相关
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="启用调试模式"
    )

    args = parser.parse_args()
    return args


def update_task_cfg_from_args(
    task_cfg: BaseTaskCfg,
    args: argparse.Namespace
) -> BaseTaskCfg:
    """根据命令行参数更新任务配置。
    
    Args:
        task_cfg: 原始任务配置
        args: 命令行参数
        
    Returns:
        更新后的任务配置
    """
    import copy
    cfg = copy.deepcopy(task_cfg)
    
    # 更新环境数量
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    
    # 更新仿真器
    cfg.simulator = args.sim
    cfg.headless = args.headless
    
    return cfg