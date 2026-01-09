"""辅助工具函数。

该模块提供任务配置与场景配置之间的转换函数。
"""

from __future__ import annotations

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