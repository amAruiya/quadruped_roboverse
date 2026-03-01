"""MyRobot 训练脚本。

该脚本用于训练 Leap 机器人的 Teacher 策略 (PPO)。
它支持 MyRobot 的配置系统，并将其适配到 rsl_rl 的训练流程中。

使用示例：
    # 训练 Teacher 策略 (默认配置)
    python MyRobot/scripts/train.py --task leap --headless

    # 使用特定实验名称
    python MyRobot/scripts/train.py --task leap --headless --experiment_name my_experiment
"""

import argparse
import os
import sys
from datetime import datetime

# 确保 workspace 根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# IsaacGym 必须在 torch 之前导入
try:
    from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
except ImportError:
    pass

import torch
import tyro

# 导入 MyRobot 模块
from MyRobot.configs.leap_cfg import LeapTaskCfg
from MyRobot.configs.train_cfg import TrainCfg
from MyRobot.runners.on_policy_runner import OnPolicyRunner
from MyRobot.tasks.env_wrapper import RslRlVecEnvWrapper
from MyRobot.tasks.leap_task import LeapTask
from MyRobot.utils.helper import set_seed

# 注册任务映射
TASK_MAP = {
    "leap": (LeapTask, LeapTaskCfg),
}

def train(args):
    """训练函数。"""
    
    # 1. 准备配置
    # -------------------------------------------------------------------------
    # 获取任务类和配置类
    if args.task not in TASK_MAP:
        raise ValueError(f"未知的任务: {args.task}. 可用任务: {list(TASK_MAP.keys())}")
    
    task_class, task_cfg_class = TASK_MAP[args.task]
    
    # 解析 CLI 参数到配置对象
    # 这里我们允许 CLI 参数覆盖默认配置
    # 注意：这里简化处理，实际应该合并 args 到 config
    
    # 实例化配置
    task_cfg = task_cfg_class()
    train_cfg = TrainCfg()

    # 应用 CLI 参数覆盖 (简单示例)
    task_cfg.env.num_envs = args.num_envs
    task_cfg.headless = args.headless
    if args.experiment_name:
        train_cfg.runner.experiment_name = args.experiment_name
    
    # 设置设备
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("logs", train_cfg.runner.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # 设置随机种子
    set_seed(args.seed)

    # 2. 初始化环境
    # -------------------------------------------------------------------------
    print(f"正在初始化任务: {args.task} ...")
    env = task_class(cfg=task_cfg, device=device)
    
    # 包装环境以适配 rsl_rl
    env_wrapper = RslRlVecEnvWrapper(env, train_cfg)

    # 3. 初始化 Runner
    # -------------------------------------------------------------------------
    print(f"正在初始化 Runner (Log dir: {log_dir})...")
    runner = OnPolicyRunner(env_wrapper, train_cfg, log_dir=log_dir, device=device)

    # 4. 开始训练
    # -------------------------------------------------------------------------
    print("开始训练...")
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    print("训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyRobot 训练脚本")
    
    parser.add_argument("--task", type=str, default="leap", help="任务名称 (例如: leap)")
    parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数量")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="无头模式运行仿真")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cpu/cuda:0)")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")

    args = parser.parse_args()
    
    train(args)
