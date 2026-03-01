"""On-Policy Runner 实现。

封装 rsl_rl 的 OnPolicyRunner，支持 MyRobot 的配置系统和环境。
"""

from __future__ import annotations

import os
from typing import Any

from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner

from MyRobot.configs.train_cfg import TrainCfg
from MyRobot.tasks.env_wrapper import RslRlVecEnvWrapper


class OnPolicyRunner(RslOnPolicyRunner):
    """MyRobot 的 On-Policy Runner。
    
    该类主要负责：
    1. 接收 MyRobot 的 EnvWrapper 和 TrainCfg。
    2. 将 TrainCfg 转换为 rsl_rl 期望的字典格式。
    3. 初始化父类 RslOnPolicyRunner。
    """

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg: TrainCfg, log_dir: str | None = None, device: str = "cpu"):
        """初始化 Runner。
        
        Args:
            env: 包装后的环境
            train_cfg: 训练配置
            log_dir: 日志目录
            device: 设备
        """
        # 将 dataclass 配置转换为 dict，以兼容 rsl_rl
        train_cfg_dict = self._cfg_to_dict(train_cfg)
        
        super().__init__(env, train_cfg_dict, log_dir, device)

    def _cfg_to_dict(self, cfg: TrainCfg) -> dict[str, Any]:
        """将 TrainCfg 转换为字典结构。"""
        # 递归转换 dataclass
        # 这里手动构建 key 以确保匹配 rsl_rl 的期望结构
        
        policy_dict = {
            k: v for k, v in vars(cfg.policy).items() if not k.startswith("_")
        }
        # 将 policy_class_name 注入到 policy 配置中
        policy_dict["class_name"] = cfg.runner.policy_class_name

        runner_dict = {
            k: v for k, v in vars(cfg.runner).items() if not k.startswith("_")
        }
        # 确保 num_steps_per_env 存在
        if "num_steps_per_env" not in runner_dict:
            runner_dict["num_steps_per_env"] = cfg.runner.num_steps_per_env
            
        # rsl_rl 期望 runner 的配置参数直接在顶层字典中
        algorithm_dict = {
            k: v for k, v in vars(cfg.algorithm).items() if not k.startswith("_")
        }
        algorithm_dict["class_name"] = cfg.runner.algorithm_class_name

        train_cfg_dict = {
            "algorithm": algorithm_dict,
            "policy": policy_dict,
            "obs_groups": cfg.obs_groups,
        }
        # 将 runner 配置合并到顶层
        train_cfg_dict.update(runner_dict)
        
        return train_cfg_dict
    
    # 可以在此添加自定义方法，例如 save/load 的具体钩子，如果 rsl_rl 的不够用
    pass
