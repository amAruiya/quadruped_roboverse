"""MyRobot 训练配置。

该模块定义了基于 rsl_rl 的训练配置类。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from metasim.utils import configclass


@configclass
class PolicyCfg:
    """策略网络配置。"""
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"  # 'elu', 'relu', 'selu', 'crelu', 'lrelu', 'tanh', 'sigmoid'


@configclass
class AlgorithmCfg:
    """PPO 算法配置。"""
    # 核心算法参数
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4  # mini batch size = num_envs * nsteps / nminibatches
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"  # could be adaptive, fixed
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class RunnerCfg:
    """运行器配置。"""
    policy_class_name: str = "ActorCritic"  # 'ActorCritic', 'ActorCriticRecurrent', 'StudentTeacher'
    algorithm_class_name: str = "PPO"
    num_steps_per_env: int = 24  # per iteration
    max_iterations: int = 1500  # number of policy updates

    # logging
    save_interval: int = 50
    experiment_name: str = "test_myrobot"
    run_name: str = ""
    resume: bool = False
    load_run: str = ""  # -1 = last run
    checkpoint: int = -1  # -1 = last saved model
    resume_path: str | None = None  # updated from load_run and checkpoint


@configclass
class TrainCfg:
    """总训练配置。"""
    runner: RunnerCfg = RunnerCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    policy: PolicyCfg = PolicyCfg()
    
    # Observation Groups 用于 Teacher-Student 或 Actor-Critic 分离
    # 格式: {"group_name": ["obs_key1", "obs_key2"]}
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],     # Student / Actor 输入
        "critic": ["critic"],     # Critic / Teacher 输入 (通常包含特权信息)
    }
