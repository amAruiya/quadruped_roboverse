"""rsl_rl 环境适配器。

该模块将 MyRobot 的 BaseLocomotionTask 适配为 rsl_rl 所需的 VecEnv 接口。
支持 Observation Groups (Teacher-Student 架构)。
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from MyRobot.tasks.base_task import BaseLocomotionTask
from MyRobot.configs.train_cfg import TrainCfg


class RslRlVecEnvWrapper(VecEnv):
    """MyRobot 到 rsl_rl 的环境适配器。"""

    def __init__(self, task: BaseLocomotionTask, train_cfg: TrainCfg):
        """初始化适配器。
        
        Args:
            task: MyRobot 任务实例
            train_cfg: 训练配置 (用于解析 obs_groups)
        """
        self.task = task
        self.train_cfg = train_cfg
        
        # 必须属性
        self.num_envs = task.num_envs
        self.num_actions = task.num_actions
        # num_obs 和 num_privileged_obs 在 rsl_rl 中通常指主要观测维度
        # 在使用 obs_groups 时，这些属性可能主要用于兼容旧代码或 logging
        self.num_obs = task.num_obs 
        
        # 如果 Task 支持特权观测，获取其维度
        # 这里假设 Task.num_privileged_obs 存在，如果不存在则默认为 num_obs
        self.num_privileged_obs = getattr(task, "num_privileged_obs", self.num_obs)
        
        self.device = task.device
        self.max_episode_length = task.max_episode_length

        # 缓存当前的观测
        self._obs_dict = None
        
        # 初始化观测
        self.reset()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """执行一步。"""
        # 调用 Task 的 step
        # MyRobot Task 返回: obs, reward, reset_buf, timeout_buf, extras
        obs, rewards, resets, timeouts, extras = self.task.step(actions)
        
        # 合并 done
        dones = resets | timeouts
        
        # 构建 obs_dict (根据配置的分组)
        # 这里我们假设 Task 可以在 extras 中提供 'privileged_obs'
        # 或者我们简单地将 obs 同时作为 policy 和 critic 的输入 (如果没有特权信息)
        
        self._obs_dict = self._process_obs(obs, extras)

        return self._obs_dict, rewards, dones, extras

    def get_observations(self) -> TensorDict:
        """获取当前观测。"""
        # 如果没有被 step 调用初始化过，则手动获取一次
        if self._obs_dict is None:
            # 通过 task 的当前状态构建
             # 注意：BaseLocomotionTask 没有直接获取当前 obs 无需 step 的公开方法，
             # 但我们可以调用 reset 时的逻辑或手动构建。
             # 为简单起见，我们假设 reset 已经被调用过，或者在第一时间调用 reset
             pass
        return self._obs_dict

    def reset(self) -> tuple[TensorDict, dict]:
        """重置所有环境。"""
        obs, extras = self.task.reset()
        self._obs_dict = self._process_obs(obs, extras)
        return self._obs_dict, extras

    def _process_obs(self, obs: torch.Tensor, extras: dict) -> TensorDict:
        """处理观测，构建 obs_groups 字典。"""
        obs_dict = {}
        
        # 1. 基础观测 (Policy/Student)
        obs_dict["policy"] = obs
        
        # 2. 评论家/教师观测 (Critic/Teacher)
        # 优先从 extras 中获取特权观测
        if "privileged_obs" in extras:
            obs_dict["critic"] = extras["privileged_obs"]
        elif hasattr(self.task, "privileged_obs_buf") and self.task.privileged_obs_buf is not None:
             obs_dict["critic"] = self.task.privileged_obs_buf
        else:
            # 如果没有特权观测，Critic 使用同样的 obs
            obs_dict["critic"] = obs
            
        # 可以在此扩展更多组，例如 'history', 'depth' 等
        
        # 转换为 TensorDict, batch_size=[num_envs]
        return TensorDict(obs_dict, batch_size=[self.num_envs])
    
    # rsl_rl 要求的其他属性/方法兼容
    @property
    def cfg(self): 
        # rsl_rl 有时访问 env.cfg
        return self.task.cfg

    def __getattr__(self, name: str):
        """将未找到的属性委托给底层 task（如 episode_length_buf 等）。"""
        if "task" in self.__dict__:
            return getattr(self.__dict__["task"], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        """对 task 中已有的缓冲区属性直接写入 task，其余写入 wrapper 自身。"""
        _task_buf_attrs = {
            "episode_length_buf",
        }
        if name in _task_buf_attrs and "task" in self.__dict__:
            setattr(self.__dict__["task"], name, value)
        else:
            super().__setattr__(name, value)
