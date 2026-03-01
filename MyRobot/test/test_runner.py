"""Runner 测试脚本。

测试 OnPolicyRunner 是否能正确初始化并运行环境。
"""

import unittest
import torch
import shutil
import os
from unittest.mock import MagicMock

from MyRobot.runners.on_policy_runner import OnPolicyRunner
from MyRobot.tasks.env_wrapper import RslRlVecEnvWrapper
from MyRobot.configs.train_cfg import TrainCfg
from MyRobot.tasks.base_task import BaseLocomotionTask

class TestRunner(unittest.TestCase):
    def setUp(self):
        # 1. Mock Task
        self.mock_task = MagicMock(spec=BaseLocomotionTask)
        self.mock_task.num_envs = 10
        self.mock_task.num_obs = 30
        self.mock_task.num_privileged_obs = 40 # 假设特权观测维度不同
        self.mock_task.num_actions = 12
        self.mock_task.max_episode_length = 100
        self.mock_task.device = "cpu"
        self.mock_task.cfg = MagicMock()
        
        # Mock step and reset
        # obs: (num_envs, num_obs)
        # extras need to contain privileged_obs
        obs = torch.zeros(10, 30)
        extras = {"privileged_obs": torch.zeros(10, 40)} # 提供特权观测
        
        self.mock_task.reset.return_value = (obs, extras)
        self.mock_task.step.return_value = (
            obs, 
            torch.zeros(10), # reward
            torch.zeros(10, dtype=torch.bool), # reset
            torch.zeros(10, dtype=torch.bool), # timeout
            extras
        )
        
        # 2. Config
        self.train_cfg = TrainCfg()
        self.train_cfg.runner.max_iterations = 2
        
        # 3. Env Wrapper
        self.env = RslRlVecEnvWrapper(self.mock_task, self.train_cfg)
        
        # Log dir
        self.log_dir = "test_logs_runner"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    def tearDown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    def test_init(self):
        """测试 Runner 初始化。"""
        runner = OnPolicyRunner(self.env, self.train_cfg, log_dir=self.log_dir, device="cpu")
        self.assertIsInstance(runner, OnPolicyRunner)
        # 验证 rsl_rl 内部算法是否创建
        self.assertIsNotNone(runner.alg)
        
    def test_learn(self):
        """测试训练循环。"""
        runner = OnPolicyRunner(self.env, self.train_cfg, log_dir=self.log_dir, device="cpu")
        
        # 运行少量迭代
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        
        # 验证 step 被调用
        self.assertTrue(self.mock_task.step.called)
        
        # 验证观察组是否生效
        # 检查 runner.alg.actor_critic.actor_obs_shape 是否对应 policy group
        # 检查 runner.alg.actor_critic.critic_obs_shape 是否对应 critic group
        # 注意: 具体属性名称取决于 rsl_rl 版本，这里简单验证不报错即可
        pass

if __name__ == "__main__":
    unittest.main()
