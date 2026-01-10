"""测试 Leap 任务的仿真渲染。

该脚本用于验证 LeapTask 类的初始化和基本运行。
"""

from __future__ import annotations

import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

try:
    import isaacgym
except ImportError as e:
    pass


import torch
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from MyRobot.configs.leap_cfg import LeapTaskCfg
from MyRobot.tasks.leap_task import LeapTask
from MyRobot.utils.helper import get_args, update_task_cfg_from_args


def test_leap_task(args):
    """测试 Leap 任务。
    
    Args:
        args: 命令行参数
    """
    log.info("="*80)
    log.info("开始测试 Leap 任务")
    log.info("="*80)
    
    # 1. 加载配置
    log.info("1. 加载任务配置...")
    task_cfg = LeapTaskCfg()
    
    # 2. 根据命令行参数更新配置
    task_cfg = update_task_cfg_from_args(task_cfg, args)
    
    log.info(f"   - 仿真器: {task_cfg.simulator}")
    log.info(f"   - 环境数量: {task_cfg.env.num_envs}")
    log.info(f"   - 无头模式: {task_cfg.headless}")
    log.info(f"   - 控制类型: {task_cfg.control.control_type}")
    log.info(f"   - 控制频率: {1.0 / (task_cfg.sim.dt * task_cfg.sim.decimation):.1f} Hz")
    
    # 3. 创建任务环境
    log.info("2. 创建任务环境...")
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"   - 使用设备: {device}")
    
    try:
        env = LeapTask(cfg=task_cfg, device=device)
        log.info("   ✓ 任务环境创建成功")
    except Exception as e:
        log.error(f"   ✗ 任务环境创建失败: {e}")
        raise
    
    # 4. 打印环境信息
    log.info("3. 环境信息:")
    log.info(f"   - 观测维度: {env.num_obs}")
    log.info(f"   - 动作维度: {env.num_actions}")
    log.info(f"   - 关节数量: {env.num_dof}")
    log.info(f"   - 刚体数量: {env.num_bodies}")
    log.info(f"   - 足部索引: {env.feet_indices}")
    
    # 5. 重置环境
    log.info("4. 重置环境...")
    try:
        obs, info = env.reset()
        log.info(f"   ✓ 环境重置成功")
        log.info(f"   - 观测形状: {obs.shape}")
    except Exception as e:
        log.error(f"   ✗ 环境重置失败: {e}")
        raise
    
    # 6. 运行测试步
    log.info(f"5. 运行测试 (最多 {100} 步)...")
    actions = torch.zeros(env.num_envs, env.num_actions, device=device)
    
    success_steps = 0
    try:
        for step in range(100):
            obs, reward, terminated, truncated, info = env.step(actions)
            success_steps += 1
            
            if step % 100 == 0:
                log.info(
                    f"   Step {step:4d}: "
                    f"reward={reward.mean():.3f}, "
                    f"terminated={terminated.sum().item()}/{env.num_envs}, "
                    f"truncated={truncated.sum().item()}/{env.num_envs}"
                )
            
            # 检查是否有环境需要重置
            if terminated.any() or truncated.any():
                reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                if args.debug:
                    log.debug(f"   - 重置环境: {reset_ids.tolist()}")
        
        log.info(f"   ✓ 成功完成 {success_steps} 步测试")
        
    except KeyboardInterrupt:
        log.warning(f"   用户中断 (已完成 {success_steps} 步)")
    except Exception as e:
        log.error(f"   ✗ 测试失败 (第 {success_steps} 步): {e}")
        raise
    finally:
        # 7. 关闭环境
        log.info("6. 清理资源...")
        env.close()
        log.info("   ✓ 环境已关闭")
    
    log.info("="*80)
    log.info("测试完成!")
    log.info("="*80)


def main():
    """主函数。"""
    args = get_args()
    
    # 设置日志级别
    if args.debug:
        log.remove()
        log.add(
            RichHandler(),
            format="{message}",
            level="DEBUG"
        )
    
    # 运行测试
    test_leap_task(args)


if __name__ == "__main__":
    main()