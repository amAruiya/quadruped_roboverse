"""Leap 机器人 GUI 控制测试。

使用 GUI 发送目标角度，通过 PD 控制计算力矩。
"""

from __future__ import annotations

import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

try:
    import isaacgym
except ImportError:
    pass

import numpy as np
import FreeSimpleGUI as sg
import torch
from loguru import logger as log
from rich.logging import RichHandler

from MyRobot.configs.leap_cfg import LeapTaskCfg
from MyRobot.robots.leap_cfg import LeapCfg
from MyRobot.tasks.leap_task import LeapTask
from MyRobot.utils.helper import get_args, update_task_cfg_from_args

# 配置日志
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

# 关节顺序（与 URDF 一致）
JOINT_NAMES = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]


def create_gui(default_angles: dict[str, float]) -> sg.Window:
    """创建 GUI 窗口。
    
    Args:
        default_angles: 默认关节角度
        
    Returns:
        PySimpleGUI 窗口对象
    """
    # 关节限制（从 LeapCfg 获取）
    robot_cfg = LeapCfg()
    joint_limits = robot_cfg.joint_limits
    
    # 按腿分组
    legs = {
        "LF": ["LF_HAA", "LF_HFE", "LF_KFE"],
        "LH": ["LH_HAA", "LH_HFE", "LH_KFE"],
        "RF": ["RF_HAA", "RF_HFE", "RF_KFE"],
        "RH": ["RH_HAA", "RH_HFE", "RH_KFE"],
    }
    
    # 创建每条腿的控制列
    columns = []
    for leg_name, joint_names in legs.items():
        col = [[sg.Text(leg_name, font=("Helvetica", 14, "bold"))]]
        
        for joint_name in joint_names:
            min_val, max_val = joint_limits[joint_name]
            default_val = default_angles[joint_name]
            
            # 关节类型标签
            joint_type = joint_name.split("_")[1]
            
            col.append([
                sg.Text(joint_type, size=(6, 1)),
                sg.Slider(
                    range=(min_val, max_val),
                    default_value=default_val,
                    resolution=0.01,
                    orientation="h",
                    size=(20, 15),
                    key=joint_name,
                ),
                sg.Text(f"{default_val:.2f}", size=(6, 1), key=f"{joint_name}_display"),
            ])
        
        columns.append(sg.Column(col, element_justification="c"))
    
    # 主布局
    layout = [
        [sg.Text("Leap 机器人 GUI 控制器", font=("Helvetica", 16))],
        [sg.Text("使用滑块设置目标关节角度，系统将通过 PD 控制计算力矩")],
        [sg.HSeparator()],
        [
            columns[0],
            sg.VSeparator(),
            columns[1],
            sg.VSeparator(),
            columns[2],
            sg.VSeparator(),
            columns[3],
        ],
        [sg.HSeparator()],
        [
            sg.Button("重置", size=(10, 1)),
            sg.Button("站立姿态", size=(10, 1)),
            sg.Button("退出", size=(10, 1)),
        ],
    ]
    
    return sg.Window("Leap 控制器", layout, finalize=True)


def compute_pd_torques(
    target_pos: torch.Tensor,
    current_pos: torch.Tensor,
    current_vel: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    torque_limit: torch.Tensor,
) -> torch.Tensor:
    """计算 PD 控制力矩。
    
    Args:
        target_pos: 目标位置 (num_envs, num_dof)
        current_pos: 当前位置 (num_envs, num_dof)
        current_vel: 当前速度 (num_envs, num_dof)
        kp: P 增益 (num_dof,)
        kd: D 增益 (num_dof,)
        torque_limit: 力矩限制 (num_dof,)
        
    Returns:
        控制力矩 (num_envs, num_dof)
    """
    # PD 控制公式: τ = Kp * (q_target - q) - Kd * q_dot
    torques = kp * (target_pos - current_pos) - kd * current_vel
    
    # 限制力矩范围
    torques = torch.clamp(torques, -torque_limit, torque_limit)
    
    return torques


def test_leap_gui_control(args):
    """测试 Leap GUI 控制。
    
    Args:
        args: 命令行参数
    """
    log.info("=" * 80)
    log.info("Leap 机器人 GUI 控制测试")
    log.info("=" * 80)
    
    # 1. 加载配置
    log.info("1. 加载任务配置...")
    task_cfg = LeapTaskCfg()
    
    # 覆盖配置：使用力控，单环境
    task_cfg = update_task_cfg_from_args(task_cfg, args)
    task_cfg.env.num_envs = 1  # 只测试一个环境
    task_cfg.control.control_type = "T"  # 使用力控
    
    log.info(f"   - 仿真器: {task_cfg.simulator}")
    log.info(f"   - 控制类型: {task_cfg.control.control_type}")
    
    # 2. 获取 PD 参数
    log.info("2. 加载 PD 参数...")
    leap_cfg = LeapCfg()
    kp_values = []
    kd_values = []
    torque_limits = []
    
    for joint_name in JOINT_NAMES:
        actuator = leap_cfg.actuators[joint_name]
        kp_values.append(actuator.stiffness)
        kd_values.append(actuator.damping)
        torque_limits.append(actuator.torque_limit)
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    kp = torch.tensor(kp_values, device=device)
    kd = torch.tensor(kd_values, device=device)
    torque_limit = torch.tensor(torque_limits, device=device)
    
    log.info(f"   - Kp: {kp_values[:3]}... (前3个关节)")
    log.info(f"   - Kd: {kd_values[:3]}... (前3个关节)")
    log.info(f"   - 力矩限制: {torque_limits[:3]}... (前3个关节)")
    
    # 3. 创建任务环境（这会自动固定基座，因为 LeapCfg 中 fix_base_link=True）
    log.info("3. 创建任务环境...")
    try:
        env = LeapTask(cfg=task_cfg, device=device)
        log.info(f"   ✓ 任务环境创建成功")
        log.info(f"   ✓ 机器人基座已固定 (LeapCfg.fix_base_link=True)")
    except Exception as e:
        log.error(f"   ✗ 任务环境创建失败: {e}")
        raise
    
    # 4. 获取默认关节位置
    default_angles = leap_cfg.default_joint_positions
    
    # 5. 创建 GUI
    log.info("4. 创建 GUI...")
    window = create_gui(default_angles)
    log.info("   ✓ GUI 创建成功")
    
    # 6. 重置环境
    log.info("5. 重置环境...")
    obs, info = env.reset()
    log.info("   ✓ 环境重置成功")
    
    # 7. 主控制循环
    log.info("6. 开始控制循环...")
    log.info("   提示：使用滑块调整关节角度，系统将自动计算并施加 PD 力矩")
    
    # 初始化目标位置
    target_pos = torch.zeros(1, len(JOINT_NAMES), device=device)
    for i, joint_name in enumerate(JOINT_NAMES):
        target_pos[0, i] = default_angles[joint_name]
    
    step = 0
    running = True
    
    try:
        while running:
            # 读取 GUI 事件
            event, values = window.read(timeout=10)
            
            if event in (sg.WIN_CLOSED, "退出"):
                log.info("用户退出")
                break
            
            # 重置按钮
            if event == "重置":
                for joint_name in JOINT_NAMES:
                    window[joint_name].update(default_angles[joint_name])
                    window[f"{joint_name}_display"].update(f"{default_angles[joint_name]:.2f}")
            
            # 站立姿态按钮
            if event == "站立姿态":
                stand_angles = {
                    "LF_HAA": 0.0, "LF_HFE": 0.8, "LF_KFE": -1.6,
                    "LH_HAA": 0.0, "LH_HFE": 0.8, "LH_KFE": -1.6,
                    "RF_HAA": 0.0, "RF_HFE": 0.8, "RF_KFE": -1.6,
                    "RH_HAA": 0.0, "RH_HFE": 0.8, "RH_KFE": -1.6,
                }
                for joint_name in JOINT_NAMES:
                    window[joint_name].update(stand_angles[joint_name])
                    window[f"{joint_name}_display"].update(f"{stand_angles[joint_name]:.2f}")
            
            # 更新目标位置和显示
            if values:
                for i, joint_name in enumerate(JOINT_NAMES):
                    if joint_name in values:
                        target_pos[0, i] = values[joint_name]
                        window[f"{joint_name}_display"].update(f"{values[joint_name]:.2f}")
            
            # 获取当前状态（从环境的缓冲区读取）
            current_pos = env.dof_pos  # (num_envs, num_dof)
            current_vel = env.dof_vel  # (num_envs, num_dof)
            
            # 计算 PD 力矩
            torques = compute_pd_torques(
                target_pos, current_pos, current_vel, kp, kd, torque_limit
            )
            
            # 执行步进（力矩作为 action）
            obs, reward, terminated, truncated, info = env.step(torques)
            
            step += 1
            if step % 100 == 0:
                log.debug(
                    f"步数: {step}, "
                    f"目标位置: [{target_pos[0, 0]:.2f}, {target_pos[0, 1]:.2f}, {target_pos[0, 2]:.2f}, ...], "
                    f"当前位置: [{current_pos[0, 0]:.2f}, {current_pos[0, 1]:.2f}, {current_pos[0, 2]:.2f}, ...]"
                )
    
    except KeyboardInterrupt:
        log.warning("用户中断")
    except Exception as e:
        log.error(f"控制循环出错: {e}")
        raise
    finally:
        # 8. 清理
        window.close()
        env.close()
        log.info("=" * 80)
        log.info("测试完成")
        log.info("=" * 80)


def main():
    """主函数。"""
    args = get_args()
    
    # 设置日志级别
    if args.debug:
        log.remove()
        log.add(RichHandler(), format="{message}", level="DEBUG")
    
    # 运行测试
    test_leap_gui_control(args)


if __name__ == "__main__":
    main()