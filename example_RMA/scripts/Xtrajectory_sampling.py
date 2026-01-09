# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR



import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path

import numpy as np
import torch
import time

import matplotlib.pyplot as plt
import pickle

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.kp_range = [25.0, 25.0]
    env_cfg.domain_rand.kd_range = [0.5, 0.5]
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]

    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_base_mass = False
    # env_cfg.domain_rand.randomize_kp_kd = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.friction_range = [1, 1]

    # env_cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0] #smooth slope
    env_cfg.terrain.terrain_proportions = [0, 1.0, 0, 0, 0] #rough slope
    # env_cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0] #stairs up
    # env_cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0] #stairs down
    # env_cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0] #discrete

    env_cfg.commands.ranges.lin_vel_x = [1, 1]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.heading = [0, 0]

    env_cfg.terrain.mesh_type = 'plane'

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    # train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_teacher_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    load_run = "BEST_Dec07_13-29-32_teacher"
    checkpoint = 4000
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    path = get_load_path(log_root, load_run=load_run,
                                        checkpoint=checkpoint)
    print(f"Loading model from: {path}")
    ppo_runner.load(path = path)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 900 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        time.sleep(0.01)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename) 
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[9,:3]
            # if(int(i / 1000) % 2 == 0):
            #     camara_position = lootat.detach().cpu().numpy() + [0,2,1]
            # else:
            #     camara_position = lootat.detach().cpu().numpy() + [2, 0, 1]
            camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
            env.set_camera(camara_position, lootat)
            # print(env.root_states[0,2])
            # camera_position += camera_vel * env.dt
            # env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            for robot_index in range(env.num_envs):
                logger.log_states(
                    {
                        f'dof_pos_target_all_{robot_index}': (actions[robot_index] * env.cfg.control.action_scale).detach().cpu().numpy(),
                        f'dof_pos_all_{robot_index}': env.dof_pos[robot_index].detach().cpu().numpy(),
                        f'dof_vel_all_{robot_index}': env.dof_vel[robot_index].detach().cpu().numpy(),
                        f'dof_torque_all_{robot_index}': env.torques[robot_index].detach().cpu().numpy(),
                        
                        f'command_x_{robot_index}': env.commands[robot_index, 0].item(),
                        f'command_y_{robot_index}': env.commands[robot_index, 1].item(),
                        f'command_yaw_{robot_index}': env.commands[robot_index, 2].item(),
                        f'base_vel_x_{robot_index}': env.base_lin_vel[robot_index, 0].item(),
                        f'base_vel_y_{robot_index}': env.base_lin_vel[robot_index, 1].item(),
                        f'base_vel_z_{robot_index}': env.base_lin_vel[robot_index, 2].item(),
                        f'base_vel_yaw_{robot_index}': env.base_ang_vel[robot_index, 2].item(),
                        
                        # contact_forces 已经是 numpy 转换逻辑了，保持不变
                        f'contact_forces_z_all_{robot_index}': env.contact_forces[robot_index, :, 2].detach().cpu().numpy()
                    }
                )
        elif i==stop_state_log:
            # logger.plot_states()
            logger.save_logs()
            # plot_kfe_phase_diagram("/opt/rui/Data/quadruped/RMA/simulation_logs.pkl")
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()


def plot_kfe_phase_diagram(log_path="simulation_logs.pkl"):
    print(f"Loading logs from {log_path}...")
    try:
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {log_path} not found. Please check the path.")
        return

    # 获取 state_log
    # 注意：根据您的保存代码，结构可能是 data['state_log']
    state_log = data.get('state_log', data) 
    
    # 检查键名是否存在
    if 'dof_pos_all' not in state_log or 'dof_torque_all' not in state_log:
        print("Error: 'dof_pos_all' or 'dof_torque_all' not found in logs.")
        print("Available keys:", state_log.keys())
        return

    # 将数据转换为 numpy 数组 (Shape: [Time_steps, 12])
    # 假设之前的代码已经用 detach().cpu().numpy() 存入了 list
    dof_pos = np.array(state_log['dof_pos_all'])
    dof_torque = np.array(state_log['dof_torque_all'])

    # 定义 KFE 关节的索引
    # 通常顺序是: [FL_HAA, FL_HFE, FL_KFE, FR_HAA, ..., RR_KFE]
    # KFE (Knee) 通常是每条腿的第三个关节 (索引 2, 5, 8, 11)
    kfe_indices = [2, 5, 8, 11]
    leg_names = ["FL (Front Left)", "FR (Front Right)", "RL (Rear Left)", "RR (Rear Right)"]

    # 创建绘图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('KFE Joint Phase Diagram (Theta vs Tau)', fontsize=16)
    
    # 扁平化 axs 以便循环
    axs_flat = axs.flatten()

    for i, ax in enumerate(axs_flat):
        idx = kfe_indices[i]
        
        # 提取当前关节的数据
        theta = dof_pos[20:, idx]
        tau = dof_torque[20:, idx]
        
        # 绘制曲线
        # 使用 alpha 设置透明度，可以看出轨迹的重叠密度
        ax.plot(theta, tau, linewidth=1.0, alpha=0.5)
        
        # 标注起点和终点（可选，帮助看方向）
        ax.plot(theta[0], tau[0], 'go', label='Start')
        ax.plot(theta[-1], tau[-1], 'ro', label='End')

        ax.set_title(f"{leg_names[i]} KFE (Idx {idx})")
        ax.set_xlabel(r'$\theta$ (Position) [rad]')
        ax.set_ylabel(r'$\tau$ (Torque) [Nm]')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    
    # 保存图片或显示
    save_name = 'kfe_phase_diagram.png'
    plt.savefig(save_name)
    print(f"Plot saved to {save_name}")
    plt.show()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
