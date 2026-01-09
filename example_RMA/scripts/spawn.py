import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import os
import PySimpleGUI as sg
import socket
import struct
import torch

# ================= 配置区域 =================
TARGET_PORT = 8001 
CMD_FMT = '60f'
CMD_SIZE = struct.calcsize(CMD_FMT)
STATE_FMT = '46f'

# PD 参数 (在 Python 端模拟)
KP = 80.0
KD = 2.0
MAX_TORQUE = 33.5 # 模拟电机的物理力矩限制 (Nm)

# 关节顺序 (用于UDP通信)
JOINT_NAMES_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE"
]
# ===========================================

class UdpSender:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.setblocking(False) 
        self.addr = None
        print(f"[UDP] Listening on 127.0.0.1:{port}")

    def recv_command(self):
        try:
            data, self.addr = self.sock.recvfrom(CMD_SIZE)
            if len(data) != CMD_SIZE:
                return None
            unpacked = struct.unpack(CMD_FMT, data)
            return unpacked 
        except BlockingIOError:
            return None
        except Exception as e:
            print(f"UDP Error: {e}")
            return None

    def send_state(self, q, dq, quat_wxyz, gyro, accel_free, tau):
        if self.addr is None:
            return
        data = list(q) + list(dq) + list(quat_wxyz) + list(gyro) + list(accel_free) + list(tau)
        packed = struct.pack(STATE_FMT, *data)
        self.sock.sendto(packed, self.addr)

def get_rotation_matrix(q):
    """ 将四元数 (x, y, z, w) 转换为旋转矩阵 """
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def main():
    # --- 1. Isaac Gym 初始化设置 ---
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="Leap Robot PySimpleGUI Control & Stream")

    sim_params = gymapi.SimParams()
    # [关键] 减小步长以保证力控稳定性
    sim_dt = 0.002  
    sim_params.dt = sim_dt
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.use_gpu = True
    sim_params.physx.num_threads = 4

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # --- 2. 加载机器人资产 ---
    # 请根据您的实际路径修改
    full_path = '/opt/rui/Data/quadruped/RMA/resources/robots/Leap_RL/urdf/Leap.urdf' 
    if not os.path.exists(full_path):
        sg.popup_error(f"URDF Not Found: {full_path}")
        return
    
    asset_root = os.path.dirname(full_path)
    asset_file = os.path.basename(full_path)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True 
    # [关键] 默认设为力控模式 (EFFORT)
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT) 
    
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # --- 3. 创建环境 ---
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 1.5, 1.0), gymapi.Vec3(0.0, 0.0, 0.4))
    
    env = gym.create_env(sim, gymapi.Vec3(-2.0, 0.0, -2.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)
    
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.32)
    actor_handle = gym.create_actor(env, robot_asset, initial_pose, "LeapRobot", 0, 1)

    # --- 4. 配置关节属性 (切换为纯力控) ---
    dof_names = gym.get_asset_dof_names(robot_asset)
    num_dofs = gym.get_asset_dof_count(robot_asset)
    dof_props = gym.get_asset_dof_properties(robot_asset)

    # [关键] 设置为 EFFORT 模式，并禁用内置 PD (设为0)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    dof_props["stiffness"].fill(0.0) 
    dof_props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    # 映射表
    dof_map = {name: i for i, name in enumerate(dof_names)}
    
    # [恢复] 您原始脚本中的默认初始角度
    default_angles = {
        'LF_HAA': -0.2, 'LH_HAA': -0.2, 'RF_HAA': 0.2, 'RH_HAA': 0.2,
        'LF_HFE': 0.83, 'LH_HFE': 0.83, 'RF_HFE': 0.83, 'RH_HFE': 0.83,
        'LF_KFE': -1.72, 'LH_KFE': -1.72, 'RF_KFE': -1.72, 'RH_KFE': -1.72,
    }

    # 目标数组 (Desired Position)
    targets_pos = np.zeros(num_dofs, dtype=np.float32)
    for name, val in default_angles.items():
        if name in dof_map:
            targets_pos[dof_map[name]] = val

    # 初始化关节状态 (Sim Physics Init)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    dof_states['pos'] = targets_pos
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_POS)

    # 导出索引映射 (Gym -> Controller Order)
    export_indices = []
    for name in JOINT_NAMES_ORDER:
        found = False
        for gym_name, gym_idx in dof_map.items():
            if name in gym_name:
                export_indices.append(gym_idx)
                found = True
                break
        if not found:
            export_indices.append(0)

    # --- 5. PySimpleGUI ---
    sg.theme('DarkBlue3')
    ranges = { 'HAA': (-0.8, 0.8), 'HFE': (-1.5, 1.5), 'KFE': (-2.5, 0.0) }
    
    def create_joint_slider(leg_name, joint_type):
        full_name = f"{leg_name}_{joint_type}"
        default_val = default_angles.get(full_name, 0.0)
        min_v, max_v = ranges.get(joint_type, (-1.0, 1.0))
        return [
            sg.Text(joint_type, size=(4,1)), 
            sg.Slider(range=(min_v, max_v), default_value=default_val, resolution=0.01, 
                      orientation='h', size=(15,15), key=full_name, enable_events=True)
        ]

    legs_order = ['LF', 'LH', 'RF', 'RH']
    columns = []
    for leg in legs_order:
        col = [
            [sg.Text(f"{leg} Leg")],
            create_joint_slider(leg, 'HAA'),
            create_joint_slider(leg, 'HFE'),
            create_joint_slider(leg, 'KFE')
        ]
        columns.append(sg.Column(col, element_justification='c'))

    # [恢复] 包含 Reset 和 Exit 按钮的布局
    layout = [
        [sg.Text("Leap Torque Control (Manual PD)", font=("Helvetica", 16))],
        [sg.HSeparator()],
        [columns[0], sg.VSeparator(), columns[1], sg.VSeparator(), columns[2], sg.VSeparator(), columns[3]],
        [sg.HSeparator()],
        [sg.Button('Reset', size=(10,1)), sg.Button('Exit', size=(10,1))] # 恢复按钮
    ]
    
    window = sg.Window('Leap Controller', layout, finalize=True)

    # --- 6. 运行时变量 ---
    udp = UdpSender(TARGET_PORT)
    last_lin_vel = np.zeros(3)
    
    # 准备 Tensor 用于高效设置力矩
    actuation_tensor = torch.zeros(num_dofs, dtype=torch.float, device='cpu') 

    print("Running...")

    while not gym.query_viewer_has_closed(viewer):
        event, values = window.read(timeout=1)
        if event == sg.WIN_CLOSED or event == 'Exit': break
        
        # [恢复] Reset 按钮逻辑
        if event == 'Reset':
            for name, val in default_angles.items():
                if name in window.key_dict:
                    window[name].update(val)
                    if name in dof_map:
                        targets_pos[dof_map[name]] = val
                        
        # 1. 更新目标位置
        if values:
            for name, val in values.items():
                if name in dof_map:
                    targets_pos[dof_map[name]] = val

        # 2. 获取当前状态
        dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
        curr_pos = dof_states['pos']
        curr_vel = dof_states['vel']
        
        # 3. [关键] 计算手动 PD 力矩
        # tau = Kp * (q_des - q) + Kd * (0 - dq)
        torques = KP * (targets_pos - curr_pos) - KD * curr_vel
        
        # 3.1 限制力矩 (模拟电机饱和)
        torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

        # 4. 应用力矩到仿真 (EFFORT模式)
        actuation_tensor[:] = torch.from_numpy(torques)
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actuation_tensor))

        # 5. 物理步进
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 6. 数据打包与 UDP 发送
        rigid_body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
        base_state = rigid_body_states[0]

        # 准备数据
        sim_q = np.zeros(12, dtype=np.float32)
        sim_dq = np.zeros(12, dtype=np.float32)
        sim_tau = np.zeros(12, dtype=np.float32)

        # [关键] 填充 sim_tau 为计算出的指令力矩 (feedback)
        for i, gym_idx in enumerate(export_indices):
            sim_q[i] = curr_pos[gym_idx]
            sim_dq[i] = curr_vel[gym_idx]
            sim_tau[i] = torques[gym_idx] # <--- 发送指令力矩，修复估计器问题

        # IMU 数据
        quat = base_state['pose']['r'] 
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
        
        lin_vel_world = np.array([base_state['vel']['linear'][0], base_state['vel']['linear'][1], base_state['vel']['linear'][2]])
        ang_vel_world = np.array([base_state['vel']['angular'][0], base_state['vel']['angular'][1], base_state['vel']['angular'][2]])
        
        R_wb = get_rotation_matrix(quat)
        gyro_body = R_wb.T @ ang_vel_world
        accel_world = (lin_vel_world - last_lin_vel) / sim_dt
        last_lin_vel = lin_vel_world

        # 通信
        _ = udp.recv_command()
        udp.send_state(sim_q, sim_dq, quat_wxyz, gyro_body, accel_world, sim_tau)

        # 渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    window.close()
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()