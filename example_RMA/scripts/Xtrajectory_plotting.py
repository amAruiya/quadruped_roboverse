import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np

# ================= 配置区域 =================
# 根据提供的刚体列表：
# 0:base, 
# 1-4:LF(hip, thigh, calf, FOOT), 
# 5-8:LH(...), 
# 9-12:RF(...), 
# 13-16:RH(...)
# 因此，FOOT 的索引为 4, 8, 12, 16
FEET_INDICES = [4, 8, 12, 16] 

# 对应的腿部名称 (顺序必须与 FEET_INDICES 一致)
LEG_NAMES = ["LF (Front Left)", "LH (Rear Left)", "RF (Front Right)", "RH (Rear Right)"]

# KFE (膝关节) 的 DOF 索引
# 假设 DOF 顺序与刚体顺序一致 (LF, LH, RF, RH)
# 每个腿 3 个自由度，KFE 通常是第 3 个 (索引 2)
# LF: 0,1,2 -> KFE=2
# LH: 3,4,5 -> KFE=5
# RF: 6,7,8 -> KFE=8
# RH: 9,10,11 -> KFE=11
KFE_DOF_INDICES = [2, 5, 8, 11]
# ===========================================

def plot_kfe_phase_diagram(log_path="simulation_logs.pkl", INDEX=0):
    print(f"Loading logs from {log_path}...")
    try:
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {log_path} not found. Please check the path.")
        return

    state_log = data.get('state_log', data)
    
    # 构造键名
    key_pos = f'dof_pos_all_{INDEX}'
    key_torque = f'dof_torque_all_{INDEX}'
    key_force = f'contact_forces_z_all_{INDEX}'

    # 检查键名
    if key_pos not in state_log or key_torque not in state_log:
        print(f"Error: keys {key_pos} or {key_torque} not found.")
        return

    # 转换为 numpy
    dof_pos = np.array(state_log[key_pos])
    dof_torque = np.array(state_log[key_torque])
    
    # 获取接触力 (Shape: [Time, 17])
    if key_force in state_log:
        contact_forces = np.array(state_log[key_force])
        print(f"Contact forces shape: {contact_forces.shape}") # 应该是 (T, 17)
    else:
        print(f"Warning: {key_force} not found.")
        contact_forces = None

    # 创建绘图
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'KFE Joint Phase Diagram (Robot {INDEX})\nLeg Order: LF, LH, RF, RH', fontsize=16)
    
    axs_flat = axs.flatten()

    cmap_impact = plt.get_cmap('Reds')
    color_air = mcolors.to_rgba('tab:blue')

    start_step = 20 

    # 遍历四条腿
    for i, ax in enumerate(axs_flat):
        # 获取当前腿对应的 关节索引 和 接触力索引
        joint_idx = KFE_DOF_INDICES[i]
        foot_idx = FEET_INDICES[i]
        leg_name = LEG_NAMES[i]

        # 1. 准备数据\
        n=100
        theta = dof_pos[start_step:start_step+n, joint_idx]
        tau = dof_torque[start_step:start_step+n, joint_idx]
        
        points = np.array([theta, tau]).T.reshape(-1, 1, 2)
        # points = np.array([range(len(theta)), theta]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 2. 颜色逻辑
        if contact_forces is not None:
            # 确保索引在范围内 (17维数据)
            if foot_idx < contact_forces.shape[1]:
                forces = contact_forces[start_step:start_step+n-1, foot_idx]
            else:
                print(f"Error: Foot index {foot_idx} out of bounds for contact_forces shape {contact_forces.shape}")
                forces = np.zeros(len(segments))

            # 初始化颜色
            colors = np.array([color_air] * len(segments))

            # 接触判断阈值 > 10N
            contact_mask = forces > 10.0

            if np.any(contact_mask):
                contact_vals = forces[contact_mask]
                
                # 颜色归一化
                f_min = 10.0
                f_max = np.max(contact_vals) + 1e-6
                norm_vals = (contact_vals - f_min) / (f_max - f_min)
                
                # 映射到 Reds 色谱的 0.3~1.0 区间
                scaled_vals = 0.3 + 0.7 * norm_vals
                colors[contact_mask] = cmap_impact(scaled_vals)
        else:
            colors = [color_air] * len(segments)

        # 3. 绘制
        lc = LineCollection(segments, colors=colors, linewidths=1.2, alpha=0.8)
        ax.add_collection(lc)

        ax.autoscale()
        ax.margins(0.1)

        # 标记起止点
        ax.plot(theta[0], tau[0], marker='o', color='green', markersize=5, label='Start')
        ax.plot(theta[-1], tau[-1], marker='o', color='gold', markersize=5, label='End')

        ax.set_title(f"{leg_name} (Joint Idx {joint_idx}, Foot Idx {foot_idx})")
        ax.set_xlabel(r'$\theta$ [rad]')
        ax.set_ylabel(r'$\tau$ [Nm]')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='tab:blue', lw=2),
                Line2D([0], [0], color='darkred', lw=2)
            ]
            ax.legend(custom_lines, ['Air (F <= 10N)', 'Ground (Darker = Higher Force)'])

    plt.tight_layout()
    save_name = f'kfe_phase_diagram_{INDEX}.png'
    # plt.savefig(save_name, dpi=150)
    print(f"Plot saved to {save_name}")
    plt.show()

if __name__ == "__main__":
    path = "/opt/rui/Data/quadruped/RMA/simulation_logs.pkl"
    # 根据你的循环逻辑
    for idx in range(1): # 测试第一个即可
        plot_kfe_phase_diagram(path, INDEX=idx)