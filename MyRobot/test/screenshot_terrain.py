"""快速截图脚本，验证地形和机器人渲染是否正确。

在 headless 模式下通过相机渲染生成截图，无需显示器。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# IsaacGym 必须在 torch 之前导入
try:
    from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
except ImportError:
    pass

import torch
import numpy as np

from MyRobot.configs.leap_cfg import LeapTaskCfg
from MyRobot.tasks.leap_task import LeapTask

OUTPUT_DIR = "MyRobot/test/output/screenshots"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = LeapTaskCfg()
    cfg.env.num_envs = 16
    cfg.headless = True
    cfg.simulator = "isaacgym"

    print("初始化 LeapTask ...")
    task = LeapTask(cfg=cfg, device="cuda")
    print(f"初始化完成: {task.num_envs} 环境, terrain_generator={task.terrain_generator is not None}")

    if task.terrain_generator is None:
        print("WARNING: 地形生成器为空！")
        return

    # 运行几步，让机器人稳定
    print("运行 50 步...")
    with torch.no_grad():
        for i in range(50):
            actions = torch.zeros(task.num_envs, task.num_actions, device=task.device)
            task.step(actions)

    # 获取机器人 z 位置，检查是否站在地形上
    states = task.handler.get_states()
    robot_pos = states.robots["leap"].root_state[:, :3]
    print(f"\n机器人位置统计 (0步随机动作后):")
    print(f"  z均值: {robot_pos[:, 2].mean().item():.3f}m")
    print(f"  z最小: {robot_pos[:, 2].min().item():.3f}m")
    print(f"  z最大: {robot_pos[:, 2].max().item():.3f}m")
    print(f"  xy范围: x=[{robot_pos[:, 0].min().item():.1f}, {robot_pos[:, 0].max().item():.1f}], "
          f"y=[{robot_pos[:, 1].min().item():.1f}, {robot_pos[:, 1].max().item():.1f}]")

    if task.env_origins is not None:
        print(f"\n地形 env_origins 统计:")
        print(f"  z均值(地面高度): {task.env_origins[:, 2].mean().item():.3f}m")
        print(f"  z最大(最高地形): {task.env_origins[:, 2].max().item():.3f}m")

    # 判断机器人是否站在地形上（z > 0.1 且 z < 2.0 认为正常）
    z_vals = robot_pos[:, 2]
    above_ground = (z_vals > 0.05).sum().item()
    print(f"\n站在地形上的机器人数量: {above_ground}/{task.num_envs}")

    if above_ground < task.num_envs // 2:
        print("WARNING: 超过一半机器人不在地面上，地形可能未正确生效！")
    else:
        print("OK: 大多数机器人成功站在地形上")

    # 尝试截图（如果 handler 有 viewer）
    try:
        import cv2
        gym = task.handler.gym
        sim = task.handler.sim

        # 询问是否可以渲染
        cam_props = gymapi.CameraProperties()
        cam_props.width = 1280
        cam_props.height = 720
        cam_props.enable_tensors = True

        # 在第一个 env 创建临时相机
        env_handle = task.handler.env_handles[0]
        cam_handle = gym.create_camera_sensor(env_handle, cam_props)

        # 设置相机位置：俯视所有 env
        cam_pos = gymapi.Vec3(0.0, 0.0, 40.0)
        cam_target = gymapi.Vec3(20.0, 20.0, 0.0)
        gym.set_camera_location(cam_handle, env_handle, cam_pos, cam_target)

        # 渲染
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)

        # 获取 RGB 图像
        image_tensor = gym.get_camera_image_gpu_tensor(sim, env_handle, cam_handle, gymapi.IMAGE_COLOR)
        torch_image = gymtorch.wrap_tensor(image_tensor)
        image_np = torch_image.cpu().numpy()  # shape: (H, W, 4) RGBA

        gym.end_access_image_tensors(sim)

        # 转换并保存
        if image_np.shape[-1] >= 3:
            bgr = cv2.cvtColor(image_np[:, :, :3], cv2.COLOR_RGB2BGR)
            out_path = os.path.join(OUTPUT_DIR, "terrain_overhead.png")
            cv2.imwrite(out_path, bgr)
            print(f"\n截图已保存: {out_path}")
        
    except Exception as e:
        print(f"\n截图失败（可能需要 cv2 或 GPU 渲染）: {e}")
        import traceback; traceback.print_exc()

    print("\n测试完成！")


if __name__ == "__main__":
    main()
