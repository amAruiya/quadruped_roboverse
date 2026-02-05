"""地形生成器测试脚本。

该脚本用于测试地形生成和注入功能，不涉及机器人任务。
直接使用 Handler 和 TerrainCfg 进行测试。

使用方法：
    python MyRobot/test/test_terrain.py --sim mujoco
    python MyRobot/test/test_terrain.py --sim isaacgym --num_envs 100
    python MyRobot/test/test_terrain.py --sim genesis --headless
"""

from __future__ import annotations

import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.scenario.scene import SceneCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_handler
from MyRobot.configs.task_cfg import TerrainCfg
from MyRobot.terrain.generator import TerrainGenerator

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


# =============================================================================
# 地形配置（在此修改地形比例和参数）
# =============================================================================

# 地形类型比例（总和应为 1.0）
TERRAIN_PROPORTIONS = {
    # "flat": 0.1,        # 平坦地形
    "rough": 0.2,       # 粗糙地形，现在还不够粗糙
    "stairs_up": 0.15,  # 上行台阶
    "stairs_down": 0.15,# 下行台阶
    "slope_up": 0.15,   # 上坡
    "slope_down": 0.15, # 下坡
    "pyramid": 0.05,    # 金字塔
    # "discrete": 0.051,   # 离散障碍物，密度不均匀,并且没有竖直向下的坑
}

# 地形网格配置
TERRAIN_GRID_CONFIG = {
    "num_rows": 10,           # 地形网格行数
    "num_cols": 10,          # 地形网格列数
    "terrain_length": 10.0,   # 单个地形块长度（米）
    "terrain_width": 10.0,    # 单个地形块宽度（米）
}

# 物理参数
TERRAIN_PHYSICS = {
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
}

# 课程学习
CURRICULUM_ENABLED = True  # 是否启用课程学习（列方向难度递增）


# =============================================================================
# 命令行参数
# =============================================================================

@dataclass
class Args:
    """命令行参数。"""
    
    sim: Literal["mujoco", "isaacgym", "isaacsim", "genesis"] = "mujoco"
    """仿真器类型"""
    
    num_envs: int = 1
    """并行环境数量（实际不创建环境，仅用于 handler 初始化）"""
    
    headless: bool = False
    """是否无头模式"""
    
    mesh_type: Literal["heightfield", "trimesh"] = "trimesh"
    """地形网格类型"""


def parse_args() -> Args:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Terrain Generator Test")
    
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        choices=["mujoco", "isaacgym", "isaacsim", "genesis"],
        help="仿真器类型"
    )
    
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="并行环境数量"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="无头模式"
    )
    
    parser.add_argument(
        "--mesh_type",
        type=str,
        default="trimesh",
        choices=["heightfield", "trimesh"],
        help="地形网格类型"
    )
    
    args_dict = vars(parser.parse_args())
    return Args(**args_dict)


# =============================================================================
# 主测试函数
# =============================================================================

def test_terrain_generator(args: Args):
    """测试地形生成器。
    
    Args:
        args: 命令行参数
    """
    log.info("=" * 80)
    log.info("地形生成器测试")
    log.info("=" * 80)
    
    # 1. 创建地形配置
    log.info("\n1. 创建地形配置...")
    terrain_cfg = TerrainCfg(
        mesh_type=args.mesh_type,
        terrain_proportions=TERRAIN_PROPORTIONS,
        curriculum=CURRICULUM_ENABLED,
        **TERRAIN_GRID_CONFIG,
        **TERRAIN_PHYSICS,
    )
    
    log.info(f"   - 地形类型: {args.mesh_type}")
    log.info(f"   - 网格尺寸: {terrain_cfg.num_rows} x {terrain_cfg.num_cols}")
    log.info(f"   - 单块尺寸: {terrain_cfg.terrain_length}m x {terrain_cfg.terrain_width}m")
    log.info(f"   - 课程学习: {CURRICULUM_ENABLED}")
    log.info(f"   - 地形比例:")
    for terrain_type, prop in TERRAIN_PROPORTIONS.items():
        log.info(f"      {terrain_type:12s}: {prop:.1%}")
    
    # 2. 创建地形生成器
    log.info("\n2. 初始化地形生成器...")
    generator = TerrainGenerator(terrain_cfg, simulator=args.sim)
    
    # 3. 生成地形数据
    log.info("\n3. 生成地形高度场...")
    height_fields = generator.generate()
    
    log.info(f"   ✓ 生成了 {len(height_fields)} 个地形块")
    
    # 打印部分地形信息
    for i in range(min(3, len(height_fields))):
        hf = height_fields[i]
        row, col = divmod(i, terrain_cfg.num_cols)
        terrain_type = generator.terrain_types[row, col]
        log.info(
            f"   - 地形 [{row},{col}]: {terrain_type:12s} | "
            f"尺寸={hf.shape} | 范围=[{hf.heights.min():.3f}, {hf.heights.max():.3f}]m"
        )
    
    if len(height_fields) > 3:
        log.info(f"   - ... (还有 {len(height_fields) - 3} 个地形块)")
    
    # 4. 创建仿真场景（仅用于地形注入）
    log.info(f"\n4. 创建 {args.sim} 仿真环境...")
    
    scenario = ScenarioCfg(
        robots=['leap'],  # 不加载机器人
        objects=[],
        scene=None,  # 禁用默认地面
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        create_ground = False
    )
    
    handler = get_handler(scenario)
    log.info(f"   ✓ Handler 创建成功: {type(handler).__name__}")
    
    # 5. 注入地形
    log.info("\n5. 注入地形到仿真环境...")
    generator.bind_handler(handler)
    generator()  # 执行注入
    
    log.info(f"   ✓ 地形注入完成 (mesh_type={args.mesh_type})")
    
    # 6. 运行仿真（可视化）
    if not args.headless:
        log.info("\n6. 启动可视化...")
        log.info("   按 Ctrl+C 退出\n")
        
        try:
            step = 0
            while True:
                handler.simulate()
                step += 1
                
                if step % 100 == 0:
                    log.debug(f"   仿真步数: {step}")
                    
        except KeyboardInterrupt:
            log.info("\n   用户中断")
    
    else:
        log.info("\n6. 无头模式，跳过可视化")
        log.info("   运行 100 步仿真验证稳定性...")
        
        for step in range(100):
            handler.simulate()
            if step % 20 == 0:
                log.debug(f"   步数: {step}/100")
        
        log.info("   ✓ 仿真稳定")
    
    # 7. 清理
    log.info("\n7. 清理资源...")
    handler.close()
    
    log.info("\n" + "=" * 80)
    log.info("✅ 地形生成器测试完成！")
    log.info("=" * 80)


def test_terrain_data_only():
    """仅测试地形数据生成（不启动仿真器）。"""
    import time
    
    log.info("=" * 80)
    log.info("地形数据生成测试（无仿真器）")
    log.info("=" * 80)
    
    # 创建配置
    terrain_cfg = TerrainCfg(
        mesh_type="heightfield",
        terrain_proportions=TERRAIN_PROPORTIONS,
        curriculum=CURRICULUM_ENABLED,
        **TERRAIN_GRID_CONFIG,
    )
    
    # 创建生成器（任意仿真器类型,因为不实际注入）
    generator = TerrainGenerator(terrain_cfg, simulator="mujoco")
    
    log.info("\n生成地形高度场...")
    start_time = time.time()
    height_fields = generator.generate()
    generate_time = time.time() - start_time
    
    log.info(f"\n✓ 成功生成 {len(height_fields)} 个地形块 (耗时: {generate_time:.3f}s)")
    log.info(f"✓ 地形类型分布:")
    
    # 统计地形类型
    from collections import Counter
    terrain_types_flat = generator.terrain_types.flatten()
    type_counts = Counter(terrain_types_flat)
    
    for terrain_type, count in sorted(type_counts.items()):
        percentage = count / len(terrain_types_flat) * 100
        log.info(f"   {terrain_type:12s}: {count:3d} / {len(terrain_types_flat)} ({percentage:.1f}%)")
    
    # 测试格式转换
    log.info("\n测试格式转换...")
    convert_start = time.time()
    trimesh = generator.to_trimesh(height_fields[0])
    convert_time = time.time() - convert_start
    log.info(f"   ✓ TriMesh: {len(trimesh.vertices)} 顶点, {len(trimesh.triangles)} 三角形 (耗时: {convert_time:.3f}s)")
    
    log.info("\n" + "=" * 80)
    log.info("✅ 地形数据测试完成！")
    log.info("=" * 80)



def main():
    """主函数。"""
    args = parse_args()
    
    # 选择测试模式
    if args.sim == "none" or args.num_envs == 0:
        # 仅测试数据生成
        test_terrain_data_only()
    else:
        # 完整测试（含仿真器）
        test_terrain_generator(args)


if __name__ == "__main__":
    main()