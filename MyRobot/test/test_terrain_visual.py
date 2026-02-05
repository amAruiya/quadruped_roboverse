
import sys
import os
import argparse
import subprocess
import time
# Add project root BEFORE importing other modules that might depend on path
sys.path.append(os.getcwd())

# CRITICAL: Force import isaacgym first, bypassing any metasim dependencies
try:
    import isaacgym
except ImportError:
    pass

# Import Handler FIRST to ensure isaacgym is imported before torch
from metasim.sim.isaacgym.isaacgym import IsaacgymHandler

import numpy as np
from loguru import logger
import cv2  # For saving images
import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg
from MyRobot.configs.task_cfg import TerrainCfg
from MyRobot.terrain.generator import TerrainGenerator
from MyRobot.robots.leap_cfg import LeapCfg

def run_test_case(mode, output_dir):
    logger.info(f"Starting Visual Terrain Test: {mode}")
    output_dir = os.path.join(output_dir, mode) if mode == "all_modes" else output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================
    # 1. Configure Terrain
    # ==========================
    terrain_cfg = TerrainCfg()
    terrain_cfg.mesh_type = "trimesh" 
    
    # Apply RMA reference parameters
    terrain_cfg.terrain_length = 8.0
    terrain_cfg.terrain_width = 8.0
    terrain_cfg.horizontal_scale = 0.1
    terrain_cfg.vertical_scale = 0.005
    terrain_cfg.border_size = 2.0 

    # Enforce step parameters (RMA default is 0.31m)
    setattr(terrain_cfg, 'step_width', 0.31)
    setattr(terrain_cfg, 'step_depth', 0.31)
    
    # Restore Mode Logic
    types_list = ["flat", "rough", "slope", "stairs_up", "stairs_down", "discrete", "stepping_stones"]
    
    if mode in types_list:
        # 1. Individual Terrain Test
        terrain_cfg.num_rows = 5
        terrain_cfg.num_cols = 5
        # Set explicitly 100% for this type
        terrain_cfg.terrain_proportions = {t: (1.0 if t == mode else 0.0) for t in types_list}
        terrain_cfg.curriculum = False
        
    elif mode == "combined":
        # 2. Combined Test
        terrain_cfg.num_rows = 6
        terrain_cfg.num_cols = 7
        terrain_cfg.terrain_proportions = {t: 1.0 for t in types_list}
        terrain_cfg.curriculum = False
        
    elif mode == "curriculum":
        # 3. Difficulty/Curriculum Test
        terrain_cfg.num_rows = 10 
        terrain_cfg.num_cols = 7
        terrain_cfg.terrain_proportions = {t: 1.0 for t in types_list}
        terrain_cfg.curriculum = True
        terrain_cfg.max_init_terrain_level = 9
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    logger.info(f"Terrain Config for {mode}: {terrain_cfg.num_rows}x{terrain_cfg.num_cols}, Curriculum={terrain_cfg.curriculum}")

    # ==========================
    # 2. Configure Scenario
    # ==========================
    # Camera position adjustment based on grid size
    grid_cx = (terrain_cfg.num_cols * terrain_cfg.terrain_width) / 2.0
    grid_cy = (terrain_cfg.num_rows * terrain_cfg.terrain_length) / 2.0
    
    scenario_cfg = ScenarioCfg(
        simulator="isaacgym",
        headless=True,
        create_ground=False,
        scene=SceneCfg(),
        robots=[
            # Explicitly set scale to tuple to avoid isaacgym handler bug
            LeapCfg(fix_base_link=True, scale=(1.0, 1.0, 1.0))
        ],
        cameras=[
            PinholeCameraCfg(
                name="Global_View",
                pos=(-5.0, -5.0, 25.0), # High up
                look_at=(grid_cx, grid_cy, 0.0), # Look at center
                data_types=["rgb"],
                width=1920,
                height=1080
            ),
            PinholeCameraCfg(
                name="Side_View",
                pos=(-5.0, grid_cy, 10.0),
                look_at=(grid_cx, grid_cy, 0.0),
                data_types=["rgb"],
                width=1920,
                height=1080
            )
        ],
        lights=[
            DistantLightCfg(
                polar=45.0,
                azimuth=45.0,
                intensity=1000.0
            )
        ]
    )
    
    # ==========================
    # 3. Run Simulation
    # ==========================
    logger.info(f"Initializing Handler for {mode}...")
    handler = IsaacgymHandler(scenario_cfg)
    handler.launch()
    
    logger.info("Generating Terrain...")
    gen = TerrainGenerator(terrain_cfg, simulator="isaacgym")
    gen.bind_handler(handler)
    gen()
    
    logger.info("Simulating and capturing...")
    # Warmup
    for _ in range(5):
        handler.simulate()
        
    # Capture
    try:
        states = handler.get_states()
        cameras = states.cameras
        
        saved_any = False
        for cam_name, cam_state in cameras.items():
            if cam_state.rgb is not None:
                rgb = cam_state.rgb[0]
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                
                # Conversion
                if rgb.dtype != np.uint8:
                     if rgb.max() <= 1.5 and rgb.dtype == np.float32:
                         rgb = (rgb * 255).astype(np.uint8)
                     else:
                         rgb = rgb.astype(np.uint8)
                
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # Naming: mode_view.png
                filename = os.path.join(output_dir, f"{mode}_{cam_name}.png")
                cv2.imwrite(filename, bgr)
                logger.success(f"Saved: {filename}")
                saved_any = True
                
        if not saved_any:
            logger.warning("No images found in camera state!")
            
    except Exception as e:
        logger.error(f"Image capture failed: {e}")
        import traceback
        traceback.print_exc()

    # No clean close in IsaacgymHandler usually, process termination handles it.
    logger.info(f"Case {mode} Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run specific mode (worker)")
    args = parser.parse_args()
    
    OUTPUT_DIR = "MyRobot/test/output/terrain_test_suite"
    
    if args.mode:
        # Worker Process
        run_test_case(args.mode, OUTPUT_DIR)
    else:
        # Master Process
        # 1. Individual Tests
        individual_types = ["flat", "rough", "slope", "stairs_up", "stairs_down", "discrete", "stepping_stones"]
        
        # 2. Combined Test
        # 3. Curriculum Test
        all_modes = individual_types + ["combined", "curriculum"]
        
        logger.info(f"Starting Test Suite. Modes: {all_modes}")
        logger.info(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")
        
        for mode in all_modes:
            logger.info(f"=== Running Subtest: {mode} ===")
            # Run self in a subprocess to ensure clean Gym environment for each test
            cmd = [sys.executable, __file__, "--mode", mode]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Subtest {mode} failed with exit code {e.returncode}")
        
        logger.success("All tests completed.")

