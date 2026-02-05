
import sys
import os
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

def test_visual_terrain():
    logger.info("Starting Visual Terrain Test...")
    
    # 1. Configure Scenario
    # We need a minimal scenario for IsaacGymHandler to work
    scenario_cfg = ScenarioCfg(
        simulator="isaacgym",
        headless=True,  # Use headless but with cameras enabled
        create_ground=False, # Disable default ground plane, we inject terrain
        scene=SceneCfg(),
        robots=[
            LeapCfg(fix_base_link=True)
        ],
        cameras=[
            PinholeCameraCfg(
                name="Camera_Overview",
                pos=(-10.0, -10.0, 20.0),
                look_at=(15.0, 15.0, 0.0), # Assuming a 30x30 grid
                data_types=["rgb"],
                width=1920,
                height=1080
            ),
            PinholeCameraCfg(
                name="Camera_Detail",
                pos=(5.0, 5.0, 5.0),
                look_at=(8.0, 8.0, 0.0),
                data_types=["rgb"],
                width=1024,
                height=1024
            )
        ],
        lights=[
            DistantLightCfg(
                polar=45.0,
                azimuth=45.0,
                intensity=1000.0 # Make sure we can see
            )
        ]
    )
    
    # 2. Configure Terrain
    # 6 rows (difficulty), 7 cols (types) to cover all algorithms
    terrain_cfg = TerrainCfg()
    terrain_cfg.mesh_type = "trimesh" # Force trimesh for visual test
    terrain_cfg.num_rows = 6
    terrain_cfg.num_cols = 7
    terrain_cfg.terrain_length = 6.0
    terrain_cfg.terrain_width = 6.0
    terrain_cfg.horizontal_scale = 0.1
    terrain_cfg.vertical_scale = 0.005
    terrain_cfg.border_size = 2.0
    terrain_cfg.max_init_terrain_level = 5
    
    # Explicitly set proportions to ensure all types appear
    # The generator chooses based on probability, so we map cols to types implicitly
    # by ensuring the "choice" variable in generator maps to our desired types.
    # To strictly control "Column 0 = Type A", we might need to hack the proportions
    # or rely on the generator's deterministic "col / num_cols" logic.
    #
    # Generator logic: choice = col / num_cols
    # If we set proportions to be equal segments, we can predict it.
    
    types = ["flat", "rough", "slope", "stairs_up", "stairs_down", "discrete", "stepping_stones"]
    # We have 7 types. num_cols = 7.
    # col 0: choice ~ 0/7 = 0.0
    # col 1: choice ~ 1/7 = 0.14
    # ...
    # We want proportions to match these bins. Each bin size = 1/7.
    
    terrain_cfg.terrain_proportions = {t: 1.0 for t in types}
    # If we pass equal weights, normalized probs are all 1/7.
    # Cumulative probs will be [1/7, 2/7, ... 1.0].
    # So col 0 (0 to 1/7) -> Type 0. Perfect.
    
    logger.info(f"Terrain Config: {terrain_cfg.num_rows}x{terrain_cfg.num_cols} grid.")
    
    # 3. Initialize Handler
    logger.info("Initializing Handler...")
    handler = IsaacgymHandler(scenario_cfg)
    handler.launch() # This creates the sim and viewer (if not headless) or windowless ctx
    
    # 4. Generate and Inject Terrain
    logger.info("Generating Terrain...")
    # NOTE: "isaacgym" matches the simulator string in ScenarioCfg
    gen = TerrainGenerator(terrain_cfg, simulator="isaacgym")
    gen.bind_handler(handler)
    gen() # This calls inject()
    
    logger.info("Terrain Injected. Starting Simulation loop for rendering...")
    
    # 5. Run Sim to Capture Images
    # We need to step the physics/graphics
    # Ideally wait a few frames for auto-exposure (if enabled) or just to settle
    
    output_dir = "MyRobot/test/output/terrain_test"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(10):
        handler.simulate()
    
    # 6. Capture Images
    logger.info("Capturing images...")
    
    # handler.get_camera_data() is the standard API? Checking Scenario/Handler structure
    # MyRobot's handler might wrap this.
    # Let's assume standard Metasim API: handler.get_camera_images() -> dict[cam_name, dict[type, data]]
    # Or scenario object has access.
    
    # Looking at RoboVerse codebase... usually it's handler.get_states() which includes sensors?
    # Or explicitly handler.get_image() ?
    
    # Let's try to find how cameras are accessed.
    # metasim/sim/base.py defines interface.
    # Let's assume we can trigger a render and get data.
    
    # Investigating metasim structure roughly:
    # Usually: images = handler.get_images()
    
    try:
        states = handler.get_states()
        cameras = states.cameras
        
        for cam_name, cam_state in cameras.items():
            if cam_state.rgb is not None:
                # Shape [num_envs, H, W, 3]
                rgb = cam_state.rgb[0] # env 0
                
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                
                # Convert to uint8
                if rgb.dtype != np.uint8:
                     if rgb.max() <= 1.5 and rgb.dtype == np.float32: # likely 0-1
                         rgb = (rgb * 255).astype(np.uint8)
                     else:
                         rgb = rgb.astype(np.uint8)
                
                # RGB to BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                safe_name = cam_name.strip("/").replace("/", "_")
                filename = os.path.join(output_dir, f"{safe_name}.png")
                cv2.imwrite(filename, bgr)
                logger.success(f"Saved image: {filename}")
                
    except Exception as e:
        logger.error(f"Failed to capture/save images: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Test Complete.")
    # handler.close() # If available

if __name__ == "__main__":
    test_visual_terrain()
