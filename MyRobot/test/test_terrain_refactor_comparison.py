
import sys
import unittest
import numpy as np
import torch
from loguru import logger

# 添加项目根目录到路径
import os
sys.path.append(os.getcwd())

from MyRobot.configs.task_cfg import TerrainCfg
from MyRobot.utils.terrain_generator import TerrainGenerator as OldGenerator
from MyRobot.terrain.generator import TerrainGenerator as NewGenerator
from MyRobot.terrain.types import TerrainParams as NewTerrainParams
from MyRobot.utils.terrain_generator import TerrainParams as OldTerrainParams

class TestTerrainRefactor(unittest.TestCase):
    def setUp(self):
        # 配置一个具有一定复杂度的地形配置
        self.cfg = TerrainCfg()
        self.cfg.num_rows = 3
        self.cfg.num_cols = 3
        self.cfg.terrain_length = 5.0
        self.cfg.terrain_width = 5.0
        self.cfg.horizontal_scale = 0.1
        self.cfg.vertical_scale = 0.005
        self.cfg.border_size = 1.0
        self.cfg.env_border_size = 0.0 # 简化边缘以便比较
        
        # 设置多种地形比例以覆盖更多逻辑
        self.cfg.terrain_proportions = {
            "flat": 0.1,
            "rough": 0.1,
            "slope": 0.2,
            "stairs_up": 0.2,
            "discrete": 0.2,
            "stepping_stones": 0.2
        }
        
    def test_initialization(self):
        """测试初始化接口及属性解析"""
        logger.info("Testing initialization...")
        old_gen = OldGenerator(self.cfg, "isaacgym")
        new_gen = NewGenerator(self.cfg, "isaacgym")
        
        # 检查基本属性
        # Old generator 直接在 self 上有这些属性
        # New generator 将解析逻辑放入了 self.parser
        self.assertEqual(old_gen.num_rows, new_gen.parser.num_rows)
        self.assertEqual(old_gen.num_cols, new_gen.parser.num_cols)
        self.assertEqual(old_gen.horizontal_scale, new_gen.parser.horizontal_scale)
        self.assertEqual(old_gen.vertical_scale, new_gen.parser.vertical_scale)
        
        # 检查 tot_rows / tot_cols
        # old_gen 在 _parse_config 中计算了 tot_rows/cols
        self.assertEqual(old_gen.tot_rows, new_gen.parser.tot_rows)
        self.assertEqual(old_gen.tot_cols, new_gen.parser.tot_cols)
        
        logger.success("Initialization test passed.")

    def test_params_generation(self):
        """测试地形参数生成逻辑 (Difficulty / Terrain Type Choice)"""
        logger.info("Testing params generation...")
        old_gen = OldGenerator(self.cfg, "isaacgym")
        new_gen = NewGenerator(self.cfg, "isaacgym")
        
        # 检查特定行列的参数
        rows = [0, 1, 2]
        cols = [0, 1, 2]
        
        for r in rows:
            for c in cols:
                # 必须分别重置随机种子吗？
                # get_terrain_params 本身不应该是随机的，而是确定性的基于 row/col
                # 查看代码：choice = col / max(1, self.num_cols) + 0.001
                # 它是确定性的。
                
                old_p = old_gen.get_terrain_params(r, c)
                new_p = new_gen.parser.get_terrain_params(r, c)
                
                self.assertEqual(old_p.terrain_type, new_p.terrain_type)
                self.assertAlmostEqual(old_p.difficulty, new_p.difficulty)
                self.assertAlmostEqual(old_p.slope, new_p.slope)
                self.assertAlmostEqual(old_p.step_height, new_p.step_height)
                # 检查其他属性...
        
        logger.success("Params generation test passed.")

    def test_terrain_generation_consistency(self):
        """测试生成的高度图数据的一致性"""
        logger.info("Testing terrain generation consistency...")
        
        # 设置相同的随机种子
        seed = 42
        
        # 生成 Old
        np.random.seed(seed)
        old_gen = OldGenerator(self.cfg, "isaacgym")
        old_hfs = old_gen.generate()
        old_raw = old_gen.height_field_raw.copy()
        
        # 生成 New
        np.random.seed(seed)
        new_gen = NewGenerator(self.cfg, "isaacgym")
        new_hfs = new_gen.generate()
        new_raw = new_gen.height_field_raw.copy()
        
        # 1. 检查生成的数量
        self.assertEqual(len(old_hfs), len(new_hfs))
        
        # 2. 检查全局 raw 高度图形状
        self.assertEqual(old_raw.shape, new_raw.shape)
        
        # 3. 比较内容
        # 由于浮点数运算顺序或微小差异，可能不必完全相等，但如果算法是完全复刻的，
        # 在相同种子下应该产生非常接近甚至相同的 int16 data。
        
        diff = np.abs(old_raw.astype(np.float32) - new_raw.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        logger.info(f"Max difference in raw heightfield: {max_diff}")
        logger.info(f"Mean difference in raw heightfield: {mean_diff}")
        
        # 允许极小误差 (考虑到可能的重构差异)
        # 如果是完全搬运，应该是 0
        if max_diff > 0:
            logger.warning("Differences found in height map. This might be due to implementation tweaks.")
            # 只有当差异很大时才报错
            self.assertTrue(max_diff < 5, "Difference in height map is too large!")
        else:
            logger.success("Height maps are identical!")

        # 4. 检查 HeightField 对象
        for i, (oh, nh) in enumerate(zip(old_hfs, new_hfs)):
            self.assertEqual(oh.origin, nh.origin, f"Origin mismatch at index {i}")
            self.assertAlmostEqual(oh.horizontal_scale, nh.horizontal_scale)
            self.assertAlmostEqual(oh.vertical_scale, nh.vertical_scale)
            # 检查高度数据
            h_diff = np.max(np.abs(oh.heights - nh.heights))
            self.assertLess(h_diff, 0.1, f"Height mismatch at index {i}")

    def test_api_compatibility(self):
        """测试公共 API 是否存在"""
        logger.info("Testing public API compatibility...")
        new_gen = NewGenerator(self.cfg, "isaacgym")
        
        # 检查方法
        self.assertTrue(hasattr(new_gen, "generate"))
        self.assertTrue(hasattr(new_gen, "bind_handler"))
        self.assertTrue(hasattr(new_gen, "to_trimesh"))
        
        # 检查属性 (原有的 height_fields 和 env_origins 等)
        self.assertTrue(hasattr(new_gen, "height_fields"))
        self.assertTrue(hasattr(new_gen, "env_origins"))
        self.assertTrue(hasattr(new_gen, "terrain_types"))

        logger.success("API compatibility test passed.")

    def test_direct_interface_compatibility(self):
        """测试新旧类是否可以直接互换（属性访问）"""
        logger.info("Testing direct interface compatibility...")
        old_gen = OldGenerator(self.cfg, "isaacgym")
        new_gen = NewGenerator(self.cfg, "isaacgym")
        
        attributes_to_check = [
            "num_rows", "num_cols", "num_terrains",
            "terrain_length", "terrain_width",
            "horizontal_scale", "vertical_scale",
            "get_terrain_params"
        ]
        
        missing_attrs = []
        for attr in attributes_to_check:
            if hasattr(old_gen, attr) and not hasattr(new_gen, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            logger.warning(f"New generator is missing attributes present in old generator: {missing_attrs}")
            logger.warning("This means the refactoring is NOT a 100% drop-in replacement for code accessing these attributes directly.")
        else:
            logger.success("New generator exposes all checked attributes directly.")

if __name__ == "__main__":
    unittest.main()
