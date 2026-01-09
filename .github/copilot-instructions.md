# RoboVerse AI 编码指南

## 项目概述

RoboVerse 是一个统一的机器人仿真平台，支持**多种物理后端**（IsaacSim、IsaacGym、MuJoCo、Genesis、PyBullet、SAPIEN）并提供通用 API。核心抽象使得编写与仿真器无关的机器人操控代码成为可能。

**当前项目阶段**：复现 `example_RMA/` 的训练逻辑，将其从 IsaacGym 专用后端迁移到 metasim handler，实现多仿真器后端的无缝切换。

## 架构

### 核心模块：`metasim/`
- **`scenario/`**：场景、机器人、物体、相机、灯光的配置类
- **`sim/`**：实现 `BaseSimHandler` 接口的后端处理器
- **`task/`**：继承 `BaseTaskEnv` 的任务定义，支持 Gymnasium 注册
- **`types.py`**：核心数据类型（`TensorState`、`DictEnvState`、`Action`）
- **`constants.py`**：`SimType` 和 `PhysicStateType` 枚举

### 数据流
```
ScenarioCfg → get_handler(scenario) → BaseSimHandler → simulate() → get_states()
                                                     ↓
                                              set_dof_targets(actions)
```

## 关键参考

### `example_RMA/`（重要参考，禁止直接运行）
这是一个清晰的 IsaacGym RL 训练实现示例，**仅供参考逻辑结构**：
- `envs/base/`：基础任务和配置类
- `envs/Leap/`：具体机器人环境实现
- `scripts/`：训练和评估脚本
- `utils/task_registry.py`：任务注册机制

⚠️ **禁止运行此目录下的文件**——它是独立示例，不兼容当前项目环境。

### `roboverse_pack/` 和 `roboverse_learn/`
这两个包实现较为臃肿。编写代码时可参考其中 `unitree_rl` 相关文件的实现思路，但**不要照搬其冗余逻辑**。

### `MyRobot/`（待构建）
我们自己的四足机器人开发包，RL 工作流尚未确立。

## 开发环境

### Conda 环境
| 环境名 | 用途 |
|--------|------|
| `metasim` | IsaacSim 和 MuJoCo |
| `metasim_genesis` | Genesis |
| `metasim_isaacgym` | IsaacGym |

### 常用命令
```bash
# 代码检查（ruff 配置在 pyproject.toml）
ruff check metasim/ --fix
ruff format metasim/

# 运行示例
python get_started/0_static_scene.py --sim mujoco --headless
```

## 关键模式

### 配置类
使用 `@configclass` 装饰器（来自 `metasim.utils.configclass`）：
```python
from metasim.utils import configclass

@configclass
class MyConfig:
    field: str = "default"  # 无需 MISSING，可变默认值自动包装
```

### 仿真器选择
在 `ScenarioCfg` 中设置 `simulator`：
- `"mujoco"` - MuJoCo（metasim 环境）
- `"isaacsim"`, `"isaaclab"` - NVIDIA Isaac（metasim 环境）
- `"isaacgym"` - IsaacGym（metasim_isaacgym 环境）
- `"genesis"` - Genesis（metasim_genesis 环境）

## 代码规范

1. **导入顺序**：使用 IsaacGym 时，`isaacgym` 必须在 `torch` 之前导入
2. **日志**：使用 `loguru.logger` 配合 `RichHandler`
3. **CLI 参数**：使用 `tyro.cli()` 配合 `@configclass`
4. **状态格式**：批量操作用 `TensorState`，单环境用 `DictEnvState`

## 注意事项

- `PhysicStateType.RIGIDBODY` 启用重力+碰撞；`XFORM` 用于静态放置
- 关节名称必须与资产文件精确匹配；使用 `handler.get_joint_names(obj_name)` 验证
- 相机需要显式设置 `data_types=["rgb"]` 才能输出图像
- **本项目不使用 HuggingFace 功能**
- 避免在核心模块中引入多余依赖，保持轻量
- 回答问题必须使用中文
