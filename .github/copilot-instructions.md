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

## MyRobot 架构规划

### 模块结构
```
MyRobot/
├─ configs/           # 配置模块
│   ├─ task_cfg.py    # TaskCfg 及所有子配置
│   └─ locomotion_task_cfg.py
├─ callbacks/         # 回调模块（按生命周期分类）
│   ├─ setup/         # terrain_randomizer 等
│   ├─ reset/         # random_state, terrain_level
│   ├─ terminate/     # contact, orientation
│   ├─ pre_step/      # action processing
│   ├─ in_step/       # push_robots 等扰动
│   └─ post_step/     # curriculum
├─ terrain/           # 地形系统
│   ├─ terrain_cfg.py
│   ├─ terrain_generator.py  # 纯算法，生成高度图/网格
│   └─ terrain_randomizer.py # 绑定 handler，注入地形
├─ tasks/             # 任务实现
│   ├─ base_task.py   # BaseLocomotionTask
│   └─ locomotion_task.py
├─ rewards/           # 奖励函数库
├─ runners/           # PPO 训练循环（后续）
└─ scripts/           # train.py, play.py
```

### 配置分离原则
| 配置类型 | 职责 | 使用时机 |
|----------|------|----------|
| `ScenarioCfg` | 场景创建（robot, scene, objects, cameras, lights） | handler.launch() |
| `TaskCfg` | 任务逻辑（env, sim, control, rewards, terrain, domain_rand） | task.__init__() |

**地形配置特殊处理**：
- `ScenarioCfg.scene.ground_type`：只管 `"plane"` / `"none"`（简单地面）
- `TaskCfg.terrain`：声明复杂地形需求（heightfield/trimesh/curriculum）
- `TerrainGenerator`：纯算法类，生成地形数据
- `TerrainRandomizer`：作为 `setup_callback`，绑定 handler 后注入地形

### 回调系统

| 回调类型 | 触发时机 | 签名 |
|----------|----------|------|
| `setup` | `__init__` 时，handler 就绪后 | `fn(task, **kwargs) -> None` |
| `reset` | `reset(env_ids)` 时 | `fn(task, env_ids: Tensor, **kwargs) -> None` |
| `terminate` | 每步检查终止条件 | `fn(task, env_states, **kwargs) -> BoolTensor` |
| `pre_step` | `step()` 开始，动作处理前 | `fn(task, actions, **kwargs) -> Tensor` |
| `in_step` | decimation 循环内 | `fn(task, step_idx, **kwargs) -> None` |
| `post_step` | simulate() 后，计算奖励前 | `fn(task, env_states, **kwargs) -> None` |

### Task 生命周期
```python
__init__(scenario, task_cfg)
    ├─ super().__init__(scenario)     # handler.launch()
    ├─ _parse_cfg(task_cfg)
    ├─ _init_buffers()
    ├─ _prepare_reward_functions()
    └─ _run_setup_callbacks()         # TerrainRandomizer 在此执行

reset(env_ids)
    ├─ _reset_idx(env_ids)
    ├─ _run_reset_callbacks(env_ids)
    └─ _observation() → obs

step(action)
    ├─ _pre_physics_step(action)      # pre_step_callback
    ├─ for _ in range(decimation):
    │      ├─ _in_physics_step()      # in_step_callback
    │      └─ handler.simulate()
    └─ _post_physics_step()
           ├─ _check_termination()    # terminate_callback
           ├─ _compute_reward()
           └─ _run_post_callbacks()   # post_step_callback
```

### 预留扩展接口
| 模块 | 预留字段 | 说明 |
|------|----------|------|
| `TerrainCfg` | `mesh_type`, `curriculum`, `proportions` | 地形类型/课程学习 |
| `DomainRandCfg` | `randomize_friction/mass/kp_kd`, `push_robots` | 域随机化 |
| `CurriculumCfg` | `enabled`, `funcs` | 课程学习函数 |

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

### `MyRobot/`（构建中）
四足机器人开发包，基于 metasim handler 实现后端无关的 RL 训练。

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
- 当你要修改已有文件的时候，必须遵循最小改动原则，尽可能的减少对已有代码的改动
- 相机需要显式设置 `data_types=["rgb"]` 才能输出图像
- **本项目不使用 HuggingFace 功能**
- 避免在核心模块中引入多余依赖，保持轻量
- 回答问题必须使用中文
