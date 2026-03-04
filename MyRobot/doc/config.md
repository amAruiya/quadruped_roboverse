# MyRobot 配置手册

> 本文档索引 MyRobot 中所有配置类及其字段，便于快速定位和修改设置。

---

## 目录

| # | 文件 | 配置类 | 说明 |
|---|------|--------|------|
| [1](#1-任务配置--configstask_cfgpy) | `configs/task_cfg.py` | 18 个基础配置类 | 任务逻辑的核心配置体系 |
| [2](#2-leap-任务配置--configsleap_cfgpy) | `configs/leap_cfg.py` | 3 个继承配置类 | Leap 机器人任务的具体配置 |
| [3](#3-训练配置--configstrain_cfgpy) | `configs/train_cfg.py` | 4 个训练配置类 | PPO 训练参数 |
| [4](#4-机器人配置--robotsleap_cfgpy) | `robots/leap_cfg.py` | 1 个机器人配置类 | Leap 硬件 / URDF 级别配置 |
| [5](#5-地形数据类--terraintypespy) | `terrain/types.py` | 3 个数据类 | 地形生成内部数据结构 |

### 快速跳转

**1. task_cfg.py**：[EnvCfg](#11-envcfg--环境基础配置) · [SimCfg](#12-simcfg--仿真参数) · [ControlCfg](#13-controlcfg--控制参数) · [InitStateCfg](#14-initstatecfg--初始状态) · [CommandRanges](#15-commandranges--命令范围) · [CommandsCfg](#16-commandscfg--命令配置) · [ObsScales](#17-obsscales--观测缩放因子) · [NormalizationCfg](#18-normalizationcfg--归一化配置) · [NoiseScales](#19-noisescales--噪声缩放因子) · [NoiseCfg](#110-noisecfg--噪声配置) · [ObservationCfg](#111-observationcfg--观测配置) · [AssetCfg](#112-assetcfg--资产配置) · [TerrainCfg](#113-terraincfg--地形配置) · [DomainRandCfg](#114-domainrandcfg--域随机化配置) · [CurriculumCfg](#115-curriculumcfg--课程学习配置) · [RewardScales](#116-rewardscales--奖励权重基类默认全-0) · [RewardsCfg](#117-rewardscfg--奖励配置) · [CallbacksCfg](#118-callbackscfg--回调配置) · [BaseTaskCfg](#119-basetaskcfg--总配置整合类)

**2. leap_cfg.py**：[LeapRewardScales](#21-leaprewardscales继承-rewardscales) · [LeapRewardsCfg](#22-leaprewardscfg继承-rewardscfg) · [LeapTaskCfg](#23-leaptaskcfg继承-basetaskcfg)

**3. train_cfg.py**：[PolicyCfg](#31-policycfg--策略网络) · [AlgorithmCfg](#32-algorithmcfg--ppo-算法) · [RunnerCfg](#33-runnercfg--运行器) · [TrainCfg](#34-traincfg--总训练配置)

**4. robots/leap_cfg.py**：[LeapCfg](#41-leapcfg继承-robotcfg)

**5. terrain/types.py**：[HeightField](#51-heightfield) · [TriMesh](#52-trimesh) · [TerrainParams](#53-terrainparams)

---

## 1. 任务配置 — `configs/task_cfg.py`

### 1.1 `EnvCfg` — 环境基础配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_envs` | `int` | `40` | 并行环境数量 |
| `episode_length_s` | `float` | `20.0` | 单集最大时长（秒） |
| `env_spacing` | `float` | `3.0` | 环境间距（米） |
| `send_timeouts` | `bool` | `True` | 是否发送 timeout 信号给 RL 算法 |
| `orientation_termination_threshold` | `float \| None` | `0.5` | 姿态终止阈值，\|projected_gravity_z\| < 此值时视为翻倒。设为 `None` 可禁用 |

### 1.2 `SimCfg` — 仿真参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dt` | `float` | `0.005` | 仿真步长（秒） |
| `decimation` | `int` | `4` | 控制降频因子。控制频率 = 1 / (dt × decimation) |
| `gravity` | `tuple[float,float,float]` | `(0,0,-9.81)` | 重力加速度 (x, y, z) |

### 1.3 `ControlCfg` — 控制参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `control_type` | `Literal["P","V","T"]` | `"P"` | 控制类型：位置 / 速度 / 扭矩 |
| `action_scale` | `float` | `0.25` | 动作缩放因子 |
| `action_offset` | `bool` | `True` | 是否以默认关节位置为动作零点偏移 |
| `stiffness` | `dict[str,float] \| None` | `None` | PD 刚度，按关节名模式匹配。**为 None 时自动从 RobotCfg.actuators 读取** |
| `damping` | `dict[str,float] \| None` | `None` | PD 阻尼，同上。**为 None 时自动从 RobotCfg.actuators 读取** |

### 1.4 `InitStateCfg` — 初始状态

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pos` | `tuple[float,float,float]` | `(0,0,0.5)` | 根体初始位置 (x, y, z) |
| `rot` | `tuple[float,float,float,float]` | `(0,0,0,1)` | 根体初始四元数 (x, y, z, w) |
| `lin_vel` | `tuple[float,float,float]` | `(0,0,0)` | 初始线速度 |
| `ang_vel` | `tuple[float,float,float]` | `(0,0,0)` | 初始角速度 |
| `default_joint_angles` | `dict[str,float] \| None` | `None` | 默认关节角度 `{joint_name: rad}`。**为 None 时自动从 RobotCfg.default_joint_positions 读取** |

### 1.5 `CommandRanges` — 命令范围

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lin_vel_x` | `tuple[float,float]` | `(-1.0, 1.0)` | x 线速度采样范围 |
| `lin_vel_y` | `tuple[float,float]` | `(-0.5, 0.5)` | y 线速度采样范围 |
| `ang_vel_yaw` | `tuple[float,float]` | `(-1.0, 1.0)` | yaw 角速度采样范围 |
| `heading` | `tuple[float,float]` | `(-3.14, 3.14)` | 航向角采样范围 |

### 1.6 `CommandsCfg` — 命令配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_commands` | `int` | `4` | 命令向量维度 |
| `resampling_time` | `float` | `10.0` | 命令重采样周期（秒） |
| `heading_command` | `bool` | `True` | 是否使用航向命令（True 时第 4 维为 heading） |
| `ranges` | `CommandRanges` | `CommandRanges()` | 命令范围子配置 |

### 1.7 `ObsScales` — 观测缩放因子

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lin_vel` | `float` | `2.0` | 线速度缩放 |
| `ang_vel` | `float` | `0.25` | 角速度缩放 |
| `dof_pos` | `float` | `1.0` | 关节位置缩放 |
| `dof_vel` | `float` | `0.05` | 关节速度缩放 |
| `height_measurements` | `float` | `5.0` | 高度测量缩放（地形相关） |

### 1.8 `NormalizationCfg` — 归一化配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `obs_scales` | `ObsScales` | `ObsScales()` | 观测缩放因子子配置 |
| `clip_observations` | `float` | `100.0` | 观测裁剪范围 |
| `clip_actions` | `float` | `100.0` | 动作裁剪范围 |

### 1.9 `NoiseScales` — 噪声缩放因子

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dof_pos` | `float` | `0.01` | 关节位置噪声 |
| `dof_vel` | `float` | `1.5` | 关节速度噪声 |
| `lin_vel` | `float` | `0.1` | 线速度噪声 |
| `ang_vel` | `float` | `0.2` | 角速度噪声 |
| `gravity` | `float` | `0.05` | 重力向量噪声 |
| `height_measurements` | `float` | `0.1` | 高度测量噪声 |

### 1.10 `NoiseCfg` — 噪声配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `add_noise` | `bool` | `True` | 是否添加观测噪声 |
| `noise_level` | `float` | `1.0` | 全局噪声缩放因子 |
| `noise_scales` | `NoiseScales` | `NoiseScales()` | 各项噪声缩放子配置 |

### 1.11 `ObservationCfg` — 观测配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `normalization` | `NormalizationCfg` | `NormalizationCfg()` | 归一化子配置 |
| `noise` | `NoiseCfg` | `NoiseCfg()` | 噪声子配置 |

### 1.12 `AssetCfg` — 资产配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `foot_name` | `str` | `"foot"` | 足部 link 名称模式（用于足部接触检测） |
| `penalize_contacts_on` | `list[str] \| None` | `None` | 接触时施加惩罚的 body 名称列表 |
| `terminate_after_contacts_on` | `list[str] \| None` | `None` | 接触后触发终止的 body 名称列表 |

### 1.13 `TerrainCfg` — 地形配置

**基础参数**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mesh_type` | `Literal["plane","heightfield","trimesh"] \| None` | `"plane"` | 地形类型 |
| `static_friction` | `float` | `1.0` | 静摩擦系数 |
| `dynamic_friction` | `float` | `1.0` | 动摩擦系数 |
| `restitution` | `float` | `0.0` | 恢复系数（弹性） |
| `measure_heights` | `bool` | `False` | 是否采集高度测量作为观测 |
| `curriculum` | `bool` | `False` | 是否启用地形课程学习 |

**网格参数**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `terrain_length` | `float` | `8.0` | 单个地形块长度（米） |
| `terrain_width` | `float` | `8.0` | 单个地形块宽度（米） |
| `num_rows` | `int` | `10` | 地形网格行数（难度方向） |
| `num_cols` | `int` | `10` | 地形网格列数（类型方向） |
| `horizontal_scale` | `float` | `0.1` | 水平分辨率（米/像素） |
| `vertical_scale` | `float` | `0.005` | 垂直缩放（米） |
| `border_size` | `float` | `2.0` | 边界平坦区域宽度（米） |
| `env_border_size` | `float` | `0.04` | 单块地形边缘平滑区域（米） |

**地形类型与比例**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `terrain_proportions` | `dict[str,float] \| list[float] \| None` | `None` | 各地形类型比例。字典如 `{"flat":0.2, "rough":0.3}`，列表按顺序 `[flat, rough, slope, stairs_up, stairs_down, discrete, stepping_stones]` |
| `slope_threshold` | `float` | `0.45` | 坡度阈值，超过视为垂直墙面 |
| `max_init_terrain_level` | `int` | `5` | 初始最大地形难度等级 |
| `terrain_smoothness` | `float` | `0.0` | 地形平滑度 (0-1) |

**课程学习**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `curriculum_type` | `Literal["linear","exponential","step"]` | `"linear"` | 课程难度递增方式 |
| `difficulty_scale` | `float` | `1.0` | 难度缩放因子 |

**高度测量**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `measured_points_x` | `int` | `10` | x 方向测量点数 |
| `measured_points_y` | `int` | `10` | y 方向测量点数 |
| `measure_distance_x` | `float` | `1.0` | x 方向测量距离（米） |
| `measure_distance_y` | `float` | `1.0` | y 方向测量距离（米） |

**结构参数**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `step_width` | `float` | `0.4` | 台阶宽度（米） |
| `step_depth` | `float` | `0.4` | 台阶深度（米） |

### 1.14 `DomainRandCfg` — 域随机化配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `randomize_friction` | `bool` | `False` | 是否随机化摩擦系数 |
| `randomize_base_mass` | `bool` | `False` | 是否随机化基座附加质量 |
| `randomize_kp_kd` | `bool` | `False` | 是否随机化 PD 增益 |
| `push_robots` | `bool` | `False` | 是否在训练中施加外部推力扰动 |
| `friction_range` | `tuple[float,float]` | `(0.5, 1.25)` | 摩擦系数范围 |
| `added_mass_range` | `tuple[float,float]` | `(-1.0, 3.0)` | 附加质量范围（kg） |
| `kp_range` | `tuple[float,float]` | `(0.8, 1.2)` | 刚度缩放范围 |
| `kd_range` | `tuple[float,float]` | `(0.8, 1.2)` | 阻尼缩放范围 |

### 1.15 `CurriculumCfg` — 课程学习配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `False` | 是否启用课程学习 |
| `terrain_levels` | `int` | `10` | 地形难度等级总数 |
| `terrain_level_step` | `int` | `1` | 等级步进值 |

### 1.16 `RewardScales` — 奖励权重（基类，默认全 0）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `tracking_lin_vel` | `0.0` | 线速度追踪奖励 |
| `tracking_ang_vel` | `0.0` | 角速度追踪奖励 |
| `lin_vel_z` | `0.0` | z 线速度惩罚（负值使用） |
| `ang_vel_xy` | `0.0` | xy 角速度惩罚 |
| `orientation` | `0.0` | 姿态偏离惩罚 |
| `torques` | `0.0` | 扭矩惩罚 |
| `dof_vel` | `0.0` | 关节速度惩罚 |
| `dof_acc` | `0.0` | 关节加速度惩罚 |
| `action_rate` | `0.0` | 动作变化率惩罚 |
| `feet_air_time` | `0.0` | 足部空中时间奖励 |
| `collision` | `0.0` | 碰撞惩罚 |
| `termination` | `0.0` | 终止惩罚 |
| `dof_pos_limits` | `0.0` | 关节位置超限惩罚 |

### 1.17 `RewardsCfg` — 奖励配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scales` | `RewardScales` | `RewardScales()` | 奖励权重子配置 |
| `only_positive_rewards` | `bool` | `True` | 是否裁剪负奖励为 0 |
| `soft_dof_pos_limit` | `float` | `0.9` | 软关节位置限制阈值 |
| `soft_dof_vel_limit` | `float` | `0.9` | 软关节速度限制阈值 |
| `soft_torque_limit` | `float` | `0.9` | 软扭矩限制阈值 |
| `base_height_target` | `float` | `0.25` | 目标基座高度（米） |
| `max_contact_force` | `float` | `100.0` | 最大允许接触力（N） |

### 1.18 `CallbacksCfg` — 回调配置

每个字段为 `{name: fn | (fn, kwargs)}` 格式的字典。

| 字段 | 触发时机 | 签名 |
|------|----------|------|
| `setup` | `__init__` 时，handler 就绪后 | `fn(task, **kw) -> None` |
| `query` | 注册查询（ContactForces 等） | `{name: BaseQueryType}` |
| `reset` | `reset(env_ids)` 时 | `fn(task, env_ids, **kw) -> None` |
| `terminate` | 每步终止条件检查 | `fn(task, env_states, **kw) -> BoolTensor` |
| `pre_step` | `step()` 入口，动作处理前 | `fn(task, actions, **kw) -> Tensor` |
| `in_step` | decimation 循环内每次 simulate 前 | `fn(task, step_idx, **kw) -> None` |
| `post_step` | simulate() 后，计算奖励前 | `fn(task, env_states, **kw) -> None` |

### 1.19 `BaseTaskCfg` — 总配置（整合类）

整合所有子配置 + 场景构建参数。

**任务逻辑子配置**

| 字段 | 类型 | 说明 |
|------|------|------|
| `env` | `EnvCfg` | 环境配置 |
| `sim` | `SimCfg` | 仿真配置 |
| `control` | `ControlCfg` | 控制配置 |
| `init_state` | `InitStateCfg` | 初始状态配置 |
| `commands` | `CommandsCfg` | 命令配置 |
| `observation` | `ObservationCfg` | 观测配置 |
| `asset` | `AssetCfg` | 资产配置 |
| `rewards` | `RewardsCfg` | 奖励配置 |
| `terrain` | `TerrainCfg` | 地形配置 |
| `domain_rand` | `DomainRandCfg` | 域随机化配置 |
| `curriculum` | `CurriculumCfg` | 课程学习配置 |
| `callbacks` | `CallbacksCfg` | 回调配置 |

**场景构建参数**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `robots` | `list[RobotCfg\|str] \| RobotCfg \| str` | `[]` | 机器人（字符串在运行时解析为 RobotCfg） |
| `objects` | `list[BaseObjCfg]` | `[]` | 场景物体 |
| `scene` | `SceneCfg \| str \| None` | `None` | 场景配置 |
| `cameras` | `list` | `[]` | 相机列表 |
| `lights` | `list[BaseLightCfg]` | `[DistantLightCfg()]` | 灯光列表 |
| `simulator` | `str \| None` | `None` | 仿真器后端（`"isaacgym"`, `"mujoco"`, `"isaacsim"`, `"genesis"`） |
| `headless` | `bool` | `False` | 是否无头模式 |
| `render` | `RenderCfg \| None` | `RenderCfg()` | 渲染配置 |
| `create_ground` | `bool` | `True` | 是否自动创建地面 |

---

## 2. Leap 任务配置 — `configs/leap_cfg.py`

继承自基类并覆盖默认值，是 Leap 四足机器人的具体任务配置。

### 2.1 `LeapRewardScales`（继承 `RewardScales`）

除继承字段外的 **Leap 新增字段**：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `leg_effort_symmetry` | `-0.00002` | 左右腿功率对称性惩罚 |
| `hip_pos` | `-1.0` | 髋关节位置偏离惩罚 |
| `thigh_pos` | `-0.8` | 大腿位置偏离惩罚 |
| `feet_contact_forces` | `-0.01` | 足部接触力惩罚 |
| `base_height` | `-0.0` | 基座高度惩罚（预留） |
| `stand_still` | `-0.0` | 静止惩罚（预留） |
| `stumble` | `-0.0` | 绊倒惩罚（预留） |

继承字段的 **Leap 覆盖默认值**：

| 字段 | 基类默认 | Leap 默认 |
|------|----------|-----------|
| `tracking_lin_vel` | 0.0 | **1.0** |
| `tracking_ang_vel` | 0.0 | **0.5** |
| `lin_vel_z` | 0.0 | **-2.0** |
| `ang_vel_xy` | 0.0 | **-0.05** |
| `orientation` | 0.0 | **-0.2** |
| `torques` | 0.0 | **-0.0002** |
| `dof_acc` | 0.0 | **-2.5e-7** |
| `action_rate` | 0.0 | **-0.01** |
| `dof_pos_limits` | 0.0 | **-10.0** |
| `feet_air_time` | 0.0 | **1.0** |
| `collision` | 0.0 | **-1.0** |

### 2.2 `LeapRewardsCfg`（继承 `RewardsCfg`）

| 字段 | 类型 | Leap 默认 | 说明 |
|------|------|-----------|------|
| `scales` | `LeapRewardScales` | `LeapRewardScales()` | 使用 Leap 特有权重 |
| `base_height_target` | `float` | `0.355` | Leap 目标基座高度 |
| `ema_alpha` | `float` | `0.4` | **新增**：leg_effort_symmetry 的 EMA 平滑系数 |

### 2.3 `LeapTaskCfg`（继承 `BaseTaskCfg`）

以下为 Leap 覆盖的关键子配置：

| 子配置 | 关键覆盖值 |
|--------|-----------|
| `sim` | `dt=0.001`, `decimation=4` → 控制频率 250Hz |
| `init_state` | `pos=(0,0,0.365)`, default_joint_angles 自动从 RobotCfg 读取 |
| `control` | stiffness/damping 自动从 RobotCfg.actuators 读取, `action_scale=0.25` |
| `terrain` | `mesh_type="trimesh"`, `curriculum=True`, 7 类地形比例 |
| `commands` | `lin_vel_x=(-1.5,1.5)`, `lin_vel_y=(-1.0,1.0)` |
| `domain_rand` | 全部启用：摩擦/质量/PD增益/外推力 |
| `asset` | `foot_name="FOOT"`, 惩罚 calf 接触，终止 base/thigh 接触 |
| `robots` | `"leap"` |
| `simulator` | `"isaacgym"` |

---

## 3. 训练配置 — `configs/train_cfg.py`

### 3.1 `PolicyCfg` — 策略网络

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `init_noise_std` | `float` | `1.0` | 初始动作噪声标准差 |
| `actor_hidden_dims` | `list[int]` | `[512, 256, 128]` | Actor MLP 隐藏层维度 |
| `critic_hidden_dims` | `list[int]` | `[512, 256, 128]` | Critic MLP 隐藏层维度 |
| `activation` | `str` | `"elu"` | 激活函数（可选：elu/relu/selu/crelu/lrelu/tanh/sigmoid） |

### 3.2 `AlgorithmCfg` — PPO 算法

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `value_loss_coef` | `float` | `1.0` | 价值损失系数 |
| `use_clipped_value_loss` | `bool` | `True` | 是否裁剪价值损失 |
| `clip_param` | `float` | `0.2` | PPO clip ε |
| `entropy_coef` | `float` | `0.01` | 熵正则化系数 |
| `num_learning_epochs` | `int` | `5` | 每次收集后的学习 epoch 数 |
| `num_mini_batches` | `int` | `4` | mini-batch 数量 |
| `learning_rate` | `float` | `1e-3` | 学习率 |
| `schedule` | `str` | `"adaptive"` | 学习率调度（adaptive / fixed） |
| `gamma` | `float` | `0.99` | 折扣因子 |
| `lam` | `float` | `0.95` | GAE λ |
| `desired_kl` | `float` | `0.01` | adaptive 调度的目标 KL 散度 |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪最大范数 |

### 3.3 `RunnerCfg` — 运行器

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `policy_class_name` | `str` | `"ActorCritic"` | 策略类名 |
| `algorithm_class_name` | `str` | `"PPO"` | 算法类名 |
| `num_steps_per_env` | `int` | `24` | 每环境每迭代采集步数 |
| `max_iterations` | `int` | `1500` | 最大训练迭代数 |
| `save_interval` | `int` | `50` | 模型保存间隔（迭代） |
| `experiment_name` | `str` | `"test_myrobot"` | 实验名称（日志目录名） |
| `run_name` | `str` | `""` | 运行名称 |
| `resume` | `bool` | `False` | 是否恢复训练 |
| `load_run` | `str` | `""` | 要加载的运行名称 |
| `checkpoint` | `int` | `-1` | 要加载的检查点编号（-1 = 最新） |
| `resume_path` | `str \| None` | `None` | 恢复路径（自动从上两项推断） |

### 3.4 `TrainCfg` — 总训练配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `runner` | `RunnerCfg` | `RunnerCfg()` | 运行器配置 |
| `algorithm` | `AlgorithmCfg` | `AlgorithmCfg()` | 算法配置 |
| `policy` | `PolicyCfg` | `PolicyCfg()` | 策略网络配置 |
| `obs_groups` | `dict[str,list[str]]` | `{"policy":["policy"], "critic":["critic"]}` | 观测分组，用于 Teacher-Student 架构 |

---

## 4. 机器人配置 — `robots/leap_cfg.py`

### 4.1 `LeapCfg`（继承 `RobotCfg`）

Leap 四足机器人（4 腿 × 3 关节 = 12 DOF）的硬件级配置。

**基础信息**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `name` | `"leap"` | 机器人标识名 |
| `num_joints` | `12` | 关节总数 |
| `urdf_path` | `roboverse_data/robots/Leap/urdf/Leap.urdf` | URDF 路径 |
| `mjcf_path` | `roboverse_data/robots/Leap/mjcf/Leap.xml` | MJCF 路径 |
| `usd_path` | `roboverse_data/robots/Leap/usd/Leap.usd` | USD 路径 |

**物理属性**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `scale` | `(1.0, 1.0, 1.0)` | 整体缩放 |
| `enabled_gravity` | `True` | 启用重力 |
| `fix_base_link` | `False` | 基座自由（非固定） |
| `enabled_self_collisions` | `False` | 关闭自碰撞 |
| `collapse_fixed_joints` | `True` | 折叠固定关节 |

**执行器通用参数**（12 个关节均相同）

| 参数 | 值 | 说明 |
|------|----|------|
| `stiffness` | `28.0` | PD 刚度 (N·m/rad) |
| `damping` | `0.8` | PD 阻尼 (N·m·s/rad) |
| `torque_limit` | `33.5` | 扭矩上限 (N·m) |
| `velocity_limit` | `21.0` | 速度上限 (rad/s) |

**关节限位**（弧度）

| 关节组 | 最小值 | 最大值 |
|--------|--------|--------|
| `*_HAA` | -0.803 | 0.803 |
| `*_HFE` | -1.047 | 4.189 |
| `*_KFE` | -3.197 | -0.616 |

**默认关节位置**（弧度）

| 关节 | LF | LH | RF | RH |
|------|----|----|----|----|
| HAA | -0.2 | -0.2 | 0.2 | 0.2 |
| HFE | 0.64 | 0.64 | 0.64 | 0.64 |
| KFE | -1.27 | -1.27 | -1.27 | -1.27 |

---

## 5. 地形数据类 — `terrain/types.py`

这些是地形生成系统的内部数据结构，一般不需要直接修改。

### 5.1 `HeightField`

| 字段 | 类型 | 说明 |
|------|------|------|
| `heights` | `np.ndarray` | 高度数据 shape=(rows, cols)，单位：米 |
| `horizontal_scale` | `float` | 水平分辨率（米/像素） |
| `vertical_scale` | `float` | 垂直缩放因子 |
| `origin` | `tuple[float,float,float]` | 原点坐标，默认 `(0,0,0)` |

### 5.2 `TriMesh`

| 字段 | 类型 | 说明 |
|------|------|------|
| `vertices` | `np.ndarray` | 顶点 shape=(N, 3) |
| `triangles` | `np.ndarray` | 三角面索引 shape=(M, 3) |
| `origin` | `tuple[float,float,float]` | 原点坐标，默认 `(0,0,0)` |

### 5.3 `TerrainParams`

地形生成器的内部参数，由 `TerrainConfigParser` 从 `TerrainCfg` 自动解析。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `terrain_type` | `Literal[...]` | *(必填)* | flat / rough / slope / slope_rough / stairs_up / stairs_down / discrete / stepping_stones / gap / pit |
| `difficulty` | `float` | `0.5` | 难度 (0-1) |
| `slope` | `float` | `0.0` | 斜坡坡度 |
| `step_height` | `float` | `0.0` | 台阶高度（米） |
| `step_width` | `float` | `0.4` | 台阶宽度（米） |
| `step_depth` | `float` | `0.4` | 台阶深度（米） |
| `discrete_obstacles_height` | `float` | `0.0` | 离散障碍物高度 |
| `stepping_stones_size` | `float` | `1.0` | 踏脚石尺寸 |
| `stone_distance` | `float` | `0.1` | 踏脚石间距 |
| `gap_size` | `float` | `0.0` | 缺口尺寸 |
| `pit_depth` | `float` | `0.0` | 坑深度 |
| `platform_size` | `float` | `3.0` | 平台尺寸 |

---

## 配置层级总览

BaseTaskCfg # task_cfg.py
├── EnvCfg # 环境
├── SimCfg # 仿真
├── ControlCfg # 控制
├── InitStateCfg # 初始状态
├── CommandsCfg # 命令
│ └── CommandRanges # 命令范围
├── ObservationCfg # 观测
│ ├── NormalizationCfg # 归一化
│ │ └── ObsScales # 缩放因子
│ └── NoiseCfg # 噪声
│ └── NoiseScales # 噪声缩放
├── AssetCfg # 资产
├── RewardsCfg # 奖励
│ └── RewardScales # 权重
├── TerrainCfg # 地形
├── DomainRandCfg # 域随机化
├── CurriculumCfg # 课程学习
├── CallbacksCfg # 回调
└── 场景参数 (robots, simulator...) # 场景构建

TrainCfg # train_cfg.py
├── RunnerCfg # 运行器
├── AlgorithmCfg # PPO 算法
└── PolicyCfg # 策略网络

LeapCfg (→ RobotCfg) # leap_cfg.py
# 硬件级：URDF/执行器/关节限位/默认位置