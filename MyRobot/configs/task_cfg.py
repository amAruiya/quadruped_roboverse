"""任务配置模块。

该模块定义了运动任务所需的所有配置类，包括：
- 环境配置（EnvCfg）
- 仿真配置（SimCfg）
- 控制配置（ControlCfg）
- 初始状态配置（InitStateCfg）
- 命令配置（CommandsCfg）
- 观测配置（ObservationCfg）
- 资产配置（AssetCfg）
- 奖励配置（RewardsCfg）
- 以及预留的扩展配置（TerrainCfg、DomainRandCfg、CurriculumCfg）
"""

from __future__ import annotations

from typing import Callable, Literal

from metasim.queries.base import BaseQueryType
from metasim.utils import configclass
from metasim.scenario.lights import BaseLightCfg, DistantLightCfg
from metasim.scenario.objects import BaseObjCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scene import SceneCfg


# =============================================================================
# 基础配置类
# =============================================================================


@configclass
class EnvCfg:
    """环境基础配置。

    Attributes:
        num_envs: 并行环境数量
        episode_length_s: 单集最大时长（秒）
        env_spacing: 环境间距（米）
        send_timeouts: 是否发送 timeout 信号
    """

    num_envs: int = 4096
    episode_length_s: float = 20.0
    env_spacing: float = 3.0
    send_timeouts: bool = True


@configclass
class SimCfg:
    """仿真参数配置。

    Attributes:
        dt: 仿真步长（秒）
        decimation: 控制频率降采样因子（控制频率 = 1 / (dt * decimation)）
        gravity: 重力加速度 (x, y, z)
    """

    dt: float = 0.005
    decimation: int = 4
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


@configclass
class ControlCfg:
    """控制参数配置。

    Attributes:
        control_type: 控制类型（"P": 位置, "V": 速度, "T": 扭矩）
        action_scale: 动作缩放因子
        action_clip: 动作裁剪范围
        action_offset: 是否使用默认关节位置作为偏移
        stiffness: PD 控制器刚度（按关节名称模式匹配）
        damping: PD 控制器阻尼（按关节名称模式匹配）
        torque_limits_factor: 扭矩限制缩放因子
        soft_joint_pos_limit_factor: 软关节位置限制因子
        soft_joint_vel_limit_factor: 软关节速度限制因子
    """

    control_type: Literal["P", "V", "T"] = "P"
    action_scale: float = 0.25
    action_clip: float = 100.0
    action_offset: bool = True
    stiffness: dict[str, float] | None = None
    damping: dict[str, float] | None = None
    torque_limits_factor: float = 1.0
    soft_joint_pos_limit_factor: float = 0.9
    soft_joint_vel_limit_factor: float = 0.9


@configclass
class InitStateCfg:
    """初始状态配置。

    Attributes:
        pos: 初始根位置 (x, y, z)
        rot: 初始根旋转四元数 (x, y, z, w)
        lin_vel: 初始线速度 (x, y, z)
        ang_vel: 初始角速度 (x, y, z)
        default_joint_angles: 默认关节角度字典 {joint_name: angle}
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.5)
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_joint_angles: dict[str, float] | None = None


@configclass
class CommandRanges:
    """命令范围配置。

    Attributes:
        lin_vel_x: x 方向线速度范围 (min, max)
        lin_vel_y: y 方向线速度范围 (min, max)
        ang_vel_yaw: yaw 角速度范围 (min, max)
        heading: 航向角范围 (min, max)
    """

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)
    heading: tuple[float, float] = (-3.14, 3.14)


@configclass
class CommandsCfg:
    """命令配置。

    Attributes:
        num_commands: 命令向量维度（通常为 3 或 4）
        resampling_time: 命令重采样周期（秒）
        heading_command: 是否使用航向命令
        ranges: 命令范围配置
    """
    num_commands: int = 4
    resampling_time: float = 10.0
    heading_command: bool = True
    ranges: CommandRanges = CommandRanges()


@configclass
class ObsScales:
    """观测缩放因子。

    Attributes:
        lin_vel: 线速度缩放
        ang_vel: 角速度缩放
        dof_pos: 关节位置缩放
        dof_vel: 关节速度缩放
        height_measurements: 高度测量缩放（地形用）
    """

    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    height_measurements: float = 5.0


@configclass
class NormalizationCfg:
    """观测归一化配置。

    Attributes:
        obs_scales: 观测缩放因子
        clip_observations: 观测裁剪范围
        clip_actions: 动作裁剪范围
    """

    obs_scales: ObsScales = ObsScales()
    clip_observations: float = 100.0
    clip_actions: float = 100.0


@configclass
class NoiseScales:
    """观测噪声缩放因子。

    Attributes:
        dof_pos: 关节位置噪声
        dof_vel: 关节速度噪声
        lin_vel: 线速度噪声
        ang_vel: 角速度噪声
        gravity: 重力向量噪声
        height_measurements: 高度测量噪声
    """

    dof_pos: float = 0.01
    dof_vel: float = 1.5
    lin_vel: float = 0.1
    ang_vel: float = 0.2
    gravity: float = 0.05
    height_measurements: float = 0.1


@configclass
class NoiseCfg:
    """观测噪声配置。

    Attributes:
        add_noise: 是否添加噪声
        noise_level: 噪声等级（全局缩放）
        noise_scales: 各观测项的噪声缩放
    """

    add_noise: bool = True
    noise_level: float = 1.0
    noise_scales: NoiseScales = NoiseScales()


@configclass
class ObservationCfg:
    """观测配置。

    Attributes:
        normalization: 归一化配置
        noise: 噪声配置
    """

    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg()


@configclass
class AssetCfg:
    """资产配置。

    Attributes:
        foot_name: 足部 link 名称模式（用于足部检测）
        penalize_contacts_on: 需要惩罚接触的 body 名称列表
        terminate_after_contacts_on: 接触后终止的 body 名称列表
    """

    foot_name: str = "foot"
    penalize_contacts_on: list[str] | None = None
    terminate_after_contacts_on: list[str] | None = None


# =============================================================================
# 预留配置类（框架级）
# =============================================================================


@configclass
class TerrainCfg:
    """地形配置（预留接口）。

    注意：
        - ScenarioCfg.scene 只负责简单地面（plane/none）
        - TaskCfg.terrain 声明任务级地形需求
        - TerrainRandomizer 在 setup_callback 中执行地形创建

    Attributes:
        mesh_type: 地形类型（"plane": 平面, "heightfield": 高度场, "trimesh": 三角网格）
        static_friction: 静摩擦系数
        dynamic_friction: 动摩擦系数
        restitution: 恢复系数
        measure_heights: 是否测量高度
        curriculum: 是否启用地形课程学习
        
        # 地形网格参数
        terrain_length: 单个地形块长度（米）
        terrain_width: 单个地形块宽度（米）
        num_rows: 地形网格行数
        num_cols: 地形网格列数
        horizontal_scale: 水平分辨率（米/像素）
        vertical_scale: 垂直缩放（米）
        border_size: 边界平坦区域大小（米）
        
        # 地形类型比例
        terrain_proportions: 各类型地形比例字典，例如：
            {"flat": 0.2, "rough": 0.3, "stairs": 0.2, "slope": 0.3}
            如果为 None，则使用单一地形类型
        
        # 地形参数范围
        slope_threshold: 坡度阈值（度）
        max_init_terrain_level: 初始最大地形难度等级
        terrain_smoothness: 地形平滑度（0-1）
        
        # 课程学习
        curriculum_type: 课程学习类型（"linear", "exponential", "step"）
        difficulty_scale: 难度缩放因子
        
        # 高度测量
        measured_points_x: x方向测量点数
        measured_points_y: y方向测量点数
        measure_distance_x: x方向测量距离（米）
        measure_distance_y: y方向测量距离（米）
    """
    mesh_type: Literal["plane", "heightfield", "trimesh"] | None = "plane"
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    measure_heights: bool = False
    curriculum: bool = False
    
    # 地形网格参数
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int = 10
    num_cols: int = 10
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 2.0
    
    # 地形类型比例
    terrain_proportions: dict[str, float] | None = None
    
    # 地形参数范围
    slope_threshold: float = 30.0
    max_init_terrain_level: int = 5
    terrain_smoothness: float = 0.0
    
    # 课程学习
    curriculum_type: Literal["linear", "exponential", "step"] = "linear"
    difficulty_scale: float = 1.0
    
    # 高度测量
    measured_points_x: int = 10
    measured_points_y: int = 10
    measure_distance_x: float = 1.0
    measure_distance_y: float = 1.0

    step_width: float = 0.4
    step_depth: float = 0.4
    
    env_border_size: float = 0.04
    """单个地形块的边缘平滑区域大小(米) - 新增参数"""


@configclass
class DomainRandCfg:
    """域随机化配置（预留接口）。

    Attributes:
        randomize_friction: 是否随机化摩擦系数
        randomize_base_mass: 是否随机化基座质量
        randomize_kp_kd: 是否随机化 PD 增益
        push_robots: 是否施加外部推力扰动
        friction_range: 摩擦系数范围 (min, max)
        added_mass_range: 添加质量范围 (min, max)
        kp_range: 刚度范围 (min, max)
        kd_range: 阻尼范围 (min, max)
    """

    randomize_friction: bool = False
    randomize_base_mass: bool = False
    randomize_kp_kd: bool = False
    push_robots: bool = False
    friction_range: tuple[float, float] = (0.5, 1.25)
    added_mass_range: tuple[float, float] = (-1.0, 3.0)
    kp_range: tuple[float, float] = (0.8, 1.2)
    kd_range: tuple[float, float] = (0.8, 1.2)


@configclass
class CurriculumCfg:
    """课程学习配置。

    Attributes:
        enabled: 是否启用课程学习
        terrain_levels: 地形难度等级数量
        terrain_level_step: 地形等级步进
    """
    # TODO: 后续可扩展更多课程学习参数
    enabled: bool = False
    terrain_levels: int = 10
    terrain_level_step: int = 1


@configclass
class RewardScales:
    """奖励权重配置。

    所有权重默认为 0，需在具体任务中覆盖。

    Attributes:
        # 追踪奖励
        tracking_lin_vel: 线速度追踪奖励
        tracking_ang_vel: 角速度追踪奖励

        # 正则化惩罚
        lin_vel_z: z 方向线速度惩罚
        ang_vel_xy: xy 方向角速度惩罚
        orientation: 姿态偏离惩罚
        torques: 扭矩惩罚
        dof_vel: 关节速度惩罚
        dof_acc: 关节加速度惩罚
        action_rate: 动作变化率惩罚

        # 其他
        feet_air_time: 足部空中时间奖励
        collision: 碰撞惩罚
        termination: 终止惩罚
        dof_pos_limits: 关节位置限制惩罚
    """

    # 追踪奖励
    tracking_lin_vel: float = 0.0
    tracking_ang_vel: float = 0.0

    # 正则化惩罚
    lin_vel_z: float = 0.0
    ang_vel_xy: float = 0.0
    orientation: float = 0.0
    torques: float = 0.0
    dof_vel: float = 0.0
    dof_acc: float = 0.0
    action_rate: float = 0.0

    # 其他
    feet_air_time: float = 0.0
    collision: float = 0.0
    termination: float = 0.0
    dof_pos_limits: float = 0.0


@configclass
class RewardsCfg:
    """奖励配置。

    Attributes:
        scales: 奖励权重
        only_positive_rewards: 是否只保留正奖励
        soft_dof_pos_limit: 软关节位置限制阈值
        soft_dof_vel_limit: 软关节速度限制阈值
        soft_torque_limit: 软扭矩限制阈值
        base_height_target: 目标基座高度
        max_contact_force: 最大接触力
    """

    scales: RewardScales = RewardScales()
    only_positive_rewards: bool = True
    soft_dof_pos_limit: float = 0.9
    soft_dof_vel_limit: float = 0.9
    soft_torque_limit: float = 0.9
    base_height_target: float = 0.25
    max_contact_force: float = 100.0


@configclass
class CallbacksCfg:
    """回调配置。

    每个字段为 {name: (func, kwargs) | func} 的字典。

    回调签名约定：
        - setup: fn(task, **kwargs) -> None
        - reset: fn(task, env_ids: Tensor, **kwargs) -> None
        - terminate: fn(task, env_states: TensorState, **kwargs) -> BoolTensor
        - pre_step: fn(task, actions: Tensor, **kwargs) -> Tensor
        - in_step: fn(task, step_idx: int, **kwargs) -> None
        - post_step: fn(task, env_states: TensorState, **kwargs) -> None
        - query: {name: BaseQueryType}

    Attributes:
        setup: 初始化时调用（handler 已就绪）
        query: 查询注册（ContactForces 等）
        reset: 每次 reset 时调用
        terminate: 终止条件检查
        pre_step: step() 开始时，动作处理前
        in_step: decimation 循环内，每次 simulate() 前
        post_step: simulate() 后，计算奖励/观测前
    """

    setup: dict[str, Callable | tuple[Callable, dict]] = {}
    query: dict[str, BaseQueryType] = {}
    reset: dict[str, Callable | tuple[Callable, dict]] = {}
    terminate: dict[str, Callable | tuple[Callable, dict]] = {}
    pre_step: dict[str, Callable | tuple[Callable, dict]] = {}
    in_step: dict[str, Callable | tuple[Callable, dict]] = {}
    post_step: dict[str, Callable | tuple[Callable, dict]] = {}


# =============================================================================
# 整合配置类
# =============================================================================


@configclass
class BaseTaskCfg:
    """运动任务基础配置。

    该配置类整合了所有子配置,用于初始化 BaseLocomotionTask。

    Attributes:
        env: 环境配置
        sim: 仿真配置
        control: 控制配置
        init_state: 初始状态配置
        commands: 命令配置
        observation: 观测配置
        asset: 资产配置
        rewards: 奖励配置
        terrain: 地形配置（预留）
        domain_rand: 域随机化配置（预留）
        curriculum: 课程学习配置（预留）
        callbacks: 回调配置
        
        # 场景构建相关
        robots: 机器人配置（字符串或 RobotCfg 列表）
        objects: 物体列表
        scene: 场景配置
        cameras: 相机列表
        lights: 灯光列表
        simulator: 仿真器类型
        headless: 是否无头模式
        render: 渲染配置
    """

    env: EnvCfg = EnvCfg()
    sim: SimCfg = SimCfg()
    control: ControlCfg = ControlCfg()
    init_state: InitStateCfg = InitStateCfg()
    commands: CommandsCfg = CommandsCfg()
    observation: ObservationCfg = ObservationCfg()
    asset: AssetCfg = AssetCfg()
    rewards: RewardsCfg = RewardsCfg()
    terrain: TerrainCfg = TerrainCfg()
    domain_rand: DomainRandCfg = DomainRandCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    callbacks: CallbacksCfg = CallbacksCfg()
    
    # =========================================================================
    # 场景构建配置（从 ScenarioCfg 迁移）
    # =========================================================================
    robots: list[RobotCfg | str] | RobotCfg | str = []
    objects: list[BaseObjCfg] = []
    scene: SceneCfg | str | None = None
    cameras: list = []
    lights: list[BaseLightCfg] = [DistantLightCfg()]
    
    simulator: str | None = None
    headless: bool = False
    render: RenderCfg | None = RenderCfg()
    create_ground: bool = True