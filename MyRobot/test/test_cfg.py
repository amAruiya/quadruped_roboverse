"""测试 Leap 机器人配置。"""

from MyRobot.configs.leap_cfg import leap_task_cfg
from MyRobot.utils.helper import task_cfg_to_scenario


def test_leap_task_cfg():
    """测试 LeapTaskCfg 基本属性。"""
    cfg = leap_task_cfg
    
    # 验证初始状态
    assert cfg.init_state.pos == (0.0, 0.0, 0.365)
    assert len(cfg.init_state.default_joint_angles) == 12
    
    # 验证控制配置
    assert cfg.control.control_type == "P"
    assert cfg.control.action_scale == 0.25
    assert "HAA" in cfg.control.stiffness
    assert cfg.control.stiffness["HAA"] == 28.0
    
    # 验证奖励配置
    assert cfg.rewards.base_height_target == 0.355
    assert cfg.rewards.scales.tracking_lin_vel == 1.0
    
    # 验证机器人配置
    assert cfg.robots == "leap"
    assert cfg.simulator == "isaacgym"
    
    print("✓ LeapTaskCfg 基本属性测试通过")


def test_scenario_conversion():
    """测试任务配置到场景配置的转换。"""
    task_cfg = leap_task_cfg
    scenario = task_cfg_to_scenario(task_cfg)
    
    # 验证场景配置
    assert len(scenario.robots) == 1
    assert scenario.robots[0].name == "leap"
    assert scenario.robots[0].num_joints == 12
    assert scenario.num_envs == task_cfg.env.num_envs
    assert scenario.decimation == task_cfg.sim.decimation
    assert scenario.simulator == "isaacgym"
    
    # 验证执行器配置
    robot = scenario.robots[0]
    assert "LF_HAA" in robot.actuators
    assert robot.actuators["LF_HAA"].stiffness == 28.0
    assert robot.actuators["LF_HAA"].damping == 0.8
    assert robot.actuators["LF_HAA"].torque_limit == 33.5
    
    # 验证关节限制
    assert "LF_KFE" in robot.joint_limits
    assert robot.joint_limits["LF_KFE"][0] < robot.joint_limits["LF_KFE"][1]
    
    print("✓ 场景配置转换测试通过")


if __name__ == "__main__":
    test_leap_task_cfg()
    test_scenario_conversion()
    print("\n✅ 所有测试通过!")