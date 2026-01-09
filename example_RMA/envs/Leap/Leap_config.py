from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class LeapCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.365] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'LF_HAA': -0.2,   # [rad]
            'LH_HAA': -0.2,   # [rad]
            'RF_HAA': 0.2,   # [rad]
            'RH_HAA': 0.2,   # [rad]

            'LF_HFE': 0.64,     # [rad]
            'LH_HFE': 0.64,     # [rad]
            'RF_HFE': 0.64,     # [rad]
            'RH_HFE': 0.64,     # [rad]

            'LF_KFE': -1.27,  # [rad]
            'LH_KFE': -1.27,  # [rad]
            'RF_KFE': -1.27,  # [rad]
            'RH_KFE': -1.27,  # [rad]
        }
    class terrain( LeggedRobotCfg.terrain ):
        terrain_proportions = [0.3, 0.3, 0.0, 0.1, 0.1, 0.1, 0.1]  

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'HAA': 28., 'HFE': 28. , 'KFE': 28.}  # [N*m/rad]
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}  
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '/opt/rui/Data/quadruped/RMA/resources/robots/Leap_RL/urdf/Leap.urdf'
        name = "leap"
        foot_name = "FOOT"  # 查找所有连杆，将其中包含foot的连杆作为足端，生成feet_name变量
        penalize_contacts_on = [ "calf"]
        terminate_after_contacts_on = ["base", "thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.75]
        randomize_base_mass = True
        added_mass_range = [0., 2.0]
        randomize_kp_kd = True
        kp_range = [20.,35.]
        kd_range = [0.4, 1.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.355
        ema_alpha = 0.4
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            leg_effort_symmetry = -0.00002
            hip_pos = -1.0
            thigh_pos = -0.8
            feet_contact_forces = -0.01

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        heading_command = True
        class ranges:
            lin_vel_x = [-1.5, 1.5]  # [m/s]
            lin_vel_y = [-1.0, 1.0]  # [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # [rad/s]
            heading = [-3.14, 3.14]

    class terrain( LeggedRobotCfg.terrain ):
        num_cols = 20
        num_rows = 15


class LeapCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'leap'
        max_iterations = 20000
        save_interval = 50
        checkpoint_interval = 2000

  