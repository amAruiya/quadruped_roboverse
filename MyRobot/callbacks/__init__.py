"""回调模块。

提供各生命周期的回调函数。
"""

from .in_step import push_robots
from .post_step import command_curriculum, push_by_setting_velocity, terrain_curriculum
from .pre_step import action_clip, action_smoothing
from .reset import random_root_state, reset_joints_by_scale
from .setup import example_setup
from .terminate import (
    orientation_termination,
    root_height_below_minimum,
    time_out_termination,
    undesired_contact_termination,
)

__all__ = [
    # Setup
    "example_setup",

    # Reset
    "random_root_state",
    "reset_joints_by_scale",

    # Pre-step
    "action_clip",
    "action_smoothing",

    # In-step
    "push_robots",

    # Post-step
    "push_by_setting_velocity",
    "terrain_curriculum",
    "command_curriculum",
    
    # Terminate
    "orientation_termination",
    "undesired_contact_termination",
    "time_out_termination",
    "root_height_below_minimum",
]