from .hzd_gait_library_cmd import GaitLibraryHZDCommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from typing import Union

HZD_EE_Q_weights_GL = [
    25.0,   250.0,    # com_x pos, vel
    300.0,   50.0,   # com_y pos, vel
    350.0,   10.0,  # com_z pos, vel
    300.0,    20.0,    # pelvis_roll pos, vel
    250.0,    10.0,    # pelvis_pitch pos, vel
    300.0,    30.0,    # pelvis_yaw pos, vel
    1500.0, 50.0,  # swing_x pos, vel
    1500.0,  50.0,  # swing_y pos, vel
    1500.0, 50.0,   # swing_z pos, vel
    30.0,    1.0,    # swing_ori_roll pos, vel
    50.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    100.0,    1.0,    # waist_yaw pos, vel
    40.0,1.0, #left shoulder pitch
    40.0,1.0, #left shoulder roll
    50.0,1.0, #left shoulder yaw
    30.0,1.0, #left elbow
    40.0,1.0, #right shoulder pitch
    40.0,1.0, #right shoulder roll
    50.0,1.0, #right shoulder yaw
    30.0,1.0, #right elbow
]


HZD_EE_R_weights_GL = [
        0.1, 0.1, 0.1,    # CoM inputs: allow moderate effort
        0.05,0.05,0.05,   # pelvis inputs: lower torque priority
        0.05,0.05,0.05,   # swing foot linear inputs
        0.02,0.02,0.02,    # swing foot orientation inputs: small adjustments
        0.1,0.01,0.01,
        0.01,0.01,0.01,
        0.01,0.01,0.01,
    ]


@configclass
class GaitLibraryHZDCommandCfg(CommandTermCfg):
    """
    Configuration for the GaitLibraryHZDCommandTerm.
    """
    class_type: type = GaitLibraryHZDCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    trajectory_tracking_visualizer_cfg: dict = {}
    Q_weights = HZD_EE_Q_weights_GL
    R_weights = HZD_EE_R_weights_GL

    clf_domain_scalar = [1.0,1.0, #double support
                        1.0,1.0, #single_support
                        1.0,1.0] #flight_phase

    # Gait library specific parameters
    trajectory_type: str = "end_effector"  # "joint" or "end_effector"
    gait_library_path: str = "source/robot_rl/robot_rl/assets/robots/gait_library/"
    config_name: str = "single_support_config"  # Base name for configuration files
    gait_velocity_ranges: Union[dict, tuple] = (0.1, 0.2, 0.1)  # (min_vel, max_vel, step) in m/s
    use_standing: bool = True


