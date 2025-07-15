from .hzd_cmd import JointTrajectoryHZDCommandTerm, EndEffectorTrajectoryHZDCommandTerm
from .hzd_stair_cmd import HZDStairCommandTerm
from .hzd_gait_library_cmd import GaitLibraryHZDCommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from typing import Union

HZD_Q_weights = [
    500.0, 1.0,  # left_hip_pitch_joint
    500.0, 1.0,  # right_hip_pitch_joint
    10.0, 1.0,  # waist_yaw_joint
    100.0, 1.0,  # left_hip_roll_joint
    100.0, 1.0,  # right_hip_roll_joint
    25.0, 1.0,  # left_shoulder_pitch_joint
    25.0, 1.0,  # right_shoulder_pitch_joint
    20.0, 1.0,  # left_hip_yaw_joint
    20.0, 1.0,  # right_hip_yaw_joint
    20.0, 1.0,  # left_shoulder_roll_joint
    20.0, 1.0,  # right_shoulder_roll_joint
    500.0, 1.0,  # left_knee_joint
    500.0, 1.0,  # right_knee_joint
    15.0, 1.0,  # left_shoulder_yaw_joint
    15.0, 1.0,  # right_shoulder_yaw_joint
    100.0, 1.0,  # left_ankle_pitch_joint
    100.0, 1.0,  # right_ankle_pitch_joint
    10.0, 1.0,  # left_elbow_joint
    10.0, 1.0,  # right_elbow_joint
    250.0, 1.0,  # left_ankle_roll_joint
    250.0, 1.0,  # right_ankle_roll_joint
]


HZD_R_weights = [
    0.1,  # left_hip_pitch_joint
    0.1,  # right_hip_pitch_joint
    0.1,  # waist_yaw_joint
    0.1,  # left_hip_roll_joint
    0.1,  # right_hip_roll_joint
    0.01,  # left_shoulder_pitch_joint
    0.01,  # right_shoulder_pitch_joint
    0.1,  # left_hip_yaw_joint
    0.1,  # right_hip_yaw_joint
    0.01,  # left_shoulder_roll_joint
    0.01,  # right_shoulder_roll_joint
    0.1,  # left_knee_joint
    0.1,  # right_knee_joint
    0.01,  # left_shoulder_yaw_joint
    0.01,  # right_shoulder_yaw_joint
    0.1,  # left_ankle_pitch_joint
    0.1,  # right_ankle_pitch_joint
    0.01,  # left_elbow_joint
    0.01,  # right_elbow_joint
    0.1,  # left_ankle_roll_joint
    0.1,  # right_ankle_roll_joint
]


@configclass
class JointTrajectoryHZDCommandCfg(CommandTermCfg):
    """
    Configuration for the JointTrajectoryHZDCommandTerm.
    """
    class_type: type = JointTrajectoryHZDCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    trajectory_tracking_visualizer_cfg: dict = {}
    Q_weights = HZD_Q_weights
    R_weights = HZD_R_weights




HZD_EE_Q_weights = [
    300.0,   200.0,    # com_x pos, vel
    600.0,   50.0,   # com_y pos, vel
    600.0,  20.0,  # com_z pos, vel
    600.0,    20.0,    # pelvis_roll pos, vel
    450.0,    10.0,    # pelvis_pitch pos, vel
    500.0,    30.0,    # pelvis_yaw pos, vel
    2500.0, 125.0,  # swing_x pos, vel
    1700.0,  125.0,  # swing_y pos, vel
    8500.0, 120.0,   # swing_z pos, vel
    200.0,    1.0,    # swing_ori_roll pos, vel
    400.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    300.0,    10.0,    # waist_yaw pos, vel
    400.0,1.0, #swing hand palm pos x
    50.0,10.0, #swing hand palm pos y
    50.0,1.0, #swing hand palm pos z
    50.0,1.0, #swing hand palm yaw
    400.0,1.0, #stance hand palm pos x
    50.0,10.0, #stance hand palm pos y
    50.0,1.0, #stance hand palm pos z
    50.0,1.0, #stance hand palm yaw
]


HZD_EE_R_weights = [
        0.1, 0.1, 0.1,    # CoM inputs: allow moderate effort
        0.05,0.05,0.05,   # pelvis inputs: lower torque priority
        0.05,0.05,0.05,   # swing foot linear inputs
        0.02,0.02,0.02,    # swing foot orientation inputs: small adjustments
        0.1,0.01,0.01,
        0.01,0.01,0.01,
        0.01,0.01,0.01,
    ]




@configclass
class EndEffectorTrajectoryHZDCommandCfg(CommandTermCfg):
    """
    Configuration for the EndEffectorTrajectoryHZDCommandTerm.
    """
    class_type: type = EndEffectorTrajectoryHZDCommandTerm
    yaml_path: str = "source/robot_rl/robot_rl/assets/robots/single_support_config_solution_ee.yaml"
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    trajectory_tracking_visualizer_cfg: dict = {}
    Q_weights = HZD_EE_Q_weights
    R_weights = HZD_EE_R_weights


# Alias for backward compatibility - defaults to joint trajectory
HZDCommandCfg = JointTrajectoryHZDCommandCfg


@configclass
class HZDStairCommandCfg(CommandTermCfg):
    """
    Configuration for the HZDStairCommandTerm.
    """
    class_type: type = HZDStairCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    Q_weights = HZD_Q_weights
    R_weights = HZD_R_weights



HZD_EE_Q_weights_GL = [
    25.0,   250.0,    # com_x pos, vel
    300.0,   50.0,   # com_y pos, vel
    250.0,   10.0,  # com_z pos, vel
    300.0,    20.0,    # pelvis_roll pos, vel
    250.0,    10.0,    # pelvis_pitch pos, vel
    300.0,    30.0,    # pelvis_yaw pos, vel
    4500.0, 125.0,  # swing_x pos, vel
    1500.0,  50.0,  # swing_y pos, vel
    8500.0, 100.0,   # swing_z pos, vel
    30.0,    1.0,    # swing_ori_roll pos, vel
    50.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    100.0,    1.0,    # waist_yaw pos, vel
    40.0,1.0, #left shoulder pitch
    30.0,1.0, #left shoulder roll
    50.0,1.0, #left shoulder yaw
    5.0,1.0, #left elbow
    40.0,1.0, #right shoulder pitch
    30.0,1.0, #right shoulder roll
    50.0,1.0, #right shoulder yaw
    5.0,1.0, #right elbow
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

    # Gait library specific parameters
    trajectory_type: str = "end_effector"  # "joint" or "end_effector"
    gait_library_path: str = "source/robot_rl/robot_rl/assets/robots/gait_library/"
    config_name: str = "single_support_config"  # Base name for configuration files
    gait_velocity_ranges: Union[dict, tuple] = (0.1, 0.2, 0.1)  # (min_vel, max_vel, step) in m/s



HZD_EE_Q_weights_GL_STAIR = [
    1000.0,   200.0,    # com_x pos, vel
    300.0,   50.0,   # com_y pos, vel
    400.0,   10.0,  # com_z pos, vel
    400.0,    20.0,    # pelvis_roll pos, vel
    250.0,    10.0,    # pelvis_pitch pos, vel
    500.0,    30.0,    # pelvis_yaw pos, vel
    4500.0, 50.0,  # swing_x pos, vel
    1500.0,  50.0,  # swing_y pos, vel
    8500.0, 100.0,   # swing_z pos, vel
    30.0,    1.0,    # swing_ori_roll pos, vel
    100.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    100.0,    1.0,    # waist_yaw pos, vel
    30.0,1.0, #swing hand palm pos x
    20.0,1.0, #swing hand palm pos y
    20.0,1.0, #swing hand palm pos z
    1.0,1.0, #swing hand palm yaw
    30.0,1.0, #stance hand palm pos x
    20.0,1.0, #stance hand palm pos y
    20.0,1.0, #stance hand palm pos z
    1.0,1.0, #stance hand palm yaw
]


HZD_EE_R_weights_GL_STAIR = [
        0.1, 0.1, 0.1,    # CoM inputs: allow moderate effort
        0.05,0.05,0.05,   # pelvis inputs: lower torque priority
        0.05,0.05,0.05,   # swing foot linear inputs
        0.02,0.02,0.02,    # swing foot orientation inputs: small adjustments
        0.05,0.01,0.01,
        0.01,0.01,0.01,
        0.01,0.01,0.01,
    ]
@configclass
class StairGaitLibraryHZDCommandCfg(CommandTermCfg):
    """
    Configuration for the StairGaitLibraryHZDCommandTerm (height-based gait selection).
    """
    class_type: type = GaitLibraryHZDCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    trajectory_tracking_visualizer_cfg: dict = {}
    Q_weights = HZD_EE_Q_weights_GL_STAIR
    R_weights = HZD_EE_R_weights_GL_STAIR

    # Gait library specific parameters
    trajectory_type: str = "end_effector"  # "joint" or "end_effector"
    gait_library_path: str = "source/robot_rl/robot_rl/assets/robots/gait_library/"
    config_name: str = "stair_config_solution"  # Base name for configuration files
    gait_height_ranges: Union[dict, tuple] = (0.01, 0.12, 0.01)  # (min_height, max_height, step) in m