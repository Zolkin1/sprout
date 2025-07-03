from .hzd_cmd import JointTrajectoryHZDCommandTerm, EndEffectorTrajectoryHZDCommandTerm
from .hzd_stair_cmd import HZDStairCommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

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
    100.0,   200.0,    # com_x pos, vel
    300.0,   50.0,   # com_y pos, vel
    600.0,  20.0,  # com_z pos, vel
    420.0,    20.0,    # pelvis_roll pos, vel
    200.0,    10.0,    # pelvis_pitch pos, vel
    500.0,    30.0,    # pelvis_yaw pos, vel
    2500.0, 125.0,  # swing_x pos, vel
    1700.0,  125.0,  # swing_y pos, vel
    6000.0, 120.0,   # swing_z pos, vel
    30.0,    1.0,    # swing_ori_roll pos, vel
    200.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    300.0,    10.0,    # waist_yaw pos, vel
    50.0,1.0, #swing hand palm pos x
    200.0,10.0, #swing hand palm pos y
    50.0,1.0, #swing hand palm pos z
    50.0,1.0, #swing hand palm yaw
    50.0,1.0, #stance hand palm pos x
    200.0,10.0, #stance hand palm pos y
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