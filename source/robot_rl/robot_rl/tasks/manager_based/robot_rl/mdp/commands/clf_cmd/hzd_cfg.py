from .hzd_cmd import JointTrajectoryHZDCommandTerm, BaseTrajectoryHZDCommandTerm
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


@configclass
class BaseTrajectoryHZDCommandCfg(CommandTermCfg):
    """
    Configuration for the BaseTrajectoryHZDCommandTerm.
    """
    class_type: type = BaseTrajectoryHZDCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    trajectory_tracking_visualizer_cfg: dict = {}
    Q_weights = HZD_Q_weights
    R_weights = HZD_R_weights


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