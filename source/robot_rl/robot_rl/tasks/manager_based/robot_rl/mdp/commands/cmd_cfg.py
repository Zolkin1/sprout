from .hlip_cmd import HLIPCommandTerm
from .hzd_cmd import HZDCommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
import torch

Q_weights = [
    100.0,   200.0,    # com_x pos, vel
    300.0,   50.0,   # com_y pos, vel
    600.0,  20.0,  # com_z pos, vel
    420.0,    20.0,    # pelvis_roll pos, vel
    200.0,    10.0,    # pelvis_pitch pos, vel
    500.0,    30.0,    # pelvis_yaw pos, vel
    2500.0, 125.0,  # swing_x pos, vel
    1700.0,  125.0,  # swing_y pos, vel
    4000.0, 100.0,   # swing_z pos, vel
    30.0,    1.0,    # swing_ori_roll pos, vel
    10.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    500.0,    10.0,    # waist_yaw pos, vel
    40.0,1.0, #left sholder pitch
    40.0,1.0, #right sholder pitch
    100,1.0, #left sholder roll
    100,1.0, #right sholder roll
    50,1.0, #left sholder yaw
    50,1.0, #right sholder yaw
    30.0,1.0, #left elbow 
    30.0,1.0, #right elbow 
]


R_weights = [
        0.1, 0.1, 0.1,    # CoM inputs: allow moderate effort
        0.05,0.05,0.05,   # pelvis inputs: lower torque priority
        0.05,0.05,0.05,   # swing foot linear inputs
        0.02,0.02,0.02,    # swing foot orientation inputs: small adjustments
        0.1,0.01,0.01,
        0.01,0.01,0.01,
        0.01,0.01,0.01,
    ]
@configclass
class HLIPCommandCfg(CommandTermCfg):
    """
    Configuration for the HLIPCommandTerm.
    """
    class_type: type = HLIPCommandTerm
    asset_name: str = "robot"
    yaw_idx: list[int] = [5,11]
    T_ds: float = 0.0          # double support duration (s)
    z0: float = 0.65           # CoM height (m)
    y_nom: float = 0.25        # nominal lateral foot offset (m)
    gait_period: float = 0.8   # gait cycle period (s)
    debug_vis: bool = False    # enable debug visualization
    z_sw_max: float = 0.1    # max swing foot z height (m); this is ankle height so different from actual foot position
    z_sw_min: float = 0.0
    v_history_len: int = 5
    pelv_pitch_ref: float = 0.0
    waist_yaw_ref: float = 0.0
    shoulder_ref: list[float] = [0.16, 0.0, 0.0]
    elbow_ref: float = 0.1
    foot_target_range_y: list[float] = [0.1,0.5]
    resampling_time_range: tuple[float, float] = (5.0, 15.0)  # Resampling time range in seconds
    # Command sampling ranges
    ranges: dict = {
        "pos_x": (-0.25, 0.25),
        "pos_y": (0.2, 0.3),
        "pos_z": (0.0, 0.5),
        "yaw": (-0.7, 0.7),
        "timing": (0.5, 1.5),
    }

    # Visualization configurations
    footprint_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/footprint",
        markers={
            "foot": sim_utils.CuboidCfg(
                size=(0.1, 0.065, 0.018),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            )
        }
    )

    foot_body_name: str = ".*_ankle_roll_link"
    upper_body_joint_name: list[str] = [
                    "waist_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ]

    Q_weights = Q_weights
    R_weights = R_weights





HZD_Q_weights = [
    500.0, 1.0,  # left_hip_pitch_joint
    500.0, 1.0,  # right_hip_pitch_joint
    10.0, 1.0,  # waist_yaw_joint
    20.0, 1.0,  # left_hip_roll_joint
    20.0, 1.0,  # right_hip_roll_joint
    10.0, 1.0,  # left_shoulder_pitch_joint
    10.0, 1.0,  # right_shoulder_pitch_joint
    10.0, 1.0,  # left_hip_yaw_joint
    10.0, 1.0,  # right_hip_yaw_joint
    10.0, 1.0,  # left_shoulder_roll_joint
    10.0, 1.0,  # right_shoulder_roll_joint
    500.0, 1.0,  # left_knee_joint
    500.0, 1.0,  # right_knee_joint
    10.0, 1.0,  # left_shoulder_yaw_joint
    10.0, 1.0,  # right_shoulder_yaw_joint
    100.0, 1.0,  # left_ankle_pitch_joint
    100.0, 1.0,  # right_ankle_pitch_joint
    10.0, 1.0,  # left_elbow_joint
    10.0, 1.0,  # right_elbow_joint
    20.0, 1.0,  # left_ankle_roll_joint
    20.0, 1.0,  # right_ankle_roll_joint
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
class HZDCommandCfg(CommandTermCfg):
    """
    Configuration for the HZDCommandTerm.
    """
    class_type: type = HZDCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    trajectory_tracking_visualizer_cfg: dict = {}
    Q_weights = HZD_Q_weights
    R_weights = HZD_R_weights
