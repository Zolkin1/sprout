from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .hzd_stair_base import HZDStairBaseCommandTerm
from .hzd_stair_joint import HZDStairJointCommandTerm
from .hzd_stair_ee import HZDStairEECommandTerm
from .hzd_cfg import HZD_Q_weights, HZD_R_weights


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
class HZDStairBaseCommandCfg(CommandTermCfg):
    """Base configuration for HZD stair command terms."""
    class_type: type = HZDStairBaseCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_outputs: int = 21
    bez_deg: int = 5
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    debug_vis: bool = False
    Q_weights = HZD_Q_weights
    R_weights = HZD_R_weights


@configclass
class HZDStairJointCommandCfg(HZDStairBaseCommandCfg):
    """Configuration for the HZDStairJointCommandTerm."""
    class_type: type = HZDStairJointCommandTerm
    Q_weights = HZD_Q_weights
    R_weights = HZD_R_weights


@configclass
class HZDStairEECommandCfg(HZDStairBaseCommandCfg):
    """Configuration for the HZDStairEECommandTerm."""
    class_type: type = HZDStairEECommandTerm
    Q_weights = HZD_EE_Q_weights
    R_weights = HZD_EE_R_weights 