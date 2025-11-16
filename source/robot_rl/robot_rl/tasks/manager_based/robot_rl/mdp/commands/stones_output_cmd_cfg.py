import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass


from .stones_output_cmd import StonesOutputCommandTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

Q_weights = [
    300.0,
    50.0,  # com_x pos, vel
    300.0,
    50.0,  # com_y pos, vel
    400.0,
    10.0,  # com_z pos, vel
    420.0,
    20.0,  # pelvis_roll pos, vel
    200.0,
    10.0,  # pelvis_pitch pos, vel
    300.0,
    10.0,  # pelvis_yaw pos, vel
    3500.0,
    125.0,  # swing_x pos, vel
    3500.0,
    125.0,  # swing_y pos, vel
    3500.0,
    100.0,  # swing_z pos, vel
    30.0,
    1.0,  # swing_ori_roll pos, vel
    10.0,
    1.0,  # swing_ori_pitch pos, vel
    400.0,
    10.0,  # swing_ori_yaw pos, vel
    500.0,
    10.0,  # waist_yaw pos, vel
    40.0,
    1.0,  # left shoulder pitch
    40.0,
    1.0,  # right shoulder pitch
    100,
    1.0,  # left shoulder roll
    100,
    1.0,  # right shoulder roll
    50,
    1.0,  # left shoulder yaw
    50,
    1.0,  # right shoulder yaw
    30.0,
    1.0,  # left elbow
    30.0,
    1.0,  # right elbow
]


R_weights = [
    0.1,
    0.1,
    0.1,  # CoM inputs: allow moderate effort
    0.05,
    0.05,
    0.05,  # pelvis inputs: lower torque priority
    0.05,
    0.05,
    0.05,  # swing foot linear inputs
    0.02,
    0.02,
    0.02,  # swing foot orientation inputs: small adjustments
    0.1,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
]


@configclass
class StonesOutputCommandCfg(CommandTermCfg):
    """
    Configuration for the StonesOutputCommandTerm.
    """

    class_type: type = StonesOutputCommandTerm
    asset_name: str = "robot"

    use_stance_foot_pos_as_ref: bool = False 
    yaw_idx: list[int] = [5, 11]
    debug_vis: bool = True  # enable debug visualization
    z_sw_max: float = 0.2  # max swing foot z height (m); this is ankle height so different from actual foot position
    z_sw_min: float = 0.0
    v_history_len: int = 5
    pelv_pitch_ref: float = 0.0
    waist_yaw_ref: float = 0.0
    shoulder_ref: list[float] = [0.16, 0.0, 0.0]
    elbow_ref: float = 0.1
    foot_target_range_y: list[float] = [0.1, 0.5]
    resampling_time_range: tuple[float, float] = (5.0, 15.0)  # Resampling time range in seconds
    use_momentum: bool = True 
    
    
    
    z0: float = 0.72  # CoM height (m)
    y_nom: float = 0.25  # nominal lateral foot offset (m)
    y_sw_min: float = 0.15
    y_sw_max: float = 0.4
    
    E_star: float = 0.6
    eps: float = 0.6 #xCOM position reference; xCOM_target[i]=eps*rel_x[i]
    TSS_max: float = 0.6  # max step time (s)
    TSS_min: float = 0.2  # min step time (s)
    
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
    

    # Visualization configurations
    #yellow foot-size cube
    foottarget_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/foottarget",
        markers={
            "foottarget": sim_utils.CuboidCfg(
                size=(0.2, 0.065, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
            )
        },
    )
    #purple foot-size cube
    swingfoot_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/swingfoot",
        markers={
            "swingfoot": sim_utils.CuboidCfg(
                size=(0.2, 0.065, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.0, 1.0))
            )
        },
    )
    #organe long cube
    currentstone_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/currentstone",
        markers={
            "currentstone": sim_utils.CuboidCfg(
                size=(0.2, 0.5, 0.12), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0))
            )
        },
    )
    
    #red long cube
    nextstone_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/stone",
        markers={
            "nextstone": sim_utils.CuboidCfg(
                size=(0.2, 0.5, 0.12), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            )
        },
    )
    #green long cube
    nextnextstone_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/nextstone",
        markers={
            "nextnextstone": sim_utils.CuboidCfg(
                size=(0.2, 0.5, 0.12), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
            )
        },
    )
    
    originframe_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/originframe",
        markers={
            "originframe": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )}
    )
    
    comrefframe_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/comrefframe",
        markers={
            "comrefframe": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )}
    )

