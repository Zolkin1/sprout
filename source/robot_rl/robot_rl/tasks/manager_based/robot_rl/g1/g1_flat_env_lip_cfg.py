from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from robot_rl.tasks.manager_based.robot_rl import mdp

##
# Environment configuration
##
@configclass
class G1FlatLipEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        # change terrain to flat
        self.rewards.clf_reward.params = {
            "command_name": "hlip_ref",
            "max_eta_err": 0.3,
        }
        self.rewards.clf_decreasing_condition.params = {
            "command_name": "hlip_ref",
            "alpha": 1.0,
            "eta_max": 0.2,
            "eta_dot_max": 0.3,
        }
        
        self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.clf_curriculum = None

@configclass 
class G1_custom_lip_clf(G1FlatLipEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        #both front and back 1.14
        #just front: 0.616
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616,0.616),
                "operation": "add",
            }
        )


class G1FlatRefTrackingEnvCfg(G1FlatLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # self.rewards.clf_reward = None
        self.curriculum.clf_curriculum = None
        self.rewards.clf_decreasing_condition = None
        self.curriculum.clf_curriculum = None

class G1FlatLipVdotEnvCfg(G1FlatLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.clf_decreasing_condition = None
        self.curriculum.clf_curriculum = None
        self.rewards.clf_vdot_tanh = RewTerm(
            func=mdp.vdot_tanh,
            weight=2.0,
            params={
                "command_name": "hlip_ref",
            }
        )

class G1FlatLipEnvCfg_PLAY(G1FlatLipEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        self.events.push_robot.interval_range_s = (5.0,5.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0,0)} #(-3.14, 3.14)},
