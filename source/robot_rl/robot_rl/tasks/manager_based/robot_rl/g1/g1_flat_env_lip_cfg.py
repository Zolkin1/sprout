from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg


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
        self.curriculum.clf_curriculum = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None




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
