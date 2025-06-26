import math
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.command.cmd_cfg import HZDCommandCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg 
##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

class G1FlatHZDCommandsCfg(CommandsCfg):

     hzd_ref = HZDCommandCfg()


##
# Environment configuration
##
@configclass
class G1FlatHZDEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    commands: G1FlatHZDCommandsCfg = G1FlatHZDCommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class G1FlatRefTrackingEnvCfg(G1FlatHZDEnvCfg):
    """Configuration for the G1 Flat environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # self.rewards.clf_reward = None
        self.rewards.clf_decreasing_condition = None
        self.curriculum.clf_curriculum = None

class G1FlatHZDVdotEnvCfg(G1FlatHZDEnvCfg):
    """Configuration for the G1 Flat environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.clf_decreasing_condition = None
        self.curriculum.clf_curriculum = None
        self.rewards.clf_vdot_tanh = RewTerm(
            func=mdp.clf_decreasing_condition,
            weight=2.0,
            params={
                "command_name": "hlip_ref",
            }
        )

class G1FlatHZDEnvCfg_PLAY(G1FlatHZDEnvCfg):
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
