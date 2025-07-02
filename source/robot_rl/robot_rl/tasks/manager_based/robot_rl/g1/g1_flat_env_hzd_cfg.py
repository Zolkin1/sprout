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
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import HZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from .g1_observation import G1FlatHZDObservationsCfg
from isaaclab.sensors import FrameTransformerCfg
#
##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

class G1FlatHZDCommandsCfg(HumanoidCommandsCfg):
     hzd_ref = HZDCommandCfg()



##
# Environment configuration
##
@configclass
class G1FlatHZDEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    commands: G1FlatHZDCommandsCfg = G1FlatHZDCommandsCfg()
    # observations: G1FlatHZDObservationsCfg = G1FlatHZDObservationsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        # self.scene.frame_sensor_cfg = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/pelvis_link",
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link/left_toe"),   # idx 0
        #         FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link/right_toe"), # idx 1
        #         FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link"),            # idx 2
        #         FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link"),             # idx 3
        #     ]
        # )



        # change the observation command name to hzd_ref
        self.observations.critic.foot_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.foot_ang_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj_vel.params["command_name"] = "hzd_ref"

        # change the reward command name to hzd_ref
        self.rewards.holonomic_constraint.params["command_name"] = "hzd_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"
        self.rewards.clf_reward.params["command_name"] = "hzd_ref"
        self.rewards.clf_decreasing_condition.params["command_name"] = "hzd_ref"
        

        self.rewards.clf_reward.params["max_clf"] = 30.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 100.0
        self.rewards.clf_decreasing_condition.params["alpha"] = 1.0
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.625,0.625)
        self.commands.base_velocity.ranges.lin_vel_y = (0,0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2,0.2)
        self.commands.base_velocity.ranges.heading = (0,0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0,0)
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
        self.curriculum.clf_curriculum.params["min_val"] = 10.0
        self.curriculum.clf_curriculum.params["min_clf_val"] = 5.0

        # self.events.push_robot = None
        # self.curriculum.clf_curriculum = None


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
                "command_name": "hzd_ref",
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
        self.events.push_robot = None
        # self.events.push_robot.interval_range_s = (5.0,5.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0,0)} #(-3.14, 3.14)},
