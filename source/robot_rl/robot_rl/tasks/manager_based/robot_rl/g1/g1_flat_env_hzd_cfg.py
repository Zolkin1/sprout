from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import (
    EndEffectorTrajectoryHZDCommandCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_FOR_FLAT_HZD_CFG

from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


class G1FlatHZDCommandsCfg_EE(HumanoidCommandsCfg):
    hzd_ref = EndEffectorTrajectoryHZDCommandCfg()


##
# Environment configuration
##
@configclass
class G1FlatHZDEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""

    commands: G1FlatHZDCommandsCfg_EE = G1FlatHZDCommandsCfg_EE()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
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

        self.rewards.clf_reward.params["max_clf"] = 100.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 100.0
        self.rewards.clf_decreasing_condition.params["alpha"] = 1.0

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = (0, 0)

        self.commands.step_period.period_range = (0.8, 0.8)
        self.events.reset_base.params["pose_range"]["yaw"] = (0, 0)
        ##
        # Scene
        ##
        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = ROUGH_FOR_FLAT_HZD_CFG
        self.curriculum.terrain_levels = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        self.curriculum.clf_curriculum.params["min_val"] = 10.0
        self.curriculum.clf_curriculum.params["min_clf_val"] = 10.0
        # no terrain curriculum


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
                "command_name": "hzd_ref",  # noqa: E203
            },
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
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (0, 0),
        }  # (-3.14, 3.14)},
