from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg,  # Inherit from the base envs
)

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import (
    G1RoughLipObservationsCfg,
)
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (
    HumanoidCommandsCfg,
    HumanoidEnvCfg,
    HumanoidRewardCfg,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.cmd_cfg import MLIPCommandCfg

##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip

#


@configclass
class G1RoughMlipCommandsCfg(HumanoidCommandsCfg):
    """Commands for the G1 Flat environment."""

    hlip_ref = MLIPCommandCfg()


@configclass
class CurriculumMlipCfg:
    """Curriculum terms for the MDP."""

    clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 1000, "min_val": 20.0})


# Lip specific rewards
##
class G1RoughMlipRewards(HumanoidRewardCfg):
    """Rewards specific to LIP Model"""

    holonomic_constraint = RewTerm(
        func=mdp.holonomic_constraint,
        weight=4.0,
        params={
            "command_name": "hlip_ref",
            "z_offset": 0.036,
        },
    )

    holonomic_constraint_vel = RewTerm(
        func=mdp.holonomic_constraint_vel,
        weight=2.0,
        params={
            "command_name": "hlip_ref",
        },
    )

    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "hlip_ref",
            "max_eta_err": 0.25,
        },
    )

    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={
            "command_name": "hlip_ref",
            "alpha": 0.5,
            "eta_max": 0.2,
            "eta_dot_max": 0.3,
        },
    )


@configclass
class G1RoughMlipEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Flat environment."""

    rewards: G1RoughMlipRewards = G1RoughMlipRewards()
    observations: G1RoughLipObservationsCfg = G1RoughLipObservationsCfg()
    commands: G1RoughMlipCommandsCfg = G1RoughMlipCommandsCfg()
    curriculum: CurriculumMlipCfg = CurriculumMlipCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos[".*_hip_pitch_joint"] = -0.25
        self.scene.robot.init_state.joint_pos[".*_knee_joint"] = 0.5
        self.scene.robot.init_state.joint_pos[".*_ankle_pitch_joint"] = 0.0
        
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        # No height scanner for now
        self.scene.height_scanner = None

        ##
        # Randomization
        ##
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1, 1),
            "y": (-1, 1),
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.4, 0.4),
        }
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.events.base_external_force_torque = None
        ##
        # Commands
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"

        ##
        # Rewards
        ##
        self.rewards.feet_air_time = None
        self.rewards.phase_contact = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.feet_clearance = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.termination_penalty = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.joint_deviation_hip = None
        self.rewards.contact_no_vel = None
        self.rewards.alive = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        # torque, acc, vel, action rate regularization
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_pos_limits.weight = -1.0
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.dof_vel_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.001

        self.rewards.dof_acc_l2 = None
        self.rewards.dof_vel_l2 = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None
        self.rewards.height_torso = None
