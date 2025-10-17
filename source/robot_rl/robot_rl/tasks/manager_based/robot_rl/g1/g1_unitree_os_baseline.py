import math

from isaaclab.utils import configclass

from .g1_rough_env_cfg import G1RoughEnvCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
# from isaaclab_tasks.manager_based.locomotion.velocity import mdp #.mdp import *  # Inherit from the base envs

##
# Environment configuration
##
@configclass
class UnitreeOSRewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0, # -0.1
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    # ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    height_torso = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    # ========================================================================
    # Rewards below are set to weight=0.0 for inheritance compatibility only.
    # These are expected by the parent class G1RoughEnvCfg but are not used
    # in the Unitree OS baseline reward configuration.
    # ========================================================================
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.5,
        },
    )

    contact_no_vel = RewTerm(
        func=mdp.contact_no_vel,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    phase_contact = RewTerm(
        func=mdp.phase_contact,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )


@configclass
class UnitreeOSG1FlatEnvCfg(G1RoughEnvCfg):
    """Configuration for the Unitree Open Source G1 Flat environment."""

    rewards: UnitreeOSRewardsCfg = UnitreeOSRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Rewards - Override parent weights with UnitreeOSRewardsCfg values
        ##
        original_rewards = UnitreeOSRewardsCfg()
        for reward_name in dir(original_rewards):
            if not reward_name.startswith("_"):
                reward_term = getattr(original_rewards, reward_name)
                if hasattr(reward_term, "weight"):
                    current_reward = getattr(self.rewards, reward_name, None)
                    if current_reward is not None:
                        current_reward.weight = reward_term.weight

        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616,0.616),
                "operation": "add",
            }
        )

        ##
        # Scene
        ##
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # TODO: Consider changing weights/params from the rough env

class UnitreeOSG1FlatEnvCfg_PLAY(UnitreeOSG1FlatEnvCfg):
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
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0, 0)}  # (-3.14, 3.14)},

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.75)  # Allow full range
