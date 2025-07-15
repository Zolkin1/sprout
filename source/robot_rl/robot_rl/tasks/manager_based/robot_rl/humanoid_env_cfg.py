# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import isaaclab.sim as sim_utils

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# TODO: Remove all of these dependencies
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (  # Inherit from the base envs
    CommandsCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

from . import mdp


@configclass
class HumanoidActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class HumanoidCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""

    # Command for the set period
    step_period = mdp.commands.GaitPeriodCfg(period_range=(0.8, 0.8), resampling_time_range=(10.0, 10.0))


@configclass
class HumanoidObservationsCfg(ObservationsCfg):
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        base_lin_vel = None  # Removed - no sensor
        height_scan = None  # Removed - not supported yet

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=1, scale=0.25)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=1,
            scale=(2.0, 2.0, 2.0),
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=1, scale=0.05)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=1)

        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"command_name": "step_period"})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"command_name": "step_period"})

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), history_length=1, scale=2.0)
        height_scan = None  # Removed - not supported yet

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HumanoidEventsCfg:
    """Event configuration."""

    randomize_ground_contact_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.25, 1.25),
            "dynamic_friction_range": (0.25, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,  # ensures dynamic friction <= static friction
        },
    )

    # startup
    # TODO: Verify if this has an effect
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # TODO: Decide if we want to keep this
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class HumanoidRewardCfg(RewardsCfg):
    """Reward terms for the MDP."""

    ##
    # Termination
    ###
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    ##
    # Tracking
    ##

    ##
    # Feet
    ##
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    phase_contact = RewTerm(
        func=mdp.phase_contact,
        weight=0.18,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    ##
    # Joint limits
    ##
    # Penalize ankle and knee joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )

    ##
    # Penalize deviation from default of the joints that are not essential for locomotion
    ##
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0,  # -0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )

    ##
    # Torso Height
    ##
    height_torso = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.78},
    )

    undesired_contacts = None

    ##
    # Regularization
    ##
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    ##
    # Feet
    ##
    feet_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.0,
        params={
            "target_height": 0.08,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    contact_no_vel = RewTerm(
        func=mdp.contact_no_vel,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )


##
# Environment configuration
##
@configclass
class HumanoidEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HumanoidRewardCfg = HumanoidRewardCfg()
    observations: HumanoidObservationsCfg = HumanoidObservationsCfg()
    events: HumanoidEventsCfg = HumanoidEventsCfg()
    actions: HumanoidActionsCfg = HumanoidActionsCfg()
    commands: HumanoidCommandsCfg = HumanoidCommandsCfg()
