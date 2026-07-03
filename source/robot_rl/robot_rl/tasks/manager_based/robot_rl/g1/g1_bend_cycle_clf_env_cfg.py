# Copyright 2026 Zachary Olkin. All rights reserved.

"""Combined bend-over + bend-up G1 environment (one periodic policy).

The two formerly-separate episodic motions ("bow forward" and "bend back up")
are treated as ONE ``full_periodic`` trajectory over the phasing variable
``phi in [0, 1]``:

- ``phi 0.0 -> 0.5`` : bend over,
- ``phi 0.5 -> 1.0`` : bend back up,
- ``phi = 0`` and ``phi = 1.0`` are the standing pose (the cycle
  stand -> bend -> stand is genuinely periodic); ``phi = 0.5`` is bent over.

A configurable fraction of episodes (``percent_hold_phi``) is a *static hold*
pinned to a phase drawn from ``hold_phi_values`` (standing at 0.0/1.0, bent-over
at 0.5), so the policy also learns to hold those poses.  Holds are realised by
:func:`mdp.reset_bend_cycle` (resets the robot into the held pose) plus the
:class:`PhasedTrajectoryCommand` fixed-phase-hold logic (pins the phase and
zeroes the reference velocity).

Built on the live :class:`G1ClfTrackingEnvCfg` (single-skill
``TrajectoryCommandCfg`` path), not the deprecated ``g1_trajopt_*`` bow/bend
configs.
"""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import (
    TrajectoryCommandCfg,
)

from .g1_clf_tracking_base import G1ClfTrackingCommandCfg, G1ClfTrackingEnvCfg
from .clf_weights import BOWING_Q_weights, BOWING_R_weights


# Single ``full_periodic`` bend-cycle clip: domain_sequence [bend_down, bend_up],
# so phi 0->~0.47 is the bend-over and ~0.47->1.0 the bend-up (durations 1.8s /
# 2.0s), with phi 0 == phi 1.0 == standing.  For the HF loader the path must point
# at the trajectory YAML *file* (the single-skill ``TrajectoryManager`` opens it
# directly; only the multi-skill library loader takes a folder).
BEND_CYCLE_HF_REPO = "zolkin/robot_rl"
BEND_CYCLE_PATH = (
    "trajectories/bow_cycle/2026-07-03_10-34-37_bow_cycle_config/bow_cycle_config_solution.yaml"
)


##
# Commands
##
@configclass
class G1BendCycleCommandsCfg(G1ClfTrackingCommandCfg):
    """Single-skill command driving the combined bend-over/bend-up periodic clip."""

    traj_ref = TrajectoryCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        manager_type="trajectory",
        hf_repo=BEND_CYCLE_HF_REPO,
        path=BEND_CYCLE_PATH,
        conditioner_generator_name="base_velocity",
        Q_weights=BOWING_Q_weights,
        R_weights=BOWING_R_weights,
        # Fixed-phase static holds: with prob ``percent_hold_phi`` an episode is
        # pinned to standing (0.0/1.0) or bent-over (0.5).
        percent_hold_phi=0.3,
        hold_phi_values=[0.0, 0.5, 1.0],
    )


##
# Environment
##
@configclass
class G1BendCycleCLFEnvCfg(G1ClfTrackingEnvCfg):
    """G1 combined bend-cycle CLF tracking environment (flat ground, one policy)."""

    commands: G1BendCycleCommandsCfg = G1BendCycleCommandsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # A few full stand->bend->stand cycles per episode (plus static holds).
        # NOTE: tune to the clip's total period once the final clip is uploaded.
        self.episode_length_s = 10.0

        ##
        # Commands — stationary motion, so zero the base-velocity command.
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        ##
        # Events — reset each env into a bend-cycle pose (free -> phase 0,
        # held -> a phase drawn from ``hold_phi_values``).  Replaces the base
        # ``reset_on_ref`` (which is coupled to the multi-skill velocity
        # conditioner the bend-cycle env does not use).
        ##
        self.events.reset_on_ref = EventTerm(
            func=mdp.reset_bend_cycle,
            mode="reset",
            params={
                "command_name": "traj_ref",
                "base_frame_name": "pelvis_link",
            },
        )

        ##
        # Rewards — retune the CLF terms to the bend (BOWING-weight) scale.
        # The base defaults (max_eta_err=6, decreasing weight -5) are tuned for
        # the looser walking CLF; the bend uses the tight bow settings.
        ##
        self.rewards.clf_reward.params["max_eta_err"] = 0.25
        self.rewards.clf_decreasing_condition.weight = 0.0

        ##
        # Terrain — flat ground, no generator / height scan / curriculum.
        ##
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None


@configclass
class G1BendCycleCLFEcEnvCfg(G1BendCycleCLFEnvCfg):
    """Bend-cycle env with the extra-compute (EC) plate mass on the torso."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        # just front: 0.616 kg
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616, 0.616),
                "operation": "add",
            },
        )


@configclass
class G1BendCycleCLFEnvCfg_PLAY(G1BendCycleCLFEnvCfg):
    """Playback config: few envs, no corruption, no domain randomization."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Disable domain randomization for clean playback.
        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
