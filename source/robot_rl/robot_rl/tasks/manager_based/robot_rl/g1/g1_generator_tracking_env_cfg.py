# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 flow-matching generator-tracking environment (flat ground).

Tracks a learned flow-matching *generator* instead of (only) the pre-made clips:
a fraction of envs follow the clip reference (the stable teacher signal), the
rest follow the generator's predicted future state. Both sources drive a single
**world-frame** CLF reward, so the tracker can ultimately be deployed sim2sim /
sim2real with the generator + tracker pair (no privileged info in the policy).

Differs from :class:`G1MotionSequenceEnvCfg` in two ways:
  1. the command term is :class:`GeneratorTrackingCommand` (clip + generator mix);
  2. because the CLF tracking vectors are now world-frame (absolute position /
     heading), the raw ``ref_traj``/``act_traj``/``*_vel`` observations move to
     the **critic-only** group, and the policy's reference base velocity is shown
     in the current base frame (heading-invariant). The policy's remaining
     reference terms (base-pose error, joint refs, ee refs) are already relative.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_motion_sequence_env_cfg import (
    DEFAULT_CLIP_PATH,
    G1MotionSeqCommandsCfg,
    G1MotionSequenceEnvCfg,
)
from ..mdp.commands.traj_tracking.generator_tracking_cmd_cfg import GeneratorTrackingCommandCfg

# Default flow-matching generator checkpoint (override per-run via the cfg / CLI).
DEFAULT_FM_CHECKPOINT = "logs/flow_matching/latest/best.pt"


##
# Commands
##
@configclass
class G1GeneratorTrackingCommandsCfg:
    """Clip + flow-matching-generator tracking command."""

    traj_ref = GeneratorTrackingCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        path=DEFAULT_CLIP_PATH,
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        # --- generator-tracking knobs ---
        fm_checkpoint=DEFAULT_FM_CHECKPOINT,
        clip_fraction=0.35,
        fm_update_every=1,
    )


##
# Observations
##
@configclass
class G1GeneratorTrackingObservationsCfg:
    """World-frame-aware observations: deployable policy, privileged critic.

    The policy never sees a world position or absolute heading: its reference
    terms are all relative (base-pose error, joint refs, ee refs, base-frame
    reference velocity). The critic additionally gets the raw world-frame CLF
    tracking vectors and a clip-vs-FM source flag.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable proprioception + base-relative reference (no world coords)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        # Reference (current step), all base-relative / heading-invariant.
        base_pose_error = ObsTerm(func=mdp.base_pose_error, params={"command_name": "traj_ref"})
        joint_pos_ref = ObsTerm(func=mdp.joint_pos_ref, params={"command_name": "traj_ref"})
        joint_vel_ref = ObsTerm(func=mdp.joint_vel_ref, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))
        ee_ref = ObsTerm(func=mdp.ee_ref_relative_to_base, params={"command_name": "traj_ref"})
        # Reference base velocity rotated into the current base frame (deployable).
        ref_base_vel = ObsTerm(
            func=mdp.ref_base_velocity_local, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic: clean state + world-frame CLF tracking vectors."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        root_quat = ObsTerm(func=mdp.root_quat_w, scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        base_pose_error = ObsTerm(func=mdp.base_pose_error, params={"command_name": "traj_ref"})
        joint_pos_ref = ObsTerm(func=mdp.joint_pos_ref, params={"command_name": "traj_ref"})
        joint_vel_ref = ObsTerm(func=mdp.joint_vel_ref, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))
        ee_ref = ObsTerm(func=mdp.ee_ref_relative_to_base, params={"command_name": "traj_ref"})
        # World-frame reference base velocity (privileged) + the local one (parity with policy).
        ref_base_vel = ObsTerm(func=mdp.ref_base_velocity, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))

        # Full world-frame stance tracking vector (privileged -- absolute coords).
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Environment
##
@configclass
class G1GeneratorTrackingEnvCfg(G1MotionSequenceEnvCfg):
    """Flat-ground generator-tracking env (clip + flow-matching mix, world-frame CLF)."""

    observations: G1GeneratorTrackingObservationsCfg = G1GeneratorTrackingObservationsCfg()
    commands: G1GeneratorTrackingCommandsCfg = G1GeneratorTrackingCommandsCfg()


@configclass
class G1GeneratorTrackingEnvCfgPlay(G1GeneratorTrackingEnvCfg):
    """Smaller, noise-free variant for play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the (clip) reference-trajectory markers in the viewer.
        self.commands.traj_ref.debug_vis = True
