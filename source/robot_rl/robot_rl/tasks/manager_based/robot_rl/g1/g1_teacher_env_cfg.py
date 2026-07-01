# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 teacher policy: open-loop motion-clip tracking environment (flat ground, v1).

Tracks pre-generated fixed motion clips (see
``scripts/rsl_rl/generate_clips.py`` and the ``clips/`` package) instead of
the online re-anchored multi-skill command.  No velocity conditioner, no
contact gates, no skill cross-fades.  Clips are stored in a single clip-world
frame and the CLF runs in the absolute world frame (the same frame as the
tracker task), so the policy's reference observations are
base-relative / heading-invariant while the privileged critic sees the raw
world-frame tracking vectors.  A single base-pose-error termination replaces
the upright / height / frame-drift terminations.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_clf_tracking_base import (
    G1ClfTrackingEnvCfg,
    G1ClfTrackingEventsCfg,
    G1ClfTrackingRewardCfg,
)
from ..mdp.commands.traj_tracking.motion_sequence_cmd_cfg import MotionSequenceCommandCfg

# Default clip library (generate with scripts/rsl_rl/generate_clips.py at the
# env control dt, e.g. --dt 0.02).  Override per-run via the cfg.  Clean
# reference clips only -- no sim-recorded clips in this paradigm.
DEFAULT_CLIP_PATH = "clips/g1_flat_v1.h5"

# Single deterministic parity clip (stand -> walk -> run -> stand; see
# scripts/rsl_rl/generate_parity_clip.py).  Shared with the tracker parity-play
# task; the teacher parity-play variant tracks it open-loop for a quick, clean
# look at how the teacher follows a known profile (and IsaacLab <-> MuJoCo checks).
DEFAULT_PARITY_CLIP_PATH = "clips/g1_parity_v1.h5"


##
# Commands
##
@configclass
class G1TeacherCommandsCfg:
    """Single open-loop clip-tracking command; no velocity conditioner."""

    traj_ref = MotionSequenceCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        path=DEFAULT_CLIP_PATH,
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,  # training: off (enabled in the -play variant)
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        # Ego-centric receding-horizon reference: half the envs have their clip
        # periodically re-placed onto the robot's root (planar pos + heading), so
        # the reference looks like an MPC plan that re-plans every few steps. The
        # rest keep the fixed world-frame clip. Set ego_anchor_fraction=0 to disable.
        ego_anchor_fraction=0.5,
        ego_anchor_every=10,  # re-plan every 10 control steps (~0.2 s at dt=0.02)
    )


##
# Observations
##
@configclass
class G1TeacherObservationsCfg:
    """World-frame-aware observations: deployable policy, privileged critic.

    The CLF tracking vectors are world-frame (absolute position / heading), so
    the raw ``ref_traj``/``act_traj``/``*_vel`` observations live in the
    critic-only group; the policy's reference terms (base-pose error, joint
    refs, ee refs, future windows) are all relative / heading-invariant, so the
    policy never sees a world coordinate.  Shared with
    :class:`G1TrackerEnvCfg` (one frame across both tasks)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable proprioception + base-relative reference (no world coords)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        # Reference (current step only), base-relative. The reference *motion* terms
        # carry observation noise so the teacher learns to track a noisy reference --
        # the generator (FM) predicts these at deploy and its output is not perfectly
        # clean. The command terms (command_vel / future_command_vel) stay clean: at
        # deploy they come from the joystick / scripted profile, not the FM.
        base_pose_error = ObsTerm(func=mdp.base_pose_error, params={"command_name": "traj_ref"},
                                  noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_pos_ref = ObsTerm(func=mdp.joint_pos_ref, params={"command_name": "traj_ref"},
                                noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel_ref = ObsTerm(func=mdp.joint_vel_ref, params={"command_name": "traj_ref"},
                                noise=Unoise(n_min=-0.5, n_max=0.5), clip=(-20.0, 20.0))
        ee_ref = ObsTerm(func=mdp.ee_ref_relative_to_base, params={"command_name": "traj_ref"},
                         noise=Unoise(n_min=-0.02, n_max=0.02))
        # Reference base velocity (current step), rotated into the base frame
        # (heading-invariant, deployable). The FM predicts this (the achieved-reference
        # velocity, not the command); noised like the other reference-motion terms.
        base_vel_ref = ObsTerm(func=mdp.ref_base_velocity_local, params={"command_name": "traj_ref"},
                               noise=Unoise(n_min=-0.1, n_max=0.1))
        # Joystick-level velocity command that generated the clip (target_vel =
        # (vx, vy, wz)); pairs with the xy_vel / yaw_vel tracking rewards. No noise:
        # this is the upstream command, not the FM-predicted reference.
        command_vel = ObsTerm(func=mdp.command_velocity, params={"command_name": "traj_ref"})

        # Look-ahead into the clip (next 0.5 s, subsampled every 5 steps): future
        # joystick command + future base pose/velocity, so the teacher can
        # anticipate upcoming slowdowns / turns. Base pose/vel are relative to the
        # current actual base (a "future base_pose_error"); noised like the
        # current-step reference (the FM predicts these too).
        future_command_vel = ObsTerm(
            func=mdp.future_command_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )
        future_base_pose = ObsTerm(
            func=mdp.future_base_pose,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        future_base_vel = ObsTerm(
            func=mdp.future_base_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
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
        # Joystick-level velocity command that generated the clip (target_vel).
        command_vel = ObsTerm(func=mdp.command_velocity, params={"command_name": "traj_ref"})

        # Look-ahead into the clip (next 0.5 s, subsampled every 5 steps), same as
        # the policy group (future command + base pose/vel rel. to current base).
        future_command_vel = ObsTerm(
            func=mdp.future_command_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )
        future_base_pose = ObsTerm(
            func=mdp.future_base_pose,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )
        future_base_vel = ObsTerm(
            func=mdp.future_base_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )

        # Full world-frame tracking vector (privileged -- absolute coords).
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
# Events
##
@configclass
class G1TeacherEventsCfg(G1ClfTrackingEventsCfg):
    """Reuse dynamics randomization; swap the reset onto a clip."""

    reset_on_ref = None  # remove the velocity-conditioned reference reset
    reset_on_clip = EventTerm(
        func=mdp.reset_on_motion_sequence,
        mode="reset",
        params={"command_name": "traj_ref", "base_frame_name": "pelvis_link"},
    )


##
# Rewards
##
@configclass
class G1TeacherRewardCfg(G1ClfTrackingRewardCfg):
    """CLF tracking + penalties; drop the velocity-command rewards and add a
    dense reference-base-velocity tracking term for shaping."""

    xy_vel = None
    yaw_vel = None

    # Dense shaped signal replacing the dropped xy_vel/yaw_vel: track the clip's
    # joystick-level velocity command (target_vel = (vx, vy, wz)) -- the same
    # signal a velocity joystick would supply -- rather than the achieved
    # reference base velocity.  Split into planar (xy) + yaw exactly like the
    # stock track_lin_vel_xy_exp / track_ang_vel_z_exp, but sourced from traj_ref.
    xy_vel = RewTerm(
        func=mdp.command_lin_vel_xy_tracking_exp,
        weight=1.0,
        params={"command_name": "traj_ref", "std": 0.5},
    )
    yaw_vel = RewTerm(
        func=mdp.command_ang_vel_z_tracking_exp,
        weight=1.0,
        params={"command_name": "traj_ref", "std": 0.5},
    )


##
# Terminations
##
@configclass
class G1TeacherTerminationCfg:
    """One failure termination (base drift) + clip-end / time-out truncations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    clip_end = DoneTerm(
        func=mdp.clip_end_reached, params={"command_name": "traj_ref"}, time_out=True
    )
    base_pose_error = DoneTerm(
        func=mdp.base_pose_error_termination,
        params={"command_name": "traj_ref", "max_pos_dist": 0.75, "max_ori_rad": 0.8},
    )


##
# Environment
##
@configclass
class G1TeacherEnvCfg(G1ClfTrackingEnvCfg):
    """Flat-ground open-loop motion-sequence tracking env."""

    observations: G1TeacherObservationsCfg = G1TeacherObservationsCfg()
    commands: G1TeacherCommandsCfg = G1TeacherCommandsCfg()
    rewards: G1TeacherRewardCfg = G1TeacherRewardCfg()
    events: G1TeacherEventsCfg = G1TeacherEventsCfg()
    terminations: G1TeacherTerminationCfg = G1TeacherTerminationCfg()

    def __post_init__(self):
        super().__post_init__()
        # Flat ground for v1 (override the rough-terrain generator).
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Episode length matches a 20 s clip at the 0.02 s control dt
        # (decimation 4 * sim.dt 0.005); regenerate clips at this dt.
        self.episode_length_s = 20.0

        # --- Parity with the multiskill baseline (g1_terrain_multiskill) ---
        # (1) No external pushes (the baseline disables them).
        self.events.push_robot = None
        # (2) Reward-knob parity: stronger undesired-contact penalty + tighter
        #     CLF normalization (baseline used these).
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.clf_reward.params["max_eta_err"] = 5.0
        self.rewards.clf_decreasing_condition.params["eta_max"] = 5.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 10.0


@configclass
class G1TeacherEnvCfgPlay(G1TeacherEnvCfg):
    """Smaller, noise-free variant for play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the reference-trajectory markers in the viewer.
        self.commands.traj_ref.debug_vis = True


@configclass
class G1TeacherParityPlayEnvCfg(G1TeacherEnvCfgPlay):
    """Deterministic single-clip playback of the trained teacher.

    Tracks the hand-authored parity clip (stand -> walk -> run -> stand) open-loop
    from step 0 in a single env, so you can watch how the teacher follows a known
    profile (and reuse it for IsaacLab <-> MuJoCo parity).  Ego re-anchoring is
    turned OFF here so the reference placement is fixed and the playback is
    deterministic; override ``env.commands.traj_ref.ego_anchor_fraction=1.0`` on
    the play command line to instead watch the receding-horizon re-anchoring on
    the parity clip.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.traj_ref.path = DEFAULT_PARITY_CLIP_PATH
        self.commands.traj_ref.random_start_phase = False   # start at clip step 0
        self.commands.traj_ref.randomize_heading = False
        self.commands.traj_ref.ego_anchor_fraction = 0.0    # fixed placement (deterministic)
