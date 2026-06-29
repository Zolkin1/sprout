# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 tracker policy: flow-matching generator-tracking environment (flat ground).

Tracks a learned flow-matching *generator* instead of (only) the pre-made clips:
a fraction of envs follow the clip reference (the stable teacher signal), the
rest follow the generator's predicted future state. Both sources drive a single
**world-frame** CLF reward, so the tracker can ultimately be deployed sim2sim /
sim2real with the generator + tracker pair (no privileged info in the policy).

Differs from :class:`G1TeacherEnvCfg` only in the command term
(:class:`GeneratorTrackingCommand`, the clip + generator mix).  Both tasks now
run a world-frame CLF, so they share one observation layout
(:class:`G1TeacherObservationsCfg`): the raw ``ref_traj``/``act_traj``/
``*_vel`` tracking vectors are critic-only (absolute world coords), and the
policy's reference terms (base-pose error, joint refs, ee refs, future windows)
are all relative / heading-invariant.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_teacher_env_cfg import (
    DEFAULT_CLIP_PATH,
    G1TeacherObservationsCfg,
    G1TeacherTerminationCfg,
    G1TeacherEnvCfg,
)
from ..mdp.commands.traj_tracking.generator_tracking_cmd_cfg import GeneratorTrackingCommandCfg

# Default flow-matching generator checkpoint (override per-run via the cfg / CLI).
DEFAULT_FM_CHECKPOINT = "logs/flow_matching/2026-06-21_17-07-59/ckpt_89.pt"

# Single deterministic parity clip (see scripts/rsl_rl/generate_parity_clip.py),
# played open-loop in the clip-only parity variant for IsaacLab <-> MuJoCo checks.
DEFAULT_PARITY_CLIP_PATH = "clips/g1_parity_v1.h5"

# Teacher-recorded counterpart of the parity clip: the motion the teacher actually
# achieved while tracking DEFAULT_PARITY_CLIP_PATH, recorded by
# scripts/rsl_rl/record_teacher_clips.py (--source-clip clips/g1_parity_v1.h5).
# Use this to run the identical sim2sim parity check on a tracker trained against
# teacher-recorded clips (an on-distribution, physically realizable reference).
DEFAULT_TEACHER_PARITY_CLIP_PATH = "clips/g1_teacher_parity_v1.h5"


##
# Commands
##
@configclass
class G1TrackerCommandsCfg:
    """Clip + flow-matching-generator tracking command."""

    traj_ref = GeneratorTrackingCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        path="clips/g1_teacher_v1.h5", #DEFAULT_CLIP_PATH,
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        # --- generator-tracking knobs ---
        fm_checkpoint=DEFAULT_FM_CHECKPOINT,
        clip_fraction=0.2,
        # Half the FM envs are closed-loop: the FM is fed a position-tracking PD
        # command over the clip and the pose-drift termination is clip-relative,
        # so stalling (which the open-loop FM lets the robot reward-hack) gets
        # punished and terminated. The policy sees no flag for the bucket.
        closed_loop_fraction=0.6,
        fm_update_every=2,
    )


##
# Terminations
##
@configclass
class G1TrackerTerminationCfg(G1TeacherTerminationCfg):
    """Inherited base-drift failure + clip-end / time-out truncations, plus a
    reference-independent fall detector.

    The generator may steer envs into states the clip never visits, so we add a
    fall termination that does not depend on the reference: terminate when the
    base drops below ``min_height`` or tilts more than 70 deg from upright (in
    roll or pitch).
    """

    base_fallen = DoneTerm(
        func=mdp.base_fallen,
        params={"min_height": 0.5, "roll_limit_deg": 70.0, "pitch_limit_deg": 70.0},
    )


##
# Environment
##
@configclass
class G1TrackerEnvCfg(G1TeacherEnvCfg):
    """Flat-ground tracker env (clip + flow-matching generator mix, world-frame CLF)."""

    # Shared world-frame observation layout (policy base-relative, critic
    # privileged); only the command term differs from the clip-only task.
    observations: G1TeacherObservationsCfg = G1TeacherObservationsCfg()
    commands: G1TrackerCommandsCfg = G1TrackerCommandsCfg()
    terminations: G1TrackerTerminationCfg = G1TrackerTerminationCfg()


@configclass
class G1TrackerEnvCfgPlay(G1TrackerEnvCfg):
    """Smaller, noise-free variant for play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the (clip) reference-trajectory markers in the viewer.
        self.commands.traj_ref.debug_vis = True


@configclass
class G1TrackerFMAutoregPlayEnvCfg(G1TrackerEnvCfgPlay):
    """Autoregressive, robot-decoupled FM playback (real-time receding horizon).

    Runs the flow-matching generator as a self-contained receding-horizon reference
    source: its history is its OWN past output (seeded from the clip at reset), so it
    never sees the robot (``fm_history_source='reference'``).  Every env runs the FM
    open-loop (``clip_fraction=0``, ``closed_loop_fraction=0``); the tracker policy
    loaded via ``--load_run`` (e.g. the ego teacher) tracks the emitted reference.

    Pass the trained **clip-conditioned** FM with ``--fm_checkpoint <flow_matching_run>/best.pt``
    -- the inherited default is the state-conditioned FM and will not behave correctly
    here.  The clip library only supplies the reset seed + the velocity command
    (``target_vel``); use ``--clip_device cpu`` for the multi-GB library.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.traj_ref.path = DEFAULT_CLIP_PATH       # FM-training clips (seed + command)
        self.commands.traj_ref.clip_fraction = 0.0            # every env runs the FM
        self.commands.traj_ref.closed_loop_fraction = 0.0     # open-loop command (clip target_vel)
        self.commands.traj_ref.fm_history_source = "reference"  # autoregressive, robot-decoupled
        self.commands.traj_ref.fm_anchor_on_prev_traj = False
        self.commands.traj_ref.random_start_phase = False     # seed from clip step 0


@configclass
class G1TrackerParityPlayEnvCfg(G1TrackerEnvCfgPlay):
    """Deterministic clip-only playback for IsaacLab <-> MuJoCo parity checks.

    Plays the single hand-authored parity clip (stand -> walk -> run -> stand)
    open-loop through the *deployed* tracker policy: ``clip_fraction = 1.0`` makes
    every env track the clip (the FM is loaded but never sampled -- the per-env
    source roll ``rand() >= 1.0`` is always False), and ``random_start_phase =
    False`` starts at clip step 0 on the clip's spawn pose, so the run is fully
    deterministic and matches the MuJoCo ``--clip`` clip-only mode reference.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.traj_ref.path = DEFAULT_PARITY_CLIP_PATH
        self.commands.traj_ref.clip_fraction = 1.0          # all envs clip, FM idle
        self.commands.traj_ref.random_start_phase = False   # start at clip step 0
        self.commands.traj_ref.randomize_heading = False


@configclass
class G1TrackerTeacherParityPlayEnvCfg(G1TrackerParityPlayEnvCfg):
    """Parity playback against the *teacher-recorded* parity clip.

    Identical clip-only, deterministic playback to ``G1TrackerParityPlayEnvCfg``,
    but tracks the motion the teacher actually achieved on the parity profile
    (``DEFAULT_TEACHER_PARITY_CLIP_PATH``) instead of the hand-authored kinematic
    clip.  Use this to run the same IsaacLab <-> MuJoCo parity check on a tracker
    trained against teacher-recorded clips, whose reference distribution is the
    teacher's realizable motion rather than the raw stitched strides.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.traj_ref.path = DEFAULT_TEACHER_PARITY_CLIP_PATH
