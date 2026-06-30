# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 tracker: flow-matching generator-tracking environment (flat ground).

Deploys a trained teacher policy against an **autoregressive, robot-decoupled**
flow-matching reference instead of a fixed clip: the FM conditions on its OWN
past predicted reference (seeded from a clip at reset + during the warmup -- it
never sees the robot) and each sampled trajectory is re-anchored onto the robot
root (a receding horizon).  The clip library only supplies the reset spawn pose
and the ``target_vel`` velocity command.  The reference drives a single
**world-frame** CLF, so the same teacher trained on world/ego-anchored clips can
track the generator with no privileged info in the policy.

There is no separate tracker *training* task: the policy is the teacher
(:mod:`g1_teacher_env_cfg`, ``G1-teacher-clf``).  These configs are play-only
deploy entry points -- pass the teacher via ``--load_run`` and the
**clip-conditioned** FM via ``--fm_checkpoint``.

Differs from :class:`G1TeacherEnvCfg` only in the command term
(:class:`GeneratorTrackingCommand`).  Both run a world-frame CLF, so they share
one observation layout (:class:`G1TeacherObservationsCfg`).
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

# Default flow-matching generator checkpoint (override per-run via --fm_checkpoint).
# MUST be a clip-conditioned, command-conditioned FM (cmd_array='target_vel',
# target_source='actual'); the command validates this at startup and errors if the
# file is missing or the checkpoint has no command conditioning.
DEFAULT_FM_CHECKPOINT = "logs/flow_matching/2026-06-29_11-00-10/best.pt"


##
# Commands
##
@configclass
class G1TrackerCommandsCfg:
    """Flow-matching-generator tracking command (autoregressive, robot-anchored)."""

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
        # Re-run FM inference every K control steps (the receding-horizon re-plan
        # cadence); the prediction advances through the F-step window in between.
        fm_update_every=10, #2,
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
    """Flat-ground generator-tracking env (FM-autoregressive, world-frame CLF).

    Shared base for the play-only deploy configs below; not registered as a
    standalone task (the policy is trained as the teacher, not here)."""

    # Shared world-frame observation layout (policy base-relative, critic
    # privileged); only the command term differs from the clip-only teacher task.
    observations: G1TeacherObservationsCfg = G1TeacherObservationsCfg()
    commands: G1TrackerCommandsCfg = G1TrackerCommandsCfg()
    terminations: G1TrackerTerminationCfg = G1TrackerTerminationCfg()


@configclass
class G1TrackerEnvCfgPlay(G1TrackerEnvCfg):
    """Smaller, noise-free play base for the deploy configs below."""

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

    Runs the flow-matching generator as a self-contained receding-horizon
    reference source: its history is its OWN past output (seeded from the clip at
    reset), so it never sees the robot.  Each sampled trajectory is re-anchored
    onto the robot root.  The tracker policy loaded via ``--load_run`` (the
    teacher) tracks the emitted reference.

    Pass the trained **clip-conditioned** FM with
    ``--fm_checkpoint <flow_matching_run>/best.pt``.  The clip library only
    supplies the reset seed + the velocity command (``target_vel``); use
    ``--clip_device cpu`` for the multi-GB library.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.traj_ref.path = DEFAULT_CLIP_PATH       # FM-training clips (seed + command)
        self.commands.traj_ref.random_start_phase = False     # seed from clip step 0
