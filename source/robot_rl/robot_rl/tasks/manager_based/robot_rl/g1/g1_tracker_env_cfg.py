# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 tracker: closed-loop robot-state-conditioned flow-matching deploy.

Deploys a trained teacher policy against the **robot-state-conditioned** flow-matching
reference: each control step the FM conditions on the LIVE robot-state history (the ``N``
most-recent measured states, phase), a root-height token, and the future velocity
command, and predicts the ``F``-step ego-anchored clip future, re-anchored onto the
robot's current base pose (node 0 = "now"). This structurally fixes the autoregressive
root-z / exposure-bias drift. The clip library only supplies the reset spawn pose and the
``target_vel`` velocity command. The reference drives a single **world-frame** CLF, so the
same teacher trained on world/ego-anchored clips tracks the generator with no privileged
info in the policy.

There is no separate tracker *training* task: the policy is the teacher
(:mod:`g1_teacher_env_cfg`, ``G1-teacher-clf-terrain``).
:class:`G1FmTrackerTerrainPlayEnvCfg` is a play-only deploy entry point -- pass the teacher
via ``--load_run``, the FM via ``--fm_checkpoint``, and the library via ``--clip``.

Differs from :class:`~.g1_teacher_terrain_env_cfg.G1TeacherTerrainEnvCfg` only in the
command term (:class:`FmTrackingCommand`) and a reference-independent fall termination.
Both run a world-frame CLF, so they share one observation layout
(:class:`~.g1_teacher_env_cfg.G1TeacherObservationsCfg`).
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_teacher_env_cfg import G1TeacherTerminationCfg
from .g1_teacher_terrain_env_cfg import G1TeacherTerrainEnvCfg
from ..mdp.commands.traj_tracking.fm_tracking_cmd_cfg import FmTrackingCommandCfg

# Default robot-state-conditioned FM checkpoint (override per-run via --fm_checkpoint).
# Trained on ego_anchored_clip_future data; carries a height_normalizer.
DEFAULT_FM_ROBOT_CKPT = "logs/flow_matching/2026-06-30_17-16-27_fm_robot_hist_v1/best.pt"


##
# Terminations
##
@configclass
class G1TrackerTerminationCfg(G1TeacherTerminationCfg):
    """Inherited base-drift failure + clip-end / time-out truncations, plus a
    reference-independent fall detector.

    The FM may steer envs into states the clip never visits, so we add a fall
    termination that does not depend on the reference: terminate when the base drops
    below ``min_height`` or tilts more than 70 deg from upright (in roll or pitch).
    """

    base_fallen = DoneTerm(
        func=mdp.base_fallen,
        params={"min_height": 0.5, "roll_limit_deg": 70.0, "pitch_limit_deg": 70.0},
    )


##
# Closed-loop deploy: teacher tracks the robot-state-conditioned FM (live robot history)
##
@configclass
class G1FmTrackerCommandsCfg:
    """Robot-state-conditioned FM tracking command (live robot history, world-frame CLF)."""

    traj_ref = FmTrackingCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        # No default library -- ``--clip`` sets this and scene.terrain.clip_library_path
        # together (see g1_teacher_terrain_env_cfg).
        path="",
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        # Put every class in the same currency as metres -- see CLF.eta_scales. Unscaled,
        # link angular velocity took 68% of V and link POSITION 5%, at comparable RELATIVE
        # accuracy, so "body-link tracking" graded rates. ori is a characteristic length
        # (m/rad), the velocities a characteristic time (s). Shared with the tracker so
        # both grade tracking the same way. Re-derive with scripts/rsl_rl/clf_balance_check.
        eta_scales={"pos": 1.0, "ori": 0.45, "lin_vel": 0.7, "ang_vel": 0.2},
        fm_checkpoint=DEFAULT_FM_ROBOT_CKPT,
        fm_update_every=12,          # 0.24 s -- MUST match the recording's ego_anchor_every
        # Warmup clip is robot-anchored on the FM re-plan cadence (matches the deploy regime).
        ego_anchor_fraction=1.0,
        ego_anchor_every=12,         # 0.24 s -- same cadence as the FM re-plan above
        # The clip library only supplies the reset spawn pose + velocity command here, so
        # keep it on CPU by default -- loading the multi-GB library onto a 12GB GPU next to
        # the sim + FM + teacher OOMs (override with --clip_device).
        clip_device="cpu",
    )


@configclass
class G1FmTrackerTerrainEnvCfg(G1TeacherTerrainEnvCfg):
    """Closed-loop FM tracking on the clip-library terrain.

    Differs from :class:`G1TeacherTerrainEnvCfg` in the command term only: the reference is
    the FM predicted from the LIVE robot-state history rather than the clip itself. The clip
    library still supplies the reset spawn pose and the ``target_vel`` command that
    conditions the FM. Both run a world-frame CLF, so the observation layout is shared.
    Adds a reference-independent ``base_fallen`` termination, since the FM may steer OOD.

    Inheriting from the teacher-terrain env means the terrain wiring (importer mesh,
    terrain-aware reset, heading) is defined once, in that class.
    """

    commands: G1FmTrackerCommandsCfg = G1FmTrackerCommandsCfg()
    terminations: G1TrackerTerminationCfg = G1TrackerTerminationCfg()


@configclass
class G1FmTrackerTerrainPlayEnvCfg(G1FmTrackerTerrainEnvCfg):
    """Play/deploy viewer for the terrain FM tracker: one env per terrain tile, noise-free,
    no pushes. Load the teacher via ``--load_run`` and the terrain-trained robot-state FM via
    ``--fm_checkpoint``; ``--clip_device cpu`` for the multi-GB terrain clip library."""

    def __post_init__(self):
        super().__post_init__()
        # One replica row + one env per tile -> every terrain is shown exactly once.
        self.scene.terrain.num_rows = 1
        self.scene.num_envs = 24
        # Terrain clip library is multi-GB; keep it off a small GPU (device-safe gather).
        self.commands.traj_ref.clip_device = "cpu"
        # Both actor-side groups: the reference window is corrupted during training too.
        self.observations.policy.enable_corruption = False
        self.observations.reference.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the FM reference markers + the height-scan ray hits on the treads.
        self.commands.traj_ref.debug_vis = True
        self.scene.height_scanner.debug_vis = True
