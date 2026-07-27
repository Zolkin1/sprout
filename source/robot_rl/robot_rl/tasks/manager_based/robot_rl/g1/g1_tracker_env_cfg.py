# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 tracker: closed-loop robot-state-conditioned flow-matching deploy (flat ground).

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
(:mod:`g1_teacher_env_cfg`, ``G1-teacher-clf``). :class:`G1FmTrackerPlayEnvCfg` is a
play-only deploy entry point -- pass the teacher via ``--load_run`` and the robot-state FM
via ``--fm_checkpoint`` (``--clip_device cpu`` for the multi-GB clip library).

Differs from :class:`G1TeacherEnvCfg` only in the command term
(:class:`FmTrackingCommand`) and a reference-independent fall termination. Both run a
world-frame CLF, so they share one observation layout (:class:`G1TeacherObservationsCfg`).
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
from .g1_teacher_terrain_env_cfg import (
    DEFAULT_TERRAIN_CLIP_PATH,
    G1TeacherTerrainSceneCfg,
)
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
        path=DEFAULT_CLIP_PATH,
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
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
class G1FmTrackerEnvCfg(G1TeacherEnvCfg):
    """Flat-ground closed-loop FM tracking env: the teacher tracks the FM predicted from
    the LIVE robot-state history. Shared world-frame observation layout; adds the
    reference-independent ``base_fallen`` termination (the FM may steer OOD)."""

    observations: G1TeacherObservationsCfg = G1TeacherObservationsCfg()
    commands: G1FmTrackerCommandsCfg = G1FmTrackerCommandsCfg()
    terminations: G1TrackerTerminationCfg = G1TrackerTerminationCfg()


@configclass
class G1FmTrackerPlayEnvCfg(G1FmTrackerEnvCfg):
    """Play/deploy viewer: 16 envs, noise-free, no pushes. Load the teacher via
    ``--load_run`` and the robot-state FM via ``--fm_checkpoint``; ``--clip_device cpu``
    for the multi-GB clip library (seed spawn + velocity command only)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        self.commands.traj_ref.debug_vis = True


##
# Terrain closed-loop deploy: the teacher tracks the terrain-trained FM ON the stairs
##
@configclass
class G1FmTrackerTerrainEnvCfg(G1FmTrackerEnvCfg):
    """Terrain counterpart of :class:`G1FmTrackerEnvCfg`: the flat ground is swapped for the
    clip-library :class:`ClipTerrainImporter` (staircases + transitions + flat patches), so
    the pelvis height scan rides the treads and the robot has real stairs to climb while the
    teacher tracks the robot-state-conditioned FM.

    Only the scene + reset differ from the flat FM tracker: the terrain-aware reset spawns
    each env on a clip that matches its tile's ``terrain_id`` (so the robot starts on the
    right staircase / lead-in), and both the importer mesh AND the command's seed clip
    library are pinned to the same terrain ``.h5``. The reference is still the FM (the clip
    only supplies the reset spawn + the ``target_vel`` command that conditions the FM), so
    a terrain-trained FM should emit climbing references from the live height scan.

    Wiring mirrors :class:`G1TeacherTerrainEnvCfg.__post_init__` (kept inline rather than
    shared so this addition is purely additive and does not touch the teacher-terrain env).
    """

    scene: G1TeacherTerrainSceneCfg = G1TeacherTerrainSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()  # G1TeacherEnvCfg: FmTrackingCommand + teacher obs + base_fallen
        # The ClipTerrainImporter builds its mesh from clip_library_path and IGNORES the
        # inherited flat terrain_generator; clear it to avoid confusion.
        self.scene.terrain.terrain_generator = None
        # Terrain-aware reset: each env spawns on a clip matching its tile's terrain_id.
        self.events.reset_on_clip.func = mdp.reset_on_terrain_motion_sequence
        # Single clip-library source of truth: importer (mesh + per-env terrain_id) and the
        # FM command's seed/velocity clips MUST read the same file, or tile terrain ids won't
        # line up with the clips' terrain ids.
        self.commands.traj_ref.path = DEFAULT_TERRAIN_CLIP_PATH
        self.scene.terrain.clip_library_path = DEFAULT_TERRAIN_CLIP_PATH
        assert self.commands.traj_ref.path == self.scene.terrain.clip_library_path, (
            "terrain clip library (scene.terrain.clip_library_path) and command clip "
            "library (commands.traj_ref.path) must be the same file."
        )
        # Clips are baked axis-aligned with their staircase -> placement is a pure translation.
        self.commands.traj_ref.randomize_heading = False


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
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the FM reference markers + the height-scan ray hits on the treads.
        self.commands.traj_ref.debug_vis = True
        self.scene.height_scanner.debug_vis = True
