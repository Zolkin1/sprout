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

**A hard barrier separates this task from the offline clips.** The clip library supplies
only the reset spawn pose (and the terrain meshed from it) and the ``target_vel`` command;
everything the policy observes is generated. So relative to
:class:`~.g1_teacher_terrain_env_cfg.G1TeacherTerrainEnvCfg` this env swaps the command term
(:class:`FmTrackingCommand`) and then removes every remaining clip consumer:

* **no rewards at all** -- there is nothing to train here, and the CLF reward terms grade
  against the clip;
* **no clip-graded termination** -- ``base_pose_error`` / ``ee_pose_error`` measure drift
  from the globally-placed clip and ``clip_end`` reads its length; what is left is
  ``time_out`` plus a reference-independent ``base_fallen``;
* **no privileged critic observation group**, which is where the last clip-derived
  observation lived (``tracking_error_global``). Nothing here trains, and ``play.py`` does
  not restore the critic's weights -- see :class:`G1TrackerObservationsCfg`.

Together with ``FmTrackingCommandCfg.strict_fm_only`` -- which NaNs every reference channel
the generator does not write -- that makes "am I accidentally tracking the clip?" a question
the env answers loudly rather than one you have to reason about.

The two ACTOR groups are the teacher's, unchanged
(:class:`~.g1_teacher_env_cfg.G1TeacherObservationsCfg`): same policy branch, same
``reference_window``, now served from the generator. Because those are identical under
either ``tracking_mode``, and the critic is gone from the shape contract, **any** teacher
checkpoint drives this task.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_teacher_env_cfg import G1TeacherObservationsCfg
from .g1_teacher_terrain_env_cfg import G1TeacherTerrainEnvCfg
from ..mdp.commands.traj_tracking.fm_tracking_cmd_cfg import FmTrackingCommandCfg

# Default robot-state-conditioned FM checkpoint (override per-run via --fm_checkpoint).
# Trained on ego_anchored_clip_future data; carries a height_normalizer.
DEFAULT_FM_ROBOT_CKPT = "logs/flow_matching/2026-06-30_17-16-27_fm_robot_hist_v1/best.pt"


##
# Terminations
##
@configclass
class G1TrackerTerminationCfg:
    """Reference-INDEPENDENT terminations only: time-out plus a fall detector.

    Deliberately NOT derived from :class:`G1TeacherTerminationCfg`. Every failure term
    there grades against the offline clip -- ``base_pose_error`` against its global
    placement, ``ee_pose_error`` against the anchored reference, ``clip_end`` against its
    length -- and none of that describes the generator the robot is actually tracking. The
    FM may legitimately steer somewhere the clip never goes; ending the episode for it would
    be measuring the wrong thing.

    What remains is a fall: the base drops below ``min_height`` or tilts more than 70 deg
    from upright (in roll or pitch). Episode length comes from ``time_out`` alone.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_fallen = DoneTerm(
        func=mdp.base_fallen,
        params={"min_height": 0.5, "roll_limit_deg": 70.0, "pitch_limit_deg": 70.0},
    )


##
# Rewards
##
@configclass
class G1TrackerRewardCfg:
    """No rewards.

    The tracker is play/deploy only -- the policy is the teacher, already trained. The
    inherited CLF and command-velocity reward terms would be computed and discarded every
    step, and every one of them reads the clip, so keeping them would put clip data back on
    the critical path for no benefit.
    """


##
# Observations
##
@configclass
class G1TrackerObservationsCfg:
    """The two ACTOR groups and nothing else -- no privileged critic group.

    The teacher's ``critic`` group is privileged and training-only: clean proprioception,
    the CLF error vectors, and ``tracking_error_global``, which measures drift against the
    globally-placed clip. Nothing here trains, and ``play.py`` does not load the critic's
    weights, so computing that group every step would only be a way to keep touching the
    clip -- the last such path in this task once the rewards and clip terminations went.

    Dropping it has a second, larger benefit: the actor's two groups are identical under
    both ``tracking_mode`` settings, so with no critic to shape-match, **any** teacher
    checkpoint drives this task. Previously a body-mode teacher failed to load here purely
    on the critic's width (``tracking_error_local`` is 168 wide in body mode, 118 in joint).

    The paired agent cfg (``FmTrackerPlayRunnerCfg``) points rsl-rl's ``critic`` observation
    set at ``policy`` so it can still build the (unused, unloaded) value net.
    """

    policy: G1TeacherObservationsCfg.PolicyCfg = G1TeacherObservationsCfg.PolicyCfg()
    reference: G1TeacherObservationsCfg.ReferenceCfg = G1TeacherObservationsCfg.ReferenceCfg()


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
        # 0.24 s. MUST equal the recording's fm_snap_every -- checked against the value
        # carried on the checkpoint, because the FM's horizon is sized from this period.
        fm_update_every=12,
        # No clip re-anchoring: nothing here reads the clip reference, so there is no
        # placement to keep on the robot. (The reset spawn pose still uses the library.)
        ego_anchor_fraction=0.0,
        # The clip library only supplies the reset spawn pose + velocity command here, so
        # keep it on CPU by default -- loading the multi-GB library onto a 12GB GPU next to
        # the sim + FM + teacher OOMs (override with --clip_device).
        clip_device="cpu",
    )


@configclass
class G1FmTrackerTerrainEnvCfg(G1TeacherTerrainEnvCfg):
    """Closed-loop FM tracking on the clip-library terrain.

    The reference is the FM predicted from the LIVE robot-state history rather than the clip.
    The clip library supplies only the reset spawn pose and the ``target_vel`` command that
    conditions the FM, so relative to :class:`G1TeacherTerrainEnvCfg` this overrides the
    command term and then strips every other clip consumer -- rewards, clip-graded
    terminations, and the critic's global-drift slot. See the module docstring.

    Inheriting from the teacher-terrain env means the terrain wiring (importer mesh,
    terrain-aware reset, heading) is defined once, in that class.
    """

    commands: G1FmTrackerCommandsCfg = G1FmTrackerCommandsCfg()
    observations: G1TrackerObservationsCfg = G1TrackerObservationsCfg()
    terminations: G1TrackerTerminationCfg = G1TrackerTerminationCfg()

    def __post_init__(self):
        """Run the teacher's setup, then drop the rewards entirely.

        The swap happens HERE rather than as a class attribute because the teacher's
        ``__post_init__`` tunes ``rewards.clf_reward`` / ``rewards.undesired_contacts``
        by name -- it would raise against an empty cfg. Let it configure the terms it
        expects, then throw them away.
        """
        super().__post_init__()
        self.rewards = G1TrackerRewardCfg()


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
