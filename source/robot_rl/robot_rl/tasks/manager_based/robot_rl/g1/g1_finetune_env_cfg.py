# Copyright 2026 Zachary Olkin. All rights reserved.

"""Fine-tune the depth teacher ON the generator: reward, critic and terminations all follow it.

The teacher is trained entirely on clips and is handed the generator's prediction only at
deploy. An earlier version of this task closed that gap from the observation side alone,
keeping the clip as the graded reference. Measured, that fine-tune LOST 15 points of plain
clip tracking (0.940 -> 0.788 paired, p<0.001) without closing the generator gap, so the
premise changed: if the deployed objective is "track what the generator asks for", then that
is what should be rewarded, and the clip's role shrinks to supplying the world.

**``reference_source="fm"`` does almost all of it, and that is the point.** The applied
``y_des`` becomes the generator's, and everything downstream follows with no plumbing: the
CLF (and so ``clf_reward``), the actor's ``reference_window``, the critic's copy of it, and
``reference_world_state``. There is no residual observation, no widened critic and no
checkpoint surgery -- all of which the clip-graded design needed and none of which survives
here.

What the CLIP still supplies, and only this:

* the terrain (the mesh is built from its manifest) and the reset spawn pose;
* the VELOCITY COMMAND -- ``command_window`` reads ``target_vel`` from the library at every
  offset, independent of ``reference_source``, so the joystick track stays real;
* ``clip_end``, so an episode stops when its motion runs out instead of walking off a finite
  tile under a saturated command for the rest of the 20 s.

**The terminations are rebuilt around the reference rather than the world**
(:class:`G1FinetuneTerminationCfg`), because an absolute datum is wrong on terrain in both
axes -- see that class.

**The generator runs autoregressively** (``history_from="generator"``), which is the deploy
condition. Teacher forcing is gone: it kept the prediction phase-locked to a clip that is no
longer what the reward grades, so it would now be answering a question nobody is asking.
"""

from __future__ import annotations

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_depth_env_cfg import G1TeacherTerrainDepthEnvCfg
from .g1_teacher_env_cfg import REPLAN_PERIOD_K
from ..mdp.commands.traj_tracking.fm_tracking_cmd_cfg import FmTrackingCommandCfg


##
# Terminations
##
@configclass
class G1FinetuneTerminationCfg:
    """Failure is measured against the REFERENCE, never against the world.

    The teacher's two clip-graded failures (``base_pose_error`` / ``ee_pose_error``) go: they
    compare the robot to the globally-placed clip, and the clip is no longer the thing being
    tracked. What replaces them is one height test and one attitude test, both relative to
    whatever the generator predicted.

    **The two are measured against different datums on purpose.** Attitude goes
    reference-relative because a fixed cone around world upright cannot work on terrain:
    climbing and jumping pitch the base deliberately, so the cone either fires on healthy
    motion or is slackened until it catches nothing. Height stays ABSOLUTE, because the
    reference-relative version has a hole -- ego re-anchoring re-datums the reference onto the
    robot at every re-plan, so a height error against it resets toward zero and a slow sink
    escapes. World z is a true fall detector as long as the terrain never drops below zero,
    which holds for the shipped libraries.

    ``clip_end`` STAYS. Nothing else bounds the episode by the motion, and without it the
    robot is commanded the clip's last velocity off the end of a finite tile.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    #: Truncation, not failure: the motion ran out, the robot did nothing wrong.
    clip_end = DoneTerm(
        func=mdp.clip_end_reached,
        params={"command_name": "traj_ref", "reserve_lookahead": True},
        time_out=True,
    )

    #: The fall detector, on ABSOLUTE world height. A reference-relative height bound looks
    #: more consistent but does not work: ego re-anchoring re-datums the reference onto the
    #: robot every re-plan, so the error resets toward zero and a slow sink is never caught.
    #: 0.3 m sits deep under any standing pose, so this fires on a collapse, not a stoop.
    base_height = DoneTerm(
        func=mdp.base_height_below_termination, params={"min_height": 0.3},
    )

    #: 60 deg between the robot's up axis and the reference's. Yaw drops out by construction,
    #: so this does not double-count heading error.
    base_tilt_error = DoneTerm(
        func=mdp.fm_base_tilt_error_termination,
        params={"command_name": "traj_ref", "max_tilt_rad": 1.047},
    )


##
# Commands
##
@configclass
class G1FinetuneCommandsCfg:
    """The teacher's clip command, with the generator APPLIED rather than run beside it.

    Every clip-side field is the depth teacher's -- the ego anchor and its cadence, the CLF
    weights, the spawn sampling -- because the clip still places the robot and still owns the
    terrain. What changes is which reference is installed.
    """

    traj_ref = FmTrackingCommandCfg(
        window_anchor="plan",
        contact_bodies=[".*_ankle_roll_link"],
        # No default library or checkpoint: --clip and --fm_checkpoint state them, and a
        # stale default is the one failure neither can detect for you.
        path="",
        fm_checkpoint="",
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        # Open on FLAT. The learned null-history embedding is what covers the first N steps
        # of every episode, and it is fitted overwhelmingly on flat ground -- which is also
        # how the robot starts on hardware. Beginning an episode already on terrain asks it
        # for the one thing it never learned. Clips with no flat lead-in are handled by
        # seeding real clip history instead; see FmTrackingCommandCfg.seed_history_from_clip.
        random_start_before_terrain_start=True,
        randomize_heading=False,
        debug_vis=False,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        eta_scales={"pos": 1.0, "ori": 0.45, "lin_vel": 0.7, "ang_vel": 0.2},
        # THE GENERATOR IS THE REFERENCE. The CLF, the actor's window, the critic's window
        # and reference_world_state all follow this one field.
        reference_source="fm",
        # Autoregressive -- the deploy condition.
        history_from="generator",
        # ...except where an episode has to start on terrain, which leaves the null embedding
        # off-distribution. Those envs get real clip history instead. Per env, at reset.
        seed_history_from_clip="on_terrain_start",
        fm_update_every=REPLAN_PERIOD_K,
        ego_anchor_fraction=1.0,
        ego_anchor_every=REPLAN_PERIOD_K,
        fm_snap_every=REPLAN_PERIOD_K,
        clip_device="cpu",
    )


##
# Env
##
@configclass
class G1FinetuneTerrainDepthEnvCfg(G1TeacherTerrainDepthEnvCfg):
    """Depth teacher, fine-tuned against its own generator.

    Resume a trained teacher into it (``--load_run`` / ``--resume``) and pass the generator
    with ``--fm_checkpoint``. The critic keeps the teacher's width, so the checkpoint loads
    with no surgery.
    """

    terminations: G1FinetuneTerminationCfg = G1FinetuneTerminationCfg()
    commands: G1FinetuneCommandsCfg = G1FinetuneCommandsCfg()
