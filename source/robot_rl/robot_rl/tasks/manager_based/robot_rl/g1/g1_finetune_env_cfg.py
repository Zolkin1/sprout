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

----

This module holds TWO tasks. :class:`G1FinetuneTerrainDepthEnvCfg` is the above, on a normal
clip library. :class:`G1FinetuneFreeformEnvCfg` is the same fine-tune with the clip removed
from the other three roles as well -- spawn, command and episode bound -- so it can run on
FREE-FORM terrain that no motion was ever generated for. That library is manifest-only (zero
clip datasets) and comes from ``scripts/rsl_rl/make_freeform_terrain.py``; see the banner above
:class:`G1FinetuneFreeformEnvCfg` for the four-way swap and what each one costs.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# The generator's own floor on a section's goal band, imported so the free-form task's
# termination and the terrain that declares the band cannot disagree about it. motion_gen is
# IsaacLab-free and this module (numpy + math only) imports nothing heavier, so this costs the
# env cfg nothing at import time.
from motion_gen.terrain.freeform import GOAL_HALF_WIDTH_FLOOR_M
from motion_gen.terrain.wall_field import DWELL_RADIUS_M as WALL_DWELL_RADIUS_M
from motion_gen.terrain.wall_field import DWELL_S as WALL_DWELL_S

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_depth_env_cfg import (
    G1TeacherTerrainDepthEnvCfg,
    G1TeacherTerrainDepthEventsCfg,
    enable_policy_far_camera,
    install_fm_depth_group,
)
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

    #: A SLACK backstop on world upright, 70 deg per axis, kept BESIDE the reference-relative
    #: attitude test rather than replacing it.
    #:
    #: **What it exists for is a hole in the absolute HEIGHT term.** That term is world z, so
    #: it only fires when the pelvis drops under 0.3 m -- which a collapse on RAISED ground
    #: never does. On the up-into-down course the spawn platform is 0.90 m up, so a robot that
    #: falls over up there sits at ~0.90 m and the height detector cannot see it; the same
    #: applies to a fall on a 0.40 m box top. Before this, such a fall was caught only by the
    #: reference-relative cone, which is a comparison against a prediction and therefore goes
    #: quiet exactly when the generator's plan has also fallen apart.
    #:
    #: **The class docstring's objection is to a TIGHT cone, not to this one.** A fixed cone
    #: around world upright cannot bound healthy terrain motion -- climbing and jumping pitch
    #: the base deliberately -- so it is either tripped by good behaviour or slackened until it
    #: catches nothing. 70 deg is the slackened end ON PURPOSE: it is far outside any
    #: deliberate pitch (a stair climb is ~20-30 deg, a box jump ~40) and inside a face-plant.
    #: It is a second opinion on a genuine collapse, not an attitude budget.
    #:
    #: PER-AXIS roll and pitch, matching `base_fallen`'s convention rather than the cone the
    #: two `max_tilt_rad` terms use -- so a combined 60/60 (about 76 deg from vertical) passes
    #: it. That is looser than "70 deg from vertical" and is the documented
    #: `TODO(align-tilt-convention)`; acceptable because this is a backstop, and a true cone
    #: would be the stricter of the two.
    base_upright = DoneTerm(
        func=mdp.base_orientation_from_upright,
        params={"roll_limit_deg": 70.0, "pitch_limit_deg": 70.0},
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

    def __post_init__(self):
        """The depth teacher's setup, then the two things running a GENERATOR requires."""
        super().__post_init__()
        # The generator runs every step here, AUTOREGRESSIVELY and conditioned on the
        # cameras, and it reads BOTH of them where the inherited policy group reads the near
        # one alone -- see g1_depth_env_cfg. Installing the group at the policy's resolution
        # points its terms at the EXISTING scene sensors, so the far camera starts being cast
        # (which is the cost of having a generator in the loop) and the near one is not cast
        # twice. Without this the generator would silently condition on one camera.
        install_fm_depth_group(self)
        # And the world it reads must be the world it was recorded in: the teacher's training
        # scene renders no decoration, the recordings the generator was fitted on did, and a
        # vertical surface it has never seen is what it misreads as a box.
        self.scene.terrain.render_decor = True


##
# ==========================================================================================
# THE FREE-FORM VARIANT: the same fine-tune, on terrain that no clip was ever made for
# ==========================================================================================
##
#
# The clip-graded fine-tune above still needs a clip library for five things, listed below,
# and the terrain is the expensive one: every tile in the shipped libraries exists to MATCH
# one recorded motion, tile by tile. Under
# ``reference_source="fm"`` the reference is generated perceptively from the cameras, so that
# constraint is gone and the ground can be free-form. What replaces the clip:
#
# | the clip supplied  | the free-form task uses                                            |
# |--------------------|--------------------------------------------------------------------|
# | the terrain mesh   | the same ``ClipTerrainImporter``, on a MANIFEST-ONLY ``.h5``        |
# | the spawn pose     | ``mdp.reset_on_freeform_section`` (the manifest's ``route`` block)  |
# | ``target_vel``     | ``FmTrackingCommandCfg.freeform_command`` -- speed draw + steering  |
# | ``clip_end``       | ``mdp.freeform_goal_reached`` -- the section's goal band            |
# | the critic's drift | ``tracking_error_global(datum="plan_anchor")``                      |
#
# Everything else -- the reward, the CLF, both windows, the depth groups, the DR -- is the
# clip fine-tune's, unchanged, which is what keeps this an ablation of the TERRAIN rather
# than of the task.

#: Floor on the goal band's half-extent (m), passed to :func:`mdp.freeform_goal_reached`.
#:
#: IMPORTED, not restated, and deliberately NOT that function's own 2.5 m default. 2.5 m is
#: mjeval's floor and mjeval's reason: nothing in that stack steers, so a band narrower than
#: the accumulated heading error measures the absence of a controller. Here the steering
#: controller runs (:func:`...fm_tracking_cmd.freeform_yaw_rate`), and the generator floors
#: every section's declared ``goal_half_width`` at this same number -- so passing it makes the
#: clamp a NO-OP and the manifest's own band is what is measured. At 2.5 m a 1.219 m staircase
#: would be scored through a 5 m-wide gate, which a robot that fell off the side and drifted
#: laterally past the goal line can pass.
FREEFORM_GOAL_MIN_HALF_WIDTH_M: float = GOAL_HALF_WIDTH_FLOOR_M


##
# Terminations
##
@configclass
class G1FinetuneFreeformTerminationCfg(G1FinetuneTerminationCfg):
    """The fine-tune's terminations with the clip-keyed bound swapped for a spatial one.

    Both FAILURE terms are inherited verbatim -- absolute-z height and reference-relative
    tilt -- because neither ever referred to a clip; they are stated against the world and
    against the generator's own prediction. Only the truncation changes.

    ``clip_end`` is disabled in :meth:`G1FinetuneFreeformEnvCfg.__post_init__` rather than
    here: :meth:`G1TeacherTerrainEnvCfg.__post_init__` reaches into ``events.reset_on_clip``
    and the ancestors assume the clip terms exist while they run, so the clip wiring is torn
    out AFTER the inherited setup rather than declared away before it.

    **Why the goal is a truncation and not a reward.** Reaching it ends the episode with the
    value bootstrapped from the final state (``time_out=True``), so walking the section is
    neither punished nor paid -- the CLF keeps grading how the robot tracked the generator
    the whole way there. A terminal bonus would make the return depend on section LENGTH,
    which varies 10-38 m across this terrain set.
    """

    #: THE STRICT TRACKING BOUNDS, as a termination rather than an offline latch.
    #:
    #: The union of lateral 0.20 m / height 0.12 m / tilt 30 deg around the SERVED node, from
    #: `tracking_error.BOOTSTRAP_SPEC` -- so this term and every offline reader of
    #: `first_violation_step` cannot disagree about what a violation is.
    #:
    #: **What it buys, and it is not primarily a fall detector.** The height bound separates a
    #: correct box dismount from a stairs-down-off-a-box one: watching the two-box course,
    #: proper jump-downs peak at 0.130-0.182 m while the wrong mode approaches the bound. So
    #: an episode that takes the wrong mode ends there and never banks the progress reward's
    #: success bonus, which is paid on the last re-plan of an ARRIVING episode.
    #:
    #: **It also makes the pipeline coherent.** Previously the recording ran past a violation
    #: and every consumer cut the episode there offline -- so a run reached the goal, banked
    #: the bonus, and the offline cut then deleted the row it was paid on, measured at 95.3%
    #: of arriving episodes. Terminating means no consumer needs to truncate anything and
    #: `reached_goal` and `rollout_outcomes.terminal_reward` become the same question.
    #:
    #: **Cost, measured on the real opening data set before switching it on** (2000 parent
    #: episodes, DR on, 3 spawn areas): clean arrivals 21.70% -> 16.85% of all episodes;
    #: windows surviving 95.7% -> 93.4%. Ground-start clean arrivals fall 4.18% -> 0.46%, so
    #: at this bound every clean success in an opening set is a box-top dismount -- a scarcity
    #: that predates this and is what box-top spawns exist for.
    #:
    #: `base_tilt_error` (60 deg) is now redundant with this term's 30 deg and is kept as a
    #: backstop: it is a different function on a different reference accessor, and on the
    #: measured opening set the tilt bound never binds before the height one anyway.
    #:
    #: **DECLARED HERE, on the free-form cfg, and deliberately NOT on the clip fine-tune it
    #: inherits from.** Two reasons, and the second is correctness rather than scope: the bound
    #: was calibrated against the GENERATOR's served node on these courses, and
    #: `_pred_valid` -- the gate that stops this firing before the first plan exists -- is
    #: absent on the clip command, where a reference always exists. On the clip path the gate
    #: would silently be a no-op and the term live from step 0, on a task nothing here
    #: measured.
    tracking_violation = DoneTerm(func=mdp.fm_tracking_violation_termination,
                                  params={"command_name": "traj_ref"})

    goal_reached = DoneTerm(
        func=mdp.freeform_goal_reached,
        params={"min_half_width": FREEFORM_GOAL_MIN_HALF_WIDTH_M,
                # Declared EXPLICITLY rather than omitted, so it is overridable: hydra's
                # struct mode refuses a key absent from the dict, and a param that cannot be
                # set from the CLI also never reaches the run's params/env.yaml -- the only
                # record of what a run was configured with. Both callers do override it
                # (`run_finetune_loop.sh`, `run_quick_eval.sh`), so this value is what a
                # DIRECT train.py/play.py launch gets.
                #
                # ON by default, at the value every course's goal spec declares. The
                # downrange-and-lateral test has no height, so on a clip-library `stairs_up`
                # tile -- 0.90 m of open floor either side of a 2.25 m flight, band clamped at
                # least that wide -- a robot that never left the floor passes the goal line and
                # scores a traverse. Off by default made that the silent case.
                #
                # 0.5 rather than something tighter because it must not bind on the DESCENDING
                # courses, whose goal is at floor level and whose height change is unavoidable.
                # Measured at arrival: box2 tops out at 0.203 m of |height_err| and
                # up-into-down at 0.399, while in-distribution binds at exactly 0.500 -- so one
                # value serves ascending and descending alike. The FUNCTION's own default stays
                # 0.0 (`terminations.freeform_goal_reached`): it is a library default, and
                # switching a height test on there would move every free-form task's
                # termination, including arms already measured.
                "height_tol_m": 0.5},
        time_out=True,
    )


##
# Events
##
@configclass
class G1FinetuneFreeformEventsCfg(G1TeacherTerrainDepthEventsCfg):
    """The depth teacher's DR, plus a spawn read off the manifest instead of off a clip.

    ``reset_on_clip`` is inherited and then disabled in the env's ``__post_init__`` (the
    ancestors write to it while they run), so what this class adds is only the replacement.
    It is declared as a SEPARATE term rather than by re-pointing ``reset_on_clip.func``
    because the two take different parameters -- there is no clip to pin, no start step to
    sample and no ``clip_draw`` -- and a term whose name says "clip" while its params say
    otherwise is the shape of config bug this pipeline has paid for twice.

    Term ORDER puts it after the camera DR, which is inherited and declared earlier. That is
    fine and worth stating: the camera terms move the CAMERAS relative to the robot, not the
    robot, so the spawn cannot invalidate a draw or vice versa, and every reset term runs
    before the first observation either way.
    """

    reset_on_section = EventTerm(
        func=mdp.reset_on_freeform_section,
        mode="reset",
        # ``spawn_z_offset`` matches the clip resets': with the soles exactly on the declared
        # surface PhysX ejects the robot and every episode opens with a kick. Here it is the
        # real fix rather than the clip path's workaround -- the manifest declares the surface
        # exactly, so 5 mm is clearance, not a correction for an inconsistent frame 0.
        #
        # The jitter is what keeps the set from being degenerate in the one variable this
        # fine-tune exists to broaden: without it every episode on a tile starts from ONE pose,
        # so the generator's first plan is conditioned on one image per tile. The box must sit
        # inside the start platform, which the 1.219 m platform minimum affords; at the 1.219 m
        # narrowest flight the lateral half-width leaves 0.36 m of apron either side of the
        # robot, so widening it needs the apron widened too.
        #
        # ``spawn_from`` is "all" here because this is the TRAINING task: the raised areas
        # (a box top, the platform between two flights) exist so the second half of a
        # traverse appears in the data at all, and AWR can only reweight what it is shown.
        # **Every EVAL overrides it to "ground"**, on every terrain, so a reported rate is
        # always the complete course --
        # `env.events.reset_on_section.params.spawn_from=ground`, which
        # `scripts/rsl_rl/run_quick_eval.sh` passes and records. See
        # `mdp.events.resets.SPAWN_FROM`.
        params={"spawn_z_offset": 0.005, "pos_jitter_m": (0.25, 0.25),
                "yaw_jitter_rad": 0.2, "spawn_from": "all"},
    )


##
# Env
##
@configclass
class G1FinetuneFreeformEnvCfg(G1FinetuneTerrainDepthEnvCfg):
    """The generator-graded fine-tune on free-form sections, driven by a joystick command.

    Pass the terrain with ``--clip`` (a manifest-only library from
    ``scripts/rsl_rl/make_freeform_terrain.py``) and the frozen generator with
    ``--fm_checkpoint``; resume a trained depth teacher with ``--load_run --resume``. There is
    no default library path here for the reason there is none anywhere in this pipeline: a
    stale one trains on the wrong terrain and nothing downstream can tell.

    **Three distribution changes ride along, all deliberate, none of them free:**

    1. **The command is STEERED**, and steering is off the generator's training distribution.
       It was commanded ``|wz| max = 0.0000`` over all 3500 stair clips of the library it was
       fitted on, and it is FROZEN here -- so on a flight it is extrapolating in yaw rate.
       Documented at :func:`...fm_tracking_cmd.freeform_yaw_rate`, which also carries the
       measurement. It is on anyway because nothing else in the loop steers and these sections
       are tens of metres long.
    2. **The command is HELD across the window offsets**, where the clip path served a
       per-offset future. That direction is a mismatch being REMOVED -- both deploy runtimes
       hold one command over the whole horizon -- but the tracker's reference-group encoder
       still goes from 13 distinct ``command_window`` rows to 13 copies of one.
    3. **The stairwell walls COLLIDE.** They are authored ``decor=False``, so they join the
       ground mesh, take a collider and appear in the height scan -- where the decoration the
       generator was fitted on did none of those things. Nothing in the MDP teaches obstacle
       avoidance: ``undesired_contacts`` is **-0.5** against ``clf_reward`` at 10 -- the
       weight THIS cfg carries, which is not the -0.1 declared beside the term in
       ``g1_clf_tracking_base``: ``G1TeacherEnvCfg.__post_init__`` overwrites it and every
       terrain cfg down to this one inherits that. Read it off a constructed cfg, never off
       the declaration -- and its six sensors are the thighs, torso, backpack and elbows,
       with no wrists or hands. So a wall is something to be walked into rather than around, and a
       scan-conditioned number measured here is not comparable to one measured on a clip
       library.

    **A CLIP-FREE LIBRARY IS NOW A DEFINED THING**, and this is the one task that takes one.
    ``FmTrackingCommand`` inherits ``MotionSequenceCommand``, which builds a
    ``MotionClipManager`` in ``_post_init`` and rebuilt the clip reference every step. Neither
    used to survive a library with no clips -- ``MotionClipManager.__init__`` reached
    ``clip_len_or_full`` -> ``num_steps`` -> ``y_des.shape[1]``, which raised ``IndexError``
    on a rank-1 placeholder before anything here ran. What changed, in three places rather
    than by special-casing this cfg:

    * the WRITER emits the clip datasets at zero extent but full rank, so ``num_steps`` is a
      real number and the manager constructs (``ClipLibraryWriter`` at ``num_clips=0``);
    * the MANAGER answers every layout question and REFUSES every gather by name
      (``MotionClipManager._require_clips``), so a clip-consuming term left wired on this task
      fails saying so instead of indexing an empty table;
    * the COMMAND skips the clip half of the step (``_advance_without_a_clip``), keeping only
      the plan-anchor latch, so ``self._step`` never marches and ``as_clip_reference()`` is
      refused.

    ``accepts_clip_free_library`` is what permits the combination, and it is True only under
    ``reference_source="fm"`` -- every other command/config pair still refuses at construction.
    ``--like <a real library>`` is worth passing: the FM checkpoint's ``frame_names`` are
    checked against the ee frames resolved from the library's OWN channel names, so the
    minimal seven-channel layout admits only a generator with no extra frames.

    **The critic's global-drift channel changes MEANING, not width** (see
    :func:`mdp.tracking_error_global`): plan-anchor drift resets toward zero at every re-plan
    and is bounded by roughly ``K * dt * speed``, where the clip datum grew without bound.
    A resumed critic therefore loads with no surgery and reads a differently-distributed six
    channels -- expect a value transient early in the fine-tune, and read it as that rather
    than as the policy collapsing.
    """

    terminations: G1FinetuneFreeformTerminationCfg = G1FinetuneFreeformTerminationCfg()
    events: G1FinetuneFreeformEventsCfg = G1FinetuneFreeformEventsCfg()

    def __post_init__(self):
        """The clip fine-tune's setup, then the clip torn out of it.

        Order is load-bearing: every ancestor's ``__post_init__`` assumes the clip wiring is
        present while it runs -- :class:`G1TeacherTerrainEnvCfg` assigns
        ``events.reset_on_clip.func``, which is an ``AttributeError`` against a ``None``. So
        the terms are disabled here, after ``super()``, not declared away above it.
        """
        super().__post_init__()

        cmd = self.commands.traj_ref
        # The velocity command is MANUFACTURED, not read from the library: a per-episode
        # forward speed after a standing hold, vy pinned to 0, and a P controller on the
        # bearing down the section. This is also what makes a clip-free library admissible --
        # ``command_velocity_available`` stops depending on a ``target_vel`` dataset.
        cmd.freeform_command = True
        # JOINT tracking, for the same reason the clip knobs below go off: there is no clip.
        # The teacher chain sets ``tracking_mode = "body"``
        # (:class:`G1TeacherTerrainEnvCfg`), which needs a library carrying root-frame body
        # channels -- and a manifest-only section library carries no channels at all, so
        # ``base_trajectory_cmd._resolve_tracked_vel_names`` raises at command-manager init:
        #
        #     tracking_mode='body' needs a clip library carrying root-frame body channels
        #     (@root), but this library has none.
        #
        # ``g1_tracker_env_cfg`` pins the same value for the same reason, and the two are the
        # generator-driven paths. Set AFTER ``super()`` because ``_apply_tracking_mode`` runs
        # as a validation during it.
        cmd.tracking_mode = "joint"
        # Every clip-phase knob is now meaningless, and each one would otherwise be a claim
        # about a library that has no clips. Off explicitly rather than left at the teacher's
        # value, so a reader of this cfg is not left inferring which of them still apply.
        cmd.random_start_phase = False                  # no phase to start at
        cmd.random_start_min_remaining_s = 0.0          # nothing runs out
        cmd.random_start_before_terrain_start = False   # sections open on their own start
        # ...including the history seeding, whose whole mechanism is reading the clip's steps
        # BEFORE the start. Every free-form episode therefore opens on the learned
        # null-history embedding -- which is the deploy condition, and is why the sections are
        # authored with a flat approach apron in front of the terrain rather than starting on
        # it.
        cmd.seed_history_from_clip = "never"

        # The reset reads the spawn off the manifest's route block. Disabled rather than
        # re-pointed: ``reset_on_motion_sequence`` and its terrain twin both SAMPLE A CLIP,
        # and on a manifest-only library that raises the moment a tile's terrain id has no
        # clips -- correct for a clip library, fatal here.
        self.events.reset_on_clip = None
        # Nothing bounds the episode by a motion any more; the section's goal band does.
        self.terminations.clip_end = None

        # The critic's global datum. Zero-filling it would keep the width (which is what lets
        # a teacher checkpoint resume unchanged) while handing the value function a constant,
        # and the clip branch is worse than useless here -- it keeps indexing a clip by step
        # under ``reference_source="fm"``, so it marches away from the robot on its own.
        self.observations.critic.tracking_error_global.params = {
            "command_name": "traj_ref", "datum": "plan_anchor",
        }


@configclass
class G1Box2FinetuneEnvCfg(G1FinetuneFreeformEnvCfg):
    """The free-form fine-tune, on the fixed TWO-BOX course, for an EXISTING tracker.

    Everything comes from :class:`G1FinetuneFreeformEnvCfg` -- the manifest spawn, the steered
    command with its standing hold, the section goal band, and the reward set (the CLF tracking
    terms, the absolute-z fall and the tilt bound).  Pass ``--clip clips/box2_train.h5``; the
    two-box tiles are ordinary free-form sections and every term reads them unchanged.

    **What this adds is the FAR camera, back in the policy observation.**  The policy reads the
    near camera alone by decision, taken ahead of a teacher retrain that has not happened -- so
    every tracker checkpoint on disk expects 3002 inputs where a near-only task builds 2222,
    and none of them can run the free-form task at all.  Recording the two-box set needs one of
    those checkpoints (it is the pair the MuJoCo box2 sweep measured, which is what makes the
    Isaac and MuJoCo numbers comparable), so the term goes back into its trained slot.

    Retire this task, not the flag, once a near-only teacher exists: at that point the plain
    free-form task takes the two-box library directly and this subclass is the stale thing.
    """

    def __post_init__(self):
        """Compose the free-form fine-tune, then restore the policy's far camera."""
        super().__post_init__()
        # AFTER super(): the ancestors build the observation groups, and this fills a slot
        # they declared. Before it, there is nothing to fill.
        enable_policy_far_camera(self)


##
# The WALL-FIELD course: enclosed rooms, and a goal you have to STOP in
##
#: Range-to-goal gain for the approach controller on this course (1/s).
#:
#: With :data:`WALLS_STOP_RADIUS_M`, a 2.5 m/s episode starts decelerating about 5.7 m out.
#: Chosen so the deceleration is gentle relative to the 1.10 m corridor rather than tuned:
#: nothing here has been measured, and the first thing to look at if the robot overshoots the
#: disc is this and ``freeform_approach_kd``, in that order.
WALLS_APPROACH_KP: float = 0.6

#: Range at which the whole command triple goes to zero (m). EQUAL to the dwell radius.
#:
#: **Equal because the command should say one thing: the moment you are in the circle, stand.**
#: **MEASURED, and the reason this is no longer the dwell radius.** With the two equal, the
#: controller's equilibrium is a SHELL on the scoring boundary rather than the goal: the P
#: term's setpoint is `to_go == stop_r` and the hard cut fires at the same range, so the robot
#: aims at the rim. Over 12,598 uncensored episodes of the first wall arm the closest approach
#: was a tight annulus -- p10 0.66, median 0.80, p90 0.89 m against a 1.0 m disc -- and ~1% of
#: episodes ever came within 0.5 m of the goal. The 0.20 m of margin the median robot had was
#: coast, not aim.
#:
#: That cost dwells, and increasingly with speed. Splitting those episodes by where they came
#: to rest: the ~8% that stopped in the outer 0.1 m held the disc 0.862 of the time at
#: 0.40 m/s and only 0.611 at 1.50, against 0.938 / 0.884 for the 91% that stopped between 0.5
#: and 0.9 m. A robot parked on the rim is one shuffle from resetting a consecutive counter.
#:
#: It also meant the course was not running its own drill. Half the goals sit 0.5-1.5 m in front
#: of a wall and the robot always arrives from the near side, so stopping 1.0 m short left it
#: ~1.5 m off the wall it was supposed to be learning to stop for.
#:
#: 0.25 m is the coast distance, rounded up: the command steps from the 0.3 m/s floor to zero
#: and momentum carries the robot the measured ~0.20 m, so it comes to rest a little short of
#: the goal point rather than past it. Short matters on the near-wall tiles -- overshooting the
#: centre there is travel toward the wall.
#:
#: THE EARLIER ARGUMENT FOR EQUALITY IS NOT REFUTED, IT IS NARROWER THAN IT LOOKED. It said a
#: smaller stop radius trains the robot to treat the goal as somewhere to pass THROUGH. That is
#: only true of a setpoint past the goal point; 0.25 m is still short of it, so the robot is
#: never commanded beyond the thing it is being scored on. (A separate earlier claim -- that a
#: smaller radius would let the speed floor walk the robot back OUT -- was simply wrong and is
#: recorded as wrong: inside the disc and outside the stop radius the floor commands motion
#: INWARD.)
#:
#: Moving the stop radius in does NOT drag the steering in with it; see
#: :data:`WALLS_YAW_QUIET_RADIUS_M`.
WALLS_STOP_RADIUS_M: float = 0.25

#: Range (m) at which the STEERING channel goes quiet -- the DISC, not the stop radius.
#:
#: These were one number while the robot stopped at the rim, and they have to come apart now
#: that it drives to the centre. Steering must end no later than forward motion, because a yaw
#: rate commanded to a stopped robot is a step input it can only answer by turning on the spot,
#: and a robot pirouetting inside the disc is one drift from breaking a dwell it had earned.
#: Left equal to the 0.25 m stop radius, that is exactly what would happen to every env that
#: comes to rest off-centre: `vx` clamps to zero on the along-track term while `rng` is still
#: above 0.25, so it would stand and turn.
#:
#: The cost is the last 0.75 m being driven open-loop in heading. That is acceptable HERE and
#: would not be everywhere: by the time the robot is inside the disc it is already scored as
#: arrived, so lateral error accumulated over that stretch cannot cost the arrival it has.
WALLS_YAW_QUIET_RADIUS_M: float = WALL_DWELL_RADIUS_M

#: Floor on the commanded forward speed while the goal is still ahead (m/s).
#:
#: Without it the P law decays through the 0.05-0.3 m/s band on the last stretch, the robot
#: stops walking wherever its own deadband lands, and it comes to rest SHORT of the goal while
#: otherwise behaving correctly -- observed directly on this course. 0.3 keeps the command
#: either zero or inside the band the generator trained on, so the approach is a ramp to 0.3,
#: held, then a step to zero at the disc.
WALLS_APPROACH_MIN_SPEED: float = 0.3

#: The dwell disc's radius (m) -- 2 m diameter. IMPORTED, not restated: the terrain module
#: stamps the same number into this course's goal spec, and the debug marker reads it back off
#: the termination below, so the spec, the rule and the picture cannot drift apart.
WALLS_GOAL_RADIUS_M: float = WALL_DWELL_RADIUS_M

#: Seconds the robot must hold the disc, upright, before the episode counts as a success.
WALLS_DWELL_S: float = WALL_DWELL_S


@configclass
class G1WallsFinetuneTerminationCfg(G1FinetuneFreeformTerminationCfg):
    """Terminations for the wall course: the pass-through goal replaced by a DWELL goal.

    ``goal_reached`` is set to ``None`` rather than left alongside the dwell term. The first
    success rule to fire ends the episode, so a task carrying both would be scored by whichever
    is looser -- and the pass-through one is looser by construction, which would make the dwell
    requirement decorative while still appearing in the config.
    """

    #: KEPT AND INERT, not removed. Both launchers unconditionally override
    #: `env.terminations.goal_reached.params.height_tol_m`, and hydra refuses an override whose
    #: key is missing or None -- so deleting this term kills every collect and every eval at
    #: config composition, before the GPU is touched. `never_terminates` keeps the key and the
    #: params addressable while the term does nothing.
    #:
    #: THE PARAMS BELOW ARE THEREFORE DOCUMENTATION, NOT CONFIGURATION. A launcher setting
    #: `height_tol_m` here gets no error and no effect; the tolerance that matters on this
    #: course is `goal_dwell`'s.
    goal_reached = DoneTerm(
        func=mdp.freeform_goal_disabled,
        params={"min_half_width": FREEFORM_GOAL_MIN_HALF_WIDTH_M, "height_tol_m": 0.5},
        time_out=True,
    )

    goal_dwell = DoneTerm(
        func=mdp.freeform_goal_dwell,
        params={"radius_m": WALLS_GOAL_RADIUS_M,
                "dwell_s": WALLS_DWELL_S,
                # DECLARED so it is overridable, exactly as `freeform_goal_reached`'s own
                # comment says of its tolerance: hydra's struct mode refuses a key absent from
                # the dict, so a param omitted here cannot be set from the CLI at all. 0 is
                # silent and is what every real run uses.
                "debug_every": 0,
                # The same height datum the pass-through goal uses, and NOT optional here: a
                # robot that has fallen inside the disc would otherwise keep accruing dwell and
                # be scored as having stopped, which is the opposite of the point.
                "height_tol_m": 0.5},
        time_out=True,
    )


@configclass
class G1WallsFinetuneEnvCfg(G1FinetuneFreeformEnvCfg):
    """The wall-field fine-tune: enclosed rooms, walls to ignore, and a goal to STOP in.

    Takes ``--clip clips/ft_walls_train.h5`` (or ``_holdout``) from
    ``scripts/rsl_rl/make_walls_terrain.py``. See ``motion_gen.terrain.wall_field`` for the
    course and its corridor invariant.

    **This is the only task that switches the approach controller on**, and that is deliberately
    visible: ``test_freeform_approach_defaults_off.py`` asserts no other G1 cfg assigns those
    knobs, so turning them on anywhere else is an edit to that test's allow-list rather than
    something a shared base class can do on a course's behalf. Every other free-form arm --
    boxes, updown, indist, multi -- keeps the constant per-episode ``vx`` its numbers were
    measured under.

    **Why the forward channel joins the yaw channel.** The other courses are traverse-and-exit:
    the goal is a downrange line, crossing it ends the episode, and nothing after that is
    observed. Here the question is whether the robot STOPS, which a constant velocity command
    makes unanswerable -- the reward's ``exp(-|v_err|/sigma)`` term pays for continuing to move,
    so holding a 0.5 m disc would be a behaviour the policy is penalised into failing. With the
    range-driven command the same term pays FULL marks for standing still at the goal, and
    stopping becomes trainable without touching the reward.

    **The GRADING follows the dwell too, and that took a change outside this file.** The
    recorded ``reached_goal`` column used to be a latch over a downrange half-plane, so a robot
    that crossed the goal line and then walked into the wall graded as an arrival -- and the
    progress reward pays on exactly that column. This course's goal spec now declares
    ``dwell_radius_m`` / ``dwell_s``, which makes ``eval.accumulate`` score ``reached_goal`` as
    "held the disc for dwell_s" instead. The reward, the critic's arrived set and
    ``terminal_reward`` all read that one column, so all three follow. ``reached_goal`` is
    therefore NOT comparable between this course and a pass-through one;
    ``accumulate`` prints which rule it scored under.
    """

    terminations: G1WallsFinetuneTerminationCfg = G1WallsFinetuneTerminationCfg()

    def __post_init__(self):
        """Switch on the approach controller. The tile replicas are deliberately inherited."""
        super().__post_init__()
        self.commands.traj_ref.freeform_approach_kp = WALLS_APPROACH_KP
        self.commands.traj_ref.freeform_stop_radius_m = WALLS_STOP_RADIUS_M
        # Pure P. The tracker is itself a closed loop on the commanded velocity, so the
        # approach is damped already; raise this only if the robot visibly overshoots.
        self.commands.traj_ref.freeform_approach_kd = 0.0
        self.commands.traj_ref.freeform_approach_min_speed = WALLS_APPROACH_MIN_SPEED
        self.commands.traj_ref.freeform_yaw_quiet_radius_m = WALLS_YAW_QUIET_RADIUS_M
        # NO num_rows OVERRIDE, and the reasoning that said there should be one was wrong
        # on both counts. `num_rows` replicates the whole tile set and env origins are tile
        # centres, so with fewer slots than envs several robots share an origin -- which looked
        # like a problem this course had and the others did not, because a wall tile declares
        # ONE small spawn rectangle inside a room where the others declare a chain.
        #
        # It is not a problem. `InteractiveSceneCfg.filter_collisions` defaults True and
        # nothing in robot_rl overrides it, so co-located robots do not interpenetrate; they
        # get correlated spawns, which every other free-form arm already has at 10 rows.
        #
        # And 86 rows would not have fixed it anyway: IsaacLab draws the ROW at random
        # (`terrain_levels = randint(0, max_init_level)`) and only the COLUMN evenly, so at
        # 2048 envs over 86x24 slots ~63% of envs still share one. What it does buy is an
        # 8.9 km world extent and 252k triangles against 2.9k, for nothing.
        #
        # So this inherits the 10 that every other free-form course runs.
