# Copyright 2026 Zachary Olkin. All rights reserved.

"""Fine-tune the depth teacher against the reference its OWN generator produces.

The teacher is trained entirely on clips: the CLF grades the clip, the critic sees the clip,
the terminations grade the clip, and the actor is shown the clip's reference window with a
synthetic per-channel noise standing in for the generator's error.  At deploy it is handed
the generator's prediction instead, and the two differ by more than that noise models: the
eval harness puts the generator arm well below the near-ceiling clip arm, and the error is
structured -- limb-dominated and knee-heavy -- rather than the i.i.d. jitter the noise term
draws.

This env closes the gap from the OBSERVATION SIDE ALONE:

* the actor's ``reference_window`` is served by the generator (:func:`mdp.fm_reference_window`);
* the reward, the critic, the terminations and every DR term are the teacher's, unchanged,
  and still grade the CLIP.

**Why the reward can be left alone rather than re-pointed.**  ``FmTrackingCommand`` builds
BOTH references every step and caches them, applies the configured one, and computes the CLF
right there -- storing ``v`` / ``vdot`` as scalars.  ``clf_reward`` reads those stored
scalars.  So with ``reference_source="clip"`` every graded quantity is the clip's, and the
observation-time swap that serves the generator cannot reach any of them.  The alternative --
``reference_source="fm"`` plus a knob re-pointing the reward at the clip -- also moves the
clip-graded TERMINATIONS (``base_pose_error``, ``ee_pose_error``, ``clip_end`` all read
``y_des``), which would change when episodes end.  That is a bigger change than asked for,
and a silent one.

**Teacher forcing is the arm implemented here** (``history_from="clip"``): the generator is
conditioned on the recorded clip history, so its prediction stays phase-locked to the motion
the reward grades.  Without it the prediction drifts, the reward stops describing what the
policy was shown, and the gradient is confused rather than merely noisy.  The closed-loop arm
is one field away (``history_from="generator"``) and is expected to want its own treatment --
it is deliberately NOT the default here.

**Why this is worth doing at all**, and it is the non-obvious part: teacher forcing recovers
only a small fraction of the generator gap, and close to none of it once the ``jump_off``
families (which jump backwards, where no forward camera can see) are excluded.  A
phase-aligned generator is therefore about as hard to track as a drifting one.  That is what
makes it a legitimate fine-tuning target rather than a solved one -- the error is in the
prediction's CONTENT, not in the loop, so there is something here for the policy to learn.
Re-measure with ``scripts/rsl_rl/eval`` (reference mode ``fm_tf``) rather than trusting a
number quoted here; it moves with every pair.
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from motion_gen.joint_names import G1_JOINT_NAMES

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_depth_env_cfg import (
    G1TeacherTerrainDepthEnvCfg,
    G1TeacherTerrainDepthObservationsCfg,
)
from .g1_teacher_env_cfg import (
    REFERENCE_NUM_EE_FRAMES,
    REFERENCE_OFFSETS,
    REFERENCE_WINDOW_BLOCKS,
    REPLAN_PERIOD_K,
)
from ..mdp.commands.traj_tracking.fm_tracking_cmd_cfg import FmTrackingCommandCfg


#: Multiplier on the teacher's synthetic reference-window noise.
#:
#: **1.0 -- the teacher's own levels, unchanged** (joint angles +-0.1 rad).  That noise stands
#: in for the generator error this env now supplies for real, so there is a case for reducing
#: it: at full strength the policy sees the generator's structured error PLUS an i.i.d. draw of
#: comparable size.  It is left at 1.0 anyway, deliberately, because reducing it changes two
#: things at once -- the reference the policy tracks AND how much jitter it is hardened against
#: -- and the first is the one being tested here.  Turning it down is a clean follow-up sweep,
#: not something to bundle into the first run.
#:
#: Not a tuned number, and the knob exists so it can be swept without touching the layout: the
#: width, the clip and the post-noise scale are shared with the teacher at any value, so a
#: checkpoint moves between the two tasks regardless.
FINETUNE_NOISE_SCALE: float = 1.0


def _scaled_reference_noise(scale: float) -> mdp.ReferenceWindowNoiseCfg:
    """The teacher's per-channel reference-window noise, scaled uniformly.

    Scales the MAGNITUDES and re-expands them over the layout, rather than scaling the
    expanded vector: the table is the thing with meaning (metres, radians, rad/s per block),
    and re-expanding keeps this in step with ``REFERENCE_WINDOW_BLOCKS`` if the selection
    ever changes.

    Args:
        scale: Multiplier on every block's half-width.

    Returns:
        A config ready to attach as an ``ObservationTermCfg`` ``noise``.
    """
    mags: dict[str, float | list[float]] = {}
    for key, value in mdp.REFERENCE_WINDOW_NOISE.items():
        mags[key] = ([v * scale for v in value] if isinstance(value, (list, tuple))
                     else value * scale)
    return mdp.ReferenceWindowNoiseCfg.from_layout(
        num_steps=len(REFERENCE_OFFSETS),
        num_joints=len(G1_JOINT_NAMES),
        num_ee_frames=REFERENCE_NUM_EE_FRAMES,
        magnitudes=mags,
        blocks=REFERENCE_WINDOW_BLOCKS,
    )


##
# Observations
##
@configclass
class G1FinetuneObservationsCfg(G1TeacherTerrainDepthObservationsCfg):
    """The depth teacher's observations, with the reference window served by the generator."""

    @configclass
    class ReferenceCfg(G1TeacherTerrainDepthObservationsCfg.ReferenceCfg):
        """Generator-served reference + the clip's joystick command.

        ``command_window`` is deliberately left on the clip.  It is the joystick command,
        which at deploy comes from the operator rather than the generator -- which is also
        why it is the one reference term that carries no noise.
        """

        reference_window = ObsTerm(
            func=mdp.fm_reference_window,
            params={
                "command_name": "traj_ref",
                "offsets": REFERENCE_OFFSETS,
                "blocks": REFERENCE_WINDOW_BLOCKS,
            },
            noise=_scaled_reference_noise(FINETUNE_NOISE_SCALE),
            clip=(-20.0, 20.0),
            scale=mdp.reference_window_scale(
                num_steps=len(REFERENCE_OFFSETS),
                num_joints=len(G1_JOINT_NAMES),
                num_ee_frames=REFERENCE_NUM_EE_FRAMES,
                blocks=REFERENCE_WINDOW_BLOCKS,
            ),
        )

    @configclass
    class CriticCfg(G1TeacherTerrainDepthObservationsCfg.CriticCfg):
        """The teacher's privileged critic, plus the generator-vs-clip residual.

        The critic's own ``reference_window`` is documented as "the same reference window the
        actor's encoder sees, uncorrupted".  Serving the actor from the generator breaks that,
        and a value function that cannot see the input the policy acted on cannot separate a
        policy error from a bad reference -- it just eats the difference as variance.  That is
        the likeliest source of the value-loss spike this task shows on resume.

        The clip window STAYS (it is what the return is graded against) and the residual is
        added, so the critic sees both halves.  Appended LAST on purpose: observation terms
        concatenate in declaration order, so the teacher's 1961 inputs keep their positions and
        a resumed critic's first layer is padded rather than permuted -- see
        ``scripts/rsl_rl/pad_critic_for_finetune.py``, which is what makes the resumed value
        function start on the same function it left off on, instead of scrambled.
        """

        reference_residual = ObsTerm(
            func=mdp.fm_reference_residual,
            params={
                "command_name": "traj_ref",
                "offsets": REFERENCE_OFFSETS,
                "blocks": REFERENCE_WINDOW_BLOCKS,
            },
            clip=(-20.0, 20.0),
        )

    reference: ReferenceCfg = ReferenceCfg()
    critic: CriticCfg = CriticCfg()


##
# Commands
##
@configclass
class G1FinetuneCommandsCfg:
    """The teacher's clip command, upgraded to also run the generator.

    Every clip-side field is the depth teacher's, verbatim -- the ego anchor and its cadence,
    the CLF weights, the spawn sampling.  What is added is the generator: a checkpoint (so
    ``fm_enabled`` is true and it runs PASSIVELY beside the applied clip), the re-plan period,
    and teacher forcing.

    ``reference_source`` stays ``"clip"``.  That is the whole safety property of this env: the
    applied reference -- and therefore ``y_des``, the CLF, the critic's error vectors and the
    clip-graded terminations -- is the clip, exactly as in teacher training.
    """

    traj_ref = FmTrackingCommandCfg(
        window_anchor="plan",
        contact_bodies=[".*_ankle_roll_link"],
        # No default library or checkpoint, for the same reason the teacher and tracker have
        # none: --clip and --fm_checkpoint state them, and a stale default is the one failure
        # neither task can detect for you.
        path="",
        fm_checkpoint="",
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        eta_scales={"pos": 1.0, "ori": 0.45, "lin_vel": 0.7, "ang_vel": 0.2},
        # THE GRADED REFERENCE IS THE CLIP. Only the actor's window comes from the
        # generator, and it does so through mdp.fm_reference_window at observation time.
        reference_source="clip",
        # TEACHER FORCING: condition the generator on the recorded clip history so its
        # prediction cannot drift out of phase with the motion the reward grades.
        history_from="clip",
        fm_update_every=REPLAN_PERIOD_K,
        # The teacher's clip placement, unchanged -- this env still grades the clip.
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
    """Depth teacher, fine-tuned against its own generator's reference window.

    Resume the deployed teacher into this task (``--load_run`` / ``--resume``) and pass the
    generator with ``--fm_checkpoint``; everything graded stays the clip's.

    Costs roughly 3x the teacher's per-iteration time (the generator runs every step to keep
    its history and re-plans every ``fm_update_every``), which is why this is a fine-tune from
    a trained checkpoint rather than a run from scratch.
    """

    observations: G1FinetuneObservationsCfg = G1FinetuneObservationsCfg()
    commands: G1FinetuneCommandsCfg = G1FinetuneCommandsCfg()


@configclass
class G1FinetuneFallOnlyTerminationsCfg:
    """Terminations with the CLIP-TRACKING pair replaced by a fall detector.

    The fine-tune shows the actor the generator's window while the clip stays the graded
    reference, and two of the teacher's terminations measure the robot against that clip:
    ``base_pose_error`` (0.25 m / 0.8 rad on the anchor mask's pinned DOFs) and
    ``ee_pose_error``. Under the generator those end episodes for tracking the clip poorly --
    which is the very thing being fine-tuned, so the policy is cut off exactly where it has
    most to learn, and the shortened episodes bias what it sees.

    What is left is the reason an episode should really end: the robot fell, or the clip ran
    out. ``base_fallen`` is trajectory-INDEPENDENT (base below ``min_height``, or roll/pitch
    past 70 deg from upright), so it fires on a collapsed robot wherever the reference is --
    the same reasoning, and the same numbers, as the tracker task's termination set.

    ``time_out`` stays: it is an episode-length cap rather than a judgement about the motion,
    and PPO bootstraps a truncated episode differently from a terminated one.

    NOTE this changes what the run's numbers MEAN. ``Episode_Termination/ee_pose_error``
    disappears, mean episode length rises because failures no longer cut episodes short, and
    neither is comparable with a fine-tune that kept the clip-tracking pair.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    clip_end = DoneTerm(func=mdp.clip_end_reached, params={"command_name": "traj_ref"})
    base_fallen = DoneTerm(
        func=mdp.base_fallen,
        params={"min_height": 0.5, "roll_limit_deg": 70.0, "pitch_limit_deg": 70.0},
    )


@configclass
class G1FinetuneFallOnlyEnvCfg(G1FinetuneTerrainDepthEnvCfg):
    """:class:`G1FinetuneTerrainDepthEnvCfg` with the clip-tracking terminations removed."""

    terminations: G1FinetuneFallOnlyTerminationsCfg = G1FinetuneFallOnlyTerminationsCfg()
