# Copyright 2026 Zachary Olkin. All rights reserved.

"""Record CLOSED-LOOP rollouts of the frozen tracker driven by the generator.

The generator drives the reference and conditions on its own served history; one WINDOW is
written per generator re-plan, and one ROW per episode saying how it ended.  Together those
are what an advantage label is computed from: the label is a function of the outcome, and the
thing it labels is the chunk.

Not ``record_data.py``.  That script records the TEACHER tracking clips, and its window format
is 22 WORLD-frame state terms per node -- root pose, twist, joints, four end-effector frames --
because a clip carries all of them.  A generator predicts three BLOCKS and no end effectors, so
on an FM-driven rollout that format cannot be filled (``get_future_world_frames`` refuses
outright) and everything that survives is a lossy re-encoding of arrays the command already
holds exactly.  See ``rollout_outcomes`` for what is stored instead and why.

Two settings are asserted rather than left to the caller, because each is silent when wrong:

* **Gaussian flow noise.**  At ``x0 = 0`` -- the default in every deploy runtime and in
  MuJoCo -- the flow is deterministic, so every chunk drawn at a given state is identical and
  ``A(s, a)`` collapses to ``A(s)``.  The label would then carry information about how hard
  the STATE was and none about which ACTION was better, which is the entire signal.  Isaac
  already defaults to Gaussian; this makes it a property of the recording rather than of a
  default that could move.
* **A calibrated goal region.**  Without one, every outcome available is a clip-CURSOR
  condition -- ``clip_end`` and the ``terrain_*`` columns compare step indices -- so
  "success" would mean "stayed upright for N steps" and a reward built on it is maximised by
  standing still.

One known gap in the recorded DISTRIBUTION, shared by every recording to date (v2 at 128 envs
and v4 at 8192 were both taken on ``-depth-play``):

* ``TODO(record-camera-dr)`` -- **the camera sees no domain randomisation, and neither does
  the reference window.**  This recorder is run on the ``-play`` task, and
  ``g1_depth_env_cfg.apply_depth_play_trims`` nulls every event whose func lives in
  ``mdp.events.depth`` -- all six of them: extrinsics, intrinsics and the noise/dropout
  resample, per camera.  ``_apply_tracker_play_trims`` additionally nulls
  ``randomize_ground_contact_friction`` and sets ``enable_corruption = False`` on BOTH actor
  obs groups.  What survives is the seven ``startup``-mode physical terms (joint defaults,
  gains, joint friction, torso mass, torso/backpack CoM, armature), so each env is a
  different robot -- but fixed for the whole run, so a draw is confounded with whichever
  clips that env happened to get.  Depth is the generator's ONLY exteroception, so a critic
  fitted here has only ever scored a perfect camera.  This is the same trap
  ``scripts/flow_matching/README.md`` already warns about for ``record_data.py``.
  **The fix is NOT "record on the train task".**  The train task has no ``clip_end``
  termination (the play trims install it), so every episode would run the padded 20 s -- a
  300-frame clip becomes 6 s of motion and then 14 s of a saturated command off a finite
  tile, which is exactly what the eval harness adds ``clip_end`` to avoid.  What is wanted is
  a recorder flag that undoes the DR half of the play trims and leaves the episode-boundary
  and scene half alone.
**One action draw per state is not a gap, and is deliberately NOT a todo.**  A clip repeats
across envs (2289 clips over 8192 envs) but no two rows share a STATE -- spawn jitter, the
env's startup DR draw and the flow noise all differ at once.  That is the ordinary situation
in RL: ``V-hat`` is fitted by regression ACROSS states, so its value at ``s`` comes from the
neighbourhood rather than from repeats at ``s``, and every standard actor-critic advantage
estimator is built on exactly one action sample per visited state.  What the fit needs is
COVERAGE, which is what moving to 8192 envs bought and what ``TODO(record-camera-dr)`` above
is asking for -- DR widens the conditioning distribution the critic must generalise over.

Repeats at a state buy an empirical baseline in place of a learned one (the GRPO/RLOO idiom),
which pays when the critic is the unreliable part and when a state can be reset to for free.
Here neither holds: there is a critic, and in Isaac a reset costs a full state copy.  Where a
paired sample IS load-bearing is MEASUREMENT -- how much of an outcome is the noise draw --
and that is already answered without any restore by ``transfer/sim/fm_variance_probe.py``,
which replays a fixed noise tape from spawn in MuJoCo and is exactly deterministic.  The
label's own acceptance test is the G2 band gate in ``label_advantage.py``.

Example::

    python scripts/rsl_rl/record_rollouts.py \\
        --task=G1-tracker-clf-fm-terrain-depth-play \\
        --load_run <teacher run> --fm_checkpoint <fm.pt> \\
        --clip clips/<lib>.h5 --clip_device cpu \\
        --goal_regions logs/terrain_eval/<name>/goal_regions.json \\
        --num_envs 256 --num_windows 200000 --output_dir logs/rollouts/<name>
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import json
import math
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 - registers the gym environments

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cli_args  # noqa: E402
import rollout_outcomes as ro  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.mdp import tracking_error  # noqa: E402
from provenance import git_snapshot  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="RL agent configuration entry point.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--fm_checkpoint", type=str, default=None,
                    help="The generator that drives the tracker. REQUIRED: without one the "
                         "command falls back to the clip and the recording is a clip arm "
                         "under a rollout's name -- same shapes, different experiment.")
parser.add_argument("--history_from", type=str, default="generator",
                    choices=["generator", "clip", "measured"],
                    help="What fills the generator's N history rows during the rollout. "
                         "'generator' (default) is the deployed closed loop and is what "
                         "every rollout before this flag recorded. 'measured' is the "
                         "robot's own achieved state, bar the base position and linear "
                         "velocity. The recording carries whatever ran, so a critic and a "
                         "fine-tune built from it inherit this arm -- which is the point "
                         "when the fine-tune is meant to MOVE the generator onto it.")
parser.add_argument("--allow_history_source_mismatch", action="store_true",
                    help="Record under an arm the generator was not fitted for. Needed to "
                         "bootstrap a fine-tune onto a new arm (the rollouts ARE the "
                         "training set), and wrong for anything being measured as-is.")
parser.add_argument("--clip", type=str, default=None, help="Clip library .h5 to run on.")
parser.add_argument("--clip_device", type=str, default=None,
                    help="'cpu' keeps a large library off the GPU.")
parser.add_argument("--num_episodes", type=int, default=0,
                    help="Stop once this many episodes have FINISHED. The natural unit: with "
                         "a terminal-only reward every window of an episode carries the same "
                         "label, so the effective sample size is the episode count and a "
                         "window quota is only a proxy for it. 0 = no episode limit.")
parser.add_argument("--episodes_per_env", type=int, default=0,
                    help="Stop once EVERY env has completed this many episodes, rather than "
                         "once a TOTAL count has finished. Use this whenever episode length "
                         "varies with the commanded speed: a total-count stop fires as soon "
                         "as the fastest episodes finish, so it keeps the short ones and "
                         "discards every slow episode still in flight -- on the box2 band "
                         "that made the kept episodes last 177 steps against a population "
                         "median of 392. Rounds keep the collected set a complete design "
                         "over (env, round), whose membership cannot depend on duration. "
                         "The cost is waiting for the slowest env, bounded by "
                         "episode_length_s. 0 = use --num_episodes instead.")
parser.add_argument("--num_windows", type=int, default=100000,
                    help="Hard cap on re-plan windows; also sizes the pre-allocated file, so "
                         "set it above what --num_episodes will produce.")
parser.add_argument("--terrain_reward", action="store_true",
                    help="Grade the plan's terrain consistency ONCE PER RE-PLAN and write it "
                         "as a `chunks` group of the episode file: one row per re-plan, keyed "
                         "(episode_uid, replan_index), holding the SAME raw terms "
                         "`inspect_terrain_reward.py --stage reward` writes per window. About "
                         "60 bytes a row against ~30 kB for the window it is graded from, so "
                         "it composes with --no_windows to give a terrain-reward eval that "
                         "records nothing else. The spec it graded under is stamped on the "
                         "group, and the offline re-grade reads that rather than its own "
                         "defaults -- see tests/test_terrain_reward_parity.py.")
parser.add_argument("--no_windows", action="store_true",
                    help="Record OUTCOMES ONLY -- write the episode table and no window "
                         "table at all. An EVAL needs success, the failure mode and "
                         "`track_reward_mean`, all of which live in the episode table; the "
                         "window table is the entire cost of a recording (~7.7 GB and most "
                         "of the per-step CPU per 5000 episodes, since `_emit` assembles the "
                         "depth rows, the proprio ring and the scan for every re-plan). It "
                         "also frees the run from `--num_windows`, which is a HARD CAP on the "
                         "loop rather than only a file size, so an eval sized for episodes "
                         "could otherwise stop early. REQUIRES --num_episodes or "
                         "--max_steps, since the window quota is what would otherwise "
                         "terminate the loop.")
parser.add_argument("--max_steps", type=int, default=0,
                    help="Also stop after this many control steps (0 = no ceiling).")
parser.add_argument("--goal_regions", default="",
                    help="goal_regions.json from build_goal_regions.py. REQUIRED -- see the "
                         "module docstring for why a rollout without one cannot carry a "
                         "reward that is not maximised by standing still.")
parser.add_argument("--tracking_spec", default="",
                    help="tracking_spec.json from measure_tracking_error.py --stage report. "
                         "Supplies the strict lateral/height/tilt bounds and the reward kernel "
                         "widths. Without one the BOOTSTRAP placeholders are used and the run "
                         "says so -- the recorded error rows make the thresholds an OFFLINE "
                         "knob, so a spec chosen later is a relabel rather than a re-record.")
parser.add_argument("--min_height", type=float, default=0.40,
                    help="base_fallen's world-z threshold, m. 0.40 rather than the "
                         "historical 0.50: the deepest jump_off_front clip's own pelvis "
                         "reaches 0.525 m and a well-tracking robot undershoots by up to "
                         "60 mm, so at 0.50 a CORRECTLY tracking robot is terminated and a "
                         "critic fitted to those episodes learns the threshold, not the fall.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Where the two .h5 files go; defaults beside the checkpoint.")
parser.add_argument("--skip_families", default="jump_off,distractor,stairs_dn_pure,stairs_up_pure",
                    help="Families this rollout REFUSES to run on. The library is checked at "
                         "load and the run aborts if any are present, naming the command to "
                         "build a filtered subset -- filtering at the reset instead leaves "
                         "envs parked on tiles with no clips. Four, for three reasons:\n"
                         "  jump_off      -- jumps BACKWARDS, so a forward half-plane puts "
                         "its goal behind the spawn and most of its episodes score free.\n"
                         "  distractor    -- travels less, in total, than the approach "
                         "margin, so the margin swallows the whole task.\n"
                         "  stairs_*_pure -- these clips OPEN ON THE STAIRS, so the "
                         "generator's first re-plan after every reset runs on the learned "
                         "NULL-history embedding while standing on a riser. That embedding "
                         "was fitted overwhelmingly on flat ground, so those first samples "
                         "are bad for a reason that belongs to the reset rather than to the "
                         "generator, and a critic fitted on them learns the artefact. They "
                         "are also the only stairs tiles with NO width variation (uniform "
                         "5.2 m against 0.81-4.14 m on the transition families), so dropping "
                         "them costs no geometric coverage.\n"
                         "Empty = run everything, and read `goal_travel_m` first.")
parser.add_argument("--start_step", type=int, default=0,
                    help="Pin every episode to open at this clip step. 0 (the default) makes "
                         "every episode run the whole clip, so every one has the full task to "
                         "do. -1 restores the clip's own random start phase, which gives more "
                         "varied initial states BUT opens many episodes at or past the clip's "
                         "own endpoint, so a large fraction of arrivals are already inside "
                         "the goal region on their first step and score a free success. "
                         "Those are flagged `spawned_in_goal` either way, so a fit "
                         "can drop them -- but do not leave the phase on and forget to.")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def _apply_overrides(env_cfg) -> None:
    """Put the env into the closed-loop configuration, and refuse anything else.

    Args:
        env_cfg: The composed env config, mutated in place.

    Raises:
        ValueError: If the task is not the FM tracker, or carries no ``base_fallen``.
    """
    cmd = getattr(getattr(env_cfg, "commands", None), "traj_ref", None)
    if cmd is None or not hasattr(cmd, "reference_source"):
        raise ValueError(
            "this task's commands.traj_ref is not an FmTrackingCommand; a rollout needs one "
            "(G1-tracker-clf-fm-terrain-depth-play and friends).")
    cmd.reference_source = "fm"
    cmd.history_from = args_cli.history_from
    # Recording under an arm the checkpoint was not fitted for is the POINT when these
    # rollouts are the fine-tune's training set, so the refusal is lifted here and only
    # here -- and only when the arm actually differs, so a matched run keeps the guard.
    cmd.allow_history_source_mismatch = bool(args_cli.allow_history_source_mismatch)
    cmd.fm_zero_noise = False
    if args_cli.fm_checkpoint is not None:
        cmd.fm_checkpoint = args_cli.fm_checkpoint
    if not getattr(cmd, "fm_checkpoint", ""):
        raise ValueError("a rollout needs --fm_checkpoint; see the flag's help.")
    # The height-fall term is spelled differently on the two task families and is the SAME
    # test either way -- `mdp.base_height_below_termination` on absolute world z, taking
    # `min_height`. The tracker calls it `base_fallen`; the fine-tune tasks call it
    # `base_height` (`G1FinetuneTerminationCfg`, whose other failure term is a
    # reference-relative tilt). Accepting both by NAME rather than renaming either: the names
    # are load-bearing in their own configs and downstream tables.
    terms = getattr(env_cfg, "terminations", None)
    fallen = next((t for t in (getattr(terms, "base_fallen", None),
                               getattr(terms, "base_height", None)) if t is not None), None)
    if fallen is None:
        raise ValueError(
            "this task has no `base_fallen` (tracker) or `base_height` (fine-tune) "
            "termination, which is the only world-z FAILURE term and therefore the only "
            "thing a reward can read as a fall.")
    fallen.params["min_height"] = float(args_cli.min_height)
    if args_cli.clip is not None:
        cmd.path = args_cli.clip
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain is not None and hasattr(terrain, "clip_library_path"):
            terrain.clip_library_path = args_cli.clip
    if args_cli.clip_device is not None:
        cmd.clip_device = args_cli.clip_device
    # "cfg:" on purpose. This runs before the generator's context is built, so it can only
    # echo what was just assigned -- which is not the value the sampler ends up reading.
    # `_assert_sampler_is_stochastic` prints the resolved truth, and measures it.
    # A clip-free task spawns from the manifest's route block (`reset_on_section`) and has no
    # clip PHASE at all, so there is no start step to pin -- and pinning one would be a
    # parameter the term does not take. Refuse a non-default `--start_step` there rather than
    # dropping it silently: it would read as having been applied.
    events = getattr(env_cfg, "events", None)
    reset = getattr(events, "reset_on_clip", None)
    if reset is None:
        section = getattr(events, "reset_on_section", None)
        if section is None:
            raise ValueError(
                "this task has neither a `reset_on_clip` event to pin the start step on nor "
                "a `reset_on_section` one to spawn from a manifest route.")
        if int(args_cli.start_step) > 0:
            raise ValueError(
                f"--start_step {args_cli.start_step} was given, but this task spawns from a "
                "free-form section's route block and has no clip phase to pin. Drop the flag.")
    else:
        reset.params["pin_start_step"] = (None if int(args_cli.start_step) < 0
                                          else int(args_cli.start_step))
    print(f"[rollout] cfg: reference_source=fm history_from={args_cli.history_from} "
          f"allow_history_source_mismatch={bool(args_cli.allow_history_source_mismatch)} "
          f"fm_zero_noise=False "
          f"fall_min_height={args_cli.min_height} "
          + (f"pin_start_step={reset.params['pin_start_step']}" if reset is not None
             else "spawn=free-form route (no clip phase to pin)"))


def _assert_families_are_filtered(codes, names) -> None:
    """Refuse a library still containing a family this rollout must not run on.

    A GUARD, not a filter, and the distinction is load-bearing: dropping families at the
    reset leaves envs parked on tiles whose clips have all been excluded, which is why the
    documented route is a filtered subset LIBRARY.  So this names the command rather than
    trying to do the job here.

    Strictness is asymmetric, matching ``dataset.resolve_skip_families``: a name in the
    default set that the library does not carry is fine and silent (a flat-only library has
    no families at all), because the default is a standing request rather than an assertion
    about this particular library.

    Args:
        codes: ``[num_clips]`` family codes, or ``None`` for a library with no manifest.
        names: The name table those codes index.

    Raises:
        ValueError: When a skipped family is present, naming the exact plan command.
            Deliberately NOT ``SystemExit``: hydra's ``main`` wrapper swallows that one, so
            the guard fired, printed nothing, wrote no data and exited 0 -- a silent success
            with an empty output directory, which is a worse failure than the one being
            guarded against.
    """
    wanted = [f.strip() for f in str(args_cli.skip_families).split(",") if f.strip()]
    if not wanted or codes is None:
        return
    drop = {n.strip() for n in _normalize_families(wanted)}
    present = sorted({nm for c, nm in zip(*_present(codes, names)) if nm in drop})
    if not present:
        return
    raise ValueError(
        f"the clip library still contains {present}, which this rollout is set to skip.\n"
        f"Build a filtered subset first:\n"
        f"  python scripts/rsl_rl/eval/evaluate.py --stage plan --lib <library> \\\n"
        f"      --load_run <teacher run> --fm_checkpoint gen=<fm.pt> \\\n"
        f"      --exclude_families {','.join(sorted(drop))} --out <dir>\n"
        f"then pass --clip <dir>/eval_clips.h5. Filtering at the reset instead would leave "
        f"envs parked on tiles whose clips were all excluded.")


def _present(codes, names):
    """``(codes, normalised names)`` for the families a library actually contains."""
    import numpy as _np  # noqa: PLC0415

    seen = sorted({int(c) for c in codes.tolist()})
    return seen, _normalize_families([names[c] if c < len(names) else "unknown" for c in seen])


def _normalize_families(raw):
    """Strip ``family_`` and ``.h5``, the way the eval's sampler spells a family."""
    return ro._normalized(list(raw))


def _assert_sampler_is_stochastic(cmd) -> None:
    """Draw the generator TWICE from identical conditioning and require the draws to differ.

    This is the one setting that silently guts the method, so it is checked by MEASUREMENT
    rather than by reading a flag.  At ``x0 = 0`` -- the default in every deploy runtime, in
    ``gen_tracker_runner.py`` and in ``mjeval`` -- the flow is deterministic, so every chunk
    drawn at a given state is identical, ``A(s, a)`` collapses to ``A(s)``, and the advantage
    label says how hard the STATE was and nothing about which ACTION was better.  A rollout
    recorded that way is well formed, joins correctly, trains without error, and carries no
    signal.

    Setting ``cfg.fm_zero_noise = False`` is not evidence that it took.  The value the
    sampler actually reads is ``ctx.zero_noise``, three hops away
    (``cfg -> load_fm_context -> FmOnlineContext -> sample_trajectory``), and any future path
    that supplies an explicit ``noise`` tensor or reuses a context would bypass the flag
    entirely.  Two draws either differ or they do not.

    Args:
        cmd: The ``FmTrackingCommand``, after its context has loaded.

    Raises:
        RuntimeError: If the two draws are bit-identical.
    """
    from robot_rl.flow_matching.online import (  # noqa: PLC0415
        build_cmd_tokens, build_extero_tokens, build_history_tokens, sample_window)

    ctx = cmd.ctx
    with torch.inference_mode():
        # A NULL history of zeros -- exactly what the model is given for the first N steps
        # after every reset, and available before any step has run (the ring is empty then,
        # and `stack_padded` rightly refuses to invent one). The conditioning's CONTENT is
        # irrelevant here: the question is only whether two calls on the same input agree.
        nodes = torch.zeros(1, ctx.N, ctx.D, device=ctx.device)
        null = torch.ones(1, dtype=torch.bool, device=ctx.device)
        hist = build_history_tokens(ctx, nodes)
        ext = build_extero_tokens(ctx, cmd._lagged_extero()[:1])
        cmds = build_cmd_tokens(ctx, cmd.get_command_velocity_horizon(ctx.F)[:1])
        a = sample_window(ctx, hist, ext, cmds, history_null_mask=null)
        b = sample_window(ctx, hist, ext, cmds, history_null_mask=null)
    delta = float((a - b).abs().max())
    if delta == 0.0:
        raise RuntimeError(
            "the generator is sampling DETERMINISTICALLY: two draws from identical "
            f"conditioning were bit-identical (ctx.zero_noise={ctx.zero_noise}). Every chunk "
            "at a given state would then be the same, so the advantage label would carry no "
            "information about the action and the whole recording would be wasted. Check "
            "that nothing overrode commands.traj_ref.fm_zero_noise.")
    print(f"[rollout] sampler is stochastic: two draws from identical conditioning differ by "
          f"max {delta:.4g} (ctx.zero_noise={ctx.zero_noise})")


ProprioRing = ro.ProprioRing
"""Re-exported: the class moved to rollout_outcomes so both consumers share one definition."""


def _twist_yaw(rd):
    """The measured base twist in the robot's YAW frame, ``(N, 6)``.

    ``yaw_quat(root_quat_w)`` is the level frame the generator predicts its window in -- z is
    gravity, roll and pitch are zero -- so this is directly comparable with the plan's own
    `base_vel` block without a transform. The body frame would tilt with the torso and swing
    the forward component on every pitch; world would defeat heading invariance.

    Args:
        rd: The converting view of ``ArticulationData``.

    Returns:
        ``(N, 6)``: linear(3) then angular(3).
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: PLC0415

    q = yaw_quat(rd.root_quat_w)
    return torch.cat([quat_apply_inverse(q, rd.root_lin_vel_w),
                      quat_apply_inverse(q, rd.root_ang_vel_w)], dim=-1)


def _replanned(cmd, tracker):
    """Claim ``(episode_uid, replan_index)`` for the envs that re-planned this step.

    **Hoisted out of :func:`_emit` because the claim must happen in BOTH modes.**  ``_emit``
    is skipped entirely under ``--no_windows`` -- it is most of the per-step cost and none of
    it reaches the episode table -- so while the claim lived inside it, a windowless run left
    ``replan_index`` at zero and wrote ``n_replans = 0`` on every episode.  The terrain chunk
    table is keyed on exactly that index and is most useful in precisely that mode, so the
    counter now advances whether or not a window is written, and both consumers are handed
    the same claim rather than each taking their own.

    Args:
        cmd: The ``FmTrackingCommand``.
        tracker: The :class:`rollout_outcomes.RolloutEpisodeTracker`.

    Returns:
        ``(idx, uid, ridx)`` for the re-planning envs, or ``None`` when none did.
    """
    mask = cmd._fm_reinfer_mask
    if not bool(mask.any()):
        return None
    idx = mask.nonzero(as_tuple=False).flatten()
    uid, ridx = tracker.on_replan(idx)
    return idx, uid, ridx


def _terrain_chunk_batch(env, command_name: str, claim, tracker) -> dict:
    """Grade the live plan for the re-planning envs, as one chunk-table batch.

    Reads :func:`mdp.plan_terrain_terms`, which is the cached producer the three terrain
    reward terms read, so the row recorded here is the number those terms would report.

    **Graded over the WHOLE plan, which is what the critic was fitted on.**  The sidecar's
    penalty is built from the unsuffixed ``contact_penalty`` / ``penetration_sum_m``, i.e. all
    62 nodes; the ``_served`` twins are recorded beside them but are not that number.  A
    served-only grade would score the 12 nodes that get executed and miss the tail, which is
    where a plan that has read a staircase as a box commits to the jump.

    Args:
        env: The unwrapped env.
        command_name: The :class:`FmTrackingCommand` term name.
        claim: The ``(idx, uid, ridx)`` from :func:`_replanned`.
        tracker: The episode tracker, for the goal anchor the downrange is measured from.

    Returns:
        A batch keyed by :data:`rollout_outcomes.ROLLOUT_CHUNK_NAMES`.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp import plan_terrain_terms  # noqa: PLC0415

    idx, uid, ridx = claim
    terms = plan_terrain_terms(env, command_name)
    # WHERE this re-plan happened, through the same function the window table and the outcome
    # fold use, so the three cannot disagree about which way a tile faces.
    from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region  # noqa: PLC0415
    down, _lat, _hz = goal_region.goal_frame_offsets(
        tracker.accum.rd.root_pos_w, tracker.accum._goal_pos_w, tracker.accum._goal_yaw)
    out = {"episode_uid": uid, "replan_index": ridx,
           "env_id": idx.detach().to("cpu").numpy().astype(np.int32),
           "goal_downrange_m": down[idx].detach().to("cpu").numpy().astype(np.float32)}
    for name in ro.TERRAIN_CHUNK_TERMS:
        if name not in terms:
            raise KeyError(
                f"terrain_consistency returned no {name!r}; the chunk table's schema and the "
                f"grader have diverged. Present: {sorted(k for k in terms)}")
        out[name] = terms[name][idx].detach().to("cpu").numpy().astype(np.float32)
    return out


def _emit(cmd, tracker, family_codes, scan_sensor, env, ring, track_ring=None,
          claim=None) -> dict | None:
    """Assemble one batch of windows for the envs that re-planned this step.

    Everything here is read at ONE instant -- after ``env.step`` has run
    ``_update_command``, so the mask, the plan, the history ring and the served exteroception
    all describe the same re-plan. Reading the plan a step later would give the next one.

    Args:
        cmd: The ``FmTrackingCommand``.
        tracker: The :class:`rollout_outcomes.RolloutEpisodeTracker`.
        family_codes: ``[num_clips]`` family codes, or ``None``.
        scan_sensor: Privileged scan sensor name, or ``None``.
        env: The unwrapped env.
        ring: The :class:`ProprioRing`.
        track_ring: The :class:`TrackErrRing`, or ``None`` when the strict checks are off.
        claim: The ``(idx, uid, ridx)`` from :func:`_replanned`.  Required -- passing it is
            what keeps the window and chunk tables on ONE claim, since ``on_replan`` bumps
            the index and calling it twice per step would number them apart.

    Returns:
        The batch, or ``None`` when no env re-planned.
    """
    if claim is None:
        return None
    idx, uid, ridx = claim
    cpu = lambda t: t.detach().to("cpu").numpy()  # noqa: E731

    nominal = int(getattr(cmd, "_depth_lag_steps", 0) or 0)
    held = len(getattr(cmd, "_extero_buf", ()) or ())
    served = min(nominal, max(held - 1, 0))

    batch = {
        "episode_uid": uid,
        "replan_index": ridx,
        "env_id": cpu(idx).astype(np.int32),
        "clip_id": cpu(cmd.clip_id[idx]).astype(np.int32),
        "family_code": (cpu(family_codes[cmd.clip_id[idx].to(family_codes.device)])
                        .astype(np.uint8) if family_codes is not None
                        else np.zeros(idx.numel(), dtype=np.uint8)),
        "history": cpu(cmd._history.stack_padded("node")[idx]).astype(np.float32),
        "history_null": cpu(cmd._history.null_mask()[idx]).astype(np.uint8),
        "extero": cpu(cmd._lagged_extero()[idx]).astype(np.float16),
        "cmd_future": cpu(cmd.get_command_velocity_horizon(cmd.ctx.F)[idx]).astype(np.float32),
        "depth_lag_used": np.full(idx.numel(), served, dtype=np.uint8),
        # The behaviour-policy arms this chunk was generated under. Read from the command
        # term's own flags at the instant of the re-plan, which is when the corruption was
        # applied -- `history` and `extero` above are the TRUE measurement either way.
        "perturb_null_history": cpu(cmd.history_nulled()[idx]).astype(np.uint8),
        "perturb_noisy_upper_cam": cpu(cmd.upper_camera_noised()[idx]).astype(np.uint8),
        "fm_plan": cpu(cmd._pred_window[idx]).astype(np.float32),
        "anchor_pos": cpu(cmd._anchor_pos[idx]).astype(np.float32),
        "anchor_quat": cpu(cmd._anchor_quat[idx]).astype(np.float32),
        "clip_step": cpu(cmd._step[idx]).astype(np.int32),
    }
    # Where the goal is FROM HERE, in the terrain frame. Read through the same function the
    # outcome fold uses, so the per-window series and the terminal row cannot disagree about
    # which way a tile faces.
    from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region  # noqa: PLC0415
    down, lat, hz = goal_region.goal_frame_offsets(
        tracker.accum.rd.root_pos_w,
        tracker.accum._goal_pos_w, tracker.accum._goal_yaw)
    batch["goal_downrange_m"] = cpu(down[idx]).astype(np.float32)
    batch["goal_lateral_m"] = cpu(lat[idx]).astype(np.float32)
    batch["goal_height_err_m"] = cpu(hz[idx]).astype(np.float32)
    batch["proprio"] = cpu(ring.buf[idx]).astype(np.float32)
    if track_ring is not None:
        # Drop the NEWEST row: the re-plan is applied inside `env.step`, so the step just
        # folded is already served by the new chunk. What is left is exactly the K steps the
        # PREVIOUS chunk drove -- hence the column's name and its one-row join shift.
        batch["track_err_prev"] = cpu(track_ring.buf[idx, :-1]).astype(np.float32)
    batch["base_twist_yaw"] = cpu(_twist_yaw(tracker.accum.rd)[idx]).astype(np.float32)
    if scan_sensor is not None:
        from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: PLC0415
        scan = mdp.scan_obs_stack(env, scan_sensor)[:, 0]
        batch["height_scan"] = cpu(scan[idx]).astype(np.float16)
    return batch


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    """Roll the frozen pair out and stream both tables."""
    if not args_cli.goal_regions:
        raise ValueError(
            "--goal_regions is required. Without it the only outcomes available are "
            "clip-CURSOR conditions, which cannot tell a robot that climbed the terrain "
            "from one that stood still while the cursor ran out -- and a reward built on "
            "those is maximised by standing still. Build one with build_goal_regions.py.")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, metadata.version("rsl-rl-lib"))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)),
            agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    out_dir = args_cli.output_dir or os.path.join(log_dir, "rollouts")

    with launch_simulation(env_cfg, args_cli):
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401
        from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: PLC0415
        from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region  # noqa: PLC0415

        sys.path.insert(0, os.path.join(_HERE, "eval"))
        import accumulate  # noqa: PLC0415

        env_cfg.scene.num_envs = (args_cli.num_envs if args_cli.num_envs is not None
                                  else env_cfg.scene.num_envs)
        env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
        _apply_overrides(env_cfg)

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                                device=agent_cfg.device)
        # ACTOR ONLY. A rollout runs the policy in inference; the critic is never asked for
        # a value, and loading it couples this script to whatever critic width the checkpoint
        # happens to carry -- `model_*_padded.pt` from the fine-tune work is 2533 against the
        # task's 2490, so a run dir holding one fails here for a reason that has nothing to do
        # with recording.
        runner.load(resume_path, load_cfg={"actor": True, "critic": False,
                                           "optimizer": False, "iteration": True,
                                           "rnd": True})
        print(f"[rollout] tracker checkpoint: {resume_path}")
        policy = runner.get_inference_policy(device=u.device)
        cmd = u.command_manager.get_term("traj_ref")

        spec = goal_region.GoalRegionSpec.load(args_cli.goal_regions)
        if spec.provenance.get("bootstrap"):
            raise ValueError(
                f"{args_cli.goal_regions} is a BOOTSTRAP spec: its tolerances are "
                "placeholders chosen only to be drawable, so every outcome recorded under "
                "it would be measured against a guess. Fit the real one from a clip-arm run "
                "(build_goal_regions.py <run_dir> --out ...).")
        family_codes, family_names = _clip_families(cmd)
        _assert_families_are_filtered(family_codes, family_names)
        accum = accumulate.EpisodeAccumulator(env, dt=float(u.step_dt),
                                              fall_min_height=float(args_cli.min_height))
        accum.attach_goal_region(spec)
        # The STRICT tracking checks. Latched, never terminating: the episode runs on, so one
        # rollout yields both the strict and the loose verdict and a threshold change
        # afterwards is an offline relabel rather than a re-collection. `--tracking_spec`
        # supplies fitted bounds; without one the BOOTSTRAP placeholders are used and said so,
        # on the same terms as a bootstrap goal region.
        _spec = tracking_error.BOOTSTRAP_SPEC
        if getattr(args_cli, "tracking_spec", ""):
            with open(args_cli.tracking_spec) as _f:
                _spec = tracking_error.TrackingErrorSpec.from_dict(json.load(_f))
        print(f"[tracking] lateral {_spec.max_lateral_m:.3f} m, height {_spec.max_height_m:.3f} m, "
              f"tilt {math.degrees(_spec.max_tilt_rad):.1f} deg"
              + ("  (BOOTSTRAP placeholders -- fit them before publishing a strict number)"
                 if _spec.is_bootstrap else f"  (from {args_cli.tracking_spec})"))
        accum.attach_tracking(_spec)
        table = ro.RolloutOutcomeTable(family_names=family_names)
        # The terrain grade is a property of the PLAN and the ground, so it needs no rollout
        # state beyond the command term -- but the spec it ran under does have to travel, or
        # an offline re-grade silently applies its own defaults to rows graded under others.
        chunks = None
        if bool(args_cli.terrain_reward):
            from motion_gen import plan_geometry as _pg  # noqa: PLC0415
            from motion_gen.terrain_consistency import TerrainConsistencySpec  # noqa: PLC0415
            _tc = TerrainConsistencySpec()
            chunks = ro.ChunkRewardTable(spec_meta={
                "contact_speed_thresh": float(_tc.contact_speed_thresh),
                "contact_dilate_nodes": int(_tc.contact_dilate_nodes),
                "contact_max_gap_nodes": int(_pg.CONTACT_MAX_GAP_NODES),
                "collider_spacing": float(_pg.DEFAULT_COLLIDER_SPACING),
                "penetration_tol_m": float(_tc.penetration_tol_m),
                "contact_deadband_m": float(_tc.contact_deadband_m),
                "contact_weight": float(_tc.contact_weight),
                "penetration_weight": float(_tc.penetration_weight),
                "penetration_default_weight": float(_tc.penetration_default_weight),
                "body_penetration_weights": dict(_tc.body_penetration_weights),
                "served_nodes": int(_tc.served_nodes),
                "graded": "whole_plan",
            })
            print(f"[terrain] grading every re-plan over the WHOLE plan "
                  f"(spacing {_pg.DEFAULT_COLLIDER_SPACING:.3f} m, "
                  f"thresh {_tc.contact_speed_thresh:.3f} m/s)")
        # The reset term the episode tracker wraps, so it can refresh the goal anchor INSIDE
        # the reset -- after the new episode has committed. A clip-free task spawns from the
        # manifest's route block instead, and the term is named for what it does.
        _reset_name = next(
            (n for n in ("reset_on_clip", "reset_on_section")
             if n in u.event_manager.active_terms.get("reset", [])), None)
        if _reset_name is None:
            raise ValueError(
                "no reset term to hang the episode tracker on: expected `reset_on_clip` "
                "(clip tasks) or `reset_on_section` (free-form). Without it the goal anchor "
                "is never refreshed and every episode folds against its predecessor's goal.")
        term_cfg = u.event_manager.get_term_cfg(_reset_name)
        tracker = ro.RolloutEpisodeTracker(term_cfg.func, env=u, accum=accum, table=table,
                                           family_codes=family_codes, goal_spec=spec)
        term_cfg.func = tracker
        # The FIRST episode never passes through the reset hook -- its clip was committed
        # before the rollout began -- so its goal anchor is primed here or the whole first
        # batch folds against a zero goal and reads as never arriving.
        tracker._refresh_goal(u, None)

        scan_sensor = mdp.fm_scan_sensor(u)
        scan_dim = (int(mdp.scan_obs_stack(u, scan_sensor).shape[-1])
                    if scan_sensor is not None else 0)
        # The scan's (height, width), read from its own contract rather than factorised out
        # of the ray count. 176 is 11x16 and also 16x11, and under a convolution a transposed
        # grid is scrambled input rather than a reordered vector -- the exact hazard
        # `test_height_scan_grid.py` exists for.
        scan_hw = None
        if scan_sensor is not None:
            scan_spec = mdp.height_scan_specs(u, scan_sensor)[0]
            scan_hw = [int(scan_spec["height"]), int(scan_spec["width"])]
        extero = cmd._lagged_extero()
        ring = ProprioRing(int(u.num_envs), u.device, ro.PROPRIO_HISTORY, ro.PROPRIO_DIM)
        # The per-chunk tracking-error block. Depth is `fm_update_every` exactly -- the steps
        # one emitted chunk drove -- so a window's `track_err` is the error the chunk in the
        # SAME row produced, not the previous one's.
        track_dim = tracking_error.NUM_FIXED_CHANNELS + int(u.scene.articulations["robot"].num_joints)
        track_hw = (int(cmd.cfg.fm_update_every), track_dim)
        # K+1 deep, one row deeper than what is WRITTEN: the newest row belongs to the chunk
        # just emitted, whose consequences have not happened yet. See `track_err_prev`.
        track_ring = ro.TrackErrRing(int(u.num_envs), u.device, track_hw[0] + 1, track_hw[1])
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"rollout_{os.path.basename(log_dir)}")
        win_path, ep_path = stem + ".h5", stem + ".episodes.h5"
        prov = {
            "checkpoint_path": os.path.abspath(resume_path),
            "fm_checkpoint": str(getattr(cmd.cfg, "fm_checkpoint", "")),
            "clip_library_path": str(getattr(cmd.cfg, "path", "")),
            "task_name": str(args_cli.task),
            "goal_regions": os.path.abspath(args_cli.goal_regions),
            "fall_min_height": float(args_cli.min_height),
            # The RESOLVED value the sampler reads, not the flag that was set. A literal
            # here would let the provenance claim Gaussian noise on a run that had none.
            "fm_zero_noise": bool(cmd.ctx.zero_noise),
            # What filled the generator's history rows, and what the checkpoint driving them
            # was FITTED on. Both, because a fine-tune bootstrapping onto a new arm runs with
            # them deliberately different, and everything built from this file -- the critic,
            # the advantages, the filtered BC -- inherits the arm without being able to see
            # it otherwise: the rows are the same width either way.
            "history_from": str(cmd.cfg.history_from),
            "fm_history_source": str(getattr(cmd.ctx, "history_source", "reference")),
            "fm_update_every": int(cmd.cfg.fm_update_every),
            "num_envs": int(u.num_envs),
            "proprio_history": int(ro.PROPRIO_HISTORY),
            "proprio_layout": ["joint_pos(29)", "joint_vel(29)", "base_ang_vel_b(3)",
                               "projected_gravity_b(3)"],
            "base_twist_yaw_layout": ["lin(3)", "ang(3)", "frame=yaw_quat(root_quat_w)"],
            "height_scan_hw": scan_hw,
            # WHAT THE STRICT COLUMNS MEAN. Without this the file's `first_violation_step`
            # and `track_reward_mean` are only interpretable by knowing which code ran, which
            # is the thing every other contract in this tree refuses. The error ROWS are
            # spec-independent, so a later spec is a relabel -- but the columns written HERE
            # were written under THIS one and have to say so.
            "tracking_spec": _spec.to_dict(),
            "tracking_spec_path": (os.path.abspath(args_cli.tracking_spec)
                                   if args_cli.tracking_spec else ""),
            # Channel names of one `track_err_prev` row, for the same reason `proprio_layout`
            # is stamped: the block is (K, NUM_FIXED + nj) and a reader that guessed the order
            # would get a plausible number from the wrong channel.
            "track_err_layout": list(
                tracking_error.error_channel_names(
                    int(u.scene.articulations["robot"].num_joints))),
            "track_err_is_previous_chunk": True,
        }
        print(f"[rollout] goal region {args_cli.goal_regions} (default margin "
              f"{spec.default.approach_margin_m} m, half-width {spec.default.half_width_m} m, "
              f"height {spec.default.height_tol_m} m)")
        _assert_sampler_is_stochastic(cmd)
        print(f"[rollout] F={cmd.ctx.F} D={cmd.ctx.D} N={cmd.ctx.N} K={cmd.cfg.fm_update_every} "
              f"extero={tuple(extero.shape[1:])} scan={scan_dim} track_err={track_hw}")

        obs = env.get_observations()
        steps = 0
        t0 = time.time()
        ceiling = int(args_cli.max_steps) or None
        no_win = bool(args_cli.no_windows)
        if no_win and not (int(args_cli.num_episodes) or int(args_cli.episodes_per_env)
                           or ceiling):
            raise SystemExit(
                "--no_windows needs --num_episodes (or --max_steps): the window quota is "
                "what bounds the loop otherwise, so without one of these the recording "
                "would never stop.")
        writer_cm = (contextlib.nullcontext(_NullWindowWriter()) if no_win
                     else ro.RolloutWindowWriter(
            win_path, num_windows=int(args_cli.num_windows), num_history=int(cmd.ctx.N),
            state_dim=int(cmd.ctx.D), future_len=int(cmd.ctx.F), cmd_dim=int(cmd.ctx.cmd_dim),
            extero_shape=(int(extero.shape[1]), int(extero.shape[2])),
            proprio_hw=(ro.PROPRIO_HISTORY, ro.PROPRIO_DIM), scan_dim=scan_dim,
            track_hw=track_hw, attrs=prov))
        with writer_cm as writer:
            with torch.inference_mode():
                want_ep = int(args_cli.num_episodes) or None
                want_rounds = int(args_cli.episodes_per_env) or None
                while no_win or writer.written < int(args_cli.num_windows):
                    if ceiling is not None and steps >= ceiling:
                        print(f"[rollout] step ceiling {ceiling} reached")
                        break
                    if want_rounds is not None and tracker.rounds_complete() >= want_rounds:
                        print(f"[rollout] every env completed {want_rounds} episode(s); "
                              f"{tracker.harvested} harvested")
                        break
                    if want_ep is not None and tracker.harvested >= want_ep:
                        print(f"[rollout] {tracker.harvested} episodes reached")
                        break
                    obs, _, dones, _ = env.step(policy(obs))
                    steps += 1
                    tracker.fold(dones)
                    # Zero the ring for envs that just reset BEFORE pushing this step, or a
                    # window emitted in the first H steps of a new episode carries rows from
                    # the previous one -- which for a fall is precisely the signature the
                    # critic is being asked to learn.
                    ring.invalidate(dones)
                    ring.push(tracker.accum.rd)
                    if track_ring is not None:
                        # Same order and the same reason as the proprio ring: zero the envs
                        # that reset BEFORE pushing, or the first windows of a new episode
                        # carry the previous one's error rows -- which for a fall is exactly
                        # the signature the critic is being asked to learn.
                        track_ring.invalidate(dones)
                        track_ring.push(tracker.accum.last_track_err)
                    # Skipped entirely under --no_windows: `_emit` is where the depth
                    # rows, the proprio ring and the scan are assembled per re-plan, so it is
                    # most of the per-step cost and none of it reaches the episode table.
                    # ONE claim per re-plan, shared: `on_replan` bumps `replan_index`, so
                    # letting the window and chunk paths each claim would number them apart
                    # and break the join they exist to support.
                    claim = _replanned(cmd, tracker)
                    if claim is not None and chunks is not None:
                        chunks.append_batch(
                            _terrain_chunk_batch(u, "traj_ref", claim, tracker))
                    if not no_win:
                        batch = _emit(cmd, tracker, family_codes, scan_sensor, u, ring,
                                      track_ring, claim=claim)
                        if batch is not None:
                            writer.write_batch(batch)
                    if steps % 500 == 0:
                        el = time.time() - t0
                        # flush=True: this is the ONLY progress signal a multi-hour run
                        # emits, and with stdout redirected to a file Python buffers it ~8 KB
                        # -- so a healthy run looks stalled for tens of minutes while the
                        # startup lines (which do flush) sit there alone.
                        print(f"[rollout] step {steps}: {writer.written} windows, "
                              f"{tracker.harvested} episodes, {el / 60:.1f} min "
                              f"({steps / max(el, 1e-9):.1f} steps/s, "
                              f"{tracker.harvested / max(el / 3600, 1e-9):.0f} ep/h)",
                              flush=True)
                # Inside inference_mode: the accumulators were last written here, which makes
                # them inference tensors, and those cannot be updated in place outside it.
                tracker.flush(u, torch.arange(int(u.num_envs), device=u.device))
        _report_goal_sanity(table)
        table.write(ep_path, attrs=prov, chunks=chunks)
        git_snapshot(stem + ".provenance")
        if no_win:
            print("[rollout] --no_windows: no window table written (outcomes only)")
        else:
            print(f"[rollout] {writer.written} windows -> {win_path}")
        print(f"[rollout] {len(table)} episodes -> {ep_path}")
        if chunks is not None:
            print(f"[rollout] {len(chunks)} terrain chunks -> {ep_path}:/chunks")
        env.close()


class _NullWindowWriter:
    """A window writer that writes nothing, for ``--no_windows``.

    A stand-in rather than a branch at every call site: the loop's progress print and its
    closing report both read ``written``, and threading ``None`` through them would put an
    ``if`` on three lines that have nothing to do with the choice.  ``write_batch`` is never
    reached under the flag (``_emit`` is skipped) and is a no-op rather than an error so that
    the object cannot half-work if that ever changes.
    """

    written: int = 0

    def write_batch(self, batch) -> None:
        """Discard a batch of windows."""
        return None


def _report_goal_sanity(table) -> None:
    """Print, per family, how far the clip travelled and how often the goal was free.

    Printed every run because the alternative is finding it by drawing six episodes and
    looking, which is how it was found the first time. A family with negative travel has its
    goal BEHIND the spawn -- a forward half-plane cannot express that -- and one whose
    spawned-inside rate is high has a goal too close to be a task.

    Args:
        table: The :class:`rollout_outcomes.RolloutOutcomeTable` about to be written.
    """
    if not len(table):
        return
    cols = {k: np.asarray([r[k] for r in table._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    live = ~cols["censored"].astype(bool)
    if not live.any():
        return
    print(f"\n[rollout] goal sanity over {int(live.sum())} finished episodes")
    print(f"    {'family':<18} {'n':>5} {'clip travel':>13} {'spawned inside':>16}")
    flagged = []
    for code in sorted(set(cols["family_code"][live].tolist())):
        m = live & (cols["family_code"] == code)
        name = (table.family_names[int(code)] if int(code) < len(table.family_names)
                else "unknown")
        name = ro._normalized([name])[0]
        travel = float(np.median(cols["goal_travel_m"][m]))
        free = float(cols["spawned_in_goal"][m].mean())
        mark = "  <-- goal BEHIND the spawn" if travel <= 0 else (
            "  <-- mostly free" if free > 0.25 else "")
        if mark:
            flagged.append(name)
        print(f"    {name:<18} {int(m.sum()):5d} {travel:12.2f} m {100*free:15.1f}%{mark}")
    if flagged:
        print(f"    ^ {flagged} contribute episodes with little or no task. Exclude them from "
              "the library (--skip_families) rather than widening the region: a goal behind "
              "the robot is not a tolerance problem.")


def _clip_families(cmd):
    """Per-clip family codes and the name table, from the library's terrain manifest.

    Args:
        cmd: The command term.

    Returns:
        ``(codes [num_clips] long | None, names)``. ``(None, ["unknown"])`` for a library
        with no manifest, which is what a flat-only library legitimately has.
    """
    import json  # noqa: PLC0415

    lib = getattr(cmd.manager, "lib", None)
    tid = getattr(lib, "terrain_id", None) if lib is not None else None
    manifest = lib.attrs.get("terrain_manifest") if lib is not None and hasattr(lib, "attrs") \
        else None
    if tid is None or tid.numel() == 0 or not manifest:
        return None, ["unknown"]
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    names, tid_to_code = ["unknown"], {}
    for entry in manifest:
        if "terrain_id" not in entry:
            continue
        fam = str(entry.get("family") or "unknown")
        fam = fam[:-3] if fam.endswith(".h5") else fam
        if fam not in names:
            names.append(fam)
        tid_to_code[int(entry["terrain_id"])] = names.index(fam)
    codes = torch.zeros(tid.shape[0], dtype=torch.long, device=tid.device)
    for t, code in tid_to_code.items():
        codes[tid.to(torch.long) == t] = code
    counts = {names[c]: int((codes == c).sum()) for c in sorted(set(codes.tolist()))}
    print(f"[rollout] clip families: {counts}")
    return codes, names


if __name__ == "__main__":
    main()
