# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play a trained RSL-RL policy and record (observation, action) pairs to disk.

The recorded dataset is intended for downstream supervised learning. Each row is
one (policy observation, action) pair, where the action is exactly what the
policy produced from that observation. Alongside the pairs we record:
  - ``terminated``: 1 if the pair led to a true termination (failure), excluding
    time-outs -- i.e. the last pair before the robot terminated.
  - ``done`` / ``time_out``: the combined episode-end flag and the time-out flag,
    so the consumer can distinguish failures from episode-length cutoffs.
  - ``episode_id`` / ``step_in_episode``: bookkeeping for episode-aware splits.
Plus provenance metadata (which checkpoint, its SHA-256, task, obs/action layout)
so the data is self-describing. See ``recorded_data_io.py`` for the on-disk
format and the loader used by the learning script.

Specify the total number of pairs with ``--num_pairs``; every run writes a new,
non-overwriting file under ``<log_dir>/recorded_data/`` (override with
``--output_dir``).

Episodes start at clip frame 0 by default (``--start_step``), which is a RECORDING
choice, not the training one: training draws a random start so every phase is an entry
point, but a random start run to the clip end visits step ``t`` with probability
``~t/clip_len``, so the recorded data would be a ramp toward late clip phases rather
than the uniform coverage the downstream FM should train on.

Example:
    python scripts/rsl_rl/record_data.py --task <Task> --num_envs 64 \
        --num_pairs 100000 --load_run <run> --headless
"""

import argparse
import contextlib
import importlib.metadata as metadata
import math
import json
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402  --ref_future_len default
from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: E402  height_scan obs fn
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from export_parameters import export_policy_parameters, robot_asset_fingerprint_for  # isort: skip
from recorded_data_io import (  # isort: skip
    RecordingMeta,
    RecordedWindowWriter,
    fingerprint_arrays,
    sha256_file,
    unique_output_path,
    save_recorded_data,
)
from provenance import auto_clip_device, git_snapshot  # isort: skip
from perturbation_scheduler import PerturbationScheduler  # isort: skip
from isaaclab.utils.io import dump_yaml  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Play a trained RSL-RL policy and record (obs, action) pairs for supervised learning."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument(
    "--num_pairs", type=int, default=None,
    help="LEGACY per-step .npz mode: total (observation, action) pairs to record "
         "(buffers all steps in RAM). Pass exactly one of --num_pairs or --num_windows.",
)
parser.add_argument(
    "--num_windows", type=int, default=None,
    help="WINDOWED mode: total FM snap windows to stream to a chunked .h5 (bounded RAM; "
         "only the snaps the FM actually trains on are stored). Pass exactly one of "
         "--num_pairs or --num_windows.",
)
parser.add_argument(
    "--history_len", type=int, default=8,
    help="[--num_windows] History-state rows H stored per window. Must be >= the largest "
         "num_history_states you will train the FM with (this is the one irreversible "
         "record-time choice). Default 8.",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Where to write the dataset. Default: <log_dir>/recorded_data.",
)
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions stochastically. Default is deterministic (mean) actions.",
)
parser.add_argument(
    "--ref_command_name", type=str, default=None,
    help="Name of the trajectory command term to record the reference state from. "
         "Default: auto-detect the command term exposing get_reference_world_state.",
)
parser.add_argument(
    "--ref_base_frame", type=str, default="pelvis_link",
    help="Trajectory output frame used as the robot root when reconstructing the "
         "reference state (must expose pos_x/y/z and ori_x/y/z/w outputs).",
)
parser.add_argument(
    "--state_frames", type=str, nargs="*", default=None,
    help="Extra robot body frames whose world pose/velocity are appended to the "
         "recorded 'state' (and 'ref_state') vectors beyond root+joints -- e.g. the "
         "feet/hands the flow-matching CLF tracks. Default: auto-detect the trajectory "
         "command's end-effector output frames (ankles/wrists). Pass nothing to record "
         "root+joints only.",
)
parser.add_argument(
    "--skip_height_scan", action="store_true", default=False,
    help="Do not record the 'height_scan' array. The recorder normally stores BOTH "
         "exteroception signals so one recording can train either FM (cfg.extero_source picks "
         "at train time); pass this when the depth path is the only one you intend to use. "
         "Note the recorded scan is CLEAN either way -- its noise lives on the obs term and the "
         "recorder calls mdp.height_scan directly -- whereas depth carries its DR inside the "
         "obs function, so the two are not symmetric.",
)
parser.add_argument(
    "--ref_future_len", type=int, default=FlowMatchingConfig().future_len,
    help="Number of FUTURE nodes F to record per step in 'ref_future' (the ego-anchored "
         "clip future = the robot-state-conditioned FM target). Defaults to (and MUST "
         "match) FlowMatchingConfig.future_len -- the dataset hard-errors on a mismatch, "
         "since windows are cut here and cannot be re-cut at train time.",
)
parser.add_argument(
    "--no_ref_future", action="store_true", default=False,
    help="Disable recording the ego-anchored clip future ('ref_future'/'cmd_future'/"
         "'reanchor_active'). By default it is recorded whenever the command exposes the "
         "look-ahead accessors (get_future_world_base) -- the data the new FM trains on.",
)
parser.add_argument(
    "--clip", type=str, default=None,
    help="Override the clip library the teacher tracks (sets commands.traj_ref.path), e.g. "
         "clips/g1_recovery_v1.h5. Default: the task config's clip (clips/g1_flat_v1.h5 for the "
         "teacher tasks). The recorded distribution mirrors THIS clip, so point it at the library "
         "you generated.",
)
parser.add_argument(
    "--start_step", type=int, default=0,
    help="Clip step every episode starts at. 0 (the default) is what makes the recorded "
         "window distribution UNIFORM in clip phase: each episode walks its clip once from "
         "frame 0, so every frame is visited exactly once per visit to that clip. -1 "
         "restores training's SAMPLED start (commands.traj_ref.random_start_phase), which is "
         "not uniform -- a start drawn uniformly from [0, clip_len) and then run to the clip "
         "end visits step t with probability ~t/clip_len, so late phases dominate (measured "
         "on a 200k-window recording: 1.1%% of windows in the first phase decile against "
         "18.1%% in the eighth). The cost of 0 is clip COVERAGE: episodes are ~2x longer, so "
         "a fixed --num_windows budget spans ~half as many distinct clips.",
)
parser.add_argument(
    "--clip_draw", type=str, default="cycle", choices=("cycle", "random"),
    help="How each reset picks its clip. 'cycle' (the default) draws from a shuffled queue "
         "per terrain group, so every clip in the library is used once before any repeats -- "
         "clip coverage is then exactly min(episodes, clips). 'random' is training's "
         "with-replacement sampling, which covers only 1-exp(-E/C) of a C-clip group in E "
         "episodes (63%% when E == C), and the misses are a random third of the library. "
         "Cycling costs nothing and is strictly better for a recording, whose clip coverage "
         "IS the deliverable; 'random' is kept to reproduce an older recording.",
)
parser.add_argument(
    "--fm_depth_resolution", type=str, default="",
    help="'WxH' to record the GENERATOR's depth at, when it should differ from the policy's. "
         "Adds a second camera pair on the SAME mounts and lenses, sampled on this grid, and "
         "the recorded 'depth' array plus its camera contract then describe THAT pair -- so a "
         "generator trained on this file conditions on this resolution while the teacher that "
         "produced the file keeps its own. Empty (default) records the scene's own cameras at "
         "the resolution they carry; it does NOT mean 'record whatever the policy observes', "
         "which is the near camera alone. Either way the recording covers BOTH cameras. "
         "Costs rays: the pass is ~W*H/780 times the policy pair's per-step depth cost, and "
         "the 'depth' dataset scales the same way (a 200k-window recording is 1.25 GB at "
         "30x26 and 5 GB at 60x52), so pick this before recording rather than after.",
)
parser.add_argument(
    "--fm_height_scan", type=str, default="",
    help="'SXxSY@RES' to give the GENERATOR its own dense height scanner, e.g. '3.0x3.0@0.025'. "
         "Adds a SECOND scanner on the same mount and alignment as the policy's, robot-centred, "
         "sampled on this grid, and the recorded 'height_scan' array plus its geometry contract "
         "then describe THAT scanner. The policy's 176-ray scanner is untouched -- it has to be, "
         "since the depth teacher's critic observes it and a changed width makes the checkpoint "
         "unloadable. PRIVILEGED: an exact noise-free ground-truth map that no deployment can "
         "reproduce, so the ONNX/MuJoCo/hardware paths refuse a recording made with it. Costs "
         "rays and disk: 3.0x3.0@0.025 is 14641 rays (83x the policy's) and 57 KiB/window f32, "
         "so pair it with --fm_scan_float16.",
)
parser.add_argument(
    "--fm_scan_float16", action="store_true", default=False,
    help="Store the 'height_scan' array as float16. Halves the file and the training-time "
         "dataset RAM; 1 mm quantisation is 25x finer than a 0.025 m grid cell, and lzf "
         "compresses f16 sensor data far better than f32 (whose low mantissa bits are noise). "
         "Recommended whenever --fm_height_scan is used at a dense grid.",
)
parser.add_argument(
    "--clip_device", type=str, default=None,
    help="Device to hold the motion-clip library on (e.g. 'cpu' to keep the multi-GB "
         "library off a small GPU; gathers copy only the per-env slice). Default: env device.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
# -- Perturbation / recovery-data options (see perturbation_scheduler.py) --
parser.add_argument(
    "--perturb", action="store_true", default=False,
    help="Gather recovery data: run a clean fraction of envs (play DR) and a perturbed "
         "fraction with windowed action-noise + push toggled on/off. Logs a per-step "
         "'perturb_active' flag. Record on a PLAY task (e.g. G1-teacher-clf-terrain-play) so the "
         "clean group is play-DR and perturbations are layered per-env in the loop.",
)
parser.add_argument("--clean-fraction", type=float, default=0.5,
                    help="[--perturb] Fraction of envs that stay clean (never perturbed).")
parser.add_argument("--perturb-on-range-s", type=float, nargs=2, default=(0.3, 1.0),
                    metavar=("MIN", "MAX"), help="[--perturb] Seconds a perturbation window stays ON.")
parser.add_argument("--perturb-off-range-s", type=float, nargs=2, default=(1.5, 4.0),
                    metavar=("MIN", "MAX"), help="[--perturb] Seconds between windows (recovery).")
parser.add_argument("--action-noise-scale", type=float, default=0.3,
                    help="[--perturb] Std of Gaussian noise added to actions while ON (0 disables).")
parser.add_argument("--push-vel", type=float, default=0.75,
                    help="[--perturb] Magnitude (m/s) of the planar x/y push impulse on each "
                         "OFF->ON edge (0 disables pushes).")
parser.add_argument("--obs-noise-scale", type=float, default=0.0,
                    help="[--perturb] Std of Gaussian noise added to the policy obs while ON "
                         "(0 disables; the recorded state is always ground truth).")
parser.add_argument("--ood-dr", action="store_true", default=False,
                    help="[--perturb] Re-apply the perturbed group's DR terms with widened "
                         "(out-of-distribution) ranges at each reset.")
parser.add_argument("--ood-dr-scale", type=float, default=1.5,
                    help="[--perturb] Range-widening factor for --ood-dr (1.5 = 50%% wider).")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_resume_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the checkpoint path the same way play.py does, without the simulator.

    Args:
        agent_cfg: The RSL-RL agent configuration (already CLI-updated).

    Returns:
        Absolute path to the checkpoint file to load.
    """
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    if getattr(args_cli, "teacher_experiment", None):
        play_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment))
    else:
        play_log_path = log_root_path
    return get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _resolve_ref_command(env: RslRlVecEnvWrapper, explicit_name: str | None):
    """Find the trajectory command term used to reconstruct the reference state.

    Args:
        env: The wrapped Isaac Lab environment.
        explicit_name: Command term name from the CLI, or ``None`` to auto-detect
            the first term exposing ``get_reference_world_state``.

    Returns:
        ``(name, term)`` for the resolved trajectory command, or ``(None, None)``
        if the task has no such command (e.g. velocity-only tasks).

    Raises:
        ValueError: If ``explicit_name`` is given but the term cannot provide a
            reference state.
    """
    cmd_mgr = env.unwrapped.command_manager
    if explicit_name is not None:
        term = cmd_mgr.get_term(explicit_name)
        if not hasattr(term, "get_reference_world_state"):
            raise ValueError(
                f"Command term '{explicit_name}' does not expose get_reference_world_state."
            )
        return explicit_name, term
    for name in cmd_mgr.active_terms:
        term = cmd_mgr.get_term(name)
        if hasattr(term, "get_reference_world_state"):
            return name, term
    return None, None


def _audit_anchor_masks(ref_cmd) -> None:
    """Warn LOUDLY if any clip in the library frees no root DOF.

    FM training windows are cut at re-anchor (snap) events, and recording runs with
    ``ego_anchor_fraction=1.0`` so the snap fires for EVERY env. A clip whose anchor mask frees
    nothing still snaps, but its placement cannot move onto the robot -- so the dataset's
    ego-anchored target encodes accumulated robot-vs-clip TRACKING DRIFT instead of a placement
    decision. Nothing downstream can distinguish those windows from good ones, which is why this
    is a shouted warning at record time rather than a quiet metric.

    Checks the per-STEP schedule, so a clip that is pinned for only part of its length (the box
    family's mount phase, legitimately) is NOT flagged -- only clips pinned at every step.

    Args:
        ref_cmd: The resolved trajectory command term (``None`` -> nothing to check).
    """
    mgr = getattr(ref_cmd, "manager", None)
    amt = getattr(mgr, "anchor_mask_t", None)
    if amt is None or amt.numel() == 0:
        return
    pinned = (amt == 0).all(dim=1)                       # [num_clips] bool: no free DOF, ever
    n_pinned = int(pinned.sum())
    if n_pinned == 0:
        return
    n_total = int(pinned.numel())
    ids = pinned.nonzero(as_tuple=False).squeeze(-1)[:10].tolist()
    bar = "!" * 100
    print(f"\n{bar}")
    print(f"[WARN] {n_pinned}/{n_total} clips in this library free NO root DOF at ANY step "
          f"(anchor_mask == 0).")
    print(f"[WARN] first ids: {ids}{' ...' if n_pinned > len(ids) else ''}")
    print("[WARN] These clips are NEVER re-anchored, but the FM snap still fires for them, so "
          "every window")
    print("[WARN] they contribute has a target made of robot-vs-clip TRACKING DRIFT rather than a "
          "placement")
    print("[WARN] decision -- and the dataset cannot tell those windows from good ones.")
    print("[WARN] Regenerate the offending family with a mask that frees at least one root DOF "
          "(e.g.")
    print("[WARN] _ANCHOR_MASK_FLAT), or exclude it from this library before recording.")
    print(f"{bar}\n")


def _distractor_clip_mask(ref_cmd):
    """Per-clip bool tensor: is this clip from the distractor family?

    Read from the library's ``terrain_manifest`` ``family`` field (the composer stamps it per
    tile), NOT from the anchor mask -- distractors are ego-anchored like flat clips now, so the
    mask no longer distinguishes them. Recorded per step so
    ``scripts/flow_matching/train.py --skip_distractors`` can drop those windows without a
    re-record.

    Args:
        ref_cmd: The resolved trajectory command term (``None`` -> no clips).

    Returns:
        ``[num_clips]`` bool tensor, or ``None`` if the library carries no manifest / terrain ids
        (a pre-terrain or flat-only library has no distractors by construction).
    """
    mgr = getattr(ref_cmd, "manager", None)
    lib = getattr(mgr, "lib", None)
    if lib is None or getattr(lib, "terrain_id", None) is None or lib.terrain_id.numel() == 0:
        return None
    manifest = lib.attrs.get("terrain_manifest") if hasattr(lib, "attrs") else None
    if not manifest:
        return None
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    distractor_tids = {
        int(e["terrain_id"]) for e in manifest
        if "terrain_id" in e and "distractor" in str(e.get("family", "")).lower()
    }
    if not distractor_tids:
        return None
    tid = lib.terrain_id.to(torch.long)
    flags = torch.zeros(tid.shape[0], dtype=torch.bool, device=tid.device)
    for t in distractor_tids:
        flags |= tid == t
    print(f"[INFO] distractor family: {int(flags.sum())}/{int(flags.numel())} clips over "
          f"{len(distractor_tids)} tiles -> recording per-step 'is_distractor' so "
          f"train.py --skip_distractors can drop them later.")
    return flags


def _report_clip_coverage(ref_cmd) -> None:
    """Print how much of the clip library the finished recording actually spanned.

    Only meaningful under ``--clip_draw cycle``, which is the mode that makes coverage a
    guarantee rather than a lottery; with-replacement sampling leaves no queue to read and
    this quietly does nothing.  Per TERRAIN GROUP rather than one number, because groups
    advance through their queues at different rates: an env consumes one clip per episode
    and a group's clips may be 3x longer than another's, so the flat group is still on its
    first pass while the stair groups are on their third.

    Args:
        ref_cmd: The resolved trajectory command term (holds the cycler, if any).
    """
    cycler = getattr(ref_cmd, "_clip_cycler", None)
    if cycler is None:
        return
    cov = cycler.coverage()
    total = sum(g["clips"] for g in cov.values())
    seen = sum(g["distinct_used"] for g in cov.values())
    full = sum(1 for g in cov.values() if g["passes"] > 0)
    print(f"[INFO] clip coverage: {seen}/{total} distinct clips used "
          f"({100.0 * seen / max(total, 1):.1f}%); {full}/{len(cov)} terrain groups completed "
          f"at least one full pass.")
    partial = {g: v for g, v in cov.items() if v["passes"] == 0 and v["distinct_used"] < v["clips"]}
    if partial:
        worst = sorted(partial.items(), key=lambda kv: kv[1]["distinct_used"] / kv[1]["clips"])[:5]
        detail = ", ".join(f"tid {g}: {v['distinct_used']}/{v['clips']}" for g, v in worst)
        print(f"[INFO]   least-covered groups -- {detail}. Record more windows to span them.")


def _clip_family_codes(ref_cmd) -> tuple["torch.Tensor | None", list[str]]:
    """Per-clip family index plus the family names, from the library's ``terrain_manifest``.

    The general form of :func:`_distractor_clip_mask`. That one answers a single yes/no
    question and so could only ever support ``--skip_distractors``; every other family
    question -- and there are eight families in the v5 library -- needed a re-record to ask.
    Recording the family as a CODE makes any of them a train-time switch
    (``train.py --skip_families``), which is the same reasoning that put the distractor flag
    in the recording rather than in the library.

    The concrete motivation: ``family_jump_off`` is the BACKWARDS jump-off-a-box family, and
    with forward-facing depth cameras the drop behind the robot is not observable at all --
    so those windows ask the generator to predict something it cannot see.

    Clips on a tile with no manifest entry (and every clip of a flat-only library) get code
    0 = ``"unknown"``, which no filter matches.

    The mapping itself lives in :func:`mdp.goal_region.clip_family_codes`, because the goal
    termination resolves its per-family tolerances from the same manifest and a family read
    two ways is a family that can be read two ways WRONG. This wrapper adds the reporting.

    Args:
        ref_cmd: The resolved trajectory command term (``None`` -> no clips).

    Returns:
        ``([num_clips] long tensor of family codes, [family names])``. ``(None, [])`` when
        the library carries no manifest or no terrain ids.
    """
    # Imported here, not at module scope: `mdp` pulls in `isaaclab.envs.mdp`, and this module
    # is imported before AppLauncher runs -- see `test_no_pre_app_pxr_import.py`.
    from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region

    codes, names = goal_region.clip_family_codes(ref_cmd)
    if codes is None:
        return None, []
    counts = {names[c]: int((codes == c).sum()) for c in sorted(set(codes.tolist()))}
    print(f"[INFO] clip families -> per-window 'clip_family' so train.py --skip_families can "
          f"drop any of them later: {counts}")
    return codes, names


def _extero_is_corrupted(env) -> bool:
    """Whether a depth noiser is installed, i.e. the recorded depth carries the sim's sensor DR.

    The ``-play`` tasks strip ``resample_depth_noise`` (and the camera extrinsics/intrinsics DR),
    so a play recording captures CLEAN depth at the nominal pose. Recording on the train task
    keeps them. Checked through the noiser registry because that is the thing ``depth_image``
    consults -- an event that is configured but never ran would not corrupt anything.

    Args:
        env: The unwrapped environment.

    Returns:
        True when at least one depth observation term has a noiser installed.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import get_depth_noiser
    for sensor_name, _ in mdp.find_depth_obs_terms(env, mdp.fm_depth_groups(env)):
        if get_depth_noiser(env, sensor_name) is not None:
            return True
    return False


def _assert_generator_sees_every_camera(env, env_cfg) -> None:
    """Refuse to record depth from fewer cameras than the robot carries.

    THE FAILURE IS SILENT AND UNRECOVERABLE DOWNSTREAM. The policy reads the near camera
    alone, so ``mdp.fm_depth_groups`` returning ``None`` ("every group") now means the
    recorder captures ONE camera, writes a one-camera ``depth_camera_spec`` beside it,
    trains a generator on that, exports an ``fm.onnx`` carrying the same one-camera
    contract, and passes the deploy-time agreement check -- every artifact self-consistent
    about a half-blind generator, and indistinguishable from a deliberate one-camera
    design. No shape check anywhere sees a problem.

    Two statements, because they fail differently: the group must be ACTIVE in the built
    env (a cfg that was never installed, or a group the observation manager dropped), and
    it must NAME every camera on the scene (a stale group, checked by the producer's own
    guard so the ``fm_``-prefix rule lives in one place).

    Args:
        env: The unwrapped environment.
        env_cfg: The env cfg it was built from.

    Raises:
        ValueError: If the task reads depth but the generator's group is absent or partial.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        assert_fm_depth_group_covers_every_camera,
    )
    if not mdp.find_depth_obs_terms(env, None):
        return                              # no depth on this task (the height-scan path)
    if mdp.fm_depth_groups(env) is None:
        raise ValueError(
            f"this task observes depth but has no active '{mdp.FM_DEPTH_OBS_GROUP}' "
            "observation group, so the recording would be scoped to the POLICY's cameras "
            f"-- {[n for n, _ in mdp.find_depth_obs_terms(env, None)]}, the near camera "
            "alone. Install the generator's group before gym.make."
        )
    assert_fm_depth_group_covers_every_camera(env_cfg)
    print(f"[INFO] recording depth from "
          f"{[n for n, _ in mdp.find_depth_obs_terms(env, mdp.fm_depth_groups(env))]} "
          f"(obs group '{mdp.FM_DEPTH_OBS_GROUP}') -- every camera the scene carries.")


def _build_recording_meta(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
    log_dir: str,
    num_pairs: int,
    state_frames: list[str],
    windowed: bool,
) -> RecordingMeta:
    """Capture provenance and obs/action layout metadata once at the start.

    Args:
        env: The wrapped Isaac Lab environment.
        agent_cfg: The RSL-RL agent configuration.
        resume_path: Absolute path to the loaded checkpoint.
        log_dir: Directory the checkpoint lives in.
        num_pairs: Final number of pairs that will be saved.
        state_frames: Extra body frames appended to the state vector (in order).
        windowed: Whether this is a windowed (.h5) recording. Only that path substitutes the
            plan anchor into the history rows, so it decides ``history_anchor``.

    Returns:
        A populated :class:`RecordingMeta`.
    """
    obs_mgr = env.unwrapped.observation_manager
    obs_term_names = list(obs_mgr.active_terms["policy"])
    obs_term_dims = [int(td[0]) for td in obs_mgr.group_obs_term_dim["policy"]]

    robot = env.unwrapped.scene.articulations["robot"]
    action_joint_names = list(robot.data.joint_names)

    # ``state_frames`` are the extra ee frames; the PARC-style FM conditions on AND
    # predicts the same set (no history/target asymmetry), so ``state_frame_names``
    # below is the single frame set. Training persists it as ``target_frame_names`` in
    # the checkpoint meta; online.load_fm_context falls back to ``state_frame_names``.
    # Raw robot-state layout recorded by collect_pairs (world frame, absolute).
    # Order and dims MUST match the torch.cat in collect_pairs.
    num_joints = len(action_joint_names)
    state_term_names = [
        "root_pos_w", "root_quat_w", "root_link_lin_vel_w", "root_link_ang_vel_w", "joint_pos", "joint_vel",
    ]
    state_term_dims = [3, 4, 3, 3, num_joints, num_joints]
    # Per extra frame: world pose + twist, appended after root + joints.
    for f in state_frames:
        state_term_names += [f"{f}:pos_w", f"{f}:quat_w", f"{f}:lin_vel_w", f"{f}:ang_vel_w"]
        state_term_dims += [3, 4, 3, 3]

    ref_blocks = mdp.reference_window_selection(env.unwrapped)

    return RecordingMeta(
        checkpoint_path=os.path.abspath(resume_path),
        checkpoint_sha256=sha256_file(resume_path),
        task_name=args_cli.task,
        experiment_name=agent_cfg.experiment_name,
        run_name=os.path.basename(log_dir),
        runner_class=agent_cfg.class_name,
        rsl_rl_version=installed_version,
        recorded_at=datetime.now().isoformat(timespec="seconds"),
        stochastic=bool(args_cli.stochastic),
        dt=float(env.unwrapped.step_dt),
        num_envs=int(env.num_envs),
        num_pairs=int(num_pairs),
        obs_term_names=obs_term_names,
        obs_term_dims=obs_term_dims,
        action_joint_names=action_joint_names,
        state_term_names=state_term_names,
        state_term_dims=state_term_dims,
        state_quat_format="xyzw",
        state_joint_names=action_joint_names,
        state_vel_frame="link",
        state_frame_names=list(state_frames),
        # Contract of the cameras behind the recorded ``depth`` array (None without them).
        # The same producer the tracker export uses, so an FM trained on this recording
        # and the tracker deployed beside it describe one physical camera pair.
        # Scoped to the cameras the recorded array actually came from. Unscoped, a task
        # carrying both pairs would describe FOUR cameras behind a two-row array, and the
        # FM dataset's len(spec) == rows check is the only thing that would notice.
        depth_camera_spec=mdp.depth_camera_specs(
            env.unwrapped, mdp.fm_depth_groups(env.unwrapped)) or None,
        # WHICH ROBOT produced these rows. CLAUDE.md's "No re-record" property for the depth
        # xyz channels holds only while the camera AND the FK model are fixed: the offline
        # unprojection rebuilds the camera pose from the recorded joint angles through
        # motion_gen's FKChain, so a recording replayed against a changed model yields
        # silently mis-registered xyz -- visible, if at all, only as a shifted
        # declared-vs-measured centroid at dataset build. Same producer as the tracker
        # export, so a recording and the policy that made it can be checked against each other.
        robot_asset=robot_asset_fingerprint_for(env.unwrapped),
        # Which reference-window blocks THIS teacher observes. The recorded arrays are
        # world-frame rows, so any selection can be packed from them -- but the generator
        # trained on this file has to predict what the policy consumes, so the teacher's
        # selection is the default the FM dataset resolves. Same producer the tracker
        # export reads, so the two artifacts cannot describe different windows.
        reference_blocks=(list(ref_blocks) if ref_blocks else None),
        # Windowed recordings substitute the plan anchor into the history rows (see
        # collect_windows); the per-step .npz path does not, and its rows stay measured.
        history_anchor="plan" if windowed else "per_step",
        # Probes the LIVE noiser rather than the event config: depth_image applies noise only
        # when resample_depth_noise installed one, so this is what actually reached the data.
        extero_corrupted=_extero_is_corrupted(env.unwrapped),
    )


def _assemble_state(robot, state_frames, frame_body_idx, n_envs) -> torch.Tensor:
    """Raw world-frame per-step state row ``[N, node_dim]`` (layout == state_term_names).

    Shared by :func:`collect_pairs` and :func:`collect_windows` so the recorded node
    layout stays in one place: root pos(3)/quat(4, xyzw)/lin(3)/ang(3), joints, then
    per extra frame pos(3)/quat(4)/lin(3)/ang(3).
    """
    parts = [
        wp.to_torch(robot.data.root_pos_w),           # [N, 3]
        wp.to_torch(robot.data.root_quat_w),          # [N, 4] (xyzw)
        wp.to_torch(robot.data.root_link_lin_vel_w),  # [N, 3]
        wp.to_torch(robot.data.root_link_ang_vel_w),  # [N, 3]
        wp.to_torch(robot.data.joint_pos),            # [N, nj]
        wp.to_torch(robot.data.joint_vel),            # [N, nj]
    ]
    if state_frames:
        fpos = wp.to_torch(robot.data.body_pos_w)[:, frame_body_idx]        # [N, K, 3]
        fquat = wp.to_torch(robot.data.body_quat_w)[:, frame_body_idx]      # [N, K, 4]
        flink = wp.to_torch(robot.data.body_link_vel_w)[:, frame_body_idx]  # [N, K, 6]
        parts.append(
            torch.cat([fpos, fquat, flink[..., :3], flink[..., 3:]], dim=-1).reshape(n_envs, -1)
        )
    return torch.cat(parts, dim=-1)


# TODO(latent-fm): also record the reference ENCODER LATENT per window, so a latent-space
# FM can be trained without re-deriving it. The teacher is a RefEncoderMLPModel: the
# `reference` group (reference_window 13 samples x D + command_window 13 x 3) goes through
# [1024, 512, 512] and the trunk only ever sees the resulting 512-d latent. That latent is
# ~half the width of the window it replaces, so recording it is cheap (~2 KB/window).
#
# It IS derivable offline from ref_future + history_state + cmd_future, but only by
# replicating the observation manager's noise -> clip -> scale ordering exactly and pinning
# the teacher checkpoint -- the kind of re-derivation that drifts silently, which is why
# recording it is worth the plumbing.
#
# Record the CLEAN latent (an FM target should not be noise-corrupted) and ideally the
# corrupted one too; the pair measures how much the encoder actually cares about the noise.
#
# Two consequences to accept before using it, not after:
#   * it welds the FM to ONE teacher checkpoint -- today's FM predicts a semantically
#     defined vector, so any reference_window-consuming teacher can use it;
#   * a latent is opaque, so it cannot be re-anchored onto the current robot pose or
#     indexed by a cursor -> fm_update_every is pinned to 1 forever.


def _assemble_ref_state(ref_cmd, ref_base_frame: str, state_frames, n_envs) -> torch.Tensor:
    """The APPLIED reference in the ``_assemble_state`` layout ``[N, node]``.

    The world-frame desired root + joints, plus the same extra body frames the measured
    state carries, so a reference row is interchangeable with a measured one column-for-
    column. That is what lets the FM dataset build a MIXED history (measured root, reference
    joints/ee) by slicing both with one term table -- see
    ``FlowMatchingConfig.history_source``.

    Timing: ``y_des``/``dy_des`` hold the value written during the previous ``env.step``, so
    reading this pre-step yields the reference applied at step ``t-1`` -- aligned with the
    pre-step ``state``/``obs`` snapshot, and matching what the runtime snapshot sees.
    """
    ref_state = ref_cmd.get_reference_world_state(ref_base_frame)
    if state_frames:
        ref_frames = ref_cmd.reference_world_frames(state_frames)  # [N, K, 13]
        ref_state = torch.cat([ref_state, ref_frames.reshape(n_envs, -1)], dim=-1)
    return ref_state


#: How many control steps of camera latency the recorder keeps frames for.
#:
#: The depth pipeline is not instantaneous on hardware: the image has to be computed and
#: shipped over ROS, which Zach measures at 20-40 ms typically and up to 60 ms. The camera runs
#: at ``_DEPTH_FPS = 50`` and the control step is ``decimation 4 * sim.dt 0.005`` = 50 Hz, so
#: ONE lag step is exactly 20 ms and lags 1..3 cover that range with nothing left over.
#:
#: The recorder stores all of lags 0..3 and lets TRAINING pick: a fresh draw every epoch, a
#: distribution that can change without re-recording, and a free no-latency ablation on the
#: same data. Lag 0 is kept for that ablation only -- it is physically unreachable on the
#: robot, since a frame cannot arrive in the step it was captured in, so the shipped
#: distribution gives it zero weight (see ``FlowMatchingConfig.depth_latency_probs``).
#:
#: Costs 4x the depth array: it is recorded as ONE channel (``[N, T, H*W]``, 2 x 780 = 1560
#: floats/window; the xyz and validity channels are derived at dataset build), so a
#: 200k-window recording goes from about 1.25 GB to 5 GB.
DEPTH_LAG_STEPS: int = 3


def _future_fits_fn(ref_cmd, max_offset: int):
    """Per-env predicate: is the whole F-node future window inside the clip's REAL frames?

    Short clips are padded into the library by repeating their last frame, and the
    look-ahead gather clamps to the library length, so a window opened within F steps of
    ``clip_len - 1`` has a partly FROZEN target. Training on those teaches "the reference
    stops moving" for conditioning that otherwise means "keep going", and because the FM is
    multi-modal it would then sample that frozen mode occasionally at deploy. Cheaper to
    drop those windows than to un-learn the mode -- and it is the record-time half of the
    no-clamping rule the tracker enforces at inference.

    ``clip_len`` lives on the clip-library device (CPU under ``--clip_device cpu``), so
    gather there and move the verdict back.

    Args:
        ref_cmd: The trajectory command (needs ``manager.clip_len``, ``_step``, ``clip_id``).
        max_offset: Largest future offset the recording will gather.

    Returns:
        A zero-arg callable returning a ``[N]`` bool tensor, or ``None`` when the command
        exposes no clip step / length (e.g. an FM-generated reference) and no gate is possible.
    """
    clip_len_all = getattr(getattr(ref_cmd, "manager", None), "clip_len", None)
    if clip_len_all is None or not (hasattr(ref_cmd, "_step") and hasattr(ref_cmd, "clip_id")):
        print("[WARN] cannot gate on clip length (no clip step/len on the command); "
              "windows near a clip end may carry a partly frozen future.")
        return None

    def fut_ok_fn():
        """True per env when ``step + max_offset`` is still a real (unpadded) clip frame."""
        cid = ref_cmd.clip_id.to(clip_len_all.device)
        cl = clip_len_all[cid].to(ref_cmd._step.device)
        return (ref_cmd._step + max_offset) <= (cl - 1)

    return fut_ok_fn


def _assemble_ref_future_node(ref_cmd, future_offsets, state_frames, n_envs) -> torch.Tensor:
    """Ego-anchored clip future ``[N, F, node_dim]`` (node layout == state).

    Shared by :func:`collect_pairs` and :func:`collect_windows`.
    """
    pos_w, quat_w, lin_w, ang_w = ref_cmd.get_future_world_base(future_offsets)  # [N,F,·]
    jp, jv = ref_cmd.get_future_joint_state(future_offsets)                       # [N,F,nj]
    node = torch.cat([pos_w, quat_w, lin_w, ang_w, jp, jv], dim=-1)              # [N,F,13+2nj]
    if state_frames:
        ff = ref_cmd.get_future_world_frames(state_frames, future_offsets)       # [N,F,K,13]
        node = torch.cat([node, ff.reshape(n_envs, ff.shape[1], -1)], dim=-1)
    return node


def collect_pairs(
    env: RslRlVecEnvWrapper,
    policy,
    policy_nn,
    initial_obs,
    num_steps: int,
    num_pairs: int,
    stochastic: bool,
    ref_cmd=None,
    ref_base_frame: str = "pelvis_link",
    state_frames: list[str] | None = None,
    scheduler: PerturbationScheduler | None = None,
    ref_future_len: int = 0,
) -> dict[str, np.ndarray]:
    """Roll out the policy and collect flattened (obs, action) pairs.

    The action recorded for an observation is ``policy(obs)`` computed *before*
    stepping the environment, so each pair is correctly aligned. The
    ``terminated``/``done``/``time_out`` flags are recorded *after* the step and
    describe the transition out of that pair: ``terminated`` excludes time-outs,
    so it marks the last pair before a true (failure) termination.

    Args:
        env: The wrapped Isaac Lab environment.
        policy: Inference policy callable returning actions from observations.
        policy_nn: The policy network for ``reset`` on rsl-rl < 4.0.0; ``None``
            when the inference policy itself exposes ``reset`` (rsl-rl >= 4.0.0).
        initial_obs: The starting observation TensorDict (from ``get_observations``).
        num_steps: Number of rollout steps (``ceil(num_pairs / num_envs)``).
        num_pairs: Total pairs to keep; arrays are trimmed to this length.
        stochastic: Whether to sample actions stochastically.
        ref_cmd: Trajectory command term exposing ``get_reference_world_state``,
            or ``None`` to skip recording the reference state.
        ref_base_frame: Trajectory output frame used as the robot root for the
            reference reconstruction.

    Returns:
        A dict of arrays each with leading dimension ``num_pairs``:
        ``obs``, ``action``, ``state``, ``terminated``, ``done``, ``time_out``,
        ``episode_id``, ``step_in_episode`` and, when ``ref_cmd`` is given,
        ``ref_state`` (the desired reference in the same layout as ``state``).
    """
    n_envs = env.num_envs
    device = env.unwrapped.device
    use_policy_reset = version.parse(installed_version) >= version.parse("4.0.0")
    robot = env.unwrapped.scene.articulations["robot"]
    # Height scan (the FM's height-conditioning token), captured when the recording task
    # has a height scanner active (e.g. G1-teacher-clf-terrain-play). Resolved through
    # mdp.fm_scan_sensor, which prefers the GENERATOR's dense scanner when --fm_height_scan
    # added one and otherwise returns the policy's -- i.e. exactly the sensor this line used to
    # name, so a run that does not opt in is unchanged.
    scan_sensor = None if args_cli.skip_height_scan else mdp.fm_scan_sensor(env.unwrapped)
    has_scan = scan_sensor is not None
    # Depth cameras (the deployable FM exteroception signal). Recorded ALONGSIDE the scan
    # whenever the task has depth observation terms, so ONE recording can train either FM
    # and the choice is made at FM-train time via cfg.extero_source.
    has_depth = bool(mdp.find_depth_obs_terms(env.unwrapped,
                                              mdp.fm_depth_groups(env.unwrapped)))

    # Body indices for the extra frames appended to the state vector (in order).
    state_frames = list(state_frames or [])
    if state_frames:
        body_names = list(robot.data.body_names)
        frame_body_idx = torch.tensor(
            [body_names.index(f) for f in state_frames], dtype=torch.long, device=device
        )

    obs = initial_obs
    # Distinct ids per env so step-0 pairs from different envs aren't conflated.
    episode_id = torch.arange(n_envs, dtype=torch.int64, device=device)
    step_in_ep = torch.zeros(n_envs, dtype=torch.int64, device=device)
    next_episode_id = int(n_envs)

    buf: dict[str, list[torch.Tensor]] = {
        "obs": [], "action": [], "state": [],
        "terminated": [], "done": [], "time_out": [],
        "episode_id": [], "step_in_episode": [],
    }
    if ref_cmd is not None:
        buf["ref_state"] = []
    # Per-step commanded velocity (the upstream command that drove the reference),
    # recorded when the clip task exposes it -- enables future-horizon command
    # conditioning in the flow-matching dataset.
    record_cmd_vel = ref_cmd is not None and getattr(ref_cmd, "command_velocity_available", False)
    if record_cmd_vel:
        buf["target_vel"] = []
    # Ego-anchored clip FUTURE target (robot-state-conditioned FM): per step an F-node
    # window of the clip ego-anchored onto the robot, plus the aligned future command and
    # the snap-event flag. Requires the look-ahead accessors + a command velocity.
    record_ref_future = (
        ref_future_len > 0 and ref_cmd is not None
        and hasattr(ref_cmd, "get_future_world_base") and record_cmd_vel
    )
    if record_ref_future:
        future_offsets = torch.arange(int(ref_future_len), device=device)
        buf["ref_future"] = []
        buf["cmd_future"] = []
        buf["reanchor_active"] = []
        # Same frozen-future gate the windowed path applies (see _future_fits_fn). Here it
        # is folded into the recorded snap flag, because ``snap_windows`` cuts training
        # windows from exactly that flag -- clearing it drops the window at train time.
        fut_ok_fn = _future_fits_fn(ref_cmd, int(future_offsets.max().item()))
        n_short = 0
    if has_scan:
        buf["height_scan"] = []
    if has_depth:
        buf["depth"] = []
    # Per-step perturbation flag (1 while a perturbation window was active for that env).
    if scheduler is not None:
        buf["perturb_active"] = []
        scheduler.reset(env, torch.arange(n_envs, device=device))  # assign groups + initial OOD DR
        print(f"[INFO] {scheduler.summary()}")

    print(f"[INFO] Collecting {num_steps} steps on {n_envs} envs "
          f"({num_steps * n_envs} pairs, trimming to {num_pairs})...")
    for step in range(num_steps):
        # Advance perturbation windows (applies pushes on OFF->ON edges) and corrupt
        # the policy's obs for active envs *before* the forward pass.
        active = None
        if scheduler is not None:
            active = scheduler.step(env)
            if scheduler.obs_noise_scale > 0.0:
                obs["policy"] = scheduler.corrupt_obs(obs["policy"], active)
        with torch.inference_mode():
            actions = policy(obs, stochastic_output=stochastic)
        # The action actually stepped is the (optionally) noise-corrupted action.
        actions_applied = (
            scheduler.corrupt_actions(actions, active) if scheduler is not None else actions
        )

        # Snapshot the (obs, action) pair *before* stepping. Clone because the
        # TensorDict tensors are overwritten by the next env.step. Record the
        # applied (noisy) action so the pair reflects what drove the transition.
        buf["obs"].append(obs["policy"].detach().cpu().clone())
        buf["action"].append(actions_applied.detach().cpu().clone())
        if scheduler is not None:
            buf["perturb_active"].append(active.detach().cpu().clone())
        # Raw robot world-state at the same timestep as obs. Order/dims MUST match
        # state_term_names/state_term_dims in _build_recording_meta.
        state = _assemble_state(robot, state_frames, frame_body_idx, n_envs)
        buf["state"].append(state.detach().cpu().clone())
        # Height scan at the same pre-step moment as state (per-ray ground clearance,
        # [N, K]); the dataset picks the snap-row scan as the FM height token.
        if has_scan:
            buf["height_scan"].append(
                mdp.scan_obs_stack(env.unwrapped, scan_sensor)[:, 0].detach().cpu().clone()
            )
        # Depth at the same pre-step moment, one token row per camera ([N, T, H*W]),
        # normalised by each camera's own obs-term params.
        if has_depth:
            buf["depth"].append(mdp.depth_obs_stack(
                env.unwrapped, mdp.fm_depth_groups(env.unwrapped)).detach().cpu().clone())
        # Desired reference at the same timestep, reconstructed to the world-frame
        # ``state`` layout. ``y_des``/``dy_des`` hold the value backing the current
        # obs (last computed during the previous env.step), so this stays aligned
        # with the pre-step ``state``/``obs`` snapshot above.
        if ref_cmd is not None:
            ref_state = _assemble_ref_state(ref_cmd, ref_base_frame, state_frames, n_envs)
            buf["ref_state"].append(ref_state.detach().cpu().clone())
        if record_cmd_vel:
            buf["target_vel"].append(ref_cmd.get_command_velocity().detach().cpu().clone())
        # Ego-anchored clip future (the FM target), captured at the same pre-step moment
        # as state/ref_state: the clip ego-anchored onto the robot at THIS step, lifted F
        # nodes forward through the current placement (no internal re-snap). Node layout
        # matches 'state': [root_pos(3), root_quat(4 xyzw), root_lin(3), root_ang(3),
        # joint_pos(nj), joint_vel(nj), <per frame pos(3)/quat(4)/lin(3)/ang(3)>].
        if record_ref_future:
            node = _assemble_ref_future_node(ref_cmd, future_offsets, state_frames, n_envs)
            buf["ref_future"].append(node.detach().cpu().clone())
            buf["cmd_future"].append(
                ref_cmd.get_future_command_velocity(future_offsets).detach().cpu().clone()
            )
            snap = ref_cmd._reanchor_active.to(torch.bool)
            if fut_ok_fn is not None:
                fits = fut_ok_fn()
                n_short += int((snap & ~fits).sum().item())
                snap = snap & fits
            buf["reanchor_active"].append(snap.detach().cpu().clone())
        buf["episode_id"].append(episode_id.detach().cpu().clone())
        buf["step_in_episode"].append(step_in_ep.detach().cpu().clone())

        with torch.inference_mode():
            obs, _, dones, extras = env.step(actions_applied)
            # Reset recurrent/transformer state for terminated envs.
            if use_policy_reset:
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # Disambiguate termination (failure) from truncation (time-out). The
        # wrapper returns dones = terminated | truncated and exposes truncation
        # via extras["time_outs"] (present when not is_finite_horizon).
        time_out = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        terminated = dones.bool() & ~time_out.bool()
        buf["done"].append(dones.detach().cpu().clone())
        buf["terminated"].append(terminated.detach().cpu().clone())
        buf["time_out"].append(time_out.detach().cpu().clone())

        # Episode bookkeeping for the *next* step's observation: envs that just
        # finished get a fresh id and a reset step counter.
        done_mask = dones.to(torch.bool)
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            new_ids = torch.arange(
                next_episode_id, next_episode_id + n_done, device=device, dtype=torch.int64
            )
            episode_id = episode_id.clone()
            episode_id[done_mask] = new_ids
            next_episode_id += n_done
            step_in_ep = step_in_ep.clone()
            step_in_ep[done_mask] = 0
            step_in_ep[~done_mask] += 1
            # Re-roll group / window / OOD DR for envs that just reset.
            if scheduler is not None:
                scheduler.reset(env, done_mask.nonzero(as_tuple=False).flatten())
        else:
            step_in_ep = step_in_ep + 1

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{num_steps}")

    if record_ref_future and n_short:
        print(f"[INFO] Cleared the snap flag on {n_short} step(s) whose "
              f"{int(future_offsets.max().item())}-step future would have crossed the clip "
              "end: those targets are partly the clip's frozen last frame, which the FM "
              "would learn as a stop-moving mode.")

    # Stack to [T, N, ...], flatten time and env into one sample axis, trim.
    out: dict[str, np.ndarray] = {}
    for key, tensors in buf.items():
        stacked = torch.stack(tensors, dim=0).numpy()  # [T, N, ...]
        if stacked.ndim >= 3:
            flat = stacked.reshape(-1, *stacked.shape[2:])
        else:
            flat = stacked.reshape(-1)
        out[key] = flat[:num_pairs]
    return out


def collect_windows(
    env: RslRlVecEnvWrapper,
    policy,
    policy_nn,
    initial_obs,
    num_windows: int,
    history_len: int,
    stochastic: bool,
    ref_cmd,
    ref_base_frame: str,
    state_frames: list[str] | None,
    future_len: int,
    writer,
    max_steps: int | None = None,
    distractor_clips=None,
    family_codes=None,
) -> int:
    """Roll out and STREAM pre-formed FM snap windows to ``writer`` (bounded RAM).

    Instead of buffering every step (like :func:`collect_pairs`) and discarding all but
    the snaps at train time, this forms one training window per env per step where the
    command's re-anchor flag fires AND the env has ``history_len`` consecutive valid (no
    reset since) history rows, and writes it straight to disk. This is equivalent to
    ``snap_windows(num_history=history_len, stride=1)`` over a full per-step recording
    (proven by ``tests/test_record_windows.py``), but never holds more than a per-env
    ring of the last ``H`` states plus one flush batch in memory.

    TWO history rings are kept per env, both ``[N, H, node]``: the MEASURED robot state, and
    the reference that was APPLIED at each of those steps (``history_ref``). The second one
    is what ``history_source="reference_partial"`` trains on; recording it unconditionally
    keeps that a train-time A/B rather than a second multi-hour recording.

    Args:
        env, policy, policy_nn, initial_obs, stochastic: as in :func:`collect_pairs`.
        num_windows: Stop after streaming this many windows.
        history_len: ``H``, history-state rows per window (>= trainable N).
        ref_cmd: Trajectory command exposing the look-ahead accessors + ``_reanchor_active``.
        state_frames: Extra body frames appended to each state row.
        future_len: ``F``, future nodes recorded per window.
        writer: A :class:`RecordedWindowWriter` to append batches to.
        max_steps: Safety cap on rollout steps (defaults to a generous multiple).

    Returns:
        The number of windows actually written.
    """
    n_envs = env.num_envs
    device = env.unwrapped.device
    use_policy_reset = version.parse(installed_version) >= version.parse("4.0.0")
    robot = env.unwrapped.scene.articulations["robot"]
    # The generator's dense scanner when --fm_height_scan added one, else the policy's.
    scan_sensor = None if args_cli.skip_height_scan else mdp.fm_scan_sensor(env.unwrapped)
    has_scan = scan_sensor is not None
    has_depth = bool(mdp.find_depth_obs_terms(env.unwrapped,
                                              mdp.fm_depth_groups(env.unwrapped)))

    state_frames = list(state_frames or [])
    frame_body_idx = None
    if state_frames:
        body_names = list(robot.data.body_names)
        frame_body_idx = torch.tensor(
            [body_names.index(f) for f in state_frames], dtype=torch.long, device=device
        )
    future_offsets = torch.arange(int(future_len), device=device)

    obs = initial_obs
    H = int(history_len)
    # Per-env history rings ([N, H, node]; row H-1 = most recent) + a consecutive-valid
    # counter reset to 0 on episode end so a window can never span a reset. ``hist`` is the
    # measured robot state; ``hist_ref`` is the reference applied at the same steps.
    hist: torch.Tensor | None = None
    hist_ref: torch.Tensor | None = None
    # The PLAN ANCHOR in force at each step: the robot root pose captured at the last
    # re-anchor (``_snap_target_*``). Ringed per step rather than read once at emit, because
    # at a snap step the re-anchor has already moved the snapshot onto the robot NOW -- reading
    # it then would give the anchor of the plan being STARTED, not the one whose tail the
    # history rows belong to. Per-step also makes a window that straddles a re-plan correct by
    # construction instead of by argument.
    hist_anchor: torch.Tensor | None = None
    # Newest LAST, so index -(k+1) is the frame k control steps ago.
    lag_frames: list[torch.Tensor] = []
    lag_states: list[torch.Tensor] = []
    valid = torch.zeros(n_envs, dtype=torch.long, device=device)
    # Distinct episode ids per env so the FM's by-episode train/val split has no leakage.
    episode_id = torch.arange(n_envs, dtype=torch.int64, device=device)
    next_episode_id = int(n_envs)

    if max_steps is None:
        # Generous cap: snaps fire ~ N/ego_anchor_every per step once rings fill; 50x the
        # ideal step count still terminates a run whose snaps are unexpectedly sparse.
        max_steps = 50 * math.ceil(num_windows / max(n_envs, 1)) + 1000

    print(f"[INFO] Streaming up to {num_windows} snap windows (H={H}, F={future_len}) "
          f"on {n_envs} envs; max_steps={max_steps}...")
    written = 0
    step = 0
    fut_ok_fn = _future_fits_fn(ref_cmd, int(future_offsets.max().item()))
    n_short = 0
    while written < num_windows and step < max_steps:
        with torch.inference_mode():
            actions = policy(obs, stochastic_output=stochastic)

        # Pre-step snapshot (aligned with obs), matching collect_pairs' snap timing.
        state_t = _assemble_state(robot, state_frames, frame_body_idx, n_envs)  # [N, node]
        if has_depth:
            # Rolling window of the last DEPTH_LAG_STEPS+1 frames AND the state each was
            # captured with. The state is what makes a lagged frame usable: the xyz channels
            # are unprojected at dataset build from the camera pose, and that pose comes from
            # the root quaternion and joint angles. Unproject a 40 ms-old image with the
            # pose at the re-plan and every pixel is displaced by the egomotion -- which is
            # exactly the registration error the xyz channels were added to remove. There is
            # no base-velocity estimate on this robot, so a lagged frame is expressed in the
            # frame it was CAPTURED in and the residual is accepted.
            dep_now = mdp.depth_obs_stack(
                env.unwrapped, mdp.fm_depth_groups(env.unwrapped))           # [N, T, H*W]
            lag_frames.append(dep_now.clone())
            lag_states.append(state_t.clone())
            del lag_frames[:-(DEPTH_LAG_STEPS + 1)]
            del lag_states[:-(DEPTH_LAG_STEPS + 1)]
        # The reference the policy was tracking when it chose this step's action. ``y_des``
        # was last written during the PREVIOUS env.step, so a pre-step read gives the
        # reference applied at step t-1 -- which is exactly the convention the runtime
        # snapshot uses (``FmTrackingCommand._robot_snapshot`` also runs before this step's
        # reference is resolved). The two must agree or the partially autoregressive history
        # is off by one.
        ref_t = _assemble_ref_state(ref_cmd, ref_base_frame, state_frames, n_envs)
        anchor_t = torch.cat([ref_cmd._snap_target_pos, ref_cmd._snap_target_quat], dim=-1)
        if hist is None:
            hist = torch.zeros(n_envs, H, state_t.shape[1], device=device)
            hist_ref = torch.zeros(n_envs, H, ref_t.shape[1], device=device)
            hist_anchor = torch.zeros(n_envs, H, anchor_t.shape[1], device=device)
        reanchor_t = ref_cmd._reanchor_active.to(torch.bool)                    # [N]

        # Push this step's rows into the rings; mark the row valid.
        hist = torch.roll(hist, shifts=-1, dims=1)
        hist[:, H - 1] = state_t
        hist_ref = torch.roll(hist_ref, shifts=-1, dims=1)
        hist_ref[:, H - 1] = ref_t
        hist_anchor = torch.roll(hist_anchor, shifts=-1, dims=1)
        hist_anchor[:, H - 1] = anchor_t
        valid = torch.clamp(valid + 1, max=H)

        # Emit a window for each env that snaps AND has a full valid ring AND whose whole
        # future window still lies inside the clip's REAL frames.
        #
        # Short clips are padded into the library by repeating their last frame, and the
        # future-window gather clamps to the library length, so a window opened within F steps
        # of ``clip_len - 1`` has a partly FROZEN target. Training on those teaches "the
        # reference stops moving" for conditioning that otherwise means "keep going", and
        # because the FM is multi-modal it would then sample that frozen mode occasionally at
        # deploy. Cheaper to drop those windows than to un-learn the mode.
        emit = reanchor_t & (valid >= H)
        if fut_ok_fn is not None:
            fits = fut_ok_fn()
            n_short += int((emit & ~fits).sum().item())
            emit = emit & fits
        if bool(emit.any()):
            idx = emit.nonzero(as_tuple=False).flatten()
            ref_future_t = _assemble_ref_future_node(ref_cmd, future_offsets, state_frames, n_envs)
            cmd_future_t = ref_cmd.get_future_command_velocity(future_offsets)  # [N, F, 3]
            # The dataset reads ONLY root_pos_w / root_quat_w out of history_state, and uses
            # each history row's copy as the FRAME that row's reference is observed in. So
            # substituting the plan anchor into rows 0..H-2 is the whole of "observe the
            # history in its plan's frame" -- no format or dataset change.
            #
            # Row H-1 keeps the MEASURED pose: that row is the TARGET's anchor, and the target
            # is observed relative to the robot as it actually is at the snap.
            #
            # The other columns of rows 0..H-2 stay measured and stay unread. That makes those
            # rows a deliberate mix, which is why RecordingMeta.history_anchor records it and
            # the dataset refuses a recording that does not say "plan".
            hist_out = hist[idx].clone()
            hist_out[:, :H - 1, 0:7] = hist_anchor[idx][:, :H - 1]
            batch = {
                "history_state": hist_out.detach().cpu().numpy(),
                "history_ref": hist_ref[idx].detach().cpu().numpy(),
                "ref_future": ref_future_t[idx].detach().cpu().numpy(),
                "cmd_future": cmd_future_t[idx].detach().cpu().numpy(),
                "episode_id": episode_id[idx].detach().cpu().numpy(),
            }
            # Per-window distractor tag (see _distractor_clip_mask): recorded so training can
            # drop these windows via --skip_distractors without re-recording. All-zero when the
            # library has no distractor family, so the field is always present and unambiguous.
            cid = ref_cmd.clip_id[idx]
            batch["is_distractor"] = (
                distractor_clips[cid.to(distractor_clips.device)].to(torch.uint8).cpu().numpy()
                if distractor_clips is not None
                else np.zeros(idx.numel(), dtype=np.uint8)
            )
            # Per-window family code (see _clip_family_codes). Superset of is_distractor:
            # that flag stays for the recordings and flags that already exist.
            batch["clip_family"] = (
                family_codes[cid.to(family_codes.device)].to(torch.uint8).cpu().numpy()
                if family_codes is not None
                else np.zeros(idx.numel(), dtype=np.uint8)
            )
            # Where this window came FROM: clip, and the step its node 0 sits at. Lets any
            # per-clip/per-step library column be joined offline -- notably anchor_mask_t,
            # whose boundaries are the ground-truth terrain transitions. `_step` is the same
            # counter the command uses to gather `ref_future`, so node 0 == this step.
            batch["clip_id"] = cid.to(torch.int32).cpu().numpy()
            batch["clip_step"] = ref_cmd._step[idx].to(torch.int32).cpu().numpy()
            # The source ENV. Its tile instance and origin are what make the recorded WORLD
            # poses interpretable off-line; clip_id fixes the tile TYPE but not which of the
            # identical instances this window sat on.
            batch["env_id"] = idx.to(torch.int32).cpu().numpy()
            if has_scan:
                scan = mdp.scan_obs_stack(env.unwrapped, scan_sensor)[:, 0]  # [N, K]
                batch["height_scan"] = scan[idx].detach().cpu().numpy()
            if has_depth:
                # ``depth`` stays the CURRENT frame, unchanged, so every existing reader and
                # every no-latency run keeps working untouched.
                dep = lag_frames[-1]                                        # [N, T, H*W]
                batch["depth"] = dep[idx].detach().cpu().numpy()
                # ``depth_lag``/``depth_lag_state`` are lags 0..DEPTH_LAG_STEPS, newest first,
                # so index k IS a lag of k steps. Early in an episode the buffer is short; the
                # oldest available frame is repeated rather than zero-filled, because a zero
                # depth row is a VALID reading (a surface at the near plane) and would train
                # the model on a wall that is not there.
                lag_d = [lag_frames[-1 - min(k, len(lag_frames) - 1)]
                         for k in range(DEPTH_LAG_STEPS + 1)]
                lag_s = [lag_states[-1 - min(k, len(lag_states) - 1)]
                         for k in range(DEPTH_LAG_STEPS + 1)]
                batch["depth_lag"] = torch.stack(lag_d, 1)[idx].detach().cpu().numpy()
                batch["depth_lag_state"] = torch.stack(lag_s, 1)[idx].detach().cpu().numpy()
            written += writer.write_batch(batch)
            if written >= num_windows:
                break

        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            if use_policy_reset:
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # Invalidate the ring for envs that just reset so the next window can't span it;
        # assign fresh episode ids (matches collect_pairs' bookkeeping).
        done_mask = dones.to(torch.bool)
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            valid[done_mask] = 0
            new_ids = torch.arange(
                next_episode_id, next_episode_id + n_done, device=device, dtype=torch.int64
            )
            episode_id = episode_id.clone()
            episode_id[done_mask] = new_ids
            next_episode_id += n_done

        step += 1
        if step % 100 == 0:
            print(f"  step {step}: {written}/{num_windows} windows")

    if written < num_windows:
        print(f"[WARN] Reached max_steps={max_steps} with {written}/{num_windows} windows "
              f"(snaps sparser than expected); saving what was collected.")
    if n_short:
        print(f"[INFO] Dropped {n_short} snap window(s) whose {int(future_offsets.max().item())}"
              f"-step future would have crossed the clip end "
              f"({100.0 * n_short / max(n_short + written, 1):.1f}% of "
              f"otherwise-eligible snaps): those targets are partly the clip's frozen last "
              f"frame, which the FM would learn as a stop-moving mode.")
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load a checkpoint, roll it out, and save recorded (obs, action) pairs.

    Two modes: legacy ``--num_pairs`` buffers every step and writes one ``.npz``;
    ``--num_windows`` streams only the FM snap windows to a chunked ``.h5`` (bounded RAM).
    """
    windowed = args_cli.num_windows is not None
    if windowed == (args_cli.num_pairs is not None):
        raise ValueError(
            "Pass exactly one of --num_pairs (legacy per-step .npz) or "
            "--num_windows (streamed snap-window .h5)."
        )
    count = args_cli.num_windows if windowed else args_cli.num_pairs
    if count <= 0:
        raise ValueError(f"--num_{'windows' if windowed else 'pairs'} must be positive; got {count}.")
    if windowed:
        if args_cli.history_len < 1:
            raise ValueError(f"--history_len must be >= 1; got {args_cli.history_len}.")
        if args_cli.perturb:
            raise ValueError("--perturb (recovery data) is only supported in --num_pairs mode.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    resume_path = _resolve_resume_path(agent_cfg)
    log_dir = os.path.dirname(resume_path)
    out_dir = args_cli.output_dir or os.path.join(log_dir, "recorded_data")

    with launch_simulation(env_cfg, args_cli):
        # Recursive contact sensor patch (matches play.py / train.py).
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = log_dir

        # Override which clip library the teacher tracks (the recorded distribution
        # mirrors this clip). Done before auto_clip_device so its size check sees it.
        if args_cli.clip is not None:
            traj_ref = getattr(getattr(env_cfg, "commands", None), "traj_ref", None)
            if traj_ref is None or not hasattr(traj_ref, "path"):
                raise ValueError("--clip given but the task has no commands.traj_ref.path to override.")
            print(f"[INFO] Overriding clip library: {traj_ref.path} -> {args_cli.clip}")
            traj_ref.path = args_cli.clip
            # A ClipTerrainImporter builds its terrain MESH from the same library the command
            # tracks, so repoint it too (mirrors play.py). Missing this silently records the
            # robot tracking library A's clips on library B's tiles: the stairs are not where
            # the clip expects, so the robot never climbs, and the height scan faithfully
            # reports geometry metres overhead (measured: a mesh reaching +20.25 m for a
            # library whose tallest authored surface is +5.44 m, and 30-37% of recorded scan
            # entries <= -1 m where a matched scene produces none).
            terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
            if terrain is not None and hasattr(terrain, "clip_library_path"):
                print(f"[INFO] Overriding terrain importer mesh library: "
                      f"{terrain.clip_library_path} -> {args_cli.clip}")
                terrain.clip_library_path = args_cli.clip

        # DECORATION ON, whichever task this is. The teacher's TRAINING scene now renders
        # none of it (tile walls, flat-tile rooms, a clutter sidecar) -- nothing in that
        # stack avoids obstacles and it only costs depth pixels over a 12k-iteration run --
        # while the -play chain turns it back on. Both are reachable here, and a recording
        # made on the train task would then fit the generator on a bare world while the
        # world it is deployed into has walls: exactly the vertical surface it already
        # misreads as a box. Asserted rather than inherited, because which task the recorder
        # is pointed at must not decide what the generator sees.
        terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain_cfg is not None and hasattr(terrain_cfg, "render_decor"):
            if not terrain_cfg.render_decor:
                print("[INFO] scene.terrain.render_decor False -> True: a recording is what "
                      "the generator conditions on, so it renders the visual decoration even "
                      "when the task it was taken from does not.")
            terrain_cfg.render_decor = True

        # Where inside its clip each episode starts. Training SAMPLES a start (so a policy
        # sees every phase as an entry point); a recording wants uniform COVERAGE of clip
        # phase, and only a start at frame 0 gives it -- a sampled start run to the clip end
        # visits step t with probability ~t/clip_len, which is the ramp measured in the
        # --start_step help. Set on the reset EVENT rather than by disabling
        # ``random_start_phase``, so the value lands in the dumped provenance env.yaml.
        if args_cli.start_step >= 0:
            reset_ev = getattr(getattr(env_cfg, "events", None), "reset_on_clip", None)
            if reset_ev is None or "pin_start_step" not in getattr(reset_ev, "params", {}):
                raise ValueError(
                    "--start_step given but the task has no events.reset_on_clip carrying a "
                    "pin_start_step param. Pass --start_step -1 to leave the task's own "
                    "(sampled) start alone."
                )
            reset_ev.params["pin_start_step"] = int(args_cli.start_step)
            print(f"[INFO] Every episode starts at clip step {args_cli.start_step} -- windows "
                  f"cover clip phase uniformly. Pass --start_step -1 to sample as training does.")

        # And which clip each reset picks: without replacement, so the run spans the library
        # rather than covering a random 63% of it (see --clip_draw).
        reset_ev = getattr(getattr(env_cfg, "events", None), "reset_on_clip", None)
        if reset_ev is None or "clip_draw" not in getattr(reset_ev, "params", {}):
            raise ValueError(
                "the task has no events.reset_on_clip carrying a clip_draw param; it is not a "
                "clip-tracking task and cannot be recorded from."
            )
        reset_ev.params["clip_draw"] = args_cli.clip_draw

        # Give the GENERATOR its observation group. UNCONDITIONAL on a depth task, because
        # the policy now reads the NEAR camera alone: without the group the recorder's
        # ``fm_depth_groups`` scoping falls back to "every group" = that one camera, and it
        # would then stamp a one-camera depth_camera_spec into the recording, train a
        # half-blind generator on it and export a matching fm.onnx. Every artifact would
        # agree with every other, and nothing downstream could tell it from a deliberate
        # one-camera design. --fm_depth_resolution only DECLARES a different sampling grid
        # (declared rather than derived, unlike play.py: there is no FM artifact yet -- this
        # run is what produces the data one will be trained on); absent, the group is still
        # installed, over the policy's own cameras and with no second ray cast.
        if any(isinstance(s, MultiMeshRayCasterCameraCfg)
               for s in vars(env_cfg.scene).values()):
            from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
                install_fm_depth_group,
            )
            fm_res: tuple[int, int] | tuple[None, None] = (None, None)
            if args_cli.fm_depth_resolution.strip():
                spec = args_cli.fm_depth_resolution.replace("x", ",").split(",")
                if len(spec) != 2:
                    raise ValueError(
                        f"--fm_depth_resolution={args_cli.fm_depth_resolution!r}; expected 'WxH'."
                    )
                fm_res = (int(spec[0]), int(spec[1]))
            install_fm_depth_group(env_cfg, *fm_res)
        elif args_cli.fm_depth_resolution.strip():
            raise ValueError(
                f"--fm_depth_resolution={args_cli.fm_depth_resolution!r} but --task "
                f"{args_cli.task!r} carries no depth cameras to sample."
            )

        # Same story for the generator's own dense height scanner: DECLARED here, and for the
        # same reason -- no FM artifact exists yet to derive it from.
        if args_cli.fm_height_scan.strip():
            from robot_rl.tasks.manager_based.robot_rl.g1.g1_fm_scan_env_cfg import (
                add_fm_height_scan,
            )
            text = args_cli.fm_height_scan.strip()
            try:
                extents, _, res = text.partition("@")
                sx, sy = (float(v) for v in extents.replace("x", ",").split(","))
                resolution = float(res)
            except ValueError as exc:
                raise ValueError(
                    f"--fm_height_scan={text!r}; expected 'SXxSY@RES' in metres, e.g. "
                    "'3.0x3.0@0.025'."
                ) from exc
            add_fm_height_scan(env_cfg, size=(sx, sy), resolution=resolution)

        # Hold the clip library off-GPU when large (auto: cpu for >1 GB libs, which
        # OOM a small card at scene init). Explicit --clip_device overrides.
        auto_clip_device(env_cfg, args_cli.clip_device)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # BEFORE a single step is recorded: the depth this file will carry must be every
        # camera the robot has, not whatever the policy happens to observe.
        _assert_generator_sees_every_camera(env.unwrapped, env_cfg)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # Recording never evaluates the value function -- it only needs the actor to
        # produce actions -- so honour a runner cfg that opts out of restoring the critic
        # (``load_critic = False``), exactly as play.py does. Without this, recording from
        # the FM-tracker task (whose runner cfg sets it, because that env has no
        # privileged critic group) fails on a critic it was never going to use. Actor
        # weights are still loaded strictly, so a real policy mismatch fails loudly.
        if getattr(agent_cfg, "load_critic", True):
            runner.load(resume_path)
        else:
            print("[INFO] Skipping the critic (load_critic=False): loading actor weights only.")
            runner.load(resume_path, load_cfg={"actor": True, "critic": False,
                                               "optimizer": False, "iteration": True, "rnd": True})
        runner.alg.eval_mode()

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # For rsl-rl < 4.0.0 the inference policy doesn't expose reset(); fetch
        # the underlying network to reset recurrent state instead (see play.py).
        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy_nn = None
        elif version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # Number of records to collect: windows (windowed) or (obs, action) pairs (legacy).
        num_records = count

        # Resolve the trajectory command to record the reference from, plus the extra
        # body frames appended to the recorded state. Default frames = the command's
        # end-effector outputs (ankles/wrists). Resolved before meta so layout matches.
        ref_name, ref_cmd = _resolve_ref_command(env, args_cli.ref_command_name)
        if args_cli.state_frames is not None:
            state_frames = list(args_cli.state_frames)
        elif ref_cmd is not None and hasattr(ref_cmd, "resolve_ee_frames"):
            state_frames = ref_cmd.resolve_ee_frames()
        else:
            state_frames = []
        if ref_cmd is not None:
            print(f"[INFO] Recording reference state from command '{ref_name}' "
                  f"(base frame '{args_cli.ref_base_frame}').")
            if getattr(ref_cmd, "command_velocity_available", False):
                print(f"[INFO] Recording commanded velocity (target_vel) from command '{ref_name}'.")
        else:
            print("[WARN] No trajectory command with get_reference_world_state found; "
                  "skipping ref_state recording.")
        if state_frames:
            print(f"[INFO] Recording {len(state_frames)} extra state frames: {state_frames}")

        # Anchor-mask audit: a clip with NO free root DOF can never be re-anchored, but the snap
        # still fires, so its FM windows are tracking drift. Warn loudly rather than silently
        # poisoning the recording (see _audit_anchor_masks).
        _audit_anchor_masks(ref_cmd)
        # Per-clip distractor flag, recorded so `train.py --skip_distractors` can drop those
        # windows later without re-recording.
        distractor_clips = _distractor_clip_mask(ref_cmd)
        family_codes, family_names = _clip_family_codes(ref_cmd)

        meta = _build_recording_meta(
            env, agent_cfg, resume_path, log_dir, num_records, state_frames, windowed=windowed
        )
        # The code -> name table for the per-window 'clip_family' tag. Meaningless without
        # it, so it travels in the recording rather than being reconstructed from a library
        # that may have moved on.
        meta.family_names = list(family_names)
        # What the per-window clip_id / clip_step index INTO. The hash is the part that
        # matters: libraries get regenerated in place under the same filename, and a join
        # against the wrong one returns plausible garbage rather than an error.
        clip_lib_path = getattr(getattr(ref_cmd, "cfg", None), "path", "") or ""
        meta.clip_library_path = str(clip_lib_path)
        meta.clip_library_sha256 = (
            sha256_file(clip_lib_path) if clip_lib_path and os.path.exists(clip_lib_path) else ""
        )

        # Ego-anchored clip-future target (robot-state-conditioned FM): recorded when the
        # command exposes the look-ahead accessors + a command velocity, unless disabled.
        rf_len = 0 if args_cli.no_ref_future else int(args_cli.ref_future_len)
        will_record_ref_future = (
            rf_len > 0 and ref_cmd is not None and hasattr(ref_cmd, "get_future_world_base")
            and getattr(ref_cmd, "command_velocity_available", False)
        )
        if will_record_ref_future:
            meta.ref_source = "ego_anchored_clip_future"
            meta.ref_future_len = rf_len
            meta.ref_future_term_names = list(meta.state_term_names)  # node layout == state
            meta.ref_future_term_dims = list(meta.state_term_dims)
            # The snap period the windows were cut at. The FM tracker's fm_update_every must
            # match it: the horizon is sized from the re-plan period, so a tracker running a
            # different one is both off-distribution and possibly short of horizon.
            meta.fm_snap_every = int(getattr(ref_cmd.cfg, "fm_snap_every", 0))
            print(f"[INFO] Recording ego-anchored clip future (ref_future): F={rf_len} nodes/step "
                  f"+ cmd_future + reanchor_active (snap every {meta.fm_snap_every} steps).")
        elif ref_cmd is not None:
            meta.ref_source = "reference"

        # Single observation fetch reused for the memory estimate, the provenance
        # export, and the start of collection. compute() defaults to
        # update_history=False, so this does not mutate observation history.
        obs = env.get_observations()

        # Warn (don't fail) on very large in-memory datasets: the legacy npz path buffers
        # everything in RAM before writing (the windowed .h5 path streams, so no warning).
        if not windowed:
            obs_dim = int(obs["policy"].shape[1])
            est_bytes = num_records * (obs_dim + int(env.num_actions)) * 4
            if est_bytes > 8 * (1 << 30):
                print(f"[WARN] Estimated ~{est_bytes / (1 << 30):.1f} GiB of obs+action data will be "
                      f"buffered in RAM before writing. Consider --num_windows (streamed .h5).")

        # Export richer provenance YAML (gains, scales, default joint angles).
        # Optional: skip if the task lacks the terms it needs.
        with torch.inference_mode():
            first_actions = policy(obs, stochastic_output=args_cli.stochastic)
        try:
            export_policy_parameters(env, obs, first_actions, log_dir)
        except (KeyError, ValueError) as exc:
            print(f"[WARN] export_policy_parameters skipped: {exc}")

        # Clear any recurrent/transformer state advanced by the export-time
        # forward pass so the first recorded pair starts from a clean state.
        if policy_nn is None:
            policy.reset()
        else:
            policy_nn.reset()

        # -- Windowed mode: stream only the FM snap windows to a chunked .h5 --------
        if windowed:
            if not will_record_ref_future:
                raise ValueError(
                    "--num_windows requires the ego-anchored clip future (ref_future): use a "
                    "clip-tracking task exposing get_future_world_base + a command velocity, "
                    "and do not pass --no_ref_future."
                )
            # 0 makes RecordedWindowWriter omit the dataset entirely. This MUST agree with
            # collect_windows' own `has_scan`, which is why both go through the same resolver:
            # the writer accepts only keys it declared, so declaring height_scan while the loop
            # never fills it is a KeyError on the first emitted batch.
            _scan_sensor = (None if args_cli.skip_height_scan
                            else mdp.fm_scan_sensor(env.unwrapped))
            if _scan_sensor is None and not mdp.find_depth_obs_terms(
                    env.unwrapped, mdp.fm_depth_groups(env.unwrapped)):
                raise ValueError(
                    "--num_windows needs SOME exteroception: a height scanner, depth obs "
                    "terms, or --skip_height_scan on a depth task. Record on a task that has "
                    "one (e.g. G1-teacher-clf-terrain-depth)."
                )
            H = int(args_cli.history_len)
            node_dim = int(sum(meta.state_term_dims))
            K = (int(mdp.scan_obs_stack(env.unwrapped, _scan_sensor).shape[-1])
                 if _scan_sensor is not None else 0)
            cmd_dim = int(
                ref_cmd.get_future_command_velocity(
                    torch.arange(1, device=env.unwrapped.device)
                ).shape[-1]
            )
            # Depth (the deployable FM signal) is recorded alongside the scan whenever the
            # task exposes depth obs terms, so one session can train either FM.
            depth_shape = None
            if mdp.find_depth_obs_terms(env.unwrapped,
                                        mdp.fm_depth_groups(env.unwrapped)):
                dep = mdp.depth_obs_stack(
                    env.unwrapped, mdp.fm_depth_groups(env.unwrapped))   # [N, T, H*W]
                depth_shape = (int(dep.shape[1]), int(dep.shape[2]))
                print(f"[INFO] recording depth alongside the height scan: {depth_shape} per window")
            meta.format = "windowed_h5"
            meta.history_len = H
            meta.height_scan_dim = K
            # The scan's GEOMETRY, not just its ray count: a convolutional encoder needs the
            # (height, width) fold, and num_rays cannot tell a grid from its transpose. Scoped
            # to the sensor actually recorded, so a depth task's untouched 176-ray critic
            # scanner is not described alongside the generator's dense one.
            if _scan_sensor is not None:
                meta.height_scan_spec = mdp.height_scan_specs(env.unwrapped, _scan_sensor)
                _spec = meta.height_scan_spec[0]
                print(f"[INFO] height scan from {_scan_sensor!r}: "
                      f"{_spec['size'][0]}x{_spec['size'][1]} m @ {_spec['resolution']} m -> "
                      f"fold ({_spec['height']}, {_spec['width']}) = {K} rays"
                      + (" [PRIVILEGED: not deployable]" if _spec["privileged"] else ""))
                # Noise-free BY CONSTRUCTION, and this records the fact so the FM dataset does
                # not have to be told twice. Every corruption route is provably absent: the
                # reading goes through the raw obs function (so the observation manager's noise
                # and clip never run), height_scan_specs has already refused any drift range on
                # a privileged sensor, and add_fm_height_scan has already refused an event term
                # that names either scanner. Depth's own flag is the opposite polarity and is
                # gated on extero_source == "depth", so a scan run falls through it entirely.
                meta.extero_noise_free = True
            # content_sha256 stays empty for the streamed format (the whole dataset is
            # never in RAM to hash); sha256_file(<.h5>) identifies the file if needed.
            out_path = unique_output_path(out_dir, prefix="recorded", ext=".h5")
            with RecordedWindowWriter(
                out_path, num_records, H, node_dim, rf_len, K, cmd_dim, meta,
                depth_shape=depth_shape,
                depth_lag_steps=(DEPTH_LAG_STEPS if depth_shape is not None
                                 else 0),
                # Always tag, even for a library with no distractor family (then all-zero): the
                # tag's PRESENCE is what lets `train.py --skip_distractors` distinguish "none to
                # drop" from "this recording predates the tag" and refuse the latter.
                tag_distractors=True,
                # And the family code it generalises. Without this the writer never declared
                # the dataset, so the key record_data assembled was silently dropped and
                # --skip_families could never match.
                tag_families=True,
                # And the index back into the library, which generalises BOTH: with
                # clip_id + clip_step any per-clip/per-step column joins offline. The one
                # that motivated it is anchor_mask_t, whose boundaries are the ground-truth
                # terrain transitions -- previously only guessable from a height threshold.
                tag_clip_index=True,
                # Same reasoning for the applied-reference ring: recording it always makes
                # `history_source=reference_partial` a train-time A/B, and its absence
                # unambiguously means "this recording predates the mode".
                record_history_ref=True,
                # Opt-in f16 for the scan only. At a dense grid this halves both the file and
                # the training-time dataset RAM, and 1 mm quantisation is far below the grid
                # cell -- but it is never the default, so the depth baseline's arrays stay
                # bit-identical and the A/B is not confounded by a storage change.
                dtypes=({"height_scan": "float16"} if args_cli.fm_scan_float16 else None),
            ) as writer:
                n_written = collect_windows(
                    env, policy, policy_nn, obs, num_records, H, args_cli.stochastic,
                    ref_cmd=ref_cmd, ref_base_frame=args_cli.ref_base_frame,
                    state_frames=state_frames, future_len=rf_len, writer=writer,
                    distractor_clips=distractor_clips, family_codes=family_codes,
                )
            prov_dir = os.path.splitext(out_path)[0] + ".provenance"
            git_snapshot(prov_dir)
            try:
                dump_yaml(os.path.join(prov_dir, "params", "env.yaml"), env_cfg)
                dump_yaml(os.path.join(prov_dir, "params", "agent.yaml"), agent_cfg)
            except Exception as exc:
                print(f"[WARN] recording params dump skipped: {exc}")
            print(f"[INFO] Streamed {n_written} snap windows to: {out_path}")
            print(f"[INFO]   H={H} F={rf_len} node_dim={node_dim} K={K} cmd_dim={cmd_dim}; "
                  f"format=windowed_h5  provenance={prov_dir}")
            _report_clip_coverage(ref_cmd)
            env.close()
            return

        # -- Legacy mode: buffer every step, write one .npz --------------------------
        num_steps = math.ceil(num_records / env.num_envs)

        # Optional perturbation scheduler for recovery-data collection.
        scheduler = None
        if args_cli.perturb:
            push_v = float(args_cli.push_vel)
            push_range = {"x": (-push_v, push_v), "y": (-push_v, push_v)} if push_v > 0 else None
            scheduler = PerturbationScheduler(
                num_envs=env.num_envs,
                dt=float(env.unwrapped.step_dt),
                device=env.unwrapped.device,
                clean_fraction=args_cli.clean_fraction,
                on_range_s=tuple(args_cli.perturb_on_range_s),
                off_range_s=tuple(args_cli.perturb_off_range_s),
                action_noise_scale=args_cli.action_noise_scale,
                push_vel_range=push_range,
                ood_dr=args_cli.ood_dr,
                ood_dr_scale=args_cli.ood_dr_scale,
                obs_noise_scale=args_cli.obs_noise_scale,
                seed=agent_cfg.seed,
            )

        data = collect_pairs(
            env, policy, policy_nn, obs, num_steps, num_records, args_cli.stochastic,
            ref_cmd=ref_cmd, ref_base_frame=args_cli.ref_base_frame, state_frames=state_frames,
            scheduler=scheduler, ref_future_len=rf_len,
        )

        if data.get("height_scan") is not None:
            meta.height_scan_dim = int(data["height_scan"].shape[1])
            _npz_scan = mdp.fm_scan_sensor(env.unwrapped)
            if _npz_scan is not None:
                meta.height_scan_spec = mdp.height_scan_specs(env.unwrapped, _npz_scan)
                meta.extero_noise_free = True
            print(f"[INFO] Recorded height_scan: {data['height_scan'].shape} "
                  f"(K={meta.height_scan_dim} rays) -- the FM height-conditioning token.")

        out_path = unique_output_path(out_dir, prefix="recorded", ext=".npz")
        # Content hash of the recorded arrays -> the dataset self-identifies.
        meta.content_sha256 = fingerprint_arrays(data)
        save_recorded_data(out_path, data, meta)
        # Per-recording provenance sidecar: git snapshot + the env/agent cfg used.
        prov_dir = os.path.splitext(out_path)[0] + ".provenance"
        git_snapshot(prov_dir)
        try:
            dump_yaml(os.path.join(prov_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(prov_dir, "params", "agent.yaml"), agent_cfg)
            if scheduler is not None:
                dump_yaml(os.path.join(prov_dir, "params", "perturb.yaml"), {
                    "clean_fraction": args_cli.clean_fraction,
                    "perturb_on_range_s": list(args_cli.perturb_on_range_s),
                    "perturb_off_range_s": list(args_cli.perturb_off_range_s),
                    "action_noise_scale": args_cli.action_noise_scale,
                    "push_vel": args_cli.push_vel,
                    "obs_noise_scale": args_cli.obs_noise_scale,
                    "ood_dr": args_cli.ood_dr,
                    "ood_dr_scale": args_cli.ood_dr_scale,
                })
        except Exception as exc:
            print(f"[WARN] recording params dump skipped: {exc}")
        print(f"[INFO] Saved {meta.num_pairs} (obs, action) pairs to: {out_path}")
        print(f"[INFO]   content_sha256={meta.content_sha256[:12]}  provenance={prov_dir}")
        ref_shape = data["ref_state"].shape if "ref_state" in data else None
        tvel_shape = data["target_vel"].shape if "target_vel" in data else None
        print(f"[INFO]   obs shape={data['obs'].shape}, action shape={data['action'].shape}, "
              f"state shape={data['state'].shape}, ref_state shape={ref_shape}, "
              f"target_vel shape={tvel_shape}, "
              f"terminations flagged={int(data['terminated'].sum())}")
        if data.get("ref_future") is not None:
            rf = data["ref_future"]; ra = data["reanchor_active"]
            print(f"[INFO]   ref_future shape={rf.shape}, cmd_future shape={data['cmd_future'].shape}, "
                  f"reanchor_active (snap steps)={int(ra.sum()):,}/{ra.size:,} "
                  f"({100.0 * float(ra.mean()):.1f}%)")
        if "perturb_active" in data and data["perturb_active"] is not None:
            pa = data["perturb_active"]
            print(f"[INFO]   perturb_active: {int(pa.sum()):,}/{pa.size:,} steps "
                  f"({100.0 * float(pa.mean()):.1f}%) perturbed")

        env.close()


if __name__ == "__main__":
    main()
