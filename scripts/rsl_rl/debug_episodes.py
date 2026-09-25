#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Record a few debug episodes (Isaac) into a bundle the viser debugger can replay.

Stage 1 of the debug-visualisation pair; ``view_debug_episodes.py`` is stage 2 (no Isaac,
so you can scrub and re-plot without holding the GPU).

Captures, per step, for a handful of envs chosen to span terrain families:

* ``cmd_vel``      -- the velocity command the POLICY receives (the ``command_vel`` obs term)
* ``ach_vel``      -- what the robot actually achieves (root lin/ang velocity, body frame)
* ``scan_obs``     -- the height scan AS OBSERVED, sliced straight out of the policy obs
                      vector (so it carries the term's noise + ``clip=(-1, 1)``)
* ``scan_raw``     -- the same scan straight from the sensor, unclipped and unnoised
* ``ref_*``        -- the reference the CLF tracks (pelvis + feet, world frame) and the
                      robot's matching measured frames
* ``fm_*``         -- when the active command is an :class:`FmTrackingCommand`, the FM's
                      predicted horizon, so the viewer can draw the FM ribbon

``scan_obs`` vs ``scan_raw`` is the pair that shows clipping: on terrain the raw scan runs
well outside +-1 m while the observation is flat-topped at the bound.

Usage (terrain teacher, clip references)::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/debug_episodes.py \
        --task G1-teacher-clf-terrain-play --checkpoint <policy.pt> \
        --clip clips/g1_mb_v1_view10.h5 --clip_device cpu \
        --episodes 6 --out debug_bundle.npz

Usage (FM in the loop -- adds the FM ribbon)::

    ... --task G1-tracker-clf-fm-terrain-play --fm_checkpoint <fm.pt> ...

Usage (the SONIC baseline instead of the trained teacher -- no checkpoint, ONNX pair)::

    ... --task G1-sonic-terrain-play --policy sonic --sonic_dir <release dir> ...

For a policy-vs-policy comparison add ``--clip_pick k --no_dr --capture_terrain_ids a,b,c`` to
BOTH runs: that fixes the tile -> clip mapping, the dynamics and the captured tiles independently
of how each policy's episodes end. ``scripts/rsl_rl/compare_trackers.py`` drives exactly that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip  (same dir; import-safe before the app launches)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True, help="Play task id (clip teacher or FM tracker).")
parser.add_argument("--fm_checkpoint", default=None, help="FM checkpoint (FM-tracker tasks).")
parser.add_argument("--fm_zero_noise", action="store_true",
                    help="Integrate the FM flow from x0=0 so its prediction is deterministic "
                         "given the conditioning. Off by default (training-matched Gaussian). "
                         "Use it to A-B against a MuJoCo run, whose runner defaults zero-noise "
                         "ON -- otherwise the two differ by an RNG draw before any dynamics.")
parser.add_argument("--clip", default=None, help="Clip library .h5 (also repoints the terrain mesh).")
parser.add_argument("--clip_device", default=None, help="Hold the clip library here, e.g. 'cpu'.")
parser.add_argument("--num_envs", type=int, default=16, help="Envs to simulate.")
parser.add_argument("--episodes", type=int, default=6, help="Envs to CAPTURE (one bundle entry each).")
parser.add_argument("--steps", type=int, default=1000, help="Steps to capture per env.")
parser.add_argument("--random-start", action="store_true", default=False,
                    help="Sample a random start step within each clip (the TRAINING behaviour). "
                         "Default OFF: every episode starts at clip step 0, so one episode == "
                         "one whole clip and the clip_end_reached truncation ends it. That makes "
                         "an episode in this bundle a single un-stitched clip.")
parser.add_argument("--policy", choices=("rsl_rl", "sonic", "zero"), default="rsl_rl",
                    help="Which policy drives the robot. 'rsl_rl' loads --checkpoint through an "
                         "OnPolicyRunner (the trained teacher). 'sonic' runs NVIDIA's released "
                         "SONIC tracker from its ONNX pair via --sonic_dir, on the G1-sonic-* "
                         "tasks; no checkpoint is involved. 'zero' commands the default pose "
                         "every step -- an asset diagnostic, not a baseline: it isolates the "
                         "robot's own dynamics (spawn interpenetration, gain blow-ups) from "
                         "whatever the policy is doing.")
parser.add_argument("--sonic_dir", default=None,
                    help="Directory with model_encoder.onnx / model_decoder.onnx / "
                         "observation_config.yaml (--policy sonic).")
parser.add_argument("--clip_pick", type=int, default=None,
                    help="Pin every env to the k-th clip of its terrain tile instead of sampling "
                         "one at random. Required for a policy-vs-policy comparison: a shared "
                         "--seed does not survive the two policies terminating at different "
                         "steps, because their reset streams then diverge.")
parser.add_argument("--capture_terrain_ids", default=None,
                    help="Comma-separated terrain ids to capture (default: the first --episodes "
                         "envs). Selects the envs sitting on those tiles, so a bundle covers "
                         "chosen terrain families rather than whatever env 0..N-1 landed on.")
parser.add_argument("--capture_families", default=None,
                    help="Comma-separated clip-library terrain FAMILIES to capture (the "
                         "manifest's 'family' field, e.g. family_stairs_up.h5), or 'all'. "
                         "Resolved against the tiles this scene actually instantiated, so it "
                         "works at any --num_envs. Takes precedence over --capture_terrain_ids.")
parser.add_argument("--per_family", type=int, default=2,
                    help="Tiles to capture per family for --capture_families.")
parser.add_argument("--ego_anchor_fraction", type=float, default=None,
                    help="Override the command term's ego_anchor_fraction. Pinning this is "
                         "REQUIRED for a reproducible A/B: at the default 0.5 the clip is "
                         "re-placed onto the robot root for a RANDOM half of the envs every "
                         "ego_anchor_every steps, so the tracked reference itself differs "
                         "between runs (measured: the same clip on the same tile diverged at "
                         "step 12 and gave 0.069 vs 0.247 m mean pelvis error). Use 0.0 to keep "
                         "the clip world-anchored, which also makes tracking error mean ABSOLUTE "
                         "error rather than error against a reference that follows the robot.")
parser.add_argument("--no_dr", action="store_true",
                    help="Null every startup domain-randomization event (joint friction, "
                         "armature, PD gains, torso mass, default-pose calibration) plus the "
                         "reset-time friction and pushes. Use it on BOTH policies of a "
                         "comparison so they run identical dynamics -- the default-pose term in "
                         "particular shifts a joint-space policy's observation AND its action "
                         "offset.")
parser.add_argument("--dump_sonic_obs", action="store_true",
                    help="Also record the exact 1762-wide encoder input and 994-wide decoder "
                         "input SONIC was fed each step, plus the clip step. The reference blocks "
                         "of the encoder input depend only on clip PHASE, so they can be compared "
                         "against a trace from NVIDIA's own env at the same clip step -- which is "
                         "the only way to validate SonicEnvBridge's reference sourcing (the "
                         "trace-replay oracle bypasses it).")
parser.add_argument("--watch", action="store_true",
                    help="Open the Isaac viewport instead of forcing headless, so you can watch "
                         "the rollout live. The bundle is still recorded. Slower, and the play "
                         "tasks already enable the reference-marker and height-scan debug vis.")
parser.add_argument("--out", default="debug_bundle.npz", help="Output bundle .npz.")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
# Split our flags from hydra's key=value overrides, then hand hydra only its own
# (it reads sys.argv directly) -- same pattern as play.py.
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.watch:
    args_cli.headless = True                  # capture does not need a window
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym                                                    # noqa: E402
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
import warp as wp                                                          # noqa: E402

import robot_rl.tasks  # noqa: F401, E402  (registers the tasks)
# Monkey-patches activate_contact_sensors to traverse NESTED bodies; the tasks put contact
# sensors on paths like ``pelvis_link/left_hip_pitch_link/.*``, which IsaacLab's
# non-recursive default never reaches ("could not find any bodies with contact reporter
# API"). Must be imported after the SimulationApp exists -- same as play.py.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401, E402
from isaaclab.managers import SceneEntityCfg                               # noqa: E402
from isaaclab_rl.rsl_rl import (                                           # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config                   # noqa: E402
from robot_rl.tasks.manager_based.robot_rl import mdp                      # noqa: E402
from rsl_rl.runners import OnPolicyRunner                                  # noqa: E402


_FRAMES = ("pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link")


def _np(x) -> np.ndarray:
    """Read a torch tensor or a warp array as numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _policy_obs(obs):
    """The concatenated policy observation tensor.

    rsl-rl / Isaac Lab may hand back a plain tensor, a dict of obs groups, or a TensorDict
    (which is NOT a ``dict`` subclass), so probe by subscripting rather than by type.
    """
    if hasattr(obs, "ndim") and not hasattr(obs, "keys"):
        return obs
    try:
        return obs["policy"]
    except Exception:                                                     # noqa: BLE001
        pass
    try:
        return next(iter(obs.values()))
    except Exception:                                                     # noqa: BLE001
        return obs


def _obs_term_slice(env, group: str, term: str) -> slice | None:
    """Offset of ``term`` inside the concatenated ``group`` obs vector, or None."""
    om = env.unwrapped.observation_manager
    names = om.active_terms[group]
    dims = [int(np.prod(s)) for s in om.group_obs_term_dim[group]]
    if term not in names:
        return None
    i = names.index(term)
    start = int(sum(dims[:i]))
    return slice(start, start + dims[i])


def _capture_envs(env, terrain_ids: str | None, episodes: int) -> np.ndarray:
    """Env indices to capture: those on the requested terrain tiles, else the first ``episodes``.

    Args:
        env: The unwrapped env (its ``scene.terrain`` supplies ``env_terrain_id``).
        terrain_ids: Comma-separated terrain ids, or None for the leading envs.
        episodes: How many envs to capture when ``terrain_ids`` is None.

    Returns:
        A 1-D int array of env indices.

    Raises:
        SystemExit: If a requested terrain id has no env on it -- silently dropping it would
            produce a bundle that looks complete but is missing a family.
    """
    if terrain_ids is None:
        return np.arange(min(episodes, env.num_envs), dtype=np.int64)
    want = [int(t) for t in terrain_ids.split(",") if t.strip() != ""]
    imp = getattr(env.scene, "terrain", None)
    if imp is None or not hasattr(imp, "env_terrain_id"):
        raise SystemExit("[debug] --capture_terrain_ids needs a ClipTerrainImporter scene terrain.")
    env_tid = _np(imp.env_terrain_id).astype(np.int64)
    picked, missing = [], []
    for t in want:
        hits = np.nonzero(env_tid == t)[0]
        if hits.size == 0:
            missing.append(t)
        else:
            picked.append(int(hits[0]))
    if missing:
        present = sorted(set(env_tid.tolist()))
        hint = "Raise --num_envs so the importer instantiates more tiles, or pick different ids."
        if missing == [-1] and -1 not in present:
            # By far the most common way to hit this: terrain id -1 means flat ground, and the
            # small terrain-only libraries contain no flat clips at all, so no --num_envs will
            # ever produce one.
            hint = ("Terrain id -1 is FLAT ground, and this library appears to contain no flat "
                    "clips -- the small terrain-only libraries (g1_mb_v5_small*.h5) are stairs "
                    "and boxes only. Use the full library (clips/g1_mb_v5.h5) for flat, or "
                    "capture a terrain family here instead, e.g. "
                    "--capture_families family_stairs_up.h5 --per_family 1.")
        raise SystemExit(
            f"[debug] terrain id(s) {missing} have no env in this scene (present: {present}). "
            + hint
        )
    return np.asarray(picked, dtype=np.int64)


def _families_to_terrain_ids(env, cmd, families: str, per_family: int) -> str:
    """Resolve clip-library terrain FAMILIES to terrain ids present in this scene.

    Selecting by family rather than by raw terrain id lets a comparison ask for "two stair-up
    tiles and two jump-on tiles" without knowing how the importer assigned tiles to envs at the
    chosen ``--num_envs``. The selection depends only on the scene, not on the policy, so two
    policies asked for the same families capture the same tiles.

    Args:
        env: The unwrapped env.
        cmd: The tracking command term (its clip library carries the manifest).
        families: Comma-separated family names, or ``"all"``. ``"flat"`` maps to terrain id -1,
            which the manifest does not list (flat tiles carry no geometry entry).
        per_family: Tiles to take per family, lowest terrain id first.

    Returns:
        A comma-separated terrain-id string for :func:`_capture_envs`.

    Raises:
        SystemExit: If a requested family has no tile in this scene, listing what is available.
    """
    lib = getattr(cmd, "_clip_lib", None)
    manifest = lib.attrs.get("terrain_manifest", []) if lib is not None else []
    by_family: dict[str, list[int]] = {}
    for tile in manifest:
        by_family.setdefault(str(tile.get("family", tile.get("type", "?"))), []).append(
            int(tile["terrain_id"]))
    present = set(_np(env.scene.terrain.env_terrain_id).astype(np.int64).tolist())
    if -1 in present:
        by_family["flat"] = [-1]
    want = sorted(by_family) if families.strip() == "all" else \
        [f.strip() for f in families.split(",") if f.strip()]

    picked, missing = [], []
    for fam in want:
        avail = [t for t in sorted(by_family.get(fam, [])) if t in present]
        if not avail:
            missing.append(fam)
        else:
            picked.extend(avail[:max(1, per_family)])
    if missing:
        here = {f: len([t for t in sorted(v) if t in present]) for f, v in by_family.items()}
        raise SystemExit(
            f"[debug] families {missing} have no tile among this scene's {len(present)} "
            f"instantiated tiles. Present per family: {here}. Raise --num_envs so the importer "
            "instantiates more tiles, or ask for different families."
        )
    print(f"[INFO] families {want} -> terrain ids {picked}")
    return ",".join(str(t) for t in picked)


def _yaw_rotate_inv(vec_w: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the yaw-only body frame (heading-invariant)."""
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    out = np.empty_like(vec_w)
    out[:, 0] = c * vec_w[:, 0] - s * vec_w[:, 1]
    out[:, 1] = s * vec_w[:, 0] + c * vec_w[:, 1]
    out[:, 2] = vec_w[:, 2]
    return out


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> int:
    """Roll out the policy and dump a debug bundle.

    Uses the hydra task config (like ``play.py``) rather than ``parse_env_cfg``: the latter
    skips the preset that enables contact reporting, so the task's foot contact sensors fail
    to initialise.
    """
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # Repoint the clip library AND the terrain importer mesh (they must agree, else the
    # robot tracks one tile set's clips on another's geometry -- see record_data.py).
    if args_cli.clip is not None:
        for name, term in vars(env_cfg.commands).items():
            if hasattr(term, "path"):
                term.path = args_cli.clip
                print(f"[INFO] clip path -> commands.{name}")
        terrain = getattr(env_cfg.scene, "terrain", None)
        if terrain is not None and hasattr(terrain, "clip_library_path"):
            terrain.clip_library_path = args_cli.clip
            print("[INFO] clip path -> scene.terrain (importer mesh)")
    if args_cli.clip_device is not None:
        for term in vars(env_cfg.commands).values():
            if hasattr(term, "clip_device"):
                term.clip_device = args_cli.clip_device
    # One clip per episode: start at step 0 so the clip_end_reached truncation lines up with
    # the end of the clip instead of chopping mid-way through a randomly-phased start.
    for name, term in vars(env_cfg.commands).items():
        if hasattr(term, "random_start_phase"):
            term.random_start_phase = bool(args_cli.random_start)
            print(f"[INFO] commands.{name}.random_start_phase = {term.random_start_phase}")

    if args_cli.fm_checkpoint is not None:
        for name, term in vars(env_cfg.commands).items():
            if hasattr(term, "fm_checkpoint"):
                term.fm_checkpoint = args_cli.fm_checkpoint
                print(f"[INFO] fm_checkpoint -> commands.{name}")
    if args_cli.fm_zero_noise:
        for name, term in vars(env_cfg.commands).items():
            if hasattr(term, "fm_zero_noise"):
                term.fm_zero_noise = True
                print(f"[INFO] fm_zero_noise=True -> commands.{name}")

    # Pin the clip choice so the tile -> clip mapping is a property of the library, not of the
    # reset RNG stream (which diverges as soon as two policies terminate at different steps).
    if args_cli.clip_pick is not None:
        reset_term = getattr(env_cfg.events, "reset_on_clip", None)
        if reset_term is None:
            raise SystemExit("[debug] --clip_pick needs an events.reset_on_clip term.")
        reset_term.params["clip_pick"] = int(args_cli.clip_pick)
        print(f"[INFO] clip_pick={args_cli.clip_pick} -> events.reset_on_clip")

    if args_cli.ego_anchor_fraction is not None:
        for name, term in vars(env_cfg.commands).items():
            if hasattr(term, "ego_anchor_fraction"):
                term.ego_anchor_fraction = float(args_cli.ego_anchor_fraction)
                print(f"[INFO] ego_anchor_fraction={args_cli.ego_anchor_fraction} "
                      f"-> commands.{name}")

    if args_cli.no_dr:
        dropped = []
        for name, term in list(vars(env_cfg.events).items()):
            if term is None:
                continue
            if getattr(term, "mode", None) == "startup" or name in (
                "randomize_ground_contact_friction", "push_robot"
            ):
                setattr(env_cfg.events, name, None)
                dropped.append(name)
        print(f"[INFO] --no_dr: dropped events {dropped}")

    # Mirror play.py: apply the CLI overrides + the rsl-rl deprecation shim before building
    # the runner, else the agent cfg carries fields the installed rsl-rl no longer accepts.
    import importlib.metadata as _md
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    if args_cli.policy == "sonic":
        # The SONIC baseline is not an rsl-rl checkpoint: it builds its own observation from the
        # command term and the articulation, so no runner (and no actor-critic) is constructed.
        if args_cli.sonic_dir is None:
            raise SystemExit("[debug] --policy sonic requires --sonic_dir <onnx dir>.")
        from robot_rl.baselines.sonic.env_bridge import SonicEnvBridge
        policy = SonicEnvBridge(env.unwrapped, args_cli.sonic_dir)
    elif args_cli.policy == "zero":
        n_act = env.unwrapped.action_manager.total_action_dim
        _zeros = torch.zeros(env.unwrapped.num_envs, n_act, device=env.unwrapped.device)
        policy = lambda _obs: _zeros                                       # noqa: E731
        print(f"[INFO] zero-action diagnostic: commanding the default pose ({n_act} actions)")
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    robot = u.scene["robot"]
    cmd = u.command_manager.get_term("traj_ref")
    has_scan = "height_scanner" in u.scene.sensors
    scan_sl = _obs_term_slice(u, "policy", "height_scan") if has_scan else None
    cmd_sl = _obs_term_slice(u, "policy", "command_vel")
    body_idx = [robot.body_names.index(f) for f in _FRAMES if f in robot.body_names]
    frames = [f for f in _FRAMES if f in robot.body_names]
    tids = args_cli.capture_terrain_ids
    if args_cli.capture_families:
        tids = _families_to_terrain_ids(u, cmd, args_cli.capture_families, args_cli.per_family)
    cap = _capture_envs(u, tids, args_cli.episodes)
    ncap = len(cap)
    print(f"[INFO] capturing {ncap} envs x {args_cli.steps} steps; envs={cap.tolist()}; "
          f"frames={frames}; scan={'yes' if has_scan else 'no'}")

    # Reference: the world state the CLF tracks, in the clip's output layout.
    pos_names = list(cmd.ordered_pos_output_names)
    ref_idx = {f: [pos_names.index(f"{f}:pos_{a}") for a in "xyz"]
               for f in frames if f"{f}:pos_x" in pos_names}
    is_fm = type(cmd).__name__ == "FmTrackingCommand"

    # Reference joint / base channels, so a joint-space tracker's error can be scored offline
    # (the CLF-facing ``ref_pos`` covers only the pelvis + feet positions).
    joint_names_clip = [n.split(":", 1)[1] for n in pos_names if n.startswith("joint:")]
    ref_jp_idx = [pos_names.index(f"joint:{n}") for n in joint_names_clip]
    base_name = cmd.cfg.base_frame_name
    ref_base_pos_idx = [pos_names.index(f"{base_name}:pos_{a}") for a in "xyz"]
    ref_base_ori_idx = [pos_names.index(f"{base_name}:ori_{a}") for a in "xyzw"]

    term_names = list(u.termination_manager.active_terms)
    buf: dict[str, list] = {k: [] for k in
                            ("cmd_vel", "ach_vel", "scan_obs", "scan_raw", "scan_hits",
                             "root_pos", "root_quat", "joint_pos", "joint_vel", "body_pos",
                             "ref_pos", "ref_joint_pos", "ref_root_pos", "ref_root_quat",
                             "action", "joint_target", "sonic_enc", "sonic_dec", "sonic_clip_step",
                             "done", "terminated", "time_out", "term_flags",
                             "fm_pred", "fm_pred_frames", "fm_pred_idx")}
    obs = env.get_observations()
    if isinstance(obs, tuple):          # rsl-rl returns (obs, extras) in some versions
        obs = obs[0]
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            act = policy(obs)
        obs_np = _np(_policy_obs(obs))[cap]
        buf["cmd_vel"].append(obs_np[:, cmd_sl] if cmd_sl else np.zeros((ncap, 3), "float32"))
        buf["scan_obs"].append(obs_np[:, scan_sl] if scan_sl else np.zeros((ncap, 1), "float32"))
        if has_scan:
            raw = _np(mdp.height_scan(u, SceneEntityCfg("height_scanner")))[cap]
        else:
            raw = np.zeros((ncap, 1), "float32")
        buf["scan_raw"].append(raw)
        if has_scan:
            # World-frame ray HIT POINTS, so the viewer can draw the scan as 3D dots without
            # re-deriving the pattern geometry. Missed rays come back non-finite; the viewer
            # filters them rather than clamping.
            sens = u.scene.sensors["height_scanner"]
            buf["scan_hits"].append(_np(sens.data.ray_hits_w)[cap].astype("float32"))

        rp, rq = _np(robot.data.root_pos_w)[cap], _np(robot.data.root_quat_w)[cap]
        lin = _np(robot.data.root_lin_vel_w)[cap] if hasattr(robot.data, "root_lin_vel_w") \
            else _np(robot.data.root_com_vel_w)[cap][:, :3]
        ang = _np(robot.data.root_ang_vel_w)[cap] if hasattr(robot.data, "root_ang_vel_w") \
            else _np(robot.data.root_com_vel_w)[cap][:, 3:]
        buf["root_pos"].append(rp)
        buf["root_quat"].append(rq)
        buf["ach_vel"].append(np.concatenate(
            [_yaw_rotate_inv(lin, rq), _yaw_rotate_inv(ang, rq)[:, 2:3]], axis=-1))  # vx, vy, vz, wz
        buf["joint_pos"].append(_np(robot.data.joint_pos)[cap])
        buf["joint_vel"].append(_np(robot.data.joint_vel)[cap])
        buf["body_pos"].append(_np(robot.data.body_pos_w)[cap][:, body_idx])
        y = _np(cmd.y_des)[cap]
        buf["ref_pos"].append(np.stack([y[:, ref_idx[f]] for f in frames if f in ref_idx], axis=1))
        buf["ref_joint_pos"].append(y[:, ref_jp_idx])
        buf["ref_root_pos"].append(y[:, ref_base_pos_idx])
        buf["ref_root_quat"].append(y[:, ref_base_ori_idx])
        # The raw action AND the joint target it becomes. With these, a poor tracking result can be
        # split: if the TARGETS follow the reference but the measured joints do not, that is
        # dynamics/gains; if the targets themselves do not follow it, the policy is being fed or
        # is ignoring the reference.
        act_np = _np(act)[cap]
        buf["action"].append(act_np)
        if args_cli.dump_sonic_obs and hasattr(policy, "last_obs"):
            lo = policy.last_obs
            if lo:
                buf["sonic_enc"].append(lo["encoder_input"][cap].astype("float32"))
                buf["sonic_dec"].append(lo["decoder_input"][cap].astype("float32"))
                buf["sonic_clip_step"].append(lo["clip_step"][cap].astype("int32"))
        buf["joint_target"].append(_np(u.action_manager.get_term("joint_pos").processed_actions)[cap])
        if is_fm:
            # The generator's whole predicted WINDOW at this step, root and extra frames, plus
            # the index within it that is currently being applied. Recorded per step (not per
            # re-plan) so a viewer can show exactly what the FM emitted at any step; the window
            # is only refreshed every ``fm_update_every`` steps, so consecutive steps repeat it
            # while ``fm_pred_idx`` walks forward through it.
            pred = getattr(cmd, "_pred_root_pos", None)
            predf = getattr(cmd, "_pred_frame_pos", None)
            pidx = getattr(cmd, "_pred_idx", None)
            buf["fm_pred"].append(_np(pred)[cap] if pred is not None
                                  else np.zeros((ncap, 1, 3), "float32"))
            buf["fm_pred_frames"].append(_np(predf)[cap] if predf is not None
                                         else np.zeros((ncap, 1, 0, 3), "float32"))
            buf["fm_pred_idx"].append(_np(pidx)[cap].astype("int32") if pidx is not None
                                      else np.zeros((ncap,), "int32"))

        with torch.inference_mode():
            obs, _, dones, _ = env.step(act)
        buf["done"].append(_np(dones)[cap].astype("uint8"))
        # Split the done into "fell over" vs "ran out of clip": survival is the headline
        # comparison metric and the two mean opposite things. Keep the PER-TERM flags too, so a
        # bundle says which criterion ended each episode instead of just that one did.
        buf["terminated"].append(_np(u.termination_manager.terminated)[cap].astype("uint8"))
        buf["time_out"].append(_np(u.termination_manager.time_outs)[cap].astype("uint8"))
        buf["term_flags"].append(np.stack(
            [_np(u.termination_manager.get_term(t))[cap].astype("uint8") for t in term_names],
            axis=-1) if term_names else np.zeros((ncap, 0), "uint8"))

    out: dict[str, np.ndarray] = {k: np.asarray(np.stack(v)) for k, v in buf.items() if v}
    # Static context the viewer needs: geometry, joint names, tile assignment, dt.
    lib_manifest, tid, corner = [], np.zeros(ncap, "int32"), np.zeros((ncap, 3), "float32")
    imp = getattr(u.scene, "terrain", None)
    if imp is not None and hasattr(imp, "env_terrain_id"):
        tid = _np(imp.env_terrain_id)[cap].astype("int32")
        corner = _np(imp.env_cell_corner)[cap].astype("float32")
        lib = getattr(cmd, "_clip_lib", None)
        if lib is not None:
            lib_manifest = lib.attrs.get("terrain_manifest", [])
    meta = {
        "dt": float(u.step_dt), "frames": frames, "joint_names": list(robot.joint_names),
        "task": args_cli.task, "is_fm": bool(is_fm), "checkpoint": args_cli.checkpoint,
        "fm_checkpoint": args_cli.fm_checkpoint, "clip": args_cli.clip,
        "fm_zero_noise": bool(args_cli.fm_zero_noise),
        "terrain_manifest": lib_manifest, "num_captured": int(ncap),
        # Order of the K extra frames in ``fm_pred_frames`` (the FM's predicted ee frames).
        "fm_frame_names": (list(cmd.ctx.frame_names) if is_fm and hasattr(cmd, "ctx") else []),
        # Which policy drove the robot, and the provenance a comparison needs to label a bundle.
        "policy": args_cli.policy,
        "sonic_dir": args_cli.sonic_dir,
        "clip_pick": args_cli.clip_pick,
        "no_dr": bool(args_cli.no_dr),
        "ego_anchor_fraction": (float(getattr(cmd.cfg, "ego_anchor_fraction", float("nan")))
                                if hasattr(cmd.cfg, "ego_anchor_fraction") else None),
        "capture_envs": cap.tolist(),
        # ``ref_joint_pos`` is in the CLIP's channel order, which is NOT the articulation's DOF
        # order that ``joint_pos`` uses -- a consumer comparing the two must remap by name.
        "ref_joint_names": joint_names_clip,
        "base_frame_name": base_name,
        # Column order of ``term_flags``.
        "term_names": term_names,
    }
    out["terrain_id"] = tid
    out["cell_corner"] = corner
    # Where the clip itself was placed in the world. This -- not ``cell_corner`` -- is the datum
    # that maps a world trace back into clip coordinates: for a terrain clip it IS the tile
    # corner, but for a flat clip the tile has no corner (``cell_corner`` stays zero) and the
    # placement is the env origin, tens of metres out in the tile grid. Consumers comparing
    # against clip-authored coordinates (e.g. a MuJoCo scene) must subtract this.
    cwp = getattr(cmd, "clip_world_pos", None)
    out["clip_world_pos"] = (_np(cwp)[cap].astype("float32") if cwp is not None
                             else np.zeros((ncap, 3), "float32"))
    out["clip_id"] = (_np(cmd.clip_id)[cap].astype("int32") if hasattr(cmd, "clip_id")
                      else np.zeros(ncap, "int32"))
    out["meta_json"] = np.asarray(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    np.savez_compressed(args_cli.out, **out)
    print(f"[INFO] wrote {args_cli.out}")
    for k, v in out.items():
        if v.ndim >= 2:
            print(f"         {k:12s} {v.shape}")
    env.close()
    return 0


if __name__ == "__main__":
    main()
    simulation_app.close()
