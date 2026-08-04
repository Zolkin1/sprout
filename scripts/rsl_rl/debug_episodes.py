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
parser.add_argument("--out", default="debug_bundle.npz", help="Output bundle .npz.")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
# Split our flags from hydra's key=value overrides, then hand hydra only its own
# (it reads sys.argv directly) -- same pattern as play.py.
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True                      # capture never needs a window
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
from robot_rl.flow_matching.window import window_to_world                  # noqa: E402
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

    # Mirror play.py: apply the CLI overrides + the rsl-rl deprecation shim before building
    # the runner, else the agent cfg carries fields the installed rsl-rl no longer accepts.
    import importlib.metadata as _md
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Mirror play.py: a runner cfg can opt out of restoring the value function
    # (``load_critic = False``), and the FM TRACKER does -- it has no privileged observation
    # group, so the critic it builds does not match the one the teacher trained. Loading
    # strictly here failed with a bare size mismatch on mlp.0 (1834 vs 2490) that reads like
    # a policy/env mismatch and is not one: the ACTOR matches exactly. Actor weights stay
    # strict, so a real mismatch still fails loudly.
    if getattr(agent_cfg, "load_critic", True):
        runner.load(args_cli.checkpoint)
    else:
        print("[INFO] Skipping the critic (play-only runner): loading actor weights only.")
        runner.load(args_cli.checkpoint, load_cfg={"actor": True, "critic": False,
                                                   "optimizer": False, "iteration": True,
                                                   "rnd": True})
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    robot = u.scene["robot"]
    cmd = u.command_manager.get_term("traj_ref")
    has_scan = "height_scanner" in u.scene.sensors
    # Depth is this policy's exteroception, and it was the one signal no trace recorded on
    # either side -- so a sim2sim divergence could never be attributed to what the two
    # policies SAW. Captured normalised (mdp.depth_obs_stack), i.e. exactly the observation,
    # so it lines up with the MuJoCo runner's <log>.depth.npy without a unit conversion.
    has_depth = any(n.startswith("depth_camera") for n in u.scene.sensors)
    scan_sl = _obs_term_slice(u, "policy", "height_scan") if has_scan else None
    cmd_sl = _obs_term_slice(u, "policy", "command_vel")
    body_idx = [robot.body_names.index(f) for f in _FRAMES if f in robot.body_names]
    frames = [f for f in _FRAMES if f in robot.body_names]
    ncap = min(args_cli.episodes, u.num_envs)
    print(f"[INFO] capturing {ncap} envs x {args_cli.steps} steps; frames={frames}; "
          f"scan={'yes' if has_scan else 'no'} depth={'yes' if has_depth else 'no'}")

    # Reference: the world state the CLF tracks, in the clip's output layout.
    pos_names = list(cmd.ordered_pos_output_names)
    ref_idx = {f: [pos_names.index(f"{f}:pos_{a}") for a in "xyz"]
               for f in frames if f"{f}:pos_x" in pos_names}
    is_fm = type(cmd).__name__ == "FmTrackingCommand"

    buf: dict[str, list] = {k: [] for k in
                            ("cmd_vel", "ach_vel", "scan_obs", "scan_raw", "scan_hits",
                             "depth", "ctrl", "torque",
                             "root_pos", "root_quat", "joint_pos", "body_pos", "ref_pos",
                             "done", "fm_pred", "fm_pred_quat", "fm_pred_frames",
                             "fm_pred_idx", "fm_anchor_pos", "fm_anchor_quat")}
    obs = env.get_observations()
    if isinstance(obs, tuple):          # rsl-rl returns (obs, extras) in some versions
        obs = obs[0]
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            act = policy(obs)
        obs_np = _np(_policy_obs(obs))[:ncap]
        buf["cmd_vel"].append(obs_np[:, cmd_sl] if cmd_sl else np.zeros((ncap, 3), "float32"))
        buf["scan_obs"].append(obs_np[:, scan_sl] if scan_sl else np.zeros((ncap, 1), "float32"))
        if has_scan:
            raw = _np(mdp.height_scan(u, SceneEntityCfg("height_scanner")))[:ncap]
        else:
            raw = np.zeros((ncap, 1), "float32")
        buf["scan_raw"].append(raw)
        if has_depth:
            buf["depth"].append(_np(mdp.depth_obs_stack(u))[:ncap].astype("float32"))
        if has_scan:
            # World-frame ray HIT POINTS, so the viewer can draw the scan as 3D dots without
            # re-deriving the pattern geometry. Missed rays come back non-finite; the viewer
            # filters them rather than clamping.
            sens = u.scene.sensors["height_scanner"]
            buf["scan_hits"].append(_np(sens.data.ray_hits_w)[:ncap].astype("float32"))

        rp, rq = _np(robot.data.root_pos_w)[:ncap], _np(robot.data.root_quat_w)[:ncap]
        lin = _np(robot.data.root_lin_vel_w)[:ncap] if hasattr(robot.data, "root_lin_vel_w") \
            else _np(robot.data.root_com_vel_w)[:ncap, :3]
        ang = _np(robot.data.root_ang_vel_w)[:ncap] if hasattr(robot.data, "root_ang_vel_w") \
            else _np(robot.data.root_com_vel_w)[:ncap, 3:]
        buf["root_pos"].append(rp)
        buf["root_quat"].append(rq)
        buf["ach_vel"].append(np.concatenate(
            [_yaw_rotate_inv(lin, rq), _yaw_rotate_inv(ang, rq)[:, 2:3]], axis=-1))  # vx, vy, vz, wz
        buf["joint_pos"].append(_np(robot.data.joint_pos)[:ncap])
        # The COMMANDED joint target and the torque the actuators actually applied -- the
        # two signals that make a cross-simulator comparison possible at all. Without them a
        # divergence can only be seen in the OUTCOME, never attributed to the policy asking
        # for something different or to the actuator delivering something different.
        # ``processed_actions`` is the action after scale + default-pose offset, i.e. the
        # same quantity MuJoCo logs as ``ctrl_<joint>``.
        try:
            term = u.action_manager.get_term(u.action_manager.active_terms[0])
            buf["ctrl"].append(_np(term.processed_actions)[:ncap].astype("float32"))
        except Exception:                                                 # noqa: BLE001
            pass
        try:
            buf["torque"].append(_np(robot.data.applied_torque)[:ncap].astype("float32"))
        except Exception:                                                 # noqa: BLE001
            pass
        buf["body_pos"].append(_np(robot.data.body_pos_w)[:ncap][:, body_idx])
        y = _np(cmd.y_des)[:ncap]
        buf["ref_pos"].append(np.stack([y[:, ref_idx[f]] for f in frames if f in ref_idx], axis=1))
        if is_fm:
            # The generator's whole predicted WINDOW at this step, root and extra frames, plus
            # the index within it that is currently being applied. Recorded per step (not per
            # re-plan) so a viewer can show exactly what the FM emitted at any step; the window
            # is only refreshed every ``fm_update_every`` steps, so consecutive steps repeat it
            # while ``fm_pred_idx`` walks forward through it.
            # Lift the WHOLE window through its OWN stored re-plan anchor, exactly as the
            # debug ribbon does (FmTrackingCommand._debug_vis_callback). The anchor is the
            # robot's FULL orientation at the re-plan, so the window it encodes pitches and
            # rolls with the torso; using the CURRENT pose here instead would shear the
            # result by the egomotion since that re-plan whenever fm_update_every > 1.
            #
            # This used to read ``_pred_root_pos``/``_pred_frame_pos``, which the
            # reference-window refactor deleted. Both were fetched with a defaulting
            # ``getattr``, so the capture silently degraded to a zero-filled [ncap, 1, 3]
            # that still LOOKED like a prediction. Reach for the live attributes instead and
            # let a rename fail loudly.
            world = window_to_world(
                cmd._pred_window[:ncap], cmd._anchor_pos[:ncap, None, :],
                cmd._anchor_quat[:ncap, None, :], cmd.ctx.nj,
                cmd.ctx.num_extra_frames, cmd._fm_blocks,
            )
            buf["fm_pred"].append(_np(world["base_pos_w"]))                    # [ncap, F, 3]
            buf["fm_pred_quat"].append(_np(world["base_quat_w"])
                                       if "base_quat_w" in world
                                       else np.zeros((ncap, cmd.ctx.F, 4), "float32"))
            buf["fm_pred_frames"].append(_np(world["ee_pos_w"]) if "ee_pos_w" in world
                                         else np.zeros((ncap, cmd.ctx.F, 0, 3), "float32"))
            buf["fm_pred_idx"].append(_np(cmd._pred_idx)[:ncap].astype("int32"))
            buf["fm_anchor_pos"].append(_np(cmd._anchor_pos)[:ncap])
            buf["fm_anchor_quat"].append(_np(cmd._anchor_quat)[:ncap])

        with torch.inference_mode():
            obs, _, dones, _ = env.step(act)
        buf["done"].append(_np(dones)[:ncap].astype("uint8"))

    out: dict[str, np.ndarray] = {k: np.asarray(np.stack(v)) for k, v in buf.items() if v}
    # Static context the viewer needs: geometry, joint names, tile assignment, dt.
    lib_manifest, tid, corner = [], np.zeros(ncap, "int32"), np.zeros((ncap, 3), "float32")
    imp = getattr(u.scene, "terrain", None)
    if imp is not None and hasattr(imp, "env_terrain_id"):
        tid = _np(imp.env_terrain_id)[:ncap].astype("int32")
        corner = _np(imp.env_cell_corner)[:ncap].astype("float32")
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
    }
    out["terrain_id"] = tid
    out["cell_corner"] = corner
    # Where the clip itself was placed in the world. This -- not ``cell_corner`` -- is the datum
    # that maps a world trace back into clip coordinates: for a terrain clip it IS the tile
    # corner, but for a flat clip the tile has no corner (``cell_corner`` stays zero) and the
    # placement is the env origin, tens of metres out in the tile grid. Consumers comparing
    # against clip-authored coordinates (e.g. a MuJoCo scene) must subtract this.
    cwp = getattr(cmd, "clip_world_pos", None)
    out["clip_world_pos"] = (_np(cwp)[:ncap].astype("float32") if cwp is not None
                             else np.zeros((ncap, 3), "float32"))
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
