#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage 1 (Motion Bricks conda env): generate raw flat motions and write intermediate ``.npz`` files.

This drives NVIDIA Motion Bricks' ``full_navigation_agent`` headlessly (the same loop as
``scripts/interactive_demo_g1.py`` with no viewer) and records the streamed ``mujoco_qpos`` per clip,
plus a coarse achieved ``(vx, vy, wz)`` command (root planar velocity + yaw-rate in the pelvis-yaw
frame). Each clip is written as a :class:`RawMotion` ``.npz`` -- the only contract with Stage 2
(``motion_gen.motionbricks.mb_ingest``), which resamples + converts these into the clip-library ``.h5``.

Runs in the ``motionbricks`` conda env (imports the external ``motionbricks`` package); it does NOT
import ``motion_gen`` (which lives in ``env_isaaclab``). The agent construction + the numpy-only raw
schema come from the shared Stage-1 harness :mod:`mb_env`, so both stages share one definition without
coupling the environments. Example::

    conda run -n motionbricks python scripts/rsl_rl/mb_generate.py \
        --mb-root ../GR00T-WholeBodyControl/motionbricks \
        --out-dir clips/mb_raw/flat_v1 --n-clips 8 --clip-length 9.0 --controller random
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_env  # noqa: E402


def _yaw_from_wxyz(quat: np.ndarray) -> np.ndarray:
    """Z-up yaw angle ``[T]`` from wxyz quaternions ``[T, 4]``."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _achieved_command(qpos: np.ndarray, fps: float) -> np.ndarray:
    """Coarse achieved ``(vx, vy, wz)`` command ``[T, 3]`` from a qpos stream.

    ``vx/vy`` are the world planar root velocity rotated into the pelvis-yaw frame (forward/left);
    ``wz`` is the yaw-rate. Central differences, edge-replicated, so the length matches ``qpos``.
    """
    root_xy = qpos[:, 0:2]
    yaw = _yaw_from_wxyz(qpos[:, 3:7])
    vel_xy = np.gradient(root_xy, axis=0) * fps                     # world-frame planar velocity
    c, s = np.cos(-yaw), np.sin(-yaw)
    vx = c * vel_xy[:, 0] - s * vel_xy[:, 1]
    vy = s * vel_xy[:, 0] + c * vel_xy[:, 1]
    yaw_unwrapped = np.unwrap(yaw)
    wz = np.gradient(yaw_unwrapped) * fps
    return np.stack([vx, vy, wz], axis=-1).astype(np.float32)


def _generate_one(demo_agent, n_steps: int, seed: int) -> np.ndarray:
    """Run the headless generation loop for ``n_steps`` and return the collected ``qpos`` ``[n_steps, 36]``."""
    np.random.seed(seed)
    t.manual_seed(seed)
    demo_agent.full_agent.reset()
    controller_dt = demo_agent.controller.get_controller_dt()
    qpos_stream = []
    for step in range(n_steps):
        force_idle = step + 60 > n_steps                             # settle to a stand near the end
        qpos = demo_agent.full_agent.get_next_frame()
        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos
        control_signals = demo_agent.controller.generate_control_signals(
            None, demo_agent.mj_model, demo_agent.mj_data, visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": getattr(demo_agent.args, "allowed_mode", None)},
        )
        control_signals["context_mujoco_qpos"] = context_mujoco_qpos
        with t.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals, controller_dt * demo_agent.args.generate_dt)
        import mujoco
        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
        qpos_stream.append(np.asarray(qpos, dtype=np.float32).copy())
    return np.stack(qpos_stream, axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mb-root", required=True, help="Path to the GR00T motionbricks package root (has out/, assets/).")
    p.add_argument("--out-dir", required=True, help="Directory to write raw motion .npz files (one per clip).")
    p.add_argument("--n-clips", type=int, default=8, help="Number of clips to generate.")
    p.add_argument("--clip-length", type=float, default=9.0, help="Clip duration (s) at the Motion Bricks fps.")
    p.add_argument("--controller", default="random", choices=["random", "wasd"], help="Motion Bricks controller.")
    p.add_argument("--allowed-mode", default=None, help="Restrict the controller to a single style mode.")
    p.add_argument("--seed", type=int, default=1234, help="Base RNG seed (per-clip = seed * (i+2333) * 2333).")
    args = p.parse_args()

    from motionbricks.motion_backbone.demo.utils import navigation_demo  # noqa: import in MB env only

    mb_env.enter_mb_root(args, ("out_dir",))
    raw_schema = mb_env.load_schema_module("raw_schema")
    os.makedirs(args.out_dir, exist_ok=True)

    # Free-run: the MB controller drives the model, so this path deliberately does NOT install the pinning
    # hooks of ``mb_env.MBAgent`` -- only the demo-args construction is shared.
    demo_agent = navigation_demo(mb_env.build_demo_args(args))
    fps = float(demo_agent.full_agent._fps)
    n_steps = max(2, int(round(args.clip_length * fps)))
    print(f"[mb_generate] fps={fps} n_steps={n_steps} controller={args.controller}")

    for i in range(args.n_clips):
        seed = args.seed * (i + 2333) * 2333 % (2 ** 32 - 1)
        qpos = _generate_one(demo_agent, n_steps, seed)
        target_vel = _achieved_command(qpos, fps)
        motion = raw_schema.RawMotion(
            mujoco_qpos=qpos, fps=fps,
            joint_names=list(mb_env.G1_JOINT_NAMES),
            target_vel=target_vel, kind="flat",
            provenance={"backend": "motionbricks", "controller": args.controller,
                        "seed": int(seed), "mb_root": os.path.abspath(args.mb_root),
                        "clip_index": i},
        )
        out = os.path.join(args.out_dir, f"flat_{i:04d}.npz")
        raw_schema.save_raw_motion(out, motion)
        print(f"[mb_generate]   wrote {out}  qpos={qpos.shape} pelvis_z~{qpos[:,2].mean():.3f}")
    print(f"[mb_generate] done: {args.n_clips} clips -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
