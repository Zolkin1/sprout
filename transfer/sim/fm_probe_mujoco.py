#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Dump FM inputs/outputs for clip-derived probe states in MuJoCo.

Counterpart of ``scripts/flow_matching/fm_probe_isaac.py``: drives the robot to a
set of identical, clip-derived states (no rollout drift) and records, per probe,
the measured frame-origin twists the FM history consumes, the assembled history
token, and the ONNX FM's prediction (run with a shared fixed noise). Compare the
two dumps with ``fm_probe_compare.py`` to localize the clip->FM sim2sim
divergence to FM *feeding* (an input mismatch) vs the *dynamics gap*.

Run from ``transfer/sim/`` (MuJoCo assets resolve relative to cwd)::

    python fm_probe_mujoco.py \
        --fm_onnx ../../logs/rsl_rl/g1/<run>/exported/fm.onnx \
        --clip ../../clips/g1_parity_v1.h5 --out /tmp/fm_probe_mujoco.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # transfer/ on path
from deploy.fm_session import build_history_token, load_fm_context  # noqa: E402
from deploy.quat_math_np import quat_apply_inverse  # noqa: E402
from deploy.tracker_session import convert_joint_order  # noqa: E402

from robot import Robot  # noqa: E402
from clip_reference import ClipReference  # noqa: E402


def main() -> int:
    """Set clip-derived probe states, measure FM inputs, run the FM, dump an npz."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fm_onnx", required=True, help="fm.onnx (defines joint/frame order, H/F/dims).")
    p.add_argument("--clip", required=True, help="Parity clip .h5 the probe states come from.")
    p.add_argument("--steps", default="50,160,300,470,540",
                   help="Comma-separated clip steps to probe (stand/walk-ramp/walk/run-ramp/run).")
    p.add_argument("--out", default="/tmp/fm_probe_mujoco.npz")
    p.add_argument("--seed", type=int, default=0, help="Seed for the shared sampling noise.")
    p.add_argument("--robot_name", default="g1_21j")
    p.add_argument("--scene", default="basic_scene")
    args = p.parse_args()

    ctx = load_fm_context(args.fm_onnx)
    H, F, Din, D = ctx.H, ctx.F, ctx.Din, ctx.D
    hist_frames = list(ctx.history_frame_names)
    K = len(hist_frames)
    joints = list(ctx.joint_names)

    robot = Robot(robot_name=args.robot_name, scene_name=args.scene,
                  joystick_scaling=np.ones(3), input_function=lambda t: np.zeros(3))
    clip = ClipReference(args.clip, joints, hist_frames, base_frame_name="pelvis_link")
    steps = [int(s) for s in args.steps.split(",") if s != ""]
    noise = np.random.default_rng(args.seed).standard_normal((1, F, D)).astype(np.float32)

    def set_state(st: dict) -> None:
        """Write a clip reference state into qpos/qvel (free-joint conventions) + FK."""
        q = st["root_quat"]  # xyzw
        robot.mj_data.qpos[:3] = st["root_pos"]
        robot.mj_data.qpos[3:7] = [q[3], q[0], q[1], q[2]]  # xyzw -> wxyz
        robot.mj_data.qpos[7:] = convert_joint_order(st["jp"], joints, robot.joint_names)
        robot.mj_data.qvel[:] = 0.0
        robot.mj_data.qvel[0:3] = st["root_lin"]                       # free-joint linear is world
        robot.mj_data.qvel[3:6] = quat_apply_inverse(q, st["root_ang"])  # free-joint angular is body-local
        robot.mj_data.qvel[6:] = convert_joint_order(st["jv"], joints, robot.joint_names)
        mujoco.mj_forward(robot.mj_model, robot.mj_data)

    out: dict[str, list] = {k: [] for k in (
        "cur_root_lin", "cur_root_ang", "cur_ee_pos", "cur_ee_quat", "cur_ee_lin", "cur_ee_ang",
        "token", "pred", "future_cmd")}
    set_err = 0.0
    for s in steps:
        window = [max(0, s - H + 1 + i) for i in range(H)]  # H steps, clamped at clip start
        rp, rq = np.zeros((1, H, 3)), np.zeros((1, H, 4))
        rl, ra = np.zeros((1, H, 3)), np.zeros((1, H, 3))
        jp, jv = np.zeros((1, H, ctx.nj)), np.zeros((1, H, ctx.nj))
        cmd = np.zeros((1, H, ctx.cmd_dim))
        frames = np.zeros((1, H, K, 13))
        for i, j in enumerate(window):
            st = clip.state_at(j)
            set_state(st)
            lin_w, ang_w = robot.get_base_world_velocity("pelvis_link")
            ee = robot.get_body_world_states(hist_frames)  # [K,13] pos,quat(xyzw),lin,ang
            set_err = max(set_err, float(np.abs(lin_w - st["root_lin"]).max()),
                          float(np.abs(ang_w - st["root_ang"]).max()))
            rp[0, i], rq[0, i] = st["root_pos"], st["root_quat"]
            rl[0, i], ra[0, i] = lin_w, ang_w
            jp[0, i], jv[0, i] = st["jp"], st["jv"]
            cmd[0, i] = clip.command_at(j)
            frames[0, i] = ee
        token = build_history_token(ctx, rp, rq, rl, ra, jp, jv, cmd, frames=frames)[0]  # (H,Din)
        fut = clip.future_command_horizon(s, F)[None].astype(np.float32)               # (1,F,cmd)
        feed = {"history_token": token[None].astype(np.float32), "future_cmd": fut, "noise": noise}
        pred = ctx.sess.run(["trajectory"], feed)[0][0]  # (F,D) denormalized relative state

        ee_cur = frames[0, -1]  # measured ee at the current step
        out["cur_root_lin"].append(rl[0, -1]); out["cur_root_ang"].append(ra[0, -1])
        out["cur_ee_pos"].append(ee_cur[:, 0:3]); out["cur_ee_quat"].append(ee_cur[:, 3:7])
        out["cur_ee_lin"].append(ee_cur[:, 7:10]); out["cur_ee_ang"].append(ee_cur[:, 10:13])
        out["token"].append(token); out["pred"].append(pred); out["future_cmd"].append(fut[0])

    np.savez(
        args.out,
        sim="mujoco", steps=np.array(steps), joint_names=np.array(joints),
        ee_frame_names=np.array(hist_frames), H=H, F=F, Din=Din, D=D, nj=ctx.nj,
        cmd_dim=ctx.cmd_dim, seed=args.seed, noise=noise,
        **{k: np.asarray(v) for k, v in out.items()},
    )
    print(f"[fm_probe_mujoco] wrote {args.out}  (probes={len(steps)}, set-state read-back err={set_err:.2e})")
    if set_err > 1e-4:
        print(f"[WARN] set-state read-back err {set_err:.2e} > 1e-4: the free-joint qvel convention "
              "may be off; the probe states may not be identical across sims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
