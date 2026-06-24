#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Dump FM inputs/outputs for clip-derived probe states in IsaacLab.

Counterpart of ``transfer/sim/fm_probe_mujoco.py``: drives the G1 articulation to
the SAME clip-derived probe states and records, per probe, the measured
frame-origin twists the FM history consumes (Isaac ``root_link_*`` / ``body_link_*``),
the assembled RAW history token, and the torch FM's prediction (run with a shared
fixed noise). Compare the two dumps with ``fm_probe_compare.py`` to localize the
clip->FM sim2sim divergence to FM feeding vs the dynamics gap.

Runs headless with gravity disabled (so a single kinematic settle step does not
perturb the velocities being read). Usage::

    python scripts/flow_matching/fm_probe_isaac.py \
        --ckpt logs/flow_matching/<run>/ckpt_<n>.pt \
        --clip clips/g1_parity_v1.h5 --out /tmp/fm_probe_isaac.npz --headless
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

# -- CLI + app launch (must precede any isaaclab/omni imports) ---------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", required=True, help="FM .pt checkpoint (defines joints/frames/dims).")
parser.add_argument("--clip", required=True, help="Parity clip .h5 the probe states come from.")
parser.add_argument("--task", default="G1-tracker-clf-parity-play")
parser.add_argument("--steps", default="50,160,300,470,540")
parser.add_argument("--out", default="/tmp/fm_probe_isaac.npz")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402  (robot.data.* are warp arrays in this fork)

import isaaclab_tasks  # noqa: F401,E402  (registers tasks)
import robot_rl  # noqa: F401,E402       (registers G1 tasks)
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402  (recursive contact sensors; after app)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from robot_rl.flow_matching.online import load_fm_context  # noqa: E402
from robot_rl.flow_matching.transforms import relative_state  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))         # deploy (for ClipReference's quat_math_np)
sys.path.insert(0, os.path.join(ROOT, "transfer", "sim"))  # ClipReference
sys.path.insert(0, os.path.join(ROOT, "scripts", "flow_matching"))
from clip_reference import ClipReference  # noqa: E402
from export_fm_onnx import _FmSamplerOnnx  # noqa: E402


def main() -> int:
    """Set clip-derived probe states in IsaacLab, read FM inputs, run the FM, dump npz."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = load_fm_context(args.ckpt, device=device, scan_dim=0)
    H, F, Din, D, nj = ctx.H, ctx.F, ctx.Din, ctx.D, ctx.nj
    hist_frames = list(ctx.history_frame_names)
    K = len(hist_frames)
    joints = list(ctx.joint_names)
    wrap = _FmSamplerOnnx(ctx).to(device).eval()

    # Build the parity env (assets + articulation); gravity off for a clean read.
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env_cfg.commands.traj_ref.fm_checkpoint = args.ckpt
    env_cfg.commands.traj_ref.path = args.clip
    env_cfg.commands.traj_ref.debug_vis = False  # no viz markers (headless probe)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    asset_joints = list(robot.data.joint_names)
    asset_bodies = list(robot.data.body_names)
    j_ctx2asset = [joints.index(n) for n in asset_joints]   # asset[a] <- ctx[j_ctx2asset[a]]
    ee_body_idx = [asset_bodies.index(f) for f in hist_frames]
    sim_dt = env.sim.get_physics_dt()
    eids = torch.tensor([0], device=device)

    clip = ClipReference(args.clip, joints, hist_frames, base_frame_name="pelvis_link")
    steps = [int(s) for s in args.steps.split(",") if s != ""]
    noise = np.random.default_rng(args.seed).standard_normal((1, F, D)).astype(np.float32)
    noise_t = torch.tensor(noise, device=device)

    def set_and_measure(st: dict):
        """Write a clip reference state, kinematically settle, return measured twists."""
        q = np.asarray(st["root_quat"], dtype=np.float32)            # xyzw (this fork)
        pose = torch.tensor(np.concatenate([st["root_pos"], q]), dtype=torch.float32, device=device)[None]
        vel = torch.tensor(np.concatenate([st["root_lin"], st["root_ang"]]),
                           dtype=torch.float32, device=device)[None]
        jp = torch.tensor(np.asarray(st["jp"], dtype=np.float32)[j_ctx2asset], device=device)[None]
        jv = torch.tensor(np.asarray(st["jv"], dtype=np.float32)[j_ctx2asset], device=device)[None]
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=eids)
        robot.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=eids)
        robot.write_joint_position_to_sim_index(position=jp, env_ids=eids)
        robot.write_joint_velocity_to_sim_index(velocity=jv, env_ids=eids)
        robot.write_data_to_sim()
        # Step-free kinematic propagation: a dynamics step integrates contacts/Coriolis
        # and pollutes the velocities being read. forward() updates the kinematic buffers
        # (link transforms + velocities from the written root+joint state) without it.
        env.sim.forward()
        robot.update(sim_dt)
        d = robot.data
        rl = wp.to_torch(d.root_link_lin_vel_w)[0].cpu().numpy()
        ra = wp.to_torch(d.root_link_ang_vel_w)[0].cpu().numpy()
        ee = np.zeros((K, 13))
        ee[:, 0:3] = wp.to_torch(d.body_link_pos_w)[0, ee_body_idx].cpu().numpy()
        ee[:, 3:7] = wp.to_torch(d.body_link_quat_w)[0, ee_body_idx].cpu().numpy()   # xyzw (this fork)
        ee[:, 7:10] = wp.to_torch(d.body_link_lin_vel_w)[0, ee_body_idx].cpu().numpy()
        ee[:, 10:13] = wp.to_torch(d.body_link_ang_vel_w)[0, ee_body_idx].cpu().numpy()
        return rl, ra, ee

    out: dict[str, list] = {k: [] for k in (
        "cur_root_lin", "cur_root_ang", "cur_ee_pos", "cur_ee_quat", "cur_ee_lin", "cur_ee_ang",
        "token", "pred", "future_cmd")}
    set_err = 0.0
    for s in steps:
        window = [max(0, s - H + 1 + i) for i in range(H)]
        rp, rq = np.zeros((1, H, 3)), np.zeros((1, H, 4))
        rl, ra = np.zeros((1, H, 3)), np.zeros((1, H, 3))
        jp, jv = np.zeros((1, H, nj)), np.zeros((1, H, nj))
        cmd = np.zeros((1, H, ctx.cmd_dim))
        frames = np.zeros((1, H, K, 13))
        for i, j in enumerate(window):
            st = clip.state_at(j)
            mrl, mra, mee = set_and_measure(st)
            set_err = max(set_err, float(np.abs(mrl - st["root_lin"]).max()),
                          float(np.abs(mra - st["root_ang"]).max()))
            rp[0, i], rq[0, i] = st["root_pos"], st["root_quat"]
            rl[0, i], ra[0, i] = mrl, mra
            jp[0, i], jv[0, i] = st["jp"], st["jv"]
            cmd[0, i] = clip.command_at(j)
            frames[0, i] = mee

        # RAW token (matches the deploy raw token; the FM wrapper normalizes internally).
        def tt(a):
            return torch.tensor(np.asarray(a, dtype=np.float32), device=device)
        m = H  # V=1
        rp_t, rq_t, rl_t, ra_t = tt(rp), tt(rq), tt(rl), tt(ra)
        jp_t, jv_t, cmd_t, fr = tt(jp), tt(jv), tt(cmd), tt(frames).reshape(m, K, 13)
        p_ref = rp_t[:, -1:].expand(1, H, 3).reshape(m, 3)
        q_ref = rq_t[:, -1:].expand(1, H, 4).reshape(m, 4)
        rel = relative_state(
            rp_t.reshape(m, 3), rq_t.reshape(m, 4), rl_t.reshape(m, 3), ra_t.reshape(m, 3),
            jp_t.reshape(m, nj), jv_t.reshape(m, nj), p_ref, q_ref,
            frame_pos_w=fr[..., 0:3], frame_quat_w=fr[..., 3:7],
            frame_lin_vel_w=fr[..., 7:10], frame_ang_vel_w=fr[..., 10:13],
        ).reshape(1, H, ctx.history_D)
        raw_token = torch.cat([rel, cmd_t], dim=-1)  # (1,H,Din)
        fut = clip.future_command_horizon(s, F).astype(np.float32)  # (F,cmd)
        with torch.no_grad():
            pred = wrap(raw_token, torch.tensor(fut, device=device)[None], noise_t).cpu().numpy()[0]

        out["cur_root_lin"].append(rl[0, -1]); out["cur_root_ang"].append(ra[0, -1])
        out["cur_ee_pos"].append(frames[0, -1, :, 0:3]); out["cur_ee_quat"].append(frames[0, -1, :, 3:7])
        out["cur_ee_lin"].append(frames[0, -1, :, 7:10]); out["cur_ee_ang"].append(frames[0, -1, :, 10:13])
        out["token"].append(raw_token[0].cpu().numpy()); out["pred"].append(pred)
        out["future_cmd"].append(fut)

    np.savez(
        args.out,
        sim="isaac", steps=np.array(steps), joint_names=np.array(joints),
        ee_frame_names=np.array(hist_frames), H=H, F=F, Din=Din, D=D, nj=nj,
        cmd_dim=ctx.cmd_dim, seed=args.seed, noise=noise,
        **{k: np.asarray(v) for k, v in out.items()},
    )
    print(f"[fm_probe_isaac] wrote {args.out}  (probes={len(steps)}, set-state read-back err={set_err:.2e})")
    if set_err > 1e-3:
        print(f"[WARN] read-back err {set_err:.2e} > 1e-3: the kinematic settle step perturbed the "
              "state; consider reading without a step or disabling contacts.")
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    main()
