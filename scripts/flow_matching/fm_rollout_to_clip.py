# Copyright 2026 Zachary Olkin. All rights reserved.

"""Roll out the clip-conditioned FM autoregressively and write it as a motion clip.

The clip-conditioned FM (``clip_to_fm_data.py`` + ``train.py``) is, by design, NOT
conditioned on the robot's state -- its reference is a deterministic function of
(seed history, velocity-command sequence). So its autoregressive output can be
precomputed offline and stored as an ordinary clip ``.h5``; that clip is identical
to what an online FM would emit. Feeding it to the ego teacher then verifies the
"teacher tracks the FM" loop with the EXISTING playback, in both sims:

    IsaacLab:  play.py --task G1-teacher-clf-play env.commands.traj_ref.path=<out.h5>
    MuJoCo:    transfer/sim/gen_tracker_runner.py --clip <out.h5>   (no FM needed)

Per control step: condition on the N most-recent reference states + the strided
command horizon, sample the F-step window, take the first novel step (index N),
append it to the rolling history, and record it -- a fully autoregressive rollout
(the bench ``rollout`` loop, but persisted as a clip). The world-frame predictions
are scattered into the source library's ``pos_output_names`` / ``vel_output_names``
layout (clip-world == world; identity placement), so the output drops into the
teacher's ``MotionSequenceCommand`` unchanged.

Torch + h5py + numpy only -- uses the vendored ``transfer/fm`` inference (no IsaacLab).

Example:
    python scripts/flow_matching/fm_rollout_to_clip.py \
        --fm_checkpoint logs/flow_matching/<run>/best.pt \
        --source_clip clips/g1_flat_v1.h5 --seed_data clips/g1_flat_v1_fm.npz \
        --out clips/g1_fm_walk.h5 --steps 400 --command 1.0,0,0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "transfer"))      # vendored torch-only fm inference
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "rsl_rl"))
from fm.online import build_cmd_tokens, build_cond_states, load_fm_context, sample_world  # noqa: E402
from recorded_data_io import load_recorded_data  # noqa: E402

_A3, _A4 = ("x", "y", "z"), ("x", "y", "z", "w")


def _cols(names: list[str], frame: str, comps, kind: str) -> list[int]:
    return [names.index(f"{frame}:{kind}_{c}") for c in comps]


def _seed_history(seed_npz: str, episode: int, N: int, ctx) -> list[dict]:
    """Extract an N-step seed history (FM format) from a clip-FM recording episode.

    The recording is in the FM state layout (``clip_to_fm_data.py``), with joints in
    ``state_joint_names`` == ``ctx.joint_names`` order and frames in
    ``state_frame_names`` == ``ctx.target_frame_names`` order, so no permutation is
    needed.
    """
    data, meta = load_recorded_data(seed_npz)
    if list(meta.state_joint_names) != list(ctx.joint_names):
        raise ValueError("seed_data joint order != FM joint order; mismatched recording.")
    sl = {}
    off = 0
    for n, d in zip(meta.state_term_names, meta.state_term_dims):
        sl[n] = (off, off + d); off += d
    rows = np.nonzero(data["episode_id"] == episode)[0]
    if len(rows) < N:
        raise ValueError(f"episode {episode} has {len(rows)} < N={N} steps.")
    rows = rows[:N]
    st = data["state"][rows].astype(np.float64)  # [N, state_dim]

    def col(name):
        a, b = sl[name]; return st[:, a:b]
    frames_per = []
    for f in meta.state_frame_names:
        frames_per.append(np.concatenate(
            [col(f"{f}:pos_w"), col(f"{f}:quat_w"), col(f"{f}:lin_vel_w"), col(f"{f}:ang_vel_w")], axis=-1))
    frames = np.stack(frames_per, axis=1) if frames_per else None  # [N, K, 13]
    hist = []
    for i in range(N):
        step = {
            "root_pos": torch.tensor(col("root_pos_w")[i]).float()[None],
            "root_quat": torch.tensor(col("root_quat_w")[i]).float()[None],
            "root_lin": torch.tensor(col("root_link_lin_vel_w")[i]).float()[None],
            "root_ang": torch.tensor(col("root_link_ang_vel_w")[i]).float()[None],
            "jp": torch.tensor(col("joint_pos")[i]).float()[None],
            "jv": torch.tensor(col("joint_vel")[i]).float()[None],
        }
        if frames is not None:
            step["frames"] = torch.tensor(frames[i]).float()[None]  # [1, K, 13]
        hist.append(step)
    return hist


@torch.no_grad()
def rollout(ctx, seed_hist: list[dict], command: np.ndarray, steps: int) -> dict:
    """Autoregressive FM rollout; returns world-frame per-step reference arrays."""
    N, F = ctx.N, ctx.F
    hist = list(seed_hist)  # rolling window of the last N reference steps
    cmd = torch.tensor(command, dtype=torch.float32).expand(1, F, 3).contiguous()
    cmd_tokens = build_cmd_tokens(ctx, cmd)
    out = {k: [] for k in ("root_pos", "root_quat", "root_lin", "root_ang", "jp", "jv",
                            "frame_pos", "frame_quat", "frame_lin", "frame_ang")}
    has_frames = "frames" in seed_hist[0]
    for _ in range(steps):
        st = lambda k: torch.cat([h[k] for h in hist], dim=0)[None]  # [1, N, .]
        frames = (torch.cat([h["frames"] for h in hist], dim=0)[None]) if has_frames else None
        cond, known = build_cond_states(
            ctx, st("root_pos"), st("root_quat"), st("root_lin"), st("root_ang"),
            st("jp"), st("jv"), frames=frames,
        )
        w = sample_world(ctx, cond, known, st("root_pos")[:, -1], st("root_quat")[:, -1], cmd_tokens=cmd_tokens)
        nf = N  # first novel (predicted) step
        rec = {
            "root_pos": w["root_pos_w"][:, nf], "root_quat": w["root_quat_w"][:, nf],
            "root_lin": w["root_link_lin_vel_w"][:, nf], "root_ang": w["root_link_ang_vel_w"][:, nf],
            "jp": w["joint_pos"][:, nf], "jv": w["joint_vel"][:, nf],
        }
        if has_frames:
            rec.update(frame_pos=w["frame_pos_w"][:, nf], frame_quat=w["frame_quat_w"][:, nf],
                       frame_lin=w["frame_lin_vel_w"][:, nf], frame_ang=w["frame_ang_vel_w"][:, nf])
        for k in out:
            if k in rec:
                out[k].append(rec[k][0].cpu().numpy())
        # Append the novel step as the next history entry (autoregressive).
        nxt = {"root_pos": w["root_pos_w"][:, nf], "root_quat": w["root_quat_w"][:, nf],
               "root_lin": w["root_link_lin_vel_w"][:, nf], "root_ang": w["root_link_ang_vel_w"][:, nf],
               "jp": w["joint_pos"][:, nf], "jv": w["joint_vel"][:, nf]}
        if has_frames:
            f13 = torch.cat([w["frame_pos_w"][:, nf], w["frame_quat_w"][:, nf],
                             w["frame_lin_vel_w"][:, nf], w["frame_ang_vel_w"][:, nf]], dim=-1)
            nxt["frames"] = f13
        hist.append(nxt)
        hist = hist[-N:]
    return {k: (np.stack(v) if v else None) for k, v in out.items()}


def scatter_to_clip(roll: dict, pos_names: list[str], vel_names: list[str], base: str,
                    ee_frames: list[str], joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Scatter world-frame rollout arrays into clip ``y_des`` / ``dy_des`` columns."""
    T = roll["root_pos"].shape[0]
    P, V = len(pos_names), len(vel_names)
    y = np.zeros((T, P), dtype=np.float64)
    dy = np.zeros((T, V), dtype=np.float64)
    y[:, _cols(pos_names, base, _A3, "pos")] = roll["root_pos"]
    y[:, _cols(pos_names, base, _A4, "ori")] = roll["root_quat"]
    dy[:, _cols(vel_names, base, _A3, "pos")] = roll["root_lin"]
    dy[:, _cols(vel_names, base, _A3, "ori")] = roll["root_ang"]
    for j, jn in enumerate(joint_names):
        y[:, pos_names.index(f"joint:{jn}")] = roll["jp"][:, j]
        dy[:, vel_names.index(f"joint:{jn}")] = roll["jv"][:, j]
    for k, fr in enumerate(ee_frames):
        y[:, _cols(pos_names, fr, _A3, "pos")] = roll["frame_pos"][:, k]
        y[:, _cols(pos_names, fr, _A4, "ori")] = roll["frame_quat"][:, k]
        dy[:, _cols(vel_names, fr, _A3, "pos")] = roll["frame_lin"][:, k]
        dy[:, _cols(vel_names, fr, _A3, "ori")] = roll["frame_ang"][:, k]
    return y, dy


def main() -> None:
    """Roll out the FM and write a single-clip library mirroring the source schema."""
    p = argparse.ArgumentParser(description="Autoregressive FM rollout -> motion clip .h5.")
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--source_clip", required=True, help="Clip .h5 whose output-name schema to mirror.")
    p.add_argument("--seed_data", required=True, help="Clip-FM .npz to seed the history from.")
    p.add_argument("--out", required=True, help="Output clip .h5.")
    p.add_argument("--steps", type=int, default=400, help="Rollout length (clip steps).")
    p.add_argument("--command", default="1.0,0,0", help="Constant velocity command 'vx,vy,wz'.")
    p.add_argument("--seed_episode", type=int, default=0, help="Seed-data episode to take the N-step seed from.")
    p.add_argument("--base_frame", default="pelvis_link")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with h5py.File(args.source_clip, "r") as f:
        pos_names = json.loads(f.attrs["pos_output_names"])
        vel_names = json.loads(f.attrs["vel_output_names"])
        src_meta = {k: json.loads(f.attrs[k]) for k in
                    ("dt", "contact_frames", "ref_frame", "skill_list") if k in f.attrs}
    ee_frames = [n.split(":", 1)[0] for n in pos_names if ":pos_" in n]
    ee_frames = sorted({f for f in ee_frames if f != args.base_frame})

    ctx = load_fm_context(args.fm_checkpoint, device=args.device)
    if list(ctx.target_frame_names) != ee_frames:
        raise ValueError(f"FM ee frames {list(ctx.target_frame_names)} != source clip ee frames {ee_frames}.")
    command = np.array([float(x) for x in args.command.split(",")], dtype=np.float64)

    seed = _seed_history(args.seed_data, args.seed_episode, ctx.N, ctx)
    roll = rollout(ctx, seed, command, args.steps)
    y, dy = scatter_to_clip(roll, pos_names, vel_names, args.base_frame, ee_frames, list(ctx.joint_names))
    T = y.shape[0]
    finite = bool(np.isfinite(y).all() and np.isfinite(dy).all())
    z = roll["root_pos"][:, 2]
    print(f"[INFO] rollout: {T} steps, base-z range=[{z.min():.3f},{z.max():.3f}], finite={finite}")

    # Write a single-clip library mirroring the source schema (zeros for contact/skill/stride).
    C = len(src_meta.get("contact_frames", []))
    with h5py.File(args.out, "w") as f:
        f.create_dataset("y_des", data=y[None].astype(np.float32))
        f.create_dataset("dy_des", data=dy[None].astype(np.float32))
        f.create_dataset("target_vel", data=np.broadcast_to(command, (1, T, 3)).astype(np.float32))
        f.create_dataset("contact", data=np.zeros((1, T, C), np.uint8))
        f.create_dataset("skill_id", data=np.zeros((1, T), np.uint8))
        f.create_dataset("stride_idx", data=np.zeros((1, T), np.int32))
        f.attrs["pos_output_names"] = json.dumps(pos_names)
        f.attrs["vel_output_names"] = json.dumps(vel_names)
        f.attrs["num_steps"] = json.dumps(T)
        f.attrs["num_clips"] = 1
        f.attrs["num_clips_written"] = 1
        for k, v in src_meta.items():
            f.attrs[k] = json.dumps(v)
    print(f"[INFO] Wrote {args.out} (1 clip, {T} steps @ dt={src_meta.get('dt')}, cmd={command.tolist()})")
    print(f"[INFO]   play (IsaacLab): play.py --task G1-teacher-clf-play "
          f"env.commands.traj_ref.path={args.out} env.commands.traj_ref.random_start_phase=False")
    print(f"[INFO]   play (MuJoCo):   transfer/sim/gen_tracker_runner.py --clip {args.out} --load_run <ego_teacher>")


if __name__ == "__main__":
    main()
