# Copyright 2026 Zachary Olkin. All rights reserved.

"""Convert a clean motion-clip library into flow-matching training data.

The clip-conditioned FM (history = past clip, target = future clip, conditioned on
the velocity command, no robot state) trains on the **clips themselves** -- nothing
recorded from sim. The clip already stores every channel the FM state needs -- base
pos/ori plus lin/ang velocity, joint pos/vel, and the end-effector frames -- in one
self-consistent clip-world frame (``dy_des`` carries the velocities). This script
re-maps the clip's ``pos_output_names`` / ``vel_output_names`` outputs into the exact
``state`` layout that ``scripts/rsl_rl/record_data.py`` emits and writes a
recorded-style ``.npz`` via ``recorded_data_io.save_recorded_data``, so
``scripts/flow_matching/train.py`` consumes it unchanged.

Because no ``ref_state`` is written, ``train.py`` auto-resolves
``target_source='actual'`` and the dataset sets ``cond = target = state`` -- i.e.
the FM conditions on the past clip and predicts the future clip, with no robot-state
dependence. ``target_vel`` is carried through as the command-token source.

Each clip becomes one "episode" (so the FM's episode-aware windower never crosses a
clip boundary). The clip-world frame is treated as the FM's world frame (identity
placement); the FM's relative-frame representation anchors each window internally, so
no stance-anchor recovery is needed.

Numpy + h5py only -- no IsaacLab / torch / simulator (mirrors transfer/sim/clip_reference.py).

Example:
    python scripts/flow_matching/clip_to_fm_data.py \
        --clip clips/g1_flat_v1.h5 --out clips/g1_flat_v1_fm.npz
    # smoke: a small subset
    python scripts/flow_matching/clip_to_fm_data.py \
        --clip clips/g1_flat_v1.h5 --out /tmp/clip_fm_smoke.npz --max_clips 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# recorded_data_io lives in scripts/rsl_rl (numpy + stdlib only; no IsaacLab).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, os.pardir, "rsl_rl"))
from recorded_data_io import (  # noqa: E402  isort: skip
    RecordingMeta,
    fingerprint_arrays,
    save_recorded_data,
)

_AXES3 = ("x", "y", "z")
_AXES4 = ("x", "y", "z", "w")  # quaternion xyzw (Isaac Lab convention)


def _idx(names: list[str], frame: str, comps: tuple[str, ...], kind: str) -> list[int]:
    """Column indices of ``{frame}:{kind}_{c}`` for each ``c`` in ``comps``."""
    return [names.index(f"{frame}:{kind}_{c}") for c in comps]


def resolve_ee_frames(pos_names: list[str], base_frame: str) -> list[str]:
    """End-effector reference frames: every output frame with a position, minus base.

    Sorted alphabetically -- the same rule as
    ``MotionSequenceCommand.resolve_ee_frames`` (the single source of truth the
    teacher's ``ee_ref`` obs uses), so the FM's predicted frames line up with the
    tracker's expectation.
    """
    frames = sorted({n.split(":", 1)[0] for n in pos_names if ":pos_" in n})
    return [f for f in frames if f != base_frame]


def convert(
    clip_path: str, base_frame: str, max_clips: int | None, stride: int
) -> tuple[dict[str, np.ndarray], list[str], list[str], float]:
    """Build the recorded-style arrays from a clip library.

    Returns ``(data, joint_names, ee_frames, dt)`` where ``data`` holds the per-step
    arrays (``state``, ``target_vel``, episode metadata, dummy ``obs``/``action``).
    """
    with h5py.File(clip_path, "r") as f:
        pos_names = json.loads(f.attrs["pos_output_names"])
        vel_names = json.loads(f.attrs["vel_output_names"])
        dt = float(json.loads(f.attrs["dt"]))
        n_all = f["y_des"].shape[0]
        n = n_all if max_clips is None else min(int(max_clips), n_all)
        # Read only the kept clips (clip-world outputs); subsample steps by ``stride``.
        sl = slice(0, n)
        y = np.asarray(f["y_des"][sl], dtype=np.float64)[:, ::stride]    # [n, T, P]
        dy = np.asarray(f["dy_des"][sl], dtype=np.float64)[:, ::stride]  # [n, T, V]
        if "target_vel" not in f:
            raise ValueError(
                f"{clip_path} has no 'target_vel'; the clip-conditioned FM is "
                "command-conditioned. Regenerate clips with a generator that writes it."
            )
        tv = np.asarray(f["target_vel"][sl], dtype=np.float64)[:, ::stride]  # [n, T, 3]

    n_clips, T = y.shape[0], y.shape[1]

    # --- channel maps (clip output order -> record_data state layout) ---
    b = base_frame
    base_pos = _idx(pos_names, b, _AXES3, "pos")
    base_ori = _idx(pos_names, b, _AXES4, "ori")   # xyzw
    base_lin = _idx(vel_names, b, _AXES3, "pos")   # linear vel lives in the pos_* vel channels
    base_ang = _idx(vel_names, b, _AXES3, "ori")   # angular vel in the ori_* vel channels
    joint_names = [n.split(":", 1)[1] for n in pos_names if n.startswith("joint:")]
    jp = [pos_names.index(f"joint:{j}") for j in joint_names]
    jv = [vel_names.index(f"joint:{j}") for j in joint_names]
    ee_frames = resolve_ee_frames(pos_names, base_frame)
    ee_maps = [
        (_idx(pos_names, fr, _AXES3, "pos"), _idx(pos_names, fr, _AXES4, "ori"),
         _idx(vel_names, fr, _AXES3, "pos"), _idx(vel_names, fr, _AXES3, "ori"))
        for fr in ee_frames
    ]

    # --- assemble the per-step state in the record_data term order ---
    # [root_pos_w(3), root_quat_w(4 xyzw), root_link_lin_vel_w(3), root_link_ang_vel_w(3),
    #  joint_pos(nj), joint_vel(nj), {ee}:pos_w(3),quat_w(4),lin_vel_w(3),ang_vel_w(3) ...]
    blocks = [y[:, :, base_pos], y[:, :, base_ori], dy[:, :, base_lin], dy[:, :, base_ang],
              y[:, :, jp], dy[:, :, jv]]
    for p_i, o_i, l_i, a_i in ee_maps:
        blocks += [y[:, :, p_i], y[:, :, o_i], dy[:, :, l_i], dy[:, :, a_i]]
    state = np.concatenate(blocks, axis=-1).reshape(n_clips * T, -1)  # [n*T, state_dim]
    target_vel = tv.reshape(n_clips * T, 3)

    # --- episode metadata: one episode per clip (windower never crosses clips) ---
    episode_id = np.repeat(np.arange(n_clips, dtype=np.int64), T)
    step_in_episode = np.tile(np.arange(T, dtype=np.int64), n_clips)
    done = np.zeros(n_clips * T, dtype=np.uint8)
    done[step_in_episode == (T - 1)] = 1          # truncate at each clip end
    time_out = done.copy()
    terminated = np.zeros(n_clips * T, dtype=np.uint8)

    nj = len(joint_names)
    data = {
        "obs": np.zeros((n_clips * T, 1), dtype=np.float32),       # unused by the FM dataset
        "action": np.zeros((n_clips * T, nj), dtype=np.float32),   # unused by the FM dataset
        "state": state.astype(np.float32),
        "target_vel": target_vel.astype(np.float32),
        "terminated": terminated,
        "done": done,
        "time_out": time_out,
        "episode_id": episode_id,
        "step_in_episode": step_in_episode,
    }
    return data, joint_names, ee_frames, dt


def build_meta(
    clip_path: str, joint_names: list[str], ee_frames: list[str], dt: float, num_pairs: int,
    content_sha256: str,
) -> RecordingMeta:
    """Recorded-data metadata describing the clip-derived state layout."""
    nj = len(joint_names)
    state_term_names = ["root_pos_w", "root_quat_w", "root_link_lin_vel_w",
                        "root_link_ang_vel_w", "joint_pos", "joint_vel"]
    state_term_dims = [3, 4, 3, 3, nj, nj]
    for fr in ee_frames:
        state_term_names += [f"{fr}:pos_w", f"{fr}:quat_w", f"{fr}:lin_vel_w", f"{fr}:ang_vel_w"]
        state_term_dims += [3, 4, 3, 3]
    return RecordingMeta(
        checkpoint_path="",
        checkpoint_sha256="",
        task_name=f"clip:{os.path.basename(clip_path)}",
        experiment_name="clip_to_fm_data",
        run_name="",
        runner_class="",
        rsl_rl_version="",
        recorded_at="",
        stochastic=False,
        dt=dt,
        num_envs=0,
        num_pairs=num_pairs,
        obs_term_names=["dummy"],
        obs_term_dims=[1],
        action_joint_names=joint_names,
        state_term_names=state_term_names,
        state_term_dims=state_term_dims,
        state_quat_format="xyzw",
        state_joint_names=joint_names,
        state_vel_frame="link",
        state_frame_names=ee_frames,
        content_sha256=content_sha256,
    )


def main() -> None:
    """Parse arguments, convert the clip library, and write the FM ``.npz``."""
    p = argparse.ArgumentParser(description="Clip library -> flow-matching training .npz.")
    p.add_argument("--clip", required=True, help="Input clip-library .h5 (e.g. clips/g1_flat_v1.h5).")
    p.add_argument("--out", required=True, help="Output recorded-style .npz path.")
    p.add_argument("--base_frame", default="pelvis_link", help="Clip base/root output frame.")
    p.add_argument("--max_clips", type=int, default=None, help="Use at most this many clips (smoke).")
    p.add_argument("--stride", type=int, default=1, help="Keep every Nth clip step (subsample).")
    args = p.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    data, joint_names, ee_frames, dt = convert(args.clip, args.base_frame, args.max_clips, args.stride)
    num_pairs = int(data["state"].shape[0])
    meta = build_meta(args.clip, joint_names, ee_frames, dt, num_pairs, fingerprint_arrays(data))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_recorded_data(args.out, data, meta)

    n_clips = int(data["episode_id"].max()) + 1
    gb = data["state"].nbytes / 1e9
    print(f"[INFO] Wrote {args.out}")
    print(f"[INFO]   clips(episodes)={n_clips}  steps={num_pairs}  state_dim={data['state'].shape[1]}"
          f"  (~{gb:.2f} GB state, dt={dt})")
    print(f"[INFO]   joints(nj={len(joint_names)}), ee_frames={ee_frames}")
    print("[INFO]   no ref_state -> train.py resolves target_source='actual' (clip->clip), "
          "cmd_array=target_vel.")


if __name__ == "__main__":
    main()
