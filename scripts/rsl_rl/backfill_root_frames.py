# Copyright 2026 Zachary Olkin. All rights reserved.

"""Widen an existing clip library with root-frame body-link channels.

The root-frame channels (``<body>@root:pos_*`` / ``:ori_*``) are a pure function of the
``joint:*`` columns a library already stores -- they are just forward kinematics from the
pelvis, with velocities from the FK geometric Jacobian.  So a library generated before those
channels existed can be widened offline, exactly, with no GPU and no re-run of the upstream
Motion Bricks stage (whose raw ``.npz`` inputs are no longer kept in the tree).

That matters for the joint-vs-body tracking A/B: both arms must train on the *same motions*,
and re-generating would resample them.

The channel layout is produced by :func:`motion_gen.clip_output.widen_output_names`, the same
function :class:`~motion_gen.clip_output.ClipOutputBuilder` uses at generation time, so a
backfilled library and a freshly generated one agree channel-for-channel.  Existing columns
keep their indices; the new block is appended.

Usage::

    python scripts/rsl_rl/backfill_root_frames.py --in clips/g1_mb_v5.h5 --out clips/g1_mb_v5_rf.h5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "motion_gen"))

from motion_gen import features as F  # noqa: E402
from motion_gen.clip_output import widen_output_names  # noqa: E402
from motion_gen.fk import FKChain  # noqa: E402
from motion_gen.math_utils import matrix_to_quat  # noqa: E402

# Datasets whose trailing axis is the channel count and therefore has to be widened.
_CHANNEL_DATASETS = {"y_des": "pos", "dy_des": "vel"}


def _joint_columns(names: list[str], joint_names: list[str]) -> list[int]:
    """Column indices of ``joint:<name>`` in ``names``, in ``joint_names`` order.

    Args:
        names: Channel names from the library.
        joint_names: The FK joint order the columns must be gathered into.

    Returns:
        One column index per joint.

    Raises:
        ValueError: If any joint the FK needs is absent from the library.
    """
    lookup = {n: i for i, n in enumerate(names)}
    missing = [j for j in joint_names if f"joint:{j}" not in lookup]
    if missing:
        raise ValueError(
            f"library is missing {len(missing)} joint channel(s) the FK needs: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    return [lookup[f"joint:{j}"] for j in joint_names]


def _central_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Central difference along time with one-sided ends -- matches ``ingest_common.central_diff``.

    Reimplemented here (rather than imported) only to keep this script's import surface to the
    torch-free-ish core; the formula is identical and is covered by the parity test.

    Args:
        x: ``[T, D]`` samples.
        dt: Timestep.

    Returns:
        ``[T, D]`` time derivative.
    """
    d = torch.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def root_frame_block(
    joints: torch.Tensor, dt: float, fk: FKChain, root_frames: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Root-frame pose + velocity channels for one clip.

    Args:
        joints: ``[T, nj]`` joint angles in ``F.G1_JOINT_NAMES`` order.
        dt: Control timestep.
        fk: Chain built over ``root_frames`` with ``base_link="pelvis_link"``.
        root_frames: Body names, matching ``fk``'s frame order.

    Returns:
        ``(pos_block [T, 7*B], vel_block [T, 6*B])`` laid out body-major, matching
        :func:`motion_gen.clip_output.root_frame_channels`.
    """
    pos_p, rot_p, jv, jw = fk.pose_and_jacobian(joints)
    qd = _central_diff(joints, dt)
    pos_cols, vel_cols = [], []
    for i, _ in enumerate(root_frames):
        pos_cols += [pos_p[:, i], matrix_to_quat(rot_p[:, i])]
        vel_cols += [
            torch.einsum("taj,tj->ta", jv[:, i], qd),
            torch.einsum("taj,tj->ta", jw[:, i], qd),
        ]
    return torch.cat(pos_cols, dim=-1), torch.cat(vel_cols, dim=-1)


def backfill(src_path: str, dst_path: str, root_frames: list[str], batch: int,
             mjcf_path: str | None, device: str) -> None:
    """Copy ``src_path`` to ``dst_path`` with the root-frame channels appended.

    Every dataset and attribute is carried across unchanged except ``y_des`` / ``dy_des``
    (widened) and ``pos_output_names`` / ``vel_output_names`` (extended).

    Args:
        src_path: Existing library.
        dst_path: Output library; must not exist.
        root_frames: Bodies to add root-frame channels for.
        batch: Clips processed per read/write (bounds peak memory).
        mjcf_path: FK model override.
        device: Torch device for the FK pass.

    Raises:
        FileExistsError: If ``dst_path`` already exists.
    """
    if os.path.exists(dst_path):
        raise FileExistsError(f"{dst_path} exists; refusing to overwrite.")

    dev = torch.device(device)
    fk = FKChain(mjcf_path or F.DEFAULT_MJCF, F.G1_JOINT_NAMES, root_frames,
                 base_link="pelvis_link").to(dev)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        pos_names = json.loads(src.attrs["pos_output_names"])
        vel_names = json.loads(src.attrs["vel_output_names"])
        new_pos, new_vel = widen_output_names(pos_names, vel_names, root_frames)
        widths = {"pos": (len(pos_names), len(new_pos)), "vel": (len(vel_names), len(new_vel))}
        jcols = _joint_columns(pos_names, F.G1_JOINT_NAMES)
        dt = float(json.loads(src.attrs["dt"]))
        n_clips = int(src["y_des"].shape[0])

        print(f"[backfill] {src_path} -> {dst_path}")
        print(f"[backfill]   {n_clips} clips, dt={dt}, "
              f"P {widths['pos'][0]}->{widths['pos'][1]}, V {widths['vel'][0]}->{widths['vel'][1]}")
        print(f"[backfill]   {len(root_frames)} root frames: {', '.join(root_frames)}")

        for key, attr in src.attrs.items():
            dst.attrs[key] = attr
        dst.attrs["pos_output_names"] = json.dumps(new_pos)
        dst.attrs["vel_output_names"] = json.dumps(new_vel)
        dst.attrs["root_frame_backfill"] = json.dumps(
            {"source": os.path.basename(src_path), "root_frames": list(root_frames)}
        )

        # Non-channel datasets copy straight across, preserving chunking/compression.
        for name, ds in src.items():
            if name in _CHANNEL_DATASETS:
                continue
            dst.create_dataset(name, data=ds[...], dtype=ds.dtype,
                               chunks=ds.chunks, compression=ds.compression)

        for name, kind in _CHANNEL_DATASETS.items():
            src_ds = src[name]
            old_w, new_w = widths[kind]
            if src_ds.shape[-1] != old_w:
                raise ValueError(
                    f"{name} has {src_ds.shape[-1]} channels but {kind}_output_names has {old_w}."
                )
            shape = src_ds.shape[:-1] + (new_w,)
            dst.create_dataset(name, shape=shape, dtype=src_ds.dtype,
                               chunks=(1,) + shape[1:], compression=src_ds.compression)

        for start in range(0, n_clips, batch):
            end = min(start + batch, n_clips)
            y_old = src["y_des"][start:end]
            dy_old = src["dy_des"][start:end]
            y_new = np.empty(y_old.shape[:-1] + (widths["pos"][1],), dtype=y_old.dtype)
            dy_new = np.empty(dy_old.shape[:-1] + (widths["vel"][1],), dtype=dy_old.dtype)
            y_new[..., : widths["pos"][0]] = y_old
            dy_new[..., : widths["vel"][0]] = dy_old
            for k in range(end - start):
                joints = torch.as_tensor(y_old[k][:, jcols], dtype=torch.float32, device=dev)
                pos_blk, vel_blk = root_frame_block(joints, dt, fk, root_frames)
                y_new[k, :, widths["pos"][0]:] = pos_blk.cpu().numpy()
                dy_new[k, :, widths["vel"][0]:] = vel_blk.cpu().numpy()
            dst["y_des"][start:end] = y_new
            dst["dy_des"][start:end] = dy_new
            print(f"[backfill]   {end}/{n_clips}", flush=True)

    print(f"[backfill] done -> {dst_path}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True, help="input clip library .h5")
    p.add_argument("--out", dest="dst", required=True, help="output .h5 (must not exist)")
    p.add_argument("--batch", type=int, default=64, help="clips per read/write chunk")
    p.add_argument("--mjcf", default=None, help="FK model override (default: features.DEFAULT_MJCF)")
    p.add_argument("--device", default="cpu", help="torch device for the FK pass")
    p.add_argument("--frames", default=None,
                   help="comma-separated body names (default: G1_BODY_TRACK_FRAMES)")
    a = p.parse_args()
    frames = ([s.strip() for s in a.frames.split(",") if s.strip()]
              if a.frames else list(F.G1_BODY_TRACK_FRAMES))
    backfill(a.src, a.dst, frames, a.batch, a.mjcf, a.device)


if __name__ == "__main__":
    main()
