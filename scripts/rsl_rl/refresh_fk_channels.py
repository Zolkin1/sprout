# Copyright 2026 Zachary Olkin. All rights reserved.

"""Recompute a clip library's FK channels against a new robot model.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
    python scripts/rsl_rl/refresh_fk_channels.py \\
        --in clips/g1_mb_v5_body.h5 \\
        --out clips/g1_mb_v5_body_mode11.h5 \\
        --mjcf transfer/sim/robots/g1/g1_perceptive.xml --device cuda

A clip library stores three kinds of position channel and they age differently:

* ``joint:*`` and ``pelvis_link:*`` come from the motion source. No robot model touches
  them, so a model change leaves them exactly right.
* ``<body>@root:*`` is pure forward kinematics from the pelvis. Change the model and every
  one of them is wrong, silently -- they are still smooth, still plausible, and now describe
  a robot that does not exist.
* the four world-frame EE blocks are the pelvis pose composed with the same FK.

Measured for the mode_11 model change, over the WHOLE library (10 000 clips x 1000 steps)
by the run that produced it: the six leg frames move at most 0.0002 mm (float32 recompute
noise -- their geometry is untouched), ``torso_link`` moves up to 10.0999 mm, and the six
arm frames up to 5.8046 mm. The torso figure is essentially the constant 10 mm the waist
chain drops; the arms cancel exactly at zero waist angles (the +10 mm shoulder offset against
the -10 mm torso move) and diverge only under waist roll/pitch. Rotations do not change at
all, because mode_11 only translates the waist and shoulder frames.

**Validated before use.** Recomputing the ``@root`` block with the OLD model reproduces
what is already stored BIT-EXACTLY -- all 91 position and 78 velocity channels, to
``0.00e+00``. So this script's arithmetic is the arithmetic that produced the library, and
its output is what a regeneration would have produced.

**What it does NOT touch, and why that is not laziness.** The 24 world-frame EE VELOCITY
channels are left as they are, because they are not recoverable. The world POSE composes
exactly (``pelvis (x) @root`` reproduces it to 0.0005 mm, i.e. float32 storage precision),
but no rigid composition reproduces the world velocities: ``V + R v_root`` misses by up to
1.5 m/s and adding the ``omega x r`` term does not fix it. They came from the ingest's own
source-motion velocities, and those ``.npz`` inputs are no longer in the tree. Recomputing
them by composition would not be a refresh -- it would silently redefine the channel, for
all four bodies including the two whose FK did not change.

The consequence is stated rather than hidden: after this runs, the two WRIST world velocity
blocks are consistent with the old model while their positions are consistent with the new
one. The script prints the size of that inconsistency. The ankles are unaffected either way.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backfill_root_frames import _joint_columns, root_frame_block  # noqa: E402
from motion_gen import features as F  # noqa: E402
from motion_gen.fk import FKChain  # noqa: E402
from motion_gen.math_utils import quat_apply, quat_mul  # noqa: E402

#: Channel components, in the order the library lays them out.
_POS_XYZ = ("pos_x", "pos_y", "pos_z")
_ORI_WXYZ = ("ori_w", "ori_x", "ori_y", "ori_z")


def _cols(names: list[str], tag: str, comps: tuple[str, ...]) -> list[int]:
    """Column indices of ``<tag><comp>`` for each component, in order.

    Args:
        names: The library's channel names.
        tag: Channel prefix, e.g. ``"left_knee_link@root:"``.
        comps: Component suffixes.

    Returns:
        One index per component.

    Raises:
        KeyError: If any component is absent.
    """
    lookup = {n: i for i, n in enumerate(names)}
    return [lookup[tag + c] for c in comps]


def world_ee_bodies(pos_names: list[str], root_frames: list[str]) -> list[str]:
    """Bodies with a WORLD-frame block that can be recomposed from the pelvis and ``@root``.

    A world block is recomposable only if the same body also has an ``@root`` block -- there
    is nothing to compose otherwise. On the shipped library all four qualify.

    Args:
        pos_names: The library's position channel names.
        root_frames: Bodies carrying ``@root`` channels.

    Returns:
        Body names, in the order their world blocks appear.
    """
    out: list[str] = []
    for name in pos_names:
        if "@root" in name or name.startswith("joint:") or not name.endswith(":pos_x"):
            continue
        body = name[: -len(":pos_x")]
        if body != "pelvis_link" and body in root_frames and body not in out:
            out.append(body)
    return out


def refresh(src_path: str, dst_path: str, mjcf_path: str, batch: int, device: str) -> None:
    """Write ``dst_path``: ``src_path`` with its FK channels recomputed against ``mjcf_path``.

    Args:
        src_path: Existing library. Never modified.
        dst_path: Output library; must not exist.
        mjcf_path: The new robot model.
        batch: Clips per read/write chunk, bounding peak memory.
        device: Torch device for the FK pass.

    Raises:
        FileExistsError: If ``dst_path`` exists.
        ValueError: If the library carries no ``@root`` block to refresh.
    """
    if os.path.exists(dst_path):
        raise FileExistsError(f"{dst_path} exists; refusing to overwrite.")

    dev = torch.device(device)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        pos_names = json.loads(src.attrs["pos_output_names"])
        vel_names = json.loads(src.attrs["vel_output_names"])
        dt = float(json.loads(src.attrs["dt"]))
        root_frames = [n[: -len("@root:pos_x")] for n in pos_names
                       if n.endswith("@root:pos_x")]
        if not root_frames:
            raise ValueError(
                f"{src_path} has no <body>@root channels. There is nothing to refresh -- "
                "use backfill_root_frames.py to add them first."
            )
        world_bodies = world_ee_bodies(pos_names, root_frames)
        jcols = _joint_columns(pos_names, F.G1_JOINT_NAMES)
        n_clips = int(src["y_des"].shape[0])

        fk = FKChain(mjcf_path, F.G1_JOINT_NAMES, root_frames,
                     base_link="pelvis_link").to(dev)

        # Destination slices, resolved once.
        root_pos_cols = [c for b in root_frames
                         for c in _cols(pos_names, f"{b}@root:", _POS_XYZ + _ORI_WXYZ)]
        root_vel_cols = [c for b in root_frames
                         for c in _cols(vel_names, f"{b}@root:", _POS_XYZ + _ORI_WXYZ[1:])]
        pel_p = _cols(pos_names, "pelvis_link:", _POS_XYZ)
        pel_q = _cols(pos_names, "pelvis_link:", _ORI_WXYZ)
        world_p = {b: _cols(pos_names, f"{b}:", _POS_XYZ) for b in world_bodies}
        world_q = {b: _cols(pos_names, f"{b}:", _ORI_WXYZ) for b in world_bodies}

        print(f"[refresh] {src_path} -> {dst_path}")
        print(f"[refresh]   model {mjcf_path}")
        print(f"[refresh]   {n_clips} clips, dt={dt}, {len(pos_names)} pos / "
              f"{len(vel_names)} vel channels")
        print(f"[refresh]   recomputing {len(root_pos_cols)} @root pos + "
              f"{len(root_vel_cols)} @root vel over {len(root_frames)} bodies")
        print(f"[refresh]   recomposing world POSE for {world_bodies}")
        print(f"[refresh]   leaving world VELOCITY untouched (not recoverable -- "
              "see the module docstring)")

        for key, attr in src.attrs.items():
            dst.attrs[key] = attr
        dst.attrs["fk_refresh"] = json.dumps({
            "source": os.path.basename(src_path),
            "mjcf": os.path.relpath(os.path.abspath(mjcf_path),
                                    os.path.dirname(os.path.dirname(
                                        os.path.dirname(os.path.abspath(__file__))))),
            "root_frames": list(root_frames),
            "world_pose_recomposed": list(world_bodies),
            "world_velocity_untouched": True,
        })

        for name, ds in src.items():
            if name in ("y_des", "dy_des"):
                dst.create_dataset(name, shape=ds.shape, dtype=ds.dtype,
                                   chunks=ds.chunks or ((1,) + ds.shape[1:]),
                                   compression=ds.compression)
            else:
                dst.create_dataset(name, data=ds[...], dtype=ds.dtype,
                                   chunks=ds.chunks, compression=ds.compression)

        worst_pos = {b: 0.0 for b in root_frames}
        worst_world = {b: 0.0 for b in world_bodies}
        for start in range(0, n_clips, batch):
            end = min(start + batch, n_clips)
            y = np.array(src["y_des"][start:end])
            dy = np.array(src["dy_des"][start:end])
            for k in range(end - start):
                joints = torch.as_tensor(y[k][:, jcols], dtype=torch.float32, device=dev)
                pos_blk, vel_blk = root_frame_block(joints, dt, fk, root_frames)
                pos_np = pos_blk.cpu().numpy()

                for b, body in enumerate(root_frames):
                    old = y[k][:, _cols(pos_names, f"{body}@root:", _POS_XYZ)]
                    worst_pos[body] = max(worst_pos[body],
                                          float(np.abs(pos_np[:, b * 7:b * 7 + 3] - old).max()))

                y[k][:, root_pos_cols] = pos_np
                dy[k][:, root_vel_cols] = vel_blk.cpu().numpy()

                # World pose = pelvis (x) @root. Recomposed from the values just written, so
                # the two blocks cannot disagree.
                q_pel = torch.as_tensor(y[k][:, pel_q], dtype=torch.float64)
                p_pel = torch.as_tensor(y[k][:, pel_p], dtype=torch.float64)
                for body in world_bodies:
                    b = root_frames.index(body)
                    p_r = torch.as_tensor(pos_np[:, b * 7:b * 7 + 3], dtype=torch.float64)
                    q_r = torch.as_tensor(pos_np[:, b * 7 + 3:b * 7 + 7], dtype=torch.float64)
                    p_w = (p_pel + quat_apply(q_pel, p_r)).numpy()
                    q_w = quat_mul(q_pel, q_r).numpy()
                    worst_world[body] = max(
                        worst_world[body], float(np.abs(p_w - y[k][:, world_p[body]]).max()))
                    y[k][:, world_p[body]] = p_w
                    y[k][:, world_q[body]] = q_w

            dst["y_des"][start:end] = y
            dst["dy_des"][start:end] = dy
            print(f"[refresh]   {end}/{n_clips}", flush=True)

        print("\n[refresh] largest change per @root body (position, mm):")
        for body in root_frames:
            d = worst_pos[body] * 1000.0
            print(f"[refresh]   {body:28s} {d:9.4f}"
                  f"{'   unchanged' if d < 1e-6 else ''}")
        print("[refresh] largest change per world block (position, mm):")
        for body in world_bodies:
            print(f"[refresh]   {body:28s} {worst_world[body] * 1000.0:9.4f}")

    print(f"[refresh] done -> {dst_path}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True, help="input clip library .h5")
    p.add_argument("--out", dest="dst", required=True, help="output .h5 (must not exist)")
    p.add_argument("--mjcf", required=True, help="the NEW robot model to recompute against")
    p.add_argument("--batch", type=int, default=64, help="clips per read/write chunk")
    p.add_argument("--device", default="cpu", help="torch device for the FK pass")
    args = p.parse_args()
    refresh(args.src, args.dst, args.mjcf, args.batch, args.device)


if __name__ == "__main__":
    main()
