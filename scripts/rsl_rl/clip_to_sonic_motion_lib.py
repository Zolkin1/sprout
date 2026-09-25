#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Convert clips from our library into SONIC's ``motion_lib`` format.

This exists for one purpose: to feed NVIDIA's own IsaacLab env the SAME clip our env tracks, so
the reference their observation code computes can be diffed against the reference our SONIC
adapter computes. Without that, a tracking discrepancy cannot be attributed to either side.

Runs in the **isolated SONIC venv** (IsaacLab 2.3.2 + gear_sonic), because it calls their
``convert_sequence`` rather than reimplementing it -- the ``pose_aa`` axis-angle construction and
the wxyz->xyzw root-rotation flip are theirs, and a reimplementation is exactly the kind of thing
that would invalidate the comparison it is meant to support.

Two conventions were verified rather than assumed:

* our clip library's joint channel order is **identical** to their ``G1_MUJOCO_ORDER``, so the
  sequence is handed over with ``joint_order="mj"`` and no permutation;
* our clips store the base orientation **wxyz** (scalar first), which is what their
  ``convert_sequence`` expects in ``body_quat_w``.

Only the pelvis entry of ``body_pos_w`` / ``body_quat_w`` is read by their converter (the rest of
the 14 tracked bodies are recomputed by their motion lib's forward kinematics from ``pose_aa``),
so the other slots are filled with identity and never used.

Usage (note: their venv, and their repo on the path)::

    cd /home/zolkin/AmberLab/Project-Isaac-RL/GR00T-WholeBodyControl
    /home/zolkin/AmberLab/Project-Isaac-RL/env_sonic232/bin/python \\
        <repo>/scripts/rsl_rl/clip_to_sonic_motion_lib.py \\
        --clip <repo>/clips/g1_mb_v5.h5 --clip_ids 7000 \\
        --out data/motion_lib_from_our_clips
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

NUM_TRACKED_BODIES = 14          # their motion_lib_cfg body_names length
TARGET_FPS = 50                  # their motion_lib_cfg.target_fps; our clips are dt=0.02


def _attr(f: h5py.File, key: str):
    """Read a clip-library attr, decoding the JSON it is stored as."""
    v = f.attrs[key]
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    return json.loads(v) if isinstance(v, str) else v


def main() -> int:
    """Convert the requested clips and write one joblib pkl per motion."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--clip", required=True, help="Our clip library .h5.")
    p.add_argument("--clip_ids", required=True,
                   help="Comma-separated clip indices to convert.")
    p.add_argument("--out", required=True, help="Output motion_lib directory.")
    p.add_argument("--session", default="our_clips",
                   help="Subdirectory name; their loader treats each subdir as a session.")
    p.add_argument("--gear_sonic_root", default=".",
                   help="GR00T-WholeBodyControl checkout root (for importing their converter).")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.gear_sonic_root))
    import joblib
    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import convert_sequence

    out_dir = os.path.join(args.out, args.session)
    os.makedirs(out_dir, exist_ok=True)
    clip_ids = [int(x) for x in args.clip_ids.split(",") if x.strip()]

    with h5py.File(args.clip, "r") as f:
        pos_names = list(_attr(f, "pos_output_names"))
        ref_frame = _attr(f, "ref_frame")
        dt = float(_attr(f, "dt"))
        if abs(dt - 1.0 / TARGET_FPS) > 1e-9:
            raise SystemExit(f"[conv] clip dt {dt} != 1/{TARGET_FPS}; their motion lib resamples "
                             "to target_fps and this script assumes they already match.")
        joint_cols = [i for i, n in enumerate(pos_names) if n.startswith("joint:")]
        base_pos_cols = [pos_names.index(f"{ref_frame}:pos_{a}") for a in "xyz"]
        # Our library stores orientation scalar-FIRST; their converter wants wxyz. No flip here.
        base_ori_cols = [pos_names.index(f"{ref_frame}:ori_{a}") for a in "wxyz"]
        print(f"[conv] {args.clip}: {len(joint_cols)} joints, ref frame '{ref_frame}', dt {dt}")

        for cid in clip_ids:
            n = int(f["clip_len"][cid])                    # real frames, excluding padding
            y = f["y_des"][cid, :n]
            seq = {
                "joint_pos": y[:, joint_cols].astype(np.float32),
                "body_pos_w": np.zeros((n, NUM_TRACKED_BODIES, 3), np.float32),
                "body_quat_w": np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, NUM_TRACKED_BODIES, 1)),
                # Verified identical to their G1_MUJOCO_ORDER, so no reordering is applied.
                "joint_order": "mj",
            }
            seq["body_pos_w"][:, 0, :] = y[:, base_pos_cols]
            seq["body_quat_w"][:, 0, :] = y[:, base_ori_cols]

            entry = convert_sequence(seq, fps=TARGET_FPS)
            name = f"clip{cid:05d}"
            path = os.path.join(out_dir, f"{name}.pkl")
            joblib.dump({name: entry}, path, compress=True)
            trav = float(np.hypot(*(y[:, base_pos_cols][:, :2].max(0)
                                   - y[:, base_pos_cols][:, :2].min(0))))
            print(f"[conv] clip {cid}: {n} frames ({n * dt:.1f} s), pelvis travel {trav:.2f} m "
                  f"-> {path}")
    print(f"[conv] wrote {len(clip_ids)} motion(s) to {out_dir}")
    print("[conv] point their env at it with:")
    print(f"         ++manager_env.commands.motion.motion_lib_cfg.motion_file={args.out}")
    print("         ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
