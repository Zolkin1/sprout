#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Behavioural gate for the SONIC adapter: it must track a FLAT clip before terrain results mean
anything.

The unit tests in ``source/robot_rl/tests/test_sonic_obs.py`` pin the observation layout block by
block (including a cross-check against the real encoder graph, which confirms which slices the g1
mode reads). What they cannot show is that the whole loop -- reference sourcing, joint permutation,
history, action scale, the robot itself -- is wired up well enough to walk. This does: SONIC has to
survive a flat clip with a bounded pelvis error.

Run this whenever the adapter, the SONIC asset config, or the reference plumbing changes. A poor
result on stairs is a finding; a poor result HERE is a bug, and this is the line between them.

Note on obs ordering: SONIC's proprioception block order was an open question when this baseline
was designed (its training ``PolicyCfg`` field order and its C++ deploy config list the five
history blocks differently). It is settled -- the two agree once you read the ``@configclass``
field order rather than the hydra ``defaults`` list, and the layout is asserted against the graphs
in ``test_sonic_obs.py`` -- so this script no longer searches over candidate orderings.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/verify_sonic_adapter.py \\
        --clip clips/g1_mb_v5.h5 \\
        --sonic_dir ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def main() -> int:
    """Roll SONIC out on a flat clip and check it tracked."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--clip", required=True,
                   help="Clip library .h5. It must contain FLAT clips (terrain_id == -1); the "
                        "small terrain-only libraries do not.")
    p.add_argument("--sonic_dir", required=True, help="SONIC release ONNX directory.")
    p.add_argument("--clip_device", default="cpu")
    p.add_argument("--num_envs", type=int, default=24,
                   help="Envs to simulate; enough that the importer instantiates a flat tile.")
    p.add_argument("--steps", type=int, default=600, help="Control steps (600 = 12 s).")
    p.add_argument("--clip_pick", type=int, default=0)
    p.add_argument("--ego_anchor_fraction", type=float, default=1.0,
                   help="Pinned so the gate is reproducible: at the cfg default (0.5) the clip is "
                        "re-anchored onto the robot for a RANDOM half of the envs, so the same "
                        "clip on the same tile gave 0.069 vs 0.247 m mean error on two runs. "
                        "1.0 (every env, every ego_anchor_every steps) is both deterministic and "
                        "the regime a policy with no root-position target can work in -- 0.0 "
                        "would let the reference translate away and trip the drift termination no "
                        "matter how well the joints were tracked, which would not tell us "
                        "anything about the adapter.")
    p.add_argument("--min_vx_frac", type=float, default=0.5,
                   help="Required achieved/commanded forward velocity. SONIC not translating was "
                        "the SYMPTOM of the reference being double-permuted, and the gate only "
                        "REPORTED velocity while asserting joint error, so it read as a marginal "
                        "tracking miss rather than a wiring bug. Correct behaviour is 0.38 of a "
                        "commanded 0.47 m/s (0.81); the broken path achieved -0.06 (negative).")
    p.add_argument("--max_joint_err_rad", type=float, default=0.12,
                   help="Required mean |joint - reference| over the first episode. This is the "
                        "PRIMARY signal: SONIC is a joint-space tracker, so a correctly wired "
                        "adapter shows up here first. A correct adapter measures 0.063 rad here, "
                        "matching the 0.069 the released policy achieves inside NVIDIA's own env "
                        "on the same clip, so 0.12 is ~2x headroom rather than an arbitrary bound.")
    p.add_argument("--min_survive_s", type=float, default=8.0,
                   help="Required time before the first termination.")
    p.add_argument("--max_pelvis_err_m", type=float, default=0.25,
                   help="Required mean pelvis |ref - actual| over the first episode. With "
                        "ego re-anchoring on this is LOCAL tracking error, not absolute progress.")
    p.add_argument("--max_root_z_err_m", type=float, default=0.06,
                   help="Required mean |reference root z - actual root z|.")
    p.add_argument("--keep", default=None, help="Keep the recorded bundle at this path.")
    args = p.parse_args()

    out = args.keep or os.path.join(tempfile.mkdtemp(prefix="sonic_gate_"), "flat.npz")
    cmd = [
        sys.executable, "-u", os.path.join(_HERE, "debug_episodes.py"),
        "--task", "G1-sonic-terrain-play",
        "--policy", "sonic", "--sonic_dir", args.sonic_dir,
        "--clip", args.clip, "--clip_device", args.clip_device,
        "--num_envs", str(args.num_envs), "--steps", str(args.steps),
        # terrain_id -1 is flat ground; flat tiles carry no manifest entry, hence the raw id.
        "--capture_terrain_ids", "-1",
        "--clip_pick", str(args.clip_pick), "--no_dr",
        "--ego_anchor_fraction", str(args.ego_anchor_fraction),
        "--out", out,
    ]
    print("[gate] " + " ".join(cmd), flush=True)
    log = out.replace(".npz", ".log")
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=_REPO, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(out):
        print(f"[gate] FAIL: rollout returned {rc}; see {log}")
        return 1

    d = np.load(out, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    dt = float(meta["dt"])
    n_ep = int(meta["num_captured"])
    print(f"[gate] {n_ep} flat episode(s) x {args.steps} steps, dt={dt}")

    ok = True
    for ep in range(n_ep):
        done = d["done"][:, ep].astype(bool)
        end = int(np.argmax(done)) + 1 if done.any() else d["done"].shape[0]
        err = np.linalg.norm(d["ref_pos"][:end, ep, 0] - d["body_pos"][:end, ep, 0], axis=-1)
        surv = end * dt
        mean_err = float(err.mean())
        perm = [meta["ref_joint_names"].index(n) for n in meta["joint_names"]]
        jerr = float(np.abs(d["joint_pos"][:end, ep]
                            - d["ref_joint_pos"][:end, ep][:, perm]).mean())
        zerr = float(np.abs(d["ref_root_pos"][:end, ep, 2] - d["root_pos"][:end, ep, 2]).mean())
        cmd_vx = float(d["cmd_vel"][:end, ep, 0].mean())
        ach_vx = float(d["ach_vel"][:end, ep, 0].mean())
        checks = {
            "survived_s": (surv, args.min_survive_s, "ge"),
            "vx_frac": (ach_vx / cmd_vx if abs(cmd_vx) > 1e-6 else 0.0, args.min_vx_frac, "ge"),
            "joint_err_rad": (jerr, args.max_joint_err_rad, "le"),
            "pelvis_err_m": (mean_err, args.max_pelvis_err_m, "le"),
            "root_z_err_m": (zerr, args.max_root_z_err_m, "le"),
        }
        good = all((v >= lim) if how == "ge" else (v <= lim) for v, lim, how in checks.values())
        ok &= good
        print(f"[gate] ep{ep} tile {int(d['terrain_id'][ep])}: " + "  ".join(
            f"{k}={v:.3f}({'>=' if how == 'ge' else '<='}{lim})"
            f"{'' if ((v >= lim) if how == 'ge' else (v <= lim)) else ' !!'}"
            for k, (v, lim, how) in checks.items()) + f"  -> {'PASS' if good else 'FAIL'}")
        print(f"[gate]      vx commanded {cmd_vx:+.2f} m/s vs achieved {ach_vx:+.2f} m/s "
              f"(ratio {ach_vx / cmd_vx if abs(cmd_vx) > 1e-6 else float('nan'):.2f})")

    if not n_ep:
        print("[gate] FAIL: no flat episode captured -- does this library have flat clips?")
        ok = False
    print(f"\n[gate] {'PASS' if ok else 'FAIL'}  (bundle {out})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
