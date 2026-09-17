# Copyright 2026 Zachary Olkin. All rights reserved.

"""Can a spline over K knots represent the correction the adjuster needs to make?

The adjuster's action is the residual it adds to the generator's 62-node reference window.
At full width that is 2728 dimensions driven by one scalar advantage, which measured out at
~97% exploration noise and ~3% learned mean. The proposed fix is to emit the residual as
``K`` knots interpolated up to 62 nodes by a FIXED linear map ``B``, shrinking the search
space ~4.8x. That only works if the correction actually lives in the span of ``B``.

This measures it, before any training. The target correction is
``d = authored_clip_window - generated_window`` on the generator's own validation windows;
the script least-squares fits ``d ~= B u`` and reports the fraction of the correction's
ENERGY the basis captures, per channel group, per knot count and per interpolation kind.

**What this is and is not.** These are OPEN-LOOP, in-distribution windows: the conditioning
is the recording's, not a drifted closed-loop history, so ``d`` here is the generator's
one-shot prediction error and not the full closed-loop correction. It is the right thing for
the question asked -- what SHAPE does the needed correction have along the node axis, and can
a coarse spline carry it -- and it under-states magnitude. Read the RMS numbers as a floor.

Example::

    python scripts/flow_matching/knot_basis_fit.py \\
        --checkpoint logs/flow_matching/n3_measured_n3/best.pt \\
        --data logs/rsl_rl/g1/<run>/recorded_data/recorded_20260822_181314.h5
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import torch

# `recorded_data_io` lives beside the RL scripts, and `fm_error_by_offset` imports it by bare
# name -- the same bootstrap `scripts/flow_matching/train.py` does, and the reason it is here
# rather than in a package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                "rsl_rl"))

from robot_rl.flow_matching.config import FlowMatchingConfig
from robot_rl.flow_matching.dataset import build_train_val
from robot_rl.flow_matching.flow import sample_trajectory
from robot_rl.flow_matching.online import load_fm_context
from robot_rl.flow_matching.knot_basis import knot_basis, knot_positions
from robot_rl.flow_matching.window import REFERENCE_WINDOW_RESIDUAL_SCALE

from fm_error_by_offset import load_capped

#: Channel groups of the 44-wide node, in packing order, as
#: :data:`REFERENCE_WINDOW_RESIDUAL_SCALE` keys them. ``joint_pos`` runs to the end.
GROUPS: tuple[tuple[str, int, int], ...] = (
    ("base_pos", 0, 3), ("base_ori", 3, 9), ("base_lin_vel", 9, 12),
    ("base_ang_vel", 12, 15), ("joint_pos", 15, 44),
)






def fit_energy(d: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares project ``d`` onto ``span(B)`` along the node axis.

    Args:
        d: ``[W, F, C]`` the correction to represent.
        b: ``[F, K]`` the basis.

    Returns:
        ``(resid, recon)`` both ``[W, F, C]`` -- the unexplained part and the projection.
    """
    w, f, c = d.shape
    flat = d.transpose(1, 0, 2).reshape(f, w * c)          # (F, W*C)
    u, *_ = np.linalg.lstsq(b, flat, rcond=None)           # (K, W*C)
    recon = (b @ u).reshape(f, w, c).transpose(1, 0, 2)
    return d - recon, recon


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--knots", default="13,21,31",
                    help="knot COUNTS to compare, each spanning the whole window")
    ap.add_argument("--allow_per_step_history", action="store_true")
    ap.add_argument("--fk_model_path", default=None,
                    help="MJCF for the dataset's FK. Defaults to this repo's copy; the "
                         "checkpoint stores an ABSOLUTE path from the box it trained on, "
                         "which does not exist elsewhere.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = load_fm_context(args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    # The stored path is absolute and belongs to the training box. Derive the replacement
    # from the package root rather than counting `..` hops -- moving a module has silently
    # broken such a path here before.
    from motion_gen.features import _REPO_ROOT  # noqa: PLC0415
    want = args.fk_model_path or str(
        pathlib.Path(_REPO_ROOT) / "transfer" / "sim" / "robots" / "g1" / "g1_perceptive.xml")
    if str(cfg.fk_model_path) != want:
        print(f"[fk] checkpoint says {cfg.fk_model_path}\n     using      {want}")
        cfg.fk_model_path = want
    print(f"[cfg] F={cfg.future_len} D={cfg.state_dim} cmd_stride={cfg.cmd_stride} "
          f"steps={cfg.num_sampling_steps} device={device}")

    data, meta = load_capped(args.data, args.cap, args.allow_per_step_history)
    _, val = build_train_val(data, meta, cfg)
    if val is None or len(val) == 0:
        raise SystemExit("No val split -- raise --cap.")
    loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)

    chunks = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            pred = sample_trajectory(
                ctx.model, batch["history"].to(device), batch["extero"].to(device),
                batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None,
                cfg.future_len, cfg.state_dim, cfg.num_sampling_steps,
            )
            p = ctx.state_norm.denormalize(pred)
            t = ctx.state_norm.denormalize(batch["state"].to(device))
            chunks.append((t - p).cpu().numpy())            # authored - generated, RAW units
    d = np.concatenate(chunks, 0).astype(np.float64)
    print(f"[data] d = authored - generated: {d.shape} (windows, nodes, channels)\n")

    scale = REFERENCE_WINDOW_RESIDUAL_SCALE
    print("=== CORRECTION MAGNITUDE (raw units), against the residual cap ===")
    print(f"  {'group':14s} {'RMS':>10s} {'p90':>10s} {'cap':>8s} {'RMS/cap':>9s}")
    for name, a, b_ in GROUPS:
        g = d[:, :, a:b_]
        cap = np.atleast_1d(np.asarray(scale[name], dtype=float)).mean()
        rms = float(np.sqrt(np.mean(g ** 2)))
        p90 = float(np.percentile(np.abs(g), 90))
        print(f"  {name:14s} {rms:10.4f} {p90:10.4f} {cap:8.3f} {rms / cap:9.3f}")

    for kind in ("linear", "catmull"):
        print(f"\n=== R^2 of the {kind.upper()} knot basis (fraction of correction energy "
              f"captured) ===")
        hdr = f"  {'group':14s}" + "".join(
            f"{'K=' + s:>10s}" for s in args.knots.split(","))
        print(hdr)
        bases = {}
        for s in args.knots.split(","):
            kn = knot_positions(cfg.future_len, int(s))
            bases[s] = (kn, knot_basis(cfg.future_len, int(s), kind))
        fits = {s: fit_energy(d, bmat) for s, (kn, bmat) in bases.items()}
        for name, a, b_ in GROUPS:
            row = f"  {name:14s}"
            for s in args.knots.split(","):
                resid, _ = fits[s]
                num = np.sum(resid[:, :, a:b_] ** 2)
                den = np.sum(d[:, :, a:b_] ** 2)
                row += f"{1.0 - num / den:10.4f}"
            print(row)
        row = f"  {'ALL':14s}"
        for s in args.knots.split(","):
            resid, _ = fits[s]
            row += f"{1.0 - np.sum(resid ** 2) / np.sum(d ** 2):10.4f}"
        print(row)
        print("  knots: " + "  ".join(
            f"K={s} at {np.round(bases[s][0], 2).tolist()[:4]}..{np.round(bases[s][0], 2).tolist()[-1]}"
            for s in args.knots.split(",")[:1]))
        s0 = args.knots.split(",")[0]
        resid, _ = fits[s0]
        per_node = np.sqrt(np.mean(resid ** 2, axis=(0, 2)))
        tot_node = np.sqrt(np.mean(d ** 2, axis=(0, 2)))
        print(f"  unexplained RMS by node at K={s0}:")
        idx = list(range(0, cfg.future_len, 10)) + [cfg.future_len - 2, cfg.future_len - 1]
        print("    node " + "".join(f"{n:>8d}" for n in idx))
        print("    resid" + "".join(f"{per_node[n]:8.4f}" for n in idx))
        print("    total" + "".join(f"{tot_node[n]:8.4f}" for n in idx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
