# Copyright 2026 Zachary Olkin. All rights reserved.

"""Aggregate two measurements over many windows: did the augmentation take, and does the
generator overshoot downrange?

Both come out of the same forward passes, so they share one pass over the data.

**(a) Did the augmentation take?** For a source window and its displaced augments, compare the
prediction against its OWN target and against the SOURCE's. Restricted to the nodes where the
two targets actually DIFFER -- elsewhere the comparison is vacuous, and reading a convergence
of the two curves as a model property when it is a property of the targets is exactly the
mistake this restriction exists to prevent. ``own << source`` means the displaced depth
changed the output.

**(b) Does it overshoot?** On a single window the prediction ran ~14 cm further downrange than
the clip over 1.24 s -- identically across all augments, so it is a systematic bias rather than
a registration effect, and ~15% more path length. The candidate mechanism is the command
channel: the recorded reference runs well below the commanded velocity (mean 0.727 vs 1.268
m/s) and slows hard on terrain while the command does not, so a model that over-trusts the
command would overshoot by exactly this signature. Reported against ``|cmd| - |ref|`` at the
snap, which tests that directly.

    python scripts/flow_matching/fm_augment_aggregate.py \\
        --checkpoint logs/flow_matching/<run>/best.pt \\
        --data <original>.h5 --augmented <augmented>.h5 --num_sources 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "source", "robot_rl"))
sys.path.insert(0, os.path.join(_REPO, "source", "motion_gen"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))

from recorded_data_io import RecordingMeta  # noqa: E402
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import FutureStateDataset  # noqa: E402
from robot_rl.flow_matching.flow import sample_trajectory  # noqa: E402
from robot_rl.flow_matching.online import load_fm_context  # noqa: E402

#: Node-level target separation below which the own-vs-source comparison carries no signal.
_SEP_FLOOR_M = 0.02


def read_meta(path: str) -> RecordingMeta:
    """RecordingMeta from an .h5's root attrs, without loading any window arrays."""
    import h5py

    with h5py.File(path, "r") as f:
        raw = {k: json.loads(v) for k, v in f.attrs.items()
               if k not in ("num_windows", "num_windows_written")}
    names = {fld.name for fld in fields(RecordingMeta)}
    return RecordingMeta(**{k: v for k, v in raw.items() if k in names})


def gather_groups(orig_path: str, aug_path: str, num_sources: int, seed: int):
    """Pick source windows that have augments, and read ONLY those rows from both files.

    Reading row-wise rather than via ``load_recorded_windows`` keeps this off the ~22 GB the
    two files would otherwise pull into RAM.

    Args:
        orig_path: Original recording.
        aug_path: Its augmented counterpart.
        num_sources: How many source windows to sample.
        seed: RNG seed.

    Returns:
        ``(data, groups)`` where ``data`` is a row-subset dict over
        ``[sources..., augments...]`` and ``groups`` maps each source's position to the
        positions of its augments.
    """
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(orig_path, "r") as fo, h5py.File(aug_path, "r") as fa:
        n_o = int(fo.attrs.get("num_windows_written", fo.attrs["num_windows"]))
        n_a = int(fa.attrs.get("num_windows_written", fa.attrs["num_windows"]))
        o_key = np.stack([fo["episode_id"][:n_o], fo["clip_step"][:n_o]], 1)
        a_key = np.stack([fa["episode_id"][:n_a], fa["clip_step"][:n_a]], 1)
        # augment rows bucketed by their source's key
        buckets: dict[tuple, list[int]] = {}
        for j, (e, s) in enumerate(a_key):
            buckets.setdefault((int(e), int(s)), []).append(j)
        cand = [i for i in rng.permutation(n_o)
                if (int(o_key[i, 0]), int(o_key[i, 1])) in buckets]
        srcs = sorted(cand[:num_sources])
        agg = [buckets[(int(o_key[i, 0]), int(o_key[i, 1]))] for i in srcs]

        keys = [k for k in fo.keys() if fo[k].shape[0] == n_o]
        data = {}
        flat_aug = [j for g in agg for j in g]
        for k in keys:
            data[k] = np.concatenate([fo[k][srcs], fa[k][sorted(flat_aug)]])
        order = {j: p for p, j in enumerate(sorted(flat_aug))}
        groups = []
        for gi, g in enumerate(agg):
            groups.append((gi, [len(srcs) + order[j] for j in g]))
    return data, groups


@torch.no_grad()
def main() -> None:
    """Run the model over all sampled rows and report both aggregates."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--augmented", required=True)
    ap.add_argument("--num_sources", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Optional .npz dump of the raw arrays.")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = read_meta(args.data)
    data, groups = gather_groups(args.data, args.augmented, args.num_sources, args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    cfg.transition_fraction = None
    ctx = load_fm_context(args.checkpoint, "cpu")     # normalizers on CPU, dataset packs on CPU
    model = ctx.model.to(dev)

    n_rows = data["history_state"].shape[0]
    ds = FutureStateDataset(data, meta, cfg, np.arange(n_rows),
                            state_normalizer=ctx.state_norm,
                            cmd_normalizer=getattr(ctx, "cmd_norm", None),
                            height_normalizer=getattr(ctx, "height_norm", None))
    pos = ds.block_slices["base_pose"]
    pos = slice(pos.start, pos.start + 3)

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    P, T = [], []
    for b in loader:
        torch.manual_seed(args.seed)
        p = sample_trajectory(model, b["history"].to(dev), b["height"].to(dev),
                              b["cmd_tokens"].to(dev) if "cmd_tokens" in b else None,
                              cfg.future_len, cfg.state_dim, cfg.num_sampling_steps)
        P.append(ctx.state_norm.denormalize(p.cpu())[:, :, pos].numpy())
        T.append(ctx.state_norm.denormalize(b["state"])[:, :, pos].numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    print(f"[agg] {len(groups)} source windows, {n_rows} rows total")

    # ---- (a) did the augmentation take? -------------------------------------------------
    # Recover the APPLIED offset per augment row from the snap-anchor difference, and split it
    # into the terrain frame. This matters because the two components are not equivalent: on a
    # terrain segment LATERAL is a free DOF, so the target moves with the robot and the window
    # target barely changes -- those rows carry almost no signal and would dilute the
    # own-vs-source comparison toward 1.0 regardless of what the model does. Only the
    # DOWNRANGE/yaw component produces a target the model must react to.
    H = data["history_state"].shape[1]
    anch = np.asarray(data["history_state"])[:, H - 1, 0:3]
    quat = np.asarray(data["history_state"])[:, H - 1, 3:7]
    hd = np.arctan2(2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
                    1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2))
    own, cross, seps, frac, d_dr, d_lat = [], [], [], [], [], []
    for si, a_idx in groups:
        for ai in a_idx:
            sep = np.linalg.norm(T[ai] - T[si], axis=1)           # (F,)
            m = sep > _SEP_FLOOR_M
            seps.append(sep)
            frac.append(m.mean())
            if m.sum() < 3:
                continue
            dw = anch[ai] - anch[si]                              # world displacement
            c, sn = np.cos(hd[si]), np.sin(hd[si])
            d_dr.append(dw[0] * c + dw[1] * sn)                   # into the robot's heading
            d_lat.append(-dw[0] * sn + dw[1] * c)
            own.append(np.linalg.norm(P[ai] - T[ai], axis=1)[m].mean())
            cross.append(np.linalg.norm(P[ai] - T[si], axis=1)[m].mean())
    own, cross = np.array(own), np.array(cross)
    d_dr, d_lat = np.abs(np.array(d_dr)), np.abs(np.array(d_lat))
    seps = np.stack(seps)
    print("\n=== (a) augmentation: prediction vs OWN target vs the SOURCE's ===")
    print(f"  usable augment rows (>=3 discriminating nodes): {len(own)}")
    print(f"  mean fraction of nodes with separation > {_SEP_FLOOR_M} m : {np.mean(frac):.2f}")
    print(f"  mean target separation on those nodes              : {seps[seps>_SEP_FLOOR_M].mean():.4f} m")
    print(f"  error vs OWN target                                : {own.mean():.4f} m")
    print(f"  error vs SOURCE target                             : {cross.mean():.4f} m")
    print(f"  ratio own/source                                   : {own.mean()/max(cross.mean(),1e-9):.3f}"
          "   (<1 = uses the displaced depth; ~1 = ignores it)")
    print(f"  rows where own < source                            : {100*(own<cross).mean():.1f}%")
    print("\n  stratified by the DOWNRANGE component of the applied offset")
    print(f"  {'|d_downrange| (m)':>20s} {'n':>6s} {'own':>8s} {'source':>8s} {'ratio':>7s} {'own<src':>8s}")
    qs = np.quantile(d_dr, [0, .25, .5, .75, 1.])
    for lo, hi in zip(qs[:-1], qs[1:]):
        mm = (d_dr >= lo) & (d_dr <= hi)
        if mm.sum() < 20:
            continue
        print(f"  {lo:8.3f}-{hi:8.3f} {int(mm.sum()):6d} {own[mm].mean():8.4f} "
              f"{cross[mm].mean():8.4f} {own[mm].mean()/max(cross[mm].mean(),1e-9):7.3f} "
              f"{100*(own[mm]<cross[mm]).mean():7.0f}%")

    # ---- (b) downrange overshoot vs the command gap -------------------------------------
    dx = P[:, -1, 0] - T[:, -1, 0]
    dz = P[:, -1, 2] - T[:, -1, 2]
    path_p = np.linalg.norm(np.diff(P, axis=1), axis=2).sum(1)
    path_t = np.linalg.norm(np.diff(T, axis=1), axis=2).sum(1)
    cmd = np.linalg.norm(np.asarray(data["cmd_future"])[:, 0, :2], axis=1)
    ref = np.linalg.norm(np.asarray(data["ref_future"])[:, 0, 7:9], axis=1)
    gap = cmd - ref
    print("\n=== (b) final-node bias, and the command/reference gap ===")
    print(f"  downrange dx : mean {dx.mean():+.4f} m   p50 {np.median(dx):+.4f}   "
          f"frac>0 {100*(dx>0).mean():.0f}%")
    print(f"  vertical  dz : mean {dz.mean():+.4f} m")
    print(f"  path length  : pred {path_p.mean():.3f} m vs target {path_t.mean():.3f} m "
          f"({100*(path_p.mean()/path_t.mean()-1):+.1f}%)")
    print(f"  corr(cmd-ref gap, dx) = {np.corrcoef(gap, dx)[0,1]:+.3f}   "
          f"corr(cmd, dx) = {np.corrcoef(cmd, dx)[0,1]:+.3f}   "
          f"corr(ref, dx) = {np.corrcoef(ref, dx)[0,1]:+.3f}")
    q = np.quantile(gap, [0, .25, .5, .75, 1.])
    print(f"  {'cmd-ref gap (m/s)':>20s} {'n':>6s} {'mean dx (m)':>12s}")
    for lo, hi in zip(q[:-1], q[1:]):
        m = (gap >= lo) & (gap <= hi)
        if m.sum() > 10:
            print(f"  {lo:8.2f}-{hi:8.2f} {int(m.sum()):6d} {dx[m].mean():+12.4f}")
    if args.out:
        np.savez_compressed(args.out, preds=P, targets=T, dx=dx, gap=gap, own=own, cross=cross)
        print(f"\n[agg] wrote {args.out}")


if __name__ == "__main__":
    main()
