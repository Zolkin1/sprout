#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Diagnostic plot: WHEN a spliced clip switches between its sub-clips (segments).

A terrain clip is spliced from several sub-clips -- a DP ``approach`` walk, the played terrain
``traversal`` (climb / descent / jump), and a DP ``recovery`` walk -- eased together at seams.  The
per-frame ANCHOR MASK changes value at each seam (flat root-DOF mask on the flat segments, the
terrain mask on the traversal), so its transitions are exactly the sub-clip switch points.  This
tool marks those switches on a side view and a time axis so you can see, e.g., how long the robot
walks before it actually steps onto the stairs (is the "transition onto the stair" too late?).

Reads only the ``.h5`` (no IsaacLab).  Respects each clip's ``clip_len`` (ignores the padding tail).

Example::

    PYTHONPATH=source/robot_rl:source/motion_gen env_isaaclab/bin/python \
        scripts/rsl_rl/plot_clip_segments.py --lib clips/_su_fix.h5 --clip 1 --out plots/su_seg.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")                                       # headless (write a PNG, no display needed)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_C = {"approach": "#0077BB", "traversal": "#EE7733", "recovery": "#009988",
      "pelvis": "#0077BB", "lfoot": "#EE7733", "rfoot": "#009988",
      "terrain": "#888888", "switch": "#CC3311"}


def _surface_xz(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Top-surface step profile (x, z) from the overlay boxes -- treads, platforms, and jump boxes."""
    ov = manifest.get("overlays", [])
    segs = []
    for o in ov:
        cx = float(o["pos_xy"][0]); sx = float(o["size_xyz"][0])
        top = float(o["base_z"]) + float(o["size_xyz"][2])
        segs.append((cx - sx / 2.0, cx + sx / 2.0, top))
    segs.sort()
    xs, zs = [], []
    for x0, x1, top in segs:
        xs += [x0, x1]; zs += [top, top]
    return np.array(xs), np.array(zs)


def _segments(anchor: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs of a constant anchor-mask value -> list of (start, end_exclusive, mask)."""
    runs = []
    s = 0
    for t in range(1, len(anchor)):
        if anchor[t] != anchor[t - 1]:
            runs.append((s, t, int(anchor[s]))); s = t
    runs.append((s, len(anchor), int(anchor[s])))
    return runs


def _label_runs(runs: list[tuple[int, int, int]]) -> list[str]:
    """Name runs: a single terrain run is the 'traversal'; with flats around it they are approach/recovery."""
    if len(runs) == 1:
        return ["traversal"]
    labels = []
    seen_traversal = False
    # the terrain (traversal) run is the one whose mask differs from the first (flat) mask
    flat_mask = runs[0][2]
    for (_a, _b, m) in runs:
        if m != flat_mask and not seen_traversal:
            labels.append("traversal"); seen_traversal = True
        elif not seen_traversal:
            labels.append("approach")
        else:
            labels.append("recovery")
    return labels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5.")
    p.add_argument("--clip", type=int, default=None, help="Clip index (default: first 'transition' clip).")
    p.add_argument("--terrain", default="transition", help="If --clip omitted, pick the first clip of this type.")
    p.add_argument("--out", default=None, help="Output PNG path (default plots/clip_segments_<clip>.png).")
    args = p.parse_args()

    from motion_gen.clips.clip_io import load_clip_library
    lib = load_clip_library(args.lib, "cpu")
    meta = json.loads(lib.meta) if isinstance(lib.meta, str) else lib.meta
    pn = meta["pos_output_names"]

    def col(sfx):
        return [i for i, n in enumerate(pn) if n.endswith(sfx)][0]
    px, pz = col("pelvis_link:pos_x"), col("pelvis_link:pos_z")
    lax, laz = col("left_ankle_roll_link:pos_x"), col("left_ankle_roll_link:pos_z")
    rax, raz = col("right_ankle_roll_link:pos_x"), col("right_ankle_roll_link:pos_z")

    tnames = lib.attrs.get("terrain_type_names", ["flat"])
    tnames = json.loads(tnames) if isinstance(tnames, str) else tnames
    codes = np.asarray(lib.terrain_type_code).reshape(-1)
    ci = args.clip
    if ci is None:
        want = tnames.index(args.terrain) if args.terrain in tnames else 2
        cand = np.where(codes == want)[0]
        ci = int(cand[0]) if cand.size else 0
    dt = float(meta["dt"])
    T = int(np.asarray(lib.clip_len).reshape(-1)[ci])            # real length (drop padding tail)
    y = np.asarray(lib.y_des[ci])[:T]
    anchor = np.asarray(lib.anchor_mask_t)[ci, :T]
    t = np.arange(T) * dt

    runs = _segments(anchor)
    labels = _label_runs(runs)
    tid = int(np.asarray(lib.terrain_id).reshape(-1)[ci])
    ttype = tnames[codes[ci]] if codes.size else "?"
    raw = lib.attrs.get("terrain_manifest"); raw = json.loads(raw) if isinstance(raw, str) else (raw or [])
    man = next((m for m in raw if int(m["terrain_id"]) == tid), None)

    # --- report to stdout ---
    print(f"[plot_clip_segments] clip {ci} [{ttype}, terrain {tid}], {T} frames = {T*dt:.2f}s, {len(runs)} sub-clips:")
    for (a, b, m), lab in zip(runs, labels):
        print(f"   {lab:10s} frames [{a:4d},{b:4d})  t=[{a*dt:5.2f},{b*dt:5.2f})s  "
              f"pelvis-x [{y[a, px]:.2f},{y[b-1, px]:.2f}]  pelvis-z [{y[a, pz]:.2f},{y[b-1, pz]:.2f}]")

    # --- figure ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 2])
    seg_col = {"approach": _C["approach"], "traversal": _C["traversal"], "recovery": _C["recovery"]}

    # Panel 0: side view (forward x vs height z), terrain + bodies, sub-clips coloured, switches marked.
    if man is not None:
        sx, sz = _surface_xz(man)
        ax0.plot(sx, sz, color=_C["terrain"], lw=4, zorder=1, label="terrain")
        ax0.fill_between(sx, sz, sz.min() - 0.2, color=_C["terrain"], alpha=0.2, zorder=0)
    else:
        ax0.axhline(0.0, color=_C["terrain"], lw=4, label="terrain")
    for (a, b, _m), lab in zip(runs, labels):
        ax0.plot(y[a:b, px], y[a:b, pz], color=seg_col.get(lab, "#333"), lw=3,
                 label=f"pelvis:{lab}", zorder=3)
    ax0.plot(y[:, lax], y[:, laz], color="#000", lw=1.0, alpha=0.5, zorder=2)
    ax0.plot(y[:, rax], y[:, raz], color="#000", lw=1.0, alpha=0.5, zorder=2, label="feet")
    for (a, _b, _m) in runs[1:]:                                  # switch markers
        ax0.axvline(y[a, px], color=_C["switch"], ls="--", lw=1.5, alpha=0.8)
        ax0.annotate(f"switch\n{t[a]:.2f}s", (y[a, px], ax0.get_ylim()[1]),
                     color=_C["switch"], fontsize=9, ha="center", va="top")
    ax0.set_xlabel("forward x (m)"); ax0.set_ylabel("height z (m)")
    ax0.set_aspect("equal", adjustable="datalim"); ax0.legend(loc="best", fontsize=9)
    ax0.set_title(f"{args.lib.split('/')[-1]}  clip {ci} [{ttype}]  -- sub-clip switches")

    # Panel 1: time axis -- sub-clip bands + pelvis z, switch lines labelled with the time.
    for (a, b, _m), lab in zip(runs, labels):
        ax1.axvspan(t[a], t[min(b, T - 1)], color=seg_col.get(lab, "#333"), alpha=0.18)
        ax1.text((t[a] + t[min(b - 1, T - 1)]) / 2.0, 0.04, lab, transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=10, color=seg_col.get(lab, "#333"))
    ax1.plot(t, y[:, pz], color=_C["pelvis"], lw=2.5, label="pelvis z")
    ax1.plot(t, np.minimum(y[:, laz], y[:, raz]), color="#000", lw=1.2, alpha=0.6, label="lower foot z")
    for (a, _b, _m) in runs[1:]:
        ax1.axvline(t[a], color=_C["switch"], ls="--", lw=1.5)
        ax1.annotate(f"{t[a]:.2f}s", (t[a], ax1.get_ylim()[1]), color=_C["switch"], fontsize=9,
                     ha="center", va="top")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("height z (m)"); ax1.legend(loc="best", fontsize=9)

    out = Path(args.out) if args.out else Path("plots") / f"clip_segments_{ci}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[plot_clip_segments] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
