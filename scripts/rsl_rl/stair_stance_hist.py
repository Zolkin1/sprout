#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where the stance foot is PLANTED relative to the first stair edge, library vs training data.

A variant of ``stair_edge_stats.py`` asking a narrower question: at the moment a foot is
stationary on the ground, how far is it from the nose of the first tread -- the step it is
about to climb? That is the placement decision, and unlike a swing trajectory it is a single
number per stance, so its distribution is readable.

Two populations, on the same axes:

* **library** -- every stance in every stairs-up transition clip. What the data CONTAINS.
* **datapoints** -- only the stances a recorded FM training window actually covers, i.e. what
  the model was SHOWN. A window spans ``[clip_step, clip_step + F)``, so coverage is a count
  per clip frame and a stance inherits the count of the frame it happens on.

If the two distributions agree, the training windows are a fair sample of the library on this
axis and any placement bias is in the clips, not the sampling. If they disagree, the sampling
is selecting placements -- worth knowing before concluding anything about the model.

A stance is one contiguous planted run of one foot, contributing ONE event at its median
position: the foot does not move while planted, so per-frame values would just weight each
placement by how long it was stood on.

Example::

    python scripts/rsl_rl/stair_stance_hist.py --lib clips/g1_mb_v5_body.h5 \
        --data logs/rsl_rl/g1/<run>/recorded_data/recorded_20260809_152805.h5 \
        --out plots/stance_edges
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # before pyplot: this runs headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in (os.path.join(_REPO, "source", "robot_rl"), os.path.join(_REPO, "source", "motion_gen"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stair_edge_stats import (  # noqa: E402
    TileGeometry,
    contact_point,
    planted_mask,
)


def stance_runs(planted: np.ndarray, min_len: int = 3) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` runs of True, at least ``min_len`` long.

    The minimum drops single-frame flickers of the geometric stance test, which would
    otherwise contribute placements the robot never actually stood on.
    """
    runs, start = [], None
    for i, v in enumerate(planted):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(planted) - start >= min_len:
        runs.append((start, len(planted)))
    return runs


def window_coverage(data_paths: list[str], num_clips: int, num_steps: int, horizon: int
                    ) -> np.ndarray:
    """``[N, T]`` count of recorded training windows covering each clip frame.

    A window at ``clip_step`` spans its own ``F`` nodes, so it covers
    ``[clip_step, clip_step + F)``. Accumulated with a difference array rather than a loop over
    frames -- at 200k windows x 62 nodes the direct form is minutes.
    """
    import h5py

    diff = np.zeros((num_clips, num_steps + 1), dtype=np.int32)
    total = 0
    for path in data_paths:
        with h5py.File(path, "r") as f:
            n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
            if not n or "clip_id" not in f:
                raise SystemExit(f"{path} has no clip_id/clip_step index; it cannot be joined "
                                 "to the library (re-record with the current record_data.py).")
            cid = f["clip_id"][:n].astype(np.int64)
            cst = f["clip_step"][:n].astype(np.int64)
        keep = (cid >= 0) & (cid < num_clips)
        cid, cst = cid[keep], cst[keep]
        lo = np.clip(cst, 0, num_steps)
        hi = np.clip(cst + horizon, 0, num_steps)
        np.add.at(diff, (cid, lo), 1)
        np.add.at(diff, (cid, hi), -1)
        total += int(keep.sum())
        print(f"[stance_hist]   {os.path.basename(path)}: {keep.sum()} windows joined")
    return np.cumsum(diff[:, :-1], axis=1)


def collect_onset(lib_path: str, family: str, point: str, cover: np.ndarray | None,
                  lookback: int = 12) -> dict:
    """One event per clip: the STANCE foot's distance to the first nose at the mask boundary.

    The boundary is where ``anchor_mask_t`` first goes flat -> terrain-pinned, i.e. the frame
    the clip commits to the staircase. At that instant one foot is (usually) planted and the
    other is swinging onto the first tread, so the stance foot's distance to the nose is the
    TAKE-OFF geometry of that step -- how far back the support foot is when the swing begins.

    Which foot is the stance one is decided at the boundary frame: if exactly one is planted it
    is that one; if both are (double support) it is the one still planted longer afterwards,
    i.e. the one that will actually support the swing; if neither is, the search walks back up
    to ``lookback`` frames to the most recent plant, and the clip is skipped if there is none.
    The other foot's distance is recorded too, as ``d_swing``.

    Returns:
        The same column layout as :func:`collect` plus ``d_swing``, ``both_planted`` and
        ``onset`` -- one row per clip rather than per stance.
    """
    import h5py

    from motion_gen.clips.clip_io import load_clip_library
    from motion_gen.motionbricks.terrain_families import sole_heights
    from motion_gen.world_lift import anchor_mask_is_pinned

    with h5py.File(lib_path, "r") as f:
        dt = json.loads(f.attrs["dt"])
        pos_names = json.loads(f.attrs["pos_output_names"])
        type_names = json.loads(f.attrs["terrain_type_names"])
        manifest = json.loads(f.attrs["terrain_manifest"])
        terrain_id = f["terrain_id"][:].astype(np.int64)
        type_code = f["terrain_type_code"][:].astype(np.int64)
    by_tid: dict[int, dict] = {}
    for e in manifest:
        by_tid.setdefault(int(e["terrain_id"]), e)
    fam = np.array([str(by_tid.get(int(t), {}).get("family", "")) for t in terrain_id],
                   dtype=object)
    clips = np.nonzero((type_code == type_names.index("transition")) & (fam == family))[0]
    if clips.size == 0:
        raise SystemExit(f"no {family!r} transition clips in {lib_path}.")
    print(f"[stance_hist] {clips.size} clips, point={point}, event=onset")

    lib = load_clip_library(lib_path, device="cpu",
                            keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    pinned_all = anchor_mask_is_pinned(lib.anchor_mask_per_step()).numpy()
    clip_len = lib.clip_len_or_full().numpy()
    geoms = {t: TileGeometry(e) for t, e in by_tid.items() if t in set(terrain_id[clips])}
    frames = ("left_ankle_roll_link", "right_ankle_roll_link")
    out = {k: [] for k in ("clip", "foot", "frame", "d_first", "d_near", "d_swing", "sole",
                           "pinned", "cover", "run_len", "both_planted", "onset")}
    skipped = 0
    with h5py.File(lib_path, "r") as f:
        for n, ci in enumerate(clips):
            if n % 200 == 0:
                print(f"  ... {n}/{clips.size}", flush=True)
            length = int(clip_len[ci])
            pin = pinned_all[ci, :length]
            edges = np.nonzero((~pin[:-1]) & pin[1:])[0]
            geom = geoms[int(terrain_id[ci])]
            if edges.size == 0 or not np.isfinite(geom.first_riser):
                skipped += 1
                continue
            onset = int(edges[0]) + 1
            first_nose = geom.first_riser        # the first edge that is a STEP
            y = f["y_des"][ci, :length, :].astype(np.float64)
            soles = sole_heights(y, pos_names, frames=frames)
            fx = np.empty((length, 2))
            planted = np.empty((length, 2), dtype=bool)
            for j, fr in enumerate(frames):
                fx[:, j], fy = contact_point(y, pos_names, fr, point)
                planted[:, j] = planted_mask(soles[:, j], geom.ground_z(fx[:, j], fy), dt)
            frame = onset
            live = np.nonzero(planted[onset])[0]
            if live.size == 0:
                # Nobody planted exactly at the boundary: walk back to the most recent plant.
                back = [t for t in range(onset - 1, max(onset - lookback, -1), -1)
                        if planted[t].any()]
                if not back:
                    skipped += 1
                    continue
                frame = back[0]
                live = np.nonzero(planted[frame])[0]
            both = live.size > 1
            if both:
                # Double support: the supporting foot is the one that stays down longer.
                after = [int(np.argmin(planted[frame:, j])) if not planted[frame:, j].all()
                         else length - frame for j in live]
                j = int(live[int(np.argmax(after))])
            else:
                j = int(live[0])
            other = 1 - j
            near, _ = geom.edge_distances(np.array([fx[frame, j]]))
            out["clip"].append(int(ci))
            out["foot"].append(j)
            out["frame"].append(frame)
            out["onset"].append(onset)
            out["d_first"].append(first_nose - float(fx[frame, j]))
            out["d_near"].append(float(near[0]))
            out["d_swing"].append(first_nose - float(fx[frame, other]))
            out["sole"].append(float(soles[frame, j]))
            out["pinned"].append(bool(pin[frame]))
            out["run_len"].append(1)
            out["both_planted"].append(bool(both))
            out["cover"].append(int(cover[ci, frame]) if cover is not None else 0)
    res = {k: np.asarray(v) for k, v in out.items()}
    res["dt"] = np.asarray(dt)
    res["point"] = np.asarray(point)
    res["num_clips"] = np.asarray(clips.size)
    print(f"[stance_hist] {res['clip'].size} onsets, {skipped} clips skipped "
          f"(no boundary, or no plant within {lookback} frames of it)")
    return res


def collect(lib_path: str, family: str, point: str, cover: np.ndarray | None) -> dict:
    """Per-stance-event distance to the first tread nose, with its window-coverage count."""
    import h5py

    from motion_gen.clips.clip_io import load_clip_library
    from motion_gen.motionbricks.terrain_families import sole_heights
    from motion_gen.world_lift import anchor_mask_is_pinned

    with h5py.File(lib_path, "r") as f:
        dt = json.loads(f.attrs["dt"])
        pos_names = json.loads(f.attrs["pos_output_names"])
        type_names = json.loads(f.attrs["terrain_type_names"])
        manifest = json.loads(f.attrs["terrain_manifest"])
        terrain_id = f["terrain_id"][:].astype(np.int64)
        type_code = f["terrain_type_code"][:].astype(np.int64)
    by_tid: dict[int, dict] = {}
    for e in manifest:
        by_tid.setdefault(int(e["terrain_id"]), e)
    fam = np.array([str(by_tid.get(int(t), {}).get("family", "")) for t in terrain_id],
                   dtype=object)
    clips = np.nonzero((type_code == type_names.index("transition")) & (fam == family))[0]
    if clips.size == 0:
        raise SystemExit(f"no {family!r} transition clips in {lib_path}.")
    print(f"[stance_hist] {clips.size} clips, point={point}")

    lib = load_clip_library(lib_path, device="cpu",
                            keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    pinned_all = anchor_mask_is_pinned(lib.anchor_mask_per_step()).numpy()
    clip_len = lib.clip_len_or_full().numpy()
    geoms = {t: TileGeometry(e) for t, e in by_tid.items() if t in set(terrain_id[clips])}
    frames = ("left_ankle_roll_link", "right_ankle_roll_link")

    out = {k: [] for k in ("clip", "foot", "frame", "d_first", "d_near", "sole", "pinned",
                           "cover", "run_len")}
    with h5py.File(lib_path, "r") as f:
        for n, ci in enumerate(clips):
            if n % 200 == 0:
                print(f"  ... {n}/{clips.size}", flush=True)
            length = int(clip_len[ci])
            geom = geoms[int(terrain_id[ci])]
            if not np.isfinite(geom.first_riser):
                continue
            first_nose = geom.first_riser        # the first edge that is a STEP
            y = f["y_des"][ci, :length, :].astype(np.float64)
            soles = sole_heights(y, pos_names, frames=frames)
            for j, fr in enumerate(frames):
                fx, fy = contact_point(y, pos_names, fr, point)
                ground = geom.ground_z(fx, fy)
                planted = planted_mask(soles[:, j], ground, dt)
                for a, b in stance_runs(planted):
                    mid = (a + b) // 2
                    x = float(np.median(fx[a:b]))
                    near, _ = geom.edge_distances(np.array([x]))
                    out["clip"].append(int(ci))
                    out["foot"].append(j)
                    out["frame"].append(mid)
                    out["d_first"].append(first_nose - x)
                    out["d_near"].append(float(near[0]))
                    out["sole"].append(float(np.median(soles[a:b, j])))
                    out["pinned"].append(bool(pinned_all[ci, mid]))
                    out["run_len"].append(b - a)
                    out["cover"].append(int(cover[ci, mid]) if cover is not None else 0)
    res = {k: np.asarray(v) for k, v in out.items()}
    res["dt"] = np.asarray(dt)
    res["point"] = np.asarray(point)
    res["num_clips"] = np.asarray(clips.size)
    return res


def plot(res: dict, out_dir: str, span: float, has_data: bool,
         edge: str = "first", x_min: float | None = None,
         x_max: float | None = None) -> list[str]:
    """Library vs datapoint-covered stance placements, and the coverage itself.

    ``edge`` picks the axis, and the two answer different questions:

    * ``first`` -- distance to the FIRST tread nose. Unbounded: it grows as the robot walks up,
      so it reads as position along the climb.
    * ``nearest`` -- distance to whichever nose is closest. Bounded by half a tread spacing once
      the foot is on the staircase, so it isolates WHERE ON A TREAD the foot lands -- the
      placement phase, independent of which step.
    """
    os.makedirs(out_dir, exist_ok=True)
    d = res["d_first" if edge == "first" else "d_near"]
    lo = -span if x_min is None else float(x_min)
    hi = span if x_max is None else float(x_max)
    keep = (d >= lo) & (d <= hi)
    fig, axes = plt.subplots(1, 2 if has_data else 1, figsize=(13 if has_data else 7, 4.4),
                             squeeze=False)
    ax = axes[0][0]
    bins = np.linspace(lo, hi, 90)
    ax.hist(d[keep], bins=bins, color="#0077BB", alpha=0.85, density=True,
            label=f"library ({int(keep.sum())} stances)")
    if has_data:
        cov = res["cover"] > 0
        sel = keep & cov
        # Weighted by coverage count: a frame covered by 40 windows was shown 40 times, and
        # the question is what the MODEL saw, not what the library holds.
        ax.hist(d[sel], bins=bins, weights=res["cover"][sel].astype(float), density=True,
                histtype="step", lw=2.2, color="#EE7733",
                label=f"datapoints ({int(res['cover'][sel].sum())} window-covers)")
    ax.axvline(0.0, color="#CC3311", ls="--", lw=1.2, label=f"{edge} tread nose")
    point = str(res.get("point", "toe"))
    ax.set_xlabel(f"{point.upper()} distance to the "
                  f"{'FIRST' if edge == 'first' else 'NEAREST'} tread nose [m]"
                  f"   (+ = nose ahead of the {point})")
    ax.set_ylabel("density")
    ax.set_title(f"stance {point} placement, {int(res['num_clips'])} stairs-up transition "
                 f"clips  (foot offsets from the ankle: toe +0.138, heel -0.066 m)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if has_data:
        ax2 = axes[0][1]
        c = res["cover"].astype(float)
        ax2.hist(c, bins=40, color="#009988")
        ax2.set_xlabel("training windows covering the stance frame")
        ax2.set_ylabel("stances")
        ax2.set_title(f"coverage: {100.0 * np.mean(c > 0):.0f}% of stances are in >=1 window",
                      fontsize=10)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir,
                     f"stance_{res.get('event', 'stance')}_{point}_{edge}_edge_hist.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def report(res: dict, span: float, has_data: bool, edge: str = "first",
           x_min: float | None = None, x_max: float | None = None) -> str:
    """Percentiles of the placement distribution, library vs datapoints."""
    d, cov = res["d_first" if edge == "first" else "d_near"], res["cover"]
    lo = -span if x_min is None else float(x_min)
    hi = span if x_max is None else float(x_max)
    lines = ["", "=" * 78,
             f"STANCE {str(res.get('point', 'toe')).upper()} -> {edge.upper()} TREAD NOSE",
             "=" * 78,
             f"  {int(d.size)} stance events over {int(res['num_clips'])} clips "
             f"({int(np.sum((d >= lo) & (d <= hi)))} within [{lo:+.2f}, {hi:+.2f}] m)"]

    def stats(name, mask, w=None):
        v = d[mask]
        if v.size == 0:
            return f"  {name:<12} (none)"
        q = (np.percentile(v, [5, 25, 50, 75, 95]) if w is None
             else _wq(v, w[mask], [5, 25, 50, 75, 95]))
        return (f"  {name:<12} p05 {q[0]:+.3f}  p25 {q[1]:+.3f}  p50 {q[2]:+.3f}  "
                f"p75 {q[3]:+.3f}  p95 {q[4]:+.3f}   n={v.size}")

    def _wq(v, w, ps):
        """Weighted percentiles."""
        o = np.argsort(v)
        v, w = v[o], np.asarray(w, float)[o]
        c = np.cumsum(w) - 0.5 * w
        return np.interp(np.asarray(ps) / 100.0 * w.sum(), c, v)

    near = (d >= lo) & (d <= hi)
    lines.append(stats("library", near))
    if has_data:
        lines.append(stats("datapoints", near & (cov > 0), w=cov))
        lines.append(f"  stances covered by >=1 window: {100.0 * np.mean(cov > 0):.1f}%   "
                     f"mean cover {cov.mean():.1f} windows")
    within = np.abs(d) < 0.05
    lines.append(f"  stances planted within 5 cm of the first nose: {int(within.sum())} "
                 f"({100.0 * within.mean():.2f}%)")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    """Parse args, collect stance events, write the figure."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5.")
    p.add_argument("--data", action="append", default=None,
                   help="Recording .h5 to compare against (repeatable). Omit for library only.")
    p.add_argument("--family", default="family_stairs_up.h5", help="Tile family.")
    p.add_argument("--point", default="toe", choices=("ankle", "toe", "heel"),
                   help="Which point of the foot the distance is measured from.")
    p.add_argument("--horizon", type=int, default=None,
                   help="Window span in nodes for the coverage count (default: the "
                        "recording's own ref_future_len).")
    p.add_argument("--event", default="stance", choices=("stance", "onset"),
                   help="'stance' = every stance in the clips. 'onset' = ONE event per clip, at "
                        "the flat->terrain anchor boundary: where the supporting foot is when "
                        "the other swings onto the first tread.")
    p.add_argument("--edge", default="first", choices=("first", "nearest"),
                   help="Measure to the FIRST tread nose (position along the climb) or the "
                        "NEAREST one (where on a tread the foot lands). Both are always saved.")
    p.add_argument("--span", type=float, default=1.5, help="Histogram half-range in metres.")
    p.add_argument("--x_min", type=float, default=None,
                   help="Lower bound of the axis, overriding -span. Use e.g. -0.1 to keep only "
                        "the APPROACH: with the first-nose axis, values below that are stances "
                        "already up on the staircase, which answer a different question.")
    p.add_argument("--x_max", type=float, default=None,
                   help="Upper bound of the axis, overriding +span.")
    p.add_argument("--out", default="plots/stance_edges", help="Output directory.")
    args = p.parse_args()

    import h5py

    with h5py.File(args.lib, "r") as f:
        n_clips, n_steps = int(f["terrain_id"].shape[0]), int(json.loads(f.attrs["num_steps"]))
    cover = None
    if args.data:
        horizon = args.horizon
        if horizon is None:
            with h5py.File(args.data[0], "r") as f:
                horizon = int(json.loads(f.attrs["ref_future_len"]))
        print(f"[stance_hist] coverage over {len(args.data)} recording(s), horizon {horizon}")
        cover = window_coverage(args.data, n_clips, n_steps, horizon)
    res = (collect_onset(args.lib, args.family, args.point, cover)
           if args.event == 'onset' else
           collect(args.lib, args.family, args.point, cover))
    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out,
                       f"stance_{args.event}_{args.point}_{args.edge}_edge.npz")
    res['event'] = np.asarray(args.event)
    np.savez_compressed(npz, **res)
    print(report(res, args.span, bool(args.data), args.edge, args.x_min, args.x_max))
    for w in [npz] + plot(res, args.out, args.span, bool(args.data), args.edge,
                          args.x_min, args.x_max):
        print(f"[stance_hist] wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
