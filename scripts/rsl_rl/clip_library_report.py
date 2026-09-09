#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""What a generated clip library actually contains, as figures and one JSON.

`mb_qc` is the text report and `mb_qc --gate` is the pass/fail; this is the picture. It exists
because the defects that mattered most on this branch were found by LOOKING -- the stairs sprint, the
box-top walk-away, the wrapped-context blow-ups -- while every scalar in the QC report stayed clean.
A number tells you a library changed; a distribution tells you how.

ONE ``collect()`` pass over the ``.h5`` fills a ``report`` dict, then ``draw_all`` renders from that
dict alone. The split matters for two reasons: ``report.json`` lets a number be diffed between builds
without re-rendering, and a figure that cannot reach the library cannot quietly recompute a metric a
different way from the one ``mb_qc`` reports.

Kept in ONE file rather than the ``eval/`` two-file split: nothing outside this script consumes the
figures, so a separate module would be indirection without a second caller.

    python3 scripts/rsl_rl/clip_library_report.py --lib clips/g1_mb_v6b.h5 \
        --recipe configs/clip_gen/g1_mb_v6.yaml --out-dir logs/clip_report/v6b
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import (COLOR_CYCLE, apply_paper_style, figsize,  # noqa: E402
                        save_figure, tex_escape)

# The three motion kinds a terrain clip is made of, and the colours they carry everywhere in this
# report. They encode the pipeline's real structure, not a decorative choice: an exact looped stride
# is motion we KNOW is correct, an in-betweening is motion the model chose, and a replay is a solved
# trajectory played verbatim.
_KIND_COLOR = {"loop": COLOR_CYCLE[0], "between": COLOR_CYCLE[1], "replay": COLOR_CYCLE[3]}
_FLAT = -1


# --------------------------------------------------------------------------------- collect
def collect(lib_path: str, max_clips: int = 0) -> dict:
    """One pass over the library -> every number the figures need.

    Masked to ``clip_len`` throughout. That is not a detail: a 300-frame clip padded to 1000 repeats
    its last command 700 times, so an unmasked histogram counts that command ~4x and the marginal it
    produces is mostly padding.
    """
    with h5py.File(lib_path, "r") as f:
        man = json.loads(f.attrs["terrain_manifest"])
        fam_of = {int(t["terrain_id"]): t.get("family", t.get("type", "?")) for t in man}
        tile_of = {int(t["terrain_id"]): t for t in man}
        cl = f["clip_len"][:]
        tid = f["terrain_id"][:]
        has_regime = bool(f.attrs.get("has_regime", False))
        n = int(len(cl))
        idx = list(range(n))
        if max_clips and n > max_clips:
            idx = idx[:: max(1, n // max_clips)][:max_clips]
        dt = float(f.attrs["dt"])

        rows = []
        for i in idx:
            m = int(cl[i])
            t = int(tid[i])
            fam = fam_of.get(t, "flat") if t != _FLAT else "flat"
            cmd = f["target_vel"][i, :m, :]
            xy = f["y_des"][i, :m, 0:2]
            z = f["y_des"][i, :m, 2]
            sp = (np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt) if m > 1 else np.zeros(1)
            si = f["stride_idx"][i, :m] if has_regime else np.full(m, -1)
            am = f["anchor_mask_t"][i, :m]
            # kind per frame: the replay is world-pinned, a loop is an exact stride, else the model
            kind = np.where(am == 2, 2, np.where(si >= 0, 0, 1))
            rows.append(dict(
                clip=i, family=str(fam), terrain_id=t, n=m,
                cmd_vx=cmd[:, 0].astype(float), cmd_vy=cmd[:, 1].astype(float),
                cmd_wz=cmd[:, 2].astype(float),
                speed=sp.astype(float), kind=kind.astype(int),
                pelvis_above_foot=(z - np.minimum(f["y_des"][i, :m, 9],
                                                  f["y_des"][i, :m, 16])).astype(float),
                size_x=float(tile_of.get(t, {}).get("size_x", 0.0)),
            ))
    return {"lib": lib_path, "dt": dt, "n_clips": n, "rows": rows,
            "families": sorted({r["family"] for r in rows}),
            "manifest_types": sorted({str(t.get("type")) for t in man})}


def _cat(rows, key, fams=None):
    sel = [r for r in rows if fams is None or r["family"] in fams]
    return np.concatenate([r[key] for r in sel]) if sel else np.zeros(0)


# --------------------------------------------------------------------------------- figures
def fig_commanded_vx(rep: dict, recipe: dict | None):
    """THE HEADLINE. Commanded forward speed, time-weighted, flat family against the declared band.

    Time-weighted because a command that is held for 4 s is four times as much of what the generator
    sees as one held for 1 s, and masked to ``clip_len`` because padding repeats the last command.
    The declared envelope is drawn behind it, so a gap reads as a decision rather than as an
    accident -- the predecessor's marginal was 10.3 : 1 max/min with 3.0-3.6 m/s entirely empty.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize(2, 1))
    vx = _cat(rep["rows"], "cmd_vx", {"flat"})
    if recipe:
        lo, hi = recipe["flat_vx_range"]
        ax.axvspan(lo, hi, color="0.90", zorder=0, label="declared envelope")
    edges = np.arange(-1.2, 3.8, 0.2)
    moving = vx[np.abs(vx) > 1e-9]
    ax.hist(moving, bins=edges, weights=np.full(moving.size, 1.0 / max(moving.size, 1)),
            color=COLOR_CYCLE[0], edgecolor="none")
    zero = float(np.mean(np.abs(vx) <= 1e-9)) if vx.size else 0.0
    ax.set_xlabel(r"commanded $v_x$ (m/s)")
    ax.set_ylabel("share of moving time")
    h, _ = np.histogram(moving, bins=edges)
    sh = h / max(h.sum(), 1)
    nz = sh[sh > 0]
    ratio = sh.max() / nz.min() if nz.size else float("nan")
    cap = (f"Commanded forward speed on the flat family, time-weighted and masked to clip length. "
           f"Bins are 0.2 m/s. max/min bin share {ratio:.2f} over {int((sh == 0).sum())} empty bins; "
           f"a further {zero:.1%} of commanded time is an exact zero (the declared stand and "
           f"sidestep masses) and is excluded here, since it is a point mass rather than part of "
           f"the continuous marginal.")
    ax.legend(loc="upper right", fontsize=8) if recipe else None
    return fig, cap


def fig_speed_by_kind(rep: dict):
    """Root speed split by what KIND of motion the frame belongs to.

    The gate's evidence, and the figure to look at first after a build. An exact looped stride is
    motion we know is correct, so its distribution bounds what "correct" looks like; an in-betweening
    is the model's own choice and is where every blow-up this branch found actually lived. A gated
    build has an empty right tail on the middle distribution.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize(2, 1))
    names = {0: "exact loop", 1: "in-betweening", 2: "terrain replay"}
    for k in (0, 1, 2):
        v = np.concatenate([r["speed"][r["kind"][1:] == k] for r in rep["rows"]
                            if (r["kind"][1:] == k).any()] or [np.zeros(0)])
        if not v.size:
            continue
        ax.hist(v, bins=np.linspace(0, 6, 60), histtype="step", lw=2, density=True,
                label=f"{names[k]} (max {v.max():.2f})",
                color=_KIND_COLOR[["loop", "between", "replay"][k]])
    ax.set_xlabel("root speed (m/s)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    allsp = _cat(rep["rows"], "speed")
    return fig, ("Root speed by motion kind. Exact loops are replays of solved strides and bound "
                 f"what correct looks like; in-betweenings are the model's own choice. Library max "
                 f"{allsp.max():.2f} m/s. Every root-speed blow-up this branch found lived in the "
                 f"in-betweening distribution and came from one cause -- a hand-off phase whose "
                 f"context window wrapped the stride's closure seam.")


def fig_commands_per_clip(rep: dict):
    """How many distinct commands a clip carries, per family.

    The whole point of the command work: a terrain clip used to carry ONE command for its entire
    traversal, so command and motion always agreed exactly where the generator most needs to see
    them disagree.
    """
    import matplotlib.pyplot as plt

    fams = [f for f in rep["families"]]
    counts = [[len(np.unique(np.round(r["cmd_vx"], 6))) for r in rep["rows"] if r["family"] == f]
              for f in fams]
    fig, ax = plt.subplots(figsize=figsize(2, 1, aspect=0.5))
    ax.boxplot(counts, tick_labels=[tex_escape(f.replace("family_", "").replace(".h5", ""))
                                    for f in fams], vert=True)
    ax.set_ylabel(r"distinct $v_x$ / clip")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    return fig, ("Distinct commanded forward speeds per clip, by family. The pure-stair families "
                 "take a matcher-free generator with no command schedule, so one is correct there.")


def fig_lateral_yaw(rep: dict):
    """|vy| against |wz|, which must be an L: at most one of the two is ever non-zero.

    A constraint the solved library satisfies exactly -- zero of its cells carry both -- so mass off
    the axes would mean the sampler had invented a command no motion exists for.
    """
    import matplotlib.pyplot as plt

    vy, wz = np.abs(_cat(rep["rows"], "cmd_vy")), np.abs(_cat(rep["rows"], "cmd_wz"))
    fig, ax = plt.subplots(figsize=figsize(2, 1))
    # A 2-D HISTOGRAM, not a point per frame: a scatter here is one artist per commanded frame, which
    # on the full library is millions of them -- it produced a 5.6 MB SVG on a 64-clip smoke build
    # alone. The density also reads better, since the whole claim is about where the mass ISN'T.
    if vy.size:
        ax.hist2d(vy, wz, bins=[np.linspace(0, 0.7, 60), np.linspace(0, 0.9, 60)],
                  cmin=1, cmap="viridis")
    both = float(np.mean((vy > 1e-9) & (wz > 1e-9))) if vy.size else 0.0
    ax.set_xlabel(r"$|v_y|$ (m/s)")
    ax.set_ylabel(r"$|\omega_z|$ (rad/s)")
    return fig, (f"Lateral against yaw command, every frame. Must be an L -- {both:.2%} of frames "
                 f"carry both, against a target of 0. Zero of the solved library's cells carry both, "
                 f"so mass off the axes would be a command no motion exists for.")


def fig_pelvis(rep: dict):
    """Pelvis height above the LOWER FOOT, which is a robot property rather than a terrain one.

    Absolute pelvis height is useless as a check here: a pure-stairs clip legitimately climbs to
    3.10 m. Measured from the foot it is the same narrow band on every family, which is what makes
    it a usable gate.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize(2, 1))
    for i, fam in enumerate(rep["families"]):
        v = _cat(rep["rows"], "pelvis_above_foot", {fam})
        if v.size:
            ax.hist(v, bins=np.linspace(0.3, 1.0, 70), histtype="step", lw=1.4, density=True,
                    color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                    label=tex_escape(fam.replace("family_", "").replace(".h5", "")))
    a = _cat(rep["rows"], "pelvis_above_foot")
    ax.set_xlabel("pelvis above lower foot (m)")
    ax.set_ylabel("density")
    ax.legend(fontsize=6, ncol=2)
    return fig, (f"Pelvis above its lower foot. Spans {a.min():.3f}-{a.max():.3f} m library-wide "
                 f"(p01 {np.percentile(a, 1):.3f}, p99 {np.percentile(a, 99):.3f}) and is the same "
                 f"band on every family, terrain included -- which is why the gate checks this "
                 f"rather than absolute height.")


def fig_family_mix(rep: dict):
    """Clips and TILES per family. Tiles, because envs round-robin over tiles.

    A family's geometric variety is its tile count, not its clip count: 26 stair tiles replaying 8
    solved cells still present 26 different staircases to the camera.
    """
    import matplotlib.pyplot as plt

    fams = rep["families"]
    clips = [sum(1 for r in rep["rows"] if r["family"] == f) for f in fams]
    tiles = [len({r["terrain_id"] for r in rep["rows"] if r["family"] == f}) for f in fams]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=figsize(2, 1, aspect=0.5))
    ax.bar(x - 0.2, clips, 0.4, label="clips", color=COLOR_CYCLE[0])
    ax.bar(x + 0.2, tiles, 0.4, label="tiles", color=COLOR_CYCLE[1])
    ax.set_xticks(x, [tex_escape(f.replace("family_", "").replace(".h5", "")) for f in fams])
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return fig, ("Clips and tiles per family. Tiles is the one that sets training exposure, because "
                 "envs round-robin over tiles -- a family's geometric variety is its tile count.")


_FIGS = {"commanded_vx": fig_commanded_vx, "speed_by_kind": fig_speed_by_kind,
         "commands_per_clip": fig_commands_per_clip, "lateral_yaw": fig_lateral_yaw,
         "pelvis": fig_pelvis, "family_mix": fig_family_mix}


def draw_all(rep: dict, out_dir: str, recipe: dict | None, which=None) -> list[str]:
    """Render every figure into ``out_dir``; one failing figure must not lose the rest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_paper_style()
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for name, fn in _FIGS.items():
        if which and name not in which:
            continue
        try:
            fig, cap = fn(rep, recipe) if name == "commanded_vx" else fn(rep)
            save_figure(fig, os.path.join(out_dir, name), caption=cap)
            plt.close(fig)
            made.append(name)
        except Exception as e:                       # one bad figure, not one bad report
            print(f"[report][WARN] {name}: {type(e).__name__}: {e}")
    return made


def _summary(rep: dict) -> dict:
    """The scalars worth diffing between builds, beside the figures."""
    vx = _cat(rep["rows"], "cmd_vx", {"flat"})
    moving = vx[np.abs(vx) > 1e-9]
    h, _ = np.histogram(moving, bins=np.arange(-1.0, 3.61, 0.2))
    sh = h / max(h.sum(), 1)
    nz = sh[sh > 0]
    sp = _cat(rep["rows"], "speed")
    pa = _cat(rep["rows"], "pelvis_above_foot")
    vy, wz = np.abs(_cat(rep["rows"], "cmd_vy")), np.abs(_cat(rep["rows"], "cmd_wz"))
    return {
        "n_clips": rep["n_clips"], "families": rep["families"],
        "vx_bin_max_over_min": float(sh.max() / nz.min()) if nz.size else None,
        "vx_empty_bins": int((sh == 0).sum()),
        "vx_distinct": int(len(np.unique(np.round(vx, 6)))),
        "root_speed_p50": float(np.median(sp)), "root_speed_max": float(sp.max()),
        "pelvis_above_foot_min": float(pa.min()), "pelvis_above_foot_max": float(pa.max()),
        "frames_with_both_vy_and_wz": int(((vy > 1e-9) & (wz > 1e-9)).sum()),
        "cmds_per_clip_median": float(np.median(
            [len(np.unique(np.round(r["cmd_vx"], 6))) for r in rep["rows"]])),
    }


def main() -> int:
    """Collect once, render, and write report.json."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--recipe", default=None,
                   help="Draws the DECLARED command envelope behind the achieved one, which is what "
                        "turns the marginal from a description into a comparison.")
    p.add_argument("--max-clips", type=int, default=0, help="Subsample for a quick look.")
    p.add_argument("--only", default=None, help="Comma-separated figure names.")
    p.add_argument("--sample-viser", type=int, default=0,
                   help="Print the view_clip_viser.py command lines for the N worst clips by peak "
                        "root speed, so 'show me the bad ones' is not an index hunt.")
    args = p.parse_args()

    rep = collect(args.lib, args.max_clips)
    recipe = None
    if args.recipe:
        import traj_library_coverage as tlc
        recipe = tlc.read_recipe_ranges(args.recipe)
    made = draw_all(rep, args.out_dir, recipe,
                    which=set(args.only.split(",")) if args.only else None)
    summ = _summary(rep)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(f"[report] {len(made)} figures -> {args.out_dir}")
    for k, v in summ.items():
        if not isinstance(v, list):
            print(f"    {k:32s} {v}")
    if args.sample_viser:
        worst = sorted(rep["rows"], key=lambda r: -float(r["speed"].max()))[:args.sample_viser]
        print(f"\n[report] the {len(worst)} worst clips by peak root speed:")
        for r in worst:
            print(f"    # {r['family']}  peak {float(r['speed'].max()):.2f} m/s\n"
                  f"    python scripts/rsl_rl/view_clip_viser.py --lib {args.lib} "
                  f"--clip {r['clip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
