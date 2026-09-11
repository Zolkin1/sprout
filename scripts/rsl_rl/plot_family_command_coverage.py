# Copyright 2026 Zachary Olkin. All rights reserved.

"""What velocity commands does a clip library cover, per TERRAIN FAMILY?

The question this answers is the one behind the terrain-misidentification work: when the
generator reads a staircase as a box, is that a regime it was ever shown?  A command is
only half of the pair -- the other half is the geometry the robot was walking at while it
was commanded -- so the figure plots the joint coverage, not just the marginals.

A clip's family is NOT stored on the clip.  It comes from the terrain manifest: ``terrain_id``
indexes a tile, and the tile carries ``family`` (``family_stairs_up.h5``, ``family_jump_on.h5``,
...).  Clips with ``terrain_id == -1`` are the flat ones and are labelled ``family_flat``.
That is the same rule ``scripts/rsl_rl/eval`` uses, so the names line up with an eval report.

Two weightings, and they are not the same claim:

* **per clip** -- one sample per clip.  What the LIBRARY was authored to cover.
* **per frame** -- one sample per valid control step (``clip_len`` bounds it; padding repeats
  the last command and would double-count a slow clip's tail).  What the model actually SEES,
  which is different whenever a slower command makes for a longer clip.

Both are drawn, because on every terrain family the command is a per-clip CONSTANT and the
two therefore differ only through clip length -- a bias worth being able to see.

numpy + h5py + matplotlib only; no IsaacLab, no torch.  Run it in either venv::

    python scripts/rsl_rl/plot_family_command_coverage.py \
        --clip clips/g1_mb_v5_body_mode11_decor.h5 --out /tmp/cmd_coverage
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style  # noqa: E402

#: Families to draw, in plot order, with the short label used in the legend.  ``stairs_up``
#: and ``jump_on`` are the pair the misidentification question is about; the others are drawn
#: as context because "what does the library cover" is not answerable from two of nine.
FAMILY_LABELS: dict[str, str] = {
    "family_stairs_up.h5": "stairs up",
    "family_stairs_up_pure.h5": "stairs up (pure)",
    "family_jump_on.h5": "jump on (box)",
    "family_stairs_down.h5": "stairs down",
    "family_stairs_dn_pure.h5": "stairs down (pure)",
    "family_jump_off.h5": "jump off",
    "family_jump_off_front.h5": "jump off (front)",
    "family_distractor.h5": "distractor",
    "family_flat": "flat",
}

#: The three drawn in the headline panels.  Everything else is context.
FOCUS: tuple[str, ...] = ("family_stairs_up.h5", "family_stairs_up_pure.h5",
                          "family_jump_on.h5")


def _family_of_tile(manifest: list[dict]) -> dict[int, str]:
    """Map ``terrain_id`` -> family string from a library's ``terrain_manifest``."""
    return {int(t["terrain_id"]): str(t.get("family", "unknown")) for t in manifest}


def _tile_obstacle_height(tile: dict) -> float | None:
    """The height of the single obstacle a robot meets head-on in this tile, in metres.

    For a staircase that is the per-step RISE (what the foot has to clear each stride); for a
    box tile it is the box's top surface (what the foot has to clear in one go).  They are the
    comparable quantity precisely because they are what a forward-facing camera sees as the
    first vertical face -- which is the thing a stair-vs-box confusion is a confusion about.

    Walkable overlays only, decoration excluded.  A step is distinguished from a platform by
    its x extent, against a 1.0 m threshold: every tread in the library is under 0.3 m and
    every approach/landing platform is over 6 m, so the threshold sits in a two-order gap and
    the exact tread set (which differs between the up and down families) does not matter.

    Returns:
        The rise (m), or ``None`` when the tile carries no walkable obstacle.
    """
    steps = [o for o in tile.get("overlays", [])
             if o.get("walkable_top") and not o.get("decor")]
    if not steps:
        return None
    tops = sorted(float(o["size_xyz"][2]) for o in steps if float(o["size_xyz"][0]) < 1.0)
    if not tops:
        return None
    if len(tops) == 1:
        return tops[0]                       # a box: one face, its full height
    rises = np.diff(np.asarray(tops))
    rises = rises[rises > 1e-4]
    return float(np.median(rises)) if rises.size else tops[0]


def _tile_lateral_width(tile: dict) -> float | None:
    """Lateral (y) extent of the tile's narrowest walkable obstacle, in metres.

    This is "the width of the staircase" as a person facing it would say it -- how wide the
    flight is across the direction of travel -- NOT the tread run.  It is the dimension
    ``motion_gen.terrain.decorate`` varies, and the one that decides whether a flight looks
    like a corridor or like a free-standing block.
    """
    steps = [o for o in tile.get("overlays", [])
             if o.get("walkable_top") and not o.get("decor")
             and float(o["size_xyz"][0]) < 1.0]
    if not steps:
        return None
    return float(min(float(o["size_xyz"][1]) for o in steps))


def load_coverage(clip_path: str) -> dict:
    """Read the per-clip command, family and tile geometry out of a clip library.

    Args:
        clip_path: Path to the clip ``.h5``.

    Returns:
        A dict with ``per_clip`` (``(N, 3)`` command at frame 0), ``per_frame`` (a dict
        family -> ``(M, 3)`` over valid frames), ``families`` (``(N,)`` strings), ``clip_len``,
        ``n_distinct`` (``(N,)`` distinct command rows per clip) and ``tile_geom``
        (family -> list of ``(terrain_id, rise, width)``).
    """
    with h5py.File(clip_path, "r") as f:
        target_vel = np.asarray(f["target_vel"][:], dtype=np.float64)
        clip_len = np.asarray(f["clip_len"][:], dtype=np.int64)
        terrain_id = np.asarray(f["terrain_id"][:], dtype=np.int64)
        manifest = json.loads(f.attrs["terrain_manifest"])

    fam_of_tile = _family_of_tile(manifest)
    families = np.array([fam_of_tile.get(int(t), "family_flat") for t in terrain_id])

    per_frame: dict[str, np.ndarray] = {}
    n_distinct = np.zeros(len(clip_len), dtype=np.int64)
    for i, n in enumerate(clip_len):
        v = target_vel[i, :n]
        n_distinct[i] = len(np.unique(np.round(v, 4), axis=0))
    for fam in np.unique(families):
        idx = np.where(families == fam)[0]
        per_frame[str(fam)] = np.concatenate([target_vel[i, :clip_len[i]] for i in idx])

    tile_geom: dict[str, list[tuple[int, float, float]]] = {}
    for t in manifest:
        fam = str(t.get("family", "unknown"))
        rise, width = _tile_obstacle_height(t), _tile_lateral_width(t)
        if rise is None or width is None:
            continue
        tile_geom.setdefault(fam, []).append((int(t["terrain_id"]), rise, width))

    return {"per_clip": target_vel[:, 0, :], "per_frame": per_frame, "families": families,
            "clip_len": clip_len, "terrain_id": terrain_id, "n_distinct": n_distinct,
            "tile_geom": tile_geom, "manifest": manifest}


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF points ``(sorted x, F)`` for a 1-D sample."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    return xs, np.arange(1, xs.size + 1) / xs.size


def _panel(ax, letter: str) -> None:
    """Put a panel letter OUTSIDE the axes, where no legend can ever land on it."""
    ax.text(0.0, 1.02, rf"({letter})", transform=ax.transAxes, va="bottom",
            ha="left", fontsize=18)


def make_figure(cov: dict, clip_name: str, colors: dict) -> tuple:
    """Build the four-panel coverage figure.

    Panels: (a) a coverage strip -- every distinct commanded ``v_x`` the family holds, one row
    per family, which shows range and GRANULARITY together (the terrain families are a 50-level
    grid, not a sample); (b) the ECDF of the two families the stair-vs-box question is about;
    (c) the lateral / yaw coverage, where every terrain clip is one point at the origin; and
    (d) the joint coverage of forward command against the vertical rise the tile presents.

    Returns:
        ``(fig, caption)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fam_colors = {
        "family_stairs_up.h5": colors["blue"],
        "family_stairs_up_pure.h5": colors["cyan"],
        "family_jump_on.h5": colors["orange"],
        "family_stairs_down.h5": colors["teal"],
        "family_stairs_dn_pure.h5": "#7BAFD4",
        "family_jump_off.h5": colors["magenta"],
        "family_jump_off_front.h5": colors["red"],
        "family_distractor.h5": "#999999",
        "family_flat": colors["black"],
    }
    esc = plot_style.tex_escape
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    per_clip, families = cov["per_clip"], cov["families"]
    stair_hi = float(per_clip[families == "family_stairs_up.h5", 0].max())
    jump_hi = float(per_clip[families == "family_jump_on.h5", 0].max())

    # (a) coverage strip: one row per family, a tick per DISTINCT commanded v_x.  A histogram
    # cannot show this -- the terrain families are a 50-level grid and the bins alias it into
    # a comb whose teeth are an artefact of the binning, not of the library.
    ax = axes[0, 0]
    rows = [f for f in FAMILY_LABELS if (families == f).any()]
    for i, fam in enumerate(rows):
        m = families == fam
        vx = per_clip[m, 0] if fam not in ("family_flat", "family_distractor.h5") \
            else cov["per_frame"][fam][:, 0]
        uni = np.unique(np.round(vx, 4))
        y = len(rows) - 1 - i
        ax.hlines(y, uni.min(), uni.max(), color=fam_colors[fam], lw=7, alpha=0.28)
        ax.eventplot([uni], lineoffsets=[y], linelengths=[0.62], linewidths=1.3,
                     colors=[fam_colors[fam]])
        ax.text(3.62, y, rf"{len(uni)}", va="center", ha="left", fontsize=12,
                color=fam_colors[fam])
    ax.axvspan(stair_hi, jump_hi, color=colors["orange"], alpha=0.13, lw=0, zorder=0)
    ax.axvline(stair_hi, color=colors["orange"], lw=1.6, ls=":", zorder=1)
    ax.text(0.5 * (stair_hi + jump_hi), 4.0, r"no stairs", ha="center", va="center",
            fontsize=13, rotation=90, color=colors["orange"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([esc(FAMILY_LABELS[f]) for f in reversed(rows)], fontsize=13)
    ax.set_xlim(-1.25, 3.95)
    ax.set_ylim(-0.75, len(rows) - 0.05)
    ax.set_xlabel(r"commanded $v_x$ (m/s)")
    _panel(ax, "a")
    ax.text(3.62, len(rows) - 0.4, r"levels", va="center", ha="left", fontsize=11,
            color="0.35")

    # (b) ECDF of the two families the question is about, per clip and per frame.
    ax = axes[0, 1]
    styles = {"family_stairs_up.h5": "-", "family_stairs_up_pure.h5": (0, (6, 2)),
              "family_jump_on.h5": "-"}
    for fam in FOCUS:
        m = families == fam
        if not m.any():
            continue
        xs, fs = _ecdf(per_clip[m, 0])
        ax.plot(xs, fs, lw=3.0, ls=styles[fam], color=fam_colors[fam],
                label=esc(FAMILY_LABELS[fam]))
        xf, ff = _ecdf(cov["per_frame"][fam][:, 0])
        ax.plot(xf, ff, lw=1.6, ls=":", color=fam_colors[fam], alpha=0.9)
    ax.axvspan(stair_hi, jump_hi, color=colors["orange"], alpha=0.13, lw=0, zorder=0)
    ax.plot([], [], lw=1.6, ls=":", color="0.35", label=r"per frame (time-weighted)")
    ax.set_xlabel(r"commanded $v_x$ (m/s)")
    ax.set_ylabel(r"cumulative fraction of clips")
    ax.set_xlim(0.35, 3.65)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=12)
    _panel(ax, "b")

    # (c) lateral / yaw coverage.  v_y is exactly zero on all 7000 terrain clips; w_z is too
    # on every stair and jump family, but NOT on distractor, which turns on 35% of its frames.
    # Folding distractor into the origin marker would have made that marker's own label false.
    ax = axes[1, 0]
    flat = cov["per_frame"].get("family_flat")
    if flat is not None and flat.size:
        hb = ax.hexbin(flat[:, 1], flat[:, 2], gridsize=44, bins="log", mincnt=1,
                       cmap="Greys", linewidths=0.0)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label(r"flat frames", fontsize=13)
        cb.ax.tick_params(labelsize=11)
    dis = cov["per_frame"].get("family_distractor.h5")
    if dis is not None and dis.size:
        n_dis = int((families == "family_distractor.h5").sum())
        ax.scatter(dis[:, 1], dis[:, 2], s=8, alpha=0.02, color=fam_colors["family_distractor.h5"],
                   edgecolors="none", rasterized=True, zorder=3)
        ax.plot([], [], "o", ms=8, color=fam_colors["family_distractor.h5"],
                label=rf"distractor ({n_dis} clips)")
    pinned = [f for f in FAMILY_LABELS
              if f not in ("family_flat", "family_distractor.h5")]
    n_pinned = int(sum((families == f).sum() for f in pinned))
    ax.plot([0.0], [0.0], "o", ms=16, mfc=colors["red"], mec="k", mew=1.6, zorder=5,
            label=rf"{n_pinned} stair + jump clips")
    ax.set_xlabel(r"commanded $v_y$ (m/s)")
    ax.set_ylabel(r"commanded $\omega_z$ (rad/s)")
    ax.legend(loc="lower left", fontsize=11)
    _panel(ax, "c")

    # (d) joint coverage: forward command vs the vertical rise the tile presents in one go.
    ax = axes[1, 1]
    marks = {"family_stairs_up.h5": ("o", 26), "family_stairs_up_pure.h5": ("s", 14),
             "family_jump_on.h5": ("^", 22)}
    for fam in FOCUS:
        geom = {tid: (rise, w) for tid, rise, w in cov["tile_geom"].get(fam, [])}
        m = families == fam
        if not m.any() or not geom:
            continue
        tids = cov["terrain_id"][m]
        keep = np.array([int(t) in geom for t in tids])
        if not keep.any():
            continue
        rises = np.array([geom[int(t)][0] for t in tids[keep]])
        mk, sz = marks[fam]
        ax.scatter(per_clip[m, 0][keep], rises, s=sz, alpha=0.4, marker=mk,
                   color=fam_colors[fam], edgecolors="none",
                   label=esc(FAMILY_LABELS[fam]))
    ax.axvspan(stair_hi, jump_hi, color=colors["orange"], alpha=0.13, lw=0, zorder=0)
    ax.axhline(0.18, color=colors["blue"], lw=1.4, ls=":", zorder=1)
    ax.text(0.42, 0.205, r"every stair rise", fontsize=12, color=colors["blue"],
            va="bottom")
    ax.set_xlabel(r"commanded $v_x$ (m/s)")
    ax.set_ylabel(r"vertical rise the tile presents (m)")
    ax.set_xlim(0.35, 3.65)
    ax.set_ylim(0.0, 0.82)
    ax.legend(loc="upper left", fontsize=12)
    _panel(ax, "d")

    caption = _caption(cov, clip_name)
    return fig, caption


def _caption(cov: dict, clip_name: str) -> str:
    """The definitions in force for the figure, plus the headline numbers it shows."""
    families, per_clip = cov["families"], cov["per_clip"]
    lines = [
        f"Commanded-velocity coverage per terrain family in {clip_name}.",
        "A clip's family comes from the terrain manifest (terrain_id -> tile -> family);",
        "terrain_id == -1 is the flat set. On the seven stair and jump families the command",
        "is a per-clip CONSTANT -- one distinct row over every valid frame -- so for those,",
        "frame 0 IS the clip. It is not constant on distractor (450 of 1000 clips change) or",
        "on flat. (a) therefore draws the seven constant families per CLIP and those two per",
        "FRAME; (b) and (d) cover only constant families, so they are per clip throughout.",
        "(b) also draws the per-FRAME ECDF dashed -- it differs from the per-clip one only",
        "through clip length, so the gap is the over-representation of slow commands in",
        "wall-clock exposure. In (c) the red marker is the 6000 stair and jump clips, every",
        "one commanded v_y = w_z = 0 EXACTLY; distractor is drawn separately because it is",
        "the one terrain family that turns (w_z in [-0.650, +0.683] on 35% of its frames),",
        "though its v_y is zero too -- so v_y is exactly zero on all 7000 terrain clips.",
        "(d) plots the per-stride rise the tile presents -- the median step rise for a",
        "staircase, the box top for a box tile -- against the command.",
        "",
    ]
    for fam in FOCUS + ("family_stairs_down.h5", "family_jump_off_front.h5", "family_flat"):
        m = families == fam
        if not m.any():
            continue
        vx = per_clip[m, 0] if fam != "family_flat" else cov["per_frame"][fam][:, 0]
        nuni = len(np.unique(np.round(vx, 4)))
        lines.append(
            f"  {FAMILY_LABELS[fam]:<18s} n={int(m.sum()):5d} clips  "
            f"v_x [{vx.min():+.3f}, {vx.max():+.3f}] m/s  "
            f"median {np.median(vx):+.3f}  distinct {nuni}"
        )
    return "\n".join(lines)


def main() -> None:
    """Parse arguments, read the library and write the figure."""
    ap = argparse.ArgumentParser(
        description="Plot commanded-velocity coverage per terrain family of a clip library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--clip", required=True, help="Clip library .h5 (no default on purpose).")
    ap.add_argument("--out", default="cmd_coverage",
                    help="Output path stem; .png/.svg/.caption.txt are written.")
    ap.add_argument("--no_usetex", action="store_true", help="Force mathtext instead of LaTeX.")
    args = ap.parse_args()

    plot_style.apply_paper_style(usetex=False if args.no_usetex else None)
    cov = load_coverage(args.clip)
    fig, caption = make_figure(cov, os.path.basename(args.clip), plot_style.COLORS)
    written = plot_style.save_figure(fig, args.out, caption=caption)
    print(caption)
    for w in written:
        print(f"[INFO] wrote {w}")


if __name__ == "__main__":
    main()
