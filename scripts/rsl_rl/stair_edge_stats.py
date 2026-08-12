#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where the stair-up reference clips put their feet relative to a tread nose, and when.

Reads only the clip library ``.h5`` (no IsaacLab, no simulator).  For every frame of every
stair-up TRANSITION clip it computes, per foot:

* the **signed distance to the nearest tread nose** -- positive = the nose is still ahead in
  the direction of travel, negative = the foot is past it, ~0 = the foot is right at the edge.
  The nose is the riser face a foot has to clear, so this is the margin that decides whether a
  step catches;
* a **gait-phase descriptor** ``z_right - z_left`` on sole heights, under three references
  (see :func:`descriptors`), because on a staircase "foot height" has no single meaning.

The question it answers: does the reference data itself step close to the edges, and at what
point in the gait -- i.e. is the tracker being asked for a margin it cannot hold, before any
generator is involved.

Everything is per-frame, so the ``.npz`` it writes supports cuts this script does not plot
(by clip, by segment, by foot, by stance/swing).

Example::

    python scripts/rsl_rl/stair_edge_stats.py --lib clips/g1_mb_v5_body.h5 \
        --out plots/stair_edges

Only the tile's own ``overlays`` are trusted for geometry -- never recomputed from riser/tread,
since a tile can be re-datumed after authoring (``mb_assemble_library._redatum_tile``) and the
manifest is what the clip's coordinates were shifted against.
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
for _p in (os.path.join(_REPO, "source", "robot_rl"), os.path.join(_REPO, "source", "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: A box is a TREAD (rather than an approach/recovery platform or a jump box) when it is
#: narrow in the travel direction. Same rule as ``motion_gen.terrain.terrain_place._tread_boxes``
#: and ``scripts/rsl_rl/mb_terrain_clips._tread_tops``, which is how the whole pipeline tells
#: the two apart -- but both of those take the AUTHORING dict form, not the manifest form, so
#: the rule is restated here rather than imported through a private.
_TREAD_SX_FACTOR = 1.5

#: Sole within this of a walkable top, with little vertical speed, counts as planted.
_STANCE_TOL_M = 0.02
_STANCE_VEL_MS = 0.05

#: Ankle-frame x of the foot box's toe and heel, from ``motion_gen.clip_output``
#: (``_FOOT_BOX_CENTER[0] +- _FOOT_BOX_HALF[0]``). The toe is what actually catches a riser.
_TOE_DX = 0.1375
_HEEL_DX = -0.0656


class TileGeometry:
    """The walkable boxes of one terrain tile, in the tile's own frame.

    The clip's ``y_des`` positions are in this same frame -- ``footprint_recenter`` datums the
    motion on the boxes' bounding box and ``overlays_for`` shifts the same boxes by the same
    corner -- so no transform is needed between a foot position and these numbers.

    Args:
        entry: One ``terrain_manifest`` entry carrying the ``overlays`` schema.

    Raises:
        ValueError: On the legacy ``dims``/``bands`` schema, where the geometry would have to
            be recomputed AND the clip is front-padded by a back-buffer -- so silently
            treating it like an overlay tile puts motion and terrain ~1 m apart in x.
    """

    def __init__(self, entry: dict) -> None:
        if "overlays" not in entry:
            raise ValueError(
                f"terrain_id {entry.get('terrain_id')} carries the legacy bands schema "
                f"(keys {sorted(entry)}), not 'overlays'. Its geometry is not in the manifest "
                "and its clip is front-padded, so foot-vs-edge distances would be offset. "
                "Regenerate the library with the current pipeline."
            )
        ov = entry["overlays"]
        if not ov:
            raise ValueError(f"terrain_id {entry.get('terrain_id')} has no overlay boxes.")
        cx = np.array([float(o["pos_xy"][0]) for o in ov])
        cy = np.array([float(o["pos_xy"][1]) for o in ov])
        sx = np.array([float(o["size_xyz"][0]) for o in ov])
        sy = np.array([float(o["size_xyz"][1]) for o in ov])
        self.x0, self.x1 = cx - sx / 2.0, cx + sx / 2.0
        self.y0, self.y1 = cy - sy / 2.0, cy + sy / 2.0
        self.top = np.array([float(o["base_z"]) + float(o["size_xyz"][2]) for o in ov])
        self.is_tread = sx <= _TREAD_SX_FACTOR * sx.min()
        # The ascent riser face of each tread. Travel is +x for every tile the current
        # pipeline emits (the placement yaw-normalizes to +x and every overlay writes
        # yaw = 0), which `assert_travel_plus_x` re-checks against the motion itself.
        self.noses = np.sort(self.x0[self.is_tread])
        self.levels = np.unique(np.round(self.top, 4))
        # The first nose is NOT the first step: on every tile this pipeline emits, tread 1's
        # top is FLUSH with the approach platform, so its nose is a seam in the mesh between
        # two surfaces at the same height. The first edge the robot must actually climb is the
        # lowest-x tread whose top is ABOVE the approach, which is one tread-depth (0.21-0.25 m)
        # further on. Measuring the approach against the seam understates the distance to the
        # step by exactly that, and puts a foot "past the first edge" while it is still on flat
        # ground.
        approach = float(np.min(self.top[~self.is_tread])) if (~self.is_tread).any() else -np.inf
        up = self.is_tread & (self.top > approach + 1e-4)
        self.riser_noses = np.sort(self.x0[up])
        self.approach_top = approach
        self.first_riser = float(self.riser_noses[0]) if self.riser_noses.size else (
            float(self.noses[0]) if self.noses.size else np.nan)
        self.terrain_id = int(entry["terrain_id"])
        self.has_floor = bool(entry.get("has_floor", False))
        self.frame_yaw = float(entry.get("frame_yaw", 0.0))

    def ground_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Top surface under each ``(x, y)``; NaN where nothing is under it.

        The highest containing box wins: a tread sits flush on the approach platform, so the
        footprints genuinely overlap and the maximum is the walkable surface.  Each box is
        tested against its OWN extent -- treads keep their authored width while platforms are
        widened to the whole tile, so a shared half-width would misplace both.

        NaN rather than 0.0 when nothing contains the point: ``has_floor`` is False on the
        current stair tiles, so off-tread is a void, and 0.0 would read as solid ground a
        metre below the foot.
        """
        inside = ((x[:, None] >= self.x0[None, :]) & (x[:, None] <= self.x1[None, :])
                  & (y[:, None] >= self.y0[None, :]) & (y[:, None] <= self.y1[None, :]))
        tops = np.where(inside, self.top[None, :], -np.inf)
        best = tops.max(axis=1)
        if self.has_floor:
            best = np.where(np.isfinite(best), best, 0.0)
        return np.where(np.isfinite(best), best, np.nan)

    def edge_distances(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(signed_nearest, next_ahead)`` distance from ``x`` to a tread nose, in metres.

        ``signed_nearest`` = ``nose - x`` for whichever nose is closest: positive means that
        nose is ahead of the foot, negative means the foot is past it, and 0 is the foot
        exactly on the edge. Defined everywhere, which is what makes it the histogram axis.

        ``next_ahead`` is the strictly-ahead nose only, NaN once the foot is past the last
        one -- the literal "how far to the next step", useful for a lead-in cut.
        """
        if self.noses.size == 0:
            nan = np.full(x.shape, np.nan)
            return nan, nan.copy()
        d = self.noses[None, :] - x[:, None]                       # [T, n_noses]
        nearest = d[np.arange(d.shape[0]), np.abs(d).argmin(axis=1)]
        ahead = np.where(d >= 0.0, d, np.inf).min(axis=1)
        return nearest, np.where(np.isfinite(ahead), ahead, np.nan)


def foot_columns(pos_names: list[str], frame: str) -> tuple[int, int, int, list[int]]:
    """Column indices of one frame's ``pos_{x,y,z}`` and its ``ori_{w,x,y,z}`` quaternion."""
    xyz = [pos_names.index(f"{frame}:pos_{a}") for a in "xyz"]
    quat = [pos_names.index(f"{frame}:ori_{a}") for a in "wxyz"]
    return xyz[0], xyz[1], xyz[2], quat


def yaw_from_wxyz(q: np.ndarray) -> np.ndarray:
    """Yaw (rad) of a ``[T, 4]`` scalar-FIRST quaternion track.

    Scalar-first because that is how the clip library stores ``ori_*`` (``ori_w`` precedes
    ``ori_x``); ``inspect_clip._quat_cols`` returns scalar-LAST, so the two are not
    interchangeable.
    """
    qw, qx, qy, qz = (q[:, i] for i in range(4))
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def contact_point(y: np.ndarray, pos_names: list[str], frame: str, point: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` of the ankle, toe or heel of one foot, per frame.

    The toe/heel offsets are along the foot's own heading, taken from the ankle
    quaternion's yaw -- the foot box's roll and pitch move the contact point vertically, which
    the sole height already carries, not along the travel direction this measures.
    """
    ix, iy, _, iq = foot_columns(pos_names, frame)
    x, y_ = y[:, ix].copy(), y[:, iy].copy()
    if point == "ankle":
        return x, y_
    dx = _TOE_DX if point == "toe" else _HEEL_DX
    yaw = yaw_from_wxyz(y[:, iq])
    return x + dx * np.cos(yaw), y_ + dx * np.sin(yaw)


def planted_mask(sole: np.ndarray, ground: np.ndarray, dt: float) -> np.ndarray:
    """``[T]`` bool: is this foot planted on a surface?

    Geometric, deliberately: the library's ``contact`` dataset is a proximity test against
    ``ground_height = 0.0``, so on a stair-transition clip -- whose approach and recovery
    platforms sit at one riser and above -- it is zero everywhere except the world-pinned
    replay window, and any stance statistic derived from it silently reports no stances at all
    on the approach.

    A frame counts when the sole is within :data:`_STANCE_TOL_M` of the surface under it and
    the sole is not moving vertically faster than :data:`_STANCE_VEL_MS`.
    """
    near = np.isfinite(ground) & (np.abs(sole - ground) < _STANCE_TOL_M)
    vel = np.gradient(sole, dt) if sole.size > 2 else np.zeros_like(sole)
    return near & (np.abs(vel) < _STANCE_VEL_MS)


def last_stance_level(ground: np.ndarray, planted: np.ndarray) -> np.ndarray:
    """Surface this foot most recently stood on, held through the swing -- ``[T]``.

    Forward-filled from the planted frames, then back-filled so the frames before the first
    touchdown carry the first level rather than NaN.  Piecewise-constant by construction,
    which is the point: the instantaneous ground under a SWING foot steps by a whole riser as
    it crosses a nose -- exactly the moment being measured -- so a descriptor built on it
    jumps where it needs to be smooth.
    """
    lvl = np.where(planted & np.isfinite(ground), ground, np.nan)
    idx = np.where(~np.isnan(lvl), np.arange(lvl.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = lvl[idx]
    if np.isnan(out).any() and not np.isnan(lvl).all():
        out[np.isnan(out)] = lvl[~np.isnan(lvl)][0]
    return out


def descriptors(soles: np.ndarray, ground: np.ndarray, stance_lvl: np.ndarray
                ) -> dict[str, np.ndarray]:
    """The gait-phase descriptor ``z_right - z_left`` under three height references.

    All three are signed and near zero at double support, ``+`` when the right foot is higher:

    * ``raw`` -- world sole heights. Simplest, but on stairs it carries a constant offset of
      one riser whenever the feet straddle a step, and the pair drifts upward through the climb.
    * ``inst`` -- each sole above the surface under it right now. Removes both of those, but is
      discontinuous mid-swing at a nose crossing (see :func:`last_stance_level`).
    * ``stance`` -- each sole above the tread that foot last stood on. Continuous and
      bias-free; the default.

    Args:
        soles: ``[T, 2]`` sole heights, (left, right).
        ground: ``[T, 2]`` instantaneous surface under each foot.
        stance_lvl: ``[T, 2]`` last-stance level per foot.

    Returns:
        ``{"raw"|"inst"|"stance": [T]}``.
    """
    return {
        "raw": soles[:, 1] - soles[:, 0],
        "inst": (soles[:, 1] - ground[:, 1]) - (soles[:, 0] - ground[:, 0]),
        "stance": (soles[:, 1] - stance_lvl[:, 1]) - (soles[:, 0] - stance_lvl[:, 0]),
    }


def collect(lib_path: str, family: str, point: str, max_clips: int | None) -> dict:
    """Walk the stair-up transition clips and gather per-frame foot/edge/phase columns.

    Returns:
        A dict of flat arrays, all indexed by the same per-frame row (foot-indexed ones are
        ``[R, 2]``, left then right), plus the ``dt`` and clip/tile bookkeeping.
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
    # One manifest entry per CLIP, not per tile -- entries for a tile are identical after
    # re-datuming, so keep the first of each.
    by_tid: dict[int, dict] = {}
    for entry in manifest:
        by_tid.setdefault(int(entry["terrain_id"]), entry)
    fam_of = {tid: str(e.get("family", "")) for tid, e in by_tid.items()}
    family_col = np.array([fam_of.get(int(t), "") for t in terrain_id], dtype=object)

    want = (type_code == type_names.index("transition")) & (family_col == family)
    clips = np.nonzero(want)[0]
    if clips.size == 0:
        raise SystemExit(
            f"{lib_path} has no clip with family={family!r} and type='transition'. "
            f"Families present: {sorted({v for v in fam_of.values() if v})}."
        )
    if max_clips is not None:
        clips = clips[:max_clips]
    print(f"[stair_edge_stats] {clips.size} clips, family={family}, contact point={point}, dt={dt}")

    lib = load_clip_library(lib_path, device="cpu",
                            keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    pinned_all = anchor_mask_is_pinned(lib.anchor_mask_per_step()).numpy()
    clip_len_all = lib.clip_len_or_full().numpy()
    geoms = {tid: TileGeometry(e) for tid, e in by_tid.items() if tid in set(terrain_id[clips])}

    frames = ("left_ankle_roll_link", "right_ankle_roll_link")
    cols: dict[str, list] = {k: [] for k in (
        "clip", "tile", "frame", "pinned", "d_edge", "d_next", "sole", "ground",
        "planted", "stance_lvl", "foot_x", "desc_raw", "desc_inst", "desc_stance")}
    with h5py.File(lib_path, "r") as f:
        for n, ci in enumerate(clips):
            if n % 200 == 0:
                print(f"  ... {n}/{clips.size}", flush=True)
            length = int(clip_len_all[ci])
            geom = geoms[int(terrain_id[ci])]
            y = f["y_des"][ci, :length, :].astype(np.float64)
            soles = sole_heights(y, pos_names, frames=frames)              # [T, 2]
            fx = np.empty((length, 2))
            fy = np.empty((length, 2))
            for j, fr in enumerate(frames):
                fx[:, j], fy[:, j] = contact_point(y, pos_names, fr, point)
            ground = np.stack([geom.ground_z(fx[:, j], fy[:, j]) for j in range(2)], axis=1)
            planted = np.stack([planted_mask(soles[:, j], ground[:, j], dt) for j in range(2)],
                               axis=1)
            stance_lvl = np.stack(
                [last_stance_level(ground[:, j], planted[:, j]) for j in range(2)], axis=1)
            near = np.empty((length, 2))
            ahead = np.empty((length, 2))
            for j in range(2):
                near[:, j], ahead[:, j] = geom.edge_distances(fx[:, j])
            desc = descriptors(soles, ground, stance_lvl)

            cols["clip"].append(np.full(length, ci))
            cols["tile"].append(np.full(length, geom.terrain_id))
            cols["frame"].append(np.arange(length))
            cols["pinned"].append(pinned_all[ci, :length])
            cols["d_edge"].append(near)
            cols["d_next"].append(ahead)
            cols["sole"].append(soles)
            cols["ground"].append(ground)
            cols["planted"].append(planted)
            cols["stance_lvl"].append(stance_lvl)
            cols["foot_x"].append(fx)
            for k, v in desc.items():
                cols[f"desc_{k}"].append(v)
    out = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    out["dt"] = np.asarray(dt)
    out["point"] = np.asarray(point)
    out["num_clips"] = np.asarray(clips.size)
    return out


def assert_travel_plus_x(lib_path: str, clips_sample: np.ndarray) -> None:
    """Fail loudly if these clips do not travel in +x, which the nose convention assumes."""
    import h5py

    with h5py.File(lib_path, "r") as f:
        pos_names = json.loads(f.attrs["pos_output_names"])
        px = pos_names.index("pelvis_link:pos_x")
        lens = f["clip_len"][:]
        for ci in clips_sample:
            x = f["y_des"][int(ci), :int(lens[ci]), px]
            if x[-1] <= x[0]:
                raise SystemExit(
                    f"clip {int(ci)} travels {x[0]:.2f} -> {x[-1]:.2f} in x, i.e. NOT +x. The "
                    "tread nose is taken as the box's minimum-x face, so every edge distance "
                    "would have the wrong sign. Handle the tile's frame_yaw before using this."
                )


def plot(data: dict, out_dir: str, ref: str, foot: str, span: float) -> list[str]:
    """2-D histogram of edge distance vs the gait descriptor, with marginals."""
    os.makedirs(out_dir, exist_ok=True)
    desc = data[f"desc_{ref}"]
    d_edge, planted = data["d_edge"], data["planted"]
    if foot == "both":
        d = np.concatenate([d_edge[:, 0], d_edge[:, 1]])
        p = np.concatenate([desc, desc])
        st = np.concatenate([planted[:, 0], planted[:, 1]])
    elif foot == "swing":
        # The airborne foot is the one about to clear the nose; where both are planted there
        # is no swing foot and the frame drops out.
        j = np.argmax(~planted, axis=1)
        keep = (~planted).any(axis=1)
        d, p, st = d_edge[np.arange(len(j)), j][keep], desc[keep], np.zeros(int(keep.sum()), bool)
    else:
        j = 0 if foot == "left" else 1
        d, p, st = d_edge[:, j], desc, planted[:, j]
    ok = np.isfinite(d) & np.isfinite(p) & (np.abs(d) <= span)
    d, p, st = d[ok], p[ok], st[ok]

    fig = plt.figure(figsize=(10.5, 8.0))
    # A dedicated colorbar column: hanging it off the marginal axis squeezes that axis and
    # collides with its tick labels.
    gs = fig.add_gridspec(2, 3, width_ratios=(4, 1, 0.18), height_ratios=(1, 4),
                          wspace=0.06, hspace=0.06)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])
    lim = float(np.nanpercentile(np.abs(p), 99.5)) or 1.0
    h = ax.hist2d(d, p, bins=(120, 100), range=((-span, span), (-lim, lim)),
                  cmap="magma", norm="log")
    fig.colorbar(h[3], cax=cax, label="foot-frames per bin")
    ax.axvline(0.0, color="#00E5FF", lw=1.0, ls="--")
    ax.axhline(0.0, color="#00E5FF", lw=0.6, ls=":")
    ax.set_xlabel("signed distance to nearest tread nose [m]   (+ = nose ahead of foot)")
    ax.set_ylabel(f"gait descriptor  z_right - z_left  [m]   (ref: {ref})")
    top.hist(d, bins=120, range=(-span, span), color="#0077BB")
    top.set_yscale("log")
    top.tick_params(labelbottom=False)
    top.set_title(f"{data['num_clips']} clips, foot={foot}, point={data['point']}, ref={ref} -- "
                  f"{d.size} of {int(np.isfinite(data['d_edge']).sum())} foot-frames "
                  f"within |d| <= {span} m", fontsize=9)
    right.hist(p, bins=100, range=(-lim, lim), orientation="horizontal", color="#EE7733")
    right.set_xscale("log")
    right.tick_params(labelleft=False)
    paths = []
    p2 = os.path.join(out_dir, f"stair_edge_hist_{ref}_{foot}.png")
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    # Stance vs swing separately: the same margin means different things when the foot is
    # planted (it is standing that close to the edge) and when it is swinging (it is passing).
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, (m, name) in zip(axes, ((st, "planted"), (~st, "swing"))):
        if int(m.sum()) == 0:
            a.set_title(f"{name}: none")
            continue
        a.hist2d(d[m], p[m], bins=(90, 80), range=((-span, span), (-lim, lim)),
                 cmap="magma", norm="log")
        a.axvline(0.0, color="#00E5FF", lw=1.0, ls="--")
        a.set_title(f"{name} ({int(m.sum())} foot-frames)")
        a.set_xlabel("signed distance to nose [m]")
    axes[0].set_ylabel(f"z_right - z_left [m] ({ref})")
    p3 = os.path.join(out_dir, f"stair_edge_stance_swing_{ref}_{foot}.png")
    fig2.tight_layout()
    fig2.savefig(p3, dpi=140)
    plt.close(fig2)
    paths.append(p3)
    return paths


def report(data: dict, span: float) -> str:
    """Text summary: how close the reference actually steps, and how the three refs differ."""
    lines = ["", "=" * 74, "STAIR-EDGE / GAIT-PHASE SUMMARY", "=" * 74]
    d = data["d_edge"]
    finite = np.isfinite(d)
    lines.append(f"foot-frames total {d.size}  (with a nose in range: {int(finite.sum())})")
    for j, name in ((0, "left "), (1, "right")):
        col = d[:, j][np.isfinite(d[:, j])]
        pl = data["planted"][:, j]
        lines.append(
            f"  {name}: |d| p05 {np.percentile(np.abs(col), 5):.3f}  p50 "
            f"{np.percentile(np.abs(col), 50):.3f} m   planted {100 * pl.mean():.0f}% of frames")
    near = np.abs(d) < 0.05
    lines.append(f"  foot-frames within 5 cm of a nose: {int(near.sum())} "
                 f"({100 * near.mean():.2f}%)")
    planted_near = near & data["planted"]
    lines.append(f"  ... of which PLANTED (standing on the edge): {int(planted_near.sum())}")
    lines.append("")
    for ref in ("raw", "inst", "stance"):
        v = data[f"desc_{ref}"]
        v = v[np.isfinite(v)]
        lines.append(f"  descriptor {ref:6s}: p05 {np.percentile(v, 5):+.3f}  p50 "
                     f"{np.percentile(v, 50):+.3f}  p95 {np.percentile(v, 95):+.3f}  "
                     f"|.|>0.05 in {100 * np.mean(np.abs(v) > 0.05):.0f}% of frames")
    lines.append(f"  ground query undefined (foot over a void) in "
                 f"{100 * np.mean(~np.isfinite(data['ground'])):.2f}% of foot-frames")
    lines.append("=" * 74)
    return "\n".join(lines)


def main() -> int:
    """Parse args, collect the columns, write the npz + figures."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5.")
    p.add_argument("--family", default="family_stairs_up.h5",
                   help="Tile family to select (with the .h5 suffix, as the manifest spells it).")
    p.add_argument("--point", default="toe", choices=("ankle", "toe", "heel"),
                   help="Which point of the foot the edge distance is measured from.")
    p.add_argument("--ref", default="stance", choices=("stance", "inst", "raw"),
                   help="Height reference for the gait descriptor (all three are always saved).")
    p.add_argument("--foot", default="both", choices=("both", "left", "right", "swing"),
                   help="Which foot the histogram uses.")
    p.add_argument("--span", type=float, default=0.6,
                   help="Edge-distance axis half-range in metres.")
    p.add_argument("--max_clips", type=int, default=None, help="Cap the clip count (debug).")
    p.add_argument("--out", default="plots/stair_edges", help="Output directory.")
    args = p.parse_args()

    data = collect(args.lib, args.family, args.point, args.max_clips)
    assert_travel_plus_x(args.lib, np.unique(data["clip"])[:5])
    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "stair_edge_frames.npz")
    np.savez_compressed(npz, **data)
    print(report(data, args.span))
    written = [npz] + plot(data, args.out, args.ref, args.foot, args.span)
    for w in written:
        print(f"[stair_edge_stats] wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
