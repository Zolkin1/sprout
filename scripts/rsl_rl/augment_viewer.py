#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""See what a registration-augmented training window actually asks the FM to learn.

For each sampled augment, one row: the ORIGINAL depth the generator would have seen, the
depth RE-RENDERED at the displaced pose, their difference, and an x-z panel with the tile's
step profile carrying both reference futures -- the original and the augmented one the model
is trained to reconstruct -- with each node coloured by whether its anchor mask leaves it FREE
to move with the robot or PINNED to the terrain.

That last colouring is the point. The augmentation displaces the robot and then moves each
reference node by only the part of the displacement its mask leaves free: a flat node takes
the whole shift (so the reference follows the robot, a no-op in the robot's frame), while a
terrain-registered node takes the lateral part only, so downrange and yaw become a correction
the model must learn to apply from what it SEES. Whether that is what actually happened is
visible here as the gap between the two curves, split by node colour.

**What moves, and what the array names suggest.** ``history_state`` is the MEASURED ROBOT
state, used as the frame each history row is expressed against -- displacing the robot moves
it, necessarily. The model's autoregressive conditioning is ``history_ref`` (the applied
reference), and it moves too. So the displacement does NOT reach the model through depth
alone; the panels label which arrays differ so the figure states the training signal rather
than an intended invariant.

Example::

    python scripts/rsl_rl/augment_viewer.py \
        --source logs/rsl_rl/g1/<run>/recorded_data/recorded_20260809_152805.h5 \
        --augmented logs/rsl_rl/g1/<run>/recorded_data/augmented_20260809.h5 \
        --lib clips/g1_mb_v5_body.h5 --num 4 --out plots/augment
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

from fm_debug_scene import _depth_image  # noqa: E402
from stair_edge_stats import TileGeometry  # noqa: E402
from transition_census import read_meta  # noqa: E402

#: Join key. Verified unique over all 200000 source windows, and every augmented row resolves
#: through it -- ``env_id`` is included because episode ids need not be unique across envs.
_KEY = ("episode_id", "clip_step", "env_id")


def state_slices(meta) -> dict[str, tuple[int, int]]:
    """``{term: (start, stop)}`` column ranges within one ``history_state`` row."""
    out, off = {}, 0
    for name, dim in zip(meta.state_term_names, meta.state_term_dims):
        out[name] = (off, off + int(dim))
        off += int(dim)
    return out


def pair_windows(src_path: str, aug_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(aug_idx, src_idx, columns)`` for every augmented row that resolves to a source row."""
    import h5py

    cols = {}
    with h5py.File(src_path, "r") as f:
        ns = int(f.attrs["num_windows_written"])
        ks = np.stack([f[k][:ns].astype(np.int64) for k in _KEY], 1)
        cols["src_clip_id"] = f["clip_id"][:ns].astype(np.int64)
        cols["src_clip_step"] = f["clip_step"][:ns].astype(np.int64)
    with h5py.File(aug_path, "r") as f:
        na = int(f.attrs["num_windows_written"])
        ka = np.stack([f[k][:na].astype(np.int64) for k in _KEY], 1)
        cols["aug_family"] = f["clip_family"][:na].astype(np.int64)
    lut = {tuple(r): i for i, r in enumerate(map(tuple, ks))}
    aug_idx, src_idx = [], []
    for j, r in enumerate(map(tuple, ka)):
        i = lut.get(r)
        if i is not None:
            aug_idx.append(j)
            src_idx.append(i)
    return np.asarray(aug_idx), np.asarray(src_idx), cols


def ref_root_pos(ref_future: np.ndarray, s_slc: dict) -> np.ndarray:
    """``[F, 3]`` root positions of a recorded reference window.

    The recording stores ``ref_future`` in the STATE-TERM layout (123 wide here: root pose,
    twists, joints, then per-frame link states), not the FM's window layout -- so the root
    position is a plain column slice, already in the simulation world frame. No lift, and in
    particular NOT ``window_to_world``, which expects the 44-wide window packing and rejects
    this outright.
    """
    a, b = s_slc["root_pos_w"]
    return np.asarray(ref_future, dtype=np.float64)[:, a:b]


def env_offset(ref_pos: np.ndarray, clip_pos: np.ndarray) -> np.ndarray:
    """World -> tile-local offset, as the median over nodes.

    The reference is in sim-world coordinates and the tile geometry is tile-local; they differ
    by the env origin, which the recording does not carry. Recovered per window by differencing
    against the clip's own pelvis track over the same steps -- the median rather than node 0 so
    a single anchoring artefact cannot set the whole frame.
    """
    n = min(ref_pos.shape[0], clip_pos.shape[0])
    return np.median(ref_pos[:n] - clip_pos[:n], axis=0)


def draw_row(axes, src, aug, geom, pinned, spec, label: str) -> None:
    """One augment: reference curves on the terrain, then original / augmented / diff depth."""
    ax = axes[0]
    xs = [src["ref"][:, 0], aug["ref"][:, 0]]
    zs_all = [src["ref"][:, 2], aug["ref"][:, 2]]
    for k, d in (("hist", src), ("hist_aug", aug)):
        h = d.get(k)
        if h is not None and len(h):
            xs.append(h[:, 0])
            zs_all.append(h[:, 2])
    lo = float(np.concatenate(xs).min()) - 0.25
    hi = float(np.concatenate(xs).max()) + 0.25
    if geom is not None:
        for x0, x1, top in sorted(zip(geom.x0, geom.x1, geom.top)):
            if x1 < lo or x0 > hi:
                continue
            ax.fill_between([x0, x1], [top, top], y2=-1.0, color="#4A4A4A", alpha=0.30, lw=0)
            ax.plot([x0, x1], [top, top], color="#4A4A4A", lw=1.4)
        if np.isfinite(geom.first_riser):
            ax.axvline(geom.first_riser, color="#CC3311", ls="--", lw=1.4)
    # The CONDITIONING past, before and after. Drawn because it moves too: the model's
    # autoregressive context is ``history_ref``, and the augmentation re-expresses it against
    # the displaced anchor -- so the displacement does not reach the model through depth alone.
    for key, c, mk, nm in (("hist", "#B35C00", "o", "original ref HISTORY"),
                           ("hist_aug", "#0A7A33", "s", "augmented ref HISTORY")):
        # Look up by KEY, not a hardcoded "hist": the augmented trail lives under
        # "hist_aug" in the aug dict, and a fixed key silently drew nothing for it.
        h = (src if key == "hist" else aug).get(key)
        if h is not None and len(h):
            ax.plot(h[:, 0], h[:, 2], color=c, lw=1.4, ls=":", marker=mk, ms=3.5,
                    label=nm, zorder=5)
    ax.plot(src["ref"][:, 0], src["ref"][:, 2], color="#F28C1E", lw=2.2,
            label="original ref FUTURE")
    ax.plot(aug["ref"][:, 0], aug["ref"][:, 2], color="#12E05A", lw=2.2,
            label="augmented ref FUTURE (the target)")
    # Node colour = what the anchor mask allowed to move.
    free = ~pinned
    for m, c, nm in ((free, "#00D5FF", "free (flat)"), (pinned, "#FF4FD8", "pinned (terrain)")):
        if m.any():
            ax.plot(aug["ref"][m, 0], aug["ref"][m, 2], ".", color=c, ms=4.5, label=nm, zorder=6)
    ax.plot([src["anchor"][0]], [src["anchor"][2]], "o", ms=9, color="#F28C1E", mec="k",
            label="robot root (original)")
    ax.plot([aug["anchor"][0]], [aug["anchor"][2]], "*", ms=15, color="#12E05A", mec="k",
            label="robot root (DISPLACED)")
    ax.set_xlim(lo, hi)
    zs = np.concatenate(zs_all)
    ax.set_ylim(zs.min() - 0.25, zs.max() + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel(f"{label}\nROOT-LINK z [m]", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    for k, (img, name) in enumerate(((src["depth"], "depth ORIGINAL"),
                                     (aug["depth"], "depth AUGMENTED"))):
        a = axes[1 + k]
        a.imshow(_depth_image(img, spec))
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(name, fontsize=8)
    a = axes[3]
    d = np.asarray(aug["depth"], float) - np.asarray(src["depth"], float)
    h, w = int(spec.get("height", 0)), int(spec.get("width", 0))
    v = float(np.nanmax(np.abs(d))) or 1.0
    a.imshow(np.nan_to_num(d).reshape(h, w), cmap="bwr", vmin=-v, vmax=v)
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(f"difference (max {v:.2f})", fontsize=8)


def main() -> int:
    """Sample augmented windows and render the comparison figure."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Original recording .h5.")
    p.add_argument("--augmented", required=True, help="Its augmented counterpart .h5.")
    p.add_argument("--lib", required=True, help="Clip library, for the tile geometry.")
    p.add_argument("--num", type=int, default=4, help="How many augments to draw.")
    p.add_argument("--family", default="family_stairs_up", help="clip_family name to keep.")
    p.add_argument("--camera", type=int, default=0, help="Which depth camera to show.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p.add_argument("--out", default="plots/augment", help="Output directory.")
    args = p.parse_args()

    import h5py

    from motion_gen.clips.clip_io import load_clip_library
    from motion_gen.world_lift import anchor_mask_is_pinned

    meta = read_meta(args.augmented)
    s_slc = state_slices(meta)
    nj = len(getattr(meta, "action_joint_names", []) or []) or 29
    blocks = list(getattr(meta, "reference_blocks", []) or []) or None
    spec = list(getattr(meta, "depth_camera_spec", []) or [])[args.camera]

    aug_idx, src_idx, cols = pair_windows(args.source, args.augmented)
    names = list(getattr(meta, "family_names", []) or [])
    fam_code = names.index(args.family) if args.family in names else -1
    keep = cols["aug_family"][aug_idx] == fam_code
    aug_idx, src_idx = aug_idx[keep], src_idx[keep]
    print(f"[augment_viewer] {aug_idx.size} augmented windows of family {args.family!r} "
          f"resolve to a source window")
    if aug_idx.size == 0:
        raise SystemExit(f"no augments of family {args.family!r}; families: {names}")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(aug_idx.size, size=min(args.num, aug_idx.size), replace=False)

    lib = load_clip_library(args.lib, device="cpu",
                            keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    pinned_all = anchor_mask_is_pinned(lib.anchor_mask_per_step()).numpy()
    with h5py.File(args.lib, "r") as f:
        manifest = json.loads(f.attrs["terrain_manifest"])
        lib_tid = f["terrain_id"][:].astype(np.int64)
        pos_names = json.loads(f.attrs["pos_output_names"])
        px = [pos_names.index(f"pelvis_link:pos_{a}") for a in "xyz"]
    by_tid: dict[int, dict] = {}
    for e in manifest:
        by_tid.setdefault(int(e["terrain_id"]), e)

    n = len(pick)
    fig, axes = plt.subplots(n, 4, figsize=(20, 3.6 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [2.6, 1, 1, 1]})
    rows = []
    with h5py.File(args.source, "r") as fs, h5py.File(args.augmented, "r") as fa, \
            h5py.File(args.lib, "r") as fl:
        for r, t in enumerate(pick):
            j, i = int(aug_idx[t]), int(src_idx[t])
            cid, cst = int(cols["src_clip_id"][i]), int(cols["src_clip_step"][i])
            hs_s, hs_a = fs["history_state"][i], fa["history_state"][j]
            ap_s = hs_s[-1, slice(*s_slc["root_pos_w"])]
            aq_s = hs_s[-1, slice(*s_slc["root_quat_w"])]
            ap_a = hs_a[-1, slice(*s_slc["root_pos_w"])]
            aq_a = hs_a[-1, slice(*s_slc["root_quat_w"])]
            rf_s, rf_a = fs["ref_future"][i], fa["ref_future"][j]
            F = int(rf_s.shape[0])
            # Lift both windows against their OWN anchors, then place the pair in tile-local
            # coordinates using the clip's pelvis at this step -- the recording carries no tile
            # corner, and the robot tracks the clip closely, so this is exact up to the
            # tracker's own error and is only used to put the terrain behind the curves.
            clip_xyz = fl["y_des"][cid, cst:cst + F, :][:, px]
            p_s, p_a = ref_root_pos(rf_s, s_slc), ref_root_pos(rf_a, s_slc)
            off = env_offset(p_s, clip_xyz)
            w_s, w_a = p_s - off, p_a - off
            base = np.asarray(ap_s, float) - off
            pin = pinned_all[cid, cst:cst + F]
            if pin.size < F:
                pin = np.pad(pin, (0, F - pin.size), constant_values=pin[-1] if pin.size else 0)
            geom = by_tid.get(int(lib_tid[cid]))
            geom = TileGeometry(geom) if geom is not None and "overlays" in geom else None
            if geom is not None:  # tile-local: the clip's own frame, same as `base`
                pass
            disp = np.asarray(ap_a) - np.asarray(ap_s)
            node_shift = w_a - w_s
            hr_s = ref_root_pos(fs["history_ref"][i], s_slc) - off
            hr_a = ref_root_pos(fa["history_ref"][j], s_slc) - off
            src = {"ref": w_s, "anchor": base, "depth": fs["depth"][i][args.camera],
                   "hist": hr_s}
            aug = {"ref": w_a, "anchor": base + disp, "depth": fa["depth"][j][args.camera],
                   "hist_aug": hr_a}
            lbl = (f"clip {cid} step {cst}\nrobot displaced "
                   f"[{disp[0]:+.3f}, {disp[1]:+.3f}] m")
            draw_row(axes[r], src, aug, geom, pin, spec, lbl)
            free = ~pin
            rows.append({
                "clip": cid, "step": cst, "disp": disp,
                "shift_free": float(np.linalg.norm(node_shift[free, :2], axis=1).mean())
                if free.any() else float("nan"),
                "shift_pinned": float(np.linalg.norm(node_shift[pin, :2], axis=1).mean())
                if pin.any() else float("nan"),
                "n_free": int(free.sum()), "n_pinned": int(pin.sum()),
                "depth_diff": float(np.nanmax(np.abs(
                    np.asarray(aug["depth"], float) - np.asarray(src["depth"], float)))),
            })
    axes[0][0].legend(fontsize=6.5, loc="upper left", ncol=2)
    for a in axes[-1]:
        a.set_xlabel("tile-local ROOT-LINK x [m]" if a is axes[-1][0] else "")
    fig.suptitle("registration augmentation, ROOT-LINK (pelvis) x-z: what the FM is trained "
                 "to reconstruct   (displacement moves FREE nodes fully, PINNED laterally only)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"augment_{args.family}_seed{args.seed}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n  clip   step   displacement [m]      mean node shift: free / pinned   depth max|d|")
    for r in rows:
        print(f"  {r['clip']:5d} {r['step']:5d}   [{r['disp'][0]:+.3f}, {r['disp'][1]:+.3f}]   "
              f"{r['shift_free']:.4f} ({r['n_free']:2d}) / {r['shift_pinned']:.4f} "
              f"({r['n_pinned']:2d})        {r['depth_diff']:.3f}")
    print(f"\n[augment_viewer] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
