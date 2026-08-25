#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Where the robot ACTUALLY went, per tile, so clutter can be placed clear of it.

Clutter has to miss the robot, and the robot is not the clip. The clip is what it was asked
to do; a recording is where the closed loop put it, and the two differ by the tracker's own
error plus the runtime lateral ego re-anchor (p99 0.21 m from the reset placement, on its
own). Since the visual re-render replays the RECORDED poses, the recorded trajectory is
exactly what the boxes must not intersect -- placing against the clips would put clutter
where the real robot walks, and no amount of care in the authoring pass would catch it.

The output is an OCCUPANCY GRID per ``terrain_id``, not an envelope. A tile hosts many clips
and a closed-loop rollout of all of them covers most of its area, so a bounding box would
reject every candidate and the clutter pass would emit nothing.

**Which tile a sample belongs to is resolved geometrically**, by containment in the tile
layout, rather than by reproducing the env->tile assignment the recorder happened to draw.
That assignment depends on ``num_envs`` and the importer's origin shuffle; a point-in-tile
lookup depends on neither, and tiles are laid out disjointly so it is unambiguous. Rows of
the layout are replicas, and several tiles can share a ``terrain_id``, so their samples pool
-- which is right, because the manifest that clutter is written into is per ``terrain_id``.

Needs IsaacLab (the tile layout is assembled by the RL-side loader). The clutter pass that
consumes this does not.

    python scripts/rsl_rl/build_clutter_occupancy.py \\
        --data <recording>.h5 --clip <library>.h5 --out occupancy.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "source", "robot_rl"))
sys.path.insert(0, os.path.join(_REPO, "source", "motion_gen"))

from motion_gen.clips.clip_io import load_clip_library  # noqa: E402
from motion_gen.terrain.clutter import occupancy_from_positions  # noqa: E402
from recorded_data_io import load_recorded_windows  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (  # noqa: E402
    assemble_clip_terrain,
)


def recorded_body_xy(data: dict, meta) -> tuple[np.ndarray, list[str]]:
    """World xy of EVERY tracked body, per recorded window: ``(N, B, 2)``.

    Not just the root. The recording stores the ankles and wrists in world alongside it, and
    they are the parts that reach: on a staircase the swing foot gets a quarter of a metre
    ahead of the pelvis, so a root-only grid has to be padded by the robot's whole reach and
    still cannot tell a foot swinging toward a box from one swinging away.

    The SNAP row -- the last history row -- is the pose the window was anchored at, which is
    the pose the re-render places the robot at. Earlier rows are the same trajectory a few
    steps back and would only blur the grid.

    Args:
        data: The loaded recording arrays.
        meta: Layout metadata; ``state_term_names`` / ``state_term_dims`` locate each body.

    Returns:
        ``((N, B, 2) world xy, body names)``.
    """
    names = list(meta.state_term_names)
    dims = list(meta.state_term_dims)
    wanted = [n for n in names if n == "root_pos_w" or n.endswith(":pos_w")]
    if "root_pos_w" not in wanted:
        raise SystemExit(f"recording has no root_pos_w in its state layout: {names}")
    rows = np.asarray(data["history_state"])[:, -1, :]
    out = []
    for n in wanted:
        st = int(sum(dims[: names.index(n)]))
        out.append(rows[:, st:st + 2])
    return np.stack(out, axis=1), [n.replace(":pos_w", "").replace("root_pos_w", "root")
                                   for n in wanted]


def densify_by_episode(xy: np.ndarray, episode: np.ndarray, step: np.ndarray,
                       spacing: float, max_gap_m: float = 3.0) -> np.ndarray:
    """Interpolate between consecutive samples of the same episode, at ``spacing`` metres.

    Args:
        xy: ``(N, 2)`` world positions, one per window.
        episode: ``(N,)`` episode id.
        step: ``(N,)`` clip step, for ordering within an episode.
        spacing: Point pitch along the path (m).
        max_gap_m: Pairs further apart than this are NOT joined -- a reset teleports the
            robot across the map, and a straight line between two episodes would sweep a
            corridor through tiles it never visited, vetoing clutter everywhere.

    Returns:
        ``(M, 2)`` positions, including the originals.
    """
    order = np.lexsort((step, episode))
    xy, episode = xy[order], episode[order]
    a, b = xy[:-1], xy[1:]
    seg = np.linalg.norm(b - a, axis=1)
    ok = (episode[1:] == episode[:-1]) & (seg > 1e-6) & (seg <= max_gap_m)
    out = [xy]
    n_mid = np.ceil(seg[ok] / max(spacing, 1e-6)).astype(int) - 1
    for k in range(1, int(n_mid.max()) + 1 if len(n_mid) else 1):
        sel = ok.copy()
        sel[ok] = n_mid >= k
        if not sel.any():
            break
        t = (k / (n_mid[n_mid >= k] + 1.0))[:, None]
        out.append(a[sel] + t * (b[sel] - a[sel]))
    return np.concatenate(out)


def main() -> int:
    """Build and write the per-tile occupancy."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Windowed recording .h5.")
    p.add_argument("--clip", default=None,
                   help="Clip library the recording was made against. Default: the one the "
                        "recording names, which is the only one whose tiles are where the "
                        "recorded world poses expect them.")
    p.add_argument("--out", required=True, help="Occupancy .npz.")
    p.add_argument("--cell", type=float, default=0.25, help="Grid pitch (m).")
    p.add_argument("--no_clip_paths", action="store_true",
                   help="Veto against the RECORDING only. Off by default: the recording is a "
                        "subset of where the clips go, so a recording-only grid leaves "
                        "clutter on the parts of a clip it happened not to cover -- safe for "
                        "re-rendering that one recording, unsafe for anything else.")
    p.add_argument("--max_clips_per_tile", type=int, default=4000,
                   help="Clips sampled per tile for the union. A terrain tile has ~50, so "
                        "this only ever bites on FLAT, which pools 3000 -- and it must not "
                        "bite there either: any flat clip can be drawn onto any flat tile, so "
                        "a sampled subset leaves clutter on the paths of the clips it "
                        "skipped. Measured at a cap of 60: terrain came out clean and every "
                        "residual intersection was flat.")
    p.add_argument("--num_rows", type=int, default=0,
                   help="Tile-set replicas, as recorded. 0 = DETECT, by taking the smallest "
                        "that leaves no recorded pose outside the layout. The recorder does "
                        "not store it and the importer's default is 1, which on a real "
                        "recording leaves ~98%% of poses unplaced -- silently, and in the "
                        "permissive direction, so the clutter pass would happily place boxes "
                        "on top of the robot.")
    p.add_argument("--max_windows", type=int, default=0,
                   help="Cap on rows read (0 = all, and 0 is what you want). A cap SUBSAMPLES "
                        "the recording, which multiplies the distance between consecutive "
                        "path samples and leaves holes a clutter box fits inside -- observed. "
                        "Capping also disables the densify pass, so it is a debugging aid "
                        "only.")
    a = p.parse_args()

    data, meta = load_recorded_windows(
        a.data, keys=["history_state", "clip_id", "episode_id", "clip_step"])
    xyb, body_names = recorded_body_xy(data, meta)
    xy = xyb[:, 0, :]                                   # the root drives episode ordering
    if a.max_windows and len(xy) > a.max_windows:
        xy = xy[np.linspace(0, len(xy) - 1, a.max_windows).astype(int)]
    else:
        # DENSIFY along each episode. A window is a point, and consecutive windows are a
        # re-plan apart -- a median 0.14 m but up to 0.93 m -- so binning the points alone
        # leaves the path perforated and a box can clear every sample while sitting on the
        # trajectory between two of them. Subsampling makes it far worse (it scales every
        # gap), which is how intersecting clutter first appeared. Interpolating turns the
        # samples back into the continuous path they came from, and makes the result
        # independent of the recording's cadence.
        ep = np.asarray(data["episode_id"]).reshape(-1)
        stp = np.asarray(data["clip_step"]).reshape(-1)
        parts = [densify_by_episode(xyb[:, b, :], ep, stp, a.cell / 2.0)
                 for b in range(xyb.shape[1])]
        xy = np.concatenate(parts)
        print(f"[occupancy] densified along {len(np.unique(ep))} episodes x "
              f"{len(body_names)} tracked bodies {body_names} -> {len(xy)} points")
    lib_path = a.clip or meta.clip_library_path
    print(f"[occupancy] {len(xy)} recorded poses; tiles from {lib_path}")

    lib = load_clip_library(lib_path, "cpu")

    def _assign(num_rows: int):
        """Point-in-tile assignment for one candidate layout."""
        _m, _o, lay = assemble_clip_terrain(lib, num_rows=num_rows)
        c, f, t = (lay.tile_corner.numpy(), lay.tile_footprint.numpy(),
                   lay.tile_terrain_id.numpy())
        acc: dict[int, list[np.ndarray]] = {}
        left = np.ones(len(xy), dtype=bool)
        for r in range(c.shape[0]):
            for k in range(c.shape[1]):
                cx, cy = float(c[r, k, 0]), float(c[r, k, 1])
                fx, fy = float(f[r, k, 0]), float(f[r, k, 1])
                ins = ((xy[:, 0] >= cx) & (xy[:, 0] < cx + fx)
                       & (xy[:, 1] >= cy) & (xy[:, 1] < cy + fy))
                if not ins.any():
                    continue
                left &= ~ins
                acc.setdefault(int(t[r, k]), []).append(xy[ins] - np.array([cx, cy]))
        return acc, left, (c, f, t)

    if a.num_rows:
        per, unplaced, (corner, fp, tid) = _assign(a.num_rows)
    else:
        # Smallest layout that contains every pose. Ascending, so a larger replica count that
        # also contains them (any multiple does) cannot be chosen over the true one.
        for cand in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32):
            per, unplaced, (corner, fp, tid) = _assign(cand)
            if not unplaced.any():
                print(f"[occupancy] detected num_rows={cand} (no pose outside the layout)")
                break
        else:
            raise SystemExit(
                "could not find a tile-set replica count that contains every recorded pose. "
                "Either --clip is not the library the recording was made against, or the "
                "layout knobs (col_gap / size_margin / flat_buffer) have moved since it was "
                "recorded. Pass --num_rows to override, but a partial fit means the grid is "
                "permissive exactly where it is wrong.")

    if unplaced.any():
        print(f"[occupancy] WARNING {unplaced.sum()} of {len(xy)} poses fell in no tile "
              "(gaps between tiles, or a layout that does not match the recording). They "
              "are dropped, which makes the grid PERMISSIVE -- check --num_rows first.")

    sizes = {int(tid[r, t]): (float(fp[r, t, 0]), float(fp[r, t, 1]))
             for r in range(corner.shape[0]) for t in range(corner.shape[1])}

    # UNION IN THE CLIP PATHS. The recording is where the robot went; the clips are where it
    # is ever ASKED to go, and the first is a subset of the second -- measured, a clip on one
    # tile spans 3.1 m of downrange while the recording covered only the last 1.4 m of it,
    # because not every clip ran and not every episode ran to the end. Vetting against the
    # recording alone is right for re-rendering THIS recording and wrong for everything else:
    # the same library used for a fresh recording, or for RL, would put the robot into boxes
    # on the parts of a clip this recording happened to miss.
    #
    # Both are needed, not either. The recording carries the tracker's error and the runtime
    # ego re-anchor, which the clip does not; the clip carries the coverage the recording
    # lacks. Their union is the only set that is safe in both directions.
    if not a.no_clip_paths:
        y_des, clip_tid, clip_len = lib.y_des, lib.terrain_id.numpy(), lib.clip_len.numpy()
        # Every WORLD body the clip stores. The ``@root`` channels are body-relative and
        # would be meaningless here -- comparing them against tile coordinates puts a wrist
        # twenty metres away.
        # ``lib.meta`` first, ``lib.attrs`` second -- the loader promotes some root
        # attributes into meta and this is one of them, so reaching straight for attrs raises
        # a KeyError. It did, and the failure was quiet in the worst way: the union block
        # died AFTER the recorded half had been binned, so the previous occupancy file stayed
        # on disk and everything downstream kept using it, one revision stale.
        raw_names = lib.meta.get("pos_output_names") if getattr(lib, "meta", None) else None
        if raw_names is None:
            raw_names = lib.attrs.get("pos_output_names")
        if raw_names is None:
            raise SystemExit(
                "clip library carries no 'pos_output_names', so the per-body columns cannot "
                "be located. Pass --no_clip_paths to veto against the recording alone.")
        pos_names = json.loads(raw_names) if isinstance(raw_names, str) else list(raw_names)
        body_cols = [[pos_names.index(f"{b}:pos_{ax}") for ax in "xy"]
                     for b in sorted({n.split(":")[0] for n in pos_names if ":pos_x" in n})
                     if "@root" not in b]
        added = 0
        for k in list(per):
            members = np.flatnonzero(clip_tid == k)
            if not len(members):
                continue
            if len(members) > a.max_clips_per_tile:
                members = members[np.linspace(0, len(members) - 1,
                                              a.max_clips_per_tile).astype(int)]
            chunks = []
            for c in members:
                n = int(clip_len[c])
                xyc = np.concatenate([np.asarray(y_des[int(c), :n, cc]) for cc in body_cols])
                if k == -1:
                    # Flat clips are re-centred on their tile by BBOX at reset, so their
                    # baked coordinates are not where the robot stands; the same rule the
                    # reset applies has to be applied here or the union is in the wrong place.
                    # Re-centred on the PELVIS bbox, the rule the reset applies -- every
                    # body shifts by the same offset, so it is computed once from the root.
                    pel = np.asarray(y_des[int(c), :n, body_cols[0]])
                    bb = 0.5 * (pel.max(axis=0) + pel.min(axis=0))
                    xyc = xyc - bb + np.array(sizes[k]) / 2.0
                chunks.append(xyc)
            pts_c = np.concatenate(chunks)
            per[k].append(pts_c)
            added += len(pts_c)
        print(f"[occupancy] unioned {added} clip-path points over {len(per)} terrain ids")
    out: dict[str, np.ndarray] = {}
    print(f"  {'terrain_id':>10s} {'poses':>8s} {'cells':>8s} {'covered':>9s}")
    for k, chunks in sorted(per.items()):
        pts = np.concatenate(chunks)
        occ = occupancy_from_positions(k, pts, *sizes[k], cell=a.cell)
        out[f"grid_{k}"] = occ.grid
        out[f"meta_{k}"] = np.asarray([occ.cell, occ.origin[0], occ.origin[1],
                                       occ.n_samples, *sizes[k]], dtype=np.float64)
        # The raw points as well as the grid. The authoring pass places against the grid and
        # then re-checks against THESE, so a discretisation bug in the grid cannot pass both
        # -- the two disagree by construction, which is the point of keeping both.
        # Decimated only if very large. Flat pools every clip in the library, so its point
        # set dwarfs a terrain tile's and a small cap would make the independent check
        # weakest exactly where the veto has the most to check.
        cap = 400000
        keep = (pts if len(pts) <= cap
                else pts[np.linspace(0, len(pts) - 1, cap).astype(int)])
        out[f"pts_{k}"] = keep.astype(np.float32)
        print(f"  {k:10d} {len(pts):8d} {occ.grid.size:8d} {occ.grid.mean():9.4f}")
    # Written via a temp and moved into place. The union block once died midway and left the
    # PREVIOUS occupancy on disk, which every downstream step then used as if it were current
    # -- a stale artifact that looks exactly like a fresh one is worse than no artifact.
    # ``.npz`` on the temp too: numpy APPENDS the extension when it is absent, so a
    # bare ".partial" is written as ".partial.npz" and the move then fails.
    tmp = a.out + ".partial.npz"
    np.savez_compressed(tmp, terrain_ids=np.asarray(sorted(per)), **out)
    os.replace(tmp, a.out)
    print(f"[occupancy] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
