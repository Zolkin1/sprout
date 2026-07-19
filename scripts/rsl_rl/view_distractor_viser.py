#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Preview distractor-box layouts: stats + a viser view of one tile's clips + boxes.

Groups a flat clip library into tiles, runs the distractor placement per tile (see
``motion_matching/distractor.py``), prints stats (clips/tile, boxes/tile), and -- if ``--group N``
is given -- renders that tile in viser: every clip's pelvis path (where the motions go), the placed
box(es), and the cold+perceivable region (the shared negative space in perception).

    PYTHONPATH=source/robot_rl:source/motion_gen <env_isaaclab py> \
        scripts/rsl_rl/view_distractor_viser.py --flat-lib clips/g1_flat_v1.h5 \
        --group-size 40 --max-clips 800 --group 0 --port 8080
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

from motion_gen.motion_matching.distractor import distractor_layout

_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _xy_idx(names: list[str], link: str) -> list[int]:
    return [names.index(f"{link}:pos_x"), names.index(f"{link}:pos_y")]


def _box_mesh(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float32)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.uint32)
    return v, f


def _load_flat_xy(path: str, max_clips: int | None, clip_len: int | None):
    with h5py.File(path, "r") as f:
        names = json.loads(f.attrs["pos_output_names"])
        n_all, T_all = f["y_des"].shape[0], f["y_des"].shape[1]
        N = n_all if max_clips is None else min(max_clips, n_all)
        T = T_all if clip_len is None else min(clip_len, T_all)
        y = f["y_des"][:N, :T]                                       # [N, T, P]
    pxy, fxy = _xy_idx(names, "pelvis_link"), [_xy_idx(names, l) for l in _FEET]
    pelvis = y[:, :, pxy].astype(np.float64)                        # [N, T, 2]
    feet = np.concatenate([y[:, :, ix] for ix in fxy], axis=1).astype(np.float64)  # [N, 2T, 2]
    body = np.concatenate([pelvis, feet], axis=1)                   # [N, 3T, 2]
    return pelvis, body


def _order_by_heading(pelvis: np.ndarray) -> np.ndarray:
    """Clip indices sorted by net travel heading, so a group shares a direction (coherent tile).

    Consecutive clips in a flat library are direction-shuffled -- grouping them makes a tile whose
    clips fan out everywhere. Sorting by ``atan2`` of the net pelvis displacement clusters
    same-heading clips, so a group is a corridor with a shared, small footprint.
    """
    net = pelvis[:, -1, :] - pelvis[:, 0, :]
    return np.argsort(np.arctan2(net[:, 1], net[:, 0]))


def _groups(order: np.ndarray, gs: int) -> list[list[int]]:
    return [order[g:g + gs].tolist() for g in range(0, len(order), gs)]


def _layout_group(pelvis, body, grp, a):
    return distractor_layout(
        [body[c] for c in grp], [pelvis[c] for c in grp],
        cell=a.cell, tile_margin=a.tile_margin, robot_radius=a.robot_radius,
        scan_fwd_offset=a.scan_fwd_offset, scan_reach=a.scan_reach,
        box_xy=tuple(a.box_size), n_boxes=a.n_boxes, min_sep=a.min_sep,
    )


def _print_stats(groups, nboxes, a):
    nb = np.asarray(nboxes)
    gsz = np.array([len(g) for g in groups])
    print("\n=== distractor layout stats ===")
    print(f"clips used         : {int(gsz.sum())}")
    print(f"tiles (groups)     : {len(groups)}  (--group-size {a.group_size})")
    print(f"clips / tile       : mean {gsz.mean():.1f}  (min {gsz.min()}, max {gsz.max()})")
    print(f"boxes / tile       : mean {nb.mean():.2f}  (min {nb.min()}, max {nb.max()}, target {a.n_boxes})")
    empty = int((nb == 0).sum())
    print(f"tiles with 0 boxes : {empty}  ({100.0 * empty / len(nb):.0f}%)  <- no cold+perceivable room")
    hist = {int(k): int((nb == k).sum()) for k in range(int(nb.max()) + 1)}
    print(f"boxes/tile hist    : {hist}")
    print(f"box footprint      : {a.box_size[0]} x {a.box_size[1]} m, h={a.box_height} | "
          f"scan fwd={a.scan_fwd_offset} reach={a.scan_reach} | robot r={a.robot_radius}")


def _render(grp, pelvis, boxes, dbg, a):
    import matplotlib.cm as cm
    import viser

    origin, shape, heat, cold = dbg
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    ext = np.array(shape) * a.cell
    server.scene.add_grid("/grid", width=float(ext[0]), height=float(ext[1]), cell_size=1.0,
                          position=(float(origin[0] + ext[0] / 2), float(origin[1] + ext[1] / 2), 0.0))
    # each clip's pelvis path (colored per clip)
    for i, c in enumerate(grp):
        p = pelvis[c]
        pts = np.concatenate([p, np.full((len(p), 1), 0.08)], axis=1).astype(np.float32)
        col = np.array(cm.tab20(i % 20)[:3], np.float32)
        server.scene.add_point_cloud(f"/paths/clip{c}", points=pts,
                                     colors=np.tile(col, (len(pts), 1)), point_size=0.03)
    # the placed boxes (green, walkable)
    for bi, (bx, by) in enumerate(boxes):
        sx, sy, sz = a.box_size[0], a.box_size[1], a.box_height
        v, f = _box_mesh(bx - sx / 2, bx + sx / 2, by - sy / 2, by + sy / 2, 0.0, sz)
        server.scene.add_mesh_simple(f"/box{bi}", vertices=v, faces=f, color=(0.2, 0.8, 0.2),
                                     side="double", flat_shading=True)
    # cold+perceivable cells (faint) = the shared negative space the boxes drop into
    ci, cj = np.where(cold)
    cpts = np.stack([origin[0] + ci * a.cell, origin[1] + cj * a.cell, np.full(len(ci), 0.02)], 1)
    if len(cpts):
        server.scene.add_point_cloud("/cold", points=cpts.astype(np.float32),
                                     colors=np.tile(np.array([0.55, 0.55, 0.9], np.float32), (len(cpts), 1)),
                                     point_size=0.02)
    print(f"[distractor] tile group {a.group}: {len(grp)} clips, {len(boxes)} boxes  ->  "
          f"http://localhost:{a.port}  (Ctrl-C to stop)")
    import time
    while True:
        time.sleep(1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flat-lib", required=True, help="Source flat clip library (.h5).")
    p.add_argument("--group-size", type=int, default=40, help="Clips per tile (-> #tiles = N/group).")
    p.add_argument("--group-by", choices=["index", "heading"], default="heading",
                   help="Tile formation: 'heading' (same-direction clips = coherent corridor tile) "
                        "or 'index' (consecutive clips; direction-shuffled -> fans out).")
    p.add_argument("--max-clips", type=int, default=800, help="Cap clips loaded (speed).")
    p.add_argument("--clip-len", type=int, default=None, help="Truncate each clip to this many steps.")
    p.add_argument("--box-size", type=float, nargs=2, default=[0.8, 0.45], help="Box full extent x y (m).")
    p.add_argument("--box-height", type=float, default=0.47)
    p.add_argument("--n-boxes", type=int, default=3, help="Max boxes to place per tile.")
    p.add_argument("--scan-fwd-offset", type=float, default=0.75)
    p.add_argument("--scan-reach", type=float, default=0.9)
    p.add_argument("--robot-radius", type=float, default=0.25)
    p.add_argument("--cell", type=float, default=0.1)
    p.add_argument("--tile-margin", type=float, default=2.0)
    p.add_argument("--min-sep", type=float, default=0.6)
    p.add_argument("--group", type=int, default=None, help="Tile index to render in viser (omit = stats only).")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()

    pelvis, body = _load_flat_xy(a.flat_lib, a.max_clips, a.clip_len)
    order = _order_by_heading(pelvis) if a.group_by == "heading" else np.arange(len(pelvis))
    groups = _groups(order, a.group_size)
    print(f"[distractor] {len(pelvis)} clips -> {len(groups)} tiles ({a.group_by}-grouped); placing ...")
    layouts = [_layout_group(pelvis, body, grp, a) for grp in groups]
    _print_stats(groups, [len(b) for b, _ in layouts], a)

    if a.group is not None:
        g = a.group
        boxes, dbg = layouts[g]
        _render(groups[g], pelvis, boxes, dbg, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
