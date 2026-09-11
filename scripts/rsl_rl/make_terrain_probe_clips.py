#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Build the probe library for ``fm_terrain_probe.py``: ONE flat clip, N tiles, N sceneries.

The question is what the generator PLANS when it sees a given piece of geometry ahead of it,
and the only honest way to ask it is to hold everything else fixed. So this library is one
flat clip copied N times onto N flat tiles, where the tiles differ **only** in their
``flat_tile_decor`` entry. Same motion, same command, same phase, same ground -- one variable.

**Everything placed here is DECOR, and that is what makes the comparison clean.** The decor
prim carries no collider and is not in the height scan, so across all N envs the physics, the
contact schedule, the reference and the privileged scan are bit-identical; the depth cameras
are the only thing that can differ. A condition that changed the ground would be measuring two
things at once.

Two geometric facts have to be got right or every placement is silently wrong:

* **Flat clips are re-centred on their tile by BBOX at reset** (``mdp/events/resets.py``:
  ``world_pos = cell_center - bbox_center``), not anchored at their origin. So the robot at
  clip step ``s`` sits at ``pel_xy[s] - bbox_center`` relative to the tile centre, and that is
  the offset every condition is authored around. ``_flat_decor_overlays`` then adds the tile
  centre back at load.
* **Overlays are AXIS-ALIGNED** -- ``build_overlay`` raises on a non-zero yaw -- so "3 m
  ahead" is only meaningful if the robot is walking along a world axis at the probe step.
  This refuses to build a library whose clip is not, rather than emitting boxes that sit off
  to one side of the path. ``--scan`` finds a segment that qualifies.

    python scripts/rsl_rl/make_terrain_probe_clips.py --clip <lib>.h5 --scan
    python scripts/rsl_rl/make_terrain_probe_clips.py --clip <lib>.h5 \\
        --clip_id 8869 --start_step 380 --out clips/_terrain_probe.h5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "source", "motion_gen"))

from motion_gen.clips.clip_io import write_clip_subset  # noqa: E402

#: Wall thickness and lateral half-width for every vertical surface here (m). The wall is made
#: WIDE (12 m across) on purpose: the clip holds only approximately to a world axis, so a
#: narrow wall could sit beside the path rather than across it and the condition would
#: silently become "a wall in my peripheral vision".
WALL_T, WALL_HALF_W = 0.15, 6.0

#: One riser of the shipped stair library. Every "is a small vertical edge read as a stair?"
#: condition uses exactly this, so the probe asks about the geometry the robot was trained on.
RISER = 0.13


def _box(x, y, z0, sx, sy, sz, cls):
    """One axis-aligned decor box, centred at ``(x, y)`` with its base at ``z0``."""
    return {"kind": "box", "walkable_top": False, "yaw": 0.0, "variant": "solid",
            "size_xyz": [float(sx), float(sy), float(sz)],
            "pos_xy": [float(x), float(y)], "base_z": float(z0),
            "decor": True, "probe_class": cls}


def conditions() -> list[tuple[str, list[dict]]]:
    """The scenery for each tile, in the robot's own forward frame at the probe step.

    ``+x`` is downrange, ``y`` is lateral, ``z = 0`` is the ground the robot walks on. The
    caller rotates/translates these onto the tile.

    Returns:
        ``(name, overlays)`` per condition; the first is the empty control every rate is read
        against.
    """
    out: list[tuple[str, list[dict]]] = [("empty", [])]
    # Distance sweep at a fixed, unambiguously-wall height.
    for d in (1.5, 2.0, 3.0, 4.0, 6.0):
        out.append((f"wall_d{d}_h2.0",
                    [_box(d, 0.0, 0.0, WALL_T, 2 * WALL_HALF_W, 2.0, "wall")]))
    # Height sweep at a fixed distance. 1.0 m is the interesting one -- waist height is where
    # a wall is most box-like.
    for h in (1.0, 1.4, 3.0):
        out.append((f"wall_d3.0_h{h}",
                    [_box(3.0, 0.0, 0.0, WALL_T, 2 * WALL_HALF_W, h, "wall")]))
    # A wall with one riser at its foot: the "false riser" -- locally a step, globally a wall.
    out.append(("false_riser_d3.0",
                [_box(3.0, 0.0, 0.0, WALL_T, 2 * WALL_HALF_W, 2.0, "wall"),
                 _box(3.0 - 0.15, 0.0, 0.0, 0.30, 2 * WALL_HALF_W, RISER, "riser")]))
    # A single riser-height ledge across the path, with nothing above it. Does one step edge
    # read as a staircase?
    out.append((f"ledge_ahead_d3.0",
                [_box(3.0, 0.0, 0.0, 0.30, 2 * WALL_HALF_W, RISER, "riser")]))
    # THE PLINTH CUE, isolated. Every stair-transition tile with a floor stands the robot on a
    # slab one riser above the ground, so a 0.13 m ledge runs the length of the corridor on
    # both sides -- present in every stairs window and in nothing else. Here it appears with
    # no stairs anywhere. If the generator plans a climb, that cue is doing work it should not.
    out.append(("plinth_edges_beside",
                [_box(4.0, +1.5, 0.0, 16.0, 0.30, RISER, "riser"),
                 _box(4.0, -1.5, 0.0, 16.0, 0.30, RISER, "riser")]))
    # Positive controls: geometry the robot SHOULD read as its two terrain kinds.
    # SANITY. A 3 m box wall on all four sides at 2.5 m: geometry the cameras cannot miss.
    # If this does not dominate the stimulus check, the placement arithmetic is wrong and
    # every other row is meaningless -- so it is a permanent condition, not a debug detour.
    out.append(("enclosure_2.5",
                [_box(+2.5, 0.0, 0.0, WALL_T, 10.0, 3.0, "sanity"),
                 _box(-2.5, 0.0, 0.0, WALL_T, 10.0, 3.0, "sanity"),
                 _box(0.0, +2.5, 0.0, 10.0, WALL_T, 3.0, "sanity"),
                 _box(0.0, -2.5, 0.0, 10.0, WALL_T, 3.0, "sanity")]))
    out.append(("box_d3.0", [_box(3.0, 0.0, 0.0, 1.0, 2.0, 0.40, "box")]))
    out.append(("stairs_d3.0",
                [_box(3.0 + i * 0.30, 0.0, 0.0, 0.30, 2.0, RISER * (i + 1), "stairs")
                 for i in range(4)]))
    return out


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a wxyz quaternion."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def clip_geometry(path: str, clip_id: int, start_step: int) -> tuple[np.ndarray, float, float, float]:
    """Where the robot is relative to its tile centre, which way it faces, and its torso pitch.

    The torso pitch is not a diagnostic extra -- it decides what the probe can see at all.
    BOTH cameras mount on ``torso_link``, so the far lens's nominal 15 deg down is measured
    from a torso that is never level: median +10 deg nose-down on flat and +16 on stairs_up.
    That drags its boresight to 25-31 deg and its ground intersection to 2.3-3.2 m against a
    nominal 4.1 m, and at the p95 nose-down instant a 2 m wall at 4 m falls entirely above
    the top of the frame. Choosing the probe step without looking at this silently decides
    which conditions are visible.

    Args:
        path: Source clip library.
        clip_id: Which clip.
        start_step: The step the probe fires on.

    Returns:
        ``(offset_xy, yaw_rad, speed, torso_pitch_deg, pose)`` -- position relative to the
        tile centre under the flat re-centring rule, heading, planar speed, world torso pitch
        (positive = nose down), and the joint configuration plus yaw-removed root rotation
        needed to draw the robot in the probe's own frame.
    """
    with h5py.File(path, "r") as f:
        n = int(f["clip_len"][clip_id])
        row = f["y_des"][clip_id][:n]
        names = json.loads(f.attrs["pos_output_names"])
    pel, quat = row[:, 0:3], row[:, 3:7]
    bbox_center = 0.5 * (pel[:, :2].max(axis=0) + pel[:, :2].min(axis=0))
    w, x, y, z = quat[start_step]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    spd = float(np.linalg.norm(pel[min(start_step + 1, n - 1), :2] - pel[start_step, :2]) / 0.02)
    q_t = np.array([row[start_step, names.index(f"torso_link@root:ori_{a}")] for a in "wxyz"])
    r_w = _rot(quat[start_step]) @ _rot(q_t)
    pitch = float(np.degrees(np.arcsin(np.clip(-r_w[2, 0], -1.0, 1.0))))

    # The robot's pose IN THE ROBOT FRAME the conditions are authored in. That frame is
    # yaw-aligned, so the heading cancels and what is left is the roll/pitch -- which must be
    # kept, not zeroed: the torso lean is the thing that decides what the cameras see, so a
    # viewer that draws the robot bolt upright would show a scene the cameras never had.
    c, s_ = np.cos(-yaw / 2.0), np.sin(-yaw / 2.0)
    q_local = _qmul(np.array([c, 0.0, 0.0, s_]), quat[start_step])
    jn = [n[len("joint:"):] for n in names if n.startswith("joint:")]
    jp = [float(row[start_step, names.index(f"joint:{n}")]) for n in jn]
    pose = {"joint_names": jn, "joint_pos": jp,
            "root_quat_wxyz": [float(v) for v in q_local],
            "root_z": float(pel[start_step, 2])}
    return pel[start_step, :2] - bbox_center, yaw, spd, pitch, pose


def scan(path: str, max_yaw: float, min_speed: float, horizon: int, n_try: int) -> None:
    """Print flat clips that hold near a world axis, with forward speed, over ``horizon`` steps."""
    with h5py.File(path, "r") as f:
        tid, clen = f["terrain_id"][:], f["clip_len"][:]
        flat = np.flatnonzero(tid == -1)
        rng = np.random.default_rng(0)
        found = []
        for c in rng.choice(flat, size=min(n_try, len(flat)), replace=False):
            n = int(clen[c])
            row = f["y_des"][int(c)][:n]
            q, pel = row[:, 3:7], row[:, 0:3]
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            spd = np.r_[0.0, np.linalg.norm(np.diff(pel[:, :2], axis=0), axis=1) / 0.02]
            for s in range(60, max(61, n - horizon), 20):
                seg, sp = yaw[s:s + horizon], spd[s:s + horizon]
                if np.abs(seg).max() < max_yaw and sp.min() > min_speed:
                    found.append((float(np.abs(seg).max()), int(c), int(s), float(sp.mean())))
                    break
    found.sort()
    print(f"{'max|yaw| (rad)':>15s} {'clip':>7s} {'step':>6s} {'mean speed':>11s}")
    for r in found[:10]:
        print(f"{r[0]:15.4f} {r[1]:7d} {r[2]:6d} {r[3]:11.2f}")
    if not found:
        print("none -- relax --max_yaw or --min_speed")


def replicate_clips(path: str, n: int) -> None:
    """Tile a one-clip library into ``n`` identical clips, in place.

    ``write_clip_subset`` refuses duplicate indices, correctly -- for its normal callers a
    repeated clip is a mistake. Here it is the design: the flat-tile count is derived from the
    flat CLIP count, so N tiles need N clips, and the clips must be identical because the
    motion is the control variable this whole probe holds fixed.

    Args:
        path: A library holding exactly one clip; rewritten in place.
        n: How many copies to end up with.
    """
    with h5py.File(path, "r+") as f:
        for key in list(f.keys()):
            arr = np.asarray(f[key][:])
            if arr.shape[0] != 1:
                raise SystemExit(f"{path}: '{key}' has {arr.shape[0]} rows, expected 1")
            tiled = np.repeat(arr, n, axis=0)
            del f[key]
            f.create_dataset(key, data=tiled)
        for k in ("num_clips", "num_clips_written"):
            if k in f.attrs:
                f.attrs[k] = n
        gc = json.loads(f.attrs["generator_config"]) if "generator_config" in f.attrs else {}
        gc.setdefault("counts", {})["flat"] = n
        f.attrs["generator_config"] = json.dumps(gc)


def main() -> int:
    """Build the probe library, or scan for a clip to build it from."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Source clip library .h5.")
    p.add_argument("--out", default="clips/_terrain_probe.h5")
    # Defaults sit at the flat family's MEDIAN torso pitch (10.3 deg). The first pair used
    # here was the p95 nose-down instant, which put every condition beyond 3 m out of frame.
    p.add_argument("--clip_id", type=int, default=9073)
    p.add_argument("--start_step", type=int, default=190)
    p.add_argument("--scan", action="store_true", help="List usable (clip, step) pairs and exit.")
    p.add_argument("--max_yaw", type=float, default=0.15,
                   help="Refuse a clip whose heading strays further than this (rad) from a "
                        "world axis over --horizon steps; overlays cannot be rotated.")
    p.add_argument("--min_speed", type=float, default=0.6,
                   help="The latch question is about a robot MOVING toward the geometry.")
    p.add_argument("--horizon", type=int, default=150, help="Steps the constraints must hold.")
    p.add_argument("--n_try", type=int, default=400, help="Flat clips to consider when scanning.")
    a = p.parse_args()

    if a.scan:
        scan(a.clip, a.max_yaw, a.min_speed, a.horizon, a.n_try)
        return 0

    conds = conditions()
    off, yaw, spd, pitch, pose = clip_geometry(a.clip, a.clip_id, a.start_step)
    if abs(yaw) > a.max_yaw:
        raise SystemExit(
            f"clip {a.clip_id} step {a.start_step} has yaw {yaw:+.3f} rad, beyond --max_yaw "
            f"{a.max_yaw}. Overlays are axis-aligned (build_overlay raises on a non-zero yaw), "
            "so a rotated heading would put every condition off to one side of the path "
            "instead of across it. Run --scan to find a segment that qualifies."
        )
    ground = 1.1 / np.tan(np.radians(max(1.0, pitch + 15.0)))
    print(f"[probe-lib] clip {a.clip_id} step {a.start_step}: yaw {yaw:+.3f} rad, "
          f"{spd:.2f} m/s, offset from tile centre [{off[0]:+.2f}, {off[1]:+.2f}] m")
    print(f"[probe-lib] torso pitch {pitch:+.2f} deg nose-down -> far camera boresight "
          f"{pitch + 15.0:.1f} deg down, meeting the ground near {ground:.1f} m "
          f"(nominal, torso level, is 4.1 m). Conditions beyond that are seen only by the "
          f"UPPER half of the frame, and a tall one can sit above it entirely.")

    # One copy of the clip per condition: flat tiles are sized from the FLAT CLIP COUNT, so N
    # tiles need N clips. They are identical, which is the point -- the motion is the control.
    write_clip_subset(a.clip, a.out, [a.clip_id])
    replicate_clips(a.out, len(conds))

    c, s = np.cos(yaw), np.sin(yaw)
    variants: list[list[dict]] = []
    for _name, boxes in conds:
        placed = []
        for b in boxes:
            bx, by = b["pos_xy"]
            placed.append({**b, "pos_xy": [float(off[0] + c * bx - s * by),
                                           float(off[1] + s * bx + c * by)]})
        variants.append(placed)

    with h5py.File(a.out, "r+") as f:
        f.attrs["flat_tile_decor"] = json.dumps(variants)
        f.attrs["terrain_probe"] = json.dumps({
            "source": os.path.basename(a.clip), "clip_id": a.clip_id,
            "start_step": a.start_step, "yaw": yaw, "speed": spd,
            "torso_pitch_deg": pitch, "far_cam_ground_m": float(ground),
            "pose": pose,
            "offset_xy": [float(off[0]), float(off[1])],
            "conditions": [n for n, _ in conds],
            # The boxes in the ROBOT's own frame at the probe instant, before placement.
            # The viewer draws every condition at the origin so they can be flipped between
            # without the camera moving, and re-deriving these from world coordinates would
            # mean inverting the placement it is supposed to be checking.
            "conditions_local": [boxes for _n, boxes in conds],
        })
    print(f"[probe-lib] wrote {a.out}: {len(conds)} tiles / conditions")
    for i, (name, boxes) in enumerate(conds):
        print(f"    tile {i:2d}  {name:22s} {len(boxes)} box(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
