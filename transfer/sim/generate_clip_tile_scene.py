# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate a MuJoCo scene holding EXACTLY ONE tile from a clip library, as Isaac builds it.

``generate_stairs_scene.py`` emits an in-distribution *playground*: stairs whose rise, run
and width are drawn from the training distribution, but which are not any particular trained
tile.  That is the right thing for a joystick session and the wrong thing for a bisect --
when the generator misreads a staircase you want the staircase it was trained on, not one
like it.  This emits that: one manifest tile, geometry for geometry.

**The geometry is not re-derived.**  The clip library's ``terrain_manifest`` is the artifact
the generators EMIT and the RL side re-reads (``clip_terrain_loader._overlays_from_manifest``
builds Isaac's tile from these same dicts), so reading it here is reading the one producer,
not a second opinion.  ``source/robot_rl/tests/test_clip_tile_scene.py`` casts a height field
over both this scene and the loader's own trimesh tile and requires them to agree.

What a stairs_up TRANSITION tile actually is, which is the thing the playground gets wrong::

    22.86 x 4.46 m floor
    + 10.1 m platform at 0.13 m          <- the robot approaches ALREADY one step up
    + 3 steps, 0.22 m tread, 0.955 m wide, at y = 1.87 of a 4.46 m tile   <- OFF-CENTRE
    + 10.1 m platform at 0.39 m

i.e. a 0.26 m climb over three narrow treads between two long platforms -- not a flight of
six 0.15 m risers on open ground.

Frame: the tile is translated so the SPAWN POINT is the world origin and the terrain under it
is z = 0, so the robot spawns exactly as it does on flat ground (no ``--spawn_z_offset`` to
remember, and nothing spawns inside a slab).  By default the spawn point is on the approach
platform, ``--approach`` metres before the first riser and laterally centred **on the
flight**, so driving straight ahead walks the stairs.

Run from ``transfer/sim`` (so the output lands next to the other scenes)::

    python generate_clip_tile_scene.py --clip ../../clips/_tile_stairs_up.h5 --robot g1_29j
    python gen_tracker_runner.py --load_run <run> --robot_name g1_29j --scene tile0_scene ...

No isaaclab and no torch: h5py + numpy only, like the rest of ``transfer/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import h5py
import numpy as np

from generate_stairs_scene import scene_xml
from overlay_geom import (DECOR_CONTACT_ATTRS, DECOR_GEOM_GROUP,
                          OVERLAY_VARIANTS, is_decor,
                          overlay_sub_boxes_from_manifest)

#: An overlay whose x extent is at most this is a STEP rather than a platform or a box.
#: Trained treads top out at 0.28 m (``generate_stairs_scene.TRAINED_TREADS``); the platforms
#: they sit between are metres deep, so anything in between is neither and is not used to
#: place the spawn.
_STEP_MAX_TREAD_M = 0.60

#: Thickness of the emitted tile floor slab (m). A finite box, not the infinite plane: the
#: Isaac tile's floor spans its footprint and stops, and walking off the edge is a real
#: property of the tile. ``--infinite_floor`` restores the plane when that gets in the way.
_FLOOR_THICKNESS_M = 0.5


def load_tile(clip_path: str, terrain_id: int | None) -> tuple[dict, dict]:
    """Read one tile's manifest entry from a clip library.

    Args:
        clip_path: Path to the clip ``.h5``.
        terrain_id: Which tile to take. ``None`` takes the only tile when the library has
            one, and refuses otherwise rather than picking for you.

    Returns:
        ``(entry, info)`` -- the manifest entry, and provenance for the scene header.

    Raises:
        ValueError: When the library has no manifest, when ``terrain_id`` is absent from it,
            or when it is ``None`` and the library describes more than one tile.
    """
    with h5py.File(clip_path, "r") as f:
        raw = f.attrs.get("terrain_manifest")
        if raw is None:
            raise ValueError(f"{clip_path} carries no 'terrain_manifest'; it describes no "
                             "terrain, so there is no tile to build.")
        manifest = json.loads(raw)
        info = {
            "clip": os.path.basename(clip_path),
            "fingerprint": str(f.attrs.get("clip_fingerprint", "?")),
            "num_clips": int(f.attrs.get("num_clips", 0)),
        }
    by_id = {int(m["terrain_id"]): m for m in manifest}
    if terrain_id is None:
        if len(by_id) != 1:
            raise ValueError(
                f"{clip_path} describes {len(by_id)} tiles ({sorted(by_id)[:8]}...); pass "
                "--terrain_id to say which one. Picking silently would put the robot on a "
                "different staircase than the one you meant to bisect."
            )
        terrain_id = next(iter(by_id))
    if terrain_id not in by_id:
        raise ValueError(f"{clip_path} has no terrain_id {terrain_id}; it has "
                         f"{sorted(by_id)[:12]}{'...' if len(by_id) > 12 else ''}.")
    entry = by_id[terrain_id]
    info["terrain_id"] = terrain_id
    info["family"] = str(entry.get("family", "?"))
    info["type"] = str(entry.get("type", "?"))
    return entry, info


#: Overlay ``kind`` values, mirroring ``terrains/blocks/overlay.py``'s ``OVERLAY_KINDS``.
#: Kept as a literal rather than imported because ``blocks/overlay.py`` pulls in IsaacLab and
#: this file runs under ``env_transfer``; ``test_clip_tile_scene.py`` pins the two together.
OVERLAY_KINDS = ("box", "obstacle", "overhead")


def emittable_overlays(entry: dict) -> list[dict]:
    """The tile's overlay volumes, refusing any this emitter cannot represent exactly.

    Neither ``kind`` nor ``variant`` restricts what is emitted any more:

    * every ``kind`` is one axis-aligned box, and the distinction (walkable ground vs. a
      hurdle vs. an overhead bar) is carried by ``walkable_top`` and ``base_z``, which
      :func:`walkable` and :func:`tile_geoms` honour;
    * every ``variant`` is a fixed set of axis-aligned sub-boxes, which
      :func:`overlay_geom.overlay_sub_boxes` produces -- the same decomposition the Isaac
      importer meshes.  This used to be refused, which meant the 23 shipped tiles carrying a
      ``floating`` overlay could not be emitted at all.

    Both are still validated, because an unrecognised value is far more likely to be a new
    shape this emitter would silently flatten than a typo.

    Args:
        entry: A manifest tile entry.

    Returns:
        The overlay dicts, in manifest order.

    Raises:
        ValueError: On an unknown kind or an unknown variant.
    """
    out = []
    for i, ov in enumerate(entry.get("overlays", [])):
        kind, variant = str(ov.get("kind", "box")), str(ov.get("variant", "solid"))
        if kind not in OVERLAY_KINDS:
            raise ValueError(
                f"tile {entry.get('terrain_id')} overlay {i} has kind='{kind}'; expected one of "
                f"{list(OVERLAY_KINDS)}. An unknown kind is most likely a new shape, which this "
                "emitter would silently flatten into a box."
            )
        if variant not in OVERLAY_VARIANTS:
            raise ValueError(
                f"tile {entry.get('terrain_id')} overlay {i} has variant='{variant}'; expected "
                f"one of {list(OVERLAY_VARIANTS)}. Emitting its bounding box instead would look "
                "right in the viewer and be wrong under the foot, which is the failure mode this "
                "whole file exists to avoid."
            )
        out.append(ov)
    return out


def walkable(overlays: list[dict]) -> list[dict]:
    """The overlays that are ground -- i.e. the ones a foot can be supported by.

    ``walkable_top`` is advisory metadata that, until now, nothing in this file acted on:
    every geometry query below treated a railing post or a wall as terrain.  Every overlay
    any generator has authored to date is ``walkable_top=True``, so this is a no-op on the
    shipped libraries and a guard on decorated ones.

    Args:
        overlays: Overlays this emitter can represent (see :func:`emittable_overlays`).

    Returns:
        The subset whose top face is a landing surface.
    """
    return [ov for ov in overlays if bool(ov.get("walkable_top", True))]


def terrain_top_at(overlays: list[dict], x: float, y: float) -> float:
    """Height of the WALKABLE terrain at a point in tile-local coordinates.

    Args:
        overlays: Overlays this emitter can represent (see :func:`emittable_overlays`).
        x: Tile-local x (m).
        y: Tile-local y (m).

    Returns:
        The top z (m) of the highest walkable overlay covering the point, or 0.0 (the tile
        floor).  Non-walkable volumes are ignored: standing the robot on top of a 1.1 m
        railing post is not a spawn height.

    Raises:
        NotImplementedError: If a covering overlay is yawed, where "covers this point" is not
            an axis-aligned test.
    """
    top = 0.0
    for ov in walkable(overlays):
        if abs(float(ov.get("yaw", 0.0))) > 1e-9:
            raise NotImplementedError(
                "terrain_top_at does not handle a yawed overlay; pass --spawn_xy explicitly."
            )
        (px, py), (sx, sy, sz) = ov["pos_xy"], ov["size_xyz"]
        if abs(x - px) <= sx / 2.0 and abs(y - py) <= sy / 2.0:
            top = max(top, float(ov.get("base_z", 0.0)) + float(sz))
    return top


def first_step(overlays: list[dict]) -> dict | None:
    """The lowest-x step-like overlay -- the first riser of the flight -- or ``None``.

    "Step-like" is WALKABLE and at most :data:`_STEP_MAX_TREAD_M` deep.  The walkable half
    matters: a railing post or a clutter box is also narrow in x, and picking one as the
    flight's first riser sends :func:`spawn_point` to that post's centreline -- measured at
    2.05 m outside the tile, over the void, on a probe tile.

    Args:
        overlays: Overlays this emitter can represent.

    Returns:
        The overlay dict, or ``None`` when the tile has no tread-sized walkable overlay (a
        box tile, a bare platform).
    """
    steps = [ov for ov in walkable(overlays) if float(ov["size_xyz"][0]) <= _STEP_MAX_TREAD_M]
    return min(steps, key=lambda ov: float(ov["pos_xy"][0]) - float(ov["size_xyz"][0]) / 2.0) \
        if steps else None


def spawn_point(entry: dict, overlays: list[dict], approach: float) -> tuple[float, float, str]:
    """Where the robot should stand, in tile-local metres.

    On the approach, ``approach`` metres before the first riser and laterally centred on the
    flight -- so "walk straight ahead" is the experiment, with no steering to confound it.
    The flight is often OFF the tile centreline (0.36 m on the shipped stairs_up tile), which
    is exactly the offset a tile-centred spawn would silently introduce.

    Args:
        entry: The manifest tile entry.
        overlays: Overlays this emitter can represent.
        approach: Distance (m) from the spawn to the first riser's front face.

    Returns:
        ``(x, y, why)`` -- the point and a one-line explanation for the log.

    Raises:
        ValueError: If the derived point falls outside the tile footprint. The spawn is
            derived from one overlay's geometry, so a misidentified "first riser" puts it
            somewhere the tile does not exist -- silently, since the emitter happily writes
            a scene whose robot starts over the void. Cheap to check, and it is the
            post-condition that pins :func:`first_step`'s predicate.
    """
    step = first_step(overlays)
    if step is None:
        return (float(entry["size_x"]) / 2.0, float(entry["size_y"]) / 2.0,
                "tile centre (no step-like overlay to aim at)")
    front = float(step["pos_xy"][0]) - float(step["size_xyz"][0]) / 2.0
    x, y = front - approach, float(step["pos_xy"][1])
    size_x, size_y = float(entry["size_x"]), float(entry["size_y"])
    if not (0.0 <= x <= size_x and 0.0 <= y <= size_y):
        raise ValueError(
            f"tile {entry.get('terrain_id')}: derived spawn ({x:.3f}, {y:.3f}) is outside the "
            f"tile footprint (0..{size_x:.3f}, 0..{size_y:.3f}). The first riser was taken to be "
            f"the overlay at {tuple(step['pos_xy'])} with size {tuple(step['size_xyz'])}; if that "
            "is not a stair tread, first_step picked the wrong overlay. Pass --spawn_xy to "
            "override, or reduce --approach."
        )
    return (x, y,
            f"{approach:.2f} m before the first riser (tile-local x={front:.3f}), "
            f"on the flight's centreline (y={y:.3f})")


def _box_geom(name: str, cx: float, cy: float, cz: float, sx: float, sy: float, sz: float,
              yaw: float, material: str = "box_material", decor: bool = False) -> str:
    """One MuJoCo box geom line.

    Args:
        name: Geom name.
        cx, cy, cz: Centre.
        sx, sy, sz: Full extents.
        yaw: Rotation about z (rad).
        material: Material name.
        decor: Emit it as VISUAL-ONLY -- group 1 rather than 2, and non-colliding. The depth
            cameras cast groups {1, 2} and the height scan casts {2}, so this is what makes a
            wall visible to the cameras, invisible to the scan, and impossible to walk into.

    Returns:
        The geom line.
    """
    quat = f' quat="{math.cos(yaw / 2.0):.9f} 0 0 {math.sin(yaw / 2.0):.9f}"' if yaw else ""
    group = DECOR_GEOM_GROUP if decor else 2
    flags = f" {DECOR_CONTACT_ATTRS}" if decor else ""
    return (f'        <geom name="{name}" type="box" material="{material}" group="{group}"'
            f'{flags} size="{sx / 2.0:.6f} {sy / 2.0:.6f} {sz / 2.0:.6f}" '
            f'pos="{cx:.6f} {cy:.6f} {cz:.6f}"{quat}/>')


def tile_geoms(entry: dict, overlays: list[dict], origin: tuple[float, float, float],
               infinite_floor: bool) -> tuple[list[str], str]:
    """Emit the tile's geometry, translated so ``origin`` (tile-local) is the world origin.

    A ``solid`` overlay is a full-height pillar from its ``base_z``, which is how the
    training tiles are built (step ``n`` spans ``z in [0, n*rise]``) -- so the tread tops
    and risers are clean and a foot cannot catch in a seam.  A ``floating*`` overlay is a
    thin top slab with an open gap beneath it, plus optional support legs; those are emitted
    as the same sub-boxes :func:`overlay_geom.overlay_sub_boxes` gives the Isaac importer,
    so the two sims build the same structure rather than the same silhouette.

    Args:
        entry: The manifest tile entry.
        overlays: Overlays this emitter can represent (see :func:`emittable_overlays`).
        origin: ``(x, y, z)`` tile-local point to place at the world origin.
        infinite_floor: Emit the checkered infinite plane instead of the tile's own floor.

    Returns:
        ``(geom_lines, floor_line)``.
    """
    ox, oy, oz = origin
    lines = []
    for i, ov in enumerate(overlays):
        decor = is_decor(ov)
        step = float(ov["size_xyz"][0]) <= _STEP_MAX_TREAD_M
        base = f"tile_decor{i}" if decor else f"tile_{'step' if step else 'slab'}{i}"
        yaw = float(ov.get("yaw", 0.0))
        # A solid overlay decomposes to exactly one sub-box with the overlay's own extents,
        # so the emitted line -- and its name -- is unchanged from before this branch existed.
        for j, (cx, cy, cz, bx, by, bz) in enumerate(overlay_sub_boxes_from_manifest(ov)):
            lines.append(_box_geom(
                base if j == 0 else f"{base}_p{j}",
                cx - ox, cy - oy, cz - oz, bx, by, bz, yaw, decor=decor,
            ))

    if infinite_floor or not bool(entry.get("has_floor", True)):
        # has_floor=False is an isolated structure (a floating staircase): Isaac gives it NO
        # ground plane, so a robot that steps off falls to infinity. Keep the plane there so
        # a run ends on the floor rather than in the void.
        return lines, ('        <geom name="floor" size="0 0 0.05" type="plane" '
                       f'material="groundplane" group="2" pos="0 0 {-oz:.6f}"/>')
    size_x, size_y = float(entry["size_x"]), float(entry["size_y"])
    floor = _box_geom("tile_floor", size_x / 2.0 - ox, size_y / 2.0 - oy,
                      -_FLOOR_THICKNESS_M / 2.0 - oz, size_x, size_y, _FLOOR_THICKNESS_M,
                      0.0, material="groundplane")
    return lines, floor


def main() -> None:
    """Parse CLI arguments and write the generated single-tile scene XML."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Clip library .h5 carrying the terrain_manifest.")
    p.add_argument("--terrain_id", type=int, default=None,
                   help="Tile to build (default: the only tile, else required).")
    p.add_argument("--robot", default="g1_perceptive",
               help="Robot model to include; names the output.")
    p.add_argument("--scene", default=None,
                   help="Scene name; output is robots/g1/<robot>_<scene>.xml "
                        "(default: tile<terrain_id>_scene).")
    p.add_argument("--out", default=None, help="Explicit output path.")
    p.add_argument("--approach", type=float, default=2.5,
                   help="Metres of approach between the spawn and the first riser (default "
                        "2.5, matching generate_stairs_scene's --x_start so the two scenes "
                        "give the robot the same run-up).")
    p.add_argument("--spawn_xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="Spawn point in TILE-LOCAL metres, overriding --approach.")
    p.add_argument("--infinite_floor", action="store_true", default=False,
                   help="Use the checkered infinite plane instead of the tile's own finite "
                        "floor. The tile floor stops at the footprint, as in Isaac, so the "
                        "robot can walk off the edge; this trades that fidelity for runs "
                        "that cannot end that way.")
    args = p.parse_args()

    entry, info = load_tile(args.clip, args.terrain_id)
    overlays = emittable_overlays(entry)
    if args.spawn_xy is not None:
        sx, sy = args.spawn_xy
        why = "--spawn_xy"
    else:
        sx, sy, why = spawn_point(entry, overlays, args.approach)
    sz = terrain_top_at(overlays, sx, sy)
    geoms, floor = tile_geoms(entry, overlays, (sx, sy, sz), args.infinite_floor)

    scene = args.scene or f"tile{info['terrain_id']}_scene"
    header = (
        f"Generated by generate_clip_tile_scene.py: terrain_id {info['terrain_id']} "
        f"({info['type']}, {info['family']})\n"
        f"         copied geometry for geometry out of {info['clip']} "
        f"(fingerprint {info['fingerprint'][:16]}).\n"
        f"         This is ONE Isaac tile, not an in-distribution lookalike; see that "
        f"script's docstring for the difference.\n"
        f"         Tile-local ({sx:.3f}, {sy:.3f}, {sz:.3f}) is the world origin: "
        f"{why}.\n"
        f"         Footprint {float(entry['size_x']):.2f} x {float(entry['size_y']):.2f} m, "
        f"{len(overlays)} overlay(s), has_floor={bool(entry.get('has_floor', True))}."
    )
    xml = scene_xml(f"{args.robot}.xml", header, "\n".join(geoms), floor,
                    model=f"g1_tile{info['terrain_id']}_scene")

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "robots", "g1", f"{args.robot}_{scene}.xml")
    with open(out_path, "w") as f:
        f.write(xml)

    print(f"[INFO] Wrote {out_path}")
    print(f"[INFO] tile {info['terrain_id']} ({info['type']}, {info['family']}) from "
          f"{info['clip']}: {float(entry['size_x']):.2f} x {float(entry['size_y']):.2f} m, "
          f"{len(overlays)} overlay(s)")
    print(f"[INFO] spawn: tile-local ({sx:.3f}, {sy:.3f}) at terrain z={sz:.3f} m -> world "
          f"origin ({why})")
    # Decoration is VISUAL ONLY -- emitted non-colliding (contype=0 conaffinity=0), so the
    # robot cannot stand on it or walk into it. It must therefore stay out of the walkable
    # geometry summary below: a 2.6 m room wall read as a riser reported "climb 2.508 m over
    # 1 riser" on a tile whose real climb is 3 risers of 0.13 m, which is a made-up number
    # pointing at a nonexistent obstacle.
    walkable = [ov for ov in overlays if not ov.get("decor")]
    n_decor = len(overlays) - len(walkable)
    for i, ov in enumerate(overlays):
        (px, py), (osx, osy, osz) = ov["pos_xy"], ov["size_xyz"]
        kind = ("decor" if ov.get("decor")
                else "step " if float(osx) <= _STEP_MAX_TREAD_M else "slab ")
        print(f"       - {kind}{i}: {osx:.3f} x {osy:.3f} x {osz:.3f} m, top "
              f"z={float(ov.get('base_z', 0.0)) + osz - sz:+.3f} m at world "
              f"(x={px - sx:+.3f}, y={py - sy:+.3f})")
    # The first RISER is the first overlay whose top is above the spawn, which is not
    # necessarily the first step-like one: on the shipped stairs_up tile the leading tread is
    # flush with the approach platform, so the tile climbs in 2 risers of 0.13 m and not the
    # 3 that counting step overlays suggests.
    ups = sorted(
        ((float(ov["pos_xy"][0]) - float(ov["size_xyz"][0]) / 2.0 - sx,
          float(ov.get("base_z", 0.0)) + float(ov["size_xyz"][2]) - sz) for ov in walkable),
        key=lambda t: t[0],
    )
    risers, last = [], 0.0
    for front, top in ups:
        if top > last + 1e-6:
            risers.append((front, top - last))
            last = top
    if n_decor:
        print(f"[INFO] {n_decor} visual-only decor overlay(s) excluded from the climb summary "
              "(non-colliding: the robot can neither stand on them nor hit them)")
    if risers:
        print("[INFO] climb from the spawn, drive +x: "
              + ", ".join(f"+{d:.3f} m at x={x:+.3f}" for x, d in risers)
              + f" (total {last:.3f} m over {len(risers)} riser(s))")


if __name__ == "__main__":
    main()
