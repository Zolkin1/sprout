# Copyright 2026 Zachary Olkin. All rights reserved.

"""``class_footprints`` must equal the floor the loader actually builds.

The decoration pass resolves wall positions OFFLINE, into the manifest, and it lives in
``motion_gen`` which cannot import IsaacLab. So it recomputes the class footprint that
``clip_terrain_loader._class_fp`` produces. Two copies of one number.

The duplication is safe only because of this test. Without it the failure is silent and
geometric: a wall lands metres inside the visible ground and simply looks like a slab
standing in a field. That is exactly how the walls first shipped -- placed at the tile's own
``size_y``, which is 4.9 to 7.1 m short of the floor on transition and box tiles.

Built against a synthetic manifest rather than the real library, so it runs without the
6.8 GB file and covers the case that actually varies: tiles of very different sizes sharing
one class.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab", reason="the loader is the other half of this")

from motion_gen.terrain.decorate import (DEFAULT_SIZE_MARGIN,  # noqa: E402
                                         DEFAULT_TRANSITION_BACK_BUFFER,
                                         class_footprints)


def _tile(terrain_id: int, kind: str, size_x: float, size_y: float) -> dict:
    """One manifest entry with a single walkable overlay."""
    return {
        "terrain_id": terrain_id, "type": kind, "size_x": size_x, "size_y": size_y,
        "has_floor": True, "frame_yaw": 0.0, "family": f"family_{kind}.h5",
        "overlays": [{"kind": "box", "pos_xy": [size_x / 2, size_y / 2],
                      "size_xyz": [1.0, 1.0, 0.2], "base_z": 0.0,
                      "walkable_top": True, "yaw": 0.0}],
    }


MANIFEST = [
    _tile(0, "terrain", 11.0, 5.2),
    _tile(1, "terrain", 9.5, 4.1),
    _tile(2, "transition", 22.9, 4.5),
    _tile(3, "transition", 18.5, 8.4),
    _tile(4, "box", 19.4, 4.7),
    _tile(5, "box", 13.2, 9.7),
]


def test_the_offline_footprint_matches_the_loaders():
    """The whole reason the duplication is allowed."""
    from robot_rl.tasks.manager_based.robot_rl.terrains import clip_terrain_loader as L

    ours = class_footprints(MANIFEST)
    terr = [m for m in MANIFEST if m["type"] == "terrain"]
    trans = [m for m in MANIFEST if m["type"] == "transition"]
    box = [m for m in MANIFEST if m["type"] == "box"]

    # Reproduce the loader's own arithmetic from its source of truth.
    def theirs(ms):
        return (max(float(m["size_x"]) for m in ms) + DEFAULT_SIZE_MARGIN,
                max(float(m["size_y"]) for m in ms) + DEFAULT_SIZE_MARGIN)

    assert ours["terrain"] == pytest.approx(theirs(terr))
    assert ours["box"] == pytest.approx(theirs(box))
    tx, ty = theirs(trans)
    assert ours["transition"] == pytest.approx((tx + DEFAULT_TRANSITION_BACK_BUFFER, ty))
    # and the defaults are the loader's, not something that drifted
    import inspect
    sig = inspect.signature(L.assemble_clip_terrain)
    assert sig.parameters["size_margin"].default == DEFAULT_SIZE_MARGIN
    assert (sig.parameters["transition_back_buffer"].default
            == DEFAULT_TRANSITION_BACK_BUFFER)


def test_the_footprint_is_a_CLASS_property_not_a_tile_one():
    """The bug in one line: every tile of a class shares one floor, sized by the LARGEST.

    A 13.2 x 9.7 box tile and a 19.4 x 4.7 one both render on 20.4 x 10.7 m, so the second
    is 6 m short in y of the ground it stands on.
    """
    fps = class_footprints(MANIFEST)
    assert fps["box"] == pytest.approx((19.4 + 1.0, 9.7 + 1.0))
    for m in (t for t in MANIFEST if t["type"] == "box"):
        assert fps["box"][0] >= m["size_x"] and fps["box"][1] >= m["size_y"]
    small = next(t for t in MANIFEST if t["terrain_id"] == 4)
    assert fps["box"][1] - small["size_y"] > 5.0, (
        "this is the gap walls used to be placed inside"
    )


def test_walls_reach_the_floor_edge_when_given_the_footprint():
    """End to end: with the footprint, a wall's outer face IS the floor boundary."""
    from motion_gen.terrain.decorate import DecorateConfig, add_walls, span
    from motion_gen.terrain.footprint import TileFootprint

    fps = class_footprints(MANIFEST)
    tile = _tile(4, "box", 19.4, 4.7)
    fp = TileFootprint(terrain_id=4, n_clips=1, y_min=2.0, y_max=2.7, x_min=6.0,
                       x_max=13.0, y_min_sized=1.7, y_max_sized=3.0)
    add_walls(tile, fp, DecorateConfig(seed=0, wall_frac=1.0, wall_window_frac=0.0),
              fps["box"])
    walls = [o for o in tile["overlays"] if o.get("decor")]
    assert walls, "no walls were built"
    fx, fy = fps["box"]
    for w in walls:
        lo_y, hi_y = span(w)
        lo_x = w["pos_xy"][0] - w["size_xyz"][0] / 2.0
        hi_x = w["pos_xy"][0] + w["size_xyz"][0] / 2.0
        touches = (abs(lo_y) < 1e-6 or abs(hi_y - fy) < 1e-6
                   or abs(lo_x) < 1e-6 or abs(hi_x - fx) < 1e-6)
        assert touches, (
            f"wall at y[{lo_y:.3f},{hi_y:.3f}] x[{lo_x:.3f},{hi_x:.3f}] does not touch the "
            f"{fx:.3f} x {fy:.3f} floor"
        )
