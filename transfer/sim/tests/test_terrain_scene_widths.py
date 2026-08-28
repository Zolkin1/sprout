# Copyright 2026 Zachary Olkin. All rights reserved.

"""The joystick playground must be made of staircases that exist.

It had drifted the other way: the lane straight ahead of spawn was 0.90 m (2.95 ft) and
``slow_tread`` 0.86 m (2.82 ft), both narrower than the narrowest real staircase measured
(4 ft), while the one realistic lane was named ``narrow``.  That is not a cosmetic problem --
hand-labelled probe trials put the generator at jump 3/3 on a 0.86 m flight and stairs 11/12
across 1.22-2.44 m, so driving the old centre lane was driving the one width where the
training library says a raised thing is more likely a box than a staircase.

These tests pin the invariant rather than the numbers, so widening a lane or adding one stays
easy and going back under 4 ft does not.
"""

from __future__ import annotations

import os
import sys

import mujoco
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_stairs_scene as gss  # noqa: E402

#: The narrowest staircase found in the field, in metres. 4 ft.
_MIN_REAL_WIDTH = gss.REAL_STAIR_WIDTHS["min"]

_SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "robots", "g1", "g1_perceptive_terrain_scene.xml")


def _lane_extents(path: str) -> dict[str, tuple[float, float]]:
    """Lateral ``(y_min, y_max)`` per terrain lane in a compiled scene.

    Selects by NAME against the lane table rather than by geom group: the robot's own bodies
    are in the model too, and a group filter alone let a backpack shell and two camera mounts
    into an earlier version of this check.
    """
    model = mujoco.MjModel.from_xml_path(path)
    known = {ln.name for ln in gss.DEFAULT_LANES}
    out: dict[str, list[float]] = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if not name:
            continue
        lane = name.rsplit("_", 1)[0]
        if lane not in known:
            continue
        y, hy = float(model.geom_pos[i][1]), float(model.geom_size[i][1])
        lo, hi = out.setdefault(lane, [y - hy, y + hy])
        out[lane] = [min(lo, y - hy), max(hi, y + hy)]
    return {k: (v[0], v[1]) for k, v in out.items()}


def test_no_default_lane_is_narrower_than_a_real_staircase():
    """Every lane in the table is at least 4 ft across."""
    for lane in gss.DEFAULT_LANES:
        assert lane.width >= _MIN_REAL_WIDTH - 1e-9, (
            f"lane {lane.name!r} is {lane.width:.2f} m ({lane.width / 0.3048:.2f} ft), under "
            f"the {_MIN_REAL_WIDTH} m narrowest real staircase measured")


def test_the_committed_scene_matches_the_lane_table():
    """The generated XML on disk must be the one this module describes.

    A self-describing scene whose header and geometry disagree with its generator is the
    stale-derived-artifact failure this repo keeps finding; the file is regenerable, so the
    check is free.
    """
    extents = _lane_extents(_SCENE)
    assert set(extents) == {ln.name for ln in gss.DEFAULT_LANES}, (
        f"scene lanes {sorted(extents)} != table {sorted(ln.name for ln in gss.DEFAULT_LANES)}"
        " -- rerun: python transfer/sim/generate_stairs_scene.py")
    for lane in gss.DEFAULT_LANES:
        lo, hi = extents[lane.name]
        assert hi - lo == pytest.approx(lane.width, abs=1e-4), (
            f"lane {lane.name!r} is {hi - lo:.3f} m in the XML but {lane.width:.3f} m in the "
            "table -- rerun generate_stairs_scene.py")


def test_the_committed_scene_has_no_lane_narrower_than_four_feet():
    """The same floor, asserted against the FILE -- the thing that actually gets driven."""
    for name, (lo, hi) in _lane_extents(_SCENE).items():
        w = hi - lo
        assert w >= _MIN_REAL_WIDTH - 1e-4, (
            f"lane {name!r} in the committed scene is {w:.2f} m ({w / 0.3048:.2f} ft)")


def test_lanes_do_not_overlap_laterally():
    """Two staircases sharing lateral space is one staircase, and an unreadable trial."""
    spans = sorted(_lane_extents(_SCENE).items(), key=lambda kv: kv[1][0])
    for (na, (_, hi)), (nb, (lo, _)) in zip(spans, spans[1:]):
        assert lo > hi, f"lanes {na!r} and {nb!r} overlap laterally: {hi:.2f} vs {lo:.2f}"


def test_the_box_lane_is_clear_of_every_staircase():
    """A jump box beside a staircase would confound every stairs-vs-box observation made here."""
    model = mujoco.MjModel.from_xml_path(_SCENE)
    boxes = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.startswith("box"):
            y, hy = float(model.geom_pos[i][1]), float(model.geom_size[i][1])
            boxes.append((y - hy, y + hy))
    assert boxes, "the playground should still carry jump boxes"
    for blo, bhi in boxes:
        for name, (lo, hi) in _lane_extents(_SCENE).items():
            assert bhi < lo or blo > hi, f"a box overlaps lane {name!r} laterally"
