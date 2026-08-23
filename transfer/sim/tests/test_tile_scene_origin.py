# Copyright 2026 Zachary Olkin. All rights reserved.

"""A tile scene must state which tile-local point it put at its world origin.

``generate_clip_tile_scene.py`` re-centres a tile on a computed spawn point -- 2.5 m before
the first riser, on the flight centreline -- while a clip's poses stay in the clip library's
tile-local frame. Nothing reconciled the two, so ``gen_tracker_runner --clip`` spawned the
robot at the clip's raw coordinates and missed by the whole offset: on tile 0 that is 8.6 m
downrange and 0.13 m up, putting the robot on the far platform instead of the approach. It
read as the tracker falling instantly, and the geometry was never in question.

It went unnoticed because ``--clip`` was built for parity libraries whose clips already sit at
the origin, and the tile scenes were previously driven by the FM, which anchors on the robot
rather than on absolute clip coordinates.

The offset was always in the scene -- in the provenance COMMENT. This pins that it is also
there as a ``<custom><numeric>`` a consumer can read off the compiled model, because the whole
point is that nobody has to type it: a hand-passed value that is wrong mis-places the robot
silently instead of raising.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

_ROBOTS = Path(__file__).resolve().parents[1] / "robots" / "g1"
_ROBOT = "g1_perceptive"
_TILES = ["tile0_scene", "tile20_scene", "tile26_scene"]


def _origin_from_model(path: Path):
    """Read ``tile_local_origin`` off the COMPILED model, as the runner does."""
    m = mujoco.MjModel.from_xml_path(str(path))
    for i in range(int(m.nnumeric)):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_NUMERIC, i) == "tile_local_origin":
            adr = int(m.numeric_adr[i])
            return [float(v) for v in m.numeric_data[adr:adr + 3]]
    return None


@pytest.mark.parametrize("scene", _TILES)
def test_the_tile_scene_declares_its_origin(scene):
    got = _origin_from_model(_ROBOTS / f"{_ROBOT}_{scene}.xml")
    assert got is not None, (
        f"{scene} carries no tile_local_origin numeric, so clip playback cannot place a clip "
        "into it and would spawn the robot by the whole tile offset"
    )
    assert len(got) == 3


@pytest.mark.parametrize("scene", _TILES)
def test_the_numeric_agrees_with_the_provenance_comment(scene):
    """Two statements of one number drift; this makes them fail together instead."""
    path = _ROBOTS / f"{_ROBOT}_{scene}.xml"
    said = re.search(r"Tile-local \(([-\d.]+), ([-\d.]+), ([-\d.]+)\) is the world origin",
                     path.read_text(encoding="utf-8"))
    assert said, f"{scene} lost the provenance line naming its origin"
    from_comment = [float(said.group(i)) for i in (1, 2, 3)]
    got = _origin_from_model(path)
    for a, b in zip(from_comment, got):
        assert abs(a - b) < 1e-3, (
            f"{scene}: comment says {from_comment}, numeric says {got} -- one of them is "
            "what a reader trusts and the other is what the runner uses"
        )


def test_the_tiles_do_not_share_an_origin():
    """A copy-paste that gave every tile the same offset would pass every check above."""
    origins = [tuple(_origin_from_model(_ROBOTS / f"{_ROBOT}_{s}.xml")) for s in _TILES]
    assert len(set(origins)) == len(origins), f"tile scenes share an origin: {origins}"


def test_a_flat_scene_has_no_origin_and_that_is_fine():
    """basic_scene is not a tile, so a clip is used in its own frame, unshifted."""
    assert _origin_from_model(_ROBOTS / f"{_ROBOT}_basic_scene.xml") is None


def test_the_runner_reads_it_rather_than_taking_a_flag():
    src = (Path(__file__).resolve().parents[1] / "gen_tracker_runner.py").read_text(
        encoding="utf-8")
    assert "_scene_tile_local_origin" in src
    assert "clip.set_placement(-tile_origin" in src, (
        "the runner must APPLY the origin: ClipReference._lift_pos is anchor_pos + p_rel, so "
        "an anchor of minus the origin puts every clip pose in scene coordinates"
    )
