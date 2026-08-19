# Copyright 2026 Zachary Olkin. All rights reserved.

"""An overlay field no consumer implements must be REFUSED, not ignored.

``decor`` is Part 2.4's visual-only channel: a wall the depth cameras see, the height scan
does not, and the robot cannot walk into. Each of those three is a different mechanism per
simulator -- a separate prim in IsaacLab, because the height scanner is a ``RayCaster`` that
takes exactly ONE mesh prim while the cameras take a list; geom group 1 with ``contype=0`` in
MuJoCo.

A consumer that merely ignores the field builds the decoration as a SOLID, COLLIDABLE box
that the height scan reads as terrain. That is worse than not having the feature: instead of
a wall the robot can see and not hit, you get a wall it can hit and the scan mistakes for
ground. So the field is refused on both sides until both sides implement it, which also makes
it impossible for the emitter to land before its consumers.

Same move Part 0 made for overlay ``yaw``, and for the same reason.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "transfer" / "sim"))


def _overlay(**extra) -> dict:
    """A valid overlay, plus whatever the caller adds."""
    return {"kind": "box", "pos_xy": [1.0, 2.0], "size_xyz": [1.0, 1.0, 0.2],
            "base_z": 0.0, "walkable_top": True, "yaw": 0.0, **extra}


def test_the_mujoco_path_refuses_a_decor_overlay():
    """``transfer/`` is the vendored copy and must refuse independently of IsaacLab."""
    from overlay_geom import overlay_sub_boxes_from_manifest
    with pytest.raises(ValueError, match="does not implement"):
        overlay_sub_boxes_from_manifest(_overlay(decor=True))


def test_the_mujoco_path_still_accepts_an_ordinary_overlay():
    """The refusal must be keyed on the field, not on anything incidental."""
    from overlay_geom import overlay_sub_boxes_from_manifest
    boxes = overlay_sub_boxes_from_manifest(_overlay())
    assert len(boxes) == 1


def test_the_isaac_loader_refuses_a_decor_overlay():
    """And so must the IsaacLab side, which builds the ground mesh."""
    pytest.importorskip("isaaclab", reason="the loader imports IsaacLab")
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
        _overlays_from_manifest)
    entry = {"terrain_id": 3, "size_x": 10.0, "size_y": 4.0, "overlays": [_overlay(decor=True)]}
    with pytest.raises(ValueError, match="does not implement"):
        _overlays_from_manifest(entry)


def test_the_isaac_loader_still_accepts_an_ordinary_overlay():
    """The undecorated library must keep loading unchanged."""
    pytest.importorskip("isaaclab", reason="the loader imports IsaacLab")
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
        _overlays_from_manifest)
    entry = {"terrain_id": 3, "size_x": 10.0, "size_y": 4.0, "overlays": [_overlay()]}
    assert len(_overlays_from_manifest(entry)) == 1


def test_both_sides_name_the_same_field():
    """One reserved list, so the two refusals cannot drift to different field names."""
    from overlay_geom import RESERVED_OVERLAY_FIELDS
    assert "decor" in RESERVED_OVERLAY_FIELDS
