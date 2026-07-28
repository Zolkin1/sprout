# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the Motion Bricks LIBRARY-scale batch layer (pass A keys, pass C tile re-datum).

These two pieces are what turn the per-clip MB terrain pipeline into a library:

* the **segment key** decides when two clips may share one generated MB segment -- too coarse and clips get
  the wrong motion, too fine and the bank never hits;
* the **tile re-datum** is what lets many clips share one ``terrain_id`` -- the RL importer meshes ONE
  geometry per tile, so every clip on it must agree on the tile's origin and footprint.

The drivers live in ``scripts/rsl_rl`` (they are CLI entry points, not package modules), so the path is
extended here rather than making the scripts importable as a package.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "rsl_rl"))

from motion_gen.motionbricks.terrain_plan import TerrainPlan  # noqa: E402

mb_plan_library = pytest.importorskip("mb_plan_library")
mb_assemble_library = pytest.importorskip("mb_assemble_library")


def _plan(*, skill="stairs_up", n=60, entry=5, exit_=50, approach_surf_z=0.15, exit_surf_z=0.90,
          approach_vx=1.0, exit_vx=1.0, entry_vx=0.62, exit_speed=0.64, approach_s=2.5, exit_s=2.5,
          z_shift=0.0) -> TerrainPlan:
    """A synthetic plan whose replay climbs steadily; ``z_shift`` lifts the whole thing rigidly."""
    q = np.zeros((n, 36), np.float32)
    q[:, 0] = np.linspace(0.0, 3.0, n)
    q[:, 1] = 0.1
    q[:, 2] = np.linspace(0.9, 1.6, n) + z_shift
    q[:, 3] = 1.0
    q[:, 7:36] = np.sin(np.arange(n)[:, None] * 0.2 + np.arange(29)[None, :] * 0.1)
    return TerrainPlan(
        skill=skill, dt=0.02, terrain_qpos=q, terrain_contact=np.ones((n, 2), np.uint8),
        entry_frame=entry, exit_frame=exit_,
        approach_surf_z=approach_surf_z + z_shift, exit_surf_z=exit_surf_z + z_shift,
        approach_vx=approach_vx, exit_vx=exit_vx, entry_vx=entry_vx, exit_speed=exit_speed,
        approach_mode="walk", exit_mode="walk", approach_s=approach_s, exit_s=exit_s,
        meta={"terrain_id": 0, "clip_type": "transition", "has_floor": True},
    )


# --------------------------------------------------------------------------- segment keys

def test_segment_keys_are_deterministic():
    keys = mb_plan_library._segment_keys(_plan(), "hard", "hard")
    assert keys == mb_plan_library._segment_keys(_plan(), "hard", "hard")


def test_segment_keys_ignore_tile_height():
    """A tile with more steps lifts the whole traversal; MB is fed surface-RELATIVE windows either way."""
    a = mb_plan_library._segment_keys(_plan(), "hard", "hard")
    b = mb_plan_library._segment_keys(_plan(z_shift=0.45), "hard", "hard")
    assert a == b


def test_segment_keys_split_on_what_mb_actually_sees():
    base = mb_plan_library._segment_keys(_plan(), "hard", "hard")
    # approach command / ramp target / duration -> different approach segment
    assert mb_plan_library._segment_keys(_plan(approach_vx=2.0), "hard", "hard")[0] != base[0]
    assert mb_plan_library._segment_keys(_plan(entry_vx=1.4), "hard", "hard")[0] != base[0]
    assert mb_plan_library._segment_keys(_plan(approach_s=3.0), "hard", "hard")[0] != base[0]
    # exit command / ramp start / duration -> different exit segment
    assert mb_plan_library._segment_keys(_plan(exit_vx=2.0), "hard", "hard")[1] != base[1]
    assert mb_plan_library._segment_keys(_plan(exit_speed=1.2), "hard", "hard")[1] != base[1]
    assert mb_plan_library._segment_keys(_plan(exit_s=3.0), "hard", "hard")[1] != base[1]
    # the pin mode changes the generation entirely
    assert mb_plan_library._segment_keys(_plan(), "soft", "hard")[0] != base[0]
    assert mb_plan_library._segment_keys(_plan(), "hard", "soft")[1] != base[1]


def test_segment_keys_split_on_the_approach_start():
    """A run-up from a stand and an arrival already at speed are different motions for one command."""
    stand = mb_plan_library._segment_keys(_plan(), "hard", "hard", "stand")
    steady = mb_plan_library._segment_keys(_plan(), "hard", "hard", "steady")
    assert stand[0] != steady[0]          # the approach differs ...
    assert stand[1] == steady[1]          # ... the exit does not depend on how the approach began
    assert stand == mb_plan_library._segment_keys(_plan(), "hard", "hard")   # 'stand' is the default


def test_soft_pin_key_ignores_the_replay():
    """With no pin the model never sees the terrain, so two different replays share one segment."""
    a = mb_plan_library._segment_keys(_plan(), "soft", "soft")
    b = mb_plan_library._segment_keys(_plan(n=80, entry=9, exit_=70), "soft", "soft")
    assert a == b
    # ... but a hard pin must distinguish them
    assert (mb_plan_library._segment_keys(_plan(), "hard", "hard")
            != mb_plan_library._segment_keys(_plan(n=80, entry=9, exit_=70), "hard", "hard"))


def test_quantize_snaps_durations_onto_a_grid():
    q = mb_plan_library._quantize
    assert q(2.27, 0.5) == 2.5 and q(1.6, 0.5) == 1.5
    assert q(2.27, 0.0) == 2.27                      # 0 disables


def test_skill_seed_is_process_stable():
    """Tile sampling is seeded with this; a salted hash() would reshuffle tiles between runs."""
    assert mb_plan_library._skill_seed("stairs_up") == mb_plan_library._skill_seed("stairs_up")
    assert mb_plan_library._skill_seed("stairs_up") != mb_plan_library._skill_seed("stairs_down")
    assert mb_plan_library._skill_seed("stairs_up") == 128208023    # pinned: must not drift


# --------------------------------------------------------------------------- tile re-datum

# A stand-in y_des layout: TWO frames (a root and a foot), each with x/y/z, so a re-datum that moves only
# the root instead of every position channel is visible.
_X_CH, _Y_CH = [0, 3], [1, 4]          # root x,y = 0,1 ; foot x,y = 3,4


def _rec(corner, size, n=10):
    """A stand-in clip record: only the fields the re-datum touches."""
    y = np.zeros((n, 6), np.float32)
    y[:, 0] = np.linspace(0.5, 1.5, n)          # root x
    y[:, 1] = 0.25                              # root y
    y[:, 3] = np.linspace(0.5, 1.5, n) + 0.1    # foot x (just behind the root)
    y[:, 4] = 0.30                              # foot y
    import torch
    return {"y": torch.as_tensor(y), "corner": corner, "size_x": size[0], "size_y": size[1],
            "manifest": {"terrain_id": 3, "type": "transition", "size_x": size[0], "size_y": size[1],
                         "overlays": [], "has_floor": True}}


def test_redatum_gives_every_clip_of_a_tile_one_origin_and_footprint():
    """The reuse invariant: one mesh per terrain_id, so identical size + overlays across the tile's clips."""
    margin = 1.0
    boxes = [{"pos": [2.0, 0.0, 0.1], "size": [1.0, 2.0, 0.2], "yaw": 0.0}]
    mb_assemble_library._SHARED.update(x_ch=_X_CH, y_ch=_Y_CH)
    # two clips datumed differently: clip B reached further -x/-y, so it got a lower corner
    recs = [_rec((-1.0, -1.0), (6.0, 4.0)), _rec((-1.5, -1.2), (6.5, 4.1))]
    y0_before = recs[0]["y"].clone()
    foot_gap_before = (recs[0]["y"][:, [0, 1]] - recs[0]["y"][:, [3, 4]]).clone()
    sx, sy = mb_assemble_library._redatum_tile(recs, boxes, margin, "stairs")

    assert recs[0]["manifest"]["size_x"] == recs[1]["manifest"]["size_x"] == sx
    assert recs[0]["manifest"]["size_y"] == recs[1]["manifest"]["size_y"] == sy
    assert recs[0]["manifest"]["overlays"] == recs[1]["manifest"]["overlays"]
    assert recs[0]["corner"] == recs[1]["corner"] == (-1.5, -1.2)      # the union (most-negative) datum
    # The tile footprint covers the UNION, which is larger than either clip's own footprint: clip B starts
    # further back (-0.2 in y) while clip A reaches further forward (2.0), so y spans 2.2 + 2*margin.
    assert sx == pytest.approx(6.5) and sy == pytest.approx(4.2)
    # the clip datumed further in shifts by exactly the datum difference -- on EVERY position channel
    assert recs[0]["y"][:, 0].numpy() == pytest.approx((y0_before[:, 0] + 0.5).numpy())
    assert recs[0]["y"][:, 3].numpy() == pytest.approx((y0_before[:, 3] + 0.5).numpy())   # foot x too
    assert recs[0]["y"][:, 1].numpy() == pytest.approx((y0_before[:, 1] + 0.2).numpy())
    assert recs[0]["y"][:, 4].numpy() == pytest.approx((y0_before[:, 4] + 0.2).numpy())   # foot y too
    # ... so the root-to-foot offset is UNCHANGED: a re-datum is a rigid move of the whole skeleton
    gap_after = recs[0]["y"][:, [0, 1]] - recs[0]["y"][:, [3, 4]]
    assert gap_after.numpy() == pytest.approx(foot_gap_before.numpy(), abs=1e-5)   # float32 noise only
    assert recs[1]["y"][:, 0].numpy() == pytest.approx(_rec((-1.5, -1.2), (6.5, 4.1))["y"][:, 0].numpy())


def test_redatum_is_a_noop_when_clips_already_share_a_datum():
    margin = 1.0
    boxes = [{"pos": [2.0, 0.0, 0.1], "size": [1.0, 2.0, 0.2], "yaw": 0.0}]
    mb_assemble_library._SHARED.update(x_ch=_X_CH, y_ch=_Y_CH)
    recs = [_rec((-1.0, -1.0), (6.0, 4.0)), _rec((-1.0, -1.0), (6.0, 4.0))]
    before = [r["y"].clone() for r in recs]
    sx, sy = mb_assemble_library._redatum_tile(recs, boxes, margin, "stairs")
    assert (sx, sy) == pytest.approx((6.0, 4.0))
    for r, b in zip(recs, before):
        assert r["y"].numpy() == pytest.approx(b.numpy())


def test_redatum_widens_stair_platforms_to_the_shared_footprint():
    """Wide platforms are stretched to the tile width so a drifting MB segment cannot step off them."""
    margin = 1.0
    boxes = [{"pos": [0.5, 0.0, 0.1], "size": [0.25, 1.0, 0.2], "yaw": 0.0},    # narrow tread
             {"pos": [3.0, 0.0, 0.1], "size": [2.5, 1.0, 0.2], "yaw": 0.0}]     # wide platform
    mb_assemble_library._SHARED.update(x_ch=_X_CH, y_ch=_Y_CH)
    recs = [_rec((-1.0, -1.0), (6.0, 4.0))]
    _sx, sy = mb_assemble_library._redatum_tile(recs, boxes, margin, "stairs")
    ov = recs[0]["manifest"]["overlays"]
    assert ov[0]["size_xyz"][1] == pytest.approx(1.0)      # tread keeps its authored width
    assert ov[1]["size_xyz"][1] == pytest.approx(sy)       # platform spans the tile
    assert ov[1]["pos_xy"][1] == pytest.approx(sy / 2.0)


def test_redatum_leaves_a_box_tiles_authored_widths_alone():
    """A box tile also has two overlays (jump box + top platform), but its widths are deliberate.

    Widening them to the tile footprint -- which includes the margin -- would lay a box-high slab over the
    whole tile, so the stair rule must key on the GEOMETRY, not on "more than one overlay".
    """
    margin = 1.0
    boxes = [{"pos": [0.5, 0.0, 0.235], "size": [0.8, 1.0, 0.47], "yaw": 0.0},   # the jump box
             {"pos": [1.65, 0.0, 0.235], "size": [1.5, 1.0, 0.47], "yaw": 0.0}]  # its top platform
    mb_assemble_library._SHARED.update(x_ch=_X_CH, y_ch=_Y_CH)
    recs = [_rec((-1.0, -1.0), (6.0, 4.0))]
    _sx, sy = mb_assemble_library._redatum_tile(recs, boxes, margin, "box")
    assert sy > 1.0                                        # the tile IS wider than the authored geometry
    assert [o["size_xyz"][1] for o in recs[0]["manifest"]["overlays"]] == pytest.approx([1.0, 1.0])
