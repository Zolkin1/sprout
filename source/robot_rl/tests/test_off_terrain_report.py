# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pure (no-library) tests for the off-terrain reference-foot check.

``report_off_terrain`` flags a clip whose reference foot drifts off the walkable
footprint: the (narrow, centered) tread while over a stair band, or the full
corridor over a flat band.  These build a synthetic 2-foot batch over a
``[flat][stairs][flat]`` corridor and check the lateral logic directly.
"""

from __future__ import annotations

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
    report_off_terrain,
)

# Corridor: flat [0,3] -> stairs [3,6] (tread center y=0.75, width 0.8 -> y in
# [0.35, 1.15]) -> flat [6,10]; corridor width 1.5 -> y in [0, 1.5].
_MANIFEST = {
    "size_x": 10.0,
    "size_y": 1.5,
    "stair_width": 0.8,
    "bands": [
        {"block_type": "FlatBlock", "aabb": [0.0, 3.0]},
        {"block_type": "StairBlock", "aabb": [3.0, 6.0],
         "stair_top_centers": [[3.5, 0.75, 0.1], [4.5, 0.75, 0.2], [5.5, 0.75, 0.3]]},
        {"block_type": "FlatBlock", "aabb": [6.0, 10.0]},
    ],
}

_NAMES = [
    "left_ankle_roll_link:pos_x", "left_ankle_roll_link:pos_y",
    "right_ankle_roll_link:pos_x", "right_ankle_roll_link:pos_y",
]


def _batch(feet_xy: list[list[tuple[float, float]]]) -> dict:
    """Build a batch dict from ``[clip][step] -> (x, y)`` applied to both feet."""
    n = len(feet_xy)
    T = len(feet_xy[0])
    y = torch.zeros(n, T, len(_NAMES))
    for ci, steps in enumerate(feet_xy):
        for t, (x, yy) in enumerate(steps):
            y[ci, t] = torch.tensor([x, yy, x, yy])
    return {"y_des": y, "meta": {"pos_output_names": _NAMES}}


def test_centered_walk_is_on_terrain():
    """A foot on the centerline stays inside both the tread and the corridor."""
    b = _batch([[(x, 0.75) for x in (0.5, 2.0, 4.5, 7.0, 9.0)]])
    off, worst = report_off_terrain(b, _MANIFEST)
    assert not bool(off.any())
    assert float(worst.max()) < 1e-6


def test_off_tread_over_stairs_is_flagged():
    """y=1.3 over the stair band (x=4.5) is 0.15 m past the tread edge (1.15)."""
    b = _batch([[(4.5, 1.30)]])
    off, worst = report_off_terrain(b, _MANIFEST)
    assert bool(off[0])
    assert abs(float(worst[0]) - 0.15) < 1e-4


def test_same_y_over_flat_band_is_ok():
    """The same y=1.3 over a FLAT band (x=1.5) is inside the wider corridor."""
    b = _batch([[(1.5, 1.30)]])
    off, worst = report_off_terrain(b, _MANIFEST)
    assert not bool(off.any())
    assert float(worst.max()) < 1e-6


def test_off_corridor_side_over_flat_is_flagged():
    """y=1.7 over a flat band exceeds the corridor width (1.5) by 0.2 m."""
    b = _batch([[(7.0, 1.70)]])
    off, worst = report_off_terrain(b, _MANIFEST)
    assert bool(off[0])
    assert abs(float(worst[0]) - 0.20) < 1e-4


def test_mixed_batch_counts_only_offenders():
    """Only clips with an off-terrain step are flagged; others stay clean."""
    b = _batch([
        [(4.5, 0.75)],   # centered on stairs -> ok
        [(4.5, 1.30)],   # off the tread -> flagged
        [(1.5, 0.20)],   # near a flat edge but inside -> ok
    ])
    off, worst = report_off_terrain(b, _MANIFEST)
    assert off.tolist() == [False, True, False]
