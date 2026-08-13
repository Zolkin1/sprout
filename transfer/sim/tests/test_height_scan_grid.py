# Copyright 2026 Zachary Olkin. All rights reserved.

"""The MuJoCo height-scan cast at the GENERATOR's dense grid.

``Robot.compute_height_scan`` was written for, and empirically verified against, the
policy's 16x11 grid.  The generator's is 121x121 over 3x3 m, and squareness costs the one
check that grid had for free: **a transposed fold cannot be caught by shape.**  It is the
same trap ``depth_obs`` guards for cameras (30x26 vs 26x30), except no width check can
notice it here.

So the dense cast is pinned against the coarse one, which is the cast already verified
against IsaacLab.  Their grids share points -- 0.1 m is a multiple of 0.025 m -- and the
coarse grid is asymmetric (16x11, centred 0.75 m forward), so agreeing on the shared points
fixes the dense grid's orientation, spacing and mount offset against a known-good reference.
:func:`test_the_overlap_check_can_fail` is the power check: the transposed grid must NOT
pass, or the comparison above proves nothing.

Poses are drawn away from the axis-aligned degenerate case on purpose.  At yaw=0 on a
staircase the two grids' shared x land on a step EDGE to within 1e-14 m, and the two casts
then differ by exactly one riser -- a real property of ray-casting a discontinuity, not a
defect, but it makes an exact-equality test a coin flip.

Needs the MuJoCo scene assets, so run from ``transfer/sim``: ``pytest tests/``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_SIM, os.path.dirname(_SIM)):     # transfer/sim, transfer/
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mujoco  # noqa: E402

from robot import Robot  # noqa: E402

#: The generator's grid (``g1_fm_scan_env_cfg._FM_SCAN_*``), robot-centred.
DENSE = {"resolution": 0.025, "size": (3.0, 3.0), "offset_fwd": 0.0}
#: The policy's grid (``G1TeacherSceneCfg.height_scanner``), 0.75 m forward.
COARSE = {"resolution": 0.1, "size": (1.5, 1.0), "offset_fwd": 0.75}


def _axes(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis-frame ``(x, y)`` of each grid column/row, matching ``compute_height_scan``."""
    hx, hy = grid["size"][0] / 2.0, grid["size"][1] / 2.0
    xs = np.arange(-hx, hx + 1e-9, grid["resolution"]) + grid["offset_fwd"]
    ys = np.arange(-hy, hy + 1e-9, grid["resolution"])
    return xs, ys


@pytest.fixture(scope="module")
def robot() -> Robot:
    """The 29-DoF robot in the sim2sim terrain scene (stairs, so the scan varies).

    ``Robot`` resolves ``robots/g1/<robot>_<scene>.xml`` relative to the WORKING DIRECTORY,
    so the chdir is what lets this file run from the repo root as well as from
    ``transfer/sim``. Restored afterwards -- a module-scoped fixture that leaves the process
    somewhere else would break whatever runs next in the same session.
    """
    cwd = os.getcwd()
    os.chdir(_SIM)
    try:
        r = Robot(robot_name="g1_29j", scene_name="terrain_scene",
                  joystick_scaling=np.ones(3), input_function=lambda t: np.zeros(3))
    finally:
        os.chdir(cwd)
    mujoco.mj_forward(r.mj_model, r.mj_data)
    return r


def _cast(robot: Robot, grid: dict, pos: np.ndarray, yaw: float) -> np.ndarray:
    """One cast, folded to ``(num_y, num_x)`` -- y outer, x inner ("xy" ordering)."""
    quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])   # xyzw
    clearance, _ = robot.compute_height_scan(
        pos, quat, resolution=grid["resolution"], size=grid["size"],
        offset_fwd=grid["offset_fwd"], sensor_offset=0.5,
    )
    xs, ys = _axes(grid)
    return clearance.reshape(ys.size, xs.size)


def _shared_points() -> tuple[list[int], list[int], list[int]]:
    """Indices of the grid points the dense and coarse casts have in common."""
    dx, dy = _axes(DENSE)
    cx, cy = _axes(COARSE)
    keep = [k for k, v in enumerate(cx) if abs(v) <= DENSE["size"][0] / 2.0 + 1e-9]
    ix = [int(np.argmin(np.abs(dx - cx[k]))) for k in keep]
    iy = [int(np.argmin(np.abs(dy - v))) for v in cy]
    return keep, ix, iy


#: Poses on and around the staircases, none axis-aligned (see the module docstring).
_POSES = [(1.35, 0.21, 0.37), (2.16, -0.44, -0.62), (2.83, 0.31, 0.94),
          (3.42, -0.12, -1.13), (0.77, 0.53, 0.28)]


def test_the_grids_coincide_exactly():
    """0.1 m is a multiple of 0.025 m, so the shared points must be the SAME points."""
    keep, ix, _ = _shared_points()
    dx, cx = _axes(DENSE)[0], _axes(COARSE)[0]
    assert keep, "the two grids share no columns; the comparison below would be vacuous"
    assert max(abs(dx[i] - cx[k]) for i, k in zip(ix, keep)) < 1e-12


@pytest.mark.parametrize("pos_yaw", _POSES)
def test_dense_cast_agrees_with_the_verified_coarse_one(robot, pos_yaw):
    """The dense grid samples the same ground, in the same place, the same way up."""
    bx, by, yaw = pos_yaw
    pos = np.array([bx, by, 0.8])
    keep, ix, iy = _shared_points()
    dense = _cast(robot, DENSE, pos, yaw)[np.ix_(iy, ix)]
    coarse = _cast(robot, COARSE, pos, yaw)[:, keep]
    both = np.isfinite(dense) & np.isfinite(coarse)
    assert both.sum() > 0.5 * coarse.size, "too few shared hits to conclude anything"
    assert np.abs(dense[both] - coarse[both]).max() < 1e-9


def test_the_overlap_check_can_fail(robot):
    """The power check: a TRANSPOSED dense grid must be rejected by the comparison above.

    Without this, ``test_dense_cast_agrees...`` would still pass on ground flat enough for
    the transpose to be indistinguishable, and would be pinning nothing.
    """
    keep, ix, iy = _shared_points()
    worst = 0.0
    for bx, by, yaw in _POSES:
        pos = np.array([bx, by, 0.8])
        flipped = _cast(robot, DENSE, pos, yaw).T[np.ix_(iy, ix)]
        coarse = _cast(robot, COARSE, pos, yaw)[:, keep]
        both = np.isfinite(flipped) & np.isfinite(coarse)
        worst = max(worst, float(np.abs(flipped[both] - coarse[both]).max()))
    assert worst > 0.05, (
        f"a transposed dense grid differs by only {worst:.4f} m on these poses, so the "
        "agreement test has no power here -- move the poses onto more varied terrain."
    )
