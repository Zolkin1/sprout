# Copyright 2026 Zachary Olkin. All rights reserved.

"""Uniform-staircase tread layout -- the SINGLE SOURCE OF TRUTH (isaaclab-free).

Both consumers import this so their geometry cannot drift:

* the RL terrain block builder (``robot_rl`` ``terrains/trimesh/stair.py::_build_stair_steps``)
  wraps these tread-top centers with the actual box meshes + decoration; and
* the offline clip generator projects a descent motion onto these exact treads, so a
  generated clip lands on precisely the staircase the RL terrain meshes.

Pure ``numpy`` -- no ``trimesh``, no IsaacLab -- so it is importable from the isaaclab-free
clip-generation environment.

Conventions (must match ``_build_stair_steps`` verbatim):
    * Along +x, stair 0 is at the span's ``xmin``; ``going_up`` -> stair 0 is the lowest
      tread, else stair 0 is the highest (descending in +x).
    * Depth tiles the span: ``N = max(1, round(size_x / d_nominal))`` and ``d = size_x / N``
      (no partial tread), unless ``allow_partial_last_step`` -- then ``num_full`` treads of
      ``d_nominal`` plus a shorter trailing tread of the leftover.
    * Vertical anchoring: centred on ``z = 0`` by default; ``start_z_zero`` puts the lowest
      tread top at ``z = 0`` (whole staircase above ground).
    * All coordinates are shifted by ``(x_offset, y_offset)``; tread-center ``y`` is
      ``y_offset + size_y / 2`` (span centre-line).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LAYOUT_TOL = 1e-6


@dataclass
class StairLayout:
    """Resolved tread geometry for one uniform staircase span (span-local, shifted).

    Attributes:
        tread_top_centers: ``[N, 3]`` float32 ``(cx, cy, top_z)`` per tread, already
            shifted by ``(x_offset, y_offset, 0)``.
        step_depths: ``[N]`` float32 per-tread x-depth (the last entry is shorter than
            ``step_depth`` when a partial trailing tread is present).
        step_height: Uniform riser (m).
        step_depth: Full (nominal) tread depth (m) -- ``size_x / N`` in the non-partial path.
        num_steps: Number of treads ``N``.
        partial_step_length: Trailing partial-tread depth (m), or ``None`` when every tread
            has the same depth.
        z_lowest_top / z_highest_top: Top-z of the lowest / highest tread (pre any caller
            re-anchoring), needed for solid-box bottoms and side decoration.
    """

    tread_top_centers: np.ndarray
    step_depths: np.ndarray
    step_height: float
    step_depth: float
    num_steps: int
    partial_step_length: float | None
    z_lowest_top: float
    z_highest_top: float


def stair_tread_layout(
    *,
    size_x: float,
    size_y: float,
    step_height: float,
    step_depth_nominal: float,
    going_up: bool,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    start_z_zero: bool = False,
    allow_partial_last_step: bool = False,
) -> StairLayout:
    """Compute the tread-top centers + step dims of a uniform staircase span.

    See the module docstring for the (fixed) conventions.  Args mirror
    ``_build_stair_steps``; returns a :class:`StairLayout`.
    """
    step_height = float(step_height)
    step_depth_nominal = float(step_depth_nominal)
    size_x = float(size_x)

    # Step depth + count to fill the span (see convention #2).
    if allow_partial_last_step:
        step_depth = step_depth_nominal
        num_full = int(size_x // step_depth)
        leftover = size_x - num_full * step_depth
        has_partial = leftover > _LAYOUT_TOL
        num_steps = num_full + (1 if has_partial else 0)
        if num_steps == 0:
            num_steps = 1
            step_depth = size_x
            has_partial = False
            leftover = 0.0
    else:
        num_steps = max(1, int(round(size_x / step_depth_nominal)))
        step_depth = size_x / num_steps
        num_full = num_steps
        has_partial = False
        leftover = 0.0
    partial_step_length: float | None = float(leftover) if has_partial else None

    # Vertical anchoring (convention #3).
    if start_z_zero:
        z_lowest_top = 0.0
        z_highest_top = (num_steps - 1) * step_height
    else:
        half_range = (num_steps - 1) * step_height / 2.0
        z_lowest_top = -half_range
        z_highest_top = half_range
    y_center = y_offset + size_y / 2.0

    centers = np.zeros((num_steps, 3), dtype=np.float32)
    depths = np.zeros((num_steps,), dtype=np.float32)
    for i in range(num_steps):
        top_z = z_lowest_top + i * step_height if going_up else z_highest_top - i * step_height
        is_partial = has_partial and i == num_steps - 1
        this_depth = leftover if is_partial else step_depth
        cx_local = num_full * step_depth + this_depth / 2.0 if is_partial else (i + 0.5) * step_depth
        centers[i] = (x_offset + cx_local, y_center, top_z)
        depths[i] = this_depth

    return StairLayout(
        tread_top_centers=centers,
        step_depths=depths,
        step_height=step_height,
        step_depth=float(step_depth),
        num_steps=int(num_steps),
        partial_step_length=partial_step_length,
        z_lowest_top=float(z_lowest_top),
        z_highest_top=float(z_highest_top),
    )
