# Copyright 2026 Zachary Olkin. All rights reserved.

"""Isaac Lab-facing (scalar-LAST) adapter over :mod:`motion_gen.world_lift`.

**Every Isaac Lab consumer must import the lift from here, never from
``motion_gen.world_lift`` directly.**  ``tests/test_world_lift_convention.py``
enforces that.

Why this file exists
--------------------
The two sides of the clip pipeline use opposite quaternion component orders, and
:mod:`motion_gen.world_lift` sits on the seam:

* **Clip data / motion_gen** is scalar-FIRST.  Clip libraries store the
  orientation channels as ``{frame}:ori_w, ori_x, ori_y, ori_z`` and
  :mod:`motion_gen.math_utils` (``quat_mul``, ``quat_apply``, ``quat_inv``,
  ``yaw_quat``) is documented and implemented for ``(w, x, y, z)``.
* **Isaac Lab 3.0 is scalar-LAST.**  ``root_quat_w`` / ``body_quat_w`` return
  ``(x, y, z, w)`` and ``write_root_pose_to_sim_index`` expects ``(x, y, z, w)``
  (see its docstring in every backend).  Isaac Lab 2.x was scalar-first, so this
  flipped under us on the 3.0 upgrade.

The ``lift_*`` helpers in ``world_lift`` are built on the motion_gen primitives,
so they *behave* scalar-first regardless of what their signatures say.  Handing
them an Isaac Lab quaternion silently produces a quaternion with the scalar in
slot 0, which the sim then reads as ``x`` -- a ~180 deg roll about X.  That was
the "robot spawns upside down" bug: the reference was upright (a 10.6 deg tilt),
and the spawn came out at ``projected_gravity_b_z = +0.997``.  It also corrupted
the *position* lift, because a mis-ordered anchor makes ``anchor_inv_yaw``
discard the anchor yaw entirely.

Contract
--------
Every quaternion crossing this module's API is ``(x, y, z, w)``, matching Isaac
Lab -- including ``idx["ori"]``, the *index* selecting a clip's base-orientation
channels, which callers already build scalar-last.  All reordering happens here
and nowhere else, so Isaac Lab code stays uniformly scalar-last and clip files
stay uniformly scalar-first.

``ego_placement`` / ``ego_placement_masked`` / :func:`yaw_from_quat` are already
scalar-last in ``world_lift`` (they do the planar algebra by hand rather than
through the motion_gen primitives), so they pass straight through -- wrapped
only so that this module is the single import site.
"""

from __future__ import annotations

import torch
from motion_gen.world_lift import _yaw_from_quat as _yaw_from_quat_xyzw
from motion_gen.world_lift import (
    anchor_inv_yaw as _anchor_inv_yaw_wxyz,
)
from motion_gen.world_lift import (
    ego_placement as _ego_placement_xyzw,
)
from motion_gen.world_lift import (
    ego_placement_masked as _ego_placement_masked_xyzw,
)
from motion_gen.world_lift import (
    lift_base_world_state as _lift_base_world_state_wxyz,
)
from motion_gen.world_lift import (
    lift_orientation as _lift_orientation_wxyz,
)
from motion_gen.world_lift import (
    lift_orientation_batched as _lift_orientation_batched_wxyz,
)
from motion_gen.world_lift import (
    lift_position as _lift_position_wxyz,
)
from motion_gen.world_lift import (
    lift_position_batched as _lift_position_batched_wxyz,
)
from motion_gen.world_lift import (
    lift_vector as _lift_vector_wxyz,
)
from motion_gen.world_lift import (
    lift_vector_batched as _lift_vector_batched_wxyz,
)

__all__ = [
    "anchor_inv_yaw",
    "ego_placement",
    "ego_placement_masked",
    "lift_base_world_state",
    "lift_orientation",
    "lift_orientation_batched",
    "lift_position",
    "lift_position_batched",
    "lift_vector",
    "lift_vector_batched",
    "to_wxyz",
    "to_xyzw",
    "yaw_from_quat",
]


# ---------------------------------------------------------------------- reorder
# Both directions are a pure permutation of the last axis: exact, no
# renormalization, no numerical drift.

def to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Reorder ``(x, y, z, w)`` -> ``(w, x, y, z)``. Shape ``(..., 4)``."""
    return quat_xyzw[..., [3, 0, 1, 2]]


def to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Reorder ``(w, x, y, z)`` -> ``(x, y, z, w)``. Shape ``(..., 4)``."""
    return quat_wxyz[..., [1, 2, 3, 0]]


# ------------------------------------------------------- scalar-last pass-through

def yaw_from_quat(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw angle (rad) about world z from an ``(x, y, z, w)`` quaternion ``[..., 4]``."""
    return _yaw_from_quat_xyzw(quat_xyzw)


def ego_placement_masked(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env placement honoring a per-clip root-DOF anchor mask (all quats xyzw).

    See :func:`motion_gen.world_lift.ego_placement_masked`. Already scalar-last;
    re-exported here so Isaac Lab code has one import site for the lift.
    """
    return _ego_placement_masked_xyzw(*args, **kwargs)


def ego_placement(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy yaw+xy+z placement onto a target root (all quats xyzw).

    See :func:`motion_gen.world_lift.ego_placement`. Already scalar-last.
    """
    return _ego_placement_xyzw(*args, **kwargs)


# ------------------------------------------------------------- converted lifts

def anchor_inv_yaw(anchor_quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation lifting anchor-relative quantities into the world.

    Args:
        anchor_quat: ``[..., 4]`` anchor world quaternions (xyzw).

    Returns:
        ``[..., 4]`` lift quaternions (xyzw).
    """
    return to_xyzw(_anchor_inv_yaw_wxyz(to_wxyz(anchor_quat)))


def lift_position(
    anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_rel: torch.Tensor,
) -> torch.Tensor:
    """Lift an anchor-relative position to the world frame.

    Args:
        anchor_pos: ``[N, 3]`` anchor world positions.
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        pos_rel: ``[N, 3]`` anchor-relative positions.

    Returns:
        ``[N, 3]`` world positions.
    """
    return _lift_position_wxyz(anchor_pos, to_wxyz(anchor_quat), pos_rel)


def lift_orientation(anchor_quat: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Lift an anchor-relative orientation to the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        quat_rel: ``[N, 4]`` anchor-relative quaternions (xyzw).

    Returns:
        ``[N, 4]`` world quaternions (xyzw).
    """
    return to_xyzw(_lift_orientation_wxyz(to_wxyz(anchor_quat), to_wxyz(quat_rel)))


def lift_vector(anchor_quat: torch.Tensor, vec_rel: torch.Tensor) -> torch.Tensor:
    """Rotate an anchor-relative vector (lin/ang velocity) into the world.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        vec_rel: ``[N, 3]`` anchor-relative vectors.

    Returns:
        ``[N, 3]`` world vectors.
    """
    return _lift_vector_wxyz(to_wxyz(anchor_quat), vec_rel)


def lift_position_batched(
    anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, pos_rel: torch.Tensor,
) -> torch.Tensor:
    """Lift ``B`` anchor-relative positions per env to the world frame.

    Args:
        anchor_pos: ``[N, 3]`` anchor world positions.
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        pos_rel: ``[N, B, 3]`` anchor-relative positions.

    Returns:
        ``[N, B, 3]`` world positions.
    """
    return _lift_position_batched_wxyz(anchor_pos, to_wxyz(anchor_quat), pos_rel)


def lift_orientation_batched(anchor_quat: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Lift ``B`` anchor-relative orientations per env to the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        quat_rel: ``[N, B, 4]`` anchor-relative quaternions (xyzw).

    Returns:
        ``[N, B, 4]`` world quaternions (xyzw).
    """
    return to_xyzw(_lift_orientation_batched_wxyz(to_wxyz(anchor_quat), to_wxyz(quat_rel)))


def lift_vector_batched(anchor_quat: torch.Tensor, vec_rel: torch.Tensor) -> torch.Tensor:
    """Rotate ``B`` anchor-relative vectors per env into the world frame.

    Args:
        anchor_quat: ``[N, 4]`` anchor world quaternions (xyzw).
        vec_rel: ``[N, B, 3]`` anchor-relative vectors.

    Returns:
        ``[N, B, 3]`` world vectors.
    """
    return _lift_vector_batched_wxyz(to_wxyz(anchor_quat), vec_rel)


def lift_base_world_state(
    y_des: torch.Tensor,
    dy_des: torch.Tensor,
    anchor_pose: torch.Tensor,
    idx: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Lift the base-frame clip channels to an Isaac Lab world state vector.

    The returned layout matches the raw robot ``state`` recorded by
    ``scripts/rsl_rl/record_data.py``, with the root quaternion **xyzw** so it
    can go straight to ``write_root_pose_to_sim_index``::

        [root_pos_w(3), root_quat_w(4, xyzw), root_lin_vel_w(3),
         root_ang_vel_w(3), joint_pos(nj), joint_vel(nj)]

    Args:
        y_des: ``[N, P]`` desired position outputs (anchor-relative).
        dy_des: ``[N, V]`` desired velocity outputs (anchor-relative).
        anchor_pose: ``[N, 7]`` anchor world pose ``[pos(3), quat(4) xyzw]`` --
            an Isaac Lab pose (e.g. built from ``body_quat_w``).
        idx: index dict with long-tensor keys ``pos, ori, lin_vel, ang_vel,
            joint_pos, joint_vel``.  ``idx["ori"]`` selects the base orientation
            channels in **xyzw** order (as ``_reference_world_indices`` builds
            them); it is permuted to the core's scalar-first order here.

    Returns:
        ``[N, 3 + 4 + 3 + 3 + 2*nj]`` world state, root quaternion xyzw.
    """
    anchor_wxyz = torch.cat([anchor_pose[:, :3], to_wxyz(anchor_pose[:, 3:7])], dim=-1)
    idx_wxyz = {**idx, "ori": to_wxyz(idx["ori"])}
    state = _lift_base_world_state_wxyz(y_des, dy_des, anchor_wxyz, idx_wxyz)
    return torch.cat([state[:, :3], to_xyzw(state[:, 3:7]), state[:, 7:]], dim=-1)
