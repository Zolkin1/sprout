# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rotate a flat clip's built outputs onto a uniform slope.

A slope clip is a flat clip whose whole body is rigidly rotated by the slope pitch
about the world y-axis, pivoting at the step-0 ground point ``[x0, y0, 0]`` -- so the
feet that started on ``z=0`` land on the tilted plane and the clip climbs.  Only the
``y_des`` (positions + orientations) and ``dy_des`` (velocities) channels are touched;
joints (the gait) and ``contact`` are invariant under the rigid rotation (a foot
planted on the flat ground at step ``t`` is planted on the tilted plane at step ``t``),
so the caller pairs the rotated outputs with the *flat* build's ``contact`` -- which
also sidesteps :meth:`ClipOutputBuilder.build`'s world-flat-ground contact test that
would mislabel a climbing clip.

Quaternions use the **wxyz** (scalar-first) convention of
:mod:`motion_gen.math_utils` (``quat_apply`` / ``quat_mul``).  Orientation channels
are read/written by name as ``(ori_w, ori_x, ori_y, ori_z)`` (exactly how
:meth:`ClipOutputBuilder.build` stored them), positions as ``(x, y, z)``, velocities
as ``(x, y, z)``.  Positions rotate about the ground pivot, orientations left-multiply
by the pitch quat, velocities rotate about the origin -- the same rigid rotation.

Sign convention: ``pitch_rad`` is the rotation about the world **+y** axis.  A forward
(+x) walk under a NEGATIVE pitch climbs (+z); :func:`slope_pitch_for_climb` maps an
uphill slope angle to the correct signed pitch.

.. warning:: SLOPES ARE DEFERRED / INCOMPLETE (2026-07-17).  TODO: come back and fix.

   Feet always land on the tilted plane (the rigid rotation is direction-agnostic),
   but the UPRIGHT-pelvis compensation in :func:`upright_slope_clip` only adds to the
   hip **pitch** joints, which is correct ONLY when the clip walks straight up/down the
   slope (heading facing the ascent, +x).  The world cross-slope tilt shows up in the
   body frame as PITCH at that heading, but as ROLL when walking across the slope
   (heading +y), and as a per-frame pitch+roll MIX at any other/turning heading.  Our
   flat source clips turn and strafe all over, so the current compensation is wrong
   wherever a clip is not facing up-slope (the torso is left leaning sideways).

   To fix properly:
     1. Decompose the un-tilt into the pelvis BODY frame per frame (using that frame's
        yaw) and distribute it to hip **pitch + roll** (reduces to today's code at
        yaw 0).  Handles cross-slope and turning clips.
     2. Even (1) only keeps the torso upright -- a true side-hill gait also has the
        uphill leg more flexed than the downhill leg, a wider stance, etc., which the
        rotate-a-flat-clip approach cannot produce.  A real fix likely needs slope
        gait data (like the stair trajectories) rather than rotated flat clips.
   Also: hip-only comp leaves ~1cm foot penetration at 10 deg (knee/ankle uncompensated);
   revisit the FIXED-yaw slope anchor mask once heading is handled; and the RL terrain
   importer / reset for slope tiles still needs sim validation.
"""

from __future__ import annotations

import math

import torch

from ..joint_names import world_frame_names
from ..math_utils import quat_apply, quat_mul


def slope_pitch_for_climb(slope_rad: float) -> float:
    """Signed +y-axis pitch that turns a forward (+x) flat walk into an uphill climb.

    A rotation by ``phi`` about +y maps ``+x -> [cos phi, 0, -sin phi]``, so forward
    motion gains height only for ``phi < 0``.  Hence an uphill slope of angle
    ``slope_rad`` (>0) uses ``pitch = -slope_rad``.
    """
    return -float(slope_rad)


def _pitch_quat_wxyz(pitch_rad: float, device, dtype) -> torch.Tensor:
    """Unit quaternion (wxyz) for a rotation of ``pitch_rad`` about the world +y axis."""
    h = 0.5 * float(pitch_rad)
    return torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0], device=device, dtype=dtype)


def rotate_clip_to_slope(
    y_des: torch.Tensor,
    dy_des: torch.Tensor,
    pos_output_names: list[str],
    vel_output_names: list[str],
    pitch_rad: float,
    pivot_xy: tuple[float, float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigidly rotate a flat clip's outputs to a slope of the given +y-axis pitch.

    Args:
        y_des: ``[T, P]`` clip-world position outputs (pelvis + EE pos + ori).
        dy_des: ``[T, V]`` clip-world velocity outputs (linear + angular).
        pos_output_names: Ordered position-output names (``{frame}:pos_*`` / ``ori_*``).
        vel_output_names: Ordered velocity-output names.
        pitch_rad: Rotation about the world +y axis (see :func:`slope_pitch_for_climb`).
        pivot_xy: World ``(x, y)`` of the rotation pivot; z is pinned to 0 (ground) so
            the step-0 stance foot stays grounded. Typically the step-0 pelvis ``xy``.

    Returns:
        ``(y_des', dy_des')`` new tensors with every WORLD frame rotated; joints and
        root-frame (``@root``) channels untouched -- both are invariant under a rigid
        rotation of the clip, so rotating them would corrupt the pose.
        ``contact`` is not an argument (invariant) -- keep the flat build's contact.
    """
    y = y_des.clone()
    dy = dy_des.clone()
    T = y.shape[0]
    dev, dtype = y.device, y.dtype
    q = _pitch_quat_wxyz(pitch_rad, dev, dtype).expand(T, 4)
    pivot = torch.tensor([float(pivot_xy[0]), float(pivot_xy[1]), 0.0], device=dev, dtype=dtype)

    def _idx(names: list[str], frame: str, comps: tuple[str, ...], kind: str) -> torch.Tensor:
        return torch.tensor(
            [names.index(f"{frame}:{kind}_{a}") for a in comps], dtype=torch.long, device=dev
        )

    # WORLD frames only -- ``@root`` channels are pelvis-relative and invariant under this
    # rigid rotation, so touching them would corrupt the pose.
    world_pos_frames = set(world_frame_names(pos_output_names))
    world_vel_frames = set(world_frame_names(vel_output_names))

    # Position frames: rotate the position about the ground pivot, left-multiply the
    # orientation by the pitch quat (both read/written wxyz, as build stored them).
    for f in sorted(f for f in world_pos_frames if f"{f}:pos_x" in pos_output_names):
        pos_i = _idx(pos_output_names, f, ("x", "y", "z"), "pos")
        y[:, pos_i] = pivot + quat_apply(q, y[:, pos_i] - pivot)
        if f"{f}:ori_w" in pos_output_names:
            ori_i = _idx(pos_output_names, f, ("w", "x", "y", "z"), "ori")
            y[:, ori_i] = quat_mul(q, y[:, ori_i])

    # Velocity frames: rotate only (linear pos_*, angular ori_*).
    for f in sorted(f for f in world_vel_frames if f"{f}:pos_x" in vel_output_names):
        lin_i = _idx(vel_output_names, f, ("x", "y", "z"), "pos")
        dy[:, lin_i] = quat_apply(q, dy[:, lin_i])
    for f in sorted(f for f in world_vel_frames if f"{f}:ori_x" in vel_output_names):
        ang_i = _idx(vel_output_names, f, ("x", "y", "z"), "ori")
        dy[:, ang_i] = quat_apply(q, dy[:, ang_i])

    return y, dy


def upright_slope_clip(
    builder,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joints: torch.Tensor,
    hip_pitch_idx,
    pitch_rad: float,
    pivot_xy: tuple[float, float] | torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an UPRIGHT-torso slope clip from a flat clip's root trajectory.

    Unlike :func:`rotate_clip_to_slope` (which rigidly tilts the whole body so the
    torso leans back parallel to the ramp), this climbs only the pelvis POSITION onto
    the slope, keeps the pelvis ORIENTATION world-vertical (as in the flat clip), and
    adds the ramp pitch to the hip-pitch joints so the legs still reach the incline --
    i.e. it flexes the hips and keeps the torso vertical, the way a real climb looks.
    The feet still land on the ramp (verified: stance foot within the ankle offset of
    the plane).  ``y_des`` / ``dy_des`` are rebuilt by ``builder`` (a
    :class:`ClipOutputBuilder`, whose FK places the feet from the compensated legs);
    ``contact`` is rotation-invariant so the caller keeps the flat clip's contact.

    Args:
        builder: a ``ClipOutputBuilder`` (FK source; its joint order must match ``joints``).
        root_pos: ``[T, 3]`` flat pelvis position.
        root_quat: ``[T, 4]`` wxyz flat pelvis orientation (kept vertical).
        joints: ``[T, nj]`` flat joints (builder's FK order).
        hip_pitch_idx: iterable of the left/right hip-pitch joint indices in ``joints``.
        pitch_rad: +y-axis pitch (use :func:`slope_pitch_for_climb` of the slope angle).
        pivot_xy: ground pivot ``(x, y)`` (typically the step-0 pelvis xy).
        dt: control dt.

    Returns:
        ``(y_des[T, P], dy_des[T, V])`` -- rebuilt outputs; pair with the flat contact.
    """
    T = root_pos.shape[0]
    dev, dtype = root_pos.device, root_pos.dtype
    h = 0.5 * float(pitch_rad)
    q = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0], device=dev, dtype=dtype).expand(T, 4)
    pivot = torch.tensor([float(pivot_xy[0]), float(pivot_xy[1]), 0.0], device=dev, dtype=dtype)
    root_pos_up = pivot + quat_apply(q, root_pos - pivot)   # climb the pelvis onto the ramp
    joints_up = joints.clone()
    for i in hip_pitch_idx:                                 # flex hips so legs reach the incline
        joints_up[:, i] = joints_up[:, i] + float(pitch_rad)
    y, dy, _contact = builder.build(root_pos_up, root_quat.clone(), joints_up, dt)
    return y, dy
