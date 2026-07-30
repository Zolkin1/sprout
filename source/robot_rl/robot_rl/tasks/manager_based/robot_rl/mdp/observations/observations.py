# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_box_minus, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ref_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_des.clone()
    return ref_traj

def act_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act.clone()
    return act_traj

def ref_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj_vel = cmd.dy_des
    return ref_traj_vel

def act_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj_vel = cmd.dy_act
    return act_traj_vel


# TODO: Delete ``_phased_cmd``, ``ref_sin_phase``, ``ref_cos_phase``
# along with :class:`PhasedTrajectoryCommand`.  The multi-skill bases
# use :func:`multiskill_phase` (below) which reads the multi-skill
# command's phase directly — replace any g1 single-skill obs that still
# wires up sin/cos phase via ``_phased_cmd`` before removing.
# ----------------------------------------------------------------------
# Motion-sequence (open-loop clip tracking) observations.
#
# The CLF runs in the world frame, so the raw reference vectors (``ref_traj``
# etc.) are absolute-world (critic-only).  The helpers here are all relative /
# heading-invariant (they difference reference-vs-actual or ee-vs-base and
# rotate into a relative frame), so they are frame-agnostic and safe for the
# policy.  Pair these with a :class:`MotionSequenceCommand` term.
# ----------------------------------------------------------------------

def _joint_output_indices(ordered_names: list[str]) -> list[int]:
    """Indices of the ``joint:*`` outputs within an ordered name list."""
    return [i for i, n in enumerate(ordered_names) if n.startswith("joint:")]


def base_pose_error(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference-vs-actual base pose error, expressed in the actual base frame.

    Returns ``[N, 6]`` = position error (3) + orientation error (3), *both*
    rotated into the robot's current base frame, so the observation is fully
    heading-invariant (no absolute world heading leaks) and deployable sim2real.
    The orientation half is the world-frame ``quat_box_minus`` rotation vector
    (a spatial error whose roll/pitch axes would otherwise rotate with heading)
    re-expressed in the base frame to match the position half.  Same base
    tracking error the CLF penalises, surfaced as a base-relative observation.
    """
    cmd = env.command_manager.get_term(command_name)
    state = cmd.reference_world_state()                 # [N, 3+4+3+3+2nj]
    ref_pos_w = state[:, 0:3]
    ref_quat_w = state[:, 3:7]                          # xyzw
    root_pos_w = wp.to_torch(cmd.robot.data.root_pos_w)
    root_quat_w = wp.to_torch(cmd.robot.data.root_quat_w)  # xyzw
    inv_root_quat = quat_inv(root_quat_w)

    pos_err_b = quat_apply(inv_root_quat, ref_pos_w - root_pos_w)
    ori_err_w = quat_box_minus(root_quat_w, ref_quat_w)  # actual ⊟ reference (world frame)
    ori_err_b = quat_apply(inv_root_quat, ori_err_w)     # -> current base frame
    return torch.cat([pos_err_b, ori_err_b], dim=-1)


def joint_pos_ref(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference joint positions (the ``joint:*`` channels of ``y_des``)."""
    cmd = env.command_manager.get_term(command_name)
    idx = _joint_output_indices(cmd.ordered_pos_output_names)
    return cmd.y_des[:, idx].clone()


def joint_vel_ref(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference joint velocities (the ``joint:*`` channels of ``dy_des``)."""
    cmd = env.command_manager.get_term(command_name)
    idx = _joint_output_indices(cmd.ordered_vel_output_names)
    return cmd.dy_des[:, idx].clone()


def _root_frame_indices(cmd, names: list[str], kinds: tuple[str, ...]) -> list[int]:
    """Channel indices of the root-frame body block, body-major then ``kinds`` order.

    Args:
        cmd: The trajectory command (supplies ``root_body_frames``).
        names: ``ordered_pos_output_names`` or ``ordered_vel_output_names``.
        kinds: Channel suffixes per body, e.g. ``("pos_x", ..., "ori_w")``.

    Returns:
        Indices into ``names``.

    Raises:
        ValueError: If the command tracks no root-frame bodies.
    """
    frames = getattr(cmd, "root_body_frames", None)
    if not frames:
        raise ValueError(
            "body_*_ref_root observations need a clip library carrying root-frame body "
            "channels; widen it with scripts/rsl_rl/backfill_root_frames.py."
        )
    return [names.index(f"{f}:{k}") for f in frames for k in kinds]


def body_pos_ref_root(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference body-link poses in the reference ROOT frame (``[N, 7 * B]``).

    The DeepMimic / Sonic style target: each tracked link's position + orientation expressed
    in the pelvis frame, straight off the stored ``<body>@root:*`` channels.  Already
    heading-invariant and free of world coordinates, so no transform is needed here and the
    observation is deployable as-is.

    Laid out body-major as ``[pos_xyz, ori_wxyz]`` per body, in ``cmd.root_body_frames``
    order.  Replaces :func:`joint_pos_ref` under ``tracking_mode="body"``.
    """
    cmd = env.command_manager.get_term(command_name)
    idx = _root_frame_indices(
        cmd, cmd.ordered_pos_output_names,
        ("pos_x", "pos_y", "pos_z", "ori_w", "ori_x", "ori_y", "ori_z"),
    )
    return cmd.y_des[:, idx].clone()


def body_vel_ref_root(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference body-link twists in the reference ROOT frame (``[N, 6 * B]``).

    The velocity counterpart of :func:`body_pos_ref_root`: each tracked link's linear and
    angular velocity relative to the pelvis, expressed in the pelvis frame.  Laid out
    body-major as ``[lin_xyz, ang_xyz]``.  Replaces :func:`joint_vel_ref` under
    ``tracking_mode="body"``.
    """
    cmd = env.command_manager.get_term(command_name)
    idx = _root_frame_indices(
        cmd, cmd.ordered_vel_output_names,
        ("pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z"),
    )
    return cmd.dy_des[:, idx].clone()


def ref_base_velocity_local(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference base linear+angular velocity rotated into the current base frame.

    Deployable counterpart to :func:`ref_base_velocity` for world-frame trackers
    (e.g. :class:`FmTrackingCommand`): there ``dy_des`` holds the base
    velocity in the world frame, which would leak absolute heading if shown to
    the policy directly. Rotating it by the inverse current base orientation
    yields a heading-invariant command (available on hardware from the IMU).
    Returns ``[N, 6]`` (lin xyz + ang xyz, current base frame).
    """
    cmd = env.command_manager.get_term(command_name)
    base = cmd.cfg.base_frame_name
    names = cmd.ordered_vel_output_names
    lin_idx = [names.index(f"{base}:pos_{a}") for a in ("x", "y", "z")]
    ang_idx = [names.index(f"{base}:ori_{a}") for a in ("x", "y", "z")]
    lin_w = cmd.dy_des[:, lin_idx]
    ang_w = cmd.dy_des[:, ang_idx]
    inv_root = quat_inv(wp.to_torch(cmd.robot.data.root_quat_w))
    return torch.cat([quat_apply(inv_root, lin_w), quat_apply(inv_root, ang_w)], dim=-1)


def command_velocity(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Joystick-level base velocity command that generated the clip (``target_vel``).

    The upstream commanded velocity ``(vx, vy, wz)`` -- the signal a velocity
    "joystick" would supply -- rather than the achieved reference base velocity
    (:func:`ref_base_velocity`, ``dy_des``).  Heading-invariant (planar linear
    velocity + yaw rate in the base/heading frame) and available at deployment.
    Pairs with the ``command_lin_vel_xy_tracking_exp`` /
    ``command_ang_vel_z_tracking_exp`` rewards so the command they grade against
    is observable to the policy.  Returns ``[N, 3]``.  Requires a clip library
    carrying ``target_vel``.
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.get_command_velocity().clone()


def _lookahead_offsets(cmd, horizon_s: float, stride: int) -> torch.Tensor:
    """Strided future step offsets covering ``horizon_s`` at the clip control dt.

    e.g. ``horizon_s=0.5, stride=5`` with ``dt=0.02`` -> ``[5, 10, 15, 20, 25]``
    (0.1 .. 0.5 s).  Always returns at least one offset.
    """
    dt = float(cmd.manager.dt)
    horizon_steps = max(1, int(round(horizon_s / dt)))
    stride = max(1, int(stride))
    offs = torch.arange(stride, horizon_steps + 1, stride, device=cmd.device, dtype=torch.long)
    if offs.numel() == 0:
        offs = torch.tensor([horizon_steps], device=cmd.device, dtype=torch.long)
    return offs


def future_command_velocity(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", horizon_s: float = 0.5, stride: int = 5,
) -> torch.Tensor:
    """Look-ahead commanded velocity over a future window (``[N, K * 3]``).

    The upstream ``target_vel`` ``(vx, vy, wz)`` (heading frame) at strided future
    steps covering ``horizon_s`` -- lets the privileged teacher anticipate an
    upcoming slowdown / direction change before the current reference changes.
    Flattened as ``[step0(3), step1(3), ...]`` (the per-step layout the symmetry
    tiler expects).  Requires a clip library carrying ``target_vel``.
    """
    cmd = env.command_manager.get_term(command_name)
    offs = _lookahead_offsets(cmd, horizon_s, stride)
    return cmd.get_future_command_velocity(offs).reshape(env.num_envs, -1)


def future_base_pose(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", horizon_s: float = 0.5, stride: int = 5,
) -> torch.Tensor:
    """Look-ahead reference base pose error over a future window (``[N, K * 6]``).

    For each strided future step covering ``horizon_s``, the reference base pose
    expressed relative to the robot's *current* base -- a "future
    :func:`base_pose_error`": position error (3) + orientation error (3), both
    rotated into the current base frame (the orientation half is the world-frame
    ``quat_box_minus`` rotation vector re-expressed in the base frame, matching
    :func:`base_pose_error`).  Fully heading-invariant.  Flattened as
    ``[step0(6), step1(6), ...]``.
    """
    cmd = env.command_manager.get_term(command_name)
    offs = _lookahead_offsets(cmd, horizon_s, stride)
    pos_w, quat_w, _, _ = cmd.get_future_world_base(offs)       # [N, K, 3], [N, K, 4]
    n, k, _ = pos_w.shape
    root_pos_w = wp.to_torch(cmd.robot.data.root_pos_w)         # [N, 3]
    root_quat_w = wp.to_torch(cmd.robot.data.root_quat_w)       # [N, 4]
    inv_root = quat_inv(root_quat_w)[:, None, :].expand(n, k, 4).reshape(n * k, 4)
    root_q = root_quat_w[:, None, :].expand(n, k, 4).reshape(n * k, 4)
    pos_err_b = quat_apply(inv_root, (pos_w - root_pos_w[:, None, :]).reshape(n * k, 3))
    ori_err_w = quat_box_minus(root_q, quat_w.reshape(n * k, 4))  # actual ⊟ reference (world)
    ori_err_b = quat_apply(inv_root, ori_err_w)                  # -> current base frame
    out = torch.cat([pos_err_b, ori_err_b], dim=-1).reshape(n, k, 6)
    return out.reshape(n, -1)


def future_base_velocity(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", horizon_s: float = 0.5, stride: int = 5,
) -> torch.Tensor:
    """Look-ahead reference base velocity over a future window (``[N, K * 6]``).

    For each strided future step covering ``horizon_s``, the reference base
    linear+angular velocity rotated into the robot's *current* base frame
    (mirrors :func:`ref_base_velocity_local`).  Heading-invariant.  Flattened as
    ``[step0(6), step1(6), ...]``.
    """
    cmd = env.command_manager.get_term(command_name)
    offs = _lookahead_offsets(cmd, horizon_s, stride)
    _, _, lin_w, ang_w = cmd.get_future_world_base(offs)        # [N, K, 3], [N, K, 3]
    n, k, _ = lin_w.shape
    root_quat_w = wp.to_torch(cmd.robot.data.root_quat_w)       # [N, 4]
    inv_root = quat_inv(root_quat_w)[:, None, :].expand(n, k, 4).reshape(n * k, 4)
    lin_b = quat_apply(inv_root, lin_w.reshape(n * k, 3)).reshape(n, k, 3)
    ang_b = quat_apply(inv_root, ang_w.reshape(n * k, 3)).reshape(n, k, 3)
    return torch.cat([lin_b, ang_b], dim=-1).reshape(n, -1)


def ee_ref_relative_to_base(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", frame_names: list[str] | None = None,
) -> torch.Tensor:
    """Reference end-effector positions relative to the reference base.

    Re-expresses each named end-effector position in the *reference base* frame
    (target posture), so no world coordinates appear.  Frame-agnostic: the ee
    and base reference positions are differenced and rotated into the reference
    base orientation, so it is correct whether ``y_des`` is clip-world or world.
    Returns ``[N, 3 * len(frame_names)]``.

    Args:
        frame_names: End-effector frame names (e.g. the ankle/wrist links).
            Defaults to all reference output frames except the base frame.
    """
    cmd = env.command_manager.get_term(command_name)
    pos_names = cmd.ordered_pos_output_names
    base = cmd.cfg.base_frame_name

    # Resolved by the command so the symmetry reflector (get_symmetric_ee) uses
    # the identical frame list/order -- single source of truth, no drift.
    frames = cmd.resolve_ee_frames(frame_names)

    base_pos = cmd.y_des[:, [pos_names.index(f"{base}:pos_{a}") for a in ("x", "y", "z")]]
    # xyzw (scalar last) to match Isaac Lab's quat math; outputs store ori_w too.
    base_quat = cmd.y_des[:, [pos_names.index(f"{base}:ori_{a}") for a in ("x", "y", "z", "w")]]
    inv_base_quat = quat_inv(base_quat)

    out = []
    for f in frames:
        ee_pos = cmd.y_des[:, [pos_names.index(f"{f}:pos_{a}") for a in ("x", "y", "z")]]
        out.append(quat_apply(inv_base_quat, ee_pos - base_pos))
    return torch.cat(out, dim=-1)


