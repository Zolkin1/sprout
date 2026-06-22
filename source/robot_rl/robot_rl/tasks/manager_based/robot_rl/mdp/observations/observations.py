# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.observations import generated_commands
from isaaclab.utils.math import quat_apply, quat_box_minus, quat_inv

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.phased_trajectory_cmd import (
    PhasedTrajectoryCommand,
)

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
def _phased_cmd(env: ManagerBasedRLEnv, command_name: str) -> PhasedTrajectoryCommand:
    cmd = env.command_manager.get_term(command_name)
    if not isinstance(cmd, PhasedTrajectoryCommand):
        raise TypeError(
            f"Phase observation requires a PhasedTrajectoryCommand for "
            f"'{command_name}', got {type(cmd).__name__}."
        )
    return cmd


def ref_sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = _phased_cmd(env, command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    sphase = torch.sin(phase)
    if sphase.ndim == 1:
        # [B] → [B, 1]
        sphase = sphase.unsqueeze(-1)

    return sphase

def ref_cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = _phased_cmd(env, command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    cphase = torch.cos(phase)
    if cphase.ndim == 1:
        # [B] → [B, 1]
        cphase = cphase.unsqueeze(-1)
    return cphase

def grace_period_active(
    env: ManagerBasedRLEnv,
    command_name: str,
    grace_period_s: float,
    one_hot: bool = True,
) -> torch.Tensor:
    """Indicator for whether the post-trajectory-change grace period is active.

    Reads ``time_since_traj_change_s`` from the named command term (populated
    by :class:`BatchedMultiSkillCommand._pre_update_phase`) and compares it
    against the configured grace window. The flag is True while
    ``time_since_traj_change_s < grace_period_s``.

    Args:
        env: The Isaac Lab RL environment.
        command_name: Trajectory command term exposing
            ``time_since_traj_change_s``.
        grace_period_s: Grace-window length in seconds. Pair with the value
            on the matching ``frame_deviation_from_reference`` termination
            to keep them in sync.
        one_hot: When True, return a ``[num_envs, 2]`` one-hot
            ``[not_in_grace, in_grace]``. When False, return a
            ``[num_envs, 1]`` 0/1 flag.

    Returns:
        Float tensor of shape ``[num_envs, 2]`` (one-hot) or
        ``[num_envs, 1]`` (single flag). If the command term doesn't expose
        ``time_since_traj_change_s`` (e.g. single-skill commands), reports
        "not in grace" for all envs.
    """
    cmd = env.command_manager.get_term(command_name)
    time_since = getattr(cmd, "time_since_traj_change_s", None)
    if time_since is None:
        return torch.zeros(
            (env.num_envs, 2 if one_hot else 1),
            device=env.device,
        )
    in_grace = time_since < grace_period_s
    if one_hot:
        return torch.nn.functional.one_hot(in_grace.long(), num_classes=2).float()
    return in_grace.float().unsqueeze(-1)


def skill_one_hot(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """One-hot encoding of the trajectory cmd's active skill for each env.

    Reads ``skill_id`` from a :class:`BatchedMultiSkillCommand` term and
    converts it into a one-hot vector of length
    ``len(cfg.velocity_buckets)``.  The active skill is bucket-derived
    from the live ``vel_target_b`` and lags the velocity cmd's sampled
    bucket across the gate-on-contact deferral window — this is the
    skill the policy is actually executing.

    Returns a tensor of shape ``(num_envs, num_buckets)``.
    """
    cmd = env.command_manager.get_term(command_name)
    num_buckets = len(cmd.cfg.velocity_buckets)
    one_hot = torch.zeros(env.num_envs, num_buckets, device=env.device)
    in_bucket = cmd.skill_id < num_buckets
    one_hot[in_bucket] = torch.nn.functional.one_hot(
        cmd.skill_id[in_bucket], num_classes=num_buckets
    ).float()
    return one_hot


# ----------------------------------------------------------------------
# Motion-sequence (open-loop clip tracking) observations.
#
# All base-relative: the policy never sees a world position.  Pair these
# with a :class:`MotionSequenceCommand` term.
# ----------------------------------------------------------------------

def _joint_output_indices(ordered_names: list[str]) -> list[int]:
    """Indices of the ``joint:*`` outputs within an ordered name list."""
    return [i for i, n in enumerate(ordered_names) if n.startswith("joint:")]


def base_pose_error(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference-vs-actual base pose error, expressed in the actual base frame.

    Returns ``[N, 6]`` = position error (3, rotated into the robot base frame)
    + orientation error (3, ``quat_box_minus(actual, reference)`` tangent).
    This is the same base tracking error the CLF penalises, surfaced as a
    base-relative observation (no world coordinates).
    """
    cmd = env.command_manager.get_term(command_name)
    state = cmd.reference_world_state()                 # [N, 3+4+3+3+2nj]
    ref_pos_w = state[:, 0:3]
    ref_quat_w = state[:, 3:7]                          # wxyz
    root_pos_w = wp.to_torch(cmd.robot.data.root_pos_w)
    root_quat_w = wp.to_torch(cmd.robot.data.root_quat_w)  # wxyz

    pos_err_b = quat_apply(quat_inv(root_quat_w), ref_pos_w - root_pos_w)
    ori_err = quat_box_minus(root_quat_w, ref_quat_w)   # actual ⊟ reference
    return torch.cat([pos_err_b, ori_err], dim=-1)


def joint_pos_ref(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference joint positions (the ``joint:*`` channels of ``y_des``)."""
    cmd = env.command_manager.get_term(command_name)
    idx = _joint_output_indices(cmd.ordered_pos_output_names)
    return cmd.y_des[:, idx].clone()


def ref_base_velocity(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference base (pelvis) linear+angular velocity from the clip (``dy_des``).

    The velocity target the ``ref_base_vel_tracking_exp`` reward grades against;
    exposing it (paired with the actual base velocity in ``act_traj_vel`` /
    ``base_*_vel``) makes that reward's error observable to the policy.  Returns
    ``[N, 6]`` (lin xyz + ang xyz, yaw-aligned anchor frame).
    """
    cmd = env.command_manager.get_term(command_name)
    base = cmd.cfg.base_frame_name
    names = cmd.ordered_vel_output_names
    idx = (
        [names.index(f"{base}:pos_{a}") for a in ("x", "y", "z")]
        + [names.index(f"{base}:ori_{a}") for a in ("x", "y", "z")]
    )
    return cmd.dy_des[:, idx].clone()


def joint_vel_ref(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference joint velocities (the ``joint:*`` channels of ``dy_des``)."""
    cmd = env.command_manager.get_term(command_name)
    idx = _joint_output_indices(cmd.ordered_vel_output_names)
    return cmd.dy_des[:, idx].clone()


def ref_base_velocity_local(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Reference base linear+angular velocity rotated into the current base frame.

    Deployable counterpart to :func:`ref_base_velocity` for world-frame trackers
    (e.g. :class:`GeneratorTrackingCommand`): there ``dy_des`` holds the base
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
    :func:`base_pose_error`": position error (3, rotated into the current base
    frame) + orientation error (3, ``quat_box_minus`` tangent).  Heading-
    invariant.  Flattened as ``[step0(6), step1(6), ...]``.
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
    ori_err = quat_box_minus(root_q, quat_w.reshape(n * k, 4))  # actual ⊟ reference
    out = torch.cat([pos_err_b, ori_err], dim=-1).reshape(n, k, 6)
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

    The clip stores frame outputs relative to the stance anchor; this
    re-expresses each named end-effector position in the *reference base*
    frame (target posture), so no world coordinates appear.  Returns
    ``[N, 3 * len(frame_names)]``.

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


def multiskill_phase(env: ManagerBasedRLEnv, frequency_list: list[float], command_name: str) -> torch.Tensor:
    """Create phasing variables at different frequencies to cover a range.

    Returns a tensor of shape (num_envs, 2 * num_frequencies) with sin and cos
    values interleaved: [sin_f0, cos_f0, sin_f1, cos_f1, ...].
    """
    # TODO: Make the frequency list read from the traj_ref command
    frequencies = torch.tensor(frequency_list, device=env.device)   # Hz
    num_freq = len(frequencies)

    ## DEBUG
    # cmd = env.command_manager.get_term(command_name)
    # print(f"Command phasing var: {cmd.get_phasing_var()}")
    ## DEBUG END

    # TODO: Should align this with the sampled reset time
    t = env.episode_length_buf * env.step_dt

    ## DEBUG
    # full_period = [2*0.299, 2*0.46]
    # phasing_vars = [(t % full_period[i])/(full_period[i]) for i in range(num_freq)]
    # commanded_velocity = env.command_manager.get_command("base_velocity")
    # print(f"Obs phasing var: {phasing_vars}\nCommanded velocity: {commanded_velocity}\n")
    ## DEBUG END

    sp = torch.zeros(env.num_envs, num_freq, device=env.device)
    cp = torch.zeros(env.num_envs, num_freq, device=env.device)

    for i in range(num_freq):
        sp[:, i] = torch.sin(2 * torch.pi * frequencies[i] * t)
        cp[:, i] = torch.cos(2 * torch.pi * frequencies[i] * t)

    return torch.cat([sp, cp], dim=-1)
