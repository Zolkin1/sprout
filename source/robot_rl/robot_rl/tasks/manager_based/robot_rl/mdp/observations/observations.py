# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_box_minus, quat_inv
from isaaclab.utils.noise.noise_cfg import NoiseCfg

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


# ----------------------------------------------------------------------
# Merged reference window (current step + strided look-ahead).
#
# One term carries the WHOLE reference trajectory -- base pose/velocity, joint
# angles, joint velocities and end-effector positions -- at ``num_steps`` clip
# steps spaced ``stride`` control steps apart, starting at the CURRENT step
# (offset 0).  ``num_steps=1`` therefore degenerates to the current step alone.
#
# The proprioceptive history is a separate mechanism (per-term
# ``ObservationTermCfg.history_length``, one entry per control step); only the
# FUTURE is decimated by ``stride``.
# ----------------------------------------------------------------------

def _window_offsets(cmd, num_steps: int, stride: int) -> torch.Tensor:
    """Clip-step offsets ``[0, stride, 2*stride, ...]`` with ``num_steps`` entries.

    ``num_steps=1`` -> ``[0]`` (current step only).  At the 0.02 s control dt,
    ``num_steps=11, stride=5`` -> ``[0, 5, ..., 50]`` = 0.0 .. 1.0 s.
    """
    n = max(1, int(num_steps))
    s = max(1, int(stride))
    return torch.arange(n, device=cmd.device, dtype=torch.long) * s


def reference_window_layout(cmd) -> list[tuple[str, int]]:
    """Per-step block layout of :func:`reference_window` as ``[(name, width), ...]``.

    The single source of truth for how one step of the window is packed, so the
    diagnostics (``scripts/rsl_rl/data_logger.py``) can slice the flat term back
    into named blocks without restating the layout.  Widths depend on the robot
    (joint count) and the clip library (ee frame count).
    """
    nj = len(cmd.robot.joint_names)
    n_ee = len(cmd.resolve_ee_frames(None))
    return [
        ("base_pose", 6),
        ("base_vel", 6),
        ("joint_pos", nj),
        ("joint_vel", nj),
        ("ee_pos", 3 * n_ee),
    ]


#: Default per-block uniform-noise HALF-WIDTHS for :func:`reference_window`.
#:
#: The window merges quantities on very different scales -- metres, radians, rad/s -- so a
#: single scalar cannot serve them all: the level that is right for a joint angle is
#: negligible for a joint velocity.  These stand in for the FM's prediction error at
#: deploy, which is what the reference noise exists to simulate.  Position-like entries are
#: per-axis ``(x, y, z)``; the vertical axis is tighter than the horizontal ones.
REFERENCE_WINDOW_NOISE: dict[str, float | list[float]] = {
    "base_pos": [0.05, 0.05, 0.01],   # reference base position error, metres
    "base_ori": 0.03,                 # reference base orientation error, radians
    "base_lin_vel": 0.25,             # m/s
    "base_ang_vel": 0.25,             # rad/s
    "joint_pos": 0.1,                 # rad
    "joint_vel": 0.5,                 # rad/s
    "ee_pos": [0.05, 0.05, 0.01],     # ee position rel. reference base, metres
}


#: Per-block multipliers applied to :func:`reference_window` AFTER noise and clipping.
#:
#: Only the joint-velocity block is scaled, by the same 0.05 the proprioceptive
#: ``joint_vel`` observation uses.  Raw reference joint velocities run to ~6.5 rad/s while
#: every other block in the window is O(1), so unscaled they would be ~10x hotter than
#: their neighbours and 20x hotter than the *same physical quantity* entering the trunk
#: through proprioception -- a pure conditioning problem for the (unnormalized) encoder,
#: since the noise is applied before the scale and so the signal-to-noise ratio is
#: untouched.
#:
#: TODO(scaling): this is a first pass, tuned only to put the blocks in the same ballpark.
#: Two things to revisit once there is a trained baseline to compare against:
#:
#: 1. These numbers are NOT normalization -- they are hand-picked, and the remaining
#:    spread is still ~5x (measured p99 after scaling: base_pose 0.68, base_vel 1.14,
#:    joint_pos 1.61, joint_vel 0.32, ee_pos 0.74).  ``agent.actor.obs_normalization=true``
#:    would handle it properly (``RefEncoderMLPModel`` normalizes the encoder branch
#:    separately), at the cost of changing training dynamics vs. the unnormalized runs.
#: 2. The scale does DOUBLE DUTY: it also re-weights the optional reconstruction loss,
#:    because an MSE weights each channel by its magnitude squared.  Measured share of that
#:    loss -- before this scale: joint_vel 92%, everything else 8% combined; after:
#:    joint_pos 62%, base_vel 18%, ee_pos 13%, base_pose 4%, joint_vel 3%.  So the
#:    ablation's objective is decided by unit magnitudes rather than deliberately.  Fix it
#:    with a per-block weight vector inside
#:    ``RefEncoderMLPModel.reconstruction_loss`` rather than by distorting this scale
#:    further -- this one also feeds the actor's forward pass, so the two cannot be tuned
#:    independently through it.
REFERENCE_WINDOW_SCALE: dict[str, float | list[float]] = {
    "base_pos": 1.0,
    "base_ori": 1.0,
    "base_lin_vel": 1.0,
    "base_ang_vel": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,   # matches the proprioceptive joint_vel term
    "ee_pos": 1.0,
}


def _expand_reference_blocks(
    num_steps: int,
    num_joints: int,
    num_ee_frames: int,
    per_block: dict[str, float | list[float]],
) -> list[float]:
    """Expand per-block values to one value per :func:`reference_window` channel.

    Shared by the noise and scale builders so a layout change cannot land in one and not
    the other.

    Args:
        num_steps: Samples in the window.
        num_joints: Robot joint count.
        num_ee_frames: End-effector frames the clip library exposes.
        per_block: Value per block name; position-like blocks accept a scalar or an
            ``(x, y, z)`` triple.

    Returns:
        ``num_steps * D`` values, in the channel order the window packs.

    Raises:
        ValueError: If the built per-step width disagrees with the block layout.
    """
    def axes(key: str) -> list[float]:
        """The block's value as three per-axis entries."""
        value = per_block[key]
        return [float(v) for v in value] if isinstance(value, (tuple, list)) else [float(value)] * 3

    per_step = (
        axes("base_pos") + axes("base_ori")                        # base_pose  (6)
        + axes("base_lin_vel") + axes("base_ang_vel")              # base_vel   (6)
        + [float(per_block["joint_pos"])] * num_joints             # joint_pos  (nj)
        + [float(per_block["joint_vel"])] * num_joints             # joint_vel  (nj)
        + axes("ee_pos") * num_ee_frames                           # ee_pos     (3F)
    )
    expected = 6 + 6 + 2 * num_joints + 3 * num_ee_frames
    if len(per_step) != expected:
        raise ValueError(
            f"Built {len(per_step)} per-step channels, expected {expected} "
            f"(nj={num_joints}, ee={num_ee_frames}). The per-block expansion has drifted "
            "from reference_window_layout."
        )
    return per_step * int(num_steps)


def reference_window_noise_vector(
    num_steps: int,
    num_joints: int,
    num_ee_frames: int,
    magnitudes: dict[str, float | list[float]] | None = None,
) -> list[float]:
    """Expand per-block noise magnitudes to one half-width per :func:`reference_window` channel.

    These are RAW-unit magnitudes: Isaac Lab applies noise before clipping and scaling, so
    a joint-velocity half-width is in rad/s regardless of
    :data:`REFERENCE_WINDOW_SCALE`.

    Returns a plain list (not a tensor) so it stays a Hydra/OmegaConf primitive -- see
    :class:`ReferenceWindowNoiseCfg`.

    Args:
        num_steps: Samples in the window (``REFERENCE_NUM_STEPS``).
        num_joints: Robot joint count.
        num_ee_frames: End-effector frames the clip library exposes.
        magnitudes: Per-block half-widths; defaults to :data:`REFERENCE_WINDOW_NOISE`.

    Returns:
        ``num_steps * D`` half-widths, in the channel order the window packs.
    """
    mags = REFERENCE_WINDOW_NOISE if magnitudes is None else magnitudes
    return _expand_reference_blocks(num_steps, num_joints, num_ee_frames, mags)


def reference_window_scale(
    num_steps: int,
    num_joints: int,
    num_ee_frames: int,
    factors: dict[str, float | list[float]] | None = None,
) -> tuple[float, ...]:
    """Per-channel multiplier for a :func:`reference_window` observation term.

    Returns a TUPLE because ``ObservationTermCfg.scale`` only accepts ``float``, ``int``
    or ``tuple`` -- a list is rejected outright by the observation manager.

    Args:
        num_steps: Samples in the window (``REFERENCE_NUM_STEPS``).
        num_joints: Robot joint count.
        num_ee_frames: End-effector frames the clip library exposes.
        factors: Per-block multipliers; defaults to :data:`REFERENCE_WINDOW_SCALE`.

    Returns:
        ``num_steps * D`` multipliers, in the channel order the window packs.
    """
    facs = REFERENCE_WINDOW_SCALE if factors is None else factors
    return tuple(_expand_reference_blocks(num_steps, num_joints, num_ee_frames, facs))


def per_channel_uniform_noise(data: torch.Tensor, cfg) -> torch.Tensor:
    """Additive uniform noise with an independent half-width per channel.

    Isaac Lab's stock ``uniform_noise`` accepts tensor bounds, but a tensor cannot live in
    an ``ObservationTermCfg``: hydra pushes the whole env cfg through OmegaConf, which
    rejects it (``UnsupportedValueType: Value 'Tensor' is not a supported primitive
    type``). So the config carries a plain list and this expands it once, caching the
    tensor on the cfg after hydra's round-trip is done.

    Args:
        data: ``[N, D]`` observation to corrupt.
        cfg: A :class:`ReferenceWindowNoiseCfg` (needs ``n_half`` and ``operation``).

    Returns:
        ``data`` plus per-channel ``U(-n_half, +n_half)``.

    Raises:
        ValueError: If the half-width vector does not match the observation width, or the
            requested operation is not ``"add"``.
    """
    if cfg.operation != "add":
        raise ValueError(f"per_channel_uniform_noise only supports operation='add', got {cfg.operation!r}.")
    half = getattr(cfg, "_half_cache", None)
    if half is None or half.device != data.device or half.dtype != data.dtype:
        half = torch.as_tensor(cfg.n_half, dtype=data.dtype, device=data.device)
        cfg._half_cache = half
    if half.shape[-1] != data.shape[-1]:
        raise ValueError(
            f"Noise vector has {half.shape[-1]} channels but the observation has "
            f"{data.shape[-1]}. Check num_joints / num_ee_frames against the clip library."
        )
    return data + (torch.rand_like(data) * 2.0 - 1.0) * half


@configclass
class ReferenceWindowNoiseCfg(NoiseCfg):
    """Per-channel uniform noise for :func:`reference_window`.

    Holds the readable per-block magnitudes rather than the expanded vector, so the dumped
    ``params/env.yaml`` stays legible and hydra-safe.  Build it with
    :meth:`from_layout`, which does the expansion.

    Note this is a plain :class:`~isaaclab.utils.noise.noise_cfg.NoiseCfg` with a custom
    ``func`` rather than a :class:`~isaaclab.utils.noise.noise_model.NoiseModel`: the
    manager calls ``func`` (which survives hydra as a lazily-resolving
    :class:`~isaaclab.utils.string.ResolvableString`) but calls ``issubclass`` on a noise
    model's ``class_type``, which does not.  Same constraint the depth noiser hit -- see
    ``mdp/sensors/noise.py``.
    """

    func: str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.observations.observations:per_channel_uniform_noise"
    )

    n_half: list[float] = MISSING
    """Per-channel noise half-width, one entry per observation channel."""

    magnitudes: dict = MISSING
    """The per-block magnitudes ``n_half`` was expanded from (recorded for provenance)."""

    @classmethod
    def from_layout(
        cls,
        num_steps: int,
        num_joints: int,
        num_ee_frames: int,
        magnitudes: dict[str, float | list[float]] | None = None,
    ) -> "ReferenceWindowNoiseCfg":
        """Build the config by expanding per-block magnitudes over the window layout.

        Args:
            num_steps: Samples in the window.
            num_joints: Robot joint count.
            num_ee_frames: End-effector frames the clip library exposes.
            magnitudes: Per-block half-widths; defaults to :data:`REFERENCE_WINDOW_NOISE`.

        Returns:
            A config ready to attach as an ``ObservationTermCfg`` ``noise``.
        """
        mags = dict(REFERENCE_WINDOW_NOISE if magnitudes is None else magnitudes)
        return cls(
            n_half=reference_window_noise_vector(num_steps, num_joints, num_ee_frames, mags),
            magnitudes=mags,
        )


def reference_window(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", num_steps: int = 11, stride: int = 5,
) -> torch.Tensor:
    """The full reference trajectory over a look-ahead window (``[N, num_steps * D]``).

    Merges what used to be five separate current-step terms and two root-only
    look-ahead terms into ONE observation.  Per step, in
    :func:`reference_window_layout` order:

    * ``base_pose`` (6) -- reference base position + orientation error relative to
      the robot's *current actual* base, matching :func:`base_pose_error` (the
      orientation half is the world-frame ``quat_box_minus`` rotation vector
      re-expressed in the base frame).
    * ``base_vel`` (6) -- reference base linear+angular velocity rotated into the
      current actual base frame, matching :func:`ref_base_velocity_local`.
    * ``joint_pos`` / ``joint_vel`` (nj each) -- frame-invariant reference joint
      state.  **In the robot's own joint order**, so these line up element-wise
      with the ``joint_pos`` / ``joint_vel`` / ``actions`` proprioception terms.
      Note this differs from :func:`joint_pos_ref`, which is in clip-OUTPUT order
      (the order the deploy metadata calls ``joint_out_names``).
    * ``ee_pos`` (3 per ee frame) -- reference end-effector positions expressed in
      *that step's* reference base frame, matching
      :func:`ee_ref_relative_to_base` (posture, free of tracking error).

    Every block is relative / heading-invariant, so no world coordinate leaks to
    the policy.  Flattened as ``[step0(D), step1(D), ...]`` -- the per-step layout
    the symmetry tiler expects.

    Note the window saturates on the clip's final padded frame near the end of a
    clip (``_future_step_indices`` clamps to the padded library length); the
    episode is truncated by ``clip_end_reached`` before that matters.
    """
    cmd = env.command_manager.get_term(command_name)
    offs = _window_offsets(cmd, num_steps, stride)

    # World-frame reference base over the window; also the anchor the ee block uses.
    pos_w, quat_w, lin_w, ang_w = cmd.get_future_world_base(offs)  # [N,K,3] [N,K,4] [N,K,3] [N,K,3]
    n, k, _ = pos_w.shape

    root_pos_w = wp.to_torch(cmd.robot.data.root_pos_w)            # [N, 3]
    root_quat_w = wp.to_torch(cmd.robot.data.root_quat_w)          # [N, 4] xyzw
    inv_root = quat_inv(root_quat_w)[:, None, :].expand(n, k, 4).reshape(n * k, 4)
    root_q = root_quat_w[:, None, :].expand(n, k, 4).reshape(n * k, 4)

    pos_err_b = quat_apply(inv_root, (pos_w - root_pos_w[:, None, :]).reshape(n * k, 3))
    ori_err_w = quat_box_minus(root_q, quat_w.reshape(n * k, 4))   # actual ⊟ reference (world)
    ori_err_b = quat_apply(inv_root, ori_err_w)                    # -> current base frame
    base_pose = torch.cat([pos_err_b, ori_err_b], dim=-1).reshape(n, k, 6)

    lin_b = quat_apply(inv_root, lin_w.reshape(n * k, 3)).reshape(n, k, 3)
    ang_b = quat_apply(inv_root, ang_w.reshape(n * k, 3)).reshape(n, k, 3)
    base_vel = torch.cat([lin_b, ang_b], dim=-1)                   # [N, K, 6]

    joint_pos, joint_vel = cmd.get_future_joint_state(offs)        # [N, K, nj] each

    # ee positions relative to the reference base AT THE SAME STEP: the world lift is
    # rigid, so differencing and unrotating recovers exactly the clip-frame posture
    # that ee_ref_relative_to_base reports at offset 0.
    frames = cmd.resolve_ee_frames(None)
    ee_w = cmd.get_future_world_frames(frames, offs)               # [N, K, F, 13]
    inv_ref_q = quat_inv(quat_w.reshape(n * k, 4))
    ee_rel = [
        quat_apply(inv_ref_q, (ee_w[:, :, i, 0:3] - pos_w).reshape(n * k, 3)).reshape(n, k, 3)
        for i in range(len(frames))
    ]
    ee_pos = torch.cat(ee_rel, dim=-1)                             # [N, K, 3F]

    out = torch.cat([base_pose, base_vel, joint_pos, joint_vel, ee_pos], dim=-1)
    return out.reshape(n, -1)


def command_window(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", num_steps: int = 11, stride: int = 5,
) -> torch.Tensor:
    """Joystick-level velocity command over the same window (``[N, num_steps * 3]``).

    The upstream ``target_vel`` ``(vx, vy, wz)`` (heading frame) at offsets
    ``[0, stride, ...]``, so index 0 is the current command and the rest is the
    look-ahead.  Deliberately a SEPARATE term from :func:`reference_window` so it
    can stay noise-free: at deploy this comes from the joystick / scripted
    profile, not from the flow-matching generator, so corrupting it would train
    the policy against noise it will never see.  Requires a clip library carrying
    ``target_vel``.  Flattened as ``[step0(3), step1(3), ...]``.
    """
    cmd = env.command_manager.get_term(command_name)
    offs = _window_offsets(cmd, num_steps, stride)
    return cmd.get_future_command_velocity(offs).reshape(env.num_envs, -1)


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


