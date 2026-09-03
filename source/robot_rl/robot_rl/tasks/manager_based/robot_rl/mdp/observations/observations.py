# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_conjugate,
    quat_inv,
    quat_mul,
    yaw_quat,
)
from isaaclab.utils.noise.noise_cfg import NoiseCfg

from robot_rl.flow_matching.transforms import quat_to_rot6d
from robot_rl.flow_matching.window import (
    REFERENCE_BLOCK_KEYS as _REFERENCE_BLOCK_KEYS,
    REFERENCE_WINDOW_BLOCK_NAMES as REFERENCE_WINDOW_BLOCKS,
    REFERENCE_WINDOW_NOISE,
    expand_window_blocks,
    resolve_reference_blocks,
    window_blocks,
    window_to_world,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def tracking_error_local(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """The per-channel tracking error the CLF actually grades (``[N, 2 * n_tracked]``).

    Exactly the CLF's ``eta``: position error (``quat_box_minus`` on the orientation
    channels) interleaved with velocity error, over the TRACKED channel subset only.
    Measured in the anchored frame, so the clip's free root DOFs have already been
    re-datumed onto the robot and this is local error, not accumulated drift.

    Privileged (critic-only): it is a function of the reward's own arguments, so it hands
    the value function the quantity it has to predict instead of making it re-derive that
    from a reference and a measurement separately.  Replaces the old ``ref_traj`` /
    ``act_traj`` pair, and is strictly more informative than either.
    """
    cmd = env.command_manager.get_term(command_name)
    clf = cmd.clf
    if getattr(clf, "_subset", False):
        y_act, y_des = cmd.y_act[:, clf.pos_gather], cmd.y_des[:, clf.pos_gather]
        dy_act, dy_des = cmd.dy_act[:, clf.vel_gather], cmd.dy_des[:, clf.vel_gather]
    else:
        y_act, y_des, dy_act, dy_des = cmd.y_act, cmd.y_des, cmd.dy_act, cmd.dy_des
    y_err = clf.compute_y_err(y_act, y_des)
    eta = torch.zeros(y_err.shape[0], 2 * y_err.shape[1], device=y_err.device)
    eta[:, 0::2] = y_err
    eta[:, 1::2] = dy_act - dy_des
    return eta


def tracking_error_global(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Base pose error against the GLOBALLY-placed clip (``[N, 6]``).

    Position error (3) plus the ``quat_box_minus`` rotation vector (3), both in the world
    frame, measured against the clip on its reset-time terrain cell -- the same reference
    the pose-drift termination uses.

    Both halves are ``ref - actual``, matching :func:`base_pose_error` and the
    ``reference_window`` -- see
    :data:`robot_rl.flow_matching.window.BASE_POSE_CONVENTION`.  Kept as a log-map
    3-vector rather than widened to the window's 6D form: this is privileged, never
    exported, and global drift is bounded well inside the branch cut by the pose-drift
    termination that grades the same datum.

    This is the drift :func:`tracking_error_local` deliberately cannot see: the reward
    forgives planar drift, but the episode still ends on it, so without this the critic
    would be predicting a termination it has no way to anticipate.
    """
    cmd = env.command_manager.get_term(command_name)
    state = cmd.termination_reference_world_state()
    root_pos_w = wp.to_torch(cmd.robot.data.root_pos_w)
    root_quat_w = wp.to_torch(cmd.robot.data.root_quat_w)
    return torch.cat([
        state[:, 0:3] - root_pos_w,
        quat_box_minus(state[:, 3:7], root_quat_w),   # reference - actual
    ], dim=-1)


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
    """The reference base pose in the robot's current YAW (heading) frame.

    Returns ``[N, 9]`` = position ``ref - actual`` (3) rotated into the robot's heading frame,
    plus ``R_yawT R_ref`` in the 6D rotation representation (6).  The anchor is yaw-only, so
    the robot's own roll and pitch never enter; the REFERENCE's tilt is still carried in full.
    Fully heading-invariant -- no absolute world heading leaks -- and deployable sim2real.

    Identical to node 0 of :func:`reference_window`'s ``base_pose`` block, and computed
    through the same :func:`~robot_rl.flow_matching.window.world_to_window` convention --
    ``test_reference_window`` asserts the two agree, which is what keeps them from drifting
    apart.  See :data:`robot_rl.flow_matching.window.BASE_POSE_CONVENTION`.

    No observation group carries this term any more (``reference_window`` replaced it, and
    node 0 IS this).  It survives as that test's independent oracle and as a diagnostic
    signal in ``data_logger``, so it has to move with the window rather than be left behind
    at the old convention.
    """
    cmd = env.command_manager.get_term(command_name)
    state = cmd.reference_world_state()                 # [N, 3+4+3+3+2nj]
    ref_pos_w = state[:, 0:3]
    ref_quat_w = state[:, 3:7]                          # xyzw
    root_pos_w = wp.to_torch(cmd.robot.data.root_pos_w)
    root_quat_w = wp.to_torch(cmd.robot.data.root_quat_w)  # xyzw

    root_yaw = yaw_quat(root_quat_w)
    return torch.cat([
        quat_apply_inverse(root_yaw, ref_pos_w - root_pos_w),
        quat_to_rot6d(quat_mul(quat_conjugate(root_yaw), ref_quat_w)),   # R_yawT R_ref
    ], dim=-1)


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
    # Yaw-only, matching the reference window's base_vel block, which this term is the
    # current-step oracle for: the anchor supplies heading, never the robot's roll/pitch.
    inv_root = quat_inv(yaw_quat(wp.to_torch(cmd.robot.data.root_quat_w)))
    return torch.cat([quat_apply(inv_root, lin_w), quat_apply(inv_root, ang_w)], dim=-1)


def command_velocity(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Joystick-level base velocity command that generated the clip (``target_vel``).

    The upstream commanded velocity ``(vx, vy, wz)`` -- the signal a velocity
    "joystick" would supply -- rather than the achieved reference base velocity
    (:func:`ref_base_velocity`, ``dy_des``).  Heading-invariant (planar linear
    velocity + yaw rate in the base/heading frame) and available at deployment.
    Returns ``[N, 3]``.  Requires a clip library carrying ``target_vel``.

    NOTE this used to pair with the ``command_lin_vel_xy_tracking_exp`` /
    ``command_ang_vel_z_tracking_exp`` rewards, which no task registers any more -- the
    teacher's CLF already grades the base twist.  The command is still worth OBSERVING
    (it is what the reference was generated from, and the deployment joystick supplies
    it); it is simply no longer separately rewarded.
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.get_command_velocity().clone()


# ----------------------------------------------------------------------
# Merged reference window (current step + look-ahead).
#
# One term carries the WHOLE reference trajectory -- base pose/velocity, joint
# angles, joint velocities and end-effector positions -- at an explicit list of
# clip-step ``offsets``, the first of which is always the CURRENT step (0).
# ``offsets=(0,)`` therefore degenerates to the current step alone.
#
# The offsets are a LIST rather than a (count, stride) pair because the window is
# deliberately non-uniform: it samples the two steps immediately after the current
# one AND a 0.1 s-spaced tail, so no single stride describes it.  See
# ``g1_teacher_env_cfg.REFERENCE_OFFSETS``.
#
# The proprioceptive history is a separate mechanism (per-term
# ``ObservationTermCfg.history_length``, one entry per control step); only the
# FUTURE is decimated.
# ----------------------------------------------------------------------

def strided_offsets(num_steps: int, stride: int) -> tuple[int, ...]:
    """Uniformly strided window offsets ``(0, stride, ..., (num_steps - 1) * stride)``.

    A builder for the strided TAIL of a window's offset list, so it does not have to be
    typed out.  At the 0.02 s control dt, ``strided_offsets(11, 5)`` spans 0.0 .. 1.0 s.

    Args:
        num_steps: Samples to emit, the leading ``0`` included.  Clamped to ``>= 1``.
        stride: Control steps between samples.  Clamped to ``>= 1``.

    Returns:
        ``num_steps`` ascending offsets, starting at 0.
    """
    return tuple(i * max(1, int(stride)) for i in range(max(1, int(num_steps))))


def _window_offsets(cmd, offsets: Sequence[int]) -> torch.Tensor:
    """Validate a window offset list and put it on the command's device.

    Args:
        cmd: The trajectory command term (supplies the device).
        offsets: Clip-step offsets, strictly ascending, starting at 0.

    Returns:
        ``[len(offsets)]`` long tensor.

    Raises:
        ValueError: If empty, if the first entry is not 0, or if the list is not strictly
            ascending.  Three consumers depend on exactly those properties and none of
            them would fail visibly: ``mdp.clip_end_reached`` reserves ``max(offsets)``
            steps of look-ahead, ``FmTrackingCommand`` checks the generator's horizon
            against the same maximum, and node 0 is both the step the CLF grades and the
            one that enters the generator's autoregressive history.
    """
    offs = tuple(int(o) for o in offsets)
    if not offs:
        raise ValueError("a reference window needs at least one offset, e.g. (0,)")
    if offs[0] != 0:
        raise ValueError(
            f"reference-window offsets must start at the current step (0), got {offs[0]}. "
            "Node 0 is what the CLF grades and what enters the generator's history."
        )
    if any(b <= a for a, b in zip(offs, offs[1:])):
        raise ValueError(f"reference-window offsets must be strictly ascending, got {offs}")
    return torch.tensor(offs, device=cmd.device, dtype=torch.long)


def reference_window_layout_from_dims(
    num_joints: int, num_ee_frames: int, blocks=None
) -> list[tuple[str, int]]:
    """Per-step block layout of :func:`reference_window` from raw dimensions.

    Kept as the name this side of the codebase reaches for (``export_parameters``,
    ``data_logger``, the noise/scale expansion below), but the layout itself lives in
    :func:`robot_rl.flow_matching.window.window_blocks` -- the flow-matching generator
    predicts in exactly this representation and honours the same block SELECTION, so the
    definition has to be shared rather than restated on each side.

    Args:
        num_joints: Robot joint count.
        num_ee_frames: End-effector frames the clip library exposes.
        blocks: Subset to include; defaults to all of them.

    Returns:
        ``[(block_name, width), ...]`` in packing order.
    """
    return window_blocks(num_joints, num_ee_frames, blocks)


def reference_window_layout(cmd, blocks=None) -> list[tuple[str, int]]:
    """Per-step block layout of :func:`reference_window` as ``[(name, width), ...]``.

    Resolves the robot's joint count and the clip library's ee-frame count off the command
    term, then defers to :func:`reference_window_layout_from_dims`.  Used by the diagnostics
    (``scripts/rsl_rl/data_logger.py``) to slice the flat term back into named blocks
    without restating the layout.

    Args:
        cmd: The trajectory command term.
        blocks: Subset to include; defaults to all of them.
    """
    return reference_window_layout_from_dims(
        len(cmd.robot.joint_names), len(cmd.resolve_ee_frames(None)), blocks
    )


def reference_window_selection(
    env: ManagerBasedRLEnv, groups: Sequence[str] | None = None
) -> tuple[str, ...] | None:
    """The block selection this environment's :func:`reference_window` term packs.

    The ONE producer of the teacher's selection, for the same reason
    :func:`~robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth.depth_camera_specs`
    is the one producer of the camera contract: the generator has to predict exactly the
    blocks the policy consumes, and both the recording (which sets the generator's training
    width) and the exported tracker parameters (which the deploy checks the generator
    against) have to state the same thing. A subset that lands in one and not the others is
    invisible -- every array is still the right shape for whoever built it.

    Args:
        env: The environment instance.
        groups: Observation groups to look in, or ``None`` for all of them.

    Returns:
        The selection in canonical packing order, or ``None`` when no group observes a
        ``reference_window`` term.

    Raises:
        ValueError: If two groups observe the term with DIFFERENT selections. They are
            fed by one command term and one layout; disagreeing means one of them is
            reading a window that is not the one being built.
    """
    obs_mgr = env.observation_manager
    wanted = list(groups) if groups is not None else list(obs_mgr.active_terms.keys())
    found: dict[str, tuple[str, ...]] = {}
    for group in wanted:
        names = list(obs_mgr.active_terms.get(group, []))
        if "reference_window" not in names:
            continue
        cfg = obs_mgr._group_obs_term_cfgs[group][names.index("reference_window")]
        found[group] = resolve_reference_blocks((cfg.params or {}).get("blocks"))
    if not found:
        return None
    selections = set(found.values())
    if len(selections) > 1:
        raise ValueError(
            f"the 'reference_window' term is configured with different block selections "
            f"per group: { {g: list(b) for g, b in found.items()} }. One command term "
            "builds one window; the groups must agree on which blocks it packs."
        )
    return next(iter(selections))


# ``REFERENCE_WINDOW_NOISE`` is defined in ``robot_rl.flow_matching.window`` and imported
# above: the FM's autoregressive history noise reads the SAME table, because both model
# the generator's prediction error. Re-exported here so the env cfgs keep importing it
# from mdp. See that module for why forking it would be a bug.


#: Per-block multipliers applied to :func:`reference_window` AFTER noise and clipping.
#:
#: CURRENTLY A NO-OP, and deliberately kept anyway.  The only non-unity entry is the
#: joint-velocity 0.05, and ``joint_vel`` is not in the live block selection
#: (``g1_teacher_env_cfg.REFERENCE_WINDOW_BLOCKS`` = base_pose, base_vel, joint_pos), so
#: :func:`reference_window_scale` expands to all ones -- :func:`expand_window_blocks`
#: only builds the SELECTED blocks.  The table survives because it is the restore path:
#: set ``REFERENCE_WINDOW_BLOCKS = None`` and the 0.05 is live again, immediately.  Do not
#: prune the unselected keys for tidiness; the expansion would then ``KeyError`` on that
#: restore.
#:
#: Why the 0.05 exists, for when it comes back: raw reference joint velocities run to
#: ~6.5 rad/s while every other block in the window is O(1), so unscaled they would be
#: ~10x hotter than their neighbours and 20x hotter than the *same physical quantity*
#: entering the trunk through proprioception (which applies the same 0.05) -- a pure
#: conditioning problem for the (unnormalized) encoder, since the noise is applied before
#: the scale and so the signal-to-noise ratio is untouched.
#:
#: DEFERRED (no compute to spend on it now, not a dead end).  Both items below were
#: measured with the FULL five-block window; dropping ``joint_vel`` and ``ee_pos`` changed
#: each of them, in opposite directions:
#:
#: 1. Conditioning -- now WEAK, and it was mostly a joint_vel problem.  These numbers are
#:    hand-picked, not normalization; the measured p99 after scaling was base_pose 0.68,
#:    base_vel 1.14, joint_pos 1.61, joint_vel 0.32, ee_pos 0.74, i.e. a ~5x spread.  Over
#:    the three LIVE blocks that is 0.68 / 1.14 / 1.61 -- 2.4x, all O(1).
#:    ``agent.actor.obs_normalization=true`` would handle it properly
#:    (``RefEncoderMLPModel`` normalizes the encoder branch separately), but it costs a run
#:    and there is much less left to win.
#: 2. Reconstruction-loss weighting -- now STEEPER, but no longer coupled to anything.  An
#:    MSE weights each channel by its magnitude squared, so unit magnitudes rather than
#:    intent decide what "reconstruct the reference" means.  Measured share of that loss
#:    with all five blocks -- before this scale: joint_vel 92%, everything else 8%
#:    combined; after: joint_pos 62%, base_vel 18%, ee_pos 13%, base_pose 4%, joint_vel 3%.
#:    Renormalizing the post-scale figures over the live blocks ESTIMATES joint_pos ~74%,
#:    base_vel ~21%, base_pose ~5% -- so the objective is now three-quarters "reconstruct
#:    the joint angles".  Treat that as an estimate, not a measurement: the encoder's
#:    per-block error will have shifted now that its input is much narrower.  Fix it with a
#:    per-block weight vector inside ``RefEncoderMLPModel.reconstruction_loss``.  That used
#:    to be forced (the scale also feeds the actor's forward pass, so the two could not be
#:    tuned independently through it); with the scale all ones the constraint is vacuous and
#:    a weight vector in the loss is a free-standing change.
#:
#: When it IS worth spending a run on: (1) is a one-flag A/B
#: (``agent.actor.obs_normalization=true``, and the critic's) and (2) is dormant while the
#: reconstruction loss is off (``reconstruction=False``, ``coeff=0.0``), so the two can be
#: taken one at a time.  The cheap moment for (1) is a round that already breaks baseline
#: comparability for other reasons -- it needs its own run either way, or it cannot be told
#: apart from whatever else moved.
REFERENCE_WINDOW_SCALE: dict[str, float | list[float]] = {
    "base_pos": 1.0,
    "base_ori": 1.0,
    "base_lin_vel": 1.0,
    "base_ang_vel": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,   # matches the proprioceptive joint_vel term; UNSELECTED, see above
    "ee_pos": 1.0,
}


def _expand_reference_blocks(
    num_steps: int,
    num_joints: int,
    num_ee_frames: int,
    per_block: dict[str, float | list[float]],
    blocks: Sequence[str] | None = None,
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
    return expand_window_blocks(num_joints, num_ee_frames, per_block, blocks) * int(num_steps)


def reference_window_noise_vector(
    num_steps: int,
    num_joints: int,
    num_ee_frames: int,
    magnitudes: dict[str, float | list[float]] | None = None,
    blocks: Sequence[str] | None = None,
) -> list[float]:
    """Expand per-block noise magnitudes to one half-width per :func:`reference_window` channel.

    These are RAW-unit magnitudes: Isaac Lab applies noise before clipping and scaling, so
    a joint-velocity half-width is in rad/s regardless of
    :data:`REFERENCE_WINDOW_SCALE`.

    Returns a plain list (not a tensor) so it stays a Hydra/OmegaConf primitive -- see
    :class:`ReferenceWindowNoiseCfg`.

    Args:
        num_steps: Samples in the window (``len(REFERENCE_OFFSETS)``).
        num_joints: Robot joint count.
        num_ee_frames: End-effector frames the clip library exposes.
        magnitudes: Per-block half-widths; defaults to :data:`REFERENCE_WINDOW_NOISE`.

    Returns:
        ``num_steps * D`` half-widths, in the channel order the window packs.
    """
    mags = REFERENCE_WINDOW_NOISE if magnitudes is None else magnitudes
    return _expand_reference_blocks(num_steps, num_joints, num_ee_frames, mags, blocks)


def reference_window_scale(
    num_steps: int,
    num_joints: int,
    num_ee_frames: int,
    factors: dict[str, float | list[float]] | None = None,
    blocks: Sequence[str] | None = None,
) -> tuple[float, ...]:
    """Per-channel multiplier for a :func:`reference_window` observation term.

    Returns a TUPLE because ``ObservationTermCfg.scale`` only accepts ``float``, ``int``
    or ``tuple`` -- a list is rejected outright by the observation manager.

    Args:
        num_steps: Samples in the window (``len(REFERENCE_OFFSETS)``).
        num_joints: Robot joint count.
        num_ee_frames: End-effector frames the clip library exposes.
        factors: Per-block multipliers; defaults to :data:`REFERENCE_WINDOW_SCALE`.

    Returns:
        ``num_steps * D`` multipliers, in the channel order the window packs.
    """
    facs = REFERENCE_WINDOW_SCALE if factors is None else factors
    return tuple(_expand_reference_blocks(num_steps, num_joints, num_ee_frames, facs, blocks))


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
        blocks: Sequence[str] | None = None,
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
            n_half=reference_window_noise_vector(num_steps, num_joints, num_ee_frames, mags, blocks),
            magnitudes=mags,
        )


def reference_window(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", *,
    offsets: Sequence[int], blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """The full reference trajectory over a look-ahead window (``[N, len(offsets) * D]``).

    Merges what used to be five separate current-step terms and two root-only
    look-ahead terms into ONE observation.  Per step, in
    :func:`reference_window_layout` order:

    * ``base_pose`` (9) -- the reference base pose in the robot's *current* YAW
      frame: position ``ref - actual`` (3) plus ``R_yawT R_ref`` as a 6D rotation
      (6).  Yaw-only, so the robot's tilt does not rotate the frame.  Matches
      :func:`base_pose_error`; see
      :data:`robot_rl.flow_matching.window.BASE_POSE_CONVENTION`.
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

    The window is built by the COMMAND (``reference_window_obs``), not here: the
    flow-matching generator predicts this representation directly, so its command term
    overrides that accessor and returns its prediction rather than round-tripping it
    through the world.  This function only resolves the offsets and flattens.

    Note the window saturates on the clip's final padded frame near the end of a
    clip (``_future_step_indices`` clamps to the padded library length); the
    episode is truncated by ``clip_end_reached`` before that matters.

    Args:
        env: The environment instance.
        command_name: The trajectory command term to read.
        offsets: Clip-step offsets to sample, ascending from 0 (see
            :func:`strided_offsets` and ``g1_teacher_env_cfg.REFERENCE_OFFSETS``).
            Deliberately has no default: the FM's horizon, the episode truncation margin
            and the exported deploy contract are all derived from it, so a silent
            fallback would put a different window in one of them.
        blocks: Layout blocks to pack; defaults to all of them.

    Returns:
        ``[N, len(offsets) * D]``.

    TODO(perf): the observation manager is the single largest phase of the env step --
    50.8 ms/frame, ~39%, tied with physics (profiled with the trained depth_v3 at 1024
    envs via ``play.py --profile``).  Roughly 19 ms of that is the two depth cameras
    (measured by A/B against ``G1-teacher-clf-terrain-body``, which is otherwise
    identical), leaving ~31 ms across this window, the critic's height scan and the
    tracking-error terms.  This term is the prime suspect for the remainder: it gathers
    ``len(offsets)`` future frames of base + joints + ee for every env every step, then
    does several quaternion ops on an ``[N*K, .]`` flattening.  Nothing here is
    instrumented individually, so the first move is a per-term breakdown rather than an
    optimisation -- the manager total is all we currently know.
    """
    cmd = env.command_manager.get_term(command_name)
    offs = _window_offsets(cmd, offsets)
    win = cmd.reference_window_obs(offs, blocks)    # [N, K, D]
    return win.reshape(win.shape[0], -1)


def reference_window_to_world(
    window: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    num_steps: int,
    num_joints: int,
    num_ee_frames: int,
    blocks: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Exact inverse of :func:`reference_window`: put the observation back in the world.

    The window is entirely base-relative by design -- that is what keeps world coordinates
    away from the policy -- which also makes it impossible to eyeball.  Given the robot's
    measured base pose at the same step (the frame the forward direction divided out), this
    reconstructs the world-frame reference the window encodes, so a viewer can draw *what
    the policy actually saw* next to what the generator predicted.  A frame bug in
    :func:`reference_window` then shows up as a visible offset or twist between the two,
    rather than as a plausible-looking vector of numbers.

    Feed the CLEAN window (as ``data_logger`` computes it), not the corrupted observation:
    noise, ``clip`` and ``scale`` are applied by the observation manager AFTER this layout
    and are not inverted here.

    Args:
        window: ``[N, num_steps * D]`` reference window, packed as
            :func:`reference_window` emits it.
        root_pos_w: ``[N, 3]`` measured base position at the same step.
        root_quat_w: ``[N, 4]`` measured base orientation (xyzw) at the same step.
        num_steps: Samples in the window.
        num_joints: Robot joint count.
        num_ee_frames: End-effector frames the window carries.
        blocks: The selection the window was packed with (the observation term's own
            ``blocks``); defaults to all of them.  Pass the term's, or this both mis-sizes
            the window and reports blocks it does not contain.

    Returns:
        Mapping with whichever of ``base_pos_w`` ``[N, K, 3]``, ``base_quat_w``
        ``[N, K, 4]`` (xyzw), ``base_lin_vel_w`` / ``base_ang_vel_w`` ``[N, K, 3]``,
        ``joint_pos`` / ``joint_vel`` ``[N, K, nj]`` (robot joint order, frame-invariant
        so passed through) and ``ee_pos_w`` ``[N, K, E, 3]`` the selection supports.  A
        dropped block is ABSENT rather than zero-filled -- a zero here would be drawn as a
        reference standing at the origin.

    Raises:
        ValueError: If ``window``'s width disagrees with the layout implied by the
            dimensions.

    Note:
        ``base_quat_w`` comes back through ``quat_box_plus``, which normalises the sign
        (``quat_unique``), so it may be the negation of the quaternion that went in.  That
        is the same rotation; compare rotations, not raw components.
    """
    layout = reference_window_layout_from_dims(num_joints, num_ee_frames, blocks)
    per_step = sum(width for _, width in layout)
    k = max(1, int(num_steps))
    if window.shape[-1] != k * per_step:
        raise ValueError(
            f"reference_window_to_world got a {window.shape[-1]}-wide window but the layout "
            f"implies {k} x {per_step} = {k * per_step} (num_steps={num_steps}, "
            f"nj={num_joints}, ee={num_ee_frames}, "
            f"blocks={[n_ for n_, _ in layout]})."
        )
    n = window.shape[0]
    return window_to_world(
        window.reshape(n, k, per_step),
        root_pos_w[:, None, :],
        root_quat_w[:, None, :],
        num_joints,
        num_ee_frames,
        blocks,
    )


def fm_reference_window(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", *,
    offsets: Sequence[int], blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """:func:`reference_window`, served from the GENERATOR while everything else stays on the clip.

    This is the RL fine-tuning seam.  The teacher trains against the clip -- reward, critic
    and terminations all grade it -- but at deploy the policy is handed the generator's
    prediction, and the two differ by more than the synthetic reference noise models.  This
    term closes that gap from the observation side ALONE: the actor sees what it will see on
    the robot, and nothing about what it is graded against moves.

    That separation is what makes the fine-tune well-posed, and it is bought by ORDERING
    rather than by a second copy of the reference.  ``FmTrackingCommand._update_command``
    builds both references every step, applies the configured one, and computes the CLF
    (``v`` / ``vdot``) there and then -- storing scalars.  ``clf_reward`` reads those stored
    scalars, so a swap performed later, at observation time, cannot reach the reward however
    long it lasts.  :meth:`~.fm_tracking_cmd.FmTrackingCommand.as_fm_reference` restores the
    configured source on the way out, so the swap does not leak into the next term either.

    Pair it with ``reference_source="clip"`` (so the graded reference IS the clip) and a
    non-empty ``fm_checkpoint`` (so ``fm_enabled`` is true and the generator runs passively
    beside it).  ``history_from`` then chooses the arm: ``"clip"`` is teacher forcing, which
    keeps the prediction phase-locked to the motion the reward grades; ``"generator"`` is the
    closed loop, where the two can drift apart and the reward stops describing what the
    policy was shown.

    Deliberately NOT applied to :func:`command_window`: that term is the joystick command,
    which at deploy comes from the operator rather than the generator, and it is noise-free
    for the same reason.

    Args:
        env: The environment.
        command_name: The tracking command term; must be an ``FmTrackingCommand``.
        offsets: Look-ahead offsets, as :func:`reference_window` takes them.
        blocks: Reference blocks, as :func:`reference_window` takes them.

    Returns:
        ``[N, len(offsets) * D]`` -- the same layout the clip-served term produces, so a
        checkpoint moves between the two without a shape change.

    Raises:
        RuntimeError: If the command has no generator loaded (``fm_checkpoint`` empty).
    """
    cmd = env.command_manager.get_term(command_name)
    with cmd.as_fm_reference():
        return reference_window(env, command_name, offsets=offsets, blocks=blocks)


def fm_reference_residual(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", *,
    offsets: Sequence[int], blocks: Sequence[str] | None = None,
) -> torch.Tensor:
    """The generator's window MINUS the clip's, for the privileged critic.

    The fine-tune breaks an invariant the critic was built on.  Its ``reference_window`` term
    is documented as "the same reference window the actor's encoder sees, uncorrupted" -- but
    once the actor is served by the generator and the critic is not, the value function can no
    longer see the input the policy is acting on.  It then cannot tell a policy error from the
    generator having asked for something wrong, and that difference is pure variance in the
    value target.

    This closes it from the critic's side.  The critic keeps its clip window and gains the
    residual, which is exactly the information it was missing: what the actor was told, versus
    what it will be graded against.

    **Residual rather than the raw generator window, on purpose.**  The critic's first layer is
    linear and already receives the clip window, so ``(clip, residual)`` and ``(clip, fm)`` span
    the same space -- the choice is one of conditioning, not information.  The residual is
    centred near zero and is small wherever the generator is right, which is where a zero-init
    column wants to start.  It is also the quantity with the meaning: "how wrong is the
    reference this step".

    Privileged, and it must stay that way -- it is built from the clip, which no robot has.

    Args:
        env: The environment.
        command_name: The tracking command term; must be an ``FmTrackingCommand``.
        offsets: Look-ahead offsets, as :func:`reference_window` takes them.
        blocks: Reference blocks, as :func:`reference_window` takes them.

    Returns:
        ``[N, len(offsets) * D]``, the same width as either window.

    Raises:
        RuntimeError: If the command has no generator loaded.
    """
    cmd = env.command_manager.get_term(command_name)
    clip = reference_window(env, command_name, offsets=offsets, blocks=blocks)
    with cmd.as_fm_reference():
        served = reference_window(env, command_name, offsets=offsets, blocks=blocks)
    return served - clip


def command_window(
    env: ManagerBasedRLEnv, command_name: str = "traj_ref", *, offsets: Sequence[int],
) -> torch.Tensor:
    """Joystick-level velocity command over the same window (``[N, len(offsets) * 3]``).

    The upstream ``target_vel`` ``(vx, vy, wz)`` (heading frame) at ``offsets``, so index 0
    is the current command and the rest is the look-ahead.  Deliberately a SEPARATE term
    from :func:`reference_window` so it can stay noise-free: at deploy this comes from the
    joystick / scripted profile, not from the flow-matching generator, so corrupting it
    would train the policy against noise it will never see.  Requires a clip library
    carrying ``target_vel``.  Flattened as ``[step0(3), step1(3), ...]``.

    Takes the SAME ``offsets`` as :func:`reference_window`, and must: the deploy runtime
    refuses to load a tracker whose two window specs disagree
    (``deploy.gen_tracker._resolve_window_offsets``), since one flat observation is
    assembled from both.

    Args:
        env: The environment instance.
        command_name: The trajectory command term to read.
        offsets: Clip-step offsets, ascending from 0.  No default, for the reason given in
            :func:`reference_window`.

    Returns:
        ``[N, len(offsets) * 3]``.
    """
    cmd = env.command_manager.get_term(command_name)
    offs = _window_offsets(cmd, offsets)
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




##
# The adjuster: what it corrects, what it was conditioned on, and where the goal is
##
def fm_plan_conditioning(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """The generator's conditioning for the live plan, ``[N, cond_width]``.

    History tokens, exteroception tokens and command tokens, flattened and concatenated in
    that order, already normalized -- the form ``encode_prefix`` takes.

    Carried as an OBSERVATION rather than recomputed by the policy because PPO's rollout runs
    under ``inference_mode``: nothing computed there can enter an autograd graph later, so the
    update re-runs the actor from stored observations. This is what it re-runs it from.

    Read from the command's stash, not rebuilt here. Rebuilding would sample a different
    instant -- observations are computed after the interval's physics -- and the exteroception
    half would be a different lag of a different frame, so the borrowed prefix would not be the
    one the plan came from.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs, cond_width]``.
    """
    return env.command_manager.get_term(command_name).plan_conditioning()


def fm_plan_window(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """The live plan and its final-step flow field, ``[N, 2 * F * D]``, STANDARDIZED.

    The two halves are the adjuster's per-node input: what the generator predicted at that
    node, and which way the flow was still moving there at the last Euler step -- near
    ``tau = 1`` a statement about how far the drawn sample sits from its conditional mode.

    Standardized, so the two halves are commensurate and both O(1): the window is normalized
    by the generator's own state normalizer (which is DECLARED, not fitted, so the number means
    the same thing in every run) and the flow field already lives in that space.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs, 2 * F * D]`` -- the window first, then the field.
    """
    cmd = env.command_manager.get_term(command_name)
    window = cmd.ctx.state_norm.normalize(cmd.plan_window())
    flow = cmd.plan_flow()
    return torch.cat([window.flatten(1), flow.flatten(1)], dim=-1)


def window_residual(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """The residual applied to the live plan, ``[N, F * D]``, in normalizer SIGMAS.

    A CRITIC input, and not an optional one. Its group already carries the reference, which
    says how wrong the served plan is; this says how much of that the adjuster caused. Without
    the second the critic cannot separate "the generator was wrong" from "the adjuster made it
    worse" and eats the difference as variance -- which is measured to turn advantages into
    noise, inflate entropy and stall a run for a thousand iterations.

    Sigmas rather than raw units for the same reason the residual reward uses them: one node
    mixes metres, dimensionless rot6d components and radians.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs, F * D]``.
    """
    cmd = env.command_manager.get_term(command_name)
    sigma = cmd.ctx.state_norm.std.to(cmd.plan_residual().device)
    return (cmd.plan_residual() / sigma).flatten(1)


def goal_geometry(env: ManagerBasedRLEnv, command_name: str = "traj_ref",
                  slack_s: float = 0.0, step_scale: float = 1.0) -> torch.Tensor:
    """Where the goal is and how long is left, ``[N, 4]``: downrange, lateral, height, steps.

    A CRITIC input, and what makes a bootstrapped truncation well posed. On this task reaching
    the goal is a truncation (so its value IS bootstrapped) while running out of time is a
    termination (so it is not) -- and that asymmetry is the whole mechanism for rewarding
    arrival. A critic that cannot see how much time is left is being asked to predict a cut it
    has no evidence for, which is the classic time-limit bootstrapping error. The fourth
    channel is precisely that evidence.

    The first three are the robot's displacement from the goal resolved in the TERRAIN's frame
    -- the same quantities :func:`mdp.in_goal` thresholds, so the critic sees the geometry the
    termination decides on rather than a proxy for it.

    ``slack_s`` MUST match the ``clip_end_reached`` term's, because that is the deadline this
    channel exists to expose. With the default 0.0 against a task running 3.0 s of slack, the
    count reaches zero 150 control steps -- 12 agent decisions -- before the episode is
    actually cut, so the last third of a short episode carries no time evidence at all and the
    cut is exactly as unpredictable as having no channel.

    ``step_scale`` exists because the critic group is un-normalized by design (every other
    term in it is O(1) by construction: velocities scaled, actions and scan clipped). A raw
    control-step count reaches ~950 and would dominate the first layer's pre-activations by
    two to three orders of magnitude. Scaling by ``1 / max_episode_length`` keeps the unit
    honest -- it is still a step count, just expressed as a fraction of the longest episode.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        slack_s: Grace period past the clip's last usable step, in seconds. Must equal the
            ``clip_end_reached`` term's ``slack_s``.
        step_scale: Multiplier on the remaining-steps channel.

    Returns:
        ``[num_envs, 4]``. Steps remaining is clamped at 0, is in units of control steps
        (times ``step_scale``) rather than seconds so it does not silently change meaning with
        ``step_dt``, and counts to the SAME deadline ``clip_end_reached`` fires on.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region

    cmd = env.command_manager.get_term(command_name)
    root_pos_w = wp.to_torch(env.scene["robot"].data.root_pos_w)
    goal_pos_w, frame_yaw = goal_region.goal_anchor_world(cmd)
    down, lat, hz = goal_region.goal_frame_offsets(root_pos_w, goal_pos_w, frame_yaw)
    cursor = (cmd.start_step + env.episode_length_buf).to(root_pos_w.device)
    deadline = goal_region.last_usable_step(cmd).to(root_pos_w.device)
    if slack_s:
        deadline = deadline + int(round(float(slack_s) / env.step_dt))
    # The episode also ends at `time_out`, and whichever deadline is nearer is the one the
    # critic has to predict.
    time_out_left = int(env.max_episode_length) - env.episode_length_buf.to(root_pos_w.device)
    remaining = torch.minimum(deadline - cursor, time_out_left).clamp_min(0)
    return torch.stack([down, lat, hz, remaining.to(down.dtype) * float(step_scale)], dim=-1)
