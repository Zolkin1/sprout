# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_box_minus,
    wrap_to_pi,
)

from motion_gen.world_lift import anchor_mask_from_free

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


#: Anchor-mask bits, computed rather than hand-typed so they stay in lock-step with the bit
#: order in ``motion_gen.world_lift``.  Bit SET = that DOF is FREE.
_MASK_DOWNRANGE = anchor_mask_from_free({"downrange"})
_MASK_LATERAL = anchor_mask_from_free({"lateral"})
_MASK_HEIGHT = anchor_mask_from_free({"height"})
_MASK_ROLL_PITCH = anchor_mask_from_free({"roll", "pitch"})
_MASK_YAW = anchor_mask_from_free({"yaw"})


def base_pose_error_termination(
    env,
    command_name: str = "traj_ref",
    max_pos_dist: float = 0.25,
    max_ori_rad: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base drifts too far from the reference, on the PINNED axes only.

    Compares the robot's world base pose against
    :meth:`MotionSequenceCommand.termination_reference_world_state` -- the **globally-placed**
    clip, deliberately NOT the per-step re-anchored reference the reward grades against.

    Only the anchor mask's PINNED DOFs are measured.  A FREE DOF is one the clip was slid
    along to sit on the robot, so error there is zero by construction and invisible to the
    policy: the reference window shows it a base error of ~0 in exactly those directions.
    Bounding something unobservable only truncates episodes.  What remains is the part of the
    pose the clip genuinely pins down, which is both graded by the CLF and visible:

    ==========================  ====================================================
    mask                        measured
    ==========================  ====================================================
    flat (downrange/lat/yaw     height, roll, pitch  -- i.e. "don't sink, stay upright"
    free)
    stairs / boxes (lateral     downrange, height, roll, pitch, yaw
    free)
    ==========================  ====================================================

    On flat that is exactly a height + orientation check, which is all SONIC terminates on.
    On terrain it additionally holds position along the treads and heading.  Because the free
    axes are no longer measured, ONE tight ``max_pos_dist`` works everywhere -- there is no
    longer a reason to inflate the bound on flat to stop it firing on drift nobody grades.

    Frames: the translational bits are named in the TERRAIN's frame, so the horizontal error
    is rotated by ``-mask_frame_yaw`` before being split into downrange / lateral.  Height is
    gravity-aligned and needs no rotation.  Orientation splits into TILT (roll+pitch together,
    as the angle between the two up-axes -- frame-free) and YAW (the heading difference).
    Roll and pitch share one test because every mask in use pins both.

    Box clips carry a PER-FRAME mask (ego approach -> pinned mount), so which axes are
    measured follows the phase within a single episode.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        max_pos_dist: Max position error over the PINNED translational axes (m).
        max_ori_rad: Max tilt, and max yaw error where yaw is pinned (rad).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    state = cmd.termination_reference_world_state()
    ref_pos_w, ref_quat_w = state[:, 0:3], state[:, 3:7]   # xyzw (Isaac Lab 3.0)
    root_pos_w = wp.to_torch(asset.data.root_pos_w)
    root_quat_w = wp.to_torch(asset.data.root_quat_w)      # xyzw

    mask = cmd.manager.gather_anchor_mask_at_step(cmd.clip_id, cmd._step).to(root_pos_w.device)
    pinned = lambda bit: (mask & bit) == 0                                       # noqa: E731

    # --- position, in the TERRAIN frame -------------------------------------------------
    d = ref_pos_w - root_pos_w
    yaw_f = cmd._mask_frame_yaw.to(d.device)
    cos_y, sin_y = torch.cos(yaw_f), torch.sin(yaw_f)
    downrange = cos_y * d[:, 0] + sin_y * d[:, 1]
    lateral = -sin_y * d[:, 0] + cos_y * d[:, 1]
    comps = torch.stack([downrange, lateral, d[:, 2]], dim=-1)
    keep = torch.stack([pinned(_MASK_DOWNRANGE), pinned(_MASK_LATERAL),
                        pinned(_MASK_HEIGHT)], dim=-1)
    pos_err = torch.norm(comps * keep.to(comps.dtype), dim=-1)

    # --- orientation: tilt (roll+pitch) and yaw, tested separately ----------------------
    _, _, ref_yaw = euler_xyz_from_quat(ref_quat_w, wrap_to_2pi=False)
    _, _, act_yaw = euler_xyz_from_quat(root_quat_w, wrap_to_2pi=False)
    yaw_err = wrap_to_pi(ref_yaw - act_yaw).abs()
    # Tilt = angle between the two body-z axes. Independent of yaw, so the two tests do not
    # double-count, and gravity-aligned so no terrain-frame rotation is needed.
    up = torch.tensor([0.0, 0.0, 1.0], device=d.device).expand(d.shape[0], 3)
    cos_tilt = (quat_apply(ref_quat_w, up) * quat_apply(root_quat_w, up)).sum(dim=-1)
    tilt_err = torch.acos(cos_tilt.clamp(-1.0, 1.0))

    over_tilt = (tilt_err > max_ori_rad) & pinned(_MASK_ROLL_PITCH)
    over_yaw = (yaw_err > max_ori_rad) & pinned(_MASK_YAW)
    return (pos_err > max_pos_dist) | over_tilt | over_yaw


def tracked_frame_error_termination(
    env,
    command_name: str = "traj_ref",
    frame_names: tuple[str, ...] = (
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    ),
    max_pos_dist: float = 0.25,
    max_ori_rad: float | Sequence[float | None] = 0.8,
) -> torch.Tensor:
    """Terminate when a tracked end-effector drifts too far from ITS reference.

    Complements :func:`base_pose_error_termination`, which watches the base against the
    globally-placed clip. This one is LOCAL: it reads ``y_act`` / ``y_des``, so the clip
    root has already been re-datumed onto the robot and what remains is limb error. The two
    catch different failures -- a robot can hold its base pose perfectly while dragging a
    foot through the wrong place, and it can track every limb while walking off the clip.

    Terminates if ANY named frame exceeds either bound.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        frame_names: Frames to watch. Defaults to the four end effectors.
        max_pos_dist: Max per-frame position error before terminating (m).
        max_ori_rad: Max per-frame geodesic orientation error before terminating (rad).
            Either one value for every frame, or one PER frame in ``frame_names`` order,
            where ``None`` disables the orientation bound for that frame (its POSITION bound
            still applies).  Per-frame exists because the wrists and the ankles are not
            comparable: profiled at 2.6k iterations, orientation drove 492 of 629 ee trips
            and ``left_wrist_yaw`` alone was 305 of them, of which 286 were on TERRAIN and
            only 4 on flat.  A wrist rides a longer kinematic chain than an ankle and its yaw
            is not pinned by ground contact, so one shared bound made this mostly a
            left-wrist-on-terrain detector.  Raising the bound does not fix that -- 0.8 ->
            1.0 rad moved it 305 -> 290 trips, so the error is far past the bound rather than
            marginally over it, and the honest options are to disable it or to fix whatever
            the terrain clips are asking of that wrist.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.

    Raises:
        ValueError: If a per-frame ``max_ori_rad`` does not match ``frame_names``.
    """
    cmd = env.command_manager.get_term(command_name)
    act_pos, act_quat, des_pos, des_quat = cmd.tracked_frame_pose(list(frame_names))
    n, f = act_pos.shape[0], act_pos.shape[1]
    pos_err = torch.norm(act_pos - des_pos, dim=-1)                                  # [N, F]
    ori_err = torch.norm(
        quat_box_minus(act_quat.reshape(n * f, 4), des_quat.reshape(n * f, 4)), dim=-1
    ).view(n, f)

    if isinstance(max_ori_rad, (int, float)):
        ori_lim = torch.full((f,), float(max_ori_rad), device=ori_err.device,
                             dtype=ori_err.dtype)
    else:
        vals = [float("inf") if v is None else float(v) for v in max_ori_rad]
        ori_lim = torch.as_tensor(vals, device=ori_err.device, dtype=ori_err.dtype)
        if ori_lim.numel() != f:
            raise ValueError(
                f"max_ori_rad has {ori_lim.numel()} entries but frame_names has {f}: "
                f"{list(frame_names)}"
            )
    return ((pos_err > max_pos_dist) | (ori_err > ori_lim)).any(dim=-1)


def clip_end_reached(env, command_name: str = "traj_ref") -> torch.Tensor:
    """Truncate (good termination) when the clip's last step is reached.

    Wire this as ``DoneTerm(..., time_out=True)`` so RSL-RL bootstraps the
    value from the final state rather than treating clip-end as a failure.
    Keyed on the clip step counter so it is correct even when episodes start
    mid-clip (``random_start_phase``).

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.

    Returns:
        Boolean ``[num_envs]`` -- True for envs at/after the clip's last step.
    """
    cmd = env.command_manager.get_term(command_name)
    # Each clip's OWN last real frame (clip_len - 1), so a short clip padded into a longer library
    # truncates at its real end instead of playing the held padding tail to num_steps - 1.  clip_len
    # lives on the clip-library device (may be CPU via --clip_device), so index it with clip_id on
    # THAT device before moving to the env device (device-safe gather).
    clip_len = cmd.manager.clip_len
    last_step = clip_len[cmd.clip_id.to(clip_len.device)].to(cmd.start_step.device) - 1
    return (cmd.start_step + env.episode_length_buf) >= last_step

def base_orientation_from_upright(
    env,
    roll_limit_deg: float = 70.0,
    pitch_limit_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the base roll or pitch deviates from world upright by more than the limits.

    The reference is the world-frame upright (identity rotation), so YAW IS IGNORED: this is a
    face-plant / fallen detector that does not depend on the tracking reference, and so is not
    subject to whatever grace gating :func:`base_pose_error_termination` uses.

    Restored 2026-07-29: this was deleted in ``a754728`` (the online reference-engine removal)
    while :func:`base_fallen` kept calling it, leaving that termination -- and therefore the
    whole ``G1-tracker-clf-fm-*`` task -- raising ``NameError`` on its first step. Implementation
    is the pre-deletion one verbatim, so termination behaviour is unchanged from when it worked.

    Args:
        env: IsaacLab environment.
        roll_limit_deg: Maximum allowed |roll| from upright in degrees.
        pitch_limit_deg: Maximum allowed |pitch| from upright in degrees.
        asset_cfg: Robot scene entity.

    Returns:
        Boolean tensor of shape ``[num_envs]`` -- ``True`` for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    act_quat = wp.to_torch(asset.data.root_quat_w)  # (x, y, z, w)
    roll, pitch, _ = euler_xyz_from_quat(act_quat, wrap_to_2pi=False)
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)

    roll_limit = torch.deg2rad(torch.tensor(roll_limit_deg, device=act_quat.device))
    pitch_limit = torch.deg2rad(torch.tensor(pitch_limit_deg, device=act_quat.device))
    return (roll.abs() > roll_limit) | (pitch.abs() > pitch_limit)


def base_fallen(
    env,
    min_height: float = 0.5,
    roll_limit_deg: float = 70.0,
    pitch_limit_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base has fallen: too low, or tilted too far from upright.

    A trajectory-independent fall detector: fires when the base world height
    drops below ``min_height`` (m) **or** the base roll/pitch deviates from
    world upright (world z) by more than ``roll_limit_deg`` / ``pitch_limit_deg``
    (yaw ignored).  The orientation half reuses
    :func:`base_orientation_from_upright`, so the two share one definition.

    Independent of the tracking reference (unlike
    :func:`base_pose_error_termination`), so it catches a collapsed / face-
    planted robot regardless of where the reference is.  On flat ground the
    base world ``z`` is the height above the ground plane.

    Args:
        env: IsaacLab environment.
        min_height: Minimum base world height before terminating (m).
        roll_limit_deg: Maximum allowed |roll| from upright (deg).
        pitch_limit_deg: Maximum allowed |pitch| from upright (deg).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    too_low = wp.to_torch(asset.data.root_pos_w)[:, 2] < min_height
    tilted = base_orientation_from_upright(env, roll_limit_deg, pitch_limit_deg, asset_cfg)
    return too_low | tilted


