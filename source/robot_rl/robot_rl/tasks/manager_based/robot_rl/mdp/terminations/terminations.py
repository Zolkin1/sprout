# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_box_minus, wrap_to_pi

from motion_gen.world_lift import anchor_mask_from_free

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


#: Anchor-mask bit for the terrain-frame traversal axis. Computed, not hand-typed, so it
#: stays in lock-step with the bit order in ``motion_gen.world_lift``.
_MASK_DOWNRANGE_FREE: int = anchor_mask_from_free({"downrange"})


def base_pose_error_termination(
    env,
    command_name: str = "traj_ref",
    max_pos_dist: float = 0.25,
    max_pos_dist_downrange_free: float | None = None,
    max_ori_rad: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base pose drifts too far from the tracking reference.

    The single failure termination for open-loop motion-sequence tracking
    (replaces the pelvis-upright / base-height / frame-drift terminations).
    Compares the robot's world base pose against the reference base pose served
    by :meth:`MotionSequenceCommand.termination_reference_world_state` -- the
    **globally-placed** clip, deliberately NOT the per-step re-anchored reference
    the reward grades against.  The reward forgives planar drift (the clip's free
    root DOFs are re-datumed onto the robot every step); this does not, so a robot
    that stalls while the clip marches on still terminates:

      * position: terminate if ``||ref_pos - root_pos|| > max_pos_dist`` (m).
      * orientation: terminate if the geodesic angle between the reference
        and actual base quaternions exceeds ``max_ori_rad`` (rad).

    **The position bound is anchor-mask dependent**, because how much drift matters -- and
    whether the policy can even perceive it -- is exactly what the mask encodes:

    * ``downrange`` PINNED (stairs, boxes): the anchored reference coincides with the global
      one along the traversal axis, so the CLF already grades that drift AND the reference
      window shows it to the policy as a nonzero base error.  Observable and correctable, so
      hold it tight -- position relative to the treads is the whole task there.
    * ``downrange`` FREE (flat): the clip is re-datumed onto the robot every step, so the
      base error the policy observes is ~0 by construction and global drift is INVISIBLE to
      it.  Terminating tightly on something unobservable just truncates episodes without
      teaching anything, and on flat ground absolute world position is arbitrary anyway --
      the motion's shape and speed are what matter.  So give it slack.

    Box clips carry a PER-FRAME mask (ego approach -> pinned mount), so the bound follows the
    phase within a single episode rather than being fixed per tile.

    Known imprecision: this switches a single scalar radius on one bit, while the error it
    bounds is the full 3-D norm.  On a stair tile lateral IS free, so lateral drift still
    counts against the tight bound.  Splitting the error into free / pinned components in the
    terrain frame would fix that; it needs the ``mask_frame_yaw`` rotation and is deliberately
    not done here.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        max_pos_dist: Max base position error where downrange is PINNED (m).
        max_pos_dist_downrange_free: Max base position error where downrange is FREE (m).
            ``None`` uses ``max_pos_dist`` everywhere, i.e. the old mask-blind behaviour.
        max_ori_rad: Max base orientation error before terminating (rad).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    state = cmd.termination_reference_world_state()
    ref_pos_w = state[:, 0:3]
    ref_quat_w = state[:, 3:7]                      # xyzw (Isaac Lab 3.0 convention)
    root_pos_w = wp.to_torch(asset.data.root_pos_w)
    root_quat_w = wp.to_torch(asset.data.root_quat_w)  # xyzw

    pos_err = torch.norm(ref_pos_w - root_pos_w, dim=-1)
    # ||quat_box_minus|| is the geodesic rotation angle (rad).
    ori_err = torch.norm(quat_box_minus(root_quat_w, ref_quat_w), dim=-1)

    if max_pos_dist_downrange_free is None:
        limit = max_pos_dist
    else:
        mask = cmd.manager.gather_anchor_mask_at_step(cmd.clip_id, cmd._step).to(pos_err.device)
        limit = torch.where(
            (mask & _MASK_DOWNRANGE_FREE) > 0,
            torch.as_tensor(max_pos_dist_downrange_free, device=pos_err.device,
                            dtype=pos_err.dtype),
            torch.as_tensor(max_pos_dist, device=pos_err.device, dtype=pos_err.dtype),
        )
    return (pos_err > limit) | (ori_err > max_ori_rad)


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


def clip_end_reached(env, command_name: str = "traj_ref", reserve_lookahead: bool = True) -> torch.Tensor:
    """Truncate (good termination) when the clip's last USABLE step is reached.

    Wire this as ``DoneTerm(..., time_out=True)`` so RSL-RL bootstraps the
    value from the final state rather than treating clip-end as a failure.
    Keyed on the clip step counter so it is correct even when episodes start
    mid-clip (``random_start_phase``).

    With ``reserve_lookahead`` the episode ends one full ``reference_window`` **before**
    the clip's last frame, rather than at it. The look-ahead gather clamps to the library
    length (``_future_step_indices``), so without the margin the final
    ``(num_steps - 1) * stride`` = 50 steps of every clip feed the policy a window that is
    progressively frozen on the last frame -- and then teach the generator to predict that
    "the reference stops moving". Reserving the margin costs the tail 1 s of every clip and
    means no observed sample is ever a clamp artefact. The margin is read off the live
    ``reference_window`` term, so it cannot drift from the observation.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        reserve_lookahead: End the episode a full observation window early (default). Set
            False to run to the clip's true last frame and accept the saturated tail.

    Returns:
        Boolean ``[num_envs]`` -- True for envs at/after the clip's last usable step.
    """
    cmd = env.command_manager.get_term(command_name)
    # Each clip's OWN last real frame (clip_len - 1), so a short clip padded into a longer library
    # truncates at its real end instead of playing the held padding tail to num_steps - 1.  clip_len
    # lives on the clip-library device (may be CPU via --clip_device), so index it with clip_id on
    # THAT device before moving to the env device (device-safe gather).
    clip_len = cmd.manager.clip_len
    last_step = clip_len[cmd.clip_id.to(clip_len.device)].to(cmd.start_step.device) - 1
    if reserve_lookahead:
        offsets = cmd.obs_window_offsets()
        if offsets is not None and offsets.numel() > 0:
            # clamp_min(0): a clip shorter than the window has no unsaturated step at all,
            # so it ends immediately rather than silently serving a frozen reference.
            last_step = (last_step - int(offsets.max().item())).clamp_min(0)
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


