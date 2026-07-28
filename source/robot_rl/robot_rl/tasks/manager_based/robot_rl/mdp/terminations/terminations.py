# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_box_minus

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def base_pose_error_termination(
    env,
    command_name: str = "traj_ref",
    max_pos_dist: float = 0.25,
    max_ori_rad: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base pose drifts too far from the tracking reference.

    The single failure termination for open-loop motion-sequence tracking
    (replaces the pelvis-upright / base-height / frame-drift terminations).
    Compares the robot's world base pose against the reference base pose served
    by :meth:`MotionSequenceCommand.termination_reference_world_state` -- the
    tracked reference for clip- and open-loop-FM envs, but the **clip** reference
    for closed-loop-FM envs (so a robot that stalls while the clip marches on
    terminates instead of riding a stationary FM prediction):

      * position: terminate if ``||ref_pos - root_pos|| > max_pos_dist`` (m).
      * orientation: terminate if the geodesic angle between the reference
        and actual base quaternions exceeds ``max_ori_rad`` (rad).

    Args:
        env: IsaacLab environment.
        command_name: The :class:`MotionSequenceCommand` term name.
        max_pos_dist: Max base position error before terminating (m).
        max_ori_rad: Max base orientation error before terminating (rad).
        asset_cfg: Robot scene entity.

    Returns:
        Boolean ``[num_envs]`` -- True for envs that should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    state = cmd.termination_reference_world_state()
    ref_pos_w = state[:, 0:3]
    ref_quat_w = state[:, 3:7]                      # wxyz
    root_pos_w = wp.to_torch(asset.data.root_pos_w)
    root_quat_w = wp.to_torch(asset.data.root_quat_w)  # wxyz

    pos_err = torch.norm(ref_pos_w - root_pos_w, dim=-1)
    # ||quat_box_minus|| is the geodesic rotation angle (rad).
    ori_err = torch.norm(quat_box_minus(root_quat_w, ref_quat_w), dim=-1)
    return (pos_err > max_pos_dist) | (ori_err > max_ori_rad)


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


