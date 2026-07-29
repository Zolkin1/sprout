# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

from ..commands.traj_tracking.world_lift_isaac import (
    lift_position,
    yaw_from_quat,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

# TODO: Break the key parts of this into another function that can be called from the play script easily.
#   Then I can more easily log initial conditions.

def reset_on_motion_sequence(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    base_frame_name: str = "pelvis_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset each env onto a randomly chosen pre-generated motion clip.

    Picks a clip (and, if ``cfg.random_start_phase``, a random start step),
    places the clip in the world so its start-step stance anchor sits at the
    env origin (re-zeroed so a mid-clip start still spawns at the origin, not
    metres downrange), and writes the clip's reference base pose / velocity /
    joint state into the sim.  Sets the command's per-env clip tracking state
    so the first ``compute`` reproduces the spawn pose (zero initial error).

    Unlike :func:`reset_on_reference`, there is no velocity conditioner, no
    skill bucket, no contact-gate state, and every env starts on a clip.

    Args:
        env: The environment.
        env_ids: Env indices to reset.
        command_name: The :class:`MotionSequenceCommand` term name.
        base_frame_name: Trajectory base frame (must match the command cfg).
        asset_cfg: Robot asset config.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    if env_ids.numel() == 0:
        return
    k = env_ids.shape[0]
    device = env.device
    num_clips = cmd.manager.num_clips
    num_steps = cmd.manager.num_steps

    # 1. Sample a clip and a start step per env.
    clip_id = torch.randint(0, num_clips, (k,), device=device)
    if cmd.cfg.random_start_phase:
        # Leave at least random_start_min_remaining_s of clip after the start.
        margin = int(round(cmd.cfg.random_start_min_remaining_s / cmd.manager.dt))
        # Per-clip: leave >= margin frames before EACH clip's OWN real end (clip_len), not num_steps,
        # so a short clip's random start doesn't land in (or past) its padding tail.  clip_len lives on
        # the clip-library device (may be CPU via --clip_device), so index it with clip_id on THAT
        # device, then move the result to the env device (device-safe gather; cf. flat_ids below).
        clip_len = cmd.manager.clip_len
        hi = (clip_len[clip_id.to(clip_len.device)].to(device) - margin).clamp(min=1)
        start_step = (torch.rand(k, device=device) * hi.float()).long()
    else:
        start_step = torch.zeros(k, dtype=torch.long, device=device)

    # 2. World placement of the clip frame (optional random yaw): re-zero only the
    #    downrange (x, y) placement so a mid-clip start spawns the base at the env
    #    origin, but KEEP the clip's base height (z).
    world_quat = torch.zeros(k, 4, device=device)
    world_quat[:, 3] = 1.0  # identity (xyzw, w last -- Isaac Lab convention)
    if getattr(cmd.cfg, "randomize_heading", False):
        yaw = torch.empty(k, device=device).uniform_(-torch.pi, torch.pi)
        zeros = torch.zeros_like(yaw)
        world_quat = quat_from_euler_xyz(zeros, zeros, yaw)
    # Clip-world base position at the start step (clips are stored clip-world, so
    # there is no per-step anchor schedule). Rotate it by the placement heading;
    # re-zero only x/y (keep z) so the base spawns at the env origin at its
    # natural clip height, with the feet resting on the ground.
    y0, _ = cmd.manager.gather_desired(clip_id, start_step)
    base_pos_cf = y0[:, cmd._base_idx["pos"]]  # [k, 3]
    rotated = lift_position(torch.zeros(k, 3, device=device), world_quat, base_pos_cf)
    rotated[:, 2] = 0.0
    world_pos = env.scene.env_origins[env_ids] - rotated

    # 3-5. Ego re-anchor mask (clip envs are ego-anchored with probability
    #      ``ego_anchor_fraction``, else world-anchored; ego-anchored envs get their
    #      clip placement periodically snapped onto the robot root during the episode,
    #      see MotionSequenceCommand._reanchor_ego_envs -- the closed-loop FM tracker
    #      sets ego_anchor_fraction=1.0), then compute + write the reference spawn
    #      state and commit the command's per-env clip state via the shared tail.
    ego_frac = float(getattr(cmd.cfg, "ego_anchor_fraction", 0.0))
    ego_mask = torch.rand(k, device=device) < ego_frac
    _commit_clip_reset(cmd, asset, env_ids, clip_id, start_step, world_pos, world_quat, ego_mask)


def _commit_clip_reset(
    cmd,
    asset: "Articulation",
    env_ids: torch.Tensor,
    clip_id: torch.Tensor,
    start_step: torch.Tensor,
    world_pos: torch.Tensor,
    world_quat: torch.Tensor,
    ego_mask: torch.Tensor,
    snap_mask: torch.Tensor | None = None,
    mask_frame_yaw: torch.Tensor | None = None,
) -> None:
    """Shared tail of the clip resets: reference world state -> sim + command state.

    Given a per-env clip assignment (``clip_id`` / ``start_step``) and world placement
    (``world_pos`` / ``world_quat``), computes the reference base pose / velocity /
    joint state at that placement, writes it to the sim, and commits the
    :class:`MotionSequenceCommand`'s per-env clip tracking state so the first
    ``compute`` (at ``episode_length_buf == 0``) reproduces the spawn pose with zero
    initial error.

    Args:
        cmd: The :class:`MotionSequenceCommand` term.
        asset: The robot articulation.
        env_ids: Env indices being reset.
        clip_id: ``[k]`` clip index per resetting env.
        start_step: ``[k]`` start step within the clip per resetting env.
        world_pos: ``[k, 3]`` world placement of each clip's frame.
        world_quat: ``[k, 4]`` world placement heading (xyzw, w last) per env.
        ego_mask: ``[k]`` bool, which envs are ego-anchored (clip re-placed onto the
            robot root on the cadence) this episode.
        snap_mask: ``[k]`` bool, which envs emit an FM snap (re-inference) event on the
            cadence. Defaults to ``ego_mask`` -- a window is recorded exactly when the clip
            is RE-ANCHORED, on whatever axes that clip's per-step mask frees. Flat and
            terrain use identical logic; the parameter stays for callers that want to
            decouple the two, but nothing does today.
        mask_frame_yaw: ``[k]`` heading (rad) of each env's TERRAIN frame, which the anchor
            mask's downrange / lateral bits resolve in. ``None`` -> 0 (world-axis masking),
            correct for flat ground and for the axis-aligned terrain tiles built today.
    """
    if snap_mask is None:
        snap_mask = ego_mask
    state = cmd.reference_world_state_at(clip_id, start_step, world_pos, world_quat)
    nj = len(asset.joint_names)
    # xyzw: what ``write_root_pose_to_sim_index`` expects (see world_lift_isaac).
    base_pose = state[:, 0:7]          # [pos(3), quat xyzw(4)]
    base_vel = state[:, 7:13]          # [lin(3), ang(3)]
    joint_pos = state[:, 13:13 + nj]
    joint_vel = state[:, 13 + nj:13 + 2 * nj]

    cmd.clip_id[env_ids] = clip_id
    cmd.start_step[env_ids] = start_step
    cmd.clip_world_pos[env_ids] = world_pos
    cmd.clip_world_quat[env_ids] = world_quat
    # Reset-time placement = the clip's terrain-cell anchor. FIXED anchor-mask DOFs
    # pin to this on the ego re-anchor cadence (so terrain-fixed axes stay put while
    # FREE axes track the robot). For flat, world_pos.z is the env-origin ground, so
    # a FIXED-z clip keeps its natural height above the ground rather than the robot's.
    cmd._reset_world_pos[env_ids] = world_pos
    cmd._reset_world_yaw[env_ids] = yaw_from_quat(world_quat)
    cmd._ego_mask[env_ids] = ego_mask
    cmd._snap_mask[env_ids] = snap_mask
    # Terrain frame the anchor mask's downrange / lateral bits resolve in (0 = world axes).
    cmd._mask_frame_yaw[env_ids] = (
        torch.zeros_like(cmd._reset_world_yaw[env_ids]) if mask_frame_yaw is None
        else mask_frame_yaw.to(cmd._mask_frame_yaw.dtype)
    )
    # Seed the re-anchor snapshot from the pose actually written to the sim, so every segment's
    # placement resolves to the terrain-cell anchor until the first ego re-anchor fires.
    cmd.seed_reanchor_snapshot(env_ids, clip_id, start_step, base_pose)

    asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim_index(root_velocity=base_vel, env_ids=env_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def sample_terrain_clips(env_tid: torch.Tensor, groups: dict[int, torch.Tensor]) -> torch.Tensor:
    """Sample, per env, a random clip whose ``terrain_id`` matches the env's tile.

    Pure-tensor helper (no sim), shared by :func:`reset_on_terrain_motion_sequence`
    and its unit test.

    Args:
        env_tid: ``[k]`` long, the terrain id of each env's tile (``-1`` = flat).
        groups: ``{terrain_id: clip_indices}`` (from
            :func:`...clips.clip_io.group_clips_by_terrain`); each value is a 1-D long
            tensor of clip indices for that terrain.

    Returns:
        ``[k]`` long clip index per env, on ``env_tid``'s device.

    Raises:
        ValueError: if any env's terrain id has no clips -- that tile can never be
            assigned a matching clip (a fatal library / terrain mismatch).
    """
    device = env_tid.device
    missing = sorted(
        {int(t) for t in env_tid.tolist() if int(t) not in groups or groups[int(t)].numel() == 0}
    )
    if missing:
        raise ValueError(
            f"terrain id(s) {missing} have an env tile but NO clips in the library -- "
            "envs on those tiles cannot be assigned a matching clip. Regenerate the "
            "terrain clip library so every terrain has clips."
        )
    clip_id = torch.empty(env_tid.shape[0], dtype=torch.long, device=device)
    for t in torch.unique(env_tid).tolist():
        mask = env_tid == t
        g = groups[int(t)].to(device)
        pick = torch.randint(0, int(g.numel()), (int(mask.sum()),), device=device)
        clip_id[mask] = g[pick]
    return clip_id


def reset_on_terrain_motion_sequence(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    base_frame_name: str = "pelvis_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset each env onto a clip whose terrain matches the env's terrain tile.

    Terrain-aware counterpart to :func:`reset_on_motion_sequence`.  The scene terrain
    is a :class:`ClipTerrainImporter` built from the same clip library the command
    tracks, so each env is pinned to a tile with a known ``terrain_id`` and tile
    geometry.  For each resetting env this:

    * samples a clip uniformly from the clips whose per-clip ``terrain_id`` matches
      the env's tile (flat tiles -> flat clips, ``terrain_id == -1``);
    * places the clip on the tile with
      :func:`...terrains.clip_terrain_loader.clip_frame_origin` (identity heading --
      clips are baked axis-aligned with the staircase), so the reference feet ride the
      treads.  Wandering flat clips are re-centered on their flat tile (their bbox is
      not centered on the clip origin);
    * restricts ego re-anchoring to flat clips: re-anchoring snaps the clip onto the
      robot root, which would decouple a terrain clip from the physical staircase, so
      terrain / transition clips are always world-anchored.

    Args:
        env: The environment (``env.scene.terrain`` must be a ClipTerrainImporter).
        env_ids: Env indices to reset.
        command_name: The :class:`MotionSequenceCommand` term name.
        base_frame_name: Trajectory base frame (must match the command cfg).
        asset_cfg: Robot asset config.
    """
    from motion_gen.clips.clip_io import (
        group_clips_by_terrain,
    )
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
        clip_frame_origin,
    )

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    if env_ids.numel() == 0:
        return
    k = env_ids.shape[0]
    device = env.device
    importer = env.scene.terrain
    if not hasattr(importer, "env_terrain_id"):
        raise TypeError(
            "reset_on_terrain_motion_sequence requires the scene terrain to be a "
            "ClipTerrainImporter (exposing env_terrain_id / env_cell_*). Use "
            "reset_on_motion_sequence for the flat-ground teacher."
        )
    num_steps = cmd.manager.num_steps

    # Cache the terrain -> clip-indices grouping on the command (built once, on device).
    groups = getattr(cmd, "_terrain_groups", None)
    if groups is None:
        # One-time single-source-of-truth check. The env cfg cannot do this (its __post_init__
        # runs BEFORE the --clip override), and a mismatch is silent-but-fatal: the tiles carry
        # one library's terrain ids while the clips carry another's.
        importer_path = getattr(importer.cfg, "clip_library_path", None)
        if importer_path != cmd.cfg.path:
            raise ValueError(
                "terrain clip library mismatch: the ClipTerrainImporter built its mesh from "
                f"'{importer_path}' but the tracking command reads '{cmd.cfg.path}'. Both must "
                "be the SAME file (tile terrain ids come from the mesh, clip terrain ids from "
                "the library). Pass one --clip <lib.h5>, which sets both."
            )
        groups = {t: idx.to(device) for t, idx in group_clips_by_terrain(cmd.manager.lib).items()}
        cmd._terrain_groups = groups

    # 1. Per-env terrain id -> sample a matching clip + a start step.
    env_tid = importer.env_terrain_id[env_ids].to(torch.long)
    clip_id = sample_terrain_clips(env_tid, groups)
    if cmd.cfg.random_start_phase:
        margin = int(round(cmd.cfg.random_start_min_remaining_s / cmd.manager.dt))
        # Per-clip: leave >= margin frames before EACH clip's OWN real end (clip_len), not num_steps,
        # so a short clip's random start doesn't land in (or past) its padding tail.  clip_len lives on
        # the clip-library device (may be CPU via --clip_device), so index it with clip_id on THAT
        # device, then move the result to the env device (device-safe gather; cf. flat_ids below).
        clip_len = cmd.manager.clip_len
        hi = (clip_len[clip_id.to(clip_len.device)].to(device) - margin).clamp(min=1)
        start_step = (torch.rand(k, device=device) * hi.float()).long()
    else:
        start_step = torch.zeros(k, dtype=torch.long, device=device)

    # 2. Place each clip on its tile (identity heading -- clips are baked axis-aligned
    #    with the staircase, so alignment is a pure translation). Terrain / transition
    #    clips are baked in the tile-LOCAL corner frame, so they go straight on the
    #    corner; flat clips land there too and are re-centered by bbox below.
    world_quat = torch.zeros(k, 4, device=device)
    world_quat[:, 3] = 1.0  # identity (xyzw, w last)
    is_flat = env_tid == -1
    world_pos = clip_frame_origin(
        importer.env_cell_corner[env_ids],
        importer.env_clip_x_offset[env_ids],
    )

    # Flat clips WANDER (turning walks up to tens of metres) and their bbox is not
    # centered on the clip origin, so anchoring that origin at the tile corner would
    # overhang the flat patch (sized to the max flat bbox). Center each flat clip's
    # base-frame bbox on its tile instead so the whole path stays on the tile.
    if bool(is_flat.any()):
        bidx = cmd._base_idx["pos"]
        bx, by = int(bidx[0]), int(bidx[1])
        y_des = cmd.manager.lib.y_des
        flat_ids = clip_id[is_flat].to(y_des.device)
        pel_xy = y_des[flat_ids][:, :, [bx, by]].to(device)             # [m, T, 2]
        bbox_center = 0.5 * (pel_xy.amax(dim=1) + pel_xy.amin(dim=1))    # [m, 2]
        cell_center_flat = importer.env_cell_center[env_ids][is_flat]
        world_pos[is_flat, 0] = cell_center_flat[:, 0] - bbox_center[:, 0]
        world_pos[is_flat, 1] = cell_center_flat[:, 1] - bbox_center[:, 1]

    # 3-5. Ego re-anchor (clip re-placement) applies to ALL envs -- the re-placement is
    #      per-DOF MASKED (see MotionSequenceCommand._reanchor_clip_ids), so a terrain env
    #      re-anchors only its FREE axes onto the robot while FIXED axes stay pinned to the
    #      terrain cell: a stair clip snaps only y (keeping x/z/yaw on the treads), a box
    #      clip follows its per-step mask (y-free-pinned approach/jump, ego recovery). So
    #      terrain no longer detaches, and every env is partial-ego with probability
    #      ego_frac.
    #
    #      Snap logic is IDENTICAL to flat: ``snap_mask`` defaults to ``ego_mask``, so an FM
    #      training window is emitted exactly when the clip is actually RE-ANCHORED, on
    #      whatever axes that clip's mask frees.  (Before the per-DOF mask existed, terrain
    #      clips could only be world-pinned or fully detached, so this passed
    #      ``ego_mask | ~is_flat`` to force a window on every non-flat env -- otherwise the
    #      FM saw no terrain at all.  With masks, a non-ego terrain env is genuinely NOT
    #      re-anchored, and recording it hands the FM accumulated tracking drift instead of a
    #      placement decision.)  Then compute + write the spawn.
    ego_frac = float(getattr(cmd.cfg, "ego_anchor_fraction", 0.0))
    ego_mask = torch.rand(k, device=device) < ego_frac
    # The tile's AUTHORED terrain frame: the anchor mask's downrange / lateral bits resolve in it, so
    # "free lateral" is across the staircase / box whatever heading the terrain carries -- rather than
    # world y, which silently became the traversal axis for a family baked on a rotated heading.
    frame_yaw = importer.env_terrain_yaw[env_ids].to(device)
    _commit_clip_reset(cmd, asset, env_ids, clip_id, start_step, world_pos, world_quat, ego_mask,
                       mask_frame_yaw=frame_yaw)
