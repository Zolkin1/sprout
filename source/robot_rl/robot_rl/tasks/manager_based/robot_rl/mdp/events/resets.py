# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_inv

from motion_gen.world_lift import (
    lift_position,
    _yaw_from_quat,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

# TODO: Break the key parts of this into another function that can be called from the play script easily.
#   Then I can more easily log initial conditions.

def reset_on_reference(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        command_name: str,
        conditioner_command_name: str,
        base_frame_name: str,
        base_z_offset: float = 0.03,    # TODO: Need to handle this better, I think the blow up issue is related to this not being correct
        joint_add_range: tuple[float, float] = (0.0, 0.0),
        rel_envs_on_ref: float = 0.5,
        special_val: float = 1.0,
        rel_envs_on_special: float = 0.4,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset the robot to a random point along the reference trajectory.

    This event samples a random time from the trajectory, extracts the base frame pose
    and joint positions at that time, and resets the robot to that state. The time offset
    is stored in the command so it knows the current phase.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to reset.
        command_name: Name of the trajectory command term.
        base_frame_name: Name of the body frame to use for root pose (must exist in trajectory outputs).
        joint_scale_range: Tuple of (min_scale, max_scale) for uniform random scaling of joint positions.
            Default (1.0, 1.0) means no scaling.
        rel_envs_on_ref: Float giving the relative amount of envs to start on the reference trajectory.
        asset_cfg: Configuration for the robot asset.

    Raises:
        ValueError: If base_frame_name is not found in trajectory outputs.
        ValueError: If any robot joint is missing from the trajectory outputs.
    """
    # Get the robot asset and trajectory command
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    cond_term = env.command_manager.get_term(conditioner_command_name)
    # Reset the velocity command for the new episode: samples a bucket
    # from the spawn cell (not the stale ``root_pos_w``), commits live
    # immediately, snaps ``vel_target_b`` past the max-acc ramp, and
    # refreshes ``time_left``.
    cond_term.reset_for_episode(env_ids)
    cond_term._update_command()
    # Snap the trajectory cmd's active ``skill_id`` to whichever bucket
    # the freshly-set ``vel_target_b`` lands in for these envs, and
    # clear any stale pending / cross-fade state.  Must run AFTER the
    # velocity cmd's reset so ``vel_target_b`` is current.
    if hasattr(cmd, "reset_for_episode"):
        cmd.reset_for_episode(env_ids)
    num_env = len(env_ids)

    if num_env == 0:
        return

    r = torch.empty(len(env_ids), device=env.device)

    ref_env = torch.zeros(num_env, device=env.device)
    ref_env = r.uniform_(0.0, 1.0) <= (rel_envs_on_ref)
    ref_ids = env_ids[ref_env]
    num_ref_envs = len(ref_ids)

    # Temporarily adjust the commands to be in the special range
    r_on_ref = torch.empty(num_ref_envs, device=env.device)
    special_envs = r_on_ref.uniform_(0.0, 1.0) < rel_envs_on_special
    command = env.command_manager.get_term(conditioner_command_name).command
    command_clone = command.clone()
    command[ref_ids[special_envs], 0] = special_val * torch.ones(len(ref_ids[special_envs]), device=env.device)

    nonref_env = ref_env == False
    nonref_ids = env_ids[nonref_env]
    num_nonref_envs = len(nonref_ids)


    # Validate base frame exists in trajectory outputs
    pos_indices = _find_output_indices(cmd.ordered_pos_output_names, base_frame_name, "pos_")
    ori_indices = _find_output_indices(cmd.ordered_pos_output_names, base_frame_name, "ori_")

    if len(pos_indices) != 3:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have pos_x, pos_y, pos_z in trajectory outputs. "
            f"Found {len(pos_indices)} position outputs."
        )
    if len(ori_indices) != 4:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have ori_x, ori_y, ori_z, ori_w in trajectory outputs. "
            f"Found {len(ori_indices)} orientation outputs."
        )

    # TODO: Can get rid of this to speed things up
    # Validate all robot joints are in trajectory outputs
    traj_joint_names = set()
    for name in cmd.ordered_pos_output_names:
        if name.startswith("joint:"):
            traj_joint_names.add(name.split(":", 1)[1])

    missing_joints = []
    for joint_name in asset.joint_names:
        if joint_name not in traj_joint_names:
            missing_joints.append(joint_name)

    if missing_joints:
        raise ValueError(
            f"The following robot joints are missing from the trajectory outputs: {missing_joints}"
        )

    # Sample a phase per ref env.  The manager caches its trajectory
    # assignments across steps and only invalidates inside
    # ``BatchedMultiSkillCommand._pre_update_phase``.  Reset events fire
    # between steps, so by default the cache here holds the *previous*
    # step's traj_idx — built from the pre-resample conditioner.
    # Invalidate so the read below picks the trajectory matching the
    # freshly-resampled conditioner (incl. any ``special_envs`` override
    # above).
    cmd.manager.invalidate_cache()
    ref_traj_idx = cmd.manager.get_current_trajectory_indices()[ref_ids]
    random_phase_ref = torch.rand(num_ref_envs, device=env.device)

    # Get trajectory outputs at sampled phase
    cmd.get_desired_outputs(random_phase_ref, env_ids=ref_ids)
    des_outputs = cmd.y_des
    y_sampled = des_outputs[ref_ids]  # Position outputs

    # Extract base frame position and orientation from outputs
    base_pos_rel = y_sampled[:, pos_indices]  # Shape: [num_env, 3]
    base_ori_quat_w = y_sampled[:, ori_indices]  # Shape: [num_env, 4] - quaternion (x, y, z, w)

    # Per-env spawn offset = stair-origin offset (per-trajectory, top-level YAML)
    # + ref-frame offset (per-domain). Spline outputs are ref-frame-relative, so
    # both the spawn pose and the stored ref pose must shift by the same amount —
    # keeps ``robot_body - ref_frame == trajectory_value`` after the reset.
    # Assumes the stair origin lives at the env origin (= 0 in stair-local coords).
    domain_idx = cmd.manager.get_current_domains(random_phase_ref, env_ids=ref_ids)
    ref_offset = cmd.manager.data["ref_frame_offset"][ref_traj_idx, domain_idx]
    stair_offset = cmd.manager.data["origin_relative_to_stair_center"][ref_traj_idx]
    spawn_offset = ref_offset + stair_offset

    # Compute world-frame base pose
    base_pos_w = base_pos_rel + env.scene.env_origins[ref_ids] + spawn_offset

    # Build pose tensor: [x, y, z, qx, qy, qz, qw]
    base_pose = torch.cat([base_pos_w, base_ori_quat_w], dim=-1)

    # Extract Base Vel
    des_doutputs = cmd.dy_des
    dy_sampled = des_doutputs[ref_ids]  # Velocity outputs (not used for now)

    lin_vel_indices = _find_output_indices(cmd.ordered_vel_output_names, base_frame_name, "pos_")
    ang_vel_indices = _find_output_indices(cmd.ordered_vel_output_names, base_frame_name, "ori_")


    # Set base velocity
    base_vel = torch.cat([dy_sampled[:, lin_vel_indices], dy_sampled[:, ang_vel_indices]], dim=-1) #torch.zeros(num_env, 6, device=env.device)


    # Set the reference frame at env_origin + spawn_offset (identity quaternion).
    cmd.ref_poses[ref_ids, :3] = env.scene.env_origins[ref_ids] + spawn_offset
    cmd.ref_poses[ref_ids, 3:6] = 0.0
    cmd.ref_poses[ref_ids, 6] = 1.0

    # Extract joint positions from trajectory output
    # Build a mapping from robot joint indices to trajectory output indices
    joint_pos = torch.zeros(num_ref_envs, len(asset.joint_names), device=env.device)
    joint_vel = torch.zeros_like(joint_pos)

    for i, joint_name in enumerate(asset.joint_names):
        traj_output_name = f"joint:{joint_name}"
        pos_traj_idx = cmd.ordered_pos_output_names.index(traj_output_name)
        vel_traj_idx = cmd.ordered_vel_output_names.index(traj_output_name)

        joint_pos[:, i] = y_sampled[:, pos_traj_idx]
        joint_vel[:, i] = dy_sampled[:, vel_traj_idx]

    # Apply optional uniform scaling to joint positions
    # TODO: Can put back for the on reference reset
    # min_scale, max_scale = joint_scale_range
    # if min_scale != 1.0 or max_scale != 1.0:
    #     scale_factors = torch.rand(num_ref_envs, 1, device=env.device) * (max_scale - min_scale) + min_scale
    #     joint_pos = joint_pos * scale_factors
    #     joint_vel = joint_vel * scale_factors

    # Anchor the trajectory clock so the next step evaluates from the
    # sampled state.
    cmd.manager.set_phase(random_phase_ref, ref_ids)

    # Write states to simulation
    asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=ref_ids)
    # NOTE: known discrepancy — the x component of the reset base velocity
    # has been observed in play to not match the trajectory reference (y
    # and z line up).  Root cause unknown; left as-is until investigated.
    asset.write_root_link_velocity_to_sim_index(root_velocity=base_vel, env_ids=ref_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=ref_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=ref_ids)

    # Restore command
    env.command_manager.get_term(conditioner_command_name).command[:] = command_clone

    if num_nonref_envs != 0:
        # Non reference resets
        root_states = wp.to_torch(asset.data.default_root_state)[nonref_ids].clone()

        base_pose = root_states[:, 0:7]
        base_pose[:, :3] += env.scene.env_origins[nonref_ids]
        base_vel = root_states[:, 7:]

        # get default joint state
        joint_pos = wp.to_torch(asset.data.default_joint_pos)[nonref_ids, asset_cfg.joint_ids].clone()
        joint_vel = wp.to_torch(asset.data.default_joint_vel)[nonref_ids, asset_cfg.joint_ids].clone()

        # Add noise
        r = torch.empty(num_nonref_envs, len(asset.joint_names), device=env.device)
        r.uniform_(joint_add_range[0], joint_add_range[1])
        joint_pos += r

        asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=nonref_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=base_vel, env_ids=nonref_ids)
        asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=nonref_ids)
        asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=nonref_ids)

        # Anchor the non-ref envs' clock at phase 0 (start of cycle) so the
        # trajectory eval roughly matches the default-pose spawn.
        nonref_phase = torch.zeros(num_nonref_envs, device=env.device)
        cmd.manager.set_phase(nonref_phase, nonref_ids)



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
        hi = max(1, num_steps - margin)
        start_step = torch.randint(0, hi, (k,), device=device)
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
            cadence. Defaults to ``ego_mask`` (flat: only ego envs snap). On terrain the
            caller passes ``ego_mask | ~is_flat`` so world-anchored stair/transition envs
            also snap (giving the FM a stair training window) without being re-placed.
    """
    if snap_mask is None:
        snap_mask = ego_mask
    state = cmd.reference_world_state_at(clip_id, start_step, world_pos, world_quat)
    nj = len(asset.joint_names)
    base_pose = state[:, 0:7]          # [pos(3), quat wxyz(4)]
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
    cmd._reset_world_yaw[env_ids] = _yaw_from_quat(world_quat)
    cmd._ego_mask[env_ids] = ego_mask
    cmd._snap_mask[env_ids] = snap_mask

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
        groups = {t: idx.to(device) for t, idx in group_clips_by_terrain(cmd.manager.lib).items()}
        cmd._terrain_groups = groups

    # 1. Per-env terrain id -> sample a matching clip + a start step.
    env_tid = importer.env_terrain_id[env_ids].to(torch.long)
    clip_id = sample_terrain_clips(env_tid, groups)
    if cmd.cfg.random_start_phase:
        margin = int(round(cmd.cfg.random_start_min_remaining_s / cmd.manager.dt))
        hi = max(1, num_steps - margin)
        start_step = torch.randint(0, hi, (k,), device=device)
    else:
        start_step = torch.zeros(k, dtype=torch.long, device=device)

    # 2. Place each clip on its tile (identity heading -- clips are baked axis-aligned
    #    with the staircase, so alignment is a pure translation). Terrain / transition
    #    clips sit on the tile centerline (clip_y_center = size_y/2); flat clips at the
    #    corner (0), then re-centered below.
    world_quat = torch.zeros(k, 4, device=device)
    world_quat[:, 3] = 1.0  # identity (xyzw, w last)
    is_flat = env_tid == -1
    csize_y = torch.tensor(
        [float(importer.clip_size_y.get(int(t), 0.0)) for t in env_tid.tolist()], device=device
    )
    clip_y_center = torch.where(is_flat, torch.zeros_like(csize_y), 0.5 * csize_y)
    world_pos = clip_frame_origin(
        importer.env_cell_corner[env_ids],
        importer.env_cell_center[env_ids],
        clip_y_center,
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
    #      ego_frac. The FM snap still fires on ALL non-flat envs regardless (snap_mask), so
    #      even a non-ego terrain env gets an FM training window. Then compute + write the spawn.
    ego_frac = float(getattr(cmd.cfg, "ego_anchor_fraction", 0.0))
    ego_mask = torch.rand(k, device=device) < ego_frac
    snap_mask = ego_mask | ~is_flat  # non-flat: always snap (even when not ego this episode)
    _commit_clip_reset(
        cmd, asset, env_ids, clip_id, start_step, world_pos, world_quat, ego_mask, snap_mask
    )


def _find_output_indices(ordered_names: list[str], frame_name: str, suffix_pattern: str) -> list[int]:
    """
    Find indices of outputs matching frame_name:suffix_pattern.

    Args:
        ordered_names: List of ordered output names.
        frame_name: The frame name to search for.
        suffix_pattern: The suffix pattern to match (e.g., "pos_" or "ori_").

    Returns:
        List of indices where the pattern matches.
    """
    indices = []
    for i, name in enumerate(ordered_names):
        if name.startswith(f"{frame_name}:") and suffix_pattern in name:
            indices.append(i)
    return indices


def _pretty_print_reset_state(
    output_pos_names: list[str],
    output_vel_names: list[str],
    y_des: torch.Tensor,
    y_act: torch.Tensor,
    dy_des: torch.Tensor,
    dy_act: torch.Tensor,
    times: torch.Tensor,
    v: torch.Tensor,
    vdot: torch.Tensor,
    v_subgroups: dict[str, torch.Tensor],
    cmd_vel: torch.Tensor,
    clf,
    phase: torch.Tensor,
    current_domain: torch.Tensor,
    cur_ref_frame_idx: torch.Tensor,
    ref_frame_names: list[str],
    ref_pose: torch.Tensor,
    env_idx: int = 0,
):
    """
    Pretty print the desired and measured outputs in a table format.

    Args:
        output_names: List of output names.
        y_des: Desired position outputs, shape [num_envs, num_outputs].
        y_act: Measured position outputs, shape [num_envs, num_outputs].
        dy_des: Desired velocity outputs, shape [num_envs, num_outputs].
        dy_act: Measured velocity outputs, shape [num_envs, num_outputs].
        times: Sampled times, shape [num_envs].
        v: Lyapunov function value, shape [num_envs].
        vdot: Lyapunov derivative, shape [num_envs].
        v_subgroups: Dict mapping subgroup names to their V contributions, each shape [num_envs].
        cmd_vel: Commanded velocity, shape [num_envs, vel_dim].
        phase: Per-env trajectory phase, shape [num_envs].
        current_domain: Per-env domain index after reset, shape [num_envs].
        cur_ref_frame_idx: Per-env index into ``ref_frame_names``, shape [num_envs].
        ref_frame_names: Ordered reference-frame names (matches ``cur_ref_frame_idx``).
        ref_pose: Per-env reference-frame pose ``[x, y, z, qx, qy, qz, qw]``, shape [num_envs, 7].
        env_idx: Index of the environment to print (default 0).
    """
    # Header
    print(f"\n{'='*94}")
    print(f"Reset State for Environment {env_idx}")
    print(f"Time: {times[0].item():.4f} s    Phase: {phase[env_idx].item():.4f}")
    print(f"Domain Index: {current_domain[env_idx].item()}")
    ref_idx = cur_ref_frame_idx[env_idx].item()
    ref_name = ref_frame_names[ref_idx] if 0 <= ref_idx < len(ref_frame_names) else "<oob>"
    ref_xyz = ref_pose[env_idx, :3].tolist()
    print(f"Reference Frame: {ref_name} (idx {ref_idx})")
    print(f"Reference Frame Pos (x,y,z): "
          f"({ref_xyz[0]:.4f}, {ref_xyz[1]:.4f}, {ref_xyz[2]:.4f})")
    print(f"Commanded Velocity: {cmd_vel[env_idx].tolist()} m/s")
    print(f"{'='*94}")

    # Position table header
    print(f"\n{'--- Positions ---':^94}")
    print(f"{'Output Name':<30} {'Desired':>12} {'Measured':>12} {'Error':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    # Position table rows
    err_val = clf.compute_y_err(y_act, y_des)[env_idx]
    err_idx = 0
    for i, name in enumerate(output_pos_names):
        des_val = y_des[env_idx, i].item()
        act_val = y_act[env_idx, i].item()
        if ":ori_w" in output_pos_names[i]:
            print(f"{name:<30} {des_val:>12.5f} {act_val:>12.5f}")
        else:
            print(f"{name:<30} {des_val:>12.5f} {act_val:>12.5f} {err_val[err_idx]:>12.5f}")
            err_idx += 1


    # Velocity table header
    print(f"\n{'--- Velocities ---':^94}")
    print(f"{'Output Name':<30} {'Desired':>12} {'Measured':>12} {'Error':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    # Velocity table rows
    for i, name in enumerate(output_vel_names):
        des_val = dy_des[env_idx, i].item()
        act_val = dy_act[env_idx, i].item()
        err_val = act_val - des_val
        print(f"{name:<30} {des_val:>12.5f} {act_val:>12.5f} {err_val:>12.5f}")

    # V subgroups section
    print(f"\n{'V Subgroups':<30} {'Value':>12}")
    print(f"{'-'*30} {'-'*12}")
    for name, values in v_subgroups.items():
        print(f"{name:<30} {values[env_idx].item():>12.6f}")

    # Footer with V and vdot
    print(f"{'-'*94}")
    print(f"V: {v[env_idx].item():.6f}    Vdot: {vdot[env_idx].item():.6f}")
    print(f"{'='*94}\n")