from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_on_reference(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        command_name: str,
        conditioner_command_name: str,
        base_frame_name: str,
        joint_scale_range: tuple[float, float] = (1.0, 1.0),
        rel_envs_on_ref: float = 0.5,
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
    # cond_cmd = env.command_manager.get_term(conditioner_command_name)
    # print(f"cond_vel: {env.command_manager.get_command(conditioner_command_name)}")
    num_env = len(env_ids)

    if num_env == 0:
        return

    r = torch.empty(len(env_ids), device=env.device)

    ref_env = torch.zeros(num_env, device=env.device)
    ref_env = r.uniform_(0.0, 1.0) <= (rel_envs_on_ref)
    ref_ids = env_ids[ref_env]
    num_ref_envs = len(ref_ids)

    nonref_env = ref_env == False
    nonref_ids = env_ids[nonref_env]
    num_nonref_envs = len(nonref_ids)

    # Resample the command



    # Validate base frame exists in trajectory outputs
    pos_indices = _find_output_indices(cmd.ordered_output_names, base_frame_name, "pos_")
    ori_indices = _find_output_indices(cmd.ordered_output_names, base_frame_name, "ori_")

    if len(pos_indices) != 3:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have pos_x, pos_y, pos_z in trajectory outputs. "
            f"Found {len(pos_indices)} position outputs."
        )
    if len(ori_indices) != 3:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have ori_x, ori_y, ori_z in trajectory outputs. "
            f"Found {len(ori_indices)} orientation outputs."
        )

    # Validate all robot joints are in trajectory outputs
    traj_joint_names = set()
    for name in cmd.ordered_output_names:
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

    # Sample random times for each environment
    total_time = cmd.manager.get_total_time()
    random_times = torch.rand(num_ref_envs, device=env.device) * total_time

    # Get trajectory outputs at sampled times
    outputs = cmd.manager.get_output(random_times, ref_ids)  # Shape: [num_env, 2, num_outputs]
    y_sampled = outputs[:, 0, :]  # Position outputs

    # Extract base frame position and orientation from outputs
    base_pos_rel = y_sampled[:, pos_indices]  # Shape: [num_env, 3]
    base_ori_euler = y_sampled[:, ori_indices]  # Shape: [num_env, 3] - roll, pitch, yaw

    # Compute world-frame base pose
    base_pos_w = base_pos_rel + env.scene.env_origins[ref_ids]

    # Convert euler angles (roll, pitch, yaw) to quaternion (wxyz format)
    base_quat = quat_from_euler_xyz(
        base_ori_euler[:, 0],  # roll
        base_ori_euler[:, 1],  # pitch
        base_ori_euler[:, 2],  # yaw
    )

    # Build pose tensor: [x, y, z, qw, qx, qy, qz]
    base_pose = torch.cat([base_pos_w, base_quat], dim=-1)

    # Extract Base Vel
    dy_sampled = outputs[:, 1, :]  # Velocity outputs (not used for now)

    # Set base velocity
    base_vel = torch.cat([dy_sampled[:, pos_indices], dy_sampled[:, ori_indices]], dim=-1) #torch.zeros(num_env, 6, device=env.device)

    # Set the reference frame position
    cmd.ref_poses[ref_ids, :3] = env.scene.env_origins[ref_ids]
    cmd.ref_poses[ref_ids, 3:] *= 0

    # Extract joint positions from trajectory output
    # Build a mapping from robot joint indices to trajectory output indices
    joint_pos = torch.zeros(num_ref_envs, len(asset.joint_names), device=env.device)
    joint_vel = torch.zeros_like(joint_pos)

    for i, joint_name in enumerate(asset.joint_names):
        traj_output_name = f"joint:{joint_name}"
        traj_idx = cmd.ordered_output_names.index(traj_output_name)
        joint_pos[:, i] = y_sampled[:, traj_idx]
        joint_vel[:, i] = dy_sampled[:, traj_idx]

    # Apply optional uniform scaling to joint positions
    min_scale, max_scale = joint_scale_range
    if min_scale != 1.0 or max_scale != 1.0:
        scale_factors = torch.rand(num_ref_envs, 1, device=env.device) * (max_scale - min_scale) + min_scale
        joint_pos = joint_pos * scale_factors
        joint_vel = joint_vel * scale_factors

    # Store time offset in command so it knows the current phase
    cmd.init_time_offset[ref_ids] = random_times

    # Write states to simulation
    asset.write_root_pose_to_sim(base_pose, env_ids=ref_ids)
    asset.write_root_velocity_to_sim(base_vel, env_ids=ref_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ref_ids)

    if num_nonref_envs != 0:
        # Non reference resets
        root_states = asset.data.default_root_state[nonref_ids].clone()

        # TODO: Consider adding some amount of noise
        base_pose = root_states[:, 0:7]
        base_pose[:, :3] += env.scene.env_origins[nonref_ids]
        base_vel = root_states[:, 7:]

        # get default joint state
        joint_pos = asset.data.default_joint_pos[nonref_ids, asset_cfg.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[nonref_ids, asset_cfg.joint_ids].clone()

        asset.write_root_pose_to_sim(base_pose, env_ids=nonref_ids)
        asset.write_root_velocity_to_sim(base_vel, env_ids=nonref_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=nonref_ids)

        # Reset the time offset
        cmd.init_time_offset[nonref_ids] = 0.0



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