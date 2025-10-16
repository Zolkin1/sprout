# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def vdot_tanh(env: ManagerBasedRLEnv, command_name: str, alpha: float = 1.0) -> torch.Tensor:
    # Retrieve the CLF-related quantities: V and its time derivative
    ref_term = env.command_manager.get_term(command_name)  # [B]
    vdot = ref_term.vdot  # [B]
    v = ref_term.v        # [B]

    # Compute the CLF decay condition violation
    clf_decay_violation = vdot + alpha * v  # [B]

    # Reward is higher when this violation is negative (i.e., condition is satisfied)
    vdot_reward = torch.tanh(-clf_decay_violation)  # [B]

    return vdot_reward


def clf_reward(env: ManagerBasedRLEnv, command_name: str, max_eta_err: float = 0.15, eps: float = 1e-6) -> torch.Tensor:
    """CLF-based reward: r = exp(-V(η) / V_max), clipped to [0, 1]."""

    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v  # [B] scalar CLF value per env
    max_clf = ref_term.clf.lambda_max * max_eta_err ** 2 + eps # principled normalization; lambda_max(P) * eta**2

    reward = torch.exp(-torch.clamp(v, max=5.0 * max_clf) / max_clf)
    return reward


def clf_decreasing_condition(
    env: ManagerBasedRLEnv,
    command_name: str,
    alpha: float = 1.0,
    eta_max: float = 0.15,
    eta_dot_max: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalty for violating CLF decrease condition: 𝑟 = clip((ΔV + αV) / max_violation, [0, 1])
    where:
        max_violation ≈ 2‖P‖ η_max η̇_max + α λ_max(P) η_max²
    """

    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v        # [B]
    vdot = ref_term.vdot  # [B]

    lambda_max = ref_term.clf.lambda_max
    norm_P = ref_term.clf.norm_P

    # Theoretical upper bound on violation
    max_violation = (
        2.0 * norm_P * eta_max * eta_dot_max + alpha * lambda_max * eta_max ** 2 + eps
    )
    # Only penalize when violation is positive
    violation = torch.clamp(vdot + alpha * v, min=0.0)
    penalty = violation / max_violation
    penalty = torch.clamp(penalty, min=0.0, max=1.0)
    return penalty


def v_dot_penalty(env: ManagerBasedRLEnv, command_name: str,eta_max: float = 0.15,
    eta_dot_max: float = 0.5,eps: float = 1e-6) -> torch.Tensor:
    ref_term = env.command_manager.get_term(command_name)                    # [B]
    vdot = ref_term.vdot # [B]

    norm_P = ref_term.clf.norm_P

    max_violation = (
        2.0 * norm_P * eta_max * eta_dot_max + eps
    )

    vdot_penalty = torch.tanh(torch.clamp(vdot, min=0.0) / max_violation) 
    return vdot_penalty


def contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet contact with zero velocity."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] * contacts.unsqueeze(-1)
    # shape [B, num_feet, 3]
    penalize = torch.square(body_vel[:,:,:3])
    return torch.sum(penalize, dim=(1,2))


def holonomic_constraint_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_vel: float = (0.1)**0.5
) -> torch.Tensor:
    """
    Unified holonomic‐velocity constraint reward:
      r = exp( – ‖[v, ω_z]‖² / σ_vel² )
    where v∈R³ is the foot’s linear velocity and ω_z its yaw rate.
    Using σ_vel=√0.1 matches the original bandwidth (denominator=0.1).
    """
    cmd = env.command_manager.get_term(command_name)

    # linear velocity [B,3] and yaw rate [B,1]
    v   = cmd.stance_foot_vel                        # [vx, vy, vz]
    wz  = cmd.stance_foot_ang_vel[:, 2].unsqueeze(-1) # [ω_z]

    # stack into [B,4] error vector
    e_vel = torch.cat([v, wz], dim=-1)

    # unified exponential‐norm reward
    return torch.exp(- (e_vel**2).sum(dim=-1) / sigma_vel**2)

def holonomic_constraint(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_pose: float = (5 * 0.01) ** 0.5,
    z_offset: float = 0.036
) -> torch.Tensor:
    """
    Unified holonomic‐pose constraint reward:
        r = exp( – ‖e_pose‖² / σ_pose² )
    where e_pose = [Δx, Δy, Δz, φ, Δψ] and
      • Δx, Δy are planar errors from the recorded foot position,
      • Δz = p_z_cur – z_offset (encourages foot to stay on the floor),
      • φ is roll,
      • Δψ is yaw error wrapped to [–π, π].
    """

    cmd = env.command_manager.get_term(command_name)

    # planar position error [B,2]
    p0_xy = cmd.stance_foot_pos_0[:, :2]
    p_xy  = cmd.stance_foot_pos[:, :2]
    delta_xy = p_xy - p0_xy

    # vertical error to the floor plane [B,1]
    z_cur    = cmd.stance_foot_pos[:, 2].unsqueeze(-1)
    delta_z  = z_cur - cmd.stance_foot_pos_0[:,2].unsqueeze(-1)

    # roll error [B,1]
    roll = cmd.stance_foot_ori[:, 0].unsqueeze(-1)

    # yaw error wrapped to [–π, π] [B,1]
    psi0 = cmd.stance_foot_ori_0[:, 2]
    psi  = cmd.stance_foot_ori[:, 2]
    delta_psi = ((psi - psi0 + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)

    # stack into [B,5] error vector
    e_pose = torch.cat([delta_xy, delta_z, roll, delta_psi], dim=-1)

    # unified Gaussian‐like reward
    return torch.exp(- (e_pose**2).sum(dim=-1) / sigma_pose**2)


def reference_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    term_std: Sequence[float],
    term_weight: Sequence[float],
) -> torch.Tensor:
    """
    Exponential reward per dimension, scaled by weight — ignores zero-weight terms.
    """
    command = env.command_manager.get_term(command_name)
    err = command.y_act - command.y_out  # [B, D]

    weight_vec = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)  # [D]
    std_vec = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)        # [D]

    # [B, D] scaled squared error per dimension
    err_sq_scaled = (err ** 2) / (std_vec ** 2)

    # Apply element-wise exp(-error²/std²) and weight
    reward_per_dim = weight_vec * torch.exp(-err_sq_scaled)  # [B, D]
    reward = reward_per_dim.sum(dim=1)/torch.sum(weight_vec)  # [B]

    return reward


def reference_vel_tracking(    env: ManagerBasedRLEnv,
    command_name: str,
    term_std: Sequence[float],
    term_weight: Sequence[float],
) -> torch.Tensor:
    """Reference tracking with element-wise term weights."""
    # 1. fetch the command and compute error [B, D]
    command = env.command_manager.get_term(command_name)
    err = command.dy_act - command.dy_out

    weight_vec = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)  # [D]
    std_vec = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)        # [D]

    # [B, D] scaled squared error per dimension
    err_sq_scaled = (err ** 2) / (std_vec ** 2)

    # Apply element-wise exp(-error²/std²) and weight
    reward_per_dim = weight_vec * torch.exp(-err_sq_scaled)  # [B, D]
    reward = reward_per_dim.sum(dim=1)/torch.sum(weight_vec)  # [B]
    return reward



def foot_clearance(env: ManagerBasedRLEnv,
                   target_height: float,
                   sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
                   height_sensor_cfg: SceneEntityCfg | None = None,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward foot clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact state
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    if height_sensor_cfg is not None:
        sensor: RayCaster = env.scene[height_sensor_cfg.name]
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[...,2],dim=1).unsqueeze(-1)
    else:
        adjusted_target_height = target_height

    # Calculate foot heights
    feet_z_err = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - adjusted_target_height
    pos_error = torch.square(feet_z_err) * ~contacts

    return torch.sum(pos_error, dim=(1))

def phase_contact(
    env: ManagerBasedRLEnv,
        period: float = 0.8,
        command_name: str | None = None,
        Tswing: float =0.4,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward foot contact with regards to phase."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get contact state
    res = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    # Contact phase
    tp = (env.sim.current_time % period) / period     # Scaled between 0-1
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)

    stance_i = int(0.5 - 0.5 * torch.sign(phi_c))


     # check if robot needs to be standing
    if command_name is not None:
        command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
        is_small_command = command_norm < 0.005
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance | is_small_command
            contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    else:
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance
            contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    return res

# ================================================= #
# ========== Unitree Open Source Rewards ========== #
# ================================================= #
"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


# def stand_still(
#     env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#
#     reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
#     return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


# def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize z-axis base linear velocity using L2 squared kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
#     return reward


# def joint_position_penalty(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
# ) -> torch.Tensor:
#     """Penalize joint position error from default on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#     reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


# def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
#     forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
#     # Penalize feet hitting vertical surfaces
#     reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
#     return reward


# def feet_height_body(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg,
#     target_height: float,
#     tanh_mult: float,
# ) -> torch.Tensor:
#     """Reward the swinging feet for clearing a specified height off the ground"""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
#     footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
#     cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
#         :, :
#     ].unsqueeze(1)
#     footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
#     for i in range(len(asset_cfg.body_ids)):
#         footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
#             asset.data.root_quat_w, cur_footpos_translated[:, i, :]
#         )
#         footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
#             asset.data.root_quat_w, cur_footvel_translated[:, i, :]
#         )
#     foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
#     foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
#     reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
#     reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


# def feet_too_near(
#     env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
#     distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
#     return (threshold - distance).clamp(min=0)


# def feet_contact_without_cmd(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
# ) -> torch.Tensor:
#     """
#     Reward for feet contact when the command is zero.
#     """
#     # asset: Articulation = env.scene[asset_cfg.name]
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
#
#     command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
#     reward = torch.sum(is_contact, dim=-1).float()
#     return reward * (command_norm < 0.1)


# def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
#     return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
#         torch.clip(last_contact_time, max=0.5), dim=1
#     )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""
# def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
#         # Cache joint positions for all pairs
#         env.joint_mirror_joints_cache = [
#             [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
#         ]
#     reward = torch.zeros(env.num_envs, device=env.device)
#     # Iterate over all joint pairs
#     for joint_pair in env.joint_mirror_joints_cache:
#         # Calculate the difference for each pair and add to the total reward
#         reward += torch.sum(
#             torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
#             dim=-1,
#         )
#     reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
#     return reward