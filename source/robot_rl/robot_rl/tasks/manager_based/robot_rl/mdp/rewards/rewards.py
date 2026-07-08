# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
import math
import re
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv

KAPPA = 0.5

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

def clf_reward(env: ManagerBasedRLEnv, command_name: str, max_eta_err: float = 0.15, eps: float = 1e-6) -> torch.Tensor:
    """CLF-based reward: r = exp(-V(η) / V_max), clipped to [0, 1]."""

    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v  # [B] scalar CLF value per env
    max_clf = ref_term.clf.lambda_max * max_eta_err ** 2 + eps # principled normalization; lambda_max(P) * eta**2

    reward = torch.exp(-v / max_clf)
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
    dt = 0.02
    max_violation = (
        dt * eta_dot_max * norm_P + alpha * lambda_max * eta_max ** 2 + eps
    )
    # Only penalize when violation is positive
    # violation = torch.clamp(vdot + alpha * v, min=0.0)
    violation = vdot + alpha * v
    penalty = violation / max_violation
    penalty = torch.clamp(penalty, min=0.0, max=1.0)
    return penalty


def _base_vel_output_indices(cmd) -> list[int]:
    """Indices of the base frame's linear (pos_*) + angular (ori_*) velocity
    channels within a trajectory command's ``ordered_vel_output_names``."""
    base = cmd.cfg.base_frame_name
    names = cmd.ordered_vel_output_names
    return (
        [names.index(f"{base}:pos_{a}") for a in ("x", "y", "z")]
        + [names.index(f"{base}:ori_{a}") for a in ("x", "y", "z")]
    )


def ref_base_vel_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.5,
) -> torch.Tensor:
    """Dense exp reward for tracking the clip's reference base velocity.

    Compares the measured vs reference base (pelvis) linear + angular velocity
    -- the ``dy_act`` / ``dy_des`` channels of the trajectory command, both in
    the same yaw-aligned anchor frame -- giving a smooth, shaped locomotion
    signal on top of the (harder) CLF tracking reward.  Analogous to the
    velocity-command ``track_lin_vel_xy_exp`` / ``track_ang_vel_z_exp`` rewards,
    but graded against the reference trajectory rather than a commanded velocity.
    """
    cmd = env.command_manager.get_term(command_name)
    idx = _base_vel_output_indices(cmd)
    err = cmd.dy_act[:, idx] - cmd.dy_des[:, idx]
    return torch.exp(-torch.sum(err * err, dim=1) / (std ** 2))


def command_lin_vel_xy_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense exp reward for tracking the clip's commanded planar (xy) velocity.

    Mirrors Isaac Lab's :func:`track_lin_vel_xy_exp` (planar linear velocity in
    the base frame, exponential kernel), but sources the command from the
    trajectory command's joystick-level ``target_vel`` (``(vx, vy, wz)`` -- what
    a velocity "joystick" would supply) rather than a separate ``base_velocity``
    command term, and grades the measured *link*-frame base velocity (no CoM, see
    the velocity-convention note on :class:`FmTrackingCommand`).  Paired
    with :func:`command_ang_vel_z_tracking_exp` for yaw.  Requires a clip library
    carrying ``target_vel``.
    """
    cmd = env.command_manager.get_term(command_name)
    target_vel = cmd.get_command_velocity()  # [N, 3] = (vx, vy, wz)
    asset: Articulation = env.scene[asset_cfg.name]
    meas = wp.to_torch(asset.data.root_link_lin_vel_b)[:, :2]  # planar linear velocity (link frame)
    err = torch.sum(torch.square(target_vel[:, :2] - meas), dim=1)
    return torch.exp(-err / (std ** 2))


def command_ang_vel_z_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense exp reward for tracking the clip's commanded yaw rate.

    Mirrors Isaac Lab's :func:`track_ang_vel_z_exp` (yaw rate in the base frame,
    exponential kernel), but sources the command from the trajectory command's
    joystick-level ``target_vel`` (the ``wz`` component) rather than a separate
    ``base_velocity`` command term.  Uses the *link*-frame yaw rate (identical to
    the CoM yaw rate, but kept link-frame for convention consistency).  Paired
    with :func:`command_lin_vel_xy_tracking_exp` for planar velocity.  Requires a
    clip library carrying ``target_vel``.
    """
    cmd = env.command_manager.get_term(command_name)
    target_vel = cmd.get_command_velocity()  # [N, 3] = (vx, vy, wz)
    asset: Articulation = env.scene[asset_cfg.name]
    meas = wp.to_torch(asset.data.root_link_ang_vel_b)[:, 2]  # yaw rate (link frame)
    err = torch.square(target_vel[:, 2] - meas)
    return torch.exp(-err / (std ** 2))


def multiple_undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfgs: Sequence[SceneEntityCfg]) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations above a threshold, aggregated across multiple sensors."""
    total = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for sensor_cfg in sensor_cfgs:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
        is_contact = (
            torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        )
        # print(f"{sensor_cfg.name} net force: {net_contact_forces.cpu()}")
        total += torch.sum(is_contact, dim=1)
    return total

def flight_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    base_vel_cmd: str,
    sensor_cfgs: Sequence[SceneEntityCfg],
    weight_scalar: float,
    start_vel: float,
) -> torch.Tensor:
    """Penalize ground contact while the reference is in its flight phase.

    Rewards a genuine flight phase at running speed: whenever the *reference*
    commands every tracked foot into swing (no desired contact) and the
    commanded forward speed is at least ``start_vel`` m/s, any measured foot
    contact force is penalized. ``weight_scalar`` starts at 0 and is ramped up
    by :func:`~robot_rl.tasks.manager_based.robot_rl.mdp.curriculums.curriculums.contact_curriculum`,
    so the penalty turns on gradually over training.

    Args:
        env: The RL environment.
        command_name: Name of the trajectory-tracking command (the reference).
        base_vel_cmd: Name of the base-velocity command (used for the speed gate).
        sensor_cfgs: Foot contact sensors whose forces are summed.
        weight_scalar: Curriculum-ramped scale on the penalty magnitude.
        start_vel: Minimum commanded forward speed (m/s) for the penalty to apply.

    Returns:
        Non-negative per-environment penalty of shape ``[N]`` (applied with a
        negative reward weight).
    """
    cmd = env.command_manager.get_term(command_name)

    # Flight = the reference commands every tracked foot into swing (no contact).
    contact_state = cmd.get_contact_state(cmd.manager.phase)  # [N, num_contacts]
    flight_mask = (contact_state.sum(dim=-1) == 0).float()    # [N]

    # Only penalize at running speed (commanded forward velocity >= start_vel).
    command_vel = env.command_manager.get_command(base_vel_cmd)[:, :3]
    speed_mask = (command_vel[:, 0] >= start_vel).float()     # [N]

    # Measured foot contact-force magnitude, summed across the foot sensors.
    contact_forces = torch.zeros(env.num_envs, device=env.device)
    for sensor_cfg in sensor_cfgs:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)  # [N, H, B, 3]
        contact_forces += torch.linalg.norm(net_forces[:, 0, sensor_cfg.body_ids], dim=-1).sum(dim=-1)

    penalty = weight_scalar * torch.tanh(contact_forces / 0.5)
    return flight_mask * speed_mask * penalty

def _active_stance_frame_state(cmd):
    """World pose + velocity of the currently-active stance reference frame, per env.

    Returns ``(pos [N,3], quat [N,4] wxyz, lin_vel [N,3], ang_vel [N,3],
    not_flight [N])``. ``not_flight`` is 1 for envs whose reference is in a
    single-support (contact) domain and 0 during flight -- the holonomic
    constraints only apply while a foot is planted.
    """
    robot = cmd.robot
    n = cmd.num_envs
    # active_body: robot body index of the stance ref frame, per env.
    # ``ref_frame_indices`` is a python list[int]; index it with the per-env
    # active-frame index (a tensor) via a tensor gather.
    ref_body_idx = torch.as_tensor(cmd.ref_frame_indices, device=cmd.device, dtype=torch.long)
    active_body = ref_body_idx[cmd.cur_ref_frame_idx]                    # [N]
    ar = torch.arange(n, device=cmd.device)
    body_pos = wp.to_torch(robot.data.body_pos_w)[ar, active_body]       # [N,3]
    body_quat = wp.to_torch(robot.data.body_quat_w)[ar, active_body]     # [N,4] wxyz
    body_vel = wp.to_torch(robot.data.body_link_vel_w)[ar, active_body]  # [N,6] (lin, ang)
    not_flight = (cmd.get_contact_state(cmd.manager.phase).sum(dim=-1) > 0).float()
    return body_pos, body_quat, body_vel[:, :3], body_vel[:, 3:], not_flight

def holonomic_constraint(env: ManagerBasedRLEnv, command_name: str, sigma_pose: float = math.sqrt(0.05)) -> torch.Tensor:
    """Reward pinning the stance foot to its anchor pose during single support.

    Exp kernel on the stance foot's drift from the re-anchor pose (the command's
    ``ref_poses``): e = [dx, dy, dz, roll, d_yaw]. Masked off during flight.
    Ported from the ICRA HZD running config (weight 4.0), adapted to the batched
    command's ``ref_poses`` anchor + the active stance-foot body world pose.

    Args:
        env: The RL environment.
        command_name: Name of the trajectory-tracking command (the reference).
        sigma_pose: Kernel width; matches the submission's sqrt(0.05).

    Returns:
        Per-environment reward in [0, 1] (zero during flight).
    """
    cmd = env.command_manager.get_term(command_name)
    pos, quat, _, _, not_flight = _active_stance_frame_state(cmd)
    anchor_pos = cmd.ref_poses[:, :3]
    anchor_quat = cmd.ref_poses[:, 3:]

    delta_xy = pos[:, :2] - anchor_pos[:, :2]                     # [N,2]
    delta_z = (pos[:, 2] - anchor_pos[:, 2]).unsqueeze(-1)        # [N,1]
    roll, _, yaw = euler_xyz_from_quat(quat, wrap_to_2pi=False)
    _, _, anchor_yaw = euler_xyz_from_quat(anchor_quat, wrap_to_2pi=False)
    delta_psi = wrap_to_pi(yaw - anchor_yaw).unsqueeze(-1)        # [N,1]
    roll = wrap_to_pi(roll).unsqueeze(-1)                         # [N,1] (near 0 for a flat foot)

    e_pose = torch.cat([delta_xy, delta_z, roll, delta_psi], dim=-1)   # [N,5]
    return not_flight * torch.exp(-(e_pose ** 2).sum(dim=-1) / sigma_pose ** 2)

def holonomic_constraint_vel(env: ManagerBasedRLEnv, command_name: str, sigma_vel: float = math.sqrt(0.1)) -> torch.Tensor:
    """Reward zero stance-foot velocity (lin xyz + yaw rate) during single support.

    Exp kernel on the active stance foot's world linear velocity and yaw rate;
    masked off during flight. Ported from the ICRA HZD running config (weight 2.0).

    Args:
        env: The RL environment.
        command_name: Name of the trajectory-tracking command (the reference).
        sigma_vel: Kernel width; matches the submission's sqrt(0.1).

    Returns:
        Per-environment reward in [0, 1] (zero during flight).
    """
    cmd = env.command_manager.get_term(command_name)
    _, _, lin_vel, ang_vel, not_flight = _active_stance_frame_state(cmd)
    e_vel = torch.cat([lin_vel, ang_vel[:, 2:3]], dim=-1)         # [N,4]
    return not_flight * torch.exp(-(e_vel ** 2).sum(dim=-1) / sigma_vel ** 2)

def torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.
    For implicit actuators, we manually compute the PD controller torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Manually compute PD controller torques for implicit actuators
    computed_torque = torch.zeros_like(wp.to_torch(asset.data.joint_pos))

    # Get current joint positions, velocities, and desired positions
    current_pos = wp.to_torch(asset.data.joint_pos)
    current_vel = wp.to_torch(asset.data.joint_vel)
    desired_pos = wp.to_torch(asset.data.joint_pos_target)
    
    # Access actuator configurations from the asset
    actuator_groups = asset.cfg.actuators
    
    for group_name, actuator_cfg in actuator_groups.items():
        # Get joint indices for this actuator group
        joint_indices = asset.find_joints(actuator_cfg.joint_names_expr)[0]
        
        # Get stiffness and damping values for this group
        if isinstance(actuator_cfg.stiffness, dict):
            # Handle per-joint stiffness values
            kp_values = torch.zeros(len(joint_indices), dtype=torch.float32, device=env.device)
            for i, joint_idx in enumerate(joint_indices):
                joint_name = asset.joint_names[joint_idx]
                # Find matching stiffness pattern
                for pattern, value in actuator_cfg.stiffness.items():
                    if re.match(pattern.replace(".*", ".*"), joint_name):
                        kp_values[i] = value
                        break
        else:
            # Single stiffness value for all joints in this group
            kp_values = torch.full((len(joint_indices),), actuator_cfg.stiffness, dtype=torch.float32, device=env.device)
        
        if isinstance(actuator_cfg.damping, dict):
            # Handle per-joint damping values
            kd_values = torch.zeros(len(joint_indices), dtype=torch.float32, device=env.device)
            for i, joint_idx in enumerate(joint_indices):
                joint_name = asset.joint_names[joint_idx]
                # Find matching damping pattern
                for pattern, value in actuator_cfg.damping.items():
                    if re.match(pattern.replace(".*", ".*"), joint_name):
                        kd_values[i] = value
                        break
        else:
            # Single damping value for all joints in this group
            kd_values = torch.full((len(joint_indices),), actuator_cfg.damping, dtype=torch.float32, device=env.device)
        
        # Compute PD torques for this group: tau = kp * (q_des - q) - kd * q_dot
        pos_error = desired_pos[:, joint_indices] - current_pos[:, joint_indices]
        pd_torque = (kp_values[None, :] * pos_error - kd_values[None, :] * current_vel[:, joint_indices])
        
        # Store computed torques
        computed_torque[:, joint_indices] = pd_torque
    
    # Compute torque limit violations
    torque_limits_upper = wp.to_torch(asset.data.joint_effort_limits)[0, asset_cfg.joint_ids]  # Upper limits

    # Get computed torques for the specified joints
    joint_torques = computed_torque[:, asset_cfg.joint_ids]
    
    # Compute violations: how much torques exceed the limits
    violation = torch.clamp(torch.abs(joint_torques) - torque_limits_upper, min=0)

    # Sum all violations
    return torch.sum(violation, dim=1)

##
# Individual tracking rewards
##
def base_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_pos = cmd_term.clf.v_subgroups["pelvis_pos"]

    return torch.exp(-KAPPA * v_base_pos / sigma)

def base_lin_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_lin_vel = cmd_term.clf.v_subgroups["pelvis_lin_vel"]

    return torch.exp(-KAPPA * v_base_lin_vel / sigma)

def base_ori_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_ori = cmd_term.clf.v_subgroups["pelvis_ori"]

    return torch.exp(-KAPPA * v_base_ori / sigma)

def base_ang_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_ang_vel = cmd_term.clf.v_subgroups["pelvis_ang_vel"]

    return torch.exp(-KAPPA * v_base_ang_vel / sigma)

def joint_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_joint_pos = cmd_term.clf.v_subgroups["joint_pos"]

    return torch.exp(-KAPPA * v_joint_pos / sigma)

def joint_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_joint_vel = cmd_term.clf.v_subgroups["joint_vel"]

    return torch.exp(-KAPPA * v_joint_vel / sigma)

def body_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_pos = cmd_term.clf.v_subgroups["other_body_pos"]

    return torch.exp(-KAPPA * v_body_pos / sigma)

def body_lin_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_lin_vel = cmd_term.clf.v_subgroups["other_body_lin_vel"]

    return torch.exp(-KAPPA * v_body_lin_vel / sigma)

def body_ori_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_ori = cmd_term.clf.v_subgroups["other_body_ori"]

    return torch.exp(-KAPPA * v_body_ori / sigma)

def body_ang_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_ang_vel = cmd_term.clf.v_subgroups["other_body_ang_vel"]

    return torch.exp(-KAPPA * v_body_ang_vel / sigma)