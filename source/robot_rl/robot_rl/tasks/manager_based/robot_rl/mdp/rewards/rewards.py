# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
import re
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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

    NOT REGISTERED by any task right now: the teacher dropped both velocity terms because
    its CLF already grades the base twist against the reference, and grading the command
    again was re-scoring the same quantity through a softer lens.  Kept because the
    tracker may want them back -- see ``G1TeacherRewardCfg``.
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

    NOT REGISTERED by any task right now -- see the note on
    :func:`command_lin_vel_xy_tracking_exp`.
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


##
# Terrain consistency: does the PLAN's own geometry agree with the ground?
##

# The SDF tiling is NOT a constant here: it is derived per call from
# `motion_gen.terrain_consistency.sdf_chunk_nodes`, the same rule the offline grader uses. A
# fixed node count scales its working set with the env count -- 8 nodes is 0.07 GB at 384 envs
# and 1.48 GB at 8192 -- so an eval scaled up for speed would run out of memory instead.


def _terrain_consistency(env: ManagerBasedRLEnv, command_name: str) -> dict[str, torch.Tensor]:
    """Grade the live plan against the env's terrain, once per step, cached.

    Three reward terms read this and the manager calls each separately, so without a cache the
    whole SDF pass (a few thousand collider points x the plan's nodes) would run three times
    per step for the same answer. Keyed on the env's step counter, so a stale result cannot
    outlive the plan it describes.

    Everything numeric is delegated and nothing here restates a threshold or a geometry: the
    contact flags come from the COMMAND's ``plan_foot_contact`` (the same producer the debug
    ribbon colours, so the flag graded is the flag drawn), the world lift from its
    ``plan_world_pose``, the boxes from the terrain importer, and the terms from
    :mod:`motion_gen.terrain_consistency`.

    Args:
        env: The environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        The :func:`~motion_gen.terrain_consistency.terrain_consistency` mapping.

    Raises:
        RuntimeError: If the command cannot produce contact flags (a window that predicts no
            joints), or the terrain has no box decomposition. Both are refusals rather than
            zeros: a zero here reads as a geometrically perfect plan.
    """
    from motion_gen.plan_geometry import plan_collider_points          # noqa: PLC0415
    from motion_gen.terrain_consistency import (                       # noqa: PLC0415
        sdf_chunk_nodes, terrain_consistency)

    cached = getattr(env, "_terrain_consistency_cache", None)
    step = int(env.common_step_counter)
    if cached is not None and cached[0] == step:
        return cached[1]

    cmd = env.command_manager.get_term(command_name)
    contact = cmd.plan_foot_contact()
    if contact is None:
        raise RuntimeError(
            f"command {command_name!r} cannot produce plan contact flags, so the "
            "terrain-consistency rewards have nothing to gate on -- this window predicts no "
            "joint positions. Returning zeros would read as a perfect plan."
        )
    model = cmd.plan_collider_model()
    base_pos, base_rot, joint_pos = cmd.plan_world_pose()
    points = plan_collider_points(base_pos, base_rot, joint_pos, model)
    terrain = env.scene.terrain.terrain_collider_boxes()
    out = terrain_consistency(
        points, contact, terrain, model,
        chunk_nodes=sdf_chunk_nodes(points.shape[0], model.num_points,
                                    terrain.boxes.shape[1]))
    env._terrain_consistency_cache = (step, out)
    return out


def plan_terrain_terms(env: ManagerBasedRLEnv, command_name: str = "traj_ref"
                       ) -> dict[str, torch.Tensor]:
    """The full terrain-consistency grade of the live plan, every reduction.

    The public face of the same cached producer the three reward terms read, so a consumer
    that wants more than one scalar -- the rollout recorder's per-re-plan chunk table -- gets
    the SAME numbers the rewards would report rather than a second grade of its own.  Calling
    this and then a reward term in the same step costs one evaluation, not two.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        The :func:`~motion_gen.terrain_consistency.terrain_consistency` mapping: the two
        penalties, their ``_served`` twins, the alternative reductions and the per-node
        arrays.
    """
    return _terrain_consistency(env, command_name)


def plan_terrain_contact(env: ManagerBasedRLEnv, command_name: str = "traj_ref"
                         ) -> torch.Tensor:
    """PARC's terrain CONTACT loss over the plan, widened by a deadband. **Wire NEGATIVE.**

    ``mean_t sum_b c_b * max(0, min_p |sd(p)| - deadband)`` over the plan's nodes, where ``b``
    runs over the feet, ``c_b`` is the plan's own kinematic contact flag, and ``p`` runs over
    that foot's sole points (PARC Eq 8; see
    :func:`~motion_gen.terrain_consistency.terrain_consistency` for the two deviations -- the
    deadband, and reading the sole rather than the whole body).

    This is the term that grades MODE selection. A plan whose own kinematics say a foot is
    planted, 15 cm above the tread it is meant to be standing on, has read the terrain wrong
    -- and it has done so BEFORE the robot falls, which is what makes it actionable in a way
    an outcome is not.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs]``, non-negative, 0 when every marked stance sits within the deadband of
        the terrain.
    """
    return _terrain_consistency(env, command_name)["contact_penalty"]


def plan_terrain_penetration(env: ManagerBasedRLEnv, command_name: str = "traj_ref"
                             ) -> torch.Tensor:
    """PARC's terrain PENETRATION loss over the plan. **Wire NEGATIVE.**

    ``mean_t sum_i max(0, -sd(p_i) - tol)`` over the plan's nodes and every collider point the
    contact buffer does not exempt (PARC Eq 7).

    **The magnitude is tied to the collider sampling density**, because a sum over points is.
    ``motion_gen.plan_geometry.DEFAULT_COLLIDER_SPACING`` is therefore part of this term's
    definition: changing it rescales the reward, so a tuned weight does not survive such a
    change. ``penetration_volume_m3`` is the area-weighted, density-invariant twin if a
    comparison across spacings is ever needed.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs]``, non-negative, 0 for a plan that stays out of the terrain.
    """
    return _terrain_consistency(env, command_name)["penetration_sum_m"]


def plan_support_fraction(env: ManagerBasedRLEnv, command_name: str = "traj_ref"
                          ) -> torch.Tensor:
    """Fraction of plan nodes with at least one foot in contact. **Log only (weight 0).**

    The guard against the contact term being satisfied vacuously: both PARC terms are gated by
    ``c_b``, so a plan that never marks a stance is charged nothing by either and a policy
    could in principle learn to keep its feet moving. This puts that in the training log
    instead of leaving it as a slow, invisible drift.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs]`` in ``[0, 1]``.
    """
    return _terrain_consistency(env, command_name)["support_fraction"]

