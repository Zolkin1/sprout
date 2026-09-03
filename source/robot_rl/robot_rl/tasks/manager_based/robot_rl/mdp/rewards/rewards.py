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
# The adjuster: one reward per re-plan interval
##
def served_track_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The interval's MEAN tracking reward, as one number per adjuster decision.

    Read off the env rather than computed here, and that is the whole point. The adjuster
    acts once per re-plan interval, so the reward it can learn from is one scalar per
    decision -- but tracking quality is a per-CONTROL-STEP quantity, and a reward term cannot
    be evaluated at a rate different from its manager's. So the env's inner loop folds
    ``tracking_error.step_reward`` every control step and this term collects the finished
    mean.

    **A mean, not a sum.** ``step_reward`` is bounded in ``(0, 1]`` by construction --
    ``TrackingErrorSpec.__post_init__`` enforces that its four weights sum to 1 precisely so
    it is -- and a mean preserves that bound while a sum scales with the cadence, putting the
    value target's range at the mercy of ``fm_update_every``. It is the same argument
    ``tracking_error.chunk_reward``'s docstring makes for itself.

    The mean is over the control steps an env was still ALIVE for, not over all K, so an env
    that fails a third of the way in is not diluted by two thirds zeros -- otherwise the
    return would encode WHEN in the interval the failure happened rather than that it did.

    Args:
        env: The environment, which must expose ``track_mean``.

    Returns:
        ``[num_envs]`` in ``(0, 1]``, zero for an interval with no served reference.

    Raises:
        RuntimeError: If the env does not fold the accumulator. Returning zeros instead would
            read as a robot that tracked nothing, which is a plausible number and would make
            the whole dense signal silently absent.
    """
    value = getattr(env, "track_mean", None)
    if value is None:
        raise RuntimeError(
            "served_track_reward needs an env that folds `track_mean` over the interval's "
            "control steps (see AdjusterEnv.step). On a per-control-step env there is no "
            "interval to average, so use tracking_error.step_reward directly instead."
        )
    return value


def _to_sigma(cmd, raw: torch.Tensor) -> torch.Tensor:
    """Convert a residual-shaped ``[N, F, D]`` tensor from RAW units to normalizer sigmas.

    Sigma units because one window node mixes metres, dimensionless rot6d components and
    radians, and a squared mean over raw channels would be dominated by whichever block
    happens to have the largest units. The FM's normalizer is DECLARED rather than fitted
    (position 0.40 m, rot6d 1.0, velocities 0.5 / 0.4, joints 0.067-0.532 rad), so a sigma
    means the same thing in every run rather than being a property of whichever recording the
    normalizer was fitted on.

    Scale only, no mean subtraction: these are DIFFERENCES, so there is no offset to remove.
    """
    return raw / cmd.ctx.state_norm.std.to(raw.device)


def plan_residual_l2(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Mean squared plan residual over nodes and channels, in normalizer sigmas.

    Wire with a NEGATIVE weight. This is what keeps the corrected plan inside the
    distribution the frozen tracker was trained on, and it is the only limit on the channels
    that carry no hard cap (velocities and joint angles) -- so its weight is not decoration.

    Reads what was APPLIED, not the action: the request passes through a squash, a per-channel
    scale, a rotation projection and an angle clamp, and pricing the request would under-price
    a correction the clamp cut down and over-price one it did not.

    **A MEAN over ``F * D``, never a sum.** A sum silently re-weights itself whenever the node
    count, the block selection or the channel widths move -- so the same configured weight
    would mean something different after a layout change, which is exactly the class of drift
    the window's one-producer tables exist to prevent.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs]``, non-negative, 0 before the first residual is applied.
    """
    cmd = env.command_manager.get_term(command_name)
    return _to_sigma(cmd, cmd.plan_residual()).pow(2).mean(dim=(1, 2))


def plan_residual_rate_l2(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """Mean squared CHANGE of the plan residual between re-plans, in normalizer sigmas.

    Wire with a NEGATIVE weight. Not decoration either: the measured conclusion about
    perturbing this tracker's reference is that a perturbation "should ramp with look-ahead
    and be drawn per re-plan, not per step" (``window.REFERENCE_WINDOW_NOISE``), and this term
    is how the second half of that becomes an incentive rather than a hope. A residual that
    reverses sign every interval is a reference that jitters, which the tracker reads as noise
    it cannot follow.

    The difference is kept by the COMMAND TERM, not stashed here: the term is what applies
    the residual, and it is what has a reset hook -- so an env's first interval after a reset
    is compared against an unadjusted plan rather than against the previous episode's
    residual, which would charge the policy once per episode for a change it did not make.
    A reward term holding its own episode state would have no way to know a reset happened.

    Args:
        env: IsaacLab environment.
        command_name: The :class:`FmTrackingCommand` term name.

    Returns:
        ``[num_envs]``, non-negative.
    """
    cmd = env.command_manager.get_term(command_name)
    return _to_sigma(cmd, cmd.plan_residual_delta()).pow(2).mean(dim=(1, 2))


def episode_ended_by(env: ManagerBasedRLEnv, term_keys: Sequence[str]) -> torch.Tensor:
    """1.0 where the interval ended because one of ``term_keys`` fired, else 0.0.

    The sparse half of the adjuster's reward, and it reads the env's per-term FIRST-CROSSING
    per-term record rather than the termination manager's live state.

    That distinction is the whole reason this exists instead of ``mdp.is_terminated_term``:
    terminations are evaluated at the control rate inside the interval while the reward is
    folded once at the end, so the manager's per-term dones describe the LAST control step. A
    term that fired early and stopped holding -- reaching the goal region and then being
    dragged out of it, say -- would read as never having fired.

    It also has to be per TERM rather than per aggregate. Reaching the goal and running out of
    time are BOTH truncations on this task, deliberately: only a truncation collects
    ``gamma * V(s)``, and that asymmetry is the mechanism for rewarding arrival. So they must
    be priced separately, and ``time_outs`` cannot tell them apart.

    Args:
        env: An :class:`AdjusterEnv` (anything exposing ``ended_by``).
        term_keys: Termination term names to count.

    Returns:
        ``[num_envs]`` float.

    Raises:
        RuntimeError: If the env records no per-term ending -- on a per-control-step env
            there is no interval to record over, and ``mdp.is_terminated_term`` is correct
            there.
        KeyError: On an unknown term name, naming the ones that exist. A silent miss would
            price a termination at zero and read as a policy that never fails.
    """
    ended = getattr(env, "ended_by", None)
    if ended is None:
        raise RuntimeError(
            "episode_ended_by needs an env that records which term ended each episode over a "
            "multi-control-step interval (see AdjusterEnv.ended_by). On a per-control-step "
            "env use mdp.is_terminated_term instead."
        )
    out = torch.zeros(env.num_envs, device=env.device)
    for key in term_keys:
        if key not in ended:
            raise KeyError(
                f"termination term {key!r} is not active; this env has "
                f"{sorted(ended)}. A miss here would price that ending at zero."
            )
        out = torch.maximum(out, ended[key].to(out.dtype))
    return out
