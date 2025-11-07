from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import generated_commands
from isaaclab.managers import SceneEntityCfg


def base_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    base_z = env.scene["robot"].data.root_pos_w[:, 2]
    return base_z.unsqueeze(-1)


def contact_state(env: ManagerBasedRLEnv, sensor_cfg, threshold: float = 50.0) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    contact_flag = (torch.abs(net_forces) > threshold).float()
    # reshape from num_env, num_bodies, 3 to num_env, num_bodies*3
    return contact_flag.reshape(env.num_envs, -1)


def step_duration(env: ManagerBasedRLEnv, command_name) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    step_duration = cmd.T
    return step_duration.unsqueeze(-1)


def step_location(env: ManagerBasedRLEnv, command_name) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    step_location = cmd.foot_target[:, 0:2]
    return step_location


def foot_vel(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_vel = cmd.robot.data.body_lin_vel_w[:, cmd.feet_bodies_idx[0], :]
    right_foot_vel = cmd.robot.data.body_lin_vel_w[:, cmd.feet_bodies_idx[1], :]

    foot_vel = torch.cat([left_foot_vel, right_foot_vel], dim=-1)

    return foot_vel


def foot_ang_vel(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:, cmd.feet_bodies_idx[0], :]
    right_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:, cmd.feet_bodies_idx[1], :]

    foot_ang_vel = torch.cat([left_foot_ang_vel, right_foot_ang_vel], dim=-1)

    return foot_ang_vel


def ref_traj(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_out.clone()
    ref_traj[:, 8] *= 50.0
    return ref_traj


def act_traj(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act.clone()
    act_traj[:, 8] *= 50.0
    return act_traj


def foot_ref_traj(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_out[:, 8]
    return ref_traj.unsqueeze(-1)


def foot_act_traj(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act[:, 8]
    return act_traj.unsqueeze(-1)


def ref_traj_vel(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj_vel = cmd.dy_out
    return ref_traj_vel


def act_traj_vel(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj_vel = cmd.dy_act
    return act_traj_vel


def joint_pos_des(env: ManagerBasedRLEnv, cmd_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(cmd_name)
    joint_pos_des = cmd.joint_pos_des
    return joint_pos_des


def v_dot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    v_dot = cmd.vdot.unsqueeze(-1)
    return v_dot


def v(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    v = cmd.v.unsqueeze(-1)
    return v


def ref_sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    phase = 2 * torch.pi * cmd.tp
    sphase = torch.sin(phase)
    if sphase.ndim == 1:
        # [B] → [B, 1]
        sphase = sphase.unsqueeze(-1)

    return sphase


def ref_cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    phase = 2 * torch.pi * cmd.tp
    cphase = torch.cos(phase)
    if cphase.ndim == 1:
        # [B] → [B, 1]
        cphase = cphase.unsqueeze(-1)
    return cphase


def sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    period = generated_commands(env, command_name).clone()

    phase = 2 * torch.pi * (env.sim.current_time / period)
    sphase = torch.sin(phase).unsqueeze(-1)

    return sphase


def cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    period = env.command_manager.get_command(command_name).clone()

    phase = 2 * torch.pi * (env.sim.current_time / period)
    cphase = torch.cos(phase).unsqueeze(-1)

    return cphase

def sincos_phase_batched(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)

    phase = cmd.phase_var.tau + cmd.stance_idx 
    sphase = torch.sin(torch.pi * phase).unsqueeze(-1)
    cphase = torch.cos(torch.pi * phase).unsqueeze(-1)

    return torch.cat([sphase, cphase], dim=-1)

def stones_position(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    curr_stone_pos = cmd.current_stone_pos
    next_stone_pos = cmd.next_stone_pos 
    diff = next_stone_pos - curr_stone_pos

    # return torch.cat([curr_stone_pos, next_stone_pos], dim=-1)
    return diff

def foot_positions_in_base(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    """Foot positions relative to base frame."""
    cmd = env.command_manager.get_term(command_name)
    
    # Get foot positions in world frame
    left_foot_pos_w = cmd.robot.data.body_pos_w[:, cmd.feet_bodies_idx[0], :]
    right_foot_pos_w = cmd.robot.data.body_pos_w[:, cmd.feet_bodies_idx[1], :]
    
    # Get base position and quaternion
    base_pos_w = cmd.robot.data.root_pos_w
    base_quat_w = cmd.robot.data.root_quat_w
    
    # Transform to base frame
    from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse
    
    # Get relative positions
    left_foot_pos_rel = left_foot_pos_w - base_pos_w
    right_foot_pos_rel = right_foot_pos_w - base_pos_w
    
    # Rotate to base frame
    left_foot_pos_base = quat_apply_inverse(base_quat_w, left_foot_pos_rel)
    right_foot_pos_base = quat_apply_inverse(base_quat_w, right_foot_pos_rel)
    
    # Concatenate [left_x, left_y, left_z, right_x, right_y, right_z]
    foot_positions = torch.cat([left_foot_pos_base, right_foot_pos_base], dim=-1)
    
    return foot_positions


def foot_velocities_in_base(env: ManagerBasedRLEnv, command_name: str = "hlip_ref") -> torch.Tensor:
    """Foot velocities relative to base frame."""
    cmd = env.command_manager.get_term(command_name)
    
    # Get foot velocities in world frame
    left_foot_vel_w = cmd.robot.data.body_lin_vel_w[:, cmd.feet_bodies_idx[0], :]
    right_foot_vel_w = cmd.robot.data.body_lin_vel_w[:, cmd.feet_bodies_idx[1], :]
    
    # Get base quaternion to rotate to base frame
    base_quat_w = cmd.robot.data.root_quat_w
    
    # Rotate to base frame
    from isaaclab.utils.math import quat_apply_inverse
    
    left_foot_vel_base = quat_apply_inverse(base_quat_w, left_foot_vel_w)
    right_foot_vel_base = quat_apply_inverse(base_quat_w, right_foot_vel_w)
    
    # Concatenate [left_vx, left_vy, left_vz, right_vx, right_vy, right_vz]
    foot_velocities = torch.cat([left_foot_vel_base, right_foot_vel_base], dim=-1)
    
    return foot_velocities