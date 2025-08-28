from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import generated_commands

def base_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    base_z = env.scene["robot"].data.root_pos_w[:,2]
    return base_z.unsqueeze(-1)

def contact_state(env: ManagerBasedRLEnv, sensor_cfg, threshold: float = 50.0) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:,-1,sensor_cfg.body_ids,:]
    contact_flag = (torch.abs(net_forces) > threshold).float()
    #reshape from num_env, num_bodies, 3 to num_env, num_bodies*3
    return contact_flag.reshape(env.num_envs, -1)

def step_duration(env: ManagerBasedRLEnv, command_name) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    step_duration = cmd.T
    return step_duration.unsqueeze(-1)

def step_location(env: ManagerBasedRLEnv, command_name) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    step_location = cmd.foot_target[:,0:2]
    return step_location

def foot_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_vel = cmd.robot.data.body_lin_vel_w[:,cmd.feet_bodies_idx[0],:]
    right_foot_vel = cmd.robot.data.body_lin_vel_w[:,cmd.feet_bodies_idx[1],:]

    foot_vel = torch.cat([left_foot_vel, right_foot_vel], dim=-1)

    return foot_vel

def foot_ang_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:,cmd.feet_bodies_idx[0],:]
    right_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:,cmd.feet_bodies_idx[1],:]

    foot_ang_vel = torch.cat([left_foot_ang_vel, right_foot_ang_vel], dim=-1)

    return foot_ang_vel

def ref_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_out.clone()
    ref_traj[:,8] *= 50.0
    return ref_traj

def act_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act.clone()
    act_traj[:,8] *= 50.0
    return act_traj

def foot_ref_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_out[:,8]
    return ref_traj.unsqueeze(-1)

def foot_act_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act[:,8]
    return act_traj.unsqueeze(-1)

def ref_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj_vel = cmd.dy_out
    return ref_traj_vel

def act_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj_vel = cmd.dy_act
    return act_traj_vel

def joint_pos_des(env: ManagerBasedRLEnv, cmd_name:str) -> torch.Tensor:
    cmd = env.command_manager.get_term(cmd_name)
    joint_pos_des = cmd.joint_pos_des
    return joint_pos_des

def v_dot(env: ManagerBasedRLEnv, command_name:str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    v_dot = cmd.vdot.unsqueeze(-1)
    return v_dot

def v(env: ManagerBasedRLEnv, command_name:str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    v = cmd.v.unsqueeze(-1)
    return v
        

def ref_sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    phase = 2*torch.pi * cmd.gait_cycle_prop
    sphase = torch.sin(phase)
    if sphase.ndim == 1:
        # [B] → [B, 1]
        sphase = sphase.unsqueeze(-1)

    return sphase

def ref_cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    phase = 2*torch.pi * cmd.gait_cycle_prop
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

def domain_flag(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return a domain flag based on which hybrid domain the reference trajectory is in."""
    cmd = env.command_manager.get_term(command_name)

    flt = cmd.get_flight_envs()
    ssp = cmd.get_ssp_envs()
    dsp = cmd.get_dsp_envs()

    return (0*flt + 1*ssp + 2*dsp).unsqueeze(-1)
