"""
Export policy parameters and environment configuration to YAML files.
"""
import os
import yaml
import torch
import numpy as np


def convert_to_serializable(obj):
    """Convert torch tensors and numpy arrays to Python native types for YAML serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def export_policy_parameters(env, obs, actions, save_dir):
    """
    Export policy parameters and environment configuration to a YAML file.

    Args:
        env: The wrapped Isaac Lab environment
        obs: Current observations tensor
        actions: Current actions tensor
        save_dir: Base directory (will create 'exported' subfolder)
    """
    # Create exported subfolder
    export_dir = os.path.join(save_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene.articulations["robot"]

    params = {
        "num_obs": obs.shape[1],
        "num_actions": actions.shape[1],
        "dt": unwrapped_env.step_dt,
    }

    # Add phase information if available
    try:
        phase_term = unwrapped_env.command_manager.get_term("phase")
        params["period"] = phase_term.cfg.phase
        params["stand_threshold"] = phase_term.cfg.stand_threshold
    except KeyError:
        pass

    # Add action scale
    try:
        params["action_scale"] = unwrapped_env.action_manager.get_term("joint_pos").cfg.scale
    except KeyError:
        pass

    # Add velocity command ranges
    try:
        base_vel_cfg = unwrapped_env.command_manager.get_term("base_velocity").cfg
        params["v_x_max"] = base_vel_cfg.ranges.lin_vel_x[1]
        params["v_x_min"] = base_vel_cfg.ranges.lin_vel_x[0]
        params["v_y_max"] = base_vel_cfg.ranges.lin_vel_y[1]
        params["v_y_min"] = base_vel_cfg.ranges.lin_vel_y[0]
        params["w_z_max"] = base_vel_cfg.ranges.ang_vel_z[1]
        params["w_z_min"] = base_vel_cfg.ranges.ang_vel_z[0]
    except KeyError:
        pass

    # Add robot joint information
    params["default_joint_angles"] = robot.data.default_joint_pos[0].detach().cpu().numpy()
    params["kp"] = robot.data.joint_stiffness[0].detach().cpu().numpy()
    params["kd"] = robot.data.joint_damping[0].detach().cpu().numpy()
    params["joint_names_isaac"] = robot.data.joint_names

    # Add observation information
    obs_manager = unwrapped_env.observation_manager
    obs_info = {}

    for group_name, group_terms in obs_manager.active_terms.items():
        obs_info[group_name] = {}
        # Get the dimensions and configs for this group
        group_dims = obs_manager._group_obs_term_dim[group_name]
        group_cfgs = obs_manager._group_obs_term_cfgs[group_name]

        for term_name, term_dim, term_cfg in zip(group_terms, group_dims, group_cfgs):
            term_info = {
                "shape": list(term_dim),
            }

            # Try to get scale information if available
            if hasattr(term_cfg, 'scale') and term_cfg.scale is not None:
                term_info["scale"] = term_cfg.scale

            obs_info[group_name][term_name] = term_info

    params["observation_terms"] = obs_info

    # Convert to serializable format
    params = convert_to_serializable(params)

    # Save to YAML file
    yaml_path = os.path.join(export_dir, "policy_parameters.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Exported policy parameters to {yaml_path}")
