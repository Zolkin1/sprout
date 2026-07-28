# Copyright 2026 Zachary Olkin. All rights reserved.

"""
Export policy parameters and environment configuration to YAML files.
"""
import os
import yaml
import torch
import warp as wp
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

    # Prefer student observations (distillation) over policy observations
    obs_key = "student" if "student" in obs else "policy"

    params = {
        "num_obs": obs[obs_key].shape[1],
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

    # Add gait period command if available
    try:
        gait_period_term = unwrapped_env.command_manager.get_term("gait_period")
        params["gait_period_range"] = gait_period_term.cfg.gait_period_range
    except KeyError:
        pass

    # Add action scale
    try:
        params["action_scale"] = unwrapped_env.action_manager.get_term("joint_pos").cfg.scale
    except KeyError:
        pass

    # Get skill type
    try:
        # TODO: Test
        params["skill_type"] = unwrapped_env.command_manager.get_term("traj_ref").trajectory_type.value
        params["total_time"] = unwrapped_env.command_manager.get_term("traj_ref").manager.get_total_time()
    except KeyError:
        raise ValueError("Could not get skill_type and/or total_time while exporting parameters!")
        # pass

    # Add velocity command ranges
    try:
        base_vel_cfg = unwrapped_env.command_manager.get_term("base_velocity").cfg
        params["v_x_max"] = base_vel_cfg.ranges.lin_vel_x[1]
        params["v_x_min"] = base_vel_cfg.ranges.lin_vel_x[0]
        params["v_y_max"] = base_vel_cfg.ranges.lin_vel_y[1]
        params["v_y_min"] = base_vel_cfg.ranges.lin_vel_y[0]
        params["w_z_max"] = base_vel_cfg.ranges.ang_vel_z[1]
        params["w_z_min"] = base_vel_cfg.ranges.ang_vel_z[0]
        if hasattr(base_vel_cfg, "max_acc"):
            params["max_acc"] = base_vel_cfg.max_acc
    except KeyError:
        pass

    # Add robot joint information
    params["default_joint_angles"] = wp.to_torch(robot.data.default_joint_pos)[0].detach().cpu().numpy()
    params["joint_names_isaac"] = robot.data.joint_names

    # CLF output joint order (the joint:* channels of the trajectory command's
    # ordered_pos_output_names). The joint_pos_ref / joint_vel_ref observations are
    # in THIS order, which differs from joint_names_isaac, so sim2sim / hardware
    # deployment (transfer/sim/gen_tracker_policy.py) needs it to place those terms.
    try:
        traj_cmd = unwrapped_env.command_manager.get_term("traj_ref")
        params["joint_out_names"] = [
            n.split(":", 1)[1] for n in traj_cmd.ordered_pos_output_names if n.startswith("joint:")
        ]
        # End-effector reference frames (the ee_ref output frames), in the single
        # source-of-truth order. Needed for tracker-only (no-FM) sim2sim clip playback
        # (transfer/sim/gen_tracker_runner.py --clip), where the ee frame names cannot
        # come from the FM context; matches the FM ctx.target_frame_names when an FM is used.
        if hasattr(traj_cmd, "resolve_ee_frames"):
            params["ee_frame_names"] = list(traj_cmd.resolve_ee_frames())
    except (KeyError, AttributeError):
        pass

    # FM generator re-inference cadence (control steps). Only the generator-tracking
    # command carries it; baking it into the metadata lets sim2sim/hardware re-run the
    # generator at the same cadence the tracker trained with instead of a runner default.
    try:
        params["fm_update_every"] = int(
            unwrapped_env.command_manager.get_term("traj_ref").cfg.fm_update_every
        )
    except (KeyError, AttributeError):
        pass

    # Build valid initial condition vectors (base first, then joints in joint_names order)
    root_pos = wp.to_torch(robot.data.root_pos_w)[0]        # (3,)
    root_quat = wp.to_torch(robot.data.root_quat_w)[0]      # (4,) x,y,z,w
    root_lin_vel = wp.to_torch(robot.data.root_lin_vel_w)[0] # (3,)
    root_ang_vel = wp.to_torch(robot.data.root_ang_vel_w)[0] # (3,)
    joint_pos = wp.to_torch(robot.data.joint_pos)[0]         # (num_joints,)
    joint_vel = wp.to_torch(robot.data.joint_vel)[0]         # (num_joints,)

    params["valid_ic_pos"] = torch.cat([root_pos, root_quat, joint_pos]).detach().cpu().numpy()
    params["valid_ic_vel"] = torch.cat([root_lin_vel, root_ang_vel, joint_vel]).detach().cpu().numpy()

    # Get base kp/kd from config if available, otherwise use current values
    # Note: robot.data values may be randomized during training
    if hasattr(robot, 'cfg') and hasattr(robot.cfg, 'actuators'):
        # Try to extract base gains from configuration in the correct joint order
        try:
            import re
            joint_names = robot.data.joint_names
            kp_list = []
            kd_list = []

            # Iterate through each joint and find its actuator group
            for joint_name in joint_names:
                found = False
                # Check each actuator group
                for actuator_name, actuator_cfg in robot.cfg.actuators.items():
                    # Check if this joint matches any of the patterns in this actuator group
                    for pattern in actuator_cfg.joint_names_expr:
                        if re.match(pattern.replace(".*", ".*?"), joint_name):
                            # Get stiffness for this joint
                            if isinstance(actuator_cfg.stiffness, dict):
                                # Find matching pattern in stiffness dict
                                kp_val = None
                                for stiff_pattern, stiff_val in actuator_cfg.stiffness.items():
                                    if re.match(stiff_pattern.replace(".*", ".*?"), joint_name):
                                        kp_val = stiff_val
                                        break
                                if kp_val is None:
                                    raise ValueError(f"No stiffness found for joint {joint_name}")
                                kp_list.append(kp_val)
                            else:
                                kp_list.append(actuator_cfg.stiffness)

                            # Get damping for this joint
                            if isinstance(actuator_cfg.damping, dict):
                                # Find matching pattern in damping dict
                                kd_val = None
                                for damp_pattern, damp_val in actuator_cfg.damping.items():
                                    if re.match(damp_pattern.replace(".*", ".*?"), joint_name):
                                        kd_val = damp_val
                                        break
                                if kd_val is None:
                                    raise ValueError(f"No damping found for joint {joint_name}")
                                kd_list.append(kd_val)
                            else:
                                kd_list.append(actuator_cfg.damping)

                            found = True
                            break
                    if found:
                        break

                if not found:
                    raise ValueError(f"No actuator found for joint {joint_name}")

            params["kp"] = kp_list
            params["kd"] = kd_list
        except (AttributeError, KeyError, ValueError) as e:
            print(f"[WARNING] Could not extract base gains from config: {e}")
            # Fall back to current values if config extraction fails
            params["kp"] = wp.to_torch(robot.data.joint_stiffness)[0].detach().cpu().numpy()
            params["kd"] = wp.to_torch(robot.data.joint_damping)[0].detach().cpu().numpy()
    else:
        params["kp"] = wp.to_torch(robot.data.joint_stiffness)[0].detach().cpu().numpy()
        params["kd"] = wp.to_torch(robot.data.joint_damping)[0].detach().cpu().numpy()

    # Custom YAML representer for lists to use flow style for numerical lists
    class FlowStyleList(list):
        pass

    def represent_flow_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(FlowStyleList, represent_flow_list)

    # Add observation information (prefer student group for distillation, else policy)
    obs_manager = unwrapped_env.observation_manager
    obs_info = {}
    obs_group_name = "student" if "student" in obs_manager.active_terms else "policy"

    for group_name, group_terms in obs_manager.active_terms.items():
        # Only include the target observation group
        if group_name != obs_group_name:
            continue

        obs_info["policy"] = {}
        # Get the dimensions and configs for this group
        group_dims = obs_manager._group_obs_term_dim[group_name]
        group_cfgs = obs_manager._group_obs_term_cfgs[group_name]

        for term_name, term_dim, term_cfg in zip(group_terms, group_dims, group_cfgs):
            # Convert shape to scalar if single dimension, otherwise keep as list
            if len(term_dim) == 1:
                shape_value = term_dim[0]
            else:
                shape_value = list(term_dim)

            term_info = {
                "shape": shape_value,
            }

            # Try to get scale information if available, otherwise default to 1.0
            if hasattr(term_cfg, 'scale') and term_cfg.scale is not None:
                term_info["scale"] = term_cfg.scale
            else:
                term_info["scale"] = 1.0

            # Export history_length if present (defaults to 0 = no history)
            history_len = getattr(term_cfg, 'history_length', 0)
            if history_len > 0:
                term_info["history_length"] = history_len

            obs_info["policy"][term_name] = term_info

    params["observation_terms"] = obs_info

    # Height-scan sensor geometry so the MuJoCo / hardware deploy can REPLICATE the scan
    # that both the `height_scan` obs term and the FM height token consume. Read from the
    # scene's RayCasterCfg (resolution/size/offset) + the height_scan obs term's
    # mdp.height_scan `offset` and clip. None when the task has no height scanner.
    hs_spec = None
    scene = unwrapped_env.scene
    scanner = scene.sensors.get("height_scanner") if hasattr(scene, "sensors") else None
    if scanner is not None:
        rc = scanner.cfg  # RayCasterCfg
        pat = rc.pattern_cfg
        term_cfgs = obs_manager._group_obs_term_cfgs.get(obs_group_name, [])
        term_names = obs_manager.active_terms.get(obs_group_name, [])
        hs_cfg = next((c for n, c in zip(term_names, term_cfgs) if n == "height_scan"), None)
        mdp_offset, clip = 0.5, None
        if hs_cfg is not None:
            mdp_offset = float((hs_cfg.params or {}).get("offset", 0.5))
            clip = list(hs_cfg.clip) if getattr(hs_cfg, "clip", None) is not None else None
        hs_spec = {
            "resolution": float(pat.resolution),
            "size": [float(pat.size[0]), float(pat.size[1])],
            "offset_pos": [float(rc.offset.pos[0]), float(rc.offset.pos[1]), float(rc.offset.pos[2])],
            "ray_alignment": str(getattr(rc, "ray_alignment", "yaw")),
            "sensor_offset": mdp_offset,   # mdp.height_scan subtracts this
            "clip": clip,                  # obs-term clip (apply in deploy to match training)
        }
    params["height_scan_spec"] = hs_spec

    # Convert to serializable format
    params = convert_to_serializable(params)

    # Helper function to recursively convert lists to FlowStyleList
    def convert_lists_to_flow_style(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    obj[key] = FlowStyleList(value)
                elif isinstance(value, dict):
                    convert_lists_to_flow_style(value)
        return obj

    # Convert numerical lists to flow style
    if 'default_joint_angles' in params:
        params['default_joint_angles'] = FlowStyleList(params['default_joint_angles'])
    if 'kp' in params:
        params['kp'] = FlowStyleList(params['kp'])
    if 'kd' in params:
        params['kd'] = FlowStyleList(params['kd'])
    if 'joint_names_isaac' in params:
        params['joint_names_isaac'] = FlowStyleList(params['joint_names_isaac'])
    if 'joint_out_names' in params:
        params['joint_out_names'] = FlowStyleList(params['joint_out_names'])
    if 'gait_period_range' in params:
        params['gait_period_range'] = FlowStyleList(params['gait_period_range'])
    if 'valid_ic_pos' in params:
        params['valid_ic_pos'] = FlowStyleList(params['valid_ic_pos'])
    if 'valid_ic_vel' in params:
        params['valid_ic_vel'] = FlowStyleList(params['valid_ic_vel'])

    # Convert all scale lists in observation_terms to flow style
    if 'observation_terms' in params:
        for group_name, group_terms in params['observation_terms'].items():
            for term_name, term_info in group_terms.items():
                if 'scale' in term_info and isinstance(term_info['scale'], list):
                    term_info['scale'] = FlowStyleList(term_info['scale'])

    # Save to YAML file
    yaml_path = os.path.join(export_dir, "policy_parameters.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"[INFO] Exported policy parameters to {yaml_path}")
