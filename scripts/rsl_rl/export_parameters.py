# Copyright 2026 Zachary Olkin. All rights reserved.

"""
Export policy parameters and environment configuration to YAML files.
"""
import os
import yaml
import torch
import warp as wp
import numpy as np

from robot_rl.flow_matching.window import BASE_POSE_CONVENTION
from robot_rl.tasks.manager_based.robot_rl import mdp


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


def resolve_actor_obs_groups(obs, actor_obs_groups=None):
    """Ordered observation groups the exported policy consumes.

    Args:
        obs: Observation dict / TensorDict from the environment.
        actor_obs_groups: Explicit ordered group list, normally
            ``agent_cfg.obs_groups["actor"]``. When None, falls back to the
            single-group default: the distillation ``student`` group if present,
            else ``policy``.

    Returns:
        List of group names, in the order the actor concatenates them.

    Raises:
        ValueError: If a named group is absent from the observations.
    """
    if actor_obs_groups is None:
        return ["student" if "student" in obs else "policy"]
    groups = list(actor_obs_groups)
    missing = [g for g in groups if g not in obs]
    if missing:
        raise ValueError(
            f"Actor observation groups {missing} are not produced by the environment "
            f"(available: {sorted(obs.keys())})."
        )
    return groups


def _find_term_cfg(obs_manager, obs_group_names, term_name):
    """Return the :class:`ObservationTermCfg` for ``term_name``, or None if absent.

    Searches the actor's groups in order and stops at the first hit.  The exported term
    table already rejects a name appearing in more than one group, so the first hit is
    the only hit.

    Args:
        obs_manager: The environment's observation manager.
        obs_group_names: Ordered observation groups the actor consumes.
        term_name: Observation term to look up.

    Returns:
        The term's config object, or None when no actor group has that term.
    """
    for group_name in obs_group_names:
        term_names = obs_manager.active_terms.get(group_name, [])
        if term_name not in term_names:
            continue
        return obs_manager._group_obs_term_cfgs[group_name][term_names.index(term_name)]
    return None


def _window_specs(unwrapped_env, obs_manager, obs_group_names, term_table):
    """Deploy contract for the ``reference_window`` / ``command_window`` observations.

    The deploy has to rebuild both terms from the generator's prediction, which needs
    the sample offsets and -- for the reference window -- the per-step block layout.
    Neither is recoverable from the flat term width, so both are exported here.  The
    layout comes from :func:`mdp.reference_window_layout`, the single source of truth
    the observation itself uses, rather than being restated.

    Args:
        unwrapped_env: The unwrapped Isaac Lab environment.
        obs_manager: The environment's observation manager.
        obs_group_names: Ordered observation groups the actor consumes.
        term_table: The exported ``observation_terms["policy"]`` mapping, used to
            recover the command width from the term's own flat shape.

    Returns:
        ``(reference_window_spec, command_window_spec)``; either is None when the
        actor does not observe that term.
    """
    ref_cfg = _find_term_cfg(obs_manager, obs_group_names, "reference_window")
    cmd_cfg = _find_term_cfg(obs_manager, obs_group_names, "command_window")
    if ref_cfg is None and cmd_cfg is None:
        return None, None

    def _offsets(cfg):
        """The term's own sample offsets in CONTROL steps, verbatim.

        Read off the live term rather than rebuilt from a (count, stride) pair: the window
        is deliberately non-uniform (it samples offsets 0, 1, 2 and then every 5th step),
        so no stride describes it and there is nothing to reconstruct it from.
        """
        return [int(o) for o in (cfg.params or {})["offsets"]]

    ref_spec = None
    if ref_cfg is not None:
        offs = _offsets(ref_cfg)
        cmd_term = unwrapped_env.command_manager.get_term((ref_cfg.params or {}).get("command_name", "traj_ref"))
        ref_spec = {
            "offsets": offs,
            # [[block_name, width], ...] for ONE step; the term is these blocks
            # concatenated, then the steps concatenated. The term's OWN selection, not
            # the full five: a teacher that drops a block observes a narrower window, and
            # a generator paired with it has to predict exactly this list
            # (checked in GenTrackerPolicy.load against fm_meta's window_blocks).
            "blocks": [[name, int(width)] for name, width in mdp.reference_window_layout(
                cmd_term, (ref_cfg.params or {}).get("blocks"))],
            # The window's joint blocks are in the ROBOT's own joint order (so they line
            # up elementwise with the joint_pos / joint_vel / actions proprioception
            # terms), NOT the `joint_out_names` order that joint_pos_ref used. Recorded
            # explicitly because the two orders differ and nothing else here says which.
            "joint_order": "isaac",
            "ee_frame_names": list(cmd_term.resolve_ee_frames(None)),
            # Direction and frame of the base_pose block's two difference terms. Recorded
            # because the SIGN of the orientation half is invisible in the obs shape: a
            # policy trained under either convention loads clean against the other and
            # walks sideways. GenTrackerPolicy.load() requires this key and refuses
            # anything it does not know how to reproduce.
            "base_pose_convention": BASE_POSE_CONVENTION,
        }

    cmd_spec = None
    if cmd_cfg is not None:
        offs = _offsets(cmd_cfg)
        flat = term_table["command_window"]["shape"]
        flat = int(np.prod(flat)) if isinstance(flat, (list, tuple)) else int(flat)
        cmd_spec = {"offsets": offs, "cmd_dim": flat // len(offs)}

    return ref_spec, cmd_spec


def export_policy_parameters(env, obs, actions, save_dir, actor_obs_groups=None):
    """
    Export policy parameters and environment configuration to a YAML file.

    Args:
        env: The wrapped Isaac Lab environment
        obs: Current observations tensor
        actions: Current actions tensor
        save_dir: Base directory (will create 'exported' subfolder)
        actor_obs_groups: Ordered observation groups the actor consumes
            (``agent_cfg.obs_groups["actor"]``). A multi-group actor concatenates
            them in this order, and the deploy rebuilds the flat observation vector
            from the exported term table -- so the table must list every group's
            terms in the same order. None keeps the legacy single-group behaviour.
    """
    # Create exported subfolder
    export_dir = os.path.join(save_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene.articulations["robot"]

    obs_group_names = resolve_actor_obs_groups(obs, actor_obs_groups)

    params = {
        # Total width the exported network expects on its single flat `obs` input.
        "num_obs": int(sum(obs[g].shape[1] for g in obs_group_names)),
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

    # Get skill type. AttributeError as well as KeyError: a command term that exists but
    # carries no `trajectory_type` / `manager` raises the former, and letting it escape
    # kills the whole parameter export over two optional fields.
    try:
        traj = unwrapped_env.command_manager.get_term("traj_ref")
        params["skill_type"] = traj.trajectory_type.value
        params["total_time"] = traj.manager.get_total_time()
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"Could not get skill_type and/or total_time while exporting parameters ({exc})."
        ) from exc

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

    # Add robot joint information.
    #
    # NOMINAL, not env 0's: add_joint_default_pos randomizes default_joint_pos in place at
    # startup to model calibration error, so the live buffer holds whatever env 0 happened to
    # draw. Shipping that put a per-joint offset of up to the DR half-width straight into the
    # deployed PD target -- verified in depth_v5_refslim's exported policy_parameters.yaml,
    # where 0 of 29 angles were exactly 0.0 despite 9 joints being authored at 0.0.
    # The event stashes the pre-randomization values for exactly this; same config-over-runtime
    # rule the kp/kd block below follows.
    nominal_default_pos = getattr(robot.data, "default_joint_pos_nominal", None)
    if nominal_default_pos is None:
        # No DR event ran (e.g. events disabled), so the live buffer IS nominal.
        nominal_default_pos = wp.to_torch(robot.data.default_joint_pos)[0]
    params["default_joint_angles"] = nominal_default_pos.detach().cpu().numpy()
    params["joint_names_isaac"] = robot.data.joint_names

    # CLF output joint order (the joint:* channels of the trajectory command's
    # ordered_pos_output_names). The joint_pos_ref / joint_vel_ref observations are
    # in THIS order, which differs from joint_names_isaac, so sim2sim / hardware
    # deployment (transfer/deploy/gen_tracker.py) needs it to place those terms.
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
        # Body links whose ROOT-frame targets the POLICY OBSERVES, in the order the
        # body_pos_ref / body_vel_ref terms stack them -- what deploy must serve
        # (transfer/sim/clip_reference.py).  Gated on the mode, not on what the clip
        # happens to carry: a joint-mode policy trained on a widened library still has the
        # channels available but never looks at them, and advertising them would make the
        # runner build reference blocks no obs term consumes.
        mode = getattr(traj_cmd.cfg, "tracking_mode", "joint")
        params["tracking_mode"] = mode
        params["root_frame_bodies"] = (
            [f.split("@", 1)[0] for f in getattr(traj_cmd, "root_body_frames", [])]
            if mode == "body" else []
        )
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

    # Observation term table for every group the actor consumes, flattened into a single
    # ordered mapping under "policy" -- the deploy concatenates the terms in exactly this
    # order to rebuild the network's flat input vector, so the group order here must match
    # obs_groups["actor"].
    obs_manager = unwrapped_env.observation_manager
    obs_info = {"policy": {}}

    for group_name in obs_group_names:
        group_terms = obs_manager.active_terms[group_name]
        # Get the dimensions and configs for this group
        group_dims = obs_manager._group_obs_term_dim[group_name]
        group_cfgs = obs_manager._group_obs_term_cfgs[group_name]

        for term_name, term_dim, term_cfg in zip(group_terms, group_dims, group_cfgs):
            if term_name in obs_info["policy"]:
                # The table is keyed by term name, so a collision across groups would
                # silently drop one term and shift every offset after it.
                raise ValueError(
                    f"Observation term '{term_name}' appears in more than one actor group "
                    f"{obs_group_names}; rename it so the exported term table stays "
                    "unambiguous."
                )
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

            # The term's clip bounds, so the deploy applies the SAME saturation the policy
            # trained under instead of a hardcoded per-term table (which silently left any
            # clipped term missing from it unclipped).
            if getattr(term_cfg, 'clip', None) is not None:
                term_info["clip"] = [float(term_cfg.clip[0]), float(term_cfg.clip[1])]

            obs_info["policy"][term_name] = term_info

    params["observation_terms"] = obs_info

    # Merged reference / command window contract.  `observation_terms` records these two
    # only as a flat width (11 * 82 = 902) plus a per-channel scale vector, which is not
    # enough for the deploy to REBUILD them: it also needs the sample offsets and the
    # per-step block layout.  Both come from the sources the observation itself uses --
    # the term's own params and mdp.reference_window_layout -- so nothing off-sim
    # restates a width.  Absent (None) for a task without the window terms.
    ref_spec, cmd_spec = _window_specs(unwrapped_env, obs_manager, obs_group_names, obs_info["policy"])
    params["reference_window_spec"] = ref_spec
    params["command_window_spec"] = cmd_spec

    # Height-scan sensor geometry so the MuJoCo / hardware deploy can REPLICATE the scan
    # that both the `height_scan` obs term and the FM height token consume. Read from the
    # scene's RayCasterCfg (resolution/size/offset) + the height_scan obs term's
    # mdp.height_scan `offset` and clip. None when the task has no height scanner.
    hs_spec = None
    scene = unwrapped_env.scene
    scanner = scene.sensors.get("height_scanner") if hasattr(scene, "sensors") else None
    hs_cfg = None
    for group_name in obs_group_names:
        term_cfgs = obs_manager._group_obs_term_cfgs.get(group_name, [])
        term_names = obs_manager.active_terms.get(group_name, [])
        hs_cfg = next((c for n, c in zip(term_names, term_cfgs) if n == "height_scan"), None)
        if hs_cfg is not None:
            break
    # Only when the EXPORTED group actually observes the scan. A depth task keeps the
    # scanner in the scene for its (privileged, never-deployed) critic, and emitting a
    # spec for that would both invent default values here and make the deploy raycast a
    # grid the policy never reads.
    if scanner is not None and hs_cfg is not None:
        rc = scanner.cfg  # RayCasterCfg
        pat = rc.pattern_cfg
        hs_spec = {
            "resolution": float(pat.resolution),
            "size": [float(pat.size[0]), float(pat.size[1])],
            "offset_pos": [float(rc.offset.pos[0]), float(rc.offset.pos[1]), float(rc.offset.pos[2])],
            "ray_alignment": str(getattr(rc, "ray_alignment", "yaw")),
            # mdp.height_scan subtracts this
            "sensor_offset": float((hs_cfg.params or {}).get("offset", 0.5)),
            # obs-term clip (apply in deploy to match training)
            "clip": list(hs_cfg.clip) if getattr(hs_cfg, "clip", None) is not None else None,
        }
    params["height_scan_spec"] = hs_spec

    # Depth-camera geometry, optics, range and normalisation, so the MuJoCo / hardware
    # deploy can REPLICATE the depth images the policy obs consumes. Derived from the
    # sensor cfgs + observation terms by mdp.depth_camera_specs -- the same producer the
    # recorder uses, so the FM's copy of this contract cannot drift from the tracker's.
    # None when the exported group has no depth cameras. Scoped to that group for the same
    # reason height_scan_spec is: describing a camera whose obs term the deployed policy
    # never reads would have the deploy cast it every step and silently discard the result
    # (which is exactly what the `-depth-image` variant's separate obs groups would do).
    params["depth_camera_spec"] = mdp.depth_camera_specs(unwrapped_env, obs_group_names) or None

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

    # Keep each camera's short numeric lists on one line
    for cam in (params.get('depth_camera_spec') or []):
        for key in ('offset_pos', 'offset_quat_xyzw', 'clip'):
            if isinstance(cam.get(key), list):
                cam[key] = FlowStyleList(cam[key])

    # Convert scale / clip / shape lists in observation_terms to flow style
    if 'observation_terms' in params:
        for group_name, group_terms in params['observation_terms'].items():
            for term_name, term_info in group_terms.items():
                for key in ('scale', 'clip', 'shape'):
                    if isinstance(term_info.get(key), list):
                        term_info[key] = FlowStyleList(term_info[key])

    # Save to YAML file
    yaml_path = os.path.join(export_dir, "policy_parameters.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"[INFO] Exported policy parameters to {yaml_path}")
