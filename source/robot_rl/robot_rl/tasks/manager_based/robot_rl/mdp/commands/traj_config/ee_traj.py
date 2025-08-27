import torch, yaml
import numpy as np
from typing import List, Dict, Tuple
from isaaclab.utils.math import wrap_to_pi, quat_apply, quat_from_euler_xyz,euler_xyz_from_quat, wrap_to_pi

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.hlip_cmd import _transfer_to_local_frame, euler_rates_to_omega

def get_euler_from_quat(quat):
    euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
    euler_x = wrap_to_pi(euler_x)
    euler_y = wrap_to_pi(euler_y)
    euler_z = wrap_to_pi(euler_z)
    return torch.stack([euler_x, euler_y, euler_z], dim=-1)

def bezier_deg(
    order: int,
    tau: torch.Tensor,  # [batch], each in [0,1]
    step_dur: torch.Tensor,  # [batch]
    control_points: torch.Tensor,  # [n_dim, degree+1]
    degree: int,
) -> torch.Tensor:
    """
    Computes (for each τ in the batch) either
      • the vector‐valued Bezier position B(τ) ∈ R^{n_dim}, or
      • its time derivative B'(τ) ∈ R^{n_dim},

    where `control_points` is shared across the whole batch and has shape [n_dim, degree+1].

    Args:
      order: 0 → position, 1 → time‐derivative.
      tau:       shape [batch], clipped to [0,1].
      step_dur:  shape [batch], positive scalars.
      control_points: shape [n_dim, degree+1].
      degree:    polynomial degree (so there are `degree+1` control points).

    Returns:
      If order==0: a tensor of shape [batch, n_dim], the Bezier‐position at each τ[i].
      If order==1: a tensor of shape [batch, n_dim], the time‐derivative at each τ[i].
    """
    # 1) Clamp tau into [0,1]
    tau = torch.clamp(tau, 0.0, 1.0)  # [batch]

    # 2) Extract n_dim from control_points
    #    control_points: [n_dim, degree+1]

    if order == 1:
        # ─── DERIVATIVE CASE ────────────────────────────────────────────────────
        # We want:
        #   B'(τ) = degree * sum_{i=0..degree-1} [
        #             (CP_{i+1} - CP_i) * C(degree-1, i)
        #             * (1-τ)^(degree-1-i) * τ^i
        #          ]  / step_dur.

        # 3) Compute CP differences along the "degree+1" axis:
        #    cp_diff: [n_dim, degree], where
        #      cp_diff[:, i] = control_points[:, i+1] - control_points[:, i].
        cp_diff = control_points[:, 1:] - control_points[:, :-1]  # [n_dim, degree]

        # 4) Binomial coefficients for (degree-1 choose i), i=0..degree-1:
        #    coefs_diff: [degree].
        coefs_diff = torch.tensor(
            [_ncr(degree - 1, i) for i in range(degree)],
            dtype=control_points.dtype,
            device=control_points.device
        )  # [degree]

        # 5) Build (τ^i) and ((1-τ)^(degree-1-i)) for i=0..degree-1:
        i_vec = torch.arange(degree, device=control_points.device)  # [degree]

        #    tau_pow:     [batch, degree],  τ^i
        tau_pow = tau.unsqueeze(1).pow(i_vec.unsqueeze(0))

        #    one_minus_pow: [batch, degree], (1-τ)^(degree-1-i)
        one_minus_pow = (1 - tau).unsqueeze(1).pow((degree - 1 - i_vec).unsqueeze(0))

        # 6) Combine into a single "weight matrix" for the derivative:
        #    weight_deriv[b, i] = degree * C(degree-1, i) * (1-τ[b])^(degree-1-i) * (τ[b])^i
        #    → shape [batch, degree]
        weight_deriv = (degree
                        * coefs_diff.unsqueeze(0)        # [1, degree]
                        * one_minus_pow                 # [batch, degree]
                        * tau_pow)                       # [batch, degree]
        # Now weight_deriv: [batch, degree]

        # 7) Multiply these weights by cp_diff to get a [batch, n_dim] result:
        #    For each batch b:  B'_b =  Σ_{i=0..degree-1} weight_deriv[b,i] * cp_diff[:,i],
        #    which is exactly a mat‐mul:  weight_deriv[b,:] @ (cp_diff^T) → [n_dim].
        #
        #    cp_diff^T: [degree, n_dim], so (weight_deriv @ cp_diff^T) → [batch, n_dim].
        Bdot = torch.matmul(weight_deriv, cp_diff.transpose(0, 1))  # [batch, n_dim]

        # 8) Finally divide by step_dur:
        return Bdot / step_dur.unsqueeze(1)  # [batch, n_dim]

    else:
        # ─── POSITION CASE ────────────────────────────────────────────────────────
        # We want:
        #   B(τ) = Σ_{i=0..degree} [
        #            CP_i * C(degree, i) * (1-τ)^(degree-i) * τ^i
        #         ].

        # 3) Binomial coefficients for (degree choose i), i=0..degree:
        #    coefs_pos: [degree+1]
        coefs_pos = torch.tensor(
            [_ncr(degree, i) for i in range(degree + 1)],
            dtype=control_points.dtype,
            device=control_points.device
        )  # [degree+1]

        # 4) Build τ^i and (1-τ)^(degree-i) for i=0..degree:
        i_vec = torch.arange(degree + 1, device=control_points.device)  # [degree+1]

        #    tau_pow:        [batch, degree+1]
        tau_pow = tau.unsqueeze(1).pow(i_vec.unsqueeze(0))

        #    one_minus_pow:  [batch, degree+1]
        one_minus_pow = (1 - tau).unsqueeze(1).pow((degree - i_vec).unsqueeze(0))

        # 5) Combine into a "weight matrix" for position:
        #    weight_pos[b, i] = C(degree, i) * (1-τ[b])^(degree-i) * (τ[b])^i.
        #    → shape [batch, degree+1]
        weight_pos = (coefs_pos.unsqueeze(0)    # [1, degree+1]
                      * one_minus_pow          # [batch, degree+1]
                      * tau_pow)               # [batch, degree+1]
        # Now weight_pos: [batch, degree+1]

        # 6) Multiply by control_points to get [batch, n_dim]:
        #    For each batch b:  B_b = Σ_{i=0..degree} weight_pos[b,i] * control_points[:,i],
        #    i.e.  weight_pos[b,:]  (shape [degree+1]) @ (control_points^T) ([degree+1, n_dim]) → [n_dim].
        #
        #    So:  B = weight_pos @ control_points^T  → [batch, n_dim].
        B = torch.matmul(weight_pos, control_points.transpose(0, 1))  # [batch, n_dim]

        return B
    
def relable_ee_hand_coeffs():
    """Build relabel matrix for end effector coefficients."""
    R = np.eye(21)

    # only need to swap left and right palm coeffs
    # com pos: [1,-1,1]
    R[1, 1] = -1
    # pelvis: [-1,1,-1]
    R[3, 3] = -1
    R[5, 5] = -1
    # swing_foot_pos:[1,-1,1]
    R[7, 7] = -1
    # swing_foot_or: [-1,1,-1]
    R[9, 9] = -1
    R[11, 11] = -1
    # waist yaw
    R[12, 12] = -1
    # swing_hand_pos: [1,-1,1]
    R[14, 14] = -1
    # swing_hand_ori: [-1,1,-1]
    R[16, 16] = -1
    # stance_hand_pos: [1,-1,1]
    R[18, 18] = -1
    # stance_hand_ori: [-1,1,-1]
    R[20, 20] = -1

    return R

def relable_ee_coeffs():
    """Build relabel matrix for end effector coefficients."""
    R = np.eye(21)

    # only need to swap left and right palm coeffs
    # com pos: [1,-1,1]
    R[1, 1] = -1
    # pelvis: [-1,1,-1]
    R[3, 3] = -1
    R[5, 5] = -1
    # swing_foot_pos:[1,-1,1]
    R[7, 7] = -1
    # swing_foot_or: [-1,1,-1]
    R[9, 9] = -1
    R[11, 11] = -1
    # waist yaw
    R[12, 12] = -1

    #swap arm coeffs
    arm_offset = 12 + 1
    left_arm = arm_offset + np.array([0, 1, 2, 3])
    right_arm = arm_offset + np.array([4, 5, 6, 7])

    tmp = R[left_arm, :].copy()
    R[left_arm, :] = R[right_arm, :]
    R[right_arm, :] = tmp

    # Sign flips: shoulder_roll, shoulder_yaw
    flip_arm = arm_offset + np.array([1, 2, 5, 6])  # left/right roll/yaw
    R[flip_arm, :] *= -1

    return R

def relable_ee_stance_coeffs():
    """Build a relabelling matrix for end effector coefficients including the stance foot."""
    R = np.eye(27)

    ##
    # COM
    ##
    # com pos: [1,-1,1]
    R[1, 1] = -1

    ##
    # Pelvis
    ##
    # pelvis: [-1,1,-1]
    R[3, 3] = -1
    R[5, 5] = -1

    ##
    # Swing foot
    ##
    # swing_foot_pos:[1,-1,1]
    R[7, 7] = -1
    # swing_foot_or: [-1,1,-1]
    R[9, 9] = -1
    R[11, 11] = -1

    ##
    # Stance Foot
    ##
    # stance_foot_pos: [1, -1, 1]
    R[13, 13] = -1
    # stance_foot_ori: [-1, 1, -1]
    R[15, 15] = -1
    R[17, 17] = -1

    ##
    # Joints
    ##
    # waist yaw
    R[18, 18] = -1

    #swap arm coeffs
    arm_offset = 18 + 1
    left_arm = arm_offset + np.array([0, 1, 2, 3])
    right_arm = arm_offset + np.array([4, 5, 6, 7])

    tmp = R[left_arm, :].copy()
    R[left_arm, :] = R[right_arm, :]
    R[right_arm, :] = tmp

    # Sign flips: shoulder_roll, shoulder_yaw
    flip_arm = arm_offset + np.array([1, 2, 5, 6])  # left/right roll/yaw
    R[flip_arm, :] *= -1

    return R

class EndEffectorTrajectoryConfig():
    """Configuration class for end effector trajectories."""
    
    def __init__(self, yaml_path="source/robot_rl/robot_rl/assets/robots/single_support_config_solution_ee.yaml"):
        self.constraint_specs = []
        self.yaml_path = yaml_path
        # Make all the dicts that will hold {domain_name: value}
        self.bezier_coeffs = {}
        self.T = {}
        self.left_coeffs = {}
        self.right_coeffs = {}
        self.bez_deg = {}  # Default Bezier degree
        self.joint_order = {}
        self.load_from_yaml()
    

    def load_from_yaml(self):
        """Load configuration from YAML file.
        
        This method loads the step period T and calls the abstract method
        _load_specific_data() for subclass-specific data loading.
        """
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        if data.get('domain_sequence') is None:
            raise ValueError("Domain sequence must be specified in the trajectory solution!")

        self.domain_seq = data['domain_sequence']
        for domain_name in self.domain_seq:
            # Load common data
            self.T[domain_name] = data[domain_name]['T'][0] if isinstance(data[domain_name]['T'], list) else data[domain_name]['T']
            self.bez_deg[domain_name] = data[domain_name]['spline_order']

            if domain_name == self.domain_seq[0]:
                # Load initial config
                init_config = data[domain_name]['q'][0]
                # Need to reorder xyzw to wxyz
                init_vel = data[domain_name]['v'][0]
                self.init_root_state = np.concatenate([init_config[:3], [init_config[6]], init_config[3:6]])  # [pos_xyz, yaw, rpy]
                self.init_root_vel = init_vel[:6]
                self.init_joint_pos = init_config[7:]
                self.init_joint_vel = init_vel[6:]

        # Load subclass-specific data
        self._load_specific_data(data)


    def _load_specific_data(self, data):
        """Load end effector specific data from YAML."""
        # Load constraint specifications

        # Read in the domain sequence
        for domain_name in self.domain_seq:
            if data[domain_name]['constraint_specs'] is None:
                raise ValueError("No constraint specs in the solution file!")

            # For now just assuming these are all the same TODO: may want to break this assumption
            self.constraint_specs = data[domain_name]['constraint_specs']

            # Reshape bezier coefficients to [num_virtual_const, num_control_points]

            ##
            # Compute the number of virtual constraints
            ##
            def count_constraint_entries(data):
                total_count = 0
                for spec in data:
                    if 'axes' in spec:
                        total_count += len(spec['axes'])
                    if 'joint_names' in spec:
                        total_count += len(spec['joint_names'])
                return total_count
            
            num_virtual_const = count_constraint_entries(self.constraint_specs)

            bezier_coeffs = data[domain_name]['bezier_coeffs']
            num_control_points = data[domain_name]['spline_order'] + 1
            bezier_coeffs_reshaped = np.array(bezier_coeffs).reshape(num_virtual_const, num_control_points)

            self.bezier_coeffs[domain_name] = bezier_coeffs_reshaped
            self.joint_order[domain_name] = data[domain_name]['joint_order']

    def get_constraint_frames(self) -> List[str]:
        """Extract frame names from constraint specs."""
        frames = []
        for spec in self.constraint_specs:
            if "frame" in spec:
                frames.append(spec["frame"])
        return frames

    def reorder_and_remap(self, cfg, device):
        """Reorder and remap end effector coefficients using hardcoded relabeling matrix."""
        # reorder for each domain
        for domain_name in self.domain_seq:
            # Load all bezier coefficients from YAML
            self.right_coeffs[domain_name] = torch.tensor(self.bezier_coeffs[domain_name], dtype=torch.float32, device=device)

            # Apply relabeling matrix to get left coefficients
            if self.bezier_coeffs[domain_name].shape[0] == 21:
                R = relable_ee_coeffs()
            elif self.bezier_coeffs[domain_name].shape[0] == 27:
                R = relable_ee_stance_coeffs()
            else:
                raise ValueError("No hard coded bezier relabelling matrix for these virtual constraints!")

            # Apply relabeling: left_coeffs = R @ right_coeffs
            left_coeffs = R @ self.bezier_coeffs[domain_name]

            self.left_coeffs[domain_name] = torch.tensor(left_coeffs, dtype=torch.float32, device=device)

            # Generate axis names for metrics
            self.generate_axis_names(domain_name)

    def get_joint_idx_list(self, hzd_cmd):
        """Get the joint index list for the given command."""
        joint_idx_list = []
        
        for joint_name in self.constraint_specs[-1]["joint_names"]:
            joint_idx, _ = hzd_cmd.robot.find_joints(joint_name)
            joint_idx_list.append(joint_idx)
        return joint_idx_list

    def generate_axis_names(self, domain_name):
        """Generate axis names for each constraint specification."""
        self.axis_names = []
        current_idx = 0
        
        for spec in self.constraint_specs:
            constraint_type = spec["type"]
            
            if constraint_type == "com_pos":
                axes = spec.get("axes", [0, 1, 2])
                axis_names = ["x", "y", "z"]
                # Generate metric names for COM position (only specified axes)
                for i, axis_idx in enumerate(axes):
                    metric_name = f"com_pos_{axis_names[axis_idx]}"
                    self.axis_names.append({
                        'name': metric_name,
                        'index': current_idx + i,
                        'domain': domain_name,
                    })
                current_idx += len(axes)
                
            elif constraint_type == "joint":
                output_dim = 1
                joint_names = spec["joint_names"]

                for joint_name in joint_names:
                    # Generate metric name for joint
                    metric_name = f"joint_{joint_name}"
                    self.axis_names.append({
                        'name': metric_name,
                        'index': current_idx,
                        'domain': domain_name,
                    })
                    current_idx += output_dim
                
            elif "frame" in spec:
                frame_name = spec["frame"]
                
                # Determine output dimension and axis names
                axes = spec.get("axes", [0, 1, 2])
                if constraint_type in ["ee_pos"]:
                    axis_names = ["x", "y", "z"]
                elif constraint_type in ["ee_ori"]:
                    axis_names = ["roll", "pitch", "yaw"]
                else:
                    axis_names = ["x", "y", "z"]
                
                # Generate metric names for each axis (only specified axes)
                for i, axis_idx in enumerate(axes):
                    metric_name = f"{frame_name}_{constraint_type}_{axis_names[axis_idx]}"
                    self.axis_names.append({
                        'name': metric_name,
                        'index': current_idx + i,
                        'domain': domain_name,
                    })
                
                current_idx += len(axes)

    def _apply_swing_modifications(self, hzd_cmd, des_pos, des_vel, base_velocity):
        """Apply end effector specific swing modifications."""
        # based on yaw velocity, update com_pos_des, com_vel_des, foot_target, foot_vel_des

        #if standing, don't modify yaw
        delta_psi = base_velocity[:, 2] * hzd_cmd.cur_swing_time

        if hzd_cmd.use_standing:
            #5,11
            stand_idx = torch.where(torch.norm(base_velocity, dim=1) < hzd_cmd.standing_threshold)[0]
            if stand_idx.numel() > 0:
                delta_psi[stand_idx] = 0
                base_velocity[stand_idx,2] = 0

        des_pos[:, hzd_cmd.yaw_output_idx] += delta_psi.unsqueeze(-1)
        des_vel[:, hzd_cmd.yaw_output_idx] += base_velocity[:, 2].unsqueeze(-1)

        q_delta_yaw = quat_from_euler_xyz(
            torch.zeros_like(delta_psi),               # roll=0
            torch.zeros_like(delta_psi),               # pitch=0
            delta_psi                                  # yaw=Δψ
        ) 

        #adjust foot target and com pos/vel to account for yaw change
        des_pos[:,[6,7,8]] = quat_apply(q_delta_yaw, des_pos[:,[6,7,8]])  # [B,3]
        des_vel[:,[6,7,8]] = quat_apply(q_delta_yaw, des_vel[:,[6,7,8]])  # [B,3]

        des_pos[:,[0,1,2]] = quat_apply(q_delta_yaw, des_pos[:,[0,1,2]])  # [B,3]
        des_vel[:,[0,1,2]] = quat_apply(q_delta_yaw, des_vel[:,[0,1,2]])  # [B,3]

        delta_y = base_velocity[:, 1] * hzd_cmd.cur_swing_time
        des_pos[:, hzd_cmd.foot_y_output_idx] += delta_y
        des_vel[:, hzd_cmd.foot_y_output_idx] += base_velocity[:, 1]

        for i in hzd_cmd.ori_idx_list:
            des_vel[:, i] = euler_rates_to_omega(des_pos[:, i], des_vel[:, i])

        return des_pos, des_vel

    def get_actual_traj(self, hzd_cmd):
        """Get actual trajectory from end effector tracker."""
        data = hzd_cmd.robot.data
        
        # Determine swing foot frame name based on stance
        # If stance_idx == 0 (left stance), then right foot is swing foot
        # If stance_idx == 1 (right stance), then left foot is swing foot
        swing_foot_idx = hzd_cmd.feet_bodies_idx[1] if hzd_cmd.stance_idx == 0 else hzd_cmd.feet_bodies_idx[0]
        stance_foot_idx = hzd_cmd.feet_bodies_idx[0] if hzd_cmd.stance_idx == 0 else hzd_cmd.feet_bodies_idx[1]

        # Get stance foot pos and velocity for relative positioning
        relative_foot_pos = hzd_cmd.stance_foot_pos_0
        
        # Get actual values for each constraint specification in order

        ##
        # COM virtual constraint
        ##
        com2stance_foot = hzd_cmd.robot.data.root_com_pos_w - relative_foot_pos
        com2stance_local = _transfer_to_local_frame(com2stance_foot, hzd_cmd.stance_foot_ori_quat_0)
       
        com_vel_w = hzd_cmd.robot.data.root_com_vel_w[:, 0:3]
        com_vel_local = _transfer_to_local_frame(com_vel_w, hzd_cmd.stance_foot_ori_quat_0)

        ##
        # Pelvis virtual constraint
        ##
        pelvis_ori = get_euler_from_quat(hzd_cmd.robot.data.root_quat_w)
        pelvis_ori[:, 2] = wrap_to_pi(pelvis_ori[:, 2] - hzd_cmd.stance_foot_ori_0[:, 2])
        pelvis_omega = hzd_cmd.robot.data.root_ang_vel_b

        def _pos_ori_vel_virtual(idx, frame_rel_pos, frame_rel_quat, frame_rel_ori):
            """Compute the locations of a given frame relative to another frame."""
            frame_pos = data.body_pos_w[:, idx, :]
            frame_quat = data.body_quat_w[:, idx, :]
            frame_ori = get_euler_from_quat(frame_quat)

            frame_pos_rel = frame_pos - frame_rel_pos
            frame_pos_rel_local = _transfer_to_local_frame(frame_pos_rel, frame_rel_quat)

            frame_ori_rel = frame_ori
            frame_ori_rel[:, 2] = wrap_to_pi(frame_ori_rel[:, 2] - frame_rel_ori[:, 2])

            frame_lin_vel_w = data.body_lin_vel_w[:, idx, :]
            frame_ang_vel_w = data.body_ang_vel_w[:, idx, :]

            frame_vel_local = _transfer_to_local_frame(frame_lin_vel_w, frame_rel_quat)
            frame_ang_vel_local= _transfer_to_local_frame(frame_ang_vel_w, frame_rel_quat)

            return frame_pos_rel_local, frame_ori_rel, frame_vel_local, frame_ang_vel_local

        ##
        # Swing foot virtual constraints
        ##
        swing_pos_rel, swing_ori_rel, swing_vel_rel, swing_ang_vel_rel = _pos_ori_vel_virtual(
            swing_foot_idx, relative_foot_pos, hzd_cmd.stance_foot_ori_quat_0, hzd_cmd.stance_foot_ori_0)

        ##
        # Joint virtual constraints
        ##
        joint_pos  = hzd_cmd.robot.data.joint_pos[:, hzd_cmd.joint_idx_list]
        joint_vel = hzd_cmd.robot.data.joint_vel[:, hzd_cmd.joint_idx_list]

        ##
        # Stance foot virtual constraints
        ##
        if self.bezier_coeffs[self.domain_seq[0]].shape[0] == 27:
            stance_pos_rel, stance_ori_rel, stance_vel_rel, stance_ang_vel_rel = _pos_ori_vel_virtual(
                stance_foot_idx, relative_foot_pos, hzd_cmd.stance_foot_ori_quat_0, hzd_cmd.stance_foot_ori_0)

            # concatenate all the position values
            y_act = torch.cat([com2stance_local, pelvis_ori, swing_pos_rel, swing_ori_rel,
                               stance_pos_rel, stance_ori_rel, joint_pos.squeeze(-1)], dim=-1)

            # concatenate all the velocity values
            dy_act = torch.cat([com_vel_local, pelvis_omega, swing_vel_rel, swing_ang_vel_rel,
                                stance_vel_rel, stance_ang_vel_rel, joint_vel.squeeze(-1)], dim=-1)

            return y_act, dy_act

        # concatenate all the position values
        y_act = torch.cat([com2stance_local, pelvis_ori, swing_pos_rel, swing_ori_rel,
                          joint_pos.squeeze(-1)], dim=-1)

        # concatenate all the velocity values
        dy_act = torch.cat([com_vel_local, pelvis_omega, swing_vel_rel, swing_ang_vel_rel,
                           joint_vel.squeeze(-1)], dim=-1)
        
        return y_act, dy_act

    def get_ref_traj(self, hzd_cmd) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get reference trajectory based on stance.
        
        Args:
            hzd_cmd: HZD command object containing stance information
            
        Returns:
            Tuple of (reference_position, reference_velocity) tensors
        """
        base_velocity = hzd_cmd.env.command_manager.get_command("base_velocity")
        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)
        
        # Choose coefficients based on stance
        if hzd_cmd.stance_idx == 1:
            ctrl_points = self.right_coeffs
        else:
            ctrl_points = self.left_coeffs
        
        phase_var_tensor = torch.full((N,), hzd_cmd.phase_var, device=hzd_cmd.device)
        
        # Import here to avoid circular imports
        from .jt_traj import bezier_deg
        
        des_pos = bezier_deg(
            0, phase_var_tensor, T, ctrl_points,
            torch.tensor(hzd_cmd.cfg.bez_deg, device=hzd_cmd.device)
        )
        
        des_vel = bezier_deg(1, phase_var_tensor, T, ctrl_points, hzd_cmd.cfg.bez_deg)
        
        # Apply stance-specific modifications
        self._apply_swing_modifications(hzd_cmd, des_pos, des_vel, base_velocity)
        
        return des_pos, des_vel
    

    def get_stance_foot_pose(self, hzd_cmd):
        """Get stance foot pose data.
        
        This method can be overridden by subclasses that need different
        stance foot pose handling.
        
        Args:
            hzd_cmd: HZD command object
        """
        # Default implementation - can be overridden
        pass
    
    def get_control_points(self, hzd_cmd) -> torch.Tensor:
        """Get control points for the current stance.
        
        Args:
            hzd_cmd: HZD command object
            
        Returns:
            Control points tensor of shape [num_env, num_outputs, degree+1]
        """
        base_velocity = hzd_cmd.env.command_manager.get_command("base_velocity")
        N = base_velocity.shape[0]
        
        # Choose coefficients based on stance
        if hzd_cmd.stance_idx == 1:
            ctrl_points = self.right_coeffs
        else:
            ctrl_points = self.left_coeffs
        
        # Expand to batch size: [num_outputs, degree+1] -> [N, num_outputs, degree+1]
        return ctrl_points.unsqueeze(0).expand(N, -1, -1)
    
    def get_phase(self, hzd_cmd) -> torch.Tensor:
        """Get current phase variable.
        
        Args:
            hzd_cmd: HZD command object
            
        Returns:
            Phase tensor of shape [num_env]
        """
        base_velocity = hzd_cmd.env.command_manager.get_command("base_velocity")
        N = base_velocity.shape[0]
        return torch.full((N,), hzd_cmd.phase_var, device=hzd_cmd.device)
    
    def get_step_duration(self, hzd_cmd) -> torch.Tensor:
        """Get step duration.
        
        Args:
            hzd_cmd: HZD command object
            
        Returns:
            Step duration tensor of shape [num_env]
        """
        base_velocity = hzd_cmd.env.command_manager.get_command("base_velocity")
        N = base_velocity.shape[0]
        return torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device) 