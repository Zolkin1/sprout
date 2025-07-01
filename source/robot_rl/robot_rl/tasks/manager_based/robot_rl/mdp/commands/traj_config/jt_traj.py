import numpy as np
import yaml
import torch
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.ref_gen import _ncr
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

def get_euler_from_quat(quat):

     euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
     euler_x = wrap_to_pi(euler_x)
     euler_y = wrap_to_pi(euler_y)
     euler_z = wrap_to_pi(euler_z)
     return torch.stack([euler_x, euler_y, euler_z], dim=-1)

def build_relabel_matrix() -> np.ndarray:
        """
        Build a relabel (mirror symmetry) matrix R for the G1 humanoid model.
        Applies left/right swaps and sign flips to roll/yaw joints.

        Returns
        -------
        R : (nq, nq) numpy.ndarray
            Orthogonal permutation matrix such that q_mirrored = R @ q
        """

        R = np.eye(21)
        # ----------------------------
        # LEG relabeling
        # ----------------------------
        # Joint ordering in legs (6 DOF each)
        # [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
        left_leg = np.array([0, 1, 2, 3, 4, 5])
        right_leg = np.array([6, 7, 8, 9, 10, 11])

        tmp = R[left_leg, :].copy()
        R[left_leg, :] = R[right_leg, :]
        R[right_leg, :] = tmp

        # Sign flips: hip_roll, hip_yaw, ankle_roll
        # left and right roll/yaw
        flip_leg = np.array([1, 2, 5, 7, 8, 11])
        R[flip_leg, :] *= -1

        # flip waist yaw
        R[12, :] *= -1

        # ----------------------------
        # ARM relabeling
        # ----------------------------
        # Starts after 12 leg joints + waist_yaw (1)
        arm_offset =  12 + 1
        left_arm = arm_offset + np.array([0, 1, 2, 3])
        right_arm = arm_offset + np.array([4, 5, 6, 7])

        tmp = R[left_arm, :].copy()
        R[left_arm, :] = R[right_arm, :]
        R[right_arm, :] = tmp

        # Sign flips: shoulder_roll, shoulder_yaw
        flip_arm = arm_offset + np.array([1, 2, 5, 6])  # left/right roll/yaw
        R[flip_arm, :] *= -1

        return R

class JointTrajectoryConfig:
    def __init__(self, yaml_path="source/robot_rl/robot_rl/assets/robots/single_support_config_solution_jt.yaml"):
        self.joint_trajectories = {}
        self.base_trajectories = {}
        self.isaac_joint_indices = []  # Store Isaac indices in YAML order
        self.yaml_path = yaml_path
        self.load_from_yaml()

    
    def load_from_yaml(self):
        """Load bezier coefficients from YAML file and find corresponding Isaac joint indices."""
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Extract bezier coefficients and joint order
        bezier_coeffs = data['bezier_coeffs']
        yaml_joint_order = data['joint_order']
        spline_order = data['spline_order']
        
        # Calculate number of control points per joint
        num_control_points = spline_order + 1
        num_joints = len(yaml_joint_order)
        
        # Reshape bezier coefficients to [num_joints, num_control_points]
        bezier_coeffs_reshaped = np.array(bezier_coeffs).reshape(num_joints, num_control_points)
        
        self.bezier_coeffs = bezier_coeffs_reshaped
        self.joint_order = yaml_joint_order
        # Store coefficients in YAML order
        for i, joint_name in enumerate(yaml_joint_order):
            self.joint_trajectories[joint_name] = bezier_coeffs_reshaped[i].tolist()
        
        
        # Store step period
        self.T = data['T'][0] if isinstance(data['T'], list) else data['T']
        
        return self
    
    def remap_jt_symmetric(self):
        """Create symmetric mapping for left/right joints."""
        symmetric_mapping = {}
        #grab R from g1_r
        R = build_relabel_matrix()
        traj = R @ self.bezier_coeffs
        
        # Create mapping from joint names to their mirrored coefficients
        for i, joint_name in enumerate(self.joint_order):
            symmetric_mapping[joint_name] = traj[i].tolist()
        
        return symmetric_mapping
    

    def reorder_and_remap_jt(self,cfg,robot,device):
        right_jt_coeffs = self.joint_trajectories
    
        left_jt_coeffs = self.remap_jt_symmetric()


        left_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=device)
        right_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=device)

        
        for key in self.joint_trajectories.keys():
            joint_idx = robot.find_joints(key)[0]
            right_coeffs[joint_idx] = torch.tensor(right_jt_coeffs[key], device=device)
            left_coeffs[joint_idx] = torch.tensor(left_jt_coeffs[key], device=device)

        self.right_coeffs = right_coeffs
        self.left_coeffs = left_coeffs
        return 

    def get_ref_traj(self,jt_hzd_cmf):
        base_velocity = jt_hzd_cmf.env.command_manager.get_command("base_velocity")  # (N,2)
        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)

        if jt_hzd_cmf.stance_idx == 1:
            ctrl_points = self.right_coeffs
        else:
            ctrl_points = self.left_coeffs
     
        phase_var_tensor = torch.full((N,), jt_hzd_cmf.phase_var, device=jt_hzd_cmf.device)
        des_jt_pos = bezier_deg(
            0, phase_var_tensor, T, ctrl_points, torch.tensor(jt_hzd_cmf.cfg.bez_deg, device=jt_hzd_cmf.device)
        )
        
        des_jt_vel = bezier_deg(1, phase_var_tensor, T, ctrl_points, jt_hzd_cmf.cfg.bez_deg)

        yaw_offset = base_velocity[:, 2]
        des_jt_pos[:, jt_hzd_cmf.hip_yaw_idx[jt_hzd_cmf.stance_idx]] += yaw_offset

        return des_jt_pos, des_jt_vel


    def get_stance_foot_pose(self,jt_hzd_cmf):
        data = jt_hzd_cmf.robot.data
          # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, jt_hzd_cmf.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, jt_hzd_cmf.feet_bodies_idx, :]

        # Store raw foot positions
        foot_lin_vel_w = data.body_lin_vel_w[:, jt_hzd_cmf.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, jt_hzd_cmf.feet_bodies_idx, :]
        jt_hzd_cmf.stance_foot_pos = foot_pos_w[:, jt_hzd_cmf.stance_idx, :]
        jt_hzd_cmf.stance_foot_ori = get_euler_from_quat(foot_ori_w[:, jt_hzd_cmf.stance_idx, :])
        jt_hzd_cmf.stance_foot_vel = foot_lin_vel_w[:, jt_hzd_cmf.stance_idx, :]
        jt_hzd_cmf.stance_foot_ang_vel = foot_ang_vel_w[:, jt_hzd_cmf.stance_idx, :]  


    def get_actul_traj(self,jt_hzd_cmf):
        data = jt_hzd_cmf.robot.data

        jt_pos = data.joint_pos
        jt_vel = data.joint_vel
        # 4. Assemble state vectors
        return jt_pos, jt_vel
        

def bezier_deg(
    order: int,
    tau: torch.Tensor,            # [batch], each in [0,1]
    step_dur: torch.Tensor,       # [batch]
    control_points: torch.Tensor, # [n_dim, degree+1]
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
    tau = torch.clamp(tau, 0.0, 1.0)         # [batch]
    batch = tau.size(0)

    # 2) Extract n_dim from control_points
    #    control_points: [n_dim, degree+1]
    n_dim = control_points.shape[0]

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