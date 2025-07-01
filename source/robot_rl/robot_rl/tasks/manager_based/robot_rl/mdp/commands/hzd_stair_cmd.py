import torch
import math
import yaml
from isaaclab.utils import configclass
import numpy as np

from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
from robot_rl.assets.robots.g1_21j import build_relabel_matrix
# from robot_rl.assets.robots.exo_cfg import JointTrajectoryConfig

from .clf import CLF
from .hzd_cmd import bezier_deg, JointTrajectoryConfig
# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz

from typing import TYPE_CHECKING




if TYPE_CHECKING:
    from .cmd_cfg import HZDStairCommandCfg


class HZDStairCommandTerm(CommandTerm):
    def __init__(self, cfg: "HZDStairCommandCfg", env):
        super().__init__(cfg, env)
       
        self.env = env
        self.robot = env.scene[cfg.asset_name]

        self.debug_vis = cfg.debug_vis


        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        hip_yaw_idx,_ = self.robot.find_joints(".*_hip_yaw_.*")

        batch_hip_yaw_idx = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        batch_hip_yaw_idx[:, 0] = hip_yaw_idx[0]
        batch_hip_yaw_idx[:, 1] = hip_yaw_idx[1]
        self.hip_yaw_idx = batch_hip_yaw_idx

        self.metrics = {}
        self.T = torch.zeros((self.num_envs), device=self.device)
     

        # load joint trajectory config from YAML
        #this is the flat ref trajectory
        yaml_path = "source/robot_rl/robot_rl/assets/robots/single_support_config_solution.yaml"
        self.jt_config = JointTrajectoryConfig()
        self.jt_config.load_from_yaml(yaml_path, self.robot)
        self.T_flat = self.jt_config.T
        
        #this is the stair ref trajectory
        yaml_path = "source/robot_rl/robot_rl/assets/robots/stair_config_solution.yaml"
        self.jt_config_stair = JointTrajectoryConfig()
        self.jt_config_stair.load_from_yaml(yaml_path, self.robot)
        self.T_upstair = self.jt_config_stair.T

        #load the downstair ref trajectory
        yaml_path = "source/robot_rl/robot_rl/assets/robots/downstair_config_solution.yaml"
        self.jt_config_downstair = JointTrajectoryConfig()
        self.jt_config_downstair.load_from_yaml(yaml_path, self.robot)
        self.T_downstair = self.jt_config_downstair.T

        right_jt_coeffs = self.jt_config.joint_trajectories
        left_jt_coeffs = self.jt_config.remap_jt_symmetric()
   

        left_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)
        right_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)

        
        for key in self.jt_config.joint_trajectories.keys():
            joint_idx = self.robot.find_joints(key)[0]
            right_coeffs[joint_idx] = torch.tensor(right_jt_coeffs[key], device=self.device)
            left_coeffs[joint_idx] = torch.tensor(left_jt_coeffs[key], device=self.device)

        self.right_coeffs = right_coeffs
        self.left_coeffs = left_coeffs

        # Remap stair trajectories once at initialization
        # Stair up trajectories
        stair_up_right_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)
        stair_up_left_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)
        
        for key in self.jt_config_stair.joint_trajectories.keys():
            joint_idx = self.robot.find_joints(key)[0]
            stair_up_right_coeffs[joint_idx] = torch.tensor(self.jt_config_stair.joint_trajectories[key], device=self.device)
        
        # Remap stair up for left side
        stair_up_left_remapped = self.jt_config_stair.remap_jt_symmetric()
        for key in stair_up_left_remapped.keys():
            joint_idx = self.robot.find_joints(key)[0]
            stair_up_left_coeffs[joint_idx] = torch.tensor(stair_up_left_remapped[key], device=self.device)
        
        self.stair_up_right_coeffs = stair_up_right_coeffs
        self.stair_up_left_coeffs = stair_up_left_coeffs
        
        # Stair down trajectories
        stair_down_right_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)
        stair_down_left_coeffs = torch.zeros((cfg.num_outputs, cfg.bez_deg+1), device=self.device)
        
        for key in self.jt_config_downstair.joint_trajectories.keys():
            joint_idx = self.robot.find_joints(key)[0]
            stair_down_right_coeffs[joint_idx] = torch.tensor(self.jt_config_downstair.joint_trajectories[key], device=self.device)
        
        # Remap stair down for left side
        stair_down_left_remapped = self.jt_config_downstair.remap_jt_symmetric()
        for key in stair_down_left_remapped.keys():
            joint_idx = self.robot.find_joints(key)[0]
            stair_down_left_coeffs[joint_idx] = torch.tensor(stair_down_left_remapped[key], device=self.device)
        
        self.stair_down_right_coeffs = stair_down_right_coeffs
        self.stair_down_left_coeffs = stair_down_left_coeffs

        self.mass = sum(self.robot.data.default_mass.T)[0]
        
        self.clf = CLF(
           cfg.num_outputs, self.env.cfg.sim.dt,
           batch_size=self.num_envs,
            Q_weights=np.array(cfg.Q_weights),
            R_weights=np.array(cfg.R_weights),
            device=self.device
        )
        
        self.v = torch.zeros((self.num_envs), device=self.device)
        self.stance_idx = None

        # Initialize stance-related variables
        self.stance_foot_pos_0 = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_ori_quat_0 = torch.zeros((self.num_envs, 4), device=self.device)
        self.stance_foot_ori_0 = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_ori = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        self.y_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.y_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.tp = torch.zeros((self.num_envs), device=self.device)
        self.z_height = torch.zeros((self.num_envs), device=self.device)
          
    @property
    def command(self):
        return self.y_out
    

    def _resample_command(self, env_ids):
        self._update_command()
        # Do nothing here
        return
    
    def _update_metrics(self):

        # Update metrics using actual joint names from the YAML file
        for i, joint_name in enumerate(self.robot.joint_names):
            error_key = f"error_{joint_name}"
            self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot
        

    def update_Stance_Swing_idx(self):
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)
        N = base_velocity.shape[0]
        device = base_velocity.device

        # Initialize stance_idx and stance_foot_pos_0 if first time
        if self.stance_idx is None:
            self.stance_idx = torch.full((N,), -1, dtype=torch.long, device=device)
            # Initialize stance foot position from current robot state
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            # Use right foot as initial stance (index 1)
            self.stance_foot_pos_0 = foot_pos_w[:, 1, :]
            self.stance_foot_ori_quat_0 = foot_ori_w[:, 1, :]
            self.stance_foot_ori_0 = self.get_euler_from_quat(foot_ori_w[:, 1, :])

        # Compute desired step lengths (Ux, Uy) based on velocity and swing time
        Ux = torch.ones((N,), device=device) * 0.25
        Uy = torch.zeros((N,), device=device)
        local_offsets = torch.stack([Ux, Uy, torch.zeros_like(Ux)], dim=-1)
            
        # Check terrain height at the next step
        box_center, _, _, stance_foot_box_center = self.check_height(local_offsets)
        self.z_height = box_center[:, 2] - stance_foot_box_center[:, 2]

        # Decide which trajectory to use based on height
        height_threshold = 0.01  # 1cm threshold
        flat_mask = torch.abs(self.z_height) < height_threshold
        stair_up_mask = self.z_height >= height_threshold
        stair_down_mask = self.z_height <= -height_threshold

        # Set self.T per environment
        T = torch.empty((N,), dtype=torch.float32, device=device)
        T[flat_mask] = self.T_flat
        T[stair_up_mask] = self.T_upstair
        T[stair_down_mask] = self.T_downstair
        self.T = T

        # Compute stance/swing index and phase per environment
        Tswing = self.T
        # Current time for each env (broadcast scalar to tensor)
        current_time = torch.full((N,), float(self.env.sim.current_time), device=device)
        tp = (current_time % (2 * Tswing)) / (2 * Tswing)
        # Compute phi_c per env
        phi_c = torch.sin(2 * torch.pi * tp) / torch.sqrt(torch.sin(2 * torch.pi * tp) ** 2 + Tswing)
        new_stance_idx = (0.5 - 0.5 * torch.sign(phi_c)).long()  # shape (N,)
        self.swing_idx = 1 - new_stance_idx

        # Update stance foot pos, ori for envs that changed stance
        changed = (self.stance_idx != new_stance_idx)
        if torch.any(changed):
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            batch = torch.arange(N, device=device)
            self.stance_foot_pos_0[changed] = foot_pos_w[batch[changed], new_stance_idx[changed], :]
            self.stance_foot_ori_quat_0[changed] = foot_ori_w[batch[changed], new_stance_idx[changed], :]
            self.stance_foot_ori_0[changed] = self.get_euler_from_quat(foot_ori_w[batch[changed], new_stance_idx[changed], :])
        self.stance_idx = new_stance_idx

        # Compute phase_var and cur_swing_time per env
        phase_var = torch.where(tp < 0.5, 2 * tp, 2 * tp - 1)

        self.phase_var = phase_var
        self.cur_swing_time = self.phase_var * Tswing

    def check_height(self, local_offsets):
        terrain_importer = self.env.scene.terrain
        env_origins = terrain_importer.env_origins           # (N, 3) world-space origin per env
        terrain_origins = terrain_importer.terrain_origins   # (rows, cols, 3)
        cfg = self.env.cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs']
        cell_x, cell_y = cfg.size

        # 4) Determine subterrain cell indices
        desired_world = self.stance_foot_pos_0 + local_offsets           # (N,3)
        idx_i, idx_j = self.find_grid_idx(self.stance_foot_pos_0, terrain_origins)

        # 5) Fetch each cell's world origin
        cell_origins = terrain_origins[idx_i, idx_j]               # (N,3)

        box_center, box_bounds_lo, box_bounds_hi = self.box_center(desired_world[:,0], desired_world[:,1], cell_origins, cfg)

        #height change relative to the initial height
        stance_foot_box_center, stance_foot_box_bounds_lo, stance_foot_box_bounds_hi = self.box_center(self.stance_foot_pos_0[:,0], self.stance_foot_pos_0[:,1], cell_origins, cfg)
        return box_center, box_bounds_lo, box_bounds_hi, stance_foot_box_center

    def generate_reference_trajectory(self):
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)
        N = base_velocity.shape[0]
        T = self.T  # Use the per-env T that was set in update_Stance_Swing_idx

        # Define height thresholds for trajectory selection
        height_threshold = 0.01  # 5cm threshold to determine if it's a step up/down
        
        # Create trajectory selection masks
        flat_mask = torch.abs(self.z_height) < height_threshold
        stair_up_mask = self.z_height >= height_threshold
        stair_down_mask = self.z_height <= -height_threshold
        
        # Create stance selection masks
        right_stance_mask = (self.stance_idx == 1)  # Right foot stance
        left_stance_mask = (self.stance_idx == 0)   # Left foot stance
        
        # Initialize output tensors
        des_jt_pos = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        des_jt_vel = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        
        # Evaluate trajectories for each terrain type and stance
        phase_var_tensor = self.phase_var
        
        # Right stance trajectories
        if torch.any(right_stance_mask):
            # Flat terrain trajectory - right stance
            flat_right_mask = flat_mask & right_stance_mask
            if torch.any(flat_right_mask):
                flat_pos = bezier_deg(0, phase_var_tensor[flat_right_mask], T[flat_right_mask], self.right_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                flat_vel = bezier_deg(1, phase_var_tensor[flat_right_mask], T[flat_right_mask], self.right_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[flat_right_mask] = flat_pos
                des_jt_vel[flat_right_mask] = flat_vel
            
            # Stair up trajectory - right stance
            stair_up_right_mask = stair_up_mask & right_stance_mask
            if torch.any(stair_up_right_mask):
                stair_up_pos = bezier_deg(0, phase_var_tensor[stair_up_right_mask], T[stair_up_right_mask], self.stair_up_right_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_up_vel = bezier_deg(1, phase_var_tensor[stair_up_right_mask], T[stair_up_right_mask], self.stair_up_right_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_up_right_mask] = stair_up_pos
                des_jt_vel[stair_up_right_mask] = stair_up_vel
            
            # Stair down trajectory - right stance
            stair_down_right_mask = stair_down_mask & right_stance_mask
            if torch.any(stair_down_right_mask):
                stair_down_pos = bezier_deg(0, phase_var_tensor[stair_down_right_mask], T[stair_down_right_mask], self.stair_down_right_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_down_vel = bezier_deg(1, phase_var_tensor[stair_down_right_mask], T[stair_down_right_mask], self.stair_down_right_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_down_right_mask] = stair_down_pos
                des_jt_vel[stair_down_right_mask] = stair_down_vel
        
        # Left stance trajectories
        if torch.any(left_stance_mask):
            # Flat terrain trajectory - left stance
            flat_left_mask = flat_mask & left_stance_mask
            if torch.any(flat_left_mask):
                flat_pos = bezier_deg(0, phase_var_tensor[flat_left_mask], T[flat_left_mask], self.left_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                flat_vel = bezier_deg(1, phase_var_tensor[flat_left_mask], T[flat_left_mask], self.left_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[flat_left_mask] = flat_pos
                des_jt_vel[flat_left_mask] = flat_vel
            
            # Stair up trajectory - left stance
            stair_up_left_mask = stair_up_mask & left_stance_mask
            if torch.any(stair_up_left_mask):
                stair_up_pos = bezier_deg(0, phase_var_tensor[stair_up_left_mask], T[stair_up_left_mask], self.stair_up_left_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_up_vel = bezier_deg(1, phase_var_tensor[stair_up_left_mask], T[stair_up_left_mask], self.stair_up_left_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_up_left_mask] = stair_up_pos
                des_jt_vel[stair_up_left_mask] = stair_up_vel
            
            # Stair down trajectory - left stance
            stair_down_left_mask = stair_down_mask & left_stance_mask
            if torch.any(stair_down_left_mask):
                stair_down_pos = bezier_deg(0, phase_var_tensor[stair_down_left_mask], T[stair_down_left_mask], self.stair_down_left_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                stair_down_vel = bezier_deg(1, phase_var_tensor[stair_down_left_mask], T[stair_down_left_mask], self.stair_down_left_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
                des_jt_pos[stair_down_left_mask] = stair_down_pos
                des_jt_vel[stair_down_left_mask] = stair_down_vel


        yaw_offset = base_velocity[:, 2] 
        batch_idx = torch.arange(N, device=self.device)

        des_jt_pos[batch_idx, self.hip_yaw_idx[batch_idx, self.stance_idx]] += yaw_offset

        self.y_out = des_jt_pos
        self.dy_out = des_jt_vel


    def get_euler_from_quat(self, quat):
        euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
        euler_x = wrap_to_pi(euler_x)
        euler_y = wrap_to_pi(euler_y)
        euler_z = wrap_to_pi(euler_z)
        return torch.stack([euler_x, euler_y, euler_z], dim=-1)

    def get_actual_state(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        data = self.robot.data

        batch_idx = torch.arange(self.num_envs, device=self.device)
        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]

        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ori = self.get_euler_from_quat(foot_ori_w[batch_idx, self.stance_idx, :])

        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[batch_idx, self.stance_idx, :]


        jt_pos = data.joint_pos
        jt_vel = data.joint_vel
        # 4. Assemble state vectors
        self.y_act = jt_pos

        self.dy_act = jt_vel

    def _update_command(self):
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, [])
        self.vdot = vdot
        self.v = vcur
       
    def find_grid_idx(self, stance_pos_world, terrain_origins):
        H, W, _ = terrain_origins.shape
        B = stance_pos_world.shape[0]

        # compute squared XY-distances: [B, H, W]
        #   broadcast stance_pos_world over the H×W grid
        dist2 = (
             stance_pos_world[:, None, None, :2]  # [B,1,1,2]
             - terrain_origins[None, :, :, :2]    # [1,H,W,2]
        ).pow(2).sum(dim=-1)                    # [B,H,W]

        # flatten H×W → (H*W), find argmin per batch
        dist2_flat = dist2.view(B, -1)          # [B, H*W]
        idx_flat   = dist2_flat.argmin(dim=1)   # [B]

        # unravel flat index to 2D grid coords
        ix = idx_flat // W                      # rows
        iy = idx_flat %  W                      # cols

        return ix, iy

    def which_step(self, x: torch.Tensor, y: torch.Tensor, origin: torch.Tensor, cfg) -> torch.LongTensor:
        """
        Batched version: for each (x, y), returns
             - -1           → outside all steps
             - 0..num_steps-1 → ring index (0 is outermost ring)
             - num_steps    → center platform
        Assumes cfg.holes == False.
        Shapes:
             x, y            → (B,) or broadcastable
             origin          → (2,) or (B, 2)
        """
        # 1) recompute num_steps
        n_x = (cfg.size[0] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
        n_y = (cfg.size[1] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
        num_steps = int(min(n_x, n_y))

        # 2) extract origin coords (broadcastable)
        ox, oy = origin[..., 0], origin[..., 1]

        # 3) compute local offsets
        dx = x - ox
        dy = y - oy
        abs_dx = dx.abs()
        abs_dy = dy.abs()

        # 4) half‐sizes of stepped region
        terrain_w = cfg.size[0] - 2*cfg.border_width
        terrain_h = cfg.size[1] - 2*cfg.border_width
        half_w = terrain_w / 2.0
        half_h = terrain_h / 2.0

        # 5) inward distance from outer edge
        delta_x = half_w - abs_dx
        delta_y = half_h - abs_dy
        delta   = torch.min(delta_x, delta_y)

        # 6) compute raw ring index
        step_w = torch.tensor(cfg.step_width, dtype=delta.dtype, device=delta.device)
        raw_k  = torch.floor(delta / step_w).long()

        # 7) clamp into [-1, num_steps]
        return raw_k.clamp(min=0, max=num_steps)

    def box_center(self, x: torch.Tensor, y: torch.Tensor, origin: torch.Tensor, cfg) -> torch.Tensor:
        """
        For each (x, y), returns the 3D center of the box it lies in:
             • outside → (nan, nan, nan)
             • ring k  → center of that ring's face
             • center  → center platform
        Outputs (B,3). origin may be (3,) or (B,3).
        """
        # 1) get step_idx
        # we need the 2D origin for which_step
        origin_xy = origin[..., :2]
        step_idx = self.which_step(x, y, origin_xy, cfg)

        # recompute num_steps & heights
        n_x = (cfg.size[0] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
        n_y = (cfg.size[1] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
        num_steps = int(min(n_x, n_y))

        cx = origin[..., 0]
        cy = origin[..., 1]
        oz = origin[..., 2]

        # derive step_height & total_height
        step_h = -oz / (num_steps + 1)
        total_h = (num_steps + 1) * step_h
        cz = oz + total_h  # should be zero

        # half-dims and width tensors
        half_w = (cfg.size[0] - 2*cfg.border_width) / 2.0
        half_h = (cfg.size[1] - 2*cfg.border_width) / 2.0
        half_w = torch.tensor(half_w, device=x.device, dtype=x.dtype)
        half_h = torch.tensor(half_h, device=y.device, dtype=y.dtype)
        step_w = torch.tensor(cfg.step_width, device=x.device, dtype=x.dtype)

        # offsets
        dx = x - cx
        dy = y - cy
        adx = dx.abs()
        ady = dy.abs()

        # compute offset along face
        offset = (step_idx.float() + 0.5) * step_w

        # masks
        in_ring   = (step_idx >= 0) & (step_idx < num_steps)
        on_middle = (step_idx == num_steps)
        top    = in_ring & (ady >= adx) & (dy >  0)
        bottom = in_ring & (ady >= adx) & (dy <= 0)
        right  = in_ring & (adx >  ady) & (dx >  0)
        left   = in_ring & (adx >  ady) & (dx <= 0)

        # center coordinates
        cx_r = cx.expand_as(x)
        cy_r = cy.expand_as(y)
        center_x = torch.where(right,  cx_r + half_w - offset, cx_r)
        center_x = torch.where(left,   cx_r - half_w + offset, center_x)
        center_y = torch.where(top,    cy_r + half_h - offset, cy_r)
        center_y = torch.where(bottom, cy_r - half_h + offset, center_y)

        effective_k = torch.clamp(step_idx+1, min=0, max=num_steps+1)
        surface_z   = - effective_k.float() * step_h
        surface_z   = torch.where(step_idx >= 0, surface_z,
                                 torch.full_like(surface_z, float('nan')))

        centers = torch.stack([center_x, center_y, surface_z], dim=-1)

        face_half_x = torch.where(
             (top|bottom),
             half_w - offset,              # full ring width in x is 2*(half_w-offset)
             torch.where((right|left),
                            step_w / 2,       # face thickness in x
                            cfg.platform_width / 2)  # center‐platform
             )
        face_half_y = torch.where(
             (top|bottom),
             step_w / 2,                   # face thickness in y
             torch.where((right|left),
                            half_h - offset,  # full ring width in y is 2*(half_h-offset)
                            cfg.platform_width / 2)
             )

        # --- now min/max bounds in world frame ---
        min_x = center_x - face_half_x
        max_x = center_x + face_half_x
        min_y = center_y - face_half_y
        max_y = center_y + face_half_y

        bounds_lo = torch.stack([min_x, min_y],     dim=-1)               # [B,2]
        bounds_hi = torch.stack([max_x, max_y],     dim=-1)               # [B,2]

        return centers, bounds_lo, bounds_hi
          
