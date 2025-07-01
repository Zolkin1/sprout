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

        self.metrics = {}
     
        # self.com_z = torch.ones((self.num_envs), device=self.device)*self.z0

        # load joint trajectory config from YAML
        #this is the flat ref trajectory
        yaml_path = "source/robot_rl/robot_rl/assets/robots/single_support_config_solution.yaml"
        self.jt_config = JointTrajectoryConfig()
        self.jt_config.load_from_yaml(yaml_path, self.robot)
        self.T = self.env.cfg.commands.step_period.period_range[0]/2
        
        #this is the stair ref trajectory
        yaml_path = "source/robot_rl/robot_rl/assets/robots/stair_config_solution.yaml"
        self.jt_config_stair = JointTrajectoryConfig()
        self.jt_config_stair.load_from_yaml(yaml_path, self.robot)

        #load the downstair ref trajectory
        yaml_path = "source/robot_rl/robot_rl/assets/robots/downstair_config_solution.yaml"
        self.jt_config_downstair = JointTrajectoryConfig()
        self.jt_config_downstair.load_from_yaml(yaml_path, self.robot)
  

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
        # device = self.env.command_manager.get_command("base_velocity").device
        
        return
    
    def _update_metrics(self):
        # Foot tracking
        # foot_pos = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :2]  # Only take x,y coordinates
        # # Contact schedule function
        # tp = (self.env.sim.current_time % (2 * self.T)) / (2 * self.T)  # Scaled between 0-1
        # phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)

        # swing_foot_pos = foot_pos[:, int(0.5 + 0.5 * torch.sign(phi_c))]
        # Only compare x,y coordinates of foot target
        
        # Update metrics using actual joint names from the YAML file
        for i, joint_name in enumerate(self.robot.joint_names):
            error_key = f"error_{joint_name}"
            self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot
        
        # return self.foot_target  # Return the foot target tensor for observation


    def update_Stance_Swing_idx(self):
        Tswing = self.T 
        tp = (self.env.sim.current_time % (2 * Tswing)) / (2 * Tswing)  
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)

        new_stance_idx = int(0.5 - 0.5 * torch.sign(phi_c))
        self.swing_idx = 1 - new_stance_idx
        
        if self.stance_idx is None or new_stance_idx != self.stance_idx:
            if self.stance_idx is None:
                self.stance_idx = new_stance_idx

            # update stance foot pos, ori
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            self.stance_foot_pos_0 = foot_pos_w[:, new_stance_idx, :]
            self.stance_foot_ori_quat_0 = foot_ori_w[:, new_stance_idx, :]
            self.stance_foot_ori_0 = self.get_euler_from_quat(foot_ori_w[:, new_stance_idx, :])
       
        self.stance_idx = new_stance_idx

        if tp < 0.5:
            self.phase_var = 2 * tp
        else:
            self.phase_var = 2 * tp - 1
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

        # TODO: select which trajectory to use
        # local offsets are the swing foot target in stance foot frame
        # Compute desired step lengths from base velocity
        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)
        
        # Compute desired step lengths (Ux, Uy) based on velocity and swing time
        Ux = torch.ones((N,), device=base_velocity.device) * 0.25
        Uy = torch.zeros((N,), device=base_velocity.device)
        


        # Create local offsets for terrain height checking
        local_offsets = torch.stack([Ux, Uy, torch.zeros_like(Ux)], dim=-1)
        
        box_center, box_bounds_lo, box_bounds_hi, stance_foot_box_center = self.check_height(local_offsets)
        self.z_height = box_center[:, 2] - stance_foot_box_center[:, 2]
        
        # based on the z_height, select the trajectory for each env
        # if z_height is positive, use the stair trajectory
        # if z_height is negative, use the downstair trajectory
        # if z_height is 0, use the flat trajectory
        
        # maybe evaluate three separate set of coeffs and select based on z height
        # assuming each trajectory has same duration for now

        # Define height thresholds for trajectory selection
        height_threshold = 0.05  # 5cm threshold to determine if it's a step up/down
        
        # Create trajectory selection masks
        flat_mask = torch.abs(self.z_height) < height_threshold
        stair_up_mask = self.z_height >= height_threshold
        stair_down_mask = self.z_height <= -height_threshold
        
        # Get coefficients for each trajectory type
        if self.stance_idx == 1:  # Right foot stance
            flat_coeffs = self.right_coeffs
            # Remap stair trajectories for right foot
            stair_up_coeffs = self.stair_up_right_coeffs
            stair_down_coeffs = self.stair_down_right_coeffs
        else:  # Left foot stance
            flat_coeffs = self.left_coeffs
            # Remap stair trajectories for left foot
            stair_up_coeffs = self.stair_up_left_coeffs
            stair_down_coeffs = self.stair_down_left_coeffs
        
        # Initialize output tensors
        des_jt_pos = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        des_jt_vel = torch.zeros((N, self.cfg.num_outputs), device=self.device)
        
        # Evaluate trajectories for each terrain type
        phase_var_tensor = torch.full((N,), self.phase_var, device=self.device)
        
        # Flat terrain trajectory
        if torch.any(flat_mask):
            flat_pos = bezier_deg(0, phase_var_tensor[flat_mask], T[flat_mask], flat_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
            flat_vel = bezier_deg(1, phase_var_tensor[flat_mask], T[flat_mask], flat_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
            des_jt_pos[flat_mask] = flat_pos
            des_jt_vel[flat_mask] = flat_vel
        
        # Stair up trajectory
        if torch.any(stair_up_mask):
            stair_up_pos = bezier_deg(0, phase_var_tensor[stair_up_mask], T[stair_up_mask], stair_up_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
            stair_up_vel = bezier_deg(1, phase_var_tensor[stair_up_mask], T[stair_up_mask], stair_up_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
            des_jt_pos[stair_up_mask] = stair_up_pos
            des_jt_vel[stair_up_mask] = stair_up_vel
        
        # Stair down trajectory
        if torch.any(stair_down_mask):
            stair_down_pos = bezier_deg(0, phase_var_tensor[stair_down_mask], T[stair_down_mask], stair_down_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
            stair_down_vel = bezier_deg(1, phase_var_tensor[stair_down_mask], T[stair_down_mask], stair_down_coeffs, torch.tensor(self.cfg.bez_deg, device=self.device))
            des_jt_pos[stair_down_mask] = stair_down_pos
            des_jt_vel[stair_down_mask] = stair_down_vel

        self.y_out = des_jt_pos
        self.dy_out = des_jt_vel

    def nom_bezier_curve(self, control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates multiple Bezier curves at given time points using the control points.

        Args:
            control_points (torch.Tensor): Tensor of shape [num_curves, num_control_points, dim]
                                        representing the control points for each curve.
            t (torch.Tensor): Tensor of shape [num_curves] representing the time point at which to evaluate each Bezier curve.
        
        Returns:
            torch.Tensor: Tensor of shape [num_curves, dim] representing the evaluated points on the Bezier curves.
        """
        dim, num_control_points = control_points.shape
        n = num_control_points - 1

        num_curves = t.shape[0]
        # Initialize the result tensor to zeros: shape [num_curves, dim]
        curve_points = torch.zeros((num_curves, dim), dtype=control_points.dtype, device=control_points.device)
        
        for i in range(n + 1):
            binomial_coeff = math.comb(n, i)  # Binomial coefficient
            bernstein_poly = binomial_coeff * (t ** i) * ((1 - t) ** (n - i))  # Bernstein basis polynomial

            # Add the contribution of each control point to each curve
            curve_points += bernstein_poly.unsqueeze(-1) * control_points[:, i]

        return curve_points

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


        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]

        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[:, self.stance_idx, :]
        self.stance_foot_ori = self.get_euler_from_quat(foot_ori_w[:, self.stance_idx, :])

        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[:, self.stance_idx, :]
        self.stance_foot_ang_vel = foot_ang_vel_w[:, self.stance_idx, :]

 

        jt_pos = data.joint_pos
        jt_vel = data.joint_vel
        # 4. Assemble state vectors
        self.y_act = jt_pos

        self.dy_act = jt_vel

    def _update_command(self):
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        # how to handle for the first step?
        # i.e. v is not defined
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
          
