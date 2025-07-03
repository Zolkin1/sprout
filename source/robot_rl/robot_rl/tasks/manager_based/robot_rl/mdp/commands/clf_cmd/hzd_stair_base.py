import torch
from abc import ABC, abstractmethod

from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cmd import HZDCommandTerm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cmd_cfg import HZDStairBaseCommandCfg


class HZDStairBaseCommandTerm(HZDCommandTerm, ABC):
    """Base class for HZD stair command terms with height checking logic."""
    
    def __init__(self, cfg: "HZDStairBaseCommandCfg", env):
        super().__init__(cfg, env)
        
        # Initialize stance-related variables
        self.stance_foot_pos_0 = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_ori_quat_0 = torch.zeros((self.num_envs, 4), device=self.device)
        self.stance_foot_ori_0 = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_ori = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.stance_foot_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Height checking variables
        self.z_height = torch.zeros((self.num_envs), device=self.device)
        self.T = torch.zeros((self.num_envs), device=self.device)
        
        # Initialize stance_idx as tensor for per-environment tracking
        self.stance_idx = None

    def update_Stance_Swing_idx(self):
        """Update stance and swing indices with height checking logic."""
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

        # Set self.T per environment based on terrain type
        T = torch.empty((N,), dtype=torch.float32, device=device)
        T[flat_mask] = self._get_flat_swing_period()
        T[stair_up_mask] = self._get_stair_up_swing_period()
        T[stair_down_mask] = self._get_stair_down_swing_period()
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
        """Check terrain height at the next step location."""
        terrain_importer = self.env.scene.terrain
        terrain_origins = terrain_importer.terrain_origins   # (rows, cols, 3)
        cfg = self.env.cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs']

        # Determine subterrain cell indices
        desired_world = self.stance_foot_pos_0 + local_offsets           # (N,3)
        idx_i, idx_j = self.find_grid_idx(self.stance_foot_pos_0, terrain_origins)

        # Fetch each cell's world origin
        cell_origins = terrain_origins[idx_i, idx_j]               # (N,3)

        box_center, box_bounds_lo, box_bounds_hi = self.box_center(
            desired_world[:, 0], desired_world[:, 1], cell_origins, cfg
        )

        # Height change relative to the initial height
        stance_foot_box_center, _, _ = self.box_center(
            self.stance_foot_pos_0[:, 0], self.stance_foot_pos_0[:, 1], cell_origins, cfg
        )
        return box_center, box_bounds_lo, box_bounds_hi, stance_foot_box_center

    def find_grid_idx(self, stance_pos_world, terrain_origins):
        """Find grid indices for terrain lookup."""
        H, W, _ = terrain_origins.shape
        B = stance_pos_world.shape[0]

        # Compute squared XY-distances: [B, H, W]
        # Broadcast stance_pos_world over the H×W grid
        dist2 = (
            stance_pos_world[:, None, None, :2]  # [B,1,1,2]
            - terrain_origins[None, :, :, :2]    # [1,H,W,2]
        ).pow(2).sum(dim=-1)                    # [B,H,W]

        # Flatten H×W → (H*W), find argmin per batch
        dist2_flat = dist2.view(B, -1)          # [B, H*W]
        idx_flat = dist2_flat.argmin(dim=1)     # [B]

        # Unravel flat index to 2D grid coords
        ix = idx_flat // W                      # rows
        iy = idx_flat % W                       # cols

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
        # Recompute num_steps
        n_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
        n_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
        num_steps = int(min(n_x, n_y))

        # Extract origin coords (broadcastable)
        ox, oy = origin[..., 0], origin[..., 1]

        # Compute local offsets
        dx = x - ox
        dy = y - oy
        abs_dx = dx.abs()
        abs_dy = dy.abs()

        # Half-sizes of stepped region
        terrain_w = cfg.size[0] - 2 * cfg.border_width
        terrain_h = cfg.size[1] - 2 * cfg.border_width
        half_w = terrain_w / 2.0
        half_h = terrain_h / 2.0

        # Inward distance from outer edge
        delta_x = half_w - abs_dx
        delta_y = half_h - abs_dy
        delta = torch.min(delta_x, delta_y)

        # Compute raw ring index
        step_w = torch.tensor(cfg.step_width, dtype=delta.dtype, device=delta.device)
        raw_k = torch.floor(delta / step_w).long()

        # Clamp into [-1, num_steps]
        return raw_k.clamp(min=0, max=num_steps)

    def box_center(self, x: torch.Tensor, y: torch.Tensor, origin: torch.Tensor, cfg) -> torch.Tensor:
        """
        For each (x, y), returns the 3D center of the box it lies in:
             • outside → (nan, nan, nan)
             • ring k  → center of that ring's face
             • center  → center platform
        Outputs (B,3). origin may be (3,) or (B,3).
        """
        # Get step_idx
        # We need the 2D origin for which_step
        origin_xy = origin[..., :2]
        step_idx = self.which_step(x, y, origin_xy, cfg)

        # Recompute num_steps & heights
        n_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
        n_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
        num_steps = int(min(n_x, n_y))

        cx = origin[..., 0]
        cy = origin[..., 1]
        oz = origin[..., 2]

        # Derive step_height
        step_h = -oz / (num_steps + 1)

        # Half-dims and width tensors
        half_w = (cfg.size[0] - 2 * cfg.border_width) / 2.0
        half_h = (cfg.size[1] - 2 * cfg.border_width) / 2.0
        half_w = torch.tensor(half_w, device=x.device, dtype=x.dtype)
        half_h = torch.tensor(half_h, device=y.device, dtype=y.dtype)
        step_w = torch.tensor(cfg.step_width, device=x.device, dtype=x.dtype)

        # Offsets
        dx = x - cx
        dy = y - cy
        adx = dx.abs()
        ady = dy.abs()

        # Compute offset along face
        offset = (step_idx.float() + 0.5) * step_w

        # Masks
        in_ring = (step_idx >= 0) & (step_idx < num_steps)
        top = in_ring & (ady >= adx) & (dy > 0)
        bottom = in_ring & (ady >= adx) & (dy <= 0)
        right = in_ring & (adx > ady) & (dx > 0)
        left = in_ring & (adx > ady) & (dx <= 0)

        # Center coordinates
        cx_r = cx.expand_as(x)
        cy_r = cy.expand_as(y)
        center_x = torch.where(right, cx_r + half_w - offset, cx_r)
        center_x = torch.where(left, cx_r - half_w + offset, center_x)
        center_y = torch.where(top, cy_r + half_h - offset, cy_r)
        center_y = torch.where(bottom, cy_r - half_h + offset, center_y)

        effective_k = torch.clamp(step_idx + 1, min=0, max=num_steps + 1)
        surface_z = -effective_k.float() * step_h
        surface_z = torch.where(step_idx >= 0, surface_z,
                               torch.full_like(surface_z, float('nan')))

        centers = torch.stack([center_x, center_y, surface_z], dim=-1)

        # Calculate face half dimensions
        face_half_x = torch.where(
            (top | bottom),
            half_w - offset,              # Full ring width in x is 2*(half_w-offset)
            torch.where((right | left),
                       step_w / 2,       # Face thickness in x
                       cfg.platform_width / 2)  # Center platform
        )
        face_half_y = torch.where(
            (top | bottom),
            step_w / 2,                   # Face thickness in y
            torch.where((right | left),
                       half_h - offset,  # Full ring width in y is 2*(half_h-offset)
                       cfg.platform_width / 2)
        )

        # Min/max bounds in world frame
        min_x = center_x - face_half_x
        max_x = center_x + face_half_x
        min_y = center_y - face_half_y
        max_y = center_y + face_half_y

        bounds_lo = torch.stack([min_x, min_y], dim=-1)               # [B,2]
        bounds_hi = torch.stack([max_x, max_y], dim=-1)               # [B,2]

        return centers, bounds_lo, bounds_hi

    def get_euler_from_quat(self, quat):
        """Convert quaternion to euler angles."""
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
        # Foot positions and orientations (world frame)
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
        # Assemble state vectors
        self.y_act = jt_pos
        self.dy_act = jt_vel

    @abstractmethod
    def _get_flat_swing_period(self) -> float:
        """Get the swing period for flat terrain."""
        pass

    @abstractmethod
    def _get_stair_up_swing_period(self) -> float:
        """Get the swing period for stair up terrain."""
        pass

    @abstractmethod
    def _get_stair_down_swing_period(self) -> float:
        """Get the swing period for stair down terrain."""
        pass

    @abstractmethod
    def generate_reference_trajectory(self):
        """Generate reference trajectory based on terrain type. Must be implemented by subclasses."""
        pass 