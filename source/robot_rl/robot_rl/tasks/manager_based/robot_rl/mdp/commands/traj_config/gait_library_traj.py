from typing import List, Dict, Union, Tuple
from pathlib import Path
import torch
import re
import math
from isaaclab.utils.math import wrap_to_pi, quat_apply, quat_from_euler_xyz,euler_xyz_from_quat, wrap_to_pi

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.hlip_cmd import _transfer_to_local_frame, euler_rates_to_omega
from .ee_traj import EndEffectorTrajectory,get_euler_from_quat #, EndEffectorTracker

def _ncr(n, r):
    return math.comb(n, r)


def bezier_deg_batched(
    order: int,
    tau: torch.Tensor,                # [num_env], each in [0,1]
    domain_dur: torch.Tensor,           # [num_env], positive scalars
    control_points: torch.Tensor,     # [num_env, jt_dim, degree+1]
    degree: int
) -> torch.Tensor:
    """
    Batched evaluation of Bezier curve (or its derivative) per environment.

    Args:
        order: 0 → position, 1 → time-derivative.
        tau: [num_env]
        domain_dur: [num_env]
        control_points: [num_env, jt_dim, degree+1]
        degree: Polynomial degree (must match control_points.shape[-1] - 1)

    Returns:
        Tensor of shape [num_env, jt_dim], either position or derivative.
    """
    num_env, jt_dim, d_plus_1 = control_points.shape
    assert d_plus_1 == degree + 1, "Mismatch between degree and control_points shape"

    # Clamp tau
    tau = torch.clamp(tau, 0.0, 1.0)  # [num_env]

    if order == 0:
        # ───── POSITION ─────
        i_vec = torch.arange(degree + 1, device=control_points.device)  # [degree+1]
        coefs = torch.tensor([_ncr(degree, i) for i in range(degree + 1)],
                             dtype=control_points.dtype,
                             device=control_points.device)  # [degree+1]

        tau_pow = tau.unsqueeze(1) ** i_vec  # [num_env, degree+1]
        one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - i_vec)  # [num_env, degree+1]

        weights = coefs.unsqueeze(0) * tau_pow * one_minus_pow  # [num_env, degree+1]

        # Weighted sum across control points dimension
        output = torch.einsum("nd,njd->nj", weights, control_points)  # [num_env, jt_dim]
        return output

    elif order == 1:
        # ───── DERIVATIVE ─────
        i_vec = torch.arange(degree, device=control_points.device)  # [degree]
        coefs = torch.tensor([_ncr(degree - 1, i) for i in range(degree)],
                             dtype=control_points.dtype,
                             device=control_points.device)  # [degree]

        tau_pow = tau.unsqueeze(1) ** i_vec  # [num_env, degree]
        one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - 1 - i_vec)  # [num_env, degree]

        weights = degree * coefs.unsqueeze(0) * tau_pow * one_minus_pow  # [num_env, degree]

        # Compute CP differences: CP_{i+1} - CP_i → shape [num_env, jt_dim, degree]
        cp_diff = control_points[:, :, 1:] - control_points[:, :, :-1]  # [num_env, jt_dim, degree]

        # Weighted sum across degree dimension
        output = torch.einsum("nd,njd->nj", weights, cp_diff)  # [num_env, jt_dim]

        # Divide by step duration
        return output / domain_dur.unsqueeze(1)  # [num_env, jt_dim]

    else:
        raise ValueError("Only order=0 (position) or order=1 (derivative) are supported.")


class GaitLibraryEndEffectorConfig(EndEffectorTrajectory):
    """Configuration class for gait library with velocity-based gait selection."""
    
    def __init__(self, gait_library_path: str, 
                 gait_velocity_ranges: Union[Dict[str, tuple], Tuple[float, float, float]], 
                 config_name: str = "single_support",
                 use_standing: bool = True):
        """
        Initialize gait library configuration.
        
        Args:
            gait_library_path: Path to directory containing gait YAML files
            gait_velocity_ranges: Either:
                - Dict mapping gait names to (min_vel, max_vel) tuples in m/s, OR
                - Tuple (min_vel, max_vel, step) for automatic discretization
            trajectory_type: "joint" or "end_effector"
            config_name: Base name for the configuration files (e.g., "single_support")
        """
        self.gait_library_path = Path(gait_library_path)
        self.trajectory_type = "end_effector"
        self.config_name = config_name

        self.left_coeffs_batched = {}
        self.right_coeffs_batched = {}
        
        min_vel, max_vel, step = gait_velocity_ranges
        self.gait_velocity_ranges = self._generate_discretized_ranges(min_vel, max_vel, step)
      
        # Cache for loaded gait configs
        self._gait_cache = {}
        
        # Current gait assignments per environment
        self.current_gaits = None
        self.num_envs = None
        
        # Load the first gait to initialize base class
        first_gait = list(self.gait_velocity_ranges.keys())[0]
        super().__init__(self._get_gait_path(first_gait))

        # Use standing or not
        self.use_standing = use_standing

        # Pre-load all gaits
        self._preload_gaits()

        ##
        # Load in other values
        ##
        self.num_gaits = len(self._gait_cache)
        self.max_domains = 0

        self.T = [{}]
        self.domain_seq = [{}]

        for gait in self._gait_cache.values():
            self.T.append(gait.T)
            self.domain_seq.append(gait.domain_seq)
            if len(gait.domain_seq) > self.max_domains:
                self.max_domains = len(gait.domain_seq)

        self.T_gait = sum(self.T[-1].values())    # Use the last gait for the length of the gaits

        ##
        # Create information for domain identification
        ##
        # Create tensors to store cumulative times and domain indices
        # Shape: [num_gaits, max_domains + 1]
        self.domain_cumulative_times = torch.zeros((self.num_gaits, self.max_domains + 1))

        # Store domain names as indices for easier batched operations
        # Create a mapping from domain names to indices
        all_domain_names = set()
        for domains in self.domain_seq:
            all_domain_names.update(domains)
        self.domain_name_to_idx = {name: idx for idx, name in enumerate(sorted(all_domain_names))}
        self.idx_to_domain_name = {idx: name for name, idx in self.domain_name_to_idx.items()}

        # Shape: [num_gaits, max_domains]
        self.domain_indices = torch.full((self.num_gaits, self.max_domains), -1, dtype=torch.long)

        # Shape: [num_gaits] - store actual number of domains for each gait
        self.num_domains_per_gait = torch.zeros(self.num_gaits, dtype=torch.long)

        # Fill the tensors
        for gait_idx in range(self.num_gaits):
            domains = self.domain_seq[gait_idx]
            self.num_domains_per_gait[gait_idx] = len(domains)

            cumulative_time = 0.0
            for domain_idx, domain_name in enumerate(domains):
                self.domain_cumulative_times[gait_idx, domain_idx] = cumulative_time
                self.domain_indices[gait_idx, domain_idx] = self.domain_name_to_idx[domain_name]

                # Access T as a list of dicts: T[gait_idx][domain_name]
                if domain_name in self.T[gait_idx]:
                    cumulative_time += self.T[gait_idx][domain_name]

            # Set the final boundary (total gait period)
            self.domain_cumulative_times[gait_idx, len(domains)] = cumulative_time


    def _generate_discretized_ranges(self, min_vel: float, max_vel: float, step: float) -> Dict[str, tuple]:
        """Generate gait velocity ranges from discretization parameters (supports negative ranges)."""
        if step < 0:
            raise ValueError("Step must be non-negative!")
        if step == 0:
            if min_vel != max_vel:
                raise ValueError("If step = 0 then the max and min velocities must match!")
            else:
                speed_cms = int(round(min_vel * 100))
                gait_ranges = {}
                gait_ranges[f"gait_{speed_cms}cms"] = (min_vel, max_vel)
        else:
            # Create inclusive list of center velocities and sort them
            velocities = torch.arange(min_vel, max_vel + step * 0.5, step).tolist()
            velocities = sorted(velocities)

            gait_ranges = {}
            for i, vel in enumerate(velocities):
               if i == 0:
                    next_vel = velocities[i + 1]
                    min_range = vel - step / 2
                    max_range = (vel + next_vel) / 2
               elif i == len(velocities) - 1:
                    prev_vel = velocities[i - 1]
                    min_range = (prev_vel + vel) / 2
                    max_range = vel + step / 2
               else:
                    prev_vel = velocities[i - 1]
                    next_vel = velocities[i + 1]
                    min_range = (prev_vel + vel) / 2
                    max_range = (vel + next_vel) / 2

               # Use velocity in cm/s as gait name
               speed_cms = int(round(vel * 100))
               gait_ranges[f"gait_{speed_cms}cms"] = (min_range, max_range)

        return gait_ranges

    
    def _get_gait_path(self, gait_name: str) -> str:
        """Get the full path to a gait YAML file based on naming convention."""
        # Convert gait name to speed in cm/s
        speed_cms = self._gait_name_to_speed_cms(gait_name)
        filename = f"{self.config_name}_solution_{speed_cms}.yaml"
        return str(self.gait_library_path / filename)
    
    def _gait_name_to_speed_cms(self, gait_name: str) -> int:
        """Convert gait name to speed in cm/s based on velocity ranges."""
        min_vel, max_vel = self.gait_velocity_ranges[gait_name]
        # Use the midpoint of the velocity range and convert to cm/s
        speed_ms = (min_vel + max_vel) / 2
        speed_cms = round(speed_ms * 100)  # Convert m/s to cm/s

        return speed_cms
    
    def _speed_cms_to_gait_name(self, speed_cms: int) -> str:
        """Convert speed in cm/s to gait name."""
        speed_ms = speed_cms / 100.0  # Convert cm/s to m/s
        
        # Find the gait that contains this speed
        for gait_name, (min_vel, max_vel) in self.gait_velocity_ranges.items():
            if min_vel <= speed_ms < max_vel:
                return gait_name
        
        # If not found, return the last gait (catch-all)
        return list(self.gait_velocity_ranges.keys())[-1]
    
    def _discover_available_gaits(self) -> Dict[str, int]:
        """Discover available gait files and their speeds."""
        available_gaits = {}
        
        if not self.gait_library_path.exists():
            return available_gaits
        
        # Pattern to match: {config_name}_solution_{speed}.yaml
        pattern = re.compile(f"{self.config_name}_solution_(-?\\d+)\\.yaml")
        
        for yaml_file in self.gait_library_path.glob("*.yaml"):
            match = pattern.match(yaml_file.name)
            if match:
                speed_cms = int(match.group(1))
                available_gaits[yaml_file.stem] = speed_cms
        
        return available_gaits
    
    def _preload_gaits(self):
        """Pre-load all gaits and precompute control points for all velocities."""
        # Discover available gait files
        available_gaits = self._discover_available_gaits()

        if self.use_standing:
            #load standing pose
            standing_yaml = self.gait_library_path / "standing.yaml"
            self.standing_config = EndEffectorTrajectory(standing_yaml)
    
        # Load each available gait
        for filename, speed_cms in available_gaits.items():
            gait_name = self._speed_cms_to_gait_name(speed_cms)
            if gait_name not in self._gait_cache:
                self._load_gait_config(gait_name, speed_cms)
        
    
    def _precompute_control_points(self):
        """Precompute control points for all velocities in batched tensors."""
        # Compute per domain
        # Use sorted gait names to ensure consistent ordering
        gait_names = sorted(self.gait_velocity_ranges.keys(),
                            key=lambda name: self.gait_velocity_ranges[name][0])
        num_vel = len(gait_names)

        # Get dimensions from the first gait
        first_gait = gait_names[0]
        first_config = self._gait_cache[first_gait]

        for domain_name in self.domain_seq:
            # Get dimensions
            output_dim = first_config.left_coeffs[domain_name].shape[0]  # num_outputs
            degree_plus_1 = first_config.left_coeffs[domain_name].shape[1]  # degree + 1
            device = first_config.left_coeffs[domain_name].device
            dtype = first_config.left_coeffs[domain_name].dtype

            # Initialize batched control point tensors
            self.left_coeffs_batched[domain_name] = torch.zeros(
                (num_vel, output_dim, degree_plus_1), device=device, dtype=dtype
            )
            self.right_coeffs_batched[domain_name] = torch.zeros(
                (num_vel, output_dim, degree_plus_1), device=device, dtype=dtype
            )

            # Fill the batched tensors
            for i, gait_name in enumerate(gait_names):
                if gait_name in self._gait_cache:
                    config = self._gait_cache[gait_name]
                    self.left_coeffs_batched[domain_name][i] = config.left_coeffs[domain_name]
                    self.right_coeffs_batched[domain_name][i] = config.right_coeffs[domain_name]

        # Store the Bezier degree
        self.bez_deg = first_config.bez_deg
    
    def _load_gait_config(self, gait_name: str, speed_cms: int = None):
        """Load a specific gait configuration."""
        gait_path = self._get_gait_path_from_speed(speed_cms)
        
        # Create appropriate trajectory config based on type
        config = EndEffectorTrajectory(gait_path)
        self._gait_cache[gait_name] = config
    
    def _get_gait_path_from_speed(self, speed_cms: int) -> str:
        """Get the full path to a gait YAML file from speed in cm/s."""
        filename = f"{self.config_name}_solution_{speed_cms}.yaml"
        return str(self.gait_library_path / filename)
    
    def select_gaits_by_velocity(self, desired_velocities: torch.Tensor) -> torch.Tensor:
          """
          Vectorized selection of gaits based on desired velocity magnitude.

          Args:
               desired_velocities: Tensor of shape (num_envs, 2) with [lin_vel_x, lin_vel_y]

          Returns:
               Tensor of gait indices (shape: [num_envs])
          """
          # Compute velocity magnitudes
          # vel_magnitudes = torch.norm(desired_velocities, dim=1)
          vel_magnitudes = desired_velocities[:, 0]
          # Create sorted list of gait range boundaries (only upper bounds)
          # Sort gait names by their minimum velocity to ensure proper ordering
          gait_names = sorted(self.gait_velocity_ranges.keys(), 
                             key=lambda name: self.gait_velocity_ranges[name][0])
          # Get upper bounds for all gaits except the last one
          boundaries = torch.tensor(
               [self.gait_velocity_ranges[name][1] for name in gait_names[:-1]],
               device=desired_velocities.device,
               dtype=vel_magnitudes.dtype,
          )

          # Bucketize assigns each velocity to a bin based on upper boundaries
          gait_indices = torch.bucketize(vel_magnitudes.contiguous(), boundaries, right=False)
          
          # Clamp indices to valid range to handle out-of-bounds velocities
          # This ensures velocities at or above the maximum gait range use the last gait
          gait_indices = torch.clamp(gait_indices, 0, len(gait_names) - 1)

          return gait_indices

    def reorder_and_remap(self, cfg, device):
        """Reorder and remap coefficients for all gaits and select based on velocity."""
        # Get desired velocities from the environment
        
        # Reorder and remap all gaits (order doesn't matter for this operation)
        gait_names = list(self.gait_velocity_ranges.keys())
        for gait_name in gait_names:
            if gait_name in self._gait_cache:
                config = self._gait_cache[gait_name]
                
                if self.trajectory_type == "end_effector":
                    if hasattr(cfg, 'ee_tracker'):
                        config.reorder_and_remap(cfg, cfg.ee_tracker, device)
                    else:
                        # ee_tracker = EndEffectorTracker(config.constraint_specs, None)
                        config.reorder_and_remap(cfg, device)
                else:
                    config.reorder_and_remap(cfg, cfg.robot, device)
                

        # Recompute batched control points after remapping
        self._precompute_control_points()

    def determine_domains(self, gait_idx: torch.Tensor, time: float) -> torch.Tensor:
        """
        Determine the domain for each environment in a batched manner.

        Args:
            gait_idx: Tensor of shape [num_envs] containing gait index for each env

        Returns:
            domain_idx: Tensor of shape [num_envs] containing domain index for each env
        """
        # Calculate time into gait for all envs
        time_into_gait = time % self.T_gait

        # Get the cumulative times for the selected gaits
        # Shape: [num_envs, max_domains + 1]
        selected_cumulative_times = self.domain_cumulative_times[gait_idx]

        # Get the domain indices for the selected gaits
        # Shape: [num_envs, max_domains]
        selected_domain_indices = self.domain_indices[gait_idx]

        # Get number of domains for each selected gait
        # Shape: [num_envs]
        num_domains = self.num_domains_per_gait[gait_idx]

        # Create a mask for valid domains
        # Shape: [num_envs, max_domains]
        domain_mask = torch.arange(self.max_domains).unsqueeze(0) < num_domains.unsqueeze(1)

        # Compare time_into_gait with boundaries
        # Shape: [num_envs, max_domains]
        in_domain = (time_into_gait.unsqueeze(1) >= selected_cumulative_times[:, :-1]) & \
                    (time_into_gait.unsqueeze(1) < selected_cumulative_times[:, 1:])

        # Apply mask to only consider valid domains
        in_domain = in_domain & domain_mask

        # Find which domain each env is in (get the index of True value)
        # Shape: [num_envs]
        domain_positions = in_domain.float().argmax(dim=1)

        # Get the actual domain indices
        # Shape: [num_envs]
        current_domain_indices = torch.gather(selected_domain_indices, 1,
                                              domain_positions.unsqueeze(1)).squeeze(1)

        return current_domain_indices

    def get_ref_traj(self, hzd_cmd, cmd_vel, gait_indices, domain_indices) -> tuple[torch.Tensor, torch.Tensor]:
        """Get reference trajectory using precomputed batched control points."""
        # Loop through all domains
        ref_pos = {}
        ref_vel = {}
        stance_coeffs = self.right_coeffs_batched if hzd_cmd.stance_idx == 1 else self.left_coeffs_batched
        for key in stance_coeffs.keys():
            # Evaluate one trajectory per gait (much more efficient!)
            # control_points: [num_vel, jt_dim, degree+1]
            # domain_dur: [1] -> expand to [num_vel]

            domain_dur = torch.tensor([self.T[key]], dtype=torch.float32, device=cmd_vel.device)  # [1]
            domain_dur_expanded = domain_dur.expand(stance_coeffs[key].shape[0])  # [num_vel]

            ref_pos[key] = bezier_deg_batched(
                            order=0,
                            tau=hzd_cmd.phase_var,
                            domain_dur=domain_dur_expanded,
                            control_points=stance_coeffs[key],
                            degree=self.bez_deg[key],
                        )
            ref_vel[key] = bezier_deg_batched(
                            order=1,
                            tau=hzd_cmd.phase_var,
                            domain_dur=domain_dur_expanded,
                            control_points=stance_coeffs[key],
                            degree=self.bez_deg[key],
                        )

        # Gather the selected trajectories
        # TODO: Generate offset for the gait indexes
        des_pos = ref_pos[self.domain_idx_to_name[domain_indices]][gait_indices]  # [N, jt_dim]
        des_vel = ref_vel[self.domain_idx_to_name[domain_indices]][gait_indices]  # [N, jt_dim]

        if self.use_standing:
            stand_idx = torch.where(torch.norm(cmd_vel, dim=1) < hzd_cmd.standing_threshold)[0]
            if stand_idx.numel() > 0:
                standing_pose = self.right_standing_pos if hzd_cmd.stance_idx == 1 else self.left_standing_pos
                des_pos[stand_idx] = standing_pose.expand(len(stand_idx), -1)
                des_vel[stand_idx] = torch.zeros_like(des_vel[stand_idx])

        des_pos, des_vel = self._apply_swing_modifications(hzd_cmd, des_pos, des_vel, cmd_vel)
   
        return des_pos, des_vel
    

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

