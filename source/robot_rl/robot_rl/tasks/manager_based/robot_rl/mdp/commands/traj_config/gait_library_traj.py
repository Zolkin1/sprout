from typing import List, Dict, Union, Tuple
from pathlib import Path
import torch
import re
import math

from .base_traj import BaseTrajectoryConfig
from .ee_traj import EndEffectorTrajectoryConfig, EndEffectorTracker
from .jt_traj import JointTrajectoryConfig
from robot_rl.tasks.manager_based.robot_rl.terrains.stair_cfg import get_step_height_at_x, get_uniform_stair_step_height_from_env

def _ncr(n, r):
    return math.comb(n, r)


def bezier_deg_batched(
    order: int,
    tau: torch.Tensor,                # [num_env], each in [0,1]
    step_dur: torch.Tensor,           # [num_env], positive scalars
    control_points: torch.Tensor,     # [num_env, jt_dim, degree+1]
    degree: int
) -> torch.Tensor:
    """
    Batched evaluation of Bezier curve (or its derivative) per environment.

    Args:
        order: 0 → position, 1 → time-derivative.
        tau: [num_env]
        step_dur: [num_env]
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
        return output / step_dur.unsqueeze(1)  # [num_env, jt_dim]

    else:
        raise ValueError("Only order=0 (position) or order=1 (derivative) are supported.")


class GaitLibraryTrajectoryConfig(BaseTrajectoryConfig):
    """Configuration class for gait library with velocity-based gait selection."""
    
    def __init__(self, gait_library_path: str, 
                 gait_velocity_ranges: Union[Dict[str, tuple], Tuple[float, float, float]], 
                 trajectory_type: str = "end_effector", config_name: str = "single_support"):
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
        self.trajectory_type = trajectory_type
        self.config_name = config_name
        
        # Handle different input types for gait_velocity_ranges
        if isinstance(gait_velocity_ranges, tuple) and len(gait_velocity_ranges) == 3:
            # Discretization mode: (min_vel, max_vel, step)
            min_vel, max_vel, step = gait_velocity_ranges
            self.gait_velocity_ranges = self._generate_discretized_ranges(min_vel, max_vel, step)
        else:
            # Dictionary mode: explicit gait names and ranges
            self.gait_velocity_ranges = gait_velocity_ranges
        
        # Cache for loaded gait configs
        self._gait_cache = {}
        
        # Current gait assignments per environment
        self.current_gaits = None
        self.num_envs = None
        
        # Load the first gait to initialize base class
        first_gait = list(self.gait_velocity_ranges.keys())[0]
        super().__init__(self._get_gait_path(first_gait))
        
        # Pre-load all gaits
        self._preload_gaits()
    
    def _generate_discretized_ranges(self, min_vel: float, max_vel: float, step: float) -> Dict[str, tuple]:
          """Generate gait velocity ranges from discretization parameters (supports negative ranges)."""
          if step <= 0:
               raise ValueError("Step must be positive")

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
        
        # Load each available gait
        for filename, speed_cms in available_gaits.items():
            gait_name = self._speed_cms_to_gait_name(speed_cms)
            if gait_name not in self._gait_cache:
                self._load_gait_config(gait_name, speed_cms)
        
    
    def _precompute_control_points(self):
        """Precompute control points for all velocities in batched tensors."""
        # Use sorted gait names to ensure consistent ordering
        gait_names = sorted(self.gait_velocity_ranges.keys(), 
                           key=lambda name: self.gait_velocity_ranges[name][0])
    
        num_vel = len(gait_names)
        
        # Get dimensions from the first gait
        first_gait = gait_names[0]
        first_config = self._gait_cache[first_gait]
        
        # Get dimensions
        output_dim = first_config.left_coeffs.shape[0]  # num_outputs
        degree_plus_1 = first_config.left_coeffs.shape[1]  # degree + 1
        device = first_config.left_coeffs.device
        dtype = first_config.left_coeffs.dtype
        
        # Initialize batched control point tensors
        self.left_coeffs_batched = torch.zeros(
            (num_vel, output_dim, degree_plus_1), device=device, dtype=dtype
        )
        self.right_coeffs_batched = torch.zeros(
            (num_vel, output_dim, degree_plus_1), device=device, dtype=dtype
        )
        
        # Fill the batched tensors
        for i, gait_name in enumerate(gait_names):
            if gait_name in self._gait_cache:
                config = self._gait_cache[gait_name]
                self.left_coeffs_batched[i] = config.left_coeffs
                self.right_coeffs_batched[i] = config.right_coeffs
        
        # Store the Bezier degree
        self.bez_deg = first_config.bez_deg
    
    def _load_gait_config(self, gait_name: str, speed_cms: int = None):
        """Load a specific gait configuration."""
        if speed_cms is None:
            speed_cms = self._gait_name_to_speed_cms(gait_name)
        
        gait_path = self._get_gait_path_from_speed(speed_cms)
        
        # Create appropriate trajectory config based on type
        if self.trajectory_type == "end_effector":
            config = EndEffectorTrajectoryConfig(gait_path)
        else:  # joint
            config = JointTrajectoryConfig(gait_path)
        
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
          gait_indices = torch.bucketize(vel_magnitudes, boundaries, right=False)
          
          # Clamp indices to valid range to handle out-of-bounds velocities
          # This ensures velocities at or above the maximum gait range use the last gait
          gait_indices = torch.clamp(gait_indices, 0, len(gait_names) - 1)

          return gait_indices

    
    def _load_specific_data(self, data):
        """Load gait library specific data from YAML."""
        # This is handled by the individual gait configs
        pass
    
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
                        ee_tracker = EndEffectorTracker(config.constraint_specs, None)
                        config.reorder_and_remap(cfg, ee_tracker, device)
                else:
                    config.reorder_and_remap(cfg, cfg.robot, device)
        
        # Recompute batched control points after remapping
        self._precompute_control_points()
    
    def get_ref_traj(self, hzd_cmd) -> tuple[torch.Tensor, torch.Tensor]:
        """Get reference trajectory using precomputed batched control points."""
        # Get current phase and step duration
        base_velocity = hzd_cmd.env.command_manager.get_command("base_velocity")
        N = base_velocity.shape[0]
        
        # Use a single tau and step_dur value (same for all gaits)
        tau = torch.tensor([hzd_cmd.phase_var], device=base_velocity.device)  # [1]
        step_dur = torch.tensor([self.T], dtype=torch.float32, device=base_velocity.device)  # [1]
        
        # Select control points based on stance
        if hzd_cmd.stance_idx == 1:
            # Right stance: use right coefficients
            control_points = self.right_coeffs_batched  # [num_vel, jt_dim, degree+1]
        else:
            # Left stance: use left coefficients
            control_points = self.left_coeffs_batched  # [num_vel, jt_dim, degree+1]
        
        # Evaluate one trajectory per gait (much more efficient!)
        # control_points: [num_vel, jt_dim, degree+1]
        # tau: [1] -> expand to [num_vel]
        # step_dur: [1] -> expand to [num_vel]
        
        # Expand tau and step_dur to match control_points batch dimension
        tau_expanded = tau.expand(control_points.shape[0])  # [num_vel]
        step_dur_expanded = step_dur.expand(control_points.shape[0])  # [num_vel]
        
        # Use batched Bezier evaluation for position (order=0)
        ref_pos = bezier_deg_batched(
            order=0,
            tau=tau_expanded,
            step_dur=step_dur_expanded,
            control_points=control_points,
            degree=self.bez_deg
        )  # [num_vel, jt_dim]
        
        # Use batched Bezier evaluation for velocity (order=1)
        ref_vel = bezier_deg_batched(
            order=1,
            tau=tau_expanded,
            step_dur=step_dur_expanded,
            control_points=control_points,
            degree=self.bez_deg
        )  # [num_vel, jt_dim]
        
        # Select the appropriate gait for each environment
        gait_indices = self.select_gaits_by_velocity(base_velocity[:, :2])  # [N]
        
        # Gather the selected trajectories
        des_pos = ref_pos[gait_indices]  # [N, jt_dim]
        des_vel = ref_vel[gait_indices]  # [N, jt_dim]

        des_pos, des_vel = self._apply_swing_modifications(hzd_cmd, des_pos, des_vel, base_velocity)
        # import pdb; pdb.set_trace()
        return des_pos, des_vel
    
    def _apply_swing_modifications(self, hzd_cmd, des_pos, des_vel, base_velocity):
        """Apply swing modifications for all gaits (same for all gait types)."""
        # Get the first gait config (all gaits use the same modifications)
        first_gait = list(self.gait_velocity_ranges.keys())[0]
        config = self._gait_cache[first_gait]
        
        # Apply swing modifications for all environments
        des_pos, des_vel = config._apply_swing_modifications(hzd_cmd, des_pos, des_vel, base_velocity)
        return des_pos, des_vel
    
    def get_actual_traj(self, hzd_cmd):
        """Get actual trajectory - use the first gait's method since they're all the same."""
        first_gait = list(self.gait_velocity_ranges.keys())[0]
        config = self._gait_cache[first_gait]
        return config.get_actual_traj(hzd_cmd)
    
    def get_available_gaits(self) -> List[str]:
        """Get list of available gait names."""
        return list(self.gait_velocity_ranges.keys())
    
    def get_available_speeds(self) -> List[int]:
        """Get list of available speeds in cm/s."""
        available_gaits = self._discover_available_gaits()
        return list(available_gaits.values())


class GaitLibraryEndEffectorConfig(GaitLibraryTrajectoryConfig):
    """Specialized gait library for end-effector trajectories."""
    
    def __init__(self, gait_library_path: str, 
                 gait_velocity_ranges: Union[Dict[str, tuple], Tuple[float, float, float]], 
                 config_name: str = "single_support"):
        super().__init__(gait_library_path, gait_velocity_ranges, "end_effector", config_name)


class GaitLibraryJointConfig(GaitLibraryTrajectoryConfig):
    """Specialized gait library for joint trajectories."""
    
    def __init__(self, gait_library_path: str, 
                 gait_velocity_ranges: Union[Dict[str, tuple], Tuple[float, float, float]], 
                 config_name: str = "single_support"):
        super().__init__(gait_library_path, gait_velocity_ranges, "joint", config_name)


class StairGaitLibraryTrajectoryConfig(GaitLibraryTrajectoryConfig):
    """Gait library for stairs: selects gait based on stair height instead of velocity."""
    def __init__(self, gait_library_path, gait_height_ranges, trajectory_type="end_effector", config_name="single_support"):
        
        # Pass dummy velocity ranges to parent (not used)
     #    dummy_velocity_ranges = {name: (0, 1) for name in gait_height_ranges}

        min_vel, max_vel, step = gait_height_ranges
        generated_gait_height_ranges = self._generate_discretized_ranges(min_vel, max_vel, step)
        self.gait_height_ranges = generated_gait_height_ranges

        super().__init__(gait_library_path, generated_gait_height_ranges, trajectory_type, config_name)

    def select_gaits_by_height(self, stair_heights: torch.Tensor) -> torch.Tensor:
        gait_names = sorted(self.gait_height_ranges.keys(), key=lambda n: self.gait_height_ranges[n][0])
        boundaries = torch.tensor(
            [self.gait_height_ranges[name][1] for name in gait_names[:-1]],
            device=stair_heights.device,
            dtype=stair_heights.dtype,
        )
        gait_indices = torch.bucketize(stair_heights, boundaries, right=False)
        gait_indices = torch.clamp(gait_indices, 0, len(gait_names) - 1)
        return gait_indices

#     def _gait_name_to_speed_cms(self, gait_name: str) -> int:
#         """Convert gait name to speed in cm/s based on velocity ranges."""
#         min_vel, max_vel = self.gait_velocity_ranges[gait_name]
#         # Use the midpoint of the velocity range and convert to cm/s
#         speed_ms = (min_vel + max_vel) / 2
#         speed_cms = int(speed_ms * 100)  # Convert m/s to cm/s
#         return speed_cms
    def get_ref_traj(self, hzd_cmd):
        # Get stair heights from hzd_cmd (assumes hzd_cmd.z_height exists)

        cfg = hzd_cmd.env.cfg.scene.terrain.terrain_generator.sub_terrains['stairs']
        env_origins = hzd_cmd.env.scene.env_origins
        stair_heights = get_uniform_stair_step_height_from_env(env_origins,cfg)
     #    import pdb; pdb.set_trace()
        tau = torch.tensor([hzd_cmd.phase_var], device=stair_heights.device)
        step_dur = torch.tensor([self.T], dtype=torch.float32, device=stair_heights.device)
        if hzd_cmd.stance_idx == 1:
            control_points = self.right_coeffs_batched
        else:
            control_points = self.left_coeffs_batched
        tau_expanded = tau.expand(control_points.shape[0])
        step_dur_expanded = step_dur.expand(control_points.shape[0])
        ref_pos = bezier_deg_batched(0, tau_expanded, step_dur_expanded, control_points, self.bez_deg)
        ref_vel = bezier_deg_batched(1, tau_expanded, step_dur_expanded, control_points, self.bez_deg)
        gait_indices = self.select_gaits_by_height(stair_heights)

        des_pos = ref_pos[gait_indices.view(-1)]
        des_vel = ref_vel[gait_indices.view(-1)]
        return des_pos, des_vel