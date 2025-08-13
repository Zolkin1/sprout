from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import torch
import yaml


class BaseTrajectoryConfig(ABC):
    """Abstract base class for trajectory configurations.

    This class provides common functionality for both joint and end effector
    trajectory configurations, including YAML loading, bezier coefficient
    management, and stance-based trajectory generation.
    """

    def __init__(self, yaml_path: str):
        """Initialize the base trajectory configuration.

        Args:
            yaml_path: Path to the YAML configuration file
        """
        self.yaml_path = yaml_path
        self.bezier_coeffs = {}
        self.T = 0.0
        self.left_coeffs = None
        self.right_coeffs = None
        self.bez_deg = 5  # Default Bezier degree
        self.load_from_yaml()

    def load_from_yaml(self):
        """Load configuration from YAML file.

        This method loads the step period T and calls the abstract method
        _load_specific_data() for subclass-specific data loading.
        """
        with open(self.yaml_path) as file:
            data = yaml.safe_load(file)
            domain_name = next(iter(data.keys()))

        # Load common data
        self.T = data[domain_name]["T"][0] if isinstance(data[domain_name]["T"], list) else data[domain_name]["T"]

        # Load initial config
        init_config = data[domain_name]["q"][0]
        # Need to reorder xyzw to wxyz

        init_vel = data[domain_name]["v"][0]
        self.init_root_state = np.concatenate(
            [init_config[:3], [init_config[6]], init_config[3:6]]
        )  # [pos_xyz, yaw, rpy]
        self.init_root_vel = init_vel[:6]
        self.init_joint_pos = init_config[7:]
        self.init_joint_vel = init_vel[6:]

        # Load subclass-specific data
        self._load_specific_data(data)

    @abstractmethod
    def _load_specific_data(self, data: dict):
        """Load subclass-specific data from YAML.

        Args:
            data: Parsed YAML data dictionary
        """
        pass

    @abstractmethod
    def reorder_and_remap(self, cfg, device):
        """Reorder and remap coefficients for left/right stance.

        Args:
            cfg: Configuration object
            device: Target device for tensors
        """
        pass

    def get_ref_traj(self, hzd_cmd) -> tuple[torch.Tensor, torch.Tensor]:
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
            0, phase_var_tensor, T, ctrl_points, torch.tensor(hzd_cmd.cfg.bez_deg, device=hzd_cmd.device)
        )

        des_vel = bezier_deg(1, phase_var_tensor, T, ctrl_points, hzd_cmd.cfg.bez_deg)

        # Apply stance-specific modifications
        self._apply_swing_modifications(hzd_cmd, des_pos, des_vel, base_velocity)

        return des_pos, des_vel

    @abstractmethod
    def _apply_swing_modifications(
        self, hzd_cmd, des_pos: torch.Tensor, des_vel: torch.Tensor, base_velocity: torch.Tensor
    ):
        """Apply swing-specific modifications to reference trajectory.

        Args:
            hzd_cmd: HZD command object
            des_pos: Desired position tensor
            des_vel: Desired velocity tensor
            base_velocity: Base velocity command
        """
        pass

    @abstractmethod
    def get_actual_traj(self, hzd_cmd) -> tuple[torch.Tensor, torch.Tensor]:
        """Get actual trajectory from robot state.

        Args:
            hzd_cmd: HZD command object

        Returns:
            Tuple of (actual_position, actual_velocity) tensors
        """
        pass

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
