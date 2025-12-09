import os
from pathlib import Path
import torch
from .trajectory_manager import TrajectoryManager


class LibraryManager:

    def __init__(self, library_folder_path: str, device):
        self.folder_path = library_folder_path
        self.device = device
        self.trajectory_managers = []
        self.conditioning_vars = None
        self.num_outputs = None

        self.load_library()


    def load_library(self):
        """Load all trajectory files from the library folder into a list."""
        # Iterate through the trajectory files in the library
        library_path = Path(self.folder_path)

        if not library_path.exists():
            raise FileNotFoundError(f"Library folder not found: {self.folder_path}")

        if not library_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.folder_path}")

        # Find all YAML files in the directory
        yaml_files = list(library_path.glob("*.yaml")) + list(library_path.glob("*.yml"))

        if len(yaml_files) == 0:
            raise ValueError(f"No YAML trajectory files found in: {self.folder_path}")

        # Make a trajectory manager object for each file
        traj_conditioner_pairs = []
        for yaml_file in yaml_files:
            traj_manager = TrajectoryManager(str(yaml_file), self.device)
            traj_conditioner_pairs.append((traj_manager, traj_manager.traj_data.conditioner))

        # Sort by first conditioning variable to allow searchsorted in get_output
        traj_conditioner_pairs.sort(key=lambda x: x[1][0])

        # Separate back into lists while maintaining sorted order
        self.trajectory_managers = [pair[0] for pair in traj_conditioner_pairs]
        conditioner_list = [pair[1] for pair in traj_conditioner_pairs]

        # Create conditioning tensor of shape (n_traj, 2)
        self.conditioning_vars = torch.tensor(conditioner_list, device=self.device)

        # Verify the trajectories are compatible (num_outputs, type, reference_frames)
        num_ouputs = self.trajectory_managers[0].traj_data.num_outputs
        output_names = self.trajectory_managers[0].traj_data.output_names
        trajectory_type = self.trajectory_managers[0].traj_data.trajectory_type
        ref_frames = self.trajectory_managers[0].traj_data.reference_frames
        for manager in self.trajectory_managers:
            if manager.traj_data.num_outputs != num_ouputs:
                raise ValueError(f"Trajectories in the library are not compatible! Varying number of outputs!")
            if manager.traj_data.trajectory_type != trajectory_type:
                raise ValueError(f"Trajectories in the library are not compatible! Varying trajectory_type!")
            if manager.traj_data.output_names != output_names:
                raise ValueError(f"Trajectories in the library are not compatible! Varying output_names!")
            if manager.traj_data.reference_frames != ref_frames:
                raise ValueError(f"Trajectories in the library are not compatible! Varying reference_frames!")

        self.trajectory_type = trajectory_type
        self.num_outputs = num_ouputs
        self.output_names = output_names
        self.ref_frames = ref_frames

    @property
    def get_output_names(self):
        return self.output_names

    def get_reference_frames(self):
        return self.ref_frames

    def get_phasing_var(self, conditioner: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        indices = self.get_traj_indices(conditioner)

        phasing_var = torch.zeros(t.shape[0], device=self.device)

        # Get the unique managers (avoid repeats)
        unique_indicies = torch.unique(indices)

        # Bin each conditioner by manager
        for idx in unique_indicies:
            # Find which environments use this trajectory
            mask = indices == idx
            env_indices = torch.where(mask)[0]

            # Get times for these environments
            t_for_manager = t[env_indices]

            # Call get_output for this manager
            manager_phasing_var = self.trajectory_managers[idx.item()].get_phasing_var(t_for_manager)

            # Place outputs in the correct positions
            phasing_var[env_indices] = manager_phasing_var

        return phasing_var

    def get_output(self, t: torch.Tensor,         # shape [N] where N is the number of environments
                   conditioner: torch.Tensor      # shape [N]
                   ) -> torch.Tensor:
        """
        Compute the outputs to be tracked by the RL.

        Args:
            t (torch.Tensor): time in each env. shape [N]
            conditioner (torch.Tensor): conditioning variable for each env. shape [N]

        Returns:
            outputs (torch.Tensor): outputs to be tracked by the RL. shape [N, 2, num_outputs]
        """
        indices = self.get_traj_indices(conditioner)

        # Initialize output tensor
        N = t.shape[0]
        outputs = torch.zeros(N, 2, self.num_outputs, device=self.device)

        # Get the unique managers (avoid repeats)
        unique_indicies = torch.unique(indices)

        # Bin each conditioner by manager
        for idx in unique_indicies:
            # Find which environments use this trajectory
            mask = indices == idx
            env_indices = torch.where(mask)[0]

            # Get times for these environments
            t_for_manager = t[env_indices]

            # Call get_output for this manager
            manager_outputs = self.trajectory_managers[idx.item()].get_output(t_for_manager)

            # Place outputs in the correct positions
            outputs[env_indices] = manager_outputs

        return outputs

    def get_ref_frames_in_use(self, conditioner: torch.Tensor,
                              t: torch.Tensor,
                              ref_frames: list[str]) -> torch.Tensor:
        """
        Determine the reference frame in use for each environment.

        Args:
            conditioner (torch.Tensor): conditioning variable for each env. shape [N]
            t (torch.Tensor): time in each env. shape [N]
            ref_frames (list[str]): list of reference frame names

        Returns:
            frame_indices (torch.Tensor): indices into ref_frames for the active frame in each env. shape [N]
        """
        indices = self.get_traj_indices(conditioner)

        # Initialize output tensor
        N = t.shape[0]
        frame_indices = torch.zeros(N, dtype=torch.long, device=self.device)

        # Get the unique managers (avoid repeats)
        unique_indicies = torch.unique(indices)

        # Bin each conditioner by manager
        for idx in unique_indicies:
            # Find which environments use this trajectory
            mask = indices == idx
            env_indices = torch.where(mask)[0]

            # Get times for these environments
            t_for_manager = t[env_indices]

            # Call get_ref_frames_in_use for this manager
            manager_frame_indices = self.trajectory_managers[idx.item()].get_ref_frames_in_use(t_for_manager, ref_frames)

            # Place outputs in the correct positions
            frame_indices[env_indices] = manager_frame_indices

        return frame_indices

    def get_contact_state(self, conditioner: torch.Tensor,     # shape: [N]
                           t: torch.Tensor,                     # shape: [N]
                           contact_frames: list[str]            # shape: [num_contacts]
                           ) -> torch.Tensor:
        """
        Get the contact states for each frame from the trajectory

        Args:
            conditioner (torch.Tensor): conditioning variable for each env. shape [N]
            t (torch.Tensor): time in each env. shape [N]
            contact_frames (list[str]): list of contact frame names

        Returns:
            contact_states (torch.Tensor): contact states for each frame from the trajectory. shape [N, num_contacts]
        """

        indices = self.get_traj_indices(conditioner)

        # Initialize output tensor
        N = t.shape[0]
        contact_states = torch.zeros(N, len(contact_frames), device=self.device)

        # Get the unique managers (avoid repeats)
        unique_indicies = torch.unique(indices)

        # Bin each conditioner by manager
        for idx in unique_indicies:
            # Find which environments use this trajectory
            mask = indices == idx
            env_indices = torch.where(mask)[0]

            # Get times for these environments
            t_for_manager = t[env_indices]

            # Call get_output for this manager
            manager_states = self.trajectory_managers[idx.item()].get_contact_state(t_for_manager, contact_frames)

            # Place outputs in the correct positions
            contact_states[env_indices] = manager_states

        return contact_states

    def get_current_domains(self, conditioner: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        """Return the domain index for each env."""
        traj_idx = self.get_traj_indices(conditioner)

        # Get the unique managers (avoid repeats)
        unique_indicies = torch.unique(traj_idx)

        domain_idx = torch.zeros(conditioner.shape[0], dtype=torch.long, device=self.device)

        # Bin each conditioner by manager
        for idx in unique_indicies:
            # Find which environments use this trajectory
            mask = traj_idx == idx
            env_indices = torch.where(mask)[0]

            # Get times for these environments
            t_for_manager = t[env_indices]

            # Call get_output for this manager
            manager_states = self.trajectory_managers[idx.item()].get_current_domains(t_for_manager)

            # Place outputs in the correct positions
            domain_idx[env_indices] = manager_states

        return domain_idx


    def get_traj_indices(self, conditioner: torch.Tensor) -> torch.Tensor:
        """Determine which trajectories are in use for each env."""
        # Determine which trajectory is in use. Use searchsorted
        indicies = torch.searchsorted(self.conditioning_vars[:, 0], conditioner, right=False) - 1

        return torch.clamp(indicies, 0, len(self.trajectory_managers) - 1)
