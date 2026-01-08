import os
from pathlib import Path
import torch
from .manager_base import ManagerBase
from .trajectory_manager import TrajectoryManager

# TODO: Test with new changes
class LibraryManager(ManagerBase):
    """Manages a library of trajectories, selecting the appropriate one based on a conditioning variable."""

    def __init__(self, library_folder_path: str, hf_repo: str, device,
                 env=None, conditioner_generator_name: str = None):
        self.folder_path = library_folder_path
        self.device = device
        self.env = env
        self.conditioner_generator_name = conditioner_generator_name
        self.trajectory_managers = []
        self.conditioning_vars = None
        self.num_outputs = None

        self.load_library(hf_repo)


    def load_library(self, hf_repo: str):
        """Load all trajectory files from the library folder into a list."""
        if hf_repo is None:
            library_path = Path(self.folder_path)
        else:
            library_path = self._get_from_hugging_face(hf_repo, self.folder_path)

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
            traj_manager = TrajectoryManager(str(yaml_file), None, self.device)
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

    def _get_from_hugging_face(self, hf_repo: str, hf_path: str) -> Path:
        """
        Load the trajectory library folder from hugging face. Download it into the hf folder.
        Make the /hf/ folder if it doesn't exist.

        Args:
            hf_repo: hugging face repo to use (e.g., 'username/repo-name')
            hf_path: the path to the trajectory folder in the hf repo (e.g., 'trajectories/library')

        Returns:
            Local Path to the downloaded trajectory folder
        """
        import os

        # Get the robot_rl root directory and go two folders above it
        root = os.getcwd() #os.environ.get("ROBOT_RL_ROOT", os.getcwd())
        hf_base = os.path.join(root)
        hf_base = os.path.abspath(hf_base)  # Resolve to absolute path

        # Create cache directory in the hf folder
        cache_dir = os.path.join(hf_base, "hf")
        os.makedirs(cache_dir, exist_ok=True)

        # The local path to the trajectory folder
        local_folder_path = os.path.join(cache_dir, hf_path)

        # Check if folder already exists locally and has YAML files
        if os.path.exists(local_folder_path) and os.path.isdir(local_folder_path):
            yaml_files = list(Path(local_folder_path).glob("*.yaml")) + list(Path(local_folder_path).glob("*.yml"))
            if len(yaml_files) > 0:
                print(f"Using cached trajectory library from {local_folder_path}")
                return Path(local_folder_path)

        # Download from Hugging Face
        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading trajectory library {hf_path} from {hf_repo}...")

            # Download the entire repo or specific folder
            snapshot_download(
                repo_id=hf_repo,
                allow_patterns=f"{hf_path}/*",  # Download only files in the specified folder
                local_dir=cache_dir,
            )

            print(f"Successfully downloaded trajectory library to {local_folder_path}")
            return Path(local_folder_path)

        except ImportError:
            raise RuntimeError("huggingface_hub is required for downloading trajectories. Install with: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"Failed to download trajectory library from Hugging Face: {e}")

    def get_conditioner_var(self) -> torch.Tensor:
        """Get the conditioner variable from the environment's command manager.

        Returns:
            torch.Tensor: The conditioning variable for each environment, shape [N].
        """
        cond_term = self.env.command_manager.get_term(self.conditioner_generator_name)
        return cond_term.command[:, 0]

    @property
    def get_output_names(self):
        return self.output_names

    def get_reference_frames(self):
        return self.ref_frames

    def get_num_outputs(self) -> int:
        """Get the total number of outputs in the trajectory.

        Returns:
            The number of outputs.
        """
        return self.num_outputs

    def get_num_domains(self):
        return self.trajectory_managers[0].get_num_domains()

    def get_trajectory_type(self):
        return self.trajectory_type

    def get_phasing_var(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the phasing variable for each environment.

        Args:
            t: Time tensor of shape [N].

        Returns:
            Phasing variable tensor of shape [N].
        """
        conditioner = self.get_conditioner_var()
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

    def get_output(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the outputs to be tracked by the RL.

        Args:
            t: Time in each env, shape [N].

        Returns:
            Outputs to be tracked by the RL, shape [N, 2, num_outputs].
        """
        conditioner = self.get_conditioner_var()
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

    def get_ref_frames_in_use(self, t: torch.Tensor,
                              ref_frames: list[str]) -> torch.Tensor:
        """Determine the reference frame in use for each environment.

        Args:
            t: Time in each env, shape [N].
            ref_frames: List of reference frame names.

        Returns:
            Frame indices into ref_frames for the active frame in each env, shape [N].
        """
        conditioner = self.get_conditioner_var()
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

    def get_contact_state(self, t: torch.Tensor,
                          contact_frames: list[str]) -> torch.Tensor:
        """Get the contact states for each frame from the trajectory.

        Args:
            t: Time in each env, shape [N].
            contact_frames: List of contact frame names.

        Returns:
            Contact states for each frame from the trajectory, shape [N, num_contacts].
        """
        conditioner = self.get_conditioner_var()
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

    def get_current_domains(self, t: torch.Tensor) -> torch.Tensor:
        """Return the domain index for each env.

        Args:
            t: Time in each env, shape [N].

        Returns:
            Domain indices, shape [N].
        """
        conditioner = self.get_conditioner_var()
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
        # Ensure tensors are contiguous to avoid performance warnings
        indicies = torch.searchsorted(
            self.conditioning_vars[:, 0].contiguous(),
            conditioner.contiguous(),
            right=False
        ) - 1

        return torch.clamp(indicies, 0, len(self.trajectory_managers) - 1)

    def get_domain_times(self, t: torch.Tensor) -> torch.Tensor:
        """Get the duration of the current domain for each environment.

        Args:
            t: Time in each env, shape [N].

        Returns:
            Domain durations, shape [N].
        """
        conditioner = self.get_conditioner_var()
        traj_idx = self.get_traj_indices(conditioner)

        # Get the unique managers (avoid repeats)
        unique_indicies = torch.unique(traj_idx)

        domain_times = torch.zeros(conditioner.shape[0], dtype=torch.long, device=self.device)

        # Bin each conditioner by manager
        for idx in unique_indicies:
            # Find which environments use this trajectory
            mask = traj_idx == idx
            env_indices = torch.where(mask)[0]

            # Get times for these environments
            t_for_manager = t[env_indices]

            # Call get_output for this manager
            domain_time = self.trajectory_managers[idx.item()].get_domain_times(t_for_manager)

            # Place outputs in the correct positions
            domain_times[env_indices] = domain_time

        return domain_times

    def get_total_time(self):
        """
        Gets the total time for the trajectory. Assumes all trajectories in the library have the same total time.
        """

        return self.trajectory_managers[0].get_total_time()

    def order_outputs(self, order_output_names: list[str]):
        for manager in self.trajectory_managers:
            manager.order_outputs(order_output_names)
