from enum import Enum
from pathlib import Path

import torch
import yaml
from dataclasses import dataclass
import math
import numpy as np

class TrajectoryType(Enum):
    HALF_PERIODIC = "half_periodic"
    FULL_PERIODIC = "full_periodic"
    EPISODIC = "episodic"
    PERPETUAL = "perpetual"

@dataclass
class DomainData:
    domain_name: str
    bezier_coeffs: dict
    bezier_tensor: torch.Tensor
    contact_frames: list[str]
    time: float
    bezier_frame: str           # Frame of reference used for the splines
    bezier_frame_domain: str    # Domain in which the frame was measured

@dataclass
class TrajectoryData:
    name: str
    domain_order: list[str]
    domain_data: dict[str, DomainData]
    num_outputs: int
    output_names: list[str]
    spline_order: int
    trajectory_type: TrajectoryType
    total_time: float
    conditioner: list[float]
    reference_frames: list[str]     # List of the bezier frames


class TrajectoryManager:
    """Manages a single trajectory. The trajectory is specified in a yaml file."""

    def __init__(self, traj_path: str, device):
        self.device = device

        # Resolve the trajectory file path
        self.traj_path = self._resolve_trajectory_path(traj_path)

        # Load the trajectory and corresponding information
        self.traj_data = self.load_from_yaml()

        # Load some data into more efficient data structures
        self.num_domains = len(self.traj_data.domain_order)

        self.T = torch.zeros(self.num_domains, device=self.device)
        self.bezier_coeffs = torch.zeros(self.num_domains, self.traj_data.num_outputs, self.traj_data.spline_order + 1, device=self.device)
        for i, domain in enumerate(self.traj_data.domain_order):
            self.T[i] = self.traj_data.domain_data[domain].time
            self.bezier_coeffs[i, :, :] = self.traj_data.domain_data[domain].bezier_tensor

        # Pre-compute relabeling matrix as a tensor
        R_numpy = self.relable_ee_stance_coeffs()
        self.R_relabel = torch.from_numpy(R_numpy).to(device=self.device, dtype=self.bezier_coeffs.dtype)


    def _resolve_trajectory_path(self, traj_path: str) -> str:
        """
        Resolve the trajectory path. If it's a folder, find the single YAML file in it.

        Args:
            traj_path: Path to a trajectory file or folder

        Returns:
            Resolved path to the trajectory YAML file

        Raises:
            FileNotFoundError: If the path doesn't exist
            ValueError: If the path is a folder with zero or multiple YAML files
        """
        path = Path(traj_path)

        if not path.exists():
            raise FileNotFoundError(f"Trajectory path does not exist: {traj_path}")

        # If it's a file, return as-is
        if path.is_file():
            return str(path)

        # If it's a directory, look for a single YAML file
        if path.is_dir():
            yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))

            if len(yaml_files) == 0:
                raise ValueError(
                    f"Trajectory folder contains no YAML files: {traj_path}"
                )
            elif len(yaml_files) > 1:
                raise ValueError(
                    f"Trajectory folder contains multiple YAML files ({len(yaml_files)} found). "
                    f"Expected exactly one file.\n"
                    f"Files found: {[f.name for f in yaml_files]}\n"
                    f"Folder: {traj_path}"
                )

            return str(yaml_files[0])

        raise ValueError(f"Path is neither a file nor a directory: {traj_path}")

    def load_from_yaml(self):
        """Loads all the relevant information from the trajectory file."""

        # Open yaml file
        with open(self.traj_path, 'r') as f:
            data = yaml.safe_load(f)

        # Verify that each domain has the same bezier outputs (frames and joints) and spline order
        # Also get the output information (count and names)
        # Do this first so we know the order for creating tensors
        num_outputs, output_names, spline_order = self._verify_consistent_outputs_and_get_info(
            {domain: (data[domain]['bezier_coeffs'], data[domain]['spline_order'])
             for domain in data['domain_sequence']},
            data['domain_sequence']
        )

        # Iterate through the domains and create tensors
        domain_data_dict = {}
        total_time = 0
        ref_frames = []
        for domain in data['domain_sequence']:
            domain_yaml = data[domain]

            # Create the bezier coefficient tensor in the same order as output_names
            bezier_tensor = self._create_bezier_tensor(
                domain_yaml['bezier_coeffs'],
                output_names
            )

            domain_data = DomainData(
                domain_name=domain,
                bezier_coeffs=domain_yaml['bezier_coeffs'],
                bezier_tensor=bezier_tensor,
                time=domain_yaml['T'][0],
                contact_frames=domain_yaml['contact_bodies'],
                bezier_frame=domain_yaml['ref_frame'],
                bezier_frame_domain=domain_yaml['ref_frame_domain']
            )

            domain_data_dict[domain] = domain_data
            total_time += domain_yaml['T'][0]
            ref_frames.append(domain_data.bezier_frame)

        return TrajectoryData(
            # Load trajectory name
            name=data['name'],
            # Load domain order
            domain_order=data['domain_sequence'],
            # Domain data
            domain_data=domain_data_dict,
            # Output information
            num_outputs=num_outputs,
            output_names=output_names,
            # Spline order (same for all domains)
            spline_order=spline_order,
            # Type
            trajectory_type=TrajectoryType(data['type']),
            # Total time
            total_time=total_time,
            # Conditioning variables
            conditioner=data['conditioner'],
            # Reference frames
            reference_frames=ref_frames,
        )

    def get_phasing_var(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the phasing variable which is a number in [0,1] that tells how far through the trajectory we are.

        For half periodic trajectories the [0, 1] is scaled to the full period. So 0.5 is the end of the provided trajectory.
        """
        # Map time based on trajectory time
        if self.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
            # Determine which half of the period the time is in
            full_period_time = self.traj_data.total_time * 2
            t_into_traj = t % full_period_time

            return t_into_traj / self.traj_data.total_time
        elif self.traj_data.trajectory_type == TrajectoryType.FULL_PERIODIC:
            # Map into the period time
            t_into_traj = t % self.traj_data.total_time
            return t_into_traj / self.traj_data.total_time

        elif self.traj_data.trajectory_type == TrajectoryType.EPISODIC:
            return torch.clamp(t / self.traj_data.total_time, 0, 1)
        elif self.traj_data.trajectory_type == TrajectoryType.PERPETUAL:
            # For now all perpetual motions are assumed to have no phasing associated with them
            return torch.zeros(t.shape[0], self.device)
        else:
            raise NotImplementedError(f"Trajectory type {self.traj_data.trajectory_type} is not implemented.")

    @property
    def get_output_names(self):
        return self.traj_data.output_names

    def get_output(self, t: torch.Tensor     # [N]
                          ) -> torch.Tensor:
        """
        Compute the bezier values at a given time.

        Args:
            t: shape [N] where N is the number of times that need to be queried.

        Returns:
            outputs: shape [N, 2, num_outputs] where num_outputs is the number of outputs of the trajectory,
                and we have one for the position and one for the velocity.

        To compute this value, we first determine which domain the times are in then we compute the values in a batched manner.
        """
        # Map time based on trajectory time
        if self.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
            # Determine which half of the period the time is in
            full_period_time = self.traj_data.total_time * 2
            t_into_traj = t % full_period_time

            second_half = t_into_traj/full_period_time > 0.5

            # In the second half
            t_into_traj[second_half] = t_into_traj[second_half] - self.traj_data.total_time

            # Need to remap the trajectory
            bezier_coeffs = self.remap_trajectory()
        elif self.traj_data.trajectory_type == TrajectoryType.FULL_PERIODIC:
            # Map into the period time
            t_into_traj = t % self.traj_data.total_time

            bezier_coeffs = self.bezier_coeffs  # TODO: Is this inefficient? Is it copying or referencing?
        elif self.traj_data.trajectory_type == TrajectoryType.EPISODIC or self.traj_data.trajectory_type == TrajectoryType.PERPETUAL:
            # Don't need to touch this
            t_into_traj = t
            bezier_coeffs = self.bezier_coeffs
        else:
            raise NotImplementedError(f"Trajectory type {self.traj_data.trajectory_type} is not implemented.")

        # Determine the domain for each time
        domain_boundaries = torch.cumsum(torch.cat([torch.tensor([0.0], device=self.device), self.T]), dim=0)

        domain_indicies = self.get_current_domains(t)

        # Normalize the time relative to the domain
        domain_start_times = domain_boundaries[domain_indicies]
        tau = (t_into_traj - domain_start_times)/self.T[domain_indicies]

        # Compute outputs
        outputs = torch.zeros(t.shape[0], 2, self.traj_data.num_outputs, device=self.device)
        outputs[:, 0, :] = self._compute_bezier_interp(0, tau, bezier_coeffs[0, :, :].squeeze(), self.T[domain_indicies])

        # Get the unique domains
        unique_domains = torch.unique(domain_indicies)

        for dom in unique_domains:
            current_domains = domain_indicies == unique_domains # TODO: Check
            outputs[current_domains, 0, :] = self._compute_bezier_interp(0, tau, bezier_coeffs[dom, :, :].squeeze(), self.T[domain_indicies])
            outputs[current_domains, 1, :] = self._compute_bezier_interp(1, tau, bezier_coeffs[dom, :, :].squeeze(), self.T[domain_indicies])

        return outputs

    def get_contact_state(self, t: torch.Tensor,    # [N]
                           contact_frames: list[str],
                           ) -> torch.Tensor:
        """
        Return the contact state of each contact point at the given time.

        Args:
            t: shape [N] where N is the number of times that need to be queried.
            contact_frames: shape: [num_contacts] list of frames to check the state for,

        Returns:
            contact_states: shape [N, num_contacts] where num_contacts is the number
             of contact points. A 1 indicates in contact, 0 otherwise.
        """
        # Determine the domain for each time
        domain_indicies = self.get_current_domains(t)

        # Initialize contact states tensor
        N = t.shape[0]
        num_contacts = len(contact_frames)
        contact_states = torch.zeros(N, num_contacts, device=self.device)

        # For each contact frame, check if it's in contact across all domains
        for i, frame in enumerate(contact_frames):
            # For each domain, check if this frame is in contact
            for domain_idx in range(self.num_domains):
                domain_name = self.traj_data.domain_order[domain_idx]
                domain_contact_frames = self.traj_data.domain_data[domain_name].contact_frames

                if frame in domain_contact_frames:
                    # Set all times in this domain to 1 for this contact frame
                    mask = domain_indicies == domain_idx
                    contact_states[mask, i] = 1.0

        return contact_states

    def get_current_domains(self, t: torch.Tensor) -> torch.Tensor:
        """
        Determine what domain each env is in given the time.
        """
        # Determine the domain for each time
        domain_boundaries = torch.cumsum(torch.cat([torch.tensor([0.0], device=self.device), self.T]), dim=0)

        domains = torch.searchsorted(domain_boundaries, t, right=False) - 1

        # Clamp to valid domain range [0, num_domains-1]
        domains = torch.clamp(domains, 0, self.num_domains - 1)

        return domains


    def get_ref_frames_in_use(self, t: torch.Tensor, ref_frames: list[str]) -> torch.Tensor:
        """
        Determine the reference frame in use.

        Args:
            t: shape [N] where N is the number envs.
            ref_frames: a list of reference frames.

        Returns:
            frame_tensor: a torch tensor of shape [N] where each scalar is the index into ref_frames for the active frame.
        """
        domain_indices = self.get_current_domains(t)

        # Create a mapping tensor: domain_idx -> ref_frame_idx
        # This is done once per call, but it's O(num_domains) not O(num_envs)
        domain_to_ref_frame_idx = torch.zeros(self.num_domains, dtype=torch.long, device=self.device)

        for domain_idx, domain_name in enumerate(self.traj_data.domain_order):
            bezier_frame = self.traj_data.domain_data[domain_name].bezier_frame
            if bezier_frame in ref_frames:
                domain_to_ref_frame_idx[domain_idx] = ref_frames.index(bezier_frame)
            else:
                raise ValueError(f"Bezier frame '{bezier_frame}' from domain '{domain_name}' not found in ref_frames list: {ref_frames}")

        # Use advanced indexing to get frame indices for all envs at once
        frame_indices = domain_to_ref_frame_idx[domain_indices]

        return frame_indices


    def remap_trajectory(self) -> torch.Tensor:
        """
        Remap the trajectory
        """
        # Apply relabeling: left_coeffs = R @ right_coeffs
        remap_coeffs = self.R_relabel @ self.bezier_coeffs

        # self.generate_axis_names()

        return remap_coeffs

    @staticmethod
    def _compute_bezier_interp(derivative: int,         # 0 -> position, 1 -> velocity
                               tau: torch.Tensor,       # [N] where tau \in [0, 1] time into the spline
                               ctrl_pts: torch.Tensor,  # [num_outputs, degree + 1]
                               T: torch.Tensor,
                               ) -> torch.Tensor:
        """
        Compute the point in the bezier curve.
        """
        # Clamp tau into [0,1]
        tau = torch.clamp(tau, 0.0, 1.0)  # [batch]

        degree = ctrl_pts.shape[1] - 1

        if derivative == 1:
            # ─── DERIVATIVE CASE ────────────────────────────────────────────────────
            # We want:
            #   B'(τ) = degree * sum_{i=0..degree-1} [
            #             (CP_{i+1} - CP_i) * C(degree-1, i)
            #             * (1-τ)^(degree-1-i) * τ^i
            #          ]  / step_dur.

            # 3) Compute CP differences along the "degree+1" axis:
            #    cp_diff: [n_dim, degree], where
            #      cp_diff[:, i] = control_points[:, i+1] - control_points[:, i].
            cp_diff = ctrl_pts[:, 1:] - ctrl_pts[:, :-1]  # [n_dim, degree]

            # 4) Binomial coefficients for (degree-1 choose i), i=0..degree-1:
            #    coefs_diff: [degree].
            coefs_diff = torch.tensor(
                [_ncr(degree - 1, i) for i in range(degree)],
                dtype=ctrl_pts.dtype,
                device=ctrl_pts.device
            )  # [degree]

            # 5) Build (τ^i) and ((1-τ)^(degree-1-i)) for i=0..degree-1:
            i_vec = torch.arange(degree, device=ctrl_pts.device)  # [degree]

            #    tau_pow:     [batch, degree],  τ^i
            tau_pow = tau.unsqueeze(1).pow(i_vec.unsqueeze(0))

            #    one_minus_pow: [batch, degree], (1-τ)^(degree-1-i)
            one_minus_pow = (1 - tau).unsqueeze(1).pow((degree - 1 - i_vec).unsqueeze(0))

            # 6) Combine into a single "weight matrix" for the derivative:
            #    weight_deriv[b, i] = degree * C(degree-1, i) * (1-τ[b])^(degree-1-i) * (τ[b])^i
            #    → shape [batch, degree]
            weight_deriv = (degree
                            * coefs_diff.unsqueeze(0)  # [1, degree]
                            * one_minus_pow  # [batch, degree]
                            * tau_pow)  # [batch, degree]
            # Now weight_deriv: [batch, degree]

            # 7) Multiply these weights by cp_diff to get a [batch, n_dim] result:
            #    For each batch b:  B'_b =  Σ_{i=0..degree-1} weight_deriv[b,i] * cp_diff[:,i],
            #    which is exactly a mat‐mul:  weight_deriv[b,:] @ (cp_diff^T) → [n_dim].
            #
            #    cp_diff^T: [degree, n_dim], so (weight_deriv @ cp_diff^T) → [batch, n_dim].
            Bdot = torch.matmul(weight_deriv, cp_diff.transpose(0, 1))  # [batch, n_dim]

            # 8) Finally divide by step_dur:
            return Bdot / T.unsqueeze(1)  # [batch, n_dim]
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
                dtype=ctrl_pts.dtype,
                device=ctrl_pts.device
            )  # [degree+1]

            # 4) Build τ^i and (1-τ)^(degree-i) for i=0..degree:
            i_vec = torch.arange(degree + 1, device=ctrl_pts.device)  # [degree+1]

            #    tau_pow:        [batch, degree+1]
            tau_pow = tau.unsqueeze(1).pow(i_vec.unsqueeze(0))

            #    one_minus_pow:  [batch, degree+1]
            one_minus_pow = (1 - tau).unsqueeze(1).pow((degree - i_vec).unsqueeze(0))

            # 5) Combine into a "weight matrix" for position:
            #    weight_pos[b, i] = C(degree, i) * (1-τ[b])^(degree-i) * (τ[b])^i.
            #    → shape [batch, degree+1]
            weight_pos = (coefs_pos.unsqueeze(0)  # [1, degree+1]
                          * one_minus_pow  # [batch, degree+1]
                          * tau_pow)  # [batch, degree+1]
            # Now weight_pos: [batch, degree+1]

            # 6) Multiply by control_points to get [batch, n_dim]:
            #    For each batch b:  B_b = Σ_{i=0..degree} weight_pos[b,i] * control_points[:,i],
            #    i.e.  weight_pos[b,:]  (shape [degree+1]) @ (control_points^T) ([degree+1, n_dim]) → [n_dim].
            #
            #    So:  B = weight_pos @ control_points^T  → [batch, n_dim].
            B = torch.matmul(weight_pos, ctrl_pts.transpose(0, 1))  # [batch, n_dim]

            return B

    def _create_bezier_tensor(self, bezier_coeffs: dict, output_names: list[str]) -> torch.Tensor:
        """
        Create a tensor of bezier coefficients in the order specified by output_names.

        Args:
            bezier_coeffs: Dictionary containing 'frames' and 'joints' with their bezier coefficients
            output_names: List of output names in the format "frame:axis" or "joint:name"

        Returns:
            bezier_tensor: Shape [num_outputs, degree+1] containing coefficients for each output
        """
        coefficient_lists = []

        for output_name in output_names:
            if output_name.startswith('joint:'):
                # Joint output
                joint_name = output_name.split(':', 1)[1]
                coeffs = bezier_coeffs['joints'][joint_name]
            else:
                # Frame output (format: "frame_name:axis")
                frame_name, axis = output_name.split(':', 1)
                coeffs = bezier_coeffs['frames'][frame_name][axis]

            coefficient_lists.append(coeffs)

        # Stack into tensor: [num_outputs, degree+1]
        bezier_tensor = torch.tensor(coefficient_lists, dtype=torch.float32, device=self.device)

        return bezier_tensor

    @staticmethod
    def _verify_consistent_outputs_and_get_info(domain_data: dict, domain_sequence: list) -> tuple[int, list[str], int]:
        """
        Verify that all domains have the same frames, joints, and spline order in their bezier coefficients.
        The numerical values can differ, but the structure must be identical.

        Args:
            domain_data: Dictionary mapping domain names to tuples of (bezier_coeffs, spline_order)
            domain_sequence: Ordered list of domain names

        Returns:
            num_outputs: Total number of outputs (sum of all frame axes + joints)
            output_names: List of output names in the format "frame:axis" or "joint:name"
            spline_order: The spline order (verified to be consistent across all domains)

        Raises:
            ValueError: If frames, joints, or spline order are inconsistent across domains
        """
        if len(domain_sequence) == 0:
            return 0, [], 0

        # Get the reference domain (first one)
        reference_domain = domain_sequence[0]
        ref_bezier, ref_spline_order = domain_data[reference_domain]

        # Extract reference frames and their axes
        ref_frames = set(ref_bezier['frames'].keys())
        ref_frame_axes = {
            frame: set(axes.keys())
            for frame, axes in ref_bezier['frames'].items()
        }

        # Extract reference joints
        ref_joints = set(ref_bezier['joints'].keys())

        # Verify all other domains match
        for domain_name in domain_sequence[1:]:
            curr_bezier, curr_spline_order = domain_data[domain_name]

            # Check spline order matches
            if curr_spline_order != ref_spline_order:
                raise ValueError(
                    f"Domain '{domain_name}' has different spline order than '{reference_domain}'.\n"
                    f"  Reference spline order: {ref_spline_order}\n"
                    f"  Current spline order: {curr_spline_order}"
                )

            # Check frames match
            curr_frames = set(curr_bezier['frames'].keys())
            if curr_frames != ref_frames:
                raise ValueError(
                    f"Domain '{domain_name}' has different frames than '{reference_domain}'.\n"
                    f"  Reference frames: {sorted(ref_frames)}\n"
                    f"  Current frames: {sorted(curr_frames)}\n"
                    f"  Missing: {sorted(ref_frames - curr_frames)}\n"
                    f"  Extra: {sorted(curr_frames - ref_frames)}"
                )

            # Check each frame has the same axes
            for frame in ref_frames:
                curr_axes = set(curr_bezier['frames'][frame].keys())
                ref_axes = ref_frame_axes[frame]

                if curr_axes != ref_axes:
                    raise ValueError(
                        f"Domain '{domain_name}' frame '{frame}' has different axes than '{reference_domain}'.\n"
                        f"  Reference axes: {sorted(ref_axes)}\n"
                        f"  Current axes: {sorted(curr_axes)}\n"
                        f"  Missing: {sorted(ref_axes - curr_axes)}\n"
                        f"  Extra: {sorted(curr_axes - ref_axes)}"
                    )

            # Check joints match
            curr_joints = set(curr_bezier['joints'].keys())
            if curr_joints != ref_joints:
                raise ValueError(
                    f"Domain '{domain_name}' has different joints than '{reference_domain}'.\n"
                    f"  Reference joints: {sorted(ref_joints)}\n"
                    f"  Current joints: {sorted(curr_joints)}\n"
                    f"  Missing: {sorted(ref_joints - curr_joints)}\n"
                    f"  Extra: {sorted(curr_joints - ref_joints)}"
                )

        # Build output names list and count outputs (preserving YAML order)
        output_names = []

        # Add frame outputs (in YAML order)
        for frame in ref_bezier['frames'].keys():
            for axis in ref_bezier['frames'][frame].keys():
                output_names.append(f"{frame}:{axis}")

        # Add joint outputs (in YAML order)
        for joint in ref_bezier['joints'].keys():
            output_names.append(f"joint:{joint}")

        num_outputs = len(output_names)

        return num_outputs, output_names, ref_spline_order

    # TODO: Clean
    def relable_ee_stance_coeffs(self):
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

    # TODO: Clean
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

def _ncr(n, r):
    return math.comb(n, r)

