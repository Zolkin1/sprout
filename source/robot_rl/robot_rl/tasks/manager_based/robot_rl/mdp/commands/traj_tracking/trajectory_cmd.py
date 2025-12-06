import numpy as np
import re
import torch
from isaaclab.managers import CommandTerm

from .clf import CLF
from .library_manager import LibraryManager
from .trajectory_manager import TrajectoryManager
from .trajectory_manager import TrajectoryType

from isaaclab.utils.math import wrap_to_pi, quat_apply, quat_from_euler_xyz,euler_xyz_from_quat, wrap_to_pi, yaw_quat, quat_inv

class TrajectoryCommand(CommandTerm):
    """Trajectory command term. This keeps track of the underlying single trajectory or library as well as CLF for tracking."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.env = env
        self.robot = env.scene[cfg.asset_name]

        # Expand wildcards in contact frames
        self.contact_frames = self._expand_wildcard_frames(cfg.contact_frames)

        # Extract the index into the robot data bodies for the contact frames
        self.contact_frame_indices = torch.zeros(len(self.contact_frames), dtype=torch.long, device=self.device)
        for i, frame_name in enumerate(self.contact_frames):
            if frame_name in self.robot.body_names:
                self.contact_frame_indices[i] = self.robot.body_names.index(frame_name)
            else:
                raise ValueError(f"Contact frame '{frame_name}' not found in robot body names.")

        self.current_contact_poses = torch.zeros(self.num_envs, len(self.contact_frames), 6, dtype=torch.float, device=self.device)
        self.desired_contact_poses = torch.zeros(self.num_envs, len(self.contact_frames), 6, dtype=torch.float, device=self.device)

        self.manager_type = cfg.manager_type
        self.conditioner_generator = cfg.conditioner_generator_name

        self.y_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)

        self.y_des = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_des = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)

        # Create trajectory/library manager
        if cfg.manager_type == "trajectory":
            self.manager = TrajectoryManager(cfg.path, env.device)
            self.trajectory_type = self.manager.traj_data.trajectory_type
        elif cfg.manager_type == "library":
            self.manager = LibraryManager(cfg.path, env.device)
            self.trajectory_type = self.manager.trajectory_type
        else:
            raise NotImplementedError(f"Manager Type {cfg.manager_type} is not implemented!")

        self.verify_contact_frames()

        # Hold a list of current domains for each env
        self.current_domain = -1 * torch.ones(self.num_envs, device=self.device)

        # Create a list of indices to be used
        self.joint_idx, self.body_idx, self.use_com = self._parse_outputs(self.manager.get_output_names)

        # For now assuming that all bodies have a yaw tracking
        self.yaw_output_idxs = []
        for body in range(len(self.body_idx)):
            start = 0
            if self.use_com:
                start += 2
            self.yaw_output_idxs = start + 5*body

        # Create a list of indices for the reference frames
        self.ref_frame_indices, self.ref_frames = self._parse_ref_frames(self.manager.traj_data.reference_frames)

        # Verify all reference frames are in contact frames
        for ref_frame in self.ref_frames:
            if ref_frame not in self.contact_frames:
                raise ValueError(f"Reference frame '{ref_frame}' is not in the contact frames list: {self.contact_frames}")

        # Create a mapping from ref_frames to contact_frames indices
        # This allows us to map ref_frame_indices to contact_state indices
        self.ref_to_contact_idx = torch.zeros(len(self.ref_frames), dtype=torch.long, device=self.device)
        for i, ref_frame in enumerate(self.ref_frames):
            self.ref_to_contact_idx[i] = self.contact_frames.index(ref_frame)

        # Current reference frame poses
        self.ref_poses = torch.zeros((self.num_envs, 7), device=self.device)    # [N, [position, quat]]

        # Create CLF
        self.clf = CLF(
            self.cfg.num_outputs, self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            Q_weights=np.array(self.cfg.Q_weights),
            R_weights=np.array(self.cfg.R_weights),
            device=self.device
        )

    def get_phasing_var(self, t: torch.Tensor) -> torch.Tensor:
        if self.manager_type == "trajectory":
            return self.manager.get_phasing_var(t)
        elif self.manager_type == "library":
            conditioner = self.get_conditioner_var()
            return self.manager.get_phasing_var(conditioner, t)
        else:
            raise ValueError(f"Cannot get phasing variable for manager of type {self.manager_type}.")

    def get_conditioner_var(self) -> torch.Tensor:
        """Get the conditioner variable."""
        cond_term = self.env.command_manager.get_term(self.conditioner_generator)

        # TODO: For now just hard code the conditioning variable as velocity
        condition_vars = cond_term[:, 0]

        # condition_vars = cond_term.get_condition_vars() # TODO: Implement. This could come from a command or from terrain (i.e. stair height, slope)
        return condition_vars

    def _expand_wildcard_frames(self, frame_patterns: list[str]) -> list[str]:
        """
        Expand wildcard patterns in contact frame names.

        Args:
            frame_patterns: List of frame names that may contain wildcards (e.g., ".*_ankle_roll_link")

        Returns:
            List of explicit frame names with wildcards expanded
        """
        expanded_frames = []

        # Get all body names from the robot
        body_names = self.robot.body_names

        for pattern in frame_patterns:
            # Check if the pattern contains wildcards (. or *)
            if '*' in pattern or '.*' in pattern:
                # Convert glob-style pattern to regex
                # Replace .* with .* (already regex), and * with .*
                regex_pattern = pattern.replace('*', '.*') if not '.*' in pattern else pattern
                regex_pattern = f'^{regex_pattern}$'

                # Find all matching body names
                matched = False
                for body_name in body_names:
                    if re.match(regex_pattern, body_name):
                        expanded_frames.append(body_name)
                        matched = True

                if not matched:
                    raise ValueError(f"Wildcard pattern '{pattern}' did not match any body names in the robot.")
            else:
                # No wildcard, add as-is
                expanded_frames.append(pattern)

        return expanded_frames

    def get_contact_state(self, t: torch.Tensor):
        """
        Gets the desired contact state at the given time for the specified contact point.

        Args:
            t: shape [N] the times in each environment

        Returns:
            contact_states: shape [N, num_contacts]. A tensor of binary values with a 1 indicating in contact and 0 otherwise.
        """
        # For a library we need the conditioner value too
        if self.manager_type == "library":
            conditioner = self.get_conditioner_var()
            return self.manager.get_contact_state(conditioner, t, self.contact_frames)
        else:
            return self.manager.get_contact_state(t, self.contact_frames)

    def get_trajectory_type(self):
        """Gets the type of trajectory: periodic or episodic."""
        return self.trajectory_type

    def verify_contact_frames(self):
        traj_frames = []
        if self.manager_type == "trajectory":
            for domain in self.manager.traj_data.domain_data.values():
                traj_frames.append(domain.contact_frames)
        elif self.manager_type == "library":
            for manager in self.manager.trajectory_managers:
                for domain in manager.traj_data.domain_data:
                    traj_frames.append(domain.contact_frames)
        else:
            raise NotImplementedError(f"Manager Type {self.manager_type} is not implemented!")

        # Verify that every frame in traj_frames appears in self.contact_frames
        for frames in traj_frames:
            for frame in frames:
                if frame not in self.contact_frames:
                    raise ValueError(f"Contact frame {frame} from a trajectory is not in the contact frames list!")

    def get_ref_frame_poses(self) -> torch.Tensor:
        """
        Get the reference frame poses.

        Returns
            poses is shape [N, num_ref_frames]
        """
        poses = torch.zeros(self.num_envs, len(self.ref_frame_indices), 7, device=self.device)

        poses[:, :, :3] = self.robot.data.body_pos_w[:, self.ref_frame_indices]
        poses[:, :, 3:] = self.robot.data.body_quat_w[:, self.ref_frame_indices]

        return poses

    def get_contact_poses(self, contact_state: torch.Tensor) -> torch.Tensor:
        """
        Determine the pose of each frame that is in contact.

        The idea here will be to always grab the pose of all the frames but then mask it with 0's when out of contact

        Args:
            contact_state: shape [N, num_contacts]

        Returns:
            poses is shape [N, num_ref_frames, 6]
        """

        poses = torch.zeros(self.num_envs, len(self.contact_frames), 6, device=self.device)

        not_in_contact = contact_state == 0

        # Get the poses of all the possible contact bodies
        poses[:, :, :3] = self.robot.data.body_pos_w[:, self.contact_frame_indices, :]
        for i in range(len(self.contact_frames)):
            poses[:, i, 3:] = get_euler_from_quat(self.robot.data.body_quat_w[:, self.contact_frame_indices[i], :])

        # Now mask
        poses[not_in_contact, :] *= 0


        return poses

    def get_desired_contact_poses(self, changed: torch.Tensor, current_poses: torch.Tensor) -> torch.Tensor:
        """
        Get the desired contact poses. This is always the pose of the frame when it first makes contact.

        Mask all the current poses by if they are in contact and if the domain just changed (because we want where we just made contact)

        TODO: Need to consider what if we want to rotate but not translate?
        """

        # Check to make sure that there is at least one True in changed
        if torch.any(changed):
            self.desired_contact_poses[changed] = current_poses[changed]

        return self.desired_contact_poses


    def get_measured_outputs(self, t: torch.Tensor):
        """
        Get the measured state then compute the measured outputs.
        """
        ref_poses = self.get_ref_frame_poses()        # Get the pose of every frame that should be in contact

        # TODO: For now assume that reference frames are always contact bodies.
        #   Then only updated the reference frame if it changes domain into a domain with contact.
        #   Each env needs its own reference frame, but it only ever needs one at a time.
        #   If there is a half periodic trajectory then assume that it switches to the other (left or right) frame.
        #   The self.ref_poses should be of shape [N, 7] and should just be holding the current in use reference frame.

        # Get the current domains
        new_domains = self.manager.get_current_domains(t)

        # Check if the domains changed
        changed = new_domains != self.current_domain

        # Update the list of current domains
        self.current_domain = new_domains

        # Determine which reference frames/bodies are in contact
        contact_state = self.get_contact_state(t)  # Shape: [N, num_contact_frames]

        self.current_contact_poses = self.get_contact_poses(contact_state)
        self.desired_contact_poses = self.get_desired_contact_poses(changed, self.current_contact_poses)

        # Get the indices into self.ref_frames for the reference frame in use for each env
        if self.manager_type == "trajectory":
            ref_frame_indices = self.manager.get_ref_frames_in_use(t, self.ref_frames)  # Shape: [N]
        else:
            ref_frame_indices = self.manager.get_ref_frames_in_use(self.get_conditioner_var(), t, self.ref_frames)  # Shape: [N]


        # Map ref_frame_indices to contact_state indices
        # ref_frame_indices indexes into self.ref_frames, but contact_state uses self.contact_frames
        # Use the mapping to convert: ref_frame_idx -> contact_frame_idx
        contact_frame_indices = self.ref_to_contact_idx[ref_frame_indices]  # Shape: [N]

        # Check if the reference frames are in contact
        # Gather the contact state for the specific frames we're interested in
        ref_frames_in_contact = torch.gather(contact_state, 1, contact_frame_indices.unsqueeze(1)).squeeze(1)  # Shape: [N]

        # Now index only envs where we are in contact and domains changed
        changed_and_contact = changed & (ref_frames_in_contact > 0)     # Shape: [N]

        # Get the correct reference frame to pass
        # Use ref_frame_indices to select the appropriate pose for each environment
        self.ref_poses[changed_and_contact, :] = ref_poses[changed_and_contact, ref_frame_indices[changed_and_contact], :]

        # Compute the measured outputs
        self.compute_measured_output(self.ref_poses[:, :3], self.ref_poses[:, 3:])


    def compute_measured_output(self, ref_frame_pos_w, ref_frame_quat) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the measured state."""
        # TODO: Make more general. For now assuming CoM then bodies then joints always

        output_idx = 0

        # Get the relevant end effector positions (in global frame)
        if self.use_com:
            # Deal with CoM as a special case
            com_pos_w = self.robot.data.root_com_pos_w
            com_vel_w = self.robot.data.root_com_vel_w[:, :3]

            # Put into the reference frame
            com_pos_local = _transfer_to_local_frame(com_pos_w - ref_frame_pos_w, ref_frame_quat)
            # import pdb; pdb.set_trace()
            com_vel_local = _transfer_to_local_frame(com_vel_w, ref_frame_quat)

            # TODO: Add to the outputs in the correct order
            self.y_act[:, output_idx:output_idx+3] = com_pos_local
            self.dy_act[:, output_idx:output_idx+3] = com_vel_local
            output_idx += 3

        def _get_pos_ori_vel_relative(idx, base_frame_pos, base_frame_quat):
            """
            Compute the position, orientation, and velocity relative to the reference frame.

            Args:
                idx: tensor of shape [num_bodies]
                base_frame_pos: tensor of shape [3]
                base_frame_quat: tensor of shape [4]
            """
            # TODO: Does this handle a tensor of idx correctly?
            base_frame_ori = get_euler_from_quat(base_frame_quat)

            frame_pos = self.robot.data.body_pos_w[:, idx, :]
            frame_quat = self.robot.data.body_quat_w[:, idx, :]
            frame_ori = torch.zeros(base_frame_pos.shape[0], len(idx), 3, device=base_frame_pos.device)
            frame_pos_rel = torch.zeros(frame_pos.shape, device=frame_pos.device)
            frame_pos_rel_local = torch.zeros(frame_pos.shape, device=frame_pos.device)
            frame_ori_rel = frame_ori
            for i in range(len(idx)):
                frame_ori[:, i, :] = get_euler_from_quat(frame_quat[:, i, :])
                frame_pos_rel[:, i, :] = frame_pos[:, i, :] - base_frame_pos

                frame_pos_rel_local[:, i, :] = _transfer_to_local_frame(frame_pos_rel[:, i, :], base_frame_quat)

                frame_ori_rel[:, i, 2] = wrap_to_pi(frame_ori_rel[:, i, 2] - base_frame_ori[:, 2])

            frame_lin_vel_w = self.robot.data.body_lin_vel_w[:, idx, :]
            frame_ang_vel_w = self.robot.data.body_ang_vel_w[:, idx, :]

            frame_vel_local = torch.zeros(frame_lin_vel_w.shape, device=frame_lin_vel_w.device)
            frame_ang_vel_local = torch.zeros(frame_ang_vel_w.shape, device=frame_ang_vel_w.device)
            for i in range(len(idx)):
                frame_vel_local[:, i, :] = _transfer_to_local_frame(frame_lin_vel_w[:, i, :], base_frame_quat)
                frame_ang_vel_local[:, i, :] = _transfer_to_local_frame(frame_ang_vel_w[:, i, :], base_frame_quat)

            # TODO: Make sure the returned shape is [N, num_bodies, 3] for each
            return frame_pos_rel_local, frame_ori_rel, frame_vel_local, frame_ang_vel_local

        if self.body_idx is not None:
            # TODO: Does the root link need to be dealt with specially?

            body_pos_local, body_ori_local, body_vel_local, body_ang_vel_local = _get_pos_ori_vel_relative(
                self.body_idx, ref_frame_pos_w, ref_frame_quat,)

            # TODO: Make sure this is being added correctly
            self.y_act[:, output_idx:output_idx+(3*len(self.body_idx))] = body_pos_local.flatten(1)
            self.dy_act[:, output_idx:output_idx+(3*len(self.body_idx))] = body_vel_local.flatten(1)

            output_idx += (3*len(self.body_idx))

        # Get the relevant joint angles
        if self.joint_idx is not None:
            joint_pos = self.robot.data.joint_pos[:, self.joint_idx]
            joint_vel = self.robot.data.joint_vel[:, self.joint_idx]

            # TODO: Make sure this is being added correctly
            self.y_act[:, output_idx:output_idx+(joint_pos.shape[1])] = joint_pos
            self.dy_act[:, output_idx:output_idx+(joint_vel.shape[1])] = joint_vel

            output_idx += joint_vel

    def get_desired_outputs(self, t: torch.Tensor):
        """
        Get the desired output to track from the trajectory.

        TODO: Need to support adjustments to the trajectory, such as yawing and lateral motion for locomoting.
        """

        if self.manager_type == "library":
            conditioner = self.get_conditioner_var()
            y = self.manager.get_output(t, conditioner)
        else:
            y = self.manager.get_output(t)

        self.y_des = y[:, 0, :]
        self.dy_des = y[:, 1, :]

    @property
    def command(self):
        return self.y_des

    def _resample_command(self, env_ids):
        """Resample the command."""
        self._update_command()
        return

    def _update_command(self):
        """Update the command values."""
        # Time in each env
        t = self.env.sim.current_time * torch.ones(self.num_envs, device=self.device)

        # Get conditioning variables (velocity, etc...)
        # cond_vars = self.env.command_manager.get_command(self.conditioner_generator)[:, 0]  # TODO: Allow conditioners to be more than scalars

        # Update the measured outputs
        self.get_measured_outputs(t)

        # Get desired output
        self.get_desired_outputs(t)

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des, self.yaw_output_idxs)
        self.vdot = vdot
        self.v = vcur

    def _update_metrics(self):
        """
        Update the metrics.

        Metrics to update:
        - Position tracking
        - Velocity tracking
        - CLF values

        """
        # TODO: Double check
        # Call parent method for base metrics
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

        # Update metrics based on trajectory type
        if hasattr(self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]], 'axis_names'):
            # End-effector trajectories
            for axis_info in self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]].axis_names:
                error_key = axis_info['name']
                index = axis_info['index']
                self.metrics[error_key] = torch.abs(
                    self.y_out[:, index] -
                    self.y_act[:, index]
                )
        else:
            # Joint trajectories
            for i, joint_name in enumerate(self.robot.joint_names):
                error_key = f"error_{joint_name}"
                self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

    def _parse_outputs(self, output_names: list[str]) -> tuple[list[int], list[int], bool]:
        """
        Parse the output names to indices to be used for getting data from the robot in sim.

        Args:
            output_names: List of output names in the format "frame:axis" or "joint:joint_name"

        Returns:
            joint_idx: List of joint indices (or None if no joints)
            body_idx: List of body indices (or None if no bodies)
            use_com: True if CoM is used, False otherwise
        """
        joint_indices = []
        body_indices = []
        use_com = False

        # Track which frames we've already added to avoid duplicates
        added_bodies = set()

        for output_name in output_names:
            if output_name.startswith('joint:'):
                # Joint output: "joint:joint_name"
                joint_name = output_name.split(':', 1)[1]

                # Get the index of this joint
                if joint_name in self.robot.joint_names:
                    joint_idx = self.robot.joint_names.index(joint_name)
                    if joint_idx not in joint_indices:
                        joint_indices.append(joint_idx)
                else:
                    raise ValueError(f"Joint '{joint_name}' not found in robot joint names.")

            elif output_name.startswith('com:'):
                # CoM output: "com:pos_x", "com:pos_y", etc.
                use_com = True

            else:
                # Frame output: "frame_name:axis"
                frame_name = output_name.split(':', 1)[0]

                # Skip if we've already added this body
                if frame_name in added_bodies:
                    continue

                # Get the index of this body
                if frame_name in self.robot.body_names:
                    body_idx = self.robot.body_names.index(frame_name)
                    body_indices.append(body_idx)
                    added_bodies.add(frame_name)
                else:
                    raise ValueError(f"Body frame '{frame_name}' not found in robot body names.")

        # Convert to None if empty
        joint_idx_result = joint_indices if len(joint_indices) > 0 else None
        body_idx_result = body_indices if len(body_indices) > 0 else None

        return joint_idx_result, body_idx_result, use_com

    def _parse_ref_frames(self, reference_frames: list[str]) -> tuple[list[int], list[str]]:
        """
        Parse the reference frame names to body indices.

        Args:
            reference_frames: List of body frame names (e.g., ["left_ankle_roll_link", "right_ankle_roll_link"])

        Returns:
            Tuple of (frame_indices, expanded_frame_names):
                - frame_indices: List of body indices corresponding to the reference frames
                - expanded_frame_names: List of expanded frame names (with left/right pairs)

        Note:
            If any frame starts with "right" or "left", the corresponding opposite side frame
            is also added automatically to ensure bilateral symmetry.
        """
        expanded_frames = []

        for frame_name in reference_frames:
            # Add the original frame
            if frame_name not in expanded_frames:
                expanded_frames.append(frame_name)

            # Check if the frame starts with "right" or "left" and add the opposite side
            if frame_name.startswith("right"):
                # Replace "right" with "left"
                opposite_frame = "left" + frame_name[5:]  # Remove "right" and add "left"
                if opposite_frame not in expanded_frames:
                    expanded_frames.append(opposite_frame)
            elif frame_name.startswith("left"):
                # Replace "left" with "right"
                opposite_frame = "right" + frame_name[4:]  # Remove "left" and add "right"
                if opposite_frame not in expanded_frames:
                    expanded_frames.append(opposite_frame)

        # Convert frame names to body indices
        frame_indices = []
        for frame_name in expanded_frames:
            if frame_name in self.robot.body_names:
                frame_idx = self.robot.body_names.index(frame_name)
                frame_indices.append(frame_idx)
            else:
                raise ValueError(f"Reference frame '{frame_name}' not found in robot body names.")

        return frame_indices, expanded_frames


def _transfer_to_local_frame(vec, root_quat):
    return quat_apply(yaw_quat(quat_inv(root_quat)), vec)

def get_euler_from_quat(quat):
    """
    Convert quaternion(s) to Euler angles.

    Args:
        quat: Quaternion tensor of shape [..., 4] (supports both single and batched inputs)

    Returns:
        Euler angles tensor of shape [..., 3] with [roll, pitch, yaw]
    """
    euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
    euler_x = wrap_to_pi(euler_x)
    euler_y = wrap_to_pi(euler_y)
    euler_z = wrap_to_pi(euler_z)
    return torch.stack([euler_x, euler_y, euler_z], dim=-1)