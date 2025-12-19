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
        self.contact_bodies = self._expand_wildcard_frames(cfg.contact_bodies)

        # Extract the index into the robot data bodies for the contact frames
        self.contact_frame_indices = torch.zeros(len(self.contact_bodies), dtype=torch.long, device=self.device)
        for i, frame_name in enumerate(self.contact_bodies):
            if frame_name in self.robot.body_names:
                self.contact_frame_indices[i] = self.robot.body_names.index(frame_name)
            else:
                raise ValueError(f"Contact frame '{frame_name}' not found in robot body names.")

        self.current_contact_poses = torch.zeros(self.num_envs, len(self.contact_bodies), 6, dtype=torch.float, device=self.device)
        self.current_contact_vels = torch.zeros(self.num_envs, len(self.contact_bodies), 6, dtype=torch.float, device=self.device)
        self.desired_contact_poses = torch.zeros(self.num_envs, len(self.contact_bodies), 6, dtype=torch.float, device=self.device)

        self.manager_type = cfg.manager_type
        self.conditioner_generator = cfg.conditioner_generator_name

        self.y_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)

        self.y_des = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_des = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)

        # Create trajectory/library manager
        if cfg.manager_type == "trajectory":
            self.manager = TrajectoryManager(cfg.path, cfg.hf_repo, env.device)
            self.trajectory_type = self.manager.traj_data.trajectory_type
        elif cfg.manager_type == "library":
            self.manager = LibraryManager(cfg.path, cfg.hf_repo, env.device)
            self.trajectory_type = self.manager.trajectory_type
        else:
            raise NotImplementedError(f"Manager Type {cfg.manager_type} is not implemented!")

        self.verify_contact_frames()

        # Hold a list of current domains for each env
        self.current_domain = -1 * torch.ones(self.num_envs, device=self.device)

        # Create a list of indices to be used
        self.joint_idx, self.body_idx, self.use_com, self.ordered_output_names, self.body_type = self._parse_outputs(self.manager.get_output_names)

        # Order the manager to match the order here
        self.manager.order_outputs(self.ordered_output_names)

        self.body_type = torch.tensor(self.body_type, dtype=torch.int, device=self.device)

        print(f"Order output names: {self.ordered_output_names}")

        if cfg.num_outputs != len(self.ordered_output_names):
            raise ValueError(f"Number of output names does not match number of outputs in the cfg! "
                             f"Len of names: {len(self.ordered_output_names)}, cfg len: {cfg.num_outputs} !")

        # For now assuming that all bodies have a yaw tracking
        self.yaw_output_idxs = []
        for body in range(len(self.body_idx)):
            start = 0
            if self.use_com:
                start += 2
            self.yaw_output_idxs = start + 5*body

        # Create a list of indices for the reference frames
        self.ref_frame_indices, self.ref_frames = self._parse_ref_frames(self.manager.get_reference_frames())

        # Verify all reference frames are in contact frames
        for ref_frame in self.ref_frames:
            if ref_frame not in self.contact_bodies:
                raise ValueError(f"Reference frame '{ref_frame}' is not in the contact frames list: {self.contact_bodies}")

        # Create a mapping from ref_frames to contact_frames indices
        # This allows us to map ref_frame_indices to contact_state indices
        self.ref_to_contact_idx = torch.zeros(len(self.ref_frames), dtype=torch.long, device=self.device)
        for i, ref_frame in enumerate(self.ref_frames):
            self.ref_to_contact_idx[i] = self.contact_bodies.index(ref_frame)

        # Current reference frame poses
        self.ref_poses = torch.zeros((self.num_envs, 7), device=self.device)    # [N, [position, quat]]

        # Create CLF
        self.clf = CLF(
            self.cfg.num_outputs, self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            ordered_output_names=self.ordered_output_names,
            Q_weights=self.cfg.Q_weights,
            R_weights=self.cfg.R_weights,
            device=self.device
        )

        self.phasing_var = 0

    def get_phasing_var(self, t: torch.Tensor) -> torch.Tensor:
        if self.manager_type == "trajectory":
            self.phasing_var = self.manager.get_phasing_var(t)
            return self.phasing_var
        elif self.manager_type == "library":
            conditioner = self.get_conditioner_var()
            self.phasing_var = self.manager.get_phasing_var(conditioner, t)
            return self.phasing_var
        else:
            raise ValueError(f"Cannot get phasing variable for manager of type {self.manager_type}.")

    def get_conditioner_var(self) -> torch.Tensor:
        """Get the conditioner variable."""
        cond_term = self.env.command_manager.get_term(self.conditioner_generator)

        # TODO: For now just hard code the conditioning variable as velocity
        condition_vars = cond_term.command[:, 0]

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
            return self.manager.get_contact_state(conditioner, t, self.contact_bodies)
        else:
            return self.manager.get_contact_state(t, self.contact_bodies)

    def get_symmetric_contacts(self, contacts):
        """
        Get the left-right symmetric reflection of the contacts.

        If a left contact is a 1 then make it 0 and make the right contact 1. And vice-versa.

        Args:
            contacts: Tensor of shape [N, num_contacts] with contact states

        Returns:
            Symmetric contacts tensor of shape [N, num_contacts]
        """
        # Create a copy to avoid modifying the original
        symmetric_contacts = contacts.clone()

        # Create a mapping of contact indices to their symmetric counterparts
        for i, frame_name in enumerate(self.contact_bodies):
            # Find the symmetric frame
            if "left" in frame_name:
                symmetric_frame = frame_name.replace("left", "right")
            elif "right" in frame_name:
                symmetric_frame = frame_name.replace("right", "left")
            else:
                # No symmetry for this frame (e.g., center frame)
                continue

            # Find the index of the symmetric frame
            if symmetric_frame in self.contact_bodies:
                j = self.contact_bodies.index(symmetric_frame)
                # Swap the contact states
                symmetric_contacts[:, i] = contacts[:, j]

        return symmetric_contacts


    def get_trajectory_type(self):
        """Gets the type of trajectory: periodic or episodic."""
        return self.trajectory_type

    def verify_contact_frames(self):
        traj_frames = []
        if self.manager_type == "trajectory":
            for domain in self.manager.traj_data.domain_data.values():
                traj_frames.append(domain.contact_bodies)
        elif self.manager_type == "library":
            for manager in self.manager.trajectory_managers:
                for domain in manager.traj_data.domain_data.values():
                    traj_frames.append(domain.contact_bodies)
        else:
            raise NotImplementedError(f"Manager Type {self.manager_type} is not implemented!")

        # Verify that every frame in traj_frames appears in self.contact_frames
        for frames in traj_frames:
            for frame in frames:
                if frame not in self.contact_bodies:
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
            poses is shape [N, num_contacts, 6]. Everything not in contact is masked to 0
        """

        poses = torch.zeros(self.num_envs, len(self.contact_bodies), 6, device=self.device)

        not_in_contact = contact_state == 0

        # Get the poses of all the possible contact bodies
        poses[:, :, :3] = self.robot.data.body_pos_w[:, self.contact_frame_indices, :]
        for i in range(len(self.contact_bodies)):
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

    def get_contact_vels(self, contact_state: torch.Tensor) -> torch.Tensor:
        """
        Get the velocity of each frame that is in contact.

        Args:
            contact_state: shape [N, num_contacts]

        Returns:
            vels is shape [N, num_contacts, 6]. Everything not in contact is masked to 0
        """
        vels = torch.zeros(self.num_envs, len(self.contact_bodies), 6, device=self.device)

        not_in_contact = contact_state == 0

        vels[:, :, :3] = self.robot.data.body_lin_vel_w[:, self.contact_frame_indices, :]
        vels[:, :, 3:] = self.robot.data.body_ang_vel_w[:, self.contact_frame_indices, :]

        # Now mask
        vels[not_in_contact, :] *= 0

        return vels

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
        if self.manager_type == "library":
            conditioner = self.get_conditioner_var()
            new_domains = self.manager.get_current_domains(conditioner, t)
        else:
            new_domains = self.manager.get_current_domains(t)

        # Check if the domains changed
        changed = new_domains != self.current_domain

        # Note that domains never change for full periodic single domain trajectories, but I think we still want to update the position.
        # The two options are (1) continually updating the position while in contact, (2) try to check based on the phasing variable
        # TODO: Come back to this to see if this is the best way to do this. For now using option (1)
        if self.manager.get_num_domains() == 1:
            changed[:] = True

        # Update the list of current domains
        self.current_domain = new_domains

        # Determine which reference frames/bodies are in contact
        contact_state = self.get_contact_state(t)  # Shape: [N, num_contact_frames]
        # print(f"contact_state: {contact_state}, contact_frames: {self.contact_bodies}")

        self.current_contact_poses = self.get_contact_poses(contact_state)
        self.current_contact_vels = self.get_contact_vels(contact_state)
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
        if torch.any(changed_and_contact):
            # Get environment indices where the condition is true
            env_indices = torch.where(changed_and_contact)[0]
            # Index ref_poses using advanced indexing: [env_idx, frame_idx, :]
            self.ref_poses[env_indices, :] = ref_poses[env_indices, ref_frame_indices[env_indices], :]

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

            self.y_act[:, output_idx:output_idx+3] = com_pos_local
            self.dy_act[:, output_idx:output_idx+3] = com_vel_local
            output_idx += 3

        def _get_pos_ori_vel_relative(ref_frame_pos, ref_frame_quat, frame_pos, frame_quad,
                                      frame_lin_vel_w, frame_ang_vel_w):
            """
            Compute the position, orientation, and velocity relative to the reference frame.

            Args:
                idx: tensor of shape [num_bodies]
                ref_frame_pos: tensor of shape [3]
                ref_frame_quat: tensor of shape [4]

            Returns:
                frame_pos_rel_local: tensor of shape [N, num_bodies, 3]
                frame_ori_rel: tensor of shape [N, num_bodies, 3]
                frame_vel_local: tensor of shape [N, num_bodies, 3]
                frame_ang_vel_local: tensor of shape [N, num_bodies, 3]
            """
            base_frame_ori = get_euler_from_quat(ref_frame_quat)

            num_idx = len(self.body_idx)

            frame_ori = torch.zeros(ref_frame_pos.shape[0], num_idx, 3, device=ref_frame_pos.device)
            frame_pos_rel = torch.zeros(frame_pos.shape, device=frame_pos.device)
            frame_pos_rel_local = torch.zeros(frame_pos.shape, device=frame_pos.device)
            frame_ori_rel = frame_ori
            for i in range(num_idx):
                frame_ori[:, i, :] = get_euler_from_quat(frame_quat[:, i, :])
                frame_pos_rel[:, i, :] = frame_pos[:, i, :] - ref_frame_pos

                frame_pos_rel_local[:, i, :] = _transfer_to_local_frame(frame_pos_rel[:, i, :], ref_frame_quat)

                frame_ori_rel[:, i, 2] = wrap_to_pi(frame_ori_rel[:, i, 2] - base_frame_ori[:, 2])

            frame_vel_local = torch.zeros(frame_lin_vel_w.shape, device=frame_lin_vel_w.device)
            frame_ang_vel_local = torch.zeros(frame_ang_vel_w.shape, device=frame_ang_vel_w.device)
            for i in range(num_idx):
                frame_vel_local[:, i, :] = _transfer_to_local_frame(frame_lin_vel_w[:, i, :], ref_frame_quat)
                frame_ang_vel_local[:, i, :] = _transfer_to_local_frame(frame_ang_vel_w[:, i, :], ref_frame_quat)

            return frame_pos_rel_local, frame_ori_rel, frame_vel_local, frame_ang_vel_local

        # Bodies
        if self.body_idx is not None:
            frame_pos = self.robot.data.body_pos_w[:, self.body_idx, :]
            frame_quat = self.robot.data.body_quat_w[:, self.body_idx, :]
            frame_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.body_idx, :]
            frame_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.body_idx, :]

            body_pos_local, body_ori_local, body_vel_local, body_ang_vel_local = _get_pos_ori_vel_relative(
                ref_frame_pos_w, ref_frame_quat, frame_pos, frame_quat, frame_lin_vel_w, frame_ang_vel_w)

            pos_bodies = (self.body_type == 1) | (self.body_type == 2)
            num_pos_bodies = pos_bodies.sum()
            ori_bodies = (self.body_type == 0) | (self.body_type == 2)
            num_ori_bodies = ori_bodies.sum()

            # Add linear
            self.y_act[:, output_idx:output_idx+(3*num_pos_bodies)] = (
                body_pos_local[:, pos_bodies, :].flatten(1))
            self.dy_act[:, output_idx:output_idx+(3*num_pos_bodies)] = (
                body_vel_local[:, pos_bodies, :].flatten(1))

            output_idx += (3*num_pos_bodies.item())

            # Add angles
            self.y_act[:, output_idx:output_idx+(3*num_ori_bodies)] = (
                body_ori_local[:, ori_bodies, :].flatten(1))
            self.dy_act[:, output_idx:output_idx+(3*num_ori_bodies)] = (
                body_ang_vel_local[:, ori_bodies, :].flatten(1))

            output_idx += (3*num_ori_bodies.item())

        # Get the relevant joint angles
        if self.joint_idx is not None:
            joint_pos = self.robot.data.joint_pos[:, self.joint_idx]
            joint_vel = self.robot.data.joint_vel[:, self.joint_idx]

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

        if self.manager.get_trajectory_type() == TrajectoryType.EPISODIC:
            phi = self.get_phasing_var(t)

            self.dy_des[phi == 1] *= 0

    def get_symmetric_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Computes the symmetric version of the trajectory.

        Goes through each of the output names and swaps the values if there is a corresponding right/left symmetric output.

        Need to negate the values for any pos_y or ori_x, ori_z.

        Args:
            traj: Trajectory tensor of shape [N, num_outputs]

        Returns:
            Symmetric trajectory tensor of shape [N, num_outputs]

        TODO: Does this work with any reference frame?
        """
        # Create a copy to avoid modifying the original
        symmetric_traj = traj.clone()

        # Create a mapping from each output to its symmetric counterpart
        for i, output_name in enumerate(self.ordered_output_names):
            # Find the symmetric output name
            if "left" in output_name:
                symmetric_name = output_name.replace("left", "right")
            elif "right" in output_name:
                symmetric_name = output_name.replace("right", "left")
            else:
                # No left/right symmetry (e.g., COM, waist), but may need sign flip
                # Check if this axis needs negation
                if any(axis in output_name for axis in ["pos_y", "ori_x", "ori_z"]):
                    symmetric_traj[:, i] = -traj[:, i]
                continue

            # Find the index of the symmetric output
            if symmetric_name in self.ordered_output_names:
                j = self.ordered_output_names.index(symmetric_name)

                # Swap the values
                symmetric_traj[:, i] = traj[:, j]

                # Apply sign flip for specific axes
                if any(axis in output_name for axis in ["pos_y", "ori_x", "ori_z"]):
                    symmetric_traj[:, i] = -symmetric_traj[:, i]

        return symmetric_traj

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
        t = self.env.episode_length_buf * self.env.step_dt

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
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

        for i, output in enumerate(self.ordered_output_names):
            self.metrics[output] = torch.abs(self.y_des[:, i] - self.y_act[:, i])

        for i, output in enumerate(self.ordered_output_names):
            self.metrics[output + "_vel"] = torch.abs(self.dy_des[:, i] - self.dy_act[:, i])

    def _parse_outputs(self, output_names: list[str]) -> tuple[list[int], list[int], bool, list[str], list[int]]:
        """
        Parse the output names to indices to be used for getting data from the robot in sim.

        Args:
            output_names: List of output names in the format "frame:axis" or "joint:joint_name"

        Returns:
            joint_idx: List of joint indices (or None if no joints)
            body_idx: List of body indices (or None if no bodies)
            use_com: True if CoM is used, False otherwise
            ordered_output_names: List of output names in the order they appear in compute_measured_output (COM, bodies, joints)
            body_type_list: List indicating type of each body (0=ori only, 1=pos only, 2=both)
        """
        joint_indices = []
        joint_names_list = []
        body_indices = []
        body_names_list = []
        use_com = False
        com_axes = []

        body_type = {}
        body_type_list = []

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
                        joint_names_list.append(joint_name)
                else:
                    raise ValueError(f"Joint '{joint_name}' not found in robot joint names.")

            elif output_name.startswith('com:'):
                # CoM output: "com:pos_x", "com:pos_y", etc.
                use_com = True
                axis = output_name.split(':', 1)[1]  # e.g., "pos_x", "pos_y", "pos_z"
                if axis not in com_axes:
                    com_axes.append(axis)

            else:
                # Frame output: "frame_name:axis"
                frame_name = output_name.split(':', 1)[0]

                if frame_name not in body_type:
                    body_type[frame_name] = [output_name.split(':', 1)[1]]
                else:
                    body_type[frame_name].append(output_name.split(':', 1)[1])

                # Skip if we've already added this body
                if frame_name in added_bodies:
                    continue

                # Try to get the index of this body
                if frame_name in self.robot.body_names:
                    body_idx = self.robot.body_names.index(frame_name)
                    body_indices.append(body_idx)
                    body_names_list.append(frame_name)
                    added_bodies.add(frame_name)
                else:
                    raise ValueError(f"Body frame '{frame_name}' not found in robot body names.")

        # Build ordered output names in the order: COM, bodies, joints
        ordered_output_names = []

        # Add COM outputs first
        if use_com:
            for axis in com_axes:
                ordered_output_names.append(f"com:{axis}")

        # Add body outputs - start with all positions
        for body_name in body_names_list:
            for btype in body_type[body_name]:
                if "pos" in btype:
                    ordered_output_names.append(f"{body_name}:{btype}")

        # Add body outputs - then do orientations
        for body_name in body_names_list:
            for btype in body_type[body_name]:
                if "ori" in btype:
                    ordered_output_names.append(f"{body_name}:{btype}")

        # Build body_type_list for bodies
        for body_name in body_names_list:
            ori = False
            pos = False
            for btype in body_type[body_name]:
                if "ori" in btype:
                    ori = True
                if "pos" in btype:
                    pos = True
            if pos and ori:
                body_type_list.append(2)
            if pos and not ori:
                body_type_list.append(1)
            if ori and not pos:
                body_type_list.append(0)

        # Add joint outputs
        for joint_name in joint_names_list:
            ordered_output_names.append(f"joint:{joint_name}")

        # Convert to None if empty
        joint_idx_result = joint_indices if len(joint_indices) > 0 else None
        body_idx_result = body_indices if len(body_indices) > 0 else None

        return joint_idx_result, body_idx_result, use_com, ordered_output_names, body_type_list

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