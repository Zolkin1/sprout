# Copyright 2026 Zachary Olkin. All rights reserved.

"""Abstract base class for trajectory command terms.

Holds the manager-agnostic, phase-free shared logic.  Concrete commands
extend either this class directly (multi-skill, no phase state) or the
:class:`PhasedTrajectoryCommand` subclass (single-skill, phase variable
exposed as observation with hold-phi logic).
"""

from __future__ import annotations

import re
from abc import abstractmethod

import torch
import warp as wp
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    wrap_to_pi,
    yaw_quat,
)

from .clf import CLF
from .manager_base import ManagerBase
from .sagittal_reflector import NamedReflector, SagittalReflectionConfig
from .terrain_aware_ref import apply_terrain_aware_ref
from .trajectory_manager import TrajectoryType
from .world_lift import lift_base_world_state


class BaseTrajectoryCommand(CommandTerm):
    """Trajectory command term base class.

    Handles measured/desired output computation, CLF tracking, ref-frame
    bookkeeping, and metrics logging.  Trajectory advancement is time-based
    only; no phasing variable is stored on this class.  Subclasses that
    want phase-as-observation or hold-phi logic should extend
    :class:`PhasedTrajectoryCommand` instead.

    Subclasses must implement ``_create_manager`` and
    ``_verify_contact_frames``.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create and return the trajectory manager.

        Args:
            cfg: The command term configuration.
            env: The IsaacLab environment.

        Returns:
            A concrete :class:`ManagerBase` subclass instance.
        """

    @abstractmethod
    def _verify_contact_frames(self) -> None:
        """Verify that every contact frame in the trajectories is present in ``self.contact_bodies``."""

    def _post_init(self) -> None:
        """Hook called at the very end of ``__init__``.

        Override in subclasses that need additional setup after the base
        initialisation is complete (e.g. building reference-frame maps).
        """

    def _record_library_fingerprint(self, cfg) -> None:
        """Persist (and verify) the loaded trajectory library's content hash.

        The manager exposes a portable content fingerprint of the library it
        actually loaded.  This method:

          * warns loudly when ``cfg.library_fingerprint`` was *already* set to a
            different hash — i.e. this policy is being replayed (play / resume)
            against a library that differs from the one recorded in the saved
            ``env.yaml`` it trained on (see ``play.py``, which injects the
            trained-against value before env construction); and
          * writes the freshly computed fingerprint + per-skill manifest back
            onto ``cfg`` so a fresh training run serialises them into
            ``params/env.yaml`` (``train.py`` dumps the cfg after env construction).

        No-op for managers that do not expose ``library_fingerprint`` (e.g. the
        single-file :class:`~.trajectory_manager.TrajectoryManager`).

        Args:
            cfg: This command term's configuration (the same instance the env
                cfg holds, so mutations are reflected in the dumped yaml).
        """
        fingerprint = getattr(self.manager, "library_fingerprint", None)
        if fingerprint is None:
            return

        local_hash = fingerprint["hash"]
        trained_hash = getattr(cfg, "library_fingerprint", None)
        if trained_hash is not None and trained_hash != local_hash:
            trained_manifest = getattr(cfg, "library_manifest", None)
            bar = "=" * 78
            print(
                f"\n{bar}\n"
                "[WARN] TRAJECTORY LIBRARY MISMATCH\n"
                f"  cfg.path                 : {cfg.path}\n"
                f"  trained-against hash     : {trained_hash}\n"
                f"  locally-loaded hash      : {local_hash}\n"
                f"  trained-against manifest : {trained_manifest}\n"
                f"  locally-loaded manifest  : {fingerprint['skill_counts']} "
                f"({fingerprint['num_trajectories']} trajectories)\n"
                "  This policy is being run against a DIFFERENT trajectory library\n"
                "  than the one it trained on -- the references may be wrong.\n"
                f"{bar}\n"
            )

        cfg.library_fingerprint = local_hash
        cfg.library_manifest = fingerprint["skill_counts"]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.env = env
        self.robot = env.scene[cfg.asset_name]

        # --- Contact bodies (expand wildcards) ----------------------------
        self.contact_bodies: list[str] = self._expand_wildcard_frames(cfg.contact_bodies)

        # --- Manager (subclass-specific) ----------------------------------
        self.manager: ManagerBase = self._create_manager(cfg, env)
        self.trajectory_type = self.manager.get_trajectory_type()

        # --- Trajectory-library provenance --------------------------------
        # Records the loaded library's portable fingerprint onto ``cfg`` (so it
        # lands in ``params/env.yaml``) and warns if it differs from a
        # trained-against value already present on ``cfg`` (play / resume).
        self._record_library_fingerprint(cfg)

        self._verify_contact_frames()

        # --- Domain tracking ----------------------------------------------
        # ``current_domain`` is the domain index from the previous step
        # (relative to the previous trajectory).  ``prev_ref_frame_idx``
        # is the index (into ``self.ref_frames``) of the body we last
        # anchored ``ref_poses`` to.  Together they drive the
        # "needs a ref_poses refresh" gate in get_measured_outputs:
        #
        #  * domain-level change (gate fire, snap) → anchor swap
        #  * ref-frame-identity change across trajectories at the same
        #    domain (e.g. cross-skill swap whose new domain 0 puts the
        #    opposite foot in stance) → anchor swap
        #
        # Using ref-frame identity instead of trajectory index avoids
        # spurious refreshes on intra-skill bucket swaps that keep the
        # stance foot the same — those would otherwise re-anchor the
        # marker to the foot's *current* world pose mid-stance, which
        # lands above ground if the foot has already started pushoff.
        # Initialised to -1 so the first step always triggers an update.
        self.current_domain = -1 * torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_ref_frame_idx = -1 * torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        # --- Output parsing -----------------------------------------------
        result = self._parse_outputs(self.manager.get_pos_output_names)
        (self.joint_idx, self.body_idx, self.use_com,
         self.ordered_pos_output_names, self.ordered_vel_output_names,
         self.body_type) = result

        # --- State tensors ------------------------------------------------
        self.y_act = torch.zeros((self.num_envs, len(self.ordered_pos_output_names)), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)
        self.y_des = torch.zeros((self.num_envs, len(self.ordered_pos_output_names)), device=self.device)
        self.dy_des = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)

        # --- Order manager outputs to match our ordering ------------------
        self.manager.order_outputs(self.ordered_pos_output_names, self.ordered_vel_output_names)
        self.body_type = torch.tensor(self.body_type, dtype=torch.int, device=self.device)

        # --- Precomputed reflectors for sagittal symmetry -----------------
        _rcfg = SagittalReflectionConfig()
        self._traj_pos_reflector = NamedReflector(_rcfg, self.ordered_pos_output_names, self.device)
        self._traj_vel_reflector = NamedReflector(_rcfg, self.ordered_vel_output_names, self.device)
        self._contact_reflector = NamedReflector(_rcfg, self.contact_bodies, self.device)

        # --- vel_to_pos_idx mapping (pos includes ori_w, vel excludes it) -
        self.vel_to_pos_idx = torch.zeros(len(self.ordered_vel_output_names), dtype=torch.long, device=self.device)
        for i, vel_name in enumerate(self.ordered_vel_output_names):
            if vel_name in self.ordered_pos_output_names:
                self.vel_to_pos_idx[i] = self.ordered_pos_output_names.index(vel_name)
            else:
                raise ValueError(f"Velocity output name '{vel_name}' not found in position output names.")

        # --- Reference frames ---------------------------------------------
        self.ref_frame_indices, self.ref_frames = self._parse_ref_frames(
            self.manager.get_reference_frames()
        )
        for ref_frame in self.ref_frames:
            if ref_frame not in self.contact_bodies:
                raise ValueError(
                    f"Reference frame '{ref_frame}' is not in the contact frames list: {self.contact_bodies}"
                )

        # Mapping from ref_frames to contact_bodies indices (needed by get_symmetric_contacts)
        self.ref_to_contact_idx = torch.zeros(len(self.ref_frames), dtype=torch.long, device=self.device)
        for i, ref_frame in enumerate(self.ref_frames):
            self.ref_to_contact_idx[i] = self.contact_bodies.index(ref_frame)

        # Current reference frame poses — [N, 7] = [pos_x, pos_y, pos_z, qx, qy, qz, qw]
        self.ref_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self.ref_poses[:, 6] = 1.0  # identity quaternion

        # Index (into self.ref_frames) of the ref frame currently in contact
        # per env.  Updated each step inside get_measured_outputs.  Exposed
        # for logging / debug visualisation.
        self.cur_ref_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- CLF ----------------------------------------------------------
        self.clf = CLF(
            sim_dt=self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            ordered_vel_output_names=self.ordered_vel_output_names,
            ordered_pos_output_names=self.ordered_pos_output_names,
            Q_weights=self.cfg.Q_weights,
            R_weights=self.cfg.R_weights,
            device=self.device,
        )

        # --- Subclass hook ------------------------------------------------
        self._post_init()

    # ------------------------------------------------------------------
    # Contact state & symmetry helpers
    # ------------------------------------------------------------------

    def get_contact_state(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the desired contact state at time *t*.

        Args:
            phase: Shape ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional subset of environments.

        Returns:
            Binary tensor of shape ``[N, num_contacts]``.
        """
        return self.manager.get_contact_state(phase, self.contact_bodies, env_ids)

    def get_symmetric_contacts(self, contacts: torch.Tensor) -> torch.Tensor:
        """Return the left-right symmetric reflection of *contacts*.

        Args:
            contacts: ``[N, num_contacts]`` or
                ``[N, history_length * num_contacts]`` contact states.

        Returns:
            Symmetric contacts of the same shape.
        """
        return self._contact_reflector.reflect_with_history(contacts)

    # ------------------------------------------------------------------
    # Trajectory type
    # ------------------------------------------------------------------

    def get_trajectory_type(self):
        """Return the trajectory type (periodic / episodic)."""
        return self.trajectory_type

    # ------------------------------------------------------------------
    # Measured outputs
    # ------------------------------------------------------------------

    def get_measured_outputs(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Compute measured outputs from the robot state.

        Args:
            phase: phase in [0, 1] tensor of shape ``[N]``.
            env_ids: Optional environment indices.
        """
        # Get poses for all reference-frame bodies: [N, num_ref_frames, 7]
        ref_poses = torch.zeros(self.num_envs, len(self.ref_frame_indices), 7, device=self.device)
        ref_poses[:, :, :3] = wp.to_torch(self.robot.data.body_pos_w)[:, self.ref_frame_indices]
        ref_poses[:, :, 3:] = wp.to_torch(self.robot.data.body_quat_w)[:, self.ref_frame_indices]

        # Look up the new domain and active ref-frame identity *before*
        # computing the refresh gate — the gate triggers on a change in
        # either the domain index or the active ref frame (see
        # ``prev_ref_frame_idx`` docstring in ``__init__``).
        # ``prev_ref_frame_idx`` is the LAST anchored ref frame and is
        # only updated when the anchor itself is refreshed below — that
        # way a deferred refresh (gate fired but the new frame isn't yet
        # in-contact per the trajectory table) keeps the trigger live
        # across steps.
        new_domains = self.manager.get_current_domains(phase, env_ids)
        if env_ids is None:
            ref_frame_indices = self.manager.get_ref_frames_in_use(phase, self.ref_frames)
            changed = (
                (new_domains != self.current_domain)
                | (ref_frame_indices != self.prev_ref_frame_idx)
            )
            self.current_domain = new_domains
            self.cur_ref_frame_idx = ref_frame_indices
        else:
            ref_frame_indices = self.manager.get_ref_frames_in_use(phase, self.ref_frames, env_ids)
            changed = (
                (new_domains != self.current_domain[env_ids])
                | (ref_frame_indices != self.prev_ref_frame_idx[env_ids])
            )
            self.current_domain[env_ids] = new_domains
            self.cur_ref_frame_idx[env_ids] = ref_frame_indices

        contact_state = self.get_contact_state(phase, env_ids)
        contact_frame_indices = self.ref_to_contact_idx[ref_frame_indices]
        ref_in_contact = torch.gather(
            contact_state, 1, contact_frame_indices.unsqueeze(1)
        ).squeeze(1)
        changed_and_contact = changed & (ref_in_contact > 0)

        if env_ids is None:
            if torch.any(changed_and_contact):
                env_indices = torch.where(changed_and_contact)[0]
                default_new = ref_poses[env_indices, ref_frame_indices[env_indices], :]
                self.ref_poses[env_indices, :] = apply_terrain_aware_ref(
                    cmd=self,
                    env_indices=env_indices,
                    default_new_ref=default_new,
                    new_domain=new_domains[env_indices],
                )
                self.prev_ref_frame_idx[env_indices] = ref_frame_indices[env_indices]
        else:
            if torch.any(changed_and_contact):
                subset_indices = torch.where(changed_and_contact)[0]
                global_env_indices = env_ids[subset_indices]
                default_new = ref_poses[
                    global_env_indices, ref_frame_indices[subset_indices], :
                ]
                self.ref_poses[global_env_indices, :] = apply_terrain_aware_ref(
                    cmd=self,
                    env_indices=global_env_indices,
                    default_new_ref=default_new,
                    new_domain=new_domains[subset_indices],
                )
                self.prev_ref_frame_idx[global_env_indices] = ref_frame_indices[subset_indices]

        # Compute measured outputs using current ref_poses
        self.compute_measured_output(self.ref_poses[:, :3], self.ref_poses[:, 3:])

    def compute_measured_output(self, ref_frame_pos_w: torch.Tensor, ref_frame_quat: torch.Tensor) -> None:
        """Extract measured positions and velocities from the robot.

        Args:
            ref_frame_pos_w: ``[N, 3]`` reference frame positions.
            ref_frame_quat: ``[N, 4]`` reference frame quaternions.
        """
        pos_output_idx = 0
        vel_output_idx = 0

        # --- CoM ---
        if self.use_com:
            com_pos_w = wp.to_torch(self.robot.data.root_com_pos_w)
            com_vel_w = wp.to_torch(self.robot.data.root_com_vel_w)[:, :3]

            com_pos_local = _align_yaw(com_pos_w - ref_frame_pos_w, ref_frame_quat)
            com_vel_local = _align_yaw(com_vel_w, ref_frame_quat)

            self.y_act[:, pos_output_idx:pos_output_idx + 3] = com_pos_local
            self.dy_act[:, vel_output_idx:vel_output_idx + 3] = com_vel_local
            pos_output_idx += 3
            vel_output_idx += 3

        def _get_pos_ori_vel_relative(ref_pos_w, ref_quat_w, frame_pos_w, frame_quat_w,
                                      frame_lin_vel_w, frame_ang_vel_w):
            """Compute position, orientation, and velocity relative to the reference frame.

            All outputs are in the yaw-aligned frame centred at the reference.

            Args:
                ref_pos_w: ``[N, 3]``
                ref_quat_w: ``[N, 4]``
                frame_pos_w: ``[N, B, 3]``
                frame_quat_w: ``[N, B, 4]``
                frame_lin_vel_w: ``[N, B, 3]``
                frame_ang_vel_w: ``[N, B, 3]``

            Returns:
                Tuple of (pos, ori, lin_vel, ang_vel), each ``[N, B, ...]``.
            """
            pos_rel = frame_pos_w - ref_pos_w.unsqueeze(1)
            pos_yaw = _align_yaw_batched(pos_rel, ref_quat_w)
            ori_yaw = _align_quat_to_yaw_batched(frame_quat_w, ref_quat_w)
            vel_yaw = _align_yaw_batched(frame_lin_vel_w, ref_quat_w)
            ang_vel_yaw = _align_yaw_batched(frame_ang_vel_w, ref_quat_w)
            return pos_yaw, ori_yaw, vel_yaw, ang_vel_yaw

        # --- Bodies ---
        if self.body_idx is not None:
            frame_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self.body_idx, :]
            frame_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self.body_idx, :]

            body_link_vel = wp.to_torch(self.robot.data.body_link_vel_w)
            frame_lin_vel_w = body_link_vel[:, self.body_idx, :3]
            frame_ang_vel_w = body_link_vel[:, self.body_idx, 3:]

            body_pos_local, body_ori_local, body_vel_local, body_ang_vel_local = _get_pos_ori_vel_relative(
                ref_frame_pos_w, ref_frame_quat, frame_pos, frame_quat, frame_lin_vel_w, frame_ang_vel_w
            )

            pos_bodies = (self.body_type == 1) | (self.body_type == 2)
            num_pos_bodies = pos_bodies.sum()
            ori_bodies = (self.body_type == 0) | (self.body_type == 2)
            num_ori_bodies = ori_bodies.sum()

            # Linear
            self.y_act[:, pos_output_idx:pos_output_idx + (3 * num_pos_bodies)] = (
                body_pos_local[:, pos_bodies, :].flatten(1))
            self.dy_act[:, vel_output_idx:vel_output_idx + (3 * num_pos_bodies)] = (
                body_vel_local[:, pos_bodies, :].flatten(1))
            pos_output_idx += (3 * num_pos_bodies.item())
            vel_output_idx += (3 * num_pos_bodies.item())

            # Orientation
            self.y_act[:, pos_output_idx:pos_output_idx + (4 * num_ori_bodies)] = (
                body_ori_local[:, ori_bodies, :].flatten(1))
            self.dy_act[:, vel_output_idx:vel_output_idx + (3 * num_ori_bodies)] = (
                body_ang_vel_local[:, ori_bodies, :].flatten(1))
            pos_output_idx += (4 * num_ori_bodies.item())
            vel_output_idx += (3 * num_ori_bodies.item())

        # --- Joints ---
        if self.joint_idx is not None:
            joint_pos = wp.to_torch(self.robot.data.joint_pos)[:, self.joint_idx]
            joint_vel = wp.to_torch(self.robot.data.joint_vel)[:, self.joint_idx]

            self.y_act[:, pos_output_idx:pos_output_idx + joint_pos.shape[1]] = joint_pos
            self.dy_act[:, vel_output_idx:vel_output_idx + joint_vel.shape[1]] = joint_vel

    # ------------------------------------------------------------------
    # Desired outputs
    # ------------------------------------------------------------------

    def _transform_desired_outputs(
        self,
        phase: torch.Tensor,
        y_pos: torch.Tensor,
        y_vel: torch.Tensor,
        env_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional hook to transform desired outputs before storage.

        Default is identity.  Subclasses override to apply a user heuristic
        or other post-processing.
        """
        return y_pos, y_vel

    def get_desired_outputs(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Get the desired output from the trajectory manager.

        Args:
            phase: phase in [0, 1] tensor of shape ``[N]``.
            env_ids: Optional environment indices.
        """
        y_pos, y_vel = self.manager.get_output(phase, env_ids)
        y_pos, y_vel = self._transform_desired_outputs(phase, y_pos, y_vel, env_ids)

        if env_ids is None:
            self.y_des = y_pos
            self.dy_des = y_vel
        else:
            self.y_des[env_ids] = y_pos
            self.dy_des[env_ids] = y_vel

        # Zero velocities at the end of episodic trajectories
        if self.manager.get_trajectory_type() == TrajectoryType.EPISODIC:
            if env_ids is None:
                self.dy_des[phase == 1] *= 0
            else:
                episodic_mask = phase == 1
                self.dy_des[env_ids[episodic_mask]] *= 0

    # ------------------------------------------------------------------
    # World-frame reference reconstruction
    # ------------------------------------------------------------------

    def get_reference_world_state(self, base_frame_name: str) -> torch.Tensor:
        """Reconstruct the current desired reference as a world-frame state vector.

        The returned layout matches the raw robot ``state`` recorded by
        ``scripts/rsl_rl/record_data.py``::

            [root_pos_w(3), root_quat_w(4, xyzw), root_lin_vel_w(3),
             root_ang_vel_w(3), joint_pos(nj), joint_vel(nj)]

        The base pose/velocity stored in ``y_des``/``dy_des`` are expressed
        yaw-aligned relative to the active contact frame ``self.ref_poses``
        (see :meth:`compute_measured_output`); this inverts that alignment to
        lift them back to the world frame.  Joint targets are reordered to
        ``self.robot.joint_names`` so the joint axis matches the recorded
        ``state``.  Mirrors the extraction in
        :func:`mdp.events.resets.reset_on_reference`.

        Args:
            base_frame_name: Trajectory output frame used as the robot root;
                must expose ``pos_x/y/z`` and ``ori_x/y/z/w`` outputs (e.g.
                ``"pelvis_link"``).

        Returns:
            World-frame reference state, shape ``[num_envs, 3 + 4 + 3 + 3 + 2*nj]``.
        """
        idx = self._reference_world_indices(base_frame_name)

        # The base pose/vel in ``y_des``/``dy_des`` are yaw-aligned relative to
        # the active contact frame ``self.ref_poses``; ``lift_base_world_state``
        # inverts that alignment to lift them (and pass joints through) into the
        # world frame.  The lift is shared with the offline clip generator so
        # generation and runtime use an identical transform.
        return lift_base_world_state(self.y_des, self.dy_des, self.ref_poses, idx)

    def _reference_world_indices(self, base_frame_name: str) -> dict[str, torch.Tensor]:
        """Resolve (and cache) the output-vector indices used by :meth:`get_reference_world_state`.

        Args:
            base_frame_name: Trajectory output frame used as the robot root.

        Returns:
            Dict of long-tensor index sets keyed by ``pos``, ``ori``, ``lin_vel``,
            ``ang_vel``, ``joint_pos``, ``joint_vel``.
        """
        cache = getattr(self, "_ref_world_idx_cache", None)
        if cache is None:
            cache = {}
            self._ref_world_idx_cache = cache
        if base_frame_name in cache:
            return cache[base_frame_name]

        pos = _find_output_indices(self.ordered_pos_output_names, base_frame_name, "pos_")
        ori = _find_output_indices(self.ordered_pos_output_names, base_frame_name, "ori_")
        if len(pos) != 3 or len(ori) != 4:
            raise ValueError(
                f"Base frame '{base_frame_name}' must expose pos_x/y/z and ori_x/y/z/w in the "
                f"trajectory outputs (found {len(pos)} pos, {len(ori)} ori)."
            )
        lin_vel = _find_output_indices(self.ordered_vel_output_names, base_frame_name, "pos_")
        ang_vel = _find_output_indices(self.ordered_vel_output_names, base_frame_name, "ori_")

        joint_pos_idx: list[int] = []
        joint_vel_idx: list[int] = []
        for name in self.robot.joint_names:
            out = f"joint:{name}"
            if out not in self.ordered_pos_output_names:
                raise ValueError(f"Joint output '{out}' missing from trajectory outputs.")
            joint_pos_idx.append(self.ordered_pos_output_names.index(out))
            joint_vel_idx.append(self.ordered_vel_output_names.index(out))

        dev = self.device
        entry = {
            "pos": torch.as_tensor(pos, dtype=torch.long, device=dev),
            "ori": torch.as_tensor(ori, dtype=torch.long, device=dev),
            "lin_vel": torch.as_tensor(lin_vel, dtype=torch.long, device=dev),
            "ang_vel": torch.as_tensor(ang_vel, dtype=torch.long, device=dev),
            "joint_pos": torch.as_tensor(joint_pos_idx, dtype=torch.long, device=dev),
            "joint_vel": torch.as_tensor(joint_vel_idx, dtype=torch.long, device=dev),
        }
        cache[base_frame_name] = entry
        return entry

    # ------------------------------------------------------------------
    # Symmetry helpers
    # ------------------------------------------------------------------

    def get_symmetric_traj(self, traj: torch.Tensor, traj_type: str) -> torch.Tensor:
        """Return the left-right symmetric reflection of a trajectory.

        Swaps left/right outputs and negates pos_y, ori_x, ori_z, roll, yaw.

        Args:
            traj: ``[N, num_outputs]`` or ``[N, history_length * num_outputs]``
                trajectory tensor.  The history-stacked layout shows up when
                the corresponding obs term has ``history_length > 1``.
            traj_type: ``"vel"`` or ``"pos"`` to select the output name list.

        Returns:
            Symmetric trajectory of the same shape.
        """
        reflector = self._traj_vel_reflector if traj_type == "vel" else self._traj_pos_reflector
        return reflector.reflect_with_history(traj)

    def get_symmetric_joint_ref(self, joints: torch.Tensor) -> torch.Tensor:
        """Return the left-right symmetric reflection of reference joint values.

        Mirrors the ``joint:*`` channels of ``y_des`` / ``dy_des`` (the
        observations :func:`mdp.joint_pos_ref` / :func:`mdp.joint_vel_ref`
        expose).  Those channels follow the *clip's* joint output order rather
        than ``robot.joint_names``, so the reflection is built from the
        command's own joint-output names instead of the robot-joint reflector.
        Position and velocity share the same joint names, so one reflector
        serves both.

        Args:
            joints: ``[N, num_ref_joints]`` or
                ``[N, history_length * num_ref_joints]`` reference joint tensor.

        Returns:
            Symmetric reference joints of the same shape.
        """
        reflector = getattr(self, "_joint_ref_reflector", None)
        if reflector is None:
            joint_names = [n for n in self.ordered_pos_output_names if n.startswith("joint:")]
            reflector = NamedReflector(SagittalReflectionConfig(), joint_names, self.device)
            self._joint_ref_reflector = reflector
        return reflector.reflect_with_history(joints)

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self):
        """Return the current desired position output."""
        return self.y_des

    def _resample_command(self, env_ids):
        """Resample the command (delegates to ``_update_command``)."""
        self._update_command()

    def _pre_update_phase(self) -> None:
        """Hook called at the top of ``_update_command``.

        Responsible for advancing ``manager.phase``, invalidating the
        trajectory-assignment cache, and running any contact-gate logic
        that mutates phase.  The default just invalidates the cache so
        downstream manager reads see a fresh trajectory assignment.
        Subclasses that own phase advance / gating should re-invalidate
        after their mutations.
        """
        self.manager.invalidate_cache()

    def _update_command(self):
        """Main per-step update: measured outputs, desired outputs, CLF."""
        self._pre_update_phase()

        phi = self.manager.phase

        self.get_measured_outputs(phi)
        self.get_desired_outputs(phi)

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)

        self.vdot = vdot
        self.v = vcur

    def _update_metrics(self):
        """Log tracking errors and CLF values."""
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

        for i, output in enumerate(self.ordered_pos_output_names):
            self.metrics[output] = torch.abs(self.y_des[:, i] - self.y_act[:, i])

        for i, output in enumerate(self.ordered_vel_output_names):
            self.metrics[output + "_vel"] = torch.abs(self.dy_des[:, i] - self.dy_act[:, i])

    # ------------------------------------------------------------------
    # Init-time parsing helpers
    # ------------------------------------------------------------------

    def _expand_wildcard_frames(self, frame_patterns: list[str]) -> list[str]:
        """Expand wildcard patterns (e.g. ``".*_ankle_roll_link"``) to explicit body names.

        Args:
            frame_patterns: Frame name patterns, possibly containing regex wildcards.

        Returns:
            List of explicit body names.
        """
        expanded_frames: list[str] = []
        body_names = self.robot.body_names

        for pattern in frame_patterns:
            if '*' in pattern or '.*' in pattern:
                regex_pattern = pattern.replace('*', '.*') if '.*' not in pattern else pattern
                regex_pattern = f'^{regex_pattern}$'
                matched = False
                for body_name in body_names:
                    if re.match(regex_pattern, body_name):
                        expanded_frames.append(body_name)
                        matched = True
                if not matched:
                    raise ValueError(f"Wildcard pattern '{pattern}' did not match any body names in the robot.")
            else:
                expanded_frames.append(pattern)

        return expanded_frames

    def _parse_outputs(
        self, pos_output_names: list[str]
    ) -> tuple[list[int] | None, list[int] | None, bool, list[str], list[str], list[int]]:
        """Parse output names into robot-data indices and ordered name lists.

        Args:
            pos_output_names: Position output names (the superset including ``ori_w``).

        Returns:
            ``(joint_idx, body_idx, use_com, ordered_pos_names, ordered_vel_names, body_type_list)``
        """
        output_names = pos_output_names
        joint_indices: list[int] = []
        joint_names_list: list[str] = []
        body_indices: list[int] = []
        body_names_list: list[str] = []
        use_com = False
        com_axes: list[str] = []
        body_type: dict[str, list[str]] = {}
        body_type_list: list[int] = []
        added_bodies: set[str] = set()

        for output_name in output_names:
            if output_name.startswith('joint:'):
                joint_name = output_name.split(':', 1)[1]
                if joint_name in self.robot.joint_names:
                    joint_idx = self.robot.joint_names.index(joint_name)
                    if joint_idx not in joint_indices:
                        joint_indices.append(joint_idx)
                        joint_names_list.append(joint_name)
                else:
                    raise ValueError(f"Joint '{joint_name}' not found in robot joint names.")
            elif output_name.startswith('com:'):
                use_com = True
                axis = output_name.split(':', 1)[1]
                if axis not in com_axes:
                    com_axes.append(axis)
            else:
                frame_name = output_name.split(':', 1)[0]
                if frame_name not in body_type:
                    body_type[frame_name] = [output_name.split(':', 1)[1]]
                else:
                    body_type[frame_name].append(output_name.split(':', 1)[1])

                if frame_name in added_bodies:
                    continue
                if frame_name in self.robot.body_names:
                    body_idx = self.robot.body_names.index(frame_name)
                    body_indices.append(body_idx)
                    body_names_list.append(frame_name)
                    added_bodies.add(frame_name)
                else:
                    raise ValueError(f"Body frame '{frame_name}' not found in robot body names.")

        # Build ordered output names: COM → body positions → body orientations → joints
        ordered_pos_output_names: list[str] = []
        ordered_vel_output_names: list[str] = []

        if use_com:
            for axis in com_axes:
                ordered_pos_output_names.append(f"com:{axis}")
                ordered_vel_output_names.append(f"com:{axis}")

        for body_name in body_names_list:
            for btype in body_type[body_name]:
                if "pos" in btype:
                    ordered_pos_output_names.append(f"{body_name}:{btype}")
                    ordered_vel_output_names.append(f"{body_name}:{btype}")

        ORI_ORDER = ["ori_x", "ori_y", "ori_z", "ori_w"]
        for body_name in body_names_list:
            ori_axes = [b for b in body_type[body_name] if "ori" in b]
            ori_axes.sort(key=lambda a: ORI_ORDER.index(a) if a in ORI_ORDER else 0)
            for btype in ori_axes:
                ordered_pos_output_names.append(f"{body_name}:{btype}")
                if "ori_w" not in btype:
                    ordered_vel_output_names.append(f"{body_name}:{btype}")

        for body_name in body_names_list:
            ori = any("ori" in b for b in body_type[body_name])
            pos = any("pos" in b for b in body_type[body_name])
            if pos and ori:
                body_type_list.append(2)
            elif pos:
                body_type_list.append(1)
            elif ori:
                body_type_list.append(0)

        for joint_name in joint_names_list:
            ordered_pos_output_names.append(f"joint:{joint_name}")
            ordered_vel_output_names.append(f"joint:{joint_name}")

        joint_idx_result = joint_indices if len(joint_indices) > 0 else None
        body_idx_result = body_indices if len(body_indices) > 0 else None

        return joint_idx_result, body_idx_result, use_com, ordered_pos_output_names, ordered_vel_output_names, body_type_list

    def _parse_ref_frames(self, reference_frames: list[str]) -> tuple[list[int], list[str]]:
        """Parse reference frame names to body indices, adding left/right pairs.

        Args:
            reference_frames: Body frame names from the trajectory data.

        Returns:
            ``(frame_indices, expanded_frame_names)``
        """
        expanded_frames: list[str] = []
        for frame_name in reference_frames:
            if frame_name not in expanded_frames:
                expanded_frames.append(frame_name)
            if frame_name.startswith("right"):
                opposite = "left" + frame_name[5:]
                if opposite not in expanded_frames:
                    expanded_frames.append(opposite)
            elif frame_name.startswith("left"):
                opposite = "right" + frame_name[4:]
                if opposite not in expanded_frames:
                    expanded_frames.append(opposite)

        frame_indices: list[int] = []
        for frame_name in expanded_frames:
            if frame_name in self.robot.body_names:
                frame_indices.append(self.robot.body_names.index(frame_name))
            else:
                raise ValueError(f"Reference frame '{frame_name}' not found in robot body names.")

        return frame_indices, expanded_frames


# ======================================================================
# Module-level helpers
# ======================================================================

def _find_output_indices(ordered_names: list[str], frame_name: str, suffix_pattern: str) -> list[int]:
    """Return indices of outputs named ``"{frame_name}:...{suffix_pattern}..."``.

    Args:
        ordered_names: Ordered output names (position or velocity list).
        frame_name: Frame the outputs belong to (e.g. ``"pelvis_link"``).
        suffix_pattern: Substring identifying the output kind (e.g. ``"pos_"`` or ``"ori_"``).

    Returns:
        Matching indices in the order they appear in ``ordered_names``.
    """
    return [
        i for i, name in enumerate(ordered_names)
        if name.startswith(f"{frame_name}:") and suffix_pattern in name
    ]


def _align_yaw(vec: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Rotate *vec* into the yaw-aligned frame of *root_quat*."""
    return quat_apply(yaw_quat(quat_inv(root_quat)), vec)


def _align_yaw_batched(vecs: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors ``[N, B, 3]`` into the yaw-aligned frame.

    Args:
        vecs: ``[N, B, 3]`` vectors.
        root_quat: ``[N, 4]`` reference quaternions.

    Returns:
        Aligned vectors of shape ``[N, B, 3]``.
    """
    N, B, _ = vecs.shape
    yaw_inv = yaw_quat(quat_inv(root_quat))
    yaw_inv_expanded = yaw_inv.unsqueeze(1).expand(N, B, 4).reshape(N * B, 4)
    vecs_flat = vecs.reshape(N * B, 3)
    result = quat_apply(yaw_inv_expanded, vecs_flat)
    return result.reshape(N, B, 3)


def _align_quat_to_yaw(quat: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Align a quaternion to the yaw-aligned frame."""
    return quat_mul(yaw_quat(quat_inv(root_quat)), quat)


def _align_quat_to_yaw_batched(quats: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Align batched quaternions ``[N, B, 4]`` to the yaw-aligned frame.

    Args:
        quats: ``[N, B, 4]`` quaternions.
        root_quat: ``[N, 4]`` reference quaternions.

    Returns:
        Aligned quaternions of shape ``[N, B, 4]``.
    """
    N, B, _ = quats.shape
    yaw_inv = yaw_quat(quat_inv(root_quat))
    yaw_inv_expanded = yaw_inv.unsqueeze(1).expand(N, B, 4).reshape(N * B, 4)
    quats_flat = quats.reshape(N * B, 4)
    result = quat_mul(yaw_inv_expanded, quats_flat)
    return result.reshape(N, B, 4)


def get_euler_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) to Euler angles ``[roll, pitch, yaw]``.

    Args:
        quat: Quaternion tensor of shape ``[..., 4]``.

    Returns:
        Euler angles tensor of shape ``[..., 3]``.
    """
    euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
    euler_x = wrap_to_pi(euler_x)
    euler_y = wrap_to_pi(euler_y)
    euler_z = wrap_to_pi(euler_z)
    return torch.stack([euler_x, euler_y, euler_z], dim=-1)
