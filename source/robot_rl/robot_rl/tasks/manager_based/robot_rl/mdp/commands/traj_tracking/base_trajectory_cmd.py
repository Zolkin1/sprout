# Copyright 2026 Zachary Olkin. All rights reserved.

"""Abstract base class for trajectory command terms.

Holds the manager-agnostic, phase-free shared logic.  Concrete commands
(:class:`MotionSequenceCommand`, :class:`FmTrackingCommand`) extend it directly.

There is NO phase-as-observation subclass: the single-skill
``PhasedTrajectoryCommand`` (phase obs + hold-phi logic) and its
``_phased_cmd`` / ``ref_sin_phase`` / ``ref_cos_phase`` observation terms were
deleted with the single-skill stack.  Clip phase is now carried as
``manager.phase`` (see :meth:`_update_command`), advanced from the clip step.
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
from motion_gen.joint_names import ROOT_FRAME_MARKER, is_root_frame, strip_root_marker
from motion_gen.manager_base import ManagerBase
from motion_gen.sagittal_reflector import NamedReflector, SagittalReflectionConfig
from motion_gen.manager_base import TrajectoryType
from .world_lift_isaac import lift_base_world_state


class BaseTrajectoryCommand(CommandTerm):
    """Trajectory command term base class.

    Handles measured/desired output computation, CLF tracking, ref-frame
    bookkeeping, and metrics logging.  Trajectory advancement is time-based
    only; no phasing variable is stored on this class -- subclasses publish
    progress as ``manager.phase`` instead (``MotionSequenceCommand`` sets it
    from the clip step).  The old phase-as-observation / hold-phi subclass is
    gone; see the module docstring.

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
            msg = (
                f"\n{bar}\n"
                "TRAJECTORY LIBRARY MISMATCH\n"
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
            if getattr(cfg, "strict_fingerprint", False):
                raise RuntimeError(msg)
            print(f"[WARN] {msg}")

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

        # NOTE: ``current_domain`` / ``prev_ref_frame_idx`` / ``cur_ref_frame_idx`` used to live
        # here. They drove the "needs a ref_poses refresh" gate in the deleted
        # ``get_measured_outputs`` (a domain change or a stance-foot identity change forced an
        # anchor swap). The foot-stance paradigm is gone, so nothing wrote them and they held their
        # init value for entire runs -- while still being logged and plotted, which rendered a
        # plausible constant rather than an obviously missing panel. All three are deleted.

        # --- Output parsing -----------------------------------------------
        result = self._parse_outputs(self.manager.get_pos_output_names)
        (self.joint_idx, self.body_idx, self.use_com,
         self.ordered_pos_output_names, self.ordered_vel_output_names,
         self.body_type, self.root_body_idx, self.root_body_frames) = result

        # Articulation index of the frame the root-frame channels are expressed in.  Read from
        # ``body_pos_w`` / ``body_link_vel_w`` rather than ``root_pos_w`` / ``root_lin_vel_w`` so
        # the base and the tracked bodies come from the SAME arrays -- ``root_*_vel_w`` is a
        # centre-of-mass quantity in Isaac Lab, which would put a lever-arm error into the
        # relative velocity.
        self._root_base_body_idx = self.robot.body_names.index(
            getattr(self.cfg, "base_frame_name", "pelvis_link")
        )

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
        # The foot-STANCE reference-frame paradigm is gone: it re-anchored a clip to a planted
        # foot and belonged to the online CLF commands, whose reference frames WERE contact
        # bodies.  Clips are now world-anchored or periodically ego-anchored through the per-DOF
        # anchor mask, and declare a NON-contact reference frame (e.g. pelvis_link).  What is
        # left here is just the name -> output-index resolution the reference frames need.
        #
        # The stance machinery is deleted: ``get_measured_outputs``, ``terrain_aware_ref.py`` and
        # the ``uses_stance_ref`` / ``ref_to_contact_idx`` bookkeeping are all gone.
        #
        # The manager ``ref_frame`` and the clip-io ``ref_frame`` metadata deliberately SURVIVE:
        # they name the frame a clip's poses are expressed RELATIVE TO, which is independent of
        # how the clip is ANCHORED (the per-DOF anchor mask).  Conflating the two is what the
        # stance paradigm did -- do not "finish the job" by deleting ``ref_frame`` as well.
        self.ref_frame_indices, self.ref_frames = self._parse_ref_frames(
            self.manager.get_reference_frames()
        )
        # Current reference frame poses — [N, 7] = [pos_x, pos_y, pos_z, qx, qy, qz, qw].
        # Always allocated: the teacher / tracker reuse this as the clip WORLD-placement scratch
        # (see motion_sequence_cmd._clip_world_frames), independent of the stance path.
        self.ref_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self.ref_poses[:, 6] = 1.0  # identity quaternion

        # --- CLF ----------------------------------------------------------
        self.tracked_vel_names = self._resolve_tracked_vel_names()
        self.clf = CLF(
            sim_dt=self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            ordered_vel_output_names=self.ordered_vel_output_names,
            ordered_pos_output_names=self.ordered_pos_output_names,
            Q_weights=self.cfg.Q_weights,
            R_weights=self.cfg.R_weights,
            tracked_vel_names=self.tracked_vel_names,
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

        # --- Root-frame bodies ---
        # The DeepMimic / Sonic parameterization: each link's pose and twist relative to the
        # robot's own root, expressed in the root frame.  Note this uses the FULL inverse root
        # rotation, not the yaw-only ``_align_yaw_batched`` the world block above uses -- so the
        # result is exactly FK(q) and pelvis roll/pitch error is charged only to the pelvis
        # channels instead of being smeared across every body.
        if self.root_body_idx is not None:
            body_pos_w = wp.to_torch(self.robot.data.body_pos_w)
            body_quat_w = wp.to_torch(self.robot.data.body_quat_w)
            body_vel_w = wp.to_torch(self.robot.data.body_link_vel_w)

            b = self._root_base_body_idx
            base_pos_w = body_pos_w[:, b, :]                      # [N, 3]
            base_quat_w = body_quat_w[:, b, :]                    # [N, 4] xyzw
            base_lin_w = body_vel_w[:, b, :3]                     # [N, 3]
            base_ang_w = body_vel_w[:, b, 3:]                     # [N, 3]
            inv_base = quat_inv(base_quat_w)

            p_w = body_pos_w[:, self.root_body_idx, :]            # [N, B, 3]
            q_w = body_quat_w[:, self.root_body_idx, :]           # [N, B, 4]
            v_w = body_vel_w[:, self.root_body_idx, :3]
            w_w = body_vel_w[:, self.root_body_idx, 3:]

            p_r = _rotate_by_inv_batched(p_w - base_pos_w.unsqueeze(1), base_quat_w)
            q_r = _quat_mul_inv_batched(q_w, base_quat_w)
            w_r = _rotate_by_inv_batched(w_w - base_ang_w.unsqueeze(1), base_quat_w)
            # d/dt (R^T (p - p0)) = R^T (v - v0) - omega_base x p_r, with omega_base expressed in
            # the base frame.  The cross term is what makes this the true derivative of ``p_r``
            # (and what makes it match the generator's ``jv @ qdot``).
            base_ang_b = quat_apply(inv_base, base_ang_w)         # [N, 3]
            v_r = (_rotate_by_inv_batched(v_w - base_lin_w.unsqueeze(1), base_quat_w)
                   - torch.linalg.cross(base_ang_b.unsqueeze(1).expand_as(p_r), p_r))

            nb = len(self.root_body_idx)
            self.y_act[:, pos_output_idx:pos_output_idx + 3 * nb] = p_r.flatten(1)
            self.dy_act[:, vel_output_idx:vel_output_idx + 3 * nb] = v_r.flatten(1)
            pos_output_idx += 3 * nb
            vel_output_idx += 3 * nb

            self.y_act[:, pos_output_idx:pos_output_idx + 4 * nb] = q_r.flatten(1)
            self.dy_act[:, vel_output_idx:vel_output_idx + 3 * nb] = w_r.flatten(1)
            pos_output_idx += 4 * nb
            vel_output_idx += 3 * nb

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

    def _update_metrics(self):
        """Log CLF values, the per-subgroup error split, and (optionally) every channel.

        The per-channel breakdown is one ``abs()`` tensor per channel per step -- 123 of them
        for the world+joint layout and 292 once root-frame body channels are added.  That is
        a lot of logging traffic for something only useful when debugging a specific channel,
        so it is gated behind ``cfg.log_per_channel_metrics``.  The per-subgroup CLF energies
        are always logged and are what you normally want.
        """
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

        for name, value in self.clf.v_subgroups.items():
            self.metrics[f"v_{name}"] = value

        if not getattr(self.cfg, "log_per_channel_metrics", False):
            return

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
    ) -> tuple[list[int] | None, list[int] | None, bool, list[str], list[str], list[int],
               list[int] | None, list[str]]:
        """Parse output names into robot-data indices and ordered name lists.

        Frames carrying the ``@root`` marker (see
        :data:`~motion_gen.joint_names.ROOT_FRAME_MARKER`) are collected into a SEPARATE
        group: their channels hold the body pose expressed in the robot's own root (pelvis)
        frame rather than the world/clip frame, so :meth:`compute_measured_output` fills them
        with a different transform.  The underlying body is resolved by stripping the marker.

        Args:
            pos_output_names: Position output names (the superset including ``ori_w``).

        Returns:
            ``(joint_idx, body_idx, use_com, ordered_pos_names, ordered_vel_names,
            body_type_list, root_body_idx, root_body_names)``
        """
        output_names = pos_output_names
        joint_indices: list[int] = []
        joint_names_list: list[str] = []
        body_indices: list[int] = []
        body_names_list: list[str] = []
        root_indices: list[int] = []
        root_names_list: list[str] = []          # marked names, e.g. "left_knee_link@root"
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
                body_name = strip_root_marker(frame_name)
                if body_name not in self.robot.body_names:
                    raise ValueError(f"Body frame '{body_name}' not found in robot body names.")
                idx = self.robot.body_names.index(body_name)
                if is_root_frame(frame_name):
                    root_indices.append(idx)
                    root_names_list.append(frame_name)
                else:
                    body_indices.append(idx)
                    body_names_list.append(frame_name)
                added_bodies.add(frame_name)

        # Root-frame channels are generated as a pose + a twist, always together.  Anything
        # else would silently mis-align the blocked write in compute_measured_output.
        for frame_name in root_names_list:
            have = set(body_type[frame_name])
            want = {"pos_x", "pos_y", "pos_z", "ori_w", "ori_x", "ori_y", "ori_z"}
            if have != want:
                raise ValueError(
                    f"Root-frame '{frame_name}' must expose exactly {sorted(want)}; got "
                    f"{sorted(have)}."
                )

        # Build ordered output names:
        #   COM -> world body positions -> world body orientations
        #       -> root body positions -> root body orientations -> joints
        # compute_measured_output writes in exactly this order.
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

        for frame_name in root_names_list:
            for a in ("x", "y", "z"):
                ordered_pos_output_names.append(f"{frame_name}:pos_{a}")
                ordered_vel_output_names.append(f"{frame_name}:pos_{a}")
        for frame_name in root_names_list:
            for a in ("x", "y", "z", "w"):
                ordered_pos_output_names.append(f"{frame_name}:ori_{a}")
                if a != "w":
                    ordered_vel_output_names.append(f"{frame_name}:ori_{a}")

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
        root_idx_result = root_indices if len(root_indices) > 0 else None

        return (joint_idx_result, body_idx_result, use_com, ordered_pos_output_names,
                ordered_vel_output_names, body_type_list, root_idx_result, root_names_list)

    def _resolve_tracked_vel_names(self) -> list[str]:
        """Velocity channels that form the CLF's eta, per ``cfg.tracking_mode``.

        ``y``/``dy`` always carry every channel; this picks the graded subset.  ``"joint"``
        keeps the world frames and the joints (the original behaviour, so the returned list is
        the full one minus nothing when the library carries no root-frame channels).
        ``"body"`` swaps the joints for the root-frame body links, keeping the world-frame
        skeleton so global placement stays penalised.

        Returns:
            Channel names, a subset of ``self.ordered_vel_output_names``.

        Raises:
            ValueError: On an unknown mode, or on ``"body"`` against a library that carries no
                root-frame channels.
        """
        mode = getattr(self.cfg, "tracking_mode", "joint")
        names = self.ordered_vel_output_names
        if mode == "joint":
            return [n for n in names if not is_root_frame(n)]
        if mode != "body":
            raise ValueError(f"tracking_mode must be 'joint' or 'body'; got {mode!r}.")
        if not self.root_body_frames:
            raise ValueError(
                "tracking_mode='body' needs a clip library carrying root-frame body channels "
                f"({ROOT_FRAME_MARKER}), but this library has none. Widen it with "
                "scripts/rsl_rl/backfill_root_frames.py, or use tracking_mode='joint'."
            )
        return [n for n in names if not n.startswith("joint:")]

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


def _rotate_by_inv_batched(vecs: torch.Tensor, frame_quat: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors ``[N, B, 3]`` into ``frame_quat``'s frame (FULL rotation).

    The root-frame counterpart of :func:`_align_yaw_batched`, which drops roll/pitch.

    Args:
        vecs: ``[N, B, 3]`` world-frame vectors.
        frame_quat: ``[N, 4]`` frame orientations (xyzw).

    Returns:
        ``[N, B, 3]`` vectors in the frame.
    """
    n, b, _ = vecs.shape
    inv = quat_inv(frame_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_apply(inv, vecs.reshape(n * b, 3)).reshape(n, b, 3)


def _quat_mul_inv_batched(quats: torch.Tensor, frame_quat: torch.Tensor) -> torch.Tensor:
    """Express batched orientations ``[N, B, 4]`` relative to ``frame_quat`` (FULL rotation).

    Args:
        quats: ``[N, B, 4]`` world-frame orientations (xyzw).
        frame_quat: ``[N, 4]`` frame orientations (xyzw).

    Returns:
        ``[N, B, 4]`` orientations relative to the frame.
    """
    n, b, _ = quats.shape
    inv = quat_inv(frame_quat).unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
    return quat_mul(inv, quats.reshape(n * b, 4)).reshape(n, b, 4)


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
