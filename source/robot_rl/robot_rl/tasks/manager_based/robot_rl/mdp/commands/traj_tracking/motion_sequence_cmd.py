# Copyright 2026 Zachary Olkin. All rights reserved.

"""Open-loop motion-sequence tracking command (world-frame CLF).

Tracks a pre-generated fixed motion clip (see ``clips/``) by a pure step
index -- no contact gates, no online re-anchoring, no skill cross-fades.
Clips are stored in a single clip-world frame (the per-step stance anchor is
baked in at generation time), so the only runtime transform is a per-env
*placement* (the clip's yaw + xy in the sim world, fixed at reset).  At reset
each env is assigned a clip and placed exactly on it; thereafter the reference
advances by ``step = episode_length_buf``.

The CLF runs in the **absolute world frame**: the desired outputs
``y_des``/``dy_des`` are the clip's outputs lifted through the placement, and
the measured outputs ``y_act``/``dy_act`` are read with an identity reference
(no yaw re-alignment).  This is the same world frame
:class:`FmTrackingCommand` uses, so the two tasks share one frame.
Because the placement is fixed per episode, there is no re-anchoring to the
measured foot during training -- the left/right tracking asymmetry of the
online command is gone.
"""

from __future__ import annotations

import os

import torch
import warp as wp
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

from robot_rl.flow_matching.window import world_to_window

from .base_trajectory_cmd import BaseTrajectoryCommand
from motion_gen.joint_names import is_root_frame, root_frame_name
from motion_gen.sagittal_reflector import NamedReflector, SagittalReflectionConfig
from motion_gen.clips.clip_io import load_clip_library
from motion_gen.clips.clip_fingerprint import fingerprint_clip_config, verify_clip_fingerprint
from motion_gen.clips.clip_manager import MotionClipManager
from .world_lift_isaac import (
    ego_placement_masked,
    ego_placement_masked_window,
    lift_base_world_state,
    lift_orientation,
    lift_orientation_batched,
    lift_position,
    lift_position_batched,
    lift_vector,
    lift_vector_batched,
)
from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg

# Per-frame debug-marker colors (RGB 0-1); matches the viser/plot tooling.
_DEBUG_FRAME_COLOR_MAP: dict[str, tuple[float, float, float]] = {
    "pelvis_link": (0.95, 0.55, 0.10),
    "left_ankle_roll_link": (0.20, 0.45, 1.00),
    "right_ankle_roll_link": (0.90, 0.20, 0.25),
    "left_wrist_yaw_link": (0.20, 0.80, 0.85),
    "right_wrist_yaw_link": (0.80, 0.30, 0.90),
}
_DEBUG_FRAME_FALLBACK_COLOR: tuple[float, float, float] = (0.6, 0.6, 0.6)

# Tracked-body spheres: what the robot IS vs what it is REWARDED AGAINST.  Two spheres per
# tracked body per env, so the reward's own error is visible as the gap between them.  The
# reference here is the clip AFTER the per-step re-datum onto the robot, which is the whole
# point -- against the global clip the pair would separate as the robot drifts even while
# tracking perfectly.
_DEBUG_ACTUAL_COLOR: tuple[float, float, float] = (0.15, 0.95, 0.35)      # green  = measured
_DEBUG_REWARDED_COLOR: tuple[float, float, float] = (1.00, 0.15, 0.85)    # magenta = reference
# The obs look-ahead ribbon: the base of the reference window the POLICY sees, which rides
# the robot (anchored) and so separates from the global clip ribbon exactly by the drift.
_DEBUG_OBS_WINDOW_COLOR: tuple[float, float, float] = (1.00, 0.95, 0.20)  # yellow
# Frame-name substrings treated as "hand" end-effectors (filtered out of the
# debug ribbon unless cfg.debug_vis_show_hands is True).
_DEBUG_HAND_FRAME_KEYS: tuple[str, ...] = ("wrist", "hand")
_DEBUG_VIZ_PRIM_PATH: str = "/Visuals/motion_seq_ref"
_DEBUG_BODY_VIZ_PRIM_PATH: str = "/Visuals/motion_seq_bodies"
_DEBUG_VIZ_EPS: float = 1e-6  # guard against zero-length segments / degenerate axes


class MotionSequenceCommand(BaseTrajectoryCommand):
    """Track fixed pre-generated motion clips open-loop."""

    cfg: MotionSequenceCommandCfg

    # Lazily built once the simulator is up (CommandTerm may call
    # ``_set_debug_vis_impl`` before ``_post_init``).  See the debug-vis block.
    _debug_markers: object | None = None
    _debug_body_markers: object | None = None
    _debug_vis_pending: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _create_manager(self, cfg: MotionSequenceCommandCfg, env) -> MotionClipManager:
        """Load the dense clip library and wrap it in a gather manager."""
        if not cfg.path:
            raise ValueError(
                "MotionSequenceCommand has no clip library path. The terrain tasks carry no "
                "default (a stale default trains against the wrong terrain); pass it with "
                "--clip <lib.h5>, which also points the ClipTerrainImporter at the same file."
            )
        self._verify_terrain_library_matches(cfg, env)
        device = cfg.clip_device or env.device
        self._clip_lib = load_clip_library(cfg.path, device)
        return MotionClipManager(self._clip_lib)

    @staticmethod
    def _verify_terrain_library_matches(cfg: MotionSequenceCommandCfg, env) -> None:
        """Fail if the terrain importer meshes a DIFFERENT library than this command tracks.

        The clip ``.h5`` carries both the reference motions and the terrain they were generated
        on, so the importer's ``clip_library_path`` and this command's ``path`` must be the same
        file -- otherwise each env spawns on a tile whose ``terrain_id`` means nothing to its
        clip, and training proceeds quietly against terrain the reference does not belong to.

        The CLI cannot produce a mismatch, and this is deliberately NOT relied on to fix one:
        ``--clip`` sets both fields and runs after hydra (so overriding only
        ``commands.traj_ref.path`` is simply overwritten), and overriding
        ``scene.terrain.clip_library_path`` is rejected outright because its cfg default is
        ``MISSING`` and hydra type-checks the override.  What this guards is the
        *programmatic* path: a new env cfg, task variant or tool that sets one field and
        forgets the other.  That failure is silent and expensive, hence a hard raise here
        rather than trust.  Compared as real paths so absolute and relative spellings of the
        same file agree.

        No-op for a terrain with no ``clip_library_path`` (i.e. not a ClipTerrainImporter).

        Args:
            cfg: This command term's configuration.
            env: The IsaacLab environment (its ``cfg.scene.terrain`` holds the importer cfg).

        Raises:
            ValueError: If both paths are set and refer to different files.
        """
        terrain = getattr(getattr(getattr(env, "cfg", None), "scene", None), "terrain", None)
        terrain_path = getattr(terrain, "clip_library_path", None)
        if not terrain_path:
            return
        if os.path.realpath(terrain_path) != os.path.realpath(cfg.path):
            raise ValueError(
                "clip library mismatch: the ClipTerrainImporter meshes\n"
                f"    scene.terrain.clip_library_path = {terrain_path}\n"
                "but the tracking command follows\n"
                f"    commands.traj_ref.path          = {cfg.path}\n"
                "These MUST be the same file -- the tiles' terrain_ids come from the importer's "
                "library and the clips' terrain_ids from the command's, so a mismatch spawns every "
                "env on terrain its reference was not generated for. Pass the library once with "
                "--clip <lib.h5>, which sets both; do not override either path on its own."
            )

    def _verify_contact_frames(self) -> None:
        """No stance reference frame to verify: teacher clips are world/ego(mask)-anchored.

        The teacher never runs the foot-stance path (it overrides ``_update_command`` and anchors
        via the per-DOF anchor mask), so the clip's ``ref_frame`` need not be a contact body -- it
        is just the clip's base/world frame (e.g. ``pelvis_link``).

        Kept as an explicit no-op so the reason is recorded at the override site. NOTE the base
        ``_verify_contact_frames`` is itself an empty extension-point hook, so nothing anywhere
        currently validates contact frames -- this override is documentation, not a behaviour
        change, and both could go if the hook is ever removed from the base class.
        """
        return

    def _post_init(self) -> None:
        """Allocate clip state, validate dt, map contacts, record fingerprint."""
        n, dev = self.num_envs, self.device
        T = self.manager.num_steps

        # The clip must be sampled at the control dt so step == step counter.
        step_dt = float(self.env.step_dt)
        if abs(self.manager.dt - step_dt) > 1e-6:
            raise ValueError(
                f"Clip dt {self.manager.dt} != env.step_dt {step_dt}. Regenerate clips "
                f"with --dt {step_dt}."
            )

        # Per-env tracking state.
        self.clip_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.start_step = torch.zeros(n, dtype=torch.long, device=dev)
        self._step = torch.zeros(n, dtype=torch.long, device=dev)
        # Per-env world placement of the clip frame (yaw-only ground frame).
        self.clip_world_pos = torch.zeros(n, 3, device=dev)
        self.clip_world_quat = torch.zeros(n, 4, device=dev)
        self.clip_world_quat[:, 3] = 1.0  # identity (xyzw, w last -- Isaac Lab convention)
        # Reset-time placement (the clip's terrain-cell anchor), set in
        # ``_commit_clip_reset``. FIXED anchor-mask DOFs pin to this on the ego
        # re-anchor cadence so terrain-fixed axes never drift onto the robot.
        self._reset_world_pos = torch.zeros(n, 3, device=dev)
        self._reset_world_yaw = torch.zeros(n, device=dev)
        # Heading (rad) of the TERRAIN's own frame, which the anchor mask's downrange / lateral
        # bits are resolved in (see world_lift.ego_placement_masked). Authored with the terrain and
        # served per env by the ClipTerrainImporter; 0 for flat ground and for any library whose
        # manifest predates the field, which reproduces world-axis masking exactly.
        self._mask_frame_yaw = torch.zeros(n, device=dev)
        # --- Re-anchor SNAPSHOT (the per-segment placement's single source) ---
        # A placement is a pure function of (terrain-cell anchor, mask, snapshot), so instead of
        # storing one solved placement we store the pose pair captured at the last re-anchor and
        # solve per FRAME against that frame's own mask. That is what lets a clip's terrain-pinned
        # segment stay welded to the terrain while its flat approach rides the robot: both are
        # derived from the same snapshot through different mask bits. Seeded at reset so the derived
        # placement is exactly the terrain-cell anchor (zero correction).
        self._snap_base_pos = torch.zeros(n, 3, device=dev)      # clip-world base pos at the snapshot step
        self._snap_base_quat = torch.zeros(n, 4, device=dev)     # clip-world base ori (xyzw)
        self._snap_base_quat[:, 3] = 1.0
        self._snap_target_pos = torch.zeros(n, 3, device=dev)    # robot root world pos at the snapshot
        self._snap_target_quat = torch.zeros(n, 4, device=dev)   # robot root world ori (xyzw)
        self._snap_target_quat[:, 3] = 1.0
        # Per-env ego-centric re-anchor assignment (re-rolled per reset in
        # reset_on_motion_sequence): True => this env's clip placement is
        # periodically snapped onto the robot root (receding-horizon reference).
        self._ego_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        # Per-env FM SNAP mask: which envs emit a snap (FM re-inference) event on the
        # re-anchor cadence, DECOUPLED from the clip re-placement (``_ego_mask``). On
        # flat this equals ``_ego_mask`` (only ego envs snap); on terrain it also
        # includes the world-anchored stair/transition envs (set in the reset) so the
        # FM still gets a training window on stairs -- the clip stays locked to the
        # treads (never re-placed) while the recorder marks the snap. Set per reset in
        # ``_commit_clip_reset``; defaults to ``_ego_mask`` when not given.
        self._snap_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        # Per-env per-step flag: True on the steps that are FM snap (re-inference) events
        # for this env (``_snap_mask`` envs on the cadence tick). Consumed by the data
        # recorder so the FM trains on snap-event anchors (the dataset ego-anchors the
        # recorded clip future to the robot base at each snap), matching the deploy
        # regime (each FM re-inference is a snap).
        self._reanchor_active = torch.zeros(n, dtype=torch.bool, device=dev)
        # Phase carrier so the base ``_update_command`` can read manager.phase.
        self.manager.phase = torch.zeros(n, device=dev)

        # Base-frame output indices for world reconstruction (placement / obs).
        # Force the orientation channels to xyzw order: clips store ori scalar-first
        # (ori_w) while Isaac Lab is scalar-last, and ``world_lift_isaac`` -- the one
        # place that reorders -- takes this index scalar-last (it permutes back to the
        # scalar-first core itself).  So the lifted base quaternion is xyzw, matching
        # root_quat_w / write_root_pose_to_sim_index.
        base = self.cfg.base_frame_name
        self._base_idx = dict(self._reference_world_indices(base))
        self._base_idx["ori"] = torch.tensor(
            [self.ordered_pos_output_names.index(f"{base}:ori_{a}") for a in ("x", "y", "z", "w")],
            dtype=torch.long, device=dev,
        )

        # Map stored contact columns (clip contact_frames order) to the
        # command's contact_bodies order; -1 where the clip has no schedule.
        clip_cf = self.manager.contact_frames
        self._contact_col = torch.tensor(
            [clip_cf.index(b) if b in clip_cf else -1 for b in self.contact_bodies],
            dtype=torch.long, device=dev,
        )

        self._record_clip_fingerprint(self.cfg)

        # --- Debug-viz setup (per reference frame: pos indices + color) ---
        # WORLD frames only -- the ribbon draws world-space polylines, and a root-frame
        # channel would be drawn as if it were a world position (a blob at the origin).
        self._debug_frame_names = [
            n[: -len(":pos_x")] for n in self.ordered_pos_output_names
            if n.endswith(":pos_x") and not is_root_frame(n)
        ]
        # Optionally drop hand/wrist end-effector frames from the ribbon.
        if not self.cfg.debug_vis_show_hands:
            self._debug_frame_names = [
                f for f in self._debug_frame_names
                if not any(k in f for k in _DEBUG_HAND_FRAME_KEYS)
            ]
        self._debug_frame_pos_idx = [
            torch.tensor(
                [self.ordered_pos_output_names.index(f"{f}:pos_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            )
            for f in self._debug_frame_names
        ]
        self._debug_frame_colors = [
            _DEBUG_FRAME_COLOR_MAP.get(f, _DEBUG_FRAME_FALLBACK_COLOR) for f in self._debug_frame_names
        ]
        # Forward-window step offsets (current step .. +horizon), subsampled.
        horizon_steps = max(1, int(round(self.cfg.debug_vis_horizon_s / self.manager.dt)))
        stride = max(1, int(self.cfg.debug_vis_stride))
        self._debug_offsets = torch.arange(0, horizon_steps, stride, dtype=torch.long, device=dev)

        # --- World-frame assembly (shared with FmTrackingCommand) ---
        # Output frames whose Cartesian channels are lifted clip-world -> world:
        # the base plus every end-effector reference frame, in resolved order.
        # (FmTrackingCommand rebuilds these in its FM frame order.)
        self._build_world_frame_maps([self.cfg.base_frame_name] + self.resolve_ee_frames())

        # Identity reference for the world-frame measured outputs (no alignment).
        self._zeros3 = torch.zeros(n, 3, device=dev)
        self._ident_quat = torch.zeros(n, 4, device=dev)
        self._ident_quat[:, 3] = 1.0  # xyzw identity
        # World +z, for rebuilding the global (pure-yaw) placement quaternion in
        # :meth:`termination_reference_world_state`.
        self._world_z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(1, 3)
        # Once-per-step guard for the re-anchor snapshot; -1 so the first step always runs.
        self._last_reanchor_step = -1
        # True only while _resample_command is running -- see _reanchor_ego_envs.
        self._resampling = False
        # CLF outputs. Initialised here because ``CommandTerm.compute`` calls
        # ``_update_metrics`` (which reads them) BEFORE ``_update_command`` (which writes
        # them), so the very first step of a run would otherwise hit an unset attribute.
        self.v = torch.zeros(n, device=dev)
        self.vdot = torch.zeros(n, device=dev)

    def _resample_command(self, env_ids) -> None:
        """Reset-time refresh, flagged so the re-anchor sits it out.

        The base class implements resampling by running a whole ``_update_command()``, and
        Isaac Lab calls this from ``_reset_idx`` -- BEFORE it zeroes ``episode_length_buf``.
        So during this call ``_advance_step`` computes ``_step`` from the DYING episode's
        length against the NEW ``start_step``, and everything derived from ``_step`` is
        nonsense for exactly this one call.  Most of it is harmless: the following
        ``command_manager.compute()`` re-runs the same chain with the counter zeroed and
        overwrites it.

        The re-anchor SNAPSHOT is the exception, because it is guarded to once per
        ``common_step_counter``: taken here, it both captures the clip base at a step far
        past ``start_step`` AND stamps the guard, so the good seed
        :meth:`seed_reanchor_snapshot` just wrote is destroyed and the corrected retake never
        happens.  The placement is then off by the clip's travel over that bogus span --
        planar only, since the mask pins z -- which is what tripped ``ee_pose_error`` on ~38%
        of resets while never firing once mid-episode.

        Args:
            env_ids: Env indices being reset (unused; the base refreshes every env).
        """
        self._resampling = True
        try:
            super()._resample_command(env_ids)
        finally:
            self._resampling = False

    def _record_clip_fingerprint(self, cfg: MotionSequenceCommandCfg) -> None:
        """Persist (and verify) the loaded clip library's fingerprint.

        Warns loudly if ``cfg.clip_fingerprint`` (injected at play/resume from
        the trained-against ``env.yaml``) differs from the loaded library's,
        then writes the loaded value back so a fresh run serialises it.
        """
        loaded = self._clip_lib.attrs.get("clip_fingerprint")
        if loaded is None:
            return
        trained = getattr(cfg, "clip_fingerprint", None)
        if trained is not None:
            ok, msg = verify_clip_fingerprint(trained, loaded)
            if not ok:
                if getattr(cfg, "strict_fingerprint", False):
                    raise RuntimeError(msg)
                print("\n" + "=" * 78 + f"\n[WARN] {msg}\n" + "=" * 78 + "\n")
        cfg.clip_fingerprint = loaded

    # ------------------------------------------------------------------
    # Lifecycle (overrides the online phase/gate machinery)
    # ------------------------------------------------------------------

    def _build_world_frame_maps(self, frames: list[str]) -> None:
        """Build the world-frame channel index maps + per-step cache for ``frames``.

        ``frames`` is ``[base] + ee`` (the base frame followed by the ee
        reference frames whose Cartesian channels get lifted clip-world ->
        world).  Populates the flat scatter indices into ``y_des``/``dy_des``,
        a name->row lookup, and the ``[N, nf, 13]`` world-frame cache served by
        :meth:`reference_world_frames` / :meth:`reference_world_state`.
        """
        self._wf_frames = list(frames)
        self._wf_frame_index = {f: i for i, f in enumerate(self._wf_frames)}
        pos_list, ori_list, lin_list, ang_list = [], [], [], []
        for f in self._wf_frames:
            fidx = self._frame_output_indices(f)
            pos_list.append(fidx["pos"]); ori_list.append(fidx["ori"])
            lin_list.append(fidx["lin_vel"]); ang_list.append(fidx["ang_vel"])
        self._wf_pos_flat = torch.cat(pos_list)   # [nf*3] -> y_des
        self._wf_ori_flat = torch.cat(ori_list)   # [nf*4] -> y_des
        self._wf_lin_flat = torch.cat(lin_list)   # [nf*3] -> dy_des
        self._wf_ang_flat = torch.cat(ang_list)   # [nf*3] -> dy_des
        self._wf_world_frames = torch.zeros(self.num_envs, len(self._wf_frames), 13, device=self.device)

    def _update_command(self) -> None:
        """Advance the step, build world-frame desired + measured, update CLF."""
        self._advance_step()
        self._reanchor_ego_envs()                  # ego-centric re-anchor snapshot (subset of envs)
        self._refresh_current_placement()          # placement for THIS step's segment mask
        self.get_desired_outputs()                 # clip-world y_des/dy_des
        frames = self._clip_world_frames()         # lift through the placement
        self._write_world_des(frames)              # base+ee Cartesian -> world
        # Measured outputs in the world frame (identity reference -> no alignment).
        self.compute_measured_output(self._zeros3, self._ident_quat)
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot = vdot
        self.v = vcur

    def _advance_step(self) -> None:
        """Set the per-env clip step from the episode step counter.

        ``episode_length_buf`` is stable within a step, so this is idempotent
        across Isaac Lab's twice-per-step command compute -- no double-advance.
        """
        T = self.manager.num_steps
        self._step = torch.clamp(self.start_step + self.env.episode_length_buf, max=T - 1)
        self.manager.phase = self._step.float() / max(T - 1, 1)

    def _reanchor_ego_envs(self) -> None:
        """Emit FM snap events and re-place the clip for ego envs, on SEPARATE cadences.

        Two things happen here and they run on **independent timers**, because the right
        rate for each is different by an order of magnitude:

        * **Clip re-placement** (``_reanchor_clip_ids``, every ``cfg.ego_anchor_every``
          steps for ``_ego_mask`` envs) -- recompute the placement so the FREE root DOFs of
          the clip's CURRENT-STEP anchor mask coincide with the robot's measured root,
          while FIXED DOFs stay pinned to the reset-time terrain-cell anchor. Because the
          re-placement is per-DOF MASKED, terrain envs can be ego too: a stair clip
          re-places only its free lateral onto the robot, keeping downrange/height/yaw
          locked to the treads, and a box clip's mask varies per step. The teacher runs
          this at **every step** so tracking error is measured locally and planar drift is
          not double-counted (see ``termination_reference_world_state`` for the other half
          of that split).
        * **FM snap flag** (``_reanchor_active``, every ``cfg.fm_snap_every`` steps for
          ``_snap_mask`` envs) -- the recorder logs it and the FM dataset builds one
          training window per snap.

        These USED to share ``ego_anchor_every``, which was fine while re-placement was
        occasional. It is not fine at every step: the recorder would emit one window per
        env per step (12x the dataset, consecutive windows overlapping in all but one
        history row), and worse, the FM target would degenerate -- node 0 always coincides
        with the robot, so the window would encode no placement decision at all, which is
        the very thing the FM is being trained to predict.

        Idempotency across Isaac Lab's twice-per-step compute used to be incidental (same
        tick + same robot pose -> same result). At period 1 it is load-bearing on every
        step rather than 1 step in 12, so it is now enforced explicitly: the snapshot is
        taken at most once per ``common_step_counter``, matching the guard
        :meth:`FmTrackingCommand._snapshot_and_infer` already uses.
        """
        # Sit out the reset-time call and DO NOT stamp the guard, so the real re-anchor still
        # happens later this step with a correct ``_step``. Stamping here would keep the
        # clip base sampled from the dying episode's step count -- see _resample_command.
        if self._resampling:
            return
        step_counter = int(self.env.common_step_counter)
        if step_counter == self._last_reanchor_step:
            return
        self._last_reanchor_step = step_counter

        ego_period = max(int(self.cfg.ego_anchor_every), 1)
        snap_period = max(int(getattr(self.cfg, "fm_snap_every", ego_period)), 1)
        self._reanchor_active = self._snap_mask & (self.env.episode_length_buf % snap_period == 0)
        ego_due = self._ego_mask & (self.env.episode_length_buf % ego_period == 0)
        self._reanchor_clip_ids(torch.nonzero(ego_due, as_tuple=False).squeeze(-1))

    def _reanchor_clip_ids(self, ids: torch.Tensor) -> None:
        """Capture the re-anchor SNAPSHOT for env ``ids`` (the clip base + robot root right now).

        Shared by the cadence-based ego re-anchor (:meth:`_reanchor_ego_envs`) and any caller that
        re-anchors on a different schedule (e.g. the generator command re-anchoring on the FM's
        re-plan steps). This no longer writes a placement: a placement is a pure function of
        ``(terrain-cell anchor, mask, snapshot)``, and the mask varies ALONG the clip, so the
        placement is solved per frame from this snapshot in :meth:`_segment_placement` /
        :meth:`_window_placement`. Storing the snapshot rather than one solved transform is what
        keeps a clip's terrain-pinned frames welded to the terrain while its flat approach rides the
        robot. No-op for empty ``ids``.
        """
        if ids.numel() == 0:
            return
        # Clip-world base pose at each env's current step.
        y, _ = self.manager.gather_desired(self.clip_id[ids], self._step[ids])
        self._snap_base_pos[ids] = y[:, self._base_idx["pos"]]   # [k, 3]
        self._snap_base_quat[ids] = y[:, self._base_idx["ori"]]  # [k, 4] xyzw
        # Target = robot root link (planar position + heading + HEIGHT for FREE DOFs).
        self._snap_target_pos[ids] = wp.to_torch(self.robot.data.root_pos_w)[ids]
        self._snap_target_quat[ids] = wp.to_torch(self.robot.data.root_quat_w)[ids]   # xyzw

    def seed_reanchor_snapshot(
        self, env_ids: torch.Tensor, clip_id: torch.Tensor, step: torch.Tensor, spawn_pose: torch.Tensor,
    ) -> None:
        """Seed the re-anchor snapshot so the derived placement IS the terrain-cell anchor.

        Called from the clip reset with the pose it wrote to the sim. The robot spawns exactly on the
        cell-anchored reference, so this snapshot carries zero correction and every segment's
        placement resolves to the cell anchor -- reproducing the old reset behavior, where
        ``clip_world_pos`` was simply the reset placement.

        Args:
            env_ids: ``[k]`` env indices being reset.
            clip_id: ``[k]`` clip index per resetting env.
            step: ``[k]`` start step within the clip.
            spawn_pose: ``[k, 7]`` world spawn pose written to the sim ``[pos(3), quat(4) xyzw]``.
        """
        y, _ = self.manager.gather_desired(clip_id, step)
        self._snap_base_pos[env_ids] = y[:, self._base_idx["pos"]]
        self._snap_base_quat[env_ids] = y[:, self._base_idx["ori"]]
        self._snap_target_pos[env_ids] = spawn_pose[:, 0:3]
        self._snap_target_quat[env_ids] = spawn_pose[:, 3:7]

    def _segment_placement(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Placement ``([N, 3], [N, 4])`` for one anchor ``mask`` per env, from the stored snapshot."""
        return ego_placement_masked(
            self._snap_base_pos, self._snap_base_quat, self._snap_target_pos, self._snap_target_quat,
            anchor_mask=mask,
            cell_anchor_pos=self._reset_world_pos,
            cell_anchor_yaw=self._reset_world_yaw,
            mask_frame_yaw=self._mask_frame_yaw,
        )

    def _window_placement(
        self, clip_id: torch.Tensor, steps: torch.Tensor, env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-FRAME placement ``([N, K, 3], [N, K, 4])`` for the window ``steps``.

        Each frame is placed through ITS OWN segment's mask, so a window that reaches from the flat
        approach into the terrain-pinned interaction shows the interaction on the terrain rather than
        dragged along with the ego-anchored approach. This is the fix for the reference ribbon / the
        teacher's future-window observation / the recorder.

        Args:
            clip_id: ``[N, K]`` (or ``[N]``-broadcastable) clip index per window frame.
            steps: ``[N, K]`` step index per window frame.
            env_ids: optional ``[N]`` env subset ``clip_id`` / ``steps`` were gathered for; the
                per-env snapshot + cell anchor are subselected to match.
        """
        sl = slice(None) if env_ids is None else env_ids
        return ego_placement_masked_window(
            self._snap_base_pos[sl], self._snap_base_quat[sl],
            self._snap_target_pos[sl], self._snap_target_quat[sl],
            anchor_mask=self.manager.gather(self.manager.anchor_mask_t, clip_id, steps),
            cell_anchor_pos=self._reset_world_pos[sl],
            cell_anchor_yaw=self._reset_world_yaw[sl],
            mask_frame_yaw=self._mask_frame_yaw[sl],
        )

    def _refresh_current_placement(self) -> None:
        """Recompute ``clip_world_pos/quat`` for the CURRENT step's anchor mask.

        Runs every step, not just on the re-anchor cadence: the mask changes at segment boundaries,
        and the placement must follow it immediately. Previously the placement was only rewritten on
        the cadence tick, so for up to ``ego_anchor_every - 1`` frames after a boundary the terrain
        segment was still played through the approach's ego-drifted placement.
        """
        mask = self.manager.gather_anchor_mask_at_step(self.clip_id, self._step)
        self.clip_world_pos, self.clip_world_quat = self._segment_placement(mask)

    def _placement(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env world placement of the clip at the current step (its segment's mask)."""
        return self.clip_world_pos, self.clip_world_quat

    def _lift_ee_world(
        self, frame_names: list[str], anchor_pos: torch.Tensor, anchor_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Lift the clip-world ``y_des``/``dy_des`` channels of ``frame_names`` through an anchor.

        Reads the (clip-world) stored outputs, so this must run before
        :meth:`_write_world_des` overwrites them.  Returns ``[N, K, 13]`` per
        frame ``[pos_w(3), quat_w(4, xyzw), lin_w(3), ang_w(3)]``.
        """
        out = []
        for f in frame_names:
            idx = self._frame_output_indices(f)
            pos_w = lift_position(anchor_pos, anchor_quat, self.y_des[:, idx["pos"]])
            quat_w = lift_orientation(anchor_quat, self.y_des[:, idx["ori"]])
            lin_w = lift_vector(anchor_quat, self.dy_des[:, idx["lin_vel"]])
            ang_w = lift_vector(anchor_quat, self.dy_des[:, idx["ang_vel"]])
            out.append(torch.cat([pos_w, quat_w, lin_w, ang_w], dim=-1))
        return torch.stack(out, dim=1)

    def _clip_world_frames(self) -> torch.Tensor:
        """Lift the current clip-world outputs through the per-env placement.

        Sets ``ref_poses`` to the placement, then lifts the base + every ee
        reference frame from the stored clip-world frame into the absolute world
        frame.  Must run before :meth:`_write_world_des` (reads clip-world
        ``y_des``).

        Returns:
            ``frames [N, nf, 13]`` -- ``[base] + ee`` in ``self._wf_frames``
            order, each ``[pos_w(3), quat_w(4, xyzw), lin_w(3), ang_w(3)]``.
        """
        wpos, wquat = self._placement()
        self.ref_poses[:, :3] = wpos
        self.ref_poses[:, 3:7] = wquat
        base13 = lift_base_world_state(self.y_des, self.dy_des, self.ref_poses, self._base_idx)[:, :13]
        if len(self._wf_frames) > 1:
            ee = self._lift_ee_world(self._wf_frames[1:], wpos, wquat)  # [N, K, 13]
            return torch.cat([base13[:, None, :], ee], dim=1)          # [N, nf, 13]
        return base13[:, None, :]

    def _write_world_des(self, frames: torch.Tensor) -> None:
        """Scatter world-frame ``[N, nf, 13]`` frames into y_des/dy_des, and cache them.

        Caches the (final, source-selected) world frames so the post-step
        recorder API (:meth:`reference_world_state` /
        :meth:`reference_world_frames`) reads them back without re-lifting the
        now-overwritten ``y_des``.
        """
        n = self.num_envs
        self.y_des[:, self._wf_pos_flat] = frames[:, :, 0:3].reshape(n, -1)
        self.y_des[:, self._wf_ori_flat] = frames[:, :, 3:7].reshape(n, -1)
        self.dy_des[:, self._wf_lin_flat] = frames[:, :, 7:10].reshape(n, -1)
        self.dy_des[:, self._wf_ang_flat] = frames[:, :, 10:13].reshape(n, -1)
        self._wf_world_frames = frames
        # MUST precede the base rewrite below: the @root bodies are lifted through the
        # ANCHORED base, which is what makes their error local.
        self._lift_root_frames_to_world(frames[:, 0, :])
        self._write_global_base_des()

    def _write_global_base_des(self) -> None:
        """Re-point the BASE channels of ``y_des`` / ``dy_des`` at the GLOBAL clip.

        Splits the CLF's job in two, which is the whole point of tracking body links:

        * the BASE says WHERE the robot should be -- graded against the globally-placed
          clip, which never moves;
        * the body links and end effectors say WHAT SHAPE it should be in -- graded against
          the re-datumed clip, so limb tracking is not punished for planar drift.

        Without this the base term does nothing at all.  Every-step anchoring re-datums the
        clip onto the robot in the mask's FREE root DOFs, so the base error in those DOFs is
        ~0 by construction -- measured at 0.3-1.9% of V.  Yet
        :func:`base_pose_error_termination` ends the episode on exactly that drift, so the
        policy was being killed for a quantity no reward mentioned.  Now the reward and the
        termination read the same reference.

        On stair / box tiles the mask already pins downrange / height / yaw to the terrain
        cell, so there the global and anchored bases coincide and this is a no-op; on flat
        it is the only thing grading global position and heading.

        Deliberately does NOT touch ``_wf_world_frames``: that cache is the ANCHORED
        placement, and the recorder / FM window / debug viz read it back through
        :meth:`reference_world_state`.  ``y_des`` being mutated past the cache is the
        pre-existing contract here (see :meth:`_write_world_des`), not a new one.

        FRAME.  The twist goes global with the position, because both come from the one
        :func:`lift_base_world_state` call -- so it is world axes rotated by the GLOBAL
        placement quaternion, not the anchored one.  That is required, not incidental: the
        CLF pairs each position channel with its velocity channel as a double integrator,
        and the ``2 * P12 * e * edot`` cross term only means anything if ``edot = d/dt e`` in
        the SAME frame.  Consequence on flat, where the mask frees yaw: a heading error now
        rotates the desired velocity vector (measured 0.046 rad mean, 0.67 rad max), so
        heading error feeds the velocity term instead of being absorbed by the frame -- which
        is the same reason the position term is global.  Exactly zero on stair / box tiles,
        where yaw is pinned and the two placements are the same rotation.

        Everything here stays in WORLD AXES on purpose.  Expressing the tracked links'
        twists base-RELATIVE instead would drop the ``omega x r`` transport term, which
        leaks base angular-velocity error into every link's linear-velocity error scaled by
        its lever arm -- genuinely unwanted.  But it would also stop grading WHERE the limbs
        are in space, and with the tracked set deduped to pelvis + the 13 ``@root`` frames
        that would leave foot placement against the stair treads graded only through the
        base pose and the anchor mask.  Terrain needs that signal, so the transport coupling
        is accepted deliberately.  (Note rotating the error into the base frame does NOT
        remove it -- rotation changes axes, and the transport term rotates with them.)
        """
        bi = self._base_idx
        state = self.termination_reference_world_state()      # independent of y_des
        self.y_des[:, bi["pos"]] = state[:, 0:3]
        self.y_des[:, bi["ori"]] = state[:, 3:7]
        self.dy_des[:, bi["lin_vel"]] = state[:, 7:10]
        self.dy_des[:, bi["ang_vel"]] = state[:, 10:13]

    def _root_frame_maps(self) -> tuple[torch.Tensor, ...] | None:
        """Flat ``y_des`` / ``dy_des`` index maps for the ``@root`` body block (cached).

        Returns ``(pos, ori, lin, ang)`` index tensors, or ``None`` when the clip library
        carries no root-frame bodies.
        """
        maps = getattr(self, "_rf_maps", "unset")
        if maps != "unset":
            return maps
        frames = list(self.root_body_frames or ())
        if not frames:
            self._rf_maps = None
            return None
        pos_names, vel_names = self.ordered_pos_output_names, self.ordered_vel_output_names
        dev = self.device

        def _idx(names, frames_, axes, kind):
            return torch.tensor([names.index(f"{f}:{kind}_{a}") for f in frames_ for a in axes],
                                dtype=torch.long, device=dev)

        self._rf_maps = (
            _idx(pos_names, frames, "xyz", "pos"),
            _idx(pos_names, frames, ("x", "y", "z", "w"), "ori"),
            _idx(vel_names, frames, "xyz", "pos"),
            _idx(vel_names, frames, "xyz", "ori"),
        )
        return self._rf_maps

    def _lift_root_frames_to_world(self, base13: torch.Tensor) -> None:
        """Lift the stored pelvis-relative ``@root`` desired channels into the world frame.

        The clip stores each tracked link's pose+twist relative to the CLIP's pelvis (raw
        FK, Jacobian-exact velocities), which is what makes those channels immune to slope
        rotation, world baking and re-anchoring.  The CLF compares in world, so they are
        lifted here through the clip's own -- already anchored -- base pose:

            p_w = p0 + R p_r
            q_w = q0 * q_r
            v_w = v0 + w0 x (R p_r) + R v_r
            w_w = w0 + R w_r

        Because ``base13`` is the anchored placement's base, this yields "the clip's FK in
        world with its root placed on the robot in the mask's free DOFs" -- the
        subtract-the-planar-root-error-and-rotate-by-yaw comparison, with no explicit
        subtraction step.

        Args:
            base13: ``[N, 13]`` world base row ``[pos(3), quat(4, xyzw), lin(3), ang(3)]``.
        """
        maps = self._root_frame_maps()
        if maps is None:
            return
        pos_i, ori_i, lin_i, ang_i = maps
        n = self.num_envs
        b = len(self.root_body_frames)

        p0, q0, v0, w0 = base13[:, 0:3], base13[:, 3:7], base13[:, 7:10], base13[:, 10:13]
        qf = q0[:, None, :].expand(n, b, 4).reshape(n * b, 4)

        p_r = self.y_des[:, pos_i].view(n, b, 3)
        q_r = self.y_des[:, ori_i].view(n, b, 4)
        v_r = self.dy_des[:, lin_i].view(n, b, 3)
        w_r = self.dy_des[:, ang_i].view(n, b, 3)

        off = quat_apply(qf, p_r.reshape(n * b, 3)).view(n, b, 3)      # R p_r
        self.y_des[:, pos_i] = (p0[:, None, :] + off).reshape(n, -1)
        self.y_des[:, ori_i] = quat_mul(qf, q_r.reshape(n * b, 4)).view(n, -1)
        self.dy_des[:, lin_i] = (
            v0[:, None, :]
            + torch.linalg.cross(w0[:, None, :].expand_as(off), off)
            + quat_apply(qf, v_r.reshape(n * b, 3)).view(n, b, 3)
        ).reshape(n, -1)
        self.dy_des[:, ang_i] = (
            w0[:, None, :] + quat_apply(qf, w_r.reshape(n * b, 3)).view(n, b, 3)
        ).reshape(n, -1)

    def get_desired_outputs(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> None:
        """Gather the (clip-world) reference outputs for the current step."""
        ids = env_ids if env_ids is not None else slice(None)
        y, dy = self.manager.gather_desired(self.clip_id[ids], self._step[ids])
        self.y_des[ids] = y
        self.dy_des[ids] = dy

    def get_contact_state(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the clip's expected contact schedule in contact_bodies order."""
        stored = self.manager.gather_contact(self.clip_id, self._step)  # [N, C_clip]
        out = torch.zeros(self.num_envs, len(self.contact_bodies), device=self.device)
        have = self._contact_col >= 0
        cols = self._contact_col[have]
        out[:, have] = stored[:, cols].float()
        if env_ids is not None:
            return out[env_ids]
        return out

    # ------------------------------------------------------------------
    # Commanded velocity (upstream command that generated the clip)
    # ------------------------------------------------------------------

    @property
    def command_velocity_available(self) -> bool:
        """True if the clip library carries a per-step commanded velocity (``target_vel``)."""
        return self.manager.has_target_vel()

    def get_command_velocity(self) -> torch.Tensor:
        """Per-env commanded base velocity ``[N, 3]`` = (vx, vy, wz) at the current step.

        This is the *command* that generated the clip (library ``target_vel``), not
        the achieved reference velocity (``dy_des``).  It is the signal to condition
        a future-horizon flow-matching model on -- upstream of the reference and
        available at deployment.  Recorded by ``scripts/rsl_rl/record_data.py``.
        """
        if not self.command_velocity_available:
            raise RuntimeError(
                "Clip library has no target_vel; regenerate clips with a generator that writes it."
            )
        return self.manager.gather_target_vel(self.clip_id, self._step)

    def get_command_velocity_horizon(self, num_future: int) -> torch.Tensor:
        """Commanded velocity over the next ``num_future`` steps, ``[N, num_future, 3]``.

        Covers steps ``s+1 .. s+num_future`` (clamped to the clip end), matching the
        flow-matching future window whose history ends at the current step ``s``.
        Used at inference to condition the decoder on the command horizon.
        """
        if not self.command_velocity_available:
            raise RuntimeError(
                "Clip library has no target_vel; regenerate clips with a generator that writes it."
            )
        T = self.manager.num_steps
        offs = torch.arange(num_future, device=self.device)
        steps = (self._step[:, None] + 1 + offs[None, :]).clamp(max=T - 1)  # [N, num_future]
        cid = self.clip_id[:, None].expand_as(steps)
        return self.manager.gather(self.manager.lib.target_vel, cid, steps)

    # ------------------------------------------------------------------
    # Future window (look-ahead into the clip for the teacher policy)
    # ------------------------------------------------------------------

    def _future_step_indices(self, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(clip_id, step)`` index grids ``[N, K]`` for the given future ``offsets``.

        Steps are ``self._step + offset`` clamped to the clip end, so the window
        saturates on the final frame near a clip's end (same convention as
        :meth:`get_command_velocity_horizon` and the debug-viz ribbon).
        """
        T = self.manager.num_steps
        steps = (self._step[:, None] + offsets[None, :]).clamp(max=T - 1)  # [N, K]
        cid = self.clip_id[:, None].expand_as(steps)
        return cid, steps

    def get_future_command_velocity(self, offsets: torch.Tensor) -> torch.Tensor:
        """Commanded base velocity ``[N, K, 3]`` = (vx, vy, wz) at the future ``offsets``.

        The upstream ``target_vel`` (heading frame, deployable) at steps
        ``self._step + offsets`` -- the look-ahead counterpart of
        :meth:`get_command_velocity`.
        """
        if not self.command_velocity_available:
            raise RuntimeError(
                "Clip library has no target_vel; regenerate clips with a generator that writes it."
            )
        cid, steps = self._future_step_indices(offsets)
        return self.manager.gather(self.manager.lib.target_vel, cid, steps)

    def get_future_world_base(
        self, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame reference base pose+velocity over the future ``offsets``.

        Returns ``(pos_w, quat_w, lin_w, ang_w)`` of shapes ``[N, K, 3]``,
        ``[N, K, 4]`` (xyzw), ``[N, K, 3]``, ``[N, K, 3]``.  Lifts the clip-world
        base channels through the per-env placement (a single yaw+xy transform),
        so ``quat_w`` matches the convention consumed by the ``base_pose_error``
        observation and the debug-viz ribbon.
        """
        cid, steps = self._future_step_indices(offsets)
        bi = self._base_idx
        y = self.manager.gather(self.manager.lib.y_des, cid, steps)    # [N, K, P] clip-world
        dy = self.manager.gather(self.manager.lib.dy_des, cid, steps)  # [N, K, V]
        wpos, wquat = self._window_placement(cid, steps)               # per-FRAME placement
        pos_w = lift_position_batched(wpos, wquat, y[:, :, bi["pos"]])
        quat_w = lift_orientation_batched(wquat, y[:, :, bi["ori"]])
        lin_w = lift_vector_batched(wquat, dy[:, :, bi["lin_vel"]])
        ang_w = lift_vector_batched(wquat, dy[:, :, bi["ang_vel"]])
        return pos_w, quat_w, lin_w, ang_w

    def get_future_joint_state(
        self, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip joint pos/vel ``([N, K, nj], [N, K, nj])`` at the future ``offsets``.

        Joints are frame-invariant (no lift), so this gathers the clip-library
        ``y_des``/``dy_des`` joint channels at steps ``self._step + offsets`` -- the
        look-ahead counterpart of the joint block in :meth:`reference_world_state`.
        Used by the data recorder to capture the ego-anchored clip future target.
        """
        cid, steps = self._future_step_indices(offsets)
        y = self.manager.gather(self.manager.lib.y_des, cid, steps)    # [N, K, P]
        dy = self.manager.gather(self.manager.lib.dy_des, cid, steps)  # [N, K, V]
        joint_pos = y[:, :, self._base_idx["joint_pos"]]
        joint_vel = dy[:, :, self._base_idx["joint_vel"]]
        return joint_pos, joint_vel

    def get_future_world_frames(
        self, frame_names: list[str], offsets: torch.Tensor
    ) -> torch.Tensor:
        """World-frame reference for extra (ee) frames over the future ``offsets``.

        The look-ahead counterpart of :meth:`reference_world_frames`: lifts each
        frame's clip-library ``y_des``/``dy_des`` channels at steps ``self._step +
        offsets`` through the per-env placement (``clip_world_pos``/``clip_world_quat``),
        exactly like :meth:`get_future_world_base`. At ``offset == 0`` this matches the
        current-step ego-anchored frames. Used by the data recorder.

        Args:
            frame_names: Output ee-frame names (each in ``self._wf_frames`` minus base).
            offsets: Future step offsets ``[K]``.

        Returns:
            ``[N, K, len(frame_names), 13]`` = ``[pos_w(3), quat_w(4 xyzw), lin_w(3),
            ang_w(3)]`` per frame, in the given order.
        """
        cid, steps = self._future_step_indices(offsets)
        y = self.manager.gather(self.manager.lib.y_des, cid, steps)    # [N, K, P]
        dy = self.manager.gather(self.manager.lib.dy_des, cid, steps)  # [N, K, V]
        wpos, wquat = self._window_placement(cid, steps)               # per-FRAME placement
        out = []
        for f in frame_names:
            fi = self._frame_output_indices(f)
            out.append(torch.cat([
                lift_position_batched(wpos, wquat, y[:, :, fi["pos"]]),
                lift_orientation_batched(wquat, y[:, :, fi["ori"]]),
                lift_vector_batched(wquat, dy[:, :, fi["lin_vel"]]),
                lift_vector_batched(wquat, dy[:, :, fi["ang_vel"]]),
            ], dim=-1))
        return torch.stack(out, dim=2)  # [N, K, F, 13]

    def reference_window_obs(self, offsets: torch.Tensor) -> torch.Tensor:
        """The reference window over ``offsets``, in the observation's representation.

        The producer behind ``mdp.reference_window``.  Lives on the command rather than in
        the observation function because the flow-matching generator predicts this
        representation *directly*: :class:`~.fm_tracking_cmd.FmTrackingCommand` overrides
        this to serve its prediction with no world round-trip, in simulation and at deploy
        alike.  Here -- the clip reference -- the world-frame look-ahead is what exists, so
        it goes through :func:`~robot_rl.flow_matching.window.world_to_window`.

        Overriding ONE accessor rather than branching inside the observation also keeps the
        clip/generator swap honest: ``as_clip_reference()`` / ``as_fm_reference()`` flip
        ``_serving()``, and every look-ahead consumer follows in step.

        Args:
            offsets: Future step offsets ``[K]`` (``0`` = the current step).

        Returns:
            ``[N, K, D]`` window nodes, ``D = window_dim(nj, num_ee_frames)``.
        """
        pos_w, quat_w, lin_w, ang_w = self.get_future_world_base(offsets)   # [N,K,·]
        joint_pos, joint_vel = self.get_future_joint_state(offsets)         # [N,K,nj]
        ee_w = self.get_future_world_frames(self.resolve_ee_frames(None), offsets)  # [N,K,E,13]
        root_pos_w = wp.to_torch(self.robot.data.root_pos_w)[:, None, :]    # [N,1,3]
        root_quat_w = wp.to_torch(self.robot.data.root_quat_w)[:, None, :]  # [N,1,4] xyzw
        return world_to_window(
            root_pos_w, root_quat_w, pos_w, quat_w, lin_w, ang_w,
            joint_pos, joint_vel, ee_w[..., 0:3],
        )

    # ------------------------------------------------------------------
    # World-frame reference (for reset placement, obs, termination)
    # ------------------------------------------------------------------

    def reference_world_state(self) -> torch.Tensor:
        """World-frame reference state ``[N, 3+4+3+3+2*nj]`` at the current step.

        ``[base_pos_w, base_quat_w(xyzw), base_lin_w, base_ang_w, joint_pos,
        joint_vel]`` -- served from the per-step world cache populated by
        :meth:`_write_world_des` (base row) plus the frame-invariant joint
        channels.  Used by the reset event, observations, and terminations.
        """
        joint_pos = self.y_des[:, self._base_idx["joint_pos"]]
        joint_vel = self.dy_des[:, self._base_idx["joint_vel"]]
        return torch.cat([self._wf_world_frames[:, 0, :], joint_pos, joint_vel], dim=-1)

    def termination_reference_world_state(self) -> torch.Tensor:
        """World-frame reference state used by the pose-drift termination.

        The **globally-placed** clip -- the reset-time terrain-cell anchor -- NOT the
        per-step re-anchored reference that drives the reward.  The two must differ, and
        the split is the whole point of anchoring:

        * the REWARD grades local error, with the clip's free root DOFs re-datumed onto the
          robot every step, so tracking the motion's shape is not punished for planar drift;
        * TERMINATION grades that drift, against a clip that never moves.

        Without this the termination silently dies under every-step anchoring: the free
        DOFs coincide with the robot by construction, so the position error collapses to
        ``|dz|`` -- and to identically zero on a pre-mask library, whose legacy mask frees
        height too.  A robot could then stand still while the clip marched away and never
        terminate.

        ``_reset_world_pos`` / ``_reset_world_yaw`` are written once per episode by
        ``_commit_clip_reset`` and are never touched by re-anchoring, so the global
        placement is exactly recoverable at any step.

        Returns:
            ``[N, 3+4+3+3+2*nj]`` world reference state, same layout as
            :meth:`reference_world_state`.
        """
        yaw = self._reset_world_yaw
        world_quat = quat_from_angle_axis(yaw, self._world_z.expand(yaw.shape[0], 3))
        return self.reference_world_state_at(
            self.clip_id, self._step, self._reset_world_pos, world_quat
        )

    def _frame_output_indices(self, frame_name: str) -> dict[str, torch.Tensor]:
        """Resolve (and cache) the y_des/dy_des indices for one output frame.

        Returns long-tensor index sets keyed by ``pos`` (3), ``ori`` (4, xyzw),
        ``lin_vel`` (3), ``ang_vel`` (3) for ``frame_name``.
        """
        cache = getattr(self, "_frame_idx_cache", None)
        if cache is None:
            cache = {}
            self._frame_idx_cache = cache
        if frame_name in cache:
            return cache[frame_name]
        pos_names = self.ordered_pos_output_names
        vel_names = self.ordered_vel_output_names
        dev = self.device
        entry = {
            "pos": torch.tensor(
                [pos_names.index(f"{frame_name}:pos_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            ),
            "ori": torch.tensor(
                [pos_names.index(f"{frame_name}:ori_{a}") for a in ("x", "y", "z", "w")],
                dtype=torch.long, device=dev,
            ),
            "lin_vel": torch.tensor(
                [vel_names.index(f"{frame_name}:pos_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            ),
            "ang_vel": torch.tensor(
                [vel_names.index(f"{frame_name}:ori_{a}") for a in ("x", "y", "z")],
                dtype=torch.long, device=dev,
            ),
        }
        cache[frame_name] = entry
        return entry

    def reference_world_frames(self, frame_names: list[str]) -> torch.Tensor:
        """World-frame reference state for extra output frames (e.g. feet/hands).

        Served from the per-step world cache populated by
        :meth:`_write_world_des` -- so it reflects the selected source (clip or
        generator) and never re-lifts the world-overwritten ``y_des``.  Used by
        the data recorder to record ee reference states.

        Args:
            frame_names: Output frame names; each must be one of the command's
                world-frame ee frames (``self._wf_frames`` minus the base).

        Returns:
            ``[N, K, 13]`` per frame ``[pos_w(3), quat_w(4, xyzw), lin_vel_w(3),
            ang_vel_w(3)]`` in the given frame order.
        """
        idxs = [self._wf_frame_index[f] for f in frame_names]
        return self._wf_world_frames[:, idxs, :]

    def get_reference_world_state(self, base_frame_name: str) -> torch.Tensor:
        """World-frame reference state, matching the base-class recording API.

        ``scripts/rsl_rl/record_data.py`` auto-detects the trajectory command via
        this method name and calls it with ``--ref_base_frame``.  The inherited
        :meth:`BaseTrajectoryCommand.get_reference_world_state` re-resolves the
        orientation indices in name-appearance order, which is wrong for the
        clip library (it stores ``ori`` scalar-first); this override delegates to
        the xyzw-correct :meth:`reference_world_state` instead.

        Args:
            base_frame_name: Trajectory output frame used as the robot root; must
                match ``cfg.base_frame_name`` (the only base frame this clip-based
                command is wired for).

        Returns:
            World-frame reference state, shape ``[N, 3 + 4 + 3 + 3 + 2*nj]``.
        """
        if base_frame_name != self.cfg.base_frame_name:
            raise ValueError(
                f"MotionSequenceCommand is wired for base frame "
                f"'{self.cfg.base_frame_name}'; got '{base_frame_name}'."
            )
        return self.reference_world_state()

    def reference_world_state_at(
        self, clip_id: torch.Tensor, step: torch.Tensor, world_pos: torch.Tensor, world_quat: torch.Tensor,
    ) -> torch.Tensor:
        """World reference state for an arbitrary (clip, step, placement).

        Used by the reset event to compute spawn poses before the per-step
        buffers are populated.

        Args:
            clip_id: ``[K]`` clip indices.
            step: ``[K]`` step indices.
            world_pos: ``[K, 3]`` clip-placement world positions.
            world_quat: ``[K, 4]`` clip-placement world quaternions (xyzw).

        Returns:
            ``[K, 3+4+3+3+2*nj]`` world reference state.
        """
        y, dy = self.manager.gather_desired(clip_id, step)
        placement = torch.cat([world_pos, world_quat], dim=-1)
        return lift_base_world_state(y, dy, placement, self._base_idx)

    # ------------------------------------------------------------------
    # Symmetry helpers (end-effector reference)
    # ------------------------------------------------------------------

    def resolve_ee_frames(self, frame_names: list[str] | None = None) -> list[str]:
        """Resolve the end-effector frame list/order for the ``ee_ref`` obs.

        Single source of truth shared by :func:`mdp.ee_ref_relative_to_base`
        and its symmetry reflector (:meth:`get_symmetric_ee`) so the two can
        never drift.  When ``frame_names`` is ``None``, returns every reference
        output frame that exposes a position, except the base frame, sorted
        alphabetically (matching the obs default).

        Root-frame (``@root``) frames are excluded: this is the WORLD end-effector
        set.  Keeping it stable as the clip layout widened is load-bearing -- it
        sizes the ``ee_ref`` observation, drives ``_build_world_frame_maps`` (so a
        root-frame body would otherwise be pointlessly lifted through the placement
        every step), and is what :class:`FmTrackingCommand` checks an FM checkpoint's
        ``frame_names`` against, so growing it would invalidate every existing
        checkpoint.

        Args:
            frame_names: Explicit frame list, or ``None`` for the default.

        Returns:
            Ordered world end-effector frame names.
        """
        if frame_names is not None:
            return list(frame_names)
        base = self.cfg.base_frame_name
        return sorted(
            {n.split(":", 1)[0] for n in self.ordered_pos_output_names
             if ":pos_" in n and not is_root_frame(n)} - {base}
        )

    def get_symmetric_ee(
        self, ee: torch.Tensor, frame_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Return the left-right symmetric reflection of base-relative ee positions.

        :func:`mdp.ee_ref_relative_to_base` stacks each frame's xyz position
        (expressed in the reference base frame).  Under sagittal reflection each
        left/right frame swaps and each frame's y component negates -- a
        :class:`NamedReflector` over ``{frame}:pos_{x,y,z}`` names does both.
        Reflectors are cached per resolved frame tuple.

        Args:
            ee: ``[N, 3*F]`` or ``[N, history_length * 3*F]`` ee-position tensor.
            frame_names: Same frame list the obs used (``None`` -> default).

        Returns:
            Symmetric ee positions of the same shape.
        """
        frames = self.resolve_ee_frames(frame_names)
        key = tuple(frames)
        cache = getattr(self, "_ee_reflector_cache", None)
        if cache is None:
            cache = {}
            self._ee_reflector_cache = cache
        reflector = cache.get(key)
        if reflector is None:
            names = [f"{f}:pos_{a}" for f in frames for a in ("x", "y", "z")]
            reflector = NamedReflector(SagittalReflectionConfig(), names, self.device)
            cache[key] = reflector
        return reflector.reflect_with_history(ee)

    # ------------------------------------------------------------------
    # Debug visualization (standard CommandTerm debug_vis hook)
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Toggle the reference-trajectory marker visualizers.

        Four things are drawn, and the point is the RELATIVE motion between them -- each
        pair isolates one frame the system uses:

        * **Clip ribbon** (per-frame colours) -- the clip at its GLOBAL placement, welded to
          the terrain cell. Where the motion was authored.
        * **Obs-window ribbon** (yellow) -- the base of the reference window the POLICY
          sees, which is anchored and therefore rides the robot. Its separation from the
          clip ribbon IS the accumulated drift: forgiven by the reward, fired on by the
          termination.
        * **Body spheres** -- measured (green) vs the reference the reward grades against
          (magenta). The gap in each pair is the reward's per-body error.

        So: green-vs-magenta = what the reward sees; yellow-vs-clip = what the termination
        sees. Per-step update is in :meth:`_debug_vis_callback`.
        """
        if debug_vis:
            if self._debug_markers is None and hasattr(self, "_debug_frame_pos_idx"):
                self._debug_markers = self._create_debug_markers()
            if self._debug_body_markers is None and self.cfg.debug_vis_body_spheres:
                self._debug_body_markers = self._create_debug_body_markers()
            if self._debug_markers is not None:
                self._debug_markers.set_visibility(True)
            else:
                self._debug_vis_pending = True
            if self._debug_body_markers is not None:
                self._debug_body_markers.set_visibility(True)
        else:
            self._debug_vis_pending = False
            for m in (self._debug_markers, self._debug_body_markers):
                if m is not None:
                    m.set_visibility(False)

    def _create_debug_markers(self):
        """Build a :class:`VisualizationMarkers` with one cylinder prototype per frame.

        Mirrors the online command's marker style: the reference ribbon is
        drawn as thin cylinder *segments* between consecutive sampled points
        (oriented + z-scaled per segment in :meth:`_debug_vis_callback`),
        rather than a point cloud of spheres.  The native cylinder height is
        ``1.0`` so the per-segment z-scale equals the segment length.
        """
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN MotionSequenceCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_marker_radius)
        prototypes = {
            name: sim_utils.CylinderCfg(
                radius=radius,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
            for name, color in zip(self._debug_frame_names, self._debug_frame_colors)
        }
        # Extra prototype, always last, for the policy's observed reference-window base.
        prototypes["obs_window_base"] = sim_utils.CylinderCfg(
            radius=radius * 1.5,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_DEBUG_OBS_WINDOW_COLOR),
        )
        cfg = VisualizationMarkersCfg(prim_path=_DEBUG_VIZ_PRIM_PATH, markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN MotionSequenceCommand] could not create VisualizationMarkers: {exc}")
            return None

    def _create_debug_body_markers(self):
        """Build the tracked-body sphere markers: prototype 0 = measured, 1 = reference."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN MotionSequenceCommand] could not import VisualizationMarkers: {exc}")
            return None
        r = float(self.cfg.debug_vis_body_sphere_radius)
        prototypes = {
            name: sim_utils.SphereCfg(
                radius=r,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
            for name, color in (("actual", _DEBUG_ACTUAL_COLOR), ("rewarded", _DEBUG_REWARDED_COLOR))
        }
        cfg = VisualizationMarkersCfg(prim_path=_DEBUG_BODY_VIZ_PRIM_PATH, markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN MotionSequenceCommand] could not create body markers: {exc}")
            return None

    def tracked_frame_pose(self, frame_names: list[str]):
        """Measured and reference pose of named frames, in the frame the REWARD uses.

        Reads ``y_act`` / ``y_des`` directly, so the pair is exactly what the CLF
        differences: world positions with the clip already re-datumed onto the robot in the
        anchor mask's free DOFs.  A consumer therefore measures LOCAL tracking error --
        planar drift has already been removed, and is the base termination's job instead.

        Prefers a frame's ``@root`` channels when the library carries them (that is the copy
        body mode grades, and its velocities are Jacobian-exact), falling back to the world
        channels otherwise, so this works in both tracking modes.

        Args:
            frame_names: Bare body names, e.g. ``["left_ankle_roll_link"]``.

        Returns:
            ``(act_pos, act_quat, des_pos, des_quat)``, shapes ``[N, F, 3]`` / ``[N, F, 4]``
            (quaternions xyzw).

        Raises:
            KeyError: If a requested frame has no channels at all.
        """
        cache = getattr(self, "_tf_pose_idx", None)
        if cache is None:
            cache = self._tf_pose_idx = {}
        key = tuple(frame_names)
        if key not in cache:
            names = self.ordered_pos_output_names
            pos_i, ori_i = [], []
            for f in frame_names:
                stem = root_frame_name(f) if f"{root_frame_name(f)}:pos_x" in names else f
                if f"{stem}:pos_x" not in names:
                    raise KeyError(
                        f"Frame '{f}' has no tracked channels (looked for '{f}' and "
                        f"'{root_frame_name(f)}'). Available frames: "
                        f"{sorted({n.split(':', 1)[0] for n in names if ':pos_' in n})}"
                    )
                pos_i += [names.index(f"{stem}:pos_{a}") for a in "xyz"]
                ori_i += [names.index(f"{stem}:ori_{a}") for a in ("x", "y", "z", "w")]
            cache[key] = (
                torch.tensor(pos_i, dtype=torch.long, device=self.device),
                torch.tensor(ori_i, dtype=torch.long, device=self.device),
            )
        pos_i, ori_i = cache[key]
        f = len(frame_names)
        return (
            self.y_act[:, pos_i].view(-1, f, 3), self.y_act[:, ori_i].view(-1, f, 4),
            self.y_des[:, pos_i].view(-1, f, 3), self.y_des[:, ori_i].view(-1, f, 4),
        )

    def _tracked_body_pos_idx(self) -> torch.Tensor | None:
        """``[B, 3]`` ``y_des``/``y_act`` column indices of every TRACKED frame's position.

        Derived from ``tracked_vel_names`` rather than a restated body list, so the spheres
        follow ``tracking_mode`` automatically: joint mode shows the world skeleton, body
        mode the 13 links.  Cached.
        """
        cached = getattr(self, "_tb_idx", "unset")
        if cached != "unset":
            return cached
        frames: list[str] = []
        for n in self.tracked_vel_names:
            if ":pos_" not in n:
                continue
            f = n.split(":", 1)[0]
            if f not in frames:
                frames.append(f)
        pos_names = self.ordered_pos_output_names
        self._tb_idx = (
            torch.tensor([[pos_names.index(f"{f}:pos_{a}") for a in "xyz"] for f in frames],
                         dtype=torch.long, device=self.device)
            if frames else None
        )
        return self._tb_idx

    def obs_window_offsets(self) -> torch.Tensor | None:
        """Step offsets of the ``reference_window`` observation, read from the obs manager.

        Read from the live term rather than restated in a cfg, so no consumer can drift from
        the window the policy is actually fed.  ``None`` if no such term exists.

        Two consumers: the debug ribbon, and ``mdp.clip_end_reached``, which truncates this
        far before the clip's last frame so the observed window never saturates on it.
        """
        cached = getattr(self, "_obs_win_offsets", "unset")
        if cached != "unset":
            return cached
        self._obs_win_offsets = None
        try:
            om = self.env.observation_manager
            for group, terms in om.active_terms.items():
                if "reference_window" in terms:
                    p = om._group_obs_term_cfgs[group][list(terms).index("reference_window")].params
                    ns = max(1, int(p.get("num_steps", 11)))
                    st = max(1, int(p.get("stride", 5)))
                    self._obs_win_offsets = torch.arange(ns, device=self.device, dtype=torch.long) * st
                    break
        except Exception as exc:
            print(f"[WARN MotionSequenceCommand] could not read reference_window params: {exc}")
        return self._obs_win_offsets

    def _clip_ribbon_env_mask(self) -> torch.Tensor | None:
        """Per-env bool mask of which envs draw the clip ribbon (``None`` => all).

        Subclasses override this to suppress the clip ribbon for envs tracking a
        different reference (e.g. the flow-matching generator). Returning
        ``None`` keeps the default behavior (draw every env).
        """
        return None

    def _scene_alive(self) -> bool:
        """Whether the env still has a scene, i.e. whether it is safe to draw.

        The debug-vis callback is an Omniverse event subscription, and it fires once more
        while the app shuts down -- AFTER ``env.close()`` has dropped ``env.scene``. Anything
        in the draw path that reaches back through the env then raises, and because the
        exception surfaces inside a ``weakref.proxy`` lambda the manager only prints it. The
        run still exits 0, so it is pure noise -- but a full traceback on every play run is
        exactly what hides a real one later.

        ``ManagerBase`` exposes ``num_envs`` and ``device`` as properties that dereference
        ``self._env``, so this cannot be replaced by caching one value; the guard has to sit
        in front of the draw.

        Returns:
            True while the scene exists, False once it has been torn down.
        """
        return getattr(self._env, "scene", None) is not None

    def _debug_vis_callback(self, event) -> None:
        """Draw the upcoming reference ribbon as cylinder segments per env.

        For each reference frame, the forward window's world positions are
        connected by one thin cylinder per consecutive pair (the online
        command's marker style): each cylinder is centred on the segment
        midpoint, oriented so its local +z lies along the segment, and
        z-scaled to the segment length.  The prototype index selects the
        frame's colour.
        """
        if not self._scene_alive():
            return
        if self._debug_markers is None:
            if not self._debug_vis_pending:
                return
            self._debug_markers = self._create_debug_markers()
            if self._debug_markers is None:
                return
            self._debug_markers.set_visibility(True)
            self._debug_vis_pending = False

        # Optionally restrict the CLIP ribbon to a subset of envs (e.g. drop envs that are
        # tracking a different reference). Subselect every per-env quantity so the batched
        # lift stays aligned. An empty selection suppresses the clip cylinders ONLY -- the
        # observed-window ribbon and the tracked-body spheres below are about what the policy
        # sees and what the reward grades, which exist whatever the reference source is.
        mask = self._clip_ribbon_env_mask()
        env_ids = None if mask is None else torch.nonzero(mask, as_tuple=False).squeeze(-1)
        sub = slice(None) if env_ids is None else env_ids
        step, clip_id = self._step[sub], self.clip_id[sub]
        self._debug_markers.set_visibility(True)

        n = step.shape[0]
        T = self.manager.num_steps
        F = len(self._debug_frame_pos_idx)
        if F == 0:
            return
        steps = (step[:, None] + self._debug_offsets[None, :]).clamp(max=T - 1)          # [n, Ns]
        ns = steps.shape[1]
        draw_clip = n > 0 and ns >= 2
        if not draw_clip:
            self._draw_obs_and_bodies(F, None)
            return
        cid = clip_id[:, None].expand(n, ns)                                            # [n, Ns]

        # Per-frame clip-world positions over the window, lifted to sim-world by that FRAME's
        # segment placement -- so the ribbon draws a terrain-pinned segment on its terrain even while
        # the robot is still on the ego-anchored approach (it used to drag the whole window along
        # with the approach, which is what made the reference look detached from the stairs / box).
        frame_idx = torch.stack(self._debug_frame_pos_idx, dim=0)                        # [F, 3]
        rel = self.manager.gather(self.manager.lib.y_des, cid, steps)[:, :, frame_idx]    # [n, Ns, F, 3] clip-world
        # GLOBAL placement (the reset-time terrain-cell anchor), NOT the anchored one. Under
        # per-step anchoring the anchored ribbon sits on the robot by construction and shows
        # nothing; the global ribbon is where the motion was authored, so the gap to the
        # yellow obs-window ribbon below reads as the accumulated drift.
        sl = slice(None) if env_ids is None else env_ids
        gpos = self._reset_world_pos[sl][:, None, :].expand(n, ns, 3)
        gyaw = self._reset_world_yaw[sl]
        gquat = quat_from_angle_axis(
            gyaw, self._world_z.expand(gyaw.shape[0], 3)
        )[:, None, :].expand(n, ns, 4)
        wpos, wquat = gpos, gquat
        # One placement per (frame), shared by that frame's F reference bodies.
        wpos_f = wpos.unsqueeze(2).expand(n, ns, F, 3).reshape(n, ns * F, 3)
        wquat_f = wquat.unsqueeze(2).expand(n, ns, F, 4).reshape(n, ns * F, 4)
        world_pos = lift_position_batched(
            wpos_f, wquat_f, rel.reshape(n, ns * F, 3),
        ).reshape(n, ns, F, 3)

        # Per-segment midpoint, direction, length (one cylinder per segment per frame).
        starts = world_pos[:, :-1]                                                       # [n, Ns-1, F, 3]
        ends = world_pos[:, 1:]
        centers = 0.5 * (starts + ends)
        vecs = ends - starts
        lens = torch.linalg.norm(vecs, dim=-1).clamp(min=_DEBUG_VIZ_EPS)                  # [n, Ns-1, F]
        dirs = vecs / lens.unsqueeze(-1)

        # Quaternion rotating each cylinder's local +z onto the segment direction.
        # axis = +z x dir = (-dir.y, dir.x, 0); degenerate (dir || +/-z) -> +x fallback.
        angle = torch.acos(dirs[..., 2].clamp(-1.0, 1.0))
        axis = torch.stack([-dirs[..., 1], dirs[..., 0], torch.zeros_like(dirs[..., 0])], dim=-1)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(axis)
        axis = torch.where(axis_norm > _DEBUG_VIZ_EPS, axis, fallback)
        quats = quat_from_angle_axis(angle.reshape(-1), axis.reshape(-1, 3))             # [M, 4] xyzw

        # Native cylinder height is 1.0; z-scale by the segment length.
        scales = torch.ones_like(dirs)
        scales[..., 2] = lens

        # Marker prototype index == frame index (insertion order in _create_debug_markers).
        marker_indices = (
            torch.arange(F, device=self.device, dtype=torch.long)
            .view(1, 1, F).expand(n, ns - 1, F).reshape(-1)
        )
        centers, quats = centers.reshape(-1, 3), quats
        scales, marker_indices = scales.reshape(-1, 3), marker_indices
        self._draw_obs_and_bodies(F, (centers, quats, scales, marker_indices))

    def _draw_obs_and_bodies(self, num_frames: int, clip_ribbon) -> None:
        """Append the observed-window ribbon to the clip ribbon, visualize, draw the bodies.

        Split out of :meth:`_debug_vis_callback` so it still runs when the clip ribbon is
        suppressed (see ``_clip_ribbon_env_mask``): the observed window and the tracked-body
        pairs describe the policy and the reward, not the clip, and are exactly what you want
        to see on a generator-driven env. Both are drawn for EVERY env -- unlike the clip
        ribbon they are per-env independent, so there is nothing to subselect for alignment.

        Args:
            num_frames: Reference-frame count; also the observed-ribbon prototype index.
            clip_ribbon: ``(centers, quats, scales, marker_indices)`` from the clip ribbon,
                or ``None`` when it was not drawn.
        """
        parts = list(clip_ribbon) if clip_ribbon is not None else None

        # The policy's observed reference-window base, as a second ribbon. Anchored, so it
        # rides the robot; its offset from the global clip ribbon is the drift.
        obs_off = self.obs_window_offsets() if self.cfg.debug_vis_obs_window else None
        if obs_off is not None and obs_off.numel() >= 2:
            opos, _, _, _ = self.get_future_world_base(obs_off)                     # [N, K, 3]
            oc, oq, os_, oi = self._ribbon_segments(opos, num_frames)               # prototype F
            if parts is None:
                parts = [oc, oq, os_, oi]
            else:
                parts = [torch.cat([a, b], dim=0) for a, b in zip(parts, (oc, oq, os_, oi))]

        if parts is None:
            self._debug_markers.set_visibility(False)
        else:
            self._debug_markers.visualize(
                translations=parts[0], orientations=parts[1],
                scales=parts[2], marker_indices=parts[3],
            )
        self._draw_body_spheres(None)

    def _ribbon_segments(self, pts: torch.Tensor, proto: int):
        """Cylinder segments joining consecutive points of a ``[n, K, 3]`` polyline.

        Args:
            pts: ``[n, K, 3]`` world points, one polyline per env.
            proto: Marker prototype index to tag every segment with.

        Returns:
            ``(centers, quats, scales, indices)`` flattened over ``n * (K - 1)``.
        """
        starts, ends = pts[:, :-1], pts[:, 1:]
        centers = 0.5 * (starts + ends)
        vecs = ends - starts
        lens = torch.linalg.norm(vecs, dim=-1).clamp(min=_DEBUG_VIZ_EPS)
        dirs = vecs / lens.unsqueeze(-1)
        angle = torch.acos(dirs[..., 2].clamp(-1.0, 1.0))
        axis = torch.stack([-dirs[..., 1], dirs[..., 0], torch.zeros_like(dirs[..., 0])], dim=-1)
        norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        axis = torch.where(norm > _DEBUG_VIZ_EPS,
                           axis, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(axis))
        quats = quat_from_angle_axis(angle.reshape(-1), axis.reshape(-1, 3))
        scales = torch.ones_like(dirs)
        scales[..., 2] = lens
        idx = torch.full((centers.shape[0] * centers.shape[1],), proto,
                         dtype=torch.long, device=self.device)
        return centers.reshape(-1, 3), quats, scales.reshape(-1, 3), idx

    def _draw_body_spheres(self, env_ids: torch.Tensor | None) -> None:
        """Spheres at each tracked body: measured (green) vs the rewarded reference (magenta).

        Both are read straight out of ``y_act`` / ``y_des``, i.e. the exact tensors the CLF
        differences -- so the visualisation cannot disagree with the reward, and it follows
        ``tracking_mode`` without restating a body list.

        EXPECTED: the green spheres visibly TRAIL their links during fast motion, and the
        faster the link the further they trail.  That is a render-cadence artifact, not a
        frame error -- ``y_act`` is a straight ``body_pos_w`` read and matches a live one to
        0.000000 m on the tick it is written.  The markers are written once per CONTROL step
        while the viewport draws faster, and a link covers up to ~0.17 m per control step
        (ankles, measured; ~0.11 m at the wrists) against a 0.03 m sphere radius -- roughly
        six sphere diameters, so the offset is obvious.
        What stays exact is the green-to-magenta SEPARATION, since both come from the same
        tick; it is only the overlay onto the rendered mesh that lags.  So read the gap, not
        a dot's absolute position on a moving link.  ``play.py --real-time`` largely hides
        it.  Redrawing green from a live read at render rate would glue the dots to the links
        but put the two colours on different ticks, corrupting the one quantity this is for.
        """
        if self._debug_body_markers is None or not self.cfg.debug_vis_body_spheres:
            return
        idx = self._tracked_body_pos_idx()
        if idx is None:
            return
        flat = idx.reshape(-1)
        sl = slice(None) if env_ids is None else env_ids
        act = self.y_act[sl][:, flat].reshape(-1, idx.shape[0], 3)      # [n, B, 3]
        des = self.y_des[sl][:, flat].reshape(-1, idx.shape[0], 3)
        n, b = act.shape[0], act.shape[1]
        pts = torch.cat([act.reshape(-1, 3), des.reshape(-1, 3)], dim=0)
        proto = torch.cat([
            torch.zeros(n * b, dtype=torch.long, device=self.device),
            torch.ones(n * b, dtype=torch.long, device=self.device),
        ])
        self._debug_body_markers.visualize(translations=pts, marker_indices=proto)
