# Copyright 2026 Zachary Olkin. All rights reserved.

"""Flow-matching generator-tracking command (world-frame CLF).

Extends :class:`MotionSequenceCommand`.  Every env is reset onto a pre-made clip
(supplying the FM velocity command ``target_vel`` and the spawn pose), but a
configurable fraction of envs track the **flow-matching generator's** predicted
future state instead of the clip reference:

  * **clip-tracking envs** (fraction ``clip_fraction``) follow the clip -- the
    stable teacher signal that lets the tracker bootstrap while its own rollout
    history (the FM input) is still garbage;
  * **FM-tracking envs** follow the generator's predicted next state, after the
    rolling history fills (they fall back to the clip during the warmup window).

Both sources drive a single **world-frame** CLF: the desired outputs
``y_des``/``dy_des`` and the measured outputs ``y_act``/``dy_act`` are expressed
in the world frame (no stance-anchor re-anchoring), so the inherited CLF reward,
observations, and terminations work uniformly for either source.  The world
frame is a single stationary inertial frame, valid for the CLF's finite-
difference ``vdot``.

Base-velocity convention: every base/ee velocity is the **link** (actor-frame)
twist.  The FM is trained and conditioned on ``root_link_{lin,ang}_vel_w`` (see
``record_data.py``) and the measured ``y_act`` base velocity comes from
``body_link_vel_w`` -- the same pelvis link frame.  No CoM velocities are used
anywhere, so the FM reference and the CLF's measured outputs share one frame
(no ``omega x r_com`` mismatch).
"""

from __future__ import annotations

import torch
import warp as wp

from robot_rl.flow_matching.online import (
    BatchedHistory,
    build_history_token,
    load_fm_context,
    sample_world,
)

from .motion_sequence_cmd import MotionSequenceCommand
from .generator_tracking_cmd_cfg import GeneratorTrackingCommandCfg


class GeneratorTrackingCommand(MotionSequenceCommand):
    """Track a flow-matching generator (or the clip) in the world frame."""

    cfg: GeneratorTrackingCommandCfg

    # Generator-tracking debug-viz marker handles (lazily built once the sim is
    # up, alongside the inherited clip-ribbon ``_debug_markers``).
    _fm_markers: object | None = None
    _source_markers: object | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _post_init(self) -> None:
        """Set up the FM context, history buffer, source mask, and world index maps."""
        super()._post_init()
        n, dev = self.num_envs, self.device

        if not self.command_velocity_available:
            raise RuntimeError(
                "GeneratorTrackingCommand requires a clip library with target_vel "
                "(the FM velocity command). Regenerate clips with a generator that writes it."
            )
        if not self.cfg.fm_checkpoint:
            raise ValueError("GeneratorTrackingCommandCfg.fm_checkpoint must be set.")

        # The motion-sequence task is flat ground -> the FM history token has no
        # height-scan columns. (Update scan_dim here if a perceptive variant is added.)
        self.ctx = load_fm_context(
            self.cfg.fm_checkpoint,
            device=self.cfg.fm_device or dev,
            num_sampling_steps=self.cfg.fm_num_sampling_steps,
            scan_dim=0,
        )
        if self.ctx.future_cmd_dim <= 0:
            raise ValueError(
                "GeneratorTrackingCommand expects a future-command (cmd_array='target_vel') "
                "flow-matching checkpoint; the loaded checkpoint has no decoder command "
                "conditioning (cmd_dim=0)."
            )

        # The FM's extra frames must be exactly the CLF's end-effector output frames.
        ee_frames = self.resolve_ee_frames()
        if set(self.ctx.frame_names) != set(ee_frames):
            raise ValueError(
                f"FM checkpoint extra frames {self.ctx.frame_names} != command ee output "
                f"frames {ee_frames}; re-record/-train the FM with --state_frames matching them."
            )

        # -- history buffer + joint/frame permutations (robot order -> FM order) --
        robot_joints = list(self.robot.data.joint_names)
        self._joint_perm = torch.tensor(
            [robot_joints.index(j) for j in self.ctx.joint_names], dtype=torch.long, device=dev
        )
        self._ee_body_idx = (
            torch.tensor([self.robot.body_names.index(f) for f in self.ctx.frame_names],
                         dtype=torch.long, device=dev)
            if self.ctx.num_extra_frames > 0 else None
        )
        self._history = BatchedHistory(n, self.ctx.H, dev)

        # -- world-output index maps: frames = [base] + FM ee frames, in FM order --
        self._wf_frames = [self.cfg.base_frame_name] + list(self.ctx.frame_names)
        pos_list, ori_list, lin_list, ang_list = [], [], [], []
        for f in self._wf_frames:
            idx = self._frame_output_indices(f)
            pos_list.append(idx["pos"]); ori_list.append(idx["ori"])
            lin_list.append(idx["lin_vel"]); ang_list.append(idx["ang_vel"])
        self._wf_pos_flat = torch.cat(pos_list)   # [nf*3] -> y_des
        self._wf_ori_flat = torch.cat(ori_list)   # [nf*4] -> y_des
        self._wf_lin_flat = torch.cat(lin_list)   # [nf*3] -> dy_des
        self._wf_ang_flat = torch.cat(ang_list)   # [nf*3] -> dy_des

        # -- joint slots (clip-output order) + FM->clip joint permutation --
        joint_out_names = [n.split(":", 1)[1] for n in self.ordered_pos_output_names if n.startswith("joint:")]
        self._joint_slots_pos = torch.tensor(
            [self.ordered_pos_output_names.index(f"joint:{n}") for n in joint_out_names],
            dtype=torch.long, device=dev,
        )
        self._joint_slots_vel = torch.tensor(
            [self.ordered_vel_output_names.index(f"joint:{n}") for n in joint_out_names],
            dtype=torch.long, device=dev,
        )
        self._fm_jperm = torch.tensor(
            [self.ctx.joint_names.index(n) for n in joint_out_names], dtype=torch.long, device=dev
        )

        # -- per-env source mask (True => FM), prediction buffers, cadence state --
        self._fm_mask = torch.zeros(n, dtype=torch.bool, device=dev)  # re-rolled each reset
        f, nj, k = self.ctx.F, self.ctx.nj, self.ctx.num_extra_frames
        self._pred_root_pos = torch.zeros(n, f, 3, device=dev)
        self._pred_root_quat = torch.zeros(n, f, 4, device=dev)
        self._pred_root_lin = torch.zeros(n, f, 3, device=dev)
        self._pred_root_ang = torch.zeros(n, f, 3, device=dev)
        self._pred_jp = torch.zeros(n, f, nj, device=dev)
        self._pred_jv = torch.zeros(n, f, nj, device=dev)
        self._pred_frame_pos = torch.zeros(n, f, k, 3, device=dev)
        self._pred_frame_quat = torch.zeros(n, f, k, 4, device=dev)
        self._pred_frame_lin = torch.zeros(n, f, k, 3, device=dev)
        self._pred_frame_ang = torch.zeros(n, f, k, 3, device=dev)
        self._pred_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self._pred_valid = torch.zeros(n, dtype=torch.bool, device=dev)

        # World-frame base reference buffer (read by reference_world_state()).
        self._ref_world_base = torch.zeros(n, 13, device=dev)
        # Constant frame for the world-frame measured outputs.
        self._zeros3 = torch.zeros(n, 3, device=dev)
        self._ident_quat = torch.zeros(n, 4, device=dev)
        self._ident_quat[:, 3] = 1.0  # xyzw identity

        self._last_step_seen = -1
        self._arange_n = torch.arange(n, device=dev)

        # --- FM-prediction debug ribbon: map each (filtered) clip-ribbon frame
        # to its slot in the FM prediction buffers. The base frame reads the
        # predicted root; each ee frame reads its index in the predicted-frame
        # buffer; frames the FM does not predict are dropped from the ribbon.
        # ``self._debug_frame_names`` / ``_colors`` are built by the base _post_init.
        self._fm_pred_specs: list[tuple[str, tuple[float, float, float], str, int]] = []
        for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
            if name == self.cfg.base_frame_name:
                self._fm_pred_specs.append((name, color, "base", -1))
            elif name in self.ctx.frame_names:
                self._fm_pred_specs.append((name, color, "frame", self.ctx.frame_names.index(name)))
        # Subsample the F-step prediction horizon for the ribbon (same stride knob).
        self._fm_debug_offsets = torch.arange(
            0, self.ctx.F, max(1, int(self.cfg.debug_vis_stride)), dtype=torch.long, device=dev
        )

    # ------------------------------------------------------------------
    # Per-step update (overrides the clip-only path)
    # ------------------------------------------------------------------

    def _update_command(self) -> None:
        """World-frame desired+measured outputs (clip or FM per env) and CLF update."""
        self._advance_step()
        self._snapshot_and_infer()

        # Clip desired outputs (stance-anchored). Joints are frame-invariant; the
        # base/ee Cartesian channels are overwritten with world values below.
        self.get_desired_outputs()

        # Clip world reference (base + ee), lifted through the clip's world anchor.
        wpos, wquat = self._world_anchor()
        self.ref_poses[:, :3] = wpos
        self.ref_poses[:, 3:7] = wquat
        clip_base = super().reference_world_state()                 # [N, 13 + 2nj]
        clip_ee = self.reference_world_frames(self.ctx.frame_names)  # [N, K, 13]

        # FM world reference at the current advance index.
        fm_base, fm_ee, fm_jp, fm_jv = self._fm_current_outputs()
        use_fm = self._fm_mask & self._pred_valid

        self._assemble_world_des(clip_base[:, :13], clip_ee, fm_base, fm_ee, fm_jp, fm_jv, use_fm)

        # Measured outputs in the world frame (identity reference -> no alignment).
        self.compute_measured_output(self._zeros3, self._ident_quat)

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)
        self.vdot = vdot
        self.v = vcur

    # ------------------------------------------------------------------
    # History snapshot + flow-matching inference (once per control step)
    # ------------------------------------------------------------------

    def _snapshot_and_infer(self) -> None:
        """Append the current world state to history and (re-)run FM inference.

        Guarded to run once per control step (Isaac Lab computes commands twice
        per step). Freshly-reset envs (``episode_length_buf == 0``) get a fresh
        source assignment and an invalidated history window.
        """
        step = int(getattr(self.env, "common_step_counter", self._last_step_seen + 1))
        if step == self._last_step_seen:
            return
        self._last_step_seen = step

        just_reset = self.env.episode_length_buf == 0
        if bool(just_reset.any()):
            # Re-roll the per-env source (True => FM) for envs that just reset.
            roll_fm = torch.rand(self.num_envs, device=self.device) >= self.cfg.clip_fraction
            self._fm_mask = torch.where(just_reset, roll_fm, self._fm_mask)
            self._pred_valid = self._pred_valid & ~just_reset
            self._pred_idx = torch.where(just_reset, torch.zeros_like(self._pred_idx), self._pred_idx)

        # Snapshot the current world state (FM joint/frame order) into the history.
        perm = self._joint_perm
        snap = {
            "root_pos": wp.to_torch(self.robot.data.root_pos_w).clone(),
            "root_quat": wp.to_torch(self.robot.data.root_quat_w).clone(),     # xyzw
            "root_lin": wp.to_torch(self.robot.data.root_link_lin_vel_w).clone(),  # link vel
            "root_ang": wp.to_torch(self.robot.data.root_link_ang_vel_w).clone(),  # link ang vel
            "jp": wp.to_torch(self.robot.data.joint_pos)[:, perm].clone(),
            "jv": wp.to_torch(self.robot.data.joint_vel)[:, perm].clone(),
            "cmd": self.get_command_velocity().clone(),                        # past target_vel [N,3]
        }
        if self._ee_body_idx is not None:
            fpos = wp.to_torch(self.robot.data.body_pos_w)[:, self._ee_body_idx]
            fquat = wp.to_torch(self.robot.data.body_quat_w)[:, self._ee_body_idx]
            flink = wp.to_torch(self.robot.data.body_link_vel_w)[:, self._ee_body_idx]
            snap["frames"] = torch.cat([fpos, fquat, flink[..., :3], flink[..., 3:]], dim=-1).clone()  # [N,K,13]

        # Invalidate reset envs' windows, then append (this snapshot = step 1 of the new episode).
        self._history.invalidate(just_reset)
        self._history.append(snap)

        # Decide which FM envs to (re-)infer vs advance the prediction index.
        valid = self._history.valid()
        self._pred_valid = self._pred_valid & valid          # lose validity if the window broke
        fm_valid = valid & self._fm_mask
        reinfer = (step % max(int(self.cfg.fm_update_every), 1)) == 0
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        infer_mask = fm_valid & (reinfer | ~self._pred_valid | exhausted)
        advance = fm_valid & ~infer_mask
        self._pred_idx = torch.where(
            advance, torch.clamp(self._pred_idx + 1, max=self.ctx.F - 1), self._pred_idx
        )
        if bool(infer_mask.any()):
            self._run_fm(infer_mask)

    @torch.no_grad()
    def _run_fm(self, mask: torch.Tensor) -> None:
        """Run batched flow-matching inference for the masked envs and store predictions."""
        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        rp = self._history.stack("root_pos")[ids]    # [V,H,3]
        rq = self._history.stack("root_quat")[ids]
        rl = self._history.stack("root_lin")[ids]
        ra = self._history.stack("root_ang")[ids]
        jp = self._history.stack("jp")[ids]          # [V,H,nj] FM order
        jv = self._history.stack("jv")[ids]
        cmd = self._history.stack("cmd")[ids]         # [V,H,3] past target_vel
        frames = self._history.stack("frames")[ids] if self._ee_body_idx is not None else None

        token = build_history_token(self.ctx, rp, rq, rl, ra, jp, jv, cmd, frames=frames, scan=None)
        # Future command horizon (steps s+1..s+F) for the decoder, current-step subset.
        fut_cmd = self.get_command_velocity_horizon(self.ctx.F)[ids]  # [V,F,3]
        # Anchor each prediction at the current base pose (the deployable relative frame).
        p_ref = rp[:, -1]
        q_ref = rq[:, -1]
        world = sample_world(self.ctx, token, p_ref, q_ref, future_cmd=fut_cmd)

        self._pred_root_pos[ids] = world["root_pos_w"]
        self._pred_root_quat[ids] = world["root_quat_w"]
        self._pred_root_lin[ids] = world["root_link_lin_vel_w"]
        self._pred_root_ang[ids] = world["root_link_ang_vel_w"]
        self._pred_jp[ids] = world["joint_pos"]
        self._pred_jv[ids] = world["joint_vel"]
        if self._ee_body_idx is not None:
            self._pred_frame_pos[ids] = world["frame_pos_w"]
            self._pred_frame_quat[ids] = world["frame_quat_w"]
            self._pred_frame_lin[ids] = world["frame_lin_vel_w"]
            self._pred_frame_ang[ids] = world["frame_ang_vel_w"]
        self._pred_idx[ids] = 0
        self._pred_valid[ids] = True

    def _fm_current_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather the FM prediction at each env's current advance index.

        Returns:
            ``(base[N,13], ee[N,K,13], jp[N,nj], jv[N,nj])`` in the world frame;
            ``jp``/``jv`` are in the FM joint order (permuted to clip slots later).
            Values are only meaningful where ``self._pred_valid`` is True.
        """
        ar, idx = self._arange_n, self._pred_idx
        base = torch.cat([
            self._pred_root_pos[ar, idx], self._pred_root_quat[ar, idx],
            self._pred_root_lin[ar, idx], self._pred_root_ang[ar, idx],
        ], dim=-1)  # [N,13]
        ee = torch.cat([
            self._pred_frame_pos[ar, idx], self._pred_frame_quat[ar, idx],
            self._pred_frame_lin[ar, idx], self._pred_frame_ang[ar, idx],
        ], dim=-1)  # [N,K,13]
        return base, ee, self._pred_jp[ar, idx], self._pred_jv[ar, idx]

    # ------------------------------------------------------------------
    # World-frame desired-output assembly
    # ------------------------------------------------------------------

    def _assemble_world_des(
        self,
        clip_base: torch.Tensor,
        clip_ee: torch.Tensor,
        fm_base: torch.Tensor,
        fm_ee: torch.Tensor,
        fm_jp: torch.Tensor,
        fm_jv: torch.Tensor,
        use_fm: torch.Tensor,
    ) -> None:
        """Write world-frame ``y_des``/``dy_des`` (clip or FM per env) in the ordered layout.

        Args:
            clip_base: Clip world base ``[N,13]`` = ``[pos3, quat4, lin3, ang3]``.
            clip_ee: Clip world ee frames ``[N,K,13]`` (same per-frame layout).
            fm_base: FM world base ``[N,13]``.
            fm_ee: FM world ee frames ``[N,K,13]``.
            fm_jp: FM joint positions ``[N,nj]`` (FM order).
            fm_jv: FM joint velocities ``[N,nj]`` (FM order).
            use_fm: ``[N]`` bool; True => take the FM reference for that env.
        """
        n = self.num_envs
        clip_frames = torch.cat([clip_base[:, None, :], clip_ee], dim=1)  # [N, 1+K, 13]
        fm_frames = torch.cat([fm_base[:, None, :], fm_ee], dim=1)        # [N, 1+K, 13]
        sel = torch.where(use_fm[:, None, None], fm_frames, clip_frames)  # [N, nf, 13]

        self.y_des[:, self._wf_pos_flat] = sel[:, :, 0:3].reshape(n, -1)
        self.y_des[:, self._wf_ori_flat] = sel[:, :, 3:7].reshape(n, -1)
        self.dy_des[:, self._wf_lin_flat] = sel[:, :, 7:10].reshape(n, -1)
        self.dy_des[:, self._wf_ang_flat] = sel[:, :, 10:13].reshape(n, -1)

        # FM envs use the FM-predicted joints (reordered to clip slots); clip envs
        # keep the gathered clip joints (joints are frame-invariant).
        fm_jp_slots = fm_jp[:, self._fm_jperm]
        fm_jv_slots = fm_jv[:, self._fm_jperm]
        self.y_des[:, self._joint_slots_pos] = torch.where(
            use_fm[:, None], fm_jp_slots, self.y_des[:, self._joint_slots_pos]
        )
        self.dy_des[:, self._joint_slots_vel] = torch.where(
            use_fm[:, None], fm_jv_slots, self.dy_des[:, self._joint_slots_vel]
        )

        # Cache the selected world base for reference_world_state().
        self._ref_world_base = sel[:, 0, :]

    # ------------------------------------------------------------------
    # World-frame reference state (base) for obs / termination
    # ------------------------------------------------------------------

    def reference_world_state(self) -> torch.Tensor:
        """World-frame reference state ``[N, 3+4+3+3+2*nj]`` (selected source).

        Built directly from the world-frame base buffer + the world-frame
        ``y_des``/``dy_des`` joint channels (joints reordered to robot order),
        so it reflects the FM prediction for FM-tracking envs and the clip for
        clip-tracking envs -- replacing the clip-only stance-anchor lift.
        """
        joint_pos = self.y_des[:, self._base_idx["joint_pos"]]
        joint_vel = self.dy_des[:, self._base_idx["joint_vel"]]
        return torch.cat([self._ref_world_base, joint_pos, joint_vel], dim=-1)

    # ------------------------------------------------------------------
    # Debug visualization (extends the inherited clip ribbon)
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Toggle the clip ribbon (inherited) plus the generator-tracking layers.

        The FM-prediction ribbon and per-env source markers are created lazily
        in :meth:`_debug_vis_callback` (the sim may not be up yet); here we only
        hide them when debug viz is turned off.
        """
        super()._set_debug_vis_impl(debug_vis)
        if not debug_vis:
            for m in (self._fm_markers, self._source_markers):
                if m is not None:
                    m.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Draw the clip ribbon (cylinders) + FM prediction (spheres) + source markers.

        The inherited callback draws the clip reference ribbon for every env (the
        clip each env was reset on, advancing by step index). On top of that we
        overlay, per the cfg flags, the flow-matching generator's predicted
        trajectory and a per-env indicator of which reference is authoritative.
        """
        super()._debug_vis_callback(event)
        if self.cfg.debug_vis_show_fm_prediction:
            self._draw_fm_prediction_ribbon()
        if self.cfg.debug_vis_show_source:
            self._draw_source_markers()

    def _clip_ribbon_env_mask(self) -> torch.Tensor:
        """Draw the clip ribbon only where the env is NOT actively tracking the FM.

        FM-active envs (``_fm_mask & _pred_valid``) show the FM-prediction
        spheres instead; clip-tracking and FM-warmup envs (still following the
        clip) keep the clip ribbon.
        """
        return ~(self._fm_mask & self._pred_valid)

    def _draw_fm_prediction_ribbon(self) -> None:
        """Draw the FM generator's predicted world trajectory as per-frame spheres.

        Only the envs currently tracking the FM (``_fm_mask & _pred_valid``) are
        drawn; the predicted positions are read straight from the world-frame
        prediction buffers (subsampled over the F-step horizon). Sphere colors
        match the clip ribbon's per-frame colors, so the FM prediction (spheres)
        can be compared directly to the clip baseline (cylinders).
        """
        if self._fm_markers is None:
            self._fm_markers = self._create_fm_markers()
            if self._fm_markers is None:
                return
        if not self._fm_pred_specs:
            self._fm_markers.set_visibility(False)
            return

        use_fm = self._fm_mask & self._pred_valid
        ids = torch.nonzero(use_fm, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            self._fm_markers.set_visibility(False)
            return

        offs = self._fm_debug_offsets
        translations, marker_indices = [], []
        for proto_idx, (_name, _color, kind, k) in enumerate(self._fm_pred_specs):
            if kind == "base":
                pts = self._pred_root_pos[ids][:, offs]        # [V, Ns, 3]
            else:
                pts = self._pred_frame_pos[ids][:, offs, k]    # [V, Ns, 3]
            pts = pts.reshape(-1, 3)
            translations.append(pts)
            marker_indices.append(
                torch.full((pts.shape[0],), proto_idx, dtype=torch.long, device=self.device)
            )

        self._fm_markers.set_visibility(True)
        self._fm_markers.visualize(
            translations=torch.cat(translations, dim=0),
            marker_indices=torch.cat(marker_indices, dim=0),
        )

    def _draw_source_markers(self) -> None:
        """Float a colored marker above each robot showing its tracking source.

        Green => FM (history full, prediction live); yellow => assigned to the FM
        but still warming up (falling back to the clip); white => clip.
        """
        if self._source_markers is None:
            self._source_markers = self._create_source_markers()
            if self._source_markers is None:
                return

        use_fm = self._fm_mask & self._pred_valid
        warmup = self._fm_mask & ~self._pred_valid
        idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # 1 => clip
        idx[use_fm] = 0   # 0 => fm
        idx[warmup] = 2   # 2 => warmup

        pos = wp.to_torch(self.robot.data.root_pos_w).clone()
        pos[:, 2] += float(self.cfg.debug_vis_source_marker_height)
        self._source_markers.set_visibility(True)
        self._source_markers.visualize(translations=pos, marker_indices=idx)

    def _create_fm_markers(self):
        """Build a sphere-per-frame marker set for the FM-prediction ribbon."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN GeneratorTrackingCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_fm_marker_radius)
        prototypes = {
            f"fm_{name}": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
            for name, color, _kind, _k in self._fm_pred_specs
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/generator_pred", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN GeneratorTrackingCommand] could not create FM markers: {exc}")
            return None

    def _create_source_markers(self):
        """Build the three-color (fm / clip / warmup) tracking-source indicator set."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:  # sim not up yet
            print(f"[WARN GeneratorTrackingCommand] could not import VisualizationMarkers: {exc}")
            return None
        radius = float(self.cfg.debug_vis_source_marker_radius)
        # Insertion order fixes the marker_indices mapping: 0 => fm, 1 => clip, 2 => warmup.
        colors = {"fm": (0.10, 0.90, 0.20), "clip": (0.90, 0.90, 0.90), "warmup": (0.95, 0.85, 0.10)}
        prototypes = {
            name: sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            for name, color in colors.items()
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/track_source", markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN GeneratorTrackingCommand] could not create source markers: {exc}")
            return None
