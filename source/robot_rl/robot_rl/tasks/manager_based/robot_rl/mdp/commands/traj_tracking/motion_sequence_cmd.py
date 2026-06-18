# Copyright 2026 Zachary Olkin. All rights reserved.

"""Open-loop motion-sequence tracking command.

Tracks a pre-generated fixed motion clip (see ``clips/``) by a pure step
index -- no contact gates, no online re-anchoring, no skill cross-fades.
At reset each env is assigned a clip and placed exactly on it; thereafter
the reference advances by ``step = episode_length_buf`` and the robot's
measured outputs are computed relative to the clip's *pre-computed* world
anchor schedule (the reference stance-foot landing poses baked in offline).

Because the anchor schedule is fixed at generation time, there is no
re-anchoring to the measured foot during training, so neither foot gets a
"free zero" -- the left/right tracking asymmetry of the online command is
gone.  The CLF tracking (``y_des``/``y_act`` in the stance-anchored,
yaw-aligned frame) and reward are otherwise unchanged.
"""

from __future__ import annotations

import torch
import warp as wp

from .base_trajectory_cmd import BaseTrajectoryCommand
from .clips.clip_io import load_clip_library
from .clips.clip_fingerprint import fingerprint_clip_config, verify_clip_fingerprint
from .clips.clip_manager import MotionClipManager
from .world_lift import lift_base_world_state, lift_orientation, lift_position
from .motion_sequence_cmd_cfg import MotionSequenceCommandCfg


class MotionSequenceCommand(BaseTrajectoryCommand):
    """Track fixed pre-generated motion clips open-loop."""

    cfg: MotionSequenceCommandCfg

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _create_manager(self, cfg: MotionSequenceCommandCfg, env) -> MotionClipManager:
        """Load the dense clip library and wrap it in a gather manager."""
        device = cfg.clip_device or env.device
        self._clip_lib = load_clip_library(cfg.path, device)
        return MotionClipManager(self._clip_lib)

    def _verify_contact_frames(self) -> None:
        """The clip's stance reference frame must be a tracked contact body."""
        ref = self.manager.ref_frame
        if ref not in self.contact_bodies:
            raise ValueError(
                f"Clip reference frame '{ref}' is not in contact_bodies {self.contact_bodies}."
            )

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
        # Phase carrier so the base ``_update_command`` can read manager.phase.
        self.manager.phase = torch.zeros(n, device=dev)

        # Base-frame output indices for world reconstruction (placement / obs).
        # Force the orientation channels to xyzw order (the YAML stores ori
        # scalar-first as ori_w, but Isaac Lab's quat math is scalar-last), so
        # the lifted base quaternion matches root_quat_w / write_root_pose.
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
                print("\n" + "=" * 78 + f"\n[WARN] {msg}\n" + "=" * 78 + "\n")
        cfg.clip_fingerprint = loaded

    # ------------------------------------------------------------------
    # Lifecycle (overrides the online phase/gate machinery)
    # ------------------------------------------------------------------

    def _update_command(self) -> None:
        """Advance the step index, gather reference + measured, update CLF."""
        self._advance_step()
        self.get_measured_outputs()
        self.get_desired_outputs()
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

    def _world_anchor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env world anchor pose = clip placement ∘ clip anchor[step]."""
        cap, caq = self.manager.gather_anchor(self.clip_id, self._step)
        wpos = lift_position(self.clip_world_pos, self.clip_world_quat, cap)
        wquat = lift_orientation(self.clip_world_quat, caq)
        return wpos, wquat

    def get_measured_outputs(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> None:
        """Compute measured outputs relative to the clip's world anchor.

        Bypasses all online re-anchoring: the anchor is read from the clip's
        pre-computed schedule (lifted into the world by the per-env clip
        placement), stored in ``ref_poses``, then handed to the inherited
        :meth:`compute_measured_output`.
        """
        wpos, wquat = self._world_anchor()
        self.ref_poses[:, :3] = wpos
        self.ref_poses[:, 3:7] = wquat
        self.compute_measured_output(wpos, wquat)

    def get_desired_outputs(self, phase: torch.Tensor = None, env_ids: torch.Tensor = None) -> None:
        """Gather the (stance-anchored) reference outputs for the current step."""
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
    # World-frame reference (for reset placement, obs, termination)
    # ------------------------------------------------------------------

    def reference_world_state(self) -> torch.Tensor:
        """World-frame reference state ``[N, 3+4+3+3+2*nj]`` at the current step.

        ``[base_pos_w, base_quat_w(wxyz), base_lin_w, base_ang_w, joint_pos,
        joint_vel]`` -- lifts the stance-anchored ``y_des`` through the current
        world anchor (``ref_poses``).  Used by the reset event and observations.
        """
        return lift_base_world_state(self.y_des, self.dy_des, self.ref_poses, self._base_idx)

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
            world_quat: ``[K, 4]`` clip-placement world quaternions (wxyz).

        Returns:
            ``[K, 3+4+3+3+2*nj]`` world reference state.
        """
        y, dy = self.manager.gather_desired(clip_id, step)
        cap, caq = self.manager.gather_anchor(clip_id, step)
        wpos = lift_position(world_pos, world_quat, cap)
        wquat = lift_orientation(world_quat, caq)
        anchor = torch.cat([wpos, wquat], dim=-1)
        return lift_base_world_state(y, dy, anchor, self._base_idx)
