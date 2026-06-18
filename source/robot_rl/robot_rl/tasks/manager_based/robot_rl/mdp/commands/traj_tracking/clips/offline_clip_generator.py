# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline stitching of per-stride references into long motion clips.

A clip is built by a control-step rollout, batched over many clips at
once.  Each clip walks through a sampled velocity profile
(:class:`OfflineVelocitySampler`); at every stride boundary the matching
source stride is selected with the *same* nearest-neighbour rule the live
sim uses (:meth:`MultiSkillManager._select_within_skills` with the terrain
filter disabled), and the stride is placed in the world by re-anchoring to
the previous stride's reference stance-foot landing pose.  This reproduces
the live contact re-anchor (anchor to the stance foot once per gait cycle),
but uses the *reference* foot pose computed offline -- so there is no
re-anchoring during training and hence no left/right tracking asymmetry.

The stored ``y_des``/``dy_des`` are exactly the source strides' anchor-
relative outputs (unchanged); stitching only computes the per-step world
anchor schedule.  See ``world_lift.py`` for the lift shared with runtime.

Output of :meth:`generate_batch` is a dict of dense ``[n, T, .]`` tensors;
HDF5 serialisation lives in the generation CLI.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import yaw_quat

from ..multiskill_manager import MultiSkillManager
from ..skill_bucket import bucket_for_velocity
from ..skill_transition import blend_outputs
from ..world_lift import lift_orientation, lift_position
from .offline_velocity_sampler import OfflineVelocitySampler

# Phase used to read the "end of stride" stance-foot landing pose.  Just shy
# of 1.0 so the lookup stays inside the stride's last domain.
_END_PHASE = 1.0 - 1e-4


class OfflineClipGenerator:
    """Generate dense, world-anchored motion clips from a stride library.

    Args:
        manager: A loaded :class:`MultiSkillManager` (``env=None``).
        sampler: An :class:`OfflineVelocitySampler` whose ``skill_list`` is
            in the manager's skill order (so its skill ids and bucket
            tables line up with the manager's).
        dt: Control-step dt of the generated clip (seconds).
        clip_length_s: Length of each clip (seconds).
        contact_frames: Frame names whose contact schedule is recorded.
        device: Torch device.
    """

    def __init__(
        self,
        manager: MultiSkillManager,
        sampler: OfflineVelocitySampler,
        *,
        dt: float,
        clip_length_s: float,
        contact_frames: Sequence[str] | None = None,
        exclude_skills: Sequence[str] | None = None,
        project_to_ground: bool = True,
        ground_height: float = 0.0,
        seam_blend_steps: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.manager = manager
        self.sampler = sampler
        self.device = torch.device(device)
        self.dt = float(dt)
        self.clip_length_s = float(clip_length_s)
        self.num_steps = int(round(self.clip_length_s / self.dt))
        # Pin each stance-foot re-anchor to the ground plane (flat terrain) so
        # the anchor z does not accumulate the per-cycle landing-height residual.
        self.project_to_ground = bool(project_to_ground)
        self.ground_height = float(ground_height)

        self.contact_frames = list(contact_frames) if contact_frames is not None else [
            "left_ankle_roll_link", "right_ankle_roll_link",
        ]

        # Validate sampler/manager skill alignment: the sampler's skill ids
        # must equal the manager's skill ids so bucket_for_velocity output
        # indexes the manager's per-skill blocks directly.
        manager_skills = [s.name for s in manager.skills]
        if sampler.skill_list != manager_skills:
            raise ValueError(
                "OfflineVelocitySampler.skill_list must match the manager's skill "
                f"order. sampler={sampler.skill_list} manager={manager_skills}"
            )
        self.num_skills = len(manager_skills)

        # Skills allowed for stride selection (excluded ones are never picked,
        # even if a ramped velocity would land in their bucket).
        excluded = set(exclude_skills or [])
        bad = excluded - set(manager_skills)
        if bad:
            raise ValueError(f"exclude_skills not in library: {sorted(bad)}")
        self._allowed = torch.tensor(
            [s not in excluded for s in manager_skills], dtype=torch.bool, device=self.device,
        )
        self.exclude_skills = sorted(excluded)

        # Per-skill bucket tables (manager order) for bucket_for_velocity.
        self._lin_x = sampler.lin_x
        self._lin_y = sampler.lin_y
        self._ang_z = sampler.ang_z

        # Output layout.
        self.pos_output_names = list(manager.pos_output_names)
        self.vel_output_names = list(manager.vel_output_names)
        self.num_pos = manager.num_pos_outputs
        self.num_vel = manager.num_vel_outputs

        # Reference (anchor) frame -- right ankle for this library; read its
        # pos/ori output indices for the seam re-anchor.
        ref_frame = manager.ref_frames[0]
        self._ra_pos_idx = torch.tensor(
            [self.pos_output_names.index(f"{ref_frame}:{a}") for a in ("pos_x", "pos_y", "pos_z")],
            dtype=torch.long, device=self.device,
        )
        # Isaac Lab math is xyzw (scalar last); the YAML stores ori scalar
        # first (ori_w), so gather as [x, y, z, w] to match the math utils.
        self._ra_ori_idx = torch.tensor(
            [self.pos_output_names.index(f"{ref_frame}:{a}") for a in ("ori_x", "ori_y", "ori_z", "ori_w")],
            dtype=torch.long, device=self.device,
        )
        self.ref_frame = ref_frame

        # Per-trajectory total time, on device.
        self._total_time = manager.data["total_time"].to(self.device)
        # Per-trajectory natural stance-ankle height (the ref-frame z offset
        # production adds at spawn): the height to pin the ground anchor to so
        # the foot rests ON the ground rather than in it.  Domain 0 (stance).
        self._ref_offset_z = manager.data["ref_frame_offset"][:, 0, 2].to(self.device)

        # Seam cross-fade: blend the new stride's first K steps with the old
        # stride continued (re-expressed in the new anchor) to smooth the
        # otherwise-hard hand-off.  0 disables.
        self.seam_blend_steps = int(seam_blend_steps)
        if self.seam_blend_steps > 0:
            self._build_blend_groups()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_strides(self, vel_cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick (skill_id, global stride index) for each clip from a velocity.

        Mirrors the live pipeline: the active skill is the box-nearest
        bucket (``bucket_for_velocity``), then the stride is the velocity
        nearest-neighbour within that skill's block
        (``_select_within_skills`` with the terrain filter off).
        """
        n = vel_cmd.shape[0]
        eligible = self._allowed.view(self.num_skills, 1).expand(self.num_skills, n)
        skill = bucket_for_velocity(vel_cmd, eligible, self._lin_x, self._lin_y, self._ang_z)
        stride = self.manager._select_within_skills(vel_cmd, skill, anchor_xy=None)
        return skill, stride

    def _stance_landing_world(
        self, stride: torch.Tensor, anchor_pos: torch.Tensor, anchor_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the stance (right-ankle) frame at the end of a stride.

        This becomes the next stride's anchor: the reference foot landing
        pose, lifted to world via the current anchor.  Orientation is
        reduced to yaw so the anchor stays a flat ground frame.
        """
        self.manager.set_trajectory_indices(stride)
        end_pos, _ = self.manager.get_output(
            torch.full((stride.shape[0],), _END_PHASE, device=self.device)
        )
        ra_pos_rel = end_pos[:, self._ra_pos_idx]   # [k, 3]
        ra_quat_rel = end_pos[:, self._ra_ori_idx]  # [k, 4] wxyz
        # x/y come from the lifted landing; z is pinned by the caller to the
        # new stride's natural ankle height (avoids cumulative z drift).
        new_pos = lift_position(anchor_pos, anchor_quat, ra_pos_rel)
        # Keep the anchor a yaw-only ground frame (the lift only uses yaw).
        new_quat = yaw_quat(lift_orientation(anchor_quat, ra_quat_rel))
        return new_pos, new_quat

    def _anchor_ground_z(self, stride: torch.Tensor) -> torch.Tensor:
        """Pinned anchor z for a stride: ground height + its stance-ankle offset."""
        return self.ground_height + self._ref_offset_z[stride]

    # ------------------------------------------------------------------
    # Seam cross-fade
    # ------------------------------------------------------------------

    def _build_blend_groups(self) -> None:
        """Per-frame orientation column groups (xyzw) for SLERP in blend_outputs.

        The old stride's continuation re-anchors to the *same* landing pose as
        the new stride (that is how the seam anchor is computed), so both are
        anchor-relative in the same frame and blend directly -- only the quat
        columns need SLERP rather than a linear blend.
        """
        frames = [n[: -len(":pos_x")] for n in self.pos_output_names if n.endswith(":pos_x")]
        frames = [f for f in frames if f"{f}:ori_w" in self.pos_output_names]
        self._blend_ori_groups = torch.tensor(
            [[self.pos_output_names.index(f"{f}:ori_{a}") for a in ("x", "y", "z", "w")] for f in frames],
            dtype=torch.long, device=self.device,
        )  # [F, 4]

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def generate_batch(self, n_clips: int) -> dict[str, torch.Tensor | dict]:
        """Generate ``n_clips`` clips as a batched control-step rollout.

        Returns:
            Dict with dense tensors (all on ``self.device``):
              ``y_des`` ``[n, T, P]``, ``dy_des`` ``[n, T, V]`` --
                  anchor-relative reference outputs (the source strides);
              ``anchor_pos`` ``[n, T, 3]``, ``anchor_quat`` ``[n, T, 4]`` --
                  per-step world anchor schedule in the clip frame;
              ``contact`` ``[n, T, C]`` uint8 -- expected contact schedule;
              ``skill_id`` ``[n, T]`` uint8, ``stride_idx`` ``[n, T]`` int32 --
                  per-step labels;
            plus a ``meta`` dict (dt, clip length, output/contact names).
        """
        n, T = int(n_clips), self.num_steps
        dev, m = self.device, self.manager
        P, V, C = self.num_pos, self.num_vel, len(self.contact_frames)

        # --- Allocate output buffers ---
        y_des = torch.empty(n, T, P, device=dev)
        dy_des = torch.empty(n, T, V, device=dev)
        anchor_pos_buf = torch.empty(n, T, 3, device=dev)
        anchor_quat_buf = torch.empty(n, T, 4, device=dev)
        contact_buf = torch.empty(n, T, C, dtype=torch.uint8, device=dev)
        skill_buf = torch.empty(n, T, dtype=torch.uint8, device=dev)
        stride_buf = torch.empty(n, T, dtype=torch.int32, device=dev)
        target_vel_buf = torch.empty(n, T, 3, device=dev)  # ramped (vx, vy, wz) command

        # --- Initial state ---
        vel = self.sampler.reset(n)                          # [n, 3]
        skill_id, stride_idx = self._select_strides(vel)     # [n], [n]
        phase = torch.zeros(n, device=dev)
        anchor_pos = torch.zeros(n, 3, device=dev)
        if self.project_to_ground:
            anchor_pos[:, 2] = self._anchor_ground_z(stride_idx)  # first landing's natural ankle height
        anchor_quat = torch.zeros(n, 4, device=dev)
        anchor_quat[:, 3] = 1.0                              # identity (xyzw, w last)

        K = self.seam_blend_steps
        if K > 0:
            blend_remaining = torch.zeros(n, dtype=torch.long, device=dev)
            blend_old_stride = stride_idx.clone()
            blend_old_phase = torch.zeros(n, device=dev)

        all_ids = torch.arange(n, device=dev)
        for t in range(T):
            # 1. Store the current frame (pre-advance state).
            m.set_trajectory_indices(stride_idx)
            ypos, yvel = m.get_output(phase)
            contact = m.get_contact_state(phase, self.contact_frames)

            # 1a. Seam cross-fade: for clips within K steps of a stride change,
            # blend the new stride with the old stride continued (re-expressed
            # in the new anchor).  (Done after contact -- it mutates the
            # manager's cached indices, which the top of the next iter resets.)
            if K > 0:
                bmask = blend_remaining > 0
                if bool(bmask.any()):
                    bids = all_ids[bmask]
                    # The old stride continued re-anchors to the same landing as
                    # the new stride, so it is anchor-relative in the *same*
                    # frame -- evaluate and blend directly (no re-expression).
                    m.set_trajectory_indices(blend_old_stride[bids])
                    oy, ody = m.get_output(blend_old_phase[bids] % 1.0)
                    # alpha ramps from ~0 (just past the seam, mostly old) to ~1.
                    alpha = (K - blend_remaining[bids] + 1).float() / (K + 1)
                    ypos[bids] = blend_outputs(oy, ypos[bids], alpha, self._blend_ori_groups)
                    yvel[bids] = blend_outputs(ody, yvel[bids], alpha, None)
                    blend_old_phase[bids] = (
                        blend_old_phase[bids] + self.dt / self._total_time[blend_old_stride[bids]]
                    )
                    blend_remaining[bids] = blend_remaining[bids] - 1

            y_des[:, t] = ypos
            dy_des[:, t] = yvel
            anchor_pos_buf[:, t] = anchor_pos
            anchor_quat_buf[:, t] = anchor_quat
            contact_buf[:, t] = contact.to(torch.uint8)
            skill_buf[:, t] = skill_id.to(torch.uint8)
            stride_buf[:, t] = stride_idx.to(torch.int32)
            # ``vel`` is the ramped command current for this frame (the reset
            # value at t=0, else the previous iteration's sampler.step()).
            target_vel_buf[:, t] = vel

            if t == T - 1:
                break

            # 2. Advance velocity ramp one control step.
            vel = self.sampler.step()

            # 3. Advance phase; handle stride-boundary crossings.
            old_tt = self._total_time[stride_idx]            # [n]
            phase = phase + self.dt / old_tt
            crossed = phase >= 1.0
            if crossed.any():
                ids = all_ids[crossed]
                # Time overshoot past the old stride end (seconds) -- carried
                # into the new stride so the global timeline stays uniform dt.
                overshoot_s = (phase[ids] - 1.0) * old_tt[ids]
                # Arm the seam cross-fade: the old stride keeps going into its
                # next cycle (re-anchored to the new landing == the new anchor),
                # so its continuation phase is the overshoot past 1.0.
                if K > 0:
                    blend_old_stride[ids] = stride_idx[ids]
                    blend_old_phase[ids] = phase[ids] - 1.0
                    blend_remaining[ids] = K
                # New anchor = reference stance landing of the old stride.
                new_pos, new_quat = self._stance_landing_world(
                    stride_idx[ids], anchor_pos[ids], anchor_quat[ids],
                )
                # New stride from the current (ramped) velocity.
                new_skill, new_stride = self._select_strides(vel[ids])
                # Pin the new anchor z to the new stride's natural ankle height
                # (foot on the ground, no cumulative z drift).
                if self.project_to_ground:
                    new_pos[:, 2] = self._anchor_ground_z(new_stride)
                # Commit.
                anchor_pos[ids] = new_pos
                anchor_quat[ids] = new_quat
                stride_idx[ids] = new_stride
                skill_id[ids] = new_skill
                phase[ids] = (overshoot_s / self._total_time[new_stride]).clamp_(min=0.0)

        meta = {
            "dt": self.dt,
            "clip_length_s": self.clip_length_s,
            "num_steps": T,
            "pos_output_names": self.pos_output_names,
            "vel_output_names": self.vel_output_names,
            "contact_frames": self.contact_frames,
            "ref_frame": self.ref_frame,
            "skill_list": list(self.sampler.skill_list),
        }
        return {
            "y_des": y_des,
            "dy_des": dy_des,
            "anchor_pos": anchor_pos_buf,
            "anchor_quat": anchor_quat_buf,
            "contact": contact_buf,
            "skill_id": skill_buf,
            "stride_idx": stride_buf,
            "target_vel": target_vel_buf,
            "meta": meta,
        }
