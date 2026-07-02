# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline stitching of per-stride references into long motion clips.

A clip is built by a control-step rollout, batched over many clips at
once.  Each clip walks through a sampled velocity profile
(:class:`OfflineVelocitySampler`); at every stride boundary the matching
source stride is selected with the *same* nearest-neighbour rule the live
sim uses (:meth:`MultiSkillManager._select_within_skills`), and the stride is
placed in the world by re-anchoring to the previous stride's reference
stance-foot landing pose.  This reproduces the live contact re-anchor (anchor
to the stance foot once per gait cycle), but uses the *reference* foot pose
computed offline -- so there is no re-anchoring during training and hence no
left/right tracking asymmetry.

Two rollout modes share the stitching core:

* **Flat** (no terrain context): the anchor z is pinned to a constant ground
  plane, selection uses the whole allowed skill set with the terrain filter
  off.  This is the original behavior.
* **Terrain** (:meth:`set_terrain_context` bound): one uniform staircase is
  active.  Stair strides re-anchor onto the tread the reference foot rests on
  (via :func:`stair_stance_ref` + an offline projector, so the anchor z climbs
  with the steps), the terrain-dim filter narrows stair-stride selection to the
  terrain's step dimensions, and a slow-down "approach" step
  (:func:`plan_slowdown_step`) lands the last flat stride at the stair entry
  before committing to the stair skill.

Stitching computes a per-step world anchor schedule, and the source strides'
anchor-relative outputs are then **baked** through that schedule
(:func:`world_lift.bake_outputs_to_world`) so the stored ``y_des``/``dy_des``
are in a single clip-world frame.  At runtime the command applies only a
per-env placement -- there is no stance re-anchoring.

Output of :meth:`generate_batch` is a dict of dense ``[n, T, .]`` tensors;
HDF5 serialisation lives in the generation CLI.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply, yaw_quat

from ...guidance import goal_pd_commands
from ..multiskill_manager import _HALF_PERIODIC_INT, MultiSkillManager
from ..skill_bucket import bucket_for_velocity
from ..terrain_aware_ref import stair_stance_ref
from ..terrain_transition import plan_slowdown_step
from ..world_lift import bake_outputs_to_world, lift_orientation, lift_position
from .offline_velocity_sampler import OfflineVelocitySampler
from .transition_strategy import TransitionStrategy, make_transition_strategy
from .step_length_table import (
    build_ref_frame_to_pos_xy_idx,
    expand_ref_frames,
    traj_step_length_table,
)

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
        exclude_skills: Skills never picked during velocity-bucketed selection.
        project_to_ground: Pin each flat re-anchor to the ground plane.
        ground_height: Flat ground-plane height (m).
        transition: Skill-transition (seam) strategy for the flat rollout --
            ``"linear"`` (default; cross-fade old->new over ``transition_blend_end_phi``
            of phase, mirroring the online blend), ``"raw"`` (hard cut, no blend),
            or a custom :class:`TransitionStrategy` instance (extension hook, e.g.
            motion matching).  Terrain/transition-clip seams are handled by the
            stair state machine and are always a hard cut (a stair commit jumps the
            anchor up a tread, so a continued old stride would float mid-air).
        transition_blend_end_phi: Phi window for the ``"linear"`` cross-fade
            (ignored by ``"raw"``); matches the online ``transition_blend_end_phi``.
        approach_threshold: Distance (m) below which the stance foot is treated
            as "at" the stair stance reference (commit the stair skill).
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
        transition: str | TransitionStrategy = "linear",
        transition_blend_end_phi: float = 0.3,
        approach_threshold: float = 0.15,
        nav_gains: tuple[float, float, float] = (1.5, 0.3, 0.5),
        nav_vy_clamp: tuple[float, float] = (-0.4, 0.4),
        nav_wz_clamp: tuple[float, float] = (-0.5, 0.5),
        device: str | torch.device = "cpu",
    ) -> None:
        self.manager = manager
        self.sampler = sampler
        self.device = torch.device(device)
        # Half-periodic strides are unsupported offline (see _reject_half_periodic).
        self._reject_half_periodic()
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

        # --- Terrain support -------------------------------------------------
        # A skill is terrain-aware iff its trajectories carry terrain
        # conditioning (name-independent -- avoids the stair_up/stairs_up
        # spelling drift).  Flat-only libraries leave this all-False so the
        # terrain code paths never fire.
        self._skill_is_terrain = manager._skill_has_terrain.to(self.device)      # [S] bool
        self._traj_skill_idx = manager.data["skill_idx"].to(self.device)         # [T]
        self._traj_is_terrain = self._skill_is_terrain[self._traj_skill_idx]     # [T] bool
        # Flat skills eligible for velocity-bucketed selection off the terrain.
        self._flat_skill_eligible = self._allowed & ~self._skill_is_terrain      # [S]
        terrain_skill_ids = self._skill_is_terrain.nonzero().flatten()
        self._stair_skill_id = int(terrain_skill_ids[0]) if terrain_skill_ids.numel() else -1
        # Per-trajectory forward swing reach for the slow-down approach step.
        # The ref-frame map needs BOTH feet present so half-periodic reflected
        # domains resolve (manager.ref_frames may list only the right foot); the
        # pos-xy index must key on the same expanded list.
        ref_frames = expand_ref_frames(list(manager.ref_frames))
        manager.build_ref_frame_map(ref_frames)
        self._ref_frame_to_pos_xy_idx = build_ref_frame_to_pos_xy_idx(manager, ref_frames, self.device)
        self._traj_step_length = traj_step_length_table(
            manager, self._ref_frame_to_pos_xy_idx, self.device,
        )                                                                        # [T]
        # Base approach slow-down candidate set: allowed, flat, gated (perpetual
        # standing has inf reach, never chosen), and STRAIGHT (vy = vyaw = 0 -- the
        # approach must be head-on).  This base is further restricted PER TERRAIN
        # to a designated set of approach SKILLS via
        # ``set_terrain_context(approach_skills=...)``.  plan_slowdown_step picks
        # the longest eligible stride whose forward step length fits the gap to the
        # stair, so leaving the skill open lets an off-nominal stride win: a
        # BACKWARD-walking skill (vx < 0, but vy = vyaw = 0 so it passes "straight")
        # is chosen when the gap is shorter than the shortest forward step, and the
        # robot walks backwards on approach.  Masking by skill (forward-walking
        # only) pins the approach to the intended gait and is the single place to
        # extend the candidate set later -- no per-trajectory velocity heuristics.
        allowed_per_traj = self._allowed[self._traj_skill_idx]
        cond = manager._global_conditioning.to(self.device)                      # [T, >=3]: vx, vy, vyaw, ...
        straight = (cond[:, 1].abs() < 0.05) & (cond[:, 2].abs() < 0.05)         # vy ~ 0 and vyaw ~ 0
        self._approach_eligible = (
            allowed_per_traj & ~self._traj_is_terrain
            & torch.isfinite(self._traj_step_length) & straight
        )                                                                        # [T]
        self.approach_threshold = float(approach_threshold)
        self._fwd_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        # Goal-directed PD gains/clamps (transition nav), reused each batch.
        self._nav_gains = torch.tensor([nav_gains], device=self.device)         # [1, 3]
        self._nav_vy = torch.tensor([nav_vy_clamp], device=self.device)         # [1, 2]
        self._nav_wz = torch.tensor([nav_wz_clamp], device=self.device)         # [1, 2]
        # Per-batch terrain context (set via set_terrain_context; None = flat).
        self._projector = None
        self._band_projectors = None
        self._terrain_id = -1
        self._terrain_type = "flat"
        self._spawn_anchor_pos: torch.Tensor | None = None
        self._spawn_anchor_quat: torch.Tensor | None = None
        self._spawn_on_stairs = False
        self._spawn_ground_z = self.ground_height
        self._band_exit_z: torch.Tensor | None = None   # [num_bands] per-band top-of-stairs z
        self._steer_to_goal = False                      # transition nav (else open-loop)
        self._nav_goal_xy: torch.Tensor | None = None    # [2] fixed goal, local-corner frame
        self._ctx_flat_eligible: torch.Tensor | None = None  # [S] per-terrain flat-skill mask override
        self._ctx_approach_eligible: torch.Tensor | None = None  # [T] per-terrain approach-stride mask
        # Straight-only terrains (transition / terrain): lay EVERY stride in the
        # global (world-aligned, yaw = 0) frame instead of carrying the previous
        # reference's heading forward, so per-stride yaw residuals never
        # accumulate into a drift off the corridor.
        self._straight_only = False
        # Per-clip yaw-correction diagnostic (set per generate_batch when
        # straight-only): the largest heading a leaving stride WOULD have imparted
        # (discarded when snapping to yaw = 0) and which source stride produced it.
        self._diag_yaw_abs: torch.Tensor | None = None
        self._diag_yaw_signed: torch.Tensor | None = None
        self._diag_yaw_stride: torch.Tensor | None = None

        # Skill-transition (seam) handling is pluggable: raw hard-cut, linear
        # cross-fade (online parity, default), or a custom TransitionStrategy.
        # The blend needs the per-frame ori column groups (SLERP), so build them
        # unconditionally.
        self._build_blend_groups()
        self.transition = make_transition_strategy(
            transition, manager=self.manager, total_time=self._total_time,
            ori_groups=self._blend_ori_groups, dt=self.dt,
            blend_end_phi=transition_blend_end_phi, device=self.device,
        )

    # ------------------------------------------------------------------
    # Terrain context
    # ------------------------------------------------------------------

    @property
    def terrain_active(self) -> bool:
        """Whether a terrain context is bound (else the flat rollout runs)."""
        return self._projector is not None

    def set_terrain_context(
        self,
        *,
        projector,
        terrain_dim_lookup,
        band_projectors=None,
        spawn_anchor_pos: torch.Tensor,
        spawn_anchor_quat: torch.Tensor,
        spawn_on_stairs: bool,
        spawn_ground_z: float,
        band_exit_z: Sequence[float],
        terrain_id: int,
        terrain_type: str,
        steer_to_goal: bool = False,
        nav_goal_xy: Sequence[float] | None = None,
        straight_only: bool = False,
        flat_exclude_skills: Sequence[str] = (),
        approach_skills: Sequence[str] | None = None,
    ) -> None:
        """Bind one uniform terrain for the next :meth:`generate_batch`.

        Registers the constant terrain-dim lookup on the manager (so stair
        strides are filtered to this terrain's step dimensions), and stores the
        (multi-band) stair projector + the per-clip spawn state.  Call once per
        terrain batch; :meth:`clear_terrain_context` restores the flat path.

        Args:
            projector: ``project(points[N,3]) -> [N,3]`` snapping a local-corner
                foot pose onto its stair step (a multi-band projector for a
                multi-transition corridor; see ``offline_stair_projector``).
                Kept for compatibility / non-band callers; the rollout uses
                ``band_projectors`` when provided.
            band_projectors: Per-band single-span projectors (one per staircase,
                ascending).  The rollout projects a clip's foot with the
                projector for the band it is climbing / approaching
                (``band_idx``) rather than the multi-band "next stair ahead"
                projector -- which would route a foot that has slightly overshot
                a staircase's top (x past its ``span_xmax``) onto the NEXT
                staircase, teleporting the clip forward over the flat.  ``None``
                falls back to ``projector`` for every band.
            terrain_dim_lookup: ``fn(xy) -> (dims[N,3], valid[N])`` for
                :meth:`MultiSkillManager.set_terrain_dim_lookup` -- a constant
                for a uniform terrain.
            spawn_anchor_pos: ``[3]`` initial reference/foot anchor position
                (local-corner frame).
            spawn_anchor_quat: ``[4]`` initial anchor quaternion (xyzw).
            spawn_on_stairs: Whether clips spawn already on the stairs
                (terrain-only) or on a flat lead-in (transition).
            spawn_ground_z: Flat ground z at spawn (lead-in level).
            band_exit_z: Per-stair-band top-of-stairs z (ascending, one per band).
                A clip that tops band k resumes flat walking at ``band_exit_z[k]``
                and targets band k+1; topping the last band ends the climb.
            terrain_id: Manifest id recorded on every clip in this batch.
            terrain_type: One of ``"terrain"`` / ``"transition"``.
            steer_to_goal: If True (transition), vy/wz are PD-steered toward
                ``nav_goal_xy`` each step; else open-loop sampler (flat/terrain).
            nav_goal_xy: ``[2]`` fixed goal (local-corner), required when
                ``steer_to_goal``.
            straight_only: If True, every stride's anchor is laid in the global
                (world-aligned, yaw = 0) frame -- the previous reference's landing
                heading is NOT carried forward -- so the clip walks perfectly
                straight up ``+x`` and per-stride yaw residuals cannot accumulate
                into a lateral drift off the corridor.  Set for the straight
                (non-steered) ``terrain`` / ``transition`` types.  Also enables the
                per-clip yaw-correction diagnostic.  Mutually exclusive with
                ``steer_to_goal`` (a steered clip must keep its heading).
            flat_exclude_skills: Skill names excluded from the flat-band stride
                selection on this terrain (e.g. ``["running"]`` so a transition's
                slow lead-in never mis-buckets a slow command onto a running
                gait).  Does not affect stair or off-terrain (flat clip) selection.
            approach_skills: Skill names the approach slow-down step may use
                (e.g. ``["walk_forward"]``).  Restricts the approach candidate set
                POSITIVELY by skill -- the single, extensible way to keep the
                slow-down on the intended (forward-walking) gait and out of
                backward / running / off-nominal skills.  ``None`` falls back to
                the flat-band mask (``flat_exclude_skills``).
        """
        if self._stair_skill_id < 0:
            raise ValueError(
                "set_terrain_context requires a terrain-aware skill in the "
                "library (a skill whose trajectories carry terrain "
                "conditioning); none found."
            )
        if steer_to_goal and nav_goal_xy is None:
            raise ValueError("steer_to_goal=True requires nav_goal_xy")
        if steer_to_goal and straight_only:
            raise ValueError("steer_to_goal and straight_only are mutually exclusive")
        self.manager.set_terrain_dim_lookup(terrain_dim_lookup)
        self._projector = projector
        # Per-band single-span projectors (fall back to the multi-band projector
        # for every band when not supplied).
        self._band_projectors = list(band_projectors) if band_projectors is not None else None
        self._spawn_anchor_pos = spawn_anchor_pos.to(self.device)
        self._spawn_anchor_quat = spawn_anchor_quat.to(self.device)
        self._spawn_on_stairs = bool(spawn_on_stairs)
        self._spawn_ground_z = float(spawn_ground_z)
        self._band_exit_z = torch.tensor(list(band_exit_z), device=self.device, dtype=torch.float32)
        self._terrain_id = int(terrain_id)
        self._terrain_type = str(terrain_type)
        self._steer_to_goal = bool(steer_to_goal)
        self._straight_only = bool(straight_only)
        self._nav_goal_xy = (
            torch.tensor(list(nav_goal_xy), device=self.device, dtype=torch.float32)
            if nav_goal_xy is not None else None
        )
        excl = set(flat_exclude_skills)
        if excl:
            keep = torch.tensor(
                [s.name not in excl for s in self.manager.skills], dtype=torch.bool, device=self.device,
            )
            self._ctx_flat_eligible = self._flat_skill_eligible & keep          # [S]
        else:
            keep = None
            self._ctx_flat_eligible = None

        # Approach candidate: restrict POSITIVELY to the designated approach
        # skills when given (the extension point); otherwise fall back to the
        # flat-band exclude mask, or the unrestricted base set.
        if approach_skills is not None:
            known = {s.name for s in self.manager.skills}
            unknown = set(approach_skills) - known
            if unknown:
                raise ValueError(f"approach_skills not in the library: {sorted(unknown)}")
            incl = set(approach_skills)
            appr_keep = torch.tensor(
                [s.name in incl for s in self.manager.skills], dtype=torch.bool, device=self.device,
            )
            self._ctx_approach_eligible = self._approach_eligible & appr_keep[self._traj_skill_idx]  # [T]
        elif keep is not None:
            self._ctx_approach_eligible = self._approach_eligible & keep[self._traj_skill_idx]  # [T]
        else:
            self._ctx_approach_eligible = None

    def clear_terrain_context(self) -> None:
        """Unbind the terrain (restore the flat rollout + selection)."""
        self.manager.set_terrain_dim_lookup(None)
        self._projector = None
        self._band_projectors = None
        self._terrain_id = -1
        self._terrain_type = "flat"
        self._steer_to_goal = False
        self._straight_only = False
        self._nav_goal_xy = None
        self._ctx_flat_eligible = None
        self._ctx_approach_eligible = None
        self._diag_yaw_abs = self._diag_yaw_signed = self._diag_yaw_stride = None

    def _band_project(self, points: torch.Tensor, bands: torch.Tensor):
        """Project each foot with the single-span projector of ITS band.

        ``points`` ``[K, 3]`` foot poses, ``bands`` ``[K]`` the stair band each
        clip is climbing / approaching (``band_idx``).  Using the per-band
        projector (not the multi-band "next stair ahead") keeps a foot that has
        overshot a staircase's top on THAT staircase (it snaps to the band's
        exit level) instead of teleporting onto the next staircase.  Falls back
        to the multi-band projector when per-band projectors were not supplied.
        """
        if self._band_projectors is None:
            return self._projector(points)
        out = torch.empty_like(points)
        for k, proj in enumerate(self._band_projectors):
            mask = bands == k
            if bool(mask.any()):
                out[mask] = proj(points[mask])
        return out

    def _flat_eligible(self, k: int) -> torch.Tensor:
        """``[num_skills, k]`` flat-band skill eligibility for the current terrain."""
        mask = self._ctx_flat_eligible if self._ctx_flat_eligible is not None else self._flat_skill_eligible
        return mask.view(self.num_skills, 1).expand(self.num_skills, k)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _reject_half_periodic(self) -> None:
        """Fail loudly if the library contains any half-periodic trajectory.

        The offline stitcher re-anchors the stance (right-ankle) frame once per
        *full* stride.  A half-periodic trajectory stores a single step and
        generates its other half by mirroring right<->left, so the per-period
        right-ankle re-anchor lands on the wrong foot and the clip snaps /
        teleports at every seam.  The library is expected to be full-periodic
        (or perpetual, e.g. standing); convert half-periodic strides to
        full-periodic before generating clips.
        """
        tt = self.manager.data["traj_type"]
        half = torch.where(tt == _HALF_PERIODIC_INT)[0].tolist()
        if not half:
            return
        names = self.manager.trajectory_names
        labels = self.manager.skill_labels
        shown = ", ".join(f"'{names[i]}' (skill '{labels[i]}')" for i in half[:8])
        more = "" if len(half) <= 8 else f", ... (+{len(half) - 8} more)"
        raise ValueError(
            f"OfflineClipGenerator does not support half-periodic trajectories, "
            f"but the library has {len(half)}/{tt.numel()}: {shown}{more}. The "
            f"offline stitcher re-anchors once per full stride, so half-periodic "
            f"strides (one stored step + a mirrored half) snap/teleport at each "
            f"seam. Convert these trajectories to full_periodic and regenerate."
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_strides(
        self,
        vel_cmd: torch.Tensor,
        anchor_xy: torch.Tensor | None = None,
        eligible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick (skill_id, global stride index) for each clip from a velocity.

        Mirrors the live pipeline: the active skill is the box-nearest
        bucket (``bucket_for_velocity``), then the stride is the velocity
        nearest-neighbour within that skill's block
        (``_select_within_skills``).  ``eligible`` (``[num_skills, n]``)
        restricts the bucket; ``anchor_xy`` activates the terrain-dim filter
        when a terrain-dim lookup is registered on the manager.
        """
        n = vel_cmd.shape[0]
        if eligible is None:
            eligible = self._allowed.view(self.num_skills, 1).expand(self.num_skills, n)
        skill = bucket_for_velocity(vel_cmd, eligible, self._lin_x, self._lin_y, self._ang_z)
        stride = self.manager._select_within_skills(vel_cmd, skill, anchor_xy=anchor_xy)
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
    # Terrain rollout helpers
    # ------------------------------------------------------------------

    def _init_terrain_state(
        self, n: int, vel: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Initial (skill, stride, anchor_pos, anchor_quat, on_stairs, ground_z, done, band_idx)."""
        dev = self.device
        anchor_pos = self._spawn_anchor_pos.unsqueeze(0).repeat(n, 1)     # [n, 3]
        anchor_quat = self._spawn_anchor_quat.unsqueeze(0).repeat(n, 1)   # [n, 4]
        on_stairs = torch.full((n,), self._spawn_on_stairs, dtype=torch.bool, device=dev)
        ground_z = torch.full((n,), self._spawn_ground_z, device=dev)
        # ``done`` latches once a clip climbs out the LAST band so it never
        # re-enters (else the exit-level projection re-commits it endlessly);
        # ``band_idx`` is the stair band a clip is climbing / approaching.
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        band_idx = torch.zeros(n, dtype=torch.long, device=dev)
        if self._spawn_on_stairs:
            stair_vec = torch.full((n,), self._stair_skill_id, dtype=torch.long, device=dev)
            stride = self.manager._select_within_skills(vel, stair_vec, anchor_xy=anchor_pos[:, :2])
            dom = self.manager._get_domain_indices(torch.zeros(n, device=dev), stride)
            # Spawn is always on the first band (band_idx = 0).
            band0_project = lambda pts: self._band_project(pts, band_idx)
            anchor_pos = stair_stance_ref(self.manager, band0_project, stride, anchor_pos, dom)
            skill = self._traj_skill_idx[stride]
        else:
            skill, stride = self._select_strides(vel, anchor_xy=None, eligible=self._flat_eligible(n))
            anchor_pos = anchor_pos.clone()
            anchor_pos[:, 2] = ground_z + self._ref_offset_z[stride]
        return skill, stride, anchor_pos, anchor_quat, on_stairs, ground_z, done, band_idx

    def _apply_goal_steer(
        self,
        vel: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_quat: torch.Tensor,
        on_stairs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Overwrite vy/wz with the goal-directed PD (transition clips only).

        Forward velocity (vx = ``vel[:, 0]``) is left as sampled; the lateral and
        yaw channels are set by :func:`goal_pd_commands` steering the current
        stance anchor toward the fixed ``self._nav_goal_xy``.  Steering is
        **suspended (forced straight)** for clips currently on the stairs so the
        swing foot stays on the narrow treads.  No-op when the terrain is not a
        steer-to-goal (transition) context, so ``flat``/``terrain`` keep the pure
        open-loop sampler output.
        """
        if not self._steer_to_goal:
            return vel
        n = vel.shape[0]
        # anchor_quat is yaw-only (xyzw): yaw = 2*atan2(qz, qw).
        heading = 2.0 * torch.atan2(anchor_quat[:, 2], anchor_quat[:, 3])
        goal = self._nav_goal_xy.unsqueeze(0).expand(n, 2)
        vy, wz = goal_pd_commands(
            anchor_pos[:, :2], heading, goal,
            self._nav_gains.expand(n, 3), self._nav_vy.expand(n, 2), self._nav_wz.expand(n, 2),
        )
        vel = vel.clone()
        vel[:, 1] = vy
        vel[:, 2] = wz
        # Straight (no steering) while climbing (feet stay on the narrow treads)
        # and once at/past the goal in x (else the PD turns the robot around to
        # chase a now-behind goal and spirals off the terrain).
        straight = anchor_pos[:, 0] >= self._nav_goal_xy[0]
        if on_stairs is not None:
            straight = straight | on_stairs
        if bool(straight.any()):
            vel[straight, 1] = 0.0
            vel[straight, 2] = 0.0
        return vel

    def _terrain_boundary(
        self,
        ids: torch.Tensor,
        vel: torch.Tensor,
        phase: torch.Tensor,
        overshoot_s: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_quat: torch.Tensor,
        stride_idx: torch.Tensor,
        skill_id: torch.Tensor,
        on_stairs: torch.Tensor,
        ground_z: torch.Tensor,
        terrain_done: torch.Tensor,
        band_idx: torch.Tensor,
    ) -> None:
        """Commit the next stride for crossing terrain clips (in place).

        State machine per clip, per stair band (flat -> approach -> climb -> top):

        * on the flat before band ``band_idx`` (not on stairs): the nominal
          sampled gait continues until the next nominal step would overshoot the
          stair stance reference; then a shortened straight walk step
          (:func:`plan_slowdown_step`) lands closest to the entry without
          stepping in, or the stair skill is committed (within
          ``approach_threshold``, or when no shorter step fits);
        * climbing band k: each stride re-anchors onto the tread the reference
          foot rests on (z climbs) until it reaches ``band_exit_z[k]``; then flat
          walking resumes at that height and ``band_idx`` advances to k+1 (the
          multi-span projector auto-targets the next staircase).  Topping the
          LAST band latches ``terrain_done`` so the clip never re-enters.
        """
        dev = self.device
        K = ids.shape[0]
        v = vel[ids]                                                     # [K, 3]
        cur_stairs = on_stairs[ids]                                      # [K] bool
        cur_ground = ground_z[ids]                                       # [K]
        cur_done = terrain_done[ids]                                     # [K] bool (climbed out)
        cur_band = band_idx[ids]                                         # [K] long

        # Old stride's stance-foot landing (world); z carries the climb.
        land_pos, land_quat = self._stance_landing_world(
            stride_idx[ids], anchor_pos[ids], anchor_quat[ids],
        )

        # Straight-only diagnostic: ``land_quat`` is the heading the leaving stride
        # (traj A) imparts relative to its (identity) start anchor -- exactly the
        # yaw we discard below when snapping to global 0.  Record, per clip, the
        # largest such correction and the stride that produced it, so a crooked
        # source stride (one that ends several degrees off straight) is findable.
        if self._straight_only and self._diag_yaw_abs is not None:
            leaving = stride_idx[ids]                                     # traj A (being left)
            corr = 2.0 * torch.atan2(land_quat[:, 2], land_quat[:, 3])    # [K] yaw of yaw-only quat
            cur = self._diag_yaw_abs[ids]
            better = corr.abs() > cur
            self._diag_yaw_abs[ids] = torch.where(better, corr.abs(), cur)
            self._diag_yaw_signed[ids] = torch.where(better, corr, self._diag_yaw_signed[ids])
            self._diag_yaw_stride[ids] = torch.where(better, leaving, self._diag_yaw_stride[ids])

        # Flat candidate (velocity-bucketed among allowed flat skills).
        _, flat_stride = self._select_strides(v, anchor_xy=None, eligible=self._flat_eligible(K))

        # Stair candidate (nearest stair stride at the landed xy; the terrain-dim
        # filter narrows it to this terrain's step dims).
        stair_vec = torch.full((K,), self._stair_skill_id, dtype=torch.long, device=dev)
        stair_stride = self.manager._select_within_skills(v, stair_vec, anchor_xy=land_pos[:, :2])
        dom = self.manager._get_domain_indices(torch.zeros(K, device=dev), stair_stride)
        # Project the foot with THIS clip's band (band_idx) -- not the multi-band
        # "next stair ahead" -- so a foot that overshot the top of band cur_band
        # snaps to that band's exit level rather than teleporting onto band+1.
        band_project = lambda pts: self._band_project(pts, cur_band)
        stair_ref = stair_stance_ref(self.manager, band_project, stair_stride, land_pos, dom)  # [K, 3]

        # Forward distance from the landed foot to the stair stance reference.
        fwd = quat_apply(yaw_quat(land_quat), self._fwd_local.expand(K, 3))   # [K, 3]
        tgt = ((stair_ref[:, :2] - land_pos[:, :2]) * fwd[:, :2]).sum(dim=1)  # [K]

        # Approach slow-down decision (only meaningful for flat clips).  Uses the
        # per-terrain approach mask when set (e.g. excludes running so the
        # slow-down never picks a big running step for a large gap).
        appr_elig = self._ctx_approach_eligible if self._ctx_approach_eligible is not None else self._approach_eligible
        elig = appr_elig.unsqueeze(0).expand(K, -1)                          # [K, T]
        do_appr, forced = plan_slowdown_step(
            tgt, self._traj_step_length, elig, self.approach_threshold,
        )

        # The nominal (sampled) gait's forward reach; the approach only triggers
        # when this nominal step would overshoot the stair stance reference (i.e.
        # the desired footstep would land into the stair) -- otherwise keep the
        # nominal gait.  Standing / gate-less strides (inf reach) don't advance,
        # so they never "overshoot".  Mirrors the online next-gate approach gate.
        nominal_step = self._traj_step_length[flat_stride]                   # [K]
        would_overshoot = torch.isfinite(nominal_step) & (nominal_step > tgt)

        # Per-band top: compare the landed z to THIS band's exit height.
        band_exit = self._band_exit_z[cur_band]                              # [K]
        last_band = self._band_exit_z.numel() - 1
        topped = land_pos[:, 2] >= (band_exit - 1e-3)                        # reached this band's top
        # ``cur_done`` clips have already climbed out the LAST band -- they never
        # re-enter (the exit-level projection would otherwise re-commit them).
        # Commit the climb when already at the entry, or when the nominal step
        # would overshoot but no shorter straight step fits (can't land cleaner).
        commit_stair = (~cur_stairs) & (~cur_done) & ((tgt <= self.approach_threshold) | (would_overshoot & ~do_appr))
        # Adjust (slow-down) only when the nominal step would overshoot AND a
        # shorter straight step fits: pick the longest straight step not past the
        # entry (``plan_slowdown_step`` / ``forced``) so the foot lands closest to
        # the nominal stair start without stepping into the stair.
        approach = (~cur_stairs) & (~cur_done) & (~commit_stair) & would_overshoot & do_appr
        stay_stairs = cur_stairs & (~topped)
        use_stair = commit_stair | stay_stairs
        just_topped = cur_stairs & topped

        # Chosen stride: stair (climb/commit) > forced walk (approach) > flat.
        new_stride = torch.where(use_stair, stair_stride, flat_stride)
        new_stride = torch.where(approach, forced, new_stride)
        new_skill = self._traj_skill_idx[new_stride]

        # Anchor: stair strides snap onto the tread (z climbs); flat strides pin
        # to the current ground (this band's connecting level once topped).
        new_ground = torch.where(just_topped, band_exit, cur_ground)
        flat_pos = land_pos.clone()
        flat_pos[:, 2] = new_ground + self._ref_offset_z[new_stride]
        new_pos = torch.where(use_stair.unsqueeze(-1), stair_ref, flat_pos)

        # Straight-only terrain (transition / terrain): lay EVERY stride in the
        # global (world-aligned, yaw = 0) frame -- do NOT carry the leaving
        # stride's landing heading forward -- so per-stride yaw residuals never
        # accumulate into a drift.  Otherwise: stair anchors use an identity
        # (world-aligned) quat -- matching the online ``apply_terrain_aware_ref``
        # -- so the climb stays axis-aligned and doesn't inherit any approach-walk
        # yaw drift, while flat strides keep the yaw-only landing quat for heading
        # continuity.
        if self._straight_only:
            new_quat = torch.zeros_like(land_quat)
            new_quat[:, 3] = 1.0
        else:
            new_quat = land_quat.clone()
            new_quat[use_stair] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)

        anchor_pos[ids] = new_pos
        anchor_quat[ids] = new_quat
        stride_idx[ids] = new_stride
        skill_id[ids] = new_skill
        on_stairs[ids] = use_stair
        ground_z[ids] = new_ground
        # Topping a non-last band advances to the next; topping the last latches done.
        is_last = cur_band >= last_band
        terrain_done[ids] = cur_done | (just_topped & is_last)
        band_idx[ids] = torch.where(just_topped & ~is_last, cur_band + 1, cur_band)
        phase[ids] = (overshoot_s / self._total_time[new_stride]).clamp_(min=0.0)

    def _print_yaw_diag(self, n: int) -> None:
        """Print, per straight-only clip, the max discarded yaw + its source stride.

        Straight-only clips snap every stride to global 0 yaw; this reports, for
        each clip, the largest heading a leaving stride WOULD have imparted (the
        yaw corrected away) and which source trajectory produced it -- so a source
        stride that ends several degrees off straight (feeding the drift) is easy
        to locate.  ``stride #`` is the global stride index into the library.
        """
        if self._diag_yaw_stride is None:
            return
        names = self.manager.trajectory_names
        labels = self.manager.skill_labels
        deg = torch.rad2deg(self._diag_yaw_signed)
        print(
            f"  [yaw-diag] terrain {self._terrain_id} ({self._terrain_type}): "
            f"max discarded yaw per clip (corrected to global 0):",
            flush=True,
        )
        for i in range(n):
            st = int(self._diag_yaw_stride[i])
            if st < 0:
                print(f"    clip[{i:>3}]: (no stride boundary crossed)")
                continue
            print(
                f"    clip[{i:>3}]: {float(deg[i]):+7.2f} deg  <- traj '{names[st]}' "
                f"(skill '{labels[st]}', stride #{st})"
            )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def generate_batch(self, n_clips: int) -> dict[str, torch.Tensor | dict]:
        """Generate ``n_clips`` clips as a batched control-step rollout.

        Returns:
            Dict with dense tensors (all on ``self.device``):
              ``y_des`` ``[n, T, P]``, ``dy_des`` ``[n, T, V]`` --
                  clip-world reference outputs (per-step stance anchor baked in);
              ``contact`` ``[n, T, C]`` uint8 -- expected contact schedule;
              ``skill_id`` ``[n, T]`` uint8, ``stride_idx`` ``[n, T]`` int32 --
                  per-step labels;
              ``target_vel`` ``[n, T, 3]`` -- ramped (vx, vy, wz) command;
              ``terrain_id`` ``[n]`` int32 -- manifest id (-1 for flat);
            plus a ``meta`` dict (dt, clip length, output/contact names) and the
            batch's ``terrain_type`` string.
        """
        n, T = int(n_clips), self.num_steps
        dev, m = self.device, self.manager
        P, V, C = self.num_pos, self.num_vel, len(self.contact_frames)
        terrain = self.terrain_active

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
        if terrain:
            skill_id, stride_idx, anchor_pos, anchor_quat, on_stairs, ground_z, terrain_done, band_idx = (
                self._init_terrain_state(n, vel)
            )
            vel = self._apply_goal_steer(vel, anchor_pos, anchor_quat, on_stairs)
            if self._straight_only:
                self._diag_yaw_abs = torch.zeros(n, device=dev)
                self._diag_yaw_signed = torch.zeros(n, device=dev)
                self._diag_yaw_stride = torch.full((n,), -1, dtype=torch.long, device=dev)
            else:
                self._diag_yaw_abs = self._diag_yaw_signed = self._diag_yaw_stride = None
        else:
            skill_id, stride_idx = self._select_strides(vel)  # [n], [n]
            anchor_pos = torch.zeros(n, 3, device=dev)
            if self.project_to_ground:
                anchor_pos[:, 2] = self._anchor_ground_z(stride_idx)  # first landing's natural ankle height
            anchor_quat = torch.zeros(n, 4, device=dev)
            anchor_quat[:, 3] = 1.0                          # identity (xyzw, w last)
            on_stairs = ground_z = terrain_done = band_idx = None
        phase = torch.zeros(n, device=dev)

        # Reset the skill-transition strategy for this batch.  Seams are armed
        # only on the flat path below; terrain-clip seams are handled by the stair
        # state machine (a hard cut -- a stair commit jumps the anchor up a tread,
        # so a continued old stride would float mid-air), so the strategy's
        # ``apply`` no-ops on terrain (nothing armed).
        self.transition.reset(n)

        all_ids = torch.arange(n, device=dev)
        for t in range(T):
            # 1. Store the current frame (pre-advance state).
            m.set_trajectory_indices(stride_idx)
            ypos, yvel = m.get_output(phase)
            contact = m.get_contact_state(phase, self.contact_frames)

            # 1a. Skill-transition blend for clips mid-transition (raw = no-op;
            # linear = cross-fade the old stride continued into the new).  The old
            # stride re-anchors to the same landing as the new, so both are
            # anchor-relative in the same frame and blend directly.  (Done after
            # contact -- it mutates the manager's cached indices, which the top of
            # the next iter resets.)
            ypos, yvel = self.transition.apply(ypos, yvel)

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

            # 2. Advance velocity ramp one control step, then (transition only)
            # overwrite vy/wz with the goal-directed PD toward the fixed nav goal.
            vel = self.sampler.step()
            vel = self._apply_goal_steer(vel, anchor_pos, anchor_quat, on_stairs)

            # 3. Advance phase; handle stride-boundary crossings.
            old_tt = self._total_time[stride_idx]            # [n]
            phase = phase + self.dt / old_tt
            crossed = phase >= 1.0
            if crossed.any():
                ids = all_ids[crossed]
                # Time overshoot past the old stride end (seconds) -- carried
                # into the new stride so the global timeline stays uniform dt.
                overshoot_s = (phase[ids] - 1.0) * old_tt[ids]
                if terrain:
                    self._terrain_boundary(
                        ids, vel, phase, overshoot_s, anchor_pos, anchor_quat,
                        stride_idx, skill_id, on_stairs, ground_z, terrain_done, band_idx,
                    )
                    continue
                # --- Flat path ---
                # Arm the transition: the old stride keeps going into its next
                # cycle (re-anchored to the new landing == the new anchor), so its
                # continuation phase is the overshoot past 1.0.  ``raw`` skips this.
                if self.transition.arms:
                    self.transition.arm(ids, stride_idx[ids], phase[ids] - 1.0)
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

        # Terrain-only clips must climb for the WHOLE clip -- they should never
        # top out and fall back to a flat gait.  If any did, the staircase is too
        # short for the clip length / velocity: shout loudly so it gets fixed.
        if self._terrain_type == "terrain" and terrain_done is not None and bool(terrain_done.any()):
            n_top = int(terrain_done.sum())
            bar = "!" * 78
            print(
                f"\n{bar}\n"
                f"!!  WARNING: {n_top}/{n} terrain-only clip(s) on terrain {self._terrain_id} "
                f"REACHED THE TOP\n"
                f"!!  of the staircase and fell back to a FLAT gait mid-clip.\n"
                f"!!  Terrain-only clips are meant to climb for the entire clip. The staircase\n"
                f"!!  is too short for --clip-length at the sampled stair velocity.\n"
                f"!!  Fix: increase --transition-length (terrain length) or lower --clip-length.\n"
                f"{bar}\n"
            )

        # Straight-only per-clip yaw-correction diagnostic (where the drift comes
        # from): the largest heading each clip's leaving strides would have
        # imparted, now corrected to global 0, with the source stride.
        if terrain and self._straight_only:
            self._print_yaw_diag(n)

        # Bake the per-step stance anchor into the stored outputs so a clip is a
        # single clip-world frame (the runtime applies only a per-env placement).
        bake_outputs_to_world(
            y_des, dy_des, anchor_pos_buf, anchor_quat_buf,
            self.pos_output_names, self.vel_output_names,
        )

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
            "contact": contact_buf,
            "skill_id": skill_buf,
            "stride_idx": stride_buf,
            "target_vel": target_vel_buf,
            "terrain_id": torch.full((n,), self._terrain_id, dtype=torch.int32, device=dev),
            "terrain_type": self._terrain_type,
            "meta": meta,
        }
