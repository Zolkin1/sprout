# Copyright 2026 Zachary Olkin. All rights reserved.

"""Dense-clip manager backing :class:`MotionSequenceCommand`.

Wraps a loaded :class:`ClipLibraryData` and exposes the minimal
:class:`ManagerBase` surface the trajectory-command base class needs at
construction (output names, ordering, reference frames, trajectory type),
plus a per-env ``gather`` of the dense clip-world reference by
``(clip_id, step)``.  All the online machinery (Bezier eval, domains,
contact gates, skill selection) is absent -- a clip is tracked open-loop by
a pure step index, so the manager is a gather, not an evaluator.
"""

from __future__ import annotations

import torch

from ..manager_base import ManagerBase
from ..manager_base import TrajectoryType
from .clip_io import ClipLibraryData


class MotionClipManager(ManagerBase):
    """Gather-based manager over a dense motion-clip library.

    Args:
        lib: Loaded dense clip library (tensors already on the target device).
    """

    def __init__(self, lib: ClipLibraryData) -> None:
        self.lib = lib
        self.device = lib.y_des.device

        self.pos_output_names: list[str] = list(lib.meta["pos_output_names"])
        self.vel_output_names: list[str] = list(lib.meta["vel_output_names"])
        self.num_pos_outputs = len(self.pos_output_names)
        self.num_vel_outputs = len(self.vel_output_names)
        # ``ref_frame`` names the frame a clip's poses are expressed RELATIVE TO -- the base frame
        # (e.g. ``pelvis_link``).  Do NOT confuse it with how a clip is ANCHORED: that is the
        # per-DOF ``anchor_mask`` below (world-pinned vs periodically ego-anchored).  The old
        # foot-STANCE paradigm conflated the two by re-anchoring to a planted foot, and it is gone
        # (see base_trajectory_cmd.py); this name is NOT a vestige of it and is still load-bearing
        # -- it feeds ``get_reference_frames`` -> ``_parse_ref_frames`` -> ``ref_poses``, which the
        # teacher reuses as the clip world-placement scratch.
        self.ref_frame: str = lib.meta["ref_frame"]
        self.contact_frames: list[str] = list(lib.meta["contact_frames"])
        self.dt: float = float(lib.meta["dt"])
        self.num_steps: int = int(lib.meta["num_steps"])
        self.num_clips: int = lib.num_clips

        # Per-clip root-DOF anchor bitmask (bit SET = FREE). A pre-mask library
        # resolves to the legacy mask for every clip here (single interpretation
        # point), so downstream re-anchoring never special-cases missing masks.
        self.anchor_mask: torch.Tensor = lib.anchor_mask_or_legacy(self.device)  # [num_clips] uint8
        # Per-clip REAL length (valid frames before padding); full ``num_steps`` for legacy libs.
        self.clip_len: torch.Tensor = lib.clip_len_or_full(self.device)          # [num_clips] long
        # Per-STEP root-DOF anchor bitmask [num_clips, T] for clips whose anchoring varies
        # WITHIN the clip (box clips: y-free-pinned approach/jump, then ego recovery). For
        # clips without a per-step schedule this broadcasts the per-clip mask across T, so
        # re-anchoring reads one uniform [N, T] regardless of clip type.
        self.anchor_mask_t: torch.Tensor = lib.anchor_mask_per_step(self.device)  # [num_clips, T] uint8
        # First and last frame of each clip's terrain-PINNED run -- the extent of its terrain
        # sequence.  Computed from ONE ``pinned`` mask so the two can never disagree.
        span = self._compute_terrain_span()
        self.terrain_start_step: torch.Tensor = span[0]                           # [num_clips] long
        self.terrain_end_step: torch.Tensor = span[1]                             # [num_clips] long

        # Per-env phase carrier (set by the owning command each step so the
        # base ``_update_command`` can read ``manager.phase``).  Allocated by
        # the command via :meth:`ensure_state`.
        self.phase: torch.Tensor | None = None

    def _compute_terrain_span(self) -> tuple[torch.Tensor, torch.Tensor]:
        """First and last frame index of each clip's terrain-PINNED run.

        A transition or box clip is a sandwich -- ego approach, a pinned run over the
        terrain (the stair climb, the box mount or dismount), then ego walking on the raised
        top or away from it.  Only the middle run is registered to the terrain geometry, so
        its first and last frames bracket the clip's terrain sequence.

        Clips with no pinned frame at all (flat) report ``(-1, clip_len - 1)``.  The ``-1``
        start is what distinguishes "no terrain span" from "pinned from frame 0" -- which
        the end alone cannot, because a clip pinned to its last frame and a clip with no
        pinned frame both report ``clip_len - 1``.  The ``clip_len - 1`` end keeps every
        existing consumer a no-op for such clips rather than needing a special case.

        Two things this must get right, both of which have a wrong-but-plausible shortcut:

        * it reads :attr:`anchor_mask_t`, which is ``ClipLibraryData.anchor_mask_per_step``
          and has already resolved the ``255 = unset`` sentinel against the per-clip mask.
          The RAW ``anchor_mask_t`` dataset must never be used here -- in its low 6 bits
          ``255`` is indistinguishable from "every DOF free".
        * it masks by :attr:`clip_len` first.  Clips are repeat-last padded, so the tail
          repeats the final valid mask; an unmasked reverse scan would run a clip that ends
          pinned all the way out to ``num_steps - 1``.

        Returns:
            ``(start, end)``, each ``[num_clips]`` long on the library device.
        """
        from ..world_lift import anchor_mask_is_pinned

        dev = self.anchor_mask_t.device
        t = self.anchor_mask_t.shape[1]
        valid = torch.arange(t, device=dev).unsqueeze(0) < self.clip_len.to(dev).unsqueeze(1)
        pinned = anchor_mask_is_pinned(self.anchor_mask_t) & valid   # [N, T]
        any_pinned = pinned.any(dim=1)
        as_u8 = pinned.to(torch.uint8)
        first = torch.argmax(as_u8, dim=1)
        # Reverse argmax = index of the LAST True; flip the index back into forward order.
        last = (t - 1) - torch.argmax(as_u8.flip(1), dim=1)
        fallback_end = self.clip_len.to(dev) - 1
        start = torch.where(any_pinned, first, torch.full_like(first, -1)).long()
        end = torch.where(any_pinned, last, fallback_end).long()
        return start, end

    # ------------------------------------------------------------------
    # ManagerBase surface used at command construction
    # ------------------------------------------------------------------

    @property
    def get_pos_output_names(self) -> list[str]:
        """Position output names (includes ori_w)."""
        return self.pos_output_names

    @property
    def get_vel_output_names(self) -> list[str]:
        """Velocity output names (excludes ori_w)."""
        return self.vel_output_names

    def get_num_pos_outputs(self) -> int:
        return self.num_pos_outputs

    def get_num_vel_outputs(self) -> int:
        return self.num_vel_outputs

    def get_reference_frames(self) -> list[str]:
        """The single stance reference frame the clips are anchored to."""
        return [self.ref_frame]

    def get_trajectory_type(self) -> TrajectoryType:
        """Clips are fixed sequences; report EPISODIC (phase advance is owned
        by the command, so the type only labels the run)."""
        return TrajectoryType.EPISODIC

    def get_total_time(self) -> torch.Tensor:
        """Total clip duration in seconds (``num_steps * dt``).

        Mirrors :meth:`MultiSkillManager.get_total_time`; consumed by the
        parameter-export tooling (``scripts/rsl_rl/export_parameters.py``).
        """
        return torch.tensor(self.num_steps * self.dt, device=self.device)

    def order_outputs(self, pos_output_names: list[str], vel_output_names: list[str]) -> None:
        """Permute the dense output dimension to the command's output order."""
        pos_perm = torch.tensor(
            [self.pos_output_names.index(n) for n in pos_output_names],
            dtype=torch.long, device=self.device,
        )
        vel_perm = torch.tensor(
            [self.vel_output_names.index(n) for n in vel_output_names],
            dtype=torch.long, device=self.device,
        )
        self.lib.y_des = self.lib.y_des[:, :, pos_perm]
        self.lib.dy_des = self.lib.dy_des[:, :, vel_perm]
        self.pos_output_names = list(pos_output_names)
        self.vel_output_names = list(vel_output_names)

    def invalidate_cache(self) -> None:
        """No per-step trajectory-assignment cache for fixed clips (no-op)."""

    # ------------------------------------------------------------------
    # Per-env gather by (clip_id, step)
    # ------------------------------------------------------------------

    def gather(self, tensor: torch.Tensor, clip_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Index a dense lib tensor ``[num_clips, T, ...]`` by ``(clip_id, step)``.

        Returns the result on ``clip_id``'s device, moving the index tensors to
        the library's device first.  This lets a CPU-held library
        (``cfg.clip_device='cpu'``, for large libs on small GPUs) work
        transparently with a CUDA env -- a per-step gather copies only the tiny
        per-env slice across the bus.  Same-device (GPU-resident lib) is a no-op.
        Works for ``[N]`` indices (``[N, ...]`` result) and ``[N, K]`` indices
        (``[N, K, ...]`` result).
        """
        out_dev = clip_id.device
        lib_dev = tensor.device
        out = tensor[clip_id.to(lib_dev), step.to(lib_dev)]
        return out.to(out_dev)

    def gather_desired(self, clip_id: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y_des, dy_des)`` ``[N, P]`` / ``[N, V]`` (clip-world) for each env."""
        return self.gather(self.lib.y_des, clip_id, step), self.gather(self.lib.dy_des, clip_id, step)

    def gather_contact(self, clip_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Return the stored contact schedule ``[N, C]`` (clip ``contact_frames`` order)."""
        return self.gather(self.lib.contact, clip_id, step)

    def gather_target_vel(self, clip_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Return the commanded base velocity ``[N, 3]`` (vx, vy, wz) for each env.

        This is the upstream command that generated the clip (not the achieved
        ``dy_des`` velocity).  Indexing matches :meth:`gather_desired`.
        """
        return self.gather(self.lib.target_vel, clip_id, step)

    def has_target_vel(self) -> bool:
        """Whether the loaded library carries a per-step commanded velocity."""
        return self.lib.target_vel.numel() > 0

    def gather_anchor_mask(self, clip_id: torch.Tensor) -> torch.Tensor:
        """Return the per-clip root-DOF anchor bitmask ``[N]`` uint8 for each env's clip.

        Step-invariant (a clip's mask is constant over its length), so this takes
        only ``clip_id``.  Follows the same CPU-lib / CUDA-env device shuttle as
        :meth:`gather`.
        """
        return self.anchor_mask[clip_id.to(self.anchor_mask.device)].to(clip_id.device)

    def gather_anchor_mask_at_step(self, clip_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Return the CURRENT-frame root-DOF anchor bitmask ``[N]`` uint8 per env.

        Indexes the per-step schedule ``anchor_mask_t[clip_id, step]`` so re-anchoring can
        vary WITHIN a clip (a box clip's y-free-pinned approach/jump vs. its ego recovery).
        For clips without a per-step schedule this equals :meth:`gather_anchor_mask` (the
        per-clip mask broadcast across the clip length).
        """
        return self.gather(self.anchor_mask_t, clip_id, step)

    def gather_terrain_end_step(self, clip_id: torch.Tensor) -> torch.Tensor:
        """Return the last terrain-PINNED frame ``[N]`` long for each env's clip.

        Step-invariant, so this takes only ``clip_id``.  Follows the same CPU-lib / CUDA-env
        device shuttle as :meth:`gather`.  See :meth:`_compute_terrain_span`.
        """
        return self.terrain_end_step[clip_id.to(self.terrain_end_step.device)].to(clip_id.device)

    def gather_terrain_start_step(self, clip_id: torch.Tensor) -> torch.Tensor:
        """Return the FIRST terrain-pinned frame ``[N]`` long for each env's clip.

        ``-1`` where the clip has no pinned frame at all.  Step-invariant, so this takes
        only ``clip_id``.  See :meth:`_compute_terrain_span`.
        """
        return self.terrain_start_step[clip_id.to(self.terrain_start_step.device)].to(clip_id.device)

    # ------------------------------------------------------------------
    # ManagerBase abstract methods (trivial for fixed clips; the command's
    # overrides drive the lifecycle, so these are defensive defaults).
    # ------------------------------------------------------------------

    def get_output(self, t: torch.Tensor):  # noqa: D401 - interface stub
        """Not used: the command gathers via :meth:`gather_desired`."""
        raise NotImplementedError("MotionClipManager: use gather_desired(clip_id, step).")

    def get_domain_times(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.num_steps * self.dt)

    def get_current_domains(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t, dtype=torch.long)

    def get_num_domains(self) -> int:
        return 1

    def get_ref_frames_in_use(self, t: torch.Tensor, ref_frames: list[str]) -> torch.Tensor:
        return torch.zeros_like(t, dtype=torch.long)

    def get_contact_state(self, t: torch.Tensor, contact_frames: list[str]) -> torch.Tensor:
        """Defensive stub: the command overrides contact via gather_contact."""
        return torch.zeros(t.shape[0], len(contact_frames), device=self.device)
