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
from ..trajectory_manager import TrajectoryType
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
        self.ref_frame: str = lib.meta["ref_frame"]
        self.contact_frames: list[str] = list(lib.meta["contact_frames"])
        self.dt: float = float(lib.meta["dt"])
        self.num_steps: int = int(lib.meta["num_steps"])
        self.num_clips: int = lib.num_clips

        # Per-env phase carrier (set by the owning command each step so the
        # base ``_update_command`` can read ``manager.phase``).  Allocated by
        # the command via :meth:`ensure_state`.
        self.phase: torch.Tensor | None = None

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
