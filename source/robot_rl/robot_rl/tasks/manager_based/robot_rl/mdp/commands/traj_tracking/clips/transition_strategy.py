# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pluggable skill-transition (seam) strategies for the offline clip stitcher.

When the offline generator crosses a stride/skill boundary it has to decide how
the reference behaves across the seam.  This module makes that decision an
*extensible* choice rather than hard-wired logic:

* :class:`RawStitchTransition` -- a hard cut; the new stride simply replaces the
  old at the seam (the original raw-stitch behavior, kept as an option).
* :class:`LinearInterpolationTransition` -- a cross-fade that mirrors the ONLINE
  ``CrossfadeState`` blend (:mod:`skill_transition`): over ``blend_end_phi`` worth
  of phase the reference is a convex blend of the old and new strides, with
  orientation quaternions SLERPed.  This is the default.

A future technique (e.g. motion matching) only needs to subclass
:class:`TransitionStrategy` and implement the three hooks the generator calls --
:meth:`reset` (start of a batch), :meth:`arm` (a seam was crossed), and
:meth:`apply` (each step, to optionally rewrite the stored reference).  Pass an
instance straight to the generator (or extend :func:`make_transition_strategy`)
to wire it in; no change to the rollout is needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..skill_transition import blend_outputs


class TransitionStrategy(ABC):
    """Base class for offline skill-transition (seam) handling.

    The offline rollout drives a strategy through three hooks:

    * :meth:`reset` once per ``generate_batch`` (allocate/clear per-clip state);
    * :meth:`arm` when a seam is crossed for some clips (begin a transition);
    * :meth:`apply` every step (optionally rewrite the current-step outputs for
      clips that are mid-transition).

    Subclasses implement a concrete technique.  ``arms`` lets a no-op strategy
    (raw) tell the generator to skip the seam-arming bookkeeping entirely.
    """

    name: str = "base"

    @property
    def arms(self) -> bool:
        """Whether seams should be armed (``False`` lets the generator skip it)."""
        return True

    @abstractmethod
    def reset(self, n: int) -> None:
        """Allocate/clear per-clip transition state for a fresh ``n``-clip batch."""

    @abstractmethod
    def arm(self, env_ids: torch.Tensor, old_stride: torch.Tensor, old_phase: torch.Tensor) -> None:
        """Begin a transition for ``env_ids`` leaving ``old_stride`` at ``old_phase``.

        Args:
            env_ids: Clips that just crossed a seam.
            old_stride: Global index of the stride each clip is leaving.
            old_phase: Phase to continue the old stride from (the overshoot past
                1.0, so it flows into its next cycle).
        """

    @abstractmethod
    def apply(self, ypos: torch.Tensor, yvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite the current-step outputs for clips mid-transition (may edit in place).

        Args:
            ypos: ``[n, P]`` current-step position outputs (new stride, anchor-relative).
            yvel: ``[n, V]`` current-step velocity outputs.

        Returns:
            ``(ypos, yvel)`` after applying the transition technique.
        """


class RawStitchTransition(TransitionStrategy):
    """Hard cut: the new stride replaces the old at the seam, no blending."""

    name = "raw"

    @property
    def arms(self) -> bool:
        return False

    def reset(self, n: int) -> None:
        pass

    def arm(self, env_ids: torch.Tensor, old_stride: torch.Tensor, old_phase: torch.Tensor) -> None:
        pass

    def apply(self, ypos: torch.Tensor, yvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ypos, yvel


class LinearInterpolationTransition(TransitionStrategy):
    """Cross-fade the old stride into the new over a phi window (online parity).

    Mirrors the online :class:`~..skill_state.CrossfadeState` blend: for
    ``blend_end_phi`` worth of phase after a seam the stored reference is
    ``(1 - alpha) * old + alpha * new`` with ``alpha = phi_elapsed / blend_end_phi``
    clamped to ``[0, 1]`` (``0`` at the seam -> pure old, ``1`` at the window end
    -> pure new); orientation quaternions are SLERPed rather than linearly blended
    (via :func:`~..skill_transition.blend_outputs`).

    The old stride is continued on its own phase clock (decoupled), re-anchored to
    the same landing pose as the new stride -- so both outputs are anchor-relative
    in one frame and blend directly.  This matches the online *decouple* branch and
    is the correct choice for the offline full-periodic library: there is no
    half-periodic reflection boundary to keep aligned, so the online sync branch is
    unnecessary.  (The offline stitcher rejects half-periodic trajectories.)

    Args:
        manager: The :class:`MultiSkillManager` (to evaluate the old stride).
        total_time: ``[T]`` per-trajectory stride duration (for the phase delta).
        ori_groups: ``[F, 4]`` orientation column groups to SLERP (from
            ``OfflineClipGenerator._build_blend_groups``).
        dt: Control-step dt (s).
        blend_end_phi: Phi window over which alpha ramps 0 -> 1 (> 0).
        device: Torch device.
    """

    name = "linear"

    def __init__(
        self,
        manager,
        total_time: torch.Tensor,
        ori_groups: torch.Tensor,
        dt: float,
        blend_end_phi: float,
        device: str | torch.device,
    ) -> None:
        if blend_end_phi <= 0.0:
            raise ValueError(
                f"linear-interpolation transition needs blend_end_phi > 0 "
                f"(got {blend_end_phi}); use the 'raw' strategy to disable blending."
            )
        self.manager = manager
        self._total_time = total_time
        self._ori_groups = ori_groups
        self.dt = float(dt)
        self.blend_end_phi = float(blend_end_phi)
        self.device = torch.device(device)
        self._active: torch.Tensor | None = None
        self._old_stride: torch.Tensor | None = None
        self._old_phase: torch.Tensor | None = None
        self._phi: torch.Tensor | None = None

    def reset(self, n: int) -> None:
        dev = self.device
        self._active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._old_stride = torch.zeros(n, dtype=torch.long, device=dev)
        self._old_phase = torch.zeros(n, device=dev)
        self._phi = torch.zeros(n, device=dev)

    def arm(self, env_ids: torch.Tensor, old_stride: torch.Tensor, old_phase: torch.Tensor) -> None:
        self._active[env_ids] = True
        self._old_stride[env_ids] = old_stride
        self._old_phase[env_ids] = old_phase
        self._phi[env_ids] = 0.0

    def apply(self, ypos: torch.Tensor, yvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._active is None or not bool(self._active.any()):
            return ypos, yvel
        ids = self._active.nonzero(as_tuple=False).flatten()
        old_stride = self._old_stride[ids]
        # Old stride continued on its own clock, re-anchored to the same landing.
        self.manager.set_trajectory_indices(old_stride)
        oy, ody = self.manager.get_output(self._old_phase[ids] % 1.0)
        alpha = (self._phi[ids] / self.blend_end_phi).clamp(0.0, 1.0)
        ypos[ids] = blend_outputs(oy, ypos[ids], alpha, self._ori_groups)
        yvel[ids] = blend_outputs(ody, yvel[ids], alpha, None)
        # Advance the old clock + the blend accumulator; end the finished fades.
        dphi = self.dt / self._total_time[old_stride]
        self._old_phase[ids] = self._old_phase[ids] + dphi
        self._phi[ids] = self._phi[ids] + dphi
        self._active[ids] = self._phi[ids] < self.blend_end_phi
        return ypos, yvel


def make_transition_strategy(
    spec: str | TransitionStrategy,
    *,
    manager,
    total_time: torch.Tensor,
    ori_groups: torch.Tensor,
    dt: float,
    blend_end_phi: float,
    device: str | torch.device,
) -> TransitionStrategy:
    """Resolve ``spec`` to a :class:`TransitionStrategy` instance.

    ``"raw"`` / ``"linear"`` build the shipped strategies; a
    :class:`TransitionStrategy` instance is returned unchanged -- the extension
    hook: construct a custom technique (e.g. motion matching) with whatever it
    needs and pass it straight through.

    Args:
        spec: ``"raw"``, ``"linear"``, or a ready :class:`TransitionStrategy`.
        manager: The manager (for the ``linear`` strategy's old-stride eval).
        total_time: ``[T]`` per-trajectory stride duration.
        ori_groups: ``[F, 4]`` orientation column groups to SLERP.
        dt: Control-step dt (s).
        blend_end_phi: Phi window for the ``linear`` cross-fade.
        device: Torch device.

    Returns:
        A concrete :class:`TransitionStrategy`.
    """
    if isinstance(spec, TransitionStrategy):
        return spec
    if spec == "raw":
        return RawStitchTransition()
    if spec == "linear":
        return LinearInterpolationTransition(
            manager, total_time, ori_groups, dt, blend_end_phi, device
        )
    raise ValueError(
        f"unknown transition strategy '{spec}'; expected 'raw', 'linear', or a "
        f"TransitionStrategy instance."
    )
