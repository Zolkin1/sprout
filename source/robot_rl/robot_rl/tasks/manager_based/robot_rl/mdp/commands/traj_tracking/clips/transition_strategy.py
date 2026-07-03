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
* :class:`QuadraticInterpolationTransition` -- the same cross-fade with a
  quadratic ease-IN (``alpha = frac**2``), holding the old (leaving) stride
  longer before ramping to the new one.

The cross-fade weight is factored into an overridable
:meth:`LinearInterpolationTransition._ease` hook, so a new easing is a two-line
subclass.  A future technique (e.g. motion matching) only needs to subclass
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
    def arm(
        self,
        env_ids: torch.Tensor,
        old_stride: torch.Tensor,
        old_phase: torch.Tensor,
        pos_shift: torch.Tensor | None = None,
        seg_len: torch.Tensor | float = 1.0,
    ) -> None:
        """Begin a transition for ``env_ids`` leaving ``old_stride`` at ``old_phase``.

        Args:
            env_ids: Clips that just crossed a seam.
            old_stride: Global index of the stride each clip is leaving.
            old_phase: Phase to continue the old stride from (the overshoot past
                1.0 for a cycle-wrap seam, so it flows into its next cycle; the
                un-wrapped mid-cycle phase for a mid-cycle seam, so it flows into
                its own second half).
            pos_shift: Optional ``[len(env_ids), 3]`` constant translation added to
                the old stride's position channels each step, re-aligning its
                stance foot to the new landing at a mid-cycle seam (where the new
                anchor is solved on a different foot).  ``None`` (cycle-wrap seams)
                means no shift -- both strides already share the anchor frame.
            seg_len: Full-period phase span of the STEP (half-cycle) being entered
                (scalar or ``[len(env_ids)]``), so a cross-fade's ``blend_end_phi``
                is a fraction of a STEP.  Defaults to 1.0 (the whole period).
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

    def debug_blend_state(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Per-clip ``(active[n] bool, alpha[n] float)`` applied at the LAST
        :meth:`apply` (alpha 0 where not mid-blend), or ``(None, None)`` for a
        strategy with no blend (e.g. ``raw``).  Used only by the debug logging."""
        return None, None

    # Set by the generator when ``debug_interp`` is on; base default off.
    _log_nominals: bool = False

    def debug_blend_outputs(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """PRE-blend ``(old_out[n,P], new_out[n,P])`` from the LAST :meth:`apply`
        (NaN where not mid-blend) when nominal logging is on; else ``(None, None)``."""
        return None, None


class RawStitchTransition(TransitionStrategy):
    """Hard cut: the new stride replaces the old at the seam, no blending."""

    name = "raw"

    @property
    def arms(self) -> bool:
        return False

    def reset(self, n: int) -> None:
        pass

    def arm(
        self,
        env_ids: torch.Tensor,
        old_stride: torch.Tensor,
        old_phase: torch.Tensor,
        pos_shift: torch.Tensor | None = None,
        seg_len: torch.Tensor | float = 1.0,
    ) -> None:
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
        blend_end_phi: Fraction of a STEP (the half-cycle segment being entered)
            over which alpha ramps 0 -> 1; in ``(0, 1]``.  The per-clip step
            length is supplied at :meth:`arm` via ``seg_len`` (phase span of the
            segment, e.g. ~0.5 of a full period for a half-cycle; defaults to 1.0
            = the whole period when not given), so the window is
            ``blend_end_phi * seg_len`` of full-period phase.
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
        pos_groups: torch.Tensor | None = None,
    ) -> None:
        if not 0.0 < blend_end_phi <= 1.0:
            raise ValueError(
                f"blend_end_phi must be in (0, 1] -- it is the fraction of a STEP "
                f"(half-cycle) over which the cross-fade completes (got "
                f"{blend_end_phi}); use the 'raw' strategy to disable blending."
            )
        self.manager = manager
        self._total_time = total_time
        self._ori_groups = ori_groups
        self.dt = float(dt)
        self.blend_end_phi = float(blend_end_phi)
        self.device = torch.device(device)
        # Per-frame (pos_x, pos_y, pos_z) column groups for the mid-cycle position
        # re-alignment shift; empty [0, 3] means "no shift channels" (raw-only).
        self._pos_groups = (
            pos_groups if pos_groups is not None
            else torch.zeros((0, 3), dtype=torch.long, device=self.device)
        )
        self._active: torch.Tensor | None = None
        self._old_stride: torch.Tensor | None = None
        self._old_phase: torch.Tensor | None = None
        self._phi: torch.Tensor | None = None
        self._pos_shift: torch.Tensor | None = None
        self._seg_len: torch.Tensor | None = None
        # Debug snapshot of the LAST apply(): which clips were mid-blend and the
        # blend weight applied (0 elsewhere).  Read via :meth:`debug_blend_state`.
        self._last_active: torch.Tensor | None = None
        self._last_alpha: torch.Tensor | None = None
        # When ``_log_nominals`` (set by the generator's ``debug_interp``), also
        # snapshot the two PRE-blend outputs -- the old-stride continuation and
        # the pure new segment -- so the plot can show the blended value against
        # its nominals.  ``[n, P]`` with NaN where a clip is not mid-blend.
        self._log_nominals: bool = False
        self._last_old_out: torch.Tensor | None = None
        self._last_new_out: torch.Tensor | None = None

    def reset(self, n: int) -> None:
        dev = self.device
        self._active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._old_stride = torch.zeros(n, dtype=torch.long, device=dev)
        self._old_phase = torch.zeros(n, device=dev)
        self._phi = torch.zeros(n, device=dev)
        self._pos_shift = torch.zeros(n, 3, device=dev)
        self._seg_len = torch.ones(n, device=dev)   # full-period default
        self._last_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._last_alpha = torch.zeros(n, device=dev)

    def arm(
        self,
        env_ids: torch.Tensor,
        old_stride: torch.Tensor,
        old_phase: torch.Tensor,
        pos_shift: torch.Tensor | None = None,
        seg_len: torch.Tensor | float = 1.0,
    ) -> None:
        self._active[env_ids] = True
        self._old_stride[env_ids] = old_stride
        self._old_phase[env_ids] = old_phase
        self._phi[env_ids] = 0.0
        self._pos_shift[env_ids] = (
            0.0 if pos_shift is None else pos_shift.to(self._pos_shift.dtype)
        )
        # Per-clip STEP length (full-period phase span of the segment being
        # entered), so ``blend_end_phi`` is a fraction of a STEP, not the period.
        self._seg_len[env_ids] = seg_len

    def _ease(self, frac: torch.Tensor) -> torch.Tensor:
        """Map the normalized window fraction to a blend weight (linear: identity).

        ``frac`` is ``phi_elapsed / blend_end_phi`` clamped to ``[0, 1]``.
        Subclasses override this to shape the cross-fade (e.g. a quadratic
        ease-in that holds the old stride longer).
        """
        return frac

    def apply(self, ypos: torch.Tensor, yvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Reset the per-step debug snapshot (0 = not mid-blend this step).
        self._last_active.zero_()
        self._last_alpha.zero_()
        if self._log_nominals:
            self._last_old_out = torch.full_like(ypos, float("nan"))
            self._last_new_out = torch.full_like(ypos, float("nan"))
        if self._active is None or not bool(self._active.any()):
            return ypos, yvel
        ids = self._active.nonzero(as_tuple=False).flatten()
        old_stride = self._old_stride[ids]
        # Old stride continued on its own clock, re-anchored to the same landing.
        self.manager.set_trajectory_indices(old_stride)
        oy, ody = self.manager.get_output(self._old_phase[ids] % 1.0)
        # Mid-cycle seams re-anchor on a DIFFERENT foot than the old stride's
        # native frame, so translate the old continuation's positions by the
        # per-clip shift to align its stance foot with the new landing before
        # blending (cycle-wrap seams have zero shift -> no-op).
        if self._pos_groups.numel():
            shift = self._pos_shift[ids].unsqueeze(1)          # [k, 1, 3]
            oy[:, self._pos_groups] = oy[:, self._pos_groups] + shift
        # Window in full-period phase = blend_end_phi (fraction of a STEP) * the
        # step's phase span, so the fade completes over ``blend_end_phi`` of the
        # half-cycle segment rather than the whole two-stride period.
        window = self.blend_end_phi * self._seg_len[ids]
        frac = (self._phi[ids] / window).clamp(0.0, 1.0)
        alpha = self._ease(frac)
        # Record the debug snapshot for this step (active clips + applied weight).
        self._last_active[ids] = True
        self._last_alpha[ids] = alpha
        if self._log_nominals:
            # PRE-blend nominals: the old-stride continuation (oy, already
            # pos-shifted) and the pure new segment (ypos before it is overwritten).
            self._last_old_out[ids] = oy
            self._last_new_out[ids] = ypos[ids].clone()
        ypos[ids] = blend_outputs(oy, ypos[ids], alpha, self._ori_groups)
        yvel[ids] = blend_outputs(ody, yvel[ids], alpha, None)
        # Advance the old clock + the blend accumulator; end the finished fades.
        dphi = self.dt / self._total_time[old_stride]
        self._old_phase[ids] = self._old_phase[ids] + dphi
        self._phi[ids] = self._phi[ids] + dphi
        self._active[ids] = self._phi[ids] < window
        return ypos, yvel

    def debug_blend_state(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._last_active, self._last_alpha

    def debug_blend_outputs(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._last_old_out, self._last_new_out


class QuadraticInterpolationTransition(LinearInterpolationTransition):
    """Quadratic ease-IN cross-fade: hold the OLD stride longer, then ramp.

    Identical machinery to :class:`LinearInterpolationTransition`, but the blend
    weight is ``alpha = frac**2`` (``frac`` = the normalized window fraction), so
    the reference stays closer to the old (leaving) stride early in the window --
    at the midpoint it is only 25% toward the new stride vs 50% for linear -- and
    catches up near the end.  Use when holding the leaving trajectory a little
    longer reads better than an even linear cross-fade.  Same ``blend_end_phi``
    window and the same guard (< the smallest half-cycle) apply.
    """

    name = "quadratic"

    def _ease(self, frac: torch.Tensor) -> torch.Tensor:
        return frac * frac


def make_transition_strategy(
    spec: str | TransitionStrategy,
    *,
    manager,
    total_time: torch.Tensor,
    ori_groups: torch.Tensor,
    dt: float,
    blend_end_phi: float,
    device: str | torch.device,
    pos_groups: torch.Tensor | None = None,
) -> TransitionStrategy:
    """Resolve ``spec`` to a :class:`TransitionStrategy` instance.

    ``"raw"`` / ``"linear"`` / ``"quadratic"`` build the shipped strategies; a
    :class:`TransitionStrategy` instance is returned unchanged -- the extension
    hook: construct a custom technique (e.g. motion matching) with whatever it
    needs and pass it straight through.

    Args:
        spec: ``"raw"``, ``"linear"``, ``"quadratic"``, or a ready
            :class:`TransitionStrategy`.
        manager: The manager (for the cross-fade strategies' old-stride eval).
        total_time: ``[T]`` per-trajectory stride duration.
        ori_groups: ``[F, 4]`` orientation column groups to SLERP.
        dt: Control-step dt (s).
        blend_end_phi: Phi window for the cross-fade.
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
            manager, total_time, ori_groups, dt, blend_end_phi, device, pos_groups=pos_groups,
        )
    if spec == "quadratic":
        return QuadraticInterpolationTransition(
            manager, total_time, ori_groups, dt, blend_end_phi, device, pos_groups=pos_groups,
        )
    raise ValueError(
        f"unknown transition strategy '{spec}'; expected 'raw', 'linear', "
        f"'quadratic', or a TransitionStrategy instance."
    )
