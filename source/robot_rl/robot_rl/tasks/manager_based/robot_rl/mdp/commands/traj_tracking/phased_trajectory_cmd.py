# Copyright 2026 Zachary Olkin. All rights reserved.

"""Single-skill trajectory command that exposes a phasing variable.

Single-skill trajectory commands feed the phasing variable ``phi in [0, 1]``
into the observation (via ``ref_sin_phase`` / ``ref_cos_phase``) and drive the
trajectory purely from the episode clock.  They should extend this class;
multi-skill commands that manage phase on the manager should extend
:class:`BaseTrajectoryCommand` directly.

This class OWNS phase advance for the single-skill path.  The live
:class:`BaseTrajectoryCommand._update_command` reads ``self.manager.phase`` each
step, so :meth:`_pre_update_phase` (called at the top of ``_update_command``)
computes the per-env phase from the episode clock and writes it onto the
manager.  It also implements the *fixed-phase static hold* used by the
bend-cycle env: with ``cfg.hold_phi_values`` set, a per-episode subset of
environments (chosen at reset, see :func:`mdp.events.resets.reset_bend_cycle`)
is pinned to a fixed phase and has its reference velocity zeroed, so the policy
learns to hold that pose statically.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase


class PhasedTrajectoryCommand(BaseTrajectoryCommand):
    """Base class for single-skill trajectory commands with a phasing variable.

    Adds:

    - ``self.phasing_var`` state (the phase in ``[0, 1]``) exposed as an
      observation input and used for phase-binned CLF logging.
    - Episode-clock phase advance in :meth:`_pre_update_phase`.
    - Fixed-phase static hold controlled by ``cfg.hold_phi_values`` /
      ``cfg.percent_hold_phi`` and the per-env ``held_mask`` /
      ``hold_phi_value`` state written by the reset event.
    """

    @abstractmethod
    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create the concrete trajectory manager."""

    @abstractmethod
    def _verify_contact_frames(self) -> None:
        """Verify trajectory contact frames match the configured list."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # Phasing variable in [0, 1]; the sin/cos phase observations read this.
        self.phasing_var = torch.zeros(self.num_envs, device=self.device)

        # Fixed-phase static-hold state (populated at reset by
        # ``reset_bend_cycle`` when ``cfg.hold_phi_values`` is set):
        #   ``held_mask[e]``      -> this env holds a fixed phase this episode.
        #   ``hold_phi_value[e]`` -> the phase it is pinned to.
        self.held_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_phi_value = torch.zeros(self.num_envs, device=self.device)

    def _pre_update_phase(self) -> None:
        """Advance the phasing variable from the episode clock, then apply holds.

        Called at the top of ``BaseTrajectoryCommand._update_command`` (which
        then reads ``self.manager.phase``).  Phase is recomputed from absolute
        episode time each step (``get_phasing_var`` wraps for periodic
        trajectories), so there is no incremental drift.  Held envs are pinned
        to their fixed phase.
        """
        t = self.env.episode_length_buf * self.env.step_dt
        phase = self.manager.get_phasing_var(t)

        if self.cfg.hold_phi_values is not None:
            phase = torch.where(self.held_mask, self.hold_phi_value, phase)

        self.manager.phase = phase
        self.phasing_var = phase
        self.manager.invalidate_cache()

    def get_phasing_var(self) -> torch.Tensor:
        """Return the current phasing variable of shape ``[N]``."""
        return self.phasing_var

    def get_desired_outputs(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Populate desired outputs, zeroing the reference velocity for held envs.

        Held (static-hold) envs must track a *stationary* pose, so their
        reference velocity is zeroed regardless of the clip's velocity at that
        phase.

        Args:
            phase: phase in [0, 1] tensor of shape ``[N]``.
            env_ids: Optional environment indices.
        """
        super().get_desired_outputs(phase, env_ids)

        if self.cfg.hold_phi_values is None:
            return
        if env_ids is None:
            self.dy_des[self.held_mask] = 0.0
        else:
            held_sub = self.held_mask[env_ids]
            self.dy_des[env_ids[held_sub]] = 0.0

    def reset_hold_state(
        self, env_ids: torch.Tensor, held: torch.Tensor, hold_phi: torch.Tensor
    ) -> None:
        """Set which of ``env_ids`` hold a fixed phase this episode (reset hook).

        Args:
            env_ids: Environment indices being reset.
            held: Bool tensor ``[len(env_ids)]`` — whether each env holds.
            hold_phi: Float tensor ``[len(env_ids)]`` — the fixed phase per env
                (ignored where ``held`` is False).
        """
        self.held_mask[env_ids] = held
        self.hold_phi_value[env_ids] = hold_phi
