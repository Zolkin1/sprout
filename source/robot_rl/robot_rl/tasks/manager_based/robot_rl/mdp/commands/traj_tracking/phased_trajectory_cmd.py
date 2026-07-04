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
manager.  It also implements the *dwell cycle* used by the bend-cycle env: with
``cfg.stand_dwell_s`` / ``cfg.bent_dwell_s`` set, the phase pauses at standing
(phase 0) and at the bent pose (``self._bent_phi``) for per-episode-random
dwells each cycle, so every env plays stand -> hold-stand -> bend -> hold-bent
-> bend-up -> stand and trains both (fragile) holds.
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
    - An optional per-env dwell at standing (phase 0) and at the bent pose
      (``self._bent_phi``), enabled by ``cfg.stand_dwell_s`` / ``cfg.bent_dwell_s``:
      the cycle pauses there each period so the policy trains holding those poses,
      with the reference velocity zeroed during each dwell.
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

        # Per-env dwell durations (s), drawn each reset by ``reset_bend_cycle``.
        # The cycle pauses at standing (phase 0) for ``_stand_dwell_s`` and at the
        # bent pose (``_bent_phi``) for ``_bent_dwell_s`` before continuing.
        self._stand_dwell_s = torch.zeros(self.num_envs, device=self.device)
        self._bent_dwell_s = torch.zeros(self.num_envs, device=self.device)
        self._uses_dwell = (cfg.bent_dwell_s is not None) or (cfg.stand_dwell_s is not None)

        # Clip period (s): the divisor that maps episode time -> phase (matches
        # what ``get_phasing_var`` divides by), used to place the dwells.
        self._period = float(self.manager.get_total_time())

        # Bent-pose phase: explicit override, else the clip's bend_down->bend_up
        # domain boundary (the deepest bend, e.g. ~0.474), else 0.5.
        if cfg.bent_hold_phi is not None:
            self._bent_phi = float(cfg.bent_hold_phi)
        elif hasattr(self.manager, "get_domain_boundary_phases"):
            boundaries = self.manager.get_domain_boundary_phases()
            self._bent_phi = float(boundaries[0]) if boundaries else 0.5
        else:
            self._bent_phi = 0.5

    def _pre_update_phase(self) -> None:
        """Advance the phasing variable from the episode clock (with the dwells).

        Called at the top of ``BaseTrajectoryCommand._update_command`` (which then
        reads ``self.manager.phase``).  Without dwells the phase is recomputed from
        absolute episode time each step (``get_phasing_var`` wraps for periodic
        clips, so there is no drift).  With a dwell configured the clock pauses at
        standing and/or the bent pose each cycle (see :meth:`_dwell_cycle_phase`).
        """
        t = self.env.episode_length_buf * self.env.step_dt
        if self._uses_dwell:
            phase = self._dwell_cycle_phase(t)
        else:
            phase = self.manager.get_phasing_var(t)

        self.manager.phase = phase
        self.phasing_var = phase
        self.manager.invalidate_cache()

    def _dwell_cycle_phase(self, t: torch.Tensor) -> torch.Tensor:
        """Phase for the periodic cycle with per-env dwells at standing and bent.

        Each period is: hold standing (phase 0) for ``_stand_dwell_s``, sweep
        ``0 -> bent`` at the clip's natural rate, hold bent for ``_bent_dwell_s``,
        sweep ``bent -> 1.0``, then wrap.  Inserting the dwells into the clock keeps
        the phase (and thus the sin/cos observation) continuous across each pause.

        Args:
            t: Elapsed episode time per env, shape ``[N]``.

        Returns:
            Phasing variable in ``[0, 1]``, shape ``[N]``.
        """
        bent = self._bent_phi
        period = self._period
        t_reach = bent * period                       # time to sweep 0 -> bent
        d_s = self._stand_dwell_s                      # [N] standing dwell (s)
        d_b = self._bent_dwell_s                       # [N] bent dwell (s)
        tau = torch.remainder(t, period + d_s + d_b)   # time within the current period

        r_stand = tau < d_s
        r_bend = (tau >= d_s) & (tau < d_s + t_reach)
        r_bent = (tau >= d_s + t_reach) & (tau < d_s + t_reach + d_b)
        # else r_up: bending up, phase = (tau - d_s - d_b) / period
        return torch.where(
            r_stand,
            torch.zeros_like(tau),
            torch.where(
                r_bend,
                (tau - d_s) / period,
                torch.where(r_bent, torch.full_like(tau, bent), (tau - d_s - d_b) / period),
            ),
        )

    def get_phasing_var(self) -> torch.Tensor:
        """Return the current phasing variable of shape ``[N]``."""
        return self.phasing_var

    def get_desired_outputs(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Populate desired outputs, zeroing the reference velocity during a dwell.

        While the phase is paused at standing (phase 0) or the bent pose
        (``self._bent_phi``) the robot holds a *stationary* pose, so its reference
        velocity is zeroed (the clip's velocity there is otherwise nonzero).
        Outside the dwells (sweeping in / out) it tracks the real motion velocity.

        Args:
            phase: phase tensor (shape ``[N]`` when ``env_ids`` is None, else
                ``[len(env_ids)]``) -- the phase these outputs are evaluated at.
            env_ids: Optional environment indices.
        """
        super().get_desired_outputs(phase, env_ids)

        if not self._uses_dwell:
            return
        dwelling = (phase == self._bent_phi) | (phase == 0.0)
        if env_ids is None:
            self.dy_des[dwelling] = 0.0
        else:
            self.dy_des[env_ids[dwelling]] = 0.0

    def reset_dwells(self, env_ids: torch.Tensor, stand_s: torch.Tensor, bent_s: torch.Tensor) -> None:
        """Set the per-env standing / bent dwell durations (s) for the resetting envs.

        Args:
            env_ids: Environment indices being reset.
            stand_s: Standing-dwell duration per env, shape ``[len(env_ids)]``.
            bent_s: Bent-dwell duration per env, shape ``[len(env_ids)]``.
        """
        self._stand_dwell_s[env_ids] = stand_s
        self._bent_dwell_s[env_ids] = bent_s
