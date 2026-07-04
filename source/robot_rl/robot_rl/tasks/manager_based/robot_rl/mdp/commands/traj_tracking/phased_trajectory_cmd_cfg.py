# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for :class:`PhasedTrajectoryCommand`.

TODO: Delete with :mod:`phased_trajectory_cmd`. Replaced by
:class:`BatchedMultiSkillCommandCfg` once g1 single-skill configs are
migrated.
"""

from isaaclab.utils import configclass

from .base_trajectory_cmd_cfg import BaseTrajectoryCommandCfg


@configclass
class PhasedTrajectoryCommandCfg(BaseTrajectoryCommandCfg):
    """Configuration shared by phased (single-skill) trajectory commands.

    Adds the hold-phi fields on top of
    :class:`BaseTrajectoryCommandCfg`.  ``percent_hold_phi`` controls the
    per-episode random hold of the trajectory at t=0.
    ``hold_phi_threshold`` is the commanded base-velocity magnitude below
    which the phase boundary-hold engages.  ``phasing_boundaries`` sets
    which boundary crossing (1st, 2nd, ...) locks the phase.

    ``bent_dwell_s`` / ``stand_dwell_s`` enable the *dwell cycle* used by the
    bend-cycle env: when set to a ``(min, max)`` range (seconds), the phase pauses
    at the bent pose (``bent_hold_phi``) / at standing (phase 0) for a
    per-episode-random dwell drawn from that range each cycle, so every env plays
    stand -> hold-stand -> bend -> hold-bent -> bend-up -> stand (the reference
    velocity is zeroed during each dwell). ``None`` on both disables the dwell
    (plain periodic cycle).

    ``bent_hold_phi`` is the phase held as "bent". ``None`` (default) auto-derives
    it from the clip's ``bend_down -> bend_up`` domain boundary (the deepest bend,
    e.g. ~0.474 for a 1.8s/2.0s split); set a float to override.
    """

    percent_hold_phi: float = -1
    hold_phi_threshold: float = -1
    phasing_boundaries: float = 1
    bent_dwell_s: tuple[float, float] | None = None
    stand_dwell_s: tuple[float, float] | None = None
    bent_hold_phi: float | None = None
