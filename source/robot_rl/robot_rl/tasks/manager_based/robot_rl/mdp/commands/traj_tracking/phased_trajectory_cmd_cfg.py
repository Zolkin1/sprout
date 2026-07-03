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

    ``hold_phi_values`` enables the *fixed-phase static hold* used by the
    bend-cycle env: when set to a list of phases (e.g. ``[0.0, 0.5, 1.0]``),
    each episode is, with probability ``percent_hold_phi``, pinned to a phase
    drawn uniformly from the list (the robot is reset into that pose and the
    reference velocity is zeroed so the CLF rewards holding it statically).
    ``None`` disables the static hold entirely.
    """

    percent_hold_phi: float = -1
    hold_phi_threshold: float = -1
    phasing_boundaries: float = 1
    hold_phi_values: list[float] | None = None
