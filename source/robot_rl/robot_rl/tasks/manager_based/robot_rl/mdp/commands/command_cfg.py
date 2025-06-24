from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .command_terms import GaitPeriodCommand


@configclass
class GaitPeriodCfg(CommandTermCfg):
    """Configure a gait period command."""

    class_type: type = GaitPeriodCommand

    period_range: tuple[float, float] = MISSING
