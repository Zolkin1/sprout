from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass

from .gaitPeriod import GaitPeriodCommand
from .vel_cmd import VelCmd

@configclass
class GaitPeriodCfg(CommandTermCfg):
    """Configure a gait period command."""

    class_type: type = GaitPeriodCommand

    period_range: tuple[float, float] = MISSING



@configclass
class CustomUniformVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = VelCmd
    