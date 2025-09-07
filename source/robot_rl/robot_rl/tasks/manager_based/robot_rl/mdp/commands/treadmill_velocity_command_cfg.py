from isaaclab.utils import configclass

from .treadmill_velocity_command import TreadmillVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

@configclass
class TreadmillVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TreadmillVelocityCommand

    y_pos_kp: float = 0.0

    y_pos_kd: float = 0.0

    rel_y_envs: float = 0.0
