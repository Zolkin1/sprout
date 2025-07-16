from isaaclab_tasks.manager_based.locomotion.velocity.mdp import UniformVelocityCommand
import isaaclab.utils.math as math_utils
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .gaitPeriodCfg import CustomUniformVelocityCommandCfg

class VelCmd(UniformVelocityCommand):
    def __init__(self, cfg: "CustomUniformVelocityCommandCfg", env):
        super().__init__(cfg, env)



    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command :
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()

            #select the envs that just enter the swing phase
            ref_term = self._env.command_manager.get_term(self.cfg.ref_cmd_name)

            if isinstance(ref_term.phase_var, float):
                swing_phase = ref_term.phase_var
                swing_phase = swing_phase < self.cfg.phase_threshold
              
                if swing_phase:

                    # compute angular velocity
                    heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])

                    self.vel_command_b[env_ids, 2] = torch.clip(
                        self.cfg.heading_control_stiffness * heading_error,
                        min=self.cfg.ranges.ang_vel_z[0],
                        max=self.cfg.ranges.ang_vel_z[1],
                    )

            else:
                swing_phase = ref_term.phase_var[env_ids]
                swing_phase = swing_phase < self.cfg.phase_threshold
                env_ids = env_ids[swing_phase]

                # compute angular velocity
                heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])

                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )
     
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0