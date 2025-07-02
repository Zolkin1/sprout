import torch
import math
import numpy as np

from isaaclab.managers import CommandTerm



from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.clf import CLF


from typing import TYPE_CHECKING
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import JointTrajectoryConfig, get_euler_from_quat




if TYPE_CHECKING:
    from ..cmd_cfg import HZDCommandCfg


class HZDCommandTerm(CommandTerm):
    def __init__(self, cfg: "HZDCommandCfg", env):
        super().__init__(cfg, env)
       
        self.env = env
        self.robot = env.scene[cfg.asset_name]

        self.debug_vis = cfg.debug_vis


        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        self.hip_yaw_idx,_ = self.robot.find_joints(".*_hip_yaw_.*")
        self.metrics = {}
     
        # load joint trajectory config from YAML

        self.ref_config = JointTrajectoryConfig()

        
        self.ref_config.reorder_and_remap_jt(cfg,self.robot,self.device)

    
        self.mass = sum(self.robot.data.default_mass.T)[0]
        
        self.clf = CLF(
           cfg.num_outputs, self.env.cfg.sim.dt,
           batch_size=self.num_envs,
            Q_weights=np.array(cfg.Q_weights),
            R_weights=np.array(cfg.R_weights),
            device=self.device
        )
        
        self.v = torch.zeros((self.num_envs), device=self.device)
        self.stance_idx = None

        self.y_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.y_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)


    @property
    def command(self):
        return self.y_out
    

    def _resample_command(self, env_ids):
        self._update_command()
        # Do nothing here
        # device = self.env.command_manager.get_command("base_velocity").device
        
        return
    
    def _update_metrics(self):
        # Update metrics using actual joint names from the YAML file
        for i, joint_name in enumerate(self.robot.joint_names):
            error_key = f"error_{joint_name}"
            self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot



    def update_Stance_Swing_idx(self):
        Tswing = self.ref_config.T
        tp = (self.env.sim.current_time % (2 * Tswing)) / (2 * Tswing)  
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.ref_config.T), device=self.env.device)

        new_stance_idx = int(0.5 - 0.5 * torch.sign(phi_c))
        self.swing_idx = 1 - new_stance_idx
        
        if self.stance_idx is None or new_stance_idx != self.stance_idx:
            if self.stance_idx is None:
                self.stance_idx = new_stance_idx

            # update stance foot pos, ori
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            self.stance_foot_pos_0 = foot_pos_w[:, new_stance_idx, :]
            self.stance_foot_ori_quat_0 = foot_ori_w[:, new_stance_idx, :]
            self.stance_foot_ori_0 = get_euler_from_quat(foot_ori_w[:, new_stance_idx, :])
       
        self.stance_idx = new_stance_idx

        if tp < 0.5:
            self.phase_var = 2 * tp
        else:
            self.phase_var = 2 * tp - 1
        self.cur_swing_time = self.phase_var * Tswing
        

    def generate_reference_trajectory(self):
        ref_pos, ref_vel = self.ref_config.get_ref_traj(self)

        self.y_out = ref_pos
        self.dy_out = ref_vel


    def get_actual_state(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        self.ref_config.get_stance_foot_pose(self)
        jt_pos, jt_vel = self.ref_config.get_actul_traj(self)

        self.y_act = jt_pos
        self.dy_act = jt_vel
        

    def _update_command(self):
        
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, [])
        self.vdot = vdot
        self.v = vcur
       
          
