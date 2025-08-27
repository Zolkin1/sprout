import torch
import math

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.gait_library_traj import (
    GaitLibraryEndEffectorConfig
)

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.ee_traj import (bezier_deg, get_euler_from_quat)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.clf import CLF
import numpy as np
from isaaclab.managers import CommandTerm




class GaitLibraryHZDCommandTerm(CommandTerm):
    """HZD command term that uses a gait library with velocity-based selection."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.env = env
        self.robot = env.scene[cfg.asset_name]
        self.debug_vis = cfg.debug_vis

        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        self.hip_yaw_idx, _ = self.robot.find_joints(".*_hip_yaw_.*")
        self.metrics = {}

        self.mass = sum(self.robot.data.default_mass.T)[0]

        self.v = torch.zeros((self.num_envs), device=self.device)
        self.stance_idx = None

        self.y_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.y_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, cfg.num_outputs), device=self.device)
        self.yaw_output_idx = []



        self.stance_foot_vel = None
        self.stance_foot_ang_vel = None
        self.stance_foot_ori = None
        self.stance_foot_pos = None
        self.current_domain = ""
        self.use_standing = cfg.use_standing

        # Initialize gait library based on trajectory type
        if hasattr(cfg, 'gait_library_path'):
            config_name = getattr(cfg, 'config_name', 'single_support')
            
            self.gait_config = GaitLibraryEndEffectorConfig(
                cfg.gait_library_path,
                cfg.gait_velocity_ranges,
                config_name,
                cfg.use_standing,
                )

        else:
            raise ValueError("Gait library configuration missing: gait_library_path required")
        


        ##
        # Indexes of the virtual constraint for modification (yaw, euler rate)
        ##
        self.waist_joint_idx, _ = self.robot.find_joints(".*waist_yaw.*")
        self.joint_idx_list = self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]].get_joint_idx_list(self)
        self.foot_yaw_output_idx = 11   # TODO: Add the stance foot yaw here
        self.foot_y_output_idx = 7      # Lateral motion    # TODO: Add the stance foot y here
        self.ori_idx_list = [[3, 4, 5], [9, 10, 11]]
        self.yaw_output_idx = [5, 11]

        if cfg.use_standing:
            self.gait_config.standing_config.reorder_and_remap(cfg, self.device)

            
            right_des_pos = bezier_deg(
                    0, torch.zeros((1,), device=self.device), self.gait_config.T, self.gait_config.standing_config.right_coeffs,
                    torch.tensor(self.gait_config.bez_deg, device=self.device)
                )
            left_des_pos = bezier_deg(
                    0, torch.zeros((1,), device=self.device), self.gait_config.T, self.gait_config.standing_config.left_coeffs,
                    torch.tensor(self.gait_config.bez_deg, device=self.device)
                )
            self.gait_config.right_standing_pos = right_des_pos
            self.gait_config.left_standing_pos = left_des_pos
            self.standing_threshold = 0.03

        
        # Reorder and remap coefficients
        self.gait_config.reorder_and_remap(cfg, self.device)
        self.gait_cycle_prop = torch.zeros((self.env.num_envs,), device=self.device)
        self.initiate_clf()


    @property
    def command(self):
        return self.y_out


    def initiate_clf(self):
        # import pdb; pdb.set_trace()
        # num_domain = len(self.gait_config.T)
        self.clf = CLF(
            self.cfg.num_outputs, self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            Q_weights=np.array(self.cfg.Q_weights),
            R_weights=np.array(self.cfg.R_weights),
            device=self.device
        )

    def _get_gait_period(self) -> float:
        """Get the swing period from the gait configuration."""
       
        first_config = self.gait_config._get_first_gait()
        return sum(first_config.T.values())

    def generate_reference_trajectory(self):
        """Generate reference trajectory using gait library."""
        ref_pos, ref_vel = self.gait_config.get_ref_traj(self)
        self.y_out = ref_pos
        self.dy_out = ref_vel

    def get_actual_state(self):
        """Get actual state for gait library trajectories."""
        # Get stance foot pose data
        self.get_stance_foot_pose()
        
        # Get actual trajectory from gait library
        act_pos, act_vel = self.gait_config.get_actual_traj(self)
        self.y_act = act_pos
        self.dy_act = act_vel

    def _resample_command(self, env_ids):
        self._update_command()
        return

    def _update_metrics(self):
        """Update metrics specific to gait library tracking."""
        # Call parent method for base metrics
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot
        
        # Update metrics based on trajectory type
        if hasattr(self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]], 'axis_names'):
            # End-effector trajectories
            for axis_info in self.gait_config._gait_cache[list(self.gait_config._gait_cache.keys())[0]].axis_names:
                error_key = axis_info['name']
                index = axis_info['index']
                self.metrics[error_key] = torch.abs(
                    self.y_out[:, index] - 
                    self.y_act[:, index]
                )
        else:
            # Joint trajectories
            for i, joint_name in enumerate(self.robot.joint_names):
                error_key = f"error_{joint_name}"
                self.metrics[error_key] = torch.abs(self.y_out[:, i] - self.y_act[:, i])

    def get_stance_foot_pose(self):
        """Get stance foot pose data similar to JointTrajectoryConfig.get_stance_foot_pose."""

        # Only update the stance foot while in a phase with the foot on the ground
        stance_foot_idx = self.feet_bodies_idx[0] if self.stance_idx == 0 else self.feet_bodies_idx[1]
        self.stance_foot_pos = self.robot.data.body_pos_w[:, stance_foot_idx, :]
        stance_foot_quat = self.robot.data.body_quat_w[:, stance_foot_idx, :]
        self.stance_foot_ori = get_euler_from_quat(stance_foot_quat)

        self.stance_foot_vel = self.robot.data.body_lin_vel_w[:, stance_foot_idx, :]
        self.stance_foot_ang_vel = self.robot.data.body_ang_vel_w[:, stance_foot_idx, :]



    def update_stance_swing_idx(self):
        """Update stance and swing indices based on phase."""
        Tgait = self._get_gait_period()

        gait_cycle_prop = (self.env.sim.current_time % (2 * Tgait)) / (2 * Tgait)
        phi_c = torch.tensor(math.sin(2 * torch.pi * gait_cycle_prop) / math.sqrt(math.sin(2 * torch.pi * gait_cycle_prop)**2 + Tgait), device=self.env.device)

        new_stance_idx = int(0.5 - 0.5 * torch.sign(phi_c))
        self.swing_idx = 1 - new_stance_idx

        ##
        # Check which domain we are in
        ##
        domain_start_time = 0
        time_into_leg = self.env.sim.current_time % Tgait

        first_gait = self.gait_config._get_first_gait()
        for domain_name in first_gait.domain_seq:
            if time_into_leg < domain_start_time + first_gait.T[domain_name]:
                self.current_domain = domain_name
                break
            else:
                domain_start_time += first_gait.T[domain_name]

        # Compute how far into the domain we are on a 0-1 scale
        self.phase_var = (time_into_leg - domain_start_time)/first_gait.T[self.current_domain]

        if self.current_domain == "":
            raise ValueError("Could not determine the current domain!")

        ##
        # Check if the stance idx changed, only check when we are not in the flight phase
        ##
        if self.stance_idx is None or new_stance_idx != self.stance_idx:
            if self.stance_idx is None:
                self.stance_idx = new_stance_idx

            # Update stance foot pose based on trajectory type
            if hasattr(self, 'ee_tracker'):
                # End-effector trajectories
                stance_foot_idx = self.feet_bodies_idx[0] if new_stance_idx == 0 else self.feet_bodies_idx[1]
                stance_foot_pos = self.robot.data.body_pos_w[:, stance_foot_idx, :]
                stance_foot_quat = self.robot.data.body_quat_w[:, stance_foot_idx, :]
                stance_foot_ori = get_euler_from_quat(stance_foot_quat)
                
                self.stance_foot_pos_0 = stance_foot_pos
                self.stance_foot_ori_quat_0 = stance_foot_quat
                self.stance_foot_ori_0 = stance_foot_ori
            else:
                # Joint trajectories
                foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
                foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
                self.stance_foot_pos_0 = foot_pos_w[:, new_stance_idx, :]
                self.stance_foot_ori_quat_0 = foot_ori_w[:, new_stance_idx, :]
                self.stance_foot_ori_0 = get_euler_from_quat(foot_ori_w[:, new_stance_idx, :])
       
        just_reset = (self.env.episode_length_buf == 0)
        # Handle episode reset: force re-alignment of stance foot
        if just_reset.any():
           # End-effector mode: assumes single-env, so just do once
           stance_foot_idx = self.feet_bodies_idx[0] if new_stance_idx == 0 else self.feet_bodies_idx[1]
           stance_foot_pos = self.robot.data.body_pos_w[:, stance_foot_idx, :]
           stance_foot_quat = self.robot.data.body_quat_w[:, stance_foot_idx, :]
           stance_foot_ori = get_euler_from_quat(stance_foot_quat)

           self.stance_foot_pos_0[just_reset] = stance_foot_pos[just_reset]
           self.stance_foot_ori_quat_0[just_reset] = stance_foot_quat[just_reset]
           self.stance_foot_ori_0[just_reset] = stance_foot_ori[just_reset]
          

        self.stance_idx = new_stance_idx

        self.gait_cycle_prop = torch.full((self.num_envs,), gait_cycle_prop, device=self.device)  # Used in observation phase variables

        if gait_cycle_prop < 0.5:
            self.cur_swing_time = gait_cycle_prop*2.0 # Used in swing yaw modification
        else:
            self.cur_swing_time = (gait_cycle_prop - 0.5) * 2.0

    def _update_command(self):
        """Update the command by generating reference and computing CLF."""
        self.update_stance_swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, self.yaw_output_idx)
        self.vdot = vdot
        self.v = vcur



