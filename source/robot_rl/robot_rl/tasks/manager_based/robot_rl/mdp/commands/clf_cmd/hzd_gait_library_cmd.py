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

        self.current_domains = torch.zeros(self.env.num_envs, device=self.device)
        self.gait_indices = torch.zeros(self.num_envs,device=self.device)

        self.stance_foot_vel = None
        self.stance_foot_ang_vel = None
        self.stance_foot_ori = None
        self.stance_foot_pos = None
        self.use_standing = cfg.use_standing
        
        # Initialize gait library based on trajectory type
        if hasattr(cfg, 'gait_library_path'):
            config_name = getattr(cfg, 'config_name', 'single_support')
            
            self.gait_config = GaitLibraryEndEffectorConfig(
                cfg.gait_library_path,
                cfg.gait_velocity_ranges,
                self.device,
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

        # Reorder and remap coefficients
        self.gait_config.reorder_and_remap(cfg, self.device)
        self.gait_cycle_prop = torch.zeros((self.env.num_envs,), device=self.device)
        self.initiate_clf()

        if cfg.use_standing:
            self.gait_config.standing_config.reorder_and_remap(cfg, self.device)

            
            right_des_pos = bezier_deg(
                    0, torch.zeros((1,), device=self.device), self.gait_config.T, self.gait_config.standing_config.right_coeffs["double_support"],
                    torch.tensor(self.gait_config.bez_deg, device=self.device)
                )
            left_des_pos = bezier_deg(
                    0, torch.zeros((1,), device=self.device), self.gait_config.T, self.gait_config.standing_config.left_coeffs["double_support"],
                    torch.tensor(self.gait_config.bez_deg, device=self.device)
                )
            self.gait_config.right_standing_pos = right_des_pos
            self.gait_config.left_standing_pos = left_des_pos
            self.standing_threshold = 0.03


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
        return self.gait_config.T_gait

    def generate_reference_trajectory(self, cmd_vel, gait_indices):
        """Generate reference trajectory using gait library."""
        ref_pos, ref_vel = self.gait_config.get_ref_traj(self, cmd_vel, gait_indices)
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
                self.metrics[error_key] =  torch.abs(
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
        half_cycle_prop = (self.env.sim.current_time % Tgait) / Tgait

        new_stance_idx = int(gait_cycle_prop >= 0.5)
        self.swing_idx = 1 - new_stance_idx

        ##
        # Check if the stance idx changed, only check when we are not in the flight phase
        ##
        if self.stance_idx is None or new_stance_idx != self.stance_idx:
            if self.stance_idx is None:
                self.stance_idx = new_stance_idx

            # Update stance foot pose
            stance_foot_idx = self.feet_bodies_idx[0] if new_stance_idx == 0 else self.feet_bodies_idx[1]
            stance_foot_pos = self.robot.data.body_pos_w[:, stance_foot_idx, :]
            stance_foot_quat = self.robot.data.body_quat_w[:, stance_foot_idx, :]
            stance_foot_ori = get_euler_from_quat(stance_foot_quat)

            self.stance_foot_pos_0 = stance_foot_pos
            self.stance_foot_ori_quat_0 = stance_foot_quat
            self.stance_foot_ori_0 = stance_foot_ori
       
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

        self.cur_swing_time = self.env.sim.current_time % Tgait

    def _update_command(self):
        """Update the command by generating reference and computing CLF."""
        # Get the commanded velocity
        commanded_velocity = self.env.command_manager.get_command("base_velocity")  # (N,3)

        # Get the active gaits for each env
        gait_indices = self.gait_config.select_gaits_by_velocity(commanded_velocity[:, :2])
        self.gait_indices = gait_indices

        # Get the active domains and phasing vars for each env
        self.current_domains, self.phase_var, self.domain_durations = self.gait_config.determine_domains(gait_indices, self.env.sim.current_time)

        # print(f"current_domains: {self.current_domains}, phase_var: {self.phase_var}, time: {self.env.sim.current_time}, domain_durations: {self.domain_durations}")

        # Update the stance and swing legs
        self.update_stance_swing_idx()

        # Get the reference trajectory
        self.generate_reference_trajectory(commanded_velocity, gait_indices)

        # Get the state
        self.get_actual_state()

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_out, self.dy_act, self.dy_out, self.yaw_output_idx)
        self.vdot = vdot
        self.v = vcur

    def get_flight_envs(self):
        """Get a masking tensor with a 1 for each environment in the flight phase."""
        # Get flight phase domain index
        flight_domain_idx = self.gait_config.domain_name_to_idx["flight_phase"]

        # Create boolean mask where current_domains equals flight_phase index
        flight_mask = (self.current_domains == flight_domain_idx)

        return flight_mask.int()


    def get_not_flight_envs(self):
        # Get flight phase domain index
        flight_domain_idx = self.gait_config.domain_name_to_idx["flight_phase"]

        # Create boolean mask where current_domains equals flight_phase index
        flight_mask = (self.current_domains != flight_domain_idx)

        return flight_mask.int()


    def get_ssp_envs(self):
        """Get a masking tensor with a 1 for each environment in the flight phase."""
        # Get flight phase domain index
        ssp_domain_idx = self.gait_config.domain_name_to_idx["single_support"]

        # Create boolean mask where current_domains equals flight_phase index
        ssp_mask = (self.current_domains == ssp_domain_idx)

        return ssp_mask.int()

    def get_dsp_envs(self):
        """Get a masking tensor with a 1 for each environment in the flight phase."""
        # Get flight phase domain index
        dsp_domain_idx = self.gait_config.domain_name_to_idx["double_support"]

        # Create boolean mask where current_domains equals flight_phase index
        dsp_mask = (self.current_domains == dsp_domain_idx)

        return dsp_mask.int()
