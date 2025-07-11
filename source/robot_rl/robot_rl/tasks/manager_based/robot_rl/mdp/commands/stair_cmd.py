import torch,math
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_from_euler_xyz,quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv, quat_apply
from .hlip_cmd import HLIPCommandTerm, euler_rates_to_omega, _transfer_to_global_frame, _transfer_to_local_frame
from .ref_gen import bezier_deg, calculate_cur_swing_foot_pos_stair, calculate_cur_swing_foot_pos
from .clf_cmd.clf import CLF
from .hlip_batch import HLIPBatch
from robot_rl.tasks.manager_based.robot_rl.terrains.stair_cfg import get_step_height_at_x, get_uniform_stair_step_height_from_env

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_config.jt_traj import get_euler_from_quat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stair_cfg import StairHLIPCommandCfg

class StairCmd(HLIPCommandTerm):
    def __init__(self, cfg: "StairHLIPCommandCfg", env):
          super().__init__(cfg, env)
          self.T = self.cfg.gait_period/2*torch.ones((self.num_envs), device=self.device)
          grav = torch.abs(torch.tensor(self.env.cfg.sim.gravity[2], device=self.device))
          self.hlip_controller = HLIPBatch(grav,self.z0,self.T_ds,self.T,self.y_nom)
          
          self.tp = torch.zeros((self.num_envs), device=self.device)
          self.z_height = torch.zeros((self.num_envs), device=self.device)
          self.stance_foot_box_z = torch.zeros((self.num_envs), device=self.device)
          self.stance_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
          self.stance_foot_pos_0       = torch.zeros((self.num_envs, 3), device=self.device)
          self.stance_foot_ori_quat_0  = torch.zeros((self.num_envs, 4), device=self.device)
          self.stance_foot_ori_0       = torch.zeros((self.num_envs, 3), device=self.device)
          self.swing2stance_foot_pos_0 = torch.zeros((self.num_envs, 3), device=self.device)


    def update_z_height(self, Ux: torch.Tensor, Uy: torch.Tensor) -> torch.Tensor:
          """
          Compute and return the stair height under a desired foot target, where Ux and Uy
          are offsets in the stance-foot frame. Analytically evaluates the MeshInvertedPyramid
          stair configuration without raycasts.

          Args:
               Ux (Tensor[N]): local X offsets in stance-foot frame
               Uy (Tensor[N]): local Y offsets in stance-foot frame

          Returns:
               height_under_foot (Tensor[N]): absolute world Z heights at each target
          """
          # 1) Terrain importer & configs
          local_offsets = torch.stack([Ux, Uy, torch.zeros_like(Ux)], dim=-1)
          desired_world = self.stance_foot_pos_0 + local_offsets           # (N,3)
          
          box_center, box_bounds_lo, box_bounds_hi, stance_foot_box_center = self.check_height(local_offsets)
          self.z_height = box_center[:, 2] - stance_foot_box_center[:, 2]
          self.stance_foot_box_z = stance_foot_box_center[:, 2]
          self.target_foot_box_center = box_center
          self.target_foot_box_bounds_lo = box_bounds_lo
          self.target_foot_box_bounds_hi = box_bounds_hi

          desired_world[:, 2] = box_center[:, 2]


          if self.cfg.debug_vis:
               print(f"z_height: {self.z_height}, stance_foot_box_center: {self.stance_foot_box_z}, box_center: {box_center[:, 2]}")


    def update_Stance_Swing_idx(self):
          base_velocity = self.env.command_manager.get_command("base_velocity")  # (N, 3)
          cfg = self.env.cfg.scene.terrain.terrain_generator.sub_terrains['stairs']
          stair_width = cfg.step_width

          # Get stair step heights per environment
          env_origins = self.env.scene.env_origins
          stair_heights = get_uniform_stair_step_height_from_env(env_origins, cfg)
          self.z_height = stair_heights  # shape: (N,)

          default_Tswing = 0.4
          epsilon = 1e-5

          # Compute Tswing for stair cases
          Tswing_stair = stair_width / torch.clamp(base_velocity[:, 0], min=epsilon)

          # If no stair height, fallback to default swing time
          T_swing = torch.where(
          self.z_height == 0,
               torch.full_like(base_velocity[:, 0], default_Tswing),
               Tswing_stair
          )

          # Clamp to safe range
          T_swing = torch.clamp(T_swing, min=0.3, max=0.6)
          self.T = T_swing

          # Compute normalized phase
          self.tp = (self.env.sim.current_time % (2 * T_swing)) / (2 * T_swing)
          phi_c = torch.sin(2 * torch.pi * self.tp) / torch.sqrt(torch.sin(2 * torch.pi * self.tp)**2 + self.T)

        # 1) compute new stance & swing indices
          new_stance_idx = (phi_c < 0).long()     # or however you get it
          self.swing_idx = 1 - new_stance_idx

          # 2) grab world-frame feet
          foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]   # [B,2,3]
          foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]  # [B,2,4]
          batch      = torch.arange(self.num_envs, device=foot_pos_w.device)

          # 3) pick out the “candidate” new pose/quaternion/euler/swing2stance
          pos      = foot_pos_w[batch, new_stance_idx]             # [B,3]
          quat     = foot_ori_w[batch, new_stance_idx]             # [B,4]
          euler    = get_euler_from_quat(quat)               # [B,3]
          rel      = foot_pos_w[batch, self.swing_idx] - pos      # [B,3]
          s2s      = _transfer_to_local_frame(rel, quat)          # [B,3]

          # 4) build a mask of “which envs actually flipped stance”
          changed = new_stance_idx != self.stance_idx              # [B] bool

          # 5) stash the new stance_idx
          self.stance_idx = new_stance_idx

          # 6) **only** overwrite those entries that changed
          self.stance_foot_pos_0      [changed] = pos      [changed]
          self.stance_foot_ori_quat_0 [changed] = quat     [changed]
          self.stance_foot_ori_0      [changed] = euler    [changed]
          self.swing2stance_foot_pos_0[changed] = s2s      [changed]


          self.phase_var = torch.where(
               self.tp < 0.5,
               2.0 * self.tp,
               2.0 * self.tp - 1.0
               )
          self.cur_swing_time = self.phase_var*T_swing


    def generate_upper_body_ref(self):
        # phase: [B]
        forward_vel = self.env.command_manager.get_command("base_velocity")[:, 0]
        N = forward_vel.shape[0]
        phase = 2 * torch.pi * self.tp
        # make it [B,1] so phase+offset broadcasts to [B,9]
        # phase = torch.ones((N,1),device=self.device) * phase

        # fetch forward_vel: [B]
        
        # unpack your cfg scalars
        sh_pitch0, sh_roll0, sh_yaw0 = self.cfg.shoulder_ref
        elb0 = self.cfg.elbow_ref
        waist_yaw0 = self.cfg.waist_yaw_ref

        # build every amp as a [B] tensor
        sh_pitch_amp = sh_pitch0 * forward_vel          # [B]
        sh_roll_amp  = sh_roll0  * torch.ones_like(forward_vel)
        sh_yaw_amp   = sh_yaw0   * torch.ones_like(forward_vel)
        elb_amp      = elb0      * forward_vel
        waist_amp    = waist_yaw0 * torch.ones_like(forward_vel)

        # stack into [B,9]
        amp = torch.stack([
            waist_amp,
            sh_pitch_amp, sh_pitch_amp,
            sh_roll_amp,  sh_roll_amp,
            sh_yaw_amp,   sh_yaw_amp,
            elb_amp,      elb_amp,
        ], dim=1).to(self.device)

        # your sign & offset stay [9] each
        sign = torch.tensor([
            1,         # waist_yaw
            1, -1,     # L/R shoulder_pitch
            1, -1,     # L/R shoulder_roll
            1, -1,     # L/R shoulder_yaw
            1, -1,     # L/R elbow
        ], device=self.device)

        offset = torch.tensor([
            torch.pi,      # waist_yaw
            torch.pi/2,    # L_sh_pitch
            torch.pi/2,    # R_sh_pitch
            torch.pi/2,    # L_sh_roll
            torch.pi/2,    # R_sh_roll
            0,             # L_sh_yaw
            0,             # R_sh_yaw
            torch.pi/2,    # L_elbow
            torch.pi/2,    # R_elbow
        ], device=self.device)

        # joint offsets: [B,9]
        joint_offset = self.robot.data.default_joint_pos[:, self.upper_body_joint_idx]

        # refs: everything now broadcast to [B,9]
        offset = offset.unsqueeze(0).expand(N, -1)
  
        ref     = amp * sign * torch.sin(phase.unsqueeze(-1) + offset) + joint_offset

        # velocity
        dphase_dt = 2 * torch.pi / (2*(self.T - self.T_ds))  # scalar
        ref_dot = amp * sign * torch.cos(phase.unsqueeze(-1) + offset) * dphase_dt.unsqueeze(-1)

        return ref, ref_dot
    def get_actual_state(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        data = self.robot.data
        root_quat = data.root_quat_w
        batch_idx = torch.arange(self.num_envs, device=self.device)

        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]



        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ori = get_euler_from_quat(foot_ori_w[batch_idx, self.stance_idx, :])

        # Convert foot positions to the robot's yaw-aligned local frame
        swing2stance_local = _transfer_to_local_frame(
            foot_pos_w[batch_idx, self.swing_idx, :]-self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )

        # Center of mass to stance foot vector in local frame
        com_w = data.root_com_pos_w
        com2stance_local = _transfer_to_local_frame(
            com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )


        # Pelvis orientation (Euler XYZ)
        pelvis_ori = get_euler_from_quat(root_quat)

        # Foot orientations (Euler XYZ)
        swing_foot_ori = get_euler_from_quat(foot_ori_w[batch_idx,self.swing_idx,:])

        # 2. Velocities (world frame)
        com_vel_w = data.root_com_vel_w[:,0:3]
        # pelvis_omega_w = data.root_ang_vel_w
        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[batch_idx,self.stance_idx,:]
        self.stance_foot_ang_vel = foot_ang_vel_w[batch_idx,self.stance_idx,:]
        # Convert velocities to local frame
        # import pdb; pdb.set_trace()
        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)
      
        pelvis_omega_local = data.root_ang_vel_b
        # foot_lin_vel_local_stance = _transfer_to_local_frame(
        #     foot_lin_vel_w[:,self.stance_idx,:], self.stance_foot_ori_quat_0
        # )
        foot_lin_vel_local_swing = _transfer_to_local_frame(
            foot_lin_vel_w[batch_idx,self.swing_idx,:], self.stance_foot_ori_quat_0
        )

        foot_ang_vel_local_swing =quat_apply(quat_inv(foot_ori_w[batch_idx,self.swing_idx,:]), foot_ang_vel_w[batch_idx,self.swing_idx,:])
        

        swing2stance_vel = foot_lin_vel_local_swing 
    
        upper_body_joint_pos = self.robot.data.joint_pos[:, self.upper_body_joint_idx]
        upper_body_joint_vel = self.robot.data.joint_vel[:, self.upper_body_joint_idx]
        # 4. Assemble state vectors
        self.y_act = torch.cat([
            com2stance_local,
            pelvis_ori,
            swing2stance_local,
            swing_foot_ori,
            upper_body_joint_pos
        ], dim=-1)

        self.dy_act = torch.cat([
            com_vel_local,
            pelvis_omega_local,
            swing2stance_vel,
            foot_ang_vel_local_swing,
            upper_body_joint_vel
        ], dim=-1)

    def generate_orientation_ref(self, base_velocity,N):
        pelvis_euler = torch.zeros((N,3), device=self.device)
        tp_tensor = self.tp
        phase_tensor = self.phase_var
        
        roll_main_amp = 0.0  # main double bump amplitude
        roll_asym_amp = -0.05  # adds asymmetry

        
        pelvis_euler[:, 0] = (
            roll_main_amp * torch.sin(4 * torch.pi * tp_tensor) +
            roll_asym_amp * torch.sin(2 * torch.pi * tp_tensor)
        )

        #add bias based on lateral velocity
        # lateral bias
        bias_lat = torch.clamp(torch.atan(base_velocity[:,1] / 9.81),-0.15,0.15)

        # turning bias
        bias_yaw = torch.clamp(torch.atan((base_velocity[:,0]*base_velocity[:,2]) / 9.81),-0.2,0.2)

        pelvis_euler[:,0] = pelvis_euler[:,0] + bias_lat + bias_yaw

        pitch_amp = 0.02
        pelvis_euler[:,1] = self.cfg.pelv_pitch_ref + torch.sin(2*torch.pi * tp_tensor) * pitch_amp
    
        yaw_amp = 0.0
        default_yaw = yaw_amp*torch.sin(2* torch.pi * tp_tensor)
        pelvis_euler[:,2] = default_yaw + self.stance_foot_ori_0[:,2] + base_velocity[:,2] * self.cur_swing_time 

        pelvis_eul_dot = torch.zeros((N,3), device=self.device)

        dtp_dt = 1/(2*(self.T-self.T_ds))
        dphase_dt = 1/(self.T-self.T_ds)
        
        pelvis_eul_dot[:, 0] = (
            roll_main_amp * 4 * torch.pi * torch.cos(4 * torch.pi * tp_tensor) * dtp_dt +
            roll_asym_amp * 2 * torch.pi * torch.cos(2 * torch.pi * tp_tensor) * dtp_dt
        )

        pelvis_eul_dot[:,1] = 2*torch.pi * torch.cos(2*torch.pi * tp_tensor) * pitch_amp * dtp_dt
        pelvis_eul_dot[:,2] = base_velocity[:,2] + yaw_amp*2* torch.pi * torch.cos(2* torch.pi * tp_tensor) * dtp_dt


        foot_eul = torch.zeros((N,3), device=self.device)
        #TODO enable foot orientation control
        foot_eul[:,2] = pelvis_euler[:,2]     
        foot_eul_dot = torch.zeros((N,3), device=self.device)
        foot_eul_dot[:,2] = pelvis_eul_dot[:,2]

        return pelvis_euler, pelvis_eul_dot, foot_eul, foot_eul_dot


    def generate_reference_trajectory(self):
          base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,3)
          N = base_velocity.shape[0]
          T = self.T
          batch_idx = torch.arange(self.num_envs, device=self.device) 
          Xdes, Ux, Ydes, Uy = self.hlip_controller.compute_orbit(
               T=T,cmd=base_velocity)

          Uy = Uy[batch_idx,self.stance_idx]
          


          foot_target = torch.stack([Ux,Uy,torch.zeros((N), device=self.device)], dim=-1)

          # based on yaw velocity, update com_pos_des, com_vel_des, foot_target,
          delta_psi = base_velocity[:,2] * self.cur_swing_time
          q_delta_yaw = quat_from_euler_xyz(
               torch.zeros_like(delta_psi),               # roll=0
               torch.zeros_like(delta_psi),               # pitch=0
               delta_psi                                  # yaw=Δψ
          ) 

          # foot_target_yaw_adjusted = quat_apply(q_delta_yaw, foot_target)  # [B,3]
          # self.update_z_height(foot_target[:,0], foot_target[:,1])
          cfg = self.env.cfg.scene.terrain.terrain_generator.sub_terrains['stairs']
         
          #clip based on the kinematics range
          foot_target[:,1] = torch.sign(Uy) * torch.clamp(torch.abs(foot_target[:,1]), min=self.cfg.foot_target_range_y[0], max=self.cfg.foot_target_range_y[1])

          
          # import pdb; pdb.set_trace()
          self.hlip_controller.compute_orbit(self.T, base_velocity)
          #select init and Xdes, Ux, Ydes, Uy
          x0 = self.hlip_controller.x_init
          y0 = self.hlip_controller.y_init[batch_idx,self.stance_idx]

          # import pdb; pdb.set_trace()
          com_x, com_xd = self.hlip_controller._compute_desire_com_trajectory(
               cur_time=self.cur_swing_time,
               Xdesire=x0,
          )
          com_y, com_yd = self.hlip_controller._compute_desire_com_trajectory(
               cur_time=self.cur_swing_time,
               Xdesire=y0,
          )
          # Concatenate x and y components
          com_z = self.com_z + self.z_height * (2 * self.phase_var - 1)
          com_zd = torch.ones((N), device=self.device) * self.z_height/T
          com_pos_des = torch.stack([com_x, com_y,com_z], dim=-1)  # Shape: (N,2)
          com_vel_des = torch.stack([com_xd, com_yd,com_zd], dim=-1)  # Shape: (N,2)


          # clip foot target based on kinematic range
          self.foot_target = foot_target[:,0:2]
      
 
          # if going down stairs, no need to modify z_sw_max only modify z_sw_neg
          z_sw_max_tensor = torch.where(self.z_height < 0, self.cfg.z_sw_max, self.cfg.z_sw_max +self.z_height)
          # z_sw_max_tensor = torch.where(z_sw_max_tensor<start_box_center[:,2], self.cfg.z_sw_max +delta_z,z_sw_max_tensor)
          z_sw_neg_tensor = self.cfg.z_sw_min + self.z_height

          # Create horizontal control points with batch dimension
          horizontal_control_points = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=self.device).repeat(N, 1)  # Shape: (N, 5)

          # Create tensors with batch dimension N
          phase_var_tensor = self.phase_var
          T_tensor = self.T
          five_tensor = torch.tensor(5, device=self.device)

          bht = bezier_deg(
               0,phase_var_tensor, T_tensor, horizontal_control_points, five_tensor
          )

          # Convert scalar parameters to tensors with batch dimension N

          z_init = self.swing2stance_foot_pos_0[:,2]
          # Convert bht to tensor if it's not already
          bht_tensor = torch.tensor(bht, device=self.device) if not isinstance(bht, torch.Tensor) else bht

          # check if sw_z is actually above the stair
          
     
          sign = torch.sign(foot_target[:, 1])
          foot_pos, sw_z = calculate_cur_swing_foot_pos_stair(
               bht_tensor, z_init, z_sw_max_tensor, phase_var_tensor,-Ux, sign*self.cfg.y_nom,T_tensor, z_sw_neg_tensor,
               foot_target[:, 0], foot_target[:, 1]
          )

          foot_pos = foot_pos
          sw_z = sw_z

          dbht = bezier_deg(1, phase_var_tensor, T_tensor, horizontal_control_points, five_tensor)
          foot_vel = torch.zeros((N,3), device=self.device)
          foot_vel[:,0] = -dbht * -foot_target[:,0]+ dbht * foot_target[:,0]
          foot_vel[:,1] = -dbht * foot_target[:,1] + dbht * foot_target[:,1]
          foot_vel[:,2] = sw_z.squeeze(-1)  # Remove the last dimension to match foot_vel[:,2] shape


          upper_body_joint_pos, upper_body_joint_vel = self.generate_upper_body_ref()
          pelvis_euler, pelvis_eul_dot, foot_eul, foot_eul_dot = self.generate_orientation_ref(base_velocity,N)

          omega_ref = euler_rates_to_omega(pelvis_euler, pelvis_eul_dot)
          omega_foot_ref = euler_rates_to_omega(foot_eul, foot_eul_dot)  # (N,3)
          #setup up reference trajectory, com pos, pelvis orientation, swing foot pos, ori
          self.y_out = torch.cat([com_pos_des, pelvis_euler, foot_pos, foot_eul,upper_body_joint_pos], dim=-1)
          self.dy_out = torch.cat([com_vel_des, omega_ref, foot_vel, omega_foot_ref,upper_body_joint_vel], dim=-1)


        
        
        