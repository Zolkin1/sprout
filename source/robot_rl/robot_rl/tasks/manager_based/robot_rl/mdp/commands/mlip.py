from math import sqrt, cosh, sinh, exp
from typing import Union
import torch


class MLIP_3D:
   def __init__(self, 
                 num_envs:int, 
                 grav:float, 
                 z0:float, 
                 TOA:float,
                 TFA:float,
                 TUA:float,
                 footlength:float,
                 use_momentum=True):
        self.use_momentum = use_momentum
        self.N = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.footlength = footlength

        self.l_heel2toe =  footlength
        self.l_flat = 0.0
        self.l_toe2heel = -footlength


        self.grav = grav
        
        # Mode constants
        self.mode_heel2toe = 1
        self.mode_flat = 0
        self.mode_toe2heel = -1
        
        self.update_mlip(z0, TOA, TFA, TUA)

        
   def update_mlip(self, z0, TOA, TFA, TUA):
      self.z0 = z0
      self.TOA = TOA
      self.TFA = TFA
      self.TUA = TUA
      self.T = TOA + TFA + TUA
      self.lam = sqrt(self.grav / z0 ) 

      if self.use_momentum:
        A = torch.tensor([
            [0, 1/self.z0, 0],
            [self.grav, 0, -self.grav],
            [0, 0, 0]
        ], dtype=torch.float32, device=self.device)
      else:
        A = torch.tensor([
            [0, 1, 0],
            [self.grav/self.z0, 0, -self.grav/self.z0],
            [0, 0, 0]
        ], dtype=torch.float32, device=self.device)
    
      self.A = A

      self.A2_S2S_h2t, self.B2_S2S_h2t, self.C2_S2S_h2t = self.get_abc_s2s(self.l_heel2toe)
      self.A2_S2S_flat, self.B2_S2S_flat, self.C2_S2S_flat = self.get_abc_s2s(self.l_flat)
      self.A2_S2S_t2h, self.B2_S2S_t2h, self.C2_S2S_t2h = self.get_abc_s2s(self.l_toe2heel)


   def update_desired_walking(self, vel: torch.Tensor, 
                              stepwidth: Union[float, torch.Tensor],
                              mask_forward: torch.Tensor, 
                              mask_backward: torch.Tensor, 
                              mask_flat: torch.Tensor):
      """
      Update desired walking parameters for N environments.
      
      Args:
          vel: Tensor of shape [N, 2] with velocities [vel_x, vel_y]
          stepwidth: Tensor of shape [N] with step widths or float
      """

      # Create l_tensor based on velocity conditions
      self.l_tensor = torch.zeros(self.N, device=self.device)
      self.l_tensor[mask_forward] = self.l_heel2toe
      self.l_tensor[mask_backward] = self.l_toe2heel
      self.l_tensor[mask_flat] = self.l_flat

      # Create identity matrices for batch solving
      eye2 = torch.eye(2, device=self.device)
      
      # ========== P1 Orbit (Sagittal plane) ==========
      # Heel to toe desired u
      self.udes_p1_h2t = vel[:, 0] * self.T - self.l_heel2toe  # [N]

      # Left-hand side (same for all envs)
      lhs_h2t = eye2 - self.A2_S2S_h2t  # [2,2]

      # Compute B u + C
      rhs_h2t = (self.B2_S2S_h2t @ self.udes_p1_h2t.unsqueeze(0)).T.unsqueeze(-1) + self.C2_S2S_h2t.unsqueeze(0)  # [N,2,1]

      # Solve (I-A)x = Bu + C
      self.xdes_p1_h2t = torch.linalg.solve(
          lhs_h2t.unsqueeze(0).expand(self.N, -1, -1),  # [N,2,2]
          rhs_h2t
      ).squeeze(-1)  # [N,2]

      
      # Flat foot
      self.udes_p1_flat = vel[:, 0] * self.T - self.l_flat  # [N]
      lhs_flat = eye2 - self.A2_S2S_flat  # [2, 2]
      rhs_flat = (self.B2_S2S_flat @ self.udes_p1_flat.unsqueeze(0)).T.unsqueeze(-1) + self.C2_S2S_flat.unsqueeze(0)  # [N, 2, 1]
      self.xdes_p1_flat = torch.linalg.solve(
          lhs_flat.unsqueeze(0).expand(self.N, -1, -1),  # [N, 2, 2]
          rhs_flat
      ).squeeze(-1)  # [N, 2]
      
      # Toe to heel
      self.udes_p1_t2h = vel[:, 0] * self.T - self.l_toe2heel  # [N]
      lhs_t2h = eye2 - self.A2_S2S_t2h  # [2, 2]
      rhs_t2h = (self.B2_S2S_t2h @ self.udes_p1_t2h.unsqueeze(0)).T.unsqueeze(-1) + self.C2_S2S_t2h.unsqueeze(0)  # [N, 2, 1]
      self.xdes_p1_t2h = torch.linalg.solve(
          lhs_t2h.unsqueeze(0).expand(self.N, -1, -1),  # [N, 2, 2]
          rhs_t2h
      ).squeeze(-1)  # [N, 2]
      
      self.xdes_p1 = self.xdes_p1_flat
      self.xdes_p1[mask_backward] = self.xdes_p1_t2h[mask_backward]
      self.xdes_p1[mask_forward] = self.xdes_p1_h2t[mask_forward]
      self.udes_p1 = self.udes_p1_flat
      #for RL, since we do not specify contact location, add back footlength
      self.udes_p1[mask_backward] = self.udes_p1_t2h[mask_backward] + self.l_toe2heel
      self.udes_p1[mask_forward] = self.udes_p1_h2t[mask_forward] + self.l_heel2toe
      self.xdes_p1[mask_backward, 0] += self.l_toe2heel/2.0
      self.xdes_p1[mask_forward, 0] += self.l_heel2toe/2.0

      # ========== P2 Orbit (Lateral plane) ==========
      # Left and right step
      # Convert stepwidth to tensor with same shape as udes_p1
      if isinstance(stepwidth, float):
          stepwidth = torch.full_like(self.udes_p1, stepwidth)
    
      self.udes_p2_left = vel[:, 1] * self.T - stepwidth  # [N]
      self.udes_p2_right = vel[:, 1] * self.T + stepwidth  # [N]
      
      # Compute A^2 for double step
      A2_squared = self.A2_S2S_flat @ self.A2_S2S_flat  # [2, 2]
      lhs_p2 = eye2 - A2_squared  # [2, 2]
      
      # Left foot desired state
      # RHS = A*B*u_left + B*u_right + A*C + C
      rhs_left = (
          (self.A2_S2S_flat @ self.B2_S2S_flat) * self.udes_p2_left.unsqueeze(-1).unsqueeze(-1) +  # [N, 2, 1]
          self.B2_S2S_flat * self.udes_p2_right.unsqueeze(-1).unsqueeze(-1) +  # [N, 2, 1]
          (self.A2_S2S_flat @ self.C2_S2S_flat).unsqueeze(0) +  # [1, 2, 1] -> [N, 2, 1]
          self.C2_S2S_flat.unsqueeze(0)  # [1, 2, 1] -> [N, 2, 1]
      )
      self.xdes_p2_left = torch.linalg.solve(
          lhs_p2.unsqueeze(0).expand(self.N, -1, -1),  # [N, 2, 2]
          rhs_left
      ).squeeze(-1)  # [N, 2]
      
      # Right foot desired state
      # RHS = A*B*u_right + B*u_left + A*C + C
      rhs_right = (
          (self.A2_S2S_flat @ self.B2_S2S_flat) * self.udes_p2_right.unsqueeze(-1).unsqueeze(-1) +  # [N, 2, 1]
          self.B2_S2S_flat * self.udes_p2_left.unsqueeze(-1).unsqueeze(-1) +  # [N, 2, 1]
          (self.A2_S2S_flat @ self.C2_S2S_flat).unsqueeze(0) +  # [1, 2, 1] -> [N, 2, 1]
          self.C2_S2S_flat.unsqueeze(0)  # [1, 2, 1] -> [N, 2, 1]
      )
      self.xdes_p2_right = torch.linalg.solve(
          lhs_p2.unsqueeze(0).expand(self.N, -1, -1),  # [N, 2, 2]
          rhs_right
      ).squeeze(-1)  # [N, 2]
      
   def get_aconv_t(self, T):
      """Get convolution matrix for time T using explicit formula"""
      Aconv = torch.zeros((3, 3), device=self.device)
      
      if self.use_momentum:
          sinh_term = torch.sinh(torch.tensor(T * self.lam, device=self.device))
          sinh_half_term = torch.sinh(torch.tensor((T * self.lam) / 2, device=self.device))
  
          Aconv[0, 0] = sinh_term / self.lam
          Aconv[0, 1] = (2 * sinh_half_term**2) / (self.lam**2 * self.z0)
          Aconv[0, 2] = T - sinh_term / self.lam
          
          Aconv[1, 0] = 2 * self.z0 * sinh_half_term**2
          Aconv[1, 1] = sinh_term / self.lam
          Aconv[1, 2] = -2 * self.z0 * sinh_half_term**2
          
          Aconv[2, 2] = T
      else:
          sinh_term = torch.sinh(torch.tensor(T * self.lam, device=self.device))
          sinh_half_term = torch.sinh(torch.tensor((T * self.lam) / 2, device=self.device))
  
          Aconv[0, 0] = sinh_term / self.lam
          Aconv[0, 1] = 2 * sinh_half_term**2 / self.lam**2
          Aconv[0, 2] = T - sinh_term / self.lam
          
          Aconv[1, 0] = 2 * sinh_half_term**2
          Aconv[1, 1] = sinh_term / self.lam
          Aconv[1, 2] = -2 * sinh_half_term**2
          
          Aconv[2, 2] = T
      return Aconv
   

   
   def get_expm_at(self, T):
      """Get exp(At) using explicit formula"""
      cosh_term = cosh(T * self.lam)
      sinh_term = sinh(T * self.lam)
      exp_term  = exp(T * self.lam)

      if self.use_momentum:
         AT = torch.tensor([
             [cosh_term, (self.lam * sinh_term) / self.grav, -(exp(-T * self.lam) * (exp_term - 1)**2) / 2],
             [(self.grav * sinh_term) / self.lam, cosh_term, -(self.grav * sinh_term) / self.lam],
             [0, 0, 1]
         ],device=self.device)
      else:
         AT = torch.tensor([
             [cosh_term, (exp_term - exp(-T * self.lam))/(2*self.lam), -(exp(-T * self.lam) * (exp_term - 1)**2) / 2],
             [self.lam * sinh_term, cosh_term, -self.lam * sinh_term],
             [0, 0, 1]
         ], device=self.device)
      return AT
           
   def get_abc_s2s(self, l):
      """Get step-to-step A, B, C matrices for foot contact length l"""
      T_eps = 0.01
  
      # Use torch.linalg.matrix_exp instead of scipy.expm
      Abar_OA = self.get_expm_at(self.TOA)
      BOA = torch.zeros((3, 1), device=self.device)
      BOA[2, 0] = 1.0 / self.TOA if self.TOA > T_eps else 0.0
      Aconv_OA = self.get_aconv_t(self.TOA)
      Bbar_OA = Aconv_OA @ BOA
  
      Bdelta = torch.tensor([[-1.], [0.], [-1.]], device=self.device)
      Bdelta[2, 0] = Bdelta[2, 0] if self.TOA > T_eps else 0.0
      Cdelta = torch.tensor([[-l], [0.], [-l]], device=self.device)

      Abar_FA = self.get_expm_at(self.TFA)
      BFA = torch.zeros((3, 1), device=self.device)
      BFA[2, 0] = 1.0 / self.TFA if self.TFA > T_eps else 0.0
      Aconv_FA = self.get_aconv_t(self.TFA)
      Cbar_FA = Aconv_FA @ BFA * l

      Abar_UA = self.get_expm_at(self.TUA)

      A3s2s = Abar_UA @ Abar_FA @ Abar_OA
      B3s2s = Abar_UA @ Abar_FA @ (Bbar_OA + Bdelta)
      C3s2s = Abar_UA @ Abar_FA @ Cdelta + Abar_UA @ Cbar_FA
  
      As2s = A3s2s[:2, :2]
      Bs2s = B3s2s[:2, :]
      Cs2s = C3s2s[:2, :]
  
      return As2s, Bs2s, Cs2s

   def get_desired_com_state(self, stance_idx: int, time_in_step: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      """
      Get desired COM state for a single environment.
      
      Args:
          stance_idx: 0=left, 1=right
          time_in_step: float - time within current step
      Returns:
          x_t_p1: Tensor of shape [N, 2] - desired sagittal state [pos, vel]
          x_t_p2: Tensor of shape [N, 2] - desired lateral state [pos, vel]
      """
      # Select desired states based on stance, if current left stance, UA minus is right stance
      xdes_ua_minus_p2 = self.xdes_p2_right if stance_idx == 0 else self.xdes_p2_left
      udes_p2 = self.udes_p2_right if stance_idx == 0 else self.udes_p2_left

      Nzeros = torch.zeros_like(self.udes_p1, device=self.device)

      # Create delta terms [N, 3]
      Cdelta = torch.stack([
          -self.l_tensor, 
          Nzeros,
          -self.l_tensor
      ], dim=-1)  # [N, 3]
      
      Bdelta = torch.tensor([-1., 0., -1.], device=self.device)  # [3]
      
      # Create 3D desired states
      xdes_p1_3 = torch.cat([self.xdes_p1, Nzeros.unsqueeze(-1)], dim=-1)  # [N, 3]
      xdes_p2_3 = torch.cat([xdes_ua_minus_p2, Nzeros.unsqueeze(-1)], dim=-1)  # [N, 3]
      #OA+ = UA-
      xdes_oa_plus_p1 = xdes_p1_3  #[self.N, 3]
      xdes_oa_plus_p2 = xdes_p2_3   #[self.N, 3]


      # Initialize outputs
      x_t_p1 = torch.zeros(self.N, 2, device=self.device)
      x_t_p2 = torch.zeros(self.N, 2, device=self.device)

      dpzmp_oa_p1 = self.udes_p1 / self.TOA if self.TOA > 0.01 else torch.zeros_like(self.udes_p1) # [N]
      dpzmp_oa_p2 = udes_p2 / self.TOA if self.TOA > 0.01 else torch.zeros_like(udes_p2) # [N]
      dpzmp_fa_p1 = self.l_tensor / self.TFA if self.TFA > 0.01 else torch.zeros_like(self.udes_p1) # [N]
      
      # ========== OA Phase ==============
      if (time_in_step >= self.TFA + self.TUA) & (time_in_step <= self.T):
          t_oa = time_in_step - (self.TFA + self.TUA)
          x_t_p1 = self.get_mlip_sol2(t_oa, xdes_oa_plus_p1, dpzmp_oa_p1)
          x_t_p2 = self.get_mlip_sol2(t_oa, xdes_oa_plus_p2, dpzmp_oa_p2)
      # ========== FA Phase ==============
      elif time_in_step >= 0. and time_in_step < self.TFA:
         #FA plus = OA minus + Bdelta * u + Cdelta
         Bdelta[2] = 0.0 if self.TOA <= 0.01 else Bdelta[2]
         x_fa_plus_p1 = self.get_mlip_sol3(self.TOA, xdes_oa_plus_p1, dpzmp_oa_p1) + Bdelta.unsqueeze(0) * self.udes_p1.unsqueeze(-1) + Cdelta  # [N, 3]
         x_fa_plus_p2 = self.get_mlip_sol3(self.TOA, xdes_oa_plus_p2, dpzmp_oa_p2) + Bdelta.unsqueeze(0) * udes_p2.unsqueeze(-1) # [N, 3]
         t_fa = time_in_step
         x_t_p1 = self.get_mlip_sol2(t_fa, x_fa_plus_p1, dpzmp_fa_p1)
         x_t_p2 = self.get_mlip_sol2(t_fa, x_fa_plus_p2, Nzeros)
      # ========== UA Phase ==============
      elif time_in_step >= self.TFA and time_in_step < self.TFA + self.TUA:
         t_ua = time_in_step - self.TFA
         xdes_ua_minus_p2 = self.xdes_p2_left if stance_idx == 0 else self.xdes_p2_right
         xdes_p2_3 = torch.cat([xdes_ua_minus_p2, Nzeros.unsqueeze(-1)], dim=-1)  # [N, 3]
         x_t_p1 = self.get_mlip_sol2(-self.TUA + t_ua, xdes_p1_3, Nzeros)
         x_t_p2 = self.get_mlip_sol2(-self.TUA + t_ua, xdes_p2_3, Nzeros)
      else:
         raise ValueError("time_in_step is out of bounds")

      return x_t_p1[:,0],x_t_p1[:,1], x_t_p2[:,0], x_t_p2[:,1]

   def get_desired_foot_placement(self, stance_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
      udes_p2 = self.udes_p2_left if stance_idx == 0 else self.udes_p2_right
      return self.udes_p1, udes_p2

   def get_mlip_sol3(self, t, X0, dpzmp):
    """
    Get 3D MLIP solution at time t for batched environments.
    
    Args:
        t: Tensor of shape [N] - time values
        X0: Tensor of shape [N, 3] - initial state
        dpzmp: Tensor of shape [N] - ZMP rate control input
        
    Returns:
        sol3: Tensor of shape [N, 3]
    """
    device = X0.device
    N = X0.shape[0]
    
    if isinstance(t, (int, float)):
        t = torch.full((N,), t, device=device)
    
    unique_times = torch.unique(t)
    sol3 = torch.zeros(N, 3, device=device)
    B = torch.tensor([[0.], [0.], [1.]], device=device)
    
    for t_val in unique_times:
        mask = t == t_val
        if not mask.any():
            continue
            
        t_scalar = t_val.item()
        # Pure PyTorch
        expAt = self.get_expm_at(t_scalar)
        Aconv = self.get_aconv_t(t_scalar)
        
        X0_masked = X0[mask]
        dpzmp_masked = dpzmp[mask]
        
        # sol3 = exp(At) @ X0 + Aconv @ B * dpzmp
        term1 = (expAt @ X0_masked.T).T  # [N_masked, 3]
        term2 = (Aconv @ B).squeeze() * dpzmp_masked.unsqueeze(-1)  # [N_masked, 3]
        
        sol3[mask] = term1 + term2
    
    return sol3

   def get_mlip_sol2(self, t, X0, dpzmp):
       """
       Get 2D MLIP solution at time t.
       
       Args:
           t: Tensor of shape [N] - time values
           X0: Tensor of shape [N, 3] - initial state
           dpzmp: Tensor of shape [N] - ZMP rate control input
           
       Returns:
           sol2: Tensor of shape [N, 2] - [pos, vel]
       """
       sol3 = self.get_mlip_sol3(t, X0, dpzmp)
       return sol3[:, :2]
    
if __name__ == "__main__":
   TOA=0.
   TFA=0.
   TUA=0.4
   footlength=0.17
   mlip = MLIP_3D(
       num_envs=5,
       grav=9.81,
       z0=0.75,
       TOA=TOA,
       TFA=TFA,
       TUA=TUA,
       footlength=footlength,
       use_momentum=False
   )
   vel = torch.tensor([[0.5, 0.1],
                       [0.0, 0.5],
                       [-0.5, -0.1],
                       [0.3, 0.0],
                       [-0.3, 0.0]], device=mlip.device)
   stepwidth = 0.25
   # mask_forward = vel[:,0] > (TOA+TFA+TUA)*footlength
   # mask_backward = vel[:,0] < -(TOA+TFA+TUA)*footlength
   # mask_flat = ~(mask_forward | mask_backward)
   mask_forward = torch.full((5,), False, device=mlip.device)
   mask_backward = torch.full((5,), False, device=mlip.device)
   mask_flat = torch.full((5,), True, device=mlip.device)
   mlip.update_desired_walking(vel, stepwidth, mask_forward, mask_backward, mask_flat)
   print("Desired sagittal states (xdes_p1):", mlip.xdes_p1)
   print("Desired sagittal foot step (udes_p1):", mlip.udes_p1)
   print("Desired lateral states left (xdes_p2_left):", mlip.xdes_p2_left)
   print("Desired lateral states right (xdes_p2_right):", mlip.xdes_p2_right)
   print("Desired lateral foot step left (udes_p2_left):", mlip.udes_p2_left)
   print("Desired lateral foot step right (udes_p2_right):", mlip.udes_p2_right)
   stance_idx = 0  # Left stance
   time_in_step = 0.4

   com_x, com_dx, com_y, com_dy = mlip.get_desired_com_state(stance_idx, time_in_step)
   print("Desired sagittal state:")
   print(com_x)
   print(com_dx)
   print("Desired coronal state:")
   print(com_y)
   print(com_dy)