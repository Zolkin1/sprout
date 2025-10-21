
from typing import Union
import torch

def solve_dare_batched(A, B, Q, R, tol=1e-6, max_iter=1000):
    """
    Batched discrete-time algebraic Riccati equation (DARE) solver.

    Args:
        A: [N, n, n]
        B: [N, n, m]
        Q: [N, n, n] or [n, n]
        R: [N, m, m] or [m, m]
    Returns:
        X: [N, n, n]
    """
    N, n, _ = A.shape
    m = B.shape[-1]
    device, dtype = A.device, A.dtype

    # Broadcast Q, R to [N, ...]
    if Q.dim() == 2:
        Q = Q.expand(N, n, n)
    if R.dim() == 2:
        R = R.expand(N, m, m)

    X = Q.clone()
    for _ in range(max_iter):
        X_prev = X
        # Compute intermediate terms
        BTXB = torch.bmm(B.transpose(1, 2), torch.bmm(X, B)) + R  # [N, m, m]
        BTXB_inv = torch.linalg.inv(BTXB)
        K = torch.bmm(BTXB_inv, torch.bmm(B.transpose(1, 2), torch.bmm(X, A)))  # [N, m, n]
        # DARE update
        X = (
            torch.bmm(A.transpose(1, 2), torch.bmm(X, A))
            - torch.bmm(A.transpose(1, 2), torch.bmm(X, torch.bmm(B, K)))
            + Q
        )
        if torch.norm(X - X_prev) / torch.norm(X) < tol:
            break
    return X
def solve_dlqr_gain_batched(A, B, Q, R, tol=1e-6, max_iter=1000):
    """
    Batched DLQR gain solver.
    Returns:
        K: [N, m, n]
    """
    X = solve_dare_batched(A, B, Q, R, tol, max_iter)
    BTXB = torch.bmm(B.transpose(1, 2), torch.bmm(X, B)) + R
    BTXB_inv = torch.linalg.inv(BTXB)
    K = torch.bmm(BTXB_inv, torch.bmm(B.transpose(1, 2), torch.bmm(X, A)))
    return K


def solve_orbital_energy_batched(x: torch.Tensor, z_tilde: torch.Tensor, use_momentum: bool, g: float = 9.81):
    """
    Solve orbital energy for HLIP model in batch.

    Args:
        x: [N,2] horizontal states, is [p,v] if not use_momentum else [p,L]
        z_tilde: [N] CoM height relative to the slope z_tilde = zcom_w - z_st_w - hdes/ldes*p
        use_momentum: whether to use angular momentum L or linear velocity v
        g: gravity constant
    Returns:
        E: [N] orbital energy
    """
    if use_momentum:
        p = x[:,0]
        L = x[:,1]
        E = (L / z_tilde)**2 - (g / z_tilde) * (p**2)
    else:
        p = x[:,0]
        v = x[:,1]
        E = v**2 - (g / z_tilde) * (p**2)
    return E

def solve_time2reach_pdes_batched(x0: torch.Tensor, pdes: torch.Tensor, z_tilde: torch.Tensor, use_momentum: bool,g: float = 9.81, eps: float = 1e-9, pos_tol: float = 1e-4):
    """
    Solve time to reach desired x com position in batch.
    Args:
        x0: [N,2] horizontal states, is [p,v] if not use_momentum else [p,L]
        pdes: [N] desired com position
        z_tilde: [N] CoM height relative to the slope z_tilde = zcom_w - z_st_w - hdes/ldes*p
        use_momentum: whether to use angular momentum L or linear velocity v
        g: gravity constant
        eps: small value to avoid nan
        pos_tol: position tolerance below which T is set to 0
    Returns:
        t: [N] time to reach desired com position    
    """
    
    p0 = x0[:, 0]
    lam = torch.sqrt(g / z_tilde)  # [N]

    # skip if target and current pos are nearly identical
    near_target = (pdes - p0).abs() < pos_tol    

    forward = (pdes >= p0 + pos_tol) 
    backward = (pdes <= p0 - pos_tol) 
    
    if use_momentum:
        L0 = x0[:, 1]
        term = L0**2 * lam**2 - g**2 * p0**2 + g**2 * pdes**2
        num_first = g * pdes
        denominator = L0 * lam + g * p0
    else:
        v0 = x0[:, 1]
        term = v0**2 - lam**2 * p0**2 + lam**2 * pdes**2
        num_first = lam * pdes
        denominator = v0 + lam * p0    
    sqrt_term = torch.sqrt(torch.clamp(term, min=eps))  # avoid nan
    num_pos = num_first + sqrt_term
    num_neg = num_first - sqrt_term
    numerator = torch.where(forward | (~backward), num_pos, num_neg)
    T = torch.log(numerator / denominator) / lam
    T = torch.where(near_target, torch.zeros_like(T), T)
    
    return T

class HLIP_P2:
    z0: torch.Tensor # [N]
    TSS: torch.Tensor # [N]
    TDS: torch.Tensor # [N]
    T: torch.Tensor # [N]
    ASS: torch.Tensor # [N,2,2]
    ADS: torch.Tensor # [N,2,2]
    A_S2S: torch.Tensor # [N,2,2]
    B_S2S: torch.Tensor # [N,2,1]
    udes_p2_left: torch.Tensor # [N,1,1]
    udes_p2_right: torch.Tensor # [N,1,1]
    xdes_p2_left: torch.Tensor # [N,2,1]
    xdes_p2_right: torch.Tensor # [N,2,1]

    def __init__(self, 
                 num_envs:int, 
                 grav:float, 
                 z0:torch.Tensor, 
                 TSS:torch.Tensor,
                 TDS:torch.Tensor,
                 use_momentum=False, 
                 use_feedback=False):
        self.use_momentum = use_momentum
        self.use_feedback = use_feedback
        self.N = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grav = grav
        
        self.update_hlip(z0, TSS, TDS)
    def update_hlip(self, z0:torch.Tensor, TSS:torch.Tensor, TDS:torch.Tensor):
        self.z0 = z0
        self.TSS = TSS
        self.TDS = TDS
        self.T = TSS + TDS

        self.ASS = self.A2_ss(z0, self.grav, self.use_momentum)
        self.ADS = self.A2_ds_constant_vel(z0, self.use_momentum)
        self.A_S2S = torch.matrix_exp(TSS.view(-1,1,1) * self.ASS) @ torch.matrix_exp(TDS.view(-1,1,1) * self.ADS)

        Busw = torch.tensor([[-1.0], [0.0]], device=z0.device).expand(z0.shape[0], 2, 1) #shape [N,2,1]
        self.B_S2S = torch.matrix_exp(TSS.view(-1,1,1) * self.ASS) @ Busw
        
        if self.use_feedback:
            self.Q = torch.eye(2,2,device=self.device)
            self.R = torch.eye(1,1,device=self.device) * 100
            self.K = -solve_dlqr_gain_batched(self.A_S2S, self.B_S2S, self.Q, self.R)

    def A2_ss(self, z0, grav, use_momentum):
        # z0: [N]
        if use_momentum:
            A2 = torch.stack([
                torch.stack([torch.zeros_like(z0), 1.0 / z0], dim=-1),
                torch.stack([grav * torch.ones_like(z0), torch.zeros_like(z0)], dim=-1)
            ], dim=-2)  # [N, 2, 2]
        else:
            A2 = torch.stack([
                torch.stack([torch.zeros_like(z0), torch.ones_like(z0)], dim=-1),
                torch.stack([grav / z0, torch.zeros_like(z0)], dim=-1)
            ], dim=-2)
        return A2  # [N,2,2]

    def A2_ds_constant_vel(self, z0, use_momentum):
        if use_momentum:
            A2 = torch.stack([
                torch.stack([torch.zeros_like(z0), 1.0 / z0], dim=-1),
                torch.stack([torch.zeros_like(z0), torch.zeros_like(z0)], dim=-1)
            ], dim=-2)
        else:
            A2 = torch.stack([
                torch.stack([torch.zeros_like(z0), torch.ones_like(z0)], dim=-1),
                torch.stack([torch.zeros_like(z0), torch.zeros_like(z0)], dim=-1)
            ], dim=-2)
        return A2  # [N,2,2]

    def update_desired_walking(self, vely: torch.Tensor, stepwidth: Union[float, torch.Tensor]):
        """
        Update desired walking parameters for N environments, like x_des_left, x_des_right, u_des_left, u_des_right.

        Args:
            vely: Tensor of shape [N] with velocities [vel_y]
            stepwidth: Tensor of shape [N] with step widths or float
        """
        if isinstance(stepwidth, float):
            stepwidth = torch.full_like(vely, stepwidth)

        self.udes_p2_left = -stepwidth #[N]
        self.udes_p2_right = 2 * vely * self.T - self.udes_p2_left # [N]

        eye2 = torch.eye(2, device=self.device).expand(self.z0.shape[0], 2, 2) #[N,2,2]
        lhs = eye2 - self.A_S2S @ self.A_S2S  # [N,2,2]
        udes_left = self.udes_p2_left.view(-1, 1, 1)
        udes_right = self.udes_p2_right.view(-1, 1, 1)
        rhs_left = self.A_S2S @ self.B_S2S * udes_left + self.B_S2S * udes_right  # [N,2,1]
        rhs_right = self.A_S2S @ self.B_S2S * udes_right + self.B_S2S * udes_left # [N,2,1]


        # Solve lhs * X = rhs
        self.xdes_p2_left = torch.linalg.solve(lhs, rhs_left)   # [N,2,1]
        self.xdes_p2_right = torch.linalg.solve(lhs, rhs_right) # [N,2,1]
         
    def get_desired_com_state(self, stance_idx: torch.Tensor, time_in_step: torch.Tensor):
        """
        Get desired com state at given time_in_step and stance_idx.

        Args:
            stance_idx: Tensor of shape [N] with stance leg index (0=left, 1=right)
            time_in_step: Tensor of shape [N] with time in current step [0,TSS+TDS] 
        Returns:
            com_y: Tensor of shape [N] with desired com y position
            com_dy: Tensor of shape [N] with desired com y velocity
        """
        mask_left = (stance_idx == 0)
        mask_right = (stance_idx == 1)
        mask_ss = (time_in_step <= self.TSS)
        mask_ds = (time_in_step > self.TSS)
        y_com_state = torch.zeros((self.N, 2, 1), device=self.device)


        #if DS, integrate forward from begining of DS
        x_ds_plus = torch.zeros((self.N, 2, 1), device=self.device)
        x_ds_plus[mask_left] = self.xdes_p2_left[mask_left]
        x_ds_plus[mask_right] = self.xdes_p2_right[mask_right]
        y_com_state[mask_ds] = torch.matrix_exp((time_in_step[mask_ds] - self.TSS[mask_ds]).view(-1,1,1) * self.ADS[mask_ds]) @ x_ds_plus[mask_ds]
        
        #if SS, integrate backward from end of SS
        x_ss_minus = x_ds_plus #happens to be the same at the beginning of DS
        y_com_state[mask_ss] = torch.matrix_exp((time_in_step[mask_ss]- self.TSS[mask_ss]).view(-1,1,1) * self.ASS[mask_ss]) @ x_ss_minus[mask_ss]

        return y_com_state.squeeze(-1)[:, 0], y_com_state.squeeze(-1)[:, 1]

    def get_desired_com_state_from_x0_sagittal(self, x0: torch.Tensor, time_in_step: torch.Tensor):
        """
        Get desired com state from initial position x0 for sagittal plane

        Args:
            x0: Tensor of shape [N, 2] with initial states [p0, L0 or v0]
            time_in_step: Tensor of shape [N] with time in current step 

        Returns:
            com_x: Tensor of shape [N] with desired com x position
            com_dx: Tensor of shape [N] with desired com x velocity or momentum
        """
        t = torch.clamp(time_in_step, 0, self.TSS)
        x_com_state = torch.matrix_exp(t.view(-1,1,1) * self.ASS) @ x0.unsqueeze(-1) #[N,2,1]

        mask_ds = (time_in_step > self.TSS)
        if torch.any(mask_ds):
            x_prev = x_com_state[mask_ds] 
            x_com_state[mask_ds] = torch.matrix_exp((time_in_step[mask_ds] - self.TSS[mask_ds]).view(-1,1,1) * self.ADS[mask_ds]) @ x_prev
        return x_com_state.squeeze(-1)[:, 0], x_com_state.squeeze(-1)[:, 1]

    def get_desired_foot_placement(self, stance_idx: torch.Tensor, com_state: torch.Tensor = None):
        """Get desired foot placement based on stance index and center of mass state.

        Args:
            stance_idx (torch.Tensor): Tensor indicating the stance leg (0=left, 1=right). shape [N]
            com_state (torch.Tensor, optional): Current center of mass state. Defaults to None. shape [N, 2]

        Returns:
            torch.Tensor: Desired foot placement.[N]
        """
        udes_p2 = torch.zeros((self.N, ), device=self.device)
        udes_p2[stance_idx == 0] = self.udes_p2_left[stance_idx == 0]
        udes_p2[stance_idx == 1] = self.udes_p2_right[stance_idx == 1]

        if self.use_feedback:
            if com_state is None:
                raise ValueError("com_state must be provided when use_feedback is True")
            xdes_p2 = self.xdes_p2_left
            xdes_p2[stance_idx == 1] = self.xdes_p2_right[stance_idx == 1]
            udes_p2 += (self.K @ (xdes_p2 - com_state.unsqueeze(-1)) ).squeeze(-1).squeeze(-1)

        return udes_p2


class HLIP_3D(HLIP_P2):
    udes_p1: torch.Tensor # [N,1,1]
    xdes_p1: torch.Tensor # [N,2,1]
    def __init__(self, 
                 num_envs:int, 
                 grav:float, 
                 z0:torch.Tensor, 
                 TSS:torch.Tensor,
                 TDS:torch.Tensor,
                 use_momentum=False, 
                 use_feedback=False):
        super().__init__(num_envs, grav, z0, TSS, TDS, use_momentum, use_feedback)
    
    def update_desired_walking(self, velx: torch.Tensor, vely: torch.Tensor, stepwidth: Union[float, torch.Tensor]):
        """
        Update desired walking parameters for N environments, like x_des_left, x_des_right, u_des_left, u_des_right.

        Args:
            velx: Tensor of shape [N] with velocities [vel_x]
            vely: Tensor of shape [N] with velocities [vel_y]
            stepwidth: Tensor of shape [N] with step widths or float
        """
        super().update_desired_walking(vely, stepwidth)

        self.udes_p1 = velx * self.T  # [N]

        eye2 = torch.eye(2, device=self.device).expand(self.z0.shape[0], 2, 2) #[N,2,2]
        lhs = eye2 - self.A_S2S  # [N,2,2]
        self.udes_p1 = self.udes_p1.view(-1, 1, 1)
        rhs =  self.B_S2S * self.udes_p1  # [N,2,1]

        # Solve lhs * X = rhs
        self.xdes_p1 = torch.linalg.solve(lhs, rhs)   # [N,2,1]    
        
    def get_desired_com_state(self, stance_idx, time_in_step):
        """
        Get desired com state at given time_in_step and stance_idx.

        Args:
            stance_idx: Tensor of shape [N] with stance leg index (0=left, 1=right)
            time_in_step: Tensor of shape [N] with time in current step [0,TSS+TDS] 
        Returns:
            com_x: Tensor of shape [N] with desired com x position
            com_dx: Tensor of shape [N] with desired com x velocity
            com_y: Tensor of shape [N] with desired com y position
            com_dy: Tensor of shape [N] with desired com y velocity
        """
        mask_left = (stance_idx == 0)
        mask_right = (stance_idx == 1)
        mask_ss = (time_in_step <= self.TSS)
        mask_ds = (time_in_step > self.TSS)
        y_com_state = torch.zeros((self.N, 2, 1), device=self.device)
        x_com_state = torch.zeros((self.N, 2, 1), device=self.device)

        #if DS, integrate forward from begining of DS
        x_ds_plus_p2 = torch.zeros((self.N, 2, 1), device=self.device)
        x_ds_plus_p2[mask_left] = self.xdes_p2_left[mask_left]
        x_ds_plus_p2[mask_right] = self.xdes_p2_right[mask_right]
        y_com_state[mask_ds] = torch.matrix_exp((time_in_step[mask_ds] - self.TSS[mask_ds]).view(-1,1,1) * self.ADS[mask_ds]) @ x_ds_plus_p2[mask_ds]

        #if SS, integrate backward from end of SS
        x_ss_minus_p2 = x_ds_plus_p2 #happens to be the same at the beginning of DS
        y_com_state[mask_ss] = torch.matrix_exp((time_in_step[mask_ss]- self.TSS[mask_ss]).view(-1,1,1) * self.ASS[mask_ss]) @ x_ss_minus_p2[mask_ss]
        
        x_ds_plus_p1 = self.xdes_p1
        x_com_state[mask_ds] = torch.matrix_exp((time_in_step[mask_ds] - self.TSS[mask_ds]).view(-1,1,1) * self.ADS[mask_ds]) @ x_ds_plus_p1[mask_ds]
        x_ss_minus_p1 = x_ds_plus_p1 #happens to be the same at the beginning of DS
        x_com_state[mask_ss] = torch.matrix_exp((time_in_step[mask_ss]- self.TSS[mask_ss]).view(-1,1,1) * self.ASS[mask_ss]) @ x_ss_minus_p1[mask_ss]

        return x_com_state.squeeze(-1)[:, 0], x_com_state.squeeze(-1)[:, 1], y_com_state.squeeze(-1)[:, 0], y_com_state.squeeze(-1)[:, 1]

    def get_desired_foot_placement(self, stance_idx, xcom_state = None, ycom_state = None):
        """Get desired foot placement based on stance index and center of mass state.

        Args:
            stance_idx (torch.Tensor): Tensor indicating the stance leg (0=left, 1=right). shape [N]
            xcom_state (torch.Tensor, optional): Current center of mass state. Defaults to None. shape [N, 2]
            ycom_state (torch.Tensor, optional): Current center of mass state. Defaults to None. shape [N, 2]

        Returns:
            udes_p1 (torch.Tensor): Desired foot placement in x.[N]
            udes_p2 (torch.Tensor): Desired foot placement in y.[N]
        """
        udes_p2 = torch.zeros((self.N, ), device=self.device)
        udes_p2[stance_idx == 0] = self.udes_p2_left[stance_idx == 0]
        udes_p2[stance_idx == 1] = self.udes_p2_right[stance_idx == 1]
        
        udes_p1 = self.udes_p1.squeeze(-1).squeeze(-1)

        if self.use_feedback:
            if xcom_state is None or ycom_state is None:
                raise ValueError("xcom_state and ycom_state must be provided when use_feedback is True")
            xdes_p2 = self.xdes_p2_left
            xdes_p2[stance_idx == 1] = self.xdes_p2_right[stance_idx == 1]
            udes_p2 += (self.K @ (xdes_p2 - ycom_state.unsqueeze(-1)) ).squeeze(-1).squeeze(-1)
            udes_p1 += (self.K @ (self.xdes_p1 - xcom_state.unsqueeze(-1)) ).squeeze(-1).squeeze(-1)

        return udes_p1, udes_p2

    def get_pre_impact_com_states(self, com_state: torch.Tensor, time2impact: torch.Tensor):
        """
        Get pre-impact com states by integrating current com_state forward by time2impact.

        Args:
            com_state: Tensor of shape [N,2] with current com states [p,v] or [p,L]
            time2impact: Tensor of shape [N] with time to impact
        Returns:
            pre_impact_state: Tensor of shape [N,2] with pre-impact com states     
        """
        pre_impact_state = torch.matrix_exp(time2impact.view(-1,1,1) * self.ASS) @ com_state.unsqueeze(-1)
        return pre_impact_state.squeeze(-1)
        
def test_hlip_p2():
   TDS= torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   TSS= torch.tensor([0.4, 0.3, 0.2, 0.5, 0.6], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   z0 = torch.tensor([0.75, 0.65, 0.55, 0.45, 0.35], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   hlip_y = HLIP_P2(
       num_envs=5,
       grav=9.81,
       z0=z0,
       TSS=TSS,
       TDS=TDS,
       use_momentum=False,
       use_feedback=True
   )
   vel = torch.tensor([0.5, 0.5, -0.5, 0.1, -0.1], device=hlip_y.device)
   stepwidth = 0.25

   hlip_y.update_desired_walking(vel, stepwidth)

   print("Desired lateral states left (xdes_p2_left):", hlip_y.xdes_p2_left)
   print("Desired lateral states right (xdes_p2_right):", hlip_y.xdes_p2_right)
   print("Desired lateral foot step left (udes_p2_left):", hlip_y.udes_p2_left)
   print("Desired lateral foot step right (udes_p2_right):", hlip_y.udes_p2_right)
   stance_idx = torch.tensor([0, 0, 0, 0, 0], device=hlip_y.device)
   time_in_step = torch.tensor([0., 0.1, 0.2, 0.3, 0.4], device=hlip_y.device)


   com_y, com_dy = hlip_y.get_desired_com_state(stance_idx, time_in_step)
   print("Desired coronal state:")
   print(com_y)
   print(com_dy)
   
   uy = hlip_y.get_desired_foot_placement(stance_idx, torch.stack([com_y, com_dy], dim=-1))
   print("Desired foot placement uy:", uy)

   xstate = torch.stack([com_y, com_dy], dim=-1)
   E = solve_orbital_energy_batched(xstate, z0, hlip_y.use_momentum)
   print("Orbital energy E:", E)


#    xT = torch.tensor([0.5, 0.5, 0.3],device=hlip_y.device)
#    x0 = torch.tensor([[-0.1, -0.2, -0.1],[0.9, 0.8, 0.7] ],device=hlip_y.device).T
#    z0 = torch.tensor([0.67, 0.7, 0.8],device=hlip_y.device)
   
#    T = solve_time2reach_pdes_batched(x0, xT, z0, hlip_y.use_momentum)
   T = solve_time2reach_pdes_batched(xstate, hlip_y.xdes_p2_left[:,0,:].squeeze(-1), z0, hlip_y.use_momentum)
   print("Time to reach desired foot placement T:", T)

def test_hlip_3d():
   TDS= torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   TSS= torch.tensor([0.4, 0.3, 0.2, 0.5, 0.6], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   z0 = torch.tensor([0.75, 0.65, 0.55, 0.45, 0.35], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   hlip = HLIP_3D(
       num_envs=5,
       grav=9.81,
       z0=z0,
       TSS=TSS,
       TDS=TDS,
       use_momentum=True,
       use_feedback=True
   )
   velx = torch.tensor([0.5, 0.2, -0.3, 0.1, -0.1], device=hlip.device)
   vely = torch.tensor([0.5, 0.5, -0.5, 0.1, -0.1], device=hlip.device)
   stepwidth = 0.25

   hlip.update_desired_walking(velx, vely, stepwidth)
   print("Desired forward states (xdes_p1):", hlip.xdes_p1)
   print("Desired forward foot step (udes_p1):", hlip.udes_p1)
   
   print("Desired lateral states left (xdes_p2_left):", hlip.xdes_p2_left)
   print("Desired lateral states right (xdes_p2_right):", hlip.xdes_p2_right)
   print("Desired lateral foot step left (udes_p2_left):", hlip.udes_p2_left)
   print("Desired lateral foot step right (udes_p2_right):", hlip.udes_p2_right)
   
   stance_idx = torch.tensor([0, 0, 0, 0, 0], device=hlip.device)
   time_in_step = torch.tensor([0., 0.1, 0.2, 0.3, 0.4], device=hlip.device)


   com_x, com_dx, com_y, com_dy = hlip.get_desired_com_state(stance_idx, time_in_step)
   print("Desired forward state:")
   print(com_x)
   print(com_dx)
   print("Desired coronal state:")
   print(com_y)
   print(com_dy)

   ux, uy = hlip.get_desired_foot_placement(stance_idx, torch.stack([com_x, com_dx], dim=-1), torch.stack([com_y, com_dy], dim=-1))
   print("Desired foot placement ux:", ux)
   print("Desired foot placement uy:", uy)

   xstate = torch.stack([com_x, com_dx], dim=-1)
   ystate = torch.stack([com_y, com_dy], dim=-1)
   Ex = solve_orbital_energy_batched(xstate, z0, hlip.use_momentum)
   Ey = solve_orbital_energy_batched(ystate, z0, hlip.use_momentum)
   print("Orbital energy Ex:", Ex)
   print("Orbital energy Ey:", Ey)

   Tx = solve_time2reach_pdes_batched(xstate, hlip.xdes_p1[:,0,:].squeeze(-1), z0, hlip.use_momentum)
   print("Time to reach desired com Tx:", Tx) 

   Ty = solve_time2reach_pdes_batched(ystate, hlip.xdes_p2_left[:,0,:].squeeze(-1), z0, hlip.use_momentum)
   print("Time to reach desired com Ty:", Ty)
   
   pre_impact_xstate = hlip.get_pre_impact_com_states(xstate, Tx)
   pre_impact_ystate = hlip.get_pre_impact_com_states(ystate, Ty)
   print("Pre-impact com states in x:", pre_impact_xstate)
   print("Pre-impact com states in y:", pre_impact_ystate)
   
   
   Ttest = solve_time2reach_pdes_batched(xstate, hlip.xdes_p2_left[:,0,:].squeeze(-1), z0, hlip.use_momentum)
   print("Time to reach desired com Ttest:", Ttest)

if __name__ == "__main__":
   test_hlip_3d()
    #test_hlip_p2()