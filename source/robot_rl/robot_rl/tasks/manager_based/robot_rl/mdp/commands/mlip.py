import numpy as np
from scipy.linalg import expm
from enum import Enum

import torch
from torch import Tensor
class StanceStatus(Enum):
    LeftStance = 0
    RightStance = 1

class OrbitPeriod(Enum):
    P1orbit = 1
    P2orbit = 2

class MLIP_3D:
    def __init__(self, use_momentum=True):
        self.use_momentum = use_momentum
        
        # Mode constants
        self.mode_heel2toe = 1
        self.mode_flat = 0
        self.mode_toe2heel = -1
        
        # Initialize parameters structure
        self.params = self.Params(use_momentum)
        
        # Initialize P1 and P2 structures
        self.p1 = self.P1()
        self.p2 = self.P2()
        
        # Control gains (will be computed in update_mlip)
        self.Kdeadbeat_h2t = None
        self.Kdeadbeat_flat = None
        self.Kdeadbeat_t2h = None
        self.Klqr_h2t = None
        self.Klqr_flat = None
        self.Klqr_t2h = None
        
        self.orbit_period = None
        
    class Params:
        def __init__(self, use_momentum=True):
            self.use_momentum = use_momentum
            self.footlength = 0.0
            
            self.l_heel2toe = 0.0
            self.l_flat = 0.0
            self.l_toe2heel = 0.0
            
            self.z0 = 0.0
            self.TOA = 0.0  # Time On Air
            self.TFA = 0.0  # Time Foot Attack
            self.TUA = 0.0  # Time Upright Approach
            self.T = 0.0
            self.lambda_val = 0.0
            self.grav = 9.81
            
            self.A = np.zeros((3, 3))
            
            # Step-to-step matrices for different contact modes
            self.A2_S2S_h2t = np.zeros((2, 2))
            self.B2_S2S_h2t = np.zeros((2, 1))
            self.C2_S2S_h2t = np.zeros((2, 1))
            
            self.A2_S2S_flat = np.zeros((2, 2))
            self.B2_S2S_flat = np.zeros((2, 1))
            self.C2_S2S_flat = np.zeros((2, 1))
            
            self.A2_S2S_t2h = np.zeros((2, 2))
            self.B2_S2S_t2h = np.zeros((2, 1))
            self.C2_S2S_t2h = np.zeros((2, 1))
            
            self.vel_des = 0.0
            
        def get_aconv_t(self, T):
            """Get convolution matrix for time T"""
            Aconv = np.zeros((3, 3))
            if self.use_momentum:
                sinh_term = np.sinh(T * self.lambda_val)
                sinh_half_term = np.sinh((T * self.lambda_val) / 2)
                
                Aconv[0, 0] = sinh_term / self.lambda_val
                Aconv[0, 1] = (2 * sinh_half_term**2) / (self.lambda_val**2 * self.z0)
                Aconv[0, 2] = T - sinh_term / self.lambda_val
                
                Aconv[1, 0] = 2 * self.z0 * sinh_half_term**2
                Aconv[1, 1] = sinh_term / self.lambda_val
                Aconv[1, 2] = -2 * self.z0 * sinh_half_term**2
                
                Aconv[2, 2] = T
            else:
                sinh_term = np.sinh(T * self.lambda_val)
                sinh_half_term = np.sinh((T * self.lambda_val) / 2)
                
                Aconv[0, 0] = sinh_term / self.lambda_val
                Aconv[0, 1] = 2 * sinh_half_term**2 / self.lambda_val**2
                Aconv[0, 2] = T - sinh_term / self.lambda_val
                
                Aconv[1, 0] = 2 * sinh_half_term**2
                Aconv[1, 1] = sinh_term / self.lambda_val
                Aconv[1, 2] = -2 * sinh_half_term**2
                
                Aconv[2, 2] = T
            return Aconv
            
        def get_abc_s2s(self, l):
            """Get step-to-step A, B, C matrices for foot contact length l"""
            T_eps = 0.01
            
            # On Air phase
            Abar_OA = expm(self.TOA * self.A)
            BOA = np.zeros((3, 1))
            BOA[2, 0] = 1.0 / self.TOA if self.TOA > T_eps else 0.0
            Aconv_OA = self.get_aconv_t(self.TOA)
            Bbar_OA = Aconv_OA @ BOA
            
            # Delta terms
            Bdelta = np.array([[-1], [0], [-1]])
            Bdelta[2, 0] = Bdelta[2, 0] if self.TOA > T_eps else 0.0
            Cdelta = np.array([[-l], [0], [-l]])
            
            # Foot Attack phase
            Abar_FA = expm(self.TFA * self.A)
            BFA = np.zeros((3, 1))
            BFA[2, 0] = 1.0 / self.TFA if self.TFA > T_eps else 0.0
            Aconv_FA = self.get_aconv_t(self.TFA)
            Cbar_FA = Aconv_FA @ BFA * l
            
            # Upright Approach phase
            Abar_UA = expm(self.TUA * self.A)
            
            # Combine phases
            A3s2s = Abar_UA @ Abar_FA @ Abar_OA
            B3s2s = Abar_UA @ Abar_FA @ (Bbar_OA + Bdelta)
            C3s2s = Abar_UA @ Abar_FA @ Cdelta + Abar_UA @ Cbar_FA
            
            # Extract 2D components
            As2s = A3s2s[:2, :2]
            Bs2s = B3s2s[:2, :]
            Cs2s = C3s2s[:2, :]
            
            return As2s, Bs2s, Cs2s
    
    class P1:
        """Period-1 orbit parameters"""
        def __init__(self):
            # Desired walking states for different contact modes
            self.Xdes_h2t = np.zeros(2)
            self.Xdes_flat = np.zeros(2)
            self.Xdes_t2h = np.zeros(2)
            
            self.Udes_h2t = 0.0
            self.Udes_flat = 0.0
            self.Udes_t2h = 0.0
            
            self.XdesFAminus_h2t = np.zeros(2)
            self.XdesFAminus_flat = np.zeros(2)
            self.XdesFAminus_t2h = np.zeros(2)
            
            self.XdesFAplus_h2t = np.zeros(2)
            self.XdesFAplus_flat = np.zeros(2)
            self.XdesFAplus_t2h = np.zeros(2)
            
            self.StepX = np.zeros(2)
            self.mode = 0
            self.is_flat_foot = False
            
            # Control gains
            self.K_h2t = None
            self.K_flat = None
            self.K_t2h = None
    
    class P2:
        """Period-2 orbit parameters"""
        def __init__(self):
            self.XleftDes = np.zeros(2)
            self.XrightDes = np.zeros(2)
            self.Xdes = np.zeros(2)
            
            self.UleftDes = 0.0
            self.UrightDes = 0.0
            self.Udes = 0.0
            
            self.StepX = np.zeros(2)
            
            self.XdesFAminus_left = np.zeros(2)
            self.XdesFAminus_right = np.zeros(2)
            self.XdesFAplus_left = np.zeros(2)
            self.XdesFAplus_right = np.zeros(2)
            
            self.K = None
    
    def compute_desired_orbit(self, 
                              vel:Tensor, 
                              TFA:float,
                              TUA:float,
                              TOA:float,
                              ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
         # Get device from input tensor
         device = vel.device

         # Compute desired orbit
         Xdes = vel * TFA
         Udes = vel * TUA
         return Xdes.to(device), Udes.to(device), TOA.to(device)

    def init(self, z0, TOA, TFA, TUA, orbit_period, vel, footlength, stepwidth=0):
        """Initialize MLIP with given parameters"""
        self.orbit_period = orbit_period
        self.params.footlength = footlength
        self.update_mlip(z0, TOA, TFA, TUA)
        self.update_desired_walking(vel, stepwidth)
    
    def update_mlip(self, z0, TOA, TFA, TUA):
        """Update MLIP dynamics parameters"""
        self.params.z0 = z0
        self.params.TOA = TOA
        self.params.TFA = TFA
        self.params.TUA = TUA
        
        self.params.l_heel2toe = self.params.footlength
        self.params.l_toe2heel = -self.params.footlength
        self.params.l_flat = 0.0
        
        self.params.T = TOA + TFA + TUA
        self.params.lambda_val = np.sqrt(self.params.grav / self.params.z0)
        
        # Set up system matrix A
        if self.params.use_momentum:
            self.params.A = np.array([
                [0, 1/self.params.z0, 0],
                [self.params.grav, 0, -self.params.grav],
                [0, 0, 0]
            ])
        else:
            self.params.A = np.array([
                [0, 1, 0],
                [self.params.grav/self.params.z0, 0, -self.params.grav/self.params.z0],
                [0, 0, 0]
            ])
        
        # Compute step-to-step matrices
        self.params.A2_S2S_h2t, self.params.B2_S2S_h2t, self.params.C2_S2S_h2t = \
            self.params.get_abc_s2s(self.params.l_heel2toe)
        self.params.A2_S2S_flat, self.params.B2_S2S_flat, self.params.C2_S2S_flat = \
            self.params.get_abc_s2s(self.params.l_flat)
        self.params.A2_S2S_t2h, self.params.B2_S2S_t2h, self.params.C2_S2S_t2h = \
            self.params.get_abc_s2s(self.params.l_toe2heel)
        
        # Compute control gains
        self.Kdeadbeat_h2t = self.solve_deadbeat_gain(self.params.A2_S2S_h2t, self.params.B2_S2S_h2t)
        self.Kdeadbeat_flat = self.solve_deadbeat_gain(self.params.A2_S2S_flat, self.params.B2_S2S_flat)
        self.Kdeadbeat_t2h = self.solve_deadbeat_gain(self.params.A2_S2S_t2h, self.params.B2_S2S_t2h)
        
        r = 5.0
        eps = 1e-8
        
        self.Klqr_h2t = -self.solve_dlqr_gain(self.params.A2_S2S_h2t, self.params.B2_S2S_h2t, 
                                              np.eye(2), np.eye(1) * r, eps)
        self.Klqr_flat = -self.solve_dlqr_gain(self.params.A2_S2S_flat, self.params.B2_S2S_flat, 
                                               np.eye(2), np.eye(1) * r, eps)
        self.Klqr_t2h = -self.solve_dlqr_gain(self.params.A2_S2S_t2h, self.params.B2_S2S_t2h, 
                                              np.eye(2), np.eye(1) * r, eps)
        
        # Assign gains to P1
        self.p1.K_h2t = self.Klqr_h2t
        self.p1.K_flat = self.Klqr_flat
        self.p1.K_t2h = self.Klqr_t2h
        
        # Recompute flat gain with different r for P2
        r = 1.0
        Klqr_flat_p2 = -self.solve_dlqr_gain(self.params.A2_S2S_flat, self.params.B2_S2S_flat, 
                                              np.eye(2), np.eye(1) * r, eps)
        self.p2.K = Klqr_flat_p2
    
    def update_desired_walking(self, vel, stepwidth):
        """Update desired walking parameters"""
        self.params.vel_des = vel
        
        if self.orbit_period == OrbitPeriod.P1orbit:
            # P1 orbit calculations
            self.p1.Udes_h2t = self.params.vel_des * self.params.T - self.params.l_heel2toe
            self.p1.Xdes_h2t = np.linalg.solve(
                np.eye(2) - self.params.A2_S2S_h2t,
                self.params.B2_S2S_h2t.flatten() * self.p1.Udes_h2t + self.params.C2_S2S_h2t.flatten()
            )
            self.p1.XdesFAminus_h2t, self.p1.XdesFAplus_h2t = self.solve_xdes_fa_minus_plus(
                self.params.l_heel2toe, self.p1.Xdes_h2t, self.p1.Udes_h2t
            )
            
            self.p1.Udes_flat = self.params.vel_des * self.params.T - self.params.l_flat
            self.p1.Xdes_flat = np.linalg.solve(
                np.eye(2) - self.params.A2_S2S_flat,
                self.params.B2_S2S_flat.flatten() * self.p1.Udes_flat + self.params.C2_S2S_flat.flatten()
            )
            self.p1.XdesFAminus_flat, self.p1.XdesFAplus_flat = self.solve_xdes_fa_minus_plus(
                self.params.l_flat, self.p1.Xdes_flat, self.p1.Udes_flat
            )
            
            self.p1.Udes_t2h = self.params.vel_des * self.params.T - self.params.l_toe2heel
            self.p1.Xdes_t2h = np.linalg.solve(
                np.eye(2) - self.params.A2_S2S_t2h,
                self.params.B2_S2S_t2h.flatten() * self.p1.Udes_t2h + self.params.C2_S2S_t2h.flatten()
            )
            self.p1.XdesFAminus_t2h, self.p1.XdesFAplus_t2h = self.solve_xdes_fa_minus_plus(
                self.params.l_toe2heel, self.p1.Xdes_t2h, self.p1.Udes_t2h
            )
            
        elif self.orbit_period == OrbitPeriod.P2orbit:
            # P2 orbit calculations
            if stepwidth == 0:
                print("Warning: stepWidth is zero, not permitted!")
            else:
                self.p2.UleftDes = -stepwidth
                self.p2.UrightDes = 2 * self.params.vel_des * self.params.T - self.p2.UleftDes
                
                A_double = self.params.A2_S2S_flat @ self.params.A2_S2S_flat
                rhs_left = (self.params.A2_S2S_flat @ self.params.B2_S2S_flat.flatten() * self.p2.UleftDes +
                           self.params.B2_S2S_flat.flatten() * self.p2.UrightDes +
                           self.params.A2_S2S_flat @ self.params.C2_S2S_flat.flatten() +
                           self.params.C2_S2S_flat.flatten())
                
                rhs_right = (self.params.A2_S2S_flat @ self.params.B2_S2S_flat.flatten() * self.p2.UrightDes +
                            self.params.B2_S2S_flat.flatten() * self.p2.UleftDes +
                            self.params.A2_S2S_flat @ self.params.C2_S2S_flat.flatten() +
                            self.params.C2_S2S_flat.flatten())
                
                self.p2.XleftDes = np.linalg.solve(np.eye(2) - A_double, rhs_left)
                self.p2.XrightDes = np.linalg.solve(np.eye(2) - A_double, rhs_right)
                
                self.p2.XdesFAminus_left, self.p2.XdesFAplus_left = self.solve_xdes_ua_minus_plus(
                    0, self.p2.XrightDes, self.p2.UrightDes
                )
                self.p2.XdesFAminus_right, self.p2.XdesFAplus_right = self.solve_xdes_ua_minus_plus(
                    0, self.p2.XleftDes, self.p2.UleftDes
                )
        else:
            print("orbit type is wrong!")
    
    def solve_xdes_fa_minus_plus(self, l, Xdes, Udes):
        """Solve for desired states at FA- and FA+ events"""
        T_eps = 0.01
        
        # On Air phase
        Abar_OA = expm(self.params.TOA * self.params.A)
        BOA = np.zeros((3, 1))
        BOA[2, 0] = 1.0 / self.params.TOA if self.params.TOA > T_eps else 0.0
        Aconv_OA = self.params.get_aconv_t(self.params.TOA)
        Bbar_OA = Aconv_OA @ BOA
        
        # Delta terms
        Bdelta = np.array([[-1], [0], [-1]])
        Bdelta[2, 0] = Bdelta[2, 0] if self.params.TOA > T_eps else 0.0
        Cdelta = np.array([[-l], [0], [-l]])
        
        # Upright Approach phase
        Abar_UA = expm(self.params.TUA * self.params.A)
        
        Xdes3 = np.array([[Xdes[0]], [Xdes[1]], [0]])
        XdesFAminus3 = np.linalg.solve(Abar_UA, Xdes3)
        XdesFAplus3 = Abar_OA @ Xdes3 + (Bbar_OA + Bdelta) * Udes + Cdelta
        
        XdesFAminus = XdesFAminus3[:2, 0]
        XdesFAplus = XdesFAplus3[:2, 0]
        
        return XdesFAminus, XdesFAplus
    
    def solve_xdes_ua_minus_plus(self, l, Xdes, Udes):
        """Solve for desired states at UA- and UA+ events"""
        T_eps = 0.01
        
        # On Air phase
        Abar_OA = expm(self.params.TOA * self.params.A)
        BOA = np.zeros((3, 1))
        BOA[2, 0] = 1.0 / self.params.TOA if self.params.TOA > T_eps else 0.0
        Aconv_OA = self.params.get_aconv_t(self.params.TOA)
        Bbar_OA = Aconv_OA @ BOA
        
        # Delta terms
        Bdelta = np.array([[-1], [0], [-1]])
        Bdelta[2, 0] = Bdelta[2, 0] if self.params.TOA > T_eps else 0.0
        Cdelta = np.array([[-l], [0], [-l]])
        
        # Upright Approach phase
        Abar_UA = expm(self.params.TUA * self.params.A)
        
        Xdes3 = np.array([[Xdes[0]], [Xdes[1]], [0]])
        XdesUAplus3 = Abar_OA @ Xdes3 + (Bbar_OA + Bdelta) * Udes + Cdelta
        XdesUAminus3 = Abar_UA @ XdesUAplus3
        
        XdesFAminus = XdesUAminus3[:2, 0]
        XdesFAplus = XdesUAplus3[:2, 0]
        
        return XdesFAminus, XdesFAplus
    
    def get_step_size_p1_fixed_mode(self, mode):
        """Get step size for P1 orbit with fixed mode"""
        if mode == self.mode_heel2toe:
            return self.p1.Udes_h2t
        elif mode == self.mode_toe2heel:
            return self.p1.Udes_t2h
        elif mode == self.mode_flat:
            return self.p1.Udes_flat
        else:
            print("WALKING MODE UNDEFINED! EXITING")
            return None
    
    def get_step_size_p2(self, stance_leg_idx):
        """Get step size for P2 orbit"""
        if stance_leg_idx == StanceStatus.LeftStance:
            self.p2.Xdes = self.p2.XleftDes
            self.p2.Udes = self.p2.UleftDes
        else:
            self.p2.Xdes = self.p2.XrightDes
            self.p2.Udes = self.p2.UrightDes
        
        return self.p2.Udes
    
    def get_mlip_sol3(self, t, X0, dpzmp):
        """Get 3D MLIP solution at time t"""
        Aconv = self.params.get_aconv_t(t)
        B = np.array([[0], [0], [1]])
        sol3 = expm(t * self.params.A) @ X0.reshape(-1, 1) + Aconv @ B * dpzmp
        return sol3.flatten()
    
    def get_mlip_sol2(self, t, X0, dpzmp):
        """Get 2D MLIP solution at time t"""
        sol3 = self.get_mlip_sol3(t, X0, dpzmp)
        return sol3[:2]
    


# Example usage
if __name__ == "__main__":
    # Create MLIP instance
    mlip = MLIP_3D(use_momentum=True)
    
    # Initialize with example parameters
    z0 = 0.9  # Height of COM
    TOA = 0.1  # Time on air
    TFA = 0.2  # Time foot attack
    TUA = 0.3  # Time upright approach
    orbit_period = OrbitPeriod.P1orbit
    vel = 1.0  # Desired velocity
    footlength = 0.2  # Foot length
    stepwidth = 0.3  # Step width (for P2 orbit)
    
    mlip.init(z0, TOA, TFA, TUA, orbit_period, vel, footlength, stepwidth)
    
    print("MLIP initialized successfully!")
    print(f"Total step time T: {mlip.params.T:.3f}")
    print(f"Lambda: {mlip.params.lambda_val:.3f}")
    print(f"P1 desired velocity states:")
    print(f"  Flat foot Xdes: {mlip.p1.Xdes_flat}")
    print(f"  Flat foot Udes: {mlip.p1.Udes_flat:.3f}")
    
    # Test MLIP solution
    X0 = np.array([0.1, 0.5, 0.0])  # Initial state
    t = 0.1  # Time
    dpzmp = 0.05  # Change in ZMP
    
    sol2 = mlip.get_mlip_sol2(t, X0, dpzmp)
    sol3 = mlip.get_mlip_sol3(t, X0, dpzmp)
    
    print(f"2D solution at t={t}: {sol2}")
    print(f"3D solution at t={t}: {sol3}")