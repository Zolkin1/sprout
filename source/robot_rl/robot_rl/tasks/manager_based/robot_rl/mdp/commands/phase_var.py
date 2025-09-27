

class MLIPPhaseVarGlobal:
    def __init__(self, T_doublestep):
        #state transition:  FA+ -> FA- -> UA+ -> UA- -> OA+ -> OA- -> FA+
        #corresponding time_in_step:
        #                   0 -> T_fa -> T_fa -> T_ss -> T_ss -> T_ss+T_ds or 0 -> 0
        self.Tstep = T_doublestep/2.0 # total time for a step = T_fa + T_ua + T_oa
        self.T_fa = self.Tstep * 0.4  # FA: 40% of a step
        self.T_ua = self.Tstep * 0.4  # UA: 40% of a step
        self.T_oa = self.Tstep * 0.2  # OA: 20% of a step
        self.T_ss = self.T_fa + self.T_ua # total single support time
        self.T_ds = self.T_oa  # total double support time
        self.time_in_step = 0.0
        self.phase = 0.0
        self.stance_idx = 0 # use to distinguish stance leg

        self.domain = "FA" # initial state
        self.phase_fa = 0.0
        self.phase_ua = 0.0
        self.phase_oa = 0.0

    def update(self, simtime):
        self.time_in_step = simtime % (self.T_ss + self.T_ds)
        self.phase = self.time_in_step / self.T_ss
        self.phase_fa = float('nan')
        self.phase_ua = float('nan')
        self.phase_oa = float('nan')
        # Fixed logical operators (& -> and)
        if self.time_in_step < self.T_fa:
            self.domain = "FA"
            self.phase_fa = self.time_in_step / self.T_fa
        elif self.time_in_step < self.T_ss and self.time_in_step >= self.T_fa:
            self.domain = "UA"
            self.phase_ua = (self.time_in_step - self.T_fa) / self.T_ua
        elif self.time_in_step < self.T_ss + self.T_oa and self.time_in_step >= self.T_ss:
            self.domain = "OA"
            self.phase_oa = (self.time_in_step - self.T_ss) / self.T_oa

        tp = (simtime % (2 * self.Tstep)) / (2 * self.Tstep)

        # per-swing normalized phase [0,1]
        if tp < 0.5:
            self.stance_idx = 0
        else:
            self.stance_idx = 1
            
import torch            
class MLIPPhaseVarEnvBatch:
    def __init__(self, T_doublestep, num_envs, device):
        #state transition:  FA+ -> FA- -> UA+ -> UA- -> OA+ -> OA- -> FA+
        #corresponding time_in_step:
        #                   0 -> T_fa -> T_fa -> T_ss -> T_ss -> T_ss+T_ds or 0 -> 0
        self.num_envs = num_envs
        self.device = device
        
        self.Tstep = T_doublestep/2.0 # total time for a step = T_fa + T_ua + T_oa
        self.T_fa = self.Tstep * 0.4  # FA: 40% of a step
        self.T_ua = self.Tstep * 0.4  # UA: 40% of a step
        self.T_oa = self.Tstep * 0.2  # OA: 20% of a step
        self.T_ss = self.T_fa + self.T_ua # total single support time
        self.T_ds = self.T_oa  # total double support time
        
        self.time_in_step = torch.zeros((num_envs), device=device)
        self.phase = torch.zeros((num_envs), device=device) # phase in [0,1] for single support
        self.stance_idx = torch.zeros((num_envs), device=device) # use to distinguish stance leg, either 0 (left) or 1 (right)

        # Initialize domain tensor (0=FA, 1=UA, 2=OA)
        self.domain = torch.zeros((num_envs,), device=device, dtype=torch.long)
        # Initialize phase tensors
        self.phase_fa = torch.full((num_envs,), float('nan'), device=device)
        self.phase_ua = torch.full((num_envs,), float('nan'), device=device)
        self.phase_oa = torch.full((num_envs,), float('nan'), device=device)

    def update(self, simtime):
        #simtime is tensor of shape (num_envs)
        self.time_in_step = simtime % (self.T_ss + self.T_ds)
        self.phase = self.time_in_step / self.T_ss
        
        # Reset all phase tensors to NaN
        self.phase_fa.fill_(float('nan'))
        self.phase_ua.fill_(float('nan'))
        self.phase_oa.fill_(float('nan'))

        # Create masks for each domain
        fa_mask = self.time_in_step < self.T_fa
        ua_mask = (self.time_in_step >= self.T_fa) & (self.time_in_step < self.T_ss)
        oa_mask = (self.time_in_step >= self.T_ss) & (self.time_in_step < self.T_ss + self.T_oa)

        # Update phase values using masks
        self.phase_fa[fa_mask] = self.time_in_step[fa_mask] / self.T_fa
        self.phase_ua[ua_mask] = (self.time_in_step[ua_mask] - self.T_fa) / self.T_ua
        self.phase_oa[oa_mask] = (self.time_in_step[oa_mask] - self.T_ss) / self.T_oa
        
        # Update domain tensor for each environment
        self.domain[fa_mask] = 0  # FA
        self.domain[ua_mask] = 1  # UA
        self.domain[oa_mask] = 2  # OA
            
        # determine stance leg
        tp = (simtime % (2 * self.Tstep)) / (2 * self.Tstep)
        # note: hardcoded for left stance start
        self.stance_idx = torch.where(tp < 0.5, 
                                     torch.zeros_like(tp), 
                                     torch.ones_like(tp))            