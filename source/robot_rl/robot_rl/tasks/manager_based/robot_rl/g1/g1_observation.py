from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg

@configclass
class G1RoughLipObservationsCfg():
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=(2.0,2.0,2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),scale=0.05)
        
        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"command_name": "step_period"})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"command_name": "step_period"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        # root_quat = ObsTerm(func=mdp.root_quat)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=(2.0,2.0,2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        sin_phase = ObsTerm(func=mdp.sin_phase, params={"command_name": "step_period"})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"command_name": "step_period"})

        foot_vel = ObsTerm(func=mdp.foot_vel, params={"command_name": "hlip_ref"},scale=1.0)
        foot_ang_vel = ObsTerm(func=mdp.foot_ang_vel, params={"command_name": "hlip_ref"},scale=1.0)
        ref_traj = ObsTerm(
            func=mdp.ref_traj,
            params={"command_name": "hlip_ref"},
            scale=tuple([5.0] * 12 + [0.1] * 9)
        )
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "hlip_ref"},scale=1.0)
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "hlip_ref"},clip=(-20.0,20.0,),scale=1.0)
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "hlip_ref"},clip=(-20.0,20.0,),scale=1.0)
        height_scan = None      # Removed - not supported yet


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()



@configclass
class G1HZDObservationsCfg():
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=(2.0,2.0,2.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),scale=0.05)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "hlip_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "hlip_ref"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
  
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        # root_quat = ObsTerm(func=mdp.root_quat)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=(2.0,2.0,2.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "hlip_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "hlip_ref"})

        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,scale=1.0)
        foot_vel = ObsTerm(func=mdp.foot_vel, params={"command_name": "hlip_ref"},scale=1.0)
        foot_ang_vel = ObsTerm(func=mdp.foot_ang_vel, params={"command_name": "hlip_ref"},scale=1.0)
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "hlip_ref"},scale=tuple([5.0] * 12 + [0.1] * 9))
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "hlip_ref"},scale=tuple([5.0] * 12 + [0.1] * 9))
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "hlip_ref"},clip=(-20.0,20.0,),scale=tuple([1.0] * 12 + [0.1] * 9))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "hlip_ref"},clip=(-20.0,20.0,),scale=tuple([1.0] * 12 + [0.1] * 9))
       
        # base_z = ObsTerm(func=mdp.base_z)
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


