from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

from robot_rl.tasks.manager_based.robot_rl import mdp

@configclass
class TeacherCfg(ObsGroup):
    """Observations for teacher group."""
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 2.0)
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
    actions = ObsTerm(func=mdp.last_action)
    sin_cos_phase = ObsTerm(func=mdp.sincos_phase_batched, params={"command_name": "hlip_ref"})
    
    stones_rel_pos = ObsTerm(func=mdp.stones_position, params={"command_name": "hlip_ref"}, scale=1.0)

    height_scan = ObsTerm(
        func=mdp.height_scan_isaaclab,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
@configclass
class StudentCfg(ObsGroup):
    """Observations for student group."""
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 2.0)
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
    actions = ObsTerm(func=mdp.last_action, history_length=1)

    height_scan = ObsTerm(
        func=mdp.height_scan_isaaclab,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
class StudentDistillationCfg(StudentCfg):
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
@configclass
class CriticCfg(ObsGroup):
    """Observations for critic group."""
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 2.0)
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action)
    sin_cos_phase = ObsTerm(func=mdp.sincos_phase_batched, params={"command_name": "hlip_ref"})
    
    contact_state = ObsTerm(
        func=mdp.contact_state,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_vel = ObsTerm(func=mdp.foot_vel, params={"command_name": "hlip_ref"}, scale=1.0)
    foot_ang_vel = ObsTerm(func=mdp.foot_ang_vel, params={"command_name": "hlip_ref"}, scale=1.0)
    ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "hlip_ref"})
    act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "hlip_ref"}, scale=1.0)
    ref_traj_vel = ObsTerm(
        func=mdp.ref_traj_vel,
        params={"command_name": "hlip_ref"},
        clip=(
            -20.0,
            20.0,
        ),
        scale=1.0,
    )
    act_traj_vel = ObsTerm(
        func=mdp.act_traj_vel,
        params={"command_name": "hlip_ref"},
        clip=(
            -20.0,
            20.0,
        ),
        scale=1.0,
    )
    stones_rel_pos = ObsTerm(func=mdp.stones_position, params={"command_name": "hlip_ref"}, scale=1.0)
    height_scan = ObsTerm(
        func=mdp.height_scan_isaaclab,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
@configclass
class G1SteppingStonesObservationsTeacherCfg:
    """Observation specifications for the G1 Flat environment."""
    # observation groups
    policy: TeacherCfg = TeacherCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class G1SteppingStonesObservationsDistillationCfg:
    """Observation specifications for the G1 Flat environment."""
    # observation groups
    teacher: TeacherCfg = TeacherCfg()
    policy: StudentDistillationCfg = StudentDistillationCfg()

@configclass
class G1SteppingStonesObservationsFinetuneCfg:    
    """Observation specifications for the G1 Flat environment."""
    # observation groups
    policy: StudentCfg = StudentCfg()
    critic: CriticCfg = CriticCfg()



@configclass
class StudentDRHeightMapCfg(ObsGroup):
    """Observations for student group."""
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 2.0)
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
    actions = ObsTerm(func=mdp.last_action, history_length=1)

    #todo: add domain randomization to height scan one by one
    #foothold smoothing is bad!
    height_scan = ObsTerm(
        func= mdp.height_scan_full_domain_rand,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,  # Can customize offset here
                "cfg": mdp.HeightMapDomainRandCfg()
                },
        noise=NoiseModelWithAdditiveBiasCfg(
        bias_noise_cfg=Unoise(n_min=-0.04, n_max=0.04, operation="add"),
        noise_cfg=Unoise(n_min=-0.06, n_max=0.06, operation="add"),
        sample_bias_per_component=False,
        ),
        clip=(-1.0, 1.0),
    )
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        
@configclass
class G1SteppingStonesObservationsFinetuneNewCfg:    
    """Observation specifications for the G1 Flat environment."""
    # observation groups
    policy: StudentDRHeightMapCfg = StudentDRHeightMapCfg()
    critic: CriticCfg = CriticCfg()