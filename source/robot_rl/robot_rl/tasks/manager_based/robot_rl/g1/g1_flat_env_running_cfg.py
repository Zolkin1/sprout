from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import G1HZDObservationsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_rough_env_lip_cfg import G1RoughLipRewards
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg, G1RoughLipCurriculumCfg
from ..humanoid_env_cfg import HumanoidEventsCfg

RUNNING_EE_Q_weights_GL = [
    25.0,   250.0,      # com_x pos, vel
    300.0,   50.0,      # com_y pos, vel
    650.0,   10.0,      # com_z pos, vel
    300.0,    20.0,     # pelvis_roll pos, vel
    250.0,    10.0,     # pelvis_pitch pos, vel
    300.0,    30.0,     # pelvis_yaw pos, vel
    1500.0, 50.0,       # swing_x pos, vel
    1500.0,  50.0,      # swing_y pos, vel
    2500.0, 50.0,       # swing_z pos, vel
    30.0,    1.0,       # swing_ori_roll pos, vel
    150.0,    1.0,       # swing_ori_pitch pos, vel
    400.0,    10.0,     # swing_ori_yaw pos, vel
    1500.0, 50.0,       # stance_x pos, vel
    1500.0,  50.0,      # stance_y pos, vel
    2500.0, 50.0,       # stance_z pos, vel
    30.0,    1.0,       # stance_ori_roll pos, vel
    150.0,    1.0,       # stance_ori_pitch pos, vel
    400.0,    10.0,     # swing_ori_yaw pos, vel
    100.0,    1.0,      # waist_yaw pos, vel
    40.0,1.0, #left shoulder pitch
    40.0,1.0, #left shoulder roll
    50.0,1.0, #left shoulder yaw
    30.0,1.0, #left elbow
    40.0,1.0, #right shoulder pitch
    40.0,1.0, #right shoulder roll
    50.0,1.0, #right shoulder yaw
    30.0,1.0, #right elbow
]


RUNNING_EE_R_weights_GL = [
        0.1, 0.1, 0.1,      # CoM inputs: allow moderate effort
        0.05,0.05,0.05,     # pelvis inputs: lower torque priority
        0.05,0.05,0.05,     # swing foot linear inputs
        0.02,0.02,0.02,     # swing foot orientation inputs: small adjustments
        0.05, 0.05, 0.05,   # stance foot linear inputs
        0.02, 0.02, 0.02,   # stance foot orientation inputs: small adjustments
        0.1,0.01,0.01,
        0.01,0.01,0.01,
        0.01,0.01,0.01,
    ]

class G1RunningGaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    hzd_ref = GaitLibraryHZDCommandCfg(
        trajectory_type="end_effector",
        gait_library_path="source/robot_rl/robot_rl/assets/robots/full_library_v1",
        config_name="full",
        # Running v1
        # gait_velocity_ranges=(1.35, 1.98, 0.09),

        # Running v2
        # gait_velocity_ranges=(1.48, 2.88, 0.14),
        # use_standing=False,

        # Full
        gait_velocity_ranges=(1.1, 2.0, .1), #(0, 3.00, 0.1),
        use_standing=False, #True,

        num_outputs=27,
        Q_weights = RUNNING_EE_Q_weights_GL,
        R_weights = RUNNING_EE_R_weights_GL
    )

@configclass
class G1RunningHZDObservationCfg(G1HZDObservationsCfg):
    """Configuration for running gait library observations."""
    @configclass
    class G1RunningPolicyCfg(G1HZDObservationsCfg.PolicyCfg):
        # Add the domain flag
        domain_flag = ObsTerm(func=mdp.domain_flag, params={"command_name": "hzd_ref"}, history_length=0)

    @configclass
    class G1RunningCriticCfg(G1HZDObservationsCfg.CriticCfg):
        # Add the domain flag
        domain_flag = ObsTerm(func=mdp.domain_flag, params={"command_name": "hzd_ref"}, history_length=0)

    # observation groups
    # TODO: Try putting back
    # policy: G1RunningPolicyCfg = G1RunningPolicyCfg()
    # critic: G1RunningCriticCfg = G1RunningCriticCfg()

@configclass
class G1RunningHZDRewardCfg(G1RoughLipRewards):
    flight_contact_penalty = RewTerm(
        func=mdp.flight_contact_penalty,
        weight=-3.0,
        params={"command_name": "hzd_ref",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "weight_scalar": 0.0},
    )

@configclass
class G1RunningCurriculumCfg(G1RoughLipCurriculumCfg):
    contact_penalty_curriculum = CurrTerm(func=mdp.contact_curriculum,
                                          params={"update_interval": 20000,
                                                   "max_weight": 1.0,
                                                   "update_amnt": 0.1})

    # commanded_vel_curriculum = CurrTerm(func=mdp.cmd_vel_curriculum,
    #                                     params={"update_interval": 15000,
    #                                             "max_vel": 3.0,
    #                                             "step": 0.1})

@configclass
class G1RunningEventsCfg(HumanoidEventsCfg):
    pass
    # randomize_contact_size = EventTerm(func=mdp.randomize_rigid_body_collider_offsets,
    #                                    mode="reset",
    #                                    params={
    #                                        "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
    #                                        "rest_offset_distribution_params": (0.02, 0.04)  # TODO tune or remove
    #                                    })

@configclass
class G1RunningGaitLibraryEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 running gait library environment."""
    commands: G1RunningGaitLibraryCommandsCfg = G1RunningGaitLibraryCommandsCfg()
    observations: G1RunningHZDObservationCfg = G1RunningHZDObservationCfg()
    rewards: G1RunningHZDRewardCfg = G1RunningHZDRewardCfg()
    curriculum: G1RunningCurriculumCfg = G1RunningCurriculumCfg()
    events: G1RunningEventsCfg = G1RunningEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Set all the environment configs
        # Running v1
        # self.commands.base_velocity.ranges.lin_vel_x = (1.31, 2.03)

        # Running v2
        # self.commands.base_velocity.ranges.lin_vel_x = (1.48, 2.88)
        # self.commands.step_period.period_range = (0.75, 0.75)

        # Full v1
        self.commands.base_velocity.ranges.lin_vel_x = (1.1, 2.00)  # Note the curriculum for increasing
        self.commands.step_period.period_range = (0.689, 0.689)

        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading = (0, 0)


        self.rewards.holonomic_constraint.params["command_name"] = "hzd_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"

        self.rewards.clf_reward.params = {
            "command_name": "hzd_ref",
            "max_eta_err": 0.25,
        }
        self.rewards.clf_decreasing_condition.params = {
            "command_name": "hzd_ref",
            "alpha": 0.5,
            "eta_max": 0.25,
            "eta_dot_max": 0.3,
        }
        self.rewards.clf_decreasing_condition.weight = -1
        # self.curriculum.clf_curriculum = None
        self.curriculum.clf_curriculum.params = {
            "min_max_err": (0.1,0.1),
            "scale": (0.005,0.005), #0.001
            "update_interval": 20000
        }

        self.curriculum.terrain_levels = None

        self.events.reset_base.params["pose_range"]["yaw"] = (0, 0)

        self.rewards.dof_acc_l2 = None
        self.rewards.dof_vel_l2 = None

        self.rewards.vdot_tanh = RewTerm(
            func=mdp.vdot_tanh,
            weight=2.0,
            params={
                "command_name": "hzd_ref",
                "alpha": 1.0,
            }
        )
        self.rewards.vdot_tanh = None

        # self.rewards.clf_decreasing_condition = None

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG

        ##
        # No holonomic constraint, use the CLF on the stance foot for all domains
        ##
        # TODO: Consider removing again
        # self.rewards.holonomic_constraint_vel = None
        # self.rewards.holonomic_constraint = None

        ##
        # No Domain randomization to start
        ##
        # self.events.randomize_ground_contact_friction = None
        # self.events.add_base_mass = None
        # self.events.base_com = None
        self.events.base_external_force_torque = None
        # self.events.push_robot = None

        # Update the ground restitution range
        self.events.randomize_ground_contact_friction.params['restitution_range'] = (0.0, 0.2)

        # Update push forces
        self.events.push_robot.params['velocity_range'] = {"x": (-0.75, 0.75), "y": (-0.25, 0.25)}

        # Make the COM randomization on the torso rather than the pelvis
        self.events.base_com.params['asset_cfg'] = SceneEntityCfg("robot", body_names="waist_yaw_link")
        self.events.add_base_mass.params['asset_cfg'] = SceneEntityCfg("robot", body_names="waist_yaw_link")


        # randomize joint parameters and actuator gains
        self.events.actuator_gain = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (-10, 10.),
                    "damping_distribution_params": (-2., 2.),
                    "operation": "add",  
                    "distribution": "uniform" 
            },
        )

        self.events.joint_params = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                    "lower_limit_distribution_params": (1.0,1.0),
                    "upper_limit_distribution_params": (1.0,1.0),
                    "friction_distribution_params": (0.95, 1.05),
                    "armature_distribution_params":(0.95,1.05),
                    "operation": "scale"},
        )
        
        ##
        # Episode length
        ##
        self.episode_length_s = 20.0

@configclass
class G1RunningGaitLibraryEnvCfgPlay(G1RunningGaitLibraryEnvCfg):
    """Configuration for the G1 running gait library play environment."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (1.1, 2.0)

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.terrain.size = (3, 3)
        self.scene.terrain.border_width = 0.0
        self.scene.terrain.num_rows = 3
        self.scene.terrain.num_cols = 2
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
