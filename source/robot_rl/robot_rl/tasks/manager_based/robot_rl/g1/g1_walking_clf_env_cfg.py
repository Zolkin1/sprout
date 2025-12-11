from isaaclab.utils import configclass
# from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
# from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1FlatHZDEnvCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidCommandsCfg,
                                                                    HumanoidRewardCfg)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip
import math

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg

##
# Lyapunov Weights
##
WALKING_Q_weights = [
    25.0,   250.0,      # com_x pos, vel
    500.0,   20.0,      # com_y pos, vel
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


WALKING_R_weights = [
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

##
# Commands
##
@configclass
class G1GaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    # hzd_ref = GaitLibraryHZDCommandCfg(
    #     trajectory_type="end_effector",
    #     gait_library_path="source/robot_rl/robot_rl/assets/robots/walking_10_13",
    #     config_name="walking",
    #     gait_velocity_ranges=(0.0, 1.0, 0.1),
    #
    #     num_outputs=27,
    #     Q_weights=WALKING_Q_weights,
    #     R_weights=WALKING_R_weights,
    # )
    traj_ref = TrajectoryCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        # manager_type = "trajectory",
        # path="source/robot_rl/robot_rl/assets/robots/test_walking_trajectories",
        manager_type="library",
        path="source/robot_rl/robot_rl/assets/robots/test_walking_library",

        conditioner_generator_name = "base_velocity",
        num_outputs = 27,
        Q_weights = WALKING_Q_weights,
        R_weights = WALKING_R_weights,
    )

##
# Curriculums
##
@configclass
class G1WalkingCLFCurriculumCfg:
    """Curriculum terms for the MDP."""

    clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 1000, "min_val": 20.0})

##
# Observations
##
@configclass
class G1TrajOptObservationsCfg():
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},
                                    scale=(2.0, 2.0, 2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "traj_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "traj_ref"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        # root_quat = ObsTerm(func=mdp.root_quat)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},
                                    scale=(2.0, 2.0, 2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "traj_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "traj_ref"})

        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        # TODO: Modify this to be more generic, I want the critic to get the values for all the end effector frames being tracked
        # TODO: Fix (above) and put back
        # foot_vel = ObsTerm(func=mdp.foot_vel, params={"command_name": "traj_ref"}, scale=1.0)
        # foot_ang_vel = ObsTerm(func=mdp.foot_ang_vel, params={"command_name": "traj_ref"}, scale=1.0)

        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

        height_scan = None  # Removed - not supported yet


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Rewards
##
@configclass
class G1WalkingCLFRewards(HumanoidRewardCfg):
    """Rewards specific to LIP Model"""

    holonomic_constraint = RewTerm(
        func=mdp.holonomic_constraint,
        weight=4.0,
        params={
            "command_name": "traj_ref",
            "z_offset": 0.036,
        }
    )

    holonomic_constraint_vel = RewTerm(
        func=mdp.holonomic_constraint_vel,
        weight=2.0,
        params={
            "command_name": "traj_ref",
        }
    )


    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "traj_ref",
            "max_eta_err": 0.25,
        }
    )

    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={
            "command_name": "traj_ref",
            "alpha": 0.5,
            "eta_max": 0.2,
            "eta_dot_max":0.3,
        }
    )

@configclass
class G1WalkingCLFEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 environment with gait library."""
    commands: G1GaitLibraryCommandsCfg = G1GaitLibraryCommandsCfg()
    observations: G1TrajOptObservationsCfg = G1TrajOptObservationsCfg()
    rewards: G1WalkingCLFRewards = G1WalkingCLFRewards()
    curriculum: G1WalkingCLFCurriculumCfg = G1WalkingCLFCurriculumCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        ##
        # Commands
        ##
        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0) # TODO: put back: (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0,0)

        self.commands.gait_period.gait_period_range = (0.71, 0.71)    # TODO: Remove?

        ##
        # Randomization
        ##
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1, 1),
            "y": (-1, 1),
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.4, 0.4),
        }
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.events.base_external_force_torque = None

        ##
        # Rewards
        ##
        self.rewards.holonomic_constraint.params["command_name"] = "traj_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "traj_ref"

        self.rewards.clf_reward.params = {
            "command_name": "traj_ref",
            "max_eta_err": 0.25,
        }
        self.rewards.clf_decreasing_condition.params = {
            "command_name": "traj_ref",
            "alpha": 0.5,
            "eta_max": 0.25,
            "eta_dot_max": 0.3,
        }
        self.rewards.clf_decreasing_condition.weight = -1

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"

        ##
        # Curriculum
        ##
        self.curriculum.clf_curriculum = None
        self.curriculum.terrain_levels = None

        self.events.reset_base.params["pose_range"]["yaw"] = (0,0)

        # self.curriculum.clf_curriculum.params = {
        #     "min_max_err": (0.1,0.1),
        #     "scale": (0.001,0.001),
        #     "update_interval": 20000
        # }

        # self.rewards.vdot_tanh = RewTerm(
        #     func=mdp.vdot_tanh,
        #     weight= 2.0,
        #     params={
        #         "command_name": "hzd_ref",
        #         "alpha": 1.0,
        #     }
        # )

        # self.rewards.clf_decreasing_condition = None

        ##
        # Terrain
        ##
        # self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.clf_curriculum = None

@configclass
class G1WalkingCLFECEnvCfg(G1WalkingCLFEnvCfg):
    """Configuration for the G1 environment with gait library."""
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        #both front and back 1.14
        #just front: 0.616
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616,0.616),
                "operation": "add",
            }
        )

@configclass
class G1WalkingCLFEnvCfg_PLAY(G1WalkingCLFEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.terrain.size = (3,3)
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
        
        
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0,0)

        # self.events.reset_base.params["pose_range"]["yaw"] = (-3.14,3.14)
        # self.events.reset_base.params["pose_range"]["x"] = (-3,3)
        # self.events.reset_base.params["pose_range"]["y"] = (-3,3)