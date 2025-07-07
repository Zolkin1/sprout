import math
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import  RayCasterCfg, patterns
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import TerminationsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1SceneCfg
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg, G1RoughLipRewards
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import STAIR_CFG, UP_STAIR_CFG, CUSTOM_STAIR_CFG


from robot_rl.tasks.manager_based.robot_rl import mdp

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.stair_cfg import StairHLIPCommandCfg
##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import G1StairObservationsCfg
#
# from robot_rl.tasks.manager_based.robot_rl.mdp.commands.cmd_cfg import HZDStairCommandCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_rough_env_lip_cfg import CurriculumCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_stair_cfg import HZDStairEECommandCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import EndEffectorTrajectoryHZDCommandCfg
@configclass
class G1StairCommandsCfg(HumanoidCommandsCfg):
    """Commands for the G1 Flat environment."""   
    hlip_ref = StairHLIPCommandCfg()
    def __post_init__(self):
        super().__post_init__()
        self.step_period.period_range = (1.0,1.0)


@configclass
class G1StairRewardsCfg(G1RoughLipRewards):
    """Rewards for the G1 Flat environment."""
    holonomic_constraint = None
    holonomic_constraint_stair: RewTerm = RewTerm(
        func=mdp.holonomic_constraint_stair,
        params={"command_name": "hlip_ref"},
        weight=4.0,
    )


@configclass
class G1StairsTerminationCfg(TerminationsCfg):
    """Events for the G1 Flat environment."""
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 1.4, 
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["pelvis_link"])
        },
    )

@configclass
class G1StairEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    scene: G1SceneCfg = G1SceneCfg()
    commands: G1StairCommandsCfg = G1StairCommandsCfg()
    rewards: G1StairRewardsCfg = G1StairRewardsCfg()
    terminations: G1StairsTerminationCfg = G1StairsTerminationCfg()
    observations: G1StairObservationsCfg = G1StairObservationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        self.curriculum.clf_curriculum = None
        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # No height scanner for now
     
        self.scene.terrain.terrain_type = "generator"
     
        self.scene.terrain.terrain_generator = STAIR_CFG
        # self.scene.terrain.terrain_generator.max_init_terrain_level = 2.0
        # self.curriculum.terrain_levels = None
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels)
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 1.2]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"


      
        ##
        # Randomization
        ##
        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {"x": (-1, 1.0), "y": (-1, 1.0), "roll": (-0.4, 0.4),
                                                           "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            
            "pose_range": {"x": (0.0,0.0), "y": (0.0,0.0), "yaw": (0,0)},
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
        # Commands
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (0.3,0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5,0.5)
        self.commands.base_velocity.ranges.heading= (0.0,0.0)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        self.terminations.no_progress = None
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        ##
        # Rewards
        ##
        self.rewards.feet_air_time = None
        self.rewards.phase_contact = None
        self.rewards.lin_vel_z_l2 = None
        # self.rewards.height_torso = None
        self.rewards.feet_clearance = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.termination_penalty = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.joint_deviation_hip = None
        self.rewards.contact_no_vel = None
        # self.rewards.alive = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        self.rewards.clf_reward.params["max_clf"] = 100.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 50.0

        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_vel_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.001

        
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None

        self.rewards.height_torso = None



HZD_Stair_Q_weights = [
    300.0,   200.0,    # com_x pos, vel
    600.0,   50.0,   # com_y pos, vel
    600.0,  20.0,  # com_z pos, vel
    600.0,    20.0,    # pelvis_roll pos, vel
    450.0,    10.0,    # pelvis_pitch pos, vel
    500.0,    30.0,    # pelvis_yaw pos, vel
    2500.0, 125.0,  # swing_x pos, vel
    1700.0,  125.0,  # swing_y pos, vel
    8500.0, 120.0,   # swing_z pos, vel
    200.0,    1.0,    # swing_ori_roll pos, vel
    400.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    300.0,    10.0,    # waist_yaw pos, vel
    400.0,1.0, #swing hand palm pos x
    50.0,10.0, #swing hand palm pos y
    50.0,1.0, #swing hand palm pos z
    50.0,1.0, #swing hand palm yaw
    400.0,1.0, #stance hand palm pos x
    50.0,10.0, #stance hand palm pos y
    50.0,1.0, #stance hand palm pos z
    50.0,1.0, #stance hand palm yaw
]


HZD_Stair_R_weights = [
        0.1, 0.1, 0.1,    # CoM inputs: allow moderate effort
        0.05,0.05,0.05,   # pelvis inputs: lower torque priority
        0.05,0.05,0.05,   # swing foot linear inputs
        0.02,0.02,0.02,    # swing foot orientation inputs: small adjustments
        0.1,0.01,0.01,
        0.01,0.01,0.01,
        0.01,0.01,0.01,
    ]

@configclass
class G1HZD_StairCommandsCfg(HumanoidCommandsCfg):
    """Commands for the G1 Flat environment."""   
    # hzd_ref = HZDStairCommandCfg()
    # hzd_ref = HZDStairEECommandCfg()
    hzd_ref = EndEffectorTrajectoryHZDCommandCfg()
    hzd_ref.Q_weights = HZD_Stair_Q_weights
    hzd_ref.R_weights = HZD_Stair_R_weights
    #TODO cha
@configclass
class G1HZDStairEnvCfg(G1StairEnvCfg):
    commands: G1HZD_StairCommandsCfg = G1HZD_StairCommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):

        super().__post_init__()

        self.observations.policy.step_duration = None
        self.observations.critic.step_duration = None
        # self.observations.policy.step_duration.params["command_name"] = "hzd_ref"
        # self.observations.critic.step_duration.params["command_name"] = "hzd_ref"

        self.observations.policy.sin_phase = ObsTerm(func=mdp.stair_sin_phase, params={"command_name": "hzd_ref"})
        self.observations.policy.cos_phase = ObsTerm(func=mdp.stair_cos_phase, params={"command_name": "hzd_ref"})

        self.observations.critic.sin_phase = ObsTerm(func=mdp.stair_sin_phase, params={"command_name": "hzd_ref"})
        self.observations.critic.cos_phase = ObsTerm(func=mdp.stair_cos_phase, params={"command_name": "hzd_ref"})

        self.observations.critic.step_duration = None
        self.observations.critic.foot_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.foot_ang_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj_vel.params["command_name"] = "hzd_ref"

        # change the reward command name to hzd_ref
        self.rewards.holonomic_constraint_stair.params["command_name"] = "hzd_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"
        self.rewards.clf_reward.params["command_name"] = "hzd_ref"
        self.rewards.clf_decreasing_condition.params["command_name"] = "hzd_ref"
        

        self.rewards.clf_reward.params["max_clf"] = 200.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 200.0
        self.rewards.clf_decreasing_condition.params["alpha"] = 2.0
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.4,0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0,0)
        self.commands.base_velocity.ranges.ang_vel_z = (0,0)

        self.scene.terrain.terrain_generator = CUSTOM_STAIR_CFG
        # self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels)
        self.curriculum.terrain_levels = None

        self.events.reset_base.params = {
            
            "pose_range": {"x": (0.0,1.0), "y": (-2.0,2.0), "yaw": (0,0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # self.episode_length_s = 7
        # self.events.push_robot.interval_range_s = (3,5)

        # self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
        self.events.base_com = None

        # self.events.reset_base = None
        # self.events.reset_robot_joints = None

        # self.events.reset_init_config = EventTerm(
        #     func=mdp.reset_init_config,
        #     params={"command_name": "hzd_ref"},
        #     mode="reset",
        # )

        self.commands.hzd_ref.yaml_path = "source/robot_rl/robot_rl/assets/robots/stair_config_solution_ee.yaml"
        # clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 1500, "min_val": 20.0})
        # self.curriculum.clf_curriculum = clf_curriculum


@configclass
class G1HZDStairEnvPlay(G1HZDStairEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        # self.events.reset_base.params["pose_range"] = {"x": (0,0), "y": (0,0), "yaw": (0,0)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.4,0.4)
        # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05,0.05)
        # self.scene.terrain.terrain_generator.num_rows = 1
        # self.scene.terrain.terrain_generator.num_cols = 2

@configclass
class G1HeightScanFlatEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    commands: G1StairCommandsCfg = G1StairCommandsCfg()
    rewards: G1StairRewardsCfg = G1StairRewardsCfg()
    terminations: G1StairsTerminationCfg = G1StairsTerminationCfg()
    observations: G1StairObservationsCfg = G1StairObservationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        self.curriculum.clf_curriculum = None
        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 6
        # No height scanner for now
     
        self.scene.terrain.terrain_type = "generator"
        STAIR_CFG.sub_terrains["pyramid_stairs"].step_height_range = (0.0,0.0)
        del STAIR_CFG.sub_terrains["pyramid_stairs_inv"]
        
        self.scene.terrain.terrain_generator = STAIR_CFG

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 1.2]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"


      
        ##
        # Randomization
        ##
        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {"x": (-1, 1.0), "y": (-1, 1.0), "roll": (-0.4, 0.4),
                                                           "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
        # self.events.push_robot.params["velocity_range"] = {"x": (-0, 0), "y": (-0, 0), "roll": (-0.0, 0.0),
        #                                                    "pitch": (-0., 0.), "yaw": (-0.0, 0.0)}
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
        # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.reset_base.params = {
            
            "pose_range": {"x": (0.0,0.0), "y": (0.0,0.0), "yaw": (-3.14,3.14)},
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
        # Commands
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (0.3,0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4,0.4)
        # self.commands.base_velocity.ranges.heading= (0.0,0.0)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        self.terminations.no_progress = None
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        ##
        # Rewards
        ##
        self.rewards.feet_air_time = None
        self.rewards.phase_contact = None
        self.rewards.lin_vel_z_l2 = None
        # self.rewards.height_torso = None
        self.rewards.feet_clearance = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.termination_penalty = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.joint_deviation_hip = None
        self.rewards.contact_no_vel = None
        # self.rewards.alive = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        self.rewards.clf_reward.params["max_clf"] = 50.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 50.0
 
        # torque, acc, vel, action rate regularization

        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_vel_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.001

        
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None

        self.rewards.height_torso = None

class G1HeightScanPlay_EnvCfg(G1HeightScanFlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.events.reset_base.params["pose_range"] = {"x": (0,0), "y": (0,0), "yaw": (0,0)} #(-3.14, 3.14)},
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.commands.hlip_ref.debug_vis = False
        self.events.push_robot = None

        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 2

        self.commands.base_velocity.ranges.lin_vel_x = (0.6,0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        self.terminations.no_progress = None
        self.scene.height_scanner.debug_vis = True


class G1StairPlay_EnvCfg(G1StairEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.events.reset_base.params["pose_range"] = {"x": (0,0), "y": (0,0), "yaw": (0,0)} #(-3.14, 3.14)},
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.commands.hlip_ref.debug_vis = False
        self.events.push_robot = None

        # STAIR_CFG.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.1,0.1)
        # STAIR_CFG.sub_terrains["pyramid_stairs"].step_height_range = (0.1,0.1)
        # self.scene.terrain.terrain_generator = STAIR_CFG
        stair_range = (0.05,0.05)
        STAIR_CFG.sub_terrains["pyramid_stairs"].step_height_range = stair_range
        STAIR_CFG.sub_terrains["pyramid_stairs_inv"].step_height_range = stair_range
        STAIR_CFG.sub_terrains["stairs_inv_w_hole"].step_height_range = stair_range
        STAIR_CFG.sub_terrains["stairs_w_hole"].step_height_range = stair_range
        # STAIR_CFG.sub_terrains["pyramid_stairs"].holes = True
        # STAIR_CFG.sub_terrains["pyramid_stairs_inv"].holes = True
        # STAIR_CFG.sub_terrains["pyramid_stairs"].step_height_range = (0.025,0.025)
        # del STAIR_CFG.sub_terrains["pyramid_stairs_inv"]
        # del STAIR_CFG.sub_terrains["pyramid_stairs"]
        # STAIR_CFG.sub_terrains["stairs_w_hole"].step_height_range = (0.1,0.1)
        # STAIR_CFG.sub_terrains["stairs_inv_w_hole"].step_height_range = (0.1,0.1)
        # del STAIR_CFG.sub_terrains["flat_stairs_inv"]
        # del STAIR_CFG.sub_terrains["flat_stairs"]
        self.scene.terrain.terrain_generator = STAIR_CFG
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 2


        self.commands.base_velocity.ranges.lin_vel_x = (0.6,0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.heading= (0.0,0.0)

        # self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        # self.terminations.no_progress = None