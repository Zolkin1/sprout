from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg, StairGaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1FlatHZDEnvCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import G1HZDObservationsCfg, G1StairHZDObservationsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_stair_env_cfg import G1StairEnvCfg, CUSTOM_STAIR_CFG
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import G1StairObservationsCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.sensors import  RayCasterCfg, patterns
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import math
class G1GaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    hzd_ref = GaitLibraryHZDCommandCfg(
        trajectory_type="end_effector",
        gait_library_path="source/robot_rl/robot_rl/assets/robots/gait_library",
        config_name="single_support_config",
        gait_velocity_ranges=(-0.75, 0.75, 0.05)
    )


class G1GaitLibraryStairCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    hzd_ref = StairGaitLibraryHZDCommandCfg(
        trajectory_type="end_effector",
        gait_library_path="source/robot_rl/robot_rl/assets/robots/stair_gait_library",
        config_name="stair_config",
        gait_height_ranges=(0.01, 0.12, 0.01)
    )


@configclass
class G1GaitLibraryEnvCfg(G1FlatHZDEnvCfg):
    """Configuration for the G1 environment with gait library."""
    commands: G1GaitLibraryCommandsCfg = G1GaitLibraryCommandsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.observations = G1HZDObservationsCfg()

        # Configure velocity ranges for different gaits
        # self.commands.base_velocity = mdp.commands.CustomUniformVelocityCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(10.0, 10.0),
        #     rel_standing_envs=0.02,
        #     rel_heading_envs=1.0,
        #     heading_command=True,
        #     heading_control_stiffness=0.5,
        #     debug_vis=True,
        #     ref_cmd_name="hzd_ref",
        #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-0.75, 0.75), lin_vel_y=(0, 0), ang_vel_z=(-0.2, 0.2), heading=(0, 0)
        #     ),
        # )
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75, 0.75)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        self.commands.base_velocity.ranges.heading = (0, 0)

        self.commands.step_period.period_range = (0.8,0.8)

        self.observations.policy.sin_phase.params["command_name"] = "hzd_ref"
        self.observations.policy.cos_phase.params["command_name"] = "hzd_ref"
        self.observations.critic.sin_phase.params["command_name"] = "hzd_ref"
        self.observations.critic.cos_phase.params["command_name"] = "hzd_ref"
        
        # Update observation and reward command names
        self.observations.critic.foot_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.foot_ang_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj_vel.params["command_name"] = "hzd_ref"

        self.rewards.holonomic_constraint.params["command_name"] = "hzd_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"
        self.rewards.clf_reward.params["command_name"] = "hzd_ref"
        self.rewards.clf_decreasing_condition.params["command_name"] = "hzd_ref"

        # self.curriculum.clf_curriculum = None
        self.rewards.clf_reward.params["max_clf"] = 40.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 150.0
        self.rewards.clf_decreasing_condition.params["alpha"] = 1.0

        self.curriculum.clf_curriculum.params["min_val"] = 10.0
        self.curriculum.clf_curriculum.params["min_clf_val"] = 5.0
        self.curriculum.clf_curriculum.params["update_interval"] = 12000


        # self.curriculum.gait_speed = CurrTerm(func=mdp.gaits_curriculum,
        #                                      params={"vel_interval": 0.1,
        #                                              "update_interval": 24000})



@configclass
class G1_M4_GaitLibraryEnvCfg(G1GaitLibraryEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.scene.robot.spawn.usd_path = "robot_assets/g1/g1_21j_urdf_v3_min_contacts_M4.usd"

@configclass
class G1_custom_plate_GaitLibraryEnvCfg(G1GaitLibraryEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (1.14,1.14),
                "operation": "add",
            }
        )
        

        
@configclass
class G1_M4_custom_plate_GaitLibraryEnvCfg(G1GaitLibraryEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.scene.robot.spawn.usd_path = "robot_assets/g1/g1_21j_urdf_v3_min_contacts_M4.usd"
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (1.14,1.14),
                "operation": "add",
            }
        )


@configclass
class G1GaitLibraryHeightMapEnvCfg(G1GaitLibraryEnvCfg):

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.observations = G1StairObservationsCfg()
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 1.2]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"
       
        self.observations.policy.ref_traj = None
        self.observations.policy.act_traj = None
        # Update observation and reward command names
        self.observations.policy.step_duration = None
        self.observations.critic.step_duration = None
        # self.observations.policy.step_duration.params["command_name"] = "hzd_ref"
        # self.observations.critic.step_duration.params["command_name"] = "hzd_ref"

        self.observations.policy.sin_phase.params["command_name"] = "hzd_ref"
        self.observations.policy.cos_phase.params["command_name"] = "hzd_ref"
        self.observations.critic.sin_phase.params["command_name"] = "hzd_ref"
        self.observations.critic.cos_phase.params["command_name"] = "hzd_ref"

        self.observations.critic.foot_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.foot_ang_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj_vel.params["command_name"] = "hzd_ref"

        self.rewards.holonomic_constraint.params["command_name"] = "hzd_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"
        self.rewards.clf_reward.params["command_name"] = "hzd_ref"
        self.rewards.clf_decreasing_condition.params["command_name"] = "hzd_ref"

class G1GaitLibraryHeightMapEnv_PlayEnvCfg(G1GaitLibraryHeightMapEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.terrain.terrain_generator.size = (3.0, 3.0)

class G1GaitLibraryStairEnvCfg(G1StairEnvCfg):
    """Configuration for the G1 environment with gait library."""
    

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0

        self.commands = G1GaitLibraryStairCommandsCfg()
        self.observations = G1StairHZDObservationsCfg()

        self.observations.policy.ref_traj = None
        self.observations.policy.act_traj = None
        # Update observation and reward command names
        self.observations.policy.step_duration = None
        self.observations.critic.step_duration = None
        # self.observations.policy.step_duration.params["command_name"] = "hzd_ref"
        # self.observations.critic.step_duration.params["command_name"] = "hzd_ref"

        self.observations.policy.sin_phase.params["command_name"] = "hzd_ref"
        self.observations.policy.cos_phase.params["command_name"] = "hzd_ref"
        self.observations.critic.sin_phase.params["command_name"] = "hzd_ref"
        self.observations.critic.cos_phase.params["command_name"] = "hzd_ref"

        self.observations.critic.foot_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.foot_ang_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj.params["command_name"] = "hzd_ref"
        self.observations.critic.ref_traj_vel.params["command_name"] = "hzd_ref"
        self.observations.critic.act_traj_vel.params["command_name"] = "hzd_ref"

        self.rewards.holonomic_constraint_stair.params["command_name"] = "hzd_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"
        self.rewards.clf_reward.params["command_name"] = "hzd_ref"
        self.rewards.clf_decreasing_condition.params["command_name"] = "hzd_ref"

        self.rewards.clf_reward.params["max_clf"] = 200.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 200.0
        self.rewards.clf_decreasing_condition.params["alpha"] = 2.0
        
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.27, 0.27)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)

        self.scene.terrain.terrain_generator = CUSTOM_STAIR_CFG
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels)
        self.curriculum.clf_curriculum = None
     #    self.curriculum.terrain_levels = None
        flat_range = CUSTOM_STAIR_CFG.border_width *0.5
        self.events.reset_base.params = {
            "pose_range": {"x": (-flat_range, flat_range), "y": (-5.0, 5.0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


class G1GL_PlayEnvCfg(G1GaitLibraryEnvCfg):
    """Configuration for the G1 environment with gait library."""
    commands: G1GaitLibraryCommandsCfg = G1GaitLibraryCommandsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # self.scene.terrain.num_rows = 1
        # self.scene.terrain.num_cols = 1
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


class G1GL_PlayStairEnvCfg(G1GaitLibraryStairEnvCfg):
    """Configuration for the G1 environment with gait library."""
    commands: G1GaitLibraryCommandsCfg = G1GaitLibraryCommandsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator.sub_terrains["stairs"].step_height_range = (0.02, 0.02)
     #    self.scene.terrain.terrain_generator.size = (10.0, 10.0)
        flat_range = CUSTOM_STAIR_CFG.border_width *0.5
        self.events.reset_base.params = {
            "pose_range": {"x": (-flat_range, flat_range), "y": (-1.0, 1.0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.5, 0.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.scene.terrain.terrain_generator.sub_terrains["stairs"].size = (2.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.scene.terrain.terrain_generator.num_rows = 1
        
        