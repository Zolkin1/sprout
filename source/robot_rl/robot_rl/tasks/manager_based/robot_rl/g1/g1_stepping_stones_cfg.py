from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from robot_rl.tasks.manager_based.robot_rl import mdp

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (
    HumanoidEnvCfg,
    HumanoidRewardCfg,
    HumanoidTerminationsCfg
)

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip


#
import isaaclab.sim as sim_utils
from dataclasses import MISSING
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


from robot_rl.tasks.manager_based.robot_rl.mdp.commands.stones_output_cmd_cfg import StonesOutputCommandCfg

from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.stones_terrain_importer import StonesTerrainImporter
from robot_rl.tasks.manager_based.robot_rl.terrains.stones_terrain_generator import StonesTerrainGenerator
from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones_cfg import (
    LongStonesFlatTerrainCfg, 
    LongStonesTerrainCfg, 
    StairsTerrainCfg, 
    TiltedStonesTerrainCfg,
    FlatGroundTestingCfg
)
from robot_rl.tasks.manager_based.robot_rl.constants import STONES, TEST_FLAT
from isaaclab.sensors import RayCasterCfg, patterns        
        
        
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import (
    ROUGH_SLOPED_FOR_FLAT_HZD_CFG,
)
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation_stepping_stones import (
    G1SteppingStonesObservationsTeacherCfg,
    G1SteppingStonesObservationsDistillationCfg,
    G1SteppingStonesObservationsFinetuneCfg,
    G1SteppingStonesObservationsFinetuneNewCfg
)

        
@configclass
class G1RoughMlipCommandsCfg:
    """Commands for the G1 Flat environment."""

    hlip_ref = StonesOutputCommandCfg()
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.6, 0.6), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )
    


@configclass
class CurriculumMlipCfg:
    """Curriculum terms for the MDP."""


    # todo: stones curriculum
    if TEST_FLAT==False:
        stones_curriculum = CurrTerm(func=mdp.stones_sagittal_terrain_levels_termination, 
                                     params={
                                            "success_term_name": "finished_long_stones",
                                             },)
        modify_reference_cfg = CurrTerm(func=mdp.modify_reference_cfg, 
                                        params={
                                                "term_name": "hlip_ref",
                                                "steps": 1000 * 24
                                               },  )



class G1RoughMlipRewards(HumanoidRewardCfg):
    """Rewards specific to LIP Model"""

    holonomic_constraint = RewTerm(
        func=mdp.holonomic_constraint,
        weight=4.0,
        params={
            "command_name": "hlip_ref",
            "z_offset": 0.036,
        },
    )

    holonomic_constraint_vel = RewTerm(
        func=mdp.holonomic_constraint_vel,
        weight=2.0,
        params={
            "command_name": "hlip_ref",
        },
    )

    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "hlip_ref",
            "max_eta_err": 0.3,
        },
    )

    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={
            "command_name": "hlip_ref",
            "alpha": 1.0,
            "eta_max": 0.2,
            "eta_dot_max": 0.3,
        },
    )


class G1SteppingStonesTerminationsCfg(HumanoidTerminationsCfg):
    """Termination terms for the G1 Stepping Stones environment."""
    if TEST_FLAT==False:
        finished_long_stones = DoneTerm(
            func=mdp.finished_long_stones,
            params={
                "output_command_name": "hlip_ref",
            },
            time_out=True)
        long_stones_deviation = DoneTerm(
            func=mdp.long_stones_deviation,
            params={
                "output_command_name": "hlip_ref",
            },
            time_out=False)
        z_com_too_low = DoneTerm(
            func=mdp.com_z_too_low,
            params={
                "output_command_name": "hlip_ref",
            },
            time_out=False)
        stationary_x_with_nonzero_vel_command = DoneTerm(
            func=mdp.stationary_x_with_nonzero_vel_command,
            params={
                "velocity_threshold": 0.1,
                "duration_threshold": 4.0,
            },
            time_out=False)


@configclass
class G1SteppingStonesEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Flat environment."""

    rewards: G1RoughMlipRewards = G1RoughMlipRewards()


    observations: G1SteppingStonesObservationsTeacherCfg = G1SteppingStonesObservationsTeacherCfg()
    commands: G1RoughMlipCommandsCfg = G1RoughMlipCommandsCfg()
    curriculum: CurriculumMlipCfg = CurriculumMlipCfg()
    terminations: G1SteppingStonesTerminationsCfg = G1SteppingStonesTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if TEST_FLAT==False:
            STONES_CFG = TerrainGeneratorCfg(
                class_type=StonesTerrainGenerator,
                size=(STONES.terrain_size_x, STONES.terrain_size_y),
                curriculum=True,
                border_width=1.0,
                border_height=0.0,
                num_rows=10,  # difficulty levels
                num_cols=100,  # corresponds to terrain types, num of sub terrain envs = proportion * num_cols
                horizontal_scale=0.1,
                vertical_scale=0.0005,
                slope_threshold=0.75,
                use_cache=False,
                difficulty_range=(0.0, 1.0),
                sub_terrains={
                    "upstairs": StairsTerrainCfg(proportion=0.1, is_upstairs=True),
                    "downstairs": StairsTerrainCfg(proportion=0.1, is_upstairs=False),
                    "flat_stones": LongStonesFlatTerrainCfg(proportion=0.4),
                    "stones": LongStonesTerrainCfg(proportion=0.4),
                    # "flat_ground": FlatGroundTestingCfg(proportion=1.0)
                    # "tilted_stones": TiltedStonesTerrainCfg(proportion=0.15),
                },
            )
            self.scene.terrain = TerrainImporterCfg(
                class_type=StonesTerrainImporter,
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=STONES_CFG,
                max_init_terrain_level=0,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                    project_uvw=True,
                    texture_scale=(0.25, 0.25),
                ),
                debug_vis=False,
            )
            self.terminations.time_out = None
        else:
            self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG

        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground/terrain_stones"],
        )

        ##
        # Randomization
        ##
        self.events.push_robot.params["velocity_range"] = {
            "x": (-.5, 0.5), 
            "y": (-0.2, 0.2), 
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.4, 0.4),
        }
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.4, 1.0)
        self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.4, 1.0)

        self.events.base_external_force_torque = None

        #remove all randomization for debugging
        from robot_rl.tasks.manager_based.robot_rl.constants import IS_DEBUG
        if IS_DEBUG: 
            self.events.reset_robot_joints = None
            self.events.reset_base = None
            self.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)
            self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
            self.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"

        ##
        # Rewards
        ##
        self.rewards.feet_air_time = None
        self.rewards.phase_contact = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.feet_clearance = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.termination_penalty = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.joint_deviation_hip = None
        self.rewards.contact_no_vel = None
        self.rewards.alive = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        # torque, acc, vel, action rate regularization
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_pos_limits.weight = -1.0
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.dof_vel_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.dof_acc_l2 = None
        self.rewards.dof_vel_l2 = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None
        self.rewards.height_torso = None

        self.curriculum.modify_reference_cfg = None
        self.commands.hlip_ref.use_stance_foot_pos_as_ref = True



class G1_custom_stepping_stones(G1SteppingStonesEnvCfg):
    observations: G1SteppingStonesObservationsTeacherCfg = G1SteppingStonesObservationsTeacherCfg()
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        # both front and back 1.14
        # just front: 0.616
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616, 0.616),
                "operation": "add",
            },
        )
class G1_custom_stepping_stones_distillation(G1SteppingStonesEnvCfg):
    observations: G1SteppingStonesObservationsDistillationCfg = G1SteppingStonesObservationsDistillationCfg()
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.observations = G1SteppingStonesObservationsDistillationCfg()
        self.scene.terrain.max_init_terrain_level = 10
        self.curriculum.stones_curriculum = None
        self.curriculum.modify_reference_cfg = None
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616, 0.616),
                "operation": "add",
            },
        )
        self.commands.hlip_ref.use_stance_foot_pos_as_ref = True
        
class G1HardwareNoHeightMapTestingCfg(G1SteppingStonesEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.curriculum.stones_curriculum = None
        self.curriculum.modify_reference_cfg = None
        from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones_cfg import FlatGroundTestingCfg
        self.scene.terrain.terrain_generator.sub_terrains = {
            "ground": FlatGroundTestingCfg(proportion=1.0),
        }
        self.scene.terrain.max_init_terrain_level = 10
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
class G1HardwareNoHeightMapTestingDistillCfg(G1SteppingStonesEnvCfg):
    observations: G1SteppingStonesObservationsDistillationCfg = G1SteppingStonesObservationsDistillationCfg()
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.curriculum.stones_curriculum = None
        self.curriculum.modify_reference_cfg = None
        from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones_cfg import FlatGroundTestingCfg
        self.scene.terrain.terrain_generator.sub_terrains = {
            "ground": FlatGroundTestingCfg(proportion=1.0),
        }
        self.observations = G1SteppingStonesObservationsDistillationCfg()
        self.scene.terrain.max_init_terrain_level = 10
        self.observations.policy.height_scan = None
        self.observations.teacher.height_scan = None
class G1HardwareNoHeightMapTestingFinetuneCfg(G1SteppingStonesEnvCfg):
    observations: G1SteppingStonesObservationsFinetuneCfg = G1SteppingStonesObservationsFinetuneCfg()
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.curriculum.stones_curriculum = None
        self.curriculum.modify_reference_cfg = None
        from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones_cfg import FlatGroundTestingCfg
        self.scene.terrain.terrain_generator.sub_terrains = {
            "ground": FlatGroundTestingCfg(proportion=1.0),
        }
        self.observations = G1SteppingStonesObservationsFinetuneCfg()
        self.scene.terrain.max_init_terrain_level = 10
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

class G1_custom_stepping_stones_finetune(G1SteppingStonesEnvCfg):
    observations: G1SteppingStonesObservationsFinetuneCfg = G1SteppingStonesObservationsFinetuneCfg()
    # observations: G1SteppingStonesObservationsFinetuneNewCfg = G1SteppingStonesObservationsFinetuneNewCfg()
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        self.observations = G1SteppingStonesObservationsFinetuneCfg()
        # self.observations = G1SteppingStonesObservationsFinetuneNewCfg()
        self.scene.terrain.max_init_terrain_level = 3
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616, 0.616),
                "operation": "add",
            },
        )
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.curriculum.modify_reference_cfg = None
        self.commands.hlip_ref.use_stance_foot_pos_as_ref = True

class G1SteppingStonesEnvCfg_PLAY(G1SteppingStonesEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        self.events.push_robot.interval_range_s = (2.0, 2.0)
        # self.events.reset_base.params["pose_range"] = {
        #     "x": (-0.3, 0.0),
        #     "y": (-0.1, 0.1),
        #     "yaw": (-0.1, 0.1),
        # }
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.commands.hlip_ref.use_stance_foot_pos_as_ref = True
        
class G1_custom_stepping_stones_distillation_PLAY(G1_custom_stepping_stones_distillation):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.events.push_robot.interval_range_s = (5.0, 5.0)
        # self.events.reset_base.params["pose_range"] = {
        #     "x": (-0.3, 0.0),
        #     "y": (-0.1, 0.1),
        #     "yaw": (-0.1, 0.1),
        # }
        
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)

class G1_custom_stepping_stones_finetune_PLAY(G1_custom_stepping_stones_finetune):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.events.push_robot.interval_range_s = (3.0, 3.0)
        # self.events.reset_base.params["pose_range"] = {
        #     "x": (-0.3, 0.0),
        #     "y": (-0.1, 0.1),
        #     "yaw": (-0.1, 0.1),
        # }
        
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)     
 