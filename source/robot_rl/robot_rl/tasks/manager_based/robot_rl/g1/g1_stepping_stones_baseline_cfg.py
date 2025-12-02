from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from robot_rl.tasks.manager_based.robot_rl import mdp

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (
    HumanoidCommandsCfg,
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
        
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import math

@configclass
class G1SteppingStonesBaselineCommandsCfg:
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
class G1SteppingStonesBaselineObservationsCfg:
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
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
    
    policy: PolicyCfg = PolicyCfg()
    
    
@configclass
class G1SteppingStonesBaselineCurriculumCfg:
    """Curriculum terms for the MDP."""
    stones_curriculum = CurrTerm(func=mdp.stones_sagittal_terrain_levels_termination, 
                                     params={
                                            "success_term_name": "finished_long_stones",
                                             },)
@configclass
class G1SteppingStonesBaselineTerminationsCfg(HumanoidTerminationsCfg):
    """Termination terms for the G1 Stepping Stones environment."""
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
class G1SteppingStonesBaselineRewardCfg:
    """Reward terms for the G1 Stepping Stones environment; used unitree baseline weights."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    
    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    # ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

@configclass
class G1SteppingStonesBaselineEnvCfg(HumanoidEnvCfg):
    commands: G1SteppingStonesBaselineCommandsCfg = G1SteppingStonesBaselineCommandsCfg()
    observations: G1SteppingStonesBaselineObservationsCfg = G1SteppingStonesBaselineObservationsCfg()
    rewards: G1SteppingStonesBaselineRewardCfg = G1SteppingStonesBaselineRewardCfg()
    curriculum: G1SteppingStonesBaselineCurriculumCfg = G1SteppingStonesBaselineCurriculumCfg()
    terminations: G1SteppingStonesBaselineTerminationsCfg = G1SteppingStonesBaselineTerminationsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
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
        
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616, 0.616),
                "operation": "add",
            },
        )
        
        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"

        self.terminations.time_out = None
        


class G1SteppingStonesBaselinePlayCfg(G1SteppingStonesBaselineEnvCfg):
    def __init__(self):
        super().__init__()
        
        self.scene.num_envs = 2
        
        self.observations.policy.enable_corruption = False
        
        self.events.push_robot.interval_range_s = (2.0, 2.0)
        
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.terrain_generator.difficulty_range = (0.2, 0.4)
        