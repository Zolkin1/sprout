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
from robot_rl.tasks.manager_based.robot_rl.terrains.stepping_stones_cfg import LongStonesTerrainCfg
from robot_rl.tasks.manager_based.robot_rl.constants import STONES, TEST_FLAT
        
        
        
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import (
    ROUGH_SLOPED_FOR_FLAT_HZD_CFG,
)


        
@configclass
class G1RoughMlipCommandsCfg(HumanoidCommandsCfg):
    """Commands for the G1 Flat environment."""

    hlip_ref = StonesOutputCommandCfg()
    


@configclass
class CurriculumMlipCfg:
    """Curriculum terms for the MDP."""

    # clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 1000, "min_val": 20.0})
    
    # todo: stones curriculum
    if TEST_FLAT==False:
        stones_curriculum = CurrTerm(func=mdp.stones_sagittal_terrain_levels_vel)
    


# Lip specific rewards
##
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

@configclass
class G1SteppingStonesEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Flat environment."""

    rewards: G1RoughMlipRewards = G1RoughMlipRewards()
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation_stepping_stones import (
    G1SteppingStonesObservationsCfg,
)

    observations: G1SteppingStonesObservationsCfg = G1SteppingStonesObservationsCfg()
    commands: G1RoughMlipCommandsCfg = G1RoughMlipCommandsCfg()
    curriculum: CurriculumMlipCfg = CurriculumMlipCfg()
    terminations: G1SteppingStonesTerminationsCfg = G1SteppingStonesTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        new_joint_pos = G1_MINIMAL_CFG.init_state.joint_pos | {
            ".*_hip_pitch_joint": -0.25,
            ".*_knee_joint": 0.46,
            ".*_ankle_pitch_joint": -0.25,
        }
        self.scene.robot = G1_MINIMAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=G1_MINIMAL_CFG.init_state.replace(joint_pos=new_joint_pos)
        )

        if TEST_FLAT==False:
            STONES_CFG = TerrainGeneratorCfg(
                class_type=StonesTerrainGenerator,
                size=(STONES.terrain_size_x, STONES.terrain_size_y),
                curriculum=True,
                border_width=1.0,
                border_height=0.0,
                num_rows=10,
                num_cols=40,
                horizontal_scale=0.1,
                vertical_scale=0.0005,
                slope_threshold=0.75,
                use_cache=False,
                difficulty_range=(0.0, 1.0),
                sub_terrains={
                    "stones": LongStonesTerrainCfg(),
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
        else:
            self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        # No height scanner for now
        self.scene.height_scanner = None

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
            "pose_range": {"x": (-0.3, 0.), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)

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


class G1_custom_stepping_stones(G1SteppingStonesEnvCfg):
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

class G1SteppingStonesEnvCfg_PLAY(G1SteppingStonesEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        self.events.push_robot.interval_range_s = (5.0, 5.0)
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.3, 0.0),
            "y": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }