from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
# from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1FlatHZDEnvCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import G1HZDObservationsCfg
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg
import math

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

class G1GaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    hzd_ref = GaitLibraryHZDCommandCfg(
        trajectory_type="end_effector",
        gait_library_path="source/robot_rl/robot_rl/assets/robots/walking_10_13",
        config_name="walking",
        gait_velocity_ranges=(0.0, 1.0, 0.1),

        num_outputs=27,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
    )


@configclass
class G1GaitLibraryEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 environment with gait library."""
    commands: G1GaitLibraryCommandsCfg = G1GaitLibraryCommandsCfg()
    observations: G1HZDObservationsCfg = G1HZDObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.75)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0,0)

        self.commands.step_period.period_range = (0.71,0.71)

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
        self.curriculum.clf_curriculum = None
        self.curriculum.terrain_levels = None

        self.events.reset_base.params["pose_range"]["yaw"] = (0,0)
        
        self.rewards.dof_acc_l2 = None
        self.rewards.dof_vel_l2 = None

        # self.curriculum.clf_curriculum.params = {
        #     "min_max_err": (0.1,0.1),
        #     "scale": (0.001,0.001),
        #     "update_interval": 20000
        # }

        self.rewards.vdot_tanh = RewTerm(
            func=mdp.vdot_tanh,
            weight= 2.0,
            params={
                "command_name": "hzd_ref",
                "alpha": 1.0,
            }
        )

        # self.rewards.clf_decreasing_condition = None

        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG

@configclass
class G1FlatRefTrackingEnvCfg(G1GaitLibraryEnvCfg):
    """Configuration for the G1 Flat environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # self.rewards.clf_reward = None
        self.rewards.clf_decreasing_condition = None
        self.curriculum.clf_curriculum = None


@configclass
class G1_custom_plate_GaitLibraryEnvCfg(G1GaitLibraryEnvCfg):
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
        




class G1GL_PlayEnvCfg(G1_custom_plate_GaitLibraryEnvCfg):
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
        
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0,0)

        # self.events.reset_base.params["pose_range"]["yaw"] = (-3.14,3.14)
        # self.events.reset_base.params["pose_range"]["x"] = (-3,3)
        # self.events.reset_base.params["pose_range"]["y"] = (-3,3)