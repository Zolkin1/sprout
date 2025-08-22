from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1FlatHZDEnvCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import G1HZDObservationsCfg
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg


class G1RunningGaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    hzd_ref = GaitLibraryHZDCommandCfg(
        trajectory_type="end_effector",
        gait_library_path="source/robot_rl/robot_rl/assets/robots/running_gait_single",
        config_name="running_config",
        gait_velocity_ranges=(2.22, 2.22, 0),
        use_standing=False,
    )

@configclass
class G1RunningGaitLibraryEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 running gait library environment."""
    commands: G1RunningGaitLibraryCommandsCfg = G1RunningGaitLibraryCommandsCfg()
    observations: G1HZDObservationsCfg = G1HZDObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Set all the environment configs
        self.commands.base_velocity.ranges.lin_vel_x = (2.22, 2.22)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading = (0, 0)

        self.commands.step_period.period_range = (0.9, 0.9)

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
        self.rewards.holonomic_constraint_vel = None
        self.rewards.holonomic_constraint = None
