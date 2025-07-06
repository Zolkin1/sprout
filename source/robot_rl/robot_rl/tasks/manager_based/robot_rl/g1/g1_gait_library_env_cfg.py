from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1FlatHZDEnvCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_observation import G1FlatHZDObservationsCfg

class G1GaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    hzd_ref = GaitLibraryHZDCommandCfg(
        trajectory_type="end_effector",
        gait_library_path="source/robot_rl/robot_rl/assets/robots/gait_library",
        config_name="single_support_config",
        gait_velocity_ranges=(0.3, 0.5, 0.1)
    )


@configclass
class G1GaitLibraryEnvCfg(G1FlatHZDEnvCfg):
    """Configuration for the G1 environment with gait library."""
    observations: G1FlatHZDObservationsCfg = G1FlatHZDObservationsCfg()
    commands: G1GaitLibraryCommandsCfg = G1GaitLibraryCommandsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.5)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0, 0)
        
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

        self.curriculum.clf_curriculum = None
        self.rewards.clf_reward.params["max_clf"] = 50.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 50.0
        self.rewards.clf_decreasing_condition.params["alpha"] = 1.0