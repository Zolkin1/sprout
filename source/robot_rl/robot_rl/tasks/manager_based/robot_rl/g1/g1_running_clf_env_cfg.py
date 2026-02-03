import torch
from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_walking_clf_env_cfg import (G1TrajOptObservationsCfg,)
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidCommandsCfg,)
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import math
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from ..humanoid_env_cfg import HumanoidEventsCfg
from ..mdp.commands.treadmill_velocity_command_cfg import TreadmillVelocityCommandCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg
from .g1_trajopt_reward import G1TrajOptCLFRewards
from .g1_trajopt_obs import G1TrajOptObservationsCfg
from robot_rl.assets.robots.g1_21j import (G1_MINIMAL_CFG, G1_ACTION_SCALE,)  # isort: skip

##
# Lyapunov Weights
##
RUNNING_Q_weights = {}
RUNNING_Q_weights["com:pos_x"] = [25.0, 250.0]
RUNNING_Q_weights["com:pos_y"] = [500.0, 20.0]
RUNNING_Q_weights["com:pos_z"] = [650.0, 10.0]

RUNNING_Q_weights["left_ankle_roll_link:pos_x"] = [1500.0, 50.0]
RUNNING_Q_weights["left_ankle_roll_link:pos_y"] = [1500.0, 50.0]
RUNNING_Q_weights["left_ankle_roll_link:pos_z"] = [2500.0, 50.0]
RUNNING_Q_weights["left_ankle_roll_link:ori_x"] = [30.0, 1.0]
RUNNING_Q_weights["left_ankle_roll_link:ori_y"] = [150.0, 1.0]
RUNNING_Q_weights["left_ankle_roll_link:ori_z"] = [400.0, 10.0]

RUNNING_Q_weights["right_ankle_roll_link:pos_x"] = [1500.0, 50.0]
RUNNING_Q_weights["right_ankle_roll_link:pos_y"] = [1500.0, 50.0]
RUNNING_Q_weights["right_ankle_roll_link:pos_z"] = [2500.0, 50.0]
RUNNING_Q_weights["right_ankle_roll_link:ori_x"] = [30.0, 1.0]
RUNNING_Q_weights["right_ankle_roll_link:ori_y"] = [150.0, 1.0]
RUNNING_Q_weights["right_ankle_roll_link:ori_z"] = [400.0, 10.0]

RUNNING_Q_weights["joint:left_hip_roll_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:left_hip_pitch_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:left_hip_yaw_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:left_knee_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:left_ankle_roll_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:left_ankle_pitch_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:right_hip_roll_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:right_hip_pitch_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:right_hip_yaw_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:right_knee_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:right_ankle_roll_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:right_ankle_pitch_joint"] = [50.0, 1.0]

RUNNING_Q_weights["pelvis_link:pos_x"] = [25.0, 250.0]
RUNNING_Q_weights["pelvis_link:pos_y"] = [500.0, 20.0]
RUNNING_Q_weights["pelvis_link:pos_z"] = [650.0, 10.0]
RUNNING_Q_weights["pelvis_link:ori_x"] = [300.0, 20.0]
RUNNING_Q_weights["pelvis_link:ori_y"] = [250.0, 10.0]
RUNNING_Q_weights["pelvis_link:ori_z"] = [300.0, 30.0]

RUNNING_Q_weights["joint:waist_yaw_joint"] = [100.0, 1.0]
RUNNING_Q_weights["joint:left_elbow_joint"] = [30.0, 1.0]
RUNNING_Q_weights["joint:left_shoulder_pitch_joint"] = [40.0, 1.0]
RUNNING_Q_weights["joint:left_shoulder_roll_joint"] = [40.0, 1.0]
RUNNING_Q_weights["joint:left_shoulder_yaw_joint"] = [50.0, 1.0]
RUNNING_Q_weights["joint:right_elbow_joint"] = [30.0, 1.0]
RUNNING_Q_weights["joint:right_shoulder_pitch_joint"] = [40.0, 1.0]
RUNNING_Q_weights["joint:right_shoulder_roll_joint"] = [40.0, 1.0]
RUNNING_Q_weights["joint:right_shoulder_yaw_joint"] = [50.0, 1.0]

RUNNING_Q_weights["right_wrist_yaw_link:pos_x"] = [50.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:pos_y"] = [50.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:pos_z"] = [50.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:ori_x"] = [15.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:ori_y"] = [15.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:ori_z"] = [15.0, 1.0]

RUNNING_Q_weights["left_wrist_yaw_link:pos_x"] = [50.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:pos_y"] = [50.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:pos_z"] = [50.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:ori_x"] = [15.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:ori_y"] = [15.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:ori_z"] = [15.0, 1.0]

RUNNING_R_weights = {}
RUNNING_R_weights["com:pos_x"] = [0.1]
RUNNING_R_weights["com:pos_y"] = [0.1]
RUNNING_R_weights["com:pos_z"] = [0.1]

RUNNING_R_weights["left_ankle_roll_link:pos_x"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:pos_y"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:pos_z"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:ori_x"] = [0.02]
RUNNING_R_weights["left_ankle_roll_link:ori_y"] = [0.02]
RUNNING_R_weights["left_ankle_roll_link:ori_z"] = [0.02]

RUNNING_R_weights["right_ankle_roll_link:pos_x"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:pos_y"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:pos_z"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:ori_x"] = [0.02]
RUNNING_R_weights["right_ankle_roll_link:ori_y"] = [0.02]
RUNNING_R_weights["right_ankle_roll_link:ori_z"] = [0.02]

RUNNING_R_weights["joint:left_hip_roll_joint"] = [0.01]
RUNNING_R_weights["joint:left_hip_pitch_joint"] = [0.01]
RUNNING_R_weights["joint:left_hip_yaw_joint"] = [0.01]
RUNNING_R_weights["joint:left_knee_joint"] = [0.01]
RUNNING_R_weights["joint:left_ankle_roll_joint"] = [0.01]
RUNNING_R_weights["joint:left_ankle_pitch_joint"] = [0.01]
RUNNING_R_weights["joint:right_hip_roll_joint"] = [0.01]
RUNNING_R_weights["joint:right_hip_pitch_joint"] = [0.01]
RUNNING_R_weights["joint:right_hip_yaw_joint"] = [0.01]
RUNNING_R_weights["joint:right_knee_joint"] = [0.01]
RUNNING_R_weights["joint:right_ankle_roll_joint"] = [0.01]
RUNNING_R_weights["joint:right_ankle_pitch_joint"] = [0.01]

RUNNING_R_weights["pelvis_link:pos_x"] = [0.05]
RUNNING_R_weights["pelvis_link:pos_y"] = [0.05]
RUNNING_R_weights["pelvis_link:pos_z"] = [0.05]
RUNNING_R_weights["pelvis_link:ori_x"] = [0.05]
RUNNING_R_weights["pelvis_link:ori_y"] = [0.05]
RUNNING_R_weights["pelvis_link:ori_z"] = [0.05]

RUNNING_R_weights["joint:waist_yaw_joint"] = [0.1]
RUNNING_R_weights["joint:left_elbow_joint"] = [0.01]
RUNNING_R_weights["joint:left_shoulder_pitch_joint"] = [0.01]
RUNNING_R_weights["joint:left_shoulder_roll_joint"] = [0.01]
RUNNING_R_weights["joint:left_shoulder_yaw_joint"] = [0.01]
RUNNING_R_weights["joint:right_elbow_joint"] = [0.01]
RUNNING_R_weights["joint:right_shoulder_pitch_joint"] = [0.01]
RUNNING_R_weights["joint:right_shoulder_roll_joint"] = [0.01]
RUNNING_R_weights["joint:right_shoulder_yaw_joint"] = [0.01]

RUNNING_R_weights["right_wrist_yaw_link:pos_x"] = [0.05]
RUNNING_R_weights["right_wrist_yaw_link:pos_y"] = [0.05]
RUNNING_R_weights["right_wrist_yaw_link:pos_z"] = [0.05]
RUNNING_R_weights["right_wrist_yaw_link:ori_x"] = [0.05]
RUNNING_R_weights["right_wrist_yaw_link:ori_y"] = [0.05]
RUNNING_R_weights["right_wrist_yaw_link:ori_z"] = [0.05]

RUNNING_R_weights["left_wrist_yaw_link:pos_x"] = [0.05]
RUNNING_R_weights["left_wrist_yaw_link:pos_y"] = [0.05]
RUNNING_R_weights["left_wrist_yaw_link:pos_z"] = [0.05]
RUNNING_R_weights["left_wrist_yaw_link:ori_x"] = [0.05]
RUNNING_R_weights["left_wrist_yaw_link:ori_y"] = [0.05]
RUNNING_R_weights["left_wrist_yaw_link:ori_z"] = [0.05]
# RUNNING_EE_Q_weights_GL = [
#     25.0,   250.0,      # com_x pos, vel
#     500.0,   20.0,      # com_y pos, vel
#     650.0,   10.0,      # com_z pos, vel
#     300.0,    20.0,     # pelvis_roll pos, vel
#     250.0,    10.0,     # pelvis_pitch pos, vel
#     300.0,    30.0,     # pelvis_yaw pos, vel
#     1500.0, 50.0,       # swing_x pos, vel
#     1500.0,  50.0,      # swing_y pos, vel
#     2500.0, 50.0,       # swing_z pos, vel
#     30.0,    1.0,       # swing_ori_roll pos, vel
#     150.0,    1.0,       # swing_ori_pitch pos, vel
#     400.0,    10.0,     # swing_ori_yaw pos, vel
#     1500.0, 50.0,       # stance_x pos, vel
#     1500.0,  50.0,      # stance_y pos, vel
#     2500.0, 50.0,       # stance_z pos, vel
#     30.0,    1.0,       # stance_ori_roll pos, vel
#     150.0,    1.0,       # stance_ori_pitch pos, vel
#     400.0,    10.0,     # swing_ori_yaw pos, vel
#     100.0,    1.0,      # waist_yaw pos, vel
#     40.0,1.0, #left shoulder pitch
#     40.0,1.0, #left shoulder roll
#     50.0,1.0, #left shoulder yaw
#     30.0,1.0, #left elbow
#     40.0,1.0, #right shoulder pitch
#     40.0,1.0, #right shoulder roll
#     50.0,1.0, #right shoulder yaw
#     30.0,1.0, #right elbow
# ]


# RUNNING_EE_R_weights_GL = [
#         0.1, 0.1, 0.1,      # CoM inputs: allow moderate effort
#         0.05,0.05,0.05,     # pelvis inputs: lower torque priority
#         0.05,0.05,0.05,     # swing foot linear inputs
#         0.02,0.02,0.02,     # swing foot orientation inputs: small adjustments
#         0.05, 0.05, 0.05,   # stance foot linear inputs
#         0.02, 0.02, 0.02,   # stance foot orientation inputs: small adjustments
#         0.1,0.01,0.01,
#         0.01,0.01,0.01,
#         0.01,0.01,0.01,
#     ]

def heuristic_modification(env, output_names, outputs, contact_bodies, contact_states,
                           phi, total_time, threshold):
    """
    Heuristically modify the gait library to allow for sideways walking and turning.

    See _apply_swing_modifications in gait_library_traj.py

    Args:
        env: Environment object.
        output_names: Names of the output variables in order.
        outputs: Output variables.
        contact_bodies: Names of the contact bodies. Of shape [num_contact_bodies]
        contact_states: tensor of shape [N, num_contact_bodies]
        time_into_domain: tensor of shape [N] giving the total time for the current domain each env is in
    """

    # Get the commanded velocity
    vel_cmd = env.command_manager.get_command("base_velocity").clone()

    # Don't apply modifications when in standing
    standing_mask = torch.abs(vel_cmd[:, 0]) < threshold
    vel_cmd[standing_mask, :] *= 0.0

    # Time into half period
    phi_half = torch.remainder(phi, 0.5)
    time_half = total_time / 2.0
    time_into_step = time_half * phi_half

    def find_idx(strings, *substrings):
        """Find index of first string containing all substrings."""
        return next((i for i, s in enumerate(strings) if all(sub in s for sub in substrings)), None)

    # Determine yaw modification
    delta_psi = vel_cmd[:, 2] * time_into_step

    # Determine horizontal modification
    delta_y = vel_cmd[:, 1] * time_into_step

    # TODO: Deal with forward vs backward direction

    # Iterate through the bodies not in contact
    for i, body in enumerate(contact_bodies):
        env_idx = torch.where(contact_states[:, i] == 0)[0]

        ##
        # Adjust this body
        ##
        # Apply yaw and horizontal modifications
        # ori_z is the yaw
        # TODO: Need to search NOT contact_bodies but the output names
        idx = find_idx(output_names, "ori_z", body)
        if idx is not None:
            outputs[env_idx, 0, idx] += delta_psi[env_idx]
            outputs[env_idx, 1, idx] += vel_cmd[env_idx, 2]


        idx = find_idx(output_names, "pos_y", body)
        if idx is not None:
            outputs[env_idx, 0, idx] += delta_y[env_idx]
            outputs[env_idx, 1, idx] += vel_cmd[env_idx, 1]


        # Adjust the hip yaw
        if "left" in body:
            idx = find_idx(output_names, "yaw", "left_hip_yaw_joint")
        else:
            idx = find_idx(output_names, "yaw", "right_hip_yaw_joint")

        if idx is not None:
            outputs[env_idx, 0, idx] += delta_psi[env_idx]
            outputs[env_idx, 1, idx] += vel_cmd[env_idx, 2]

        ##
        # Hip Roll
        ##
        # Adjust hip roll based on the height of the foot
        if "left" in body:
            hip_roll_link_name = "left_hip_roll_link"
            hip_roll_joint_name = "left_hip_roll_joint"
            ankle_link_name = "left_ankle_roll_link"
        else:
            hip_roll_link_name = "right_hip_roll_link"
            hip_roll_joint_name = "right_hip_roll_joint"
            ankle_link_name = "right_ankle_roll_link"

        # Get the height of the hip roll link and the foot
        robot = env.scene["robot"]
        hip_roll_idx = robot.body_names.index(hip_roll_link_name)
        ankle_idx = robot.body_names.index(ankle_link_name)

        hip_roll_height = robot.data.body_pos_w[:, hip_roll_idx, 2]
        foot_height = robot.data.body_pos_w[:, ankle_idx, 2]

        # Vertical distance from hip roll to foot (adjacent side of right triangle)
        vertical_distance = torch.abs(hip_roll_height - foot_height)

        # Use trig: tan(theta) = opposite / adjacent = delta_y / vertical_distance
        # Therefore: theta = atan(delta_y / vertical_distance)
        required_roll_angle = torch.atan2(delta_y, vertical_distance)

        # Find and update the hip roll joint output
        idx = find_idx(output_names, hip_roll_joint_name)
        if idx is not None:
            outputs[env_idx, 0, idx] += required_roll_angle[env_idx]
            # Velocity: d(theta)/dt = (1 / vertical_distance) * d(delta_y)/dt for small angles
            outputs[env_idx, 1, idx] += vel_cmd[env_idx, 1] / vertical_distance[env_idx]


    # Adjust hands yaw
    idx = find_idx(output_names, "ori_z", "right_wrist_yaw_link")
    if idx is not None:
        outputs[:, 0, idx] += delta_psi
        outputs[:, 1, idx] += vel_cmd[:, 2]

    idx = find_idx(output_names, "ori_z", "left_wrist_yaw_link")
    if idx is not None:
        outputs[:, 0, idx] += delta_psi
        outputs[:, 1, idx] += vel_cmd[:, 2]

    # Adjust hands y
    idx = find_idx(output_names, "pos_y", "right_wrist_yaw_link")
    if idx is not None:
        outputs[:, 0, idx] += delta_y
        outputs[:, 1, idx] += vel_cmd[:, 1]

    idx = find_idx(output_names, "pos_y", "left_wrist_yaw_link")
    if idx is not None:
        outputs[:, 0, idx] += delta_y
        outputs[:, 1, idx] += vel_cmd[:, 1]

    # Adjust pelvis yaw
    idx = find_idx(output_names, "ori_z", "pelvis_link")
    if idx is not None:
        outputs[:, 0, idx] += delta_psi
        outputs[:, 1, idx] += vel_cmd[:, 2]

    # Adjust pelvis y
    idx = find_idx(output_names, "pos_y", "pelvis_link")
    if idx is not None:
        outputs[:, 0, idx] += delta_y
        outputs[:, 1, idx] += vel_cmd[:, 1]

    # Adjust COM y
    idx = find_idx(output_names, "pos_y", "com")
    if idx is not None:
        outputs[:, 0, idx] += delta_y
        outputs[:, 1, idx] += vel_cmd[:, 1]

    return outputs

@configclass
class G1RunningGaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    # hzd_ref = GaitLibraryHZDCommandCfg(
    #     trajectory_type="end_effector",
    #     gait_library_path="source/robot_rl/robot_rl/assets/robots/running_library_v7",
    #     config_name="full",
    #
    #     gait_velocity_ranges=(1.1, 3.0, 0.1),
    #     use_standing=True,
    #
    #     num_outputs=27,
    #     Q_weights = RUNNING_EE_Q_weights_GL,
    #     R_weights = RUNNING_EE_R_weights_GL
    # )
    traj_ref = TrajectoryCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        # manager_type = "trajectory",
        # path="source/robot_rl/robot_rl/assets/robots/test_walking_trajectories",

        manager_type="library",
        # path="source/robot_rl/robot_rl/assets/robots/test_walking_library",
        hf_repo = "zolkin/robot_rl",
        path = "trajectories/running",

        conditioner_generator_name = "base_velocity",
        num_outputs = 45, #27, #45, #51, #31, #27,
        Q_weights = RUNNING_Q_weights,
        R_weights = RUNNING_R_weights,
        hold_phi_threshold = 0.1,
        heuristic_func=heuristic_modification,
        phasing_boundaries=4,
    )

    base_velocity = TreadmillVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.6,
        rel_y_envs=0.6,
        heading_command=True,
        heading_control_stiffness=0.5,
        y_pos_kp=1.5, #0.4,
        y_pos_kd=0.3,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ))

@configclass
class G1RunningObservationCfg(G1TrajOptObservationsCfg):
    """Configuration for running gait library observations."""
    pass
    # @configclass
    # class G1RunningPolicyCfg(G1HZDObservationsCfg.PolicyCfg):
    #     # Add the domain flag
    #     domain_flag = ObsTerm(func=mdp.domain_flag, params={"command_name": "hzd_ref"}, history_length=0)
    #     root_quat_w = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.2, n_max=0.2))
    #     # base_z = ObsTerm(func=mdp.base_z, noise=Unoise(n_min=-0.2, n_max=0.2))
    #
    #
    # @configclass
    # class G1RunningCriticCfg(G1HZDObservationsCfg.CriticCfg):
    #     # Add the domain flag
    #     domain_flag = ObsTerm(func=mdp.domain_flag, params={"command_name": "hzd_ref"}, history_length=0)
    #     root_quat_w = ObsTerm(func=mdp.root_quat_w)

    # observation groups
    # policy: G1RunningPolicyCfg = G1RunningPolicyCfg()
    # critic: G1RunningCriticCfg = G1RunningCriticCfg()

@configclass
class G1RunningRewardCfg(G1TrajOptCLFRewards):
    # TODO: Update flight contact penalty
    flight_contact_penalty = RewTerm(
        func=mdp.contact_schedule_penalty,
        weight=-3.0,
        params={"command_name": "traj_ref",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                "weight_scalar": 0.0},
    )

    torque_lims = RewTerm(
        func=mdp.torque_limits,
        weight=-1.0,
    )

@configclass
class G1RunningCurriculumCfg:
    pass
    contact_penalty_curriculum = CurrTerm(func=mdp.contact_curriculum,
                                          params={"update_interval": 40000, #20000,
                                                   "max_weight": 1.0,
                                                   "update_amnt": 0.1})

    clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 30000, "min_max_err": (0.25, 0.3, 0.2) })


@configclass
class G1RunningEventsCfg(HumanoidEventsCfg):
    pass

@configclass
class G1RunningGaitLibraryEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 running gait library environment."""
    commands: G1RunningGaitLibraryCommandsCfg = G1RunningGaitLibraryCommandsCfg()
    observations: G1RunningObservationCfg = G1RunningObservationCfg()
    rewards: G1RunningRewardCfg = G1RunningRewardCfg()
    curriculum: G1RunningCurriculumCfg = G1RunningCurriculumCfg()
    events: G1RunningEventsCfg = G1RunningEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.commands.base_velocity.ranges.lin_vel_x = (1.1, 3.0)  # Note the curriculum for increasing

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0, 0)


        self.rewards.holonomic_constraint.params["command_name"] = "traj_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "traj_ref"

        self.rewards.clf_reward.params = {
            "command_name": "traj_ref",
            "max_eta_err": 0.3,
        }
        self.rewards.clf_decreasing_condition.params = {
            "command_name": "traj_ref",
            "alpha": 0.5,
            "eta_max": 0.25,
            "eta_dot_max": 0.3,
        }
        self.rewards.clf_decreasing_condition.weight = -1
        # self.curriculum.clf_curriculum = None
        # self.curriculum.clf_curriculum.params = {
        #     "min_max_err": (0.1,0.1,0.2),
        #     "scale": (0.005,0.005,0.005), #0.001
        #     "update_interval": 20000
        # }

        self.curriculum.terrain_levels = None

        self.rewards.dof_acc_l2 = None
        self.rewards.dof_vel_l2 = None

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG

        # Other rewards
        self.rewards.dof_torques_l2.weight = -1.0e-5

        ##
        # Domain randomization
        ##
        # self.events.randomize_ground_contact_friction = None
        # self.events.add_base_mass = None
        # self.events.base_com = None
        self.events.base_external_force_torque = None
        # self.events.push_robot = None

        # Update the ground restitution range
        self.events.randomize_ground_contact_friction.params['restitution_range'] = (0.0, 0.2)

        # Update push forces
        self.events.push_robot.params['velocity_range'] = {"x": (-0.75, 0.75), "y": (-0.75, 0.75)}

        # Make the COM randomization on the torso rather than the pelvis
        self.events.base_com.params['asset_cfg'] = SceneEntityCfg("robot", body_names="waist_yaw_link")
        self.events.add_base_mass.params['asset_cfg'] = SceneEntityCfg("robot", body_names="waist_yaw_link")


        # randomize joint parameters and actuator gains
        # self.events.actuator_gain = EventTerm(
        #     func=mdp.randomize_actuator_gains,
        #     mode="startup",
        #     params={
        #             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        #             "stiffness_distribution_params": (-10, 10.),
        #             "damping_distribution_params": (-2., 2.),
        #             "operation": "add",
        #             "distribution": "uniform"
        #     },
        # )
        self.events.gain_randomization.params['operation'] = "scale"
        self.events.gain_randomization.params['stiffness_distribution_params'] = (0.9,1.1)
        self.events.gain_randomization.params['damping_distribution_params'] = (0.9,1.1)

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

        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
        self.commands.base_velocity.ranges.resampling_time_range=(4.0, 4.0)
        self.commands.base_velocity.rel_y_envs = 1.0
        self.commands.base_velocity.debug_vis = False

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
