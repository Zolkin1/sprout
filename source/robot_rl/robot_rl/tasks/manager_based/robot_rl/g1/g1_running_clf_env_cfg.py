import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_apply

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidCommandsCfg,)
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import math
from ..humanoid_env_cfg import HumanoidEventsCfg
from ..mdp.commands.velocity_commands_cfg import VelocityTrackingCommandCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg
from .g1_trajopt_reward import G1TrajOptCLFRewards
from .g1_trajopt_obs import G1TrajOptObservationsCfg
from robot_rl.assets.robots.g1_21j import (G1_MINIMAL_CFG, G1_ACTION_SCALE,)  # isort: skip

##
# Lyapunov Weights
##
RUNNING_Q_weights = {}
# RUNNING_Q_weights["com:pos_x"] = [25.0, 250.0]
# RUNNING_Q_weights["com:pos_y"] = [500.0, 20.0]
# RUNNING_Q_weights["com:pos_z"] = [650.0, 10.0]
#
# RUNNING_Q_weights["left_ankle_roll_link:pos_x"] = [1500.0, 50.0]
# RUNNING_Q_weights["left_ankle_roll_link:pos_y"] = [1500.0, 50.0]
# RUNNING_Q_weights["left_ankle_roll_link:pos_z"] = [2500.0, 50.0]
# RUNNING_Q_weights["left_ankle_roll_link:ori_x"] = [30.0, 1.0]
# RUNNING_Q_weights["left_ankle_roll_link:ori_y"] = [150.0, 1.0]
# RUNNING_Q_weights["left_ankle_roll_link:ori_z"] = [400.0, 10.0]
#
# RUNNING_Q_weights["right_ankle_roll_link:pos_x"] = [1500.0, 50.0]
# RUNNING_Q_weights["right_ankle_roll_link:pos_y"] = [1500.0, 50.0]
# RUNNING_Q_weights["right_ankle_roll_link:pos_z"] = [2500.0, 50.0]
# RUNNING_Q_weights["right_ankle_roll_link:ori_x"] = [30.0, 1.0]
# RUNNING_Q_weights["right_ankle_roll_link:ori_y"] = [150.0, 1.0]
# RUNNING_Q_weights["right_ankle_roll_link:ori_z"] = [400.0, 10.0]
#
# RUNNING_Q_weights["joint:left_hip_roll_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:left_hip_pitch_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:left_hip_yaw_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:left_knee_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:left_ankle_roll_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:left_ankle_pitch_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:right_hip_roll_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:right_hip_pitch_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:right_hip_yaw_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:right_knee_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:right_ankle_roll_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:right_ankle_pitch_joint"] = [50.0, 1.0]
#
# RUNNING_Q_weights["pelvis_link:pos_x"] = [25.0, 250.0]
# RUNNING_Q_weights["pelvis_link:pos_y"] = [500.0, 20.0]
# RUNNING_Q_weights["pelvis_link:pos_z"] = [650.0, 10.0]
# RUNNING_Q_weights["pelvis_link:ori_x"] = [300.0, 20.0]
# RUNNING_Q_weights["pelvis_link:ori_y"] = [250.0, 10.0]
# RUNNING_Q_weights["pelvis_link:ori_z"] = [300.0, 30.0]
#
# RUNNING_Q_weights["joint:waist_yaw_joint"] = [100.0, 1.0]
# RUNNING_Q_weights["joint:left_elbow_joint"] = [30.0, 1.0]
# RUNNING_Q_weights["joint:left_shoulder_pitch_joint"] = [40.0, 1.0]
# RUNNING_Q_weights["joint:left_shoulder_roll_joint"] = [40.0, 1.0]
# RUNNING_Q_weights["joint:left_shoulder_yaw_joint"] = [50.0, 1.0]
# RUNNING_Q_weights["joint:right_elbow_joint"] = [30.0, 1.0]
# RUNNING_Q_weights["joint:right_shoulder_pitch_joint"] = [40.0, 1.0]
# RUNNING_Q_weights["joint:right_shoulder_roll_joint"] = [40.0, 1.0]
# RUNNING_Q_weights["joint:right_shoulder_yaw_joint"] = [50.0, 1.0]
#
# RUNNING_Q_weights["right_wrist_yaw_link:pos_x"] = [50.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:pos_y"] = [50.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:pos_z"] = [50.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:ori_x"] = [15.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:ori_y"] = [15.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:ori_z"] = [15.0, 1.0]
#
# RUNNING_Q_weights["left_wrist_yaw_link:pos_x"] = [50.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:pos_y"] = [50.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:pos_z"] = [50.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:ori_x"] = [15.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:ori_y"] = [15.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:ori_z"] = [15.0, 1.0]


RUNNING_Q_weights["com:pos_x"] = [1.0, 1.0]
RUNNING_Q_weights["com:pos_y"] = [1.0, 1.0]
RUNNING_Q_weights["com:pos_z"] = [1.0, 1.0]

RUNNING_Q_weights["left_ankle_roll_link:pos_x"] = [1.0, 1.0]
RUNNING_Q_weights["left_ankle_roll_link:pos_y"] = [1.0, 1.0]
RUNNING_Q_weights["left_ankle_roll_link:pos_z"] = [1.0, 1.0]
RUNNING_Q_weights["left_ankle_roll_link:ori_x"] = [1.0, 1.0]
RUNNING_Q_weights["left_ankle_roll_link:ori_y"] = [1.0, 1.0]
RUNNING_Q_weights["left_ankle_roll_link:ori_z"] = [1.0, 1.0]

RUNNING_Q_weights["right_ankle_roll_link:pos_x"] = [1.0, 1.0]
RUNNING_Q_weights["right_ankle_roll_link:pos_y"] = [1.0, 1.0]
RUNNING_Q_weights["right_ankle_roll_link:pos_z"] = [1.0, 1.0]
RUNNING_Q_weights["right_ankle_roll_link:ori_x"] = [1.0, 1.0]
RUNNING_Q_weights["right_ankle_roll_link:ori_y"] = [1.0, 1.0]
RUNNING_Q_weights["right_ankle_roll_link:ori_z"] = [1.0, 1.0]

RUNNING_Q_weights["joint:left_hip_roll_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_hip_pitch_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_hip_yaw_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_knee_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_ankle_roll_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_ankle_pitch_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_hip_roll_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_hip_pitch_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_hip_yaw_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_knee_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_ankle_roll_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_ankle_pitch_joint"] = [1.0, 1.0]

RUNNING_Q_weights["pelvis_link:pos_x"] = [1.0, 1.0]
RUNNING_Q_weights["pelvis_link:pos_y"] = [1.0, 1.0]
RUNNING_Q_weights["pelvis_link:pos_z"] = [1.0, 1.0]
RUNNING_Q_weights["pelvis_link:ori_x"] = [1.0, 1.0]
RUNNING_Q_weights["pelvis_link:ori_y"] = [1.0, 1.0]
RUNNING_Q_weights["pelvis_link:ori_z"] = [1.0, 1.0]

RUNNING_Q_weights["joint:waist_yaw_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_elbow_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_shoulder_pitch_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_shoulder_roll_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:left_shoulder_yaw_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_elbow_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_shoulder_pitch_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_shoulder_roll_joint"] = [1.0, 1.0]
RUNNING_Q_weights["joint:right_shoulder_yaw_joint"] = [1.0, 1.0]

RUNNING_Q_weights["right_wrist_yaw_link:pos_x"] = [1.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:pos_y"] = [1.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:pos_z"] = [1.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:ori_x"] = [1.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:ori_y"] = [1.0, 1.0]
RUNNING_Q_weights["right_wrist_yaw_link:ori_z"] = [1.0, 1.0]

RUNNING_Q_weights["left_wrist_yaw_link:pos_x"] = [1.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:pos_y"] = [1.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:pos_z"] = [1.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:ori_x"] = [1.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:ori_y"] = [1.0, 1.0]
RUNNING_Q_weights["left_wrist_yaw_link:ori_z"] = [1.0, 1.0]

# RUNNING_R_weights = {}
# RUNNING_R_weights["com:pos_x"] = [0.1]
# RUNNING_R_weights["com:pos_y"] = [0.1]
# RUNNING_R_weights["com:pos_z"] = [0.1]
#
# RUNNING_R_weights["left_ankle_roll_link:pos_x"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:pos_y"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:pos_z"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:ori_x"] = [0.02]
# RUNNING_R_weights["left_ankle_roll_link:ori_y"] = [0.02]
# RUNNING_R_weights["left_ankle_roll_link:ori_z"] = [0.02]
#
# RUNNING_R_weights["right_ankle_roll_link:pos_x"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:pos_y"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:pos_z"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:ori_x"] = [0.02]
# RUNNING_R_weights["right_ankle_roll_link:ori_y"] = [0.02]
# RUNNING_R_weights["right_ankle_roll_link:ori_z"] = [0.02]
#
# RUNNING_R_weights["joint:left_hip_roll_joint"] = [0.01]
# RUNNING_R_weights["joint:left_hip_pitch_joint"] = [0.01]
# RUNNING_R_weights["joint:left_hip_yaw_joint"] = [0.01]
# RUNNING_R_weights["joint:left_knee_joint"] = [0.01]
# RUNNING_R_weights["joint:left_ankle_roll_joint"] = [0.01]
# RUNNING_R_weights["joint:left_ankle_pitch_joint"] = [0.01]
# RUNNING_R_weights["joint:right_hip_roll_joint"] = [0.01]
# RUNNING_R_weights["joint:right_hip_pitch_joint"] = [0.01]
# RUNNING_R_weights["joint:right_hip_yaw_joint"] = [0.01]
# RUNNING_R_weights["joint:right_knee_joint"] = [0.01]
# RUNNING_R_weights["joint:right_ankle_roll_joint"] = [0.01]
# RUNNING_R_weights["joint:right_ankle_pitch_joint"] = [0.01]
#
# RUNNING_R_weights["pelvis_link:pos_x"] = [0.05]
# RUNNING_R_weights["pelvis_link:pos_y"] = [0.05]
# RUNNING_R_weights["pelvis_link:pos_z"] = [0.05]
# RUNNING_R_weights["pelvis_link:ori_x"] = [0.05]
# RUNNING_R_weights["pelvis_link:ori_y"] = [0.05]
# RUNNING_R_weights["pelvis_link:ori_z"] = [0.05]
#
# RUNNING_R_weights["joint:waist_yaw_joint"] = [0.1]
# RUNNING_R_weights["joint:left_elbow_joint"] = [0.01]
# RUNNING_R_weights["joint:left_shoulder_pitch_joint"] = [0.01]
# RUNNING_R_weights["joint:left_shoulder_roll_joint"] = [0.01]
# RUNNING_R_weights["joint:left_shoulder_yaw_joint"] = [0.01]
# RUNNING_R_weights["joint:right_elbow_joint"] = [0.01]
# RUNNING_R_weights["joint:right_shoulder_pitch_joint"] = [0.01]
# RUNNING_R_weights["joint:right_shoulder_roll_joint"] = [0.01]
# RUNNING_R_weights["joint:right_shoulder_yaw_joint"] = [0.01]
#
# RUNNING_R_weights["right_wrist_yaw_link:pos_x"] = [0.05]
# RUNNING_R_weights["right_wrist_yaw_link:pos_y"] = [0.05]
# RUNNING_R_weights["right_wrist_yaw_link:pos_z"] = [0.05]
# RUNNING_R_weights["right_wrist_yaw_link:ori_x"] = [0.05]
# RUNNING_R_weights["right_wrist_yaw_link:ori_y"] = [0.05]
# RUNNING_R_weights["right_wrist_yaw_link:ori_z"] = [0.05]
#
# RUNNING_R_weights["left_wrist_yaw_link:pos_x"] = [0.05]
# RUNNING_R_weights["left_wrist_yaw_link:pos_y"] = [0.05]
# RUNNING_R_weights["left_wrist_yaw_link:pos_z"] = [0.05]
# RUNNING_R_weights["left_wrist_yaw_link:ori_x"] = [0.05]
# RUNNING_R_weights["left_wrist_yaw_link:ori_y"] = [0.05]
# RUNNING_R_weights["left_wrist_yaw_link:ori_z"] = [0.05]
RUNNING_R_weights = {}
RUNNING_R_weights["com:pos_x"] = [0.05]
RUNNING_R_weights["com:pos_y"] = [0.05]
RUNNING_R_weights["com:pos_z"] = [0.05]

RUNNING_R_weights["left_ankle_roll_link:pos_x"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:pos_y"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:pos_z"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:ori_x"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:ori_y"] = [0.05]
RUNNING_R_weights["left_ankle_roll_link:ori_z"] = [0.05]

RUNNING_R_weights["right_ankle_roll_link:pos_x"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:pos_y"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:pos_z"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:ori_x"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:ori_y"] = [0.05]
RUNNING_R_weights["right_ankle_roll_link:ori_z"] = [0.05]

RUNNING_R_weights["joint:left_hip_roll_joint"] = [0.05]
RUNNING_R_weights["joint:left_hip_pitch_joint"] = [0.05]
RUNNING_R_weights["joint:left_hip_yaw_joint"] = [0.05]
RUNNING_R_weights["joint:left_knee_joint"] = [0.05]
RUNNING_R_weights["joint:left_ankle_roll_joint"] = [0.05]
RUNNING_R_weights["joint:left_ankle_pitch_joint"] = [0.05]
RUNNING_R_weights["joint:right_hip_roll_joint"] = [0.05]
RUNNING_R_weights["joint:right_hip_pitch_joint"] = [0.05]
RUNNING_R_weights["joint:right_hip_yaw_joint"] = [0.05]
RUNNING_R_weights["joint:right_knee_joint"] = [0.05]
RUNNING_R_weights["joint:right_ankle_roll_joint"] = [0.05]
RUNNING_R_weights["joint:right_ankle_pitch_joint"] = [0.05]

RUNNING_R_weights["pelvis_link:pos_x"] = [0.05]
RUNNING_R_weights["pelvis_link:pos_y"] = [0.05]
RUNNING_R_weights["pelvis_link:pos_z"] = [0.05]
RUNNING_R_weights["pelvis_link:ori_x"] = [0.05]
RUNNING_R_weights["pelvis_link:ori_y"] = [0.05]
RUNNING_R_weights["pelvis_link:ori_z"] = [0.05]

RUNNING_R_weights["joint:waist_yaw_joint"] = [0.05]
RUNNING_R_weights["joint:left_elbow_joint"] = [0.05]
RUNNING_R_weights["joint:left_shoulder_pitch_joint"] = [0.05]
RUNNING_R_weights["joint:left_shoulder_roll_joint"] = [0.05]
RUNNING_R_weights["joint:left_shoulder_yaw_joint"] = [0.05]
RUNNING_R_weights["joint:right_elbow_joint"] = [0.05]
RUNNING_R_weights["joint:right_shoulder_pitch_joint"] = [0.05]
RUNNING_R_weights["joint:right_shoulder_roll_joint"] = [0.05]
RUNNING_R_weights["joint:right_shoulder_yaw_joint"] = [0.05]

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

def _find_idx(strings, *substrings):
    """Find index of first string containing all substrings."""
    return next((i for i, s in enumerate(strings) if all(sub in s for sub in substrings)), None)


def _build_heuristic_cache(output_pos_names, output_vel_names, contact_bodies, env):
    """Build all index lookups once for the heuristic function."""
    cache = {}
    robot = env.scene["robot"]

    # Cache per-body output indices
    cache['body_output_indices'] = {}
    cache['body_robot_indices'] = {}

    for body in contact_bodies:
        side = "left" if "left" in body else "right"

        # Output indices for this body
        cache['body_output_indices'][body] = {
            'quat': [_find_idx(output_pos_names, "ori_w", body),
                     _find_idx(output_pos_names, "ori_x", body),
                     _find_idx(output_pos_names, "ori_y", body),
                     _find_idx(output_pos_names, "ori_z", body)],

            # 'ori_z_vel': _find_idx(output_vel_names, "ori_z", body),

            'pos': [_find_idx(output_pos_names, "pos_x", body),
                    _find_idx(output_pos_names, "pos_y", body),
                    _find_idx(output_pos_names, "pos_z", body)],

            'lin_vel': [_find_idx(output_vel_names, "pos_x", body),
                        _find_idx(output_vel_names, "pos_y", body),
                        _find_idx(output_vel_names, "pos_z", body)],

            'ang_vel': [_find_idx(output_vel_names, "ori_x", body),
                        _find_idx(output_vel_names, "ori_y", body),
                        _find_idx(output_vel_names, "ori_z", body)],

            'hip_yaw': _find_idx(output_pos_names, "yaw", f"{side}_hip_yaw_joint"),
            'hip_yaw_vel': _find_idx(output_vel_names, "yaw", f"{side}_hip_yaw_joint"),

            'hip_roll': _find_idx(output_pos_names, f"{side}_hip_roll_joint"),
            'hip_roll_vel': _find_idx(output_vel_names, f"{side}_hip_roll_joint"),
        }

        # Robot body indices for this contact body
        cache['body_robot_indices'][body] = {
            'hip_roll': robot.body_names.index(f"{side}_hip_roll_link"),
            'ankle': robot.body_names.index(f"{side}_ankle_roll_link"),
        }

    cache['com_pos_y'] = _find_idx(output_pos_names, "pos_y", "com")
    cache['com_pos_y_vel'] = _find_idx(output_vel_names, "pos_y", "com")

    for body in ["pelvis_link", "right_wrist_yaw_link", "left_wrist_yaw_link"]:
        if _find_idx(output_pos_names, "ori_w", body) is not None:
            cache[body + '_quat'] = [_find_idx(output_pos_names, "ori_w", body),
                                     _find_idx(output_pos_names, "ori_x", body),
                                     _find_idx(output_pos_names, "ori_y", body),
                                     _find_idx(output_pos_names, "ori_z", body)]

            # cache[body + '_ori_z_vel'] = _find_idx(output_vel_names, "ori_z", body),

            cache[body + '_ang_vel'] = [_find_idx(output_vel_names, "ori_x", body),
                                        _find_idx(output_vel_names, "ori_y", body),
                                        _find_idx(output_vel_names, "ori_z", body)],

        if _find_idx(output_pos_names, "pos_y", body) is not None:
            cache[body + '_pos'] = [_find_idx(output_pos_names, "pos_x", body),
                                    _find_idx(output_pos_names, "pos_y", body),
                                    _find_idx(output_pos_names, "pos_z", body)],

            cache[body + '_lin_vel'] = [_find_idx(output_vel_names, "pos_x", body),
                                        _find_idx(output_vel_names, "pos_y", body),
                                        _find_idx(output_vel_names, "pos_z", body)],

    return cache


# Module-level cache for heuristic function (Hydra requires function to be findable by name)
_HEURISTIC_CACHE = {}


def heuristic_modification(env,
                           output_pos_names,
                           output_vel_names,
                           pos_outputs,
                           vel_outputs,
                           contact_bodies,
                           contact_states,
                           phi,
                           total_time,
                           env_ids,
                           threshold):
    """Heuristically modify the gait library for sideways walking and turning.

    Indices are cached on first call for performance.

    Args:
        env: Environment object.
        output_pos_names: Names of the output variables in order.
        output_vel_names: Names of the output variables in order.
        pos_outputs: Output variables.
        vel_outputs: Output variables.
        contact_bodies: Names of the contact bodies. Of shape [num_contact_bodies]
        contact_states: tensor of shape [N, num_contact_bodies]
        phi: Phasing variable tensor of shape [N]
        total_time: Total time for the trajectory
        threshold: Velocity threshold for standing detection
    """
    global _HEURISTIC_CACHE

    # Initialize cache on first call
    if not _HEURISTIC_CACHE:
        _HEURISTIC_CACHE.update(_build_heuristic_cache(output_pos_names, output_vel_names, contact_bodies, env))

    # Get the commanded velocity
    vel_cmd = env.command_manager.get_command("base_velocity").clone()

    # TODO: Do I need to account for total time in the air so that the stance phase is accounted for
    if env_ids is None:
        # Don't apply modifications when in standing
        standing_mask = torch.abs(vel_cmd[:, 0]) < threshold
        vel_cmd[standing_mask, :] *= 0.0

        # Time into half period
        phi_half = torch.remainder(phi, 0.5) / 0.5
        time_half = total_time / 2.0
        time_into_step = time_half * phi_half


        # Determine yaw modifications
        delta_psi = vel_cmd[:, 2] * time_into_step

        # Create yaw quaternion
        delta_psi_quat = quat_from_euler_xyz(torch.zeros_like(delta_psi), torch.zeros_like(delta_psi), delta_psi)

        # Horizontal modification
        delta_y = vel_cmd[:, 1] * time_into_step

    else:
        # Don't apply modifications when in standing
        standing_mask = torch.abs(vel_cmd[:, 0]) < threshold
        vel_cmd[standing_mask, :] *= 0.0

        # Time into half period
        phi_half = torch.remainder(phi, 0.5) / 0.5
        time_half = total_time / 2.0
        time_into_step = time_half * phi_half

        # Determine yaw modifications
        delta_psi = vel_cmd[env_ids, 2] * time_into_step

        # Create yaw quaternion
        delta_psi_quat = quat_from_euler_xyz(torch.zeros_like(delta_psi), torch.zeros_like(delta_psi), delta_psi)

        # Horizontal modification
        delta_y = vel_cmd[env_ids, 1] * time_into_step

    # Get robot reference once
    # robot = env.scene["robot"]

    # Iterate through the bodies not in contact
    for i, body in enumerate(contact_bodies):
        env_idx = torch.where(contact_states[:, i] == 0)[0]
        if len(env_idx) != 0:

            body_indices = _HEURISTIC_CACHE['body_output_indices'][body]
            robot_indices = _HEURISTIC_CACHE['body_robot_indices'][body]

            # Apply yaw modification
            quat_idx = body_indices['quat']
            ang_vel_idx = body_indices['ang_vel']

            pos_outputs[env_idx, quat_idx[0]:quat_idx[-1]+1] = quat_mul(delta_psi_quat[env_idx, :], pos_outputs[env_idx, quat_idx[0]:quat_idx[-1]+1])
            # Use global indices for vel_cmd when in subset mode
            if env_ids is None:
                vel_outputs[env_idx, ang_vel_idx[2]] += vel_cmd[env_idx, 2]
            else:
                vel_outputs[env_idx, ang_vel_idx[2]] += vel_cmd[env_ids[env_idx], 2]
            # vel_outputs[env_idx, ang_vel_idx[0]:ang_vel_idx[-1]+1] = quat_apply(delta_psi_quat, )

            # Apply horizontal modification (pos_y)
            pos_idx = body_indices['pos']
            lin_vel_idx = body_indices['lin_vel']

            pos_outputs[env_idx, pos_idx[1]] += delta_y[env_idx]
            if env_ids is None:
                vel_outputs[env_idx, lin_vel_idx[1]] += vel_cmd[env_idx, 1]
            else:
                vel_outputs[env_idx, lin_vel_idx[1]] += vel_cmd[env_ids[env_idx], 1]

            # Apply yaw modification to position and linear vel
            pos_outputs[env_idx, pos_idx[0]:pos_idx[-1]+1] = quat_apply(delta_psi_quat[env_idx], pos_outputs[env_idx, pos_idx[0]:pos_idx[-1]+1])

            # # Add cross product term?
            # omega = vel_outputs[env_idx, ang_vel_idx]
            # vel_outputs[env_idx, lin_vel_idx] += torch.cross(omega, pos_outputs[env_idx, pos_idx], dim=1)#[env_idx].squeeze(0)

        # TODO: Try with and without the yaw compensation on the stance leg joint
        # env_idx = torch.where(contact_states[:, i] == 1)[0]
        # if len(env_idx) != 0:
        #     body_indices = _HEURISTIC_CACHE['body_output_indices'][body]
        #     robot_indices = _HEURISTIC_CACHE['body_robot_indices'][body]
        #
        #     # Adjust hip yaw
        #     idx = body_indices['hip_yaw']
        #     idx_vel = body_indices['hip_yaw_vel']
        #     if idx is not None:
        #         pos_outputs[env_idx, idx] -= delta_psi[env_idx]
        #         vel_outputs[env_idx, idx_vel] -= vel_cmd[env_idx, 2]

            # # Adjust hip roll based on foot height
            # hip_roll_idx = robot_indices['hip_roll']
            # ankle_idx = robot_indices['ankle']
            #
            # hip_roll_height = robot.data.body_pos_w[:, hip_roll_idx, 2]
            # foot_height = robot.data.body_pos_w[:, ankle_idx, 2]
            # vertical_distance = torch.abs(hip_roll_height - foot_height)
            #
            # required_roll_angle = torch.atan2(delta_y, vertical_distance)
            #
            # idx = body_indices['hip_roll']
            # if idx is not None:
            #     pos_outputs[env_idx, idx] -= required_roll_angle[env_idx]
            #     vel_outputs[env_idx, idx] -= vel_cmd[env_idx, 1] / vertical_distance[env_idx]

    # Apply global modifications (all envs)
    for body in ["pelvis_link", "right_wrist_yaw_link", "left_wrist_yaw_link"]:
        # Yaw
        if body + '_quat' in _HEURISTIC_CACHE:
            quat_idx = _HEURISTIC_CACHE[body + '_quat']
            vel_idx = _HEURISTIC_CACHE[body + '_ang_vel']

            if env_ids is None:
                pos_outputs[:, quat_idx[0]:quat_idx[-1]+1] = quat_mul(delta_psi_quat, pos_outputs[:, quat_idx[0]:quat_idx[-1]+1])
                vel_outputs[:, vel_idx[0][2]] += vel_cmd[:, 2]
            else:
                # Subset mode: pos_outputs/vel_outputs are subset-sized, use : for them; vel_cmd is full-sized, use env_ids
                pos_outputs[:, quat_idx[0]:quat_idx[-1]+1] = quat_mul(delta_psi_quat, pos_outputs[:, quat_idx[0]:quat_idx[-1]+1])
                vel_outputs[:, vel_idx[0][2]] += vel_cmd[env_ids, 2]

        # Position/linear vel
        if body + '_pos' in _HEURISTIC_CACHE:
            pos_idx = _HEURISTIC_CACHE[body + '_pos']
            lin_vel_idx = _HEURISTIC_CACHE[body + '_lin_vel']

            if env_ids is None:
                # Lateral velocity
                pos_outputs[:, pos_idx[0][1]] += delta_y
                vel_outputs[:, lin_vel_idx[0][1]] += vel_cmd[:, 1]

                # Adjustment due to yaw
                pos_outputs[:, pos_idx[0][0]:pos_idx[0][-1]+1] = quat_apply(delta_psi_quat, pos_outputs[:, pos_idx[0][0]:pos_idx[0][-1]+1])
                vel_outputs[:, lin_vel_idx[0][0]:lin_vel_idx[0][-1]+1] = quat_apply(delta_psi_quat, vel_outputs[:, lin_vel_idx[0][0]:lin_vel_idx[0][-1]+1])
            else:
                # Subset mode: pos_outputs/vel_outputs are subset-sized, vel_cmd is full-sized
                pos_outputs[:, pos_idx[0][1]] += delta_y
                vel_outputs[:, lin_vel_idx[0][1]] += vel_cmd[env_ids, 1]

                # Adjustment due to yaw
                pos_outputs[:, pos_idx[0][0]:pos_idx[0][-1]+1] = quat_apply(delta_psi_quat, pos_outputs[:, pos_idx[0][0]:pos_idx[0][-1]+1])
                vel_outputs[:, lin_vel_idx[0][0]:lin_vel_idx[0][-1]+1] = quat_apply(delta_psi_quat, vel_outputs[:, lin_vel_idx[0][0]:lin_vel_idx[0][-1]+1])

    return pos_outputs, vel_outputs


@configclass
class G1RunningGaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    traj_ref = TrajectoryCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        manager_type="library",
        # path="source/robot_rl/robot_rl/assets/robots/test_walking_library",
        hf_repo = "zolkin/robot_rl",
        path = "trajectories/running",

        conditioner_generator_name = "base_velocity",
        num_outputs = 48, #45, #27, #45, #51, #31, #27,
        Q_weights = RUNNING_Q_weights,
        R_weights = RUNNING_R_weights,
        hold_phi_threshold = 0.1,
        heuristic_func=heuristic_modification,
        phasing_boundaries = 4,
    )

    base_velocity = VelocityTrackingCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.0, 10.0), #(10.0, 10.0),
        rel_standing_envs=0.0, #0.05, #0.02,
        rel_closed_loop=0.55, #0.55,
        rel_closed_loop_yaw=0.25,
        rel_open_loop=0.2,
        debug_vis=True,
        ranges=VelocityTrackingCommandCfg.VelRanges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
            y_pos_offset=(-0.5, 0.5),
            y_kp=(1.2, 1.8),
            y_kd=(0.2, 0.4),
        ))
        # rel_heading_envs=0.6,
        # rel_y_envs=0.6,
        # heading_command=True,
        # heading_control_stiffness=0.5,
        # y_pos_kp=1.5, #0.4,
        # y_pos_kd=0.3,
        # debug_vis=True,
        # ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        # ))

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
    # flight_contact_penalty = RewTerm(
    #     func=mdp.contact_schedule_penalty,
    #     weight=-3.0,
    #     params={"command_name": "traj_ref",
    #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    #             "weight_scalar": 0.0},
    # )

    torque_lims = RewTerm(
        func=mdp.torque_limits,
        weight=-1.0,
    )

    # Base
    base_pos = RewTerm(
        func=mdp.base_pos_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.4}
    )
    base_ori = RewTerm(
        func=mdp.base_ori_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.5}
    )
    base_lin_vel = RewTerm(
        func=mdp.base_lin_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.6}
    )
    base_ang_vel = RewTerm(
        func=mdp.base_ang_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 1.5}
    )

    # Joints
    joint_pos = RewTerm(
        func=mdp.joint_pos_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.3*math.sqrt(21)}
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 6.5*math.sqrt(21)}
    )

    # Bodies
    body_pos = RewTerm(
        func=mdp.body_pos_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.2*math.sqrt(4)}
    )
    body_ori = RewTerm(
        func=mdp.body_ori_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.4 * math.sqrt(4)}
    )
    body_lin_vel = RewTerm(
        func=mdp.body_lin_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 2.0 * math.sqrt(4)}
    )
    body_ang_vel = RewTerm(
        func=mdp.body_ang_vel_reward,
        weight=1.0, #0.0,
        params={"command_name": "traj_ref",
                "sigma": 1.0 * math.sqrt(4)}
    )

    # Goal conditioned rewards
    xy_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity",
                "std": 0.75,}
    )

    yaw_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity",
                "std": 0.75,}
    )

    clf_reward = None


@configclass
class G1RunningCurriculumCfg:
    # contact_penalty_curriculum = CurrTerm(func=mdp.contact_curriculum,
    #                                       params={"update_interval": 40000, #20000,
    #                                                "max_weight": 1.0,
    #                                                "update_amnt": 0.1})

    # clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 30000, "min_max_err": (0.25, 0.3, 0.2) })
    pass

@configclass
class G1RunningEventsCfg(HumanoidEventsCfg):
    # Reset on the trajectory
    reset_on_ref = EventTerm(
        func=mdp.reset_on_reference,
        mode="reset",
        params={"command_name": "traj_ref",
                "base_frame_name": "pelvis_link",
                "conditioner_command_name": "base_velocity",
                "rel_envs_on_ref": 0.5}
    )

    # Randomize joint friction
    joint_friction_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "static_friction_distribution_params": (0.3, 1.6),
                "dynamic_friction_distribution_params": (0.3, 1.2),
                "viscous_friction_distribution_params": (0.01, 0.1),
                "operation": "add"},
    )

    # Randomize armature
    joint_armature_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.95, 1.05),
                "operation": "scale"},
    )

    # PD Gain randomization
    gain_randomization = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        },
    )

    # Adjust torso mass
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "mass_distribution_params": (0.85, 1.15), #(-5.0, 5.0),
            "operation": "scale",
        },
    )

    reset_base = None
    reset_robot_joints = None

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

        self.commands.base_velocity.ranges.lin_vel_x = (1.1, 3.7)  # Note the curriculum for increasing

        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.2, 0.2)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }

        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (0, 0)


        self.rewards.holonomic_constraint.params["command_name"] = "traj_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "traj_ref"

        # self.rewards.clf_reward.params = {
        #     "command_name": "traj_ref",
        #     "max_eta_err": 0.3,
        # }
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
        self.events.base_external_force_torque = None

        # Update the ground restitution range
        self.events.randomize_ground_contact_friction.params['restitution_range'] = (0.0, 0.2)

        # Update push forces
        self.events.push_robot.params['velocity_range'] = {"x": (-0.75, 0.75), "y": (-0.75, 0.75)}

        # Make the COM randomization on the torso rather than the pelvis
        self.events.base_com.params['asset_cfg'] = SceneEntityCfg("robot", body_names="waist_yaw_link")

        ##
        # Episode length
        ##
        self.episode_length_s = 20.0

@configclass
class G1RunningGaitLibraryEnvCfgPlay(G1RunningGaitLibraryEnvCfg):
    """Configuration for the G1 running gait library play environment."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (1.1, 3.7) #(1.1, 3.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.resampling_time_range=(4.0, 4.0) #(3.0, 4.0)
        self.commands.base_velocity.rel_y_envs = 1.0
        self.commands.base_velocity.debug_vis = False

        self.episode_length_s = 4.0 #6.0


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
