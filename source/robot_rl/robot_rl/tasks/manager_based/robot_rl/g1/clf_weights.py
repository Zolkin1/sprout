# Copyright 2026 Zachary Olkin. All rights reserved.

import copy

##
# Lyapunov Weights
##
WALKING_Q_weights = {}
WALKING_Q_weights["com:pos_x"] = [1.0, 1.0]
WALKING_Q_weights["com:pos_y"] = [1.0, 1.0]
WALKING_Q_weights["com:pos_z"] = [1.0, 1.0]

WALKING_Q_weights["left_ankle_roll_link:pos_x"] = [1.0, 1.0]
WALKING_Q_weights["left_ankle_roll_link:pos_y"] = [1.0, 1.0]
WALKING_Q_weights["left_ankle_roll_link:pos_z"] = [1.0, 1.0]
WALKING_Q_weights["left_ankle_roll_link:ori_x"] = [1.0, 1.0]
WALKING_Q_weights["left_ankle_roll_link:ori_y"] = [1.0, 1.0]
WALKING_Q_weights["left_ankle_roll_link:ori_z"] = [1.0, 1.0]

WALKING_Q_weights["right_ankle_roll_link:pos_x"] = [1.0, 1.0]
WALKING_Q_weights["right_ankle_roll_link:pos_y"] = [1.0, 1.0]
WALKING_Q_weights["right_ankle_roll_link:pos_z"] = [1.0, 1.0]
WALKING_Q_weights["right_ankle_roll_link:ori_x"] = [1.0, 1.0]
WALKING_Q_weights["right_ankle_roll_link:ori_y"] = [1.0, 1.0]
WALKING_Q_weights["right_ankle_roll_link:ori_z"] = [1.0, 1.0]

WALKING_Q_weights["joint:left_hip_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_hip_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_hip_yaw_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_knee_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_ankle_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_ankle_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_hip_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_hip_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_hip_yaw_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_knee_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_ankle_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_ankle_pitch_joint"] = [3.0, 0.1]

WALKING_Q_weights["joint:waist_yaw_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_elbow_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_shoulder_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_shoulder_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_shoulder_yaw_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_elbow_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_shoulder_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_shoulder_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_shoulder_yaw_joint"] = [3.0, 0.1]

WALKING_Q_weights["pelvis_link:pos_x"] = [1.0, 10.0]
WALKING_Q_weights["pelvis_link:pos_y"] = [1.0, 10.0]
WALKING_Q_weights["pelvis_link:pos_z"] = [1.0, 1.0]
WALKING_Q_weights["pelvis_link:ori_x"] = [1.0, 1.0]
WALKING_Q_weights["pelvis_link:ori_y"] = [1.0, 1.0]
WALKING_Q_weights["pelvis_link:ori_z"] = [1.0, 10.0]

WALKING_Q_weights["right_wrist_yaw_link:pos_x"] = [1.0, 1.0]
WALKING_Q_weights["right_wrist_yaw_link:pos_y"] = [1.0, 1.0]
WALKING_Q_weights["right_wrist_yaw_link:pos_z"] = [1.0, 1.0]
WALKING_Q_weights["right_wrist_yaw_link:ori_x"] = [1.0, 1.0]
WALKING_Q_weights["right_wrist_yaw_link:ori_y"] = [1.0, 1.0]
WALKING_Q_weights["right_wrist_yaw_link:ori_z"] = [1.0, 1.0]

WALKING_Q_weights["left_wrist_yaw_link:pos_x"] = [1.0, 1.0]
WALKING_Q_weights["left_wrist_yaw_link:pos_y"] = [1.0, 1.0]
WALKING_Q_weights["left_wrist_yaw_link:pos_z"] = [1.0, 1.0]
WALKING_Q_weights["left_wrist_yaw_link:ori_x"] = [1.0, 1.0]
WALKING_Q_weights["left_wrist_yaw_link:ori_y"] = [1.0, 1.0]
WALKING_Q_weights["left_wrist_yaw_link:ori_z"] = [1.0, 1.0]

WALKING_R_weights = {}
WALKING_R_weights["com:pos_x"] = [0.05]
WALKING_R_weights["com:pos_y"] = [0.05]
WALKING_R_weights["com:pos_z"] = [0.05]

WALKING_R_weights["left_ankle_roll_link:pos_x"] = [0.05]
WALKING_R_weights["left_ankle_roll_link:pos_y"] = [0.05]
WALKING_R_weights["left_ankle_roll_link:pos_z"] = [0.05]
WALKING_R_weights["left_ankle_roll_link:ori_x"] = [0.05]
WALKING_R_weights["left_ankle_roll_link:ori_y"] = [0.05]
WALKING_R_weights["left_ankle_roll_link:ori_z"] = [0.05]

WALKING_R_weights["right_ankle_roll_link:pos_x"] = [0.05]
WALKING_R_weights["right_ankle_roll_link:pos_y"] = [0.05]
WALKING_R_weights["right_ankle_roll_link:pos_z"] = [0.05]
WALKING_R_weights["right_ankle_roll_link:ori_x"] = [0.05]
WALKING_R_weights["right_ankle_roll_link:ori_y"] = [0.05]
WALKING_R_weights["right_ankle_roll_link:ori_z"] = [0.05]

WALKING_R_weights["joint:left_hip_roll_joint"] = [0.05]
WALKING_R_weights["joint:left_hip_pitch_joint"] = [0.05]
WALKING_R_weights["joint:left_hip_yaw_joint"] = [0.05]
WALKING_R_weights["joint:left_knee_joint"] = [0.05]
WALKING_R_weights["joint:left_ankle_roll_joint"] = [0.05]
WALKING_R_weights["joint:left_ankle_pitch_joint"] = [0.05]
WALKING_R_weights["joint:right_hip_roll_joint"] = [0.05]
WALKING_R_weights["joint:right_hip_pitch_joint"] = [0.05]
WALKING_R_weights["joint:right_hip_yaw_joint"] = [0.05]
WALKING_R_weights["joint:right_knee_joint"] = [0.05]
WALKING_R_weights["joint:right_ankle_roll_joint"] = [0.05]
WALKING_R_weights["joint:right_ankle_pitch_joint"] = [0.05]

WALKING_R_weights["pelvis_link:pos_x"] = [1.0]
WALKING_R_weights["pelvis_link:pos_y"] = [1.0]
WALKING_R_weights["pelvis_link:pos_z"] = [0.05]
WALKING_R_weights["pelvis_link:ori_x"] = [0.05]
WALKING_R_weights["pelvis_link:ori_y"] = [0.05]
WALKING_R_weights["pelvis_link:ori_z"] = [1.0]

WALKING_R_weights["joint:waist_yaw_joint"] = [0.05]
WALKING_R_weights["joint:left_elbow_joint"] = [0.05]
WALKING_R_weights["joint:left_shoulder_pitch_joint"] = [0.05]
WALKING_R_weights["joint:left_shoulder_roll_joint"] = [0.05]
WALKING_R_weights["joint:left_shoulder_yaw_joint"] = [0.05]
WALKING_R_weights["joint:right_elbow_joint"] = [0.05]
WALKING_R_weights["joint:right_shoulder_pitch_joint"] = [0.05]
WALKING_R_weights["joint:right_shoulder_roll_joint"] = [0.05]
WALKING_R_weights["joint:right_shoulder_yaw_joint"] = [0.05]

WALKING_R_weights["right_wrist_yaw_link:pos_x"] = [0.05]
WALKING_R_weights["right_wrist_yaw_link:pos_y"] = [0.05]
WALKING_R_weights["right_wrist_yaw_link:pos_z"] = [0.05]
WALKING_R_weights["right_wrist_yaw_link:ori_x"] = [0.05]
WALKING_R_weights["right_wrist_yaw_link:ori_y"] = [0.05]
WALKING_R_weights["right_wrist_yaw_link:ori_z"] = [0.05]

WALKING_R_weights["left_wrist_yaw_link:pos_x"] = [0.05]
WALKING_R_weights["left_wrist_yaw_link:pos_y"] = [0.05]
WALKING_R_weights["left_wrist_yaw_link:pos_z"] = [0.05]
WALKING_R_weights["left_wrist_yaw_link:ori_x"] = [0.05]
WALKING_R_weights["left_wrist_yaw_link:ori_y"] = [0.05]
WALKING_R_weights["left_wrist_yaw_link:ori_z"] = [0.05]


##
# Running: feet-dominant CLF weighting (ICRA-like).
##
# Bump the ankle_roll (foot) frames far above the rest so the CLF prioritizes
# foot Cartesian tracking (~10x the joint weight, echoing the ICRA feet:joints
# ratio) instead of the uniform WALKING weights, where the joints (3.0) actually
# outweighed the feet (1.0). Pairs with the holonomic stance-foot constraints.
# R is left unchanged (reuse WALKING_R_weights).
RUNNING_Q_weights = copy.deepcopy(WALKING_Q_weights)
for _name in list(RUNNING_Q_weights.keys()):
    if "ankle_roll_link:pos" in _name:
        RUNNING_Q_weights[_name] = [30.0, 3.0]
    elif "ankle_roll_link:ori" in _name:
        RUNNING_Q_weights[_name] = [15.0, 3.0]


# ##
# # Lyapunov Weights
# ##
# RUNNING_Q_weights = {}
# RUNNING_Q_weights["com:pos_x"] = [1.0, 1.0]
# RUNNING_Q_weights["com:pos_y"] = [1.0, 1.0]
# RUNNING_Q_weights["com:pos_z"] = [1.0, 1.0]
#
# RUNNING_Q_weights["left_ankle_roll_link:pos_x"] = [1.0, 1.0]
# RUNNING_Q_weights["left_ankle_roll_link:pos_y"] = [1.0, 1.0]
# RUNNING_Q_weights["left_ankle_roll_link:pos_z"] = [1.0, 1.0]
# RUNNING_Q_weights["left_ankle_roll_link:ori_x"] = [1.0, 1.0]
# RUNNING_Q_weights["left_ankle_roll_link:ori_y"] = [1.0, 1.0]
# RUNNING_Q_weights["left_ankle_roll_link:ori_z"] = [1.0, 1.0]
#
# RUNNING_Q_weights["right_ankle_roll_link:pos_x"] = [1.0, 1.0]
# RUNNING_Q_weights["right_ankle_roll_link:pos_y"] = [1.0, 1.0]
# RUNNING_Q_weights["right_ankle_roll_link:pos_z"] = [1.0, 1.0]
# RUNNING_Q_weights["right_ankle_roll_link:ori_x"] = [1.0, 1.0]
# RUNNING_Q_weights["right_ankle_roll_link:ori_y"] = [1.0, 1.0]
# RUNNING_Q_weights["right_ankle_roll_link:ori_z"] = [1.0, 1.0]
#
# if REWARD_TYPE == "CLF":
#     RUNNING_Q_weights["joint:left_hip_roll_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_hip_pitch_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_hip_yaw_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_knee_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_ankle_roll_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_ankle_pitch_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_hip_roll_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_hip_pitch_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_hip_yaw_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_knee_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_ankle_roll_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_ankle_pitch_joint"] = [3.0, 0.1]
#
#     RUNNING_Q_weights["joint:waist_yaw_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_elbow_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_shoulder_pitch_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_shoulder_roll_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:left_shoulder_yaw_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_elbow_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_shoulder_pitch_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_shoulder_roll_joint"] = [3.0, 0.1]
#     RUNNING_Q_weights["joint:right_shoulder_yaw_joint"] = [3.0, 0.1]
#
#     RUNNING_Q_weights["pelvis_link:pos_x"] = [1.0, 10.0]
#     RUNNING_Q_weights["pelvis_link:pos_y"] = [1.0, 10.0]
#     RUNNING_Q_weights["pelvis_link:pos_z"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:ori_x"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:ori_y"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:ori_z"] = [1.0, 10.0]
# else:
#     RUNNING_Q_weights["joint:left_hip_roll_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_hip_pitch_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_hip_yaw_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_knee_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_ankle_roll_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_ankle_pitch_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_hip_roll_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_hip_pitch_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_hip_yaw_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_knee_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_ankle_roll_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_ankle_pitch_joint"] = [1.0, 1.0]
#
#     RUNNING_Q_weights["joint:waist_yaw_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_elbow_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_shoulder_pitch_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_shoulder_roll_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:left_shoulder_yaw_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_elbow_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_shoulder_pitch_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_shoulder_roll_joint"] = [1.0, 1.0]
#     RUNNING_Q_weights["joint:right_shoulder_yaw_joint"] = [1.0, 1.0]
#
#     RUNNING_Q_weights["pelvis_link:pos_x"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:pos_y"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:pos_z"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:ori_x"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:ori_y"] = [1.0, 1.0]
#     RUNNING_Q_weights["pelvis_link:ori_z"] = [1.0, 1.0]
#
#
# RUNNING_Q_weights["right_wrist_yaw_link:pos_x"] = [1.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:pos_y"] = [1.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:pos_z"] = [1.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:ori_x"] = [1.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:ori_y"] = [1.0, 1.0]
# RUNNING_Q_weights["right_wrist_yaw_link:ori_z"] = [1.0, 1.0]
#
# RUNNING_Q_weights["left_wrist_yaw_link:pos_x"] = [1.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:pos_y"] = [1.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:pos_z"] = [1.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:ori_x"] = [1.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:ori_y"] = [1.0, 1.0]
# RUNNING_Q_weights["left_wrist_yaw_link:ori_z"] = [1.0, 1.0]
#
# RUNNING_R_weights = {}
# RUNNING_R_weights["com:pos_x"] = [0.05]
# RUNNING_R_weights["com:pos_y"] = [0.05]
# RUNNING_R_weights["com:pos_z"] = [0.05]
#
# RUNNING_R_weights["left_ankle_roll_link:pos_x"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:pos_y"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:pos_z"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:ori_x"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:ori_y"] = [0.05]
# RUNNING_R_weights["left_ankle_roll_link:ori_z"] = [0.05]
#
# RUNNING_R_weights["right_ankle_roll_link:pos_x"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:pos_y"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:pos_z"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:ori_x"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:ori_y"] = [0.05]
# RUNNING_R_weights["right_ankle_roll_link:ori_z"] = [0.05]
#
# RUNNING_R_weights["joint:left_hip_roll_joint"] = [0.05]
# RUNNING_R_weights["joint:left_hip_pitch_joint"] = [0.05]
# RUNNING_R_weights["joint:left_hip_yaw_joint"] = [0.05]
# RUNNING_R_weights["joint:left_knee_joint"] = [0.05]
# RUNNING_R_weights["joint:left_ankle_roll_joint"] = [0.05]
# RUNNING_R_weights["joint:left_ankle_pitch_joint"] = [0.05]
# RUNNING_R_weights["joint:right_hip_roll_joint"] = [0.05]
# RUNNING_R_weights["joint:right_hip_pitch_joint"] = [0.05]
# RUNNING_R_weights["joint:right_hip_yaw_joint"] = [0.05]
# RUNNING_R_weights["joint:right_knee_joint"] = [0.05]
# RUNNING_R_weights["joint:right_ankle_roll_joint"] = [0.05]
# RUNNING_R_weights["joint:right_ankle_pitch_joint"] = [0.05]
#
# if REWARD_TYPE == "CLF":
#     RUNNING_R_weights["pelvis_link:pos_x"] = [1.0]
#     RUNNING_R_weights["pelvis_link:pos_y"] = [1.0]
#     RUNNING_R_weights["pelvis_link:pos_z"] = [0.05]
#     RUNNING_R_weights["pelvis_link:ori_x"] = [0.05]
#     RUNNING_R_weights["pelvis_link:ori_y"] = [0.05]
#     RUNNING_R_weights["pelvis_link:ori_z"] = [1.0]
# else:
#     RUNNING_R_weights["pelvis_link:pos_x"] = [0.05]
#     RUNNING_R_weights["pelvis_link:pos_y"] = [0.05]
#     RUNNING_R_weights["pelvis_link:pos_z"] = [0.05]
#     RUNNING_R_weights["pelvis_link:ori_x"] = [0.05]
#     RUNNING_R_weights["pelvis_link:ori_y"] = [0.05]
#     RUNNING_R_weights["pelvis_link:ori_z"] = [0.05]
#
# RUNNING_R_weights["joint:waist_yaw_joint"] = [0.05]
# RUNNING_R_weights["joint:left_elbow_joint"] = [0.05]
# RUNNING_R_weights["joint:left_shoulder_pitch_joint"] = [0.05]
# RUNNING_R_weights["joint:left_shoulder_roll_joint"] = [0.05]
# RUNNING_R_weights["joint:left_shoulder_yaw_joint"] = [0.05]
# RUNNING_R_weights["joint:right_elbow_joint"] = [0.05]
# RUNNING_R_weights["joint:right_shoulder_pitch_joint"] = [0.05]
# RUNNING_R_weights["joint:right_shoulder_roll_joint"] = [0.05]
# RUNNING_R_weights["joint:right_shoulder_yaw_joint"] = [0.05]
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