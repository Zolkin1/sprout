# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-channel CLF weights.

``CLF.__init__`` indexes these dicts by channel name, so a channel the command tracks but the
dict does not name is a ``KeyError`` at env construction.  The world-frame + joint channels are
written out explicitly below; the root-frame body channels are filled programmatically by
:func:`add_root_frame_weights` (13 bodies x 6 channels would otherwise be 156 near-identical
lines to hand-maintain).
"""

from motion_gen.joint_names import G1_BODY_TRACK_FRAMES, root_frame_name

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

# Newly-unlocked DOFs (waist roll/pitch + the 6 wrists): tracked like the other arm/waist joints.
WALKING_Q_weights["joint:waist_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:waist_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_wrist_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_wrist_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:left_wrist_yaw_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_wrist_roll_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_wrist_pitch_joint"] = [3.0, 0.1]
WALKING_Q_weights["joint:right_wrist_yaw_joint"] = [3.0, 0.1]

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

# Newly-unlocked DOFs (waist roll/pitch + the 6 wrists): same control weight as the other arm/waist joints.
WALKING_R_weights["joint:waist_roll_joint"] = [0.05]
WALKING_R_weights["joint:waist_pitch_joint"] = [0.05]
WALKING_R_weights["joint:left_wrist_roll_joint"] = [0.05]
WALKING_R_weights["joint:left_wrist_pitch_joint"] = [0.05]
WALKING_R_weights["joint:left_wrist_yaw_joint"] = [0.05]
WALKING_R_weights["joint:right_wrist_roll_joint"] = [0.05]
WALKING_R_weights["joint:right_wrist_pitch_joint"] = [0.05]
WALKING_R_weights["joint:right_wrist_yaw_joint"] = [0.05]

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
# Root-frame body-link weights (``tracking_mode="body"``)
##

# ``[q_pose, q_twist]`` per channel: for a ``pos_`` channel that is (position, linear
# velocity); for an ``ori_`` channel it is (orientation, angular velocity).
#
# Deliberately uniform ACROSS LINKS -- the point of the joint-vs-body A/B is the
# parameterization, so a hand-tuned per-link profile would confound it.  But pose and twist
# are NOT treated alike: measured on a rollout with the twist weight at 1.0, the unweighted
# error energy split was
#
#     root_body_ang_vel 150.97 | root_body_lin_vel 10.51 | root_body_ori 5.38 | root_body_pos 0.30
#
# i.e. link angular velocity outweighed link POSITION by ~500x, so "body-link tracking"
# was in practice "match the links' angular velocities".  Velocities are inherently larger
# and noisier than the positions they differentiate, so the twist weight is 0.1 -- a 10x
# cut -- to put pose back in charge of the reward.
_ROOT_Q_DEFAULT = [1.0, 0.1]
_ROOT_R_DEFAULT = [0.05]


def add_root_frame_weights(
    q_weights: dict,
    r_weights: dict,
    bodies: list[str] | None = None,
    q_overrides: dict | None = None,
    r_overrides: dict | None = None,
) -> None:
    """Fill in Q/R entries for every root-frame body channel, in place.

    Args:
        q_weights: Q dict to extend (channel name -> ``[q_pos, q_vel]``).
        r_weights: R dict to extend (channel name -> ``[r]``).
        bodies: Body names to add channels for; defaults to
            :data:`~motion_gen.joint_names.G1_BODY_TRACK_FRAMES`.
        q_overrides: Per-BODY Q override, e.g. ``{"torso_link": [3.0, 1.0]}``.
        r_overrides: Per-BODY R override.
    """
    bodies = list(G1_BODY_TRACK_FRAMES if bodies is None else bodies)
    q_overrides = q_overrides or {}
    r_overrides = r_overrides or {}
    for body in bodies:
        rf = root_frame_name(body)
        q = list(q_overrides.get(body, _ROOT_Q_DEFAULT))
        r = list(r_overrides.get(body, _ROOT_R_DEFAULT))
        for kind in ("pos", "ori"):
            for axis in ("x", "y", "z"):
                q_weights[f"{rf}:{kind}_{axis}"] = list(q)
                r_weights[f"{rf}:{kind}_{axis}"] = list(r)


add_root_frame_weights(WALKING_Q_weights, WALKING_R_weights)


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