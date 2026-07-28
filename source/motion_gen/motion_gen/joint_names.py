# Copyright 2026 Zachary Olkin. All rights reserved.

"""The canonical G1 joint order.

Lives on its own because it is the one piece of the (deprecated) motion-matching feature
code that the LIVE motionbricks ingest needs: ``motionbricks.joint_map`` maps a raw Motion
Bricks joint vector onto this order by NAME, and ``motionbricks.raw_schema`` documents the
contract against it.
"""

# Canonical G1 29-joint order (matches the MJCF joint names + the FM/clip joint set).  This is the
# FULL Unitree G1 body-DOF order -- identical to ``ingest.bones_ingest.CSV_JOINT_NAMES`` so the
# seed-CSV / clean / terrain ingests map straight through (their 29->this reindex is the identity).
# Was the 21-DOF subset (waist_yaw + shoulders/elbow only); widened to unlock waist roll/pitch +
# the 6 wrists.
G1_JOINT_NAMES: list[str] = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
