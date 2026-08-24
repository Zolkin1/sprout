# Copyright 2026 Zachary Olkin. All rights reserved.

"""Look at a `user_pose` in MuJoCo before the robot holds it.

``user_pose`` is what the G1's driver snaps to in the FSM's USER_POSE state, before low-level
control starts. This just renders it.

**THE JOINT ORDER IS A THIRD ORDER, and that is the whole reason this is a script rather than
a paste.** The policy's is interleaved (``left_hip_pitch, right_hip_pitch, waist_yaw, ...``);
MuJoCo's is per-limb with the three waist joints CONSECUTIVE; the driver's appends the two
waist joints its 27-DoF interface did not have to the END::

    ... right_ankle_roll, waist_yaw, left_shoulder_pitch ... right_wrist_yaw, waist_roll, waist_pitch
                                  ^ 12                                        ^ 27        ^ 28

MuJoCo has ``waist_roll`` / ``waist_pitch`` at 13 and 14. So dropping a ``user_pose`` straight
into ``qpos[7:]`` shows a DIFFERENT pose -- arms shifted by two, waist wearing the wrists'
angles -- and it looks like a perfectly plausible robot. The permutation here is resolved BY
NAME from ``default_joint_names`` in the driver config itself, so it cannot drift from what
the robot will actually do.

Usage::

    python transfer/sim/view_user_pose.py                          # the config's current pose
    python transfer/sim/view_user_pose.py --pose-file candidate.yml
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import yaml

_HERE = pathlib.Path(__file__).resolve().parent
#: The driver config: the source of BOTH the pose and the joint order it is written in.
DRIVER_CFG = _HERE.parent / "obelisk" / "g1_control" / "config" / "g1_robot_hw.yml"
DEFAULT_SCENE = _HERE / "robots" / "g1" / "g1_perceptive_basic_scene.xml"


def load_driver_pose(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    """The driver's joint names and its current ``user_pose``.

    Read with a plain scan rather than a YAML load: the obelisk configs carry ``${VAR}``
    interpolation that a strict loader rejects, and all this needs is two flat lists.

    Args:
        path: ``g1_robot_hw.yml``.

    Returns:
        ``(joint_names, user_pose)``.

    Raises:
        SystemExit: If either list is missing, or their lengths disagree -- every array in
            that file is indexed by position in ``default_joint_names``.
    """
    text = path.read_text()

    def block(name: str) -> str:
        m = re.search(rf"^\s*{name}:\s*\[(.*?)\]", text, re.S | re.M)
        if not m:
            sys.exit(f"[ERROR] no '{name}:' list in {path}")
        return m.group(1)

    names = re.findall(r'"([a-z_]+_joint)"', block("default_joint_names"))
    raw = re.sub(r"#[^\n]*", "", block("user_pose"))
    pose = np.array([float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", raw)])
    if pose.size != len(names):
        sys.exit(f"[ERROR] {path.name}: user_pose has {pose.size} entries but "
                 f"default_joint_names has {len(names)}.")
    return names, pose


def to_mujoco_order(pose: np.ndarray, driver_names: list[str],
                    model: mujoco.MjModel) -> np.ndarray:
    """Permute a driver-ordered vector into the model's ``qpos[7:]`` order.

    BY NAME, never by position -- see the module docstring.

    Raises:
        SystemExit: If a driver joint has no hinge in the model, naming it.
    """
    out = np.zeros(model.nq - 7)
    for i, name in enumerate(driver_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            sys.exit(f"[ERROR] the driver commands {name!r} and the model has no such joint. "
                     "The two describe different robots.")
        out[model.jnt_qposadr[jid] - 7] = pose[i]
    return out


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose-file", default=None,
                   help="YAML/JSON holding 29 joint angles in the DRIVER's order (a bare list, "
                        "or a mapping with a 'user_pose' key). Default: the pose already in "
                        "g1_robot_hw.yml.")
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    args = p.parse_args()

    names, pose = load_driver_pose(DRIVER_CFG)
    source = DRIVER_CFG.name
    if args.pose_file:
        loaded = yaml.safe_load(open(args.pose_file))
        if isinstance(loaded, dict):
            loaded = loaded.get("user_pose", loaded)
        pose = np.asarray(loaded, dtype=float).ravel()
        source = args.pose_file
        if pose.size != len(names):
            sys.exit(f"[ERROR] {args.pose_file} has {pose.size} values; the driver commands "
                     f"{len(names)} joints.")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[7:] = to_mujoco_order(pose, names, model)
    mujoco.mj_forward(model, data)

    print(f"[pose] {source}: {pose.size} joints, permuted from the driver's order by name")
    print("[pose] close the viewer window to exit")
    # PASSIVE: nothing is stepped, so what is on screen is the pose as written. Re-forwarding
    # each sync keeps it there rather than letting anything drift.
    with mj_viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)
            v.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
