# Copyright 2026 Zachary Olkin. All rights reserved.

"""The ribbon's FK moved from MuJoCo to pinocchio; this is the claim that let it.

``deploy.fk`` FKs the ROS URDF, but the generator's own FK consistency loss is defined against
``FlowMatchingConfig.fk_model_path`` -- the MJCF at ``transfer/sim/robots/g1/g1_29j.xml``. The
standing objection to using anything else is that the drawn feet would become a SECOND OPINION
about where the feet are, and a display whose purpose is to expose a discrepancy must not be
able to manufacture one.

This measures the objection instead of arguing about it: the two models, the four frames the
ribbon actually draws, random configurations. MuJoCo appears ONLY here, as a test-only oracle,
which is the same arrangement ``transfer/torch_math`` has with the numpy ports.

If this ever fails, the models have genuinely diverged and the ribbon is drawing something the
generator does not agree with -- fix the models, do not raise the tolerance.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
mujoco = pytest.importorskip("mujoco")

from deploy.fk import DEFAULT_FK_MODEL, WindowFK  # noqa: E402

#: The MJCF the generator's FK loss is defined against. Off this file's package root rather
#: than a count of ".." hops.
_MJCF = str(
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "sim" / "robots" / "g1" / "g1_29j.xml"
)

#: What the tracker's `ee_frame_names` holds, and therefore what the ribbon draws.
_FRAMES = [
    "left_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_ankle_roll_link",
    "right_wrist_yaw_link",
]

#: Measured worst case over 200 random configurations is 4.6e-7 m, at the wrists -- the end of
#: the longest chain, where the MJCF's rounded literals have accumulated the most. The bound is
#: an order above that and still 300x below a marker anyone could see.
_TOL_M = 5e-6


def _mjcf_frame_positions(joint_pos_isaac: np.ndarray, isaac_names: list[str]) -> np.ndarray:
    """The oracle: the same frames off the MJCF, relative to the root body.

    Args:
        joint_pos_isaac: ``[N, nj]`` angles in the policy's joint order.
        isaac_names: That order.

    Returns:
        ``[N, len(_FRAMES), 3]`` positions in the root frame.
    """
    model = mujoco.MjModel.from_xml_path(_MJCF)
    data = mujoco.MjData(model)
    # By qpos ADDRESS, never by joint id order -- the two are not the same map, and assuming
    # they were put the feet 1.4 m out while every shape stayed right.
    adr = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        adr[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = int(model.jnt_qposadr[j])
    # The MJCF calls the root body `pelvis_link` and the URDF calls it `pelvis`; same body.
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    assert root >= 0, "the MJCF's root body is not pelvis_link any more"
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in _FRAMES]
    assert all(i >= 0 for i in ids), f"MJCF is missing one of {_FRAMES}"

    out = np.zeros((joint_pos_isaac.shape[0], len(_FRAMES), 3))
    for row in range(joint_pos_isaac.shape[0]):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0            # identity base quaternion (wxyz), free joint at qpos[0:7]
        for name, angle in zip(isaac_names, joint_pos_isaac[row]):
            data.qpos[adr[name]] = angle
        mujoco.mj_kinematics(model, data)
        rot = data.xmat[root].reshape(3, 3)
        for f, bid in enumerate(ids):
            out[row, f] = rot.T @ (data.xpos[bid] - data.xpos[root])
    return out


def _isaac_joint_names() -> list[str]:
    """The policy's joint order, taken from the URDF so the test needs no artifact."""
    model = pin.buildModelFromUrdf(DEFAULT_FK_MODEL)
    return [model.names[i] for i in range(1, model.njoints)]


def test_the_urdf_and_the_mjcf_place_the_ribbon_frames_identically():
    """Over random postures, the two models are the same kinematics for these frames."""
    names = _isaac_joint_names()
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.6, 0.6, size=(64, len(names)))

    fk = WindowFK(DEFAULT_FK_MODEL, _FRAMES, names)
    assert not fk.missing, f"URDF is missing {fk.missing}"
    # Identity base pose, so the world positions ARE the root-frame ones the oracle returns.
    got = fk.frame_positions(
        np.zeros((q.shape[0], 3)),
        np.tile([0.0, 0.0, 0.0, 1.0], (q.shape[0], 1)),
        q,
    )
    want = _mjcf_frame_positions(q, names)
    worst = float(np.abs(got - want).max())
    assert worst <= _TOL_M, f"URDF and MJCF disagree by {worst:.3e} m on {_FRAMES}"


def test_the_base_pose_carries_the_frames_into_the_world():
    """The other half: FK is relative, and the predicted base pose is what places it.

    A fixed-base model gets this from the caller rather than from a free joint, so it is a
    line of arithmetic that could be dropped without any shape changing.
    """
    names = _isaac_joint_names()
    fk = WindowFK(DEFAULT_FK_MODEL, _FRAMES, names)
    q = np.zeros((1, len(names)))
    at_origin = fk.frame_positions(np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), q)
    # A pure translation moves every frame by exactly it.
    shifted = fk.frame_positions(np.array([[1.0, -2.0, 0.5]]), np.array([[0, 0, 0, 1.0]]), q)
    assert np.allclose(shifted - at_origin, [1.0, -2.0, 0.5])
    # A yaw of 180 degrees negates x and y and leaves z.
    turned = fk.frame_positions(
        np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0, 0.0]]), q
    )
    assert np.allclose(turned[..., 0], -at_origin[..., 0], atol=1e-12)
    assert np.allclose(turned[..., 1], -at_origin[..., 1], atol=1e-12)
    assert np.allclose(turned[..., 2], at_origin[..., 2], atol=1e-12)


def test_a_frame_the_model_does_not_have_is_NaN_not_an_exception():
    """One renamed frame must not cost the whole drawing."""
    names = _isaac_joint_names()
    fk = WindowFK(DEFAULT_FK_MODEL, ["left_ankle_roll_link", "not_a_link"], names)
    assert fk.missing == ["not_a_link"]
    out = fk.frame_positions(
        np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, len(names)))
    )
    assert np.isfinite(out[0, 0]).all()
    assert np.isnan(out[0, 1]).all()
