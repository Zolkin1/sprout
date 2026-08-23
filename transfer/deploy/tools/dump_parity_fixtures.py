# Copyright 2026 Zachary Olkin. All rights reserved.

"""Emit the cross-language parity fixtures the C++ deploy port is checked against.

The python runtime stays the oracle (see ``transfer/deploy/``); this writes down what it
computes so a C++ build with no python in it can be held to the same numbers.

Format is deliberately dumb: one raw little-endian ``float64`` blob per array plus a JSON
manifest. NOT ``.npz`` -- reading that from C++ needs a zip reader and an npy parser, which
is real code that can itself be wrong, in the one place where being wrong looks like a
passing test. A blob is an ``ifstream`` and a ``resize``.

The manifest records the shape, dtype and sha256 of every blob, plus the numpy version,
platform and seed that produced them, so a stale or half-regenerated fixture set is
detected rather than silently compared against.

Both target architectures are little-endian; that is asserted here rather than assumed.

Run (from the repo root, in ``env_transfer`` or ``env_isaaclab``)::

    python transfer/deploy/tools/dump_parity_fixtures.py
    python transfer/deploy/tools/dump_parity_fixtures.py --out /tmp/fixtures --count 2048

The primitive fixtures this writes are small (~1 MB) and are meant to be committed. The
episode-scale fixtures are generated on demand from a recorded bag and are not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from typing import Any, Callable

import numpy as np

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy import depth_obs as do  # noqa: E402
from deploy.fixture_io import MANIFEST_FORMAT, FixtureWriter  # noqa: E402
from deploy import quat_math_np as qn  # noqa: E402
from deploy import window_np as wn  # noqa: E402


#: Absolute tolerance per function, and how to compare. Lives HERE, travels in the
#: manifest, and is read by the C++ side -- one number per function in one place, so the
#: two languages cannot drift on what "agrees" means.
#:
#: The values are not uniform because the disagreement mechanisms are not:
#:
#: * 1e-14 -- pure algebra, four-term sums at most, identical operation order. The C++ is
#:   transcribed expression for expression from the numpy and built with -ffp-contract=off,
#:   so there is nothing left to differ except the last bit.
#: * 1e-13 -- one sqrt (correctly rounded in both) plus a length-3 dot: Gram-Schmidt.
#: * 1e-12 -- anything through atan2/sin/cos/asin. numpy may route these through SVML on
#:   x86 while calling libm on aarch64, so this is the one row whose slack is really about
#:   the host, not the algorithm.
#: * 1e-9 with `quat_rotation` -- quat_from_matrix picks among four candidates by argmax.
#:   Near a pivot tie the two languages can select different candidates; they are the same
#:   ROTATION to ~1e-8 but differ in sign pattern, so comparing componentwise is simply the
#:   wrong question. test_deploy_parity.py already sets this precedent (atol=1e-8, "asR").
#: * 1e-11 -- composites of the above evaluated at robot scale (metres, m/s).
TOLERANCES: dict[str, tuple[float, str]] = {
    "quat_mul": (1e-14, "elementwise"),
    "quat_conjugate": (1e-14, "elementwise"),
    "quat_inv": (1e-14, "elementwise"),
    "quat_apply": (1e-14, "elementwise"),
    "quat_apply_inverse": (1e-14, "elementwise"),
    "quat_unique": (1e-14, "elementwise"),
    "matrix_from_quat": (1e-14, "elementwise"),
    "matrix_to_rot6d": (1e-14, "elementwise"),
    "quat_to_rot6d": (1e-14, "elementwise"),
    "rot6d_to_matrix": (1e-13, "elementwise"),
    "combine_frame_transforms": (1e-13, "elementwise"),
    "subtract_frame_transforms": (1e-13, "elementwise"),
    "yaw_quat": (1e-12, "elementwise"),
    "axis_angle_from_quat": (1e-12, "elementwise"),
    "quat_from_axis_angle": (1e-12, "elementwise"),
    "quat_box_plus": (1e-12, "elementwise"),
    "quat_box_minus": (1e-12, "elementwise"),
    "quat_from_matrix": (1e-9, "quat_rotation"),
    "rot6d_to_quat": (1e-9, "quat_rotation"),
    "world_to_window": (1e-11, "elementwise"),
    "reanchor_window": (1e-11, "elementwise"),
    # EXACT. No reductions, no transcendentals -- just comparisons and one affine -- so a
    # single ulp of disagreement would be a real behavioural difference, not arithmetic
    # noise. test_depth_obs.py already holds the python to exact equality against its torch
    # original; this inherits that rather than loosening it.
    "normalize_depth": (0.0, "elementwise"),
    "sanitize_extero": (0.0, "elementwise"),
    # The 5-channel exteroception row. NOT exact, unlike the two above: these DO have
    # transcendentals (atan-free, but a normalise and a division by the ray cosine) and a
    # 3x3 matrix product, so an ulp of difference is arithmetic rather than behaviour.
    # Still tight -- the quantity is metres of terrain position, and 1e-12 m is far below
    # anything the sensor or the model can distinguish.
    "pinhole_ray_directions": (1e-14, "elementwise"),
    "invalid_depth_mask": (0.0, "elementwise"),   # a 0/1 plane: exact or wrong
    "unproject_to_yaw_frame": (1e-12, "elementwise"),
    # FK through a kinematic chain, so a few ulps accumulate per joint. 1e-11 m is ~10 pm --
    # eleven orders below anything the cameras or the model can distinguish, and still tight
    # enough that a wrong joint permutation or a scalar-first quaternion fails by kilometres.
    "camera_pose_yaw_frame": (1e-11, "elementwise"),
}


class _ParityWriter(FixtureWriter):
    """A :class:`FixtureWriter` that takes each case's tolerance from :data:`TOLERANCES`.

    The table stays in this file, where the REASON for each number is written down, while the
    format lives in ``deploy.fixture_io`` with the C++ reader on the other side of it. The
    KeyError on an unknown ``fn`` is deliberate and preserved: a new comparison must be given
    an explicit tolerance rather than inheriting a default nobody chose.
    """

    def add(self, case: str, fn: str, inputs: dict[str, Any], outputs: dict[str, Any],
            attrs: dict[str, Any] | None = None) -> None:
        """Record one case, looking its tolerance up by ``fn``."""
        atol, compare = TOLERANCES[fn]
        super().add(case, fn, inputs, outputs, atol=atol, compare=compare, attrs=attrs)


#: The repo root, off this file's location rather than a count of ".." hops elsewhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          os.pardir, os.pardir, os.pardir))

#: The POLICY's joint order (interleaved), which is what a joint_pos block is in. Pinocchio's
#: own order is per limb and differs from this on 27 of the 29, so the permutation is the
#: thing the camera-pose fixtures exist to pin.
_ISAAC_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

#: The shipped pair's mounts and NOMINAL offsets, as tracker.onnx publishes them. Two
#: different lenses on one mount body -- see the depth fixtures for why that matters.
#:
#: These are the AS-BUILT CAD poses (the left imager of each stereo pair, hence the non-zero y
#: -- half of a 50 mm and a 120 mm baseline), and they must track
#: ``g1_depth_env_cfg._NEAR`` / ``_FAR``, which
#: ``source/robot_rl/tests/test_camera_mount_matches_urdf.py`` pins against the URDF. They
#: previously held the PLACEHOLDER pair -- near at z = 0.0994 and far at z = 0.38 pitched 35
#: deg -- and because the C++ test restated the same wrong numbers, the camera-FK parity case
#: passed while describing a robot and a mount pair that ship nowhere. The offsets now travel
#: in the fixture's own attrs so the two languages cannot disagree about them again.
_CAMERAS: tuple[dict, ...] = (
    {"mount_body": "torso_link", "offset_pos": [0.107, 0.025, 0.045],
     "offset_quat_xyzw": [0.0, 0.5372996083468239, 0.0, 0.8433914458128857]},
    {"mount_body": "torso_link", "offset_pos": [0.085, 0.06, 0.285],
     "offset_quat_xyzw": [0.0, 0.13052619222005157, 0.0, 0.9914448613738104]},
)


def _mat_from_quat_xyzw(q) -> np.ndarray:
    """Rotation matrix from a scalar-LAST quaternion."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Scalar-LAST quaternion for an intrinsic Z-Y-X rotation."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def _roll_pitch_matrix(quat_xyzw) -> np.ndarray:
    """The root's tilt with its HEADING divided out: R(yaw_quat(q)^-1 * q).

    This is what carries a point from the root frame into the yaw-aligned one, and building it
    from the orientation rather than from a yaw estimate is what makes the camera pose
    independent of odometry.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    yaw = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    q_yaw_inv = np.array([0.0, 0.0, -np.sin(yaw / 2), np.cos(yaw / 2)])
    return _mat_from_quat_xyzw(q_yaw_inv) @ _mat_from_quat_xyzw(q)



def _rand_quat(rng: np.random.Generator, n: int) -> np.ndarray:
    """``[n, 4]`` uniformly random unit quaternions (xyzw)."""
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _adversarial_quats() -> np.ndarray:
    """Quaternions that break naive implementations, as ``[m, 4]`` (xyzw).

    Covers the cases a uniform random draw essentially never produces: the identity and its
    negation (``quat_unique`` / ``w >= 0`` canonicalisation), rotations by almost exactly pi
    about each axis (where ``quat_from_matrix``'s four-candidate argmax changes pivot and
    ``axis_angle_from_quat``'s small-angle branch is furthest from its expansion), and
    near-identity rotations (where that same branch is closest).
    """
    eps = 1e-8
    out = [
        [0.0, 0.0, 0.0, 1.0],                       # identity
        [0.0, 0.0, 0.0, -1.0],                      # identity, negated
        [1.0, 0.0, 0.0, 0.0],                       # pi about x
        [0.0, 1.0, 0.0, 0.0],                       # pi about y
        [0.0, 0.0, 1.0, 0.0],                       # pi about z
        [eps, 0.0, 0.0, 1.0],                       # near identity
        [0.0, eps, 0.0, 1.0],
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],     # pi about the xy diagonal: pivot tie
        [np.sqrt(1.0 / 3), np.sqrt(1.0 / 3), np.sqrt(1.0 / 3), 0.0],  # three-way tie
        [1.0 - 1e-12, 0.0, 0.0, np.sqrt(2e-12)],    # just short of pi
    ]
    arr = np.asarray(out, dtype=np.float64)
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


def dump_quat_math(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit fixtures for every ``quat_math_np`` entry point the C++ port implements."""
    q_rand = _rand_quat(rng, n)
    q_adv = _adversarial_quats()
    q = np.concatenate([q_rand, q_adv], axis=0)
    p = np.concatenate([_rand_quat(rng, n), q_adv], axis=0)
    # Realistic magnitudes: metres for positions, m/s and rad/s for velocities. Parity at
    # 1e-14 on unit-scale inputs can hide an error that only shows at robot scale.
    v = rng.standard_normal((q.shape[0], 3)) * 2.0
    t = rng.standard_normal((q.shape[0], 3)) * 5.0

    unary: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "quat_conjugate": qn.quat_conjugate,
        "quat_inv": qn.quat_inv,
        "yaw_quat": qn.yaw_quat,
        "quat_unique": qn.quat_unique,
        "matrix_from_quat": qn.matrix_from_quat,
        "axis_angle_from_quat": qn.axis_angle_from_quat,
    }
    for name, fn in unary.items():
        w.add(name, name, {"q": q}, {"result": fn(q)})

    w.add("quat_mul", "quat_mul", {"a": q, "b": p}, {"result": qn.quat_mul(q, p)})
    w.add("quat_apply", "quat_apply", {"q": q, "v": v}, {"result": qn.quat_apply(q, v)})
    w.add("quat_apply_inverse", "quat_apply_inverse", {"q": q, "v": v},
          {"result": qn.quat_apply_inverse(q, v)})
    w.add("quat_box_minus", "quat_box_minus", {"a": q, "b": p},
          {"result": qn.quat_box_minus(q, p)})
    w.add("quat_box_plus", "quat_box_plus", {"q": q, "delta": v},
          {"result": qn.quat_box_plus(q, v)})
    w.add("quat_from_axis_angle", "quat_from_axis_angle", {"delta": v},
          {"result": qn.quat_from_axis_angle(v)})

    # quat_from_matrix is fed matrices built from the SAME quaternions, so the expected
    # answer is a known rotation. Compared as a rotation, not componentwise -- near a pivot
    # tie the two languages can pick different candidates of the four, which are the same
    # rotation to ~1e-8 but differ in sign pattern.
    w.add("quat_from_matrix", "quat_from_matrix", {"matrix": qn.matrix_from_quat(q)},
          {"result": qn.quat_from_matrix(qn.matrix_from_quat(q))})

    p01, q01, p12, q12 = t, q, v, p
    cp, cq = qn.combine_frame_transforms(p01, q01, p12, q12)
    w.add("combine_frame_transforms", "combine_frame_transforms",
          {"p01": p01, "q01": q01, "p12": p12, "q12": q12}, {"pos": cp, "quat": cq})
    sp, sq = qn.subtract_frame_transforms(p01, q01, t + v, p)
    w.add("subtract_frame_transforms", "subtract_frame_transforms",
          {"p01": p01, "q01": q01, "p02": t + v, "q02": p}, {"pos": sp, "quat": sq})


def dump_window(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit fixtures for the rot6d helpers, ``world_to_window`` and ``reanchor_window``."""
    q = np.concatenate([_rand_quat(rng, n), _adversarial_quats()], axis=0)
    m = qn.matrix_from_quat(q)
    w.add("matrix_to_rot6d", "matrix_to_rot6d", {"matrix": m}, {"result": wn.matrix_to_rot6d(m)})
    w.add("quat_to_rot6d", "quat_to_rot6d", {"q": q}, {"result": wn.quat_to_rot6d(q)})

    # rot6d_to_matrix must survive raw generator output, which is only APPROXIMATELY a
    # rotation -- that is the whole reason it Gram-Schmidts. So perturb, and include a
    # zero row to exercise the 1e-8 norm guard that stops it dividing by zero.
    d6 = wn.matrix_to_rot6d(m) + rng.standard_normal((m.shape[0], 6)) * 0.05
    d6 = np.concatenate([d6, np.zeros((1, 6))], axis=0)
    w.add("rot6d_to_matrix", "rot6d_to_matrix", {"d6": d6}, {"result": wn.rot6d_to_matrix(d6)})
    w.add("rot6d_to_quat", "rot6d_to_quat", {"d6": d6}, {"result": wn.rot6d_to_quat(d6)})

    # The live artifact's layout: 29 joints, no ee frames, base_pose + base_vel + joint_pos
    # (D = 44). Also a full five-block case, so a future policy that carries joint_vel and
    # ee_pos is covered before it exists rather than after it breaks.
    for tag, blocks, nj, nee in (
        ("slim", ("base_pose", "base_vel", "joint_pos"), 29, 0),
        ("full", None, 21, 4),
    ):
        k = 13                                   # the artifact serves 13 window offsets
        root_pos = rng.standard_normal((1, 3)) * 3.0
        root_quat = _rand_quat(rng, 1)
        ref_pos = rng.standard_normal((k, 3)) * 3.0
        ref_quat = _rand_quat(rng, k)
        ref_lin = rng.standard_normal((k, 3))
        ref_ang = rng.standard_normal((k, 3))
        joint_pos = rng.standard_normal((k, nj))
        joint_vel = rng.standard_normal((k, nj))
        ee_pos = rng.standard_normal((k, max(nee, 1), 3))
        window = wn.world_to_window(root_pos, root_quat, ref_pos, ref_quat, ref_lin,
                                    ref_ang, joint_pos, joint_vel, ee_pos, blocks)
        w.add(f"world_to_window_{tag}", "world_to_window", {
            "root_pos_w": root_pos, "root_quat_w": root_quat,
            "ref_pos_w": ref_pos, "ref_quat_w": ref_quat,
            "ref_lin_w": ref_lin, "ref_ang_w": ref_ang,
            "joint_pos": joint_pos, "joint_vel": joint_vel, "ee_pos_w": ee_pos,
        }, {"window": window}, attrs={
            "blocks": list(wn.resolve_reference_blocks(blocks)),
            "num_joints": nj, "num_ee_frames": nee, "num_nodes": k,
        })

        # Re-anchor onto a pose a plausible K-1 steps of egomotion away. Also emit the
        # same-anchor case: reanchor onto the anchor it is already in must be the identity
        # to ~2e-15, which is the cheapest possible check that the round trip is not
        # accumulating error.
        new_pos = root_pos + rng.standard_normal((1, 3)) * 0.2
        new_quat = _rand_quat(rng, 1)
        w.add(f"reanchor_window_{tag}", "reanchor_window", {
            "window": window,
            "old_pos_w": root_pos, "old_quat_w": root_quat,
            "new_pos_w": new_pos, "new_quat_w": new_quat,
        }, {"window": wn.reanchor_window(window, root_pos, root_quat, new_pos, new_quat,
                                         nj, nee, blocks)},
            attrs={"blocks": list(wn.resolve_reference_blocks(blocks)),
                   "num_joints": nj, "num_ee_frames": nee})
        w.add(f"reanchor_window_identity_{tag}", "reanchor_window", {
            "window": window,
            "old_pos_w": root_pos, "old_quat_w": root_quat,
            "new_pos_w": root_pos, "new_quat_w": root_quat,
        }, {"window": wn.reanchor_window(window, root_pos, root_quat, root_pos, root_quat,
                                         nj, nee, blocks)},
            attrs={"blocks": list(wn.resolve_reference_blocks(blocks)),
                   "num_joints": nj, "num_ee_frames": nee})


def dump_depth(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit depth-normalization and exteroception-sanitising fixtures.

    Weighted towards the values that distinguish a correct port from a plausible one. The
    ordering inside ``normalize_depth_np`` is the whole game: the too-near test runs first,
    so ``-inf`` becomes ``min_clip_val``, while ``+inf`` and ``NaN`` fall through to the
    non-finite test and become ``max_clip_val``. An implementation that reorders those two
    lines, or that uses non-strict comparisons, passes on random inputs and fails only on
    the readings a real camera actually produces at the edge of its range.
    """
    cam = {"offset": 0.75, "scale": 0.5, "min_dist": 0.2, "max_dist": 9.0,
           "min_clip_val": 0.0, "max_clip_val": 0.0}
    edge = np.array([
        -np.inf, np.inf, np.nan,
        cam["min_dist"], cam["max_dist"],                       # exactly at: still VALID
        np.nextafter(cam["min_dist"], -np.inf),                 # one ulp below: invalid
        np.nextafter(cam["max_dist"], np.inf),                  # one ulp above: invalid
        0.0, -1.0, 1e-30, 1e30,
        cam["offset"],                                          # normalises to exactly 0
    ], dtype=np.float64)
    d = np.concatenate([rng.uniform(-0.5, 10.0, size=n), edge])[:, None]
    w.add("normalize_depth", "normalize_depth", {"d": d},
          {"result": mdp_normalize(d, cam)}, attrs=dict(cam))

    # A second camera contract with non-zero clip substitutions, so a port that hardcodes
    # the 0.0 defaults is caught. Those values are what a real spec carries when "no
    # reading" has to be distinguishable from a legitimate depth of `offset` metres.
    cam2 = {"offset": 1.0, "scale": 0.4, "min_dist": 0.3, "max_dist": 6.0,
            "min_clip_val": 0.25, "max_clip_val": 7.5}
    w.add("normalize_depth_nonzero_clips", "normalize_depth", {"d": d},
          {"result": mdp_normalize(d, cam2)}, attrs=dict(cam2))

    for source in ("height_scan", "depth"):
        clip = do.EXTERO_CLIP_NP[source]
        raw = np.concatenate([
            rng.standard_normal(n) * clip * 1.5,
            np.array([np.nan, np.inf, -np.inf, clip, -clip, 0.0], dtype=np.float64),
        ])[:, None]
        w.add(f"sanitize_extero_{source}", "sanitize_extero", {"values": raw},
              {"result": do.sanitize_extero_np(raw, source)},
              attrs={"source": source, "clip": clip})


def dump_depth_xyz(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit the fixtures for the 5-channel depth row's geometry.

    The three numerical pieces the C++ has to reproduce before it can assemble a row at all:
    where each pixel's ray points, which pixels carry no reading, and where a reading lands
    in the root's yaw-aligned frame. The row ASSEMBLY on top of them (channel-major order,
    the depth plane passed through untouched) is a layout property rather than a numerical
    one, and is tested directly in C++ -- the same split the python side uses between
    test_depth_unproject.py and test_depth_xyz_rows.py.

    A small 4x5 grid on purpose: these blobs are committed, and the geometry is per-pixel, so
    a bigger image would multiply the fixture size without testing anything new. The camera
    is pitched steeply DOWN like the real pair -- with a horizontal boresight every ray would
    have a near-identical cosine, which is exactly the factor a wrong z-depth-to-ray-length
    conversion gets wrong.

    Args:
        w: The fixture writer.
        rng: Source of randomness.
        n: Number of independent cases for the pose-dependent functions.
    """
    h, wid = 4, 5
    cam = {"width": wid, "height": h, "fx": 2.5, "fy": 2.2, "cx": wid / 2.0, "cy": h / 2.0,
           "offset": 0.75, "scale": 0.5, "min_dist": 0.1, "max_dist": 7.9,
           "min_clip_val": 0.0, "max_clip_val": 0.0}
    k = h * wid

    dirs = do.pinhole_ray_directions(wid, h, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    w.add("pinhole_ray_directions", "pinhole_ray_directions", {},
          {"dirs": dirs.reshape(1, -1)}, attrs=dict(cam))

    # Images that carry both real readings and every class of miss, since validity is found
    # by RANGE and the unprojection places a miss at max_dist of RAY LENGTH rather than of
    # z-depth. Random depths in metres, then normalised through the contract.
    metres = rng.uniform(0.05, 9.0, size=(n, k))
    metres[:, ::5] = np.inf                                   # rays that hit nothing
    metres[:, 1::7] = 0.02                                    # nearer than min_dist
    img = np.stack([mdp_normalize(metres[i], cam) for i in range(n)])

    w.add("invalid_depth_mask", "invalid_depth_mask", {"image": img},
          {"mask": do.invalid_depth_mask(cam, img.reshape(-1)).reshape(n, k).astype(np.float64)},
          attrs=dict(cam))

    # One pose per case: a camera forward of and above the root, pitched down, with the
    # rotation carried as a flat 3x3 so the C++ reads it without a quaternion convention in
    # the middle (which is its own parity question, covered by quat_math).
    pos = rng.uniform(-0.4, 0.6, size=(n, 3))
    rot = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(n)])
    rot *= np.sign(np.linalg.det(rot))[:, None, None]          # proper rotations only
    pts = np.stack([do.unproject_to_yaw_frame(cam, img[i], pos[i], rot[i]) for i in range(n)])
    w.add("unproject_to_yaw_frame", "unproject_to_yaw_frame",
          {"image": img, "cam_pos": pos, "cam_rot": rot.reshape(n, 9)},
          {"points": pts.reshape(n, -1)}, attrs=dict(cam))


def dump_camera_fk(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit the camera-pose fixtures the C++ FK is held to.

    The oracle here is PINOCCHIO ON THE ROS URDF, which is itself checked against the
    training-side ``dataset._camera_pose_yaw_frame`` -- exactly, to 0.000000 m and 4e-13 of
    rotation, over zero and five random configurations. Two steps rather than one because
    this generator must not import isaaclab: the chain is C++ -> python pinocchio ->
    robot_rl, and the second link is asserted in robot_rl's own suite where isaaclab exists.

    What the cases are built to catch, since a wrong camera pose is a plausible cloud in the
    wrong place rather than a crash:

    * the JOINT PERMUTATION. Pinocchio orders joints per limb, the policy interleaves them,
      and 27 of the G1's 29 differ. Random angles on every joint make a mismatch enormous;
      zeros would hide it completely.
    * the YAW DIVIDING OUT. Cases share joint angles across different root headings, so an
      implementation that leaves psi in disagrees on those alone.
    * the quaternion CONVENTION. Scalar-last everywhere here; pinocchio's Eigen constructor
      is scalar-first, and mixing them is a silent ~180 degree error.

    Args:
        w: The fixture writer.
        rng: Source of randomness.
        n: Independent robot configurations.
    """
    import pinocchio as pin  # noqa: PLC0415  -- only this case needs it

    # The robot that was TRAINED -- generated from its URDF by g1_model/urdf/make_ros_urdf.py.
    # This said g1_29j.urdf, which is the pre-mode_11 robot: `torso_link` sits 10.0 mm higher
    # on it, so the oracle was computing camera poses for a machine nothing runs.
    urdf = os.path.join(_REPO_ROOT, "transfer", "obelisk", "g1_model", "urdf",
                        "g1_perceptive.urdf")
    if not os.path.exists(urdf):
        print(f"[fixtures] no URDF at {urdf}; skipping camera_pose_yaw_frame")
        return
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    qidx = [model.joints[model.getJointId(j)].idx_q for j in _ISAAC_JOINTS]
    root = model.getFrameId(model.frames[1].name)

    joints = np.zeros((n, len(_ISAAC_JOINTS)))
    joints[1:] = rng.uniform(-0.5, 0.5, size=(n - 1, len(_ISAAC_JOINTS)))
    rpy = np.zeros((n, 3))
    rpy[1:] = rng.uniform(-0.4, 0.4, size=(n - 1, 3))
    # The last case REPEATS the previous joints under a different heading: yaw must divide out,
    # so those two rows have to agree, and an implementation that keeps psi fails on them alone.
    joints[-1] = joints[-2]
    rpy[-1] = [rpy[-2][0], rpy[-2][1], rpy[-2][2] + 1.7]

    quats = np.stack([_quat_from_rpy(*r) for r in rpy])
    pos_out, rot_out = [], []
    for i in range(n):
        q = np.zeros(model.nq)
        q[qidx] = joints[i]
        pin.framesForwardKinematics(model, data, q)
        r_rp = _roll_pitch_matrix(quats[i])
        ps, rs = [], []
        for cam in _CAMERAS:
            mount = data.oMf[root].inverse() * data.oMf[model.getFrameId(cam["mount_body"])]
            cam_in_root = mount * pin.SE3(_mat_from_quat_xyzw(cam["offset_quat_xyzw"]),
                                          np.asarray(cam["offset_pos"]))
            ps.append(r_rp @ cam_in_root.translation)
            rs.append((r_rp @ cam_in_root.rotation).reshape(-1))
        pos_out.append(np.concatenate(ps))
        rot_out.append(np.concatenate(rs))

    w.add("camera_pose_yaw_frame", "camera_pose_yaw_frame",
          {"joint_pos": joints, "base_quat_xyzw": quats},
          {"cam_pos": np.stack(pos_out), "cam_rot": np.stack(rot_out)},
          attrs={"urdf": os.path.relpath(urdf, _REPO_ROOT),
                 "joint_names": list(_ISAAC_JOINTS),
                 "mount_bodies": [c["mount_body"] for c in _CAMERAS],
                 # FLAT, camera-major, because the manifest carries lists and not matrices.
                 # The C++ reshapes by len(mount_bodies). Restating them there is what let the
                 # placeholder pair survive a contract change unnoticed.
                 "offset_pos": [v for c in _CAMERAS for v in c["offset_pos"]],
                 "offset_quat_xyzw": [v for c in _CAMERAS for v in c["offset_quat_xyzw"]]})


def mdp_normalize(d: np.ndarray, cam: dict) -> np.ndarray:
    """``normalize_depth_np`` with a camera contract dict, for brevity above."""
    return do.normalize_depth_np(
        d, offset=cam["offset"], scale=cam["scale"], min_dist=cam["min_dist"],
        max_dist=cam["max_dist"], min_clip_val=cam["min_clip_val"],
        max_clip_val=cam["max_clip_val"])


def main() -> int:
    """Write the fixture set. Returns a process exit code."""
    default_out = os.path.join(_TRANSFER, "deploy", "tests", "fixtures", "parity")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=default_out, help="output directory")
    ap.add_argument("--count", type=int, default=512, help="random cases per function")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")
    args = ap.parse_args()

    if sys.byteorder != "little":
        print(f"ERROR: fixtures are little-endian raw blobs; this host is {sys.byteorder}.")
        return 1

    rng = np.random.default_rng(args.seed)
    w = _ParityWriter(args.out, "transfer/deploy/tools/dump_parity_fixtures.py")
    dump_quat_math(w, rng, args.count)
    dump_window(w, rng, args.count)
    dump_depth(w, rng, args.count)
    dump_depth_xyz(w, rng, min(args.count, 64))
    dump_camera_fk(w, rng, min(args.count, 32))
    path = w.write_manifest(args.seed)

    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"[fixtures] {len(w.cases)} cases -> {path} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
