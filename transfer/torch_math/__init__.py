# Copyright 2026 Zachary Olkin. All rights reserved.

"""Isaaclab-free torch frame/quaternion math -- the reference oracle for the numpy ports.

:mod:`quat_math` is a standalone pure-torch implementation of the xyzw quaternion and
frame-transform primitives the deploy runtime needs. It imports **no** isaaclab and no
robot_rl, so it can be exercised in a torch-only environment (no IsaacSim install).

It sits in the middle of a two-step oracle chain:

* ``tests/test_quat_math_parity.py`` pins it against ``isaaclab.utils.math`` -- the repo's
  own math -- up to fp tolerance and the ``q`` vs ``-q`` sign ambiguity for matrix->quat;
* ``transfer/deploy/tests/test_deploy_parity.py`` then pins the torch-free
  ``deploy/quat_math_np.py`` against THIS, so the deployed runtime stays faithful.

Both tests are dev-only. **Deployment does not use this package** -- sim2sim and hardware
run the ONNX artifacts through ``transfer/deploy``.

History: this was ``transfer/fm``, a full vendored copy of ``robot_rl.flow_matching``'s
inference stack (config / model / normalizer / online / flow / transforms), kept as a torch
oracle for the FM. Once deployment moved to ONNX and
``scripts/flow_matching/fm_test_bench.py`` was deleted, nothing loaded a model through it
and those six modules became unreachable -- so they were dropped rather than kept in
lock-step (a burden that had already let them drift). The frame-transform half of the
parity test now uses ``robot_rl.flow_matching.transforms`` directly, guarded by
``pytest.importorskip``; only the isaaclab-free quaternion math still needs a local copy.
"""

from .quat_math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_conjugate,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    yaw_quat,
)

__all__ = [
    "axis_angle_from_quat",
    "combine_frame_transforms",
    "matrix_from_quat",
    "quat_apply",
    "quat_apply_inverse",
    "quat_box_minus",
    "quat_conjugate",
    "quat_from_matrix",
    "quat_inv",
    "quat_mul",
    "subtract_frame_transforms",
    "yaw_quat",
]
