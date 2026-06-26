# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline hardware-readiness smoke for the torch-free deploy runtime.

No robot / ROS2 / HF needed: loads the local ONNX artifacts, builds the
``create_obs`` state from synthetic estimator readings (with a scrambled joint
order to exercise the joint-order conversion), warm-starts, and steps the policy
for a few cycles. Asserts finite PD targets and that **torch is never imported**.

Run: ``python transfer/deploy/tests/test_hardware_offline.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.hardware import build_hardware_state  # noqa: E402

EXP = os.path.join(ROOT, "logs/rsl_rl/g1/2026-06-22_09-47-52_TM2_generator_tracking_test/exported")


def test_hardware_offline() -> None:
    """Synthetic estimator readings -> finite PD targets, torch-free."""
    policy = GenTrackerPolicy(os.path.join(EXP, "tracker.onnx"), os.path.join(EXP, "fm.onnx"))
    policy.load()
    isaac = policy.get_joint_names()
    nj = len(isaac)

    # Estimator joint order != isaac order (reversed) -> exercises convert_joint_order.
    enc_names = list(reversed(isaac))
    rng = np.random.default_rng(0)

    # The FM conditions on (and predicts) the same ee frames, so the hardware state
    # supplies them every step (from FK on the measured joints).
    khist = policy.ctx.num_extra_frames

    def reading(cmd):
        frames = None
        if khist > 0:
            q = rng.standard_normal((khist, 4)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
            frames = np.concatenate([rng.standard_normal((khist, 3)) * 0.1, q,
                                     rng.standard_normal((khist, 6)) * 0.05], axis=-1)
        return build_hardware_state(
            policy,
            root_pos=np.array([0.0, 0.0, 0.78]),
            base_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            base_lin_vel_w=rng.standard_normal(3) * 0.05,
            base_ang_vel_b=rng.standard_normal(3) * 0.05,
            joint_pos=rng.standard_normal(nj) * 0.1,
            joint_vel=rng.standard_normal(nj) * 0.1,
            joint_names=enc_names,
            command=np.asarray(cmd, dtype=np.float64),
            frames=frames,
        )

    policy.warm_start(reading(np.zeros(3)))
    last = None
    for _ in range(8):
        obs = policy.create_obs(reading(np.array([0.4, 0.0, 0.1])))
        last = policy.get_action(obs, isaac)
        assert obs.shape == (1, policy.get_num_obs())
        assert last.shape == (nj,)
        assert np.isfinite(obs).all() and np.isfinite(last).all()

    assert "torch" not in sys.modules, "deploy path must be torch-free"
    print(f"obs {policy.get_num_obs()}  action {last.shape}  finite=True  pred_valid={policy._pred_valid}")
    print("HARDWARE OFFLINE SMOKE PASSED (torch-free)")


if __name__ == "__main__":
    test_hardware_offline()
