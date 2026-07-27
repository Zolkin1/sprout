# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline hardware-readiness smoke for the torch-free deploy runtime.

No robot / ROS2 / HF needed: loads the local ONNX artifacts, builds the
``create_obs`` state from synthetic estimator readings (with a scrambled joint
order to exercise the joint-order conversion), warm-starts, and steps the policy
for a few cycles. Asserts finite PD targets and that **torch is never imported**.

Runs against the most recently exported run that has a ``tracker.onnx``, and skips when
nothing is exported. Whichever exteroception the run trained on is fed a synthetic
reading through the same ``build_hardware_state`` arguments the robot uses -- depth in
METRES (the camera's own frame), or a per-ray height scan.

Run: ``python transfer/deploy/tests/test_hardware_offline.py``
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.hardware import build_hardware_state  # noqa: E402


def _latest_export() -> str | None:
    """The ``exported`` dir to test against.

    ``DEPLOY_EXPORT_DIR`` picks one explicitly -- the artifacts under ``logs/`` are
    whatever was last exported there, which may predate the current exporter; point this
    at a fresh ``export_deploy_onnx.py`` output to test the artifacts you actually mean.
    Otherwise the most recently modified ``exported`` dir holding a ``tracker.onnx``.
    """
    override = os.environ.get("DEPLOY_EXPORT_DIR")
    if override:
        return override
    hits = glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "*", "*", "exported", "tracker.onnx"))
    return os.path.dirname(max(hits, key=os.path.getmtime)) if hits else None


def test_hardware_offline() -> None:
    """Synthetic estimator readings -> finite PD targets, torch-free."""
    exp = _latest_export()
    if exp is None:
        print("[SKIP] No exported tracker.onnx under logs/rsl_rl/*/*/exported/.")
        return
    fm = os.path.join(exp, "fm.onnx")
    print(f"[INFO] artifacts: {exp} (fm.onnx: {'yes' if os.path.exists(fm) else 'no'})")
    policy = GenTrackerPolicy(os.path.join(exp, "tracker.onnx"),
                              fm if os.path.exists(fm) else None)
    policy.load()
    isaac = policy.get_joint_names()
    nj = len(isaac)
    depth_spec = policy.get_depth_camera_spec()
    # Width of the height scan to synthesise, 0 when this policy does not read one. The
    # tracker's own term is authoritative (a tracker-only ctx has no real height_dim).
    scan_dim = 0
    if policy.needs_height_scan():
        terms = {t["name"]: t["shape"] for t in policy.get_obs_terms()}
        scan_dim = int(terms.get("height_scan", policy.ctx.height_dim))

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
        # Exteroception goes in RAW, exactly as the robot supplies it: metres along each
        # camera's optical axis (with a few invalid pixels, which every camera produces),
        # or metres of ground clearance. build_hardware_state owns the normalisation, so
        # this exercises the path the controller actually takes.
        depth_raw = None
        if depth_spec:
            depth_raw = []
            for cam in depth_spec:
                px = rng.uniform(float(cam["min_dist"]), float(cam["max_dist"]),
                                 int(cam["width"]) * int(cam["height"]))
                px[rng.random(px.size) < 0.05] = np.nan       # dropouts
                depth_raw.append(px)
        height_scan = rng.uniform(-0.3, 0.3, scan_dim) if scan_dim else None
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
            depth_raw=depth_raw,
            height_scan=height_scan,
        )

    policy.warm_start(reading(np.zeros(3)))
    last = None
    for _ in range(8):
        state = reading(np.array([0.4, 0.0, 0.1]))
        # Without a generator the clip reference drives the tracker, so the obs must be
        # built directly rather than via create_obs (which samples the FM).
        if policy.fm_enabled:
            obs = policy.create_obs(state)
        else:
            obs = policy._build_obs(state, _clip_like_reference(policy, state, rng))
        last = policy.get_action(obs, isaac)
        assert obs.shape == (1, policy.get_num_obs())
        assert last.shape == (nj,)
        assert np.isfinite(obs).all() and np.isfinite(last).all()

    assert "torch" not in sys.modules, "deploy path must be torch-free"
    print(f"obs {policy.get_num_obs()}  action {last.shape}  finite=True  "
          f"depth_cameras={len(depth_spec)}  height_scan={scan_dim}  "
          f"pred_valid={policy._pred_valid}")
    print("HARDWARE OFFLINE SMOKE PASSED (torch-free)")


def _clip_like_reference(policy, state, rng) -> dict:
    """A synthetic reference in the shape the clip source supplies (tracker-only mode)."""
    nj, k = policy.ctx.nj, policy.ctx.target_num_extra_frames
    node = lambda: {
        "root_pos": np.asarray(state["root_pos"]) + rng.standard_normal(3) * 0.01,
        "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "root_lin": rng.standard_normal(3) * 0.05,
        "root_ang": rng.standard_normal(3) * 0.05,
        "jp": rng.standard_normal(nj) * 0.1,
        "jv": rng.standard_normal(nj) * 0.1,
        "frame_pos": rng.standard_normal((k, 3)) * 0.1,
    }
    return {"current": node(), "future": [node() for _ in range(len(policy.future_offsets))]}


if __name__ == "__main__":
    test_hardware_offline()
