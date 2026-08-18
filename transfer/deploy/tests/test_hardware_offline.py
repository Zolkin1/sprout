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
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.hardware import build_hardware_state  # noqa: E402

#: True only when this file is executed directly, which is the one context where
#: ``"torch" not in sys.modules`` says anything about the DEPLOY path rather than about
#: whatever else the interpreter has already imported.  See
#: :func:`test_deploy_path_is_torch_free`.
_STANDALONE = __name__ == "__main__"


def _latest_export() -> str | None:
    """The ``exported`` dir to test against.

    ``DEPLOY_EXPORT_DIR`` picks one explicitly -- the artifacts under ``logs/`` are
    whatever was last exported there, which may predate the current exporter; point this
    at a fresh ``play.py`` export to test the artifacts you actually mean.
    Otherwise the most recently modified ``exported`` dir holding a ``tracker.onnx`` --
    skipping any pair whose generator conditions on a PRIVILEGED signal.  Such a pair is
    sim-only by construction (``GenTrackerPolicy`` refuses it without an explicit opt-in, and
    that refusal is what this test would otherwise be reporting as a hardware failure), so
    "the newest DEPLOYABLE export" is the thing this test means by "the newest export".
    """
    override = os.environ.get("DEPLOY_EXPORT_DIR")
    if override:
        return override
    hits = glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "*", "*", "exported", "tracker.onnx"))
    hits = [h for h in hits if not _is_privileged_pair(os.path.dirname(h))]
    return os.path.dirname(max(hits, key=os.path.getmtime)) if hits else None


def _is_privileged_pair(exported: str) -> bool:
    """Whether ``exported`` holds a generator conditioned on a privileged signal.

    Read from the ONNX metadata directly rather than by loading the pair, so this stays a
    cheap filter and does not depend on the very refusal it is filtering for.

    Args:
        exported: An ``exported`` directory.

    Returns:
        True when its ``fm.onnx`` declares a privileged exteroception contract.
    """
    fm = os.path.join(exported, "fm.onnx")
    if not os.path.exists(fm):
        return False
    import json

    import onnxruntime as ort

    meta = ort.InferenceSession(fm, providers=["CPUExecutionProvider"]) \
        .get_modelmeta().custom_metadata_map.get("fm_meta")
    if not meta:
        return False
    return any(bool(s.get("privileged")) for s in (json.loads(meta).get("extero_spec") or []))


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
    try:
        policy.load()
    except ValueError as exc:
        stale = any(marker in str(exc) for marker in
                    ("base_pose_convention", "window_blocks", "pre-rename exteroception keys"))
        if not stale:
            raise
        # Refusing an artifact from before the reference-window representation (or before
        # the base_pose sign fix, or before the exteroception fields were renamed) is the
        # runtime doing its job -- the whole point of those guards. Note the rename case is
        # the CHEAP one: unlike the others it needs only a re-export, not a retrain, so if
        # this skip fires re-run export_fm_onnx.py on the run's checkpoint rather than
        # waiting for one.
        # NOTE: a K>1 artifact is no longer in this list -- the runtime serves any
        # fm_update_every whose horizon fits. A horizon that does NOT fit
        # ("needs F >= ...") is an unservable export rather than a stale one, so it is
        # deliberately left to raise: the fix is a config or a retrain, not a skip.
        print(f"[SKIP] {exp} cannot be served by this runtime: {exc}")
        return
    stacked = [t["name"] for t in policy.get_obs_terms() if int(t.get("history_length", 0)) > 0]
    print(f"[INFO] history-stacked terms: {', '.join(stacked) if stacked else '(none)'}")
    isaac = policy.get_joint_names()
    nj = len(isaac)
    depth_spec = policy.get_depth_camera_spec()
    # Width of the height scan to synthesise, 0 when this policy does not read one. The
    # tracker's own term is authoritative (a tracker-only ctx has no real extero_sensor_dim).
    scan_dim = 0
    if policy.needs_height_scan():
        terms = {t["name"]: t["shape"] for t in policy.get_obs_terms()}
        scan_dim = int(terms.get("height_scan", policy.ctx.extero_sensor_dim))

    # Estimator joint order != isaac order (reversed) -> exercises convert_joint_order.
    enc_names = list(reversed(isaac))
    rng = np.random.default_rng(0)

    # NOTE: no measured end-effector frames are supplied, and that is the point. The
    # generator's history is its own previous output, so nothing on the robot needs forward
    # kinematics; build_hardware_state used to demand `frames` here and would kill the
    # controller one tick after arming. If this test ever needs FK again, something has
    # reintroduced a measured-state dependency into the FM conditioning.

    def reading(cmd):
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
            depth_raw=depth_raw,
            height_scan=height_scan,
        )

    policy.reset()
    last = None
    for _ in range(8):
        state = reading(np.array([0.4, 0.0, 0.1]))
        # Without a generator the clip reference drives the tracker, so the obs must be
        # built directly rather than via create_obs (which samples the FM).
        if policy.fm_enabled:
            obs = policy.create_obs(state)
        else:
            obs = policy.create_obs_from_reference(state, _clip_like_reference(policy, state, rng))
        last = policy.get_action(obs, isaac)
        assert obs.shape == (1, policy.get_num_obs())
        assert last.shape == (nj,)
        assert np.isfinite(obs).all() and np.isfinite(last).all()

    # Only meaningful in a fresh interpreter. Under pytest, sibling tests have imported
    # torch long before this file runs, so sys.modules describes the session rather than
    # the deploy path -- asserting here would fail for a reason that has nothing to do with
    # deploy. test_deploy_path_is_torch_free re-runs this file standalone, where it holds.
    if _STANDALONE:
        assert "torch" not in sys.modules, "deploy path must be torch-free"
        # Same argument, different package. deploy.fk_mujoco exists for the RViz ribbon and
        # the sim2sim viewer, and MuJoCo must stay off the ON-ROBOT import graph: it is tens
        # of MB the control path never calls, and the robot is not required to have it. The
        # import is deliberately inside that module; this is what stops a convenience
        # `from .fk_mujoco import ...` at the top of hardware.py or gen_tracker.py from
        # quietly dragging it in.
        assert "mujoco" not in sys.modules, (
            "the deploy control path must not import mujoco -- keep it inside "
            "deploy/fk_mujoco.py, which only the viewers touch"
        )
    print(f"obs {policy.get_num_obs()}  action {last.shape}  finite=True  "
          f"depth_cameras={len(depth_spec)}  height_scan={scan_dim}  "
          f"pred_valid={policy._pred_valid}")
    print("HARDWARE OFFLINE SMOKE PASSED (torch-free)")


def test_deploy_path_is_torch_free() -> None:
    """The deploy runtime must never import torch -- checked in a FRESH interpreter.

    The invariant is real (``transfer/deploy`` is numpy + onnxruntime only, so it can run
    on hardware without a torch install), but it is unobservable from inside the pytest
    session: by the time this file is collected, sibling tests have already imported torch.
    Re-running the module standalone isolates it, and its in-process assertion then applies.

    Skips silently when nothing is exported -- the child prints ``[SKIP]`` and exits 0, same
    as :func:`test_hardware_offline`.
    """
    proc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, (
        f"standalone run failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
    if "[SKIP]" not in proc.stdout:
        assert "PASSED (torch-free)" in proc.stdout, (
            f"standalone run did not reach the torch-free check:\n{proc.stdout}"
        )


def _clip_like_reference(policy, state, rng) -> dict:
    """A synthetic reference in the shape the clip source supplies (tracker-only mode).

    A tracker trained with ``tracking_mode="body"`` additionally needs the root-frame body
    block, which real playback gets from ``ClipReference(root_frame_bodies=...)``.
    """
    nj, k = policy.ctx.nj, policy.ctx.target_num_extra_frames
    nb = len(policy._trk.get_root_frame_bodies())

    def node():
        n = {
            "root_pos": np.asarray(state["root_pos"]) + rng.standard_normal(3) * 0.01,
            "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "root_lin": rng.standard_normal(3) * 0.05,
            "root_ang": rng.standard_normal(3) * 0.05,
            "jp": rng.standard_normal(nj) * 0.1,
            "jv": rng.standard_normal(nj) * 0.1,
            "frame_pos": rng.standard_normal((k, 3)) * 0.1,
        }
        if nb:
            n["body_pos_root"] = rng.standard_normal(7 * nb) * 0.1
            n["body_vel_root"] = rng.standard_normal(6 * nb) * 0.1
        return n

    # One node per window offset, offset 0 (the current step) first.
    return {"window": [node() for _ in policy.window_offsets]}


if __name__ == "__main__":
    test_hardware_offline()
