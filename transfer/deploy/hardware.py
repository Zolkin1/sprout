# Copyright 2026 Zachary Olkin. All rights reserved.

"""Hardware-facing helpers for the torch-free generator+tracker deploy runtime.

Thin glue for the obelisk controller to use the shared
:class:`deploy.gen_tracker.GenTrackerPolicy` (numpy + onnxruntime) on the robot:

* :func:`load_gen_tracker_from_hf` -- download ``tracker.onnx`` + ``fm.onnx`` from
  HuggingFace (cached) and return a loaded policy.
* :func:`build_hardware_state` -- assemble the ``create_obs`` state dict from the
  estimator's readings (joint encoders + IMU/base estimate + velocity command) and the
  exteroception the policy was trained on.

The ROS2 control loop lives in the obelisk ``g1_control`` controller node; it calls
``policy.warm_start(build_hardware_state(...))`` on entering low-level control, then each
50 Hz cycle ``policy.get_action(policy.create_obs(build_hardware_state(...)), hw_names)``.

One frame convention to get right on the estimator side: the FM history needs the base
**world-frame** linear velocity.  ``nav_msgs/Odometry`` carries ``pose`` in the header
frame but ``twist`` in the CHILD frame, so a FAST-LIO twist is body-frame and must be
rotated by the base quaternion before being passed as ``base_lin_vel_w``.  Getting this
wrong is silent -- every history token is simply wrong.
"""

from __future__ import annotations

import os

import numpy as np

from .depth_obs import normalize_camera_images
from .gen_tracker import GenTrackerPolicy
from .quat_math_np import quat_apply
from .tracker_session import convert_joint_order


#: The two files a deployment needs, relative to the policy folder in the repo.
ARTIFACT_NAMES = ("tracker.onnx", "fm.onnx")

#: Bounds on the online fetch. The download happens inside the controller's lifecycle
#: ``on_configure``; an unbounded HTTP call there hangs BRING-UP, and the only cure is
#: killing the launch. A robot LAN with no route out must fail in seconds, not block.
_ETAG_TIMEOUT_S = 15.0
_DOWNLOAD_TIMEOUT_S = 15.0
_ATTEMPTS = 3
_BACKOFF_S = 3.0


def fetch_artifacts(
    hf_repo_id: str,
    hf_policy_folder: str,
    cache_root: str,
    refresh: bool = False,
) -> tuple[str, ...]:
    """Resolve ``tracker.onnx`` + ``fm.onnx`` locally, downloading only if needed.

    Offline-FIRST: a cache hit never touches the network, because recent
    ``huggingface_hub`` fires a best-effort telemetry request on every ``hf_hub_download``
    -- even for a cache hit -- and on the robot that is a bring-up hang, not a slow start.
    Only an actual miss (or ``refresh``) lifts offline mode, and then under a timeout.

    Args:
        hf_repo_id: HuggingFace repo id (``user/repo``).
        hf_policy_folder: Folder within the repo holding the two ONNX files.
        cache_root: Local directory to cache downloads under.
        refresh: Re-download even on a cache hit, to pick up a policy re-pushed to the
            same path. Use before going to the robot, never on the robot.

    Returns:
        Absolute paths to the artifacts, in :data:`ARTIFACT_NAMES` order.

    Raises:
        RuntimeError: If an artifact is neither cached nor downloadable, with the failure
            named -- offline with a cold cache is the common case and should say so.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    base = os.path.join(cache_root, hf_repo_id.replace("/", "_"))
    paths = []
    for name in ARTIFACT_NAMES:
        local = os.path.join(base, hf_policy_folder, name)
        if os.path.exists(local) and not refresh:
            paths.append(local)
            continue
        paths.append(_download(hf_repo_id, f"{hf_policy_folder}/{name}", base))
    return tuple(paths)


def _download(repo_id: str, filename: str, local_dir: str) -> str:
    """Fetch one artifact, bounded and retrying, then restore offline mode.

    Args:
        repo_id: HuggingFace repo id.
        filename: Path within the repo.
        local_dir: Directory to download into.

    Returns:
        The local path to the downloaded file.

    Raises:
        RuntimeError: If every attempt fails.
    """
    import time

    from huggingface_hub import hf_hub_download

    prior = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(_ETAG_TIMEOUT_S))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(_DOWNLOAD_TIMEOUT_S))
    try:
        last = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
            except Exception as exc:  # noqa: BLE001 -- reported below with context
                last = exc
                print(f"[WARN deploy.hardware] fetch of {filename} failed "
                      f"(attempt {attempt}/{_ATTEMPTS}): {exc}")
                if attempt < _ATTEMPTS:
                    time.sleep(_BACKOFF_S)
        raise RuntimeError(
            f"Could not obtain {repo_id}/{filename}: not in the cache under {local_dir} and "
            f"the download failed ({last}). Pre-warm the cache while the machine still has "
            "a route out -- transfer/obelisk/scripts/prefetch_policy.py."
        )
    finally:
        if prior is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior


def load_gen_tracker_from_hf(
    hf_repo_id: str,
    hf_policy_folder: str,
    cache_root: str,
    fm_update_every: int = 1,
    num_threads: int = 0,
    refresh: bool = False,
    providers=None,
) -> GenTrackerPolicy:
    """Resolve ``tracker.onnx`` + ``fm.onnx`` from HF (cached) and load the policy.

    Args:
        hf_repo_id: HuggingFace repo id (``user/repo``).
        hf_policy_folder: Folder within the repo holding the two ONNX files.
        cache_root: Local directory to cache downloads under.
        fm_update_every: Generator re-inference cadence (control steps). Fallback only:
            overridden by 'fm_update_every' in the tracker metadata if present.
        num_threads: onnxruntime intra-op threads (0 = auto).
        refresh: Re-download even on a cache hit.
        providers: onnxruntime execution providers (see
            ``tracker_session.resolve_providers``). None = CPU.

    Returns:
        A loaded :class:`GenTrackerPolicy`.
    """
    tracker, fm = fetch_artifacts(hf_repo_id, hf_policy_folder, cache_root, refresh=refresh)
    policy = GenTrackerPolicy(tracker, fm, fm_update_every=fm_update_every,
                              num_threads=num_threads, providers=providers)
    policy.load()
    return policy


def build_hardware_state(
    policy: GenTrackerPolicy,
    *,
    root_pos: np.ndarray,
    base_quat_xyzw: np.ndarray,
    base_lin_vel_w: np.ndarray,
    base_ang_vel_b: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_names: list[str],
    command: np.ndarray,
    base_ang_vel_w: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    depth_raw=None,
    height_scan: np.ndarray | None = None,
) -> dict:
    """Assemble the ``GenTrackerPolicy.create_obs`` state dict from robot readings.

    Args:
        policy: The loaded policy (for the isaac joint order).
        root_pos: Base position in world ``(3,)``.
        base_quat_xyzw: Base orientation quaternion ``(4,)`` xyzw.
        base_lin_vel_w: Base **world-frame** linear velocity ``(3,)`` (link frame).
        base_ang_vel_b: Base **body-frame** angular velocity ``(3,)`` (the
            ``base_ang_vel`` obs term; from the IMU).
        joint_pos/joint_vel: Joint encoder readings ``(nj,)`` in ``joint_names`` order.
        joint_names: The encoder joint order (mapped to the isaac order here).
        command: Velocity command ``(vx, vy, wz)``.
        base_ang_vel_w: Base world-frame angular velocity ``(3,)``; if ``None`` it is
            derived as ``quat_apply(base_quat, base_ang_vel_b)``.
        frames: End-effector frame world states ``(K, 13)`` =
            ``[pos(3), quat(4 xyzw), lin_vel(3), ang_vel(3)]`` in
            ``policy.ctx.frame_names`` order, required when the FM conditions on extra
            frames (compute them from forward kinematics on the measured joints).
        depth: Already-normalised depth ``(T, H*W)`` in camera-contract order. Use this
            when something upstream normalised for you; otherwise pass ``depth_raw``.
        depth_raw: One raw reading per camera in **metres along the optical axis**, in
            camera-contract order, normalised here via
            :func:`~deploy.depth_obs.normalize_camera_images`. This is the usual hardware
            path -- the ROS callback hands over what the camera published.
        height_scan: Per-ray ground clearance, for a height-scan policy.

    Returns:
        State dict consumed by :meth:`GenTrackerPolicy.create_obs`.

    Raises:
        ValueError: If the policy needs an exteroception reading that was not supplied,
            or if both ``depth`` and ``depth_raw`` are given.
    """
    isaac = policy.get_joint_names()
    q = np.asarray(base_quat_xyzw, dtype=np.float64)
    ang_b = np.asarray(base_ang_vel_b, dtype=np.float64)
    ang_w = np.asarray(base_ang_vel_w, dtype=np.float64) if base_ang_vel_w is not None else quat_apply(q, ang_b)
    state = {
        "root_pos": np.asarray(root_pos, dtype=np.float64),
        "root_quat": q,
        "root_lin_vel_w": np.asarray(base_lin_vel_w, dtype=np.float64),
        "root_ang_vel_w": ang_w,
        "base_ang_vel": ang_b,
        "joint_pos": convert_joint_order(np.asarray(joint_pos, dtype=np.float64), list(joint_names), isaac),
        "joint_vel": convert_joint_order(np.asarray(joint_vel, dtype=np.float64), list(joint_names), isaac),
        "command": np.asarray(command, dtype=np.float64),
    }
    if policy.ctx.num_extra_frames > 0:
        if frames is None:
            raise ValueError(
                f"This FM conditions on {policy.ctx.num_extra_frames} end-effector frames "
                f"({policy.ctx.frame_names}); pass `frames` (K, 13) in that order "
                "(compute them from forward kinematics on the measured joints)."
            )
        fr = np.asarray(frames, dtype=np.float64)
        state.update(frame_pos=fr[:, 0:3], frame_quat=fr[:, 3:7], frame_lin=fr[:, 7:10], frame_ang=fr[:, 10:13])

    # Exteroception. Required-ness comes from the artifacts, not from the caller: a
    # policy that reads terrain and is handed none would otherwise fail deep inside the
    # obs assembly (or, for a term the tracker lacks but the FM wants, not at all).
    if depth is not None and depth_raw is not None:
        raise ValueError("Pass either `depth` (normalised) or `depth_raw` (metres), not both.")
    spec = policy.get_depth_camera_spec()
    if depth_raw is not None:
        depth = normalize_camera_images(spec, depth_raw)
    if spec:
        if depth is None:
            cams = ", ".join(str(c.get("sensor") or c["obs_term"]) for c in spec)
            raise ValueError(
                f"This policy reads {len(spec)} depth camera(s) ({cams}); pass "
                "`depth_raw` (metres, one reading per camera in contract order) or an "
                "already-normalised `depth`."
            )
        state["depth"] = np.asarray(depth, dtype=np.float64)
    elif depth is not None:
        raise ValueError("`depth` supplied but this policy has no depth cameras.")

    if policy.needs_height_scan():
        if height_scan is None:
            raise ValueError(
                "This policy reads a height scan; pass `height_scan` (per-ray ground "
                "clearance, in metres)."
            )
        state["height_scan"] = np.asarray(height_scan, dtype=np.float64)
    elif height_scan is not None:
        state["height_scan"] = np.asarray(height_scan, dtype=np.float64)
    return state
