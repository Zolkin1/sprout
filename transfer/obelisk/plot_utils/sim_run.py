# Copyright 2026 Zachary Olkin. All rights reserved.

"""A MuJoCo sim2sim replay stream, wearing the same interface as a decoded obelisk bag.

So the two stacks can be plotted by ONE piece of code. ``bag_run.Run`` is the shape every
figure and the frame stepper already consume; this presents a sim2sim run the same way, which
means a comparison is an overlay rather than a second plotting pipeline that drifts from the
first.

The source is a ``--dump_replay`` stream rather than ``sim_log.csv`` on purpose: the stream
carries the per-tick DEPTH IMAGES, the served reference window and the raw action, which the
CSV does not, and those are most of what the stepper shows.

WHAT IS NOT THERE, and is exposed as ``None`` rather than faked:

* ``joint_target_isaac`` -- the stream records the raw action, not the PD target, and
  reconstructing it needs the action scale and default pose.
* ``base_lin_b`` -- not recorded; the generator does not read it.
* ``latency`` / ``fm_infer_ms`` -- wall-clock costs of a run whose whole point was to be
  unpaced, so they would compare a headless sweep against a real-time robot.

Time is measured FROM THE FIRST CONTROL TICK on both sides. The sim2sim runner drives from
step 0, and a bag's clock starts at launch -- ~10 s before arming -- so the comparator shifts
the bag onto its own arm instant instead of trusting two zeros to mean the same thing.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TRANSFER not in sys.path:
    sys.path.insert(0, _TRANSFER)

from deploy.replay_stream import load as load_stream  # noqa: E402

from bag_run import block_slices, rot6d_to_rpy  # noqa: E402


class SimRun:
    """A sim2sim replay stream presented as :class:`bag_run.Run`.

    Args:
        stream_dir: Directory holding ``manifest.json`` and the blobs.

    Raises:
        ValueError: If the stream carries no window layout -- an older dump, which cannot be
            sliced into blocks and would silently plot the wrong columns.
    """

    #: What this is, for figure titles and so a reader can tell which stack a trace came from.
    label = "sim2sim"

    def __init__(self, stream_dir: str) -> None:
        arrays, manifest = load_stream(stream_dir)
        attrs = manifest["cases"][0]["attrs"]
        self.stream_dir = stream_dir
        self.manifest = manifest
        self._a = arrays

        for key in ("joint_names_isaac", "window_offsets", "reference_blocks", "node_dim"):
            if key not in attrs:
                raise ValueError(
                    f"{stream_dir} carries no '{key}'. It was dumped before the stream "
                    "recorded the window layout; re-dump it. Guessing the layout would slice "
                    "the window into the wrong blocks and plot confident nonsense.")

        dt = float(attrs.get("dt", 0.02))
        self.t = arrays["ctrl_step"].ravel() * dt
        self.ticks = list(range(len(self.t)))
        self.armed = np.ones(len(self.t), dtype=bool)   # the runner drives from step 0
        self.clip_mode = False

        self.joint_names_isaac = [str(n) for n in attrs["joint_names_isaac"]]
        self.joint_names_hw = list(self.joint_names_isaac)
        self.num_joints = len(self.joint_names_isaac)
        self.joint_pos_isaac = arrays["joint_pos"]
        self.joint_target_isaac = None
        self.action = arrays["action"]
        self.command = arrays["command"]
        self.base_quat = arrays["base_quat_xyzw"]
        self.base_ang_b = arrays["base_ang_vel_b"]
        self.base_lin_b = None
        self.latency = None
        self.fm_infer_ms = None
        self.trk_infer_ms = None
        self.fm_cursor = None
        self.clip_step = None

        self.offsets = np.asarray(attrs["window_offsets"])
        self.blocks = [str(b) for b in attrs["reference_blocks"]]
        self.node_dim = int(attrs["node_dim"])
        self.slices = block_slices(self.blocks, self.node_dim, self.num_joints)
        k, d = len(self.offsets), self.node_dim
        self.window = arrays["window"].reshape(len(self.t), k, d)
        # Identical slicing to the bag side, deliberately: base_pose is
        # [R_yawT (p_ref - p_act) (3), rot6d (6)], so the position error is IN the window.
        if "base_pose" in self.slices:
            bp = self.window[:, :, self.slices["base_pose"]]
            self.ref_pos_err = bp[:, :, :3]
            self.ref_rpy_err = rot6d_to_rpy(bp[:, :, 3:9])
        else:
            self.ref_pos_err = self.ref_rpy_err = None
        self.ref_joint = (self.window[:, :, self.slices["joint_pos"]]
                          if "joint_pos" in self.slices else None)

        # Depth, one image per camera per tick, in METRES -- the stream records it raw because
        # that is what the C++ runtime is fed. Shaped from the contract in the manifest.
        spec = _camera_shapes(manifest)
        self._depth = arrays["depth_raw"]
        self._cam_shapes = spec
        self._cam_offsets = np.cumsum([0] + [h * w for h, w in spec])

    @property
    def survived(self) -> bool | None:
        """The runner's own verdict, or None if the stream predates it."""
        return self.manifest["cases"][0]["attrs"].get("survived")

    def depth_at(self, tick: int, cam: int) -> np.ndarray | None:
        """The camera's image at this tick, as ``(H, W)`` metres, or None.

        Args:
            tick: Index into :attr:`t`.
            cam: Camera index in contract order.

        Returns:
            The image, or None when the stream has no such camera.
        """
        if cam >= len(self._cam_shapes) or not self._depth.shape[1]:
            return None
        h, w = self._cam_shapes[cam]
        lo, hi = int(self._cam_offsets[cam]), int(self._cam_offsets[cam + 1])
        return self._depth[tick, lo:hi].reshape(h, w)

    def nearest_tick(self, t: float) -> int:
        """Index of the tick closest to time ``t``."""
        return int(np.argmin(np.abs(self.t - t)))


def _camera_shapes(manifest: dict) -> list[tuple[int, int]]:
    """Per-camera ``(H, W)`` from the stream's own artifact metadata.

    Read from the tracker.onnx the stream names rather than assumed: a transposed camera is
    shape-silent (30x26 against 26x30 has the same pixel count), and the whole point of a
    contract travelling with the artifact is that no consumer restates it.

    Args:
        manifest: The stream manifest.

    Returns:
        One ``(H, W)`` per camera, or ``[]`` when the artifact cannot be read.
    """
    path = manifest.get("artifacts", {}).get("tracker_onnx", "")
    if not path or not os.path.exists(path):
        return []
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        params = json.loads({p.key: p.value for p in model.metadata_props}["policy_parameters"])
        return [(int(c["height"]), int(c["width"])) for c in params["depth_camera_spec"]]
    except Exception:                                                     # noqa: BLE001
        # A missing or unreadable artifact costs the depth panels, not the whole comparison.
        return []
