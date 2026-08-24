# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 low-level controller: the flow-matching generator + RL tracker, on hardware.

DEPRECATED. ``g1_control_cpp`` (``config/g1_cpp.yml``) is the supported controller and the
default in EVERY run script and every mode, clip playback included; this one is kept for one
shrinking reason and will be removed: bisecting a regression against a known-good
implementation, by launching ``g1.yml`` (or ``g1_debug_clip.yml``) instead of its C++ twin
and changing nothing else.

Clip playback is no longer a reason. ``deploy.clip_reference`` is ported --
``g1_deploy::ClipReference``, fed by a binary pack rather than the ``.h5`` the image has no
HDF5 to read -- and the two implementations are held to 1e-11 by
``deploy_cpp/test/clip_reference_parity_test.cc``, which is what makes running one against
the other a bisect rather than a guess.

It is not merely unsupported on the robot, it is UNUSABLE there: aarch64 rclpy is a cp310 C
extension and the only onnxruntime GPU build that runs on a Thor is cp312, so two C
extensions built for different CPython ABIs cannot share a process and this node reaches the
CPU only -- ~471 ms per generator tick against a 20 ms control period. That is the whole
reason the C++ port exists.

Do not add features here. Anything new belongs in ``transfer/deploy_cpp`` (the numeric half,
tested against the python oracle with no ROS) or in ``g1_control_cpp`` (the ROS half).

The policy itself is NOT implemented here. ``transfer/deploy.GenTrackerPolicy`` is the one
runtime shared by the MuJoCo sim2sim runner and this node, so the observation assembly,
the generator cadence and the action scaling are identical in sim and on the robot by
construction rather than by two implementations agreeing. This file is the ROS half:
sensors in, PD targets out, and the safety state machine around them.

Everything about the policy comes from the artifacts' embedded metadata -- joint order,
gains, observation terms and their clips, the depth-camera contract. There is no
``policy_parameters.yaml``.

The bits that matter for not breaking the robot:

* **Arming follows the robot, not a button.** The controller publishes nothing at all
  until the G1's own execution FSM reports ``LOW_LEVEL_CONTROL``; the interface's
  passive-hold timer drives USER_POSE and DAMPING on its own.
* **Arming hands straight to the policy.** There is no PD warm-up phase. Whatever is
  feeding the tracker -- the generator or a clip -- drives from the first tick, which is
  what the MuJoCo sim2sim runner does and what the policy was trained for. A stiffened
  hold would put the robot in a state no training episode ever started from, and in the
  generator's case it also fed the history rows of zeros (``_make_snapshot`` reads the
  runtime's own previously-served node, which does not exist yet) -- which flipped
  ``null_mask`` off and denied the model the learned null-history embedding meant to cover
  exactly those first steps.
* **Sensors are watchdogged.** A stale depth frame or velocity command zeroes the command
  rather than letting the robot keep walking on old data.
* **The joint sets must match exactly.** A policy that does not command every hardware
  joint is refused, rather than the remainder being filled from a guess.

**Debug-clip mode** (``debug_clip``) replaces the generator with a pre-made motion clip.
The generator is then not loaded AT ALL -- not merely unused -- so a run in this mode
cannot be contaminated by it, and a failure is unambiguously the tracker or this stack.
It is the first thing to reach for on a stack that does not work: the ``stand`` case asks
the robot to hold still, so if that falls, nothing downstream of the observation is worth
debugging yet. See ``scripts/run_sim_clip.sh`` and ``config/g1_debug_clip.yml``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from g1_msgs.msg import ControllerDebug, FmPlan
from obelisk_control_msgs.msg import ExecutionFSM, PDFeedForward, VelocityCommand
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_py.core.control import ObeliskController
from obelisk_py.core.utils.ros import spin_obelisk
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# The torch-free deploy runtime lives in the repo, not in this ROS package: it is shared
# with the MuJoCo sim2sim runner and must not fork. This file sits at
# <repo>/transfer/obelisk/g1_control/g1_control/, so <repo>/transfer is three levels up.
#
# REALPATH, not abspath. colcon --symlink-install imports this module through
# build/g1_control/g1_control/, a symlink into the source tree; abspath does not resolve
# symlinks, so the three hops landed on <repo>/transfer/obelisk and the node died at import
# with "ModuleNotFoundError: No module named 'deploy'" -- while running the file just fine
# from a source checkout. realpath resolves back into the source tree first, so both layouts
# give the same answer.
_TRANSFER = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
)
if not os.path.isdir(os.path.join(_TRANSFER, "deploy")):
    raise RuntimeError(
        f"the shared deploy runtime is not at {_TRANSFER}/deploy (derived from "
        f"{os.path.realpath(__file__)}). In the container this means REPO_ROOT is not "
        "mounted, or this package was copied out of <repo>/transfer/obelisk/."
    )
if _TRANSFER not in sys.path:
    sys.path.insert(0, _TRANSFER)

#: Where the policy artifacts are cached on disk. Anchored on ``_TRANSFER`` rather than
#: counting ``..`` from ``__file__`` so it survives the symlink-install layout, and because
#: it MUST equal ``prefetch_policy.DEFAULT_CACHE`` -- which derives the same anchor the same
#: way. It previously did not: two ``..`` from an unresolved ``__file__`` pointed at
#: ``transfer/obelisk/build/resource/hf_cache``, so the node re-downloaded (and offline,
#: failed) even though prefetch_policy had warmed the real directory.
HF_CACHE_DIR = os.path.join(_TRANSFER, "obelisk", "g1_control", "resource", "hf_cache")

from deploy.diagnostics import RateTracker, depth_health, obs_checksum  # noqa: E402
from deploy.hardware import build_hardware_state, load_gen_tracker_from_hf  # noqa: E402

#: Diagnostic topics. Not obelisk-managed: these are first-party debug output with no place
#: in the control graph, and routing them through the framework's keyed config would imply
#: a node contract they are not part of. Prefixed so one `ros2 bag record` pattern catches
#: them all and so they are obviously not part of the robot interface.
DEBUG_TICK_TOPIC = "/obelisk/g1/debug/tick"
DEBUG_FM_PLAN_TOPIC = "/obelisk/g1/debug/fm_plan"

#: Reliable and deep enough to ride out a recorder hiccup: a dropped diagnostic sample is a
#: hole in the record of a fall, and the whole point is to have the record. Still bounded --
#: back-pressure must never reach the control tick.
DEBUG_QOS = QoSProfile(
    depth=50,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

#: ExecFSMState::USER_CTRL. Mirrors obelisk's unitree_fsm.h, where it is named
#: "LOW_LEVEL_CONTROL"; this is the ONLY state in which the policy may drive the robot.
EXEC_FSM_LOW_LEVEL_CONTROL = 3

#: Consecutive non-finite commands before the controller disarms rather than keep holding
#: the last good target. 10 ticks = 0.2 s at 50 Hz: long enough to ride out a one-off bad
#: estimate, short enough that a dead input does not leave the robot frozen mid-stride.
_NONFINITE_DISARM_TICKS = 10

#: Depth frames arrive on their own clock and are not part of the control graph; best
#: effort matches how they are produced and keeps a slow subscriber from stalling the camera.
SENSOR_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class G1Controller(ObeliskController):
    """Drives the G1 from the generator+tracker ONNX pair at 50 Hz."""

    def __init__(self, node_name: str = "g1_controller") -> None:
        """Load the policy, wire the sensors, and come up disarmed."""
        super().__init__(node_name, PDFeedForward, EstimatedState)

        self.register_obk_subscription("sub_vel_cmd", self.vel_cmd_callback, VelocityCommand)
        self.register_obk_subscription("sub_fsm_state", self.fsm_state_callback, ExecutionFSM)

        self._declare_params()
        self.policy = self._load_policy()
        self._setup_depth()
        self._setup_clip()

        # Latest estimate.
        self.q_base = np.array([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0])
        self.v_base = np.zeros(6)
        self.joint_pos = np.zeros(0)
        self.joint_vel = np.zeros(0)
        self.hw_joint_names: list[str] = []
        self.have_estimate = False
        # WHICH estimate a tick used. The estimator runs at 1 kHz against a 50 Hz tick, so
        # without this a recording cannot say which of twenty candidate samples fed a given
        # control step -- and that alignment is a prime suspect whenever sim and robot
        # disagree. Also the closest thing to the est_state watchdog this node still lacks.
        self._est_stamp = None
        self._est_time: float | None = None

        # Command, and the watchdogs over it.
        self.vel_cmd = np.zeros(3)
        self.last_vel_cmd_time: float | None = None
        self._vel_cmd_stale = False
        self._depth_stale = False

        # Arming / warm-up.
        self.armed = False
        self.hold_pose: np.ndarray | None = None
        self._policy_joint_idx: np.ndarray | None = None

        # Degenerate-quaternion guard: a dropped IMU frame publishes a zero quaternion,
        # which would otherwise yield a garbage gravity vector for that tick.
        self._last_good_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._bad_quat_count = 0
        self._quat_samples = 0

        # Non-finite-command guard (see _finite_targets).
        self._last_good_targets: np.ndarray | None = None
        self._nonfinite_streak = 0

        self._infer_ms: list[float] = []
        self._last_report = 0.0
        # Previous tick's clock and the gap it implies, for ControllerDebug.tick_period_ms --
        # the ROS timer's own jitter, which _record_latency deliberately does not measure (it
        # times the policy call, not the gap between calls). Advanced by compute_control, read
        # by _build_debug_msg; see the note there for why the builder must not touch it.
        self._last_tick_t: float | None = None
        self._tick_period_ms = 0.0

        # LAST, because it dry-runs a diagnostic message against every attribute above.
        self._setup_debug()
        # Loud, once, at bring-up: a config file is easy to launch out of habit, and the
        # difference between the two controllers is otherwise invisible in the log until
        # something is slow.
        self.get_logger().warn(
            "this is the DEPRECATED python controller. g1_control_cpp (config/g1_cpp.yml) is "
            "the supported one and the default in every run script; on aarch64 this node "
            "cannot reach the GPU at all. Clip playback is the only mode that still needs it."
        )
        self.get_logger().info("g1_controller up, DISARMED (waiting for LOW_LEVEL_CONTROL).")

    # ------------------------------------------------------------------ setup

    def _declare_params(self) -> None:
        """Declare and read every ROS parameter this node uses."""
        self.declare_parameter("hf_repo_id", "")
        self.declare_parameter("hf_policy_folder", "")
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("onnx_threads", 2)
        self.declare_parameter("depth_topics", [""])
        self.declare_parameter("depth_timeout", 0.25)
        self.declare_parameter("vel_cmd_timeout", 1.0)
        self.declare_parameter("verbose", False)

        # ---- Debug-clip playback (see the module docstring) ----
        # Empty path = the normal generator path. Anything else replaces the generator with
        # a baked clip, and the generator is then not loaded at all.
        self.declare_parameter("debug_clip", "")
        self.declare_parameter("debug_clip_case", "")
        self.declare_parameter("debug_clip_id", 0)
        self.declare_parameter("debug_clip_start_time", 0.0)
        # ON by default, unlike the MuJoCo runner. MuJoCo spawns the robot on the clip's
        # own step-0 pose; a real robot is wherever it is standing, so the clip has to be
        # re-placed onto it instead. Without this the reference starts metres away.
        # Seconds of clip to play before holding, 0 = the whole clip. The clip is the
        # command source here, so `run_sim_low.sh --duration` (which bounds the velocity
        # PROFILE node) does not reach this mode at all.
        self.declare_parameter("debug_clip_duration", 0.0)
        self.declare_parameter("debug_clip_ego_anchor", True)
        self.declare_parameter("debug_clip_ego_anchor_every", 0)

        # ---- Diagnostics ----
        # On by default: ~330 kB/s is nothing next to being unable to explain a fall, and a
        # run that was not recorded cannot be re-run. Turn it off only if the latency line
        # says publishing is eating the control period.
        self.declare_parameter("debug_topics", True)

        self.depth_timeout = float(self.get_parameter("depth_timeout").value)
        self.vel_cmd_timeout = float(self.get_parameter("vel_cmd_timeout").value)
        self.verbose = bool(self.get_parameter("verbose").value)
        self.debug_topics = bool(self.get_parameter("debug_topics").value)
        self.debug_clip_path = str(self.get_parameter("debug_clip").value)
        self.clip_duration = float(self.get_parameter("debug_clip_duration").value)
        self.clip_ego_anchor = bool(self.get_parameter("debug_clip_ego_anchor").value)
        # 0 = DERIVE from the tracker's own fm_update_every, resolved in _setup_clip once the
        # artifact is loaded. Under window_anchor='plan' the clip placement and the window
        # anchor must be solved from the robot on the same step, so this is a property of the
        # policy rather than a tuning knob -- see GenTrackerPolicy.assert_clip_reference_compatible.
        self.clip_ego_anchor_every = max(
            0, int(self.get_parameter("debug_clip_ego_anchor_every").value))

    def _load_policy(self):
        """Fetch and load the ONNX pair.

        Returns:
            The loaded :class:`~deploy.gen_tracker.GenTrackerPolicy`.

        Raises:
            RuntimeError: If the repo/folder parameters are unset, or the artifacts are
                neither cached nor reachable.
        """
        repo = str(self.get_parameter("hf_repo_id").value)
        folder = str(self.get_parameter("hf_policy_folder").value)
        if not repo or not folder or folder == "REPLACE_ME":
            raise RuntimeError(
                "hf_repo_id / hf_policy_folder must name the policy to run "
                f"(got {repo!r} / {folder!r}); set them in the controller config."
            )
        cache = HF_CACHE_DIR
        # `use_cuda` was declared and then read by nothing, so the config documented a GPU
        # option the node could not take. It now reaches onnxruntime; resolve_providers
        # appends CPU as the fallback, so a CUDA request on a CPU-only wheel raises at load
        # (loudly, at bring-up) rather than silently running on the CPU it promised to leave.
        want_cuda = bool(self.get_parameter("use_cuda").value)
        providers = ["cuda"] if want_cuda else None
        clip_mode = bool(self.debug_clip_path)
        # No cadence argument: the re-plan period comes from tracker.onnx's metadata, and
        # GenTrackerPolicy raises rather than falling back when an fm.onnx is loaded without
        # it. A ROS parameter here would read as a live knob that can never take effect.
        policy = load_gen_tracker_from_hf(
            repo, folder, cache,
            num_threads=int(self.get_parameter("onnx_threads").value),
            providers=providers,
            load_fm=not clip_mode,
        )
        active = policy.active_providers()
        if want_cuda and not clip_mode:
            self._require_cuda(active)
        elif want_cuda and clip_mode:
            # The CUDA refusal is about the GENERATOR's 100 ms CPU cost. Without it the
            # tracker alone runs in ~0.3 ms, so a clip run is perfectly usable on the CPU
            # and refusing would block the one mode that works on a box with no GPU.
            self.get_logger().info(
                "Debug-clip mode: not requiring CUDA. The generator is not loaded, and the "
                "tracker alone fits the control period on the CPU.")
        # A >1-channel generator unprojects its own depth image, which needs each camera's
        # pose in the root's yaw-aligned frame -- i.e. forward kinematics. This node has none
        # and will not grow any: it is deprecated, and the C++ controller already FKs the
        # cameras every tick (g1_deploy::CameraPoseFk). Refused HERE rather than one tick
        # after arming, where the same missing key surfaces as a KeyError from deep inside
        # the observation assembly.
        channels = 1
        if not clip_mode:
            channels = int(getattr(policy.ctx, "extero_channels", 1) or 1)
        if channels > 1:
            raise RuntimeError(
                f"this generator conditions on {channels} channels per pixel, so it needs "
                "camera forward kinematics, which the DEPRECATED python controller does not "
                "have. Launch config/g1_cpp.yml (the default in every run script) instead."
            )
        self.get_logger().info(
            f"Policy {repo}/{folder}: obs={policy.get_num_obs()} act={policy.get_num_actions()} "
            f"dt={policy.get_dt():.4f}s cameras={len(policy.get_depth_camera_spec())} "
            f"providers={active}"
            + (" [DEBUG CLIP: generator NOT loaded]" if clip_mode else "")
        )
        return policy

    def _require_cuda(self, active: dict[str, list[str]]) -> None:
        """Refuse to run on the CPU after asking for the GPU.

        ``resolve_providers`` only checks that the wheel WAS BUILT with a CUDA provider --
        a static property. It cannot see a session that was created and then fell back:
        missing CUDA runtime libraries, no nvidia container runtime, the no-GPU compose
        file, a driver too old for the CUDA minor. onnxruntime logs that and continues, so
        without this the robot comes up looking healthy and misses its control period by
        5x. Refusing is the right call because CPU is not a degraded mode for this pair, it
        is an unusable one (103.5 ms against a 20 ms budget).

        Args:
            active: Session name -> registered providers, from ``active_providers()``.

        Raises:
            RuntimeError: If any loaded session did not register the CUDA provider.
        """
        fell_back = {name: got for name, got in active.items()
                     if "CUDAExecutionProvider" not in got}
        if not fell_back:
            return
        raise RuntimeError(
            "use_cuda is true but onnxruntime put "
            f"{' and '.join(sorted(fell_back))} on the CPU: {fell_back}. The wheel "
            "advertises CUDA, so this is the runtime, not the install -- check that the "
            "container has the nvidia runtime (`nvidia-smi` inside it) and that the launch "
            "picked docker-compose.yml rather than docker-compose-no-gpu.yml. To run on "
            "the CPU deliberately, set use_cuda:=false and expect ~103 ms per generator "
            "tick against a 20 ms budget."
        )

    def _setup_depth(self) -> None:
        """Subscribe one topic per camera in the artifact's contract.

        Plain ROS subscriptions rather than obelisk-managed ones: the camera COUNT comes
        from the policy metadata at runtime, which the static key registration cannot
        express. The count is cross-checked against the contract, so a two-camera policy
        cannot be run against one topic.

        Raises:
            RuntimeError: If the configured topics do not match the contract.
        """
        self.depth_spec = self.policy.get_depth_camera_spec()
        topics = [t for t in self.get_parameter("depth_topics").value if t]
        if len(topics) != len(self.depth_spec):
            names = [c.get("sensor") or c["obs_term"] for c in self.depth_spec]
            raise RuntimeError(
                f"This policy reads {len(self.depth_spec)} depth camera(s) {names}, but "
                f"{len(topics)} depth_topics are configured {topics}. They must correspond "
                "one-to-one, in contract order."
            )
        self.depth_raw: list[np.ndarray | None] = [None] * len(self.depth_spec)
        self.last_depth_time: list[float | None] = [None] * len(self.depth_spec)
        # Created here, not in _setup_debug: depth_callback writes to these, and the
        # subscriptions below are live from this point on.
        self._depth_rate = [RateTracker() for _ in self.depth_spec]
        self._depth_subs = []
        for i, (topic, cam) in enumerate(zip(topics, self.depth_spec)):
            self._depth_subs.append(self.create_subscription(
                Image, topic, lambda msg, i=i: self.depth_callback(msg, i), SENSOR_QOS))
            self.get_logger().info(
                f"depth[{i}] '{cam.get('sensor') or cam['obs_term']}' "
                f"{int(cam['width'])}x{int(cam['height'])} <- {topic}")

    @staticmethod
    def _missing_clip_message(path: str) -> str:
        """Explain a missing clip library, naming the dangling-symlink case explicitly.

        "No such file" is the least useful true statement available here. The way this
        actually fails is that the path exists perfectly well on the host and is invisible
        inside the container: compose mounts only ``$REPO_ROOT``, and in a git worktree
        ``clips`` is a SYMLINK pointing OUT of that tree, so the link resolves to nothing
        once mounted. A real robot has no ``clips/`` directory at all. Both look identical
        to ``os.path.exists``, and neither is guessable from the plain message.

        Args:
            path: The absolute path that was not found.

        Returns:
            The error text.
        """
        base = f"debug_clip is set but there is no clip library at {path}."
        # A broken symlink anywhere along the path is the tell.
        probe = os.path.dirname(path)
        while probe and probe != os.path.dirname(probe):
            if os.path.islink(probe) and not os.path.exists(os.path.realpath(probe)):
                return (
                    f"{base}\n  '{probe}' is a SYMLINK to "
                    f"'{os.readlink(probe)}', which does not exist here. That directory is "
                    "outside the container's mount (compose mounts only $REPO_ROOT), so "
                    "the link dangles. Put the library INSIDE the repo -- "
                    "scripts/rsl_rl/make_debug_clips.py defaults to "
                    "transfer/obelisk/g1_control/resource/debug_clips/ for exactly this "
                    "reason."
                )
            probe = os.path.dirname(probe)
        return (
            f"{base}\n  Build it with:\n"
            "    PYTHONPATH=source/robot_rl:source/motion_gen \\\n"
            "        python scripts/rsl_rl/make_debug_clips.py\n"
            "  The path is resolved against the repo root, and must be INSIDE it: the "
            "container mounts only $REPO_ROOT, so anything reached through a symlink out "
            "of the tree is invisible here."
        )

    def _setup_clip(self) -> None:
        """Load the debug clip, if one is configured.

        Deliberately after ``_load_policy``: the clip has to be read in the tracker's joint
        order, ee-frame order and root-frame-body set, all of which come from the artifact
        metadata. Reading it in any other order would silently mis-assign joints.

        The import is local because ``deploy.clip_reference`` needs h5py, which a robot
        image only carries for this mode -- a normal FM run must not require it.

        Raises:
            RuntimeError: If the clip cannot be loaded, or its dt disagrees with the
                policy's. A dt mismatch would play the motion at the wrong speed, which
                looks like a tracking failure rather than a configuration error.
        """
        self._clip = None
        self._clip_step = 0
        self._clip_started = False
        self._clip_exhausted = False
        if not self.debug_clip_path:
            return

        from deploy.clip_reference import ClipReference, resolve_case

        path = self.debug_clip_path
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(_TRANSFER), path)
        if not os.path.exists(path):
            raise RuntimeError(self._missing_clip_message(path))

        case = str(self.get_parameter("debug_clip_case").value)
        clip_id = resolve_case(path, case) if case else int(
            self.get_parameter("debug_clip_id").value)
        self._clip = ClipReference(
            path,
            self.policy.get_joint_names(),
            list(self.policy.ctx.target_frame_names),
            self.policy.base_frame_name,
            clip_id=clip_id,
            start_time=float(self.get_parameter("debug_clip_start_time").value),
            ego_anchor=self.clip_ego_anchor,
            ego_anchor_every=self.clip_ego_anchor_every or self.policy.fm_update_every,
            root_frame_bodies=self.policy._trk.get_root_frame_bodies(),
        )
        # Whatever the cadence resolved to, it is the clip's now -- read it back rather than
        # keeping a second copy that could disagree with what is actually being served.
        self.clip_ego_anchor_every = self._clip.ego_anchor_every
        self.policy.assert_clip_reference_compatible(self._clip)
        # Last step the clip will PLAY. Beyond it the reference holds and the command is
        # zeroed -- see _tick_clip. Bounded by the clip's own length, so a --duration longer
        # than the clip is a no-op rather than an error.
        self._clip_stop_step = self._clip.real_steps - 1
        if self.clip_duration > 0.0:
            self._clip_stop_step = min(
                self._clip_stop_step,
                self._clip.start_step + int(round(self.clip_duration / self._clip.dt)))
        if abs(self._clip.dt - self.policy.get_dt()) > 1e-6:
            raise RuntimeError(
                f"clip dt {self._clip.dt} != policy dt {self.policy.get_dt()}: the motion "
                "would play at the wrong speed. Regenerate the clip at the control dt."
            )
        self.get_logger().info(
            f"DEBUG CLIP {os.path.basename(path)} "
            f"{'case ' + case + ' -> ' if case else ''}clip {clip_id}: "
            f"{self._clip.real_steps} steps ({self._clip.real_length_s:.2f}s), "
            f"start={self._clip.start_step * self._clip.dt:.2f}s, "
            f"anchor={'ego/every %d' % self.clip_ego_anchor_every if self.clip_ego_anchor else 'world'}"
            f" ({self.policy.window_anchor}). "
            "The generator is NOT in this loop."
        )

    def _setup_debug(self) -> None:
        """Create the diagnostic publishers and the per-camera rate trackers.

        Plain rclpy publishers rather than obelisk-managed ones: the camera COUNT is only
        known once the policy metadata is read, and these are first-party debug output that
        is not part of any node's interface contract.
        """
        self._debug_pub = None
        self._fm_plan_pub = None
        self._last_plan_id = 0
        if not self.debug_topics:
            self.get_logger().warn(
                "debug_topics is FALSE: this run records nothing beyond log lines.")
            return
        self._debug_pub = self.create_publisher(ControllerDebug, DEBUG_TICK_TOPIC, DEBUG_QOS)
        self._fm_plan_pub = self.create_publisher(FmPlan, DEBUG_FM_PLAN_TOPIC, DEBUG_QOS)
        self._assert_debug_msg_current()
        self.get_logger().info(
            f"Diagnostics on: {DEBUG_TICK_TOPIC} per tick, {DEBUG_FM_PLAN_TOPIC} per re-plan.")

    def _assert_debug_msg_current(self) -> None:
        """Fail at bring-up if the built g1_msgs is older than this file.

        Adding a field to a ``.msg`` needs g1_msgs REBUILT; colcon --symlink-install
        symlinks Python sources but message bindings are generated, so editing the .msg and
        rebuilding only g1_control leaves a stale ``ControllerDebug``. The mismatch then
        surfaces as an AttributeError inside the 50 Hz tick, which ``_publish_debug``
        catches so a diagnostics bug cannot stop the robot -- correct, but it means the run
        completes having recorded NOTHING while printing one throttled line every 5 s.

        Cheaper to find it here: build the message once against the same code the tick uses.

        Raises:
            RuntimeError: If a field this node sets does not exist on the built message.
        """
        try:
            self._build_debug_msg(None, None)
        except AttributeError as exc:
            raise RuntimeError(
                f"the built g1_msgs is out of date with this controller ({exc}). Message "
                "bindings are GENERATED, so a .msg change needs the package rebuilt:\n"
                "    rm -rf build/g1_msgs install/g1_msgs && scripts/build.sh\n"
                "Refusing to start: the run would look healthy and record nothing."
            ) from exc

    def _publish_debug(self, state: dict, obs: np.ndarray) -> None:
        """Emit this tick's diagnostic record, and the generator's plan if it just changed.

        Everything here is either unavailable outside the process or unprovable without it;
        anything already on the network (proprioception, raw depth, the published command)
        is deliberately NOT repeated. Wrapped in a broad except because a diagnostics bug
        must never be able to stop the robot -- losing the record is bad, losing control is
        worse.

        Args:
            state: This tick's state dict, or None when the tick did not get far enough to
                build one (disarmed, no estimate yet, or waiting on a first depth frame).
            obs: The assembled observation, or None when no policy call was made.
        """
        if self._debug_pub is None:
            return
        try:
            self._debug_pub.publish(self._build_debug_msg(state, obs))
            plan = self.policy.last_plan()
            if plan is not None and plan["plan_id"] != self._last_plan_id:
                self._last_plan_id = plan["plan_id"]
                self._fm_plan_pub.publish(self._build_fm_plan_msg(plan))
        except Exception as exc:                                    # noqa: BLE001
            self.get_logger().error(f"diagnostics publish failed: {exc}",
                                    throttle_duration_sec=5.0)

    def _build_debug_msg(self, state: dict, obs: np.ndarray | None) -> ControllerDebug:
        """Fill one :class:`ControllerDebug`.

        Args:
            state: This tick's state dict, or None when none was built (see
                :meth:`_publish_debug`).
            obs: The assembled observation, or None when no policy call was made.

        Returns:
            The message to publish.
        """
        m = ControllerDebug()
        m.header.stamp = self.get_clock().now().to_msg()
        m.ctrl_step = max(0, int(getattr(self.policy, "_ctrl_step", 0)))
        m.latency_ms = self._infer_ms[-1] if self._infer_ms else 0.0

        # The two networks separately -- see the .msg comments for why one total will not do.
        # Non-destructive, unlike pop_avg_infer_ms, which the 5 s report line drains.
        split = self.policy.last_infer_ms()
        m.fm_infer_ms = float(split["fm_ms"])
        m.trk_infer_ms = float(split["trk_ms"])
        # Read-only here on purpose. `_assert_debug_msg_current` dry-runs this whole builder
        # during __init__ to prove the installed g1_msgs is current, so a `self._last_tick_t =
        # self.t` side effect in here would be stamped at CONSTRUCTION time -- before the
        # control timer exists -- and the first real tick would report the entire
        # configure-to-activate gap as its control period. `compute_control` advances it.
        m.tick_period_ms = self._tick_period_ms
        # The schedule-health counters have no python equivalent: this runtime samples the
        # generator SYNCHRONOUSLY inside the tick, so there is no mailbox to drop a plan, no
        # busy worker to skip a request, and a plan cannot be late for its own boundary. They
        # are left at zero rather than faked, and plot_from_bag reads them only when nonzero.

        # Provenance: WHICH sample, not its value. See the .msg comments.
        if self._est_stamp is not None:
            m.est_state_stamp = self._est_stamp
        m.est_state_age = (self.t - self._est_time) if self._est_time is not None else -1.0
        m.depth_age = [(self.t - t) if t is not None else -1.0 for t in self.last_depth_time]
        m.vel_cmd_age = ((self.t - self.last_vel_cmd_time)
                         if self.last_vel_cmd_time is not None else -1.0)

        m.command_effective = (np.asarray(state["command"], dtype=np.float64).tolist()
                               if state is not None else [0.0, 0.0, 0.0])
        m.joint_names_isaac = list(self.policy.get_joint_names())
        m.action_isaac = np.asarray(
            getattr(self.policy._trk, "action_isaac", np.zeros(0)), dtype=np.float64).tolist()

        spec = self.policy.reference_window_spec
        m.reference_blocks = [str(b) for b in self.policy.reference_blocks]
        m.reference_offsets = [int(o) for o in self.policy.window_offsets]
        m.reference_node_dim = int(sum(int(w) for _, w in spec["blocks"]))
        served = self.policy.last_reference_window()
        m.reference_window = np.asarray(served, dtype=np.float64).tolist() \
            if served is not None else []

        plan = self.policy.last_plan()
        m.fm_plan_id = int(plan["plan_id"]) if plan is not None else 0
        m.fm_cursor = int(self.policy.served_node_index) if plan is not None else -1
        m.fm_anchor_pos = np.asarray(self.policy._anchor_pos, dtype=np.float64).tolist()
        m.fm_anchor_quat = np.asarray(self.policy._anchor_quat, dtype=np.float64).tolist()

        m.clip_mode = self._clip is not None
        m.clip_step = int(self._clip_step) if self._clip is not None else -1

        m.armed = bool(self.armed)
        m.depth_stale = bool(self._depth_stale)
        m.vel_cmd_stale = bool(self._vel_cmd_stale)
        m.nonfinite_streak = int(self._nonfinite_streak)

        hz, jit, dmin, dmax, dmean, dinv = [], [], [], [], [], []
        for i, cam in enumerate(self.depth_spec):
            hz.append(self._depth_rate[i].hz())
            jit.append(self._depth_rate[i].jitter())
            raw = self.depth_raw[i]
            h = depth_health(cam, raw) if raw is not None else {
                "min": float("nan"), "max": float("nan"), "mean": float("nan"),
                "invalid_fraction": 1.0}
            dmin.append(h["min"]); dmax.append(h["max"])
            dmean.append(h["mean"]); dinv.append(h["invalid_fraction"])
        m.depth_hz, m.depth_jitter = hz, jit
        m.depth_min, m.depth_max, m.depth_mean, m.depth_invalid_fraction = dmin, dmax, dmean, dinv

        m.obs_checksum = obs_checksum(obs) if obs is not None else 0
        m.obs_size = int(np.size(obs)) if obs is not None else 0
        return m

    def _build_fm_plan_msg(self, plan: dict) -> FmPlan:
        """Fill one :class:`FmPlan` from the generator's last sample.

        Args:
            plan: ``GenTrackerPolicy.last_plan()``.

        Returns:
            The message to publish.
        """
        flat = lambda a: np.asarray(a, dtype=np.float64).ravel().tolist()  # noqa: E731
        m = FmPlan()
        m.header.stamp = self.get_clock().now().to_msg()
        m.plan_id = int(plan["plan_id"])
        m.ctrl_step = max(0, int(plan["ctrl_step"]))
        window = self.policy.predicted_window()
        m.trajectory = flat(window) if window is not None else []
        m.num_nodes = int(self.policy.ctx.F)
        m.node_dim = int(self.policy.ctx.D)
        m.window_blocks = [str(b) for b in self.policy.reference_blocks]
        # So a viewer can FK the predicted joints without restating a tracker contract.
        m.ee_frame_names = [str(n) for n in self.policy.ctx.target_frame_names]
        m.anchor_pos = flat(self.policy._anchor_pos)
        m.anchor_quat = flat(self.policy._anchor_quat)
        m.history_nodes = flat(plan["history_nodes"])
        m.history_cmd = flat(plan["history_cmd"])
        m.history_length = int(self.policy.ctx.N)
        m.history_null = bool(plan["history_null"])
        m.extero_tokens = flat(plan["extero_tokens"])
        m.extero_source = str(self.policy.ctx.extero_source)
        m.cmd_tokens = flat(plan["cmd_tokens"])
        m.cmd_dim = int(self.policy.ctx.cmd_dim)
        m.infer_ms = float(plan["infer_s"]) * 1e3
        return m

    # -------------------------------------------------------------- callbacks

    def update_x_hat(self, msg: EstimatedState) -> None:
        """Latch the state estimate.

        Args:
            msg: Fused estimate from ``g1_estimator``.
        """
        self.q_base = np.asarray(msg.q_base, dtype=np.float64)
        self.v_base = np.asarray(msg.v_base, dtype=np.float64)
        self.joint_pos = np.asarray(msg.q_joints, dtype=np.float64)
        self.joint_vel = np.asarray(msg.v_joints, dtype=np.float64)
        self._est_stamp = msg.header.stamp
        self._est_time = self.t
        if list(msg.joint_names) != self.hw_joint_names:
            self.hw_joint_names = list(msg.joint_names)
            self._resolve_joint_mapping()
        self.have_estimate = True

    def vel_cmd_callback(self, msg: VelocityCommand) -> None:
        """Latch the commanded planar velocity.

        Args:
            msg: Velocity command from the joystick FSM (or a high-level node).
        """
        self.vel_cmd = np.array([msg.v_x, msg.v_y, msg.w_z], dtype=np.float64)
        self.last_vel_cmd_time = self.t

    def depth_callback(self, msg: Image, index: int) -> None:
        """Latch one camera's raw depth frame.

        Non-finite pixels need no guard here: ``normalize_depth_np`` maps NaN/inf to
        ``max_clip_val``, so depth cannot be the source of a non-finite command (see
        ``_finite_targets``, which backstops everything that is not so protected).

Stream health -- measured Hz, inter-arrival jitter, range and invalid-pixel fraction --
        goes out on ``/obelisk/g1/debug/tick`` (see ``deploy.diagnostics``), because
        staleness alone cannot tell a degraded camera from a healthy one: an all-invalid
        frame, a wrong exposure or a half-saturated image all arrive perfectly on time.

        TODO(depth-validation): ``msg.encoding`` / ``width`` / ``height`` / ``step`` are
        still unvalidated -- a wrong-sized frame raises out of ``normalize_camera_images``
        inside the 50 Hz timer and kills the node.

        Args:
            msg: 32FC1 depth image, row-major, in METRES along the optical axis.
            index: Which camera, in contract order.
        """
        self.depth_raw[index] = np.frombuffer(msg.data, dtype=np.float32).copy()
        # Local receipt time on purpose: the camera's clock is not synchronised to this
        # machine, so a header stamp cannot tell us whether the stream is still alive.
        self.last_depth_time[index] = self.t
        self._depth_rate[index].tick(self.t)

    def fsm_state_callback(self, msg: ExecutionFSM) -> None:
        """Arm or disarm from the robot's ACTUAL execution-FSM state.

        Args:
            msg: Execution FSM state, published on every transition plus a heartbeat.
        """
        in_low_level = msg.cmd_exec_fsm_state == EXEC_FSM_LOW_LEVEL_CONTROL
        if in_low_level and not self.armed:
            self._arm()
        elif not in_low_level and self.armed:
            self.armed = False
            self.get_logger().warn(
                f"DISARMED: execution FSM left LOW_LEVEL_CONTROL (now {msg.cmd_exec_fsm_state}).")

    # ------------------------------------------------------------------ arming

    def _resolve_joint_mapping(self) -> None:
        """Map the policy's joint order onto the hardware's, requiring an exact match.

        Raises:
            RuntimeError: If the two joint SETS differ at all. A policy that does not
                command every motor leaves the rest to be filled from somewhere, and every
                choice of "somewhere" is a way to move a joint nobody asked to move.
        """
        policy_joints = self.policy.get_joint_names()
        missing = [j for j in self.hw_joint_names if j not in policy_joints]
        extra = [j for j in policy_joints if j not in self.hw_joint_names]
        if missing or extra:
            raise RuntimeError(
                "Policy and robot joint sets differ, so this policy cannot drive this "
                f"robot.\n  commanded by the policy but absent on the robot: {extra}\n"
                f"  present on the robot but not commanded: {missing}\n"
                f"  ({len(policy_joints)} policy joints vs {len(self.hw_joint_names)} robot "
                "joints -- a 29-DoF robot needs `fixed_waist: false` AND a 29-DoF policy.)"
            )
        # Position of each hardware joint within the policy's ordering.
        self._policy_joint_idx = np.array([policy_joints.index(j) for j in self.hw_joint_names])
        self.get_logger().info(f"Joint mapping resolved for {len(self.hw_joint_names)} joints.")

    def _arm(self) -> None:
        """Enter low-level control: reset the policy and begin the generator warm-up."""
        if not self.have_estimate:
            self.get_logger().error("Refusing to arm: no state estimate yet.")
            return
        if self._policy_joint_idx is None:
            self.get_logger().error("Refusing to arm: joint mapping unresolved.")
            return
        self.policy.reset()
        # Kept only as the non-finite guard's last-resort fallback (see _finite_targets);
        # nothing holds this pose deliberately.
        self.hold_pose = self.joint_pos.copy()
        self.armed = True
        self._last_good_targets = None
        self._nonfinite_streak = 0
        if self._clip is not None:
            self._clip_step = self._clip.start_step
            self._clip_started = False
            self._clip_exhausted = False
        self.get_logger().info(
            "ARMED. The generator drives the tracker from this tick."
            if self._clip is None else
            f"ARMED (DEBUG CLIP). The clip drives the tracker from this tick, step "
            f"{self._clip_step}.")

    def _start_clip(self, state: dict) -> None:
        """Place the clip onto the robot and report the offset the tracker must close.

        Called once, at the warm-up -> clip handover. The robot cannot be teleported onto
        the clip's own pose the way the MuJoCo runner spawns it, so the clip is re-placed
        onto the robot instead (planar position, heading and height). What re-placement
        CANNOT fix is the joint-space difference between the clip's posture and the pose
        the robot happens to be holding -- so that offset is measured and logged rather
        than blended away. A large value here is the finding, not a nuisance: it means the
        robot is being asked to step to a posture it is nowhere near, and every case in the
        debug library starts from a settled stand precisely so that it should be small.

        Args:
            state: This tick's state dict, whose ``joint_pos`` is already in the policy's
                joint order (``build_hardware_state`` converts it).
        """
        if self._clip is None:
            return
        s = min(self._clip_step, self._clip.real_steps - 1)
        if self.clip_ego_anchor:
            self._clip.reanchor_ego(s, state["root_pos"], state["root_quat"])
        err = np.asarray(self._clip.state_at(s)["jp"], dtype=np.float64) - state["joint_pos"]
        worst = int(np.argmax(np.abs(err)))
        self.get_logger().info(
            f"CLIP HANDOVER at step {s}: joint offset |e|={np.linalg.norm(err):.3f} rad "
            f"over {err.size} joints, max {err[worst]:+.3f} on "
            f"{self.policy.get_joint_names()[worst]}. The tracker has to close this."
        )

    # --------------------------------------------------------------- watchdogs
    #
    # TODO(est-watchdog): there is still NO est_state watchdog. Every other input has a
    # timeout; if the estimator or the driver dies, the controller keeps commanding at
    # 50 Hz off frozen arrays. `update_x_hat` now records a receipt time and every tick
    # publishes `est_state_age`, so the failure is at least VISIBLE in a recording -- but
    # nothing acts on it. Same for the edge-triggered dropout warnings: the bag now carries
    # the latch state per tick, which recovers the "has it been degraded for 90 s?" question
    # after the fact, but a periodic "degraded" heartbeat is still the right live answer.
    # Related, in estimator.py: odometry staleness is known there and structurally
    # invisible here.

    def _depth_is_stale(self) -> bool:
        """Whether any camera has stopped delivering.

        Returns:
            True while a camera is overdue. Inert before its first frame -- never having
            arrived is handled by refusing to step, not by the watchdog.
        """
        if not self.depth_spec or self.depth_timeout <= 0.0:
            return False
        return any(t is not None and (self.t - t) > self.depth_timeout for t in self.last_depth_time)

    def _vel_cmd_is_stale(self) -> bool:
        """Whether the velocity command has gone quiet.

        Returns:
            True when nothing has arrived within ``vel_cmd_timeout``.
        """
        if self.vel_cmd_timeout <= 0.0 or self.last_vel_cmd_time is None:
            return False
        return (self.t - self.last_vel_cmd_time) > self.vel_cmd_timeout

    def _update_watchdogs(self, check_vel_cmd: bool = True) -> bool:
        """Refresh the watchdog latches, logging every edge.

        Split out of :meth:`_effective_command` because debug-clip mode needs the same
        latches and the same log lines but a different response to them -- it freezes the
        reference rather than zeroing a command it does not use.

        Args:
            check_vel_cmd: Include the velocity-command watchdog. False in clip mode,
                where the clip supplies the command and nothing publishes one.

        Returns:
            True while any consulted watchdog is latched.
        """
        depth_stale = self._depth_is_stale()
        if depth_stale != self._depth_stale:
            self._depth_stale = depth_stale
            log = self.get_logger().warn if depth_stale else self.get_logger().info
            log("DEPTH DROPOUT: the policy is running on a frozen image."
                if depth_stale else "Depth recovered.")
        if not check_vel_cmd:
            return depth_stale
        cmd_stale = self._vel_cmd_is_stale()
        if cmd_stale != self._vel_cmd_stale:
            self._vel_cmd_stale = cmd_stale
            log = self.get_logger().warn if cmd_stale else self.get_logger().info
            log("VELOCITY COMMAND DROPOUT: commanding zero." if cmd_stale else "Command recovered.")
        return depth_stale or cmd_stale

    def _effective_command(self) -> np.ndarray:
        """The velocity command actually handed to the policy, after the watchdogs.

        Returns:
            The commanded velocity, or zeros while a sensor or the command is stale. The
            policy keeps running either way -- it still has to balance -- it is just told
            to stand still.
        """
        return np.zeros(3) if self._update_watchdogs() else self.vel_cmd

    def _base_quat(self) -> np.ndarray:
        """The base orientation, guarded against a degenerate IMU reading.

        Returns:
            A unit quaternion (xyzw). A near-zero quaternion -- what a dropped IMU frame
            looks like -- reuses the last good one rather than producing a garbage gravity
            vector for that tick.
        """
        quat = self.q_base[3:7]
        self._quat_samples += 1
        if np.linalg.norm(quat) < 0.5:
            self._bad_quat_count += 1
            if self._bad_quat_count % 50 == 1:
                pct = 100.0 * self._bad_quat_count / max(1, self._quat_samples)
                self.get_logger().warn(
                    f"Degenerate base quaternion ({pct:.1f}% of samples); reusing the last good one.")
            return self._last_good_quat
        self._last_good_quat = quat
        return quat

    # ------------------------------------------------------------ control loop

    def _build_state(self, command: np.ndarray | None = None) -> dict:
        """Assemble the policy's state dict from the latest readings.

        Args:
            command: Velocity command to use instead of the live one. Debug-clip mode
                passes the clip's own authored ``target_vel``, which is what the reference
                was generated against -- a joystick command would disagree with the motion
                the robot is being asked to track.

        Returns:
            The state dict for ``GenTrackerPolicy``.
        """
        return build_hardware_state(
            self.policy,
            root_pos=self.q_base[0:3],
            base_quat_xyzw=self._base_quat(),
            # Already rotated into the world frame by the estimator.
            base_lin_vel_w=self.v_base[0:3],
            base_ang_vel_b=self.v_base[3:6],
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
            joint_names=self.hw_joint_names,
            command=self._effective_command() if command is None else command,
            depth_raw=self.depth_raw if self.depth_spec else None,
        )

    def compute_control(self) -> PDFeedForward | None:
        """The 50 Hz control tick.

        Returns:
            The published command, or ``None`` when disarmed, not yet ready, or the
            command was not finite -- in which case NOTHING is published and the robot
            interface keeps holding its pose.
        """
        # A tick is recorded even when nothing is commanded. Otherwise "disarmed, waiting
        # for the FSM" and "the node died" look identical in a bag -- both are simply an
        # absent topic -- and the first is the normal state of a stack nobody has armed yet.
        if not self.armed or not self.have_estimate:
            # The tick clock stops with the loop, so a re-arm does not charge the whole
            # disarmed interval to the first armed tick's control period.
            self._last_tick_t = None
            self._tick_period_ms = 0.0
            self._publish_debug(None, None)
            return None

        # Sampled at the TOP, so the guards below that return early still count as ticks that
        # consumed a period -- and here rather than in _build_debug_msg, which is dry-run once
        # at construction and so must have no side effects.
        self._tick_period_ms = ((self.t - self._last_tick_t) * 1e3
                                if self._last_tick_t is not None else 0.0)
        self._last_tick_t = self.t
        if self.depth_spec and any(d is None for d in self.depth_raw):
            self.get_logger().warn("Waiting for the first frame from every depth camera...",
                                   throttle_duration_sec=2.0)
            self._publish_debug(None, None)
            return None

        if self._clip is not None:
            return self._clip_tick()

        state = self._build_state()
        t0 = time.perf_counter()
        obs = self.policy.create_obs(state)
        action = self.policy.get_action(obs, self.hw_joint_names)
        self._record_latency((time.perf_counter() - t0) * 1e3)
        msg = self._publish(action)
        # AFTER publishing: the command goes out first, diagnostics second, so a slow
        # recorder can never delay the robot's own command.
        self._publish_debug(state, obs)
        return msg

    def _clip_tick(self) -> PDFeedForward | None:
        """One control step driven by the baked clip instead of the generator.

        Mirrors ``gen_tracker_runner.run_clip_loop`` so the MuJoCo run and this one differ
        in the ROS plumbing and nothing else -- which is the whole point of the mode.

        A latched watchdog FREEZES the clip rather than zeroing a velocity command. In
        generator mode zeroing the command makes the robot stand still; here the command
        does not drive anything (the reference does), so the equivalent safe action is to
        stop advancing the reference and let the robot hold the pose it is tracking.

        Returns:
            The published command, or None (see :meth:`_publish`).
        """
        s = min(self._clip_step, self._clip_stop_step)
        # Once the reference stops advancing the command must go to ZERO. Playback saturates
        # on its last played frame, and `command_at` there returns whatever that frame was
        # authored with -- so a clip that ends mid-walk otherwise holds a walk command
        # against a stationary reference. Measured: a backward clip held -0.94 m/s for 16 s
        # after its last frame and put the robot down, which reads as a policy failure and
        # is not one.
        held = self._clip_step >= self._clip_stop_step
        command = ([0.0, 0.0, 0.0] if held else self._clip.command_at(s))
        state = self._build_state(command=command)
        if not self._clip_started:
            self._clip_started = True
            self._start_clip(state)

        t0 = time.perf_counter()
        # ONE number for both decisions: the clip's own elapsed tick, which the depth
        # watchdog freezes along with the reference. The policy latches its window anchor on
        # the same number, so under window_anchor='plan' the placement and the anchor are
        # always solved from the robot at the same instant -- which is what keeps the served
        # window a pure function of the clip. A separate counter here drifted by exactly the
        # length of any freeze, silently.
        clip_tick = self._clip_step - self._clip.start_step
        if self.clip_ego_anchor and clip_tick % self.clip_ego_anchor_every == 0:
            self._clip.reanchor_ego(s, state["root_pos"], state["root_quat"])
        obs = self.policy.create_obs_from_reference(
            state, self._clip.reference_at(s, self.policy.window_offsets), clip_tick=clip_tick)
        action = self.policy.get_action(obs, self.hw_joint_names)
        self._record_latency((time.perf_counter() - t0) * 1e3)

        # Depth only. The velocity-command watchdog is meaningless here -- the clip supplies
        # the command and nothing has to publish one -- and honouring it would let an absent
        # joystick freeze the reference forever.
        frozen = self._update_watchdogs(check_vel_cmd=False)
        if self._clip_step >= self._clip_stop_step:
            if not self._clip_exhausted:
                self._clip_exhausted = True
                self._report_clip_exhausted(s)
        elif not frozen:
            self._clip_step += 1
        msg = self._publish(action)
        self._publish_debug(state, obs)
        return msg

    def _report_clip_exhausted(self, step: int) -> None:
        """Say, once, that the clip ran out -- and whether its last frame is safe to hold.

        Playback saturates on the final real frame rather than stopping, so the robot keeps
        being told to do whatever that frame was. For the ``stand`` and ``stand_walk_stand``
        cases that is a settled stand and the robot can sit there indefinitely. It is NOT
        true in general: the ``terrain`` case ends mid-stride partway up a staircase, and
        holding a stride pose there is not a state to leave a robot in. So this reports what
        the final frame actually is instead of assuming, and escalates to a warning when it
        is not a stand.

        Args:
            step: The clip step now being held.
        """
        speed = float(np.linalg.norm(self._clip.command_at(step)[:2]))
        where = f"CLIP EXHAUSTED at step {step} ({step * self._clip.dt:.2f}s), holding its final frame"
        if speed <= 0.05:
            self.get_logger().info(
                f"{where}. It is a stand, so this is a safe state to sit in -- disarm when "
                "you have what you need.")
        else:
            self.get_logger().warn(
                f"{where}, which is commanded at {speed:.2f} m/s -- the motion ends "
                "mid-stride, so the robot is now holding a pose it was passing through. "
                "DISARM.")

    def _diagnose_nonfinite(self) -> str:
        """Name the latched inputs that are non-finite, for the guard's error message.

        Only ever called on the failure path, so the cost does not matter -- but knowing
        whether the NaN entered through the joints, the base pose or the base velocity is
        the difference between "the estimator died" and "the policy diverged".

        Returns:
            A comma-separated list of bad inputs, or ``"none (the policy itself diverged)"``
            when every input is clean.
        """
        checks = {
            "q_base (base pos+quat)": self.q_base, "v_base (base lin+ang vel)": self.v_base,
            "joint_pos": self.joint_pos, "joint_vel": self.joint_vel,
            "vel_cmd": self.vel_cmd,
        }
        bad = [name for name, arr in checks.items()
               if arr is not None and not np.isfinite(np.asarray(arr, dtype=np.float64)).all()]
        # Depth NaNs are already absorbed by normalize_depth_np, so a camera can only
        # contribute by going ENTIRELY invalid -- worth naming, but it is not the NaN source.
        if any(d is not None and not np.isfinite(d).any() for d in getattr(self, "depth_raw", [])):
            bad.append("depth (an all-invalid frame)")
        return ", ".join(bad) if bad else "none (the policy itself diverged)"

    def _finite_targets(self, joint_targets: np.ndarray) -> np.ndarray | None:
        """Gate the command on being finite, substituting the last good one if it is not.

        Nothing upstream validates proprioception: a NaN in ``msg.q_joints`` survives
        ``update_x_hat``, ``build_hardware_state`` and the ``np.clip`` in ``_assemble``
        (which propagates NaN unchanged), and would otherwise land in ``msg.pos_target`` on
        a live robot. Depth is already safe -- ``normalize_depth_np`` maps NaN/inf to
        ``max_clip_val`` -- so this is the backstop for everything else.

        The fallback is the PREVIOUS tick's command rather than ``hold_pose``: it is the
        nearest thing to what the robot is already tracking, whereas the arm-time pose can
        be far away and would command a jump. A sustained fault escalates to a disarm
        rather than freezing on a stale target forever, since holding one mid-stride is not
        a safe state to sit in indefinitely.

        Args:
            joint_targets: Candidate targets, hardware order.

        Returns:
            Finite targets to publish, or None to publish nothing (the robot interface
            then holds its pose -- the same passive state a disarm produces).
        """
        if joint_targets is not None and np.isfinite(joint_targets).all():
            self._last_good_targets = np.asarray(joint_targets, dtype=np.float64).copy()
            self._nonfinite_streak = 0
            return joint_targets

        self._nonfinite_streak += 1
        fallback = self._last_good_targets if self._last_good_targets is not None else self.hold_pose
        self.get_logger().error(
            f"NON-FINITE COMMAND (tick {self._nonfinite_streak} of a run): refusing to "
            f"publish it. Non-finite inputs: {self._diagnose_nonfinite()}. "
            f"Holding {'the last good target' if self._last_good_targets is not None else 'the arm-time pose'}.",
            throttle_duration_sec=1.0,
        )
        if self._nonfinite_streak >= _NONFINITE_DISARM_TICKS:
            self.get_logger().error(
                f"DISARMING: {self._nonfinite_streak} consecutive non-finite commands "
                f"({self._nonfinite_streak * self.policy.get_dt():.2f}s). Fix the input "
                "before re-entering LOW_LEVEL_CONTROL."
            )
            self.armed = False
            return None
        if fallback is None or not np.isfinite(fallback).all():
            return None
        return fallback

    def _publish(self, joint_targets: np.ndarray) -> PDFeedForward | None:
        """Send PD position targets in the hardware's joint order.

        Always the policy's gains. There is no second gain regime -- the controller only
        ever publishes the policy's output, so a stiffened-hold scale would have nothing to
        apply to.

        Args:
            joint_targets: Target joint positions, hardware order.

        Returns:
            The published message, or None when the command was not finite and no finite
            fallback was available (nothing is published; the interface holds pose).
        """
        joint_targets = self._finite_targets(joint_targets)
        if joint_targets is None:
            return None
        kp = np.asarray(self.policy.get_kp(), dtype=np.float64)
        kd = np.asarray(self.policy.get_kd(), dtype=np.float64)
        # Metadata gains are in the POLICY's joint order; the robot wants its own.
        kp, kd = kp[self._policy_joint_idx], kd[self._policy_joint_idx]

        msg = PDFeedForward()
        zeros = np.zeros_like(joint_targets)
        msg.pos_target = joint_targets.tolist()
        msg.vel_target = zeros.tolist()
        msg.feed_forward = zeros.tolist()
        msg.u_mujoco = np.concatenate([joint_targets, zeros, zeros]).tolist()
        msg.joint_names = self.hw_joint_names
        msg.kp = kp.tolist()
        msg.kd = kd.tolist()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.obk_publishers["pub_ctrl"].publish(msg)
        return msg

    def _record_latency(self, ms: float) -> None:
        """Track inference latency and report percentiles periodically.

        Two ONNX graphs run here and the generator's cadence makes the cost uneven tick to
        tick, so a mean would hide exactly the spikes that break a 50 Hz loop.

        This covers create_obs + get_action only. The actual timer PERIOD and its jitter
        are recoverable from the debug tick timestamps (``plot_from_bag.plot_health`` plots
        both against the budget), and ``est_state_age`` says how stale the estimate was --
        neither is measured here, on purpose: this line exists to warn live, and the bag is
        where a run gets analysed. ``transfer/deploy/benchmark.py`` reports the same
        percentiles off-robot, so the two are directly comparable.

        Args:
            ms: Wall time of this tick's policy call, in milliseconds.
        """
        self._infer_ms.append(ms)
        if self.t - self._last_report < 5.0:
            return
        self._last_report = self.t
        arr = np.asarray(self._infer_ms)
        self._infer_ms.clear()
        budget = self.policy.get_dt() * 1e3
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        split = self.policy.pop_avg_infer_ms()
        line = (f"policy latency p50={p50:.1f} p95={p95:.1f} p99={p99:.1f} ms "
                f"(budget {budget:.1f}) fm={split.get('fm_ms', 0.0):.1f} "
                f"trk={split.get('trk_ms', 0.0):.1f}")
        if p99 > budget:
            self.get_logger().warn(f"OVER BUDGET: {line}")
        elif self.verbose:
            self.get_logger().info(line)


def main(args: list | None = None) -> None:
    """Spin the controller.

    A multi-threaded executor so the depth callbacks are not serialised behind the 50 Hz
    control tick; every piece of shared state is latched as a whole array, never read
    half-updated.
    """
    spin_obelisk(args, G1Controller, MultiThreadedExecutor)


if __name__ == "__main__":
    main()
