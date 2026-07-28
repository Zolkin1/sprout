# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 low-level controller: the flow-matching generator + RL tracker, on hardware.

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
* **The generator is primed before it is trusted.** Its history tokens are measured robot
  state, so on arming the controller holds the standing pose on stiffened gains for
  ``warmup_time``, feeding the history, before letting the policy drive.
* **Sensors are watchdogged.** A stale depth frame or velocity command zeroes the command
  rather than letting the robot keep walking on old data.
* **The joint sets must match exactly.** A policy that does not command every hardware
  joint is refused, rather than the remainder being filled from a guess.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
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
_TRANSFER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _TRANSFER not in sys.path:
    sys.path.insert(0, _TRANSFER)

from deploy.hardware import build_hardware_state, load_gen_tracker_from_hf  # noqa: E402

#: ExecFSMState::USER_CTRL. Mirrors obelisk's unitree_fsm.h, where it is named
#: "LOW_LEVEL_CONTROL"; this is the ONLY state in which the policy may drive the robot.
EXEC_FSM_LOW_LEVEL_CONTROL = 3

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

        # Latest estimate.
        self.q_base = np.array([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0])
        self.v_base = np.zeros(6)
        self.joint_pos = np.zeros(0)
        self.joint_vel = np.zeros(0)
        self.hw_joint_names: list[str] = []
        self.have_estimate = False

        # Command, and the watchdogs over it.
        self.vel_cmd = np.zeros(3)
        self.last_vel_cmd_time: float | None = None
        self._vel_cmd_stale = False
        self._depth_stale = False

        # Arming / warm-up.
        self.armed = False
        self.arm_time: float | None = None
        self.hold_pose: np.ndarray | None = None
        self._policy_joint_idx: np.ndarray | None = None

        # Degenerate-quaternion guard: a dropped IMU frame publishes a zero quaternion,
        # which would otherwise yield a garbage gravity vector for that tick.
        self._last_good_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._bad_quat_count = 0
        self._quat_samples = 0

        self._infer_ms: list[float] = []
        self._last_report = 0.0
        self.get_logger().info("g1_controller up, DISARMED (waiting for LOW_LEVEL_CONTROL).")

    # ------------------------------------------------------------------ setup

    def _declare_params(self) -> None:
        """Declare and read every ROS parameter this node uses."""
        self.declare_parameter("hf_repo_id", "")
        self.declare_parameter("hf_policy_folder", "")
        self.declare_parameter("fm_update_every", 2)
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("onnx_threads", 2)
        self.declare_parameter("depth_topics", [""])
        self.declare_parameter("depth_timeout", 0.25)
        self.declare_parameter("vel_cmd_timeout", 1.0)
        self.declare_parameter("warmup_time", 1.0)
        self.declare_parameter("hold_kp_scale", 2.0)
        self.declare_parameter("hold_kd_scale", 1.5)
        self.declare_parameter("verbose", False)

        self.depth_timeout = float(self.get_parameter("depth_timeout").value)
        self.vel_cmd_timeout = float(self.get_parameter("vel_cmd_timeout").value)
        self.warmup_time = float(self.get_parameter("warmup_time").value)
        self.hold_kp_scale = float(self.get_parameter("hold_kp_scale").value)
        self.hold_kd_scale = float(self.get_parameter("hold_kd_scale").value)
        self.verbose = bool(self.get_parameter("verbose").value)

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
        cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "resource", "hf_cache"))
        policy = load_gen_tracker_from_hf(
            repo, folder, cache,
            fm_update_every=int(self.get_parameter("fm_update_every").value),
            num_threads=int(self.get_parameter("onnx_threads").value),
        )
        self.get_logger().info(
            f"Policy {repo}/{folder}: obs={policy.get_num_obs()} act={policy.get_num_actions()} "
            f"dt={policy.get_dt():.4f}s cameras={len(policy.get_depth_camera_spec())}"
        )
        return policy

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
        self._depth_subs = []
        for i, (topic, cam) in enumerate(zip(topics, self.depth_spec)):
            self._depth_subs.append(self.create_subscription(
                Image, topic, lambda msg, i=i: self.depth_callback(msg, i), SENSOR_QOS))
            self.get_logger().info(
                f"depth[{i}] '{cam.get('sensor') or cam['obs_term']}' "
                f"{int(cam['width'])}x{int(cam['height'])} <- {topic}")

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

        Args:
            msg: 32FC1 depth image, row-major, in METRES along the optical axis.
            index: Which camera, in contract order.
        """
        self.depth_raw[index] = np.frombuffer(msg.data, dtype=np.float32).copy()
        # Local receipt time on purpose: the camera's clock is not synchronised to this
        # machine, so a header stamp cannot tell us whether the stream is still alive.
        self.last_depth_time[index] = self.t

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
            self.arm_time = None
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
        self.hold_pose = self.joint_pos.copy()
        self.arm_time = self.t
        self.armed = True
        self.get_logger().info(
            f"ARMED. Holding pose and priming the generator for {self.warmup_time:.2f}s.")

    # --------------------------------------------------------------- watchdogs

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

    def _effective_command(self) -> np.ndarray:
        """The velocity command actually handed to the policy, after the watchdogs.

        Returns:
            The commanded velocity, or zeros while a sensor or the command is stale. The
            policy keeps running either way -- it still has to balance -- it is just told
            to stand still.
        """
        depth_stale, cmd_stale = self._depth_is_stale(), self._vel_cmd_is_stale()
        if depth_stale != self._depth_stale:
            self._depth_stale = depth_stale
            log = self.get_logger().warn if depth_stale else self.get_logger().info
            log("DEPTH DROPOUT: commanding zero velocity." if depth_stale else "Depth recovered.")
        if cmd_stale != self._vel_cmd_stale:
            self._vel_cmd_stale = cmd_stale
            log = self.get_logger().warn if cmd_stale else self.get_logger().info
            log("VELOCITY COMMAND DROPOUT: commanding zero." if cmd_stale else "Command recovered.")
        return np.zeros(3) if (depth_stale or cmd_stale) else self.vel_cmd

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

    def _build_state(self) -> dict:
        """Assemble the policy's state dict from the latest readings.

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
            command=self._effective_command(),
            depth_raw=self.depth_raw if self.depth_spec else None,
        )

    def compute_control(self) -> PDFeedForward | None:
        """The 50 Hz control tick.

        Returns:
            The published command, or ``None`` when disarmed or not yet ready -- in which
            case NOTHING is published and the robot interface keeps holding its pose.
        """
        if not self.armed or not self.have_estimate:
            return None
        if self.depth_spec and any(d is None for d in self.depth_raw):
            self.get_logger().warn("Waiting for the first frame from every depth camera...",
                                   throttle_duration_sec=2.0)
            return None

        state = self._build_state()
        elapsed = self.t - (self.arm_time or self.t)
        if elapsed < self.warmup_time:
            # Warm-up: feed the generator's history with measured state but do NOT act on
            # its prediction -- with a cold history it has no phase to work from.
            self.policy.observe(state)
            return self._publish(self.hold_pose, warmup=True)

        t0 = time.perf_counter()
        obs = self.policy.create_obs(state)
        action = self.policy.get_action(obs, self.hw_joint_names)
        self._record_latency((time.perf_counter() - t0) * 1e3)
        return self._publish(action, warmup=False)

    def _publish(self, joint_targets: np.ndarray, warmup: bool) -> PDFeedForward:
        """Send PD position targets in the hardware's joint order.

        Args:
            joint_targets: Target joint positions, hardware order.
            warmup: Whether this is the stiffened hold command or the policy's output.

        Returns:
            The published message.
        """
        kp = np.asarray(self.policy.get_kp(), dtype=np.float64)
        kd = np.asarray(self.policy.get_kd(), dtype=np.float64)
        # Metadata gains are in the POLICY's joint order; the robot wants its own.
        kp, kd = kp[self._policy_joint_idx], kd[self._policy_joint_idx]
        if warmup:
            kp, kd = kp * self.hold_kp_scale, kd * self.hold_kd_scale

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
