# Copyright 2026 Zachary Olkin. All rights reserved.

"""Draw the generator's plan in RViz: the predicted root path, and the feet via FK.

The plots answer "what did the reference do over time"; this answers "where is the robot
being asked to put itself, right now, in space". Those fail differently, and the second one
is the question you actually have while watching a robot walk badly.

WHAT IS DRAWN, and why each namespace earns its place:

* ``fm/root_path`` -- the whole F-node plan, lifted out of its anchor frame into the world.
  Should extend ahead of the robot by roughly ``commanded speed * F * dt``.
* ``fm/root_frames`` -- orientation triads every few nodes. A yaw error is INVISIBLE in a
  bare path, and the yaw-only anchor convention is exactly the thing most likely to be wrong.
* ``fm/frame/<name>`` -- forward kinematics of each predicted node's joints, one path per
  target frame. The feet are what a walking failure shows up in first.
* ``fm/served`` -- the 13 window rows the tracker ACTUALLY observed this tick, lifted onto
  the measured base. These should lie on the plan at ``cursor + offsets``; a systematic
  divergence is the re-anchoring, and it is the highest-value thing on this display.
* ``fm/anchor`` -- the pose the plan is expressed relative to, captured at the re-plan.

Deliberately a PLAIN rclpy node launched with ``ros2 run``, following fsm_state_echo_node's
reasoning: as an obelisk entry it would share the lifecycle and node-name space of the real
controller, and a cross-config lifecycle transition would then kill a live node.

Nothing here restates a contract. The block layout, the node width, the joint order, the
frame names and the window offsets all arrive on the wire -- ``FmPlan.window_blocks`` /
``ee_frame_names`` and ``ControllerDebug.joint_names_isaac`` / ``reference_offsets`` are
published every plan and every tick precisely so a consumer never has to know them.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from obelisk_estimator_msgs.msg import EstimatedState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from g1_msgs.msg import ControllerDebug, FmPlan

# The torch-free deploy runtime lives in the repo, not in this ROS package -- see the same
# block in controller.py for why this is realpath and not abspath (colcon --symlink-install
# imports through build/, and abspath does not resolve the symlink).
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

from deploy.fk_mujoco import DEFAULT_FK_MODEL, WindowFK  # noqa: E402
from deploy.quat_math_np import quat_apply  # noqa: E402
from deploy.window_np import (  # noqa: E402
    base_pose_block_to_world,
    window_block_slices,
    window_blocks,
)

#: Match the controller's debug publishers (QoS(50), reliable, volatile). A mismatch here is
#: silent: the subscription simply never pairs and the display stays empty.
DEBUG_QOS = QoSProfile(
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)

#: Orange for the root, cyan for the feet -- the same colours the MuJoCo sim2sim viewer uses
#: for the same quantities (gen_tracker_runner). Two views of one thing should look like it.
ROOT_RGBA = (1.0, 0.55, 0.0, 0.85)
FOOT_RGBA = (0.0, 0.9, 1.0, 0.95)
OTHER_FRAME_RGBA = (0.7, 0.7, 0.7, 0.4)
SERVED_RGBA = (0.2, 1.0, 0.3, 0.9)
CURSOR_RGBA = (1.0, 0.1, 0.1, 1.0)


def _is_foot(name: str) -> bool:
    """True for a target frame on a foot, which is drawn brighter than the rest."""
    low = name.lower()
    return "ankle" in low or "foot" in low


def build_plan_geometry(
    trajectory: np.ndarray,
    blocks: list[str],
    node_dim: int,
    num_joints: int,
    num_ee_frames: int,
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    fk: WindowFK | None,
) -> dict[str, np.ndarray]:
    """Turn one flattened plan into world-frame geometry.

    Pure numpy and free of rclpy on purpose, so the arithmetic that this display exists to
    check can itself be tested without a ROS graph.

    Args:
        trajectory: ``[F * D]`` flattened plan, RAW, in its own anchor frame.
        blocks: Block names making up a node, in packing order (``FmPlan.window_blocks``).
        node_dim: ``D``, used only to reshape and to cross-check the block layout.
        num_joints: ``nj``, from ``ControllerDebug.joint_names_isaac``.
        num_ee_frames: ``len(FmPlan.ee_frame_names)``.
        anchor_pos: ``[3]`` re-plan anchor position, world frame.
        anchor_quat: ``[4]`` xyzw re-plan anchor orientation.
        fk: Resolved :class:`WindowFK`, or None to skip the frame paths.

    Returns:
        ``{"root_pos": [F,3], "root_quat": [F,4]}``, plus ``"frame_pos": [F,K,3]`` when the
        window carries joints and ``fk`` is available. An empty dict if the window has no
        ``base_pose`` block, which is the one thing that cannot be worked around.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if node_dim <= 0 or traj.size < node_dim:
        return {}
    nodes = traj.reshape(-1, node_dim)
    slices = window_block_slices(num_joints, num_ee_frames, blocks)
    if "base_pose" not in slices:
        return {}

    root_pos, root_quat = base_pose_block_to_world(
        nodes[:, slices["base_pose"]], anchor_pos, anchor_quat
    )
    out = {"root_pos": root_pos, "root_quat": root_quat}
    if fk is not None and fk.resolved and "joint_pos" in slices:
        out["frame_pos"] = fk.frame_positions(root_pos, root_quat, nodes[:, slices["joint_pos"]])
    return out


class FmPlanViz(Node):
    """Subscribe to the generator's plan and republish it as RViz markers."""

    def __init__(self) -> None:
        """Declare parameters, resolve FK lazily, and wire the three subscriptions."""
        super().__init__("fm_plan_viz")
        # Must equal the parent frame default_robot_viz puts the base transform under -- it
        # HARDCODES "world" (obelisk_viz_robot_default.h) -- and the fixed frame in
        # g1_deploy.rviz. Otherwise the ribbon lands in a frame RViz cannot relate to the
        # robot and simply does not appear, with no error anywhere.
        self.frame_id = self.declare_parameter("frame_id", "world").value
        self.mjcf_path = self.declare_parameter("mjcf_path", DEFAULT_FK_MODEL).value
        self.marker_topic = self.declare_parameter(
            "marker_topic", "/obelisk/g1/viz/fm_plan"
        ).value
        #: Draw every Nth plan node. F is ~62 and a line strip of all of them is cheap, so
        #: this is for turning the sphere clutter down rather than for cost.
        self.node_stride = int(self.declare_parameter("node_stride", 1).value)
        #: The served window arrives at 50 Hz; redrawing it that fast is unreadable and
        #: pointless next to a 4 Hz plan.
        self.served_decimation = int(self.declare_parameter("served_decimation", 5).value)
        self.draw_served = bool(self.declare_parameter("draw_served", True).value)
        self.draw_other_frames = bool(self.declare_parameter("draw_other_frames", False).value)
        #: So a dead node's markers disappear instead of hanging in the scene, frozen, looking
        #: like a live plan that has stopped moving.
        self.marker_lifetime = float(self.declare_parameter("marker_lifetime", 1.0).value)

        self._fk: WindowFK | None = None
        #: (frame_names, isaac_joint_names) the cached chain was built from. See _resolve_fk.
        self._fk_key: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._tick: ControllerDebug | None = None
        #: (stamp_s, pos, quat) per estimator sample, for aligning the served window to the
        #: sample the tick actually consumed. See _on_est.
        self._est_buf: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._served_count = 0

        self._pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.create_subscription(FmPlan, "/obelisk/g1/debug/fm_plan", self._on_plan, DEBUG_QOS)
        self.create_subscription(ControllerDebug, "/obelisk/g1/debug/tick", self._on_tick,
                                 DEBUG_QOS)
        self.create_subscription(EstimatedState, "/obelisk/g1/est_state", self._on_est, 10)
        self.get_logger().info(
            f"fm_plan_viz up: markers on {self.marker_topic} in frame '{self.frame_id}', "
            f"FK from {self.mjcf_path}"
        )

    # ------------------------------------------------------------------ callbacks
    @staticmethod
    def _stamp_s(stamp) -> float:
        """A builtin_interfaces/Time as float seconds."""
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _on_est(self, msg: EstimatedState) -> None:
        """Buffer measured base poses BY STAMP, for the served window's lift.

        A ring, not a single latch. The estimator runs at 1 kHz against the controller's 50 Hz
        tick, and the tick re-anchored the served window onto the ONE sample it copied at its
        top -- identified by ControllerDebug.est_state_stamp, which exists for exactly this.
        Lifting through "whatever arrived most recently here" instead would offset the green
        dots from the orange ribbon by up to a tick of egomotion (~2 cm at 1 m/s) on a
        perfectly healthy run, and this overlay's whole purpose is that a divergence between
        them means the re-anchoring is wrong.
        """
        if len(msg.q_base) != 7:
            return
        self._est_buf.append((
            self._stamp_s(msg.header.stamp),
            np.asarray(msg.q_base[0:3], dtype=np.float64),
            np.asarray(msg.q_base[3:7], dtype=np.float64),
        ))
        # ~1 s at 1 kHz. The tick that consumed a sample publishes within one control period
        # of it, so anything older than that can never be asked for again.
        if len(self._est_buf) > 1200:
            del self._est_buf[:-1200]

    def _base_at(self, stamp) -> tuple[np.ndarray, np.ndarray] | None:
        """The buffered base pose whose stamp matches `stamp`, or None.

        Nearest by stamp rather than exact: the buffer is what this node received, and a
        dropped sample should degrade the overlay's alignment rather than blank it. Returns
        None only when nothing has been buffered at all.
        """
        if not self._est_buf:
            return None
        want = self._stamp_s(stamp)
        if want <= 0.0:                       # an older controller that never filled it
            return self._est_buf[-1][1], self._est_buf[-1][2]
        best = min(self._est_buf, key=lambda e: abs(e[0] - want))
        return best[1], best[2]

    def _on_tick(self, msg: ControllerDebug) -> None:
        """Latch the tick, and draw the served window on a decimated schedule."""
        self._tick = msg
        if not self.draw_served or not msg.reference_window:
            return
        self._served_count += 1
        if self._served_count % max(1, self.served_decimation):
            return
        markers = self._served_markers(msg)
        if markers:
            self._pub.publish(MarkerArray(markers=markers))

    def _on_plan(self, msg: FmPlan) -> None:
        """Lift, FK and publish one re-plan's worth of markers."""
        if self._tick is None or not self._tick.joint_names_isaac:
            # nj and the isaac order both come from the tick; without one the block layout
            # cannot be resolved and any drawing would be a guess.
            self.get_logger().warn("waiting for a debug tick before drawing a plan",
                                   throttle_duration_sec=5.0)
            return
        nj = len(self._tick.joint_names_isaac)
        frame_names = list(msg.ee_frame_names)
        geom = build_plan_geometry(
            np.asarray(msg.trajectory, dtype=np.float64),
            [str(b) for b in msg.window_blocks],
            int(msg.node_dim),
            nj,
            len(frame_names),
            np.asarray(msg.anchor_pos, dtype=np.float64),
            np.asarray(msg.anchor_quat, dtype=np.float64),
            self._resolve_fk(frame_names, self._tick.joint_names_isaac),
        )
        if not geom:
            self.get_logger().warn(
                "the plan carries no base_pose block, so there is nothing to draw in world "
                "coordinates", throttle_duration_sec=10.0)
            return

        markers = self._plan_markers(geom, frame_names, msg)
        self._pub.publish(MarkerArray(markers=markers))

    # ------------------------------------------------------------------ helpers
    def _resolve_fk(self, frame_names: list[str], isaac_names: list[str]) -> WindowFK | None:
        """Build the FK chain, rebuilding it if the contract it was built from has changed.

        Keyed on the frames AND the joint order, not cached once forever: only the controller
        pane is restarted when iterating with `--policy`, so this node routinely outlives the
        artifact it first saw. A cached permutation from the previous policy would FK every
        node with the wrong joint mapping and draw feet that are confidently, plausibly wrong
        -- the one failure this display must not be capable of.
        """
        key = (tuple(frame_names), tuple(isaac_names))
        if key == self._fk_key:
            return self._fk
        # A new contract deserves a fresh attempt even if the last one failed.
        self._fk_key = key
        self._fk = None
        if not frame_names:
            # An older publisher, before FmPlan carried the names. Root path only.
            self.get_logger().warn(
                "this FmPlan carries no ee_frame_names, so no FK foot paths -- rebuild "
                "g1_msgs and the controller", throttle_duration_sec=30.0)
            return None
        try:
            self._fk = WindowFK(self.mjcf_path, frame_names, list(isaac_names))
        except Exception as exc:                                      # noqa: BLE001
            # Not retried, and not fatal: the cached key holds the failure, so this logs once
            # per contract rather than at 4 Hz, and the root ribbon still draws without feet.
            self.get_logger().error(
                f"no FK, so the root path is drawn without the frames: {exc}")
            return None
        if self._fk.missing:
            self.get_logger().warn(
                f"FK frames absent from {self.mjcf_path}: {self._fk.missing}")
        return self._fk

    def _marker(self, ns: str, mid: int, mtype: int, rgba, scale: float) -> Marker:
        """A marker with this node's frame, lifetime and colour conventions applied."""
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        sec = int(self.marker_lifetime)
        m.lifetime = Duration(sec=sec, nanosec=int((self.marker_lifetime - sec) * 1e9))
        return m

    @staticmethod
    def _points(xyz: np.ndarray) -> list[Point]:
        """Finite rows of an ``[N, 3]`` array as marker points; NaN rows are dropped."""
        return [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in xyz if np.all(np.isfinite(p))]

    def _triads(self, ns: str, mid: int, pos: np.ndarray, quat: np.ndarray,
                length: float, stride: int) -> list[Marker]:
        """One LINE_LIST per axis, so orientation is visible where a path shows nothing."""
        out = []
        idx = np.arange(0, pos.shape[0], max(1, stride))
        for axis, rgba in enumerate(((1.0, 0.2, 0.2, 0.8), (0.2, 1.0, 0.2, 0.8),
                                     (0.2, 0.4, 1.0, 0.8))):
            m = self._marker(ns, mid + axis, Marker.LINE_LIST, rgba, 0.004)
            unit = np.zeros(3)
            unit[axis] = length
            for i in idx:
                tip = pos[i] + quat_apply(quat[i], unit)
                m.points.append(Point(x=float(pos[i][0]), y=float(pos[i][1]), z=float(pos[i][2])))
                m.points.append(Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])))
            if m.points:
                out.append(m)
        return out

    def _plan_markers(self, geom: dict, frame_names: list[str], msg: FmPlan) -> list[Marker]:
        """Markers for one re-plan: the root ribbon, its triads, and the FK'd frames."""
        root = geom["root_pos"]
        markers: list[Marker] = []

        path = self._marker("fm/root_path", 0, Marker.LINE_STRIP, ROOT_RGBA, 0.01)
        path.points = self._points(root)
        markers.append(path)

        dots = self._marker("fm/root_nodes", 0, Marker.SPHERE_LIST, ROOT_RGBA, 0.02)
        dots.points = self._points(root[:: max(1, self.node_stride)])
        markers.append(dots)

        markers += self._triads("fm/root_frames", 0, root, geom["root_quat"], 0.12,
                                stride=max(1, root.shape[0] // 10))

        # The anchor, drawn longer than the node triads so the frame the whole plan hangs off
        # is distinguishable from the plan itself.
        markers += self._triads(
            "fm/anchor", 0,
            np.asarray(msg.anchor_pos, dtype=np.float64).reshape(1, 3),
            np.asarray(msg.anchor_quat, dtype=np.float64).reshape(1, 4), 0.25, stride=1)

        if "frame_pos" in geom:
            for k, name in enumerate(frame_names):
                foot = _is_foot(name)
                if not foot and not self.draw_other_frames:
                    continue
                rgba = FOOT_RGBA if foot else OTHER_FRAME_RGBA
                line = self._marker(f"fm/frame/{name}", 0, Marker.LINE_STRIP, rgba, 0.008)
                line.points = self._points(geom["frame_pos"][:, k, :])
                markers.append(line)

        # The node being served right now, so "where in the plan are we" is visible without
        # reading fm_cursor off a plot.
        cursor = int(self._tick.fm_cursor) if self._tick is not None else -1
        if 0 <= cursor < root.shape[0] and np.all(np.isfinite(root[cursor])):
            dot = self._marker("fm/cursor", 0, Marker.SPHERE, CURSOR_RGBA, 0.05)
            dot.pose.position = Point(x=float(root[cursor][0]), y=float(root[cursor][1]),
                                      z=float(root[cursor][2]))
            markers.append(dot)
        return markers

    def _served_markers(self, msg: ControllerDebug) -> list[Marker]:
        """The window the tracker actually observed, lifted onto the MEASURED base.

        This is the payoff display. ``reference_window`` has already been re-anchored onto the
        robot's current pose on the way into the observation, so it lifts through the measured
        base rather than the plan's anchor -- and it should land on the orange ribbon at
        ``cursor + offsets``. Where it does not, the re-anchoring is what is wrong.
        """
        base = self._base_at(msg.est_state_stamp)
        if base is None or not msg.reference_blocks or not msg.reference_node_dim:
            return []
        nj = len(msg.joint_names_isaac)
        if nj == 0:
            return []
        node_dim = int(msg.reference_node_dim)
        rows = np.asarray(msg.reference_window, dtype=np.float64)
        if rows.size < node_dim:
            return []
        rows = rows.reshape(-1, node_dim)
        # ControllerDebug names the blocks but not the frame COUNT, and ee_pos is the one
        # block whose width depends on it. It is absent from the live artifact's selection, so
        # this is normally 0 -- derived from the leftover width rather than assumed, so a
        # window that does carry the block still resolves instead of silently mis-slicing
        # everything after it.
        blocks = [str(b) for b in msg.reference_blocks]
        n_ee = 0
        if "ee_pos" in blocks:
            without_ee = sum(w for _, w in window_blocks(nj, 0, blocks))
            n_ee = max(0, (node_dim - without_ee) // 3)
        slices = window_block_slices(nj, n_ee, blocks)
        if "base_pose" not in slices:
            return []
        pos, _ = base_pose_block_to_world(rows[:, slices["base_pose"]], base[0], base[1])
        dots = self._marker("fm/served", 0, Marker.SPHERE_LIST, SERVED_RGBA, 0.03)
        dots.points = self._points(pos)
        line = self._marker("fm/served", 1, Marker.LINE_STRIP, SERVED_RGBA, 0.006)
        line.points = dots.points
        return [dots, line]


def main() -> None:
    """Spin the marker node."""
    rclpy.init()
    node = FmPlanViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
