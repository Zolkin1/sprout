#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Measure the rate and the capture-to-receipt latency of the ZED depth topics.

This answers two questions about what the ZED box costs the control loop:

* **Rate.** How often a depth frame actually lands, and how evenly.  The controller reads
  whatever is latest at its own 50 Hz tick, so a camera that averages 50 Hz in bursts of
  two is not the same as one that delivers every 20 ms, and only the jitter distinguishes
  them.
* **Latency.** How old the pixels are when a subscriber first sees them.
  ``zed_depth_node`` stamps the header with ``getTimestamp(TIME_REFERENCE::IMAGE)`` -- the
  camera's own capture time -- so ``now - header.stamp`` is the WHOLE pipeline: exposure,
  GMSL/CSI transport, ISP, the NEURAL depth network, the blur/clip/downsample pipeline,
  serialization and DDS.  Nothing here has to be inferred from a benchmark.

That subtraction is only meaningful when both sides read the same clock, which is exactly
the caveat ``ControllerDebug.depth_capture_age`` carries on hardware.  Run this ON THE BOX
THAT PUBLISHES and the caveat disappears.  It is not assumed either way: the stamp is
compared against BOTH ``CLOCK_REALTIME`` and ``CLOCK_MONOTONIC`` and the script reports
which base the camera is actually stamping in, because a Jetson capture path can plausibly
hand back either and the wrong choice is a silently enormous or negative "latency".

Usage, inside the ZED container (ROS 2 must be sourced)::

    python3 zed_ros_timing.py --topic /depth/image_processed --seconds 30
    python3 zed_ros_timing.py --topic /depth/near/image_processed \
                              --topic /depth/far/image_processed \
                              --expect 30x26 --seconds 60 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, not a code path
    print("numpy is required (it ships in the Stereolabs image)", file=sys.stderr)
    raise

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

#: A stamp is judged to be in a given clock base if the implied age lands in this window.
#: Generous on the high side -- a NEURAL_PLUS frame on a loaded Orin can be far from fast --
#: but nowhere near the hours that separate CLOCK_REALTIME from CLOCK_MONOTONIC on a box
#: that has been up for a while, which is the confusion this exists to resolve.
PLAUSIBLE_AGE_S = (-0.010, 2.0)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, so the tool carries no dependency for one number.

    Args:
        values: Samples; need not be sorted.  Must be non-empty.
        q: Percentile in [0, 100].

    Returns:
        The interpolated percentile of ``values``.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass
class TopicStats:
    """Everything accumulated for one depth topic."""

    topic: str
    #: Receipt instants on CLOCK_MONOTONIC, which is what rate and jitter must be measured
    #: against -- CLOCK_REALTIME can be stepped by NTP mid-run.
    recv_monotonic: list[float] = field(default_factory=list)
    #: Age of each frame under each candidate clock base for the header stamp.
    age_realtime: list[float] = field(default_factory=list)
    age_monotonic: list[float] = field(default_factory=list)
    #: Capture instants (header stamps), for the two-camera sync analysis.
    capture_stamps: list[float] = field(default_factory=list)
    #: Fraction of pixels carrying no reading, per frame.
    invalid_fraction: list[float] = field(default_factory=list)
    finite_min: list[float] = field(default_factory=list)
    finite_max: list[float] = field(default_factory=list)
    finite_mean: list[float] = field(default_factory=list)
    shape: tuple[int, int] | None = None
    encoding: str = ""
    #: Frames dropped for a shape/encoding the caller said not to expect.
    mismatched: int = 0
    #: Frames whose header stamp was zero -- no capture time to measure against.
    unstamped: int = 0


class DepthTimingNode(Node):
    """Subscribes to each depth topic and accumulates timing and pixel statistics."""

    def __init__(self, topics: list[str], expect: tuple[int, int] | None) -> None:
        """Create the node and one best-effort subscription per topic.

        Args:
            topics: Fully-resolved topic names to subscribe to.
            expect: Optional ``(width, height)`` every frame must match, so a
                wrong-shaped frame is counted rather than quietly averaged in.
        """
        super().__init__("zed_ros_timing")
        self._expect = expect
        self.stats: dict[str, TopicStats] = {t: TopicStats(topic=t) for t in topics}

        # Must MATCH the publisher: zed_depth_node uses KeepLast(1) + BEST_EFFORT +
        # VOLATILE.  A RELIABLE subscriber is incompatible with a BEST_EFFORT publisher and
        # simply never connects -- which looks identical to a camera that is not running.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        for topic in topics:
            self.create_subscription(
                Image, topic, lambda msg, t=topic: self._on_image(msg, t), qos
            )

    def _on_image(self, msg: Image, topic: str) -> None:
        """Record one frame's arrival time, age and pixel health.

        Both clocks are read FIRST, before any decoding, so the measured age is the
        subscriber's arrival instant rather than arrival plus this function's own cost.

        Args:
            msg: The received depth image.
            topic: The topic it arrived on.
        """
        now_mono = time.clock_gettime(time.CLOCK_MONOTONIC)
        now_real = time.clock_gettime(time.CLOCK_REALTIME)

        st = self.stats[topic]
        st.recv_monotonic.append(now_mono)
        st.shape = (msg.width, msg.height)
        st.encoding = msg.encoding

        if self._expect is not None and (msg.width, msg.height) != self._expect:
            st.mismatched += 1
            return

        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if stamp <= 0.0:
            st.unstamped += 1
        else:
            st.capture_stamps.append(stamp)
            st.age_realtime.append(now_real - stamp)
            st.age_monotonic.append(now_mono - stamp)

        if msg.encoding != "32FC1":
            return
        # msg.data is an array.array('B') under rclpy and bytes under some readers; both
        # expose the buffer protocol, so this never copies the payload.
        pixels = np.frombuffer(memoryview(msg.data), dtype=np.float32)
        finite = np.isfinite(pixels)
        # A miss reaches the policy as a sentinel, and the SDK writes non-finite for "no
        # reading" while the NaN-aware downsample writes NaN for a fully-empty block.  Both
        # are the same thing here.
        st.invalid_fraction.append(float(1.0 - finite.mean()))
        if finite.any():
            good = pixels[finite]
            st.finite_min.append(float(good.min()))
            st.finite_max.append(float(good.max()))
            st.finite_mean.append(float(good.mean()))


def summarise(st: TopicStats) -> dict[str, Any]:
    """Reduce one topic's samples to the numbers worth reporting.

    Args:
        st: The accumulated statistics for a topic.

    Returns:
        A JSON-serialisable summary.  ``clock_base`` names which clock the publisher's
        header stamp appears to be in, or ``"unknown"`` when neither candidate yields a
        plausible age -- in which case the latency figures are omitted rather than
        reported wrong.
    """
    out: dict[str, Any] = {
        "topic": st.topic,
        "frames": len(st.recv_monotonic),
        "shape": list(st.shape) if st.shape else None,
        "encoding": st.encoding,
        "shape_mismatched_frames": st.mismatched,
        "unstamped_frames": st.unstamped,
    }
    if len(st.recv_monotonic) < 2:
        return out

    gaps = [
        b - a for a, b in zip(st.recv_monotonic[:-1], st.recv_monotonic[1:], strict=True)
    ]
    span = st.recv_monotonic[-1] - st.recv_monotonic[0]
    out["rate_hz_overall"] = (len(st.recv_monotonic) - 1) / span if span > 0 else None
    out["interarrival_ms"] = {
        "mean": 1e3 * statistics.fmean(gaps),
        "median": 1e3 * statistics.median(gaps),
        "stdev": 1e3 * (statistics.stdev(gaps) if len(gaps) > 1 else 0.0),
        "min": 1e3 * min(gaps),
        "p99": 1e3 * percentile(gaps, 99),
        "max": 1e3 * max(gaps),
    }

    # Pick the clock base from the data rather than assuming one.  Whichever candidate puts
    # the MEDIAN age inside the plausible window is the publisher's base; a Jetson capture
    # path can hand back either, and guessing wrong turns a 60 ms latency into a number the
    # size of the uptime (or a negative one).
    candidates = {"CLOCK_REALTIME": st.age_realtime, "CLOCK_MONOTONIC": st.age_monotonic}
    chosen: str = "unknown"
    for name, ages in candidates.items():
        if ages and PLAUSIBLE_AGE_S[0] <= statistics.median(ages) <= PLAUSIBLE_AGE_S[1]:
            chosen = name
            break
    out["clock_base"] = chosen
    if chosen != "unknown":
        ages = candidates[chosen]
        out["capture_to_receipt_ms"] = {
            "mean": 1e3 * statistics.fmean(ages),
            "median": 1e3 * statistics.median(ages),
            "p90": 1e3 * percentile(ages, 90),
            "p99": 1e3 * percentile(ages, 99),
            "min": 1e3 * min(ages),
            "max": 1e3 * max(ages),
        }
    else:
        # Report the raw offsets so the mismatch is diagnosable instead of just absent.
        out["median_offset_vs_realtime_s"] = (
            statistics.median(st.age_realtime) if st.age_realtime else None
        )
        out["median_offset_vs_monotonic_s"] = (
            statistics.median(st.age_monotonic) if st.age_monotonic else None
        )

    if st.invalid_fraction:
        out["pixels"] = {
            "invalid_fraction_mean": statistics.fmean(st.invalid_fraction),
            "invalid_fraction_max": max(st.invalid_fraction),
            "depth_min_m": min(st.finite_min) if st.finite_min else None,
            "depth_max_m": max(st.finite_max) if st.finite_max else None,
            "depth_mean_m": statistics.fmean(st.finite_mean) if st.finite_mean else None,
        }
    return out


def pair_sync(a: TopicStats, b: TopicStats) -> dict[str, Any]:
    """Quantify how far apart in CAPTURE time two cameras' frames actually are.

    Delivery age and exposure alignment are different questions: a pair can be perfectly
    hardware-synced and still arrive at different ages because one queues behind the other
    on the GPU.  This measures the exposure side, which is the one that decides whether the
    two views the policy fuses were taken at the same moment.

    For every frame on ``a`` it finds the closest capture instant on ``b``.  Free-running
    sensors give an offset spread roughly uniformly over half a frame period and drifting
    steadily as the two clocks diverge; a hardware-synced pair gives a small, flat offset.

    Args:
        a: Statistics for the first topic.
        b: Statistics for the second topic.

    Returns:
        The offset distribution in milliseconds, plus a drift rate in ms per second
        estimated by least squares over the run.
    """
    import bisect

    sa, sb = a.capture_stamps, sorted(b.capture_stamps)
    if len(sa) < 10 or len(sb) < 10:
        return {"error": "not enough stamped frames on both topics"}

    offsets, times = [], []
    for s in sa:
        i = bisect.bisect_left(sb, s)
        best = min(
            (abs(s - sb[j]) for j in (i - 1, i, i + 1) if 0 <= j < len(sb)),
            default=None,
        )
        if best is not None:
            offsets.append(best)
            times.append(s)

    t0 = times[0]
    xs = [x - t0 for x in times]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(offsets) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, offsets, strict=True)) / denom if denom else 0.0

    return {
        "frames_compared": n,
        "offset_ms": {
            "median": 1e3 * statistics.median(offsets),
            "mean": 1e3 * statistics.fmean(offsets),
            "min": 1e3 * min(offsets),
            "p90": 1e3 * percentile(offsets, 90),
            "max": 1e3 * max(offsets),
        },
        # A synced pair holds a constant offset; a free-running pair walks through the whole
        # frame period as the two sensor clocks diverge.
        "drift_ms_per_s": 1e3 * slope,
        "span_s": xs[-1],
    }


def render(summary: dict[str, Any]) -> str:
    """Format one topic summary as a human-readable block.

    Args:
        summary: The output of :func:`summarise`.

    Returns:
        A multi-line string.
    """
    lines = [f"topic {summary['topic']}"]
    shape = summary.get("shape")
    head = f"  frames        {summary['frames']}"
    if shape:
        head += f"   shape {shape[0]}x{shape[1]} {summary['encoding']}"
    lines.append(head)
    if summary["shape_mismatched_frames"]:
        lines.append(f"  WRONG SHAPE   {summary['shape_mismatched_frames']} frames")
    if summary["unstamped_frames"]:
        lines.append(f"  UNSTAMPED     {summary['unstamped_frames']} frames")
    if "rate_hz_overall" not in summary:
        lines.append("  (too few frames to summarise)")
        return "\n".join(lines)

    ia = summary["interarrival_ms"]
    lines.append(f"  rate          {summary['rate_hz_overall']:.2f} Hz")
    lines.append(
        f"  inter-arrival mean {ia['mean']:.2f}  median {ia['median']:.2f}  "
        f"sd {ia['stdev']:.2f}  min {ia['min']:.2f}  p99 {ia['p99']:.2f}  "
        f"max {ia['max']:.2f}  ms"
    )
    if "capture_to_receipt_ms" in summary:
        lt = summary["capture_to_receipt_ms"]
        lines.append(f"  stamp clock   {summary['clock_base']}")
        lines.append(
            f"  capture->recv median {lt['median']:.2f}  mean {lt['mean']:.2f}  "
            f"p90 {lt['p90']:.2f}  p99 {lt['p99']:.2f}  max {lt['max']:.2f}  ms"
        )
    else:
        lines.append(
            "  stamp clock   UNKNOWN -- the header stamp matches neither CLOCK_REALTIME "
            f"({summary.get('median_offset_vs_realtime_s')!r} s) nor CLOCK_MONOTONIC "
            f"({summary.get('median_offset_vs_monotonic_s')!r} s). Latency NOT reported."
        )
    if "pixels" in summary:
        px = summary["pixels"]
        lines.append(
            f"  pixels        invalid mean {100 * px['invalid_fraction_mean']:.2f}%  "
            f"max {100 * px['invalid_fraction_max']:.2f}%   "
            f"depth [{px['depth_min_m']:.3f}, {px['depth_max_m']:.3f}] m  "
            f"mean {px['depth_mean_m']:.3f} m"
        )
    return "\n".join(lines)


def main() -> int:
    """Run the subscriber for the requested duration and print a summary.

    Returns:
        0 if every requested topic produced at least two frames, 1 otherwise -- so this is
        usable as a check and not only as a report.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--topic", action="append", required=True,
        help="depth image topic; repeat for several cameras",
    )
    parser.add_argument("--seconds", type=float, default=20.0, help="how long to listen")
    parser.add_argument(
        "--expect", default=None, metavar="WxH",
        help="required frame shape, e.g. 30x26; frames of any other shape are counted, "
             "not averaged in",
    )
    parser.add_argument("--json", default=None, help="also write the summary here")
    args = parser.parse_args()

    expect: tuple[int, int] | None = None
    if args.expect:
        w, _, h = args.expect.partition("x")
        expect = (int(w), int(h))

    rclpy.init()
    node = DepthTimingNode(args.topic, expect)
    print(f"listening for {args.seconds:.0f} s on: {', '.join(args.topic)}", flush=True)
    deadline = time.monotonic() + args.seconds
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    summaries = [summarise(node.stats[t]) for t in args.topic]
    sync = pair_sync(node.stats[args.topic[0]], node.stats[args.topic[1]]) if len(args.topic) == 2 else None
    node.destroy_node()
    rclpy.shutdown()

    print()
    for s in summaries:
        print(render(s))
        print()
    if sync is not None:
        print("capture-time offset between the two cameras")
        if "error" in sync:
            print(f"  {sync['error']}")
        else:
            o = sync["offset_ms"]
            print(f"  compared      {sync['frames_compared']} frames over {sync['span_s']:.1f} s")
            print(f"  |dt| capture  median {o['median']:.2f}  mean {o['mean']:.2f}  "
                  f"min {o['min']:.2f}  p90 {o['p90']:.2f}  max {o['max']:.2f}  ms")
            print(f"  drift         {sync['drift_ms_per_s']:+.3f} ms per second "
                  f"(~0 = locked; nonzero = free-running clocks diverging)")
        print()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"topics": summaries, "pair_sync": sync}, fh, indent=2)
        print(f"wrote {args.json}")

    return 0 if all(s["frames"] >= 2 for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
