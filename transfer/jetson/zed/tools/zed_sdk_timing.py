#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Measure the ZED SDK's own share of the depth latency, with no ROS in the picture.

``zed_ros_timing.py`` measures the whole pipeline end to end.  This measures the part of it
that belongs to the camera and the SDK, so the remainder can be attributed rather than
guessed:

* ``capture -> grab()`` = exposure + GMSL/CSI transport + ISP + rectification + the NEURAL
  depth network.  Taken as ``TIME_REFERENCE.CURRENT - TIME_REFERENCE.IMAGE`` immediately
  after ``grab()`` returns, which is the same IMAGE stamp ``zed_depth_node`` puts in the
  ROS header -- so the two tools are measuring from the same origin.
* ``retrieve_measure(DEPTH, CPU)`` = the device-to-host copy of the full-resolution float
  depth map, which the node does on every frame before it crops anything.
* the grab cadence itself, which is what caps the publish rate.

Everything left over between this and ``zed_ros_timing.py`` is the node's own processing
(blur / FoV clip / downsample / message build) plus DDS.

It also prints the device list, which is the only way to learn the serials -- and
``sl.InitParameters.set_from_serial_number`` here is exactly the call ``zed_depth_publisher``
is missing, so this doubles as a check that a specific camera can be opened on demand.

Usage, inside the ZED container::

    python3 zed_sdk_timing.py --list
    python3 zed_sdk_timing.py --serial 44889004 --frames 300
    python3 zed_sdk_timing.py --serial 59249806 --resolution HD720 --fps 60 \
                              --depth-mode NEURAL --json far.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from typing import Any

import pyzed.sl as sl

#: Where the Orin exposes the GPU's current clock. Sampled alongside every frame, because a
#: rate measured while the governor is parked at the minimum is a statement about the
#: governor and not about the camera -- which is exactly the ambiguity that cost a sweep.
GPU_FREQ_PATH = "/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"
GPU_POWER_CONTROL_PATH = "/sys/devices/platform/gpu.0/power/control"


def read_int(path: str) -> int | None:
    """Read a single integer out of a sysfs file.

    Args:
        path: The sysfs file to read.

    Returns:
        The integer, or None if the file is absent or unreadable.
    """
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def read_str(path: str) -> str | None:
    """Read a single line out of a sysfs file.

    Args:
        path: The sysfs file to read.

    Returns:
        The stripped contents, or None if the file is absent or unreadable.
    """
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile.

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
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def describe(values: list[float], scale: float = 1e3) -> dict[str, float]:
    """Summarise a sample as the percentiles worth quoting.

    Args:
        values: Samples in seconds.
        scale: Multiplier applied to every output, 1e3 to report milliseconds.

    Returns:
        Mean/median/p90/p99/min/max, scaled.
    """
    return {
        "mean": scale * statistics.fmean(values),
        "median": scale * statistics.median(values),
        "p90": scale * percentile(values, 90),
        "p99": scale * percentile(values, 99),
        "min": scale * min(values),
        "max": scale * max(values),
    }


def list_devices() -> list[dict[str, Any]]:
    """Enumerate the attached ZED cameras.

    Returns:
        One entry per camera with its id, serial, model and availability.
    """
    out = []
    for dev in sl.Camera.get_device_list():
        out.append(
            {
                "id": dev.id,
                "serial_number": dev.serial_number,
                "camera_model": str(dev.camera_model),
                "camera_state": str(dev.camera_state),
                "path": dev.path,
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Open one camera, grab ``args.frames`` frames and time each stage.

    Args:
        args: Parsed command line.

    Returns:
        A JSON-serialisable summary.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.camera_fps = args.fps
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.METER
    init.depth_stabilization = args.depth_stabilization
    init.sdk_verbose = 0
    if args.serial:
        # The selection zed_depth_publisher has no parameter for.  Without it the SDK opens
        # whichever camera it finds first, which on a two-camera box is a coin flip.
        init.set_from_serial_number(args.serial)

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        # INVALID RESOLUTION here almost always means a USB-camera mode on a GMSL camera; the
        # SDK enum accepts it and the hardware does not, so the error names neither the mode
        # nor the camera.
        hint = ""
        if status == sl.ERROR_CODE.INVALID_RESOLUTION:
            hint = (f" -- {args.resolution} is not a mode this camera offers. GMSL ZED X "
                    f"supports HD1200 / HD1080 / SVGA")
        raise RuntimeError(f"open failed: {status}{hint}")

    info = cam.get_camera_information()
    conf = info.camera_configuration
    left = conf.calibration_parameters.left_cam
    width, height = int(left.image_size.width), int(left.image_size.height)
    # The native cone, which is the ceiling on what clip_fov can produce: that stage only
    # CROPS, and asking for more than this silently clamps to the image bounds.
    hfov = 2.0 * math.degrees(math.atan2(width / 2.0, left.fx))
    vfov = 2.0 * math.degrees(math.atan2(height / 2.0, left.fy))

    # Exposure, and the readback. AEC_AGC must go off FIRST or the auto loop overwrites the
    # value on the next frame; and the readback is what gets recorded, never the request,
    # because the camera quantises to whole sensor rows and can clamp to the FPS period.
    if args.exposure_us > 0:
        cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE_TIME, args.exposure_us)
    _, aec = cam.get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC)
    _, exp_us = cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE_TIME)

    summary: dict[str, Any] = {
        "auto_exposure": bool(aec),
        "exposure_us": int(exp_us),
        "serial_number": info.serial_number,
        "camera_model": str(info.camera_model),
        "resolution": [width, height],
        "configured_fps": conf.fps,
        "depth_mode": args.depth_mode,
        "depth_stabilization": args.depth_stabilization,
        "retrieve": args.retrieve,
        "intrinsics": {"fx": left.fx, "fy": left.fy, "cx": left.cx, "cy": left.cy},
        "native_fov_deg": {"h": hfov, "v": vfov},
        "depth_range_m": [
            float(init.depth_minimum_distance), float(init.depth_maximum_distance)
        ],
    }

    depth = sl.Mat()
    runtime = sl.RuntimeParameters()
    runtime.enable_depth = True
    runtime.confidence_threshold = args.confidence
    runtime.texture_confidence_threshold = args.texture_confidence

    capture_to_grab: list[float] = []
    grab_wall: list[float] = []
    retrieve_wall: list[float] = []
    grab_return_mono: list[float] = []
    gpu_freq_hz: list[int] = []
    failures = 0

    # Discard the first frames: the first grab after open carries pipeline warm-up (and, if
    # the engine were not already built, the whole TensorRT build), which is a startup cost
    # and not the steady-state number this is measuring.
    total = args.frames + args.warmup
    if args.start_at:
        # Unbounded: frames burned at the barrier are not measured frames, so a fixed total
        # would end the run before the window even opens.
        total = 10 ** 9
    measured = 0
    for i in range(total):
        t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
        err = cam.grab(runtime)
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
        if err != sl.ERROR_CODE.SUCCESS:
            failures += 1
            continue
        image_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        current_ns = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()

        t2 = time.clock_gettime(time.CLOCK_MONOTONIC)
        if args.retrieve == "cpu":
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
        elif args.retrieve == "gpu":
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.GPU)
        t3 = time.clock_gettime(time.CLOCK_MONOTONIC)

        if i < args.warmup:
            continue
        # Hold at the barrier while still grabbing AND retrieving, so the other probe sees
        # this camera's full load during its own warm-up rather than an idle GPU.
        if args.start_at and time.clock_gettime(time.CLOCK_REALTIME) < args.start_at:
            continue
        freq = read_int(GPU_FREQ_PATH)
        if freq is not None:
            gpu_freq_hz.append(freq)
        capture_to_grab.append(1e-9 * (current_ns - image_ns))
        grab_wall.append(t1 - t0)
        retrieve_wall.append(t3 - t2)
        grab_return_mono.append(t1)
        measured += 1
        if measured >= args.frames:
            break

    cam.close()

    summary["frames"] = len(capture_to_grab)
    summary["grab_failures"] = failures
    if len(capture_to_grab) < 2:
        return summary

    span = grab_return_mono[-1] - grab_return_mono[0]
    gaps = [b - a for a, b in zip(grab_return_mono[:-1], grab_return_mono[1:], strict=True)]
    summary["grab_rate_hz"] = (len(grab_return_mono) - 1) / span if span > 0 else None
    summary["grab_interval_ms"] = describe(gaps)
    # The headline: how old the pixels already are the instant the SDK hands them over.
    summary["capture_to_grab_ms"] = describe(capture_to_grab)
    # Wall time blocked inside grab(); at a fixed fps most of this is waiting for the next
    # frame, so it bounds rather than measures the compute.
    summary["grab_blocking_ms"] = describe(grab_wall)
    summary["retrieve_target"] = args.retrieve
    summary["retrieve_measure_ms"] = describe(retrieve_wall)
    if gpu_freq_hz:
        summary["gpu_clock_mhz"] = {
            "median": statistics.median(gpu_freq_hz) / 1e6,
            "min": min(gpu_freq_hz) / 1e6,
            "max": max(gpu_freq_hz) / 1e6,
        }
    summary["gpu_power_control"] = read_str(GPU_POWER_CONTROL_PATH)
    return summary


def main() -> int:
    """Parse arguments, run the probe and print the result.

    Returns:
        0 on success, 1 if the camera produced too few frames to summarise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list cameras and exit")
    parser.add_argument("--serial", type=int, default=0, help="camera serial to open")
    parser.add_argument("--frames", type=int, default=300, help="frames to measure")
    parser.add_argument("--warmup", type=int, default=30, help="frames to discard first")
    parser.add_argument("--start-at", type=float, default=0.0, metavar="UNIX_TS",
                        help="after warm-up, keep grabbing but discard until this wall-clock "
                             "instant, then measure. Two probes given the same value measure "
                             "over the SAME window, which is the only way a two-camera "
                             "contention number means anything -- camera open alone varies by "
                             "seconds, so unsynchronised runs would partly measure each other's "
                             "absence")
    # SVGA, not HD720. `sl.RESOLUTION.HD720` EXISTS in the enum, so nothing rejects it until
    # `open()` returns INVALID RESOLUTION -- but it is a USB-camera mode and no GMSL ZED X
    # offers it. The GMSL modes are HD1200 / HD1080 / SVGA, and SVGA is what the deployment
    # runs: measured concurrently on both cameras, SVGA+NEURAL_LIGHT holds 44.8 Hz where
    # HD1200+NEURAL manages 15.7.
    parser.add_argument("--resolution", default="SVGA",
                        help="GMSL ZED X: HD1200/HD1080/SVGA (HD720 is USB-only and will "
                             "fail at open)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--depth-mode", default="NEURAL",
                        help="NEURAL_LIGHT / NEURAL / NEURAL_PLUS")
    parser.add_argument("--exposure-us", type=int, default=-1, metavar="US",
                        help="fix the exposure time in microseconds (disables AEC_AGC). "
                             "-1 = leave auto. Exposure is the one part of capture->grab that "
                             "is a CHOICE rather than a fixed hardware cost, so measuring "
                             "against a pinned value is how you learn whether auto-exposure "
                             "is buying latency with light")
    parser.add_argument("--depth-stabilization", type=int, default=30,
                        help="0 disables; the node's default is 30, and it is temporal")
    parser.add_argument("--retrieve", default="cpu", choices=("cpu", "gpu", "none"),
                        help="where retrieve_measure lands. The node uses cpu; gpu and none "
                             "exist to separate the device-to-host copy from the depth "
                             "network, which the SDK may only finish when the result is asked "
                             "for -- in which case the cost is compute wearing a copy's name")
    parser.add_argument("--confidence", type=int, default=95)
    parser.add_argument("--texture-confidence", type=int, default=100)
    parser.add_argument("--json", default=None, help="also write the summary here")
    args = parser.parse_args()

    if args.list:
        devices = list_devices()
        for d in devices:
            print(f"id={d['id']}  sn={d['serial_number']}  {d['camera_model']}  "
                  f"{d['camera_state']}  {d['path']}")
        if not devices:
            print("no cameras found")
        return 0 if devices else 1

    summary = run(args)
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0 if summary.get("frames", 0) >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
