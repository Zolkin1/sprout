#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Run zed_sdk_timing.py on BOTH cameras over ONE synchronized measurement window.
#
# Two probes given the same --start-at measure the same window, which is the only way a
# two-camera number means anything: camera open alone varies by seconds, so unsynchronised
# runs partly measure each other's ABSENCE. Contention is not a detail here -- it is 17 ms
# of the 42 ms end-to-end latency, and one camera alone reaches 33.7 ms where the pair
# reaches 51.0.
#
# Run it from the HOST (it docker-execs into the container). The cameras must be free: the
# node holds them, and sl::Camera::close() takes 25-30 s to release one.
#
#   tools/measure_pair.sh <tag> [extra zed_sdk_timing.py flags...]
#
#   tools/measure_pair.sh shipped   --resolution SVGA --depth-mode NEURAL_LIGHT --fps 120 \
#                                   --depth-stabilization 30
#   tools/measure_pair.sh stab0     --resolution SVGA --depth-mode NEURAL_LIGHT --fps 120 \
#                                   --depth-stabilization 0
#   tools/measure_pair.sh noretrieve --resolution SVGA --fps 120 --retrieve none
#
# For the END-TO-END number (capture -> subscriber, through the node and DDS) use
# zed_ros_timing.py against a running pair instead; this measures the camera and SDK alone.
set -euo pipefail

TAG="${1:-}"
[ -n "$TAG" ] || { echo "usage: measure_pair.sh <tag> [flags...]" >&2; exit 2; }
shift

OUT="${ZED_MEASURE_OUT:-/tmp/zed_measure}"
CONTAINER="${ZED_CONTAINER:-zed_depth}"
# near = the LOWER mount (ZED X Mini), far = the UPPER (ZED X). Roles are positions, and the
# serials are how config/{near,far}.yaml bind them.
NEAR_SN="${ZED_NEAR_SN:-59249806}"
FAR_SN="${ZED_FAR_SN:-44889004}"
FRAMES="${ZED_FRAMES:-300}"

mkdir -p "$OUT"
# The barrier has to clear a camera open, which is ~7 s and occasionally much worse.
START=$(python3 -c 'import time; print(time.time()+45)')

for pair in "near:$NEAR_SN" "far:$FAR_SN"; do
    n=${pair%%:*}; sn=${pair##*:}
    timeout 420 docker exec "$CONTAINER" bash -c \
        "python3 /opt/zed_tools/zed_sdk_timing.py --serial $sn $* --frames $FRAMES \
         --start-at $START --json $OUT/${TAG}_${n}.json" > "$OUT/${TAG}_${n}.txt" 2>&1 &
done
wait

echo "##### $TAG : $* #####"
for n in near far; do
    python3 - "$OUT/${TAG}_${n}.json" "$n" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  {sys.argv[2]:4s} FAILED: {exc}")
    raise SystemExit
iv, ctg = d['grab_interval_ms'], d['capture_to_grab_ms']
rt = d.get('retrieve_measure_ms') or {}
print(f"  {sys.argv[2]:4s} rate={1000/iv['mean']:5.1f}Hz  capture->grab mean={ctg['mean']:6.2f} "
      f"p99={ctg['p99']:6.2f}  retrieve={rt.get('mean', 0):6.2f}  "
      f"exp={d.get('exposure_us', '?')}us auto={d.get('auto_exposure', '?')}")
PY
done
