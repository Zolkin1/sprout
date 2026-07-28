#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Seed the ZED's per-camera factory calibration onto the Jetson.
#
# The SDK fetches settings/SN<serial>.conf from stereolabs.com the first time it opens a
# camera. The robot LAN has no DNS, so on the robot that fetch hangs ~20 s and then FAILS,
# and camera open fails with it. The file is per-serial and lives in the zed_settings
# volume, so it survives down/up -- but a fresh Jetson, a new camera, or a `docker volume rm`
# needs it seeded again.
#
# RUN THIS FROM A MACHINE WITH INTERNET (the dev box), not from the robot.
#
#   ./seed_zed_calib.sh <serial>                      # serial from the camera or an SDK log
#   JETSON_HOST=unitree@192.168.123.164 ./seed_zed_calib.sh 12345678
set -euo pipefail

SERIAL="${1:-}"
JETSON_HOST="${JETSON_HOST:-unitree@192.168.123.164}"
CONTAINER="${JETSON_CONTAINER:-zed_depth}"

if [ -z "$SERIAL" ]; then
    echo "usage: $0 <camera-serial>" >&2
    echo "  The serial is printed by the SDK when the calibration fetch fails; on the" >&2
    echo "  Jetson: docker exec $CONTAINER run_node.sh zed   and read the SN= in the error." >&2
    exit 2
fi

conf="SN${SERIAL}.conf"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "[seed] downloading $conf ..."
curl -fsS -L "http://www.stereolabs.com/developers/calib/?SN=${SERIAL}" \
     -A 'Mozilla/4.0' -o "$tmp/$conf"

# The endpoint returns an HTML error page with HTTP 200 for an unknown serial, so check the
# content rather than the status code -- otherwise you ship an HTML file to the robot and
# the failure resurfaces there as an unhelpful camera-open error.
if ! head -1 "$tmp/$conf" | grep -q '^\[LEFT_CAM'; then
    echo "[seed] ERROR: that is not a calibration file (wrong serial?). First line:" >&2
    head -1 "$tmp/$conf" >&2
    exit 1
fi

echo "[seed] copying to $JETSON_HOST:$CONTAINER ..."
scp -o StrictHostKeyChecking=accept-new "$tmp/$conf" "$JETSON_HOST:~/$conf"
ssh -o StrictHostKeyChecking=accept-new "$JETSON_HOST" \
    "docker cp ~/$conf $CONTAINER:/usr/local/zed/settings/$conf && rm -f ~/$conf"
echo "[seed] done. Persisted in the zed_settings volume; survives down/up."
