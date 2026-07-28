#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Build the ZED SDK's per-device TensorRT engines for the NEURAL depth modes.
#
# Why this exists as a separate step: the SDK otherwise downloads and optimises the engine
# LAZILY, on the first `open()` that needs it. On the robot that is the worst possible
# moment -- it needs internet (the robot LAN has none) and it occupies the camera, which is
# on the Jetson's only USB-C port. `optimizeAIModel` runs on an UNOPENED camera object, so
# this needs the GPU and a network route but NOT the camera: run it on the bench.
#
# Engines land in /usr/local/zed/resources, a named volume, so they survive down/up.
#
#   optimize_ai_models.sh              all three NEURAL tiers
#   optimize_ai_models.sh NEURAL       just one
set -euo pipefail

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(NEURAL_LIGHT NEURAL NEURAL_PLUS)
fi

for model in "${MODELS[@]}"; do
    echo "[optimize] $model ..."
    python3 - "$model" <<'PY'
import sys
import pyzed.sl as sl

name = sys.argv[1]
mode = getattr(sl.DEPTH_MODE, name, None)
if mode is None:
    raise SystemExit(f"unknown depth mode {name!r}; expected one of "
                     f"{[m.name for m in sl.DEPTH_MODE]}")
# No camera is opened: this only needs the GPU and a route to stereolabs.com.
status = sl.Camera.optimize_ai_model(mode)
if status != sl.ERROR_CODE.SUCCESS:
    raise SystemExit(f"optimize_ai_model({name}) failed: {status}")
print(f"[optimize] {name} OK")
PY
done
echo "[optimize] done; engines cached in /usr/local/zed/resources (persisted volume)."
