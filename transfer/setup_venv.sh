#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Build the downstream (Isaac-free) venv for MuJoCo sim2sim + the ONNX deploy runtime.
#
#   bash transfer/setup_venv.sh            # -> <repo>/env_transfer
#   bash transfer/setup_venv.sh /path/to/env
#
# Uses uv when it is on PATH, plain python3 -m venv otherwise -- a fresh deploy box has
# neither uv nor conda, and this has to work there without a bootstrap step.
#
# Activate with:  source env_transfer/bin/activate
#
# This env deliberately does NOT contain Isaac Sim or torch; see transfer/requirements.txt
# for what is in it and why. env_isaaclab remains the env for anything under source/ or
# scripts/ that imports isaaclab.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-${REPO_ROOT}/env_transfer}"
REQUIREMENTS="${REPO_ROOT}/transfer/requirements.txt"

echo "[setup_venv] repo:         ${REPO_ROOT}"
echo "[setup_venv] venv:         ${VENV_DIR}"
echo "[setup_venv] requirements: ${REQUIREMENTS}"

if command -v uv >/dev/null 2>&1; then
    echo "[setup_venv] using uv"
    uv venv "${VENV_DIR}"
    VIRTUAL_ENV="${VENV_DIR}" uv pip install -r "${REQUIREMENTS}"
else
    echo "[setup_venv] uv not found; using python3 -m venv"
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS}"
fi

# Smoke test in the built env, not the caller's: an aarch64 box with no wheel for one of
# these installs everything else and then fails at the first import inside MuJoCo, which
# is a much worse place to find out.
"${VENV_DIR}/bin/python" - <<'PY'
import importlib
import sys

mods = ["numpy", "mujoco", "onnxruntime", "yaml", "scipy", "h5py", "pygame",
        "huggingface_hub", "matplotlib", "mcap", "mcap_ros2"]
missing = []
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<16} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001 -- reported together below
        missing.append(f"{name}: {exc}")
if missing:
    sys.exit("[setup_venv] FAILED to import: " + "; ".join(missing))

import onnxruntime as ort
print(f"  onnxruntime providers: {ort.get_available_providers()}")
PY

echo "[setup_venv] done. Activate with: source ${VENV_DIR}/bin/activate"
