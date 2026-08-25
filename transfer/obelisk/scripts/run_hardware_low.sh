#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The full stack on HARDWARE: the brain (g1_cpp.yml) + the real G1 driver (g1_robot_hw.yml) +
# the Jetson ZED depth node.
#
# g1_cpp.yml is the SAME config run_sim_low.sh launches -- only the robot source differs (the
# real driver here, the MuJoCo world there). No FSM echo: the G1 interface publishes its
# real execution-FSM state, which is the ONLY thing that arms the controller.
#
#   --sim-test   do NOT launch the real hardware -- neither the G1 driver nor the Jetson
#                ZED node. Expect them from a desktop `run_sim_world.sh` over the ROS graph.
#                Use this to exercise the real bring-up path with no robot powered.
#
#   window "main": controller (AUTO-RUN) | MCAP recorder (AUTO-RUN)
#                  [| robot driver | ZED depth]
#
# The controller AUTO-RUNS but comes up UNARMED (the robot is in damping). Support the
# robot, walk the FSM to LOW_LEVEL_CONTROL, and it arms itself and primes the generator.
#
# NOTE: from LOW_LEVEL_CONTROL the only FSM exits are DAMPING and ESTOP (obelisk 9f52216
# removed the step back to user_pose). Plan aborts around damping.
#
# FIRST RUN ON A ROBOT: use --clip. It swaps the brain for the debug-clip config, which
# loads the tracker ALONE (no generator at all) and drives it from a baked motion. The
# default `stand` case asks the robot to hold still, so it exercises the encoders, frames,
# joint order, gains and PD output with nothing that can improvise. Only once that holds is
# there any point running the generator on a real robot.
#
# Usage:
#   scripts/run_hardware_low.sh                  brain + robot + depth
#   scripts/run_hardware_low.sh --clip           ... driven by the `stand` debug clip
#   scripts/run_hardware_low.sh --clip terrain   ... or another case
#   scripts/run_hardware_low.sh --sim-test       brain only; pair with run_sim_world.sh
#   scripts/run_hardware_low.sh --policy FOLDER  override the checkpoint for this run
#   scripts/run_hardware_low.sh --build          colcon build first
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"
OBK_SESSION=lowhw

usage() {
    cat <<'EOF'
run_hardware_low.sh -- brain + real G1 driver + Jetson ZED depth (tmux session lowhw).

Options:
  --clip [CASE]    Drive the tracker from a pre-made clip with the GENERATOR NOT LOADED.
                   CASE is stand (default) | stand_walk_stand | walk_backward | terrain.
                   This is the bring-up mode: run `stand` before anything else. Runs the
                   C++ controller like every other path; add --python for the deprecated one.
                   Needs the clip PACK on this machine -- build it with
                   scripts/rsl_rl/make_debug_clips.py then
                   transfer/deploy/tools/dump_clip_pack.py (neither the .h5 nor the pack is
                   committed; they come from a clip library the robot does not carry).
  --sim-test       Do NOT launch the robot driver or the Jetson ZED node.
  --policy FOLDER  Override the controller's hf_policy_folder for this run.
  --no-depth       Skip the Jetson ZED pane (for a height-scan or blind policy).
  --no-bag         Do not record. On by default -- an unrecorded hardware run cannot be
                   explained afterwards. Bags land in transfer/obelisk/log/bags/.
  --build          colcon build first.
  --build-clean    colcon clean rebuild first.
  -h, --help       Show this help.

BEFORE THE FIRST RUN ON A ROBOT WITH NO INTERNET:
  python3 scripts/prefetch_policy.py --hf_policy_folder <name>
The controller fetches offline-first, so a cold cache is a bring-up failure.

BEFORE THE FIRST RUN ON A NEW ZED BOX (or a reinstalled one):
  scripts/setup_zed_link.sh              pubkey auth to it, from inside this container
  sudo scripts/setup_clock_sync.sh --master     then --client over there
The depth pane is an ssh to that box, so with no key it hangs -- looking like a camera
fault. This script now checks the link before opening the pane and says which it is.
EOF
}

ORIG_ARGS=("$@")
SENSORS=1 ; DEPTH=1 ; POLICY="" ; BUILD="" ; CLIP_CASE="" ; BAG=1 ; PYTHON_CTRL=""
# Which controller a clip run uses. Only read when CLIP_CASE is set, but initialised here
# anyway: this script runs under `set -u`, so a future edit that reads it on another path
# would abort the launch rather than default.
CLIP_LANG=cpp
while [ $# -gt 0 ]; do
    case "$1" in
        --sim-test)    SENSORS=0; shift ;;
        --no-depth)    DEPTH=0; shift ;;
        # --clip takes an OPTIONAL case name; bare --clip means the safest one.
        --clip)
            if [ -n "${2:-}" ] && [ "${2#-}" = "${2:-}" ]; then CLIP_CASE="$2"; shift 2
            else CLIP_CASE=stand; shift; fi ;;
        --no-bag)      BAG=""; shift ;;
        # The python controller is DEPRECATED. On this machine it is also the slow one --
        # aarch64 rclpy is cp310 and the only Thor onnxruntime GPU wheel is cp312, so it
        # cannot reach the GPU at all (~471 ms per re-plan against a 20 ms period).
        --python)      PYTHON_CTRL=1; shift ;;
        --cpp)         shift ;;   # the default now; kept so old commands still run
        --policy)      POLICY="${2:-}"; shift 2 ;;
        --build)       BUILD=1; shift ;;
        --build-clean) BUILD=clean; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "run_hardware_low.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

if [ -z "${OBK_IN_CONTAINER:-}" ]; then
    if [ "$BUILD" = clean ]; then "$HERE/build.sh" --clean; fi
    if [ "$BUILD" = 1 ];     then "$HERE/build.sh"; fi
    obk_host_launch "${BASH_SOURCE[0]}" "${ORIG_ARGS[@]}"
    exit 0
fi

SESSION="$OBK_SESSION"
# --clip swaps the whole brain config for the debug-clip one, which loads the tracker ALONE
# (no fm.onnx) and drives it from a baked motion. On a robot that is the safe way in: the
# `stand` case asks it to hold still, so it exercises encoders, frames, joint order, gains
# and PD output without the generator ever having a say.
if [ -n "$CLIP_CASE" ] && [ -z "$PYTHON_CTRL" ]; then
    # C++, like every other path here. The runtime plays clips from a binary PACK
    # (transfer/deploy/tools/dump_clip_pack.py) rather than the .h5, because the image ships
    # no HDF5 -- so nothing about clip mode needs the deprecated controller any more.
    CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_cpp_clip.yml"
    CLIP_LANG=cpp
elif [ -n "$CLIP_CASE" ]; then
    echo "[run_hardware_low] WARNING: --clip --python is the DEPRECATED controller. It reads" >&2
    echo "[run_hardware_low]          the .h5 directly and cannot reach this machine's GPU." >&2
    CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_debug_clip.yml"
    CLIP_LANG=python
elif [ -n "$PYTHON_CTRL" ]; then
    echo "[run_hardware_low] WARNING: the python controller is DEPRECATED and cannot reach" >&2
    echo "[run_hardware_low]          this machine's GPU. Drop --python for the C++ one." >&2
    CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1.yml"
else
    CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_cpp.yml"
fi
ROBOT_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_robot_hw.yml"
[ -n "$POLICY" ] && CTRL_CFG="$(obk_policy_config "$CTRL_CFG" "$POLICY")"
# The LANGUAGE decides which clip parameters the overlay emits -- the C++ node takes a pack
# directory, the python one an .h5 -- and an override for a parameter the node never declares
# is silently ignored. See obk_clip_config.
[ -n "$CLIP_CASE" ] && CTRL_CFG="$(obk_clip_config "$CTRL_CFG" "$CLIP_CASE" "" "" "" "" "$CLIP_LANG")"

# The depth pane is an ssh to the ZED box (obk_jetson_zed_cmd), so check the link BEFORE
# building a session around it. A missing key or a stale OBK_JETSON_HOST otherwise shows up
# as a pane that never prints a frame, which on a robot reads as a camera fault and sends
# you to the cameras. obk_jetson_preflight names the actual cause and the fix for it.
#
# Fatal on purpose: every policy this stack runs is a DEPTH policy -- the tracker takes depth
# images even in --clip mode, where only the generator is absent -- so a run with no camera
# is not a degraded run, it is a robot acting on a stale image. --no-depth is the deliberate
# way past this, and is also the right flag when the far side is already up and you only
# want to skip supervising it.
if [ "$SENSORS" = 1 ] && [ "$DEPTH" = 1 ]; then
    if ! obk_jetson_preflight; then
        echo "[run_hardware_low] refusing to start: the depth pane cannot come up." >&2
        exit 1
    fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null && [ -z "${OBK_RESTART:-}" ]; then
    exec tmux attach -t "$SESSION"
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

# The recorder runs in its own pane rather than through obk-launch's `bag=True`, which
# hardcodes `-a` and the default sqlite3 storage. See obk_bag_cmd for why MCAP and why an
# explicit topic list. It writes under $REPO_ROOT (bind-mounted); obelisk's own ROS_LOG_DIR
# is not, so `stop.sh --down` would destroy anything left there.
BAG_CMD="$(obk_bag_cmd "hw${CLIP_CASE:+_clip_$CLIP_CASE}")"
tmux new-session  -d -s "$SESSION" -n main "$(obk_pane_run ctrl "obk-launch config=$CTRL_CFG")"
if [ -n "$BAG" ]; then
    tmux split-window -h -t "$SESSION":main.0 "$(obk_pane_run bag "$BAG_CMD")"
else
    tmux split-window -h -t "$SESSION":main.0 "$(obk_pane bag "$BAG_CMD")"
fi
if [ "$SENSORS" = 1 ]; then
    tmux split-window -v -t "$SESSION":main.0 "$(obk_pane_run robot "obk-launch config=$ROBOT_CFG")"
    if [ "$DEPTH" = 1 ]; then
        # The ZED node runs in the Jetson's own container tmux, so a flaky ssh link is
        # just Up+Enter here and never takes the camera down with it.
        tmux split-window -v -t "$SESSION":main.2 "$(obk_pane_run depth "$(obk_jetson_zed_cmd)")"
        # Record that a node went onto the ZED box. Without this marker `stop.sh --zed`
        # skips the remote teardown silently -- it tests for the marker before spending an
        # ssh -- and the pair stays holding the cameras after the stack is down.
        obk_mark_jetson 1
    else
        obk_mark_jetson 0
    fi
else
    obk_mark_jetson 0
fi

tmux select-window -t "$SESSION":main
tmux select-pane   -t "$SESSION":main.0
exec tmux attach   -t "$SESSION"
