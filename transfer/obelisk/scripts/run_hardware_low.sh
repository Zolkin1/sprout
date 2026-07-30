#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The full stack on HARDWARE: the brain (g1.yml) + the real G1 driver (g1_robot_hw.yml) +
# the Jetson ZED depth node.
#
# g1.yml is the SAME config run_sim_low.sh launches -- only the robot source differs (the
# real driver here, the MuJoCo world there). No FSM echo: the G1 interface publishes its
# real execution-FSM state, which is the ONLY thing that arms the controller.
#
#   --sim-test   do NOT launch the real hardware -- neither the G1 driver nor the Jetson
#                ZED node. Expect them from a desktop `run_world.sh` over the ROS graph.
#                Use this to exercise the real bring-up path with no robot powered.
#
#   window "main": controller (AUTO-RUN) | debug shell [| robot driver | ZED depth]
#
# The controller AUTO-RUNS but comes up UNARMED (the robot is in damping). Support the
# robot, walk the FSM to LOW_LEVEL_CONTROL, and it arms itself and primes the generator.
#
# NOTE: from LOW_LEVEL_CONTROL the only FSM exits are DAMPING and ESTOP (obelisk 9f52216
# removed the step back to user_pose). Plan aborts around damping.
#
# Usage:
#   scripts/run_hardware_low.sh                  brain + robot + depth
#   scripts/run_hardware_low.sh --sim-test       brain only; pair with run_world.sh
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
  --sim-test       Do NOT launch the robot driver or the Jetson ZED node.
  --policy FOLDER  Override the controller's hf_policy_folder for this run.
  --no-depth       Skip the Jetson ZED pane (for a height-scan or blind policy).
  --build          colcon build first.
  --build-clean    colcon clean rebuild first.
  -h, --help       Show this help.

BEFORE THE FIRST RUN ON A ROBOT WITH NO INTERNET:
  python3 scripts/prefetch_policy.py --hf_policy_folder <name>
The controller fetches offline-first, so a cold cache is a bring-up failure.
EOF
}

ORIG_ARGS=("$@")
SENSORS=1 ; DEPTH=1 ; POLICY="" ; BUILD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sim-test)    SENSORS=0; shift ;;
        --no-depth)    DEPTH=0; shift ;;
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
CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1.yml"
ROBOT_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_robot_hw.yml"
[ -n "$POLICY" ] && CTRL_CFG="$(obk_policy_config "$CTRL_CFG" "$POLICY")"

if tmux has-session -t "$SESSION" 2>/dev/null && [ -z "${OBK_RESTART:-}" ]; then
    exec tmux attach -t "$SESSION"
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

# TODO(logging-refactor): a hardware run currently records NOTHING. obk-launch takes
# `bag=True` (ros2 bag record, obelisk_bringup.launch.py) and no launcher here passes it,
# so if the robot does something wrong the entire forensic record is tmux scrollback -- in
# a container whose ROS_LOG_DIR is not even bind-mounted, so `stop.sh --down` destroys it.
# Also: the `dbg` pane below is a bare shell where every other pane gets its command
# pre-seeded; it wants `ros2 topic hz` on /depth/image_processed, /obelisk/g1/est_state,
# /obelisk/g1/ctrl and /Odometry -- the four rates that say the stack is alive.
# See "Observability" in transfer/obelisk/README.md.
tmux new-session  -d -s "$SESSION" -n main "$(obk_pane_run ctrl "obk-launch config=$CTRL_CFG")"
tmux split-window -h -t "$SESSION":main.0  "$(obk_pane_shell dbg)"
if [ "$SENSORS" = 1 ]; then
    tmux split-window -v -t "$SESSION":main.0 "$(obk_pane_run robot "obk-launch config=$ROBOT_CFG")"
    if [ "$DEPTH" = 1 ]; then
        # The ZED node runs in the Jetson's own container tmux, so a flaky ssh link is
        # just Up+Enter here and never takes the camera down with it.
        tmux split-window -v -t "$SESSION":main.2 "$(obk_pane_run depth "$(obk_jetson_zed_cmd)")"
    fi
fi

tmux select-window -t "$SESSION":main
tmux select-pane   -t "$SESSION":main.0
exec tmux attach   -t "$SESSION"
