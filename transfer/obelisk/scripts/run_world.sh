#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The MuJoCo WORLD alone (g1_world.yml) plus the FSM echo -- no brain.
#
# Two uses:
#   * pair with `run_hardware_low.sh --sim-test` on the minipc to exercise the REAL
#     bring-up path across two machines with no robot powered;
#   * restart the world without cycling a brain that took a careful bring-up to arm.
#
# Usage:
#   scripts/run_world.sh [--scene NAME] [--build|--build-clean]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"
OBK_SESSION=world

ORIG_ARGS=("$@")
SCENE="" ; BUILD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --scene)       SCENE="${2:-}"; shift 2 ;;
        --build)       BUILD=1; shift ;;
        --build-clean) BUILD=clean; shift ;;
        -h|--help)
            echo "run_world.sh [--scene NAME] [--build|--build-clean]"
            echo "  Available scenes: \$(ls transfer/sim/robots/g1/*.xml)"
            exit 0 ;;
        *) echo "run_world.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

if [ -z "${OBK_IN_CONTAINER:-}" ]; then
    if [ "$BUILD" = clean ]; then "$HERE/build.sh" --clean; fi
    if [ "$BUILD" = 1 ];     then "$HERE/build.sh"; fi
    obk_host_launch "${BASH_SOURCE[0]}" "${ORIG_ARGS[@]}"
    exit 0
fi

SESSION="$OBK_SESSION"
WORLD_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_world.yml"
[ -n "$SCENE" ] && WORLD_CFG="$(obk_scene_config "$WORLD_CFG" "$SCENE")"

if tmux has-session -t "$SESSION" 2>/dev/null && [ -z "${OBK_RESTART:-}" ]; then
    exec tmux attach -t "$SESSION"
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session  -d -s "$SESSION" -n main "$(obk_pane_run world "obk-launch config=$WORLD_CFG")"
tmux split-window -h -t "$SESSION":main.0  "$(obk_pane_run fsm "ros2 run g1_control fsm_state_echo")"
tmux select-pane  -t "$SESSION":main.0
exec tmux attach  -t "$SESSION"
