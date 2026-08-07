#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The ROBOT half in simulation: the MuJoCo world (g1_world.yml) + the FSM echo. No brain.
#
# This is the sim counterpart of the real robot -- it publishes exactly what the hardware
# publishes (joint encoders, pelvis IMU, both depth images, /livox/*, /Odometry), so the
# brain cannot tell the difference. Pairs with:
#
#   scripts/run_sim_low.sh                  the brain, on this machine (LAUNCH IT FIRST)
#   scripts/run_hardware_low.sh --sim-test  the brain on the minipc, for the split-machine
#                                           test with no robot powered
#
# It is also how you restart the world without cycling a brain that took a careful
# bring-up to arm.
#
# MuJoCo has no execution FSM, so `fsm_state_echo` republishes the joystick's commanded
# state as the achieved one; without it the controller would never arm. It lives HERE and
# not with the brain because on hardware that state comes from the ROBOT -- keeping the
# split honest means the sim robot supplies it too. (It is also why running it in both
# would collide: two nodes of the same name on one graph.)
#
# Usage:
#   scripts/run_sim_world.sh [--scene NAME] [--build|--build-clean]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"
OBK_SESSION=simworld

ORIG_ARGS=("$@")
SCENE="" ; BUILD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --scene)       SCENE="${2:-}"; shift 2 ;;
        --build)       BUILD=1; shift ;;
        --build-clean) BUILD=clean; shift ;;
        -h|--help)
            echo "run_sim_world.sh [--scene NAME] [--build|--build-clean]"
            echo "  The MuJoCo world + FSM echo, in tmux session 'simworld'."
            echo "  Launch scripts/run_sim_low.sh (the brain) FIRST -- it needs ~10 s to"
            echo "  load the policy, and the robot free-falls once this comes up."
            echo "  --scene NAME picks transfer/sim/robots/g1/<NAME>.xml; the config"
            echo "  default is the terrain scene. Available:"
            ls "$HERE/../../sim/robots/g1/"*.xml 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/^/    /'
            exit 0 ;;
        *) echo "run_sim_world.sh: unknown arg '$1'" >&2; exit 2 ;;
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
