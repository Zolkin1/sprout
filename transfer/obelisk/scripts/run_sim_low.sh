#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The full stack in SIMULATION: the brain (g1.yml) + the MuJoCo world (g1_world.yml) +
# the FSM echo, in one tmux session.
#
# The brain config launched here is the SAME FILE run_hardware_low.sh launches on the
# robot -- only the world differs. That is the point of the split: what you test in sim is
# literally what runs on hardware, not a sim-flavoured variant of it.
#
# MuJoCo has no execution FSM, so `fsm_state_echo` republishes the joystick's commanded
# state as the achieved one; without it the controller would never arm.
#
#   window "main": controller (AUTO-RUN) | world (AUTO-RUN) | fsm echo (AUTO-RUN) | shell
#
# Usage:
#   scripts/run_sim_low.sh                       brain + world + echo
#   scripts/run_sim_low.sh --scene g1_29j_basic_scene
#   scripts/run_sim_low.sh --policy <hf_folder>  override the checkpoint for this run
#   scripts/run_sim_low.sh --build               colcon build first (needed on first run)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"
OBK_SESSION=simlow

usage() {
    cat <<'EOF'
run_sim_low.sh -- brain + MuJoCo world + FSM echo, in one tmux session (simlow).

Options:
  --scene NAME     transfer/sim/robots/g1/<NAME>.xml instead of the config's default.
  --policy FOLDER  Override the controller's hf_policy_folder for this run.
  --build          colcon build first.
  --build-clean    colcon clean rebuild first.
  -h, --help       Show this help.

The controller comes up DISARMED. Walk the joystick FSM to LOW_LEVEL_CONTROL to arm it;
it then primes the generator for warmup_time before the policy takes over.
EOF
}

ORIG_ARGS=("$@")
SCENE="" ; POLICY="" ; BUILD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --scene)       SCENE="${2:-}"; shift 2 ;;
        --policy)      POLICY="${2:-}"; shift 2 ;;
        --build)       BUILD=1; shift ;;
        --build-clean) BUILD=clean; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "run_sim_low.sh: unknown arg '$1'" >&2; exit 2 ;;
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
WORLD_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_world.yml"
[ -n "$POLICY" ] && CTRL_CFG="$(obk_policy_config "$CTRL_CFG" "$POLICY")"
[ -n "$SCENE" ]  && WORLD_CFG="$(obk_scene_config "$WORLD_CFG" "$SCENE")"

if tmux has-session -t "$SESSION" 2>/dev/null && [ -z "${OBK_RESTART:-}" ]; then
    exec tmux attach -t "$SESSION"
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

# 2x2: controller | world on top, fsm echo | debug shell below. tmux renumbers panes by
# POSITION after each split, so once the left column (.0) is split the right column's
# index shifts from .1 to .2 -- hence the .2 target.
tmux new-session  -d -s "$SESSION" -n main "$(obk_pane_run ctrl  "obk-launch config=$CTRL_CFG")"
tmux split-window -h -t "$SESSION":main.0  "$(obk_pane_run world "obk-launch config=$WORLD_CFG")"
tmux split-window -v -t "$SESSION":main.0  "$(obk_pane_run fsm   "ros2 run g1_control fsm_state_echo")"
tmux split-window -v -t "$SESSION":main.2  "$(obk_pane_shell dbg)"

tmux select-window -t "$SESSION":main
tmux select-pane   -t "$SESSION":main.0
exec tmux attach   -t "$SESSION"
