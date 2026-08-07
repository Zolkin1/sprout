#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The BRAIN in simulation with the GENERATOR REMOVED: g1_debug_clip.yml drives the tracker
# from a pre-made motion clip instead. Records to MCAP by default.
#
# This is the diagnostic to run FIRST when the normal stack falls over. It removes the flow
# matching generator from the loop entirely -- the fm.onnx is not even loaded -- so what is
# left is the tracker plus everything this stack does around it: encoders in, estimator,
# frames, joint order, gains, depth, timing, PD targets out. Walk the cases upward:
#
#   --case stand             hold still. The reference NEVER MOVES. If the robot cannot do
#                            this, the fault is in this stack and the generator is
#                            irrelevant.
#   --case stand_walk_stand  the same clip untruncated: stand, walk, stand. It shares a
#                            bit-identical prefix with `stand`, so if that one held and this
#                            one falls, the difference is the walk transition and nothing
#                            else.
#   --case terrain           a stairs transition. Adds the depth cameras and terrain
#                            anchoring; pair with `run_sim_world.sh --scene terrain_scene`.
#
# Each case is known to survive in MuJoCo through the SAME deploy runtime
# (transfer/sim/gen_tracker_runner.py --clip --clip_case <case>), so a case that falls here
# and not there is a difference in this stack, not in the policy or the clip. Run the MuJoCo
# one first if you have any doubt -- it is the control.
#
#   window "main": controller (AUTO-RUN) | recorder (AUTO-RUN) | debug shell
#
# Usage:
#   scripts/run_sim_clip.sh                            stand, recorded
#   scripts/run_sim_clip.sh --case stand_walk_stand    then this
#   scripts/run_sim_clip.sh --case terrain             ... with --scene terrain_scene
#   scripts/run_sim_clip.sh --start_time 11.0          skip most of the standing prefix
#   scripts/run_sim_clip.sh --no-bag                   do not record
#
# THEN, in a second terminal:
#   scripts/run_sim_world.sh [--scene terrain_scene]
#
# Order is load-bearing for the same reason as run_sim_low.sh: the brain takes ~10 s to come
# up and MuJoCo has nothing holding the robot upright, so a world-first start hands the
# policy an already-collapsed robot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"
OBK_SESSION=simclip

usage() {
    cat <<'EOF'
run_sim_clip.sh -- the brain with a baked clip instead of the generator, in tmux `simclip`.

Options:
  --case NAME      stand (default) | stand_walk_stand | terrain. Resolved through the clip
                   library's JSON sidecar, so names survive a library rebuild.
  --clip PATH      Clip library .h5, relative to the repo root
                   (default: the config's
                   transfer/obelisk/g1_control/resource/debug_clips/g1_debug_v1.h5).
                   It must be INSIDE the repo: the container mounts only $REPO_ROOT,
                   and in a worktree `clips` is a symlink out of the tree.
  --start_time S   Seconds into the clip to begin (default 0). The stand cases open with
                   ~12.5 s of standing; pass 11.0 to reach the walk quickly.
  --world-anchor   Turn OFF ego re-anchoring (the MuJoCo parity mode). Only meaningful if
                   the robot happens to start on the clip's own pose -- on hardware it does
                   not, which is why ego anchoring is the default here.
  --policy FOLDER  Override the controller's hf_policy_folder for this run.
  --no-bag         Do not record. The record command is still seeded in the third pane.
  --build          colcon build first.
  --build-clean    colcon clean rebuild first.
  -h, --help       Show this help.

Run this FIRST, then `scripts/run_sim_world.sh` in another terminal.
The controller comes up DISARMED; run_sim_world.sh auto-arms it. There is no PD warm-up:
the clip drives the tracker from the first tick, as it does in MuJoCo. The joint offset
between the robot and the clip's first frame is logged at handover; a large one is a
finding, not a nuisance.
EOF
}

ORIG_ARGS=("$@")
CASE="" ; CLIP="" ; START="" ; POLICY="" ; BUILD="" ; BAG=1 ; EGO=""
while [ $# -gt 0 ]; do
    case "$1" in
        --case)         CASE="${2:-}"; shift 2 ;;
        --clip)         CLIP="${2:-}"; shift 2 ;;
        --start_time)   START="${2:-}"; shift 2 ;;
        --world-anchor) EGO=false; shift ;;
        --policy)       POLICY="${2:-}"; shift 2 ;;
        --no-bag)       BAG=""; shift ;;
        --build)        BUILD=1; shift ;;
        --build-clean)  BUILD=clean; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "run_sim_clip.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

if [ -z "${OBK_IN_CONTAINER:-}" ]; then
    if [ "$BUILD" = clean ]; then "$HERE/build.sh" --clean; fi
    if [ "$BUILD" = 1 ];     then "$HERE/build.sh"; fi
    obk_host_launch "${BASH_SOURCE[0]}" "${ORIG_ARGS[@]}"
    exit 0
fi

SESSION="$OBK_SESSION"
CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_debug_clip.yml"
[ -n "$POLICY" ] && CTRL_CFG="$(obk_policy_config "$CTRL_CFG" "$POLICY")"
if [ -n "$CASE$CLIP$START$EGO" ]; then
    CTRL_CFG="$(obk_clip_config "$CTRL_CFG" "$CASE" "$CLIP" "$START" "$EGO")"
fi

if tmux has-session -t "$SESSION" 2>/dev/null && [ -z "${OBK_RESTART:-}" ]; then
    exec tmux attach -t "$SESSION"
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

BAG_CMD="$(obk_bag_cmd "clip${CASE:+_$CASE}")"
tmux new-session  -d -s "$SESSION" -n main "$(obk_pane_run ctrl "obk-launch config=$CTRL_CFG")"
if [ -n "$BAG" ]; then
    tmux split-window -h -t "$SESSION":main.0 "$(obk_pane_run bag "$BAG_CMD")"
else
    tmux split-window -h -t "$SESSION":main.0 "$(obk_pane bag "$BAG_CMD")"
fi
tmux split-window -v -t "$SESSION":main.1 "$(obk_pane_shell dbg)"

tmux select-window -t "$SESSION":main
tmux select-pane   -t "$SESSION":main.0
exec tmux attach   -t "$SESSION"
