#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The BRAIN in simulation: g1.yml (controller + estimator + joystick FSM), and nothing else.
#
# This is the SAME file, launched the SAME way, that run_hardware_low.sh runs on the robot.
# The robot half comes from a separately launched config, so what you test in sim is
# literally what runs on hardware rather than a sim-flavoured variant of it.
#
#   sim       ->  scripts/run_sim_world.sh   (MuJoCo + the FSM echo)
#   hardware  ->  scripts/run_hardware_low.sh
#
# LAUNCH THIS FIRST, THEN run_sim_world.sh, and the ordering is not cosmetic. Loading the
# policy pair takes ~10 s (ONNX + CUDA init) while the MuJoCo world activates in ~2 s, and
# MuJoCo has no `user_pose` equivalent holding the robot up. Start the world first and it
# collapses on the floor before the controller can be armed -- you then measure a policy
# recovering from a heap, which it was never trained to do.
#
#   window "main": controller (AUTO-RUN) | MCAP recorder (AUTO-RUN)
#   window "viz" (--rviz only): FM plan markers (AUTO-RUN)
#
# Usage:
#   scripts/run_sim_low.sh                       the brain, recorded to MCAP
#   scripts/run_sim_low.sh --no-bag              ... without recording
#   scripts/run_sim_low.sh --rviz                ... with RViz + the FM plan ribbon
#   scripts/run_sim_low.sh --policy <hf_folder>  override the checkpoint for this run
#   scripts/run_sim_low.sh --build               colcon build first (needed on first run)
#
# To take the generator OUT of the loop and drive the tracker from a pre-made clip, use
# scripts/run_sim_clip.sh instead. That is the first thing to reach for when this falls.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"
OBK_SESSION=simlow

usage() {
    cat <<'EOF'
run_sim_low.sh -- the BRAIN alone (g1.yml), in tmux session `simlow`.

Options:
  --cpp            Use the C++ controller (g1_cpp.yml) instead of the python one.
                   On aarch64 this is the only one that can reach the GPU: rclpy is a
                   cp310 C extension and the only onnxruntime GPU build for a Thor is
                   cp312, so the python node runs the generator on the CPU at ~471 ms
                   per re-plan against a 20 ms period. The C++ one runs it at ~16 ms,
                   on its own thread.
  --policy FOLDER  Override the controller's hf_policy_folder for this run.
  --rviz           Turn the viz block on: RViz showing the robot in its estimated pose
                   (default_robot_viz + robot_state_publisher + the g1_model URDF), plus
                   the generator's predicted root path and FK'd foot paths as markers.
                   The first three come from obelisk's own bringup; only the marker node
                   (g1_control fm_plan_viz) gets a pane, in a second tmux window.
                   To view from a LAPTOP instead, still pass this -- everything the remote
                   RViz needs (/robot_description, /joint_states, TF, the markers) goes
                   out over DDS, and the local rviz2 is simply one more window. Obelisk's
                   bringup starts rviz2 unconditionally with the viz block, so there is no
                   way to ask for the publishers without it.
  --no-bag         Do NOT record. Recording is ON by default -- a run you cannot
                   explain afterwards is a run you have to repeat. Bags go to
                   $REPO_ROOT/transfer/obelisk/log/bags/<timestamp>_fm/.
  --build          colcon build first.
  --build-clean    colcon clean rebuild first.
  -h, --help       Show this help.

Run this FIRST, then `scripts/run_sim_world.sh` in another terminal (or detach with
Ctrl-b d). The world is a separate launch on purpose: the brain needs ~10 s to load the
policy, and the robot free-falls the moment the world comes up.

There is no --scene here; the scene belongs to the world (run_sim_world.sh --scene NAME).

The controller comes up DISARMED; run_sim_world.sh auto-arms it. There is no PD warm-up
-- the generator drives the tracker from the first tick, sampled through its learned
null-history embedding until the history fills, exactly as in training.
EOF
}

ORIG_ARGS=("$@")
POLICY="" ; BUILD="" ; BAG=1 ; CPP="" ; RVIZ=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cpp)         CPP=1; shift ;;
        --rviz)        RVIZ=1; shift ;;
        --policy)      POLICY="${2:-}"; shift 2 ;;
        --no-bag)      BAG=""; shift ;;
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
# One line apart, deliberately: the two controllers share every other part of the config, so
# a difference between them cannot be a difference in what else was launched.
if [ -n "$CPP" ]; then
    CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1_cpp.yml"
else
    CTRL_CFG="$ROBOT_RL_ROOT/g1_control/config/g1.yml"
fi
[ -n "$POLICY" ] && CTRL_CFG="$(obk_policy_config "$CTRL_CFG" "$POLICY")"
# After --policy: each overlay includes the previous one by absolute path, so the order here
# is the include chain and both flags compose.
[ -n "$RVIZ" ] && CTRL_CFG="$(obk_viz_config "$CTRL_CFG")"

if tmux has-session -t "$SESSION" 2>/dev/null && [ -z "${OBK_RESTART:-}" ]; then
    exec tmux attach -t "$SESSION"
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session  -d -s "$SESSION" -n main "$(obk_pane_run ctrl "obk-launch config=$CTRL_CFG")"
# Recording by default, like run_sim_clip.sh and run_hardware_low.sh. It used to be opt-in
# on the theory that a generator run is the one you iterate on -- but the run you most want
# a recording of is the one that surprised you, and by then it is over.
if [ -n "$BAG" ]; then
    tmux split-window -h -t "$SESSION":main.0 "$(obk_pane_run bag "$(obk_bag_cmd fm)")"
else
    tmux split-window -h -t "$SESSION":main.0 "$(obk_pane dbg "$(obk_bag_cmd fm)")"
fi

# A SECOND window, so --rviz does not disturb the layout of `main`. Only the two marker /
# cloud nodes need panes: default_robot_viz, robot_state_publisher and rviz2 are all
# launched by obelisk's bringup in the controller pane's launch, off the viz block.
if [ -n "$RVIZ" ]; then
    # depth_cloud_viz resolves the camera contract from tracker.onnx, so it has to be told
    # WHICH policy -- and the same one the controller runs, or it would draw a different
    # camera than the one feeding the network. --policy wins; otherwise read the config's
    # own default rather than restating it here, so the two cannot drift.
    VIZ_POLICY="$POLICY"
    if [ -z "$VIZ_POLICY" ]; then
        VIZ_POLICY="$(awk '/^[[:space:]]*hf_policy_folder:/ {gsub(/"/, "", $2); print $2; exit}' \
                      "$ROBOT_RL_ROOT/g1_control/config/common/low_level.yml")"
    fi
    tmux new-window   -t "$SESSION" -n viz \
        "$(obk_pane_run fmviz "ros2 run g1_control fm_plan_viz")"
    tmux split-window -h -t "$SESSION":viz.0 \
        "$(obk_pane_run depthviz "ros2 run g1_control depth_cloud_viz --ros-args -p hf_policy_folder:=$VIZ_POLICY")"
fi

tmux select-window -t "$SESSION":main
tmux select-pane   -t "$SESSION":main.0
exec tmux attach   -t "$SESSION"
