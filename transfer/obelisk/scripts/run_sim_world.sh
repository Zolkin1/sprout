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
# It AUTO-ARMS by default (SIM ONLY): the FSM echo comes up already reporting
# LOW_LEVEL_CONTROL, so the controller arms as soon as it has a state estimate and no
# joystick input is needed. MuJoCo has no `user_pose` holding the robot up, so waiting for a
# button press just means watching it collapse. A joystick command still overrides at any
# time -- damping and e-stop included -- and `--no-auto-arm` restores the hardware-like flow.
#
# This cannot leak to hardware: fsm_state_echo exists only because MuJoCo has no execution
# FSM and is launched only from here. On a robot the G1 interface publishes the real state
# and this node never runs.
#
# Usage:
#   scripts/run_sim_world.sh [--scene NAME] [--no-auto-arm] [--build|--build-clean]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"
OBK_SESSION=simworld

ORIG_ARGS=("$@")
VEL_PROFILE="" ; VEL_DURATION=""
SCENE="" ; BUILD="" ; AUTO_ARM=1
while [ $# -gt 0 ]; do
    case "$1" in
        --scene)       SCENE="${2:-}"; shift 2 ;;
        # A SCRIPTED velocity profile instead of the joystick, by name from
        # transfer/sim/experiments/velocity_commands.py -- the same names the MuJoCo sim2sim
        # runner's --command takes, from the same file. This is what makes an
        # obelisk-vs-sim2sim comparison an input-for-input one.
        --command)     VEL_PROFILE="${2:-}"; shift 2 ;;
        --duration)    VEL_DURATION="${2:-}"; shift 2 ;;
        --no-auto-arm) AUTO_ARM=""; shift ;;
        --build)       BUILD=1; shift ;;
        --build-clean) BUILD=clean; shift ;;
        -h|--help)
            echo "run_sim_world.sh [--scene NAME] [--no-auto-arm] [--build|--build-clean]"
            echo "  --no-auto-arm  Come up in INIT and wait for a joystick FSM command,"
            echo "                 the way hardware does. Default AUTO-ARMS (sim only)."
            echo "  The MuJoCo world + FSM echo, in tmux session 'simworld'."
            echo "  Launch scripts/run_sim_low.sh (the brain) FIRST -- it needs ~10 s to"
            echo "  load the policy, and the robot free-falls once this comes up."
            echo "  --command NAME drives from a scripted velocity profile instead of the"
            echo "                 joystick (experiments.velocity_commands, the same names"
            echo "                 the sim2sim runner takes). Its clock starts when the"
            echo "                 controller ARMS, not at launch."
            echo "  --duration S   hold zeros after S seconds of profile time (0 = forever)."
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
# AUTO-ARM lives here, in the SIM-ONLY FSM echo, and deliberately not as a controller
# option: this node exists only because MuJoCo has no execution FSM and is launched only by
# this script, so on hardware the capability structurally does not exist rather than merely
# being switched off. MuJoCo also has no `user_pose` holding the robot up, so every second
# the controller spends disarmed is a second of the robot collapsing -- waiting for a button
# press is the wrong default in sim even though it is the only safe one on a robot.
FSM_CMD="ros2 run g1_control fsm_state_echo"
[ -n "$AUTO_ARM" ] && FSM_CMD="$FSM_CMD --ros-args -p initial_state:=3"
tmux split-window -h -t "$SESSION":main.0  "$(obk_pane_run fsm "$FSM_CMD")"
# The scripted command, when asked for. Its own pane so a run can be read afterwards: the
# profile it used, and when its clock started, are printed there. Launched from the WORLD side
# because it stands in for the joystick, which lives here -- the brain is unchanged either way,
# so a scripted run and a driven run put the same controller under test.
if [ -n "$VEL_PROFILE" ]; then
    VEL_CMD="ros2 run g1_control scripted_vel_cmd --ros-args -p profile:=$VEL_PROFILE"
    # Emitted as a FLOAT literal. `-p duration:=14` is an INTEGER to rclpy and the node
    # declares a double; the node now accepts either, but sending the unambiguous form keeps
    # the intent visible in the pane's command line.
    if [ -n "$VEL_DURATION" ]; then
        VEL_CMD="$VEL_CMD -p duration:=$(printf '%.3f' "$VEL_DURATION")"
    fi
    tmux split-window -v -t "$SESSION":main.1  "$(obk_pane_run vel "$VEL_CMD")"
fi
tmux select-pane  -t "$SESSION":main.0
exec tmux attach  -t "$SESSION"
