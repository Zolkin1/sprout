# shellcheck shell=bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Shared helpers for the obk launch scripts (run_sim_low.sh, run_sim_world.sh,
# run_hardware_low.sh, stop.sh). Source this from a variant script.
#
# Each variant runs in two halves:
#
#   HOST      -> obk_host_launch "$0": docker compose up -d, then re-exec THIS script
#                inside the container with OBK_IN_CONTAINER=1 over a TTY.
#   CONTAINER -> the variant builds a tmux session out of "seeded" panes and attaches.
#
# Pane model: every pane is an interactive `bash --rcfile <tmpfile>` whose rcfile does
#   source ~/.bashrc     # ROS + defines obk / obk-launch
#   obk                  # obelisk env (RMW, ROS_DOMAIN_ID) + sources obelisk_ws
#   source $ROBOT_RL_ROOT/install/setup.bash   # this workspace
#   history -s '<cmd>'   # the pane's target command, queued in history
# so a pane comes up fully sourced at a clean [obk] prompt with its command one Up-arrow
# away. A node that dies therefore leaves you somewhere useful instead of killing the pane.

OBK_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # .../obelisk/scripts
OBK_DOCKER_DIR="$(cd "$OBK_SCRIPTS_DIR/../docker" && pwd)"
OBK_SERVICE="${OBK_SERVICE:-sim}"

# Make docker/.env AUTHORITATIVE for compose interpolation.
#
# Compose resolves `$VAR` from the SHELL environment FIRST and only then from .env, so any
# of these exported in a login shell silently overrides the file that is documented as the
# source of truth. That is not hypothetical: a stale `export ROBOT_RL_ROOT=...` in ~/.bashrc
# (pointing at a previous repo layout) produced a container whose $ROBOT_RL_ROOT named a
# directory that did not exist, while $REPO_ROOT -- not exported, so read from .env -- was
# correct. The mount was right and the paths inside it were wrong, which reads as "the repo
# is not mounted" and sends you looking in the wrong place entirely.
#
# Exporting .env here means the shell env and .env AGREE, so precedence stops mattering.
# Only affects this shell and the compose/exec children it spawns.
#
# Assigned line by line rather than `set -a; . .env`: bash's UID is READONLY, so sourcing
# the file errors on that line -- and under the callers' `set -e` that aborts the launcher
# outright. Skipping UID costs nothing, because bash does not export UID either, so compose
# finds no shell value and reads it from .env directly, which is what we want anyway.
if [ -z "${OBK_IN_CONTAINER:-}" ] && [ -f "$OBK_DOCKER_DIR/.env" ]; then
    while IFS= read -r _obk_line || [ -n "$_obk_line" ]; do
        case "$_obk_line" in ''|'#'*|*[!A-Za-z0-9_]*=*) continue ;; esac
        case "${_obk_line%%=*}" in UID|EUID|PPID|SHELLOPTS|BASHOPTS) continue ;; esac
        export "${_obk_line?}"
    done < "$OBK_DOCKER_DIR/.env"
    unset _obk_line
fi

# Compose file selection. An explicit OBK_COMPOSE_FILE always wins. Otherwise auto-detect
# on the HOST: the GPU compose file requests `runtime: nvidia`, which docker can only
# honour with the nvidia container runtime registered; without it `docker compose up`
# errors out, so fall back. In-container the value is unused (compose only runs host-side).
if [ -z "${OBK_COMPOSE_FILE:-}" ]; then
    if [ -n "${OBK_IN_CONTAINER:-}" ]; then
        OBK_COMPOSE_FILE="docker-compose.yml"
    elif docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then
        OBK_COMPOSE_FILE="docker-compose.yml"
    else
        OBK_COMPOSE_FILE="docker-compose-no-gpu.yml"
        echo "[obk] no nvidia docker runtime detected -> using $OBK_COMPOSE_FILE" \
             "(set OBK_COMPOSE_FILE to override)" >&2
    fi
fi

# ---------------------------------------------------------------------------
# HOST side
# ---------------------------------------------------------------------------

# obk_build [--clean]: bring the container up and build the colcon workspace inside it.
# --clean removes build/ install/ log/ first.
obk_build() {
    cd "$OBK_DOCKER_DIR" || return 1   # so docker compose picks up docker/.env
    local dc=(docker compose -f "$OBK_COMPOSE_FILE")
    echo "[obk] bringing up container ($OBK_SERVICE)..."
    "${dc[@]}" up -d
    if [ "${1:-}" = --clean ]; then
        echo "[obk] cleaning colcon workspace (rm build/ install/ log/)..."
        "${dc[@]}" exec -T "$OBK_SERVICE" bash -c 'cd "$ROBOT_RL_ROOT" && rm -rf build install log'
    fi
    echo "[obk] building in container: obk + colcon build --symlink-install ..."
    # `bash -c` (not -ic) avoids "no job control" warnings under exec -T; ~/.bash_aliases
    # supplies `obk`, which sets the obelisk env and sources obelisk_ws so colcon finds
    # the obelisk messages.
    "${dc[@]}" exec -T "$OBK_SERVICE" bash -c '
        set -e
        source ~/.bash_aliases 2>/dev/null || true
        obk
        cd "$ROBOT_RL_ROOT"
        colcon build --symlink-install
    ' || { echo "[obk] build FAILED"; return 1; }
}

# obk_host_launch <abs path of the calling variant script> [args...]
# Brings the container up, then re-execs the variant inside it (the repo is bind-mounted at
# the same path), forwarding any extra args. Does NOT build; pass --build to the variant.
obk_host_launch() {
    local self name
    self="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    name="$(basename "$self")"
    shift

    cd "$OBK_DOCKER_DIR" || return 1
    local dc=(docker compose -f "$OBK_COMPOSE_FILE")

    echo "[obk] bringing up container ($OBK_SERVICE)..."
    "${dc[@]}" up -d

    # Re-attach fast path. OBK_RESTART=1 forces a fresh session.
    if [ -n "${OBK_SESSION:-}" ] && [ -z "${OBK_RESTART:-}" ] \
       && "${dc[@]}" exec -T "$OBK_SERVICE" bash -c "tmux has-session -t '$OBK_SESSION' 2>/dev/null"; then
        echo "[obk] session '$OBK_SESSION' already running; attaching (OBK_RESTART=1 for a fresh one)."
        "${dc[@]}" exec -it "$OBK_SERVICE" bash -c "tmux attach -t '$OBK_SESSION'"
        return 0
    fi

    # Stale-launch guard. An ORPHANED obelisk bring-up from a previous session poisons the
    # new one: every config reuses the same node names, so a surviving launch reacts to the
    # new nodes' lifecycle events and re-emits configure/activate at them, which is an
    # invalid transition on an already-active node and kills it. Only true orphans are
    # flagged -- a launch whose pane died reparents to PID 1, while a legitimately
    # coexisting session (a world next to a brain) keeps its pane shell as parent.
    local stale
    stale="$("${dc[@]}" exec -T "$OBK_SERVICE" bash -c '
        for p in $(pgrep -f "obelisk_bringup[.]launch[.]py" 2>/dev/null); do
            [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")" = "1" ] && ps -o pid=,cmd= -p "$p"
        done' || true)"
    if [ -n "$stale" ]; then
        echo "[obk] WARNING: ORPHANED obelisk launch process(es) alive in the container:"
        echo "$stale"
        echo "[obk]          these WILL kill the new session's nodes. Run scripts/stop.sh"
        echo "[obk]          first. Continuing in 5s (Ctrl-C to abort)..."
        sleep 5
    fi

    echo "[obk] starting tmux session (attach; detach with Ctrl-b then d)..."
    "${dc[@]}" exec -it "$OBK_SERVICE" \
        env OBK_IN_CONTAINER=1 OBK_VARIANT="$name" \
        bash -c 'bash "$ROBOT_RL_ROOT/scripts/$OBK_VARIANT" "$@"' _ "$@"
}

# ---------------------------------------------------------------------------
# CONTAINER side (tmux pane construction)
# ---------------------------------------------------------------------------

# Pre-create obelisk's shared log directory, because two obk-launches racing to create it
# is a coin flip over whether the stack comes up.
#
# obelisk's setup_logging_dir() does `if not os.path.exists(d): os.makedirs(d)` -- a TOCTOU
# race. Every stack here launches the brain and the robot as TWO obk-launches started
# together, so on a container whose obk_logs does not exist yet they both see it missing and
# the loser dies with "[Errno 17] File exists: .../obk_logs". Only the SECOND launch is lost,
# so the symptom is a world with no brain (or vice versa) and one buried [ERROR] line.
# Deterministic to pre-create; nothing here depends on the directory being fresh.
# TODO(obelisk-upstream): `os.makedirs(..., exist_ok=True)` in launch_utils.setup_logging_dir
# removes the need for this.
if [ -n "${OBK_IN_CONTAINER:-}" ]; then
    mkdir -p "${ROS_HOME:-${OBELISK_ROOT:-$HOME/obelisk}}/obk_logs" 2>/dev/null || true
fi

# obk_mk_rcfile <name> <command> [mode]: write a per-pane rcfile that sources the full env
# and, depending on <mode>:
#   seed (default) -- queue <command> in history only (press Up then Enter).
#   run            -- queue it AND run it on startup; when it exits or you Ctrl-C you land
#                     at the [obk] prompt with the command one Up-arrow away.
# Echoes the rcfile path.
obk_mk_rcfile() {
    local name="$1" cmd="${2:-}" mode="${3:-seed}"
    local f="/tmp/obk_rc_$name"
    {
        echo 'source ~/.bashrc'
        # RE-ASSERT the container's paths AFTER ~/.bashrc, which is bind-mounted from the
        # HOST and is free to export whatever it likes. A stale `export ROBOT_RL_ROOT=...`
        # in a developer's .bashrc (pointing at a previous repo layout) broke EVERY pane of
        # EVERY run script, and did so invisibly: .bashrc returns early for non-interactive
        # shells, so `bash -lc` was unaffected and only the panes -- which are interactive --
        # picked up the stale value. `install/setup.bash` then did not exist, the workspace
        # went unsourced, and the failure surfaced three layers later as
        # "package 'g1_control' not found" from ros2 launch.
        #
        # Baked in as literals because this rcfile is generated INSIDE the container, where
        # the values came from compose (which obk_launch.sh already forces to agree with
        # docker/.env) and are correct at this moment.
        printf 'export ROBOT_RL_ROOT=%q\n' "$ROBOT_RL_ROOT"
        [ -n "${REPO_ROOT:-}" ] && printf 'export REPO_ROOT=%q\n' "$REPO_ROOT"
        echo 'obk'
        # No `2>/dev/null`: swallowing the error here is what turned a one-line "no such
        # file" into the package-not-found hunt above.
        echo 'source "$ROBOT_RL_ROOT/install/setup.bash" \
              || echo "[pane] NOT SOURCED: $ROBOT_RL_ROOT/install/setup.bash -- run scripts/build.sh"'
        if [ -n "$cmd" ]; then
            # %q keeps the history entry intact (including literal ${ROBOT_RL_ROOT}) so
            # Up-arrow shows exactly what you would have typed.
            printf 'history -s %q\n' "$cmd"
            if [ "$mode" = run ]; then
                printf '%s\n' "$cmd"
            fi
        fi
    } >"$f"
    printf '%s' "$f"
}

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

# Topics a run records. An EXPLICIT list, not `ros2 bag record -a`, for two reasons: `-a`
# also pulls /livox/lidar (a point cloud we do not use in sim) and every obelisk internal,
# and -- more importantly -- a named list is a statement of what a run is expected to
# contain, so a missing topic shows up as an empty one in the bag rather than as nothing
# at all.
#
# Between them these reconstruct the whole policy input. /obelisk/g1/est_state gives the
# proprioception terms, the depth images give the exteroception, and debug/tick carries the
# per-tick internals that exist nowhere else -- so the observation vector itself is never
# recorded, only proven via its checksum. See g1_msgs/msg/ControllerDebug.msg.
OBK_BAG_TOPICS=(
    # First-party diagnostics: the only source for what happens inside a control tick.
    /obelisk/g1/debug/tick
    /obelisk/g1/debug/fm_plan
    # The control loop proper.
    /obelisk/g1/ctrl
    /obelisk/g1/est_state
    /obelisk/g1/vel_cmd
    /obelisk/g1/exec_fsm
    /obelisk/g1/fsm_state
    # Raw sensing, as the policy received it.
    /obelisk/g1/joint_encoders
    /obelisk/g1/pelvis_imu
    /depth/near/image_processed
    /depth/far/image_processed
    /Odometry
)

# obk_bag_cmd [tag]: the `ros2 bag record` command line for a run, echoed for a tmux pane.
#
# MCAP, not the default sqlite3, for two reasons that both bite off the robot: MCAP embeds
# each message's SCHEMA in the file, so a bag full of custom obelisk messages can be read on
# a laptop with `pip install mcap` and no ROS workspace at all; and Foxglove reads MCAP
# natively while it dropped ROS 2 .db3. It is also chunked and append-only, so a bag whose
# process was killed mid-run -- i.e. every interesting one -- is recoverable.
#
# Writes under $REPO_ROOT, which IS bind-mounted, unlike obelisk's own ROS_LOG_DIR: that
# lives in the container's writable layer and `stop.sh --down` destroys it.
obk_bag_cmd() {
    local tag="${1:-run}"
    local dir="\$REPO_ROOT/transfer/obelisk/log/bags/\$(date +%Y-%m-%d_%H-%M-%S)_${tag}"
    printf 'mkdir -p "$(dirname %s)" && ros2 bag record -s mcap -o %s %s' \
        "$dir" "$dir" "${OBK_BAG_TOPICS[*]}"
}

# obk_pane <name> <command>: pane that SEEDS the command in history (Up then Enter to run).
obk_pane() { printf 'bash --rcfile %q -i' "$(obk_mk_rcfile "$1" "$2" seed)"; }

# obk_pane_run <name> <command>: like obk_pane but AUTO-RUNS on startup.
obk_pane_run() { printf 'bash --rcfile %q -i' "$(obk_mk_rcfile "$1" "$2" run)"; }

# obk_pane_shell <name>: a plain sourced shell for debugging (ros2 topic list, etc.).
obk_pane_shell() { printf 'bash --rcfile %q -i' "$(obk_mk_rcfile "$1")"; }

# obk_policy_config <base_abs_config> <hf_folder>: write a temp controller config that
# `include`s <base> and keyed-merge-overrides ONLY the low-level controller's
# hf_policy_folder. Lets --policy select a checkpoint without editing YAML. Obelisk merges
# list entries by `name:`, so redeclaring `- name: low_level` merges into that entry rather
# than adding a second controller. <base> must be ABSOLUTE (the temp lives in /tmp, and an
# include resolves relative to the including file's directory). Echoes the temp path.
obk_policy_config() {
    local base="$1" folder="$2"
    local f
    f="/tmp/obk_policy_$(basename "$base")"
    local cfgname
    cfgname="$(grep -m1 '^config:' "$base" | awk '{print $2}')"
    cfgname="${cfgname:-rl_policy}"
    cat >"$f" <<EOF
# Auto-generated by run_*.sh --policy ${folder}. Thin overlay: include the controller
# config and keyed-merge-override ONLY the policy folder.
config: ${cfgname}
include:
  - ${base}
control:
  - name: low_level
    params:
      hf_policy_folder: "${folder}"
EOF
    printf '%s' "$f"
}

# obk_clip_config <base_abs_config> <case> <clip> <start_time> <ego>: write a temp controller
# config overriding only the debug-clip selection. Empty arguments are omitted, so the base
# config's value survives -- which is why every field is emitted conditionally rather than
# with a default restated here (a default in two places is a default that will disagree).
obk_clip_config() {
    local base="$1" case="$2" clip="$3" start="$4" ego="$5"
    local f="/tmp/obk_clip_$(basename "$base")"
    local cfgname
    cfgname="$(grep -m1 '^config:' "$base" | awk '{print $2}')"
    cfgname="${cfgname:-rl_debug_clip}"
    {
        echo "# Auto-generated by run_sim_clip.sh. Thin overlay: include the debug-clip"
        echo "# config and keyed-merge-override ONLY the clip selection."
        echo "config: ${cfgname}"
        echo "include:"
        echo "  - ${base}"
        echo "control:"
        echo "  - name: low_level"
        echo "    params:"
        [ -n "$case" ]  && echo "      debug_clip_case: \"${case}\""
        [ -n "$clip" ]  && echo "      debug_clip: \"${clip}\""
        [ -n "$start" ] && echo "      debug_clip_start_time: ${start}"
        [ -n "$ego" ]   && echo "      debug_clip_ego_anchor: ${ego}"
    } >"$f"
    printf '%s' "$f"
}

# obk_viz_config <base_abs_config>: write a temp controller config that `include`s <base> and
# turns the viz block ON. That flips three obelisk-launched processes on -- default_robot_viz,
# robot_state_publisher and rviz2 -- with no other change.
#
# ONLY the `on:` scalar is overridden, and that is a constraint rather than a choice: `viz` is
# a MAPPING (deep-merged, so a single key can be set here) but `viz_nodes` is a LIST whose
# entries carry no name/id/key/ros_parameter/topic. Obelisk keys list merges off those fields,
# so it has no identity for a viz_nodes entry and redeclaring one APPENDS a second viz node --
# two robot_state_publishers fighting over /robot_description. Which node runs, which URDF and
# which rviz config therefore all live in common/low_level.yml.
#
# Chains after obk_policy_config: each overlay includes the previous temp file by ABSOLUTE
# path, so `--policy X --rviz` composes.
# obk_scripted_cmd_config <base_brain_abs_config>: write a temp brain config that `include`s
# <base> and RELAYS THE JOYSTICK'S velocity command away from /obelisk/g1/vel_cmd, so a
# scripted profile is the only publisher on it.
#
# Without this the two compete and the controller latches whichever arrived last: measured on
# a real run, 50 Hz of scripted 0.8 against 12 Hz of joystick 0.0 zeroed about a quarter of
# the ticks, which reads as a command dropout and makes any comparison meaningless. The
# joystick's own config comment names this mechanism -- "a stack with a high-level node in the
# loop keyed-merges `- name: joystick_fsm` to relay it somewhere else instead".
#
# The FSM half is untouched: the joystick still arms the robot, which is how a scripted run is
# started and, more importantly, stopped.
obk_scripted_cmd_config() {
    local base="$1"
    local f="/tmp/obk_scripted_$(basename "$base")"
    local cfgname
    cfgname="$(grep -m1 '^config:' "$base" | awk '{print $2}')"
    cfgname="${cfgname:-rl_policy}"
    cat >"$f" <<EOF
# Auto-generated by run_sim_low.sh --command. Thin overlay: include the brain config and point
# the joystick's velocity publisher at a dead topic so the scripted node owns /vel_cmd.
config: ${cfgname}
include:
  - ${base}
control:
  - name: joystick_fsm
    publishers:
      - ros_parameter: pub_ctrl_setting
        topic: /obelisk/g1/vel_cmd_joy
EOF
    printf '%s' "$f"
}

obk_viz_config() {
    local base="$1"
    local f="/tmp/obk_viz_$(basename "$base")"
    local cfgname
    cfgname="$(grep -m1 '^config:' "$base" | awk '{print $2}')"
    cfgname="${cfgname:-rl_policy}"
    cat >"$f" <<EOF
# Auto-generated by run_*.sh --rviz. Thin overlay: include the controller config and turn the
# viz block on. Everything else about viz lives in common/low_level.yml -- see obk_viz_config.
config: ${cfgname}
include:
  - ${base}
viz:
  on: True
EOF
    printf '%s' "$f"
}

# Point the controller at DELAYED depth topics, for testing capture-time pose matching.
#
# MuJoCo casts depth synchronously, so in simulation the pose always matches the image and the
# matching code is exercised but never tested. depth_delay_relay republishes each frame late
# with its header stamp untouched -- the shape of the real ZED fault -- and this overlay is
# what makes the controller actually read the delayed copy. The two belong together: launching
# the relay without repointing the topics measures nothing, quietly.
obk_depth_delay_config() {
    local base="$1" delay="$2"
    local f="/tmp/obk_depthdelay_$(basename "$base")"
    local cfgname
    cfgname="$(grep -m1 '^config:' "$base" | awk '{print $2}')"
    cfgname="${cfgname:-rl_policy}"
    cat >"$f" <<EOF
# Auto-generated by run_sim_low.sh --depth-delay ${delay}. A TEST overlay: it makes the
# controller read frames that depth_delay_relay has held back by ${delay} s.
config: ${cfgname}
control:
  - name: low_level
    params:
      depth_topics: ["/depth/near/image_delayed", "/depth/far/image_delayed"]
include:
  - ${base}
EOF
    printf '%s' "$f"
}

# obk_scene_config <base_world_abs_config> <scene_name>: write a temp world config that
# `include`s <base> and keyed-merge-overrides only the MuJoCo scene, so any
# transfer/sim/robots/g1/<scene>.xml can be launched without a per-scene config file.
obk_scene_config() {
    local base="$1" scene="$2"
    local f="/tmp/obk_world_${scene}.yml"
    local xml="$ROBOT_RL_ROOT/../sim/robots/g1/${scene}.xml"
    # FATAL, not a warning. Warning and then writing the config anyway launches a world with
    # no robot, and a bag of zero armed ticks is indistinguishable from a policy failure until
    # somebody opens it. This is the case it exists for: the five g1_29j_*_scene.xml were
    # renamed to g1_perceptive_* with the training refresh, so every remembered `--scene`
    # argument and every copied command line now names a file that is not there. ab_repeat.sh
    # already did this check on its own; it belongs here, where every caller gets it.
    if [ ! -f "$xml" ]; then
        echo "[obk] no such scene: $xml" >&2
        echo "[obk]   available:" >&2
        ls "$ROBOT_RL_ROOT/../sim/robots/g1/"*_scene.xml 2>/dev/null \
            | xargs -n1 basename | sed 's/\.xml$//;s/^/[obk]     /' >&2
        return 2
    fi
    cat >"$f" <<EOF
# Auto-generated by run_*.sh --scene ${scene}. Keyed-merge override of the scene only.
config: g1_world
include:
  - ${base}
robot:
  - name: g1_sim
    sim:
      - ros_parameter: mujoco_setting
        model_xml_path: \${ROBOT_RL_ROOT}/../sim/robots/g1/${scene}.xml
EOF
    printf '%s' "$f"
}

# ---------------------------------------------------------------------------
# Jetson (the depth camera's GPU host)
# ---------------------------------------------------------------------------

# obk_jetson_zed_cmd: echo a command that ssh's to the Jetson and runs the zed_depth
# container's baked-in run_node.sh, which launches the ZED depth node in its OWN container
# tmux and attaches. The onboard minipc has no GPU and the ZED SDK needs one, so the camera
# lives on the Jetson and reaches us over the shared DDS graph.
#
# All the launch logic is baked into the image at /usr/local/bin/run_node.sh, so there is
# nothing to copy and no nested quoting -- just `docker exec` with plain token args. The
# node runs in a DETACHED tmux, so it survives the ssh dropping; run_node.sh then attaches.
# Re-running this whole command (Up+Enter in the pane) is idempotent: it re-attaches to the
# SAME node rather than starting a second one that would fight over the camera.
#
# `docker start` first is a no-op when already running and covers a manual `compose stop`;
# a container that was never created fails with a clear "No such container", and you bring
# it up once with `docker compose up -d` in transfer/jetson/zed/docker.
# `run_node.sh pair` launches BOTH cameras from the bind-mounted per-camera configs. The
# stock `zed` case starts ONE node from default_params.yaml, which asks for HD720 -- a mode
# no GMSL ZED X can open -- and has no camera selector, so it is not the deployment path.
#
# The ssh target when docker/.env does not set OBK_JETSON_HOST -- which it normally does,
# and compose's value WINS inside the container. So this is the fallback, not the authority;
# tests/test_zed_link_target.py pins the two to the same address so a stale .env cannot
# quietly aim the depth pane at a box that no longer has the cameras.
OBK_JETSON_HOST_FALLBACK="${OBK_JETSON_HOST_FALLBACK:-zolkin@192.168.123.221}"

# The brain's own address on the robot LAN, and therefore the time master (see
# scripts/setup_clock_sync.sh). Only ever read to check a clock, never to move a control
# packet, so it stays a plain default rather than another required .env line.
OBK_TIME_MASTER="${OBK_TIME_MASTER:-192.168.123.220}"

# The ZED box on the robot network. All Thor<->ZED traffic goes over that wired link: the
# box is multi homed (wifi for internet, plus docker and l4t bridges) and Cyclone would
# otherwise pick an interface "arbitrarily" in its own words, so config/cyclonedds.xml pins
# it to this address on the ZED side.
#
# Requires pubkey auth to that host reachable from INSIDE this container --
# scripts/setup_zed_link.sh sets it up and obk_jetson_preflight checks it.
# Overrides: OBK_JETSON_HOST, OBK_JETSON_CONTAINER.
obk_jetson_zed_cmd() {
    local host="${OBK_JETSON_HOST:-$OBK_JETSON_HOST_FALLBACK}"
    local container="${OBK_JETSON_CONTAINER:-zed_depth}"
    local opts="-o StrictHostKeyChecking=accept-new"
    printf 'ssh -t %s %q "docker start %s >/dev/null 2>&1; docker exec -it %s run_node.sh pair"' \
        "$opts" "$host" "$container" "$container"
}

# obk_jetson_preflight: say whether the depth pane can actually reach the ZED box, BEFORE
# any pane is opened. Run it where the pane's ssh runs -- inside the container.
#
# Every failure of this link presents the same way: a pane that sits there, or dies, with no
# depth arriving. On a robot that reads as a CAMERA fault and sends you to the cameras. So
# name the cause here instead. Four of them, with four different fixes:
#
#   1. the box is not answering               -- off, cable out, or OBK_JETSON_HOST is stale
#   2. no pubkey auth from in here            -- scripts/setup_zed_link.sh (the usual one)
#   3. the offered identity has a passphrase  -- ssh blocks on a prompt in a pane, forever
#   4. the zed_depth container does not exist -- the cameras are fine, the image is not up
#
# (3) is why setup_zed_link.sh makes a DEDICATED passphrase-free key instead of reusing
# ~/.ssh/id_ed25519, which on this box has one: BatchMode turns that block into a clean
# "Permission denied" here, but an interactive pane would just hang at a prompt.
#
# Returns 0 only when a `docker exec` on the far side would work.
obk_jetson_preflight() {
    local host="${OBK_JETSON_HOST:-$OBK_JETSON_HOST_FALLBACK}"
    local container="${OBK_JETSON_CONTAINER:-zed_depth}"
    local addr="${host#*@}"
    local out rc=0

    # BatchMode: never prompt. A prompt IS the failure being looked for, not a question.
    out="$(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
              "$host" 'echo obk_ssh_ok' 2>&1)" || rc=$?
    if [ "$rc" -ne 0 ] || [ "${out#*obk_ssh_ok}" = "$out" ]; then
        echo "[obk] the depth pane cannot ssh to the ZED box ($host)." >&2
        echo "[obk]   ssh: ${out##*$'\n'}" >&2
        case "$out" in
            *"Permission denied"*)
                echo "[obk]   CAUSE: no pubkey auth from in here (the box may have no" >&2
                echo "[obk]          authorized_keys at all, or our key is not in it)." >&2
                echo "[obk]   FIX:   scripts/setup_zed_link.sh   -- once, needs that box's password" >&2 ;;
            *"timed out"*|*"No route to host"*|*"Connection refused"*|*"Network is unreachable"*)
                echo "[obk]   CAUSE: $addr is not answering." >&2
                echo "[obk]   FIX:   ping $addr; if that works, OBK_JETSON_HOST in docker/.env" >&2
                echo "[obk]          is aimed at the wrong box (it named the onboard minipc" >&2
                echo "[obk]          for a while after the cameras moved to their own)." >&2 ;;
            *"Host key verification failed"*|*"REMOTE HOST IDENTIFICATION"*)
                echo "[obk]   CAUSE: the box's host key changed -- reinstalled, or a new box." >&2
                echo "[obk]   FIX:   ssh-keygen -R $addr, then scripts/setup_zed_link.sh" >&2 ;;
            *)
                echo "[obk]   FIX:   scripts/setup_zed_link.sh --check   for the full diagnosis" >&2 ;;
        esac
        echo "[obk]   This is NOT a camera fault. --no-depth skips the pane entirely." >&2
        return 1
    fi

    rc=0
    out="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
              "docker inspect -f '{{.State.Status}}' $container" 2>&1)" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[obk] ssh to $host works, but '$container' is not usable there:" >&2
        echo "[obk]   ${out##*$'\n'}" >&2
        echo "[obk]   CAUSE: the container was never created (or this user cannot reach" >&2
        echo "[obk]          docker). The cameras are fine either way." >&2
        echo "[obk]   FIX:   on $addr: cd ~/sprout/transfer/jetson/zed/docker && docker compose up -d" >&2
        return 1
    fi
    echo "[obk] ZED box $addr: ssh ok, container '$container' is ${out##*$'\n'}."
    return 0
}

# obk_jetson_clock_report: print how far the ZED box's clock sits from this one's, reading
# the FAR SIDE's own chrony rather than timing a round trip.
#
# Not `date` over ssh: an ssh round trip is a few hundred ms of process startup and the
# remote reading lands at an unknown point inside it, so that method's own error is ~100x
# the offset it would be measuring. chrony's number is an NTP measurement against this box.
#
# Diagnostic only, and deliberately never fatal: the controller's depth watchdog uses LOCAL
# RECEIPT time (g1_controller.cc), so a skewed camera clock cannot stale-latch a good camera.
# What it does corrupt is `depth_capture_age` -- the only number that says how old a frame
# was when the policy used it -- and every cross-box comparison in the bag.
obk_jetson_clock_report() {
    local host="${OBK_JETSON_HOST:-$OBK_JETSON_HOST_FALLBACK}"
    local out rc=0 ref off
    out="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
              'command -v chronyc >/dev/null && chronyc tracking' 2>&1)" || rc=$?
    if [ "$rc" -ne 0 ] || [ -z "$out" ]; then
        echo "[obk] clock: cannot read chrony on $host -- clocks are UNSYNCHRONISED unless"
        echo "[obk]        something else keeps them together. See scripts/setup_clock_sync.sh."
        return 1
    fi
    ref="$(printf '%s\n' "$out" | sed -n 's/^Reference ID *: *//p')"
    off="$(printf '%s\n' "$out" | sed -n 's/^Last offset *: *\([-+0-9.]*\).*/\1/p')"
    if [ -z "$off" ]; then off=0; fi
    printf '[obk] clock: ZED box is %s ms off its source (%s)\n' \
        "$(awk -v o="$off" 'BEGIN{printf "%+.2f", o*1000}')" "${ref:-unknown}"
    case "$ref" in
        *"$OBK_TIME_MASTER"*) ;;
        *)  echo "[obk]        WARNING: its source is NOT this box ($OBK_TIME_MASTER), so the two"
            echo "[obk]        clocks agree only as far as their separate uplinks do." ;;
    esac
    awk -v o="$off" 'BEGIN{exit !(o*1000 > 5 || o*1000 < -5)}' \
        && echo "[obk]        WARNING: >5 ms is an eighth of the depth pipeline's ~42 ms latency."
    return 0
}

# obk_mark_jetson <0|1>: record whether THIS launch put a node on the Jetson, so stop.sh
# knows whether to reach out to it. The marker lives in this container's /tmp, so it is
# naturally wiped by a down/up and teardown simply skips.
obk_mark_jetson() {
    if [ "${1:-0}" = 1 ]; then : > /tmp/obk_jetson_active
    else rm -f /tmp/obk_jetson_active 2>/dev/null || true
    fi
}

# obk_jetson_stop_cmd: echo a command (run from INSIDE this container, same ssh path as the
# launch) that stops the Jetson-side ZED node, leaving the container itself up.
# Best-effort: BatchMode + a short ConnectTimeout so an unreachable Jetson fails fast, and
# the remote side tolerates a stopped container or no tmux server.
#
# Killing the tmux SESSION is not enough: the `ros2 launch` tree under it survives a pane
# SIGHUP -- the same orphan problem obk_stop reaps locally -- and a surviving launch would
# then fight the next one over the camera. So pkill the launch tree too: SIGTERM first for
# a clean rcl shutdown, then -9 for stragglers. The whole remote script is kept quote-free
# so it survives the bash -> ssh -> jetson-shell -> docker-exec quoting layers, and `set -f`
# stops the `[.]` anti-self-match patterns being pathname-expanded.
#
# The ${OBK_JETSON_HOST:-...} refs are emitted UNEXPANDED on purpose: obk_stop runs this via
# `docker compose exec ... bash -c "$(obk_jetson_stop_cmd)"`, and that $( ) substitution runs
# on the HOST, where the variable is unset (compose sets it only inside the container).
# Expanding here would bake in the wrong default.
obk_jetson_stop_cmd() {
    # `zed_depth.*[.]launch[.]py`, not `zed_depth[.]launch[.]py`: the deployment launch file
    # is zed_depth_PAIR.launch.py and the old pattern does not match it, so teardown would
    # silently leave both cameras held. The session name follows run_node.sh's `obk_<node>`,
    # so the pair is obk_pair; obk_zed is still killed for a single-camera debug run.
    #
    # Killing the LAUNCH process is what stops the pair, and it matters more than it used to:
    # the launch file respawns a died node after 10 s, so pkill'ing the nodes alone would just
    # have them come back. The launch tree goes first, then any stragglers.
    local reap='set -f; tmux kill-session -t obk_pair >/dev/null 2>&1 || true; tmux kill-session -t obk_zed >/dev/null 2>&1 || true; pkill -f zed_depth.*[.]launch[.]py >/dev/null 2>&1 || true; sleep 2; pkill -9 -f zed_depth.*[.]launch[.]py >/dev/null 2>&1 || true; pkill -9 -f [z]ed_depth_node >/dev/null 2>&1 || true'
    printf 'host="${OBK_JETSON_HOST:-zolkin@192.168.123.221}"; container="${OBK_JETSON_CONTAINER:-zed_depth}"; ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$host" "docker exec $container bash -c '\''%s'\''"' \
        "$reap"
}

# ---------------------------------------------------------------------------
# Shutdown (HOST side)
# ---------------------------------------------------------------------------

# obk_stop [--zed] [--stop|--down]: kill the tmux server in the container (SIGHUPs every
# pane, tearing down obk-launch and its nodes), then reap orphans. Then:
#   (no arg) leave the container running.
#   --stop   `docker compose stop`  -- stop but KEEP the container (writable layer intact).
#   --down   `docker compose down`  -- stop and REMOVE it.
#
# The Jetson ZED node runs in its OWN tmux inside the Jetson container -- launched over ssh
# FROM here, and deliberately outliving this tmux -- so killing the local server does NOT
# stop it. By DEFAULT it is left running: the ZED is slow and flaky to bring back up (the
# camera link often needs retries), and the next launch's run_node.sh finds the surviving
# session and re-attaches instead of starting a second one. Pass --zed to tear it down too.
obk_stop() {
    cd "$OBK_DOCKER_DIR" || return 1
    local dc=(docker compose -f "$OBK_COMPOSE_FILE")
    # Accept --zed and one mode in any order.
    local mode="" kill_zed=0 a
    for a in "$@"; do
        case "$a" in
            --zed) kill_zed=1 ;;
            *)     mode="$a" ;;
        esac
    done
    if "${dc[@]}" ps --status running --services 2>/dev/null | grep -qx "$OBK_SERVICE"; then
        if [ "$kill_zed" = 1 ] && "${dc[@]}" exec -T "$OBK_SERVICE" \
                bash -c 'test -f /tmp/obk_jetson_active' 2>/dev/null; then
            echo "[obk] stopping the Jetson ZED node over ssh (best-effort)..."
            "${dc[@]}" exec -T "$OBK_SERVICE" bash -c "$(obk_jetson_stop_cmd)" || true
            "${dc[@]}" exec -T "$OBK_SERVICE" bash -c 'rm -f /tmp/obk_jetson_active 2>/dev/null || true'
        fi
        # SIGINT the recorder FIRST and let it finalise. rosbag2 writes metadata.yaml on a
        # clean shutdown and only then; killed hard, it leaves a directory holding a perfectly
        # good .mcap that `ros2 bag` refuses to open ("read failed"), recoverable only via
        # `ros2 bag reindex`. Every bag this stack recorded before this was in that state.
        # 3 s is generous -- finalising is a summary write, not a flush of the whole file.
        #
        # `[r]os2 bag record`, NOT the plain string: `pgrep -f` matches against every process's
        # full command line INCLUDING THIS `bash -c` ITSELF, which contains the pattern. Spelled
        # plainly, the test was true on every teardown -- so a run with no recorder still printed
        # "stopping the bag recorder", waited its 3 s, and then warned that a bag which never
        # existed would need reindexing. The bracket makes the pattern not match its own text.
        if "${dc[@]}" exec -T "$OBK_SERVICE" bash -c 'pgrep -f "[r]os2 bag record" >/dev/null'; then
            echo "[obk] stopping the bag recorder gracefully (so metadata.yaml is written)..."
            "${dc[@]}" exec -T "$OBK_SERVICE" bash -c '
                pkill -INT -f "[r]os2 bag record" 2>/dev/null || true
                for _ in $(seq 30); do
                    pgrep -f "[r]os2 bag record" >/dev/null || break
                    sleep 0.1
                done
                pgrep -f "[r]os2 bag record" >/dev/null \
                    && echo "[obk] recorder still running after 3s; it will be killed -- the bag will need \`ros2 bag reindex\`" \
                    || true' || true
        fi
        echo "[obk] killing tmux server in container (stops all panes/nodes)..."
        "${dc[@]}" exec -T "$OBK_SERVICE" bash -c 'tmux kill-server 2>/dev/null || true'
        # Reap orphaned launch trees. tmux kill-server only SIGHUPs the panes; a ros2 launch
        # that survives keeps its lifecycle event handlers and change_state clients alive on
        # the graph. Because the configs reuse node names via the shared includes
        # (obelisk_control_low_level, obelisk_estimation_g1_estimator, g1_world, ...), ONE
        # stale launch poisons the NEXT session: it reacts to the new nodes' transition
        # events and re-emits configure/activate at them, and an invalid transition on an
        # active rclpy node raises and kills it. [_]/[.] break pkill self-matching.
        echo "[obk] reaping orphaned ros2 launch trees + obelisk nodes..."
        "${dc[@]}" exec -T "$OBK_SERVICE" bash -c '
            pkill -f "obelisk_bringup[.]launch[.]py" 2>/dev/null || true
            pkill -f "ros2 la[u]nch" 2>/dev/null || true
            sleep 2
            pkill -9 -f "obelisk_bringup[.]launch[.]py" 2>/dev/null || true
            pkill -9 -f "ros2 la[u]nch" 2>/dev/null || true
            pkill -f "obelisk[_]unitree|lib/joy/joy[_]node|global[_]state|fsm[_]state[_]echo" 2>/dev/null || true
            sleep 1
            left=$(pgrep -af "ros2 la[u]nch|obelisk_bringup[.]launch|obelisk[_]unitree|global[_]state" 2>/dev/null | grep -cv pgrep || true)
            if [ "${left:-0}" -gt 0 ]; then
                echo "[obk] WARNING: $left launch/node process(es) still alive after reap:"
                pgrep -af "ros2 la[u]nch|obelisk_bringup[.]launch|obelisk[_]unitree|global[_]state" | grep -v pgrep || true
            fi' || true
    else
        echo "[obk] container '$OBK_SERVICE' is not running."
    fi
    case "$mode" in
        --down)
            echo "[obk] docker compose down (stop + REMOVE container)..."
            "${dc[@]}" down ;;
        --stop)
            echo "[obk] docker compose stop (container kept)..."
            "${dc[@]}" stop ;;
        "")
            echo "[obk] nodes stopped; container left running (--stop to stop it, --down to remove it)." ;;
        *)
            echo "[obk] unknown option '$mode' (use --stop or --down)."; return 2 ;;
    esac
}
