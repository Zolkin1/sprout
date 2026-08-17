#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Preflight for running the WORLD and the BRAIN on two different machines.
#
# The split needs no new launch path -- run_sim_low.sh already launches the brain alone and
# run_sim_world.sh the world alone -- so when it does not work, it fails as silence: the two
# halves come up, log nothing unusual, and never see each other. This turns that silence into
# one of four named causes:
#
#   1. the two boxes are on different ROS_DOMAIN_ID / RMW_IMPLEMENTATION;
#   2. discovery cannot cross the link (Cyclone's default is UDP MULTICAST, which campus and
#      guest wifi routinely drop between clients -- see CYCLONEDDS_URI below);
#   3. the two images were built from different OBELISK_REF, so the message TYPE HASHES differ
#      and endpoints never pair -- visible here as a topic with a publisher and a subscriber
#      that both report zero matched counterparts;
#   4. everything pairs but a rate is wrong, which is the only one of the four that is about
#      the network being slow rather than the graph being wrong.
#
# Run it on BOTH boxes while both halves are up. It only reads the graph -- no node is
# launched, nothing is published, and it is safe to run against a live robot.
#
#   scripts/net_check.sh                 # 3 s per topic
#   scripts/net_check.sh --seconds 10    # longer, for a jitter question
#   scripts/net_check.sh --quick         # graph only, no rate measurement
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"

SECONDS_PER_TOPIC=3
QUICK=""
while [ $# -gt 0 ]; do
    case "$1" in
        --seconds) SECONDS_PER_TOPIC="${2:-3}"; shift 2 ;;
        --quick)   QUICK=1; shift ;;
        -h|--help)
            sed -n '3,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "net_check.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

# Host side: run inside the container, where the ROS environment actually lives. `exec -T`
# rather than obk_host_launch: this is a diagnostic that prints to stdout, not a tmux session.
#
# Deliberately NO `docker compose up -d`. Every other script here starts the container, but
# this one is the thing you run WHILE the stack is up and misbehaving -- and `up -d` recreates
# a container whose compose config has changed since it was started, which would kill the very
# run being diagnosed. Refuse instead, and say so.
if [ -z "${OBK_IN_CONTAINER:-}" ]; then
    cd "$OBK_DOCKER_DIR" || exit 1
    if ! docker compose -f "$OBK_COMPOSE_FILE" ps --status running --services 2>/dev/null \
         | grep -qx "$OBK_SERVICE"; then
        echo "net_check.sh: the '$OBK_SERVICE' container is not running." >&2
        echo "  Start the stack first (scripts/run_sim_low.sh / run_sim_world.sh), or just" >&2
        echo "  bring the container up:  cd docker && docker compose up -d" >&2
        exit 2
    fi
    args=(--seconds "$SECONDS_PER_TOPIC")
    [ -n "$QUICK" ] && args+=(--quick)
    exec docker compose -f "$OBK_COMPOSE_FILE" exec -T -e OBK_IN_CONTAINER=1 "$OBK_SERVICE" \
        bash -c 'source ~/.bash_aliases 2>/dev/null || true
                 obk >/dev/null
                 source "$ROBOT_RL_ROOT/install/setup.bash" 2>/dev/null || true
                 exec "$ROBOT_RL_ROOT/scripts/net_check.sh" "$@"' _ "${args[@]}"
fi

# --------------------------------------------------------------------------------------
# Topics that cross the WORLD <-> BRAIN boundary, with the rate each is configured for.
#
# From g1_control/config/g1_world.yml (the publish side) and common/low_level.yml (the
# subscribe side). NOT every topic on the graph: /obelisk/g1/est_state and the debug topics
# are brain-internal by construction (the estimator lives in the brain config), and a split
# run must not put them on the wire.
# --------------------------------------------------------------------------------------
# `event` instead of a rate marks a topic that only publishes when something happens -- the
# joystick ones. Silence there is the CORRECT state with nobody holding the stick, and
# flagging it trains you to ignore this script's output, which is the one thing it must not do.
#
#            topic                          expected_hz  direction
CROSSING=(
    "/obelisk/g1/joint_encoders             500    world->brain"
    "/obelisk/g1/pelvis_imu                 1000   world->brain"
    "/depth/near/image_processed            50     world->brain"
    "/depth/far/image_processed             50     world->brain"
    "/Odometry                              10     world->brain"
    "/obelisk/g1/fsm_state                  5      world->brain"
    "/obelisk/g1/ctrl                       50     brain->world"
    "/obelisk/g1/vel_cmd                    event  brain->world"
    "/obelisk/g1/exec_fsm                   event  brain->world"
)

# Colour only on a terminal: this output is routinely piped into a file or a paste, and
# escape codes there make a PASS/FAIL list harder to read rather than easier.
if [ -t 1 ]; then
    GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YEL=$'\033[0;33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
    GREEN=""; RED=""; YEL=""; DIM=""; OFF=""
fi
fails=0
warns=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '  %sPASS%s  %s\n' "$GREEN" "$OFF" "$*"; }
bad()  { printf '  %sFAIL%s  %s\n' "$RED" "$OFF" "$*"; fails=$((fails + 1)); }
warn() { printf '  %sWARN%s  %s\n' "$YEL" "$OFF" "$*"; warns=$((warns + 1)); }
note() { printf '        %s%s%s\n' "$DIM" "$*" "$OFF"; }

# --------------------------------------------------------------------------------------
say ""
say "=== 1. DDS environment (must be IDENTICAL on both boxes) ==="
say "  hostname             $(hostname)"
say "  ROS_DOMAIN_ID        ${ROS_DOMAIN_ID:-<unset, defaults to 0>}"
say "  RMW_IMPLEMENTATION   ${RMW_IMPLEMENTATION:-<unset, defaults to fastrtps>}"
say "  CYCLONEDDS_URI       ${CYCLONEDDS_URI:-<unset -- multicast discovery>}"
say "  ROS_DISTRO           ${ROS_DISTRO:-<unset>}"
if [ -z "${ROS_DOMAIN_ID:-}" ] || [ -z "${RMW_IMPLEMENTATION:-}" ]; then
    bad "the obelisk environment is not applied -- run \`obk\` first"
    note "every pane the run scripts open does this for you; a bare shell does not"
fi

# --------------------------------------------------------------------------------------
say ""
say "=== 2. The link ==="
# Interfaces with an address, and whether the kernel will even carry multicast on them.
# `network_mode: host`, so these are the HOST's interfaces -- what the peer actually has to
# be reachable over.

# Emit "<ifname> <ipv4> <FLAGS>" per interface. `ip` is not in this image (net-tools is), so
# ifconfig is the fallback rather than the exception -- and the fallback has to keep working
# after `iproute2` lands in the Dockerfile, since a running container predates the rebuild.
interfaces() {
    if command -v ip >/dev/null 2>&1; then
        ip -o -4 addr show 2>/dev/null | awk '{print $2, $4}' | while read -r n a; do
            printf '%s %s %s\n' "$n" "$a" \
                "$(ip link show "$n" 2>/dev/null | sed -n '1s/.*<\([^>]*\)>.*/\1/p')"
        done
    elif command -v ifconfig >/dev/null 2>&1; then
        ifconfig 2>/dev/null | awk '
            /^[a-zA-Z0-9_.-]+:/ {
                name = $1; sub(":$", "", name)
                flags = $2; sub(/^flags=[0-9]*</, "", flags); sub(/>$/, "", flags)
                addr = ""
            }
            /^[[:space:]]*inet / { addr = $2; if (name != "") print name, addr, flags; name = "" }'
    fi
}

routable=0
while read -r ifname addr flags; do
    [ "$ifname" = lo ] && continue
    [ -z "$addr" ] && continue
    kind="wired"
    case "$ifname" in wl*|wlan*) kind="WIFI" ;; docker*|br-*|veth*|l4tbr*) kind="local bridge" ;; esac
    if [[ "$flags" == *MULTICAST* ]]; then
        say "  $(printf '%-12s %-20s %-13s multicast-capable' "$ifname" "$addr" "$kind")"
    else
        warn "$ifname $addr has NO MULTICAST flag -- default discovery cannot use it"
    fi
    case "$kind" in "local bridge") ;; *) routable=$((routable + 1)) ;; esac
    [ "$kind" = WIFI ] && note "wifi: if the other box is on the same AP, client isolation" \
        "is the single most likely reason discovery does not cross"
done < <(interfaces)
if [ "$routable" -gt 1 ]; then
    note "more than one routable interface is up -- Cyclone picks ONE for discovery and it"
    note "need not be the one that reaches the other box (<NetworkInterface> settles it)"
fi

if [ -n "${CYCLONEDDS_URI:-}" ]; then
    ok "CYCLONEDDS_URI is set -- discovery is not relying on multicast"
    note "peers: $(sed -n 's/.*<Peer[^>]*[Aa]ddress="\([^"]*\)".*/\1/p' \
                 "${CYCLONEDDS_URI#file://}" 2>/dev/null | tr '\n' ' ')"
else
    warn "no CYCLONEDDS_URI -- discovery is UDP multicast on the default interface"
    note "fine on a wired switch; wifi with client isolation silently drops it, and a"
    note "multi-homed box may also pick the wrong interface. See docker/cyclonedds.xml."
fi

# --------------------------------------------------------------------------------------
say ""
say "=== 3. Who is on the graph ==="
nodes="$(ros2 node list 2>/dev/null)"
if [ -z "$nodes" ]; then
    bad "ros2 node list is EMPTY -- not even the local half is visible"
    note "nothing is running here, or the domain/RMW above is not what the nodes use"
else
    printf '%s\n' "$nodes" | sed 's/^/  /'
fi
have_world=$(printf '%s\n' "$nodes" | grep -cE 'g1_sim|obelisk_robot' || true)
have_brain=$(printf '%s\n' "$nodes" | grep -cE 'obelisk_control_low_level|g1_estimator' || true)
say ""
say "  world half visible: $([ "$have_world" -gt 0 ] && echo yes || echo NO)"
say "  brain half visible: $([ "$have_brain" -gt 0 ] && echo yes || echo NO)"
if [ "$have_world" -gt 0 ] && [ "$have_brain" -gt 0 ]; then
    ok "both halves are on one graph"
elif [ "$have_world" -eq 0 ] && [ "$have_brain" -eq 0 ]; then
    bad "neither half is visible -- start one before running this"
else
    bad "only one half is visible: discovery is not crossing the link"
    note "cause 1 (domain/RMW) or cause 2 (multicast) above. Check section 1 on BOTH boxes,"
    note "then set CYCLONEDDS_URI on both and retry."
fi

# --------------------------------------------------------------------------------------
say ""
say "=== 4. Crossing topics ==="
say "  Endpoint counts are the type-hash check: a topic with both a publisher and a"
say "  subscriber that are NOT paired means the two images disagree about the message."
say ""
for row in "${CROSSING[@]}"; do
    read -r topic want dir <<<"$row"
    info="$(ros2 topic info "$topic" 2>/dev/null)"
    if [ -z "$info" ]; then
        bad "$(printf '%-32s %-12s not on the graph' "$topic" "$dir")"
        continue
    fi
    pubs="$(printf '%s\n' "$info" | sed -n 's/.*Publisher count: *//p')"
    subs="$(printf '%s\n' "$info" | sed -n 's/.*Subscription count: *//p')"
    pubs="${pubs:-0}"; subs="${subs:-0}"

    if [ "$pubs" -eq 0 ]; then
        bad "$(printf '%-32s %-12s no publisher (pub=%s sub=%s)' "$topic" "$dir" "$pubs" "$subs")"
        continue
    fi
    if [ -n "$QUICK" ]; then
        ok "$(printf '%-32s %-12s pub=%s sub=%s' "$topic" "$dir" "$pubs" "$subs")"
        continue
    fi

    # Measured rate. `ros2 topic hz` prints running averages; take the last one it emits.
    hz="$(timeout "$((SECONDS_PER_TOPIC + 3))" ros2 topic hz "$topic" 2>/dev/null \
          | sed -n 's/^average rate: *\([0-9.]*\).*/\1/p' | tail -1)"
    if [ "$want" = event ]; then
        if [ -z "$hz" ]; then
            ok "$(printf '%-32s %-12s %8s     (event-driven)  pub=%s sub=%s' \
                    "$topic" "$dir" "idle" "$pubs" "$subs")"
        else
            ok "$(printf '%-32s %-12s %8.1f Hz  (event-driven)  pub=%s sub=%s' \
                    "$topic" "$dir" "$hz" "$pubs" "$subs")"
        fi
        continue
    fi
    if [ -z "$hz" ]; then
        bad "$(printf '%-32s %-12s pub=%s sub=%s but NOTHING ARRIVED in %ss' \
                "$topic" "$dir" "$pubs" "$subs" "$SECONDS_PER_TOPIC")"
        note "a publisher that is advertised but never received is the type-hash symptom"
        continue
    fi
    # 25% either side. Wide on purpose: these are wall-clock averages over a few seconds and
    # a real problem is a factor, not a percent.
    lo=$(awk -v w="$want" 'BEGIN{printf "%.2f", 0.75*w}')
    hi=$(awk -v w="$want" 'BEGIN{printf "%.2f", 1.25*w}')
    line="$(printf '%-32s %-12s %8.1f Hz (want %s)  pub=%s sub=%s' \
            "$topic" "$dir" "$hz" "$want" "$pubs" "$subs")"
    if awk -v h="$hz" -v l="$lo" -v u="$hi" 'BEGIN{exit !(h>=l && h<=u)}'; then
        ok "$line"
    else
        warn "$line"
    fi
done

# --------------------------------------------------------------------------------------
say ""
if [ "$fails" -eq 0 ] && [ "$warns" -eq 0 ]; then
    say "${GREEN}net_check: PASS${OFF} -- the split graph is healthy from this box."
elif [ "$fails" -eq 0 ]; then
    say "${YEL}net_check: $warns warning(s)${OFF}, no failures."
    say "A low rate on /obelisk/g1/pelvis_imu or /obelisk/g1/joint_encoders is the link, not"
    say "the sim: they are the 1 kHz and 500 Hz streams and the only ones that stress it."
    say "The fix is to lower those sensors' dt for split runs, NOT to move the estimator to"
    say "the world side -- the estimator lives in the brain on hardware and must stay there."
else
    say "${RED}net_check: $fails failure(s), $warns warning(s).${OFF}"
    say "Work down from section 1: an environment mismatch makes every later section wrong."
fi
say ""
exit $((fails > 0))
