#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Run the same configuration N times, unattended, and print the bags it produced.
#
# WHY THIS EXISTS: outcomes on these stairs are not repeatable run to run -- four identical
# local runs measured 3 climbs and 1 fall. A single run therefore cannot compare two
# configurations, and this makes N runs cheap enough to compare distributions.
#
# WHY RUNS DIFFER AT ALL, since the networks here are deterministic (the FM sampler is pinned
# and zero_noise is on): the stack is ASYNCHRONOUS. The camera timers and the control loop are
# independent, drift about 1 ms/s against each other, and the phase wraps every ~20 s -- so a
# 30 s run already sweeps the whole period, and what changes between runs is where in that
# cycle the robot happens to be when it reaches a stair edge. Nothing needs to be injected to
# get that spread; `--stagger` exists only to shift the world's start within a tick if you ever
# want to bias it deliberately, and is OFF by default because the asynchrony samples it.
#
#   scripts/ab_repeat.sh --runs 5 --brain thor  --tag thor_tf32on
#   scripts/ab_repeat.sh --runs 5 --brain local --tag local_tf32on --scene tile1_scene
#
# TODO(ab-stats): the comparison this was built for is DEFERRED, not done. The open question
# is whether the Thor is actually worse than the desktop, or whether both fail some fraction
# of runs and the single-run difference we saw (Thor tilt p99 14.8 deg vs local 8.5) was
# scatter. Ten runs settles it: 5 local + 5 Thor, TF32 MATCHED on both boxes so it is not a
# second variable, lead 0.04, one constant command, one tile. Score with
# plot_utils/score_runs.py, which reports fall rate per group rather than per run.
#
# Deferred because the stack became good enough to drive by hand and the question stopped
# blocking anything -- not because it was answered. Two things already known that bear on it:
# a chronically LATE plan is fatal (65 clamps of 65 plans fell, while 1 of 76 is harmless),
# and the teacher randomizes no latency at all, so the policy has never been trained to
# tolerate a lagged loop.
#
# It does NOT touch the configuration. Set fm_lead_sec / NVIDIA_TF32_OVERRIDE / the policy
# folder yourself first, then name the result with --tag; the tag is only for your notes,
# since bags are named by timestamp.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBK_DIR="$(dirname "$HERE")"

RUNS=5; BRAIN=local; TAG=run; SCENE=""; RUN_TIME=30; SETTLE=40; STAGGER=0
THOR_HOST="${OBK_THOR_HOST:-zolkin@192.168.50.20}"
THOR_DIR="${OBK_THOR_DIR:-/home/zolkin/sprout/transfer/obelisk}"
CMD_PROFILE="${OBK_AB_COMMAND:-smooth_ramp}"

while [ $# -gt 0 ]; do
    case "$1" in
        --runs)     RUNS="$2"; shift 2 ;;
        --brain)    BRAIN="$2"; shift 2 ;;   # local | thor
        --tag)      TAG="$2"; shift 2 ;;
        --scene)    SCENE="$2"; shift 2 ;;
        --run-time) RUN_TIME="$2"; shift 2 ;;
        --command)  CMD_PROFILE="$2"; shift 2 ;;
        --stagger) STAGGER=1; shift ;;
        -h|--help)  sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ab_repeat.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done
[ "$BRAIN" = local ] || [ "$BRAIN" = thor ] || { echo "--brain must be local or thor" >&2; exit 2; }

thor() { ssh -o BatchMode=yes -o ConnectTimeout=6 "$THOR_HOST" "$@"; }

stop_all() {
    (cd "$OBK_DIR" && timeout 90 scripts/stop.sh >/dev/null 2>&1) || true
    [ "$BRAIN" = thor ] && timeout 90 thor "cd $THOR_DIR && scripts/stop.sh" >/dev/null 2>&1 || true
    sleep 3
}

# The brain's bags land on whichever box RECORDS them, which is the box the brain runs on.
bag_dir()  { if [ "$BRAIN" = thor ]; then thor "ls -1dt $THOR_DIR/log/bags/*/ 2>/dev/null | head -1"; \
                                     else ls -1dt "$OBK_DIR"/log/bags/*/ 2>/dev/null | head -1; fi | tr -d '\r'; }

# Readiness is POLLED, not slept on: the ONNX pair takes ~10 s to load on one box and longer
# on another, and starting the world early hands the policy an already-collapsed robot (there
# is no user_pose holding it up).
wait_ready() {
    local i
    for i in $(seq 1 "$SETTLE"); do
        local pane
        if [ "$BRAIN" = thor ]; then
            pane=$(thor "docker exec docker-sim-1 bash -c \"tmux capture-pane -p -S -200 -t simlow.0 2>/dev/null\"" 2>/dev/null)
        else
            pane=$(cd "$OBK_DIR" && docker exec docker-sim-1 bash -c "tmux capture-pane -p -S -200 -t simlow.0 2>/dev/null" 2>/dev/null)
        fi
        case "$pane" in *"policy ready"*) return 0 ;; esac
        sleep 1
    done
    return 1
}

echo "[ab] tag=$TAG brain=$BRAIN runs=$RUNS command=$CMD_PROFILE scene=${SCENE:-default} stagger=$STAGGER"
RESULTS=()
for n in $(seq 1 "$RUNS"); do
    stop_all
    BEFORE="$(bag_dir)"
    # Brain first: the world has nothing holding the robot up, so a world-first start begins
    # from a pose no training episode ever saw.
    if [ "$BRAIN" = thor ]; then
        timeout 60 ssh -tt -o BatchMode=yes "$THOR_HOST" \
            "cd $THOR_DIR && OBK_RESTART=1 scripts/run_sim_low.sh --command $CMD_PROFILE" >/dev/null 2>&1 &
    else
        (cd "$OBK_DIR" && timeout 60 script -qec "OBK_RESTART=1 scripts/run_sim_low.sh --command $CMD_PROFILE" /dev/null) >/dev/null 2>&1 &
    fi
    if ! wait_ready; then echo "[ab] run $n: brain never reported 'policy ready' -- skipping"; continue; fi

    # Sample a different tick phase each run. dt is 20 ms, so a fifth of it per step walks the
    # whole period over five runs without ever landing on the same offset twice.
    if [ "$STAGGER" -eq 1 ]; then
        local_off=$(awk -v n="$n" 'BEGIN{printf "%.3f", ((n-1)%5)*0.004}')
        sleep "$local_off"
    fi

    WORLD_ARGS=""; [ -n "$SCENE" ] && WORLD_ARGS="--scene $SCENE"
    (cd "$OBK_DIR" && timeout 60 script -qec "OBK_RESTART=1 scripts/run_sim_world.sh $WORLD_ARGS" /dev/null) >/dev/null 2>&1 &
    sleep "$RUN_TIME"
    stop_all
    AFTER="$(bag_dir)"
    if [ "$AFTER" != "$BEFORE" ] && [ -n "$AFTER" ]; then
        echo "[ab] run $n/$RUNS -> $(basename "$AFTER")"
        RESULTS+=("$(basename "$AFTER")")
    else
        echo "[ab] run $n/$RUNS -> NO NEW BAG (the brain may not have recorded)"
    fi
done
echo
echo "[ab] $TAG produced ${#RESULTS[@]} bag(s):"
for b in "${RESULTS[@]:-}"; do echo "  $b"; done
[ "$BRAIN" = thor ] && echo "[ab] they are on the Thor -- pull them with scripts/fetch_brain_bags.sh"
exit 0
