#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Queue the four fine-tunes, their evals, and the PARENT's evals, on whatever GPUs are ours.
#
#   scripts/flow_matching/run_finetune_queue.sh <parent_fm.pt> <tracker_run_name>
#
# TWELVE jobs: 4 fine-tunes + 4 fine-tune evals + 4 parent evals. The parent evals are the
# control arm -- without them a fine-tune's number is a level with nothing to be a level
# ABOVE, and the four courses are different enough that one shared baseline would not do.
#
# **EVERY FINE-TUNE STARTS FROM THE SAME PARENT.** `run_finetune_loop.sh` chains its own
# ITERATIONS (iteration k trains from k-1, which is the AWR loop), but the four CONFIGS must
# not chain with each other: each is a separate answer to "what does this course do to the
# parent", and a chain would make three of them answers about a model that had already been
# fine-tuned on something else. That is why PARENT is passed per run and never re-read from a
# previous config's output.
#
# **The tracker is pinned to the one the FM was RECORDED from**, not to whatever is newest.
# The generator was fitted on that tracker's rollouts, so grading it against a different one
# measures the pair rather than the generator -- and on this box the same tracker run is often
# still training, so "newest" is a moving target.
#
# ---------------------------------------------------------------------------- GPU claiming
# **A GPU is claimed only when NOBODY ELSE is on it.** talos is shared (meg / yao / xiang all
# run here), so the pool is split: OURS is claimed as soon as it is idle, and OPPORTUNISTIC is
# borrowed only while it is completely empty and released back the moment a job ends. Nothing
# is ever killed and nothing co-locates: two Isaac jobs on one H100 fit in memory and then
# halve each other's throughput, which is the failure that looks like a slow GPU rather than a
# scheduling mistake.
#
# The check is by PID OWNER rather than by memory: a GPU sitting at 400 MiB is not free, it is
# somebody's process between allocations.
set -uo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"

PARENT="${1:?usage: run_finetune_queue.sh <parent_fm.pt> <tracker_run_name>}"
TRK="${2:?usage: run_finetune_queue.sh <parent_fm.pt> <tracker_run_name>}"

#: GPUs that are ours to use as soon as they are idle.
OURS="${OURS:-1 2 3}"
#: GPUs we may borrow while they are completely empty. Not guaranteed: another user can take
#: one at any time, so a job is only ever STARTED on one -- never migrated onto one.
OPPORTUNISTIC="${OPPORTUNISTIC:-0 6}"
#: Configs to run, in priority order. The fine-tunes are the long pole (~3.5 h each against
#: well under an hour for an eval), so they are queued first and the evals fill the gaps.
CONFIGS="${CONFIGS:-boxes updown indist multi}"
#: Speeds the quick evals sweep, one run each. Extended down to 0.40 to match the training
#: band's new floor -- widening the band and then evaluating only from 0.83 up would leave the
#: part that changed unmeasured. Still does NOT reach the band's 2.5 ceiling; that gap predates
#: this queue and costs one more run per eval to close.
SPEEDS="${SPEEDS:-0.40 0.83 1.17 1.50}"
ROOT="${ROOT:-/mnt/zolkin_new/sprout/ftqueue}"
POLL="${POLL:-60}"
PY="${PY:-$W/env_isaaclab/bin/python}"
export PYTHONPATH="$W/source/robot_rl:$W/source/motion_gen${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$ROOT"/logs
say() { echo "[queue $(date +%H:%M:%S)] $*" | tee -a "$ROOT/queue.log"; }

# --------------------------------------------------------------------------------- helpers
me="$(id -un)"

gpu_is_free() {
  # True when no compute process on GPU $1 belongs to anyone -- including us. Our own finished
  # jobs release it; our RUNNING ones must not be doubled up on.
  local want="$1" uuid idx pid owner
  uuid=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader \
         | awk -F', *' -v i="$want" '$1==i{print $2}')
  [ -n "$uuid" ] || return 1
  while IFS=', ' read -r u pid _; do
    [ "$u" = "$uuid" ] || continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$owner" ] && return 1
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
  return 0
}

declare -A BUSY        # gpu -> pid of the job we put there

claim_gpu() {
  # Echo a free GPU, ours first. Opportunistic ones are only taken when truly empty.
  local g
  for g in $OURS $OPPORTUNISTIC; do
    [ -n "${BUSY[$g]:-}" ] && continue
    if gpu_is_free "$g"; then echo "$g"; return 0; fi
  done
  return 1
}

cfg_env() {  # config -> the loop's env vars, straight from the tracked YAML
  "$PY" scripts/flow_matching/finetune_config.py "configs/finetune/$1.yaml"
}

cfg_get() { cfg_env "$1" | awk -F= -v k="$2" '$1==k{print substr($0, index($0,"=")+1)}'; }

ft_root()   { echo "$ROOT/ft_$1"; }
ft_result() { echo "$(ft_root "$1")/$(printf 'iter_%03d' "$(cfg_get "$1" ITERS)")/ft/best.pt"; }

# --------------------------------------------------------------------------------- the jobs
launch() {   # kind config gpu
  local kind="$1" cfg="$2" gpu="$3" log="$ROOT/logs/${1}_${2}.log"
  local clip goals task nenv
  clip=$(cfg_get "$cfg" CLIP); goals=$(cfg_get "$cfg" GOALS)
  task=$(cfg_get "$cfg" TASK); nenv=$(cfg_get "$cfg" NENV)
  case "$kind" in
    ft)
      # Every knob the run needs comes from the config; only the PARENT, the tracker and the
      # GPU are the queue's to say.
      ( set -a; eval "$(cfg_env "$cfg")"; set +a
        PARENT="$PARENT" TRK="$TRK" GPU="$gpu" ROOT="$(ft_root "$cfg")" \
        bash scripts/flow_matching/run_finetune_loop.sh ) > "$log" 2>&1 &
      ;;
    parent_eval|ft_eval)
      local fm tag
      if [ "$kind" = parent_eval ]; then fm="$PARENT"; tag="parent_on_$cfg"
      else fm="$(ft_result "$cfg")"; tag="ft_${cfg}_on_$cfg"; fi
      # The HOLDOUT library where one exists: unseen geometry from the same distribution.
      # `indist` has none by construction -- its 60 tiles are 60 of the 75 real ones, so a
      # disjoint holdout cannot be cut -- and it is evaluated on its training tiles, which is
      # what an in-distribution CONTROL is for anyway.
      local evalclip="${clip/_train.h5/_holdout.h5}"
      [ -f "$evalclip" ] || evalclip="$clip"
      ( GPU="$gpu" TRK="$TRK" TASK="$task" CLIP="$evalclip" GR="$goals" \
        SPEEDS="$SPEEDS" OUT="$ROOT/eval_$tag" \
        bash scripts/rsl_rl/run_quick_eval.sh "$fm" "$tag" ) > "$log" 2>&1 &
      ;;
  esac
  BUSY[$gpu]=$!
  say "  -> $kind:$cfg on GPU $gpu (pid ${BUSY[$gpu]}), log $log"
}

ready() {    # kind config -> is it runnable yet
  case "$1" in
    ft|parent_eval) [ -f "$PARENT" ] ;;
    ft_eval)        [ -f "$(ft_result "$2")" ] ;;
  esac
}

# --------------------------------------------------------------------------------- schedule
PENDING=()
for c in $CONFIGS; do PENDING+=("ft:$c"); done
for c in $CONFIGS; do PENDING+=("parent_eval:$c"); done
for c in $CONFIGS; do PENDING+=("ft_eval:$c"); done

say "queue starting: ${#PENDING[@]} jobs"
say "  parent  $PARENT"
say "  tracker $TRK"
say "  gpus    ours=[$OURS] opportunistic=[$OPPORTUNISTIC]"
say "  root    $ROOT"
[ -f "$PARENT" ] || say "  waiting for the parent checkpoint to appear..."

while [ ${#PENDING[@]} -gt 0 ]; do
  # Reap finished jobs and free their GPUs.
  for g in "${!BUSY[@]}"; do
    if ! kill -0 "${BUSY[$g]}" 2>/dev/null; then
      wait "${BUSY[$g]}" 2>/dev/null; rc=$?
      say "  GPU $g free (pid ${BUSY[$g]} exit $rc)"
      unset "BUSY[$g]"
    fi
  done
  # Start whatever is ready, in priority order.
  progressed=0
  for i in "${!PENDING[@]}"; do
    kind="${PENDING[$i]%%:*}"; cfg="${PENDING[$i]##*:}"
    ready "$kind" "$cfg" || continue
    gpu=$(claim_gpu) || break
    launch "$kind" "$cfg" "$gpu"
    unset "PENDING[$i]"; PENDING=("${PENDING[@]}")
    progressed=1
    break
  done
  [ "$progressed" = 1 ] && continue
  sleep "$POLL"
done

say "all jobs dispatched; waiting for the last ${#BUSY[@]} to finish"
for g in "${!BUSY[@]}"; do wait "${BUSY[$g]}" 2>/dev/null; done
say "QUEUE COMPLETE"
