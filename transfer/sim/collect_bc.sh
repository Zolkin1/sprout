#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Filtered-BC data collection: run the K>1 selector over a tile x speed grid and record the
# CHOSEN chunk at every re-plan, plus (with CANDIDATES=1) every rejected one.
#
# Parameterised so the SAME code path runs on a training server and on a laptop GPU -- the
# presets differ only in the grid and the shard count, never in what is recorded, so a local
# file and a talos file are the same format and can be merged together.
#
#   PRESET=talos  21 tiles x 3 speeds x 40 episodes   (the full grid)
#   PRESET=local   2 tiles x 1 speed  x 10 episodes   (a smoke run, minutes not hours)
#
# SHARDS is the parallelism. `live_select` is mostly CPU (MuJoCo physics + the depth raycast)
# with the GPU used only for the FM re-plan, and measures ~1 core and ~1 GB of VRAM per shard,
# so on a many-core box the shard count -- not the GPU -- is what sets throughput. Measured
# baseline to beat: 6 shards sustained 20.4 cells/h on talos. RAISE THIS AND MEASURE rather
# than assuming; and on a shared machine, leave headroom for other users.
#
# Usage:
#   PRESET=talos SHARDS=16 GPU=1 ./collect_bc.sh 0     # shard 0 of 16
#   PRESET=local ./collect_bc.sh 0                     # single shard, local GPU
set -u
SHARD=${1:-0}
PRESET=${PRESET:-local}
SHARDS=${SHARDS:-1}
GPU=${GPU:-0}
K=${K:-4}
CANDIDATES=${CANDIDATES:-1}

if [ "$PRESET" = "talos" ]; then
  ROOT=/mnt/zolkin_new/sprout
  REPO=$ROOT/sprout
  WT=$ROOT/bestofk
  PY=$ROOT/env_sweep/bin/python
  TILES=("6:" "7:" "8:" "9:" "10:" "11:"
         "20:" "21:" "22:" "23:" "24:"
         "40:" "41:" "42:" "43:" "44:"
         "75:box" "76:box" "77:box" "81:box" "82:box")
  SPEEDS=(0.8 1.6 2.4)
  EPISODES=${EPISODES:-40}
else
  WT=/home/zolkin/AmberLab/Project-Isaac-RL/sprout/.claude/worktrees/bestofk
  REPO=$WT
  PY=/home/zolkin/AmberLab/Project-Isaac-RL/sprout/env_isaaclab/bin/python
  TILES=("20:" "40:")
  SPEEDS=(2.4)
  EPISODES=${EPISODES:-10}
fi
OUT=${OUT:-$REPO/logs/value/bc_data_v2}

export PYTHONPATH=$WT/source/robot_rl:$WT/source/motion_gen
export CUDA_VISIBLE_DEVICES=$GPU
cd "$WT/transfer/sim" || exit 1

CRITIC=$REPO/logs/value/v2/V_priv.fold0.pt
TRACKER=$REPO/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/tracker.onnx
FM=$REPO/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx
CLIPS=$REPO/clips/g1_mb_v5_body_mode11_decor.h5

echo "[collect $SHARD/$SHARDS] preset=$PRESET K=$K candidates=$CANDIDATES gpu=$GPU -> $OUT"
i=0
for entry in "${TILES[@]}"; do
  tile="${entry%%:*}"; klass="${entry##*:}"
  for sp in "${SPEEDS[@]}"; do
    if [ $((i % SHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    cell=$OUT/t${tile}_s${sp}
    # Idempotent: a shard that dies mid-grid is restarted with the same command line.
    if [ -f "$cell/selected.h5" ]; then
      echo "[collect $SHARD] SKIP (done): t$tile s$sp"; continue
    fi
    mkdir -p "$cell"
    args=(--out "$cell" --critic_ckpt "$CRITIC" --tracker_onnx "$TRACKER" --fm_onnx "$FM"
          --clip_lib "$CLIPS" --tiles "$tile" --k "$K" --episodes "$EPISODES" --speed "$sp"
          --approach 2.5 --device cuda --scene_tag "_bc${SHARD}"
          --record_selected "$cell/selected.h5")
    [ -n "$klass" ] && args+=(--klass "$klass")
    [ "$CANDIDATES" = "1" ] && args+=(--record_candidates)
    echo "[collect $SHARD] START $(date +%H:%M:%S) t$tile s$sp"
    "$PY" -u live_select.py "${args[@]}" > "$cell/run.log" 2>&1
    echo "[collect $SHARD] DONE  $(date +%H:%M:%S) t$tile s$sp :: $(grep -h 'recorded\|SUCCESS' "$cell/run.log" | tr '\n' ' ')"
  done
done
echo "[collect $SHARD] ALL DONE $(date +%H:%M:%S)"
