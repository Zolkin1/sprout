#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Fetch the two from-scratch N=3 generators from talos and evaluate them, MuJoCo first then
# Isaac. Waits for the running Isaac cell to finish so the two never contend for the GPU.
#
# BOTH models, not just the measured one. `measured_n3` alone is a number with no baseline:
# the whole reason `reference_n3` was trained on the same data, same seed and same N is that
# it is the only control which isolates the history source. The deployed generator is a
# third reference point but differs in N (4 vs 3) and in training run, so it is not the
# control.
#
# EACH ON THE ARM IT WAS TRAINED FOR, so no guard override is needed anywhere:
#   measured_n3  -> history_from=measured   (declares history_source=measured)
#   reference_n3 -> history_from=generator  (declares history_source=reference)
# Serving either the other arm is what `assert_history_source_matches` refuses, and here
# there is no reason to ask for it.
#
# Resumable: every stage skips when its output exists.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$W/env_isaaclab/bin/python"
export PYTHONPATH="$W/source/robot_rl:$W/source/motion_gen"
cd "$W" || exit 1

TALOS=talos.caltech.edu
TD=/mnt/zolkin_new/sprout/measured-history/logs/flow_matching
M_RUN=2026-09-03_01-42-04_2026-09-03_measured_n3
R_RUN=2026-09-03_01-42-08_2026-09-03_reference_n3
LOCAL="$W/logs/flow_matching"
EXPORTED="$W/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported"
CLIPLIB="$W/clips/g1_mb_v5_body_mode11_decor.h5"
LOG="$W/logs/flow_matching/eval_n3_pair.log"
say () { echo "[n3 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- 0. wait for the GPU ---------------------------------------------------------------
if pgrep -f "[e]val/evaluate.py" >/dev/null 2>&1; then
  say "waiting for the running Isaac cell to finish (one GPU)"
  while pgrep -f "[e]val/evaluate.py" >/dev/null 2>&1; do sleep 120; done
fi
say "GPU free"

# ---- 1. fetch ---------------------------------------------------------------------------
# best.pt is EMA weights selected on val_ema/flow_mse. Fetched at whatever epoch the run has
# reached; the file is rewritten as training improves, so this is a snapshot, not the final.
for pair in "measured_n3:$M_RUN" "reference_n3:$R_RUN"; do
  name=${pair%%:*}; run=${pair#*:}
  dst="$LOCAL/n3_$name"
  mkdir -p "$dst"
  if [ ! -f "$dst/best.pt" ]; then
    say "fetching $name best.pt"
    scp -q "$TALOS:$TD/$run/best.pt" "$dst/best.pt" || { say "scp $name FAILED"; exit 1; }
    scp -q "$TALOS:$TD/$run/params/"* "$dst/" 2>/dev/null || true
  fi
  say "$name: $(du -h "$dst/best.pt" | cut -f1)  epoch $("$PY" -c "
import torch,sys; print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch','?'))
" "$dst/best.pt" 2>/dev/null)"
done

# ---- 2. export --------------------------------------------------------------------------
# --out is an output .onnx PATH, not a directory: a directory writes an extensionless file
# and still exits 0.
for name in measured_n3 reference_n3; do
  d="$LOCAL/n3_$name"; onnx="$d/exported/n3_${name}_fm.onnx"
  mkdir -p "$d/exported"
  if [ ! -f "$onnx" ]; then
    say "exporting $name"
    "$PY" scripts/flow_matching/export_fm_onnx.py \
        --checkpoint "$d/best.pt" --out "$onnx" >>"$LOG" 2>&1 || { say "export $name FAILED"; exit 2; }
    [ -f "$onnx" ] || { say "export $name exited 0 but wrote nothing"; exit 2; }
  fi
  say "$name onnx: $(du -h "$onnx" | cut -f1)"
done

# ---- 3. MuJoCo: the quick tile sweep ----------------------------------------------------
for pair in "measured_n3:measured" "reference_n3:generator"; do
  name=${pair%%:*}; arm=${pair#*:}
  onnx="$LOCAL/n3_$name/exported/n3_${name}_fm.onnx"
  for SPEED in 0.8 1.6 2.4; do
    out="$W/logs/value/n3_${name}_s${SPEED}"
    [ -f "$out/live_select.json" ] && { say "mujoco $name @$SPEED skipped"; continue; }
    rm -rf "$out"
    say "mujoco $name @$SPEED (history_from=$arm)"
    ( cd "$W/transfer/sim" && "$PY" live_select.py \
        --out "$out" \
        --tracker_onnx "$EXPORTED/tracker.onnx" --fm_onnx "$onnx" \
        --clip_lib "$CLIPLIB" \
        --tiles 6,20,40 --k 1 --episodes 40 --speed "$SPEED" \
        --history_from "$arm" --scene_tag "_n3$name" ) >>"$LOG" 2>&1 \
      || say "mujoco $name @$SPEED FAILED (continuing)"
  done
done

say "MuJoCo done -- paired table (reference_n3 vs measured_n3, each on its own arm):"
"$PY" scripts/rsl_rl/report_measured_pairs.py \
    --a "$W/logs/value/n3_reference_n3_s*" --b "$W/logs/value/n3_measured_n3_s*" \
    --label_a ref_n3 --label_b meas_n3 2>&1 | tee -a "$LOG"

# ---- 4. Isaac: one grid per generator, each with only its valid mode --------------------
# Separate runs rather than one grid with two labels: the grid crosses modes x labels, which
# would build `fm__train__meas` (a measured generator served the generator arm) -- a cell
# the guard refuses and that nothing should want.
run_isaac () {   # $1 name, $2 mode, $3 checkpoint
  local out="logs/terrain_eval/n3_$1"
  if [ -f "$W/$out/plan.json" ]; then say "isaac $1 plan exists"; else
    say "isaac $1: plan"
    "$PY" scripts/rsl_rl/eval/evaluate.py --stage plan --out "$out" \
        --lib clips/g1_mb_v5_body_mode11_decor.h5 \
        --checkpoint logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/model_19999.pt \
        --fm_checkpoint "n3=$3" \
        --clips_per_tile 2 --flat_tiles 12 --min_clip_len 52 \
        --exclude_families distractor \
        --tile_pick spread --clip_pick spread --plan_seed 20260820 \
        --reference_modes "$2" --dr_presets train \
        --goal_regions "$W/logs/rollouts/v2/goal_regions.json" >>"$LOG" 2>&1 \
      || { say "isaac $1 plan FAILED"; return 1; }
  fi
  say "isaac $1: grid ($2)"
  "$PY" scripts/rsl_rl/eval/evaluate.py --stage grid --out "$out" --num_envs 256 \
      --goal_regions "$W/logs/rollouts/v2/goal_regions.json" >>"$LOG" 2>&1 \
    || say "isaac $1 grid returned nonzero"
  "$PY" scripts/rsl_rl/eval/report.py "$W/$out" >>"$LOG" 2>&1 || say "isaac $1 report nonzero"
}
run_isaac measured  fm_meas "$LOCAL/n3_measured_n3/best.pt"
run_isaac reference fm      "$LOCAL/n3_reference_n3/best.pt"
say "ALL DONE"
