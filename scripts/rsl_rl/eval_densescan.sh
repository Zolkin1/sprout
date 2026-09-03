#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Evaluate the PRIVILEGED dense-height-scan generator, which finished training on
# 2026-08-30 and was never tested. It is the control arm that isolates what camera
# exteroception costs the generator: matched to 2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff
# on every non-extero hyperparameter (26.9M vs 27M params, same tracker, same clip library),
# differing only in extero_source=height_scan on the dense 121x121 grid, a 4-layer CNN to
# 256 tokens, and extero_clip 1.5.
#
# **SIM ONLY, and the numbers mean something narrower than usual.** The dense scan is exact,
# noise-free ground truth that no robot can produce; MuJoCo can, casting the same grid
# against the same terrain at ~20 ms/step for 14641 rays. So this measures the model GIVEN
# PERFECT TERRAIN -- an upper bound on what better exteroception could buy, not a deployable
# configuration. Both the export and the runner refuse it without --allow_privileged_extero.
#
# It trained on REFERENCE history (it predates the history_source flag, so from_dict resolves
# it to 'reference'), hence --history_from generator: the arm it was fitted for.
#
# One asymmetry against the depth arm, recorded rather than hidden: depth_latency_probs
# cannot carry over, because depth_lag is depth-shaped (N, 4, 2, 780) and can never serve a
# 14641-wide scan. So the depth generator trained with a 48 ms camera lag and this one with
# none.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$W/env_isaaclab/bin/python"
export PYTHONPATH="$W/source/robot_rl:$W/source/motion_gen"
cd "$W" || exit 1

TALOS=talos.caltech.edu
SRC=/mnt/zolkin_new/sprout/sprout/logs/flow_matching/2026-08-29_11-59-05_densescan_cnn256_v1
DST="$W/logs/flow_matching/densescan_cnn256_v1"
ONNX="$DST/exported/densescan_cnn256_v1_fm.onnx"
EXPORTED="$W/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported"
LOG="$DST/eval.log"
mkdir -p "$DST/exported"
say () { echo "[dscan $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- wait for the GPU (the Isaac grid has it) --------------------------------------------
if pgrep -f "[e]val/evaluate.py" >/dev/null 2>&1; then
  say "waiting for the Isaac grid to finish (one local GPU)"
  while pgrep -f "[e]val/evaluate.py" >/dev/null 2>&1; do sleep 120; done
fi

# ---- fetch --------------------------------------------------------------------------------
if [ ! -f "$DST/best.pt" ]; then
  say "fetching best.pt (431 MB) from talos"
  scp -q "$TALOS:$SRC/best.pt" "$DST/best.pt" || { say "scp FAILED"; exit 1; }
fi
say "checkpoint epoch $("$PY" -c "
import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
cfg=c['config']; print(c.get('epoch'), 'extero', cfg.get('extero_source'), cfg.get('extero_sensor_dim'),
      'clip', cfg.get('extero_clip'), 'hist_src', cfg.get('history_source','<absent->reference>'))
" "$DST/best.pt")"

# ---- export -------------------------------------------------------------------------------
# --out is a .onnx PATH, not a directory.
if [ ! -f "$ONNX" ]; then
  say "exporting (--allow_privileged_extero)"
  "$PY" scripts/flow_matching/export_fm_onnx.py \
      --checkpoint "$DST/best.pt" --out "$ONNX" \
      --allow_privileged_extero >>"$LOG" 2>&1 || { say "export FAILED"; exit 2; }
  [ -f "$ONNX" ] || { say "export exited 0 but wrote nothing"; exit 2; }
fi
say "onnx $(du -h "$ONNX" | cut -f1)"

# ---- the quick 9-cell MuJoCo sweep --------------------------------------------------------
for SPEED in 0.8 1.6 2.4; do
  OUT="$W/logs/value/dscan_s${SPEED}"
  [ -f "$OUT/live_select.json" ] && { say "@$SPEED skipped"; continue; }
  rm -rf "$OUT"
  say "@$SPEED (dense scan cast in MuJoCo, ~20 ms/step)"
  ( cd "$W/transfer/sim" && "$PY" live_select.py \
      --out "$OUT" \
      --tracker_onnx "$EXPORTED/tracker.onnx" --fm_onnx "$ONNX" \
      --clip_lib "$W/clips/g1_mb_v5_body_mode11_decor.h5" \
      --tiles 6,20,40 --k 1 --episodes 40 --speed "$SPEED" \
      --history_from generator --allow_privileged_extero \
      --scene_tag "_dscan" ) >>"$LOG" 2>&1 \
    || say "@$SPEED FAILED (continuing)"
done

say "done -- dense scan vs the DEPTH generator on the same tiles/speeds/seeds:"
"$PY" scripts/rsl_rl/report_measured_pairs.py \
    --a "$W/logs/value/hist_ab_s*_generator" --b "$W/logs/value/dscan_s*" \
    --label_a depth --label_b dense_scan 2>&1 | tee -a "$LOG"
