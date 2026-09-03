#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The dense-scan vs depth comparison on a SECOND, independent tile per family.
#
# Why this matters more than more episodes: every sweep tonight rests on tiles 6/20/40, so
# three clusters, and per-cell effects have failed to replicate three separate times. A
# second tile per family is the only thing that can tell a FAMILY property from a TILE
# quirk -- specifically whether the dense scan's large tile-6 advantage is about stairs_up
# or about tile 6, whose failures are known to be dominated by heading drift.
#
#   tile  7  stairs_up    +0.680 m   (6 is +0.850)
#   tile 21  stairs_down  -0.390 m   (20 is -0.777)
#   tile 42  box          +0.600 m   (40 is +0.400)
#
# 42 rather than 41 (+0.250) on purpose: tile 40 saturated at 1.000 with ZERO discordant
# pairs at the low speeds, so the informative box is a harder one, not an easier one.
#
# BOTH ARMS, which is 18 cells rather than 9. The depth baseline does not exist on these
# tiles, and a dense-scan number with nothing to pair against measures nothing -- the whole
# value is the paired delta on the same tiles, speeds and episode seeds.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$W/env_isaaclab/bin/python"
export PYTHONPATH="$W/source/robot_rl:$W/source/motion_gen"
cd "$W" || exit 1

TILES=${TILES:-7,21,42}
TAG=${TAG:-t2}
EXPORTED="$W/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported"
DSCAN_ONNX="$W/logs/flow_matching/densescan_cnn256_v1/exported/densescan_cnn256_v1_fm.onnx"
# The generator the dense-scan run was MATCHED to on every non-extero hyperparameter.
DEPTH_ONNX="$EXPORTED/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx"
LOG="$W/logs/flow_matching/densescan_cnn256_v1/eval_${TAG}.log"
say () { echo "[dscan2 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for f in "$DSCAN_ONNX" "$DEPTH_ONNX" "$EXPORTED/tracker.onnx"; do
  [ -f "$f" ] || { say "missing $f"; exit 2; }
done

# One local GPU, and live_select writes robots/g1/<robot>_<course>.xml from the course before
# loading it -- two runs on one tile tear that file silently.
if pgrep -f "[l]ive_select.py" >/dev/null 2>&1 || pgrep -f "[e]val/evaluate.py" >/dev/null 2>&1; then
  say "waiting for the running eval to finish"
  while pgrep -f "[l]ive_select.py" >/dev/null 2>&1 || pgrep -f "[e]val/evaluate.py" >/dev/null 2>&1; do
    sleep 120
  done
fi
say "GPU free; tiles $TILES"

# Depth first: it is ~5x faster per cell (mj_multiRay casts both cameras in ~3.9 ms, while a
# height scan's rays do not share an origin so the dense grid is a 14641-iteration mj_ray
# loop at ~20 ms/step). Finishing the cheap arm first means a partial run still pairs.
for pair in "depth:$DEPTH_ONNX" "dscan:$DSCAN_ONNX"; do
  arm=${pair%%:*}; onnx=${pair#*:}
  extra=""; [ "$arm" = dscan ] && extra="--allow_privileged_extero"
  for SPEED in 0.8 1.6 2.4; do
    out="$W/logs/value/${TAG}_${arm}_s${SPEED}"
    [ -f "$out/live_select.json" ] && { say "$arm @$SPEED skipped"; continue; }
    rm -rf "$out"
    say "$arm @$SPEED"
    ( cd "$W/transfer/sim" && "$PY" live_select.py \
        --out "$out" \
        --tracker_onnx "$EXPORTED/tracker.onnx" --fm_onnx "$onnx" \
        --clip_lib "$W/clips/g1_mb_v5_body_mode11_decor.h5" \
        --tiles "$TILES" --k 1 --episodes 40 --speed "$SPEED" \
        --history_from generator $extra \
        --scene_tag "_${TAG}${arm}" ) >>"$LOG" 2>&1 \
      || say "$arm @$SPEED FAILED (continuing)"
  done
done

say "done -- dense scan vs depth on tiles $TILES:"
"$PY" scripts/rsl_rl/report_measured_pairs.py \
    --a "$W/logs/value/${TAG}_depth_s*" --b "$W/logs/value/${TAG}_dscan_s*" \
    --label_a depth --label_b dense_scan 2>&1 | tee -a "$LOG"

say "and the ORIGINAL tiles 6/20/40, for the replication check:"
"$PY" scripts/rsl_rl/report_measured_pairs.py \
    --a "$W/logs/value/hist_ab_s*_generator" --b "$W/logs/value/dscan_s*" \
    --label_a depth --label_b dense_scan 2>&1 | tee -a "$LOG"
