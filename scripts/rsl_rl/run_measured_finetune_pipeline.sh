#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The whole filtered-BC pipeline, with the generator conditioned on the ROBOT'S OWN state.
#
#   1. record  -- RL + FM rollouts in Isaac under history_from=measured
#   2. critic  -- fit_value, two heads (failure + quality)
#   3. label   -- advantages from the QUALITY head alone (w_fail 0)
#   4. finetune-- filtered BC on the top 30%, stratified by family x speed x progress
#   5. eval    -- MuJoCo, the fine-tuned pair against its parent
#
# RESUMABLE BY DESIGN: every stage skips when its output exists, so a failure or a kill
# costs only the stage it was in. Stage 1 is hours and stage 4 is the one most likely to
# want its knobs changed, and neither should force the other to be redone.
#
# READ THE CEILING FIRST. This exact pipeline was run to completion on the generator arm on
# 2026-09-02 and STOPPED on a causal ceiling: from an exactly-restored MuJoCo state the
# chunk choice is only 1.6% of the tracking reward's variance, a perfect best-of-4 is worth
# +0.0207, and the previous fine-tunes already took +0.011-0.014 of that. Success never
# moved, and a no-critic `survived` filter matched or beat the critic one. The hypothesis
# here is that measured conditioning changes that; the prior is that it does not, and the
# arm's own eval measured it as worth +0.025 (p=0.36) before any fine-tuning.
#
# Usage: bash scripts/rsl_rl/run_measured_finetune_pipeline.sh [stage]
#   stage: optional first stage to run (record|critic|label|finetune|eval); default record.
set -u

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$W/env_isaaclab/bin/python"
export PYTHONPATH="$W/source/robot_rl:$W/source/motion_gen"
cd "$W" || exit 1

# ---- the artifacts this run is defined by, named once ---------------------------------
FM_PARENT="$W/logs/flow_matching/2026-08-22_19-14-39_plan_v1_lag48_aug/best.pt"
TRACKER="$W/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/model_19999.pt"
CLIPS="$W/logs/rollouts/v2/eval_clips.h5"
GOALS="$W/logs/rollouts/v2/goal_regions.json"
TASK="G1-tracker-clf-fm-terrain-depth-play"

RUN="$W/logs/rollouts/v5_meas"
CRITIC="$W/logs/value/critic_meas_v1"
FT="$W/logs/flow_matching/ft_meas_v1"
SPEC="$RUN/tracking_spec_v1.json"

# The rollout stem is DERIVED, never assumed. ``record_rollouts`` names its output after the
# TRACKER RUN (`rollout_<run>.h5`), not after anything this script passes -- `--output_dir`
# only chooses the directory. Hard-coding a stem here cost a stage: `du` printed an empty
# size, the script carried on, and `fit_value` died on a missing file 41 minutes later with
# the recording sitting on disk under its real name.
resolve_stem () {
  local hits=()
  shopt -s nullglob
  for f in "$RUN"/rollout_*.h5; do
    case "$f" in *.episodes.h5) ;; *) hits+=("$f");; esac
  done
  shopt -u nullglob
  [ ${#hits[@]} -eq 1 ] || return 1
  STEM="${hits[0]%.h5}"
}
STEM=""

# Local card is 12 GB; the reference rollout used 8192 envs on a 95 GB H100.
NUM_ENVS=${NUM_ENVS:-512}
NUM_WINDOWS=${NUM_WINDOWS:-300000}

mkdir -p "$RUN" "$CRITIC" "$(dirname "$FT")"
LOG="$RUN/pipeline.log"
say () { echo "[pipe $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die () { say "FAILED at $1 (exit $2). Nothing later ran."; exit "$2"; }

for f in "$FM_PARENT" "$TRACKER" "$CLIPS" "$GOALS"; do
  [ -f "$f" ] || { say "missing required artifact: $f"; exit 2; }
done

STAGE="${1:-record}"
run_from () { case "$STAGE" in record) return 0;; critic) [ "$1" != record ];;
  label) [ "$1" != record ] && [ "$1" != critic ];;
  finetune) [ "$1" = finetune ] || [ "$1" = eval ];; eval) [ "$1" = eval ];; esac; }

# ---- 0. the tracking spec ---------------------------------------------------------------
# Lifted out of the reference rollout's own attributes rather than re-fitted: the critic's
# quality head predicts a normalized return under ONE set of sigmas and gamma, and
# label_advantage's G has to be the same quantity. Re-fitting would silently make it another.
if [ ! -f "$SPEC" ]; then
  say "writing tracking spec from the v4_big rollout's attrs -> $SPEC"
  "$PY" - "$SPEC" <<'EOF' >>"$LOG" 2>&1 || die spec $?
import h5py, json, sys
src = ("logs/rollouts/v4_big/rollout_2026-08-20_16-36-10_mode11_decor_plan_v1.h5")
with h5py.File(src, "r") as f:
    spec = f.attrs["tracking_spec"]
spec = spec.decode() if isinstance(spec, bytes) else str(spec)
json.dump(json.loads(spec), open(sys.argv[1], "w"), indent=1)
print("tracking spec:", spec[:160])
EOF
fi

# ---- 1. record rollouts under the MEASURED arm ------------------------------------------
if run_from record && ! resolve_stem; then
  say "stage 1/5 record: $NUM_WINDOWS windows, $NUM_ENVS envs, history_from=measured"
  "$PY" scripts/rsl_rl/record_rollouts.py \
      --task "$TASK" --headless \
      --checkpoint "$TRACKER" \
      --fm_checkpoint "$FM_PARENT" \
      --clip "$CLIPS" --clip_device cpu \
      --goal_regions "$GOALS" \
      --tracking_spec "$SPEC" \
      --history_from measured \
      --allow_history_source_mismatch \
      --num_envs "$NUM_ENVS" --num_windows "$NUM_WINDOWS" \
      --output_dir "$RUN" >>"$LOG" 2>&1 || die record $?
fi
# Resolved (or re-resolved) unconditionally, and a hard failure: every later stage reads it,
# and the whole point of the block above is that this name is not ours to predict.
resolve_stem || { say "expected exactly one rollout_*.h5 in $RUN; found $(ls "$RUN"/rollout_*.h5 2>/dev/null | grep -vc episodes) -- cannot continue"; exit 4; }
say "rollout stem: $STEM ($(du -h "$STEM.h5" | cut -f1))"

# ---- 2. the critic: failure head + quality head ------------------------------------------
if run_from critic && [ ! -f "$CRITIC/predictions.npz" ]; then
  say "stage 2/5 critic: fit_value, two heads"
  "$PY" scripts/flow_matching/fit_value.py "$STEM" \
      --fm_checkpoint "$FM_PARENT" \
      --out "$CRITIC" \
      --quality_weight 1.0 \
      --no_wandb >>"$LOG" 2>&1 || die critic $?
  say "stage 2 done"
else
  say "stage 2 skipped"
fi

# ---- 3. advantages from the QUALITY head alone -------------------------------------------
# The critic has ONE head and the advantage is A = G - V, so "quality only" is structural
# rather than a flag: the survival head that `--w_fail 0` used to switch off is gone.
# Stratified on all three axes: family is the per-family percentile the labeller already
# applies, plus equal-count speed bins and normalized-progress (phase) bins.
LABELS="$CRITIC/labels_top30.npz"
if run_from label && [ ! -f "$LABELS" ]; then
  say "stage 3/5 label: quality-only advantages, top 30% positive, 3-way stratified"
  "$PY" scripts/flow_matching/label_advantage.py "$CRITIC/predictions.npz" \
      --out "$LABELS" \
      --variant V_priv \
      --percentile 70.0 \
      --rollout "$STEM" \
      --speed_bins 3 \
      --progress_bins 5 >>"$LOG" 2>&1 || die label $?
  say "stage 3 done"
else
  say "stage 3 skipped"
fi

# ---- 4. filtered BC on the positives -----------------------------------------------------
if run_from finetune && [ ! -f "$FT/best.pt" ]; then
  say "stage 4/5 finetune: filtered BC on the positive label"
  "$PY" scripts/flow_matching/finetune_generator.py \
      --pretrained "$FM_PARENT" \
      --rollout "$STEM.h5" \
      --labels "$LABELS" \
      --filter_label positive \
      --out "$FT" \
      --no_wandb >>"$LOG" 2>&1 || die finetune $?
  say "stage 4 done -> $FT"
else
  say "stage 4 skipped"
fi

# ---- 5. MuJoCo eval, fine-tune against its parent ---------------------------------------
# Both arms served the MEASURED history, which is what the fine-tune was trained for. The
# parent under that arm is the same comparison already measured at +0.025 -- so the number
# that matters is fine-tune-vs-parent, both measured, not either against the generator arm.
if run_from eval; then
  say "stage 5/5 eval: export, then tiles 6/20/40 x 3 speeds, BOTH arms"
  # --out is an output .onnx PATH, not a directory: passing a directory writes a 130 MB
  # extensionless file called `exported` and still exits 0, which is how this cost a stage.
  # Named to the deploy convention (`<run>_fm.onnx`) so resolve_fm_onnx would accept it.
  mkdir -p "$FT/exported"
  FT_TARGET="$FT/exported/$(basename "$FT")_fm.onnx"
  if [ ! -f "$FT_TARGET" ]; then
    "$PY" scripts/flow_matching/export_fm_onnx.py \
        --checkpoint "$FT/best.pt" --out "$FT_TARGET" >>"$LOG" 2>&1 || die export $?
    [ -f "$FT_TARGET" ] || { say "export exited 0 but wrote no $FT_TARGET"; exit 5; }
  fi

  # Resolved by GLOB, not by name: export_fm_onnx names the file after the run it came from
  # (fm_artifact_name), so a hard-coded name rots the moment the directory is renamed. More
  # than one match is REFUSED for the same reason resolve_fm_onnx refuses it -- defaulting to
  # the newest would reinstate as a silent default the ambiguity the naming removed.
  shopt -s nullglob
  FT_ONNX=("$FT/exported"/*_fm.onnx)
  shopt -u nullglob
  [ ${#FT_ONNX[@]} -eq 1 ] || { say "want exactly one *_fm.onnx in $FT/exported, got ${#FT_ONNX[@]}"; exit 3; }
  EXPORTED="$W/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported"
  PARENT_ONNX="$EXPORTED/2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx"

  # BOTH arms run here rather than reusing tonight's measured-arm sweep, which looks like a
  # free baseline and is not: that sweep drove the 2026-08-25 nojumpoff generator, while
  # this fine-tune's parent is the deployed 2026-08-22 one. Pairing across two different
  # parents would charge their difference to the fine-tune.
  #
  # Sequential, and one --scene_tag per arm: CourseRig writes robots/g1/<robot>_<course>.xml
  # from the course before loading it, so two concurrent runs on one tile tear that file
  # silently rather than failing to load.
  for SPEED in 0.8 1.6 2.4; do
    for ARM in parent finetune; do
      OUT="$W/logs/value/ftmeas_s${SPEED}_$ARM"
      if [ -f "$OUT/live_select.json" ]; then say "  eval $ARM @$SPEED skipped"; continue; fi
      FM=$([ "$ARM" = parent ] && echo "$PARENT_ONNX" || echo "${FT_ONNX[0]}")
      say "  eval $ARM @$SPEED"
      ( cd "$W/transfer/sim" && "$PY" live_select.py \
          --out "$OUT" \
          --tracker_onnx "$EXPORTED/tracker.onnx" --fm_onnx "$FM" \
          --clip_lib "$W/clips/g1_mb_v5_body_mode11_decor.h5" \
          --tiles 6,20,40 --k 1 --episodes 40 --speed "$SPEED" \
          --history_from measured --scene_tag "_ftm$ARM" ) >>"$LOG" 2>&1 \
        || say "  eval $ARM @$SPEED FAILED (continuing: the remaining cells still run)"
    done
  done

  say "stage 5 done -- paired table:"
  "$PY" scripts/rsl_rl/report_measured_pairs.py \
      --a "$W/logs/value/ftmeas_s*_parent" \
      --b "$W/logs/value/ftmeas_s*_finetune" \
      --label_a parent --label_b finetune 2>&1 | tee -a "$LOG"
fi
say "DONE (from stage $STAGE)"
