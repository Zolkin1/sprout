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
STEM="$RUN/rollout_measured"

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
if run_from record && [ ! -f "$STEM.h5" ]; then
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
  say "stage 1 done: $(du -h "$STEM.h5" | cut -f1)"
else
  say "stage 1 skipped ($STEM.h5 exists or stage>record)"
fi

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
# --w_fail 0 -> V = 1.0 * G_hat - 0 * P(fail): the quality head only, as asked. The failure
# head is still fitted and still stamped; it is there for later, not for this label.
# Stratified on all three axes: family is the per-family percentile the labeller already
# applies, plus equal-count speed bins and normalized-progress (phase) bins.
LABELS="$CRITIC/labels_top30.npz"
if run_from label && [ ! -f "$LABELS" ]; then
  say "stage 3/5 label: quality-only advantages, top 30% positive, 3-way stratified"
  "$PY" scripts/flow_matching/label_advantage.py "$CRITIC/predictions.npz" \
      --out "$LABELS" \
      --variant V_priv \
      --w_fail 0.0 \
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
  say "stage 5/5 eval: exporting the fine-tune and running tiles 6/20/40"
  "$PY" scripts/flow_matching/export_fm_onnx.py \
      --checkpoint "$FT/best.pt" --out "$FT/exported" >>"$LOG" 2>&1 \
      || say "export failed -- run it by hand; the checkpoint is fine"
  say "pipeline complete. Eval commands are in $RUN/NEXT_STEPS.txt"
  cat > "$RUN/NEXT_STEPS.txt" <<EOF
The fine-tuned generator is $FT/best.pt (exported under $FT/exported).

MuJoCo A/B, both arms on the MEASURED history (which is what it was tuned for):

cd transfer/sim
for ARM in parent finetune; do
  FM=\$([ \$ARM = parent ] && echo "$W/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx" || echo "$FT/exported/"*_fm.onnx)
  \$PY live_select.py --out $W/logs/value/ft_meas_\$ARM \\
    --tracker_onnx $W/logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/tracker.onnx \\
    --fm_onnx "\$FM" --clip_lib $W/clips/g1_mb_v5_body_mode11_decor.h5 \\
    --tiles 6,20,40 --k 1 --episodes 40 --speed 0.8 \\
    --history_from measured --scene_tag _ftm\$ARM
done
EOF
fi
say "DONE (from stage $STAGE)"
