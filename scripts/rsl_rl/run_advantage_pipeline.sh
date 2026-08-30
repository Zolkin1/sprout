#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The whole advantage-conditioning chain, in the one order it works in:
#
#   plan -> record -> fit critic -> label -> fine-tune -> export both arms
#
# Every stage feeds the next by PATH, and each is `set -e` fatal, so a failure at 02:00
# stops the chain rather than handing the next stage a truncated or absent input. Each
# stage's stdout goes to its own log beside the artifacts.
#
# Usage:  scripts/rsl_rl/run_advantage_pipeline.sh <tag> [<clips_per_tile>] [<num_episodes>]
set -euo pipefail

TAG="${1:?usage: run_advantage_pipeline.sh <tag> [clips_per_tile] [num_episodes]}"
CLIPS_PER_TILE="${2:-31}"
NUM_EPISODES="${3:-15000}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# The frozen pair: the tracker that drove the robot and the generator trained against it.
# Named here rather than passed, because the whole point of the experiment is that the
# TRACKER never moves -- so a run that used a different one is not comparable and should
# not be reachable by editing a command line.
TEACHER="2026-08-20_16-36-10_mode11_decor_plan_v1"
TRACKER_CKPT="logs/rsl_rl/g1/${TEACHER}/model_19999.pt"
FM_CKPT="logs/flow_matching/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff/best.pt"
LIB="clips/g1_mb_v5_body_mode11_decor.h5"
SKIP="jump_off,distractor,stairs_dn_pure,stairs_up_pure"

ROLL="logs/rollouts/${TAG}"
VAL="logs/value/${TAG}"
FT="logs/flow_matching/$(basename "$(dirname "$FM_CKPT")")_ac_${TAG}"
VAL_F1="logs/value/${TAG}_fold1"
FT_F1="${FT}_fold1"
STEM="${ROLL}/rollout_${TEACHER}"

# `--flat_clips` is deliberately a small share (~7%). Flat succeeds ~99% of the time, so it
# contributes almost no failures and therefore almost nothing to a failure classifier -- and
# its goal region is the one under-calibrated by the fit (7 surviving clip-arm episodes).
FLAT_CLIPS=175
NUM_ENVS=128
FIT_EPOCHS=20
FIT_FOLDS=2
ADV_N=8
FT_EPOCHS=6

export PYTHONPATH="source/robot_rl:source/motion_gen"
# shellcheck disable=SC1091
source env_isaaclab/bin/activate

mkdir -p "$ROLL" "$VAL" "$FT" "$VAL_F1" "$FT_F1"
echo "[chain] tag=$TAG  clips/tile=$CLIPS_PER_TILE  episodes=$NUM_EPISODES"
echo "[chain] rollout=$ROLL  value=$VAL  finetune=$FT"
date

# ---------------------------------------------------------------- 1. plan + subset library
# Also fits the goal regions from the CLIP arm, so the region is never tighter than the
# tracker's own demonstrated ability and a failure is attributable to the generator.
if [ ! -f "${ROLL}/eval_clips.h5" ]; then
  echo "[chain] 1/6 plan"
  python scripts/rsl_rl/eval/evaluate.py --stage plan \
      --lib "$LIB" --load_run "$TEACHER" --checkpoint "$TRACKER_CKPT" \
      --fm_checkpoint "gen_a=${FM_CKPT}" \
      --exclude_families "$SKIP" \
      --clips_per_tile "$CLIPS_PER_TILE" --flat_tiles 8 --flat_clips "$FLAT_CLIPS" \
      --num_envs "$NUM_ENVS" --out "$ROLL" 2>&1 | tee "${ROLL}/plan.log"
else
  echo "[chain] 1/6 plan -- eval_clips.h5 exists, skipping"
fi
# The goal regions are the previous run's, refitted only when absent: they are a property of
# the TRACKER, which never moves here, and refitting per run would make two rollouts score
# against subtly different regions.
if [ ! -f "${ROLL}/goal_regions.json" ]; then
  cp logs/rollouts/v1/goal_regions.json "${ROLL}/goal_regions.json"
  echo "[chain] copied goal_regions.json from logs/rollouts/v1"
fi

# --------------------------------------------------------------------------- 2. the rollout
if [ ! -f "${STEM}.h5" ]; then
  echo "[chain] 2/6 record  ($NUM_EPISODES episodes)"
  python scripts/rsl_rl/record_rollouts.py \
      --task=G1-tracker-clf-fm-terrain-depth-play \
      --load_run "$TEACHER" --checkpoint "$TRACKER_CKPT" --fm_checkpoint "$FM_CKPT" \
      --clip "${ROLL}/eval_clips.h5" --clip_device cpu \
      --goal_regions "${ROLL}/goal_regions.json" \
      --skip_families "$SKIP" \
      --num_envs "$NUM_ENVS" --num_episodes "$NUM_EPISODES" --num_windows 450000 \
      --output_dir "$ROLL" 2>&1 | tee "${ROLL}/record.log"
else
  echo "[chain] 2/6 record -- ${STEM}.h5 exists, skipping"
fi
date

# ----------------------------------------------------------------------------- 3. the critic
# TWO arms, and the ORDER matters: the cross-fit one is the primary, so it completes end to
# end first and a night that runs short still produces the deliverable.
#
#   --folds 2  every window is scored by a model that did not train on its episode.
#   --folds 1  the naive fit, in sample -- the CONTROL for how much the fold structure buys.
#
# Both cost the same wall-clock per variant (each fold model trains on 1/K of the data), and
# the fold-1 arm fits V_priv only, since V_onboard's role is the information-gap diagnostic
# and the primary arm already measures it.
#
# What the comparison measures: an in-sample critic can memorise the outcome of the episode
# it is scoring, and every window of an episode carries the SAME label, so the effective
# sample size is the episode count. Under memorisation V(s_t) and V(s_{t+n}) both collapse
# onto the realized outcome and the advantage goes to ZERO -- the label degrades to noise
# while still looking entirely normal on disk. If the two arms' labels and G2 gates agree,
# that did not happen here and the folds are not buying anything.
fit_and_label () {                       # $1 = out dir, $2 = folds, $3 = extra fit args
  local out="$1" folds="$2"
  shift 2
  mkdir -p "$out"
  echo "[chain] fit critic -> $out  ($FIT_EPOCHS epochs, $folds fold(s))"
  python scripts/flow_matching/fit_value.py "$STEM" \
      --fm_checkpoint "$FM_CKPT" --out "$out" \
      --epochs "$FIT_EPOCHS" --folds "$folds" "$@" \
      --run_name "value_$(basename "$out")" 2>&1 | tee "${out}/fit.log"
  echo "[chain] label -> $out  (n=$ADV_N)"
  python scripts/flow_matching/label_advantage.py "${out}/predictions.npz" \
      --out "${out}/advantage.npz" --n "$ADV_N" --variant V_priv \
      2>&1 | tee "${out}/label.log"
}

echo "[chain] 3/6 + 4/6 critic + label, cross-fit arm"
fit_and_label "$VAL" "$FIT_FOLDS"
date

# --------------------------------------------------------------------------- 5. the fine-tune
echo "[chain] 5/6 fine-tune ($FT_EPOCHS epochs)"
python scripts/flow_matching/finetune_advantage.py \
    --pretrained "$FM_CKPT" --rollout "${STEM}.h5" --labels "${VAL}/advantage.npz" \
    --out "$FT" --epochs "$FT_EPOCHS" --drop_tile_holdout \
    2>&1 | tee "${FT}/finetune.log"
date

# ----------------------------------------------------------------- 6. both arms, for the check
# The negative arm is not optional and not a later refinement: if it performs the same as the
# positive one the model ignored the label, and every other number from this run is noise.
echo "[chain] 6/6 export positive + negative"
for LABEL in positive negative; do
  python scripts/flow_matching/export_fm_onnx.py --checkpoint "${FT}/best.pt" \
      --advantage_label "$LABEL" 2>&1 | grep -E "^\[INFO\] (Wrote|batch)"
done
echo "[chain] PRIMARY ARM DONE."
date

# ------------------------------------------------------- 7. the single-critic control, after
# Everything above is complete at this point, so this is additive: if it is still running in
# the morning, kill it and nothing is lost.
echo "[chain] 7/8 single-critic control"
fit_and_label "$VAL_F1" 1 --variants V_priv
python scripts/flow_matching/compare_advantage_labels.py \
    "${VAL}/advantage.npz" "${VAL_F1}/advantage.npz" \
    --names cross_fit in_sample 2>&1 | tee "${VAL_F1}/compare.log"
date

echo "[chain] 8/8 fine-tune on the in-sample label"
python scripts/flow_matching/finetune_advantage.py \
    --pretrained "$FM_CKPT" --rollout "${STEM}.h5" --labels "${VAL_F1}/advantage.npz" \
    --out "$FT_F1" --epochs "$FT_EPOCHS" --drop_tile_holdout \
    2>&1 | tee "${FT_F1}/finetune.log"
for LABEL in positive negative; do
  python scripts/flow_matching/export_fm_onnx.py --checkpoint "${FT_F1}/best.pt" \
      --advantage_label "$LABEL" 2>&1 | grep -E "^\[INFO\] (Wrote|batch)"
done

echo "[chain] DONE."
date
ls -la "$FT" "$FT_F1"
echo
echo "NEXT (by hand, in the morning):"
echo "  1. read ${VAL_F1}/compare.log -- do the two labellings even differ?"
echo "  2. the negative-conditioning control on ${FT}, before any sweep."
echo "  3. the paired eval against the base generator."
