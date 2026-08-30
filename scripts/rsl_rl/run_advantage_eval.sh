#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Isaac eval of an advantage-conditioned fine-tune, against the generator it came from.
#
# FOUR arms, three of which are generators, all paired on clip id against ONE shared clip
# arm (the generator is a cell DIMENSION in this harness, so the ceiling is run once):
#
#   clip                          the tracker's own ceiling -- did anything move that should not have
#   base   <pretrained>           the baseline, unconditioned
#   ac_pos <ft>/best_positive.pt  THE DELIVERABLE
#   ac_neg <ft>/best_negative.pt  THE CONTROL, and it is not optional
#
# `ac_neg` is what makes the rest interpretable. If it performs the same as `ac_pos` the
# model ignored the label, and `ac_pos` beating `base` would then be measuring "six more
# epochs of training on rollout data", not the conditioning. Read it first.
#
# The arm each checkpoint serves rides IN the checkpoint (`advantage_serve_label`), so the
# harness needs no new flag -- which matters here, since a flag added to the parser is not
# forwarded to the cell subprocess unless `stage_grid` restates it, and reverts to its
# argparse default in the child silently. A mode that rides in an artifact cannot be dropped.
#
# Usage:  scripts/rsl_rl/run_advantage_eval.sh <tag> [<clips_per_tile>]
set -euo pipefail

TAG="${1:?usage: run_advantage_eval.sh <tag> [clips_per_tile]}"
CLIPS_PER_TILE="${2:-3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TEACHER="2026-08-20_16-36-10_mode11_decor_plan_v1"
TRACKER_CKPT="logs/rsl_rl/g1/${TEACHER}/model_19999.pt"
FM_CKPT="logs/flow_matching/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff/best.pt"
LIB="clips/g1_mb_v5_body_mode11_decor.h5"
SKIP="jump_off,distractor,stairs_dn_pure,stairs_up_pure"

FT="logs/flow_matching/$(basename "$(dirname "$FM_CKPT")")_ac_${TAG}"
OUT="logs/terrain_eval/advantage_${TAG}"

export PYTHONPATH="source/robot_rl:source/motion_gen"
# shellcheck disable=SC1091
source env_isaaclab/bin/activate

for f in "${FT}/best_positive.pt" "${FT}/best_negative.pt"; do
  [ -f "$f" ] || { echo "[eval] MISSING $f -- did the fine-tune finish?"; exit 1; }
done

mkdir -p "$OUT"
echo "[eval] out=$OUT  finetune=$FT  clips/tile=$CLIPS_PER_TILE"
date

# Which tiles the fine-tune was NOT trained on, recorded so the morning analysis can split
# seen from unseen. The fine-tune ran with --drop_tile_holdout, so those tiles are clean --
# but this eval picks its own tiles and does not know about that split, so the honest reading
# needs the two lists side by side rather than a single headline.
python - "logs/value/${TAG}/predictions.npz" "${OUT}/finetune_holdout_tiles.json" <<'PY'
import json, sys
import numpy as np
d = np.load(sys.argv[1])
held = sorted({int(t) for t in d["tile"][d["val_tile"].astype(bool)]})
seen = sorted({int(t) for t in d["tile"][~d["val_tile"].astype(bool)]})
json.dump({"finetune_trained_tiles": seen, "finetune_holdout_tiles": held}, open(sys.argv[2], "w"), indent=2)
print(f"[eval] fine-tune saw {len(seen)} tiles, held out {len(held)}: {held}")
PY

# --------------------------------------------------------------------------------- plan
if [ ! -f "${OUT}/plan.json" ]; then
  python scripts/rsl_rl/eval/evaluate.py --stage plan \
      --lib "$LIB" --load_run "$TEACHER" --checkpoint "$TRACKER_CKPT" \
      --fm_checkpoint "base=${FM_CKPT}" \
      --fm_checkpoint "ac_pos=${FT}/best_positive.pt" \
      --fm_checkpoint "ac_neg=${FT}/best_negative.pt" \
      --exclude_families "$SKIP" \
      --clips_per_tile "$CLIPS_PER_TILE" --flat_tiles 4 --flat_clips 30 \
      --dr_presets train --num_envs 128 \
      --goal_regions "logs/rollouts/${TAG}/goal_regions.json" \
      --out "$OUT" 2>&1 | tee "${OUT}/plan.log"
fi

# --------------------------------------------------------------------------------- grid
# `--stage grid` is a SEPARATE argparse invocation from `--stage plan` -- it does not re-read
# flags off plan.json, only off its own command line. Anything set at plan time that stage_grid
# also reads (see its own forwarding list) has to be restated here or it silently reverts to
# argparse's default in every cell subprocess. Bit once already in this exact run: an eval with
# no --goal_regions here ran with reached_goal permanently 0 for its whole duration, and
# report.py refused the table only at the very end.
python scripts/rsl_rl/eval/evaluate.py --stage grid --out "$OUT" \
    --num_envs 200 --goal_regions "logs/rollouts/${TAG}/goal_regions.json" \
    2>&1 | tee "${OUT}/grid.log"
date

# ------------------------------------------------------------------------------- report
python scripts/rsl_rl/eval/report.py "$OUT" 2>&1 | tee "${OUT}/report.log"

echo "[eval] DONE -> $OUT"
echo
echo "READ IN THIS ORDER:"
echo "  1. ac_neg vs ac_pos. Same => the model ignored the label; nothing else here means anything."
echo "  2. ac_pos vs base, paired on clip id. This is the result."
echo "  3. clip. It must not have moved -- the tracker is frozen, so if it did, something else did too."
echo "  4. the per-family table against ${OUT}/finetune_holdout_tiles.json."
