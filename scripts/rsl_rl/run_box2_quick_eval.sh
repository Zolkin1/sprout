#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# box2_quick -- the ~30 minute two-box comparator.
#
# 2 approach distances x 3 speeds instead of the full 5 x 6, with every episode concentrated
# on the two tiles rather than spread over five, so a short run still reaches n ~ 700 per
# cell. The tile GEOMETRY is identical to the full grid's, so a box2_quick cell and a
# full-grid cell at the same (distance, speed) are the same course and are directly
# comparable -- which is what lets a quick run be scored against the full-grid parent
# baseline instead of needing its own control.
#
#   distances  1.50 m, 3.00 m   -- the two where the first fine-tune moved most
#   speeds     0.83, 1.17, 1.50 -- around the parent's peak, i.e. the highest base rates,
#                                  which is where a fixed episode budget has the most power
#
# WHAT IT MEASURES. Two things, and the SECOND is the primary one:
#   reached_goal        -- the by-product, and the weaker signal (see the resolution note)
#   the terrain grade   -- one row per re-plan in the episode file's `chunks` group, holding
#                          the same raw terms the offline sidecar writes and the critic is
#                          fitted on. `--terrain_reward` costs ~60 bytes a re-plan against
#                          ~30 kB for the window it grades, so it composes with --no_windows:
#                          nothing else is recorded and the run stays this fast.
#
# THE STOP RULE IS ROUND-BASED, AND THIS IS NOT A DETAIL. `--num_episodes` stops when a TOTAL
# count has finished, which fires as soon as the FASTEST episodes complete and cuts every slow
# episode still in flight -- so the kept sample is a policy's fast-failing tail, and the bias
# runs against the SLOWER arm. Measured on exactly this eval: the parent at 1024 envs came out
# 20% censored with its uncensored episodes reaching the full 1999-step budget, while an AWR
# arm at 4096 envs came out 74.5% censored with a hard ceiling of 869 steps. Scored that way
# the arm read 0.0005 against the parent's 0.1960 -- a policy that looks destroyed and is not.
# `--episodes_per_env` keeps the collected set a complete design over (env, round), whose
# membership cannot depend on duration. NENV therefore also has to MATCH across arms: it sets
# how many episodes are in flight at the stop, so it decides the censored fraction.
#
# WHAT IT CAN AND CANNOT RESOLVE. At ~700 episodes/cell the per-cell 95% interval is about
# +-0.027 and the pooled one about +-0.015. So this is a TRIAGE instrument: it separates
# "clearly better/worse" from "no big effect". It cannot resolve the ~+0.008 that the full
# grid measured for filtered BC, and a null here is not evidence of no effect -- run the full
# grid for that.
#
# Usage:
#   scripts/rsl_rl/run_box2_quick_eval.sh <fm_checkpoint> <out_tag> [episodes_per_speed]
#
# Example:
#   scripts/rsl_rl/run_box2_quick_eval.sh logs/box2/awr_b1.0_w0/best.pt awr_b1.0
set -u
FM=${1:?usage: run_box2_quick_eval.sh <fm_checkpoint> <out_tag> [episodes_per_speed]}
TAG=${2:?usage: run_box2_quick_eval.sh <fm_checkpoint> <out_tag> [episodes_per_speed]}
NEPS=${3:-1400}

#: Envs, overridable because the right number is a property of the BOX not the eval: 384 fits
#: a 12 GB dev card, 8192 suits an H100's 96 GB and is ~21x the throughput. The terrain grade
#: tiles its SDF pass to a fixed element budget (`motion_gen.terrain_consistency`), so its
#: working set is ~7.4 GB at either -- the env itself is what scales.
NENV=${NENV:-384}
#: Rounds per env, derived so the total is at least NEPS. Ceil, so the run never collects
#: fewer than asked; the surplus is a COMPLETE round, which is what keeps the design balanced.
#: At NENV >= NEPS this is 1 round, i.e. one episode per env -- already a complete design.
ROUNDS=$(( (NEPS + NENV - 1) / NENV ))

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH=source/robot_rl:source/motion_gen
TRK=2026-08-20_16-36-10_mode11_decor_plan_v1
#: The comparator library. Overridable because v1 and v2 are NOT comparable courses -- v2
#: puts the goal 3 m further out with the boxes unmoved, so a v2 rate is lower than a v1 rate
#: on the same policy for a reason that is not the policy (see make_box2_terrain --variant).
#: An arm trained on v2 data therefore needs a v2 BASELINE, which means both arms of a
#: comparison must name the same library here.
CLIP=${CLIP:-clips/box2_quick_train.h5}
GR=logs/box2/goal_regions.json
OUT=logs/box2/quick/$TAG
#: The distances box2_quick's tiles were built at, in TILE ORDER. The report needs them
#: because `terrain_id` is a tile index: 3.00 m is tile 1 here and tile 3 in the full grid.
DISTS="1.5 3.0"

[ -f "$FM" ] || { echo "no such checkpoint: $FM" >&2; exit 2; }
[ -f "$CLIP" ] || { echo "missing $CLIP -- build it with: python scripts/rsl_rl/make_box2_terrain.py \
--out_dir clips --prefix $(basename "$CLIP" _train.h5) --train_distances $DISTS \
--variant <v1|v2> --like clips/g1_mb_v5_body_mode11_decor.h5" >&2; exit 2; }

mkdir -p "$OUT"
echo "[quick] $TAG  fm=$FM  $NENV envs x $ROUNDS rounds = $((NENV * ROUNDS)) episodes/speed"
echo "[quick]   (round-based stop; an arm compared against this one must match BOTH numbers)"
echo "[quick]   -> $OUT"
for spd in 0.83 1.17 1.50; do
  d="$OUT/train_v${spd}"
  if [ -f "$d/rollout_${TRK}.episodes.h5" ]; then echo "[quick] SKIP $spd (exists)"; continue; fi
  echo "[quick] speed $spd  $(date +%H:%M)"
  ./env_isaaclab/bin/python scripts/rsl_rl/record_rollouts.py \
    --task=G1-box2-finetune --load_run "$TRK" --fm_checkpoint "$FM" \
    --clip "$CLIP" --clip_device cpu --goal_regions "$GR" \
    --num_envs $NENV --episodes_per_env "$ROUNDS" --no_windows --terrain_reward \
    --output_dir "$d" \
    env.commands.traj_ref.freeform_speed_min=$spd \
    env.commands.traj_ref.freeform_speed_max=$spd \
    env.episode_length_s=40.0 \
    > "$d.log" 2>&1 || { echo "[quick] speed $spd FAILED, see $d.log" >&2; exit 1; }
done
echo "[quick] done $(date +%H:%M)"
./env_isaaclab/bin/python scripts/rsl_rl/box2_ic_report.py "$OUT" --distances $DISTS
