#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Fit a goal-region set for ONE clip library, the way the shipped set was fitted.
#
# Usage:
#   scripts/rsl_rl/calibrate_goal_regions.sh <clip_library.h5> <tracker_run_name> [<out.json>]
#
# EVERY LIBRARY NEEDS ITS OWN. The tolerances are a measurement of where the CLIP ARM finishes,
# so they answer "did the policy get where its own reference goes" -- and a set fitted on one
# library carries no guarantee of being REACHABLE on another, because clip lengths, speeds and
# terrain geometry all move between versions. Reusing `goal_regions_anchor_v3.json` (fitted
# pre-v6, from a different tracker on the older library) would score v6 clips against endpoints
# its own reference may never reach, and an unreachable goal makes the rate meaningless rather
# than merely conservative.
#
# Fitting per library does NOT cost comparability: each set is sized to ITS library's clip arm
# at the same quantile, so "fraction of the way the tracker's own reference goes" is the same
# question on both, which is the question an old-vs-new comparison is asking.
#
# THE CIRCULARITY, and how it is broken (see build_goal_regions.py): nothing records where an
# episode ended unless a goal region is attached, and the region is what we are fitting. So
# pass 1 runs the clip arm against a BOOTSTRAP spec -- infinite tolerances, which record the
# raw terminal offsets exactly as any spec would while making `reached_goal` trivially 1. That
# is flagged in the spec, copied into the table's meta, and refused by report.py, so a
# bootstrap table can never be read as a 100%-success result.
set -euo pipefail

LIB=${1:?usage: calibrate_goal_regions.sh <clip_library.h5> <tracker_run_name> [<out.json>]}
RUN_NAME=${2:?usage: calibrate_goal_regions.sh <clip_library.h5> <tracker_run_name> [<out.json>]}
OUT=${3:-$(dirname "$LIB")/$(basename "$LIB" .h5)_goal_regions.json}

# ---------------------------------------------------------------- knobs
# Sized against the shipped set's own provenance: it fitted from 1931 surviving clip-arm
# episodes. The fit SKIPS flat, the pure families and jump_off, so what matters is the episode
# count on the four fitted ones -- 8 tiles x 4 clips x R=8 = 256 each, ~1000 after falls.
# Spend budget on TILES over robots: the eval's own measurement is that 2x the robots buys
# nothing while 2x the tiles at half the robots is free and 21% tighter.
# 8, because that is what the SMALLEST family has: v6 authors 8 pure-stair tiles per
# direction against 33 for the transitions, and select_clips refuses a request it cannot
# fill rather than quietly returning fewer -- so a number above the smallest family's tile
# count fails the whole plan, not just that family. Raise it only alongside the recipe.
TILES_PER_FAMILY=${TILES_PER_FAMILY:-8}
CLIPS_PER_TILE=${CLIPS_PER_TILE:-4}
FLAT_TILES=${FLAT_TILES:-6}
ROBOTS=${ROBOTS:-8}
NUM_ENVS=${NUM_ENVS:-512}
GPU=${GPU:-0}
WORK=${WORK:-$(mktemp -d /tmp/goalcal_XXXX)}

PY="./env_isaaclab/bin/python"
export PYTHONPATH="source/robot_rl:source/motion_gen${PYTHONPATH:+:$PYTHONPATH}"
say() { echo "[goalcal $(date +%H:%M)] $*"; }

[ -f "$LIB" ] || { say "REFUSING: no clip library: $LIB"; exit 1; }
say "library  $LIB"
say "tracker  $RUN_NAME"
say "work     $WORK"
say "out      $OUT"

# ---------------------------------------------------------------- 1. bootstrap spec
say "=== 1/3 bootstrap spec"
$PY scripts/rsl_rl/build_goal_regions.py --write_bootstrap "$WORK/goal_regions.bootstrap.json"

# ---------------------------------------------------------------- 2. the CLIP arm
# clip ONLY: the region is sized to what the tracker following a pre-made reference achieves,
# which is the best this stack does at the task. A generator arm that misses a region sized
# that way missed for a reason that is the generator's.
say "=== 2/3 clip arm against the bootstrap spec"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/rsl_rl/eval/evaluate.py \
    --lib "$LIB" \
    --load_run "$RUN_NAME" \
    --reference_modes clip \
    --dr_presets train \
    --tiles_per_family "$TILES_PER_FAMILY" \
    --clips_per_tile "$CLIPS_PER_TILE" \
    --flat_tiles "$FLAT_TILES" \
    --robots_per_clip "clip/train=$ROBOTS" \
    --goal_regions "$WORK/goal_regions.bootstrap.json" \
    --num_envs "$NUM_ENVS" \
    --out "$WORK/boot"

# ---------------------------------------------------------------- 3. fit
# The skip list is build_goal_regions.py's own default, which carries BOTH spellings of the
# pure families -- the name changed between library versions and a stale spelling matches
# nothing, which fits the family rather than skipping it, silently.
say "=== 3/3 fit -> $OUT"
$PY scripts/rsl_rl/build_goal_regions.py "$WORK/boot" --out "$OUT"

say "done. Scored runs on this library want --goal_regions $OUT"
say "provenance (fitted_from, n_surviving_episodes, per-family) is inside the file."
