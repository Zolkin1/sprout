#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# record -> augment -> train, the whole generator pipeline in one place.
#
# Reproduces how the SHIPPED generator (2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff) was
# actually built, read off its own params/data.yaml rather than from memory:
#
#   1. record_data.py   on the tracker, windowed, stochastic OFF, clip phase UNIFORM
#   2. augment_registration.py  displaces the robot so depth stops being ignorable
#   3. flow_matching/train.py --data <recorded>,<augmented>   -- BOTH, comma-separated
#
# Step 3 is the one that is easy to get wrong: the generator trains on the raw recording AND
# the augmented one concatenated, not on the augmented one alone.
#
# Usage:
#   scripts/flow_matching/run_fm_pipeline.sh <tracker_run_name> [<clip_library>]
#
# RUN_NAME is the run FOLDER NAME under logs/rsl_rl/<experiment>, e.g.
# 2026-09-11_11-45-14_paper_v6g_v1 -- that is what --load_run takes ("name of the run folder"),
# not a path. Everything else is a knob below.
set -euo pipefail

RUN_NAME=${1:?usage: run_fm_pipeline.sh <tracker_run_name> [<clip_library>]}
CLIP=${2:-clips/g1_mb_v6g_decor.h5}
EXPERIMENT=${EXPERIMENT:-g1}
RUN_DIR="logs/rsl_rl/$EXPERIMENT/$RUN_NAME"

# ---------------------------------------------------------------- knobs
# 1024 envs, NOT more. Window coverage is what this trades: windows snap every ~12 control
# steps, so a full clip yields ~77 of them, and an env only reaches a clip's late phase if its
# share of the budget (num_windows / num_envs) comfortably exceeds that. Measured on the
# shipped recording: 200k windows at 256 envs = 781/env ~ 10 episodes, covering 6065 distinct
# clips with clip_step spanning 12..924. At 1024 envs that is still 195/env (2.5+ full
# episodes) and coverage holds; at 4096 it would be 49/env -- LESS than one episode, so the
# run would end mid-first-episode for nearly every env and every window would come from the
# early phase of its clip. That bias is invisible in the output, which is why the number is
# pinned here with its reason.
NUM_ENVS=${NUM_ENVS:-1024}
NUM_WINDOWS=${NUM_WINDOWS:-200000}
NUM_AUGMENTS=${NUM_AUGMENTS:-3}
TASK=${TASK:-G1-teacher-clf-terrain-depth}
EXTRA_TRAIN=${EXTRA_TRAIN:---wandb}

REC_DIR="$RUN_DIR/recorded_data"
STAMP=$(date +%Y%m%d_%H%M%S)
RECORDED="$REC_DIR/recorded_${STAMP}.h5"
AUGMENTED="$REC_DIR/augmented_${STAMP}.h5"

PY="./env_isaaclab/bin/python"
export PYTHONPATH="source/robot_rl:source/motion_gen${PYTHONPATH:+:$PYTHONPATH}"

say() { echo "[fm_pipeline $(date +%H:%M)] $*"; }

[ -d "$RUN_DIR" ] || { say "REFUSING: no tracker run dir: $RUN_DIR"; exit 1; }
[ -f "$CLIP" ]    || { say "REFUSING: no clip library: $CLIP"; exit 1; }
ls "$RUN_DIR"/model_*.pt >/dev/null 2>&1 || { say "REFUSING: $RUN_DIR has no model_*.pt"; exit 1; }
mkdir -p "$REC_DIR"

say "tracker $RUN_NAME  ($RUN_DIR)"
say "clip    $CLIP"
say "envs $NUM_ENVS  windows $NUM_WINDOWS  augments $NUM_AUGMENTS"

# ---------------------------------------------------------------- 1. record
# --start_step 0 keeps the recorded window distribution UNIFORM in clip phase; training's
# sampled start puts 18% of windows in the eighth phase decile against 1% in the first.
# --clip_draw cycle makes clip coverage exactly min(episodes, clips) instead of 63%.
# The fm_depth observation group is installed by record_data itself, UNCONDITIONALLY on a
# depth task -- it has to be, now that the policy reads the near camera alone: without it the
# recorder would scope to the policy's one camera and stamp a one-camera spec into the
# recording, and every artifact downstream would agree with every other about a half-blind
# generator. The sanity check after this stage is the guard that would catch it anyway.
say "=== 1/3 record -> $RECORDED"
$PY scripts/rsl_rl/record_data.py \
    --task "$TASK" \
    --load_run "$RUN_NAME" \
    --experiment_name "$EXPERIMENT" \
    --clip "$CLIP" \
    --num_windows "$NUM_WINDOWS" \
    --num_envs "$NUM_ENVS" \
    --start_step 0 \
    --clip_draw cycle \
    --headless

# record_data timestamps its own output; adopt whatever it just wrote.
RECORDED=$(ls -t "$REC_DIR"/recorded_*.h5 | head -1)
say "recorded: $RECORDED"

# THE check worth doing by hand once: two cameras, not one.
$PY - "$RECORDED" <<'PY'
import h5py, json, sys
f = h5py.File(sys.argv[1], "r")
spec = f.attrs.get("depth_camera_spec") or f.attrs.get("extero_spec") or "[]"
n = len(json.loads(spec)) if isinstance(spec, str) else len(spec)
print(f"[fm_pipeline] recorded cameras: {n}")
if n < 2:
    raise SystemExit(
        f"REFUSING: the recording carries {n} camera(s). The generator reads BOTH; a "
        "one-camera recording trains a half-blind generator that every downstream artifact "
        "will agree with. Check install_fm_depth_group ran."
    )
f.close()
PY

# ---------------------------------------------------------------- 2. augment
# Every recorded window has the robot sitting exactly on its clip, so depth and the correct
# answer are perfectly correlated and depth can be ignored outright -- measured: displace the
# robot 0.10/0.20 m downrange and the predicted stair segment slides 0.098/0.168 m with it,
# where the anchor mask requires 0.000. This breaks that correlation offline: no physics, no
# controller, Isaac as a renderer only.
say "=== 2/3 augment -> $AUGMENTED"
$PY scripts/rsl_rl/augment_registration.py \
    --task "$TASK" \
    --data "$RECORDED" \
    --out "$AUGMENTED" \
    --num_augments "$NUM_AUGMENTS" \
    --headless

# ---------------------------------------------------------------- 3. train
# BOTH files. The exteroception constants (extero_xyz_offset, the envelope, the nominal ground
# plane) are DERIVED here from the recording's own camera contract -- there is no default to
# pass and none should be passed, or a stale declared offset rides along instead.
say "=== 3/3 train on recorded + augmented"
$PY scripts/flow_matching/train.py \
    --data "$RECORDED,$AUGMENTED" \
    $EXTRA_TRAIN

say "done"
say "  recorded : $RECORDED"
say "  augmented: $AUGMENTED"
