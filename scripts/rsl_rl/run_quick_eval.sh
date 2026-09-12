#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# ONE eval entry point for every fine-tune terrain: box2, up-into-down, varied, in-distribution.
#
# Was `run_box2_quick_eval.sh`, which defaulted to a box2 library, rebuilt one from
# box2-specific flags, and had to be copied and edited for every other course -- so the four
# terrains were evaluated by four slightly different procedures and the differences were
# invisible in the output. `TERRAIN=` selects a BUNDLE instead; every field of it is still
# individually overridable, so "consistent by default" costs nothing at the CLI.
#
# WHAT IS THE SAME ON EVERY TERRAIN, and each one is a thing that was not:
#   the spawn rule       SPAWN_FROM=ground -- the INITIAL area only. Training declares raised
#                        areas (a box top, the platform between two flights) because a
#                        complete traverse is rare enough that the behaviour policy never
#                        produced one; scoring from those would report a fraction of the
#                        course as a traverse. See `mdp.events.resets.SPAWN_FROM`.
#   the success rule     arrival BEFORE the first fall (`rollout_outcomes.terminal_reward`),
#                        applied by the reducer, printed in its header.
#   the reduction        first-per-env. A round-based stop leaves the fast envs with extra
#                        SHORT episodes, so pooling tilts the sample toward fast failures.
#   the speeds           the same three rungs on every course, and REFUSED if they fall
#                        outside the library's own declared band.
#   NENV                 one number, because it sets the censored fraction and therefore has
#                        to match across the arms of a comparison.
#   the budget           episode_length_s=40, non-binding on the slow x far cells.
#
# THE STOP RULE IS ROUND-BASED, AND THIS IS NOT A DETAIL. `--num_episodes` stops when a TOTAL
# count has finished, which fires as soon as the FASTEST episodes complete and cuts every slow
# episode still in flight -- so the kept sample is a policy's fast-failing tail, and the bias
# runs against the SLOWER arm. Measured on exactly this eval: the parent at 1024 envs came out
# 20% censored with its uncensored episodes reaching the full 1999-step budget, while an AWR
# arm at 4096 envs came out 74.5% censored with a hard ceiling of 869 steps. Scored that way
# the arm read 0.0005 against the parent's 0.1960 -- a policy that looks destroyed and is not.
#
# WHAT IT CAN AND CANNOT RESOLVE. At ~700 episodes/cell the per-cell 95% interval is about
# +-0.027 and the pooled one about +-0.015. So this is a TRIAGE instrument: it separates
# "clearly better/worse" from "no big effect". It cannot resolve the ~+0.008 that the full
# grid measured for filtered BC, and a null here is not evidence of no effect.
#
# Usage:
#   scripts/rsl_rl/run_quick_eval.sh <fm_checkpoint> <out_tag> [episodes_per_speed]
#
# Examples:
#   TERRAIN=box2   scripts/rsl_rl/run_quick_eval.sh logs/box2/awr/best.pt awr_b1.0
#   TERRAIN=updown GPU=1 scripts/rsl_rl/run_quick_eval.sh logs/box2/vm30/best.pt vm30
#   TERRAIN=indist SPEEDS="1.17" scripts/rsl_rl/run_quick_eval.sh <ckpt> indist_check
set -u
FM=${1:?usage: run_quick_eval.sh <fm_checkpoint> <out_tag> [episodes_per_speed]}
TAG=${2:?usage: run_quick_eval.sh <fm_checkpoint> <out_tag> [episodes_per_speed]}
NEPS=${3:-1400}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=source/robot_rl:source/motion_gen
PY=./env_isaaclab/bin/python

#: The course. Everything below is a DEFAULT for it, not a constraint.
TERRAIN=${TERRAIN:-box2}
case "$TERRAIN" in
  box2)
    #: The reproducible 2-tile grid. `box2_v2_quick_train.h5` was the frozen one and is
    #: retired: it predates spawn areas, so today's generator cannot emit it -- and it does
    #: not need to, because a ground-only draw on THIS library is bit-identical to that
    #: library's synthesised single start (`test_eval_spawn_ground_only.py` pins it), so
    #: every box2 rate ever measured stays comparable.
    DEF_CLIP=clips/box2_v2spawn_quick_train.h5
    DEF_GR=logs/box2/goal_regions.json
    DEF_TOL=0.5
    DEF_ROOT=logs/box2/quick
    DEF_BUILD="python scripts/rsl_rl/make_box2_terrain.py --out_dir clips --prefix box2_v2spawn_quick --variant v2 --train_distances 1.5 3.0 --like clips/g1_mb_v5_body_mode11_decor.h5";;
  updown)
    DEF_CLIP=clips/updown_p020_r030_train.h5
    DEF_GR=logs/updown/goal_regions.json
    DEF_TOL=0.5
    DEF_ROOT=logs/updown/quick
    DEF_BUILD="python scripts/rsl_rl/make_step_courses_terrain.py --course updown --platform_depth 2.0 --runout 3.0 --out_dir clips --like clips/g1_mb_v5_body_mode11_decor.h5";;
  varied)
    DEF_CLIP=clips/varied60_holdout.h5
    DEF_GR=logs/varied/goal_regions.json
    DEF_TOL=0.5
    DEF_ROOT=logs/varied/quick
    DEF_BUILD="python scripts/rsl_rl/make_varied_terrain.py --out_dir clips --prefix varied60 --goals logs/varied --like clips/g1_mb_v5_body_mode11_decor.h5";;
  indist)
    #: The IN-DISTRIBUTION control: 75 real clip-library tiles, so this is what an OOD
    #: fine-tune is asked not to have cost. `height_tol_m` matters here and only here --
    #: stairs_up and jump_on put the goal ABOVE the spawn, and the termination tests
    #: `|root_z - goal_z|`, so a 0 tolerance can never fire on them.
    DEF_CLIP=clips/indist_train.h5
    DEF_GR=logs/indist/goal_regions.json
    DEF_TOL=0.5
    DEF_ROOT=logs/indist/quick
    DEF_BUILD="python scripts/rsl_rl/make_indist_routes.py --source clips/g1_mb_v5_body_mode11_decor.h5 --out_dir clips --prefix indist";;
  *)
    echo "REFUSING: TERRAIN must be box2, updown, varied or indist; got '$TERRAIN'." >&2
    exit 2;;
esac

CLIP=${CLIP:-$DEF_CLIP}
GR=${GR:-$DEF_GR}
OUT=${OUT:-$DEF_ROOT/$TAG}
TRK=${TRK:-2026-08-20_16-36-10_mode11_decor_plan_v1}
TASK=${TASK:-G1-box2-finetune}

#: Envs, overridable because the right number is a property of the BOX not the eval: 384 fits
#: a 12 GB dev card, 8192 suits an H100's 96 GB and is ~21x the throughput. The terrain grade
#: tiles its SDF pass to a fixed element budget (`motion_gen.terrain_consistency`), so its
#: working set is ~7.4 GB at either -- the env itself is what scales. IT MUST MATCH ACROSS THE
#: ARMS OF A COMPARISON: it sets how many episodes are in flight at the stop, so it decides
#: the censored fraction.
NENV=${NENV:-384}
#: Rounds per env, derived so the total is at least NEPS. Ceil, so the run never collects
#: fewer than asked; the surplus is a COMPLETE round, which keeps the design balanced.
ROUNDS=$(( (NEPS + NENV - 1) / NENV ))

#: WHICH SPAWN AREAS. `ground` is the eval rule on every terrain; `all` reproduces the
#: TRAINING distribution and is not a success rate for the course.
#: What fills the generator's history rows. MUST match the arm the checkpoint was TRAINED
#: for -- the fine-tune stamps `history_source` from its recording and the runtimes refuse a
#: mismatched pair, because the two kinds of history row are identical in WIDTH and a
#: mismatch merely steers wrong.
HISTORY_FROM=${HISTORY_FROM:-generator}
case "$HISTORY_FROM" in generator|measured) ;; *)
  echo "REFUSING: HISTORY_FROM must be generator or measured; got '$HISTORY_FROM'." >&2
  exit 2;; esac

SPAWN_FROM=${SPAWN_FROM:-ground}
case "$SPAWN_FROM" in ground|all) ;; *)
  echo "REFUSING: SPAWN_FROM must be ground or all; got '$SPAWN_FROM'." >&2; exit 2;; esac

#: Goal height tolerance (m), applied to |root_z - goal_z|. COERCED TO A FLOAT because
#: IsaacLab type-checks hydra overrides: `height_tol_m=0` is refused where `0.0` is accepted,
#: and it fails at hydra composition before the GPU is touched -- which cost two runs. And 0
#: means OPPOSITE things at the two ends: it DISABLES the test in the termination and is an
#: EXACT-EQUALITY test in `goal_region.in_goal`, so a spec of 0 scored an entire arm at 0.0000
#: over 8400 episodes while looking like a broken policy.
HEIGHT_TOL=$(awk -v v="${HEIGHT_TOL:-$DEF_TOL}" 'BEGIN{printf "%.6f", v+0}')
# awk's `v+0` turns ANYTHING un-numeric into 0 -- and 0 is the value this comment says scored
# an arm at 0.0000 over 8400 episodes. So the coercion is checked, not trusted: `HEIGHT_TOL=p5`
# would otherwise silently disable the height test on the one terrain where it decides arrival.
awk -v v="${HEIGHT_TOL:-$DEF_TOL}" 'BEGIN{exit !(v+0 > 0 && v+0 == v+0)}' 2>/dev/null \
  && [ "$(awk -v v="$HEIGHT_TOL" 'BEGIN{exit !(v>0)}'; echo $?)" = "0" ] || {
  echo "REFUSING: HEIGHT_TOL='${HEIGHT_TOL:-$DEF_TOL}' is not a positive number (it coerced" >&2
  echo "  to $HEIGHT_TOL). 0 does not mean 'off' here -- goal_region.in_goal reads it as an" >&2
  echo "  EXACT-equality test, which is never satisfiable." >&2
  exit 2; }

#: The commanded speeds, one recording each, pinned per cell. THE SAME THREE ON EVERY COURSE,
#: which is what makes a box2 cell and an up-into-down cell the same command -- and they are
#: checked against the library's OWN declared band below rather than assumed to be inside it.
SPEEDS=${SPEEDS:-"0.83 1.17 1.50"}

[ -f "$FM" ] || { echo "no such checkpoint: $FM" >&2; exit 2; }
#: The library is REQUIRED, never rebuilt here. The old script rebuilt it from box2-specific
#: flags, which silently produced the wrong course for any other terrain; and a build inside
#: an eval can overwrite the very grid an earlier arm was scored on.
[ -f "$CLIP" ] || { echo "missing $CLIP -- build it with:
  $DEF_BUILD" >&2; exit 2; }
[ -f "$GR" ] || { echo "missing goal spec $GR -- its builder writes it (see $DEF_BUILD)" >&2; exit 2; }

#: What this library's tiles vary, in TILE ORDER, read OFF THE LIBRARY. The report needs it
#: because `terrain_id` is an index into it: 3.00 m is tile 1 on the quick grid and tile 3 on
#: the full one, so a hand-passed list that did not match the file silently relabelled every
#: cell and paired it against the wrong control.
DISTS=${DISTS:-$($PY scripts/rsl_rl/terrain_axis.py --clip "$CLIP" --values)}
AXIS=$($PY scripts/rsl_rl/terrain_axis.py --clip "$CLIP" --name)

#: Every commanded speed must be inside the library's own band, or the cell is out of the
#: distribution the course declares and its rate is not comparable to a training run's.
$PY scripts/rsl_rl/terrain_axis.py --clip "$CLIP" --check_speeds $SPEEDS || exit 2

mkdir -p "$OUT"
echo "[eval] $TAG  terrain=$TERRAIN  fm=$FM"
echo "[eval]   clip   $CLIP"
echo "[eval]   goals  $GR  height_tol=$HEIGHT_TOL"
echo "[eval]   hist   $HISTORY_FROM"
echo "[eval]   spawn  $SPAWN_FROM  (the eval rule is 'ground' -- the INITIAL area only)"
echo "[eval]   axis   $AXIS = [$DISTS]"
echo "[eval]   speeds $SPEEDS m/s"
echo "[eval]   $NENV envs x $ROUNDS rounds = $((NENV * ROUNDS)) episodes/speed"
echo "[eval]   (round-based stop; an arm compared against this one must match BOTH numbers)"
echo "[eval]   -> $OUT"

for spd in $SPEEDS; do
  d="$OUT/train_v${spd}"
  if [ -f "$d/rollout_${TRK}.episodes.h5" ]; then echo "[eval] SKIP $spd (exists)"; continue; fi
  echo "[eval] speed $spd  $(date +%H:%M)"
  # `--history_from` is a CLI FLAG, not a hydra override: record_rollouts has its own flag
  # for it and applies it after composition, so an override is silently discarded.
  $PY scripts/rsl_rl/record_rollouts.py \
    --task="$TASK" --load_run "$TRK" --fm_checkpoint "$FM" \
    --clip "$CLIP" --clip_device cpu --goal_regions "$GR" \
    --num_envs $NENV --episodes_per_env "$ROUNDS" --no_windows --terrain_reward \
    --history_from "$HISTORY_FROM" \
    --output_dir "$d" \
    env.commands.traj_ref.freeform_speed_min=$spd \
    env.commands.traj_ref.freeform_speed_max=$spd \
    env.events.reset_on_section.params.spawn_from=$SPAWN_FROM \
    env.terminations.goal_reached.params.height_tol_m=$HEIGHT_TOL \
    env.episode_length_s=40.0 \
    > "$d.log" 2>&1 || { echo "[eval] speed $spd FAILED, see $d.log" >&2; exit 1; }
done
echo "[eval] done $(date +%H:%M)"
# `--distances` only when the course HAS a scalar axis. `varied` and `indist` have none --
# their tiles vary wholesale geometry and are 75 real library tiles -- so `terrain_axis.py`
# returns an empty list by design, and passing the flag with no values made argparse refuse
# AFTER the entire GPU eval had finished.
if [ -n "$DISTS" ]; then
  $PY scripts/rsl_rl/box2_ic_report.py "$OUT" --distances $DISTS
else
  echo "[eval] $AXIS: no scalar axis on this course; cells are labelled by tile index"
  $PY scripts/rsl_rl/box2_ic_report.py "$OUT"
fi

# PER TILE TYPE, on any library holding more than one family. The report above reduces per
# TILE, which on a multi-terrain library is 70 rows nobody reads, and its marginals pool across
# families -- a pooled rate over a mix moves when the MIX moves, with no policy change at all.
# Skipped rather than failed on a single-family library: there the two reports are the same
# number, so it would be a second copy of the row above.
NFAM=$($PY - "$CLIP" <<'PYEOF'
import json, sys
import h5py
try:
    with h5py.File(sys.argv[1], "r") as f:
        raw = f.attrs.get("terrain_manifest")
    man = json.loads(raw if isinstance(raw, str) else raw.tobytes().decode()) if raw is not None else []
    print(len({str(t.get("family") or "?") for t in man}))
except Exception:
    print(0)
PYEOF
)
if [ "${NFAM:-0}" -gt 1 ]; then
  echo "[eval] $NFAM tile types in $CLIP -- per-type breakdown:"
  # No --lib: the recording's own stamp is the only map that is correct, and passing the
  # shell's $CLIP would skip the check that the two agree.
  $PY scripts/rsl_rl/tile_type_report.py "$OUT" --json "$OUT/by_tile_type.json"
fi
