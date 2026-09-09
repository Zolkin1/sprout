#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The full AWR loop: collect -> grade -> churn the buffer -> fit the critic -> label -> update
# the policy, N times, each iteration collecting with the policy the LAST one produced.
#
# What makes this a loop rather than the single advantage-weighted-BC step already measured
# (+0.125 success on the box2 v2 course, 30/30 matched cells): the behaviour policy moves with
# the target policy, so the advantage is re-estimated against a policy that has changed. The
# buffer is a fixed number of EPISODES -- iteration 1 fills it, and every later iteration
# collects one refresh and drops a random refresh-worth, so the critic sees a mix of about
# four policy generations rather than only the newest.
#
# RESUMABLE BY DESIGN. Every stage skips when its output exists, and the buffer manifest is
# the commit point -- once `iter_<NNN>/buffer.json` is written the drop is never redrawn,
# because a redraw after a half-written merge yields a buffer that does not match the manifest
# every later stage reads. A kill costs the stage it was in, not the iteration.
#
# TWO THINGS THAT ARE DELIBERATE AND LOOK LIKE OVERSIGHTS:
#   * `last.pt` is chained, never `best.pt`. `best.pt` is selected on UNWEIGHTED
#     val_flow_mse, which AWR moves away from by design, so it favours the checkpoint that
#     changed LEAST -- on three arms it picked the least-changed or the luckiest epoch.
#   * the critic's ENCODER stays pinned to the parent for every iteration. Only its value
#     heads refit. A moving encoder would give each iteration a different feature basis, so
#     the returns would not be comparable across the series the loop exists to produce.
#
# The quick eval is NOT in here: it is ~50 min against a ~37 min iteration, and it needs a
# GPU this loop is using. Run it afterwards against any iteration's `ft/last.pt`.
#
# Usage:
#   bash scripts/flow_matching/run_awr_loop.sh --tag a4k --gpu 1 --buffer 4000 --refresh 1000
#   bash scripts/flow_matching/run_awr_loop.sh --tag b2k --gpu 2 --buffer 2000 --refresh 500
set -uo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W" || exit 1

# ---- knobs ----------------------------------------------------------------------------
TAG=""; GPU=""; BUFFER=4000; REFRESH=1000; ITERS=5; ROOT=""
PARENT="${PARENT:-/mnt/zolkin_new/sprout/terrain_reward_smoke/fm/best.pt}"
TRK="${TRK:-2026-08-20_16-36-10_mode11_decor_plan_v1}"
CLIP="${CLIP:-clips/box2_v2_train.h5}"
GOALS="${GOALS:-logs/box2/goal_regions.json}"
TASK="${TASK:-G1-box2-finetune}"
NENV="${NENV:-2048}"
EPOCHS_CRITIC="${EPOCHS_CRITIC:-5}"
EPOCHS_FT="${EPOCHS_FT:-5}"
#: The PROGRESS reward's constants. Off by default would be the wrong default here -- the
#: whole point of this arm is that the plain terrain grade is maximised by standing still
#: (measured: 5 iterations took forward progress 11.50 -> 3.95 m per 1000 steps while
#: survival ROSE), so the loop grades and fits the progress reward unless told otherwise.
PROGRESS="${PROGRESS:-1}"
GATE_S="${GATE_S:-2.0}"                 # velocity gate sharpness
SUCCESS_GATE_S="${SUCCESS_GATE_S:-3.0}" # success gate sharpness, on the episode's WORST chunk
VEL_SIGMA="${VEL_SIGMA:-0.5}"
SUCCESS_C="${SUCCESS_C:-10.0}"
#: Rounds of FITTED VALUE ITERATION per iteration. Round 0 is the Monte Carlo return; each
#: later round refits against `r + gamma^N V` with V FROZEN as the previous round's
#: predictions on disk. 1 = pure MC, i.e. the old behaviour.
FVI_ROUNDS="${FVI_ROUNDS:-2}"
#: The SPEED-MISMATCH arm's fraction, within the eligible population. Off by default: it is a
#: third change on top of the reward and the spawns, and an arm nobody asked for should not be
#: in a run's data silently. Set it per run (SPEED_MISMATCH=0.5) once the reward itself has a
#: baseline. The reward scores velocity against the NOMINAL speed, so this arm is a
#: corruption to reject rather than a command to follow -- see the cfg field's docstring.
SPEED_MISMATCH="${SPEED_MISMATCH:-0.0}"
BETA="${BETA:-0.5}"
WORKERS="${WORKERS:-16}"
PROJECT="${PROJECT:-robot_rl_awr}"
#: The commanded-speed band. Defaults come from the CLIP LIBRARY when it declares one
#: (`clip_io.read_commanded_speed_range`), because the band is a property of the course --
#: a 6 x 0.15 m staircase and a two-box jump want different ones, and before this the band
#: lived only in whichever launcher happened to pass an override. An explicit SPEED_MIN /
#: SPEED_MAX still wins, so a deliberate sweep is unaffected; a library that declares
#: nothing falls back to these, which is box2's historical band.
#: The goal's HEIGHT tolerance (m), 0 = the test off.  It has to be settable from here: on a
#: course whose goal is ABOVE the spawn the downrange-and-lateral goal test scores a robot
#: that never climbed, and that inflated `reached_goal` feeds the reward's success term -- so
#: the fine-tune would be trained on arrivals that did not happen. Needed for the
#: in-distribution `stairs_up` / `jump_on` tiles; harmless (and off) everywhere else.
HEIGHT_TOL="${HEIGHT_TOL:-0.0}"
SPEED_MIN="${SPEED_MIN:-}"
SPEED_MAX="${SPEED_MAX:-}"
PY="${PY:-$W/env_isaaclab/bin/python}"

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --buffer) BUFFER="$2"; shift 2;;
    --refresh) REFRESH="$2"; shift 2;;
    --iterations) ITERS="$2"; shift 2;;
    --root) ROOT="$2"; shift 2;;
    --parent) PARENT="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$TAG" ] || { echo "REFUSING: --tag is required; it names the output tree, the wandb"; \
                   echo "  group and the drop seed, so two arms cannot collide."; exit 2; }
[ -n "$GPU" ] || { echo "REFUSING: --gpu is required."; exit 2; }

#: Resolve the speed band: the library's declaration unless the caller overrode it. Read with
#: the repo's own reader rather than by poking at h5 attrs here, so the storage location is
#: stated in exactly one place.
if [ -z "$SPEED_MIN" ] || [ -z "$SPEED_MAX" ]; then
  _rng=$("$PY" - "$CLIP" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, "source/motion_gen")
from motion_gen.clips.clip_io import read_commanded_speed_range
r = read_commanded_speed_range(sys.argv[1])
print(f"{r[0]} {r[1]}" if r else "")
PYEOF
)
  if [ -n "$_rng" ]; then
    SPEED_MIN="${SPEED_MIN:-$(printf '%s' "$_rng" | cut -d' ' -f1)}"
    SPEED_MAX="${SPEED_MAX:-$(printf '%s' "$_rng" | cut -d' ' -f2)}"
    SPEED_SRC="declared by $(basename "$CLIP")"
  else
    SPEED_MIN="${SPEED_MIN:-0.5}"; SPEED_MAX="${SPEED_MAX:-2.2}"
    SPEED_SRC="loop default ($(basename "$CLIP") declares none)"
  fi
else
  SPEED_SRC="set on the command line"
fi
for pair in "buffer:$BUFFER" "refresh:$REFRESH" "iterations:$ITERS"; do
  case "${pair#*:}" in
    ''|*[!0-9]*) echo "REFUSING: --${pair%%:*} must be a positive integer, got '${pair#*:}'."
                 exit 2;;
  esac
done
[ "$REFRESH" -le "$BUFFER" ] || {
  echo "REFUSING: --refresh ($REFRESH) exceeds --buffer ($BUFFER); there would be nothing"
  echo "  left to keep."; exit 2; }
[ "$ITERS" -ge 1 ] || { echo "REFUSING: --iterations must be at least 1."; exit 2; }
ROOT="${ROOT:-/mnt/zolkin_new/sprout/awr_$TAG}"
# The drop seed is derived from the tag, so the two arms churn differently and a re-run of
# one arm reproduces its own churn exactly.
SEED=$(( $(printf '%s' "$TAG" | cksum | cut -d' ' -f1) % 100000 ))

export PYTHONPATH="$W/source/robot_rl:$W/source/motion_gen"
export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT="$PROJECT"

if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSING: the worktree is dirty; the run would not match the pushed tree."; exit 2
fi
[ -f "$PARENT" ] || { echo "REFUSING: parent checkpoint $PARENT not found."; exit 3; }
[ -f "$CLIP" ]   || { echo "REFUSING: clip library $CLIP not found."; exit 3; }
[ -f "$GOALS" ]  || { echo "REFUSING: goal regions $GOALS not found."; exit 3; }

mkdir -p "$ROOT"
HEAD="$(git rev-parse --short HEAD)"
echo "==================================================================="
echo "[awr] tag $TAG  gpu $GPU  worktree $HEAD"
echo "[awr] buffer $BUFFER episodes, refresh $REFRESH, $ITERS iterations"
echo "[awr] parent $PARENT"
echo "[awr] root   $ROOT"
echo "[awr] wandb  $PROJECT / ONE run 'loop_$TAG' / seed $SEED"
echo "[awr] reward $([ "$PROGRESS" = "1" ] && echo "PROGRESS (s_vel=$GATE_S s_success=$SUCCESS_GATE_S sigma=$VEL_SIGMA C=$SUCCESS_C)" || echo "terrain only")"
echo "[awr] critic $FVI_ROUNDS FVI round(s)$([ "$PROGRESS" = "1" ] && echo ", worst-penalty-so-far context")"
echo "[awr] arms   camera 0.5 (dead, per-replan), history 0.0, speed-mismatch $SPEED_MISMATCH"
echo "[awr] clip   $CLIP"
echo "[awr] speed  $SPEED_MIN-$SPEED_MAX m/s, continuous per episode ($SPEED_SRC)"
echo "[awr] goal   height tolerance $HEIGHT_TOL m $(awk -v v="$HEIGHT_TOL" 'BEGIN{exit !(v+0==0)}' && echo "(height test OFF -- a goal ABOVE the spawn can be scored from the floor)" || echo "(pelvis must arrive at the goal height)")"
echo "==================================================================="

fail () { echo "[awr] FAILED at $1 (iteration $2)" >&2; exit 1; }

# Create the run NOW, before any iteration finishes. The loop's own row is written only at
# the end of an iteration, so without this nothing reaches wandb for the first 25-40 minutes
# and a crash inside iteration 1 leaves no record at all -- the arm's absence from the
# sidebar has to mean "launch failed", not "wait longer".
"$PY" -u scripts/flow_matching/awr_log_iteration.py --mode start \
  --root "$ROOT" --project "$PROJECT" --run_id "loop_$TAG" --run_name "awr_$TAG" \
  --tag "$TAG" --buffer "$BUFFER" --refresh "$REFRESH" --iterations "$ITERS" \
  --epochs_per_iter "$EPOCHS_FT" --parent "$PARENT" --git_head "$HEAD" \
  --gpu "$GPU" --seed "$SEED" || true

for (( it=1; it<=ITERS; it++ )); do
  INC="$ROOT/$(printf 'inc_%03d' "$it")"
  ITD="$ROOT/$(printf 'iter_%03d' "$it")"
  CRITIC="$ITD/critic"
  FT="$ITD/ft"
  mkdir -p "$INC" "$ITD"

  # Iteration 1 fills the buffer; every later one collects a single refresh.
  if [ "$it" -eq 1 ]; then NEPS="$BUFFER"; POLICY="$PARENT";
  else NEPS="$REFRESH"; POLICY="$ROOT/$(printf 'iter_%03d' $((it-1)))/ft/last.pt"; fi
  # **COLLECT IN WHOLE ROUNDS, NOT TO A TOTAL COUNT.** A total-count stop fires as soon as
  # the fastest episodes have finished -- and under a per-episode speed draw the fast ones
  # end in ~150 steps while a slow one runs 600+, so it keeps the short episodes and
  # discards every slow one still in flight. Measured on the box2 band: the kept episodes
  # had a median of 177 steps against a population median of 392, and their terrain-penalty
  # tail was HALF as wide -- the tail being the signal AWR feeds on. Running until every env
  # has finished ROUNDS episodes makes membership independent of duration, and it scales the
  # right way: more envs collect more per round rather than a worse sample.
  # Never run MORE envs than episodes wanted. A round is all-or-nothing, so 500 episodes on
  # 2048 envs would collect 2048 and discard 1548 -- paying for four times the sim and four
  # times the grade to keep a quarter of it. Capping at the request makes the small refresh
  # collections exact (500 on 500 envs, nothing discarded) while a large one still uses the
  # whole card: 8000 wanted keeps all 2048 envs and runs 4 rounds. This is NOT the earlier
  # "use fewer robots" idea -- the bias is already gone, killed by the round-based stop; this
  # only stops over-collecting.
  ENVS=$(( NENV < NEPS ? NENV : NEPS ))
  ROUNDS=$(( (NEPS + ENVS - 1) / ENVS ))
  COLLECT=$(( ROUNDS * ENVS ))
  # `--num_windows` is a HARD CAP *and* the file pre-allocation, so it is set far above what
  # the episode budget will produce: as the policy improves episodes survive longer and make
  # MORE re-plans each, so a cap sized for iteration 1 would silently truncate iteration 5
  # and quietly shrink the buffer. `awr_buffer adopt` refuses a run where it bound.
  # Sized from the EPISODE LENGTH CAP, not a guessed windows-per-episode. At
  # episode_length_s=40, a 0.02 s control step and K=12 that is 40/0.24 = 167 re-plans for an
  # episode that runs the full clock, so an env can contribute at most ROUNDS * 167 windows
  # once the claim is gated by the quota. The old `COLLECT * 150` was calibrated on
  # total-count collections whose episodes were TRUNCATED at ~30 windows, and it bound at
  # 300k on the first round-based run. `adopt` refuses a run where the cap bound, so an
  # under-estimate costs a re-collection rather than a silently short buffer.
  WIN_PER_EP="${WIN_PER_EP:-170}"
  NWIN=$(( ENVS * ROUNDS * WIN_PER_EP ))
  # HALF the perturbation gate after the first iteration. Iteration 1 collects from the parent
  # and is the one increment that has to carry the widest behaviour spread on its own; from
  # iteration 2 the buffer already holds several generations, so a narrower gate keeps more of
  # each increment pristine while the mix stays broad. 0.5 x the 0.5 camera arm is ~25% of
  # plans perturbed on iteration 1 and ~12.5% thereafter.
  PERTURB_FRAC=$([ "$it" -eq 1 ] && echo 0.5 || echo 0.25)

  echo ""
  echo "[awr] ####### iteration $it/$ITERS  ($NEPS episodes)  $(date +%H:%M) #######"
  [ -f "$POLICY" ] || fail "policy $POLICY missing" "$it"
  echo "[awr] behaviour policy: $POLICY"

  # ---- 1. collect -------------------------------------------------------------------
  if [ -f "$INC/rollout_${TRK}.episodes.h5" ]; then
    echo "[awr] 1/8 collect     SKIP (exists)"
  else
    echo "[awr] 1/8 collect     $ROUNDS round(s) x $ENVS envs = $COLLECT episodes"\
         " (sampling $NEPS), speed $SPEED_MIN-$SPEED_MAX"
    # ONE launch over the whole speed band: the commanded speed is drawn per EPISODE, so
    # pinning it (as the v2 collection did) would mean six Isaac starts for no extra
    # coverage. The band matches the range those six pinned runs covered.
    "$PY" -u scripts/rsl_rl/record_rollouts.py \
      --task="$TASK" --load_run "$TRK" --fm_checkpoint "$POLICY" \
      --clip "$CLIP" --clip_device cpu --goal_regions "$GOALS" \
      --num_envs "$ENVS" --episodes_per_env "$ROUNDS" --num_windows "$NWIN" \
      --output_dir "$INC" \
      env.commands.traj_ref.fm_perturb_fraction="$PERTURB_FRAC" \
      env.commands.traj_ref.fm_upper_camera_noise_fraction=0.5 \
      env.commands.traj_ref.fm_upper_camera_noise_mode=dead \
      env.commands.traj_ref.fm_upper_camera_noise_cadence=replan \
      env.commands.traj_ref.fm_history_null_fraction=0.0 \
      env.commands.traj_ref.fm_speed_mismatch_fraction="$SPEED_MISMATCH" \
      env.commands.traj_ref.fm_speed_mismatch_cadence=replan \
      env.commands.traj_ref.freeform_speed_min="$SPEED_MIN" \
      env.commands.traj_ref.freeform_speed_max="$SPEED_MAX" \
      env.episode_length_s=40.0 \
      env.terminations.goal_reached.params.height_tol_m="$HEIGHT_TOL" \
      > "$INC/collect.log" 2>&1 || fail "collect (see $INC/collect.log)" "$it"
  fi

  # ---- 2. grade ---------------------------------------------------------------------
  # Graded on the INCREMENT, before adoption, and never again: an episode is graded once
  # ever, which is what keeps a re-materialised buffer cheap.
  if [ -f "$INC/terrain_reward_sidecar.npz" ]; then
    echo "[awr] 2/8 grade       SKIP (exists)"
  else
    if [ "$PROGRESS" = "1" ]; then
      echo "[awr] 2/8 grade       PROGRESS reward (terrain + gated velocity + terminal success)"
      PROGRESS_ARGS=(--progress --gate_sharpness "$GATE_S"
                     --success_gate_sharpness "$SUCCESS_GATE_S"
                     --vel_sigma "$VEL_SIGMA" --success_const "$SUCCESS_C")
    else
      echo "[awr] 2/8 grade       terrain reward, every window"
      PROGRESS_ARGS=()
    fi
    "$PY" -u scripts/rsl_rl/inspect_terrain_reward.py --stage reward --reward_windows 0 \
      "${PROGRESS_ARGS[@]}" \
      --rollout "$INC/rollout_${TRK}" --clip "$CLIP" \
      --isaac_joint_names "logs/rsl_rl/g1/$TRK/exported/policy_parameters.yaml" \
      --device cuda --out "$INC/grade" \
      --reward_out "$INC/terrain_reward_sidecar.npz" \
      > "$INC/grade.log" 2>&1 || fail "grade (see $INC/grade.log)" "$it"
  fi

  # ---- 3-5. adopt, churn, materialise -----------------------------------------------
  echo "[awr] 3/8 adopt"
  "$PY" -u scripts/flow_matching/awr_buffer.py --root "$ROOT" adopt \
    --iteration "$it" --expect_episodes "$NEPS" --policy "$POLICY" \
    --rounds "$ROUNDS" --trim_seed "$SEED" || fail adopt "$it"
  echo "[awr] 4/8 plan        (drop $([ "$it" -eq 1 ] && echo 0 || echo "$REFRESH"))"
  "$PY" -u scripts/flow_matching/awr_buffer.py --root "$ROOT" plan \
    --iteration "$it" --capacity "$BUFFER" \
    --refresh "$([ "$it" -eq 1 ] && echo 0 || echo "$REFRESH")" \
    --seed "$SEED" || fail plan "$it"
  # STAGES 5-8 ARE SKIPPED WHOLESALE once this iteration's fine-tune has finished, and the
  # buffer is what makes that worth doing: it is deleted after the NEXT iteration verifies,
  # so a restart would otherwise fail `verify` here and re-merge several GB for stages that
  # are all about to be skipped anyway. Nothing after the fine-tune reads the buffer.
  if [ -f "$FT/finetune.json" ]; then
    echo "[awr] 5-8/8          SKIP (this iteration's fine-tune is complete)"
  else
  echo "[awr] 5/8 materialise"
  "$PY" -u scripts/flow_matching/awr_buffer.py --root "$ROOT" verify --iteration "$it" \
    || "$PY" -u scripts/flow_matching/awr_buffer.py --root "$ROOT" materialise \
         --iteration "$it" || fail materialise "$it"

  BUF_STEM=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['buffer']['stem'])" \
             "$ITD/buffer.json") || fail "reading the manifest" "$it"
  BUF_NPZ=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['buffer']['reward_npz'])" \
            "$ITD/buffer.json") || fail "reading the manifest" "$it"
  echo "[awr]     buffer -> $BUF_STEM"

  # ---- 6. critic --------------------------------------------------------------------
  if [ -f "$CRITIC/predictions.npz" ]; then
    echo "[awr] 6/8 critic      SKIP (exists)"
  else
    # `--no_wandb`: the CHILD stages log nothing. One run per arm is assembled by
    # `awr_log_iteration.py` from the artifacts instead -- see its docstring for why
    # per-stage runs were tried and abandoned (20 runs, three of them all named `ft`, and a
    # fine-tune curve split across runs that cannot be overlaid).
    #
    # **`--clip_c` SELECTS THE REDUCER, and 2.5 would silently keep the old reward.** The
    # clipped-penalty path maps `r = -min(p, c)` into [-1, 0], which is the wrong target for
    # a goodness in roughly [0, 12] with a one-off terminal bonus. `--clip_c 0` takes the
    # progress branch (`bootstrapped_window_return`, bounded by 2 + (1-gamma)*C).
    if [ "$PROGRESS" = "1" ]; then
      REWARD_ARGS=(--clip_c 0 --worst_context)
    else
      REWARD_ARGS=(--clip_c 2.5)
    fi
    # FITTED VALUE ITERATION. Round 0 fits the Monte Carlo return; each later round refits
    # against the previous round's FROZEN predictions. Frozen because it is a file: a target
    # recomputed from the live model is a moving target, and the failure is a slow divergence
    # no metric on the fit reports. Only the LAST round's output feeds the label stage, but
    # every round's gates.json is kept so the effect of bootstrapping is visible.
    ROUNDS_FVI=1
    [ "$PROGRESS" = "1" ] && ROUNDS_FVI=$FVI_ROUNDS
    PREV_PRED=""
    for (( fr=0; fr<ROUNDS_FVI; fr++ )); do
      RDIR="$CRITIC"
      [ "$ROUNDS_FVI" -gt 1 ] && RDIR="$CRITIC/round_$fr"
      if [ -f "$RDIR/predictions.npz" ]; then
        echo "[awr] 6/8 critic      round $fr SKIP (exists)"
        PREV_PRED="$RDIR/predictions.npz"; continue
      fi
      BOOT_ARGS=()
      [ -n "$PREV_PRED" ] && BOOT_ARGS=(--bootstrap_v "$PREV_PRED" --bootstrap_variant V_priv)
      echo "[awr] 6/8 critic      round $fr/$((ROUNDS_FVI - 1)), $EPOCHS_CRITIC epochs, 2 folds"\
           "$([ -n "$PREV_PRED" ] && echo "(bootstrapped)" || echo "(Monte Carlo)")"
      "$PY" -u scripts/flow_matching/fit_value.py "$BUF_STEM" --no_wandb \
        --fm_checkpoint "$PARENT" \
        --reward_npz "$BUF_NPZ" \
        "${REWARD_ARGS[@]}" "${BOOT_ARGS[@]}" \
        --window 3 --quality_kind mse --quality_only \
        --folds 2 --epochs "$EPOCHS_CRITIC" --variants V_priv --device cuda \
        --out "$RDIR" >> "$ITD/critic.log" 2>&1 \
        || fail "critic round $fr (see $ITD/critic.log)" "$it"
      PREV_PRED="$RDIR/predictions.npz"
    done
    # The label stage reads `$CRITIC/predictions.npz`, so the last round's answer has to be
    # THERE rather than one directory down. Copied, not moved: the round dir stays a complete
    # record of the fit that produced it.
    #
    # GLOBBED, because the checkpoint's NAME depends on the fold count: `fit_value` writes
    # `V_priv.pt` at one fold and `V_priv.fold0.pt` / `V_priv.fold1.pt` at two. A literal
    # `V_priv.pt` therefore always failed here -- and this script is `set -uo pipefail` with
    # NO `-e`, so it failed silently and left $CRITIC holding predictions and gates but no
    # weights at all. `|| fail` is what makes that loud.
    if [ "$ROUNDS_FVI" -gt 1 ]; then
      RD="$(dirname "$PREV_PRED")"
      cp "$PREV_PRED" "$CRITIC/predictions.npz" || fail "collecting predictions.npz" "$it"
      cp "$RD"/V_priv*.pt "$CRITIC/" || fail "collecting the critic weights" "$it"
      cp "$RD/gates.json" "$CRITIC/gates.json" || fail "collecting gates.json" "$it"
    fi
  fi

  # ---- 7. label ---------------------------------------------------------------------
  if [ -f "$CRITIC/advantage.npz" ]; then
    echo "[awr] 7/8 label       SKIP (exists)"
  else
    echo "[awr] 7/8 label       MC advantage, terrain only"
    "$PY" -u scripts/flow_matching/label_advantage.py "$CRITIC/predictions.npz" \
      --terrain_only --w_fail 0 --estimator mc --variant V_priv \
      --out "$CRITIC/advantage.npz" >> "$ITD/critic.log" 2>&1 || fail label "$it"
  fi

  # ---- 8. fine-tune -----------------------------------------------------------------
  # GUARDED ON `finetune.json`, NOT `last.pt`. `last.pt` is rewritten after every epoch, so
  # a kill during epoch 2 of 3 would satisfy a `last.pt` guard: the iteration would be
  # skipped on resume and every later iteration would chain from a part-trained policy, with
  # nothing recording the shortfall. `finetune.json` is written once, at the end.
  if [ -f "$FT/finetune.json" ]; then
    echo "[awr] 8/8 finetune    SKIP (exists)"
  else
    echo "[awr] 8/8 finetune    $EPOCHS_FT epochs from $(basename "$(dirname "$POLICY")")"
    "$PY" -u scripts/flow_matching/finetune_generator.py --no_wandb \
      --pretrained "$POLICY" \
      --rollout "${BUF_STEM}.h5" \
      --labels "$CRITIC/advantage.npz" \
      --awr_beta "$BETA" --awr_normalize global --awr_center none --awr_balance none \
      --awr_clip 20.0 --batch_size 256 --num_workers "$WORKERS" \
      --epochs "$EPOCHS_FT" --ckpt_every 1 --device cuda \
      --out "$FT" > "$ITD/ft.log" 2>&1 || fail "finetune (see $ITD/ft.log)" "$it"
  fi
  fi                                       # end of the stages-5-8 block

  # ---- iteration-level telemetry, and the previous buffer's disk ---------------------
  "$PY" -u scripts/flow_matching/awr_log_iteration.py \
    --root "$ROOT" --iteration "$it" --critic_dir "$CRITIC" --ft_dir "$FT" \
    --project "$PROJECT" --run_id "loop_$TAG" --run_name "awr_$TAG" \
    --epochs_per_iter "$EPOCHS_FT" \
    >> "$ITD/critic.log" 2>&1 || true

  # Drop iteration k-1's MERGED buffer now that k's has verified. Increments are never
  # deleted -- they are the reproducible source, and the manifest names episodes inside them.
  if [ "$it" -gt 1 ]; then
    PREV="$ROOT/$(printf 'iter_%03d' $((it-1)))/buffer"
    for suffix in .h5 .episodes.h5 _reward.npz; do
      [ -f "$PREV$suffix" ] && rm -f "$PREV$suffix" && echo "[awr]     freed $PREV$suffix"
    done
  fi
  echo "[awr] iteration $it done $(date +%H:%M) -> $FT/last.pt"
done

echo ""
echo "[awr] ALL $ITERS ITERATIONS DONE $(date +%H:%M)"
echo "[awr] policies:"
for (( it=1; it<=ITERS; it++ )); do
  echo "    iter $it -> $ROOT/$(printf 'iter_%03d' "$it")/ft/last.pt"
done
echo "[awr] evaluate with: NENV=4096 bash scripts/rsl_rl/run_box2_quick_eval.sh"
