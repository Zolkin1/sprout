<!-- Copyright 2026 Zachary Olkin. All rights reserved. -->

# Fine-tuning the generator on terrain (AWR and its controls)

Take a trained flow-matching generator and improve it on ONE hard course by re-weighting its
own behaviour, without a simulator in the training loop: collect rollouts, grade them, fit a
value function, weight or filter the chunks, and update the generator. `run_finetune_loop.sh`
runs the whole thing and is resumable per stage.

This is NOT the teacher PPO fine-tune that `README.md` §4c documents; nothing here calls
`train.py`.

## THE ARTIFACTS, and the one that has been mistaken for the other

**Everything in this section was fine-tuned from this pair, and it is not the deployed pair.**
Both are exported side by side in the same `exported/` directory, which is exactly how they
get confused -- the mistake reads as a plausible run against a plausible generator.

**Two parents, and only these two, until a new generator is trained** (decided 2026-09-09):

| arm | checkpoint | serve with |
|---|---|---|
| **reference history** | `logs/flow_matching/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff/best.pt` | `HISTORY_FROM=generator` |
| **measured history** | `logs/flow_matching/2026-09-03_01-42-04_2026-09-03_measured_n3/best.pt` | `HISTORY_FROM=measured` |

| role | artifact |
|---|---|
| **tracker** | `logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1` (the loop's `TRK` default) |
| task | `G1-box2-finetune` |
| *the DEPLOYED generator, for contrast* | `2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx` -- **the pair that drove the robot, and NOT either parent** |

Three things about that pair that are easy to get wrong:

- **`nojumpoff` drops only the BACKWARDS dismount.** `skip_families=['family_jump_off']`;
  `family_jump_off_front`, the forward dismount, is still in the training set -- which matters
  because dismounting a box is forward. Both parents carry the same skip list.
- **There is no runtime arm called "reference".** `HISTORY_ARM_SOURCE` is
  `{"generator": "reference", "measured": "measured"}`, so a reference-trained checkpoint is
  served by `history_from="generator"`, its own previous output. The runtimes refuse a
  mismatched pair: the two kinds of history row are identical in WIDTH, so a mismatch loads
  clean and merely steers wrong.
- **They are not architecture-matched.** The reference parent is `N=4` and every measured
  generator is `N=3`, and they trained on different datasets. So a difference between the two
  is not attributable to the history kind alone; the one-variable comparison is the
  `ref_n3` / `meas_n3` pair, which is what the matched test used to measure +0.003.

Two wrong turns, both of which cost time on 2026-09-09: the deployed generator sits in the
same `exported/` directory and was never a parent; and the three `lag48_aug` runs have
`skip_families=None`, i.e. they INCLUDE the backwards jump-off, so none of them is "the
no-jump-off one".

`nojumpoff` names the difference that matters: it was trained with `family_jump_off` excluded
(the BACKWARDS dismount, which no forward-facing camera can see). Recorded in every arm's own
`finetune.json` under `args.pretrained`, which is the authority if this table ever rots.

## The terrain libraries

One `.h5` per course, manifest-only (no clips): the file defines the ground and nothing else.
Every one regenerates bit-for-bit -- the fingerprints below were reproduced from the builders
on 2026-09-09, and `generator_config` is deliberately outside the fingerprint so a band can be
re-declared without moving the geometry.

| course | library | tiles | fingerprint | band (m/s) | per-tile axis |
|---|---|---|---|---|---|
| two-box | `box2_v2spawn_train.h5` | 5 | `d6e71e0d1c77` | 0.5-2.2 | spawn distance |
| two-box, eval grid | `box2_v2spawn_quick_train.h5` | 2 | `735d39f57cac` | 0.5-2.2 | spawn distance |
| two-box holdout | `box2_v2spawn_holdout.h5` | 1 | `d0154f365c49` | 0.5-2.2 | 2.25 m |
| up-into-down | `updown_p020_r030_train.h5` | 5 | `113b552f9e19` | 0.6-2.5 | approach |
| up-into-down holdout | `updown_p020_r030_holdout.h5` | 1 | `edde0931a5d7` | 0.6-2.5 | 2.25 m |
| varied multi-terrain | `varied60_train.h5` | 60 | `1149eae02028` | 0.6-2.5 | none (geometry) |
| varied holdout | `varied60_holdout.h5` | 60 | `b21be22109d1` | 0.6-2.5 | none (geometry) |
| in-distribution control | `indist_train.h5` | 75 | `8ffdcba61db7` | 0.5-2.2 | none (real tiles) |

Two things were retired on 2026-09-09 and are in the git history rather than here: the **curb**
course (a single wide step, no mjeval twin) and the **randomised free-form section families**
(`flat` / `boxes` / `stairs`). Nothing was ever fine-tuned on either. `motion_gen.terrain.freeform`
still holds the shared foundation every course above is built on -- spawn areas, the route block,
the goal geometry -- and only its section FAMILIES are deprecated.

### Rebuilding them

Every command below was run on 2026-09-09 and reproduced the fingerprint beside it. `LIKE` is
the source library whose CHANNEL LAYOUT is copied -- no motion is copied, and the FM
checkpoint's `frame_names` are checked against the ee frames resolved from the library's own
channel names, so the minimal layout admits only a generator with no extra frames.

```bash
LIKE=clips/g1_mb_v5_body_mode11_decor.h5     # 6.7 GB; on talos it is under bestofk_p4/clips

# two-box: the training grid (5 tiles) + holdout, then the 2-tile eval grid
python scripts/rsl_rl/make_box2_terrain.py --out_dir clips --prefix box2_v2spawn \
    --variant v2 --like $LIKE                            # d6e71e0d1c77 / d0154f365c49
python scripts/rsl_rl/make_box2_terrain.py --out_dir clips --prefix box2_v2spawn_quick \
    --variant v2 --train_distances 1.5 3.0 --like $LIKE   # 735d39f57cac

# up-into-down. --runout 3.0 is NOT the default: it is the variant these results used, and it
# gets its own family because more ground past the bottom riser is more chances to fall.
python scripts/rsl_rl/make_step_courses_terrain.py --course updown --platform_depth 2.0 \
    --runout 3.0 --out_dir clips --like $LIKE             # 113b552f9e19 / edde0931a5d7

# the varied multi-terrain set
python scripts/rsl_rl/make_varied_terrain.py --out_dir clips --prefix varied60 \
    --boxes 30 --updown 30 --like $LIKE                   # 1149eae02028 / b21be22109d1

# the in-distribution control: real clip-library tiles, routes authored over them
python scripts/rsl_rl/make_indist_routes.py --source $LIKE --out_dir clips --prefix indist
                                                          # 8ffdcba61db7
```

Each also writes its `goal_regions.json` (`--goals`, defaulting to the path its course already
uses). That matters because `logs/` is GITIGNORED, so a hand-kept spec does not travel with a
git sync -- talos once held a stale `height_tol_m: 0` after a pull, which is the value that
mistrained an arm for five iterations.

`scripts/rsl_rl/terrain_axis.py --clip <lib> --values` reads the per-tile axis back off the
library, so no reducer is handed a hand-typed one; `varied` and `indist` have no scalar axis
and it correctly returns nothing.

## Running it

```bash
# five AWR iterations on the two-box course
TAG=my_run GPU=0 bash scripts/flow_matching/run_finetune_loop.sh --parent <parent best.pt>

# then evaluate -- NENV must MATCH across the arms of a comparison
NENV=4096 TERRAIN=box2 bash scripts/rsl_rl/run_quick_eval.sh <iter_005/ft/last.pt> my_run
```

One iteration is collect -> grade -> critic (FVI) -> label -> update, ~40 min on one GPU
with the loop's default buffer; the arms that skip the critic (`bc`, `arrived`) run in about
half that, and iteration 1 is longer than the rest because it fills the whole buffer rather
than refreshing part of it.

### Reading the result

`run_quick_eval.sh` reduces its own run at the end, but every number in the paper comes from
one of three scripts, and none of them is reachable by guessing its interface:

```bash
# ONE arm. Pass the TAG directory, not a per-speed directory and not a stem: the reducer
# globs <root>/*/*.episodes.h5, i.e. it looks ONE LEVEL DOWN, so the tag dir reduces every
# speed together. Pointing it at train_v1.17/ finds nothing and it refuses -- correctly,
# since an empty report is indistinguishable from a finished sweep.
python scripts/rsl_rl/box2_ic_report.py logs/box2/quick/my_run

# PAIRED arms -> the figure's numbers, with tile-clustered intervals and a per-cell sign test.
# --arm takes `label:parent:finetuned`, paths relative to --root. It refuses a bare tag.
python scripts/rsl_rl/fig_reduce.py --root logs \
  --arm "two-box:box2/quick/parent_tag:box2/quick/my_run" \
  --arm "in-distribution:indist/quick/parent_tag:indist/quick/my_run" \
  --out fig_numbers.json

# the figure itself -- imports no IsaacLab and no torch, so it runs anywhere
python scripts/rsl_rl/plot_finetune_bars.py fig_numbers.json --out logs/paper/finetune_bars
```

`paired_delta.py` is the third reducer: the arms are 100% spawn-matched, so its paired
interval is the correct one and `fig_reduce`'s level intervals are conservative by comparison.

**Both reducers default to `--metric terminal_reward --failure_rule strict` and to
first-episode-per-env**, which is the definition the loop trains against -- see "Success is
defined ONCE" below. `--pooled` exists and is biased: a round-based stop gives a fast-failing
env more episodes than a slow one, measured at 0.15 on one in-distribution tile.

## The four algorithms are one selectivity ladder

`ALGO=` selects the update and **nothing else** -- collect, grade and buffer churn never branch
on it, so a difference between arms is the objective and not the data
(`test_finetune_loop_algos.py` pins that by parsing the script).

| `ALGO` | update | critic | selects |
|---|---|---|---|
| `awr` | `w = clip(exp(A / (beta * sd A)), 0, 20)` | yes | ~10-15% effective |
| `fbc` | filtered BC, top 30% by advantage | yes | 30% |
| `arrived` | BC on episodes that ARRIVED before failing | **no** | 45-65%, see below |
| `bc` | plain BC on every row | **no** | 100% |

**`arrived`'s yield is a property of the COURSE, not of the algorithm, and the 8.8% this table
used to quote is stale.** Measured on the 2026-09-15 ladder, per iteration:

| course | iter 1 | iter 5 | trend |
|---|---|---|---|
| boxes | 0.450 | 0.535 | rises as the policy improves |
| updown | 0.624 | 0.627 (it 4) | |
| multi | 0.552 | 0.556 (it 2) | |

Two consequences. **A starved `arrived` arm is a real hazard and it is not the current regime** --
the 8.8%, and the 0.40% the `loose` section below records, come from a weaker parent scored
differently, so do not reason about this arm from those numbers. And **`arrived` adapts its data
budget while `fbc` cannot**: fbc keeps exactly 30% at every iteration because it is a top-quantile
filter, so late in training it is still discarding 70% of rollouts when half of them succeeded,
where arrived keeps the half that did. That is a plausible mechanism for arrived beating fbc by
+0.0773 paired (p=2.3e-10) on boxes, and it is measured rather than inferred -- but it is one
course, and the updown/multi arms are what would make it a finding.

`awr`'s figure is an EFFECTIVE sample size, not a keep fraction, and it is current: the ladder
arm logged `awr/ess_frac 0.160`. `bc` writes no label file at all, which is what its 100% means.

`bc` passes no `--labels` at all -- not a hot `beta`. AWR's hot limit is unweighted BC on the
labelled rows and its sharp limit under `global` normalization IS filtered BC, so an arm
reported as plain BC has to actually be plain BC or the ladder has no bottom rung.

## Success is defined ONCE, and three places have to agree

- **the reward** pays `gate_worst * C * reached_goal * is_last` (C = 10) on the episode's last
  re-plan, gated by the episode's WORST terrain chunk -- so arriving destructively pays nothing;
- **the critic** keeps every window up to the first failure and drops the rest;
- **the eval** reports `rollout_outcomes.terminal_reward >= 0`: arrival BEFORE the first failure,
  so a fall in the run-out after the goal is a success and a robot dragged over the line after
  failing is not.

`FAILURE_RULE` decides what "failure" means for the last two, and **it is `strict` -- the fall
OR the tracking bounds -- because Isaac now TERMINATES on those bounds**
(`terminations.tracking_violation`, height 0.12 m). On a recording made with that termination
the two rules agree by construction: an episode that leaves the bounds ends there, so it cannot
arrive afterwards.

**On a recording made BEFORE the termination, score it with `loose`.** There the run continued
past a violation, and strict then fires on 99.5% of episodes at a median step 378 against a
median ARRIVAL at 765 -- consequences that read as policy results --

- the `arrived` filter keeps 33 of 8310 episodes (0.40%) instead of 728 (8.8%) -- both
  figures are THAT recording's, made with a weaker parent; the same filter runs at
  45-65% on the 2026-09-15 ladder, so read this as the size of the `strict`/`loose`
  effect and not as what the arm keeps today;
- the critic's window cut removes the episode's LAST re-plan, the only row the success bonus is
  paid on, from **95.3%** of the arriving episodes -- so the critic gets a reward with
  essentially no success term in it.

Of the strict violations on arriving episodes, **88% are the HEIGHT bound firing first**
(`max_height_m = 0.12` since 2026-09-09, chosen from the measured arrived-vs-fell separation
rather than fitted); tilt 9.9%, lateral 3.3%. Those bounds are marked
`provenance["bootstrap"]` and were never fitted. `TODO(tracking-height-bound)`.

## Spawn areas: training draws all of them, every eval draws the ground

Raised areas -- a box top, the platform between two flights -- exist because a complete
traverse is rare enough that the behaviour policy never produced one (of 352 successful two-box
episodes, every one reached the goal by dismounting), and AWR can only reweight what it is
shown. Scoring from those same areas would report a fraction of the course as a traverse, so
`run_quick_eval.sh` passes `SPAWN_FROM=ground` on every terrain and prints it.

It costs nothing in comparability: every family authors its ground area at exactly
`freeform.FREEFORM_SPAWN_JITTER_M`, which is also what the reset synthesises for a tile
declaring none -- so a ground-only draw is bit-identical to the pre-spawn-areas libraries every
earlier number was measured on (`test_eval_spawn_ground_only.py`).

## Watching what the bounds catch

`transfer/sim/watch_tracking.py` replays episodes in MuJoCo and then plots position-vs-served-node
per axis with the bounds drawn on, the latch marked, and the goal-entry step marked -- so
"violated, recovered, arrived anyway" is a shape rather than two numbers. On the two-box course:

```bash
cd transfer/sim
python watch_tracking.py --clip_lib ../../clips/box2_v2spawn_train.h5 \
  --tile 0 --klass box --obstacle_from raised --spawn_from route \
  --speed 1.17 --seeds 0-7 --rt 0.5 \
  --tracker_onnx ../../logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/tracker.onnx \
  --fm_onnx ../../logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx
```

To watch a FINE-TUNED arm on the same course, swap the generator. Fine-tune checkpoints are
`.pt` and this script wants a graph, so export once:

```bash
python scripts/flow_matching/export_fm_onnx.py --checkpoint logs/box2/prog_s3_iter013.pt \
  --out logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/awr_prog_s3_iter013_fm.onnx
```

The arms worth watching, and where their checkpoints live:

Success rates below are under the PRE-TERMINATION rule (`--metric reached_goal
--failure_rule loose`), which is what existed when they were collected. See CLAUDE.md for why
the current default reports something different on the same files.

| arm | box2 success | checkpoint |
|---|---|---|
| parent | 0.2440 | `logs/flow_matching/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff/best.pt` |
| `awr_prog_s3` (progress reward) | 0.5950 @ iter 5 | `logs/box2/prog_s3_iter013.pt` is iter 13, local |
| **`awr_vm30`** (+ velocity mismatch 0.30) | **0.7257** @ iter 5 | talos `/mnt/zolkin_new/sprout/awr_vm30/iter_005/ft/last.pt` |

`awr_vm30` iteration 5 is the best box2 arm. It is talos-only, so export it THERE and copy the
graph rather than the 428 MB checkpoint -- `CUDA_VISIBLE_DEVICES=` keeps it off the GPUs a run
may be using:

```bash
ssh talos 'cd /mnt/zolkin_new/sprout/sprout && CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=source/robot_rl:source/motion_gen ./env_isaaclab/bin/python \
  scripts/flow_matching/export_fm_onnx.py \
    --checkpoint /mnt/zolkin_new/sprout/awr_vm30/iter_005/ft/last.pt \
    --out /tmp/awr_vm30_iter005_last_fm.onnx'
scp talos:/tmp/awr_vm30_iter005_last_fm.onnx \
  logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1/exported/
```

Its `fm_meta` carries `history_source: null` where the newer exports say `reference`; that is
the pre-flag spelling and `assert_history_source_matches` reads a null AS `reference`, which is
what it is. Not a mismatch.

`--obstacle_from raised` is required on a box course: the default tread filter refuses a 1.20 m
box by design. Nothing else in `transfer/sim` applies the strict bounds -- `mode_probe.py` and
`gen_tracker_runner.py` pass no `tracking_spec`, so every clip they produce is fall-only. In
ISAAC the violation is now a TERMINATION (`terminations.tracking_violation`), so a recording
made after 2026-09-09 ends at the bound; recordings made before it ran on past one, which is
why the reducers take `--failure_rule`. In MuJoCo this script is still the only place the
strict rule is visible.

## Traps, each already paid for

- **`height_tol_m = 0` means OPPOSITE things at the two ends.** It DISABLES the height test in
  `terminations.freeform_goal_reached` and is an EXACT-EQUALITY test in `goal_region.in_goal`.
  A spec of 0 scored an arm at 0.0000 over 8400 episodes AND zeroed the reward's success term
  for five whole training iterations.
- **IsaacLab type-checks hydra overrides**: `height_tol_m=0` (int) is refused where `0.0` is
  accepted, and it fails at hydra composition before the GPU is touched.
- **`--num_episodes` selects the SHORTEST episodes.** A total-count stop cuts every slow episode
  still in flight. Use `--episodes_per_env`, and reduce FIRST-PER-ENV: a round-based stop leaves
  the fast envs with extra short episodes, so pooling tilts the sample toward fast failures
  (measured bias 0.15 on one in-distribution tile).
- **`NENV` is part of the comparison, not of the box.** It sets how many episodes are in flight
  at the stop, so it decides the censored fraction; 4096 on talos.
