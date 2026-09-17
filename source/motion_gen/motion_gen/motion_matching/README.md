# `motion_matching` — DEPRECATED, retained for ablations

Motion matching over a cached candidate graph: the clip generator that preceded the
Motion Bricks (MB) pipeline. **Nothing on the live pipeline imports this package.** It is
kept so the MB clips can be compared against a matcher baseline in an ablation.

## What is here

| module | role |
|---|---|
| `graph.py` | cached candidate graph over the motion DB |
| `vanilla.py` | greedy local-lookahead ("trajectory cost") matcher |
| `beam.py` | approximate beam search |
| `dp.py` | exact backward dynamic program — the strongest of the three |
| `locomotion_db.py` | assembles the flat bones + clean-stride + jump-lead-in DB the matchers search |
| `slope_rotate.py` | rotates flat clips onto a slope — shelved (works only straight up/down-slope; cross-slope and turning clips lean sideways) |

The matching feature space and its FK moved to `motion_gen/features.py` (see "What moved
out" below); this table listed it here long after it left.

`trajopt_ingest.py` and `mm_clip_generator.py` were removed when the trajectory-manager stack
was deleted: the former needed the manager's full Bezier evaluation, and the latter was only
reachable through it. Neither was on any runnable path (no CLI imported them) — the ablation
drivers below all seed from the bones cache or the clean stride library.

## Running it

Its drivers live in `scripts/rsl_rl/` and each carries a `DEPRECATED:` banner naming the MB
equivalent:

**The committed bones cache is STALE and every matcher path is blocked on it.**
`clips/bones_cache_locomotion.h5` (2.7 GB, built 2026-07-08) carries `joint_pos` of width **21**
against this tree's 29-joint robot, and its `feature_frames` name `left_palm_link` /
`right_palm_link` -- the Dex3 hands, which the perceptive robot replaced with passive rubber ones.
So it describes neither the joint set nor the bodies anything here now has. Rebuilding it is one
`precompute_bones_cache.py` run and NOT a code change; nothing else in the matcher stack is known
to be stale. Do that first -- an ablation against this cache would be an ablation against a
different robot.

```bash
# once: build the bones feature cache every matcher needs. REQUIRED before anything below --
# see the note above; the committed cache is 21-joint and cannot be used with this robot.
python scripts/rsl_rl/precompute_bones_cache.py --out clips/bones_cache_locomotion.h5

# flat clips (DP), and the box / stair traversal families
python scripts/rsl_rl/generate_mm_clips.py  --bones-cache clips/bones_cache_locomotion.h5 ...
python scripts/rsl_rl/generate_box_clips.py --bones-cache clips/bones_cache_locomotion.h5 ...

# whole-library driver (the DP counterpart of generate_mb_library.py)
python scripts/rsl_rl/generate_terrain_library.py ...

# matcher comparison (vanilla vs beam vs DP) — the actual ablation harness
python scripts/rsl_rl/mm_experiment.py ...
```

`trajectory` cost at `cut_penalty=3` is the tuned configuration (velocity cost has a
foot-slide failure mode).

## The PAPER ablation: motion matching as a third arm

`mb_baseline.py` scores our clips against stock Motion Bricks. Motion matching joins as a third
arm of the same comparison, and the only thing that makes the three comparable is that all of
them are driven by **one command profile**, taken from our own flat clips:

```bash
# 1. the MM arm, on OUR commands (not its own random schedule)
python scripts/rsl_rl/generate_mm_clips.py \
    --source hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
    --bones-cache clips/bones_cache_locomotion_29j.h5 \
    --out clips/mm_ablation_v1.h5 --device cpu \
    --command-from clips/g1_mb_v6f_decor.h5 --command-n-clips 30 --command-seed 1234

# 2. score all three
python scripts/rsl_rl/mb_baseline.py --stage batch \
    --clip clips/g1_mb_v6f_decor.h5 --n-clips 30 --seed 1234 \
    --mm-arm clips/mm_ablation_v1.h5 --out-dir <paper_data>/... \
    --mb-root /abs/path/to/GR00T-WholeBodyControl/motionbricks
```

**`--command-n-clips` and `--command-seed` MUST equal the scorer's `--n-clips` and `--seed`.**
The selection is `rng.choice(flat, size=n)`, so a different `n` draws a different SET, not a
prefix of the same one -- at `n=3` it picks clips 9929/9936/9963 where `n=30` starts at 7312.
Getting this wrong would pair clip *i* of one arm against a different motion in the other, with
both arms running and both reporting. It cannot happen silently: the generator records the
indices it used and `mb_baseline` refuses a mismatch, naming the rebuild command. A library with
no `command_source` attr is refused outright as the authored MM family, which draws its own
speeds and is not comparable to anything.

Both arms are scored on velocity tracking (bucketed by commanded vx, clip-clustered bootstrap,
paired) and on **joint jerk RMS**, the jitter number. Read the jitter table's notes before
quoting it: ours and MM are native 50 Hz and directly comparable, the Motion Bricks arm is
resampled from 30 fps and its cost is measured per run by the control row.

## What moved out

The geometry, ingest and clip-writing modules this package used to hold are **not**
deprecated — the MB pipeline depends on them — so they moved to
`motion_gen/terrain/`, `motion_gen/ingest/`, and `motion_gen/{clip_output,gen_parallel,playback}.py`.
`G1_JOINT_NAMES` moved to `motion_gen/joint_names.py`.
