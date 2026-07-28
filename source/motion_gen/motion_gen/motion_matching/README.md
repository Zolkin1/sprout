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
| `features.py` | the matching feature space (trajectory + pose features) and its FK |
| `mm_clip_generator.py` | clip assembly around a matcher result |
| `slope_rotate.py` | rotates flat clips onto a slope — shelved (works only straight up/down-slope; cross-slope and turning clips lean sideways) |
| `trajopt_ingest.py` | trajopt/Bezier motion source, an alternative to the bones seed |

## Running it

Its drivers live in `scripts/rsl_rl/` and each carries a `DEPRECATED:` banner naming the MB
equivalent:

```bash
# once: build the bones feature cache every matcher needs
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

## What moved out

The geometry, ingest and clip-writing modules this package used to hold are **not**
deprecated — the MB pipeline depends on them — so they moved to
`motion_gen/terrain/`, `motion_gen/ingest/`, and `motion_gen/{clip_output,gen_parallel,playback}.py`.
`G1_JOINT_NAMES` moved to `motion_gen/joint_names.py`.
