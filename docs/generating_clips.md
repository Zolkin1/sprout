# Generating a clip library

The clip library (`.h5`) is the spine of the whole stack: it carries both the reference motions and
the terrain they were generated on, so one file defines what the robot tracks AND what it walks on.
This document is how to build one.

Everything here is driven by **`scripts/rsl_rl/generate_mb_library.py`**, which orchestrates 13
script invocations across two conda environments. You should not need to run the stage scripts by
hand; where you might want to, it is called out.

---

## TL;DR

```bash
cd <repo root>            # or your worktree root -- both work; paths are repo-relative
source env_isaaclab/bin/activate
export PYTHONPATH=source/robot_rl:source/motion_gen

# smoke build -- 15 tiles, ~30 s, run this after ANY change to the pipeline
python scripts/rsl_rl/generate_mb_library.py \
    --config   configs/clip_gen/quick.yaml \
    --work-dir mb_raw/quick \
    --out      clips/g1_mb_quick.h5 \
    --source   hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
    --library  trajectories/skills/library_v2 \
    --mb-root  /abs/path/to/GR00T-WholeBodyControl/motionbricks

# full build -- 200 tiles x 50 clips = 10 000 clips, ~20 min, needs the GPU
python scripts/rsl_rl/generate_mb_library.py \
    --config   configs/clip_gen/g1_mb_v6.yaml \
    --work-dir mb_raw/lib_v7 \
    --out      clips/g1_mb_v7.h5 \
    --source   hf/trajectories/retargeted/2026-06-13_15-10-58_merged \
    --library  trajectories/skills/library_v2 \
    --mb-root  /abs/path/to/GR00T-WholeBodyControl/motionbricks
```

**The finished library is `clips/g1_mb_v7_decor.h5`, not `clips/g1_mb_v7.h5`.** Stage d2 decorates
the compose output and writes a new file; `--out` names the *undecorated* intermediate. Point
training, recording and every viewer at the `_decor` one. An undecorated library has no stair-width
variation at all — every platform is exactly `size_y` — which is what stage e refuses.

All six flags above are **required**. There are no defaults, deliberately: a stale library or
source silently trains on motion nobody solved for this robot, and the solved-library identity is
stamped into the finished `.h5` as provenance.

---

## The stages

`--stage` runs one alone (e.g. to run the GPU stage on the machine that has a GPU). A killed build
resumes where it stopped, but **skip-if-done is not uniform**: `0`, `a`, `c` and `d2` skip on their
output file; `b` skips per *segment* inside `mb_terrain_segments` (`_is_stale` compares the generation
knobs, so a knob change regenerates where a plain resume does not); and `d`, `e` and `f` have no guard
at all — `d` recomposes `--out` from the per-family `.h5` on every run (cheap, fully derived), and
`e`/`f` are reports that should always describe the library in hand. The cost either way is that a
stale intermediate is reused **silently**, which is why a fresh `--work-dir` is the right move after a
recipe change.

| stage | script | env | GPU | produces |
|---|---|---|---|---|
| `0` | `export_stride_bank.py` | env_isaaclab | no | the exact-loop stride bank |
| `a` | `mb_plan_library.py` | env_isaaclab | no | tiles, per-clip plans, MB requests |
| `b` | `mb_terrain_segments.py`, `mb_flat_transients.py` | **motionbricks** | **yes** | the raw MB in-betweenings |
| `c` | `mb_assemble_library.py`, `mb_pure_stair_clips.py`, `mb_flat_clips.py` | env_isaaclab | no | per-family `.h5` |
| `d` | `compose_terrain_library.py` | env_isaaclab | no | ONE library |
| `d2` | `decorate_terrain.py` | env_isaaclab | no | **the finished `_decor.h5`** |
| `e` | `mb_qc.py --gate` | env_isaaclab | no | PASS/FAIL — a bad library fails the build |
| `f` | `clip_library_report.py` | env_isaaclab | no | figures + `report.json` |

Stage b is the only one that needs Motion Bricks and the only one that needs a GPU. It is invoked
through `conda run -n motionbricks`; each of its two scripts loads the model **once** and generates
everything it owns, which is the difference between one checkpoint load and thousands.

Pure stair tiles need no Motion Bricks at all — a pure tile is one long platform-less staircase with
no approach or exit, so there is nothing to in-between. They are built in stage c and folded into
stage d automatically.

---

## Traps

**`--mb-root` — the DRIVER handles the worktree case, the stage scripts do not.** The default is
written relative (`../GR00T-.../motionbricks`), which resolves from the main checkout and so points
at nothing from a git worktree. `generate_mb_library._resolve_mb_root` covers that: it tries the
argument, then walks up from the repo looking for the clone, and prints what it resolved to
(`[mb] motionbricks root: ...`). Read that line — it means a typo'd `--mb-root` is silently
corrected rather than reported, so if you meant a *different* clone, check it took the one you
meant. Running `mb_terrain_segments.py`, `mb_flat_transients.py` or `stitch_lab.py` **by hand**
has no such fallback: pass an absolute path there.

**The `motionbricks` env has no h5py.** Anything crossing into it must go as `.npz` — that is why
plans are `.npz` and why `mb_baseline.py` has a separate `export` stage. Do not add an HDF5 read to
a stage-b script.

**Skip-if-done hides recipe changes.** Stages check for their output file, not for whether the
recipe that produced it still matches. After changing `configs/clip_gen/*.yaml`, use a fresh
`--work-dir` and a fresh `--out`, or you will compose old segments under a new recipe.

**Run from the repo root** (or your worktree root -- either is fine, they are equivalent
here). Several paths in the recipes and defaults are repo-relative.

---

## After the build

Stage e (the gate) and stage f (the report) run automatically and describe the **finished**
`_decor.h5`. The gate fails the build on an unambiguous defect — a root speed nothing correct
reaches, a pelvis outside the range a standing robot can occupy, an undecorated library — because a
9.8% blow-up once shipped in `g1_mb_v8.h5` and was found only by watching a clip render.

Read `<work-dir>/report/` next. The figure to read first is `commanded_vx.png` (the command marginal
against the declared envelope) and then `speed_by_kind.png`, which separates exact loops from
in-betweenings — every root-speed blow-up this pipeline has had lived in the in-betweening
distribution.

**Then look at clips.** Nothing above catches a motion that is merely bad:

```bash
python scripts/rsl_rl/view_clip_viser.py --lib clips/g1_mb_v7_decor.h5 --clip 0
python scripts/rsl_rl/view_terrain_viser.py --lib clips/g1_mb_v7_decor.h5 --traj 3
```

Clutter (`build_clutter_occupancy.py` → `author_clutter.py`) is **not** part of the build and is off
by default; see CLAUDE.md for why the RL terrain is deliberately uncluttered.

---

## Changing the distribution

Every knob lives in `configs/clip_gen/*.yaml` and is forwarded by the driver to the stage that
consumes it. The recipe is loaded strictly in both directions — a missing key and an unknown key are
both hard errors — and `test_recipe_forwards_every_parameter.py` fails the build if a stage grows a
flag the recipe does not state.

That test is a **heuristic**: it checks that a flag string appears somewhere in the driver, not that
the two declarations agree, and not that the flag reaches *that particular* stage invocation. Three
real gaps hid behind exactly that — `--seed` and `--alpha-max` never reached the segments stage at
all, and `--seam-blend-s` reached only one of the two flat calls.

So every build now writes **`<work-dir>/stage_commands.txt`**: the resolved argv of all 13 stage
invocations, appended per run with a timestamp. When adding a knob, forward it and then grep that
file — it answers "did this stage actually receive it" for the build in hand, which the static test
cannot.

Two recipes are tracked: `g1_mb_v6.yaml` (full) and `quick.yaml` (the smoke build). **Keep them in
sync** for anything that is not a scale parameter — the smoke build is only useful as a smoke build
if it exercises the same code paths.

---

## Debugging a bad seam

When a transition into or out of terrain looks wrong, `scripts/rsl_rl/stitch_lab.py` is the tool:
it plans one tile at every possible hand-off frame, lets you scrub them in viser, and then runs the
**real** pipeline for the frame you pick, so what you see is the clip rather than a preview.

```bash
python scripts/rsl_rl/stitch_lab.py --stage sweep --out /tmp/stitch \
    --skill stairs_up --riser 0.15 --tread 0.25 --n-steps 4 --terrain-speed 0.69
python scripts/rsl_rl/stitch_lab.py --stage view  --out /tmp/stitch
```

It also exposes the two knobs that turned out to matter — the transient's token horizon and whether
Motion Bricks owns the root — so a proposed change can be looked at before it becomes a rebuild.

---

## Related

- `CLAUDE.md` — the conventions, and why the terrain is uncluttered
- `source/motion_gen/motion_gen/motion_matching/README.md` — the DEPRECATED matcher, kept for
  ablations. It is **blocked on a stale 21-joint bones cache** that must be rebuilt before any
  matcher runs; that README says which command and why.
- `scripts/rsl_rl/mb_baseline.py` — scores this library's tracking against the stock Motion Bricks stack
