# Sprout
## :construction: This repo is under active development :construction:
Since we actively develop on this repo we may push major/breaking changes.
We do not guarantee any kind of API/environment stability right now, although this could come in the future.
That being said, we attempt to keep the repo as usable and documented as possible given its developmental status.

## Overview

This project is a set of tools for end-to-end development of RL for robots. Specifically, we support:

- RL development using IsaacLab and IsaacSim.
- sim2sim transfer using Mujoco.
- Hardware transfer using Obelisk (ROS2) - note that ROS2 is NOT a dependency of this project - the hardware
interface can be run through a docker/dev-container provided in this repo. See below for more information.

## Installation
We use a [`uv`](https://docs.astral.sh/uv/) virtual environment named `env_isaaclab` (at the repo root) with a
**pip-installed Isaac Sim** and an **editable Isaac Lab** checked out as a sibling directory. This repo currently tracks
Isaac Lab **`v3.0.0-beta`**. For the exact Isaac Sim pip command for your version/CUDA, see the Isaac Lab pip-install
guide — both the
[`v3.0.0-beta` page](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/setup/installation/pip_installation.html)
(the version we track) and the
[latest page](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) are useful
references. The steps below reflect the setup this repo is currently developed against.

Prerequisites: a recent NVIDIA driver (the pinned `torch` wheels target CUDA 12.8), [`uv`](https://docs.astral.sh/uv/),
and [`git-lfs`](https://git-lfs.com/) (run `git lfs install` once). Asset files (`*.usd`, `*.pt`, `*.jit`, ...) are
stored with **Git LFS**, so clone with LFS enabled.

1. Mirror the directory layout — Isaac Lab lives next to (not inside) this repo:

    ```bash
    mkdir -p Project-Isaac-RL && cd Project-Isaac-RL
    ```

2. Clone this repo (with LFS) and Isaac Lab, then check out the pinned Isaac Lab tag:

    ```bash
    git clone git@github.com:Zolkin1/sprout.git
    git -C sprout lfs pull
    git clone https://github.com/isaac-sim/IsaacLab.git
    git -C IsaacLab fetch --tags
    git -C IsaacLab checkout v3.0.0-beta
    ```

3. Create and activate the `uv` venv (Python 3.12) at the sprout root:

    ```bash
    cd sprout
    uv venv env_isaaclab --python 3.12
    source env_isaaclab/bin/activate
    ```

4. Install Isaac Sim (pinned to match this repo) from NVIDIA's pip index. The two extra `uv` flags are required:
   `--index-strategy unsafe-best-match` lets `uv` pull dependencies such as `mujoco-usd-converter` from PyPI while taking
   the Isaac Sim wheels from NVIDIA's index, and `--prerelease=allow` permits pinned pre-release deps like
   `tinyobjloader==2.0.0rc13`. (Plain `pip` does both implicitly, which is why the online guide doesn't mention them.)

    ```bash
    uv pip install --upgrade pip
    uv pip install 'isaacsim[all,extscache]==6.0.0' \
      --extra-index-url https://pypi.nvidia.com \
      --index-strategy unsafe-best-match \
      --prerelease=allow
    ```

5. Let Isaac Lab install all its extensions and RL libraries (e.g. `rsl-rl-lib`) in editable mode:

    ```bash
    (cd ../IsaacLab && ./isaaclab.sh -i)
    ```

6. Install this project in editable mode:

    ```bash
    uv pip install -e source/robot_rl
    ```

7. Verify the install by listing tasks and running a short headless training. The task
   needs a clip library (`--clip`); see [The pipeline](#the-pipeline) for how to build one.

    ```bash
    python scripts/list_envs.py --keyword G1        # expect 5 G1-* tasks
    python scripts/rsl_rl/train.py --task=G1-teacher-clf-terrain --headless \
        --num_envs=64 --max_iterations=2 --clip <lib.h5> --clip_device cpu
    ```

## The pipeline

Everything runs through one chain. Each stage's output is the next stage's input:

```
motionbricks clip generation  ->  teacher RL  ->  flow-matching generator  ->  tracker  ->  sim2sim  ->  hardware
   generate_mb_library.py         train.py         flow_matching/train.py      play.py     compare_sim2sim.py
```

The **clip library** (`.h5`) is the spine. It carries the reference motions AND the terrain
they were generated on, so a single file defines both what the robot tracks and what it walks
on. Every task takes it via `--clip`; there is deliberately no default, because a stale
library silently trains on the wrong terrain.

List the tasks:

```bash
python scripts/list_envs.py --keyword G1
```

### 1. Generate a clip library

Motion Bricks runs in its own conda env (GPU), so generation is staged: the driver
shells out and skips stages that are already done.

```bash
python scripts/rsl_rl/generate_mb_library.py --out clips/g1_mb_v5.h5 [--stage N] [--quick]
```

It orchestrates `export_stride_bank` -> `mb_plan_library` -> `mb_terrain_segments` /
`mb_encode` (in the `motionbricks` env) -> `mb_assemble_library` / `generate_mb_clips` /
`mb_distractor_clips` -> `compose_terrain_library`.

Then check what came out:

```bash
python scripts/rsl_rl/mb_qc.py --lib clips/g1_mb_v5.h5        # objective quality report
python scripts/rsl_rl/inspect_clip.py --mode timeseries --lib clips/g1_mb_v5.h5 --clip 0
python scripts/rsl_rl/view_clip_viser.py --lib clips/g1_mb_v5.h5    # interactive
python scripts/rsl_rl/view_terrain_isaac.py --lib clips/g1_mb_v5.h5 # per-family tile split
```

### 2. Train the teacher

```bash
python scripts/rsl_rl/train.py --task=G1-teacher-clf-terrain --headless \
    --num_envs=4096 --clip clips/g1_mb_v5.h5 --run_name my_run
```

`--clip` sets the terrain importer's mesh AND the tracked clips together; both must read the
same file or the tile terrain ids won't line up with the clips'. Add `--run_name` to label
the log dir. Large libraries need `--clip_device cpu` on a small GPU.

Play it back:

```bash
python scripts/rsl_rl/play.py --task=G1-teacher-clf-terrain-play \
    --load_run <run_dir> --clip clips/g1_mb_v5.h5 --clip_device cpu
```

`play.py` also writes the deploy artifacts: `exported/policy.pt`,
`policy_parameters.yaml`, and (with `--fm_checkpoint`) `tracker.onnx` + the generator,
named for the FM checkpoint it came from (`<fm run>_fm.onnx`) so a run's `exported/` can
hold every generator trained against that teacher instead of overwriting one file. Pass
`--no_deploy_onnx` to skip the ONNX pair. `tracker.onnx` is the ONLY ONNX form of the
tracker — it is `policy.pt` re-exported self-contained, batch-dynamic and with the
deployment parameters embedded, and it is what MuJoCo and the robot load.

### 3. Train the flow-matching generator

Record teacher rollouts, then fit the FM to them:

```bash
python scripts/rsl_rl/record_data.py --task G1-teacher-clf-terrain-play \
    --load_run <run_dir> --clip clips/g1_mb_v5.h5 --num_windows 200000 --num_envs 256
python scripts/flow_matching/train.py --data <windows.h5> --wandb
```

`--num_windows` streams only the FM snap windows to a chunked `.h5` (bounded RAM); omit it
for the older buffered per-step `.npz`. `command_distribution_stats.py` reports the
command-dynamics split of either format.

**Two families are excluded from FM training by default**, both because the generator cannot
be asked to predict what it cannot see:

* `--skip_distractors` (on) drops the distractor windows;
* `--skip_families family_jump_off` (on) drops the BACKWARDS jump-off-a-box family. The robot
  has two forward-facing depth cameras and nothing behind, so the drop it steps off is not
  observable. `family_jump_off_FRONT` is the forward dismount, is observable, and is kept.

Both are train-time switches — every window carries its family and its distractor tag, so
A/B-ing either costs no re-record. `--skip_families ""` trains on everything. A family you
NAME must exist in the recording (a typo would otherwise drop nothing, which reads as
success); a family carried only by the default is skipped where absent with a printed line,
because a flat-only library has no terrain manifest and therefore no family table at all.

Note the two flags spell the same family differently and deliberately: `--skip_families`
wants the recording's own `family_names` entry (`family_jump_off`), while the eval harness's
`--exclude_families` normalizes (`jump_off`). The raw names are what the recording records.

**What a recording contains is a choice, not an accident.** Three knobs decide it, and the
defaults are the ones a dataset wants rather than the ones training wants:

* `--start_step 0` (default) starts every episode at its clip's first frame, so each clip
  is walked once end to end and the windows cover clip PHASE uniformly. Training's sampled
  start (`--start_step -1`) run to the clip end visits step `t` with probability `~t/len`,
  which measured out as 1.1% of windows in the first phase decile against 18.1% in the
  eighth.
* `--clip_draw cycle` (default) draws clips from a shuffled queue per terrain group, so
  every clip is used once before any repeats. With-replacement sampling covers only
  `1-exp(-E/C)` of a group — a 200k-window recording spanned 6393 of 10000 clips.
* the MIX across clip families is set afterwards, offline:

```bash
python scripts/flow_matching/compose_dataset.py <windows.h5> --list
python scripts/flow_matching/compose_dataset.py <windows.h5> -o mix.h5 --by terrain \
    --mix crossing=40000 --mix on_terrain=40000 --mix flat=40000
```

`compose_dataset.py` selects rows by quota, spreading each evenly over the bucket's clips
and evenly along each clip's timeline. It decides from the recording's small index columns,
so record ONCE broadly and compose as many mixes as are worth trying. `--by` picks what a
bucket IS, and the choice matters because the useful questions live at two levels:

| `--by` | buckets | describes |
| --- | --- | --- |
| `type` | flat / terrain / transition / box | the CLIP's terrain (library `terrain_type_code`) |
| `family` | `family_stairs_up`, `family_jump_off`, … | the CLIP's family (recorded tag) |
| `terrain` | flat / crossing / on_terrain | what the WINDOW's own span does |

`terrain` is the one no clip-level axis can express: a transition clip is a flat approach,
then the crossing, then the stair run, so `--by type --mix transition=40000` buys mostly
ordinary flat walking — measured on a real recording, a transition clip's windows are 42%
flat, 46% crossing, 12% on-terrain. It is classified from the library's authored per-frame
anchor mask (`anchor_mask_is_pinned` = the reference is registered to terrain geometry
rather than to the robot), not from a height threshold. Axes compose:
`--by type,terrain --mix box/crossing=20000` asks for the box mounts specifically.

Two things to size the recording against, both measured on `g1_mb_v5_body`:

* **Its natural mix is set by TILE counts, not by anything you asked for** — 31% flat, 34%
  box, 20% transition, 15% terrain, because every env emits a window at the same cadence
  and envs are spread over the tiles. That is the distribution the quotas exist to override.
* **Flat is the group that binds clip coverage.** Flat clips are 999 frames against a stair
  clip's ~350, so a flat env gets through its 3000 clips far slower than a stair env gets
  through its 50-clip group. At 200k windows the terrain/transition/box groups are already
  fully covered and flat is at 57%; **~353k windows** is where every clip in the library has
  been used at least once.

The composer can only spread what the recording took, so phase uniformity comes from
`--start_step 0` — composing an old random-start recording still yields the ramp.

### 4. Deploy the tracker, in Isaac and in MuJoCo

The tracker is the *same* teacher policy, tracking the FM instead of the clip:

```bash
python scripts/rsl_rl/play.py --task=G1-tracker-clf-fm-terrain-play \
    --load_run <run_dir> --fm_checkpoint <fm.pt> --clip clips/g1_mb_v5.h5 --clip_device cpu
```

For a like-for-like Isaac-vs-MuJoCo comparison, `compare_sim2sim.py` builds a single-clip
library **and a MuJoCo scene from that clip's own tiles**, so the two sims get identical
geometry rather than a parametric staircase that merely resembles it:

```bash
python scripts/rsl_rl/compare_sim2sim.py --stage prep   --lib clips/g1_mb_v5.h5 --out-dir logs/sim2sim
python scripts/rsl_rl/compare_sim2sim.py --stage isaac  --case flat   --out-dir logs/sim2sim ...
python scripts/rsl_rl/compare_sim2sim.py --stage mujoco --case flat   --out-dir logs/sim2sim ...
python scripts/rsl_rl/compare_sim2sim.py --stage plot   --case flat   --out-dir logs/sim2sim
```

For a free-roaming joystick playground instead of one clip, generate a standalone scene and
drive it directly:

```bash
python transfer/sim/generate_stairs_scene.py   # symmetric up/platform/down lanes
cd transfer/sim && python gen_tracker_runner.py --load_run <run_dir> ...
```

Deeper Isaac-side debugging (height-scan depth images, reference ribbons):

```bash
python scripts/rsl_rl/debug_episodes.py --task G1-teacher-clf-terrain-play ... --out bundle.npz
python scripts/rsl_rl/view_debug_episodes.py bundle.npz     # viser replay + plots, no Isaac
```

#### Debugging the generator: `fm_debug.py`

When the tracker follows the clips well but degrades on the FM, this is the tool. It freezes
**three** clips (a stair transition, a flat walk, a box jump-up) into a small library, rolls
the tracker over them twice, plots both, and replays both in viser:

```bash
python scripts/rsl_rl/fm_debug.py --lib clips/g1_mb_v5_body.h5 \
    --checkpoint logs/rsl_rl/g1/<teacher>/model_XXXX.pt \
    --fm_checkpoint logs/flow_matching/<fm_run>/best.pt
# then iterate without touching the GPU:
python scripts/rsl_rl/fm_debug.py --stage plot --out logs/fm_debug/<run>
python scripts/rsl_rl/fm_debug.py --stage view --out logs/fm_debug/<run>
```

The two rollouts differ in exactly one field, `commands.traj_ref.reference_source`:

* **`clip`** — the policy tracks the pre-made clip (the known-good baseline) while the
  generator runs **passively** every step and its output is recorded. So both references are
  evaluated at *identical* robot states, and a difference between them is the generator's
  rather than a consequence of the robot having already diverged.
* **`fm`** — the policy tracks the generator, from the same spawn pose and the same
  `target_vel` profile.

Everything is persisted in `<out>/fm_debug.json`, so no argument is retyped across stages
(and no two stages can disagree). Read the output in this order: `contract_<mode>.txt`
(the pre-rollout agreement report — checkpoint vs live obs layout, horizon vs window,
generator frames vs command frames, the depth-camera contract),
`plots/<mode>/obs_clip_fm_diff_env*.csv` (the reference-observation columns where the two
sources diverge most, ranked), `plots/compare/` (where the rollouts part), then the viewer.

In viser, the load-bearing comparison is the **FM horizon** layer against the **reference
window** layer: the window is the generator's prediction after the whole conversion chain
(world lift → base-relative differencing → observation packing) and back, so they must
coincide. A constant offset or a yaw twist between them is a frame bug in
`mdp.reference_window`; agreement moves the search to the generator itself. `--corrupt_obs`
re-enables training-time observation noise, which the play trims switch off, so the
"observed" window variant shows how much of the reference the noise eats.

### 4b. Evaluate: `scripts/rsl_rl/eval/`

The measurement no training curve makes: how often the policy survives, how long failures
take, how well it tracks the commanded velocity, and whether it gets through the terrain --
split by terrain family, under three domain-randomization regimes, for the clip arm and the
generator arm side by side.

Three stages sharing one manifest, so nothing is retyped and no two stages can disagree:

```bash
# 1. PLAN (no simulator): pick tiles and clips, build a self-contained subset library
python scripts/rsl_rl/eval/evaluate.py --stage plan \
    --lib clips/g1_mb_v5_body_mode11_decor.h5 \
    --load_run <teacher run> \
    --fm_checkpoint gen_a=logs/flow_matching/<fm run>/best.pt \
    --num_envs 256 --out logs/terrain_eval/<name>

# 2. GRID: run every cell that is not already complete, one process each
python scripts/rsl_rl/eval/evaluate.py --stage grid --out logs/terrain_eval/<name> --num_envs 256

# 3. REPORT (no simulator, and `--stage reduce` needs no matplotlib either)
python scripts/rsl_rl/eval/report.py logs/terrain_eval/<name>
python scripts/rsl_rl/eval/report.py logs/terrain_eval/{a,b}   # comparison
```

`--quick` (3 tiles/family, 1 clip/tile) smoke-tests the whole pipeline in a couple of minutes.

**A cell is `(reference_mode, dr_preset, fm_label)`.** `--fm_checkpoint` is repeatable, and the
generator is a cell DIMENSION rather than a per-run constant: several generators are compared
against one tracker with the clip arm run ONCE and shared, so every cell is paired on clip id
by construction and the grid is `P + P*G` cells instead of `P*(1+G)`.

**Both arms run on the tracker task**, the clip arm with `reference_source=clip`, so "success"
means the same thing in each (survived the clip without falling). `--clip_task` runs the clip
arm on a teacher task instead, where success additionally requires not tripping the
clip-graded bounds; the report then prints the two as separate blocks rather than on one axis.

Four things the harness does that are easy to get wrong by hand:

* **It adds `clip_end` to the terminations.** The tracker task deliberately has none, so
  without it every episode runs the full 20 s regardless of clip length -- a 300-frame
  distractor clip becomes 6 s of motion plus 14 s of holding a frozen pose, and in the FM arm
  14 s of being commanded the final frame's velocity, walking off a finite tile.
* **It reallocates envs across tile columns and re-dispatches on demand.** An even split
  leaves the 20 s flat lane setting the wall clock; work-conserving dispatch runs at 91-99% of
  the perfectly-balanced time and works even with fewer envs than columns.
* **DR presets are built from the canonical event configs**, not from the task flavour: the
  `-play` trims are PARTIAL (they null the reset-mode friction, both corruption flags and the
  camera terms, and leave all seven startup physical terms firing). `ood` widens each range
  about its midpoint with a declared table, clamped at zero, and asserts the result is never
  narrower than training's -- a blind widen turns `static_friction (0.3, 1.6)` into
  `(-0.025, 1.925)`.
* **Five of the eight physical DR terms re-draw every episode**, giving each episode an
  independent draw so `robots_per_clip` really is the DR sample size. The other three cannot:
  `add_joint_default_pos` adds to the live buffer and would random-walk, and the two CoM terms
  raise on a partial reset. Which is which is recorded per term.

**Read the report in this order.** `DEFINITIONS IN FORCE` (what "success" meant, which stratum,
which interval), then `CELL INVENTORY`, then `HEADLINE`, then `BY FAMILY`, then the paired
contrasts. Every rate is **clip-averaged with an interval bootstrapped over TILES**, and
carries the level it came from: a binomial interval over pooled episodes has an actual coverage
of 14-70% against a nominal 95% on this design, because 25 draws share a clip and two clips
share a tile.

Two readings the report will not let you take by accident:

* **Terrain completion is degenerate on four of the nine families.** `flat` has no
  terrain-pinned frame at all; `distractor`, `stairs_up_pure` and `stairs_dn_pure` are pinned
  to their clip's last frame, so "got through the terrain" IS "survived". Those print `--`, and
  `terrain_progress` -- how far along the pinned span the robot got, denominated by the
  REACHABLE span so a success scores exactly 1.0 -- carries the information instead.
* **`*/none` is the noise floor, not a determinism check.** Measured: with every randomization
  term nulled, the observation noise off and the clip pinned, two envs on the same clip and the
  same tile still diverge. It is not tile sharing (one env per tile gives the same spread), so
  it is simulator-level. That cell is the resolution below which no difference between any two
  other cells means anything.

Artifacts, per run: `plan.json`, `manifest.json` (git state, checkpoint sha256s, generator
contracts, both library fingerprints, resolved DR ranges), `eval_clips.h5` (the subset, so the
run is self-contained), `git/`, and per cell `episodes.h5` (one row per episode, 57 columns)
plus `cell.json` (resolved DR, coverage proof, throughput). `report/` holds `report.txt`,
`report.json` -- the text is rendered FROM the json, so nothing downstream ever parses prose --
`summary_by_family.csv`, and `figures/` as png + svg + a caption sidecar carrying that figure's
definitions.

### 5. Hardware

Upload an exported run to HuggingFace; the obelisk controller pulls from there.

```bash
python scripts/rsl_rl/export_to_hardware.py --load_run <run_dir> \
    --hf_repo_id <username/repo> --folder_name <name>
```

The policy must already be exported (step 2). See [Obelisk Transfer](#obelisk-transfer) for
the ROS2 side.

## RL Tasks

Five tasks, all terrain-based. The scene ground is a `ClipTerrainImporter` meshed from the
clip library, so a "flat" run is just a library whose tiles are flat — there is no separate
flat task.

| Task | Purpose |
|---|---|
| `G1-teacher-clf-terrain` | **Train** the teacher: open-loop clip tracking, world-frame CLF. |
| `G1-teacher-clf-terrain-play` | Play / record / export the teacher. One env per terrain tile. |
| `G1-teacher-clf-terrain-sym` | Symmetry-augmented training. **Currently broken — see below.** |
| `G1-teacher-clf-terrain-sym-play` | Play variant of the above. **Currently broken.** |
| `G1-tracker-clf-fm-terrain-play` | Closed-loop deploy: the teacher tracks the FM predicted from live robot state. |

There is no separate tracker *training* task — the tracker policy **is** the teacher.

### Teacher observations

The teacher's actor takes **two** observation groups and routes them to two branches:

| Group | Contents | Path |
|---|---|---|
| `policy` | 5 proprioception terms, each stacked over `PROPRIO_HISTORY_LENGTH` = 10 consecutive 50 Hz control steps (current + 9 back), plus the height scan (no history) | straight to the trunk |
| `reference` | `reference_window` (base pose/vel + joint pos/vel + ee positions at the 13 `REFERENCE_OFFSETS` = `0, 1, 2, 5, 10, … 50` control steps: the current step, the two right after it, then a 0.1 s tail out to 1.0 s) and the clean `command_window` | through a `[1024, 512, 512]` MLP encoder |

The constants live at the top of `g1_teacher_env_cfg.py`; `PROPRIO_HISTORY_LENGTH = 1` or
`REFERENCE_OFFSETS = (0,)` degenerates to current-step-only. `REFERENCE_OFFSETS` is the one
window contract — the observation, its noise/scale vectors, the truncation margin, the
generator's required horizon and the exported deploy spec are all derived from it, and the
`reference` and `critic` groups' two window terms must all declare the same list (the deploy
runtime refuses a tracker whose two exported specs disagree). The critic sees the same
history and the same (uncorrupted) window, plus the privileged world-frame CLF vectors.

An **optional** auxiliary loss reconstructs the reference window from the encoder latent.
It is off by default so it can be ablated, and needs no code change to turn on:

```bash
python scripts/rsl_rl/train.py --task=G1-teacher-clf-terrain --headless --clip <lib.h5> \
    agent.actor.reconstruction=true agent.algorithm.reference_reconstruction_coeff=0.1
```

`AuxPPO` raises if the coefficient is non-zero without `actor.reconstruction=true`.

### The generator/clip barrier

`G1-tracker-clf-fm-terrain-play` shares the teacher's observation layout, but the clip
supplies it only two things: the **reset spawn pose** (and the terrain meshed from the same
library) and the **`target_vel` command** — the joystick stand-in. Everything the policy
observes is generated. Enforcing that, rather than assuming it:

- the tracker env has **zero reward terms** and no clip-graded termination (only `time_out`
  and `base_fallen`);
- it has **no privileged `critic` observation group at all**, and `play.py` does not restore
  the critic's weights (`FmTrackerPlayRunnerCfg.load_critic = False`). That group held the
  last clip-derived observation (`tracking_error_global`, drift against the globally-placed
  clip), and nothing in a play-only task evaluates a value function. It also makes the task
  **teacher-agnostic**: the actor's two groups have the same layout under either
  `tracking_mode`, so a body-mode and a joint-mode teacher both load here — previously a
  body-mode checkpoint was rejected purely on the critic's width;
- there is **no clip warmup**: before the history window fills, the generator is sampled with
  the learned `null_history` embedding it was trained with — the same thing deployment does;
- `FmTrackingCommandCfg.strict_fm_only` (on by default) fills every `y_des` / `dy_des`
  channel the FM does not write with **NaN**, so an accidental read of clip-derived data
  fails immediately instead of quietly tracking the wrong thing.

The generator's horizon is sized from the re-plan period, not guessed:

```
future_len >= max(REFERENCE_OFFSETS) + 1
           =  50                     + 1  = 51
```

There is no `fm_update_every - 1` slack term: the period is pinned to 1, so the requirement
is exact. Note the bound is the FURTHEST offset, not the sample count — the near samples
(offsets 1 and 2) cost no horizon at all. `FlowMatchingConfig.future_len = 51`.
`FmTrackingCommand` re-derives the requirement from the offsets it is actually asked for and
**raises** if the checkpoint is short — nothing clamps, because a clamp would silently freeze
the tail of the observed window. The same rule applies at record time (windows whose future
would cross the clip end are dropped) and in training (`clip_end_reached` truncates one full
observation window before the clip's last frame).

Observation history off-sim is handled: `_assemble` in `transfer/deploy/gen_tracker.py` pushes
each term through its own window using the `history_length` the export records, so no consumer
restates a window length. What is still unimplemented there is the BODY-tracking FM path — a
tracker trained with `tracking_mode="body"` wants root-frame body targets (`body_pos_ref` /
`body_vel_ref`), which clip playback supplies but the generator cannot, since turning its
predicted joint angles into root-frame link poses needs a torch-free FK. `_assemble` raises
rather than guessing.

> **The two `-sym` tasks do not currently run.** They raise
> `ValueError: Symmetry augmentation has no handler for obs term ...` at the first PPO
> update. The teacher's reference observations (now `reference_window` / `command_window`)
> have no reflection rules in `motion_gen.sagittal_reflector.SagittalReflectionConfig`, and
> the augmentation deliberately refuses to run rather than silently identity-copying the
> mirrored half of the batch. This pre-dates the repo slim-down. To revive them, add both
> terms with their correct reflection — `reference_window` is `[step0(D), step1(D), ...]`
> and must be reflected per block within each step, using
> `mdp.observations.reference_window_layout` for the widths — extend
> `source/robot_rl/tests/test_sagittal_reflector.py`, then smoke-train.

## Deprecated: motion-matching / DP clip generation

The clip generator that preceded Motion Bricks is retained for ablation studies. Nothing on
the live pipeline imports it; see `source/motion_gen/motion_gen/motion_matching/README.md`
for the module map and how to run it. Its drivers each carry a `DEPRECATED:` banner naming
the Motion Bricks equivalent:

`precompute_bones_cache.py`, `generate_mm_clips.py`, `generate_box_clips.py`,
`generate_terrain_library.py`, `generate_mm_toy.py`, `mm_experiment.py`.

`generate_distractor_clips.py` is **partly** deprecated: its CLI is superseded, but
`mb_distractor_clips.py` imports `_box_pool` / `_make_tile` from it so both motion sources
emit byte-identical tile geometry.

## Profiling

`scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` ship with an opt-in profiling harness
(`scripts/rsl_rl/profiling.py`) that breaks the per-step time down by phase — physics vs. the
MDP managers (observations / rewards / terminations / commands / events) vs. the
trajectory-command internals. It is zero-cost when off. Use it to find where time is going
*before* optimizing.

### Quick start

Training — run a short, bounded profile (use a small `--max_iterations`):
```bash
python scripts/rsl_rl/train.py --task=<TASK_NAME> --headless --num_envs=4096 \
    --max_iterations=5 --profile
```

Play — matches your normal play setup; reports on a rolling window:
```bash
python scripts/rsl_rl/play.py --task=<TASK_NAME>-play --num_envs=8 --profile
```

Each prints a per-phase table to the console and appends it to `profile_report.txt` in the
run's log directory.

### Reading the output

The table is split into two parts:

- **top-level** — phases that *partition* the step: `obs`, `reward`, `termination`,
  `command`, `events`, plus an `unaccounted` row that is everything else inside `env.step`
  (physics + render + scene update). On a headless training run, `unaccounted ≈ physics`.
- **detail** — phases *nested* inside the top-level ones, shown as % of the frame but **not**
  subtracted (so they may overlap). This is where the trajectory command is split into
  selection (`traj.pre_update`), Bezier eval (`traj.get_output`), the CLF (`traj.clf`),
  measured outputs (`traj.measured`), and the per-event-term breakdown (`evt.reset.*`,
  `evt.interval.*`).

```
==================== TRAIN per-step breakdown ====================
recorded frames: 118   device: cuda
frame mean: 84.304 ms/frame  (~11.9 FPS)
-- top-level (partition of frame) --
  events                         16.331 ms/frame   19.4%
  command                         4.147 ms/frame    4.9%
  ...
  unaccounted (physics/render)   57.841 ms/frame   68.6%
-- detail (nested; % of frame) --
  evt.reset.reset_on_reference    9.870 ms/frame   11.7%
  traj.measured                   7.283 ms/frame    8.6%
  ...
==================================================================
```

rsl_rl already logs `Perf/collection_time`, `Perf/learning_time`, and `Perf/total_fps` every
iteration — read those alongside the table for the rollout-vs-PPO-update split.

### Flags

`train.py`:
- `--profile` — enable the harness.
- `--profile_report_every N` — print a cumulative snapshot every `N` steps (default `50`; also
  appended to `profile_report.txt`, so a snapshot survives a hard kill). `0` = report only at
  the end.
- `--profile_sync` — synchronous mode (see below).
- `--profile_trace` — also run `torch.profiler` over the (bounded) run, dumping a Chrome trace
  (`profile_trace.json`) and a top-op table. Use with a small `--max_iterations`.

`play.py`:
- `--profile` — enable; rolling report every `--profile_steps` frames (default `100`).
- `--profile_sync` — synchronous mode.
- `--no-debug-vis` — force the trajectory debug markers off, to A/B their cost.
- `--no-logging` — disable per-step data logging, to A/B its cost.

### How it works

The harness times named spans with CUDA events (falling back to the wall clock on CPU) and
also emits NVTX ranges, so the same phase names line up on an
[Nsight Systems](https://developer.nvidia.com/nsight-systems) timeline if you run the script
under `nsys`. It wraps the manager `compute` methods (and the trajectory-command / event-term
functions) at runtime — no Isaac Lab edits. Phase times are read after a single
`torch.cuda.synchronize()`.

To **stop early, use Ctrl-C** (not `kill`) so the final report prints. The periodic snapshots
are written to `profile_report.txt` during the run, so even a hard kill leaves the latest
snapshot on disk.

### Sync mode and a timing caveat

Isaac Lab runs physics/sensors through Warp (a separate CUDA stream) while the timers use
Torch's stream. With the default **async** timers, a phase that is the first to read physics
results can absorb the physics wait across the stream boundary — so a *detail* row can read
larger than its parent. That is an attribution artifact, not real nested cost. When you see
it, re-run with **`--profile_sync`**: it drains the whole device at each phase boundary and
times with the wall clock, so phases nest correctly. Sync mode is slower (it serializes the
pipeline), so read the *proportions*, not the absolute FPS. The script prints
`sync mode: ON/OFF` at startup so you can confirm it engaged.

For **throughput decisions** (e.g. comparing `decimation` or `num_envs`), prefer the
artifact-free `Perf/total_fps` that rsl_rl logs every iteration, and change one knob at a time.

## Generating the USD
IsaacLab's built-in URDF -> USD converter, run from the IsaacLab checkout. The robot is
29-DOF (waist roll/pitch + the 6 wrists unlocked), and `robot_rl/assets/robots/g1_21j.py`
points at the `_29j` USD despite the `g1_21j` module name:
```
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    <sprout>/robot_assets/g1/g1_21j/g1_21j_box_merged_29j.urdf \
    <sprout>/robot_assets/g1/g1_21j/g1_21j_box_merged_29j
```
The MuJoCo / FK counterpart is `transfer/sim/robots/g1/g1_29j.xml`, which
`motion_gen.features.DEFAULT_MJCF` resolves to.

## Copying checkpoints from remote server 
First mount the server to your local desktop
 
```
bash scripts/mount_remote.sh
```

Note: this script might require a few changes as somethings are hard-coded.

## sim2sim Transfer

Policies trained in IsaacLab run in MuJoCo without retraining. See
[step 4 of the pipeline](#4-deploy-the-tracker-in-isaac-and-in-mujoco) — `compare_sim2sim.py`
for a like-for-like comparison against a specific clip, or
`transfer/sim/gen_tracker_runner.py` for a joystick playground on a generated scene.

Both read the artifacts `play.py` exports (`exported/policy_parameters.yaml` +
`tracker.onnx` + a `*_fm.onnx`), so export before running either. With several generators
in one `exported/`, name the one you mean with `--fm_onnx`; the runners refuse to guess. To add a robot, drop its sim
files in `transfer/sim/robots/` and add a runner.

Neither needs Isaac: on a box without `env_isaaclab`, build the small downstream venv
(`bash transfer/setup_venv.sh`, see [Other Dependencies](#other-dependencies)) and pull a
published pair instead of exporting one. `gen_tracker_runner.py` resolves the pair from
`logs/rsl_rl/<experiment>/<run>/exported/`, so point that at the fetched folder:

```bash
RUN=<hf folder>                                  # e.g. a run-named folder in zolkin/robot_rl
python3 transfer/obelisk/scripts/prefetch_policy.py --hf_policy_folder $RUN   # + validates the pair
mkdir -p logs/rsl_rl/g1/$RUN
ln -s "$PWD/transfer/obelisk/g1_control/resource/hf_cache/zolkin_robot_rl/$RUN" \
      logs/rsl_rl/g1/$RUN/exported
cd transfer/sim && python gen_tracker_runner.py --mode eval --command smooth_ramp_running \
    --load_run $RUN --scene basic_scene --headless --total_time 20
```

The run prints mean FM and tracker inference times every 5 sim seconds. For the numbers
that decide a deploy box — p50/p95/p99 and `step_worst` against the control period —
use `transfer/deploy/benchmark.py`, which needs only the two `.onnx` files.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Obelisk Transfer
First, set the environment variable `ROBOT_RL_ROOT` to the path to the `/transfer/obelisk` folder.
Now we can being building the docker container.

### Prerequisites
- Docker and Docker Compose plugin installed
  - Installation guide: [Docker Compose Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
- VSCode with devcontainer support (recommended)

We recommend using VSCode's devcontainer feature to run the Docker environment, although in theory you can run this
as a vanilla docker container.

### Initial Setup (VSCode devcontainer)
1. Navigate to the Obelisk folder in VSCode.
2. Open VSCode command palette (`Ctrl+Shift+P`)
3. Select "Dev Container: Rebuild and Reopen in Container"
4. Choose your preferred configuration:
   - GPU-based (recommended for better performance)
   - No-GPU (if GPU is not available)

At anytime you can open the folder locally by using `Ctrl+Shift+P` then "Dev Container: Open Folder Locally".

At this point you are now inside the docker container and can now use Obelisk and ROS2. Please see the readme in the
Obelisk folder for further instructions.


## Other Dependencies
Everything under `transfer/` that does **not** import IsaacLab — MuJoCo sim2sim and the
torch-free ONNX deploy runtime — runs in its own small venv, built from
`transfer/requirements.txt`:

```bash
bash transfer/setup_venv.sh          # -> env_transfer/  (uses uv if present, else python3 -m venv)
source env_transfer/bin/activate
```

That is the whole install on a box that only plays or benchmarks policies (a Jetson, a
candidate deploy machine, a laptop): no Isaac Sim, no torch. On a training box
`env_isaaclab` already has these and you can keep using it.

The venv is CPU-only by default. To run inference on a GPU (`--providers cuda`), swap in
the matching `onnxruntime-gpu` wheel afterwards — the per-platform recipes, including the
Jetson ones, are in the GPU section at the bottom of `transfer/requirements.txt`.

## Updating IsaacLab
Sometimes you will want to updated the version of IsaacLab you are using. To do this, go to the IsaacLab directory
(where you cloned it). Then pull the version you want from git.
Then in that folder run `./isaaclab.sh --install`.


## Citing
If you use this repo in any publications, please consider citing the relevant paper (see below).

### Walking
For the original CLF-RL idea and basic walking:
```
@article{li_clf-rl_2026,
  author={Li, Kejun and Olkin, Zachary and Yue, Yisong and Ames, Aaron D.},
  journal={IEEE Robotics and Automation Letters}, 
  title={CLF-RL: Control Lyapunov Function Guided Reinforcement Learning}, 
  year={2026},
  volume={11},
  number={3},
  pages={3230-3237},
  doi={10.1109/LRA.2026.3653329}}
```
### Running
```
@misc{olkin2026chasingautonomydynamicretargeting,
      title={Chasing Autonomy: Dynamic Retargeting and Control Guided RL for Performant and Controllable Humanoid Running}, 
      author={Zachary Olkin and William D. Compton and Ryan M. Bena and Aaron D. Ames},
      year={2026},
      eprint={2603.25902},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.25902}, 
}
```
and
```
@misc{olkin_chasing_2025,
      title={Chasing Stability: Humanoid Running via Control Lyapunov Function Guided Reinforcement Learning}, 
      author={Zachary Olkin and Kejun Li and William D. Compton and Aaron D. Ames},
      year={2025},
      eprint={2509.19573},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.19573}, 
}
```
### Stepping stones
```
@misc{dai_walk_2026,
      title={Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds}, 
      author={Min Dai and William D. Compton and Junheng Li and Lizhi Yang and Aaron D. Ames},
      year={2026},
      eprint={2601.06286},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.06286}, 
}
```

### Theoretical Results
```
@misc{olkin2026stabilitycontrollyapunovfunction,
      title={Stability of Control Lyapunov Function Guided Reinforcement Learning}, 
      author={Zachary Olkin and William D. Compton and Aaron D. Ames},
      year={2026},
      eprint={2605.01978},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2605.01978}, 
}
```

