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

`play.py` also writes the deploy artifacts: `exported/policy.pt`, `policy.onnx`,
`policy_parameters.yaml`, and (with `--fm_checkpoint`) `tracker.onnx` + `fm.onnx`. Pass
`--no_deploy_onnx` to skip the ONNX pair.

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
python transfer/sim/generate_stairs_scene.py --robot g1_29j   # symmetric up/platform/down lanes
cd transfer/sim && python gen_tracker_runner.py --load_run <run_dir> ...
```

Deeper Isaac-side debugging (height-scan depth images, reference ribbons):

```bash
python scripts/rsl_rl/debug_episodes.py --task G1-teacher-clf-terrain-play ... --out bundle.npz
python scripts/rsl_rl/view_debug_episodes.py bundle.npz     # viser replay + plots, no Isaac
```

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
| `G1-tracker-clf-fm-terrain-play` | Closed-loop deploy: the teacher tracks the FM predicted from live robot state. **Currently broken — see below.** |

There is no separate tracker *training* task — the tracker policy **is** the teacher.

### Teacher observations

The teacher's actor takes **two** observation groups and routes them to two branches:

| Group | Contents | Path |
|---|---|---|
| `policy` | 5 proprioception terms, each stacked over `PROPRIO_HISTORY_LENGTH` = 10 consecutive 50 Hz control steps (current + 9 back), plus the height scan (no history) | straight to the trunk |
| `reference` | `reference_window` (base pose/vel + joint pos/vel + ee positions at `REFERENCE_NUM_STEPS` = 11 samples, `REFERENCE_STRIDE` = 5 steps apart → 0.0 … 1.0 s) and the clean `command_window` | through a `[1024, 512, 512]` MLP encoder |

The three constants live at the top of `g1_teacher_env_cfg.py`; setting either count to 1
degenerates to current-step-only. The critic sees the same history and the same
(uncorrupted) window, plus the privileged world-frame CLF vectors.

An **optional** auxiliary loss reconstructs the reference window from the encoder latent.
It is off by default so it can be ablated, and needs no code change to turn on:

```bash
python scripts/rsl_rl/train.py --task=G1-teacher-clf-terrain --headless --clip <lib.h5> \
    agent.actor.reconstruction=true agent.algorithm.reference_reconstruction_coeff=0.1
```

`AuxPPO` raises if the coefficient is non-zero without `actor.reconstruction=true`.

> **`G1-tracker-clf-fm-terrain-play` does not currently run.** It shares the teacher's
> observation config, and `reference_window` calls two `FmTrackingCommand` accessors that
> deliberately raise (`get_future_joint_state` / `get_future_world_frames`) because they
> would return clip data while the FM drives. Separately, the FM only predicts 0.78 s
> (`FlowMatchingConfig.future_len = 40`) versus the teacher's 1.0 s window. Reviving it
> needs both: implement the per-env FM/clip select in `fm_tracking_cmd.py`, then raise
> `future_len` to ≥ 51 and **re-record + retrain** the FM (windows are cut at record time).
> The MuJoCo/hardware deploy is blocked on the same change plus observation history in
> `transfer/deploy/gen_tracker.py:_assemble` — see the TODO there.

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
`tracker.onnx` / `fm.onnx`), so export before running either. To add a robot, drop its sim
files in `transfer/sim/robots/` and add a runner.

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
To run the sim2sim transfer, you will to install these dependencies in your conda environment:
- `pygame`
- `mujoco`
- `huggingface_hub`

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

