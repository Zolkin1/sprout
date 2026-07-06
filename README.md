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

7. Verify the install by listing tasks and running a short headless training:

    ```bash
    python scripts/list_envs.py --keyword G1
    python scripts/rsl_rl/train.py --task=<TASK_NAME> --headless --num_envs=64
    ```

## Running Tasks
Get a list of available tasks:
```
python scripts/list_envs.py --keyword clf
```
or you could use `G1` as the keyword.

To train a policy run:
```bash
python scripts/rsl_rl/train_policy.py --env_type=<ENV_NAME> --viz none
```

To add a run name add `--run_name=my_run_name`. This will add the name after the date on the folder with the run name.

Note that right now the only RL library that is tested in `RSL_RL`.

To play the most recently trained policy for a given task run:
```bash
python scripts/rsl_rl/play_policy.py --env_type=<ENV_NAME> --log_data --export_policy --viz none
```

for a specific run you can pass in additional config such as `--load_run=<run_dir>`
If you want to play from a specific checkpoint then you can run the play script with `--checkpoint=<checkpoint>` (`<checkpoint>` is something like "model_1800").

For both `train` and `play` you can also specify a number of envs with `--num_envs=###`.

### Preparing for Hardware/Uploading to Hugging Face
We suggest uploading the policy to hugging face when you are ready to run it on hardware so that this repo (the code repo) is kept clean.
You can upload to hugging face automatically with
```bash
python scripts/rsl_rl/export_to_hardware.py --env_type=<env-type> --load_run=<run_dir> --hf_repo_id=<username/repo> --policy_name=<policy_name>
```
This will load a already exported run and upload the exported policy to hugging face in the specified repo with the specified name.

NOTE: The policy must already be exported (see `play_policy` above).

For now we are uploading the policies to the hugging face repo [here](https://huggingface.co/zolkin/robot_rl/tree/main).

These policies are automatically downloaded in the `transfer/obelisk` controller.

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

## RL Tasks

RL Task list:

| Env. Type             |   Robot    |      Hardware Tested?      | Description                                                                          |
|-----------------------|:----------:|:--------------------------:|--------------------------------------------------------------------------------------|
| `vanilla`             |     G1     |     :white_check_mark:     | Basic, hand-tuned, RL walking on the G1 humanoid on flat ground.                     |
| `lip_clf`             |     G1     |     :white_check_mark:     | Basic, LIP CLF RL walking on the G1 humanoid on flat ground.                         |
| `walking_clf`         |     G1     |             No             | A HZD gait library; CLF RL walking on the G1 humanoid on flat ground.                |
| `walking_clf_sym`     |     G1     |     :white_check_mark:     | A HZD gait library; CLF RL walking on the G1 humanoid on flat ground, uses symmetry. |
| `running_clf_sym`     | G1 |     :white_check_mark:     | Running controller using a gait library.                                             |
| `G1-running-refresh-clf` | G1 |          No (sim)          | Running, submission refresh: trajectory-clip reference, submission-matched DR, revived flight-contact penalty. Gym IDs `G1-running-refresh-clf[-symmetric\|-play]`. |
| `waving_clf`          | G1 |             No             | Robot stands still and waves. Uses trajectories and CLFs.                            |
| `bow_forward_clf_sym` | G1 |     :white_check_mark:     | Robot bows forward then holds its position. Uses symmetry.                           |
| `bend_up_clf_sym`     | G1 |     :white_check_mark:     | Robot bends back up then holds in the standing position. Uses symmetry.              |

You can also append "_ec" to any of the above tasks to run them on the robot model that has the additional weight from the extra compute (EC).

## Generating the USD
Now we use the default IsaacLab built in urdf to usd converter:
```
./isaaclab.sh -p scripts/tools/convert_urdf.py robot_rl/robot_assets/g1/g1_21j/g1_21j_merged.urdf robot_rl/robot_assets/g1/g1_21j/isaac_converter
```
TODO: Update this to not have different paths

## Copying checkpoints from remote server 
First mount the server to your local desktop
 
```
bash scripts/mount_remote.sh
```

Note: this script might require a few changes as somethings are hard-coded.

## sim2sim Transfer
This code base has a built in sim2sim transfer (i.e. the policy is trained in IsaacLab and can be run in Mujoco).
Currently, we only support the G1 (as that is the only policy we have right now), but the code is easily extended to other robots.
To run the sim2sim transfer, go to the `transfer/sim/` directory. From this directory run
```
python g1_runner.py --env_type=<ENV_NAME>
```
where `ENV_NAME` is the same as above: it is an entry into the table. The arguments here follow the same structure as the `play_policy.py` script.
Optionally you can also add `--load_run=<run_dir>` to run a specific policy, otherwise the newest one will be run.
`--log` enables logging to the same directory. You can choose the logging directory with `--log_dir`.

This uses information that must have been exported from the `play_policy.py` script.

Then we can generate plots for the mujoco simulation. We can run
```
python plot_from_sim.py --env_type=<ENV_NAME>
```
and optionally you can pass `--load_run=<run_dir>` where `<run_dir>` is something like `025-11-17_15-04-12_walking_test`.
You can also pass `--log_session=<folder>` to choose a folder from within those mujoco logs. 
If either of these are unspecified then we plot the data for the newest run.

To add a new robot, the associated robot sim files will need to be added into the `transfer/sim/robots/` folder,
the `rl_policy_wrapper` will need to be adjusted a bit, and a new `runner` file will need to be made.

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

## Branch Overview

Snapshot as of 2026-06-08, after merging `multiskill-stair-slowdown` into `main` (PR #1).
`ahead`/`behind` are relative to `main`.

| Branch | ahead / behind | Description | Recommendation |
|---|---|---|---|
| `multiskill-stair-slowdown` | 0 / 1 | Active multiskill work; superset of the whole multiskill chain. Merged into `main`. | Prune (merged) |
| `multiskill-terrain-blocks` | 0 / 4 | ⊂ `multiskill-stair-slowdown`. Merged. | Prune (merged) |
| `multiskill-terrain` | 0 / 8 | ⊂ `multiskill-stair-slowdown`. Merged. | Prune (merged) |
| `multiskill` | 0 / 33 | ⊂ `multiskill-stair-slowdown`. Merged. | Prune (merged) |
| `clf-running-opt-sim` | 0 / 153 | Fully merged into `main`. | Prune (merged) |
| `clf-running` | 1 / 125 | Single unique commit (obelisk readme note) already contained in `demo`. | Prune (redundant) |
| `dual-cmd` | 3 / 54 | "Dual command" feature v1 + the idempotent-guard bug fix. **Not** in the multiskill line. | Decide — unique work |
| `clf-analytic` | 4 / 124 | Analytic Vdot, double-integrator, paper plots ("initial submission"). | Keep? (paper-relevant) |
| `demo` | 7 / 125 | Hardware demo configs (bow / bend-over, state machine). | Keep? (HW demos) |
| `mlip` | 64 / 441 | Older HW transfer + data-processing code (incl. Talos), 2025-12. | Prune? (old experiment) |
| `revision` | 7 / 449 | Paper rebuttal/revision code, mechanical cost-of-transport (2025-10/11). | Prune (stale) |
| `clf-rl` | 3 / 449 | Old CLF-RL work, cleanup/formatting commits (2025-08). | Prune (stale) |

> Already deleted: `spc-rl`, `longtest_real`, `amber5`, `distillation`.

### TODO: prune remaining branches

Not dealing with this right now — revisit and delete the branches that are no longer needed.
Delete a remote branch with `git push origin --delete <branch>` (then `git fetch --prune`).

- [ ] Prune merged/redundant (zero unique work lost): `multiskill-stair-slowdown`, `multiskill-terrain-blocks`, `multiskill-terrain`, `multiskill`, `clf-running-opt-sim`, `clf-running`
- [ ] Prune stale: `revision`, `clf-rl`, `mlip`
- [ ] Review before pruning (have unique commits): `dual-cmd`, `clf-analytic`, `demo`