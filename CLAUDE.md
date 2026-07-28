# Code Overview
This code deals with Reinforcement Learning (RL) for robots, specifically humanoid robots right now.
The code holds all the information to train the RL algorithm and then deploy it in simulation and on hardware.

# Code Architecture
The code is using IsaacLab at its core and thus uses the suggested file structure.

## The pipeline
One chain, each stage feeding the next:
`motionbricks clip generation -> teacher RL -> flow-matching generator -> tracker -> sim2sim -> hardware`.

The **clip library** (`.h5`) is the spine: it carries both the reference motions and the
terrain they were generated on, so one file defines what the robot tracks AND what it walks
on. Every task takes it via `--clip` and there is deliberately no default (a stale library
silently trains on the wrong terrain). See `README.md` for the stage-by-stage runbook.

## Layout
`source/motion_gen/` — IsaacLab-free offline clip generation:
    - `motionbricks/` — the live generator (raw Motion Bricks npz -> clip ingest)
    - `terrain/` — sim-free terrain geometry, EMITTED by the generators and re-read by the RL
      side so nothing is recomputed or allowed to drift
    - `ingest/` — readers turning a reference-motion library on disk into tensors
    - `clips/` — clip `.h5` IO, the RL-runtime `MotionClipManager`, fingerprinting
    - `motion_matching/` — DEPRECATED, kept only for ablations (see its README)

`source/robot_rl/robot_rl/` — the RL side:
    - `tasks/manager_based/robot_rl/g1/` — the 5 env cfgs + agent cfgs. `g1_clf_tracking_base`
      is abstract (scene / actions / events / CLF rewards); `g1_teacher_env_cfg` adds the
      clip command + observations; `g1_teacher_terrain_env_cfg` supplies the ground;
      `g1_tracker_env_cfg` swaps the clip reference for the flow-matching one
    - `tasks/manager_based/robot_rl/mdp/` — rewards, observations, terminations, events, and
      `commands/traj_tracking/` (the clip + FM command terms, the CLF, the world-lift adapter)
    - `tasks/manager_based/robot_rl/terrains/` — the `ClipTerrainImporter` + block builders
    - `flow_matching/` — the FM model, dataset and viz

`scripts/rsl_rl/` — entry points (`train.py`, `play.py`, `record_data.py`, the `mb_*`
generation stages, `compare_sim2sim.py`) plus diagnostics. `scripts/flow_matching/` trains
and exports the FM. Runs are logged to `logs/`.

`transfer/` — deployment: `transfer/sim` (MuJoCo sim2sim), `transfer/deploy` (the shared
torch-free ONNX runtime), `transfer/fm` (isaaclab-free FM), `transfer/obelisk` (ROS2 hardware).

# General Guidelines
- Don't care for backwards compatibility.
- Use the uv virtual environment `env_isaaclab` (located at the repo root) for all related python code; activate it with `source env_isaaclab/bin/activate`.
- When writing a function, always provide a doc string.
- In general, we should always use type hints where it is idiomatic.
- In python variable names should be snake_case and function should be snake_case and classes should be UpperCamelCase.
- When creating a new file, add a comment on the very first line (after a shebang, if present): `Copyright <current year> Zachary Olkin. All rights reserved.` Use the file's comment syntax (`#` for Python/shell/YAML, `//` for C/C++, `%` for MATLAB).
- Many files under `source/robot_rl/.../tasks/` and `scripts/rsl_rl/` are derived from IsaacLab and carry a BSD-3-Clause header (Isaac Lab Project Developers). Retain those headers when editing; do not replace or remove them, and do not add the personal copyright line to them.
- Training and playing are run via `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` (with `env_isaaclab` active, or via IsaacLab's `./isaaclab.sh -p`). See `README.md` for task names and examples.
- `source/motion_gen` must stay IsaacLab-free — it is imported by generation scripts and tests that run without a simulator. Don't add an `isaaclab` import to it; put anything that needs one on the `robot_rl` side.
- Both packages need to be importable. `source/robot_rl` is the editable install, but `source/motion_gen` is not, so run with `PYTHONPATH=source/robot_rl:source/motion_gen`. In a git worktree this also shadows the editable install, which otherwise points at the main checkout.
- Don't build paths with a hard-coded number of `..` hops from `__file__` — derive them from the package root (see `motion_gen.features._REPO_ROOT`). Moving a module between packages has silently broken such a path before.