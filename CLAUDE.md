# Code Overview
This code deals with Reinforcement Learning (RL) for robots, specifically humanoid robots right now.
The code holds all the information to train the RL algorithm and then deploy it in simulation and on hardware.

# Code Architecture
The code is using IsaacLab at its core and thus uses the suggested file structure. 

source is where most of the core code is housed, as two editable packages:
    - `source/robot_rl` - the IsaacLab side (RL envs, MDP terms, networks, flow matching).
    - `source/motion_gen` - IsaacLab-free offline motion/clip generation (torch/numpy/h5py only).

source/robot_rl/robot_rl/tasks/manager_based/robot_rl/g1 is where the configs for each of the different actions are.
source/robot_rl/robot_rl/tasks/manager_based/robot_rl/mdp is where all the mdp code is including:
    - rewards
    - curriculums
    - events
    - commands
    - observations
    - sensors
    - terminations
source/robot_rl/robot_rl/tasks/manager_based/robot_rl/terrains holds the custom terrain generators (stairs, meta-terrain).
source/robot_rl/robot_rl/flow_matching is the flow-matching (FM) reference generator; its scripts are in
scripts/flow_matching (see that README).

scripts/rsl_rl is where the code to actually run the training and the play the result is.
Training and playing is logged to logs/ (logs/rsl_rl/<exp>/<run>, logs/flow_matching/<run>).

transfer holds code to transfer the RL code to mujoco (in transfer/sim) and to the hardware (transfer/obelisk).
transfer/deploy is the torch-free ONNX runtime shared by both, and transfer/fm is a vendored IsaacLab-free
copy of the FM math.

# Motion / Clip Generation
The RL pipeline is offline-clip -> teacher -> FM -> tracker: a reference clip library (`.h5` under `clips/`)
is generated offline, a teacher policy tracks it, and the FM model trains on recorded teacher rollouts.
All clip generation lives in `source/motion_gen` and never imports IsaacLab.

- **Motion Bricks (current primary path)** - keyframe in-betweening via NVIDIA Motion Bricks.
  `scripts/rsl_rl/generate_mb_library.py` is the top-level driver. It spans TWO environments that cannot
  share a process (`conda run -n motionbricks` for the GPU model, `env_isaaclab` for the clip machinery),
  so it needs `--mb-root ../GR00T-WholeBodyControl/motionbricks`. Every stage is skip-if-done, so a killed
  run resumes; `--stage` runs one stage alone (e.g. on the machine with the GPU).
- **Motion matching** - matches a motion database against a velocity command over a cached,
  content-addressed candidate graph. Three matchers: `vanilla` (greedy), `beam` (approximate), `dp`
  (exact backward solve; what the clip generators use).
- **Terrain families** - `scripts/rsl_rl/generate_terrain_library.py` runs each family's generator
  (flat / boxes / stairs / distractors) and merges them into ONE library via `compose_terrain_library.py`,
  which sets the per-family tile proportions.

# Commands
```bash
source env_isaaclab/bin/activate

# tests
python -m pytest source/motion_gen/tests source/robot_rl/tests \
    source/robot_rl/robot_rl/flow_matching/tests transfer/fm/tests transfer/sim/tests
python -m pytest transfer/deploy/tests/test_hardware_offline.py   # own process, see Gotchas
```

Style is configured in `.pre-commit-config.yaml` and `.flake8`: black with `--line-length 120`, isort
`--profile black`, flake8 `max-line-length = 120`. Match that when writing code. `pre-commit` is NOT
installed in `env_isaaclab` and no git hook is wired up; to run it: `uv pip install pre-commit && pre-commit run --all-files`.

# Gotchas
- BOTH packages must be installed editable (`robot_rl` imports `motion_gen` for the stair terrain, so
  `import robot_rl` fails without it): `uv pip install -e source/robot_rl` and
  `uv pip install -e source/motion_gen --no-deps` (the `--no-deps` keeps uv from re-resolving the pinned
  Isaac Sim / CUDA torch stack). `pyproject.toml` finds packages under `source/robot_rl`, not the repo
  root - installing from the root gives a broken editable install.
- Large clip libraries (>1 GB) OOM a small GPU - pass `--clip_device cpu` to train/play/record.
- `transfer/deploy` must stay torch-free; `test_hardware_offline.py` asserts `torch not in sys.modules`,
  so it must run in its OWN pytest process - even its sibling ONNX-parity test imports torch and breaks it.
- `transfer/sim/tests` and `transfer/deploy/tests/test_gen_tracker_onnx_parity.py` need a local FM
  checkpoint under `logs/flow_matching/` and fail (not skip) without one.

# General Guidelines
- Don't care for backwards compatibility.
- Use the uv virtual environment `env_isaaclab` (located at the repo root) for all related python code; activate it with `source env_isaaclab/bin/activate`.
- When writing a function, always provide a doc string.
- In general, we should always use type hints where it is idiomatic.
- In python variable names should be snake_case and function should be snake_case and classes should be UpperCamelCase.
- When creating a new file, add a comment on the very first line (after a shebang, if present): `Copyright <current year> Zachary Olkin. All rights reserved.` Use the file's comment syntax (`#` for Python/shell/YAML, `//` for C/C++, `%` for MATLAB).
- Many files under `source/robot_rl/.../tasks/` and `scripts/rsl_rl/` are derived from IsaacLab and carry a BSD-3-Clause header (Isaac Lab Project Developers). Retain those headers when editing; do not replace or remove them, and do not add the personal copyright line to them.
- Training and playing are run via `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` (with `env_isaaclab` active, or via IsaacLab's `./isaaclab.sh -p`). See `README.md` for task names and examples.
- To train or play WITH A VIEWER, pass `--viz kit`. `--headless` is DEPRECATED as of IsaacLab v3.0 (its help
  text says so) - headless is now the default when no visualizer is requested, so omit the flag entirely for
  headless runs, or pass `--viz none` to force-disable visualizers. `--visualizer`/`--viz` takes a CSV of
  backends (`kit,newton,rerun,viser`).
