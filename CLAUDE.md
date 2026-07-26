# Code Overview
This code deals with Reinforcement Learning (RL) for robots, specifically humanoid robots right now.
The code holds all the information to train the RL algorithm and then deploy it in simulation and on hardware.

# Code Architecture
The code is using IsaacLab at its core and thus uses the suggested file structure. 

source is where most of the core code is housed. 
source/robot_rl/robot_rl/tasks/manager_based/robot_rl/g1 is where the configs for each of the different actions are.
source/robot_rl/robot_rl/tasks/manager_based/robot_rl/mdp is where all the mdp code is including:
    - rewards
    - curriculums
    - events
    - commands
    - terminations

scripts/rsl_rl is where the code to actually run the training and the play the result is.
Training and playing is logged to log.

transfer holds code to transfer the RL code to mujoco (in transfer/sim) and to the hardware (transfer/obelisk).

# Depth Cameras (this branch)
Two self-scanning depth cameras on the terrain teacher, as an alternative exteroception to the height scan.
- `g1/g1_depth_env_cfg.py` + `g1/g1_self_scan.py` - camera cfgs and the generated `MultiMeshRayCaster`
  target prim paths. For the robot to appear in its own depth image every moving link must be its OWN
  target with `track_mesh_transforms=True`, aimed at the INNER same-named Xform: a target on the OUTER
  link Xform welds the whole downstream limb into one rigid body frozen at its spawn pose (so e.g. knee
  flexion would be invisible).
- `mdp/observations/depth.py`, `mdp/events/depth.py`, `mdp/sensors/noise.py` - the observation terms,
  mount-pose randomization + FOV edge-crop, and datasheet optics/range/rate.
- Tasks: `G1-teacher-clf-terrain-depth{,-image}` (+ each `-play`). Depth REPLACES the height scan in
  the POLICY obs (the hardware has cameras, not a heightmap); the CRITIC keeps the scan, since it is
  privileged and never deployed. The `-image` variant differs only in obs SHAPE (per-camera image
  groups for a CNN) — it is untested in training and `PPORunnerCfg` is an MLP.
- The FM selects depth OR height-scan exteroception tokens (`robot_rl/flow_matching/config.py`).
- `scripts/rsl_rl/verify_depth_self_scan.py` verifies self-scanning at the sim level.

## The camera contract (how depth reaches MuJoCo / hardware)
`mdp.depth_camera_specs(env)` derives ONE contract per camera — mount body + offset, pinhole
intrinsics, range, and the `(d - offset) * scale` normalization — from the sensor cfgs and obs terms.
It is the single producer for all consumers, so nothing off-sim restates a camera number:
- `tracker.onnx` metadata `policy_parameters.depth_camera_spec` (via `export_parameters.py`);
- `fm.onnx` metadata `fm_meta.extero_spec` (+ `extero_source`, `extero_tokens`), plumbed env cfg ->
  `RecordingMeta.depth_camera_spec` -> FM dataset -> checkpoint -> export;
- `GenTrackerPolicy.load()` requires the two to AGREE (`transfer/deploy/depth_obs.py`) — there is one
  physical camera pair, so a tracker and generator running together cannot describe different ones.
Relatedly, `export_parameters.py` now emits each obs term's `clip`, so `_assemble` saturates every
term the way training did instead of consulting a hardcoded table (kept only as a legacy fallback).

## sim2sim (MuJoCo)
- `transfer/sim/depth_camera.py` — `mj_multiRay` re-implementation (~3.9 ms/frame for both cameras).
  Casts geom groups {1 = per-link visual meshes, 2 = terrain}: self-occlusion comes from group 1, and
  group 0 (collision) stays out because the foot there is a box, not the visual foot.
- `transfer/deploy/depth_obs.py` — the torch-free normalization port + spec parse/agreement, shared
  with hardware. `transfer/sim/verify_depth_cameras.py` checks geometry/coverage/self-occlusion,
  `scripts/rsl_rl/compare_depth_sim2sim.py` diffs IsaacLab vs MuJoCo pixel-by-pixel (they agree
  exactly except for the hand delta below).
- KNOWN DELTA: MuJoCo group 1 includes the HAND meshes, which `SELF_SCAN_LINKS` excludes (~60/780
  near-camera px). Kept deliberately until the hardware camera's occlusion geometry is known; TODOs
  in `g1_self_scan.py` and `transfer/sim/depth_camera.py`.
- sim2sim is ONNX-ONLY (the torch `gen_tracker_policy.py` is gone). `transfer/fm/` remains as the
  torch oracle for the numpy ports + `fm_test_bench.py`, and is height-scan-only.

# General Guidelines
- Don't care for backwards compatibility.
- Use the uv virtual environment `env_isaaclab` (located at the repo root) for all related python code; activate it with `source env_isaaclab/bin/activate`.
- When writing a function, always provide a doc string.
- In general, we should always use type hints where it is idiomatic.
- In python variable names should be snake_case and function should be snake_case and classes should be UpperCamelCase.
- When creating a new file, add a comment on the very first line (after a shebang, if present): `Copyright <current year> Zachary Olkin. All rights reserved.` Use the file's comment syntax (`#` for Python/shell/YAML, `//` for C/C++, `%` for MATLAB).
- Many files under `source/robot_rl/.../tasks/` and `scripts/rsl_rl/` are derived from IsaacLab and carry a BSD-3-Clause header (Isaac Lab Project Developers). Retain those headers when editing; do not replace or remove them, and do not add the personal copyright line to them.
- Training and playing are run via `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` (with `env_isaaclab` active, or via IsaacLab's `./isaaclab.sh -p`). See `README.md` for task names and examples.