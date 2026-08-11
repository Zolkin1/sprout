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

`scripts/rsl_rl/fm_debug.py` — the FM-vs-clip debugger, and the tool to reach for when the
tracker follows clips but not the generator. Freezes three clips (stair transition / flat /
box jump-up) into a small library and rolls the tracker over them twice, differing in ONE
field, `commands.traj_ref.reference_source`: `"clip"` tracks the pre-made clip while the
generator runs PASSIVELY (both references evaluated at identical robot states), `"fm"`
tracks the generator. `FmTrackingCommand` computes both every step and applies one;
`as_clip_reference()` / `as_fm_reference()` serve the other to a diagnostic — they swap
`y_des` AND the three look-ahead accessors, since `mdp.reference_window` reads the latter.
Stages `prep`/`run`/`plot`/`view` share `<out>/fm_debug.json`; `plot` and `view` are
IsaacLab-free. `mdp.reference_window_to_world` is the exact inverse of the window
observation, so the viewer can draw what the policy SAW and check it against what the
generator predicted.

`transfer/` — deployment: `transfer/sim` (MuJoCo sim2sim), `transfer/deploy` (the shared
torch-free ONNX runtime), `transfer/torch_math` (isaaclab-free torch quaternion math, the
test-only oracle for the numpy ports), `transfer/obelisk` (ROS2 hardware).

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
- These extend the BODY-tracking terrain teacher, so they default to `tracking_mode="body"` and pair
  with a widened (`@root`) clip library — the depth-vs-height-scan A/B is against
  `G1-teacher-clf-terrain-body` with exteroception as the only variable. Override per-run with
  `env.commands.traj_ref.tracking_mode=joint`. Costs ~19% more per iteration than the scan
  (139 vs 116 ms/frame at 1024 envs), almost all of it in `obs`.
- PROFILE (trained depth_v3, 1024 envs, `play.py --profile`; profile with a LOADED policy — an
  untrained one dies every step and doubles the apparent reset cost): of a 129 ms env step,
  `obs` 50.8 ms (39%) and physics/render ~47 ms (37%) dominate, then `command` 13.1, `events`
  7.8, `reward` 7.6, `termination` 2.5. ~19 ms of `obs` is the two depth cameras. `TODO(perf)`
  notes sit on `mdp.observations.reference_window` and `MotionSequenceCommand._update_command`;
  neither is instrumented per-term yet, so a breakdown comes before any optimisation.
- The FM's `attention_mode` decides which way the conditioning/future dependency runs.
  `prefix` (DEFAULT) has history+extero+cmd attend only to THEMSELVES and keeps the flow-time
  embedding off them, so their representation is a fixed statement rather than a function of
  the noise level — and is therefore encoded ONCE per re-plan and reused by every Euler step
  (`model.encode_prefix` / `decode_future`, hoisted in `flow.sample_trajectory` and in the
  unrolled ONNX graph). `full` is what every checkpoint before the flag was trained under.
  **The two modes share one parameter set**, so a mislabelled checkpoint runs silently under
  the wrong attention — `load_state_dict` cannot catch it. `from_dict` defaults a pre-flag
  checkpoint to `full`, the mode is printed at load, and it is in `fm_meta`; that is the whole
  defence. Measured: torch gets 1.34–1.39x on the CNN model at Isaac batch sizes, but the
  exported graph is ~0.82x at the deploy batch of 1 (fewer FLOPs, ~27% MORE nodes, and
  onnxruntime is launch-bound there). `prefix` is implemented as the cached decomposition, not
  as a mask: with a 2-D `attn_mask` the fused kernel materializes `(B, nhead, S, S)` inside
  `merge_masks` — gigabytes at Isaac env counts — so the masked pass is test-reference only.
- The FM selects depth OR height-scan exteroception (`extero_source`), and how one reading
  becomes TOKENS (`extero_encoder`, `robot_rl/flow_matching/extero_encoder.py`): `cnn` gives
  each camera its OWN 3-layer stride-2 trunk → a 4x4 grid of tokens with a learned spatial
  slot (16/camera, sequence 111); `mlp` is the original one-token-per-reading flat path
  (sequence 81), and the only option for the height scan. The flat path is
  permutation-invariant over pixels, so nothing in it can attend to WHERE a step edge is —
  which is the shape of the registration failure `fm_registration_probe.py` measures. The
  per-camera `(H, W)` is DERIVED from the recording's `depth_camera_spec`, travels into
  `fm.onnx`, and is checked at deploy: `extero_sensor_dim` cannot tell 30x26 from 26x30, and
  under a convolution a transposed camera is scrambled input rather than a reordered vector.
  `extero_norm_mode=per_sensor` pools the normalizer per camera instead of per pixel (a
  per-pixel affine is a spatially varying transform, which weight sharing assumes away).
- `scripts/rsl_rl/verify_depth_self_scan.py` verifies self-scanning at the sim level.

## The camera contract (how depth reaches MuJoCo / hardware)
`mdp.depth_camera_specs(env)` derives ONE contract per camera — mount body + offset, pinhole
intrinsics, range, and the `(d - offset) * scale` normalization — from the sensor cfgs and obs terms.
It is the single producer for all consumers, so nothing off-sim restates a camera number:
- `tracker.onnx` metadata `policy_parameters.depth_camera_spec` (via `export_parameters.py`);
- `fm.onnx` metadata `fm_meta.extero_spec` (+ `extero_source`, `extero_sensor_count`,
  `extero_sensor_dim` — sensor-neutral names; the old `extero_tokens`/`height_dim` are shimmed
  on `.pt` load but a stale `fm.onnx` is refused and must be re-exported), plumbed env cfg ->
  `RecordingMeta.depth_camera_spec` -> FM dataset -> checkpoint -> export;
- `GenTrackerPolicy.load()` requires the two to AGREE (`transfer/deploy/depth_obs.py`) — there is one
  physical camera pair, so a tracker and generator running together cannot describe different ones.

**The generator may SAMPLE that pair on its own grid** (so its vision input can change without
retraining the teacher). `add_fm_depth_pair(env_cfg, W, H)` in `g1_depth_env_cfg.py` adds a second
pair built from the SAME `_CameraSpec` — same mount, apertures, range and normalization, only the
ray grid differs — in its own `fm_depth` obs group, and EXTENDS the three DR events to
`sensor_cfgs` lists so one draw moves both samplings (extrinsics copy directly; intrinsics
transfer as a ratio, which is why `operation="scale"` is now required for a list). It is a no-op
when the resolutions match, and `mdp.fm_depth_groups(env)` then returns `None` = "all groups" =
the old behaviour, which is what keeps every scoped call site free.
- Who states the resolution: `record_data.py --fm_depth_resolution WxH` DECLARES it (no artifact
  exists yet); `play.py` and `augment_registration.py` DERIVE it (from the checkpoint via
  `online.read_fm_extero_meta`, and from the recording's own `depth_camera_spec`).
- Relaxing the deploy agreement is NOT "drop width/height": `fx/fy/cx/cy` are linear in the
  resolution, so they go free at the same time and a narrower cone on a bigger grid would pass.
  `depth_obs.py` exchanges the sampling fields for a positive check on the four resolution-invariant
  pinhole half-angles (all four, not their sum — a compensating principal-point shift preserves the
  sum). `REQUIRE_SAME_DEPTH_RESOLUTION = False` carries the TODO to re-tighten.
- MuJoCo casts BOTH grids (`gen_tracker_runner` builds a second `MujocoDepthCameras` from
  `get_fm_depth_camera_spec()` → `state["fm_depth"]`, with `mirror_extrinsics_from` so the jittered
  mount is shared). **Hardware is not wired for a split sampling** — `export_sensor_configs.py`
  raises rather than emitting configs that cover only the tracker.
Relatedly, `export_parameters.py` now emits each obs term's `clip`, so `_assemble` saturates every
term the way training did instead of consulting a hardcoded table (kept only as a legacy fallback).

## sim2sim (MuJoCo)
- `transfer/sim/depth_camera.py` — `mj_multiRay` re-implementation (~3.9 ms/frame for both cameras).
  Casts geom groups {1 = per-link visual meshes, 2 = terrain}: self-occlusion comes from group 1, and
  group 0 (collision) stays out because the foot there is a box, not the visual foot.
- `transfer/deploy/depth_obs.py` — the torch-free normalization port + spec parse/agreement, shared
  with hardware. `transfer/sim/verify_depth_cameras.py` checks geometry/coverage/self-occlusion,
  `scripts/rsl_rl/compare_depth_sim2sim.py` diffs IsaacLab vs MuJoCo pixel-by-pixel (they now
  agree everywhere, hands included, to a few mm). `--save_dir` writes labelled colour panels and
  `--dump` feeds `scripts/rsl_rl/view_depth_compare_viser.py`, which shows both sims' robots,
  camera frames and hit clouds in 3-D.
- The HANDS must stay real fixed-joint links (`robot_assets/g1/g1_21j/make_hand_links_urdf.py`,
  asset `g1_21j_box_merged_29j_hands`). As bare `<visual>` elements on the wrist — how the stock
  URDF ships them — the importer gives them a non-identity mesh->body offset, and
  `MultiMeshRayCaster` does not intersect meshes on that path: the cameras saw every limb EXCEPT
  the hands while the viewer looked perfect. They are the only meshes that ever used that path,
  which is why nothing else was affected. Joint count/order are unchanged by the fix (29, same
  order), so checkpoints still load; only `body_names` grows 30 -> 46, old list an exact prefix.
- sim2sim is ONNX-ONLY (the torch `gen_tracker_policy.py` is gone). The vendored torch FM copy
  (`transfer/fm`) is gone too — once deployment moved to ONNX and `fm_test_bench.py` was deleted,
  nothing loaded a model through it. What survives is `transfer/torch_math/quat_math.py`, the
  isaaclab-free torch quaternion math that `deploy/quat_math_np.py` is parity-tested against;
  `transforms_np.py` is now tested against the real `robot_rl.flow_matching.transforms`
  (`importorskip`-guarded, so that half skips without IsaacSim).

# obelisk (ROS2 deploy) — the default run pipeline

Three commands, in this order, every time:

```bash
cd transfer/obelisk
scripts/run_sim_low.sh                 # 1. BRAIN   (terminal A) — records MCAP by default
scripts/run_sim_world.sh               # 2. WORLD   (terminal B) — auto-arms the controller
python plot_utils/plot_from_bag.py     # 3. PLOTS   -> <bag>/plots/*.png
```

**Order is load-bearing**: the brain takes ~10 s to load the ONNX pair and MuJoCo has no
`user_pose` holding the robot up, so a world-first start hands the policy an already
collapsed robot. `scripts/stop.sh` tears both down (and now SIGINTs the recorder first so
the bag is finalised).

**When it falls over, take the generator out of the loop first.** `scripts/run_sim_clip.sh
--case stand` swaps in `g1_debug_clip.yml`, which loads the tracker ALONE (no `fm.onnx` at
all) and drives it from a baked clip. Cases, easiest first: `stand` (reference never moves —
if this falls the fault is this stack, not any policy), `stand_walk_stand` (the SAME clip
untruncated, so a failure isolates the walk transition), `terrain`. Each is known to survive
in MuJoCo through the same `transfer/deploy` runtime, so a case that falls here and not
there is a difference in the ROS stack. The clip library is built by
`scripts/rsl_rl/make_debug_clips.py` out of the library the teacher trained on.

- **No PD warm-up.** Whatever feeds the tracker drives it from the first tick, matching
  `gen_tracker_runner.py` (`--warmup_time 0`, ignored outright in clip mode). A stiffened
  hold put the robot in a state no training episode started from, and fed the generator's
  history rows of ZEROS — `_make_snapshot` reads the runtime's own previously-served node,
  which does not exist yet — which flipped `null_mask` off and denied the model the learned
  null-history embedding meant to cover exactly those steps.
- **Auto-arm is SIM ONLY and structurally so**: it is a parameter of `fsm_state_echo`, a
  node that exists only because MuJoCo has no execution FSM and is launched only by
  `run_sim_world.sh`. The controller has no auto-arm and must not grow one
  (`tests/test_auto_arm_is_sim_only.py`).

## Logging: MCAP, and what is deliberately NOT in it

Bags land in `transfer/obelisk/log/bags/<timestamp>_<tag>/`; figures next to them. The
observation VECTOR is not recorded — every term is either already on the network
(`est_state`, the depth topics) or reconstructible — but `obs_checksum` is, so an offline
reconstruction can PROVE it rebuilt what the network saw. `/obelisk/g1/debug/tick` carries
only what exists nowhere else: the post-watchdog command, the raw `action_isaac`, the served
reference window, arm/watchdog state, depth health, and **the stamps of the input samples
each tick consumed** (the estimator runs at 1 kHz against a 50 Hz tick, so without them no
recording can say which sample fed which step). `/obelisk/g1/debug/fm_plan` carries the
generator's output plus everything it was conditioned on, once per re-plan.

Three traps, each of which produced confident wrong numbers before being found:

- **Joint order.** The policy's is interleaved (`left_hip_pitch, right_hip_pitch,
  waist_yaw, ...`), the robot's is per limb — 27 of 29 positions differ on the G1.
  `action_isaac` and the reference window are in the POLICY's; `/ctrl` and `/est_state` are
  in the robot's. Subtracting one from the other gives a large, smooth, fictitious error.
  Every tick carries `joint_names_isaac`; the plotter refuses to compare without both.
- **`rosbag2_py` deserialises with the INSTALLED message definition**, not the bag's, so a
  bag recorded before a `.msg` edit is read through the wrong layout — silently, returning
  plausible values for the wrong fields. The `mcap` reader uses the bag's own schema and is
  the default; it also needs no ROS and no `metadata.yaml`.
- **A `.msg` change needs g1_msgs REBUILT** (`rm -rf build/g1_msgs install/g1_msgs &&
  scripts/build.sh`) — bindings are generated, not symlinked. The controller now refuses to
  start on a mismatch rather than recording nothing for a whole run. Relatedly, a literal
  `*/` anywhere in a `.msg` comment terminates the generated C block comment and the rest
  compiles as code (`tests/test_msg_comments.py`).

Also pinned by tests: a config parameter the node never declares is silently ignored by
rclpy, so `tests/test_params_are_live.py` fails the build on one (this bit `log`,
`log_decimation`, `config_name` and `fm_update_every`). The generator's re-plan cadence
comes from `tracker.onnx` metadata and nowhere else.

# General Guidelines
- Don't care for backwards compatibility.
- Use the uv virtual environment `env_isaaclab` (located at the repo root) for all related python code; activate it with `source env_isaaclab/bin/activate`. The exception is `transfer/` off-sim work (MuJoCo sim2sim + the ONNX deploy runtime), which imports no IsaacLab and runs in the much smaller `env_transfer` — built by `bash transfer/setup_venv.sh` from `transfer/requirements.txt`, so a Jetson or any candidate deploy box can play and benchmark policies without a 10 GB Isaac install. On a training box `env_isaaclab` covers both.
- When writing a function, always provide a doc string.
- In general, we should always use type hints where it is idiomatic.
- In python variable names should be snake_case and function should be snake_case and classes should be UpperCamelCase.
- When creating a new file, add a comment on the very first line (after a shebang, if present): `Copyright <current year> Zachary Olkin. All rights reserved.` Use the file's comment syntax (`#` for Python/shell/YAML, `//` for C/C++, `%` for MATLAB).
- Many files under `source/robot_rl/.../tasks/` and `scripts/rsl_rl/` are derived from IsaacLab and carry a BSD-3-Clause header (Isaac Lab Project Developers). Retain those headers when editing; do not replace or remove them, and do not add the personal copyright line to them.
- Training and playing are run via `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` (with `env_isaaclab` active, or via IsaacLab's `./isaaclab.sh -p`). See `README.md` for task names and examples.
- `source/motion_gen` must stay IsaacLab-free — it is imported by generation scripts and tests that run without a simulator. Don't add an `isaaclab` import to it; put anything that needs one on the `robot_rl` side.
- Both packages need to be importable. `source/robot_rl` is the editable install, but `source/motion_gen` is not, so run with `PYTHONPATH=source/robot_rl:source/motion_gen`. In a git worktree this also shadows the editable install, which otherwise points at the main checkout.
- Don't build paths with a hard-coded number of `..` hops from `__file__` — derive them from the package root (see `motion_gen.features._REPO_ROOT`). Moving a module between packages has silently broken such a path before.