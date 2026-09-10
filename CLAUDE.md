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

# The robot is `g1_perceptive`, and FOUR files describe it

`robot_assets/g1_perceptive/` is the trained robot: mode_11 motors, a 2.64 kg compute backpack
(Thor + ZED), passive rubber hands in place of the Dex3-1s, the torso collider on the torso, and
the depth-camera mounts on their as-built CAD pose. Everything else is DERIVED from its URDF, and
that is the point -- the same robot used to be described by hand in several places at once.

| file | derived by | read by |
|---|---|---|
| `robot_assets/g1_perceptive/g1_perceptive.urdf` | authored (`make_perceptive_urdf.py`) | the source of truth |
| `robot_assets/g1_perceptive/g1_perceptive/` (USD) | `urdf_to_usd_conversion.py` | IsaacLab |
| `transfer/sim/robots/g1/g1_perceptive.xml` | `make_g1_mjcf.py` | MuJoCo sim2sim + `motion_gen` FK |
| `transfer/obelisk/g1_model/urdf/g1_perceptive.urdf` | `make_ros_urdf.py` | the deploy camera FK + RViz |

**The fourth one is the cautionary tale.** It was a hand-kept copy called `g1_29j.urdf`, and it
was byte-identical on main, the deploy branch and the training-refresh branch -- so while
training moved to the perceptive robot, `g1_deploy::CameraPoseFk` went on FK'ing the old one.
Nothing errored: `torso_link` and all 29 joints exist in both, so every name it looks up
resolved, and it placed both depth cameras **exactly 10.0 mm high on every unprojected xyz
channel**. A wrong camera pose is a plausible cloud in the wrong place, so no image inspection
would ever have found it. `test_ros_urdf_is_generated.py` now fails if the generated file drifts
from its source, and `test_fk_parity.py` imports both models from where they are DEFINED rather
than restating paths, so the URDF and the MJCF cannot name different robots.

**Joint count and order are UNCHANGED -- 29, identical -- and that is what makes a stale model
dangerous.** `joint_names_isaac` is the only exported field that came close to identifying a
robot, and it matches across a payload change, a hand swap and a gear-ratio change. So:
- `robot_asset` (`motion_gen/robot_asset.py`) fingerprints urdf/mjcf/usd hashes + joint names +
  sorted body names + total mass. `export_parameters` stamps it into `tracker.onnx`,
  `export_fm_onnx` into `fm.onnx`, and BOTH runtimes refuse a pair that disagrees
  (`_assert_robot_assets_agree`, `AssertRobotAssetsAgree`). A missing side is silent -- it is a
  guard against a new mismatch, not a migration gate -- but "absent" and "empty" are
  deliberately different things, since an empty fingerprint would compare equal to anything.
- `terrain_fingerprint` is stamped and **deliberately compared nowhere on the deploy path**; see
  `TrackerSession.get_terrain_fingerprint` for why both obvious comparisons are wrong rather
  than merely unimplemented.
- **Pre-mode_11 checkpoints load and are numerically meaningless**: `G1_ACTION_SCALE` is
  `0.125 * effort / stiffness`, and hip pitch moved from the 14.3 to the 22.5 gear, so its action
  scale went 0.2738 -> 0.1753. Same width, different meaning. `test_actuator_table.py` pins the
  whole table.
- Body names changed in BOTH directions (39 bodies; the 16 Dex3 links per side went, the hands,
  camera mounts and backpack arrived), so `body_names` is no longer a prefix of the old list and
  anything indexing it positionally shifts silently.

# The reference window has TWO conventions, and deploy serves only one

`window_anchor` decides what the `base_pose` block's POSITION is measured from
(`robot_rl/flow_matching/window.py`). The two arms are **identical in width and in every other
block**, so a tracker trained under one loads clean against the other and merely steers wrong --
which is why the convention travels in the artifact and is checked rather than inferred.

- `robot_now` -- `p_ref - p_actual(t)`, the tracking error against the robot as it is this step.
  Serving it means re-anchoring the prediction onto the measured base every control step, which
  needs a base POSITION, which on a robot means odometry.
- `plan` -- `p_ref - p_plan`, measured from the root at the last re-plan. The generator already
  predicts in that frame, so its output IS the observation: served as made, no base position read.

**The C++ deploy runtime serves `plan` ONLY, and refuses `robot_now` by design rather than by
omission** (`AnchorFromConvention`). That is what makes the deploy path odometry-free: under
`plan`, `root_pos` reaches only the anchor latch and from there only `TickDiagnostics` and the
FmPlan ribbon in RViz. The FAST-LIO plumbing stays in the repo, unused, and a drifting or absent
base position costs a debug ribbon rather than the robot. The PYTHON runtime serves both arms --
it is the MuJoCo driver and the parity oracle, and MuJoCo has a true base pose, so a `robot_now`
A/B belongs there.

**One `fm.onnx` serves either arm.** The generator is invariant to the choice: `_run_fm` latches
the anchor on both, and the history stash is the raw served node on both. Only the serve differs.

# Depth Cameras
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
  `extero_norm_mode` decides how the normalizer is pooled; see "The DEFAULTS are the depth
  training config" below (`per_channel` is the default, `per_element` is gone).
- `scripts/rsl_rl/verify_depth_self_scan.py` verifies self-scanning at the sim level.

## The generator's depth is DELAYED, in training and in both simulators

The real depth pipeline puts tens of milliseconds between capture and the frame reaching the
network (~42 ms median measured on the ZED pair, 48 ms modelled), and a generator fed a fresh
frame in simulation is off-distribution in exactly the variable it was fitted to. So the
generator's conditioning carries a lag on purpose:

- TRAINING: `depth_latency_probs` is a distribution over whole control steps, sampled per window
  by `dataset._select_depth_lag`, and it rides `fm_meta` end to end so a checkpoint brings its
  own number rather than inheriting whatever the runner defaults to.
- SIM: `FmDepthLag` in `gen_tracker_runner.py`, derived from the graph's own
  `depth_latency_probs` unless `--fm_depth_lag_steps` overrides it. **Only the GENERATOR is
  delayed**; the tracker's own `depth` observation stays live, because it was trained on fresh
  frames.
- **The POSE travels with the image.** The 5-channel row's xyz planes are unprojected from the
  camera pose, so a lagged image unprojected with the CURRENT pose is displaced by the egomotion
  -- precisely the registration error those channels exist to remove.
- HARDWARE adds none of this: its lag is physical and already in the frames. What the controller
  does instead is match the camera POSE to the frame's capture stamp (`g1_deploy::PoseHistory`,
  `depth_pose_match`), which is the same consistency requirement solved from the other end.
  `depth_delay_relay` manufactures the missing latency in MuJoCo so that matching code is
  actually exercised rather than merely present.

**`--fm_lead_ticks` and `--fm_depth_lag_steps` do NOT compose.** Both exist to line the sim
runner up against the C++ runtime, which applies no synthetic lag, so leaving both on stacks two
delays and silently makes the documented matched comparison (`--fm_lead_ticks 1`) a 3-tick one.
The lag is forced off for `--fm_lead_ticks` / `--dump_replay`, and asking for both explicitly is
a hard exit. That also keeps `--dump_replay` streams coherent: the replay records the tracker's
LIVE `depth_raw`, so a lagged `fm_camera_pose` beside it would make the C++ rebuild the extero
row from live pixels and a stale pose, and `compare_replay.py` would report a divergence that is
the harness's fault.

## The camera contract (how depth reaches MuJoCo / hardware)
`mdp.depth_camera_specs(env)` derives ONE contract per camera — mount body + offset, pinhole
intrinsics, range, and the `(d - offset) * scale` normalization — from the sensor cfgs and obs terms.
It is the single producer for all consumers, so nothing off-sim restates a camera number:
- `tracker.onnx` metadata `policy_parameters.depth_camera_spec` (via `export_parameters.py`);
- `fm.onnx` metadata `fm_meta.extero_spec` (+ `extero_source`, `extero_sensor_count`,
  `extero_sensor_dim` — sensor-neutral names; the old `extero_tokens`/`height_dim` are shimmed
  on `.pt` load but a stale `fm.onnx` is refused and must be re-exported), plumbed env cfg ->
  `RecordingMeta.depth_camera_spec` -> FM dataset -> checkpoint -> export;
- `GenTrackerPolicy.load()` requires the two to AGREE on every camera they SHARE
  (`transfer/deploy/depth_obs.py`) — there is one physical camera pair, so a tracker and
  generator running together cannot describe the same lens differently.

**The POLICY reads the NEAR camera alone; the GENERATOR reads both.** The tracker follows an
already-placed reference, so terrain ahead is the generator's problem, not its. That makes the
tracker's camera set a strict, name-matched SUBSET of the generator's, and it is the shipped
configuration — so the two contracts have DIFFERENT LENGTHS and `assert_depth_specs_agree`
matches by NAME (`camera_key`, which strips the `fm_` sampling prefix) rather than by list
index. Pairing by index would compare the policy's near camera against the generator's far one
and report two genuinely different cameras as a contract violation. The other direction stays an
error: a camera the tracker reads and the generator does not is one the pair cannot supply.

**The generator's group is INSTALLED, never inferred.** `install_fm_depth_group(cfg, W=None,
H=None)` in `g1_depth_env_cfg.py` (it replaced `add_fm_depth_pair`) puts every scene camera in
its own `fm_depth` obs group. At the policy's own 30x26 the group's terms point at the EXISTING
sensors — no duplicate ray cast — and at any other grid it builds `fm_`-prefixed duplicates from
the SAME `_CameraSpec` (same mount, apertures, range and normalization; only the ray grid
differs) and EXTENDS the three DR events to `sensor_cfgs` lists so one draw moves both samplings
(extrinsics copy directly; intrinsics transfer as a ratio, which is why `operation="scale"` is
now required for a list). `mdp.fm_depth_groups(env)` returns that group and
`mdp.rl_depth_groups(env)` everything else.
- **A missing group is a FAULT, not an opt-out.** Without it `fm_depth_groups` falls back to
  `None` = "all groups", which on this task is the policy's NEAR camera alone: the recorder
  would stamp a one-camera `depth_camera_spec`, the FM would train on it, `export_fm_onnx` would
  emit a matching graph and the deploy agreement check would PASS — every artifact
  self-consistent about a half-blind generator. `assert_fm_depth_group_covers_every_camera` is
  the guard; every task and script that runs or records the generator installs the group
  unconditionally.
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

## The height-scan contract, and running a PRIVILEGED generator in sim2sim
`fm_meta.extero_spec` holds whichever sensor the checkpoint trained on, so it is a camera list
for one FM and a ray grid for another. `transfer/deploy/scan_obs.py` is the scan half:
`parse_extero_spec` picks the parser from the contract's own `extero_source` (parsing a scan as
depth failed for a missing `fx`), and `assert_scan_specs_agree` compares the two artifacts over
the fields they SHARE — `fm.onnx` embeds the full `mdp.height_scan_specs` dict, `tracker.onnx`
the six `export_parameters` emits — with a floor so a sparse contract cannot agree by silence.
- A **depth tracker + scan generator** is a legitimate pair (the robot has cameras, the generator
  reads the ground). `_resolve_depth_cameras` only reconciles `extero_spec` against the tracker's
  cameras when the FM's source IS depth.
- The dense scan is marked `privileged`, and that gate is now **sim-only rather than absolute**:
  `export_fm_onnx --allow_privileged_extero` (plumbed through `play.py`) emits the graph,
  `gen_tracker_runner.py --allow_privileged_extero` runs it, and MuJoCo casts the same 121x121
  grid via `Robot.compute_height_scan` (~20 ms/step for 14641 rays — the per-ray `mj_ray` loop is
  fine at this size; `mj_multiRay` does NOT apply, since a scan's rays do not share an origin).
  **Hardware still refuses**, because `GenTrackerPolicy` defaults the flag off and nothing on the
  robot path passes it. Anything measured this way is a statement about the model given perfect
  terrain.
- The dense grid is SQUARE, so a transposed fold is shape-silent. `transfer/sim/tests/`
  `test_height_scan_grid.py` pins it against the coarse 16x11 cast where the grids coincide (and
  asserts the transpose would fail). Note an exact-equality check is a coin flip at yaw=0 on
  stairs: the shared points land on a step edge to 1e-14 m and the two casts differ by one riser.
- `torch.isclose` has no ONNX lowering, so the miss-sentinel channel in
  `flow_matching/extero_encoder.py` is written out as `|img - sentinel| > atol + rtol*|sentinel|`
  (identical on every finite input). Before that, a CNN extero model with a sentinel could train
  but not export.

## The depth row carries CHANNELS, not just depth (`extero_channels`)
The FM sees no joint angles and no root orientation, so a waist turn or a pelvis pitch is
indistinguishable from the terrain moving (`fm_registration_probe`: the predicted stair
segment slides 1.10 of a downrange displacement, where the anchor mask requires 0). So a
pixel can carry more than depth: `1` = depth alone (every older checkpoint), `4` = `+ xyz` in
the root's yaw-aligned frame, `5` = `+ validity`. `extero_sensor_dim` COUNTS channels (it is
the row width), so everything comparing it against an image shape divides by
`extero_channels` first — `check_extero_encoder`, the model ctor, `read_fm_extero_meta`,
`_resolve_extero_image_hw` and `gen_tracker._resolve_depth_cameras`. The fold is
CHANNEL-MAJOR; interleaving scrambles every kernel while the shapes stay right.
- **The frame is LEVEL** — orientation `yaw_quat(root_quat_w)`, so its z is gravity and the
  frame has zero roll/pitch. **Static terrain therefore does NOT move when the robot tilts**,
  and that invariance is what the channels buy: a pitch rewrites every pixel but moves the
  CAMERA by the same amount, and the unprojection composes it back out. What moves within the
  level frame is the robot's own geometry (the camera, and the limbs it sees), so the tilt
  stays observable without being confusable with the ground moving. The heading cancels
  (`R_world_root = Rz(psi) @ R_rp`), so a deployment needs only FK to the camera and the IMU's
  roll/pitch — **never odometry**, and a drifting yaw cannot corrupt it. That is a claim about
  the COMPUTATION, not a yaw invariance: the frame turns with the robot, so a yaw does rotate
  a static point's coordinates — consistently with the predicted window, which turns too.
- **Nominal extrinsics, never the DR draw**: a robot knows its calibration and nothing more,
  so the DR perturbs the IMAGE relative to the stated geometry (which is what calibration
  error IS). Training on the randomised pose would teach a registration it never gets.
- **No re-record.** The offline path (`dataset._camera_pose_yaw_frame`) rebuilds the pose
  from the recorded root quaternion + joint angles + the contract's mount, via `motion_gen`'s
  `FKChain` — validated against the limb frames the recording stores (0.67 mm).
  `mdp.depth_xyz_stack` is the live twin; they must agree.
- **Validity is found by RANGE, not sentinel equality.** `DepthNoiser`'s 3x3 blur leaves a
  geometric miss NEAR the sentinel; only the 2.5% per-frame dropout lands exactly on it.
  Measured over the full recording: 6.4% (near) / 8.9% (far) of pixels carry no reading. The
  height scan keeps equality, where it is exact because that signal is noise-free.
- **A camera's position embedding is ANGULAR and token-level** (`pixel_angles_rad`), and
  **per sensor**: the two cameras are 100x70 and 70x42 deg, so one shared extent mislabels
  the far one's tokens by 1.43x, silently. The metric ground grid (the scan) keeps its coord
  CHANNELS; a ray has a direction, not a planar coordinate.
- `extero_cnn_grid` larger than the trunk's native grid used to upsample silently (a token
  count that is a claim, not a fact); now refused.

## The DEFAULTS are the depth training config (nothing to remember at the CLI)
Every extero knob resolves per SOURCE, so one default set is correct for a camera run and a
scan run alike, and a checkpoint records the resolved number rather than "whatever the default
meant that day". `extero_channels=None` -> 5 for depth / 1 for a scan
(`resolve_extero_channels`), joining `extero_encoder="auto"` (cnn / mlp) and
`extero_clip=None` (±5.575 / ±1.0 m) in that idiom. Defaults now: `depth`, `cnn`, 5 channels,
`strides (1,2,2)` = **7x8 = 56 tokens/camera** (sequence 191), `per_channel` normalization.
- `EXTERO_CLIP["depth"]` is now **±5.575, measured**: at ±2.0 the band saturated at 4.75 m
  while the far camera sees 11.9 m (1.30% of pixels, 13.8% of windows, 99th pct 5.22 m).
  Recordings store depth UNCLIPPED, so widening recovers signal already on disk. Changing the
  per-source default RE-RESOLVES any checkpoint that stored `extero_clip=None`, which is why
  `train.py` prints the band and whether it was declared.
- **There is NO train-time exteroception noise.** `noise_depth` / `noise_height_scan` and the
  `noise_extero` property are gone, along with the two gates (`extero_corrupted`,
  `extero_noise_free`) that kept them from double-counting; both meta flags survive as
  statements about a recording that nothing reacts to. The sensor model belongs to the RL env,
  whose structured DR (per-episode plane tilt + blur sigma, extrinsics/intrinsics jitter,
  per-step additive, 2.5% dropout) is already baked into the recording — a scalar uniform term
  on top over-noised the one signal the generator uses to place a reference, and 0.1 m on the
  privileged scan was four cells of a 0.025 m grid. Pinned by
  `test_there_is_no_train_time_exteroception_noise`. The history/reference-window noise
  (`noise_scale`, `REFERENCE_WINDOW_NOISE`) is NOT this and stays: it models the generator's
  own prediction error feeding back into its history at deploy — exposure bias, not a sensor.
- **`contract` normalization is the default, and fits NOTHING.** Every constant answers to the
  camera contract and the robot's nominal stance, so the same robot yields the same numbers in
  every recording — a checkpoint's notion of "one metre from the robot" stops being a property
  of whichever terrain was on disk the day the normalizer ran.
  - `d` → **identity**. It already arrives declared-normalised, `(depth − 0.75) × 0.5` out of
    the contract, and clipped to `extero_clip`. Fitting it again would demote a contract
    constant to a dataset statistic.
  - `xyz` → **one declared scale** (`extero_xyz_scale`, 3.0 m) shared by both cameras and all
    three axes, so a metre reads the same number everywhere — across cameras too, which matters
    because both feed one transformer and the model has to compare them. The **offset is per
    camera and geometric**: the boresight's intersection with the nominal ground plane, derived
    at dataset build, rounded to 5 cm, recorded. For the shipped pair that is
    `[[0.55, 0.0, −0.79], [1.8, 0.0, −0.79]]` with ground z = −0.7869 (the ankle at zero joints,
    less the sole). Per camera because the two look 65° and 35° down from different heights.
  - `v` → identity, in every mode.
  - **It rides the existing `Normalizer`**, so it is still one `(x − mean)/std` of shape
    `(T*K,)`: the checkpoint, the affine folded into `fm.onnx` and the whole deploy path are
    untouched. Declared instead of measured costs no new plumbing to a robot.
  - **`extero_xyz_envelope` is a safety net, not a band.** Derived from each camera's own cone
    (±5.0 at scale 3.0) and asserted never to bind — sizing it to bite real geometry would
    repeat the ±2.0 depth-band mistake. The envelope comes from the clip band divided by the
    **corner cosine**, NOT from `max_dist`: a valid pixel carries a z-DEPTH, which becomes a ray
    length once divided by that cosine, so a frame corner reaches far past the band's nominal
    range. `max_dist` under-bounds the cone by exactly that factor. Widening `extero_clip`
    therefore widens this envelope too.
  - **Known cost, and it is real**: the offsets centre the NOMINAL FLAT-GROUND view, not the
    terrain a recording sampled. On one drop-off tile the far camera's empirical centroid sat
    5.77 m away (1.9 of the scale) while the near camera's was 1.54 m (0.5). The dataset PRINTS
    declared-vs-measured per camera at build so the residual is visible rather than assumed; if
    it stays that large on the real recording, the fix is a larger declared nominal range, not
    a fitted one.
- Other modes: `per_channel` fits one (mean, std) per sensor per channel group (`d` alone;
  `xyz` sharing one SCALE but keeping per-axis OFFSETS, since subtracting a per-axis mean is a
  rigid translation that centres the channel while preserving every relative distance; `v`
  identity). `per_sensor` pools across channels too, which lets xyz dominate the std and
  squashes validity — fine for a 1-channel scan, wrong for a 5-channel depth row. `metric` is
  one declared affine for everything, still right for a privileged metric scan. `per_element` is
  DELETED: a per-pixel affine is a spatially varying transform, exactly what a shared
  convolution kernel assumes away.

## One exported generator per FM run (`transfer/deploy/artifacts.py`)
Exports are named for the checkpoint they came from — `<fm run>_fm.onnx`, plus the checkpoint
stem when it is not `best` — so a teacher's `exported/` holds every generator trained against
it instead of each `play.py` silently replacing the last. `fm_artifact_name()` is the one rule
(used by `play.py` and `export_fm_onnx.py`); `resolve_fm_onnx()` is how every consumer finds
one (`gen_tracker_runner.py --fm_onnx`, `benchmark.py --fm`, `export_sensor_configs.py`,
`export_to_hardware.py --fm_onnx`).
- **With several in a directory the resolver RAISES and lists them.** Defaulting to the
  newest would reinstate, as a silent default, the ambiguity the naming removed.
- **`fm.onnx` is still the DEPLOYMENT name**: a robot installs one pair and the obelisk
  controller reads a fixed filename, so `export_to_hardware.py` uploads whichever local
  artifact it was given AS `fm.onnx`. The rename happens at that boundary and nowhere else;
  the resolver still accepts a bare `fm.onnx` as input.
- Pre-rename `fm.onnx` files (anything exported before `extero_tokens` ->
  `extero_sensor_count`) are refused at load and need one `export_fm_onnx.py` call; the `.pt`
  is unaffected.

## MVP: the depth-camera generator RUNS in sim2sim (2026-08-18)

The first generator trained on the 5-channel depth row drives the tracker through MuJoCo
sim2sim, and the architecture is **working** — this is the minimum viable product for
camera-only exteroception, replacing the privileged height scan on the deploy path.

Covered in MuJoCo, with the generator in the loop: **flat ground, stairs UP, jump-ups,
jump-downs.** **Stairs DOWN still needs work** and is the open item — the one locomotion mode
the pair does not yet handle.

The pair, so the result is reproducible:
- generator `logs/flow_matching/2026-08-17_12-53-18_depth_xyz5_contract_v1` (epoch 94 of 100,
  EMA weights, 27M), exported as `<run>_best_banded_fm.onnx`
- tracker `logs/rsl_rl/g1/2026-08-04_21-55-17_yaw_k12_v1` (`G1-teacher-clf-terrain-depth`),
  trained + recorded on `clips/g1_mb_v5_body.h5`
- `extero_channels=5`, `extero_norm_mode=contract`, `strides (1,2,2)` = 56 tok/camera
  (sequence 191), band ±5.575, `attention_mode=prefix`, `fm_update_every=12` (from the
  tracker's metadata, not K=1)
- scenes: `--scene terrain_scene` (in-distribution playground) and `--scene tile0_scene` (ONE
  real Isaac tile, manifest-identical to `g1_mb_v5_body.h5` tile 0 — verified, not assumed)

**This checkpoint stores `extero_clip` only because it was stamped by hand.** It was trained
inside the window between the depth band moving 2.0 → 5.575 and the stamping fix, so its own
config carries `null`; the band was recovered forensically (the saved envelope of 5.0 is
reachable only from 5.575, since 2.0 gives 3.5) and written into a `best_banded.pt` copy.
Anything trained after that fix stamps itself. See the band note above for why a `null` is
ambiguous rather than merely missing.

## The terrain is rebuilt from the manifest, so the GROUND can change without regenerating motion

A clip's only tie to its tile is an integer `terrain_id` plus a corner translation. The library
carries a `terrain_manifest` attribute -- per-tile lists of explicit boxes -- and the RL importer,
both MuJoCo emitters and every viewer rebuild the world from it at load. So terrain can be
rewritten offline while the motion stays exactly as recorded, which is what the passes in
`source/motion_gen/motion_gen/terrain/` exist for.

- `decorate.py` varies STAIR WIDTHS by rewriting the manifest. The shipped library had
  effectively none -- treads 0.801-1.156 m and approach/landing platforms ALWAYS exactly
  `size_y`, because all three generators forced it -- so against a 16x11 scan spanning ±0.5 m an
  0.8 m flight straddles both edges in every frame. Three arms per transition tile:
  `wide_stairs`, `narrow_landing`, `matched`. **Grow in place, never re-centre**: `size_y` is the
  bounding box of the tile's CLIPS' lateral travel, not of its flight, so `size_y / 2` is where
  the motion wandered and the flight sits off it by a median 0.149 m.
- The **decoration channel is VISUAL ONLY** and lives on its own prim (`decor.DECOR_PRIM_PATH`
  = `/World/decor`), which the depth cameras cast against and physics ignores. It is the one
  thing that changes what the robot SEES without changing what it stands on.
- **The RL TEACHER renders none of it** (`ClipTerrainImporterCfg.render_decor`, default `True`,
  set `False` on `G1TeacherTerrainSceneCfg`). Nothing in this stack avoids obstacles, so walls
  and rooms teach the tracker nothing while costing render time and depth pixels over ~12k
  iterations. It is a RENDER-time drop, not a second library: one `.h5` serves both roles, so
  no clip fingerprint or sha256 moves. Everything with the GENERATOR in the loop or being
  RECORDED turns it back on -- the `-play` cfgs, the FM tracker, the fine-tune task,
  `record_data.py`, `augment_registration.py` -- because a vertical surface it has never seen
  is what the generator misreads as a box.
- **What is decoration and what is not.** Flight and landing WIDTHS and the box SUPPORT
  variants (`solid` / `floating` / `floating_center_leg` / `floating_corner_legs`) are manifest
  rewrites of the WALKABLE ground and are unaffected by the flag. Only `decor=True` overlays,
  the `flat_tile_decor` rooms and a schema-1 clutter sidecar are dropped; a schema-2 ground
  variant is not decoration and passes through (asking for clutter with `render_decor=False`
  is refused rather than silently rendering nothing).
- **`render_decor` is invisible in every artifact.** `terrain_fingerprint` hashes the MANIFEST,
  so a decor-on and a decor-off run export byte-identical metadata, `tracker.onnx` included.
  The only record is the run's `params/env.yaml` and the importer's startup print, which is why
  it prints. Folding it into the fingerprint would make a decoration-only difference read as a
  GROUND difference to every check that compares it.
- Walls ring the RENDERED FLOOR rather than the manifest tile, up to four per tile, and the flat
  tiles get rooms. `test_wall_footprint.py` and `test_decor_prim_is_mandatory.py` pin both.
- `decor.py` imports NOTHING on purpose. The depth camera cfgs need the prim path at module
  scope, and importing the importer to read one string drags in `pxr` -- which, because hydra
  composes the env cfg BEFORE `AppLauncher` runs, binds pip's `usd-core` instead of Kit's and
  then segfaults when Kit boots a mismatched core. `test_no_pre_app_pxr_import.py` fails the
  build if it comes back.
- **Overlay fields that would diverge between simulators are REFUSED, not best-efforted**, and
  floating overlays render as floating in every MuJoCo/viser path rather than as solid.

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

# NOTHING IN THIS STACK AVOIDS OBSTACLES, and that is deliberate

There is no obstacle term anywhere in the MDP -- no reward, no observation, no termination.
The tracker follows a reference; the generator predicts a reference from depth and a command.
Neither plans around anything. `decorate.py`'s `wall_collidable: bool = False` records the
measured reason a wall is visual-only: a collidable wall the reference's arm sweeps through
blocks a wrist, fires `ee_pose_error` at 0.3 m and ends the episode for a reason the policy
cannot fix, and `undesired_contacts` at -0.1 against `clf_reward` at 10 is far too weak to
teach avoidance.

So "the robot walked at a wall" is partly CORRECT behaviour and is not a bug to fix here.
What is a real fault is the robot planning a JUMP or a CLIMB at one -- that is a
classification failure, and `scripts/rsl_rl/eval/terrain_class.py` is the instrument for it.
Keep the two apart: obstacle avoidance is an operator-level or planner-level concern that
this stack does not have and should not grow by accident.

# The generator's history has THREE producers, and only two say the same kind of thing

`history_from` decides what fills the FM's `N` context rows. They are reference-window
nodes, identical in width whatever fills them, so a mislabelled run is invisible to every
shape check -- which is why the arm is printed unconditionally by both runtimes and stamped
into every manifest.

| arm | a row is | available on a robot |
|---|---|---|
| `generator` (default) | the node this runtime SERVED that step -- its own output | yes, and it is what deploys |
| `clip` | the CLIP's node for that step (teacher forcing) | no -- there is no clip |
| `measured` | the ROBOT'S OWN achieved state that step | yes, in principle (see below) |

**The first two both mean "the reference applied at that step"; only `measured` is a
different kind of quantity**, and that is the point of it. The model is FITTED on reference
rows (`dataset` builds history from `history_ref`, using `history_state` only for the
per-row anchor), so the closed loop is distribution-consistent in kind and merely drifting,
while `measured` is real and off-distribution.

**Two of the five node blocks fall back to the prediction, because no robot can measure
them** (`window.MEASURED_HISTORY_FALLBACK`, mirrored in `window_np`): `base_pose`'s POSITION
half (no odometry) and `base_vel`'s LINEAR half (no base velocity estimate). Named at
HALF-block granularity because that is where the answer changes -- falling back over the
whole parent blocks would replace the robot's own orientation and angular velocity too,
leaving an arm barely measured at all, at exactly the same width. The rest is measured:
orientation from the IMU plus the heading change since the plan anchor, angular velocity
from the gyro, joints from the encoders. **38 of the deployed layout's 44 dims.**

**Under `window_anchor="plan"` this arm is not hardware-impossible.** History rows share the
PREVIOUS plan's anchor, so the position half is the robot's displacement over at most
`fm_update_every` control steps (0.24 s at K=12) -- short-horizon relative odometry, which
leg odometry can supply, not the global base position the `plan` convention exists to avoid.
`TODO(measured-history-odometry)` sits on the fallback list; the gap numbers below say the
expected gain from closing it is small.

**Both stashes must describe the SAME step.** The row pushed at step `t` describes `t-1`, so
the substitution pairs `_prev_measured_node` with `_prev_served_node` at PUSH time, where
both hold `t-1`. Doing it at stash time instead pairs a measured row for `t` with a served
node for `t-1` -- off by one step of travel, in the three columns that carry where the robot
is. A row is also **all-or-nothing**: before the first plan, three columns can come from
nowhere, and part-measurement-part-zero is a row training never presented, so the whole row
goes in as zeros under the null-history embedding.

## How far off-distribution it is, measured (2026-09-02)

`scripts/flow_matching/measured_history_gap.py`, one script over two sources, reporting the
gap in the units the model sees (the state normalizer's per-dimension sigmas, so 1.0 = a
full sigma of that block's training spread). Against the deployed pair:

| block | role | sigma p50 | physical p50 |
|---|---|---|---|
| `base_pos` | fallback | 0.019 | 1.3 cm |
| `base_rot6d` | **measured** | **0.029** | 3.9 deg |
| `base_lin_vel` | fallback | 0.150 | 0.13 m/s |
| `base_ang_vel` | **measured** | **0.694** | 0.48 rad/s |
| `joint_pos` | **measured** | **0.540** | 0.079 rad |

- **The two blocks a robot CANNOT measure are the two that move LEAST** (0.019 / 0.150). So
  the hardware constraint costs almost nothing -- the hybrid captures essentially all the
  available grounding -- and odometry would buy little. That is a result, not an assumption.
- **The arm IS substantially off-distribution**: 0.54 and 0.69 sigma on two measured blocks
  is a different input distribution, not a perturbation. Read any `fm_meas` result against
  this table, not as a clean grounding experiment. A matched retrain (recorder change ->
  re-record -> `history_source` in the dataset) is what would fix it, and is NOT done.
- Two sources agree, which is the cross-check: `--rollout` (served-vs-measured, closed loop,
  `logs/rollouts/v4_big`, joint_pos only) gives 0.518 against `--recording`'s 0.540
  (applied-reference-vs-measured, the FM's own 200k-window training set on talos).
- `stairs_dn_pure` is the outlier again -- 0.866 joint_pos, 0.084 rot6d, ~3x the others.
- **Only the per-block NORM is anchor-free**, and that is what makes the snap-row measurement
  valid at all: the anchor position cancels out of a DIFFERENCE and its yaw is a rotation,
  which preserves norms (every block's normalizer entries are equal within the block, so
  normalizing does not break it). The per-COMPONENT split of the base blocks is NOT anchor-
  free, so it is not reported. `--self_check` asserts both halves -- norms invariant under a
  deliberately wrong anchor, base components moved by it -- and the two tolerances are 100x
  apart on purpose: the recorded rows are float32, so an invariant moves ~1e-7, while a
  random yaw moves a component by O(0.1).
- The offset sweep in `--rollout` mode is a real check, not a convenience: `history[3]`
  pairs with `proprio[8]`, and the sweep MEASURES that (interior minimum at offset 5)
  rather than trusting a reading of the push order. The test is that the minimum is
  INTERIOR, not that it is deep -- one control step barely moves a joint next to the
  tracking error, so the curve is legitimately shallow.

## Where the arm is wired

- Isaac: `history_from="measured"` on the command; eval cell `fm_meas` joins `clip`/`fm`/
  `fm_tf` (`reference_overrides` gives it `ego_anchor_fraction=None` like `fm`, NOT `1.0`
  like `fm_tf` -- no clip row reaches the generator, and pinning it would make the arm differ
  from `fm` in two ways at once and destroy the pairing).
- MuJoCo/deploy: `gen_tracker_runner.py --history_from measured`, and an mjeval ARM
  dimension (`baseline_meas`) so one tracker can appear twice differing only in this.
  **There is no `clip` arm on the deploy side** and the ctor refuses one by name.
- **`--history_from measured` is a hard exit with `--fm_lead_ticks` / `--dump_replay`**: both
  exist to line this runner up against the C++ runtime, which has no measured arm at all, so
  `compare_replay.py` would report a divergence that is the harness's fault.
- Verified running in MuJoCo sim2sim on flat at 0.8 m/s: both arms survive 8 s, and 130 of
  135 logged columns differ (the 5 that do not are `time`, the three command channels and
  `fell`) -- so the arm is live rather than a silent no-op, and the generator's own plan
  moves by up to 0.37 m.

## The measured arm looks CATASTROPHIC on x0=0 and roughly NEUTRAL on real draws (2026-09-03)

3 stairs_up courses x 3 speeds = 9 cells per arm, paired exactly on (course, speed) -- same
tracker, same generator (`2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff`), only the history
source differs. Run once on the deterministic draw and then on four Gaussian seeds:

| run | `baseline` (served) | `baseline_meas` | paired delta |
|---|---|---|---|
| `x0 = 0` (`measured_history_9cell`) | **0.889** | **0.000** | **-0.889** |
| seed 0 (`measured_history_seed0`) | 0.556 | 0.222 | -0.333 |
| seed 1 | 0.333 | 0.111 | -0.222 |
| seed 2 | 0.222 | 0.222 | 0.000 |
| seed 3 | 0.111 | 0.222 | **+0.111** |
| **4 seeds pooled (36 cells)** | **0.306** | **0.194** | **-0.111** |

**The deterministic draw overstates the gap by ~8x, and the single-run conclusion drawn
from it was wrong.** Almost all of the -0.889 is the BASELINE being flattered: it goes
0.889 -> 0.306 the moment the noise is the noise the model was trained under. The measured
arm moves far less (0.000 -> 0.194). Over the four seeds the delta ranges -0.333 to +0.111
and the SIGN FLIPS -- seed 3 has the measured arm ahead, seed 2 ties -- so at 36 cells this
is "not resolvable", not "catastrophic".

That is `fm_variance_probe`'s finding again, harder: `zero_noise` is not a neutral sample,
it sits near the conditional's mode, and **any single-draw A/B on this stack is measuring
the draw as much as the arms.** Both mjeval and the seed sweep default to `x0 = 0` because
that is what the hardware controller runs, so the trap is on by default -- pass
`--gaussian_noise --fm_seed N` across several `--out` directories for anything that has to
hold for the model.

**What survives:** the measured arm is not better, and there is no evidence it helps. What
does NOT survive is "it breaks the generator". The eval-only question is still answered
"not worth shipping", but the honest reason is "no measurable gain", not "collapse".

**The seed sweep needed three fixes before it could mean anything**, all of the same shape:
- `runner.py` wrote `ZERO_NOISE_SEED` into every row's `seed` column unconditionally, so a
  seeded run recorded `-1` -- the one thing a seed sweep exists to record. Now
  `ArmRun.recorded_seed`.
- the table's and manifest's `zero_noise` were the literal `True`, so a Gaussian run
  recorded itself as deterministic. Now read off the loaded policy.
- **the noise stream was never re-keyed per episode.** `GenTrackerPolicy` builds one RNG at
  construction and draws once per re-plan, so an episode's noise depended on how many
  re-plans every episode before it made -- and an arm that falls early makes fewer. The two
  arms would have been scored on different draws from the first differing cell onward,
  drifting further apart the longer the sweep ran, with clean-looking tables throughout.
  `GenTrackerPolicy.reseed` plus a per-cell key `(fm_seed, terrain_idx, speed_idx)` fixes
  it; `reseed` is deliberately NOT part of `reset()`, since whether an episode resumes or
  restarts a stream is the harness's experiment design.

Read with the limits: **3 courses, all `stairs_up`** (`--max_courses 3` takes the
catalogue's first three, and the six ascending flights come first), so it is not a
cross-family read; and the pooled baseline of 0.306 says this cell set is hard for both
arms. The sign-flip p on any single run is **0.2499 against a floor of 0.2500** -- fully
SATURATED at three courses, so quote effects, never the p. The gap table was measured on the
deployed `2026-08-22` generator and on this `2026-08-25` one (0.540 vs 0.518 on `joint_pos`),
so the reading does not hinge on which of the two ran.

`view_cell.py` replays any of these cells and asserts the replay reproduced the recorded
row; it now reads `history_from` / `zero_noise` / `fm_seed` off the table rather than
defaulting them (it would otherwise have replayed a `baseline_meas` cell as the generator
arm, silently) and lists the arms and cells a run actually covered rather than every one the
`ARMS` table declares.

## AND THE ANSWER, ON REAL TILES: the history source is worth nothing; HEADING is worth 0.27

**mjeval's courses are the wrong instrument for this**, and its own docstring says why:
they are AUTHORED to vary one geometric parameter and are deliberately not trained tiles,
so a measured-history result there is confounded with perceptive distribution shift
(`isaac_tile_course`). The tell was in the numbers -- the K=1 baseline pools **0.306** on
three authored `stairs_up` courses against **0.875-1.000** on real library tiles at the same
speed. Everything above this line is therefore about a floor effect as much as about the arm.

Re-measured through `live_select.py` on tiles **6 / 20 / 40** of
`g1_mb_v5_body_mode11_decor.h5` (the `ksweep_meas` configuration: 3 speeds x 40 episodes,
K=1, seed-paired per episode because `live_select` keys each draw on
`[episode_seed, replan_index]`), 360 paired episodes:

| | served history | measured history | delta |
|---|---|---|---|
| pooled (360 ep) | 0.797 | 0.822 | **+0.025**, 42/75 discordant, p=0.36 |

**Indistinguishable from zero.** A tile-6-helps / tile-20-hurts split at 1.6 m/s (+0.225,
p=0.012 and -0.200, p=0.008) did NOT replicate at 2.4 -- both signs reversed -- and those
were 2 of 7 tests on one dataset. Read them as what multiple comparisons look like, not as a
family interaction. `report_measured_pairs.py` is the reducer (`live_select` has none of its
own); it prints exact McNemar on the DISCORDANT pairs, because a percentile bootstrap over
three tiles floors out.

**The real lever on tile 6 is HEADING, not history.** With `--steer_kp 1.5` holding the
flight centreline, the served arm goes **0.658 -> 0.925** and the measured arm **0.750 ->
0.892** -- and the measured arm's apparent advantage vanishes (-0.033, 8/20 discordant,
p=0.50). At 0.8 m/s it is 40/40 with zero failures against 20 successes and 17 tilt falls +
3 `fall/height` unsteered. So tile 6's unsteered failures were mostly the robot drifting in
heading and meeting the risers off-centre; the measured-history A/B there was measuring drift
recovery, not riser handling.

That is a coherent mechanism -- measured history tells the generator where the robot ACTUALLY
is, which only matters once it has drifted -- and it is the first account in this
investigation that survived its own controls. Two limits: `live_select` records no heading
error, so "the advantage scales with accumulated yaw error" is the untested prediction rather
than a measurement; and steering is OFF-DISTRIBUTION on stairs (|wz| = 0.0000 on every stair
clip across all 3500), so those rates are not comparable to the unsteered baselines and
`ksweep_meas` is `only_nosteer`. Tile 40 (box) DOES turn in training and is where steering
is in-distribution.

**Practical reading: on tile 6, heading control is worth ~+0.27 of success and the history
source is worth nothing.** That points at the operator/planner level, which per this file's
own framing is a thing this stack deliberately does not have.

## Filtered BC on measured history buys +0.019, i.e. the ceiling (2026-09-03)

The whole pipeline, on the measured arm: record 300k windows under
`history_from=measured` -> two-head critic -> advantages from the QUALITY head alone
(`--w_fail 0`) -> filtered BC on the top 30% (`--percentile 70`), stratified per-family x 3
speed bins x 5 progress bins. `scripts/rsl_rl/run_measured_finetune_pipeline.sh`, resumable
per stage.

| | parent | fine-tune | delta |
|---|---|---|---|
| pooled, 360 seed-paired episodes | 0.800 | 0.819 | **+0.019**, 40/73 discordant, p=0.48 |

**That is the ceiling, not an escape from it**: a perfect best-of-4 is worth +0.0207 and the
earlier generator-arm fine-tunes already took +0.011-0.014. Measured conditioning did not
change the causal structure, which is consistent with the conditioning alone being worth
+0.025 (p=0.36) untuned. One cell reaches p=0.039 (tile 6 @ 1.6) and is the 8th test in a
9-cell table whose neighbour at 2.4 goes the other way -- the same non-replicating shape as
the two retracted results above.

**It is a real null, not a broken run**, and each precondition was checked because any of
them could have failed silently:
- critic **AUC 0.904 on UNSEEN tiles** (the split is `val_UNSEEN_tiles`; 0.928 seen), against
  the 0.906 Isaac reference -- so the critic transfers to measured-history data;
- quality head **rho 0.752 against a label-shuffle control of 0.003**, `pred_sd 0.190` vs
  `target_sd 0.253` (under-confident, not collapsed to the mean);
- labels **78,413 positive / 182,899 negative / 0 null**, exactly 30.0% per stratum with a
  per-stratum threshold (+0.21 to +0.29), so `null_spread` nulled nothing wholesale;
- the fine-tune's `history_source` re-stamped `reference -> measured` from the recording.

**The parent ran its `generator` arm and the fine-tune its `measured` arm** -- each on the arm
it was TRAINED for, because `GenTrackerPolicy` refuses a mismatched pair unconditionally and
has no sim override by design. So conditioning and fine-tuning are confounded here; with the
conditioning alone at +0.025 there is no room in +0.019 for the confound to be hiding a real
fine-tune effect. A 2x2 would need that override, which does not exist and should not be
added casually.

What this does NOT answer is a generator TRAINED on measured rows from scratch rather than
adapted to them -- see `history_source` above; that pair (N=3 measured vs N=3 reference, same
data, same seed) is the matched experiment. It was run, and it is below.

## THE MATCHED TEST: training on measured rows is worth +0.003 (2026-09-03)

Two generators, `history_source=measured` and `history_source=reference`, same 361k-window
dataset, same seed, same `num_history_states=3`, same epoch, each served the arm it was
TRAINED for -- so no guard override anywhere, unlike every measured-arm eval before it.
`scripts/rsl_rl/eval_n3_pair.sh`. MuJoCo tiles 6/20/40 x 3 speeds x 40 seed-paired episodes:

| | ref_n3 | meas_n3 | delta |
|---|---|---|---|
| pooled, 360 episodes | 0.744 | 0.747 | **+0.003**, 38/75 discordant, p=1.000 |

**38 of 75 discordant pairs is a coin flip.** Two cells look interesting -- tile 20 @2.4
(+0.250, p=0.041) and tile 6 @1.6 (-0.150, p=0.070) -- and they point in OPPOSITE directions,
2 of 9 tests on one dataset. That is the third time this exact shape has appeared in this
investigation and the third time it has not survived; do not quote either.

**Four independent measurements now agree that the history source does not matter:**

| experiment | episodes | delta |
|---|---|---|
| measured arm fed to a reference generator, MuJoCo tiles | 360 | +0.025, p=0.36 |
| filtered BC onto the measured arm | 360 | +0.019, p=0.48 |
| measured arm fed to a reference generator, Isaac grid | 19,800 | +0.021, CIs overlap |
| **TRAINED on measured, matched control** | 360 | **+0.003, p=1.00** |

The matched one is both the most informative and the smallest. Fed, fine-tuned onto, or
trained on from scratch, conditioning the generator on the robot's own state does not move
outcomes -- against a clip ceiling of 0.907 and `fm` at 0.533, the generator is still the
limit and this is not the lever. Compare the ~+0.27 that heading control buys on tile 6.

Read at the scale it was measured: these are mid-training snapshots (~epoch 46 of 100, both
losses still falling and within 0.0001 of each other), 3 tiles = 3 clusters, and `best.pt`
is EMA weights selected on `val_ema/flow_mse`. The Isaac grid at 264 clips is the
better-powered version of the same contrast.

# Two families are excluded from FM training BY DEFAULT

Both because the generator cannot be asked to predict what it cannot see, and both are
train-time switches, so A/B-ing either costs no re-record:

- `skip_distractors` (True) -- the distractor windows.
- `skip_families` (`DEFAULT_SKIP_FAMILIES` = `["family_jump_off"]`) -- the BACKWARDS
  jump-off-a-box family. Two forward-facing cameras and nothing behind. `family_jump_off_FRONT`
  is the forward dismount, IS observable, and is NOT dropped -- do not add it by reflex.
  The eval harness measured those windows contaminating the exposure-bias diagnostic as well:
  teacher forcing buys +0.078 overall, +0.026 with `jump_off` removed, +0.006 with both jump
  families removed.

**Strictness is asymmetric BY NAME, and the rule lives in
`dataset.resolve_skip_families`** -- not in `train.py` -- because `build_train_val` is a
library entry point the probes, the composer and the tests all reach directly, and a default
that hard-failed every table-less recording would break all of them. A name in the DEFAULT set
is a standing REQUEST (drop it where it exists, print a line where it does not); any other name
was TYPED and its absence is a typo, which raises, because dropping nothing would read as
success. Dropping every window raises either way. `--skip_families ""` is the off switch, and
`test_the_family_exclusion_is_reachable_in_both_directions` pins that it is reachable.

The absent case is NOT only stale recordings: a flat-only library has no terrain manifest, so
`record_data._clip_family_codes` writes no family table at all, and a perfectly current flat
recording legitimately carries no families to name.

**`from_dict` maps a stored `skip_families: null` to `[]`, never to the default.** That value
MEANT "drop nothing", and a checkpoint's saved config is the only record of what its training
set contained -- letting it re-resolve would restate a later decision as a property of an
earlier run. This is the FOURTH instance of that trap in this file (`extero_source`,
`extero_encoder`, `attention_mode`, now this), and the only one that cannot corrupt inference.

The legacy per-step `.npz` path applies NEITHER filter, and needs no guard saying so:
`_build_train_val_windowed` already refuses any recording whose `history_anchor` is not
`"plan"`, and a per-step recording has none, so the format is rejected outright and for a
stronger reason.

**The two exclusion flags spell the same family differently, deliberately.**
`train.py --skip_families` wants the recording's own `family_names` entry
(`family_jump_off`); the eval harness's `evaluate.py --exclude_families` normalizes
(`jump_off`, via `sampler.normalize_family`). Do not unify them -- the raw names are what the
recording records. The harness's exclusion is a FLAG rather than a hardcode on purpose: a
family you can no longer measure is a family you can no longer make a decision about.

# The evaluation harness (`scripts/rsl_rl/eval/`)

`evaluate.py --stage plan|cell|grid` then `report.py <run> [<run> ...]`. Stages share one
manifest so nothing is retyped; `plan` and `report` import no IsaacLab, and `report --stage
reduce` imports no matplotlib. A cell is `(reference_mode, dr_preset, fm_label)`; the
generator is a cell DIMENSION, so several generators share ONE clip arm and every cell is
paired on clip id. See README "4b. Evaluate" for the runbook.

Things here that are load-bearing and were each found by a failure:

- **The tracker task has no `clip_end` termination**, so the harness ADDS one. Without it every
  episode runs the padded 20 s: a 300-frame clip becomes 6 s of motion plus 14 s of a frozen
  pose, and in the FM arm 14 s of being commanded the last frame's velocity off a finite tile.
- **`terrain_columns` returns only the DERIVED terrain flags.** The caller owes
  `terrain_start_step` / `terrain_end_step`; trusting it for the whole block zero-filled them,
  which left the derived values correct but made the report's degeneracy gate read every family
  as informative. Pinned by `test_terrain_columns_does_not_supply_the_primitives`.
- **Terrain completion is degenerate on 4 of 9 families** (`flat` has no pinned frame;
  `distractor` / `stairs_*_pure` are pinned to their last frame). The gate is
  `terrain_end_step < last_usable_step`, per clip -- NOT `has_terrain_span`, which is true for
  8 of 9. `terrain_progress` is the metric that still says something, and its denominator is
  the REACHABLE span (`min(terrain_end, last_usable) - terrain_start`) so a success scores
  exactly 1.0 on every family; the full span would score 0.83-0.91 on the terminal families and
  smear the success point-mass.
- **`*/none` is a NOISE FLOOR, not a determinism assertion.** With all DR and obs noise off and
  the clip pinned, two envs on the same clip and tile still diverge (measured ~0.3 within-clip
  relative spread of `steps_survived`). One env per tile gives the same spread, so it is not
  tile sharing -- it is simulator-level. The DR-uniformity check ("every env is the same
  robot") does hold and is asserted.
- **Only 5 of 8 physical DR terms can re-draw per reset.** `add_joint_default_pos` adds to the
  live `default_joint_pos` buffer (random walk), and `randomize_rigid_body_com` hands the FULL
  CoM array to `set_coms_index` alongside a SUBSET of env ids, which raises on a partial reset.
  `RESET_SAFE_PHYSICAL_TERMS` encodes the split and the manifest records it per term, because
  the three that stay `startup` leave their draw confounded with whichever clips an env runs.
- **`_widen` cannot be used for the OOD preset.** No clamping (turns `static_friction
  (0.3, 1.6)` into `(-0.025, 1.925)`), and its `len(value) == 2` test silently passes the
  camera extrinsics 3-tuples through UNTOUCHED, which would make `ood` identical to `train` on
  that whole axis. The safeguard is not a floor but an assertion that a widened range is never
  narrower -- a floor of 0.05 makes `viscous_friction`'s OOD low end higher than training's.
- **Envs may share a tile.** `filter_collisions` is on and the depth raycasts are per-env, so a
  dozen robots in one tile neither collide nor see each other -- confirmed empirically, not just
  from the config default. That is what lets `num_envs` be smaller than the column count.
- **Statistics are clip-averaged with a TILE-clustered bootstrap.** A binomial interval over
  pooled episodes has an actual coverage of 14-70% against a nominal 95% here, and gets worse
  with more draws per clip. Spend budget on tiles, not on `robots_per_clip`: 2x the robots
  measured as buying nothing, while 2x the tiles at half the robots is free and 21% tighter.
- `scripts/rsl_rl/eval/_paths.py` is the `sys.path` bootstrap that keeps `provenance`,
  `cli_args`, `play_plots` and `plot_style` reachable by bare name from the subfolder.
- `scripts/rsl_rl/plot_style.py` is the shared paper style (it lives in the PARENT because four
  plotters outside the eval use it, and `import robot_rl.anything` drags in IsaacLab).
  Importing it does not touch `rcParams`; only `apply_paper_style()` does.
- **A flag added to the parser is NOT forwarded to the cell subprocess unless `stage_grid`
  restates it.** It reverts to its argparse default in the child, silently. Measured twice:
  `--num_envs` ran a grid at 256 while the manifest recorded 1024, and `--terrain_ref`
  produced 3950 episodes with `n_plans == 0` and no error. This is the whole reason `fm_tf`
  is a reference MODE and not a flag -- a mode rides in `plan.json` and cannot be dropped.
  Pinned by `test_stage_grid_forwards_every_flag_the_cell_stage_reads` (a heuristic: a flag
  read via `getattr` is invisible to it).
- **The derived fall flag once inherited the PREVIOUS episode's orientation.** `_reset_idx`
  refreshes `root_pos_w`/`root_quat_w` but `projected_gravity_b` lagged, so the first fold of
  a new episode read the old tilt while reading the new height -- inflating `fell` by a fixed
  ~18.5% of the PRECEDING fall rate (0.188 generator arm / 0.182 clip arm). Per-step folds now
  use `root_quat_w`; `fold_terminal` takes `max(quat, gravity)` because inside the reset hook
  the LAGGING buffer is the correct one. Success rates were never affected -- they come from
  termination terms, never the accumulator.

## The generator's OWN sampling variance decides ~1 in 5 outcomes (2026-08-29)

`transfer/sim/fm_variance_probe.py`. One initial condition, one constant velocity command, one
REAL clip-library tile, N repeats, and **the only difference between repeats is the
flow-matching noise draw** -- no DR, no depth noise, no clip, same standing spawn, and MuJoCo
is deterministic. `terrains.isaac_tile_course` makes a library tile a first-class mjeval
`Course` (reading the manifest, the same producer `generate_clip_tile_scene` reads), because
terrain the generator never saw contributes its own failures and they would be
indistinguishable from sampling variance.

Deployed pair (`2026-08-20_16-36-10_mode11_decor_plan_v1` + `2026-08-22_19-14-39_plan_v1_lag48_aug`),
tile 6 = stairs_up 5 risers +0.850 m, tile 20 = stairs_down 7 steps -0.777 m, spawn 2.5 m
before the first riser on the flight's centreline:

| approach | speed | stairs_up success | stairs_down success |
|---|---|---|---|
| 2.50 m | 0.8 m/s | 0.798 (n=500) | 0.816 (n=500) |
| 2.25 m | 0.8 m/s | 0.836 (n=250) | 0.852 (n=250) |
| 2.50 m | **1.6 m/s** | **0.624** (n=250) | **0.668** (n=250) |

- **The outcome is bimodal, not a graded spread**: on stairs_up `progress_x_m` is p50 = p90 =
  8.06 m against a falls-only p50 of 3.61 m. A trial either walks the whole flight or dies on
  it, so distance-before-failure is mostly reporting that split.
- **Speed ~doubles the failure rate** (1.8x on both tiles independently). The variance is far
  from saturated at walking speed.
- **Start position does NOT matter**: 0.25 m closer moves success by +0.038 / +0.036 on n=250,
  inside binomial noise. That rules out a spawn-gait-phase artifact -- a third of a step
  length should have moved it if failures were phase-aligned.
- **Where it fails differs by family and moves with speed.** stairs_up always fails on the
  UPPER risers (r3 + r5 carry 80%+; r1 never fails). stairs_down at 0.8 m/s fails mostly OFF
  the flight (62% past it, 29% before it, ~8% on it); at 1.6 m/s that inverts (pre-flight
  29% -> 4%, on-flight 8% -> 42%). Falls are ~100% the TILT bound, not height.
- **The deterministic `x0 = 0` draw -- what the hardware controller runs -- is not a neutral
  sample.** It succeeds in 5 of the 6 cells, including `down_fast` where a third of Gaussian
  draws fail, consistent with it sitting at the conditional's mode. The exception is
  **stairs_up at 1.6 m/s, where it falls on riser 2 -- and riser 2 carries only 9% of the
  Gaussian failures**, so there the shipped configuration fails deterministically every time
  while 62% of alternative draws would have made it.
- **`--zero_noise_control` is the blocking check and it passes EXACTLY**: sd = 0.000e+00 on
  both progress and distance across repeats, so nothing other than the draw varies. That is
  the check an Isaac state-restore harness cannot pass, and it is why this instrument is
  trusted where the per-window one was not (see below).
- **`--scene_tag` is required when two of these run concurrently on one tile.** `CourseRig`
  writes `robots/g1/<robot>_<course>.xml` from the course before loading it, and for a speed
  sweep the geometry is identical, so the race tears the file silently rather than failing to
  load. The generated scenes are derived and deliberately NOT committed.

**The Isaac per-state best-of-K harness was ABANDONED, and why matters.** It branched K
generator samples from one state by copying an env's state onto K others. The copy matches on
the four PHYSICS buffers -- root pose/vel and joint pos/vel, all 0.000e+00 immediately after --
but NOT on what the controller actually READS: `obs_policy` differed by 2.375 in that same
instant, and one step later joint positions had diverged 0.2195 rad
(`logs/headroom/v1/headroom/restore_probe.json`). PhysX's contact manifolds, friction anchors
and solver warm start are not restorable either. A same-noise control -- every candidate handed ONE draw, so zero signal by
construction -- produced a best-of-K rise of +0.005 to +0.031 per family, matching or exceeding
the real arm's at every horizon from 0.24 s to 1.20 s (ratio 0.80-0.93). Independent
per-candidate noise MANUFACTURES a best-of-K rise of about its own flip probability, and two
estimates of that agreed: candidate-0-vs-parent survival disagreement 0.0157 against the
control's rise 0.0151. So the instrument had no resolving power and its "the gap is
bias-limited" reading was not supported. A second defect compounded it: the first action after
a restore was computed from the PRE-restore observation. The MuJoCo probe has no restore at
all, which is why it replaced it.

## The failure critic READS IN MUJOCO (2026-08-30), and the goal anchor is what unblocked it

`fit_value.py` trains a `P(fail)` head over the generator's frozen encoder. It could only be
read in Isaac, because three of its context features are goal-region quantities needing a live
`MotionSequenceCommand` -- which is why `fm_sample_explorer.py` records outcomes rather than
critic readings and why `TODO(attribution-critic-trace)` existed. `transfer/sim/mjeval/`
`goal_anchor.py` + `critic.py` close it. This matters because best-of-K needs to branch K
candidates from one state, and MuJoCo needs no state restore at all (a fixed noise tape replayed
from spawn is exactly deterministic), where the Isaac attempt at the same thing was abandoned.

**The anchor is EXACT and free, and it is NOT `Course.target_xy`.** Measured over 14703
episodes / 2289 clips / 75 real tiles:

    goal_pos_w - y_des[clip, clip_len - 51, 0:3]   is CONSTANT within a tile
        max within-tile spread   x 7.6e-06   y 1.5e-05   z 0.0   m

so the anchor in tile-local coords IS the clip's own terminal pelvis position, straight from
the clip library -- no layout rebuild, no torch, no IsaacLab. `goal_frame_yaw == 0.0` on every
episode, so the terrain rotation is the identity and `frame_offsets` is plain subtraction
(`assert_yaw_free` refuses a library that ever authors turned tiles).
- **`target_xy` is the far platform's CENTRE** and those platforms are 7.5-10.1 m deep while
  the clip steps only 0.8-2.1 m onto them. Bias: stairs_up **-3.06 m**, stairs_down **-3.17**,
  jump_on **-0.44**, jump_off_front **+4.92** (sign flips). That is 0.61-0.63 of a normalized
  context unit of pure offset on a feature the critic reads directly. It keeps owning
  `in_goal` / success and is untouched, so every mjeval number already measured stays
  comparable; the two goals answer different questions and are deliberately not unified.
- A per-tile MEDIAN costs `|x - median|` of 0.000 (jump_on) / 0.083 (stairs_down) / 0.176
  (stairs_up) m at the median, 4.3-6.5x better than `target_xy` at p90 -- and the residual is
  real clip-to-clip variation that the critic, whose Isaac anchor is per CLIP, was fitted
  across. `target_xy`'s error is a systematic offset, which it was not.
- **Tile -1 is REFUSED, not approximated**: the flat sentinel has no tile frame at all
  (spread 129 m), so the identity is undefined rather than noisy.

**Transfer measured against MuJoCo's OWN outcomes** (`critic_gate.py`, 48 episodes / 11 falls,
tiles 6 + 20): AUC(`p_fail_mean`) **0.946** with a label-shuffle control at **0.517 +/- 0.097**,
AUC(`p_fail_max`) 0.846, failures 0.772 vs successes 0.368. Isaac's held-out-tile reference is
0.906. The shuffle landing at 0.50 is what makes the rest mean anything -- it is the check the
abandoned Isaac harness failed. Read with care: 2 tiles, and the 2.52 s median lead time counts
only the 8 of 11 falls that ever crossed 0.5, so it says "when it fires it fires early", NOT
"it catches falls 2.5 s out".

**Two arms of the sabotage suite are BLIND, and they are findings about the CRITIC**: a
transposed scan fold moves `p_fail` 0.0049 against 0.0704 for zeroing the scan, and a
history rolled one row 0.0024 against 0.0577 for zeroing it. So it reads the scan's CONTENT
but barely its spatial LAYOUT, and the history's content but barely its temporal ORDER.
Neither is catchable by a delta test; `SCAN_HW` is `(11, 16)` and deliberately not square so
the wrong fold raises on shape instead.
- `ProprioRing` must stay an exact mirror of `record_rollouts.ProprioRing` -- control rate not
  re-plan rate, ABSOLUTE joint_pos, zeroed on reset. **Do not substitute
  `policy._obs_history`** despite its `history_length=10`: it stores
  `joint_pos - default_joint_angles`, post-clip, and its first push FILLS all ten slots
  instead of zero-padding. Three silent mismatches at once.
- The runtime is XYZW everywhere (`build_state` stores `wxyz_to_xyzw(qpos[3:7])`), so the
  frame helpers delegate to `deploy.quat_math_np` rather than restating the algebra. Hand-
  rolling it was wrong once, and the error was finite and plausible.
- **`isaaclab` is not IsaacSim.** `robot_rl.flow_matching.online` imports in 1.1 s pulling 152
  `isaaclab.*` modules and **zero** `isaacsim`/`omni`/`pxr`/`carb`, and coexists with mujoco
  and onnxruntime. The critic arm needs `env_isaaclab` only because `env_transfer` carries no
  torch, deliberately; without `--critic_ckpt` mjeval stays torch-free.

**Two live bugs this surfaced, both fixed here:**
- **35 tiles could not be built as courses at all.** `isaac_tile_course`'s `klass` inference
  substring-scanned for `"box"`/`"stairs_down"` against manifest families like
  `family_jump_on.h5` (20 tiles) and `family_stairs_dn_pure.h5` (15) -- while
  `family_stairs_up_pure.h5` DID resolve, purely because `"stairs_up"` is a substring of it.
  A scan that silently succeeds on one `_pure` family and refuses its mirror is matching
  spelling, not meaning; it is an explicit `FAMILY_KLASS` table now.
- **The re-plan edge only advanced when a classifier was attached** -- `last_plan_id` lived
  inside that branch -- so any second consumer of the edge would have seen EVERY step as a
  re-plan, silently, the flag being plausible either way.


## The best-of-K selector harness (`scripts/rsl_rl/eval/headroom*.py`)

`headroom.py --stage select` runs a V(s) best-of-K selector to the END of each episode: at
every re-plan it copies one state into `K` envs, draws a different chunk in each, rolls one
chunk, scores the resulting states with a `fit_value.py` critic and carries the winner
forward. `--stage restore_probe` is the state-copy gate; `headroom_select_report.py` pairs
arms on clip id, per family.

**The state copy is bit-identical, and getting there needed four fixes.** IsaacLab's
`write_*_to_sim` invalidates seven `ArticulationData` caches by hand and
`_projected_gravity_b` is NOT among them, so a restored env served its own orientation from a
current-timestamped cache -- the same buffer, and the same shape of bug, as the eval
harness's fall flag. `joint_acc` is finite-differenced, so `_previous_joint_vel` has to travel
too. `RayCasterCameraData.output` is a DICT, so a walk that only accepted bare tensors copied
each camera's pose and intrinsics while leaving the IMAGE behind. And the roll loop reused the
pre-restore observation for the first action of every window. `restore_probe` now checks every
buffer AND every observation term for EXACT equality and exits non-zero otherwise.
`_body_com_acc_w` / `_body_incoming_joint_wrench_b` are excluded by name: they rebind `.data`
on every read from PhysX solver output, so a copy cannot stick, and nothing in `robot_rl`
reads either.

**Isaac reproduces exactly across envs given identical state; the odd one out is an env that
was never written to.** Two branches restored from one state are bit-identical to each other
(median 0.000e+00, most pairs exactly equal), while a branch is never bit-identical to its
un-restored parent -- and self-restoring the parent does not close it. So a control that
compares RESTORED against UNRESTORED is asking a different question, which is why the selector
carries the group's state in the winning env rather than in a parent. `enable_enhanced_
determinism` and higher solver iteration counts were both measured and neither fixes the
residual; the amplification is the robot, not the solver.

**The arms only mean something against the SAME-NOISE control.** `--same_noise` serves every
candidate one draw, so every candidate is the same plan and any best-of-K rise is manufactured
by the simulator. `--no_selection` builds and restores all K envs but always follows candidate
0, so it must reproduce the `k=0` baseline; a gap there belongs to the K-env structure rather
than to selection. Both are cheap and both have caught a defect that produced clean, plausible,
entirely wrong numbers.

**Success is `rollout_outcomes.terminal_reward`, and it is NOT `report.py`'s rule.** The two
disagree on episodes that arrive and then fall (success here, failure there) and on ones that
fall and are later dragged over the line (the reverse). To compare this harness against the
episode harness, score the EVAL's own table under `terminal_reward` rather than assuming the
two agree -- done that way, the do-nothing control (`--no_restore --no_force_replan`) and the
episode harness agree to within one standard error on the same clips and generator.
`--goal_regions` is REQUIRED for the same reason `record_rollouts.py` requires it.

**Two ordering traps, both of which read as a model result.** The goal anchor must be resolved
INSIDE the reset, after the clip commits -- resolved when the allocator hands out the next
clip, every episode is scored against its predecessor's goal. And it must be seeded for every
group's FIRST episode, which no advance ever covers; without that those episodes are scored
against a goal at the world origin with zero tolerance and can never succeed.

## What the harness has established about the depth generator (2026-08-24)

Tracker `2026-08-20_09-04-12_mode11_decor_v1`, generator
`2026-08-22_12-26-34_mode11_decor_lag48_aug` (epoch 39/100). Clip-averaged, tile-clustered.

- **The generator, not the tracker, is the limit**: fm/train 0.468 against clip/train 0.930,
  paired delta -0.462 [-0.513, -0.411], p<0.001 over 264 clips.
- **Exposure bias is largely RULED OUT.** Teacher forcing (`fm_tf`) buys +0.078 overall, but
  removing `jump_off` drops that to +0.026 and removing both jump_off families to +0.006, with
  4 of 7 remaining families NEGATIVE. `jump_off` jumps BACKWARDS, which no forward-facing
  camera can see, so teacher forcing there smuggles in an unobservable -- that family is to be
  dropped from the eval AND from FM training.
- **DR is irrelevant**: fm/none 0.495 -> fm/train 0.468, inside the interval. And `dr_preset=none`
  sets `enable_corruption=False`, so the headline gap was measured with the tracker's
  reference-window noise OFF; turning on ALL DR costs at most 2.7 points. The tracker is already
  near-indifferent to i.i.d. jitter, so adding MORE of it is not the lever.
- **The FM error is structured, and differently per family.** Fraction of window-error energy in
  {const, const+ramp} over the 13 observed offsets (i.i.d. baseline 1/13=0.077, 2/13=0.154):
  base_pos 0.49/0.87 on the transition families vs 0.93/0.99 on the `*_pure` ones; `ang_vel` is
  0.04/0.11, i.e. SUB-white, nothing to model there. Growth from node 0 to node 50 is 5.1x, so
  the band over-covers the near node and under-covers the far one. The rigid component is
  under-covered 3.2-4.5x. Any perturbation built for the tracker should ramp with look-ahead and
  be drawn per re-plan, not per step.
- **`stairs_dn_pure` is the outlier and is NOT explained by any of the above**: 0.048 closed
  loop, 0.073 teacher-forced, 0.087 after the lag-ring fix, against a clip ceiling of 0.939.
  Normalised per tread all four stair families share an ~8-14%/tread hazard, so its 0.048 is
  25.4 treads of exposure rather than a unique defect. Deprioritised in favour of the four
  families that matter on hardware: `stairs_up`, `stairs_down`, `jump_on`, `jump_off_front`.
- **~48% of FM-attributable falls on the transition families happen on FLAT ground**, before the
  terrain is reached.
- **The augmentation was VERIFIED SOUND**, not a suspect: the delta-0 self-check's scan residual
  is exactly 0.0 (the script states a non-zero scan diff at zero offset is what would mean bad
  placement), and displaced poses were achieved to 0.35 mm / 0.24 deg median. The alarming
  `frac_gt_1mm = 0.515` is the depth DR being re-drawn.


## Measuring what SAMPLING is worth: the estimator traps (`eval/seed_headroom.py`)

`evaluate.py --fm_seed K` names the generator's initial-noise draw so it can be varied with
everything else pinned -- `--seed` cannot, because it also moves the DR draws, the spawn jitter
and the observation noise. The draw is keyed `(fm_seed, within-episode re-plan index)` and
**deliberately not by env**: several robots on one clip under one seed must be served the
IDENTICAL noise, because their spread is the simulator floor, and keying by env puts the effect
being measured inside its own control. `docs/seed_headroom_report.md` is the write-up.

**`max over seeds` is biased upward, and the obvious fix is biased DOWNWARD -- by more.** A
maximum over K noisy per-seed rates finds a high one whether or not the seed matters, so the raw
gain overstates. The natural correction is a permutation control (repartition a clip's episodes
into pseudo-seeds), and it is a valid TEST -- it returns zero on a synthetic null. It is not a
valid ESTIMATOR: it measures the inflation under the null of NO effect, and where an effect is
real the max is partly driven by genuinely better seeds and is less noise-inflated than the null
predicts, so subtracting the full null bias over-corrects. Measured against synthetic ground
truth it recovered 31%. What works is HELD-OUT selection -- pick the best seed on some robots,
score it on the rest -- plus a sweep of the selection sample: still climbing means the ceiling is
above the estimate. **Validate any estimator here against synthetic data with a known answer
before quoting it**; all three were, and two of the three failed.

Also measured while building it, and each cost a run:
- **Isaac is NOT reproducible run-to-run here.** The same `fm_seed` in two processes reproduced
  per-clip rates exactly in 58% of clips. The docs' determinism claim covers re-running a
  simulation; it did not hold. So a same-seed repeat is a REQUIRED arm, not a nicety.
- **Do not use `p(1-p)/R` as the within-clip floor.** Robots sharing a draw are strongly
  correlated -- measured same-seed variance ~8x smaller than independence predicts.
- `--num_envs` above what a plan can populate is refused with `pinned to clip -1`, not run;
  1000 envs OOMs a 12 GB card where 750 does not.

## Terrain misidentification: real, and CLOSED-LOOP ONLY (`terrain_class.py`)

Field observation: the robot sometimes reads a staircase as a box and plans to jump onto it.
`terrain_class.py` turns that into a number -- k-NN over a reference set of real clip windows,
in the generator's own output space, classifying each re-plan flat/jump/stairs. Matched to
`stairs_up`+`stairs_down`, matched classifier:

| | stairs->jump |
|---|---|
| open loop, true clip window (the control) | 0.103 [0.089, 0.121] |
| open loop, FM prediction | 0.104 [0.084, 0.131] |
| **closed loop, FM prediction** | **0.174 [0.146, 0.205]** |

So given in-distribution conditioning the generator reads stairs exactly as well as the ground
truth does; drive it with its own drifted history and depth from a drifted pose and the
confusion appears. **The fix is on the loop/conditioning side, not "the model cannot see
stairs".** NEXT: dump the depth image alongside the flagged plans to separate a perception
failure (the image genuinely looks box-like) from a generation failure (image fine, plan wrong)
-- the one split this number cannot make.

**The classifier is a POPULATION statistic and NOT a per-plan label.** It was only ever
validated against real CLIP windows (control accuracy 0.859) and is then applied to GENERATOR
output, which lives elsewhere in that space -- so its accuracy where it is actually used had
never been measured. Measured now, against |served - clip| as a label-free ground truth:
within matched `clip_step` bands a flagged plan is no more divergent from the clip than an
unflagged one (~1.0x; 0.62/1.36/1.05/1.07 over four bands). It separates only at the extreme
tail (2-4x). So the rate above is usable -- the population really has shifted -- but "this
plan is a misread" is not a claim the label supports, and selecting plans to eyeball by it
gives a sample that says nothing about whether anything was wrong. `--dump_flagged` therefore
selects on |served - clip| and keeps BOTH ends within a step band, since prediction difficulty
rises through an episode (rank correlation 0.29). That metric has the mirror-image limit: a
plan 5 m from the clip is a robot already lost, so its extreme end selects consequences rather
than causes. Neither proxy isolates the cause alone.

**The wall-as-box behaviour REPRODUCES in MuJoCo sim2sim, on a joystick** (2026-08-26).
`generate_stairs_scene.py --wall` puts a visual-only 90-degree corner behind the spawn;
driving at it, the policy reads the wall as a box and tries to jump onto it. That takes the
hardware observation out of a one-off field report and into a local loop that costs nothing to
re-run -- and it is the sharpest confirmation that this failure lives on the exteroception
side rather than the loop/conditioning one.

Build the reference with `terrain_class.py --windows <pred_tgt.npz> --holdout 0 --out ...`;
pass it to the eval with `--terrain_ref`. **Always read the rate against the control**, never
as an absolute: the classifier's own stairs->jump confusion is ~0.10 on these families and was
unstable (0.010 to 0.116) until the reference reached ~12k windows.

## Visual clutter: re-render a recording's pixels, don't re-record (2026-08-26)

The generator had essentially never seen a vertical surface within camera range -- 0.0% of
flat clips have a wall in reach, terrain tiles' walls sit a median 9.6 m past where the motion
stops, and `family_distractor` is skipped by default. A wall at 2 m is therefore out of
distribution in the direction that most resembles a box, which is what the robot does with it.

The fix needs no re-record and no teacher retrain. `augment_registration.py --visual_only
--clutter <sidecar>` re-renders an existing recording's depth at every recorded pose with
scenery added; the sidecar carries only `decor` overlays, so the ground mesh and every tile
corner are bit-identical and the recorded world poses stay valid. Verified: `ref_future` comes
back bit-identical, and two different clutter variants give bit-identical height scans.

- `motion_gen/terrain/clutter.py` authors a variant, vetted against the CLOSED-LOOP recording
  (`build_clutter_occupancy.py`) rather than the open-loop clip -- the robot is where it
  actually went, not where the clip said. `author_clutter.py` writes a ~200 kB sidecar, not a
  6 GB library copy; `ClipTerrainImporter.decor_overlay_path` merges it at load.
- **What limits exposure is ANGULAR SIZE, not range.** Pillars sit closer than walls (1.64 m
  vs 2.24 m median) and are 13x less visible, and every class realises only 6-10% of the frame
  it would fill if centred. The far camera is 30x26 px over 70x42 deg, rides the torso, and
  pitches 10-16 deg nose-down while walking, so it stares at the ground 2-3 m out. Overhangs
  at 2.1-2.6 m changed ZERO pixels in ZERO windows and are off by default.
- **Wall HEIGHT is unobservable.** At 3 m the frame's top edge cuts a wall at z = 0.87 m, so
  1.0 m and 3.0 m walls are byte-identical images. Vary it for the viewer, not the model.
- Displacing the robot AND adding clutter needs the penetration filter: the veto is one body
  radius plus clearance and the displacement spends most of it, so `--num_augments` together
  with `--clutter` drops windows whose displaced reference enters a volume.
- `fm_clutter_sensitivity.py` is the before/after number: one checkpoint over the clean and
  cluttered halves with ONE shared noise draw, so conditioning is the only difference. Windows
  whose depth did not change are a control that must return ~0 -- and it refuses to report
  anything when that group is empty, which is what running it against DR-ON files looks like.
  Its threshold is the control's **p99.9, never its max**: across two clutter draws of one
  checkpoint the max moved 0.0066 -> 0.0241 m and swung the headline 47% -> 13%, where p99.9
  held and the two draws agreed to five points.
- `merge_recordings.py --variant_groups`: re-renders of ONE rollout must share an episode
  numbering, or `build_train_val`'s by-episode split puts one variant in train and another in
  val. With k re-renders the chance a val episode has no counterpart in train is
  `val_fraction ** (k-1)` -- at seven files and a 5% split, 2e-8. Not only a reported number:
  `best.pt` is selected on the val metric.

# The TWO-BOX overfit set: one OOD obstacle, an IC grid, and a critic (2026-09-06)

`worktree-box2-finetune` = `worktree-measured-history` (the filtered-BC/AWR tip) + a merge of
`worktree-fm-reward-finetune` (the clip-free free-form terrain). **NOT** off
`worktree-fm-adjuster`, which is that same tip plus 15 adjuster commits none of this needs.
`worktree-bestofk` is deliberately UNMERGED -- 8 conflict hunks in `branch_probe.py`, which
nothing here reads -- so its `oracle_gain.py` and its branch-probe defect fixes are still out.

**The obstacle is `mjeval_box2_h040`, cloned into Isaac** (`motion_gen/terrain/box2.py`,
`scripts/rsl_rl/make_box2_terrain.py`): two 0.40 m boxes 1.20 m deep, 3 m apart. Chosen because
it is the worst course in the mjeval catalogue and the only one that fails at EVERY speed --
0.060 unsteered / 0.145 steered over 20 flow-noise seeds x 10 speeds, and 0/10 at every speed on
the deterministic `x0 = 0` draw -- while the SINGLE 0.40 m box scores 0.70. The second box is
what breaks it, so it is a MODE choice rather than a capability limit, and the generator visibly
proposes different modes there (`view_cell.py --plans`). Overfitting to it is the point.

- **Geometry is MIRRORED, not imported**: `source/motion_gen` stays free of `transfer` as it
  stays free of IsaacLab, and `transfer/sim/tests/test_box2_matches_mjeval.py` imports both and
  requires them to agree -- comparing distances FROM THE SPAWN, since mjeval puts its origin at
  the spawn and a free-form tile is corner-anchored.
- **The goal is mjeval's, MAPPED**: mjeval scores arrival inside a 2.0 m-deep rectangle,
  `freeform_goal_reached` tests a downrange LINE plus a lateral band, so the line sits at the
  rectangle's NEAR EDGE and the two agree on the boundary that decides the outcome.
- **The IC grid is a grid over TILES**: one tile per approach distance {1.5, 2.0, 2.5, 3.0, 4.0}
  m, identical boxes on every one, because the reset already reads the spawn off the route
  block. The speed axis is a COMMAND, pinned per recording run by
  `freeform_speed_min == freeform_speed_max`. Held out: **(2.25 m, 1.00 m/s)**, off the training
  grid on both axes and INTERIOR to it, in its own `box2_holdout.h5` -- a separate file rather
  than a "do not sample" flag, because a runtime filter is the shape of leak the FM's episode
  split already paid for.

## The far camera had to go back, and the restore was silently wrong first

Since `efe93b4e` the POLICY reads the near camera alone -- taken ahead of a teacher retrain that
**has not happened**. So every tracker checkpoint on disk expects 3002 inputs where a near-only
task builds 2222, and **none of them can run the free-form task at all**.
`enable_policy_far_camera` + the `G1-box2-finetune` task put the term back.

**It must go back in its TRAINED SLOT, and the obvious spelling puts it in the wrong one.**
Written as an annotated field (`depth_far: ObsTerm | None = None`) it lands BEFORE `depth_near`,
because `configclass` orders ANNOTATED fields ahead of un-annotated ones and `depth_near` carries
no annotation. The vector is then the right WIDTH with two 780-wide blocks transposed:
`load_state_dict` passes and the tracker reads the far camera as the near one. Measured over 64
episodes:

| order | reached_goal | steps_survived (med) | track_reward |
|---|---|---|---|
| transposed | 0.000 | 143 | 0.57 |
| correct | **0.188** | **517** | **0.69** |

Nothing in the first row reads as a config bug; it reads as a policy that cannot do the task.
`test_box2_task_obs_order.py` pins it against the deployed artifact's OWN `observation_terms`
block, plus a mechanism check on the declared field order that still runs where `logs/` is absent.
Note `reference_window` / `command_window` appear in that block but are NOT cfg observation
terms -- they are `RefEncoderMLPModel` inputs, which is why the MLP takes 2490 raw + 512 encoded
= 3002 rather than the block's full 3101.

## Five clip-shaped assumptions the recorder had, and the pelvis-height trap

A manifest-only library refuses every gather by name, so each of these failed loudly -- but the
FIX for one of them is where the silent hazard was:

- `goal_region.goal_anchor_world` recovered the anchor from the placed clip's terminal pose; it
  now dispatches to `freeform_goal_anchor_world`, the exact inverse of the lift
  `freeform_goal_reached` applies to the robot. Both read the SAME route row.
- **The anchor is a PELVIS height, not the walkable surface.** `height_tol_m` is applied to
  `|root_z - goal_z|` and the clip anchor is a BASE pose, so a surface-height goal puts a
  standing robot ~0.79 m outside a 0.5 m tolerance: `in_goal` could never fire while the
  termination still did, and every arrival would have recorded as a failure. The standing height
  is read off the asset's own `default_root_pose` -- the same quantity the spawn adds to
  `start_z`, so the two ends of an episode are measured against one number.
- `goal_travel_m` and `last_usable_step` likewise gathered the clip (the section's own
  start-to-goal run, and the time-out horizon).
- `record_rollouts` required terms named `base_fallen` / `reset_on_clip`; the fine-tune tasks
  spell the same things `base_height` / `reset_on_section`.
- `rollout_outcomes` derived `terrain_id` from a per-CLIP table, so it was **-1 on every row** --
  and on this terrain the tile IS the initial condition, so the IC an episode ran at was
  unrecoverable from the table. It falls back to the importer's `env_terrain_id`.

`merge_rollouts.py` joins the per-speed stems for `fit_value`, renumbering `episode_uid` past
the running max: the uid is assigned per RUN from 0, and `fit_value` splits BY EPISODE precisely
so near-duplicate windows cannot straddle the split -- a collision merges two unrelated episodes
into one group and deals their windows to both sides.

## What the first run measured, and why it is 12x below MuJoCo

5001 episodes at 0.50 m/s, ~800-1200 per tile: **pooled reached_goal 0.059**, against mjeval's
0.15 unsteered / 0.75 steered on the same course and the same pair. The gap is REAL and is
DOMAIN RANDOMISATION, verified on the composed cfg rather than assumed: all six depth DR terms
live (extrinsics, intrinsics, noise resample, per camera), seven physical DR terms live, and
`enable_corruption=True` on the policy obs. mjeval has none of it.

**That is the desirable direction**, and it closes `record_rollouts`' own
`TODO(record-camera-dr)`: every previous recording ran on a `-play` task, where
`apply_depth_play_trims` nulls all six camera events, so "a critic fitted here has only ever
scored a perfect camera". This is the first recording where it has not.

Not a scoring artifact, checked: of 616 truncations, 293 are genuine arrivals (terminal
downrange 0.009 m, height error -0.12 m, both well inside tolerance) and 323 are plain
time-outs. 88% of episodes terminate on a fall.

# FINE-TUNING THE GENERATOR ON TERRAIN: the loop, and what it has established (2026-09-09)

Take a trained generator and improve it on ONE hard course by re-weighting its own behaviour,
with no simulator in the training loop: collect rollouts, grade them, fit a value function,
weight or filter the chunks, update the generator, x5. `scripts/flow_matching/run_finetune_loop.sh`
runs the whole thing and is resumable per stage; **`scripts/flow_matching/FINETUNE.md` is the
runbook** and carries the artifact provenance, the library fingerprints and the traps.

This is NOT the teacher PPO fine-tune `README.md` §4c documents. Nothing here calls `train.py`.

**WHAT IT BUYS, on the two-box course, paired at the ENV level** (both arms draw the same reset
stream, so every env's first episode opens at a bit-identical downrange -- max |diff| 0.00e+00,
which is why the paired interval is the honest one and the level intervals are conservative):

| arm | paired delta vs parent | McNemar |
|---|---|---|
| AWR on the progress reward | +0.3510 [+0.2514, +0.4476] | p=1.8e-226 |
| **+ the velocity-mismatch knob (0.30)** | **+0.4817 [+0.3905, +0.5831]** | p underflows |
| filtered BC, top 30% | +0.3531 [+0.2124, +0.5124] | p=9.2e-242 |

**Filtered BC MATCHES AWR here** -- the intervals overlap almost entirely and fBC's is 55%
wider. On this course the advantage WEIGHTING buys nothing over a top-30% filter, which is the
ablation's real finding. Up-into-down is +0.2069 and the in-distribution control (routes over
the original clip tiles) is +0.1046, so the OOD fine-tune did not cost in-distribution skill.

Every number above is regenerable from the repo: `fig_reduce.py` -> `plot_finetune_bars.py` for
the figure, `paired_delta.py` for the paired test, `variance_audit.py` for what is inside the
episode count. They route their episode selection through `awr_buffer.first_per_env` and their
success rule through `rollout_outcomes.terminal_reward`, so the figure, the report and the
paired test cannot describe one recording differently.

## The four algorithms are ONE selectivity ladder, and only the update differs

`ALGO=awr|fbc|arrived|bc`. Collect, grade and buffer churn never branch on it, so a difference
between arms is the objective and not the data -- pinned by parsing the script
(`test_finetune_loop_algos.py`), because that property lives in shell where nothing type-checks
it.

| `ALGO` | update | critic | selects |
|---|---|---|---|
| `awr` | `w = clip(exp(A / (beta * sd A)), 0, 20)` | yes | ~10-15% effective |
| `fbc` | filtered BC on the top 30% by advantage | yes | 30% |
| `arrived` | BC on episodes that ARRIVED before failing | **no** | ~8.8% |
| `bc` | plain BC on every row | **no** | 100% |

**AWR's hot limit is unweighted BC on the labelled rows and its sharp limit under `global`
normalization IS filtered BC**, so the ladder only means something if each arm is actually its
own algorithm: `bc` passes no `--labels` at all rather than a large beta, and `fbc`/`arrived`
filter rather than weight.

**`bc` also needs `--unconditioned`, and that is not obvious.** `is_plain_arm` is true for
`filter_label`, `unconditioned` or `awr_beta > 0` -- plain BC passes NONE of those, which is
the whole point of the arm, so without the flag it takes the advantage-CONDITIONED path,
grows an embedding no label ever reaches, and writes a checkpoint with no serve arm that every
consumer then refuses. That is not hypothetical: `eval_bc_s3_i1` is three log files and no
data. BC was the FOURTH plain arm and `is_plain_arm`'s docstring had already recorded the rule
going stale once, "the moment AWR was added as a third plain arm".

## The progress reward, and why the terrain-only one is maximised by standing still

`R_t = R_terrain + gate * exp(-|v_err| / sigma) + gate_worst * C * reached_goal * is_last`,
`C = 10`. The terrain-consistency grade alone rewards not touching the terrain, and it showed:
a terrain-only run drove forward progress 11.50 -> 3.95 m per 1000 steps WHILE survival rose.
So the velocity term and an ARRIVAL bonus are what make it un-gameable by waiting.

**The success bonus is TERMINAL and gated on the episode's WORST chunk.** Two other shapes were
measured and rejected: adding `C` to every chunk made the target near-binary (unsuccessful rows
0.11, successful 0.95, with terrain and velocity squeezed into the bottom sixth of the range),
and distributing `C/n` diluted it to 2% of the reward. A terminal bonus keeps it where it is
earned and lets the CRITIC propagate it backwards, which is what bootstrapping is for.

**The gate means arriving destructively pays nothing** -- of 352 successful two-box episodes,
every one reached the goal by dismounting a box, because a correct two-jump traverse is rare
enough that the behaviour policy never produced one and AWR can only reweight what it is shown.
That is also why the course declares box-top SPAWN AREAS.

## Isaac TERMINATES on the strict tracking bounds, and the bound is 0.12 m

`terminations.tracking_violation` (`mdp.fm_tracking_violation_termination`) ends the episode at
the union of lateral 0.20 / height 0.12 / tilt 30 deg around the SERVED node. It goes through
`tracking_error.violation_checks` rather than restating the tests, so the termination and every
offline reader of `first_violation_step` cannot disagree about what a violation is.

**It replaced an offline LATCH, and the latch was quietly destroying the success signal.** The
recording used to run on past a violation and every consumer cut the episode there afterwards
-- so a run reached the goal, banked the terminal bonus, and the offline cut then deleted the
row it was paid on. Measured: the last re-plan survived on **4.7%** of arriving episodes, so of
728 arrivals ~34 kept their bonus in the return. The critic was fitted on a reward with
essentially no success term in it, and every log read healthy. Terminating means nothing
downstream truncates anything, and `reached_goal` and `rollout_outcomes.terminal_reward` become
the same question.

**0.12 is measured, not fitted.** Watching the course, proper box jump-downs peak at
0.130-0.182 m against the old 0.20 bound -- nowhere near it -- while stairs-down-off-a-box
dismounts approach it, so the bound separates a correct dismount from the wrong MODE rather
than catching a fall. Its cost was measured on the real opening data set BEFORE the change
(`awr_vm30/inc_001`, 2000 parent episodes, DR on): clean arrivals 21.70% -> 16.85% of all
episodes, windows surviving 95.7% -> 93.4%. Ground-start clean arrivals collapse 4.18% ->
0.46%, so at this bound every clean success in an opening set is a box-top dismount -- a
scarcity that predates the change. The tilt bound never binds before the height one on that
data. `fit_tracking_thresholds.py` still does not exist, so the spec is still `bootstrap: True`
and publishing consumers still refuse it.

**The bound's discriminative power is a property of the POLICY, not the terrain**, which is the
thing to remember before quoting any of it. Peak |height error|, arrived vs fell: the vm30 arm
separates (0.092 / 0.224 median; at 0.12 it flags 0.219 of box-top arrivals and 0.866 of
falls) while the PARENT is inverted -- the strict union fires on 0.953 of its arrivals against
0.807 of its falls. And restricted to ground starts the "cost" on vm30 rises from 0.308 to
0.586, because pooling box-top dismounts with full traverses compares a single dismount against
a two-jump course.

**Whether the bound tracks the wrong MODE is UNMEASURED.** It is a hypothesis from watching
episodes, and there is no automated mode detector: `terrain_class.py` is validated as a
POPULATION statistic and its own numbers say "this plan is a misread" is not a claim it
supports. `transfer/sim/mode_probe.py` -- human labels, blind and interleaved, N noise draws on
one tile -- is the only instrument, and `watch_tracking.py --spec` is how to see what a given
bound catches (it pauses the replay ON the latch; press a key with the VIEWER focused).

## Success is defined ONCE, and three places have to agree

* the REWARD pays on `reached_goal`, gated by the worst chunk;
* the CRITIC keeps every window up to the first failure and drops the rest -- a no-op now that
  the episode ends there;
* the EVAL reports `terminal_reward >= 0`: arrival BEFORE the first failure, so a fall in the
  run-out after the goal is a success and a robot dragged over the line after failing is not.

`FAILURE_RULE` decides what "failure" means for the last two and is `strict` -- what Isaac now
terminates on. **Score a recording made BEFORE the termination with `loose`**, or the strict
rule is applied to episodes it never ended: it then fires on 99.5% of them and reports 0.0049
against the fall rule's 0.1051. `rollout_outcomes.assert_tracking_bounds_agree` refuses a
comparison whose two arms were latched at different bounds, because nothing in the numbers
shows it -- both arms have the column, both are populated, both are plausible.

## Training draws every spawn area; every EVAL draws the initial GROUND one

Raised areas exist so the second half of a traverse appears in the data at all. Scoring from
them reports a fraction of the course as a traverse, so `run_quick_eval.sh` passes
`SPAWN_FROM=ground` on every terrain and prints it. It costs nothing in comparability, checked
rather than argued: every family authors its ground area at exactly
`freeform.FREEFORM_SPAWN_JITTER_M`, which is also what the reset synthesises for a tile
declaring none -- so a ground-only draw is BIT-IDENTICAL to the pre-spawn-areas libraries every
earlier number was measured on.

## Reduce FIRST-PER-ENV, always

A round-based stop ends when the SLOWEST env finishes its k-th episode, and an env finishes
more episodes precisely when its episodes were short -- so the surplus is drawn from the
fast-failing tail and pooling it tilts the sample back the way it came. Measured bias 0.15 on
one in-distribution tile. Worse is `--num_episodes`, a TOTAL-count stop, which cuts every slow
episode still in flight: one arm read 0.0005 against a parent's 0.1960 that way, a policy that
looks destroyed and is not.

## The terrain contract every course satisfies

One `.h5` per course, manifest-only: terrain and zero clips, still passed with `--clip`. All
nine regenerate with the terrain fingerprint UNCHANGED (`generator_config` sits outside
`fingerprint_terrain_manifest`, because a commanded speed is not geometry). Each declares
`commanded_speed_range_mps` under the key `clip_io.read_commanded_speed_range` actually reads
-- box2 and the curb shipped an `intended_speeds_mps` LADDER for weeks, which nothing reads, so
the whole declared-band path was a silent no-op and the loop used its own default.

`varied_courses` is the one where a tile is GEOMETRY rather than an initial condition: 30 box +
30 up-into-down tiles in one library, with the spawn-area chain and a continuous band carrying
the variation a fixed course puts on separate tiles.

## TRAPS, each of which cost real time

* **`height_tol_m` MEANS OPPOSITE THINGS AT ZERO.** `terminations.freeform_goal_reached`
  guards with `if height_tol_m > 0.0` (0 = the test is OFF); `goal_region.in_goal` applies
  `|height_err| <= height_tol_m` unconditionally (0 = EXACT equality, never satisfiable). A
  spec of 0 scored an arm 0.0000 over 8400 episodes AND zeroed the reward's success term for
  five whole training iterations. `TODO(height-tol-conventions)`.
* **IsaacLab type-checks hydra overrides**: `height_tol_m=0` (an int) is refused where `0.0` is
  accepted, and it fails at hydra composition before the GPU is touched.
* **`record_rollouts.py` has its own `--history_from`**, applied AFTER hydra composition, so
  `env.commands.traj_ref.history_from=` is silently discarded. Pass the FLAG.
* **A comment inside a `\`-continued shell command ENDS the command**, making every argument
  after it a separate command. `bash -n` accepts it and these scripts have no `set -e`, so the
  flag simply vanishes and the callee uses its default. `test_shell_continuations.py` scans for
  it.
* **`robot.data`'s buffers are WARP arrays in a termination.** `eval.accumulate` reads them
  through a shim that converts on every access, so copying its accessor NAMES without the shim
  hands `tracking_error` a `vec3f` and it dies at the first step of collection -- and the
  suite passed, because nothing executed the function.
* **`ArmRun` passes `allow_robot_asset_mismatch=True` unconditionally**, so the robot
  fingerprint guard warns and continues on every mjeval run. Anything measured there is a
  statement about a pair the fingerprint says is not matched.
* **mjeval's `in_goal` is a +-`GOAL_DEPTH/2` BAND**; the free-form `goal_reached` is a
  downrange HALF-PLANE. The band's near edge cuts arriving episodes short, which is what
  `stop_on_goal=False` exists for when RECORDING rather than scoring.
* **A declared spawn area REPLACES `pos_jitter_m`** for the whole tile, so an authored ground
  area sized to anything but `FREEFORM_SPAWN_JITTER_M` silently changes the spawn distribution.
  box2 paid this once at 4x the intended lateral spread.

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

**The brain is the C++ controller** (`g1_cpp.yml`, `g1_control_cpp`); the python one
(`g1.yml`) is DEPRECATED, warns at startup, and is reachable only via `--python`. It survives
for bisecting a regression and nothing else. **CLIP PLAYBACK IS NOW C++ TOO** (`g1_cpp_clip.yml`,
the default for `scripts/run_sim_clip.sh`; `--python` reaches the old one). Nothing new goes
there: the numeric half belongs in `transfer/deploy_cpp` (tested against the python oracle with
no ROS), the ROS half in `g1_control_cpp`.

**The clip reaches C++ as a PACK, never as HDF5.** `transfer/deploy/tools/dump_clip_pack.py`
writes the library into the `manifest.json` + raw-float64-blob format `deploy/fixture_io.py`
already writes and `g1_deploy::BlobPack` reads (promoted out of `test/fixture.h` when this
became its second consumer). The image ships no HDF5, and the library is `lzf`-compressed,
which the plain HDF5 C library cannot decode without h5py's filter plugin. The pack carries the
channel NAMES, so a pack built for another robot fails loudly instead of reading the right
columns of the wrong motion, and it is committed beside the `.h5`.
`clip_reference_parity_test.cc` holds the port to the python oracle at 1e-11 over the real pack.

**Where the reference comes from is chosen ONCE, at load** (`ReferenceSource` in
`policy_runtime.h`: `FmReferenceSource` or `ClipReferenceSource`), so the tick does not branch
on it. The clip supplies its own per-offset `command_window`; the generator holds the current
command, as the deployed joystick does. A >1-channel generator still needs camera FK, which
clip mode never builds because it loads no generator at all.

**Pinocchio is the robot's kinematics, in both languages, from one apt package**
(`ros-humble-pinocchio`, 4.0 in the image; the dev box has 3.9 from pip and both are pinned by
the same tests). `g1_deploy::CameraPoseFk` FKs the depth cameras every tick — a generator with
`extero_channels > 1` unprojects its own image and cannot do that without knowing where the
cameras are — and `deploy.fk` draws the RViz/viewer plan ribbon, which used to be MuJoCo on
`FlowMatchingConfig.fk_model_path`. The "FK'ing the ROS URDF would be a SECOND OPINION"
objection is measured, not argued: URDF vs MJCF is 4e-13 on the camera chain and 4.6e-7 m at
the wrists (`deploy/tests/test_fk_parity.py`, MuJoCo kept as a test-only oracle), so the
deploy code has ONE kinematics library rather than two. It is not an image-size win — obelisk's own sim is a
MuJoCo sim, so the image installs it regardless. The URDF is a controller PARAMETER
(`urdf_path`), not a constant.
- Two CMake traps, both of which fail pointing somewhere else: pinocchio 4.0's
  `pinocchio_parsers` reaches `coal::coal` and its own config does not `find_dependency` it, and
  an imported target is DIRECTORY-SCOPED — so `deploy_cpp` finds coal and promotes it
  (`IMPORTED_GLOBAL`), or the consumer's own `target_link_libraries` line is what errors.
- **A green build on the dev box says nothing about four files.** `find_path(onnxruntime)`
  fails there, so `ort_session` / `tracker_session` / `fm_session` / `policy_runtime` are NOT
  COMPILED and a syntax error in them survives (it did: a botched extraction ate the head of
  `GaussianNoise::Fill` and was committed). The configure now WARNS; compile in the container.

**When it falls over, take the generator out of the loop first.** `scripts/run_sim_clip.sh
--case stand` swaps in `g1_cpp_clip.yml`, which loads the tracker ALONE (no `fm.onnx` at all)
and drives it from a baked clip through the SAME C++ runtime the generator path uses — so a
case changes exactly one thing against a normal run: where the reference comes from. Cases,
easiest first: `stand` (reference never moves — if this falls the fault is this stack, not
any policy), `stand_walk_stand` (the SAME clip untruncated, so a failure isolates the walk
transition), `walk_backward`, `terrain`. Each is known to survive in MuJoCo through the same
`transfer/deploy` runtime, so a case that falls here and not there is a difference in the ROS
stack. `--python` reaches the deprecated controller for a bisect; the two implementations are
held to 1e-11 (`deploy_cpp/test/clip_reference_parity_test.cc`), so a BEHAVIOURAL difference
between them is not in the reference construction.

**Two artifacts, neither committed** (both come from a clip library the robot does not
carry): `scripts/rsl_rl/make_debug_clips.py` writes the `.h5` the python controller reads,
then `transfer/deploy/tools/dump_clip_pack.py` writes the binary PACK the C++ one reads. The
pack exists because the image ships no HDF5 and the library is lzf-compressed, which the
plain HDF5 C library cannot decode. Build both before a clip run on a fresh machine.

**The cadence is not a knob.** Under `window_anchor="plan"` the clip's ego re-anchor and the
reference window's anchor latch must fire on the SAME step, or up to `K-1` steps of the
robot's own egomotion leak into `base_pose` — a window matching neither training nor an
odometry-free serve, and exactly as wide as a correct one. Both configs DERIVE it from the
tracker's `fm_update_every` (`0` = derive) and both runtimes refuse a mismatch. That
cancellation is what makes C++ clip playback odometry-free: measured 8.3e-17 m between a
walking and a stalled robot, against 0.132 m at a mismatched cadence.

- **No PD warm-up, and no way to add one.** Whatever feeds the tracker drives it from the
  first tick, matching `gen_tracker_runner.py`. The warm-up paths are GONE — the stand hold,
  its stiffened gains, the start-button gate, `warm_start()` and `BatchedHistory.fill()` were
  all deleted, so nothing can fabricate a generator history any more. Every one of them put
  the robot in a state no training episode started from and fed the generator history rows of
  ZEROS: `_make_snapshot` reads the runtime's own previously-served node, which does not exist
  yet, which flipped `null_mask` off and denied the model the learned null-history embedding
  meant to cover exactly those steps. What survives is `advance_dormant_step()`, which moves
  the cadence counter and touches nothing else — the clip path needs it so the first re-plan
  after a clip→FM handoff lands on the right phase. Pinned by
  `transfer/deploy/tests/test_history_null.py` on both sides of the port.
- **Auto-arm is SIM ONLY and structurally so**: it is a parameter of `fsm_state_echo`, a
  node that exists only because MuJoCo has no execution FSM and is launched only by
  `run_sim_world.sh`. The controller has no auto-arm and must not grow one
  (`tests/test_auto_arm_is_sim_only.py`).

## The generator's ONNX graph runs entirely on the GPU (and is captured)

The graph is ~3000 kernels for one 62-node plan at **batch 1**, so onnxruntime is
LAUNCH-bound there: wall time tracks the kernel COUNT, not the FLOPs. Two changes follow
from that, and the first is a precondition for the second.

**1. The batch axis is FIXED at 1 in the export** (`export_fm_onnx._export_graph`;
`--dynamic_batch` restores the old symbolic axis for throughput experiments). With a symbolic
batch, every head reshape in the traced attention blocks computes its shape at RUN time, so
the graph carried a `Shape -> Gather -> Unsqueeze -> Concat` chain of int64 scalars per block.
onnxruntime keeps shape arithmetic on the HOST, so **2687 of 6606 kernel nodes ran on the
CPU** with 71 `MemcpyFromHost` nodes between them — the "Memcpy nodes are added" warning at
load. Fixing the axis lets constant folding delete all of it, which also lets the fusion pass
match patterns the shape ops were splitting (`SkipLayerNormalization`, `QuickGelu`, a fused
QKV `Split`). Result: **0 host nodes, 3052 kernel nodes.** `tracker.onnx` was never part of
this — 17 nodes, already all-CUDA, 0.18 ms.
- The GPU-free way to check it is ORT's OWN optimizer, not the file: the file keeps its
  `Shape` nodes either way and folding them is the work being measured. 0 surviving `Shape`
  against 358 lines up exactly with 3052 all-CUDA against 6606-with-2687-on-host
  (`test_fm_onnx_static_batch`).
- Numerics: ~1e-6 from the old graph (torch folds a few more constants once shapes are known),
  three orders below the ~2.6e-3 CPU-vs-CUDA fp32 spread the graph already had. Freezing an
  EXISTING graph instead (`onnxruntime.tools.make_dynamic_shape_fixed --dim_param V
  --dim_value 1`) is bit-identical, if a forensic comparison is ever needed.

**2. CUDA Graph capture, which the host nodes had been blocking.** onnxruntime refuses capture
outright while any node is on the host ("all compute graph nodes have not been partitioned to
the CUDAExecutionProvider") — it fails to CREATE the session, so it cannot half-apply. On:
`fm_cuda_graph: true` in `g1_cpp.yml`, `--fm_cuda_graph` on `gen_tracker_runner.py`,
`load_fm_context(cuda_graph=True)`. Generator only; the tracker has nothing to win.

**Every wrong way of feeding a captured graph fails SILENTLY**, and that is why there are
tests either side of the port (`deploy/tests/test_fm_cuda_graph.py`,
`deploy_cpp/test/ort_sessions_test.cc`). A capture holds raw DEVICE POINTERS, so:
- a plain `sess.run()` on a capture-enabled session returns an **empty output list** once the
  capture is recorded — an `IndexError` far from the cause, or no plan at all;
- binding host memory, or rebinding fresh host buffers, returns the **first** answer forever,
  bit-identical, with no error;
- even mutating one bound host buffer IN PLACE is not picked up, because the host-to-device
  copy is itself inside the capture.
That last shape is the dangerous one: "the plan stopped responding to the cameras" is
indistinguishable on a walking robot from conditioning that has not moved. So the inputs live
in device buffers written in place — `update_inplace` in python, `Ort::Env::CopyTensors` in
C++ — and the tests assert the replay tracks its input, not merely that it is finite.
- **`CopyTensors` resolves its copy from the providers registered on the `Ort::Env`, which is a
  DIFFERENT registry from the one `SessionOptions` appends to.** A process can hold a perfectly
  good CUDA session and still get "Data transfer implementation between source and destination
  device was not found" on the first copy. `EnsureCudaDataTransfer()` registers the provider
  library on the Env once, by bare soname (it sits beside `libonnxruntime.so`, so an absolute
  `/opt` path would just be a second statement of where ORT is installed). Needs ORT >= 1.23;
  the images ship 1.24.4.
- Linking CUDA to do the copy ourselves was the alternative and was rejected: the images carry
  `libcudart.so.13` but no CUDA headers, so it would mean growing them.
- The C++ warm-up must go through the binding under capture (a plain warm-up would both
  mis-warm the session and leave the capture cost on the first live tick), and profiling is
  skipped there entirely — placement is established by construction.

**Measured, per re-plan** (dynamic -> fixed batch -> fixed + capture):
`11.6 -> 9.8 -> 8.3 ms` on an RTX 4070, `22.4 -> 19.3 -> 14-15 ms` on the Jetson Thor. Quote
Thor numbers as a RANGE: repeated passes on one graph vary ~8% there (clock/thermal) where the
desktop is stable to ~1%. **The Thor baseline sat ABOVE the 20 ms control period**, so this is
aimed at the chronically-late plan the two-box bring-up traced its falls to, not just at
throughput.

**Two things this surfaced, both FIXED 2026-08-24:**
- `G1_DEPLOY_POLICY_DIR` in `deploy_cpp/CMakeLists.txt` was missing the `policies/` element the
  controller loads under, so the C++ session tests SILENTLY SKIPPED on every dev box.
  `test_policy_cache_layout.py` pins that prefix across the language boundary and could not see
  this default. It now points at the live `plan` run.
- With the path fixed, `policy_runtime_test` failed for real: it fed no camera poses to a
  5-channel generator, which the runtime refuses BY DESIGN. `MakeReadings` now supplies each
  camera's nominal mount pose, and the test runs against the live pair (4 passed, 1 skipped on
  the CPU provider, where deadlines are not assertable).
- The capture path is verified on the Thor through the PYTHON runtime on the host (3/3 tests,
  19.33 -> 15.45 ms). The C++ half compiles clean on aarch64 and passes 12/12 on the CPU there,
  but its GPU behaviour in the Thor CONTAINER is unverified: `docker run --runtime nvidia`
  gives that container no working CUDA (`GPU=-1`, "driver version is insufficient"), and the
  untouched `ort_smoke` fails identically, so it is the launch, not the change. Running the
  brain for real is what closes it.

## Running the BRAIN on another machine (the Jetson Thor), over ssh

The split needs no new launch path: `run_sim_low.sh` is the brain alone and `run_sim_world.sh`
the world alone, so the brain runs on the Thor and the MuJoCo world on the dev desktop. Both
halves must agree on `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` (obk sets them) and each box
needs its OWN `docker/cyclonedds.xml` with the addresses swapped. That file is GITIGNORED, so
the link is described here rather than in it: `192.168.50.10` desktop (world),
`192.168.50.20` Thor (brain), `zolkin@`, direct cable on a private subnet chosen to collide
with nothing else on the desktop. `scripts/net_check.sh` on BOTH boxes while both halves are
up is the preflight, and it names the four ways discovery fails.

Host prerequisites, both boxes, and neither is optional:
- `net.core.rmem_max=16777216`. Cyclone treats a smaller kernel as FATAL and every ROS node
  dies at startup with `rmw_create_node: failed to create domain`.
- `<MaxAutoParticipantIndex>32</MaxAutoParticipantIndex>` in the cyclone config. Unicast peers
  require `ParticipantIndex auto`, which caps participants at 9 by default, and the world's
  bringup exceeds that. Cyclone's own default is `none`, which has no cap but needs multicast.

Then, in order, and the ORDER IS LOAD-BEARING (the world has no `user_pose` holding the robot
up, so a world-first start hands the policy an already-collapsed robot):

```bash
ssh -t zolkin@192.168.50.20 'cd ~/sprout/transfer/obelisk && scripts/run_sim_low.sh'  # brain
scripts/run_sim_world.sh --scene g1_perceptive_tile0_scene                            # world, here
scripts/fetch_brain_bags.sh          # bags are recorded ON THE BRAIN's box
```

**Traps, each of which cost a debugging cycle:**
- **`stop.sh` is LOCAL ONLY** -- there is no ssh in it. Tearing down here leaves the remote
  brain running and still recording. Stop both.
- **`run_sim_low.sh` RE-ATTACHES** when its tmux session exists, so a config change is ignored
  until `OBK_RESTART=1`.
- **The Thor reaches GitHub only through a running ssh AGENT.** The key has a passphrase, so
  whether `git push` works is a property of the CONTEXT, not the box: from a desktop session
  the gnome-keyring agent (`SSH_AUTH_SOCK=/run/user/1001/keyring/ssh`) holds it unlocked and a
  push just works, while the same command with the agent stripped fails with "make sure you
  have the correct access rights" -- which is what a cron job, a detached script, or the obk
  container (no agent socket mounted, only `~/.ssh`) all get. That last one is why the ZED
  depth link needs its own passphrase-free key rather than this one.
  Where there is no agent, move commits with a bundle over the private link, and FETCH BEFORE
  reverting anything local: `git bundle create /tmp/x.bundle <their-head>..<branch>`, scp, then
  `git fetch /tmp/x.bundle <branch> && git merge --ff-only FETCH_HEAD`.
- **`ssh -tt` needs a LOCAL tty** to allocate a pty from, so it fails when backgrounded inside
  a script. Run `script -qec '<cmd>' /dev/null` on the far side instead.
- **There is no tmux on the Thor HOST** (only inside the container).
- **`timeout` cannot exec a shell function.** `timeout N my_ssh_fn ...` fails instantly and
  silently; put the timeout on the far side of the ssh.
- **`pgrep -f <pattern>` matches its own command line**, and over ssh that line arrives
  containing the pattern. Bracket it (`'[r]os2 bag record'`) and keep the literal out of any
  echo in the same command -- this produced four separate false conclusions in one session.
- **Never copy a bag from a live run**: it comes back torn at message ~0. Stop first.
- **Remote bags go to `log/bags_thor/`**, never `log/bags/`. Bags are named by timestamp only,
  so mixing two machines in one directory makes their origin unrecoverable.
- **The clocks are not synchronised** on this desktop/Thor pair (chrony here,
  systemd-timesyncd there, ~8 ms apart). Cross-box `depth_capture_age` is offset by that and
  can read SMALLER than `depth_age`, which is impossible. Rate matches to 1 ppm, so only the
  absolute value is affected. Nothing in the control path uses a remote stamp; the watchdog
  deliberately uses receipt time. On the ROBOT LAN this is fixed rather than tolerated -- the
  brain serves time there (see the ZED box section) -- and the same
  `setup_clock_sync.sh --client` would make this desktop follow it if a split sim run ever
  needs comparable stamps.

## The ZED box: the depth cameras arrive over an ssh link, and one clock

On hardware the cameras are NOT on the brain. `run_hardware_low.sh`'s depth pane ssh's to the
ZED box (`zolkin@192.168.123.221`, `OBK_JETSON_HOST` in `docker/.env`) and runs `run_node.sh
pair` in the container there, so the link is part of the control path. Robot LAN, all three
boxes: robot `.161`, brain `.220`, ZED `.221`. Two one-time setups, both idempotent:
`scripts/setup_zed_link.sh` (ssh) and `scripts/setup_clock_sync.sh --master` here plus
`--client` there (time).

**Every failure of this link presents as a depth pane that prints nothing, which on a robot
reads as a CAMERA fault.** So `obk_jetson_preflight` runs before any pane is built and names
the cause instead: no route, no pubkey, changed host key, or no `zed_depth` container.
`run_hardware_low.sh` refuses to start on it (every policy here is a depth policy, `--clip`
included -- only the generator is absent there); `--no-depth` is the deliberate way past.

- **The key is NOT `~/.ssh/id_ed25519`**, which has a passphrase on both these boxes. A tmux
  pane cannot answer a prompt, so it would hang forever showing nothing. `setup_zed_link.sh`
  makes `~/.ssh/id_obk_zed` with no passphrase plus an `~/.ssh/config` block with
  `IdentitiesOnly yes` (without which ssh still tries the default identity first). `~/.ssh` is
  bind-mounted, so the key made on the host IS the key the pane uses.
- **`docker/.env` OVERRIDES the fallback in `obk_launch.sh`**, because compose puts it in the
  container's environment -- and it named the onboard minipc (`unitree@192.168.123.164`) long
  after the cameras moved, which answers ssh and would have been a wrong-box success. Worse, a
  RUNNING container keeps the value it was CREATED with, so editing the file changes nothing
  until compose recreates it. `test_zed_link_target.py` pins the tracked pair; `--check` prints
  the container's own value beside the file's.
- **The BRAIN is the time master**, not the ZED box, even though the ZED box is the one with
  internet: `depth_capture_age` is a receipt time made here minus a capture stamp made there,
  the bags are here, and every other stamp in them is this box's -- so one client's reported
  offset replaces a difference between two independently steered clocks. The ZED box's UTC
  comes over WIFI and the robot must work with no AP, which `local stratum 10 orphan` here
  covers. The cost, stated because the other direction was free: `systemd-timesyncd` is an
  SNTP client and cannot serve, so chrony replaces it here.
- **Live since 2026-08-22**: the ZED box tracks `192.168.123.220` (stratum 4 off the brain's
  3) at tens of microseconds. `trust` on the client means the internet pool there is excluded
  for disagreeing with the brain (`^?` in `chronyc sources`) -- deliberate, since the two
  boxes agreeing with each other matters more than either agreeing with UTC, and the pool is
  still polled so it takes over if the brain goes away.
- **Clock skew is a DIAGNOSTIC problem, not a safety one.** The depth watchdog times frames by
  local receipt on purpose (`g1_controller.cc`), so a skewed camera clock cannot stale-latch a
  healthy camera. It corrupts `depth_capture_age` -- the only number saying how old a frame was
  when the policy used it -- and every cross-box comparison in a bag.
- `obk_mark_jetson` had never been CALLED, so `stop.sh --zed` skipped the remote teardown
  silently and left the pair holding the cameras. `run_hardware_low.sh` sets it now.
- **`chronyc tracking` on the master is NOT what clients get.** While the `local` reference is
  active -- from startup until a real source's root distance falls under the 1 s `distance`
  threshold, and again if every upstream is lost -- tracking reports `7F7F0101`
  (127.127.1.1) / stratum 10 / zero root delay, so the master reads as unsynchronised for the
  first couple of minutes while it is already serving the real upstream. Measured here:
  tracking said stratum 10 while a query to `192.168.123.220` was answered stratum 3,
  ref 51.81.226.229, 108 ms root delay. `setup_clock_sync.sh` therefore probes the server over
  the wire (`served`), which also proves `allow` is scoped -- the same query to 127.0.0.1 must
  time out.

## The first REAL hardware run: three config defects, all in the robot driver

`run_hardware_low.sh` came up end to end on the robot on 2026-08-22 (brain + driver + ZED
pair + recorder, unarmed). Everything that blocked it was in the DRIVER path -- the sim path
had been exercised for months and the hardware one never had -- and each failure named
something other than its cause:

- **`docker/cyclonedds.xml` was still pinned to `192.168.50.20`**, the desktop split-sim
  cable, which this box no longer has. Cyclone treats an absent pinned interface as FATAL, so
  EVERY ros node died at `rmw_create_node: failed to create domain` with the real reason one
  line earlier ("does not match an available interface"). The file is gitignored per box, so
  nothing in the repo could catch it. It now lists BOTH roles' addresses with
  `presence_required="false"`, so the same file works plugged into the robot, the desktop, or
  both, and the absent one is a warning.
- **`AllowMulticast` must be ON for the robot LAN.** It was false (correct for the desktop
  cable, where unicast peers are safer). The ZED box cannot be reached that way: its config
  uses multicast and lists no peers, so its participants keep Cyclone's DEFAULT ephemeral
  ports -- and a portless `<Peer>` is resolved by probing participant INDEXES, which an
  unindexed participant never answers. Brain and cameras would simply never see each other,
  both looking healthy alone. The peers stay for the desktop role.
- **`target_ip` is THIS box's address, not the robot's.** obelisk uses it only to pick the
  local interface (`GetInterfaceName(target_ip, ...)`) and defaults to `192.168.123.222`,
  Unitree's user-PC convention. The brain is `.220`, so the node aborted with "No interface
  found with IP: 192.168.123.222" -- which reads as a robot connection fault. Set by ADDRESS
  rather than `network_interface_name` on purpose: an interface name changes when the cable
  moves between an onboard NIC and a USB adapter.
- **`g1_robot_hw.yml` had no `timers:` block**, so `timer_ctrl` -- the passive-hold timer
  `G1Interface` registers in `on_configure`, which drives the motors while the FSM sits in
  USER_POSE or DAMPING -- had no period, and the configure transition threw "Timer entry is
  missing required field 'timer_period_sec'". A missing entry only WARNS; it is a registered
  timer with no entry that throws.

Measured, unarmed, with the real cameras (tick 50.0 Hz, depth 47.2 / 47.7 Hz):
`depth_age` 1.7 / 0.6 ms (the controller's own contribution) and `depth_capture_age`
53.7 / 36.9 ms (whole pipeline from the ZED's own capture stamp). Both capture ages exceed
both receipt ages, which is TASKS 2a's done-when for the clock sync -- the number that used to
be able to read impossibly small. `fm_plan` and `ctrl` are silent while `armed: False`.

**Do not measure a stamp age with `ros2 topic echo --once` + `date`.** That method carries
~250 ms of its own process startup: a topic stamped on THIS box read +249 ms through it while
the depth topic read +273 ms. Only the difference means anything. The controller's own
`depth_capture_age` is the number, and it is in the bag.

## THE PAIR DROVE THE ROBOT (2026-08-25) — first armed hardware run, and it worked

The whole stack ran end to end on the real G1, on the first attempt: ZED pair over the ssh
link, the perceptive generator and the tracker on the brain, the C++ controller commanding the
motors. Everything above this line is bring-up; this is the first time any of it was ARMED.

What was actually tested, and nothing beyond it:
- **Flat ground: standing, and walking forwards, backwards and turning.** All fine.
- **A box jump-up: it got onto the box but stumbled at the top.**
- **Stairs: NOT tested.** Neither up nor down.

So the claim this supports is "the deployed pair walks on flat ground and can climb a box",
and specifically not that terrain works. The stumble at the top of the box is the first real
hardware finding and is unexplained.

It is worth being clear about what this does and does not vindicate. The stack had been run in
the obelisk sim for weeks; what had never been exercised was the ROBOT half -- the driver, the
motors, the real cameras, the real IMU -- and that is the part that worked first time. The
three defects in "The first REAL hardware run" above were all in that half and were all found
unarmed, which is why this run had none left to find.

The 3-DoF waist rode along untested-in-anger and was fine: this is also the first policy
deployed with `fixed_waist: false`, i.e. with `waist_roll` and `waist_pitch` actually driven.
The driver's own ordering (`joint_names_` = the 27-DoF set then the two extra waist joints)
matches `default_joint_names` exactly, and every command path is matched BY NAME, so nothing
here depended on the two orders agreeing -- but the `user_pose` array does, and it is indexed
by position in `default_joint_names` (`user_pose_[def_ind]` in `g1_interface.h`), which is why
waist roll and pitch are its LAST two entries.

The pair that did it: tracker `2026-08-20_16-36-10_mode11_decor_plan_v1`, generator
`2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx`.

## The obelisk SIM applies the MJCF's PD gains, and they were STALE (fixed 2026-08-24)

The tracker fell in the obelisk sim on motions MuJoCo sim2sim walked through cleanly, and the
cause was neither policy nor stack: **both hip pitches were 2.46x too soft there**, kp 40.18
against the 99.10 the policy exports. Same tracker, same clip, same world -- `walk_backward`
went from 3/5 falls to 0/5, and travel tightened from a 1.99-3.75 m spread to 2.48-2.50 m.

**Three consumers, and only ONE reads the MJCF's gains**, which is why this was invisible:

| consumer | gain source |
|---|---|
| `transfer/sim` sim2sim | the policy's, via `Robot.set_pd_gains_from_policy` -- **overwrites the MJCF** |
| the hardware driver | the policy's, via `msg.kp` / `msg.kd` on the wire |
| **the obelisk MuJoCo sim** | **the MJCF** |

Obelisk's sim is `void ApplyControl(const ControlMessageT& msg) { SetSharedData(msg.u_mujoco); }`
-- straight into `data.ctrl`. It never reads the published gains, so the MJCF alone decides how
stiff the robot is there, and both controllers publish `kp`/`kd` that only the robot consumes.

**Cause:** `make_g1_mjcf.py`'s `TEMPLATE_SECTIONS` copies `default` and `actuator` VERBATIM
from its template, `g1_29j.xml` -- the pre-mode_11 robot. mode_11 moved hip pitch from the
14.3 gear to the 22.5 (`STIFFNESS_7520_22`), and hip pitch is the only joint whose gear
changed, so it was the only mismatch. The generator already had the right armature in
`JOINT_ARMATURE` and was using it for `<joint armature=...>` while the ACTUATOR gains came
from the old robot.

**Fix:** `apply_actuator_gains` derives them from that same table via the same law the RL side
uses (`kp = J w^2`, `kd = 2 z J w`), so no number is duplicated and a future gear change moves
both. It refuses rather than guesses on an actuated joint with no armature, a `<position>`
with no default class, and a class shared by joints whose gains differ.
`test_actuator_table.py::test_mjcf_actuator_gains_match_the_cfg` pins the result -- it
resolves gains the way MuJoCo does (element attribute wins, else the named class) and fails
naming the joint, both numbers and the regeneration command.

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

# STATE: the training refresh and the deploy stack are MERGED (2026-08-23)

`worktree-robot-terrain-refresh` (the perceptive robot, terrain decoration, the `plan` anchor,
the camera-lag model) and `cpp-deploy-port` (the C++ deploy runtime, the obelisk C++ controller,
the two-box split, the ZED bring-up) are one tree. 12 files were touched by both; the four that
conflicted were `gen_tracker.py`, `gen_tracker_metrics.py`, `robot.py` and `gen_tracker_runner.py`.

Two resolutions worth knowing, because both branches had independently invented the same thing:
- `robot.py` -- terrain-relative fall detection plus a tilt cone, twice, with different APIs. The
  merged form keeps the ctor/CLI-driven one and exposes rtr's string verdict as
  `failure_reason()`, with `failed()` as its predicate half. The two datums were measured equal on
  all five scenes first (0.0 / 0.0 / −0.130 / −0.450 / −0.907), so no measurement moved.
- `gen_tracker_metrics.py` -- the same "the `fell` column is written before the fall check, so
  every fall read as a survival" bug, fixed twice. Both keys survive: `verdict_source` says which
  answer was used, `failure_reason` says what the runner saw.

**The warm-up paths stay deleted.** rtr still had them and threaded `fm_lag` through them, but
cpp had removed `GenTrackerPolicy.warm_start` / `.observe` entirely, so keeping the hold would
`AttributeError` on the first non-clip run.

## The `plan` teacher is exported and DEPLOYED (was: blocked on it)

`logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1` has its `exported/` --
`tracker.onnx`, `policy_parameters.yaml`, and `2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx`
-- and that is the pair that drove the robot. All three follow-ons are done:

1. `config/scan/depth_{near,far}.yml` are regenerated from that run and carry the far camera at
   15 deg (`rot: [0.99144, 0, 0.13053, 0]`), not the 35 deg placeholder.
2. `low_level.yml`'s `hf_policy_folder` names it.
3. `dump_metadata_fixtures.py` then `dump_obs_fixtures.py` re-run against it. The metadata
   fixture now agrees with the deployed `tracker.onnx` on all 530 keys, where before it
   differed on 89 and was missing 93 (the whole `robot_asset` block).

**The two halves had come apart very differently, and only one of them mattered.** Every obs
`.bin` regenerated BYTE-IDENTICAL -- the observation contract (term names, widths, history
lengths, clips, scales) had not moved between the two artifacts at all, so the assembled
vector was the same and `obs_assembler_test` had been testing exactly the right thing all
along. Only the manifest's recorded numpy version changed, 1.26.4 -> 2.4.6, which is itself a
small free result: the assembly is bit-stable across that major version.

**The METADATA half is where the rot was, and re-running it found a real bug.**
`policy_params_test.cc` asserted the PLACEHOLDER camera mounts -- near z 0.099426, far 0.38,
against the as-built 0.045 / 0.285 -- and had been passing for weeks because the fixture it
read described the same stale robot. That is precisely the failure the camera contract exists
to prevent: a wrong mount is a plausible cloud in the wrong place, invisible to every shape
check, sitting inside the one test meant to catch it. The intrinsics were unaffected (`fx`
12.586 / 21.422 both matched); only the mount pose had moved.

**Both stale-fixture and stale-artifact paths had the same root cause**, and it is worth
recognising a third time: `_DEFAULT_RUN_DIR` in `dump_metadata_fixtures.py` was missing the
`policies/` element AND naming a pre-refresh run -- the identical defect to
`G1_DEPLOY_POLICY_DIR` in `deploy_cpp/CMakeLists.txt`, which silently skipped two whole test
binaries. A default path to an artifact is a thing that rots quietly.

## Still open, and deliberately not done here

- **`transfer/obelisk/g1_model/mujoco/`** is a FOURTH, unreferenced copy of the robot
  (`g1_27dof.xml` + 4 scenes). Nothing in the launch path reads it. It only adds ways to be wrong.
- The `plan` arm has no torch-vs-numpy parity test at the `_MockCmd` seam
  (`test_gen_tracker_obs_parity.py`'s `window_anchor()` still returns the measured root, so it
  covers `robot_now` only). The CLIP half of that gap is closed --
  `deploy/tests/test_clip_window_anchor.py` and `deploy_cpp/test/clip_reference_parity_test.cc`
  pin the `plan` arm on both sides -- but the generator's Isaac-vs-numpy seam is still
  `robot_now`-only.
- `refresh_fk_channels.py` left the 24 world-frame EE VELOCITY channels on the old model (not
  recoverable -- the source `.npz` are gone), so post-refresh the two wrist world-velocity blocks
  are consistent with the old robot while their positions are consistent with the new one.
  Libraries record this in an `fk_refresh` attr.
- `d66420e7`'s local scene tuning (jump-up boxes near spawn, `depth_near.yml viz` re-enabled) was
  lost in an earlier merge and has not been restored.

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