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
  `extero_norm_mode` decides how the normalizer is pooled; see "The DEFAULTS are the depth
  training config" below (`per_channel` is the default, `per_element` is gone).
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

**The brain is the C++ controller** (`g1_cpp.yml`, `g1_control_cpp`); the python one
(`g1.yml`) is DEPRECATED, warns at startup, and is reachable only via `--python`. It survives
for bisecting a regression and for CLIP PLAYBACK, which the C++ runtime cannot do at all (no
HDF5 clip reader — it refuses to load without an `fm.onnx` rather than pretending;
`TODO(cpp-clip)`). It also REFUSES a >1-channel generator outright, because that needs camera
FK it does not have. Nothing new goes there: the numeric half belongs in `transfer/deploy_cpp`
(tested against the python oracle with no ROS), the ROS half in `g1_control_cpp`.

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
--case stand` swaps in `g1_debug_clip.yml`, which loads the tracker ALONE (no `fm.onnx` at
all) and drives it from a baked clip. Cases, easiest first: `stand` (reference never moves —
if this falls the fault is this stack, not any policy), `stand_walk_stand` (the SAME clip
untruncated, so a failure isolates the walk transition), `terrain`. Each is known to survive
in MuJoCo through the same `transfer/deploy` runtime, so a case that falls here and not
there is a difference in the ROS stack. The clip library is built by
`scripts/rsl_rl/make_debug_clips.py` out of the library the teacher trained on.

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
scripts/run_sim_world.sh --scene g1_29j_tile1_scene                                   # world, here
scripts/fetch_brain_bags.sh          # bags are recorded ON THE BRAIN's box
```

**Traps, each of which cost a debugging cycle:**
- **`stop.sh` is LOCAL ONLY** -- there is no ssh in it. Tearing down here leaves the remote
  brain running and still recording. Stop both.
- **`run_sim_low.sh` RE-ATTACHES** when its tmux session exists, so a config change is ignored
  until `OBK_RESTART=1`.
- **The Thor cannot reach GitHub** (the key needs a passphrase, so non-interactive ssh fails).
  Move commits with a bundle over the private link, and FETCH BEFORE reverting anything local:
  `git bundle create /tmp/x.bundle <their-head>..<branch>`, scp, then
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
- **The clocks are not synchronised** (chrony here, systemd-timesyncd there, ~8 ms apart).
  Cross-box `depth_capture_age` is offset by that and can read SMALLER than `depth_age`, which
  is impossible. Rate matches to 1 ppm, so only the absolute value is affected. Nothing in the
  control path uses a remote stamp; the watchdog deliberately uses receipt time.

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