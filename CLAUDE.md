# Code Overview

Reinforcement learning for humanoid robots (the Unitree G1): train, then deploy in simulation and
on hardware. IsaacLab at the core, so the suggested file structure.

## Where the rest of this lives

This file holds what is true across the whole stack. Everything else sits beside the code it
governs and loads when you work there:

| file | covers |
|---|---|
| `source/motion_gen/motion_gen/terrain/CLAUDE.md` | the terrain manifest, decoration, clutter, the free-form course contract, box2 |
| `scripts/rsl_rl/eval/CLAUDE.md` | the eval harness, best-of-K, the failure critic, seed/noise-draw traps |
| `scripts/flow_matching/CLAUDE.md` | the generator fine-tune loop, its reward, its success rule |
| `transfer/sim/CLAUDE.md` | MuJoCo sim2sim and mjeval |
| `transfer/deploy/CLAUDE.md` | the ONNX runtime, the exported graph, artifact naming, spec agreement |
| `transfer/obelisk/CLAUDE.md` | the ROS2 deploy pipeline, the robot, the ZED box, logging |

`README.md` is the stage-by-stage runbook. `scripts/flow_matching/FINETUNE.md` is the fine-tune
runbook. `docs/claude_md_archive_2026-09-09.md` is the previous single-file version of all of this,
kept verbatim -- reach for it when you need the reasoning behind a rule, or a number someone quoted.
Also `docs/seed_headroom_report.md`, `docs/critic_advantage_eval_report.md`,
`docs/seed_variance_mujoco_report.md`.

# WHICH ARTIFACTS TO USE

**The generator to run and to fine-tune from is `2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff`**
(exported as `..._nojumpoff_fm.onnx`), with `history_from="generator"`. The measured-history
counterpart is `2026-09-03_01-42-04_2026-09-03_measured_n3`, served with
`history_from="measured"` -- the runtimes REFUSE a mismatched pair, and the two kinds of history
row are identical in width, so a mismatch merely steers wrong. Use these two and nothing else
until a newer one is trained and named here.

**The trap: `2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx` sits in the SAME `exported/`
directory and is NOT it.** That one is the DEPLOYED generator -- the pair that drove the robot,
which is why the hardware section below still names it -- and nothing was ever fine-tuned from
it. Reaching for it because it is the one this file quotes is a mistake that has been made.
Related: the three `lag48_aug` runs carry `skip_families=None`, i.e. they INCLUDE the backwards
jump-off, so `2026-08-23_18-25-31_plan_v1_lag48_aug_h64_27M` is not "the no-jump-off one"
either. **`nojumpoff` drops only the BACKWARDS dismount**; `family_jump_off_front`, the forward
one, is still in the training set, which is what the box work needs.

`resolve_fm_onnx()` RAISES and lists them when a directory holds several, which is the guard --
there is no "newest wins" default. Each fine-tune's `finetune.json` records its own parent under
`args.pretrained`, and that is the authority if any of this drifts.
`scripts/flow_matching/FINETUNE.md` carries the full provenance.

# Architecture

One chain, each stage feeding the next:
`motionbricks clip generation -> teacher RL -> flow-matching generator -> tracker -> sim2sim -> hardware`.

The **clip library** (`.h5`) is the spine: it carries both the reference motions and the terrain
they were generated on, so one file defines what the robot tracks AND what it walks on. Every task
takes it via `--clip` and there is deliberately **no default** (a stale library silently trains on
the wrong terrain).

## Layout

`source/motion_gen/` -- IsaacLab-free offline clip generation:
- `motionbricks/` -- the live generator (raw Motion Bricks npz -> clip ingest)
- `terrain/` -- sim-free terrain geometry, EMITTED by the generators and re-read by the RL side so
  nothing is recomputed or allowed to drift
- `ingest/` -- readers turning a reference-motion library on disk into tensors
- `clips/` -- clip `.h5` IO, the RL-runtime `MotionClipManager`, fingerprinting
- `motion_matching/` -- DEPRECATED, kept only for ablations (see its README)

`source/robot_rl/robot_rl/` -- the RL side:
- `tasks/manager_based/robot_rl/g1/` -- the env + agent cfgs. `g1_clf_tracking_base` is abstract
  (scene / actions / events / CLF rewards); `g1_teacher_env_cfg` adds the clip command + obs;
  `g1_teacher_terrain_env_cfg` supplies the ground; `g1_tracker_env_cfg` swaps the clip reference
  for the flow-matching one
- `tasks/manager_based/robot_rl/mdp/` -- rewards, observations, terminations, events, and
  `commands/traj_tracking/` (the clip + FM command terms, the CLF, the world-lift adapter)
- `tasks/manager_based/robot_rl/terrains/` -- the `ClipTerrainImporter` + block builders
- `flow_matching/` -- the FM model, dataset and viz

`scripts/rsl_rl/` -- entry points (`train.py`, `play.py`, `record_data.py`, the `mb_*` generation
stages, `compare_sim2sim.py`) plus diagnostics. `scripts/flow_matching/` trains and exports the FM.
Runs log to `logs/`.

`transfer/` -- deployment: `transfer/sim` (MuJoCo sim2sim), `transfer/deploy` (the shared torch-free
ONNX runtime), `transfer/deploy_cpp` (the C++ port, tested against the python oracle with no ROS),
`transfer/torch_math` (isaaclab-free torch quaternion math, the test-only oracle for the numpy
ports), `transfer/obelisk` (ROS2 hardware).

`scripts/rsl_rl/fm_debug.py` -- the FM-vs-clip debugger, and the tool to reach for when the tracker
follows clips but not the generator. It rolls the tracker over three frozen clips twice, differing
in ONE field (`commands.traj_ref.reference_source`), with both references evaluated at identical
robot states. `mdp.reference_window_to_world` is the exact inverse of the window observation, so
the viewer can draw what the policy SAW.

# The robot is `g1_perceptive`, and FOUR files describe it

`robot_assets/g1_perceptive/` is the trained robot: mode_11 motors, a 2.64 kg compute backpack
(Thor + ZED), passive rubber hands in place of the Dex3-1s, the torso collider on the torso, the
depth-camera mounts on their as-built CAD pose. Everything else is DERIVED from its URDF.

| file | derived by | read by |
|---|---|---|
| `robot_assets/g1_perceptive/g1_perceptive.urdf` | authored (`make_perceptive_urdf.py`) | the source of truth |
| `robot_assets/g1_perceptive/g1_perceptive/` (USD) | `urdf_to_usd_conversion.py` | IsaacLab |
| `transfer/sim/robots/g1/g1_perceptive.xml` | `make_g1_mjcf.py` | MuJoCo sim2sim + `motion_gen` FK |
| `transfer/obelisk/g1_model/urdf/g1_perceptive.urdf` | `make_ros_urdf.py` | the deploy camera FK + RViz |

**Joint count and order are UNCHANGED -- 29, identical -- and that is what makes a stale model
dangerous.** A hand-kept fourth copy once left `g1_deploy::CameraPoseFk` FK'ing the pre-perceptive
robot: every name resolved, and it placed both depth cameras exactly 10.0 mm high on every
unprojected xyz channel. A wrong camera pose is a plausible cloud in the wrong place, so no image
inspection would have found it. Hence:

- `robot_asset` (`motion_gen/robot_asset.py`) fingerprints urdf/mjcf/usd hashes + joint names +
  sorted body names + total mass. `export_parameters` stamps it into `tracker.onnx`,
  `export_fm_onnx` into `fm.onnx`, and BOTH runtimes refuse a pair that disagrees. A missing side is
  silent (a guard against a new mismatch, not a migration gate) but "absent" and "empty" are
  deliberately different -- an empty fingerprint would compare equal to anything.
- `terrain_fingerprint` is stamped and **deliberately compared nowhere on the deploy path**; see
  `TrackerSession.get_terrain_fingerprint` for why both obvious comparisons are wrong.
- **Pre-mode_11 checkpoints load and are numerically meaningless**: `G1_ACTION_SCALE` is
  `0.125 * effort / stiffness`, and hip pitch moved from the 14.3 to the 22.5 gear, so its action
  scale went 0.2738 -> 0.1753. Same width, different meaning. `test_actuator_table.py` pins the table.
- `body_names` is no longer a prefix of the pre-refresh list (39 bodies; Dex3 links out, hands /
  camera mounts / backpack in), so anything indexing it positionally shifts silently.
- `test_ros_urdf_is_generated.py` fails if the generated ROS URDF drifts from its source;
  `test_fk_parity.py` imports both models from where they are DEFINED rather than restating paths.
- `refresh_fk_channels.py` left the 24 world-frame EE VELOCITY channels on the old model (the source
  `.npz` are gone), so post-refresh those two blocks are consistent with the OLD robot while their
  positions are consistent with the new one. Libraries record this in an `fk_refresh` attr.

# The reference window has TWO conventions, and deploy serves only one

`window_anchor` (`robot_rl/flow_matching/window.py`) decides what the `base_pose` block's POSITION
is measured from. The two arms are **identical in width and in every other block**, so a tracker
trained under one loads clean against the other and merely steers wrong -- which is why the
convention travels in the artifact and is checked rather than inferred.

- `robot_now` -- `p_ref - p_actual(t)`. Re-anchors onto the measured base every control step, which
  needs a base POSITION, which on a robot means odometry.
- `plan` -- `p_ref - p_plan`, from the root at the last re-plan. The generator already predicts in
  that frame, so its output IS the observation: served as made, no base position read.

**The C++ deploy runtime serves `plan` ONLY and refuses `robot_now` by design**
(`AnchorFromConvention`). That is what makes the deploy path odometry-free: under `plan`, `root_pos`
reaches only the anchor latch, then `TickDiagnostics` and the RViz ribbon. The FAST-LIO plumbing
stays in the repo, unused. The PYTHON runtime serves both -- it is the MuJoCo driver and the parity
oracle, and MuJoCo has a true base pose.

**One `fm.onnx` serves either arm.** `_run_fm` latches the anchor on both and the history stash is
the raw served node on both; only the serve differs.

Gap: the generator's Isaac-vs-numpy parity test at the `_MockCmd` seam still covers `robot_now` only
(`test_gen_tracker_obs_parity.py`). The CLIP half is closed on both sides
(`deploy/tests/test_clip_window_anchor.py`, `deploy_cpp/test/clip_reference_parity_test.cc`).

# Depth cameras

Two self-scanning depth cameras on the terrain teacher, as an alternative exteroception to the
height scan. `g1/g1_depth_env_cfg.py` + `g1/g1_self_scan.py` (camera cfgs and the generated
`MultiMeshRayCaster` target prim paths), `mdp/observations/depth.py`, `mdp/events/depth.py`,
`mdp/sensors/noise.py`. Tasks: `G1-teacher-clf-terrain-depth{,-image}` (+ each `-play`).
`scripts/rsl_rl/verify_depth_self_scan.py` verifies self-scanning at the sim level.

- **Every moving link must be its OWN raycast target** with `track_mesh_transforms=True`, aimed at
  the INNER same-named Xform: a target on the OUTER link Xform welds the whole downstream limb into
  one rigid body frozen at its spawn pose (knee flexion would be invisible).
- **The HANDS must stay real fixed-joint links** (asset `g1_21j_box_merged_29j_hands`). As bare
  `<visual>` elements on the wrist -- how the stock URDF ships them -- the importer gives them a
  non-identity mesh->body offset and `MultiMeshRayCaster` does not intersect meshes on that path:
  the cameras saw every limb EXCEPT the hands while the viewer looked perfect.
- Depth REPLACES the height scan in the POLICY obs (the hardware has cameras, not a heightmap); the
  CRITIC keeps the scan, since it is privileged and never deployed. The `-image` variant differs
  only in obs SHAPE; it is untested in training and `PPORunnerCfg` is an MLP.
- These extend the BODY-tracking terrain teacher, so they default to `tracking_mode="body"` and pair
  with a widened (`@root`) clip library. Override with `env.commands.traj_ref.tracking_mode=joint`.
  Costs ~19% more per iteration than the scan.
- Profile a LOADED policy, never an untrained one (it dies every step and doubles the apparent reset
  cost). Of a 129 ms env step at 1024 envs: `obs` 50.8 ms (~19 of it the two cameras),
  physics/render ~47, `command` 13.1, `events` 7.8, `reward` 7.6, `termination` 2.5.

## The policy reads NEAR only, and the far camera has ONE correct slot

**The POLICY reads the NEAR camera alone; the GENERATOR reads both.** The tracker follows an
already-placed reference, so terrain ahead is the generator's problem. `assert_depth_specs_agree`
therefore matches by NAME (`camera_key`, stripping the `fm_` prefix), not by list index -- pairing
by index would compare the policy's near camera against the generator's far one and report two
different cameras as a contract violation. A camera the tracker reads and the generator does not
stays an error.

That near-only change (`efe93b4e`) was taken ahead of a teacher retrain that **has not happened**,
so every tracker checkpoint on disk expects 3002 inputs where a near-only task builds 2222.
`enable_policy_far_camera` puts the term back -- and **the obvious spelling puts it in the wrong
slot.** Written as an annotated field (`depth_far: ObsTerm | None = None`) it lands BEFORE
`depth_near`, because `configclass` orders ANNOTATED fields ahead of un-annotated ones and
`depth_near` carries no annotation. The vector is then the right WIDTH with two 780-wide blocks
transposed: `load_state_dict` passes and the tracker reads the far camera as the near one. Over 64
episodes, transposed vs correct: `reached_goal` 0.000 -> 0.188, median `steps_survived` 143 -> 517.
**Nothing in the first row reads as a config bug; it reads as a policy that cannot do the task.**
`test_box2_task_obs_order.py` pins it against the deployed artifact's OWN `observation_terms`
block, plus a mechanism check on the declared field order that runs where `logs/` is absent. Note
`reference_window` / `command_window` appear in that block but are NOT cfg obs terms -- they are
`RefEncoderMLPModel` inputs, which is why the MLP takes 2490 raw + 512 encoded = 3002.

## The camera contract

`mdp.depth_camera_specs(env)` derives ONE contract per camera -- mount body + offset, pinhole
intrinsics, range, and the `(d - offset) * scale` normalization -- from the sensor cfgs and obs
terms. It is the single producer, so nothing off-sim restates a camera number: `tracker.onnx`
metadata `policy_parameters.depth_camera_spec`; `fm.onnx` metadata `fm_meta.extero_spec` (+
`extero_source`, `extero_sensor_count`, `extero_sensor_dim`); `GenTrackerPolicy.load()` requires
the two to AGREE on every camera they SHARE (`transfer/deploy/depth_obs.py`).

- **The generator's group is INSTALLED, never inferred.** `install_fm_depth_group(cfg, W, H)` puts
  every scene camera in its own `fm_depth` obs group. At the policy's own 30x26 the terms point at
  the EXISTING sensors; at any other grid it builds `fm_`-prefixed duplicates from the SAME
  `_CameraSpec` and EXTENDS the three DR events to `sensor_cfgs` lists so one draw moves both
  samplings (intrinsics transfer as a ratio, hence `operation="scale"` is required for a list).
- **A missing group is a FAULT, not an opt-out.** Without it `fm_depth_groups` falls back to `None`
  = all groups, which on this task is the policy's NEAR camera alone -- and every downstream
  artifact would then be self-consistent about a half-blind generator.
  `assert_fm_depth_group_covers_every_camera` is the guard; every task and script that runs or
  records the generator installs the group unconditionally.
- Who states the resolution: `record_data.py --fm_depth_resolution WxH` DECLARES it; `play.py` and
  `augment_registration.py` DERIVE it (from the checkpoint, and from the recording's own spec).
- MuJoCo casts BOTH grids (`mirror_extrinsics_from` shares the jittered mount). **Hardware is not
  wired for a split sampling** -- `export_sensor_configs.py` raises rather than emitting configs
  that cover only the tracker.
- `export_parameters.py` emits each obs term's `clip`, so `_assemble` saturates every term the way
  training did instead of consulting a hardcoded table (kept only as a legacy fallback).

## The generator's depth is DELAYED, in training and in both simulators

Real depth puts tens of ms between capture and the network (~42 ms median measured on the ZED pair,
48 modelled), so the generator's conditioning carries a lag on purpose.

- TRAINING: `depth_latency_probs`, a distribution over whole control steps sampled per window by
  `dataset._select_depth_lag`, riding `fm_meta` end to end.
- SIM: `FmDepthLag` in `gen_tracker_runner.py`, derived from the graph unless
  `--fm_depth_lag_steps` overrides. **Only the GENERATOR is delayed**; the tracker's own `depth` obs
  stays live, because it was trained on fresh frames.
- **The POSE travels with the image.** The 5-channel row's xyz planes are unprojected from the
  camera pose, so a lagged image unprojected with the CURRENT pose is displaced by the egomotion.
- HARDWARE adds none of this (its lag is physical); it matches the camera POSE to the frame's
  capture stamp (`g1_deploy::PoseHistory`, `depth_pose_match`). `depth_delay_relay` manufactures the
  missing latency in MuJoCo so that matching code is exercised.
- **`--fm_lead_ticks` and `--fm_depth_lag_steps` do NOT compose.** Both exist to line the sim runner
  up against the C++ runtime, which applies no synthetic lag. The lag is forced off for
  `--fm_lead_ticks` / `--dump_replay`, and asking for both is a hard exit.

## The depth row carries CHANNELS (`extero_channels`)

The FM sees no joint angles and no root orientation, so a waist turn is otherwise indistinguishable
from the terrain moving. `1` = depth alone (older checkpoints), `4` = `+ xyz` in the root's
yaw-aligned frame, `5` = `+ validity`. `extero_sensor_dim` COUNTS channels (it is the row width), so
everything comparing it against an image shape divides by `extero_channels` first. **The fold is
CHANNEL-MAJOR**; interleaving scrambles every kernel while the shapes stay right.

- **The frame is LEVEL** -- orientation `yaw_quat(root_quat_w)`, zero roll/pitch. Static terrain
  therefore does NOT move when the robot tilts: a pitch rewrites every pixel but moves the CAMERA by
  the same amount and the unprojection composes it back out. Deployment needs only FK to the camera
  plus the IMU's roll/pitch -- **never odometry**, and a drifting yaw cannot corrupt it. That is a
  claim about the COMPUTATION, not a yaw invariance: the frame turns with the robot.
- **Nominal extrinsics, never the DR draw**: a robot knows its calibration and nothing more, so the
  DR perturbs the IMAGE relative to the stated geometry (which is what calibration error IS).
- **No re-record.** `dataset._camera_pose_yaw_frame` rebuilds the pose offline from the recorded root
  quaternion + joint angles + the contract's mount via `motion_gen`'s `FKChain` (validated against
  the recording's limb frames to 0.67 mm). `mdp.depth_xyz_stack` is the live twin; they must agree.
- **Validity is found by RANGE, not sentinel equality** -- `DepthNoiser`'s 3x3 blur leaves a
  geometric miss NEAR the sentinel. 6.4% (near) / 8.9% (far) of pixels carry no reading. The height
  scan keeps equality, where it is exact.
- **A camera's position embedding is ANGULAR, token-level and PER SENSOR** (`pixel_angles_rad`): the
  two cameras are 100x70 and 70x42 deg, so one shared extent mislabels the far one's tokens by
  1.43x, silently. The metric ground grid keeps its coord CHANNELS.
- `extero_cnn_grid` larger than the trunk's native grid is refused (it used to upsample silently).

## The DEFAULTS are the depth training config

Every extero knob resolves per SOURCE, so one default set is correct for a camera run and a scan run
alike, and a checkpoint records the RESOLVED number. `extero_channels=None` -> 5 for depth / 1 for a
scan (`resolve_extero_channels`), joining `extero_encoder="auto"` (cnn / mlp) and `extero_clip=None`
(+-5.575 / +-1.0 m). Defaults: `depth`, `cnn`, 5 channels, `strides (1,2,2)` = **7x8 = 56
tokens/camera** (sequence 191), `contract` normalization.

- `EXTERO_CLIP["depth"]` is **+-5.575, measured**: at +-2.0 the band saturated at 4.75 m while the
  far camera sees 11.9 m. Recordings store depth UNCLIPPED, so widening recovers signal already on
  disk. Changing a per-source default RE-RESOLVES any checkpoint that stored `None`, which is why
  `train.py` prints the band and whether it was declared.
- **There is NO train-time exteroception noise.** `noise_depth` / `noise_height_scan` /
  `noise_extero` are gone. The sensor model belongs to the RL env, whose structured DR is already
  baked into the recording; a scalar uniform term on top over-noised the one signal the generator
  uses to place a reference. Pinned by `test_there_is_no_train_time_exteroception_noise`. The
  history/reference-window noise (`noise_scale`, `REFERENCE_WINDOW_NOISE`) is NOT this and stays --
  it models exposure bias, not a sensor.
- **`contract` normalization fits NOTHING**, so the same robot yields the same numbers in every
  recording. `d` -> identity (it already arrives declared-normalised out of the contract). `xyz` ->
  one declared scale (`extero_xyz_scale`, 3.0 m) shared by both cameras and all three axes, with a
  per-camera geometric OFFSET (the boresight's intersection with the nominal ground plane, rounded
  to a centimetre, ground z = -0.7869). `v` -> identity. It rides the existing `Normalizer`, so the
  checkpoint, the folded affine and the deploy path are untouched.
  **The offsets are DERIVED per recording and there is no default to quote**: the config field is
  `None` until the dataset resolves it from that recording's own camera contract. On the as-built
  cameras (`d514f0b1`, 2026-08-19) they come out `[[0.51, 0.03, -0.79], [4.25, 0.06, -0.79]]`,
  which is what `..._h64_27M_nojumpoff` carries. This file used to quote
  `[[0.55, 0, -0.79], [1.8, 0, -0.79]]` as if it were the shipped value; those are the PRE-CAD
  35-degree far camera, they predate the mount change by two days, and the only place they appear in
  code is a CLI-parsing test that passes them in explicitly. The 2.45 m gap between the two is the
  far camera going 35 -> 15 deg, which is exactly what `_XYZ_OFFSET_DRIFT_WARN_M` exists to shout
  about -- so a stale declared offset warns rather than silently mis-normalising.
- **`extero_xyz_envelope` is a safety net, not a band** -- derived from each camera's own cone and
  asserted never to bind. It comes from the clip band divided by the **corner cosine**, NOT from
  `max_dist`: a valid pixel carries a z-DEPTH, which becomes a ray length once divided by that
  cosine. Widening `extero_clip` widens this too.
- Known cost: the offsets centre the NOMINAL FLAT-GROUND view, not the terrain a recording sampled
  (on one drop-off tile the far camera's empirical centroid sat 1.9 scales out). The dataset PRINTS
  declared-vs-measured per camera at build; if the residual stays large the fix is a larger declared
  nominal range, not a fitted one.
- Other modes: `per_channel` (one mean/std per sensor per channel group), `per_sensor` (pools across
  channels -- fine for a 1-channel scan, wrong for a 5-channel row), `metric` (one declared affine,
  still right for a privileged metric scan). `per_element` is DELETED: a per-pixel affine is a
  spatially varying transform, exactly what a shared convolution assumes away.

## The height-scan contract, and a PRIVILEGED generator in sim2sim

`fm_meta.extero_spec` holds whichever sensor the checkpoint trained on, so it is a camera list for
one FM and a ray grid for another. `transfer/deploy/scan_obs.py` is the scan half:
`parse_extero_spec` picks the parser from the contract's own `extero_source`, and
`assert_scan_specs_agree` compares the fields the two artifacts SHARE, with a floor so a sparse
contract cannot agree by silence.

- A **depth tracker + scan generator** is a legitimate pair (the robot has cameras, the generator
  reads the ground). `_resolve_depth_cameras` only reconciles `extero_spec` against the tracker's
  cameras when the FM's source IS depth.
- The dense scan is `privileged`, and that gate is **sim-only**: `export_fm_onnx
  --allow_privileged_extero` (plumbed through `play.py`) and `gen_tracker_runner.py
  --allow_privileged_extero`. **Hardware still refuses** -- the flag defaults off and nothing on the
  robot path passes it. Anything measured this way is a statement about the model given perfect
  terrain.
- `torch.isclose` has no ONNX lowering, so the miss-sentinel channel in
  `flow_matching/extero_encoder.py` is written as `|img - sentinel| > atol + rtol*|sentinel|`.

## The FM's `attention_mode`

`prefix` (DEFAULT) has history+extero+cmd attend only to THEMSELVES and keeps the flow-time
embedding off them, so their representation is encoded ONCE per re-plan and reused by every Euler
step (`model.encode_prefix` / `decode_future`). `full` is what every pre-flag checkpoint trained
under. **The two modes share one parameter set**, so a mislabelled checkpoint runs silently under
the wrong attention -- `load_state_dict` cannot catch it. `from_dict` defaults a pre-flag checkpoint
to `full`, the mode is printed at load and is in `fm_meta`; that is the whole defence. Measured:
torch gets 1.34-1.39x at Isaac batch sizes, but the exported graph is ~0.82x at batch 1. `prefix` is
the cached decomposition, NOT a mask -- a 2-D `attn_mask` materializes `(B, nhead, S, S)` inside
`merge_masks`, gigabytes at Isaac env counts.

# NOTHING IN THIS STACK AVOIDS OBSTACLES, and that is deliberate

There is no obstacle term anywhere in the MDP -- no reward, no observation, no termination. The
tracker follows a reference; the generator predicts a reference from depth and a command. Neither
plans around anything. `decorate.py`'s `wall_collidable: bool = False` records the measured reason a
wall is visual-only: a collidable wall the reference's arm sweeps through blocks a wrist, fires
`ee_pose_error` at 0.3 m and ends the episode for a reason the policy cannot fix, and
`undesired_contacts` at -0.1 against `clf_reward` at 10 is far too weak to teach avoidance.

So "the robot walked at a wall" is partly CORRECT behaviour and is not a bug to fix here. What IS a
real fault is planning a JUMP or a CLIMB at one -- a classification failure, and
`scripts/rsl_rl/eval/terrain_class.py` is the instrument. Keep the two apart: obstacle avoidance is
an operator- or planner-level concern this stack does not have and should not grow by accident.

# The generator's history has THREE producers

`history_from` decides what fills the FM's `N` context rows. They are reference-window nodes,
identical in width whatever fills them, so a mislabelled run is invisible to every shape check --
which is why the arm is printed unconditionally by both runtimes and stamped into every manifest.

| arm | a row is | on a robot |
|---|---|---|
| `generator` (default) | the node this runtime SERVED that step -- its own output | yes, and it is what deploys |
| `clip` | the CLIP's node for that step (teacher forcing) | no -- there is no clip |
| `measured` | the ROBOT'S OWN achieved state that step | yes, in principle |

The first two both mean "the reference applied at that step"; only `measured` is a different kind of
quantity. The model is FITTED on reference rows, so the closed loop is distribution-consistent in
kind and merely drifting, while `measured` is real and off-distribution.

- **Two of the five node blocks fall back to the prediction** (`window.MEASURED_HISTORY_FALLBACK`,
  mirrored in `window_np`): `base_pose`'s POSITION half (no odometry) and `base_vel`'s LINEAR half.
  Named at HALF-block granularity because that is where the answer changes. The rest is measured --
  38 of the deployed layout's 44 dims.
- **Both stashes must describe the SAME step.** The row pushed at step `t` describes `t-1`, so the
  substitution pairs `_prev_measured_node` with `_prev_served_node` at PUSH time. A row is also
  all-or-nothing: before the first plan the whole row goes in as zeros under the null-history
  embedding.
- Wiring: `history_from="measured"` on the Isaac command (eval cell `fm_meas`, whose
  `reference_overrides` give it `ego_anchor_fraction=None` like `fm`, NOT `1.0` like `fm_tf`);
  `gen_tracker_runner.py --history_from measured` plus the mjeval `baseline_meas` arm. **There is no
  `clip` arm on the deploy side** and the ctor refuses one by name. `--history_from measured` is a
  hard exit with `--fm_lead_ticks` / `--dump_replay`.

**VERDICT: the history source is worth nothing.** Four independent measurements agree -- fed to a
reference generator on MuJoCo tiles (+0.025, p=0.36), filtered BC onto the measured arm (+0.019,
p=0.48), fed on the Isaac grid at 19,800 episodes (+0.021, CIs overlap), and TRAINED on measured rows
with a matched control (**+0.003, p=1.00**). Against a clip ceiling of 0.907 and `fm` at 0.533, the
generator is still the limit and this is not the lever. **The real lever on tile 6 is HEADING:
`--steer_kp 1.5` is worth ~+0.27 of success**, which points at the operator/planner level this stack
deliberately does not have. Two notes if you revisit: the measured arm sits 0.54 / 0.69 sigma out on
`joint_pos` / `base_ang_vel` (the two blocks a robot CANNOT measure are the two that move LEAST,
0.019 / 0.150 -- so odometry would buy little), and steering is off-distribution on stairs (`|wz| = 0`
on every stair clip). Details: `scripts/flow_matching/measured_history_gap.py` and the archive.

# Two families are excluded from FM training BY DEFAULT

Both because the generator cannot be asked to predict what it cannot see, and both are train-time
switches, so A/B-ing either costs no re-record:

- `skip_distractors` (True) -- the distractor windows.
- `skip_families` (`DEFAULT_SKIP_FAMILIES = ["family_jump_off"]`) -- the BACKWARDS jump-off-a-box
  family. Two forward-facing cameras and nothing behind. `family_jump_off_FRONT` is the forward
  dismount, IS observable, and is NOT dropped -- do not add it by reflex.

**Strictness is asymmetric BY NAME, and the rule lives in `dataset.resolve_skip_families`** -- not in
`train.py` -- because `build_train_val` is a library entry point the probes, the composer and the
tests reach directly. A name in the DEFAULT set is a standing REQUEST (drop it where it exists, print
where it does not); any other name was TYPED, so its absence is a typo and raises. Dropping every
window raises either way. `--skip_families ""` is the off switch. The absent case is not only stale
recordings: a flat-only library has no terrain manifest, so a perfectly current flat recording
legitimately carries no families to name.

**`from_dict` maps a stored `skip_families: null` to `[]`, never to the default.** That value MEANT
"drop nothing", and a checkpoint's saved config is the only record of what its training set
contained. This is the FOURTH instance of that trap (`extero_source`, `extero_encoder`,
`attention_mode`, now this) and the only one that cannot corrupt inference.

**The two exclusion flags spell the same family differently, deliberately.** `train.py
--skip_families` wants the recording's own `family_names` entry (`family_jump_off`); `evaluate.py
--exclude_families` normalizes (`jump_off`, via `sampler.normalize_family`). Do not unify them. The
harness's exclusion is a FLAG rather than a hardcode on purpose: a family you can no longer measure
is a family you can no longer make a decision about.

The legacy per-step `.npz` path applies NEITHER filter and needs no guard:
`_build_train_val_windowed` refuses any recording whose `history_anchor` is not `"plan"`.

# Still open, and deliberately not done

- **`transfer/obelisk/g1_model/mujoco/`** is a FOURTH, unreferenced copy of the robot
  (`g1_27dof.xml` + 4 scenes). Nothing in the launch path reads it. It only adds ways to be wrong.
- The generator's Isaac-vs-numpy parity test at the `_MockCmd` seam covers `robot_now` only.
- `refresh_fk_channels.py`'s 24 world-frame EE VELOCITY channels are still on the old robot.
- `d66420e7`'s local scene tuning (jump-up boxes near spawn, `depth_near.yml viz` re-enabled) was
  lost in an earlier merge and has not been restored.
- `fit_tracking_thresholds.py` does not exist, so the tracking spec is still `bootstrap: True`.

# General Guidelines

- Don't care for backwards compatibility.
- Use the uv virtual environment `env_isaaclab` (at the repo root) for all related python code;
  activate it with `source env_isaaclab/bin/activate`. The exception is `transfer/` off-sim work
  (MuJoCo sim2sim + the ONNX deploy runtime), which imports no IsaacLab and runs in the much smaller
  `env_transfer` -- built by `bash transfer/setup_venv.sh` from `transfer/requirements.txt`, so a
  Jetson or any candidate deploy box can play and benchmark policies without a 10 GB Isaac install.
  On a training box `env_isaaclab` covers both.
- When writing a function, always provide a doc string.
- Use type hints where it is idiomatic.
- Python: variables and functions `snake_case`, classes `UpperCamelCase`.
- When creating a new file, add a comment on the very first line (after a shebang, if present):
  `Copyright <current year> Zachary Olkin. All rights reserved.` Use the file's comment syntax (`#`
  for Python/shell/YAML, `//` for C/C++, `%` for MATLAB).
- Many files under `source/robot_rl/.../tasks/` and `scripts/rsl_rl/` are derived from IsaacLab and
  carry a BSD-3-Clause header (Isaac Lab Project Developers). Retain those headers when editing; do
  not replace or remove them, and do not add the personal copyright line to them.
- Training and playing run via `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` (with
  `env_isaaclab` active, or via IsaacLab's `./isaaclab.sh -p`). See `README.md` for task names.
- `source/motion_gen` must stay IsaacLab-free -- it is imported by generation scripts and tests that
  run without a simulator. Put anything needing `isaaclab` on the `robot_rl` side.
- Both packages need to be importable. `source/robot_rl` is the editable install, but
  `source/motion_gen` is not, so run with `PYTHONPATH=source/robot_rl:source/motion_gen`. In a git
  worktree this also shadows the editable install, which otherwise points at the main checkout.
- Don't build paths with a hard-coded number of `..` hops from `__file__` -- derive them from the
  package root (see `motion_gen.features._REPO_ROOT`). Moving a module between packages has silently
  broken such a path before.
