# MuJoCo sim2sim

sim2sim is ONNX-ONLY (the torch `gen_tracker_policy.py` is gone, and so is the vendored torch FM
copy `transfer/fm` -- do not re-vendor one to dodge an isaaclab import). What survives is
`transfer/torch_math/quat_math.py`, the isaaclab-free torch quaternion math that
`deploy/quat_math_np.py` is parity-tested against; `transforms_np.py` is tested against the real
`robot_rl.flow_matching.transforms` (`importorskip`-guarded, so that half skips without IsaacSim).

Runs in `env_transfer` (see the root `CLAUDE.md`) -- no IsaacLab, so a Jetson can play and benchmark
policies. The one exception is the mjeval critic arm, which needs `env_isaaclab` because
`env_transfer` carries no torch: `robot_rl.flow_matching.online` imports in 1.1 s pulling 152
`isaaclab.*` modules and ZERO `isaacsim`/`omni`/`pxr`/`carb`, and coexists with mujoco and
onnxruntime. Without `--critic_ckpt` mjeval stays torch-free.

## Depth cameras

- `depth_camera.py` -- an `mj_multiRay` re-implementation (~3.9 ms/frame for both cameras). Casts
  geom groups {1 = per-link visual meshes, 2 = terrain}: self-occlusion comes from group 1, and
  group 0 (collision) stays out because the foot there is a box, not the visual foot.
- `verify_depth_cameras.py` checks geometry / coverage / self-occlusion.
  `scripts/rsl_rl/compare_depth_sim2sim.py` diffs IsaacLab vs MuJoCo pixel-by-pixel (they agree
  everywhere, hands included, to a few mm); `--save_dir` writes labelled colour panels and `--dump`
  feeds `scripts/rsl_rl/view_depth_compare_viser.py`, which shows both sims' robots, camera frames
  and hit clouds in 3-D.
- `gen_tracker_runner` casts BOTH grids when the generator samples at a different resolution: a
  second `MujocoDepthCameras` from `get_fm_depth_camera_spec()` -> `state["fm_depth"]`, with
  `mirror_extrinsics_from` so the jittered mount is shared.
- `Robot.compute_height_scan` casts the privileged 121x121 grid (~20 ms/step for 14641 rays -- the
  per-ray `mj_ray` loop is fine at this size; `mj_multiRay` does NOT apply, since a scan's rays do
  not share an origin). The grid is SQUARE, so a transposed fold is shape-silent:
  `tests/test_height_scan_grid.py` pins it against the coarse 16x11 cast where the grids coincide
  and asserts the transpose would fail. An exact-equality check is a coin flip at yaw=0 on stairs --
  the shared points land on a step edge to 1e-14 m and the two casts differ by one riser.
- **USD and MJCF self-collision sets are DISJOINT**: USD has 11 hand-authored primitives, MuJoCo 36
  visual-mesh bodies with NOTHING on the legs above the ankle -- so training's hand-vs-thigh
  constraint is absent in sim2sim.

# STATUS: what the depth-camera pair covers in sim2sim

The 5-channel depth generator drives the tracker through MuJoCo sim2sim, and the architecture works
-- camera-only exteroception replacing the privileged height scan on the deploy path. Covered with
the generator in the loop: **flat ground, stairs UP, jump-ups, jump-downs.** **Stairs DOWN still
needs work** and is the open item -- the one locomotion mode the pair does not yet handle.

Scenes: `--scene terrain_scene` (in-distribution playground) and `--scene tile0_scene` (ONE real
Isaac tile, manifest-identical to `g1_mb_v5_body.h5` tile 0 -- verified, not assumed).

**Provenance note on the first such checkpoint** (`2026-08-17_12-53-18_depth_xyz5_contract_v1`): it
stores `extero_clip` only because it was stamped BY HAND. It was trained inside the window between
the depth band moving 2.0 -> 5.575 and the stamping fix, so its own config carries `null`; the band
was recovered forensically (the saved envelope of 5.0 is reachable only from 5.575, since 2.0 gives
3.5) and written into a `best_banded.pt` copy. Anything trained after that fix stamps itself.

# The generator's OWN sampling variance decides ~1 in 5 outcomes

`fm_variance_probe.py`. One initial condition, one constant velocity command, one REAL clip-library
tile, N repeats, and **the only difference between repeats is the flow-matching noise draw** -- no
DR, no depth noise, no clip, same standing spawn, and MuJoCo is deterministic.
`terrains.isaac_tile_course` makes a library tile a first-class mjeval `Course` (reading the
manifest, the same producer `generate_clip_tile_scene` reads), because terrain the generator never
saw contributes its own failures and they would be indistinguishable from sampling variance.

Deployed pair, tile 6 = stairs_up 5 risers +0.850 m, tile 20 = stairs_down 7 steps -0.777 m, spawn
2.5 m before the first riser on the flight's centreline:

| approach | speed | stairs_up | stairs_down |
|---|---|---|---|
| 2.50 m | 0.8 m/s | 0.798 (n=500) | 0.816 (n=500) |
| 2.25 m | 0.8 m/s | 0.836 (n=250) | 0.852 (n=250) |
| 2.50 m | **1.6 m/s** | **0.624** (n=250) | **0.668** (n=250) |

- **The outcome is bimodal, not a graded spread**: on stairs_up `progress_x_m` is p50 = p90 = 8.06 m
  against a falls-only p50 of 3.61 m. A trial either walks the whole flight or dies on it, so
  distance-before-failure is mostly reporting that split.
- **Speed ~doubles the failure rate** (1.8x on both tiles independently).
- **Start position does NOT matter**: 0.25 m closer moves success by +0.038 / +0.036 on n=250, inside
  binomial noise. That rules out a spawn-gait-phase artifact.
- **Where it fails differs by family and moves with speed.** stairs_up always fails on the UPPER
  risers (r3 + r5 carry 80%+; r1 never fails). stairs_down at 0.8 m/s fails mostly OFF the flight
  (62% past it, 29% before it, ~8% on it); at 1.6 m/s that inverts (pre-flight 29% -> 4%, on-flight
  8% -> 42%). Falls are ~100% the TILT bound, not height.
- **The deterministic `x0 = 0` draw -- what the hardware controller runs -- is not a neutral sample.**
  It succeeds in 5 of the 6 cells, including `down_fast` where a third of Gaussian draws fail. The
  exception is **stairs_up at 1.6 m/s, where it falls on riser 2 -- and riser 2 carries only 9% of
  the Gaussian failures**, so there the shipped configuration fails deterministically every time
  while 62% of alternative draws would have made it.
- **`--zero_noise_control` is the blocking check and it passes EXACTLY**: sd = 0.000e+00 on both
  progress and distance across repeats, so nothing other than the draw varies. That is the check an
  Isaac state-restore harness cannot pass, and it is why this instrument is trusted where the
  per-window one was not (see `scripts/rsl_rl/eval/CLAUDE.md`).
- **`--scene_tag` is required when two of these run concurrently on one tile.** `CourseRig` writes
  `robots/g1/<robot>_<course>.xml` from the course before loading it, and for a speed sweep the
  geometry is identical, so the race tears the file silently rather than failing to load. The
  generated scenes are derived and deliberately NOT committed.

# mjeval

`mjeval/` is the MuJoCo eval harness: `ARMS` x courses x speeds, seed-paired per cell.

- **mjeval's courses are AUTHORED to vary one geometric parameter and are deliberately not trained
  tiles**, so any perceptive result there is confounded with distribution shift. The tell: a K=1
  baseline pools **0.306** on three authored `stairs_up` courses against **0.875-1.000** on real
  library tiles at the same speed. Use `live_select.py` on real tiles for anything that has to hold
  for the model; `report_measured_pairs.py` is its reducer (`live_select` has none of its own) and
  prints exact McNemar on the DISCORDANT pairs, because a percentile bootstrap over three tiles
  floors out.
- **`ArmRun` passes `allow_robot_asset_mismatch=True` unconditionally**, so the robot fingerprint
  guard warns and continues on every mjeval run. Anything measured there is a statement about a pair
  the fingerprint says is not matched.
- **mjeval's `in_goal` is a +-`GOAL_DEPTH/2` BAND**; the free-form `goal_reached` is a downrange
  HALF-PLANE. The band's near edge cuts arriving episodes short, which is what `stop_on_goal=False`
  exists for when RECORDING rather than scoring.
- **The goal anchor for the critic is EXACT and free, and it is NOT `Course.target_xy`.** Measured
  over 14703 episodes / 2289 clips / 75 real tiles, `goal_pos_w - y_des[clip, clip_len - 51, 0:3]` is
  CONSTANT within a tile (max spread x 7.6e-06, y 1.5e-05, z 0.0 m), so the anchor in tile-local
  coords IS the clip's own terminal pelvis position -- straight from the clip library, no layout
  rebuild, no torch, no IsaacLab. `goal_frame_yaw == 0.0` on every episode, so `frame_offsets` is
  plain subtraction (`assert_yaw_free` refuses a library that ever authors turned tiles).
  `target_xy` is the far platform's CENTRE and those platforms are 7.5-10.1 m deep while the clip
  steps only 0.8-2.1 m onto them: bias stairs_up -3.06 m, stairs_down -3.17, jump_on -0.44,
  jump_off_front +4.92 (the sign flips). It keeps owning `in_goal` / success and is untouched, so
  every mjeval number already measured stays comparable. **Tile -1 is REFUSED, not approximated** --
  the flat sentinel has no tile frame at all (spread 129 m).
- Critic transfer measured against MuJoCo's OWN outcomes (`critic_gate.py`, 48 episodes / 11 falls,
  tiles 6 + 20): AUC(`p_fail_mean`) **0.946** with a label-shuffle control at **0.517 +/- 0.097**,
  AUC(`p_fail_max`) 0.846. The shuffle landing at 0.50 is what makes the rest mean anything. Read
  with care: 2 tiles, and the 2.52 s median lead time counts only the 8 of 11 falls that ever crossed
  0.5, so it says "when it fires it fires early", NOT "it catches falls 2.5 s out".
- `ProprioRing` must stay an exact mirror of `record_rollouts.ProprioRing` -- control rate not
  re-plan rate, ABSOLUTE joint_pos, zeroed on reset. **Do not substitute `policy._obs_history`**
  despite its `history_length=10`: it stores `joint_pos - default_joint_angles`, post-clip, and its
  first push FILLS all ten slots instead of zero-padding. Three silent mismatches at once.
- The runtime is XYZW everywhere (`build_state` stores `wxyz_to_xyzw(qpos[3:7])`), so the frame
  helpers delegate to `deploy.quat_math_np` rather than restating the algebra. Hand-rolling it was
  wrong once, and the error was finite and plausible.
- `isaac_tile_course`'s family->class mapping is an explicit `FAMILY_KLASS` table. A substring scan
  silently resolved `family_stairs_up_pure` (because `"stairs_up"` is a substring) while refusing
  `family_jump_on.h5` and `family_stairs_dn_pure.h5` -- 35 tiles could not be built as courses at all.
- The re-plan edge (`last_plan_id`) must advance unconditionally, not inside the classifier branch;
  any second consumer of the edge would otherwise see EVERY step as a re-plan, silently.
- `view_cell.py` replays any recorded cell and asserts the replay reproduced the recorded row. It
  reads `history_from` / `zero_noise` / `fm_seed` off the table rather than defaulting them (it would
  otherwise replay a `baseline_meas` cell as the generator arm, silently) and lists the arms and
  cells a run actually covered rather than every one `ARMS` declares. `--plans` draws the proposed
  modes.
- `mode_probe.py` -- human labels, blind and interleaved, N noise draws on ONE Isaac-manifest tile --
  is the only mode-selection instrument. Capture must be headless. Its `first_step` 0.60 m tread
  filter makes a 1.2 m box tile unbuildable (its own TODO).
- **The wall-as-box behaviour reproduces here, on a joystick**: `generate_stairs_scene.py --wall`
  puts a visual-only 90-degree corner behind the spawn; driving at it, the policy reads the wall as a
  box and tries to jump onto it. That takes the hardware observation out of a one-off field report
  and into a local loop, and it is the sharpest confirmation that the failure lives on the
  exteroception side rather than the loop/conditioning one.
- **Terrain-relative fall detection**: `robot.py` exposes the string verdict as `failure_reason()`
  with `failed()` as its predicate half. `gen_tracker_metrics.py` records `verdict_source` (which
  answer was used) beside `failure_reason` (what the runner saw) -- the `fell` column was once
  written BEFORE the fall check, so every fall read as a survival.
