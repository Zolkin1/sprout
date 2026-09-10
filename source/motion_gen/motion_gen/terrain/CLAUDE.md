# Terrain

Sim-free terrain geometry, EMITTED here and re-read by the RL side so nothing is recomputed or
allowed to drift. This package must stay IsaacLab-free AND free of `transfer` (see the root
`CLAUDE.md`).

# Terrain is rebuilt from the manifest, so the GROUND can change without regenerating motion

A clip's only tie to its tile is an integer `terrain_id` plus a corner translation. The library
carries a `terrain_manifest` attribute -- per-tile lists of explicit boxes -- and the RL importer,
both MuJoCo emitters and every viewer rebuild the world from it at load. So terrain can be rewritten
offline while the motion stays exactly as recorded, which is what the passes here exist for.

- `decorate.py` varies STAIR WIDTHS by rewriting the manifest (`wide_stairs` / `narrow_landing` /
  `matched`). The shipped library had effectively none -- treads 0.801-1.156 m and approach/landing
  platforms ALWAYS exactly `size_y`, because all three generators forced it -- so against a 16x11
  scan spanning +-0.5 m an 0.8 m flight straddles both edges in every frame.
- **Grow in place, never re-centre**: `size_y` is the bounding box of the tile's CLIPS' lateral
  travel, not of its flight, so `size_y / 2` is where the motion wandered and the flight sits off it
  by a median 0.149 m.
- The **decoration channel is VISUAL ONLY** and lives on its own prim (`decor.DECOR_PRIM_PATH` =
  `/World/decor`), which the depth cameras cast against and physics ignores. It is the one thing that
  changes what the robot SEES without changing what it stands on. Walls ring the RENDERED FLOOR
  rather than the manifest tile, up to four per tile, and the flat tiles get rooms
  (`test_wall_footprint.py`, `test_decor_prim_is_mandatory.py`).
- **The RL TEACHER renders none of it** (`ClipTerrainImporterCfg.render_decor`, default `True`, set
  `False` on `G1TeacherTerrainSceneCfg`) -- nothing in this stack avoids obstacles, so walls and
  rooms teach the tracker nothing while costing render time and depth pixels over ~12k iterations.
  It is a RENDER-time drop, not a second library, so no clip fingerprint or sha256 moves. Everything
  with the GENERATOR in the loop or being RECORDED turns it back on (the `-play` cfgs, the FM
  tracker, the fine-tune tasks, `record_data.py`, `augment_registration.py`) -- a vertical surface it
  has never seen is what the generator misreads as a box.
- **`render_decor` is invisible in every artifact**: `terrain_fingerprint` hashes the MANIFEST, so a
  decor-on and a decor-off run export byte-identical metadata, `tracker.onnx` included. The only
  record is the run's `params/env.yaml` and the importer's startup print, which is why it prints.
  Folding it into the fingerprint would make a decoration-only difference read as a GROUND
  difference to every check that compares it.
- What is NOT decoration: flight/landing WIDTHS and the box SUPPORT variants (`solid` / `floating` /
  `floating_center_leg` / `floating_corner_legs`) are manifest rewrites of the WALKABLE ground and
  are unaffected by the flag. Only `decor=True` overlays, the `flat_tile_decor` rooms and a schema-1
  clutter sidecar are dropped; a schema-2 ground variant passes through (asking for clutter with
  `render_decor=False` is refused rather than silently rendering nothing).
- **`decor.py` imports NOTHING on purpose.** The depth camera cfgs need the prim path at module
  scope, and importing the importer to read one string drags in `pxr` -- which, because hydra
  composes the env cfg BEFORE `AppLauncher` runs, binds pip's `usd-core` instead of Kit's and then
  segfaults when Kit boots a mismatched core. `test_no_pre_app_pxr_import.py` fails the build if it
  comes back.
- **Overlay fields that would diverge between simulators are REFUSED, not best-efforted**, and
  floating overlays render as floating in every MuJoCo/viser path rather than as solid.

## Visual clutter: re-render a recording's pixels, don't re-record

The generator had essentially never seen a vertical surface within camera range -- 0.0% of flat
clips have a wall in reach, terrain tiles' walls sit a median 9.6 m past where the motion stops, and
`family_distractor` is skipped by default. A wall at 2 m is therefore out of distribution in the
direction that most resembles a box, which is what the robot does with it.

The fix needs no re-record and no teacher retrain: `augment_registration.py --visual_only --clutter
<sidecar>` re-renders an existing recording's depth at every recorded pose with scenery added. The
sidecar carries only `decor` overlays, so the ground mesh and every tile corner are bit-identical
and the recorded world poses stay valid (verified: `ref_future` comes back bit-identical, and two
different clutter variants give bit-identical height scans).

- `clutter.py` authors a variant, vetted against the CLOSED-LOOP recording
  (`build_clutter_occupancy.py`) rather than the open-loop clip -- the robot is where it actually
  went, not where the clip said. `author_clutter.py` writes a ~200 kB sidecar, not a 6 GB library
  copy; `ClipTerrainImporter.decor_overlay_path` merges it at load.
- **What limits exposure is ANGULAR SIZE, not range.** Pillars sit closer than walls (1.64 vs
  2.24 m median) and are 13x less visible, and every class realises only 6-10% of the frame it would
  fill if centred. The far camera is 30x26 px over 70x42 deg, rides the torso, and pitches 10-16 deg
  nose-down while walking, so it stares at the ground 2-3 m out. Overhangs at 2.1-2.6 m changed ZERO
  pixels in ZERO windows and are off by default.
- **Wall HEIGHT is unobservable.** At 3 m the frame's top edge cuts a wall at z = 0.87 m, so 1.0 m
  and 3.0 m walls are byte-identical images. Vary it for the viewer, not the model.
- Displacing the robot AND adding clutter needs the penetration filter: the veto is one body radius
  plus clearance and the displacement spends most of it, so `--num_augments` together with
  `--clutter` drops windows whose displaced reference enters a volume.
- `fm_clutter_sensitivity.py` is the before/after number: one checkpoint over the clean and
  cluttered halves with ONE shared noise draw. Windows whose depth did not change are a control that
  must return ~0 -- and it refuses to report anything when that group is empty, which is what
  running it against DR-ON files looks like. Its threshold is the control's **p99.9, never its max**:
  across two clutter draws of one checkpoint the max moved 0.0066 -> 0.0241 m and swung the headline
  47% -> 13%, where p99.9 held and the two draws agreed to five points.
- `merge_recordings.py --variant_groups`: re-renders of ONE rollout must share an episode numbering,
  or `build_train_val`'s by-episode split puts one variant in train and another in val. With k
  re-renders the chance a val episode has no counterpart in train is `val_fraction ** (k-1)` -- at
  seven files and a 5% split, 2e-8. Not only a reported number: `best.pt` is selected on the val
  metric.

## The free-form course contract every course satisfies

One `.h5` per course, manifest-only: terrain and zero clips, still passed with `--clip`. All nine
regenerate with the terrain fingerprint UNCHANGED (`generator_config` sits outside
`fingerprint_terrain_manifest`, because a commanded speed is not geometry). Each declares
`commanded_speed_range_mps` under the key `clip_io.read_commanded_speed_range` actually reads --
box2 and the curb shipped an `intended_speeds_mps` LADDER for weeks, which nothing reads, so the
whole declared-band path was a silent no-op and the loop used its own default. Every builder EMITS
its goal spec.

`varied_courses` is the one where a tile is GEOMETRY rather than an initial condition: 30 box + 30
up-into-down tiles in one library, with the spawn-area chain and a continuous band carrying the
variation a fixed course puts on separate tiles.

- **A declared spawn area REPLACES `pos_jitter_m`** for the whole tile, so an authored ground area
  sized to anything but `freeform.FREEFORM_SPAWN_JITTER_M` silently changes the spawn distribution.
  box2 paid this once at 4x the intended lateral spread. That constant is also what the reset
  synthesises for a tile declaring none, which is what makes a ground-only draw bit-identical to the
  pre-spawn-areas libraries every earlier number was measured on.
- **`height_tol_m` MEANS OPPOSITE THINGS AT ZERO.** `terminations.freeform_goal_reached` guards with
  `if height_tol_m > 0.0` (0 = the test is OFF); `goal_region.in_goal` applies
  `|height_err| <= height_tol_m` unconditionally (0 = EXACT equality, never satisfiable). A spec of 0
  scored an arm 0.0000 over 8400 episodes AND zeroed the reward's success term for five whole
  training iterations. `TODO(height-tol-conventions)`.
- **The goal anchor is a PELVIS height, not the walkable surface.** `height_tol_m` is applied to
  `|root_z - goal_z|` and the clip anchor is a BASE pose, so a surface-height goal puts a standing
  robot ~0.79 m outside a 0.5 m tolerance: `in_goal` could never fire while the termination still
  did, and every arrival would have recorded as a failure. The standing height is read off the
  asset's own `default_root_pose` -- the same quantity the spawn adds to `start_z`, so the two ends
  of an episode are measured against one number.

## box2, the two-box overfit obstacle

`box2.py` + `scripts/rsl_rl/make_box2_terrain.py` clone `mjeval_box2_h040` into Isaac: two 0.40 m
boxes 1.20 m deep, 3 m apart. Chosen because it is the worst course in the mjeval catalogue and the
only one that fails at EVERY speed (0.060 unsteered / 0.145 steered over 20 flow-noise seeds x 10
speeds, and 0/10 at every speed on the deterministic `x0 = 0` draw) while the SINGLE 0.40 m box
scores 0.70. The second box is what breaks it, so it is a MODE choice rather than a capability
limit, and the generator visibly proposes different modes there. Overfitting to it is the point.

- **Geometry is MIRRORED, not imported**, and `transfer/sim/tests/test_box2_matches_mjeval.py`
  imports both and requires them to agree -- comparing distances FROM THE SPAWN, since mjeval puts
  its origin at the spawn and a free-form tile is corner-anchored.
- **The goal is mjeval's, MAPPED**: mjeval scores arrival inside a 2.0 m-deep rectangle,
  `freeform_goal_reached` tests a downrange LINE plus a lateral band, so the line sits at the
  rectangle's NEAR EDGE and the two agree on the boundary that decides the outcome.
- **The IC grid is a grid over TILES**: one tile per approach distance {1.5, 2.0, 2.5, 3.0, 4.0} m,
  identical boxes on every one, because the reset already reads the spawn off the route block. The
  speed axis is a COMMAND, pinned per recording run by `freeform_speed_min == freeform_speed_max`.
  Held out: **(2.25 m, 1.00 m/s)**, off the training grid on both axes and INTERIOR to it, in its own
  `box2_holdout.h5` -- a separate FILE rather than a "do not sample" flag, because a runtime filter
  is the shape of leak the FM's episode split already paid for.
- The course declares box-top SPAWN AREAS so the second half of a traverse appears in the data at
  all; see `scripts/flow_matching/CLAUDE.md` for why every EVAL draws the ground one instead.

## Other terrain notes

- **Terrain clip anchoring**: one placement per anchor-mask segment plus a terrain-AUTHORED yaw
  frame. `goal_frame_yaw == 0.0` on every episode of the shipped libraries, and
  `assert_yaw_free` refuses a library that ever authors turned tiles.
- Two manifest SCHEMAS exist. Stair `decor` works only on the LEGACY bands schema; both live
  pipelines emit the OVERLAY one.
