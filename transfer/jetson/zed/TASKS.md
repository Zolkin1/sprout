# ZED depth bring-up: what is left

Status as of 2026-08-22, on `zed-box-2` (Jetson Orin NX, L4T R36.4.4, ZED SDK 5.2.3).

**Done:** the container builds and runs; both cameras open by serial; the pair launches
together on the topics the controller subscribes to; the config is measured and committed
(SVGA / NEURAL_LIGHT / 120 fps / stabilization 30 -> 44.7 Hz, capture->subscriber 41.9 ms
median, 53.8 p99). See `docker/README.md` for the numbers and how they were taken.

**Not done:** the depth images are the wrong SHAPE for any policy. Nothing below is
optional before this runs on the robot, except where marked.

---

## 1. Blocking: the images do not match what the policy trained on

- [ ] **1a. Get `tracker.onnx` + `fm.onnx` onto this box.**
  Needed by everything else here; the box currently has no policy artifact at all.
  - This box has NO python venv. Build one first: `bash transfer/setup_venv.sh` (creates
    `env_transfer`; IsaacLab is not needed for any of this).
  - Then `python3 transfer/obelisk/scripts/prefetch_policy.py --hf_policy_folder <NAME>`
    against `zolkin/robot_rl`. **The folder name is unknown to me** -- it is the run that
    matches the cameras currently being trained with, not necessarily the MVP pair recorded
    in CLAUDE.md.
  - The script loads the pair before exiting, so a tracker/generator that disagree about the
    depth cameras are caught here rather than on the robot.
  - Done when: `<cache>/policies/<run>/{tracker,fm}.onnx` exist and the script exits clean.

- [ ] **1b. Set the real FoV, resolution, range and sentinel.**
  `config/{near,far}.yaml` carry `clip_fov: 60x50`, `downsample: 30x26` and
  `depth_{min,max}_distance: -1.0`. Those are UPSTREAM DEFAULTS and match no policy; both
  files are currently identical, so at least one is certainly wrong.
  - `python scripts/rsl_rl/export_sensor_configs.py --run_dir <run>` -> `zed_targets.yml`,
    which gives `width, height, hfov_deg, vfov_deg, min_range_m, max_range_m,
    invalid_range_m, rate_hz` per camera. It imports only the torch-free `deploy` runtime,
    so it runs here in `env_transfer` -- no IsaacLab, no training box needed.
  - Map those onto the node's params. Field names differ; the numbers must not.
  - `invalid_range_m` matters as much as the FoV: a miss has to arrive as the same sentinel
    a missed ray produces in sim.
  - Ceiling: the native cone is **106.2 x 79.5 deg on BOTH cameras** at SVGA (fx = fy ~ 360).
    `clip_fov` only CROPS -- asking for more silently clamps to the image bounds. A 100 deg
    horizontal target leaves only ~3 deg of margin per side for mount tolerance.
  - Done when: the published images match `zed_targets.yml` in shape and FoV, and
    `rate_hz` from the contract is reconciled against the measured rate (see 4a).

- [ ] **1c. Confirm which physical camera is `near`.**
  Currently `near` = ZED X Mini (59249806, the LOWER mount) and `far` = ZED X (44889004,
  UPPER). That binding is the physical mounting read against the contract's geometry -- near
  looks 65 deg down onto ground 0.55 m ahead, far 35 deg onto 1.8 m, putting far ~8 cm
  higher -- and it is an INFERENCE, not a reading.
  - Check it against `tracker.onnx`'s `depth_camera_spec` (mount body + offset per camera)
    once 1a lands.
  - This is worth care: both cameras have the same native cone, so a swap is invisible in
    the imagery. The controller cross-checks the topic COUNT against `tracker.onnx` and
    cannot notice that the two are transposed. The failure mode is a policy that walks
    confidently into things.

---

## 2. Blocking: deployment plumbing

- [ ] **2a. Synchronise this box's clock with the brain's.**
  `depth_capture_age` subtracts a stamp made HERE from a receipt time made THERE. CLAUDE.md
  already records ~8 ms of skew between the desktop and the Thor, enough that the age reads
  SMALLER than `depth_age`, which is impossible. Against a 42 ms median and 54 ms p99, 8 ms
  is a fifth of the margin.
  - chrony or PTP between the ZED box and whichever box runs the controller, or a watchdog
    threshold that explicitly absorbs the skew.
  - Done when: cross-box `depth_capture_age` is never less than `depth_age`.

- [ ] **2b. Decide how the pair is started on the robot.**
  `run_node.sh pair` works by hand. The onboard stack needs to invoke it (or the launch file)
  as part of bringup, in the right order relative to the controller.

- [ ] **2c. Switch camera selection from serial to `gmsl_port`.** (after the mounts are final)
  A serial names a DEVICE; a port names a MOUNTING POSITION, and near/far are positions.
  Replacing a camera then cannot silently transpose the pair. The parameter already exists
  (`docker/patches/0001-camera-selection.patch`); only the config changes.

---

## 3. Robustness

- [ ] **3a. Survive an nvargus crash.** `nvargus-daemon` crashed 3 times in one session
  (`NRestarts=3`), taking a node pair down with it. Partly handled: the launch file now
  respawns a died node after 10 s, and the `/tmp` directory mount lets the container follow
  the recreated socket without being recreated (verified). Still open:
  - the node has NO retry around `open()`, so a camera that is briefly unavailable is a
    startup crash rather than a wait;
  - nothing supervises the launch process itself if it exits;
  - the respawn delay (10 s) is a guess sized to `close()` taking 25-30 s. Watch whether a
    real blip actually recovers, or thrashes.

- [ ] **3b. Make journald persistent.** `/var/log/journal` does not exist, so the box keeps
  no logs across a reboot. A hard lockup on 2026-08-21 was undiagnosable for exactly this
  reason. One `Storage=persistent` and a reboot. Cheap; do it before the next lockup.

---

## 4. Validation, before committing to this config

- [ ] **4a. Re-measure the rate.** The 30 s run gives 44.7 Hz/camera; a later 5 s spot check
  through the launch file read 49.1 Hz. 5 s is too short to trust, but the gap is unexplained
  and the documented number should be the one that survives a long run.

- [ ] **4b. Check NEURAL_LIGHT's depth QUALITY on a real scene.** It was chosen on rate
  alone (44.7 Hz vs NEURAL's 19.2). The bench scene showed 40-52% invalid pixels at 0.36 m
  mean depth, which is a close-range artifact and not a verdict.
  - Fold in the 120 fps exposure cap while testing: 120 fps caps auto-exposure at 8.3 ms
    against 16.7 at 60 fps, which only bites in dim light. Both knobs trade against image
    quality, so test them together and in the real lighting.
  - If quality fails, note that NEURAL is NOT a drop-in fallback: 19.2 Hz concurrent, well
    under the control tick. The fallback needs a different lever than the depth mode.

---

## 5. Housekeeping (non-blocking)

- [ ] **5a. Send `0001-camera-selection.patch` upstream** to `wdc3iii/zed-interface`. It is
  carried locally only because `ZED_REF` is pinned deliberately -- the depth geometry has to
  match training, and a moving ref is a silent way to change it. Upstreaming it plus a
  `ZED_REF` bump deletes the patch.
- [ ] **5b. Upstream `default_params.yaml`'s `resolution: HD720`**, which no GMSL ZED X can
  open. Our configs override it, so this only bites someone using the stock launch.
