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
  - DONE: `env_transfer` is built and both `prefetch_policy.py` and
    `export_sensor_configs.py` import and run in it. See 5c for the one package missing.
  - Remaining: `./env_transfer/bin/python transfer/obelisk/scripts/prefetch_policy.py
    --hf_policy_folder <NAME>`
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

- [x] **1c. Which physical camera is `near`. DONE.**
  `near` = ZED X Mini (59249806, LOWER mount, looking 65 deg down), `far` = ZED X (44889004,
  UPPER) -- confirmed by the operator and consistent with the contract geometry.

  **The mount OFFSETS are not this box's problem**, contrary to an earlier note here. This
  node knows no mount geometry: it publishes a clipped, downsampled depth image and nothing
  else. The extrinsics are consumed by `g1_deploy::CameraPoseFk` in the CONTROLLER, which
  FKs the cameras from the URDF plus the encoders and the IMU's roll/pitch, on the Thor.
  `zed_targets()` emits no extrinsics for exactly this reason -- only shape, FoV, range,
  sentinel and rate cross the boundary. The controller does not even read the message's
  `frame_id`. What this box must get right is the ROLE binding above, plus 1b.
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

- [ ] **2b. Finish wiring `run_hardware_low.sh` to this box.**
  DONE: `obk_jetson_zed_cmd` now runs `run_node.sh pair` (was the single stock `zed` node),
  and `obk_jetson_stop_cmd` matches `zed_depth_pair.launch.py` and kills the launch tree --
  the old pattern matched neither, so teardown would have left both cameras held, and the
  new respawn would have restarted any node killed on its own.
  - Remaining: `OBK_JETSON_HOST` still defaults to `unitree@192.168.123.164`. This box is
    `zed-box-2`, the login is `zolkin`, and its ethernet NIC (`enP8p1s0`) is NO-CARRIER --
    it has no address on the shared network yet. Pin the default once the three boxes are
    cabled, and confirm pubkey auth works FROM INSIDE the controller container.
  - Untested end to end: nothing has yet driven this path from the Thor.

- [ ] **2c. Switch camera selection from serial to `gmsl_port`.** (after the mounts are final)
  A serial names a DEVICE; a port names a MOUNTING POSITION, and near/far are positions.
  Replacing a camera then cannot silently transpose the pair. The parameter already exists
  (`docker/patches/0001-camera-selection.patch`); only the config changes.

---

- [ ] **2d. DDS across the three boxes.** The Thor, this box and the robot share one
  ethernet network, so they share a DDS graph. Three things must line up:
  - `ROS_DOMAIN_ID` -- this container has 2, from `.env` via `env_file`. It must equal the
    onboard graph's, or the controller simply never sees a depth frame.
  - `net.core.rmem_max` is **1048576 here**, 16x under the 16777216 the other boxes require.
    It is not biting yet ONLY because this box has no `cyclonedds.xml` and Cyclone's defaults
    fit. The moment this box gets a config like the others', every ROS node dies at startup
    with `rmw_create_node: failed to create domain`. Raise it in `/etc/sysctl.d/` (a live
    `sysctl -w` is lost on reboot) at the same time as adding the config.
  - whether a `cyclonedds.xml` is needed at all: the desktop/Thor pair uses unicast peers
    because they are on a direct cable. On a real shared LAN with working multicast, the
    default may be fine -- decide deliberately rather than by inheriting one.

- [ ] **2e. (BRAIN, not this box) The URDF's camera mounts must match the real ones.**
  Listed here only so it is not lost: `CameraPoseFk` FKs the cameras from `urdf_path` and
  the contract, so if the URDF's mount frames disagree with where the cameras are physically
  bolted, every xyz channel is mis-registered while the depth channel still looks perfect.
  Nothing on the ZED box can detect that.

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
- [ ] **5c. `pin` (pinocchio) is NOT in this box's `env_transfer`.** It is unsatisfiable
  here: every `pin>=3.1` wheel pulls `cmeel-boost`, which requires numpy>=2, while
  `requirements.txt` pins numpy<2 for the onnxruntime ABI. The venv was built with `pin`
  filtered out, and `requirements.txt` is UNCHANGED -- relaxing the numpy pin affects every
  box and is not mine to decide. Nothing needed here uses it (`deploy/fk.py` and the viewer
  ribbon do; the robot's FK is the C++ `CameraPoseFk` in the container anyway), but
  `transfer/sim` viewers will not run on this box until it is resolved.
- [ ] **5b. Upstream `default_params.yaml`'s `resolution: HD720`**, which no GMSL ZED X can
  open. Our configs override it, so this only bites someone using the stock launch.
