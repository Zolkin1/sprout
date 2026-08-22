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

- [x] **1a. Get the artifacts onto this box. DONE.**
  `env_transfer` is built (see the footnote on `pin`), and `tracker.onnx` + `fm.onnx` for
  `TEMP_mode11_decor_lag48_aug_NOT_HW_READY` are fetched. `prefetch_policy.py` 404s on that
  folder because of where it sits, so they were pulled by hand -- see 1d.

- [x] **1b. Set the real FoV, resolution, range and sentinel. DONE, and verified.**
  Derived from `TEMP_mode11_decor_lag48_aug_NOT_HW_READY`'s `tracker.onnx` and applied to
  `config/{near,far}.yaml`:

  | | near | far |
  |---|---|---|
  | FoV | 100 x 70 deg | 70 x 42 deg |
  | output | 30 x 26 | 30 x 26 |
  | range | 0.1 - 7.9 m | 0.15 - 11.9 m |

  Verified at the camera, not just in the file: the node crops to 859x505 and 504x276 px
  (the sizes those angles imply at fx=fy=360), and its own processed `CameraInfo` matches
  the contract's intrinsics to within 0.2% -- near 12.60/18.57 against 12.586/18.566, far
  21.44/33.93 against 21.422/33.866. The residual is integer ROI rounding; you cannot crop
  a fractional pixel. Timing is unchanged by the wider FoV (44.9 / 46.2 Hz, 41.8 / 41.5 ms
  median), and the published minima now sit at the contract floors (0.100 / 0.176 m).

  An invalid pixel needs nothing here: the SDK's NaN/inf passes through and the
  CONTROLLER's `normalize_depth_np` maps every invalid class onto the sentinel.

- [ ] **1d. Swap in the real policy when it exists.** The current checkpoint is
  `TEMP`/`NOT_HW_READY` and sits at the repo ROOT, where `prefetch_policy.py` and the
  controller cannot find it (both expect `policies/<run>/`, pinned by
  `test_policy_cache_layout.py`). The real one WILL be uploaded in the right place and is
  expected to carry the same camera geometry, so this is a re-verify rather than a redo:
  run `export_sensor_configs.py` against it and confirm `zed_targets.yml` still says
  100x70 / 70x42, 30x26, 0.1-7.9 / 0.15-11.9. Cheap, and the alternative is trusting that
  nothing moved.

- [x] **1e. ~45 Hz against the contract's 50. ACCEPTED for now.**
  `depth_stabilization: 30` stays. The contract's `rate_hz` is 50.0 and we measure 44.9 /
  46.2, so the controller reuses the previous frame on roughly one tick in ten -- judged
  acceptable, since sim testing held up at 30 Hz. Turning stabilization off measured 50.4 Hz
  and remains the lever if this ever needs revisiting.

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

- [ ] **2c. `gmsl_port` WORKS -- but decide whether we want it.** Tested through the node
  on all four ports:

  | GMSL port | camera | role |
  |---|---|---|
  | 0 | ZED X, 44889004 | far / upper |
  | 1 | not detected | -- |
  | 2 | ZED X Mini, 59249806 | near / lower |
  | 3 | not detected | -- |

  Two separable things, and only one was ever broken. SELECTION works: the C++
  `setFromGMSLPort` exists and opens the right camera (the PYTHON binding lacks it, which is
  why `pyzed` could not test it -- `sl.InputType` has no `set_from_gmsl_port`). DISCOVERY is
  what is missing: `get_device_list()` reports `gmsl_port: -1`, so nothing can look the
  mapping up. The trial above replaces that permanently -- the ports are physical connectors,
  so the numbers do not move unless a cable does.

  **The tradeoff runs the other way from what task 2c assumed**, because our two cameras are
  DIFFERENT MODELS:
  - `serial_number` (today): a swapped pair of cables changes nothing -- each role opens its
    own camera wherever it is plugged. A REPLACED camera needs a one-line config edit, and
    until it gets one the node fails loudly at `open()`.
  - `gmsl_port`: a replaced camera needs no edit. But a swapped pair of cables SILENTLY
    transposes near and far, and since both cameras share a native cone (106.2 x 79.5 deg)
    nothing in the imagery reveals it -- it surfaces as a policy that misjudges the ground.

  Serial fails loudly, port fails silently. Recommendation: **keep `serial_number`**, and
  keep the mapping above recorded so switching stays a one-line change if that calculus
  changes (e.g. if cameras start getting swapped often).

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

- [ ] **2e. (BRAIN, not this box) Merge the updated URDF before any hardware testing.**
  `CameraPoseFk` FKs the cameras from `urdf_path` plus the contract's mount body and offset,
  every tick, to unproject the depth image. This policy's contract puts both cameras on
  `torso_link_origin_site`:

  | camera | offset (m) | tilt about Y |
  |---|---|---|
  | near | `[0.107, 0.025, 0.045]` | 65 deg |
  | far  | `[0.085, 0.06, 0.285]`  | 15 deg |

  The PHYSICAL mounts were designed to match these numbers exactly, and the matching URDF
  exists on another branch. So the remaining work is the merge, not a measurement -- but it
  is a hard prerequisite for hardware testing, because a URDF that does not describe the
  real mounts mis-registers every xyz channel while the DEPTH channel still looks perfect.
  Nothing on the ZED box, and no image inspection, can reveal that.

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

- [x] **4a. Rate settled, no new measurement needed.** The 49.1 Hz reading was a 5 s
  sample and is discarded. Every 30 s run agrees: 44.87, 44.67 / 44.89, and 44.92 / 46.17
  Hz. Call it ~45 Hz per camera. Nothing operational turns on it.

- [ ] **4b. Check NEURAL_LIGHT's depth QUALITY on a real scene.** It was chosen on rate
  alone (44.7 Hz vs NEURAL's 19.2). The bench scene showed 40-52% invalid pixels at 0.36 m
  mean depth, which is a close-range artifact and not a verdict.
  - Fold in the 120 fps exposure cap while testing: 120 fps caps auto-exposure at 8.3 ms
    against 16.7 at 60 fps, which only bites in dim light. Both knobs trade against image
    quality, so test them together and in the real lighting.
  - If quality fails, note that NEURAL is NOT a drop-in fallback: 19.2 Hz concurrent, well
    under the control tick. The fallback needs a different lever than the depth mode.

---

## 5. Decided against

- **Upstreaming the camera-selection patch to `wdc3iii/zed-interface`, and the HD720
  default with it.** Not needed. The patch mechanism is the steady state, not a stopgap:
  `ZED_REF` is pinned deliberately because the depth geometry has to match training, and
  `git apply` fails the build if the patch ever stops applying -- so it cannot silently
  skip. Upstreaming would trade a pinned ref for a moving one.

- **Resolving the `pin` (pinocchio) pip conflict.** Irrelevant to this box: pinocchio is
  not in the ZED image at all, and the controller gets it from `ros-humble-pinocchio` via
  apt rather than pip, so no deployment path touches the pip package. `env_transfer` here
  is bring-up tooling only (it fetched the policy and derived the geometry, neither of
  which needs FK). It would only matter if someone ran `transfer/sim`'s viewers on this
  box, which is not what it is for.
