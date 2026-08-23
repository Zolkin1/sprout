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

- [x] **2a. Synchronise this box's clock with the brain's. DONE, both halves, 2026-08-22.**
  The BRAIN serves and this box follows: `chronyc tracking` here now reports
  `Reference ID C0A87BDC (192.168.123.220)`, stratum 4, with offsets settling in the tens of
  microseconds to ~2 ms while chrony converges (RMS 0.64 ms after the first few polls).
  Before this it followed `time-usw6.crimpac.net` -- close to the brain by luck, through two
  separate wifi uplinks.
  - Brain: chrony 4.5, `allow 192.168.123.0/24` + `local stratum 10 orphan`. A query to
    192.168.123.220 is answered `stratum 3`; the same query to 127.0.0.1 is refused, so
    serving is scoped to the robot LAN.
  - Here: `server 192.168.123.220 iburst minpoll 4 maxpoll 6 prefer trust`, written by
    `setup_clock_sync.sh --client`, on top of the Ubuntu pools which stay as a fallback.
  - **VERIFIED on the robot, 2026-08-22**: with the full stack up, the controller reports
    `depth_capture_age` 53.7 / 36.9 ms (near / far) against `depth_age` 1.7 / 0.6 ms. Both
    capture ages exceed both receipt ages, which is this task's done-when: before the sync
    that comparison could come out impossible.
  - **`trust` has a real consequence worth knowing**: the pool sources now read `^?` in
    `chronyc sources` -- excluded for not agreeing with the trusted one -- so this box will
    NOT use the internet to sanity-check the brain. That is the intent (the two agreeing with
    each other matters more than either agreeing with UTC), and the pool is still polled, so
    it takes over if the brain goes away.
  That writes `/etc/chrony/conf.d/obk-robot-lan.conf` with
  `server 192.168.123.220 iburst minpoll 4 maxpoll 6 prefer trust` and restarts chrony. The
  brain runs the `--master` half (chrony there, `allow 192.168.123.0/24`, `local stratum 10
  orphan`). `prefer trust` is the point: being close to the BRAIN matters more here than
  being close to UTC, so the internet pool this box still has stays a fallback and cannot
  outvote the brain.
  - **The direction runs the other way from this file's earlier suggestion**, and not because
    this box's clock is worse. `depth_capture_age` is a receipt time made THERE minus a
    capture stamp made HERE, bags are recorded there, and every other stamp in them is that
    box's -- so making the brain the reference turns a difference between two independently
    steered clocks into one client's offset, which this box reports in one line
    (`chronyc tracking`). And this box's UTC comes over WIFI, while the robot has to work with
    no AP at all; `local stratum 10 orphan` on the brain keeps one self-consistent base either
    way. The cost is a package: `systemd-timesyncd` cannot serve time, so the brain installs
    chrony and apt removes timesyncd. Pointing timesyncd here would have been zero-install.
  - **Correction to what this task claimed.** The watchdog does NOT read this number. The
    controller times depth frames by LOCAL RECEIPT, deliberately (`g1_controller.cc`, and the
    `ControllerDebug.msg` comment says so), so skew here can never stale-latch a healthy
    camera. What it corrupts is `depth_capture_age` -- the only number that says how old a
    frame was when the policy used it -- and every cross-box comparison in a bag. A
    diagnostic, then, not a safety gate: worth doing before any latency claim, not before
    powering the robot.
  - Done when: `scripts/setup_clock_sync.sh --check` on the brain reports this box following
    `192.168.123.220` with a sub-millisecond offset, and cross-box `depth_capture_age` is
    never less than `depth_age`.

- [x] **2b. `run_hardware_low.sh` -> this box. DONE, and exercised end to end 2026-08-22.**
  `obk_jetson_zed_cmd` runs `run_node.sh pair`, `obk_jetson_stop_cmd` matches the pair's launch
  file and kills the launch tree, and the address is pinned to `zolkin@192.168.123.221`.
  - **ssh keys: done from the brain side, one command there** --
    `transfer/obelisk/scripts/setup_zed_link.sh`. It makes a DEDICATED passphrase-free key
    (`~/.ssh/id_obk_zed`) plus an `~/.ssh/config` block with `IdentitiesOnly yes`, then
    `ssh-copy-id`s it here (the one interactive step: this box's password, once). Reusing the
    brain's `~/.ssh/id_ed25519` would not have worked -- it has a passphrase, and a tmux pane
    cannot answer a prompt, so the pane would hang forever showing nothing.
  - Two brain-side traps found while wiring it, both of which produce exactly the "camera
    fault" this task warned about: `docker/.env` still named the onboard minipc
    (`unitree@192.168.123.164`) and that value OVERRIDES the built-in default because compose
    puts it in the container's environment; and a long-running container keeps the value it was
    CREATED with, so editing the file changes nothing until compose recreates it. Both are now
    pinned by `transfer/obelisk/tests/test_zed_link_target.py`, and `setup_zed_link.sh --check`
    prints the container's own value next to the file's.
  - `run_hardware_low.sh` now PREFLIGHTS the link before opening any pane and refuses to start
    with the cause named (no route / no pubkey / host key changed / no `zed_depth` container),
    so a missing key can no longer present as a camera problem. `--no-depth` skips it.
  - **Verified**: a full smoke test driven from the brain brought the cameras and the pair up
    here, over the ssh link and the wired DDS graph. That was the last path nothing had
    exercised from end to end.
  - Also fixed on the way: nothing ever called `obk_mark_jetson`, so `stop.sh --zed` silently
    skipped the remote teardown and left the pair holding the cameras.
  - **DONE END TO END, 2026-08-22.** `run_hardware_low.sh` on the Thor preflighted the link,
    opened the depth pane, ssh'd here, launched `run_node.sh pair` in the `zed_depth`
    container, and both cameras came up (`960x600 -> 30x26`, "Grab loop started"). The brain
    subscribed and measured 47.2 / 47.7 Hz -- slightly BETTER than the 44.9 / 46.2 measured
    locally on this box -- with the controller ticking at 50.0 Hz and `camera FK ready:
    2 camera(s) ... (5 channels per pixel)`. Unarmed, so nothing was commanded.
  - Nothing on the ZED box had to change for that. What blocked it was three defects in the
    BRAIN's robot-driver config (a stale DDS interface pin, `AllowMulticast` off, obelisk's
    `target_ip` default, and a missing `timers:` block) -- see CLAUDE.md.

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

- [x] **2d. DDS across the three boxes. DONE.**
  Addresses are set and all three reachable: robot `.161`, Thor `.220`, this box `.221`,
  on `192.168.123.0/24`. The default route deliberately stays on wifi
  (`ipv4.never-default`), so the box keeps internet without the robot link claiming it.
  - DONE, and it was a real trap: this box is MULTI HOMED (wifi, robot link, l4tbr0,
    docker0) and Cyclone reported `using network interface enP8p1s0 ... selected
    ARBITRARILY from: enP8p1s0, wlP7p1s0, l4tbr0, docker0`. All four rank equally, so the
    choice could have landed on wifi or a docker bridge on any restart, and the brain would
    then see no depth frame while this box looked perfectly healthy.
    `config/cyclonedds.xml` pins it to `192.168.123.221`; Cyclone now reports
    `selected interfaces: enP8p1s0` with no "arbitrarily". Per box, so gitignored, with
    `cyclonedds.xml.example` tracked.
  - Deliberately NOT unicast peers: we are on a wired switch that carries multicast, and
    `AllowMulticast=false` would oblige every box to list every other one.
  - `net.core.rmem_max` is still 1 MB here. Nothing depends on it, because this box's config
    does not request a big receive buffer (it receives nothing and publishes two 30x26 float
    images). Raising it to 16 MB remains worthwhile for fleet consistency, so that copying
    the obelisk config here later does not kill every ROS node at startup.
  - `ROS_DOMAIN_ID=2` here, and CONFIRMED against the onboard graph by the 2026-08-22 smoke
    test: the brain subscribed to this box's depth topics, which a domain mismatch makes
    impossible (it presents as silence on both sides, not an error).

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
  (`NRestarts=3`), taking a node pair down with it. **Now `NRestarts=6`**, last active
  2026-08-22 17:24:41 PDT, measured after the first hardware run.
  - **Untested hypothesis worth one experiment: our own teardown may be causing some of
    these.** `obk_jetson_stop_cmd` ends with `pkill -9` on the launch tree and the node, and
    a process killed with SIGKILL while it holds a camera cannot close it, which is exactly
    the state nvargus handles worst. The 17:24 restart sits close to a `stop.sh --zed`, and
    close is not caused-by: nothing recorded which came first. Settle it by reading
    `systemctl show nvargus-daemon -p NRestarts` on this box immediately before and after a
    `stop.sh --zed`, twice. If the teardown is the trigger, the fix is to let SIGTERM finish
    (`close()` takes 25-30 s here) before escalating, rather than to harden the crash path.
  - Verified separately: a DETACHED session does survive the brain's tmux being killed, as
    `run_node.sh` claims -- the pair kept running after a plain `stop.sh`. What ends it is
    the launch TREE exiting, whether reaped deliberately by `--zed` or by nodes dying and
    respawns being exhausted; the session is the only one, so the tmux server goes with it,
    and the container never restarts (`RestartCount 0`), leaving no record inside it. The
    consequence either way is that the depth pane's "re-running re-attaches to the SAME
    node" quietly becomes a cold start. Partly handled: the launch file now
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
