# obelisk (ROS2 deploy) -- the default run pipeline

Three commands, in this order, every time:

```bash
cd transfer/obelisk
scripts/run_sim_low.sh                 # 1. BRAIN   (terminal A) -- records MCAP by default
scripts/run_sim_world.sh               # 2. WORLD   (terminal B) -- auto-arms the controller
python plot_utils/plot_from_bag.py     # 3. PLOTS   -> <bag>/plots/*.png
```

**Order is load-bearing**: the brain takes ~10 s to load the ONNX pair and MuJoCo has no `user_pose`
holding the robot up, so a world-first start hands the policy an already collapsed robot.
`scripts/stop.sh` tears both down (and SIGINTs the recorder first so the bag is finalised).

**The brain is the C++ controller** (`g1_cpp.yml`, `g1_control_cpp`); the python one (`g1.yml`) is
DEPRECATED, warns at startup, and is reachable only via `--python`. It survives for bisecting a
regression and nothing else. **CLIP PLAYBACK IS C++ TOO** (`g1_cpp_clip.yml`, the default for
`scripts/run_sim_clip.sh`). Nothing new goes in the python one: the numeric half belongs in
`transfer/deploy_cpp` (tested against the python oracle with no ROS), the ROS half here.

- **Where the reference comes from is chosen ONCE, at load** (`ReferenceSource` in
  `policy_runtime.h`: `FmReferenceSource` or `ClipReferenceSource`), so the tick does not branch on
  it. The clip supplies its own per-offset `command_window`; the generator holds the current command,
  as the deployed joystick does. A >1-channel generator still needs camera FK, which clip mode never
  builds because it loads no generator at all.
- **The clip reaches C++ as a PACK, never as HDF5** -- see `transfer/deploy/CLAUDE.md`.
  `clip_reference_parity_test.cc` holds the port to the python oracle at 1e-11 over the real pack.
  **Two artifacts, neither committed** (both come from a clip library the robot does not carry):
  `scripts/rsl_rl/make_debug_clips.py` writes the `.h5` the python controller reads, then
  `transfer/deploy/tools/dump_clip_pack.py` writes the PACK. Build both before a clip run on a fresh
  machine.
- **When it falls over, take the generator out of the loop first.** `scripts/run_sim_clip.sh --case
  stand` swaps in `g1_cpp_clip.yml`, which loads the tracker ALONE (no `fm.onnx` at all) and drives
  it from a baked clip through the SAME C++ runtime the generator path uses -- so a case changes
  exactly one thing against a normal run: where the reference comes from. Cases, easiest first:
  `stand` (the reference never moves -- if this falls the fault is this stack, not any policy),
  `stand_walk_stand` (the SAME clip untruncated, so a failure isolates the walk transition),
  `walk_backward`, `terrain`. Each is known to survive in MuJoCo through the same `transfer/deploy`
  runtime, so a case that falls here and not there is a difference in the ROS stack. The two
  implementations are held to 1e-11, so a BEHAVIOURAL difference between them is not in the reference
  construction.
- **Pinocchio is the robot's kinematics, in both languages, from one apt package**
  (`ros-humble-pinocchio`, 4.0 in the image; the dev box has 3.9 from pip and both are pinned by the
  same tests). `g1_deploy::CameraPoseFk` FKs the depth cameras every tick -- a generator with
  `extero_channels > 1` unprojects its own image and cannot do that without knowing where the cameras
  are -- and `deploy.fk` draws the RViz/viewer plan ribbon. The "FK'ing the ROS URDF would be a
  SECOND OPINION" objection is measured, not argued: URDF vs MJCF is 4e-13 on the camera chain and
  4.6e-7 m at the wrists, so the deploy code has ONE kinematics library rather than two. It is not an
  image-size win -- obelisk's own sim is a MuJoCo sim, so the image installs it regardless. The URDF
  is a controller PARAMETER (`urdf_path`), not a constant.
  - Two CMake traps, both of which fail pointing somewhere else: pinocchio 4.0's `pinocchio_parsers`
    reaches `coal::coal` and its own config does not `find_dependency` it, and an imported target is
    DIRECTORY-SCOPED -- so `deploy_cpp` finds coal and promotes it (`IMPORTED_GLOBAL`), or the
    consumer's own `target_link_libraries` line is what errors.
  - **A green build on the dev box says nothing about four files.** `find_path(onnxruntime)` fails
    there, so `ort_session` / `tracker_session` / `fm_session` / `policy_runtime` are NOT COMPILED and
    a syntax error in them survives (it did: a botched extraction ate the head of
    `GaussianNoise::Fill` and was committed). The configure WARNS; compile in the container.
- **The cadence is not a knob** and **there is no PD warm-up** -- both rules live in
  `transfer/deploy/CLAUDE.md`, and both runtimes refuse a cadence mismatch.
- **Auto-arm is SIM ONLY and structurally so**: it is a parameter of `fsm_state_echo`, a node that
  exists only because MuJoCo has no execution FSM and is launched only by `run_sim_world.sh`. The
  controller has no auto-arm and must not grow one (`tests/test_auto_arm_is_sim_only.py`).
- CUDA Graph capture for the generator is `fm_cuda_graph: true` in `g1_cpp.yml`; the traps are in
  `transfer/deploy/CLAUDE.md` and every one of them fails silently.

## The obelisk SIM applies the MJCF's PD gains, and they were STALE (fixed 2026-08-24)

The tracker fell in the obelisk sim on motions MuJoCo sim2sim walked through cleanly, and the cause
was neither policy nor stack: **both hip pitches were 2.46x too soft there**, kp 40.18 against the
99.10 the policy exports. Same tracker, same clip, same world -- `walk_backward` went from 3/5 falls
to 0/5, and travel tightened from a 1.99-3.75 m spread to 2.48-2.50 m.

**Three consumers, and only ONE reads the MJCF's gains**, which is why this was invisible:

| consumer | gain source |
|---|---|
| `transfer/sim` sim2sim | the policy's, via `Robot.set_pd_gains_from_policy` -- **overwrites the MJCF** |
| the hardware driver | the policy's, via `msg.kp` / `msg.kd` on the wire |
| **the obelisk MuJoCo sim** | **the MJCF** |

Obelisk's sim is `void ApplyControl(const ControlMessageT& msg) { SetSharedData(msg.u_mujoco); }` --
straight into `data.ctrl`. It never reads the published gains, so the MJCF alone decides how stiff the
robot is there, and both controllers publish `kp`/`kd` that only the robot consumes.

**Cause:** `make_g1_mjcf.py`'s `TEMPLATE_SECTIONS` copied `default` and `actuator` VERBATIM from its
template, `g1_29j.xml` -- the pre-mode_11 robot. mode_11 moved hip pitch from the 14.3 gear to the
22.5, and hip pitch is the only joint whose gear changed, so it was the only mismatch.
**Fix:** `apply_actuator_gains` derives them from the same `JOINT_ARMATURE` table via the same law
the RL side uses (`kp = J w^2`, `kd = 2 z J w`), so no number is duplicated and a future gear change
moves both. It refuses rather than guesses on an actuated joint with no armature, a `<position>` with
no default class, and a class shared by joints whose gains differ.
`test_actuator_table.py::test_mjcf_actuator_gains_match_the_cfg` pins the result -- it resolves gains
the way MuJoCo does (element attribute wins, else the named class) and fails naming the joint, both
numbers and the regeneration command.

## Logging: MCAP, and what is deliberately NOT in it

Bags land in `log/bags/<timestamp>_<tag>/`; figures next to them. The observation VECTOR is not
recorded -- every term is either already on the network (`est_state`, the depth topics) or
reconstructible -- but `obs_checksum` is, so an offline reconstruction can PROVE it rebuilt what the
network saw. `/obelisk/g1/debug/tick` carries only what exists nowhere else: the post-watchdog
command, the raw `action_isaac`, the served reference window, arm/watchdog state, depth health, and
**the stamps of the input samples each tick consumed** (the estimator runs at 1 kHz against a 50 Hz
tick, so without them no recording can say which sample fed which step).
`/obelisk/g1/debug/fm_plan` carries the generator's output plus everything it was conditioned on,
once per re-plan.

Three traps, each of which produced confident wrong numbers before being found:

- **Joint order.** The policy's is interleaved (`left_hip_pitch, right_hip_pitch, waist_yaw, ...`),
  the robot's is per limb -- 27 of 29 positions differ on the G1. `action_isaac` and the reference
  window are in the POLICY's; `/ctrl` and `/est_state` are in the robot's. Subtracting one from the
  other gives a large, smooth, fictitious error. Every tick carries `joint_names_isaac`; the plotter
  refuses to compare without both.
- **`rosbag2_py` deserialises with the INSTALLED message definition**, not the bag's, so a bag
  recorded before a `.msg` edit is read through the wrong layout -- silently, returning plausible
  values for the wrong fields. The `mcap` reader uses the bag's own schema and is the default; it
  also needs no ROS and no `metadata.yaml`.
- **A `.msg` change needs g1_msgs REBUILT** (`rm -rf build/g1_msgs install/g1_msgs &&
  scripts/build.sh`) -- bindings are generated, not symlinked. The controller now refuses to start on
  a mismatch rather than recording nothing for a whole run. Relatedly, a literal `*/` anywhere in a
  `.msg` comment terminates the generated C block comment and the rest compiles as code
  (`tests/test_msg_comments.py`).

Also pinned by tests: a config parameter the node never declares is silently ignored by rclpy, so
`tests/test_params_are_live.py` fails the build on one (this bit `log`, `log_decimation`,
`config_name` and `fm_update_every`). The generator's re-plan cadence comes from `tracker.onnx`
metadata and nowhere else.

**Do not measure a stamp age with `ros2 topic echo --once` + `date`.** That method carries ~250 ms of
its own process startup: a topic stamped on THIS box read +249 ms through it while the depth topic
read +273 ms. Only the difference means anything. The controller's own `depth_capture_age` is the
number, and it is in the bag.

# Hardware: what has actually run

**The pair drove the real G1 on 2026-08-25** -- ZED pair over the ssh link, the perceptive generator
and the tracker on the brain, the C++ controller commanding the motors, first attempt. What was
tested, and nothing beyond it:
- **Flat ground: standing, and walking forwards, backwards and turning.** All fine.
- **A box jump-up: it got onto the box but stumbled at the top.** Unexplained.
- **Stairs: NOT tested.** Neither up nor down.

So the claim this supports is "the deployed pair walks on flat ground and can climb a box", and
specifically not that terrain works. The pair: tracker `2026-08-20_16-36-10_mode11_decor_plan_v1`,
generator `2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx`. **That generator is a record of what
drove the robot, NOT the one to run now** -- see "WHICH ARTIFACTS TO USE" in the root
`CLAUDE.md`, which names the current one and why this one keeps being picked up by mistake.

The stack had been run in the obelisk sim for weeks; what had never been exercised was the ROBOT half
-- the driver, the motors, the real cameras, the real IMU -- and that is the part that worked first
time. This is also the first policy deployed with `fixed_waist: false`, i.e. with `waist_roll` and
`waist_pitch` actually driven. The driver's own ordering (`joint_names_` = the 27-DoF set then the
two extra waist joints) matches `default_joint_names` exactly, and every command path is matched BY
NAME, so nothing depended on the two orders agreeing -- but the `user_pose` array does, and it is
indexed by position in `default_joint_names` (`user_pose_[def_ind]` in `g1_interface.h`), which is
why waist roll and pitch are its LAST two entries.

## The three config defects that blocked bring-up were ALL in the driver path

Each named something other than its cause, and the sim path had been exercised for months while the
hardware one never had:

- **`docker/cyclonedds.xml` was still pinned to `192.168.50.20`**, the desktop split-sim cable, which
  the robot box does not have. Cyclone treats an absent pinned interface as FATAL, so EVERY ros node
  died at `rmw_create_node: failed to create domain` with the real reason one line earlier ("does not
  match an available interface"). The file is GITIGNORED per box, so nothing in the repo could catch
  it. It now lists BOTH roles' addresses with `presence_required="false"`.
- **`AllowMulticast` must be ON for the robot LAN.** It was false (correct for the desktop cable,
  where unicast peers are safer). The ZED box cannot be reached that way: its config uses multicast
  and lists no peers, so its participants keep Cyclone's DEFAULT ephemeral ports -- and a portless
  `<Peer>` is resolved by probing participant INDEXES, which an unindexed participant never answers.
  Brain and cameras would simply never see each other, both looking healthy alone.
- **`target_ip` is THIS box's address, not the robot's.** obelisk uses it only to pick the local
  interface (`GetInterfaceName(target_ip, ...)`) and defaults to `192.168.123.222`, Unitree's user-PC
  convention. The brain is `.220`, so the node aborted with "No interface found with IP:
  192.168.123.222" -- which reads as a robot connection fault. Set by ADDRESS rather than
  `network_interface_name` on purpose: an interface name changes when the cable moves between an
  onboard NIC and a USB adapter.
- **`g1_robot_hw.yml` had no `timers:` block**, so `timer_ctrl` -- the passive-hold timer
  `G1Interface` registers in `on_configure`, which drives the motors while the FSM sits in USER_POSE
  or DAMPING -- had no period, and the configure transition threw "Timer entry is missing required
  field 'timer_period_sec'". A missing entry only WARNS; it is a registered timer with no entry that
  throws.

Host prerequisites on every box, neither optional:
- `net.core.rmem_max=16777216`. Cyclone treats a smaller kernel as FATAL and every ROS node dies at
  startup with `rmw_create_node: failed to create domain`.
- `<MaxAutoParticipantIndex>32</MaxAutoParticipantIndex>` in the cyclone config. Unicast peers
  require `ParticipantIndex auto`, which caps participants at 9 by default, and the world's bringup
  exceeds that. Cyclone's own default is `none`, which has no cap but needs multicast.

Measured unarmed with the real cameras (tick 50.0 Hz, depth 47.2 / 47.7 Hz): `depth_age` 1.7 /
0.6 ms (the controller's own contribution) and `depth_capture_age` 53.7 / 36.9 ms (the whole pipeline
from the ZED's own capture stamp). Both capture ages exceed both receipt ages, which is the done-when
for the clock sync -- the number that used to be able to read impossibly small.

## The ZED box: the depth cameras arrive over an ssh link, and one clock

On hardware the cameras are NOT on the brain. `run_hardware_low.sh`'s depth pane ssh's to the ZED box
(`zolkin@192.168.123.221`, `OBK_JETSON_HOST` in `docker/.env`) and runs `run_node.sh pair` in the
container there, so the link is part of the control path. Robot LAN, all three boxes: robot `.161`,
brain `.220`, ZED `.221`. Two one-time setups, both idempotent: `scripts/setup_zed_link.sh` (ssh) and
`scripts/setup_clock_sync.sh --master` on the brain plus `--client` on the ZED box (time).

**Every failure of this link presents as a depth pane that prints nothing, which on a robot reads as
a CAMERA fault.** So `obk_jetson_preflight` runs before any pane is built and names the cause
instead: no route, no pubkey, changed host key, or no `zed_depth` container.
`run_hardware_low.sh` refuses to start on it (every policy here is a depth policy, `--clip` included
-- only the generator is absent there); `--no-depth` is the deliberate way past.

- **The key is NOT `~/.ssh/id_ed25519`**, which has a passphrase on both boxes. A tmux pane cannot
  answer a prompt, so it would hang forever showing nothing. `setup_zed_link.sh` makes
  `~/.ssh/id_obk_zed` with no passphrase plus an `~/.ssh/config` block with `IdentitiesOnly yes`
  (without which ssh still tries the default identity first). `~/.ssh` is bind-mounted, so the key
  made on the host IS the key the pane uses.
- **`docker/.env` OVERRIDES the fallback in `obk_launch.sh`**, because compose puts it in the
  container's environment -- and it named the onboard minipc (`unitree@192.168.123.164`) long after
  the cameras moved, which answers ssh and would have been a wrong-box success. Worse, a RUNNING
  container keeps the value it was CREATED with, so editing the file changes nothing until compose
  recreates it. `test_zed_link_target.py` pins the tracked pair; `--check` prints the container's own
  value beside the file's.
- **The BRAIN is the time master**, not the ZED box, even though the ZED box is the one with
  internet: `depth_capture_age` is a receipt time made on the brain minus a capture stamp made on the
  ZED box, the bags are on the brain, and every other stamp in them is the brain's -- so one client's
  reported offset replaces a difference between two independently steered clocks. The ZED box's UTC
  comes over WIFI and the robot must work with no AP, which `local stratum 10 orphan` covers. The
  cost, stated because the other direction was free: `systemd-timesyncd` is an SNTP client and cannot
  serve, so chrony replaces it on the brain.
- Live since 2026-08-22: the ZED box tracks `192.168.123.220` (stratum 4 off the brain's 3) at tens
  of microseconds. `trust` on the client means the internet pool there is excluded for disagreeing
  with the brain (`^?` in `chronyc sources`) -- deliberate, since the two boxes agreeing with each
  other matters more than either agreeing with UTC, and the pool is still polled so it takes over if
  the brain goes away.
- **Clock skew is a DIAGNOSTIC problem, not a safety one.** The depth watchdog times frames by local
  receipt on purpose (`g1_controller.cc`), so a skewed camera clock cannot stale-latch a healthy
  camera. It corrupts `depth_capture_age` -- the only number saying how old a frame was when the
  policy used it -- and every cross-box comparison in a bag.
- **`chronyc tracking` on the master is NOT what clients get.** While the `local` reference is active
  -- from startup until a real source's root distance falls under the 1 s `distance` threshold, and
  again if every upstream is lost -- tracking reports `7F7F0101` / stratum 10 / zero root delay, so
  the master reads as unsynchronised for the first couple of minutes while it is already serving the
  real upstream. `setup_clock_sync.sh` therefore probes the server over the wire (`served`), which
  also proves `allow` is scoped -- the same query to 127.0.0.1 must time out.
- `obk_mark_jetson` had never been CALLED, so `stop.sh --zed` skipped the remote teardown silently
  and left the pair holding the cameras. `run_hardware_low.sh` sets it now.

## Running the BRAIN on another machine (the Jetson Thor), over ssh

The split needs no new launch path: `run_sim_low.sh` is the brain alone and `run_sim_world.sh` the
world alone, so the brain runs on the Thor and the MuJoCo world on the dev desktop. Both halves must
agree on `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` (obk sets them) and each box needs its OWN
`docker/cyclonedds.xml` with the addresses swapped. That file is GITIGNORED, so the link is described
here rather than in it: `192.168.50.10` desktop (world), `192.168.50.20` Thor (brain), `zolkin@`,
direct cable on a private subnet. `scripts/net_check.sh` on BOTH boxes while both halves are up is
the preflight, and it names the four ways discovery fails.

```bash
ssh -t zolkin@192.168.50.20 'cd ~/sprout/transfer/obelisk && scripts/run_sim_low.sh'  # brain
scripts/run_sim_world.sh --scene g1_perceptive_tile0_scene                            # world, here
scripts/fetch_brain_bags.sh          # bags are recorded ON THE BRAIN's box
```

**The known root cause of the two-box falls: the generator cannot tolerate a stale depth image, with
a cliff between 20 and 40 ms. The Thor is NOT worse.**

Traps, each of which cost a debugging cycle:
- **`stop.sh` is LOCAL ONLY** -- there is no ssh in it. Tearing down here leaves the remote brain
  running and still recording. Stop both.
- **`run_sim_low.sh` RE-ATTACHES** when its tmux session exists, so a config change is ignored until
  `OBK_RESTART=1`.
- **The Thor reaches GitHub only through a running ssh AGENT.** The key has a passphrase, so whether
  `git push` works is a property of the CONTEXT, not the box: from a desktop session the
  gnome-keyring agent holds it unlocked and a push just works, while the same command with the agent
  stripped fails with "make sure you have the correct access rights" -- which is what a cron job, a
  detached script, or the obk container (no agent socket mounted, only `~/.ssh`) all get. That last
  one is why the ZED depth link needs its own passphrase-free key. Where there is no agent, move
  commits with a bundle over the private link, and FETCH BEFORE reverting anything local:
  `git bundle create /tmp/x.bundle <their-head>..<branch>`, scp, then
  `git fetch /tmp/x.bundle <branch> && git merge --ff-only FETCH_HEAD`.
- **`ssh -tt` needs a LOCAL tty** to allocate a pty from, so it fails when backgrounded inside a
  script. Run `script -qec '<cmd>' /dev/null` on the far side instead.
- **There is no tmux on the Thor HOST** (only inside the container).
- **`timeout` cannot exec a shell function.** `timeout N my_ssh_fn ...` fails instantly and silently;
  put the timeout on the far side of the ssh.
- **`pgrep -f <pattern>` matches its own command line**, and over ssh that line arrives containing
  the pattern. Bracket it (`'[r]os2 bag record'`) and keep the literal out of any echo in the same
  command -- this produced four separate false conclusions in one session.
- **Never copy a bag from a live run**: it comes back torn at message ~0. Stop first.
- **Remote bags go to `log/bags_thor/`**, never `log/bags/`. Bags are named by timestamp only, so
  mixing two machines in one directory makes their origin unrecoverable.
- **The clocks are not synchronised** on this desktop/Thor pair (chrony here, systemd-timesyncd
  there, ~8 ms apart). Cross-box `depth_capture_age` is offset by that and can read SMALLER than
  `depth_age`, which is impossible. Rate matches to 1 ppm, so only the absolute value is affected.
  Nothing in the control path uses a remote stamp; the watchdog deliberately uses receipt time. On
  the ROBOT LAN this is fixed rather than tolerated (see the ZED box section).

# The deployed artifacts

`logs/rsl_rl/g1/2026-08-20_16-36-10_mode11_decor_plan_v1` holds `tracker.onnx`,
`policy_parameters.yaml`, and `2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx` -- the pair that drove
the robot. `config/scan/depth_{near,far}.yml` are regenerated from that run and carry the far camera
at 15 deg (`rot: [0.99144, 0, 0.13053, 0]`), not the 35 deg placeholder. `low_level.yml`'s
`hf_policy_folder` names it.

**A default path to an artifact is a thing that rots quietly**, and this stack has been bitten three
times by the same defect -- a `_DEFAULT_RUN_DIR` or `G1_DEPLOY_POLICY_DIR` missing the `policies/`
element and/or naming a pre-refresh run. Once it silently SKIPPED two whole C++ test binaries; once
it left `policy_params_test.cc` asserting PLACEHOLDER camera mounts (near z 0.099426, far 0.38,
against the as-built 0.045 / 0.285) and passing for weeks against a fixture describing the same stale
robot. That is precisely the failure the camera contract exists to prevent: a wrong mount is a
plausible cloud in the wrong place, invisible to every shape check, sitting inside the one test meant
to catch it.

When re-running `dump_metadata_fixtures.py` / `dump_obs_fixtures.py`, note the two halves come apart
very differently: every obs `.bin` regenerated BYTE-IDENTICAL across the refresh (the observation
contract -- term names, widths, history lengths, clips, scales -- had not moved, and the assembly is
bit-stable across numpy 1.26 -> 2.4), while the metadata fixture differed on 89 keys and was missing
93. Check the metadata half.
