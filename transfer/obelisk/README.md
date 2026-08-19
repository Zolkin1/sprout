# Quick start

```bash
cd transfer/obelisk/docker && cp .env_example .env && $EDITOR .env   # once, per machine
cd ..
scripts/build.sh                 # once, after a clone or image rebuild
python3 scripts/prefetch_policy.py --hf_policy_folder <name>   # once per policy

scripts/run_sim_low.sh                                # 1. the BRAIN  (terminal A)
scripts/run_sim_world.sh --scene g1_29j_basic_scene   # 2. the WORLD  (terminal B)
python plot_utils/plot_from_bag.py                    # 3. the PLOTS  -> <bag>/plots/

scripts/stop.sh                  # tear both down
```

**That is the default pipeline for every obelisk run: brain -> world -> plots.** The brain
records an MCAP bag by default (`--no-bag` opts out) and the world auto-arms the controller,
so a run needs no joystick input and always leaves something to look at afterwards.

**Order matters, and it is not cosmetic.** The brain needs ~10 s to load the policy pair
(ONNX + CUDA) while the MuJoCo world activates in ~2 s, and MuJoCo has no `user_pose`
equivalent holding the robot up. Bring the world up first and the robot is lying on the
floor before you can walk the FSM to `LOW_LEVEL_CONTROL` -- you then measure a policy trying
to recover from a heap, which is not what it was trained to do. Wait for
`g1_controller up, DISARMED` in terminal A before starting the world.

Each script attaches its own tmux session, so use two terminals, or `Ctrl-b d` to detach the
first. They coexist on one ROS graph by design; `stop.sh` stops both.

On the robot, the only change is which robot config comes up alongside the same brain:

```bash
python3 scripts/prefetch_policy.py --hf_policy_folder <name>   # while it still has internet
scripts/run_hardware_low.sh
```

Every launcher runs on the host, brings the container up, and re-execs itself inside it, so
you never `docker exec` by hand. Each tmux pane comes up fully sourced with its command one
Up-arrow away, so a node you Ctrl-C is one keystroke from restarting.

| script | what it launches |
|---|---|
| `run_sim_low.sh` | the BRAIN alone -- `g1_cpp.yml`, i.e. controller + estimator + joystick FSM (session `simlow`) |
| `run_sim_clip.sh` | the brain with the GENERATOR REMOVED, driven by a pre-made clip (session `simclip`) |
| `run_sim_world.sh` | the sim ROBOT -- MuJoCo world + FSM echo (session `simworld`) |
| `run_hardware_low.sh` | brain + real G1 driver + Jetson ZED depth (session `lowhw`) |
| `build.sh` | colcon build inside the container |
| `stop.sh` | kill the tmux server, reap orphaned launches (`--zed`, `--stop`, `--down`) |
| `net_check.sh` | preflight for a two-machine run: domain/RMW, discovery, endpoint pairing, rates |

## Running the world on another machine

The brain and the world are already two processes with two configs and two obelisk `config:`
identities (`rl_policy` vs `g1_world`), which is exactly what lets them share one ROS graph.
Nothing about splitting them across two boxes needs a new launch path -- `run_sim_low.sh`
launches the brain alone and `run_sim_world.sh` the world alone, unchanged:

```bash
# dev desktop
scripts/run_sim_world.sh --scene g1_29j_basic_scene
# the Thor
scripts/run_sim_low.sh
```

Worth doing when the controller box is also paying for the sim. The world side is not cheap:
1 kHz MuJoCo, two ray-cast depth cameras at 50 Hz, and a 21600-ray lidar scan at 10 Hz. Order
still matters and for the same reason -- **brain first**, then the world.

**What has to agree.** Discovery is stock Cyclone with no configuration anywhere in this repo,
so the requirements are the ones DDS imposes:

- `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION`. `obk` sets both (`2` and `rmw_cyclonedds_cpp`) and
  every pane the launchers open runs `obk`, so this is automatic -- but only inside the
  container. A laptop running `rviz2` outside one has to set them by hand.
- **The same `OBELISK_REF` in both images.** A different pin means different generated message
  definitions, and DDS then refuses to pair endpoints on type hash. There is no error for
  this: both halves come up, both log normally, and no message ever crosses.
- The repo checked out on the desktop with its own `docker/.env` (the world config resolves
  `${ROBOT_RL_ROOT}/../sim/robots/g1/*.xml` from it), and `scripts/build.sh` run there --
  `run_sim_world.sh` also launches `fsm_state_echo` out of `g1_control`.
- `scripts/stop.sh` on **each** box. The orphan reaper is container-local.

**When discovery does not cross.** Cyclone discovers over UDP multicast by default. That is
fine on a wired switch and is why the Jetson depth camera has always just worked. It is *not*
fine over wifi -- campus and guest APs routinely enable client isolation, which drops
multicast between clients -- and it is ambiguous on a box with several interfaces up, where
Cyclone picks one and it need not be the one that reaches the other machine. The fix is an
explicit unicast peer list: copy `docker/cyclonedds.xml.example` to `docker/cyclonedds.xml`,
list the OTHER box (**on both boxes** -- a peer list is not symmetric by itself), and set
`CYCLONEDDS_URI` in `docker/.env`. Then `docker compose up -d --force-recreate`, since compose
reads `.env` only when it creates a container.

**Diagnosing it.** `scripts/net_check.sh`, run on both boxes while both halves are up. It
reads the graph only -- it launches nothing and is safe against a live robot -- and separates
the four ways this fails: an environment mismatch, discovery not crossing, a message-hash
mismatch from differing obelisk pins (a topic with a publisher and a subscriber that never
pair), and everything pairing but a rate being wrong.

**What actually crosses, and what must not.** ~1.5 MB/s total, and depth is not the concern:
the images are 30x26 float32, 3.1 kB a frame. The load is `/obelisk/g1/joint_encoders`
(500 Hz) and `/obelisk/g1/pelvis_imu` (1 kHz), both reliable-QoS. `/obelisk/g1/est_state` is
also 1 kHz but stays entirely on the brain box, because the estimator lives in the brain
config -- which is also where it lives on hardware. If the link cannot carry the sensor
streams, lower those two sensors' `dt` in the world config for split runs; do **not** move the
estimator to the world side to make the split cheaper, because that is a topology hardware
never has.

Clocks need not be synchronised for the run to work -- every watchdog is deliberately on local
receipt time -- but a bag records header stamps from both boxes, so run NTP/chrony if you
intend to compare them.

## When the robot falls over, start here

Do not debug the generator first. `run_sim_clip.sh` replaces it with a baked motion clip and
does not load `fm.onnx` at all, so what remains is the tracker plus everything this stack
does around it -- encoders, estimator, frames, joint order, gains, depth, timing, PD output.
Walk the cases upward and stop at the first that fails:

```bash
scripts/run_sim_clip.sh --case stand                  # 1. hold still. Reference NEVER moves.
scripts/run_sim_clip.sh --case stand_walk_stand       # 2. stand -> walk -> stand
scripts/run_sim_clip.sh --case terrain                # 3. + depth and terrain anchoring
scripts/run_sim_world.sh                              #    (second terminal, as always)
```

**In sim the controller arms itself.** `run_sim_world.sh` starts the FSM echo already
reporting `LOW_LEVEL_CONTROL`, so no joystick input is needed -- MuJoCo has nothing holding
the robot up, so waiting for a button press only means watching it collapse. Pass
`--no-auto-arm` for the hardware-like flow (come up in INIT, walk the FSM by hand). A
joystick command still overrides at any time, damping and e-stop included.

This cannot reach hardware: `fsm_state_echo` exists only because MuJoCo has no execution FSM
and is launched only by `run_sim_world.sh`; on a robot the G1 interface publishes the real
state and the node never runs. The controller itself has no auto-arm and must not grow one
(`tests/test_auto_arm_is_sim_only.py` enforces both halves).

`stand` is the one that matters. Its reference is a settled stand held for the whole clip,
so a robot that cannot hold it has a problem that has nothing to do with any policy. And
`stand_walk_stand` is the SAME clip untruncated -- they share a bit-identical prefix -- so if
`stand` holds and it falls, the difference is the walk transition and nothing else.

All three are known to survive in MuJoCo through the same `transfer/deploy` runtime:

```bash
cd ../sim && python gen_tracker_runner.py --load_run <run> --robot_name g1_29j \
    --clip transfer/obelisk/g1_control/resource/debug_clips/g1_debug_v1.h5 \\
    --clip_case stand --clip_ego_anchor --log
```

so a case that falls in obelisk and not there is a difference in THIS stack. Run the MuJoCo
one first if there is any doubt -- it is the control, not a formality.

On hardware the same mode is `run_hardware_low.sh --clip [CASE]`, and it is how a robot
should be brought up: nothing improvises.

The clip library is built by `scripts/rsl_rl/make_debug_clips.py` out of the same library
the teacher trained on, so the motions are in-distribution rather than invented for the
test. Cases are referred to by NAME (via the library's JSON sidecar) because indices move
whenever the library is rebuilt.

The sim is two launches, not one, for the same reason hardware is: the brain config is robot-less
and the robot comes from its own config, so the brain you test in sim is byte-identical to
the one that runs on the robot. `run_sim_world.sh` stands in for `run_hardware_low.sh`'s
robot half -- it publishes the same topics, including the FSM state (`fsm_state_echo` lives
there because on hardware that state comes from the robot, and running it on both sides
would put two same-named nodes on one graph). It also pairs with
`run_hardware_low.sh --sim-test` for the split-machine test with no robot powered.

Either can be restarted without cycling the other, which is the point: the world can be
reset without losing a brain that took a careful bring-up to arm.

`stop.sh` reaps orphaned `ros2 launch` trees, not just the panes. This matters: the configs
reuse node names, so one surviving launch re-emits lifecycle transitions at the *next*
session's nodes and kills them.

# Running the Docker
For manual/devcontainer use. The launch scripts above do this for you.

## Prerequisites
- Docker and Docker Compose plugin installed
  - Installation guide: [Docker Compose Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
- VSCode with devcontainer support (recommended)

## Environment Setup
Set the `ROBOT_RL_ROOT` environment variable:
```bash
export ROBOT_RL_ROOT=/your/path/to/robot_rl/transfer/obelisk
```

## Creating a `.env`
`cp docker/.env_example docker/.env` and edit it -- it documents every variable. Beyond
user/uid/gid you must set `REPO_ROOT` (the repo root is what gets mounted, so the container
can see `transfer/deploy` and `transfer/sim`), and `INPUT_GID` / `DIALOUT_GID` from your own
host (`getent group input | cut -d: -f3`).

Compose reads `.env` only when it CREATES a container; after editing it,
`docker compose up -d --force-recreate`.

## Running in Docker
We recommend using VSCode's devcontainer feature to run the Docker environment.

### Initial Setup
1. Navigate to the obelisk folder
2. Open VSCode command palette (Ctrl+Shift+P)
3. Select "Dev Container: Rebuild and Reopen in Container"
4. Choose your preferred configuration:
   - GPU-based (recommended for better performance)
   - No-GPU (if GPU is not available)

## Pushing and Pulling from a Registry - DOES NOT WORK RIGHT NOW
Login to your registry of choice (for now I am using github packages).
(i.e. `docker login ghcr.io -u YOUR_GITHUB_USERNAME`)

To push:
```
docker compose -f <compose_file>.yml build
```
then
```
docker compose -f <compose_file>.yml push
```

To pull (i.e. on the on-board computer):
```
docker pull ghcr.io/github_username/robot-rl-obk:no-gpu-latest
```

```
docker compose -f <compose_file>.yaml up
```

# Setup within the Docker
To build the local ROS workspace you can run
```
source scripts/build.sh
```
or do it manually:
Run
```
obk
```
To build and source ROS (per terminal).

After Obelisk has been built you can just run
```
obk-build
```
and
```
obk-clean
```
to clean the Obelisk build folder.

After `obk` we can build this package:
```
colcon build --symlink-install --parallel-workers $(nproc)
```
Then we can source this package:
```
source install/setup.bash
```

## The config tree

The stack is **two** `obk-launch`es on one ROS graph, not one: the brain is robot-less, and
the robot comes from its own config. Distinct top-level `config:` names give each its own
obelisk `global_state`, so either can be restarted without cycling the other.

```
g1_control/config/
  common/low_level.yml   fragment: controller + estimator + joystick FSM (no `config:`)
  g1_cpp.yml             THE BRAIN -- launched identically in sim and on hardware
  g1.yml                 the same brain on the DEPRECATED python controller (see below)
  g1_debug_clip.yml      the same brain with the GENERATOR REMOVED, driven by a baked clip
  g1_robot_hw.yml        the real Unitree driver, alone
  g1_world.yml           the MuJoCo world, alone
  scan/                  ray-caster configs (the depth ones are GENERATED -- see below)
```

Sim (two terminals, or use the launch scripts):
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_cpp.yml
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_world.yml
```

Hardware -- identical brain, different robot:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_cpp.yml
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_robot_hw.yml
```

**The python controller (`g1.yml`) is DEPRECATED.** `g1_cpp.yml` is what the run scripts
launch and what the robot runs; `--python` selects the old one and prints a warning. Two
things keep it alive: bisecting a regression against a known-good implementation, and CLIP
PLAYBACK, which the C++ runtime cannot do at all (no HDF5 clip reader -- it refuses to load
without an `fm.onnx` rather than pretending). On aarch64 the python node cannot reach the GPU
in any case: rclpy is a cp310 C extension and the only onnxruntime GPU build for a Thor is
cp312, so it runs the generator on the CPU at ~471 ms against a 20 ms period.

Debug clip -- same robot half, brain with no generator:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_debug_clip.yml
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_world.yml
```

Note there is no `device=` argument any more. `obk-launch`'s own `bag=` is NOT used: it
hardcodes `-a` and sqlite3 storage, so the launchers run their own `ros2 bag record -s mcap`
against an explicit topic list instead (see "Observability").

### Selecting a policy

There is no per-policy config file. The controller pulls `tracker.onnx` + `fm.onnx` from
HuggingFace and reads every deployment parameter -- joint order, gains, observation terms and
their clips, the depth-camera geometry -- out of their embedded metadata. To run a different
policy, change `hf_policy_folder` in `config/common/low_level.yml` (or keyed-merge-override it
from a launch script); there is no `policy_parameters.yaml` and no rebuild.

Upload a policy with:
```
python scripts/rsl_rl/export_to_hardware.py --run_dir logs/rsl_rl/g1/<run> \
    --hf_repo_id <user/repo> --folder_name <name>
```
which refuses to upload a tracker/generator pair that disagrees about the cameras.

### Generated sensor configs

`config/scan/depth_*.yml` and `config/scan/zed_targets.yml` are **generated** from the
policy artifact so the sim, the real camera and training cannot drift apart:
```
python scripts/rsl_rl/export_sensor_configs.py --run_dir logs/rsl_rl/g1/<run>
```
Re-run it whenever the camera configuration changes, and re-export the policy with it.
`config/scan/mid360.yml` is hand-written -- no policy reads the lidar.

# Run the Lidar
May need to run
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

```
source /home/${USER}/lidar_odom_ws/install/setup.bash
```
```
ros2 launch lidar_scan fastlio_launch.py 
```

# Running on hardware
Follow the above steps to make sure everything work in simulation.

When connected to the robot, attempt to ping the IP address at `192.168.123.161`.
Then, update the `network_interface_name` parameter in the robot section of
`config/g1_robot_hw.yml` to match the name of your network interface (can be seen via `ifconfig`).

Now we can launch the hardware stack -- the brain and the robot, as two separate launches:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_cpp.yml
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_robot_hw.yml
```

The robot interface in Obelisk has a statemachine that we need to "navigate" to enter into low level control mode.
For the G1, we will follow this diagram (can get to L1 by holding down Left Bumper, see Obelisk docs for more info):

```
    Left Trigger        L1 DPAD Left       L1 DPAD down
init ----------> damping -------> user_pose ---------> low_level_ctrl
```
At `user_pose`, the robot will snap to the default position specified by `user_pose` in
`config/g1_robot_hw.yml` and hold that.

At `low_level_ctrl`, the output from the controller node will be applied to robot.

**The controller arms itself off this FSM, not off a button.** Until the robot actually
reports `LOW_LEVEL_CONTROL` it publishes nothing at all and the interface's passive-hold
timer drives the robot; on entering it, whatever feeds the tracker -- the generator or a
debug clip -- drives from the FIRST tick. There is no PD warm-up phase: it put the robot in
a state no training episode started from, and on the generator path it fed the history rows
of zeros instead of letting the learned null-history embedding cover the first N steps.
In sim `run_sim_world.sh` supplies the FSM state and auto-arms; see below.

E-STOP is right trigger.

> **Changed in obelisk `9f52216`:** the `low_level_ctrl -> user_pose` transition was removed.
> From low-level control the only exits are DAMPING and E-STOP -- you can no longer step back
> to the held pose, so plan aborts around damping.

# Setting up the Xbox remote
You can now run the script:
```
bash scripts/joystick.sh
```

Or you can do it manually:
You can make sure that you can see the remote control with
```
sudo evtest
```

Then you can run
```
sudo chmod 666 /dev/input/eventX
```
where `X` is the number that you saw from evtest.

Then we can verify that ROS2 can see it with:
```
ros2 run joy joy_enumerate_devices
```


# Running the docker through tmux session (detached)
#TODO: check if we need --pull always

Start a tmux session
```tmux```

Check existing tmux sessions
```tmux ls```

To dettach from a tmux session
```ctrl +b then d```

To reattach to a tmux session
```tmux attach -t <id>```

If you want to build the container with no cache:
```docker compose -f docker-compose-no-gpu.yml build --no-cache```

To create and start the container, run the following command:
```docker compose -f docker-compose-no-gpu.yml up ```
Note this would also build the container if necessary.

To check existing docker images
```docker ps```

To check running containers
```docker container ls```

To open a new bash terminal for running container
```docker exec -it <container_id or container_name> /bin/bash```

# Observability

## Live, in RViz

```bash
scripts/run_sim_low.sh --rviz           # then run_sim_world.sh as usual
```

Shows the robot in its ESTIMATED pose (so it is a view of what the controller believes, not of
MuJoCo's ground truth -- the two diverging is itself a finding), plus what the generator is
asking it to do:

| what | where it comes from |
|---|---|
| the robot's joint configuration and base pose | `default_robot_viz` + `robot_state_publisher`, off `/obelisk/g1/est_state` and the `g1_model` URDF |
| **orange** root ribbon, the generator's whole F-node plan lifted to world | `/obelisk/g1/debug/fm_plan` -> `g1_control fm_plan_viz` |
| **cyan** foot paths, forward-kinematic from each predicted node's joints | the same, FK'd on the robot's URDF with pinocchio |
| orientation triads every few nodes, and the plan's anchor frame | the same. A yaw error is INVISIBLE in a bare path |
| **green** the 13 window rows the tracker actually observed this tick | `/obelisk/g1/debug/tick` + `/obelisk/g1/est_state` |

The colours match the MuJoCo sim2sim viewer's ribbon deliberately, and both now call the same
FK (`deploy.fk.WindowFK`) on the same model, so the two views cannot disagree by construction
and a discrepancy between them is real. That model is the ROS URDF rather than
`FlowMatchingConfig.fk_model_path`, the MJCF the generator's own FK loss is defined against:
the two agree on these four frames to 4.6e-7 m over random postures
(`transfer/deploy/tests/test_fk_parity.py`, with MuJoCo kept as a test-only oracle), so the
ribbon is not a second opinion, and the robot carries one kinematics library instead of two.

**The green-vs-orange overlay is the highest-value thing on this display.** The served window
should lie on the plan at `cursor + offsets`. It is drawn by a different route -- lifted onto
the MEASURED base rather than the plan's stored anchor -- so a systematic divergence between
them is the re-anchoring, which is otherwise invisible.

The feet are FK'd from the predicted JOINTS, so they are exactly where those joints put them
rather than a second opinion about it; the joint error reaches them amplified by the link
lengths, which is why a gait problem shows up there first.

To view from a laptop instead, see "Running the world on another machine" -- everything RViz
needs crosses DDS, and the requirements are the same.

## Afterwards, from the bag

**Every run is recorded to an MCAP bag**, by every launcher, by default (`--no-bag` opts
out). Bags land in `transfer/obelisk/log/bags/<timestamp>_<tag>/`, under `$REPO_ROOT`
because that is bind-mounted -- obelisk's own `ROS_LOG_DIR` is not, and `stop.sh --down`
destroys anything left there.

```bash
python plot_utils/plot_from_bag.py                    # newest bag -> <bag>/plots/*.png
python plot_utils/plot_from_bag.py <bag> --figures health,joints
python plot_utils/plot_from_bag.py <bag> --interactive
```

Figures are written NEXT TO THE BAG, so runs never overwrite each other, and split by the
question they answer: `health` (arming, watchdogs, input ages, loop period and latency vs
budget, per-camera rate and invalid fraction), `command` (commanded vs achieved velocity),
`tracking` (base error, one axis per subplot: x/y/z and roll/pitch/yaw), `joints` (one
subplot per joint -- reference, measured and PD target, by joint name), `reference` (how far
the window reaches and whether it jumps between re-plans), `generator` (re-plan cadence,
cursor, inference time).

The `health` figure's timing panel splits the two networks, because they are judged against
DIFFERENT budgets and a single total cannot say which one moved: the tracker runs inside the
20 ms tick, while the generator runs on its own thread and is asked for a plan `lead_ticks`
before the boundary it is for, so its deadline is `lead_ticks * dt` (60 ms at the shipped
lead). Below it, the generator's cumulative schedule counters -- late events, plans dropped in
the mailbox, requests skipped because the worker was still busy. A flat line there is a
healthy stretch; any slope at all is the generator failing to keep its cadence, and dropped or
skipped shows up before lateness does. The C++ controller prints the same numbers live, once
every 5 s:

```
timing (5.0s): tick p50 0.74 p95 1.51 p99 1.86 / 20.0 ms | period p50 20.00 p99 20.46 ms
  | trk p50 0.58 p99 1.70 ms | fm p50 17.55 p95 18.47 p99 18.54 / 60.0 ms (20 plans, 21
  expected) | late 0 dropped 0 skipped 0 clamped 0
```

`period` is the ROS timer's own jitter, measured with `steady_clock` between ticks -- worth
watching on its own, because the re-plan cadence is counted in TICKS, so a loop that is not
actually at 50 Hz silently rescales it and every reference offset with it. The line goes to
WARN if any budget is exceeded or any counter moves; otherwise INFO.

`--interactive` opens a frame stepper: both depth images, the reference root path and joint
trajectory, joints at the cursor, and the velocity trace. LEFT/RIGHT step one tick, UP/DOWN
ten, PAGEUP/PAGEDOWN a hundred, HOME/END the ends. Every axis is held FIXED across ticks, so
stepping shows the data changing rather than the axes chasing it.

**Joint order is a trap here.** The policy's order is interleaved
(`left_hip_pitch, right_hip_pitch, waist_yaw, ...`) and the robot's is per limb
(`left_hip_pitch, left_hip_roll, ...`) -- 27 of 29 positions differ on the G1. `action_isaac`
and the reference window are in the POLICY's; `/obelisk/g1/ctrl` and `/obelisk/g1/est_state`
are in the robot's. Subtracting one from the other yields a large, smooth, entirely
fictitious error. Every tick therefore carries `joint_names_isaac`, and the plotter refuses
to compare when it cannot resolve both orders.

`plot_from_bag.py` needs **no ROS installation** off the robot -- MCAP embeds each message's
schema in the file, so `pip install mcap mcap-ros2-support matplotlib numpy` on a laptop is
enough to read a bag full of custom obelisk messages. That, plus Foxglove reading MCAP
natively (it dropped ROS 2 `.db3`), is why the image installs
`ros-humble-rosbag2-storage-mcap` and the recorder passes `-s mcap`. MCAP is also chunked
and append-only, so a bag whose process was killed mid-run -- i.e. every interesting one --
is still readable.

The recorder is a plain `ros2 bag record` in its own tmux pane, **not** `obk-launch
bag=True`: that hardcodes `-a` and the default sqlite3 storage. `OBK_BAG_TOPICS` in
`scripts/lib/obk_launch.sh` is the explicit list, which doubles as a statement of what a run
is expected to contain -- a topic that went missing shows up as an empty one rather than as
nothing at all.

## What is recorded, and what deliberately is not

The observation vector itself is **not** recorded. Every term in it is either already on the
network or reconstructible from what is:

| the policy needs | it comes from |
|---|---|
| `base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel` | `/obelisk/g1/est_state` |
| `depth_near`, `depth_far` | `/depth/{near,far}/image_processed` + the ONNX camera spec |
| `actions` | `action_isaac` on the debug tick |
| `reference_window`, `command_window` | the debug tick carries both directly |

What the two first-party topics add is everything that exists ONLY inside a control tick:

* **`/obelisk/g1/debug/tick`** (`g1_msgs/ControllerDebug`, one per 50 Hz step) -- the
  *effective* command after the watchdogs (which differs from `/vel_cmd` whenever one
  latches, and in clip mode comes from the clip); the raw `action_isaac` before scaling and
  reordering; the served `reference_window`; the arm/warm-up/watchdog state; per-camera
  rate, jitter, range and invalid-pixel fraction; the generator's plan id, cursor and
  anchor; and the **stamps of the input samples the tick actually consumed**. That last one
  is why nothing has to be double-logged: the estimator runs at 1 kHz against a 50 Hz tick,
  so without the consumed stamp there is no way to tell offline which of twenty candidate
  messages fed a given step -- and that alignment is a prime suspect whenever sim and robot
  disagree.
* **`/obelisk/g1/debug/fm_plan`** (`g1_msgs/FmPlan`, one per re-plan, so ~4 Hz at K=12) --
  the generator's sampled window in its own anchor frame, plus everything it was
  conditioned on: the autoregressive history nodes, the command history, the exteroception
  tokens, and the `history_null` flag. The history is the runtime's own previous output, so
  an error there is invisible from outside and compounds; this is the only place it can be
  seen. The plan is published RAW, not re-anchored, so a frame bug in the per-tick
  re-anchoring shows up as a disagreement against the tick's `reference_window` rather than
  being baked identically into both.

`obs_checksum` on every tick closes the loop: rebuild the observation offline from the
bagged inputs, compare, and an exact match proves the reconstruction is the vector the
network actually saw. Without it a replay bug and a controller bug look identical.

Volume is ~330 kB/s, dominated by `reference_window` and the FM plan. `debug_topics:=false`
turns the lot off if the latency line ever says publishing is eating the control period.

## Still outstanding

* **No `est_state` watchdog.** Every other input has a timeout. If the estimator or the
  driver dies, the controller commands at 50 Hz off frozen arrays, forever. The tick's
  `est_state_age` now makes it visible after the fact; it still does not stop the robot.
* **Odometry staleness is invisible to the controller.** The estimator knows;
  `EstimatedState` carries no validity flag; `low_level.yml` claims the controller acts on
  it and nothing does. The FM history is built from base position and world velocity, so a
  FAST-LIO dropout corrupts the reference the tracker chases.
* **Dropout warnings are edge-triggered**; wants a periodic "degraded" heartbeat. The bag
  makes this recoverable offline, which is most of the value.
* **Motor temperatures and error states are not published.** The upstream robot node has
  `pub_motor_temp` (default false), so `motor_surface_temps` / `motor_winding_temps` /
  `motor_states` are empty in every `ObkJointEncoders`. One line in `g1_robot_hw.yml`.
* **`log` / `log_decimation` / `config_name` in `low_level.yml` do nothing** -- undeclared
  parameters, silently ignored by rclpy. Labelled in place; delete them with the next config
  pass.
* **No FAST-LIO pane in `run_hardware_low.sh`.** Nothing starts
  `lidar_scan/launch/localization.launch.py`, so a hardware run comes up with
  `base_pos`/`base_lin_vel` pinned at zero and the odom watchdog inert.
  `scripts/launch_lidar_odom.sh` also points at workspaces the current Dockerfile does not
  build. **In SIM this does not apply** -- `g1_world.yml` publishes `/Odometry` from a MuJoCo
  ground-truth sensor quartet (`TODO(sim-odometry)`) -- which makes the gap *harder* to
  notice, not smaller: the sim has no drift, no latency and no dropouts, so everything
  downstream of localisation is still untested.
* **`/obelisk/g1/odom` from the driver is published and subscribed by nobody** -- a free
  cross-check against FAST-LIO, discarded. It is not in `OBK_BAG_TOPICS` either.
* **`obk_mark_jetson` is defined and never called**, so `/tmp/obk_jetson_active` never
  exists and `stop.sh --zed` can never tear the Jetson node down.
* **Obelisk is pinned to a commit SHA, not a release tag**, with the Dockerfile's own TODO
  saying that is not acceptable on the robot -- and the pin is duplicated by hand in the
  Jetson image.
* **`TODO(mcap-sim2sim)`**: `transfer/sim/gen_tracker_runner.py --log` still writes
  `sim_log.csv` plus `.npy` sidecars. Having it write MCAP with these same schemas would let
  one set of tooling read both sims, which is the natural next step now that the robot side
  is MCAP.

# Setting the IP setting through the terminal

Check the names:
```
nmcli con show
```

To set it to manual:
```
nmcli con mod "Wired connection 1" ipv4.addresses 192.168.123.222/24 ipv4.method manual
nmcli con up "Wired connection 1"
```

To set to automatic:
```
nmcli con mod "Wired connection 1" ipv4.method auto ipv4.addresses ""
nmcli con up "Wired connection 1"
```

# Using Optitrack
We are using the natnet driver located [here](https://github.com/L2S-lab/natnet_ros2). In the docker it is installed to /home/{USER}.

Once you are inside the docker you need to build the package using colcon, but in order to do that you need to activate obelisk so you have ros2. So run:

```
obk
```

make sure you are in `~/natnet_ros2` then run

```
colcon build --symlink-install
. install/setup.bash
```

to build it.

Run it with:
```
ros2 launch natnet_ros2 gui_natnet_ros2.launch.py
```

You may need to run
```
mkdir -p /tmp/runtime-$USER
chmod 700 /tmp/runtime-$USER
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
```

but I don't know for sure.

To run the optitrack forwarding node:
```
ros2 run g1_control optitrack_forwarding
```