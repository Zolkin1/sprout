# Quick start

```bash
cd transfer/obelisk/docker && cp .env_example .env && $EDITOR .env   # once, per machine
cd ..
scripts/build.sh                 # once, after a clone or image rebuild
scripts/run_sim_low.sh           # brain + MuJoCo world + FSM echo, in tmux
scripts/stop.sh                  # tear it all down
```

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
| `run_sim_low.sh` | brain + MuJoCo world + FSM echo (session `simlow`) |
| `run_hardware_low.sh` | brain + real G1 driver + Jetson ZED depth (session `lowhw`) |
| `run_world.sh` | the MuJoCo world alone, for the split-machine test (session `world`) |
| `build.sh` | colcon build inside the container |
| `stop.sh` | kill the tmux server, reap orphaned launches (`--zed`, `--stop`, `--down`) |

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
  g1.yml                 THE BRAIN -- launched identically in sim and on hardware
  g1_robot_hw.yml        the real Unitree driver, alone
  g1_world.yml           the MuJoCo world, alone
  scan/                  ray-caster configs (the depth ones are GENERATED -- see below)
```

Sim (two terminals, or use the launch scripts):
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1.yml
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_world.yml
```

Hardware -- identical brain, different robot:
```
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1.yml
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1_robot_hw.yml
```

Note there is no `device=` argument any more, and `bag` now defaults to off.

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
obk-launch config=$ROBOT_RL_ROOT/g1_control/config/g1.yml
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
timer drives the robot; on entering it, the controller first spends `warmup_time` holding
the standing pose on stiffened gains to prime the flow-matching generator's state history,
then hands over to the policy.

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

# Observability — known gaps (deferred to the MCAP refactor)

**A hardware run currently records nothing.** This section is the inventory, so it does
not have to be re-derived. The plan is to replace the whole story with MCAP logging
rather than revive the deleted CSV writer; until then, treat everything below as
outstanding. Greppable in the code as `TODO(logging-refactor)`.

**What exists today.** ROS 2 logger lines only, and almost all of them are one-shot or
edge-triggered: policy identity and camera wiring at configure; arm/disarm; dropout
edges for depth, velocity command and odometry; a degenerate-quaternion warning every
50th bad sample; and a 5 s p50/p95/p99 policy-latency line that is only printed when it
exceeds budget unless `verbose: true`. Text goes to the tmux pane and to obelisk's
`ROS_LOG_DIR`. There are no first-party debug topics.

### Blocking, in the order I'd fix them

1. **Nothing is recorded.** No bag, no CSV, no ring buffer. `obk-launch` accepts
   `bag=True` (record-all, via `obelisk_bringup.launch.py`) and no launcher passes it.
   Cheapest real win: pass it in `scripts/run_hardware_low.sh` with an explicit topic
   list rather than `-a`, which would otherwise pull the full `/depth/image_processed`
   stream.
2. **The obelisk log dir is not bind-mounted.** `docker-compose.yml` mounts only
   `$REPO_ROOT` and a few dotfiles, so `ROS_LOG_DIR` lives in the container's writable
   layer and `stop.sh --down` destroys it. Mount it, or set `ROS_HOME` into `$REPO_ROOT`.
3. **Nothing about the depth cameras is recorded or health-checked** beyond staleness, so
   a silently degraded camera reads as healthy. Wants measured Hz, inter-arrival jitter,
   min/max/mean range, and invalid-pixel fraction on a diagnostics topic
   (`depth_obs.invalid_depth_value` already computes the threshold).
4. ~~**No finite check on the action before publish.**~~ **DONE** — `_finite_targets`
   gates every command, falls back to the previous tick's target (not `hold_pose`, which
   can be far away and would command a jump), names which input went non-finite, and
   disarms after 10 consecutive bad ticks rather than freezing on a stale target.
5. **No `est_state` watchdog.** Every other input has a timeout. If the estimator or the
   driver dies, the controller commands at 50 Hz off frozen arrays, forever, silently.
6. **Odometry staleness is invisible to the controller.** The estimator knows;
   `EstimatedState` carries no validity flag; `low_level.yml` claims the controller acts
   on it and nothing does. The FM history is built from base position and world velocity,
   so a FAST-LIO dropout corrupts the reference the tracker chases.
7. **Motor temperatures and error states are not published.** The upstream robot node has
   `pub_motor_temp` (default false), so `motor_surface_temps` / `motor_winding_temps` /
   `motor_states` are empty in every `ObkJointEncoders` message. One line in
   `g1_robot_hw.yml`. Likewise its `logging: true` motor CSV — which additionally needs a
   `timer_logging` entry under `timers:`, or you get a header-only file.

### Also outstanding

- **`log: true` / `log_decimation` / `config_name` in `low_level.yml` do nothing** — they
  are undeclared parameters, which rclpy silently ignores. Labelled in place.
- **Dropout warnings are edge-triggered only**; wants a periodic "degraded" heartbeat.
- **No loop-period / jitter measurement.** The latency line covers `create_obs` +
  `get_action` only, not timer overrun, executor contention, or estimate age.
  `transfer/deploy/benchmark.py` reports the same percentiles off-robot, so the two are
  directly comparable — what is missing is the on-robot loop itself.
- **Arm / warm-up state is not observable off-node**, only as three log lines.
- **The `dbg` tmux pane is empty** while every other pane gets a seeded command.
- **No FAST-LIO pane in `run_hardware_low.sh`.** The estimator subscribes `/Odometry` and
  nothing in the hardware launcher starts `lidar_scan/launch/localization.launch.py`, so a
  run comes up with `base_pos`/`base_lin_vel` pinned at zero and the odom watchdog inert
  (it returns False until the first message ever arrives). `scripts/launch_lidar_odom.sh`
  also points at workspaces the current Dockerfile does not build.
  **In SIM this no longer applies**: `g1_world.yml` publishes `/Odometry` from a MuJoCo
  ground-truth sensor quartet on the pelvis (`TODO(sim-odometry)`), so `base_pos` tracks.
  That makes the gap *harder* to notice, not smaller — the sim has no drift, no latency and
  no dropouts, so everything downstream of localisation is still untested. Running the real
  FAST-LIO in sim is the fix; when it lands, delete the world's `/Odometry` entry or two
  publishers will fight over the topic.
- **`/obelisk/g1/odom` from the driver is published and subscribed by nobody** — a free
  cross-check against FAST-LIO, discarded.
- **`obk_mark_jetson` is defined and never called**, so `/tmp/obk_jetson_active` never
  exists and `stop.sh --zed` can never tear the Jetson node down.
- **Dead tooling that will mislead the next operator:** `plot_utils/plot_ctrl.py`,
  `plot_odom.py`, `plot_trajectory_video.py` and `transfer/g1_log_analysis.py` all read
  log formats nothing writes; the two plotting sections below document them as live. The
  Optitrack section references a `optitrack_forwarding` entry point that no longer exists.
- **Obelisk is pinned to a commit SHA, not a release tag**, with the Dockerfile's own TODO
  saying that is not acceptable on the robot — and the pin is duplicated by hand in the
  Jetson image.

# Plotting Odomety

> **Note:** nothing in the current stack writes the CSVs this and the next section read
> (the controller's writer was removed in `f5bc1ad`, and no node has ever written
> `odom_logs/`). See "Observability" above.

To plot the odometry run
```
python plot_odom.py
```
or 
```
python plot_odom.py /path/to/odom_data.csv
```

You can also use `--start-time` and `--end-time` to specify and start and end time.

# Plotting the Observations and Actions
To plot these, run:
```
python plot_ctrl.py
```

You can also use `--start-time` and `--end-time` to specify and start and end time.

## Setting the IP setting through the terminal

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