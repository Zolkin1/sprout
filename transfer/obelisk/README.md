# Running the Docker
For now, we suggest bringing the docker up in a `devcontainer` within VSCode.

## Prerequisites
- Docker and Docker Compose plugin installed
  - Installation guide: [Docker Compose Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
- VSCode with devcontainer support (recommended)

## Environment Setup
Set the `ROBOT_RL_ROOT` environment variable:
```bash
export ROBOT_RL_ROOT=/your/path/to/robot_rl/transfer/obelisk
```

## Running in Docker
We recommend using VSCode's devcontainer feature to run the Docker environment.

### Initial Setup
1. Navigate to the obelisk folder
2. Open VSCode command palette (Ctrl+Shift+P)
3. Select "Dev Container: Rebuild and Reopen in Container"
4. Choose your preferred configuration:
   - GPU-based (recommended for better performance)
   - No-GPU (if GPU is not available)


# Setup within the Docker
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

Then finally we can run the stack in sim:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/rl_policy_config.yaml device_name=onboard bag=false
```

## Different Configs/Policies
LIP:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/hardware_config_clf.yaml device_name=onboard bag=false
```

HZD:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/hardware_config_hzd_gl.yaml device_name=onboard bag=false
```
Baseline:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/hardware_config.yaml device_name=onboard bag=false
```

# Running on hardware
Follow the above steps to make sure everything work in simulation.

When connected to the robot, attempt to ping the IP address at `192.168.123.161`.
Then, update the `network_interface_name` parameter in the robot section of `hardware_config.yaml` to match the name of your network interface (can be seen via `ifconfig`).

Now we can launch the hardware stack:
```
obk-launch config_file_path=$ROBOT_RL_ROOT/g1_control/configs/hardware_config.yaml device_name=onboard bag=false
```

The robot interface in Obelisk has a statemachine that we need to "navigate" to enter into low level control mode.
For the G1, we will follow this diagram:

```
     dpad right          squares            dpad down
init ----------> damping -------> user_pose ---------> low_level_ctrl
```
At `user_pose`, the robot will snap to the default position specified by `user_pose` in `hardware_config.yaml` and hold that.

At `low_level_ctrl`, the output from the controller node will be applied to robot.

# Setting up the Xbox remote
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
To create and start the container, run the following command:
```docker compose -f docker-compose-no-gpu.yml up ```
Note this would also build the container if necessary.

To check existing docker images
```docker ps```

To check running containers
```docker container ls```

To open a new bash terminal for running container
```docker exec -it <container_id or container_name> /bin/bash```



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