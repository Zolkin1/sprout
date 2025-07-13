# Robot RL

## Overview

This project is a set of tools for end-to-end development of RL for robots. Specifically, we support:

- RL development using IsaacLab and IsaacSim.
- sim2sim transfer using Mujoco.
- Hardware transfer using Obelisk (ROS2) - note that ROS2 is NOT a dependency of this project - the hardware
interface can be run through a docker/dev-container provided in this repo. See below for more information.

## Installation
When you clone this repo, please use Git Large File System (lfs).

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/robot_rl

- Verify that the extension is correctly by attempting to train:

    - Running a task (see below for a full list of tasks):

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

## Running Tasks
To train a policy run:
```bash
python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
```

Note that right now the only RL_LIBRARY that is tested in `RSL_RL`.

To play the most recently trained policy for a given task run:
```bash
python scripts/<RL_LIBRARY>/play.py --task=<TASK_NAME>
```

If you want to play from a specific checkpoint then you can run the play script with `--checkpoint=<log_dir_checkpoint>`.

For both `train` and `play` you can also specify a number of envs with `--num_envs=###`.

TODO: Discuss custom train and play scripts.

## RL Tasks

RL Task list:

| Task          |   Robot    |   Hardware Tested?   | Description                                                      |
|---------------|:----------:|:--------------------:|------------------------------------------------------------------|
| `G1-flat-vel` |     G1     |  :white_check_mark:  | Basic, hand-tuned, RL walking on the G1 humanoid on flat ground. |


## sim2sim Transfer
This code base has a built in sim2sim transfer (i.e. the policy is trained in IsaacLab and can be run in Mujoco).
Currently, we only support the G1 (as that is the only policy we have right now), but the code is easily extended to other robots.
To run the sim2sim transfer, go to the `transfer/sim/` directory. From this directory run
```
python g1_runner.py --config_file=/path/to/config/file
```

The config file holds all the information about how the RL policy is used including which policy to load, scaling of
observations and actions, and default angles.

To add an additional robot, the associated robot sim files will need to be added into the `transfer/sim/robots/` folder,
the `rl_policy_wrapper` will need to be adjusted a bit, and a new `runner` file will need to be made.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Obelisk Transfer
First, set the environment variable `ROBOT_RL_ROOT` to the path to the `/transfer/obelisk` folder.
Now we can being building the docker container.

### Prerequisites
- Docker and Docker Compose plugin installed
  - Installation guide: [Docker Compose Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
- VSCode with devcontainer support (recommended)

We recommend using VSCode's devcontainer feature to run the Docker environment, although in theory you can run this
as a vanilla docker container.

### Initial Setup (VSCode devcontainer)
1. Navigate to the Obelisk folder in VSCode.
2. Open VSCode command palette (`Ctrl+Shift+P`)
3. Select "Dev Container: Rebuild and Reopen in Container"
4. Choose your preferred configuration:
   - GPU-based (recommended for better performance)
   - No-GPU (if GPU is not available)

At anytime you can open the folder locally by using `Ctrl+Shift+P` then "Dev Container: Open Folder Locally".

At this point you are now inside the docker container and can now use Obelisk and ROS2. Please see the readme in the
Obelisk folder for further instructions.


## Other Dependencies
To run the sim2sim transfer, you will to install these dependencies in your conda environment:
- `pygame`
- `mujoco`

## Updating IsaacLab
Sometimes you will want to updated the version of IsaacLab you are using. To do this, go to the IsaacLab directory
(where you cloned it). Then pull the version you want from git.
Then in that folder run `./isaaclab.sh --install`.

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/robot_rl"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
