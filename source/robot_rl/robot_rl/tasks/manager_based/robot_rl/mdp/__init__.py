# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

# Lazy-load isaaclab.envs.mdp to avoid triggering pxr imports before SimulationApp.
# Wildcard import from this module resolves all lazy attributes, which pulls in pxr.
# Instead, we import the module and fall back to it via __getattr__.
import isaaclab.envs.mdp as _isaaclab_envs_mdp

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # Inherit from the base envs

from .rewards.rewards import *  # noqa: F401, F403
from .observations.observations import *  # noqa: F401, F403
from .observations.depth import (  # noqa: F401
    DEPTH_OBS_FUNCS,
    DEPTH_XYZ_CHANNELS,
    FM_DEPTH_OBS_GROUP,
    depth_camera_pose_yaw_frame,
    depth_camera_specs,
    depth_image,
    depth_image_4d,
    depth_obs_stack,
    depth_xyz_stack,
    find_depth_obs_terms,
    fm_depth_groups,
    invalid_depth_mask,
    normalize_depth,
    pinhole_ray_directions,
    rl_depth_groups,
    unproject_to_yaw_frame,
)
# Explicit, never `import *`, and deliberately WITHOUT a `height_scan` name: IsaacLab's
# `mdp.height_scan` reaches four existing obs terms plus the recorder through the lazy
# __getattr__ below, and re-exporting a same-named function here would shadow it silently.
from .observations.height_scan import (  # noqa: F401
    FM_SCAN_PREFIX,
    FM_SCAN_SENSOR,
    RL_SCAN_SENSOR,
    fm_scan_sensor,
    height_scan_specs,
    rl_scan_sensor,
    scan_obs_stack,
)
from .terminations.terminations import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .events.physical_randomization import *  # noqa: F401, F403
from .events.resets import *  # noqa: F401, F403
from .events.depth import (  # noqa: F401
    randomize_camera_extrinsics,
    randomize_camera_intrinsics,
    resample_depth_noise,
)


def __getattr__(name):
    """Lazily resolve attributes from isaaclab.envs.mdp."""
    return getattr(_isaaclab_envs_mdp, name)