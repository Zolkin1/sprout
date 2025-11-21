# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stones_sagittal_terrain_levels_termination(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    success_term_name: str,
    neutral_term_names: list[str] = None,
    consecutive_successes_required: int = 3,
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain = env.scene.terrain
    
    # Initialize tracking on first call
    if not hasattr(terrain, "consecutive_successes"):
        terrain.consecutive_successes = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        terrain.initialized = True
        return torch.mean(terrain.terrain_levels.float())
    
    if not getattr(terrain, "initialized", False):
        terrain.initialized = True
        return torch.mean(terrain.terrain_levels.float())

    #check success term
    success_term = env.termination_manager.get_term(success_term_name)
    is_success = success_term[env_ids]
    if neutral_term_names is None:
        neutral_term_names = []
        
    
    # Get all neutral terminations
    neutral_terms = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    for neutral_name in neutral_term_names:
        neutral_term = env.termination_manager.get_term(neutral_name)
        neutral_terms |= neutral_term[env_ids]  # OR operation

    # Any other termination means failure (boolean tensor)
    all_terminated = env.termination_manager.dones[env_ids]
    is_failure = all_terminated & ~is_success & ~neutral_terms
    
    # Update consecutive success counter
    # On success: increment
    # On failure: reset to 0
    # On neutral: keep current count (no change)
    terrain.consecutive_successes[env_ids] = torch.where(
        is_success,
        terrain.consecutive_successes[env_ids] + 1,
        torch.where(
            is_failure,
            torch.zeros_like(terrain.consecutive_successes[env_ids]),
            terrain.consecutive_successes[env_ids]  # neutral: keep current count
        )
    )
    # Move up only if reached required consecutive successes
    move_up = terrain.consecutive_successes[env_ids] >= consecutive_successes_required
    
    # Reset counter for envs that are moving up
    terrain.consecutive_successes[env_ids] = torch.where(
        move_up,
        torch.zeros_like(terrain.consecutive_successes[env_ids]),
        terrain.consecutive_successes[env_ids]
    )
    
    # Never move down - only stay or move up
    move_down = torch.zeros_like(move_up, dtype=torch.bool)
    
    # update terrain levels
    terrain.update_env_origins_and_infos(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())



def modify_reference_cfg(
    env,  
    env_ids: Sequence[int], 
    term_name: str, 
    steps: int = 0
) -> int:
    """Curriculum that modifies the reference Cfg after a certain number of steps."""
    
    command_cfg = env.command_manager.get_term(term_name).cfg

    if env.common_step_counter > steps:
        command_cfg.use_stance_foot_pos_as_ref = True
        return 1  # Signal curriculum change occurred
    
    return 0