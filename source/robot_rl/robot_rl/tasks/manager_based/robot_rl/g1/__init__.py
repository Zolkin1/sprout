# Copyright 2026 Zachary Olkin. All rights reserved.
"""Gym registrations for the G1 offline-clip pipeline.

The pipeline is: motionbricks clip generation -> teacher RL on the clip library ->
flow-matching generator -> closed-loop tracker -> sim2sim -> hardware.  Every task
here is terrain-based: the scene ground is a
:class:`~robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer.ClipTerrainImporter`
built from the clip library itself, and each env spawns on a clip matching the tile it
stands on.  A "flat" run is just a library whose tiles happen to be flat, so there is no
separate flat task -- see ``scripts/rsl_rl/compare_sim2sim.py``, which drives the flat and
stair cases through the same task with different single-clip libraries.

The clip library is REQUIRED from the CLI (``--clip``); there is deliberately no default,
because a stale one silently trains on the wrong terrain.
"""

import gymnasium as gym

from . import agents

# Guard to prevent multiple registrations
_registered = False

##
# Register Gym environments.
##

if not _registered:
    ## =========================================
    # Teacher: open-loop clip tracking on the clip-library terrain (height scan)
    ## =========================================
    gym.register(
        id="G1-teacher-clf-terrain",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_terrain_env_cfg:G1TeacherTerrainEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-terrain-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_terrain_env_cfg:G1TeacherTerrainEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    # Symmetry-augmented variant (sagittal-plane data augmentation).  The clips are full
    # episodic motions with no gait-phase obs, so the Episodic runner is the correct
    # choice (phase negation would be a no-op here).
    gym.register(
        id="G1-teacher-clf-terrain-sym",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_terrain_env_cfg:G1TeacherTerrainEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricEpisodicPPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-terrain-sym-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_terrain_env_cfg:G1TeacherTerrainEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricEpisodicPPORunnerCfg",
        },
    )

    ## =========================================
    # Tracker: closed-loop robot-state-conditioned flow-matching deploy (world-frame CLF)
    ## =========================================
    # The teacher (--load_run) tracks the FM predicted from the LIVE robot-state history
    # each step. Pass the FM via --fm_checkpoint and the library via --clip
    # (--clip_device cpu for the multi-GB libraries).
    gym.register(
        id="G1-tracker-clf-fm-terrain-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_tracker_env_cfg:G1FmTrackerTerrainPlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )
    _registered = True
