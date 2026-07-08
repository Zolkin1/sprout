# Copyright 2026 Zachary Olkin. All rights reserved.

import gymnasium as gym

from . import agents

# Guard to prevent multiple registrations
_registered = False

##
# Register Gym environments.
##

if not _registered:
    ## =========================================
    # Teacher: open-loop motion-clip tracking (pre-generated clips, flat ground)
    ## =========================================
    gym.register(
        id="G1-teacher-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_env_cfg:G1TeacherEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_env_cfg:G1TeacherEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    # Symmetry-augmented variant (sagittal-plane data augmentation).  The clips
    # are full episodic motions with no gait-phase obs, so the Episodic runner
    # is the correct choice (phase negation is a no-op here).
    gym.register(
        id="G1-teacher-clf-sym",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_env_cfg:G1TeacherEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricEpisodicPPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-sym-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_env_cfg:G1TeacherEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricEpisodicPPORunnerCfg",
        },
    )

    # Deterministic single-clip playback of the trained teacher on the parity clip
    # (stand -> walk -> run -> stand); ego re-anchoring off by default (override
    # env.commands.traj_ref.ego_anchor_fraction=1.0 to watch the re-anchoring).
    gym.register(
        id="G1-teacher-clf-parity-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_env_cfg:G1TeacherParityPlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    ## =========================================
    # Teacher on TERRAIN: terrain-aware clip tracking (stairs + flat, height scan)
    # The scene ground is a ClipTerrainImporter built from the clip library, and
    # each env is assigned a clip matching the terrain tile it stands on.
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

    ## =========================================
    # Tracker: closed-loop robot-state-conditioned flow-matching deploy (world-frame CLF)
    ## =========================================
    # Closed-loop deploy of the ROBOT-STATE-conditioned FM: the teacher (--load_run)
    # tracks the FM predicted from the LIVE robot-state history each step (the new
    # paradigm). Pass the robot-state FM via --fm_checkpoint and --clip_device cpu.
    gym.register(
        id="G1-tracker-clf-fm-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_tracker_env_cfg:G1FmTrackerPlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    # Terrain variant: same closed-loop FM deploy, but on the ClipTerrainImporter stairs
    # (height scan rides the treads). Load the teacher via --load_run + terrain FM via
    # --fm_checkpoint, --clip_device cpu.
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


