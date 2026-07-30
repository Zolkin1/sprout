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

    ## =========================================
    # Body-link tracking variant (DeepMimic / Sonic style).  Same env as above, but the CLF
    # grades -- and the policy observes -- each tracked link's pose + twist in the PELVIS
    # frame instead of the reference joint angles.  The world-frame skeleton (pelvis + the
    # four end-effectors) is tracked in both, so global foot placement stays penalised.
    #
    # Needs a clip library carrying root-frame channels; widen one with
    # ``scripts/rsl_rl/backfill_root_frames.py``.  Point BOTH tasks at the SAME widened
    # library to keep the joint-vs-body comparison controlled.
    ## =========================================
    gym.register(
        id="G1-teacher-clf-terrain-body",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_terrain_env_cfg:G1TeacherTerrainBodyEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-terrain-body-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_teacher_terrain_env_cfg:G1TeacherTerrainBodyEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    # Symmetry-augmented variant (sagittal-plane data augmentation).  The clips are full
    # episodic motions with no gait-phase obs, so the Episodic runner is the correct
    # choice (phase negation would be a no-op here).
    #
    # !! CURRENTLY BROKEN -- these two raise at the first PPO update:
    #      ValueError: Symmetry augmentation has no handler for obs term 'base_vel_ref'
    #    The teacher's reference observations were added without registering reflection
    #    rules in motion_gen.sagittal_reflector.SagittalReflectionConfig, and the
    #    augmentation deliberately refuses to run rather than silently identity-copying the
    #    mirrored half of the batch (which would poison training). Pre-dates the repo
    #    slim-down: the same error reproduces on the old G1-teacher-clf-sym at 681a624.
    #    The unhandled terms are now `reference_window` and `command_window` (the merged
    #    look-ahead that replaced base_vel_ref / ee_ref / future_base_*). reference_window
    #    is the harder one: it is [step0(D), step1(D), ...] where each step packs
    #    base_pose(6) + base_vel(6) + joint_pos(nj) + joint_vel(nj) + ee_pos(3*n_ee), so
    #    the handler must reflect per BLOCK within each step -- reuse
    #    mdp.observations.reference_window_layout for the widths, and note the joint block
    #    is in ROBOT joint order (so the existing joint reflection permutation applies
    #    directly, unlike the clip-output-ordered joint_pos_ref).
    #    To revive: add both terms to SagittalReflectionConfig, extend
    #    tests/test_sagittal_reflector.py, then smoke-train.
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
    # Teacher on TERRAIN + two self-scanning depth cameras (torso-mounted).
    # Same terrain / clip tracking as above, but the policy sees the ground ONLY
    # through the cameras -- depth REPLACES the height scan, which is what the
    # hardware can actually deploy.  The critic keeps the scan (privileged).
    ## =========================================
    gym.register(
        id="G1-teacher-clf-terrain-depth",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1TeacherTerrainDepthEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-terrain-depth-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1TeacherTerrainDepthEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-terrain-depth-image",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1TeacherTerrainDepthImageEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-terrain-depth-image-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1TeacherTerrainDepthImageEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
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
            # Play-only runner: no privileged critic group, and its weights are not loaded,
            # so any teacher checkpoint drives this task regardless of tracking_mode.
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FmTrackerPlayRunnerCfg",
        },
    )
    _registered = True
