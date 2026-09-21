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

**One task breaks the "a clip per tile" rule**, and it is the last one registered:
``G1-teacher-clf-terrain-depth-finetune-freeform`` takes a MANIFEST-ONLY library -- terrain
and no clips at all -- because its reference is predicted from the cameras rather than read
off disk. ``--clip`` still states it, and still states it for both the importer and the
command, so nothing about how a library is passed changes.
"""

import gymnasium as gym

from . import agents
# For the adjuster task's entry point. Importing the PACKAGE is safe pre-AppLauncher: its
# ``__init__`` re-exports only the cfg, never the env class, which is the whole reason that
# split exists (see ``envs/__init__.py``).
from .. import envs

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
    #    is the harder one: it is [sample0(D), sample1(D), ...] over the 13
    #    REFERENCE_OFFSETS, where each sample packs
    #    base_pose(9) + base_vel(6) + joint_pos(nj) + joint_vel(nj) + ee_pos(3*n_ee), so
    #    -- and base_pose's orientation half is a 6D rotation, which does NOT mirror as a
    #    sign pattern (see motion_gen.features._rot6d for the one that does), so
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

    ## =========================================
    # RL FINE-TUNE: the depth teacher, shown its own generator's reference window while
    # every graded quantity stays the clip's. Resume a trained teacher into it and pass
    # --fm_checkpoint. See g1_finetune_env_cfg for why the reward needs no re-pointing.
    ## =========================================
    gym.register(
        id="G1-teacher-clf-terrain-depth-finetune",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.g1_finetune_env_cfg:G1FinetuneTerrainDepthEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:FinetunePPORunnerCfg"
            ),
        },
    )

    ## =========================================
    # The SAME fine-tune on FREE-FORM terrain: flat extents, scattered boxes and straight
    # staircases that no clip was ever generated for. Under ``reference_source="fm"`` the
    # reference is predicted from the cameras, so the tile-matches-one-motion constraint that
    # shapes every other library here is gone. ``--clip`` therefore takes a MANIFEST-ONLY
    # ``.h5`` (zero clip datasets) from ``scripts/rsl_rl/make_freeform_terrain.py``: the spawn
    # comes from each section's route block, the velocity command is generated with a steering
    # controller, and the episode is bounded by the section's goal band instead of by a clip
    # running out. See g1_finetune_env_cfg for the three distribution changes that buys.
    ## =========================================
    gym.register(
        id="G1-teacher-clf-terrain-depth-finetune-freeform",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.g1_finetune_env_cfg:G1FinetuneFreeformEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:FinetunePPORunnerCfg"
            ),
        },
    )

    # The WALL-FIELD fine-tune: enclosed rooms with walls to be IGNORED rather than climbed,
    # and a goal the robot has to STOP inside for several seconds. The only task that switches
    # the free-form approach controller on (range-driven vx), which is what makes stopping a
    # behaviour the reward can pay for. Takes `--clip clips/ft_walls_train.h5` from
    # `scripts/rsl_rl/make_walls_terrain.py`; see `motion_gen.terrain.wall_field`.
    ## =========================================
    gym.register(
        id="G1-teacher-clf-terrain-depth-finetune-walls",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.g1_finetune_env_cfg:G1WallsFinetuneEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:FinetunePPORunnerCfg"
            ),
        },
    )

    # The same fine-tune pinned to the FIXED two-box course, and with the far camera back in
    # the POLICY observation so an existing (two-camera) tracker checkpoint can run it. Takes
    # `--clip clips/box2_train.h5` or `clips/box2_holdout.h5` from
    # `scripts/rsl_rl/make_box2_terrain.py`. See `motion_gen.terrain.box2` for why this course.
    ## =========================================
    gym.register(
        id="G1-box2-finetune",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.g1_finetune_env_cfg:G1Box2FinetuneEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:FinetunePPORunnerCfg"
            ),
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

    ## =========================================
    # The DEPLOY-ISOLATION teacher: no exteroception at all, on flat ground.
    # Everything the depth teacher has except the cameras and the terrain, so a
    # hardware failure can be attributed to perception or to tracking rather than
    # to "the policy". Pair with a flat-only clip library.
    ## =========================================
    gym.register(
        id="G1-teacher-clf-flat-blind",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_flat_blind_env_cfg:G1TeacherFlatBlindEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-teacher-clf-flat-blind-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.g1_flat_blind_env_cfg:G1TeacherFlatBlindEnvCfgPlay",
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

    # Same tracker, DEPTH exteroception on both sides -- the depth teacher's policy branch
    # plus a depth-trained generator. Needed because FmTrackingCommand feeds the sensor the
    # CHECKPOINT names, so a depth FM cannot run in the height-scan task above (no cameras in
    # its scene), and depth is what FlowMatchingConfig trains by default.
    gym.register(
        id="G1-tracker-clf-fm-terrain-depth-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.g1_tracker_env_cfg:G1FmTrackerTerrainDepthPlayEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FmTrackerPlayRunnerCfg",
        },
    )

    ## =========================================
    ## The ADJUSTER
    ## =========================================
    # Trains a residual on the generator's predicted window while BOTH the generator and the
    # tracker stay frozen. The RL agent acts once per re-plan (0.24 s), so the entry point is
    # not the stock env: `AdjusterEnv.step` runs a whole interval with the tracker inside it.
    #
    # Requires four artifacts and refuses to guess any of them:
    #   --clip                (library + terrain; use --clip_device cpu)
    #   --fm_checkpoint       (a PREFIX-attention, measured-history generator)
    #   --tracker_run         (the frozen tracker's run dir)
    #   --tracking_spec / --goal_regions   (the two FITTED specs)
    #
    # Run it with scripts/rsl_rl/train.py, which wires the four adjuster flags and swaps
    # in RslRlAdjusterVecEnvWrapper on `hasattr(env, 'replan_period')`. The wrapper has to
    # override `num_actions`, because the action manager holds the TRACKER's joints.
    gym.register(
        id="G1-adjuster-fm-terrain-depth",
        # The first non-`ManagerBasedRLEnv` entry point in this repo. Resolved by gym when the
        # env is MADE, i.e. after AppLauncher -- which is what lets the concrete env class
        # import `ManagerBasedRLEnv` (and therefore reach pxr) without breaking the
        # pre-AppLauncher cfg compose. See envs/__init__.py.
        entry_point=f"{envs.__name__}.adjuster_env:AdjusterEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_adjuster_env_cfg:G1AdjusterEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AdjusterPPORunnerCfg",
        },
    )
    # Watch an adjuster checkpoint on a small tile set, with the terrain ray casts DRAWN.
    # `load_critic = False` on its runner cfg is what lets a checkpoint from before a
    # critic-group change still load: play never evaluates the value function, and this
    # task's critic width has moved while its actor has not.
    gym.register(
        id="G1-adjuster-fm-terrain-depth-play",
        entry_point=f"{envs.__name__}.adjuster_env:AdjusterEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_adjuster_env_cfg:G1AdjusterEnvCfgPlay",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:AdjusterPlayPPORunnerCfg"
            ),
        },
    )

    # The PPO arm of the algorithm ladder: the SAME adjuster, on the MDP `configs/finetune/
    # boxes.yaml` names, so a PPO number drops into the same forest-plot block as awr / fbc /
    # arrived / bc. A separate id rather than a flag on the one above, because the two differ
    # in terrain, reward and terminations -- everything a task id exists to name -- and the
    # clip-terrain arm stays runnable.
    gym.register(
        id="G1-adjuster-ft-freeform",
        entry_point=f"{envs.__name__}.adjuster_env:AdjusterEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_adjuster_env_cfg:G1AdjusterFreeformEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:AdjusterFreeformPPORunnerCfg"
            ),
        },
    )

    # Watch the PPO arm. `play.py` picks the interval-stepping wrapper off the env's
    # `replan_period`, so the same script drives it as drives every other task.
    gym.register(
        id="G1-adjuster-ft-freeform-play",
        entry_point=f"{envs.__name__}.adjuster_env:AdjusterEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.g1_adjuster_env_cfg:G1AdjusterFreeformEnvCfgPlay"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:AdjusterFreeformPlayPPORunnerCfg"
            ),
        },
    )

    _registered = True
