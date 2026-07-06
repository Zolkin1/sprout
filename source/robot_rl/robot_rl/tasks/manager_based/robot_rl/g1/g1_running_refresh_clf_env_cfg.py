# Copyright 2026 Zachary Olkin. All rights reserved.

"""Refreshed running config reproducing the conference-submission running task.

Rebuilds the intent of the old ``G1RunningGaitLibraryEnvCfg`` (task
``G1-running-hzd`` at the ``ICRA_2026_submission`` tag) on the current runtime.

Built on the **batched multiskill** base (:class:`G1MultiSkillCLFEnvCfg`) with a
single ``running`` skill on flat ground -- the single-skill
``TrajectoryCommandCfg(manager_type="library")`` path is broken on the current
setup (``reset_on_reference`` requires the multiskill manager API), so this rides
the maintained path that actually runs. The multiskill framework degenerates
cleanly to one skill: skill id is pinned to 0, cross-fade/terrain-transition
logic stays dormant, and the flat meta-terrain is semantically the same flat
ground as the old ``terrain_type="plane"`` (it just also advertises the ``skill``
machinery the command expects).

Carried over faithfully from the submission: the domain randomization (see
``__post_init__``) and the running-specific flight-contact penalty + its ramping
curriculum (revived in ``mdp/rewards/rewards.py`` + ``mdp/curriculums``).
"""

import copy

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl import mdp

from .g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg
from ..mdp.commands.multiskill_velocity_commands_cfg import VelocityBucketCfg
from ..terrains.meta_terrain_generator_cfg import MetaTerrainGeneratorCfg
from ..terrains.meta_terrain_importer_cfg import MetaTerrainImporterCfg
from ..terrains.trimesh.flat_cfg import MeshFlatTerrainCfg


# Flat-only meta-terrain that advertises a single ``running`` skill on every
# cell.  Semantically "flat ground everywhere" (like the old
# ``terrain_type="plane"``) but exposes the ``skill_probs``/``skill_list``
# machinery the multiskill velocity command requires.  The flat cell's default
# ``skill_probs`` is overridden to 100% running so the command only ever samples
# the running skill.
RUNNING_FLAT_MULTISKILL_CFG = MetaTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=72.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshFlatTerrainCfg(proportion=1.0, skill_probs={"running": 1.0}),
    },
)


@configclass
class G1RunningRefreshCurriculumCfg:
    """Curriculum for the refreshed running config.

    Ramps the ``flight_contact_penalty`` reward term's ``weight_scalar`` from 0
    up to ``max_weight`` (submission values), so the flight-contact penalty
    turns on gradually during training.
    """

    contact_penalty_curriculum = CurrTerm(
        func=mdp.contact_curriculum,
        params={"update_interval": 20000, "max_weight": 1.0, "update_amnt": 0.1},
    )


@configclass
class G1RunningRefreshCLFEnvCfg(G1MultiSkillCLFEnvCfg):
    """Running config reproducing the submission task on the current setup."""

    curriculum: G1RunningRefreshCurriculumCfg = G1RunningRefreshCurriculumCfg()

    def __post_init__(self):
        # Post init of parent (multiskill base -> clf tracking base).
        super().__post_init__()

        ##
        # Commands -- single running skill
        ##
        # FILL IN: point at the regenerated submission running trajectory once it
        # is ready. The batched loader wants a folder with a ``running/``
        # subfolder of velocity-indexed running YAMLs (subfolder name must match
        # the ``skill_list``/bucket key ``running``). Smoke-tested here against
        # the walk-run merged library, which already contains a ``running`` skill.
        self.commands.traj_ref.path = "trajectories/retargeted/2026-05-11_12-34-50_merged"

        # Single skill: only running. lin_vel_x is the submission running sweep;
        # adjust to the regenerated trajectory's actual speed coverage.
        self.commands.traj_ref.velocity_buckets = {
            "running": VelocityBucketCfg(lin_vel_x=(1.1, 3.0)),
        }
        # Contact gate on (gait phase-alignment). Non-None is required by the
        # default ``gate_skill_change_on_contact=True`` consistency assert; with
        # one skill the pending-skill-change queue simply never has entries.
        self.commands.traj_ref.contact_gate_window_frac = 0.1

        # Base-velocity envelope (submission running values). Keep
        # ``skill_transition_prob=None`` (single skill).
        self.commands.base_velocity.ranges.lin_vel_x = (1.1, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.max_acc = 1.0
        self.commands.base_velocity.max_acc_frac = 1.0

        ##
        # Rewards
        ##
        # CLF thresholds at the current (new-formulation) running scale; the
        # submission's 0.25/0.3 were on the old CLF formulation and do not map 1:1.
        self.rewards.clf_reward.params["max_eta_err"] = 12.0
        self.rewards.clf_decreasing_condition.params["eta_max"] = 12.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 24.0

        self.rewards.xy_vel.params["std"] = 0.75
        self.rewards.yaw_vel.params["std"] = 0.75

        # Running-specific flight-contact penalty (revived). Starts inert at
        # weight_scalar=0.0 and is ramped by contact_penalty_curriculum. The term
        # MUST be named ``flight_contact_penalty`` (contact_curriculum looks it up
        # by that name). Depends only on ``traj_ref`` (get_contact_state /
        # manager.phase) and ``base_velocity`` -- both supplied by the batched cmd.
        self.rewards.flight_contact_penalty = RewTerm(
            func=mdp.flight_contact_penalty,
            weight=-3.0,
            params={
                "command_name": "traj_ref",
                "base_vel_cmd": "base_velocity",
                "sensor_cfgs": [
                    SceneEntityCfg("left_foot_contact"),
                    SceneEntityCfg("right_foot_contact"),
                ],
                "weight_scalar": 0.0,
                "start_vel": 2.0,
            },
        )
        # ``torque_lims`` (mdp.torque_limits, weight -1.0) already lives in the base.

        ##
        # Events -- domain randomization matched to the submission
        ##
        # Actuator gains: additive absolute offsets (submission), not multiplicative.
        self.events.gain_randomization.params["stiffness_distribution_params"] = (-10.0, 10.0)
        self.events.gain_randomization.params["damping_distribution_params"] = (-2.0, 2.0)
        self.events.gain_randomization.params["operation"] = "add"

        # Torso mass: additive +-5 kg on waist_yaw_link (submission).
        self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.add_base_mass.params["operation"] = "add"

        # COM randomization on the torso (waist_yaw_link).
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")

        # Ground contact friction ranges (submission).
        self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.25, 1.25)
        self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.25, 1.25)
        # restitution_range already (0.0, 0.2)

        # Push forces (submission).
        self.events.push_robot.params["velocity_range"] = {"x": (-0.75, 0.75), "y": (-0.75, 0.75)}

        # The submission disabled the external force/torque; the joint-default-pos
        # calibration event did not exist then.
        self.events.base_external_force_torque = None
        self.events.add_joint_default_pos = None
        # NOTE: joint friction stays on the current multi-friction term -- Isaac
        # v3 removed the old dynamic/viscous friction knobs the submission used,
        # so its +-5% friction scale is not reproducible as-is.

        ##
        # Terrain -- flat meta-terrain exposing the running skill
        ##
        base_terrain = self.scene.terrain
        self.scene.terrain = MetaTerrainImporterCfg(
            prim_path=base_terrain.prim_path,
            terrain_type="generator",
            terrain_generator=RUNNING_FLAT_MULTISKILL_CFG,
            max_init_terrain_level=0,
            collision_group=base_terrain.collision_group,
            physics_material=base_terrain.physics_material,
            visual_material=base_terrain.visual_material,
            debug_vis=base_terrain.debug_vis,
            skill_list=["running"],
        )
        self.scene.height_scanner = None


@configclass
class G1RunningRefreshCLFEnvCfgPlay(G1RunningRefreshCLFEnvCfg):
    """Play/eval variant: few envs, no corruption, no domain randomization."""

    def __post_init__(self):
        super().__post_init__()

        # Fixed play speed (within the running bucket).
        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.debug_vis = False
        self.commands.traj_ref.debug_vis = False

        self.episode_length_s = 10.0

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Shrink the play terrain. The grid is driven by the terrain_generator;
        # deepcopy first so we don't mutate the shared module-level generator.
        self.scene.terrain.terrain_generator = copy.deepcopy(self.scene.terrain.terrain_generator)
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2

        # Disable domain randomization for eval.
        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None
