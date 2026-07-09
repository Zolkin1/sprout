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
from .clf_weights import RUNNING_Q_weights
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
        # Single-skill folder (flat, no subfolder) -> the multiskill loader names
        # the skill "default"; skill_probs/skill_list/velocity_buckets must agree.
        "flat": MeshFlatTerrainCfg(proportion=1.0, skill_probs={"default": 1.0}),
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
        # Regenerated submission running trajectory (21 half-periodic solutions,
        # conditioner vel_x in [1.38, 3.29] m/s). Flat folder of YAMLs -> the
        # multiskill loader names the single skill "default".
        self.commands.traj_ref.path = "trajectories/running/2026-07-06_11-46-52_running_config_icra_port"

        # Single skill ("default"); lin_vel_x = the trajectory's actual speed
        # coverage so commands map to an in-distribution trajectory.
        self.commands.traj_ref.velocity_buckets = {
            "default": VelocityBucketCfg(lin_vel_x=(1.38, 3.29)),
        }
        # Feet-dominant CLF weighting (ICRA-like): prioritize foot Cartesian
        # tracking far above the joints/arms. See clf_weights.RUNNING_Q_weights.
        self.commands.traj_ref.Q_weights = RUNNING_Q_weights
        # Time-based reference anchor (match the ICRA anchor). Disable the contact
        # gate so the phase advances purely on the episode clock and the world
        # anchor re-anchors at time-based domain boundaries -- no physical-contact
        # phase resync, i.e. the open-loop time-based reference ICRA used.
        self.commands.traj_ref.contact_gate_window_frac = None
        self.commands.traj_ref.gate_skill_change_on_contact = False

        # Base-velocity envelope matched to the trajectory's speed coverage. Keep
        # ``skill_transition_prob=None`` (single skill).
        self.commands.base_velocity.ranges.lin_vel_x = (1.38, 3.29)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.max_acc = 1.0
        self.commands.base_velocity.max_acc_frac = 1.0
        # Disable command debug viz for (headless) training -- the multiskill base
        # leaves base_velocity.debug_vis=True, which spawns the velocity-goal
        # point-instancer markers and trips omni/fabric even headless.
        self.commands.base_velocity.debug_vis = False
        self.commands.traj_ref.debug_vis = False

        ##
        # Rewards
        ##
        # CLF error normalizer. Lowered 12 -> 5 to sharpen tracking: V_max scales
        # with max_eta_err^2, so this makes every output's tracking error ~5.8x
        # more penalized (converged clf_reward was 9.0/10 at 12, so there is
        # headroom). eta_max in the decreasing condition kept consistent at 5.
        # CLF error normalizer, retuned for the feet-dominant weights (Option 2):
        # with the bumped foot Q, lambda_max(P)~13 and V at good tracking ~12, so
        # max_eta_err=2.0 puts clf_reward ~0.8 at good tracking (demanding, with
        # headroom). ICRA's 0.3 does NOT transfer -- the 56-output set still
        # aggregates joints/wrists into V. Main knob to tune: raise toward 2.5-3
        # if the reward stays too low, lower toward 1.5 if it saturates high.
        self.rewards.clf_reward.params["max_eta_err"] = 2.0
        # CLF decreasing-condition: match the ICRA weight (-1.0) and alpha (0.5);
        # eta_max kept consistent with clf_reward's max_eta_err. eta_dot_max left
        # at 24 (per-step ΔV velocity scale -- see the CLF discrete-vs-continuous note).
        self.rewards.clf_decreasing_condition.weight = -1.0
        self.rewards.clf_decreasing_condition.params["alpha"] = 0.5
        self.rewards.clf_decreasing_condition.params["eta_max"] = 2.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 24.0

        # Match the ICRA action-rate penalty (current base default is -0.01).
        self.rewards.action_rate_l2.weight = -0.001

        # Match the tag: drop explicit velocity-tracking rewards (ICRA relied on
        # the CLF + holonomic constraints for base motion, not track_lin/ang_vel).
        self.rewards.xy_vel = None
        self.rewards.yaw_vel = None

        # Holonomic stance-foot constraints (revived from the ICRA HZD running
        # config): pin the stance foot's pose (weight 4.0) and velocity (weight
        # 2.0) to its anchor during single support. Adapted to the batched
        # command's ref_poses anchor + active stance-foot body world state.
        self.rewards.holonomic_constraint = RewTerm(
            func=mdp.holonomic_constraint,
            weight=4.0,
            params={"command_name": "traj_ref"},
        )
        self.rewards.holonomic_constraint_vel = RewTerm(
            func=mdp.holonomic_constraint_vel,
            weight=2.0,
            params={"command_name": "traj_ref"},
        )

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

        # Torso mass on waist_yaw_link: SCALE (0.8, 1.2) -- the submission's
        # effective DR (set in G1RoughLipEnvCfg), i.e. +-20% of torso mass, not
        # an additive +-5 kg.
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"

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
            skill_list=["default"],
        )
        self.scene.height_scanner = None


@configclass
class G1RunningRefreshCLFFastEnvCfg(G1RunningRefreshCLFEnvCfg):
    """Single-reference training: pin the command to ONE library speed so the
    conditioner always selects that solution, straight ahead -- trains on just
    that one running reference. ``fixed_speed`` picks which clip and must be a
    conditioner vel_x present in the library (~1.38-3.29 m/s); subclass and
    override it for another speed. Inherits the submission DR, the flight-contact
    penalty, and the feet-dominant CLF from the parent.
    """

    # Plain class var (no type annotation -> NOT a config field). Library top speed.
    fixed_speed = 3.29

    def __post_init__(self):
        super().__post_init__()

        # For the multiskill velocity cmd the sampled speed comes from the bucket,
        # so pin the bucket (and the envelope) to ``fixed_speed``; zero y/yaw so it
        # trains purely on the single straight-running reference.
        s = self.fixed_speed
        self.commands.traj_ref.velocity_buckets = {
            "default": VelocityBucketCfg(lin_vel_x=(s, s)),
        }
        self.commands.base_velocity.ranges.lin_vel_x = (s, s)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1RunningRefreshCLFMidEnvCfg(G1RunningRefreshCLFFastEnvCfg):
    """Single-reference training at a mid library speed (conditioner vel_x=2.3 m/s)."""

    fixed_speed = 2.3


@configclass
class G1RunningRefreshCLFEnvCfgPlay(G1RunningRefreshCLFEnvCfg):
    """Play/eval variant: few envs, no corruption, no domain randomization."""

    def __post_init__(self):
        super().__post_init__()

        # Fixed play speed. For the multiskill velocity command the SAMPLED speed
        # comes from the per-skill bucket (velocity_buckets), NOT ranges.lin_vel_x
        # -- so pin the bucket to hold a constant eval velocity. (2.5 m/s -> the
        # nearest trajectory, conditioner vel_x=2.48, is tracked.)
        self.commands.traj_ref.velocity_buckets = {
            "default": VelocityBucketCfg(lin_vel_x=(3.29, 3.29)),
        }
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
