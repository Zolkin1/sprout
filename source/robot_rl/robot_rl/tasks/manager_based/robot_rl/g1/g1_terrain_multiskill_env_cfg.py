# Copyright 2026 Zachary Olkin. All rights reserved.

import copy

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..mdp.commands.multiskill_velocity_commands_cfg import VelocityBucketCfg

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import G1ClfTrackingSceneCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg, G1MultiSkillObservationCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import StairBlockCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.config.stair_dim_scan import scan_stair_step_dims
from robot_rl.tasks.manager_based.robot_rl.terrains.config.terrain_cfgs import FLAT_STAIR_MULTISKILL_CFG, FLAT_MULTISKILL_CFG
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_composite_importer_cfg import MetaCompositeTerrainImporterCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_terrain_generator_cfg import MetaTerrainGeneratorCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg import (
    MeshXStairsDownTerrainCfg,
    MeshXStairsUpTerrainCfg,
)


# Body-contact sensors (declared on the base scene cfg) that should terminate
# the episode when they touch the terrain.  Feet are intentionally excluded.
_TERRAIN_CONTACT_TERMINATION_SENSORS = (
    "torso_contact",
    "left_thigh_contact",
    "right_thigh_contact",
    "left_elbow_contact",
    "right_elbow_contact",
)


def _terrain_has_stairs(generator: MetaTerrainGeneratorCfg) -> bool:
    """Return whether a terrain generator contains any stairs.

    Inspects every sub-terrain for either a pure stair terrain type
    (:class:`MeshXStairsUpTerrainCfg` / :class:`MeshXStairsDownTerrainCfg`) or a
    composite sub-terrain whose explicit ``blocks`` / randomized ``choices``
    include a :class:`StairBlockCfg`.  Used to gate the stair-dimension scan so
    flat-only terrains don't require a ``stair_up`` folder in the trajectory
    library.

    Args:
        generator: The terrain generator config to inspect.

    Returns:
        ``True`` if any sub-terrain is or contains stairs, otherwise ``False``.
    """
    for sub in generator.sub_terrains.values():
        if isinstance(sub, (MeshXStairsUpTerrainCfg, MeshXStairsDownTerrainCfg)):
            return True
        # Composite sub-terrains expose their blocks either as a fixed
        # ``blocks`` list or as weighted ``choices`` (each holding a ``cfg``).
        blocks = list(getattr(sub, "blocks", None) or [])
        blocks += [choice.cfg for choice in getattr(sub, "choices", None) or []]
        if any(isinstance(block, StairBlockCfg) for block in blocks):
            return True
    return False


@configclass
class G1TerrainMultiskillSceneCfg(G1ClfTrackingSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.75, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class G1TerrainMultiskillObservationCfg(G1MultiSkillObservationCfg):

    @configclass
    class PolicyCfg(G1MultiSkillObservationCfg.PolicyCfg):
        sin_phase = None
        cos_phase = None

        # Trying the actor with the traj ref and traj actual
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

    @configclass
    class CriticCfg(G1MultiSkillObservationCfg.CriticCfg):
        sin_phase = None
        cos_phase = None

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

    @configclass
    class StudentCfg(PolicyCfg):
        ref_traj = None
        act_traj = None
        ref_traj_vel = None
        act_traj_vel = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    student: StudentCfg = StudentCfg()

@configclass
class G1TerrainMultiskillCurriculumCfg:
    """Curriculum terms for the terrain multiskill MDP."""

    # Anneal the frame-drift termination's post-transition grace window upward so
    # that post-skill-transition transients are penalized progressively less and
    # eventually (once grace covers a full episode) frame drift never fires after a
    # transition. Steady-state / fresh-episode frame drift stays active.
    frame_drift_grace = CurrTerm(
        func=mdp.grace_period_curriculum,
        params={
            "term_name": "frame_drift",
            "increment_s": 2.0,
            "interval_iters": 3000,
            "num_steps_per_env": 24,  # keep in sync with agents/rsl_rl_ppo_cfg.py
            # max_grace_period_s=None -> caps at the episode length (20 s)
        },
    )


@configclass
class G1TerrainMultiskillCLFEnvCfg(G1MultiSkillCLFEnvCfg):
    """Config for the G1 for multiple skills with terrain."""

    scene: G1TerrainMultiskillSceneCfg = G1TerrainMultiskillSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1TerrainMultiskillObservationCfg = G1TerrainMultiskillObservationCfg()
    curriculum: G1TerrainMultiskillCurriculumCfg = G1TerrainMultiskillCurriculumCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        # to HuggingFace is done (walk_forward / walk_backwards / running /
        # stair_up / standing subfolders).
        self.commands.traj_ref.path = "trajectories/retargeted/2026-06-16_16-30-26_merged" #"trajectories/retargeted/2026-06-13_15-10-58_merged" #"trajectories/retargeted/2026-06-12_merged"
        # Cross-fade window after a skill commit — half a cycle of the new
        # trajectory's phase.  Default is 1.0 (a full cycle).
        self.commands.traj_ref.transition_blend_end_phi = 0.3 #0.5
        self.commands.traj_ref.skill_query_buffer = 0.2

        # Global velocity ranges.  lin_vel_y / ang_vel_z are the OUTER clip on
        # the closed-loop PD overrides (y-position / heading control); the
        # per-skill clamp in ``MultiskillVelocityTrackingCommand`` is what
        # keeps stair_up / standing envs at zero lateral / yaw.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 3.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-3.14,3.14)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)

        self.commands.base_velocity.max_acc_frac = 1.0
        self.commands.base_velocity.max_acc = 1.0

        # Force every env fully open-loop for now: no y-position / heading PD
        # overrides on the sampled velocity. The base multiskill cfg uses
        # rel_closed_loop=0.55 / rel_closed_loop_yaw=0.25 / rel_open_loop=0.2;
        # restore those four rel_* values to re-enable the PD. (The four must
        # still sum to 1.0 — validated in VelocityTrackingCommand.__init__.)
        self.commands.base_velocity.rel_open_loop = 1.0
        self.commands.base_velocity.rel_closed_loop = 0.0
        self.commands.base_velocity.rel_closed_loop_yaw = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Mid-episode skill transitions: on each velocity resample an env may
        # switch to another skill from its current cell's distribution (prob
        # ``skill_transition_prob``; the cell probs are renormalised to exclude
        # the current skill).  ``reset_for_episode`` still draws the per-episode
        # starting skill independently from the spawn cell.  0.6 matches the
        # prior walk/run multiskill runs; set to 0.0 to lock one skill per
        # episode (see MultiskillVelocityTrackingCommand._sample_skill).
        self.commands.base_velocity.skill_transition_prob = 0.6

        # Per-skill velocity ranges.  Keys MUST equal the terrain importer's
        # ``skill_list`` exactly (validated at construction).  The
        # trajectory cmd owns this dict; the velocity cmd reads it via
        # the trajectory cmd's cfg at init.
        # Ranges mirror the gait-library grids (walk_forward/walk_backwards/
        # running library configs); stair_up keeps lateral / yaw at (0, 0)
        # for now even though the stair library covers vel_y ±0.4.
        self.commands.traj_ref.velocity_buckets = {
            "standing":       VelocityBucketCfg(lin_vel_x=(-0.1, 0.1)),
            "walk_forward":   VelocityBucketCfg(
                lin_vel_x=(0.1, 1.5), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-0.8, 0.8),
            ),
            # TEMP: backwards walking disabled for now. Re-enable together with
            # the skill_list entry below and remove the skill_probs strip in the
            # terrain block.
            # "walk_backwards": VelocityBucketCfg(
            #     lin_vel_x=(-0.9, -0.1), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-0.8, 0.8),
            # ),
            "running":        VelocityBucketCfg(
                lin_vel_x=(1.7, 3.6), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-0.8, 0.8),
            ),
            # "stair_up":       VelocityBucketCfg(lin_vel_x=(0.2, 0.8)),
        }


        ##
        # Rewards
        ##

        ## CLF Based
        self.rewards.clf_reward.params["max_eta_err"] = 5.0
        self.rewards.clf_decreasing_condition.params["eta_max"] = 5.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 10.0

        # Velocity Tracking
        self.rewards.xy_vel.params["std"] = 0.5
        self.rewards.yaw_vel.params["std"] = 0.5

        ##
        # Events
        ##
        # Update push forces
        # TODO: Should probably make it so i only get pushes on flat ground
        # TODO: Need to be careful about pushes with the frame drift termination
        self.events.push_robot = None


        ##
        # Terrain
        ##
        # terrain_generator = copy.deepcopy(FLAT_STAIR_MULTISKILL_CFG)
        terrain_generator = copy.deepcopy(FLAT_MULTISKILL_CFG)

        # TEMP: backwards walking disabled for now (walk_backwards commented out
        # of skill_list / velocity_buckets above). The shared _FLAT_SKILL_PROBS
        # (terrains/config/terrain_cfgs.py) still lists walk_backwards, and the
        # meta importer raises KeyError on any skill_probs entry not in
        # skill_list, so strip it from this deepcopy's flat cells + border and
        # renormalize to 1.0. To restore: delete this block.
        def _drop_skill_renorm(probs: dict, skill: str) -> dict:
            """Remove ``skill`` from a skill_probs dict and renormalize to 1.0."""
            kept = {k: v for k, v in probs.items() if k != skill}
            total = sum(kept.values())
            return {k: v / total for k, v in kept.items()}

        if terrain_generator.border_skill_probs:
            terrain_generator.border_skill_probs = _drop_skill_renorm(
                terrain_generator.border_skill_probs, "walk_backwards"
            )
        for _sub in terrain_generator.sub_terrains.values():
            sub_probs = getattr(_sub, "skill_probs", None)
            if sub_probs and "walk_backwards" in sub_probs:
                _sub.skill_probs = _drop_skill_renorm(sub_probs, "walk_backwards")

        # Stair sizes come from the gait library so every stair span has a
        # matching stair_up trajectory (see the per-env terrain-dim filter in
        # ``MultiSkillManager``).  Skip for flat-only terrains: the scan needs a
        # ``stair_up`` folder in the trajectory library, which a flat-only gait
        # library doesn't have.
        if _terrain_has_stairs(terrain_generator):
            step_dims = scan_stair_step_dims(
                self.commands.traj_ref.path, hf_repo=self.commands.traj_ref.hf_repo
            )
            if "pure_stair_up" in terrain_generator.sub_terrains:
                pure_stair_up = terrain_generator.sub_terrains["pure_stair_up"]
                pure_stair_up.step_dim_options = step_dims
                # Spawn on tread 0 for every depth in the grid (origin tread
                # index is ``distance // step_depth`` — don't tie it to one
                # nominal depth).
                pure_stair_up.origin_distance_from_back = 0.1
            if "flat_stair_up" in terrain_generator.sub_terrains:
                for choice in terrain_generator.sub_terrains["flat_stair_up"].choices:
                    if isinstance(choice.cfg, StairBlockCfg):
                        choice.cfg.step_dim_options = step_dims

        base_terrain = self.scene.terrain
        self.scene.terrain = MetaCompositeTerrainImporterCfg(
            prim_path=base_terrain.prim_path,
            terrain_type="generator",
            terrain_generator=terrain_generator,
            max_init_terrain_level=10,
            collision_group=base_terrain.collision_group,
            physics_material=base_terrain.physics_material,
            visual_material=base_terrain.visual_material,
            debug_vis=base_terrain.debug_vis,
            # skill_list=["stair_up", "walk_forward", "walk_backwards", "running", "standing"],
            # TEMP: backwards walking disabled for now — add "walk_backwards"
            # back here (and re-enable its velocity bucket above + drop the
            # skill_probs strip in the terrain block) to restore it.
            # skill_list=["walk_forward", "walk_backwards", "running", "standing"],
            skill_list=["walk_forward", "running", "standing"],
        )

        self.commands.base_velocity.debug_vis = False

        ##
        # Terminations
        ##

        self.terminations.frame_drift = DoneTerm(
          func=mdp.frame_deviation_from_reference,
          params={
              "debug": False,
              "cmd_name": "traj_ref",
              "frame_names": ["pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link"],
              "max_frac": 0.3,
              "min_dist": 0.1,
              "grace_period_s": 100.0,
          },
        )

        # Face-plant / fallen-robot detector.  Not gated by the trajectory
        # grace window — catches falls even while ``frame_drift`` is being
        # suppressed by a post-skill-transition grace period.
        self.terminations.pelvis_upright = DoneTerm(
            func=mdp.base_orientation_from_upright,
            params={"roll_limit_deg": 50.0, "pitch_limit_deg": 50.0},
        )

        # Per-skill breakdown of frame_drift for logging
        # (wandb: Episode_Termination/frame_drift_<skill>). Generated from the
        # terrain's skill_list so new skills are picked up automatically. Each
        # term is an exact subset of frame_drift, so training dynamics are
        # unchanged. Registered after frame_drift so it computes first and the
        # subset terms can read its done via the termination manager.
        for skill in self.scene.terrain.skill_list:
            setattr(
                self.terminations,
                f"frame_drift_{skill}",
                DoneTerm(
                    func=mdp.frame_drift_in_skill,
                    params={"cmd_name": "traj_ref", "skill_name": skill},
                ),
            )

        # Frame-drift terminations that land within ``window_s`` after the grace
        # period ends — a proxy for "the transition itself failed (just late)".
        # Logging only: an exact subset of frame_drift, so training is unchanged.
        # wandb: Episode_Termination/frame_drift_post_grace. ``grace_period_s`` is
        # read from frame_drift's params automatically. Registered after
        # frame_drift so it computes first and can read its done.
        self.terminations.frame_drift_post_grace = DoneTerm(
            func=mdp.frame_drift_post_grace,
            params={"cmd_name": "traj_ref", "window_s": 0.25},
        )

        # Per-skill breakdown of the post-grace metric
        # (wandb: Episode_Termination/frame_drift_post_grace_<skill>). Reuses the
        # generic frame_drift_in_skill masker by pointing base_term at the
        # post-grace term. Registered after it so it computes first.
        for skill in self.scene.terrain.skill_list:
            setattr(
                self.terminations,
                f"frame_drift_post_grace_{skill}",
                DoneTerm(
                    func=mdp.frame_drift_in_skill,
                    params={
                        "cmd_name": "traj_ref",
                        "skill_name": skill,
                        "base_term": "frame_drift_post_grace",
                    },
                ),
            )

@configclass
class G1TerrainMultiskillCLFDistillationEnvCfg(G1TerrainMultiskillCLFEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.commands.base_velocity.resampling_time_range = (2.0, 6.0)
        # Per-skill velocity ranges are inherited from the parent
        # ``G1TerrainMultiskillCLFEnvCfg`` on ``self.commands.traj_ref``.
        # Override individual buckets here (e.g.
        # ``self.commands.traj_ref.velocity_buckets["running"].lin_vel_x = (2.0, 3.5)``)
        # if distillation needs tighter ranges.
        self.commands.base_velocity.max_acc = 1.0
        self.commands.base_velocity.max_acc_frac = 1.0

        # Inherit no-transition behavior from the parent (skill locked per
        # episode).  Bump back toward 0.6 here if distillation should see
        # mid-episode skill churn.
        self.commands.base_velocity.skill_transition_prob = 0.0

        self.commands.traj_ref.contact_gate_window_frac = 0.1


@configclass
class G1TerrainMultiskillCLFTransformerDistillationEnvCfg(G1TerrainMultiskillCLFDistillationEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set uniform history for all policy obs terms (each timestep = 1 token)
        history_length = 50
        self.observations.student.base_ang_vel.history_length = history_length
        self.observations.student.projected_gravity.history_length = history_length
        self.observations.student.velocity_commands.history_length = history_length
        self.observations.student.joint_pos.history_length = history_length
        self.observations.student.joint_vel.history_length = history_length
        self.observations.student.actions.history_length = history_length

@configclass
class G1TerrainMultiskillCLFEnvCfgPlay(G1TerrainMultiskillCLFEnvCfg):
    """Configuration for the G1 running gait library play environment."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_velocity.debug_vis = False

        self.commands.traj_ref.debug_skill_log = False


        self.scene.num_envs = 2
        self.observations.policy.enable_corruption = False

        # Shrink the play terrain.  The actual grid is driven by the
        # ``terrain_generator`` (STAIR_WALK_CFG) — overriding fields on the
        # importer cfg has no effect.  Deepcopy first so we don't mutate the
        # shared module-level config used by the training env.
        self.scene.terrain.terrain_generator = copy.deepcopy(
            self.scene.terrain.terrain_generator
        )
        self.scene.terrain.terrain_generator.num_rows = self.scene.num_envs
        self.scene.terrain.terrain_generator.num_cols = 4
        # The training cfg's 7 buffer columns don't fit in 4 columns — drop
        # the buffer for the small play grid.
        self.scene.terrain.terrain_generator.inter_column_borders = []
        # self.scene.terrain.terrain_generator.sub_terrains["pure_flat"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["pure_stair_up"].proportion = 0.0
        # self.scene.terrain.terrain_generator.sub_terrains["flat_stair_up"].proportion = 1.0


        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None
        # self.events.joint_friction_params = None  # Can't use this - friction goes out of distribution

        self.commands.traj_ref.debug_vis = True
        self.commands.base_velocity.debug_vis = True #False
        self.scene.height_scanner.debug_vis = True
        self.scene.terrain.terrain_generator.debug_vis = True
        # Per-block outline rectangles around every FlatBlock / StairBlock
        # in the composite cells. Drawn by MetaCompositeTerrainImporter.
        # set_debug_vis. Colors come from _DEFAULT_BLOCK_OUTLINE_COLORS in
        # meta_composite_importer.py — override here via
        # ``self.scene.terrain.block_outline_colors = {...}`` if needed.
        self.scene.terrain.debug_vis = True