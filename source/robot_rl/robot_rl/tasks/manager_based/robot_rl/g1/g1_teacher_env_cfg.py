# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 teacher policy: open-loop motion-clip tracking -- the shared, terrain-agnostic half.

Tracks pre-generated fixed motion clips (see ``scripts/rsl_rl/generate_mb_library.py``).
No velocity conditioner, no contact gates, no skill cross-fades.  Clips are stored in a
single clip-world frame and the CLF runs in the absolute world frame (the same frame as
the tracker task), so the policy's reference observations are base-relative /
heading-invariant while the privileged critic sees the raw world-frame tracking vectors.
A single base-pose-error termination covers failure.

:class:`G1TeacherEnvCfg` here is ABSTRACT -- it declares no ground.  The concrete env is
:mod:`g1_teacher_terrain_env_cfg`, whose scene meshes the clip library itself; a "flat"
run is just a library whose tiles are flat.  The split is kept (rather than folded into
one file) because the depth-camera work builds on both halves.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

from motion_gen.joint_names import G1_JOINT_NAMES

from robot_rl.tasks.manager_based.robot_rl import mdp

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_clf_tracking_base import (
    G1ClfTrackingEnvCfg,
    G1ClfTrackingEventsCfg,
    G1ClfTrackingRewardCfg,
    G1ClfTrackingSceneCfg,
)
from ..mdp.commands.traj_tracking.motion_sequence_cmd_cfg import MotionSequenceCommandCfg


# Flat TRIANGLE-MESH terrain at z=0.  A RayCaster cannot hit the infinite "plane"
# primitive, so any tool that needs a ray-castable flat ground builds one from this.
#
# No TASK uses it -- the flat-mesh record env that owned it was pruned in 0f26c06.  It
# lives on for the two depth diagnostics that import it from here
# (``scripts/rsl_rl/verify_depth_self_scan.py`` and ``compare_depth_sim2sim.py``), both of
# which want a minimal ray-castable scene rather than the terrain teacher's tiles.  It was
# deleted with the record env and the scripts were left dangling on it; restored rather
# than inlined twice.
FLAT_MESH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
    horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
    sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
)


##
# Observation window sizes -- one knob each, shared by the policy and the critic.
##

#: Proprioception window length in CONTROL steps, current step included.  10 = the
#: current step plus the 9 before it, i.e. 0.18 s of history at the 0.02 s control dt.
#: These are *consecutive* steps (Isaac Lab's per-term ``history_length`` buffers every
#: step); only the reference look-ahead below is decimated.  Set to 1 for no history.
PROPRIO_HISTORY_LENGTH = 10

#: Samples in the STRIDED tail of the reference window, and their spacing in control
#: steps.  5 * 0.02 s = 0.1 s, so 11 samples span 0.0 .. 1.0 s of look-ahead.  These two
#: only parameterize the tail; :data:`REFERENCE_OFFSETS` below is the actual contract.
REFERENCE_TAIL_NUM_STEPS = 11
REFERENCE_TAIL_STRIDE = 5

#: Clip-step offsets the reference window samples, current step (0) first.  THE window
#: contract: the observation, its noise/scale vectors, the episode truncation margin
#: (``mdp.clip_end_reached``), the flow-matching generator's required horizon and the
#: exported deploy spec are all derived from this one tuple.
#:
#: Two regimes, deliberately non-uniform:
#:
#: * ``0, 1, 2`` -- the current step and the two immediately after it.  The strided tail
#:   is spaced 0.1 s apart, which is far too coarse to show what happens NEXT; these give
#:   the policy the 50 Hz structure around the current step, so it can see where it is
#:   going rather than only where it will be in a tenth of a second.
#: * ``5, 10, ... 50`` -- the 0.1 s-spaced tail, out to 1.0 s.
#:
#: 13 samples.  ``max(REFERENCE_OFFSETS) + 1`` = 51 is the number of nodes the generator
#: must predict (``FlowMatchingConfig.future_len``); ``test_fm_tracking_lookahead``
#: asserts it, so adding a FURTHER offset without widening ``future_len`` fails there
#: rather than at the first play step.
REFERENCE_OFFSETS: tuple[int, ...] = (
    0, 1, 2, *mdp.strided_offsets(REFERENCE_TAIL_NUM_STEPS, REFERENCE_TAIL_STRIDE)[1:],
)

#: End-effector frames the reference window carries (clip output frames minus the base:
#: both ankles + both wrists).  Only needed to size the per-channel noise vector below;
#: the observation itself takes the list from the command's ``resolve_ee_frames``.  A
#: mismatch fails loudly when the noise is broadcast against the observation.
REFERENCE_NUM_EE_FRAMES = 4

#: Which reference blocks the window carries.  The full set is
#: ``mdp.REFERENCE_WINDOW_BLOCKS`` = base_pose, base_vel, joint_pos, joint_vel, ee_pos.
#:
#: Currently REDUCED to pose + velocity + joint angles: the reference joint VELOCITIES and
#: the ee positions are dropped, to test whether the policy needs them at all.  The ee block
#: is largely redundant with joint_pos (forward kinematics of the same posture) and the
#: velocity block is the noisiest thing the FM has to predict at deploy, so if tracking
#: holds without them the reference gets cheaper to produce and cheaper to observe.
#:
#: One knob: the noise vector, the scale vector and the observation itself are all built
#: from this, so they cannot disagree.  Restore by setting it to ``None``.
REFERENCE_WINDOW_BLOCKS = ("base_pose", "base_vel", "joint_pos")

#: Shared by BOTH window terms.  ``command_window`` takes no ``blocks`` -- it is a single
#: 3-channel command, not a packed layout -- so the selection is added only for
#: ``reference_window`` below.
_REF_PARAMS = {
    "command_name": "traj_ref",
    "offsets": REFERENCE_OFFSETS,
}

_REF_WINDOW_PARAMS = {**_REF_PARAMS, "blocks": REFERENCE_WINDOW_BLOCKS}

#: Per-channel noise for the merged reference window.  A single scalar cannot serve it:
#: the window mixes metres, radians and rad/s, so the level that is right for a joint
#: angle is negligible for a joint velocity.  Magnitudes live in
#: ``mdp.REFERENCE_WINDOW_NOISE``; this expands them over whatever
#: ``REFERENCE_WINDOW_BLOCKS`` selects.
_REF_NOISE = mdp.ReferenceWindowNoiseCfg.from_layout(
    num_steps=len(REFERENCE_OFFSETS),
    num_joints=len(G1_JOINT_NAMES),
    num_ee_frames=REFERENCE_NUM_EE_FRAMES,
    blocks=REFERENCE_WINDOW_BLOCKS,
)

#: Per-channel multiplier for the reference window, applied AFTER noise and clipping.
#: Only the joint-velocity block moves (x0.05, matching the proprioceptive ``joint_vel``
#: term) so the encoder's inputs sit in a comparable range; see
#: ``mdp.REFERENCE_WINDOW_SCALE``.  Because the scale comes last, the noise half-widths
#: and the +-20 clip above stay in raw units, exactly as the old joint_vel_ref term had.
_REF_SCALE = mdp.reference_window_scale(
    num_steps=len(REFERENCE_OFFSETS),
    num_joints=len(G1_JOINT_NAMES),
    num_ee_frames=REFERENCE_NUM_EE_FRAMES,
    blocks=REFERENCE_WINDOW_BLOCKS,
)


##
# Scene
##
@configclass
class G1TeacherSceneCfg(G1ClfTrackingSceneCfg):
    """Teacher scene + a downward height-scan ray-caster on the pelvis (176 rays).

    The scan is the FM's height-conditioning token AND a teacher observation.  It needs a
    raycast-able MESH terrain -- the RayCaster cannot hit the infinite "plane" primitive --
    which the clip-library terrain importer provides (see
    :class:`~.g1_teacher_terrain_env_cfg.G1TeacherTerrainSceneCfg`).
    """

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.75, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


##
# Commands
##
@configclass
class G1TeacherCommandsCfg:
    """Single open-loop clip-tracking command; no velocity conditioner."""

    traj_ref = MotionSequenceCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        # No default library: the terrain mesh AND the tracked clips both come from one
        # ``.h5``, and a stale default silently trains on the wrong terrain. ``--clip``
        # sets this and scene.terrain.clip_library_path together.
        path="",
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,  # training: off (enabled in the -play variant)
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        # Put every class in the same currency as metres -- see CLF.eta_scales. Unscaled,
        # link angular velocity took 68% of V and link POSITION 5%, at comparable RELATIVE
        # accuracy, so "body-link tracking" graded rates. ori is a characteristic length
        # (m/rad), the velocities a characteristic time (s). Shared with the tracker so
        # both grade tracking the same way. Re-derive with scripts/rsl_rl/clf_balance_check.
        eta_scales={"pos": 1.0, "ori": 0.45, "lin_vel": 0.7, "ang_vel": 0.2},
        # LOCAL reference frame (BeyondMimic / SONIC style). Every env, every step, the
        # clip is re-datumed onto the robot's root in whatever root DOFs its anchor mask
        # frees -- planar + yaw on flat, leaving z/roll/pitch at the clip's own value.
        # So the reward measures the motion's SHAPE and does not double-count planar
        # drift; on stairs/boxes the mask pins downrange/height/yaw, so drift there is
        # still penalised against the terrain.
        #
        # Drift is not un-graded, it is graded ONCE, by the termination, against the
        # globally-fixed clip -- see termination_reference_world_state.
        ego_anchor_fraction=1.0,
        ego_anchor_every=1,
        # The FM snap flag keeps its own, much slower cadence; sharing this one would
        # flood the recorder and collapse the FM's training target. Must match the
        # tracker's fm_update_every.
        fm_snap_every=12,
    )


##
# Observations
##
@configclass
class G1TeacherObservationsCfg:
    """Three groups: proprioception, reference, privileged critic.

    The actor consumes ``policy`` and ``reference`` as SEPARATE branches (see
    ``agents/rsl_rl_ppo_cfg.py``): ``policy`` feeds the trunk directly while
    ``reference`` goes through its own MLP encoder.  Keeping them apart at the
    observation-manager level is what makes that routing declarative -- rsl-rl's
    ``obs_groups`` maps group names onto model inputs.

    * ``policy`` -- deployable proprioception, each term carrying
      :data:`PROPRIO_HISTORY_LENGTH` consecutive control steps, plus the height
      scan (exteroception, no history).
    * ``reference`` -- the whole reference trajectory over
      :data:`REFERENCE_OFFSETS` samples, all relative / heading-invariant so no
      world coordinate reaches the policy.
    * ``critic`` -- the same history and reference, uncorrupted, plus privileged
      state and the world-frame CLF tracking vectors.

    Shared with :class:`G1TrackerEnvCfg` (one frame across both tasks).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable proprioception (history-stacked) + the height scan.

        Every term here is history-stacked EXCEPT the height scan: the history is
        proprioception (what the robot felt), and Isaac Lab stacks oldest-first with
        noise/clip/scale baked into each slot independently -- which is the physical
        thing, each past step carries its own sensor noise.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),
                               history_length=PROPRIO_HISTORY_LENGTH)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
                                    history_length=PROPRIO_HISTORY_LENGTH)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                            history_length=PROPRIO_HISTORY_LENGTH)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05,
                            history_length=PROPRIO_HISTORY_LENGTH)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0),
                          history_length=PROPRIO_HISTORY_LENGTH)

        # Terrain awareness: per-ray ground clearance from the pelvis height scanner
        # (the same scan the FM conditions on). Flat ground -> ~constant; carries terrain
        # shape on rough ground. Noised like the other deployable sensor terms. NOT
        # history-stacked -- exteroception, and 176 rays x 10 steps buys little.
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.04, n_max=0.04), clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # NB: history is declared PER TERM above, never via
            # ObservationGroupCfg.history_length -- that field blanket-overrides every
            # term in the group, which would stack the height scan too.

    @configclass
    class ReferenceCfg(ObsGroup):
        """The reference trajectory: current step + look-ahead, base-relative.

        Fed to the actor's dedicated encoder rather than the trunk.  The reference
        *motion* carries observation noise so the teacher learns to track a noisy
        reference -- the generator (FM) predicts it at deploy and its output is not
        perfectly clean.  The joystick command stays clean: at deploy it comes from
        the joystick / scripted profile, not the FM.
        """

        # Per-CHANNEL noise (see _REF_NOISE): position blocks ~cm, velocity blocks
        # 0.25-0.5 in their own units. The clip saturates the velocity channels the way
        # the old joint_vel_ref term did; it is a no-op for the position channels.
        reference_window = ObsTerm(func=mdp.reference_window, params=_REF_WINDOW_PARAMS,
                                   noise=_REF_NOISE, clip=(-20.0, 20.0), scale=_REF_SCALE)
        command_window = ObsTerm(func=mdp.command_window, params=_REF_PARAMS)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic: clean state (history-stacked) + world-frame CLF vectors."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0,
                               history_length=PROPRIO_HISTORY_LENGTH)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0,
                               history_length=PROPRIO_HISTORY_LENGTH)
        root_quat = ObsTerm(func=mdp.root_quat_w, scale=1.0,
                            history_length=PROPRIO_HISTORY_LENGTH)
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    history_length=PROPRIO_HISTORY_LENGTH)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            history_length=PROPRIO_HISTORY_LENGTH)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05,
                            history_length=PROPRIO_HISTORY_LENGTH)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0),
                          history_length=PROPRIO_HISTORY_LENGTH)

        # Same reference window the actor's encoder sees, uncorrupted.
        # Same window the actor's encoder sees, uncorrupted -- but identically clipped and
        # scaled, so the two networks read the reference in the same units.
        reference_window = ObsTerm(func=mdp.reference_window, params=_REF_WINDOW_PARAMS,
                                   clip=(-20.0, 20.0), scale=_REF_SCALE)
        command_window = ObsTerm(func=mdp.command_window, params=_REF_PARAMS)

        # The two errors, privileged. Together they are what the value function has to
        # predict, and they are deliberately in DIFFERENT frames because the reward and
        # the termination are:
        #
        #  * LOCAL -- the CLF's own eta, in the anchored frame, so planar drift is already
        #    forgiven. This is the reward's argument.
        #  * GLOBAL -- base pose error against the clip on its reset-time terrain cell.
        #    This is the drift the reward ignores and the TERMINATION fires on, so without
        #    it the critic would be predicting an episode end it cannot see coming.
        #
        # These replace the old ref_traj / act_traj / *_vel quadruple, which handed over
        # both raw sides in absolute world coords and made the critic re-derive the error.
        tracking_error_local = ObsTerm(func=mdp.tracking_error_local,
                                       params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))
        tracking_error_global = ObsTerm(func=mdp.tracking_error_global,
                                        params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))

        # Terrain height scan (clean; critic corruption is off).
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    reference: ReferenceCfg = ReferenceCfg()
    critic: CriticCfg = CriticCfg()


##
# Events
##
@configclass
class G1TeacherEventsCfg(G1ClfTrackingEventsCfg):
    """Reuse dynamics randomization; add the reset onto a clip."""

    reset_on_clip = EventTerm(
        func=mdp.reset_on_motion_sequence,
        mode="reset",
        # ``pin_*`` are evaluation knobs and default to OFF (sample a clip + start step per
        # reset, as training wants). Declared here rather than left to the function
        # signature because hydra refuses to override a key absent from the config.
        #
        # The sentinels are typed, not None: Isaac Lab's configclass infers a param's type
        # from its default, so a ``None`` default makes the field NoneType and rejects the
        # string/int the override supplies. Hence "" and -1 for "off".
        # See mdp.events.resets.resolve_pinned_clips.
        params={"command_name": "traj_ref", "base_frame_name": "pelvis_link",
                "pin_clip_ids": "", "pin_start_step": -1},
    )


##
# Rewards
##
@configclass
class G1TeacherRewardCfg(G1ClfTrackingRewardCfg):
    """CLF tracking + penalties.

    The two dense commanded-velocity terms that used to live here (``xy_vel`` /
    ``yaw_vel``, weight 1.0 each against ``clf_reward``'s 10, grading the clip's
    joystick-level ``target_vel`` against the measured base twist) are GONE: the CLF
    already grades the base twist against the reference, so they were re-scoring the same
    quantity through a second, softer lens.

    Removing them does change what is graded, and the change is the point.  The CLF grades
    reference SHAPE in the locally-anchored frame and deliberately does not double-count
    planar drift (see the ``ego_anchor_fraction`` note on the command above), so the only
    remaining pressure to actually travel at the commanded velocity is
    ``base_pose_error_termination`` against the GLOBAL clip.  If the policy starts
    under-tracking the command, that is the term to look at first.

    Evaluation is unaffected: ``play_plots.py`` still reports
    ``vel_track_{forward_vx,lateral_vy,yaw_wz}_mse_*`` from the recorded ``target_vel``.
    The reward functions themselves are kept in ``mdp.rewards`` for the tracker task.
    """


##
# Terminations
##
@configclass
class G1TeacherTerminationCfg:
    """Two failure terminations (global base drift, local ee drift) + truncations.

    The pair is deliberate and mirrors the reward/termination frame split: the base term
    watches the GLOBAL clip (drift the reward forgives), the ee term watches each limb
    against the LOCAL, anchored reference (error the reward grades but only softly).  A
    robot can fail either way independently -- hold its base pose while dragging a foot
    through the wrong place, or track every limb while walking off the clip.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    clip_end = DoneTerm(
        func=mdp.clip_end_reached, params={"command_name": "traj_ref"}, time_out=True
    )
    # The 0.8 rad bounds are unchanged.
    #
    # The base term measures only the anchor mask's PINNED axes -- on flat that is height +
    # tilt (exactly SONIC's condition), on stairs/boxes it adds downrange and yaw. A FREE
    # axis is one the clip was slid along to sit on the robot, so error there is zero by
    # construction and invisible to the policy; bounding it only truncated episodes, and was
    # 90% of all base terminations at 2.6k iterations. Because the free axes are no longer
    # measured, ONE tight 0.3 works everywhere -- no separate inflated bound for flat.
    base_pose_error = DoneTerm(
        func=mdp.base_pose_error_termination,
        params={"command_name": "traj_ref", "max_pos_dist": 0.3, "max_ori_rad": 0.8},
    )
    ee_pose_error = DoneTerm(
        func=mdp.tracked_frame_error_termination,
        params={
            "command_name": "traj_ref",
            "frame_names": ("left_ankle_roll_link", "right_ankle_roll_link",
                            "left_wrist_yaw_link", "right_wrist_yaw_link"),
            "max_pos_dist": 0.3,
            # Per frame, in frame_names order: ankles 0.8 rad, wrists 1.0.
            #
            # The wrists get slack but KEEP a bound, because the constraint is demonstrably
            # satisfiable: body_v2's left_wrist_yaw orientation trips at a 0.8 bound went
            # 25 -> 12 -> 5 over iterations 2.6k -> 5k -> 8.2k. It is learnable, so removing
            # it entirely would discard a real signal.
            #
            # Slack because body_v3 is far worse on this one axis than body_v2 was at the
            # SAME iteration -- 305 trips vs 25, while being 120x better on v_root_body_pos.
            # Part of that is exposure: v3's episodes run 196 steps vs 132, so it reaches
            # more of each clip's dynamic arm motion. 1.0 is a mild easing (it only moved v3
            # 305 -> 290 on its own) that keeps the signal while the rest converges.
            #
            # Watch it: if v3 does not follow v2's downward curve, the terrain clips may be
            # asking for wrist motion that is not trackable, which is a clip-generation
            # question rather than a threshold one. 286 of those 305 trips were on TERRAIN
            # and only 4 on flat.
            "max_ori_rad": (0.8, 0.8, 1.0, 1.0),
        },
    )


##
# Environment
##
@configclass
class G1TeacherEnvCfg(G1ClfTrackingEnvCfg):
    """Open-loop motion-sequence tracking env -- ABSTRACT (no ground assigned).

    Everything that does not depend on the ground lives here: the clip-tracking command,
    the CLF + commanded-velocity rewards, the base-relative policy observations, the
    privileged critic, the clip reset, and the pelvis height scanner.  The ground itself
    comes from :class:`~.g1_teacher_terrain_env_cfg.G1TeacherTerrainEnvCfg`, which meshes
    the clip library; this class is not registered.
    """

    scene: G1TeacherSceneCfg = G1TeacherSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1TeacherObservationsCfg = G1TeacherObservationsCfg()
    commands: G1TeacherCommandsCfg = G1TeacherCommandsCfg()
    rewards: G1TeacherRewardCfg = G1TeacherRewardCfg()
    events: G1TeacherEventsCfg = G1TeacherEventsCfg()
    terminations: G1TeacherTerminationCfg = G1TeacherTerminationCfg()

    def __post_init__(self):
        super().__post_init__()
        # Episode length matches a 20 s clip at the 0.02 s control dt
        # (decimation 4 * sim.dt 0.005); regenerate clips at this dt.
        self.episode_length_s = 20.0
        # No external pushes.
        self.events.push_robot = None
        # Stronger undesired-contact penalty + tighter CLF normalization.
        self.rewards.undesired_contacts.weight = -0.5
        # CLF reward normaliser: reward = 10 * exp(-V / (lambda_max * max_eta_err^2)).
        # Must be re-derived whenever ANY of Q, R or ang_vel_scale move, since they change
        # V's scale underneath it. The sizing rule, applied by clf_balance_check.py: put
        # half-reward at ~3x the V a live policy actually averages, which lands the
        # instantaneous reward near 8/10 in steady tracking and ~5/10 as an episode dies.
        #
        # 1.15 goes with the eta_scales above PLUS the reset fix and the now-GLOBAL base
        # channels: measured mean V 1.12 on a live policy. Most of the drop from 2.20 is the
        # reset fix (df7a1de) removing the spawn-step spikes that were inflating V, not the
        # base change. History: 1.6 for eta_scales with the local base and the reset bug; 2.25
        # for ang_vel-only scaling at 0.3; 4.1 unscaled with _ROOT_Q_DEFAULT = [10.0, 0.1];
        # 3.5 for the old [1.0, 0.1] at a measured V ~= 41; 5.0 before that, where
        # half-reward sat at V ~= 77 and the policy only got there as it was dying.
        #
        # ``clf_decreasing_condition`` needs NO matching change: eta_scales is folded into P,
        # not into eta, so eta_max / eta_dot_max keep their physical units.
        #
        # RE-DERIVED and left at 1.15. The worry was that this value came from body_v2's
        # model_5800, trained under both the old local-base reward and the reset-step
        # placement bug -- two reward generations stale. Measured against depth_v3's
        # model_4200 in its own env (clf_balance_check.py, 64 envs x 200 steps): mean V 1.10
        # versus the 1.12 that 1.15 was sized against, i.e. V did not move. The 3x rule wants
        # 1.09 and the instantaneous reward is already 8.098/10, so the correction is inside
        # the noise and not worth breaking comparability with depth_v1/v2/v3 for.
        #
        # For reference, the same policy measures mean V 1.42 (-> 1.25) in the CURRENT env,
        # but that is a policy on out-of-distribution depth after the near-camera move plus
        # harder spawns from the reset cap -- an upper bound that training should recover,
        # not a reason to move the scale.
        #
        # Two traps when re-tuning this from the dashboard:
        #   * ``Episode_Reward/clf_reward`` is an episode SUM x dt divided by
        #     max_episode_length_s, so it reads ~7% of the instantaneous value while
        #     episodes are ~1.3 s of a 20 s window. It tracks SURVIVAL as much as quality.
        #   * ``Metrics/traj_ref/v`` is logged at reset, so it is V at the instant of
        #     FAILURE, not the episode average. Back the average out of the reward instead.
        self.rewards.clf_reward.params["max_eta_err"] = 1.15
        self.rewards.clf_decreasing_condition.params["eta_max"] = 5.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 10.0
        self._apply_tracking_mode()

    def _apply_tracking_mode(self) -> None:
        """Validate ``commands.traj_ref.tracking_mode``.

        The OBSERVATION is deliberately identical in both modes: the reference window keeps
        its joint + ee content, so the A/B isolates the thing under test -- what the reward
        grades -- rather than confounding it with a change in what the policy can see.  The
        reward side needs nothing here either; the command reads ``tracking_mode`` itself and
        builds the CLF over the matching channel subset.

        So this is only a fail-fast on a typo'd mode, which would otherwise surface much
        later as a confusing channel-subset error inside the CLF.
        """
        mode = self.commands.traj_ref.tracking_mode
        if mode not in ("joint", "body"):
            raise ValueError(f"tracking_mode must be 'joint' or 'body'; got {mode!r}.")
