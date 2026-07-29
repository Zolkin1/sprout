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
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns

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


##
# Observation window sizes -- one knob each, shared by the policy and the critic.
##

#: Proprioception window length in CONTROL steps, current step included.  10 = the
#: current step plus the 9 before it, i.e. 0.18 s of history at the 0.02 s control dt.
#: These are *consecutive* steps (Isaac Lab's per-term ``history_length`` buffers every
#: step); only the reference look-ahead below is decimated.  Set to 1 for no history.
PROPRIO_HISTORY_LENGTH = 10

#: Reference window length in samples, current step included.  11 = the current step
#: plus 10 future ones.  Set to 1 to observe only the current reference.
REFERENCE_NUM_STEPS = 11

#: Control steps between reference samples.  5 * 0.02 s = 0.1 s, so 11 samples span
#: 0.0 .. 1.0 s of look-ahead.
REFERENCE_STRIDE = 5

#: End-effector frames the reference window carries (clip output frames minus the base:
#: both ankles + both wrists).  Only needed to size the per-channel noise vector below;
#: the observation itself takes the list from the command's ``resolve_ee_frames``.  A
#: mismatch fails loudly when the noise is broadcast against the observation.
REFERENCE_NUM_EE_FRAMES = 4

_REF_PARAMS = {
    "command_name": "traj_ref",
    "num_steps": REFERENCE_NUM_STEPS,
    "stride": REFERENCE_STRIDE,
}

#: Per-channel noise for the merged reference window.  A single scalar cannot serve it:
#: the window mixes metres, radians and rad/s, so the level that is right for a joint
#: angle is negligible for a joint velocity.  Magnitudes live in
#: ``mdp.REFERENCE_WINDOW_NOISE``; this expands them over the full 11 x 82 layout.
_REF_NOISE = mdp.ReferenceWindowNoiseCfg.from_layout(
    num_steps=REFERENCE_NUM_STEPS,
    num_joints=len(G1_JOINT_NAMES),
    num_ee_frames=REFERENCE_NUM_EE_FRAMES,
)

#: Per-channel multiplier for the reference window, applied AFTER noise and clipping.
#: Only the joint-velocity block moves (x0.05, matching the proprioceptive ``joint_vel``
#: term) so the encoder's inputs sit in a comparable range; see
#: ``mdp.REFERENCE_WINDOW_SCALE``.  Because the scale comes last, the noise half-widths
#: and the +-20 clip above stay in raw units, exactly as the old joint_vel_ref term had.
_REF_SCALE = mdp.reference_window_scale(
    num_steps=REFERENCE_NUM_STEPS,
    num_joints=len(G1_JOINT_NAMES),
    num_ee_frames=REFERENCE_NUM_EE_FRAMES,
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
        # Ego-centric receding-horizon reference: half the envs have their clip
        # periodically re-placed onto the robot's root (planar pos + heading), so
        # the reference looks like an MPC plan that re-plans every few steps. The
        # rest keep the fixed world-frame clip. Set ego_anchor_fraction=0 to disable.
        ego_anchor_fraction=0.5,
        ego_anchor_every=12,  # re-plan every 12 control steps (0.24 s at dt=0.02)
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
      :data:`REFERENCE_NUM_STEPS` samples, all relative / heading-invariant so no
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
        reference_window = ObsTerm(func=mdp.reference_window, params=_REF_PARAMS,
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
        reference_window = ObsTerm(func=mdp.reference_window, params=_REF_PARAMS,
                                   clip=(-20.0, 20.0), scale=_REF_SCALE)
        command_window = ObsTerm(func=mdp.command_window, params=_REF_PARAMS)

        # Full world-frame tracking vector (privileged -- absolute coords). NOT
        # redundant with reference_window: act_traj/act_traj_vel are the only MEASURED
        # Cartesian ee state anywhere in the observations, ref_traj carries ee
        # reference ORIENTATION and velocity (the window keeps ee positions only), and
        # the four together are exactly the CLF's arguments
        # (clf.compute_vdot(y_act, y_des, dy_act, dy_des)) -- i.e. the arguments of the
        # reward this critic has to predict.
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))

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
        params={"command_name": "traj_ref", "base_frame_name": "pelvis_link"},
    )


##
# Rewards
##
@configclass
class G1TeacherRewardCfg(G1ClfTrackingRewardCfg):
    """CLF tracking + penalties, plus dense commanded-velocity tracking."""

    # Track the clip's joystick-level velocity command (target_vel = (vx, vy, wz)) -- the
    # same signal a velocity joystick would supply -- rather than the achieved reference
    # base velocity.  Split into planar (xy) + yaw exactly like the stock
    # track_lin_vel_xy_exp / track_ang_vel_z_exp, but sourced from traj_ref.
    xy_vel = RewTerm(
        func=mdp.command_lin_vel_xy_tracking_exp,
        weight=1.0,
        params={"command_name": "traj_ref", "std": 0.5},
    )
    yaw_vel = RewTerm(
        func=mdp.command_ang_vel_z_tracking_exp,
        weight=1.0,
        params={"command_name": "traj_ref", "std": 0.5},
    )


##
# Terminations
##
@configclass
class G1TeacherTerminationCfg:
    """One failure termination (base drift) + clip-end / time-out truncations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    clip_end = DoneTerm(
        func=mdp.clip_end_reached, params={"command_name": "traj_ref"}, time_out=True
    )
    base_pose_error = DoneTerm(
        func=mdp.base_pose_error_termination,
        params={"command_name": "traj_ref", "max_pos_dist": 0.75, "max_ori_rad": 0.8},
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
        self.rewards.clf_reward.params["max_eta_err"] = 5.0
        self.rewards.clf_decreasing_condition.params["eta_max"] = 5.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 10.0
