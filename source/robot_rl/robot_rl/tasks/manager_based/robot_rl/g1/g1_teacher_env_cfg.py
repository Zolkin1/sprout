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
    """World-frame-aware observations: deployable policy, privileged critic.

    The CLF tracking vectors are world-frame (absolute position / heading), so
    the raw ``ref_traj``/``act_traj``/``*_vel`` observations live in the
    critic-only group; the policy's reference terms (base-pose error, joint
    refs, ee refs, future windows) are all relative / heading-invariant, so the
    policy never sees a world coordinate.  Shared with
    :class:`G1TrackerEnvCfg` (one frame across both tasks)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable proprioception + base-relative reference (no world coords)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        # Reference (current step only), base-relative. The reference *motion* terms
        # carry observation noise so the teacher learns to track a noisy reference --
        # the generator (FM) predicts these at deploy and its output is not perfectly
        # clean. The command terms (command_vel / future_command_vel) stay clean: at
        # deploy they come from the joystick / scripted profile, not the FM.
        base_pose_error = ObsTerm(func=mdp.base_pose_error, params={"command_name": "traj_ref"},
                                  noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_pos_ref = ObsTerm(func=mdp.joint_pos_ref, params={"command_name": "traj_ref"},
                                noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel_ref = ObsTerm(func=mdp.joint_vel_ref, params={"command_name": "traj_ref"},
                                noise=Unoise(n_min=-0.5, n_max=0.5), clip=(-20.0, 20.0))
        ee_ref = ObsTerm(func=mdp.ee_ref_relative_to_base, params={"command_name": "traj_ref"},
                         noise=Unoise(n_min=-0.02, n_max=0.02))
        # Reference base velocity (current step), rotated into the base frame
        # (heading-invariant, deployable). The FM predicts this (the achieved-reference
        # velocity, not the command); noised like the other reference-motion terms.
        base_vel_ref = ObsTerm(func=mdp.ref_base_velocity_local, params={"command_name": "traj_ref"},
                               noise=Unoise(n_min=-0.1, n_max=0.1))
        # Joystick-level velocity command that generated the clip (target_vel =
        # (vx, vy, wz)); pairs with the xy_vel / yaw_vel tracking rewards. No noise:
        # this is the upstream command, not the FM-predicted reference.
        command_vel = ObsTerm(func=mdp.command_velocity, params={"command_name": "traj_ref"})

        # Look-ahead into the clip (next 0.5 s, subsampled every 5 steps): future
        # joystick command + future base pose/velocity, so the teacher can
        # anticipate upcoming slowdowns / turns. Base pose/vel are relative to the
        # current actual base (a "future base_pose_error"); noised like the
        # current-step reference (the FM predicts these too).
        future_command_vel = ObsTerm(
            func=mdp.future_command_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )
        future_base_pose = ObsTerm(
            func=mdp.future_base_pose,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        future_base_vel = ObsTerm(
            func=mdp.future_base_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # Terrain awareness: per-ray ground clearance from the pelvis height scanner
        # (the same scan the FM conditions on). Flat ground -> ~constant; carries terrain
        # shape on rough ground. Noised like the other deployable sensor terms.
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic: clean state + world-frame CLF tracking vectors."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        root_quat = ObsTerm(func=mdp.root_quat_w, scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        base_pose_error = ObsTerm(func=mdp.base_pose_error, params={"command_name": "traj_ref"})
        joint_pos_ref = ObsTerm(func=mdp.joint_pos_ref, params={"command_name": "traj_ref"})
        joint_vel_ref = ObsTerm(func=mdp.joint_vel_ref, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0))
        ee_ref = ObsTerm(func=mdp.ee_ref_relative_to_base, params={"command_name": "traj_ref"})
        # Joystick-level velocity command that generated the clip (target_vel).
        command_vel = ObsTerm(func=mdp.command_velocity, params={"command_name": "traj_ref"})

        # Look-ahead into the clip (next 0.5 s, subsampled every 5 steps), same as
        # the policy group (future command + base pose/vel rel. to current base).
        future_command_vel = ObsTerm(
            func=mdp.future_command_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )
        future_base_pose = ObsTerm(
            func=mdp.future_base_pose,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )
        future_base_vel = ObsTerm(
            func=mdp.future_base_velocity,
            params={"command_name": "traj_ref", "horizon_s": 0.5, "stride": 5},
        )

        # Full world-frame tracking vector (privileged -- absolute coords).
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
        self._apply_tracking_mode()

    def _apply_tracking_mode(self) -> None:
        """Swap the reference observation terms to match ``commands.traj_ref.tracking_mode``.

        The reward side needs nothing here -- the command reads ``tracking_mode`` itself and
        builds the CLF over the matching channel subset.  This is only the observation half:
        under ``"body"`` the policy and critic see the root-frame body targets instead of the
        reference joint angles, so the two arms of the A/B differ in *both* what is graded and
        what is observed, which is the comparison being made.

        ``ee_ref`` stays in both modes: the world end-effector frames are tracked either way.
        Proprioception (``joint_pos`` / ``joint_vel`` / ``actions``) also stays -- the action
        space is joint PD targets and joint encoders are what the hardware measures.
        """
        mode = self.commands.traj_ref.tracking_mode
        if mode == "joint":
            return
        if mode != "body":
            raise ValueError(f"tracking_mode must be 'joint' or 'body'; got {mode!r}.")

        for group, pos_noise, vel_noise in (
            (self.observations.policy, Unoise(n_min=-0.02, n_max=0.02),
             Unoise(n_min=-0.5, n_max=0.5)),
            (self.observations.critic, None, None),
        ):
            group.joint_pos_ref = None
            group.joint_vel_ref = None
            group.body_pos_ref = ObsTerm(
                func=mdp.body_pos_ref_root, params={"command_name": "traj_ref"},
                noise=pos_noise,
            )
            group.body_vel_ref = ObsTerm(
                func=mdp.body_vel_ref_root, params={"command_name": "traj_ref"},
                noise=vel_noise, clip=(-20.0, 20.0),
            )
