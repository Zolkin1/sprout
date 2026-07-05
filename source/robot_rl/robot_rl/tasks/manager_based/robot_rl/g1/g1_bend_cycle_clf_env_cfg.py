# Copyright 2026 Zachary Olkin. All rights reserved.

"""Combined bend-over + bend-up G1 environment (one periodic policy).

The two formerly-separate episodic motions ("bow forward" and "bend back up")
are treated as ONE ``full_periodic`` trajectory over the phasing variable
``phi in [0, 1]``:

- ``phi 0.0 -> 0.5`` : bend over,
- ``phi 0.5 -> 1.0`` : bend back up,
- ``phi = 0`` and ``phi = 1.0`` are the standing pose (the cycle
  stand -> bend -> stand is genuinely periodic); ``phi = 0.5`` is bent over.

Every env runs the cycle with per-episode-random dwells at standing and at the
bent pose: stand -> hold standing (``stand_dwell_s``) -> bend -> hold bent
(``bent_dwell_s``) -> bend up -> stand -> repeat.  So every env trains both
(fragile) holds before/after the motion.  The bent pose is the clip's
``bend_down -> bend_up`` domain boundary (the deepest bend, ~0.474), auto-derived.
The dwells are realised by :func:`mdp.reset_bend_cycle` (draws the per-env dwells
+ spawns standing) plus :meth:`PhasedTrajectoryCommand._dwell_cycle_phase` (pauses
the phase at standing / bent and zeroes the reference velocity during each pause).

Built on the live :class:`G1ClfTrackingEnvCfg` (single-skill
``TrajectoryCommandCfg`` path), not the deprecated ``g1_trajopt_*`` bow/bend
configs.
"""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import (
    TrajectoryCommandCfg,
)

from .g1_clf_tracking_base import G1ClfTrackingCommandCfg, G1ClfTrackingEnvCfg
from .clf_weights import BOWING_Q_weights, BOWING_R_weights


# Single ``full_periodic`` bend-cycle clip: domain_sequence [bend_down, bend_up],
# so phi 0->~0.47 is the bend-over and ~0.47->1.0 the bend-up (durations 1.8s /
# 2.0s), with phi 0 == phi 1.0 == standing.  For the HF loader the path must point
# at the trajectory YAML *file* (the single-skill ``TrajectoryManager`` opens it
# directly; only the multi-skill library loader takes a folder).
BEND_CYCLE_HF_REPO = "zolkin/robot_rl"
BEND_CYCLE_PATH = (
    "trajectories/bow_cycle/2026-07-03_10-34-37_bow_cycle_config/bow_cycle_config_solution.yaml"
)


##
# Commands
##
@configclass
class G1BendCycleCommandsCfg(G1ClfTrackingCommandCfg):
    """Single-skill command driving the combined bend-over/bend-up periodic clip."""

    traj_ref = TrajectoryCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        manager_type="trajectory",
        hf_repo=BEND_CYCLE_HF_REPO,
        path=BEND_CYCLE_PATH,
        conditioner_generator_name="base_velocity",
        Q_weights=BOWING_Q_weights,
        R_weights=BOWING_R_weights,
        # Every cycle holds standing (phase 0) for 1-4 s and the bent pose for
        # 2-5 s (per-episode-random), so every env trains both holds. The bent
        # phase auto-derives from the bend_down->bend_up domain boundary
        # (bent_hold_phi=None -> ~0.474, the deepest bend).
        stand_dwell_s=(1.0, 4.0),
        bent_dwell_s=(2.0, 5.0),
    )


##
# Environment
##
@configclass
class G1BendCycleCLFEnvCfg(G1ClfTrackingEnvCfg):
    """G1 combined bend-cycle CLF tracking environment (flat ground, one policy)."""

    commands: G1BendCycleCommandsCfg = G1BendCycleCommandsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Long enough for one full cycle even at the max dwells: up to 4s stand
        # hold + ~1.9s bend + up to 5s bent hold + ~1.9s bend up = ~12.8s, so
        # bending back up keeps getting trained.
        self.episode_length_s = 15.0

        ##
        # Commands — stationary motion, so zero the base-velocity command.
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        ##
        # Events — reset every env into the phase-0 standing pose + draw its
        # per-episode bent-dwell duration.  Replaces the base ``reset_on_ref``
        # (which is coupled to the multi-skill velocity conditioner the
        # bend-cycle env does not use).
        ##
        self.events.reset_on_ref = EventTerm(
            func=mdp.reset_bend_cycle,
            mode="reset",
            params={
                "command_name": "traj_ref",
                "base_frame_name": "pelvis_link",
            },
        )

        ##
        # Domain randomization — the bend policy must be robust to carrying a load on
        # its back while bent over.
        ##
        # Widen the torso (waist) mass scale from ±15% to 0.7–1.3.
        self.events.add_base_mass.params["mass_distribution_params"] = (0.7, 1.3)
        # Widen the vertical (z) torso-COM jitter from ±1 cm to ±5 cm (x/y unchanged).
        self.events.base_com.params["com_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }
        # Randomize the UPPER-BODY (waist lump) CoM position — the training USD
        # uses a ``box_merged`` model that lumps torso+head+arms into a single
        # ``waist_yaw_link`` CoM, whereas the real robot (≈ the distributed stock
        # MuJoCo model) carries that mass ~+1.7 cm higher and ~+0.6 cm more forward.
        # ``base_com`` only jitters the *pelvis*, and ``add_base_mass`` scales the
        # waist mass but keeps its CoM fixed, so the policy otherwise never trains
        # against the higher/forward upper CoM that tips it in sim2sim. Use a broad
        # symmetric ±5 cm jitter (well past the measured stock−training delta) so
        # the policy is robust wherever the real upper CoM actually lands.
        self.events.waist_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
        # Half the envs carry a 5–7 kg back payload: a point mass at a random position
        # within a 5 cm cube centered ~10 cm behind and ~15 cm above the torso frame
        # (a backpack-like load that pulls the robot backward/up when it bends).
        self.events.add_back_payload = EventTerm(
            func=mdp.add_payload_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_range": (5.0, 7.0),
                "position_range": {
                    "x": (-0.125, -0.075),
                    "y": (-0.025, 0.025),
                    "z": (0.125, 0.175),
                },
                "fraction": 0.5,
                "debug_vis": False,
            },
        )

        ##
        # Rewards — retune the CLF terms to the bend (BOWING-weight) scale.
        # The base defaults (max_eta_err=6, decreasing weight -5) are tuned for
        # the looser walking CLF; the bend uses the tighter ``max_eta_err``.
        # Keep the CLF *decreasing* (stability) term ON: it penalizes the
        # Lyapunov function failing to decrease (tracking error growing / ``η̇``
        # not going negative), which directly damps the oscillation around the
        # bent hold rather than just the mean tracking error.
        ##
        self.rewards.clf_reward.params["max_eta_err"] = 0.25
        self.rewards.clf_decreasing_condition.weight = -5.0

        ##
        # Terrain — flat ground, no generator / height scan / curriculum.
        ##
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None


@configclass
class G1BendCycleCLFEcEnvCfg(G1BendCycleCLFEnvCfg):
    """Bend-cycle env with the extra-compute (EC) plate mass on the torso."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        # just front: 0.616 kg
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616, 0.616),
                "operation": "add",
            },
        )


@configclass
class G1BendCycleCLFEnvCfg_PLAY(G1BendCycleCLFEnvCfg):
    """Playback config: few envs, no corruption, no domain randomization."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Disable domain randomization for clean playback.
        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.waist_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Keep the back payload ON so its placement is visible: every env carries one
        # (fraction=1) and the debug sphere is drawn at the payload's live position.
        self.events.add_back_payload.params["fraction"] = 0.0 #1.0
        self.events.add_back_payload.params["debug_vis"] = True


@configclass
class G1BendCycleCLFEnvCfg_PLAY_DR(G1BendCycleCLFEnvCfg):
    """Playback with the mass / CoM domain randomization left ON.

    Same few-env, corruption-free viewing as ``_PLAY`` but keeps the persistent
    physical DR — torso mass scale, pelvis + waist-lump CoM jitter (±5 cm), the
    back payload, friction/gains — so you can watch the *current* policy meet the
    randomized upper-body distribution: confirm it tips under a forward/high waist
    CoM *before* retraining, and confirm it no longer does after. Transient
    disturbances (pushes / external force) are switched OFF so any tip is
    attributable to the mass distribution and not a random kick. Uses more envs
    than ``_PLAY`` so the randomized range is well sampled (some envs land in the
    tipping regime).
    """

    def __post_init__(self):
        # Post init of parent (the TRAIN cfg -> all DR present)
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Drop only the transient disturbances so a tip is a pure mass/CoM result;
        # the mass-distribution DR (add_base_mass, base_com, waist_com,
        # add_back_payload) stays exactly as configured for training.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
