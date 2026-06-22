# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 flow-matching generator-tracking environment (flat ground).

Tracks a learned flow-matching *generator* instead of (only) the pre-made clips:
a fraction of envs follow the clip reference (the stable teacher signal), the
rest follow the generator's predicted future state. Both sources drive a single
**world-frame** CLF reward, so the tracker can ultimately be deployed sim2sim /
sim2real with the generator + tracker pair (no privileged info in the policy).

Differs from :class:`G1MotionSequenceEnvCfg` only in the command term
(:class:`GeneratorTrackingCommand`, the clip + generator mix).  Both tasks now
run a world-frame CLF, so they share one observation layout
(:class:`G1MotionSeqObservationsCfg`): the raw ``ref_traj``/``act_traj``/
``*_vel`` tracking vectors are critic-only (absolute world coords), and the
policy's reference terms (base-pose error, joint refs, ee refs, future windows)
are all relative / heading-invariant.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from .g1_motion_sequence_env_cfg import (
    DEFAULT_CLIP_PATH,
    G1MotionSeqObservationsCfg,
    G1MotionSequenceEnvCfg,
)
from ..mdp.commands.traj_tracking.generator_tracking_cmd_cfg import GeneratorTrackingCommandCfg

# Default flow-matching generator checkpoint (override per-run via the cfg / CLI).
DEFAULT_FM_CHECKPOINT = "logs/flow_matching/2026-06-21_17-07-59/ckpt_89.pt"


##
# Commands
##
@configclass
class G1GeneratorTrackingCommandsCfg:
    """Clip + flow-matching-generator tracking command."""

    traj_ref = GeneratorTrackingCommandCfg(
        contact_bodies=[".*_ankle_roll_link"],
        path=DEFAULT_CLIP_PATH,
        base_frame_name="pelvis_link",
        random_start_phase=True,
        random_start_min_remaining_s=2.0,
        randomize_heading=False,
        debug_vis=False,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        # --- generator-tracking knobs ---
        fm_checkpoint=DEFAULT_FM_CHECKPOINT,
        clip_fraction=0.35,
        fm_update_every=2,
    )


##
# Environment
##
@configclass
class G1GeneratorTrackingEnvCfg(G1MotionSequenceEnvCfg):
    """Flat-ground generator-tracking env (clip + flow-matching mix, world-frame CLF)."""

    # Shared world-frame observation layout (policy base-relative, critic
    # privileged); only the command term differs from the clip-only task.
    observations: G1MotionSeqObservationsCfg = G1MotionSeqObservationsCfg()
    commands: G1GeneratorTrackingCommandsCfg = G1GeneratorTrackingCommandsCfg()


@configclass
class G1GeneratorTrackingEnvCfgPlay(G1GeneratorTrackingEnvCfg):
    """Smaller, noise-free variant for play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.randomize_ground_contact_friction = None
        # Draw the (clip) reference-trajectory markers in the viewer.
        self.commands.traj_ref.debug_vis = True
