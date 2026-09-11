# Copyright 2026 Zachary Olkin. All rights reserved.

"""G1 teacher with NO exteroception at all, on flat ground only.

The deploy-isolation policy. When the camera pair misbehaves on hardware there is no way to
tell a perception fault from a tracking fault, because every deployable teacher on this branch
reads the ground through the cameras. This one reads nothing: the policy group is
proprioception plus the reference window, and that is all. If it walks and the depth policy
does not, the difference is the cameras.

Two deliberate choices, both of which cost something:

* **The critic KEEPS its height scan.** It is privileged and never deployed, so removing it
  would change the value function's problem for no deployment benefit, and would make this
  teacher's learning curve incomparable with the others'. On flat ground the scan is nearly
  constant anyway. To strip it too, set ``observations.critic.height_scan`` and
  ``scene.height_scanner`` to ``None`` -- and drop the play variant's inherited
  ``scene.height_scanner.debug_vis = True``, which would otherwise raise on ``None``.
* **The terrain still comes from the clip library**, through the usual ``ClipTerrainImporter``.
  "Flat only" is a property of the LIBRARY you pass to ``--clip``, not of this config: pair it
  with a flat-only subset (see ``scripts/rsl_rl/make_flat_clips.py``). Nothing here enforces
  that, so passing the full terrain library gives a blind policy on stairs, which will simply
  fall. The run's ``terrain_fingerprint`` records which library was used.

It extends the BODY-tracking terrain teacher, exactly as the depth teacher does, so the only
difference from ``G1-teacher-clf-terrain-depth`` is the exteroception and the ground -- which
is what makes it a usable control.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .g1_teacher_env_cfg import G1TeacherObservationsCfg
from .g1_teacher_terrain_env_cfg import G1TeacherTerrainBodyEnvCfg, G1TeacherTerrainBodyEnvCfgPlay

#: Action-rate penalty, back to the ``-0.01`` every teacher inherits from
#: :class:`~.g1_clf_tracking_base.G1ClfTrackingRewardCfg`. It was raised 10x for the runs
#: queued 2026-08-21 and is returned here, so this constant no longer overrides the base --
#: it is kept, rather than deleted, because it is what holds the blind and depth arms to the
#: SAME smoothness cost. A different one on each would confound the blind-vs-camera
#: comparison these configs exist for, and that is a property worth stating even when the
#: value happens to match the default.
ACTION_RATE_WEIGHT = -0.01


@configclass
class G1TeacherFlatBlindObservationsCfg(G1TeacherObservationsCfg):
    """Teacher observations with the height scan REMOVED from the policy group.

    Only the ``policy`` group changes. The proprioceptive history, the ``reference`` encoder
    group and the privileged ``critic`` -- scan included -- all come straight from
    :class:`~.g1_teacher_env_cfg.G1TeacherObservationsCfg`.
    """

    @configclass
    class PolicyCfg(G1TeacherObservationsCfg.PolicyCfg):
        """Proprioception and nothing else.

        ``height_scan = None`` deletes the inherited term. The remaining terms keep their
        ``PROPRIO_HISTORY_LENGTH`` stacking untouched, so the trunk sees exactly the history
        the other teachers do, minus 176 rays.
        """

        height_scan = None

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1TeacherFlatBlindEnvCfg(G1TeacherTerrainBodyEnvCfg):
    """Blind body-tracking teacher. Pair with a flat-only clip library."""

    observations: G1TeacherFlatBlindObservationsCfg = G1TeacherFlatBlindObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = ACTION_RATE_WEIGHT


@configclass
class G1TeacherFlatBlindEnvCfgPlay(G1TeacherTerrainBodyEnvCfgPlay):
    """Play variant: one env per tile, no observation corruption."""

    observations: G1TeacherFlatBlindObservationsCfg = G1TeacherFlatBlindObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = ACTION_RATE_WEIGHT
        # The play base turns corruption off on both actor groups; do the same here, since
        # this class re-instantiates ``observations`` and would otherwise get the training
        # default back.
        self.observations.policy.enable_corruption = False
        self.observations.reference.enable_corruption = False
