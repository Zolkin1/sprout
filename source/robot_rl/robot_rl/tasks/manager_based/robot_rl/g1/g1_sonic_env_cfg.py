# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play envs for the SONIC baseline: our terrain and clips, SONIC's robot.

These are deliberately *thin* subclasses of the teacher play envs. Everything that decides what
"tracking well" means -- the terrain importer and its clip library, the tracking command, the
terminations, the episode length, the reset placement, the height scanner -- is inherited
unchanged, so a SONIC rollout is graded by exactly the same yardstick as the teacher's. Only two
things are swapped:

* ``scene.robot`` -> SONIC's own G1 (:func:`~robot_rl.assets.robots.g1_sonic.g1_sonic_cfg`), and
* ``actions.joint_pos.scale`` -> SONIC's action scale, which is derived from that robot's PD
  gains (``0.25 * effort / stiffness`` where ours is ``0.125 *``). Mixing one robot's gains with
  the other's scale would silently halve or double the policy's effective joint range, so the two
  always travel together.

The policy itself is not part of the env: SONIC builds its own observation from the command term
and the articulation via :class:`~robot_rl.baselines.sonic.env_bridge.SonicEnvBridge`. The
inherited observation groups are still computed (they are what the recorder logs, and the critic
group is harmless), they just do not drive the actions.

Domain randomization is left as the teacher play cfg has it. For a controlled A/B use the
recorder's ``--no_dr`` flag, which nulls the startup randomization on BOTH policies at once --
in particular ``add_joint_default_pos``, which perturbs ``default_joint_pos`` and therefore
shifts both SONIC's ``joint_pos_rel`` observation and its action offset.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from robot_rl.assets.robots.g1_sonic import G1_SONIC_ACTION_SCALE, g1_sonic_cfg

from .g1_teacher_terrain_env_cfg import G1TeacherTerrainEnvCfgPlay
from .g1_tracker_env_cfg import G1FmTrackerTerrainPlayEnvCfg


def _use_sonic_robot(cfg) -> None:
    """Swap in SONIC's G1 and its matching action scale, in place."""
    cfg.scene.robot = g1_sonic_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = dict(G1_SONIC_ACTION_SCALE)


@configclass
class G1SonicTerrainPlayEnvCfg(G1TeacherTerrainEnvCfgPlay):
    """SONIC tracking the pre-made clips, on the teacher's terrain and terminations."""

    def __post_init__(self):
        super().__post_init__()
        _use_sonic_robot(self)


@configclass
class G1SonicFmTerrainPlayEnvCfg(G1FmTrackerTerrainPlayEnvCfg):
    """SONIC tracking the flow-matching generator's prediction.

    The FM reference reaches SONIC through
    :meth:`~...fm_tracking_cmd.FmTrackingCommand.get_future_joint_state`, which serves the
    predicted joint window rather than the clip's.
    """

    def __post_init__(self):
        super().__post_init__()
        _use_sonic_robot(self)
