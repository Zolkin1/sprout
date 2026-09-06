# Copyright 2026 Zachary Olkin. All rights reserved.

"""Environment classes that need more than a manager config.

Everything else in this package is configured rather than subclassed --
``entry_point="isaaclab.envs:ManagerBasedRLEnv"`` -- so a class here has to justify itself.
:class:`~.adjuster_env.AdjusterEnv` does: its agent acts once per re-plan interval while the
robot is driven at the control rate by a frozen policy, so one env step spans many control
steps. That is a change to the step LOOP, which no config expresses.

**This module deliberately does NOT import the env class**, and the rule is the same one
``terrains/__init__.py`` states. ``robot_rl/__init__.py`` runs ``import_packages``, which
imports every subpackage's ``__init__`` at ``import robot_rl`` time -- i.e. before
``AppLauncher`` -- and the concrete ``ManagerBasedRLEnv`` reaches ``pxr``. Re-exporting the
class here bound pip's ``usd-core`` ahead of Kit's and made *every* task in the repo
unlaunchable, `import robot_rl` included; ``tests/test_no_pre_app_pxr_import.py`` is what
caught it.

Gym needs no help from us: an ``entry_point`` is a ``"module:Class"`` STRING that gym resolves
when the env is actually made, which is after Kit is up. The CONFIG is re-exported because
hydra does need it at compose time, and it costs nothing -- ``ManagerBasedRLEnvCfg`` is
``pxr``-free where the env class is not.
"""

from .adjuster_env_cfg import AdjusterEnvCfg
from .token_adjuster_env_cfg import TokenAdjusterEnvCfg
from .token_planner_env_cfg import TokenPlannerEnvCfg

__all__ = ["AdjusterEnvCfg", "TokenAdjusterEnvCfg", "TokenPlannerEnvCfg"]
