# Copyright 2026 Zachary Olkin. All rights reserved.

"""The adjuster env's CONFIG, kept apart from the env class on purpose.

Hydra composes an ``env_cfg_entry_point`` module BEFORE ``AppLauncher`` runs, and at that
point the only ``pxr`` importable is pip's ``usd-core`` rather than Kit's -- binding it makes
Kit segfault about a second into startup (see ``tests/test_no_pre_app_pxr_import.py``). The
concrete ``ManagerBasedRLEnv`` reaches that; ``ManagerBasedRLEnvCfg`` does not. So the cfg
lives here, importing only the cfg base, and :mod:`.adjuster_env` holds the class.

This split is not optional bookkeeping: ``robot_rl/__init__.py`` runs ``import_packages``,
which imports every subpackage's ``__init__``, so a package that re-exports its env class
makes ``import robot_rl`` itself unlaunchable. That is how this file came to exist.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


@configclass
class AdjusterEnvCfg(ManagerBasedRLEnvCfg):
    """A :class:`ManagerBasedRLEnvCfg` plus what the frozen tracker and the fold need.

    Attributes:
        tracker_run_dir: ``logs/rsl_rl/<experiment>/<run>`` of the tracker to drive the robot.
            Empty is refused: there is no sensible default, and a stale default path to an
            artifact is a thing this repo has watched rot silently more than once.
        tracker_checkpoint: Checkpoint file name inside that run.
        tracking_spec_path: The fitted :class:`TrackingErrorSpec`. ONE field, because the
            dense reward's kernel widths and the failure termination's bounds must come from
            one file -- the env's fold and the ``TrackingViolation`` term read the same value.
        allow_tracker_mismatch: Downgrade the tracker's observation-contract failures to
            warnings. For a deliberate cross-pair experiment only.
        command_name: The :class:`FmTrackingCommand` term name.
        steer_kp: Gain of a yaw controller pointing the robot at its goal; 0 disables.
        steer_wz_max: Clip on that controller's output (rad/s).
        zero_residual: Force the applied residual to EXACTLY zero, whatever the policy emits
            -- the bare-generator control. The served plan is then bit-identical to the
            generator's own (that identity is exact by construction; see
            ``apply_plan_residual``'s per-node ``moved`` mask, which exists so the
            Gram-Schmidt round trip cannot perturb a zero residual), so a run with it on and
            a run with it off differ in exactly one thing.

            Mainly for LOOKING: the debug ribbon draws ``_pred_window``, which
            ``apply_plan_residual`` has already modified in place, so without this the plan
            on screen is the generator's PLUS whatever the policy currently does -- early in
            training, noise with a 0.2 m position bound. Loading ``model_0.pt`` achieves the
            same thing, but only by accident of the head being zero-initialised.
    """

    steer_kp: float = 0.0
    """Proportional gain (1/s) of a yaw controller that points the robot at its goal.

    ``0.0`` (the default) leaves the command's ``wz`` channel exactly as the clip states it,
    which is what every task here has always done.

    Nothing else in this stack steers: the command is a body-frame velocity and the generator
    plans from depth, so a few degrees of accumulated heading error over several metres
    becomes a lateral miss -- and on a thin flight of stairs a lateral miss is a fall. Closing
    that one loop is the largest single effect measured on this pair to date, worth far more
    than any conditioning change tried.

    **It is OFF-DISTRIBUTION on the stair families**, whose clips carry ``|wz| max = 0.0000``
    across the whole library, so a steered stair result is a different measurement rather than
    a cleaner one. Applied to the COMMAND, so the generator's conditioning, the reward and the
    ``command_window`` observation all move together -- see
    :meth:`MotionSequenceCommand.set_yaw_rate_source`.
    """

    steer_wz_max: float = 0.68
    """Symmetric clip on the steering yaw rate (rad/s). Matches the eval's default."""

    zero_residual: bool = False
    tracker_run_dir: str = ""
    tracker_checkpoint: str = "model_19999.pt"
    tracking_spec_path: str = ""
    allow_tracker_mismatch: bool = False
    command_name: str = "traj_ref"
