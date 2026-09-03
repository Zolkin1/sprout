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
    """

    tracker_run_dir: str = ""
    tracker_checkpoint: str = "model_19999.pt"
    tracking_spec_path: str = ""
    allow_tracker_mismatch: bool = False
    command_name: str = "traj_ref"
