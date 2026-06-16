# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared configuration for trajectory command terms."""

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class BaseTrajectoryCommandCfg(CommandTermCfg):
    """Base configuration shared by all trajectory command variants.

    Subclasses set ``class_type`` to point at the concrete command class.
    """

    asset_name: str = "robot"
    contact_bodies: list[str] = None
    conditioner_generator_name: str = ""
    path: str = ""
    hf_repo: str = None
    Q_weights: list[float] = None
    R_weights: list[float] = None
    resampling_time_range: tuple[float, float] = (5.0, 15.0)

    library_fingerprint: str | None = None
    """Portable content hash of the trajectory library actually loaded.

    Populated by the command at construction (see
    ``BaseTrajectoryCommand._record_library_fingerprint``) and serialised into
    ``params/env.yaml`` so a run records exactly which library version it
    trained against.  At play / resume, inject the trained-against value here
    *before* env construction (play.py does this from the saved ``env.yaml``);
    the command then warns loudly if the locally loaded library hashes
    differently.  Leave ``None`` for a fresh training run."""

    library_manifest: dict | None = None
    """Per-skill trajectory counts of the loaded library (``{skill: count}``),
    recorded alongside :attr:`library_fingerprint` for human-readable provenance
    in ``params/env.yaml``."""
