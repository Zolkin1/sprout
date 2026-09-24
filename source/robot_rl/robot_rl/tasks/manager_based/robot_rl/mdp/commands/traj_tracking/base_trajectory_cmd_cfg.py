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

    eta_scales: dict[str, float] | None = None
    """Per-class unit conversion applied to the CLF's channels (see :class:`CLF`).

    V adds up metres, radians, m/s and rad/s as if they were one quantity; these are the
    factors that make them commensurable, keyed by ``CLF.ETA_SCALE_CLASSES``.  Each class's
    share of V moves as the SQUARE of its factor.  ``None`` is the historical all-1.0.
    """
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

    log_per_channel_metrics: bool = False
    """If True, log a per-output-channel absolute tracking error every step.

    Off by default: it is one tensor per channel per step (292 with root-frame body tracking
    on), which is heavy logging traffic for something only useful when chasing one specific
    channel.  The per-subgroup CLF energies (``v_pelvis_pos``, ``v_joint_pos``,
    ``v_root_body_pos``, ...) are always logged and cover the usual need."""

    strict_fingerprint: bool = True
    """If True, a trajectory/clip library fingerprint mismatch at construction *raises*
    instead of warning -- a guardrail for deploy / cross-machine runs where the loaded
    library must match the trained-against one.
    ``env.commands.traj_ref.strict_fingerprint=false`` downgrades it to a warning.

    **True is the intent, but as of 2026-09-10 this flag is UNREACHABLE on every current
    task, in both directions.** Both call sites need a *trained-against* fingerprint to
    have been injected into this cfg, and on the clip stack neither is:

    - :meth:`BaseTrajectoryCommand._record_library_fingerprint` reads
      ``self.manager.library_fingerprint``, and the clip tasks' manager is
      ``MotionClipManager``, which has no such attribute -- so it returns on its first
      line. ``play.py`` does inject ``cfg.library_fingerprint`` from the saved
      ``env.yaml``, but that injection is dead code for this stack.
    - :meth:`MotionSequenceCommand._record_clip_fingerprint` compares
      ``cfg.clip_fingerprint``, which **nothing injects** -- it is only ever written BACK
      after a load. So ``trained is None`` on every run.

    The deployed teacher ``2026-08-20_16-36-10_mode11_decor_plan_v1`` recorded
    ``strict_fingerprint: true``, passed on its command line; that was already a no-op.
    The default is flipped so the stated intent lives in the repo rather than in a shell
    history, and so wiring the injection later needs no second decision.

    **Wiring it is NOT free, which is why it is not done here.** Injecting
    ``clip_fingerprint`` at play time the way ``library_fingerprint`` is injected would
    make every deliberate cross-library run fatal -- the box2 / varied / indist / curb
    evals all run the deployed tracker against a course library it never trained on, by
    design, and so does ``fm_debug.py``. Those paths need an explicit exemption before the
    guard can be turned on for real."""
