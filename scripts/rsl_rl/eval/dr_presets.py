# Copyright 2026 Zachary Olkin. All rights reserved.

"""Domain-randomization presets: ``none``, ``train``, ``ood``, ``ood_push``.

Each preset is an **authoritative specification over the whole DR term set**, applied the
same way whichever task flavour the cell started from.  That is not fussiness: the tracker
task exists only in a ``-play`` flavour whose trims are PARTIAL -- they null the reset-mode
ground friction, both actor groups' observation corruption and the six camera terms, while
leaving all seven ``mode="startup"`` physical terms firing.  So neither "leave the cfg alone"
nor "use the play variant" means what it looks like, in either direction.

The ``train`` preset is built by deep-copying terms off the canonical event cfg classes
rather than transcribing their numbers, so a change to training's DR automatically changes
what "in-distribution" means here instead of silently diverging from it.

Two things this module deliberately does NOT reuse from
``scripts/rsl_rl/perturbation_scheduler.py``:

* ``_widen`` itself.  It is ``c +- scale*(c-lo)`` with no clamping, which turns
  ``static_friction (0.3, 1.6)`` into ``(-0.025, 1.925)`` and ``restitution (0.0, 0.5)`` into
  ``(-0.125, 0.625)``.  It also tests ``len(value) == 2``, so the camera extrinsics ranges --
  3-tuples of per-axis half-widths -- pass through **untouched**, which would leave the whole
  extrinsics axis identical between ``train`` and ``ood`` with no sign that anything was
  skipped.  Only the arithmetic is borrowed, through :func:`widen_range`.
* ``_apply_ood_dr``'s error handling.  It ``continue``s past an unresolvable term name and
  swallows every exception ("DR is best-effort; never crash recording").  For an eval a
  silently skipped term means a cell labelled OOD that is not OOD, so everything here is
  strict.

The floor is 0 for non-negative quantities, and the real safeguard is an assertion that a
widened range never narrows.  A *positive* floor is wrong: ``viscous_friction`` trains on
(0.01, 0.1), so a 0.05 floor would raise its OOD low end above training's and produce a
narrower range wearing the OOD label.  The assertion catches that; a floor cannot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

#: The seven startup terms plus the reset-mode ground material.
PHYSICAL_DR_TERMS: tuple[str, ...] = (
    "joint_friction_params",
    "joint_armature_params",
    "gain_randomization",
    "add_torso_mass",
    "add_joint_default_pos",
    "torso_com",
    "backpack_com",
    "randomize_ground_contact_friction",
)

#: Camera DR, all reset-mode, present only on the depth tasks.
DEPTH_DR_TERMS: tuple[str, ...] = (
    "randomize_camera_extrinsics_near",
    "randomize_camera_extrinsics_far",
    "randomize_camera_intrinsics_near",
    "randomize_camera_intrinsics_far",
    "resample_depth_noise_near",
    "resample_depth_noise_far",
)

#: The physical terms that are SAFE to re-draw on every reset, and therefore the ones that
#: give each episode an independent DR draw.  Each reads its pristine default afresh (the two
#: friction/armature terms read ``asset.data.default_*`` every call; the gain and mass terms
#: snapshot theirs and restore before drawing), so N applications have the same marginal as
#: one.
#:
#: The three that are NOT here stay ``startup``, for two different reasons, both verified:
#:
#: * ``add_joint_default_pos`` adds to the LIVE ``default_joint_pos`` buffer, so repeated
#:   application is a random walk rather than a re-draw -- it would drift without bound.
#: * ``torso_com`` / ``backpack_com`` call ``randomize_rigid_body_com``, which reads the live
#:   CoM and then hands the FULL array to ``set_coms_index`` alongside a SUBSET of env ids.
#:   That is fine at startup, where every env resets at once, and raises on a partial reset
#:   (``Shape mismatch: torch.Size([52, 39, 7]) != (1, 39, 7)``).
#:
#: Consequence, recorded per term in the manifest rather than glossed: the CoM and joint-
#: calibration draws stay fixed per env for a whole cell, so those two axes remain confounded
#: with whichever clips that env happens to run.  The other five re-draw per episode.
RESET_SAFE_PHYSICAL_TERMS: tuple[str, ...] = (
    "joint_friction_params",
    "joint_armature_params",
    "gain_randomization",
    "add_torso_mass",
    "randomize_ground_contact_friction",   # already reset-mode in training
)

#: Present but with zero ranges, so it randomizes nothing.  Listed so the term-set check
#: does not flag it as an unexpected extra.
INERT_DR_TERMS: tuple[str, ...] = ("base_external_force_torque",)

#: Nulled by ``G1TeacherEnvCfg.__post_init__``, so pushes are NOT in the training
#: distribution.  Only ``ood_push`` restores them, and that is a separate preset rather than
#: a flag, so a cell key never means two different things.
OFF_IN_TRAINING: tuple[str, ...] = ("push_robot",)

PRESETS: tuple[str, ...] = ("none", "train", "ood", "ood_push")

#: Observation groups whose corruption the presets drive.  The critic group is always clean
#: and is absent from the tracker task altogether.
CORRUPTED_OBS_GROUPS: tuple[str, ...] = ("policy", "reference")


class DrError(RuntimeError):
    """A DR preset could not be applied or verified exactly as specified."""


def widen_range(lo: float, hi: float, scale: float, *,
                clamp_min: float | None = None,
                clamp_max: float | None = None) -> tuple[float, float]:
    """Widen ``(lo, hi)`` about its midpoint by ``scale``, clamped, never narrowing.

    Args:
        lo: Lower bound.
        hi: Upper bound.
        scale: Widening factor; 1.0 is a no-op.
        clamp_min: Physical floor, e.g. 0 for a friction coefficient.
        clamp_max: Physical ceiling, e.g. 1 for a probability.

    Returns:
        The widened, clamped range.

    Raises:
        ValueError: If ``scale`` is below 1, if ``lo > hi``, or if clamping would make the
            result NARROWER than the input on either side -- which means the clamp is wrong
            for this quantity, not that the range is unusual.
    """
    lo, hi, scale = float(lo), float(hi), float(scale)
    if hi < lo:
        raise ValueError(f"range ({lo}, {hi}) is inverted")
    if scale < 1.0:
        raise ValueError(f"ood scale must be >= 1.0 (1.0 = no widening), got {scale}")
    c = 0.5 * (lo + hi)
    out_lo, out_hi = c - scale * (c - lo), c + scale * (hi - c)
    if clamp_min is not None:
        out_lo = max(out_lo, clamp_min)
    if clamp_max is not None:
        out_hi = min(out_hi, clamp_max)
    eps = 1e-12
    if out_lo > lo + eps or out_hi < hi - eps:
        raise ValueError(
            f"widening ({lo}, {hi}) by {scale} with clamps "
            f"[{clamp_min}, {clamp_max}] gave ({out_lo}, {out_hi}), which is NARROWER than "
            "the training range. An OOD range must contain the training one; the clamp is "
            "wrong for this quantity."
        )
    return out_lo, out_hi


def scale_all(values, scale: float, *, clamp_min: float | None = 0.0) -> tuple[float, ...]:
    """Multiply every entry of a per-axis half-width tuple.

    Needed because ``_EXTRINSICS_POS_RANGE = (0.01, 0.01, 0.01)`` is three half-widths, not
    a ``(lo, hi)`` pair.  A generic ``len(value) == 2`` widener passes it through unchanged,
    so the camera extrinsics would be identical in ``train`` and ``ood`` with nothing to
    show that a whole axis had been skipped.
    """
    scale = float(scale)
    out = tuple(float(v) * scale for v in values)
    if clamp_min is not None:
        out = tuple(max(v, clamp_min) for v in out)
    return out


def scale_scalar(value: float, scale: float, *, clamp_max: float | None = None) -> float:
    """Multiply a lone scalar.

    A scalar has no midpoint to expand about, so it widens MULTIPLICATIVELY.  That is a
    different operation from range widening and is reported as such rather than conflated
    with it.
    """
    out = float(value) * float(scale)
    if clamp_max is not None:
        out = min(out, clamp_max)
    return out


@dataclass(frozen=True)
class Rule:
    """How one DR parameter is widened for the ``ood`` presets."""

    kind: str                     # widen | widen_dict | scale_all | scale_scalar | widen_sym
    clamp_min: float | None = None
    clamp_max: float | None = None


#: ``term name -> {param path -> Rule}``.  Paths address the term's ``params`` dict, and may
#: reach into a nested cfg object with dots (``noise_cfg.dropout_prob``).
#:
#: Declared rather than inferred, because three of these get the wrong answer from any
#: generic recursion: the camera 3-tuples (see :func:`scale_all`), the ``DepthNoiseCfg``
#: object (not a tuple at all, so it is skipped entirely), and its two bare scalars.
OOD_RULES: dict[str, dict[str, Rule]] = {
    "joint_friction_params": {
        "static_friction_distribution_params": Rule("widen", clamp_min=0.0),
        "dynamic_friction_distribution_params": Rule("widen", clamp_min=0.0),
        "viscous_friction_distribution_params": Rule("widen", clamp_min=0.0),
    },
    "joint_armature_params": {
        "armature_distribution_params": Rule("widen", clamp_min=0.0),
    },
    "gain_randomization": {
        "stiffness_distribution_params": Rule("widen", clamp_min=0.0),
        "damping_distribution_params": Rule("widen", clamp_min=0.0),
    },
    "add_torso_mass": {
        "mass_distribution_params": Rule("widen", clamp_min=0.0),
    },
    "add_joint_default_pos": {
        "pos_distribution_params": Rule("widen"),
    },
    "torso_com": {"com_range": Rule("widen_dict")},
    "backpack_com": {"com_range": Rule("widen_dict")},
    "randomize_ground_contact_friction": {
        "static_friction_range": Rule("widen", clamp_min=0.0),
        "dynamic_friction_range": Rule("widen", clamp_min=0.0),
        "restitution_range": Rule("widen", clamp_min=0.0, clamp_max=1.0),
    },
    **{
        name: {
            "pos_range": Rule("scale_all", clamp_min=0.0),
            "rot_range": Rule("scale_all", clamp_min=0.0),
        }
        for name in ("randomize_camera_extrinsics_near", "randomize_camera_extrinsics_far")
    },
    **{
        name: {
            "focal_length_distribution_params": Rule("widen", clamp_min=0.0),
            "principal_point_distribution_params": Rule("widen", clamp_min=0.0),
        }
        for name in ("randomize_camera_intrinsics_near", "randomize_camera_intrinsics_far")
    },
    **{
        name: {
            "noise_cfg.noise_cfg": Rule("widen_sym"),
            "noise_cfg.sigma_noise_cfg": Rule("widen_sym", clamp_min=0.0),
            "noise_cfg.max_plane_deviation": Rule("scale_scalar", clamp_min=0.0),
            "noise_cfg.dropout_prob": Rule("scale_scalar", clamp_min=0.0, clamp_max=1.0),
        }
        for name in ("resample_depth_noise_near", "resample_depth_noise_far")
    },
}

#: Params the term's implementation does not actually apply, so widening them is a no-op.
#: ``randomize_joint_parameters_multi_friction`` has the dynamic and viscous friction blocks
#: commented out, writing only static friction.  Kept in :data:`OOD_RULES` for when the
#: implementation returns, but reported as inert so the OOD arm's magnitude is not overstated.
INERT_PARAMS: dict[str, tuple[str, ...]] = {
    "joint_friction_params": (
        "dynamic_friction_distribution_params",
        "viscous_friction_distribution_params",
    ),
}


def _get_path(params: dict, path: str):
    """Follow a dotted path through dict keys then object attributes."""
    cur = params
    for i, part in enumerate(path.split(".")):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"{path!r}: no key {part!r} at depth {i}")
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                raise KeyError(f"{path!r}: no attribute {part!r} at depth {i}")
            cur = getattr(cur, part)
    return cur


def _set_path(params: dict, path: str, value) -> None:
    """Assign through a dotted path (see :func:`_get_path`)."""
    parts = path.split(".")
    cur = params
    for part in parts[:-1]:
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def apply_rule(params: dict, path: str, rule: Rule, scale: float) -> tuple:
    """Apply one :class:`Rule` in place and return ``(before, after)``.

    Raises:
        DrError: If the path is absent, or the value has a shape the rule cannot handle --
            either is a sign the cfg changed under the rule table, which must be loud.
    """
    try:
        before = _get_path(params, path)
    except KeyError as exc:
        raise DrError(
            f"OOD rule for {path!r} does not match the live config ({exc}). The DR term's "
            "params changed; update OOD_RULES rather than skipping the parameter."
        ) from exc

    if rule.kind == "widen":
        if not (isinstance(before, (tuple, list)) and len(before) == 2):
            raise DrError(f"{path!r}: 'widen' needs a (lo, hi) pair, got {before!r}")
        after = widen_range(before[0], before[1], scale,
                            clamp_min=rule.clamp_min, clamp_max=rule.clamp_max)
    elif rule.kind == "widen_dict":
        if not isinstance(before, dict):
            raise DrError(f"{path!r}: 'widen_dict' needs a dict of ranges, got {before!r}")
        after = {
            k: widen_range(v[0], v[1], scale,
                           clamp_min=rule.clamp_min, clamp_max=rule.clamp_max)
            for k, v in before.items()
        }
    elif rule.kind == "scale_all":
        if not (isinstance(before, (tuple, list)) and len(before) >= 1):
            raise DrError(f"{path!r}: 'scale_all' needs a tuple, got {before!r}")
        after = scale_all(before, scale, clamp_min=rule.clamp_min)
    elif rule.kind == "scale_scalar":
        if isinstance(before, (tuple, list, dict)):
            raise DrError(f"{path!r}: 'scale_scalar' needs a scalar, got {before!r}")
        after = scale_scalar(before, scale, clamp_max=rule.clamp_max)
    elif rule.kind == "widen_sym":
        # A NoiseCfg with n_min/n_max: widen the interval, then write both fields back.
        n_min, n_max = float(before.n_min), float(before.n_max)
        lo, hi = widen_range(n_min, n_max, scale,
                             clamp_min=rule.clamp_min, clamp_max=rule.clamp_max)
        after = copy.deepcopy(before)
        after.n_min, after.n_max = lo, hi
        _set_path(params, path, after)
        return ((n_min, n_max), (lo, hi))
    else:
        raise DrError(f"unknown OOD rule kind {rule.kind!r} for {path!r}")

    _set_path(params, path, after)
    return (before, after)


#: Module basenames a DR term's ``func`` lives in.  Used to DISCOVER the live DR term set
#: rather than trusting a hard-coded list, so a term added to training shows up here as an
#: unexpected extra instead of quietly sitting outside "in-distribution".
_DR_FUNC_MODULES: tuple[str, ...] = ("physical_randomization", "depth", "events")


def _term_module(term) -> str:
    """Basename of the module a term's ``func`` comes from (``""`` if undiscoverable)."""
    func = getattr(term, "func", None)
    mod = getattr(func, "__module__", "") or ""
    return mod.rsplit(".", 1)[-1]


def discover_dr_terms(events_cfg) -> set[str]:
    """Names of every live DR term on an events config.

    Discovered from each term's implementing module, not from a list, because the failure
    this guards against is precisely a term that exists in training and is absent from the
    eval's list.
    """
    out: set[str] = set()
    for name, term in vars(events_cfg).items():
        if term is None or name.startswith("_"):
            continue
        if name in ("reset_on_clip",):        # the clip reset is not randomization
            continue
        if _term_module(term) in _DR_FUNC_MODULES:
            out.add(name)
    return out


def canonical_events(has_cameras: bool):
    """A fresh instance of the events config training actually uses.

    Terms are deep-copied off this rather than transcribed, so a change to training's DR
    changes what ``train`` means here automatically.  Imported lazily: this module must stay
    importable (and unit-testable) without IsaacLab.
    """
    if has_cameras:
        from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (  # noqa: PLC0415
            G1TeacherTerrainDepthEventsCfg)
        return G1TeacherTerrainDepthEventsCfg()
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import (  # noqa: PLC0415
        G1ClfTrackingEventsCfg)
    return G1ClfTrackingEventsCfg()


def apply_dr_preset(
    env_cfg,
    preset: str,
    *,
    ood_scale: float = 1.5,
    ood_depth_noise: bool = True,
    dr_resample: str = "reset",
    has_cameras: bool | None = None,
) -> dict:
    """Install ``preset`` on ``env_cfg.events``, before ``gym.make``.

    Applied to the config rather than per reset because seven of the eight physical terms
    are ``mode="startup"``: re-applying them at reset through a generic helper would be a
    DIFFERENT distribution, not a wider one.  Per-episode re-draw is arranged separately, by
    moving those terms to ``mode="reset"`` (see ``dr_resample``), which is a one-field change
    because ``min_step_count_between_reset`` defaults to 0 and the event manager then applies
    them on every reset.

    Args:
        env_cfg: The environment config, mutated in place.
        preset: One of :data:`PRESETS`.
        ood_scale: Widening factor for the ``ood`` presets.
        ood_depth_noise: Widen the depth sensor model too.  On by default: depth is the
            exteroception the policy actually deploys on, so its margin is the interesting
            one.
        dr_resample: ``reset`` re-draws every physical term each episode (independent draws
            from the same marginal training uses, which is what makes R the DR sample size);
            ``startup`` keeps training's one-draw-per-env and leaves the draw confounded with
            whatever clips that env runs.
        has_cameras: Override the camera detection; ``None`` infers it from the scene.

    Returns:
        The resolved spec, for printing and for the cell manifest.

    Raises:
        DrError: On an unknown preset or resample mode, or if a rule does not match the live
            config.
    """
    if preset not in PRESETS:
        raise DrError(f"unknown dr preset {preset!r}; expected one of {PRESETS}")
    if dr_resample not in ("reset", "startup"):
        raise DrError(f"dr_resample must be 'reset' or 'startup', got {dr_resample!r}")

    events = env_cfg.events
    if has_cameras is None:
        has_cameras = any(
            getattr(env_cfg.scene, n, None) is not None
            for n in ("depth_camera_near", "depth_camera_far")
        )

    wanted_physical = list(PHYSICAL_DR_TERMS)
    wanted_depth = list(DEPTH_DR_TERMS) if has_cameras else []
    all_names = wanted_physical + wanted_depth + list(INERT_DR_TERMS) + list(OFF_IN_TRAINING)

    resolved: dict = {
        "preset": preset, "ood_scale": float(ood_scale),
        "ood_depth_noise": bool(ood_depth_noise), "dr_resample": dr_resample,
        "has_cameras": bool(has_cameras), "obs_corruption": preset != "none",
        "terms": {}, "changed": [], "inert_params": [], "absent": [],
    }

    if preset == "none":
        for name in all_names:
            if hasattr(events, name):
                setattr(events, name, None)
        _set_corruption(env_cfg, False, resolved)
        resolved["terms"] = {}
        return resolved

    # --- train (and the base of every ood preset) ---------------------------------------
    canon = canonical_events(has_cameras)
    for name in wanted_physical + wanted_depth + list(INERT_DR_TERMS):
        src = getattr(canon, name, None)
        if src is None:
            resolved["absent"].append(name)
            continue
        setattr(events, name, copy.deepcopy(src))
    # Pushes are off in training, so they stay off unless the preset is explicitly about them.
    for name in OFF_IN_TRAINING:
        if hasattr(events, name):
            setattr(events, name, copy.deepcopy(getattr(canon, name, None))
                    if preset == "ood_push" else None)
    _set_corruption(env_cfg, True, resolved)

    # --- per-episode re-draw --------------------------------------------------------------
    if dr_resample == "reset":
        for name in wanted_physical:
            term = getattr(events, name, None)
            if term is None:
                continue
            if name in RESET_SAFE_PHYSICAL_TERMS:
                term.mode = "reset"

    # --- ood widening ---------------------------------------------------------------------
    if preset in ("ood", "ood_push"):
        for name, rules in OOD_RULES.items():
            term = getattr(events, name, None)
            if term is None:
                continue
            if not ood_depth_noise and name.startswith("resample_depth_noise"):
                continue
            for path, rule in rules.items():
                before, after = apply_rule(term.params, path, rule, ood_scale)
                entry = {"term": name, "param": path, "kind": rule.kind,
                         "before": _jsonable(before), "after": _jsonable(after)}
                if path in INERT_PARAMS.get(name, ()):
                    entry["inert"] = True
                    resolved["inert_params"].append(f"{name}.{path}")
                resolved["changed"].append(entry)

    resolved["reset_safe_terms"] = [n for n in wanted_physical
                                   if n in RESET_SAFE_PHYSICAL_TERMS]
    resolved["startup_only_terms"] = [n for n in wanted_physical
                                      if n not in RESET_SAFE_PHYSICAL_TERMS]
    for name in wanted_physical + wanted_depth + list(INERT_DR_TERMS) + list(OFF_IN_TRAINING):
        term = getattr(events, name, None)
        if term is None:
            continue
        resolved["terms"][name] = {
            "mode": getattr(term, "mode", "?"),
            "func": f"{getattr(term.func, '__module__', '?')}."
                    f"{getattr(term.func, '__name__', type(term.func).__name__)}",
            "params": {k: _jsonable(v) for k, v in term.params.items()},
        }
    return resolved


def _set_corruption(env_cfg, on: bool, resolved: dict) -> None:
    """Turn observation corruption on/off for every group that has the flag."""
    touched = []
    for group in CORRUPTED_OBS_GROUPS:
        grp = getattr(env_cfg.observations, group, None)
        if grp is not None and hasattr(grp, "enable_corruption"):
            grp.enable_corruption = bool(on)
            touched.append(group)
    resolved["obs_corruption_groups"] = touched


def _jsonable(value):
    """Best-effort JSON-friendly rendering of a cfg parameter."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    for attrs in (("n_min", "n_max", "operation"),):
        if all(hasattr(value, a) for a in attrs):
            return {a: _jsonable(getattr(value, a)) for a in attrs}
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items()
                if not k.startswith("_") and not callable(v)}
    return repr(value)


def verify_dr_terms(env, resolved: dict) -> None:
    """Check the LIVE env's DR term set against what the preset intended.

    Fails in both directions.  A missing term means the cell is less randomized than its
    label claims; an extra one means training grew a DR term the eval's list does not know
    about, so the ``train`` arm would no longer be the training distribution.

    Raises:
        DrError: On any difference.
    """
    live = discover_dr_terms(env.unwrapped.cfg.events)
    expected = set(resolved["terms"])
    missing, extra = sorted(expected - live), sorted(live - expected)
    if missing or extra:
        raise DrError(
            f"DR term set does not match preset {resolved['preset']!r}: "
            f"missing={missing} unexpected={extra}. An unexpected term means training gained "
            "randomization this eval does not model; add it to PHYSICAL_DR_TERMS / "
            "DEPTH_DR_TERMS (and OOD_RULES) rather than ignoring it."
        )


def dr_fingerprint(env) -> dict:
    """Per-env hash of the realised randomization, so the preset's claim is checkable.

    Same five quantities ``play.py --dr_fingerprint`` uses: the preset says what was asked
    for, this says what was applied.
    """
    import hashlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    robot = env.unwrapped.scene.articulations["robot"]
    out: dict = {}
    for name in ("joint_friction_coeff", "joint_armature", "joint_stiffness",
                 "joint_damping", "default_joint_pos"):
        buf = getattr(robot.data, name, None)
        if buf is None:
            continue
        arr = buf if isinstance(buf, torch.Tensor) else torch.as_tensor(buf)
        arr = arr.detach().to("cpu").to(torch.float64).numpy()
        out[name] = {
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
            "per_env_spread": float(arr.max(axis=0).max() - arr.min(axis=0).min())
            if arr.ndim == 2 else 0.0,
            "constant_across_envs": bool(arr.ndim == 2 and (np.ptp(arr, axis=0) == 0).all()),
        }
    return out


def assert_no_dr_variation(fingerprint: dict) -> None:
    """For the ``none`` preset: every env must be the same robot, exactly.

    Exact equality is right here -- with randomization off these buffers are copies of one
    authored value, not draws -- and it is the cheapest possible proof that the preset took.

    Raises:
        DrError: Naming the first buffer that varies across envs.
    """
    for name, info in fingerprint.items():
        if not info.get("constant_across_envs", True):
            raise DrError(
                f"dr_preset='none' but {name} differs across envs (spread "
                f"{info['per_env_spread']:.3g}). Some randomization term is still live; the "
                "preset did not take."
            )
