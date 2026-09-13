#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Schema + STRICT loader for a fine-tune course config (``configs/finetune/*.yaml``).

``run_finetune_loop.sh`` is configured entirely by environment variables with ``${VAR:-default}``
fallbacks. With one course that was fine. With four it has two failure modes that a tracked file
removes:

* **Nothing pairs a library with its goal regions.** ``CLIP`` and ``GOALS`` are independent
  variables, so a half-updated command runs the up-into-down library against the two-box regions
  and reports a rate. The tolerances are per-FAMILY and the family names differ, so the mismatch
  lands on ``default`` and the run looks ordinary.
* **A typo is a default.** ``REFRESH=1O00`` (letter O) is not an error in shell; it is an empty
  expansion and the default. Loading here is strict in BOTH directions: unknown keys and missing
  keys are both errors.

Same shape and the same reasoning as ``scripts/rsl_rl/clip_gen_config.py``, which exists for the
clip recipes for exactly these reasons -- and deliberately so, because two config systems in one
repo is how the second one ends up half-maintained.

Validation is shallow on purpose: types, arity, ``lo <= hi``, and that every stair cell is a
``[rise, run]`` pair. It catches typos and transposed bounds. It does not encode what makes a good
course; that judgement lives in the comments of the config files themselves.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import yaml

#: key -> (kind, arity), the same vocabulary ``clip_gen_config`` uses.
#: ``cells`` is this module's own addition: a list of ``[rise, run]`` pairs, which is how a
#: staircase geometry is drawn rather than two independent scalars.
_SCHEMA: dict[str, tuple] = {
    # -- identity -------------------------------------------------------------------------
    "name": ("str", 1),
    "task": ("str", 1),
    "clip": ("str", 1),
    "goals": ("str", 1),

    # -- the AWR buffer -------------------------------------------------------------------
    # Episodes held, and how many are replaced per iteration. The churn is what re-estimates
    # the advantage against a policy that has MOVED; too small and the critic is fitted to a
    # policy several iterations stale.
    "buffer": ("int", 1),
    "refresh": ("int", 1),
    "iters": ("int", 1),
    "num_envs": ("int", 1),

    # -- BALLAST: chunks from the original pre-training set, at weight 1 -------------------
    # Not graded, not fitted, not advantage-weighted -- they exist to stop the generator
    # drifting off the distribution it was pre-trained on while AWR pulls it toward one
    # course. 0 disables. ``ballast_flat_frac`` is the share drawn from FLAT windows, which
    # are the ones a terrain course contains none of.
    "ballast_chunks": ("int", 1),
    "ballast_flat_frac": ("float", 1),
    # WHICH recording the ballast comes from. Empty when ``ballast_chunks`` is 0, and
    # required otherwise: a count with no source is the shape of half-configured run this
    # loader exists to refuse, and the loop would either fail deep inside training or -- worse
    # -- fall back to a default and carry ballast from some other distribution entirely.
    "ballast_rollout": ("str", 1),

    # -- the course mix, for a multi-terrain library ---------------------------------------
    # Tile counts per type. A single-terrain config sets its own type and zeroes the rest,
    # which is what makes "config 4 is configs 1-3 plus the entry stairs" literally true in
    # the file rather than by convention.
    "tiles_boxes": ("int", 1),
    "tiles_updown": ("int", 1),
    "tiles_indist": ("int", 1),
    "tiles_entry_stairs": ("int", 1),

    # -- up-into-down geometry -------------------------------------------------------------
    # Drawn per tile from ``updown_cells``, with ``updown_measured_frac`` of tiles drawn from
    # ``updown_measured_cells`` instead. Rise and run are ONE draw because a staircase has one
    # geometry: drawing them independently produces rise/run combinations no building has.
    "updown_cells": ("cells", 0),
    "updown_measured_cells": ("cells", 0),
    "updown_measured_frac": ("float", 1),
    "updown_steps_range": ("int_range", 2),
    "updown_width_range_m": ("range", 2),
    "updown_platform_range_m": ("range", 2),
    "updown_wall_height_range_m": ("range", 2),

    # -- box geometry ----------------------------------------------------------------------
    "boxes_per_tile_range": ("int_range", 2),
    "box_height_range_m": ("range", 2),
    "box_depth_range_m": ("range", 2),
    "box_width_range_m": ("range", 2),
    "box_gap_range_m": ("range", 2),

    # -- shared ----------------------------------------------------------------------------
    "commanded_speed_range_mps": ("range", 2),
    "seed": ("int", 1),
}


def _validate(key: str, kind: str, arity, val, where: str):
    """One value against its schema entry, returning the coerced value.

    Args:
        key: The config key, for the error message.
        kind: Schema kind.
        arity: Expected length, or the choice set.
        val: The raw YAML value.
        where: Prefix naming the file and key.

    Returns:
        The coerced value.

    Raises:
        ValueError: On any type, arity or ordering violation.
    """
    if kind == "str":
        if not isinstance(val, str):
            raise ValueError(f"{where}: expected a string, got {val!r}")
        return val
    if kind == "int":
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{where}: expected an integer, got {val!r}")
        return int(val)
    if kind == "float":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{where}: expected a number, got {val!r}")
        return float(val)
    if kind in ("range", "int_range"):
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"{where}: expected a [min, max] pair, got {val!r}")
        ints = kind == "int_range"
        for v in val:
            ok = isinstance(v, int) if ints else isinstance(v, (int, float))
            if isinstance(v, bool) or not ok:
                raise ValueError(f"{where}: expected two {'integers' if ints else 'numbers'}, "
                                 f"got {val!r}")
        lo, hi = (int(val[0]), int(val[1])) if ints else (float(val[0]), float(val[1]))
        if lo > hi:
            raise ValueError(f"{where}: min {lo} > max {hi} -- bounds transposed?")
        return [lo, hi]
    if kind == "cells":
        # A list of [rise, run] pairs. MAY be empty -- a config with no up-into-down tiles
        # states an empty list rather than omitting the key, so the schema stays total.
        if not isinstance(val, (list, tuple)):
            raise ValueError(f"{where}: expected a list of [rise, run] pairs, got {val!r}")
        out = []
        for i, cell in enumerate(val):
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise ValueError(f"{where}[{i}]: expected a [rise, run] pair, got {cell!r}")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in cell):
                raise ValueError(f"{where}[{i}]: expected two numbers, got {cell!r}")
            rise, run = float(cell[0]), float(cell[1])
            # Ordering is the check that matters here: a transposed pair is two plausible
            # numbers that build a staircase with a 0.3 m riser and a 0.15 m tread, which no
            # shape check downstream would question.
            if rise >= run:
                raise ValueError(
                    f"{where}[{i}]: rise {rise} >= run {run}. Cells are [rise, run] and every "
                    "real staircase has a run longer than its rise -- transposed?")
            out.append([rise, run])
        return out
    raise AssertionError(f"unhandled schema kind {kind!r}")


def load_finetune_config(path: str) -> SimpleNamespace:
    """Load and validate one course config.

    Args:
        path: Path to the ``.yaml``.

    Returns:
        A namespace of the validated values.

    Raises:
        SystemExit: On a missing file, unknown keys, missing keys, or a failed validation.
            Exits rather than raises because every caller is a script whose useful response is
            to stop and print, and a traceback through yaml would bury the key that is wrong.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"fine-tune config not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise SystemExit(f"fine-tune config {path}: expected a mapping at the top level")

    unknown = sorted(set(raw) - set(_SCHEMA))
    missing = sorted(set(_SCHEMA) - set(raw))
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unknown key(s): {', '.join(unknown)}")
        if missing:
            parts.append(f"missing key(s): {', '.join(missing)}")
        raise SystemExit(
            f"fine-tune config {path}: " + "; ".join(parts) +
            ".\nBoth directions are errors on purpose: an unknown key is a typo that would "
            "otherwise do nothing, and a missing key would take a default this file exists to "
            "state.")

    out = {}
    for key, (kind, arity) in _SCHEMA.items():
        try:
            out[key] = _validate(key, kind, arity, raw[key], f"fine-tune config {path}: {key}")
        except ValueError as e:
            raise SystemExit(str(e)) from None

    # Cross-key checks: the two that are wrong in a way no single value reveals.
    if out["refresh"] > out["buffer"]:
        raise SystemExit(
            f"fine-tune config {path}: refresh {out['refresh']} > buffer {out['buffer']}. The "
            "churn replaces a slice of the buffer; larger than the buffer means every episode "
            "is new every iteration, which is not a replay buffer at all.")
    if out["updown_measured_frac"] > 0.0 and not out["updown_measured_cells"]:
        raise SystemExit(
            f"fine-tune config {path}: updown_measured_frac is "
            f"{out['updown_measured_frac']} but updown_measured_cells is empty, so that "
            "fraction of tiles has nothing to draw from.")
    if not (0.0 <= out["updown_measured_frac"] <= 1.0):
        raise SystemExit(f"fine-tune config {path}: updown_measured_frac must be in [0, 1]")
    if not (0.0 <= out["ballast_flat_frac"] <= 1.0):
        raise SystemExit(f"fine-tune config {path}: ballast_flat_frac must be in [0, 1]")
    if out["ballast_chunks"] > 0 and not out["ballast_rollout"].strip():
        raise SystemExit(
            f"fine-tune config {path}: ballast_chunks is {out['ballast_chunks']} but "
            "ballast_rollout is empty. The ballast is a slice of a SPECIFIC pre-training "
            "recording -- which one is part of what the run was, and it is invisible in the "
            "buffer, the labels and the sidecar.")
    if out["ballast_chunks"] == 0 and out["ballast_rollout"].strip():
        raise SystemExit(
            f"fine-tune config {path}: ballast_rollout names "
            f"{out['ballast_rollout']!r} but ballast_chunks is 0, so nothing would be drawn "
            "from it. One of the two was meant to change.")
    if out["ballast_chunks"] == 0 and out["ballast_flat_frac"] > 0.0:
        raise SystemExit(
            f"fine-tune config {path}: ballast_flat_frac is {out['ballast_flat_frac']} with "
            "ballast_chunks 0, which declares a mixture of nothing.")
    if out["tiles_updown"] > 0 and not out["updown_cells"] and not out["updown_measured_cells"]:
        raise SystemExit(
            f"fine-tune config {path}: tiles_updown is {out['tiles_updown']} but no stair cells "
            "are declared to draw their geometry from.")
    return SimpleNamespace(**out)


def config_keys() -> list[str]:
    """Every schema key, for a test that a config file states all of them."""
    return sorted(_SCHEMA)


#: Flags ``run_finetune_loop.sh`` parses from the COMMAND LINE. It assigns these plainly
#: (``TAG=""; GPU=""; BUFFER=4000; ...``) before parsing, so an exported environment variable
#: of the same name is OVERWRITTEN and silently ignored -- unlike the rest, which it reads as
#: ``${VAR:-default}``. Passing TAG in the environment therefore does not fail loudly at the
#: point of the mistake; it fails at the loop's own guard, "REFUSING: --tag is required",
#: after the queue has already claimed a GPU and reported the job as dispatched.
_LOOP_FLAGS: dict[str, str] = {
    "name": "--tag", "buffer": "--buffer", "refresh": "--refresh", "iters": "--iterations",
}


def as_env(cfg: SimpleNamespace) -> dict[str, str]:
    """The subset ``run_finetune_loop.sh`` reads from the ENVIRONMENT.

    Deliberately excludes everything in :data:`_LOOP_FLAGS`: those are command-line flags and
    the loop clobbers any same-named environment variable before parsing. Returning them here
    is what made a queue look correctly configured and refuse on first contact.

    Args:
        cfg: A loaded config.

    Returns:
        ``{VAR: value}`` ready to export.
    """
    return {
        "TASK": cfg.task,
        "CLIP": cfg.clip,
        "GOALS": cfg.goals,
        "NENV": str(cfg.num_envs),
        "BALLAST_CHUNKS": str(cfg.ballast_chunks),
        "BALLAST_FLAT_FRAC": str(cfg.ballast_flat_frac),
        "BALLAST_ROLLOUT": cfg.ballast_rollout,
    }


def as_flags(cfg: SimpleNamespace) -> list[str]:
    """The command-line half of the same bridge, in order.

    Args:
        cfg: A loaded config.

    Returns:
        ``["--tag", "<name>", "--buffer", "<n>", ...]``.
    """
    return [str(part) for key, flag in _LOOP_FLAGS.items()
            for part in (flag, getattr(cfg, key))]


if __name__ == "__main__":  # pragma: no cover - a convenience for the shell
    import sys

    cfg = load_finetune_config(sys.argv[1])
    for k, v in as_env(cfg).items():
        print(f"{k}={v}")
