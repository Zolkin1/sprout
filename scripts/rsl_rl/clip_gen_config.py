#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Schema + STRICT loader for a clip-library recipe (``configs/clip_gen/*.yaml``).

``generate_mb_library.py`` used to carry ~40 argparse flags, 28 of them distribution and scale
parameters. That had three problems this module exists to remove:

* **Defaults declared twice.** The driver redeclared ``--stair-vel-range`` / ``--box-vel-range`` and
  forwarded them explicitly, so its stale copies silently overrode the bands ``mb_plan_library``
  declared. A library's parameters now live in ONE tracked file.
* **Knobs the driver could not reach.** The flat family's velocity bands are ``mb_encode`` args the
  driver never forwarded, so the only way to change them was to edit that file's defaults -- which
  means a library built from a command line was not reproducible from it.
* **Silent defaults.** A mistyped parameter fell back to a default and produced a library that
  looked fine and trained on the wrong distribution. Loading here is strict in BOTH directions:
  unknown keys and missing keys are both errors.

Validation is deliberately shallow -- types, arity, and ``lo <= hi`` on ranges. It is here to catch
typos and transposed bounds, not to encode what makes a good library.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import yaml

# key -> (kind, arity). kind: "float" | "int" | "bool" | "str" | "range" | "ints" | "choice".
# A "range" is a 2-list validated as lo <= hi; "ints" is a fixed-length int list.
_SCHEMA: dict[str, tuple] = {
    "seed": ("int", 1),
    # scale
    "clips_per_tile": ("int", 1),
    "stair_tiles": ("int", 1),
    "pure_stair_tiles": ("int", 1),
    "box_tiles": ("ints", 3),
    "flat_tiles": ("int", 1),
    "distractor_tiles": ("int", 1),
    "distractor_s": ("float", 1),
    # flat family
    "flat_clip_length": ("float", 1),
    "flat_seg_dur_range": ("range", 2),
    "flat_walk_vx_range": ("range", 2),
    "flat_run_vx_range": ("range", 2),
    "flat_back_vx_range": ("range", 2),
    "flat_side_vy": ("float", 1),
    "flat_turn_max": ("float", 1),
    # stair families
    "stair_vel_range": ("range", 2),
    "stair_floor_frac": ("float", 1),
    "stair_float_frac": ("float", 1),
    "stair_float_thick_range": ("range", 2),
    "pure_seconds": ("float", 1),
    # box families
    "box_vel_range": ("range", 2),
    "box_approach_s_range": ("range", 2),
    "box_exit_s_range": ("range", 2),
    "box_top_vx_range": ("range", 2),
    "box_two_sided_frac": ("float", 1),
    "box_platform_buffer": ("float", 1),
    "box_nav_width": ("float", 1),
    "backward_vx_max": ("float", 1),
    # shared terrain
    "steady_approach_frac": ("float", 1),
    "pin_lead": ("float", 1),
    "pin_cmd": ("choice", ("ramp", "hold")),
    "ramp_s": ("float", 1),
    # RL-visible command
    "rl_cmd_vary": ("bool", 1),
    "rl_cmd_seg_dur_range": ("range", 2),
    "segment_reuse": ("bool", 1),
    "rl_vels": ("choice", ("raw", "snapped")),
    # composition
    "extra_mix": ("str", 1),
}


def _check(key: str, kind: str, arity, val):
    """Validate and coerce ONE config value. Raises ``ValueError`` with the key named."""
    where = f"clip-gen config key {key!r}"
    if kind == "bool":
        if not isinstance(val, bool):
            raise ValueError(f"{where}: expected true/false, got {val!r}")
        return val
    if kind == "int":
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{where}: expected an integer, got {val!r}")
        return int(val)
    if kind == "float":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{where}: expected a number, got {val!r}")
        return float(val)
    if kind == "str":
        if not isinstance(val, str):
            raise ValueError(f"{where}: expected a string, got {val!r}")
        return val
    if kind == "choice":
        if val not in arity:
            raise ValueError(f"{where}: expected one of {list(arity)}, got {val!r}")
        return val
    if kind == "ints":
        if not isinstance(val, (list, tuple)) or len(val) != arity:
            raise ValueError(f"{where}: expected {arity} integers, got {val!r}")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in val):
            raise ValueError(f"{where}: expected integers, got {val!r}")
        return [int(v) for v in val]
    if kind == "range":
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"{where}: expected a [min, max] pair, got {val!r}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in val):
            raise ValueError(f"{where}: expected two numbers, got {val!r}")
        lo, hi = float(val[0]), float(val[1])
        if lo > hi:
            raise ValueError(f"{where}: min {lo} > max {hi} -- bounds transposed?")
        return [lo, hi]
    raise AssertionError(f"unhandled schema kind {kind!r}")


def load_clip_gen_config(path: str) -> SimpleNamespace:
    """Load + validate a clip-gen recipe. Every schema key required, no extras allowed.

    Args:
        path: Path to a ``configs/clip_gen/*.yaml``.

    Returns:
        A namespace with one attribute per schema key, plus ``config_path`` (absolute) so the
        driver can record which recipe built a library.

    Raises:
        SystemExit: on a missing file, unknown keys, missing keys, or a value that fails validation.
            Loud rather than defaulted: a wrong distribution parameter is invisible in the finished
            library until someone measures it.
    """
    if not os.path.exists(path):
        raise SystemExit(f"clip-gen config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SystemExit(f"clip-gen config {path}: expected a mapping at the top level")

    unknown = sorted(set(raw) - set(_SCHEMA))
    missing = sorted(set(_SCHEMA) - set(raw))
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unknown key(s): {', '.join(unknown)}")
        if missing:
            parts.append(f"missing key(s): {', '.join(missing)}")
        raise SystemExit(
            f"clip-gen config {path} does not match the schema -- " + "; ".join(parts)
            + ".\nEvery parameter must be stated explicitly: a defaulted one is a library that "
              "trained on a distribution nobody chose. See configs/clip_gen/g1_mb_v6.yaml.")

    out = {}
    for key, (kind, arity) in _SCHEMA.items():
        try:
            out[key] = _check(key, kind, arity, raw[key])
        except ValueError as e:
            raise SystemExit(f"clip-gen config {path}: {e}") from None
    out["config_path"] = os.path.abspath(path)
    return SimpleNamespace(**out)


def config_keys() -> list[str]:
    """The schema's keys, for tests and for the driver's provenance dump."""
    return sorted(_SCHEMA)
