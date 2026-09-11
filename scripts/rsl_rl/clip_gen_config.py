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

# key -> (kind, arity). kind: "float" | "int" | "bool" | "str" | "range" | "int_range" | "ints"
# | "choice" | "float_or_null".
# A "range" is a 2-list of floats validated as lo <= hi; "int_range" is the same for ints (a
# transposed step-count range is exactly the silent error this loader exists to catch); "ints" is a
# fixed-length int list with no ordering; "float_or_null" is a number or an explicit null, for a
# knob whose None means "derive it" rather than "unset".
_SCHEMA: dict[str, tuple] = {
    "seed": ("int", 1),
    # scale
    "clips_per_tile": ("int", 1),
    "stair_tiles": ("int", 1),
    "pure_stair_tiles": ("int", 1),
    "box_tiles": ("ints", 3),
    "flat_tiles": ("int", 1),
    # flat family
    "flat_clip_length": ("float", 1),
    # stair families
    "stair_vel_range": ("range", 2),
    "stair_allow_non_converged": ("bool", 1),
    "stair_floor_frac": ("float", 1),
    "stair_float_frac": ("float", 1),
    "stair_float_thick_range": ("range", 2),
    "pure_seconds": ("float", 1),
    # ---- pass A: terrain tile geometry + boundary-frame selection.
    # These shape the tile POPULATION and the gait phase every clip hands off at, so they are
    # distribution parameters in the same sense the velocity bands are. They reached a build only
    # through mb_plan_library's own argparse defaults, which is what this file exists to prevent.
    "n_steps_range": ("int_range", 2),
    "tile_width_range": ("range", 2),
    "entry_pick": ("choice", ("first", "middle", "last", "sample")),
    "exit_pick": ("choice", ("first", "middle", "last")),
    "contact_tol": ("float", 1),
    "swing_clear": ("float_or_null", 1),
    "swing_band": ("float_or_null", 1),
    "footprint_margin": ("float", 1),
    # ---- the approach/exit TRAVEL budget, in metres.
    # It used to be a duration per TILE, and the tile's platforms were then sized as
    # duration x the tile's FASTEST clip -- so a 3.5 s budget on a 2.6 m/s stair tile authored a
    # 10.1 m platform on both ends of every transition tile, and the seam sat wherever that
    # happened to put it. Distance is the quantity the geometry and the approach zone are both in.
    #
    # The band is still stated in SECONDS and converted at each clip's own speed, then clamped in
    # metres: a fixed distance would make a slow clip take proportionally longer (6 m at 0.2 m/s is
    # 30 s), and a fixed duration is what produced the speed-dependent geometry. This is the one
    # place a duration is still the natural unit, and it never reaches the tile.
    "approach_flat_s_range": ("range", 2),
    "exit_flat_s_range": ("range", 2),
    "flat_dist_clamp": ("range", 2),
    "box_top_dist_clamp": ("range", 2),
    "dist_step": ("float", 1),
    "platform_buffer_m": ("float", 1),
    # ---- pass C: the splice
    "seams": ("choice", ("ease", "weld", "auto")),
    "seam_blend_s": ("float", 1),
    "weld_search": ("int", 1),
    "arrival_search": ("int", 1),
    # ---- pure stair tiles. SEPARATE keys, because the matcher-free path ships DIFFERENT
    # geometry defaults from the transition path -- width (1.0, 1.6) against (0.8, 1.2) and
    # (4, 10) steps against (3, 7). One shared key would have silently changed one of them.
    "pure_end_buffer": ("int", 1),
    "pure_tile_width_range": ("range", 2),
    "pure_n_steps_range": ("int_range", 2),
    "pure_footprint_margin": ("float", 1),
    # ---- flat family assembly
    "flat_seam_blend_s": ("float", 1),
    "flat_anchor": ("str", 1),
    # ---- segment generation
    "pin_future": ("choice", ("auto", "on", "off")),
    # box families
    "box_vel_range": ("range", 2),
    "box_top_vx_range": ("range", 2),
    "box_two_sided_frac": ("float", 1),
    "box_nav_width": ("float", 1),
    "backward_vx_max": ("float", 1),
    # shared terrain
    "steady_approach_frac": ("float", 1),
    # BOUNDED approach: an exact looped stride covers the tile's travel budget and Motion Bricks only
    # bridges the last stretch. The duration-driven run-up it replaces did not track its command
    # (measured 78% overspeed), which is how 41% of jump_on walked off its platform.
    # FLAT via exact looped strides + bounded Motion Bricks transients. The old path had Motion Bricks
    # generate the whole clip, which tracked the commanded speed to ~0.10 m/s; the loops make steady
    # state exact and leave the model only the transitions.
    "flat_loops": ("bool", 1),
    "flat_hold_s_range": ("range", 2),
    # The driver's ENVELOPE: one continuous range per channel. Replaces the six mode/band knobs
    # (walk/run/back vx, side_vy, turn_max, run_frac) whose structure showed up in the output as
    # cliffs at 0.73 and 0.82 m/s and a 3.3x density step at 1.0.
    "flat_vx_range": ("range", 2),
    "flat_vy_range": ("range", 2),
    "flat_wz_range": ("range", 2),
    "flat_mass_stand": ("float", 2),
    "flat_mass_sidestep": ("float", 2),
    "flat_strata_width": ("float", 2),
    "flat_speed_quota": ("choice", ("stratified", "iid")),
    # Probability a flat command segment is a full stop.  A standing HOLD (the bank has true
    # zero-velocity strides), entered via an ordinary in-betweening -- NOT the standing reset a clip
    # used to open from.
    "flat_ramp_dv": ("float", 1),
    "flat_a_max": ("float", 1),
    "flat_alpha_max": ("float", 1),
    "flat_min_transient_s": ("float", 1),
    "flat_hop_slack": ("float", 1),
    "approach_a_max": ("float", 1),
    "approach_min_transient_s": ("float", 1),
    "approach_hop_slack": ("float", 1),
    "gen_attempts": ("int", 1),
    "terrain_cmd_changes_per_side": ("int", 1),
    "terrain_cmd_segments": ("int", 1),
    # How the terrain BOUNDARY transients are generated. See mb_env.walk_control(free_root) and
    # mb_terrain_segments --token-band; both were debug knobs until the seam work settled them.
    "terrain_transient_tokens": ("int", 1),
    "terrain_alpha_max": ("float", 1),
    "terrain_free_root": ("bool", 1),
    "terrain_exit_swing_clear": ("float", 1),
    # The PURE-stair families get their own three, because they are the only place lateral/yaw on
    # terrain is kept. A pure tile is 100% terrain with no approach to route, so a lateral command
    # there cannot walk the robot off an entry pose -- and a transition clip carrying none means the
    # confound (lateral commanded <=> terrain underfoot) does not exist inside it at all.
    "pure_cmd_segments": ("int", 1),
    "pure_cmd_vy_range": ("range", 2),
    "pure_cmd_wz_range": ("range", 2),
    "pure_cmd_p_straight": ("float", 1),
    "terrain_cmd_vy_range": ("range", 2),
    "terrain_cmd_wz_range": ("range", 2),
    "min_hold_m": ("float", 2),
    # RL-visible command
    "segment_reuse": ("bool", 1),
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
    if kind == "float_or_null":
        if val is None:
            return None
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{where}: expected a number or null, got {val!r}")
        return float(val)
    if kind == "int_range":
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"{where}: expected a [min, max] pair, got {val!r}")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in val):
            raise ValueError(f"{where}: expected two integers, got {val!r}")
        lo, hi = int(val[0]), int(val[1])
        if lo > hi:
            raise ValueError(f"{where}: min {lo} > max {hi} -- bounds transposed?")
        return [lo, hi]
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
