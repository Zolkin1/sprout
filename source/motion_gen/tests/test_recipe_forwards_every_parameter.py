# Copyright 2026 Zachary Olkin. All rights reserved.

"""A distribution parameter must live in the RECIPE, and the driver must forward it.

``configs/clip_gen/*.yaml`` claims to BE the library's definition: "every distribution and scale knob
lives here, nothing is defaulted behind your back... a default declared in two places is a default
that drifts". That claim was false for 22 parameters -- the tile width and step-count ranges, the
boundary-gate tolerances, which gait phase the terrain entry hands off at, the whole splice
configuration, the pure-stair tile geometry -- every one of which reached a build only through a
stage script's own argparse default.

This is the THIRD recorded instance of the flag-forwarding trap here. CLAUDE.md records it twice for
the eval harness ("A flag added to the parser is NOT forwarded to the cell subprocess unless
``stage_grid`` restates it... Measured twice"), where the fix is a heuristic test of this shape.

Two directions, and the second is the one that catches a regression:

* every recipe key is READ by the driver -- a key nobody reads is decoration;
* every stage flag is either FORWARDED by the driver or named here with a reason.

Both are heuristics over source text, which is stated plainly because it bounds what they can catch:
a key consumed by some other route (built into a dict, or read back from ``jobs.json``) is invisible.
The allow-lists are therefore part of the test -- adding to one is a decision someone writes down.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest


def _repo_root() -> pathlib.Path:
    """The checkout this test lives in (searched, not counted in ``..`` hops -- see CLAUDE.md)."""
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "configs" / "clip_gen").is_dir() and (parent / "scripts" / "rsl_rl").is_dir():
            return parent
    raise RuntimeError(f"no checkout with configs/clip_gen + scripts/rsl_rl above {__file__}")


_REPO = _repo_root()
sys.path.insert(0, str(_REPO / "scripts" / "rsl_rl"))

from clip_gen_config import config_keys, load_clip_gen_config  # noqa: E402

_RSL = _REPO / "scripts" / "rsl_rl"
_DRIVER_SRC = (_RSL / "generate_mb_library.py").read_text()

#: Recipe keys the driver reads under a name other than ``cfg.<key>``, or not at all.
_KEYS_NOT_READ_DIRECTLY: dict[str, str] = {
    "extra_mix": "read as args.extra_mix on the CLI, merged into the compose --mix string",
}

#: Stage flags the driver deliberately leaves at the stage's own argparse default.
#: Anything NOT here must be forwarded, i.e. stated in the recipe.
_ARGPARSE_DEFAULT_IS_FINE: dict[str, str] = {
    # --- infrastructure / addressing, not distribution
    "--hf-repo": "HuggingFace repo id; addressing, not a distribution parameter",
    "--terrain-hf-repo": "as --hf-repo",
    "--warn-steps": "a warning threshold on clip length, not an input",
    # --- debug / single-item entry points the driver never uses
    "--families": "subset selection for debugging; the driver builds all of them",
    "--plan": "single-clip path; the driver runs the batch (--requests)",
    "--side": "single-clip path",
    "--tag": "single-clip path output naming",
    "--approach-dist": "single-clip override of the plan's own travel budget",
    "--exit-dist": "single-clip override of the plan's own travel budget",
    "--approach-start": "per-clip in batch mode: the request carries it, drawn from "
                        "steady_approach_frac",
    "--types": "compose-time filter for building a subset library by hand",
    "--num-steps": "compose-time override; the driver lets it derive from the sources",
    # --- vestigial
    "--controller": "unused; kept so Motion Bricks' build_demo_args is satisfied",
    "--allowed-mode": "unused",
    "--approach-pin": "generation pin: selects nothing now that the bounded path is the only "
                      "one, and survives only as a provenance/staleness field. "
                      "TODO(pin-vestigial)",
    "--exit-pin": "as --approach-pin",
}

_STAGE_SCRIPTS = ("mb_plan_library", "mb_terrain_segments", "mb_assemble_library",
                  "mb_pure_stair_clips", "mb_flat_clips", "mb_flat_transients",
                  "compose_terrain_library")


def _driver_flags() -> set[str]:
    return set(re.findall(r'"(--[a-z0-9-]+)"', _DRIVER_SRC))


def _stage_flags(script: str) -> list[str]:
    """Optional ``--flags`` a stage declares (required ones are always supplied).

    A ``BooleanOptionalAction`` flag is reported as both ``--x`` and ``--no-x``, because the driver
    forwards whichever matches the recipe -- ``*([] if cfg.rl_cmd_vary else ["--no-rl-cmd-vary"])``
    -- so the positive string may never appear in the driver at all. Counting only ``--x`` there
    reported a forwarded parameter as unforwarded.
    """
    src = (_RSL / f"{script}.py").read_text()
    out = []
    for flag, body in re.findall(
            r'p\.add_argument\("(--[a-z0-9-]+)"(.*?)(?=\n    p\.add_argument|\n    args =)', src, re.S):
        if "required=True" in body:
            continue
        out.append(("--no-" + flag[2:], flag) if "BooleanOptionalAction" in body else flag)
    return out


@pytest.mark.parametrize("key", sorted(config_keys()))
def test_every_recipe_key_is_read_by_the_driver(key):
    """A recipe key nobody reads is decoration -- and reads worse than a missing key, because it
    looks like it is doing something."""
    if key in _KEYS_NOT_READ_DIRECTLY:
        return
    assert f"cfg.{key}" in _DRIVER_SRC, (
        f"recipe key {key!r} is declared in the schema but generate_mb_library.py never reads "
        f"cfg.{key}. Forward it to the stage that consumes it, or record why not in "
        f"_KEYS_NOT_READ_DIRECTLY."
    )


@pytest.mark.parametrize("script", _STAGE_SCRIPTS)
def test_every_stage_flag_is_forwarded_or_allow_listed(script):
    """THE regression test: a new stage flag must be stated in the recipe, or excused here.

    This is the direction that bites. A flag added to a stage script works when that script is run
    by hand, and then silently takes its argparse default on every driver run -- which is how
    ``entry_pick``, the tile width and step-count ranges and the whole splice configuration came to
    be set from a second place.
    """
    fwd = _driver_flags()
    def _ok(f):
        alts = f if isinstance(f, tuple) else (f,)
        return any(a in fwd or a in _ARGPARSE_DEFAULT_IS_FINE for a in alts)

    missing = [f for f in _stage_flags(script) if not _ok(f)]
    assert not missing, (
        f"{script}.py declares {missing} which generate_mb_library.py never passes, so a driver run "
        f"takes that script's argparse default. Add the parameter to configs/clip_gen/*.yaml and "
        f"forward it, or add it to _ARGPARSE_DEFAULT_IS_FINE with the reason."
    )


def test_the_allow_lists_have_no_stale_entries():
    stale_keys = set(_KEYS_NOT_READ_DIRECTLY) - set(config_keys())
    assert not stale_keys, f"_KEYS_NOT_READ_DIRECTLY names non-keys: {sorted(stale_keys)}"
    declared = {a for s in _STAGE_SCRIPTS for f in _stage_flags(s)
                for a in (f if isinstance(f, tuple) else (f,))}
    stale_flags = set(_ARGPARSE_DEFAULT_IS_FINE) - declared
    assert not stale_flags, (
        f"_ARGPARSE_DEFAULT_IS_FINE names flags no stage declares any more: {sorted(stale_flags)}")


@pytest.mark.parametrize("recipe", ["g1_mb_v6.yaml", "quick.yaml"])
def test_the_shipped_recipes_state_every_key(recipe):
    """Strict loading raises on a missing key, so this passing says the two tracked recipes were
    updated together with the schema. They have fallen out of step before, the other way round:
    ``--quick`` was once a FLAG that mutated values in code and silently missed every knob added
    after it was written."""
    cfg = load_clip_gen_config(str(_REPO / "configs" / "clip_gen" / recipe))
    for key in config_keys():
        assert hasattr(cfg, key), key


def test_the_inert_pin_knobs_are_gone():
    """The knobs that did nothing must not come back.

    ``pin_cmd`` / ``ramp_s`` / ``pin_lead`` / ``bounded_approach`` were recipe keys forwarded to
    ``mb_terrain_segments``, where they selected an approach generator UNREACHABLE on every driver
    run: every family walks, both travel budgets are positive, and both recipes turned the bounded
    path on. So ``pin_cmd: ramp`` -- with nine lines of measured justification above it -- changed
    nothing, and the -6.8 -> -2.7 m/s^2 improvement it is credited with came from the bounded
    rewrite instead.
    """
    dead = {"pin_cmd", "ramp_s", "pin_lead", "pin_tail", "pin_place", "pin_dist", "bounded_approach"}
    present = dead & set(config_keys())
    assert not present, f"inert knobs are back in the recipe schema: {sorted(present)}"
    seg = (_RSL / "mb_terrain_segments.py").read_text()
    for fn in ("def _generate_approach(", "def _generate_exit("):
        assert fn not in seg, f"the unbounded generator {fn!r} is back in mb_terrain_segments.py"
