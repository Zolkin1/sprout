# Copyright 2026 Zachary Olkin. All rights reserved.

"""There is ONE Motion-Bricks in-betweening generator, and cached transients go stale honestly.

Two properties, each pinned because its absence was measured rather than imagined:

* ``mb_transient.generate`` is the only implementation. It used to live inside
  ``mb_flat_transients``, and the terrain driver -- which is about to need ordinary loop-to-loop
  transitions during a clip's lead-up -- would otherwise have grown a second one. The one that
  exists does multi-hop chaining (``ramp_waypoints``); a fresh single-hop copy would not, and the
  difference is 1.70 vs 5.08 m/s^2 of peak acceleration on a large change.

* the flat pass-B cache is keyed on the GENERATION KNOBS, not on a filename existing. It was a bare
  ``os.path.exists`` skip, and the quick build's 47 cached transients were measured to differ in
  SHAPE from what the same command produces today on 38 of them, with nothing saying so.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "scripts" / "rsl_rl" / "mb_transient.py").is_file():
        _REPO = _p
        break
else:                                                          # pragma: no cover - layout guard
    raise RuntimeError("could not locate the checkout above this test")
_SCRIPTS = _REPO / "scripts" / "rsl_rl"
sys.path.insert(0, str(_SCRIPTS))


def _defs(path: pathlib.Path) -> set[str]:
    """Names defined at module scope in ``path``."""
    tree = ast.parse(path.read_text())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_the_transient_generator_lives_in_exactly_one_module():
    shared = _defs(_SCRIPTS / "mb_transient.py")
    assert {"generate", "ramp_waypoints", "seed_overlap"} <= shared

    flat = _defs(_SCRIPTS / "mb_flat_transients.py")
    reinvented = flat & {"_generate", "generate", "_ramp_waypoints", "ramp_waypoints",
                         "_seed_overlap", "seed_overlap", "_min_tokens_for"}
    assert not reinvented, (
        f"mb_flat_transients re-defines {sorted(reinvented)}; it must call mb_transient instead, or "
        f"the two copies drift and only one of them chains hops")


def test_the_flat_transient_cache_is_keyed_on_the_generation_knobs():
    """Every flag that changes what pass B PRODUCES must be in ``_GEN_KNOBS``.

    Heuristic, and deliberately so: it reads the flags ``mb_flat_transients`` forwards into
    ``mb_transient.generate`` out of the source, so a new knob wired straight through is caught while
    one read via ``getattr`` is not. Same shape as the eval harness's forwarding test, and the same
    caveat.
    """
    import mb_flat_transients as ft

    src = (_SCRIPTS / "mb_flat_transients.py").read_text()
    tree = ast.parse(src)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "generate")
    # `a_max=args.a_max` -> "a_max"; the arg NAME on the args namespace is what the knob dict holds.
    forwarded = {kw.value.attr for kw in call.keywords
                 if isinstance(kw.value, ast.Attribute) and isinstance(kw.value.value, ast.Name)
                 and kw.value.value.id == "args"}
    assert forwarded, "could not find the generate() call's args.* keywords"
    missing = forwarded - set(ft._GEN_KNOBS)
    assert not missing, (
        f"{sorted(missing)} reach the generator but are not in _GEN_KNOBS, so moving one silently "
        f"reuses every cached transient")


def test_gen_params_records_exactly_the_knobs_that_are_compared():
    import mb_flat_transients as ft

    args = type("A", (), {"stride_bank": "/tmp/bank.npz", "a_max": 2.0, "alpha_max": 2.0,
                          "min_transient_s": 0.35, "ramp_dv": 1.0, "hop_slack": 1.3})()
    want = ft._gen_params(args)
    assert set(want) == set(ft._GEN_KNOBS), (
        "_gen_params and _GEN_KNOBS disagree; a knob recorded but not compared is inert, and one "
        "compared but not recorded marks every segment stale forever")
    # BASENAME, so moving the work dir does not invalidate the whole cache.
    assert want["stride_bank"] == "bank.npz"


def test_a_cache_with_no_gen_version_is_stale():
    """The pre-mechanism files must not compare equal. They predate the token upper bound."""
    import mb_flat_transients as ft

    class _Raw:
        @staticmethod
        def load_raw_motion(_p):
            return type("M", (), {"provenance": {"stride_bank": "bank.npz", "a_max": 2.0,
                                                 "alpha_max": 2.0, "min_transient_s": 0.35,
                                                 "ramp_dv": 1.0, "hop_slack": 1.3}})()

    want = dict.fromkeys(ft._GEN_KNOBS, 0.0)
    want.update({"gen_version": ft._GEN_VERSION, "stride_bank": "bank.npz", "a_max": 2.0,
                 "alpha_max": 2.0, "min_transient_s": 0.35, "ramp_dv": 1.0, "hop_slack": 1.3})
    why = ft._is_stale("ignored", want, _Raw)
    assert why is not None and "gen_version" in why


# --------------------------------------------------------------- hop_tokens: the horizon is the MODEL's
class _MB:
    """The four fields ``hop_tokens`` reads, at the shipped Motion Bricks values."""

    nfpt = 4
    fps = 30.0
    min_tokens = 6
    max_tokens = 16


def test_the_band_leaves_the_model_a_choice():
    """``slack`` is what makes the duration Motion Bricks' own, and it must stay a real band.

    A ``min_travel_m`` floor briefly forced the terrain entry transient to span the 2 m approach zone,
    and to make that bite it also collapsed the band to one token count. Both are gone: a clip
    commanded 0.2 m/s into a 0.30 m/s stair entry has |dv| = 0.1 m/s and nothing to transition, so
    covering 2 m bought 1.87 s of coasting against a pinned pose -- the exact failure ``slack`` exists
    to prevent -- and it drove every approach to the 16-token ceiling, making the latch 1.87 s x speed.
    """
    import mb_env

    n_min, n_max, _ = mb_env.hop_tokens(_MB(), (2.6, 0, 0), (0.36, 0, 0), a_max=2.0, alpha_max=2.0,
                                        floor_s=0.35, slack=1.3)
    assert n_max > n_min, "the model must be left a range of horizons to choose from"
    assert n_min >= _MB.min_tokens and n_max <= _MB.max_tokens


def test_a_small_change_gets_a_SHORT_transient():
    """The slow-stair case that the travel floor got wrong: |dv| = 0.1 m/s needs almost no time."""
    import mb_env

    n_min, _n_max, t_min = mb_env.hop_tokens(_MB(), (0.2, 0, 0), (0.3, 0, 0), a_max=2.0, alpha_max=2.0,
                                             floor_s=0.35, slack=1.3)
    assert n_min == _MB.min_tokens, "nothing to transition, so ask for the fewest tokens"
    assert t_min == pytest.approx(0.35), "the floor binds, not the velocity change"


def test_a_large_deceleration_is_bounded_by_the_change_not_by_a_distance():
    """``T >= |dv| / a_max`` is the only lower bound; the exact loop covers the travel budget."""
    import mb_env

    n_min, _n, t_min = mb_env.hop_tokens(_MB(), (3.3, 0, 0), (0.1, 0, 0), a_max=2.0, alpha_max=2.0,
                                         floor_s=0.35, slack=1.3)
    assert t_min == pytest.approx(3.2 / 2.0)
    assert n_min > _MB.min_tokens


def test_hop_tokens_takes_no_distance_argument():
    """Pinned so the travel floor is not reintroduced without reading why it went.

    The approach zone is a PLANNING quantity -- pass A reserves it beside the terrain and the clutter
    pass vetoes decoys inside it -- not a length to force on one generation.
    """
    import inspect

    import mb_env

    params = set(inspect.signature(mb_env.hop_tokens).parameters)
    assert not (params & {"min_travel_m", "min_travel", "travel_m"}), sorted(params)
