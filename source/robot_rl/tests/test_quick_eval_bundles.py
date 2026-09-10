# Copyright 2026 Zachary Olkin. All rights reserved.
"""The eval entry point's per-terrain bundles, read off the script itself.

``run_quick_eval.sh`` is where four terrains are made to agree, and every one of the things it
makes agree was once a difference nobody could see in the output: the training library
evaluated as if it were the eval grid, a spawn distribution that included the box tops, a
``height_tol_m`` that could not be set at all, and a commanded speed below the course's own
floor. None of that is type-checked -- it is a shell script -- and the eval had zero tests
where the fine-tune loop has twelve, so it is pinned here the same way, by parsing the text.

**Why not run it.** One cell is a simulator launch and a GPU. What can go wrong without one is
the CONFIGURATION -- a bundle pointing at a stale library, an override whose name does not
match the env cfg, a default that silently reverts an arm to the training distribution -- and
that is text.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
EVAL = _REPO / "scripts" / "rsl_rl" / "run_quick_eval.sh"

#: Every terrain the eval must offer.
TERRAINS = ("box2", "updown", "curb", "varied", "indist")


@pytest.fixture(scope="module")
def text() -> str:
    """The eval script's source."""
    if not EVAL.is_file():
        pytest.skip(f"{EVAL} not present")
    return EVAL.read_text()


def test_the_old_box2_only_script_is_gone(text: str) -> None:
    """It defaulted to a v1 library while the loop trained on v2 -- a different course."""
    assert not (_REPO / "scripts" / "rsl_rl" / "run_box2_quick_eval.sh").exists(), (
        "the box2-only eval is back; four terrains evaluated by four procedures is what "
        "this replaced")
    assert "TERRAIN=" in text


@pytest.mark.parametrize("terrain", TERRAINS)
def test_every_terrain_has_a_bundle(terrain: str, text: str) -> None:
    """A bundle is (library, goal spec, height tol, output root, build command)."""
    m = re.search(rf"^  {terrain}\)(.*?);;", text, re.S | re.M)
    assert m, f"{terrain} has no bundle"
    body = m.group(1)
    for field in ("DEF_CLIP=", "DEF_GR=", "DEF_TOL=", "DEF_ROOT=", "DEF_BUILD="):
        assert field in body, f"{terrain}'s bundle is missing {field}"


def test_an_unknown_terrain_is_refused(text: str) -> None:
    assert "REFUSING: TERRAIN must be" in text


def test_the_eval_spawns_from_the_GROUND_by_default(text: str) -> None:
    """The rule the whole eval design turns on.

    ``all`` would include the box tops and the platform between two flights -- starts that
    hand the robot the jump the course exists to measure -- and it would still produce a
    plausible number.
    """
    assert re.search(r'^SPAWN_FROM=\$\{SPAWN_FROM:-ground\}', text, re.M), (
        "the eval must default to the initial ground area on every terrain")
    assert "REFUSING: SPAWN_FROM must be ground or all" in text
    assert "env.events.reset_on_section.params.spawn_from=$SPAWN_FROM" in text, (
        "the rule is declared and never passed to the env")


def test_the_spawn_rule_and_the_speeds_are_ANNOUNCED(text: str) -> None:
    """A setting that changes what a number MEANS has to be in the run's own output.

    The same argument as `--terrain_ref` in the episode harness: the log is the only durable
    record of which arm a directory of episode tables belongs to.
    """
    assert re.search(r'echo "\[eval\][^"]*spawn[^"]*\$SPAWN_FROM', text), (
        "the banner never names the spawn rule")
    assert re.search(r'echo "\[eval\][^"]*\$SPEEDS', text)
    assert re.search(r'echo "\[eval\][^"]*height_tol=\$HEIGHT_TOL', text)


def test_the_height_tolerance_is_coerced_to_a_float(text: str) -> None:
    """IsaacLab type-checks hydra overrides.

    ``height_tol_m=0`` (int) is refused where ``0.0`` is accepted, and it fails at hydra
    composition before the GPU is touched -- two runs died on exactly this. The awk coercion
    is the fix at the point of use rather than in whichever launcher happened to set it.
    """
    assert re.search(r"HEIGHT_TOL=\$\(awk .*printf \"%\.6f\"", text), (
        "HEIGHT_TOL is not coerced to a float")
    assert "env.terminations.goal_reached.params.height_tol_m=$HEIGHT_TOL" in text


def test_no_terrain_defaults_the_tolerance_to_zero(text: str) -> None:
    """0 means OPPOSITE things at the two ends.

    It DISABLES the test in ``terminations.freeform_goal_reached`` and is an EXACT-EQUALITY
    test in ``goal_region.in_goal``, so a spec of 0 scored an entire arm at 0.0000 over 8400
    episodes -- and mistrained it, because the reward's success term went with it.
    """
    for tol in re.findall(r"^\s*DEF_TOL=(\S+)", text, re.M):
        assert float(tol) > 0.0, f"a bundle defaults height_tol to {tol}"


def test_the_library_is_required_and_never_rebuilt_here(text: str) -> None:
    """The old script rebuilt it from box2-specific flags.

    Which was wrong for every other course, and worse: a build inside an eval can overwrite
    the very grid an earlier arm was scored on, so two arms stop sharing a course.
    """
    assert "missing $CLIP -- build it with:" in text
    # The build command may only be QUOTED (as advice), never executed.
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("#") or "DEF_BUILD=" in st or "echo" in st:
            continue
        assert not re.match(r"^(\./env_isaaclab/bin/)?(python|\$PY) scripts/rsl_rl/make_", st), (
            f"the eval builds a library: {st}")


def test_the_tile_axis_is_read_off_the_library(text: str) -> None:
    """``terrain_id`` is an INDEX; a hand-passed list that did not match relabelled every cell.

    3.00 m is tile 1 on the quick grid and tile 3 on the full one, so the same list cannot be
    right for both -- and being wrong is silent, then paired against the wrong control.
    """
    assert re.search(r'DISTS=\$\{DISTS:-\$\(\$PY scripts/rsl_rl/terrain_axis\.py', text), (
        "the per-tile axis is not derived from the library")


def test_the_speeds_are_checked_against_the_librarys_own_band(text: str) -> None:
    """One shared speed list is not automatically legal on every course.

    box2's band starts at 0.5 and the up-into-down course's at 0.6, so a sweep that reached
    0.5 would be out of distribution on one of them -- a different distribution, not a harder
    cell.
    """
    assert "--check_speeds $SPEEDS" in text
    assert re.search(r"--check_speeds \$SPEEDS \|\| exit 2", text), (
        "the band check runs but its result is discarded")


def test_the_env_count_and_the_round_stop_are_stated_together(text: str) -> None:
    """NENV decides the censored fraction, so it is part of the comparison, not of the box."""
    assert re.search(r"^NENV=\$\{NENV:-\d+\}", text, re.M)
    assert "ROUNDS=$(( (NEPS + NENV - 1) / NENV ))" in text
    assert "match" in text.lower() and "censored" in text.lower()


def test_the_reducer_is_called_with_the_derived_axis(text: str) -> None:
    """An eval that collects and does not reduce leaves the answer to whoever runs it next."""
    assert re.search(r"box2_ic_report\.py \"\$OUT\" --distances \$DISTS", text)


def test_the_loop_points_at_the_new_script(text: str) -> None:
    """The loop's closing line is how anyone finds the eval at all."""
    loop = (_REPO / "scripts" / "flow_matching" / "run_finetune_loop.sh").read_text()
    assert "run_box2_quick_eval.sh" not in loop, "the loop still names the retired script"
    assert "run_quick_eval.sh" in loop
