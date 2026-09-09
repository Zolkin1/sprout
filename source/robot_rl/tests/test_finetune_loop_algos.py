# Copyright 2026 Zachary Olkin. All rights reserved.
"""The fine-tune loop's ALGO switch, read off the script itself.

The point of the switch is that three arms share every stage except the update, so an ablation
attributes a difference to the objective and to nothing else.  That property lives in a shell
script, where nothing type-checks it, and it is exactly the kind of thing that rots when a flag
is added to one branch and not the others -- so it is pinned here by parsing the script.

**Why not run the loop.**  One iteration is ~40 minutes on a GPU and needs a clip library, a
tracker checkpoint and a simulator.  What can go wrong without any of that is the
CONFIGURATION: an arm that quietly keeps a flag it should have dropped, or a percentile that
defaults to a number nobody chose.  Those are text, and text is what this reads.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
LOOP = _REPO / "scripts" / "flow_matching" / "run_finetune_loop.sh"

#: The algorithms the loop must offer, and what each one's update stage is allowed to pass.
ALGOS = ("awr", "fbc", "bc", "arrived")


@pytest.fixture(scope="module")
def text() -> str:
    """The loop script's source."""
    if not LOOP.is_file():
        pytest.skip(f"{LOOP} not present")
    return LOOP.read_text()


def _algo_args(algo: str) -> list[str]:
    """The ``ALGO_ARGS`` array the script builds for one algorithm.

    Evaluated by the SHELL rather than parsed with a regex: the array is built by a ``case``
    over quoted words, and a regex that agreed with bash on that would be a second, worse
    implementation of bash.
    """
    block = re.search(r"case \"\$ALGO\" in\n(\s*awr\) ALGO_ARGS=.*?)\n\s*esac",
                      LOOP.read_text(), re.S)
    assert block, "the ALGO_ARGS case block is not where this test expects it"
    script = (f'ALGO={algo}\nCRITIC=/c\nBETA=0.5\ncase "$ALGO" in\n{block.group(1)}\nesac\n'
              'printf "%s\\n" "${ALGO_ARGS[@]}"')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return [w for w in out.stdout.split("\n") if w]


def test_the_script_exists_under_its_new_name(text: str) -> None:
    """Renamed from ``run_awr_loop.sh`` once it stopped running only AWR."""
    assert "run_awr_loop.sh" not in text, "a stale self-reference to the old name"


def test_every_algo_is_accepted_and_anything_else_is_refused(text: str) -> None:
    for a in ALGOS:
        assert re.search(rf"\b{a}\b", text), f"{a} is not mentioned"
    assert "ALGO must be awr, fbc, bc or arrived" in text, (
        "an unknown ALGO must be refused, not run")


def test_plain_bc_passes_no_labels_at_all() -> None:
    """Not a hot beta.

    AWR's hot limit is unweighted BC on every LABELLED row and the two agree only in that
    limit -- so an arm reported as plain BC has to actually be plain BC, with no label pathway
    and nothing to tune.  This is the control that separates "the advantage selected good
    behaviour" from "the model finally saw this terrain", and it is worthless if it is
    secretly AWR with a large number in it.
    """
    args = _algo_args("bc")
    assert args == [], f"bc must pass nothing, got {args}"


def test_filtered_bc_filters_rather_than_weighting() -> None:
    args = _algo_args("fbc")
    assert "--filter_label" in args and "positive" in args
    assert not any(a.startswith("--awr") for a in args), (
        f"filtered BC must not also weight: {args}")
    assert "--labels" in args, "the filter reads the sidecar's label column"


def test_awr_weights_and_does_not_filter() -> None:
    args = _algo_args("awr")
    assert "--awr_beta" in args and "--labels" in args
    assert "--filter_label" not in args, f"AWR keeps every row: {args}"


def test_the_three_arms_differ_ONLY_in_the_update(text: str) -> None:
    """Collect, grade and buffer churn must not branch on ALGO.

    If they did, an arm's data distribution would differ from another's and the comparison
    would attribute a data effect to the objective.  The only permitted branches are the
    update stage and the two stages `bc` skips because nothing reads them.
    """
    allowed = ("ROUNDS_FVI=0", "7/8 label", "ALGO_ARGS", "algo   ", "ALGO must be",
               # the label stage's branches, the second of the two stages `bc` drops
               '[ "$ALGO" = "bc" ]; then', '[ "$ALGO" = "arrived" ]',
               "case \"$ALGO\" in bc|arrived)",
               # `$ALGO` inside a SKIP message names which arm it was; that is reporting.
               "6/8 critic      SKIP")
    for m in re.finditer(r'^.*\$ALGO.*$', text, re.M):
        line = m.group(0)
        if line.strip().startswith("#:") or line.strip().startswith("#"):
            continue
        if any(tok in line for tok in allowed) or "case \"$ALGO\"" in line:
            continue
        pytest.fail(f"ALGO branches somewhere it should not:\n  {line.strip()}")


def test_bc_skips_the_critic_and_the_label(text: str) -> None:
    """Skipped, not run-and-ignored.

    ~28 minutes an iteration, and an arm that computes a value function it does not use
    invites the reader to wonder whether it used it after all.
    """
    assert 'case "$ALGO" in bc|arrived) ROUNDS_FVI=0;; esac' in text
    assert 'ALGO" = "bc" ]; then' in text, "the label stage must skip under bc"


def test_the_percentile_is_passed_explicitly_not_inherited(text: str) -> None:
    """``label_advantage.py``'s own default is 35, which nothing here chose.

    Filtered BC's threshold IS its definition, so it has to come from the arm's config and be
    recorded in the sidecar, rather than being whatever the labeller's default happened to be
    on the day.
    """
    assert '--percentile "$PERCENTILE"' in text
    assert re.search(r'^PERCENTILE="\$\{PERCENTILE:-70(\.0)?\}"', text, re.M), (
        "the default must be the top-30% threshold the ablation was specified at")


def test_the_banner_states_which_algorithm_ran(text: str) -> None:
    """A run's own log is the only durable record of which arm it was."""
    for a in ("AWR", "FILTERED BC", "PLAIN BC"):
        assert a in text, f"the banner never names {a}"


def test_the_arrived_arm_skips_the_critic() -> None:
    """A no-critic baseline that needs a critic is not a no-critic baseline.

    That is the whole content of the arm: it asks whether spending a value function buys
    anything over "keep the chunks of episodes that did not fail".  On the tracking reward the
    equivalent comparison found this rule matched or beat the critic one.
    """
    t = LOOP.read_text()
    assert 'case "$ALGO" in bc|arrived) ROUNDS_FVI=0;; esac' in t, (
        "arrived must skip the critic exactly as bc does")


def test_the_arrived_arm_uses_its_own_labeller() -> None:
    """``label_advantage.py`` reads the critic's predictions, so it cannot serve this arm."""
    t = LOOP.read_text()
    assert "label_arrived.py" in t
    # And it must produce the sidecar the SAME fine-tune stage consumes, so the arms differ in
    # the selection rule alone.
    assert re.search(r"fbc\|arrived\) ALGO_ARGS=\(--labels[^)]*--filter_label positive", t)


def test_the_failure_rule_is_explicit_and_matches_the_critic_arms() -> None:
    """strict vs loose moves the positive set from 33 rows to 728 on the box2 grid.

    ``fit_value.py``'s default is strict, so an arm compared against a critic arm has to be
    strict too or the two select from differently-defined populations.
    """
    t = LOOP.read_text()
    assert re.search(r'^FAILURE_RULE="\$\{FAILURE_RULE:-strict\}"', t, re.M)
    assert '--failure_rule "$FAILURE_RULE"' in t
