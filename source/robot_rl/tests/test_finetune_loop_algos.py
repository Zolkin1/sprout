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
    assert "--labels" not in args, f"bc must pass no label sidecar, got {args}"
    assert not any(a.startswith("--awr") for a in args), f"bc must not weight: {args}"
    assert "--filter_label" not in args, f"bc must not filter: {args}"
    # It DOES pass `--unconditioned`, and must: `finetune_generator.is_plain_arm` is true for
    # `filter_label`, `unconditioned` or `awr_beta > 0`, so a run with none of those silently
    # takes the advantage-CONDITIONED path -- trains an embedding no label ever reaches and
    # writes a checkpoint with no serve arm, which the eval then refuses outright. That is
    # what killed `eval_bc_s3_i1` at all three speeds.
    assert "--unconditioned" in args, (
        f"bc must take the PLAIN architecture path, got {args}")


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


def test_the_failure_rule_is_STRICT_because_isaac_now_terminates_on_it() -> None:
    """strict vs loose moved the positive set from 33 rows to 728 on a pre-termination grid.

    That difference was the tell for an incoherence rather than a preference: the recording ran
    PAST a violation and every consumer then cut the episode there offline, so a run reached
    the goal, banked the progress reward's success bonus, and the cut deleted the row it was
    paid on -- 95.3% of arriving episodes. Now ``terminations.tracking_violation`` ends the
    episode at the bounds, so strict is what the simulator does and the two rules agree by
    construction. ``loose`` survives for scoring a recording made before that.
    """
    t = LOOP.read_text()
    assert re.search(r'^FAILURE_RULE="\$\{FAILURE_RULE:-strict\}"', t, re.M), (
        "the loop must default to the rule Isaac terminates on")
    assert '--failure_rule "$FAILURE_RULE"' in t
    assert "REFUSING: FAILURE_RULE must be strict or loose" in t


def test_ONE_failure_rule_reaches_both_the_critic_and_the_arrived_filter() -> None:
    """They are two uses of one definition, and the loop used to pass it to only one.

    ``fit_value.py`` cuts every window at or after the first bad step, and the progress
    reward pays its success bonus on the episode's LAST re-plan alone -- so under strict that
    cut removed the bonus from 95.3% of the ARRIVING episodes, and the critic was fitted on a
    reward with essentially no success term in it while every log read as healthy. The loop
    never passed the flag to the critic at all, so the two stages silently used different
    definitions of the same word.
    """
    t = LOOP.read_text()
    passes = re.findall(r'--failure_rule "\$FAILURE_RULE"', t)
    assert len(passes) >= 2, (
        f"the failure rule reaches {len(passes)} stage(s); the critic's window cut and the "
        "arrived filter both need it")
    # The INVOCATION, not the prose: anchored on the interpreter, since the flag's own
    # rationale is written out in a comment that mentions the script by name.
    fit = re.search(r'"\$PY" -u scripts/flow_matching/fit_value\.py(.*?)\n\n', t, re.S)
    assert fit, "fit_value.py's invocation is not where this test expects it"
    assert '--failure_rule "$FAILURE_RULE"' in fit.group(1), (
        "fit_value.py is called without the failure rule, so it falls back to its own "
        "default -- which is how the critic and the arrived filter came to disagree")


def test_the_banner_states_the_failure_rule_for_every_arm() -> None:
    """A run's log is the only durable record of which definition of "failed" it used."""
    t = LOOP.read_text()
    assert re.search(r'echo "\[awr\] failed \$FAILURE_RULE', t), (
        "the banner never names the failure rule outside the arrived arm's line")


def test_the_history_arm_is_passed_as_a_CLI_FLAG_not_a_hydra_override() -> None:
    """``record_rollouts.py`` has its OWN ``--history_from`` and applies it AFTER composition.

    So ``env.commands.traj_ref.history_from=`` is silently discarded: the run serves the
    generator arm to whatever checkpoint it was given, and on a measured-trained one
    ``assert_history_source_matches`` then kills it -- after the simulator has launched, with
    the refusal naming the checkpoint rather than the override that did not take. That cost a
    GPU launch on 2026-09-09.

    This is the same shape as the eval harness's own rule that a flag added to a parser is not
    forwarded unless the caller restates it; here the caller restated it in the WRONG channel.
    """
    t = LOOP.read_text()
    assert '--history_from "$HISTORY_FROM"' in t, (
        "the loop must pass the arm as a CLI flag; the hydra override is overwritten")
    # An ARGUMENT line, not the string anywhere: the comment above the flag names the
    # override precisely to say why it is not used, and must not fail its own test.
    assert not re.search(r"^\s*env\.commands\.traj_ref\.history_from=", t, re.M), (
        "the hydra override is back as an argument, and record_rollouts will discard it")


def test_the_eval_passes_the_history_arm_the_same_way() -> None:
    """Both sides must agree, or an arm is evaluated on a history it was not trained for."""
    ev = (_REPO / "scripts" / "rsl_rl" / "run_quick_eval.sh").read_text()
    assert '--history_from "$HISTORY_FROM"' in ev
    assert not re.search(r"^\s*env\.commands\.traj_ref\.history_from=", ev, re.M)
