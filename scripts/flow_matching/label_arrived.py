#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Label every chunk of an ARRIVING episode positive -- the no-critic baseline.

    python scripts/flow_matching/label_arrived.py <rollout stem> --out advantage.npz

Writes the same sidecar ``label_advantage.py`` writes, so ``finetune_generator.py
--filter_label positive`` consumes it unchanged and the arm differs from filtered BC in the
SELECTION RULE alone.

**Why this arm exists, and why it is the one to beat.**  Every other arm here spends a value
function to decide which chunks to keep.  This one asks whether that buys anything over the
crudest possible rule: keep the chunks of episodes that reached the goal.  It has form -- on
the TRACKING reward the equivalent comparison found the no-critic outcome filter *matched or
beat* the critic one, which is part of why that line of work was stopped.  If the
progress-reward advantage beats it, that is a real result; if it does not, that is the more
important thing to know, and knowing it before the paper is the point.

**ARRIVAL, not survival, and the difference is not cosmetic.**  ``terminal_reward`` scores 0
only for an episode that entered the goal before it ever failed, and -1 for everything else
INCLUDING a robot that stayed upright and never got there -- its docstring says why: "if
merely staying upright scored 0, standing still would score identically to climbing the
staircase with none of the risk".  A falls-only "survived" rule keeps 96.5% of windows on a
real buffer (measured), which is barely a filter and lands on top of the plain-BC arm -- and
it positively selects the pathology the progress reward exists to remove, since the
terrain-only run drove forward progress 11.50 -> 3.95 m per 1000 steps WHILE survival rose.
Arrival keeps 56.5%, which is a real selection and comparable to filtered BC's 30%.

**A separate script rather than a mode of ``label_advantage.py``.**  That script's whole body
reads ``predictions.npz`` -- the critic's output -- so a critic-free path through it would
still need a critic to have run, which defeats the baseline.  Here nothing is fitted: the
label is a property of the recording.

**The failure rule is IMPORTED, not restated**, and it decides what "before it failed"
means, so it moves this filter a lot: strict keeps 55.8% of episodes on a real buffer against
loose's 67.1%, because 41.7% of them trip a tracking violation somewhere.  ``rollout_outcomes.terminal_reward`` is the
one definition of "this episode failed", and the strict/loose choice moves the answer
enormously -- on the box2 grid the strict rule labels 8967 of 9000 episodes failed against the
loose rule's 8272, because 695 episodes REACHED THE GOAL and tripped the bootstrap tolerances
on the way, so the positive set is 33 rows or 728 depending on the flag.  An arm that
reimplemented the rule could differ from the arm it is being compared against for a reason
that has nothing to do with selection.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "source" / "robot_rl"), str(_ROOT / "source" / "motion_gen"),
           str(_ROOT / "scripts" / "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rollout_outcomes as ro  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stem", help="Rollout stem (the buffer's, without .h5).")
    p.add_argument("--out", required=True, help="Output .npz sidecar.")
    p.add_argument("--failure_rule", default="strict", choices=("strict", "loose"),
                   help="What counts as failing BEFORE arrival, which is what this filter "
                        "tests against. MUST match the rule the arm being "
                        "compared against used -- fit_value.py's own default is strict, so "
                        "that is the default here. Strict counts the tracking violation "
                        "alongside the fall; loose is the fall alone, which is what MuJoCo "
                        "scores. On the box2 grid the two differ by 695 episodes of 9000.")
    p.add_argument("--drop_censored", action="store_true", default=True,
                   help="Drop chunks of episodes cut off by the recording's stop rather than "
                        "by an outcome. A censored episode neither survived nor failed, and "
                        "calling it either way puts a fabricated label in the training set.")
    return p.parse_args()


def build(stem: str, *, strict: bool, drop_censored: bool) -> dict:
    """Read the rollout's two tables and label every chunk by its episode's outcome.

    Args:
        stem: Rollout stem.
        strict: Use the strict failure rule (see :func:`parse_args`).
        drop_censored: Exclude censored episodes entirely.

    Returns:
        The sidecar's arrays, keyed as ``label_advantage.py`` writes them.

    Raises:
        SystemExit: If the two tables do not join exactly on ``episode_uid``. A window whose
            episode has no outcome row cannot be labelled, and dropping it silently would
            change the training distribution -- the same refusal ``fit_value._load`` makes.
    """
    # The rollout is TWO FILES, `<stem>.h5` (windows) and `<stem>.episodes.h5` (outcomes),
    # and `rollout_outcomes` owns both readers -- so they are used rather than h5py directly.
    # Reading them by hand is how a caller ends up disagreeing with every other consumer about
    # the layout; this script's first version looked for a `windows` GROUP and found none.
    w, _wa = ro.read_rollout_windows(
        f"{stem}.h5", columns=["episode_uid", "replan_index", "family_code"])
    E, _ea = ro.read_rollout_episodes(f"{stem}.episodes.h5")
    uid = np.asarray(w["episode_uid"], np.int64)
    rep = np.asarray(w["replan_index"], np.int64)
    fam = np.asarray(w["family_code"], np.int64)

    # `terminal_reward` is the shared definition and returns only {-1, 0}: 0 for an episode
    # that entered the goal BEFORE it ever failed, -1 for everything else -- failed first, or
    # stayed upright and never arrived. So `>= 0` is ARRIVED, and a timeout is excluded.
    reward = ro.terminal_reward(E, strict=strict)
    arrived_ep = reward >= 0.0
    censored = E["censored"].astype(bool)
    by_uid = {int(u): (bool(a), bool(c))
              for u, a, c in zip(E["episode_uid"], arrived_ep, censored)}

    missing = sorted({int(u) for u in uid} - set(by_uid))
    if missing:
        raise SystemExit(
            f"{len(missing)} window episode(s) have no outcome row (e.g. {missing[:5]}). The "
            "two tables must join exactly on episode_uid.")

    w_arrived = np.array([by_uid[int(u)][0] for u in uid], bool)
    w_censored = np.array([by_uid[int(u)][1] for u in uid], bool)

    keep = ~w_censored if drop_censored else np.ones(uid.size, bool)
    # +1 arrived, -1 did not, 0 = not labelled (censored). `finetune_generator
    # --filter_label positive` keeps only the +1 rows, so a 0 drops the row without
    # asserting anything about it.
    label = np.where(w_arrived, 1, -1).astype(np.int8)
    label[~keep] = 0
    # The `advantage` column is what an AWR arm would weight by; nothing reads it here, but a
    # sidecar whose columns disagree with its label would be a trap for the next reader, so it
    # carries the same information as a number.
    adv = np.where(label > 0, 1.0, np.where(label < 0, -1.0, 0.0)).astype(np.float32)
    return {
        "label": label,
        "advantage": adv,
        "bootstrapped": np.zeros(uid.size, bool),
        "episode_uid": uid,
        "replan_index": rep,
        "family_code": fam,
        # One stratum per family, matching what `label_advantage` produces with the default
        # single speed/progress bin -- the label here is not a percentile, so there is no cut
        # to stratify, but the column has to exist and mean the same thing.
        "stratum": fam.astype(np.int64),
        # No critic, so no held-out tiles: nothing was fitted that a holdout would protect.
        "val_tile": np.zeros(uid.size, bool),
    }


def main() -> int:
    """Write the sidecar."""
    args = parse_args()
    strict = args.failure_rule == "strict"
    out = build(args.stem, strict=strict, drop_censored=bool(args.drop_censored))
    n = out["label"].size
    pos = int((out["label"] > 0).sum())
    neg = int((out["label"] < 0).sum())
    nul = n - pos - neg
    meta = {
        "labeller": "label_arrived",
        "selection": "every chunk of an episode that reached the goal before failing",
        "failure_rule": args.failure_rule,
        "drop_censored": bool(args.drop_censored),
        "critic": None,
        "n_windows": n, "n_positive": pos, "n_negative": neg, "n_null": nul,
        "episodes": int(np.unique(out["episode_uid"]).size),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, meta=json.dumps(meta), **out)
    print(f"[arrived] {pos:,} positive / {neg:,} negative / {nul:,} null (censored) "
          f"over {meta['episodes']:,} episodes, {args.failure_rule} failure rule")
    print(f"[arrived] keep rate {pos / max(n, 1):.4f}")
    print(f"[arrived] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
