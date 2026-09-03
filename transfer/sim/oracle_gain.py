#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""What is CHOOSING a chunk worth, at one re-plan, to a selector that cannot see the future?

`branch_probe.py` rolls K sibling chunks from one state to completion, and now rolls each of
them TWICE on two independently-keyed continuation streams.  That second column is what makes
a selection claim falsifiable rather than circular.  With one column, "pick the best sibling"
scores the very draw it picked on, and the number it returns is the spread of the outcome
noise dressed up as a policy improvement.  With two, the pick is made on column A and PAID
OUT on column B, so any gain that survives is a property of the CHUNK and not of its luck.

    honest      select on A, score on B (and the mirror), averaged
    hindsight   select on A, score on A -- the same draw.  Reported only as a contrast
    any-of-K    max_k y[g,k,A] - mean_k y[g,k,A].  Algebraically IDENTICAL to hindsight on
                the A column (the argmax's own value IS the max); printed beside it because
                the two names circulate as if they were different measurements, and seeing
                them agree to the last digit is the point rather than a copy-paste bug

**What this bounds, and what it does not.**  The honest gain is what a selector with a
one-rollout-quality predictor would collect at ONE decision, given the generator this pair
ships.  It is an upper bound on best-of-K, filtered BC and advantage conditioning AT THE
PER-DECISION LEVEL, because no predictor of the chunk's expected outcome can beat knowing a
whole independent realization of it.  It is NOT a trajectory-level ceiling: a selector acts at
every re-plan, and 50 decisions each worth a little may compound.  It is also not a critic
result -- no critic is read here at all.  The critic's own ranking is `branch_probe --stage
report`; this is the ceiling that report has to be read against.

**Three ways this measurement can lie, and the arm that catches each.**

* *The pipeline leaks a draw into its own selector.*  Caught by ``--control``: a
  ``--same_noise`` run hands every sibling ONE branch draw, so the siblings are bit-identical,
  every group's spread is exactly zero and the gain MUST be 0.000000.  Anything else means the
  two columns are not what they claim.
* *The gain is the maximum of K noisy numbers.*  Caught by the permutation null, which
  re-links column B to a random sibling within each group and so preserves every marginal
  while destroying the only thing under test.
* *The estimator itself is wrong.*  Caught by ``--stage selfcheck``, which runs the whole
  computation on synthetic data whose answer is known in closed form -- the discipline
  ``eval/seed_headroom.py`` paid for the hard way, where three estimators were validated this
  way and two of the three failed.

**The drop rule is a BIAS, stated rather than hidden.**  A group is usable only if every
sibling carries both continuations.  The one thing that makes a group partial is a sibling
that terminated INSIDE its own chunk -- it has no continuation to roll, and `branch_probe`
gives it ``g_real = NaN`` on purpose (a single own-chunk reward is a different random variable
from a rolled-out return).  Those are exactly the decisive states: a sibling that fell in
0.24 s is the clearest thing any selector could have avoided, so dropping them biases the
ceiling DOWN.  ``--include_terminal_binary`` measures how much: on the BINARY outcome a
terminal branch's outcome is not noisy at all (a branch that reached the goal reached it under
every continuation), so it can be included at zero continuation variance.  The continuous
outcome has no such repair and the headline drops them either way.

Usage::

    python oracle_gain.py --stage analyze --out logs/value/oracle_gain_v1 \\
        --run logs/value/critic_eval_v4/bprobe_t6s16 \\
        --run logs/value/critic_eval_v4/bprobe_t20s16 \\
        --control logs/value/critic_eval_v4/bprobe_ctrl
    python oracle_gain.py --stage selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if not os.path.isfile(os.path.join(REPO_ROOT, "scripts", "rsl_rl", "plot_style.py")):
    raise RuntimeError(f"{__file__} derived REPO_ROOT={REPO_ROOT}, which has no "
                       "scripts/rsl_rl/plot_style.py -- the file moved and this derivation "
                       "did not follow it.")
# The same bootstrap `branch_probe.py` uses, plus `scripts/rsl_rl` for the shared paper plot
# style. `transfer/` is in it because `transfer/sim/__init__.py` re-exports `Simulation`,
# whose own imports are written as `sim.robot` -- so anything that imports this directory as a
# PACKAGE (pytest does, when collecting a test under it) needs the parent on the path too.
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval"),
           os.path.join(REPO_ROOT, "scripts", "rsl_rl")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: Spread bins for the CONTINUOUS outcome, as upper edges.  The first is exact zero and is
#: kept even though a continuous return practically never ties: it is the bin the
#: ``--same_noise`` control must land ALL of its groups in, so an empty first bin on a
#: treatment run and a full one on the control is itself a check.
SPREAD_EDGES: tuple[float, ...] = (0.0, 0.15, 0.30)

#: E[max of K i.i.d. standard normals], for the closed-form selfcheck.  K=4 only; the
#: selfcheck fixes K rather than deriving this at run time, because the constant is what makes
#: the expected answer a NUMBER instead of another simulation.
_EMAX_K4 = 1.0293753730039641


# ======================================================================================
# loading
# ======================================================================================
def _decode(arr) -> np.ndarray:
    """Byte-or-str HDF5 column to a numpy array of ``str``."""
    return np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                     for x in arr], dtype=object)


@dataclass
class Cell:
    """One run directory's usable groups, plus the cell it was collected in.

    ``y_a`` / ``y_b`` are ``(G, K)``: one row per probe state, one column per sibling, the
    two independently-keyed continuations of the SAME sibling side by side.  The pairing
    across the two arrays is the entire content of this file; a shuffle of either one's second
    axis is the permutation null.
    """

    run: str
    tile: int
    speed: float
    klass: str
    same_noise: bool
    k: int
    y_a: np.ndarray
    y_b: np.ndarray
    seed: np.ndarray
    replan: np.ndarray
    ep_frac: np.ndarray
    degenerate: np.ndarray
    drops: dict = field(default_factory=dict)
    n_groups_total: int = 0
    #: Continuation-verdict histogram over EVERY branch in the file, before any drop. A run
    #: dominated by `timeout` is a statement about the clock rather than about the robot, and
    #: the binary outcome scores a timeout as a failure -- so it would move the baseline
    #: without moving anything real. That is the shape of the `--isaac_matched` retraction
    #: (84.7% of episodes timed out and the label was almost entirely the budget), so the
    #: histogram travels beside the number rather than being available on request.
    verdicts: dict = field(default_factory=dict)


def _klass_for(attrs: dict, tile: int) -> str:
    """The course class for a run, from the file if it says so and the library if not.

    New runs stamp ``klass`` (``branch_probe._write``).  Older ones do not, and the honest
    recovery is the manifest the course was built from -- not a tile-id table here, which
    would be a second statement of what a tile is and would rot the moment a library is
    regenerated.
    """
    if attrs.get("klass"):
        k = attrs["klass"]
        return k.decode() if isinstance(k, bytes) else str(k)
    lib = attrs.get("clip_lib", "")
    lib = lib.decode() if isinstance(lib, bytes) else str(lib)
    # The recorded path may be RELATIVE, and to the launching cwd rather than to anything this
    # process knows -- `branch_probe` is run from `transfer/sim`, so a library recorded as
    # `../../clips/x.h5` resolves against THAT and not against here. Tried in order rather
    # than assumed, and the winner is printed, because silently resolving to a different
    # library than the run used would mislabel a family with no error at all.
    for base in ("", _HERE, REPO_ROOT):
        cand = lib if not base else os.path.normpath(os.path.join(base, lib))
        if cand and os.path.isfile(cand):
            if base:
                print(f"[oracle_gain] tile {tile}: clip_lib {lib!r} resolved against "
                      f"{base} -> {cand}")
            lib = cand
            break
    else:
        print(f"[oracle_gain] tile {tile}: clip_lib {lib!r} is not readable from here; "
              "stratifying it as 'unknown'")
        return "unknown"
    try:
        import generate_clip_tile_scene as gcts  # noqa: PLC0415
        from mjeval.terrains import klass_for_family  # noqa: PLC0415

        entry, _info = gcts.load_tile(lib, int(tile))
        return klass_for_family(str(entry.get("family", "")), int(tile))
    except Exception as exc:                                   # noqa: BLE001
        print(f"[oracle_gain] tile {tile}: could not resolve a family ({exc}); "
              "stratifying it as 'unknown'")
        return "unknown"


def load_cell(dirpath: str, *, outcome: str, include_terminal: bool) -> Cell:
    """Read one ``branches.h5`` and fold it into ``(G, K)`` outcome matrices.

    Args:
        dirpath: A ``branch_probe --stage run`` output directory.
        outcome: ``"binary"`` (reached the goal without violating) or ``"continuous"``
            (``normalized_return`` over the continuation, in ``[0, 1]``).
        include_terminal: Keep siblings that terminated inside their own chunk, giving them
            their branch verdict as BOTH columns.  Binary only -- there is no continuous
            outcome for them.  See the module docstring on why this is a bias arm and not a
            default.

    Returns:
        The :class:`Cell`.  Groups that do not carry two continuations for every sibling are
        absent from it and counted in ``drops``.

    Raises:
        SystemExit: If the file predates the all-sibling second continuation, or carries no
            second continuation at all.  Mixing a sibling-0-only file in would analyse one
            sibling of every group as if it were K, silently.
    """
    import h5py  # noqa: PLC0415

    path = os.path.join(dirpath, "branches.h5")
    if not os.path.isfile(path):
        raise SystemExit(f"no branches.h5 in {dirpath}")
    with h5py.File(path, "r") as f:
        d = {k: f[k][:] for k in f}
        attrs = dict(f.attrs)
    if "cont_verdict_b" not in d:
        raise SystemExit(
            f"{path} has no second continuation at all (no `cont_verdict_b`). Re-run "
            "branch_probe with --second_seed_every set; a one-column file cannot support a "
            "selection that is scored on a draw it did not pick on.")

    k = int(attrs["k"])
    seed = d["seed"].astype(np.int64)
    replan = d["replan"].astype(np.int64)
    cand = d["cand"].astype(np.int64)
    bverd, cverd = _decode(d["branch_verdict"]), _decode(d["cont_verdict"])
    bverd_b = _decode(d["cont_verdict_b"])
    term = (d["terminal_in_branch"].astype(bool) if "terminal_in_branch" in d
            else np.array([v != "" for v in bverd]))

    # A sibling-0-only file: every group that has ANY second continuation has exactly one.
    # Refused rather than analysed, per the guardrail -- a partial column is a different
    # estimator wearing the same name.
    has_b = np.array([v != "" for v in bverd_b])
    per_group_b: dict[tuple[int, int], int] = {}
    for i in np.flatnonzero(has_b):
        per_group_b[(seed[i], replan[i])] = per_group_b.get((seed[i], replan[i]), 0) + 1
    scope = attrs.get("second_seed_scope", b"")
    scope = scope.decode() if isinstance(scope, bytes) else str(scope)
    if per_group_b and max(per_group_b.values()) < 2 and k > 1:
        raise SystemExit(
            f"{path} second-seeded ONE sibling per group (scope {scope or 'unstamped'}): "
            f"{len(per_group_b)} groups, none with more than one B continuation. That column "
            "cannot rank a group. Re-collect with the all-sibling second continuation.")

    # `ep_frac` from ALL probed re-plans of the episode, before any drop: the position of a
    # group in its own episode is a property of the episode, and computing it after dropping
    # would move a mid-episode group to "late" because the later ones happened to be partial.
    last_replan: dict[int, int] = {}
    for s, r in zip(seed, replan):
        last_replan[int(s)] = max(last_replan.get(int(s), 0), int(r))

    if outcome == "binary":
        y1 = (cverd == "success").astype(np.float64)
        y2 = (bverd_b == "success").astype(np.float64)
        ok = has_b.copy()
        if include_terminal:
            # A terminal branch's outcome has NO continuation variance: it reached the goal,
            # fell or violated inside its own chunk, under every continuation there could be.
            # So both columns are its branch verdict and the pair is exact, not imputed.
            y1 = np.where(term, (bverd == "success").astype(np.float64), y1)
            y2 = np.where(term, (bverd == "success").astype(np.float64), y2)
            ok = ok | term
    elif outcome == "continuous":
        y1, y2 = d["g_real"].astype(np.float64), d["g_real_b"].astype(np.float64)
        ok = has_b & np.isfinite(y1) & np.isfinite(y2)
    else:
        raise SystemExit(f"unknown outcome {outcome!r}")

    rows: dict[tuple[int, int], list] = {}
    for i in range(seed.size):
        rows.setdefault((int(seed[i]), int(replan[i])), []).append(i)

    ya, yb, gs, gr, gf, gd = [], [], [], [], [], []
    drops = {"partial_second_continuation": 0, "wrong_sibling_count": 0,
             "duplicate_sibling": 0}
    for (s, r), idx in sorted(rows.items()):
        if len(idx) != k:
            drops["wrong_sibling_count"] += 1
            continue
        order = np.argsort(cand[idx])
        idx = [idx[j] for j in order]
        if sorted(int(cand[i]) for i in idx) != list(range(k)):
            drops["duplicate_sibling"] += 1
            continue
        if not all(ok[i] for i in idx):
            drops["partial_second_continuation"] += 1
            continue
        ya.append([y1[i] for i in idx])
        yb.append([y2[i] for i in idx])
        gs.append(s)
        gr.append(r)
        gf.append(r / max(last_replan[s], 1))
        gd.append(bool(any(term[i] for i in idx)))

    vhist: dict[str, int] = {}
    for v in cverd:
        vhist[str(v)] = vhist.get(str(v), 0) + 1
    tile = int(attrs["tile"])
    return Cell(run=os.path.basename(os.path.normpath(dirpath)), tile=tile,
                speed=float(attrs["speed"]), klass=_klass_for(attrs, tile),
                same_noise=bool(attrs.get("same_noise", False)), k=k,
                y_a=np.asarray(ya, dtype=np.float64).reshape(-1, k),
                y_b=np.asarray(yb, dtype=np.float64).reshape(-1, k),
                seed=np.asarray(gs, dtype=np.int64), replan=np.asarray(gr, dtype=np.int64),
                ep_frac=np.asarray(gf, dtype=np.float64),
                degenerate=np.asarray(gd, dtype=bool), drops=drops,
                n_groups_total=len(rows), verdicts=vhist)


@dataclass
class Stacked:
    """Several cells concatenated, with the cell labels kept per group."""

    y_a: np.ndarray
    y_b: np.ndarray
    cell: np.ndarray          # "t6_s1.6"
    klass: np.ndarray
    episode: np.ndarray       # bootstrap CLUSTER: successive re-plans of one episode are
    ep_frac: np.ndarray       # not independent, so the cluster is the episode, not the group
    degenerate: np.ndarray
    k: int
    drops: dict
    n_groups_total: int
    verdicts: dict


def stack(cells: list[Cell]) -> Stacked:
    """Concatenate cells, keying the bootstrap cluster by ``(run, seed)``.

    The group key must carry the RUN: every cell was collected with ``seed_offset 0``, so
    ``seed`` alone collides across cells and would silently pool four different episodes into
    one cluster.
    """
    if not cells:
        raise SystemExit("no cells")
    ks = {c.k for c in cells}
    if len(ks) != 1:
        raise SystemExit(f"cells disagree on K: {sorted(ks)}; a gain over K is not comparable "
                         "across different K and pooling them would average two estimands.")
    drops: dict = {}
    vhist: dict = {}
    for c in cells:
        for key, val in c.drops.items():
            drops[key] = drops.get(key, 0) + val
        for key, val in c.verdicts.items():
            vhist[key] = vhist.get(key, 0) + val
    return Stacked(
        y_a=np.concatenate([c.y_a for c in cells]),
        y_b=np.concatenate([c.y_b for c in cells]),
        cell=np.concatenate([np.full(c.y_a.shape[0], f"t{c.tile}_s{c.speed:g}") for c in cells]),
        klass=np.concatenate([np.full(c.y_a.shape[0], c.klass) for c in cells]),
        # The cluster key carries the cell's INDEX as well as its directory name: two --run
        # paths can end in the same basename, and then two cells' episodes would merge into
        # one cluster and widen nothing.
        episode=np.concatenate([np.array([f"{i}:{c.run}:{s}" for s in c.seed])
                                for i, c in enumerate(cells)]),
        ep_frac=np.concatenate([c.ep_frac for c in cells]),
        degenerate=np.concatenate([c.degenerate for c in cells]),
        k=cells[0].k, drops=drops,
        n_groups_total=sum(c.n_groups_total for c in cells), verdicts=vhist)


# ======================================================================================
# the estimator
# ======================================================================================
def _pick(y_sel: np.ndarray, rng: np.random.Generator, tie_break: str) -> np.ndarray:
    """Row-wise argmax of ``y_sel``, ties broken uniformly at random.

    Args:
        y_sel: ``(G, K)`` selection column.
        rng: Seeded generator for the tie-break.
        tie_break: ``"random"`` puts all the weight on ONE uniformly drawn member of the
            argmax set, which is the estimator as specified; ``"expected"`` spreads it
            uniformly over that set, which is the exact expectation over the same draw and
            carries none of its variance.  Ties are the common case on a binary outcome --
            most groups are all-1 or all-0 -- so the draw is a real variance source and both
            are worth having side by side.

    Returns:
        ``(G, K)`` weights summing to 1 per row.
    """
    top = (y_sel == y_sel.max(axis=1, keepdims=True))
    if tie_break == "expected":
        return top / top.sum(axis=1, keepdims=True)
    u = np.where(top, rng.random(y_sel.shape), -1.0)
    w = np.zeros_like(y_sel)
    w[np.arange(y_sel.shape[0]), np.argmax(u, axis=1)] = 1.0
    return w


def arms(y_a: np.ndarray, y_b: np.ndarray, rng: np.random.Generator,
         tie_break: str = "random") -> dict[str, np.ndarray]:
    """Every per-group quantity the report needs, from the two outcome columns.

    Args:
        y_a: ``(G, K)`` first continuation.
        y_b: ``(G, K)`` second continuation.
        rng: Seeded generator for tie-breaking.
        tie_break: ``"random"`` (the estimator as specified) or ``"expected"``.

    Returns:
        Per-group vectors: ``gain`` (the honest, swap-averaged one), its two arms ``gain_ab``
        and ``gain_ba``, ``baseline`` and ``oracle`` so the gain reads as "from X to Y",
        ``hindsight`` (scored on the draw it picked on) and ``any_of_k``, and the realized
        selection-column ``spread``.
    """
    wa, wb = _pick(y_a, rng, tie_break), _pick(y_b, rng, tie_break)
    # THE GUARDRAIL, in one line each: the weights come from one column and the score from the
    # other, never the same array twice.
    score_ab = (wa * y_b).sum(axis=1)
    score_ba = (wb * y_a).sum(axis=1)
    base_b, base_a = y_b.mean(axis=1), y_a.mean(axis=1)
    hind_a = y_a.max(axis=1) - base_a
    hind_b = y_b.max(axis=1) - base_b
    return {"gain_ab": score_ab - base_b, "gain_ba": score_ba - base_a,
            "gain": 0.5 * ((score_ab - base_b) + (score_ba - base_a)),
            "oracle": 0.5 * (score_ab + score_ba), "baseline": 0.5 * (base_a + base_b),
            "hindsight": 0.5 * (hind_a + hind_b), "any_of_k": hind_a,
            "spread": y_a.max(axis=1) - y_a.min(axis=1)}


def bootstrap_ci(v: np.ndarray, *, reps: int, rng: np.random.Generator,
                 cluster: np.ndarray | None = None) -> tuple[float, float]:
    """Percentile 95% CI for ``mean(v)``, resampling groups or CLUSTERS of them.

    ``cluster`` is the episode.  Successive probed re-plans of one episode share a robot that
    is either coping or already in trouble, so they are not independent draws; the eval
    harness measured a plain interval's coverage at 14-70% against a nominal 95% under the
    analogous clip clustering.  Both intervals are reported and the honest one to quote is the
    wider.
    """
    if v.size == 0:
        return (float("nan"), float("nan"))
    if cluster is None:
        draws = rng.integers(0, v.size, size=(reps, v.size))
        means = v[draws].mean(axis=1)
    else:
        keys, inv = np.unique(cluster, return_inverse=True)
        groups = [np.flatnonzero(inv == j) for j in range(keys.size)]
        means = np.empty(reps, dtype=np.float64)
        for i in range(reps):
            pick = rng.integers(0, len(groups), size=len(groups))
            means[i] = v[np.concatenate([groups[j] for j in pick])].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def permutation_p(y_a: np.ndarray, y_b: np.ndarray, *, reps: int,
                  rng: np.random.Generator, tie_break: str,
                  obs: float | None = None) -> tuple[float, float, float]:
    """One-sided p-value for the honest gain against a within-group re-linking of column B.

    The null permutes the SIBLING LABELS of column B inside each group.  That preserves every
    marginal -- the same K outcomes, the same group means, the same maxima, so the baseline and
    the hindsight arm are numerically untouched -- and destroys only the correspondence between
    a sibling's two draws, which is the hypothesis.  It is the same shape of null as
    ``branch_probe``'s within-group prediction shuffle, and deliberately not a global shuffle:
    those return their chance line analytically and so cannot fail.

    Returns:
        ``(p, null mean, null sd)``.
    """
    if y_a.shape[0] == 0:
        return (float("nan"), float("nan"), float("nan"))
    # The observed value is passed in by the caller that REPORTS it, so the p-value refers to
    # the number in the table. Recomputing it here would draw the tie-break again and score
    # the null against a value that appears nowhere.
    if obs is None:
        obs = float(arms(y_a, y_b, rng, tie_break)["gain"].mean())
    idx = np.tile(np.arange(y_a.shape[1]), (y_a.shape[0], 1))
    null = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        perm = rng.permuted(idx, axis=1)
        null[i] = arms(y_a, np.take_along_axis(y_b, perm, axis=1),
                       rng, tie_break)["gain"].mean()
    p = float((1 + int((null >= obs).sum())) / (reps + 1))
    return (p, float(null.mean()), float(null.std()))


def reliability(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """Within-group demeaned correlation between the two continuations.

    This is the SELECTION RELIABILITY: how much of a sibling's outcome is the sibling rather
    than the draw.  Demeaning per group is what removes "this state was hard", which is not a
    quantity a chunk selector can act on and which would inflate the correlation.
    """
    if y_a.shape[0] == 0:
        return float("nan")
    a = y_a - y_a.mean(axis=1, keepdims=True)
    b = y_b - y_b.mean(axis=1, keepdims=True)
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = np.argsort(x, kind="mergesort")
    out = np.empty(x.size, dtype=np.float64)
    srt = x[order]
    rr = np.arange(1, x.size + 1, dtype=np.float64)
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        rr[i:j + 1] = rr[i:j + 1].mean()
        i = j + 1
    out[order] = rr
    return out


def reliability_spearman(y_a: np.ndarray, y_b: np.ndarray) -> tuple[float, int]:
    """Mean WITHIN-GROUP Spearman between the two continuations, and the group count.

    Reported beside :func:`reliability` because it is the statistic the earlier label-ceiling
    measurement used (``critic_branch_metrics.label_ceiling``, 0.268 over 247 groups), and the
    two are NOT the same number on the same data -- and this reproduces that 0.268 to four
    digits on it, which is what establishes that they are two statistics rather than two
    estimates.  This one gives every group equal weight however little its siblings differed;
    the pooled demeaned Pearson weights a group by its own spread.  On the continuous outcome
    reliability rises with spread, so there the pooled figure is the higher of the two; the
    ordering is not guaranteed, and on a binary outcome with few informative groups it can
    invert on noise alone.  Reported together so a reader comparing them does not conclude one
    of the two instruments is broken.

    The extrapolation uses the PEARSON one, deliberately: ``gain / sqrt(r)`` comes from a
    variance decomposition of ``y = mu + e``, and the mean of a per-group rank statistic is
    not that variance ratio.  This one is for continuity with the earlier number.
    """
    vals = []
    for i in range(y_a.shape[0]):
        a, b = y_a[i], y_b[i]
        if np.all(a == a[0]) or np.all(b == b[0]):
            continue
        ra, rb = _rank(a), _rank(b)
        if ra.std() == 0 or rb.std() == 0:
            continue
        vals.append(float(np.corrcoef(ra, rb)[0, 1]))
    return ((float(np.mean(vals)) if vals else float("nan")), len(vals))


def summarize(st: Stacked, a: dict, a_exp: dict, *, reps_boot: int, reps_perm: int,
              rng: np.random.Generator, tie_break: str, with_null: bool = True) -> dict:
    """The full aggregate block for one outcome version.

    Args:
        st: The stacked groups.
        a: Precomputed :func:`arms` under the headline tie-break.  Passed in rather than
            recomputed because the tie draw is a RANDOM part of the estimator: computing it
            here and again in :func:`strata` would give the headline and the per-stratum rows
            different draws, and the strata would then not average back to the headline --
            a discrepancy that reads as an arithmetic bug and is not one.
        a_exp: Precomputed :func:`arms` with the tie draw integrated out.
        reps_boot: Bootstrap reps.
        reps_perm: Permutation reps (0 skips the null).
        rng: Seeded generator.
        tie_break: Passed to :func:`arms`.
        with_null: Compute the permutation null.

    Returns:
        A JSON-ready dict.  ``perfect_gain`` is the bivariate-normal extrapolation and is
        labelled as such: selecting on one noisy draw correlates ``sqrt(r)`` with the true
        conditional mean, and under that approximation the gain is linear in the predictor's
        correlation, so dividing by ``sqrt(r)`` reads off what a PERFECT predictor of the
        chunk's expectation would collect.  The selfcheck shows it is exact under the model it
        assumes; nothing here says the outcomes obey it.
    """
    exp_gain = (float(a_exp["gain"].mean()) if st.y_a.shape[0] else float("nan"))
    n = int(st.y_a.shape[0])
    out = {"n_groups": n, "k": st.k, "n_groups_before_drop": st.n_groups_total,
           "drops": st.drops, "tie_break": tie_break, "verdicts": st.verdicts}
    if n == 0:
        return out
    r = reliability(st.y_a, st.y_b)
    rho_s, n_rho_s = reliability_spearman(st.y_a, st.y_b)
    g = a["gain"]
    out.update({
        "mean_baseline": float(a["baseline"].mean()),
        "mean_oracle_score": float(a["oracle"].mean()),
        "mean_gain": float(g.mean()),
        "mean_gain_expected_tiebreak": exp_gain,
        "ci_group": list(bootstrap_ci(g, reps=reps_boot, rng=rng)),
        "ci_episode_clustered": list(bootstrap_ci(g, reps=reps_boot, rng=rng,
                                                  cluster=st.episode)),
        "gain_ab": float(a["gain_ab"].mean()),
        "gain_ba": float(a["gain_ba"].mean()),
        # The two arms must agree within noise, so the asymmetry gets an INTERVAL rather than
        # a number to eyeball. The two continuations are rolled from the same post-chunk
        # snapshot, so they are exchangeable by construction; a persistent asymmetry would
        # mean they are not -- the most likely cause being a restore that is not idempotent,
        # which the same-noise control cannot see (it restores both columns equally for every
        # sibling, so an inexact second restore still leaves the siblings agreeing).
        "swap_asymmetry": float(a["gain_ab"].mean() - a["gain_ba"].mean()),
        "ci_swap_asymmetry": list(bootstrap_ci(a["gain_ab"] - a["gain_ba"],
                                               reps=reps_boot, rng=rng,
                                               cluster=st.episode)),
        "hindsight": float(a["hindsight"].mean()),
        "any_of_k": float(a["any_of_k"].mean()),
        "mean_spread_a": float(a["spread"].mean()),
        "frac_informative": float((a["spread"] > 0).mean()),
        "frac_degenerate_groups": float(st.degenerate.mean()),
        "reliability_r": r,
        "reliability_spearman_mean": rho_s,
        "reliability_spearman_n_groups": n_rho_s,
        "perfect_gain_extrapolation": (float(g.mean() / np.sqrt(r))
                                       if np.isfinite(r) and r > 0 else float("nan")),
        "n_episodes": int(np.unique(st.episode).size),
    })
    if with_null and reps_perm > 0:
        p, nm, nsd = permutation_p(st.y_a, st.y_b, reps=reps_perm, rng=rng,
                                   tie_break=tie_break, obs=float(g.mean()))
        out.update({"perm_p": p, "perm_null_mean": nm, "perm_null_sd": nsd,
                    "perm_reps": reps_perm})
    return out


# ======================================================================================
# stratification
# ======================================================================================
def _spread_bins(spread: np.ndarray, binary: bool) -> tuple[np.ndarray, list[str]]:
    """Bin index and labels for the realized selection-column spread."""
    if binary:
        lab = ["all siblings agree in A", "disagree in A"]
        return (spread > 0).astype(int), lab
    # Named in WORDS, not with a `>`: `tex_escape` has no case for it and under `usetex` a
    # bare `>` in text mode renders as an inverted question mark -- silently, in the figure
    # only, which is where it was found.
    lab = ["0 (exact)", "(0, 0.15]", "(0.15, 0.30]", "above 0.30"]
    idx = np.digitize(spread, SPREAD_EDGES, right=True)
    return idx, lab


def strata(st: Stacked, a: dict, a_exp: dict, *, binary: bool, reps_boot: int,
           rng: np.random.Generator) -> dict:
    """Honest gain in every stratum the report asks for.

    Returns:
        ``{stratification: [{name, n, mean_gain, ci, hindsight, any_of_k}, ...]}``.
    """
    # `a` / `a_exp` are the SAME arrays the headline was computed from -- see
    # :func:`summarize` on why they are not recomputed here.
    #
    # `a_exp` is the same estimator with the tie draw integrated out. It matters most in exactly the
    # stratum the spread axis creates: a group whose siblings all TIE in the selection column
    # contributes zero honest gain "by construction", and that is exact in the B->A arm (every
    # A is equal, so the score IS the baseline) but only true IN EXPECTATION in the A->B arm,
    # where a uniform tie draw still lands on some sibling's B outcome. Under `expected` the
    # row is exactly 0.0 and the claim can be checked rather than asserted.
    sidx, slab = _spread_bins(a["spread"], binary)
    tidx = np.clip((st.ep_frac * 3).astype(int), 0, 2)
    axes = {
        "family": [(str(v), st.klass == v) for v in sorted(set(st.klass.tolist()))],
        "cell": [(str(v), st.cell == v) for v in sorted(set(st.cell.tolist()))],
        "spread_bin": [(slab[j], sidx == j) for j in range(len(slab))],
        "episode_position": [(nm, tidx == j) for j, nm in
                             enumerate(("early (first third)", "middle", "late (last third)"))],
    }
    out: dict = {}
    for axis, items in axes.items():
        rows = []
        for name, m in items:
            g = a["gain"][m]
            row = {"name": name, "n": int(m.sum())}
            if m.any():
                row.update({
                    "mean_gain": float(g.mean()),
                    "mean_gain_expected_tiebreak": float(a_exp["gain"][m].mean()),
                    "ci_group": list(bootstrap_ci(g, reps=reps_boot, rng=rng)),
                    "ci_episode_clustered": list(bootstrap_ci(
                        g, reps=reps_boot, rng=rng, cluster=st.episode[m])),
                    "mean_baseline": float(a["baseline"][m].mean()),
                    "mean_oracle_score": float(a["oracle"][m].mean()),
                    "hindsight": float(a["hindsight"][m].mean()),
                    "any_of_k": float(a["any_of_k"][m].mean()),
                    "reliability_r": reliability(st.y_a[m], st.y_b[m]),
                })
            rows.append(row)
        out[axis] = rows
    return out


# ======================================================================================
# output
# ======================================================================================
def _fmt_ci(row: dict, key: str = "ci_episode_clustered") -> str:
    """``[lo, hi]`` to 4 dp, or a dash when the stratum is empty."""
    ci = row.get(key)
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "--"


def write_markdown(path: str, blocks: dict, meta: dict) -> None:
    """One table per stratification, per outcome version."""
    L = ["# Split-continuation selection gain, by stratum", "",
         "Honest gain = select the best sibling on one continuation, score it on the OTHER.",
         "`hindsight` scores on the draw it selected with and is an upper bound with no",
         "predictive content; `any_of_K` is algebraically the same number on column A.",
         "Intervals are 95% percentile bootstrap, clustered on the EPISODE -- successive",
         "probed re-plans of one episode share a robot that is either coping or already in",
         "trouble, so the group is not the independent unit.", "",
         "The spread axis is defined on the SELECTION column A while the statistic is",
         "swap-averaged over both directions, so its `all siblings agree` row is exactly zero",
         "in the B->A arm and zero only in expectation in the A->B arm; the tie-integrated",
         "column is where that row reads exactly 0.0000.", ""]
    for oc in ("continuous", "binary", "binary_with_terminal"):
        if oc not in blocks:
            continue
        b = blocks[oc]
        L += [f"## {oc}", "",
              f"n = {b['agg']['n_groups']} groups, K = {b['agg'].get('k', '?')}, "
              f"{b['agg'].get('n_episodes', '?')} episodes", ""]
        for axis, rows in b["strata"].items():
            L += [f"### by {axis}", "",
                  "| stratum | n | base | oracle | honest gain | 95% CI (episode-clustered) "
                  "| gain, tie draw integrated | hindsight | any-of-K | r |",
                  "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
            for r in rows:
                if not r["n"]:
                    L.append(f"| {r['name']} | 0 | -- | -- | -- | -- | -- | -- | -- | -- |")
                    continue
                L.append(
                    f"| {r['name']} | {r['n']} | {r['mean_baseline']:.4f} | "
                    f"{r['mean_oracle_score']:.4f} | {r['mean_gain']:+.4f} | "
                    f"{_fmt_ci(r)} | {r['mean_gain_expected_tiebreak']:+.4f} | "
                    f"{r['hindsight']:+.4f} | {r['any_of_k']:+.4f} | "
                    f"{r['reliability_r']:+.3f} |")
            L.append("")
    L += ["## provenance", "", "```json", json.dumps(meta, indent=2), "```", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def make_figure(path_no_ext: str, blocks: dict) -> None:
    """Honest gain against realized spread, hindsight and any-of-K as faint references.

    Two panels, one per outcome version.  The spread axis is the one stratification where the
    three arms separate for a reason worth seeing: a group whose siblings all came out the
    same cannot be chosen between, so the honest arm is pinned at zero there while both
    same-draw arms are already positive -- which is the whole bias, drawn.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import plot_style  # noqa: PLC0415

    plot_style.apply_paper_style()
    C = plot_style.COLORS
    esc = plot_style.tex_escape
    names = {"continuous": "continuous return", "binary": "reached the goal"}
    shown = [oc for oc in ("continuous", "binary") if oc in blocks]
    # Derived from the style's own full-width size rather than restated: two panels of
    # rotated categorical labels need about 1.7x the width of one.
    base_w, base_h = plot_style.DOUBLE_COL
    fig, axes = plt.subplots(1, max(len(shown), 1), squeeze=False,
                             figsize=(base_w * 0.85 * max(len(shown), 1), base_h * 1.25))
    for ax, oc in zip(axes[0], shown):
        rows = blocks[oc]["strata"]["spread_bin"]
        x = np.arange(len(rows))
        gain = np.array([r.get("mean_gain", np.nan) for r in rows], dtype=np.float64)
        ci = np.array([r.get("ci_episode_clustered", [np.nan, np.nan]) for r in rows],
                      dtype=np.float64)
        ax.axhline(0.0, color="0.65", lw=1.2, zorder=1)
        ax.plot(x, [r.get("hindsight", np.nan) for r in rows], "s--", color=C["grey"],
                alpha=0.9, lw=2.0, ms=7, label="hindsight (same draw)", zorder=2)
        ax.plot(x, [r.get("any_of_k", np.nan) for r in rows], "^:", color=C["orange"],
                alpha=0.55, lw=2.0, ms=7, label="any-of-K", zorder=2)
        ax.errorbar(x, gain, yerr=np.abs(np.vstack([gain - ci[:, 0], ci[:, 1] - gain])),
                    fmt="o-", color=C["blue"], lw=3.0, ms=10, capsize=5, capthick=2,
                    label="honest (held-out draw)", zorder=3)
        for xi, r in zip(x, rows):
            ax.annotate(f"n={r['n']}", (xi, 0.02), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=13, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels([esc(r["name"]) for r in rows], rotation=18, ha="right",
                           fontsize=14)
        ax.set_xlim(-0.45, len(rows) - 0.55)
        ax.set_xlabel("between-sibling spread, selection draw")
        # On BOTH panels: the two outcomes are on different scales and share no axis, and one
        # label between them would read as a shared one.
        ax.set_ylabel("gain from choosing best of K")
        ax.legend(title=names[oc], frameon=False, fontsize=14, loc="upper left",
                  title_fontsize=15)
    fig.align_xlabels(axes[0])
    plot_style.save_figure(
        fig, path_no_ext,
        caption="Split-continuation selection gain against the realized between-sibling "
                "spread. The honest arm selects the best sibling on one continuation draw and "
                "is paid out on an independent one; the hindsight and any-of-K arms score the "
                "draw they selected with and are shown only to size how much of the apparent "
                "gain is the maximum of K noisy numbers (they are algebraically the same "
                "quantity on the selection column). Bars are 95% percentile bootstrap "
                "intervals clustered on the episode. Groups whose siblings all agree in the "
                "selection column contribute exactly zero honest gain by construction, so "
                "the leftmost populated bin is the estimator's own floor.")
    plt.close(fig)


def paragraph(blocks: dict) -> str:
    """The short statement the report asks for, in the units the numbers are in."""
    out = []
    for oc in ("continuous", "binary"):
        if oc not in blocks:
            continue
        a = blocks[oc]["agg"]
        if not a.get("n_groups"):
            continue
        ci = a["ci_episode_clustered"]
        hind = a["hindsight"]
        luck = (100 * (1 - a["mean_gain"] / hind)) if abs(hind) > 1e-12 else float("nan")
        out.append(
            f"[{oc}] Over {a['n_groups']} probe states ({a['n_episodes']} episodes, K="
            f"{a['k']}), a random pick scores {a['mean_baseline']:.4f} and a pick made on one "
            f"continuation and paid out on an independent one scores "
            f"{a['mean_oracle_score']:.4f}: a gain of {a['mean_gain']:+.4f} "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] (95% episode-clustered bootstrap, permutation "
            f"p={a.get('perm_p', float('nan')):.4f}). "
            + (f"Selection was informative in {100 * a['frac_informative']:.1f}% of groups -- "
               "the rest have every sibling tied in the selection column and contribute "
               "exactly zero by construction. "
               if oc == "binary" else
               f"Every group is informative on a continuous outcome (no ties); the mean "
               f"realized between-sibling spread is {a['mean_spread_a']:.4f}. ")
            + f"The "
            f"hindsight arm, which scores the draw it selected with, reads "
            f"{hind:+.4f}, so {luck:.0f}% of the apparent best-of-K gain is the maximum of "
            "K noisy numbers."
            + (f" Selection reliability r={a['reliability_r']:.3f}, so the perfect-predictor "
               f"extrapolation is {a['perfect_gain_extrapolation']:+.4f} (bivariate-normal, "
               "order of magnitude only)." if np.isfinite(a["reliability_r"]) else ""))
    return "\n\n".join(out)


# ======================================================================================
# stages
# ======================================================================================
def control_verdict(c: dict) -> str:
    """PASS/FAIL for a ``--same_noise`` control block.

    An EMPTY control is a FAIL, not a pass. A blocking check that no data reached asserts
    nothing, and reporting it green is worse than not running it -- the same defect as a
    sensitivity threshold computed from an empty control group.
    """
    if not c.get("n_groups"):
        return ("FAIL: the control has NO usable groups, so it proves nothing. Collect it "
                "with --second_seed_every set, like the treatment.")
    if abs(c["mean_gain"]) < 1e-12 and c["mean_spread_a"] == 0.0:
        return "PASS"
    return ("FAIL: the two columns are not what they claim (a --same_noise run's siblings are "
            "bit-identical) and no number above is usable")


def stage_analyze(args) -> int:
    """Load, compute, write, print."""
    rng_seed = int(args.seed)
    versions = [("continuous", False), ("binary", False)]
    if args.include_terminal_binary:
        versions.append(("binary_with_terminal", True))

    blocks: dict = {}
    control: dict = {}
    cells_meta: list[dict] = []
    for label, incl in versions:
        oc = "binary" if label.startswith("binary") else "continuous"
        cells = [load_cell(d, outcome=oc, include_terminal=incl) for d in args.run]
        bad = [c.run for c in cells if c.same_noise]
        if bad:
            raise SystemExit(
                f"--run {bad} is a --same_noise control. Pass it as --control: its siblings "
                "are bit-identical by construction, so pooling it with treatment runs dilutes "
                "the gain toward zero with groups that cannot inform it and turns the blocking "
                "check into part of the estimate.")
        st = stack(cells)
        # One generator per version, re-seeded, so adding a version cannot move an earlier
        # one's number by consuming from a shared stream.
        rng = np.random.default_rng(rng_seed)
        # ONE tie draw for this dataset, then everything downstream reads it, so the strata
        # average back to the headline exactly.
        a = arms(st.y_a, st.y_b, rng, args.tie_break)
        a_exp = arms(st.y_a, st.y_b, rng, "expected")
        blocks[label] = {
            "agg": summarize(st, a, a_exp, reps_boot=args.boot, reps_perm=args.perm, rng=rng,
                             tie_break=args.tie_break),
            "strata": strata(st, a, a_exp, binary=oc == "binary", reps_boot=args.boot,
                             rng=rng),
        }
        if label == "continuous":
            cells_meta = [{"run": c.run, "tile": c.tile, "speed": c.speed, "klass": c.klass,
                           "groups_total": c.n_groups_total, "groups_used": int(c.y_a.shape[0]),
                           "drops": c.drops, "verdicts": c.verdicts} for c in cells]

    for label, incl in versions:
        oc = "binary" if label.startswith("binary") else "continuous"
        if not args.control:
            continue
        ccells = [load_cell(d, outcome=oc, include_terminal=incl) for d in args.control]
        loose = [c.run for c in ccells if not c.same_noise]
        if loose:
            raise SystemExit(f"--control {loose} was NOT collected with --same_noise; it is a "
                             "treatment run and would pass a control that proves nothing.")
        cst = stack(ccells)
        rng = np.random.default_rng(rng_seed + 1)
        control[label] = summarize(
            cst, arms(cst.y_a, cst.y_b, rng, args.tie_break),
            arms(cst.y_a, cst.y_b, rng, "expected"),
            reps_boot=args.boot, reps_perm=0, rng=rng, tie_break=args.tie_break)
        # Resolved HERE, before the json is written. It used to be stamped during the printed
        # report, which runs after the write -- so the file never carried the verdict of the
        # one check that decides whether the rest of it means anything.
        control[label]["verdict"] = control_verdict(control[label])

    meta = {"runs": args.run, "controls": args.control, "cells": cells_meta,
            "seed": rng_seed, "bootstrap_reps": args.boot, "permutation_reps": args.perm,
            "tie_break": args.tie_break, "numpy": np.__version__}
    os.makedirs(args.out, exist_ok=True)
    payload = {"provenance": meta, "outcomes": blocks, "control_same_noise": control,
               "paragraph": paragraph(blocks)}
    with open(os.path.join(args.out, "oracle_gain.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    write_markdown(os.path.join(args.out, "oracle_gain_by_stratum.md"), blocks, meta)
    if not args.no_figure:
        make_figure(os.path.join(args.out, "oracle_gain"), blocks)

    # ---- printed report ----------------------------------------------------------------
    print("\n=== split-continuation selection gain ===")
    for c in cells_meta:
        print(f"   {c['run']:>22s}  tile {c['tile']:>3d} @ {c['speed']:.1f} m/s "
              f"({c['klass']})  {c['groups_used']:>4d}/{c['groups_total']:>4d} groups usable"
              f"   verdicts {c['verdicts']}")
    for label in blocks:
        a = blocks[label]["agg"]
        print(f"\n-- {label} --")
        if not a["n_groups"]:
            print("   no usable groups")
            continue
        print(f"   groups {a['n_groups']} of {a['n_groups_before_drop']} probed  "
              f"(dropped: {a['drops']})")
        print(f"   continuation verdicts over ALL branches: {a['verdicts']}")
        print(f"   base (random pick)      {a['mean_baseline']:.4f}")
        print(f"   oracle-selected, held-out draw  {a['mean_oracle_score']:.4f}")
        print(f"   HONEST GAIN             {a['mean_gain']:+.4f}  "
              f"group CI {a['ci_group'][0]:+.4f}..{a['ci_group'][1]:+.4f}  "
              f"episode-clustered {a['ci_episode_clustered'][0]:+.4f}.."
              f"{a['ci_episode_clustered'][1]:+.4f}")
        asym = a["ci_swap_asymmetry"]
        print(f"   arms: A->B {a['gain_ab']:+.4f}   B->A {a['gain_ba']:+.4f}   "
              f"asymmetry {a['swap_asymmetry']:+.4f} [{asym[0]:+.4f}, {asym[1]:+.4f}]"
              f"{'' if asym[0] <= 0.0 <= asym[1] else '  <- EXCLUDES ZERO: the two '
                 'continuations are not exchangeable, which is a defect and not a result'}")
        print(f"   expected-tiebreak gain  {a['mean_gain_expected_tiebreak']:+.4f} "
              "(the same estimator with the tie draw integrated out)")
        print(f"   hindsight {a['hindsight']:+.4f}   any-of-K {a['any_of_k']:+.4f}   "
              "<- both score the draw they picked on")
        if "perm_p" in a:
            print(f"   permutation null {a['perm_null_mean']:+.4f} +/- {a['perm_null_sd']:.4f}"
                  f"   p={a['perm_p']:.4f} ({a['perm_reps']} reps)")
        print(f"   informative groups {100 * a['frac_informative']:.1f}%   "
              f"reliability r {a['reliability_r']:+.4f} (pooled, demeaned Pearson)   "
              f"perfect-predictor extrapolation {a['perfect_gain_extrapolation']:+.4f}")
        print(f"   the SAME data as a mean within-group Spearman -- the statistic "
              f"`critic_branch_metrics.label_ceiling` reports -- is "
              f"{a['reliability_spearman_mean']:+.4f} over "
              f"{a['reliability_spearman_n_groups']} groups. A DIFFERENT statistic, not a "
              "second estimate: it weights every group equally where the pooled figure "
              "weights a group by its own spread, so the two separate exactly as far as "
              "reliability varies with spread.")
        if label in control:
            c = control[label]
            print(f"   *** SAME-NOISE CONTROL: n={c['n_groups']} groups, gain "
                  f"{c.get('mean_gain', float('nan')):+.9f}, spread "
                  f"{c.get('mean_spread_a', float('nan')):.9f} -- {c['verdict']} ***")
    print("\n" + paragraph(blocks) + "\n")
    print(f"[oracle_gain] wrote {args.out}/oracle_gain.json, "
          f"oracle_gain_by_stratum.md{'' if args.no_figure else ', oracle_gain.png'}")
    return 0


def _synth(n_groups, k, sig_mu, sig_eps, rng):
    """Two noisy draws of K siblings whose true means differ by a KNOWN amount."""
    mu = rng.normal(0.0, sig_mu, size=(n_groups, k))
    ya = mu + rng.normal(0.0, sig_eps, size=(n_groups, k))
    yb = mu + rng.normal(0.0, sig_eps, size=(n_groups, k))
    return mu, ya, yb


def stage_selfcheck(args) -> int:
    """Run the estimator on synthetic data whose answer is known in closed form.

    Four cases, and every one of them has a number this can be wrong against:

    1. **No sibling signal** (``sig_mu = 0``): the honest gain must be ~0 while the hindsight
       arm is large.  This is the case a hindsight estimator gets confidently wrong.
    2. **Known signal**: for ``y = mu + eps`` with both columns i.i.d. Gaussian, selecting on
       one column and scoring the other has expectation ``rho * sig_mu * E[max of K]`` with
       ``rho = sig_mu / sqrt(sig_mu^2 + sig_eps^2)``.
    3. **The extrapolation**: dividing the honest gain by ``sqrt(r)`` must recover the TRUE
       oracle gain ``sig_mu * E[max of K]``, exactly, under this model.
    4. **Same noise**: siblings identical within a group -> gain exactly 0.0, and a binary
       deterministic group ``{0,0,0,1}`` -> gain exactly ``1 - 1/K``.
    """
    rng = np.random.default_rng(int(args.seed))
    k, n = 4, 40000
    fails = []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        print(f"   {'PASS' if ok else 'FAIL'}  {name:52s} got {got:+.4f}  want "
              f"{want:+.4f} +/- {tol:.4f}")
        if not ok:
            fails.append(name)

    print("\n=== oracle_gain selfcheck (synthetic, closed-form answers) ===")
    print("\n-- 1. no sibling signal: the honest arm must return 0, hindsight must not --")
    _mu, ya, yb = _synth(n, k, 0.0, 0.25, rng)
    a = arms(ya, yb, rng)
    check("no-signal honest gain", float(a["gain"].mean()), 0.0, 0.006)
    check("no-signal hindsight (the WRONG estimator)", float(a["hindsight"].mean()),
          0.25 * _EMAX_K4, 0.01)

    print("\n-- 2. known signal: honest gain = rho * sig_mu * E[max K] --")
    for sig_mu, sig_eps in ((0.30, 0.30), (0.10, 0.40), (0.40, 0.10)):
        _mu, ya, yb = _synth(n, k, sig_mu, sig_eps, rng)
        rho = sig_mu / np.sqrt(sig_mu ** 2 + sig_eps ** 2)
        a = arms(ya, yb, rng)
        check(f"honest gain (sig_mu {sig_mu}, sig_eps {sig_eps})",
              float(a["gain"].mean()), rho * sig_mu * _EMAX_K4, 0.012)
        r = reliability(ya, yb)
        check(f"reliability r == rho^2 ({sig_mu}, {sig_eps})", r, rho ** 2, 0.02)
        check(f"perfect-predictor extrapolation == sig_mu * E[max K] ({sig_mu}, {sig_eps})",
              float(a["gain"].mean()) / np.sqrt(max(r, 1e-9)), sig_mu * _EMAX_K4, 0.03)

    print("\n-- 3. the permutation null is a real null --")
    _mu, ya, yb = _synth(4000, k, 0.0, 0.25, rng)
    p0, nm, _sd = permutation_p(ya, yb, reps=200, rng=rng, tie_break="random")
    check("null-data permutation null mean", nm, 0.0, 0.006)
    print(f"   null-data p = {p0:.4f} (must NOT be small; a real null cannot reject noise)")
    if p0 < 0.01:
        fails.append("permutation p on signal-free data")
    _mu, ya, yb = _synth(4000, k, 0.30, 0.30, rng)
    p1, _nm, _sd = permutation_p(ya, yb, reps=200, rng=rng, tie_break="random")
    print(f"   signal p = {p1:.4f} (must be small; the null must have power)")
    if p1 > 0.01:
        fails.append("permutation p on signal data")

    print("\n-- 4. degenerate cases, exact --")
    same = rng.normal(0, 1, size=(500, 1)) @ np.ones((1, k))
    a = arms(same, same.copy(), rng)
    check("identical siblings -> gain exactly 0", float(a["gain"].mean()), 0.0, 0.0)
    det = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (500, 1))
    a = arms(det, det.copy(), rng)
    check("binary {0,0,0,1} -> gain exactly 1 - 1/K", float(a["gain"].mean()),
          1.0 - 1.0 / k, 0.0)

    print("\n-- 5. the guardrail: the honest arm never reads its own selection column --")
    # B is pure noise, uncorrelated with A. An estimator that leaked A into the score would
    # return the hindsight number here; the honest one must return 0.
    _mu, ya, _yb = _synth(n, k, 0.40, 0.10, rng)
    yb_indep = rng.normal(0.0, 0.41, size=ya.shape)
    a = arms(ya, yb_indep, rng)
    check("independent-B honest gain", float(a["gain"].mean()), 0.0, 0.01)

    print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
    return 1 if fails else 0


def main() -> int:
    """Entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="analyze", choices=("analyze", "selfcheck"))
    p.add_argument("--out", default="")
    p.add_argument("--run", action="append", default=[],
                   help="a branch_probe --stage run directory whose siblings ALL carry two "
                        "continuations. Repeatable; cells are pooled and also reported "
                        "separately.")
    p.add_argument("--control", action="append", default=[],
                   help="a --same_noise branch_probe directory. The blocking check: its "
                        "siblings are bit-identical, so the measured gain must be 0.")
    p.add_argument("--include_terminal_binary", action="store_true",
                   help="additionally report the BINARY outcome with in-chunk-terminal "
                        "siblings kept, using their branch verdict as both columns (they have "
                        "no continuation variance). Sizes the bias the headline drop rule "
                        "introduces by removing exactly the decisive states.")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the tie-break, the bootstrap and the permutation null. Logged "
                        "into oracle_gain.json.")
    p.add_argument("--tie_break", default="random", choices=("random", "expected"),
                   help="'random' is the estimator as specified: one of the tied siblings, "
                        "uniformly. 'expected' integrates that draw out analytically; both "
                        "are always reported, this picks the headline.")
    p.add_argument("--no_figure", action="store_true")
    args = p.parse_args()
    if args.stage == "selfcheck":
        return stage_selfcheck(args)
    if not args.run or not args.out:
        raise SystemExit("--stage analyze needs --out and at least one --run")
    return stage_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
