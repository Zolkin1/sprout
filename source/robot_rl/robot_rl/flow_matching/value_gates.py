# Copyright 2026 Zachary Olkin. All rights reserved.

"""The three things worth asking of a trackability critic, and why aggregate AUC is not one.

A single held-out AUC over every recorded state is the obvious gate and it is close to
uninformative here.  Most states in a rollout are on flat approach where nothing has been
decided yet, so a critic that is right about those and blind at the decisive moment scores
well, and one that is uninformative early and sharp at the riser scores badly.  Worse, a high
aggregate AUC is exactly what a critic that has merely learned WHICH FAMILY it is looking at
would produce -- state difficulty, not the thing the advantage needs.

So:

1. :func:`calibration` -- **testable even where discrimination is impossible.** Bin states by
   predicted ``P(fail)`` and compare against the empirical failure frequency in each bin. A
   critic that says 0.2 and is right one time in five is working, and it is working even in a
   region where no critic could separate individual episodes. This is the check that does not
   presuppose predictability, which is why it comes first.

2. :func:`auc_by_phase` -- **discrimination as a function of PHASE, never in aggregate.** The
   healthy signature is a curve rising from ~0.5 early to high near the decisive moment. Flat
   at 0.5 everywhere is the failure. Flat at 0.5 only at the START is expected and correct:
   nothing has happened yet, so nothing is predictable, and a critic claiming otherwise on the
   first chunk of a stairs episode has learned the tile rather than the state.

3. :func:`lead_time` -- **the one that decides whether the method can work at all.** For each
   failure, how many chunks before the end does ``P(fail)`` rise and stay risen? If it only
   rises once the base is already past recovery, the advantage arrives AFTER the chunk that
   caused the failure and nothing is attributable to any action -- which would be a red flag
   pointing at bias rather than at sampling variance, since it would mean no chunk the
   generator emitted was identifiable as the cause. It also **sets n empirically**: the n-step
   advantage window should span the lead time, and measuring it beats guessing it from where
   falls happen to land.

   Measured against ``first_fall_step`` as well as against the end, because those are
   different questions. Rising before the episode ENDS is weak -- the fall predicate may have
   been crossed long before. Rising before the fall PREDICATE is the claim that matters.

The cheap version of (3) is worth running before any of this: plot ``P(fail)`` over a handful
of stairs episodes, overlay successes and failures, and look at where the two families
separate.  :func:`value_traces` returns exactly that.
"""

from __future__ import annotations

import numpy as np


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC, by the Mann-Whitney rank statistic.

    Exact, ties handled by mid-ranks, and no scikit-learn dependency -- this package is
    imported by scripts that run without one.

    Args:
        scores: ``[n]`` predictions; higher should mean more likely positive.
        positive: ``[n]`` boolean labels.

    Returns:
        The AUC, or NaN when either class is empty (there is nothing to separate, which is a
        different statement from 0.5).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Mid-ranks for ties, or a critic that outputs a constant would score 1.0 or 0.0 rather
    # than the 0.5 it has earned.
    srt = s[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration(p_fail: np.ndarray, failed: np.ndarray, *, n_bins: int = 10
                ) -> dict[str, np.ndarray | float]:
    """Predicted failure probability against the observed frequency, by bin.

    Args:
        p_fail: ``[n]`` predictions in ``[0, 1]``.
        failed: ``[n]`` boolean outcomes.
        n_bins: Equal-width bins over ``[0, 1]``.

    Returns:
        ``bin_lo``, ``bin_hi``, ``n``, ``predicted`` (mean prediction in the bin),
        ``observed`` (empirical failure rate), and two scalars: ``ece`` (expected calibration
        error, the count-weighted mean ``|predicted - observed|``) and ``mce`` (the worst
        bin). Empty bins carry NaN and contribute nothing.
    """
    p = np.asarray(p_fail, dtype=np.float64)
    y = np.asarray(failed).astype(bool)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    pred = np.full(n_bins, np.nan)
    obs = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            pred[b], obs[b] = float(p[m].mean()), float(y[m].mean())
    ok = cnt > 0
    w = cnt[ok] / max(int(cnt.sum()), 1)
    gap = np.abs(pred[ok] - obs[ok])
    return {
        "bin_lo": edges[:-1], "bin_hi": edges[1:], "n": cnt,
        "predicted": pred, "observed": obs,
        "ece": float((w * gap).sum()) if ok.any() else float("nan"),
        "mce": float(gap.max()) if ok.any() else float("nan"),
    }


def auc_by_phase(p_fail: np.ndarray, failed: np.ndarray, phase: np.ndarray, *,
                 n_bins: int = 10) -> dict[str, np.ndarray]:
    """Discrimination as a function of where in the episode the state sits.

    Args:
        p_fail: ``[n]`` predictions.
        failed: ``[n]`` boolean episode outcomes, broadcast to every state of the episode.
        phase: ``[n]`` in ``[0, 1]``; 0 at the episode's first chunk, 1 at its last.
        n_bins: Equal-width phase bins.

    Returns:
        ``bin_lo``, ``bin_hi``, ``n``, ``auc`` per bin. A bin holding one class only gets
        NaN, which is not the same as 0.5 and must not be plotted as if it were.
    """
    ph = np.asarray(phase, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(ph, edges[1:-1], right=False), 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            out[b] = auc(np.asarray(p_fail)[m], np.asarray(failed)[m])
    return {"bin_lo": edges[:-1], "bin_hi": edges[1:], "n": cnt, "auc": out}


def lead_time(p_fail: np.ndarray, episode: np.ndarray, index: np.ndarray,
              failed: np.ndarray, *, tau: float = 0.5,
              reference: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """How many chunks before the end ``P(fail)`` rises and STAYS risen, per failed episode.

    Defined as a sustained crossing rather than a first touch: the earliest chunk ``t`` such
    that ``p_fail[s] >= tau`` for every ``s >= t``. A single spike three seconds out is not
    the critic knowing something; it is noise, and a first-touch definition would report it
    as a three-second lead.

    Args:
        p_fail: ``[n]`` predictions, one per chunk.
        episode: ``[n]`` episode ids.
        index: ``[n]`` chunk index within the episode.
        failed: ``[n]`` boolean episode outcome, constant within an episode.
        tau: The level that counts as "risen".
        reference: Optional ``[n]`` per-state reference index to measure the lead against
            instead of the episode's last chunk -- pass the chunk containing
            ``first_fall_step`` to ask the sharper question, whether the critic rises before
            the FALL rather than merely before the episode ends.

    Returns:
        ``episode`` (the failed episodes, in order), ``lead`` (chunks of warning; 0 means it
        never rose before the reference, NaN means it never rose at all), and ``n_chunks``.
    """
    p = np.asarray(p_fail, dtype=np.float64)
    ep = np.asarray(episode)
    ix = np.asarray(index)
    bad = np.asarray(failed).astype(bool)
    ref = None if reference is None else np.asarray(reference, dtype=np.float64)

    eps, leads, lens = [], [], []
    for e in np.unique(ep[bad]):
        m = ep == e
        order = np.argsort(ix[m])
        pe = p[m][order]
        end = float(ix[m][order][-1]) if ref is None else float(np.nanmedian(ref[m]))
        if not np.isfinite(end):
            continue
        risen = pe >= float(tau)
        # Earliest index from which it is risen for the rest of the episode.
        sustained = np.flip(np.cumprod(np.flip(risen.astype(np.int64))))
        hit = np.flatnonzero(sustained > 0)
        eps.append(e)
        lens.append(int(pe.size))
        leads.append(float("nan") if hit.size == 0
                     else max(0.0, end - float(ix[m][order][hit[0]])))
    return {"episode": np.asarray(eps), "lead": np.asarray(leads, dtype=np.float64),
            "n_chunks": np.asarray(lens, dtype=np.int64)}


def value_traces(p_fail: np.ndarray, episode: np.ndarray, index: np.ndarray,
                 failed: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Per-episode ``P(fail)`` traces, for the cheap look that comes before any statistic.

    Plotting these for a handful of episodes with successes and failures overlaid, and seeing
    WHERE the two families separate, is worth more than any of the numbers above as a first
    pass -- it is the check PI's Figure 4 is making, that the value drops at the mistake
    rather than at its consequence.

    Args:
        p_fail: ``[n]`` predictions.
        episode: ``[n]`` episode ids.
        index: ``[n]`` chunk index within the episode.
        failed: ``[n]`` boolean episode outcome.

    Returns:
        ``{episode_id: {"index": [k], "p_fail": [k], "failed": bool}}``, each sorted by index.
    """
    ep, ix = np.asarray(episode), np.asarray(index)
    p, bad = np.asarray(p_fail, dtype=np.float64), np.asarray(failed).astype(bool)
    out: dict[int, dict[str, np.ndarray]] = {}
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(ix[m])
        out[int(e)] = {"index": ix[m][order], "p_fail": p[m][order],
                       "failed": bool(bad[m][0])}
    return out
