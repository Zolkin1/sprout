# Copyright 2026 Zachary Olkin. All rights reserved.

"""Estimators for the eval report. numpy only -- no matplotlib, no scipy, no argparse.

The one thing this module exists to get right is the **unit of analysis**.  Episodes are
nested: R draws within a clip, C clips within a tile, T tiles within a family.  Pooling
episodes and computing a binomial interval treats all of them as independent, and simulated
against the real design shape that interval has an actual coverage of **14-70%** against a
nominal 95% -- and it gets *worse* with more draws per clip, because it keeps shrinking around
the wrong quantity.  A tile-clustered bootstrap holds ~90%.

So every rate here is computed as **clip-averaged, then bootstrapped over TILES**, and every
returned number carries which level its interval came from.  Two clips on one tile share the
terrain geometry, so the tile is the resampling unit, not the clip.

``no scipy`` is deliberate and pinned by a test: scipy IS installed in this env, so nothing
but an explicit assertion stops it drifting in, and the report has to stay runnable in the
small ``env_transfer`` too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Fields every reported number carries, so "which level is this error bar?" is answered by
#: the data rather than by prose.
@dataclass
class Estimate:
    """One reported number with its interval and provenance."""

    value: float
    ci_lo: float
    ci_hi: float
    method: str
    level: str
    n_episodes: int = 0
    n_clips: int = 0
    n_tiles: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-friendly form."""
        return {"value": self.value, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
                "method": self.method, "level": self.level,
                "n_episodes": self.n_episodes, "n_clips": self.n_clips,
                "n_tiles": self.n_tiles, **({"extra": self.extra} if self.extra else {})}


def summarise(col: np.ndarray) -> dict:
    """``{n, mean, median, p25, p75, min, max}``, NaN-ignoring.

    Same field set and spelling as ``clip_ab_sweep._summarise``, so the two tools' JSON stays
    diffable rather than merely similar.
    """
    v = np.asarray(col, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p25": float("nan"),
                "p75": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max())}


def _z(conf: float) -> float:
    """Two-sided normal quantile, from the stdlib (no scipy)."""
    from statistics import NormalDist  # noqa: PLC0415

    return float(NormalDist().inv_cdf(0.5 + conf / 2.0))


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a proportion.

    Used where Wald is indefensible: at ``n`` of 4 (a small cell) or at ``p`` of 0 or 1, Wald
    gives zero width or bounds outside [0, 1).

    Returns:
        ``(point, lo, hi)``.
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    z = _z(conf)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def clip_table(clip_ids: np.ndarray, tile_ids: np.ndarray, values: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse episodes to one row per clip.

    Args:
        clip_ids: ``[n]`` clip id per episode.
        tile_ids: ``[n]`` tile (cluster) id per episode.
        values: ``[n]`` the per-episode quantity (0/1 for a rate).

    Returns:
        ``(clips, tiles, means, counts)`` -- one entry per distinct clip, with the clip's
        tile, the mean of its episodes, and how many it had.
    """
    clip_ids = np.asarray(clip_ids)
    order = np.argsort(clip_ids, kind="stable")
    c, t, v = clip_ids[order], np.asarray(tile_ids)[order], np.asarray(values, float)[order]
    uniq, start = np.unique(c, return_index=True)
    ends = np.append(start[1:], c.size)
    means = np.array([np.nanmean(v[a:b]) if b > a else np.nan
                      for a, b in zip(start, ends)])
    counts = (ends - start).astype(np.int64)
    tiles = t[start]
    return uniq, tiles, means, counts


def cluster_bootstrap(clip_means: np.ndarray, clip_tiles: np.ndarray,
                      clip_counts: np.ndarray | None = None,
                      *, binary: bool = False, boot: int = 2000, conf: float = 0.95,
                      seed: int = 0) -> tuple[float, float, float, str]:
    """Clip-averaged point estimate with a tile-clustered percentile interval.

    Two stages per replicate: resample TILES with replacement (carrying all their clips),
    then -- for a binary outcome -- redraw each selected clip's successes from
    ``Binomial(n_c, p_c)``.  The second stage matters because one report spans cells with very
    different draw counts: at R=25 the within-clip noise is a few percent and negligible
    against tile-to-tile spread, at R=4 it dominates.

    The point estimate is the unweighted mean over clips, not over episodes: ``n_c`` is a
    design choice, and pooling would silently reweight towards whichever clips happened to run
    more episodes.

    Args:
        clip_means: Per-clip value.
        clip_tiles: Per-clip cluster id.
        clip_counts: Per-clip episode count (needed when ``binary``).
        binary: Whether the quantity is a proportion, enabling the within-clip stage.
        boot: Replicates.
        conf: Interval mass.
        seed: RNG seed; a report that changes on re-run is not a report.

    Returns:
        ``(point, lo, hi, method)``.  When every clip agrees exactly the percentile interval
        collapses to a point, which is not a defensible interval -- so it falls back to a
        Wilson interval at the CLUSTER level and says so in ``method``.
    """
    means = np.asarray(clip_means, dtype=float)
    tiles = np.asarray(clip_tiles)
    ok = np.isfinite(means)
    means, tiles = means[ok], tiles[ok]
    counts = (np.asarray(clip_counts, dtype=np.int64)[ok] if clip_counts is not None
              else np.ones(means.size, dtype=np.int64))
    if means.size == 0:
        return (float("nan"), float("nan"), float("nan"), "empty")
    point = float(means.mean())

    uniq_tiles, tile_index = np.unique(tiles, return_inverse=True)
    n_tiles = uniq_tiles.size
    by_tile = [np.flatnonzero(tile_index == i) for i in range(n_tiles)]
    if n_tiles < 2:
        # One cluster makes every resample identical, so the percentile interval would be a
        # silently zero-width point. Say so instead.
        return (point, float("nan"), float("nan"), "single_cluster")

    rng = np.random.default_rng(seed)
    reps = np.empty(boot, dtype=float)
    for b in range(boot):
        pick = rng.integers(0, n_tiles, n_tiles)
        idx = np.concatenate([by_tile[i] for i in pick])
        vals = means[idx]
        if binary:
            n = counts[idx]
            vals = rng.binomial(n, np.clip(vals, 0.0, 1.0)) / np.maximum(n, 1)
        reps[b] = vals.mean()
    lo, hi = np.percentile(reps, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    if not np.isfinite(lo) or hi - lo < 1e-12:
        k = int(round(point * n_tiles))
        _, wlo, whi = wilson(k, n_tiles, conf)
        return (point, float(wlo), float(whi), "wilson_clusters")
    return (point, float(lo), float(hi), "cluster_bootstrap")


def rate(clip_ids, tile_ids, values, *, boot: int = 2000, conf: float = 0.95,
         seed: int = 0, binary: bool = True) -> Estimate:
    """Clip-averaged rate (or mean) with a tile-clustered interval.

    Also records the episode-POOLED value in ``extra``, always: the two coincide only when
    every clip ran the same number of episodes, so any divergence is a signal that the design
    is unbalanced rather than a rounding artefact.
    """
    clips, tiles, means, counts = clip_table(clip_ids, tile_ids, values)
    point, lo, hi, method = cluster_bootstrap(
        means, tiles, counts, binary=binary, boot=boot, conf=conf, seed=seed)
    v = np.asarray(values, float)
    pooled = float(np.nanmean(v)) if v.size else float("nan")
    return Estimate(
        value=point, ci_lo=lo, ci_hi=hi, method=method, level="clip_mean/tile_cluster",
        n_episodes=int(np.isfinite(v).sum()), n_clips=int(clips.size),
        n_tiles=int(np.unique(tiles).size),
        extra={"episode_pooled": pooled,
               "unbalanced": bool(counts.size and counts.min() != counts.max()),
               "episodes_per_clip_min": int(counts.min()) if counts.size else 0,
               "episodes_per_clip_max": int(counts.max()) if counts.size else 0,
               "across_clip_sd": float(np.nanstd(means)) if means.size else float("nan")},
    )


def kaplan_meier(duration: np.ndarray, event: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Product-limit survival estimate.

    Successes have no failure time; they are right-CENSORED, not events.  Averaging a
    failure-time column over failures only conditions on failing, and biases in the
    flattering direction: a policy that fails only at the hard late moments gets a longer
    mean time-to-failure and looks better.

    Tie rule, stated because it changes the curve: a censoring tied with an event counts as
    at risk at that time (censoring after event).

    Args:
        duration: ``[n]`` observed time (or step) per episode.
        event: ``[n]`` 1 = failure observed, 0 = censored.

    Returns:
        ``(times, survival, at_risk)`` at the distinct event times, with ``S`` starting at 1.
    """
    t = np.asarray(duration, dtype=float)
    e = np.asarray(event).astype(bool)
    good = np.isfinite(t)
    t, e = t[good], e[good]
    if t.size == 0:
        return (np.array([0.0]), np.array([1.0]), np.array([0]))
    times = np.unique(t[e])
    surv, risk, s = [], [], 1.0
    for ti in times:
        n_at = int((t >= ti).sum())          # censoring after event
        d = int(((t == ti) & e).sum())
        if n_at > 0:
            s *= (1.0 - d / n_at)
        surv.append(s)
        risk.append(n_at)
    return (np.concatenate([[0.0], times]),
            np.concatenate([[1.0], np.asarray(surv)]),
            np.concatenate([[int(t.size)], np.asarray(risk, dtype=np.int64)]))


def rmst(times: np.ndarray, surv: np.ndarray, tau: float) -> float:
    """Restricted mean survival time: the area under ``S`` up to ``tau``.

    The censoring-correct scalar for a table.  The mean is undefined whenever ``S`` never
    reaches 0 (any mostly-surviving policy) and the median is undefined unless ``S`` crosses
    0.5, so neither can be tabulated unconditionally; this always exists.
    """
    t = np.asarray(times, dtype=float)
    s = np.asarray(surv, dtype=float)
    keep = t <= tau
    t2 = np.concatenate([t[keep], [tau]])
    s2 = np.concatenate([s[keep], [s[keep][-1] if keep.any() else 1.0]])
    return float(np.sum(np.diff(t2) * s2[:-1]))


def median_survival(times: np.ndarray, surv: np.ndarray) -> float:
    """First time ``S`` drops to or below 0.5, or NaN when it never does.

    NaN rather than the last observed time: reporting "the median is 20 s" when the curve
    never crossed 0.5 states a fact that was not measured.
    """
    t = np.asarray(times, dtype=float)
    s = np.asarray(surv, dtype=float)
    hit = np.flatnonzero(s <= 0.5)
    return float(t[hit[0]]) if hit.size else float("nan")


def paired_delta(clip_ids_a, tile_a, values_a, clip_ids_b, tile_b, values_b,
                 *, boot: int = 2000, conf: float = 0.95, seed: int = 0) -> Estimate:
    """Difference B - A, paired on clip id, with a tile-clustered interval.

    Cells within a run see the same clips, so pairing removes clip-difficulty variance -- the
    dominant component.  The SAME tile resample is applied to both arms in each replicate,
    which is what makes it a paired interval rather than two independent ones subtracted.

    Deliberately not episode-paired: episode *i* of a clip in one cell is a different draw
    from episode *i* in another, so there is no one-to-one correspondence to exploit.
    """
    ca, ta, ma, na = clip_table(clip_ids_a, tile_a, values_a)
    cb, tb, mb, nb = clip_table(clip_ids_b, tile_b, values_b)
    common, ia, ib = np.intersect1d(ca, cb, return_indices=True)
    if common.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), "no_common_clips", "clip")
    d = mb[ib] - ma[ia]
    tiles = ta[ia]
    point, lo, hi, method = cluster_bootstrap(d, tiles, binary=False, boot=boot,
                                              conf=conf, seed=seed)
    p = sign_flip_test(d, tiles, boot=boot, seed=seed)
    return Estimate(point, lo, hi, f"paired_{method}", "clip_delta/tile_cluster",
                    n_episodes=int(na[ia].sum() + nb[ib].sum()), n_clips=int(common.size),
                    n_tiles=int(np.unique(tiles).size),
                    extra={"permutation_p": p, "n_pairs": int(common.size),
                           "n_positive": int((d > 0).sum()), "n_zero": int((d == 0).sum())})


def sign_flip_test(deltas: np.ndarray, tiles: np.ndarray, *, boot: int = 2000,
                   seed: int = 0) -> float:
    """Two-sided p from flipping the sign of whole TILES.

    A cluster-level randomisation test: no distributional assumption, and it respects the
    nesting.  Flipping individual clips would treat two clips on one tile as independent and
    return an impossibly small p.

    Not a paired t-test: with ~10 clusters and a bounded outcome the normal approximation has
    nothing to stand on.
    """
    d = np.asarray(deltas, dtype=float)
    t = np.asarray(tiles)
    ok = np.isfinite(d)
    d, t = d[ok], t[ok]
    if d.size == 0:
        return float("nan")
    uniq, inv = np.unique(t, return_inverse=True)
    obs = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(boot):
        sign = rng.choice(np.array([-1.0, 1.0]), uniq.size)[inv]
        if abs(float((d * sign).mean())) >= obs - 1e-15:
            hits += 1
    return float((hits + 1) / (boot + 1))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment. Monotone and bounded at 1."""
    items = sorted(((v, k) for k, v in pvalues.items() if np.isfinite(v)))
    m = len(items)
    out: dict[str, float] = {k: float("nan") for k in pvalues}
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
