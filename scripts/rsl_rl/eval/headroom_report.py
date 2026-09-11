# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reduce the branched-state records into the four tables and their figures.

Imports no IsaacLab; matplotlib is imported only when the figures are asked for.

**Best-of-K is computed in CLOSED FORM, not by subsampling.**  For a state where ``m`` of
``K`` candidates survived, the probability that a uniformly random ``k``-subset contains at
least one survivor is ``1 - C(K-m, k) / C(K, k)`` exactly.  Drawing one random subset per
state instead would add a whole binomial's worth of variance to a curve whose entire shape is
the result -- and the exact form costs nothing, since ``m`` is already the good-chunk count of
Table 2.

**The graded score is composed HERE, from divisors measured on this run.**  Each deviation
channel is divided by its own median over the SURVIVING candidate rows, so every term is O(1)
in the units of ordinary operation and no constant had to be guessed before the data existed.
The divisors are printed and written into the report, which is what makes the weighting a
reporting decision that can be redone against the same raw file.

**The noise floor is a first-class row of the output.**  Candidate 0 and the parent row are
the same computation up to the state restore, so their disagreement bounds what any spread in
these tables can mean.  Read Table 3 against it; a spread at the floor is the harness, not the
generator.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import _paths  # noqa: F401
import headroom_schema as hs
import stats

#: The K values the best-of-K curve is reported at.
K_GRID: tuple[int, ...] = (1, 2, 4, 8, 16)



def read_ceiling(path: str, cell: str) -> dict[str, float]:
    """Per-family success from an EPISODE-harness run, to print beside the best-of-K curve.

    The ceiling is what the tracker achieves with a PERFECT reference, so it is the number
    best-of-K saturates towards -- and it is not 1.0 on any family.  Read out of
    ``report/summary_by_family.csv`` rather than recomputed, so this cannot disagree with the
    table it is quoted from.

    Args:
        path: An eval run directory (the one ``report.py`` wrote into).
        cell: Which cell to read, e.g. ``clip__none__-``.  Match the DR preset of the
            headroom arm: the clip arm moves a point or two between ``none`` and ``train``.

    Returns:
        family -> success rate.  Empty (with a printed reason) if the file or cell is absent,
        because a missing ceiling must cost a column and not the whole report.
    """
    import csv  # noqa: PLC0415

    f = pathlib.Path(path) / "report" / "summary_by_family.csv"
    if not f.is_file():
        print(f"[headroom] no ceiling: {f} does not exist (run report.py on that eval run)")
        return {}
    out: dict[str, float] = {}
    with f.open() as fh:
        for row in csv.DictReader(fh):
            if row["cell"] == cell and row["scope"] == "family" and row["metric"] == "success":
                out[row["family"]] = float(row["value"])
    if not out:
        print(f"[headroom] no ceiling: {f} has no cell {cell!r} / scope=family / "
              "metric=success rows")
    return out


def load_cells(out_dir: str, wanted: list[str] | None) -> list[dict]:
    """Read every completed headroom cell under a run directory.

    Args:
        out_dir: The run directory (the one ``evaluate.py --stage plan`` wrote).
        wanted: Cell keys to keep, or ``None`` for all.

    Returns:
        One dict per cell with its records, episode table and metadata.

    Raises:
        SystemExit: If nothing is there, naming where it looked.
    """
    root = pathlib.Path(out_dir) / "headroom"
    cells = []
    for d in sorted(root.glob("*")):
        if wanted and d.name not in wanted:
            continue
        states = d / "states.h5"
        if not states.is_file():
            continue
        rec = hs.read_records(str(states))
        info = json.loads((d / "headroom.json").read_text()) if (d / "headroom.json").is_file() \
            else {}
        cells.append({"key": d.name, "dir": d, "data": rec, "meta": rec["meta"],
                      "info": info, "episodes": _read_episodes(d / "episodes.h5")})
    if not cells:
        raise SystemExit(
            f"no completed headroom cells under {root}. Run `headroom.py --stage run` first; "
            f"a killed run leaves a states.h5.inprogress, which is skipped on purpose."
        )
    return cells


def _read_episodes(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read the per-episode side table, or an empty dict if it is absent."""
    if not path.is_file():
        return {}
    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f if isinstance(f[k], h5py.Dataset)}


def best_of_k(survived: np.ndarray, k_grid=K_GRID) -> dict[int, np.ndarray]:
    """Exact P(at least one of k survives) per state, for each k in the grid.

    Args:
        survived: ``[n_states, K]`` 0/1 array of candidate survival.
        k_grid: The subset sizes to evaluate.

    Returns:
        ``k -> [n_states]`` probabilities.  ``k`` above ``K`` is dropped.
    """
    from math import comb  # noqa: PLC0415

    n, big_k = survived.shape
    m = survived.sum(axis=1).astype(np.int64)
    out: dict[int, np.ndarray] = {}
    for k in k_grid:
        if k > big_k:
            continue
        # 1 - C(K-m, k)/C(K, k): the fraction of k-subsets that avoid every survivor.
        denom = comb(big_k, k)
        fail = np.array([comb(big_k - int(mi), k) if big_k - int(mi) >= k else 0
                         for mi in m], dtype=np.float64)
        out[k] = 1.0 - fail / float(denom)
    return out



def survived_at(d: dict, k: int, horizon_steps: int) -> np.ndarray:
    """Candidate survival re-scored at a SHORTER horizon than the rolled window.

    Free, and on identical states: ``steps_before_term`` records when each candidate fell,
    so a candidate survived to ``horizon_steps`` exactly when it never fell at all or fell
    after that step.  A truncation (``clip_end``) is survival, and it is already covered by
    the first clause -- ``survived`` is 1 there.

    This is what makes the window-sensitivity check a property of ONE run rather than a
    comparison of two, which matters because two runs branch from different states and the
    difference between their curves would confound horizon with state distribution.

    Args:
        d: The record dict.
        k: Candidates per state.
        horizon_steps: The shorter horizon, in control steps.

    Returns:
        ``[n_states, K]`` 0/1 survival at that horizon.
    """
    surv = candidate_grid(d, "survived", k).astype(bool)
    fell_at = candidate_grid(d, "steps_before_term", k).astype(np.int64)
    return (surv | (fell_at > int(horizon_steps))).astype(np.int64)


def window_sensitivity(cell: dict, horizons_chunks: tuple[int, ...] | None = None) -> list[dict]:
    """The best-of-K rise at a ladder of horizons, all from the same rolled window.

    The spec's window check asks whether ``W`` is long enough for a bad chunk's consequence
    to have shown up.  It cuts the OPPOSITE way from the state restore: too short a horizon
    scores every candidate "survived" and flattens the curve because the instrument cannot
    resolve the difference, where restore noise inflates it.  So the ladder has to be read
    before the curve is.

    Returns:
        One dict per horizon: the K=1 and K=max rates and their difference.
    """
    d, k = cell["data"], int(cell["meta"]["k"])
    chunk = int(cell["meta"]["replan_chunk_steps"])
    rolled = int(cell["meta"]["window_steps"])
    dt = float(cell["meta"]["dt"])
    if horizons_chunks is None:
        horizons_chunks = tuple(range(1, rolled // chunk + 1))
    out = []
    for hc in horizons_chunks:
        h = hc * chunk
        if h > rolled:
            continue
        curves = best_of_k(survived_at(d, k, h))
        ks = sorted(curves)
        lo, hi = float(curves[ks[0]].mean()), float(curves[ks[-1]].mean())
        # DILUTION. A candidate whose window was truncated by clip_end before this horizon
        # counts as a survivor at every longer horizon -- correctly, it reached the end of
        # its clip -- but it cannot contribute to the rise there, so it flattens the tail of
        # the ladder. Reported, because otherwise a horizon that looks saturated may only be
        # short of candidates still at risk.
        short = candidate_grid(d, "window_steps", k).astype(np.int64) < h
        out.append({"chunks": hc, "seconds": h * dt, "k_lo": ks[0], "k_hi": ks[-1],
                    "lo": lo, "hi": hi, "rise": hi - lo,
                    "truncated_frac": float(short.mean())})
    return out


def candidate_grid(d: dict, column: str, k: int) -> np.ndarray:
    """Reshape one candidate column into ``[n_states, K]``.

    The rows of one state are contiguous and ordered ``candidate 0..K-1`` then the parent, so
    the reshape is exact rather than a groupby -- and it is asserted, because a silently
    ragged file would reshape into a plausible wrong grid.

    Raises:
        SystemExit: If the row count is not a whole number of states, or the candidate index
            pattern is not the expected one.
    """
    per = k + 1
    n = d[column].shape[0]
    if n % per:
        raise SystemExit(f"{n} rows is not a multiple of K+1={per}; the file is ragged.")
    idx = d["candidate_idx"].reshape(-1, per)
    if not (idx[:, k] == hs.PARENT_ROW).all() or not (idx[:, :k]
                                                      == np.arange(k)).all():
        raise SystemExit(
            "candidate_idx is not [0..K-1, PARENT] per state; the row order this reduce "
            "relies on does not hold."
        )
    return np.asarray(d[column]).reshape(-1, per)[:, :k]


def parent_column(d: dict, column: str, k: int) -> np.ndarray:
    """The parent row's value of a column, one per state."""
    return np.asarray(d[column]).reshape(-1, k + 1)[:, k]


def state_column(d: dict, column: str, k: int) -> np.ndarray:
    """A state-level column, de-duplicated to one value per state."""
    return np.asarray(d[column]).reshape(-1, k + 1)[:, 0]


def graded(d: dict, k: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compose the graded score from run-measured divisors.

    Returns:
        ``(candidate scores [n_states, K], parent scores [n_states], divisors)``.
    """
    channels = [f"err_{c}" for c in hs.ERROR_CHANNELS] + list(hs.COST_CHANNELS)
    surv = np.asarray(d["survived"]).astype(bool)
    divisors: dict[str, float] = {}
    for c in channels:
        v = np.asarray(d[c], dtype=np.float64)
        good = v[surv & np.isfinite(v)]
        med = float(np.median(good)) if good.size else np.nan
        # A channel whose median is zero or missing carries no information; dividing by it
        # would make the whole score NaN or infinite, so it is dropped and SAID to be.
        divisors[c] = med if np.isfinite(med) and med > 0 else np.nan

    steps = np.maximum(np.asarray(d["window_steps"], dtype=np.float64), 1.0)
    frac = np.asarray(d["steps_before_term"], dtype=np.float64) / steps
    score = 10.0 * np.asarray(d["survived"], dtype=np.float64) + 10.0 * frac
    for c, div in divisors.items():
        if not np.isfinite(div):
            continue
        score -= np.nan_to_num(np.asarray(d[c], dtype=np.float64), nan=0.0) / div
    per = k + 1
    grid = score.reshape(-1, per)
    return grid[:, :k], grid[:, k], divisors


def reduce_cell(cell: dict, *, boot: int, ci: float, seed: int = 0) -> dict:
    """Everything the four tables need, for one cell."""
    d = cell["data"]
    k = int(cell["meta"]["k"])
    fam_codes = {int(v): kk for kk, v in cell["meta"]["family_codes"].items()}
    surv = candidate_grid(d, "survived", k).astype(np.int64)
    clip = state_column(d, "clip_id", k)
    tile = state_column(d, "scene_tile_id", k)
    fam = state_column(d, "family_code", k)
    phase = state_column(d, "phase_code", k)
    cand_score, par_score, divisors = graded(d, k)
    par_surv = parent_column(d, "survived", k).astype(np.int64)

    curves = best_of_k(surv)
    out: dict = {"key": cell["key"], "meta": cell["meta"], "info": cell["info"],
                 "k": k, "n_states": int(surv.shape[0]), "divisors": divisors,
                 "families": {}, "overall": {}, "noise_floor": {}, "episodes": {}}

    def _rate(values, mask):
        est = stats.rate(clip[mask], tile[mask], values[mask], boot=boot, conf=ci,
                         seed=seed, binary=False)
        return {"value": est.value, "ci_lo": est.ci_lo, "ci_hi": est.ci_hi,
                "n_states": int(mask.sum()), "n_clips": est.n_clips, "n_tiles": est.n_tiles}

    good_frac = surv.mean(axis=1)
    spread = cand_score.max(axis=1) - cand_score.min(axis=1)
    iqr = (np.percentile(cand_score, 75, axis=1) - np.percentile(cand_score, 25, axis=1))
    airborne = state_column(d, "terminal_airborne", k).astype(bool)

    for code, name in sorted(fam_codes.items()):
        m = fam == code
        if not m.any():
            continue
        n_win = float(np.mean(np.maximum(
            state_column(d, "last_usable_step", k)[m], 1)) / float(cell["meta"]["window_steps"]))
        entry = {
            "n_states": int(m.sum()),
            "best_of_k": {str(kk): _rate(v, m) for kk, v in curves.items()},
            # The UNRESTORED rate: the parent env is never written to under commit=random, so
            # this is the honest per-window deployment hazard, and its gap to candidate 0 --
            # which is the same chunk through the state restore -- is the restore's own
            # systematic cost. Read the best-of-K RISE against that, not the absolute.
            "parent_survival": float(par_surv[m].mean()),
            "candidate0_survival": float(surv[m, 0].mean()),
            "restore_penalty": float(par_surv[m].mean() - surv[m, 0].mean()),
            # Windows per episode, for the per-window <-> per-episode conversion below.
            "windows_per_episode": n_win,
            "good_chunk_frac": {
                "mean": float(good_frac[m].mean()),
                "p10": float(np.percentile(good_frac[m], 10)),
                "p50": float(np.percentile(good_frac[m], 50)),
                "p90": float(np.percentile(good_frac[m], 90)),
            },
            "spread": {"mean": float(spread[m].mean()),
                       "p50": float(np.median(spread[m])),
                       "p90": float(np.percentile(spread[m], 90)),
                       "iqr_p50": float(np.median(iqr[m]))},
            "terminal_airborne_frac": float(airborne[m].mean()),
            "phases": {},
        }
        for pcode, pname in enumerate(hs.PHASES):
            pm = m & (phase == pcode)
            if pm.sum() < 20:
                continue
            entry["phases"][pname] = {
                "n_states": int(pm.sum()),
                "good_chunk_mean": float(good_frac[pm].mean()),
                "good_chunk_p10": float(np.percentile(good_frac[pm], 10)),
                "good_chunk_p50": float(np.percentile(good_frac[pm], 50)),
                "good_chunk_p90": float(np.percentile(good_frac[pm], 90)),
                "spread_p50": float(np.median(spread[pm])),
                "spread_p90": float(np.percentile(spread[pm], 90)),
                "best_of_k_max": float(curves[max(curves)][pm].mean()),
                "k1": float(curves[1][pm].mean()),
            }
        out["families"][name] = entry

    allm = np.ones(fam.shape, dtype=bool)
    out["overall"] = {
        "n_states": int(allm.sum()),
        "best_of_k": {str(kk): _rate(v, allm) for kk, v in curves.items()},
        "good_chunk_mean": float(good_frac.mean()),
        "spread_p50": float(np.median(spread)),
    }

    # -- the floor: candidate 0 against the parent's own un-restored window ---------------
    c0_surv = surv[:, 0]
    # The comparison that matters for Table 1, and it is a like-for-like pair of
    # PROBABILITIES OF DISAGREEMENT on a binary outcome:
    #   floor  = P(candidate 0 and the parent disagree) -- pure harness, same computation.
    #   signal = P(two DISTINCT candidates of one state disagree), which for m survivors of
    #            K is 2 m (K - m) / (K (K - 1)), averaged over states.
    # Table 1's shape is only meaningful where the second is well above the first. The
    # graded-score floor below is a different and much softer question: on a state where
    # every candidate survives, the score difference is pure tracking-error noise and the
    # ratio SHOULD be near 1.
    m = surv.sum(axis=1).astype(np.float64)
    pair_disagree = float(np.mean(2.0 * m * (k - m) / (k * max(k - 1, 1))))
    out["noise_floor"] = {
        "survival_disagreement": float(np.mean(c0_surv != par_surv)),
        "candidate_pair_disagreement": pair_disagree,
        "score_abs_diff_p50": float(np.median(np.abs(cand_score[:, 0] - par_score))),
        "score_abs_diff_p90": float(np.percentile(np.abs(cand_score[:, 0] - par_score), 90)),
        "between_candidate_spread_p50": float(np.median(spread)),
        "floor_over_spread": float(
            np.median(np.abs(cand_score[:, 0] - par_score)) / max(np.median(spread), 1e-9)),
        "parent_survival": float(par_surv.mean()),
        "candidate0_survival": float(c0_surv.mean()),
    }

    out["window_sensitivity"] = window_sensitivity(cell)
    ep = cell["episodes"]
    if ep:
        out["episodes"] = _reduce_episodes(ep, cell["meta"], fam_codes, boot=boot, ci=ci,
                                           seed=seed)
        ok, lost = episode_table_is_complete(cell["info"], ep)
        out["episodes"]["complete"] = bool(ok)
        out["episodes"]["lost_rows"] = int(lost)
    return out


def episode_table_is_complete(info: dict, ep: dict) -> tuple[bool, int]:
    """Whether every parent reset produced an episode row.

    ``alloc.done`` counts parent resets; the table should hold one row per reset plus the
    end-of-run censored flush.  A SHORTFALL means rows were lost, and they are not lost at
    random: the defect that caused it collapsed several resets inside one window into one
    row, so what goes missing is the fastest failures and every rate reads high.

    Returns:
        ``(complete, lost)`` -- ``lost`` is 0 when complete or undeterminable.
    """
    done = int(((info or {}).get("alloc") or {}).get("done", -1))
    if done < 0 or not ep:
        return True, 0
    censored = int(ep["admin_censored"].astype(bool).sum())
    observed = int(len(ep["episode_id"])) - censored
    return observed >= done, max(done - observed, 0)


def _reduce_episodes(ep: dict, meta: dict, fam_codes: dict, *, boot: int, ci: float,
                     seed: int) -> dict:
    """Episode-level success by family, defined exactly as the episode harness defines it.

    ``success`` = no FAILURE termination fired and the episode was not administratively
    censored.  ``clip_end`` and ``time_out`` are truncations, not failures.
    """
    terms = list(meta.get("termination_terms") or [])
    trunc = {"time_out", "clip_end"}
    bits = ep["termination_bits"].astype(np.int64)
    failed = np.zeros(bits.shape, dtype=bool)
    for name in terms:
        if name in trunc:
            continue
        failed |= ((bits >> terms.index(name)) & 1).astype(bool)
    censored = ep["admin_censored"].astype(bool)
    success = (~failed & ~censored).astype(float)
    keep = ~censored
    out = {"n_episodes": int(keep.sum()), "n_censored": int(censored.sum()),
           "failure_terms": [t for t in terms if t not in trunc], "by_family": {}}
    if not keep.any():
        return out
    est = stats.rate(ep["clip_id"][keep], ep["scene_tile_id"][keep], success[keep],
                     boot=boot, conf=ci, seed=seed)
    out["overall"] = {"value": est.value, "ci_lo": est.ci_lo, "ci_hi": est.ci_hi,
                      "n_episodes": est.n_episodes, "n_clips": est.n_clips}
    for code, name in sorted(fam_codes.items()):
        m = keep & (ep["family_code"] == code)
        if m.sum() < 5:
            continue
        e = stats.rate(ep["clip_id"][m], ep["scene_tile_id"][m], success[m], boot=boot,
                       conf=ci, seed=seed)
        out["by_family"][name] = {"value": e.value, "ci_lo": e.ci_lo, "ci_hi": e.ci_hi,
                                  "n_episodes": e.n_episodes, "n_clips": e.n_clips}
    return out


def format_report(reduced: list[dict]) -> str:
    """Render the four tables plus the floor, as plain text."""
    lines: list[str] = []
    a = lines.append
    a("=" * 96)
    a("STATE-LEVEL BEST-OF-K HEADROOM")
    a("=" * 96)
    for r in reduced:
        m = r["meta"]
        a("")
        a(f"CELL {r['key']}")
        a(f"  K={r['k']}  states={r['n_states']}  window={m['window_s']:.2f} s "
          f"({m['window_steps']} steps, chunk {m['replan_chunk_steps']})  "
          f"commit={m['commit_mode']}  branch stride={m['branch_stride_chunks']} chunk(s)")
        a(f"  dr_preset={m['cell']['dr_preset']}  generator={m['cell']['fm_label']}")

        nf = r["noise_floor"]
        a("")
        a("  -- NOISE FLOOR (candidate 0 vs the parent's own window: same computation up to")
        a("     the state restore, so this bounds what any spread below can mean) ----------")
        a(f"     survival disagreement      {nf['survival_disagreement']:.4f}   "
          f"(parent {nf['parent_survival']:.3f} vs candidate0 {nf['candidate0_survival']:.3f})")
        a(f"     two-candidate disagreement {nf['candidate_pair_disagreement']:.4f}   "
          "<- the SIGNAL Table 1 measures, on the same scale")
        ratio = (nf["candidate_pair_disagreement"]
                 / max(nf["survival_disagreement"], 1e-9))
        a(f"     signal / floor (survival)  {ratio:>6.1f}x   "
          + ("<- Table 1's shape is real" if ratio >= 3.0 else
             "<- TOO CLOSE: Table 1's shape may be the state restore, not the generator"))
        a(f"     |score| diff  p50/p90      {nf['score_abs_diff_p50']:.3f} / "
          f"{nf['score_abs_diff_p90']:.3f}")
        a(f"     between-candidate spread   {nf['between_candidate_spread_p50']:.3f} (p50)")
        a(f"     floor / spread             {nf['floor_over_spread']:.3f}   "
          + ("<- spread is at the floor; treat every ranking below as uninformative"
             if nf["floor_over_spread"] > 0.5 else
             "<- spread is well above the floor"))

        ks = sorted(int(k) for k in next(iter(r["families"].values()))["best_of_k"]) \
            if r["families"] else []
        a("")
        a(f"  -- TABLE 1: best-of-K survival over ONE {m['window_s']:.2f} s WINDOW, by family --")
        a("     UNITS. These are PER-WINDOW rates, not per-episode success. An episode is")
        a("     several windows, so an episode number and a cell here are different")
        a("     quantities and must not be compared directly. The 'ceil/win' column is the")
        a("     clip arm's episode success converted to a per-window equivalent by")
        a("     ceiling**(1/windows), which assumes a CONSTANT hazard over the episode --")
        a("     an approximation, and the 'K=1 -> ep' column is the same conversion run")
        a("     backwards as a tie-out against report.py's own `fm` cell.")
        ceil = r.get("ceiling") or {}
        if ceil:
            a(f"     ceiling source = {r['ceiling_source']}")
        a("     " + f"{'family':<15}" + "".join(f"{'K=' + str(k):>16}" for k in ks)
          + f"{'rise':>7}{'parent':>8}{'restore':>8}{'win/ep':>7}{'ceil/win':>9}"
          + f"{'K=1->ep':>8}{'states':>7}")
        for name, f in r["families"].items():
            row = "".join(
                f"{f['best_of_k'][str(k)]['value']:>8.3f}"
                f"[{f['best_of_k'][str(k)]['ci_lo']:.2f},{f['best_of_k'][str(k)]['ci_hi']:.2f}]"
                for k in ks)
            k_lo, k_hi = f["best_of_k"][str(min(ks))], f["best_of_k"][str(max(ks))]
            rise = k_hi["value"] - k_lo["value"]
            nw = f["windows_per_episode"]
            cw = f"{ceil[name] ** (1.0 / max(nw, 1e-9)):>9.4f}" if name in ceil else f"{'--':>9}"
            implied = k_lo["value"] ** nw
            a(f"     {name:<15}{row}{rise:>+7.3f}{f['parent_survival']:>8.3f}"
              f"{f['restore_penalty']:>+8.3f}{nw:>7.1f}{cw}{implied:>8.3f}"
              f"{f['n_states']:>7}")
        row = "".join(
            f"{r['overall']['best_of_k'][str(k)]['value']:>8.3f}"
            f"[{r['overall']['best_of_k'][str(k)]['ci_lo']:.2f},"
            f"{r['overall']['best_of_k'][str(k)]['ci_hi']:.2f}]" for k in ks)
        a(f"     {'OVERALL':<18}{row}{r['overall']['n_states']:>9}")

        a("")
        a("  -- TABLE 1b: the SAME numbers as a per-EPISODE equivalent --------------------")
        a("     Each cell is the Table 1 rate raised to that family's windows-per-episode,")
        a("     i.e. what the rate would imply for episode success IF the per-window hazard")
        a("     were constant over the episode. It is a CONVERSION of Table 1, not a second")
        a("     measurement -- the direct per-episode number is Table 4 of the argmax arm,")
        a("     and the 'meas' column is this arm's own Table 4 so the conversion's error is")
        a("     visible rather than assumed. The clip arm's episode success is 'ceil'.")
        eps = r.get("episodes") or {}
        ep = eps.get("by_family") or {}
        if ep and not eps.get("complete", True):
            a(f"     WARNING: the 'meas' column is BIASED UPWARD -- {eps.get('lost_rows')} "
              "episodes are missing from this cell's episode table (see Table 4). Compare "
              "the conversion against report.py's `fm` arm, not against 'meas'.")
        a("     " + f"{'family':<18}" + "".join(f"{'K=' + str(k):>9}" for k in ks)
          + f"{'oracle+':>9}{'ceil':>8}{'meas':>8}{'gap left':>10}")
        for name, f in r["families"].items():
            nw = f["windows_per_episode"]
            vals = [f["best_of_k"][str(k)]["value"] ** nw for k in ks]
            row = "".join(f"{v:>9.3f}" for v in vals)
            gain = vals[-1] - vals[0]
            c = ceil.get(name)
            meas = (ep.get(name) or {}).get("value")
            cs = f"{c:>8.3f}" if c is not None else f"{'--':>8}"
            ms = f"{meas:>8.3f}" if meas is not None else f"{'--':>8}"
            # What an oracle over K leaves on the table against the clip ceiling.
            left = (f"{(c - vals[-1]) / max(c - vals[0], 1e-9):>10.0%}"
                    if c is not None and c > vals[0] else f"{'--':>10}")
            a(f"     {name:<18}{row}{gain:>+9.3f}{cs}{ms}{left}")
        a("     'oracle+' = K=max minus K=1, i.e. what a perfect selector over K samples buys.")
        a("     'gap left' = the fraction of the K=1-to-ceiling gap STILL open at K=max.")

        a("")
        a("  -- TABLE 2: good-chunk fraction P(good chunk | state), by family and phase ---")
        a(f"     {'family / phase':<32}{'mean':>8}{'p10':>8}{'p50':>8}{'p90':>8}{'states':>9}")
        for name, f in r["families"].items():
            g = f["good_chunk_frac"]
            a(f"     {name:<32}{g['mean']:>8.3f}{g['p10']:>8.3f}{g['p50']:>8.3f}"
              f"{g['p90']:>8.3f}{f['n_states']:>9}")
            for pname, p in f["phases"].items():
                a(f"       {pname:<30}{p['good_chunk_mean']:>8.3f}{p['good_chunk_p10']:>8.3f}"
                  f"{p['good_chunk_p50']:>8.3f}{p['good_chunk_p90']:>8.3f}{p['n_states']:>9}")

        a("")
        a("  -- TABLE 3: graded-score spread across the K candidates ----------------------")
        a(f"     {'family / phase':<32}{'mean':>8}{'p50':>8}{'p90':>8}{'IQR p50':>10}")
        for name, f in r["families"].items():
            s = f["spread"]
            a(f"     {name:<32}{s['mean']:>8.2f}{s['p50']:>8.2f}{s['p90']:>8.2f}"
              f"{s['iqr_p50']:>10.2f}")
            for pname, p in f["phases"].items():
                a(f"       {pname:<30}{'':>8}{p['spread_p50']:>8.2f}{p['spread_p90']:>8.2f}")

        if r["episodes"] and "overall" in r["episodes"]:
            e = r["episodes"]
            a("")
            a(f"  -- TABLE 4: episode success under commit={m['commit_mode']} --------------")
            a(f"     failure terms {e['failure_terms']}; {e['n_censored']} censored episode(s) "
              "excluded")
            if not e.get("complete", True):
                tot = e["n_episodes"] + e["lost_rows"]
                a(f"     *** BIASED UPWARD: {e['lost_rows']} of {tot} episodes "
                  f"({e['lost_rows'] / max(tot, 1):.1%}) are MISSING from this table. The "
                  "cell predates the per-reset fix, which collapsed several resets inside "
                  "one window into one row -- so what is missing is the FASTEST FAILURES, "
                  "not a random sample.")
                a(f"     *** Ceiling on the true overall rate: "
                  f"{e['overall']['value'] * e['n_episodes'] / max(tot, 1):.3f} "
                  "(every lost episode counted as a failure). Re-run before quoting it; "
                  "use report.py's own `fm` arm as the tie-out instead.")
            if m["commit_mode"] == "argmax":
                info = r.get("info") or {}
                gr = int(info.get("rounds", 0)) * int(m.get("num_groups", 0))
                unc = int(info.get("uncommitted_group_rounds", -1))
                a("     CAVEAT: a round in which the parent RESET cannot be committed -- its "
                  "one-chunk-in snapshot belongs to the episode that has since ended.")
                if gr and unc >= 0:
                    a(f"     Un-oracled: {unc} of {gr} group-rounds ({unc / gr:.1%}). An "
                      "un-committed round advances the parent a whole window (60 steps) "
                      "against a committed round's 12, so multiply by ~5 to read it as a "
                      "fraction of episode PROGRESS. This number is a lower bound on the "
                      "oracle.")
                elif unc < 0:
                    a("     Un-oracled fraction NOT RECORDED -- this cell predates "
                      "`uncommitted_group_rounds`, and was run when the whole round after a "
                      "reset was skipped rather than merely unrecorded (~57% of episode "
                      "progress un-oracled). Re-run before quoting it.")
            else:
                a("     This is the DEPLOYMENT arm -- random commit is statistically ordinary "
                  "deployment -- so it is the tie-out against report.py's `fm` cell, not a "
                  "result in its own right.")
            a(f"     {'family':<20}{'success':>10}{'ci':>18}{'episodes':>10}{'clips':>8}")
            for name, v in e["by_family"].items():
                a(f"     {name:<20}{v['value']:>10.3f}"
                  f"{'[' + f'{v['ci_lo']:.3f}, {v['ci_hi']:.3f}' + ']':>18}"
                  f"{v['n_episodes']:>10}{v['n_clips']:>8}")
            o = e["overall"]
            a(f"     {'OVERALL':<20}{o['value']:>10.3f}"
              f"{'[' + f'{o['ci_lo']:.3f}, {o['ci_hi']:.3f}' + ']':>18}"
              f"{o['n_episodes']:>10}{o['n_clips']:>8}")

        a("")
        a("  -- graded-score divisors (median over surviving candidate rows) --------------")
        a("     " + "  ".join(f"{k.replace('err_', '')}={v:.4g}"
                              for k, v in r["divisors"].items() if np.isfinite(v)))
        dropped = [k for k, v in r["divisors"].items() if not np.isfinite(v)]
        if dropped:
            a(f"     dropped (median 0 or absent, so uninformative): {dropped}")

    a("")
    for r in reduced:
        rows = r.get("window_sensitivity") or []
        if not rows:
            continue
        a(f"-- WINDOW SENSITIVITY ({r['key']}) " + "-" * 30)
        a("     Re-scored from the SAME rolled window at shorter horizons, so horizon is the")
        a("     only thing that changes. Too short a horizon flattens the curve because a bad")
        a("     chunk's consequence has not landed yet -- read this BEFORE Table 1. If the")
        a("     rise is still growing at the longest horizon, the window is too short.")
        a(f"     {'horizon':>9}{'K=1':>9}{'K=' + str(rows[0]['k_hi']):>9}{'rise':>9}"
          f"{'d(rise)':>9}{'trunc':>8}")
        for i, w in enumerate(rows):
            dr = w["rise"] - rows[i - 1]["rise"] if i else w["rise"]
            a(f"     {w['seconds']:>7.2f} s{w['lo']:>9.4f}{w['hi']:>9.4f}{w['rise']:>+9.4f}"
              f"{dr:>+9.4f}{w['truncated_frac']:>8.2f}")
        # Verdict from the mean of the LAST THREE increments, not the last one: a single
        # increment is noisy, and on this data the final step alone read "saturated" while
        # the three before it were all ~+0.003.
        incs = [rows[i]["rise"] - rows[i - 1]["rise"] for i in range(1, len(rows))]
        tail = incs[-3:] if len(incs) >= 3 else incs
        mean_tail = float(np.mean(tail)) if tail else 0.0
        a(f"     mean of the last {len(tail)} increment(s): {mean_tail:+.4f} per chunk"
          + ("  <- STILL GROWING: the window is too short to resolve the headroom; "
             "re-run with a larger --window_chunks"
             if mean_tail > 0.1 * max(rows[-1]["rise"], 1e-9) else
             "  <- saturated, the horizon is long enough"))
        if rows[-1]["truncated_frac"] > 0.15:
            a(f"     NOTE: {rows[-1]['truncated_frac']:.0%} of candidates were truncated by "
              "clip_end before the longest horizon, so the tail of this ladder is diluted "
              "and the saturation verdict there is weak either way.")
        a("")
    a(_control_contrast(reduced))
    a("-- HOW TO READ THIS " + "-" * 76)
    a("  READ THE RISE, NOT THE ABSOLUTE. Every candidate carries the state restore and the")
    a("    parent does not, so the level of the curve includes the 'restore' penalty column")
    a("    while the K=1 -> K=16 delta is a difference of two equally-restored numbers.")
    a("  CHECK signal/floor FIRST. Below ~3x the rise is not distinguishable from the state")
    a("    restore, and the fix is the same-noise control (all K given ONE draw): its")
    a("    two-candidate disagreement is pure harness, with no chunk effect at all.")
    a("  best-of-K rises steeply to the clip ceiling -> variance-limited; the good chunks")
    a("    exist and a distribution-sharpening method should recover most of the gap.")
    a("  best-of-K plateaus below it -> the plateau height is the realistic target.")
    a("  best-of-K flat but Table 3 spread well above the floor -> weak but real ranking")
    a("    signal; sharpening is viable and slower.")
    a("  flat in BOTH -> bias-limited. The conditional has no good tail and supervised fixes")
    a("    (perception, data balance, output parameterization) come first.")
    a("  Stratify on phase: mid_feature spread with flat approach spread is a state-")
    a("    distribution problem; flat everywhere is the model unable to do the skill at all.")
    a("  The clip ceiling is NOT 1.0 -- take it from `report.py` on the same plan's clip arm.")
    a("  jump_off_front is confounded: the 0.50 m base_fallen threshold sits ~25 mm inside")
    a("    the reference clip's own base-height envelope, so read min_height_margin_m there")
    a("    before believing its survival numbers.")
    return "\n".join(lines)


def make_plots(reduced: list[dict], cells: list[dict], out: pathlib.Path) -> None:
    """Best-of-K curves, good-chunk histograms, and spread by phase."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    import plot_style  # noqa: PLC0415

    plot_style.apply_paper_style()
    out.mkdir(parents=True, exist_ok=True)

    for r, cell in zip(reduced, cells):
        k = r["k"]
        d = cell["data"]
        fam_codes = {int(v): kk for kk, v in r["meta"]["family_codes"].items()}

        fig, ax = plt.subplots(figsize=plot_style.figsize(1, 1))
        for name, f in r["families"].items():
            ks = sorted(int(x) for x in f["best_of_k"])
            ys = [f["best_of_k"][str(x)]["value"] for x in ks]
            lo = [f["best_of_k"][str(x)]["ci_lo"] for x in ks]
            hi = [f["best_of_k"][str(x)]["ci_hi"] for x in ks]
            ax.plot(ks, ys, marker="o", label=name)
            ax.fill_between(ks, lo, hi, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(K_GRID))
        ax.set_xticklabels([str(x) for x in K_GRID])
        ax.set_xlabel("K candidates")
        ax.set_ylabel(f"P(at least one survives {r['meta']['window_s']:.2f} s)")
        ax.set_title(f"best-of-K headroom ({r['key']})")
        ax.legend(fontsize="small")
        ax.grid(alpha=0.3)
        plot_style.save_figure(fig, str(out / f"best_of_k__{r['key']}"))
        plt.close(fig)

        surv = candidate_grid(d, "survived", k).astype(float)
        fam = state_column(d, "family_code", k)
        good = surv.mean(axis=1)
        fig, ax = plt.subplots(figsize=plot_style.figsize(1, 1))
        bins = np.linspace(0, 1, k + 2)
        for code, name in sorted(fam_codes.items()):
            m = fam == code
            if m.sum() < 20:
                continue
            ax.hist(good[m], bins=bins, histtype="step", density=True, label=name, lw=2)
        ax.set_xlabel("fraction of K candidates that survived")
        ax.set_ylabel("density")
        ax.set_title(f"good-chunk fraction ({r['key']})")
        ax.legend(fontsize="small")
        plot_style.save_figure(fig, str(out / f"good_chunk_frac__{r['key']}"))
        plt.close(fig)

        cand_score, par_score, _ = graded(d, k)
        spread = cand_score.max(axis=1) - cand_score.min(axis=1)
        floor = np.abs(cand_score[:, 0] - par_score)
        phase = state_column(d, "phase_code", k)
        fig, ax = plt.subplots(figsize=plot_style.figsize(1, 1))
        labels, data, floors = [], [], []
        for pcode, pname in enumerate(hs.PHASES):
            m = phase == pcode
            if m.sum() < 20:
                continue
            labels.append(f"{pname}\nn={int(m.sum())}")
            data.append(spread[m])
            floors.append(np.median(floor[m]))
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.plot(range(1, len(floors) + 1), floors, "r_", markersize=18,
                    label="noise floor (median)")
            ax.legend(fontsize="small")
        ax.set_ylabel("graded-score spread across K")
        ax.set_title(f"spread by phase ({r['key']})")
        plot_style.save_figure(fig, str(out / f"spread_by_phase__{r['key']}"))
        plt.close(fig)


def stage_report(args) -> int:
    """Reduce every cell, write the tables, and (unless suppressed) the figures."""
    wanted = [c.strip() for c in args.cells.split(",") if c.strip()] or None
    cells = load_cells(args.out, wanted)
    reduced = [reduce_cell(c, boot=int(args.boot), ci=float(args.ci)) for c in cells]
    if getattr(args, "ceiling", ""):
        ceil = read_ceiling(args.ceiling, args.ceiling_cell)
        for r in reduced:
            r["ceiling"] = ceil
            r["ceiling_source"] = f"{args.ceiling}::{args.ceiling_cell}"
    out = pathlib.Path(args.out) / "headroom" / "report"
    out.mkdir(parents=True, exist_ok=True)
    text = format_report(reduced)
    (out / "report.txt").write_text(text)
    (out / "report.json").write_text(json.dumps(reduced, indent=2, default=float))
    _write_csv(reduced, out)
    print(text)
    if not args.no_plots:
        try:
            make_plots(reduced, cells, out / "figures")
        except Exception as exc:  # noqa: BLE001 - a plot failure must not lose the tables
            print(f"[headroom] figures skipped: {type(exc).__name__}: {exc}")
    print(f"\n[headroom] wrote {out}/report.txt, report.json, tables.csv")
    return 0



def _control_contrast(reduced: list[dict]) -> str:
    """The decisive comparison: the real arm's best-of-K rise against the control's.

    The same-noise control gives every candidate ONE draw, so its K branches differ only in
    which env slot they landed in.  Whatever rise it shows is the state restore, with no chunk
    effect in it at all -- so it is the number the real arm's rise has to beat, and it is the
    only clean form of this check in a simulator whose solver state cannot be restored.

    Candidate-0-vs-parent (the per-cell floor above) is a DIFFERENT and softer question: it
    compares restored against UNRESTORED, so it carries a systematic penalty the real arm's
    K=1 and K=16 both carry equally and which therefore partly cancels in the rise.

    Returns:
        The block, or a line saying the control has not been run.
    """
    ctrl = {r["key"]: r for r in reduced if r["key"].endswith("__samenoise")}
    real = [r for r in reduced if not r["key"].endswith("__samenoise")
            and r["meta"]["commit_mode"] == "random"]
    out = ["-- CONTROL CONTRAST: is the best-of-K rise the generator or the restore? " + "-" * 20]
    if not ctrl:
        out.append("     NOT RUN. Re-run with --same_noise; without it the rise below cannot")
        out.append("     be separated from the state restore. This is the blocking check.")
        return "\n".join(out) + "\n"
    if not real:
        out.append("     the control is present but no random-commit arm is, so there is")
        out.append("     nothing to contrast it against.")
        return "\n".join(out) + "\n"
    c = next(iter(ctrl.values()))
    r = real[0]
    out.append(f"     control cell = {c['key']}  ({c['n_states']} states)")
    out.append(f"     {'family':<18}{'real rise':>11}{'control':>10}{'net':>9}"
               f"{'verdict':>26}")
    for name in r["families"]:
        if name not in c["families"]:
            continue
        def _rise(entry):
            ks = sorted(int(x) for x in entry["best_of_k"])
            return (entry["best_of_k"][str(max(ks))]["value"]
                    - entry["best_of_k"][str(min(ks))]["value"])
        rr, cc = _rise(r["families"][name]), _rise(c["families"][name])
        net = rr - cc
        verdict = ("real" if net > 2.0 * max(abs(cc), 1e-4) else
                   "NOT above the restore" if net <= abs(cc) else "marginal")
        out.append(f"     {name:<18}{rr:>+11.4f}{cc:>+10.4f}{net:>+9.4f}{verdict:>26}")
    return "\n".join(out) + "\n"


def _write_csv(reduced: list[dict], out: pathlib.Path) -> None:
    """One long-form CSV of every table cell, for pasting elsewhere."""
    import csv  # noqa: PLC0415

    with (out / "tables.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "table", "family", "phase", "metric", "value", "ci_lo", "ci_hi",
                    "n_states"])
        for r in reduced:
            for name, f in r["families"].items():
                for kk, v in f["best_of_k"].items():
                    w.writerow([r["key"], "best_of_k", name, "", f"K={kk}", v["value"],
                                v["ci_lo"], v["ci_hi"], v["n_states"]])
                g = f["good_chunk_frac"]
                for stat, v in g.items():
                    w.writerow([r["key"], "good_chunk", name, "", stat, v, "", "",
                                f["n_states"]])
                for pname, p in f["phases"].items():
                    w.writerow([r["key"], "good_chunk", name, pname, "mean",
                                p["good_chunk_mean"], "", "", p["n_states"]])
                    w.writerow([r["key"], "spread", name, pname, "p50", p["spread_p50"],
                                "", "", p["n_states"]])
            for name, v in (r["episodes"].get("by_family") or {}).items():
                w.writerow([r["key"], "episode_success", name, "", "success", v["value"],
                            v["ci_lo"], v["ci_hi"], v["n_episodes"]])
