# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tables and figures for the critic-advantage evaluation.

Emits the numbers and the plots; the ARGUMENT is written by hand in ``docs/``.  That split is
deliberate -- an auto-generated conclusion is a conclusion nobody checked, and every number
here needs a caveat that only a reader who knows how it was measured can supply.

Reads ``metrics.json`` and ``advantages.npz`` from the eval directory, plus
``branch_metrics.json`` when the counterfactual arm has run, and writes ``tables.md`` and a
handful of figures beside them.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(os.path.dirname(_HERE), "rsl_rl"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def _fmt(x, nd: int = 3) -> str:
    """Format a number for a markdown cell, tolerating None and NaN.

    Args:
        x: The value.
        nd: Decimal places.

    Returns:
        The rendered cell.
    """
    if x is None:
        return "--"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "--" if not np.isfinite(f) else f"{f:.{nd}f}"


def _table(rows: list[list], header: list[str]) -> str:
    """Render a markdown table.

    Args:
        rows: Row cells, already stringified.
        header: Column names.

    Returns:
        The table, newline-terminated.
    """
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _memorization_table(M: dict) -> str:
    """Step 1: the in-fold / out-of-fold gap, per head.

    Args:
        M: The loaded ``metrics.json``.

    Returns:
        Markdown.
    """
    rows = []
    for key, blk in M["heads"].items():
        m = blk["memorization"]
        rows.append([key, m["n"], _fmt(m["auc_out"]), _fmt(m["auc_in"]),
                     _fmt(m["auc_gap"]), _fmt(m["spearman_out"]),
                     _fmt(m["spearman_in"]), _fmt(m["spearman_gap"])])
    return _table(rows, ["head", "n", "AUC out", "AUC in", "AUC gap",
                         "rho(V,G) out", "rho in", "rho gap"])


def _discrimination_table(M: dict) -> str:
    """Step 2: AUROC against progress, seen and unseen tiles.

    Args:
        M: The loaded ``metrics.json``.

    Returns:
        Markdown.
    """
    rows = []
    for key, blk in M["heads"].items():
        d = blk["discrimination"]
        for scope in ("auc_by_progress_seen_tiles", "auc_by_progress_unseen_tiles"):
            b = d[scope]
            if not b.get("value"):
                continue
            rows.append([key, scope.replace("auc_by_progress_", "")]
                        + [_fmt(v) for v in b["value"]])
    if not rows:
        return "_(no buckets)_\n"
    n = max(len(r) - 2 for r in rows)
    return _table(rows, ["head", "scope"] + [f"b{i}" for i in range(n)])


def _estimator_table(M: dict) -> str:
    """Step 3 + the structural questions, one row per estimator.

    Args:
        M: The loaded ``metrics.json``.

    Returns:
        Markdown.
    """
    rows = []
    for key, blk in M["heads"].items():
        for name, e in blk["estimators"].items():
            if name.startswith("A_GAE") and name not in ("A_GAE0", "A_GAE0.5",
                                                         "A_GAE0.8", "A_GAE0.95", "A_GAE1"):
                continue
            rows.append([key, name, _fmt(e["sd"]),
                         _fmt(e["var_frac_within_episode"]),
                         _fmt(e["r2_from_episode_outcome"]),
                         _fmt(e["r2_from_outcome_family_progress"]),
                         _fmt(e["reliability_across_folds_unseen"]),
                         _fmt(e.get("reliability_within_episode_unseen")),
                         _fmt(e["within_signal_to_noise"], 2),
                         _fmt(e["spearman_with_A_base"])])
    return _table(rows, ["head", "estimator", "sd", "var frac within ep",
                         "R2 from outcome", "R2 +fam +prog", "reliability",
                         "reliability WITHIN ep", "within SNR", "rho with A_base"])


def _weight_table(M: dict, estimators=("A_base", "A_MC", "A_TD", "A_GAE0.95")) -> str:
    """Step 5: effective sample size and where the weight mass lands.

    Args:
        M: The loaded ``metrics.json``.
        estimators: Which estimators to show.

    Returns:
        Markdown.
    """
    rows = []
    for key, blk in M["heads"].items():
        for name in estimators:
            e = blk["estimators"].get(name)
            if e is None:
                continue
            for w in e["weights"]:
                if w["w_max"] is None:
                    continue        # the clipped arm is the one a trainer would use
                rows.append([key, name, w["beta"], _fmt(w["ess_frac"]),
                             _fmt(w.get("ess_frac_episodes")),
                             _fmt(w["mass_on_success"]), _fmt(w["frac_rows_success"]),
                             _fmt(w["mass_near_failure"]), _fmt(w["frac_rows_near_failure"]),
                             _fmt(w["max_weight_share"], 5)])
    return _table(rows, ["head", "estimator", "beta", "ESS frac (rows)",
                         "ESS frac (EPISODES)", "mass on success",
                         "row frac success", "mass near failure", "row frac near failure",
                         "max single weight"])


def _credit_table(M: dict) -> str:
    """Step 4: where the most negative advantage lands in a failed episode.

    Args:
        M: The loaded ``metrics.json``.

    Returns:
        Markdown.
    """
    rows = []
    for key, blk in M["heads"].items():
        for name, e in blk["estimators"].items():
            c = e.get("credit") or {}
            if not c.get("n"):
                continue
            rows.append([key, name, c["n"], _fmt(c["median_chunks_before_end"], 1),
                         _fmt(c["p25"], 1), _fmt(c["p75"], 1),
                         _fmt(c["frac_on_last_chunk"]), _fmt(c["frac_within_2_chunks"])])
    return _table(rows, ["head", "estimator", "n failed ep", "median offset (chunks)",
                         "p25", "p75", "frac on last chunk", "frac within 2"])


def _novelty_table(M: dict) -> str:
    """How different a filtered-BC set from each estimator is from the outcome-only one.

    Args:
        M: The loaded ``metrics.json``.

    Returns:
        Markdown.
    """
    rows = []
    for key, blk in M["heads"].items():
        for name, e in blk["estimators"].items():
            n = e.get("label_novelty")
            if not n:
                continue
            rows.append([key, name, _fmt(n["positive_frac"]),
                         _fmt(n["outcome_positive_frac"]),
                         _fmt(n["positives_from_failed_episodes"]),
                         _fmt(n["negatives_from_survived_episodes"]),
                         _fmt(n["novel_frac"])])
    return _table(rows, ["head", "estimator", "positive frac", "outcome positive frac",
                         "positives from FAILED ep", "negatives from SURVIVED ep",
                         "rows classified differently"])


def _branch_tables(B: dict) -> str:
    """The counterfactual arm, read against its own ceiling.

    Args:
        B: The loaded ``branch_metrics.json``.

    Returns:
        Markdown.
    """
    out = ["### Realized between-sibling spread (read this first)\n"]
    rs = B["realized_spread"]
    out.append(_table([[B["n_groups"], B["n_branches"],
                        rs["survival_groups_with_disagreement"],
                        _fmt(rs["survival_frac_informative"]),
                        _fmt(rs["quality_spread_median"]),
                        _fmt(rs["quality_spread_p90"]),
                        _fmt(rs["failure_rate_corrected"]),
                        _fmt(rs["failure_rate_raw"])]],
                      ["groups", "branches", "outcome-mixed groups", "frac informative",
                       "median dG", "p90 dG", "fail rate", "fail rate (uncorrected)"]))

    if "label_ceiling" in B:
        out.append("\n### Label-noise ceiling (the best any predictor could do)\n")
        rows = []
        for head, c in B["label_ceiling"].items():
            if not isinstance(c, dict) or not c.get("n_groups"):
                continue
            rows.append([head, c["n_groups"], _fmt(c.get("rho")),
                         _fmt(c.get("rho_sem")), _fmt(c.get("rho_oracle")),
                         _fmt(c.get("best_pick"))])
        out.append(_table(rows, ["head", "groups", "single-draw reliability r", "sem",
                                 "ORACLE ceiling sqrt(r)", "best-pick ceiling"]))

    out.append("\n### Critic ranking of siblings, against a within-group permutation null\n")
    rows = []
    for nm, blk in B["heads"].items():
        for head in ("survival", "quality"):
            r = blk.get(head)
            if not r or not r.get("n_informative"):
                continue
            rows.append([nm, head, r["n_informative"], _fmt(r["rho"]), _fmt(r.get("rho_sem")),
                         _fmt(r["rho_null_mean"]), _fmt(r["rho_null_sd"]),
                         _fmt(r["p_value"]), _fmt(r["best_pick"]),
                         _fmt(r["best_pick_null_mean"])])
    out.append(_table(rows, ["critic", "head", "informative groups", "rho", "sem",
                             "null mean", "null sd", "p", "best-pick", "null"]))

    h2h = []
    for nm, blk in B["heads"].items():
        for head in ("survival", "quality"):
            d = blk.get(f"{head}_head_to_head")
            if not d:
                continue
            for est in ("critic", "mc"):
                r = d.get(est) or {}
                if not r.get("n_informative"):
                    continue
                h2h.append([nm, head,
                            "A_TD (critic at s')" if est == "critic" else "A_MC (realized)",
                            r["n_informative"], _fmt(r["rho"]), _fmt(r.get("rho_sem")),
                            _fmt(r["rho_null_mean"]), _fmt(r["p_value"]),
                            _fmt(r["best_pick"]), _fmt(r["best_pick_null_mean"])])
    if h2h:
        out.append("\n### HEAD TO HEAD: both estimators against the SAME independent draw\n")
        out.append(_table(h2h, ["critic", "head", "estimator", "informative groups", "rho",
                                "sem", "null", "p", "best-pick", "best-pick null"]))

    strat = []
    for nm, blk in B["heads"].items():
        for r in blk.get("quality_by_spread", []):
            strat.append([nm, f"{r['spread_lo']:.2f}-"
                          + ("inf" if r["spread_hi"] is None else f"{r['spread_hi']:.2f}"),
                          r["n_informative"], _fmt(r["rho"]), _fmt(r.get("rho_sem")),
                          _fmt(r["rho_null_mean"]), _fmt(r["p_value"]),
                          _fmt(r["best_pick"]), _fmt(r["best_pick_null_mean"])])
    if strat:
        out.append("\n### Stratified by how much the siblings' outcomes ACTUALLY differed\n")
        out.append(_table(strat, ["critic", "realized dG band", "groups", "rho", "sem",
                                  "null", "p", "best-pick", "null"]))

    out.append("\n### What WEIGHTING buys, against selection's oracle\n")
    rows = []
    for nm, blk in B["heads"].items():
        for head in ("survival_awr", "quality_awr"):
            for w in blk.get(head, []):
                rows.append([nm, head.replace("_awr", ""), w["beta"], w["n_groups"],
                             _fmt(w["uniform"]), _fmt(w["weighted"]), _fmt(w["oracle"]),
                             _fmt(w["gain_over_uniform"], 4), _fmt(w["captured"])])
    out.append(_table(rows, ["critic", "head", "beta", "groups", "uniform", "weighted",
                             "oracle", "gain", "frac of oracle captured"]))

    if "critic_resolution" in B:
        out.append("\n### Can the critic RESOLVE siblings at all?\n")
        rows = [[k, v.get("n_groups", "--"), _fmt(v["within_group_median_pairwise"], 5),
                 _fmt(v["fold_disagreement_median_abs"], 5), _fmt(v["snr"], 3),
                 " vs ".join(v["folds"])]
                for k, v in B["critic_resolution"].items()]
        out.append(_table(rows, ["quantity", "groups", "within-group median |pairwise|",
                                 "fold disagreement median |pairwise|", "SNR", "folds"]))
    return "\n".join(out)


def _plots(M: dict, out_dir: str) -> list[str]:
    """Every figure, saved beside the tables.

    Args:
        M: The loaded ``metrics.json``.
        out_dir: Where to write.

    Returns:
        The basenames written.

    Raises:
        ImportError: If matplotlib is unavailable; the tables still land without it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import plot_style as ps  # noqa: PLC0415

    ps.apply_paper_style()
    written = []

    # ---- AUROC against progress, both scopes, both variants -----------------------------
    fig, ax = plt.subplots(figsize=ps.DOUBLE_COL)
    for i, (key, blk) in enumerate(M["heads"].items()):
        if not key.endswith("survival"):
            continue
        d = blk["discrimination"]
        for scope, ls in (("auc_by_progress_seen_tiles", "-"),
                          ("auc_by_progress_unseen_tiles", "--")):
            b = d[scope]
            if not b.get("value"):
                continue
            ax.plot([(x + 0.5) / 10 for x in b["bucket"]], b["value"], ls,
                    color=ps.COLOR_CYCLE[i % len(ps.COLOR_CYCLE)], marker="o", ms=3,
                    label=f"{key.split('/')[0]} {'seen' if 'seen_tiles' in scope and 'unseen' not in scope else 'unseen'} tiles")
    ax.axhline(0.5, color=ps.COLORS["grey"], lw=1, zorder=0)
    ax.set_xlabel("normalized episode progress")
    ax.set_ylabel("AUROC (failure)")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("Discrimination rises through the episode")
    ps.save_figure(fig, os.path.join(out_dir, "auc_vs_progress"))
    written.append("auc_vs_progress")
    plt.close(fig)

    # ---- ESS against beta ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=ps.DOUBLE_COL)
    for i, (key, blk) in enumerate(M["heads"].items()):
        for j, name in enumerate(("A_base", "A_MC", "A_TD", "A_GAE0.95")):
            e = blk["estimators"].get(name)
            if e is None:
                continue
            ws = [w for w in e["weights"] if w["w_max"] is not None]
            ax.plot([w["beta"] for w in ws], [w["ess_frac"] for w in ws],
                    marker="os^D"[j % 4], ms=3,
                    color=ps.COLOR_CYCLE[j % len(ps.COLOR_CYCLE)],
                    ls="-" if "priv" in key else "--",
                    label=f"{key} {name}")
    ax.set_xscale("log")
    ax.set_xlabel(r"AWR temperature $\beta$ (on the standardized advantage)")
    ax.set_ylabel("effective sample size / rows")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("How much of the dataset survives the weighting")
    ps.save_figure(fig, os.path.join(out_dir, "ess_vs_beta"))
    written.append("ess_vs_beta")
    plt.close(fig)

    # ---- where the advantage's variance lives -------------------------------------------
    fig, ax = plt.subplots(figsize=ps.DOUBLE_COL)
    keys, vals, cols = [], [], []
    for i, (key, blk) in enumerate(M["heads"].items()):
        for name, e in blk["estimators"].items():
            if name in ("A_GAE0", "A_GAE1"):
                continue
            keys.append(f"{key.split('/')[0][2:]}\n{key.split('/')[1][:4]} {name}")
            vals.append(e["var_frac_within_episode"])
            cols.append(ps.COLOR_CYCLE[i % len(ps.COLOR_CYCLE)])
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=5)
    ax.set_ylabel("fraction of Var(A) WITHIN an episode")
    ax.axhline(0.5, color=ps.COLORS["grey"], lw=1, ls=":")
    ax.set_title("Below this, weighting by A is weighting by the episode's outcome")
    fig.tight_layout()
    ps.save_figure(fig, os.path.join(out_dir, "advantage_variance"))
    written.append("advantage_variance")
    plt.close(fig)
    return written


def _traces(out_dir: str, n_fail: int = 20, n_ok: int = 10, seed: int = 0) -> str:
    """Step 4: per-episode value and advantage traces, failed and successful.

    Args:
        out_dir: The eval directory, holding ``advantages.npz``.
        n_fail: Failed episodes to draw.
        n_ok: Successful ones.
        seed: RNG seed for the sample.

    Returns:
        The basename written, or ``""`` if the npz is absent.
    """
    path = os.path.join(out_dir, "advantages.npz")
    if not os.path.exists(path):
        return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import plot_style as ps  # noqa: PLC0415

    ps.apply_paper_style()
    A = np.load(path, allow_pickle=True)
    uid, ri, failed = A["episode_uid"], A["replan_index"], A["failed"]
    vt = A["val_tile"]
    # Held-out tiles ONLY: these traces are the qualitative half of the memorisation check, so
    # drawing them from tiles the critic trained on would show a picture nobody can act on.
    pool_f = np.unique(uid[failed & vt])
    pool_s = np.unique(uid[~failed & vt])
    rng = np.random.default_rng(seed)
    pick_f = rng.choice(pool_f, size=min(n_fail, pool_f.size), replace=False)
    pick_s = rng.choice(pool_s, size=min(n_ok, pool_s.size), replace=False)

    keys = [k for k in A.files if k.endswith("__v_out")]
    key = next((k for k in keys if k.startswith("V_priv__survival")), keys[0] if keys else None)
    if key is None:
        return ""
    stem = key[:-len("__v_out")]
    v = A[key]
    a_td = A.get(f"{stem}__A_TD")


    fig, axes = plt.subplots(2, 1, figsize=(ps.DOUBLE_COL[0], ps.DOUBLE_COL[1] * 1.15),
                             sharex=True)
    for ep_set, col, lab in ((pick_f, ps.COLORS["red"], "failed"),
                             (pick_s, ps.COLORS["blue"], "survived")):
        for e in ep_set:
            m = uid == e
            o = np.argsort(ri[m])
            x = ri[m][o]
            axes[0].plot(x, -v[m][o], color=col, lw=0.7, alpha=0.5)
            if a_td is not None:
                axes[1].plot(x, a_td[m][o], color=col, lw=0.7, alpha=0.5)
        axes[0].plot([], [], color=col, label=lab)
    axes[0].set_ylabel(r"$P(\mathrm{fail}\mid s_t)$  (cross-fit)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Held-out tiles: the trace, and the blame it implies")
    axes[1].axhline(0.0, color=ps.COLORS["grey"], lw=1)
    axes[1].set_ylabel(r"$A_{\mathrm{TD}}$ (one chunk)")
    axes[1].set_xlabel("re-plan index (chunks of 0.24 s)")
    fig.tight_layout()
    ps.save_figure(fig, os.path.join(out_dir, "value_traces"))
    plt.close(fig)
    return "value_traces"


def stage_report(args) -> int:
    """Write ``tables.md`` and the figures.

    Args:
        args: Parsed CLI arguments; only ``out`` is read.

    Returns:
        Process exit status.
    """
    M = json.load(open(os.path.join(args.out, "metrics.json")))
    parts = ["# Critic advantage evaluation -- tables\n",
             f"Windows {M['n_windows']}, episodes {M['n_episodes']} "
             f"({M['n_failed_episodes']} failed), tiles {M['n_tiles']} "
             f"({M['n_held_out_tiles']} held out; {100 * M['frac_rows_held_out_tile']:.1f}% "
             f"of rows). Quality-head gamma {M['gamma_quality']}.\n",
             "\n## 1. Memorization (training tiles; the only rows with an in-fold reading)\n",
             _memorization_table(M),
             "\n## 2. Discrimination against episode progress\n",
             _discrimination_table(M),
             "\n## 2b. The critic as an AWR BASELINE (its role under A_MC)\n",
             _table([[k, _fmt(b["sd_A_base"]), _fmt(b["sd_A_MC"]),
                      _fmt(b["variance_reduction"])]
                     for k, blk in M["heads"].items()
                     if (b := blk.get("baseline"))],
                    ["head", "sd(G - mean G)", "sd(G - V)", "variance removed"]),
             "\n## 3. Estimators, and where their variance comes from\n",
             _estimator_table(M),
             "\n## 4. Credit localisation on failed episodes\n",
             _credit_table(M),
             "\n## 5. AWR weight statistics (clipped at 20x the mean weight)\n",
             _weight_table(M),
             "\n## 5b. Would a filtered-BC set differ from the outcome-only one?\n",
             _novelty_table(M)]

    bp = os.path.join(args.out, "branch_metrics.json")
    if os.path.exists(bp):
        parts += ["\n## 6. Counterfactual: K siblings from one state\n",
                  _branch_tables(json.load(open(bp)))]
    else:
        parts += ["\n## 6. Counterfactual\n\n_(branch_metrics.json absent -- not yet run)_\n"]

    try:
        figs = _plots(M, args.out)
        tr = _traces(args.out)
        if tr:
            figs.append(tr)
        parts.append("\n## Figures\n\n" + "\n".join(f"- `{f}.png`" for f in figs) + "\n")
    except Exception as exc:                                            # noqa: BLE001
        parts.append(f"\n_(figures skipped: {type(exc).__name__}: {exc})_\n")

    path = os.path.join(args.out, "tables.md")
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    print(f"[report] wrote {path}")
    return 0
