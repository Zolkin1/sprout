#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reduce one or several eval runs into statistics and figures. IsaacLab-free.

``--stage reduce`` writes the numbers (and imports no matplotlib, so it runs on a box with no
display); ``--stage plot`` draws from what reduce wrote.  One CLI, so the two cannot disagree
about which cells or definitions were used.

Two things this report refuses to do, both because the alternative produces a confident wrong
number:

* **Quote a rate over pooled episodes.**  Every rate is clip-averaged and bootstrapped over
  TILES, and carries the level its interval came from.  The pooled value is printed beside it
  so any divergence -- which only happens when the design is unbalanced -- is visible.
* **Print a terrain-completion rate for a family where the metric is degenerate.**  On four of
  the nine families the terrain-pinned span reaches the end of the episode, so "got through
  the terrain" IS "survived", and a number there would read as an independent result. Those
  print a dash, and ``terrain_progress`` carries the information instead.

Run::

    <py> scripts/rsl_rl/eval/report.py logs/terrain_eval/<run> [<run2> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import _paths  # noqa: F401

import numpy as np

import schema
import stats

#: How ``success`` is defined, by TERMINATION TERM rather than by IsaacLab's truncated flag.
#: The term sets differ by task -- the teacher has clip-graded failures and no fall detector,
#: the tracker the reverse -- so the partition is read from the artifact, not hard-coded.
TRUNCATION_TERMS = ("time_out", "clip_end")


def build_parser() -> argparse.ArgumentParser:
    """CLI for the reduce/plot stages."""
    p = argparse.ArgumentParser(prog="report.py", description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", help="one or more eval run directories")
    p.add_argument("--stage", default="all", choices=("reduce", "plot", "all"))
    p.add_argument("--out", default=None,
                   help="default <run>/report for one run, logs/eval_compare/<stamp> for many")
    p.add_argument("--labels", default="", help="comma list, one per run dir")
    p.add_argument("--cells", default="", help="comma list of cell keys; empty = all present")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-clusters", type=int, default=3, dest="min_clusters",
                   help="below this many tiles a family's interval is reported as NaN rather "
                        "than as a misleadingly narrow one")
    p.add_argument("--baseline", default="",
                   help="series every contrast is taken against, as '<run label>:<cell key>'; "
                        "default is the first run's clip__train__- cell")
    p.add_argument("--success_rule", default="spatial",
                   choices=("spatial", "legacy_temporal"),
                   help="'spatial' (default) requires the robot to have entered the goal "
                        "region, so a stalled or lagging robot is not a success. "
                        "'legacy_temporal' is the pre-goal-region rule -- survived to a step "
                        "index -- and exists ONLY to read tables written before the goal "
                        "columns. It is not a fallback: a table without them is refused "
                        "under 'spatial' rather than silently reduced under a rule that "
                        "cannot support the claim.")
    p.add_argument("--figures", default="default")
    p.add_argument("--no-svg", action="store_true")
    p.add_argument("--allow-plan-mismatch", action="store_true", dest="allow_plan_mismatch")
    return p


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
def load_run(run_dir: str, cells: list[str] | None = None) -> dict:
    """Read a run's plan, manifest and every complete cell table."""
    root = pathlib.Path(run_dir)
    plan = json.loads((root / "plan.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    out = {"dir": str(root), "plan": plan, "manifest": manifest, "cells": {}}
    for cell_dir in sorted((root / "cells").glob("*")):
        info_path = cell_dir / "cell.json"
        table = cell_dir / "episodes.h5"
        if not (info_path.is_file() and table.is_file()):
            continue
        info = json.loads(info_path.read_text())
        if not info.get("complete"):
            continue
        key = cell_dir.name
        if cells and key not in cells:
            continue
        data, meta = schema.read_episode_table(str(table))
        out["cells"][key] = {"info": info, "data": data, "meta": meta}
    return out


def derive_labels(runs: list[dict], explicit: str) -> list[str]:
    """One label per run: explicit, else the minimal distinguishing field.

    Falls back to the run-dir basename with a leading timestamp stripped, so
    ``2026-08-21_10-00-00_teacher_v7`` labels itself ``teacher_v7`` and a "same tracker,
    different generator" comparison is self-labelling with nothing typed.
    """
    if explicit.strip():
        want = [x.strip() for x in explicit.split(",") if x.strip()]
        if len(want) != len(runs):
            raise SystemExit(f"--labels gave {len(want)} labels for {len(runs)} run dirs")
        return want
    import re  # noqa: PLC0415

    out = []
    for r in runs:
        name = pathlib.Path(r["dir"]).name
        out.append(re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", name) or name)
    return out


def _clip_identity(plan: dict) -> str:
    """The hash that decides whether two runs may be paired on clip id.

    Prefers the plan's own ``clip_plan_hash``; falls back to ``plan_hash`` for plans written
    before that field existed. The distinction matters because ``plan_hash`` also covers the
    creation timestamp and the subset's output path, so two runs over an identical clip set
    carry different ``plan_hash`` values and would be refused.
    """
    return str(plan.get("clip_plan_hash") or plan.get("plan_hash", ""))


def align_runs(runs: list[dict], labels: list[str], *, allow_mismatch: bool) -> list[str]:
    """Refuse comparisons whose clip plans or libraries differ. Returns collected warnings.

    Two distinguishable problems, treated differently on purpose:

    * different clip PLAN but the same library -- narrower, and overridable, because
      intersecting silently changes the estimand (the same run would show two different
      headline numbers in two reports);
    * different clip LIBRARY -- not narrower but WRONG, because the same ``clip_id`` is a
      different motion, and there is no correct intersection. No flag enables it.
    """
    warnings: list[str] = []
    refusals: list[str] = []
    base_lib = runs[0]["plan"]["source_library"]["sha256"]
    base_plan = _clip_identity(runs[0]["plan"])
    # The cross-run checks below need two runs; the per-run tracking_mode check does not, and
    # applies to a single-run report just as much.
    for lab, r in zip(labels[1:], runs[1:]):
        if r["plan"]["source_library"]["sha256"] != base_lib:
            refusals.append(
                f"{lab}: source clip library differs from {labels[0]}. The same clip_id is a "
                "different motion in the two runs, so pairing is not narrower but wrong, and "
                "there is no correct intersection."
            )
        elif _clip_identity(r["plan"]) != base_plan:
            msg = (f"{lab}: clip plan differs from {labels[0]} "
                   f"({_clip_identity(r['plan'])[:12]} vs {base_plan[:12]})")
            (warnings if allow_mismatch else refusals).append(msg)
    for lab, r in zip(labels, runs):
        pol = r["manifest"]["policy"]
        if pol.get("trained_tracking_mode") and pol.get("rollout_tracking_mode") and \
                pol["trained_tracking_mode"] != pol["rollout_tracking_mode"]:
            warnings.append(
                f"{lab}: checkpoint trained tracking_mode="
                f"{pol['trained_tracking_mode']!r} but the rollout ran "
                f"{pol['rollout_tracking_mode']!r}. The actor's observation groups are "
                "identical in both, so the policy behaves as trained, but the "
                "reference-tracking columns are computed against a different reference "
                "decomposition -- do not compare them across runs that disagree here."
            )
    if refusals:
        raise SystemExit("refusing to compare these runs:\n  - " + "\n  - ".join(refusals))
    return warnings


# ---------------------------------------------------------------------------
# Reduction.
# ---------------------------------------------------------------------------
def term_partition(meta: dict) -> tuple[list[str], list[str]]:
    """Split a cell's active termination terms into truncations and failures."""
    terms = list(meta.get("termination_terms") or [])
    trunc = [t for t in terms if t in TRUNCATION_TERMS]
    fail = [t for t in terms if t not in TRUNCATION_TERMS]
    return trunc, fail


def derived_columns(cell: dict, plan: dict,
                    success_rule: str = "spatial") -> dict[str, np.ndarray]:
    """Everything the report needs that the table stores as primitives.

    ``success`` is derived rather than stored precisely because it is a contested definition:
    on the tracker task it means "did not fall", on the teacher task "did not fall AND did not
    trip a clip-graded bound".  Deriving it from the recorded bits means the report can say
    which, and a reader can recompute the other.

    **Under ``spatial`` (the default) success additionally requires the robot to have entered
    the goal region.** Without that clause every outcome signal in the table is a clip-CURSOR
    condition -- ``clip_end`` fires on a step index and every ``terrain_*`` column compares
    ``steps_survived`` against one -- so "success" means *stayed upright for N steps*, not
    *got anywhere*, and a robot that stalls in place, lags several metres behind or drifts
    laterally off the flight scores one.  That is a trivially reachable optimum with none of
    the risk of the task, which makes it fatal to anything that reshapes a policy toward
    outcomes that score well.

    ``legacy_temporal`` reproduces the old rule EXACTLY and exists to read tables written
    before the goal columns.  It is not a fallback: a table without the columns is refused
    under ``spatial`` rather than silently reduced under a rule it cannot support, because a
    spatial claim quietly resting on a temporal measurement is the whole defect.

    Args:
        cell: The loaded cell (``data`` + ``meta``).
        plan: The run's plan.
        success_rule: ``"spatial"`` or ``"legacy_temporal"``.

    Returns:
        The derived columns.

    Raises:
        ValueError: Under ``spatial`` when the table has no ``reached_goal`` column, or has
            one that is entirely un-populated (a cell run without ``--goal_regions``).
    """
    d, meta = cell["data"], cell["meta"]
    terms = list(meta.get("termination_terms") or [])
    _, fail_terms = term_partition(meta)
    bits = d["termination_bits"]
    failed = np.zeros(bits.shape, dtype=bool)
    for name in fail_terms:
        failed |= ((bits >> terms.index(name)) & 1).astype(bool)
    censored = d["admin_censored"].astype(bool)
    fall_h = float((meta.get("overrides") or {}).get(
        "fall_min_height", schema.FALL_MIN_HEIGHT_M))

    if success_rule not in ("spatial", "legacy_temporal"):
        raise ValueError(f"unknown success_rule {success_rule!r}; expected 'spatial' or "
                         "'legacy_temporal'.")
    arrived = np.ones(bits.shape, dtype=bool)
    if success_rule == "spatial":
        if "reached_goal" not in d:
            raise ValueError(
                f"cell {meta.get('cell', '?')} has no `reached_goal` column, so it was "
                "written before the spatial goal existed or was run without "
                "--goal_regions. Its only outcome signals are clip-CURSOR conditions, "
                "which cannot distinguish climbing the terrain from standing still while "
                "the cursor ran out. Re-run the cell with --goal_regions, or pass "
                "--success_rule legacy_temporal to read it under the old rule and know "
                "that is what you are reading.")
        if bool((meta.get("overrides") or {}).get("goal_regions_bootstrap", False)):
            raise ValueError(
                f"cell {meta.get('cell', '?')} was run under a BOOTSTRAP goal region: its "
                "tolerances are PLACEHOLDERS, chosen only to be finite enough to draw, so "
                "its reached_goal -- and therefore its success rate -- is a number produced "
                "by a tool working exactly as designed and meaning nothing. That run exists "
                "to put a gate on screen and to feed build_goal_regions.py. Fit the real "
                "spec from it and re-run the cell against goal_regions.json.")
        arrived = d["reached_goal"].astype(bool)
        if not arrived.any():
            raise ValueError(
                f"cell {meta.get('cell', '?')} has a `reached_goal` column that is zero on "
                f"every one of its {arrived.size} episodes. That is what a cell run WITHOUT "
                "--goal_regions looks like (the column exists and was never populated), and "
                "it is not distinguishable here from a policy that never arrived -- so it is "
                "refused rather than reported as 0.000 success. Check "
                "`overrides.goal_regions` in the cell meta.")
    # THE SUCCESS RULE, and it is now `schema.arrived_before_failing` rather than a second
    # opinion stated here. The old rule was `~failed & ~censored & arrived`, which fails any
    # episode whose fall TERMINATION bit is set regardless of WHEN it fired -- so an episode
    # that reached the goal and then fell in the run-out scored a failure here and a success
    # under `rollout_outcomes.terminal_reward`, on 51 of 298 episodes of one measured rollout.
    # Two arms cannot be compared under two success rules.
    #
    # `legacy_temporal` keeps its own shape: it drops the arrival requirement entirely, so
    # there is no first-event ordering to apply and "upright without going anywhere" counts.
    if success_rule == "spatial":
        strict_ok = schema.arrived_before_failing(d, strict=True)
        loose_ok = schema.arrived_before_failing(d, strict=False)
        has_strict = "first_violation_step" in d and bool((d["first_violation_step"] >= 0).any())
    else:
        strict_ok = loose_ok = ~failed
        has_strict = False

    out = {
        # THE HEADLINE IS STRICT WHENEVER THE STRICT DATA EXISTS. It used to be `loose_ok`
        # unconditionally, so a run that collected the tracking checks reported the fall-only
        # rate under the name "success" and buried the strict one in the reduced table -- the
        # tables carried the answer and the printed report did not. Measured on `strict_v1`:
        # the loose headline read clip 0.9188 / fm 0.6406 while the strict rule those cells
        # actually ran under is clip 0.9089 / fm 0.6061, and the arms are NOT equally
        # affected (the checks cost the generator 3.5x what they cost the clip), so the
        # substitution changes a contrast and not just a level.
        #
        # `success_loose` is kept beside it, never folded in: the pair is what says how much
        # the checks cost. With no tracking columns the two are identical BY CONSTRUCTION,
        # which is why `strict_checks_active` travels with them.
        "success": ((strict_ok if has_strict else loose_ok) & ~censored).astype(float),
        "success_strict": (strict_ok & ~censored).astype(float),
        "success_loose": (loose_ok & ~censored).astype(float),
        "track_reward_mean": (d["track_reward_mean"].astype(float)
                              if "track_reward_mean" in d
                              else np.full(bits.shape, np.nan)),
        "strict_checks_active": np.full(bits.shape, float(has_strict)),
        "failed": failed.astype(float),
        # At the threshold THIS CELL RAN, not the module default. The two came apart when
        # --fall_min_height was added: the termination fired at 0.40 while the derived flag
        # still read 0.50, so `fell` counted episodes the run had not ended and the printed
        # definition named a number nothing had used.
        "fell": schema.fell(d["min_base_height_w"], d["max_tilt_rad"],
                            min_height=fall_h).astype(float),
        "terrain_end_reached": d["terrain_end_reached"].astype(float),
        "terrain_completed": d["terrain_completed"].astype(float),
        "terrain_progress": d["terrain_progress"].astype(float),
        "vel_rmse_lin": np.sqrt(d["vel_mse_vx"] + d["vel_mse_vy"]),
        "vel_rmse_yaw": np.sqrt(d["vel_mse_wz"]),
        "ref_rmse_joint": np.sqrt(d["ref_joint_pos_mse"]),
        "ref_rmse_base_ori": np.sqrt(d["ref_base_ori_mse_rad"]),
        "cost_of_transport": d["cost_of_transport"].astype(float),
        "torque_util_max": d["torque_util_max"].astype(float),
        "action_rate_mean": d["action_rate_mean"].astype(float),
    }
    # Episode duration as a fraction of what was ACHIEVABLE, so families whose clips differ
    # 3.3x in length are comparable. Denominator named, never clip_len (which no episode can
    # reach: clip_end reserves the observation window).
    lu = np.maximum(d["last_usable_step"], 1)
    out["duration_frac"] = d["steps_survived"] / lu
    out["event_observed"] = d["event_observed"].astype(float)
    # Whether the completion metric says anything beyond success, per episode.
    out["completion_informative"] = schema.completion_is_informative(
        d["terrain_end_step"], d["last_usable_step"]).astype(float)
    # Only under `spatial`. Reporting an all-ones column under `legacy_temporal` would state
    # that every episode arrived, when in fact nothing measured whether any of them did.
    if success_rule == "spatial":
        out["reached_goal"] = arrived.astype(float)
        out["min_dist_to_goal_m"] = d["min_dist_to_goal_m"].astype(float)
    return out


#: Metrics reported per cell and per family: ``(column, binary?, label)``.
REPORTED = (
    ("success", True, "success rate"),
    # The pair BESIDE the headline, because what the tracking checks cost is a result in its
    # own right and it is not the same on every arm: measured on `strict_v1` the checks cost
    # the clip arm 0.010 and the generator 0.035, so reading only one of these changes a
    # contrast. Identical to `success` when no checks ran.
    ("success_loose", True, "success rate (fall-only)"),
    ("track_reward_mean", False, "tracking reward"),
    ("fell", True, "fall rate (derived)"),
    # Reported BESIDE success rather than folded silently into it: success is
    # `arrived & ~fell & ~censored`, so seeing the two limbs apart is what says whether a
    # policy is falling over or merely not getting there. Absent under legacy_temporal.
    ("reached_goal", True, "reached goal region"),
    ("min_dist_to_goal_m", False, "min dist to goal (m)"),
    ("terrain_end_reached", True, "reached terrain end"),
    ("terrain_completed", True, "terrain completed"),
    ("terrain_progress", False, "terrain progress"),
    ("vel_rmse_lin", False, "vel RMSE lin (m/s)"),
    ("vel_rmse_yaw", False, "vel RMSE yaw (rad/s)"),
    ("ref_rmse_joint", False, "joint RMSE (rad)"),
    ("duration_frac", False, "duration / achievable"),
    ("cost_of_transport", False, "cost of transport"),
    ("torque_util_max", False, "peak torque util"),
)


def reduce_cell(cell: dict, plan: dict, args) -> dict:
    """Every reported number for one cell, overall and per family."""
    d = cell["data"]
    der = derived_columns(cell, plan, getattr(args, "success_rule", "spatial"))
    fam_names = cell["meta"].get("family_codes") or {}
    code_to_fam = {int(v): k for k, v in fam_names.items()}
    fam = np.array([code_to_fam.get(int(c), "?") for c in d["family_code"]])
    clip, tile = d["clip_id"], d["scene_tile_id"]

    def _est(col: str, binary: bool, mask=None) -> stats.Estimate:
        m = np.ones(len(clip), dtype=bool) if mask is None else mask
        if not m.any():
            return stats.Estimate(float("nan"), float("nan"), float("nan"), "empty", "none")
        return stats.rate(clip[m], tile[m], der[col][m], boot=args.boot, conf=args.ci,
                          seed=args.seed, binary=binary)

    out: dict = {
        "cell": cell["info"]["cell"], "key": cell["info"]["cell"]["key"],
        "n_episodes": int(len(clip)),
        "n_clips": int(np.unique(clip).size),
        "n_tiles": int(np.unique(tile).size),
        "n_censored": int(d["admin_censored"].sum()),
        "termination_terms": cell["meta"].get("termination_terms"),
        "failure_terms": term_partition(cell["meta"])[1],
        # CELL METADATA, not a metric, which is why it is here rather than in REPORTED: the
        # reduce step aggregates only REPORTED columns, so a per-episode flag put there would
        # print as a spurious "rate" row while still being what the definitions block needs.
        # Distinguishes "the checks ran and nothing fired" from "there were no checks" --
        # `success` and `success_loose` are identical in both cases.
        "strict_checks_active": bool(float(np.mean(der["strict_checks_active"])) > 0.5),
        "throughput": cell["info"].get("throughput"),
        "dr": {k: cell["info"]["dr"].get(k)
               for k in ("preset", "ood_scale", "dr_resample", "reset_safe_terms",
                         "startup_only_terms", "obs_corruption")},
        # Surfaced so the report can PRINT the rules it applied rather than the module
        # defaults. A definitions block that names a threshold nothing used is worse than
        # none: it reads as provenance.
        "fall_min_height": (cell["meta"].get("overrides") or {}).get("fall_min_height"),
        "goal_regions": (cell["meta"].get("overrides") or {}).get("goal_regions"),
        "overall": {}, "by_family": {}, "survival": {}, "terminations": {},
    }
    # `reached_goal` / `min_dist_to_goal_m` exist only under the spatial rule, so REPORTED is
    # a superset of what any one reduction produces. Skipping an absent column is right here
    # and would NOT be right for a missing primitive -- the difference is that these two are
    # derived, and their absence is already a stated property of the rule in force.
    reported = [(c, b, lbl) for c, b, lbl in REPORTED if c in der]
    for col, binary, _label in reported:
        out["overall"][col] = _est(col, binary).as_dict()

    # Terrain completion is only reported where it is not identical to success.
    informative = der["completion_informative"].astype(bool)
    for col in ("terrain_end_reached", "terrain_completed"):
        out["overall"][col + "_informative_only"] = (
            _est(col, True, informative).as_dict() if informative.any()
            else {"value": None, "note": "degenerate for every clip in this cell"})

    for f in sorted(set(fam.tolist())):
        m = fam == f
        entry = {"n_episodes": int(m.sum()), "n_clips": int(np.unique(clip[m]).size),
                 "n_tiles": int(np.unique(tile[m]).size),
                 "completion_informative": bool(informative[m].any())}
        if entry["n_tiles"] < args.min_clusters:
            entry["note"] = (f"only {entry['n_tiles']} tile cluster(s); intervals reported "
                             "as NaN rather than as misleadingly narrow ones")
        for col, binary, _label in reported:
            if col in ("terrain_end_reached", "terrain_completed") and not entry[
                    "completion_informative"]:
                entry[col] = {"value": None, "note": "degenerate (= success) for this family"}
                continue
            e = _est(col, binary, m)
            if entry["n_tiles"] < args.min_clusters:
                e.ci_lo = e.ci_hi = float("nan")
                e.method += "+too_few_clusters"
            entry[col] = e.as_dict()
        out["by_family"][f] = entry

    # Survival, on the achievable-fraction axis so families are comparable.
    times, surv, risk = stats.kaplan_meier(der["duration_frac"], der["event_observed"])
    out["survival"] = {
        "axis": "duration / achievable steps",
        "times": times.tolist(), "survival": surv.tolist(), "at_risk": risk.tolist(),
        "rmst_frac": stats.rmst(times, surv, 1.0),
        "median_frac": stats.median_survival(times, surv),
    }
    terms = list(cell["meta"].get("termination_terms") or [])
    bits = d["termination_bits"]
    for i, name in enumerate(terms):
        out["terminations"][name] = float(((bits >> i) & 1).mean())
    out["terminations"]["_multiple_terms"] = float(
        np.mean([bin(int(b)).count("1") > 1 for b in bits]) if len(bits) else 0.0)
    return out


def paired_contrasts(runs: list[dict], labels: list[str], baseline: str, args) -> list[dict]:
    """Paired clip-level differences of every cell against ``baseline``.

    Cells within a run see the same clips, so pairing removes clip-difficulty variance -- the
    dominant component -- and the same tile resample drives both arms in each replicate.
    """
    series: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    for lab, r in zip(labels, runs):
        for key, cell in r["cells"].items():
            d = cell["data"]
            series[f"{lab}:{key}"] = (d["clip_id"], d["scene_tile_id"],
                                      derived_columns(cell, r["plan"],
                                                      getattr(args, "success_rule", "spatial")))
    if baseline not in series:
        print(f"[report] baseline series {baseline!r} not found; no paired contrasts. "
              f"Available: {sorted(series)}")
        return []
    ca, ta, da = series[baseline]
    out = []
    for name, (cb, tb, db) in series.items():
        if name == baseline:
            continue
        row = {"vs": baseline, "series": name}
        for col in ("success", "terrain_progress", "vel_rmse_lin", "duration_frac"):
            est = stats.paired_delta(ca, ta, da[col], cb, tb, db[col],
                                     boot=args.boot, conf=args.ci, seed=args.seed)
            row[col] = est.as_dict()
        out.append(row)
    holm_in = {r["series"]: r["success"].get("extra", {}).get("permutation_p", float("nan"))
               for r in out}
    adjusted = stats.holm(holm_in)
    for r in out:
        r["success"]["extra"]["permutation_p_holm"] = adjusted.get(r["series"])
    return out


def _fmt(e: dict | None, pct: bool = False) -> str:
    """Render one estimate as ``value [lo, hi]``, or a dash when it is not reportable."""
    if not e or e.get("value") is None:
        return "     --    "
    v, lo, hi = e["value"], e.get("ci_lo"), e.get("ci_hi")
    if pct:
        v, lo, hi = (x * 100 if x is not None and np.isfinite(x) else x for x in (v, lo, hi))
    if lo is None or not np.isfinite(lo):
        return f"{v:7.3f}  (n/a)"
    return f"{v:7.3f} [{lo:6.3f},{hi:6.3f}]"


def format_report(reduced: dict) -> str:
    """The human-readable report. Rendered FROM the JSON, never the other way round.

    That direction matters: the repo already has a plotter that regex-scrapes a text report
    for its numbers, which is exactly the coupling this avoids.
    """
    L: list[str] = []
    a = L.append
    a("=" * 100)
    a("TERRAIN EVALUATION REPORT")
    a("=" * 100)
    if reduced["warnings"]:
        a("")
        a("!! WARNINGS")
        for w in reduced["warnings"]:
            a(f"   - {w}")

    a("")
    a("-- PROVENANCE " + "-" * 85)
    for lab, r in zip(reduced["labels"], reduced["runs"]):
        pol, lib = r["policy"], r["clip_library"]
        a(f"  {lab}")
        a(f"    checkpoint     {pol['checkpoint']}")
        a(f"    sha256         {str(pol['checkpoint_sha256'])[:16]}   tracking_mode "
          f"trained={pol.get('trained_tracking_mode')} rollout={pol.get('rollout_tracking_mode')}")
        a(f"    source library {lib['source']['path']}  ({lib['source']['num_clips']} clips)")
        a(f"    eval subset    {lib['subset']['num_clips']} clips, "
          f"{lib['subset']['flat_tiles']} flat tiles, sha {str(lib['subset']['sha256'])[:12]}")
        a(f"    plan_hash      {r['plan_hash'][:16]}")
        for label, g in (r.get("generators") or {}).items():
            a(f"    generator {label}: {g.get('extero_source')}/{g.get('extero_encoder')} "
              f"ch={g.get('extero_channels')} F={g.get('future_len')} "
              f"band={g.get('extero_clip')} sha {str(g.get('sha256'))[:12]}")

    a("")
    a("-- DEFINITIONS IN FORCE " + "-" * 75)
    rule = str(reduced.get("success_rule", "spatial"))
    if rule == "spatial":
        a("  success        = the base ENTERED THE GOAL REGION, no FAILURE termination fired, "
          "and the episode")
        a("                   was not censored. The goal is a downrange half-plane at the "
          "clip's own endpoint,")
        a("                   intersected with a lateral band and a height band, in the "
          "TERRAIN's frame.")
        a("                   Arrival is spatial on purpose: `clip_end` and every terrain_* "
          "column compare")
        a("                   STEP INDICES, so a rule built on them scores a robot that "
          "stalled in place.")
    else:
        a("  success        = no FAILURE termination fired and the episode was not censored.")
        a("                   *** LEGACY TEMPORAL RULE: arrival is NOT tested, so a robot "
          "that stayed")
        a("                   upright without going anywhere counts as a success. ***")
    a(f"                   truncations (not failures): {list(TRUNCATION_TERMS)}")
    # WHICH RULE THE HEADLINE IS, stated rather than inferable. The column is named `success`
    # either way, so without this line a strict table and a loose one print identically --
    # and they differ by 1.0 point on the clip arm and 3.5 on the generator.
    _strict_on = [c["key"] for c in reduced["cells"] if c.get("strict_checks_active")]
    if _strict_on:
        a("                   *** STRICT: a tracking VIOLATION (lateral / height / tilt vs the")
        a("                   served reference) counts as a failure, and the first event wins.")
        a(f"                   Cells with the checks active: {_strict_on}")
        a("                   `success_loose` in the reduced table is the fall-only rate; the")
        a("                   pair is what says what the checks cost.")
    else:
        a("                   LOOSE: no tracking checks in this run -- `success` is fall-only,")
        a("                   and `success_strict` equals it by construction.")
    for c in reduced["cells"]:
        a(f"                   {c['key']}: failure terms {c['failure_terms']}")
        if rule == "spatial":
            gr = (c.get("goal_regions") or "")
            a(f"                   {c['key']}: goal region {gr or '(none recorded)'}")
        break
    falls = sorted({c.get("fall_min_height") for c in reduced["cells"]
                    if c.get("fall_min_height") is not None})
    fall_txt = ", ".join(f"{v:g}" for v in falls) or f"{schema.FALL_MIN_HEIGHT_M:g}"
    a("  fall           = derived from the recorded extremes at "
      f"height < {fall_txt} m or tilt > "
      f"{np.rad2deg(schema.FALL_TILT_RAD):.0f} deg; NOT a termination term, so adding it "
      "could not change when episodes ended.")
    if len(falls) > 1:
        a("                   *** CELLS RAN AT DIFFERENT FALL HEIGHTS; their success rates "
          "are not comparable. ***")
    a("  terrain done   = survived to min(terrain_end + 2 s, last usable step).")
    a("  progress       = fraction of the REACHABLE pinned span traversed; denominated by "
      "min(terrain_end, last usable) so a success scores exactly 1.0 on every family.")
    a("  velocity error = sqrt of the per-episode MSE against the commanded target_vel; MSE "
      "is stored and square-rooted here, because means of RMSEs are biased low.")
    a("  unit of analysis = the CLIP (mean over its episodes); interval = percentile "
      f"bootstrap resampling TILES, B={reduced['boot']}, seed={reduced['seed']}, "
      f"level {reduced['ci']:.2f}.")
    a("  duration       = steps survived / last usable step, so families whose clips differ "
      "3.3x in length are comparable.")

    a("")
    a("-- CELL INVENTORY " + "-" * 81)
    a(f"  {'cell':<26} {'eps':>7} {'clips':>6} {'tiles':>6} {'cens':>5} {'DR':>10} "
      f"{'resample':>9} {'corrupt':>8} {'steps/s':>8}")
    for c in reduced["cells"]:
        t = c.get("throughput") or {}
        a(f"  {c['key']:<26} {c['n_episodes']:>7} {c['n_clips']:>6} {c['n_tiles']:>6} "
          f"{c['n_censored']:>5} {str(c['dr']['preset']):>10} "
          f"{str(c['dr']['dr_resample']):>9} {str(c['dr']['obs_corruption']):>8} "
          f"{t.get('steps_per_s', float('nan')):>8.1f}")

    a("")
    a("-- HEADLINE " + "-" * 87)
    a(f"  {'cell':<26} {'success':>24} {'fall':>24} {'RMST(frac)':>11} {'median':>8}")
    for c in reduced["cells"]:
        a(f"  {c['key']:<26} {_fmt(c['overall']['success']):>24} "
          f"{_fmt(c['overall']['fell']):>24} "
          f"{c['survival']['rmst_frac']:>11.3f} "
          f"{c['survival']['median_frac']:>8.3f}")
    a("")
    a(f"  {'cell':<26} {'vel RMSE lin':>24} {'vel RMSE yaw':>24} {'progress':>24}")
    for c in reduced["cells"]:
        a(f"  {c['key']:<26} {_fmt(c['overall']['vel_rmse_lin']):>24} "
          f"{_fmt(c['overall']['vel_rmse_yaw']):>24} "
          f"{_fmt(c['overall']['terrain_progress']):>24}")
    a("")
    a("  Every rate above is clip-averaged with a tile-clustered interval. The "
      "episode-pooled value is in report.json;")
    a("  the two coincide only when every clip ran the same number of episodes.")

    a("")
    a("-- BY FAMILY " + "-" * 86)
    for c in reduced["cells"]:
        a(f"  [{c['key']}]")
        # `reached goal` sits beside `success`, not folded into it. Success is
        # `arrived AND not fallen`, and the two limbs fail for opposite reasons -- a family
        # where they diverge is one where the robot stays up and does not get there, which
        # reads as the same number as falling if only the product is shown.
        spatial = any("reached_goal" in v for v in c["by_family"].values())
        head = (f"    {'family':<18} {'tiles':>5} {'eps':>6} {'success':>24} ")
        head += f"{'reached goal':>24} " if spatial else ""
        a(head + f"{'terrain done':>24} {'progress':>24}")
        for fam in sorted(c["by_family"]):
            f = c["by_family"][fam]
            row = (f"    {fam:<18} {f['n_tiles']:>5} {f['n_episodes']:>6} "
                   f"{_fmt(f['success']):>24} ")
            row += f"{_fmt(f.get('reached_goal')):>24} " if spatial else ""
            a(row + f"{_fmt(f.get('terrain_completed')):>24} "
              f"{_fmt(f.get('terrain_progress')):>24}")
        degen = [k for k, v in c["by_family"].items() if not v["completion_informative"]]
        if degen:
            a(f"    -- : terrain_end_step >= last usable step for every clip of "
              f"{sorted(degen)}, so completion equals success by construction and is not an "
              "independent result. Read 'progress' for those.")

    a("")
    a("-- TERMINATION BREAKDOWN " + "-" * 74)
    for c in reduced["cells"]:
        parts = ", ".join(f"{k}={v:.3f}" for k, v in c["terminations"].items()
                          if not k.startswith("_"))
        a(f"  {c['key']:<26} {parts}   (>1 term on one step: "
          f"{c['terminations']['_multiple_terms']:.3f})")

    if reduced["contrasts"]:
        a("")
        a("-- PAIRED CONTRASTS " + "-" * 79)
        a(f"  baseline: {reduced['contrasts'][0]['vs']}")
        a(f"  {'series':<34} {'d success':>24} {'perm p':>8} {'holm':>8} {'pairs':>6}")
        for row in reduced["contrasts"]:
            ex = row["success"].get("extra", {})
            a(f"  {row['series']:<34} {_fmt(row['success']):>24} "
              f"{ex.get('permutation_p', float('nan')):>8.3f} "
              f"{ex.get('permutation_p_holm', float('nan')):>8.3f} "
              f"{ex.get('n_pairs', 0):>6}")
        a("  Paired on clip id, differences bootstrapped over the same tile resample in both")
        a("  arms; p from flipping the sign of whole TILES (a cluster-level randomisation).")

    a("")
    a("-- CAVEATS " + "-" * 88)
    for c in reduced["cells"]:
        if c["n_tiles"] < 10:
            a(f"  {c['key']}: only {c['n_tiles']} tile clusters. A percentile bootstrap at "
              "this cluster count is anticonservative; add tiles, not robots per clip.")
    unbalanced = [c["key"] for c in reduced["cells"]
                  if c["overall"]["success"].get("extra", {}).get("unbalanced")]
    if unbalanced:
        a(f"  unbalanced episode counts per clip in {unbalanced}: the clip-averaged and "
          "episode-pooled rates will differ, and only the former is the estimand.")
    a("  Conditional metrics (anything computed over survivors only) are NOT reported here; "
      "the survival curve and the unconditional means are.")
    a("=" * 100)
    return "\n".join(L) + "\n"


def write_csvs(reduced: dict, out: pathlib.Path) -> None:
    """Long-form CSVs: one row per (run, cell, family, metric).

    Long rather than wide because a wide layout breaks every time a metric is added, and this
    is the join surface for a later MuJoCo arm.
    """
    rows = []
    for c in reduced["cells"]:
        for scope, fam, entry in ([("overall", "-", c["overall"])] +
                                  [("family", f, v) for f, v in c["by_family"].items()]):
            for col, _binary, label in REPORTED:
                e = entry.get(col)
                if not isinstance(e, dict) or e.get("value") is None:
                    continue
                rows.append({
                    "run": c.get("run_label", ""), "cell": c["key"], "scope": scope,
                    "family": fam, "metric": col, "label": label,
                    "value": e["value"], "ci_lo": e.get("ci_lo"), "ci_hi": e.get("ci_hi"),
                    "method": e.get("method"), "level": e.get("level"),
                    "n_episodes": e.get("n_episodes"), "n_clips": e.get("n_clips"),
                    "n_tiles": e.get("n_tiles"),
                    "episode_pooled": (e.get("extra") or {}).get("episode_pooled"),
                })
    path = out / "summary_by_family.csv"
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[report] wrote {path} ({len(rows)} rows)")


def main() -> int:
    """Reduce, report, and (unless ``--stage reduce``) draw."""
    args = build_parser().parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    runs = [load_run(r, cells or None) for r in args.run_dirs]
    labels = derive_labels(runs, args.labels)
    warnings = align_runs(runs, labels, allow_mismatch=args.allow_plan_mismatch)

    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path(runs[0]["dir"]) / "report" if len(runs) == 1
        else pathlib.Path("logs") / "eval_compare" / "latest")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    reduced: dict = {"labels": labels, "boot": args.boot, "seed": args.seed, "ci": args.ci,
                     "success_rule": args.success_rule,
                     "warnings": warnings, "runs": [], "cells": []}
    for lab, r in zip(labels, runs):
        reduced["runs"].append({
            "label": lab, "dir": r["dir"], "plan_hash": r["plan"]["plan_hash"],
            "policy": r["manifest"]["policy"], "clip_library": r["manifest"]["clip_library"],
            "generators": r["manifest"].get("generators"),
        })
        for key in sorted(r["cells"]):
            red = reduce_cell(r["cells"][key], r["plan"], args)
            red["run_label"] = lab
            # Keep the RAW cell key alongside the display key: ``paired_contrasts`` indexes
            # its series by "<run label>:<raw cell key>", so building the baseline from the
            # display key double-prefixes it and silently matches no series at all.
            red["cell_key"] = key
            red["key"] = f"{lab}:{key}" if len(runs) > 1 else key
            reduced["cells"].append(red)
    if not reduced["cells"]:
        raise SystemExit(f"no complete cells found under {args.run_dirs}")

    # The reference every contrast is taken against. Default: the first run's clip/train
    # cell, which is the in-distribution baseline; --baseline names another explicitly.
    if args.baseline:
        base_series = args.baseline
    else:
        pick = next((c for c in reduced["cells"] if "clip__train" in c["cell_key"]),
                    reduced["cells"][0])
        base_series = f"{pick['run_label']}:{pick['cell_key']}"
    reduced["contrasts"] = paired_contrasts(runs, labels, base_series, args)

    (out / "report.json").write_text(json.dumps(reduced, indent=2, default=str))
    text = format_report(reduced)
    (out / "report.txt").write_text(text)
    print(text)
    write_csvs(reduced, out)

    if args.stage in ("plot", "all"):
        import figures  # noqa: PLC0415 - imports matplotlib, so only when drawing

        made = figures.draw_all(reduced, out / "figures", which=args.figures,
                                svg=not args.no_svg)
        print(f"[report] wrote {len(made)} figure file(s) under {out / 'figures'}")
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
