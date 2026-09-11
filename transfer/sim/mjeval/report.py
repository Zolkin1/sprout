# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reduce one or more MuJoCo eval run directories into tables, JSON and figures.

One deterministic episode per (authored course, commanded speed) cell, ten speeds per course,
no repeats -- so what this has to get right differs from the Isaac harness's job.  There the
danger was pooling repeated draws; here there are none.

* **The unit of analysis is still the COURSE.**  Two speeds on one course share its geometry
  exactly as two clips on one tile shared theirs, so ``scripts/rsl_rl/eval/stats.py`` transfers
  wholesale under ``clip_id -> cell id`` and ``tile_id -> terrain_idx``.  One episode per cell
  makes the clip-averaging stage the identity, so the whole interval is the course spread.
* **Per-family significance is at or past the test's floor, and the floor is printed.**  The
  sign-flip test randomises whole clusters, so with ``N`` courses the smallest attainable
  two-sided p is ``2 / 2**N`` -- 1.0 at the single two-box course, 0.0625 at five, 0.0312 at
  six.  A family cannot reach p<0.05 until it has six courses, and then only when every one of
  them moves the same way; the all-courses row is the one with room for a claim.  Those floors
  are DERIVED from the catalogue, never restated -- it has already grown once under this
  reducer, 18 courses to 20 and the stair families 5 to 6.

Pairing between arms is EXACT rather than matched: ``--zero_noise`` integrates the flow from
``x0 = 0`` (what the hardware controller runs), depth noise is off and MuJoCo is deterministic,
so a repeat at fixed ``(course, speed)`` is bit-identical and the tracker is the only thing
that differs.  Hence one trial per cell and no within-cell variance term anywhere.

Run ``python report.py <run_dir> [<run_dir> ...] [--out DIR] [--no_plots]``.  ``--no_plots`` is
the reduce stage: tables + JSON, and no matplotlib import on any path it takes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

import numpy as np

# <repo>/transfer/sim/mjeval/report.py -> parents[2] == <repo>. Counted from THIS file and
# verified against a marker, the idiom `scripts/rsl_rl/eval/_paths.py` settled on: a
# plausible-but-wrong root fails much later, as a missing artifact rather than as a bad path.
_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[2]
if not (REPO_ROOT / "scripts" / "rsl_rl" / "eval" / "stats.py").is_file():
    raise RuntimeError(f"{__file__} derived REPO_ROOT={REPO_ROOT}, which has no "
                       "scripts/rsl_rl/eval/stats.py; fix the parents[] index if this module "
                       "moved rather than adding a '..' hop.")
# APPENDED, never inserted at 0: `scripts/rsl_rl/eval` carries its own `schema.py` and its own
# `report.py`, and the Isaac `schema` imports perfectly cleanly while exposing a different
# reader (`read_episode_table`, not `read_table`) -- so shadowing this package's siblings would
# surface as an AttributeError far from the cause.
for _d in (REPO_ROOT / "scripts" / "rsl_rl" / "eval", REPO_ROOT / "scripts" / "rsl_rl"):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import schema  # noqa: E402
import stats  # noqa: E402
import terrains  # noqa: E402

#: Classes in catalogue order, which is the order a reader expects them argued.
CLASS_ORDER: tuple[str, ...] = tuple(terrains.CLASS_CODES)
_CODE_TO_CLASS: dict[int, str] = {v: k for k, v in terrains.CLASS_CODES.items()}

#: Columns without which there is no report: identity and outcome. Every other column
#: degrades to a printed dash, so a table missing one still yields what does not depend on it.
REQUIRED_COLUMNS: tuple[str, ...] = ("terrain_idx", "speed_idx", "terrain_class", "verdict")

#: Below this many course clusters a percentile bootstrap has too few distinct resamples to
#: mean anything -- one at n=1, three at n=2 -- so the interval is withheld rather than printed
#: narrow. ``two_box_course`` has 1 course and ``up_into_down`` 2.
MIN_CLUSTERS: int = 3

#: Speed levels per course. The cell id is ``terrain_idx * SPEED_LEVELS + speed_idx``, a
#: FORMULA rather than an enumeration of what a run happened to contain, so two arms that ran
#: different subsets still pair on the cells they share.
SPEED_LEVELS: int = len(terrains.STAIR_SPEEDS)
if len(terrains.BOX_SPEEDS) != SPEED_LEVELS:
    raise RuntimeError("the two speed ladders differ in length; the cell id formula and the "
                       "by-speed table both assume one level count")

_VEL_AXES: tuple[str, ...] = ("vx", "vy", "wz")

#: Courses per class, counted from the catalogue rather than restated. The catalogue is
#: authored and has already grown under this reducer (18 courses to 20, the stair families 5
#: to 6), and a hardcoded family size would have silently mislabelled every p floor by a
#: factor of two while still printing a plausible number.
COURSES_PER_CLASS: dict[str, int] = {
    kl: sum(1 for c in terrains.catalogue() if c.klass == kl) for kl in CLASS_ORDER}
N_COURSES: int = sum(COURSES_PER_CLASS.values())



def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    Returns:
        Parser for ``report.py <run_dir> ... [--out DIR] [--no_plots]``.
    """
    p = argparse.ArgumentParser(prog="report.py", description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", help="one or more mjeval run directories")
    p.add_argument("--out", default=None, help="default <first run_dir>/report")
    p.add_argument("--no_plots", action="store_true",
                   help="the reduce stage: tables + json only, and no matplotlib import")
    p.add_argument("--labels", default="", help="comma list, one arm label per run dir")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_svg", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Loading and identity.
# ---------------------------------------------------------------------------
def speed_ladders() -> dict[str, tuple[float, ...]]:
    """The commanded ladder each class's courses share, read from the catalogue.

    Not from a column: the two ladders differ, so there is no single x axis and a report
    assuming one would mislabel half the sweep.

    Returns:
        ``class -> ladder``.

    Raises:
        RuntimeError: If one class carries two ladders, which would make ``speed_idx`` mean
            two things within one row of the by-speed table.
    """
    out: dict[str, tuple[float, ...]] = {}
    for c in terrains.catalogue():
        if out.setdefault(c.klass, c.speeds) != c.speeds:
            raise RuntimeError(f"class {c.klass} carries two different speed ladders")
    return out


def ladder_summary() -> str:
    """One line naming each distinct speed ladder and the classes that share it.

    Derived rather than written out: the stair and box ladders are authored from the training
    library's own measured command range, so a restated endpoint here would go stale the
    moment either is retuned -- as the box ladder already was.

    Returns:
        e.g. ``10 levels per course: stairs_up/stairs_down 0.60-2.60; box 0.50-3.50 m/s``.
    """
    groups: dict[tuple[float, ...], list[str]] = {}
    for kl, lad in speed_ladders().items():
        groups.setdefault(lad, []).append(kl)
    parts = [f"{'/'.join(kls)} {lad[0]:.2f}-{lad[-1]:.2f}" for lad, kls in groups.items()]
    return f"{SPEED_LEVELS} levels per course: " + "; ".join(parts) + " m/s"


def cell_ids(cols: dict[str, np.ndarray]) -> np.ndarray:
    """The pairing key: one unique int per ``(terrain_idx, speed_idx)``.

    Args:
        cols: Episode columns.

    Returns:
        ``[n]`` int64 cell ids.
    """
    return (cols["terrain_idx"].astype(np.int64) * SPEED_LEVELS
            + cols["speed_idx"].astype(np.int64))


def load_run(run_dir: str) -> dict:
    """Read one run's episode table and manifest.

    Args:
        run_dir: A run directory holding ``episodes.h5`` and ``manifest.json``.

    Returns:
        ``{"dir", "cols", "attrs", "manifest"}``.

    Raises:
        SystemExit: If the table is absent or missing a required column.
    """
    root = pathlib.Path(run_dir)
    table = root / "episodes.h5"
    if not table.is_file():
        raise SystemExit(f"{table} not found -- is {run_dir} an mjeval run directory?")
    cols, attrs = schema.read_table(str(table))
    absent = [c for c in REQUIRED_COLUMNS if c not in cols]
    if absent:
        raise SystemExit(f"{table} has no column(s) {absent}; present: {sorted(cols)}")
    man = root / "manifest.json"
    return {"dir": str(root), "cols": cols, "attrs": attrs,
            "manifest": json.loads(man.read_text()) if man.is_file() else {}}


def derive_labels(runs: list[dict], explicit: str) -> list[str]:
    """One arm label per run: explicit, else the run-dir name with a leading timestamp cut.

    Args:
        runs: Loaded runs.
        explicit: The ``--labels`` comma list, possibly empty.

    Returns:
        One label per run.

    Raises:
        SystemExit: If ``--labels`` gives the wrong count.
    """
    if explicit.strip():
        want = [x.strip() for x in explicit.split(",") if x.strip()]
        if len(want) != len(runs):
            raise SystemExit(f"--labels gave {len(want)} labels for {len(runs)} run dirs")
        return want
    stamp = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?"
    names = [pathlib.Path(r["dir"]).name for r in runs]
    return [re.sub(stamp, "", n) or n for n in names]


def artifact_rows(manifest: dict) -> list[tuple[str, str, str]]:
    """Tracker/generator identity for the inventory, as ``(role, name, sha256 prefix)``.

    Provenance absence prints a line rather than raising: a refusal would hide the numbers as
    well as the gap, and the gap is what the reader needs told.

    Args:
        manifest: The run's ``manifest.json`` contents.

    Returns:
        One row per recorded artifact.
    """
    rows = []
    for role, e in (manifest.get("artifacts") or {}).items():
        if not isinstance(e, dict):
            rows.append((str(role), str(e), ""))
            continue
        name = str(e.get("name") or e.get("path") or "?")
        rows.append((str(role), os.path.basename(name), str(e.get("sha256") or "")[:12]))
    return rows


# ---------------------------------------------------------------------------
# Derived per-episode quantities.
# ---------------------------------------------------------------------------
def derived_columns(cols: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """The 0/1 outcome columns the report reduces, plus warnings raised deriving them.

    ``success`` comes from ``schema.derive_success`` (reached the goal rectangle AND traversed
    the course), not from the verdict enum, and the two are CROSS-CHECKED: they are written by
    different code on the run side, so a disagreement is a real defect and is reported rather
    than silently resolved in favour of either.

    Args:
        cols: Episode columns.

    Returns:
        ``(derived, warnings)``.
    """
    v = terrains.VERDICT_CODES
    verdict = cols["verdict"].astype(int)
    # LOOSE is the headline, so every number published before the strict checks existed stays
    # comparable; STRICT sits beside it and the difference between them is what the checks
    # cost. Never folded together.
    success = np.asarray(schema.derive_success(cols, strict=False)).astype(float)
    strict = np.asarray(schema.derive_success(cols, strict=True)).astype(float)
    has_strict = bool("first_violation_step" in cols
                      and (np.asarray(cols["first_violation_step"]) >= 0).any())
    der = {"success": success,
           "success_strict": strict,
           # A table without the checks reports the two as equal BY CONSTRUCTION. The flag is
           # what stops a reader mistaking that for "measured and nothing ever fired".
           "strict_checks_active": np.full(success.shape, float(has_strict)),
           "fall": (verdict == v["fall"]).astype(float),
           "timeout": (verdict == v["timeout"]).astype(float),
           "violation": (verdict == v["violation"]).astype(float),
           "interrupted": (verdict == v["interrupted"]).astype(float)}
    if "track_reward_mean" in cols:
        der["track_reward_mean"] = cols["track_reward_mean"].astype(float)
    warnings: list[str] = []
    disagree = int((success.astype(bool) != (verdict == v["success"])).sum())
    if disagree:
        warnings.append(
            f"{disagree} of {len(verdict)} episodes have derive_success() disagreeing with "
            "verdict==success. Those are written by different code on the run side; the "
            "tables below use derive_success (reached_target AND traversed).")
    # The event indicator is DERIVED from the verdict rather than read: a fall is the only
    # observed failure, and successes and timeouts are right-censored, which is exactly the
    # definition printed in the report. Storing it too would be one more way to disagree.
    der["event_observed"] = der["fall"]
    if "progress_frac" in cols:
        der["progress_frac"] = cols["progress_frac"].astype(float)
    if "n_plans" in cols and "n_plans_jump" in cols:
        n = cols["n_plans"].astype(float)
        der["plan_jump_rate"] = np.where(
            n > 0, cols["n_plans_jump"].astype(float) / np.maximum(n, 1.0), np.nan)
    return der, warnings


def velocity_blocks(names) -> dict[tuple[str, str], dict[str, str]]:
    """Group every ``vel_mse_*`` column by ``(segment, frame)``.

    The AXIS tokens (``vx``/``vy``/``wz``) and the FRAME token (``body`` = the full-body frame
    kept for parity with the Isaac harness's columns; its absence means the yaw frame the
    headline uses) are this report's vocabulary.  Whatever tokens remain name the SEGMENT,
    which is the writer's -- ``pre`` before the first riser or box face, ``post`` from there
    on, and no token at all for the whole episode.  Parsing rather than matching one fixed
    spelling means a third segment costs the reducer nothing, while a renamed axis yields an
    empty block loudly.

    Args:
        names: The table's column names.

    Returns:
        ``(segment, frame) -> {axis: column name}``.
    """
    out: dict[tuple[str, str], dict[str, str]] = {}
    for n in names:
        if not n.startswith("vel_mse_"):
            continue
        toks = n[len("vel_mse_"):].split("_")
        axis = next((t for t in toks if t in _VEL_AXES), None)
        if axis is None:
            continue
        rest = [t for t in toks if t != axis]
        frame = "body" if "body" in rest else "yaw"
        segment = "_".join(t for t in rest if t != "body") or "all"
        out.setdefault((segment, frame), {})[axis] = n
    return out


def velocity_summary(cols: dict[str, np.ndarray], block: dict[str, str],
                     mask: np.ndarray) -> dict[str, float]:
    """RMSE for one ``(segment, frame)`` block over the masked episodes.

    The root is taken at the JOIN, never per episode: a mean of per-episode RMSEs is biased low
    by Jensen, which is why the table stores squares at all.  The linear RMSE sums only the
    lateral axes the block carries, and names them -- the segmented columns are written for
    ``vx`` alone, so folding an absent ``vy`` in as NaN would erase the one number those
    columns exist to give.

    Args:
        cols: Episode columns.
        block: ``{axis: column name}`` from :func:`velocity_blocks`.
        mask: ``[n]`` which episodes to include.

    Returns:
        ``{"rmse_lin", "axes", "rmse_yaw", "cmd_vx", "n"}``; NaN where an axis is absent.
    """
    def _mean(axis: str) -> float:
        """Mean of one axis's stored MSE over the mask; NaN when that column is absent."""
        name = block.get(axis)
        if name is None:
            return float("nan")
        v = cols[name][mask].astype(float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    lin = [ax for ax in ("vx", "vy") if ax in block]
    # `cmd_vx` is the SWEPT command and is constant within an episode, so its mean over a
    # scope is what makes the error beside it interpretable.
    cmd = (float(np.nanmean(cols["cmd_vx"][mask].astype(float)))
           if "cmd_vx" in cols and mask.any() else float("nan"))
    return {"rmse_lin": float(np.sqrt(sum(_mean(ax) for ax in lin))) if lin else float("nan"),
            "axes": "+".join(lin) or "-", "rmse_yaw": float(np.sqrt(_mean("wz"))),
            "cmd_vx": cmd, "n": int(mask.sum())}


def p_floor(n_clusters: int) -> float:
    """Smallest two-sided p a whole-cluster sign-flip test can attain, ``2 / 2**N``.

    Both the all-positive and the all-negative assignment reproduce the observed statistic, so
    two of the ``2**N`` assignments always count as hits.

    Args:
        n_clusters: Courses in the family being tested.

    Returns:
        The floor, NaN when there is nothing to flip.
    """
    return float(2.0 ** (1 - n_clusters)) if n_clusters > 0 else float("nan")


# ---------------------------------------------------------------------------
# Reduction.
# ---------------------------------------------------------------------------
def reduce_arm(run: dict, label: str, args: argparse.Namespace) -> dict:
    """Every reported number for one arm.

    Args:
        run: A loaded run.
        label: Its arm label.
        args: Parsed CLI arguments.

    Returns:
        The arm's JSON block: headline, by class, by speed, velocity, plans.
    """
    cols = run["cols"]
    der, warnings = derived_columns(cols)
    cell = cell_ids(cols)
    course = cols["terrain_idx"].astype(np.int64)
    klass = np.array([_CODE_TO_CLASS.get(int(c), "?") for c in cols["terrain_class"]])
    ladders = speed_ladders()
    all_mask = np.ones(cell.size, dtype=bool)

    def _est(col: str, mask: np.ndarray, binary: bool = True) -> dict:
        """One clip-averaged, course-clustered estimate, or a note saying why there is none."""
        if col not in der:
            return {"value": None, "note": f"no {col} column in this table"}
        if not mask.any():
            return {"value": None, "note": "no episodes"}
        e = stats.rate(cell[mask], course[mask], der[col][mask], boot=args.boot, conf=args.ci,
                       seed=args.seed, binary=binary)
        if e.n_tiles < MIN_CLUSTERS:
            e.ci_lo = e.ci_hi = float("nan")
            e.method += "+too_few_clusters"
        return e.as_dict()

    out: dict = {
        "label": label, "dir": run["dir"], "warnings": warnings,
        "n_episodes": int(cell.size), "n_courses": int(np.unique(course).size),
        "n_cells": int(np.unique(cell).size),
        "n_interrupted": int(der["interrupted"].sum()),
        "artifacts": artifact_rows(run["manifest"]),
        "headline": {k: _est(k, all_mask) for k in ("success", "fall", "timeout")},
        "by_class": {}, "by_speed": {}, "velocity": {}, "plans": {},
    }
    out["headline"]["p_floor"] = p_floor(out["n_courses"])
    if "progress_frac" in der:
        t, s, _ = stats.kaplan_meier(der["progress_frac"], der["event_observed"])
        out["headline"]["progress_rmst"] = stats.rmst(t, s, 1.0)
        out["headline"]["progress_median"] = stats.median_survival(t, s)

    for kl in CLASS_ORDER:
        m = klass == kl
        n_courses = int(np.unique(course[m]).size) if m.any() else 0
        entry = {"n_episodes": int(m.sum()), "n_courses": n_courses,
                 "p_floor": p_floor(n_courses), "interval_possible": n_courses >= MIN_CLUSTERS}
        entry.update({k: _est(k, m) for k in ("success", "fall", "timeout")})
        out["by_class"][kl] = entry
        ladder = ladders.get(kl, ())
        levels = [m & (cols["speed_idx"].astype(int) == i) for i in range(len(ladder))]
        out["by_speed"][kl] = {
            "speeds": list(ladder), "n": [int(ms.sum()) for ms in levels],
            "success": [float(der["success"][ms].mean()) if ms.any() else float("nan")
                        for ms in levels]}

    # Ordered approach -> on-course -> whole episode, and yaw before body, so the table reads
    # in the order the robot met the course rather than alphabetically ("all" first).
    order = {"pre": 0, "post": 1, "all": 2}
    for (segment, frame), block in sorted(
            velocity_blocks(cols.keys()).items(),
            key=lambda kv: (order.get(kv[0][0], 3), kv[0][0], kv[0][1] == "body")):
        entry = {"overall": velocity_summary(cols, block, all_mask)}
        entry.update({kl: velocity_summary(cols, block, klass == kl) for kl in CLASS_ORDER})
        out["velocity"][f"{segment}/{frame}"] = entry

    if "plan_jump_rate" in der and float(np.nansum(cols["n_plans"].astype(float))) > 0:
        out["plans"] = {"overall": _est("plan_jump_rate", all_mask, binary=False)}
        out["plans"].update({kl: _est("plan_jump_rate", klass == kl, binary=False)
                             for kl in CLASS_ORDER})
    return out


def paired_contrasts(runs: list[dict], labels: list[str],
                     args: argparse.Namespace) -> list[dict]:
    """Paired success deltas of every arm against the first, overall and per class.

    Paired on cell id, clustered on course.  Every trial being deterministic makes the pair the
    same course at the same speed with the same generator, so the delta is attributable to the
    tracker alone and there is no draw-to-draw component to average out.

    Args:
        runs: Loaded runs, the first being the baseline.
        labels: One label per run.
        args: Parsed CLI arguments.

    Returns:
        One row per non-baseline arm, with a Holm adjustment across the families it holds.
    """
    base = runs[0]
    base_der, _ = derived_columns(base["cols"])
    base_cell = cell_ids(base["cols"])
    base_course = base["cols"]["terrain_idx"].astype(int)
    base_class = np.array([_CODE_TO_CLASS.get(int(c), "?")
                           for c in base["cols"]["terrain_class"]])
    rows = []
    for lab, r in zip(labels[1:], runs[1:]):
        der, _ = derived_columns(r["cols"])
        cell, course = cell_ids(r["cols"]), r["cols"]["terrain_idx"].astype(int)
        kls = np.array([_CODE_TO_CLASS.get(int(c), "?") for c in r["cols"]["terrain_class"]])
        est = stats.paired_delta(base_cell, base_course, base_der["success"],
                                 cell, course, der["success"],
                                 boot=args.boot, conf=args.ci, seed=args.seed)
        row: dict = {"baseline": labels[0], "arm": lab, "overall": est.as_dict(),
                     "by_class": {}}
        row["overall"]["p_floor"] = p_floor(est.n_tiles)
        ps: dict[str, float] = {}
        for kl in CLASS_ORDER:
            ma, mb = base_class == kl, kls == kl
            if not (ma.any() and mb.any()):
                continue
            e = stats.paired_delta(base_cell[ma], base_course[ma], base_der["success"][ma],
                                   cell[mb], course[mb], der["success"][mb],
                                   boot=args.boot, conf=args.ci, seed=args.seed)
            d = e.as_dict()
            d["p_floor"] = p_floor(e.n_tiles)
            if e.n_tiles < MIN_CLUSTERS:
                d["ci_lo"] = d["ci_hi"] = float("nan")
            row["by_class"][kl] = d
            ps[kl] = d.get("extra", {}).get("permutation_p", float("nan"))
        for kl, adj in stats.holm(ps).items():
            row["by_class"][kl]["permutation_p_holm"] = adj
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Rendering. The text is built FROM the JSON, never the other way round.
# ---------------------------------------------------------------------------
_DEFINITIONS = """\
  success       = reached the goal rectangle AND traversed the course -- the root moved {frac:.0%}
                  of the flight height in the course's direction (datum {datum} m). A circle
                  alone can be walked around: the lanes are finite in y and the floor is not.
  verdict       = a three-way partition, success / fall / timeout. 'interrupted' is
                  administrative (the rollout was cut) and is counted apart, never as a failure.
  event_observed= 1 for a FALL only. Successes and timeouts are right-CENSORED, so every
                  survival quantity below is Kaplan-Meier; a mean over falls alone conditions
                  on failing and flatters a policy that fails only late.
  cluster       = the COURSE (terrain_idx). Two speeds on one course share its geometry,
                  exactly as two clips on one tile shared theirs. ONE deterministic trial per
                  cell, so the clip-averaging stage is the identity and the whole interval
                  comes from the courses.
  interval      = percentile bootstrap over courses, B={boot}, seed={seed}, level {ci:.2f};
                  withheld below {min_clusters} clusters rather than printed narrow.
  P-VALUE FLOOR = the sign-flip test randomises whole COURSES, so with N courses the smallest
                  attainable two-sided p is 2/2**N. Derived from the catalogue, per family:
{floors}
                  A family below six courses cannot reach p<0.05 at any effect size, and at six
                  only when every course moves the same way. ALL {n_courses} courses together
                  floor at {f_all:.2e} -- that row is the one with room for a significance claim."""

_BY_CLASS_NOTE = """\
    NO INTERVAL POSSIBLE = fewer than {min_clusters} course clusters; today {classes}.
    A one-cluster bootstrap has ONE distinct resample and a two-cluster one three, so a
    percentile interval there is a zero-width point dressed as an error bar."""

_BY_SPEED_NOTE = """\
  The curve the whole sweep exists for. The ladders differ per family because the training
  library commands them differently, so one shared axis would be out of distribution at an end
  for every course:
    {ladders}
  Each entry is a fraction over that class's courses at that level (counts in report.json); no
  interval, because at a handful of clusters the bar would be wider than the axis."""

_VELOCITY_NOTE = """\
  sqrt of the MEAN of the stored MSE, rooted at the JOIN: a mean of per-episode RMSEs is biased
  low. The headline columns are the YAW frame; a '/body' block is the full-body frame, kept for
  parity with the Isaac harness's columns. Segments: 'pre' is the approach before the first
  riser or box face, 'post' is on the course, 'all' is the whole episode. 'axes' names which
  lateral channels the block carries -- the segmented columns are vx-only. cmd_vx sits beside
  them because a velocity error is uninterpretable without the speed it was committed at."""

_CONTRAST_NOTE = """\
    Read every per-family p against its own floor: none can go below it, so a p of 0.0625 on
    five courses is the test SATURATING, not weak evidence of no effect. The ALL COURSES row is
    the only one whose floor leaves room for a significance claim."""

_PLANS_NOTE = """\
  READ AGAINST A CONTROL, NEVER AS AN ABSOLUTE: the classifier's own stairs->jump confusion is
  ~0.10 on these families, which is most of any small number here. It is a population statistic
  and not a per-plan label -- within matched difficulty bands a flagged plan is no more
  divergent from its reference than an unflagged one."""


def _fmt(e: dict | None, width: int = 22) -> str:
    """Render one estimate as ``value [lo, hi]``, or a dash when it is not reportable.

    Args:
        e: An ``Estimate.as_dict()``, or a ``{"value": None}`` note.
        width: Column width to right-align into.

    Returns:
        The formatted cell.
    """
    if not e or e.get("value") is None:
        return "--".rjust(width)
    v, lo, hi = e["value"], e.get("ci_lo"), e.get("ci_hi")
    if lo is None or not np.isfinite(lo):
        return f"{v:7.3f}   (no interval)".rjust(width)
    return f"{v:7.3f} [{lo:6.3f},{hi:6.3f}]".rjust(width)


def format_report(reduced: dict) -> str:
    """The human-readable report, in the order the argument has to be made.

    Args:
        reduced: The JSON block :func:`main` assembles.

    Returns:
        The full text.
    """
    L: list[str] = []
    a = L.append
    a("=" * 100)
    a("MUJOCO TERRAIN EVALUATION -- closed-loop generator + tracker")
    a("=" * 100)
    for arm in reduced["arms"]:
        for w in arm["warnings"]:
            a(f"!! {arm['label']}: {w}")

    a("\n-- DEFINITIONS IN FORCE " + "-" * 75)
    floors = "\n".join(
        f"                    {kl:<16} {n:>2} course{'s' if n != 1 else ' '} -> floor "
        f"{p_floor(n):.4f}" + ("" if p_floor(n) < 0.05 else "   (p<0.05 unreachable)")
        for kl, n in COURSES_PER_CLASS.items())
    a(_DEFINITIONS.format(frac=terrains.TRAVERSAL_FRAC, datum=terrains.SPAWN_BASE_Z,
                          boot=reduced["boot"], seed=reduced["seed"], ci=reduced["ci"],
                          min_clusters=MIN_CLUSTERS, floors=floors, n_courses=N_COURSES,
                          f_all=p_floor(N_COURSES)))

    a("\n-- RUN INVENTORY " + "-" * 82)
    for arm in reduced["arms"]:
        a(f"  {arm['label']}\n    dir        {arm['dir']}")
        for role, name, sha in arm["artifacts"]:
            a(f"    {role:<10} {name}  sha {sha or '(not recorded)'}")
        if not arm["artifacts"]:
            a("    artifacts  NOT RECORDED in manifest.json -- this report cannot say which "
              "pair produced these numbers")
        a(f"    episodes   {arm['n_episodes']} over {arm['n_courses']} courses "
          f"({arm['n_cells']} cells); interrupted {arm['n_interrupted']}")
        a(f"    speeds     {ladder_summary()}")

    a("\n-- HEADLINE (all courses) " + "-" * 73)
    a(f"  {'arm':<22} {'success':>22} {'fall':>22} {'timeout':>22}")
    for arm in reduced["arms"]:
        h = arm["headline"]
        a(f"  {arm['label']:<22} {_fmt(h['success'])} {_fmt(h['fall'])} {_fmt(h['timeout'])}")
        if "progress_rmst" in h:
            a(f"  {'':<22} progress RMST {h['progress_rmst']:.3f} of the course "
              f"(Kaplan-Meier, falls are the events); median {h['progress_median']:.3f}")

    a("\n-- BY COURSE CLASS " + "-" * 80)
    for arm in reduced["arms"]:
        a(f"  [{arm['label']}]")
        a(f"    {'class':<16} {'crs':>4} {'eps':>4} {'success':>22} {'fall':>22} "
          f"{'timeout':>22}")
        for kl in CLASS_ORDER:
            c = arm["by_class"][kl]
            mark = "" if c["interval_possible"] else "   <- NO INTERVAL POSSIBLE"
            a(f"    {kl:<16} {c['n_courses']:>4} {c['n_episodes']:>4} {_fmt(c['success'])} "
              f"{_fmt(c['fall'])} {_fmt(c['timeout'])}{mark}")
    thin = [f"{kl} ({n})" for kl, n in COURSES_PER_CLASS.items() if n < MIN_CLUSTERS]
    a(_BY_CLASS_NOTE.format(min_clusters=MIN_CLUSTERS,
                            classes=", ".join(thin) or "no class"))

    a("\n-- BY SPEED (success rate vs commanded forward speed) " + "-" * 46)
    a(_BY_SPEED_NOTE.format(ladders=ladder_summary()))
    for arm in reduced["arms"]:
        a(f"  [{arm['label']}]")
        for kl in CLASS_ORDER:
            row = arm["by_speed"][kl]
            if not row["speeds"]:
                continue
            a(f"    {kl:<16} cmd  " + " ".join(f"{s:5.2f}" for s in row["speeds"]))
            a(f"    {'':<16} succ " + " ".join("   --" if not np.isfinite(v) else f"{v:5.2f}"
                                               for v in row["success"]))

    a("\n-- VELOCITY TRACKING " + "-" * 79)
    a(_VELOCITY_NOTE)
    for arm in reduced["arms"]:
        a(f"  [{arm['label']}]")
        if not arm["velocity"]:
            a("    no vel_mse_* columns in this table")
        for key in arm["velocity"]:
            a(f"    {key}\n      {'scope':<16} {'lin RMSE':>10} {'axes':>6} "
              f"{'yaw RMSE':>10} {'cmd vx':>8} {'eps':>5}")
            for scope in ("overall",) + CLASS_ORDER:
                v = arm["velocity"][key][scope]
                a(f"      {scope:<16} {v['rmse_lin']:>10.3f} {v['axes']:>6} "
                  f"{v['rmse_yaw']:>10.3f} {v['cmd_vx']:>8.3f} {v['n']:>5}")

    if reduced["contrasts"]:
        a("\n-- PAIRED CONTRASTS " + "-" * 79)
        a("  Paired on cell id (course, speed), clustered on course. Every trial is "
          "deterministic, so the\n  pairing is EXACT: same course, same speed, same "
          "generator, only the tracker differs.")
        for row in reduced["contrasts"]:
            o, ex = row["overall"], row["overall"].get("extra", {})
            a(f"  [{row['arm']} - {row['baseline']}]  d success")
            a(f"    {'ALL COURSES':<16} {_fmt(o)} p={ex.get('permutation_p', float('nan')):.4f}"
              f"  floor={o['p_floor']:.2e}  pairs={ex.get('n_pairs', 0)}")
            a(f"    {'class':<16} {'delta':>22} {'perm p':>9} {'holm':>9} {'floor':>9} "
              f"{'pairs':>6}")
            for kl, d in row["by_class"].items():
                dx = d.get("extra", {})
                a(f"    {kl:<16} {_fmt(d)} {dx.get('permutation_p', float('nan')):>9.4f} "
                  f"{d.get('permutation_p_holm', float('nan')):>9.4f} {d['p_floor']:>9.4f} "
                  f"{dx.get('n_pairs', 0):>6}")
            a(_CONTRAST_NOTE)

    if any(arm["plans"] for arm in reduced["arms"]):
        a("\n-- PLAN CLASSIFICATION (n_plans_jump / n_plans) " + "-" * 52)
        a(_PLANS_NOTE)
        for arm in reduced["arms"]:
            if not arm["plans"]:
                continue
            a(f"  [{arm['label']}]")
            a(f"    {'scope':<16} {'jump fraction':>22}")
            for scope in ("overall",) + CLASS_ORDER:
                if scope in arm["plans"]:
                    a(f"    {scope:<16} {_fmt(arm['plans'][scope])}")
    a("=" * 100)
    return "\n".join(L) + "\n"


def draw_figures(reduced: dict, out_dir: pathlib.Path, *, svg: bool = True) -> list[str]:
    """The success-vs-speed curves and the verdict partition. Imports matplotlib.

    Args:
        reduced: The JSON block.
        out_dir: Where to write.
        svg: Also write SVG beside each PNG.

    Returns:
        Paths written.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import plot_style  # noqa: PLC0415

    plot_style.apply_paper_style()
    colours = {kl: plot_style.COLOR_CYCLE[i % len(plot_style.COLOR_CYCLE)]
               for i, kl in enumerate(CLASS_ORDER)}
    styles, arms = ("-", "--", ":", "-."), reduced["arms"]
    written: list[str] = []

    fig, ax = plt.subplots(figsize=plot_style.figsize(1))
    for j, arm in enumerate(arms):
        for kl in CLASS_ORDER:
            row = arm["by_speed"][kl]
            if not row["speeds"]:
                continue
            ax.plot(row["speeds"], row["success"], styles[j % len(styles)], marker="o", ms=3,
                    color=colours[kl],
                    label=f"{kl} ({arm['label']})" if len(arms) > 1 else kl)
    ax.set_xlabel("commanded forward speed [m/s]")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=6, ncol=2)
    written += plot_style.save_figure(
        fig, out_dir / "success_vs_speed", svg=svg,
        caption="Success (reached the target AND traversed the course) against commanded "
                "forward speed, per course class. Each point is a fraction over that class's "
                "courses at one level; no interval, because five clusters give a bar wider "
                "than the axis.")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=plot_style.figsize(1))
    width, x = 0.8 / max(len(arms), 1), np.arange(len(CLASS_ORDER))
    for j, arm in enumerate(arms):
        bottom = np.zeros(len(CLASS_ORDER))
        for key, colour in (("success", plot_style.COLORS["teal"]),
                            ("fall", plot_style.COLORS["red"]),
                            ("timeout", plot_style.COLORS["grey"])):
            vals = np.array([arm["by_class"][kl][key].get("value") or 0.0
                             for kl in CLASS_ORDER], dtype=float)
            ax.bar(x + j * width - 0.4 + width / 2, vals, width * 0.9, bottom=bottom,
                   color=colour, label=key if j == 0 else None)
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([plot_style.tex_escape(k) for k in CLASS_ORDER], rotation=20,
                       fontsize=6)
    ax.set_ylabel("fraction of episodes")
    ax.legend(fontsize=6)
    written += plot_style.save_figure(
        fig, out_dir / "verdict_by_class", svg=svg,
        caption="Verdict partition per course class"
                + (f"; bars grouped {[a['label'] for a in arms]} left to right."
                   if len(arms) > 1 else "."))
    plt.close(fig)
    return written


def main() -> int:
    """Load, reduce, write ``report.txt`` / ``report.json``, and unless ``--no_plots`` draw.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args()
    runs = [load_run(d) for d in args.run_dirs]
    labels = derive_labels(runs, args.labels)
    out = (pathlib.Path(args.out) if args.out
           else pathlib.Path(runs[0]["dir"]) / "report").resolve()
    out.mkdir(parents=True, exist_ok=True)

    reduced = {"boot": args.boot, "ci": args.ci, "seed": args.seed, "labels": labels,
               "min_clusters": MIN_CLUSTERS, "speed_levels": SPEED_LEVELS,
               "arms": [reduce_arm(r, lab, args) for r, lab in zip(runs, labels)],
               "contrasts": paired_contrasts(runs, labels, args) if len(runs) > 1 else []}
    (out / "report.json").write_text(json.dumps(reduced, indent=2, default=str))
    text = format_report(reduced)
    (out / "report.txt").write_text(text)
    print(text)
    if not args.no_plots:
        made = draw_figures(reduced, out / "figures", svg=not args.no_svg)
        print(f"[report] wrote {len(made)} figure file(s) under {out / 'figures'}")
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
