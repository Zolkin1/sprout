# Copyright 2026 Zachary Olkin. All rights reserved.

"""Size the goal region from where the CLIP ARM actually finishes.

The tolerances in ``goal_regions.json`` are a measurement, not a guess, and this is the
script that makes them one.  It reads a clip-arm ``episodes.h5`` and fits each family's
region so that essentially every SURVIVING clip-arm episode lands inside it.

**Why the clip arm and not a round number.**  The clip arm succeeds at ~0.94: it is the
tracker following a pre-made reference, i.e. the best this stack does at the task.  Sizing the
region to contain its endpoints guarantees the region is never tighter than the tracker's own
demonstrated ability, so a generator arm that misses it missed for a reason that is the
generator's.  A hand-picked tolerance has no such guarantee, and the failure is silent in the
direction that matters -- too tight, and every arm is penalised for something no policy in the
repo can do.

mjeval is the cautionary precedent.  Its goal started as a 1 m circle, and that was one of
four harness defects each of which moved measured success by MORE than the difference between
the two policies being compared.  It is now a 5-6 m x 2 m rectangle sized per course.

**The circularity, and how it is broken.**  Nothing records where an episode ended unless a
goal region is attached, and the region is what this script produces.  So the calibration run
uses :meth:`GoalRegionSpec.bootstrap` -- infinite tolerances, which record the three raw
terminal offsets exactly as any other spec would while making ``reached_goal`` trivially 1.
That last part is flagged in the spec, copied into the table's meta, and refused by
``report.py``, so a bootstrap run can never be read as a 100%-success result.

Two steps::

    # 1. emit the bootstrap spec, and run the CLIP arm against it
    python scripts/rsl_rl/build_goal_regions.py --write_bootstrap logs/eval/<run>/goal_regions.bootstrap.json
    python scripts/rsl_rl/eval/evaluate.py --stage grid --out logs/eval/<run> \\
        --goal_regions logs/eval/<run>/goal_regions.bootstrap.json --cells clip/train

    # 2. fit the real one from that table
    python scripts/rsl_rl/build_goal_regions.py logs/eval/<run> --out logs/eval/<run>/goal_regions.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _d in (str(_HERE / "eval"), str(_HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import schema  # noqa: E402
import sampler  # noqa: E402

#: Quantile of the surviving clip-arm endpoints each tolerance must contain, before the pad.
#:
#: Not 1.0.  A single clip whose episode ended two metres wide would set the lateral band for
#: its whole family, and the point is to contain the BULK of what the tracker does, not its
#: worst outlier -- the outlier is more likely a near-fall than a definition.  The pad below
#: then buys margin back, so the region is generous without being hostage to one episode.
DEFAULT_QUANTILE: float = 0.99

#: Additive slack on each fitted tolerance, metres.  Deliberately generous: the asymmetry of
#: this parameter is severe.  Too WIDE costs a little discrimination at the margin; too TIGHT
#: penalises every arm for something no policy here can do, and looks exactly like a policy
#: result.  When in doubt, widen.
DEFAULT_PAD_M: float = 0.5

#: Additive slack on the HEIGHT tolerance specifically, metres.
#:
#: Separate from :data:`DEFAULT_PAD_M` because the asymmetry argument above is about DOWNRANGE
#: and LATERAL slack, where half a metre costs a little discrimination and buys a lot of safety.
#: Vertically it does not transfer: a stair riser is 0.13-0.18 m, so a 0.5 m height tolerance is
#: three to four steps and a robot standing most of a flight below the goal scores as arrived --
#: on exactly the families the eval exists to separate. Measured on the v6g clip arm, the
#: tracker's own height error at the goal is 0.040-0.072 m, so a 0.5 m pad is ~10x the thing it
#: is padding. 0.1 is what the shipped goal_regions_anchor_v3.json was fitted with (its
#: provenance records `height_pad_m: 0.1`), before the two pads were collapsed into one.
DEFAULT_HEIGHT_PAD_M: float = 0.1

#: Floors, so a family whose clip arm happens to be very tight cannot produce a region so
#: small that ordinary variation misses it.
MIN_MARGIN_M: float = 1.0
MIN_HALF_WIDTH_M: float = 1.5
#: Below one stair riser (0.13-0.18 m), so the floor cannot by itself merge two steps. 0.35
#: dominated every fitted height (0.040-0.072 + 0.1) and would have made the per-axis pad above
#: a no-op; v3's fitted heights ran 0.12-0.23, i.e. under the old floor entirely.
MIN_HEIGHT_TOL_M: float = 0.12


#: The only columns the fit reads.  Named so the loader can ask for them rather than pulling
#: all 68 -- on a full grid that is most of a gigabyte to compute nine quantiles from.
FIT_COLUMNS: tuple[str, ...] = (
    "termination_bits", "admin_censored", "min_base_height_w", "max_tilt_rad",
    "family_code", "goal_downrange_m", "goal_lateral_m", "goal_height_err_m",
)


def _load_goal_region():
    """Import ``mdp/goal_region.py`` WITHOUT running the ``mdp`` package's ``__init__``.

    The module itself is pure torch + json, but its package ``__init__`` imports
    ``isaaclab.envs.mdp`` and every observation module, which needs a simulator on the box and
    (composed before ``AppLauncher``) binds pip's ``usd-core`` ahead of Kit's.  This script
    reads and writes a JSON file; it has no business booting Isaac to do it.

    Returns:
        The ``goal_region`` module.
    """
    import importlib.util  # noqa: PLC0415

    repo = _HERE.parent.parent
    path = (repo / "source" / "robot_rl" / "robot_rl" / "tasks" / "manager_based"
            / "robot_rl" / "mdp" / "goal_region.py")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; goal_region.py moved, and this path is derived from the "
            "package root rather than counted in '..' hops, so fix the derivation.")
    spec = importlib.util.spec_from_file_location("goal_region", str(path))
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: @dataclass resolves annotations through
    # `sys.modules[cls.__module__].__dict__`, which is None for a module that is being
    # executed but not yet registered, and fails with an AttributeError far from the cause.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_clip_cells(run_dir: str) -> list[tuple[str, dict, dict]]:
    """Every CLIP-arm cell in ``run_dir``.

    Args:
        run_dir: An eval run directory (the one ``evaluate.py --out`` wrote).

    Returns:
        ``[(cell_key, columns, meta), ...]`` for cells whose reference mode is ``clip``.

    Raises:
        FileNotFoundError: When the directory holds no finished clip-arm table.  Fitting from
            a generator arm would size the region to what the thing under test can reach,
            which is circular in the direction that always passes.
    """
    out = []
    seen = set()
    for path in sorted(list(pathlib.Path(run_dir).glob("*/episodes.h5"))
                       + list(pathlib.Path(run_dir).glob("cells/*/episodes.h5"))):
        if path in seen:
            continue
        seen.add(path)
        key = str(path.parent.name)
        if not key.startswith("clip__"):
            continue
        # Only the columns the fit reads. `read_episode_table` loads all 68 otherwise, which
        # on a full grid is most of a gigabyte to compute nine quantiles from.
        try:
            cols, meta = schema.read_episode_table(str(path), columns=list(FIT_COLUMNS))
        except ValueError as exc:
            if "goal_" not in str(exc):
                raise
            raise ValueError(
                f"{key} has no goal columns: it was run without --goal_regions, so nothing "
                "recorded where its episodes ended. Re-run the clip arm against the "
                "bootstrap spec (--write_bootstrap).") from exc
        out.append((key, cols, meta))
    if not out:
        raise FileNotFoundError(
            f"{run_dir} has no clip-arm episodes.h5 (a directory named 'clip__<dr>__'). The "
            "region must be fitted from the CLIP arm: fitting it from a generator arm would "
            "size the goal to what the thing under test can already reach.")
    return out


def _fit_family(down: np.ndarray, lat: np.ndarray, hz: np.ndarray,
                quantile: float, pad: float, height_pad: float) -> dict:
    """Tolerances containing ``quantile`` of one family's surviving endpoints, plus ``pad``.

    Args:
        down: ``[k]`` terminal downrange offsets, metres (negative = short of the clip's end).
        lat: ``[k]`` terminal lateral offsets.
        hz: ``[k]`` terminal height errors.
        quantile: See :data:`DEFAULT_QUANTILE`.
        pad: Downrange and lateral slack. See :data:`DEFAULT_PAD_M`.
        height_pad: Vertical slack, which does NOT share the horizontal number. See
            :data:`DEFAULT_HEIGHT_PAD_M`.

    Returns:
        A dict of the three tolerances plus the raw fitted quantiles, for the report.
    """
    # The margin is how far SHORT the clip arm ends up, so it reads off the negative tail.
    short = np.quantile(np.clip(-down, 0.0, None), quantile)
    wide = np.quantile(np.abs(lat), quantile)
    high = np.quantile(np.abs(hz), quantile)
    return {
        "approach_margin_m": float(max(short + pad, MIN_MARGIN_M)),
        "half_width_m": float(max(wide + pad, MIN_HALF_WIDTH_M)),
        "height_tol_m": float(max(high + height_pad, MIN_HEIGHT_TOL_M)),
        "fitted_q_short_m": float(short),
        "fitted_q_lateral_m": float(wide),
        "fitted_q_height_m": float(high),
        "n_survivors": int(down.size),
    }


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default="",
                   help="eval run directory holding a CLIP-arm episodes.h5")
    p.add_argument("--out", default="", help="where to write goal_regions.json")
    p.add_argument("--write_bootstrap", default="",
                   help="emit an UNCALIBRATED spec here and exit; run the clip arm against "
                        "it to produce the table this script then fits")
    p.add_argument("--skip_families",
                   default=("jump_off,distractor,stairs_dn_pure,stairs_up_pure,"
                            "pure_stairs_down,pure_stairs_up,flat"),
                   help="Families excluded from the FIT. The same two the rest of the "
                        "pipeline drops: `jump_off` travels backwards so its goal sits "
                        "behind the spawn, `distractor` travels less in total than a typical "
                        "approach margin, and the "
                        "`*_pure` families open ON the stairs so their first re-plan runs on "
                        "a null-history embedding fitted on flat. Fitting tolerances from any "
                        "of them would size the region to a task that is not one. BOTH "
                        "spellings of the pure families are listed because the name changed "
                        "with the library: `stairs_up_pure` in the v5-era libraries, "
                        "`pure_stairs_up` in v6. `normalize_family` only strips the `family_` "
                        "prefix and the `.h5` suffix, so a stale spelling matches nothing and "
                        "the family is fitted rather than skipped -- silently, and in the "
                        "direction that sizes the region to a task that is not one. "
                        "`flat` is skipped for a different reason and the same effect: it has "
                        "no terrain feature to reach, so its clip end is an arbitrary point on "
                        "a plane, and a tolerance fitted there measures the sampler\'s speed "
                        "draw rather than the policy. It is scored by the FALL rule instead -- "
                        "see report.FALL_ONLY_FAMILIES.")
    p.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    p.add_argument("--pad_m", type=float, default=DEFAULT_PAD_M,
                   help="Downrange and lateral slack (m) added to each fitted quantile.")
    p.add_argument("--height_pad_m", type=float, default=DEFAULT_HEIGHT_PAD_M,
                   help="VERTICAL slack (m), separate from --pad_m: a stair riser is "
                        "0.13-0.18 m, so the horizontal number applied here would merge "
                        "several steps into one goal.")
    args = p.parse_args()

    goal_region = _load_goal_region()

    if args.write_bootstrap:
        goal_region.GoalRegionSpec.bootstrap().save(args.write_bootstrap)
        print(f"wrote UNCALIBRATED bootstrap spec -> {args.write_bootstrap}\n"
              "Run the CLIP arm against it, then re-run this script on the output "
              "directory. Its own success column is meaningless and report.py refuses it.")
        return 0

    if not args.run_dir or not args.out:
        p.error("need both a run_dir and --out (or --write_bootstrap)")

    skip = {sampler.normalize_family(f) for f in args.skip_families.split(",") if f.strip()}
    fams: dict[str, list[np.ndarray]] = {}
    total = n_skipped = 0
    for key, cols, meta in _load_clip_cells(args.run_dir):
        # "Ran the clip out" is read off the RECORDED `clip_end` bit, not reconstructed from
        # step counts. `steps_survived` counts FOLDED steps, which excludes the terminal
        # frame, and episodes may start mid-clip (`start_step`), so every arithmetic version
        # of this test is off by one or by the start phase -- and both errors are silent.
        fell = schema.fell(cols["min_base_height_w"], cols["max_tilt_rad"]).astype(bool)
        censored = cols["admin_censored"].astype(bool)
        terms = list(meta.get("termination_terms") or [])
        if "clip_end" not in terms:
            raise ValueError(
                f"{key} has no `clip_end` termination ({terms}), so nothing in it says which "
                "episodes ran their clip out. The eval harness adds that term; a table "
                "without it was produced some other way.")
        finished = ((cols["termination_bits"] >> terms.index("clip_end")) & 1).astype(bool)
        alive = ~fell & ~censored & finished
        fam_names = json.loads(json.dumps(meta.get("family_codes", {}))) or {}
        code_to_name = {int(v): k for k, v in fam_names.items()}
        codes = cols["family_code"].astype(int)
        for code in np.unique(codes):
            m = alive & (codes == code)
            if not m.any():
                continue
            name = sampler.normalize_family(code_to_name.get(int(code), "unknown"))
            if name in skip:
                n_skipped += int(m.sum())
                continue
            fams.setdefault(name, []).append(
                np.stack([cols["goal_downrange_m"][m], cols["goal_lateral_m"][m],
                          cols["goal_height_err_m"][m]], axis=-1))
        total += int(alive.sum())
        print(f"[{key}] {int(alive.sum())} of {alive.size} episodes ran the clip out "
              f"upright ({int(fell.sum())} fell, {int(censored.sum())} censored, "
              f"{int((~finished & ~fell & ~censored).sum())} did not reach clip_end)")

    by_family, rows = {}, []
    if not fams:
        raise SystemExit(
            "no episode in this run ran its clip out upright, so there is nothing to fit. "
            "That is what a run cut short by --max_steps looks like (every episode is "
            "admin-censored at the ceiling); give the clip arm enough steps to finish.")
    all_pts = np.concatenate([np.concatenate(v) for v in fams.values()], axis=0)
    default = _fit_family(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                          args.quantile, args.pad_m, args.height_pad_m)
    for name, chunks in sorted(fams.items()):
        pts = np.concatenate(chunks, axis=0)
        fit = _fit_family(pts[:, 0], pts[:, 1], pts[:, 2], args.quantile, args.pad_m,
                          args.height_pad_m)
        by_family[name] = {k: fit[k] for k in
                           ("approach_margin_m", "half_width_m", "height_tol_m")}
        rows.append((name, fit))

    spec = goal_region.GoalRegionSpec(
        default=goal_region.GoalTolerances(
            **{k: default[k] for k in ("approach_margin_m", "half_width_m", "height_tol_m")}),
        by_family={k: goal_region.GoalTolerances(**v) for k, v in by_family.items()},
        provenance={
            "bootstrap": False,
            "fitted_from": os.path.abspath(args.run_dir),
            "arm": "clip",
            "quantile": float(args.quantile),
            "pad_m": float(args.pad_m),
            "height_pad_m": float(args.height_pad_m),
            "n_surviving_episodes": int(total),
            "per_family": {n: f for n, f in rows},
        },
    )
    spec.save(args.out)

    print(f"\n{'family':<18} {'n':>6}  {'margin':>8} {'half-w':>8} {'height':>8}   "
          f"(q{args.quantile:g} + {args.pad_m:g} m pad, {args.height_pad_m:g} m vertical)")
    print("-" * 72)
    for name, fit in rows:
        print(f"{name:<18} {fit['n_survivors']:>6}  {fit['approach_margin_m']:>8.2f} "
              f"{fit['half_width_m']:>8.2f} {fit['height_tol_m']:>8.2f}")
    print(f"{'(default)':<18} {total:>6}  {default['approach_margin_m']:>8.2f} "
          f"{default['half_width_m']:>8.2f} {default['height_tol_m']:>8.2f}")
    if n_skipped:
        print(f"\nexcluded {n_skipped} episode(s) of {sorted(skip)} from the fit.")
    print(f"\nwrote {args.out}\n"
          "CHECK THIS BEFORE USING IT: re-run the clip arm against the fitted spec and "
          "confirm reached_goal is ~1.0 on every family. A family below that is a region "
          "too tight for the tracker itself, and every generator arm would inherit the "
          "penalty as if it were a policy result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
