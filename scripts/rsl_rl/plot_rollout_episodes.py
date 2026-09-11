# Copyright 2026 Zachary Olkin. All rights reserved.

"""Draw a few rollout episodes against their goal region, to check the verdict by eye.

The success rule is spatial and the region is a measured artifact, so "does the label match
where the robot actually went" is a question worth answering by looking rather than by
reading a rate.  This picks N successes and N failures and draws each one twice:

* **left, plan view** -- the robot's path in the TERRAIN frame, with the goal region drawn.
  The region is a downrange half-plane intersected with a lateral band, so the shaded area
  extends off the right of the axes on purpose: anything inside it has arrived.
* **right, downrange over time** -- the same path against the arrival boundary, which shows
  WHEN it crossed (or how far short it stopped).

Reading them together is what separates the two ways a failure happens: falling on the
terrain, and staying upright without getting anywhere.  A rate cannot tell those apart and
they want completely different fixes.

The path is the plan ANCHOR, one point per re-plan (0.24 s), not the full 50 Hz trajectory --
that is what a rollout records, and it is plenty to see where the robot went.

Usage::

    python scripts/rsl_rl/plot_rollout_episodes.py <rollout stem> --out <dir>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import plot_style  # noqa: E402
import rollout_outcomes as ro  # noqa: E402


def _pick(episodes: dict, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose ``n`` successes and ``n`` failures, spread over families where possible.

    Args:
        episodes: Columns from :func:`rollout_outcomes.read_rollout_episodes`.
        n: How many of each.
        seed: RNG seed, so a re-run shows the same episodes.

    Returns:
        ``(success_uids, failure_uids)``.

    Raises:
        SystemExit: When the run has none of one class -- there is nothing to compare, and a
            figure of six successes labelled "3 successes, 3 failures" would be worse than
            no figure.
    """
    live = ~episodes["censored"].astype(bool)
    reward = ro.terminal_reward(episodes)
    ok = live & (reward == 0.0)
    bad = live & (reward == -1.0)
    if not ok.any() or not bad.any():
        raise SystemExit(
            f"this run has {int(ok.sum())} success(es) and {int(bad.sum())} failure(s) among "
            f"{int(live.sum())} finished episodes; need at least one of each to compare.")
    rng = np.random.default_rng(seed)

    def take(mask):
        """One per family first, then fill at random -- a spot check wants variety."""
        uids, fams = episodes["episode_uid"][mask], episodes["family_code"][mask]
        chosen = []
        for f in rng.permutation(np.unique(fams)):
            pool = uids[fams == f]
            chosen.append(pool[rng.integers(len(pool))])
            if len(chosen) >= n:
                break
        rest = [u for u in uids if u not in set(chosen)]
        while len(chosen) < n and rest:
            chosen.append(rest.pop(rng.integers(len(rest))))
        return np.asarray(chosen[:n])

    return take(ok), take(bad)


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stem", help="rollout stem: <stem>.h5 and <stem>.episodes.h5")
    p.add_argument("--out", default="", help="output directory (default: beside the stem)")
    p.add_argument("--n", type=int, default=3, help="episodes per class")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    stem = args.stem[:-3] if args.stem.endswith(".h5") else args.stem
    stem = stem[:-9] if stem.endswith(".episodes") else stem
    episodes, ea = ro.read_rollout_episodes(stem + ".episodes.h5")
    windows, _ = ro.read_rollout_windows(
        stem + ".h5", columns=["episode_uid", "replan_index", "anchor_pos",
                               "goal_downrange_m", "goal_lateral_m", "goal_height_err_m"])
    fam_names = ea.get("family_names", ["unknown"])
    # The region as RECORDED, read from the spec the run names in its own provenance -- not
    # a path retyped here, which could point at a different calibration.
    spec_path = ea.get("goal_regions", "")
    if not spec_path or not os.path.isfile(spec_path):
        raise SystemExit(
            f"the recording names goal region {spec_path!r}, which is not readable. The "
            "drawn region has to come from the spec the run actually used; a default would "
            "make this figure agree with itself whatever the recording did.")
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "..", "source", "robot_rl"))
    from robot_rl.tasks.manager_based.robot_rl.mdp.goal_region import GoalRegionSpec
    spec = GoalRegionSpec.load(spec_path)
    ok_uids, bad_uids = _pick(episodes, args.n, args.seed)

    plot_style.apply_paper_style()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = len(ok_uids) + len(bad_uids)
    fig, axes = plt.subplots(rows, 2, figsize=(11.0, 2.6 * rows), squeeze=False)
    C = plot_style.COLORS

    for r, (uid, good) in enumerate([(u, True) for u in ok_uids]
                                    + [(u, False) for u in bad_uids]):
        e = {k: v[episodes["episode_uid"] == uid][0] for k, v in episodes.items()}
        m = windows["episode_uid"] == uid
        order = np.argsort(windows["replan_index"][m])
        down = windows["goal_downrange_m"][m][order]
        lat = windows["goal_lateral_m"][m][order]
        t = windows["replan_index"][m][order] * 0.24
        colour = C["teal"] if good else C["red"]
        raw_fam = (fam_names[int(e["family_code"])]
                   if int(e["family_code"]) < len(fam_names) else "unknown")
        fam = ro._normalized([raw_fam])[0]
        # Derived from the SAME ordering the reward uses, never from `terminated` alone.
        # The two disagree: the derived fall predicate is a superset of the termination (its
        # tilt limb is the total angle from upright, where `base_fallen` bounds roll and
        # pitch separately), so an episode can cross the fall predicate, be scored a failure,
        # and still end by truncation. Labelling that "stalled" in the tool whose job is to
        # check the labels is exactly the wrong error.
        fell_first = (e["first_fall_step"] >= 0) and (
            e["first_goal_step"] < 0 or e["first_fall_step"] < e["first_goal_step"])
        verdict = ("arrived" if good else
                   ("fell before arriving" if fell_first and e["first_goal_step"] >= 0 else
                    "fell" if fell_first else "stalled"))

        # --- plan view, terrain frame ---------------------------------------------------
        ax = axes[r][0]
        # From the SPEC this run recorded under, and per family. Hardcoding the region's
        # size in the tool whose job is to check the region would make the drawing agree
        # with itself no matter what the recording did.
        tol = spec.for_family(fam)
        half, margin = float(tol.half_width_m), float(tol.approach_margin_m)
        ax.axvspan(-margin, max(down.max(), 2.0) + 2.0, color=C["cyan"], alpha=0.18, lw=0)
        ax.axvline(-margin, color=C["cyan"], lw=2.5)
        ax.axhline(half, color=C["cyan"], lw=1.5, ls=":")
        ax.axhline(-half, color=C["cyan"], lw=1.5, ls=":")
        ax.plot(down, lat, color=colour, lw=3, marker="o", ms=3.5, mew=0)
        ax.plot(down[0], lat[0], marker="s", ms=9, color=C["black"], mew=0)
        ax.plot(down[-1], lat[-1], marker="X", ms=12, color=colour, mew=0)
        ax.set_ylabel(r"lateral (m)")
        ax.set_ylim(-4.0, 4.0)
        if r == rows - 1:
            ax.set_xlabel(r"downrange to goal anchor (m)")
        ax.text(0.02, 0.90, rf"\#{int(uid)} {plot_style.tex_escape(fam)} -- {verdict}",
                transform=ax.transAxes, fontsize=13, va="top", color=colour)

        # --- downrange over time --------------------------------------------------------
        ax = axes[r][1]
        ax.axhspan(-margin, max(down.max(), 2.0) + 2.0, color=C["cyan"], alpha=0.18, lw=0)
        ax.axhline(-margin, color=C["cyan"], lw=2.5)
        ax.plot(t, down, color=colour, lw=3)
        if int(e["first_goal_step"]) >= 0:
            ax.axvline(float(e["first_goal_step"]) * 0.02, color=C["teal"], lw=2, ls="--")
        if int(e["first_fall_step"]) >= 0:
            ax.axvline(float(e["first_fall_step"]) * 0.02, color=C["red"], lw=2, ls="--")
        ax.set_ylabel(r"downrange (m)")
        if r == rows - 1:
            ax.set_xlabel(r"time (s)")
        ax.text(0.02, 0.90,
                rf"min dist {float(e['min_dist_to_goal_m']):.2f} m, "
                rf"{int(e['steps_survived'])} steps",
                transform=ax.transAxes, fontsize=13, va="top", color="0.3")

    fig.tight_layout()
    out_dir = args.out or os.path.dirname(os.path.abspath(stem))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rollout_spot_check")
    plot_style.save_figure(fig, path)
    print(f"wrote {path}.png")
    print("shaded = inside the goal region (a half-plane, so it runs off the right); "
          "dotted = the lateral band; square = first re-plan, X = last.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
