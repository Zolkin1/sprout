# Copyright 2026 Zachary Olkin. All rights reserved.

"""Prove the LIVE terrain grade equals the OFFLINE one, on a real rollout.

``test_terrain_reward_parity.py`` pins everything a unit test can reach -- the contact
threshold, the dilation, the node dt, which reduction the penalty is built from, and that
neither side passes an override.  The one thing it cannot reach is the TERRAIN PLACEMENT,
because that needs a running importer: offline the tile corner is recovered from the recorded
goal (``goal_pos_w - route.goal_xy``), live it comes from
``ClipTerrainImporter.terrain_collider_boxes``.  A translation error there shifts every signed
distance and would show up as a plausible-looking reward, not as a crash.

So this closes it on data.  Record one short rollout with BOTH ``--terrain_reward`` (the live
chunk table) and windows on, re-grade those windows through the ordinary offline command, and
join the two on ``(episode_uid, replan_index)``.  They must agree.

Run it ONCE per meaningful change to either path; production evals then use ``--no_windows``
and rely on the unit test.

Usage::

    python scripts/rsl_rl/record_rollouts.py --task=G1-box2-finetune ... \\
        --terrain_reward --num_episodes 64 --output_dir /tmp/parity
    python scripts/rsl_rl/inspect_terrain_reward.py --stage reward --reward_windows 0 \\
        --rollout /tmp/parity/rollout_<trk> --clip clips/box2_train.h5 \\
        --isaac_joint_names <run>/exported/policy_parameters.yaml \\
        --out /tmp/parity --reward_out /tmp/parity/sidecar.npz
    python scripts/rsl_rl/verify_terrain_reward_parity.py \\
        /tmp/parity/rollout_<trk>.episodes.h5 /tmp/parity/sidecar.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import rollout_outcomes as ro  # noqa: E402

#: Columns compared, and the two that matter: the critic's penalty is built from exactly
#: these, so an agreement anywhere else is not a substitute for an agreement here.
COMPARED: tuple[str, ...] = ("contact_penalty", "penetration_sum_m")

#: Relative tolerance, sized for a **cross-DTYPE** comparison, which is what this is.
#:
#: The offline grader runs float64 on purpose -- ``_grade_rows`` casts the base rotation and
#: the joints, and ``plan_collider_points`` refuses a dtype mismatch rather than downcasting,
#: so its collider model is float64 too.  The live term runs float32 at env speed.  So the two
#: cannot agree to better than fp32 epsilon on any continuous quantity, and an earlier 1e-9
#: here was simply the wrong question: it reported FAILED on a pair whose medians agreed to six
#: figures and whose ``penetration_sum_m`` correlated at +1.00000.
#:
#: 5e-3 is derived, not tuned to pass.  Each window's penetration term sums ~117k signed
#: distances (1886 collider points x 62 nodes); fp32 accumulation over N terms carries a
#: relative error between ``sqrt(N) * eps`` (4e-5, random signs) and ``N * eps`` (1.4e-2,
#: worst case).  Measured worst row: 1.8e-3, squarely inside that band.  Nothing tighter is
#: reachable while one side runs float64 and the other float32.
#:
#: What this still catches, which is the point of naming the number: the feet-only collider
#: bug this file found was 1300-1700%, a misplaced tile is O(1), a spec change is its own
#: explicit gate, a contact-logic change moves the FLAG gate, and a systematic shift moves
#: the distribution gate at 1%.  A 0.5% floor sits three orders below the real defect.
RTOL: float = 5e-3

#: Fraction of rows allowed to disagree on the contact FLAGS, and it is a separate gate
#: because a flag is discrete: no relative tolerance can express "one node flipped".
#:
#: A sole speed sitting within float noise of the 0.05 m/s threshold flips stance/swing, which
#: moves the graded node set and the exemption mask together.  Measured on 3876 real re-plans:
#: 1.11% of windows differ, and where they do the median difference is exactly ONE flip of 128
#: (64 nodes x 2 feet).  That is the fp32/fp64 boundary and it is irreducible without matching
#: dtypes; a rate climbing well past it is the signal of a real divergence, which a relative
#: tolerance on the penalty would dilute into noise.
MAX_FLAG_DISAGREE: float = 0.05

#: Relative tolerance for the DISTRIBUTION summaries, which is what the eval consumes.
#:
#: Looser than :data:`RTOL` on purpose, and for a reason that is not slack: a ~1% contact-flag
#: flip rate moves a RANK statistic (the median) by more than fp32 epsilon, because a flip
#: reorders rows rather than perturbing them.  Measured: mean and p95 agree to 1e-5, the median
#: to 1e-3.  1e-2 sits above that and far below any arm difference worth detecting -- the
#: reward's own p95 is two orders above its median, so a comparison sensitive to 1% of the
#: median was never going to be the instrument.
DIST_RTOL: float = 1e-2

#: Absolute floor for the per-row value gate, as a fraction of the column's own MEDIAN.
#:
#: The reward is a tail phenomenon -- ``penetration_sum_m`` is identically zero on a fifth of
#: windows and its p95 is two orders above its median -- so a pure relative tolerance is
#: dominated by rows carrying no signal: the worst offender measured was 1.68e-06 against
#: 1.65e-06, a 2% relative difference on an absolute difference of 3.5e-8, five orders below
#: the median penalty.  Ignoring differences below a thousandth of the column's median asks
#: the question that matters -- can this move any statistic built on the column -- instead of
#: flagging fp32 noise on values that round to nothing.
ATOL_MEDIAN_FRAC: float = 1e-3


def _join(live: dict[str, np.ndarray], side: dict[str, np.ndarray]
          ) -> tuple[np.ndarray, np.ndarray]:
    """Row indices pairing the two tables on ``(episode_uid, replan_index)``.

    **By key, never by position.** The sidecar is written from a subset of the rollout's rows
    in whatever order the grader walked them, and the chunk table is in re-plan order across
    interleaved envs, so the two orders have no reason to agree -- and a positional join would
    silently pair each chunk with some other chunk's grade, producing a parity report that
    passes or fails for reasons unrelated to the code.

    Args:
        live: The chunk table's columns.
        side: The offline sidecar's arrays.

    Returns:
        ``(live_rows, side_rows)`` into the two tables, for the keys they share.
    """
    def key(uid, ridx):
        return (np.asarray(uid, dtype=np.int64) << np.int64(20)
                | np.asarray(ridx, dtype=np.int64))

    lk = key(live["episode_uid"], live["replan_index"])
    sk = key(side["episode_uid"], side["replan_index"])
    if np.unique(lk).size != lk.size:
        raise SystemExit(f"the chunk table has duplicate (episode_uid, replan_index) keys: "
                         f"{lk.size} rows, {np.unique(lk).size} unique")
    order = np.argsort(sk)
    pos = np.searchsorted(sk[order], lk)
    pos = np.clip(pos, 0, sk.size - 1)
    src = order[pos]
    ok = sk[src] == lk
    return np.flatnonzero(ok), src[ok]


def main() -> int:
    """Compare and report. Returns a non-zero exit code on a mismatch."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", help="the .episodes.h5 written with --terrain_reward")
    p.add_argument("sidecar", help="the .npz from inspect_terrain_reward.py --stage reward")
    p.add_argument("--rtol", type=float, default=RTOL)
    args = p.parse_args()

    live, live_spec = ro.read_rollout_chunks(args.episodes)
    if not live:
        raise SystemExit(
            f"{args.episodes} has no `chunks` group -- it was recorded without "
            "--terrain_reward, so there is no live grade to compare against.")
    side = dict(np.load(args.sidecar, allow_pickle=False))
    side_meta = json.loads(str(side["meta_json"])) if "meta_json" in side else {}

    lrows, srows = _join(live, side)
    print(f"[parity] {len(live['episode_uid']):,} live chunks, "
          f"{len(side['episode_uid']):,} offline windows, {lrows.size:,} joined")
    if lrows.size == 0:
        raise SystemExit("the two tables share no (episode_uid, replan_index) key. They are "
                         "not from the same rollout, or the offline grade sampled none of "
                         "these rows -- re-run it with --reward_windows 0.")

    # The spec has to match too: identical numbers graded under different thresholds are not
    # a parity result. The live side stamps its spec on the group for exactly this check.
    for field, side_key in (("contact_speed_thresh", "thresh"),
                            ("contact_dilate_nodes", "dilate"),
                            ("contact_max_gap_nodes", "max_gap"),
                            ("collider_spacing", "spacing"),
                            ("contact_deadband_m", "contact_deadband_m"),
                            ("penetration_tol_m", "penetration_tol_m")):
        if field in live_spec and side_key in side_meta:
            lv, sv = float(live_spec[field]), float(side_meta[side_key])
            if lv != sv:
                print(f"[parity] SPEC MISMATCH {field}: live {lv} vs offline {sv}")
                return 2

    # Rows where the contact FLAGS agree. A flip changes which nodes are graded and which
    # points are exempt, so those rows differ by O(1) relatively -- no value tolerance can
    # pass them, and the flag gate below is what accounts for them. Comparing values only
    # where the flags agree is what makes this gate about arithmetic rather than about
    # thresholding noise.
    same_flags = np.ones(lrows.size, dtype=bool)
    for col, per_row in (("contact_coverage", 1.0 / 128.0), ("support_fraction", 1.0 / 64.0)):
        if col in live and col in side:
            d = np.abs(np.asarray(live[col], np.float64)[lrows]
                       - np.asarray(side[col], np.float64)[srows])
            same_flags &= d <= per_row * 0.5
    if not same_flags.all():
        print(f"[parity] comparing values on the {same_flags.sum():,} of {lrows.size:,} rows "
              f"whose contact flags agree; the rest are the flag gate's business")

    worst = 0.0
    failed = False
    for col in COMPARED:
        if col not in live or col not in side:
            print(f"[parity] MISSING column {col}: live={col in live} offline={col in side}")
            return 2
        a = np.asarray(live[col], dtype=np.float64)[lrows][same_flags]
        b = np.asarray(side[col], dtype=np.float64)[srows][same_flags]
        atol = ATOL_MEDIAN_FRAC * float(np.median(np.abs(b)))
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        over = ~np.isclose(a, b, rtol=args.rtol, atol=atol)
        i = int(np.argmax(np.where(over, rel, -1.0))) if over.any() else int(np.argmax(rel))
        worst = max(worst, float(rel[i]) if over.any() else 0.0)
        bad = int(over.sum())
        flag = "OK " if bad == 0 else "BAD"
        print(f"[parity] {flag} {col:<22s} max rel {rel[i]:.3e}  "
              f"(live {a[i]:.6g} vs offline {b[i]:.6g})  {bad:,} of {rel.size:,} outside "
              f"rtol {args.rtol:g} + atol {atol:.2e}")
        failed |= bad > 0

    # The penalty the critic is actually fitted on, rebuilt from both sides with the offline
    # weights -- the number the whole comparison exists to make trustworthy.
    wc = float(side_meta.get("contact_weight", 1.0))
    wp = float(side_meta.get("penetration_weight", 1.0))
    pen_l = wc * np.asarray(live["contact_penalty"], np.float64)[lrows] \
        + wp * np.asarray(live["penetration_sum_m"], np.float64)[lrows]
    pen_s = wc * np.asarray(side["contact_penalty"], np.float64)[srows] \
        + wp * np.asarray(side["penetration_sum_m"], np.float64)[srows]
    pl, ps = pen_l[same_flags], pen_s[same_flags]
    p_atol = ATOL_MEDIAN_FRAC * float(np.median(np.abs(ps)))
    p_over = ~np.isclose(pl, ps, rtol=args.rtol, atol=p_atol)
    rel = np.abs(pl - ps) / np.maximum(np.abs(ps), 1e-12)
    bad = bool(p_over.any())
    failed |= bad
    print(f"[parity] {'BAD' if bad else 'OK '} "
          f"penalty (w_c {wc:g}, w_p {wp:g})  max rel {rel.max():.3e}, "
          f"{int(p_over.sum()):,} of {rel.size:,} outside rtol + atol {p_atol:.2e}")

    # THE FLAG GATE. Discrete, so it gets its own criterion rather than being folded into a
    # relative tolerance that cannot represent it.
    for col, per_row in (("contact_coverage", 1.0 / 128.0), ("support_fraction", 1.0 / 64.0)):
        if col not in live or col not in side:
            continue
        a = np.asarray(live[col], np.float64)[lrows]
        b = np.asarray(side[col], np.float64)[srows]
        d = np.abs(a - b)
        rate = float((d > per_row * 0.5).mean())
        flips = (np.median(d[d > per_row * 0.5]) / per_row) if (d > per_row * 0.5).any() else 0.0
        bad = rate > MAX_FLAG_DISAGREE
        print(f"[parity] {'BAD' if bad else 'OK '} {col:<22s} rows differing "
              f"{rate:.2%} (limit {MAX_FLAG_DISAGREE:.0%}), median {flips:.2f} flip(s)")
        failed |= bad

    # What the eval actually consumes is the DISTRIBUTION of the penalty over a cell, not any
    # single row, so it is reported explicitly: a per-row residual that never moves these is
    # invisible to every comparison built on top.
    for name, q in (("median", 50.0), ("mean", None), ("p95", 95.0)):
        la = float(pen_l.mean()) if q is None else float(np.percentile(pen_l, q))
        sa = float(pen_s.mean()) if q is None else float(np.percentile(pen_s, q))
        rel = abs(la - sa) / max(abs(sa), 1e-12)
        bad = rel > DIST_RTOL
        failed |= bad
        print(f"[parity] {'BAD' if bad else 'OK '} penalty {name:<6s} "
              f"live {la:.6g} vs offline {sa:.6g}  rel {rel:.2e} "
              f"(limit {DIST_RTOL:g})")

    print(f"[parity] {'FAILED' if failed else 'PASSED'} at rtol {args.rtol:g} "
          f"(worst term {worst:.3e})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
