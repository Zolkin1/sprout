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

#: Relative tolerance.  The two sides tile the SDF pass differently -- the live term by node
#: count, the offline one by a point budget -- so the same per-point depths are summed in a
#: different order.  Measured at ~1 ULP on a synthetic plan; 1e-9 leaves four orders of slack
#: for fp32 accumulation over a real plan while still failing any real disagreement, which
#: would be O(1) rather than O(1e-9).
RTOL: float = 1e-9


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

    worst = 0.0
    failed = False
    for col in COMPARED:
        if col not in live or col not in side:
            print(f"[parity] MISSING column {col}: live={col in live} offline={col in side}")
            return 2
        a = np.asarray(live[col], dtype=np.float64)[lrows]
        b = np.asarray(side[col], dtype=np.float64)[srows]
        denom = np.maximum(np.abs(b), 1e-12)
        rel = np.abs(a - b) / denom
        i = int(np.argmax(rel))
        worst = max(worst, float(rel[i]))
        bad = int((rel > args.rtol).sum())
        flag = "OK " if bad == 0 else "BAD"
        print(f"[parity] {flag} {col:<22s} max rel {rel[i]:.3e}  "
              f"(live {a[i]:.6g} vs offline {b[i]:.6g})  {bad:,} of {rel.size:,} over rtol")
        failed |= bad > 0

    # The penalty the critic is actually fitted on, rebuilt from both sides with the offline
    # weights -- the number the whole comparison exists to make trustworthy.
    wc = float(side_meta.get("contact_weight", 1.0))
    wp = float(side_meta.get("penetration_weight", 1.0))
    pen_l = wc * np.asarray(live["contact_penalty"], np.float64)[lrows] \
        + wp * np.asarray(live["penetration_sum_m"], np.float64)[lrows]
    pen_s = wc * np.asarray(side["contact_penalty"], np.float64)[srows] \
        + wp * np.asarray(side["penetration_sum_m"], np.float64)[srows]
    rel = np.abs(pen_l - pen_s) / np.maximum(np.abs(pen_s), 1e-12)
    print(f"[parity] {'OK ' if rel.max() <= args.rtol else 'BAD'} "
          f"penalty (w_c {wc:g}, w_p {wp:g})  max rel {rel.max():.3e}  "
          f"median live {np.median(pen_l):.5f} vs offline {np.median(pen_s):.5f}")
    failed |= bool(rel.max() > args.rtol)

    print(f"[parity] {'FAILED' if failed else 'PASSED'} at rtol {args.rtol:g} "
          f"(worst term {worst:.3e})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
