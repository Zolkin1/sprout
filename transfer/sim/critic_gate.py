# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does the Isaac-fitted failure critic still discriminate when read in MuJoCo?

The critic (``scripts/flow_matching/fit_value.py``) was fitted on Isaac rollouts and scored
against Isaac outcomes.  :mod:`mjeval.critic` makes it readable inside the MuJoCo loop; this
asks the only question that matters afterwards -- whether the number still means anything
there -- and it asks it against MuJoCo's OWN outcomes, so nothing is assumed to transfer.

Three gates, deliberately the same three ``value_gates`` applies in Isaac, so the numbers are
directly comparable to the fit's own (v2, held-out tiles: ``V_priv`` AUC 0.906 / ECE 0.039,
median lead 4.75 chunks):

1. **Discrimination** -- AUC of ``p_fail`` against the episode's real MuJoCo verdict, with a
   **label-shuffle control on the identical pipeline**.  The shuffle must land at 0.50.  If it
   does not, the estimator is broken and no other number here means anything -- which is the
   lesson of the abandoned Isaac best-of-K harness, whose control matched its signal.
2. **Lead time** -- how many re-plans before the fall ``p_fail`` first crosses 0.5.  A critic
   that only rises once recovery is impossible scores well on AUC and is useless for
   selection, and only this number separates those.
3. **Calibration** -- reliability of the predicted rate against the observed one.

**What a poor result here does and does not mean.**  With the unit and reproduction tests
passing (``tests/test_mjeval_critic.py``), a low MuJoCo AUC is a statement about the critic's
SIM2SIM TRANSFER, not about the port.  Separating those needs a matched-pose comparison
against Isaac, which this does not attempt.

The generator must be the one the critic was fitted over, or the conditioning it reads is a
different distribution; ``FailureCritic`` refuses a mismatched ``.pt`` by content hash, but the
ONNX graph is chosen here and is not checked against it -- so pass the matching export.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve, by the rank identity (no sklearn on this path).

    Args:
        score: Predicted values.
        label: Boolean outcomes; True is the positive class.

    Returns:
        AUC, or NaN when one class is absent -- which is a real possibility on a tile where
        every episode survives, and reporting 0.5 there would look like a measurement.
    """
    label = np.asarray(label, dtype=bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks within ties, or a critic that saturates at 1.000 (which this one does on
    # a fall) scores by array order rather than by value.
    s = np.asarray(score, dtype=np.float64)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run(args) -> int:
    """Roll ``--seeds`` episodes per tile, score every re-plan, and report the three gates."""
    import terrains  # noqa: PLC0415

    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    from fm_noise_attribution import build_arm  # noqa: PLC0415

    crit = C.FailureCritic(args.critic_ckpt, fm_pt=(args.critic_fm_pt or None),
                           device=args.device)
    rows: list[dict] = []
    for tile in [int(t) for t in args.tiles.split(",")]:
        course = terrains.isaac_tile_course(args.clip_lib, tile, speeds=(float(args.speed),),
                                            goal_anchor_max_offset=args.max_offset,
                                            isaac_matched=args.isaac_matched)
        course = dataclasses.replace(course, name=f"tile{tile}{args.scene_tag}")
        # The BUDGET and the VELOCITY are one choice, not two. Isaac's episode window is the
        # clip's own duration, and in Isaac the clip DRIVES the robot -- so the distance and
        # the time are the same fact and they imply a speed. Taking the clip's time while
        # commanding an unrelated 0.8 m/s times the robot against a reference it was never
        # asked to keep up with, which shows up as a timeout and reads as a policy failure.
        # So under --isaac_matched the command comes from the course, not the CLI.
        speed = float(args.speed)
        if args.isaac_matched:
            speed = float(course.params["clip_pace_mps"])
            print(f"[isaac_matched] tile {tile}: budget {course.budget_s:.1f} s, goal "
                  f"{course.target_xy[0]:.2f} m, commanded pace {speed:.2f} m/s "
                  f"(--speed {args.speed} ignored: it is not free once the budget is fixed)")
        cfg, policy, fm_lag, spawn, rig = build_arm(
            types.SimpleNamespace(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                                  out=args.out, providers=args.providers,
                                  fall_height=args.fall_height,
                                  fall_tilt_deg=args.fall_tilt_deg,
                                  cmd_hold_s=terrains.CMD_HOLD_S), course)
        print(f"[gate] {course.name} klass={course.klass} "
              f"anchor={np.round(course.goal_anchor_xyz, 3)}", flush=True)
        cmd = np.array([speed, 0.0, 0.0])
        for seed in range(args.seeds):
            scorer = C.CriticScorer(crit, course.goal_anchor_xyz, rig.robot)
            policy._rng = np.random.default_rng(seed)
            mjrunner._reset_episode_state(policy, fm_lag)
            scorer.reset()
            verdict, failure, _ = mjrunner.run_episode(
                rig, policy, fm_lag, spawn, cmd, None, steer_kp=cfg.steer_kp,
                steer_wz_max=cfg.steer_wz_max, cfg_hold=cfg.cmd_hold_s, probe=scorer)
            p = np.asarray(scorer.p_fail, dtype=np.float64)
            failed = verdict != "success"
            # Lead time: re-plans between the FIRST 0.5 crossing and the end. Only defined on
            # an episode that actually failed and actually crossed.
            lead = float("nan")
            if failed and p.size and (p >= 0.5).any():
                lead = float(p.size - int(np.argmax(p >= 0.5)))
            rows.append({"tile": tile, "seed": seed, "verdict": verdict, "failure": failure,
                         "failed": bool(failed), "n": int(p.size), "lead_replans": lead,
                         **scorer.summary(), "trace": [round(float(x), 4) for x in p]})
            print(f"  seed {seed:>2}: {verdict:8s} n={p.size:>3} "
                  f"max={p.max() if p.size else float('nan'):.3f} lead={lead}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "critic_gate.json"), "w") as f:
        json.dump({"rows": rows, "args": vars(args)}, f, indent=1)

    # ---- the three gates -----------------------------------------------------------------
    ok = [r for r in rows if r["n"] > 0]
    y = np.array([r["failed"] for r in ok])
    for stat in ("p_fail_max", "p_fail_mean"):
        s = np.array([r[stat] for r in ok])
        rng = np.random.default_rng(0)
        null = [auc(s, rng.permutation(y)) for _ in range(200)]
        print(f"\n[gate 1] AUC({stat}) = {auc(s, y):.3f}   "
              f"shuffled control {np.nanmean(null):.3f} +/- {np.nanstd(null):.3f}"
              f"   (must be ~0.50)")
    lead = np.array([r["lead_replans"] for r in ok if r["failed"]], dtype=np.float64)
    lead = lead[np.isfinite(lead)]
    if lead.size:
        print(f"[gate 2] lead time over {lead.size} falls: median {np.median(lead):.2f} "
              f"re-plans = {np.median(lead) * 12 * 0.02:.2f} s   "
              f"(Isaac held-out tiles: 4.75 chunks)")
    else:
        print("[gate 2] no failed episode crossed 0.5 -- nothing to select on")
    print(f"[gate 3] failures  max p_fail: {np.array([r['p_fail_max'] for r in ok if r['failed']]).mean():.3f}"
          f"   successes: {np.array([r['p_fail_max'] for r in ok if not r['failed']]).mean():.3f}")
    print(f"\n[gate] {len(ok)} episodes, {int(y.sum())} falls -> {args.out}/critic_gate.json")
    return 0


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--critic_ckpt", required=True)
    p.add_argument("--critic_fm_pt", default="")
    p.add_argument("--tracker_onnx", required=True)
    p.add_argument("--fm_onnx", required=True,
                   help="MUST be the export of the generator the critic was fitted over.")
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--tiles", default="6,20")
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--isaac_matched", action="store_true",
                   help="run each course under the ISAAC episode window -- clip_end budget, "
                        "the clip's own endpoint as the goal, and its approach margin. Use it "
                        "whenever the critic's reading has to be in-distribution: the default "
                        "mjeval budget runs 1.4-3.6x longer than any training episode. Moves "
                        "budget AND goal together; either alone is incoherent (a matched "
                        "budget with the default goal times out every tile-6 run).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--scene_tag", default="_gate",
                   help="REQUIRED to differ between concurrent runs on one tile: CourseRig "
                        "rewrites robots/g1/<robot>_<course>.xml before loading it.")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
