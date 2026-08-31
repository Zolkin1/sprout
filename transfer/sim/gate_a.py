# Copyright 2026 Zachary Olkin. All rights reserved.

"""Gate A: is there anything for best-of-K to SELECT, and where?

The generator's own sampling variance decides ~1 in 5 outcomes (``fm_variance_probe.py``), so
picking a better draw is worth something in principle.  This measures how much, and it is the
gate that decides whether the Q-critic / best-of-K pipeline is worth building at all.

**The question is asked per STATE, not per episode.**  From a frozen state, draw ``K``
candidate chunks and roll each to TERMINATION, so every candidate carries a real outcome rather
than an estimate.  Then compare:

* ``random`` -- the mean outcome over the K candidates, i.e. what the deployed policy gets
  today by drawing one of them;
* ``oracle`` -- 1.0 if ANY candidate survives, i.e. the ceiling a perfect selector reaches.

``oracle - random`` is the headroom.  A selector can only ever recover part of it.

**Harvesting is the whole game and it is why the critic had to be ported first.**  Branching
uniformly wastes most of the budget: on a recorded episode 8 of 10 branch states had zero
headroom because the failure was already committed, and one had none because it was already
safe.  So states are harvested by the critic's own ``p_fail`` band -- the region where the
outcome is genuinely undecided.  ``--bands`` reports the headroom per band, which is also the
answer to "where should a selector spend its samples".

**A branch RESTORES NOTHING.**  It replays from spawn with the reference's own recorded draws
for the first ``pick`` chunks and one fresh draw at ``pick``.  MuJoCo is deterministic and the
tape fixes the only stochastic input, so the branch arrives at exactly the reference state --
no contact manifolds, no friction anchors, no solver warm start, none of what made the Isaac
version of this measurement unusable.  It costs a full re-roll per candidate, which is the
price of not needing a restore to be exact.

**The blocking control is replay determinism**, and it runs before anything is reported: a
branch handed the reference's OWN draw at ``pick`` must reproduce the reference outcome
exactly.  If it does not, the tape is not capturing every stochastic input and the whole
measurement is meaningless -- which is the failure mode that killed the Isaac harness, whose
same-noise control matched its real signal.
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


def _roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, mjrunner, *, inject=None, seed=0,
          scorer=None):
    """One episode with the noise taped, optionally critic-scored.

    Args:
        rig, policy, fm_lag, spawn, cmd, cfg: The loaded arm.
        tape: The :class:`fm_noise_attribution.NoiseTape` wrapping ``policy``.
        mjrunner: The ``runner`` module.
        inject: Draws to replay for the leading chunks, or ``None`` to draw fresh.
        seed: The policy RNG seed for anything past the injected prefix.
        scorer: A :class:`critic.CriticScorer`, or ``None``.

    Returns:
        ``(verdict, failure, taped draws, p_fail list)``.
    """
    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)
    tape.reset(inject)
    if scorer is not None:
        scorer.reset()
    verdict, failure, _ = mjrunner.run_episode(
        rig, policy, fm_lag, spawn, cmd, None, steer_kp=cfg.steer_kp,
        steer_wz_max=cfg.steer_wz_max, cfg_hold=cfg.cmd_hold_s,
        probe=scorer)
    return verdict, failure, list(tape.recorded), (list(scorer.p_fail) if scorer else [])


def select_clips(clip_h5: str, tiles, per_tile: int) -> list[int]:
    """Clips SPREAD across each tile's commanded-speed range, not its first N.

    A tile's clips are laid out in ascending command: taking the head of the list gives the
    slowest ones (0.60-0.64 m/s against a range to 2.60), which is an easy slice and reports a
    baseline the deployed policy never sees.  Speed is also the single biggest driver of
    failure measured on this pair -- ``fm_variance_probe`` puts stairs_up at 0.798 success at
    0.8 m/s and 0.624 at 1.6 -- so a selection biased in speed is biased in exactly the
    variable Gate A is about.

    Args:
        clip_h5: The clip library.
        tiles: Terrain ids to draw from.
        per_tile: How many clips per tile.

    Returns:
        Clip ids, ascending.
    """
    import h5py  # noqa: PLC0415

    out: list[int] = []
    with h5py.File(clip_h5, "r") as f:
        tid = f["terrain_id"][:]
        for t in tiles:
            rows = np.flatnonzero(tid == int(t))
            if rows.size == 0:
                raise ValueError(f"no clips on terrain_id {t}")
            v = np.array([f["target_vel"][int(c), 0, 0] for c in rows])
            order = rows[np.argsort(v)]
            # Even quantiles of the command distribution, endpoints included.
            idx = np.unique(np.linspace(0, order.size - 1, min(per_tile, order.size)).round()
                            .astype(int))
            out.extend(int(c) for c in order[idx])
    return sorted(out)


def run(args) -> int:
    """Reference pass, harvest, branch, report."""
    import terrains  # noqa: PLC0415

    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    from fm_noise_attribution import NoiseTape, build_arm  # noqa: PLC0415

    crit = C.FailureCritic(args.critic_ckpt, device=args.device)
    lo, hi = args.band
    rows, refs = [], []
    tiles = [int(t) for t in args.tiles.split(",")]
    clips = select_clips(args.clip_lib, tiles, int(args.clips_per_tile))
    print(f"[gateA] {len(clips)} clips over {len(tiles)} tiles: {clips}", flush=True)
    for clip_id in clips:
        # One course per CLIP: its terrain, start, goal, duration and command. Nothing is
        # aggregated over the tile -- a `max` duration beside a `median` endpoint describes a
        # trajectory no clip ever ran, and measured 0 successes in 6 reference episodes.
        course = terrains.isaac_clip_course(args.clip_lib, clip_id,
                                            max_offset=args.max_offset)
        course = dataclasses.replace(course, name=f"clip{clip_id}{args.scene_tag}")
        tile = int(course.params["terrain_id"])
        # The command is the clip's OWN `target_vel`, the signal Isaac's teacher was driven
        # by -- not a pace inferred from distance/duration, which is a different and much
        # smaller number (0.86 m/s inferred against a 1.60 m/s real command on tile 6) because
        # the clip climbs and turns and its ground speed is far below what it was told to do.
        if not course.params["clip_constant_command"]:
            raise SystemExit(
                f"clip {clip_id}'s target_vel varies within the clip, and `run_episode` holds "
                "one command for the episode. ~6% of clips are like this; either exclude them "
                "explicitly or add the per-step command path -- do not silently command the "
                "first row, which would run a different experiment than the clip describes.")
        cfg, policy, fm_lag, spawn, rig = build_arm(
            types.SimpleNamespace(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                                  out=args.out, providers=args.providers,
                                  fall_height=args.fall_height,
                                  fall_tilt_deg=args.fall_tilt_deg, cmd_hold_s=0.0), course)
        # Isaac starts the robot ON the clip -- pose AND velocity. mjeval's default standing
        # spawn is a different initial condition, and on stairs it is the difference between
        # arriving in phase with the treads and arriving from a standstill.
        spawn = mjrunner.clip_spawn(course, policy.get_joint_names())
        rig = mjrunner.CourseRig(course, policy, cfg, spawn)
        tape = NoiseTape(policy)
        scorer = C.CriticScorer(crit, course.goal_anchor_xyz, rig.robot)
        cmd = np.array(course.params["clip_command"][0], dtype=np.float64)
        print(f"\n[gateA] {course.name} tile {tile} {course.klass} cmd {cmd[0]:.2f} m/s "
              f"budget {course.budget_s:.2f}s dist {course.course_length_m:.2f}m", flush=True)

        # ---- reference pass: roll, score, keep the tape ----------------------------------
        for seed in range(args.seeds):
            v, f, tp, p = _roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, mjrunner,
                                seed=seed, scorer=scorer)
            refs.append({"tile": tile, "clip_id": clip_id, "seed": seed,
                         "verdict": v, "p": p, "n": len(tp), "cmd": float(cmd[0])})
            # Harvest: re-plans the critic says are genuinely undecided. `k < len(tp)` because
            # a re-plan with no taped draw after it cannot be branched.
            picks = [k for k, q in enumerate(p) if lo <= q <= hi and k < len(tp)]
            if args.max_states_per_episode:
                picks = picks[:args.max_states_per_episode]
            print(f"  seed {seed:>2}: {v:8s} n={len(tp):>3} "
                  f"in-band {len(picks)}/{len(p)}", flush=True)

            # BLOCKING control, once per reference episode: replaying the WHOLE tape must
            # reproduce the outcome exactly. It has to be the whole tape -- injecting only a
            # prefix and letting the tail draw fresh diverges BY CONSTRUCTION, which is not a
            # determinism failure but a different experiment, and reading it as one is how
            # this check first fired against itself. A DIFFERENT rng seed is used so that
            # anything stochastic OUTSIDE the tape would separate the two runs; the same seed
            # would hide exactly that.
            ctl_v, _, _, _ = _roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, mjrunner,
                                   inject=tp, seed=90000 + seed)
            if ctl_v != v:
                raise SystemExit(
                    f"REPLAY CONTROL FAILED at tile {tile} seed {seed}: replaying the full "
                    f"tape under a different rng seed gave {ctl_v!r}, not {v!r}. The tape is "
                    "not capturing every stochastic input, so a 'candidate' differs from the "
                    "reference by more than its draw and every headroom number would be "
                    "measuring the harness rather than the generator.")

            for pick in picks:
                outs, boots, trunc = [], [], []
                for k in range(args.k):
                    rng = np.random.default_rng(700000 + 1000 * pick + k)
                    inj = [tp[j] for j in range(pick)] + [rng.standard_normal(tp[0].shape)]
                    # The scorer rides the branch too, so each candidate carries BOTH labels:
                    # its true rolled-to-termination outcome and the critic's bootstrapped
                    # V(s_H). They are measured on the SAME branch, which is the only way the
                    # comparison says anything -- a bootstrap evaluated on different rollouts
                    # would confound the label with the trajectory.
                    bv, bf, _, bp = _roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, mjrunner,
                                          inject=inj, seed=800000 + 1000 * pick + k,
                                          scorer=scorer)
                    outs.append(bv == "success")
                    h = pick + int(args.bootstrap_replans)
                    # A branch that ended before the horizon has no V there. Fall back to its
                    # last reading and FLAG it, rather than substituting the true outcome --
                    # that would leak the answer into the label being tested.
                    trunc.append(h >= len(bp))
                    boots.append(float(bp[min(h, len(bp) - 1)]) if bp else float("nan"))
                # What a V-based selector would actually get: pick the branch the critic
                # likes best, then read that branch's TRUE outcome. Sits between `random`
                # (no selection) and `oracle` (perfect selection), and the gap to `oracle` is
                # what the critic's error costs.
                bo = np.asarray(boots, dtype=np.float64)
                v_sel = (float(outs[int(np.nanargmin(bo))]) if np.isfinite(bo).any()
                         else float("nan"))
                rows.append({"tile": tile, "clip_id": clip_id, "seed": seed, "pick": pick,
                             "p_fail": float(p[pick]), "ref_verdict": v,
                             "survived": [bool(o) for o in outs],
                             "boot_v": [float(b) for b in boots],
                             "boot_truncated": [bool(t) for t in trunc],
                             "v_selector": v_sel,
                             "random": float(np.mean(outs)), "oracle": float(any(outs))})
                print(f"      pick {pick:>3} p_fail={p[pick]:.3f}  "
                      f"{sum(outs)}/{args.k} survive  random={np.mean(outs):.3f} "
                      f"v_sel={v_sel:.3f} oracle={float(any(outs)):.3f}"
                      f"{'  (some branches truncated)' if any(trunc) else ''}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate_a.json"), "w") as f:
        json.dump({"rows": rows, "refs": refs, "args": {**vars(args), "band": list(args.band)}},
                  f, indent=1)
    _report(rows, args)
    print(f"\n[gateA] {len(rows)} states -> {args.out}/gate_a.json")
    return 0


def _report(rows: list[dict], args) -> None:
    """Headroom pooled and by ``p_fail`` band."""
    if not rows:
        print("\n[gateA] NO states were harvested. Either the band is wrong for this critic "
              "or these episodes are all decided -- widen --band before concluding anything.")
        return
    rnd = np.array([r["random"] for r in rows])
    orc = np.array([r["oracle"] for r in rows])
    vsl = np.array([r.get("v_selector", np.nan) for r in rows], dtype=np.float64)
    print(f"\n[gateA] pooled over {len(rows)} harvested states, K={args.k}")
    print(f"  random pick (deployed today) : {rnd.mean():.3f}")
    print(f"  V-selector (this critic)     : {np.nanmean(vsl):.3f}")
    print(f"  oracle best-of-K (ceiling)   : {orc.mean():.3f}")
    print(f"  HEADROOM (oracle - random)   : {orc.mean() - rnd.mean():+.3f}")
    print(f"  REALISED  (V-sel  - random)  : {np.nanmean(vsl) - rnd.mean():+.3f}"
          f"   = {100 * (np.nanmean(vsl) - rnd.mean()) / max(orc.mean() - rnd.mean(), 1e-9):.0f}%"
          " of the ceiling")
    n_tr = sum(any(r.get("boot_truncated", [])) for r in rows)
    if n_tr:
        print(f"  ({n_tr}/{len(rows)} states had at least one branch end before the "
              f"{args.bootstrap_replans}-re-plan horizon; those fall back to the last reading)")
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    print(f"\n  {'p_fail band':>14} {'n':>4} {'random':>8} {'V-sel':>8} {'oracle':>8} "
          f"{'headroom':>9} {'realised':>9}")
    p = np.array([r["p_fail"] for r in rows])
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if not m.any():
            continue
        print(f"  {f'[{a:.1f}, {b:.1f})':>14} {int(m.sum()):>4} {rnd[m].mean():>8.3f} "
              f"{np.nanmean(vsl[m]):>8.3f} {orc[m].mean():>8.3f} "
              f"{orc[m].mean() - rnd[m].mean():>+9.3f} "
              f"{np.nanmean(vsl[m]) - rnd[m].mean():>+9.3f}")
    print("\n  Read the BAND table, not the pooled row: a state that is already safe or "
          "already\n  committed has no headroom by construction, so pooling mixes the "
          "question with its\n  own sampling distribution.")


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--critic_ckpt", required=True)
    p.add_argument("--tracker_onnx", required=True)
    p.add_argument("--fm_onnx", required=True,
                   help="MUST be the export of the generator the critic was fitted over.")
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--tiles", default="6,20,40")
    p.add_argument("--clips_per_tile", type=int, default=4,
                   help="clips drawn per tile, SPREAD across its commanded-speed range.")
    p.add_argument("--seeds", type=int, default=3, help="reference episodes per CLIP.")
    p.add_argument("--k", type=int, default=8, help="candidates per harvested state.")
    p.add_argument("--band", type=float, nargs=2, default=(0.2, 0.8),
                   metavar=("LO", "HI"),
                   help="harvest re-plans whose p_fail lies here -- the undecided region.")
    p.add_argument("--bootstrap_replans", type=int, default=8,
                   help="horizon for the bootstrapped label, in re-plans (8 = 1.92 s at "
                        "K=12). Each branch is ALSO scored by the critic this far past the "
                        "branch point, so the true and bootstrapped labels are compared on "
                        "identical branches -- which is what says whether the cheap label "
                        "can drive the real selector.")
    p.add_argument("--max_states_per_episode", type=int, default=3,
                   help="cap, so one long episode cannot dominate the budget. 0 = no cap.")
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--scene_tag", default="_gatea")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
