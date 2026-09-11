# Copyright 2026 Zachary Olkin. All rights reserved.

"""Look at what the generator ACTUALLY proposes at one moment, and what each proposal does.

The aggregate numbers say the outcome is dominated by which noise draw you got, that failure
needs a conjunction of draws, and that a chunk-level label is weak.  None of them show you a
single decision.  This does: freeze one state, draw ``K`` candidate chunks from the generator,
and roll each one to the end of the episode so every candidate carries a REAL outcome rather
than an estimate.

Two stages, because the viewer must not be in the physics loop:

* ``record`` -- headless.  Runs one course, and at every ``--every`` re-plan branches ``K``
  candidates, each rolled to termination.  Writes ``<out>/samples.npz``.
* ``view`` -- viser, imports no MuJoCo and no onnxruntime.  Step between re-plans with the
  keyboard and see the candidates.

**Why capture-then-replay rather than a live viewer.**  ``mjeval/view_cell.py`` measured that
attaching the MuJoCo viewer perturbs ``mjData`` enough to change the trajectory -- one course
survived 16 s headless and fell reproducibly at 6.04 s with the viewer attached.  A live
stepper would therefore show a rollout that is not the one whose candidates were scored, which
is the failure mode this whole file exists to avoid.

**A branch RESTORES NOTHING; it replays.**  Each candidate re-runs the episode from spawn with
the reference's own recorded noise draws for the first ``pick`` chunks and one fresh draw at
``pick``.  MuJoCo is deterministic and the tape fixes the only stochastic input, so the branch
arrives at exactly the reference state -- no state copy, and therefore none of the
contact-manifold / friction-anchor / solver-warm-start restore problem that abandoned the Isaac
harness (see CLAUDE.md).  It costs a full re-roll per candidate, which is the price of not
needing a restore to be exact.

**Why the outcomes are ground truth rather than a critic reading.**  Every candidate is rolled
to termination, so its recorded outcome IS what that chunk leads to, not a prediction of it.
That is the stronger quantity, and it is also what a critic would have to be checked against:
colouring by an estimate would beg the question the tool is meant to answer.

The critic's ``P(fail)`` is deliberately NOT recorded here.  Both variants take three
goal-offset features that are Isaac goal-region quantities, and the mjeval courses define
arrival by a downrange target line, so there is no faithful mapping.  See
``TODO(attribution-critic-trace)`` in ``fm_noise_attribution.py``; a goal-free critic would
unblock both at once.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

# Same bootstrap fm_variance_probe.py uses: mjeval's modules import each other by bare name.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, "mjeval"), os.path.dirname(_HERE),
           os.path.dirname(os.path.dirname(_HERE))):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def roll_capture(rig, policy, fm_lag, spawn, cmd, cfg, tape, inject=None, seed: int = 0,
                 want_window_at: int | None = None) -> dict:
    """One episode, additionally capturing the WINDOW the generator emitted at one re-plan.

    ``roll`` records the base trajectory; the candidate CHUNK is what a PCA of "what did the
    generator propose" has to live in, and it is not reconstructible from the trajectory.

    Captured from ``policy._pred``, watching ``tape.idx`` for the re-plan boundary: the noise
    is drawn BEFORE ``sample_window`` writes ``_pred``, so the window belonging to draw ``i``
    is only readable on a control step after that draw. Reading it at draw time would return
    the PREVIOUS re-plan's window, off by one and entirely plausible-looking.

    Args:
        want_window_at: Re-plan index whose emitted window to keep, or ``None`` for none.

    Returns:
        ``roll``'s dict plus ``window`` -- ``(F, D)`` or ``None``.
    """
    import runner as mjrunner

    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)
    tape.reset(inject)
    traj: list[np.ndarray] = []
    grabbed = {"window": None, "qpos": None}

    def _rec(robot, pol):
        traj.append(np.asarray(robot.mj_data.qpos[:3], dtype=np.float64).copy())
        if (want_window_at is not None and grabbed["window"] is None
                and tape.idx == want_window_at + 1):
            grabbed["window"] = np.asarray(pol._pred[0], dtype=np.float32).copy()
            # The FULL pose at the branch instant, so the viewer can stand a robot exactly
            # where the candidates diverge. qpos is [base_pos(3), base_quat_wxyz(4), joints].
            grabbed["qpos"] = np.asarray(robot.mj_data.qpos, dtype=np.float32).copy()

    verdict, failure, acc = mjrunner.run_episode(
        rig, policy, fm_lag, spawn, cmd, None, recorder=_rec,
        steer_kp=cfg.steer_kp, steer_wz_max=cfg.steer_wz_max, cfg_hold=cfg.cmd_hold_s)
    row = acc.row(episode_id=seed, terrain_idx=0, speed_idx=0, arm=0, seed=seed,
                  verdict=verdict, failure_code=failure)
    return {"verdict": verdict, "failure": failure, "row": row,
            "tape": list(tape.recorded), "traj": np.array(traj),
            "window": grabbed["window"], "qpos": grabbed["qpos"],
            "n_fresh": tape.n_fresh}


def stage_record(args) -> int:
    """Roll one course, branching K candidates at every ``--every`` re-plan."""
    import mujoco  # noqa: F401  (imported for its side effect on the model handle)
    import runner as mjrunner
    import terrains
    from fm_noise_attribution import NoiseTape, build_arm, _course

    course = _course(terrains, args.clip_lib, int(args.tile), float(args.speed), args.scene_tag)
    cfg, policy, fm_lag, spawn, rig = build_arm(args, course)
    tape = NoiseTape(policy)
    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)
    print(f"[explore] {course.name}: {course.note}")

    # ---- the REFERENCE rollout, taped so every branch starts from a known trajectory -------
    ref = roll_capture(rig, policy, fm_lag, spawn, cmd, cfg, tape, seed=args.seed)
    n_chunks = len(ref["tape"])
    print(f"[explore] reference: {ref['verdict']}/{ref['failure']}  {n_chunks} chunks, "
          f"{len(ref['traj'])} steps")

    # Which re-plans to branch at.
    picks = list(range(0, n_chunks, max(1, int(args.every))))
    print(f"[explore] branching {len(picks)} re-plan(s) x {args.k} candidates = "
          f"{len(picks) * args.k} rollouts")

    records = []
    for pick in picks:
        for k in range(int(args.k)):
            # The prefix is the reference's own draws, so the branch starts from EXACTLY the
            # reference state at `pick`; only the draw at `pick` differs, and every later
            # chunk is re-inferred normally.
            inj = [ref["tape"][j] for j in range(pick)]
            rng = np.random.default_rng(700000 + 1000 * pick + k)
            inj.append(rng.standard_normal(ref["tape"][0].shape))
            r = roll_capture(rig, policy, fm_lag, spawn, cmd, cfg, tape,
                             inject=inj, seed=800000 + 1000 * pick + k,
                             want_window_at=pick)
            # The candidate chunk is the window the generator emitted from that draw. Recorded
            # in the generator's own window space, which is what a PCA of "what did it propose"
            # has to live in.
            records.append({
                "replan": pick, "k": k,
                "verdict": r["verdict"], "failure": r["failure"],
                "steps": int(len(r["traj"])),
                "progress_x_m": float(r["row"]["progress_x_m"]),
                "traj": r["traj"].astype(np.float32),
                "window": r["window"],
                "qpos": r["qpos"],
            })
        done = sum(1 for x in records if x["replan"] == pick and x["verdict"] != "success")
        print(f"[explore] replan {pick:>3}: {args.k - done}/{args.k} candidates survive",
              flush=True)
    tape.detach()

    os.makedirs(args.out, exist_ok=True)
    # Ragged trajectories: stored flat with an index, so the viewer needs no padding rule.
    trajs = [r.pop("traj") for r in records]
    lens = np.array([len(t) for t in trajs], dtype=np.int32)
    wins = [r.pop("window") for r in records]
    n_missing = sum(1 for w in wins if w is None)
    if n_missing:
        # A branch that terminated inside its own re-plan never reached a control step where
        # the window was readable. Refusing beats silently zero-filling, which would put a
        # fabricated chunk into the PCA looking exactly like a real one.
        raise SystemExit(
            f"{n_missing} of {len(wins)} branches never exposed their emitted window (the "
            "episode ended within the re-plan). Lower --every or pick a longer reference.")
    windows = np.stack(wins).astype(np.float32)
    qp = [r.pop("qpos") for r in records]
    if any(q is None for q in qp):
        raise SystemExit("a branch exposed a window but no qpos; the two are captured on the "
                         "same control step, so this cannot happen -- investigate.")
    qpos = np.stack(qp).astype(np.float32)
    np.savez_compressed(
        os.path.join(args.out, "samples.npz"),
        ref_traj=ref["traj"].astype(np.float32),
        ref_tape=np.stack(ref["tape"]).astype(np.float32),
        traj_flat=np.concatenate(trajs, axis=0) if trajs else np.zeros((0, 3), np.float32),
        traj_len=lens,
        windows=windows,
        qpos=qpos,
        replan=np.array([r["replan"] for r in records], np.int32),
        k=np.array([r["k"] for r in records], np.int32),
        failed=np.array([r["verdict"] != "success" for r in records], bool),
        steps=np.array([r["steps"] for r in records], np.int32),
        progress=np.array([r["progress_x_m"] for r in records], np.float32),
        meta=json.dumps({
            "course": course.name, "tile": int(args.tile), "speed": float(args.speed),
            "clip_lib": args.clip_lib, "every": int(args.every), "k": int(args.k),
            "n_chunks": n_chunks,
            "fm_update_every": int(getattr(policy, "fm_update_every", 12)),
            "ref_verdict": ref["verdict"], "ref_failure": ref["failure"],
            "tracker_onnx": args.tracker_onnx, "fm_onnx": args.fm_onnx,
        }),
    )
    print(f"[explore] wrote {args.out}/samples.npz")
    return 0


def main() -> int:
    """Entry point."""
    import terrains  # noqa: PLC0415  the shared CMD_HOLD_S default
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("record", "view"))
    p.add_argument("--out", required=True)
    p.add_argument("--tracker_onnx", default="")
    p.add_argument("--fm_onnx", default="")
    p.add_argument("--clip_lib", default="clips/g1_mb_v5_body_mode11_decor.h5")
    p.add_argument("--tile", type=int, default=6)
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0, help="which reference rollout to explore.")
    p.add_argument("--k", type=int, default=12, help="candidate chunks per re-plan.")
    p.add_argument("--every", type=int, default=3, help="branch every Nth re-plan.")
    p.add_argument("--scene_tag", default="_expl",
                   help="REQUIRED to differ between concurrent runs on one tile: CourseRig "
                        "rewrites robots/g1/<robot>_<course>.xml before loading it.")
    p.add_argument("--port", type=int, default=8080, help="viser port (view stage).")
    p.add_argument("--cmd_hold_s", type=float, default=terrains.CMD_HOLD_S)
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--providers", default="auto")
    args = p.parse_args()
    if args.stage == "record":
        return stage_record(args)
    from fm_sample_explorer_view import stage_view  # noqa: PLC0415
    return stage_view(args)


if __name__ == "__main__":
    raise SystemExit(main())
