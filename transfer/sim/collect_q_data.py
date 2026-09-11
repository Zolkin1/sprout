# Copyright 2026 Zachary Olkin. All rights reserved.

"""Collect ``(state, chunk, outcome)`` for fitting a Q-critic over the generator's samples.

The failure critic is ``V(s)``: it scores a STATE.  Best-of-K needs ``Q(s, a)`` -- it has to
rank the K chunks the generator drew from one state, which is a question about the ACTION.
Gate A measured that ranking is worth having (headroom +0.116 pooled, +0.472 in the
``V(s)`` band where outcomes are genuinely undecided, and a 2 s bootstrapped label already
recovers half of it).  This collects what a Q-head would be fitted on.

**Two streams, because they answer different halves and cost differently.**

* ``onpolicy`` -- every re-plan of an ordinary rollout: the conditioning, the chunk the
  generator actually served, and the episode's eventual outcome.  ~20 samples per episode, so
  it is cheap and dense.  On its own it is **not enough**: it holds exactly one action per
  state, so nothing in it distinguishes ``Q(s, a)`` from ``V(s)`` and a head fitted on it can
  score well while ranking chunks at a fixed state arbitrarily.
* ``branch`` -- K chunks drawn from ONE state, each rolled to its own outcome.  This is the
  only stream that contains the comparison the selector has to make.  It costs a full episode
  per sample, because a branch replays from spawn rather than restoring state.

So the ratio between them is the real budget decision, not an implementation detail:
``--branch_every`` sets how many re-plans are branched, and the run reports what it actually
collected of each.

**The label is the episode's outcome, and it is Monte-Carlo, not bootstrapped.**  A
bootstrapped label needs a critic, and using the critic being replaced to label its own
replacement is circular in the direction that hides failure.  ``V(s_H)`` is recorded ALONGSIDE
so the cheap label can be fitted and compared later without re-collecting -- which is how Gate
A established the 2 s label is worth 50% of the ceiling.

**Unattended-run discipline.**  Disk is the binding constraint on this box (211 GB free at 88%
used when this was written), so the run stops on whichever of ``--max_hours``,
``--max_samples`` or ``--min_free_gb`` binds first, and says which.  Datasets are resizable and
flushed every ``--flush_every`` samples, so a kill loses at most that much; a clip that raises
is recorded and skipped rather than ending the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import signal
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)


class Writer:
    """Append-only HDF5 sink with resizable datasets, flushed as it goes.

    Args:
        path: Destination ``.h5``.
        flush_every: Samples between flushes.
    """

    def __init__(self, path: str, flush_every: int = 200) -> None:
        import h5py  # noqa: PLC0415

        self.path = path
        self.f = h5py.File(path, "w")
        self.n = 0
        self.flush_every = int(flush_every)
        self._ds: dict = {}

    def append(self, row: dict) -> None:
        """Write one sample, creating datasets on the first call.

        Args:
            row: Field name -> array or scalar.  The shape after the leading axis must be the
                same on every call.
        """
        import h5py  # noqa: PLC0415

        if not self._ds:
            for k, v in row.items():
                a = np.asarray(v)
                # float16 for the exteroception row (it dominates the file and arrives as
                # fp16 from the camera path anyway); float32 elsewhere; ints kept exact.
                dt = (np.float16 if k == "extero" else
                      a.dtype if a.dtype.kind in "iub" else np.float32)
                self._ds[k] = self.f.create_dataset(
                    k, shape=(0, *a.shape), maxshape=(None, *a.shape), dtype=dt,
                    chunks=(min(64, max(1, 1)), *a.shape), compression="lzf")
        for k, d in self._ds.items():
            d.resize(self.n + 1, axis=0)
            d[self.n] = np.asarray(row[k])
        self.n += 1
        if self.n % self.flush_every == 0:
            self.f.flush()

    def close(self, meta: dict) -> None:
        """Stamp provenance and close.

        Args:
            meta: Written as JSON into the file's attrs.
        """
        self.f.attrs["meta"] = json.dumps(meta)
        self.f.flush()
        self.f.close()

    @property
    def bytes(self) -> int:
        """Current file size on disk.

        Reads the cached path, not ``self.f.filename`` -- the handle is gone after
        :meth:`close`, and the final size report is made after closing.
        """
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0


def _sample(inp, window, cmd, value, meta: dict) -> dict:
    """One training row: the conditioning, the chunk, and the labels.

    Args:
        inp: The :class:`critic.CriticInputs` the state was scored with.
        window: ``(F, D)`` the chunk the generator emitted -- the ACTION.
        cmd: ``(3,)`` the commanded twist.
        value: The critic's V(s) at this state, kept as a baseline and so the cheap
            bootstrapped label can be fitted later without re-collecting.
        meta: Identifiers and the outcome.

    Returns:
        The row.
    """
    return {
        "history": np.asarray(inp.history, np.float32),
        "extero": np.asarray(inp.extero, np.float16).reshape(-1),
        "cmd_future": np.asarray(inp.cmd_future, np.float32),
        "history_null": np.uint8(bool(inp.history_null)),
        "scan": np.asarray(inp.scan, np.float32).reshape(-1),
        "proprio": np.asarray(inp.proprio, np.float32),
        "goal": np.asarray(inp.goal, np.float32),
        "twist": np.asarray(inp.twist, np.float32),
        "window": np.asarray(window, np.float32),
        "command": np.asarray(cmd, np.float32),
        "value": np.float32(value),
        **{k: np.int64(v) for k, v in meta.items()},
    }


def run(args) -> int:
    """Roll, branch, and write until a stop condition binds."""
    import terrains  # noqa: PLC0415

    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    from fm_noise_attribution import NoiseTape, build_arm  # noqa: PLC0415
    from gate_a import select_clips  # noqa: PLC0415

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    # A kill already leaves a READABLE file -- verified on a live mid-write copy and on a
    # crashed shard -- but `meta` is stamped only by `Writer.close()`, so a killed run loses
    # its own record of how it was built. Catching the signal turns "recoverable" into
    # "clean": the loop finishes its current episode, the last partial batch is flushed, and
    # `stopped_on` says a signal did it. Re-raising is deliberately NOT done; a second signal
    # still kills outright, so a wedged run can always be forced.
    signalled: list = []
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, lambda sg, _f: (signalled.append(sg),
                                            print(f"\n[collect] signal {sg}: finishing the "
                                                  "current episode, then closing cleanly",
                                                  flush=True)))
    crit = C.ValueCritic(args.critic_ckpt, device=args.device)
    tiles = [int(t) for t in args.tiles.split(",")]
    clips = select_clips(args.clip_lib, tiles, int(args.clips_per_tile))
    # SHUFFLED, deterministically. `select_clips` returns them sorted, so a run that stops on
    # its time budget before exhausting the list covers only the low tile ids -- and the
    # families differ enough in clip length (stairs ~6.3 s, jump_on ~10.5 s) that the slow
    # worker reaches barely a quarter of its clips in the same wall time. Shuffling makes an
    # early stop an unbiased SUBSAMPLE of the tiles rather than a prefix of them, which is the
    # same coverage bias that spoiled the first attempt, arriving by a different route.
    np.random.default_rng(int(args.shuffle_seed)).shuffle(clips)
    print(f"[collect] {len(clips)} clips over {len(tiles)} tiles, shuffled "
          f"(seed {args.shuffle_seed}) so an early stop is a subsample, not a prefix",
          flush=True)

    w = Writer(os.path.join(args.out, f"q_data_{args.shard}.h5"), args.flush_every)
    counts = {"onpolicy": 0, "branch": 0, "episodes": 0, "clips_failed": 0}
    stop = None
    # One-element list so the clip loop can populate it on the first pass without `nonlocal`.
    arm: list = [None]

    def _budget() -> str | None:
        """Which stop condition binds, or None."""
        if signalled:
            return f"signal {signalled[0]}"
        if (time.time() - t0) / 3600.0 >= args.max_hours:
            return f"max_hours ({args.max_hours})"
        if w.n >= args.max_samples:
            return f"max_samples ({args.max_samples})"
        free_gb = shutil.disk_usage(args.out).free / 1e9
        if free_gb < args.min_free_gb:
            return f"min_free_gb (only {free_gb:.1f} GB left)"
        return None

    for clip_id in clips:
        if (stop := _budget()):
            break
        try:
            course = terrains.isaac_clip_course(args.clip_lib, clip_id,
                                               max_offset=args.max_offset)
            if not course.params["clip_constant_command"]:
                print(f"[collect] clip {clip_id}: time-varying command, skipped "
                      "(run_episode holds one command)", flush=True)
                continue
            course = dataclasses.replace(course, name=f"clip{clip_id}{args.scene_tag}")
            # Load the ONNX pair ONCE for the whole run and rebuild only the rig per clip.
            # `build_arm` per clip creates a fresh tracker + generator session every time and
            # never releases the previous one, so the CUDA arena grows without bound: three
            # workers over ~120 clips exhausted a 12 GB card, which surfaced as
            # `bfc_arena.cc: Exception during initialization` on one worker and as 103 of 120
            # clips "FAILED to build" on another -- a resource leak wearing the costume of a
            # data problem. The policy is clip-INDEPENDENT (it is the tracker/generator pair);
            # only the course and its scene change.
            if arm[0] is None:
                arm[0] = build_arm(
                    type("A", (), dict(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                                       out=args.out, providers=args.providers,
                                       fall_height=args.fall_height,
                                       fall_tilt_deg=args.fall_tilt_deg,
                                       cmd_hold_s=terrains.CMD_HOLD_S))(),
                    course)[:3]
            cfg, policy, fm_lag = arm[0]
            spawn = mjrunner.clip_spawn(course, policy.get_joint_names())
            rig = mjrunner.CourseRig(course, policy, cfg, spawn)
            tape = NoiseTape(policy)
            cmd = np.asarray(course.params["clip_command"][0], np.float64)
        except Exception:  # noqa: BLE001 - one bad clip must not end an overnight run
            counts["clips_failed"] += 1
            print(f"[collect] clip {clip_id} FAILED to build:\n{traceback.format_exc()}",
                  flush=True)
            continue

        anchor = course.goal_anchor_xyz
        for seed in range(args.seeds):
            if (stop := _budget()):
                break
            # ---- the reference episode, capturing every re-plan ---------------------------
            cap = _Capture(crit, anchor, rig.robot, policy)
            policy._rng = np.random.default_rng(seed)
            mjrunner._reset_episode_state(policy, fm_lag)
            tape.reset(None)
            v, f, _ = mjrunner.run_episode(
                rig, policy, fm_lag, spawn, cmd, None, steer_kp=cfg.steer_kp,
                steer_wz_max=cfg.steer_wz_max, cfg_hold=cfg.cmd_hold_s, probe=cap)
            counts["episodes"] += 1
            failed = int(v != "success")
            tp = list(tape.recorded)
            for i, (inp, win, pf) in enumerate(cap.rows):
                w.append(_sample(inp, win, cmd, pf,
                                 {"clip_id": clip_id, "terrain_id": course.params["terrain_id"],
                                  "seed": seed, "replan": i, "failed": failed,
                                  "stream": 0, "branch_k": -1}))
                counts["onpolicy"] += 1

            # ---- branches: K actions from ONE state, the comparison a selector needs ------
            picks = list(range(0, max(len(tp) - 1, 0), max(1, args.branch_every)))
            for pick in picks:
                if (stop := _budget()):
                    break
                for k in range(args.k):
                    rng = np.random.default_rng(910000 + 1000 * pick + k)
                    inj = ([tp[j] for j in range(pick)]
                           + [rng.standard_normal(tp[0].shape)])
                    bcap = _Capture(crit, anchor, rig.robot, policy, only_replan=pick)
                    policy._rng = np.random.default_rng(920000 + 1000 * pick + k)
                    mjrunner._reset_episode_state(policy, fm_lag)
                    tape.reset(inj)
                    bv, _bf, _ = mjrunner.run_episode(
                        rig, policy, fm_lag, spawn, cmd, None, steer_kp=cfg.steer_kp,
                        steer_wz_max=cfg.steer_wz_max, cfg_hold=cfg.cmd_hold_s, probe=bcap)
                    counts["episodes"] += 1
                    if not bcap.rows:
                        continue
                    inp, win, pf = bcap.rows[0]
                    w.append(_sample(inp, win, cmd, pf,
                                     {"clip_id": clip_id,
                                      "terrain_id": course.params["terrain_id"],
                                      "seed": seed, "replan": pick,
                                      "failed": int(bv != "success"),
                                      "stream": 1, "branch_k": k}))
                    counts["branch"] += 1
            print(f"[collect] clip {clip_id} seed {seed}: {v:8s} "
                  f"n={w.n} ({counts['onpolicy']} onpolicy / {counts['branch']} branch) "
                  f"{w.bytes/1e6:.0f} MB  {(time.time()-t0)/3600:.2f} h", flush=True)

    w.close({"stopped_on": stop or "clip list exhausted", **counts,
             "hours": (time.time() - t0) / 3600.0, "args": vars(args)})
    print(f"\n[collect] STOPPED ON: {stop or 'clip list exhausted'}")
    print(f"[collect] {w.n} samples ({counts['onpolicy']} onpolicy, {counts['branch']} branch) "
          f"from {counts['episodes']} episodes over {(time.time()-t0)/3600:.2f} h")
    print(f"[collect] {counts['clips_failed']} clip(s) failed to build")
    print(f"[collect] -> {args.out}/q_data_{args.shard}.h5 ({w.bytes/1e9:.2f} GB)")
    return 0


class _Capture:
    """Rides ``run_episode``'s probe, keeping the conditioning + chunk at each re-plan.

    Args:
        crit: The failure critic.
        anchor: Goal anchor in the course frame.
        robot: The MuJoCo robot.
        policy: The loaded pair.
        only_replan: Keep only this re-plan index, or ``None`` for all.  A branch needs just
            the one it differs at; keeping the rest would fill the file with rows whose action
            was not the one being varied.
    """

    def __init__(self, crit, anchor, robot, policy, only_replan: int | None = None) -> None:
        self.crit, self.anchor, self.robot, self.policy = crit, anchor, robot, policy
        self.only = only_replan
        self.ring = C_ring()
        self.rows: list[tuple] = []
        self._i = 0

    def __call__(self, robot, policy, state, is_replan: bool) -> None:
        """One control step: ring always, capture on the wanted re-plan."""
        import critic as C  # noqa: PLC0415

        self.ring.push(state["joint_pos"], state["joint_vel"], state["base_ang_vel"],
                       C.projected_gravity_b(state["root_quat"]))
        if not is_replan:
            return
        idx = self._i
        self._i += 1
        if self.only is not None and idx != self.only:
            return
        win = policy.predicted_window()
        if win is None:
            return
        inp = C.CriticInputs.from_runtime(policy, robot, state, anchor=self.anchor,
                                          ring=self.ring, want_scan=True)
        # The label is V(s) now, not P(fail|s): higher is better, and any consumer
        # ordering these rows has to flip its comparison.
        self.rows.append((inp, np.asarray(win, np.float32), self.crit.value(inp)))


def C_ring():
    """A fresh proprioception ring (import-local so the module loads without torch)."""
    import critic as C  # noqa: PLC0415

    return C.ProprioRing()


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="0", help="suffix, so concurrent workers do not collide.")
    p.add_argument("--critic_ckpt", required=True)
    p.add_argument("--tracker_onnx", required=True)
    p.add_argument("--fm_onnx", required=True)
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--tiles", default="6,20,40")
    p.add_argument("--clips_per_tile", type=int, default=20)
    p.add_argument("--seeds", type=int, default=4, help="reference episodes per clip.")
    p.add_argument("--k", type=int, default=8, help="branches per branched re-plan.")
    p.add_argument("--branch_every", type=int, default=6,
                   help="branch every Nth re-plan. Branch samples cost a full episode EACH; "
                        "on-policy ones are ~20 per episode and free. This is the budget "
                        "split between 'many states, one action' and 'one state, many "
                        "actions', and only the second contains the comparison a selector "
                        "has to make.")
    p.add_argument("--shuffle_seed", type=int, default=0,
                   help="orders the clip list. Deterministic, so a resumed or re-run shard "
                        "covers the same clips in the same order.")
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--scene_tag", default="_qc")
    p.add_argument("--max_hours", type=float, default=9.0)
    p.add_argument("--max_samples", type=int, default=400_000)
    p.add_argument("--min_free_gb", type=float, default=60.0,
                   help="stop while this much disk is still free. The box was at 88% used "
                        "with 211 GB free when this was written; an overnight run that fills "
                        "a shared disk is worse than a short one.")
    p.add_argument("--flush_every", type=int, default=200)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
