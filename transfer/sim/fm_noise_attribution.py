# Copyright 2026 Zachary Olkin. All rights reserved.

"""WHEN does a failing rollout become committed to failing, and does one chunk decide it?

Every advantage-based fine-tune under consideration -- conditioning, AWR, chunk-level
filtered BC -- needs one thing to be true: a chunk's noise draw has to change the outcome
GIVEN the state.  Two fine-tunes have come back null.  This measures whether that signal
exists at all, and where it lives.

The whole test rests on one property, which ``fm_variance_probe --zero_noise_control``
already measured at sd = 0.000e+00: with the noise sequence fixed, a MuJoCo rollout is
deterministic.  So the noise sequence ``z = (z_1 ... z_T)`` -- one draw per re-plan -- is the
ONLY free variable, and substituting parts of it between a successful rollout ``A`` and a
failing rollout ``B`` from the SAME spawn isolates which draws mattered.

Three stages, sharing ``<out>/attribution.json``:

* ``pairs``  -- roll N seeds, tape every draw, keep the successes and failures.  Also runs
  the BLOCKING replay-determinism check (§2): a tape is re-injected under a DIFFERENT rng
  seed, so anything stochastic that the tape does not capture shows up as a divergence.
* ``prefix`` -- the primary experiment.  For each pair and each ``k``, run
  ``(z_1^B..z_k^B, z_{k+1}^A..z_T^A)``.  Draws ``1..k`` reproduce B's trajectory exactly, so
  this asks: from B's state at step k, can A's remaining draws still rescue it?  The mirror
  direction brackets the same window from the other side.
* ``single`` -- at each step in the window the prefix sweep identifies, replace ONE draw with
  ``K`` fresh ones and report the failure RATE.  B's draw at step i is not special (flow noise
  is sampled independently of state), so a single B-substitution is just K=2 branching; fresh
  draws are strictly more informative for the same code path.

**Only the NOISE is substituted, never the emitted chunk** -- every later chunk is re-inferred
normally, which is the point: the model gets to respond to the perturbed state.

NOT INCLUDED, and deliberately: the critic ``V(s_t)`` cross-check of §8.2.  Both critic
variants take three GOAL-OFFSET features (``goal_downrange_m`` / ``goal_lateral_m`` /
``goal_height_err_m``) that are Isaac goal-region quantities; the mjeval courses define
arrival by a downrange target line instead, so there is no faithful mapping and a guessed one
would produce a confident-looking V trace that is off-distribution for the critic.  See the
module TODO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# The same bootstrap fm_variance_probe.py uses: mjeval's modules import each other by bare
# name, so `mjeval/` itself has to be on the path, not just its parent.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, "mjeval"), os.path.dirname(_HERE),
           os.path.dirname(os.path.dirname(_HERE))):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# TODO(attribution-critic-trace): §8.2 wants V(s_t) beside the flip point, to say whether the
# critic drops at COMMITMENT or only at MANIFESTATION. It needs the three goal-offset features
# the critic was fit with, which are Isaac goal-region quantities. Options, neither free: run
# this test in Isaac (where they exist, but where the state-restore harness was abandoned as
# untrustworthy and determinism is not established), or define a mjeval-course equivalent of
# the goal region and re-fit the critic on features that exist in both. Decide before guessing.


class NoiseTape:
    """Capture and/or inject the per-re-plan flow-matching noise draw.

    Wraps :meth:`GenTrackerPolicy._sample_noise`, which is the single place the sampler's
    ``x0`` comes from: one ``(1, F, D)`` draw per re-plan, consumed by ``_run_fm``.  Nothing
    else in the deploy runtime is stochastic, which is what the replay check proves rather
    than assumes.

    Args:
        policy: The loaded :class:`GenTrackerPolicy` to wrap.  Restored by :meth:`detach`.
    """

    def __init__(self, policy):
        self.policy = policy
        self._orig = policy._sample_noise
        self.recorded: list[np.ndarray] = []
        self.inject: list[np.ndarray] | None = None
        self.idx = 0
        self.n_fresh = 0
        policy._sample_noise = self._hook

    def _hook(self) -> np.ndarray:
        """Serve the injected draw for this step, or a fresh one past the end of the tape."""
        if self.inject is not None and self.idx < len(self.inject):
            z = np.asarray(self.inject[self.idx], dtype=np.float64)
        else:
            z = self._orig()
            self.n_fresh += 1
        self.recorded.append(np.asarray(z, dtype=np.float64).copy())
        self.idx += 1
        return z

    def reset(self, inject: list[np.ndarray] | None = None) -> None:
        """Start a new episode, optionally replaying ``inject`` from step 0."""
        self.recorded = []
        self.inject = inject
        self.idx = 0
        self.n_fresh = 0

    def detach(self) -> None:
        """Put the policy's own sampler back."""
        self.policy._sample_noise = self._orig


def roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, inject=None, seed: int = 0) -> dict:
    """One deterministic episode, returning its outcome, tape and base trajectory.

    Args:
        rig / policy / fm_lag / spawn / cmd / cfg: the loaded mjeval arm.
        tape: A :class:`NoiseTape` already wrapping ``policy``.
        inject: Noise draws to replay from step 0, or ``None`` for all-fresh.
        seed: Seeds the policy's own RNG, which supplies any draw past ``inject``.

    Returns:
        Dict with ``verdict``, ``failure``, ``row``, ``tape`` and ``traj`` (the per-control-step
        base position, for the divergence curve).
    """
    import runner as mjrunner

    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)
    tape.reset(inject)
    traj: list[np.ndarray] = []

    def _rec(robot, _pol):
        traj.append(np.asarray(robot.mj_data.qpos[:3], dtype=np.float64).copy())

    verdict, failure, acc = mjrunner.run_episode(
        rig, policy, fm_lag, spawn, cmd, None, recorder=_rec,
        steer_kp=cfg.steer_kp, steer_wz_max=cfg.steer_wz_max, cfg_hold=cfg.cmd_hold_s)
    row = acc.row(episode_id=seed, terrain_idx=0, speed_idx=0, arm=0, seed=seed,
                  verdict=verdict, failure_code=failure)
    return {"verdict": verdict, "failure": failure, "row": row,
            "tape": list(tape.recorded), "traj": np.array(traj), "n_fresh": tape.n_fresh}


def divergence_curve(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-step distance between a rollout's base trajectory and the reference's.

    Compared over the overlap only; a rollout that ended earlier simply has a shorter curve.
    This is what separates "diverged immediately, failed later" from "tracked the reference
    until step 14, then failed for an unrelated reason" -- without it a late failure after an
    early substitution cannot be read at all.
    """
    n = min(len(traj), len(ref))
    if n == 0:
        return np.zeros(0)
    return np.linalg.norm(np.asarray(traj[:n]) - np.asarray(ref[:n]), axis=-1)


def _course(terrains, clip_lib: str, tile: int, speed: float, scene_tag: str):
    """The library tile as a first-class mjeval course.

    The scene NAME carries the tag: ``CourseRig`` writes ``robots/g1/<robot>_<course>.xml``
    from the course before loading it, so two processes sharing a course name tear that file
    silently rather than failing to load.
    """
    return terrains.isaac_tile_course(clip_lib, tile, speeds=(speed,),
                                      name=f"tile{tile}{scene_tag}")


def build_arm(args, course, history_from: str = "generator"):
    """Load the pair once and rig the course; returns everything ``roll`` needs.

    Args:
        args: Parsed CLI, for the artifact paths and the physical thresholds.
        course: The course to rig.
        history_from: What fills the generator's ``N`` history rows -- ``"generator"`` (its
            own served nodes, the deployed loop) or ``"measured"`` (the robot's achieved
            state). An explicit keyword with the deployed default rather than something read
            off ``args``, because nine callers share this function and only the ones running
            that A/B declare the flag; a ``getattr`` fallback here would silently absorb a
            rename of it.

    Returns:
        ``(cfg, policy, fm_lag, spawn, rig)``.
    """
    import runner as mjrunner

    cfg = mjrunner.ArmRun(
        history_from=history_from,
        arm=0, label=f"attr_{course.name}", tracker_onnx=args.tracker_onnx,
        fm_onnx=args.fm_onnx, out_dir=args.out,
        providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
        fall_height=float(args.fall_height), fall_tilt_deg=float(args.fall_tilt_deg),
        cmd_hold_s=float(args.cmd_hold_s),
        # Held at 0 for the same reason the variance probe holds it: a heading controller is a
        # second thing varying between rollouts, and it is off-distribution on stairs anyway.
        steer_kp=0.0,
        # GAUSSIAN, necessarily. At zero_noise there is one trajectory per conditioning and
        # nothing to substitute -- the experiment does not exist.
        zero_noise=False, fm_seed=0,
    )
    policy, fm_lag, spawn = mjrunner.load_arm(cfg)
    rig = mjrunner.CourseRig(course, policy, cfg, spawn)
    return cfg, policy, fm_lag, spawn, rig


def stage_pairs(args) -> int:
    """Roll N seeds, tape every draw, and check replay determinism. BLOCKING."""
    import terrains

    course = _course(terrains, args.clip_lib, int(args.tile), float(args.speed),
                     args.scene_tag)
    cfg, policy, fm_lag, spawn, rig = build_arm(args, course)
    tape = NoiseTape(policy)
    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)
    print(f"[attr] {course.name}: {course.note}")
    print(f"[attr] cmd={cmd.tolist()} zero_noise=False trials={args.trials}")

    # ---- §2 replay determinism, BLOCKING -------------------------------------------------
    a = roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, seed=0)
    # A DIFFERENT rng seed on replay: if anything outside the tape were stochastic, it would
    # differ here and the trajectories would separate. Same seed would hide exactly that.
    b = roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, inject=a["tape"], seed=99991)
    n = min(len(a["traj"]), len(b["traj"]))
    dmax = float(np.abs(a["traj"][:n] - b["traj"][:n]).max()) if n else float("inf")
    same_len = len(a["traj"]) == len(b["traj"])
    print(f"[attr] REPLAY CHECK: len {len(a['traj'])} vs {len(b['traj'])}, "
          f"max |dx| = {dmax:.3e}, fresh draws on replay = {b['n_fresh']}")
    if not same_len or dmax > 1e-9:
        raise SystemExit(
            "REPLAY IS NOT DETERMINISTIC -- the noise tape is not the only free variable, so "
            "every substitution below would confound the draw with whatever else is varying.\n"
            f"  trajectory lengths {len(a['traj'])} vs {len(b['traj'])}, max |dx| {dmax:.3e}\n"
            "  Find the other source before continuing (§2).")
    print("[attr] replay is exact -- the noise sequence is the only free variable.")

    rows, tapes = [], []
    t0 = time.time()
    for seed in range(int(args.trials)):
        r = roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, seed=seed)
        rows.append({"seed": seed, "verdict": r["verdict"], "failure": r["failure"],
                     "n_chunks": len(r["tape"]), "n_steps": int(len(r["traj"])),
                     "progress_x_m": float(r["row"]["progress_x_m"])})
        tapes.append(np.stack(r["tape"]) if r["tape"] else np.zeros((0, 1, 1)))
        if seed < 3 or (seed + 1) % 10 == 0:
            print(f"[attr] seed {seed:>3}  {r['verdict']:<9}{r['failure']:<7} "
                  f"chunks={len(r['tape']):>3}  x={r['row']['progress_x_m']:5.2f}m "
                  f"({time.time() - t0:.0f}s)", flush=True)
    tape.detach()

    succ = [r["seed"] for r in rows if r["verdict"] == "success"]
    fail = [r["seed"] for r in rows if r["verdict"] == "fall"]
    print(f"[attr] {len(succ)} success / {len(fail)} fall of {len(rows)}")
    if not succ or not fail:
        raise SystemExit(
            f"need BOTH outcomes to form pairs, got {len(succ)} success / {len(fail)} fall. "
            "Pick a tile/speed with an intermediate success rate -- a near-0 or near-1 tile "
            "has no contrast to attribute.")

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "tapes.npz"),
                        **{f"seed{r['seed']}": t for r, t in zip(rows, tapes)})
    n_pairs = min(int(args.pairs), len(succ) * len(fail))
    pairs = [{"pair_id": i, "seed_A": int(succ[i % len(succ)]), "seed_B": int(fail[i % len(fail)])}
             for i in range(n_pairs)]
    state = {"course": course.name, "tile": int(args.tile), "speed": float(args.speed),
             "clip_lib": args.clip_lib, "tracker_onnx": args.tracker_onnx,
             "fm_onnx": args.fm_onnx, "replay_max_dx": dmax,
             "rows": rows, "pairs": pairs}
    with open(os.path.join(args.out, "attribution.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"[attr] wrote {args.out}/attribution.json  ({n_pairs} pairs)")
    return 0


def _my_pairs(state: dict, args) -> list[dict]:
    """This shard's pairs, and the scene tag that keeps its scene file private.

    Rollouts here are embarrassingly parallel -- every (pair, direction, k) hybrid is an
    independent deterministic episode -- and one worker uses ~0.5 GB of GPU and a single core,
    so a 32-core box runs a whole sweep in the time one shard takes.

    THE SCENE TAG MUST DIFFER PER SHARD.  ``CourseRig`` writes
    ``robots/g1/<robot>_<course>.xml`` from the course before loading it, so two workers
    sharing a course name tear that file silently rather than failing to load -- the same
    hazard ``fm_variance_probe`` documents for concurrent speed sweeps.
    """
    pairs = [q for q in state["pairs"] if q["pair_id"] % args.n_shards == args.shard]
    tag = args.scene_tag if args.n_shards == 1 else f"{args.scene_tag}s{args.shard}"
    return pairs, tag


def _hybrid(tape_a: np.ndarray, tape_b: np.ndarray, k: int, b_first: bool) -> list[np.ndarray]:
    """The hybrid noise sequence for prefix index ``k``.

    ``b_first=True``  -> ``(z_1^B .. z_k^B, z_{k+1}^A ..)``: k=0 reproduces A, k=len(B) B.
    ``b_first=False`` -> ``(z_1^A .. z_k^A, z_{k+1}^B ..)``: k=0 reproduces B, k=len(A) A.
    """
    head, tail = (tape_b, tape_a) if b_first else (tape_a, tape_b)
    out = [head[i] for i in range(min(k, len(head)))]
    out += [tail[i] for i in range(k, len(tail))]
    return out


def stage_prefix(args) -> int:
    """§3: the prefix sweep, both directions, for every pair."""
    import terrains

    with open(os.path.join(args.out, "attribution.json")) as f:
        state = json.load(f)
    tapes = np.load(os.path.join(args.out, "tapes.npz"))
    my_pairs, tag = _my_pairs(state, args)
    course = _course(terrains, state["clip_lib"], int(state["tile"]),
                     float(state["speed"]), tag)
    cfg, policy, fm_lag, spawn, rig = build_arm(args, course)
    tape = NoiseTape(policy)
    cmd = np.array([float(state["speed"]), 0.0, 0.0], dtype=np.float64)
    print(f"[attr] shard {args.shard}/{args.n_shards}: pairs "
          f"{[q['pair_id'] for q in my_pairs]}")

    ref = {}
    results = []
    t0 = time.time()
    for pair in my_pairs:
        ta = tapes[f"seed{pair['seed_A']}"]
        tb = tapes[f"seed{pair['seed_B']}"]
        # The reference trajectory for the divergence curves is A's own.
        ra = roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, inject=list(ta),
                  seed=pair["seed_A"])
        ref[pair["pair_id"]] = ra["traj"]
        for b_first in (True, False):
            n_k = len(tb) if b_first else len(ta)
            for k in range(n_k + 1):
                inj = _hybrid(ta, tb, k, b_first)
                r = roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, inject=inj,
                         seed=100000 + pair["pair_id"])
                dc = divergence_curve(r["traj"], ra["traj"])
                results.append({
                    "pair_id": pair["pair_id"],
                    "direction": "prefix_B_then_A" if b_first else "prefix_A_then_B",
                    "k": k, "seed_A": pair["seed_A"], "seed_B": pair["seed_B"],
                    "verdict": r["verdict"], "failure": r["failure"],
                    "termination_step": int(len(r["traj"])),
                    "progress_x_m": float(r["row"]["progress_x_m"]),
                    "n_fresh": int(r["n_fresh"]),
                    "div_max": float(dc.max()) if dc.size else 0.0,
                    "div_curve": [round(float(v), 4) for v in dc[::args.div_stride]],
                })
        print(f"[attr] pair {pair['pair_id']} (A={pair['seed_A']} B={pair['seed_B']}) "
              f"done ({time.time() - t0:.0f}s)", flush=True)
    tape.detach()

    # §3.1's free correctness checks: the two endpoints must reproduce their parents.
    bad = []
    for r in results:
        b_first = r["direction"] == "prefix_B_then_A"
        if r["k"] == 0 and b_first and r["verdict"] != "success":
            bad.append((r["pair_id"], "k=0 of B_then_A should reproduce A (success)"))
        if r["k"] == 0 and not b_first and r["verdict"] != "fall":
            bad.append((r["pair_id"], "k=0 of A_then_B should reproduce B (fall)"))
    if bad:
        raise SystemExit(
            "ENDPOINT CHECK FAILED -- a hybrid at k=0 did not reproduce its parent, so the "
            "replay determinism established in `pairs` does not hold here:\n  "
            + "\n  ".join(f"pair {p}: {m}" for p, m in bad))
    print("[attr] endpoint checks pass: every k=0 reproduced its parent.")

    _write_shard(args, "prefix", results)
    print(f"[attr] wrote {len(results)} prefix rollouts")
    return 0


def flip_points(rows: list[dict]) -> dict:
    """Per (pair, direction): the first k after which the outcome stays flipped.

    Reported with an ``oscillates`` flag, because §8.4 turns on it: a curve that flips back
    and forth means the outcome depends on a CONJUNCTION of draws rather than on a prefix, so
    no single commitment point exists and chunk-level advantage is ill-defined for that pair.
    """
    out = {}
    for pair_id in sorted({r["pair_id"] for r in rows}):
        for direction in ("prefix_B_then_A", "prefix_A_then_B"):
            sel = sorted([r for r in rows if r["pair_id"] == pair_id
                          and r["direction"] == direction], key=lambda r: r["k"])
            if not sel:
                continue
            fell = [r["verdict"] != "success" for r in sel]
            target = True if direction == "prefix_B_then_A" else False
            flip, n_changes = None, 0
            for i in range(1, len(fell)):
                if fell[i] != fell[i - 1]:
                    n_changes += 1
            # First index from which the outcome equals the far endpoint and never changes back.
            for i in range(len(fell)):
                if all(f == target for f in fell[i:]):
                    flip = sel[i]["k"]
                    break
            # B's own termination is the row at the FAR endpoint of the B_then_A sweep
            # (k = len(z^B) reproduces B exactly). Taking a max over the sweep instead read
            # A's length -- k=0 of that direction IS A, and a success runs ~2x longer than a
            # fall, so the "gap" column was A's duration minus a flip index.
            endpoint = sel[-1] if direction == "prefix_B_then_A" else sel[0]
            out[f"{pair_id}:{direction}"] = {
                "pair_id": pair_id, "direction": direction, "flip_k": flip,
                "n_changes": n_changes, "oscillates": n_changes > 1,
                "n_k": len(sel),
                "termination_step_B": int(endpoint["termination_step"]),
                "termination_chunk_B": int(round(endpoint["termination_step"] / 12.0)),
            }
    return out


def stage_single(args) -> int:
    """§4: replace ONE draw with K fresh ones, over the window the prefix sweep found."""
    import terrains

    with open(os.path.join(args.out, "attribution.json")) as f:
        state = json.load(f)
    if "prefix" not in state:
        raise SystemExit("run --stage prefix first: the window comes from its flip points.")
    tapes = np.load(os.path.join(args.out, "tapes.npz"))
    my_pairs, tag = _my_pairs(state, args)
    course = _course(terrains, state["clip_lib"], int(state["tile"]),
                     float(state["speed"]), tag)
    cfg, policy, fm_lag, spawn, rig = build_arm(args, course)
    tape = NoiseTape(policy)
    cmd = np.array([float(state["speed"]), 0.0, 0.0], dtype=np.float64)

    flips = flip_points(state["prefix"])
    results = []
    t0 = time.time()
    for pair in my_pairs:
        fp = flips.get(f"{pair['pair_id']}:prefix_B_then_A", {})
        centre = fp.get("flip_k")
        ta = tapes[f"seed{pair['seed_A']}"]
        if centre is None:
            print(f"[attr] pair {pair['pair_id']}: no clean flip, sweeping the whole episode")
            lo, hi = 0, len(ta)
        else:
            lo = max(0, centre - args.window)
            hi = min(len(ta), centre + args.window + 1)
        for i in range(lo, hi):
            for k_draw in range(int(args.k_fresh)):
                rng = np.random.default_rng(900000 + 1000 * i + k_draw)
                inj = [ta[j] for j in range(len(ta))]
                # A FRESH draw, not B's: flow noise is sampled independently of state, so B's
                # draw at step i is statistically an ordinary sample. Substituting only B's
                # would be K=2 branching; K fresh draws give a failure RATE at each i.
                inj[i] = rng.standard_normal(ta[i].shape)
                r = roll(rig, policy, fm_lag, spawn, cmd, cfg, tape, inject=inj,
                         seed=900000 + 1000 * i + k_draw)
                results.append({
                    "pair_id": pair["pair_id"], "direction": "single_sub", "i": i,
                    "draw": k_draw, "seed_A": pair["seed_A"], "seed_B": pair["seed_B"],
                    "verdict": r["verdict"], "failure": r["failure"],
                    "termination_step": int(len(r["traj"])),
                    "progress_x_m": float(r["row"]["progress_x_m"]),
                })
        print(f"[attr] pair {pair['pair_id']} single-sub {lo}..{hi - 1} done "
              f"({time.time() - t0:.0f}s)", flush=True)
    tape.detach()
    _write_shard(args, "single", results)
    print(f"[attr] wrote {len(results)} single-substitution rollouts")
    return 0


def _write_shard(args, key: str, results: list[dict]) -> None:
    """Write one shard's results beside the manifest, or fold them straight in at N=1."""
    if args.n_shards == 1:
        with open(os.path.join(args.out, "attribution.json")) as f:
            state = json.load(f)
        state[key] = results
        with open(os.path.join(args.out, "attribution.json"), "w") as f:
            json.dump(state, f, indent=2)
        return
    path = os.path.join(args.out, f"{key}.shard{args.shard}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"[attr] wrote {path}")


def stage_merge(args) -> int:
    """Fold every shard file into ``attribution.json``.

    Refuses on a MISSING shard rather than reducing what happened to be on disk: a sweep
    silently short a worker would still plot, and its commitment-point histogram would be a
    statement about the shards that finished rather than about the pairs that were planned.
    """
    with open(os.path.join(args.out, "attribution.json")) as f:
        state = json.load(f)
    for key in ("prefix", "single"):
        found, merged = [], []
        for i in range(args.n_shards):
            path = os.path.join(args.out, f"{key}.shard{i}.json")
            if os.path.exists(path):
                found.append(i)
                with open(path) as f:
                    merged.extend(json.load(f))
        if not found:
            continue
        missing = [i for i in range(args.n_shards) if i not in found]
        if missing:
            raise SystemExit(
                f"{key}: shards {missing} are missing of {args.n_shards}. Merging anyway would "
                "report a result over the pairs that happened to finish. Re-run them, or pass "
                "--n_shards equal to the number actually launched.")
        state[key] = merged
        print(f"[attr] merged {len(merged)} {key} rollouts from {len(found)} shard(s)")
    with open(os.path.join(args.out, "attribution.json"), "w") as f:
        json.dump(state, f, indent=2)
    return 0


def main() -> int:
    """Entry point."""
    import terrains  # noqa: PLC0415  the shared CMD_HOLD_S default
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=("pairs", "prefix", "single", "merge", "plot"))
    p.add_argument("--shard", type=int, default=0, help="this worker's index.")
    p.add_argument("--n_shards", type=int, default=1,
                   help="split the pairs across N workers. Each uses ~0.5 GB of GPU and one "
                        "core, so a 32-core box runs the whole sweep in one shard's time.")
    p.add_argument("--out", required=True)
    p.add_argument("--tracker_onnx", default="")
    p.add_argument("--fm_onnx", default="")
    p.add_argument("--clip_lib", default="clips/g1_mb_v5_body_mode11_decor.h5")
    p.add_argument("--tile", type=int, default=6, help="library tile id (6=stairs_up).")
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--trials", type=int, default=40,
                   help="seeds rolled in the `pairs` stage to find successes and failures.")
    p.add_argument("--pairs", type=int, default=10, help="(A, B) pairs to sweep.")
    p.add_argument("--window", type=int, default=3,
                   help="single-sub sweep half-width around the flip point.")
    p.add_argument("--k_fresh", type=int, default=8,
                   help="fresh draws per step in the single-substitution stage.")
    p.add_argument("--div_stride", type=int, default=5,
                   help="store every Nth point of a divergence curve, to bound the json.")
    p.add_argument("--scene_tag", default="attr",
                   help="REQUIRED to differ between concurrent runs on one tile: CourseRig "
                        "rewrites robots/g1/<robot>_<course>.xml before loading it, so two "
                        "processes sharing a name tear the file silently.")
    p.add_argument("--cmd_hold_s", type=float, default=terrains.CMD_HOLD_S)
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--providers", default="auto")
    args = p.parse_args()

    if args.stage == "pairs":
        return stage_pairs(args)
    if args.stage == "prefix":
        return stage_prefix(args)
    if args.stage == "single":
        return stage_single(args)
    if args.stage == "merge":
        return stage_merge(args)
    from fm_noise_attribution_plots import stage_plot  # noqa: PLC0415
    return stage_plot(args)


if __name__ == "__main__":
    raise SystemExit(main())
