# Copyright 2026 Zachary Olkin. All rights reserved.

"""A LIVE 1-step-lookahead selector: try K chunks for real, keep the one the critic likes.

Gate A measures a CEILING -- what a perfect selector could reach from a frozen state.  This
runs the selector, so its output is an episode success RATE rather than a per-window number,
and it is rank-based end to end: only the ORDER of ``V_priv`` over K candidates matters, so
every calibration question hanging over Gate A is irrelevant here.  Its result stands however
the calibration work turns out.

At each re-plan: snapshot, execute 0.24 s of each of K candidate chunks, evaluate ``V_priv``
on the resulting state, pick the argmin, and continue from THAT branch's state.

**The winner is not re-executed.**  The branch length equals the re-plan cadence, so the
winner's post-branch state IS the state the episode continues from -- restoring and replaying
it would cost a (K+1)th chunk and, worse, is the kind of duplicated execution that silently
diverges if the restore is imperfect.

**Snapshot = whole ``mjData`` + policy state, and both halves are load-bearing.**  Measured on
one 0.24 s chunk:

* ``mj_copyData`` + policy state -> **0.000e+00** on qpos and qvel, holding at 12, 36 and 120
  control steps;
* ``mj_copyData`` alone, policy left running -> **1.787 m**;
* ``mj_setState(mjSTATE_INTEGRATION)`` + policy state -> 8.99e-03 m.

``mjSTATE_INTEGRATION`` is insufficient and cannot be repaired.  Contacts genuinely re-derive
from ``qpos``, but the FORWARD KINEMATICS do not, and the policy reads them (``cvel`` for base
world velocity, ``xpos``/``xmat``/``geom_xpos`` for the depth raycast) in ``build_state``
BEFORE the first ``mj_step`` would re-derive anything.  Straight after ``mj_setState`` those
arrays still describe the previous branch: ``xpos`` off by 0.420 m.  ``mj_forward`` does not
fix it and cannot -- MuJoCo's derived kinematics trail ``qpos`` by one physics step, so a
re-derived state is self-consistent but is not the state the policy observed (measured: it
moves ``xpos`` by 2.1e-03 m on an UNTOUCHED post-step ``mjData``).

**Cost.**  Snapshot+restore is 0.22 ms against a 95.6 ms candidate chunk -- 0.2%, free.  The
whole cost is K forward passes, of which ~55% is depth raycasting.  K=8 is 766 ms per re-plan,
3.2x real time.  Against Gate A's replay-from-spawn on the same course this is **14.7x less
work** (21.7 s vs 317.8 s per episode), because a branch at re-plan n costs n chunks there and
one here.

**The harness check is K=1, and it is exact rather than statistical.**  With one candidate the
selector has no choice to make, so its trajectory must be BIT-IDENTICAL to the plain runner at
the same seed.  That tests the snapshot machinery directly.  Checking instead that K=1
reproduces a published success rate would be far weaker -- and in this case impossible, since
``fm_variance_probe``'s 0.798/0.816 were measured with a DIFFERENT generator
(``lag48_aug``) from the one the critic was fitted over (``nojumpoff``).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: Policy attributes that must be captured for an exact restore, with the measured cost of
#: omitting each (max |dqpos| in metres after one 0.24 s chunk).  Established by ablation --
#: leave one out, branch, measure -- rather than by reading the code, because "what the policy
#: carries across a step" is not something the class states anywhere.
#:
#:   _history            1.653e+00   the generator's emitted-node ring
#:   _obs_history        1.225e-01   the tracker's per-term obs windows
#:   _prev_served_node   1.043e-01
#:   _rng                5.748e-02
#:   _ctrl_step          IndexError  at a branch length not a multiple of fm_update_every
#:
#: The plan block (_pred .. _clip_anchor_*) measures 0.0 ONLY because a snapshot taken on a
#: re-plan boundary has step 0 overwrite it.  It is captured anyway: a mid-chunk snapshot
#: makes it live, and a field that is dead by coincidence of timing is not dead.
_POLICY_FIELDS = (
    "_history", "_obs_history", "_prev_served_node", "_rng", "_ctrl_step",
    "_pred", "_pred_valid", "_pred_idx", "_anchor_pos", "_anchor_quat", "_plan_id",
    "_last_plan", "_last_reference_window", "_last_served_window", "_lead_buf",
    "_future_cmd_override", "_clip_step",
)


class Snapshot:
    """An exact restore point for the sim and the policy together.

    Args:
        robot: The ``transfer.sim.robot.Robot``.
        policy: The ``GenTrackerPolicy``.
        fm_lag: The camera-lag ring.
    """

    def __init__(self, robot, policy, fm_lag) -> None:
        import mujoco  # noqa: PLC0415

        self._mj = mujoco
        self.data = mujoco.MjData(robot.mj_model)
        mujoco.mj_copyData(self.data, robot.mj_model, robot.mj_data)
        self.policy = {k: copy.deepcopy(getattr(policy, k))
                       for k in _POLICY_FIELDS if hasattr(policy, k)}
        self.action = copy.deepcopy(getattr(policy._trk, "action_isaac", None))
        self.lag = copy.deepcopy(getattr(fm_lag, "_buf", None))
        # Anything clip-mode carries. Absent on the generator path; captured when present so
        # this class does not quietly depend on which reference source is loaded.
        self.clip_anchor = {k: copy.deepcopy(getattr(policy, k))
                            for k in dir(policy) if k.startswith("_clip_anchor")}

    def restore(self, robot, policy, fm_lag) -> None:
        """Put both halves back.

        Args:
            robot: The robot to restore into.
            policy: The policy to restore into.
            fm_lag: The camera-lag ring to restore into.
        """
        self._mj.mj_copyData(robot.mj_data, robot.mj_model, self.data)
        for k, v in self.policy.items():
            setattr(policy, k, copy.deepcopy(v))
        for k, v in self.clip_anchor.items():
            setattr(policy, k, copy.deepcopy(v))
        if self.action is not None:
            policy._trk.action_isaac = copy.deepcopy(self.action)
        if self.lag is not None:
            fm_lag._buf = copy.deepcopy(self.lag)


def _chunk(rig, policy, fm_lag, cmd, n_steps, cfg, zex, scorer=None):
    """Advance ``n_steps`` control steps, returning early on a terminal condition.

    Mirrors ``runner.run_episode``'s inner loop and its ORDERING, which is load-bearing there
    and here: the fall test runs BEFORE the goal test, so a robot that face-plants across the
    line is a fall and the verdicts stay mutually exclusive.

    Args:
        rig: The course rig.
        policy: The loaded pair.
        fm_lag: The camera-lag ring.
        cmd: ``(3,)`` commanded twist.
        n_steps: Control steps to run.
        cfg: The ``ArmRun``.
        zex: Mutable ``[max_base_z, min_base_z]`` for the traversal gate, carried across
            chunks because it is an EPISODE-level quantity.  ``run_episode`` keeps it in the
            accumulator; this loop has no accumulator, and omitting it makes ``in_goal`` alone
            decide success -- which awards it to a robot that walked BESIDE the course on the
            infinite floor, exactly what the gate exists to refuse.
        scorer: Optional per-step callback ``(robot, policy, state)``.

    Returns:
        ``(verdict, failure, last_state)`` -- ``verdict`` is ``""`` if nothing terminal
        happened, and ``last_state`` is the state dict of the final control step, which the
        caller scores on.  Returned rather than rebuilt by the caller because ``build_state``
        MUTATES the depth-lag ring: calling it again to peek pushes an extra frame, and since
        the post-chunk snapshot is taken after that, every candidate was being committed with
        a desynchronised lag. It presented as the selector making outcomes WORSE than no
        selection at all (K=4 fell on three seeds that K=1 survived), which reads as a weak
        critic and is not.
    """
    from gen_tracker_runner import build_state  # noqa: PLC0415

    robot, course = rig.robot, rig.course
    last = None
    for _ in range(int(n_steps)):
        if robot.mj_data.time > course.budget_s:
            return "timeout", "", last
        state = build_state(robot, policy, cmd, rig.depth_cams, want_depth_hits=False,
                            fm_depth_cams=rig.fm_depth_cams, fm_lag=fm_lag)
        last = state
        obs = policy.create_obs(state)
        robot.apply_action(policy.get_action(obs, robot.joint_names))
        if robot.failed():
            return "fall", robot.failure_reason(), last
        if scorer is not None:
            scorer(robot, policy, state)
        for _ in range(rig.decim):
            robot.step()
        z = float(robot.mj_data.qpos[2])
        zex[0], zex[1] = max(zex[0], z), min(zex[1], z)
        if (course.in_goal(float(robot.mj_data.qpos[0]), float(robot.mj_data.qpos[1]))
                and course.traversed(zex[0], zex[1])):
            return "success", "", last
    return "", "", last


def run_episode_selected(rig, policy, fm_lag, spawn, cmd, cfg, crit, anchor, *,
                         k: int, seed: int, mjrunner):
    """One episode driven by a live 1-step-lookahead selector over ``k`` candidate chunks.

    Args:
        rig: The course rig.
        policy: The loaded pair.
        fm_lag: The camera-lag ring.
        spawn: The initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        crit: A ``critic.FailureCritic``, or ``None`` at ``k == 1``.
        anchor: Goal anchor for the critic's goal features.
        k: Candidates per re-plan.  ``k == 1`` takes the selector's own path with no choice to
            make, so it must reproduce the plain runner exactly -- that is the harness check,
            not a separate code path.
        seed: Episode seed.
        mjrunner: The ``runner`` module.

    Returns:
        ``(verdict, failure, n_replans, mean_candidates_scored)``.
    """
    import critic as C  # noqa: PLC0415
    import mujoco  # noqa: PLC0415

    robot, course = rig.robot, rig.course
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)

    cadence = int(getattr(policy, "fm_update_every", 12))
    ring = C.ProprioRing() if (crit is not None and crit.privileged) else None

    def _ring(robot_, policy_, state):
        if ring is not None:
            ring.push(state["joint_pos"], state["joint_vel"], state["base_ang_vel"],
                      C.projected_gravity_b(state["root_quat"]))

    z0 = float(robot.mj_data.qpos[2])
    zex = [z0, z0]
    n_replans, scored = 0, []
    while robot.mj_data.time <= course.budget_s:
        if k <= 1:
            v, f, _st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, zex, scorer=_ring)
            if v:
                return v, f, n_replans, 0.0
            n_replans += 1
            continue

        snap = Snapshot(robot, policy, fm_lag)
        best = None
        for cand in range(k):
            if cand:
                snap.restore(robot, policy, fm_lag)
            # The ONLY difference between candidates. One flow sample is drawn per re-plan,
            # so re-seeding here varies the chunk and nothing else.
            #
            # THE EPISODE SEED IS PART OF IT. Without it every K>1 episode draws the identical
            # candidate sequence, so all N episodes of an arm are ONE episode repeated -- the
            # measured "success rate" is then 0.000 or 1.000 and nothing else, which reads as
            # a catastrophically bad selector. K=1 is unaffected (it advances policy._rng from
            # the episode seed naturally), which is exactly what made the two arms look so far
            # apart. Seeded as a SEQUENCE rather than an arithmetic combination so no
            # (seed, replan, cand) triple can alias another.
            policy._rng = np.random.default_rng([seed, n_replans, cand])
            # Each candidate gets its OWN copy of the episode's z extremes: a branch that
            # climbs must not hand its traversal credit to a sibling that did not.
            cz = list(zex)
            v, f, st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, cz, scorer=_ring)
            if v == "fall":
                # A candidate that already fell is worse than any survivor. Scored 1.0 rather
                # than skipped, so a re-plan where EVERY candidate falls still has an argmin
                # and the episode ends the way it would have anyway.
                score = 1.0
            elif v == "success":
                score = -1.0
            elif st is None:
                score = 1.0
            else:
                # The chunk's own last state, NOT a rebuilt one. See _chunk's docstring: a
                # rebuild advances the depth-lag ring and corrupts the committed state.
                x = C.CriticInputs.from_runtime(policy, robot, st, anchor=anchor, ring=ring,
                                                want_scan=crit.privileged)
                score = crit.p_fail(x)
            post = Snapshot(robot, policy, fm_lag)
            if best is None or score < best[0]:
                best = (score, post, v, f, cz)
        scored.append(k)
        # Commit the winner by RESTORING ITS POST-CHUNK STATE -- the branch length is the
        # cadence, so that state is exactly where the episode continues from. Re-executing the
        # winning draw instead would cost a (K+1)th chunk and would diverge silently if the
        # restore were imperfect.
        best[1].restore(robot, policy, fm_lag)
        zex = list(best[4])
        n_replans += 1
        if best[2]:
            return best[2], best[3], n_replans, float(np.mean(scored))
    return "timeout", "", n_replans, (float(np.mean(scored)) if scored else 0.0)
def run(args) -> int:
    """Run the K arms, seed-matched, and report the episode success rate of each."""
    import terrains  # noqa: PLC0415

    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    from fm_noise_attribution import build_arm  # noqa: PLC0415

    ks = [int(x) for x in args.k.split(",")]
    crit = (None if ks == [1] else
            C.FailureCritic(args.critic_ckpt, device=args.device))
    os.makedirs(args.out, exist_ok=True)
    rows, t0 = [], time.time()

    for tile in [int(t) for t in args.tiles.split(",")]:
        # The fm_variance_probe configuration, exactly: tile course, 2.5 m standing approach,
        # constant 0.8 m/s. That harness measured 0.798 (tile 6) / 0.816 (tile 20) and is the
        # reference this benchmark is defined against.
        course = terrains.isaac_tile_course(args.clip_lib, tile, speeds=(args.speed,),
                                            approach=args.approach,
                                            goal_anchor_max_offset=args.max_offset)
        course = dataclasses.replace(course, name=f"tile{tile}{args.scene_tag}")
        cfg, policy, fm_lag, spawn, rig = build_arm(
            type("A", (), dict(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                               out=args.out, providers=args.providers,
                               fall_height=args.fall_height,
                               fall_tilt_deg=args.fall_tilt_deg, cmd_hold_s=0.0))(), course)
        cmd = np.array([args.speed, 0.0, 0.0])
        print(f"\n[live] tile {tile} {course.klass} anchor "
              f"{np.round(course.goal_anchor_xyz, 3)}", flush=True)

        for k in ks:
            wins = 0
            for ep in range(args.seed_offset, args.seed_offset + args.episodes):
                v, f, nr, sc = run_episode_selected(
                    rig, policy, fm_lag, spawn, cmd, cfg, crit, course.goal_anchor_xyz,
                    k=k, seed=ep, mjrunner=mjrunner)
                wins += int(v == "success")
                rows.append({"tile": tile, "k": k, "episode": ep, "verdict": v,
                             "failure": f, "n_replans": nr})
                done = ep - args.seed_offset + 1
                if done % max(1, args.episodes // 10) == 0:
                    print(f"  tile {tile} K={k}: {done}/{args.episodes} "
                          f"success {wins / done:.3f}  ({(time.time() - t0) / 60:.1f} min)",
                          flush=True)
            print(f"  tile {tile} K={k}: SUCCESS {wins / args.episodes:.3f} "
                  f"({wins}/{args.episodes})", flush=True)

    with open(os.path.join(args.out, "live_select.json"), "w") as fh:
        json.dump({"rows": rows, "args": vars(args)}, fh, indent=1)
    print(f"\n{'tile':>5} {'K':>3} {'episodes':>9} {'success':>9}")
    for tile in sorted({r["tile"] for r in rows}):
        for k in ks:
            sel = [r for r in rows if r["tile"] == tile and r["k"] == k]
            if sel:
                s = sum(r["verdict"] == "success" for r in sel) / len(sel)
                print(f"{tile:>5} {k:>3} {len(sel):>9} {s:>9.3f}")
    print("\nK=1 is the harness check: with one candidate the selector has no choice, so its "
          "\ntrajectory must match the plain runner exactly. Its success rate is this "
          "benchmark's\nbaseline -- NOT fm_variance_probe's 0.798/0.816, which were measured "
          "with a different\ngenerator (lag48_aug, not the nojumpoff one the critic was "
          "fitted over).")
    print(f"\n[live] {len(rows)} episodes -> {args.out}/live_select.json")
    return 0


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--critic_ckpt", required=True)
    p.add_argument("--tracker_onnx", required=True)
    p.add_argument("--fm_onnx", required=True)
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--tiles", default="6,20")
    p.add_argument("--k", default="1,4,8", help="comma list of candidate counts.")
    p.add_argument("--episodes", type=int, default=200,
                   help="per arm per tile, seed-matched across arms (seed = episode index).")
    p.add_argument("--seed_offset", type=int, default=0,
                   help="first episode seed. Shards an arm across processes WITHOUT breaking "
                        "the seed matching, since the seed is the episode index and not a "
                        "position in a per-process sequence: shard [0,100) and [100,200) of "
                        "K=8 cover exactly the seeds K=1 ran, so the arms still pair.")
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--approach", type=float, default=2.5)
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--scene_tag", default="_ls")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
