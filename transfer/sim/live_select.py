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
import math
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


def arrived(course, robot, zex) -> bool:
    """Did the robot arrive, under whichever rule this course carries?

    ISAAC's rule when ``course.isaac_tol`` is set -- a downrange half-plane plus lateral and
    height bands, no traversal gate.  mjeval's own otherwise -- a rectangle around
    ``target_xy`` plus the traversal gate.

    They are not interchangeable and disagree three ways: Isaac counts OVERSHOOT (a
    half-plane, where mjeval's band can be walked out of), Isaac has a HEIGHT band that mjeval
    lacks, and mjeval has a traversal gate that Isaac does not need -- its height band already
    refuses a robot that never went up.

    Args:
        course: The course.
        robot: The MuJoCo robot.
        zex: ``[max_base_z, min_base_z]`` so far, for the mjeval gate.

    Returns:
        Whether the episode should end as a success.
    """
    q = robot.mj_data.qpos
    if course.isaac_tol is not None:
        from goal_anchor import in_isaac_goal  # noqa: PLC0415

        return in_isaac_goal((float(q[0]), float(q[1]), float(q[2])),
                             course.goal_anchor_xyz, course.isaac_tol)
    return bool(course.in_goal(float(q[0]), float(q[1]))
                and course.traversed(zex[0], zex[1]))


def _chunk(rig, policy, fm_lag, cmd, n_steps, cfg, zex, scorer=None,
          steer_kp: float = 0.0, steer_wz_max: float = 0.68):
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
        steer_kp: P gain (1/s) on heading error toward the course's goal; ``0`` (default)
            holds ``cmd`` flat, matching ``runner.run_episode``'s own ``steer_kp`` semantics
            exactly (same function, ``runner.steer_yaw_rate``, called fresh every step).
        steer_wz_max: Symmetric clip on the steering yaw rate, rad/s.

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
    from runner import steer_yaw_rate  # noqa: PLC0415

    robot, course = rig.robot, rig.course
    last = None
    for _ in range(int(n_steps)):
        if robot.mj_data.time > course.budget_s:
            return "timeout", "", last
        step_cmd = cmd
        if steer_kp > 0.0:
            qw, qx, qy, qz = (float(v) for v in robot.mj_data.qpos[3:7])   # MuJoCo wxyz
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            wz = steer_yaw_rate(course, float(robot.mj_data.qpos[0]), float(robot.mj_data.qpos[1]),
                                yaw, steer_kp, steer_wz_max)
            step_cmd = np.array([cmd[0], cmd[1], wz])
        state = build_state(robot, policy, step_cmd, rig.depth_cams, want_depth_hits=False,
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
        if arrived(course, robot, zex):
            return "success", "", last
    return "", "", last


def run_episode_selected(rig, policy, fm_lag, spawn, cmd, cfg, crit, anchor, *,
                         k: int, seed: int, mjrunner, steer_kp: float = 0.0,
                         steer_wz_max: float = 0.68):
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
        steer_kp: P gain (1/s) on heading error toward the goal; ``0`` (default) disables
            steering. Forwarded to every :func:`_chunk` call, so it applies identically to
            the ``k == 1`` path and every ``k > 1`` candidate.
        steer_wz_max: Symmetric clip on the steering yaw rate, rad/s.

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
            v, f, _st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, zex, scorer=_ring,
                              steer_kp=steer_kp, steer_wz_max=steer_wz_max)
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
            v, f, st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, cz, scorer=_ring,
                             steer_kp=steer_kp, steer_wz_max=steer_wz_max)
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

    if args.catalogue:
        # Matches evaluate.py's own wiring exactly: STAIR_WALLS is a module global read at
        # course-build time, not a per-course field, so every hand-authored stairs course in
        # one run shares this setting.
        terrains.STAIR_WALLS = bool(args.stair_walls)
        print(f"[live] stair walls: {'ON (touching the treads)' if args.stair_walls else 'OFF'}")

    ks = [int(x) for x in args.k.split(",")]
    use_q = bool(args.q_ckpt)
    if args.steer_kp > 0.0:
        if use_q:
            raise SystemExit(
                "--steer_kp is only wired into run_episode_selected (the V selector, "
                "--critic_ckpt/--state_critic_ckpt); run_episode_q_selected (--q_ckpt) does "
                "not read it yet. Refusing rather than silently running unsteered.")
        print(f"[live] steering ON: P={args.steer_kp:.3f}, wz clipped to "
              f"+-{args.steer_wz_max:.3f} rad/s (runner.steer_yaw_rate). OFF-DISTRIBUTION ON "
              "STAIRS -- the training library commands zero turn rate on every stair clip; "
              "this measures a DIFFERENT thing, not a cleaner one.")
    scorer = None
    if args.state_critic_ckpt and ks != [1]:
        scorer = StateCritic(args.state_critic_ckpt, device=args.device,
                             fm_override=args.fm_pt)
    elif use_q and ks != [1]:
        scorer = QScorer(args.q_ckpt, device=args.device, fm_override=args.fm_pt)
    elif ks != [1]:
        if not args.critic_ckpt:
            raise SystemExit("need --critic_ckpt (V selector) or --q_ckpt (Q selector)")
        scorer = C.FailureCritic(args.critic_ckpt, device=args.device)
    crit = scorer
    os.makedirs(args.out, exist_ok=True)
    rows, t0 = [], time.time()

    if args.catalogue:
        units = [("course", int(c)) for c in args.catalogue.split(",") if c.strip()]
    elif args.clips:
        units = [("clip", int(c)) for c in args.clips.split(",") if c.strip()]
    else:
        units = [("tile", int(t)) for t in args.tiles.split(",")]
    for kind, uid in units:
        if kind == "clip":
            # ISAAC-MATCHED: the clip's own terrain, start (pose AND velocity), goal, duration
            # and command. No standing approach, and therefore no cmd_hold_s -- the robot
            # begins already moving at the clip's velocity, so there is no standing-start
            # transient to settle. That knob is worth +-0.26 on these courses and this is how
            # it is removed rather than tuned.
            course = terrains.isaac_clip_course(args.clip_lib, uid,
                                                max_offset=args.max_offset,
                                                goal_regions=args.goal_regions)
            course = dataclasses.replace(course, name=f"clip{uid}{args.scene_tag}")
        elif kind == "course":
            # HAND-AUTHORED: terrains.catalogue(), the general mjeval cell/grid harness's own
            # terrain -- no clip ever walked it, so there is no true terminal pose to anchor
            # the critic on (see the module's Course.goal_anchor_xyz docstring). target_xy is
            # what IS available -- the far platform's centre, already used for this course's
            # own mjeval success rule -- and using it for the critic too is a KNOWN, measured
            # approximation (0.44-3.17 m off a real clip anchor by family, see
            # mjeval.goal_anchor), not a substitute for one. isaac_tol stays unset, so the
            # SUCCESS rule (unlike the critic's goal features) is untouched by this.
            course = terrains.catalogue()[uid]
            anchor_z = course.floor_z + max(course.flight_height_m, 0.0) + terrains.SPAWN_BASE_Z
            course = dataclasses.replace(
                course, name=f"cat{uid}{args.scene_tag}",
                goal_anchor_xyz=(course.target_xy[0], course.target_xy[1], anchor_z),
                goal_anchor_source="target_xy_approx")
            print(f"[live] course {uid} ({course.klass}): critic anchor APPROXIMATED from "
                  "target_xy -- expect a 0.44-3.17 m bias against a real clip anchor.",
                  flush=True)
        else:
            # The fm_variance_probe configuration: tile course, 2.5 m standing approach,
            # constant --speed.
            course = terrains.isaac_tile_course(args.clip_lib, uid, speeds=(args.speed,),
                                                approach=args.approach,
                                                goal_anchor_max_offset=args.max_offset)
            course = dataclasses.replace(course, name=f"tile{uid}{args.scene_tag}")
        cfg, policy, fm_lag, spawn, rig = build_arm(
            type("A", (), dict(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                               out=args.out, providers=args.providers,
                               fall_height=args.fall_height,
                               fall_tilt_deg=args.fall_tilt_deg,
                               cmd_hold_s=(0.0 if kind == "clip" else args.cmd_hold_s)))(),
            course)
        if kind == "clip":
            spawn = mjrunner.clip_spawn(course, policy.get_joint_names())
            rig = mjrunner.CourseRig(course, policy, cfg, spawn)
            cmd = np.array(course.params["clip_command"][0], dtype=np.float64)
        else:
            cmd = np.array([args.speed, 0.0, 0.0])
        tile = int(course.params.get("terrain_id", uid))
        rule = "ISAAC (half-plane + lateral + height)" if course.isaac_tol else "mjeval (band + gate)"
        print(f"\n[live] {kind} {uid} tile {tile} {course.klass} cmd {cmd[0]:.2f} m/s "
              f"budget {course.budget_s:.2f}s anchor {np.round(course.goal_anchor_xyz, 3)}"
              f"\n       arrival rule: {rule}", flush=True)

        for k in ks:
            wins = 0
            for ep in range(args.seed_offset, args.seed_offset + args.episodes):
                fn = run_episode_q_selected if use_q else run_episode_selected
                extra = ({} if use_q else
                        {"steer_kp": args.steer_kp, "steer_wz_max": args.steer_wz_max})
                v, f, nr, sc = fn(
                    rig, policy, fm_lag, spawn, cmd, cfg, crit, course.goal_anchor_xyz,
                    k=k, seed=ep, mjrunner=mjrunner, **extra)
                wins += int(v == "success")
                rows.append({"tile": tile, "unit": f"{kind}{uid}", "k": k,
                         "episode": ep, "verdict": v,
                             "failure": f, "n_replans": nr, "spread": float(sc),
                             "selector": "Q" if use_q else "V"})
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
    p.add_argument("--critic_ckpt", default="", help="V(s) critic, for the rollout selector.")
    p.add_argument("--q_ckpt", default="",
                   help="A fit_q checkpoint. Selects with Q(s, a), which scores each candidate "
                        "chunk WITHOUT executing it -- K generator calls and no physics, "
                        "against the V selector needing K rollouts of 0.24 s. This is also the "
                        "only one of the two a robot could run: it cannot execute eight "
                        "futures and undo seven.")
    p.add_argument("--fm_pt", default="", help="Local generator .pt, when the Q checkpoint "
                                               "names a path from another machine.")
    p.add_argument("--state_critic_ckpt", default="",
                   help="An ABLATED fit_q checkpoint, driven as V(s) by the ROLLOUT selector "
                        "(candidates are executed 0.24 s and the resulting state is scored). "
                        "Use this to run the same selector that worked with V_priv, but with "
                        "a V trained on the fit_q split.")
    p.add_argument("--tracker_onnx", required=True)
    p.add_argument("--fm_onnx", required=True)
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--tiles", default="6,20")
    p.add_argument("--clips", default="",
                   help="comma list of CLIP ids. Builds Isaac-matched per-clip courses -- the "
                        "clip's own terrain, start pose AND velocity, goal, duration and "
                        "command -- instead of tile courses with a standing approach. "
                        "Overrides --tiles, --speed and --cmd_hold_s.")
    p.add_argument("--catalogue", default="",
                   help="comma list of indices into terrains.catalogue() (0-19), the "
                        "hand-authored courses used by the general mjeval cell/grid harness -- "
                        "separate from --tiles/--clips, which read a real clip-library tile. "
                        "Overrides --tiles and --clips. These courses carry NO goal_anchor_xyz "
                        "(no clip ever walked them to record one), so the critic's goal "
                        "features are built from target_xy instead -- a KNOWN, measured "
                        "approximation, biased 0.44-3.17 m depending on family versus the real "
                        "clip-anchor a tile course would give (see mjeval.goal_anchor). Every "
                        "row from this arm is tagged goal_anchor_source=target_xy_approx.")
    p.add_argument("--goal_regions", default="",
                   help="a fitted goal_regions.json; makes arrival ISAAC's rule (downrange "
                        "half-plane + lateral + height bands, no traversal gate) rather than "
                        "mjeval's rectangle plus gate. Clip courses only.")
    p.add_argument("--no_stair_walls", dest="stair_walls", action="store_false", default=True,
                   help="--catalogue only, matches evaluate.py's own flag: build hand-authored "
                        "stairs courses without the visual stairwell walls.")
    p.add_argument("--steer_kp", type=float, default=0.0,
                   help="P gain (1/s) on heading error toward the goal, recomputed every "
                        "control step via runner.steer_yaw_rate; 0 (default) holds the flat "
                        "commanded twist. V selector only (--critic_ckpt/--state_critic_ckpt) "
                        "-- refused with --q_ckpt, which does not read it. "
                        "OFF-DISTRIBUTION ON STAIRS: the training library commands zero turn "
                        "rate on every stair clip (3500 of them), so a stairs run with this "
                        "on is a DIFFERENT measurement, not a cleaner one.")
    p.add_argument("--steer_wz_max", type=float, default=0.68,
                   help="Symmetric clip on the steering yaw rate, rad/s. Matches "
                        "runner.STEER_WZ_MAX. Only used with --steer_kp.")
    p.add_argument("--cmd_hold_s", type=float, default=0.0,
                   help="seconds of zero command before stepping to --speed, on TILE courses. "
                        "Measured worth -0.26 on stairs_up and +0.10 on stairs_down, so it is "
                        "a real experimental variable and neither value is neutral. Clip "
                        "courses force 0: they start already moving.")
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




class QScorer:
    """Scores a candidate CHUNK from the current state, without executing it.

    This is the difference between a ``Q`` selector and the ``V`` one above, and it is not a
    detail.  ``V(s')`` needs a state to score, so every candidate must first be ROLLED 0.24 s
    -- 95.6 ms each, measured.  ``Q(s, a)`` scores the chunk directly, so the candidates cost
    one generator call apiece and no physics at all.  That makes the Q selector roughly an
    order of magnitude cheaper AND deployable: a robot cannot execute eight futures and undo
    seven of them, but it can sample eight plans and pick one.

    Args:
        ckpt: A ``fit_q`` checkpoint.
        device: Torch device.
        fm_override: Path to the generator ``.pt`` the checkpoint borrows, when the recorded
            one does not resolve here -- a checkpoint trained on another machine names ITS
            path. Overriding is safe only because the trunk is frozen and both sides must be
            the same file; the loader below refuses a DIFFERENT generator by content hash
            rather than trusting the caller.
    """

    def __init__(self, ckpt: str, device: str = "cuda", fm_override: str = "") -> None:
        import copy as _copy  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from robot_rl.flow_matching import online, qvalue  # noqa: PLC0415

        self._torch = torch
        self._qv = qvalue
        self.device = device
        k = torch.load(ckpt, map_location="cpu", weights_only=False)
        if bool(k.get("ablate_action", False)):
            raise ValueError(
                f"{ckpt} is the ACTION-ABLATED control arm -- its action tokens were zeroed "
                "during training, so it is a V(s) in a Q's clothing and would rank every "
                "candidate at a state identically. Selecting on it is a no-op that looks "
                "like a working selector.")
        fm = str(k["fm_checkpoint"])
        if fm_override:
            import hashlib as _h  # noqa: PLC0415

            def _sha(q):
                h = _h.sha256()
                with open(q, "rb") as fh:
                    for c in iter(lambda: fh.read(1 << 20), b""):
                        h.update(c)
                return h.hexdigest()
            if os.path.exists(fm) and _sha(fm) != _sha(fm_override):
                raise ValueError(
                    f"fm_override {fm_override!r} is a DIFFERENT generator from the one this Q "
                    f"was fitted over ({fm!r}). The trunk is frozen, so a different one loads "
                    "with the right shapes and three wrong normalizers -- silently.")
            fm = fm_override
        elif not os.path.exists(fm):
            raise FileNotFoundError(
                f"the Q checkpoint names {fm!r}, which does not exist here. It was probably "
                "trained on another machine; pass fm_override with the local copy.")
        self.ctx = online.load_fm_context(fm, device=device)
        self.model = qvalue.FlowQModel(
            _copy.deepcopy(self.ctx.model), action_dim=int(k["action_dim"]),
            action_steps=int(k["action_steps"]), scan_hw=tuple(k["scan_hw"]),
            proprio_dim=int(k["proprio_hw"][-1]), proprio_steps=int(k["proprio_hw"][0]),
            context_dim=int(k["context_dim"])).to(device)
        self.model.load_state_dict(k["model"])
        self.model.eval()
        self.scan_hw = tuple(k["scan_hw"])
        self.steps = torch.as_tensor(self.ctx.cmd_steps, dtype=torch.long, device=device)

    def score(self, inp, window) -> float:
        """``P(fail | s, a)`` for one candidate; lower is better.

        Normalizer order mirrors ``fit_q._batch`` exactly, including the action being
        normalized with ``state_norm`` -- the emitted plan nodes live in the same space as the
        history nodes, which is what that normalizer was fitted on.

        Args:
            inp: A :class:`critic.CriticInputs` for the current state.
            window: ``(F, D)`` the candidate chunk.

        Returns:
            The score.
        """
        import numpy as _np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from robot_rl.flow_matching import transforms, value  # noqa: PLC0415

        from goal_anchor import GOAL_SCALE_M  # noqa: PLC0415

        t = lambda a, d=torch.float32: torch.as_tensor(  # noqa: E731
            _np.asarray(a), dtype=d, device=self.device).unsqueeze(0)
        c = self.ctx
        cmd = (c.cmd_norm.normalize(t(inp.cmd_future)[:, self.steps, :])
               if c.cmd_dim > 0 else None)
        goal = _np.nan_to_num(_np.asarray(inp.goal, dtype=_np.float64) / GOAL_SCALE_M,
                              nan=0.0, posinf=0.0, neginf=0.0)
        st = value.ValueInputs(
            history=c.state_norm.normalize(t(inp.history)),
            extero=c.extero_norm.normalize(t(inp.extero).reshape(1, -1)),
            cmd_tokens=cmd,
            history_null=t(_np.array([inp.history_null]), torch.bool).reshape(1),
            scan=transforms.sanitize_extero(t(inp.scan), "height_scan").reshape(1, *self.scan_hw),
            proprio=t(inp.proprio),
            context=t(_np.concatenate([goal, _np.asarray(inp.twist, dtype=_np.float32)])))
        a = c.state_norm.normalize(t(window)[:, self.steps, :])
        with torch.no_grad():
            return float(self.model.p_fail(self._qv.QInputs(state=st, action=a)).item())


class StateCritic:
    """A ``fit_q`` ABLATED checkpoint used as ``V(s)``, for the rollout selector.

    The ablated arm was trained with its action tokens zeroed, so it is a state critic that
    happens to be a :class:`FlowQModel`.  This exposes it behind the same two members the
    rollout selector reads from ``critic.FailureCritic`` -- ``privileged`` and ``p_fail`` --
    so the same loop can be driven by either.

    **The action is passed as ZEROS, matching training.**  Feeding it a real chunk would put
    the model off the distribution it was fitted on: it never saw a non-zero action, so
    whatever it does with one is unconstrained rather than merely unhelpful.

    Args:
        ckpt: An ablated ``fit_q`` checkpoint.
        device: Torch device.
        fm_override: Local generator ``.pt``, when the checkpoint names another machine's path.

    Raises:
        ValueError: If the checkpoint is NOT ablated.  A live-action Q used here would be
            scored with a zero action at every candidate, making it a state critic by
            accident and silently discarding the thing it was trained for.
    """

    def __init__(self, ckpt: str, device: str = "cuda", fm_override: str = "") -> None:
        import torch  # noqa: PLC0415

        k = torch.load(ckpt, map_location="cpu", weights_only=False)
        if not bool(k.get("ablate_action", False)):
            raise ValueError(
                f"{ckpt} is a LIVE-action Q, not the ablated control. Driving the rollout "
                "selector with it would pass a zero action at every candidate -- turning it "
                "into a state critic by accident and throwing away exactly what it was "
                "trained to add. Use QScorer for a live Q.")
        # Built through a loader that skips QScorer's ablation guard: that guard protects the
        # Q selector, where an ablated model would rank every candidate the same. Here the
        # ablation is the POINT.
        self._q = _load_q_unchecked(ckpt, device, fm_override)
        self.privileged = True
        self._torch = torch

    def p_fail(self, inp) -> float:
        """``P(fail | s)`` with the action held at zero.

        Args:
            inp: A :class:`critic.CriticInputs`.

        Returns:
            The score.
        """
        import numpy as _np  # noqa: PLC0415

        zero = _np.zeros((64, self._q.model.action_in[0].in_features), dtype=_np.float32)
        return self._q.score(inp, zero)


def _load_q_unchecked(ckpt: str, device: str, fm_override: str):
    """A :class:`QScorer` built without the ablation guard.

    Args:
        ckpt: The checkpoint.
        device: Torch device.
        fm_override: Local generator path.

    Returns:
        The scorer.
    """
    import copy as _copy  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from robot_rl.flow_matching import online, qvalue  # noqa: PLC0415

    o = QScorer.__new__(QScorer)
    k = torch.load(ckpt, map_location="cpu", weights_only=False)
    fm = fm_override or str(k["fm_checkpoint"])
    o._torch, o._qv, o.device = torch, qvalue, device
    o.ctx = online.load_fm_context(fm, device=device)
    o.model = qvalue.FlowQModel(
        _copy.deepcopy(o.ctx.model), action_dim=int(k["action_dim"]),
        action_steps=int(k["action_steps"]), scan_hw=tuple(k["scan_hw"]),
        proprio_dim=int(k["proprio_hw"][-1]), proprio_steps=int(k["proprio_hw"][0]),
        context_dim=int(k["context_dim"])).to(device)
    o.model.load_state_dict(k["model"])
    o.model.eval()
    o.scan_hw = tuple(k["scan_hw"])
    o.steps = torch.as_tensor(o.ctx.cmd_steps, dtype=torch.long, device=device)
    return o


def run_episode_q_selected(rig, policy, fm_lag, spawn, cmd, cfg, scorer, anchor, *,
                           k: int, seed: int, mjrunner):
    """One episode where each re-plan picks the best of ``k`` chunks by ``Q(s, a)``.

    Unlike :func:`run_episode_selected`, **no candidate is executed to be scored**.  At a
    re-plan the generator is sampled ``k`` times against the SAME conditioning, each sample is
    scored where it stands, and only the winner is committed.  Cost is ``k`` generator calls,
    not ``k`` chunk rollouts.

    The state half of ``Q``'s input is built ONCE and reused across candidates: it is the
    conditioning at this re-plan, which every candidate shares by construction.  Rebuilding it
    per candidate would additionally re-cast depth, which mutates the lag ring -- the bug that
    made the V selector look worse than no selection at all.

    Args:
        rig: The course rig.
        policy: The loaded pair.
        fm_lag: The camera-lag ring.
        spawn: Initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        scorer: A :class:`QScorer`, or ``None`` for ``k == 1``.
        anchor: Goal anchor for the goal features.
        k: Candidates per re-plan.
        seed: Episode seed.
        mjrunner: The ``runner`` module.

    Returns:
        ``(verdict, failure, n_replans, mean_spread)`` where ``mean_spread`` is the mean
        max-minus-min of ``Q`` across candidates -- if that is ~0 the model is not
        discriminating and any success change is not selection.
    """
    import critic as C  # noqa: PLC0415
    import mujoco  # noqa: PLC0415

    from gen_tracker_runner import build_state  # noqa: PLC0415

    robot, course = rig.robot, rig.course
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)
    ring = C.ProprioRing()
    z0 = float(robot.mj_data.qpos[2])
    zex = [z0, z0]
    n_replans, spreads = 0, []
    last_plan_id = -1

    while robot.mj_data.time <= course.budget_s:
        state = build_state(robot, policy, cmd, rig.depth_cams, want_depth_hits=False,
                            fm_depth_cams=rig.fm_depth_cams, fm_lag=fm_lag)
        ring.push(state["joint_pos"], state["joint_vel"], state["base_ang_vel"],
                  C.projected_gravity_b(state["root_quat"]))
        # Snapshot BEFORE create_obs. `_ctrl_step` is the re-plan cadence counter, so a
        # snapshot taken AFTER it has already advanced restores to MID-CHUNK -- and a
        # create_obs there does not re-plan, it consumes the plan already made. The
        # "candidates" are then different offsets into ONE plan rather than different plans:
        # their windows genuinely differ, so nothing looks wrong, but plan_id never
        # increments and the selector is choosing among readings of the same decision.
        pre = Snapshot(robot, policy, fm_lag)
        obs = policy.create_obs(state)
        plan = policy.last_plan()
        is_replan = plan is not None and int(plan["plan_id"]) != last_plan_id

        if is_replan and k > 1 and scorer is not None:
            last_plan_id = int(plan["plan_id"])
            snap = pre
            # Built once: this is the conditioning at THIS re-plan, shared by every candidate.
            inp = C.CriticInputs.from_runtime(policy, robot, state, anchor=anchor, ring=ring,
                                              want_scan=True)
            w0 = policy.predicted_window()
            cands = [(scorer.score(inp, w0), 0)] if w0 is not None else []
            for c in range(1, k):
                snap.restore(robot, policy, fm_lag)
                policy._rng = np.random.default_rng([seed, n_replans, c])
                policy.create_obs(state)
                w = policy.predicted_window()
                if w is not None:
                    cands.append((scorer.score(inp, w), c))
            if cands:
                sc = [s for s, _ in cands]
                spreads.append(max(sc) - min(sc))
                best_c = min(cands)[1]
                snap.restore(robot, policy, fm_lag)
                # Candidate 0 used whatever rng the episode had reached; reproduce that by
                # restoring rather than re-seeding, so the committed plan is the one scored.
                if best_c:
                    policy._rng = np.random.default_rng([seed, n_replans, best_c])
                obs = policy.create_obs(state)
            n_replans += 1
        elif is_replan:
            last_plan_id = int(plan["plan_id"])
            n_replans += 1

        robot.apply_action(policy.get_action(obs, robot.joint_names))
        if robot.failed():
            return "fall", robot.failure_reason(), n_replans, float(np.mean(spreads or [0.0]))
        for _ in range(rig.decim):
            robot.step()
        z = float(robot.mj_data.qpos[2])
        zex[0], zex[1] = max(zex[0], z), min(zex[1], z)
        if arrived(course, robot, zex):
            return "success", "", n_replans, float(np.mean(spreads or [0.0]))
    return "timeout", "", n_replans, float(np.mean(spreads or [0.0]))


if __name__ == "__main__":
    raise SystemExit(main())
