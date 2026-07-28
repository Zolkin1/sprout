#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage 1 (Motion Bricks env): generate the APPROACH + EXIT segments around a planned terrain traversal.

One driver for every terrain family.  It reads the :class:`~motion_gen.motionbricks.terrain_plan.TerrainPlan`
written by ``mb_terrain_clips.py --dump-plan`` (which already placed the replay on the tile and picked its
entry / exit frames) and generates the two flat segments that bracket it -- the terrain half of the
keyframe in-betweening:

* **approach** -- steady locomotion (or a standing settle, on a box) that ramps to the replay's speed at the
  entry frame.  With ``--approach-pin hard`` the target keyframe is switched from our optimized steady
  stride to the REPLAY'S OWN FRAMES once the end of the approach comes inside Motion Bricks' planning
  horizon, so the generated motion in-betweens into the entry pose (and, on stairs, starts rising with it)
  and its last frame can be welded to the entry with no ease.  With ``soft`` the approach stays a free
  steady walk and the seam is eased when the clip is spliced.
* **exit** -- with ``--exit-pin hard``, Motion Bricks is CONTEXT-SEEDED with the replay's last frames, so
  the walk-away starts ON the terrain and is continuous with it by construction; the command ramps from the
  replay's exit speed back up to the steady walk-away.  With ``soft`` it is a free steady walk, eased in.

Both segments are written as ordinary :class:`~motion_gen.motionbricks.raw_schema.RawMotion` files (with the
pin mode in their provenance, so pass 2 splices them the way they were generated).  Runs in the
``motionbricks`` conda env; paths may be repo-relative (the harness resolves them before it has to ``chdir``
into the Motion Bricks root, whose configs reference assets relatively)::

    conda run -n motionbricks python scripts/rsl_rl/mb_terrain_segments.py \
        --mb-root ../GR00T-WholeBodyControl/motionbricks --stride-bank mb_raw/stride_bank_v2.npz \
        --plan mb_raw/plan_stairs_up.npz --out-dir mb_raw/seg_stairs_up
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_env  # noqa: E402


# Width (m) over which the stride pose pin relaxes as the boundary comes into range, so the hand-over
# to the terrain pin is a blend rather than a switch.
_PIN_BLEND_M = 1.0

# Seconds a STAND start spends getting into gait before it covers meaningful ground (measured on a
# jump_on run-up: 0.30 m in the first 1.57 s).  Discounted from the anchor distance so the boundary is
# placed somewhere the robot can actually reach.
_ACCEL_TAX_S = 1.1


def _ramp(v0: float, v1: float, frac: float) -> float:
    """Linear blend ``v0 -> v1`` at ``frac`` in ``[0, 1]`` (clamped)."""
    a = min(max(float(frac), 0.0), 1.0)
    return float(v0 + (v1 - v0) * a)


def _generate_approach(mb: mb_env.MBAgent, targeter: mb_env.StrideTargeter, plan, *,
                       pin: str, pin_future: bool, ramp_s: float, secs: float, seed: int,
                       pin_lead: float | None = None, pin_tail: float = 0.5,
                       start: str = "stand", place_pin: bool = True, hold_cmd: bool = False,
                       pin_dist: float = 2.0):
    """The approach segment: steady locomotion into the terrain entry pose.

    The command holds ``plan.approach_vx`` and ramps to ``plan.entry_vx`` over the last ``ramp_s`` seconds
    (the replay's own entry speed), so the hand-off is not a velocity step.  Under a HARD pin the target
    pose switches from our steady stride to the REPLAY ITSELF for the last ``pin_lead`` seconds
    (:meth:`TerrainPlan.approach_pin_window` reads the replay at the model's lookahead times), so the model
    in-betweens toward the entry pose instead of walking flat into it.  Two details matter:

    * The pin is a target at a ~1 s LOOKAHEAD, so it asks "be at the terrain pose one second from now".
      Once the approach's end is inside that horizon the target walks forward into the replay's own future
      (``pin_future``) -- which is what keeps the arrival MOVING at the entry velocity rather than settling
      onto a frozen keyframe, and what an ASCENT wants.  Otherwise the target is HELD at the entry pose,
      which is what a DESCENT (whose future drops below the platform) and an episodic boundary (whose future
      is the takeoff itself) want -- see ``terrain_families.pin_future`` for the measured comparison.
    * The model passes NEAR the pinned pose rather than landing on it at an exact frame, so ``pin_tail``
      extra seconds are generated past the nominal arrival and the splice picks the best-matching frame
      (:func:`~motion_gen.motionbricks.terrain_splice.match_arrival`).

    ``start`` picks what the approach BEGINS from:

    * ``"stand"`` -- the agent's reset (standing) state, so the clip opens with an acceleration run-up.
    * ``"steady"`` -- CONTEXT-SEEDED from the matched steady stride (the same priming the exit uses), so the
      robot is already travelling at the approach command on frame 0. This is the deployment case -- the
      robot reaches terrain while walking -- and it stops the run-up from eating the clip at high speed.

    Both are wanted in a library: a stand start teaches the approach-from-rest, a steady start teaches the
    arrival at speed. The caller decides per clip (``mb_plan_library --steady-approach-frac``).

    Returns ``(qpos[T,36], cmd[T,3])``.
    """
    mb.reset(seed)
    horizon = float(mb.lookahead_times[0])
    lead = horizon if pin_lead is None else float(pin_lead)
    n_steps = max(2, int(round((secs + max(0.0, pin_tail)) * mb.fps)))
    standing = plan.approach_mode == "stand"
    # STEADY start: prime the buffer from a stride already moving at the approach command, so frame 0 is
    # mid-gait rather than a stand.  The bank's strides are feet-at-zero at the MB frame rate, which is the
    # frame every MB segment is generated in, so the seed needs no shifting (unlike the exit's, which comes
    # off the terrain and is lowered by its surface height).  A STANDING approach (settling on a box top)
    # has no steady state to seed from.
    if start == "steady" and not standing:
        cmd0 = np.array([float(plan.approach_vx), 0.0, 0.0], np.float32)
        win = targeter.window(targeter.nearest(cmd0), advance=1)
        mb.set_target_pose(win)
        mb.set_regime(1.0, 0.0)
        seed_ctx = t.as_tensor(np.asarray(win, np.float32), dtype=t.float32,
                               device=mb.agent._device)[None]
        mb.prime(mb_env.walk_control(seed_ctx, cmd0, mb.lookahead_times))
    # WHERE the entry is, fixed once from the plan: this clip's share of the run-up the tile reserves, so
    # the boundary sits that far ahead of wherever the segment starts.  A place, not a moving carrot.
    #
    # A STAND start spends its first ~_ACCEL_TAX_S seconds getting into gait and covers almost no ground
    # (measured: 0.30 m of the first 1.57 s), so the nominal ``vx * secs`` is further than the robot can
    # reach and it just sprints at an anchor it never arrives at -- which is how a 3 m/s jump_on run-up
    # produced a 3.97 m/s seam.  Discount it.  A steady start is already at speed on frame 0 and needs no
    # discount (measured: 3.15-3.18 m travelled against a 3.21 m anchor).
    cx0, cy0, _cz0, cyaw0 = mb_env.cur_pose(mb.context())
    reach_s = float(secs) - (_ACCEL_TAX_S if start == "stand" and not standing else 0.0)
    d0 = float(plan.approach_vx) * max(0.2, reach_s)
    entry_world = (cx0 + math.cos(cyaw0) * d0, cy0 + math.sin(cyaw0) * d0, cyaw0)
    e_xy = plan.terrain_qpos[plan.entry_frame, 0:2]          # the entry in TILE coords, for the offsets below
    c0, s0 = math.cos(cyaw0), math.sin(cyaw0)

    def _target_at(t_offset: float):
        """The boundary as a world pose, advanced ``t_offset`` s into the replay for a pin_future family.

        A ``pin_future`` approach tracks the replay's FUTURE frames as its target POSE, so its target PLACE
        has to advance with them -- otherwise the two disagree ("be at a frame further up the stairs, at
        the entry's position"), which collapses stairs_up onto its measured "hold" behaviour (1.04 rad vs
        0.66 for a proper future pin).  For a HOLD family the offset is 0 and this is just ``entry_world``.
        """
        if t_offset <= 0.0:
            return entry_world
        idx = np.array([plan.entry_frame + t_offset / float(plan.dt)])
        r = mb_env.load_schema_module('terrain_plan').sample_qpos(plan.terrain_qpos, idx)[0]
        dx, dy = float(r[0] - e_xy[0]), float(r[1] - e_xy[1])
        return (entry_world[0] + c0 * dx - s0 * dy, entry_world[1] + s0 * dx + c0 * dy, cyaw0)

    qpos_stream, cmd_stream = [], []
    pin_on = False
    for step in range(n_steps):
        t_now = step * mb.dt
        t_left = secs - t_now
        # With the boundary pinned as a PLACE, the ramp is the redundant half: the target no longer
        # travels with the robot, so the gap -- and with it the speed -- closes because the robot
        # arrives.  ``hold_cmd`` drops the ramp and lets the geometry do it.
        vx = 0.0 if standing else (float(plan.approach_vx) if hold_cmd else
                                   _ramp(plan.approach_vx, plan.entry_vx,
                                         1.0 - t_left / max(1e-6, ramp_s)))
        cmd = np.array([vx, 0.0, 0.0], np.float32)
        # Hand over to the terrain pin on PROXIMITY, not on a stopwatch: the run-up should be driven by our
        # steady stride until the boundary is actually close.  ``pin_lead`` stays as a floor so the pin
        # always engages for the last stretch even if the robot never closes the distance (it stalls, or
        # the placement over-reached).  Latched -- once handed over, never hand back.
        cx, cy, _cz, _cyaw = mb_env.cur_pose(mb.context())
        dist = math.hypot(entry_world[0] - cx, entry_world[1] - cy)
        pin_on = pin_on or (dist <= pin_dist) or (t_left <= lead)
        if pin == "hard" and pin_on:
            # ``t_offset`` is where the FIRST lookahead frame falls relative to the entry: negative while
            # the arrival is still beyond the horizon (clamped to the entry pose), then walking forward into
            # the replay once it is inside.
            t_offset = horizon - t_left
            if not pin_future:
                t_offset = min(0.0, t_offset)
            # ``pin_future=False`` means HOLD the entry, so pin exactly that one keyframe -- clamping
            # t_offset alone left the window's tail sliding forward into the replay (see
            # TerrainPlan.approach_pin_window).
            mb.set_target_pose(plan.approach_pin_window(t_offset, mb.nfpt, mb.dt, hold=not pin_future),
                               kind="terrain")
            mb.set_regime(1.0, t_now)                      # keep the pose pin ON through the hand-off
        else:
            mb.set_target_pose(targeter.window(targeter.nearest(cmd), advance=1))
            # Relax the stride pose pin as the boundary comes within reach, so the model blends into the
            # terrain pin instead of switching to it cold.  Same smoothstep as before, just measured in
            # METRES now that the hand-over is a distance -- full pin a blend-width out, released at the
            # switch.
            mb.set_regime(mb_env.regime_weight([pin_dist], dist, ramp_s=_PIN_BLEND_M), t_now)
        qpos = mb.next_frame()
        # Under the pin, aim at the boundary's PLACE; before it, the ordinary command rollout.
        tgt = _target_at(t_offset if (pin == 'hard' and pin_on and pin_future) else 0.0) \
            if (pin == "hard" and pin_on and place_pin) else None
        mb.step(mb_env.walk_control(mb.context(), cmd, mb.lookahead_times,
                                    standing=standing or None, target=tgt))
        qpos_stream.append(qpos)
        cmd_stream.append(cmd)
    return np.stack(qpos_stream, 0), np.stack(cmd_stream, 0)


def _generate_exit(mb: mb_env.MBAgent, targeter: mb_env.StrideTargeter, plan, *,
                   pin: str, ramp_s: float, secs: float, seed: int):
    """The exit segment: locomotion away from the terrain exit pose (or a standing hold, on a box top).

    Under a HARD pin the agent is CONTEXT-SEEDED with the replay's last frames
    (:meth:`TerrainPlan.exit_seed_window`), so the walk-away continues FROM the terrain rather than being
    stitched onto it.  The seed is translated so the exit SURFACE sits at z = 0 (see below), which is also
    the frame every other Motion Bricks segment lives in, so the splice grounds / welds it the same way.
    Under a SOFT pin there is no seed at all -- an ordinary steady flat segment, placed and eased.

    Motion Bricks' height prior is ABSOLUTE (it generates for a floor at z = 0), so seeding it with a pose
    1.2 m up on a box makes the model walk back DOWN to its ~0.78 m pelvis height -- the walk-away sinks
    through the platform (measured: -0.12 m over 2.5 s on a stair top platform).  Shifting the seed down by
    the surface height is a rigid translation of the history, so continuity with the replay is untouched,
    but the prior is now correct and the segment stays on its (virtual) ground.

    The command ramps from the replay's signed exit speed up to ``plan.exit_vx`` over ``ramp_s`` seconds
    (the mirror of the approach ramp), so the walk-away ends up tracking the steady command the RL sees.

    Returns ``(qpos[T,36], cmd[T,3])``.
    """
    mb.reset(seed)
    n_steps = max(2, int(round(secs * mb.fps)))
    standing = plan.exit_mode == "stand"

    def _cmd_at(t_now: float) -> np.ndarray:
        vx = 0.0 if standing else _ramp(plan.exit_speed, plan.exit_vx, t_now / max(1e-6, ramp_s))
        return np.array([vx, 0.0, 0.0], np.float32)

    def _pose_for(cmd: np.ndarray) -> np.ndarray:
        return targeter.window(targeter.nearest(cmd), advance=1)

    if pin == "hard":
        # PRIME: force one generation whose context IS the replay's exit window (surface -> z=0) -> the
        # buffer becomes the walk-away continuing from the terrain (``next_frame`` then pops those frames).
        win = plan.exit_seed_window(mb.nfpt, mb.dt)
        win[:, 2] -= float(plan.exit_surf_z)
        seed_ctx = t.as_tensor(win, dtype=t.float32, device=mb.agent._device)[None]
        cmd0 = _cmd_at(0.0)
        mb.set_target_pose(_pose_for(cmd0), kind="stride")
        mb.set_regime(1.0, 0.0)
        mb.prime(mb_env.walk_control(seed_ctx, cmd0, mb.lookahead_times, standing=standing or None))

    qpos_stream, cmd_stream = [], []
    for step in range(n_steps):
        t_now = step * mb.dt
        cmd = _cmd_at(t_now)
        # ALWAYS the stride: unlike the approach, the exit never pins a terrain pose -- it is
        # context-SEEDED from the replay instead, so continuity is in the history, not the target.
        mb.set_target_pose(_pose_for(cmd), kind="stride")
        mb.set_regime(mb_env.regime_weight([ramp_s], t_now), t_now)
        qpos = mb.next_frame()
        mb.step(mb_env.walk_control(mb.context(), cmd, mb.lookahead_times, standing=standing or None))
        qpos_stream.append(qpos)
        cmd_stream.append(cmd)
    return np.stack(qpos_stream, 0), np.stack(cmd_stream, 0)


def _emit_segment(mb, targeter, plan, family, side: str, pin: str, secs: float, out: str,
                  args, raw_schema, quiet: bool = False, plan_path: str | None = None,
                  start: str = "stand") -> str:
    """Generate ONE segment (``side`` of ``plan``) and write it as a ``RawMotion``; returns the path."""
    seed = (args.seed * (1 + (side == "exit")) * 2333) % (2 ** 32 - 1)
    if side == "approach":
        pin_future = family.pin_future if args.pin_future == "auto" else (args.pin_future == "on")
        qpos, cmd = _generate_approach(mb, targeter, plan, pin=pin, pin_future=pin_future,
                                       ramp_s=args.ramp_s, secs=secs, seed=seed,
                                       pin_lead=args.pin_lead, pin_tail=args.pin_tail, start=start,
                                       place_pin=(not family.periodic if args.pin_place == 'auto'
                                                  else args.pin_place == 'terrain'),
                                       hold_cmd=(args.pin_cmd == 'hold'),
                                       pin_dist=args.pin_dist)
    else:
        qpos, cmd = _generate_exit(mb, targeter, plan, pin=pin, ramp_s=args.ramp_s, secs=secs, seed=seed)
    motion = raw_schema.RawMotion(
        mujoco_qpos=qpos, fps=mb.fps, joint_names=list(mb_env.G1_JOINT_NAMES),
        target_vel=cmd, kind="terrain",
        provenance={"backend": "motionbricks", "mode": f"terrain_{side}", "pin": pin,
                    # Every segment is generated with its walking surface at z = 0 ("flat"), so the
                    # splice grounds / welds both sides identically; ``seeded`` says whether Motion
                    # Bricks was context-seeded from the terrain (its output then opens by re-predicting
                    # the seed frames, which the splice skips).
                    "frame": "flat", "seeded": bool(side == "exit" and pin == "hard"),
                    "start": start if side == "approach" else "",
                    "skill": plan.skill, "seed": int(seed),
                    "plan": os.path.abspath(plan_path) if plan_path else "",
                    # The full set of knobs that shape this motion -- ``_gen_params`` compares them on
                    # resume so a settings change regenerates instead of silently reusing a stale segment
                    # (they are NOT in the request key, which only describes WHAT to reach, not HOW).
                    "secs": secs, "ramp_s": args.ramp_s, "pin_future": args.pin_future,
                    "pin_lead": args.pin_lead, "pin_tail": args.pin_tail,
                    "pin_place": args.pin_place, "pin_cmd": args.pin_cmd,
                    "pin_dist": args.pin_dist,
                    "stride_bank": os.path.abspath(args.stride_bank),
                    "mb_root": os.path.abspath(args.mb_root)},
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    raw_schema.save_raw_motion(out, motion)
    if not quiet:
        z = qpos[:, 2]
        boundary = "last" if side == "approach" else "first"
        bz = float(z[-1] if side == "approach" else z[0])
        print(f"[mb_seg]   {side:8s} pin={pin:4s} {qpos.shape[0]}f ({secs:.1f}s) cmd "
              f"{cmd[0, 0]:+.2f} -> {cmd[-1, 0]:+.2f} m/s  pelvis_z {z[0]:.2f} -> {z[-1]:.2f} "
              f"({boundary} frame z={bz:.2f})  -> {os.path.basename(out)}")
    return out


# Generation knobs that change the MOTION for the same request. The key deliberately does not include
# them (it says which boundary pose to reach and at what command), so resume compares them explicitly.
_GEN_KNOBS = ("ramp_s", "pin_future", "pin_lead", "pin_tail", "start", "secs", "pin", "pin_place",
              "pin_cmd", "pin_dist")


def _gen_params(args, side: str, pin: str, secs: float, start: str) -> dict:
    """The generation knobs this run would use for one segment."""
    return {"ramp_s": args.ramp_s, "pin_future": args.pin_future, "pin_lead": args.pin_lead,
            "pin_tail": args.pin_tail, "start": start if side == "approach" else "",
            "secs": float(secs), "pin": pin, "pin_place": args.pin_place,
            "pin_cmd": args.pin_cmd, "pin_dist": args.pin_dist}


def _is_stale(path: str, want: dict, raw_schema) -> str | None:
    """``None`` if the segment on disk was generated with ``want``; else the first differing knob.

    Without this, changing e.g. ``--pin-lead`` would leave every already-generated segment in place: the
    request key is unchanged, so resume would treat stale motion as done.
    """
    try:
        prov = dict(raw_schema.load_raw_motion(path).provenance or {})
    except Exception:
        return "unreadable"
    for k in _GEN_KNOBS:
        if k not in prov:
            return f"{k} (not recorded)"
        a, b = prov[k], want[k]
        if isinstance(b, float) and isinstance(a, (int, float)):
            if abs(float(a) - b) > 1e-9:
                return f"{k}: {a} != {b}"
        elif a != b:
            return f"{k}: {a!r} != {b!r}"
    return None


def _run_batch(args, mb, targeter, raw_schema, plan_mod, families) -> int:
    """BATCH mode: generate every unique segment in ``requests.json``, in this one process.

    Each request names a representative plan + which side to generate; plans are cached because many
    requests share one. A segment whose ``.npz`` already exists is skipped, so an interrupted pass resumes,
    and a failure is recorded per request instead of taking down the run.
    """
    work = args.work_dir or os.path.dirname(os.path.abspath(args.requests))
    with open(args.requests) as f:
        reqs = json.load(f)["requests"]
    plans: dict[str, object] = {}
    done = fail = skipped = stale = 0
    print(f"[mb_seg] batch: {len(reqs)} unique segments (work-dir {work})")
    for i, r in enumerate(reqs):
        out = os.path.join(work, r["out"])
        try:
            ppath = os.path.join(work, r["plan"])
            if ppath not in plans:
                plans[ppath] = plan_mod.load_terrain_plan(ppath)
            plan = plans[ppath]
            family = families.family_for(plan.skill)
            secs = float(plan.approach_s if r["side"] == "approach" else plan.exit_s)
            if os.path.exists(out):
                why = _is_stale(out, _gen_params(args, r["side"], r["pin"], secs,
                                                 r.get("start", "stand")), raw_schema)
                if why is None:
                    skipped += 1
                    continue
                if stale == 0:
                    print(f"[mb_seg] regenerating: on-disk segments differ ({why})", flush=True)
                stale += 1
            _emit_segment(mb, targeter, plan, family, r["side"], r["pin"], secs, out, args,
                          raw_schema, quiet=True, plan_path=ppath, start=r.get("start", "stand"))
            done += 1
        except (Exception, SystemExit) as e:     # one bad segment costs one clip, not the library
            fail += 1
            print(f"[mb_seg][FAIL] {r['side']} {r['key'][:8]} ({r.get('plan')}): {type(e).__name__}: {e}")
        if (i + 1) % 25 == 0 or i + 1 == len(reqs):
            print(f"[mb_seg]   {i + 1}/{len(reqs)}  generated={done} reused={skipped} failed={fail}",
                  flush=True)
    print(f"[mb_seg] batch done: {done} generated ({stale} of them stale + regenerated), "
          f"{skipped} reused, {fail} FAILED")
    return 1 if (fail and not done) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mb-root", required=True, help="GR00T motionbricks package root.")
    p.add_argument("--stride-bank", required=True, help="Flat stride bank .npz (export_stride_bank.py).")
    p.add_argument("--plan", default=None, help="Terrain plan .npz (mb_terrain_clips.py --dump-plan).")
    p.add_argument("--out-dir", default=None, help="Directory for the approach / exit RawMotion .npz files.")
    p.add_argument("--requests", default=None,
                   help="BATCH mode: requests.json from mb_plan_library.py. Generates every unique segment "
                        "the library needs in THIS process -- the whole point, since loading the model "
                        "dominates the cost of one segment. Paths inside it are relative to --work-dir.")
    p.add_argument("--work-dir", default=None,
                   help="[batch] Root the requests' plan/out paths resolve against (default: the "
                        "requests.json directory).")
    p.add_argument("--approach-pin", default="hard", choices=["hard", "soft"],
                   help="hard: in-between into the terrain entry pose (weld, no ease). "
                        "soft: free steady walk (placed + eased at splice time).")
    p.add_argument("--exit-pin", default="hard", choices=["hard", "soft"],
                   help="hard: context-seed from the terrain exit pose (weld, no ease). "
                        "soft: free steady walk (placed + eased at splice time).")
    p.add_argument("--approach-s", type=float, default=None, help="Override the plan's approach duration (s).")
    p.add_argument("--exit-s", type=float, default=None, help="Override the plan's exit duration (s).")
    p.add_argument("--ramp-s", type=float, default=0.8,
                   help="Command ramp between the steady command and the terrain boundary speed (s).")
    p.add_argument("--pin-lead", type=float, default=None,
                   help="[hard approach] how long before the arrival the terrain pose pin engages (s); "
                        "default = the model's planning horizon (~1 s). Longer = more replans under the pin.")
    p.add_argument("--pin-future", default="auto", choices=["auto", "on", "off"],
                   help="[hard approach] once the arrival is inside the planning horizon, track the replay's "
                        "own FUTURE frames (on) or hold the entry pose (off). auto = the family's own "
                        "setting (on for ascents only -- see terrain_families.pin_future for the measured "
                        "comparison).")
    p.add_argument("--approach-start", default="stand", choices=["stand", "steady"],
                   help="[single-plan] what the approach begins from: 'stand' (reset, run-up) or 'steady' "
                        "(context-seeded from the matched stride, already at the command). In batch mode "
                        "this comes per request from mb_plan_library --steady-approach-frac.")
    p.add_argument("--pin-place", default="auto", choices=["auto", "terrain", "command"],
                   help="[hard approach] WHERE the model is told the boundary is. 'terrain' pins the "
                        "entry as a fixed world pose taken from the plan (the boundary is a place). "
                        "'command' is the legacy behaviour: the target is re-derived every replan as "
                        "robot_pose + command * lookahead_horizon, so it slides along with the robot and "
                        "the terrain's real location never reaches the model. 'auto' (default) picks per "
                        "family -- EPISODIC interactions get 'terrain' because a box has one takeoff point "
                        "the approach must hit, PERIODIC ones get 'command' because a stair traversal "
                        "re-anchors onto whichever tread its entry frame lands on (boundary_candidates "
                        "picks that frame from several equally good ones), so a fixed position "
                        "over-constrains it. Measured on one clip per family, approach joint_res: "
                        "stairs_up 0.709 cmd / 0.846 terrain, stairs_down 0.563 / 0.945, "
                        "jump_on 0.496 terrain (2.198 before the reachable-anchor fix).")
    p.add_argument("--pin-dist", type=float, default=2.0,
                   help="[hard approach] hand over from the steady stride to the terrain pin once the "
                        "boundary is within this many metres. Proximity rather than a stopwatch: the "
                        "run-up should be driven by our optimized stride until the terrain is actually "
                        "close. --pin-lead remains a FLOOR, so the pin still engages for the last "
                        "stretch if the robot never closes the distance. Only meaningful with "
                        "--pin-place terrain, which is what makes the boundary a known location.")
    p.add_argument("--pin-cmd", default="hold", choices=["ramp", "hold"],
                   help="[hard approach] the command under the pin. 'ramp' decelerates to the entry "
                        "speed; 'hold' keeps the steady approach command and lets --pin-place terrain "
                        "supply the deceleration (the gap to a FIXED boundary closes on arrival).")
    p.add_argument("--pin-tail", type=float, default=1.0,
                   help="[hard approach] extra seconds generated past the nominal arrival, so pass 2 can pick "
                        "the frame that actually matches the entry pose. With --pin-place terrain the arrival "
                        "time is EMERGENT (the robot gets there when it gets there), so it needs room to land "
                        "and settle into the boundary pose: measured on a jump_on approach, 0.5 s arrives still "
                        "moving at 0.38 m/s against an entry speed of 0.10, while 1.0 s lands at 0.15. 2.0 s is "
                        "identical to 1.0, so it has converged. The extra frames cost generation time only -- "
                        "the splice cuts them at the matched arrival.")
    p.add_argument("--tag", default="", help="Suffix for the output filenames (e.g. the pin combination).")
    p.add_argument("--side", default="both", choices=["both", "approach", "exit"],
                   help="Generate only one side (e.g. to re-roll a seam without redoing the other).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--controller", default="random")          # unused; keeps build_demo_args happy
    p.add_argument("--allowed-mode", default=None)
    args = p.parse_args()

    if bool(args.requests) == bool(args.plan):
        raise SystemExit("pass exactly one of --plan (single) or --requests (batch)")
    if args.plan and not args.out_dir:
        raise SystemExit("--plan needs --out-dir")

    mb_env.enter_mb_root(args, ("stride_bank", "plan", "out_dir", "requests", "work_dir"))
    raw_schema = mb_env.load_schema_module("raw_schema")
    plan_mod = mb_env.load_schema_module("terrain_plan")
    families = mb_env.load_schema_module("terrain_families")

    strides, steady_vel, bank_meta = mb_env.load_bank(args.stride_bank)
    mb = mb_env.MBAgent(args, capture_keyframes=True)
    targeter = mb_env.StrideTargeter(strides, steady_vel, mb.nfpt)
    print(f"[mb_seg] bank: {len(strides)} strides (dt={bank_meta.get('dt')});  {mb.describe()}")
    if args.requests:
        return _run_batch(args, mb, targeter, raw_schema, plan_mod, families)

    plan = plan_mod.load_terrain_plan(args.plan)
    family = families.family_for(plan.skill)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[mb_seg] {plan.skill}: {family.blurb}")
    print(f"[mb_seg]   plan: replay {plan.replay.shape[0]}f, entry {plan.entry_frame} exit {plan.exit_frame}, "
          f"surfaces approach z={plan.approach_surf_z:.2f} exit z={plan.exit_surf_z:.2f}")

    approach_s = float(args.approach_s if args.approach_s is not None else plan.approach_s)
    exit_s = float(args.exit_s if args.exit_s is not None else plan.exit_s)
    tag = f"_{args.tag}" if args.tag else ""
    written = []
    for side in (["approach", "exit"] if args.side == "both" else [args.side]):
        pin = args.approach_pin if side == "approach" else args.exit_pin
        secs = approach_s if side == "approach" else exit_s
        written.append(_emit_segment(mb, targeter, plan, family, side, pin, secs,
                                     os.path.join(args.out_dir, f"{side}{tag}.npz"), args, raw_schema,
                                     plan_path=args.plan, start=args.approach_start))

    kf_t, kf_qpos, kf_kind = mb.captured_keyframes()
    sidecar = os.path.join(args.out_dir, "_keyframes.npz")
    np.savez_compressed(sidecar, clip_idx=np.zeros(len(kf_t), np.int32), t=kf_t, qpos=kf_qpos,
                        kind=np.array(json.dumps(list(kf_kind))),
                        joint_names=np.array(json.dumps(list(mb_env.G1_JOINT_NAMES))), fps=np.array(mb.fps))
    print(f"[mb_seg] done: {len(written)} segments -> {args.out_dir} "
          f"({len(kf_t)} pinned keyframes of the last segment -> {sidecar})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
