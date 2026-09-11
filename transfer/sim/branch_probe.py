#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Can the critic rank SIBLING chunks branched from one state?  Measured, not inferred.

Every number we have about this critic is an average over STATES: AUC 0.76-0.78 within cell,
ECE 0.05, lead time 5.8 s.  A selector needs something else entirely -- it has to order K
futures that are 0.24 s downstream of the SAME state, differing only by a flow-noise draw.
Those are different questions, and a good answer to the first implies nothing about the
second.  Best-of-K has now failed to pay across three interventions (the critic-read fix, the
quality term, and lookahead 2 and 4), every one of them aimed at a mechanism nobody had
measured.  This measures it.

**The design, and why each piece is there.**

* **K branches per probed state, each rolled to COMPLETION.**  Only a completed continuation
  gives a branch a realized outcome, and without one there is no counterfactual: the live
  selector executes exactly one sibling, which is why 80250 recorded re-plans could show the
  within-state spread and never whether the ordering was right.
* **RANDOM commit.**  The episode follows a uniformly chosen sibling, so the visited state
  distribution is not shaped by the critic being evaluated.  Committing the argmin would make
  every downstream state one the critic already liked.
* **COMMON RANDOM NUMBERS on the continuations.**  Each sibling's own chunk must differ -- that
  is the treatment -- but all siblings share one continuation noise key, so the comparison is
  paired and the flow draw stops being a nuisance variable.  The key deliberately carries
  ``_CONT_TAG`` and NOT ``cand``: a key that collided with a branch key would hand a sibling
  its own branch draw as its continuation.
* **A SECOND continuation seed on a subset**, because CRN pairs the siblings without removing
  label noise.  Without it, "the critic cannot rank them" and "the labels are noisy" are the
  same observation.
* **EVERY critic on hand scores the SAME branches.**  The dataset is critic-agnostic -- states,
  chunks, outcomes -- so extra models cost one forward pass each.  The two cross-fit folds
  disagreeing with each other is the NOISE FLOOR: a within-state spread of 0.009 means nothing
  until it is compared against how much two honest estimates of the same quantity differ.

**Each head is judged against its own target**, which is the whole point of logging both:

    survival head   ->  the strict fail/success of the completion rollout
    quality  head   ->  the realized normalized return G over that rollout, built by
                        `tracking_error.normalized_return` from the same per-chunk rewards, the
                        same gamma and the same absorbing-tail rule the fit trained on
    measured 0.24 s ->  logged separately, so "does that term carry signal or jitter" is
                        answerable rather than assumed

Comparing the quality head against episode success would judge it on a question it was never
asked; a windowed discounted tracking return and a binary outcome are different quantities.

**Two things to read before believing any output.**

1. **Report the realized between-sibling outcome spread FIRST.**  The siblings are drawn from
   identical conditioning, so they are similar by construction.  If their realized outcomes
   barely differ, no critic could rank them and the finding is about the GENERATOR, not the
   critic -- the selection premise itself is then what is wrong.  ``--stage report`` prints this
   above everything else for that reason.
2. **Power is asymmetric between the heads.**  The quality target is continuous, so every group
   informs it.  The survival target is binary, so only groups whose siblings actually disagree
   do -- probability ``1 - p^K - (1-p)^K``, maximised at ``p = 0.5``.  ``t6_s0.8`` was chosen
   for that reason (K=1 success 0.500 measured, so ~0.88 of groups informative); on a cell like
   ``t40_s0.8`` at 1.000 the survival half would return nothing at all and would look identical
   to a critic with no signal.  The informative-group count is printed beside every statistic.

The null is a WITHIN-GROUP permutation of the predictions across siblings.  That is a real
permutation test here -- it preserves the group structure and destroys only the within-group
association, which is the hypothesis.  Note this is NOT the global label shuffle used elsewhere
in this tree, which is vacuous: under a permutation of all labels ``E[AUC] = 0.5`` analytically
for any score vector, so it cannot fail and validates nothing.

Usage::

    python branch_probe.py --stage run --out logs/value/bprobe \\
        --tracker_onnx <...>/tracker.onnx --fm_onnx <...>_fm.onnx \\
        --clip_lib clips/g1_mb_v5_body_mode11_decor.h5 \\
        --critic V_priv_f0=logs/value/v4/V_priv.fold0.pt \\
        --critic V_priv_f1=logs/value/v4/V_priv.fold1.pt \\
        --tile 6 --speed 0.8 --k 4 --episodes 40 --probe_every 4
    python branch_probe.py --stage report --out logs/value/bprobe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: RNG-key tag for the shared continuation stream. Distinct from `live_select`'s
#: `_LOOKAHEAD_TAG` and `_SELECT_TAG`, and from any branch key, so a continuation can never
#: replay a branch's own draw.
_CONT_TAG = 90210

#: Tag for the SECOND continuation seed, which measures the label noise CRN leaves behind.
_CONT_TAG_B = 90211

#: Tag for the uniform commit draw. Its own literal so it cannot alias either tape.
_COMMIT_TAG = 77003


# ======================================================================================
# run
# ======================================================================================
def _probe_episode(rig, policy, fm_lag, spawn, cmd, cfg, crits, spec, anchor, *,
                   k, seed, mjrunner, cadence, probe_every, second_seed, same_noise,
                   rows):
    """One episode: branch at every ``probe_every``-th re-plan, roll each branch out.

    Args:
        rig: Course rig.
        policy: The loaded pair.
        fm_lag: Camera-lag ring.
        spawn: Initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        crits: ``{name: FailureCritic}``, all scoring the same branches.
        spec: ``TrackingErrorSpec`` for the probe and the return target.
        anchor: Goal anchor for the critic's goal features.
        k: Siblings per probed state.
        seed: Episode seed.
        mjrunner: ``mjeval.runner``.
        cadence: Control steps per chunk.
        probe_every: Probe one re-plan in this many.
        second_seed: Also roll a SECOND, independently-keyed continuation for every sibling
            of this group, so the group carries two independent realized outcomes per branch
            and any ranking statistic has a measurable ceiling.
        same_noise: BLOCKING CONTROL -- hand every sibling one draw, so they must be
            identical and no group can be informative.
        rows: Output list, appended in place.

    Returns:
        ``(verdict, n_replans, n_groups)``.
    """
    import critic as C  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    from live_select import Snapshot, _chunk  # noqa: PLC0415
    from mjeval import tracking  # noqa: PLC0415
    from deploy.tracking_error import normalized_return  # noqa: PLC0415

    robot, course = rig.robot, rig.course
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    # The SAME reset `live_select.run_episode_selected` performs, field for field: a clip
    # spawn is not at rest, so dropping the velocity half silently changes the initial
    # condition rather than erroring.
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    mjrunner._reset_episode_state(policy, fm_lag)
    policy._rng = np.random.default_rng(seed)

    ep_ring = C.ProprioRing() if any(c.privileged for c in crits.values()) else None

    def push(target):
        def _cb(_r, _p, state):
            if target is not None:
                target.push(state["joint_pos"], state["joint_vel"], state["base_ang_vel"],
                            C.projected_gravity_b(state["root_quat"]))
        return _cb

    hold = float(cfg.cmd_hold_s)
    # THE EPISODE'S OWN PROBE. Without it the non-probed chunks run with `probe=None`, so the
    # strict violation cannot terminate them while every branch chunk and every continuation
    # DOES terminate on it -- the episode then walks past states termination would have ended
    # at, which are exactly the states `fit_value`'s drop-after-latch removes from training
    # (7.49% of windows). That both puts the critic on off-distribution states and shapes
    # WHICH states get probed. It is the same rule asymmetry as the K=1-vs-K=4 defect fixed in
    # `live_select` (K=1 came back `fall 0.211, violation 0.000` against K=4's
    # `violation 0.255, fall 0.000`), reintroduced here by omission.
    ep_probe = tracking.TrackingProbe(spec, hold_until_s=hold)
    z0 = float(robot.mj_data.qpos[2])
    zex = [z0, z0]
    n_replans, n_groups = 0, 0

    def roll_out(tag_seed, start_replan, cz, ring, probe):
        """Roll to COMPLETION with no branching. Returns (verdict, chunk_rewards)."""
        r = start_replan
        while robot.mj_data.time <= course.budget_s:
            # ONE shared stream for every sibling: the key carries no `cand`.
            policy._rng = np.random.default_rng([seed, start_replan, r, tag_seed])
            v, _f, _st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, cz,
                                scorer=push(ring), hold_until=hold, probe=probe,
                                step0=r * cadence)
            r += 1
            if v:
                probe.finish()
                return v, list(probe.chunk_rewards)
        probe.finish()
        return "timeout", list(probe.chunk_rewards)

    while robot.mj_data.time <= course.budget_s:
        probing = (n_replans % max(int(probe_every), 1)) == 0
        if not probing:
            v, f, _st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, zex,
                               scorer=push(ep_ring), hold_until=hold, probe=ep_probe,
                               step0=n_replans * cadence)
            n_replans += 1
            if v:
                return v, n_replans, n_groups
            continue

        snap = Snapshot(robot, policy, fm_lag)
        group = []
        for cand in range(k):
            if cand:
                snap.restore(robot, policy, fm_lag)
            # THE TREATMENT: the branch chunk differs per sibling and nothing else does.
            # Under --same_noise every sibling is handed ONE key, so the siblings must come
            # out bit-identical and the report's informative-group count must be EXACTLY 0.
            # Any non-zero count proves the restore is not exact and invalidates every number
            # in the run. This is the arm the abandoned Isaac harness lacked -- it was retired
            # because its same-noise control matched its real signal.
            policy._rng = np.random.default_rng(
                [seed, n_replans] if same_noise else [seed, n_replans, cand])
            cz = list(zex)
            cring = C.ProprioRing() if ep_ring is not None else None
            if cring is not None and ep_ring is not None:
                cring.buf[:] = ep_ring.buf
            cprobe = tracking.TrackingProbe(spec, hold_until_s=hold)
            v, f, st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, cz,
                              scorer=push(cring), hold_until=hold, probe=cprobe,
                              step0=n_replans * cadence)
            post = Snapshot(robot, policy, fm_lag)
            cprobe.finish()
            r_exec = (float(np.mean(cprobe.chunk_rewards))
                      if cprobe.chunk_rewards else float("nan"))
            # THE CRITIC READ, at the re-plan: one more control step reaches step 0 of the
            # next chunk, where `last_plan()` carries this sibling's own executed tail.
            preds = {}
            if not v and st is not None:
                _sv, _sf, st_rp = _chunk(rig, policy, fm_lag, cmd, 1, cfg, list(cz),
                                         scorer=push(cring), hold_until=hold,
                                         step0=(n_replans + 1) * cadence)
                sc = st_rp if st_rp is not None else st
                for nm, cr in crits.items():
                    x = C.CriticInputs.from_runtime(policy, robot, sc, anchor=anchor,
                                                    ring=cring, want_scan=cr.privileged)
                    p, g = cr.read(x)
                    preds[nm] = (p, g)
            # THE LABEL: roll this sibling to completion, on the SHARED continuation stream.
            post.restore(robot, policy, fm_lag)
            cont_probe = tracking.TrackingProbe(spec, hold_until_s=hold)
            cverd, crew = ("", []) if v else roll_out(
                _CONT_TAG, n_replans + 1, list(cz),
                (C.ProprioRing() if cring is not None else None), cont_probe)
            # A BRANCH THAT TERMINATED INSIDE ITS OWN CHUNK GETS g_real = NaN, not a number.
            # `crew` would then be its single own chunk reward, so `normalized_return` returns
            # (1-gamma)*R ~ 0.04 against a survivor's 0.5-0.9 -- two different random variables
            # in one column. That contaminated the "read this first" spread with a definition
            # change rather than generator behaviour, and made `r_exec vs realized G` partly
            # CIRCULAR, since for those rows g_real is exactly 0.05 * r_exec.
            failed = (v != "") or (cverd != "success")
            g_real = float("nan")
            if v == "" and crew:
                g_real = normalized_return(crew, spec, failed=failed)
            row = {"seed": seed, "replan": n_replans, "cand": cand,
                   "branch_verdict": v, "cont_verdict": (v or cverd),
                   "failed": bool(failed), "g_real": float(g_real),
                   "terminal_in_branch": bool(v != ""),
                   "r_exec": float(r_exec), "n_cont_chunks": len(crew)}
            for nm, (p, g) in preds.items():
                row["p_" + nm] = float(p)
                row["g_" + nm] = float("nan") if g is None else float(g)
            # A SECOND continuation for EVERY sibling, to size the label noise CRN leaves.
            #
            # It used to run on sibling 0 alone, which gives a per-branch noise MAGNITUDE but
            # no ceiling: the quantity every ranking claim has to be read against is how well
            # the group's realized outcomes rank the SAME group under an independent
            # continuation draw, and that needs a second label on every sibling, not on one.
            # Without it "the critic ranks siblings at tau = 0.1" and "no ranking of these
            # siblings is stable at all" are the same observation.
            #
            # It costs the second-seed groups K continuations instead of 1;
            # `--second_seed_every` is what keeps that affordable, and a group either has the
            # full set or none, never a partial one that would silently be a different
            # estimator.
            if second_seed and not v:
                post.restore(robot, policy, fm_lag)
                p2 = tracking.TrackingProbe(spec, hold_until_s=hold)
                v2, r2 = roll_out(_CONT_TAG_B, n_replans + 1, list(cz),
                                  (C.ProprioRing() if cring is not None else None), p2)
                row["cont_verdict_b"] = v2
                row["failed_b"] = bool(v2 != "success")
                row["g_real_b"] = float(normalized_return(r2, spec, failed=v2 != "success")
                                        if r2 else float("nan"))
            group.append((row, post, v, cz, cring))
        n_groups += 1
        rows.extend(g[0] for g in group)

        # RANDOM commit: its own stream, never `policy._rng`, so choosing cannot consume from
        # the sequence that produced the candidates.
        # Its OWN literal, not `_CONT_TAG + 1` -- which IS `_CONT_TAG_B`, and was safe only
        # because the two keys differ in length. One edit from aliasing the second-seed tape.
        pick = int(np.random.default_rng([seed, n_replans, _COMMIT_TAG]).integers(k))
        rows[-k + pick]["committed"] = True
        chosen = group[pick]
        chosen[1].restore(robot, policy, fm_lag)
        zex = list(chosen[3])
        if chosen[4] is not None and ep_ring is not None:
            ep_ring.buf[:] = chosen[4].buf
        n_replans += 1
        if chosen[2]:
            return chosen[2], n_replans, n_groups
    return "timeout", n_replans, n_groups


def stage_run(args) -> int:
    """Collect the branch dataset."""
    import h5py  # noqa: PLC0415
    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    import terrains  # noqa: PLC0415
    from fm_noise_attribution import build_arm  # noqa: PLC0415
    from deploy.tracking_error import TrackingErrorSpec  # noqa: PLC0415

    crits = {}
    for item in args.critic:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--critic wants NAME=path, got {item!r}")
        # A critic called `real` would write `g_real`, silently OVERWRITING the realized-return
        # LABEL with its own prediction -- after which the report scores g_real against g_real
        # and prints tau +1.0000, a perfect result that is pure aliasing. A name with `/`
        # creates an HDF5 group and breaks the reader instead.
        if name in ("real", "real_b") or not name.replace("_", "").isalnum():
            raise SystemExit(
                f"--critic name {name!r} is not allowed: it must be alphanumeric/underscore "
                "and must not be 'real' or 'real_b', which would collide with the g_real / "
                "g_real_b LABEL columns and score the label against itself.")
        crits[name] = C.FailureCritic(path, fm_pt=(args.fm_pt or None), device=args.device)
        print(f"[probe] critic {name}: {os.path.basename(path)} "
              f"privileged={crits[name].privileged} quality={crits[name].has_quality}",
              flush=True)
    if not crits:
        raise SystemExit("--critic is required; pass at least one NAME=path")

    spec = None
    stamped = [c.tracking_spec for c in crits.values() if c.tracking_spec is not None]
    if args.tracking_spec:
        with open(args.tracking_spec) as fh:
            spec = TrackingErrorSpec.from_dict(json.load(fh))
    elif stamped:
        spec = stamped[0]
    if spec is None:
        raise SystemExit(
            "no tracking spec: the quality head's target is a return under one set of sigmas, "
            "bounds and gamma, and the strict failure rule is the same object. Pass "
            "--tracking_spec, or use a critic whose checkpoint stamps one.")
    for c in crits.values():
        if c.tracking_spec is not None and c.tracking_spec != spec:
            raise SystemExit(f"critic {c.variant} was fitted under a DIFFERENT tracking spec "
                             "than the one in force; its G_hat is not on this scale.")
    print(f"[probe] spec {'FITTED' if not spec.is_bootstrap else 'BOOTSTRAP'}, "
          f"gamma {spec.gamma}", flush=True)

    course = terrains.isaac_tile_course(args.clip_lib, int(args.tile),
                                        speeds=(float(args.speed),),
                                        goal_anchor_max_offset=args.max_offset)
    import dataclasses  # noqa: PLC0415
    course = dataclasses.replace(course, name=f"tile{args.tile}{args.scene_tag}")
    cfg, policy, fm_lag, spawn, rig = build_arm(
        types.SimpleNamespace(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                              out=args.out, providers=args.providers,
                              fall_height=args.fall_height, fall_tilt_deg=args.fall_tilt_deg,
                              cmd_hold_s=terrains.CMD_HOLD_S), course)
    cadence = int(getattr(policy, "fm_update_every", 12))
    cmd = np.array([float(args.speed), 0.0, 0.0])
    print(f"[probe] {course.name} klass={course.klass} budget {course.budget_s:.1f}s "
          f"cadence {cadence}  probing every {args.probe_every} re-plan(s)", flush=True)

    rows: list[dict] = []
    import time  # noqa: PLC0415
    t0 = time.time()
    for ep in range(args.seed_offset, args.seed_offset + args.episodes):
        v, nr, ng = _probe_episode(
            rig, policy, fm_lag, spawn, cmd, cfg, crits, spec, course.goal_anchor_xyz,
            k=int(args.k), seed=ep, mjrunner=mjrunner, cadence=cadence,
            probe_every=int(args.probe_every), same_noise=bool(args.same_noise),
            second_seed=((ep - args.seed_offset) % max(int(args.second_seed_every), 1) == 0),
            rows=rows)
        print(f"  seed {ep:>3}: {v:8s} replans={nr:>3} groups={ng:>3} "
              f"rows={len(rows):>5}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if (ep - args.seed_offset + 1) % max(int(args.flush_every), 1) == 0:
            _write(args, rows, spec, crits)

    _write(args, rows, spec, crits)
    print(f"[probe] {len(rows)} branch rows -> {os.path.join(args.out, 'branches.h5')}",
          flush=True)
    return 0


def _write(args, rows, spec, crits) -> None:
    """Write the dataset. Called PERIODICALLY, not only at the end.

    A probe episode costs ~2 min and a run is tens of them, all held in memory until the last
    one -- so a crash at episode 39 of 40 used to lose the whole run, and ``--stage report``
    could not be pointed at a run in flight. Rewriting the file whole each time is fine at
    this scale (a few thousand small rows) and keeps the reader trivially correct.
    """
    import h5py  # noqa: PLC0415

    if not rows:
        return
    os.makedirs(args.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    path = os.path.join(args.out, "branches.h5")
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        for k in keys:
            vals = [r.get(k) for r in rows]
            if all(isinstance(v, (bool, np.bool_)) or v is None for v in vals):
                f.create_dataset(k, data=np.array([bool(v) for v in vals], dtype=np.uint8))
            elif any(isinstance(v, str) for v in vals):
                f.create_dataset(k, data=np.array([str(v or "") for v in vals],
                                                  dtype=h5py.string_dtype()))
            else:
                f.create_dataset(k, data=np.array([np.nan if v is None else float(v)
                                                   for v in vals], dtype=np.float64))
        f.attrs["tile"] = int(args.tile)
        f.attrs["speed"] = float(args.speed)
        f.attrs["k"] = int(args.k)
        f.attrs["probe_every"] = int(args.probe_every)
        f.attrs["episodes"] = int(args.episodes)
        f.attrs["critics"] = json.dumps(sorted(crits))
        f.attrs["same_noise"] = bool(args.same_noise)
        # PROVENANCE. "A default path to an artifact is a thing that rots quietly" -- and this
        # file's output decides whether best-of-K is abandoned, so the pair that produced it
        # travels in it.
        for key, val in (("tracker_onnx", args.tracker_onnx), ("fm_onnx", args.fm_onnx),
                         ("clip_lib", args.clip_lib), ("fm_pt", args.fm_pt),
                         ("critic_paths", json.dumps(sorted(args.critic))),
                         ("seed_offset", int(args.seed_offset)),
                         ("second_seed_every", int(args.second_seed_every)),
                         ("max_offset", int(args.max_offset)),
                         ("fall_height", float(args.fall_height)),
                         ("fall_tilt_deg", float(args.fall_tilt_deg))):
            f.attrs[key] = val
        f.attrs["tracking_spec"] = json.dumps(spec.to_dict())
    # Atomic replace, so a reader pointed at a run in flight never sees a half-written file.
    os.replace(tmp, path)


# ======================================================================================
# report
# ======================================================================================
def _within_group_tau(pred, real, gid, *, lower_is_better):
    """Mean within-group Kendall tau-b, ORIENTED so that positive always means "ranks well".

    ``lower_is_better`` is not a convenience. A raw concordance gives a good ``p_fail`` -1 and
    a good ``G_hat`` +1, because one is a risk and the other a quality -- two opposite sign
    conventions in one table, which is how a correct ranking gets read as a broken one. The
    flag folds that in so every tau in the report means the same thing.

    Args:
        pred: Per-branch prediction.
        real: Per-branch realized target.
        gid: Group id per branch.
        lower_is_better: True when a LOW prediction should mean a HIGH target (a risk score).

    Returns:
        ``(mean tau over informative groups, that group count)``.
    """
    sign = -1.0 if lower_is_better else 1.0
    taus = []
    for g in np.unique(gid):
        m = gid == g
        a, b = pred[m], real[m]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
            continue
        n_c = n_d = 0
        for i in range(a.size):
            for j in range(i + 1, a.size):
                s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                if s > 0:
                    n_c += 1
                elif s < 0:
                    n_d += 1
        if n_c + n_d:
            taus.append(sign * (n_c - n_d) / (n_c + n_d))
    return (float(np.mean(taus)) if taus else float("nan")), len(taus)


def _pick_rate(pred, real, gid, *, lower_is_better):
    """P(the argmin/argmax of `pred` is a best-in-group by `real`), and the group count."""
    hits, n = 0, 0
    for g in np.unique(gid):
        m = gid == g
        a, b = pred[m], real[m]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if a.size < 2 or np.all(b == b[0]):
            continue
        pick = int(np.argmin(a)) if lower_is_better else int(np.argmax(a))
        hits += int(b[pick] == b.max())
        n += 1
    return (hits / n if n else float("nan")), n


def stage_report(args) -> int:
    """Read the dataset and print the statistics, spread first."""
    import h5py  # noqa: PLC0415

    with h5py.File(os.path.join(args.out, "branches.h5"), "r") as f:
        d = {k: f[k][:] for k in f}
        attrs = dict(f.attrs)
    names = json.loads(attrs["critics"])
    k = int(attrs["k"])
    gid = d["seed"].astype(np.int64) * 100000 + d["replan"].astype(np.int64)
    n_groups = int(np.unique(gid).size)
    print(f"\n=== branch probe: tile {int(attrs['tile'])} @ {float(attrs['speed'])} m/s, "
          f"K={k}, {n_groups} groups over {int(attrs['episodes'])} episodes ===")

    # ---- 1. IS THERE ANYTHING TO RANK? -----------------------------------------------
    fail = d["failed"].astype(bool)
    greal = d["g_real"]
    mixed, gspread = 0, []
    for g in np.unique(gid):
        m = gid == g
        if fail[m].any() and not fail[m].all():
            mixed += 1
        gg = greal[m][np.isfinite(greal[m])]
        if gg.size > 1:
            gspread.append(gg.max() - gg.min())
    print("\n-- 1. realized between-sibling spread (READ THIS FIRST) --")
    print(f"   survival: {mixed}/{n_groups} groups have siblings that DISAGREE "
          f"({100 * mixed / max(n_groups, 1):.1f}%) -- only these can inform a survival ranking")
    print(f"   quality : realized G spread within a group, median "
          f"{np.median(gspread):.4f}  p90 {np.percentile(gspread, 90):.4f}"
          if gspread else "   quality : no groups")
    print(f"   overall failure rate {fail.mean():.4f} over {fail.size} branches")
    # THE VERDICT MIX, printed because a run dominated by timeouts is a statement about the
    # clock and not the robot -- the shape of the `--isaac_matched` retraction, where 84.7% of
    # episodes timed out and the failure label was almost entirely the budget.
    for col in ("branch_verdict", "cont_verdict"):
        if col in d:
            vv = [x.decode() if isinstance(x, bytes) else str(x) for x in d[col]]
            hist = {a: vv.count(a) for a in sorted(set(vv))}
            print(f"   {col:15s} {hist}")
    term = d["terminal_in_branch"].astype(bool) if "terminal_in_branch" in d else None
    if term is not None and term.any():
        print(f"   terminal INSIDE its own chunk: {term.sum()}/{term.size} branches "
              f"({100 * term.mean():.1f}%) -- these have g_real = NaN and no critic read, so "
              "they are CENSORED from sections 4 and 5. A live selector ranks them correctly "
              "for free by sentinel, so a null there is a null about the HEAD, not about "
              "best-of-K.")
    if bool(attrs.get("same_noise", False)):
        print("\n   *** --same_noise RUN: this is the BLOCKING CONTROL. Every sibling had one "
              f"draw, so informative groups MUST be 0. Observed {mixed}. "
              f"{'PASS' if mixed == 0 else 'FAIL -- the state restore is not exact and every '
               'number from a normal run is invalid.'} ***")
    if mixed < 0.1 * n_groups:
        print("   *** the siblings almost never differ in outcome. No critic could rank them; "
              "this is a statement about the GENERATOR, not the critic. ***")

    # ---- 2. label noise that CRN did not remove ---------------------------------------
    if "failed_b" in d:
        m = np.isfinite(d["g_real_b"]) & np.isfinite(greal)
        if m.any():
            fb = d["failed_b"].astype(bool)[m]
            print("\n-- 2. label noise: the SAME branch rolled on a second continuation seed --")
            print(f"   n={int(m.sum())}  survival label flips {100 * (fb != fail[m]).mean():.1f}%"
                  f"   |dG| median {np.median(np.abs(d['g_real_b'][m] - greal[m])):.4f}")
            print("   a within-group signal smaller than this is not resolvable")

    # ---- 3. the noise floor: two honest estimates of the same quantity ----------------
    print("\n-- 3. critic noise floor (fold disagreement) vs within-state SPREAD --")
    for stat, lbl in (("p_", "p_fail"), ("g_", "G_hat")):
        cols = [stat + n for n in names if stat + n in d]
        if len(cols) < 2:
            continue
        a, b = d[cols[0]], d[cols[1]]
        m = np.isfinite(a) & np.isfinite(b)
        disagree = float(np.mean(np.abs(a[m] - b[m]))) if m.any() else float("nan")
        spreads = []
        for g in np.unique(gid):
            v = a[(gid == g) & np.isfinite(a)]
            if v.size > 1:
                spreads.append(v.max() - v.min())
        sp = float(np.median(spreads)) if spreads else float("nan")
        # Six decimals on purpose: at four, the two heads' disagreements printed as an
        # identical 0.0324 and read as a copy-paste bug rather than a coincidence (they are
        # 0.032375 and 0.032366). A number that invites that misreading costs a verification
        # pass every time it is looked at.
        print(f"   {lbl:7s} fold |disagreement| {disagree:.6f}   within-state spread (median) "
              f"{sp:.6f}   SNR {sp / disagree if disagree else float('nan'):.2f}")

    # ---- 4. can it RANK? each head against its own target ----------------------------
    print("\n-- 4. within-group ranking, each head against ITS OWN target --")
    print("   tau is ORIENTED: positive always means the head ranks siblings correctly.")
    rng = np.random.default_rng(0)
    for nm in names:
        for stat, target, lower, tlabel in (
                ("p_" + nm, (~fail).astype(float), True, "survived"),
                ("g_" + nm, greal, False, "realized G")):
            if stat not in d:
                continue
            pred = d[stat]
            tau, ntau = _within_group_tau(pred, target, gid, lower_is_better=lower)
            rate, nr = _pick_rate(pred, target, gid, lower_is_better=lower)
            null = []
            for _ in range(200):
                sh = pred.copy()
                for g in np.unique(gid):
                    m = np.where(gid == g)[0]
                    sh[m] = pred[rng.permutation(m)]
                null.append(_pick_rate(sh, target, gid, lower_is_better=lower)[0])
            print(f"   {stat:16s} vs {tlabel:11s}  tau {tau:+.4f} (n={ntau:4d})   "
                  f"best-pick {rate:.4f} vs null {np.nanmean(null):.4f}"
                  f" +/- {np.nanstd(null):.4f} (n={nr:4d})")

    # ---- 5. the measured 0.24 s reward ------------------------------------------------
    print("\n-- 5. the MEASURED 0.24 s reward: signal or jitter? --")
    for target, lbl, lower in ((greal, "realized G", False),
                               ((~fail).astype(float), "survived", False)):
        tau, ntau = _within_group_tau(d["r_exec"], target, gid, lower_is_better=lower)
        rate, nr = _pick_rate(d["r_exec"], target, gid, lower_is_better=lower)
        print(f"   r_exec vs {lbl:11s}  tau {tau:+.4f} (n={ntau:4d})   "
              f"best-pick {rate:.4f} (n={nr:4d})")
    return 0


def main() -> int:
    """Entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="run", choices=("run", "report"))
    p.add_argument("--out", required=True)
    p.add_argument("--critic", action="append", default=[],
                   help="NAME=path, repeatable. Pass BOTH cross-fit folds: their "
                        "disagreement is the noise floor the within-state spread is read "
                        "against, and it costs one forward pass per branch.")
    p.add_argument("--fm_pt", default="")
    p.add_argument("--tracking_spec", default="")
    p.add_argument("--tracker_onnx", default="")
    p.add_argument("--fm_onnx", default="")
    p.add_argument("--clip_lib", default="")
    p.add_argument("--tile", type=int, default=6)
    p.add_argument("--speed", type=float, default=0.8,
                   help="pick a cell whose success rate is NEAR 0.5: the survival half of this "
                        "probe only learns from groups whose siblings disagree, with "
                        "probability 1 - p^K - (1-p)^K.")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--probe_every", type=int, default=4,
                   help="branch at one re-plan in this many. Rolling every branch of every "
                        "re-plan to completion costs ~70x a plain episode.")
    p.add_argument("--second_seed_every", type=int, default=4,
                   help="on one episode in this many, roll sibling 0 on a SECOND continuation "
                        "seed, to size the label noise CRN leaves behind.")
    p.add_argument("--same_noise", action="store_true",
                   help="BLOCKING CONTROL: hand every sibling ONE branch draw. They must then "
                        "be bit-identical, so the report's informative-group count must be "
                        "EXACTLY 0. Any non-zero count proves the state restore is not exact "
                        "and invalidates the whole run. Run this before quoting anything -- it "
                        "is the arm whose absence retired the Isaac best-of-K harness.")
    p.add_argument("--flush_every", type=int, default=2,
                   help="rewrite branches.h5 every N episodes. A run is tens of ~2 min "
                        "episodes; without this a crash at the last one loses all of them, "
                        "and --stage report cannot read a run in flight.")
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--scene_tag", default="_bprobe")
    args = p.parse_args()
    return stage_run(args) if args.stage == "run" else stage_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
