#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Quality report for a generated clip library -- does the motion actually do what the command says?

Reads a clip ``.h5`` and nothing else (no Motion Bricks, no IsaacLab), so it works on any library
regardless of backend and is the objective answer to questions that would otherwise be eyeballed in viser:

* **command tracking, split by REGIME** -- the Motion Bricks flat path pins our optimized stride as the
  target pose only when the command is steady, and relaxes the pin through velocity transitions (the
  regime gate). So the interesting number is not one average error but STEADY vs TRANSITION error: the
  steady error says whether the pin lands the commanded velocity, the transition error says whether the
  in-betweening is doing its job. Reported per SPEED BUCKET too, because a pin that holds at 1 m/s can
  fail at 2.5 m/s, and that is exactly what a whole-library average hides.
* **foot slide** -- horizontal travel of a foot while it is in contact. The main artefact a pose pin or a
  spliced seam introduces, and invisible in a root-trajectory plot.
* **seam spikes** -- per-frame root speed, and specifically its value where the per-step anchor mask
  changes (i.e. at the approach->terrain->exit seams), so a discontinuous splice shows up as a number.
* **pelvis height** -- range per clip; a sanity check that a clip is neither sinking nor floating.

Flat and terrain clips are reported separately: a terrain clip's ``target_vel`` is deliberately held at
the approach speed for the whole traversal (the robot slows through the interaction), so its tracking
error is expected and only the slide / seam / height numbers are defects. Example::

    PYTHONPATH=source/motion_gen env_isaaclab/bin/python scripts/rsl_rl/mb_qc.py \
        --lib clips/g1_flat_mb_v1.h5 --top 10
"""

from __future__ import annotations

import argparse

import numpy as np

from motion_gen.clips.clip_io import load_clip_library
from motion_gen.world_lift import _ANCHOR_MASK_FLAT

_TERRAIN_TYPE_NAMES = ["flat", "terrain", "transition", "slope", "box"]


def _yaw_wxyz(q: np.ndarray) -> np.ndarray:
    """Z-up yaw ``[T]`` from wxyz quaternions ``[T, 4]``."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pct(a: np.ndarray, q: float) -> float:
    """Percentile of ``a``, or ``nan`` for an empty array (a clip may have no contact/steady frames)."""
    return float(np.percentile(a, q)) if a.size else float("nan")


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``[start, end)`` runs where ``flag`` is True."""
    d = np.diff(np.concatenate([[0], flag.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _stance_slide(xy: np.ndarray, flag: np.ndarray, dt: float,
                  min_stance: int, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-stance net foot slide ``(displacement_m[K], median_speed[K])``.

    ``contact`` in a clip library is a GEOMETRIC proximity flag (a foot-box bottom corner within 2 cm of
    the ground, ``ClipOutputBuilder``), so it is also true through touchdown, liftoff and any low swing
    pass -- measuring instantaneous foot speed over every flagged frame mostly measures swing feet moving
    at 2-5 m/s, not slide. So: keep only runs long enough to be a real stance, trim ``edge`` frames off
    each end (the touchdown/liftoff transitions), and measure how far the foot TRAVELS while planted.
    """
    disp, spd = [], []
    for a, b in _runs(flag):
        a, b = a + edge, b - edge
        if b - a < min_stance:
            continue
        seg = xy[a:b]
        disp.append(float(np.linalg.norm(seg[-1] - seg[0])))
        step = np.linalg.norm(np.diff(seg, axis=0), axis=-1) / dt
        spd.append(float(np.median(step)) if step.size else 0.0)
    return np.asarray(disp, np.float64), np.asarray(spd, np.float64)


def _regime_mask(cmd: np.ndarray, dt: float, ramp_s: float) -> np.ndarray:
    """``True`` where the command has been steady for at least ``ramp_s`` on both sides.

    Mirrors ``mb_env.regime_weight``: the command is piecewise-constant, so its change frames are the
    segment boundaries, and a frame is STEADY when it sits at least ``ramp_s`` away from every boundary.
    Recovering the regime from the stored command means the split needs no generator-side bookkeeping.
    """
    changed = np.any(np.abs(np.diff(cmd, axis=0)) > 1e-4, axis=1)
    bnds = np.nonzero(changed)[0] + 1
    if bnds.size == 0:
        return np.ones(cmd.shape[0], dtype=bool)
    idx = np.arange(cmd.shape[0])
    dist = np.min(np.abs(idx[:, None] - bnds[None, :]), axis=1) * dt
    return dist >= ramp_s


def _cycle_avg_err(vx: np.ndarray, vy: np.ndarray, cmd: np.ndarray, steady: np.ndarray,
                   win: int) -> tuple[float, float]:
    """Velocity error AVERAGED OVER ABOUT A GAIT CYCLE -- the only honest tracking number here.

    ``err_steady_*`` compares the INSTANTANEOUS body velocity to the command frame by frame, and a real
    gait's pelvis accelerates and decelerates within every step.  A library of mathematically EXACT
    looped replays -- velocity correct to 1e-6 m/s over whole periods -- still scores ``err_steady_p50``
    of 0.188 m/s on that metric, so it cannot tell a perfect reference from a mediocre one.  Smoothing
    over a cycle removes the oscillation and leaves the part that is actually a tracking error: measured
    that way, exact loops score 0.017 and Motion Bricks' own flat motion 0.099 m/s.

    Windows are only scored where the command is steady THROUGHOUT, so a smoothed sample never straddles
    a command change.  Returns ``(p50, p95)`` of the forward-velocity error, NaN if nothing qualifies.
    """
    n = vx.shape[0]
    if n < win + 2:
        return float("nan"), float("nan")
    k = np.ones(win) / win
    sm = np.convolve(vx, k, mode="valid")                  # [n - win + 1]
    tgt = cmd[win // 2: win // 2 + sm.size, 0]
    ok = np.convolve(steady.astype(float), k, mode="valid") > 0.999   # steady across the whole window
    if ok.sum() < 3:
        return float("nan"), float("nan")
    e = np.abs(sm[ok] - tgt[ok])
    return float(np.percentile(e, 50)), float(np.percentile(e, 95))


def _seam_accel_ratios(root_speed: np.ndarray, seams: np.ndarray) -> np.ndarray:
    """PER-SEAM: how far each seam's velocity jump stands out against the clip's own variation.

    ``seam_speed`` is the root speed AT the seam, which is dominated by how fast the clip is walking --
    a 3 m/s clip reports a 3 m/s "seam speed" whether or not anything is wrong.  What matters is whether
    the velocity CHANGES discontinuously there, and "discontinuous" only means anything relative to the
    accelerations the gait already contains.  So: the seam's |dv| over the p95 of |dv| everywhere else.
    ~1 means the seam looks like ordinary gait; >> 1 is a pop.

    Returned PER SEAM, not reduced to the clip's worst.  A max-per-clip summary is not comparable across
    families: a flat clip has 6-7 seams and a terrain clip 2, so the flat one scores worse for identical
    per-seam quality simply by having more chances to be unlucky.  That structural unfairness read as a
    flat regression (3.77 vs terrain 0.02) that did not exist -- pooled per-seam, flat was the better of
    the two.  Pool these across the library and take percentiles of the pool.
    """
    if seams.size == 0 or root_speed.size < 4:
        return np.zeros(0)
    dv = np.abs(np.diff(root_speed))
    # root_speed[k] spans the interval [k, k+1], so the interval STRADDLING a seam at frame b is
    # root_speed[b-1], and the jump can show up on either side of it: dv[b-2] or dv[b-1].  Which one
    # depends on the splice -- a weld duplicates a frame at the seam, making the straddling interval
    # equal to a neighbour and zeroing that side.  Reading a single fixed offset therefore reported a
    # flat 0.00 for every terrain ENTRY seam (all 20 of them, p50 through max) purely because the
    # duplicate landed on the side being sampled.  Take the worst of the two.
    ref_mask = np.ones(dv.size, dtype=bool)
    for b in seams:
        ref_mask[max(0, b - 3):min(dv.size, b + 2)] = False   # keep seam neighbourhoods out of the
    if ref_mask.sum() < 8:                                    # reference, else a pop inflates its own
        ref_mask[:] = True                                    # normalizer and hides itself
    ref = float(np.percentile(dv[ref_mask], 95))
    if ref <= 1e-9:
        return np.zeros(0)
    lo = np.clip(seams - 2, 0, dv.size - 1)
    hi = np.clip(seams - 1, 0, dv.size - 1)
    return np.maximum(dv[lo], dv[hi]) / ref


def _clip_metrics(y: np.ndarray, dy: np.ndarray, contact: np.ndarray, cmd: np.ndarray,
                  amt: np.ndarray | None, dt: float, ramp_s: float,
                  min_stance: int, edge: int, stride_idx: np.ndarray | None = None,
                  cycle_win: int = 40) -> dict:
    """Per-clip metrics over the clip's REAL frames (padding already stripped by the caller)."""
    yaw = _yaw_wxyz(y[:, 3:7])
    c, s = np.cos(yaw), np.sin(yaw)
    vx = c * dy[:, 0] + s * dy[:, 1]                      # world planar velocity -> pelvis-yaw frame
    vy = -s * dy[:, 0] + c * dy[:, 1]
    wz = dy[:, 5]
    err = np.stack([vx - cmd[:, 0], vy - cmd[:, 1], wz - cmd[:, 2]], axis=-1)
    speed_err = np.linalg.norm(err[:, :2], axis=-1)       # planar velocity error magnitude
    # Prefer the regime the GENERATOR recorded over one inferred from command change points: with exact
    # looped steady stretches the two no longer coincide, because a transient's extent is Motion Bricks'
    # choice rather than the command's.  ``stride_idx >= 0`` IS "this frame is an exact loop replay".
    if stride_idx is not None and stride_idx.size == cmd.shape[0] and bool((stride_idx >= 0).any()):
        # Only when this clip actually HAS looped frames.  A composed library declares ``has_regime``
        # library-wide, but only the flat family carries loops -- terrain writes -1 throughout, so taking
        # this branch there would mark the whole clip "not steady" and report no tracking at all.
        steady = stride_idx >= 0
    elif amt is not None and amt.size == cmd.shape[0] and np.unique(amt).size > 1:
        # TERRAIN, and specifically terrain with NO exact loops -- the pure-stair and one-sided box
        # families, which are a replay plus in-betweenings and nothing else.  The traversal is exactly
        # where command and motion are SUPPOSED to diverge (the terrain overrides the command), so
        # score only the ego-anchored approach and exit, where the contract says they must agree.
        #
        # AND intersect with the command's own regime.  This branch used to be the anchor mask alone,
        # on the stated grounds that "its command is a deliberate per-clip CONSTANT" -- true when it
        # was written, false since a terrain clip's lead-up got a varying command.  With a constant
        # command the mask was harmless; with a varying one it scores every in-betweening frame as
        # steady, and an in-betweening is a deliberate acceleration between two commanded speeds.
        # Measured on the quick build: clips scored this way report cyc_err ~1.2 m/s against ~0.03 for
        # loop-scored clips, so a family moving into this bucket looks like a large regression that is
        # entirely the metric's.
        steady = (amt == _ANCHOR_MASK_FLAT) & _regime_mask(cmd, dt, ramp_s)
    else:
        steady = _regime_mask(cmd, dt, ramp_s)

    disp, spd, stance_s = [], [], []                       # per-stance net slide, both feet pooled
    for f, base in enumerate((7, 14)):                      # left / right ankle_roll pos_x
        d, s = _stance_slide(y[:, base:base + 2], contact[:, f], dt, min_stance, edge)
        disp.append(d); spd.append(s)
        stance_s += [(b - a) * dt for a, b in _runs(contact[:, f]) if (b - a) >= min_stance + 2 * edge]
    slide_disp = np.concatenate(disp) if disp else np.zeros(0)
    slide_spd = np.concatenate(spd) if spd else np.zeros(0)
    # Duty factor + stance duration make the slide number INTERPRETABLE: the contact flag is geometric, so
    # a low-clearance gait keeps it true through swing.  Duty ~0.6 with stance ~0.4 s = a real gait (slide
    # is slide); duty near 1.0 or stance ~ the whole stride = the flag is capturing swing, and the slide
    # figure is an upper bound rather than a defect.
    duty = float(contact.mean()) if contact.size else 0.0

    root_speed = np.linalg.norm(y[1:, 0:3] - y[:-1, 0:3], axis=-1) / dt
    seam_speed, seam_ratios = float("nan"), np.zeros(0)
    # Seams come from the per-step anchor mask (terrain) or, on a loop-spliced flat clip, from where the
    # exact-replay stretches start and stop.
    # Whichever source actually VARIES marks this clip's seams: terrain segments them with the per-step
    # anchor mask, flat with stride_idx.  In a COMPOSED library a flat clip still carries a per-step mask,
    # but a constant one (its whole clip is ego-anchored), so keying on presence alone found no seams at
    # all and reported n/a for the entire flat family.
    # Both sources are read and the seams LABELLED by which one moved, because they mark different
    # things and are worth scoring apart: the anchor mask changes only where the clip enters or leaves
    # the terrain, while stride_idx changes at every loop<->transient handoff.  Keying on one source
    # alone hides the other's seams entirely -- terrain reported 2 seams per clip (its boundaries) and
    # none of its internal handoffs, so "25% of terrain seams are bad" could not be attributed.
    seam_kinds: list[str] = []
    parts = []
    for cand, kind in ((amt, "boundary"), (stride_idx, "loop")):
        if cand is not None and cand.size == y.shape[0] and np.unique(cand).size > 1:
            idx = np.nonzero(np.diff(cand.astype(np.int32)) != 0)[0] + 1
            parts.append((idx, kind))
    if parts:
        # A boundary and a loop handoff can coincide (the replay starts exactly where the approach loop
        # ends); credit those to the boundary, which is the stronger constraint.
        bidx = np.concatenate([i for i, k in parts if k == "boundary"]) if any(
            k == "boundary" for _, k in parts) else np.zeros(0, dtype=int)
        seams_l, kinds_l = [], []
        for idx, kind in parts:
            for j in idx:
                if kind == "loop" and bidx.size and np.abs(bidx - j).min() <= 1:
                    continue
                seams_l.append(int(j)); kinds_l.append(kind)
        order = np.argsort(np.asarray(seams_l))
        seams = np.asarray(seams_l, dtype=int)[order]
        seam_kinds = [kinds_l[i] for i in order]
        # The two boundaries of a terrain clip are its entry and its exit, in timeline order.
        bpos = [i for i, k in enumerate(seam_kinds) if k == "boundary"]
        if len(bpos) == 2:
            seam_kinds[bpos[0]] = "entry"
            seam_kinds[bpos[1]] = "exit"
        if seams.size:
            seam_speed = float(root_speed[np.clip(seams - 1, 0, root_speed.size - 1)].max())
            seam_ratios = _seam_accel_ratios(root_speed, seams)
    cyc50, cyc95 = _cycle_avg_err(vx, vy, cmd, steady, cycle_win)

    return {
        "n": int(y.shape[0]),
        "cmd_speed": float(np.linalg.norm(cmd[:, :2], axis=-1).max()),
        "err_steady_p50": _pct(speed_err[steady], 50), "err_steady_p95": _pct(speed_err[steady], 95),
        "err_trans_p50": _pct(speed_err[~steady], 50), "err_trans_p95": _pct(speed_err[~steady], 95),
        "err_wz_p95": _pct(np.abs(err[:, 2]), 95),
        # The instantaneous errors above are dominated by within-stride gait oscillation; these two are
        # the ones to read for tracking quality (see _cycle_avg_err).
        "cyc_err_p50": cyc50, "cyc_err_p95": cyc95,
        # The clip's WORST seam, for ranking bad clips only -- never compare it across families (see
        # _seam_accel_ratios). The pooled per-seam distribution is reported separately.
        "seam_accel_max": float(seam_ratios.max()) if seam_ratios.size else float("nan"),
        "seam_ratios": seam_ratios,
        "seam_kinds": seam_kinds,
        "steady_frac": float(steady.mean()),
        "n_stances": int(slide_disp.size),
        "contact_duty": duty,
        "stance_s_med": float(np.median(stance_s)) if stance_s else float("nan"),
        "slide_p95": _pct(slide_disp, 95),                 # metres of travel per planted stance
        "slide_max": float(slide_disp.max()) if slide_disp.size else 0.0,
        "slide_speed_p95": _pct(slide_spd, 95),            # m/s while planted
        "root_speed_max": float(root_speed.max()) if root_speed.size else 0.0,
        "seam_speed_max": seam_speed,
        "pelvis_z_min": float(y[:, 2].min()), "pelvis_z_max": float(y[:, 2].max()),
    }


def _summarize(rows: list[dict], label: str, keys: list[tuple[str, str]]) -> None:
    """Print p50/p95/max of each metric across clips."""
    if not rows:
        return
    print(f"\n  --- {label} ({len(rows)} clips) ---")
    print(f"    {'metric':22s} {'p50':>9s} {'p95':>9s} {'max':>9s}")
    for key, unit in keys:
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not v.size:
            print(f"    {key:22s} {'n/a':>9s} {'n/a':>9s} {'n/a':>9s}   {unit}")
            continue
        print(f"    {key:22s} {np.percentile(v, 50):9.3f} {np.percentile(v, 95):9.3f} "
              f"{v.max():9.3f}   {unit}")


def _speed_buckets(rows: list[dict], edges: list[float]) -> None:
    """Steady/transition tracking error and foot slide per commanded-speed band.

    The point of the whole report: if the pose pin degrades with speed, it shows up as a rising steady
    error in the fast buckets while the library average stays acceptable.
    """
    print(f"\n  --- tracking + slide by commanded speed ---")
    print(f"    {'speed band':>14s} {'clips':>6s} {'steady p50':>11s} {'steady p95':>11s} "
          f"{'trans p50':>10s} {'slide p95':>10s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["cmd_speed"] < hi]
        if not sel:
            continue
        g = lambda k: np.array([r[k] for r in sel], dtype=np.float64)
        gm = lambda k: float(np.nanmedian(g(k))) if np.any(np.isfinite(g(k))) else float("nan")
        gp = lambda k: (float(np.nanpercentile(g(k), 95)) if np.any(np.isfinite(g(k))) else float("nan"))
        print(f"    {f'[{lo:.1f},{hi:.1f})':>14s} {len(sel):6d} {gm('err_steady_p50'):11.3f} "
              f"{gp('err_steady_p95'):11.3f} {gm('err_trans_p50'):10.3f} {gm('slide_p95'):10.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib", required=True, help="Clip library .h5 to report on.")
    p.add_argument("--gate", action="store_true",
                   help="Exit non-zero if the library fails the quality thresholds below. The build "
                        "driver runs this as its last stage, so a bad library fails the build instead "
                        "of being written and looked at later -- which is how a 9.8%% flat root-speed "
                        "blow-up shipped in g1_mb_v8 and was only found by watching a clip.")
    p.add_argument("--gate-max-root-speed", type=float, default=6.0,
                   help="No clip may contain a frame faster than this (m/s). The library's own exact "
                        "looped replays peak at 3.82 and the worst sound in-betweening at 4.84, so "
                        "6.0 sits above everything correct with margin; the blow-ups it exists to "
                        "catch ran 19-31.")
    p.add_argument("--gate-pelvis-band", type=float, nargs=2, default=(0.40, 0.95),
                   metavar=("MIN", "MAX"),
                   help="Pelvis height above the LOWER FOOT (m), not above the world. Absolute height "
                        "is wrong on terrain -- a pure-stairs clip legitimately reaches 3.10 m as it "
                        "climbs -- while the pelvis-to-foot distance is a property of the ROBOT and "
                        "needs no terrain knowledge at all. Measured over the whole library it spans "
                        "0.511-0.757 m (p01 0.614, p99 0.753), so this band clears everything correct "
                        "and still catches a robot collapsed on the floor or launched into the air.")
    p.add_argument("--top", type=int, default=10, help="List this many worst clips per ranking key.")
    p.add_argument("--ramp-s", type=float, default=0.4,
                   help="Regime window (s) around a command change; matches mb_env.regime_weight's default.")
    p.add_argument("--max-clips", type=int, default=0,
                   help="Evenly subsample to at most this many clips (0 = all). For a 10k library.")
    p.add_argument("--cycle-s", type=float, default=0.8,
                   help="Averaging window (s) for cyc_err_*, about one gait cycle. The instantaneous "
                        "err_steady_* cannot separate tracking error from within-stride oscillation: an "
                        "EXACT looped replay scores 0.188 on it. Smoothing over a cycle can.")
    p.add_argument("--min-stance", type=int, default=5,
                   help="Minimum contact-run length (frames) counted as a stance; shorter runs are low "
                        "swing passes of the geometric contact flag, not stances.")
    p.add_argument("--stance-edge", type=int, default=2,
                   help="Frames trimmed from each end of a stance before measuring slide -- the contact "
                        "flag is geometric, so its edges are touchdown / rolling toe-off, where the foot "
                        "is legitimately still moving.")
    args = p.parse_args()

    lib = load_clip_library(args.lib, "cpu")
    dt = float(lib.meta["dt"])
    pn = list(lib.meta["pos_output_names"])
    for want in ("pelvis_link:pos_x", "left_ankle_roll_link:pos_x", "right_ankle_roll_link:pos_x"):
        if pn.index(want) not in (0, 7, 14):
            raise SystemExit(f"unexpected channel layout: {want} at index {pn.index(want)}; "
                             f"mb_qc assumes pelvis@0, left ankle@7, right ankle@14")
    y_all = np.asarray(lib.y_des)
    dy_all = np.asarray(lib.dy_des)
    ct_all = np.asarray(lib.contact)
    tv_all = np.asarray(lib.target_vel)
    clen = lib.clip_len_or_full().numpy()
    tcode = np.asarray(lib.terrain_type_code)
    amt_all = np.asarray(lib.anchor_mask_per_step()) if lib.anchor_mask_t.numel() else None
    # ``stride_idx`` marks the exact-loop frames, but ONLY where the generator actually wrote it: the
    # dataset is created with h5py's default fill of 0, so an older library reads as "stride 0
    # everywhere" == 100% steady.  Gate on the explicit attr rather than on the dataset being present.
    si_all = np.asarray(lib.stride_idx) if bool(lib.attrs.get("has_regime", False)) else None
    if si_all is not None:
        print("[mb_qc] using the generator-recorded regime (stride_idx >= 0 == exact loop replay)")

    idx = list(range(int(lib.num_clips)))
    if args.max_clips > 0 and len(idx) > args.max_clips:
        idx = idx[:: max(1, len(idx) // args.max_clips)][:args.max_clips]
    print(f"[mb_qc] {args.lib}: {lib.num_clips} clips x {lib.num_steps} steps, dt={dt}, "
          f"generator={lib.attrs.get('generator')} (reporting on {len(idx)})")

    flat_rows, terr_rows = [], []
    for i in idx:
        n = max(2, int(clen[i]))
        r = _clip_metrics(y_all[i, :n], dy_all[i, :n], ct_all[i, :n].astype(bool), tv_all[i, :n],
                          (amt_all[i, :n] if amt_all is not None else None), dt, args.ramp_s,
                          args.min_stance, args.stance_edge,
                          stride_idx=(si_all[i, :n] if si_all is not None else None),
                          cycle_win=max(4, int(round(args.cycle_s / dt))))
        r["clip"] = i
        r["type"] = _TERRAIN_TYPE_NAMES[int(tcode[i])] if int(tcode[i]) < len(_TERRAIN_TYPE_NAMES) else "?"
        (flat_rows if r["type"] == "flat" else terr_rows).append(r)

    common = [("slide_p95", "m/stance"), ("slide_max", "m/stance"), ("slide_speed_p95", "m/s"),
              ("contact_duty", "frac"), ("stance_s_med", "s"), ("n_stances", "-"),
              ("root_speed_max", "m/s"), ("seam_speed_max", "m/s"),
              ("pelvis_z_min", "m"), ("pelvis_z_max", "m")]
    tracking = [("cyc_err_p50", "m/s"), ("cyc_err_p95", "m/s"),
                ("err_steady_p50", "m/s"), ("err_steady_p95", "m/s"),
                ("err_trans_p50", "m/s"), ("err_trans_p95", "m/s"), ("err_wz_p95", "rad/s"),
                ("seam_accel_max", "x"), ("steady_frac", "-")]
    def _seam_pool_by_kind(rows: list[dict], label: str) -> None:
        """Pool per-seam ratios per seam KIND, so entry / exit / internal handoffs are ranked apart."""
        buckets: dict[str, list[float]] = {}
        for r in rows:
            for ratio, kind in zip(r.get("seam_ratios", []), r.get("seam_kinds", [])):
                buckets.setdefault(kind, []).append(float(ratio))
        for kind in ("entry", "exit", "boundary", "loop"):
            v = np.asarray(buckets.get(kind, []))
            if not v.size:
                continue
            print(f"    {label} {kind:9s}: {v.size:4d} seams  ratio p50={np.percentile(v, 50):.2f} "
                  f"p95={np.percentile(v, 95):.2f} max={v.max():.2f}  "
                  f"({100.0 * float((v > 2.0).mean()):.0f}% above 2x the clip's own variation)")

    def _seam_pool(rows: list[dict], label: str) -> None:
        """Pooled PER-SEAM ratio -- the comparable number, unlike the per-clip max above."""
        pool = np.concatenate([r["seam_ratios"] for r in rows if r["seam_ratios"].size]) \
            if any(r["seam_ratios"].size for r in rows) else np.zeros(0)
        if not pool.size:
            return
        print(f"    {label}: {pool.size} seams  ratio p50={np.median(pool):.2f} "
              f"p95={np.percentile(pool, 95):.2f} max={pool.max():.2f}  "
              f"({100.0 * (pool > 2.0).mean():.0f}% above 2x the clip's own variation)")

    if flat_rows:
        print(f"\n[mb_qc] === FLAT clips ===")
        _summarize(flat_rows, "command tracking (steady vs transition regime)", tracking)
        _summarize(flat_rows, "artefacts", common)
        print("\n  --- seam discontinuity (pooled per seam; the per-clip max above is NOT comparable "
              "across families) ---")
        _seam_pool(flat_rows, "flat")
        _seam_pool_by_kind(flat_rows, "flat")
        _speed_buckets(flat_rows, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 99.0])
    if terr_rows:
        _summarize(terr_rows, "command tracking (approach + exit only; the traversal is excluded)",
                   tracking)
        print(f"\n[mb_qc] === TERRAIN clips (tracking error is EXPECTED: the command is held at the "
              f"approach speed through the traversal) ===")
        _summarize(terr_rows, "artefacts", common)
        print("\n  --- seam discontinuity (pooled per seam) ---")
        _seam_pool(terr_rows, "terrain")
        _seam_pool_by_kind(terr_rows, "terrain")
        by_type: dict[str, list[dict]] = {}
        for r in terr_rows:
            by_type.setdefault(r["type"], []).append(r)
        for t, rows in sorted(by_type.items()):
            # A family with no per-step anchor mask (pure tiles, distractors) has no seams to measure, so
            # every seam value is nan -- report that rather than letting nanmedian warn on an all-nan slice.
            seams = [r["seam_speed_max"] for r in rows if np.isfinite(r["seam_speed_max"])]
            seam_txt = f"{np.median(seams):.3f} m/s" if seams else "n/a (no spliced seams)"
            print(f"    type '{t}': {len(rows)} clips, seam_speed_max median {seam_txt}, "
                  f"slide_p95 median {np.nanmedian([r['slide_p95'] for r in rows]):.3f} m/stance")

    for key in ("slide_p95", "seam_speed_max", "err_steady_p95"):
        rows = (flat_rows + terr_rows) if key != "err_steady_p95" else flat_rows
        rows = [r for r in rows if np.isfinite(r[key])]
        if not rows or args.top <= 0:
            continue
        worst = sorted(rows, key=lambda r: -r[key])[:args.top]
        print(f"\n  --- worst {len(worst)} by {key} ---")
        for r in worst:
            print(f"    clip {r['clip']:5d} [{r['type']:10s}] {key}={r[key]:6.3f}  "
                  f"cmd_speed={r['cmd_speed']:.2f}  slide_p95={r['slide_p95']:.3f}  "
                  f"pelvis_z=[{r['pelvis_z_min']:.2f},{r['pelvis_z_max']:.2f}]")

    # Seam diagnostics the terrain generator already measured, if this library carries them.
    plan = lib.attrs.get("mb_terrain_plan")
    if isinstance(plan, dict) and "seam" in plan:
        print(f"\n  --- generator-reported seams ({plan.get('skill')}: "
              f"{plan.get('approach')} -> {plan.get('exit')}) ---")
        for k, v in plan["seam"].items():
            print(f"    {k:24s} {v}")
    if args.gate:
        _names = list(lib.meta["pos_output_names"])
        try:
            _LZ = _names.index("left_ankle_roll_link:pos_z")
            _RZ = _names.index("right_ankle_roll_link:pos_z")
        except ValueError:
            raise SystemExit("--gate needs the ankle position channels; this library has none")
        # THE BACKSTOP. Everything above is a report a human reads; this is the number that fails a
        # build. It deliberately checks only what is unambiguous -- a speed nothing correct reaches,
        # and a pelvis outside the range a standing robot can occupy -- because a gate that argues
        # about quality gets turned off, and one that catches broken motion does not.
        fails = []
        for i in idx:
            n = int(lib.clip_len[i])
            p_xy = np.asarray(lib.y_des[i, :n, 0:2])
            sp = float((np.linalg.norm(np.diff(p_xy, axis=0), axis=1) / dt).max()) if n > 1 else 0.0
            # SURFACE-RELATIVE by construction: the lower foot IS the surface, whatever height it
            # is at, so this needs nothing from the terrain manifest and cannot be fooled by a clip
            # that climbs.
            h = (np.asarray(lib.y_des[i, :n, 2])
                 - np.minimum(np.asarray(lib.y_des[i, :n, _LZ]), np.asarray(lib.y_des[i, :n, _RZ])))
            lo_b, hi_b = float(min(args.gate_pelvis_band)), float(max(args.gate_pelvis_band))
            if sp > args.gate_max_root_speed:
                fails.append(f"clip {i}: root speed {sp:.2f} > {args.gate_max_root_speed} m/s")
            if float(h.max()) > hi_b:
                fails.append(f"clip {i}: pelvis {float(h.max()):.2f} m above its lower foot > {hi_b}")
            if float(h.min()) < lo_b:
                fails.append(f"clip {i}: pelvis {float(h.min()):.2f} m above its lower foot < {lo_b}")
        if fails:
            print(f"\n[mb_qc] GATE FAILED: {len(fails)} problem(s) in {len(idx)} clips")
            for line in fails[:20]:
                print(f"  {line}")
            if len(fails) > 20:
                print(f"  ... and {len(fails) - 20} more")
            return 1
        print(f"\n[mb_qc] gate PASSED over {len(idx)} clips (root speed <= "
              f"{args.gate_max_root_speed} m/s, pelvis above its lower foot in "
              f"{tuple(args.gate_pelvis_band)} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
