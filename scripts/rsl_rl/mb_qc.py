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


def _clip_metrics(y: np.ndarray, dy: np.ndarray, contact: np.ndarray, cmd: np.ndarray,
                  amt: np.ndarray | None, dt: float, ramp_s: float,
                  min_stance: int, edge: int) -> dict:
    """Per-clip metrics over the clip's REAL frames (padding already stripped by the caller)."""
    yaw = _yaw_wxyz(y[:, 3:7])
    c, s = np.cos(yaw), np.sin(yaw)
    vx = c * dy[:, 0] + s * dy[:, 1]                      # world planar velocity -> pelvis-yaw frame
    vy = -s * dy[:, 0] + c * dy[:, 1]
    wz = dy[:, 5]
    err = np.stack([vx - cmd[:, 0], vy - cmd[:, 1], wz - cmd[:, 2]], axis=-1)
    speed_err = np.linalg.norm(err[:, :2], axis=-1)       # planar velocity error magnitude
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
    seam_speed = float("nan")
    if amt is not None and amt.size == y.shape[0]:        # per-step anchor mask changes AT the seams
        seams = np.nonzero(np.diff(amt.astype(np.int16)) != 0)[0]
        if seams.size:
            seam_speed = float(root_speed[np.clip(seams, 0, root_speed.size - 1)].max())

    return {
        "n": int(y.shape[0]),
        "cmd_speed": float(np.linalg.norm(cmd[:, :2], axis=-1).max()),
        "err_steady_p50": _pct(speed_err[steady], 50), "err_steady_p95": _pct(speed_err[steady], 95),
        "err_trans_p50": _pct(speed_err[~steady], 50), "err_trans_p95": _pct(speed_err[~steady], 95),
        "err_wz_p95": _pct(np.abs(err[:, 2]), 95),
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
    p.add_argument("--top", type=int, default=10, help="List this many worst clips per ranking key.")
    p.add_argument("--ramp-s", type=float, default=0.4,
                   help="Regime window (s) around a command change; matches mb_env.regime_weight's default.")
    p.add_argument("--max-clips", type=int, default=0,
                   help="Evenly subsample to at most this many clips (0 = all). For a 10k library.")
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
                          args.min_stance, args.stance_edge)
        r["clip"] = i
        r["type"] = _TERRAIN_TYPE_NAMES[int(tcode[i])] if int(tcode[i]) < len(_TERRAIN_TYPE_NAMES) else "?"
        (flat_rows if r["type"] == "flat" else terr_rows).append(r)

    common = [("slide_p95", "m/stance"), ("slide_max", "m/stance"), ("slide_speed_p95", "m/s"),
              ("contact_duty", "frac"), ("stance_s_med", "s"), ("n_stances", "-"),
              ("root_speed_max", "m/s"), ("seam_speed_max", "m/s"),
              ("pelvis_z_min", "m"), ("pelvis_z_max", "m")]
    tracking = [("err_steady_p50", "m/s"), ("err_steady_p95", "m/s"),
                ("err_trans_p50", "m/s"), ("err_trans_p95", "m/s"), ("err_wz_p95", "rad/s"),
                ("steady_frac", "-")]
    if flat_rows:
        print(f"\n[mb_qc] === FLAT clips ===")
        _summarize(flat_rows, "command tracking (steady vs transition regime)", tracking)
        _summarize(flat_rows, "artefacts", common)
        _speed_buckets(flat_rows, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 99.0])
    if terr_rows:
        print(f"\n[mb_qc] === TERRAIN clips (tracking error is EXPECTED: the command is held at the "
              f"approach speed through the traversal) ===")
        _summarize(terr_rows, "artefacts", common)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
