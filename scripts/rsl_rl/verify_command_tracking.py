#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does a clip's MOTION track the command the RL sees, where it is supposed to?

The contract the terrain families are built to:

* the RL-visible command may change ANYWHERE in a clip;
* the motion tracks it everywhere EXCEPT an override window -- the last ``--ramp-s`` of the approach,
  and the terrain interaction itself -- where the terrain dictates what the robot does instead.

So "is the command aligned with the motion" is a PER-PHASE question, and a whole-clip statistic
cannot answer it. Phases come from ``anchor_mask_t``, which already marks them per step (FLAT for the
approach and exit, BOX for the interaction), so nothing has to be re-derived from the geometry.

Two numbers per phase, both in m/s so they are readable as speeds rather than as a unitless score:

* ``mean|err|`` -- over every step in the phase;
* ``settled``   -- excluding ``--settle-s`` after each command change. A robot cannot step-change its
  velocity, so a fresh command is followed by an unavoidable acceleration transient; counting it as
  tracking error measures the accelerometer, not the alignment.

CAVEAT on the ``vs flat`` column: the achieved speed here is the pelvis's WORLD dx/dt, which equals
the commanded FORWARD speed only while the robot is not turning. The terrain families are authored
straight along the tile's +x so theirs is clean, but the flat family commands yaw and lateral motion,
which inflates its apparent error. Read flat as a loose upper bound, not as a floor to beat.

Deliberately NOT a correlation: correlation is scale- and offset-blind (two signals can correlate 1.0
while one is triple the other) and is destroyed by the phase lag that acceleration necessarily
introduces, so it reports badly on a correctly-tracking clip. Ask for the speed error directly.

    python scripts/rsl_rl/verify_command_tracking.py --clip clips/g1_mb_quick_v6d.h5 --plot-clips 3
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

_FLAT_MASK = 35                 # motion_gen.world_lift._ANCHOR_MASK_FLAT: approach / exit
_BOX_MASK = 2                   # _ANCHOR_MASK_BOX: the terrain interaction
_NAMES = ["flat", "terrain", "transition", "slope", "box"]


def _phases(mask_t: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Boolean step masks for ``approach`` / ``interaction`` / ``exit`` from ``anchor_mask_t``.

    A terrain clip is FLAT -> BOX -> FLAT, so the interaction is the BOX run and the two flat runs
    around it are the approach and the exit. A clip with no BOX steps (flat family) is all approach;
    one that is entirely BOX (a pure stair tile) is all interaction.
    """
    m = np.asarray(mask_t[:n]).astype(int)
    box = m == _BOX_MASK
    out = {"approach": np.zeros(n, bool), "interaction": box.copy(), "exit": np.zeros(n, bool)}
    if not box.any():
        out["approach"][:] = True
        return out
    first, last = int(np.argmax(box)), int(n - 1 - np.argmax(box[::-1]))
    out["approach"][:first] = True
    out["exit"][last + 1:] = True
    return out


def _achieved(dy: np.ndarray, n: int, smooth: int) -> np.ndarray:
    """Achieved forward speed: the pelvis world dx/dt the library stores, boxcar-smoothed.

    Smoothed because a per-step velocity carries the gait's within-stride ripple (a leg swings, the
    pelvis surges) that no command describes and no policy is asked to follow.
    """
    v = np.asarray(dy[:n, 0], dtype=np.float64)
    if smooth <= 1:
        return v
    k = np.ones(int(smooth)) / float(smooth)
    return np.convolve(v, k, mode="same")


def _settled(cmd: np.ndarray, n: int, settle_steps: int) -> np.ndarray:
    """Steps at least ``settle_steps`` after the last command CHANGE (the acceleration transient)."""
    ok = np.ones(n, bool)
    changes = np.flatnonzero(np.abs(np.diff(cmd[:n])) > 1e-6) + 1
    for c in changes:
        ok[c:c + int(settle_steps)] = False
    ok[:int(settle_steps)] = False                      # the initial ramp from a stand
    return ok


def analyse(path: str, ramp_s: float, settle_s: float, smooth_s: float) -> dict:
    """Per-family, per-phase tracking error for a clip library."""
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        code = np.asarray(f["terrain_type_code"])
        clen = np.asarray(f["clip_len"]).astype(int)
        tv, dy, amt = f["target_vel"], f["dy_des"], f["anchor_mask_t"]
        ramp_steps = int(round(float(ramp_s) / dt))
        settle_steps = int(round(float(settle_s) / dt))
        smooth = max(1, int(round(float(smooth_s) / dt)))
        rows: dict[int, dict] = {}
        for c in np.unique(code):
            acc = {p: {"all": [], "settled": []} for p in ("approach", "interaction", "exit")}
            for i in np.flatnonzero(code == c):
                n = int(clen[i])
                if n < 60:
                    continue
                cmd = np.asarray(tv[i])[:n, 0].astype(np.float64)
                ach = _achieved(np.asarray(dy[i]), n, smooth)
                ph = _phases(np.asarray(amt[i]), n)
                # The last ramp_steps of the approach are the terrain override -- excluded from the
                # approach, because the motion is SUPPOSED to leave the command there.
                appr = ph["approach"].copy()
                idx = np.flatnonzero(appr)
                if idx.size:
                    appr[idx[-1] - ramp_steps + 1:idx[-1] + 1] = False
                settle = _settled(cmd, n, settle_steps)
                err = np.abs(ach - cmd)
                for name, m in (("approach", appr), ("interaction", ph["interaction"]),
                                ("exit", ph["exit"])):
                    if m.sum() >= 10:
                        acc[name]["all"].append(float(err[m].mean()))
                        ms = m & settle
                        if ms.sum() >= 10:
                            acc[name]["settled"].append(float(err[ms].mean()))
            rows[int(c)] = acc
        return {"rows": rows, "dt": dt}


def report(res: dict) -> None:
    """Print the per-family, per-phase table."""
    # The yardstick is FLAT, not zero. A flat clip's motion has always been generated by tracking
    # its schedule with nothing overriding it, so its error is this pipeline's INTRINSIC
    # command-following gap -- mostly the target pose being snapped to a discrete stride bank (the
    # reason --rl-vels raw|snapped exists). An approach that matches it is tracking as well as
    # anything here can.
    base = None
    flat = res["rows"].get(0, {}).get("approach", {}).get("settled")
    if flat:
        base = float(np.mean(flat))

    print(f"{'family':<12s}{'phase':<13s}{'clips':>6s}{'mean|err|':>11s}{'settled':>10s}"
          f"{'vs flat':>9s}")
    print("-" * 61)
    for c, acc in sorted(res["rows"].items()):
        first = True
        for phase in ("approach", "interaction", "exit"):
            a, s = acc[phase]["all"], acc[phase]["settled"]
            if not a:
                continue
            sv = float(np.mean(s)) if s else float("nan")
            ratio = (f"{sv / base:.2f}x" if base and np.isfinite(sv) else "--")
            print(f"{(_NAMES[c] if first else ''):<12s}{phase:<13s}{len(a):>6d}"
                  f"{np.mean(a):>11.3f}{sv:>10.3f}{ratio:>9s}")
            first = False
        print()
    if base:
        print(f"all figures m/s. FLAT's own approach error ({base:.3f} m/s) is the baseline: its")
        print("motion tracks its schedule with nothing overriding, so that IS this pipeline's")
        print("intrinsic command-following gap (stride-bank snapping). Approach and exit near 1x")
        print("means aligned; interaction well above 1x is the terrain override, as intended.")


def plot(path: str, out: str, n_clips: int, family: int, ramp_s: float, smooth_s: float) -> None:
    """Commanded vs achieved speed for a few sample clips, phases shaded."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"})
    try:
        _f, _a = plt.subplots(); _a.set_title(r"$\alpha$"); _f.canvas.draw(); plt.close(_f)
    except Exception:
        print("LaTeX unavailable -- falling back to mathtext.")
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams.update({
        "lines.linewidth": 3, "axes.linewidth": 1.5, "axes.labelsize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "xtick.direction": "in",
        "ytick.direction": "in", "legend.fontsize": 14, "legend.framealpha": 0.9,
        "legend.edgecolor": "0.4", "legend.fancybox": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
    C = {"cmd": "#0077BB", "ach": "#EE7733", "ovr": "#CC3311", "terr": "#BBBBBB"}

    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        code = np.asarray(f["terrain_type_code"]); clen = np.asarray(f["clip_len"]).astype(int)
        ids = [int(i) for i in np.flatnonzero(code == family)][:int(n_clips)]
        if not ids:
            print(f"[plot] no clips of family {_NAMES[family]}"); return
        smooth = max(1, int(round(float(smooth_s) / dt)))
        ramp_steps = int(round(float(ramp_s) / dt))
        fig, axes = plt.subplots(len(ids), 1, figsize=(11.0, 2.7 * len(ids)), squeeze=False)
        for ax, i in zip(axes[:, 0], ids):
            n = int(clen[i]); t = np.arange(n) * dt
            cmd = np.asarray(f["target_vel"][i])[:n, 0]
            ach = _achieved(np.asarray(f["dy_des"][i]), n, smooth)
            ph = _phases(np.asarray(f["anchor_mask_t"][i]), n)
            if ph["interaction"].any():
                a, b = np.flatnonzero(ph["interaction"])[[0, -1]]
                ax.axvspan(t[a], t[b], color=C["terr"], alpha=0.55, lw=0,
                           label=r"terrain (override)")
                ap = np.flatnonzero(ph["approach"])
                if ap.size:
                    lo = max(0, ap[-1] - ramp_steps + 1)
                    ax.axvspan(t[lo], t[ap[-1]], color=C["ovr"], alpha=0.18, lw=0,
                               label=r"approach ramp (override)")
            ax.plot(t, cmd, color=C["cmd"], lw=3, label=r"commanded (what the RL sees)")
            ax.plot(t, ach, color=C["ach"], lw=2.5, label=r"achieved (motion)")
            ax.set_ylabel(r"$v_x$ (m/s)")
            ax.set_xlim(t[0], t[-1])
            ax.grid(True, alpha=0.3, linewidth=0.8)
            ax.annotate(rf"\texttt{{{_NAMES[family]}}} clip {i}", xy=(0.012, 0.88),
                        xycoords="axes fraction", fontsize=14, va="top")
        axes[0, 0].legend(loc="upper right", ncol=2)
        axes[-1, 0].set_xlabel(r"Time (s)")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{out}.{ext}")
        print(f"[plot] wrote {out}.png / .svg  (clips {ids})")


def main() -> int:
    """Measure and plot command-vs-motion agreement per phase."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Clip library .h5 (no default on purpose).")
    p.add_argument("--ramp-s", type=float, default=1.0,
                   help="Approach override window (s) before the terrain -- must match "
                        "mb_terrain_segments --ramp-s, which is what generated the motion.")
    p.add_argument("--settle-s", type=float, default=1.5,
                   help="Ignored after each command change: the acceleration transient no robot can "
                        "skip. The 'settled' column is the steady-state tracking error. 1.5 s is "
                        "MEASURED, not guessed -- at 0.6 s a correctly-tracking transition approach "
                        "still reads 0.725 m/s of 'error', at 1.5 s it reads 0.188. Note what that "
                        "implies: a command segment SHORTER than ~1.5 s is one the motion never "
                        "reaches, so its alignment is nominal only.")
    p.add_argument("--smooth-s", type=float, default=0.5,
                   help="Boxcar over the achieved speed, to drop within-stride ripple.")
    p.add_argument("--plot-clips", type=int, default=0, help="Sample clips to plot (0 = none).")
    p.add_argument("--plot-family", default="transition", choices=_NAMES,
                   help="Which family to plot.")
    p.add_argument("--out", default="logs/command_tracking", help="Plot path stem.")
    args = p.parse_args()

    if not os.path.exists(args.clip):
        raise SystemExit(f"clip library not found: {args.clip}")
    print(f"{args.clip}\n  ramp {args.ramp_s}s (override)  settle {args.settle_s}s  "
          f"smooth {args.smooth_s}s\n")
    res = analyse(args.clip, args.ramp_s, args.settle_s, args.smooth_s)
    report(res)
    if args.plot_clips > 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plot(args.clip, args.out, args.plot_clips, _NAMES.index(args.plot_family),
             args.ramp_s, args.smooth_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
