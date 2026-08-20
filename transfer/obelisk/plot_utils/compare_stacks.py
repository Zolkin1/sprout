# Copyright 2026 Zachary Olkin. All rights reserved.

"""Overlay an obelisk run and a MuJoCo sim2sim run, statically or tick by tick.

The two stacks run the same artifact pair and behave differently, and until now each had its
own plotting pipeline: figures from a bag on one side, figures from a CSV on the other, and a
human comparing two sets of PNGs. This puts both on one pair of axes.

Both sources present the same interface (``bag_run.Run`` / ``sim_run.SimRun``), so every trace
here is computed once and drawn twice. That is the point: a comparison whose two halves are
produced by different code is a comparison of the code as much as of the runs.

WHAT MUST BE TRUE FOR THE COMPARISON TO MEAN ANYTHING, each of which has been got wrong here
before:

* TIME. Both are shifted onto their own ARM instant. A bag's clock starts at launch, ~10 s
  before the controller arms, and the sim2sim runner drives from step 0 -- taking two zeros
  to mean the same instant puts the stacks seconds apart in the same profile.
* JOINT ORDER. Both sides are indexed in the POLICY's order and the names are compared, not
  assumed. The robot's order differs on 27 of the G1's 29 joints, and subtracting one from
  the other yields a large, smooth, entirely fictitious error.
* THE SAMPLER. The FM is stochastic and the C++ NoiseSource is not a bit-port of numpy's, so
  two stacks each drawing their own x0 differ by the DRAW. Run both with zero noise (the
  sim2sim default; the C++ controller's `zero_noise` parameter, default true) or this
  measures the sampling.
* THE VIEWER. A headed sim2sim run is a different trajectory from a headless one, measured.
  Pick one mode for both arms of a comparison.

Usage::

    python plot_utils/compare_stacks.py --bag <bag> --stream <replay dir>
    python plot_utils/compare_stacks.py --bag <bag> --stream <dir> --interactive
    python plot_utils/compare_stacks.py --stream <dir_a> --stream <dir_b>   # sim vs sim
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: Per-run styling. Distinct colour AND linestyle, so the figures survive being printed or
#: read by someone who cannot distinguish the two hues.
STYLES = [
    {"color": "tab:blue", "ls": "-"},
    {"color": "tab:red", "ls": "--"},
]


def _aligned_time(run) -> np.ndarray:
    """Time from the run's own ARM instant.

    Args:
        run: A ``Run`` or ``SimRun``.

    Returns:
        ``t`` shifted so 0 is the first armed tick (or the first tick, if the run never
        reports arming -- a sim2sim run is armed throughout by construction).
    """
    armed = np.asarray(run.armed, dtype=bool) if run.armed is not None else None
    if armed is None or not armed.any():
        return np.asarray(run.t) - run.t[0]
    return np.asarray(run.t) - run.t[int(np.argmax(armed))]


def _common_span(runs) -> tuple[float, float]:
    """The time range both runs actually cover, in arm-aligned seconds.

    Overlaying a 6 s sim2sim run on a 50 s bag squeezes the interesting part into the first
    eighth of the axis, so the figures trim to the intersection by default. It is the
    INTERSECTION rather than the shorter run's own span because the two need not start at the
    same offset from arm -- a bag begins several seconds BEFORE arming.

    Args:
        runs: The runs.

    Returns:
        ``(lo, hi)`` seconds from arm.
    """
    spans = [_aligned_time(r) for r in runs]
    return float(max(t[0] for t in spans)), float(min(t[-1] for t in spans))


def _autoscale_y_to_span(ax, span: tuple[float, float]) -> None:
    """Rescale y to the data INSIDE ``span``, after x has been trimmed.

    ``set_xlim`` changes the view, not the autoscale: matplotlib keeps sizing y from every
    point in the artist, so trimming a 14 s run against a 33 s one left the height-error panel
    spanning -0.6..0.8 for data that never leaves +-0.1 inside the visible window. The
    interesting signal then occupies a tenth of the axis and looks flat.

    Lines with two points are skipped: those are the axhline/axvline guides, whose x data is in
    axes coordinates and would mask to nothing.

    Args:
        ax: The axis, already x-limited.
        span: ``(lo, hi)`` in the same units as the x data.
    """
    lo, hi = span
    vals = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size <= 2 or y.size != x.size:
            continue
        m = (x >= lo) & (x <= hi) & np.isfinite(y)
        if m.any():
            vals.append(y[m])
    if not vals:
        return
    flat = np.concatenate(vals)
    y0, y1 = float(flat.min()), float(flat.max())
    pad = max(0.05 * (y1 - y0), 1e-3)
    ax.set_ylim(y0 - pad, y1 + pad)


def _load(bags: list[str], streams: list[str]) -> list:
    """Build the run objects, in the order they were given.

    Args:
        bags: Bag directories or .mcap paths.
        streams: Replay-stream directories.

    Returns:
        The runs, each carrying a ``label``.

    Raises:
        SystemExit: On fewer than two runs -- there is nothing to compare.
    """
    runs = []
    for b in bags:
        from bag_reader import read_bag
        from bag_run import Run
        r = Run(read_bag(b))
        r.label = f"obelisk:{os.path.basename(b.rstrip('/'))}"
        runs.append(r)
    for s in streams:
        from sim_run import SimRun
        r = SimRun(s)
        r.label = f"sim2sim:{os.path.basename(s.rstrip('/'))}"
        runs.append(r)
    if len(runs) < 2:
        sys.exit("[ERROR] give at least two runs (--bag and/or --stream); one run is not a "
                 "comparison.")
    return runs


def _check_joint_orders(runs: list) -> None:
    """Refuse to compare joints across two different orders.

    Raises:
        SystemExit: If the runs disagree on the joint names. Plotting anyway would produce
            smooth, plausible, wrong per-joint traces -- the exact failure the bag plotter
            already refuses.
    """
    names = [tuple(r.joint_names_isaac or ()) for r in runs]
    if len({n for n in names if n}) > 1:
        sys.exit("[ERROR] the runs report different joint orders, so per-joint traces would "
                 "compare different joints:\n"
                 + "\n".join(f"  {r.label}: {list(n)[:4]}..." for r, n in zip(runs, names)))


def fig_command(runs, plt):
    """Commanded velocity, overlaid. The input both stacks were given."""
    fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for r, st in zip(runs, STYLES):
        t = _aligned_time(r)
        for i, name in enumerate(("v_x", "v_y", "w_z")):
            ax[i].plot(t, r.command[:, i], lw=1.0, label=r.label, **st)
            ax[i].set_ylabel(name)
    ax[-1].set_xlabel("time from arm [s]")
    ax[0].set_title("commanded velocity -- if these differ, nothing below is comparable")
    return fig, ax


def fig_tracking(runs, plt):
    """Reference-vs-actual base error at offset 0, position and orientation.

    Separately, because they fail differently and an aggregate norm hides which one moved.
    """
    fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    ax = ax.ravel()
    for r, st in zip(runs, STYLES):
        t = _aligned_time(r)
        if r.ref_pos_err is not None:
            for i, name in enumerate("xyz"):
                ax[i].plot(t, r.ref_pos_err[:, 0, i], lw=0.9, label=r.label, **st)
                ax[i].set_ylabel(f"pos {name} [m]")
        if r.ref_rpy_err is not None:
            for i, name in enumerate(("roll", "pitch", "yaw")):
                ax[3 + i].plot(t, r.ref_rpy_err[:, 0, i], lw=0.9, label=r.label, **st)
                ax[3 + i].set_ylabel(f"{name} [rad]")
    for a in ax[3:]:
        a.set_xlabel("time from arm [s]")
    ax[0].set_title("base reference error at offset 0")
    return fig, ax


def fig_error_norm(runs, plt):
    """One trace per run: the norm of the base position and orientation error.

    The summary view. The trained termination bounds are drawn, because "worse" is only
    meaningful against the threshold the policy was trained to respect.
    """
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for r, st in zip(runs, STYLES):
        t = _aligned_time(r)
        if r.ref_pos_err is not None:
            ax[0].plot(t, np.linalg.norm(r.ref_pos_err[:, 0, :], axis=1), lw=1.0,
                       label=r.label, **st)
        if r.ref_rpy_err is not None:
            ax[1].plot(t, np.linalg.norm(r.ref_rpy_err[:, 0, :], axis=1), lw=1.0,
                       label=r.label, **st)
    # The teacher's own DoneTerm bounds (g1_teacher_env_cfg): 0.3 m and 0.8 rad.
    ax[0].axhline(0.3, color="k", ls=":", lw=1.0, label="trained bound 0.3 m")
    ax[1].axhline(0.8, color="k", ls=":", lw=1.0, label="trained bound 0.8 rad")
    ax[0].set_ylabel("|pos err| [m]")
    ax[1].set_ylabel("|ori err| [rad]")
    ax[1].set_xlabel("time from arm [s]")
    ax[0].set_title("base tracking error against the bounds the policy was trained under")
    return fig, ax


def fig_joints(runs, plt):
    """Per-joint tracking error, and the worst joints named.

    An aggregate says something is wrong; this says which joint, which is the only version
    that leads anywhere.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    worst_lines = []
    for r, st in zip(runs, STYLES):
        t = _aligned_time(r)
        if r.ref_joint is None or r.joint_pos_isaac is None:
            continue
        err = np.abs(r.ref_joint[:, 0, :] - r.joint_pos_isaac)
        ax[0].plot(t, np.linalg.norm(err, axis=1), lw=1.0, label=r.label, **st)
        rms = np.sqrt(np.nanmean(err ** 2, axis=0))
        order = np.argsort(rms)[::-1][:5]
        worst_lines.append(f"{r.label}: " + ", ".join(
            f"{r.joint_names_isaac[j]} {rms[j]:.3f}" for j in order))
        for j in order:
            ax[1].plot(t, err[:, j], lw=0.8, alpha=0.8,
                       label=f"{r.label} {r.joint_names_isaac[j]}", **st)
    ax[0].set_ylabel("|joint err| [rad]")
    ax[1].set_ylabel("worst joints [rad]")
    ax[1].set_xlabel("time from arm [s]")
    ax[0].set_title("joint tracking error (top 5 RMS joints per run below)")
    for line in worst_lines:
        print(f"[compare] worst joints by RMS -- {line}")
    return fig, ax


FIGURES = {"command": fig_command, "tracking": fig_tracking,
           "error_norm": fig_error_norm, "joints": fig_joints}


class CompareStepper:
    """Tick-by-tick, both stacks at once, one shared cursor.

    Rows are runs and columns are panels, so the eye compares down a column: the same camera
    at the same instant in both stacks, then the same predicted path, then the same joints.
    The bottom row is shared: the base position error PER AXIS -- downrange, lateral, height
    -- plus the norm, each carrying the cursor. Those axes are the anchor's YAW-ALIGNED frame,
    so z is gravity and a torso pitch does not rotate them.

    Keys match ``plot_from_bag``'s stepper: LEFT/RIGHT one tick, UP/DOWN ten, PAGEUP/PAGEDOWN
    a hundred, HOME/END the ends.

    Args:
        runs: Two or more runs.
        plt: The pyplot module (imported by the caller, after the backend is chosen).
    """

    def __init__(self, runs: list, plt, trim: bool = True) -> None:
        self.runs = runs
        self.plt = plt
        #: Trim the shared traces to the span both runs cover. The CURSOR was always confined
        #: to the intersection, but the traces were drawn over each run's full extent -- so a
        #: 14 s sim against a 33 s bag showed a cursor that stopped less than halfway across
        #: the axis, which reads as the stepper being stuck rather than the runs being
        #: different lengths.
        self.trim = trim
        self.times = [_aligned_time(r) for r in runs]
        # The cursor is a TIME, not an index: the two runs need not have the same tick count
        # or the same start, and stepping by index would slide them apart.
        self.t_now = float(max(tt[0] for tt in self.times))
        self.t_min = self.t_now
        self.t_max = float(min(tt[-1] for tt in self.times))
        self.dt = float(np.median(np.diff(self.times[0]))) if len(self.times[0]) > 1 else 0.02

        n = len(runs)
        self.fig = plt.figure(figsize=(15, 3.2 * n + 2.4))
        gs = self.fig.add_gridspec(n + 1, 4, height_ratios=[3] * n + [2])
        self.ax = [[self.fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(n)]
        # Bottom row: the base position error SPLIT PER AXIS, plus the norm. The axes are the
        # anchor's YAW-ALIGNED frame -- heading-relative with z along gravity -- because that
        # is the frame the window's base_pose block is expressed in
        # (`pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d`). So x is downrange, y is
        # lateral and z is height, and none of them is the full body frame: a torso pitch does
        # not rotate these. A norm alone hides which of the three moved, and on a staircase it
        # is usually z or downrange specifically.
        self.ax_err = [self.fig.add_subplot(gs[n, j]) for j in range(3)]
        self.ax_trace = self.fig.add_subplot(gs[n, 3])
        self._cid = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        # Held so the connection is not garbage-collected -- the window would appear fully
        # drawn with dead arrow keys, which looks like a frozen plot rather than a lost
        # callback. Same reason plot_from_bag's Stepper keeps its own.
        self._keep_alive = self._cid
        self._draw()

    def _on_key(self, event) -> None:
        """Move the cursor."""
        step = {"left": -1, "right": 1, "up": 10, "down": -10,
                "pageup": 100, "pagedown": -100}.get(event.key)
        if step is not None:
            self.t_now = float(np.clip(self.t_now + step * self.dt, self.t_min, self.t_max))
        elif event.key == "home":
            self.t_now = self.t_min
        elif event.key == "end":
            self.t_now = self.t_max
        else:
            return
        self._draw()

    def _draw(self) -> None:
        """Redraw every panel at the current cursor time."""
        for row, (run, t) in enumerate(zip(self.runs, self.times)):
            i = int(np.argmin(np.abs(t - self.t_now)))
            for a in self.ax[row]:
                a.clear()
            # Both cameras.
            for cam, name in ((0, "near"), (1, "far")):
                a = self.ax[row][cam]
                img = run.depth_at(i, cam) if hasattr(run, "depth_at") else None
                if img is None:
                    a.text(0.5, 0.5, f"no {name} depth", ha="center", va="center",
                           transform=a.transAxes, fontsize="small")
                else:
                    a.imshow(img, aspect="auto")
                a.set_title(f"{run.label} {name}", fontsize="small")
                a.set_xticks([]); a.set_yticks([])
            # The predicted root path in the window's own frame: where the reference asks the
            # base to go over the horizon, relative to where the base is now.
            a = self.ax[row][2]
            if run.ref_pos_err is not None:
                p = run.ref_pos_err[i]
                a.plot(p[:, 0], p[:, 1], "o-", ms=3, lw=1.0, color=STYLES[row]["color"])
                a.plot(0, 0, "k+", ms=8)
                a.set_xlabel("x [m]", fontsize="small"); a.set_ylabel("y [m]", fontsize="small")
            a.set_title("reference path (offsets)", fontsize="small")
            a.grid(alpha=0.3)
            # Joints now: reference against measured, by index.
            a = self.ax[row][3]
            if run.ref_joint is not None and run.joint_pos_isaac is not None:
                idx = np.arange(run.num_joints)
                a.plot(idx, run.ref_joint[i, 0, :], "o", ms=3, label="reference")
                a.plot(idx, run.joint_pos_isaac[i], "x", ms=4, label="measured")
                a.legend(fontsize="xx-small")
            a.set_title(f"joints @ t={t[i]:+.2f}s (tick {i})", fontsize="small")
            a.grid(alpha=0.3)

        # The shared traces, with the cursor on each.
        for a in self.ax_err:
            a.clear()
        self.ax_trace.clear()
        axis_names = ("x downrange", "y lateral", "z height")
        for run, t, st in zip(self.runs, self.times, STYLES):
            if run.ref_pos_err is None:
                continue
            for i in range(3):
                self.ax_err[i].plot(t, run.ref_pos_err[:, 0, i], lw=1.0, label=run.label, **st)
            self.ax_trace.plot(t, np.linalg.norm(run.ref_pos_err[:, 0, :], axis=1),
                               lw=1.0, label=run.label, **st)
        for i, name in enumerate(axis_names):
            self.ax_err[i].axhline(0.0, color="k", lw=0.6, alpha=0.5)
            self.ax_err[i].axvline(self.t_now, color="k", lw=1.2)
            self.ax_err[i].set_xlabel("time from arm [s]")
            self.ax_err[i].set_ylabel(f"{name} [m]")
            self.ax_err[i].grid(alpha=0.3)
        self.ax_err[0].legend(fontsize="xx-small", ncol=len(self.runs))
        # The bound is on the NORM, so it is drawn only here -- putting 0.3 on a per-axis
        # panel would imply a per-axis budget the termination does not define.
        self.ax_trace.axhline(0.3, color="k", ls=":", lw=0.8, label="bound 0.3 m")
        self.ax_trace.axvline(self.t_now, color="k", lw=1.2)
        self.ax_trace.set_xlabel("time from arm [s]")
        self.ax_trace.set_ylabel("|pos err| [m]")
        self.ax_trace.legend(fontsize="xx-small")
        self.ax_trace.grid(alpha=0.3)
        if self.trim:
            for a in (*self.ax_err, self.ax_trace):
                a.set_xlim(self.t_min, self.t_max)
                _autoscale_y_to_span(a, (self.t_min, self.t_max))
        self.fig.suptitle(f"t = {self.t_now:+.3f} s from arm    "
                          "LEFT/RIGHT 1 tick, UP/DOWN 10, PAGEUP/PAGEDOWN 100, HOME/END ends",
                          fontsize="medium")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


def main() -> int:
    """Load the runs and either save the figures or open the stepper."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", action="append", default=[],
                    help="An obelisk bag directory or .mcap. Repeatable.")
    ap.add_argument("--stream", action="append", default=[],
                    help="A sim2sim --dump_replay directory. Repeatable.")
    ap.add_argument("--interactive", action="store_true",
                    help="Open the tick stepper instead of writing figures.")
    ap.add_argument("--save_dir", default=None, help="Where the PNGs go.")
    ap.add_argument("--no_trim", action="store_true",
                    help="Do NOT trim the figures OR the stepper to the time range both runs "
                         "cover. The "
                         "default trims, because a 6 s run overlaid on a 50 s one is squeezed "
                         "into a corner of the axis.")
    ap.add_argument("--figures", default=None,
                    help=f"Comma-separated subset of {','.join(FIGURES)}.")
    args = ap.parse_args()

    if not args.interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _load(args.bag, args.stream)
    _check_joint_orders(runs)
    for r in runs:
        t = _aligned_time(r)
        print(f"[compare] {r.label}: {len(t)} ticks, t {t[0]:+.2f}..{t[-1]:+.2f} s from arm")

    if args.interactive:
        backend = matplotlib.get_backend()
        if backend.lower() in ("agg", "pdf", "ps", "svg", "template"):
            sys.exit(f"[ERROR] matplotlib is on the non-interactive '{backend}' backend, so a "
                     "window would open with dead arrow keys. Install a GUI backend "
                     "(`pip install PyQt5`) or set MPLBACKEND=TkAgg, and make sure DISPLAY is "
                     "set if you are over ssh.")
        stepper = CompareStepper(runs, plt, trim=not args.no_trim)   # keep the reference
        print(f"[compare] stepper on '{backend}': LEFT/RIGHT 1 tick, UP/DOWN 10, "
              "PAGEUP/PAGEDOWN 100, HOME/END ends")
        plt.show()
        return 0 if stepper else 0

    span = _common_span(runs)
    if not args.no_trim:
        print(f"[compare] trimming the figures to the shared span "
              f"{span[0]:+.2f}..{span[1]:+.2f} s from arm (--no_trim to keep both in full)")

    wanted = args.figures.split(",") if args.figures else list(FIGURES)
    save_dir = args.save_dir or os.path.join(os.getcwd(), "compare_plots")
    os.makedirs(save_dir, exist_ok=True)
    for name in wanted:
        if name not in FIGURES:
            sys.exit(f"[ERROR] unknown figure {name!r}; have {', '.join(FIGURES)}")
        fig, axes = FIGURES[name](runs, plt)
        for a in np.atleast_1d(axes).ravel():
            a.grid(alpha=0.3)
            if not args.no_trim:
                a.set_xlim(*span)
                _autoscale_y_to_span(a, span)
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize="x-small", ncol=2)
        fig.tight_layout()
        out = os.path.join(save_dir, f"compare_{name}.png")
        fig.savefig(out, dpi=120)
        print(f"[compare] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
