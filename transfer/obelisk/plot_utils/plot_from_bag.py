#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Plot a recorded obelisk run, or step through it frame by frame.

**Runs outside the container.** MCAP embeds every message schema in the bag, so this needs
no ROS workspace and no ``metadata.yaml``::

    pip install mcap mcap-ros2-support matplotlib numpy

Figures are written NEXT TO THE BAG (``<bag>/plots/``) so runs never overwrite each other,
split by the question they answer:

``health.png``      was the stack healthy? arming, watchdogs, input ages, loop period and
                    latency against budget, per-camera rate and invalid fraction.
``command.png``     commanded vs achieved base velocity.
``tracking.png``    reference-window base error, ONE AXIS PER SUBPLOT: x/y/z and
                    roll/pitch/yaw, so it is clear WHICH degree of freedom drifted.
``joints.png``      one subplot PER JOINT -- reference, measured and PD target together,
                    labelled by joint name. An aggregate error says something is wrong;
                    this says which joint.
``reference.png``   the reference window itself over time -- how far ahead it reaches and
                    whether it jumps between re-plans.
``generator.png``   re-plan cadence, cursor walk, and generator inference time. In debug-clip
                    mode this shows the clip step instead.

``--interactive`` opens a frame stepper instead: both depth images, the reference window's
root path and joint trajectory, the current joint positions against the reference, and the
velocity trace with a cursor. LEFT/RIGHT step one tick, UP/DOWN ten, PAGEUP/PAGEDOWN a
hundred, HOME/END jump to the ends.

Usage::

    python plot_utils/plot_from_bag.py                        # newest bag, all figures
    python plot_utils/plot_from_bag.py <bag>
    python plot_utils/plot_from_bag.py <bag> --interactive
    python plot_utils/plot_from_bag.py <bag> --start 5 --end 20
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bag_reader import read_bag  # noqa: E402
from bag_run import FOOT_FRAMES, TOPICS, Run  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_BAG_ROOT = _HERE.parent / "log" / "bags"

def find_latest_bag(root: pathlib.Path) -> str | None:
    """Newest bag directory under ``root``, or None."""
    bags = [p for p in glob.glob(str(root / "*")) if os.path.isdir(p)]
    return max(bags, key=os.path.getmtime) if bags else None


# ---------------------------------------------------------------------------
# Static figures
# ---------------------------------------------------------------------------

def fig_health(run: Run, plt):
    """Arming, watchdogs, input ages, loop timing, generator schedule and depth health."""
    fig, ax = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    m = run.ticks

    ax[0].step(run.t, run.armed, where="post", label="armed")
    ax[0].step(run.t, [x.clip_mode for x in m], where="post", label="clip mode")
    ax[0].step(run.t, [x.depth_stale for x in m], where="post", label="depth stale")
    ax[0].step(run.t, [x.vel_cmd_stale for x in m], where="post", label="vel cmd stale")
    # IMU hold. `getattr` because a bag recorded before these fields existed is still worth
    # plotting -- the mcap reader decodes with the BAG's schema, so the attribute is genuinely
    # absent there rather than zero.
    if hasattr(m[0], "quat_held"):
        ax[0].step(run.t, [x.quat_held for x in m], where="post", label="imu held")
        bad = [getattr(x, "bad_quat_samples", 0) for x in m]
        tot = [getattr(x, "quat_samples", 0) for x in m]
        if tot and tot[-1] > 0:
            # The cumulative rate, in the label rather than as a curve: what matters is
            # whether the IMU misbehaved at all and how often, not its shape in time.
            ax[0].plot([], [], " ",
                       label=f"bad IMU samples: {bad[-1]}/{tot[-1]} "
                             f"({100.0 * bad[-1] / tot[-1]:.2f}%)")
    ax[0].set_ylabel("state"); ax[0].set_ylim(-0.1, 1.5)

    # -1 is the "never arrived" sentinel; plotting it as -1000 ms swamps the axis.
    age = lambda v: np.where(np.asarray(v) < 0, np.nan, np.asarray(v) * 1e3)  # noqa: E731
    ax[1].plot(run.t, age([x.est_state_age for x in m]), label="est_state")
    for i in range(len(m[0].depth_age)):
        ax[1].plot(run.t, age([x.depth_age[i] for x in m]), label=f"depth[{i}]")
    # Capture-time pose matching. Plotted as capture age (how old the pixels are) and the
    # RESIDUAL pose_age - capture_age (how far the pose used sits from the moment they were
    # taken). The residual is the diagnostic: flat at ~0 means the pose came from the capture
    # instant, and anything else is the unprojection composing out a rotation the robot was not
    # in. Drawn as a difference rather than as two overlapping curves because the two agree to
    # a fraction of a millisecond when it works, which is invisible on a 100 ms axis.
    if hasattr(m[0], "depth_pose_age"):
        for i in range(len(m[0].depth_capture_age)):
            ax[1].plot(run.t, age([x.depth_capture_age[i] for x in m]), ls="--", lw=0.8,
                       label=f"capture[{i}]")
            # -1 is "the live pose was used" -- matching off, or the stamp was rejected as
            # unusable. Left as a gap: there is no residual to report, and drawing it as a
            # number would invent one.
            resid = [(x.depth_pose_age[i] - x.depth_capture_age[i]) * 1e3
                     if (i < len(x.depth_pose_age) and x.depth_pose_age[i] >= 0) else np.nan
                     for x in m]
            ax[1].plot(run.t, resid, lw=0.9, label=f"pose - capture[{i}]")
    ax[1].set_ylabel("input age [ms]\n(gaps = never arrived)")
    ax[1].legend(fontsize="xx-small", ncol=4)

    # The two networks separately: they are judged against different budgets (the tracker runs
    # inside the tick, the generator gets its whole lead window off-thread), so a single total
    # cannot say which one moved. `tick_period_ms` is the controller's own steady_clock gap and
    # is the trustworthy one -- diff(run.t) is over HEADER stamps, which on a split-machine run
    # come from a different box's clock.
    if len(run.t) > 1:
        ax[2].plot(run.t[1:], np.diff(run.t) * 1e3, lw=0.6, alpha=0.5,
                   label="period (header stamps)")
    ax[2].plot(run.t, run.tick_period_ms, lw=0.8, label="period (steady_clock)")
    ax[2].plot(run.t, run.latency, lw=0.8, label="tick total")
    ax[2].plot(run.t, run.trk_infer_ms, lw=0.8, label="tracker.onnx")
    # The generator holds the same value between re-plans; drawn stepped so the flat stretches
    # read as "did not run", not as a measurement.
    ax[2].step(run.t, run.fm_infer_ms, where="post", lw=0.8, label="fm.onnx (per re-plan)")
    ax[2].axhline(20.0, color="r", ls="--", lw=0.8, label="20 ms tick budget")
    ax[2].set_ylabel("[ms]")

    # Generator schedule health. Cumulative counters, so a flat line is a healthy stretch and
    # any slope at all is the generator failing to keep its cadence. `fm_late` is shaded
    # because it is the state, not a count -- the ticks where the reference stopped refreshing.
    ax[3].plot(run.t, run.fm_late_events, label="late events")
    ax[3].plot(run.t, run.fm_plans_dropped, label="plans dropped")
    ax[3].plot(run.t, run.fm_requests_skipped, label="requests skipped")
    ax[3].plot(run.t, run.fm_offsets_clamped, ls=":", label="offsets clamped")
    if run.fm_late.any():
        ax[3].fill_between(run.t, 0, 1, where=run.fm_late, transform=ax[3].get_xaxis_transform(),
                           color="r", alpha=0.15, label="late (state)")
    ax[3].set_ylabel("generator schedule\n(cumulative)")

    for i in range(len(m[0].depth_hz)):
        ax[4].plot(run.t, [x.depth_hz[i] for x in m], label=f"depth[{i}] Hz")
        ax[4].plot(run.t, [100.0 * x.depth_invalid_fraction[i] for x in m], ls=":",
                   label=f"depth[{i}] % invalid")
    ax[4].set_ylabel("depth health"); ax[4].set_xlabel("time [s]")
    return fig, ax


def fig_command(run: Run, plt):
    """Commanded vs achieved base velocity."""
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    achieved = (run.base_lin_b[:, 0], run.base_lin_b[:, 1], run.base_ang_b[:, 2])
    for i, (name, unit) in enumerate((("v_x", "m/s"), ("v_y", "m/s"), ("w_z", "rad/s"))):
        ax[i].plot(run.t, run.command[:, i], label=f"commanded {name}")
        ax[i].plot(run.t, achieved[i], lw=0.8, label=f"achieved {name}")
        ax[i].set_ylabel(f"{name} [{unit}]")
    ax[-1].set_xlabel("time [s]")
    return fig, ax


def fig_tracking(run: Run, plt):
    """Reference-window error, ONE AXIS PER SUBPLOT.

    Position and orientation both, because they fail differently and an aggregate norm hides
    which one moved. Every trace is the reference at offset 0 minus the robot, expressed in
    the robot's own yaw frame -- so zero is perfect and the sign says which way it is off.
    """
    fig, ax = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
    if run.ref_pos_err is not None:
        for i, name in enumerate("xyz"):
            ax[i].plot(run.t, run.ref_pos_err[:, 0, i], lw=0.9, label=f"base {name} err")
            ax[i].axhline(0.0, color="r", lw=0.6)
            ax[i].set_ylabel(f"{name} [m]")
    if run.ref_rpy_err is not None:
        for i, name in enumerate(("roll", "pitch", "yaw")):
            ax[3 + i].plot(run.t, run.ref_rpy_err[:, 0, i], lw=0.9,
                           color="tab:orange", label=f"base {name} err")
            ax[3 + i].axhline(0.0, color="r", lw=0.6)
            ax[3 + i].set_ylabel(f"{name} [rad]")
    ax[0].set_title("reference minus robot, at offset 0, in the robot's yaw frame",
                    fontsize="small")
    ax[-1].set_xlabel("time [s]")
    return fig, ax


def fig_joints(run: Run, plt):
    """One subplot PER JOINT: reference, measured and PD target together.

    An aggregate joint error says something is wrong; this says which joint, which is the
    only version you can act on. All three traces are in the POLICY's joint order and
    labelled with the joint's name -- the robot publishes a different order, and subtracting
    one from the other produces a large, smooth, entirely fictitious error.
    """
    ref, meas, tgt = run.ref_joint, run.joint_pos_isaac, run.joint_target_isaac
    if ref is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "no joint_pos block in this tracker's reference window",
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    nj = ref.shape[2]
    names = run.joint_names_isaac or [f"j{i}" for i in range(nj)]
    cols = 4
    rows = int(np.ceil(nj / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.1 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for j in range(nj):
        a = axes[j]
        a.plot(run.t, ref[:, 0, j], lw=0.9, label="reference")
        if meas is not None:
            a.plot(run.t, meas[:, j], lw=0.9, label="measured")
        if tgt is not None:
            a.plot(run.t, tgt[:, j], lw=0.7, ls=":", label="PD target")
        a.set_title(names[j].replace("_joint", ""), fontsize="x-small")
        a.tick_params(labelsize="xx-small")
    for a in axes[nj:]:
        a.axis("off")
    if meas is None:
        axes[0].set_title(f"{axes[0].get_title()}  [measured unavailable: joint order "
                          "unresolved]", fontsize="x-small", color="r")
    axes[0].legend(fontsize="xx-small")
    fig.supxlabel("time [s]"); fig.supylabel("[rad]")
    return fig, axes


def fig_reference(run: Run, plt):
    """The served reference window itself: reach, and whether it jumps."""
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    if run.ref_pos_err is not None:
        reach = np.linalg.norm(run.ref_pos_err[:, -1, :], axis=1)
        ax[0].plot(run.t, reach, lw=0.8,
                   label=f"|reference at offset {run.offsets[-1]}| (1 s ahead)")
        ax[0].plot(run.t, np.linalg.norm(run.ref_pos_err[:, 0, :], axis=1), lw=0.8,
                   label="|reference at offset 0| (now)")
    ax[0].set_ylabel("reach [m]")

    # Tick-to-tick change of the WHOLE window. A re-plan shows up as a spike; a smooth trace
    # means the reference is continuous across the cadence boundary.
    if run.window.size:
        jump = np.linalg.norm(np.diff(run.window.reshape(len(run.t), -1), axis=0), axis=1)
        ax[1].plot(run.t[1:], jump, lw=0.8, label="|window(t) - window(t-1)|")
    ax[1].set_ylabel("window step")

    if run.ref_joint is not None:
        for j in range(min(6, run.ref_joint.shape[2])):
            ax[2].plot(run.t, run.ref_joint[:, 0, j], lw=0.8, label=f"ref j{j} @0")
    ax[2].set_ylabel("reference joints [rad]"); ax[2].set_xlabel("time [s]")
    return fig, ax


def fig_generator(run: Run, plt):
    """Re-plan cadence and cursor walk, or the clip step in debug-clip mode."""
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(run.t, run.fm_cursor, lw=0.8, label="fm cursor (-1 = no plan)")
    ax[0].plot(run.t, run.clip_step, lw=0.8, label="clip step (-1 = no clip)")
    ax[0].set_ylabel("cursor")

    if run.plans:
        tp = np.array([t for t, _ in run.plans])
        if len(tp) > 1:
            ax[1].plot(tp[1:], np.diff(tp) * 1e3, ".", ms=3, label="re-plan interval [ms]")
        ax[1].plot(tp, [m.infer_ms for _, m in run.plans], ".", ms=3, label="fm infer [ms]")
    else:
        ax[1].text(0.5, 0.5, "no generator in this run (debug-clip mode)",
                   ha="center", va="center", transform=ax[1].transAxes)
    ax[1].set_ylabel("[ms]"); ax[1].set_xlabel("time [s]")
    return fig, ax


def fig_feet(run: Run, plt):
    """Foot position over the run: what the reference asked for, and where the foot was.

    The window carries no ``ee_pos`` block, so both sides are FK'd from joint angles in the
    BASE's frame -- see :meth:`bag_run.Run.feet_at` for why that frame and not a world one.
    A base-frame foot position is the gait itself: stride reach in x, stance width in y,
    swing height in z. Reading it against the reference is how a "tracks joints but walks
    badly" failure becomes visible, because the joint plot shows 29 small errors while this
    shows the one thing they add up to.
    """
    fig, ax = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    probe = run.feet_at(0)
    if probe is None:
        for a in ax.ravel():
            a.text(0.5, 0.5, "no FK available (pinocchio / URDF)",
                   ha="center", va="center", transform=a.transAxes)
        return fig, ax.ravel()

    n = len(run.t)
    ref = np.full((n, len(FOOT_FRAMES), 3), np.nan)
    act = np.full((n, len(FOOT_FRAMES), 3), np.nan)
    for i in range(n):
        got = run.feet_at(i)
        if got is not None:
            ref[i] = got[0][0]          # offset 0 == the reference for THIS tick
            act[i] = got[1]

    for f, foot in enumerate(FOOT_FRAMES):
        for c, name in enumerate("xyz"):
            a = ax[c, f]
            a.plot(run.t, ref[:, f, c], lw=1.0, label="reference")
            a.plot(run.t, act[:, f, c], lw=0.8, ls="--", label="measured")
            a.set_ylabel(f"{name} [m]")
            a.grid(alpha=0.3)
            if c == 0:
                a.set_title(foot, fontsize="small")
                a.legend(fontsize="x-small", ncol=2)
    err = np.linalg.norm(ref - act, axis=-1)
    for f in range(len(FOOT_FRAMES)):
        ax[2, f].set_xlabel("time [s]")
        finite = err[np.isfinite(err[:, f]), f]
        if finite.size:
            ax[0, f].set_title(f"{FOOT_FRAMES[f]}   |err| mean {finite.mean()*1e3:.0f} mm, "
                               f"p95 {np.percentile(finite, 95)*1e3:.0f} mm",
                               fontsize="small")
    # Flat: main() iterates the returned axes to apply shared styling, and a 2-D array
    # yields ROWS there rather than axes.
    return fig, ax.ravel()


FIGURES = {"health": fig_health, "command": fig_command, "tracking": fig_tracking,
           "joints": fig_joints, "reference": fig_reference,
           "generator": fig_generator, "feet": fig_feet}


# ---------------------------------------------------------------------------
# Interactive frame stepper
# ---------------------------------------------------------------------------

class Stepper:
    """Step through a run tick by tick, showing what the policy saw and did.

    The static figures answer "what happened over the run"; this answers "what was the robot
    looking at, and being asked for, at THIS instant" -- which is the question a fall
    actually poses.

    Args:
        run: The decoded run.
        plt: The pyplot module (imported by the caller, after the backend is chosen).
    """

    def __init__(self, run: Run, plt) -> None:
        self.run = run
        self.i = int(np.argmax(run.armed)) if run.armed.any() else 0
        # constrained layout, because four rows of titled panels overlap their neighbours'
        # x-labels otherwise -- and the panels most worth reading here are the small ones.
        self.fig = plt.figure(figsize=(16, 12), layout="constrained")
        gs = self.fig.add_gridspec(4, 4, height_ratios=[1.05, 0.95, 0.95, 0.7])

        self.ax_near = self.fig.add_subplot(gs[0, 0])
        self.ax_far = self.fig.add_subplot(gs[0, 1])
        self.ax_path = self.fig.add_subplot(gs[0, 2])
        self.ax_z = self.fig.add_subplot(gs[0, 3])
        # Row 1: the base_pose block SPLIT PER AXIS. A single x-y path hides which axis a
        # divergence is in, and z never appeared on it at all -- so a height error and a
        # downrange error looked the same.
        self.ax_ex = self.fig.add_subplot(gs[1, 0])
        self.ax_ey = self.fig.add_subplot(gs[1, 1])
        self.ax_ez = self.fig.add_subplot(gs[1, 2])
        self.ax_rpy = self.fig.add_subplot(gs[1, 3])
        # Row 2: the feet, FK'd (the window carries no ee_pos block), then the joints.
        self.ax_fl = self.fig.add_subplot(gs[2, 0])
        self.ax_fr = self.fig.add_subplot(gs[2, 1])
        self.ax_jnow = self.fig.add_subplot(gs[2, 2:])
        self.ax_vel = self.fig.add_subplot(gs[3, :])
        self._axes_dyn = (self.ax_near, self.ax_far, self.ax_path, self.ax_z,
                          self.ax_ex, self.ax_ey, self.ax_ez, self.ax_rpy,
                          self.ax_fl, self.ax_fr, self.ax_jnow)

        self._limits = self._compute_limits()
        self._init_static()
        self._cid = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        # Matplotlib's callback registry holds BOUND METHODS WEAKLY, so a Stepper nobody
        # keeps a reference to is collected the moment the constructor returns -- the window
        # still appears, fully drawn, and the arrow keys silently do nothing. Parking a
        # reference on the figure ties the handler's lifetime to the window's.
        self.fig._stepper_ref = self
        self._draw()

    def _compute_limits(self) -> dict:
        """Axis limits held FIXED across every tick.

        Autoscaling each frame is actively misleading in a stepper: the reference root path
        rescales as you step, so a window that shrank to a tenth of its size looks
        identical to one that did not, and a drift reads as stationary. Fixed limits make
        the eye do the comparison instead of the axis.

        Computed over the WHOLE run so nothing ever clips, with the origin (the robot)
        always included -- the path is meaningless without it.

        Returns:
            Limits keyed by panel; missing keys mean that panel has no data.
        """
        r, out = self.run, {}
        pad = lambda lo, hi, f=0.08: (lo - f * max(hi - lo, 1e-6),  # noqa: E731
                                      hi + f * max(hi - lo, 1e-6))

        if r.ref_pos_err is not None and np.isfinite(r.ref_pos_err).any():
            xy = r.ref_pos_err[:, :, :2]
            # Square and centred on the data+origin, so `equal` aspect cannot re-crop it.
            lo = min(np.nanmin(xy), 0.0)
            hi = max(np.nanmax(xy), 0.0)
            mid, half = 0.5 * (lo + hi), 0.5 * max(hi - lo, 1e-3) * 1.15
            out["path"] = (mid - half, mid + half)
            z = r.ref_pos_err[:, :, 2]
            out["z"] = pad(float(np.nanmin(z)), float(np.nanmax(z)))

        if r.ref_pos_err is not None and np.isfinite(r.ref_pos_err).any():
            # Per-axis limits, each over its OWN axis: sharing one scale across x/y/z lets
            # the largest axis flatten the other two into a straight line.
            for ax_i, key in enumerate(("ex", "ey", "ez")):
                v = r.ref_pos_err[:, :, ax_i]
                out[key] = pad(float(np.nanmin(v)), float(np.nanmax(v)))
        if r.ref_rpy_err is not None and np.isfinite(r.ref_rpy_err).any():
            out["rpy"] = pad(float(np.nanmin(r.ref_rpy_err)), float(np.nanmax(r.ref_rpy_err)))

        if r.ref_joint is not None and np.isfinite(r.ref_joint).any():
            stack = [r.ref_joint]
            nj = r.ref_joint.shape[2]
            for arr in (r.joint_pos_isaac, r.joint_target_isaac):
                if arr is not None and arr.size:
                    stack.append(arr[:, :nj])
            lo = min(float(np.nanmin(a)) for a in stack)
            hi = max(float(np.nanmax(a)) for a in stack)
            out["jnow"] = pad(lo, hi)

        # Feet: one scale shared by BOTH feet and by reference and measured alike, so the
        # left and right panels can be read against each other and a swing height can be
        # compared to its own reference rather than to a rescaled axis.
        probe = r.feet_at(int(np.argmax(r.armed)) if r.armed.any() else 0)
        if probe is not None:
            every = [r.feet_at(i) for i in range(0, len(r.t), max(1, len(r.t) // 200))]
            vals = np.concatenate([np.concatenate([a.ravel(), b.ravel()])
                                   for a, b in every if a is not None])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out["feet"] = pad(float(vals.min()), float(vals.max()))

        # One colour scale per camera, over every frame: stepping should show the scene
        # changing, not the colormap re-normalising to it.
        out["depth"] = []
        for frames in r.depth_frames:
            if not frames:
                out["depth"].append(None)
                continue
            allpx = np.concatenate([f[np.isfinite(f)].ravel() for f in frames]) \
                if frames else np.zeros(1)
            if allpx.size == 0:
                out["depth"].append(None)
            else:
                lo, hi = np.percentile(allpx, [1, 99])
                out["depth"].append((float(lo), float(max(hi, lo + 1e-3))))
        return out

    def _init_static(self) -> None:
        """Draw everything that does not change with the cursor."""
        r = self.run
        for i, (name, unit) in enumerate((("v_x", "m/s"), ("v_y", "m/s"), ("w_z", "rad/s"))):
            achieved = (r.base_lin_b[:, 0], r.base_lin_b[:, 1], r.base_ang_b[:, 2])[i]
            self.ax_vel.plot(r.t, r.command[:, i], lw=1.0, label=f"cmd {name}")
            self.ax_vel.plot(r.t, achieved, lw=0.7, ls="--", label=f"act {name}")
        self.ax_vel.set_xlabel("time [s]"); self.ax_vel.set_ylabel("velocity")
        self.ax_vel.legend(fontsize="x-small", ncol=6); self.ax_vel.grid(alpha=0.3)
        self._cursor = self.ax_vel.axvline(r.t[self.i], color="k", lw=1.2)

    def _on_key(self, event) -> None:
        """Move the cursor. Arrow keys step; home/end jump."""
        step = {"left": -1, "right": 1, "down": -10, "up": 10,
                "pagedown": -100, "pageup": 100}.get(event.key)
        if step is not None:
            self.i = int(np.clip(self.i + step, 0, len(self.run.t) - 1))
        elif event.key == "home":
            self.i = 0
        elif event.key == "end":
            self.i = len(self.run.t) - 1
        else:
            return
        self._draw()

    def _draw(self) -> None:
        """Redraw everything that depends on the cursor."""
        r, i = self.run, self.i
        for ax in self._axes_dyn:
            ax.clear()

        for cam, (ax, name) in enumerate(((self.ax_near, "near"), (self.ax_far, "far"))):
            img = r.depth_at(i, cam)
            if img is None:
                ax.text(0.5, 0.5, f"no {name} camera", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            finite = img[np.isfinite(img)]
            # ONE colour scale for the whole run, so stepping shows the scene change rather
            # than the colormap chasing it. The title carries this frame's own range.
            scale = self._limits["depth"][cam]
            lo, hi = scale if scale else (0.0, 1.0)
            ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi)
            bad = 100.0 * (1.0 - finite.size / img.size) if img.size else 100.0
            here = (f"{finite.min():.2f}-{finite.max():.2f}" if finite.size else "all invalid")
            ax.set_title(f"depth {name}  {img.shape[0]}x{img.shape[1]}   "
                         f"scale {lo:.2f}-{hi:.2f} m\n"
                         f"this frame {here} m, {bad:.0f}% non-finite", fontsize="small")
            ax.set_xticks([]); ax.set_yticks([])

        if r.ref_pos_err is not None:
            p = r.ref_pos_err[i]                       # [K, 3] in the ANCHOR's yaw frame
            self.ax_path.plot(p[:, 0], p[:, 1], "-o", ms=3, label="reference")
            # The origin is the WINDOW's anchor, which under `plan` is the root at the last
            # re-plan -- NOT the robot. Labelling it "robot" made the robot's own commanded
            # displacement (0.42 m/s of it, resetting every 12 ticks) read as drift.
            self.ax_path.plot(0, 0, "k+", ms=12, mew=2, label="anchor")
            robot = r.robot_in_anchor_frame(i)
            if robot is not None:
                self.ax_path.plot(robot[0], robot[1], "r*", ms=12, label="robot")
            self.ax_path.set_title(
                "reference root path (anchor yaw frame)\n"
                + ("robot drawn from odometry -- DIAGNOSTIC" if robot is not None
                   else "no anchor latched (clip playback)"), fontsize="small")
            self.ax_path.set_xlabel("x [m]"); self.ax_path.set_ylabel("y [m]")
            self.ax_path.legend(fontsize="x-small")
            if "path" in self._limits:
                self.ax_path.set_xlim(*self._limits["path"])
                self.ax_path.set_ylim(*self._limits["path"])
            # After the limits, and 'box' so it cannot widen them back out.
            self.ax_path.set_aspect("equal", adjustable="box")

            self.ax_z.plot(r.offsets, p[:, 2], "-o", ms=3)
            self.ax_z.axhline(0.0, color="r", lw=0.8)
            self.ax_z.set_title("reference height vs offset", fontsize="small")
            self.ax_z.set_xlabel("offset [steps]"); self.ax_z.set_ylabel("z [m]")
            if "z" in self._limits:
                self.ax_z.set_ylim(*self._limits["z"])

            # ---- per-axis, which is what a single x-y path cannot show ----
            for ax, comp, name in ((self.ax_ex, 0, "x"), (self.ax_ey, 1, "y"),
                                   (self.ax_ez, 2, "z")):
                ax.plot(r.offsets, p[:, comp], "-o", ms=3)
                ax.axhline(0.0, color="r", lw=0.8)
                if robot is not None:
                    ax.axhline(robot[comp], color="tab:green", lw=0.9, ls=":",
                               label="robot")
                    ax.legend(fontsize="x-small")
                ax.set_title(f"base {name}: reference - anchor", fontsize="small")
                ax.set_xlabel("offset [steps]"); ax.set_ylabel(f"{name} [m]")
                key = ("ex", "ey", "ez")[comp]
                if key in self._limits:
                    ax.set_ylim(*self._limits[key])

        if r.ref_rpy_err is not None:
            rpy = r.ref_rpy_err[i]
            for c, name in enumerate(("roll", "pitch", "yaw")):
                self.ax_rpy.plot(r.offsets, np.degrees(rpy[:, c]), "-o", ms=3, label=name)
            self.ax_rpy.axhline(0.0, color="r", lw=0.8)
            self.ax_rpy.set_title("base orientation: reference - anchor", fontsize="small")
            self.ax_rpy.set_xlabel("offset [steps]"); self.ax_rpy.set_ylabel("[deg]")
            self.ax_rpy.legend(fontsize="x-small")
            if "rpy" in self._limits:
                lo, hi = self._limits["rpy"]
                self.ax_rpy.set_ylim(np.degrees(lo), np.degrees(hi))

        feet = r.feet_at(i)
        for f, (ax, label) in enumerate(((self.ax_fl, "left foot"),
                                         (self.ax_fr, "right foot"))):
            if feet is None:
                ax.text(0.5, 0.5, "no FK\n(pinocchio / URDF)", ha="center", va="center",
                        transform=ax.transAxes, fontsize="small")
                continue
            ref_f, act_f = feet
            for c, (name, colour) in enumerate((("x", "tab:blue"), ("y", "tab:orange"),
                                                ("z", "tab:green"))):
                ax.plot(r.offsets, ref_f[:, f, c], "-", color=colour, lw=1.2,
                        label=f"ref {name}")
                ax.plot(0, act_f[f, c], "o", color=colour, ms=7, mfc="none", mew=1.8)
            ax.set_title(f"{label} (base frame)\nline = reference, ring = measured",
                         fontsize="small")
            ax.set_xlabel("offset [steps]"); ax.set_ylabel("[m]")
            ax.legend(fontsize="x-small", ncol=3)
            if "feet" in self._limits:
                ax.set_ylim(*self._limits["feet"])

        if r.ref_joint is not None:
            nj = r.ref_joint.shape[2]
            idx = np.arange(nj)
            self.ax_jnow.plot(idx, r.ref_joint[i, 0, :], "o-", ms=3, label="reference @0")
            if r.joint_pos_isaac is not None:
                self.ax_jnow.plot(idx, r.joint_pos_isaac[i, :nj], "s--", ms=3, label="measured")
            if r.joint_target_isaac is not None:
                self.ax_jnow.plot(idx, r.joint_target_isaac[i, :nj], "^:", ms=3,
                                  label="PD target")
            self.ax_jnow.set_title("joints at this tick (POLICY order)", fontsize="small")
            self.ax_jnow.set_xlabel("joint index"); self.ax_jnow.set_ylabel("[rad]")
            self.ax_jnow.legend(fontsize="x-small")
            if "jnow" in self._limits:
                self.ax_jnow.set_ylim(*self._limits["jnow"])

        for ax in self._axes_dyn[2:]:
            ax.grid(alpha=0.3)
        self._cursor.set_xdata([r.t[i], r.t[i]])

        m = r.ticks[i]
        self.fig.suptitle(
            f"tick {i}/{len(r.t) - 1}   t={r.t[i]:.2f}s   "
            f"{'ARMED' if m.armed else 'disarmed'}   "
            f"{'clip step ' + str(m.clip_step) if m.clip_mode else 'fm cursor ' + str(m.fm_cursor)}"
            f"   tick {m.latency_ms:.1f} ms"
            f" (trk {getattr(m, 'trk_infer_ms', float('nan')):.1f},"
            f" fm {getattr(m, 'fm_infer_ms', float('nan')):.1f})"
            f"{'   LATE' if getattr(m, 'fm_late', False) else ''}\n"
            "LEFT/RIGHT = 1 tick,  UP/DOWN = 10,  PAGEUP/PAGEDOWN = 100,  HOME/END = ends",
            fontsize="medium")
        self.fig.canvas.draw_idle()


def main() -> int:
    """Read a bag and either write the figures or open the stepper."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", nargs="?", default=None,
                   help=f"Bag directory or .mcap (default: newest in {DEFAULT_BAG_ROOT}).")
    p.add_argument("--save_dir", default=None,
                   help="Where figures go (default: <bag>/plots, so runs never clobber "
                        "each other).")
    p.add_argument("--interactive", action="store_true",
                   help="Open the frame stepper instead of writing figures.")
    p.add_argument("--show", action="store_true",
                   help="Display the static figures instead of writing them.")
    p.add_argument("--figures", default=None,
                   help=f"Comma-separated subset of {','.join(FIGURES)}.")
    p.add_argument("--start", type=float, default=None, help="Trim to after this second.")
    p.add_argument("--end", type=float, default=None, help="Trim to before this second.")
    args = p.parse_args()

    bag = args.bag or find_latest_bag(DEFAULT_BAG_ROOT)
    if not bag:
        sys.exit(f"[ERROR] No bags under {DEFAULT_BAG_ROOT}. Record one with "
                 "`scripts/run_sim_clip.sh` or `scripts/run_sim_low.sh --bag`.")
    print(f"[INFO] reading {bag}")
    run = Run(read_bag(bag, TOPICS), start=args.start, end=args.end)
    print(f"[INFO] {run.summary()}")

    import matplotlib
    interactive = args.interactive or args.show
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.interactive:
        backend = matplotlib.get_backend()
        if backend.lower() in ("agg", "pdf", "ps", "svg", "template"):
            sys.exit(
                f"[ERROR] matplotlib is using the non-interactive '{backend}' backend, so a "
                "window would open with dead arrow keys (or not open at all). Install a GUI "
                "backend (`pip install PyQt5`), or set MPLBACKEND=TkAgg, and make sure "
                "DISPLAY is set if you are over ssh."
            )
        stepper = Stepper(run, plt)          # keep the reference; see Stepper.__init__
        print(f"[INFO] stepper on the '{backend}' backend: LEFT/RIGHT 1 tick, UP/DOWN 10, "
              "PAGEUP/PAGEDOWN 100, HOME/END ends")
        plt.show()
        return 0 if stepper else 0

    wanted = args.figures.split(",") if args.figures else list(FIGURES)
    save_dir = args.save_dir or os.path.join(
        bag if os.path.isdir(bag) else os.path.dirname(bag), "plots")
    if not args.show:
        os.makedirs(save_dir, exist_ok=True)
    for name in wanted:
        if name not in FIGURES:
            sys.exit(f"[ERROR] unknown figure {name!r}; have {', '.join(FIGURES)}")
        fig, axes = FIGURES[name](run, plt)
        for ax in np.atleast_1d(axes):
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize="x-small", ncol=3)
        fig.suptitle(f"{os.path.basename(bag)} — {name}", y=0.995)
        fig.tight_layout()
        if not args.show:
            out = os.path.join(save_dir, f"{name}.png")
            fig.savefig(out, dpi=120)
            print(f"[INFO] wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
