#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""The paper's velocity-tracking figure, from one ``vel_tracking_sweep.py`` output directory.

A grid: one COLUMN per terrain, and per column a pair of stacked panels --

* the root's x-z path, over a filled outline of the terrain the robot actually walked on, so
  the height the pair gives up to the feature is visible against the feature itself; and
* the commanded velocity against the achieved one, so the same excursion is readable as a
  tracking result.

The claim the figure is making is the contrast BETWEEN columns: flat tracks the command
closely, and the terrain columns still track it while the top panel shows the root doing what
the terrain requires.  So every column shares one velocity scale and one time (or downrange)
scale, because a per-column autoscale would flatten exactly that comparison.

**The band is across FLOW-NOISE SEEDS and nothing else** -- the sweep holds the initial
condition, the command, the terrain and the DR, so the spread is the generator's own sampling
variance. It is a 10-90 percentile band, not a standard error: the outcome on terrain is
bimodal (a trial either traverses or dies on the feature), so a symmetric interval around the
mean would describe a distribution that is not there.

**Failures are drawn, not dropped.** Every trial's path is drawn faintly, a failed one in red
and marked with an x where it ended, and the caption carries the rate. The median and the band
are computed over SURVIVING trials only, because a fallen robot's velocity after the fall is
not a tracking measurement -- so a column with a high fall rate has a median drawn from fewer
trials, which the caption states rather than the line implying.

IsaacLab-free and MuJoCo-free: it reads ``traces.npz`` and ``summary.json`` only, so it runs
anywhere the sweep's output can be copied to.

    python plot_vel_tracking.py ../../logs/vel_tracking/v1
    python plot_vel_tracking.py ../../logs/vel_tracking/v1 --channels vx,wz --vel_axis x
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import warnings
from collections.abc import Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# `plot_style` lives in scripts/rsl_rl because four plotters outside the eval use it; reached
# by path because `transfer` is not on the same install as `scripts`.
for _d in (os.path.join(_REPO, "scripts", "rsl_rl"), _HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import plot_style  # noqa: E402

#: Human labels for the terrain columns, in the order they are drawn.
KIND_LABEL: dict[str, str] = {
    "flat": "flat", "box": "box", "stairs_up": "stairs up", "stairs_down": "stairs down",
}

#: Axis labels for the velocity channels, and which trace column pairs make each one.
CHANNELS: dict[str, tuple[str, str, str]] = {
    "vx": ("vx", "cmd_vx", r"$v_x$ [m/s]"),
    "vy": ("vy", "cmd_vy", r"$v_y$ [m/s]"),
    "wz": ("wz", "cmd_wz", r"$\omega_z$ [rad/s]"),
}


def parse_run_spec(spec: str) -> tuple[str, tuple[str, ...]]:
    """Split ``path`` or ``path:kind[,kind...]`` into a directory and its terrain selection.

    Per-run selection exists because a run holds ONE geometry per terrain -- ``--box_depth``
    is a per-run flag -- so "the parent on every terrain, with the IN-DISTRIBUTION box" is
    necessarily assembled from two runs: flat and the stairs from the sweep that had them, and
    the box from the sweep that used the trained depth. A single ``--terrain`` list cannot say
    that, because it applies to every run at once and would pull in the other box too.

    The suffix is only treated as a selection when every part of it names a real terrain, so a
    directory containing a colon still resolves as a path.

    Args:
        spec: The command-line token.

    Returns:
        ``(run_dir, kinds)``; ``kinds`` empty means "every terrain this run holds".
    """
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        parts = tuple(k.strip() for k in tail.split(",") if k.strip())
        if head and parts and all(k in vt_terrain_kinds() for k in parts):
            return head, parts
    return spec, ()


def vt_terrain_kinds() -> frozenset[str]:
    """The terrain names a run spec may select, from the sweep module itself.

    Imported lazily and by name rather than restated here: a terrain added to the sweep must
    become selectable without editing the plotter, and a typo in a spec must be refused rather
    than silently swallowed as part of a path.
    """
    global _KINDS
    if _KINDS is None:
        sys.path.insert(0, _HERE)
        import vel_tracking_sweep as _vt  # noqa: PLC0415
        _KINDS = frozenset(_vt.TERRAIN_KINDS)
    return _KINDS


_KINDS: frozenset[str] | None = None


def load_run(run_dir: str) -> tuple[dict, dict]:
    """Read one sweep's ``summary.json`` and ``traces.npz``.

    Args:
        run_dir: A ``vel_tracking_sweep.py`` output directory.

    Returns:
        ``(summary, traces)``, the second a dict of the ``.npz``'s arrays.

    Raises:
        SystemExit: When either file is absent, naming the directory rather than the file --
            the usual cause is pointing at a parent of the run.
    """
    s_path = os.path.join(run_dir, "summary.json")
    t_path = os.path.join(run_dir, "traces.npz")
    for p in (s_path, t_path):
        if not os.path.isfile(p):
            raise SystemExit(f"{run_dir}: no {os.path.basename(p)} -- is this a "
                             "vel_tracking_sweep output directory?")
    with open(s_path) as fh:
        summary = json.load(fh)
    with np.load(t_path) as z:
        traces = {k: z[k] for k in z.files}
    return summary, traces


class Column:
    """One panel column: one terrain of one run, with everything needed to draw it.

    A column rather than a terrain is the unit because the comparisons this harness exists to
    make are BETWEEN RUNS on the same terrain -- a box at two depths, a parent generator
    against a fine-tuned one -- and a run holds one geometry per terrain (``--box_depth`` and
    ``--stair_steps`` are per-run flags). Drawing several runs' columns in one figure is what
    puts those side by side on a shared scale; a reader comparing two separately-rendered
    figures is comparing two autoscales.
    """

    def __init__(self, label: str, kind: str, summary: dict, traces: dict) -> None:  # noqa: D401
        """Bind one (run, terrain) pair.

        Args:
            label: Short run name, shown under the terrain in the column title when more than
                one run is drawn.  Empty for a single-run figure, where it would be noise.
            kind: The terrain key within that run.
            summary: The run's ``summary.json``.
            traces: The run's ``traces.npz`` arrays.
        """
        self.label, self.kind = label, kind
        self.summary, self.traces = summary, traces
        self.info = summary["terrains"][kind]
        self.slices = episode_slices(traces)
        self.code = summary["meta"]["terrain_codes"][kind]
        self.success_code = summary["meta"]["verdict_codes"]["success"]

    def pick(self, speed: float) -> np.ndarray:
        """Episode indices for this column at ``speed``.

        Keyed on ``meta.terrain_codes``, never on the position of the terrain in the run's own
        list: ``ep_terrain`` stores the GLOBAL code, and the two coincide only when a run
        happens to request every terrain in code order.  A four-terrain run does; a
        stairs-only run does not, and the mismatch returns an empty selection rather than an
        error.
        """
        t, s = self.traces["ep_terrain"], self.traces["ep_speed"]
        return np.flatnonzero((t == self.code) & np.isclose(s, speed))

    def survived(self, idx: np.ndarray) -> np.ndarray:
        """Of ``idx``, those that traversed."""
        return idx[self.traces["ep_verdict"][idx] == self.success_code]

    def traversed(self, speeds: Sequence[float]) -> tuple[int, int]:
        """How many of this column's DRAWN episodes traversed, pooled over ``speeds``.

        Pooled rather than per speed because the heading names the panel and the panel holds
        every speed; the per-speed split stays in the caption, which is where a reader who
        needs it looks.  Counted over the speeds actually drawn, not over the run's own list,
        so a ``--speed``-filtered figure reports the figure rather than the run behind it.

        Args:
            speeds: The commanded speeds this figure draws.

        Returns:
            ``(traversed, total)``.  ``(0, 0)`` when the column draws nothing.
        """
        ok = total = 0
        for speed in speeds:
            pick = self.pick(speed)
            if pick.size == 0:
                continue
            ok += int(self.survived(pick).size)
            total += int(pick.size)
        return ok, total

    def title(self, multi: bool, speeds: Sequence[float] = ()) -> str:
        """Column heading: the terrain, the run when several are drawn, and what traversed.

        The traversed share is shown as a PERCENTAGE; the raw counts behind it stay in the
        caption, per speed, which is where the denominator belongs.  One decimal because at
        the usual 50 trials x 3 speeds one episode is 0.67 %, so the digit is resolution
        rather than noise -- it is not a claim about precision.

        Args:
            multi: Whether the figure draws more than one run.
            speeds: The speeds drawn, which the traversed share pools over.  Empty omits the
                share entirely, which is what a caller that has not resolved its speeds yet
                gets rather than a wrong denominator.

        Returns:
            The heading, one or two lines.
        """
        base = KIND_LABEL.get(self.kind, self.kind)
        ok, total = self.traversed(speeds) if speeds else (0, 0)
        if total:
            import matplotlib as mpl  # noqa: PLC0415  -- lazy, as plot_run imports pyplot
            # `%` opens a comment in LaTeX, so under `text.usetex` it must be escaped -- and
            # under mathtext it must NOT be, or the backslash renders literally.  Read off the
            # live rcParam rather than off `--no_tex`, because apply_paper_style also falls
            # back to mathtext on its own when LaTeX is simply not installed.
            mark = r"\%" if mpl.rcParams["text.usetex"] else "%"
            # Parenthesised on the TERRAIN line, which costs no extra line in the figure --
            # vertical space is the scarce thing in a paper.  It goes here rather than on the
            # run-name line because `base` is a short fixed word from KIND_LABEL, so the
            # heading's width stays bounded; appending it to `label` instead made a three-run
            # heading as wide as the caller's `--label` and it collided with its neighbour.
            base = f"{base} ({100.0 * ok / total:.1f}{mark})"
        return f"{base}\n{self.label}" if (multi and self.label) else base


def build_columns(runs: list, only_kinds: tuple[str, ...] = ()) -> list[Column]:
    """Every (run, terrain) pair, TERRAIN-major so the runs sit adjacent per terrain.

    Terrain-major rather than run-major because the eye compares neighbours: for a box at two
    depths the two box columns must touch, and grouping by run would put a flat and two stair
    columns between them.

    Args:
        runs: ``(label, summary, traces)`` per run.
        only_kinds: Terrains to keep, in this order; empty keeps each run's own list.

    Returns:
        The columns to draw.

    Raises:
        SystemExit: When a requested terrain is in no run, naming what is available -- an
            unmatched name would otherwise silently drop a column and read as a missing arm.
    """
    def offers(run) -> list[str]:
        """The terrains a run entry contributes: its own list, narrowed by its spec."""
        _label, s, _t = run[0], run[1], run[2]
        pick = run[3] if len(run) > 3 else ()
        return [k for k in s["meta"]["terrains"] if not pick or k in pick]

    have = {k for run in runs for k in offers(run)}
    kinds = list(only_kinds) if only_kinds else [k for k in KIND_LABEL if k in have]
    missing = [k for k in kinds if k not in have]
    if missing:
        raise SystemExit(f"no run offers terrain(s) {missing}; available: {sorted(have)}")
    return [Column(run[0], k, run[1], run[2])
            for k in kinds
            for run in runs if k in offers(run)]


def episode_slices(traces: dict) -> list[slice]:
    """One slice into the flat per-step arrays per episode, in table order."""
    return [slice(int(s), int(s) + int(n))
            for s, n in zip(traces["ep_start"], traces["ep_len"])]


def resample(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate one episode's series onto a shared grid, NaN outside its own extent.

    NaN rather than an edge hold, because the episodes end at different places: holding the
    last value would draw a fallen robot as standing still at its final velocity for the rest
    of the panel, and that value would then enter the median as if it were a measurement.

    Args:
        xs: The episode's own abscissa (time or downrange), non-decreasing.
        ys: The series to resample, same length.
        grid: The shared abscissa.

    Returns:
        ``ys`` on ``grid``, with NaN where ``grid`` is outside ``xs``.
    """
    if xs.size < 2:
        return np.full(grid.shape, np.nan)
    out = np.interp(grid, xs, ys, left=np.nan, right=np.nan)
    return np.where((grid >= xs[0]) & (grid <= xs[-1]), out, np.nan)


def band(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and 10-90 percentile band over resampled curves, plus the per-point count.

    Args:
        curves: One resampled curve per episode, all on the same grid.

    Returns:
        ``(median, p10, p90, n)``; every entry NaN where no episode covers that grid point.
    """
    if not curves:
        empty = np.zeros(0)
        return empty, empty, empty, empty
    a = np.vstack(curves)
    n = np.sum(np.isfinite(a), axis=0)
    # An all-NaN column is the NORMAL case past the shortest episode's end, and the reducers
    # warn on it rather than raising -- so the warning is filtered rather than the column
    # dropped, and the result is blanked below. `errstate` does not cover this: it is a
    # RuntimeWarning from the reducer, not a floating-point error.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        med = np.nanmedian(np.where(n > 0, a, np.nan), axis=0)
        p10 = np.nanpercentile(np.where(n > 0, a, np.nan), 10, axis=0)
        p90 = np.nanpercentile(np.where(n > 0, a, np.nan), 90, axis=0)
    blank = n == 0
    for arr in (med, p10, p90):
        arr[blank] = np.nan
    return med, p10, p90, n


def trial_alpha(n: int) -> float:
    """Per-trial curve alpha for ``n`` overlaid trials, so total ink stays roughly constant.

    A fixed alpha cannot serve both ends of this: 0.18 is a readable ghost for two trials and a
    solid block for fifty, and the block hides the median and band drawn on top of it -- the
    figure then shows the envelope of the trials rather than their centre, which is the
    opposite of what the panel is for.
    """
    return float(min(0.25, max(0.03, 3.0 / max(1, n))))


#: Alpha for the failure-end markers.  A CONSTANT, deliberately: it used to scale as
#: ``12 / n_failures`` per (column, speed), on the constant-ink reasoning behind
#: :func:`trial_alpha`, and the side effect was that transparency silently encoded how badly a
#: speed did -- the cell with the most failures drew the FAINTEST crosses, and two panels of
#: one figure ended up on different scales.  Density still reads, because the crosses sit at
#: distinct end positions and pile up darker on their own; it now reads the same way in every
#: panel of every figure instead of being rescaled per cell.
FAIL_MARKER_ALPHA: float = 0.75

#: Panel text sizes for a multi-column grid.  The shared paper rcParams are sized for a
#: single full-width plot (22 pt titles, 20 pt labels); four columns of it at journal width
#: give each panel under 2 inches, where those sizes collide -- the first draft of this figure
#: rendered "stairs upstairs down".  Overriding here rather than in ``plot_style`` keeps the
#: style one thing: this is what a 4-column grid of it costs, not a second style.
GRID_FONTS: dict[str, float] = {"title": 15, "label": 13, "tick": 11, "legend": 10}


#: Fewest contributing episodes a median/band point is allowed to be drawn from.
#:
#: The abscissa of a time panel runs to the LONGEST surviving episode, but the survivors do
#: not all last that long -- the tail belongs to the slowest few, so the last seconds of a
#: median are an average over a handful of trials drawn at the same weight as the first
#: seconds' fifty. Below this many the curve is cut rather than thinned: a line whose support
#: silently falls from 50 to 1 invites reading the noisiest part of it as a trend.
MIN_TRIALS: int = 10


def trim_support(med, p10, p90, n, min_n: int):
    """Blank median/band points supported by fewer than ``min_n`` episodes.

    Args:
        med: Median curve.
        p10: Lower band edge.
        p90: Upper band edge.
        n: Per-point count of contributing episodes, from :func:`band`.
        min_n: Minimum support.

    Returns:
        ``(med, p10, p90, last)`` with the unsupported points set to NaN, and ``last`` the
        index of the final supported point (``-1`` when none is).
    """
    if med.size == 0:
        return med, p10, p90, -1
    ok = n >= min_n
    for arr in (med, p10, p90):
        arr[~ok] = np.nan
    idx = np.flatnonzero(ok)
    return med, p10, p90, (int(idx[-1]) if idx.size else -1)


def draw_profile(ax, prof_x, prof_z) -> None:
    """Fill the terrain outline under the trajectory panel.

    Filled from far below the lowest surface, because the panel's y limits are set from the
    trajectory DATA afterwards: filling to the axis bottom instead would make the fill itself
    part of the autoscale, and one 0.75 m staircase would then set the shared height scale for
    every column and squash the flat control into a line.

    Args:
        ax: The trajectory axes.
        prof_x: Downrange samples of the walkable surface (m).
        prof_z: Surface height at each sample (m).
    """
    x = np.asarray(prof_x, dtype=float)
    z = np.asarray(prof_z, dtype=float)
    ax.fill_between(x, float(z.min()) - 50.0, z, color="0.86", linewidth=0.0, zorder=0)
    ax.plot(x, z, color="0.45", linewidth=1.5, zorder=1)


def resolve_block_shape(ncols: int, mode: str,
                        shape: tuple[int, int] | None) -> tuple[int, int]:
    """``(block_rows, block_cols)`` for ``ncols`` terrain blocks.

    Args:
        ncols: Number of columns (terrain/run pairs).
        mode: ``"rows"`` or ``"grid"``.
        shape: An explicit arrangement, or ``None`` for near-square.

    Returns:
        The block arrangement; ``rows`` is always one row of ``ncols`` blocks.

    Raises:
        SystemExit: When an explicit shape cannot hold every block -- dropping a terrain
            silently would read as a run that produced nothing.
    """
    if mode == "rows":
        return 1, ncols
    if shape is None:
        rows = int(math.ceil(math.sqrt(ncols)))
        return rows, int(math.ceil(ncols / rows))
    if shape[0] * shape[1] < ncols:
        raise SystemExit(f"--grid_shape {shape[0]}x{shape[1]} holds {shape[0] * shape[1]} "
                         f"blocks but there are {ncols} columns to draw")
    return shape


def layout_figsize(nrows: int, block_rows: int, block_cols: int, ncols: int, mode: str,
                   multi: bool) -> tuple[float, float]:
    """Figure size, computed before the figure exists.

    ``grid`` is sized per BLOCK rather than per column, so four terrains stay near journal
    width instead of stretching into four full-height columns.
    """
    if mode == "rows":
        return (max(7.0, 1.9 * ncols), 2.15 * nrows + 0.45 + (0.25 if multi else 0.0))
    return (max(7.0, 3.7 * block_cols),
            (1.72 * nrows + 0.75) * block_rows + (0.2 if multi else 0.0))


class Layout:
    """Where each column's stack of panels sits, and which of them carry labels.

    Two arrangements, one drawing code. ``rows`` puts every terrain in a column and every
    channel in a row -- best for scanning ONE quantity across terrains, since a row is then a
    single shared y scale read left to right. ``grid`` gives each terrain its own block of
    stacked panels laid out near-square, which reads as four small self-contained figures and
    keeps a terrain's height and velocity physically together; blocks fill DOWN each grid
    column first, so with four terrains the order is top-left, bottom-left, top-right,
    bottom-right.

    Either way the axes are exposed as ``axes[row][column]`` so nothing in the drawing code
    branches on the arrangement, and every scale is still shared across all columns -- the
    comparison is between terrains in both.
    """

    def __init__(self, fig, nrows: int, ncols: int, mode: str,
                 shape: tuple[int, int]) -> None:
        """Create the axes.

        Args:
            fig: The figure to populate.
            nrows: Panels per column (1 trajectory + one per channel).
            ncols: Number of columns (terrain/run pairs).
            mode: ``"rows"`` or ``"grid"``.
            shape: ``(block_rows, block_cols)``, from :func:`resolve_block_shape`.
        """
        self.mode, self.nrows, self.ncols = mode, nrows, ncols
        self.block_rows, self.block_cols = shape
        if mode == "rows":
            gs = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.10)
            self.axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)]
                         for r in range(nrows)]
            self._pos = [(0, c) for c in range(ncols)]
            return
        br, bc = shape
        outer = fig.add_gridspec(br, bc, hspace=0.30, wspace=0.18)
        cells: list[list] = [[None] * ncols for _ in range(nrows)]
        self._pos = []
        for ci in range(ncols):
            # Column-major: fill down a grid column before moving right.
            gr, gc = ci % br, ci // br
            self._pos.append((gr, gc))
            inner = outer[gr, gc].subgridspec(nrows, 1, hspace=0.62)
            for r in range(nrows):
                cells[r][ci] = fig.add_subplot(inner[r, 0])
        self.axes = cells

    def wants_ylabel(self, ci: int) -> bool:
        """Only the leftmost block column carries y labels; the rest share its scale."""
        return self._pos[ci][1] == 0


def plot_run(runs: list[tuple[str, dict, dict]], *, channels: tuple[str, ...],
             vel_axis: str, show_all: bool, grid_points: int,
             only_speeds: tuple[float, ...] = (), only_kinds: tuple[str, ...] = (),
             min_trials: int = MIN_TRIALS, layout: str = "rows",
             grid_shape: tuple[int, int] | None = None):
    """Build the figure: one column per (run, terrain), one row per channel plus the path.

    **The rows do not always share an abscissa, and that is why nothing here uses
    ``sharex="col"``.** Under ``vel_axis="time"`` the trajectory row is downrange metres and
    the velocity rows are seconds; a column-wide share silently puts one row on the other's
    scale, and because a 20 m lane at ~0.8 m/s takes ~25 s the two ranges are numerically
    similar enough that the result looks right. The axes are linked per ROW GROUP instead, and
    the trajectory row's ticks move to the TOP when it is on its own scale.

    **Every scale is shared across columns**, which is the whole point of drawing several runs
    in one figure: the comparison being made is between columns, and a per-column autoscale
    would flatten exactly that.

    Args:
        runs: ``(label, summary, traces)`` per run, in the order their columns appear.  A
            single run passes one entry and its label is not drawn.
        channels: Velocity channels to give a row each, keys of :data:`CHANNELS`.
        vel_axis: ``"time"`` or ``"x"`` for the velocity rows' abscissa.  ``"x"`` shares the
            trajectory row's axis, which lines a velocity excursion up with the step that
            caused it; ``"time"`` keeps the opening command STEP visible, which a downrange
            axis collapses into a single x because the robot is standing through the hold.
        show_all: Draw every trial's own faint curve behind the median.
        grid_points: Samples on the shared abscissa.
        only_speeds: Draw only these speeds; empty draws every speed the runs hold.
        only_kinds: Draw only these terrains; empty draws every terrain each run holds.
        min_trials: Cut every curve where fewer than this many episodes still contribute, and
            end the axis at the last supported point.  See :data:`MIN_TRIALS`.
        layout: ``"rows"`` (terrains across, channels down) or ``"grid"`` (a block of stacked
            panels per terrain, near-square).  See :class:`Layout`.
        grid_shape: ``(block_rows, block_cols)`` for ``"grid"``; ``None`` picks near-square.

    Returns:
        ``(fig, caption)``.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415  -- after apply_paper_style

    cols = build_columns(runs, only_kinds)
    multi = len(runs) > 1
    speeds = list(only_speeds if only_speeds else runs[0][1]["meta"]["speeds"])
    shared_x = vel_axis == "x"
    nrows = 1 + len(channels)

    shape = resolve_block_shape(len(cols), layout, grid_shape)
    fig = plt.figure(figsize=layout_figsize(nrows, *shape, len(cols), layout, multi))
    lay = Layout(fig, nrows, len(cols), layout, shape)
    axes = lay.axes
    for ci in range(len(cols)):
        for row in range(2, nrows):
            axes[row][ci].sharex(axes[1][ci])
        if shared_x:
            axes[0][ci].sharex(axes[1][ci])

    colours = plot_style.COLOR_CYCLE
    fail_colour = plot_style.COLORS["red"]
    counts: list[str] = []
    z_lo, z_hi, x_hi = [], [], []
    traj_hi, vel_hi = [], []

    for ci, c in enumerate(cols):
        tr, sl = c.traces, c.slices
        ax_traj = axes[0][ci]
        draw_profile(ax_traj, c.info["profile_x"], c.info["profile_z"])
        z_lo.append(float(min(c.info["profile_z"])))
        z_hi.append(float(max(c.info["profile_z"])))
        x_max = float(c.info["goal_x"]) + 1.0
        x_hi.append(x_max)
        for row in range(nrows):
            if row == 0 or shared_x:
                axes[row][ci].axvspan(c.info["x_enter"],
                                      max(c.info["x_exit"], c.info["x_enter"]),
                                      color="0.72", alpha=0.6, linewidth=0.8, zorder=1)

        for i_s, speed in enumerate(speeds):
            colour = colours[i_s % len(colours)]
            pick = c.pick(speed)
            if pick.size == 0:
                continue
            ok = c.survived(pick)
            bad = np.setdiff1d(pick, ok, assume_unique=False)
            tag = f"{c.label} " if (multi and c.label) else ""
            counts.append(f"{tag}{KIND_LABEL.get(c.kind, c.kind)} @ {speed:.2f} m/s "
                          f"{ok.size}/{pick.size}")
            faint = trial_alpha(pick.size)

            gx = np.linspace(0.0, x_max, grid_points)
            if show_all:
                for i in pick:
                    col = colour if i in set(ok.tolist()) else fail_colour
                    ax_traj.plot(tr["x"][sl[i]], tr["z"][sl[i]], color=col,
                                 linewidth=0.5, alpha=faint, zorder=2)
            med, p10, p90, nsup = band([resample(tr["x"][sl[i]], tr["z"][sl[i]], gx)
                                        for i in ok])
            med, p10, p90, last = trim_support(med, p10, p90, nsup, min_trials)
            if last >= 0:
                traj_hi.append(float(gx[last]))
            if med.size:
                ax_traj.fill_between(gx, p10, p90, color=colour, alpha=0.25, linewidth=0.0,
                                     zorder=3)
                ax_traj.plot(gx, med, color=colour, linewidth=2.0, zorder=4,
                             label=f"{speed:.2f} m/s")
                z_lo.append(float(np.nanmin(p10))); z_hi.append(float(np.nanmax(p90)))
            if bad.size:
                ax_traj.plot(tr["ep_end_x"][bad], tr["ep_end_z"][bad], "x", color=fail_colour,
                             markersize=5, markeredgewidth=1.4, linestyle="none", zorder=5,
                             alpha=FAIL_MARKER_ALPHA,
                             label="failed" if (ci == 0 and i_s == 0) else None)

            if shared_x:
                abscissa, agrid = "x", gx
            else:
                ends = [float(tr["t"][sl[i]][-1]) for i in pick if sl[i].stop > sl[i].start]
                abscissa = "t"
                agrid = np.linspace(0.0, max(ends) if ends else 1.0, grid_points)
            okset = set(ok.tolist())
            for row, ch in enumerate(channels, start=1):
                ax = axes[row][ci]
                got, want, _ = CHANNELS[ch]
                if show_all:
                    for i in pick:
                        col = colour if i in okset else fail_colour
                        ax.plot(tr[abscissa][sl[i]], tr[got][sl[i]], color=col,
                                linewidth=0.5, alpha=faint, zorder=2)
                med, p10, p90, nsup = band([resample(tr[abscissa][sl[i]], tr[got][sl[i]],
                                                     agrid) for i in ok])
                med, p10, p90, last = trim_support(med, p10, p90, nsup, min_trials)
                if last >= 0:
                    vel_hi.append(float(agrid[last]))
                if med.size:
                    ax.fill_between(agrid, p10, p90, color=colour, alpha=0.25, linewidth=0.0,
                                    zorder=3)
                    ax.plot(agrid, med, color=colour, linewidth=2.0, zorder=4)
                # The COMMAND from the same trials rather than from the CLI: under steering wz
                # is a controller OUTPUT, so restating the requested speed would draw a
                # command that was never issued.
                cmed, cl, ch, cn = band([resample(tr[abscissa][sl[i]], tr[want][sl[i]], agrid)
                                         for i in ok])
                cmed, _cl, _ch, _last = trim_support(cmed, cl, ch, cn, min_trials)
                if cmed.size:
                    ax.plot(agrid, cmed, color="0.10", linewidth=1.5, linestyle="--",
                            zorder=6, label="commanded" if (ci == 0 and i_s == 0) else None)

    pad = 0.08 * max(1e-3, max(z_hi) - min(z_lo))
    # The trajectory axis ends at the last point any column still had `min_trials` episodes
    # on, falling back to the lane length when nothing was drawn.
    traj_limit = max(traj_hi) if traj_hi else max(x_hi)
    for a in axes[0]:
        a.set_ylim(min(z_lo) - pad, max(z_hi) + pad)
        a.set_xlim(0.0, min(max(x_hi), traj_limit * 1.02))
    if not shared_x and vel_hi:
        for row in range(1, nrows):
            for a in axes[row]:
                a.set_xlim(0.0, max(vel_hi) * 1.02)
    for row in range(1, nrows):
        lo = min(a.get_ylim()[0] for a in axes[row])
        hi = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(lo, hi)

    f = GRID_FONTS
    for row in range(nrows):
        for ci, a in enumerate(axes[row]):
            a.grid(True, alpha=0.25, linewidth=0.7)
            a.tick_params(labelsize=f["tick"])
            # Y tick labels only on the leftmost block column: every other column shares that
            # scale exactly, so repeating the numbers costs width and says nothing.
            if not lay.wants_ylabel(ci):
                a.tick_params(labelleft=False)
    for ci, c in enumerate(cols):
        axes[0][ci].set_title(c.title(multi, speeds), fontsize=f["title"])
        if lay.wants_ylabel(ci):
            axes[0][ci].set_ylabel(r"root $z$ [m]", fontsize=f["label"])
            for row, ch in enumerate(channels, start=1):
                axes[row][ci].set_ylabel(CHANNELS[ch][2], fontsize=f["label"])
    # Interior velocity rows never carry x labels; the block's LAST row always does, in both
    # arrangements -- in "grid" each block is a self-contained figure and needs its own axis.
    for row in range(1, nrows - 1):
        for a in axes[row]:
            a.tick_params(labelbottom=False)
    x_label = r"$x$ [m]" if shared_x else r"$t$ [s]"
    for a in axes[nrows - 1]:
        a.set_xlabel(x_label, fontsize=f["label"])
    if shared_x and nrows > 1:
        for a in axes[0]:
            a.tick_params(labelbottom=False)
    else:
        # Row 0 is on its OWN abscissa (downrange) while the rows below are in seconds, so it
        # carries its own ticks and label at the BOTTOM of its panel.
        for a in axes[0]:
            a.xaxis.set_label_position("bottom")
            a.tick_params(labeltop=False, labelbottom=True, top=False, labelsize=f["tick"])
            a.set_xlabel(r"$x$ [m]", fontsize=f["label"], labelpad=2)
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        axes[0][0].legend(h, l, loc="upper left", fontsize=f["legend"], handlelength=1.4,
                          borderpad=0.3, labelspacing=0.25)
    h2, l2 = axes[1][0].get_legend_handles_labels()
    if h2:
        axes[1][0].legend(h2, l2, loc="best", fontsize=f["legend"], handlelength=1.4,
                          borderpad=0.3)

    m0 = runs[0][1]["meta"]
    who = "; ".join(f"{r[0] or 'run'}: generator {os.path.basename(r[1]['meta']['fm_onnx'])}, "
                    f"{r[1]['meta']['trials']} trials" for r in runs)
    caption = (
        f"Velocity tracking across terrain. Per column: the root's x-z path over the terrain "
        f"outline (top) and the commanded (dashed) against the achieved velocity (bottom, "
        f"{'downrange' if shared_x else 'time'} axis). "
        f"{m0['approach_m']:.0f} m of flat approach and {m0['runout_m']:.0f} m of runout about "
        f"each feature; the shaded span is the feature, and on a flat control column it marks "
        f"where a feature would have been. The command steps from zero to the stated speed at "
        f"t = {m0['cmd_hold_s']:.2f} s, with a heading P controller "
        f"(kp = {m0['steer_kp']:.2f} 1/s) holding the lane centreline -- which is "
        f"off-distribution on stairs, where no training clip was ever commanded a turn. "
        f"Trials within a cell differ ONLY in the generator's flow-noise draw, and are seed-"
        f"paired across cells; line = median, band = 10-90 percentile over the trials that "
        f"traversed, faint curves = every trial, red x = where a failed trial ended. "
        f"Height and velocity scales are shared across all columns. "
        f"Traversed: {'; '.join(counts)}. {who}. "
        f"Velocities are in the root's yaw-aligned (gravity-aligned) frame, the frame the "
        f"command is issued in; omega_z is the world-frame yaw rate."
    )
    return fig, caption


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="+",
                   help="One or more vel_tracking_sweep.py output directories, each optionally "
                        "suffixed ':terrain[,terrain]' to take only those terrains from that "
                        "run -- which is how 'the parent on every terrain, with the "
                        "in-distribution box' is assembled from two sweeps. Several are "
                        "drawn as adjacent column groups on ONE shared scale, which is how a "
                        "box at two depths or a parent against a fine-tune are compared -- a "
                        "run holds one geometry per terrain, so the contrast cannot live "
                        "inside a single run.")
    p.add_argument("--label", action="append", default=None,
                   help="Repeatable, one per run_dir in order, shown under the terrain in the "
                        "column title. Default: the directory's basename.")
    p.add_argument("--out", default="",
                   help="Figure path WITHOUT an extension. Default <first run_dir>/"
                        "fig_vel_tracking.")
    p.add_argument("--channels", default="vx",
                   help="Comma-separated velocity channels, one row each, from "
                        f"{','.join(CHANNELS)}. 'vx,wz' shows the steering too.")
    p.add_argument("--vel_axis", choices=("time", "x"), default="time",
                   help="Abscissa for the velocity rows. 'time' keeps the opening command "
                        "STEP visible; 'x' shares the trajectory axis so an excursion lines "
                        "up with the step that caused it, at the cost of collapsing the hold.")
    p.add_argument("--no_all_trials", action="store_true",
                   help="Draw only the median and the band, not every trial's own curve.")
    p.add_argument("--terrain", action="append", default=None,
                   help="Repeatable. Draw only these terrains, in this order. Default: every "
                        "terrain the runs hold. Use it to compare ONE terrain across runs -- "
                        "e.g. --terrain box over a trained-depth run and a deep-box run.")
    p.add_argument("--speed", type=float, action="append", default=None,
                   help="Repeatable. Draw only these speeds. Default: every speed in the "
                        "first run. A speed a run does not hold is an error, not an empty "
                        "panel.")
    p.add_argument("--layout", choices=("rows", "grid"), default="rows",
                   help="'rows': terrains across, channels down -- best for reading ONE "
                        "quantity across terrains on a shared scale. 'grid': one block of "
                        "stacked panels per terrain, near-square, filling DOWN each grid "
                        "column first (so four terrains go top-left, bottom-left, top-right, "
                        "bottom-right). Scales are shared across blocks either way.")
    p.add_argument("--grid_shape", default="",
                   help="'RxC' block arrangement for --layout grid. Default: near-square.")
    p.add_argument("--min_trials", type=int, default=MIN_TRIALS,
                   help="Cut each curve where fewer than this many episodes still contribute, "
                        "and end the axis there. The time axis otherwise runs to the slowest "
                        "surviving trial, where the median is an average over a handful. "
                        "0 disables the trim.")
    p.add_argument("--grid_points", type=int, default=600,
                   help="Samples on the shared abscissa the median is taken over.")
    p.add_argument("--no_tex", action="store_true", help="Force mathtext instead of LaTeX.")
    return p


def main() -> int:
    """Load the runs and write the figure plus its caption sidecar."""
    args = build_parser().parse_args()
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    unknown = [c for c in channels if c not in CHANNELS]
    if unknown:
        raise SystemExit(f"unknown channel(s) {unknown}; expected from {list(CHANNELS)}")
    if not channels:
        raise SystemExit("--channels needs at least one of " + ", ".join(CHANNELS))
    specs = [parse_run_spec(s) for s in args.run_dir]
    labels = args.label or [os.path.basename(os.path.normpath(d)) for d, _ in specs]
    if len(labels) != len(specs):
        raise SystemExit(f"{len(specs)} run dir(s) but {len(labels)} --label(s); "
                         "pass one label per run, in order")

    runs = []
    for label, (d, pick) in zip(labels, specs):
        summary, traces = load_run(d)
        unknown = [k for k in pick if k not in summary["meta"]["terrains"]]
        if unknown:
            raise SystemExit(f"{d} holds {summary['meta']['terrains']}, not {unknown}")
        runs.append((label if len(specs) > 1 else "", summary, traces, pick))
    only = tuple(args.speed or ())
    # Checked against EVERY run, not just the first: a speed one arm is missing would
    # otherwise draw that arm's column empty and read as a policy that never traversed.
    for label, s, *_ in runs:
        have = list(s["meta"]["speeds"])
        missing = [sp for sp in only if not any(abs(sp - h) < 1e-9 for h in have)]
        if missing:
            raise SystemExit(f"{label or args.run_dir[0]} holds speeds {have}, not {missing}")

    shape = None
    if args.grid_shape:
        try:
            r, _, c = args.grid_shape.lower().partition("x")
            shape = (int(r), int(c))
        except ValueError as exc:
            raise SystemExit(f"--grid_shape wants 'RxC', got {args.grid_shape!r}") from exc
    plot_style.apply_paper_style(usetex=False if args.no_tex else None)
    fig, caption = plot_run(runs, channels=channels, vel_axis=args.vel_axis,
                            show_all=not args.no_all_trials, grid_points=args.grid_points,
                            only_speeds=only, only_kinds=tuple(args.terrain or ()),
                            min_trials=max(0, int(args.min_trials)),
                            layout=args.layout, grid_shape=shape)
    stem = args.out or os.path.join(specs[0][0], "fig_vel_tracking")
    # The command goes in the caption sidecar, because a figure that cannot say how it was
    # made is not reproducible in practice however deterministic the code is: these are
    # assembled from several runs with per-run terrain selection, a layout and a label per
    # arm, and none of that is recoverable from the image or from the sweeps it drew on.
    cmd = " ".join(shlex.quote(a) for a in [os.path.basename(sys.argv[0]), *sys.argv[1:]])
    caption = f"{caption}\n\nRegenerate: cd transfer/sim && python {cmd}"
    for w in plot_style.save_figure(fig, stem, caption=caption):
        print(f"[plot] {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
