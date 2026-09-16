# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared paper-figure style: rcParams, palette, sizes, and a saving helper.

Lives in ``scripts/rsl_rl/`` rather than in either source package, because it has to be
importable from three places with three different paths and, in one of them, a venv that has
neither package installed:

* ``source/robot_rl/robot_rl/`` is out: its ``__init__`` does ``from .tasks import *`` and
  imports ``isaaclab_tasks``, so *any* ``import robot_rl.x`` drags in IsaacLab -- fatal for a
  plotting tool whose point is running without it, and impossible in ``env_transfer``.
* ``source/motion_gen/`` is out on its stated dependency contract (torch/numpy/h5py/pyyaml,
  explicitly not IsaacLab); adding matplotlib to an offline clip-generation package to hold a
  plot style is the wrong shape, and it is not installed in ``env_transfer`` either.

Consumers reach it by bare name after putting ``scripts/rsl_rl`` on ``sys.path`` -- which
``scripts/flow_matching/fm_augment_probe.py`` and ``scripts/rsl_rl/eval/_paths.py`` already do.

**Importing this module does not touch global state.**  Only :func:`apply_paper_style` writes
``rcParams``.  That matters because a module which restyles on import poisons every other plot
in the same process -- exactly what would happen when a test imports both a paper figure and
``play_plots``' diagnostic grids.

Deliberately NOT applied to ``scripts/rsl_rl/play_plots.py``.  That is 2000 lines of
per-channel diagnostic grids with 10-pt subplot titles and up to 155 axes; 20-pt paper labels,
no-titles and ``usetex`` would make them slower (LaTeX across 155 axes) and less readable, for
figures that are never published.  It borrows nothing from here on purpose.
"""

from __future__ import annotations

#: Paul Tol vibrant, colourblind-friendly.
COLORS: dict[str, str] = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
    "black": "#000000",
}

COLOR_CYCLE: list[str] = [COLORS[k] for k in
                          ("blue", "orange", "teal", "red", "cyan", "magenta", "grey", "black")]

#: Journal single-column and full-width sizes.
SINGLE_COL: tuple[float, float] = (3.5, 2.8)
DOUBLE_COL: tuple[float, float] = (7.0, 4.5)

#: Trade dpi rather than emit a figure of absurd pixel count.  Same budget and bounds as
#: ``play_plots._fig_dpi``, which exists because a wide subplot grid otherwise renders at
#: hundreds of megapixels.
_FIG_PIXEL_BUDGET = 15e6
_DPI_MAX, _DPI_MIN = 300, 70

#: An SVG of a large scatter or an image is tens of megabytes of unopenable XML, so it is
#: skipped past this many artists.
_SVG_ARTIST_MAX = 20_000

_LATEX_OK: bool | None = None
_APPLIED = False


def latex_available() -> bool:
    """Whether ``text.usetex`` actually renders here. Probed once, then cached.

    The probe draws a real figure, because a LaTeX install can be present but incomplete.
    Cached because 14 figures must not mean 14 LaTeX subprocess round-trips.
    """
    global _LATEX_OK
    if _LATEX_OK is not None:
        return _LATEX_OK
    import matplotlib as mpl  # noqa: PLC0415

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        prev = mpl.rcParams["text.usetex"]
        mpl.rcParams["text.usetex"] = True
        fig, ax = plt.subplots()
        ax.set_title(r"$\alpha$")
        fig.canvas.draw()
        plt.close(fig)
        mpl.rcParams["text.usetex"] = prev
        _LATEX_OK = True
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        _LATEX_OK = False
    return _LATEX_OK


def apply_paper_style(usetex: bool | None = None) -> bool:
    """Install the paper rcParams. Idempotent; safe to call from every entry point.

    Args:
        usetex: Force LaTeX on/off; ``None`` probes for it and falls back to mathtext with a
            printed warning, since CI boxes have no LaTeX.

    Returns:
        Whether LaTeX rendering is active.
    """
    global _APPLIED
    import matplotlib as mpl  # noqa: PLC0415

    use = latex_available() if usetex is None else bool(usetex)
    mpl.rcParams.update({
        "text.usetex": use,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "lines.linewidth": 4, "lines.markersize": 10, "lines.markeredgewidth": 2,
        "axes.linewidth": 1.5, "axes.labelsize": 20, "axes.titlesize": 22,
        "axes.titlepad": 12, "axes.labelpad": 8,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "xtick.major.width": 1.5, "ytick.major.width": 1.5,
        "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.fontsize": 15, "legend.framealpha": 0.9, "legend.edgecolor": "0.4",
        "legend.fancybox": False, "legend.handlelength": 2.0,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
    })
    if use:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
    else:
        mpl.rcParams["mathtext.fontset"] = "cm"
        if not _APPLIED:
            print("[plot_style] LaTeX not available -- falling back to mathtext. Fonts will "
                  "not match Computer Modern exactly.")
    _APPLIED = True
    return use


def tex_escape(s: str) -> str:
    """Escape a plain string for use as a LaTeX label.

    Not cosmetic: under ``text.usetex`` a bare underscore in a tick label raises a LaTeX
    error mid-figure, and family names, cell keys and run labels are full of them.  Centralised
    here because getting it wrong is the single most likely first-run failure --
    ``play_stats_comparison._wrap_label`` feeds values like ``walking_clf`` straight into
    ``set_xticklabels`` under its own ``usetex=True`` today.
    """
    out = str(s)
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def figsize(cols: int = 2, rows: int = 1, *, aspect: float = 0.64) -> tuple[float, float]:
    """Figure size for a ``rows``-row figure at journal width.

    ``cols`` selects the WIDTH (1 = single column, >=2 = full width); ``rows`` scales the
    height proportionally, as the skill prescribes for subplot grids.  One row at full width
    reproduces ``DOUBLE_COL``.
    """
    base = DOUBLE_COL if cols >= 2 else SINGLE_COL
    if rows <= 1:
        return base                     # exactly the standard size, no surprises
    return (base[0], max(2.0, base[0] * aspect * rows))


def pixel_budget_dpi(fig) -> int:
    """Largest dpi that keeps the raster under the pixel budget."""
    w, h = fig.get_size_inches()
    area = max(1e-6, float(w) * float(h))
    dpi = int((_FIG_PIXEL_BUDGET / area) ** 0.5)
    return max(_DPI_MIN, min(_DPI_MAX, dpi))


def _artist_count(fig) -> int:
    """Rough artist count, for deciding whether an SVG is worth writing."""
    n = 0
    for ax in fig.get_axes():
        n += len(ax.lines) + len(ax.patches) + len(ax.collections) + len(ax.texts)
        n += sum(getattr(c, "get_offsets", lambda: [])().shape[0]
                 if hasattr(c, "get_offsets") else 0 for c in ax.collections)
        if ax.images:
            n += _SVG_ARTIST_MAX + 1        # an image in an SVG is always the wrong choice
    return int(n)


def save_figure(fig, path_no_ext, *, svg: bool = True, caption: str | None = None,
                exact_size: bool = False) -> list[str]:
    """Save PNG (+ SVG) and, optionally, a caption sidecar. Never PDF.

    The caption file is what makes the "no titles, use captions" rule workable: it holds the
    definitions in force for that figure -- which success notion, which stratum, which
    interval -- so a reader is never left guessing and the text can be pasted into LaTeX.

    Args:
        fig: The figure.
        path_no_ext: Output path without an extension.
        svg: Also write an SVG, unless the artist count makes it unopenable.
        caption: Caption text for the sidecar.
        exact_size: Write the figure at its DECLARED size instead of cropping to content.
            The default ``savefig.bbox="tight"`` trims each figure to its own ink, so two
            figures declared at the same width come out a few pixels apart and their panel
            columns no longer line up when one is STACKED above the other. Set this for
            figures meant to be stacked; it costs a little whitespace at the margins.

    Returns:
        The paths written.
    """
    import pathlib  # noqa: PLC0415

    p = pathlib.Path(str(path_no_ext))
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # `bbox_inches=None` does NOT mean "the whole canvas": matplotlib falls back to
    # rcParams["savefig.bbox"], which this style sets to "tight". Only the rcParam turns
    # the crop off, so exact_size goes through an rc_context rather than the kwarg.
    #
    # A FRESH context per save: `rc_context` is a generator-based context manager and
    # entering one twice raises, which made the second (SVG) write fail whenever the figure
    # was small enough for an SVG to be written at all.
    overrides = {"savefig.bbox": "standard"} if exact_size else {}
    written = []
    png = str(p) + ".png"
    with plt.rc_context(overrides):
        fig.savefig(png, dpi=pixel_budget_dpi(fig))
    written.append(png)
    if svg:
        if _artist_count(fig) > _SVG_ARTIST_MAX:
            print(f"[plot_style] {p.name}: skipping SVG ({_artist_count(fig)} artists or an "
                  "image); the vector file would be tens of MB and unopenable.")
        else:
            with plt.rc_context(overrides):
                fig.savefig(str(p) + ".svg")
            written.append(str(p) + ".svg")
    if caption:
        cap = str(p) + ".caption.txt"
        pathlib.Path(cap).write_text(caption.rstrip() + "\n")
        written.append(cap)
    return written
