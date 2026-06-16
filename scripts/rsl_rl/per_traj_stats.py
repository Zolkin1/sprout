# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-trajectory CLF/tracking + termination statistics for a headless eval.

The multiskill task tracks ~1300 reference trajectories, each a global index
annotated by skill and conditioning (commanded velocity + terrain dims). This
module captures, per trajectory:

* **CLF / tracking quality** — sample-weighted mean of the Lyapunov value ``V``
  and the frame tracking error ``||y_act - y_des||`` over "warm" (non-transient,
  non-terminal) steps.
* **Terminations** — per-termination-term fire counts and episode-end counts,
  giving the fraction of episode-ends via each cause for that trajectory.

Capture is an online, device-tensor accumulator keyed by the global trajectory
index (memory O(num_trajectories x num_terms), independent of rollout length).
Display scales to ~1300 trajectories by aggregating over the
(skill x velocity x terrain) grid the trajectories live on: a full CSV (one row
per trajectory), a human-scannable text triage report (per-skill aggregates +
top-K worst + low-coverage warnings), and per-skill metric-vs-conditioning plots.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
import torch

# Frames the ``frame_drift`` termination monitors — the natural default for the
# per-trajectory tracking-error metric so the two line up.
DEFAULT_TRACK_FRAMES = ("pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link")

# Conditioning column layout (ConditionerData.to_continuous_tensor).
_COND_NAMES = ("vel_x", "vel_y", "vel_yaw", "terrain_w", "terrain_h", "terrain_len")

# Termination terms shown in the compact text report (the wide per-term breakdown
# lives in the CSV). Intersected with the active terms; falls back to the first
# few if none of these are present.
_HEADLINE_TERMS = ("time_out", "frame_drift", "pelvis_upright", "base_height")


def _cond_label(row) -> str:
    """Compact conditioning label: ``vx=..,vy=..,vyaw=..`` (+ terrain when set).

    The full commanded velocity is always shown so trajectories that share a
    ``vx`` but differ in lateral / yaw command stay distinct; terrain dims are
    appended only when non-zero (i.e. for terrain skills).
    """
    parts = [f"vx={float(row[0]):.2f}", f"vy={float(row[1]):.2f}", f"vyaw={float(row[2]):.2f}"]
    if len(row) >= 6:
        tw, th, tl = float(row[3]), float(row[4]), float(row[5])
        if tw:
            parts.append(f"tw={tw:.2f}")
        if th:
            parts.append(f"th={th:.2f}")
        if tl:
            parts.append(f"tl={tl:.2f}")
    return ",".join(parts)


def build_traj_names(manager) -> list[str] | None:
    """Build per-trajectory display names ``"<skill>/vx=..,vy=..,vyaw=..[,terrain]"``.

    Trajectories frequently share a skill name and even a forward velocity, so
    the name carries the full velocity conditioning (and terrain dims when
    present) to disambiguate. The global index is still the unique key (CSV
    ``traj_idx`` column / ``#`` column in the text tables); this is the readable
    label. Falls back to ``"<skill>/<idx>"`` when conditioning is unavailable.

    Args:
        manager: The multiskill manager exposing ``num_trajectories``,
            ``skill_labels`` and ``_global_conditioning``.

    Returns:
        List of length ``num_trajectories``, or ``None`` if the manager has no
        trajectories.
    """
    n = int(getattr(manager, "num_trajectories", 0))
    if n <= 0:
        return None
    skills = list(getattr(manager, "skill_labels", []) or [])
    cond = getattr(manager, "_global_conditioning", None)
    cond_np = None
    if cond is not None:
        try:
            cond_np = cond.detach().cpu().numpy()
        except Exception:
            cond_np = None
    names: list[str] = []
    for i in range(n):
        skill = skills[i] if i < len(skills) else "?"
        if cond_np is not None and i < len(cond_np):
            names.append(f"{skill}/{_cond_label(cond_np[i])}")
        else:
            names.append(f"{skill}/{i}")
    return names


def resolve_frame_pos_indices(
    output_names: list[str], frame_names, device
) -> tuple[torch.Tensor | None, list[str]]:
    """Resolve ``{frame}:pos_{x,y,z}`` column indices in a command's outputs.

    Same lookup pattern as ``frame_deviation_from_reference`` but tolerant:
    frames lacking a full position triplet are skipped (this is a diagnostic,
    not a termination), and the frames actually used are returned.

    Args:
        output_names: ``cmd.ordered_pos_output_names``.
        frame_names: Frames to monitor.
        device: Device for the returned index tensor.

    Returns:
        ``(pos_idx, used_frames)`` where ``pos_idx`` is a ``[F, 3]`` long tensor
        (or ``None`` if no frame resolved) and ``used_frames`` lists the frames
        that contributed.
    """
    pos_indices: list[list[int]] = []
    used: list[str] = []
    for name in frame_names:
        try:
            xyz = [output_names.index(f"{name}:pos_{a}") for a in ("x", "y", "z")]
        except ValueError:
            continue
        pos_indices.append(xyz)
        used.append(name)
    if not pos_indices:
        return None, []
    return torch.tensor(pos_indices, dtype=torch.long, device=device), used


def frame_tracking_error(cmd, pos_idx: torch.Tensor) -> torch.Tensor:
    """Mean over monitored frames of ``||y_act - y_des||`` per env.

    Args:
        cmd: Trajectory command term exposing ``y_act`` / ``y_des``.
        pos_idx: ``[F, 3]`` long tensor of position-output columns per frame
            (from :func:`resolve_frame_pos_indices`).

    Returns:
        ``[num_envs]`` float tensor of the mean per-frame position-error norm.
    """
    err = (cmd.y_act - cmd.y_des)[:, pos_idx]            # [N, F, 3]
    return torch.linalg.norm(err, dim=-1).mean(dim=-1)   # [N]


class PerTrajectoryStats:
    """Online per-trajectory accumulator for CLF/tracking + termination stats.

    Usage::

        stats = PerTrajectoryStats(num_trajectories, term_names, device,
                                   skill_labels=..., conditioning=..., traj_names=...)
        for step in rollout:
            stats.collect_step(traj_idx, v, track_err, active, term_fired, dones)
        stats.finalize()
        stats.write_csv(...); stats.write_text_report(...); stats.make_plots(...)
    """

    def __init__(
        self,
        num_trajectories: int,
        term_names: list[str],
        device: torch.device | str,
        skill_labels: list[str] | None = None,
        conditioning: torch.Tensor | None = None,
        traj_names: list[str] | None = None,
    ) -> None:
        """Allocate accumulators and stash trajectory metadata.

        Args:
            num_trajectories: Number of global trajectories ``T``.
            term_names: Ordered termination-term names ``K`` to track.
            device: Device for the accumulator tensors.
            skill_labels: Per-trajectory skill name, length ``T``.
            conditioning: ``[T, 6]`` conditioning tensor (vel + terrain dims).
            traj_names: Per-trajectory display names, length ``T``.
        """
        self.T = int(num_trajectories)
        self.term_names = list(term_names)
        self.K = len(self.term_names)
        self.device = torch.device(device)

        self.sum_v = torch.zeros(self.T, dtype=torch.float32, device=self.device)
        self.sum_track_err = torch.zeros(self.T, dtype=torch.float32, device=self.device)
        self.count_metric = torch.zeros(self.T, dtype=torch.long, device=self.device)
        self.term_count = torch.zeros((self.T, self.K), dtype=torch.long, device=self.device)
        self.episode_ends = torch.zeros(self.T, dtype=torch.long, device=self.device)

        self.skill_labels = list(skill_labels) if skill_labels is not None else None
        self.conditioning = (
            conditioning.detach().cpu().numpy() if conditioning is not None else None
        )
        self.traj_names = list(traj_names) if traj_names is not None else None

        self._steps = 0
        self._saw_v = False
        self._saw_track = False
        self._result: dict[str, Any] | None = None

    def collect_step(
        self,
        traj_idx: torch.Tensor,
        v: torch.Tensor | None,
        track_err: torch.Tensor | None,
        active: torch.Tensor,
        term_fired: dict[str, torch.Tensor],
        dones: torch.Tensor,
    ) -> None:
        """Fold one rollout step into the per-trajectory accumulators.

        Args:
            traj_idx: ``[N]`` long global trajectory index active during this
                step (snapshot taken BEFORE ``env.step`` so terminations are
                attributed to the trajectory that was live, not a post-reset
                resample).
            v: ``[N]`` CLF value, or ``None`` if the command has none.
            track_err: ``[N]`` frame tracking error, or ``None``.
            active: ``[N]`` bool mask of warm, non-terminal steps for the
                CLF/tracking means.
            term_fired: ``{term_name: [N] bool}`` of which envs ended via each
                term this step.
            dones: ``[N]`` bool of episode-ends this step.
        """
        self._steps += 1
        traj_idx = traj_idx.to(self.device)

        if active.any():
            ia = traj_idx[active]
            ones = torch.ones_like(ia)
            self.count_metric.index_add_(0, ia, ones)
            if v is not None:
                self._saw_v = True
                self.sum_v.index_add_(0, ia, v.to(self.device)[active].float())
            if track_err is not None:
                self._saw_track = True
                self.sum_track_err.index_add_(0, ia, track_err.to(self.device)[active].float())

        # Terminations: count every fire (no warm gate), keyed by the active traj.
        fired_mat = torch.stack(
            [term_fired[n].to(self.device).long() for n in self.term_names], dim=1
        )  # [N, K]
        self.term_count.index_add_(0, traj_idx, fired_mat)

        if dones.any():
            de = traj_idx[dones.to(self.device)]
            self.episode_ends.index_add_(0, de, torch.ones_like(de))

    def finalize(self) -> dict[str, Any]:
        """Compute per-trajectory arrays and cache them for the writers.

        Returns:
            Dict of numpy arrays: ``mean_v`` / ``mean_track_err`` (NaN where no
            samples), ``samples``, ``episode_ends``, ``term_frac`` ``[T, K]``
            (NaN where no episode-ends), plus ``term_count`` and stashed metadata.
        """
        samples = self.count_metric.cpu().numpy()
        ends = self.episode_ends.cpu().numpy()
        sum_v = self.sum_v.cpu().numpy()
        sum_te = self.sum_track_err.cpu().numpy()
        term_count = self.term_count.cpu().numpy()

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_v = np.where(samples > 0, sum_v / np.maximum(samples, 1), np.nan)
            mean_te = np.where(samples > 0, sum_te / np.maximum(samples, 1), np.nan)
            ends_col = ends[:, None]
            term_frac = np.where(ends_col > 0, term_count / np.maximum(ends_col, 1), np.nan)

        self._result = {
            "mean_v": mean_v if self._saw_v else None,
            "mean_track_err": mean_te if self._saw_track else None,
            "samples": samples,
            "episode_ends": ends,
            "term_count": term_count,
            "term_frac": term_frac,
            "term_names": self.term_names,
            "skill_labels": self.skill_labels,
            "conditioning": self.conditioning,
            "traj_names": self.traj_names,
            "steps": self._steps,
        }
        return self._result

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _require_result(self) -> dict[str, Any]:
        if self._result is None:
            self.finalize()
        assert self._result is not None
        return self._result

    def write_csv(self, path: str) -> None:
        """Write the authoritative one-row-per-trajectory CSV."""
        r = self._require_result()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        names = r["traj_names"] or [f"traj_{i}" for i in range(self.T)]
        skills = r["skill_labels"] or ["?"] * self.T
        cond = r["conditioning"]
        mean_v, mean_te = r["mean_v"], r["mean_track_err"]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            header = ["traj_idx", "name", "skill", *_COND_NAMES,
                      "samples", "episode_ends", "mean_v", "mean_track_err"]
            header += [f"{n}_frac" for n in self.term_names]
            w.writerow(header)
            for i in range(self.T):
                cvals = list(cond[i]) if cond is not None else [""] * len(_COND_NAMES)
                row = [i, names[i], skills[i], *[f"{c:.4f}" if c != "" else "" for c in cvals],
                       int(r["samples"][i]), int(r["episode_ends"][i]),
                       _fmt(mean_v[i] if mean_v is not None else np.nan),
                       _fmt(mean_te[i] if mean_te is not None else np.nan)]
                row += [_fmt(r["term_frac"][i, k]) for k in range(self.K)]
                w.writerow(row)

    def write_text_report(
        self, path: str, num_envs: int | None = None,
        top_k: int = 20, min_samples: int = 20, min_eps: int = 5,
    ) -> None:
        """Write the human-scannable triage report (not all T rows)."""
        report = self.format_text_report(
            num_envs=num_envs, top_k=top_k, min_samples=min_samples, min_eps=min_eps
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(report)

    def format_text_report(
        self, num_envs: int | None = None,
        top_k: int = 20, min_samples: int = 20, min_eps: int = 5,
    ) -> str:
        """Render the triage report string."""
        r = self._require_result()
        names = r["traj_names"] or [f"traj_{i}" for i in range(self.T)]
        skills = np.array(r["skill_labels"] or ["?"] * self.T, dtype=object)
        samples, ends = r["samples"], r["episode_ends"]
        mean_v, mean_te, term_frac = r["mean_v"], r["mean_track_err"], r["term_frac"]
        term_count = r["term_count"]
        visited = int(((samples > 0) | (ends > 0)).sum())

        headline = [t for t in _HEADLINE_TERMS if t in self.term_names] or self.term_names[:4]
        hl_idx = [self.term_names.index(t) for t in headline]

        L: list[str] = ["=== PER-TRAJECTORY STATS ==="]
        hdr = f"steps: {r['steps']}   episode-ends: {int(ends.sum())}   trajectories: {self.T} (visited: {visited})"
        if num_envs is not None:
            hdr += f"   envs: {num_envs}"
        L.append(hdr)
        L.append("metric means are sample-weighted over warm, non-terminal steps; "
                 "term fracs are per episode-end.")

        # -- Per-skill aggregate --
        L.append("")
        L.append("-- Per-skill aggregate --")
        cols = ["skill", "trajs", "visited", "samples", "ends", "mean_v", "track_err", *headline]
        L.append(_row(cols, [16, 6, 7, 9, 7, 9, 10] + [11] * len(headline)))
        for sk in _unique_preserve(skills):
            m = skills == sk
            s_samp = int(samples[m].sum())
            s_ends = int(ends[m].sum())
            s_vis = int(((samples[m] > 0) | (ends[m] > 0)).sum())
            mv = _wmean(mean_v, samples, m) if mean_v is not None else np.nan
            mt = _wmean(mean_te, samples, m) if mean_te is not None else np.nan
            fr = [(_safe_div(term_count[m, k].sum(), s_ends)) for k in hl_idx]
            L.append(_row(
                [sk, int(m.sum()), s_vis, s_samp, s_ends, _fmt(mv), _fmt(mt), *[_fmt(x) for x in fr]],
                [16, 6, 7, 9, 7, 9, 10] + [11] * len(headline),
            ))

        # -- Top-K worst by mean_v --
        if mean_v is not None:
            L.append("")
            L.append(f"-- Top {top_k} worst by mean_v (samples >= {min_samples}) --")
            L += self._top_k_block(mean_v, samples >= min_samples, names, skills,
                                    samples, ends, mean_v, mean_te, term_frac, hl_idx, headline, top_k)

        # -- Top-K worst by frame_drift (or first non-timeout term) --
        rank_term = "frame_drift" if "frame_drift" in self.term_names else self._first_non_timeout()
        if rank_term is not None:
            ti = self.term_names.index(rank_term)
            L.append("")
            L.append(f"-- Top {top_k} worst by {rank_term}_frac (episode-ends >= {min_eps}) --")
            key = np.where(ends > 0, term_frac[:, ti], -np.inf)
            L += self._top_k_block(key, ends >= min_eps, names, skills,
                                    samples, ends, mean_v, mean_te, term_frac, hl_idx, headline, top_k)

        # -- Low coverage --
        low = (samples < min_samples)
        zero = (ends == 0)
        L.append("")
        L.append("-- Low coverage --")
        ex = [f"#{i}:{names[i]}" for i in np.argsort(samples)[:8]]
        L.append(f"{int(low.sum())} trajectories with samples < {min_samples} "
                 f"({int(zero.sum())} with zero episode-ends). least-sampled: {', '.join(ex)}")
        return "\n".join(L) + "\n"

    def _top_k_block(self, key, mask, names, skills, samples, ends,
                     mean_v, mean_te, term_frac, hl_idx, headline, top_k) -> list[str]:
        # ``#`` is the global trajectory index — the unique key, since two
        # trajectories can share identical conditioning (and thus a name).
        widths = [6, 52, 13, 8, 6, 10, 10] + [11] * len(headline)
        head = _row(["#", "trajectory", "skill", "samples", "ends", "mean_v", "track_err", *headline], widths)
        out = [head]
        order = np.argsort(-np.where(mask, key, -np.inf))
        shown = 0
        for i in order:
            if not mask[i] or not np.isfinite(key[i]):
                break
            mv = _fmt(mean_v[i]) if mean_v is not None else "-"
            mt = _fmt(mean_te[i]) if mean_te is not None else "-"
            fr = [_fmt(term_frac[i, k]) for k in hl_idx]
            out.append(_row([int(i), names[i], str(skills[i]), int(samples[i]), int(ends[i]), mv, mt, *fr], widths))
            shown += 1
            if shown >= top_k:
                break
        if shown == 0:
            out.append("  (none meet the coverage threshold)")
        return out

    def _first_non_timeout(self) -> str | None:
        for n in self.term_names:
            if n != "time_out":
                return n
        return None

    def make_plots(
        self, out_dir: str, term_for_plot: str | None = None,
        max_facets: int = 6, cmap: str = "viridis",
    ) -> list[str]:
        """Per-skill metric-as-color-over-conditioning plots.

        For each skill we detect which conditioning dims actually vary, then
        render each metric as **color** over the two highest-variance dims
        (x, y), with marker size = #samples. A 3rd varying dim is faceted into
        columns; any further dims are marginalized (noted in the title). With a
        single varying dim it degenerates to a metric-on-Y scatter. One figure
        per ``(skill, metric)`` written as ``per_traj_<skill>__<metric>.png``.

        Args:
            out_dir: Directory to write PNGs into.
            term_for_plot: Termination term whose fraction to plot
                (default ``frame_drift``, else the first non-timeout term).
            max_facets: Max facet columns for the 3rd varying dim.
            cmap: Matplotlib colormap (high metric = bright = worse).

        Returns:
            List of written file paths.
        """
        r = self._require_result()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        cond = r["conditioning"]
        if cond is None or r["skill_labels"] is None:
            return []
        skills = np.array(r["skill_labels"], dtype=object)
        samples = r["samples"]
        mean_v, mean_te, term_frac = r["mean_v"], r["mean_track_err"], r["term_frac"]

        term = term_for_plot or ("frame_drift" if "frame_drift" in self.term_names
                                 else self._first_non_timeout())
        term_series = term_frac[:, self.term_names.index(term)] if term else None

        metrics: list[tuple[str, np.ndarray]] = []
        if mean_v is not None:
            metrics.append(("mean_v", mean_v))
        if mean_te is not None:
            metrics.append(("track_err", mean_te))
        if term_series is not None:
            metrics.append((f"{term}_frac", term_series))
        if not metrics:
            return []

        written: list[str] = []
        for sk in _unique_preserve(skills):
            m = (skills == sk) & (samples > 0)
            if m.sum() < 2:
                continue
            idx = np.where(m)[0]
            # Conditioning dims that vary for this skill, ordered by spread.
            std = cond[idx].std(axis=0)
            varying = [d for d in range(cond.shape[1]) if std[d] > 1e-3]
            if not varying:
                continue
            # Prefer vel_x on the x-axis when it varies; rest by descending std.
            if 0 in varying:
                order = [0] + sorted((d for d in varying if d != 0), key=lambda d: -std[d])
            else:
                order = sorted(varying, key=lambda d: -std[d])
            for label, series in metrics:
                p = _plot_metric_landscape(
                    plt, out_dir, sk, idx, cond, order, label,
                    series[idx], samples[idx], max_facets, cmap,
                )
                if p:
                    written.append(p)
        return written


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _facet_bins(fvals: np.ndarray, max_facets: int):
    """Partition ``fvals`` into facet masks + labels.

    Uses the (rounded) unique values when there are few of them — the trajectory
    library is on a discrete conditioning grid — else falls back to quantile
    bins capped at ``max_facets``.
    """
    rounded = np.round(fvals, 2)
    uniq = np.unique(rounded)
    if len(uniq) <= max_facets:
        return [rounded == u for u in uniq], [f"{u:.2f}" for u in uniq]
    qs = np.quantile(fvals, np.linspace(0.0, 1.0, max_facets + 1))
    qs[-1] += 1e-9
    masks, labels = [], []
    for i in range(max_facets):
        masks.append((fvals >= qs[i]) & (fvals < qs[i + 1]))
        labels.append(f"{qs[i]:.2f}-{qs[i + 1]:.2f}")
    return masks, labels


def _plot_metric_landscape(
    plt, out_dir, sk, idx, cond, order, label, vals, samples, max_facets, cmap,
) -> str | None:
    """Render one ``(skill, metric)`` figure: metric as color over conditioning.

    ``idx`` are the skill's visited trajectory indices; ``order`` is the
    varying conditioning dims (descending spread). Color = ``vals``, size =
    ``samples``. Two dims -> single 2D panel; >=3 dims -> faceted by the 3rd.
    """
    finite = np.isfinite(vals)
    if not finite.any():
        return None
    s = samples.astype(float)
    sizes = 15.0 + 60.0 * np.sqrt(np.clip(s, 0, None) / max(s.max(), 1.0))
    vmin, vmax = np.nanpercentile(vals[finite], [2.0, 98.0])  # robust to outliers
    if vmin == vmax:
        vmax = vmin + 1e-6
    xdim = order[0]
    x = cond[idx, xdim]
    nd = len(order)

    if nd == 1:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        ax.scatter(x[finite], vals[finite], s=sizes[finite], alpha=0.7)
        ax.set_xlabel(_COND_NAMES[xdim])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{sk}: {label} vs {_COND_NAMES[xdim]}")
    else:
        ydim = order[1]
        y = cond[idx, ydim]
        if nd >= 3:
            fdim = order[2]
            masks, labels = _facet_bins(cond[idx, fdim], max_facets)
        else:
            fdim = None
            masks, labels = [np.ones(len(idx), dtype=bool)], [None]
        ncols = len(masks)
        fig, axes = plt.subplots(
            1, ncols, figsize=(4.5 * ncols, 4.0), squeeze=False,
            sharex=True, sharey=True, layout="constrained",
        )
        sc = None
        for ax, bmask, blab in zip(axes[0], masks, labels):
            mm = bmask & finite
            sc = ax.scatter(x[mm], y[mm], c=vals[mm], s=sizes[mm], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel(_COND_NAMES[xdim])
            ax.grid(True, alpha=0.3)
            if blab is not None:
                ax.set_title(f"{_COND_NAMES[fdim]}≈{blab}")
        axes[0][0].set_ylabel(_COND_NAMES[ydim])
        if sc is not None:
            fig.colorbar(sc, ax=axes[0].tolist(), label=label, fraction=0.046, pad=0.02)
        title = f"{sk}: {label}  (x={_COND_NAMES[xdim]}, y={_COND_NAMES[ydim]}"
        if fdim is not None:
            title += f", facet={_COND_NAMES[fdim]}"
        extra = [_COND_NAMES[d] for d in order[3:]]
        if extra:
            title += f"; marginalized: {', '.join(extra)}"
        title += ")"
        fig.suptitle(title)

    fname = f"per_traj_{sk}__{label}.png".replace("/", "_")
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    """Format a float, rendering NaN as a dash."""
    return "  -   " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _row(vals, widths) -> str:
    """Left-align ``vals`` into fixed-width columns with two-space gutters."""
    cells = []
    for v, w in zip(vals, widths):
        s = str(v)
        cells.append(f"{s:<{w}}")
    return "  " + "  ".join(cells)


def _wmean(values, weights, mask) -> float:
    """Sample-weighted mean of ``values[mask]`` weighted by ``weights[mask]``."""
    v = values[mask]
    w = weights[mask].astype(np.float64)
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def _safe_div(num, den) -> float:
    """Divide, returning NaN when the denominator is zero."""
    return float(num) / float(den) if den else np.nan


def _unique_preserve(arr) -> list:
    """Unique values in first-seen order."""
    seen, out = set(), []
    for x in arr.tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
