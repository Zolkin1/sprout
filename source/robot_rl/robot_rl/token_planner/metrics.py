# Copyright 2026 Zachary Olkin. All rights reserved.

"""Reconstruction and codebook metrics for the trajectory tokenizer.

Everything reported here is in PHYSICAL units. The autoencoder trains in normalized node
space, and a normalized MSE is not a quantity anyone can gate on -- the state normalizer
divides base position by a declared 0.4 m and each joint by its own fitted std, so one
normalized number is metres on three channels, radians on twenty-nine, and dimensionless on
six. Since a mean cancels in a difference, converting is exactly ``err * std``.

The channel groups are the ones :data:`~robot_rl.flow_matching.window.REFERENCE_BLOCK_KEYS`
already names, and they are obtained by expanding a group ID through
:func:`~robot_rl.flow_matching.window.expand_window_blocks` rather than by restating the
layout's widths. That matters more than it looks: ``base_pose`` is one block but two groups
in two different unit systems (3 position channels in metres, 6 rot6d channels that are
dimensionless), and the widths ``9 / 6 / nj / nj / 3K`` have to agree with the packing in
five other places.

Nothing here imports IsaacLab at module scope -- the one function that needs the layout
imports it lazily -- so the accumulator and its reductions are usable in ``env_transfer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


def channel_group_labels(
    num_joints: int, num_ee_frames: int, blocks: Sequence[str] | None = None
) -> tuple[list[str], torch.Tensor]:
    """Per-channel group labels for one window node.

    Args:
        num_joints: Robot joint count ``nj``.
        num_ee_frames: End-effector frame count ``K``.
        blocks: The node's block selection, or ``None`` for all five.

    Returns:
        ``(group_names, labels)`` with ``labels`` an ``int64`` ``(D,)`` index into
        ``group_names``. Only groups the selection actually contains appear.
    """
    from robot_rl.flow_matching.window import (  # noqa: PLC0415  (keeps this module Isaac-free)
        REFERENCE_BLOCK_KEYS,
        expand_window_blocks,
        resolve_reference_blocks,
    )

    selected = resolve_reference_blocks(blocks)
    keys: list[str] = []
    for block in selected:
        keys.extend(REFERENCE_BLOCK_KEYS[block])
    per_block = {k: float(i) for i, k in enumerate(keys)}
    expanded = expand_window_blocks(num_joints, num_ee_frames, per_block, selected)
    return keys, torch.tensor([int(v) for v in expanded], dtype=torch.long)


#: Physical unit of each group, for the report. Descriptive only.
GROUP_UNITS: dict[str, str] = {
    "base_pos": "m",
    "base_ori": "-",          # two orthonormal rot6d columns; dimensionless by construction
    "base_lin_vel": "m/s",
    "base_ang_vel": "rad/s",
    "joint_pos": "rad",
    "joint_vel": "rad/s",
    "ee_pos": "m",
}


def _geodesic_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between two 6-D rotation representations, in radians.

    Args:
        a: ``(..., 6)`` rot6d.
        b: ``(..., 6)`` rot6d.

    Returns:
        ``(...)`` angles.
    """
    from robot_rl.flow_matching.transforms import rot6d_to_matrix  # noqa: PLC0415

    rel = rot6d_to_matrix(a).transpose(-1, -2) @ rot6d_to_matrix(b)
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.arccos(cos)


@dataclass
class ReconAccumulator:
    """Streaming reconstruction + codebook statistics over a whole split.

    Accumulates in float64 so a hundred thousand chunks cannot drift the mean, and keeps the
    per-node group errors so the 99th percentile is EXACT rather than an estimate off a
    histogram. The spec asks for that percentile specifically because a tokenizer with a good
    mean and a bad tail is the failure the tracker would feel: it follows an average
    trajectory fine and falls on the one node that jumped.

    Args:
        group_names: Group names, from :func:`channel_group_labels`.
        labels: ``(D,)`` group label per channel.
        mean: ``(D,)`` normalizer mean. Only the geodesic needs it (an error cancels the
            mean, a rotation matrix does not).
        std: ``(D,)`` normalizer std, for the conversion to physical units.
        chunk_len: Nodes per chunk ``H``, for the per-position curve.
        vocab_size: Token vocabulary ``V``, for the utilisation histogram.
        num_families: Family code count, for the by-terrain split.
        family_names: Family names for the report.
    """

    group_names: list[str]
    labels: torch.Tensor
    std: torch.Tensor
    mean: torch.Tensor
    chunk_len: int
    vocab_size: int
    num_families: int
    family_names: list[str]

    _sq_by_group: torch.Tensor = field(init=False)          # (G,) sum of squared phys err
    _n_by_group: torch.Tensor = field(init=False)           # (G,) channel-node count
    _sq_by_pos: torch.Tensor = field(init=False)            # (H, G)
    _n_by_pos: torch.Tensor = field(init=False)             # (H, G)
    _sq_by_family: torch.Tensor = field(init=False)         # (F, G)
    _n_by_family: torch.Tensor = field(init=False)          # (F, G)
    _token_counts: torch.Tensor = field(init=False)         # (V,)
    _node_err: list[torch.Tensor] = field(init=False)       # per-batch (B*H, G) node RMS
    _node_family: list[torch.Tensor] = field(init=False)    # per-batch (B*H,) family code
    _geo_sum: float = field(init=False, default=0.0)         # sum of geodesic angles (rad)
    _geo_n: int = field(init=False, default=0)
    _chunks: int = field(init=False, default=0)

    def __post_init__(self):
        g = len(self.group_names)
        self._sq_by_group = torch.zeros(g, dtype=torch.float64)
        self._n_by_group = torch.zeros(g, dtype=torch.float64)
        self._sq_by_pos = torch.zeros(self.chunk_len, g, dtype=torch.float64)
        self._n_by_pos = torch.zeros(self.chunk_len, g, dtype=torch.float64)
        self._sq_by_family = torch.zeros(self.num_families, g, dtype=torch.float64)
        self._n_by_family = torch.zeros(self.num_families, g, dtype=torch.float64)
        self._token_counts = torch.zeros(self.vocab_size, dtype=torch.int64)
        self._node_err = []
        self._node_family = []
        self._geo_sum = 0.0
        self._geo_n = 0
        self._chunks = 0

    @torch.no_grad()
    def update(
        self, target: torch.Tensor, recon: torch.Tensor,
        ids: torch.Tensor, family: torch.Tensor,
    ) -> None:
        """Fold one batch in.

        Args:
            target: Normalized ground-truth chunks ``(B, H, D)``.
            recon: Normalized reconstructions ``(B, H, D)``.
            ids: Token ids ``(B, N)``.
            family: Family codes ``(B,)``.
        """
        dev = target.device
        labels = self.labels.to(dev)
        std = self.std.to(dev)
        err = (recon - target).double() * std.double()             # (B, H, D) physical
        sq = err.pow(2)
        g = len(self.group_names)
        onehot = torch.zeros(g, target.shape[-1], dtype=torch.float64, device=dev)
        onehot[labels, torch.arange(target.shape[-1], device=dev)] = 1.0
        # (B, H, G): squared error summed over each group's channels, and the channel count.
        sq_g = sq @ onehot.T
        cnt_g = onehot.sum(dim=1)                                   # (G,)

        self._sq_by_group += sq_g.sum(dim=(0, 1)).cpu()
        self._n_by_group += (cnt_g * target.shape[0] * target.shape[1]).cpu()
        self._sq_by_pos += sq_g.sum(dim=0).cpu()
        self._n_by_pos += (cnt_g.unsqueeze(0) * target.shape[0]).expand(
            self.chunk_len, g).cpu()

        fam = family.to(dev).long().clamp_(0, self.num_families - 1)
        per_chunk = sq_g.sum(dim=1)                                 # (B, G)
        self._sq_by_family.index_add_(0, fam.cpu(), per_chunk.cpu())
        self._n_by_family.index_add_(
            0, fam.cpu(),
            (cnt_g.unsqueeze(0) * target.shape[1]).expand(target.shape[0], g).cpu(),
        )

        # Per-node group RMS, kept for the exact tail.
        node_rms = (sq_g / cnt_g.clamp(min=1.0)).sqrt()             # (B, H, G)
        self._node_err.append(node_rms.reshape(-1, g).float().cpu())
        self._node_family.append(
            fam.cpu().unsqueeze(1).expand(-1, target.shape[1]).reshape(-1))

        self._token_counts += torch.bincount(
            ids.reshape(-1).cpu(), minlength=self.vocab_size)
        self._chunks += int(target.shape[0])

        # Orientation, as an ANGLE. A Frobenius RMSE on the rot6d channels is a fine loss but
        # is not a quantity anyone can compare against anything -- the generator reports
        # ``ori_geodesic_rad``, and matching that definition is the only way the tokenizer's
        # information loss can be read against the generator's own prediction error. Going
        # through the rotation group also projects out the Gram-Schmidt residual rather than
        # counting it as orientation error.
        if "base_ori" in self.group_names:
            col = self.labels.to(dev) == self.group_names.index("base_ori")
            m, sd = self.mean.to(dev), std
            self._geo_sum += float(_geodesic_rad(
                (recon * sd + m)[..., col], (target * sd + m)[..., col]).sum())
            self._geo_n += int(target.shape[0] * target.shape[1])

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def ori_geodesic_rad(self) -> float:
        """Mean angle between the reconstructed and true base orientations, in radians.

        The same definition ``scripts/flow_matching/train.py``'s ``channel_metrics`` reports
        as ``ori_geodesic_rad``, so the two are directly comparable.
        """
        return self._geo_sum / max(self._geo_n, 1)

    def group_rmse(self) -> dict[str, float]:
        """Physical RMSE per channel group, over the whole split."""
        rmse = (self._sq_by_group / self._n_by_group.clamp(min=1.0)).sqrt()
        return {n: float(v) for n, v in zip(self.group_names, rmse)}

    def group_rmse_by_family(self) -> dict[str, dict[str, float]]:
        """Physical RMSE per group, split by terrain family. Empty families are omitted."""
        rmse = (self._sq_by_family / self._n_by_family.clamp(min=1.0)).sqrt()
        out: dict[str, dict[str, float]] = {}
        for f in range(self.num_families):
            if float(self._n_by_family[f].sum()) == 0.0:
                continue
            name = (self.family_names[f] if f < len(self.family_names) else f"family_{f}")
            out[name] = {n: float(v) for n, v in zip(self.group_names, rmse[f])}
        return out

    def rmse_by_position(self) -> dict[str, list[float]]:
        """Physical RMSE per group at each of the ``H`` chunk positions.

        The curve that catches boundary degradation: a chunk-local autoencoder pads at the
        ends, so if the padding mode is wrong the first and last nodes reconstruct worse than
        the interior and nothing else in the report would say so.
        """
        rmse = (self._sq_by_pos / self._n_by_pos.clamp(min=1.0)).sqrt()
        return {n: [float(v) for v in rmse[:, i]] for i, n in enumerate(self.group_names)}

    def boundary_penalty(self, edge: int = 4) -> dict[str, dict[str, float]]:
        """Edge-vs-interior RMSE ratio per group, reported for the two ends SEPARATELY.

        ``1.0`` means that end reconstructs as well as the middle, which is the property the
        padding mode is chosen for; the padding is what makes the first and last nodes hard,
        so this is the number that says whether ``replicate`` is doing its job.

        **Pooling the two ends is confounded and was, briefly.** Several groups' error grows
        monotonically with look-ahead -- ``base_pos`` most of all, since node 0 is the
        robot's own pose (a near-zero residual) and node ``H-1`` is up to a couple of metres
        of travel away. A single pooled ratio mixes a low-signal end with a high-signal one
        and can read near 1.0 while both ends are wrong in opposite directions. Split, the
        confound is legible instead: a padding defect raises BOTH ends, while the
        signal-growth effect raises ``hi`` and lowers ``lo``.

        The interior slope is reported alongside for exactly that reason -- it is the size of
        the effect the ratios have to be read against.

        Args:
            edge: Nodes at each end counted as the boundary.

        Returns:
            Group name -> ``{"lo", "hi", "interior_slope"}``. The ratios are ``inf`` if the
            interior mean is exactly zero; ``interior_slope`` is the ratio of the interior's
            last-quarter mean to its first-quarter mean, i.e. how much the error grows with
            look-ahead on its own.
        """
        curve = self.rmse_by_position()
        out: dict[str, dict[str, float]] = {}
        for name, vals in curve.items():
            v = torch.tensor(vals, dtype=torch.float64)
            inner = v[edge:-edge] if v.numel() > 2 * edge else v
            mid = float(inner.mean())
            q = max(inner.numel() // 4, 1)
            lo_q, hi_q = float(inner[:q].mean()), float(inner[-q:].mean())
            out[name] = {
                "lo": float(v[:edge].mean() / mid) if mid > 0.0 else math.inf,
                "hi": float(v[-edge:].mean() / mid) if mid > 0.0 else math.inf,
                "interior_slope": (hi_q / lo_q) if lo_q > 0.0 else math.inf,
            }
        return out

    def node_error_quantile(self, q: float = 0.99) -> dict[str, float]:
        """Exact ``q``-quantile of the per-node group RMSE, in physical units.

        Args:
            q: The quantile.

        Returns:
            Group name -> quantile value.
        """
        if not self._node_err:
            return {n: 0.0 for n in self.group_names}
        allv = torch.cat(self._node_err, dim=0)
        return {
            n: float(torch.quantile(allv[:, i].double(), q))
            for i, n in enumerate(self.group_names)
        }

    def node_error_max(self) -> dict[str, float]:
        """Largest per-node group RMSE seen, in physical units."""
        if not self._node_err:
            return {n: 0.0 for n in self.group_names}
        allv = torch.cat(self._node_err, dim=0)
        return {n: float(allv[:, i].max()) for i, n in enumerate(self.group_names)}

    def codebook(self) -> dict[str, float]:
        """Codebook health: utilisation, entropy and perplexity of the token distribution.

        ``utilisation`` is the fraction of ``V`` ids that appeared at all;
        ``entropy_bits`` is the empirical Shannon entropy and ``entropy_frac`` is it as a
        fraction of ``log2(V)``, i.e. 1.0 for a uniform token distribution.
        ``perplexity`` is ``2 ** entropy_bits`` -- the number of tokens the model is
        EFFECTIVELY using, which is the more honest figure when a few ids carry most of the
        mass and the utilisation number looks healthy.
        """
        counts = self._token_counts.double()
        total = float(counts.sum())
        if total <= 0.0:
            return {"utilisation": 0.0, "entropy_bits": 0.0, "entropy_frac": 0.0,
                    "perplexity": 0.0, "distinct": 0.0, "vocab_size": float(self.vocab_size)}
        p = counts / total
        nz = p[p > 0]
        ent = float(-(nz * nz.log2()).sum())
        return {
            "utilisation": float((counts > 0).double().mean()),
            "distinct": float((counts > 0).sum()),
            "vocab_size": float(self.vocab_size),
            "entropy_bits": ent,
            "entropy_frac": ent / math.log2(self.vocab_size),
            "perplexity": float(2.0 ** ent),
        }

    def report(self) -> dict[str, Any]:
        """Everything, as a JSON-able dict."""
        return {
            "chunks": self._chunks,
            "group_units": {n: GROUP_UNITS.get(n, "?") for n in self.group_names},
            "rmse": self.group_rmse(),
            "rmse_by_family": self.group_rmse_by_family(),
            "rmse_by_position": self.rmse_by_position(),
            "boundary_penalty": self.boundary_penalty(),
            "node_rmse_p99": self.node_error_quantile(0.99),
            "node_rmse_max": self.node_error_max(),
            "codebook": self.codebook(),
            "ori_geodesic_rad": self.ori_geodesic_rad(),
        }


def format_report(rep: dict[str, Any], max_families: int = 12) -> str:
    """Render :meth:`ReconAccumulator.report` as the table the milestone asks for.

    Args:
        rep: A report dict.
        max_families: Cap on family rows printed.

    Returns:
        A multi-line string.
    """
    groups = list(rep["rmse"].keys())
    units = rep["group_units"]
    head = "  ".join(f"{g}[{units.get(g, '?')}]" for g in groups)
    lines = [f"chunks: {rep['chunks']}", "", f"{'split':<26}{head}"]

    def row(name: str, vals: dict[str, float]) -> str:
        return f"{name:<26}" + "  ".join(
            f"{vals.get(g, float('nan')):>{max(len(g) + len(units.get(g, '?')) + 2, 9)}.5f}"
            for g in groups
        )

    lines.append(row("overall", rep["rmse"]))
    for i, (fam, vals) in enumerate(sorted(rep["rmse_by_family"].items())):
        if i >= max_families:
            lines.append(f"  ... {len(rep['rmse_by_family']) - max_families} more families")
            break
        lines.append(row(fam, vals))
    bp = rep["boundary_penalty"]
    lines += [
        "",
        row("node p99", rep["node_rmse_p99"]),
        row("node max", rep["node_rmse_max"]),
        row("edge_lo/interior", {g: bp[g]["lo"] for g in groups if g in bp}),
        row("edge_hi/interior", {g: bp[g]["hi"] for g in groups if g in bp}),
        row("interior slope", {g: bp[g]["interior_slope"] for g in groups if g in bp}),
        "",
    ]
    lines.append(f"base orientation geodesic: {rep.get('ori_geodesic_rad', 0.0):.5f} rad "
                 f"({math.degrees(rep.get('ori_geodesic_rad', 0.0)):.2f} deg)")
    cb = rep["codebook"]
    lines.append(
        f"codebook: {int(cb['distinct'])}/{int(cb['vocab_size'])} ids used "
        f"({100 * cb['utilisation']:.1f}%), entropy {cb['entropy_bits']:.2f} bits "
        f"({100 * cb['entropy_frac']:.1f}% of uniform), perplexity {cb['perplexity']:.0f}"
    )
    return "\n".join(lines)
