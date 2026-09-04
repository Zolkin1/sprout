# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline metrics for the autoregressive planner.

Four quantities, and the second and fourth are here for the RL stage rather than for the
supervised one:

1. **cross-entropy and top-1 accuracy** -- how well the tokens are predicted at all.
2. **per-position predictive entropy**, as a curve over the ``N`` positions. This is the
   EXPLORATION-HEALTH metric: the token distribution at supervised convergence is the
   distribution PPO will explore with, and a planner annealed to a near-deterministic argmax
   has nothing to explore. It is recorded in the checkpoint for that reason, not as a
   diagnostic. Expect it to fall with position -- the first token is chosen under the most
   uncertainty about the whole plan.
3. **trajectory error** of a greedily decoded plan against the true future, in physical
   units and per terrain family. Reuses :class:`~robot_rl.token_planner.metrics.ReconAccumulator`,
   so it is the same number the tokenizer's own report gives and the two are subtractable:
   the difference between them is what the PLANNER adds to what quantization already cost.
4. **sample diversity** -- mean pairwise distance between plans drawn from one context. It has
   to be clearly non-zero (a deterministic planner cannot be fine-tuned by policy gradient)
   and should be LARGER where the context is genuinely ambiguous.

Nothing here imports IsaacLab at module scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class TokenAccumulator:
    """Streaming cross-entropy, accuracy and per-position entropy over a split.

    Args:
        num_tokens: Output positions ``N``.
        vocab_size: Vocabulary ``V``, for the entropy ceiling and the utilisation figure.
    """

    num_tokens: int
    vocab_size: int

    _ce: torch.Tensor = field(init=False)          # (N,) summed CE per position
    _correct: torch.Tensor = field(init=False)     # (N,) top-1 hits
    _ent: torch.Tensor = field(init=False)         # (N,) summed predictive entropy
    _n: int = field(init=False, default=0)
    _pred_counts: torch.Tensor = field(init=False)  # (V,) argmax histogram

    def __post_init__(self):
        n = int(self.num_tokens)
        self._ce = torch.zeros(n, dtype=torch.float64)
        self._correct = torch.zeros(n, dtype=torch.float64)
        self._ent = torch.zeros(n, dtype=torch.float64)
        self._pred_counts = torch.zeros(int(self.vocab_size), dtype=torch.int64)
        self._n = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Fold one teacher-forced batch in.

        Args:
            logits: ``(B, N, V)`` predictive logits.
            target: ``(B, N)`` true token ids.
        """
        logp = F.log_softmax(logits.float(), dim=-1)
        # UNSMOOTHED cross-entropy, whatever the training loss used. Label smoothing adds a
        # constant floor that depends on V, so a smoothed CE is not comparable across
        # vocabulary sizes -- and the level-table ablation changes V.
        self._ce += (-logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)).sum(0).double().cpu()
        pred = logp.argmax(-1)
        self._correct += (pred == target).sum(0).double().cpu()
        self._ent += (-(logp.exp() * logp).sum(-1)).sum(0).double().cpu()
        self._pred_counts += torch.bincount(
            pred.reshape(-1).cpu(), minlength=int(self.vocab_size))
        self._n += int(target.shape[0])

    def report(self) -> dict[str, Any]:
        """Everything, as a JSON-able dict."""
        n = max(self._n, 1)
        ce = self._ce / n
        ent = self._ent / n
        acc = self._correct / n
        counts = self._pred_counts.double()
        total = float(counts.sum())
        p = counts / max(total, 1.0)
        nz = p[p > 0]
        return {
            "samples": self._n,
            "ce": float(ce.mean()),
            "ce_by_position": [float(v) for v in ce],
            "top1": float(acc.mean()),
            "top1_by_position": [float(v) for v in acc],
            "entropy_nats": float(ent.mean()),
            "entropy_bits": float(ent.mean() / math.log(2.0)),
            "entropy_by_position_nats": [float(v) for v in ent],
            # As a fraction of a uniform distribution's entropy: 1.0 means the model is
            # saying nothing, 0.0 means it is deterministic. Neither end is what RL wants.
            "entropy_frac_of_uniform": float(ent.mean() / math.log(float(self.vocab_size))),
            "argmax_distinct": int((counts > 0).sum()),
            "argmax_utilisation": float((counts > 0).double().mean()),
            "argmax_perplexity": float(math.exp(float(-(nz * nz.log()).sum())))
            if nz.numel() else 0.0,
        }


@torch.no_grad()
def sample_diversity(
    planner,
    history: torch.Tensor,
    extero: torch.Tensor,
    cmd_tokens: torch.Tensor | None,
    num_samples: int = 16,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    std: torch.Tensor | None = None,
) -> dict[str, float]:
    """Mean pairwise distance between plans drawn from the SAME context.

    The number RL depends on. A planner whose diversity is ~0 is deterministic given its
    conditioning, so a policy gradient has nothing to push on; one whose diversity is huge is
    not committing to a plan at all. Reported in physical units when ``std`` is given, since a
    normalized distance is not a quantity anyone can judge.

    Both a TOKEN distance and a TRAJECTORY distance are returned, and they can disagree
    informatively: many distinct token sequences that decode to nearly the same motion means
    the vocabulary is redundant, which is harmless for behaviour and bad for credit
    assignment.

    Args:
        planner: A :class:`~robot_rl.token_planner.planner.TokenPlanner`.
        history: ``(B, N_hist, history_dim)`` conditioning.
        extero: ``(B, T, extero_sensor_dim)`` conditioning.
        cmd_tokens: ``(B, num_cmd_tokens, cmd_dim)`` or ``None``.
        num_samples: Draws per context.
        temperature: Sampling temperature.
        generator: Explicit RNG, on the same device as the conditioning.
        std: ``(node_dim,)`` normalizer std, to report the trajectory distance in physical
            units. ``None`` reports it in normalized units.

    Returns:
        ``{"traj_rmsd", "token_hamming", "distinct_plans"}``. ``traj_rmsd`` is the mean over
        contexts of the mean pairwise RMS node distance; ``token_hamming`` the mean pairwise
        fraction of differing tokens; ``distinct_plans`` the mean number of distinct token
        sequences among the draws.
    """
    if planner.training:
        raise RuntimeError(
            "sample_diversity() needs the planner in eval mode -- dropout would make each "
            "draw come from a different network, so the spread measured would be the "
            "dropout's rather than the policy's.")
    b = history.shape[0]
    plans, toks = [], []
    for _ in range(int(num_samples)):
        t, _, _ = planner.sample(history, extero, cmd_tokens, temperature=temperature,
                                 generator=generator)
        toks.append(t)
        plans.append(planner.decode(t))                       # (B, H, node_dim), normalized
    tok = torch.stack(toks, 1)                                # (B, S, N)
    traj = torch.stack(plans, 1)                              # (B, S, H, D)
    if std is not None:
        traj = traj * std.to(traj.device).view(1, 1, 1, -1)

    s = tok.shape[1]
    iu = torch.triu_indices(s, s, offset=1)
    if iu.numel() == 0:
        return {"traj_rmsd": 0.0, "token_hamming": 0.0, "distinct_plans": 1.0}
    a, c = iu[0], iu[1]
    # Pairwise, over the upper triangle only -- the full matrix double-counts and includes
    # the zero diagonal, which would drag every mean towards zero by 1/S.
    d_traj = (traj[:, a] - traj[:, c]).pow(2).mean(dim=(-1, -2)).sqrt()   # (B, P)
    d_tok = (tok[:, a] != tok[:, c]).float().mean(-1)                     # (B, P)
    distinct = torch.tensor(
        [len({tuple(row.tolist()) for row in tok[i]}) for i in range(b)], dtype=torch.float64)
    return {
        "traj_rmsd": float(d_traj.mean()),
        "token_hamming": float(d_tok.mean()),
        "distinct_plans": float(distinct.mean()),
    }


def format_report(tok: dict[str, Any], div: dict[str, float] | None = None) -> str:
    """Render a token report (and optionally a diversity report) as text.

    Args:
        tok: A :meth:`TokenAccumulator.report`.
        div: A :func:`sample_diversity` result, or ``None``.

    Returns:
        A multi-line string.
    """
    lines = [
        f"samples: {tok['samples']}",
        f"cross-entropy {tok['ce']:.4f} nats   top-1 {100 * tok['top1']:.2f}%",
        f"predictive entropy {tok['entropy_nats']:.4f} nats "
        f"({tok['entropy_bits']:.2f} bits, "
        f"{100 * tok['entropy_frac_of_uniform']:.1f}% of uniform)",
        f"argmax spread: {tok['argmax_distinct']} distinct ids "
        f"({100 * tok['argmax_utilisation']:.1f}%), perplexity "
        f"{tok['argmax_perplexity']:.0f}",
        "",
        "per position:  " + "  ".join(f"{i:>6d}" for i in range(len(tok['ce_by_position']))),
        "  CE           " + "  ".join(f"{v:>6.3f}" for v in tok["ce_by_position"]),
        "  top-1 %      " + "  ".join(f"{100 * v:>6.1f}" for v in tok["top1_by_position"]),
        "  entropy      " + "  ".join(f"{v:>6.3f}"
                                      for v in tok["entropy_by_position_nats"]),
    ]
    if div is not None:
        lines += [
            "",
            f"diversity at T=1: trajectory RMSD {div['traj_rmsd']:.4f}, "
            f"token hamming {div['token_hamming']:.3f}, "
            f"{div['distinct_plans']:.1f} distinct plans per context",
        ]
    return "\n".join(lines)
