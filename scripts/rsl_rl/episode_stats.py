# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play-time accumulator for training-comparable episode metrics.

During play the environment still publishes the same aggregate episode metrics
it does during training in ``extras["log"]`` — the dict the RSL-RL logger reads
to populate the ``Episode_Termination/*``, ``Episode_Reward/*``, ``Metrics/*``
and ``Curriculum/*`` series in tensorboard/wandb. ``play.py`` normally discards
that dict; this accumulator folds it in over the run so the resulting numbers
line up with where the training curves settle.

Two aggregation modes, by metric kind:

* **Termination terms** are tallied by *direct per-episode-end counts*:
  ``(# episode-ends via term) / (# episode-ends)``. This is the unbiased
  fraction of episodes ending via each cause, and the primary (non-subset)
  causes sum to ~1.0. We do NOT step-mean ``Episode_Termination/*`` from
  ``extras["log"]`` because that quantity (``_last_episode_dones`` meaned over
  envs) is biased low at play time: every env starts a fresh episode with an
  all-zero row, so averaging from step 0 includes a cold-start window that
  training — always "warm" — never sees. Counting real end-events sidesteps
  that and recovers the same value training settles at.

* **All other scalars** (rewards, command metrics, curriculum) are step-meaned,
  matching RSL-RL's per-iteration aggregation (``rsl_rl/utils/logger.py``): the
  env rewrites ``extras["log"]`` only on reset steps and leaves it stale
  otherwise, and the logger means each key over *every* collected step (stale
  snapshots included). We fold ``extras["log"]`` in on every step likewise.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch

_TERM_PREFIX = "Episode_Termination/"


class EpisodeStatsAccumulator:
    """Accumulate the env's ``extras["log"]`` scalars over a play run.

    Usage::

        ep_stats = EpisodeStatsAccumulator()
        for step in range(N):
            obs, _, dones, infos = env.step(actions)
            tm = env.unwrapped.termination_manager
            term_fired = {n: tm.get_term(n) for n in tm.active_terms}
            ep_stats.collect_step(infos, dones, term_fired=term_fired)
        ep_stats.write_report(path, num_envs=env.unwrapped.num_envs)
    """

    def __init__(self) -> None:
        """Initialize empty running sums/counts."""
        # Step-mean accumulators for non-termination scalars.
        self._sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        # Direct per-episode-end termination fire counts.
        self._term_count: dict[str, int] = defaultdict(int)
        self._have_term_counts: bool = False
        self._steps: int = 0
        self._episode_ends: int = 0

    def collect_step(
        self,
        extras: dict[str, Any],
        dones: torch.Tensor | None = None,
        term_fired: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Fold one step's metrics into the running aggregates.

        Args:
            extras: The info dict returned as the 4th value of the wrapped
                ``env.step`` (the env's ``extras``). Read-only here. Steps before
                the first reset (no ``"log"`` key) are counted but contribute no
                scalars.
            dones: Optional per-env done flags for this step, used to count
                completed episodes (the denominator for termination fractions).
            term_fired: Optional ``{term_name: bool tensor [num_envs]}`` of which
                envs ended via each termination term this step (read from the
                termination manager). When supplied, termination terms are
                tallied as direct per-episode-end counts and the corresponding
                ``Episode_Termination/*`` keys in ``extras["log"]`` are skipped.
                When omitted, those keys fall back to the (cold-start-biased)
                step-mean.
        """
        self._steps += 1
        if dones is not None:
            self._episode_ends += int(dones.sum().item())
        if term_fired is not None:
            self._have_term_counts = True
            for name, fired in term_fired.items():
                self._term_count[name] += int(fired.sum().item())
        log = extras.get("log") if isinstance(extras, dict) else None
        if not log:
            return
        for key, val in log.items():
            if self._have_term_counts and key.startswith(_TERM_PREFIX):
                continue  # reported via direct counts instead
            if isinstance(val, torch.Tensor):
                if val.numel() != 1:
                    continue
                val = float(val.item())
            else:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
            self._sum[key] += val
            self._count[key] += 1

    @property
    def num_steps(self) -> int:
        """Number of steps folded in so far."""
        return self._steps

    def means(self) -> dict[str, float]:
        """Per-key step-mean of every captured non-termination scalar."""
        return {
            k: self._sum[k] / self._count[k]
            for k in sorted(self._sum)
            if self._count[k] > 0
        }

    def termination_fractions(self) -> dict[str, tuple[float, int]]:
        """Per-term ``(fraction_of_episode_ends, fire_count)``.

        The fraction denominator is the total number of episode-ends seen; with
        no ends yet it is treated as 0.0 fraction. Primary (non-subset) causes
        sum to ~1.0; subset/breakdown terms (e.g. ``*_post_grace``, per-skill)
        double-count their parent.
        """
        denom = self._episode_ends if self._episode_ends > 0 else 0
        out: dict[str, tuple[float, int]] = {}
        for name in sorted(self._term_count):
            cnt = self._term_count[name]
            frac = (cnt / denom) if denom > 0 else 0.0
            out[name] = (frac, cnt)
        return out

    def format_report(
        self, num_envs: int | None = None, title: str = "EPISODE STATS (play)"
    ) -> str:
        """Render a grouped, human-readable summary table.

        Termination terms are shown as ``fraction (count)`` of episode-ends;
        every other scalar is shown as its step-mean. Non-termination keys are
        grouped by their top-level prefix (the part before the first ``/``).

        Args:
            num_envs: Number of parallel envs, shown in the header for context.
            title: Heading for the report.

        Returns:
            The formatted multi-line report (trailing newline included).
        """
        lines = [f"=== {title} ==="]
        header = f"steps: {self._steps}   episode-ends: {self._episode_ends}"
        if num_envs is not None:
            header += f"   envs: {num_envs}"
        lines.append(header)

        term_fracs = self.termination_fractions()
        means = self.means()

        if not term_fracs and not means:
            lines.append("(no extras['log'] captured — did any episode reset?)")
            return "\n".join(lines) + "\n"

        # Width spans every label in the report so columns line up across groups.
        label_pool = [name for name in term_fracs]
        grouped_means: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for key, val in means.items():
            group, sep, rest = key.partition("/")
            if sep:
                grouped_means[group].append((rest, val))
                label_pool.append(rest)
            else:
                grouped_means["(misc)"].append((key, val))
                label_pool.append(key)
        width = max((len(lbl) for lbl in label_pool), default=0)

        # Termination block (direct per-episode-end fractions).
        if term_fracs:
            lines.append("")
            lines.append(f"{_TERM_PREFIX}  (fraction of episode-ends [count])")
            lines.append(
                "  note: primary causes sum to ~1.0; subset/breakdown terms "
                "(*_post_grace, per-skill) double-count their parent."
            )
            for name, (frac, cnt) in term_fracs.items():
                lines.append(f"  {name:<{width}}  {frac: .4f}  [{cnt}]")

        # Step-mean blocks for everything else.
        for group in sorted(grouped_means):
            lines.append("")
            lines.append(f"{group}/  (step-mean)")
            for label, val in sorted(grouped_means[group]):
                lines.append(f"  {label:<{width}}  {val: .4f}")
        return "\n".join(lines) + "\n"

    def write_report(self, out_path: str, num_envs: int | None = None) -> None:
        """Write the formatted report to ``out_path`` (creating its dir)."""
        report = self.format_report(num_envs=num_envs)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(report)
