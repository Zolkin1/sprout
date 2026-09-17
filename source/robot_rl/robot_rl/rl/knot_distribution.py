# Copyright 2026 Zachary Olkin. All rights reserved.

"""A Gaussian whose std is 18 numbers instead of 572, factored group x knot.

The adjuster's exploration width was a per-dimension parameter, and at full width that meant
2728 of them. Measured over 850 iterations of run v6 they went nowhere: mean 0.150 -> 0.1478,
spread across dimensions 0.0025, every channel group within 1% of every other. The reason is
sample count, not gradient direction -- with ~3 samples per dimension per iteration the
second-moment statistic that moves a per-dimension std has nothing to average.

So the std is factored:

    log sigma[g, j] = log s[g] + log h[j]

with ``g`` one of five CHANNEL GROUPS (base position, base rot6d, base lin vel, base ang vel,
joint positions) and ``j`` one of ``K`` KNOTS. At ``K = 13`` that is 18 learned numbers
sharing all 8192-per-iteration samples between them, ~450x more samples each than the
per-dimension version got.

Two things it buys beyond sample count:

* the groups are the units the residual is CAPPED in, and the offline correction fit says
  they need different widths -- the needed correction is 0.19-0.52 of the cap depending on
  the group -- so a single scalar was never right;
* the knot factor can learn to explore less on knots that execute before the next re-plan
  and more further out, which is the shape the generator's own error has (it grows ~5x from
  node 0 to node 50).

**The gauge is pinned.** ``log s + log h`` is over-parameterised: adding ``c`` to every
``log h[j]`` and subtracting it from every ``log s[g]`` leaves sigma unchanged, so the two
factors would drift together and neither would be readable. ``log h`` is therefore
mean-centred inside the property, which makes ``s[g]`` the group's geometric-mean width and
``h[j]`` a pure per-knot deviation about 1.0. Both are then meaningful to log, which is what
the diagnostics want.

**It presents itself as a plain 572-wide Gaussian.** ``log_std_param`` is a PROPERTY that
expands the factors, so ``GaussianDistribution.update`` samples unchanged and ``ChunkPPO``'s
layout assertion -- which reads that attribute's width to check ``K * D`` -- still sees the
action's true width rather than 18.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import Distribution, GaussianDistribution

#: Channel-group names, in the window's own packing order. Not a hand-written list: they are
#: the sub-keys of :data:`~robot_rl.flow_matching.window.REFERENCE_BLOCK_KEYS` for whichever
#: parent blocks the window carries, so a layout change moves them.
def group_names(blocks: tuple[str, ...]) -> list[str]:
    """Channel-group names for ``blocks``, in packing order.

    Args:
        blocks: Parent block selection the window is packed with.

    Returns:
        The sub-key names, e.g. ``["base_pos", "base_ori", ...]``.
    """
    from robot_rl.flow_matching.window import REFERENCE_BLOCK_KEYS  # noqa: PLC0415

    return [k for parent in blocks for k in REFERENCE_BLOCK_KEYS[parent]]


def channel_groups(num_joints: int, num_ee_frames: int,
                   blocks: tuple[str, ...]) -> tuple[list[int], list[str]]:
    """Per-channel group index for ONE window node, plus the group names.

    Built through :func:`~robot_rl.flow_matching.window.expand_window_blocks`, the same
    producer the observation's noise and the residual's cap table go through, so the grouping
    cannot come to disagree with the packing it describes.

    Args:
        num_joints: Robot joint count.
        num_ee_frames: End-effector frame count.
        blocks: Parent block selection.

    Returns:
        ``([D] group index per channel, [G] group names)``.
    """
    from robot_rl.flow_matching.window import (  # noqa: PLC0415
        REFERENCE_BLOCK_KEYS,
        expand_window_blocks,
    )

    names = group_names(blocks)
    table: dict[str, float] = {k: float(i) for i, k in enumerate(names)}
    # Keys outside the selection are never read, but `expand_window_blocks` builds its
    # lambdas eagerly, so they must be present.
    for ks in REFERENCE_BLOCK_KEYS.values():
        for k in ks:
            table.setdefault(k, 0.0)
    idx = [int(v) for v in expand_window_blocks(num_joints, num_ee_frames, table, blocks)]
    return idx, names


class GroupedKnotGaussian(GaussianDistribution):
    """Diagonal Gaussian with a ``group x knot`` factored, state-independent std.

    Attributes:
        log_group_scale: ``[G]`` learned log width per channel group.
        log_knot_scale: ``[K]`` learned log width per knot, mean-centred when read.
    """

    def __init__(
        self,
        output_dim: int,
        num_knots: int,
        channel_group_index: list[int],
        group_names_: list[str],
        init_std: float | dict[str, float] = 0.15,
        std_type: str = "log",
    ) -> None:
        """Build the factored std.

        Args:
            output_dim: ``K * D``, the action width.
            num_knots: ``K``.
            channel_group_index: ``[D]`` group index per channel, from
                :func:`channel_groups`.
            group_names_: ``[G]`` group names, for per-group ``init_std`` lookup and logging.
            init_std: A scalar for every group, or a per-group-name mapping. A missing name
                RAISES rather than defaulting: the offline correction fit shows the groups
                need widths differing by ~2.7x, so silently handing one the wrong default is
                the failure this argument exists to avoid.
            std_type: Must be ``"log"``; the factorisation is defined in log space.

        Raises:
            ValueError: On a non-log ``std_type``, a width that is not ``K * D``, or an
                ``init_std`` mapping missing a group.
        """
        if std_type != "log":
            raise ValueError(
                f"std_type={std_type!r}: GroupedKnotGaussian is defined in log space, where "
                "the group and knot factors ADD. A 'scalar' parameterisation would make them "
                "multiply and the mean-centring gauge meaningless."
            )
        dim = int(output_dim) // int(num_knots)
        if int(num_knots) * dim != int(output_dim):
            raise ValueError(
                f"output_dim {int(output_dim)} is not divisible by num_knots {int(num_knots)}"
            )
        if len(channel_group_index) != dim:
            raise ValueError(
                f"channel_group_index has {len(channel_group_index)} entries but the action "
                f"implies D = output_dim / K = {dim}."
            )
        Distribution.__init__(self, int(output_dim))
        self.std_type = "log"
        self.num_knots = int(num_knots)
        self.node_dim = dim
        self.group_names = list(group_names_)

        if isinstance(init_std, dict):
            missing = [n for n in self.group_names if n not in init_std]
            if missing:
                raise ValueError(
                    f"init_std is missing channel group(s) {missing}; it lists "
                    f"{sorted(init_std)}. Every group must be stated -- the measured "
                    "correction magnitudes differ by ~2.7x across them."
                )
            per_group = [float(init_std[n]) for n in self.group_names]
        else:
            per_group = [float(init_std)] * len(self.group_names)
        if any(v <= 0.0 for v in per_group):
            raise ValueError(f"init_std must be positive per group, got {per_group}")

        self.log_group_scale = nn.Parameter(torch.log(torch.tensor(per_group)))
        # Flat, so at init sigma is exactly the group's own init_std.
        self.log_knot_scale = nn.Parameter(torch.zeros(self.num_knots))
        self.register_buffer("channel_group",
                             torch.tensor(list(channel_group_index), dtype=torch.long),
                             persistent=True)

        self._distribution = None
        torch.distributions.Normal.set_default_validate_args(False)

    @property
    def log_std_param(self) -> torch.Tensor:
        """``[K * D]`` log std, expanded from the 18 factors.

        A property, not a Parameter, so the inherited ``update`` and ``ChunkPPO``'s layout
        check both see the action's true width. ``log_knot_scale`` is mean-centred here,
        which is what pins the gauge -- see the module docstring.
        """
        knot = self.log_knot_scale - self.log_knot_scale.mean()
        per_channel = self.log_group_scale[self.channel_group]          # (D,)
        return (knot.unsqueeze(-1) + per_channel.unsqueeze(0)).reshape(-1)

    @property
    def group_std(self) -> torch.Tensor:
        """``[G]`` the learned per-group width ``s_g``, for logging."""
        return torch.exp(self.log_group_scale.detach())

    @property
    def knot_std(self) -> torch.Tensor:
        """``[K]`` the learned per-knot factor ``h_j`` about 1.0, for logging."""
        c = self.log_knot_scale.detach()
        return torch.exp(c - c.mean())
