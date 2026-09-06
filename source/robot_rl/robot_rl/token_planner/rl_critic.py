# Copyright 2026 Zachary Olkin. All rights reserved.

"""The value head both token RL stages share.

It lives here rather than in either trainer because both need it and neither can import the
other: those scripts boot ``SimulationApp`` at import time, so anything defined after that
line is unreachable to anything that is not itself an Isaac entry point. Duplicating it
instead would leave two normalizers that have to be kept identical by hand -- and the
normalizer is stateful, so a drift between them would show up as one stage's value loss
being on a different scale than the other's, with nothing to say why.
"""

from __future__ import annotations

import torch
from torch import nn


class Critic(nn.Module):
    """A plain MLP state-value head with a running INPUT normalizer.

    Deliberately NOT the policy's trunk: the spec asks for a separate network, and sharing
    would couple the value loss's gradients into the very encoder whose drift the KL term
    exists to bound.

    Only the INPUT is normalized, despite what a reader might assume from "normalization":
    returns are left in their own units so the divergence gate can see them. Explained
    because the docstring this replaces claimed both and only did one.

    Args:
        obs_dim: Width of the critic observation.
        hidden: Hidden widths.
    """

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (512, 256, 128)) -> None:
        super().__init__()
        layers, d = [], int(obs_dim)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("_mu", torch.zeros(obs_dim))
        self.register_buffer("_var", torch.ones(obs_dim))
        self.register_buffer("_n", torch.tensor(1e-4))

    def update_norm(self, x: torch.Tensor) -> None:
        """Welford update of the input normalizer over a rollout batch.

        Args:
            x: ``[B, obs_dim]`` observations as they ARRIVE, before any update step -- so the
                statistics describe the distribution the values were estimated under.
        """
        b = x.shape[0]
        mu, var = x.mean(0), x.var(0, unbiased=False)
        d = mu - self._mu
        tot = self._n + b
        self._mu += d * b / tot
        self._var.copy_((self._var * self._n + var * b + d.pow(2) * self._n * b / tot) / tot)
        self._n.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Value estimate ``[B]`` for an unnormalized observation.

        Args:
            x: ``[B, obs_dim]``.

        Returns:
            ``[B]``.
        """
        z = (x - self._mu) / (self._var.sqrt() + 1e-6)
        return self.net(z.clamp(-10.0, 10.0)).squeeze(-1)
