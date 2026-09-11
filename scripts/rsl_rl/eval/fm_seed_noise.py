# Copyright 2026 Zachary Olkin. All rights reserved.

"""Make the generator's noise draw a named, reproducible input to an eval.

The flow's only stochasticity is its initial noise ``x0``.  Ordinarily it comes from the
global RNG, which means the only way to change it is ``--seed`` -- and that also moves the DR
draws, the spawn jitter and the observation noise, so a difference between two runs is
attributable to nothing in particular.  This makes the draw addressable on its own, so an
experiment can vary it and hold everything else fixed.

**The key is ``(fm_seed, replan index WITHIN the episode)`` and deliberately NOT the env.**
That is what lets several robots running one clip under one ``fm_seed`` act as the simulator
floor: they are served the identical noise sequence, so any difference in their outcomes is
the simulator and not the generator.  Keying by env instead -- the obvious way -- would make
those robots differ by noise AND by simulator at once, and the floor would contain the very
quantity it exists to measure.

The index is derived from ``episode_length_buf`` rather than counted, so it needs no state of
its own and cannot drift: two envs at the same point of their own episodes have the same
index by construction, whatever else they are doing, and an env that resets starts again at 0.
"""

from __future__ import annotations

import torch

from headroom_noise import derive

#: Sentinel for "leave the global RNG alone", so `0` stays a usable seed.
OFF: int = -1


class FmSeedNoise:
    """Per-env flow noise addressed by ``(fm_seed, within-episode re-plan index)``."""

    def __init__(self, *, fm_seed: int, future_len: int, state_dim: int, device,
                 replan_every: int, episode_len_buf) -> None:
        """Bind the draw to a seed and to the env's own step counter.

        Args:
            fm_seed: The named seed. Every env sees noise from this one draw stream.
            future_len: ``F``, the generator's horizon in nodes.
            state_dim: ``D``, the per-node state width.
            device: Torch device the noise must land on.
            replan_every: Control steps per re-plan (``fm_update_every``).
            episode_len_buf: Callable returning the env's ``[N]`` episode step counter.
        """
        self.fm_seed = int(fm_seed)
        self.future_len = int(future_len)
        self.state_dim = int(state_dim)
        self.device = device
        self.replan_every = max(1, int(replan_every))
        self._buf = episode_len_buf
        self._cache: dict[int, torch.Tensor] = {}
        self.calls = 0

    def draw(self, replan_index: int) -> torch.Tensor:
        """The ``[F, D]`` draw for one re-plan index, cached.

        Cached because every env at that index must get the SAME tensor, and re-deriving it
        per env would be both slower and an invitation to key it per env by accident.
        """
        k = int(replan_index)
        hit = self._cache.get(k)
        if hit is None:
            g = torch.Generator(device="cpu")
            g.manual_seed(derive(self.fm_seed, k) & ((1 << 63) - 1))
            hit = torch.randn(self.future_len, self.state_dim, generator=g).to(self.device)
            self._cache[k] = hit
        return hit

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """``[V, F, D]`` noise for the envs about to be sampled."""
        self.calls += 1
        idx = (self._buf()[ids] // self.replan_every).detach().to("cpu").tolist()
        return torch.stack([self.draw(i) for i in idx]).to(self.device)
