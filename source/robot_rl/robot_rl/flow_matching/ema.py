# Copyright 2026 Zachary Olkin. All rights reserved.

"""Exponential moving average of the model weights.

Standard practice for diffusion / flow models: the weights that minimise the training
objective step-to-step are noisier than an average over the recent trajectory, and the
average usually samples better. Cheap -- one extra copy of the parameters and an
elementwise update per step.

Two things are deliberately separate, and conflating them is the easy mistake:

* **update** happens every OPTIMIZER STEP. That is what makes it an average of the
  trajectory. An EMA refreshed only when a checkpoint is written is just a subsample of
  the weights, with none of the averaging.
* **saving** happens whenever a checkpoint is written, into the same file as the raw
  weights (``model_ema`` beside ``model``), so one file can be loaded either way and
  nothing has to be decided at training time.

The training loop should also VALIDATE through :meth:`swapped_in`, and log both, so the
question "did EMA help this run?" is answerable from the run itself rather than assumed.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


class WeightEMA:
    """Shadow copy of a model's weights, updated as an exponential moving average.

    Args:
        model: The model whose ``state_dict`` is shadowed.
        decay: Target decay per step. 0.999 keeps an effective window of ~1000 steps.
        warmup: Ramp the decay in over the first steps (see :meth:`update`). Leave on --
            without it the average is dominated by the random initialisation for the first
            ``1 / (1 - decay)`` steps, which on a short run is most of the run.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup: bool = True):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.step = 0
        # Shadow EVERY state_dict entry so the EMA is a drop-in replacement. Only floating
        # point tensors are averaged; integer buffers (counters, index tables) are copied,
        # because an average of indices is meaningless.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._avg = {k for k, v in self.shadow.items() if v.is_floating_point()}

    def effective_decay(self) -> float:
        """Decay for the current step, ramped in when ``warmup``.

        ``(1 + step) / (10 + step)`` starts near 0.1 and approaches 1, so early updates
        track the live weights closely instead of the initialisation.
        """
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Fold one optimizer step into the average. Call after ``optimizer.step()``."""
        self.step += 1
        d = self.effective_decay()
        for key, value in model.state_dict().items():
            if key in self._avg:
                self.shadow[key].mul_(d).add_(value.detach(), alpha=1.0 - d)
            else:
                self.shadow[key].copy_(value.detach())

    @contextmanager
    def swapped_in(self, model: torch.nn.Module):
        """Temporarily install the EMA weights on ``model``, restoring the raw ones after.

        Lets validation, sampling and export run through the averaged weights without a
        second model instance. Restores in a ``finally``, so an exception mid-evaluation
        cannot leave the trainer optimising the shadow copy.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(backup)

    def state_dict(self) -> dict:
        """Serialisable state: the shadow weights plus the step count and settings."""
        return {"shadow": self.shadow, "step": self.step,
                "decay": self.decay, "warmup": self.warmup}

    def load_state_dict(self, state: dict) -> None:
        """Restore from :meth:`state_dict` (for resuming a run)."""
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        self.step = int(state.get("step", 0))
        self.decay = float(state.get("decay", self.decay))
        self.warmup = bool(state.get("warmup", self.warmup))
        self._avg = {k for k, v in self.shadow.items() if v.is_floating_point()}

    def to(self, device) -> "WeightEMA":
        """Move the shadow weights to ``device`` (in place); returns self."""
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self
