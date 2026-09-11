# Copyright 2026 Zachary Olkin. All rights reserved.

"""Common random numbers over the generator's only source of stochasticity.

The flow's sole randomness is ``x0``, the Gaussian the Euler integration starts from, so
fixing it makes a plan a deterministic function of its conditioning.  This turns that into an
addressing scheme: every draw the harness needs is named by ``(group, round, tag)`` and
derived from one run seed, so nothing depends on call order and a rerun reproduces the same
K chunks at the same states.

**Why this is the highest-leverage piece.**  Within a round the K branches must differ in
exactly one thing -- the noise for the branched chunk.  If the four CONTINUATION re-plans
inside the scoring window also drew independently, the score ranking would be dominated by
which branch happened to get a good continuation, and the measured spread would be
continuation variance wearing the label "chunk quality".  So every branch of a group is
handed the SAME continuation draw at each of the four subsequent re-plans.

Three properties worth stating because each was a choice:

* **The parent shares the group's continuation stream.**  Together with candidate 0 taking
  the parent's own plan by copy rather than by re-inference, that makes candidate 0's rolled
  window and the parent's own next W seconds the same computation up to the state restore --
  which is the harness noise floor, measured at every branched state.
* **Seeds are derived, not counted.**  ``splitmix64`` over ``(run, group, round, tag)``, so
  arming a phase needs no history and a resumed or reordered run draws the same numbers.
* **A draw is an ordinary Gaussian.**  This is not ``fm_zero_noise``, which integrates from
  ``x0 = 0`` and is off the training distribution.  Only WHICH sample is under control.
"""

from __future__ import annotations

import numpy as np
import torch

#: Tag mixed into a branched-chunk draw, so candidate ``c`` of a state is addressable.
TAG_BRANCH: int = 0x9E3779B97F4A7C15
#: Tag mixed into a continuation draw, shared by every branch of a group.
TAG_CONTINUATION: int = 0xC0FFEE0BADF00D01

_MASK64: int = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """One round of splitmix64 -- an avalanching 64-bit integer hash.

    Used rather than ``hash()`` (salted per process, so a rerun would draw differently) and
    rather than a counter (order-dependent, so a resumed run would diverge).

    Args:
        x: Any 64-bit value.

    Returns:
        A well-mixed 64-bit value.
    """
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def derive(*parts: int) -> int:
    """Mix an arbitrary tuple of ints into one 64-bit seed."""
    acc = 0
    for p in parts:
        acc = splitmix64((acc ^ (int(p) & _MASK64)) & _MASK64)
    return acc


class CrnNoise:
    """Per-env flow noise addressed by ``(group, round, tag)``.

    Install with ``cmd.set_noise_source(crn)``.  The command calls it with the env ids it is
    about to sample; the phase currently armed decides what those ids get.
    """

    def __init__(self, *, num_envs: int, group_size: int, future_len: int, state_dim: int,
                 device, run_seed: int) -> None:
        """Allocate the addressing tables.

        Args:
            num_envs: Total envs.  Must be a whole number of groups.
            group_size: ``K + 1`` -- local index 0 is the parent, ``j + 1`` is candidate
                ``j``.
            future_len: ``F``, the generator's horizon in nodes.
            state_dim: ``D``, the per-node state width.
            device: Torch device the noise must land on.
            run_seed: The run's master seed.

        Raises:
            ValueError: If ``num_envs`` is not a multiple of ``group_size`` -- a partial
                group would silently give some candidate index two envs.
        """
        if num_envs % group_size:
            raise ValueError(
                f"num_envs={num_envs} is not a multiple of group_size={group_size}; the "
                "last group would be partial and its candidate indices would not line up."
            )
        self.num_envs = int(num_envs)
        self.group_size = int(group_size)
        self.k = int(group_size) - 1
        self.future_len = int(future_len)
        self.state_dim = int(state_dim)
        self.device = device
        self.run_seed = int(run_seed)
        self.num_groups = self.num_envs // self.group_size

        idx = np.arange(self.num_envs, dtype=np.int64)
        #: Group index of each env.
        self.group = idx // self.group_size
        #: Local slot: 0 for the parent, ``j + 1`` for candidate ``j``.
        self.slot = idx % self.group_size
        #: Candidate index each env carries.  The PARENT carries candidate 0 as well: its
        #: round-start plan is the draw candidate 0's env then inherits by copy, so the two
        #: must name the same seed.
        self.candidate = np.maximum(self.slot - 1, 0)
        #: Which envs are branch evaluators rather than parents.
        self.is_branch = self.slot > 0

        #: When set, every candidate is served the seed of THIS candidate index instead of
        #: its own -- the same-noise CONTROL.  With it on, the K branches of a state differ
        #: in nothing but which env slot they landed in, so their pairwise disagreement is
        #: pure state-restore noise with no chunk effect at all, and any best-of-K rise the
        #: control shows is an artifact of the harness rather than a property of the model.
        #: It is the only clean way to read the real arm's rise, because candidate-0-vs-parent
        #: compares RESTORED against UNRESTORED and is therefore a different question.
        self.force_candidate: int | None = None
        self._round = np.zeros(self.num_groups, dtype=np.int64)
        self._phase: tuple[str, int] = ("branch", 0)
        #: Every seed served, for the run manifest and for the reproducibility check.
        self.calls: int = 0

    # ------------------------------------------------------------------ addressing
    def state_seed(self, group: int, round_index: int) -> int:
        """The seed naming one branched state; recorded on every row of that state."""
        return derive(self.run_seed, int(group), int(round_index))

    def branch_seed(self, group: int, round_index: int, candidate: int) -> int:
        """Seed for candidate ``candidate``'s chunk at one state."""
        return derive(self.state_seed(group, round_index), TAG_BRANCH, int(candidate))

    def continuation_seed(self, group: int, round_index: int, replan: int) -> int:
        """Seed for the ``replan``-th re-plan inside a window -- shared by the whole group."""
        return derive(self.state_seed(group, round_index), TAG_CONTINUATION, int(replan))

    # ------------------------------------------------------------------ phase
    def set_round(self, group_ids, round_index) -> None:
        """Set the round counter for some groups.

        Args:
            group_ids: Group indices, or ``None`` for all of them.
            round_index: Scalar or per-group array.
        """
        if group_ids is None:
            self._round[:] = round_index
        else:
            self._round[np.asarray(group_ids, dtype=np.int64)] = round_index

    def arm_branch(self, round_offset: int = 0) -> None:
        """Serve branched-chunk draws for ``current round + round_offset``.

        ``round_offset=1`` is armed before the LAST step of a window: the command computes
        the next step's reference at the end of the step, so the re-plan that opens the next
        round is drawn there.
        """
        self._phase = ("branch", int(round_offset))

    def arm_continuation(self, replan: int) -> None:
        """Serve the shared continuation draw for the ``replan``-th re-plan of the window."""
        self._phase = ("continuation", int(replan))

    @property
    def phase(self) -> tuple[str, int]:
        """The armed ``(kind, index)``, for logging."""
        return self._phase

    # ------------------------------------------------------------------ serve
    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """Produce ``[V, F, D]`` noise for the envs about to be sampled.

        Args:
            ids: ``[V]`` long tensor of env indices, in the order the caller will use them.

        Returns:
            The initial-noise tensor, on ``self.device``.
        """
        env_ids = ids.detach().to("cpu").numpy().astype(np.int64)
        kind, index = self._phase
        self.calls += 1
        out = torch.empty(env_ids.size, self.future_len, self.state_dim, device=self.device)
        if kind == "continuation":
            # One draw per GROUP, broadcast: this is what stops the ranking from being about
            # which branch got a lucky continuation.
            cache: dict[int, torch.Tensor] = {}
            for row, env in enumerate(env_ids):
                g = int(self.group[env])
                if g not in cache:
                    cache[g] = self._draw(
                        self.continuation_seed(g, int(self._round[g]), index))
                out[row] = cache[g]
            return out
        for row, env in enumerate(env_ids):
            g = int(self.group[env])
            r = int(self._round[g]) + index
            cand = (int(self.candidate[env]) if self.force_candidate is None
                    else int(self.force_candidate))
            out[row] = self._draw(self.branch_seed(g, r, cand))
        return out

    def _draw(self, seed: int) -> torch.Tensor:
        """One ``[F, D]`` Gaussian from a named seed.

        Drawn on the CPU and moved, deliberately: a CUDA generator's stream is not portable
        across devices or driver versions, so a run reproduced on another GPU would get
        different numbers from the same seed.  The tensor is 62x~130 floats and there are a
        few thousand draws per hour of rollout, so the copy does not show up.
        """
        gen = torch.Generator(device="cpu")
        # torch.Generator.manual_seed takes a SIGNED 64-bit value; the derived seeds are
        # unsigned, so fold rather than truncate.
        gen.manual_seed(int(seed & ((1 << 63) - 1)))
        return torch.randn(self.future_len, self.state_dim, generator=gen).to(self.device)
