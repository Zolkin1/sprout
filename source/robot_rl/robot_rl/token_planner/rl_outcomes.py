# Copyright 2026 Zachary Olkin. All rights reserved.

"""How an episode ENDED, bucketed into success / fall / timeout.

One function, because the training loop and the `--validate_only` gate were counting the
same three things differently and the difference was invisible in the numbers: the gate
derived success as a RESIDUAL (``done - fell - timeout``) while subtracting only
``tracking_violation``, so once the task grew a second fall term (``base_fallen``) every
global-fall ending was scored as a completed traverse. A residual cannot notice a
termination key nobody told it about; a direct count can.

The convention is ARRIVAL WINS: an episode that enters the goal region and falls on the same
control step is a success, because reaching the goal is the task and the episode ends there
either way. That is the user's rule, and it matches `compute_gae`'s refusal to guess -- the
ambiguous state is resolved here, once, by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

#: Termination keys that mean the robot failed. Both are real terms on the planner task:
#: the trajectory-RELATIVE tracking bound and the GLOBAL height/tilt bound.
FALL_KEYS: tuple[str, ...] = ("tracking_violation", "base_fallen")

#: Termination keys that mean the episode simply ran out, having neither arrived nor fallen.
TIMEOUT_KEYS: tuple[str, ...] = ("time_out", "clip_end")


@dataclass
class EndingCounts:
    """Tally of episode endings.

    ``success + fell + timeout + unattributed == episodes`` by construction, so a key the
    task grows and nobody classifies shows up as ``unattributed`` rather than silently
    inflating one of the three buckets.

    Attributes:
        episodes: Episodes that ended.
        success: Endings that reached the goal region (arrival wins).
        fell: Endings that did NOT reach the goal and hit a fall term.
        timeout: Endings that neither reached the goal nor fell.
        goal_and_fall: Successes that also tripped a fall term on the same step.
        unattributed: Endings matching no known key -- always a defect to chase.
        by_key: Raw per-key counts. These OVERLAP (one ending can set several keys).
    """

    episodes: int = 0
    success: int = 0
    fell: int = 0
    timeout: int = 0
    goal_and_fall: int = 0
    unattributed: int = 0
    by_key: dict[str, int] = field(default_factory=dict)

    def update(self, done: torch.Tensor, trunc: torch.Tensor,
               ended_by: dict[str, torch.Tensor] | None) -> None:
        """Fold one step's endings in.

        Args:
            done: ``[N]`` bool, envs whose episode ended this step.
            trunc: ``[N]`` bool, the goal-reached signal (the env's truncation).
            ended_by: Per-termination-key ``[N]`` flags, or None/empty if the env exposes none.
        """
        if not bool(done.any()):
            return
        by = ended_by or {}
        self.episodes += int(done.sum())
        for key, flag in by.items():
            self.by_key[key] = self.by_key.get(key, 0) + int(flag[done].sum())

        goal = done & trunc
        fall_f = torch.zeros_like(done)
        for key in FALL_KEYS:
            if key in by:
                fall_f |= by[key].bool()
        tout_f = torch.zeros_like(done)
        for key in TIMEOUT_KEYS:
            if key in by:
                tout_f |= by[key].bool()

        self.success += int(goal.sum())
        self.goal_and_fall += int((goal & fall_f).sum())
        rest = done & ~trunc
        self.fell += int((rest & fall_f).sum())
        self.timeout += int((rest & tout_f & ~fall_f).sum())
        self.unattributed += int((rest & ~fall_f & ~tout_f).sum())

    def rates(self) -> dict[str, float]:
        """Return the three headline rates over ended episodes (0.0 when nothing ended)."""
        n = max(self.episodes, 1)
        return {"success": self.success / n, "fell": self.fell / n,
                "timeout": self.timeout / n, "unattributed": self.unattributed / n,
                "goal_and_fall": self.goal_and_fall / n}

    def as_dict(self) -> dict:
        """Return a JSON-serialisable record of everything counted."""
        return {"episodes": self.episodes, "success": self.success, "fell": self.fell,
                "timeout": self.timeout, "goal_and_fall": self.goal_and_fall,
                "unattributed": self.unattributed, "by_key": dict(self.by_key)}
