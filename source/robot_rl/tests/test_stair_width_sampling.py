# Copyright 2026 Zachary Olkin. All rights reserved.

"""``sample_terrain_specs`` samples the stair tread width per terrain.

When a stair library declares ``terrain_width`` as a range (a *don't-care*
dim), the offline orchestrator samples each terrain's tread width in that
range instead of pinning a single value; a fixed-width library (``lo == hi``)
still builds every terrain at that one width.  ``sample_terrain_specs`` is
pure (numpy only, no trajectory library), so this runs without the HF cache.
"""

from __future__ import annotations

import numpy as np

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
    sample_terrain_specs,
)

_HD = [(0.135, 0.233)]


def _specs(rng, *, n_terrain, n_transition, width_range, nav_width_range=(1.5, 3.0)):
    return sample_terrain_specs(
        rng, n_terrain=n_terrain, n_transition=n_transition, hd_grid=_HD,
        width_range=width_range,
        num_steps_range=(3, 4), lead_in_range=(1.5, 4.0), end_platform=1.0,
        num_transitions_range=(2, 2), inter_flat_range=(3.0, 5.0),
        nav_width_range=nav_width_range, flat_width_margin=0.3, transition_length=20.0,
    )


def test_range_width_sampled_within_bounds_and_varies():
    """Terrain-only widths land inside the range and are not all identical."""
    lo, hi = 1.0, 2.0
    widths = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        specs = _specs(rng, n_terrain=1, n_transition=0, width_range=(lo, hi))
        w = specs[0].width
        assert lo - 1e-9 <= w <= hi + 1e-9, f"width {w} out of [{lo}, {hi}]"
        # terrain-only: corridor width tracks the tread width.
        assert specs[0].nav_width == w
        widths.append(w)
    assert max(widths) - min(widths) > 0.1, "widths should vary across terrains"


def test_fixed_width_is_constant():
    """A degenerate range (lo == hi) pins every terrain to that width."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        specs = _specs(rng, n_terrain=2, n_transition=2, width_range=(0.777, 0.777))
        for s in specs:
            assert abs(s.width - 0.777) < 1e-9, f"expected fixed width, got {s.width}"


def test_transition_width_capped_at_corridor():
    """A transition's tread width never exceeds its (narrower) corridor width."""
    for seed in range(40):
        rng = np.random.default_rng(seed)
        # Wide tread range, fixed narrow corridor -> the cap must bind.
        specs = _specs(
            rng, n_terrain=0, n_transition=1, width_range=(1.0, 3.0),
            nav_width_range=(1.2, 1.2),
        )
        s = specs[0]
        assert s.ttype == "transition"
        assert s.width <= s.nav_width + 1e-9, f"tread {s.width} > corridor {s.nav_width}"
        assert s.width >= 1.0 - 1e-9
