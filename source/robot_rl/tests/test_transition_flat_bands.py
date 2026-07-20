# Copyright 2026 Zachary Olkin. All rights reserved.

"""Transition flat bands: lead-in + inter-stair flats sampled IN range, TOP platform absorbs slack.

``sample_terrain_specs`` is pure (numpy only, no trajectory library), so this runs without the HF
cache.  Build transitions with tiny stairs (lots of flat slack) and check the inter-stair flats STAY
within ``inter_flat_range`` (the slack piles onto the top platform, not the inter-flats), the lead-in
stays in ``lead_in_range``, and the total length is exactly ``transition_length``.  Also covers the
single-step curb path.
"""

from __future__ import annotations

import numpy as np

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
    sample_terrain_specs,
)


def test_inter_flats_in_range_top_absorbs_slack():
    hd = [(0.15, 0.25)]
    L = 20.0                            # long terrain -> large flat slack (would inflate flats)
    min_lead, max_lead, min_inter, max_inter, min_end = 1.5, 4.0, 3.0, 5.0, 1.0
    big_top_seen = False
    for seed in range(60):
        rng = np.random.default_rng(seed)
        specs = sample_terrain_specs(
            rng, n_terrain=0, n_transition=1, hd_grid=hd, width_range=(0.777, 0.777),
            num_steps_range=(3, 3), lead_in_range=(min_lead, max_lead), end_platform=min_end,
            num_transitions_range=(2, 2), inter_flat_range=(min_inter, max_inter),
            nav_width_range=(1.0, 1.0), flat_width_margin=0.3, transition_length=L,
        )
        s = specs[0]
        assert s.ttype == "transition"
        # Lead-in stays in range.
        assert min_lead - 1e-6 <= s.lead_in <= max_lead + 1e-6, f"lead_in {s.lead_in} out of range"
        # Inter-stair flats (all but the last) stay within [min_inter, max_inter] -- NOT slack-inflated.
        for f in s.flat_after[:-1]:
            assert min_inter - 1e-6 <= f <= max_inter + 1e-6, f"inter-flat {f} out of [{min_inter},{max_inter}]"
        # Top platform absorbs the slack (>= floor; large here).
        assert s.flat_after[-1] >= min_end - 1e-6, "top platform below floor"
        if s.flat_after[-1] > max_inter:
            big_top_seen = True
        # Total length preserved exactly.
        total = s.lead_in + sum(s.stair_steps) * s.step_depth + sum(s.flat_after)
        assert abs(total - L) < 1e-4, f"total {total} != {L}"
    assert big_top_seen, "top platform never exceeded inter-flat max -- slack not landing on the top"


def test_single_step_curbs():
    hd = [(0.15, 0.25)]
    rng = np.random.default_rng(0)
    specs = sample_terrain_specs(
        rng, n_terrain=0, n_transition=200, hd_grid=hd, width_range=(0.777, 0.777),
        num_steps_range=(2, 7), lead_in_range=(1.5, 4.0), end_platform=1.0,
        num_transitions_range=(2, 3), inter_flat_range=(3.0, 5.0),
        nav_width_range=(1.0, 1.0), flat_width_margin=0.3, transition_length=16.0,
        single_step_frac=0.5,
    )
    curbs = [s for s in specs if s.stair_steps == [1]]
    multis = [s for s in specs if s.stair_steps != [1]]
    assert curbs and multis, "expected both single-step curbs and multi-staircase transitions"
    for s in curbs:                                          # curb: one riser, lead-in + top platform
        assert s.stair_steps == [1]
        assert s.lead_in >= 1.5 - 1e-6
        assert len(s.flat_after) == 1 and s.flat_after[-1] >= 1.0 - 1e-6
    for s in multis:                                         # every staircase in [2, 7]
        assert all(2 <= n <= 7 for n in s.stair_steps)
    assert 0.3 < len(curbs) / len(specs) < 0.7               # ~half at frac=0.5
