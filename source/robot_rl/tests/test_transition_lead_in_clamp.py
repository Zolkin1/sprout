# Copyright 2026 Zachary Olkin. All rights reserved.

"""The transition lead-in (start platform) is clamped to ``[min, max]``.

``sample_terrain_specs`` is pure (numpy only, no trajectory library), so this
runs without the HF cache: build transitions with tiny stairs (lots of flat
slack) so the fixed-length fill *would* push the lead-in well past the cap, and
check that (a) the lead-in is clamped to ``[min, max]``, (b) the other flat
bands keep their floors, and (c) the total length is still ``transition_length``
(the clamped excess was redistributed, not dropped).
"""

from __future__ import annotations

import numpy as np

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
    sample_terrain_specs,
)


def test_transition_lead_in_clamped_with_redistribution():
    hd = [(0.15, 0.25)]                 # single (step_height, step_depth)
    L = 20.0                            # long terrain -> large flat slack
    min_lead, max_lead, min_inter, min_end = 1.5, 4.0, 3.0, 1.0
    clamped_seen = False
    for seed in range(60):
        rng = np.random.default_rng(seed)
        specs = sample_terrain_specs(
            rng, n_terrain=0, n_transition=1, hd_grid=hd, width_range=(0.777, 0.777),
            num_steps_range=(3, 3),     # tiny stairs -> the lead-in wants the slack
            lead_in_range=(min_lead, max_lead), end_platform=min_end,
            num_transitions_range=(2, 2), inter_flat_range=(min_inter, 5.0),
            nav_width_range=(1.0, 1.0), flat_width_margin=0.3, transition_length=L,
        )
        s = specs[0]
        assert s.ttype == "transition"
        # (a) lead-in clamped to [min, max].
        assert min_lead - 1e-6 <= s.lead_in <= max_lead + 1e-6, f"lead_in {s.lead_in} out of range"
        if s.lead_in >= max_lead - 1e-6:
            clamped_seen = True
        # (b) inter-stair flats and top platform keep their floors.
        assert all(f >= min_inter - 1e-6 for f in s.flat_after[:-1]), "inter-stair flat below floor"
        assert s.flat_after[-1] >= min_end - 1e-6, "top platform below floor"
        # (c) total length preserved (redistribution, not truncation).
        total = s.lead_in + sum(s.stair_steps) * s.step_depth + sum(s.flat_after)
        assert abs(total - L) < 1e-4, f"total {total} != {L}"
    assert clamped_seen, "no draw hit the cap -- test not exercising the clamp path"
