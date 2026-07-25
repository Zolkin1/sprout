# Copyright 2026 Zachary Olkin. All rights reserved.

"""Boundary-value DP hooks: approach-pin (terminal) + exit-seed (start frame).

Builds a real :class:`DPMatcher` + candidate graph from the committed ``merged_library``
fixture and checks that pinning the terminal toward a target entry descriptor makes the
optimal path END near that state, and that seeding the reconstruction start makes it BEGIN
there.  This is the DP machinery the box-clip generator uses to fill the approach/exit
free-locomotion segments around a played terrain interaction.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from motion_gen.motion_matching import features as F
from motion_gen.motion_matching.dp import DPConfig, DPMatcher
from motion_gen.motion_matching.graph import GraphConfig, build_candidate_graph
from motion_gen.motion_matching.trajopt_ingest import ingest_trajopt
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig

_FIX = os.path.join(os.path.dirname(__file__), "..", "..", "robot_rl", "tests",
                    "test_data", "merged_library")
pytestmark = pytest.mark.skipif(
    not (os.path.isfile(F.DEFAULT_MJCF) and os.path.isdir(_FIX)),
    reason="G1 MJCF or library fixture not found",
)


def _matcher(tmp_path):
    from motion_gen.multiskill_manager import MultiSkillManager
    os.environ["ROBOT_RL_TRAJ_CACHE_DIR"] = str(tmp_path / "cache")
    m = MultiSkillManager(_FIX, "cpu", env=None)
    srcs = ingest_trajopt(m, device="cpu")
    vcfg = VanillaConfig(dt=0.02)
    dcfg = DPConfig(dt=0.02, w_traj=1.0, w_track=0.0, cut_penalty=3.0)
    matcher = DPMatcher(srcs, vcfg, dcfg, device="cpu")
    gcfg = GraphConfig(k_min=4, k_max=8, k_target=6, dt=0.02)
    cost_feat = matcher._norm[:, matcher._traj_dim:]
    select_feat = cost_feat[:, POSE_POSTURAL_IDX]
    graph = build_candidate_graph(select_feat, cost_feat, matcher.feat_raw, matcher.owner,
                                  matcher.src_start, matcher.src_len, matcher.yaw_all, gcfg)
    matcher.attach_graph(graph)
    return matcher, srcs


def _cmd(_t):
    return (1.0, 0.0, 0.0)


def test_boundary_cost_minimal_at_source_frame0(tmp_path):
    """A source's own entry descriptor is (near) zero-distance at that source's first frame."""
    matcher, srcs = _matcher(tmp_path)
    entry_desc, _ = matcher.boundary_descriptors(srcs[0])
    bc = matcher.boundary_cost(entry_desc)
    frame0 = int(matcher.src_start[0])
    assert int(bc.argmin()) == frame0
    assert bc[frame0] < 1e-3                              # recomputed features == the DB row


def test_approach_pin_ends_near_entry(tmp_path):
    """Pinning the terminal toward an entry descriptor makes the path END near that state."""
    matcher, srcs = _matcher(tmp_path)
    # use a DIFFERENT source's frame 0 as the target "terrain entry"
    target = srcs[min(2, len(srcs) - 1)]
    entry_desc, _ = matcher.boundary_descriptors(target)
    bc = matcher.boundary_cost(entry_desc)
    T = 80

    matcher.generate(_cmd, T)                            # unpinned
    end_unpinned = float(bc[int(matcher.frame_path[-1])])

    matcher.generate_approach(_cmd, T, entry_desc, boundary_weight=200.0)
    end_pinned = float(bc[int(matcher.frame_path[-1])])

    assert end_pinned <= end_unpinned + 1e-6             # the pin never hurts
    assert end_pinned < float(np.median(bc))             # and lands much closer than a random frame


def test_exit_seed_starts_at_exit_frame(tmp_path):
    """Seeding the exit DP makes the reconstructed path BEGIN at the exit-nearest frame."""
    matcher, srcs = _matcher(tmp_path)
    _, exit_desc = matcher.boundary_descriptors(srcs[min(1, len(srcs) - 1)])
    start = int(matcher.boundary_cost(exit_desc).argmin())
    matcher.generate_exit(_cmd, 60, exit_desc)
    assert int(matcher.frame_path[0]) == start
