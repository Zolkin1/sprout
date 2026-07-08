# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generator smoke/regression test on the small committed library fixture.

Uses the flat ``merged_library`` fixture (one 'default' skill, mixed speeds) and a
speed-change schedule, so it exercises the full match -> play -> handoff -> steady
machinery + output assembly without needing the bones dataset or the big library.
"""

from __future__ import annotations

import os

import pytest
import torch

from motion_gen.motion_matching import features as F
from motion_gen.motion_matching.mm_clip_generator import MotionMatchClipGenerator
from motion_gen.motion_matching.motion_db import MotionDB
from motion_gen.motion_matching.trajopt_ingest import ingest_trajopt

_FIX = os.path.join(os.path.dirname(__file__), "..", "..", "robot_rl", "tests",
                    "test_data", "merged_library")
pytestmark = pytest.mark.skipif(
    not (os.path.isfile(F.DEFAULT_MJCF) and os.path.isdir(_FIX)),
    reason="G1 MJCF or library fixture not found",
)


def _gen(tmp_path):
    from motion_gen.multiskill_manager import MultiSkillManager
    os.environ["ROBOT_RL_TRAJ_CACHE_DIR"] = str(tmp_path / "cache")
    m = MultiSkillManager(_FIX, "cpu", env=None)
    cfg = F.FeatureConfig(dt=0.02)
    fk = F.make_fk(cfg, device="cpu")
    srcs = ingest_trajopt(m, device="cpu")
    db = MotionDB(cfg, "cpu").build(srcs, fk)
    return MotionMatchClipGenerator(m, db, srcs, cfg, dt=0.02, device="cpu"), m


def test_generate_valid_clip(tmp_path):
    gen, m = _gen(tmp_path)
    schedule = [
        {"skill": "default", "vx": 0.0, "wz": 0.0, "duration_s": 1.0},
        {"skill": "default", "vx": 3.0, "wz": 0.0, "duration_s": 1.5},
        {"skill": "default", "vx": 1.0, "wz": 0.0, "duration_s": 1.5},
    ]
    batch = gen.generate(schedule, clip_length_s=4.0)
    T = 200
    assert batch["y_des"].shape == (1, T, gen.P)
    assert batch["dy_des"].shape == (1, T, gen.V)
    assert batch["contact"].shape == (1, T, 2)
    assert batch["skill_id"].shape == (1, T) and batch["target_vel"].shape == (1, T, 3)
    y, dy = batch["y_des"][0], batch["dy_des"][0]
    assert torch.isfinite(y).all() and torch.isfinite(dy).all()
    # joints within plausible limits
    jj = y[:, gen._p_joint]
    assert float(jj.abs().max()) < 3.2
    # root C0 continuity (no teleports): per-step pelvis move stays small
    pel = y[:, gen._p_pos["pelvis_link"]]
    assert float((pel[1:] - pel[:-1]).norm(dim=1).max()) < 0.3
    # transition frames were spliced in (stride_idx == -1 for MM frames)
    assert int((batch["stride_idx"][0] == -1).sum()) > 0
    # terrain tags
    assert batch["terrain_type"] == "flat" and int(batch["terrain_id"][0]) == -1
    assert isinstance(gen.warnings, list)


def test_output_dtypes_match_writer(tmp_path):
    """Dtypes/keys line up with what ClipLibraryWriter.write_batch expects."""
    gen, _ = _gen(tmp_path)
    batch = gen.generate([{"skill": "default", "vx": 0.0, "wz": 0.0, "duration_s": 2.0}],
                         clip_length_s=2.0)
    assert batch["skill_id"].dtype == torch.uint8
    assert batch["stride_idx"].dtype == torch.int32
    assert batch["target_vel"].dtype == torch.float32
    assert batch["contact"].dtype == torch.uint8
