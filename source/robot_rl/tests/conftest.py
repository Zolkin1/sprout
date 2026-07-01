# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared fixtures for trajectory tracking tests."""

import os
from pathlib import Path

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import TrajectoryManager
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import LibraryManager

TEST_DATA = Path(__file__).parent / "test_data"
STANDING_YAML = str(TEST_DATA / "standing.yaml")
WALKING_YAML = str(TEST_DATA / "walking_100.yaml")
RUNNING_YAML = str(TEST_DATA / "running_200.yaml")
MERGED_LIBRARY_DIR = str(TEST_DATA / "merged_library")

DEVICE = "cpu"


@pytest.fixture(scope="session", autouse=True)
def _isolated_traj_cache(tmp_path_factory):
    """Redirect MultiSkillManager's merged-tensor cache to a throwaway dir.

    Session-scoped + autouse so it's in effect before any module-scoped
    manager fixture is built — keeps the test suite from reading or writing
    the developer's real ``~/.cache/robot_rl`` cache.  Individual tests can
    still override ``ROBOT_RL_TRAJ_CACHE_DIR`` (e.g. via ``monkeypatch``) to
    isolate their own cache.
    """
    cache_dir = tmp_path_factory.mktemp("msm_tensor_cache")
    prev = os.environ.get("ROBOT_RL_TRAJ_CACHE_DIR")
    os.environ["ROBOT_RL_TRAJ_CACHE_DIR"] = str(cache_dir)
    yield
    if prev is None:
        os.environ.pop("ROBOT_RL_TRAJ_CACHE_DIR", None)
    else:
        os.environ["ROBOT_RL_TRAJ_CACHE_DIR"] = prev


@pytest.fixture(scope="module")
def standing_manager() -> TrajectoryManager:
    """Standing trajectory: periodic, 1 domain (double_support), T=0.4."""
    return TrajectoryManager(STANDING_YAML, None, DEVICE)


@pytest.fixture(scope="module")
def walking_manager() -> TrajectoryManager:
    """Walking trajectory: half_periodic, 1 domain (single_support), T=0.46."""
    return TrajectoryManager(WALKING_YAML, None, DEVICE)


@pytest.fixture(scope="module")
def running_manager() -> TrajectoryManager:
    """Running trajectory: half_periodic, 2 domains (single_support + flight_phase), T=0.299."""
    return TrajectoryManager(RUNNING_YAML, None, DEVICE)


@pytest.fixture(scope="module")
def merged_library() -> LibraryManager:
    """Merged walk/run library: 19 YAML files, mixed speeds."""
    return LibraryManager(MERGED_LIBRARY_DIR, None, DEVICE)
