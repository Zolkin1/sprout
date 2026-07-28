# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared paths for the robot_rl tests.

The trajectory-manager fixtures that used to live here went with the managers themselves
(the offline clip pipeline replaced them). The YAML/library fixtures under ``test_data/`` are
still referenced directly by a few tests as *paths*, so the constants stay.
"""

from pathlib import Path

TEST_DATA = Path(__file__).parent / "test_data"
STANDING_YAML = str(TEST_DATA / "standing.yaml")
WALKING_YAML = str(TEST_DATA / "walking_100.yaml")
RUNNING_YAML = str(TEST_DATA / "running_200.yaml")
MERGED_LIBRARY_DIR = str(TEST_DATA / "merged_library")

DEVICE = "cpu"
