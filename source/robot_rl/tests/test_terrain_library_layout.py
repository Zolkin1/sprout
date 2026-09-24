# Copyright 2026 Zachary Olkin. All rights reserved.

"""A clip-free course library must still name one output per ROBOT JOINT.

This is the check that turns a three-minute failure into a one-second one. The channel layout
was documented in ``scripts/rsl_rl/terrain_library.py`` as "pure bookkeeping -- nothing reads
it without clips to index", and that claim is false:
``BaseTrajectoryCommand._reference_world_indices`` resolves one output index per robot joint
at COMMAND CONSTRUCTION, before anything knows whether a clip will ever be gathered.

So a library built without ``--like`` writes clean, loads clean, passes the clip-manager
round trip in ``verify`` -- and then dies inside Isaac with::

    ValueError: Joint output 'joint:left_hip_pitch_joint' missing from trajectory outputs.

after the simulator has booted and the env has been built. That happened to all four fine-tune
libraries at once, and nothing between the build and the launch said a word. Every clip-free
course on disk that has ever RUN carries the full 155-position / 137-velocity layout; the
seven-channel one has never been used by anything.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (str(_ROOT / "scripts" / "rsl_rl"), str(_ROOT / "source" / "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

tl = pytest.importorskip("terrain_library")


def test_library_meta_refuses_a_missing_like() -> None:
    """``None`` used to hand back an unusable layout; it must now say so."""
    with pytest.raises(ValueError, match="--like"):
        tl.library_meta(None)


def test_the_minimal_schema_is_refused_as_a_layout() -> None:
    """It is the KEY SET, and it must never again be mistaken for a usable layout."""
    with pytest.raises(ValueError, match="joint:"):
        tl.assert_layout_is_usable(dict(tl._MINIMAL_META), source="_MINIMAL_META")


def test_a_joint_bearing_layout_passes() -> None:
    """The positive case, so the guard is not merely refusing everything."""
    meta = {"pos_output_names": ["pelvis_link:pos_x", "joint:left_hip_pitch_joint"]}
    tl.assert_layout_is_usable(meta, source="<synthetic>")


def test_the_minimal_schema_still_covers_every_metadata_key() -> None:
    """It stays the schema: ``_META_KEYS`` is derived from it, so it cannot shrink."""
    assert set(tl._META_KEYS) == set(tl._MINIMAL_META)
    for key in ("dt", "num_steps", "clip_length_s", "pos_output_names", "vel_output_names",
                "contact_frames", "ref_frame", "skill_list"):
        assert key in tl._META_KEYS, f"{key} dropped from the clip-free library's schema"


def test_the_finetune_builder_defaults_like_to_a_real_library() -> None:
    """The builder that paid for this must not leave the flag optional-and-silent."""
    sys.path.insert(0, str(_ROOT / "scripts" / "flow_matching"))
    mft = pytest.importorskip("make_finetune_terrain")
    assert mft.DEFAULT_LIKE, "make_finetune_terrain must name a default --like library"
    assert mft.DEFAULT_LIKE.endswith(".h5")
    # The default is a repo-relative path the builder opens directly; if it has moved, say so
    # here rather than at the next build.
    if os.path.isdir(_ROOT / "clips"):
        assert (_ROOT / mft.DEFAULT_LIKE).exists(), (
            f"make_finetune_terrain.DEFAULT_LIKE points at {mft.DEFAULT_LIKE}, which is not "
            "in clips/ -- pick a library that is, or every build fails at open()")
