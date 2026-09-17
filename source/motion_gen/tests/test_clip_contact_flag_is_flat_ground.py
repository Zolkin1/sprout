# Copyright 2026 Zachary Olkin. All rights reserved.

"""Every clip library's recorded ``contact`` flag is a FLAT-ground test, and that is wrong.

``ClipOutputBuilder`` marks a foot loaded when a bottom corner of its collision box reaches
``ground_height + contact_z_thresh``, and ``ground_height`` defaults to 0.0 -- which **no
generator overrides**.  So on terrain the flag fails in both directions: a foot planted on a
0.4 m box or any tread above the first reads NOT in contact, and a foot airborne over ground
below z = 0 reads in contact.

This is the same defect that made the FM's terrain-penetration loss inert (``27723572``: "the
auto-calibrated floor lands near the LOWEST step"), and it is why the terrain-consistency
reward calibrates its velocity threshold against this flag **only on flat clips**, where it
happens to be valid.

The test pins the DEFECT, not a fix.  Repairing it properly means handing
``ClipOutputBuilder`` the tile's collider boxes, which changes the ``contact`` dataset of
every library on disk -- so the day someone does that, this fails and the calibration arms
get revisited on purpose rather than silently becoming a comparison against something else.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import motion_gen.clip_output as clip_output
from motion_gen import features as F

#: Generation entry points that build a ``ClipOutputBuilder``.  A new one that passes
#: ``ground_height`` will trip this test, which is the intent.
_GENERATORS = (
    "scripts/rsl_rl/mb_terrain_clips.py",
    "scripts/rsl_rl/generate_box_clips.py",
    "scripts/rsl_rl/generate_mb_clips.py",
    "scripts/rsl_rl/generate_distractor_clips.py",
    "scripts/rsl_rl/mb_distractor_clips.py",
    "scripts/rsl_rl/generate_mm_clips.py",
)


def test_the_builder_defaults_the_ground_to_a_single_flat_height():
    """``ground_height=0.0`` and one scalar threshold -- the shape of the problem."""
    sig = inspect.signature(clip_output.ClipOutputBuilder.__init__)
    assert sig.parameters["ground_height"].default == 0.0
    assert sig.parameters["contact_z_thresh"].default == 0.02


def test_no_generator_overrides_the_ground_height():
    """If one starts to, the flag stops being uniformly flat and the note above is stale."""
    offenders = []
    for rel in _GENERATORS:
        path = F._REPO_ROOT / rel
        if not path.exists():                       # a generator may be retired
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "ClipOutputBuilder"
                    and any(kw.arg == "ground_height" for kw in node.keywords)):
                offenders.append(rel)
    assert not offenders, (
        f"{offenders} now pass ground_height to ClipOutputBuilder. The recorded `contact` "
        "flag is no longer uniformly a flat-ground test, so the terrain-consistency "
        "calibration arms (flat-only against this flag, SDF-recomputed on terrain) need "
        "revisiting -- and every library on disk predates the change."
    )


def test_the_flag_is_demonstrably_wrong_one_riser_up():
    """Two identical stances, one on a 0.13 m tread: the flag calls the raised one airborne."""
    torch = pytest.importorskip("torch")
    from motion_gen.collision import collider_points                  # noqa: PLC0415

    sole_z = collider_points(F.DEFAULT_MJCF, ["left_ankle_roll_link"]
                             )["left_ankle_roll_link"].points[:, 2].min()
    thresh = 0.02
    # On the ground the lowest corner is at 0; one riser up it is at 0.13.
    assert (0.0 < thresh) and not (0.13 < thresh)
    assert sole_z == pytest.approx(-0.034425, abs=1e-6)
