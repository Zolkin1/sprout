# Copyright 2026 Zachary Olkin. All rights reserved.

"""Exported generators are named for their checkpoint, and resolved by name.

The behaviour being pinned is mostly a REFUSAL: with several generators in one ``exported/``
directory, silently loading the newest would put back exactly the ambiguity that naming them
removed -- and it would do it as a default, where nobody looks.

Also pinned: ``fm.onnx`` still resolves. It is the DEPLOYMENT name (the robot installs one
pair and its controller reads a fixed filename), so it has to keep working as an input even
though nothing writes it locally any more.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.artifacts import (  # noqa: E402
    DEPLOY_FM_NAME,
    fm_artifact_name,
    list_fm_onnx,
    resolve_fm_onnx,
)


def _touch(path: str, when: float | None = None) -> str:
    """Create an empty file, optionally with a specific mtime (to order a directory)."""
    with open(path, "w") as f:
        f.write("")
    if when is not None:
        os.utime(path, (when, when))
    return path


##
# The name
##

def test_best_checkpoint_is_named_for_its_run():
    """The run directory is the identity -- it is what every other artifact is filed under."""
    assert fm_artifact_name("logs/flow_matching/2026-08-12_17-07-47_scan_pin1/best.pt") == \
        "2026-08-12_17-07-47_scan_pin1_fm.onnx"


def test_a_non_best_checkpoint_keeps_its_stem():
    """Two epochs of one run must not collide, which is the same bug one level down."""
    assert fm_artifact_name("logs/flow_matching/2026-08-12_17-07-47_scan_pin1/ckpt_60.pt") == \
        "2026-08-12_17-07-47_scan_pin1_ckpt_60_fm.onnx"


def test_the_name_is_independent_of_how_the_path_was_written():
    """A relative path, a trailing './', an absolute one -- all the same artifact."""
    base = "logs/flow_matching/run_a/best.pt"
    assert fm_artifact_name(base) == fm_artifact_name("./" + base) \
        == fm_artifact_name(os.path.abspath(base))


##
# The resolution
##

def test_one_generator_needs_no_flag(tmp_path):
    """The common case stays a bare --load_run."""
    want = _touch(str(tmp_path / "run_a_fm.onnx"))
    assert resolve_fm_onnx(str(tmp_path)) == want


def test_the_deploy_name_still_resolves(tmp_path):
    """``fm.onnx`` is what a robot installs; it must remain loadable."""
    want = _touch(str(tmp_path / DEPLOY_FM_NAME))
    assert resolve_fm_onnx(str(tmp_path)) == want


def test_several_generators_refuse_to_be_guessed(tmp_path):
    """THE point of the change: no silent default, and the message lists the candidates."""
    _touch(str(tmp_path / "run_a_fm.onnx"), when=1_700_000_000)
    _touch(str(tmp_path / "run_b_fm.onnx"), when=1_800_000_000)   # newest
    with pytest.raises(ValueError, match=r"holds 2 generators"):
        resolve_fm_onnx(str(tmp_path))


def test_naming_one_resolves_it(tmp_path):
    """A bare filename is resolved inside the directory, not just an absolute path."""
    _touch(str(tmp_path / "run_a_fm.onnx"))
    want = _touch(str(tmp_path / "run_b_fm.onnx"))
    assert resolve_fm_onnx(str(tmp_path), "run_b_fm.onnx") == want
    assert resolve_fm_onnx(str(tmp_path), want) == want


def test_a_named_generator_that_does_not_exist_is_an_error(tmp_path):
    """Better than falling back to the directory, which would load a different model."""
    _touch(str(tmp_path / "run_a_fm.onnx"))
    with pytest.raises(FileNotFoundError, match=r"no generator at"):
        resolve_fm_onnx(str(tmp_path), "typo_fm.onnx")


def test_an_empty_directory_can_be_optional(tmp_path):
    """A tracker-only deployment (clip playback) legitimately has no generator."""
    assert resolve_fm_onnx(str(tmp_path), required=False) is None
    with pytest.raises(FileNotFoundError, match=r"no generator in"):
        resolve_fm_onnx(str(tmp_path))


def test_listing_is_ordered_and_deduplicated(tmp_path):
    """``list_fm_onnx`` backs the error message, so its order has to be stable."""
    old = _touch(str(tmp_path / "run_a_fm.onnx"), when=1_700_000_000)
    new = _touch(str(tmp_path / DEPLOY_FM_NAME), when=1_800_000_000)
    assert list_fm_onnx(str(tmp_path)) == [old, new]


def test_unrelated_onnx_files_are_not_generators(tmp_path):
    """``tracker.onnx`` shares the directory and must never be offered as a generator."""
    _touch(str(tmp_path / "tracker.onnx"))
    _touch(str(tmp_path / "policy.pt"))
    want = _touch(str(tmp_path / "run_a_fm.onnx"))
    assert resolve_fm_onnx(str(tmp_path)) == want
