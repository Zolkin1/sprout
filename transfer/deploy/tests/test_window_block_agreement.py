# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generator and the tracker must pack the SAME reference-window blocks.

Artifact-free: :meth:`GenTrackerPolicy._build_window` reads only the tracker's exported
metadata and the FM context, both stubbed here, so this runs without an export under
``logs/`` (unlike ``test_hardware_offline`` / ``test_gen_tracker_obs_parity``, which need
one and skip without it). That matters because this guard is the last thing standing
between two independently-trained artifacts and a scrambled observation.

A teacher may drop reference-window blocks it does not need, and the generator it is paired
with has to predict exactly the ones that remain -- its output goes into that observation
verbatim, with nothing in between to translate. Two failure shapes:

* different widths -- caught eventually, but as a shape error deep in the obs assembly,
  far from the artifacts that disagree;
* the SAME width with different blocks (``joint_pos`` and ``joint_vel`` are both ``nj``
  wide) -- never caught at all. The policy reads velocities where it expects positions.

Run: ``python transfer/deploy/tests/test_window_block_agreement.py``
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.gen_tracker import _BASE_POSE_CONVENTION, GenTrackerPolicy  # noqa: E402
from deploy.window_np import window_blocks  # noqa: E402

NJ, K = 29, 4


def _policy(tracker_blocks, fm_blocks):
    """A bare policy whose two artifacts declare the given block selections.

    Args:
        tracker_blocks: What the tracker's ``reference_window_spec`` names.
        fm_blocks: What the FM context says the generator predicts, or ``None`` for
            tracker-only playback (no generator to disagree).

    Returns:
        An unloaded :class:`GenTrackerPolicy` carrying just what ``_build_window`` reads.
    """
    pol = object.__new__(GenTrackerPolicy)
    pol.fm_enabled = fm_blocks is not None
    pol.ctx = SimpleNamespace(reference_blocks=tuple(fm_blocks or ()))
    spec = {
        "num_steps": 11, "stride": 5, "offsets": [i * 5 for i in range(11)],
        "blocks": [[n, w] for n, w in window_blocks(NJ, K, tracker_blocks)],
        "joint_order": "isaac", "ee_frame_names": [],
        "base_pose_convention": _BASE_POSE_CONVENTION,
    }
    pol._trk = SimpleNamespace(
        get_obs_terms=lambda: [{"name": "reference_window"}],
        get_reference_window_spec=lambda: spec,
        get_command_window_spec=lambda: None,
    )
    return pol


def test_matching_selections_load():
    """The ordinary case: one teacher, one generator trained for it."""
    slim = ("base_pose", "base_vel", "joint_pos")
    pol = _policy(slim, slim)
    pol._build_window()
    assert pol.reference_blocks == slim
    assert pol.window_offsets == [i * 5 for i in range(11)]


@pytest.mark.parametrize("tracker_blocks,fm_blocks", [
    # Different widths: the generator predicts more than the policy reads.
    (("base_pose", "base_vel", "joint_pos"),
     ("base_pose", "base_vel", "joint_pos", "joint_vel")),
    # SAME width, different meaning -- joint_pos and joint_vel are both nj wide, so
    # nothing downstream can tell these apart.
    (("base_pose", "base_vel", "joint_pos"),
     ("base_pose", "base_vel", "joint_vel")),
])
def test_disagreeing_selections_are_refused(tracker_blocks, fm_blocks):
    """Either artifact can be the odd one out; both must fail at load, not at runtime."""
    pol = _policy(tracker_blocks, fm_blocks)
    with pytest.raises(ValueError, match="predicts window blocks"):
        pol._build_window()


def test_tracker_only_playback_has_nothing_to_agree_with():
    """Without a generator the clip reference drives directly; the check does not apply."""
    slim = ("base_pose", "base_vel", "joint_pos")
    pol = _policy(slim, None)
    pol._build_window()
    assert pol.reference_blocks == slim


if __name__ == "__main__":
    test_matching_selections_load()
    for tb, fb in [(("base_pose", "base_vel", "joint_pos"),
                    ("base_pose", "base_vel", "joint_pos", "joint_vel")),
                   (("base_pose", "base_vel", "joint_pos"),
                    ("base_pose", "base_vel", "joint_vel"))]:
        test_disagreeing_selections_are_refused(tb, fb)
    test_tracker_only_playback_has_nothing_to_agree_with()
    print("WINDOW-BLOCK AGREEMENT CHECKS PASSED")
