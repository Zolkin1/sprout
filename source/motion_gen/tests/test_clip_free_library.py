# Copyright 2026 Zachary Olkin. All rights reserved.

"""A library with ZERO clips is a well-defined library, and every gather on it refuses.

The free-form fine-tune passes ``--clip`` a ``.h5`` carrying a terrain manifest and no motion:
its reference is predicted from the depth cameras, so nothing is ever read off disk. That file
still has to be a real clip library on every axis except the clips themselves, because
:class:`~motion_gen.clips.clip_manager.MotionClipManager` is built before anything knows whether
a clip will be read.

**The defect this pins.** The generator used to simply OMIT the clip datasets, the loader's
placeholder for an absent dataset is a rank-1 ``torch.empty(0)``, and ``num_steps`` is
``y_des.shape[1]`` -- so building the manager on a real generated library raised

    clip_manager.py:56   self.clip_len = lib.clip_len_or_full(self.device)
    clip_io.py:263       return torch.full((self.num_clips,), self.num_steps, ...)
    clip_io.py:240       return self.y_des.shape[1]
    IndexError: tuple index out of range

three frames deep, with nothing in the traceback naming the library or the task. The registered
task ``G1-teacher-clf-terrain-depth-finetune-freeform`` could not be constructed at all.

The other half matters as much: a zero-clip library that quietly returned EMPTY tensors from a
sampler would be worse than the ``IndexError``, because a mis-wired clip term would then present
as a reference that never moves. So each gather refuses by name.

No simulator and no IsaacLab here -- this is all ``motion_gen``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motion_gen.clips.clip_io import (  # noqa: E402
    ClipLibraryWriter,
    group_clips_by_terrain,
    load_clip_library,
)
from motion_gen.clips.clip_manager import ClipFreeLibraryError, MotionClipManager  # noqa: E402

# <repo>/source/motion_gen/tests/<this> -> parents[3] == <repo>. Derived, never counted.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The channel layout a terrain-only library carries. Deliberately NOT imported from the maker
#: script: this test is about what ``ClipLibraryWriter`` does with ``num_clips=0``, and reusing
#: the script's constant would make the two move together and hide a change in either.
META: dict = {
    "dt": 0.02,
    "num_steps": 37,
    "clip_length_s": 0.74,
    "pos_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
                         "pelvis_link:ori_w", "pelvis_link:ori_x", "pelvis_link:ori_y",
                         "pelvis_link:ori_z"],
    "vel_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
                         "pelvis_link:ori_x", "pelvis_link:ori_y", "pelvis_link:ori_z"],
    "contact_frames": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "ref_frame": "pelvis_link",
    "skill_list": ["standing"],
}

T = META["num_steps"]
P = len(META["pos_output_names"])
V = len(META["vel_output_names"])
C = len(META["contact_frames"])


def _write_empty(path: str) -> str:
    """Write a clip-free library through the shared writer.

    Args:
        path: Output ``.h5``.

    Returns:
        ``path``, for chaining.
    """
    with ClipLibraryWriter(path, num_clips=0, meta=META) as w:
        w.set_attr("terrain_manifest", [{"terrain_id": 0, "overlays": []}])
    return path


def _write_with_clips(path: str, n: int = 3) -> str:
    """Write an ordinary library of ``n`` clips, as the control arm.

    Every assertion about the clip-free path has a twin here: the point is that nothing about
    a library WITH motion changed, so a regression in the shared writer shows up as a failure
    in this file rather than in a training run.

    Args:
        path: Output ``.h5``.
        n: Clip count.

    Returns:
        ``path``.
    """
    with ClipLibraryWriter(path, num_clips=n, meta=META) as w:
        w.write_batch(0, {
            "y_des": torch.arange(n * T * P, dtype=torch.float32).reshape(n, T, P),
            "dy_des": torch.zeros(n, T, V),
            "contact": torch.ones(n, T, C, dtype=torch.uint8),
            "skill_id": torch.zeros(n, T, dtype=torch.uint8),
            "stride_idx": torch.zeros(n, T, dtype=torch.int32),
            "target_vel": torch.ones(n, T, 3),
            "terrain_id": torch.full((n,), -1, dtype=torch.int32),
            "clip_len": torch.full((n,), T, dtype=torch.int32),
        })
    return path


# ---------------------------------------------------------------------------------------------
# The file on disk
# ---------------------------------------------------------------------------------------------
def test_the_writer_keeps_the_rank_of_every_clip_dataset(tmp_path):
    """Zero clips, full rank. The rank is what ``num_steps`` is read off.

    h5py refuses the per-clip ``(1, ...)`` chunk against a zero-extent axis, which is why these
    datasets used to be omitted rather than emitted empty. Dropping the CHUNKING is the correct
    concession (there is nothing to chunk); dropping the SHAPES was not.
    """
    lib = load_clip_library(_write_empty(str(tmp_path / "empty.h5")))
    assert lib.y_des.shape == (0, T, P)
    assert lib.dy_des.shape == (0, T, V)
    assert lib.contact.shape == (0, T, C)
    assert lib.target_vel.shape == (0, T, 3)
    assert lib.skill_id.shape == (0, T)
    assert lib.terrain_id.shape == (0,)
    assert lib.clip_len.shape == (0,)


def test_a_clip_free_library_knows_its_own_length_and_says_it_has_no_clips(tmp_path):
    """``num_steps`` is a real number and ``has_clips`` is False.

    This is the assertion that raised ``IndexError`` before the fix.
    """
    lib = load_clip_library(_write_empty(str(tmp_path / "empty.h5")))
    assert lib.num_clips == 0
    assert lib.has_clips is False
    assert lib.num_steps == T


def test_the_length_falls_back_to_the_metadata_when_the_shape_cannot_state_it(tmp_path):
    """A ``keys=`` subset load leaves un-requested datasets rank-1, and must still answer.

    The same hole the clip-free library fell into, reached from the other direction -- and this
    one is on the LIVE path: ``robot_rl.flow_matching.dataset`` loads libraries with a key
    subset.
    """
    lib = load_clip_library(_write_with_clips(str(tmp_path / "clips.h5")), keys=["target_vel"])
    assert lib.y_des.ndim == 1                    # not requested -> placeholder
    assert lib.num_steps == T                     # ...and T is still known


def test_a_library_with_clips_is_still_chunked_and_compressed(tmp_path):
    """The control arm: the concession is scoped to ``num_clips == 0`` and nowhere else."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(_write_with_clips(str(tmp_path / "clips.h5")), "r") as f:
        assert f["y_des"].chunks == (1, T, P)
        assert f["y_des"].compression == "lzf"
    with h5py.File(_write_empty(str(tmp_path / "empty.h5")), "r") as f:
        assert f["y_des"].chunks is None
        assert f["y_des"].compression is None


# ---------------------------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------------------------
def test_the_manager_constructs_and_answers_every_layout_question(tmp_path):
    """The whole point: the command term builds one of these before reading any clip.

    Drives each accessor the construction path actually reaches, so "it constructs" is not
    mistaken for "it works" -- a manager that raised on ``order_outputs`` would be no more
    usable than one that raised in ``__init__``.
    """
    manager = MotionClipManager(load_clip_library(_write_empty(str(tmp_path / "empty.h5"))))

    assert manager.has_clips is False
    assert manager.num_clips == 0
    assert manager.num_steps == T
    assert manager.dt == META["dt"]
    assert manager.ref_frame == META["ref_frame"]
    assert manager.get_pos_output_names == META["pos_output_names"]
    assert manager.get_vel_output_names == META["vel_output_names"]
    assert manager.get_num_pos_outputs() == P
    assert manager.get_num_vel_outputs() == V
    assert manager.get_reference_frames() == [META["ref_frame"]]
    assert manager.get_num_domains() == 1
    assert float(manager.get_total_time()) == pytest.approx(T * META["dt"])
    assert manager.has_target_vel() is False

    # The per-clip tables are empty but SHAPED, so anything that reduces over them is defined.
    assert manager.anchor_mask.shape == (0,)
    assert manager.clip_len.shape == (0,)
    assert manager.anchor_mask_t.shape == (0, T)
    assert manager.terrain_start_step.shape == (0,)
    assert manager.terrain_end_step.shape == (0,)

    # Reordering the output columns is done at command construction, on the real arrays.
    manager.order_outputs(list(reversed(META["pos_output_names"])), META["vel_output_names"])
    assert manager.pos_output_names == list(reversed(META["pos_output_names"]))
    assert manager.lib.y_des.shape == (0, T, P)
    manager.invalidate_cache()


@pytest.mark.parametrize("call", [
    lambda m: m.gather_desired(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)),
    lambda m: m.gather_contact(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)),
    lambda m: m.gather_target_vel(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)),
    lambda m: m.gather_anchor_mask(torch.zeros(2, dtype=torch.long)),
    lambda m: m.gather_anchor_mask_at_step(torch.zeros(2, dtype=torch.long),
                                           torch.zeros(2, dtype=torch.long)),
    lambda m: m.gather_terrain_start_step(torch.zeros(2, dtype=torch.long)),
    lambda m: m.gather_terrain_end_step(torch.zeros(2, dtype=torch.long)),
    lambda m: m.gather(m.lib.y_des, torch.zeros(2, dtype=torch.long),
                       torch.zeros(2, dtype=torch.long)),
])
def test_every_gather_refuses_and_names_the_task(tmp_path, call):
    """No gather may return an empty tensor, and none may raise a bare ``IndexError``.

    The message has to carry the way out, because the reader of it is someone who passed a
    perfectly good terrain library to a task that still has a clip term wired on.
    """
    manager = MotionClipManager(load_clip_library(_write_empty(str(tmp_path / "empty.h5"))))
    with pytest.raises(ClipFreeLibraryError) as e:
        call(manager)
    assert "finetune-freeform" in str(e.value)
    assert "make_freeform_terrain" in str(e.value)


def test_an_empty_index_would_otherwise_have_returned_silently(tmp_path):
    """The refusal fires on an EMPTY index too -- the case that would not have errored at all.

    ``_reanchor_clip_ids`` masks its ids and can legitimately pass none, so an empty gather is
    reachable. Without the check it returns an empty result and the caller carries on.
    """
    manager = MotionClipManager(load_clip_library(_write_empty(str(tmp_path / "empty.h5"))))
    empty = torch.zeros(0, dtype=torch.long)
    with pytest.raises(ClipFreeLibraryError):
        manager.gather_desired(empty, empty)


def test_a_manager_with_clips_still_gathers(tmp_path):
    """The control arm for the refusal: nothing about a real library's gather moved."""
    manager = MotionClipManager(load_clip_library(_write_with_clips(str(tmp_path / "c.h5"))))
    assert manager.has_clips is True
    y, dy = manager.gather_desired(torch.tensor([0, 2]), torch.tensor([1, 5]))
    assert y.shape == (2, P) and dy.shape == (2, V)
    assert float(y[0, 0]) == pytest.approx(1 * P)          # clip 0, step 1, channel 0
    assert manager.has_target_vel() is True


def test_a_library_that_disagrees_with_itself_about_its_length_is_refused(tmp_path):
    """``T`` is stated by the metadata AND by the step axis; they must match.

    Only the metadata survives on a clip-free library, so the metadata is what the manager
    reads -- which makes a disagreement on a library WITH clips a silent out-of-range index
    somewhere else entirely.
    """
    h5py = pytest.importorskip("h5py")
    import json
    path = _write_with_clips(str(tmp_path / "c.h5"))
    with h5py.File(path, "r+") as f:
        f.attrs["num_steps"] = json.dumps(T + 5)
    with pytest.raises(ValueError, match="disagrees with itself about its length"):
        MotionClipManager(load_clip_library(path))


def test_grouping_by_terrain_names_the_clip_free_case_rather_than_the_stale_one(tmp_path):
    """Both present as an empty ``terrain_id`` and the two fixes are opposite.

    A pre-terrain library must be REGENERATED; a free-form one is correct and simply has no
    clips, so telling its user to regenerate would send them the wrong way.
    """
    lib = load_clip_library(_write_empty(str(tmp_path / "empty.h5")))
    with pytest.raises(ValueError) as e:
        group_clips_by_terrain(lib)
    assert "ZERO clips" in str(e.value)
    assert "generate_mb_library" not in str(e.value)


# ---------------------------------------------------------------------------------------------
# The real generated library, end to end
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_freeform_library(tmp_path_factory) -> str:
    """A library written by the actual CLI, not by this file's hand-rolled writer.

    Generated rather than fixtured for the reason ``test_freeform_env_cfg`` states: a checked-in
    file would freeze one half of the very agreement under test. This is the arm that would have
    caught the defect -- the hand-rolled writer above is a statement about
    ``ClipLibraryWriter``, and the CLI is what a user runs.
    """
    ff = pytest.importorskip("motion_gen.terrain.freeform")
    maker_path = _REPO_ROOT / "scripts" / "rsl_rl" / "make_freeform_terrain.py"
    spec = importlib.util.spec_from_file_location("make_freeform_terrain", maker_path)
    maker = importlib.util.module_from_spec(spec)
    sys.modules["make_freeform_terrain"] = maker
    spec.loader.exec_module(maker)

    cfg = ff.FreeformConfig(seed=3, num_flat=2, num_boxes=2, num_stairs=2)
    manifest = ff.generate_freeform_manifest(cfg)
    ff.validate_manifest(manifest)
    path = str(tmp_path_factory.mktemp("freeform") / "lib.h5")
    # The bare SCHEMA, deliberately: this file never builds a trajectory command, so it
    # needs no joint channels. `library_meta` refuses None because a COURSE built that
    # way dies inside Isaac; that refusal is right, and this is the honest way to ask
    # for the layout it does not apply to.
    maker.write_terrain_library(path, manifest, maker.schema_only_layout(), cfg)
    return path


def test_the_real_generated_library_builds_a_manager(real_freeform_library):
    """The exact reproduction from the bug report, now passing.

    ``load_clip_library(...)`` then ``MotionClipManager(lib)`` on the file the shipped CLI
    writes. Everything else in this file is a statement about the pieces; this is the one that
    says the deliverable is constructible.
    """
    lib = load_clip_library(real_freeform_library, "cpu")
    assert lib.num_clips == 0
    assert lib.y_des.ndim == 3, "the CLI must emit shape-correct empty clip datasets"

    manager = MotionClipManager(lib)
    assert manager.has_clips is False
    assert manager.num_steps == lib.num_steps
    assert manager.terrain_start_step.numel() == 0
    with pytest.raises(ClipFreeLibraryError):
        manager.gather_desired(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))


def test_the_real_generated_library_still_carries_its_terrain(real_freeform_library):
    """Nothing about the manifest half moved when the clip half was given shapes."""
    lib = load_clip_library(real_freeform_library, "cpu")
    manifest = lib.attrs["terrain_manifest"]
    assert len(manifest) == 6
    assert all("route" in m for m in manifest)
    assert np.asarray(lib.terrain_id).size == 0
