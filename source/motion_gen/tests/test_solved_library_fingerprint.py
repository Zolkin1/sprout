# Copyright 2026 Zachary Olkin. All rights reserved.

"""A clip library must record WHICH solved traj-opt library produced it.

Only the DEPRECATED motion-matching generators ever recorded a source fingerprint, so the live
Motion Bricks chain produced libraries that could not answer "which solved library is this?".
That is a problem now rather than in principle, for a specific reason:
``hf_cache.download_from_hf`` reuses whatever is already in the local ``hf/`` cache and never
re-checks the remote, so two machines can hold DIFFERENT libraries under one path and nothing
downstream would notice.

The function is deliberately in ``clips.clip_fingerprint``, which is stdlib-only so the standalone
checker can import it under a bare ``python3``. That property is load-bearing and is tested here.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from motion_gen.clips.clip_fingerprint import (fingerprint_solved_library,
                                               verify_solved_fingerprint)


def _repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "scripts" / "rsl_rl").is_dir() and (parent / "source" / "motion_gen").is_dir():
            return parent
    raise RuntimeError(f"no checkout above {__file__}")


_REPO = _repo_root()
_LIB = "trajectories/skills/library_v2"
_have_lib = (_REPO / "hf" / _LIB).is_dir()


def test_clip_fingerprint_stays_stdlib_only():
    """The module must not pull in numpy/torch/h5py.

    ``scripts/rsl_rl/clip_library_fingerprint.py`` documents that it runs "under a bare python3
    without the IsaacLab environment". Importing anything heavy here would quietly break that, and
    the breakage would only show on a machine that has no venv -- i.e. exactly where the checker is
    most useful.
    """
    code = ("import sys; import motion_gen.clips.clip_fingerprint as m; "
            "print([x for x in ('numpy','torch','h5py','scipy') if x in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": str(_REPO / "source" / "motion_gen"), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"clip_fingerprint pulled in heavy modules: {out.stdout}"


def test_a_missing_fingerprint_is_not_a_failure():
    """Libraries built before this was wired carry none; refusing them would be a migration gate.

    This mirrors the rule the robot-asset fingerprint already follows: a missing side is silent
    because it guards against a NEW mismatch, not against history.
    """
    cur = {"hash": "abc", "short": "abc", "source": "x", "num_clips": 1}
    ok, msg = verify_solved_fingerprint(None, cur)
    assert ok and "predates" in msg


def test_an_EMPTY_fingerprint_is_treated_as_absent_not_as_a_match():
    """"absent" and "empty" must not become the same thing.

    An empty dict compared field-by-field would equal anything, which is the failure mode the
    robot-asset fingerprint calls out explicitly: an empty fingerprint that compares equal is worse
    than no fingerprint at all.
    """
    cur = {"hash": "abc", "short": "abc", "source": "x", "num_clips": 1}
    ok, msg = verify_solved_fingerprint({}, cur)
    assert ok and "predates" in msg
    # ...and it must NOT report a match, which would be an affirmative false claim.
    assert "matches" not in msg


def test_a_different_library_is_a_mismatch():
    a = {"hash": "aaa", "short": "aaa", "source": "library_v2", "num_clips": 307}
    b = {"hash": "bbb", "short": "bbb", "source": "library_v1", "num_clips": 272}
    ok, msg = verify_solved_fingerprint(a, b)
    assert not ok
    assert "library_v2" in msg and "library_v1" in msg, "the message must name BOTH sides"


@pytest.mark.skipif(not _have_lib, reason="library_v2 not in the local hf/ cache")
def test_the_real_library_hashes_stably_and_is_cwd_independent():
    """Same answer from any working directory.

    The cache used to be anchored on ``os.getcwd()``, so one call from a subdirectory downloaded a
    second full copy (308 files, 103 MB) into ``source/motion_gen/hf`` and then fingerprinted THAT.
    A cache that moves with the cwd is a cache that cannot be reasoned about.
    """
    a = fingerprint_solved_library(_LIB)
    b = fingerprint_solved_library(_LIB)
    assert a == b, "not deterministic"
    assert a["num_clips"] > 0 and len(a["hash"]) == 64

    code = ("from motion_gen.clips.clip_fingerprint import fingerprint_solved_library as f; "
            f"print(f({_LIB!r})['hash'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(_REPO / "scripts" / "rsl_rl"),
                         env={"PYTHONPATH": str(_REPO / "source" / "motion_gen"), "PATH": "/usr/bin:/bin",
                              "HOME": str(pathlib.Path.home())})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == a["hash"], "fingerprint depends on the cwd"
    assert not (_REPO / "scripts" / "rsl_rl" / "hf").exists(), "a stray cache was created"
