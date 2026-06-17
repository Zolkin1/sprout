# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for ``BaseTrajectoryCommand._record_library_fingerprint``.

Covers the play / resume provenance path: warn on a library that differs from
the trained-against value, stay silent on a match, no-op when nothing was
recorded, and always write the locally-loaded fingerprint back onto the cfg
(so a fresh training run serialises it into ``params/env.yaml``).
"""

import types

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.base_trajectory_cmd import (
    BaseTrajectoryCommand,
)


def _record(trained_hash, local_hash):
    """Invoke the (unbound) recorder with stub manager/cfg; return the cfg.

    Avoids constructing a full command (which needs a live env) by calling the
    method against lightweight stand-ins that expose only what it touches.
    """
    manager = types.SimpleNamespace(
        library_fingerprint={
            "hash": local_hash,
            "short": local_hash[:5],
            "num_trajectories": 3,
            "skill_counts": {"walk_forward": 2, "running": 1},
        }
    )
    cfg = types.SimpleNamespace(
        path="trajectories/retargeted/lib",
        library_fingerprint=trained_hash,
        library_manifest=None,
    )
    me = types.SimpleNamespace(manager=manager)
    BaseTrajectoryCommand._record_library_fingerprint(me, cfg)
    return cfg


def test_warns_on_mismatch(capsys):
    """A different trained-against hash prints the mismatch banner."""
    _record("trained_aaa", "local_bbb")
    out = capsys.readouterr().out
    assert "TRAJECTORY LIBRARY MISMATCH" in out
    assert "trained_aaa" in out and "local_bbb" in out


def test_silent_on_match(capsys):
    """Matching hashes produce no warning."""
    _record("same_hash", "same_hash")
    assert "MISMATCH" not in capsys.readouterr().out


def test_silent_when_unrecorded(capsys):
    """An unset trained-against hash (old run) is a no-op comparison."""
    _record(None, "local_bbb")
    assert "MISMATCH" not in capsys.readouterr().out


def test_writes_back_local_fingerprint():
    """The locally-loaded hash + manifest are always written onto the cfg."""
    cfg = _record(None, "local_bbb")
    assert cfg.library_fingerprint == "local_bbb"
    assert cfg.library_manifest == {"walk_forward": 2, "running": 1}


def test_noop_without_manager_fingerprint():
    """Managers without a fingerprint (single-file TrajectoryManager) are skipped."""
    cfg = types.SimpleNamespace(library_fingerprint="trained_aaa", library_manifest=None)
    me = types.SimpleNamespace(manager=types.SimpleNamespace())  # no library_fingerprint
    BaseTrajectoryCommand._record_library_fingerprint(me, cfg)
    assert cfg.library_fingerprint == "trained_aaa"  # untouched
