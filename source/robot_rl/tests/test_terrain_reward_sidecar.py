# Copyright 2026 Zachary Olkin. All rights reserved.

"""The terrain reward reaches the critic by KEY, and "terrain only" is checked not trusted.

Two failure modes here, and neither would raise on its own:

* a per-window reward sidecar joined by POSITION rather than by
  ``(episode_uid, replan_index)`` pairs every window with some other window's reward. The
  critic then trains happily on a shuffled target.
* an advantage whose return ``G`` is built from the terrain reward while its baseline
  ``G_hat`` was fitted on the tracking reward. ``G_terrain - G_hat_tracking`` has the right
  shape and the right units and means nothing.

So the join is by key and the reward source is stamped at fit time and checked at label time.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "flow_matching"


def _load_script(name: str):
    """Import a ``scripts/flow_matching`` module by path, without running its ``main``."""
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def label_mod():
    """``label_advantage``, for its guard."""
    return _load_script("label_advantage")


@pytest.fixture(scope="module")
def fit_mod():
    """``fit_value``, for its sidecar loader."""
    return _load_script("fit_value")


# -----------------------------------------------------------------------------------------
# the sidecar join
# -----------------------------------------------------------------------------------------
def _sidecar(tmp_path, uid, ridx, reward, source="terrain"):
    """Write a minimal sidecar in the shape ``--stage reward`` emits."""
    path = tmp_path / "sidecar.npz"
    np.savez_compressed(
        path, episode_uid=np.asarray(uid, np.int64),
        replan_index=np.asarray(ridx, np.int32), reward=np.asarray(reward, np.float64),
        meta_json=json.dumps({"reward_source": source, "rollout": "/some/stem"}),
        reward_hash="deadbeef")
    return str(path)


def test_the_reward_is_joined_by_key_and_survives_a_reordering(fit_mod, tmp_path, capsys):
    """The whole point: a positional join would give every window the wrong reward."""
    uid = np.array([7, 7, 7, 9, 9], np.int64)
    ridx = np.array([0, 1, 2, 0, 1], np.int32)
    reward = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    path = _sidecar(tmp_path, uid, ridx, reward)

    # Ask in a DIFFERENT order from the one the sidecar was written in.
    perm = np.array([3, 0, 4, 2, 1])
    got, ok, meta = fit_mod._reward_sidecar(path, uid[perm], ridx[perm])
    assert bool(ok.all())
    assert np.allclose(got, reward[perm]), "joined by position, not by key"
    assert meta["reward_source"] == "terrain"
    assert meta["reward_hash"] == "deadbeef"


def test_windows_the_sidecar_does_not_cover_are_flagged_not_zero_filled(fit_mod, tmp_path):
    """The sidecar is written from a subset, so partial coverage is normal and must be VISIBLE."""
    path = _sidecar(tmp_path, [7, 7], [0, 1], [0.25, 0.75])
    got, ok, _meta = fit_mod._reward_sidecar(
        path, np.array([7, 7, 7], np.int64), np.array([0, 1, 5], np.int32))
    assert ok.tolist() == [True, True, False]
    assert np.allclose(got[:2], [0.25, 0.75])
    assert got[2] == 0.0


def test_a_sidecar_for_a_different_rollout_is_refused(fit_mod, tmp_path):
    """An empty join would fit the critic on nothing but zeros, without complaint."""
    path = _sidecar(tmp_path, [1, 1], [0, 1], [0.5, 0.5])
    with pytest.raises(SystemExit, match="covers none"):
        fit_mod._reward_sidecar(path, np.array([99, 99], np.int64),
                                np.array([0, 1], np.int32))


# -----------------------------------------------------------------------------------------
# the "terrain only" guard
# -----------------------------------------------------------------------------------------
def test_terrain_only_accepts_the_one_consistent_combination(label_mod, capsys):
    """w_fail 0 + mc + a critic fitted on terrain, and nothing else."""
    label_mod.check_reward_consistency("terrain", 0.0, "mc", True, "abc123")
    assert "TERRAIN ONLY" in capsys.readouterr().out


def test_terrain_only_refuses_a_critic_fitted_on_the_TRACKING_reward(label_mod):
    """The dangerous one: nothing about this mismatch shows up in the numbers."""
    with pytest.raises(SystemExit, match="fitted on 'track_err'"):
        label_mod.check_reward_consistency("track_err", 0.0, "mc", True)


def test_terrain_only_refuses_a_nonzero_survival_weight(label_mod):
    """At w_fail > 0 the advantage mixes in the realized fall outcome."""
    with pytest.raises(SystemExit, match="w_fail 0"):
        label_mod.check_reward_consistency("terrain", 1.0, "mc", True)
    with pytest.raises(SystemExit, match="w_fail 0"):
        label_mod.check_reward_consistency("terrain", 0.25, "mc", True)


def test_terrain_only_refuses_the_bootstrapped_estimator(label_mod):
    """n_step's quality half differences against the critic, not against measured reward."""
    with pytest.raises(SystemExit, match="estimator mc"):
        label_mod.check_reward_consistency("terrain", 0.0, "n_step", True)


def test_a_terrain_critic_used_WITHOUT_terrain_only_warns(label_mod, capsys):
    """Silently mixing is the mistake; it should at least say so."""
    label_mod.check_reward_consistency("terrain", 1.0, "n_step", False)
    assert "WARNING" in capsys.readouterr().out


def test_the_default_path_is_unaffected(label_mod, capsys):
    """A tracking critic labelled the old way must stay silent, or the warning is noise."""
    label_mod.check_reward_consistency("track_err", 1.0, "n_step", False)
    assert capsys.readouterr().out == ""


# -----------------------------------------------------------------------------------------
# a quality-only critic
# -----------------------------------------------------------------------------------------
def test_a_quality_only_critic_refuses_any_survival_weight(label_mod):
    """Its ``p_fail`` was never trained, so reading it at all is a bug, not a setting.

    Refused rather than warned: an untrained sigmoid head still emits numbers in [0, 1] that
    look exactly like probabilities, so nothing downstream would notice.
    """
    for w in (1.0, 0.5, 0.01):
        with pytest.raises(SystemExit, match="UNTRAINED"):
            label_mod.check_reward_consistency("terrain", w, "mc", False, quality_only=True)


def test_a_quality_only_critic_is_fine_at_w_fail_zero(label_mod, capsys):
    """Which is exactly the terrain-only configuration."""
    label_mod.check_reward_consistency("terrain", 0.0, "mc", True, "abc",
                                        (-1.0, 0.0), "mse", True)
    out = capsys.readouterr().out
    assert "TERRAIN ONLY" in out
    assert "survival head UNTRAINED" in out
    assert "[-1, 0]" in out


def test_the_range_and_kind_are_reported_so_a_mismatch_is_visible(label_mod, capsys):
    """[0,1] and [-1,0] are both plausible-looking; an advantage across them is nonsense."""
    label_mod.check_reward_consistency("terrain", 0.0, "mc", True, "h",
                                        (0.0, 1.0), "two_hot", False)
    out = capsys.readouterr().out
    assert "two_hot head over [0, 1]" in out
    assert "UNTRAINED" not in out


def test_fit_value_offers_quality_only_and_drops_the_bce():
    """The loss must be dropped, not down-weighted: a small weight still shapes the encoder."""
    src = (_SCRIPTS / "fit_value.py").read_text()
    assert '"--quality_only"' in src
    assert "torch.zeros((), device=device) if quality_only" in src
    assert "quality_only=bool(args.quality_only)" in src
    # ...and the model must not be selected on an untrained head's AUC.
    assert 'select_name = "val quality MSE" if quality_only else "val AUC"' in src

