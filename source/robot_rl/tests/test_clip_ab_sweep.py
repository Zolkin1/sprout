# Copyright 2026 Zachary Olkin. All rights reserved.

"""The pieces that make an Isaac-vs-MuJoCo clip comparison sound.

Three things have to hold for the comparison in ``scripts/rsl_rl/clip_ab_sweep.py`` to mean
anything, and none of them is visible in the aggregate number if it breaks:

1. **Clip pinning** -- Isaac Lab samples ``clip_id`` at every reset, so without pinning the
   two simulators play different clips and the comparison is between different work.
2. **Per-env stats** -- two numbers (mean and std) are not a distribution. ``play_plots`` has
   to emit the per-ENV column behind each scalar, which with clips pinned is per-CLIP.
3. **The metric join** -- the driver reads Isaac MSE and MuJoCo RMS and has to put them in
   the same units before comparing, or one side reads systematically low.

The heavy halves (a GPU rollout, a MuJoCo rollout) are not tested here; these are the joins
between them, which is where a silent mistake would survive.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

REPO = pathlib.Path(__file__).resolve().parents[3]


def _load(name: str, path: pathlib.Path):
    """Import a script by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Clip pinning.
# ---------------------------------------------------------------------------
def _pin():
    """``resolve_pinned_clips``, skipped when isaaclab is unavailable."""
    resets = pytest.importorskip(
        "robot_rl.tasks.manager_based.robot_rl.mdp.events.resets", exc_type=ImportError)
    return resets.resolve_pinned_clips


def test_pinning_keys_on_the_env_index_not_the_reset_order():
    """A PARTIAL reset must leave every env on its own clip.

    The subtle failure: indexing by position within ``env_ids`` instead of by env index. It
    looks right on a full reset (env_ids = 0..N-1) and silently reshuffles clips the first
    time a subset of envs terminates -- which is every step after the first.
    """
    pin = _pin()
    ids = [10, 20, 30, 40]
    full = pin(torch.arange(4), ids, torch.device("cpu"))
    assert full.tolist() == ids
    # Envs 1 and 3 terminate on their own: they must keep clips 20 and 40.
    subset = pin(torch.tensor([1, 3]), ids, torch.device("cpu"))
    assert subset.tolist() == [20, 40]


def test_pinning_wraps_when_there_are_more_envs_than_clips():
    """More envs than pinned clips repeats the list rather than failing."""
    pin = _pin()
    got = pin(torch.arange(5), [7, 8], torch.device("cpu"))
    assert got.tolist() == [7, 8, 7, 8, 7]


def test_pinning_reads_a_json_file():
    """The file form exists because a hydra override of 100+ ids is unwieldy."""
    pin = _pin()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([3, 1, 4], f)
        path = f.name
    assert pin(torch.arange(3), path, torch.device("cpu")).tolist() == [3, 1, 4]


@pytest.mark.parametrize("off", [None, "", "   "])
def test_no_pin_means_sample_as_usual(off):
    """"Off" must fall through to the random draw, not to clip 0.

    ``""`` is off as well as ``None`` because Isaac Lab's configclass types an event param
    from its default, so the event cannot default to ``None`` and still accept a string
    override -- the empty string is the typed stand-in.
    """
    assert _pin()(torch.arange(3), off, torch.device("cpu")) is None


def test_pinning_accepts_a_comma_separated_string():
    """A short set should not need a file: hydra passes it as one string."""
    got = _pin()(torch.arange(3), "5, 6,7", torch.device("cpu"))
    assert got.tolist() == [5, 6, 7]


@pytest.mark.parametrize("bad", [[], [0, -1]])
def test_pinning_refuses_nonsense(bad):
    """Empty and negative both silently degrade to "some clip" if not checked."""
    with pytest.raises(ValueError):
        _pin()(torch.arange(2), bad, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 2. Per-env stats out of play_plots.
# ---------------------------------------------------------------------------
def test_play_plots_emits_a_per_env_column_per_metric(tmp_path):
    """``tracking_per_env.csv`` must have one ROW per env, and survival must be per env.

    Built from synthetic logger output so it needs no rollout. Env 1 is given a fall (its
    timeout mask is all False) so the survival column has both values in it.
    """
    pp = _load("play_plots_ab", REPO / "scripts" / "rsl_rl" / "play_plots.py")
    n_t, n_env, n_dim = 20, 3, 3
    rng = np.random.default_rng(0)
    mask = np.ones((n_t, n_env), dtype=bool)
    mask[:, 1] = False                                  # env 1 fell
    data = {
        "y_des": rng.normal(size=(n_t, n_env, n_dim)),
        "y_act": rng.normal(size=(n_t, n_env, n_dim)),
        "dy_des": rng.normal(size=(n_t, n_env, n_dim)),
        "dy_act": rng.normal(size=(n_t, n_env, n_dim)),
        "target_vel": rng.normal(size=(n_t, n_env, 3)),
        "robot_base_vel": rng.normal(size=(n_t, n_env, 3)),
        "_timeout_mask": mask,
    }
    meta = {"pos_names": ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z"],
            "vel_names": ["a", "b", "c"], "dt": 0.02}
    pp.compute_and_save_stats(data, meta, str(tmp_path))

    path = tmp_path / "tracking_per_env.csv"
    assert path.is_file(), "no per-env CSV; the Isaac distribution would be empty"
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == n_env
    assert [int(r["env_id"]) for r in rows] == [0, 1, 2]
    # Survival is per env, and env 1 is the one that fell.
    assert [float(r["survived"]) for r in rows] == [1.0, 0.0, 1.0]
    # The velocity channels the driver joins on are present and finite.
    for chan in ("forward_vx", "lateral_vy", "yaw_wz"):
        col = [float(r[f"vel_track_{chan}_mse"]) for r in rows]
        assert all(np.isfinite(col)) and all(c >= 0 for c in col)


def test_per_env_mean_matches_the_scalar_it_came_from(tmp_path):
    """The per-env column has to be the SAME reduction as the published mean.

    Guards the join: if the vector were, say, the unmasked series while the scalar was
    masked, the two halves of the comparison would disagree and only the distribution would
    be wrong.
    """
    pp = _load("play_plots_ab2", REPO / "scripts" / "rsl_rl" / "play_plots.py")
    n_t, n_env = 15, 4
    rng = np.random.default_rng(1)
    data = {
        "target_vel": rng.normal(size=(n_t, n_env, 3)),
        "robot_base_vel": rng.normal(size=(n_t, n_env, 3)),
    }
    pp.compute_and_save_stats(data, {"dt": 0.02}, str(tmp_path))
    scal = {r["metric"]: float(r["value"])
            for r in csv.DictReader(open(tmp_path / "tracking_stats.csv"))}
    rows = list(csv.DictReader(open(tmp_path / "tracking_per_env.csv")))
    per_env = np.array([float(r["vel_track_forward_vx_mse"]) for r in rows])
    assert per_env.mean() == pytest.approx(scal["vel_track_forward_vx_mse_mean"], rel=1e-5)


# ---------------------------------------------------------------------------
# 3. The metric join in the driver.
# ---------------------------------------------------------------------------
def _write_sim_log(run: pathlib.Path, upright: bool, n: int = 40) -> None:
    """A minimal ``sim_log.csv`` carrying just the columns success is judged on."""
    z = 0.78 if upright else 0.21
    pitch = 0.0 if upright else 1.8                      # 103 deg, well past the limit
    with open(run / "sim_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "base_z", "act_roll", "act_pitch"])
        for i in range(n):
            w.writerow([i * 0.02, z, 0.0, pitch])


def _driver():
    """The A/B driver module."""
    return _load("clip_ab_sweep", REPO / "scripts" / "rsl_rl" / "clip_ab_sweep.py")


def test_isaac_mse_is_square_rooted_before_comparison(tmp_path):
    """Isaac publishes MSE, MuJoCo publishes RMS. Comparing them raw reads ~10x low."""
    d = _driver()
    with open(tmp_path / "isaac_tracking_per_env.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_id", "survived", "vel_track_forward_vx_mse"])
        w.writerow([0, 1.0, 0.25])                      # MSE 0.25 -> RMSE 0.5
        w.writerow([1, 0.0, 0.09])                      # MSE 0.09 -> RMSE 0.3
    cols = d.collect_isaac(tmp_path, [11, 22])
    assert cols["vel_rmse_vx"].tolist() == pytest.approx([0.5, 0.3])
    assert cols["success"].tolist() == [1.0, 0.0]
    # Row i is clip ids[i] -- the whole point of pinning.
    assert cols["clip_id"].tolist() == [11.0, 22.0]


def test_mujoco_side_reduces_orientation_and_joints_the_same_way(tmp_path):
    """The two reduced metrics must be an RMS over their members, not a mean."""
    d = _driver()
    for cid, roll in ((5, 0.3), (6, 0.0)):
        run = tmp_path / "mj" / f"clip_{cid:05d}"
        run.mkdir(parents=True)
        (run / "metrics_summary.json").write_text(json.dumps({
            "survival": {"survived": cid == 5},
            "command_velocity_rms": {"vx": 0.1, "vy": 0.2, "wz": 0.3},
            "reference_tracking": {"roll": roll, "pitch": 0.4, "yaw": 0.0},
            "joint_tracking": {"rms": {"a": 0.3, "b": 0.4}},
        }))
        # clip 5 stays up, clip 6 goes down: success comes from the TRAJECTORY now.
        _write_sim_log(run, upright=(cid == 5))
    cols = d.collect_mujoco(tmp_path, [5, 6])
    assert cols["clip_id"].tolist() == [5.0, 6.0]
    assert cols["success"].tolist() == [1.0, 0.0]
    # RMS over (0.3, 0.4, 0.0) = sqrt((0.09+0.16)/3)
    assert cols["orient_rmse"][0] == pytest.approx(np.sqrt(0.25 / 3))
    # RMS over (0.3, 0.4) = sqrt((0.09+0.16)/2) -- a MEAN would give 0.35
    assert cols["joint_rmse"][0] == pytest.approx(np.sqrt(0.25 / 2))
    assert cols["joint_rmse"][0] != pytest.approx(0.35)


def test_missing_mujoco_runs_are_dropped_not_zero_filled(tmp_path):
    """A crashed clip must shrink n, not enter the distribution as a zero."""
    d = _driver()
    run = tmp_path / "mj" / "clip_00009"
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(json.dumps({"survival": {"survived": True}}))
    _write_sim_log(run, upright=True)
    cols = d.collect_mujoco(tmp_path, [9, 99])          # clip 99 never ran
    assert cols["clip_id"].tolist() == [9.0]
    assert cols["success"].tolist() == [1.0]


def test_base_position_is_not_in_the_compared_set():
    """Matched ego anchoring zeroes base position error, so comparing it is meaningless.

    Pinned as a test because the column is trivially easy to re-add and would look like a
    free extra metric while actually reading 1e-17 in both sims.
    """
    names = [n for n, _i, _m in _driver().COMPARED]
    assert not [n for n in names if "base_pos" in n], names
    assert "orient_rmse" in names and "joint_rmse" in names


def test_clip_choice_is_deterministic_and_length_filtered(tmp_path):
    """Same seed, same clips -- otherwise a re-run is not the same experiment."""
    h5py = pytest.importorskip("h5py", exc_type=ImportError)
    lib = tmp_path / "lib.h5"
    with h5py.File(lib, "w") as f:
        f["clip_len"] = np.array([999] * 8 + [100] * 4)   # 4 too-short clips
    d = _driver()
    a = d.choose_clips(lib, 5, seed=3, min_len=990)
    b = d.choose_clips(lib, 5, seed=3, min_len=990)
    assert a == b == sorted(a)
    assert all(c < 8 for c in a), f"picked a short clip: {a}"
    with pytest.raises(ValueError, match="only 8 of 12"):
        d.choose_clips(lib, 9, seed=3, min_len=990)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_mujoco_success_uses_isaac_fall_criterion_not_the_runners(tmp_path):
    """The two sims must be scored by the SAME predicate.

    MuJoCo's own ``Robot.failed()`` is ``qpos[2] < 0.2`` -- height only, at less than half
    Isaac's 0.5 m, with no tilt check -- and the run stops the moment it trips, so the
    failure usually never reaches the log. Measured on a real 100-clip sweep:
    ``metrics_summary.json`` flagged 1 fall where the runner itself reported 30. Scoring each
    sim by its own rule had MuJoCo beating Isaac 99 to 91; rescoring put MuJoCo at 68.
    """
    d = _driver()
    # Upright by MuJoCo's rule (0.45 > 0.2) but DOWN by Isaac's (0.45 < 0.5).
    run = tmp_path / "mj" / "clip_00001"
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(json.dumps({"survival": {"survived": True}}))
    with open(run / "sim_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "base_z", "act_roll", "act_pitch"])
        w.writerow([0.0, 0.78, 0.0, 0.0])
        w.writerow([0.02, 0.45, 0.0, 0.0])
    assert d.collect_mujoco(tmp_path, [1])["success"].tolist() == [0.0]

    # Upright and level throughout -> survived, even though tilt is near the limit.
    run2 = tmp_path / "mj" / "clip_00002"
    run2.mkdir(parents=True)
    (run2 / "metrics_summary.json").write_text(json.dumps({"survival": {"survived": False}}))
    with open(run2 / "sim_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "base_z", "act_roll", "act_pitch"])
        w.writerow([0.0, 0.78, 1.2, 0.0])                # 69 deg, just inside
    assert d.collect_mujoco(tmp_path, [2])["success"].tolist() == [1.0]


def test_tilt_alone_counts_as_a_fall(tmp_path):
    """Height-only scoring called a robot face-down at 0.21 m and 96 deg a SURVIVOR."""
    d = _driver()
    run = tmp_path / "mj" / "clip_00003"
    run.mkdir(parents=True)
    (run / "metrics_summary.json").write_text(json.dumps({"survival": {"survived": True}}))
    with open(run / "sim_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "base_z", "act_roll", "act_pitch"])
        w.writerow([0.0, 0.80, 0.0, 1.7])                # tall enough, but 97 deg over
    assert d.collect_mujoco(tmp_path, [3])["success"].tolist() == [0.0]
