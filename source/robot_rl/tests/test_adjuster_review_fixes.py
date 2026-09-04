# Copyright 2026 Zachary Olkin. All rights reserved.

"""Regression guards for defects that reached a live training run.

Each of these was found by review AFTER the code had trained, and each was silent -- the run
came up, the losses moved, and nothing pointed at any of them. They are grouped here because
they share a shape: a value that is WRONG rather than missing, so no shape check can see it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _NoManagerEnv:
    """An env mid-construction: the observation manager does not exist yet."""

    device = "cpu"


class _BuiltEnv:
    """An env whose observation manager exists and carries a ``reference_window`` term."""

    device = "cpu"

    def __init__(self, offsets) -> None:
        class _Cfg:
            def __init__(self, o):
                self.params = {"offsets": list(o)}

        class _OM:
            def __init__(self, o):
                self.active_terms = {"policy": ["reference_window"]}
                self._group_obs_term_cfgs = {"policy": [_Cfg(o)]}

        self.observation_manager = _OM(offsets)


def _cmd(env):
    """A stand-in exposing only what ``obs_window_offsets`` touches."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
        motion_sequence_cmd,
    )

    obj = object.__new__(motion_sequence_cmd.MotionSequenceCommand)
    # Both spellings: the method under test reads `self.env`, while the inherited `device`
    # property reads `self._env`.
    obj.env = env
    obj._env = env
    return obj


def test_a_probe_before_the_observation_manager_exists_does_not_poison_the_cache() -> None:
    """The bug that displaced the goal for a whole run, and printed one WARN line.

    ``ObservationManager._prepare_terms`` calls every term once to discover its shape, and it
    does that BEFORE ``env.observation_manager`` is assigned. The adjuster's ``goal_geometry``
    reaches ``obs_window_offsets`` during that probe. Memoising the failure made
    ``last_usable_step`` fall back to ``clip_len - 1`` FOREVER -- which its own docstring
    names as the mis-anchoring that puts the goal a full observation window past anything an
    episode can reach.
    """
    cmd = _cmd(_NoManagerEnv())
    assert cmd.obs_window_offsets() is None
    # The manager arrives. The answer must now resolve, not stay stuck on the probe's failure.
    cmd.env = _BuiltEnv([0, 5, 50])
    got = cmd.obs_window_offsets()
    assert got is not None, "the failed probe was cached and the real answer is unreachable"
    assert [int(v) for v in got] == [0, 5, 50]


def test_a_resolved_answer_is_still_cached() -> None:
    """The memo must survive -- it is read once per control step by `clip_end_reached`."""
    cmd = _cmd(_BuiltEnv([0, 7]))
    first = cmd.obs_window_offsets()
    cmd.env = _BuiltEnv([1, 2, 3])       # a different answer, which must NOT be re-read
    assert [int(v) for v in cmd.obs_window_offsets()] == [int(v) for v in first]


def test_the_conditioning_row_carries_the_null_history_flag() -> None:
    """``cond_width`` must include it, or the row cannot describe the prefix it came from.

    Where the flag is set, ``encode_prefix`` replaces every history token with a learned
    embedding -- and ``BatchedHistory.stack_padded`` deliberately leaves those rows stale
    BECAUSE the flag covers them. A consumer re-encoding without it therefore feeds the
    previous episode's rows, from a different tile, as real history.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
        fm_tracking_cmd,
    )

    class _Ctx:
        N, D = 3, 44
        extero_sensor_count, extero_sensor_dim = 2, 3900
        num_cmd_tokens, cmd_dim = 13, 3

    obj = object.__new__(fm_tracking_cmd.FmTrackingCommand)
    obj.ctx = _Ctx()
    streams = (_Ctx.N * _Ctx.D
               + _Ctx.extero_sensor_count * _Ctx.extero_sensor_dim
               + _Ctx.num_cmd_tokens * _Ctx.cmd_dim)
    assert obj.cond_width == streams + 1


def test_the_actor_expects_the_flag_too() -> None:
    """The two widths must move together, or the env refuses its own observation group."""
    import inspect

    from robot_rl.rl import adjuster_model

    src = inspect.getsource(adjuster_model.WindowAdjusterModel._get_obs_dim)
    assert "+ 1" in src, (
        "the actor's expected conditioning width no longer accounts for the null-history "
        "flag that FmTrackingCommand.cond_width appends"
    )


def test_goal_geometry_counts_to_the_deadline_the_episode_is_actually_cut_on() -> None:
    """The fourth channel exists to make a bootstrapped truncation well posed.

    With ``slack_s`` left at 0 against a task running 3.0 s of it, the count reached zero 150
    control steps -- 12 whole agent decisions -- before the episode was cut, so the last third
    of a short episode carried no time evidence and the cut was as unpredictable as having no
    channel at all.
    """
    import inspect

    from robot_rl.tasks.manager_based.robot_rl.mdp.observations import observations

    sig = inspect.signature(observations.goal_geometry)
    assert "slack_s" in sig.parameters and "step_scale" in sig.parameters

    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_adjuster_env_cfg as cfg

    params = cfg.G1AdjusterObservationsCfg.CriticCfg().goal_geometry.params
    assert params["slack_s"] == cfg.GOAL_SLACK_S, (
        "the observed deadline and the clip_end termination's must be the same deadline"
    )
    # The critic group is un-normalized by design, and every other channel in it is O(1).
    horizon = cfg.EPISODE_LENGTH_S / cfg.CONTROL_DT
    assert params["step_scale"] * horizon == pytest.approx(1.0)


def test_every_critic_channel_is_scaled_to_order_one() -> None:
    """The critic group is un-normalized, so a raw metre channel is a conditioning defect.

    MEASURED over 60 agent steps at 32 envs: with the offsets raw, `goal_geometry` reached
    -14.1 m and had a std of 2.105, against <= 0.93 for every other term in the group. It was
    the only outlier -- `window_residual`, which is 60% of the group, sits at std 0.277.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_adjuster_env_cfg as cfg

    params = cfg.G1AdjusterObservationsCfg.CriticCfg().goal_geometry.params
    scale = params["offset_scale"]
    assert len(scale) == 3 and all(s > 0 for s in scale)
    # 14 m downrange, and the goal's own fitted half-width / height tolerance, must all land
    # near 1 rather than near 10.
    assert 14.0 * scale[0] < 2.0, "downrange still reaches an order of magnitude above its peers"
    assert 1.5 * scale[1] == pytest.approx(1.0), "lateral should read 1.0 at the goal half-width"
    assert 0.6 * scale[2] == pytest.approx(1.0), "height should read 1.0 at the tolerance"


def test_the_wrapper_leaves_max_episode_length_in_control_steps() -> None:
    """Its one rsl-rl reader seeds a buffer this env keeps in CONTROL steps.

    Converting it to agent steps divided ``init_at_random_ep_len``'s randomization range by
    the re-plan period, opening every env inside the first 1/K of the episode -- the opposite
    of what that flag is for.
    """
    import inspect

    from robot_rl.rl import adjuster_wrapper

    src = inspect.getsource(adjuster_wrapper.RslRlAdjusterVecEnvWrapper.__init__)
    assert "self.max_episode_length =" not in src, (
        "max_episode_length is overridden again; rsl-rl seeds episode_length_buf from it and "
        "that buffer counts control steps on this env"
    )


def test_the_reference_patch_actually_samples_under_the_robot() -> None:
    """A STATIC check, where it is unambiguous -- not a runtime one on hit data.

    `_reference_height` measures "the ground under the robot" from the rays of the base patch
    that land near the sensor's own origin. If the patch's offset ever moves it entirely off
    the robot, every terrain height would silently be measured against ground a metre away.
    That is a property of the PATTERN, so it is checkable here; at runtime the same symptom is
    produced by a robot standing over a void, which is routine and must NOT raise.
    """
    from isaaclab.sensors.ray_caster.patterns import patterns, patterns_cfg

    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_adjuster_env_cfg as cfg
    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.observations import (
        _REF_RADIUS_M,
    )

    env = cfg.G1AdjusterEnvCfg()
    patch = env.scene.terrain_base_patch
    starts, _ = patterns.grid_pattern(patch.pattern_cfg, "cpu")
    assert isinstance(patch.pattern_cfg, patterns_cfg.GridPatternCfg)
    # Distance from the SENSOR origin, which is where `_reference_height` measures from.
    assert float(starts[:, :2].norm(dim=-1).min()) < _REF_RADIUS_M, (
        "the base patch has no ray within the reference radius of its own origin, so "
        "'terrain height under the robot' would be measured somewhere else entirely"
    )


def test_the_play_cfg_shows_each_tile_once_and_training_still_replicates() -> None:
    """``num_rows`` is an env-CAPACITY knob, and the two tasks need opposite values.

    Rows are pure REPLICAS of the whole tile set, so a play cfg that inherits the training
    value renders the library ten times over -- an 8-tile set becomes 80 tiles marching off
    into the distance, which is what it did before this was pinned.

    Training must keep the replicas: measured at 256 envs on the 75-tile 4-family library,
    all 75 distinct terrains are covered with 3-4 envs each. One row there would put ~13 envs
    on identical geometry.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_adjuster_env_cfg import (
        G1AdjusterEnvCfg,
        G1AdjusterEnvCfgPlay,
    )

    play = G1AdjusterEnvCfgPlay()
    assert play.scene.terrain.num_rows == 1, (
        "the play cfg replicates the tile set; every terrain should appear exactly once"
    )
    train = G1AdjusterEnvCfg()
    assert train.scene.terrain.num_rows > 1, (
        "training lost its replica rows, which are what give env capacity and spatial "
        "separation across a small library"
    )
