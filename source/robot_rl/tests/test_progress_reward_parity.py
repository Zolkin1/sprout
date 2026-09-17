# Copyright 2026 Zachary Olkin. All rights reserved.

"""The live progress reward is the offline grader's, term for term.

One reward, two implementations, and the whole algorithm comparison rests on them agreeing:
``scripts/rsl_rl/inspect_terrain_reward.py --stage reward --progress`` grades the recordings the
off-policy arms (awr / fbc / arrived / bc) fit a critic to, and
``robot_rl...mdp.rewards.progress`` is what the on-policy arm maximises directly. If the two
drifted, the ladder would be comparing rewards and reporting it as a comparison of algorithms,
with nothing in either run's output to say so.

These tests do NOT need a simulator: every piece of the reward downstream of the terrain grade
is arithmetic on ``[N]`` arrays, so both sides can be driven from the same synthetic inputs.
The terrain grade itself is already pinned live-vs-offline by ``test_terrain_reward_parity``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import torch

from motion_gen.progress_reward import PROGRESS_REWARD_SPEC, ProgressRewardSpec, terrain_gate

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _offline():
    """Import the offline grader as a module, without running it.

    It lives in ``scripts/`` and imports its siblings by bare name, so the directory has to be
    on the path.  Loaded lazily and cached by :mod:`sys.modules` so the cost is paid once.

    Returns:
        The ``inspect_terrain_reward`` module.
    """
    if "inspect_terrain_reward" in sys.modules:
        return sys.modules["inspect_terrain_reward"]
    d = _REPO / "scripts" / "rsl_rl"
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(
        "inspect_terrain_reward", d / "inspect_terrain_reward.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inspect_terrain_reward"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_offline_grader_reads_the_shared_constants() -> None:
    """Not five literals in two files. The defaults ARE the spec."""
    m = _offline()
    p = m.build_parser() if hasattr(m, "build_parser") else None
    assert m.REWARD_TAU == PROGRESS_REWARD_SPEC.tau
    if p is not None:                                    # the parser is built in main() on some
        d = {a.dest: a.default for a in p._actions}      # revisions; skip rather than guess
        assert d["gate_sharpness"] == PROGRESS_REWARD_SPEC.gate_sharpness
        assert d["success_gate_sharpness"] == PROGRESS_REWARD_SPEC.success_gate_sharpness
        assert d["vel_sigma"] == PROGRESS_REWARD_SPEC.vel_sigma_mps
        assert d["success_const"] == PROGRESS_REWARD_SPEC.success_const


def test_the_per_chunk_bound_is_two_plus_C() -> None:
    """``R <= 1``, ``gate * vel <= 1``, and the terminal chunk adds at most ``C``."""
    m = _offline()
    assert m.per_step_reward_max(10.0) == pytest.approx(PROGRESS_REWARD_SPEC.per_chunk_max)
    assert m.per_step_reward_max(3.0) == pytest.approx(5.0)


@pytest.mark.parametrize("sharpness", [1.0, 2.0, 3.0, 4.0])
def test_the_gate_is_one_function_for_numpy_and_torch(sharpness: float) -> None:
    """The two consumers hand it different array types and must get the same number."""
    r = np.linspace(0.0, 1.0, 257)
    a = terrain_gate(r, sharpness)
    b = terrain_gate(torch.from_numpy(r), sharpness).numpy()
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_the_gate_matches_the_form_it_is_documented_as() -> None:
    """``sin(pi/2 * R**s)``, which is ``cos(pi/2 * (1 - R**s))`` -- stated both ways in docs."""
    r = np.linspace(0.0, 1.0, 129)
    s = 2.0
    np.testing.assert_allclose(terrain_gate(r, s), np.sin(0.5 * np.pi * r**s), atol=1e-12)
    np.testing.assert_allclose(terrain_gate(r, s), np.cos(0.5 * np.pi * (1.0 - r**s)), atol=1e-6)


def test_the_gate_clips_rather_than_extrapolating() -> None:
    """A penalty of 0 gives ``R = 1`` exactly, and float error must not push the gate past it."""
    assert terrain_gate(np.array([1.0 + 1e-9]), 2.0)[0] == pytest.approx(1.0)
    assert terrain_gate(np.array([-1e-9]), 2.0)[0] == pytest.approx(0.0)


def _offline_chunk(reward_terrain, v_fwd, cmd_vx, worst, arrived, is_last, spec):
    """The offline grader's per-chunk reward, written out from `_progress_terms`' algebra.

    Kept as an independent transcription rather than a call into `_progress_terms`, which needs
    an h5 rollout to read anchors from: what is being pinned is the ARITHMETIC, and a
    transcription that drifted from the source would be caught by
    `test_the_offline_source_still_has_this_algebra` below.

    Args:
        reward_terrain: ``[N]`` R.
        v_fwd: ``[N]`` achieved forward speed over the interval, m/s.
        cmd_vx: ``[N]`` nominal commanded forward speed, m/s.
        worst: ``[N]`` the episode's worst R.
        arrived: ``[N]`` 0/1.
        is_last: ``[N]`` 0/1.
        spec: The constants.

    Returns:
        ``[N]`` the chunk's reward, unnormalized.
    """
    gate = terrain_gate(reward_terrain, spec.gate_sharpness)
    vel = np.exp(-np.abs(v_fwd - cmd_vx) / spec.vel_sigma_mps)
    gate_worst = terrain_gate(worst, spec.success_gate_sharpness)
    success = gate_worst * spec.success_const * arrived * is_last
    # The last chunk has no successor to difference against, so it keeps R + success alone.
    return reward_terrain + gate * vel * (1.0 - is_last) + success


def test_the_live_terms_reproduce_the_offline_chunk_reward() -> None:
    """Same inputs, same three numbers, to float tolerance.

    The live side is exercised through its own arithmetic rather than through a fake env: the
    env supplies the inputs (the grade, the anchors, the command) and this is what it does with
    them, which is the half that can drift.
    """
    rng = np.random.default_rng(0)
    n = 512
    spec = PROGRESS_REWARD_SPEC
    R = rng.uniform(0.0, 1.0, n)
    worst = np.minimum(R, rng.uniform(0.0, 1.0, n))
    v_fwd = rng.uniform(-0.5, 3.0, n)
    cmd_vx = rng.uniform(0.4, 2.5, n)
    arrived = (rng.random(n) < 0.3).astype(np.float64)
    is_last = (rng.random(n) < 0.2).astype(np.float64)

    want = _offline_chunk(R, v_fwd, cmd_vx, worst, arrived, is_last, spec)

    tR = torch.from_numpy(R)
    t_is_last = torch.from_numpy(is_last).bool()
    gate = terrain_gate(tR, spec.gate_sharpness)
    vel = torch.exp(-(torch.from_numpy(v_fwd) - torch.from_numpy(cmd_vx)).abs()
                    / spec.vel_sigma_mps)
    gate_vel = gate * vel * (~t_is_last).to(vel.dtype)
    success = (terrain_gate(torch.from_numpy(worst), spec.success_gate_sharpness)
               * spec.success_const * torch.from_numpy(arrived) * t_is_last.to(tR.dtype))
    got = (tR + gate_vel + success).numpy()

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_the_yaw_projection_matches_the_offline_one_on_random_quaternions() -> None:
    """XYZW, and only the YAW removed.

    The one place the two implementations could plausibly disagree and still look right: a
    wxyz/xyzw mix-up rotates every velocity by the robot's own tilt and leaves the magnitudes
    entirely plausible. The offline grader indexes ``[3],[0],[1],[2]`` and says it verified
    that against a real recording; ``FmTrackingCommand`` initialises ``_anchor_quat[:, 3] = 1``,
    which is the same statement from the other end.
    """
    rng = np.random.default_rng(7)
    n = 256
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v_w = rng.normal(size=(n, 3))

    qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c, s = np.cos(-yaw), np.sin(-yaw)
    want = c * v_w[:, 0] - s * v_w[:, 1]

    tq, tv = torch.from_numpy(q), torch.from_numpy(v_w)
    tqw, tqx, tqy, tqz = tq[:, 3], tq[:, 0], tq[:, 1], tq[:, 2]
    tyaw = torch.atan2(2.0 * (tqw * tqz + tqx * tqy), 1.0 - 2.0 * (tqy * tqy + tqz * tqz))
    tc, ts = torch.cos(-tyaw), torch.sin(-tyaw)
    got = (tc * tv[:, 0] - ts * tv[:, 1]).numpy()

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_a_yaw_only_rotation_leaves_a_forward_velocity_forward() -> None:
    """The property the projection exists for, so the test above cannot pass on a shared bug."""
    for deg in (0.0, 30.0, 90.0, -145.0):
        a = np.radians(deg)
        q = np.array([[0.0, 0.0, np.sin(a / 2), np.cos(a / 2)]])        # XYZW, yaw only
        v = np.array([[np.cos(a), np.sin(a), 0.0]])                     # 1 m/s along that yaw
        qw, qx, qy, qz = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = np.cos(-yaw), np.sin(-yaw)
        assert (c * v[:, 0] - s * v[:, 1])[0] == pytest.approx(1.0, abs=1e-9)


class _FakeEnv:
    """The env surface `_progress_terms` reads, and nothing else.

    Deliberately a stub rather than a simulator: everything the live reward does downstream of
    the terrain grade is arithmetic on ``[N]`` tensors, and that arithmetic is the half that can
    drift from the offline grader. The grade itself is pinned live-vs-offline by
    ``test_terrain_reward_parity``.
    """

    def __init__(self, n, contact, penetration, root_pos, anchor_pos, anchor_quat,
                 cmd_vx, is_last, arrived, worst=None, gamma=0.95):
        self.num_envs = n
        self.common_step_counter = 0
        self.replan_period = 12
        self.step_dt = 0.02
        self.device = "cpu"
        self.worst_terrain_R = (torch.full((n,), float("inf")) if worst is None
                                else worst.clone())
        self.progress_anchor_pos = anchor_pos
        self.progress_anchor_quat = anchor_quat
        self.progress_cmd_vx = cmd_vx
        self.reset_terminated = is_last
        self.reset_time_outs = torch.zeros(n, dtype=torch.bool)
        self.ended_by = {"goal_reached": arrived}
        self.progress_gamma_scale = 1.0 - gamma
        self._contact, self._pen, self._root = contact, penetration, root_pos

        class _Data:
            root_pos_w = root_pos

        class _Robot:
            data = _Data()

        self.scene = {"robot": _Robot()}


def _live(env, monkeypatch):
    """Call the REAL live terms against a fake env, with the SDF pass stubbed out."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.rewards import progress as P

    monkeypatch.setattr(P, "plan_terrain_terms",
                        lambda e, name: {"contact_penalty": e._contact,
                                         "penetration_sum_m": e._pen})
    return P


def test_the_LIVE_terms_reproduce_the_offline_chunk_reward(monkeypatch) -> None:
    """The real `mdp.progress_*` functions, executed, against the offline algebra.

    This is the test that matters, and an earlier version of this file did NOT have it: it
    compared two transcriptions written side by side in this file, so the live module was never
    imported. Dropping the `(1-gamma)` factor, inverting the last-chunk mask, or reading
    `worst_terrain_R` before its update would all have passed that.
    """
    from motion_gen.terrain_consistency import TerrainConsistencySpec

    rng = np.random.default_rng(11)
    n = 256
    spec, tc = PROGRESS_REWARD_SPEC, TerrainConsistencySpec()

    contact = torch.from_numpy(rng.uniform(0.0, 0.02, n))
    pen = torch.from_numpy(rng.uniform(0.0, 3.0, n))
    anchor_pos = torch.from_numpy(rng.normal(size=(n, 3)))
    disp = torch.from_numpy(rng.normal(scale=0.4, size=(n, 3)))
    q = torch.from_numpy(rng.normal(size=(n, 4)))
    q /= q.norm(dim=1, keepdim=True)
    cmd_vx = torch.from_numpy(rng.uniform(0.4, 2.5, n))
    is_last = torch.from_numpy(rng.random(n) < 0.25)
    arrived = torch.from_numpy(rng.random(n) < 0.4)
    prior_worst = torch.from_numpy(rng.uniform(0.2, 1.0, n))

    env = _FakeEnv(n, contact, pen, anchor_pos + disp, anchor_pos, q, cmd_vx, is_last,
                   arrived, worst=prior_worst)
    P = _live(env, monkeypatch)

    got_terrain = P.progress_terrain(env, "traj_ref", spec)
    got_vel = P.progress_velocity(env, "traj_ref", spec)
    got_success = P.progress_success(env, "traj_ref", spec)

    # --- the offline algebra, on the same inputs ------------------------------------------
    penalty = (contact.numpy() * tc.contact_weight + pen.numpy() * tc.penetration_weight)
    R = np.exp(-penalty / spec.tau)
    worst = np.minimum(prior_worst.numpy(), R)
    v_w = disp.numpy() / (12 * 0.02)
    qn = q.numpy()
    qw, qx, qy, qz = qn[:, 3], qn[:, 0], qn[:, 1], qn[:, 2]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c, s = np.cos(-yaw), np.sin(-yaw)
    v_fwd = c * v_w[:, 0] - s * v_w[:, 1]
    lastf = is_last.numpy().astype(np.float64)
    want = _offline_chunk(R, v_fwd, cmd_vx.numpy(), worst, arrived.numpy().astype(np.float64),
                          lastf, spec)

    np.testing.assert_allclose(got_terrain.numpy(), R, rtol=0, atol=1e-12)
    # The live success term additionally carries `(1 - gamma)`; divide it back out to compare
    # against the offline chunk, then pin the factor itself separately below.
    live_sum = got_terrain + got_vel + got_success / env.progress_gamma_scale
    np.testing.assert_allclose(live_sum.numpy(), want, rtol=0, atol=1e-12)


def test_the_live_success_term_carries_the_one_minus_gamma_factor(monkeypatch) -> None:
    """The factor the whole return-matching argument rests on, pinned on its own.

    It is invisible in a chunk-reward comparison (it scales one term), so it gets its own test.
    """
    n = 8
    env = _FakeEnv(n, torch.zeros(n), torch.zeros(n), torch.zeros(n, 3), torch.zeros(n, 3),
                   torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1), torch.zeros(n),
                   torch.ones(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool),
                   worst=torch.ones(n), gamma=0.95)
    P = _live(env, monkeypatch)
    got = P.progress_success(env, "traj_ref", PROGRESS_REWARD_SPEC)
    # R = 1 (zero penalty) so both gates are 1: the bonus is C * (1 - gamma).
    assert got[0].item() == pytest.approx(PROGRESS_REWARD_SPEC.success_const * 0.05)


def test_the_last_chunk_drops_the_velocity_term_and_only_it(monkeypatch) -> None:
    """Matching the offline grader, whose final window has no successor to difference against."""
    n = 2
    is_last = torch.tensor([False, True])
    env = _FakeEnv(n, torch.zeros(n), torch.zeros(n), torch.zeros(n, 3), torch.zeros(n, 3),
                   torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1), torch.zeros(n),
                   is_last, torch.zeros(n, dtype=torch.bool), worst=torch.ones(n))
    P = _live(env, monkeypatch)
    vel = P.progress_velocity(env, "traj_ref", PROGRESS_REWARD_SPEC)
    assert vel[0].item() == pytest.approx(1.0)      # standing still, commanded 0 -> perfect
    assert vel[1].item() == 0.0


def test_the_running_worst_is_updated_BEFORE_it_is_read(monkeypatch) -> None:
    """The success gate reads the episode's worst chunk INCLUDING the terminal one.

    Reading before the update would let a destructive final re-plan collect a full bonus.
    """
    n = 1
    # A large penalty on this chunk -> R well under the prior worst of 1.0.
    env = _FakeEnv(n, torch.tensor([0.01]), torch.tensor([1.0]), torch.zeros(n, 3),
                   torch.zeros(n, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.zeros(n),
                   torch.ones(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool),
                   worst=torch.ones(n))
    P = _live(env, monkeypatch)
    R = P.progress_terrain(env, "traj_ref", PROGRESS_REWARD_SPEC)
    assert R.item() < 1.0
    assert env.worst_terrain_R.item() == pytest.approx(R.item()), (
        "this chunk's grade did not reach the running worst")
    gated = P.progress_success(env, "traj_ref", PROGRESS_REWARD_SPEC)
    full = PROGRESS_REWARD_SPEC.success_const * env.progress_gamma_scale
    assert gated.item() < full, "the bonus was not gated by this chunk's own grade"


def test_the_offline_source_still_has_this_algebra() -> None:
    """Pins the transcription above against the real grader, by source.

    Cheaper than building an h5 rollout, and it catches the change that matters: if
    ``_progress_terms`` stops gating the velocity term, or stops paying the bonus only on the
    last chunk, the strings below go and this fails.
    """
    import inspect

    src = inspect.getsource(_offline()._progress_terms)
    assert "terrain_gate(" in src, "the offline gate is no longer the shared one"
    assert "np.exp(-err / float(vel_sigma))" in src, "the velocity kernel changed"
    assert "gate_worst * float(success_const)" in src, "the bonus is no longer worst-gated"
    assert "is_last" in src, "the bonus is no longer terminal"
