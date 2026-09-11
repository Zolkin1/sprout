# Copyright 2026 Zachary Olkin. All rights reserved.

"""The online critic batch must equal the offline one, field for field.

``scripts/rsl_rl/eval/headroom_value.py`` rebuilds ``fit_value._batch`` from live Isaac
buffers so a best-of-K selector can score candidates without a device round trip.  Every way
that rebuild can go wrong is shape-silent: a normalizer applied in the wrong order, the
command striding applied after normalization instead of before, the height scan left
unsanitised, the twist sliced on the wrong half, the goal on the wrong scale -- each gives a
finite batch of exactly the right widths that the critic scores in an input space it was never
fitted in.  Nothing downstream can detect that.

So the test is not "does it run".  It builds one set of tensors, hands the numpy equivalent to
the REAL ``fit_value._batch`` and the tensors themselves to the online scorer, and asserts the
two ``ValueInputs`` are bit-identical.

No simulator and no GPU.  One seam needs a built scene -- ``OnlineValueScorer._read_scan``,
which calls ``mdp.scan_obs_stack`` -- and that is exactly why it is a method: it is overridden
here and everything else in the assembly is exercised for real, including the generator's own
normalizers, ``transforms.sanitize_extero``, ``goal_region.goal_frame_offsets`` and a real
``FlowValueModel`` forward pass.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "isaaclab",
    reason="the normalizers, sanitizer and goal geometry live in robot_rl, which imports "
           "isaaclab (152 modules, no isaacsim -- no simulator is booted here)")

_REPO = pathlib.Path(__file__).resolve().parents[3]
for _p in (_REPO / "scripts" / "flow_matching",
           _REPO / "scripts" / "rsl_rl",
           _REPO / "scripts" / "rsl_rl" / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_value  # noqa: E402
import headroom_value  # noqa: E402

from robot_rl.flow_matching import model as fm_model  # noqa: E402
from robot_rl.flow_matching import normalizer as fm_normalizer  # noqa: E402
from robot_rl.flow_matching import value  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region  # noqa: E402

DEVICE = "cpu"
NUM_ENVS = 7
N_HIST, D_NODE, F_FUT = 4, 11, 20
CMD_DIM, CMD_STRIDE = 3, 5
EXTERO_SENSORS, EXTERO_SENSOR_DIM = 2, 24
EXTERO_FLAT = EXTERO_SENSORS * EXTERO_SENSOR_DIM
#: Deliberately NOT square and deliberately not the exteroception width, so a transposed fold
#: or a swapped field raises on shape instead of scoring a plausible wrong number.
SCAN_HW = (11, 16)
PROPRIO_HW = (10, 64)
#: The subset the caller asks for -- unsorted, and not all of them, because ``env_ids`` is
#: whatever the selector wants scored and never necessarily ``arange(num_envs)``.
IDX = np.array([5, 0, 3, 6], dtype=np.int64)


# ----------------------------------------------------------------------
# The recorder's own twist, executed from its own source.
# ----------------------------------------------------------------------
def _record_rollouts_twist_yaw():
    """``record_rollouts._twist_yaw``, executed from the file rather than copied here.

    ``record_rollouts.py`` cannot be imported: it builds an argparse parser and calls
    ``parse_known_args`` at module scope, and imports ``isaaclab_tasks.utils``, which needs a
    full Isaac install.  Restating the function here instead would make this test compare the
    online scorer against a COPY -- so an edit to the recorder would leave both the copy and
    the test passing while the two implementations had diverged.  Executing the real source
    keeps the comparison honest.

    Returns:
        The ``_twist_yaw`` function object.

    Raises:
        AssertionError: If the function is no longer defined at module scope there.
    """
    src = (_REPO / "scripts" / "rsl_rl" / "record_rollouts.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_twist_yaw":
            # Its only module-scope name; the isaaclab math it needs is imported inside
            # the function body, which is why executing the definition alone is enough.
            ns: dict = {"torch": torch}
            exec(compile(ast.Module(body=[node], type_ignores=[]),  # noqa: S102
                         "<record_rollouts._twist_yaw>", "exec"), ns)
            return ns["_twist_yaw"]
    raise AssertionError(
        "record_rollouts._twist_yaw is gone or is no longer a module-scope function; this "
        "test executes its real source so the online mirror cannot drift from it.")


# ----------------------------------------------------------------------
# Stubs holding known tensors.
# ----------------------------------------------------------------------
class _StubHistory:
    """The two ``BatchedHistory`` accessors ``_emit`` reads."""

    def __init__(self, nodes: torch.Tensor, null: torch.Tensor) -> None:
        """Store the padded node stack and the null mask.

        Args:
            nodes: ``(num_envs, N, D)``.
            null: ``(num_envs,)`` bool.
        """
        self._nodes, self._null = nodes, null

    def stack_padded(self, key: str) -> torch.Tensor:
        """Return the node stack, refusing any other key.

        Args:
            key: Must be ``"node"``.

        Returns:
            ``(num_envs, N, D)``.
        """
        assert key == "node", f"the critic reads the emitted nodes, not {key!r}"
        return self._nodes

    def null_mask(self) -> torch.Tensor:
        """``(num_envs,)`` bool: True while the history ring had not filled."""
        return self._null


class _StubCmd:
    """The ``FmTrackingCommand`` surface ``OnlineValueScorer.inputs`` touches."""

    def __init__(self, ctx, nodes, null, extero, cmd_future) -> None:
        """Hold the buffers a live command would expose.

        Args:
            ctx: The generator context.
            nodes: ``(num_envs, N, D)`` emitted history nodes.
            null: ``(num_envs,)`` bool null-history mask.
            extero: ``(num_envs, T, K)`` the served exteroception row.
            cmd_future: ``(num_envs, F, cmd_dim)`` the command horizon.
        """
        self.ctx = ctx
        self._history = _StubHistory(nodes, null)
        self._extero = extero
        self._cmd_future = cmd_future

    def _lagged_extero(self) -> torch.Tensor:
        """The row the generator conditioned on, ``(num_envs, T, K)``."""
        return self._extero

    def get_command_velocity_horizon(self, horizon: int) -> torch.Tensor:
        """The future command, ``(num_envs, F, cmd_dim)``.

        Args:
            horizon: Must be the context's ``F`` -- the recorder asks for exactly that, and a
                shorter horizon would silently change which steps ``cmd_steps`` selects.

        Returns:
            ``(num_envs, F, cmd_dim)``.
        """
        assert horizon == self.ctx.F, f"expected F={self.ctx.F}, got {horizon}"
        return self._cmd_future


class _StubRd:
    """The converting view of ``ArticulationData`` the accumulator exposes."""

    def __init__(self, pos, quat, lin, ang) -> None:
        """Store the four root buffers the critic's context needs.

        Args:
            pos: ``(num_envs, 3)`` world position.
            quat: ``(num_envs, 4)`` world orientation, wxyz.
            lin: ``(num_envs, 3)`` world linear velocity.
            ang: ``(num_envs, 3)`` world angular velocity.
        """
        self.root_pos_w, self.root_quat_w = pos, quat
        self.root_lin_vel_w, self.root_ang_vel_w = lin, ang


class _StubTracker:
    """``RolloutEpisodeTracker`` reduced to the accumulator fields ``_emit`` reads."""

    def __init__(self, rd, goal_pos_w, goal_yaw) -> None:
        """Wrap the robot view and the goal anchor.

        Args:
            rd: A :class:`_StubRd`.
            goal_pos_w: ``(num_envs, 3)`` goal anchor, world.
            goal_yaw: ``(num_envs,)`` terrain heading.
        """
        self.accum = types.SimpleNamespace(rd=rd, _goal_pos_w=goal_pos_w, _goal_yaw=goal_yaw)


class _StubRing:
    """``record_rollouts.ProprioRing`` reduced to its buffer."""

    def __init__(self, buf: torch.Tensor) -> None:
        """Hold the ring.

        Args:
            buf: ``(num_envs, H, PROPRIO_DIM)``, newest last.
        """
        self.buf = buf


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _normalizer(width: int, gen: torch.Generator):
    """A non-trivial normalizer, with one identity dimension.

    Args:
        width: Feature width.
        gen: Seeded RNG.

    Returns:
        A ``Normalizer`` whose mean and std are both non-degenerate, so applying it in the
        wrong place cannot pass by looking like the identity.
    """
    mean = torch.randn(width, generator=gen)
    std = torch.rand(width, generator=gen) + 0.5
    mask = torch.ones(width, dtype=torch.bool)
    mask[0] = False                       # one identity dim, as a real fit produces
    return fm_normalizer.Normalizer(
        torch.where(mask, mean, torch.zeros_like(mean)),
        torch.where(mask, std, torch.ones_like(std)), mask)


def _context(gen: torch.Generator):
    """A generator context carrying real normalizers and the strided command steps.

    Args:
        gen: Seeded RNG.

    Returns:
        A ``SimpleNamespace`` with the fields ``fit_value._batch`` and the online scorer read.
    """
    return types.SimpleNamespace(
        state_norm=_normalizer(D_NODE, gen),
        extero_norm=_normalizer(EXTERO_FLAT, gen),
        cmd_norm=_normalizer(CMD_DIM, gen),
        cmd_dim=CMD_DIM,
        cmd_steps=list(range(0, F_FUT, CMD_STRIDE)),
        F=F_FUT, N=N_HIST, D=D_NODE, device=torch.device(DEVICE))


def _world(seed: int = 0):
    """One set of live buffers plus the numpy recording of the very same values.

    The two are the same numbers by construction, so any difference the assertions find is a
    difference between the two code paths and nothing else.

    Args:
        seed: RNG seed.

    Returns:
        ``(ctx, cmd, tracker, ring, columns)`` where ``columns`` is the recorded-table dict
        ``fit_value._batch`` indexes.
    """
    gen = torch.Generator().manual_seed(seed)
    ctx = _context(gen)

    nodes = torch.randn(NUM_ENVS, N_HIST, D_NODE, generator=gen)
    null = torch.tensor([i % 3 == 0 for i in range(NUM_ENVS)])
    extero = torch.randn(NUM_ENVS, EXTERO_SENSORS, EXTERO_SENSOR_DIM, generator=gen)
    cmd_future = torch.randn(NUM_ENVS, F_FUT, CMD_DIM, generator=gen)
    proprio = torch.randn(NUM_ENVS, *PROPRIO_HW, generator=gen)

    # Missed rays record -inf and one of them takes the whole batch's logit with it, so the
    # scan must exercise `sanitize_extero` rather than being uniformly finite. The large
    # finite value checks the CLAMP half of it at the same time.
    scan = torch.randn(NUM_ENVS, SCAN_HW[0] * SCAN_HW[1], generator=gen)
    scan[0, 3] = -float("inf")
    scan[2, 17] = 12.5

    pos = torch.randn(NUM_ENVS, 3, generator=gen)
    quat = torch.randn(NUM_ENVS, 4, generator=gen)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    lin = torch.randn(NUM_ENVS, 3, generator=gen)
    ang = torch.randn(NUM_ENVS, 3, generator=gen)
    goal_pos = torch.randn(NUM_ENVS, 3, generator=gen)
    goal_yaw = torch.randn(NUM_ENVS, generator=gen)
    # An episode with no goal records NaN, and `_batch`'s nan_to_num is what keeps one from
    # taking the batch down. Exercise it.
    goal_pos[4] = float("nan")

    rd = _StubRd(pos, quat, lin, ang)
    tracker = _StubTracker(rd, goal_pos, goal_yaw)
    cmd = _StubCmd(ctx, nodes, null, extero, cmd_future)
    ring = _StubRing(proprio)

    down, lat, hz = goal_region.goal_frame_offsets(pos, goal_pos, goal_yaw)
    twist = _record_rollouts_twist_yaw()(rd)
    cols = {
        "history": nodes.numpy(),
        "history_null": null.numpy().astype(np.uint8),
        "extero": extero.numpy(),
        "cmd_future": cmd_future.numpy(),
        "proprio": proprio.numpy(),
        "height_scan": scan.numpy(),
        "goal_downrange_m": down.numpy(),
        "goal_lateral_m": lat.numpy(),
        "goal_height_err_m": hz.numpy(),
        "base_twist_yaw": twist.numpy(),
    }
    return ctx, cmd, tracker, ring, cols, scan


def _scorer(ctx, scan: torch.Tensor, *, priv: bool, model=None, goal_context: bool = True):
    """An ``OnlineValueScorer`` whose scan read is stubbed.

    ``_read_scan`` is the module's only dependency on a built scene; overriding it is what
    lets the rest of the assembly be compared for real without a simulator.

    Args:
        ctx: The generator context.
        scan: ``(num_envs, num_rays)`` the stubbed sensor read.
        priv: Which variant to build.
        model: A model to score with; a stub with the right token attributes by default.
        goal_context: Whether the goal offsets are in the context token.

    Returns:
        The scorer.
    """
    m = model if model is not None else _stub_model(priv, goal_context=goal_context)
    s = headroom_value.OnlineValueScorer(
        m, ctx, priv=priv, scan_hw=(SCAN_HW if priv else None), device=DEVICE,
        scan_sensor=("height_scanner" if priv else None), goal_context=goal_context)
    s._read_scan = types.MethodType(lambda self, env: scan, s)
    return s


def _stub_model(priv: bool, *, goal_context: bool = True):
    """A stand-in carrying only the attributes the scorer's constructor validates.

    Args:
        priv: Which variant it claims to be.
        goal_context: Whether its context token includes the goal offsets.

    Returns:
        An object with ``scan_enc`` / ``proprio_in`` / ``context_in`` shaped like the real
        model's, and a ``training`` flag so ``score``'s eval-mode round trip is exercisable.
    """
    ctx_in = torch.nn.Sequential(torch.nn.Linear(
        value.context_width(privileged=priv, goal_context=goal_context), 8))
    return types.SimpleNamespace(
        scan_enc=(types.SimpleNamespace(image_hw=SCAN_HW) if priv else None),
        proprio_in=(object() if priv else None),
        context_in=ctx_in, training=False,
        eval=lambda: None, train=lambda mode=True: None)


def _assert_same(online, offline) -> None:
    """Every field of two ``ValueInputs`` is bit-identical, including which are ``None``.

    Args:
        online: What the scorer built.
        offline: What ``fit_value._batch`` built.
    """
    for f in ("history", "extero", "cmd_tokens", "history_null", "scan", "proprio", "context"):
        a, b = getattr(online, f), getattr(offline, f)
        assert (a is None) == (b is None), f"{f}: online={a is None} offline={b is None}"
        if a is None:
            continue
        assert a.shape == b.shape, f"{f}: {tuple(a.shape)} vs {tuple(b.shape)}"
        assert a.dtype == b.dtype, f"{f}: {a.dtype} vs {b.dtype}"
        assert torch.equal(a, b), (
            f"{f} differs; max |online - offline| = "
            f"{float((a.to(torch.float64) - b.to(torch.float64)).abs().max())}")


# ----------------------------------------------------------------------
# The measurement.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("priv", [True, False])
@pytest.mark.parametrize("goal_context", [True, False])
def test_online_inputs_are_bit_identical_to_fit_value_batch(priv, goal_context):
    """The whole point: same state, same normalizers, same numbers.

    A divergence here means the critic scores in a different input space from the one it was
    fitted in, with every shape still agreeing -- which nothing downstream can see.

    Parametrised over ``goal_context`` because the no-goal arm is the one place the two
    assemblies could diverge WITHOUT a shape error: both drop three numbers, and if one
    dropped them from the front while the other dropped them from the back the twist would
    be read out of the goal's columns and every value would still be finite.
    """
    ctx, cmd, tracker, ring, cols, scan = _world()
    online = _scorer(ctx, scan, priv=priv, goal_context=goal_context).inputs(
        cmd, tracker, ring, object(), torch.as_tensor(IDX))
    offline = fit_value._batch(ctx, cols, IDX, priv, DEVICE, SCAN_HW,
                               goal_context=goal_context)
    _assert_same(online, offline)


@pytest.mark.parametrize("priv", [True, False])
def test_dropping_the_goal_removes_the_GOAL_columns_and_moves_no_others(priv):
    """The ablation takes columns 0:3 away and leaves the twist exactly where it was.

    The goal is PREPENDED in both assemblies precisely so this holds: a with-goal checkpoint's
    twist lives at 3: and a no-goal one's at 0:, and the shared tail is bit-identical. Had
    either side appended instead, the widths would still be 9 and 6 and every column would be
    off by three -- a shift no shape check can see and no value can look wrong.
    """
    ctx, cmd, tracker, ring, _, scan = _world()
    args = (cmd, tracker, ring, object(), torch.as_tensor(IDX))
    with_goal = _scorer(ctx, scan, priv=priv, goal_context=True).inputs(*args).context
    without = _scorer(ctx, scan, priv=priv, goal_context=False).inputs(*args).context
    n_tw = 6 if priv else 3
    assert with_goal.shape == (IDX.size, 3 + n_tw)
    assert without.shape == (IDX.size, n_tw)
    assert torch.equal(without, with_goal[:, 3:])


@pytest.mark.parametrize("priv", [True, False])
def test_a_goal_context_mismatch_between_scorer_and_checkpoint_is_refused(priv):
    """A no-goal model handed a with-goal scorer, and the reverse.

    Both directions are a width disagreement the ctor can see; neither is left to a shape
    error at forward, because for ``priv`` the wrong pairing is 9-against-6 and for the
    onboard variant it is 6-against-3 -- and a 6-wide context is reachable BOTH ways, which
    is why the flag travels as a stamp rather than being read off the width.
    """
    ctx, _, _, _, _, scan = _world()
    for model_goal, scorer_goal in ((False, True), (True, False)):
        with pytest.raises(ValueError, match="context token"):
            _scorer(ctx, scan, priv=priv, goal_context=scorer_goal,
                    model=_stub_model(priv, goal_context=model_goal))


def test_the_privileged_variant_is_the_only_one_carrying_the_linear_twist():
    """``priv`` decides the twist SLICE, not just the extra tokens.

    Both variants' context blocks are 3 goal offsets plus a twist, so the onboard one is 6
    wide and the privileged one 9 -- and the onboard's twist is the ANGULAR half, because
    nothing on the robot measures the linear half without odometry. Getting the halves the
    wrong way round is a 6-wide context either way.
    """
    ctx, cmd, tracker, ring, _, scan = _world()
    on = _scorer(ctx, scan, priv=False).inputs(cmd, tracker, None, None, torch.as_tensor(IDX))
    pr = _scorer(ctx, scan, priv=True).inputs(cmd, tracker, ring, object(),
                                              torch.as_tensor(IDX))
    assert on.context.shape == (IDX.size, 6)
    assert pr.context.shape == (IDX.size, 9)
    # The onboard block is the goal plus the LAST three of the privileged twist.
    assert torch.equal(on.context[:, :3], pr.context[:, :3])
    assert torch.equal(on.context[:, 3:], pr.context[:, 6:])


def test_the_scan_is_sanitised_and_the_extero_row_is_not():
    """The two exteroception fields go through different paths, deliberately.

    ``fit_value._batch`` sanitises the height scan (``-inf`` for a missed ray would take the
    batch's whole logit) and does NOT sanitise the served ``extero`` row, because the
    recording's column is not sanitised either. Doing both would be a second, disagreeing
    clamp on a signal the generator already normalised.
    """
    from robot_rl.flow_matching import transforms  # noqa: PLC0415

    ctx, cmd, tracker, ring, cols, scan = _world()
    out = _scorer(ctx, scan, priv=True).inputs(cmd, tracker, ring, object(),
                                               torch.as_tensor(IDX))
    assert torch.isfinite(out.scan).all(), "a missed ray survived into the batch"
    band = float(out.scan.abs().max())
    assert band <= 1.0 + 1e-6, f"the scan escaped its default band: {band}"
    # A miss maps to +clip, the DROP-OFF end, not to -clip: "no ground within range" is
    # ground very far BELOW, and -clip would spell a hole the same way ascending stairs
    # spell a wall (``transforms.sanitize_height_scan``).
    row0 = int(np.flatnonzero(IDX == 0)[0])
    assert float(out.scan[row0].reshape(-1)[3]) == pytest.approx(1.0)

    # The served row, by contrast, reaches the critic exactly as normalized -- unsanitised
    # and unclamped. The raw row genuinely leaves the band, so this separates the two paths
    # rather than passing because there was nothing to clamp.
    raw = torch.as_tensor(cols["extero"][IDX]).reshape(IDX.size, -1)
    assert float(raw.abs().max()) > 1.0, "the fixture no longer exercises the clamp"
    assert torch.equal(out.extero, ctx.extero_norm.normalize(raw))
    assert not torch.equal(
        out.extero, ctx.extero_norm.normalize(transforms.sanitize_extero(raw, "height_scan")))


def test_the_goal_scale_is_imported_not_restated():
    """One statement of ``GOAL_SCALE_M`` in this process, and it is ``fit_value``'s.

    A second copy is how a scale that is DECLARED rather than fitted quietly becomes two
    scales -- and the feature it divides is one the critic reads directly, so a factor there
    is a systematic offset on every score.
    """
    ctx, _, _, _, _, scan = _world()
    assert _scorer(ctx, scan, priv=False)._goal_scale_m == fit_value.GOAL_SCALE_M


def test_a_boolean_env_id_mask_is_refused():
    """``cmd._fm_reinfer_mask`` is exactly this shape, and coercing it is silent.

    ``as_tensor(mask, dtype=long)`` gives a vector of 0s and 1s: it scores envs 0 and 1 over
    and over, at full width, with every downstream shape still right.
    """
    ctx, cmd, tracker, ring, _, scan = _world()
    mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
    mask[IDX] = True
    with pytest.raises(ValueError, match="boolean mask"):
        _scorer(ctx, scan, priv=False).inputs(cmd, tracker, None, None, mask)


def test_a_variant_mismatch_between_scorer_and_checkpoint_is_refused():
    """``priv`` must agree with the model it was built over, in both directions."""
    ctx, _, _, _, _, scan = _world()
    with pytest.raises(ValueError, match="scan tokens"):
        headroom_value.OnlineValueScorer(_stub_model(True), ctx, priv=False, scan_hw=None,
                                         device=DEVICE, scan_sensor=None)
    with pytest.raises(ValueError, match="scan tokens"):
        headroom_value.OnlineValueScorer(_stub_model(False), ctx, priv=True, scan_hw=SCAN_HW,
                                         device=DEVICE, scan_sensor="height_scanner")


def test_a_transposed_scan_fold_is_refused():
    """176 rays is both 11x16 and 16x11, and the wrong one scrambles every kernel."""
    ctx, _, _, _, _, scan = _world()
    with pytest.raises(ValueError, match="scan encoder was built for"):
        headroom_value.OnlineValueScorer(_stub_model(True), ctx, priv=True,
                                         scan_hw=(SCAN_HW[1], SCAN_HW[0]), device=DEVICE,
                                         scan_sensor="height_scanner")


def test_the_recordings_float16_columns_are_the_only_difference_from_training():
    """What the critic was FITTED on, versus what it is served live.

    ``record_rollouts._emit`` stores ``extero`` and ``height_scan`` as float16 while every
    other field is float32, so the fit saw quantised versions of exactly those two. The live
    buffers are float32 and this module uses them as they are. This pins the difference to
    those two fields and measures it, rather than leaving it as a claim in a docstring.
    """
    ctx, cmd, tracker, ring, cols, scan = _world()
    quantised = dict(cols)
    quantised["extero"] = cols["extero"].astype(np.float16)
    quantised["height_scan"] = cols["height_scan"].astype(np.float16)

    online = _scorer(ctx, scan, priv=True).inputs(cmd, tracker, ring, object(),
                                                  torch.as_tensor(IDX))
    as_fitted = fit_value._batch(ctx, quantised, IDX, True, DEVICE, SCAN_HW)
    for f in ("history", "cmd_tokens", "history_null", "proprio", "context"):
        assert torch.equal(getattr(online, f), getattr(as_fitted, f)), \
            f"{f} is float32 on both sides and must not move"
    for f, tol in (("extero", 5e-3), ("scan", 5e-3)):
        d = float((getattr(online, f) - getattr(as_fitted, f)).abs().max())
        assert 0.0 < d < tol, f"{f}: expected a small fp16 residual, got {d}"


def test_score_runs_the_real_model_and_returns_one_probability_per_env():
    """The batch must actually FEED the critic, not merely look like it.

    Built against a real (tiny) ``FlowValueModel`` over a real prefix-attention
    ``FlowMatchingTransformer``, so the token widths, the readout stream and the borrowed
    encoder all have to agree with what ``inputs`` assembles.
    """
    ctx, cmd, tracker, ring, _, scan = _world()
    fm = fm_model.FlowMatchingTransformer(
        state_dim=D_NODE, future_len=F_FUT, num_history_states=N_HIST, history_dim=D_NODE,
        extero_sensor_dim=EXTERO_SENSOR_DIM, extero_sensor_count=EXTERO_SENSORS,
        cmd_dim=CMD_DIM, cmd_stride=CMD_STRIDE, d_model=32, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0.0, attention_mode="prefix")
    model = value.FlowValueModel(fm, scan_hw=SCAN_HW, proprio_dim=PROPRIO_HW[1],
                                 proprio_steps=PROPRIO_HW[0], context_dim=9)
    model.train()                      # score() must force eval and put it back
    p = _scorer(ctx, scan, priv=True, model=model).score(
        cmd, tracker, ring, object(), torch.as_tensor(IDX))
    assert p.shape == (IDX.size,)
    assert torch.isfinite(p).all() and float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    assert model.training, "score() must restore the model's training flag"


def test_score_never_leaves_the_device_it_was_given():
    """A CPU round trip is the one thing that would make this useless in the physics loop."""
    ctx, cmd, tracker, ring, _, scan = _world()
    out = _scorer(ctx, scan, priv=True).inputs(cmd, tracker, ring, object(),
                                               torch.as_tensor(IDX))
    for f in ("history", "extero", "cmd_tokens", "history_null", "scan", "proprio", "context"):
        t = getattr(out, f)
        assert t.device == torch.device(DEVICE), f"{f} landed on {t.device}"
