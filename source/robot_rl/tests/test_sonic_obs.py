# Copyright 2026 Zachary Olkin. All rights reserved.

"""Unit tests for the SONIC observation port.

These pin the details that a reimplementation gets wrong *silently* -- a wrong slice offset, a
history running newest-first, a 6D rotation flattened column-wise. All of them produce vectors of
exactly the right shape, so nothing but an explicit check catches them.

The tests that need the third-party weights (the ONNX cross-checks) skip when the release has not
been downloaded; everything else runs headless with no simulator and no GR00T clone.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.baselines.sonic.obs import (
    HistoryBuffer,
    MotionReference,
    RobotState,
    anchor_ori_6d,
    build_decoder_input,
    build_encoder_input,
    history_depth,
    history_dims,
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
)
from robot_rl.baselines.sonic.spec import Block, SonicSpec, load_spec

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

NJ = 29
SQRT_HALF = float(np.sqrt(0.5))


# --------------------------------------------------------------------------- fixtures

def _release_config():
    """Path to the shipped release observation config, or None if the clone is absent."""
    try:
        from robot_rl.assets.robots.g1_sonic import find_groot_root
        p = find_groot_root() / "gear_sonic_deploy/policy/release/observation_config.yaml"
    except FileNotFoundError:
        return None
    return p if p.is_file() else None


def _release_onnx():
    """Directory holding the downloaded release ONNX pair, or None."""
    cfg = _release_config()
    if cfg is None:
        return None
    return cfg.parent if (cfg.parent / "model_encoder.onnx").is_file() else None


@pytest.fixture
def spec() -> SonicSpec:
    """The shipped release layout, skipping if the GR00T clone is not available."""
    cfg = _release_config()
    if cfg is None:
        pytest.skip("GR00T-WholeBodyControl clone with the release obs config not available")
    return load_spec(cfg)


# --------------------------------------------------------------------------- layout

def test_release_layout_offsets(spec: SonicSpec) -> None:
    """The g1-mode blocks land on the offsets the released encoder actually reads.

    These three numbers were confirmed against the graph by perturbing each slice in turn with
    the mode id pinned to 0 and watching the 64-d token: only these moved it.
    """
    assert spec.encoder_dim == 1762
    assert spec.decoder_dim == 994
    assert spec.token_dim == 64
    got = {b.name: (b.offset, b.stop) for b in spec.mode_blocks("g1")}
    assert got["encoder_mode_4"] == (0, 4)
    assert got["motion_joint_positions_10frame_step5"] == (4, 294)
    assert got["motion_joint_velocities_10frame_step5"] == (294, 584)
    assert got["motion_anchor_orientation_10frame_step5"] == (601, 661)
    # Reference-root height is present in the layout but NOT required by the g1 mode: this is
    # the "noz" tokenizer, so SONIC gets no root-position target of any kind.
    assert "motion_root_z_position_10frame_step5" not in got
    assert "motion_root_z_position" not in got


def test_release_reference_window(spec: SonicSpec) -> None:
    """The default release samples 10 future frames at stride 5 (0.1 s apart at 50 Hz)."""
    assert spec.reference_frames("g1") == (10, 5)


def test_decoder_layout_order(spec: SonicSpec) -> None:
    """Proprioception blocks appear in the order the graph expects, all 10-frame stride-1."""
    names = [b.name for b in spec.decoder_blocks]
    assert names == [
        "token_state",
        "his_base_angular_velocity_10frame_step1",
        "his_body_joint_positions_10frame_step1",
        "his_body_joint_velocities_10frame_step1",
        "his_last_actions_10frame_step1",
        "his_gravity_dir_10frame_step1",
    ]
    for b in spec.decoder_blocks[1:]:
        assert (b.frames, b.step) == (10, 1)
    assert history_depth(spec) == 10
    assert history_dims(spec) == {"base_ang_vel": 3, "joint_pos_rel": 29,
                                 "joint_vel": 29, "last_action": 29, "gravity_dir": 3}


def test_unknown_observation_name_raises(tmp_path) -> None:
    """An unported observation name fails loudly instead of shifting every later block."""
    cfg = tmp_path / "obs.yaml"
    cfg.write_text(
        "observations:\n"
        "  - name: \"token_state\"\n    enabled: true\n"
        "  - name: \"some_new_thing_we_have_not_ported\"\n    enabled: true\n"
        "encoder:\n  dimension: 64\n  encoder_observations: []\n  encoder_modes: []\n"
    )
    with pytest.raises(NotImplementedError, match="not in the ported registry"):
        load_spec(cfg)


def test_validate_dims_rejects_mismatch(spec: SonicSpec) -> None:
    """A layout that does not sum to the graph's input width is refused."""
    spec.validate_dims(1762, 994)
    with pytest.raises(ValueError, match="sums to 1762 but the ONNX graph declares 1"):
        spec.validate_dims(1, 994)


# --------------------------------------------------------------------------- 6D orientation

def test_anchor_ori_6d_is_two_columns_row_major() -> None:
    """The 6D block is ``[R00, R01, R10, R11, R20, R21]`` of the relative rotation."""
    q_id = np.array([[0.0, 0.0, 0.0, 1.0]])                      # xyzw identity
    q_z90 = np.array([[[0.0, 0.0, SQRT_HALF, SQRT_HALF]]])       # +90 deg about z
    out = anchor_ori_6d(q_id, q_z90)
    assert out.shape == (1, 1, 6)
    np.testing.assert_allclose(out[0, 0], [0, -1, 1, 0, 0, 0], atol=1e-12)


def test_anchor_ori_6d_uses_full_robot_quat_not_heading() -> None:
    """A robot roll/pitch offset changes the block: upstream's mode 0 is not heading-corrected."""
    q_ref = np.array([[[0.0, 0.0, 0.0, 1.0]]])
    upright = np.array([[0.0, 0.0, 0.0, 1.0]])
    # 30 deg about x -- pure roll, which a heading-only formulation would ignore entirely.
    rolled = np.array([[np.sin(np.pi / 12), 0.0, 0.0, np.cos(np.pi / 12)]])
    assert not np.allclose(anchor_ori_6d(upright, q_ref), anchor_ori_6d(rolled, q_ref))


def test_anchor_ori_6d_matches_relative_rotation_matrix() -> None:
    """Cross-check against composing the rotation explicitly, for a random pair."""
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(1, 4)), rng.normal(size=(1, 1, 4))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    rel = quat_mul(quat_conjugate(a)[:, None, :], b)
    expect = matrix_from_quat(rel)[..., :2].reshape(1, 1, 6)
    np.testing.assert_allclose(anchor_ori_6d(a, b), expect, atol=1e-12)


def test_quaternion_helpers_match_the_deploy_runtime() -> None:
    """Parity with ``transfer/deploy/quat_math_np.py``, the repo's tested numpy quat module."""
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[3]
    if not (root / "transfer" / "deploy" / "quat_math_np.py").is_file():
        pytest.skip("transfer/deploy not present")
    sys.path.insert(0, str(root))
    try:
        from transfer.deploy import quat_math_np as ref
    finally:
        sys.path.remove(str(root))
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    np.testing.assert_allclose(quat_mul(a, b), ref.quat_mul(a, b), atol=1e-12)
    np.testing.assert_allclose(quat_conjugate(a), ref.quat_conjugate(a), atol=1e-12)
    np.testing.assert_allclose(matrix_from_quat(a), ref.matrix_from_quat(a), atol=1e-12)


# --------------------------------------------------------------------------- history buffer

def _state(n: int, fill: float) -> RobotState:
    """A RobotState whose every field is a constant, for identifying frames by value."""
    return RobotState(
        root_quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        base_ang_vel=np.full((n, 3), fill),
        joint_pos_rel=np.full((n, NJ), fill),
        joint_vel=np.full((n, NJ), fill),
        gravity_dir=np.full((n, 3), fill),
        last_action=np.full((n, NJ), fill),
    )


def test_history_warm_start_fills_the_whole_depth() -> None:
    """The first append after a reset fills every frame, matching IsaacLab's CircularBuffer.

    Without this the first ten control steps would be fed a part-zero history the policy never
    saw in training.
    """
    h = HistoryBuffer(2, 10, {"base_ang_vel": 3})
    h.reset()
    h.append(_state(2, 5.0))
    np.testing.assert_allclose(h.stack("base_ang_vel", 10, 1), np.full((2, 10, 3), 5.0))


def test_history_is_oldest_first() -> None:
    """``stack`` returns ``[oldest, ..., newest]``, as both IsaacLab and the C++ deploy do."""
    h = HistoryBuffer(1, 10, {"base_ang_vel": 3})
    h.reset()
    for v in range(1, 5):
        h.append(_state(1, float(v)))
    got = h.stack("base_ang_vel", 4, 1)[0, :, 0]
    np.testing.assert_allclose(got, [1.0, 2.0, 3.0, 4.0])


def test_history_stride_samples_every_step_th_frame() -> None:
    """A stride-2 window takes every other sample, still ending at the newest."""
    h = HistoryBuffer(1, 10, {"base_ang_vel": 3})
    h.reset()
    for v in range(1, 8):
        h.append(_state(1, float(v)))
    np.testing.assert_allclose(h.stack("base_ang_vel", 3, 2)[0, :, 0], [3.0, 5.0, 7.0])


def test_history_reset_is_per_env() -> None:
    """Resetting one env re-warm-starts only that env, as terminations do mid-rollout."""
    h = HistoryBuffer(2, 4, {"base_ang_vel": 3})
    h.reset()
    for v in (1.0, 2.0):
        h.append(_state(2, v))
    h.reset(np.array([1]))
    h.append(_state(2, 9.0))
    got = h.stack("base_ang_vel", 4, 1)[:, :, 0]
    np.testing.assert_allclose(got[0], [1.0, 1.0, 2.0, 9.0])     # env 0 keeps its history
    np.testing.assert_allclose(got[1], [9.0, 9.0, 9.0, 9.0])     # env 1 warm-started


def test_history_rejects_a_window_deeper_than_the_buffer() -> None:
    """Asking for more history than is retained is an error, not a silent short window."""
    h = HistoryBuffer(1, 4, {"base_ang_vel": 3})
    h.reset()
    h.append(_state(1, 1.0))
    with pytest.raises(ValueError, match="needs 5 samples"):
        h.stack("base_ang_vel", 5, 1)


# --------------------------------------------------------------------------- assembly

def _reference(n: int, frames: int) -> MotionReference:
    """A reference window whose values encode (frame, joint) so packing order is visible."""
    jp = np.arange(frames * NJ, dtype=np.float64).reshape(1, frames, NJ).repeat(n, axis=0)
    return MotionReference(
        joint_pos=jp,
        joint_vel=jp + 1000.0,
        root_quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, frames, 1)),
    )


def test_encoder_packs_reference_frame_major(spec: SonicSpec) -> None:
    """Frame 0's 29 joints come first, then frame 1's -- not joint-major."""
    frames, _ = spec.reference_frames("g1")
    out = build_encoder_input(spec, "g1", _reference(1, frames),
                              np.array([[0.0, 0.0, 0.0, 1.0]]))
    jp = spec.encoder_block("motion_joint_positions_10frame_step5")
    block = out[0, jp.offset:jp.stop].reshape(frames, NJ)
    np.testing.assert_allclose(block, np.arange(frames * NJ).reshape(frames, NJ))


def test_encoder_zero_fills_everything_the_mode_does_not_use(spec: SonicSpec) -> None:
    """Only the g1 mode's own blocks are written; the teleop and SMPL slots stay zero."""
    frames, _ = spec.reference_frames("g1")
    out = build_encoder_input(spec, "g1", _reference(1, frames),
                              np.array([[0.0, 0.0, 0.0, 1.0]]))
    used = {b.name for b in spec.mode_blocks("g1")}
    for b in spec.encoder_blocks:
        if b.name not in used:
            np.testing.assert_allclose(out[0, b.offset:b.stop], 0.0,
                                       err_msg=f"block '{b.name}' should be zero for mode g1")
    assert out[0, 0] == 0.0                                     # g1 mode id
    np.testing.assert_allclose(out[0, 1:4], 0.0)                # mode padding


def test_encoder_rejects_a_mismatched_reference_window(spec: SonicSpec) -> None:
    """A window with the wrong frame count is refused rather than silently truncated."""
    bad = _reference(1, 3)
    with pytest.raises(ValueError, match="built 87 values, layout wants 290"):
        build_encoder_input(spec, "g1", bad, np.array([[0.0, 0.0, 0.0, 1.0]]))


def test_encoder_requires_root_z_only_when_the_mode_asks_for_it() -> None:
    """A variant whose mode needs ``motion_root_z_position`` says so instead of feeding zeros."""
    spec = SonicSpec(
        decoder_blocks=(Block("token_state", 0, 64, 1, 1),),
        encoder_blocks=(Block("encoder_mode_4", 0, 4, 1, 1),
                        Block("motion_root_z_position", 4, 1, 1, 1)),
        modes={"g1": __import__("robot_rl.baselines.sonic.spec", fromlist=["Mode"]).Mode(
            "g1", 0, frozenset({"encoder_mode_4", "motion_root_z_position"}))},
        token_dim=64,
        path=__import__("pathlib").Path("synthetic"),
    )
    ref = MotionReference(joint_pos=np.zeros((1, 1, NJ)), joint_vel=np.zeros((1, 1, NJ)),
                          root_quat=np.tile([0.0, 0.0, 0.0, 1.0], (1, 1, 1)))
    with pytest.raises(ValueError, match="root_z is None"):
        build_encoder_input(spec, "g1", ref, np.array([[0.0, 0.0, 0.0, 1.0]]))


def test_decoder_packs_history_blocks_in_layout_order(spec: SonicSpec) -> None:
    """Each proprioception block lands on its own slice, oldest-first within the block."""
    h = HistoryBuffer(1, history_depth(spec), history_dims(spec))
    h.reset()
    for v in range(1, 11):
        h.append(_state(1, float(v)))
    out = build_decoder_input(spec, np.full((1, 64), 0.5), h)
    np.testing.assert_allclose(out[0, :64], 0.5)
    for name, width in (("his_base_angular_velocity_10frame_step1", 3),
                        ("his_body_joint_positions_10frame_step1", NJ),
                        ("his_gravity_dir_10frame_step1", 3)):
        b = spec.decoder_block(name)
        got = out[0, b.offset:b.stop].reshape(10, width)
        np.testing.assert_allclose(got[:, 0], np.arange(1.0, 11.0), err_msg=name)


# --------------------------------------------------------------------------- against the graphs

@pytest.mark.parametrize("mode_id,expected_live", [
    (0, {"motion_joint_positions_10frame_step5", "motion_joint_velocities_10frame_step5",
         "motion_anchor_orientation_10frame_step5"}),
])
def test_unused_slices_are_inert(spec: SonicSpec, mode_id: int, expected_live: set[str]) -> None:
    """With the mode id pinned, only that mode's blocks move the token.

    This is the empirical basis for zero-filling the other 1100-odd inputs, and it is what makes
    the g1 mode usable at all: the released encoder runs all three encoders and selects one by a
    one-hot weighted sum, so the unused branches' inputs cannot matter -- but only a measurement
    proves the offsets in this repo's layout line up with the graph's.
    """
    model_dir = _release_onnx()
    if model_dir is None:
        pytest.skip("release ONNX pair not downloaded")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_dir / "model_encoder.onnx"),
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    base = np.zeros((1, spec.encoder_dim), np.float32)
    base[0, 0] = float(mode_id)
    tok0 = sess.run(None, {name: base})[0]
    rng = np.random.default_rng(0)
    for b in spec.encoder_blocks:
        if b.name == "encoder_mode_4":
            continue
        pert = base.copy()
        pert[0, b.offset:b.stop] = rng.normal(0.0, 0.5, b.dim).astype(np.float32)
        moved = not np.array_equal(sess.run(None, {name: pert})[0], tok0)
        assert moved == (b.name in expected_live), (
            f"block '{b.name}' {'moved' if moved else 'did not move'} the token for mode "
            f"{mode_id}; layout and graph disagree about its offset."
        )


def test_policy_loads_and_steps_against_the_real_graphs() -> None:
    """End-to-end: the layout validates against both graphs and a step produces 29 actions."""
    model_dir = _release_onnx()
    if model_dir is None:
        pytest.skip("release ONNX pair not downloaded")
    from robot_rl.baselines.sonic.policy import ACTION_CLIP, SonicPolicy
    pol = SonicPolicy(model_dir, num_envs=2).load()
    frames, step = pol.reference_frames
    assert (frames, step) == (10, 5)
    pol.reset()
    act = pol.act(_state(2, 0.0), _reference(2, frames))
    assert act.shape == (2, 29) and act.dtype == np.float32
    assert np.all(np.abs(act) <= ACTION_CLIP)
    # The action feedback must reach the next step's history, or the policy is driven open-loop.
    np.testing.assert_allclose(pol.last_action, act, atol=0)
