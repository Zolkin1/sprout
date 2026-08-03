# Copyright 2026 Zachary Olkin. All rights reserved.

"""``reference_window`` / ``command_window`` against a stub command term.

The window observation is pure geometry over the command term's look-ahead accessors, so
it can be checked without a simulator by standing in a fake command whose clip is a known
rigid transform of a known trajectory.

The load-bearing property is the **offset-0 anchor**: sample 0 of the window must equal
the current-step observations it replaced (:func:`base_pose_error`,
:func:`ref_base_velocity_local`, :func:`ee_ref_relative_to_base`), otherwise the merged
term has quietly changed the reference the policy tracks.  The joint blocks are the one
deliberate exception -- the window reports them in ROBOT joint order while
:func:`joint_pos_ref` reports clip-OUTPUT order.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
# The observation module needs isaaclab's quaternion math and warp's tensor bridge; both
# import fine without a running simulator, but skip cleanly where they are absent.
wp = pytest.importorskip("warp", exc_type=ImportError)
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

# The stub hands the observation functions warp arrays (as ArticulationData does), and
# warp's torch bridge needs a live runtime to resolve the device.
try:
    wp.init()
except Exception as exc:  # pragma: no cover - environment without a usable warp runtime
    pytest.skip(f"warp runtime unavailable: {exc}", allow_module_level=True)

from isaaclab.utils.math import quat_apply, quat_box_minus, quat_conjugate, quat_mul  # noqa: E402

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (  # noqa: E402
    MotionSequenceCommand,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.observations import (  # noqa: E402
    REFERENCE_WINDOW_NOISE,
    ReferenceWindowNoiseCfg,
    _window_offsets,
    base_pose_error,
    command_window,
    ee_ref_relative_to_base,
    joint_pos_ref,
    per_channel_uniform_noise,
    ref_base_velocity_local,
    reference_window,
    reference_window_layout,
    reference_window_layout_from_dims,
    reference_window_noise_vector,
    reference_window_scale,
    reference_window_to_world,
)

N_ENVS, N_JOINTS, T = 3, 4, 60
EE_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
BASE = "pelvis_link"


def _rand_quat(n: int) -> torch.Tensor:
    """``n`` random unit quaternions, xyzw (Isaac Lab's convention)."""
    q = torch.randn(n, 4)
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def _wp(tensor: torch.Tensor):
    """Wrap a tensor as a warp array, as the real ``ArticulationData`` stores root state.

    The observation functions read the root pose through ``wp.to_torch``, so the stub has
    to hand back warp arrays rather than plain tensors.
    """
    return wp.from_torch(tensor.contiguous())


class _StubCommand:
    """Minimal stand-in for ``MotionSequenceCommand``'s look-ahead surface.

    Holds a per-(env, step) world-frame trajectory for the base and each ee frame plus
    joint states, and serves exactly the accessors the observation functions call. The
    current step is index ``step0``, so offset ``k`` reads ``step0 + k``.
    """

    def __init__(self, seed: int = 0) -> None:
        """Build a random but self-consistent reference trajectory."""
        torch.manual_seed(seed)
        self.device = "cpu"
        self.num_envs = N_ENVS
        self._step0 = torch.arange(N_ENVS) * 2  # different phase per env
        self.cfg = SimpleNamespace(base_frame_name=BASE)
        self.manager = SimpleNamespace(dt=0.02, num_steps=T)

        self.base_pos = torch.randn(N_ENVS, T, 3)
        self.base_quat = _rand_quat(N_ENVS * T).view(N_ENVS, T, 4)
        self.base_lin = torch.randn(N_ENVS, T, 3)
        self.base_ang = torch.randn(N_ENVS, T, 3)
        self.joint_pos = torch.randn(N_ENVS, T, N_JOINTS)
        self.joint_vel = torch.randn(N_ENVS, T, N_JOINTS)
        # ee poses in the base frame, lifted to world by the base transform.
        self.ee_local = torch.randn(N_ENVS, T, len(EE_FRAMES), 3)
        self.target_vel = torch.randn(N_ENVS, T, 3)

        # The measured robot root, which the base blocks are expressed relative to.
        # ``joint_names`` sits on both the articulation and its data, as on the real one.
        joint_names = [f"joint_{i}" for i in range(N_JOINTS)]
        self.robot = SimpleNamespace(
            joint_names=joint_names,
            data=SimpleNamespace(
                root_pos_w=_wp(torch.randn(N_ENVS, 3)),
                root_quat_w=_wp(_rand_quat(N_ENVS)),
                joint_names=joint_names,
            ),
        )

        # Clip-output vectors for the current step, in CLIP-OUTPUT order: base pose/ori
        # first, then the ee frames, then the joints (reversed, so output order and robot
        # order genuinely differ and the test can tell them apart).
        self.ordered_pos_output_names = (
            [f"{BASE}:pos_{a}" for a in "xyz"]
            + [f"{BASE}:ori_{a}" for a in ("x", "y", "z", "w")]
            + [f"{f}:pos_{a}" for f in EE_FRAMES for a in "xyz"]
            + [f"joint:joint_{i}" for i in reversed(range(N_JOINTS))]
        )
        self.ordered_vel_output_names = (
            [f"{BASE}:pos_{a}" for a in "xyz"]
            + [f"{BASE}:ori_{a}" for a in "xyz"]
            + [f"joint:joint_{i}" for i in reversed(range(N_JOINTS))]
        )
        self.y_des = torch.cat([self._cur(self.base_pos), self._cur(self.base_quat)]
                               + [self._ee_world(i) for i in range(len(EE_FRAMES))]
                               + [self._cur(self.joint_pos).flip(-1)], dim=-1)
        self.dy_des = torch.cat(
            [self._cur(self.base_lin), self._cur(self.base_ang), self._cur(self.joint_vel).flip(-1)], dim=-1
        )

    # -- helpers ----------------------------------------------------------
    def _cur(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather ``tensor[env, step0]`` for every env."""
        return tensor[torch.arange(N_ENVS), self._step0]

    def _ee_world(self, frame_idx: int) -> torch.Tensor:
        """Current-step world position of one ee frame."""
        return quat_apply(self._cur(self.base_quat), self._cur(self.ee_local)[:, frame_idx]) + self._cur(
            self.base_pos
        )

    def _idx(self, offsets: torch.Tensor) -> torch.Tensor:
        """``[N, K]`` step indices, clamped to the clip end like the real command."""
        return (self._step0[:, None] + offsets[None, :]).clamp(max=T - 1)

    def _gather(self, tensor: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Gather a ``[N, T, ...]`` trajectory at ``[N, K]`` step indices."""
        steps = self._idx(offsets)
        return tensor[torch.arange(N_ENVS)[:, None].expand_as(steps), steps]

    # -- the accessor surface the observations use ------------------------
    def resolve_ee_frames(self, frame_names=None) -> list[str]:
        """Ordered ee frame names."""
        return list(EE_FRAMES) if frame_names is None else list(frame_names)

    def get_future_world_base(self, offsets):
        """World-frame base pose+velocity over ``offsets``."""
        return (
            self._gather(self.base_pos, offsets),
            self._gather(self.base_quat, offsets),
            self._gather(self.base_lin, offsets),
            self._gather(self.base_ang, offsets),
        )

    def get_future_joint_state(self, offsets):
        """Joint pos/vel over ``offsets``, in ROBOT joint order."""
        return self._gather(self.joint_pos, offsets), self._gather(self.joint_vel, offsets)

    def get_future_world_frames(self, frame_names, offsets):
        """``[N, K, F, 13]`` world-frame ee state over ``offsets``."""
        pos_w, quat_w, _, _ = self.get_future_world_base(offsets)
        local = self._gather(self.ee_local, offsets)  # [N, K, F, 3]
        n, k = pos_w.shape[:2]
        out = []
        for i, _ in enumerate(frame_names):
            world = quat_apply(quat_w.reshape(n * k, 4), local[:, :, i].reshape(n * k, 3)).reshape(n, k, 3)
            world = world + pos_w
            out.append(torch.cat([world, quat_w, torch.zeros(n, k, 6)], dim=-1))
        return torch.stack(out, dim=2)

    def get_future_command_velocity(self, offsets):
        """Joystick command over ``offsets``."""
        return self._gather(self.target_vel, offsets)

    def reference_world_state(self) -> torch.Tensor:
        """``[base_pos, base_quat, base_lin, base_ang, joint_pos, joint_vel]`` at the current step."""
        return torch.cat(
            [self._cur(self.base_pos), self._cur(self.base_quat), self._cur(self.base_lin),
             self._cur(self.base_ang), self._cur(self.joint_pos), self._cur(self.joint_vel)],
            dim=-1,
        )

    # The window builder itself is the PRODUCTION method, bound onto the stub rather than
    # re-implemented here: it is the thing under test, and a copy in the test would agree
    # with itself no matter what the command actually does.  It only touches the accessor
    # surface above plus ``self.robot.data``, so it runs unmodified against the stub.
    reference_window_obs = MotionSequenceCommand.reference_window_obs


class _StubEnv:
    """An environment exposing just ``command_manager.get_term`` and ``num_envs``."""

    def __init__(self, cmd: _StubCommand) -> None:
        """Wrap a stub command term."""
        self.num_envs = N_ENVS
        self.command_manager = SimpleNamespace(get_term=lambda _name: cmd)


@pytest.fixture
def env_cmd():
    """A stub env / command pair."""
    cmd = _StubCommand()
    return _StubEnv(cmd), cmd


def test_window_offsets_include_the_current_step():
    """Offsets start at 0, so ``num_steps=1`` is the current step alone."""
    cmd = SimpleNamespace(device="cpu")
    assert _window_offsets(cmd, 1, 5).tolist() == [0]
    assert _window_offsets(cmd, 11, 5).tolist() == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # Degenerate inputs clamp rather than producing an empty window.
    assert _window_offsets(cmd, 0, 0).tolist() == [0]


def test_layout_matches_the_robot_and_clip(env_cmd):
    """Block widths follow the joint count and ee frame count."""
    _, cmd = env_cmd
    assert reference_window_layout(cmd) == [
        ("base_pose", 6), ("base_vel", 6),
        ("joint_pos", N_JOINTS), ("joint_vel", N_JOINTS),
        ("ee_pos", 3 * len(EE_FRAMES)),
    ]


def test_window_shape(env_cmd):
    """``[N, num_steps * D]`` with D from the layout."""
    env, cmd = env_cmd
    per_step = sum(w for _, w in reference_window_layout(cmd))
    assert reference_window(env, "traj_ref", num_steps=11, stride=5).shape == (N_ENVS, 11 * per_step)
    assert reference_window(env, "traj_ref", num_steps=1, stride=5).shape == (N_ENVS, per_step)


def test_sample_zero_matches_the_current_step_terms(env_cmd):
    """Offset 0 reproduces the single-step observations the window replaced."""
    env, cmd = env_cmd
    per_step = sum(w for _, w in reference_window_layout(cmd))
    first = reference_window(env, "traj_ref", num_steps=11, stride=5)[:, :per_step]

    torch.testing.assert_close(first[:, 0:6], base_pose_error(env, "traj_ref"))
    torch.testing.assert_close(first[:, 6:12], ref_base_velocity_local(env, "traj_ref"))
    torch.testing.assert_close(first[:, -3 * len(EE_FRAMES):], ee_ref_relative_to_base(env, "traj_ref"))


def test_base_pose_uses_the_legacy_mixed_convention(env_cmd):
    """``base_pose`` is position ``ref - actual`` and orientation ``actual - ref``.

    Pinned against an independent expression rather than against the window's own inverse,
    because every other check here is a round trip or a cross-check between two functions
    that share the convention -- flip them together and nothing notices.

    The two halves genuinely disagree (see
    :data:`robot_rl.flow_matching.window.BASE_POSE_CONVENTION`), and this test exists to
    keep that FROM CHANGING BY ACCIDENT rather than to endorse it: the observation width is
    identical either way, so a flip silently invalidates every trained policy. When the
    convention is unified, this test and ``BASE_POSE_CONVENTION`` move together, and a
    teacher retrain follows.
    """
    env, cmd = env_cmd
    offs = _window_offsets(cmd, 11, 5)
    ref_p, ref_q, _, _ = cmd.get_future_world_base(offs)
    n, k = ref_p.shape[:2]
    act_p = wp.to_torch(cmd.robot.data.root_pos_w)[:, None, :].expand(n, k, 3).reshape(-1, 3)
    act_q = wp.to_torch(cmd.robot.data.root_quat_w)[:, None, :].expand(n, k, 4).reshape(-1, 4)

    per_step = sum(w for _, w in reference_window_layout(cmd))
    base_pose = reference_window(env, "traj_ref", num_steps=11, stride=5)
    base_pose = base_pose.reshape(n, k, per_step)[..., 0:6].reshape(-1, 6)

    # position: R_act^T (p_ref - p_act)   -- points robot -> reference
    want_pos = quat_apply(quat_conjugate(act_q), ref_p.reshape(-1, 3) - act_p)
    # orientation: R_act^T Log(R_act R_ref^T) == Log(R_ref^T R_act)  -- the OTHER way
    want_ori = quat_apply(
        quat_conjugate(act_q), quat_box_minus(act_q, ref_q.reshape(-1, 4))
    )
    torch.testing.assert_close(base_pose[:, 0:3], want_pos, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(base_pose[:, 3:6], want_ori, atol=1e-6, rtol=1e-6)

    # ...and both vanish together when the robot sits exactly on the reference. True under
    # either convention, so not the discriminating check -- but it is the property a reader
    # expects, and it fails loudly if either half ever picks up an offset.
    cmd.robot.data.root_pos_w = _wp(ref_p[:, 0, :].contiguous())
    cmd.robot.data.root_quat_w = _wp(ref_q[:, 0, :].contiguous())
    on_ref = reference_window(env, "traj_ref", num_steps=11, stride=5)
    on_ref = on_ref.reshape(n, k, per_step)[:, 0, 0:6]
    torch.testing.assert_close(on_ref, torch.zeros_like(on_ref), atol=1e-6, rtol=1e-6)


def test_layout_from_dims_matches_the_command_layout(env_cmd):
    """The dimension-only layout is the same table the command-based one builds."""
    _, cmd = env_cmd
    assert reference_window_layout_from_dims(N_JOINTS, len(EE_FRAMES)) == reference_window_layout(cmd)


def _to_world(env, cmd, num_steps: int = 11, stride: int = 5) -> dict:
    """Round-trip the window back to world through :func:`reference_window_to_world`."""
    window = reference_window(env, "traj_ref", num_steps=num_steps, stride=stride)
    return reference_window_to_world(
        window,
        wp.to_torch(cmd.robot.data.root_pos_w),
        wp.to_torch(cmd.robot.data.root_quat_w),
        num_steps=num_steps, num_joints=N_JOINTS, num_ee_frames=len(EE_FRAMES),
    )


def test_window_to_world_recovers_the_reference(env_cmd):
    """The inverse reproduces exactly the world-frame reference the command served.

    This is the load-bearing check behind the viser overlay: the debugger draws the
    reference the POLICY saw by inverting the observation, so if the inverse and the
    forward transform ever disagree the overlay would show a phantom frame error.
    """
    env, cmd = env_cmd
    offs = _window_offsets(cmd, 11, 5)
    pos_w, quat_w, lin_w, ang_w = cmd.get_future_world_base(offs)
    out = _to_world(env, cmd)

    torch.testing.assert_close(out["base_pos_w"], pos_w, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["base_lin_vel_w"], lin_w, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["base_ang_vel_w"], ang_w, atol=1e-5, rtol=1e-5)
    # quat_box_plus normalises the sign, so compare the ROTATION: q_out * q_in^-1 is
    # identity up to sign, i.e. |w| == 1 and the vector part is zero.
    n, k = quat_w.shape[:2]
    diff = quat_mul(out["base_quat_w"].reshape(n * k, 4),
                    quat_conjugate(quat_w.reshape(n * k, 4)))
    torch.testing.assert_close(diff[:, :3], torch.zeros(n * k, 3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(diff[:, 3].abs(), torch.ones(n * k), atol=1e-5, rtol=1e-5)


def test_window_to_world_recovers_the_ee_frames_and_joints(env_cmd):
    """ee positions come back in world; the joint blocks pass through untouched."""
    env, cmd = env_cmd
    offs = _window_offsets(cmd, 11, 5)
    ee_w = cmd.get_future_world_frames(EE_FRAMES, offs)[..., 0:3]   # [N, K, F, 3]
    joint_pos, joint_vel = cmd.get_future_joint_state(offs)
    out = _to_world(env, cmd)

    torch.testing.assert_close(out["ee_pos_w"], ee_w, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["joint_pos"], joint_pos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["joint_vel"], joint_vel, atol=1e-5, rtol=1e-5)


def test_window_to_world_rejects_a_width_mismatch(env_cmd):
    """A window built for a different layout fails loudly instead of reshaping silently."""
    env, cmd = env_cmd
    window = reference_window(env, "traj_ref", num_steps=11, stride=5)
    with pytest.raises(ValueError, match="implies"):
        reference_window_to_world(
            window, wp.to_torch(cmd.robot.data.root_pos_w), wp.to_torch(cmd.robot.data.root_quat_w),
            num_steps=11, num_joints=N_JOINTS + 1, num_ee_frames=len(EE_FRAMES),
        )


def test_joint_block_is_in_robot_order(env_cmd):
    """The window reports joints in robot order -- a permutation of ``joint_pos_ref``.

    Deliberate: it lines the reference joints up element-wise with the policy's own
    ``joint_pos`` / ``joint_vel`` / ``actions`` terms. The stub's clip-output order is
    reversed relative to robot order, so a naive implementation would fail here.
    """
    env, cmd = env_cmd
    first = reference_window(env, "traj_ref", num_steps=1, stride=5)
    joint_block = first[:, 12:12 + N_JOINTS]
    torch.testing.assert_close(joint_block, cmd._cur(cmd.joint_pos))
    # Same values, opposite order: a permutation, not a different quantity.
    torch.testing.assert_close(joint_block, joint_pos_ref(env, "traj_ref").flip(-1))


def test_later_samples_read_later_clip_steps(env_cmd):
    """Sample k is the reference at step0 + k*stride, not a repeat of the present."""
    env, cmd = env_cmd
    num_steps, stride = 4, 5
    per_step = sum(w for _, w in reference_window_layout(cmd))
    win = reference_window(env, "traj_ref", num_steps=num_steps, stride=stride).view(
        N_ENVS, num_steps, per_step
    )
    offsets = _window_offsets(cmd, num_steps, stride)
    expected = cmd._gather(cmd.joint_pos, offsets)
    torch.testing.assert_close(win[:, :, 12:12 + N_JOINTS], expected)


def test_base_blocks_are_heading_invariant(env_cmd):
    """Rotating robot and reference together about z leaves the window unchanged.

    Everything the policy sees must be relative, so a pure world-heading change cannot
    move the observation -- that is what makes it deployable.
    """
    env, cmd = env_cmd
    before = reference_window(env, "traj_ref", num_steps=6, stride=5)

    yaw = torch.tensor(0.7)
    rot = torch.tensor([0.0, 0.0, torch.sin(yaw / 2), torch.cos(yaw / 2)]).expand(N_ENVS, 4).contiguous()

    def _rot_traj(vecs):
        n, t = vecs.shape[:2]
        return quat_apply(rot[:, None, :].expand(n, t, 4).reshape(n * t, 4), vecs.reshape(n * t, 3)).reshape(n, t, 3)

    cmd.base_pos = _rot_traj(cmd.base_pos)
    cmd.base_lin = _rot_traj(cmd.base_lin)
    cmd.base_ang = _rot_traj(cmd.base_ang)
    n, t = cmd.base_quat.shape[:2]
    cmd.base_quat = quat_mul(
        rot[:, None, :].expand(n, t, 4).reshape(n * t, 4), cmd.base_quat.reshape(n * t, 4)
    ).reshape(n, t, 4)
    cmd.robot.data.root_pos_w = _wp(quat_apply(rot, wp.to_torch(cmd.robot.data.root_pos_w)))
    cmd.robot.data.root_quat_w = _wp(quat_mul(rot, wp.to_torch(cmd.robot.data.root_quat_w)))

    torch.testing.assert_close(reference_window(env, "traj_ref", num_steps=6, stride=5), before,
                               atol=1e-5, rtol=1e-5)


def test_ee_block_is_relative_to_its_own_reference_base(env_cmd):
    """ee positions are posture (rel. the reference base at that step), free of drift."""
    env, cmd = env_cmd
    num_steps, stride = 3, 5
    per_step = sum(w for _, w in reference_window_layout(cmd))
    win = reference_window(env, "traj_ref", num_steps=num_steps, stride=stride).view(
        N_ENVS, num_steps, per_step
    )
    offsets = _window_offsets(cmd, num_steps, stride)
    torch.testing.assert_close(
        win[:, :, -3 * len(EE_FRAMES):],
        cmd._gather(cmd.ee_local, offsets).flatten(start_dim=2),
        atol=1e-5, rtol=1e-5,
    )


def test_command_window(env_cmd):
    """The joystick window is the raw command at the same offsets."""
    env, cmd = env_cmd
    num_steps, stride = 4, 5
    got = command_window(env, "traj_ref", num_steps=num_steps, stride=stride)
    expected = cmd._gather(cmd.target_vel, _window_offsets(cmd, num_steps, stride))
    assert got.shape == (N_ENVS, num_steps * 3)
    torch.testing.assert_close(got.view(N_ENVS, num_steps, 3), expected)


def test_window_saturates_at_the_clip_end(env_cmd):
    """Offsets past the clip clamp to the last frame instead of wrapping or erroring."""
    env, cmd = env_cmd
    cmd._step0 = torch.full((N_ENVS,), T - 3)
    per_step = sum(w for _, w in reference_window_layout(cmd))
    win = reference_window(env, "traj_ref", num_steps=5, stride=5).view(N_ENVS, 5, per_step)
    # Samples 1..4 all land past the end, so they hold the final frame.
    for k in range(2, 5):
        torch.testing.assert_close(win[:, k], win[:, 1])


# ----------------------------------------------------------------------
# Per-channel noise
# ----------------------------------------------------------------------

def test_noise_vector_matches_the_window_layout(env_cmd):
    """One half-width per channel, in the same block order the window packs."""
    _, cmd = env_cmd
    num_steps = 4
    vec = reference_window_noise_vector(num_steps, N_JOINTS, len(EE_FRAMES))
    per_step = sum(w for _, w in reference_window_layout(cmd))
    assert len(vec) == num_steps * per_step

    step = vec[:per_step]
    m = REFERENCE_WINDOW_NOISE
    assert step[0:3] == list(m["base_pos"])                       # per-axis, z tighter
    assert step[3:6] == [m["base_ori"]] * 3
    assert step[6:9] == [m["base_lin_vel"]] * 3
    assert step[9:12] == [m["base_ang_vel"]] * 3
    assert step[12:12 + N_JOINTS] == [m["joint_pos"]] * N_JOINTS
    assert step[12 + N_JOINTS:12 + 2 * N_JOINTS] == [m["joint_vel"]] * N_JOINTS
    assert step[12 + 2 * N_JOINTS:] == list(m["ee_pos"]) * len(EE_FRAMES)
    # Every sample in the window carries the same magnitudes.
    assert vec[-per_step:] == step


def test_noise_vector_rejects_a_layout_mismatch():
    """A magnitudes dict that cannot fill the layout fails loudly, not silently."""
    bad = dict(REFERENCE_WINDOW_NOISE, ee_pos=[0.05, 0.05])  # 2 axes, not 3
    with pytest.raises(ValueError, match="drifted from window_blocks"):
        reference_window_noise_vector(2, N_JOINTS, len(EE_FRAMES), bad)


def test_per_channel_noise_respects_each_half_width():
    """Noise stays inside the per-channel bound and is not shared across channels."""
    half = [0.0, 0.01, 0.5]
    cfg = ReferenceWindowNoiseCfg(n_half=half, magnitudes={})
    data = torch.zeros(4096, 3)
    out = per_channel_uniform_noise(data, cfg)
    peak = out.abs().amax(dim=0)
    assert peak[0] == 0.0                       # a zero half-width injects nothing
    assert 0.005 < peak[1] <= 0.01              # tight channel stays tight
    assert 0.25 < peak[2] <= 0.5                # wide channel spans its own range
    assert (out.abs() <= torch.tensor(half)).all()


def test_per_channel_noise_rejects_a_width_mismatch():
    """A vector that does not match the observation width raises rather than broadcasting."""
    cfg = ReferenceWindowNoiseCfg(n_half=[0.1] * 5, magnitudes={})
    with pytest.raises(ValueError, match="channels but the observation has"):
        per_channel_uniform_noise(torch.zeros(2, 7), cfg)


def test_per_channel_noise_rejects_unsupported_operation():
    """Only additive noise is implemented; anything else must not silently add."""
    cfg = ReferenceWindowNoiseCfg(n_half=[0.1], magnitudes={}, operation="scale")
    with pytest.raises(ValueError, match="only supports operation='add'"):
        per_channel_uniform_noise(torch.zeros(2, 1), cfg)


def test_scale_is_a_tuple_matching_the_layout(env_cmd):
    """``ObservationTermCfg.scale`` rejects lists, so this must be a tuple of the right width."""
    _, cmd = env_cmd
    num_steps = 3
    scale = reference_window_scale(num_steps, N_JOINTS, len(EE_FRAMES))
    per_step = sum(w for _, w in reference_window_layout(cmd))
    assert isinstance(scale, tuple)
    assert len(scale) == num_steps * per_step


def test_only_the_joint_velocity_block_is_scaled(env_cmd):
    """Joint velocities drop to the proprioceptive 0.05; every other block is untouched."""
    _, cmd = env_cmd
    num_steps = 3
    per_step = sum(w for _, w in reference_window_layout(cmd))
    step = reference_window_scale(num_steps, N_JOINTS, len(EE_FRAMES))[:per_step]
    jv = slice(12 + N_JOINTS, 12 + 2 * N_JOINTS)
    assert set(step[jv]) == {0.05}
    assert set(step[:jv.start]) == {1.0}
    assert set(step[jv.stop:]) == {1.0}
    # Count across the whole window, the way the dumped config reads.
    full = reference_window_scale(num_steps, N_JOINTS, len(EE_FRAMES))
    assert sum(1 for v in full if v == 0.05) == num_steps * N_JOINTS


def test_scale_and_noise_share_one_layout():
    """Both builders expand through the same helper, so a bad layout fails in both."""
    bad = {k: 1.0 for k in ("base_pos", "base_ori", "base_lin_vel", "base_ang_vel",
                            "joint_pos", "joint_vel")}
    bad["ee_pos"] = [1.0, 1.0]  # 2 axes, not 3
    with pytest.raises(ValueError, match="drifted from window_blocks"):
        reference_window_scale(2, N_JOINTS, len(EE_FRAMES), bad)
    with pytest.raises(ValueError, match="drifted from window_blocks"):
        reference_window_noise_vector(2, N_JOINTS, len(EE_FRAMES), bad)


def test_noise_is_specified_before_scaling(env_cmd):
    """Noise half-widths stay in RAW units -- Isaac Lab noises before it scales.

    Pins the invariant that makes the joint-velocity numbers line up: the reference and
    proprioceptive joint velocities both carry +-0.5 rad/s and both are then multiplied by
    0.05, so scaling changes conditioning and not signal-to-noise.
    """
    _, cmd = env_cmd
    per_step = sum(w for _, w in reference_window_layout(cmd))
    jv = slice(12 + N_JOINTS, 12 + 2 * N_JOINTS)
    noise = reference_window_noise_vector(1, N_JOINTS, len(EE_FRAMES))[:per_step]
    scale = reference_window_scale(1, N_JOINTS, len(EE_FRAMES))[:per_step]
    assert set(noise[jv]) == {0.5}      # rad/s, pre-scale
    assert set(scale[jv]) == {0.05}
    # Post-scale the injected noise matches the proprioceptive term's 0.5 * 0.05.
    assert noise[jv.start] * scale[jv.start] == pytest.approx(0.025)


def test_noise_cfg_from_layout_is_hydra_safe(env_cmd):
    """The config carries plain primitives -- a tensor here is rejected by OmegaConf."""
    _, cmd = env_cmd
    cfg = ReferenceWindowNoiseCfg.from_layout(3, N_JOINTS, len(EE_FRAMES))
    per_step = sum(w for _, w in reference_window_layout(cmd))
    assert isinstance(cfg.n_half, list) and len(cfg.n_half) == 3 * per_step
    assert all(isinstance(v, float) for v in cfg.n_half)
    assert cfg.magnitudes == REFERENCE_WINDOW_NOISE
