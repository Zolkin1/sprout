# Copyright 2026 Zachary Olkin. All rights reserved.

"""``FmTrackingCommand`` against a library with no clips: what it skips, and what it refuses.

The free-form fine-tune passes ``--clip`` a terrain manifest carrying zero clips, and this class
normally builds BOTH references every step and applies one. Two defects blocked that, and both
are pinned here:

* **The command could not be constructed.** ``MotionSequenceCommand._post_init`` builds a
  ``MotionClipManager``, which used to raise ``IndexError`` on a zero-clip library. That half is
  pinned in ``source/motion_gen/tests/test_clip_free_library.py``; what this file adds is that
  the command's own construction gate now ADMITS such a library under
  ``reference_source="fm"`` and refuses it everywhere else, by name.
* **``_build_clip_reference()`` ran unconditionally**, as the first statement of
  ``_update_command`` after the free-form command rebuild. Its first call is ``_advance_step``,
  so ``self._step`` kept marching under ``reference_source="fm"`` -- it is NOT a harmless
  constant -- and everything after it indexed an empty clip table.

Every test here drives the SHIPPED methods on a bare instance (``object.__new__``) carrying a
REAL ``MotionClipManager`` over a REAL clip-free ``.h5``, so the thing under test is the thing
that runs. That is as close to constructing the task as is reachable without a simulator: the
env, the scene and the generator checkpoint are all stubbed, and `_post_init` proper is exercised
only up to its clip-free gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp", exc_type=ImportError)
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
wp.init()

from motion_gen.clips.clip_io import ClipLibraryWriter, load_clip_library  # noqa: E402
from motion_gen.clips.clip_manager import ClipFreeLibraryError, MotionClipManager  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402
    FmTrackingCommand,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (  # noqa: E402
    MotionSequenceCommand,
)

N_ENVS = 4
STEP_DT = 0.02
K = 12                                    # the shipped re-plan / snap cadence

_META: dict = {
    "dt": STEP_DT,
    "num_steps": 400,                     # deliberately LARGE: a marching _step would show
    "clip_length_s": 400 * STEP_DT,
    "pos_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
                         "pelvis_link:ori_w", "pelvis_link:ori_x", "pelvis_link:ori_y",
                         "pelvis_link:ori_z"],
    "vel_output_names": ["pelvis_link:pos_x", "pelvis_link:pos_y", "pelvis_link:pos_z",
                         "pelvis_link:ori_x", "pelvis_link:ori_y", "pelvis_link:ori_z"],
    "contact_frames": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "ref_frame": "pelvis_link",
    "skill_list": ["standing"],
}


#: The frame set of the shipped body-tracking library ``clips/g1_mb_v5_body_mode11_decor.h5``,
#: transcribed from it -- base, two feet, two wrists, then the 13 ``@root`` body links. What
#: makes it worth restating here rather than testing against the minimal layout alone: the
#: free-form task is meant to be generated with ``--like <that library>``, so its ``_post_init``
#: resolves 155 position channels, 137 velocity channels, 29 joints and a 13-frame root-body
#: block -- each of which is a separate index map, and none of which the 7-channel minimal
#: layout exercises at all.
_REAL_FRAMES: tuple[str, ...] = (
    "pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "right_wrist_yaw_link", "left_wrist_yaw_link",
    "left_hip_roll_link@root", "left_knee_link@root", "left_ankle_roll_link@root",
    "right_hip_roll_link@root", "right_knee_link@root", "right_ankle_roll_link@root",
    "torso_link@root", "left_shoulder_roll_link@root", "left_elbow_link@root",
    "left_wrist_yaw_link@root", "right_shoulder_roll_link@root", "right_elbow_link@root",
    "right_wrist_yaw_link@root",
)
_REAL_JOINTS: tuple[str, ...] = tuple(f"j{i:02d}" for i in range(29))


def _real_layout_meta() -> dict:
    """The channel layout ``--like`` copies off the shipped training library.

    Returns:
        A metadata dict with the real frame set, 29 joints and the real widths.
    """
    pos = [f"{f}:{c}" for f in _REAL_FRAMES
           for c in ("pos_x", "pos_y", "pos_z", "ori_w", "ori_x", "ori_y", "ori_z")]
    vel = [f"{f}:{c}" for f in _REAL_FRAMES
           for c in ("pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z")]
    pos += [f"joint:{j}" for j in _REAL_JOINTS]
    vel += [f"joint:{j}" for j in _REAL_JOINTS]
    return dict(_META, pos_output_names=pos, vel_output_names=vel)


@pytest.fixture(scope="module")
def clipless_manager(tmp_path_factory) -> MotionClipManager:
    """A real :class:`MotionClipManager` over a real zero-clip library on disk.

    Written through the shared :class:`ClipLibraryWriter`, which is what
    ``scripts/rsl_rl/make_freeform_terrain.py`` uses -- so this is the same file shape a run
    would be given, not a hand-built stand-in.
    """
    path = str(tmp_path_factory.mktemp("clipless") / "terrain_only.h5")
    with ClipLibraryWriter(path, num_clips=0, meta=_META) as w:
        w.set_attr("terrain_manifest", [{"terrain_id": 0, "overlays": []}])
    return MotionClipManager(load_clip_library(path, "cpu"))


class _BareCommand(FmTrackingCommand):
    """``FmTrackingCommand`` with ``device`` / ``num_envs`` as plain class attributes.

    The base exposes both as read-only properties off the env; shadowing them lets a bare
    instance exist without one. Same idiom as ``test_freeform_steering``; every method under
    test is inherited unchanged.
    """

    device = torch.device("cpu")
    num_envs = N_ENVS
    #: ``CommandTerm.__del__`` touches this; a bare instance never ran the ctor that sets it.
    _debug_vis_handle = None


def _cmd(manager: MotionClipManager, **over) -> FmTrackingCommand:
    """A bare command wired for ``_update_command``, carrying ``manager``.

    Everything the generator half would do is stubbed (there is no checkpoint here); everything
    the CLIP half would do is left real, because whether it runs at all is the thing under test.

    Args:
        manager: The clip manager to attach -- real, clip-free or not.
        **over: Config overrides.

    Returns:
        The bare instance.
    """
    cmd = object.__new__(_BareCommand)
    cfg = dict(
        path="terrain_only.h5", reference_source="fm", window_anchor="plan",
        ego_anchor_every=K, fm_snap_every=K, fm_update_every=K,
        freeform_command=True, freeform_speed_min=1.0, freeform_speed_max=1.0,
        freeform_cmd_hold_s=0.0, freeform_steer_kp=1.0, freeform_lookahead_m=5.0,
        freeform_wz_max=0.68,
    )
    cfg.update(over)
    cmd.cfg = SimpleNamespace(**cfg)
    cmd.manager = manager

    # --- the free-form command's buffers ---
    cmd._ff_cmd = torch.zeros(N_ENVS, 3)
    cmd._ff_speed = torch.ones(N_ENVS)
    cmd._ff_origin_w = torch.zeros(N_ENVS, 2)
    cmd._ff_heading_w = torch.zeros(N_ENVS)
    cmd._ff_hold_steps = 0

    # --- the clip-side state MotionSequenceCommand._post_init allocates ---
    cmd.clip_id = torch.zeros(N_ENVS, dtype=torch.long)
    cmd.start_step = torch.zeros(N_ENVS, dtype=torch.long)
    cmd._step = torch.zeros(N_ENVS, dtype=torch.long)
    cmd._ego_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    cmd._snap_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    cmd._reanchor_active = torch.zeros(N_ENVS, dtype=torch.bool)
    cmd._plan_anchor_pos = torch.zeros(N_ENVS, 3)
    cmd._plan_anchor_quat = torch.zeros(N_ENVS, 4)
    cmd._plan_anchor_quat[:, 3] = 1.0
    cmd._resampling = False
    cmd._last_reanchor_step = -1
    cmd._serve_override = None

    # --- the robot and the env ---
    root_quat = torch.zeros(N_ENVS, 4)
    root_quat[:, 3] = 1.0                                        # identity, xyzw
    cmd.robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=wp.from_torch(torch.zeros(N_ENVS, 3).contiguous()),
        root_quat_w=wp.from_torch(root_quat.contiguous()),
    ))
    cmd.env = SimpleNamespace(
        step_dt=STEP_DT, scene=None, common_step_counter=0,
        episode_length_buf=torch.zeros(N_ENVS, dtype=torch.long),
    )

    # --- the generator half and the CLF, stubbed: no checkpoint exists here ---
    cmd.fm_enabled = True
    cmd._push_extero_lag = lambda: None
    cmd._build_fm_reference = lambda: None
    cmd._apply_source = lambda source: None
    cmd.compute_measured_output = lambda pos, quat: None
    cmd.clf = SimpleNamespace(compute_vdot=lambda *a: (0.0, 0.0))
    cmd._zeros3 = cmd._ident_quat = None
    cmd.y_act = cmd.y_des = cmd.dy_act = cmd.dy_des = None
    return cmd


def _tick(cmd: FmTrackingCommand, steps: int, speed: float = 0.0) -> None:
    """Drive ``_update_command`` for ``steps`` control steps, walking the robot at ``speed``.

    Advances exactly what the env advances between compute calls: the global step counter (which
    the re-anchor's idempotency guard keys on), the per-env episode counter (which the cadences
    key on) and the root pose.

    Args:
        cmd: The bare command.
        steps: How many steps to run.
        speed: Metres per step to translate the root by, along +x.
    """
    for _ in range(steps):
        FmTrackingCommand._update_command(cmd)
        cmd.env.common_step_counter += 1
        cmd.env.episode_length_buf = cmd.env.episode_length_buf + 1
        pos = wp.to_torch(cmd.robot.data.root_pos_w).clone()
        pos[:, 0] += speed
        cmd.robot.data.root_pos_w = wp.from_torch(pos.contiguous())


# ---------------------------------------------------------------------------------------------
# The construction gate
# ---------------------------------------------------------------------------------------------
def test_the_clip_tracker_refuses_a_clip_free_library(clipless_manager):
    """``MotionSequenceCommand`` tracks clips, so a library with none is a hard error.

    The refusal is in ``_post_init`` and BEFORE any allocation, so what a reader gets is the
    library path and the one task that is allowed to do this -- not a gather refusal from
    inside the first step, and not the ``IndexError`` from inside the manager that came before.
    """
    class _Bare(MotionSequenceCommand):
        device = torch.device("cpu")
        num_envs = N_ENVS
        _debug_vis_handle = None

    cmd = object.__new__(_Bare)
    cmd.manager = clipless_manager
    cmd.cfg = SimpleNamespace(path="terrain_only.h5")
    assert cmd.has_clips is False
    assert cmd.accepts_clip_free_library is False
    with pytest.raises(ValueError) as e:
        MotionSequenceCommand._post_init(cmd)
    assert "CLIP-FREE" in str(e.value)
    assert "terrain_only.h5" in str(e.value)
    assert "finetune-freeform" in str(e.value)


@pytest.mark.parametrize("source,allowed", [("fm", True), ("clip", False)])
def test_only_the_generator_driven_configuration_admits_one(clipless_manager, source, allowed):
    """The permission is per CONFIGURATION, not per class.

    Under ``reference_source="clip"`` this term still serves the clip, so a library with none
    is exactly as useless here as on the base class -- that is the pure-clip control run, and
    it must not silently start serving zeros.
    """
    cmd = _cmd(clipless_manager, reference_source=source)
    assert cmd.accepts_clip_free_library is allowed


def test_the_clip_free_gate_lets_the_generator_configuration_past_post_init(clipless_manager):
    """The positive arm: the gate is what would have stopped the free-form task here.

    ``_post_init`` cannot run to completion on a bare instance (it needs a scene), so what is
    asserted is that whatever it fails on next, it is NOT the clip-free refusal.
    """
    cmd = _cmd(clipless_manager)
    with pytest.raises(Exception) as e:                # something later: no env/scene here
        MotionSequenceCommand._post_init(cmd)
    assert "CLIP-FREE" not in str(e.value)


def test_the_registered_freeform_task_is_the_configuration_that_is_admitted():
    """The cfg the task registers must be on the permitted side of that gate.

    Otherwise the whole arrangement is correct and the shipped task still refuses its own
    library at construction.
    """
    pytest.importorskip("isaaclab")
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (
        G1FinetuneFreeformEnvCfg,
    )
    cfg = G1FinetuneFreeformEnvCfg()
    assert cfg.commands.traj_ref.reference_source == "fm"
    # ...and every clip term it would otherwise reach is off. Each of these indexes a clip.
    assert cfg.events.reset_on_clip is None
    assert cfg.terminations.clip_end is None
    assert cfg.commands.traj_ref.seed_history_from_clip == "never"
    assert cfg.commands.traj_ref.random_start_phase is False


@pytest.mark.parametrize("layout", ["minimal", "real"])
def test_post_init_runs_to_completion_on_a_clip_free_library(tmp_path, layout):
    """The deepest layer reachable without a simulator: the whole of ``_post_init``.

    Not just the clip-free gate but every allocation, index map and cache after it -- the dt
    check against ``env.step_dt``, the base-frame channel indices, the contact-column map, the
    fingerprint record, the debug-vis frame maps and the world-frame maps. Each of those is a
    place a library with no clips could still have failed AFTER the manager was fixed, and each
    reads the library's METADATA, which a clip-free file carries in full.

    Only ``env``, ``scene`` and the articulation are stubbed; ``BaseTrajectoryCommand.__init__``
    is what normally derives the four attributes seeded below, and it needs a real articulation.
    So this is the last layer before the simulator, and the layers above it -- the env, the
    managers, the generator checkpoint -- remain unproven here by construction.
    """
    meta = _META if layout == "minimal" else _real_layout_meta()
    joints = [n.split("joint:", 1)[1] for n in meta["pos_output_names"]
              if n.startswith("joint:")]
    frames = list(dict.fromkeys(n.rsplit(":", 1)[0] for n in meta["pos_output_names"]
                                if not n.startswith("joint:")))
    path = str(tmp_path / f"{layout}.h5")
    with ClipLibraryWriter(path, num_clips=0, meta=meta) as w:
        w.set_attr("terrain_manifest", [])
    lib = load_clip_library(path, "cpu")

    class _Bare(MotionSequenceCommand):
        device = torch.device("cpu")
        num_envs = N_ENVS
        _debug_vis_handle = None
        accepts_clip_free_library = True         # stands in for reference_source="fm"

    cmd = object.__new__(_Bare)
    cmd.manager = MotionClipManager(lib)
    cmd._clip_lib = lib
    cmd.cfg = SimpleNamespace(
        path=path, base_frame_name="pelvis_link", reference_source="fm",
        debug_vis_show_hands=False, debug_vis_horizon_s=1.0, debug_vis_stride=5,
        clip_fingerprint=None, strict_fingerprint=False,
    )
    cmd.env = SimpleNamespace(step_dt=STEP_DT)
    cmd.robot = SimpleNamespace(joint_names=joints, body_names=frames)
    # Derived by ``BaseTrajectoryCommand.__init__`` from the library's own channel names.
    cmd.ordered_pos_output_names = list(meta["pos_output_names"])
    cmd.ordered_vel_output_names = list(meta["vel_output_names"])
    cmd.contact_bodies = list(meta["contact_frames"])
    cmd.root_body_frames = [f for f in frames if f.endswith("@root")]
    cmd.y_des = torch.zeros(N_ENVS, len(meta["pos_output_names"]))
    cmd.dy_des = torch.zeros(N_ENVS, len(meta["vel_output_names"]))

    MotionSequenceCommand._post_init(cmd)

    assert torch.equal(cmd._step, torch.zeros(N_ENVS, dtype=torch.long))
    assert torch.equal(cmd.clip_id, torch.zeros(N_ENVS, dtype=torch.long))
    assert cmd._contact_col.tolist() == [0, 1]
    if layout == "minimal":
        assert cmd._wf_frames == ["pelvis_link"]
        assert cmd._wf_world_frames.shape == (N_ENVS, 1, 13)
    else:
        # Base + the four world ee frames; the 13 ``@root`` links are lifted separately.
        assert len(cmd._wf_frames) == 5
        assert cmd._wf_world_frames.shape == (N_ENVS, 5, 13)
        assert len(cmd.root_body_frames) == 13
        assert cmd._base_idx["joint_pos"].numel() == 29
    # The dt check is real and reads the library: a mismatch must still be caught here.
    cmd.env = SimpleNamespace(step_dt=STEP_DT * 2)
    with pytest.raises(ValueError, match="Clip dt"):
        MotionSequenceCommand._post_init(cmd)


@pytest.mark.parametrize("field,value", [
    ("history_from", "clip"),          # teacher forcing: the one non-default value
    ("seed_history_from_clip", "on_terrain_start"),
    ("random_start_phase", True),
    ("random_start_before_terrain_start", True),
])
def test_a_clip_phase_knob_left_on_is_refused_at_construction(clipless_manager, field, value):
    """Each of these is a claim about a motion that is not in the file.

    Three are read only at a RESET, so without this they would fail after a simulator launch
    and a first episode; ``random_start_phase`` is read only by the clip reset event this task
    disables, so it would never fail at all and would sit in ``params/env.yaml`` describing a
    start that never happened.
    """
    cmd = _cmd(clipless_manager, **{field: value})
    with pytest.raises(ValueError) as e:
        FmTrackingCommand._refuse_clip_phase_knobs(cmd)
    assert field in str(e.value)
    assert "CLIP-FREE" in str(e.value)


def test_the_shipped_freeform_config_passes_that_check():
    """The positive arm -- otherwise the check is free to be arbitrarily strict."""
    pytest.importorskip("isaaclab")
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (
        G1FinetuneFreeformEnvCfg,
    )
    cmd = SimpleNamespace(cfg=G1FinetuneFreeformEnvCfg().commands.traj_ref,
                          _CLIP_PHASE_KNOBS=FmTrackingCommand._CLIP_PHASE_KNOBS)
    FmTrackingCommand._refuse_clip_phase_knobs(cmd)          # must not raise


# ---------------------------------------------------------------------------------------------
# The per-step gate
# ---------------------------------------------------------------------------------------------
def test_the_clip_half_is_skipped_entirely(clipless_manager):
    """``_build_clip_reference`` must not run, and ``_advance_without_a_clip`` must."""
    cmd = _cmd(clipless_manager)
    calls: list[str] = []
    cmd._build_clip_reference = lambda: calls.append("clip")
    cmd._advance_without_a_clip = lambda: calls.append("clipless")
    _tick(cmd, 5)
    assert calls == ["clipless"] * 5


def test_the_clip_step_does_not_march(clipless_manager):
    """The measured hazard: ``_advance_step`` runs first inside ``_build_clip_reference``.

    It is driven by ``episode_length_buf``, so under the old unconditional call ``self._step``
    walked up to ``num_steps - 1`` on a library where no step indexes anything. That is a live
    number -- ``termination_reference_world_state`` and the history seeding both read it -- so
    "harmless constant" was never available; it either marches or it does not exist.
    """
    cmd = _cmd(clipless_manager)
    _tick(cmd, 60, speed=0.02)
    assert torch.equal(cmd._step, torch.zeros(N_ENVS, dtype=torch.long))
    assert cmd.manager.num_steps == 400          # ...and there was room to march


def test_running_the_clip_half_anyway_marches_the_step_and_then_refuses(clipless_manager):
    """What the gate prevents, measured rather than argued -- both halves of it.

    Calling :meth:`FmTrackingCommand._build_clip_reference` directly is what
    ``_update_command`` used to do unconditionally. Driven at episode step 37 it (a) marches
    ``self._step`` to 37 on a library where no step means anything, and only THEN (b) refuses,
    from the placement refresh. The order is the point: the state corruption happens on the
    first statement and the error on the third, so a caught-and-logged failure would leave a
    marched step behind.
    """
    cmd = _cmd(clipless_manager)
    cmd.env.episode_length_buf = torch.full((N_ENVS,), 37, dtype=torch.long)
    with pytest.raises(ClipFreeLibraryError):
        FmTrackingCommand._build_clip_reference(cmd)
    assert torch.equal(cmd._step, torch.full((N_ENVS,), 37, dtype=torch.long))


def test_the_plan_anchor_still_latches_onto_the_walking_robot(clipless_manager):
    """The ONE thing ``_advance_without_a_clip`` keeps, and why it is not optional.

    ``_reanchor_ego_envs`` is the only writer of ``_plan_anchor_*`` on a running episode (the
    other, ``seed_reanchor_snapshot``, is called by the clip reset, which the free-form task
    disables). Drop it with the rest of the clip half and the anchor sits at the world origin
    for the whole run, which is finite, plausible and wrong.
    """
    cmd = _cmd(clipless_manager)
    # Episode steps 0..K, robot walking +x. Latches at 0 (the spawn, x = 0) and at K.
    _tick(cmd, K + 1, speed=0.1)
    assert float(cmd._plan_anchor_pos[0, 0]) == pytest.approx(K * 0.1)
    _tick(cmd, K - 1, speed=0.1)                 # ...and HELD between re-plans
    assert float(cmd._plan_anchor_pos[0, 0]) == pytest.approx(K * 0.1)


def test_the_free_form_command_is_still_rebuilt_first(clipless_manager):
    """Order is unchanged by the gate: the command, then whichever reference half runs."""
    cmd = _cmd(clipless_manager)
    order: list[str] = []
    cmd._update_freeform_command = lambda: order.append("freeform")
    cmd._advance_without_a_clip = lambda: order.append("clipless")
    cmd._build_fm_reference = lambda: order.append("fm")
    _tick(cmd, 1)
    assert order == ["freeform", "clipless", "fm"]


def test_a_library_with_clips_still_builds_the_clip_reference(tmp_path):
    """The control arm. The gate keys on the LIBRARY, not on ``freeform_command``.

    A clip task that quietly stopped building its clip reference would be a far worse bug than
    the one being fixed, and nothing else in this file would notice.
    """
    path = str(tmp_path / "with_clips.h5")
    n, t = 2, _META["num_steps"]
    p, v, c = (len(_META["pos_output_names"]), len(_META["vel_output_names"]),
               len(_META["contact_frames"]))
    with ClipLibraryWriter(path, num_clips=n, meta=_META) as w:
        w.write_batch(0, {
            "y_des": torch.zeros(n, t, p), "dy_des": torch.zeros(n, t, v),
            "contact": torch.zeros(n, t, c, dtype=torch.uint8),
            "skill_id": torch.zeros(n, t, dtype=torch.uint8),
            "stride_idx": torch.zeros(n, t, dtype=torch.int32),
            "target_vel": torch.zeros(n, t, 3),
        })
    cmd = _cmd(MotionClipManager(load_clip_library(path, "cpu")))
    calls: list[str] = []
    cmd._build_clip_reference = lambda: calls.append("clip")
    cmd._advance_without_a_clip = lambda: calls.append("clipless")
    _tick(cmd, 3)
    assert calls == ["clip"] * 3


# ---------------------------------------------------------------------------------------------
# Serving the other source
# ---------------------------------------------------------------------------------------------
def test_serving_the_clip_reference_is_refused(clipless_manager):
    """``as_clip_reference()`` would install caches that were allocated and never written.

    Zero-filled ``y_des`` is a reference the diagnostic would draw as a robot standing at the
    world origin -- finite, plotted, and meaningless. The A/B needs a real clip library, and on
    free-form terrain there is no clip to compare against by construction.
    """
    cmd = _cmd(clipless_manager)
    with pytest.raises(RuntimeError) as e:
        with cmd.as_clip_reference():
            pass
    assert "CLIP-FREE" in str(e.value)
    assert "fm_debug" in str(e.value)


def test_serving_the_generator_is_unaffected(clipless_manager):
    """The other arm of the same context manager still works -- it is the only reference there."""
    cmd = _cmd(clipless_manager)
    with cmd.as_fm_reference() as served:
        assert served is cmd
        assert cmd._serving() == "fm"
