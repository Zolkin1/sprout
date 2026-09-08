# Copyright 2026 Zachary Olkin. All rights reserved.

"""The ribbon must draw the contact flag the REWARD gates on, from the same producer.

The whole value of colouring contacts in the debug viz is that you are looking at the thing
being graded.  If the ribbon computed its own flag -- from the ankle frame's speed, say,
rather than the slowest sole point -- it would show a different contact set from the one the
terrain-consistency reward uses, and the viz would quietly stop being evidence.

So ``FmTrackingCommand.plan_foot_contact`` is the single producer, and these are source-level
guards on that. They are source-level because ``_post_init`` needs a live scene: constructing
the term to test its wiring would need Isaac Sim, which is exactly what the rest of this file
tree avoids.
"""

from __future__ import annotations

import inspect
import pathlib


def _cmd_source() -> str:
    """The command term's source, located via the MODULE rather than by counting hops.

    ``..``-counting from a test file has silently broken paths in this repo before, and it is
    what CLAUDE.md asks not to do -- so ask the import system where the module actually is.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
        fm_tracking_cmd)
    return pathlib.Path(inspect.getsourcefile(fm_tracking_cmd)).read_text()


def test_the_ribbon_reads_the_shared_contact_producer():
    """Not its own finite difference, and not the ankle frame's speed."""
    src = _cmd_source()
    draw = src[src.index("def _draw_fm_prediction_ribbon"):]
    draw = draw[:draw.index("\n    def ")]
    assert "self.plan_foot_contact()" in draw, (
        "the ribbon must call plan_foot_contact, so the flag drawn is the flag graded")
    for forbidden in ("plan_contact(", "node_speed(", "torch.diff"):
        assert forbidden not in draw, (
            f"the ribbon computes {forbidden} itself; that is a second contact definition")


def test_the_producer_derives_its_dt_from_the_env():
    """A restated 0.02 would silently mis-scale every speed after a decimation change."""
    src = _cmd_source()
    fn = src[src.index("def plan_foot_contact"):]
    fn = fn[:fn.index("\n    def ")]
    assert "self._env.step_dt" in fn
    assert "0.02" not in fn


def test_the_producer_reads_the_soles_not_the_ankle_frames():
    """The collider model is feet-only and surface-sampled, which is what has soles at all."""
    src = _cmd_source()
    fn = src[src.index("def plan_collider_model"):]
    fn = fn[:fn.index("\n    def ")]
    assert "body_names=list(FOOT_BODIES)" in fn
    assert "PlanColliderModel" in fn


def test_the_contact_colour_is_distinguishable_from_every_frame_colour():
    """It is read AGAINST the frame colours, so it must not be one of them."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd \
        import DEBUG_CONTACT_COLOR
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking \
        .motion_sequence_cmd import _DEBUG_FRAME_COLOR_MAP, _DEBUG_FRAME_FALLBACK_COLOR

    others = list(_DEBUG_FRAME_COLOR_MAP.values()) + [_DEBUG_FRAME_FALLBACK_COLOR]
    for c in others:
        dist = sum((a - b) ** 2 for a, b in zip(DEBUG_CONTACT_COLOR, c)) ** 0.5
        assert dist > 0.35, f"contact colour is too close to {c} (distance {dist:.3f})"


def test_the_foot_body_names_come_from_one_place():
    """The ribbon indexes ``contact`` by FOOT_BODIES order, so it must import that order."""
    src = _cmd_source()
    assert "from motion_gen.plan_geometry import FOOT_BODIES" in src
    assert 'FOOT_BODIES.index(name)' in src
    # ...and it must not restate the names, which would let the two orders diverge.
    assert '"left_ankle_roll_link"' not in src.split("FOOT_BODIES")[-1]


def test_the_history_null_arm_is_drawn_and_ors_into_the_served_mask():
    """`fm_history_null_fraction` must reach the MODEL and be visible in the ribbon.

    The knob exists to break an autoregressively committed mode: the generator conditions on
    its own past output, so once it reads a box top as a descending staircase it stays
    conditioned on a plan that says so.  Nulling the history makes the mode be re-decided from
    exteroception and the command alone.

    Two things have to hold for the play-script look to mean anything: the deliberate arm must
    be OR'd into the same `history_null_mask` the post-reset transient uses (the model has one
    null embedding and cannot tell why a row is nulled), and the affected envs must be
    distinguishable on screen.
    """
    src = _cmd_source()

    # It rides on the same mask, not a separate model input.
    infer = src[src.index("null_hist = self._history.null_mask()"):]
    infer = infer[:infer.index("self._fm_reinfer_mask")]
    assert "_force_null_hist" in infer and "|" in infer, (
        "the deliberate arm must OR into `history_null_mask`; a second channel would need a "
        "second embedding the model does not have")

    # Re-drawn per EPISODE, so a whole episode is attributable to one arm.
    reset = src[src.index("    def reset(self, env_ids=None):"):]
    reset = reset[:reset.index("\n    # ---")]
    assert "_draw_history_null" in reset

    # Randomly, never strided: envs map to tiles through a structured grid lookup, so every
    # n-th env could land the whole arm on one tile.
    draw = src[src.index("    def _draw_history_null"):]
    draw = draw[:draw.index("\n    def ")]
    assert "torch.rand" in draw
    for strided in ("% ", "arange(0,", "::"):
        assert strided not in draw.replace("torch.arange(self.num_envs", ""), (
            f"the draw looks strided ({strided!r}); that risks putting every history-nulled "
            "env on the same tile")

    # And it is visible: its own root-path prototype, selected per env.
    assert "DEBUG_NULL_HISTORY_COLOR" in src
    assert "_fm_arm_protos" in src
    ribbon = src[src.index("def _draw_fm_prediction_ribbon"):]
    ribbon = ribbon[:ribbon.index("\n    def ")]
    assert "_fm_arm_protos" in ribbon, "the arm is invisible in the ribbon"


def test_history_null_is_off_by_default():
    """A default above 0 would silently change every existing run's serve path."""
    import dataclasses

    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
        fm_tracking_cmd_cfg)

    # Read the dataclass FIELD, not a class attribute: `@configclass` leaves no class
    # attribute and rewrites a scalar default into a `default_factory`, so both forms are
    # accepted rather than assuming which one this decorator version produces.
    fields = {f.name: f for f in dataclasses.fields(fm_tracking_cmd_cfg.FmTrackingCommandCfg)}
    assert "fm_history_null_fraction" in fields, sorted(fields)
    f = fields["fm_history_null_fraction"]
    got = (f.default_factory() if f.default is dataclasses.MISSING else f.default)
    assert got == 0.0, got


def test_the_upper_camera_is_resolved_from_mount_height_not_a_hardcoded_index():
    """Which sensor is "upper" must be DERIVED, or the knob can corrupt the wrong camera.

    The extero sensor axis is fixed by the obs group's term order (`fm_depth_near` then
    `fm_depth_far`).  A hardcoded index 1 would keep working right up until someone reorders
    those terms, at which point it would corrupt the NEAR camera while the flag went on
    claiming to test forward terrain preview -- a wrong answer with no error.  So the index
    comes from the checkpoint's own `extero_spec` mount poses.
    """
    src = _cmd_source()
    fn = src[src.index("    def _resolve_upper_camera"):]
    fn = fn[:fn.index("\n    def ")]
    assert "extero_spec" in fn and "offset_pos" in fn, (
        "the upper camera must be resolved from the spec's mount heights")
    # No name matching either: 'far' is a label, the mount height is the physical fact.
    assert '"far"' not in fn and "'far'" not in fn
    # And it refuses rather than guessing when it cannot tell.
    assert "return None" in fn


def test_the_upper_camera_knob_refuses_when_it_cannot_identify_the_camera():
    """A silent fallback to sensor 1 is the failure this must not have."""
    src = _cmd_source()
    ctor = src[src.index("self._upper_cam_index = self._resolve_upper_camera()"):]
    ctor = ctor[:ctor.index("[FmTrackingCommand] upper camera")]
    assert "raise ValueError" in ctor
    assert "Refusing rather" in ctor or "Refusing" in ctor


def test_dead_mode_folds_the_channels_channel_major():
    """`d, x, y, z, v` channel-major; interleaving would scramble every CNN kernel.

    The 'dead' mode has to write the validity channel specifically, so it must fold the row
    the same way the encoder does.
    """
    src = _cmd_source()
    fn = src[src.index("    def _corrupt_upper_camera"):]
    fn = fn[:fn.index("\n    def ")]
    assert "view(sel.numel(), nch, k // nch)" in fn, (
        "the fold must be (channels, pixels) -- channel-major")
    assert "% nch" in fn, "an indivisible row width must fail rather than silently mis-fold"
    # Gaussian corrupts validity too, and that is deliberate, not an oversight.
    assert "validity included" in fn


def test_the_upper_camera_noise_is_off_by_default():
    """A non-zero default would silently corrupt an input on every existing run."""
    import dataclasses

    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
        fm_tracking_cmd_cfg)

    fields = {f.name: f for f in dataclasses.fields(fm_tracking_cmd_cfg.FmTrackingCommandCfg)}
    f = fields["fm_upper_camera_noise_fraction"]
    got = (f.default_factory() if f.default is dataclasses.MISSING else f.default)
    assert got == 0.0, got
    m = fields["fm_upper_camera_noise_mode"]
    got_m = (m.default_factory() if m.default is dataclasses.MISSING else m.default)
    assert got_m == "gaussian", got_m


def test_both_diagnostic_arms_are_visually_distinct_and_composable():
    """Four states must be tellable apart: neither arm, either one, and both.

    The two knobs are independent, so a run with both on has four populations and a viewer
    that only marked one of them would attribute the wrong cause to what it saw.  Index 0 is
    deliberately absent from the colour table: an env on no arm keeps its ordinary frame
    colour, so a scene with every knob off looks exactly as it always did.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd \
        import DEBUG_ARM_COLORS, DEBUG_CONTACT_COLOR, DEBUG_NULL_HISTORY_COLOR

    assert set(DEBUG_ARM_COLORS) == {1, 2, 3}, "arm 0 must have no colour of its own"
    colors = [c for _n, c in DEBUG_ARM_COLORS.values()]
    assert len({tuple(c) for c in colors}) == 3, "two arms share a colour"

    # Distinct from every frame colour already on screen, and from the contact green.
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd \
        import _DEBUG_FRAME_COLOR_MAP, _DEBUG_FRAME_FALLBACK_COLOR

    existing = ([tuple(v) for v in _DEBUG_FRAME_COLOR_MAP.values()]
                + [tuple(_DEBUG_FRAME_FALLBACK_COLOR), tuple(DEBUG_CONTACT_COLOR)])
    for name, c in DEBUG_ARM_COLORS.values():
        assert tuple(c) not in existing, f"{name} reuses a colour already on screen"

    # The history arm keeps the colour the play-script instructions already quote.
    assert tuple(DEBUG_NULL_HISTORY_COLOR) == tuple(DEBUG_ARM_COLORS[1][1])


def test_the_arm_index_composes_the_two_flags():
    """`history_nulled + 2 * upper_camera_noised`, so all four states are reachable."""
    src = _cmd_source()
    ribbon = src[src.index("def _draw_fm_prediction_ribbon"):]
    ribbon = ribbon[:ribbon.index("\n    def ")]
    assert "_force_null_hist.long()" in ribbon
    assert "2 * self._noise_upper_cam.long()" in ribbon, (
        "the two arms must compose into one index, or 'both' is drawn as one of them")


def test_the_replan_cadence_draws_immediately_before_sampling():
    """The flag gating the corruption must be the flag the plan was sampled under.

    Under the per-re-plan cadence the draw has to happen for exactly the re-planning envs and
    right before `_run_fm`.  Drawing it a step early or for all envs would corrupt a different
    set than the one that then samples, and the ribbon -- which colours by the same flag --
    would attribute each plan to the wrong condition.
    """
    src = _cmd_source()
    infer = src[src.index("self._fm_reinfer_mask = infer_mask"):]
    infer = infer[:infer.index("self._run_fm(infer_mask, null_hist)")]
    assert 'fm_upper_camera_noise_cadence) == "replan"' in infer
    assert "infer_mask.nonzero" in infer, (
        "the draw must take the re-planning env IDS; a bool mask would size the draw wrong")


def test_the_two_cadences_are_mutually_exclusive():
    """Exactly one of reset / re-plan draws, or the arm is re-rolled twice a step."""
    src = _cmd_source()
    reset = src[src.index("    def reset(self, env_ids=None):"):]
    reset = reset[:reset.index("return out")]
    assert 'fm_upper_camera_noise_cadence) == "episode"' in reset, (
        "reset must draw only under the per-episode cadence")
    # The history arm is unconditional -- it has no cadence knob yet, and that is deliberate.
    assert "self._draw_history_null(env_ids)" in reset


def test_an_unknown_cadence_is_refused():
    """A typo must not silently fall through to one of the two behaviours."""
    src = _cmd_source()
    assert "fm_upper_camera_noise_cadence must be 'episode' or 'replan'" in src


def test_the_recording_stores_the_TRUE_conditioning_not_the_perturbed_one():
    """The perturbations are a BEHAVIOUR POLICY; the recording must be the measurement.

    The generator is conditioned on a dead camera / null history to widen the distribution of
    states and plans a run visits, but the critic and the AWR fine-tune must learn from what
    actually happened -- so the window table has to store the real camera row and the real
    history.

    **This currently holds by a copy-vs-view accident and that is why it is pinned here.**
    `_run_fm` corrupts ``self._lagged_extero()[ids]``, and advanced indexing with an index
    TENSOR returns a copy, so the underlying buffer is untouched and the recorder's own
    ``_lagged_extero()[idx]`` call sees the truth.  Change ``ids`` to a slice anywhere on that
    path and the corruption silently leaks into the recording, with nothing in the data saying
    so.
    """
    import numpy as np
    import torch

    # The semantics the correctness rests on.
    buf = torch.zeros(4, 2, 3)
    view_of_slice = buf[0:2]
    copy_of_index = buf[torch.tensor([0, 2])]
    copy_of_index[:] = 9.0
    assert buf.abs().sum().item() == 0.0, "index-tensor selection stopped being a copy"
    view_of_slice[:] = 7.0
    assert buf.abs().sum().item() > 0.0, "slice selection stopped being a view"

    src = _cmd_source()
    fn = src[src.index("    def _run_fm"):]
    fn = fn[:fn.index("\n    def ")]
    # The corruption must target the INDEXED copy, never the buffer or a slice of it.
    assert "extero_row = self._lagged_extero()[ids]" in fn
    assert "_corrupt_upper_camera(extero_row, ids)" in fn
    corrupt = src[src.index("    def _corrupt_upper_camera"):]
    corrupt = corrupt[:corrupt.index("\n    def ")]
    assert "self._extero" not in corrupt and "_lagged_extero" not in corrupt, (
        "the corruption must not reach back into the extero buffer")

    # And the history is nulled via the MASK handed to the model, never by editing the stash.
    assert "self._history.stack_padded" not in corrupt
    for editor in ("self._history.push", "self._history._steps"):
        assert editor not in src[src.index("def _corrupt_upper_camera"):
                                 src.index("def _draw_history_null")], editor
    del np


def test_the_perturbation_gate_guarantees_a_control_population():
    """An ineligible env must be pristine however the per-arm draws land.

    Independent draws at 0.5 each would leave only 25% untouched and a different 25% every
    time; the gate fixes a control population of a stated size inside the same run, on the
    same terrain and command distribution, which is what makes the arms comparable.
    """
    src = _cmd_source()
    for fn_name in ("_draw_upper_cam_noise", "_draw_history_null"):
        fn = src[src.index(f"    def {fn_name}"):]
        fn = fn[:fn.index("\n    def ")]
        assert "self._perturb_eligible[ids]" in fn, (
            f"{fn_name} ignores the eligibility gate")
        assert "&" in fn, f"{fn_name} must AND with eligibility, not overwrite it"

    # The gate itself is per EPISODE even when an arm re-draws per re-plan: a control
    # population that changed mid-episode would not be one.
    gate = src[src.index("    def _draw_perturb_eligible"):]
    gate = gate[:gate.index("\n    def ")]
    assert "torch.rand" in gate
    reset = src[src.index("    def reset(self, env_ids=None):"):]
    reset = reset[:reset.index("return out")]
    assert "_draw_perturb_eligible(env_ids)" in reset
