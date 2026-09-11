# Copyright 2026 Zachary Olkin. All rights reserved.

"""Read the failure critic from LIVE Isaac buffers, on the GPU, at a re-plan.

``scripts/flow_matching/fit_value.py`` fits ``V(s) = -P(fail | s)`` over the generator's own
frozen encoder.  Best-of-K needs that number ONLINE -- draw K candidate chunks, roll each one
``fm_update_every`` steps, score the state each produced, commit the best -- and two halves of
that already existed, neither of them usable as it stands.

* ``record_rollouts._emit`` reads exactly the right live buffers at exactly the right instant,
  and then calls ``.detach().to("cpu").numpy()`` on every one of them.  That is right for a
  recorder streaming HDF5 and ruinous here: this runs inside the physics loop for thousands of
  envs, and a device round trip per field would cost more than the forward pass it feeds.
* ``fit_value._batch`` applies the transformation the critic was fitted under, but only to
  arrays already read back off a recording -- numpy indexing, ``np.stack``, ``np.asarray``.

This module is the fusion: ``_emit``'s reads, ``_batch``'s transformation, in torch, on the
device the buffers already live on.  Nothing here touches host memory.

**``_batch`` is the SPECIFICATION, not the inspiration.**  Every way of diverging from it is
shape-silent -- the critic would score in a different input space from the one it was fitted
in while every tensor still had the right width, which is the failure this repo has been bitten
by in four other places.  So ``source/robot_rl/tests/test_headroom_value.py`` asserts
field-by-field equality between this and the real ``fit_value._batch`` fed the numpy equivalent
of the same tensors, and ``GOAL_SCALE_M`` is IMPORTED from ``fit_value`` rather than restated
so the two cannot drift apart at all.

The four subtleties that carry the risk, all of them inherited rather than invented here:

* the normalizers are the GENERATOR's own (``ctx.state_norm`` / ``ctx.extero_norm`` /
  ``ctx.cmd_norm``); re-fitting anything would put critic and generator in different spaces;
* ``cmd_future`` is indexed by ``ctx.cmd_steps`` BEFORE normalizing;
* the height scan carries ``-inf`` for a missed ray and one non-finite entry takes the whole
  batch's logit with it, so it goes through ``transforms.sanitize_extero(..., "height_scan")``
  -- at that source's DEFAULT band, which is what the fit used, never ``ctx.extero_clip``;
* ``priv`` keeps the scan, proprioception and the FULL twist; the onboard variant drops both
  and keeps only ``twist[:, 3:]``, because nothing on the real robot measures the linear half
  without odometry.  Slicing it away is what makes the two variants' gap a statement about
  information rather than about architecture.

**One thing is deliberately NOT reproduced, and it lives in the RECORDING rather than in
``_batch``.**  ``record_rollouts._emit`` stores ``extero`` and ``height_scan`` as float16;
``_batch`` upcasts them to float32, so the critic was FITTED on fp16-quantised versions of
those two fields (~5e-4 relative).  The live buffers are float32 and are used as they are --
the same choice ``transfer/sim/mjeval/critic.py`` makes on the MuJoCo side, and the smaller of
the two available errors, since re-quantising would be a second lossy step rather than an
undo.  It is a transfer question, not a port bug, and it is the only respect in which what
this module hands the critic differs from what training saw.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import torch

import _paths  # noqa: F401  - puts scripts/rsl_rl and this dir on sys.path

if TYPE_CHECKING:  # pragma: no cover - import cost, not behaviour
    from robot_rl.flow_matching import value


def _fit_value() -> Any:
    """Import ``scripts/flow_matching/fit_value.py``, the module this one must agree with.

    Imported rather than mirrored so ``GOAL_SCALE_M`` has ONE statement in the Isaac process.
    ``transfer/sim/mjeval/goal_anchor.py`` restates it because ``env_transfer`` cannot import
    torch or ``robot_rl``; that excuse does not apply here, and a third copy of a declared
    constant is how a "declared, not fitted" scale quietly becomes two scales.

    Lazy because it pulls ``rollout_outcomes`` and ``robot_rl.flow_matching`` -- fine inside a
    live Isaac process, needless cost at import time for anything that only wants the class.

    Returns:
        The ``fit_value`` module.
    """
    d = str(_paths.REPO_ROOT / "scripts" / "flow_matching")
    if d not in sys.path:
        sys.path.insert(0, d)
    import fit_value  # noqa: PLC0415

    return fit_value


class OnlineValueScorer:
    """``P(fail | s)`` for a set of envs, built from live buffers without leaving the device.

    Reads the same fields ``record_rollouts._emit`` records and applies exactly the transform
    ``fit_value._batch`` applies to them.  ``inputs`` is the part under test; ``score`` is that
    plus one forward pass.

    Args:
        model: A fitted ``robot_rl.flow_matching.value.FlowValueModel``.
        ctx: The ``FmOnlineContext`` whose normalizers the critic was fitted under.  In an
            Isaac rollout this is ``cmd.ctx``, and passing that one is strictly better than
            loading a second copy: the history nodes and the exteroception row are produced by
            THAT generator, so normalizing them with another checkpoint's statistics is a
            silent input-space shift (the shapes agree across checkpoints).
        priv: Whether this is the ``V_priv`` variant.  Decides both the scan/proprio tokens
            and the twist slice, so it must match the checkpoint's own ``privileged`` flag.
        scan_hw: ``(H, W)`` fold of the privileged height scan.  Required when ``priv``,
            refused otherwise.  A transposed fold is scrambled input to a convolution while
            the element count still agrees, so it is checked against the model's own encoder.
        device: Device to build the batch on.  Must be the device the env's buffers live on;
            anything else reintroduces the copy this module exists to avoid.
        scan_sensor: Name of the privileged scanner, passed to ``mdp.scan_obs_stack``.
            Required when ``priv``, refused otherwise.
        worst_penalty_context: Whether the worst-terrain-penalty-so-far block is in the
            context token.  From the checkpoint's own stamp, for the same reason as
            ``goal_context``: such a fit is one block WIDER, so a check omitting it rejects a
            good critic.  This path cannot SERVE one -- see the refusal below.
        goal_context: Whether the three goal offsets go into the context token.  Comes from
            the checkpoint's own ``goal_context`` stamp; it cannot be inferred from the width,
            since 6 is both ``V_onboard`` with the goal and ``V_priv`` without it.

    Raises:
        ValueError: When ``priv`` disagrees with the model's own tokens, when the declared
            ``scan_hw`` disagrees with the scan encoder's, when the context token width
            disagrees with the twist slice ``priv`` selects, or when a privileged scorer is
            built with no scanner to read.
    """

    def __init__(self, model, ctx, *, priv: bool, scan_hw, device,
                 scan_sensor: str | None, goal_context: bool = True,
                 worst_penalty_context: bool = False) -> None:
        self.model = model
        self.ctx = ctx
        self.priv = bool(priv)
        self.goal_context = bool(goal_context)
        self.worst_penalty_context = bool(worst_penalty_context)
        self.device = torch.device(device)
        self.scan_sensor = scan_sensor
        self.scan_hw = None if scan_hw is None else (int(scan_hw[0]), int(scan_hw[1]))

        has_scan = getattr(model, "scan_enc", None) is not None
        has_proprio = getattr(model, "proprio_in", None) is not None
        if self.priv != has_scan or self.priv != has_proprio:
            raise ValueError(
                f"priv={self.priv} but the model carries scan tokens={has_scan}, proprio "
                f"tokens={has_proprio}. The flag decides the TWIST SLICE as well as the "
                "tokens, so a scorer that disagrees with its checkpoint feeds a context "
                "vector of the wrong half of the twist -- and on the onboard variant that is "
                "3 numbers wide either way for the goal, which is why it is checked here "
                "rather than left to a shape error deeper in.")
        if self.priv:
            if self.scan_hw is None:
                raise ValueError("a privileged scorer needs scan_hw; 176 rays is both 11x16 "
                                 "and 16x11 and a transposed fold is scrambled input to the "
                                 "scan convolution while the ray count still agrees.")
            if self.scan_sensor is None:
                raise ValueError("a privileged scorer needs scan_sensor -- the name "
                                 "mdp.scan_obs_stack reads. Without one there is no scan to "
                                 "fold, and the model would refuse the batch at forward.")
            enc_hw = tuple(int(v) for v in model.scan_enc.image_hw)
            if self.scan_hw != enc_hw:
                raise ValueError(
                    f"scan_hw={self.scan_hw} but the checkpoint's scan encoder was built for "
                    f"{enc_hw}. Folding one into the other keeps the element count and "
                    "scrambles every kernel.")
        else:
            if self.scan_hw is not None or self.scan_sensor is not None:
                raise ValueError(
                    f"priv=False, so neither scan_hw ({self.scan_hw}) nor scan_sensor "
                    f"({self.scan_sensor}) is read. Passing one is a statement that the scan "
                    "reaches the critic, and it does not -- refused rather than ignored.")

        from robot_rl.flow_matching import value as _fmvalue  # noqa: PLC0415

        if self.worst_penalty_context:
            # Nothing in the best-of-K selector computes a terrain penalty, so there is no
            # running worst to put in the context token. Refused at load rather than at the
            # builder, so the message names the missing plumbing.
            raise SystemExit(
                "this critic was fitted with --worst_context (its context token carries the "
                "worst terrain penalty BEFORE each chunk) and the selector has no terrain "
                "grader to supply it. Fit without --worst_context to use it here.")
        want_ctx = _fmvalue.context_width(privileged=self.priv,
                                          goal_context=self.goal_context,
                                          worst_penalty_context=self.worst_penalty_context)
        ctx_in = getattr(model, "context_in", None)
        if ctx_in is not None and int(ctx_in[0].in_features) != want_ctx:
            raise ValueError(
                f"the checkpoint's context token is {int(ctx_in[0].in_features)} wide but "
                f"priv={self.priv} / goal_context={self.goal_context} / "
                f"worst_penalty_context={self.worst_penalty_context} builds {want_ctx} "
                f"({'3 goal offsets + ' if self.goal_context else 'NO goal + '}"
                f"{'6' if self.priv else '3'} twist). The arms differ ONLY here and in the "
                "tokens, so this is where a mislabelled checkpoint shows up -- and note a "
                "width of 6 is reachable two ways, which is why `goal_context` travels as a "
                "stamp rather than being read off this number.")

        # Precomputed once: `_batch` builds this tensor per call, and the strided indexing is
        # the step most easily done in the wrong order (index the batch, then the steps).
        self._cmd_steps = (torch.as_tensor(ctx.cmd_steps, dtype=torch.long, device=self.device)
                           if ctx.cmd_dim > 0 else None)
        self._goal_scale_m = float(_fit_value().GOAL_SCALE_M)

    # ------------------------------------------------------------------
    # Live reads
    # ------------------------------------------------------------------
    def _read_scan(self, env) -> torch.Tensor:
        """The privileged height scan for every env, ``(num_envs, num_rays)``.

        The one seam in this module that needs a built scene, which is why it is a method:
        ``source/robot_rl/tests/test_headroom_value.py`` overrides it to compare the whole
        rest of the assembly against ``fit_value._batch`` with no simulator.  Everything else
        reads plain tensors off objects a stub can supply.

        ``mdp.scan_obs_stack`` is the single reader for this signal -- no observation noise
        and no clip on either side -- which is what makes a live rollout and the recording the
        critic was fitted on comparable at all.  ``[:, 0]`` drops the sensor axis exactly as
        ``record_rollouts._emit`` does.

        Args:
            env: The unwrapped env.

        Returns:
            ``(num_envs, num_rays)`` on the env's device.
        """
        from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: PLC0415

        return mdp.scan_obs_stack(env, self.scan_sensor)[:, 0]

    def _twist_yaw(self, rd, ids: torch.Tensor) -> torch.Tensor:
        """The measured base twist in the robot's YAW frame, ``(V, 6)``, linear then angular.

        The torch twin of ``record_rollouts._twist_yaw``, differing only in taking the env
        subset first -- every operation in it is elementwise over envs, so the result is
        bit-identical to slicing afterwards and the work scales with ``V`` instead of with
        ``num_envs``.

        ``yaw_quat(root_quat_w)`` is the LEVEL frame the generator predicts its window in --
        z is gravity, roll and pitch are zero -- so this is directly comparable with the
        plan's own ``base_vel`` block without a transform.  The body frame would tilt with the
        torso and swing the forward component on every pitch; world would defeat heading
        invariance.

        Args:
            rd: The accumulator's converting view of ``ArticulationData``.
            ids: ``(V,)`` env indices.

        Returns:
            ``(V, 6)``.
        """
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: PLC0415

        q = yaw_quat(rd.root_quat_w[ids])
        return torch.cat([quat_apply_inverse(q, rd.root_lin_vel_w[ids]),
                          quat_apply_inverse(q, rd.root_ang_vel_w[ids])], dim=-1)

    def _goal_offsets(self, tracker, ids: torch.Tensor) -> torch.Tensor:
        """``(V, 3)`` downrange / lateral / height error, in METRES.

        Read through ``goal_region.goal_frame_offsets`` -- the same function the outcome fold
        and ``_emit`` use -- so a per-window score and a terminal row cannot disagree about
        which way a tile faces.  Subset first for the same reason as :meth:`_twist_yaw`.

        Args:
            tracker: The ``rollout_outcomes.RolloutEpisodeTracker``.
            ids: ``(V,)`` env indices.

        Returns:
            ``(V, 3)`` metres, unscaled.
        """
        from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region  # noqa: PLC0415

        acc = tracker.accum
        down, lat, hz = goal_region.goal_frame_offsets(
            acc.rd.root_pos_w[ids], acc._goal_pos_w[ids], acc._goal_yaw[ids])
        return torch.stack([down, lat, hz], dim=-1)

    # ------------------------------------------------------------------
    # The batch
    # ------------------------------------------------------------------
    def inputs(self, cmd, tracker, ring, env, env_ids) -> "value.ValueInputs":
        """Assemble one normalized batch for ``env_ids``, entirely on ``self.device``.

        Must be called at the instant ``record_rollouts._emit`` is: after ``env.step`` has run
        ``_update_command``, so the history ring, the served exteroception, the command
        horizon and the robot state all describe the same control step.  A step later the
        generator has re-planned and ``cmd._history`` has rolled, so every field would
        describe a different state while none of them changed shape.

        ``env_ids`` is whatever the caller wants scored -- it is NOT required to be
        ``cmd._fm_reinfer_mask``.  All of these buffers are maintained for every env on every
        control step, so any subset is readable; what the caller owes is that the STATE it
        names is the one it means (see the module docstring).

        Args:
            cmd: The ``FmTrackingCommand`` driving the reference.
            tracker: The ``rollout_outcomes.RolloutEpisodeTracker`` (for the robot state and
                the goal anchor).
            ring: The ``record_rollouts.ProprioRing``; ``None`` is accepted when not ``priv``.
            env: The unwrapped env, for the privileged scan; ``None`` when not ``priv``.
            env_ids: ``(V,)`` long tensor of env indices.

        Returns:
            A ``ValueInputs`` whose every field is on ``self.device``.

        Raises:
            ValueError: On a boolean or non-1-D ``env_ids``, or on a privileged call with no
                ring or no env.
        """
        from robot_rl.flow_matching import transforms, value  # noqa: PLC0415

        ids = self._env_ids(env_ids)
        n = int(ids.numel())
        ctx = self.ctx

        hist = ctx.state_norm.normalize(self._f32(cmd._history.stack_padded("node")[ids]))
        # `_lagged_extero` is what `record_rollouts._emit` records, and there are two
        # shape-identical wrong neighbours: `cmd._extero_stash` is the row the generator
        # conditioned on at its LAST re-plan (right for attributing that plan, wrong for
        # scoring the state now), and a fresh `_read_extero()` skips the camera lag the
        # generator was fitted under. Flattened exactly as `_batch` flattens the recorded
        # column -- row-major over (sensor, pixel); a different fold scrambles every
        # convolution kernel while the width still agrees.
        ext = ctx.extero_norm.normalize(self._f32(cmd._lagged_extero()[ids]).reshape(n, -1))

        cmd_tokens = None
        if ctx.cmd_dim > 0:
            fut = self._f32(cmd.get_command_velocity_horizon(ctx.F)[ids])
            cmd_tokens = ctx.cmd_norm.normalize(fut[:, self._cmd_steps, :])

        scan = proprio = None
        if self.priv:
            if ring is None or env is None:
                raise ValueError(
                    f"priv={self.priv} needs both a proprioception ring ({ring is not None}) "
                    f"and an env to read the scan from ({env is not None}). Proprioception is "
                    "half of what the privileged variant adds, so a None here would make "
                    "V_priv quietly worse than V_onboard rather than better.")
            # The scan records -inf for a MISSED RAY and one non-finite value takes the whole
            # batch's logit with it. `sanitize_extero` is the generator's own guard, and the
            # band is that source's DEFAULT -- not `ctx.extero_clip` -- because the default is
            # what `fit_value._batch` clamped the training rows with.
            scan = transforms.sanitize_extero(
                self._f32(self._read_scan(env)[ids]), "height_scan").reshape(n, *self.scan_hw)
            proprio = self._f32(ring.buf[ids])

        # Goal offsets on a DECLARED metre scale rather than a fitted statistic, so a critic's
        # notion of "far" is a property of the robot and not of whichever rollout was on disk.
        goal = None
        if self.goal_context:
            goal = self._goal_offsets(tracker, ids).to(torch.float32) / self._goal_scale_m
            goal = torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0)
        twist = self._f32(self._twist_yaw(tracker.accum.rd, ids))
        # ONBOARD keeps only the ANGULAR half: an IMU measures body rates and the yaw frame is
        # reachable from those plus its own roll/pitch, but nothing on the robot measures the
        # linear half without odometry.
        # Assembled by `value.build_context`, which owns the order -- the same call the fit,
        # fit_q and the mjeval reader make, so the layout cannot differ between the process
        # that trains a critic and the one that reads it.
        from robot_rl.flow_matching import value as _fmvalue  # noqa: PLC0415

        pieces = {"twist": twist}
        if goal is not None:
            pieces["goal"] = goal
        context = _fmvalue.build_context(privileged=self.priv,
                                         goal_context=goal is not None, **pieces)

        return value.ValueInputs(
            history=hist, extero=ext, cmd_tokens=cmd_tokens,
            history_null=cmd._history.null_mask()[ids].to(device=self.device,
                                                          dtype=torch.bool),
            scan=scan, proprio=proprio, context=context)

    def score(self, cmd, tracker, ring, env, env_ids) -> torch.Tensor:
        """``P(fail | s)`` for ``env_ids``, ``(V,)`` on ``self.device``.

        LOWER IS BETTER: the fitted quantity is a failure probability and the value is
        ``V(s) = -P(fail | s)``, so a best-of-K selector takes the ``argmin`` of this.

        Eval mode is forced and restored around the pass, exactly as ``fit_value._predict``
        does.  The borrowed encoder has dropout, so a model left in training mode returns a
        different number on every call with nothing to say so -- and on a selector that reads
        as the generator's sampling variance rather than as a bug.

        Args:
            cmd: The ``FmTrackingCommand``.
            tracker: The ``rollout_outcomes.RolloutEpisodeTracker``.
            ring: The proprioception ring, or ``None`` when not ``priv``.
            env: The unwrapped env, or ``None`` when not ``priv``.
            env_ids: ``(V,)`` long tensor of env indices.

        Returns:
            ``(V,)`` failure probabilities in ``[0, 1]``.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self.model.p_fail(self.inputs(cmd, tracker, ring, env, env_ids))
        finally:
            self.model.train(was_training)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _f32(self, t: torch.Tensor) -> torch.Tensor:
        """Float32 on ``self.device``, and a no-op when it already is.

        ``fit_value._batch`` casts every field to float32 on its device; matching that is what
        makes the two comparable.  ``Tensor.to`` returns ``self`` when nothing changes, so on
        the hot path -- sim buffers already float32 on the sim device -- this copies nothing,
        which is the whole reason this module exists.

        Args:
            t: Any tensor read off a live buffer.

        Returns:
            The same values as float32 on ``self.device``.
        """
        return t.to(device=self.device, dtype=torch.float32)

    def _env_ids(self, env_ids) -> torch.Tensor:
        """Validate and normalize the env selection to a ``(V,)`` long tensor.

        A BOOLEAN mask is refused rather than converted.  ``torch.as_tensor(mask,
        dtype=torch.long)`` turns one into a vector of 0s and 1s, which indexes envs 0 and 1
        repeatedly -- a full-width, finite, entirely wrong batch, and the caller's own mask
        (``cmd._fm_reinfer_mask``) is exactly that shape.

        Args:
            env_ids: The caller's selection.

        Returns:
            ``(V,)`` long on ``self.device``.

        Raises:
            ValueError: On a boolean or non-1-D selection.
        """
        t = torch.as_tensor(env_ids, device=self.device)
        if t.dtype == torch.bool:
            raise ValueError(
                "env_ids is a boolean mask; pass indices (`mask.nonzero().flatten()`). A mask "
                "coerced to long would index envs 0 and 1 over and over and every downstream "
                "shape would still be right.")
        if t.ndim != 1:
            raise ValueError(f"env_ids must be 1-D, got shape {tuple(t.shape)}")
        return t.to(torch.long)
