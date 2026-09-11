# Copyright 2026 Zachary Olkin. All rights reserved.

"""Score ``P(fail)`` inside the MuJoCo loop, with the critic fitted on Isaac rollouts.

``scripts/flow_matching/fit_value.py`` trains a failure-probability head over the generator's
own frozen encoder.  Until now it could only be read in Isaac, because three of its context
features are goal-region quantities that need a live ``MotionSequenceCommand``.
:mod:`goal_anchor` removes that dependency, and this module is the other half: it assembles the
seven :class:`~robot_rl.flow_matching.value.ValueInputs` from the MuJoCo runtime's own state and
runs the head.

**Why here and not in Isaac.**  Best-of-K needs to branch K candidate chunks from one state and
compare them.  The Isaac attempt at that was abandoned -- its state copy matched the four
physics buffers exactly but not what the controller reads (``obs_policy`` differed by 2.375 in
the same instant), and its same-noise control matched the real signal, so it had no resolving
power.  MuJoCo needs no restore at all: replaying from spawn with a fixed noise tape is exactly
deterministic, which is what ``fm_sample_explorer.py`` already relies on.

**Import cost, measured rather than assumed.**  ``robot_rl.flow_matching.online`` pulls 152
``isaaclab.*`` modules but **zero** ``isaacsim`` / ``omni`` / ``pxr`` / ``carb`` -- it is pure
torch, 1.1 s, no Kit and no ``AppLauncher`` -- and coexists with ``mujoco`` and ``onnxruntime``
in one process.  So this imports the real modules directly rather than stubbing package parents
to dodge ``robot_rl/__init__``; "drags in IsaacLab" is true of the package but does not mean
"boots a simulator".  The cost is that the critic arm needs ``env_isaaclab``: ``env_transfer``
carries no torch, deliberately, and a ``FlowValueModel`` is an ``nn.Module``.  Without
``--critic_ckpt`` nothing here is imported and mjeval stays torch-free.

**The exteroception row is the input most likely to be silently wrong**, because three
shape-identical wrong answers sit beside the right one: ``policy.last_plan()["extero_tokens"]``
is already normalized AND sanitized (feeding it normalizes twice), a row-major fold scrambles
every convolution kernel while the shape stays right, and the clamped-vs-unclamped question
below.  :meth:`CriticInputs.from_runtime` is therefore the only supported way to build one --
it takes the runtime, not an array, so the wrong array is unreachable at the call site.

**Known distribution shift, stated because it is real.**  ``_fm_extero_row`` clamps the DEPTH
channel before unprojecting xyz (deliberately -- sanitising the widened row moved points by up
to 6 m), whereas the recording ``fit_value`` trains on stores depth unclipped.  ~3.1% of row
entries differ, worth mean 0.022 / max 0.297 in ``p_fail`` over 32 windows.  This module uses
the runtime's row anyway, so the critic scores exactly what the generator was conditioned on;
the mismatch is with TRAINING, which is a transfer question, not a port bug.  ``TODO(critic-
extero-clamp)``: quantify it against an unclamped row built from ``state["fm_depth"]``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

#: Row width of one proprioception step: joint_pos(29) + joint_vel(29) + ang_vel_b(3) +
#: projected_gravity_b(3). Pinned by ``rollout_outcomes.PROPRIO_DIM``.
PROPRIO_DIM = 64

#: Height-scan fold. ``transforms.scan_grid_shape(0.1, [1.5, 1.0], "xy")`` -> ``(n_y, n_x)``,
#: the row-major fold of the y-outer/x-inner cast ``Robot.compute_height_scan`` returns. The
#: grid is NOT square, so a transposed fold fails loudly here -- unlike the privileged 121x121
#: scan, where it is shape-silent.
SCAN_HW = (11, 16)


class ProprioRing:
    """The last ``H`` control steps of proprioception, newest last.

    A numpy, single-environment mirror of ``record_rollouts.ProprioRing``, and it must stay an
    exact one -- the critic reads these rows as tokens, so any disagreement is a silent
    distribution shift rather than an error.  Three specifics that are easy to get wrong and
    each of which the Isaac original states deliberately:

    * kept at the CONTROL rate, not the re-plan rate (a fall develops over tens of ms; sampled
      every 0.24 s the onset is simply not in the data, and the onset is what the lead time is
      about);
    * ``joint_pos`` is ABSOLUTE, not offset from the default pose;
    * zeroed on reset, so no window spans two episodes.

    **Do not substitute ``policy._obs_history``** even though the depth tracker's terms carry
    ``history_length=10``: it stores ``joint_pos - default_joint_angles``, post-clip, and its
    first push FILLS all ten slots with that sample instead of zero-padding (IsaacLab
    ``CircularBuffer`` semantics).  Three silent mismatches at once.

    Args:
        history: Rows kept.
        dim: Row width; must equal :data:`PROPRIO_DIM`.
    """

    def __init__(self, history: int = 10, dim: int = PROPRIO_DIM) -> None:
        if int(dim) != PROPRIO_DIM:
            raise ValueError(f"proprio row width is {PROPRIO_DIM}, got {dim}")
        self.history = int(history)
        self.buf = np.zeros((self.history, int(dim)), dtype=np.float32)

    def push(self, joint_pos: np.ndarray, joint_vel: np.ndarray,
             ang_vel_b: np.ndarray, projected_gravity_b: np.ndarray) -> None:
        """Roll in this control step's row.

        Args:
            joint_pos: ``(29,)`` ABSOLUTE joint positions, Isaac order.
            joint_vel: ``(29,)`` joint velocities, Isaac order.
            ang_vel_b: ``(3,)`` base angular velocity, BODY frame.
            projected_gravity_b: ``(3,)`` gravity direction in the body frame.
        """
        row = np.concatenate([np.asarray(joint_pos, dtype=np.float32).ravel(),
                              np.asarray(joint_vel, dtype=np.float32).ravel(),
                              np.asarray(ang_vel_b, dtype=np.float32).ravel(),
                              np.asarray(projected_gravity_b, dtype=np.float32).ravel()])
        if row.size != PROPRIO_DIM:
            raise ValueError(f"proprio row is {row.size} wide, expected {PROPRIO_DIM} "
                             "(joint_pos 29 + joint_vel 29 + ang_vel 3 + gravity 3)")
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = row

    def invalidate(self) -> None:
        """Zero the ring, so no window spans two episodes."""
        self.buf[:] = 0.0


def projected_gravity_b(root_quat_xyzw: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the base frame.

    Delegates to ``deploy.quat_math_np``, the tree's parity-tested numpy quaternion port,
    rather than restating the algebra.  Hand-rolling it here was wrong once already, and the
    error was the kind this whole module is built to avoid: a finite, plausible rotation.

    Args:
        root_quat_xyzw: ``(4,)`` base orientation, **x-first** -- exactly what
            ``build_state`` puts in ``state["root_quat"]`` (``wxyz_to_xyzw(qpos[3:7])``,
            ``gen_tracker_runner.py:239``).  The whole numpy deploy path is xyzw, so there is
            no conversion anywhere on this path and none should be added.

    Returns:
        ``(3,)`` unit gravity in the body frame -- ``[0, 0, -1]`` when upright.
    """
    from deploy.quat_math_np import quat_apply_inverse  # noqa: PLC0415

    q = np.asarray(root_quat_xyzw, dtype=np.float64).ravel()
    return quat_apply_inverse(q, np.array([0.0, 0.0, -1.0])).astype(np.float32)


def twist_yaw(root_quat_xyzw: np.ndarray, lin_vel_w: np.ndarray,
              ang_vel_w: np.ndarray) -> np.ndarray:
    """The measured base twist in the robot's YAW frame, ``(6,)`` linear then angular.

    The numpy twin of ``record_rollouts._twist_yaw``.  ``yaw_quat`` is the level frame the
    generator predicts its window in -- z is gravity, roll and pitch are zero -- so this is
    directly comparable with the plan's own ``base_vel`` block without a transform.  The body
    frame would tilt with the torso and swing the forward component on every pitch; world
    would defeat heading invariance.

    Args:
        root_quat_xyzw: ``(4,)`` base orientation, **x-first** (see
            :func:`projected_gravity_b`).
        lin_vel_w: ``(3,)`` base linear velocity, WORLD frame.
        ang_vel_w: ``(3,)`` base angular velocity, WORLD frame.

    Returns:
        ``(6,)`` float32.
    """
    from deploy.quat_math_np import quat_apply_inverse, yaw_quat  # noqa: PLC0415

    q = yaw_quat(np.asarray(root_quat_xyzw, dtype=np.float64).ravel())
    return np.concatenate([
        quat_apply_inverse(q, np.asarray(lin_vel_w, dtype=np.float64).ravel()),
        quat_apply_inverse(q, np.asarray(ang_vel_w, dtype=np.float64).ravel()),
    ]).astype(np.float32)


@dataclass
class CriticInputs:
    """The seven raw arrays a :class:`FailureCritic` consumes, batch of 1.

    Raw means PRE-normalizer: :meth:`FailureCritic.p_fail` applies the generator's own
    ``state_norm`` / ``extero_norm`` / ``cmd_norm`` itself, exactly as ``fit_value._batch``
    does.  Handing it an already-normalized array is the port's most likely silent failure and
    is why :meth:`from_runtime` exists.

    Attributes:
        history: ``(N, D)`` the generator's last N emitted nodes, oldest first.
        extero: ``(T, K)`` or flat -- the exteroception row it was served.
        cmd_future: ``(F, cmd_dim)`` UNSTRIDED future command; striding is applied downstream.
        history_null: True while the ring had not filled.
        scan: ``(11, 16)`` privileged height scan, or ``None``.
        proprio: ``(10, 64)`` proprioception ring, newest last, or ``None``.
        goal: ``(3,)`` downrange / lateral / height error, METRES (scaled downstream).
        twist: ``(6,)`` yaw-frame base twist, linear then angular.
    """

    history: np.ndarray
    extero: np.ndarray
    cmd_future: np.ndarray
    history_null: bool
    goal: np.ndarray
    twist: np.ndarray
    scan: np.ndarray | None = None
    proprio: np.ndarray | None = None

    @classmethod
    def from_runtime(cls, policy, robot, state: dict, *, anchor: np.ndarray,
                     ring: ProprioRing | None, want_scan: bool) -> "CriticInputs":
        """Assemble from the live MuJoCo runtime at a re-plan.

        Takes the runtime rather than arrays on purpose: every one of the seven has at least
        one shape-identical wrong source sitting beside it, and a call site that never names
        an array cannot pick the wrong one.

        Must be called on a control step where ``policy.last_plan()`` describes the re-plan
        just made -- the same instant ``record_rollouts._emit`` reads, and the same edge
        ``runner.run_episode`` already detects via ``plan["plan_id"]``.

        Args:
            policy: The ``GenTrackerPolicy``.
            robot: The ``transfer.sim.robot.Robot``, for the height-scan cast.
            state: The dict ``build_state`` produced for this step.
            anchor: ``(3,)`` goal anchor in MuJoCo world coords (:mod:`goal_anchor`).
            ring: The proprioception ring, or ``None`` for a non-privileged critic.
            want_scan: Cast the 176-ray height scan (privileged critic only).

        Returns:
            The assembled inputs.
        """
        from goal_anchor import frame_offsets  # noqa: PLC0415

        plan = policy.last_plan()
        hist = np.asarray(plan["history_nodes"], dtype=np.float32).reshape(
            -1, int(np.asarray(plan["history_nodes"]).shape[-1]))
        # RAW: `_fm_extero_row` is the un-normalized fused row. `plan["extero_tokens"]` is the
        # SAME shape but already through extero_norm + sanitize -- see the module docstring.
        extero = np.asarray(policy._fm_extero_row(state), dtype=np.float32)
        cmd_future = np.asarray(plan["future_cmd"], dtype=np.float32).reshape(
            -1, int(np.asarray(plan["future_cmd"]).shape[-1]))

        scan = None
        if want_scan:
            # `state["root_quat"]` is XYZW -- `build_state` stores wxyz_to_xyzw(qpos[3:7])
            # (gen_tracker_runner.py:239) and `compute_height_scan` wants exactly that, which
            # is why it is passed through unconverted here and converted below.
            clr, _ = robot.compute_height_scan(state["root_pos"], state["root_quat"])
            # -inf for a miss, matching the Isaac recording; `transforms.sanitize_extero`
            # downstream is what maps it into the band. A finite fill here would be a second,
            # disagreeing sanitiser.
            scan = np.where(np.isnan(clr), -np.inf, clr).astype(np.float32).reshape(*SCAN_HW)

        return cls(
            history=hist, extero=extero, cmd_future=cmd_future,
            history_null=bool(plan["history_null"]),
            goal=frame_offsets(np.asarray(state["root_pos"], dtype=np.float64), anchor),
            # No conversion: state["root_quat"] and deploy.quat_math_np are both XYZW.
            twist=twist_yaw(state["root_quat"],
                            state["root_lin_vel_w"], state["root_ang_vel_w"]),
            scan=scan,
            proprio=(None if ring is None else ring.buf.copy()),
        )


def _sha256(path: str) -> str:
    """Content hash of a file, for refusing a mismatched pair.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FailureCritic:
    """A trained ``P(fail)`` head, loaded for inference in the MuJoCo process.

    Args:
        ckpt: A ``fit_value.py`` checkpoint (``V_priv.fold0.pt`` and friends).
        fm_pt: The generator ``.pt`` the head was fitted over.  Defaults to the path recorded
            in the checkpoint.  A DIFFERENT generator loads with the right shapes and three
            wrong normalizers, which is silent, so the two are compared by content hash.
        device: Torch device.

    Raises:
        FileNotFoundError: If either artifact is missing.
        ValueError: If ``fm_pt`` is not the generator the checkpoint names.
    """

    def __init__(self, ckpt: str, *, fm_pt: str | None = None, device: str = "cuda") -> None:
        import copy  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from robot_rl.flow_matching import online, value  # noqa: PLC0415

        self._torch = torch
        self.device = device
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        named = str(ck["fm_checkpoint"])
        use = fm_pt or named
        if not os.path.exists(use):
            raise FileNotFoundError(
                f"generator checkpoint {use!r} not found (named by {os.path.basename(ckpt)})")
        if fm_pt is not None and os.path.abspath(fm_pt) != os.path.abspath(named):
            # THE CHECKPOINT'S OWN HASH FIRST, when it has one. `fm_checkpoint` is an absolute
            # PATH, so a critic fitted on one machine and read on another names a location
            # that does not exist -- and a path is not what identifies a generator anyway.
            # With the hash stamped, a relocated-but-identical artifact verifies; without it,
            # the pair genuinely cannot be checked and the load still refuses.
            want = str(ck.get("fm_checkpoint_sha256", "") or "")
            if want:
                got = _sha256(fm_pt)
                if got != want:
                    raise ValueError(
                        f"critic {os.path.basename(ckpt)} was fitted over a generator with "
                        f"sha256 {want[:16]}..., but {fm_pt!r} hashes {got[:16]}.... The "
                        "encoder, state_norm, extero_norm and cmd_norm all come from it; a "
                        "mismatch is silent.")
            elif not os.path.exists(named):
                raise ValueError(
                    f"--critic_fm_pt {fm_pt!r} differs from the one the critic names "
                    f"({named!r}), that one is missing, and this checkpoint carries no "
                    "`fm_checkpoint_sha256` -- so the pair cannot be verified by content. A "
                    "different generator loads with the right shapes and three WRONG "
                    "normalizers, which nothing downstream can detect. Re-fit to get the hash "
                    "stamped, or run where the named path resolves.")
            # ELIF, not a second check: once the stamped hash has verified, the named path is
            # irrelevant and hashing it would fail on exactly the cross-machine case the hash
            # exists to support. These two routes are alternatives, never both.
            elif _sha256(fm_pt) != _sha256(named):
                raise ValueError(
                    f"critic {os.path.basename(ckpt)} was fitted over {named!r}, but "
                    f"{fm_pt!r} has different content. The encoder, state_norm, extero_norm "
                    "and cmd_norm all come from it; a mismatch is silent.")

        self.privileged = bool(ck["privileged"])
        self.variant = str(ck.get("variant", "?"))
        # WHETHER THE GOAL OFFSETS ARE IN THE CONTEXT TOKEN. Read from the stamp, never
        # inferred from the width: 9 is V_priv-with-goal and 6 is BOTH V_onboard-with-goal
        # and V_priv-without, so the width alone cannot tell the arms apart. `True` for an
        # absent stamp is a fact about pre-flag checkpoints -- all of them carried the goal --
        # rather than today's default re-resolved onto an older artifact, which is the trap
        # CLAUDE.md records for `extero_clip`, `attention_mode` and `skip_families`.
        self.goal_context = bool(ck.get("goal_context", True))
        #: Whether the critic's context token carries the worst-terrain-penalty-so-far block.
        #: Read from the checkpoint and never assumed: such a fit is one block WIDER, so a
        #: check omitting the flag rejects a good critic -- and, if the widths ever coincided,
        #: would accept a wrong one.
        self.worst_penalty_context = bool(ck.get("worst_penalty_context", False))
        self.ctx = online.load_fm_context(use, device=device)
        # The QUALITY head's width has to be known BEFORE the load, because the head is built
        # by the ctor and `load_state_dict` is strict. Preference order, and the fallback is
        # not a guess: `quality_support` is a persistent buffer of exactly `quality_bins` bin
        # centres, so an unstamped checkpoint still STATES its own width -- it just states it
        # in the tensor rather than in a scalar. Stamped is better only because it also
        # carries the spec, which no shape can recover.
        self.quality_bins = int(ck.get("quality_bins", 0)) or int(
            ck["model"]["quality_support"].numel() if "quality_support" in ck["model"] else 0)
        self.model = value.FlowValueModel(
            copy.deepcopy(self.ctx.model),
            scan_hw=(tuple(ck["scan_hw"]) if self.privileged else None),
            proprio_dim=(int(ck["proprio_hw"][-1]) if self.privileged else 0),
            proprio_steps=(int(ck["proprio_hw"][0]) if self.privileged else 0),
            context_dim=int(ck["context_dim"]),
            quality_bins=self.quality_bins).to(device)
        # The stamp against the width the two flags imply. `load_state_dict` checks the
        # checkpoint against the MODEL and both come from `context_dim`, so it cannot see a
        # stamp that disagrees with them -- and the failure mode is not a shape error but a
        # linear layer reading the twist out of the goal's columns.
        if self.worst_penalty_context:
            # A critic fitted with `--worst_context` reads the worst terrain penalty seen so
            # far in the episode, and NOTHING on this path computes a terrain penalty -- there
            # is no grader in this loop and no running worst to hand it. Refused here, at load,
            # rather than at the context builder: the builder would report a missing piece,
            # which reads as a bug in this file instead of as the missing plumbing it is.
            raise SystemExit(
                f"{os.path.basename(ckpt)} was fitted with --worst_context, whose context "
                "block is the worst terrain penalty BEFORE each chunk. This consumer has no "
                "terrain grader and cannot supply it. Fit the critic without --worst_context "
                "to score it here, or add a running worst-penalty accumulator to this path.")
        want = value.context_width(privileged=self.privileged,
                                   goal_context=self.goal_context,
                                   worst_penalty_context=self.worst_penalty_context)
        if int(ck["context_dim"]) != want:
            raise ValueError(
                f"{os.path.basename(ckpt)} stamps context_dim={int(ck['context_dim'])} but "
                f"privileged={self.privileged} + goal_context={self.goal_context} + "
                f"worst_penalty_context={self.worst_penalty_context} build "
                f"{want}. One of the three is wrong, and every combination of the other two "
                "produces a context vector that is finite, plausible and reading the wrong "
                "columns.")
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        # The reward the quality head predicts a RETURN of. `None` on a pre-stamp checkpoint,
        # which is why `live_select` requires one from somewhere and refuses to invent it: the
        # head's G_hat is a return under one set of sigmas, bounds and gamma, and adding a
        # reward measured under a different spec to it sums two different quantities. Both are
        # plausible floats and nothing downstream can tell them apart.
        self.tracking_spec = None
        if ck.get("tracking_spec"):
            from deploy.tracking_error import TrackingErrorSpec  # noqa: PLC0415
            self.tracking_spec = TrackingErrorSpec.from_dict(dict(ck["tracking_spec"]))
        self.gamma = float(ck.get("gamma", 0.0)) or (
            float(self.tracking_spec.gamma) if self.tracking_spec is not None else 0.0)
        self.scan_hw = tuple(ck["scan_hw"]) if self.privileged else None
        if self.privileged and self.scan_hw != SCAN_HW:
            raise ValueError(f"critic expects scan {self.scan_hw}, this port casts {SCAN_HW}")

    @property
    def has_quality(self) -> bool:
        """Whether this checkpoint carries a quality head at all."""
        return self.quality_bins > 0

    def p_fail(self, x: CriticInputs) -> float:
        """Failure probability for one state.

        Args:
            x: The raw inputs.

        Returns:
            ``P(fail)`` in [0, 1].
        """
        import torch  # noqa: PLC0415

        with torch.no_grad():
            return float(self.model.p_fail(self._inputs(x)).item())

    def read(self, x: CriticInputs) -> tuple[float, float | None]:
        """Both heads for one state, from ONE encoder pass.

        What the selector calls. Two separate calls would run the borrowed layer stack twice
        over identical tokens, and the stack is ~97% of the parameters and essentially all of
        the cost -- so reading the second head is free and reading it separately is not.

        Args:
            x: The raw inputs.

        Returns:
            ``(P(fail), G_hat)``, the second ``None`` if this checkpoint has no quality head.
        """
        import torch  # noqa: PLC0415

        with torch.no_grad():
            v = self._inputs(x)
            fail_logit, q_logits = self.model.heads(v)
            p = float(torch.sigmoid(fail_logit).item())
            if q_logits is None:
                return p, None
            probs = torch.softmax(q_logits, dim=-1)
            return p, float((probs * self.model.quality_support).sum(dim=-1).item())

    def _inputs(self, x: CriticInputs):
        """Normalize and assemble one state into a ``value.ValueInputs``.

        Normalizer order mirrors ``fit_value._batch`` exactly: ``state_norm`` on the history,
        ``extero_norm`` on the flattened row (NO sanitize -- the recording's column is not
        sanitised either), ``cmd_norm`` on the STRIDED command, ``sanitize_extero`` on the
        scan alone, and the goal on the declared ``GOAL_SCALE_M``.

        Args:
            x: The raw inputs.

        Returns:
            The assembled batch of one.
        """
        import torch  # noqa: PLC0415
        from goal_anchor import GOAL_SCALE_M  # noqa: PLC0415
        from robot_rl.flow_matching import transforms, value  # noqa: PLC0415

        t = lambda a, d=torch.float32: torch.as_tensor(  # noqa: E731
            np.asarray(a), dtype=d, device=self.device).unsqueeze(0)
        ctx = self.ctx
        hist = ctx.state_norm.normalize(t(x.history))
        ext = ctx.extero_norm.normalize(t(x.extero).reshape(1, -1))
        cmd = None
        if ctx.cmd_dim > 0:
            steps = torch.as_tensor(ctx.cmd_steps, dtype=torch.long, device=self.device)
            cmd = ctx.cmd_norm.normalize(t(x.cmd_future)[:, steps, :])
        scan = proprio = None
        if self.privileged:
            if x.scan is None or x.proprio is None:
                raise ValueError(f"{self.variant} is privileged and needs both a height scan "
                                 "and a proprioception ring; got "
                                 f"scan={x.scan is None and 'None' or 'ok'}, "
                                 f"proprio={x.proprio is None and 'None' or 'ok'}")
            scan = transforms.sanitize_extero(t(x.scan), "height_scan").reshape(1, *SCAN_HW)
            proprio = t(x.proprio)
        tw = np.asarray(x.twist, dtype=np.float32)
        # ONBOARD keeps only the ANGULAR half: an IMU measures body rates, but nothing on the
        # robot measures the linear half without odometry. Slicing it away is what makes the
        # two variants' gap a statement about information rather than about architecture.
        # `value.build_context` owns the order -- the same call the fit makes, so this reader
        # and the process that trained the checkpoint cannot disagree about which column is
        # which. Built as tensors rather than numpy for that reason.
        pieces = {"twist": t(np.asarray(tw, dtype=np.float64))}
        if self.goal_context:
            pieces["goal"] = t(np.nan_to_num(
                np.asarray(x.goal, dtype=np.float64) / GOAL_SCALE_M,
                nan=0.0, posinf=0.0, neginf=0.0))
        context = value.build_context(privileged=self.privileged,
                                      goal_context=self.goal_context, **pieces)
        return value.ValueInputs(
            history=hist, extero=ext, cmd_tokens=cmd,
            history_null=t(np.array([x.history_null]), torch.bool).reshape(1),
            scan=scan, proprio=proprio, context=context)


class CriticScorer:
    """Drives a :class:`FailureCritic` from inside ``runner.run_episode``.

    Passed as that function's ``probe``.  It owns the proprioception ring because the ring
    runs at the CONTROL rate while the critic is only read at a re-plan, so the two cadences
    have to live in one object or the ring ends up sampled at the wrong one -- which is the
    single mistake that would make ``V_priv`` quietly worse than ``V_onboard`` rather than
    better, since proprioception is half of what the privileged variant adds.

    Args:
        critic: The loaded head.
        anchor: ``(3,)`` goal anchor in the course's world frame -- ``Course.goal_anchor_xyz``.
        robot: The ``transfer.sim.robot.Robot``, for the height-scan cast.

    Raises:
        ValueError: If ``anchor`` is ``None``.  A course with no anchor cannot be scored, and
            substituting ``target_xy`` would read as an anchor while being off by 0.44-3.17 m
            depending on family (see :mod:`goal_anchor`).
    """

    def __init__(self, critic: FailureCritic, anchor, robot) -> None:
        if anchor is None:
            raise ValueError(
                "this course carries no goal_anchor_xyz, so the critic's three goal features "
                "have nothing to measure from. Build it with "
                "`isaac_tile_course(..., goal_anchor_max_offset=50)`; do NOT fall back to "
                "target_xy, which is the far platform's centre and differs from the clip's "
                "own endpoint by a family-systematic 0.44-3.17 m.")
        self.critic = critic
        self.anchor = np.asarray(anchor, dtype=np.float64)
        self.robot = robot
        self.ring = ProprioRing() if critic.privileged else None
        self.p_fail: list[float] = []
        self.replan_step: list[int] = []
        self._step = 0

    def reset(self) -> None:
        """Clear the ring and the record, between episodes.

        The ring MUST be zeroed here: a window spanning two episodes is ten rows of a robot
        that no longer exists, and it is finite and plausible.  ``runner._reset_episode_state``
        is the matching call site.
        """
        if self.ring is not None:
            self.ring.invalidate()
        self.p_fail.clear()
        self.replan_step.clear()
        self._step = 0

    def __call__(self, robot, policy, state: dict, is_replan: bool) -> None:
        """One control step: always ring, sometimes score.

        Args:
            robot: The MuJoCo robot.
            policy: The ``GenTrackerPolicy``.
            state: The dict the policy was just given.
            is_replan: Whether the generator re-planned on this step.
        """
        if self.ring is not None:
            self.ring.push(state["joint_pos"], state["joint_vel"], state["base_ang_vel"],
                           projected_gravity_b(state["root_quat"]))
        if is_replan:
            x = CriticInputs.from_runtime(
                policy, robot, state, anchor=self.anchor, ring=self.ring,
                want_scan=self.critic.privileged)
            self.p_fail.append(self.critic.p_fail(x))
            self.replan_step.append(self._step)
        self._step += 1

    def summary(self) -> dict:
        """Per-episode aggregates, for a results row.

        ``first`` and ``last`` are kept alongside the moments because the shape of the trace
        is the whole point -- a critic that only rises after recovery is impossible scores
        well on ``max`` while being useless for selection, and only the trace shows that.

        Returns:
            ``{}`` when nothing was scored, else the aggregates.
        """
        if not self.p_fail:
            return {}
        p = np.asarray(self.p_fail, dtype=np.float64)
        return {"p_fail_mean": float(p.mean()), "p_fail_max": float(p.max()),
                "p_fail_first": float(p[0]), "p_fail_last": float(p[-1]),
                "n_critic_calls": int(p.size)}
