# Copyright 2026 Zachary Olkin. All rights reserved.

"""The action -> residual map, with ONE producer.

:meth:`AdjusterEnv._residual_from_action` applies this during training.  Anything that drives a
trained adjuster from outside that env -- the eval harness, a viewer -- must apply the IDENTICAL
map, because the lift, the squash order and the per-channel caps together are what the policy's
output means.  A second transcription of it is a silent way for a scored rollout to differ from
the rollouts that trained the policy.

Kept in its own module rather than on the env class so an off-env consumer can import it
without importing :class:`ManagerBasedRLEnv`, which reaches ``pxr`` and cannot be bound before
AppLauncher runs.

:class:`AdjusterPolicy` drives a trained one from OUTSIDE :class:`AdjusterEnv`, which is what
lets the eval harness score this arm against the off-policy ones: the eval must run the PLAIN
free-form task at the control rate, because that is the task every other arm was recorded on
and the comparison is only worth anything if the two recordings differ in one thing.
"""

from __future__ import annotations

import os

import torch


def residual_from_action(action: torch.Tensor, knot_lift: torch.Tensor,
                         residual_scale: torch.Tensor) -> torch.Tensor:
    """``[B, K*D]`` dimensionless action -> ``[B, F, D]`` residual in raw units.

    **Lift, then squash, then scale, and the order is load-bearing.** Catmull-Rom overshoots by
    up to 25% between knots (worst row L1 1.2499), so squashing the knots first would let the
    interpolated window exceed the per-channel cap it is supposed to respect. ``tanh`` rather
    than a clamp so the gradient never vanishes and ``init_std`` stays meaningful.

    At one knot per node the lift is exactly the identity, so this is also the full-width path.

    Args:
        action: ``[B, K*D]``, the policy's output.
        knot_lift: ``[F, K]`` the fixed basis, from :func:`~robot_rl.flow_matching.knot_basis`.
        residual_scale: ``[D]`` per-channel authority in raw units.

    Returns:
        ``[B, F, D]``.

    Raises:
        ValueError: If the action width is not ``K * D``.
    """
    k, d = int(knot_lift.shape[1]), int(residual_scale.shape[-1])
    if action.shape[-1] != k * d:
        raise ValueError(
            f"expected an action of width K * D = {k} * {d} = {k * d}, got "
            f"{int(action.shape[-1])}. The actor must have been built with this K.")
    shaped = action.view(-1, k, d)
    lifted = torch.einsum("fk,bkd->bfd", knot_lift, shaped)
    return torch.tanh(lifted) * residual_scale




def actor_cfg_from_checkpoint(state: dict, cmd) -> dict:
    """The SHAPE-CRITICAL actor config, derived from the checkpoint's own tensors.

    The authoritative source, and the reason this exists rather than trusting the run's yaml:
    ``params/agent.yaml`` parsed correctly standalone and came back WITHOUT ``num_knots`` inside
    a running Kit process, deterministically, on the same file (`KeyError: 'num_knots'`, all
    five eval cells). Whatever Omniverse does to the import environment, the checkpoint's own
    tensors cannot be shadowed by it -- ``distribution.log_knot_scale`` IS ``[K]`` and
    ``distribution.log_group_scale`` IS ``[G]``.

    Only the two fields that decide the parameter SET are taken here. Everything else is either
    inert or caught: the caller's ``load_state_dict`` check refuses any mismatch, so a wrong
    ``num_frozen_layers`` fails loudly rather than loading a partial actor.

    Args:
        state: The actor ``state_dict`` from the checkpoint.
        cmd: The live :class:`FmTrackingCommand`, for the channel layout.

    Returns:
        Keyword arguments for :class:`WindowAdjusterModel`; possibly empty.
    """
    from robot_rl.rl.knot_distribution import channel_groups                # noqa: PLC0415

    cfg: dict = {}
    knots = state.get("distribution.log_knot_scale")
    if knots is not None:
        cfg["num_knots"] = int(knots.shape[0])
    if "distribution.log_group_scale" in state:
        # The NAMES are what size the grouping; the values are overwritten by the load. Taken
        # from the same producer the model uses, so the two cannot disagree about the packing.
        _, names = channel_groups(cmd.ctx.nj, cmd.ctx.target_num_extra_frames,
                                  cmd.ctx.reference_blocks)
        cfg["group_init_std"] = {n: 0.1 for n in names}
    return cfg


def actor_cfg_from_run(checkpoint: str) -> dict:
    """The actor config the run was trained with, BEST EFFORT, from ``params/agent.yaml``.

    Supplies the fields the checkpoint's tensors cannot show -- ``num_frozen_layers``,
    ``activation``, the observation group names. Best effort on purpose: this file has been
    observed to parse incompletely inside a running Kit process, so a failure here warns and
    returns what it has rather than killing the eval. Everything it could get wrong is verified
    by the caller's ``load_state_dict`` check.

    Args:
        checkpoint: Path to ``<run>/model_*.pt``.

    Returns:
        Keyword arguments for :class:`WindowAdjusterModel`; possibly empty.
    """
    p = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "params", "agent.yaml")
    try:
        import yaml                                                         # noqa: PLC0415

        with open(p) as f:
            d = yaml.safe_load(f)
        actor = dict((d or {}).get("actor") or {})
        actor.pop("class_name", None)
        if not actor:
            raise ValueError("no `actor` block")
        return actor
    except Exception as exc:                                       # noqa: BLE001 -- see above
        print(f"[WARN] AdjusterPolicy: could not read {p} ({exc}); falling back to the "
              "checkpoint's own tensors plus defaults. The state-dict check below is what "
              "makes that safe.")
        return {}


class AdjusterPolicy:
    """A trained adjuster, applied once per re-plan on a task that is not :class:`AdjusterEnv`.

    Three things must match a training rollout or the scored episodes are not the policy's:

    * the action -> residual map, which is :func:`residual_from_action` and is shared;
    * WHICH step it lands on. The env applies it at an interval boundary where every env has
      just re-planned; here the loop runs one CONTROL step at a time, so it is masked to the
      1-in-K that re-planned. Applying it on the other eleven would stack a second correction
      onto a window already corrected;
    * the residual being DETERMINISTIC. Training samples the Gaussian; scoring takes its mean
      (``MLPModel.forward`` defaults to ``stochastic_output=False``).

    The generator whose frozen prefix it borrows is the one the COMMAND loaded, published via
    ``online.active_fm_context()`` -- so the eval must pass the same ``--fm_checkpoint`` the
    adjuster trained against. That is what you want anyway: a residual is a correction to one
    generator's output and means nothing applied to another's.

    Attributes:
        model: The :class:`WindowAdjusterModel`, in eval mode.
        iteration: The training iteration the checkpoint was written at, or -1.
    """

    def __init__(self, checkpoint: str, env, command_name: str = "traj_ref",
                 num_knots: int | None = None) -> None:
        """Rebuild the actor against the live env and restore its weights.

        Args:
            checkpoint: A ``model_*.pt`` from the adjuster's runner.
            env: The live UNWRAPPED env, already built.
            command_name: The :class:`FmTrackingCommand` term name.
            num_knots: Override for ``K``. Normally omitted -- the whole actor config is
                read from the run's ``params/agent.yaml``, which is the only record of it.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
            KeyError: If it carries no actor weights, or they do not fit.
        """
        from tensordict import TensorDict                                   # noqa: PLC0415

        from robot_rl.flow_matching.knot_basis import knot_basis            # noqa: PLC0415
        from robot_rl.flow_matching.window import residual_scale_vector     # noqa: PLC0415
        from robot_rl.rl.adjuster_model import WindowAdjusterModel          # noqa: PLC0415
        from robot_rl.tasks.manager_based.robot_rl import mdp               # noqa: PLC0415

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"adjuster checkpoint not found: {checkpoint}")
        self._env, self._name = env, command_name
        cmd = env.command_manager.get_term(command_name)
        self._cmd = cmd
        payload = torch.load(checkpoint, map_location=env.device, weights_only=False)
        state = payload.get("actor_state_dict") or payload.get("model_state_dict")
        if state is None:
            raise KeyError(f"{os.path.basename(checkpoint)} carries no actor weights "
                           f"(keys: {sorted(payload)}).")
        # yaml first for the descriptive fields, then the CHECKPOINT over the top for the two
        # that decide the parameter set. The checkpoint wins because it is the thing being
        # loaded: if the two ever disagree, the yaml is the one that is wrong.
        cfg = actor_cfg_from_run(checkpoint)
        cfg.update(actor_cfg_from_checkpoint(state, cmd))
        if num_knots is not None:
            cfg["num_knots"] = int(num_knots)
        if "num_knots" not in cfg:
            # SELF-DESCRIBING, because this has now failed twice for reasons that were not
            # visible from the message. Prints inside a Kit process are unreliable (Kit
            # captures stdout during startup), but an exception's text always reaches the log.
            import hashlib                                                  # noqa: PLC0415

            try:
                h = hashlib.md5(open(checkpoint, "rb").read(1 << 20)).hexdigest()[:8]
                size = os.path.getsize(checkpoint)
            except Exception as exc:                                        # noqa: BLE001
                h, size = f"<{exc}>", -1
            raise KeyError(
                "cannot determine K. "
                f"checkpoint={checkpoint!r} exists={os.path.isfile(checkpoint)} "
                f"size={size} md5_1mb={h} | payload keys={sorted(payload)} | "
                f"state type={type(state).__name__} n={len(state)} "
                f"first={list(state)[:6]} | "
                f"distribution keys={[k for k in state if 'distribution' in k]} | "
                f"yaml gave={sorted(actor_cfg_from_run(checkpoint))}")
        self._k = int(cfg["num_knots"])
        kind = str(cfg.get("knot_kind", "catmull"))

        # Both groups are pure reads of the command's stash, so the model is sized from a
        # synthetic dict rather than by installing observation groups on the eval task -- which
        # would add ~7.3k floats per env per control step and reorder the observation manager,
        # for nothing.
        cond = mdp.fm_plan_conditioning(env, command_name)
        plan = mdp.fm_plan_window(env, command_name)
        obs = TensorDict({"fm_cond": cond, "fm_plan": plan}, batch_size=[cond.shape[0]])
        f, d = int(cmd.ctx.F), int(cmd.ctx.D)
        self.model = WindowAdjusterModel(
            obs=obs, obs_groups={"actor": ["fm_cond", "fm_plan"]}, obs_set="actor",
            output_dim=f * d, **cfg).to(env.device)

        # The borrowed generator is held in a plain list so it stays out of `state_dict`, so a
        # `fm.*` key missing is expected and anything else is not: a silent partial load is the
        # difference between scoring the trained policy and scoring a freshly initialised one.
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        real = [k for k in missing if not k.startswith(("fm.", "frozen"))]
        if real or unexpected:
            raise KeyError(
                f"adjuster weights do not fit the rebuilt actor: missing={real[:6]} "
                f"unexpected={list(unexpected)[:6]}. K is probably wrong (used {self._k}).")
        self.model.eval()
        self.iteration = int(payload.get("iter", -1))

        self._lift = torch.as_tensor(knot_basis(f, self._k, kind),
                                     dtype=torch.float32, device=env.device)
        self._scale = torch.tensor(
            residual_scale_vector(cmd.ctx.nj, cmd.ctx.target_num_extra_frames,
                                  cmd.ctx.reference_blocks), device=env.device)
        print(f"[INFO] AdjusterPolicy: {os.path.basename(checkpoint)} (iter {self.iteration}), "
              f"K={self._k} {kind}, action {self._k * d} -> residual {f}x{d}, DETERMINISTIC "
              "mean, applied on the re-plan step only.")

    @torch.no_grad()
    def apply(self) -> None:
        """Write the mean residual onto the plan, for the envs that just re-planned."""
        from tensordict import TensorDict                                   # noqa: PLC0415

        from robot_rl.tasks.manager_based.robot_rl import mdp               # noqa: PLC0415

        mask = self._cmd.fm_reinfer_mask
        if not bool(mask.any()):
            return
        obs = TensorDict({"fm_cond": mdp.fm_plan_conditioning(self._env, self._name),
                          "fm_plan": mdp.fm_plan_window(self._env, self._name)},
                         batch_size=[int(mask.shape[0])])
        residual = residual_from_action(self.model(obs), self._lift, self._scale)
        self._cmd.apply_plan_residual(residual * mask.view(-1, 1, 1).to(residual.dtype))
