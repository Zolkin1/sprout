# Copyright 2026 Zachary Olkin. All rights reserved.

"""A trained tracker, loaded frozen and driven from inside the environment.

The adjuster's action is a residual on the generator's plan, so ``env.num_actions`` is that
residual and the policy that actually commands the joints has to live inside the env. From the
adjuster's point of view the tracker is part of the robot's dynamics: it is not trained, it is
not observed, and none of its rewards carry over.

**Not loaded through ``OnPolicyRunner``.** That would build a critic against the *adjuster's*
critic group, size a ``RolloutStorage`` for the adjuster's action width, and put the tracker's
weights inside the object ``train.py`` is about to train and checkpoint. This builds the actor
alone, from the run's own ``params/agent.yaml``, so the class and the hyperparameters are the
ones the run was trained with rather than a restatement of them.

**It must not be an attribute of any ``nn.Module`` in the algorithm.** ``PPO`` chains
``actor.parameters()`` with ``critic.parameters()`` for its single optimizer, clips gradients
over both, and saves both state dicts; a tracker reachable from either would be added to every
checkpoint and demanded back by ``load(strict=True)``. Held by the env, which is not a Module,
it cannot be reached by any of them. ``requires_grad_(False)`` alone would not be enough --
Adam does skip ``None``-grad parameters, but the weights would still be *in* the checkpoint,
and a resume with a different tracker would silently restore the wrong ones.

## The observation contract is asserted from three oracles

A tracker whose expected observation width does not match the env's loads clean when the
widths happen to agree and merely steers wrong when they do not, so this checks rather than
assumes -- and deliberately does not use ``exported/policy_parameters.yaml``, which for the
shipped tracker reports ``num_obs: 2490`` and omits the 611-wide ``reference`` group entirely,
disagreeing with the ``tracker.onnx`` beside it (``[N, 3101]``).

1. **The checkpoint's own weights**, always available and self-consistent by construction:
   ``encoder.0.weight.shape[1]`` is the encoder groups' total width, ``mlp.0.weight.shape[1]``
   minus the encoder latent is the direct groups' width, and
   ``distribution.log_std_param.shape[0]`` is the action width. Without this the only symptom
   of a mismatch is ``load_state_dict``'s shape complaint, which names a tensor and no
   observation term.
2. **The generator's recorded term table**, when one is supplied. The generator was recorded
   *through* this tracker, so ``fm_meta`` carries the per-term names and widths -- which is
   what turns a width error into *"observation term 'depth_far' (780) is missing from group
   'policy'"*.
3. **The recorded checkpoint hash**, so a tracker that is not the one the generator was
   recorded against is refused by identity rather than by coincidence of width.
"""

from __future__ import annotations

import hashlib
import os

import torch
import yaml
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict


def _sha256(path: str) -> str:
    """Hex digest of a file, read in chunks so a 100 MB checkpoint is not held twice."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FrozenTracker:
    """A trained tracker actor, in eval mode, with gradients off.

    Attributes:
        run_dir: The run the weights and the architecture came from.
        checkpoint_path: The loaded checkpoint.
        obs_groups: The observation groups the actor reads, in the order it concatenates them.
        action_dim: Width of the joint action it emits.
    """

    def __init__(
        self,
        env,
        run_dir: str,
        checkpoint: str = "model_19999.pt",
        fm_meta: dict | None = None,
        allow_mismatch: bool = False,
    ) -> None:
        """Load the actor and assert it fits this environment.

        Args:
            env: The (unwrapped) environment, for the observation manager and the device.
            run_dir: A ``logs/rsl_rl/<experiment>/<run>`` directory holding ``params/agent.yaml``.
            checkpoint: Checkpoint file name inside ``run_dir``.
            fm_meta: The generator checkpoint's ``meta`` dict, when available. Supplies the
                per-term contract and the recorded tracker hash for oracles 2 and 3; omitting
                it leaves only the weight-derived widths.
            allow_mismatch: Downgrade every contract failure to a printed warning. For a
                deliberate cross-pair experiment; it prints loudly because a mismatched pair
                is the failure this class exists to prevent.

        Raises:
            FileNotFoundError: If the run directory, its ``params/agent.yaml`` or the
                checkpoint is missing.
            ValueError: On a contract failure, unless ``allow_mismatch``.
        """
        self.run_dir = os.path.abspath(run_dir)
        self.checkpoint_path = os.path.join(self.run_dir, checkpoint)
        cfg_path = os.path.join(self.run_dir, "params", "agent.yaml")
        for path in (self.run_dir, cfg_path, self.checkpoint_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FrozenTracker needs {path}; run_dir must be a completed training run "
                    "carrying params/agent.yaml (the architecture) beside its checkpoints."
                )

        with open(cfg_path) as f:
            agent_cfg = yaml.unsafe_load(f)
        actor_cfg = dict(agent_cfg["actor"])
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        self.obs_groups: list[str] = list(agent_cfg["obs_groups"]["actor"])
        self._encoder_groups: list[str] = list(actor_cfg.get("encoder_obs_groups", []))

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["actor_state_dict"]
        self.action_dim = int(state["distribution.log_std_param"].shape[0])
        self._check_contract(env, state, actor_cfg, fm_meta, allow_mismatch)

        device = env.device
        probe = TensorDict(
            {g: torch.zeros(env.num_envs, int(env.observation_manager.group_obs_dim[g][0]),
                            device=device)
             for g in self.obs_groups},
            batch_size=[env.num_envs],
        )
        model = actor_class(
            probe, {"actor": self.obs_groups}, "actor", self.action_dim, **actor_cfg
        )
        model.load_state_dict(state, strict=True)
        self._model = model.to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        print(f"[INFO] FrozenTracker: {os.path.basename(self.run_dir)}/{checkpoint}, "
              f"{actor_class.__name__}, groups {self.obs_groups}, "
              f"{self.action_dim} joints, {sum(p.numel() for p in self._model.parameters())} "
              "frozen parameters")

    def _check_contract(
        self, env, state: dict, actor_cfg: dict, fm_meta: dict | None, allow_mismatch: bool
    ) -> None:
        """Assert the three oracles agree with this env; see the module docstring.

        Args:
            env: The environment.
            state: The checkpoint's ``actor_state_dict``.
            actor_cfg: The architecture cfg, minus ``class_name``.
            fm_meta: The generator's recorded meta, or ``None``.
            allow_mismatch: Warn instead of raising.

        Raises:
            ValueError: On any disagreement, unless ``allow_mismatch``.
        """
        problems: list[str] = []
        dims = {g: int(env.observation_manager.group_obs_dim[g][0]) for g in self.obs_groups}

        # -- oracle 1: widths implied by the weights themselves ---------------------------
        latent = int(actor_cfg.get("encoder_dims", [0, 0])[-1])
        want_encoder = int(state["encoder.0.weight"].shape[1]) if "encoder.0.weight" in state else 0
        want_direct = int(state["mlp.0.weight"].shape[1]) - latent
        have_encoder = sum(dims[g] for g in self.obs_groups if g in self._encoder_groups)
        have_direct = sum(dims[g] for g in self.obs_groups if g not in self._encoder_groups)
        if want_encoder != have_encoder:
            problems.append(
                f"encoder groups {self._encoder_groups} are {have_encoder} wide in this env "
                f"but the checkpoint's encoder expects {want_encoder}"
            )
        if want_direct != have_direct:
            direct = [g for g in self.obs_groups if g not in self._encoder_groups]
            problems.append(
                f"direct groups {direct} are {have_direct} wide in this env but the "
                f"checkpoint's trunk expects {want_direct} "
                f"(mlp.0 {int(state['mlp.0.weight'].shape[1])} - encoder latent {latent})"
            )
        if self.action_dim != int(env.action_manager.total_action_dim):
            problems.append(
                f"the checkpoint emits {self.action_dim} joint targets but this env's action "
                f"manager holds {int(env.action_manager.total_action_dim)}"
            )

        # -- oracle 2: the generator's recorded per-TERM table ----------------------------
        # The one that names a term rather than a tensor. The recording was made through this
        # tracker's own policy group, so a term added or dropped since shows up by name.
        names = list((fm_meta or {}).get("obs_term_names") or [])
        widths = list((fm_meta or {}).get("obs_term_dims") or [])
        if names and widths and len(names) == len(widths):
            live = dict(zip(env.observation_manager.active_terms["policy"],
                            [int(td[0]) for td in
                             env.observation_manager.group_obs_term_dim["policy"]]))
            for name, width in zip(names, widths):
                if name not in live:
                    problems.append(
                        f"observation term {name!r} ({int(width)}) is missing from group "
                        "'policy'; the generator this pair was recorded with conditions on it"
                    )
                elif live[name] != int(width):
                    problems.append(
                        f"observation term {name!r} is {live[name]} wide here but "
                        f"{int(width)} in the recording the generator was trained on"
                    )
            for name in live:
                if name not in names:
                    problems.append(
                        f"observation term {name!r} ({live[name]}) is in group 'policy' but "
                        "was not in the recording; the tracker has never seen it"
                    )

        # -- oracle 3: identity, not width ------------------------------------------------
        recorded = str((fm_meta or {}).get("checkpoint_sha256") or "")
        if recorded:
            actual = _sha256(self.checkpoint_path)
            if actual != recorded:
                problems.append(
                    f"this checkpoint hashes {actual[:12]} but the generator was recorded "
                    f"against {recorded[:12]} -- a different tracker, of the same width"
                )

        if not problems:
            return
        detail = "\n  - ".join(problems)
        message = f"FrozenTracker contract failed against this environment:\n  - {detail}"
        if allow_mismatch:
            print(f"[WARN] {message}\n[WARN] continuing because allow_mismatch=True; every "
                  "number from this run describes a pair that was never trained together.")
            return
        raise ValueError(message)

    @property
    def model(self) -> torch.nn.Module:
        """The loaded actor. Read-only by convention; it is frozen and in eval mode."""
        return self._model

    @torch.no_grad()
    def __call__(self, obs) -> torch.Tensor:
        """The DETERMINISTIC joint action for one control step.

        Deterministic because that is what ``play.py`` and the deploy runtime serve:
        ``MLPModel.forward`` defaults ``stochastic_output=False`` and returns the
        distribution's mean. Sampling here would add exploration to a policy that is not being
        trained, and would make the adjuster's credit assignment noisier for nothing.

        Args:
            obs: The env's observation mapping. Only :attr:`obs_groups` are read, in the order
                the actor concatenates them, so extra groups are ignored rather than
                mis-assembled.

        Returns:
            ``[num_envs, action_dim]``.
        """
        batch = obs[self.obs_groups[0]].shape[0]
        td = TensorDict({g: obs[g] for g in self.obs_groups}, batch_size=[batch])
        return self._model(td)
