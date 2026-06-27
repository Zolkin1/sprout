# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-env, windowed perturbation scheduler for teacher data recording.

Used by ``scripts/rsl_rl/record_data.py`` to gather *recovery* data: run the
teacher with a clean fraction of envs (play-level DR only) and a perturbed
fraction where action-noise + a push impulse are toggled ON for a sampled
window then OFF, so the robot is driven off-reference and then recovers. Each
step exposes a per-env ``perturb_active`` mask the recorder logs, so the
flow-matching trainer can later condition on the bad-state history while
predicting the clean recovery (see ``flow_matching/dataset.py``).

Design (matches the approved plan):

* **Groups** -- each env is ``clean`` or ``perturbed`` (fraction
  ``1 - clean_fraction`` perturbed), re-rolled on reset so the mix is kept as
  episodes end. Clean envs are never touched.
* **Windows** -- each perturbed env runs an OFF/ON state machine (durations
  sampled from ``off_range_s`` / ``on_range_s``), starting OFF (a clean lead-in).
  On every OFF->ON edge it receives a push impulse; while ON its actions (and
  optionally its policy obs) are corrupted.
* **Per-episode DR** -- dynamics DR is *not* windowed. Optionally
  (``ood_dr=True``) the perturbed group's existing DR terms are re-applied with
  widened (out-of-distribution) ranges at each reset. The clean group keeps the
  base (play) DR baked in at env startup.

The window / group / noise logic is pure torch (no Isaac Lab), so it is
unit-testable; the push and DR calls are the only env-touching methods.
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import torch


class PerturbationScheduler:
    """Assign clean/perturbed groups and drive per-env perturbation windows.

    Args:
        num_envs: Number of parallel environments.
        dt: Control-step period (seconds), used to convert window durations to steps.
        device: Torch device for the per-env state tensors.
        clean_fraction: Fraction of envs that are clean (never perturbed).
        on_range_s: ``(min, max)`` seconds a perturbation window stays ON.
        off_range_s: ``(min, max)`` seconds between windows (recovery / clean).
        action_noise_scale: Std of the Gaussian noise added to actions while ON
            (in action units; ``0`` disables action noise).
        push_vel_range: ``{axis: (lo, hi)}`` root-velocity ranges (m/s, rad/s) for
            the push impulse applied on each OFF->ON edge (keys: x, y, z, roll,
            pitch, yaw). ``None`` disables pushes.
        ood_dr: If True, re-apply the perturbed group's DR terms with widened
            ranges at each reset (out-of-distribution dynamics).
        ood_dr_scale: Multiplicative widening of each DR range about its center
            (e.g. 1.5 = 50% wider) when ``ood_dr`` is set.
        dr_term_names: Event-term names to re-apply for OOD DR (skipped if absent
            or disabled in the env config).
        obs_noise_scale: Std of Gaussian noise added to the policy obs while ON
            (``0`` disables obs noise).
        seed: Seeds the internal RNG (group draw, window durations, noise).
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: torch.device | str,
        *,
        clean_fraction: float = 0.5,
        on_range_s: tuple[float, float] = (0.3, 1.0),
        off_range_s: tuple[float, float] = (1.5, 4.0),
        action_noise_scale: float = 0.0,
        push_vel_range: dict[str, tuple[float, float]] | None = None,
        ood_dr: bool = False,
        ood_dr_scale: float = 1.5,
        dr_term_names: Sequence[str] = (
            "joint_friction_params", "joint_armature_params", "gain_randomization",
            "add_base_mass", "base_com", "randomize_ground_contact_friction",
        ),
        obs_noise_scale: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.clean_fraction = float(clean_fraction)
        self.on_range_s = (float(on_range_s[0]), float(on_range_s[1]))
        self.off_range_s = (float(off_range_s[0]), float(off_range_s[1]))
        self.action_noise_scale = float(action_noise_scale)
        self.push_vel_range = push_vel_range
        self.ood_dr = bool(ood_dr)
        self.ood_dr_scale = float(ood_dr_scale)
        self.dr_term_names = list(dr_term_names)
        self.obs_noise_scale = float(obs_noise_scale)

        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(int(seed))

        # Per-env state (CPU bookkeeping; masks moved to device on demand).
        self.is_perturbed = torch.zeros(self.num_envs, dtype=torch.bool)  # group
        self.window_on = torch.zeros(self.num_envs, dtype=torch.bool)     # ON vs OFF
        self.steps_left = torch.zeros(self.num_envs, dtype=torch.int64)   # in current state
        # Warn once if pushes were requested but no env was wired to apply them.
        self._push_warned = False

    # ------------------------------------------------------------------
    # Pure window / group logic (no Isaac Lab)
    # ------------------------------------------------------------------

    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, generator=self._gen)

    def _sample_steps(self, n: int, rng_s: tuple[float, float]) -> torch.Tensor:
        """Sample ``n`` window durations (steps), uniform in ``rng_s`` seconds."""
        lo, hi = rng_s
        secs = lo + (hi - lo) * self._rand(n)
        return torch.clamp((secs / self.dt).round().to(torch.int64), min=1)

    def assign_groups(self, env_ids: torch.Tensor) -> None:
        """(Re)assign clean/perturbed groups and reset windows for ``env_ids``.

        Uses an *exact* split: exactly ``round((1 - clean_fraction) * k)`` of the
        ``k`` env_ids are flagged perturbed (chosen at random), so a full
        ``clean_fraction=0.5`` assignment over N envs yields exactly N/2 perturbed
        rather than a noisy Bernoulli draw.
        """
        idx = env_ids.detach().cpu()
        k = idx.numel()
        if k == 0:
            return
        n_perturbed = int(round((1.0 - self.clean_fraction) * k))
        perm = torch.randperm(k, generator=self._gen)
        perturbed = torch.zeros(k, dtype=torch.bool)
        perturbed[perm[:n_perturbed]] = True
        self.is_perturbed[idx] = perturbed
        # All start OFF (a clean lead-in); first ON edge fires after off-duration.
        self.window_on[idx] = False
        self.steps_left[idx] = self._sample_steps(k, self.off_range_s)

    def step_windows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one step; flip expired windows.

        Returns:
            ``(active_mask, newly_on_ids)`` -- ``active_mask`` is ``[num_envs]``
            bool (perturbed envs currently ON) on ``self.device``; ``newly_on_ids``
            are env indices that flipped OFF->ON this step (need a push), on
            ``self.device``.
        """
        pert = self.is_perturbed
        self.steps_left[pert] -= 1
        expired = pert & (self.steps_left <= 0)
        newly_on = torch.zeros(self.num_envs, dtype=torch.bool)
        if expired.any():
            eids = expired.nonzero(as_tuple=False).flatten()
            new_state = ~self.window_on[eids]              # flip
            self.window_on[eids] = new_state
            # Sample the new duration from the bucket for the new state.
            durs = torch.empty(eids.numel(), dtype=torch.int64)
            on_mask = new_state
            if on_mask.any():
                durs[on_mask] = self._sample_steps(int(on_mask.sum()), self.on_range_s)
            if (~on_mask).any():
                durs[~on_mask] = self._sample_steps(int((~on_mask).sum()), self.off_range_s)
            self.steps_left[eids] = durs
            newly_on[eids] = new_state
        active = (pert & self.window_on).to(self.device)
        return active, newly_on.nonzero(as_tuple=False).flatten().to(self.device)

    def corrupt_actions(self, actions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Return ``actions`` with Gaussian noise added on rows where ``active``."""
        if self.action_noise_scale <= 0.0 or not bool(active.any()):
            return actions
        out = actions.clone()
        noise = torch.randn(out.shape, generator=self._gen).to(out.device) * self.action_noise_scale
        out[active] = out[active] + noise[active]
        return out

    def corrupt_obs(self, obs_tensor: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Return ``obs_tensor`` with Gaussian noise added on rows where ``active``."""
        if self.obs_noise_scale <= 0.0 or not bool(active.any()):
            return obs_tensor
        out = obs_tensor.clone()
        noise = torch.randn(out.shape, generator=self._gen).to(out.device) * self.obs_noise_scale
        out[active] = out[active] + noise[active]
        return out

    # ------------------------------------------------------------------
    # Env-touching methods (push + per-episode OOD DR)
    # ------------------------------------------------------------------

    def step(self, env) -> torch.Tensor:
        """Advance windows and apply pushes for OFF->ON edges; return ``active_mask``."""
        active, newly_on = self.step_windows()
        if newly_on.numel() > 0:
            self._apply_push(env, newly_on)
        return active

    def reset(self, env, env_ids: torch.Tensor) -> None:
        """Re-roll groups/windows for ``env_ids`` and (optionally) re-apply OOD DR."""
        self.assign_groups(env_ids)
        if self.ood_dr:
            perturbed_ids = env_ids[self.is_perturbed[env_ids.detach().cpu()].to(env_ids.device)]
            self._apply_ood_dr(env, perturbed_ids)

    def _apply_push(self, env, env_ids: torch.Tensor) -> None:
        """Apply a push impulse (root-velocity jump) to ``env_ids``."""
        if self.push_vel_range is None or env_ids.numel() == 0:
            return
        try:
            from isaaclab.envs.mdp import push_by_setting_velocity
        except ImportError:
            if not self._push_warned:
                print("[PerturbationScheduler] WARN: could not import push_by_setting_velocity; "
                      "pushes disabled.")
                self._push_warned = True
            return
        push_by_setting_velocity(env.unwrapped, env_ids, velocity_range=self.push_vel_range)

    @staticmethod
    def _widen(value: Any, scale: float) -> Any:
        """Widen a numeric ``(lo, hi)`` range about its center by ``scale`` (recursive)."""
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in value
        ):
            lo, hi = float(value[0]), float(value[1])
            c = 0.5 * (lo + hi)
            return (c - scale * (c - lo), c + scale * (hi - c))
        if isinstance(value, dict):
            return {k: PerturbationScheduler._widen(v, scale) for k, v in value.items()}
        return value

    def _apply_ood_dr(self, env, env_ids: torch.Tensor) -> None:
        """Re-apply each configured DR term to ``env_ids`` with widened ranges."""
        if env_ids.numel() == 0:
            return
        event_manager = env.unwrapped.event_manager
        for name in self.dr_term_names:
            try:
                cfg = event_manager.get_term_cfg(name)
            except (ValueError, AttributeError):
                continue  # term not present in this env config (e.g. disabled in play)
            func = getattr(cfg, "func", None)
            if func is None:
                continue
            # Class-based terms are instantiated lazily by the EventManager; resolve here.
            if inspect.isclass(func):
                func = func(cfg=cfg, env=env.unwrapped)
                cfg.func = func
            params = {k: self._widen(v, self.ood_dr_scale) for k, v in cfg.params.items()}
            try:
                func(env.unwrapped, env_ids, **params)
            except Exception as exc:  # noqa: BLE001 - DR is best-effort; never crash recording
                print(f"[PerturbationScheduler] WARN: OOD DR term '{name}' failed: {exc}")

    def summary(self) -> str:
        """One-line description of the configured perturbation regime."""
        n_pert = int(self.is_perturbed.sum())
        return (
            f"PerturbationScheduler: {n_pert}/{self.num_envs} perturbed "
            f"(clean_fraction={self.clean_fraction}); on={self.on_range_s}s off={self.off_range_s}s; "
            f"action_noise={self.action_noise_scale} obs_noise={self.obs_noise_scale}; "
            f"push={self.push_vel_range}; ood_dr={self.ood_dr} (x{self.ood_dr_scale})"
        )
