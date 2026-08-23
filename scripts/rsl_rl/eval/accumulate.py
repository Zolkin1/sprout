# Copyright 2026 Zachary Olkin. All rights reserved.

"""Streaming per-episode metric accumulation.

RAM is ``O(num_envs x ~50 scalars)``, not ``O(episodes x steps)``: running sums live on the
device and are harvested when an episode ends.  ``scripts/rsl_rl/data_logger.py`` buffers
every step and is therefore unusable at tens of thousands of episodes -- but it is the
authoritative reference for HOW each quantity is read off the env, and the accessors here
follow it.

**The off-by-one that shapes this file.**  ``ManagerBasedRLEnv.step`` computes terminations,
then calls ``_reset_idx`` on the done envs, then returns.  So once ``env.step`` returns:

* ``termination_manager.dones`` / ``.time_outs`` / ``.get_term(n)`` still describe the episode
  that just ENDED -- they are buffers, not recomputed;
* but ``robot.data``, ``cmd.clip_id`` and ``episode_length_buf`` already describe the NEW
  episode for any env that reset.

So per-step folding must mask out the done envs, and anything that needs the terminal state
has to be read from inside a reset event -- which is what :class:`EvalResetHook` in
``runner.py`` is for.  ``steps_survived`` comes from this accumulator's own counter and never
from ``episode_length_buf``, which is already zeroed by then.
"""

from __future__ import annotations

import numpy as np
import torch

import schema

#: Name patterns that split the clip's output channels into comparable groups.  Same
#: patterns ``play_plots._print_group_summaries`` uses, so the eval's reference errors group
#: exactly like the existing stats do.
_JOINT_PREFIX = "joint:"
_POS_INFIX = ":pos_"
_ORI_INFIX = ":ori_"
#: Body-tracking links carry an ``@root`` suffix on the frame name.  The generator does not
#: predict them, so under ``strict_fm_only`` they arrive as NaN and must not poison a sum.
_ROOT_SUFFIX = "@root"


def _to_torch(data):
    """Convert a robot-data buffer to a torch tensor, handling warp arrays.

    ``ArticulationData`` hands back warp arrays for several buffers, so a bare ``.sum()`` or
    slice fails.  Same conversion ``scripts/rsl_rl/data_logger.py:_to_torch`` uses; replicated
    rather than imported because that module pulls in IsaacLab, and this one is unit-tested
    without it.
    """
    if isinstance(data, torch.Tensor):
        return data
    try:
        import warp as wp  # noqa: PLC0415

        return wp.to_torch(data)
    except (ImportError, TypeError):
        return torch.as_tensor(data)


class _Data:
    """Attribute view over ``robot.data`` that converts warp arrays on access."""

    def __init__(self, data) -> None:
        self._d = data

    def __getattr__(self, name):
        return _to_torch(getattr(self._d, name))


def channel_groups(pos_names: list[str], vel_names: list[str], ref_frame: str) -> dict[str, np.ndarray]:
    """Index arrays for each reference-error group, from the command's LIVE output names.

    Read off ``cmd.manager.pos_output_names`` rather than the library file, because
    ``order_outputs`` permutes them to the robot's own ordering at command construction.

    Args:
        pos_names: Position-channel names (includes ``ori_w``).
        vel_names: Velocity-channel names.
        ref_frame: The clip's reference frame, e.g. ``pelvis_link``.

    Returns:
        Group name -> integer index array into ``y_des`` / ``dy_des``.
    """
    base = ref_frame.strip('"').strip("'")

    def _idx(names, pred):
        return np.array([i for i, n in enumerate(names) if pred(n)], dtype=np.int64)

    is_root = lambda n: _ROOT_SUFFIX in n  # noqa: E731
    return {
        "base_pos": _idx(pos_names, lambda n: n.startswith(base + ":") and _POS_INFIX in n),
        "base_ori": _idx(pos_names, lambda n: n.startswith(base + ":") and _ORI_INFIX in n),
        "joint_pos": _idx(pos_names, lambda n: n.startswith(_JOINT_PREFIX)),
        "ee_pos": _idx(pos_names, lambda n: (_POS_INFIX in n and not is_root(n)
                                            and not n.startswith(base + ":"))),
        "base_lin_vel": _idx(vel_names, lambda n: n.startswith(base + ":") and _POS_INFIX in n),
        "base_ang_vel": _idx(vel_names, lambda n: n.startswith(base + ":") and _ORI_INFIX in n),
        "joint_vel": _idx(vel_names, lambda n: n.startswith(_JOINT_PREFIX)),
    }


def _quat_angle(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two quaternion batches, in radians.

    A component-wise MSE between quaternions is not an angle, so it is not reported as one.
    Sign-invariant, because ``q`` and ``-q`` are the same rotation.
    """
    dot = (q_a * q_b).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


class EpisodeAccumulator:
    """Running per-episode metrics for every env at once.

    Call :meth:`fold` once per step with the done envs masked out, and :meth:`harvest` from
    inside the clip reset for the envs that just ended.
    """

    def __init__(self, env, *, dt: float, fall_min_height: float = schema.FALL_MIN_HEIGHT_M,
                 fall_tilt_rad: float = schema.FALL_TILT_RAD) -> None:
        """Allocate the per-env buffers and resolve the channel groups once.

        Args:
            env: The wrapped env (``env.unwrapped`` is used throughout).
            dt: Control step, seconds.
            fall_min_height: Base height below which the robot counts as fallen.
            fall_tilt_rad: Tilt from upright above which the robot counts as fallen.
        """
        u = env.unwrapped
        self.dt = float(dt)
        self.n = int(u.num_envs)
        self.device = u.device
        self.fall_min_height = float(fall_min_height)
        self.fall_tilt_rad = float(fall_tilt_rad)

        self.cmd = u.command_manager.get_term("traj_ref")
        self.robot = u.scene.articulations["robot"]
        self.groups = channel_groups(
            list(self.cmd.manager.pos_output_names), list(self.cmd.manager.vel_output_names),
            str(self.cmd.manager.ref_frame))
        self._gidx = {k: torch.as_tensor(v, device=self.device, dtype=torch.long)
                      for k, v in self.groups.items() if v.size}
        self.rd = _Data(self.robot.data)
        self.mass_kg = (float(self.rd.default_mass[0].sum())
                        if getattr(self.robot.data, "default_mass", None) is not None
                        else float("nan"))
        limits = getattr(self.robot.data, "joint_effort_limits", None)
        self._effort = (_to_torch(limits)[0].clone().to(self.device)
                        if limits is not None
                        else torch.ones(self.robot.num_joints, device=self.device))
        self._effort = torch.where(self._effort > 0, self._effort,
                                   torch.ones_like(self._effort))
        self._foot_ids = self._resolve_feet(u)
        self._contact_sensors = [s for name, s in u.scene.sensors.items()
                                 if "foot" in name and hasattr(s.data, "net_forces_w")]

        z = lambda: torch.zeros(self.n, device=self.device)  # noqa: E731
        self.steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.vel_sq = torch.zeros(self.n, 3, device=self.device)
        self.vel_absmax = torch.zeros(self.n, 3, device=self.device)
        self.cmd_speed = z()
        self.ref_sq = {k: z() for k in self._gidx}
        self.ref_n = {k: z() for k in self._gidx}
        self.ori_sq = z()
        self.ori_max = z()
        self.finite_frac = z()
        self.clf_v = z()
        self.min_h = torch.full((self.n,), float("inf"), device=self.device)
        self.max_tilt = z()
        self.first_fall = torch.full((self.n,), -1, device=self.device, dtype=torch.long)
        self.act_rate = z()
        self.act_rate_max = z()
        self.torque_util = z()
        self.torque_sq = z()
        self.work = z()
        self.dist = z()
        self.slip = z()
        self.contact_steps = z()
        self._prev_action = torch.zeros(self.n, self.robot.num_joints, device=self.device)
        self._prev_xy = self.rd.root_pos_w[:, :2].clone()
        self._have_prev = torch.zeros(self.n, dtype=torch.bool, device=self.device)

    def _resolve_feet(self, u) -> torch.Tensor:
        """Body indices of the two ankle-roll links, for the foot-slip term."""
        names = list(self.robot.data.body_names)
        want = [i for i, n in enumerate(names) if n.endswith("ankle_roll_link")]
        return torch.as_tensor(want, device=self.device, dtype=torch.long)

    # ------------------------------------------------------------------
    def fold(self, mask: torch.Tensor) -> None:
        """Fold one step into the running sums, for the envs in ``mask``.

        ``mask`` must exclude envs that reset on this step: their ``robot.data`` already
        describes the new episode, so folding them would mix two episodes together.
        """
        # ``dones`` comes back from the wrapper as a Long tensor on some paths, and
        # ``torch.where`` requires a genuine bool condition, so normalise here rather than at
        # every call site.
        m = mask.to(self.device).to(torch.bool)
        mf = m.to(torch.float32)
        d = self.rd

        self.steps += m.long()

        # -- commanded-velocity tracking (the pairing data_logger uses) -------------------
        if getattr(self.cmd, "command_velocity_available", False):
            target = self.cmd.get_command_velocity()
            meas = torch.stack([d.root_lin_vel_b[:, 0], d.root_lin_vel_b[:, 1],
                                d.root_ang_vel_b[:, 2]], dim=-1)
            err = meas - target
            self.vel_sq += (err * err) * mf.unsqueeze(-1)
            self.vel_absmax = torch.maximum(self.vel_absmax,
                                            err.abs() * mf.unsqueeze(-1))
            self.cmd_speed += torch.linalg.vector_norm(target[:, :2], dim=-1) * mf

        # -- reference tracking, NaN-safe (strict_fm_only NaNs whole blocks) --------------
        y_des, y_act = self.cmd.y_des, self.cmd.y_act
        dy_des, dy_act = self.cmd.dy_des, self.cmd.dy_act
        finite_total = torch.zeros_like(mf)
        finite_count = torch.zeros_like(mf)
        for name, idx in self._gidx.items():
            src_d, src_a = ((dy_des, dy_act) if name.endswith("_vel")
                            else (y_des, y_act))
            diff = src_d[:, idx] - src_a[:, idx]
            good = torch.isfinite(diff)
            sq = torch.where(good, diff * diff, torch.zeros_like(diff)).sum(dim=-1)
            cnt = good.sum(dim=-1).to(torch.float32)
            self.ref_sq[name] += sq * mf
            self.ref_n[name] += cnt * mf
            finite_total += cnt
            finite_count += float(idx.numel())
        self.finite_frac += torch.where(finite_count > 0, finite_total / finite_count,
                                        torch.zeros_like(finite_count)) * mf

        # base orientation as an ANGLE
        if "base_ori" in self._gidx:
            oi = self._gidx["base_ori"]
            ang = _quat_angle(y_des[:, oi], y_act[:, oi])
            ang = torch.nan_to_num(ang, nan=0.0)
            self.ori_sq += (ang * ang) * mf
            self.ori_max = torch.maximum(self.ori_max, ang * mf)

        v = getattr(self.cmd, "v", None)
        if isinstance(v, torch.Tensor):
            self.clf_v += torch.nan_to_num(v.reshape(self.n), nan=0.0) * mf

        # -- robot state ------------------------------------------------------------------
        height = d.root_pos_w[:, 2]
        tilt = self._tilt(d)
        self.min_h = torch.where(m, torch.minimum(self.min_h, height), self.min_h)
        self.max_tilt = torch.maximum(self.max_tilt, tilt * mf)
        crossed = m & (self.first_fall < 0) & (
            (height < self.fall_min_height) | (tilt > self.fall_tilt_rad))
        self.first_fall = torch.where(crossed, self.steps, self.first_fall)

        # -- effort / smoothness ----------------------------------------------------------
        action = self._processed_action()
        if action is not None:
            rate = torch.linalg.vector_norm(action - self._prev_action, dim=-1)
            rate = torch.where(self._have_prev, rate, torch.zeros_like(rate))
            self.act_rate += rate * mf
            self.act_rate_max = torch.maximum(self.act_rate_max, rate * mf)
            self._prev_action = action.clone()
        tau = d.applied_torque
        self.torque_util = torch.maximum(
            self.torque_util, (tau.abs() / self._effort).amax(dim=-1) * mf)
        self.torque_sq += (tau * tau).sum(dim=-1) * mf
        self.work += (tau * d.joint_vel).abs().sum(dim=-1) * self.dt * mf

        xy = d.root_pos_w[:, :2]
        step_dist = torch.linalg.vector_norm(xy - self._prev_xy, dim=-1)
        step_dist = torch.where(self._have_prev, step_dist, torch.zeros_like(step_dist))
        self.dist += step_dist * mf
        self._prev_xy = xy.clone()
        self._have_prev = self._have_prev | m

        # -- foot slip --------------------------------------------------------------------
        if self._foot_ids.numel() and self._contact_sensors:
            contact = self._foot_contact()
            vel_xy = d.body_link_lin_vel_w[:, self._foot_ids, :2]
            tang = torch.linalg.vector_norm(vel_xy, dim=-1)
            self.slip += (tang * contact).sum(dim=-1) * self.dt * mf
            self.contact_steps += contact.sum(dim=-1) * mf

    def _tilt(self, d) -> torch.Tensor:
        """Angle between the robot's up-axis and world up, in radians.

        Derived from ``root_quat_w`` rather than ``projected_gravity_b``, and that is a
        CORRECTNESS fix, not a preference.  ``_reset_idx`` writes the new root state to sim,
        which refreshes the pose buffers, but ``projected_gravity_b`` lagged one step -- so
        the first fold of a new episode read the PREVIOUS episode's orientation.  Height,
        read straight off ``root_pos_w``, was always fresh, which is why the corruption
        appeared only in the tilt channel.

        Measured on the runs that found it (fm__none__lag48 / clip__none__-): episodes flagged
        as fallen that never fired the ``base_fallen`` termination were 635 and 64
        respectively, of which 98.7% and 100% had ``first_fall_step == 1``, ALL tripping on
        tilt and NONE on height, at a normal median height of 0.685 m and a median 333 steps
        survived.  The leak was a fixed fraction of the PRECEDING episode's fall rate --
        0.0962/0.513 = 0.188 for the generator arm against 0.0097/0.0533 = 0.182 for the clip
        arm -- which is what identified it as carry-over rather than a physical transient.

        The two agree to floating-point on any settled step; only the step after a reset
        differed.  ``cos(tilt)`` is the world-z component of the body's up-axis, which for a
        ``(w, x, y, z)`` quaternion is ``1 - 2 (x^2 + y^2)``.

        Args:
            d: The converting view over ``robot.data``.

        Returns:
            ``[num_envs]`` tilt angle in radians.
        """
        q = d.root_quat_w
        cos = (1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])).clamp(-1.0, 1.0)
        return torch.acos(cos)

    def _tilt_terminal(self, d) -> torch.Tensor:
        """Tilt for the TERMINAL fold, where the two buffers disagree on purpose.

        :meth:`fold_terminal` runs inside the clip-reset hook, and by then the buffers have
        come apart: ``scene.reset`` has already written the new spawn pose into
        ``root_pos_w`` / ``root_quat_w``, while ``projected_gravity_b`` still holds the
        orientation the episode ENDED at.  So at this one call site the lagging buffer is the
        correct one, and the fresh buffer describes an episode that has not started.

        Taking the MAXIMUM of the two is what makes this robust to the ordering rather than
        dependent on it: whichever buffer holds the terminal pose, a fall is beyond the
        threshold in that one, and the other reads an upright spawn that cannot mask it.  If
        IsaacLab ever refreshes both before the hook, this degrades to the fresh value and
        :meth:`EvalResetHook._check_terminal_state` reports ``post_reset`` rather than
        silently under-counting.

        The asymmetry with :meth:`_tilt` is the whole fix: per-step folds must use ONLY the
        fresh buffer, or the first fold of a new episode inherits the previous episode's
        orientation.

        Args:
            d: The converting view over ``robot.data``.

        Returns:
            ``[num_envs]`` tilt angle in radians.
        """
        grav = d.projected_gravity_b
        from_grav = torch.acos((-grav[:, 2]).clamp(-1.0, 1.0))
        return torch.maximum(self._tilt(d), from_grav)

    def _processed_action(self) -> torch.Tensor | None:
        """The action actually sent to the PD targets (``data_logger``'s accessor)."""
        try:
            term = self.robot._env.action_manager.get_term("joint_pos")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            term = None
        if term is None:
            return None
        val = getattr(term, "_processed_actions", None)
        return val if isinstance(val, torch.Tensor) else None

    def _foot_contact(self) -> torch.Tensor:
        """``[n, n_feet]`` 1.0 where the foot is loaded."""
        forces = []
        for sensor in self._contact_sensors[: self._foot_ids.numel()]:
            f = _to_torch(sensor.data.net_forces_w)
            mag = torch.linalg.vector_norm(f.reshape(self.n, -1, 3), dim=-1).amax(dim=-1)
            forces.append((mag > 1.0).to(torch.float32))
        if not forces:
            return torch.zeros(self.n, self._foot_ids.numel(), device=self.device)
        out = torch.stack(forces, dim=-1)
        if out.shape[-1] < self._foot_ids.numel():
            pad = torch.zeros(self.n, self._foot_ids.numel() - out.shape[-1],
                              device=self.device)
            out = torch.cat([out, pad], dim=-1)
        return out

    # ------------------------------------------------------------------
    def fold_terminal(self, env_ids: torch.Tensor) -> None:
        """Fold the terminal frame's height/tilt into the extremes for ``env_ids``.

        Called from the clip reset, which is the only place the state of the episode that
        just ended is still readable.  Only the extremes and the fall crossing are updated --
        the running SUMS deliberately are not, because their divisor is the folded step count
        and adding a step to one without the other would bias every mean.

        Without this the accumulator's ``~dones`` mask excludes exactly the frame that ended
        the episode, which for a fall is exactly the frame that crossed the threshold.
        """
        d = self.rd
        ids = env_ids.to(self.device)
        height = d.root_pos_w[:, 2][ids]
        # The TERMINAL variant: inside the reset hook the pose buffers already describe the
        # new spawn while projected_gravity_b still holds the ending orientation.
        tilt = self._tilt_terminal(d)[ids]
        self.min_h[ids] = torch.minimum(self.min_h[ids], height)
        self.max_tilt[ids] = torch.maximum(self.max_tilt[ids], tilt)
        crossed = (self.first_fall[ids] < 0) & (
            (height < self.fall_min_height) | (tilt > self.fall_tilt_rad))
        self.first_fall[ids] = torch.where(crossed, self.steps[ids], self.first_fall[ids])

    def snapshot(self, env_ids: torch.Tensor) -> dict[str, np.ndarray]:
        """Reduce the running sums for ``env_ids`` into per-episode values.

        Returns numpy arrays; one host transfer per quantity per done-batch, and none per
        step, so the hot loop never synchronises.
        """
        ids = env_ids.to(self.device)
        steps = self.steps[ids].clamp(min=1).to(torch.float32)
        cpu = lambda t: t.detach().to("cpu").numpy()  # noqa: E731

        out: dict[str, np.ndarray] = {
            "steps_survived": cpu(self.steps[ids]).astype(np.int64),
            "n_steps_scored": cpu(self.steps[ids]).astype(np.int64),
            "cmd_speed_mean": cpu(self.cmd_speed[ids] / steps),
            "ref_base_ori_mse_rad": cpu(self.ori_sq[ids] / steps),
            "ref_base_ori_max_rad": cpu(self.ori_max[ids]),
            "ref_finite_frac": cpu(self.finite_frac[ids] / steps),
            "clf_v_mean": cpu(self.clf_v[ids] / steps),
            "min_base_height_w": cpu(self.min_h[ids]),
            "max_tilt_rad": cpu(self.max_tilt[ids]),
            "first_fall_step": cpu(self.first_fall[ids]).astype(np.int64),
            "action_rate_mean": cpu(self.act_rate[ids] / steps),
            "action_rate_max": cpu(self.act_rate_max[ids]),
            "torque_util_max": cpu(self.torque_util[ids]),
            "torque_rms": cpu(torch.sqrt(self.torque_sq[ids] / steps)),
            "mech_work_j": cpu(self.work[ids]),
            "distance_m": cpu(self.dist[ids]),
            "foot_slip_dist_m": cpu(self.slip[ids]),
            "foot_contact_frac": cpu(self.contact_steps[ids]
                                     / (steps * max(1, int(self._foot_ids.numel())))),
        }
        vel = cpu(self.vel_sq[ids] / steps.unsqueeze(-1))
        vmax = cpu(self.vel_absmax[ids])
        for k, axis in enumerate(("vx", "vy", "wz")):
            out[f"vel_mse_{axis}"] = vel[:, k]
            out[f"vel_maxabs_{axis}"] = vmax[:, k]
        for name in ("base_pos", "joint_pos", "ee_pos", "base_lin_vel", "base_ang_vel",
                     "joint_vel"):
            col = f"ref_{name}_mse"
            if name in self.ref_sq:
                cnt = self.ref_n[name][ids]
                out[col] = cpu(torch.where(cnt > 0, self.ref_sq[name][ids] / cnt,
                                           torch.full_like(cnt, float("nan"))))
            else:
                out[col] = np.full(int(ids.numel()), np.nan, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = self.mass_kg * 9.81 * out["distance_m"]
            out["cost_of_transport"] = np.where(denom > 1e-6, out["mech_work_j"] / denom,
                                                np.nan)
        return out

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Zero every accumulator for ``env_ids``, ready for their next episode."""
        ids = env_ids.to(self.device)
        self.steps[ids] = 0
        self.vel_sq[ids] = 0.0
        self.vel_absmax[ids] = 0.0
        self.cmd_speed[ids] = 0.0
        for k in self.ref_sq:
            self.ref_sq[k][ids] = 0.0
            self.ref_n[k][ids] = 0.0
        self.ori_sq[ids] = 0.0
        self.ori_max[ids] = 0.0
        self.finite_frac[ids] = 0.0
        self.clf_v[ids] = 0.0
        self.min_h[ids] = float("inf")
        self.max_tilt[ids] = 0.0
        self.first_fall[ids] = -1
        self.act_rate[ids] = 0.0
        self.act_rate_max[ids] = 0.0
        self.torque_util[ids] = 0.0
        self.torque_sq[ids] = 0.0
        self.work[ids] = 0.0
        self.dist[ids] = 0.0
        self.slip[ids] = 0.0
        self.contact_steps[ids] = 0.0
        self._have_prev[ids] = False
