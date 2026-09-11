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

# Torch twin of ``transfer.deploy.tracking_error``, held to it by
# ``transfer/deploy/tests/test_tracking_error_parity.py``. Safe at module scope: it imports
# only torch and ``isaaclab.utils.math``, and pulls no ``isaacsim``/``omni``/``pxr``/``carb``
# -- unlike ``mdp.goal_region``, which is attached after AppLauncher for exactly that reason
# (see ``test_no_pre_app_pxr_import.py``).
from robot_rl.tasks.manager_based.robot_rl.mdp import tracking_error

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
        # Per-episode tally of what KIND of motion each re-plan produced. Stays all-zero
        # unless a classifier is attached (the clip arm loads no generator).
        self.plan_counts = torch.zeros(self.n, 3, device=self.device, dtype=torch.long)
        self.plan_clf = None
        #: Optional :class:`FlaggedPlanDump`. Off unless the harness asks for it.
        self.plan_dump = None
        #: Running |served - clip| since each env's last re-plan, for the dump's selection.
        self._div_sum = torch.zeros(self.n, device=self.device)
        self._div_n = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self._arange = torch.arange(self.n, device=self.device)
        self.act_rate = z()
        self.act_rate_max = z()
        self.torque_util = z()
        self.torque_sq = z()
        self.work = z()
        self.dist = z()
        self.slip = z()
        self.contact_steps = z()
        self._prev_action = torch.zeros(self.n, self.robot.num_joints, device=self.device)
        # -- the SPATIAL goal ------------------------------------------------------------
        # Every other outcome above is a clip-CURSOR condition and cannot tell a robot that
        # climbed the staircase from one that stood still while the cursor ran out. These
        # are the buffers that can.
        #
        # The ANCHOR is episode-constant by construction (the globally placed clip's own
        # terminal base pose), so it is resolved once per reset in `refresh_goal_anchor`
        # rather than gathered every step -- cheaper, and it makes the constancy explicit
        # instead of incidental.
        # `goal_region` sits behind the `mdp` package __init__, which imports
        # `isaaclab.envs.mdp`; `runner.py` imports THIS module before AppLauncher runs, so a
        # module-level import here would bind pip's usd-core ahead of Kit's and segfault.
        # See `test_no_pre_app_pxr_import.py`. Attached instead, after the env exists.
        self._gr = None
        self.goal_spec = None
        self._goal_pos_w = torch.zeros(self.n, 3, device=self.device)
        self._goal_yaw = torch.zeros(self.n, device=self.device)
        self._goal_valid = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self._goal_tol = torch.zeros(self.n, 3, device=self.device)
        self.goal_reached = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.first_goal = torch.full((self.n,), -1, device=self.device, dtype=torch.long)
        self.min_goal_dist = torch.full((self.n,), float("inf"), device=self.device)
        self.max_goal_down = torch.full((self.n,), -float("inf"), device=self.device)
        self.end_goal = torch.full((self.n, 3), float("nan"), device=self.device)
        self._prev_xy = self.rd.root_pos_w[:, :2].clone()
        self._have_prev = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        # -- the STRICT tracking checks --------------------------------------------------
        # Off until `attach_tracking`, on the same terms as the goal region: with no spec the
        # columns come out "never violated" / NaN and a reader must be able to tell that from
        # "measured and never fired".
        self._track_spec = None
        self._track_offsets = torch.zeros(1, dtype=torch.long, device=self.device)
        self.first_violation = torch.full((self.n,), -1, device=self.device, dtype=torch.long)
        self.first_check = {}
        self.track_reward_sum = torch.zeros(self.n, device=self.device)
        self.track_reward_n = torch.zeros(self.n, device=self.device)
        #: The error row this step, ``[N, n]``, or None before the first served reference.
        #: Stashed rather than recomputed by the recorder: it is built here from the served
        #: window at the fold instant, and rebuilding it a step later would measure a
        #: different pairing.
        self.last_track_err: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # The STRICT tracking checks.  See transfer/deploy/tracking_error.py for the contract.
    # ------------------------------------------------------------------
    def attach_tracking(self, spec) -> None:
        """Turn on the strict tracking columns, with ``spec``'s bounds and kernel widths.

        Off until called, like :meth:`attach_goal_region` and for the same reason: a run
        without it reports ``first_violation_step == -1`` everywhere, and a reader has to be
        able to tell that from "measured, never fired".  ``report.py`` refuses to derive
        strict success from a table whose column is absent rather than reading it as "nothing
        ever went wrong".

        The reference is the SERVED one -- ``get_future_world_base`` / ``get_future_joint_state``
        at offset 0 answer for whichever arm is applied, so the clip arm and the generator arm
        go through one call site and cannot drift apart.

        Args:
            spec: A :class:`~robot_rl...mdp.tracking_error.TrackingErrorSpec`.

        Raises:
            ValueError: If the served generator predicts no ``base_vel`` or no ``joint_pos``.
                Checked HERE rather than per step: the accessors raise on that checkpoint, and
                a run that dies mid-episode after twenty minutes of rollout costs more than
                one that refuses at setup.
        """
        blocks = getattr(self.cmd, "_fm_blocks", None)
        if blocks is not None:
            missing = {"base_vel", "joint_pos"} - set(blocks)
            if missing:
                raise ValueError(
                    f"the strict tracking checks need {sorted(missing)} from the served "
                    f"reference, and this generator's window packs {list(blocks)}. There is "
                    "no useful reduced form: without base_vel the twist term is undefined, "
                    "and without joint_pos the reward is a base-only number wearing the name "
                    "of a whole-body one.")
        self._track_spec = spec
        self.first_check = {
            name: torch.full((self.n,), -1, device=self.device, dtype=torch.long)
            for name in tracking_error.VIOLATION_CHECK_NAMES}

    def _fold_tracking(self, m: torch.Tensor) -> None:
        """Latch the strict checks and accumulate the tracking reward, for the envs in ``m``.

        Called from :meth:`fold` and from :meth:`fold_terminal`, on the same terms as the fall
        crossing: the frame that ended the episode is exactly the one most likely to have
        crossed, so omitting it would leave a violated episode reading clean.

        Args:
            m: Boolean ``[N]`` mask of envs to fold.
        """
        # NO WARM-UP EXCLUSION HERE, and the asymmetry with MuJoCo is deliberate.
        # `mjeval.tracking.TrackingProbe` skips the opening command hold, which is a MuJoCo-only
        # artifact: its courses spawn a standing robot 2.5 m out and then command full speed,
        # and nothing on this side has a `cmd_hold_s` at all.
        #
        # Isaac DOES have the analogous mechanism -- the generator's history ring is empty at
        # reset, so its first plans are poor. Measured on a 74-episode rollout, mean chunk
        # reward by re-plan index: 0.40 (1), 0.48 (2), 0.50 (3), 0.52 (4-5), 0.55 (6-10),
        # 0.64 (21-40). It is KEPT ON PURPOSE, and "the ring is empty" is the reason to keep
        # it rather than a reason to filter it: the generator has a LEARNED null-history
        # embedding trained for exactly those steps, so they are a regime the model was fitted
        # for and not a degenerate warm-up. The return from an early state genuinely includes
        # those chunks, the critic conditions on the state (history_null included), and the
        # steps at the start of an episode are real data about real behaviour.
        #
        # What it does mean is that `track_reward_mean` is NOT symmetric between the two
        # harnesses -- this one carries the ramp, mjeval's does not -- so the two columns are
        # not directly comparable, and this one carries the same short-episode bias mjeval's
        # hold exclusion exists to remove. Compare within a harness, never across.
        if self._track_spec is None:
            return
        # No served reference yet -- before the generator's first plan there is nothing to
        # measure, which is a real state rather than an error. `_pred_valid` is absent on the
        # clip command, where a reference always exists.
        valid = getattr(self.cmd, "_pred_valid", None)
        if valid is not None:
            m = m & valid.to(self.device).reshape(self.n)
        if not bool(m.any()):
            self.last_track_err = None
            return

        pos_w, quat_w, lin_w, ang_w = self.cmd.get_future_world_base(self._track_offsets)
        jp, _ = self.cmd.get_future_joint_state(self._track_offsets)
        d = self.rd
        # `root_link_*`, NOT `root_lin_vel_w`: the latter is the CoM twist, and the MuJoCo
        # half measures the frame-ORIGIN twist via `mj_jacBody` (`Robot.get_base_world_velocity`
        # documents the distinction and warns against `mj_objectVelocity`). Mixing the two
        # would put a systematic offset on the velocity term in one simulator only.
        err = tracking_error.tracking_error(
            pos_w[:, 0], quat_w[:, 0], lin_w[:, 0], ang_w[:, 0], jp[:, 0],
            d.root_pos_w, d.root_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w,
            d.joint_pos)

        self.last_track_err = err
        mf = m.to(torch.float32)
        self.track_reward_sum += tracking_error.step_reward(err, self._track_spec) * mf
        self.track_reward_n += mf

        checks = tracking_error.violation_checks(err, self._track_spec)
        for name, hit in checks.items():
            crossed = m & (self.first_check[name] < 0) & hit
            self.first_check[name] = torch.where(crossed, self.steps, self.first_check[name])
        any_hit = torch.zeros_like(m)
        for hit in checks.values():
            any_hit = any_hit | hit
        crossed = m & (self.first_violation < 0) & any_hit
        self.first_violation = torch.where(crossed, self.steps, self.first_violation)

    # ------------------------------------------------------------------
    # The SPATIAL goal.  See mdp/goal_region.py for why it has to exist.
    # ------------------------------------------------------------------
    def attach_goal_region(self, spec, family_of_env=None) -> None:
        """Turn on the goal columns, with ``spec``'s tolerances.

        Off until called: with no spec the goal columns come out "never reached" / NaN and
        ``report.py`` REFUSES to derive spatial success from them.  That is deliberate --
        a spatial success rule silently falling back to a temporal one would be the exact
        defect this whole module exists to remove, and "absent" has to be distinguishable
        from "never arrived".

        Args:
            spec: A :class:`mdp.goal_region.GoalRegionSpec`.
            family_of_env: Optional ``[n]`` sequence of normalised family names, so each env
                gets its own family's calibrated tolerances.  ``None`` gives every env the
                spec's default.
        """
        from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region

        self._gr = goal_region
        self.goal_spec = spec
        self.set_goal_families(family_of_env)

    def set_goal_families(self, family_of_env=None, env_ids: torch.Tensor | None = None) -> None:
        """Write per-env tolerances from each env's family.

        Called again whenever an env is re-pinned to a clip of a different family -- the
        tolerances are per family, and a batch of 1024 envs spans several.

        Args:
            family_of_env: Sequence of normalised family names, one per entry of ``env_ids``
                (or per env when ``env_ids`` is None).  ``None`` uses the spec default.
            env_ids: Optional subset to write.
        """
        if self.goal_spec is None:
            return
        ids = (torch.arange(self.n, device=self.device) if env_ids is None
               else env_ids.to(self.device))
        if family_of_env is None:
            t = self.goal_spec.default
            row = torch.tensor([t.approach_margin_m, t.half_width_m, t.height_tol_m],
                               device=self.device, dtype=self._goal_tol.dtype)
            self._goal_tol[ids] = row
            return
        rows = [self.goal_spec.for_family(f) for f in family_of_env]
        self._goal_tol[ids] = torch.tensor(
            [[r.approach_margin_m, r.half_width_m, r.height_tol_m] for r in rows],
            device=self.device, dtype=self._goal_tol.dtype)

    def refresh_goal_anchor(self, env_ids: torch.Tensor | None = None) -> None:
        """Resolve where the clip itself finishes, for ``env_ids``, once per episode.

        The anchor is episode-CONSTANT -- it is the globally placed clip's own base pose at
        ``last_usable_step``, and neither the placement nor that step moves once the episode
        has started.  Resolving it per reset rather than per step is therefore not just
        cheaper (it is a library gather over every env), it states that constancy instead of
        leaving it incidental.

        Must be called AFTER the reset has committed the new clip and its placement, or the
        anchor describes the previous episode's clip -- the same shape of carry-over bug that
        made ``projected_gravity_b`` inflate the derived fall flag.

        Args:
            env_ids: Envs to refresh, or ``None`` for all.
        """
        if self._gr is None:
            return
        pos, yaw = self._gr.goal_anchor_world(self.cmd)
        ids = (torch.arange(self.n, device=self.device) if env_ids is None
               else env_ids.to(self.device))
        self._goal_pos_w[ids] = pos[ids].to(self._goal_pos_w.dtype)
        self._goal_yaw[ids] = yaw[ids].to(self._goal_yaw.dtype)
        self._goal_valid[ids] = True

    def _fold_goal(self, m: torch.Tensor) -> None:
        """Fold this step's goal geometry for the envs in ``m``."""
        if self._gr is None:
            return
        down, lat, hz = self._gr.goal_frame_offsets(
            self.rd.root_pos_w, self._goal_pos_w, self._goal_yaw)
        live = m & self._goal_valid
        inside = self._gr.in_goal(down, lat, hz, self._goal_tol) & live
        dist = self._gr.goal_distance_m(down, lat, hz, self._goal_tol)

        newly = inside & ~self.goal_reached
        self.first_goal = torch.where(newly, self.steps, self.first_goal)
        self.goal_reached = self.goal_reached | inside
        self.min_goal_dist = torch.where(live, torch.minimum(self.min_goal_dist, dist),
                                         self.min_goal_dist)
        self.max_goal_down = torch.where(live, torch.maximum(self.max_goal_down, down),
                                         self.max_goal_down)
        # The terminal offsets are simply the LAST folded ones, so no separate terminal hook
        # is needed: `fold_terminal` folds the ending frame through this same path.
        self.end_goal = torch.where(live.unsqueeze(-1),
                                    torch.stack([down, lat, hz], dim=-1), self.end_goal)

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

        # -- the strict tracking checks, on the same mask and the same clock --------------
        self._fold_tracking(m)

        # -- what kind of motion did the generator just plan? -----------------------------
        # Gated on a FRESH plan (`_pred_idx == 0`, set on re-plan and advanced one node per
        # step), so each plan is counted exactly once per env rather than 12 times.
        if self.plan_clf is not None:
            idx = getattr(self.cmd, "_pred_idx", None)
            win = getattr(self.cmd, "_pred_window", None)
            valid = getattr(self.cmd, "_pred_valid", None)
            if idx is not None and win is not None:
                fresh = m & (idx == 0)
                if valid is not None:
                    fresh = fresh & valid
                ids = torch.nonzero(fresh).flatten()
                if ids.numel():
                    codes = self.plan_clf.classify(win[ids])
                    codes = torch.as_tensor(codes, device=self.device, dtype=torch.long)
                    self.plan_counts[ids, codes] += 1
                    if self.plan_dump is not None:
                        # Close out the plan that just ENDED (its divergence is complete
                        # only now) before opening the new one.
                        self.plan_dump.finalize(ids, self._div_sum, self._div_n)
                        self.plan_dump.offer(ids, codes, win, self.cmd)
                        self._div_sum[ids] = 0.0
                        self._div_n[ids] = 0

        # -- served-vs-clip divergence, the ground-truth half of the plan dump -------------
        # The classifier says what a plan LOOKS like; this says how far it is from what the
        # clip actually does, with no learned label in the way. _clip_history_node returns
        # the clip's node in the PLAN's frame -- the same frame the served node is in -- so
        # the difference is meaningful without any re-anchoring here.
        if self.plan_dump is not None:
            try:
                served = self.cmd._pred_window[self._arange, self.cmd._pred_idx]
                clip_node = self.cmd._clip_history_node()
                # Base POSITION only (window channels 0:3). The node mixes position, rot6d
                # and 29 joint angles; an unweighted norm over all of them is dominated by
                # whichever block happens to have the largest units, which is not a claim
                # about the plan.
                # NOT `d`: that name is the _Data wrapper further down this method, and
                # shadowing it makes fold() die on `d.applied_torque` several branches later.
                div_step = torch.linalg.norm(served[:, :3] - clip_node[:, :3], dim=-1)
                div_ok = m & self.cmd._pred_valid
                self._div_sum = self._div_sum + torch.where(
                    div_ok, div_step, torch.zeros_like(div_step))
                self._div_n = self._div_n + div_ok.long()
            except (AttributeError, RuntimeError):
                pass

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

        self._fold_goal(m)

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

    # TODO(terminal-tilt): `max_tilt_rad` is CONTAMINATED and `first_fall_step` may be too.
    #
    # A non-trivial fraction of SUCCESSFUL episodes record `max_tilt_rad` ABOVE the 70 deg
    # termination limit, some of them far above it. That is impossible if the column measured
    # what it claims -- `base_fallen` would have ended those episodes -- so a transient is
    # being folded in. First seen on a closed-loop rollout; re-derive the rate from any
    # `episodes.h5` by taking `max_tilt_rad` over the episodes the reward scores a success.
    #
    # The suspect is the `max()` below. It is deliberate and well argued for ordering
    # robustness, but max(a, b) where one input is a mid-teleport pose is that pose. If
    # `root_quat_w` is momentarily garbage inside the reset hook rather than merely fresh,
    # this picks it up and nothing downstream can tell.
    #
    # WHAT IS AFFECTED, and what is not:
    #   * `max_tilt_rad` -- unusable as reported.
    #   * `first_fall_step` -- `fold_terminal` sets it when EITHER limb crosses, so a
    #     spurious tilt marks an episode as having fallen when it did not. That makes the
    #     fall-vs-stall SPLIT unreliable.
    #   * the success verdict is SAFE: it comes from `reached_goal` and the first-event
    #     ordering, and an episode that reached the goal before the terminal frame keeps its
    #     success regardless. That is why this hid.
    #   * the eval harness shares this accumulator, so its `max_tilt_rad` and derived `fell`
    #     column may carry the same defect. CLAUDE.md records a PREVIOUS bug in this exact
    #     place (the derived fall flag inheriting the previous episode's orientation), so
    #     check whether this is a recurrence rather than a new fault.
    #
    # To diagnose: log both limbs separately at the terminal fold for a few hundred episodes
    # and compare against the per-step maximum, which is computed from a buffer that is not
    # mid-reset. Blocks TODO(jump-off-front).
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

        # The goal too, and for the same reason: under `clip_end` truncation the LAST step is
        # exactly when a robot is most likely to be standing in the goal, so omitting the
        # terminal frame would systematically under-count arrivals.
        #
        # This inherits the same assumption the height limb above already makes -- that
        # `root_pos_w` still holds the terminal pose here -- which `_check_terminal_state`
        # verifies empirically once per run. If that ever breaks, the failure is benign in
        # the one direction that matters: a post-reset read is the clip's SPAWN, which is
        # maximally far from the goal, so it can corrupt `min_dist_to_goal_m` but can never
        # falsely mark an episode as having arrived.
        m = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        m[ids] = True
        self._fold_goal(m)
        # And the strict checks, for the reason the fall crossing is folded here: the frame
        # that ended the episode is the one most likely to have crossed, so omitting it would
        # leave a violated episode reading clean.
        self._fold_tracking(m)

    def snapshot(self, env_ids: torch.Tensor) -> dict[str, np.ndarray]:
        """Reduce the running sums for ``env_ids`` into per-episode values.

        Returns numpy arrays; one host transfer per quantity per done-batch, and none per
        step, so the hot loop never synchronises.
        """
        ids = env_ids.to(self.device)
        steps = self.steps[ids].clamp(min=1).to(torch.float32)
        cpu = lambda t: t.detach().to("cpu").numpy()  # noqa: E731

        pc = cpu(self.plan_counts[ids]).astype(np.int64)
        # NaN/inf out of the goal buffers means "never folded with a valid anchor", which is
        # what an un-attached goal region looks like; `report.py` refuses to derive spatial
        # success from that rather than reading it as "never arrived".
        end_goal = cpu(self.end_goal[ids]).astype(np.float32)
        min_gd = cpu(self.min_goal_dist[ids]).astype(np.float32)
        max_gdown = cpu(self.max_goal_down[ids]).astype(np.float32)
        out: dict[str, np.ndarray] = {
            "reached_goal": cpu(self.goal_reached[ids]).astype(np.uint8),
            "first_goal_step": cpu(self.first_goal[ids]).astype(np.int64),
            "min_dist_to_goal_m": np.where(np.isfinite(min_gd), min_gd, np.nan),
            "goal_downrange_m": end_goal[:, 0],
            "goal_lateral_m": end_goal[:, 1],
            "goal_height_err_m": end_goal[:, 2],
            "max_goal_downrange_m": np.where(np.isfinite(max_gdown), max_gdown, np.nan),
            "steps_survived": cpu(self.steps[ids]).astype(np.int64),
            "n_steps_scored": cpu(self.steps[ids]).astype(np.int64),
            "n_plans": pc.sum(1),
            "n_plans_flat": pc[:, 0],
            "n_plans_jump": pc[:, 1],
            "n_plans_stairs": pc[:, 2],
            "cmd_speed_mean": cpu(self.cmd_speed[ids] / steps),
            "ref_base_ori_mse_rad": cpu(self.ori_sq[ids] / steps),
            "ref_base_ori_max_rad": cpu(self.ori_max[ids]),
            "ref_finite_frac": cpu(self.finite_frac[ids] / steps),
            "clf_v_mean": cpu(self.clf_v[ids] / steps),
            "min_base_height_w": cpu(self.min_h[ids]),
            "max_tilt_rad": cpu(self.max_tilt[ids]),
            "first_fall_step": cpu(self.first_fall[ids]).astype(np.int64),
            "first_violation_step": cpu(self.first_violation[ids]).astype(np.int64),
            "track_reward_mean": cpu(torch.where(
                self.track_reward_n[ids] > 0,
                self.track_reward_sum[ids] / self.track_reward_n[ids],
                torch.full_like(self.track_reward_n[ids], float("nan")))),
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
        # Per-CHECK first-crossing steps. Named individually rather than pooled into the
        # one `first_violation_step`: lateral and height are chunk-local while tilt is
        # measured against gravity, so a pooled rate hides which bound is doing the work.
        for _name in tracking_error.VIOLATION_CHECK_NAMES:
            _buf = self.first_check.get(_name)
            out[f"first_{_name}_step"] = (
                cpu(_buf[ids]).astype(np.int64) if _buf is not None
                else np.full(int(ids.numel()), -1, dtype=np.int64))
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
        self.first_violation[ids] = -1
        for _buf in self.first_check.values():
            _buf[ids] = -1
        self.track_reward_sum[ids] = 0.0
        self.track_reward_n[ids] = 0.0
        self.plan_counts[ids] = 0
        self.act_rate[ids] = 0.0
        self.act_rate_max[ids] = 0.0
        self.torque_util[ids] = 0.0
        self.torque_sq[ids] = 0.0
        self.work[ids] = 0.0
        self.dist[ids] = 0.0
        self.slip[ids] = 0.0
        self.contact_steps[ids] = 0.0
        self._have_prev[ids] = False
        self.goal_reached[ids] = False
        self.first_goal[ids] = -1
        self.min_goal_dist[ids] = float("inf")
        self.max_goal_down[ids] = -float("inf")
        self.end_goal[ids] = float("nan")
        # NOT `_goal_valid`: the anchor is refreshed by `refresh_goal_anchor` AFTER the clip
        # reset commits the new clip, and clearing it here would only widen the window in
        # which a stale anchor could be folded.
