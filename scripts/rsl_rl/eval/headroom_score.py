# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-window scoring for one branch: survival plus the graded components.

Binary survival over a 1.2 s window is nearly flat -- most branches survive -- so the signal
that separates a good chunk from a bad one lives in how close the bad ones came.  This folds
both, once per control step, for every env at once.

**Every deviation is measured against the branch's OWN generator reference**, never against a
clip and never against a nominal stance.  ``cmd.y_des`` under ``reference_source='fm'`` IS
the generator's window lifted to the tracked channels, so ``y_des - y_act`` is already the
right difference and no forward kinematics enters: on this task the tracker's observation is
the generator's window and ``tracking_mode`` is ``joint``, so there are no ``@root`` body
channels to derive.  (``FmTrackingCommand`` refuses ``body`` + ``fm`` outright.)  The channel
grouping is imported from ``accumulate`` rather than restated, so a window error here and a
``ref_rmse_*`` column in the episode harness mean the same thing.

**No component is weighted here.**  ``headroom_report`` normalizes each by its own median
over the run and sums; picking weights before seeing the distribution would bake a guess into
the raw record.
"""

from __future__ import annotations

import numpy as np
import torch

import accumulate
import headroom_schema as hs
import schema

#: Contact sensors that should not be touching anything.  The list is
#: ``G1ClfTrackingRewardCfg.undesired_contacts``' own ``sensor_cfgs`` -- the tracker task has
#: no rewards, so it cannot be read off the config here and is restated with the same
#: threshold.  The shins are deliberately absent, matching that term: they have their own
#: termination in the teacher and brushing a riser is normal on stairs.
UNDESIRED_CONTACT_SENSORS: tuple[str, ...] = (
    "left_thigh_contact", "right_thigh_contact", "torso_contact", "backpack_contact",
    "left_elbow_contact", "right_elbow_contact",
)
#: Newtons above which a contact sensor counts as touching, as in that reward term.
CONTACT_THRESHOLD_N: float = 1.0


class WindowScorer:
    """Fold one W-step window per env into the columns of ``headroom_schema``.

    Usage per round::

        scorer.begin(env_ids)
        for _ in range(window_steps):
            env.step(...)
            scorer.fold(alive_mask, term_bits, truncated)
        cols = scorer.harvest(env_ids)
    """

    def __init__(self, env, *, dt: float, command_name: str = "traj_ref") -> None:
        """Resolve the channel groups and the sensors, and allocate the per-env buffers.

        Args:
            env: The unwrapped environment.
            dt: Control step, seconds.
            command_name: Command term supplying ``y_des`` / ``y_act``.

        Raises:
            SystemExit: If a named undesired-contact sensor is absent from the scene, or if
                the command exposes no ``base_pos`` channel group -- either would silently
                zero a score component.
        """
        self.env = env
        self.dt = float(dt)
        self.n = int(env.num_envs)
        self.device = env.device
        self.cmd = env.command_manager.get_term(command_name)
        self.robot = env.scene.articulations["robot"]
        self.rd = accumulate._Data(self.robot.data)

        groups = accumulate.channel_groups(
            list(self.cmd.manager.pos_output_names), list(self.cmd.manager.vel_output_names),
            str(self.cmd.manager.ref_frame))
        self.gidx = {k: torch.as_tensor(v, device=self.device, dtype=torch.long)
                     for k, v in groups.items() if v.size}
        for want in ("base_pos", "joint_pos"):
            if want not in self.gidx:
                raise SystemExit(
                    f"the command exposes no {want!r} channel group; every tracking-error "
                    "component would be zero. channel_groups saw pos names "
                    f"{list(self.cmd.manager.pos_output_names)[:6]}..."
                )

        missing = [s for s in UNDESIRED_CONTACT_SENSORS if s not in env.scene.sensors]
        if missing:
            raise SystemExit(
                f"undesired-contact sensors absent from the scene: {missing}. The scene is "
                "not the CLF tracking base scene this scorer was written against; fix "
                "UNDESIRED_CONTACT_SENSORS rather than letting the component read zero."
            )
        self.undesired = [env.scene.sensors[s] for s in UNDESIRED_CONTACT_SENSORS]
        self.feet = [s for name, s in env.scene.sensors.items()
                     if name.endswith("foot_contact")]

        limits = accumulate._to_torch(getattr(self.robot.data, "joint_effort_limits", None)) \
            if getattr(self.robot.data, "joint_effort_limits", None) is not None else None
        self._effort = (limits[0].clone().to(self.device) if limits is not None
                        else torch.ones(self.robot.num_joints, device=self.device))
        self._effort = torch.where(self._effort > 0, self._effort,
                                   torch.ones_like(self._effort))

        z = lambda: torch.zeros(self.n, device=self.device)  # noqa: E731
        self.steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.alive_steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.err_sq = {k: z() for k in self.gidx}
        self.err_n = {k: z() for k in self.gidx}
        self.ori_sq = z()
        self.min_h = torch.full((self.n,), float("inf"), device=self.device)
        self.max_tilt = z()
        self.undesired_steps = z()
        self.foot_steps = z()
        self.act_rate = z()
        self.torque = z()
        self.term_step = torch.full((self.n,), -1, device=self.device, dtype=torch.long)
        self.term_bits = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.truncated = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.airborne = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self._prev_action = torch.zeros(self.n, self.robot.num_joints, device=self.device)
        self._term_names: list[str] = list(env.termination_manager.active_terms)

    # ------------------------------------------------------------------
    @property
    def termination_terms(self) -> list[str]:
        """The active termination term names, in bit order for ``termination_bits``."""
        return list(self._term_names)

    def begin(self, env_ids: torch.Tensor) -> None:
        """Zero every accumulator for ``env_ids`` and seed the action-rate reference."""
        ids = env_ids.to(self.device)
        self.steps[ids] = 0
        self.alive_steps[ids] = 0
        for k in self.gidx:
            self.err_sq[k][ids] = 0.0
            self.err_n[k][ids] = 0.0
        self.ori_sq[ids] = 0.0
        self.min_h[ids] = float("inf")
        self.max_tilt[ids] = 0.0
        self.undesired_steps[ids] = 0.0
        self.foot_steps[ids] = 0.0
        self.act_rate[ids] = 0.0
        self.torque[ids] = 0.0
        self.term_step[ids] = -1
        self.term_bits[ids] = 0
        self.truncated[ids] = False
        self.airborne[ids] = False
        self._prev_action[ids] = self.env.action_manager.action[ids]

    def fold(self, alive: torch.Tensor) -> None:
        """Fold one control step.

        Must be called immediately after ``env.step``, before anything else reads the
        termination manager: its buffers still describe the step that just ended, while
        ``robot.data`` already describes the new episode for any env that reset.  ``alive``
        is what keeps the two apart.

        Args:
            alive: ``[num_envs]`` bool -- envs whose window is still running AND which did
                not terminate on this step.  A ``False`` env is frozen: every accumulator
                except the termination record ignores it.
        """
        m = alive.to(self.device).to(torch.bool)
        mf = m.to(torch.float32)
        d = self.rd
        self.steps += m.long()
        self.alive_steps += m.long()

        y_des, y_act = self.cmd.y_des, self.cmd.y_act
        dy_des, dy_act = self.cmd.dy_des, self.cmd.dy_act
        for name, idx in self.gidx.items():
            src_d, src_a = ((dy_des, dy_act) if name.endswith("_vel") else (y_des, y_act))
            diff = src_d[:, idx] - src_a[:, idx]
            good = torch.isfinite(diff)
            self.err_sq[name] += torch.where(good, diff * diff,
                                             torch.zeros_like(diff)).sum(dim=-1) * mf
            self.err_n[name] += good.sum(dim=-1).to(torch.float32) * mf
        if "base_ori" in self.gidx:
            oi = self.gidx["base_ori"]
            ang = torch.nan_to_num(accumulate._quat_angle(y_des[:, oi], y_act[:, oi]), nan=0.0)
            self.ori_sq += (ang * ang) * mf

        height = d.root_pos_w[:, 2]
        grav = d.projected_gravity_b
        tilt = torch.arccos(torch.clamp(-grav[:, 2], -1.0, 1.0))
        self.min_h = torch.where(m, torch.minimum(self.min_h, height), self.min_h)
        self.max_tilt = torch.maximum(self.max_tilt, tilt * mf)

        touching = self._contact_any(self.undesired)
        self.undesired_steps += touching.to(torch.float32) * mf
        foot = self._contact_any(self.feet)
        self.foot_steps += foot.to(torch.float32) * mf
        # Airborne is the LAST alive step's answer, not an OR over the window: the question
        # is whether the window ended mid-flight (spec §4.4), so it must be overwritten, not
        # accumulated.
        self.airborne = torch.where(m, ~foot, self.airborne)

        act = self.env.action_manager.action
        self.act_rate += (act - self._prev_action).abs().mean(dim=-1) * mf
        self._prev_action = torch.where(m.unsqueeze(-1), act, self._prev_action)
        self.torque += (d.applied_torque.abs() / self._effort).mean(dim=-1) * mf

    def note_terminations(self, in_window: torch.Tensor) -> torch.Tensor:
        """Record which envs ended on this step and how, and return the ones that did.

        Args:
            in_window: ``[num_envs]`` bool -- envs whose window is still running.  An env
                already ended is not re-recorded, so the FIRST termination wins.

        Returns:
            ``[num_envs]`` bool of envs that terminated on this step.
        """
        tm = self.env.termination_manager
        fresh = in_window & (self.term_step < 0)
        bits = torch.zeros(self.n, device=self.device, dtype=torch.long)
        for b, name in enumerate(self._term_names):
            bits |= tm.get_term(name).to(torch.long) << b
        ended = fresh & (bits != 0)
        self.term_bits = torch.where(ended, bits, self.term_bits)
        self.term_step = torch.where(ended, self.steps, self.term_step)
        self.truncated = torch.where(ended, tm.time_outs.to(torch.bool), self.truncated)
        return ended

    def _contact_any(self, sensors: list) -> torch.Tensor:
        """Whether any of ``sensors`` reports a force above the threshold, per env."""
        out = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        for s in sensors:
            f = accumulate._to_torch(s.data.net_forces_w)
            out |= (torch.linalg.vector_norm(f, dim=-1) > CONTACT_THRESHOLD_N).any(dim=-1)
        return out

    # ------------------------------------------------------------------
    def harvest(self, env_ids: torch.Tensor) -> dict[str, np.ndarray]:
        """Reduce the folded window for ``env_ids`` into the candidate columns.

        ``survived`` is ``base_fallen`` not having fired -- the TERMINATION, not the derived
        ``fell`` column of the episode harness, which is looser on the tilt axis (it
        thresholds the total angle from upright where the termination limits roll and pitch
        separately, so roll 60 deg with pitch 60 deg is 75.5 deg total and reads as a fall
        that never terminated).  A truncation -- ``clip_end`` or ``time_out`` -- is survival.

        Args:
            env_ids: ``[m]`` env indices, in the order the rows should come out.

        Returns:
            Column name -> ``[m]`` array, covering every entry of
            ``headroom_schema.CANDIDATE_COLUMNS`` plus the error and cost channels.
        """
        ids = env_ids.to(self.device)
        cpu = lambda t: t[ids].detach().to("cpu").numpy()  # noqa: E731
        steps = cpu(self.steps).astype(np.int64)
        denom = np.maximum(steps, 1).astype(np.float64)

        bits = cpu(self.term_bits).astype(np.int64)
        fell_bit = (1 << self._term_names.index("base_fallen")
                    if "base_fallen" in self._term_names else 0)
        fell = (bits & fell_bit) != 0
        term_step = cpu(self.term_step).astype(np.int64)

        out: dict[str, np.ndarray] = {
            "survived": (~fell).astype(np.uint8),
            "truncated": cpu(self.truncated).astype(np.uint8),
            "window_steps": steps.astype(np.int32),
            "steps_before_term": np.where(term_step >= 0, term_step, steps).astype(np.int32),
            "time_to_term_s": np.where(term_step >= 0, term_step * self.dt,
                                       np.nan).astype(np.float32),
            "termination_bits": bits.astype(np.uint16),
            "terminal_airborne": cpu(self.airborne).astype(np.uint8),
            "min_height_margin_m": (cpu(self.min_h)
                                    - schema.FALL_MIN_HEIGHT_M).astype(np.float32),
            "max_tilt_rad": cpu(self.max_tilt).astype(np.float32),
            "foot_contact_frac": (cpu(self.foot_steps) / denom).astype(np.float32),
            "undesired_contact_frac": (cpu(self.undesired_steps) / denom).astype(np.float32),
            "action_rate": (cpu(self.act_rate) / denom).astype(np.float32),
            "torque_util": (cpu(self.torque) / denom).astype(np.float32),
        }
        for chan in hs.ERROR_CHANNELS:
            if chan == "base_ori":
                # A geodesic angle, not the component-wise quaternion difference the generic
                # group sum would give -- the same distinction accumulate.py draws.
                out["err_base_ori"] = np.sqrt(cpu(self.ori_sq) / denom).astype(np.float32)
                continue
            if chan not in self.gidx:
                out[f"err_{chan}"] = np.full(steps.size, np.nan, dtype=np.float32)
                continue
            # NaN, not zero, where the block carried no finite channel at all: this
            # generator predicts base_pose / base_vel / joint_pos, so ``strict_fm_only``
            # NaN-fills ee_pos for every row -- and a zero there reads in a table as
            # perfect tracking of a channel that was never predicted.
            counted = cpu(self.err_n[chan])
            out[f"err_{chan}"] = np.where(
                counted > 0, np.sqrt(cpu(self.err_sq[chan]) / np.maximum(counted, 1.0)),
                np.nan).astype(np.float32)
        return out


#: Per-channel divisors for the ONLINE commit score.  Declared rather than fitted, because
#: ``--commit_mode argmax`` has to rank candidates while the run is happening and cannot wait
#: for the run's own medians.  Each is roughly the median error of a surviving candidate on
#: this pair, so every term contributes O(1); they are recorded in the run manifest, and
#: Table 4 is conditional on them in the same way any test-time policy is conditional on its
#: scorer.  The REPORT's graded score does not use these -- it normalizes by the observed
#: medians instead, which is why the two are separate functions.
COMMIT_SCALES: dict[str, float] = {
    "err_base_pos": 0.10, "err_base_ori": 0.20, "err_joint_pos": 0.15, "err_ee_pos": 0.10,
    "err_base_lin_vel": 0.50, "err_base_ang_vel": 1.00, "undesired_contact_frac": 0.10,
}


def tracking_cost(cols: dict[str, np.ndarray],
                  scales: dict[str, float] | None = None) -> np.ndarray:
    """Sum of the per-channel deviations, each divided by its own scale.

    NaN channels (a block the generator does not write, NaN-filled by ``strict_fm_only``)
    contribute zero rather than poisoning the sum -- the same NaN-safety
    ``accumulate.EpisodeAccumulator`` applies to the episode-level errors.

    Args:
        cols: Column name -> array, as returned by :meth:`WindowScorer.harvest`.
        scales: Divisors, defaulting to :data:`COMMIT_SCALES`.

    Returns:
        A non-negative cost per row, higher being worse.
    """
    scales = COMMIT_SCALES if scales is None else scales
    total = None
    for name, scale in scales.items():
        if name not in cols:
            continue
        term = np.nan_to_num(np.asarray(cols[name], dtype=np.float64), nan=0.0) / float(scale)
        total = term if total is None else total + term
    return np.zeros(0) if total is None else total


def commit_score(cols: dict[str, np.ndarray]) -> np.ndarray:
    """The online rule ``--commit_mode argmax`` ranks candidates by.

    Survival dominates, partial survival breaks ties among the ones that fell, and tracking
    cost breaks ties among the ones that did not.  Composed so it is strictly ordered rather
    than a weighted blend: a candidate that survives must never lose to one that fell,
    whatever its tracking error, because the question Table 4 answers is how far an oracle
    could get and falling ends the episode.

    Args:
        cols: Column name -> array from :meth:`WindowScorer.harvest`.

    Returns:
        A score per row, higher being better.
    """
    steps = np.maximum(np.asarray(cols["window_steps"], dtype=np.float64), 1.0)
    frac = np.asarray(cols["steps_before_term"], dtype=np.float64) / steps
    survived = np.asarray(cols["survived"], dtype=np.float64)
    # The tracking term is SATURATED into [0, 10) rather than used raw, so the ordering the
    # docstring claims is a property of the formula instead of an assumption about how bad
    # a candidate can get. Raw, a candidate whose channels sit at 10x their scale scores
    # -337 and loses to one that fell -- which is exactly backwards, and is what
    # test_commit_score_never_ranks_a_faller_above_a_survivor caught. The map is strictly
    # increasing, so it still ranks two bad candidates against each other; it reaches 5 when
    # every channel sits at its own scale.
    cost = tracking_cost(cols)
    n_terms = float(sum(1 for c in COMMIT_SCALES if c in cols)) or 1.0
    bounded = 10.0 * cost / (cost + n_terms)
    return 100.0 * survived + 10.0 * frac - bounded
