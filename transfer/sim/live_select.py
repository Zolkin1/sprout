# Copyright 2026 Zachary Olkin. All rights reserved.

"""A LIVE 1-step-lookahead selector: try K chunks for real, keep the one the critic likes.

Gate A measures a CEILING -- what a perfect selector could reach from a frozen state.  This
runs the selector, so its output is an episode success RATE rather than a per-window number,
and it is rank-based end to end: only the ORDER of ``V_priv`` over K candidates matters, so
every calibration question hanging over Gate A is irrelevant here.  Its result stands however
the calibration work turns out.

At each re-plan: snapshot, execute 0.24 s of each of K candidate chunks, evaluate ``V_priv``
on the resulting state, pick the argmin, and continue from THAT branch's state.

``--lookahead n`` deepens the evaluation without changing the decision: each candidate is
carried a further ``n - 1`` chunks with NO branching before ``V`` is read, and only the FIRST
chunk is committed.  Breadth and depth then cost the same thing per decision, which is what
makes ``--k 4 --lookahead 2`` and ``--k 8 --lookahead 1`` a controlled comparison.

**The winner is not re-executed.**  The branch length equals the re-plan cadence, so the
winner's post-branch state IS the state the episode continues from -- restoring and replaying
it would cost a (K+1)th chunk and, worse, is the kind of duplicated execution that silently
diverges if the restore is imperfect.

**Snapshot = whole ``mjData`` + policy state, and both halves are load-bearing.**  Measured on
one 0.24 s chunk:

* ``mj_copyData`` + policy state -> **0.000e+00** on qpos and qvel, holding at 12, 36 and 120
  control steps;
* ``mj_copyData`` alone, policy left running -> **1.787 m**;
* ``mj_setState(mjSTATE_INTEGRATION)`` + policy state -> 8.99e-03 m.

``mjSTATE_INTEGRATION`` is insufficient and cannot be repaired.  Contacts genuinely re-derive
from ``qpos``, but the FORWARD KINEMATICS do not, and the policy reads them (``cvel`` for base
world velocity, ``xpos``/``xmat``/``geom_xpos`` for the depth raycast) in ``build_state``
BEFORE the first ``mj_step`` would re-derive anything.  Straight after ``mj_setState`` those
arrays still describe the previous branch: ``xpos`` off by 0.420 m.  ``mj_forward`` does not
fix it and cannot -- MuJoCo's derived kinematics trail ``qpos`` by one physics step, so a
re-derived state is self-consistent but is not the state the policy observed (measured: it
moves ``xpos`` by 2.1e-03 m on an UNTOUCHED post-step ``mjData``).

**Cost.**  Snapshot+restore is 0.22 ms against a 95.6 ms candidate chunk -- 0.2%, free.  The
whole cost is K forward passes, of which ~55% is depth raycasting.  K=8 is 766 ms per re-plan,
3.2x real time.  Against Gate A's replay-from-spawn on the same course this is **14.7x less
work** (21.7 s vs 317.8 s per episode), because a branch at re-plan n costs n chunks there and
one here.

**The harness check is K=1, and it is exact rather than statistical.**  With one candidate the
selector has no choice to make, so its trajectory must be BIT-IDENTICAL to the plain runner at
the same seed.  That tests the snapshot machinery directly.  Checking instead that K=1
reproduces a published success rate would be far weaker -- and in this case impossible, since
``fm_variance_probe``'s 0.798/0.816 were measured with a DIFFERENT generator
(``lag48_aug``) from the one the critic was fitted over (``nojumpoff``).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: Policy attributes that must be captured for an exact restore, with the measured cost of
#: omitting each (max |dqpos| in metres after one 0.24 s chunk).  Established by ablation --
#: leave one out, branch, measure -- rather than by reading the code, because "what the policy
#: carries across a step" is not something the class states anywhere.
#:
#:   _history            1.653e+00   the generator's emitted-node ring
#:   _obs_history        1.225e-01   the tracker's per-term obs windows
#:   _prev_served_node   1.043e-01
#:   _rng                5.748e-02
#:   _ctrl_step          IndexError  at a branch length not a multiple of fm_update_every
#:
#: The plan block (_pred .. _clip_anchor_*) measures 0.0 ONLY because a snapshot taken on a
#: re-plan boundary has step 0 overwrite it.  It is captured anyway: a mid-chunk snapshot
#: makes it live, and a field that is dead by coincidence of timing is not dead.
_POLICY_FIELDS = (
    "_history", "_obs_history", "_prev_served_node", "_rng", "_ctrl_step",
    "_pred", "_pred_valid", "_pred_idx", "_anchor_pos", "_anchor_quat", "_plan_id",
    "_last_plan", "_last_reference_window", "_last_served_window", "_lead_buf",
    "_future_cmd_override", "_clip_step",
    # Read by `served_reference_w()`, which the tracking probe calls every step, and cleared
    # by `reset()` -- so it is live policy state. Its partner `_last_served_window` was here
    # and it was not; the omission was invisible only because `create_obs` always precedes
    # the read within a step. Any caller reading the served reference straight after a
    # restore would have lifted the winner's window through candidate K-1's anchor.
    "_last_served_anchor",
)


class Snapshot:
    """An exact restore point for the sim and the policy together.

    Args:
        robot: The ``transfer.sim.robot.Robot``.
        policy: The ``GenTrackerPolicy``.
        fm_lag: The camera-lag ring.
    """

    def __init__(self, robot, policy, fm_lag) -> None:
        import mujoco  # noqa: PLC0415

        self._mj = mujoco
        self.data = mujoco.MjData(robot.mj_model)
        mujoco.mj_copyData(self.data, robot.mj_model, robot.mj_data)
        self.policy = {k: copy.deepcopy(getattr(policy, k))
                       for k in _POLICY_FIELDS if hasattr(policy, k)}
        self.action = copy.deepcopy(getattr(policy._trk, "action_isaac", None))
        self.lag = copy.deepcopy(getattr(fm_lag, "_buf", None))
        # Anything clip-mode carries. Absent on the generator path; captured when present so
        # this class does not quietly depend on which reference source is loaded.
        self.clip_anchor = {k: copy.deepcopy(getattr(policy, k))
                            for k in dir(policy) if k.startswith("_clip_anchor")}

    def restore(self, robot, policy, fm_lag) -> None:
        """Put both halves back.

        Args:
            robot: The robot to restore into.
            policy: The policy to restore into.
            fm_lag: The camera-lag ring to restore into.
        """
        self._mj.mj_copyData(robot.mj_data, robot.mj_model, self.data)
        for k, v in self.policy.items():
            setattr(policy, k, copy.deepcopy(v))
        for k, v in self.clip_anchor.items():
            setattr(policy, k, copy.deepcopy(v))
        if self.action is not None:
            policy._trk.action_isaac = copy.deepcopy(self.action)
        if self.lag is not None:
            fm_lag._buf = copy.deepcopy(self.lag)


class SelectedChunkWriter:
    """Record the chunk the critic CHOSE at each re-plan, in the fine-tune's own input format.

    This is the filtered-BC dataset: one row per re-plan, holding the conditioning the
    generator saw and, as ``fm_plan``, the candidate the selector committed -- so training on
    it is "reproduce the chunk that was picked".  ``finetune_generator.py`` with no
    ``--labels`` consumes exactly this and calls it unconditioned self-distillation.

    **Only the five columns the trainer reads are written** (``fm_plan``, ``history``,
    ``extero``, ``cmd_future``, ``history_null``) plus the join keys, rather than the full
    ``rollout_outcomes`` schema.  Writing the rest as zeros would produce a file that looks
    like a rollout recording and silently answers wrong for proprio, the scan and the goal
    offsets; ``is_selected_chunk_recording`` marks it so nothing mistakes it for one.

    **``record_candidates`` adds the REJECTED chunks, and costs almost nothing extra**,
    because every candidate at one re-plan branches from the same snapshot and re-plans at the
    same state -- they differ only in the flow noise draw.  So the conditioning is shared and
    already stored once, and a row grows by ``K`` plans plus ``K`` scores rather than by ``K``
    copies of a 7800-wide exteroception row.  These columns are a DIAGNOSTIC, never training
    data: ``ft_canary.PairGapCanary`` reads them to ask whether the model is pulling away from
    the chunks the critic rejected or improving on all of them together.

    ``cand_plans`` is fp32 where ``extero`` is fp16 on purpose.  ``extero`` is model INPUT,
    where fp16 is well inside the sensor's own noise; these are flow-loss TARGETS, and the
    paired gaps being measured can be ~1e-3, which is fp16's resolution there.

    Args:
        path: Output ``.h5``.
        meta: Provenance stamped into the file's attrs.
        record_candidates: Also store every candidate's plan, score and chunk verdict.
    """

    #: ``cand_verdict`` codes.  CHUNK-level -- what happened during that candidate's own
    #: 0.24 s branch, which for a REJECTED candidate is the only outcome that exists, since the
    #: episode never followed it.  Episode-level outcome is still available to a consumer:
    #: both members of a pair share ``episode_uid``, so `live_select.json`'s verdict conditions
    #: the pair (which is how ``merge_selected.py --successful_only`` already works).
    #:
    #: **NOT ``terrains.VERDICT_CODES``, and the two must never be crossed.** That table has
    #: no ``""`` member (an episode always ends) and numbers the rest differently -- there
    #: ``success`` is 0 and ``timeout`` 2, here they are 2 and 3 -- so reading one recording's
    #: codes against the other table silently swaps success for timeout. ``fall`` is 1 in both,
    #: which is exactly enough coincidence to make a spot check pass. ``violation`` is appended
    #: as 4 rather than inserted, so every code in an existing recording keeps its meaning.
    VERDICT_CODES = {"": 0, "fall": 1, "success": 2, "timeout": 3, "violation": 4}

    def __init__(self, path: str, meta: dict, record_candidates: bool = False) -> None:
        self.path = str(path)
        self.meta = dict(meta)
        self.record_candidates = bool(record_candidates)
        self.rows: list[dict] = []

    def add(self, *, episode_uid: int, replan_index: int, family_code: int,
            history: np.ndarray, history_null: bool, extero: np.ndarray,
            cmd_future: np.ndarray, fm_plan: np.ndarray,
            cand_plans: np.ndarray | None = None, cand_scores: np.ndarray | None = None,
            cand_verdict: np.ndarray | None = None, sel_index: int = 0,
            cand_pfail: np.ndarray | None = None, cand_ghat: np.ndarray | None = None,
            cand_reward: np.ndarray | None = None) -> None:
        """Append one selected chunk with the conditioning it was drawn from.

        Args:
            episode_uid: Join key to ``live_select.json``'s per-episode verdict.
            replan_index: Re-plan ordinal within the episode.
            family_code: Terrain family.
            history: ``(N, D)`` history conditioning.
            history_null: Whether the learned null history stood in.
            extero: RAW fused exteroception row (the dataset normalizes it).
            cmd_future: ``(T, 3)`` future command tokens.
            fm_plan: The COMMITTED chunk.
            cand_plans: ``(K, F, D)`` every candidate's chunk, in candidate order.
            cand_scores: ``(K,)`` the value each was ranked on.
            cand_verdict: ``(K,)`` per-candidate CHUNK verdict codes.
            sel_index: Which candidate was committed.
            cand_pfail: ``(K,)`` the survival head's ``P(fail)`` per candidate.
            cand_ghat: ``(K,)`` the quality head's ``G_hat`` per candidate, ``None`` on the
                survival-only control arm.
            cand_reward: ``(K,)`` MEASURED mean chunk reward per candidate; NaN where every
                step fell inside the opening command hold.
        """
        row = {
            "episode_uid": np.int64(episode_uid), "replan_index": np.int32(replan_index),
            "family_code": np.uint8(family_code),
            "history": np.asarray(history, dtype=np.float32),
            "history_null": np.uint8(bool(history_null)),
            "extero": np.asarray(extero, dtype=np.float16),
            "cmd_future": np.asarray(cmd_future, dtype=np.float32),
            "fm_plan": np.asarray(fm_plan, dtype=np.float32),
        }
        if self.record_candidates:
            row["cand_plans"] = np.asarray(cand_plans, dtype=np.float32)
            row["cand_scores"] = np.asarray(cand_scores, dtype=np.float32)
            row["cand_verdict"] = np.asarray(cand_verdict, dtype=np.uint8)
            row["sel_index"] = np.uint8(sel_index)
            # The score DECOMPOSED into the three things that made it, so a sweep can ask
            # whether the heads disagreed rather than only what the sum ranked. Optional in
            # the row because the survival-only control arm produces no G_hat -- absent, not
            # zero-filled: a zero G_hat is the WORST possible return and would read as the
            # quality head condemning every candidate.
            for key, val, dt in (("cand_pfail", cand_pfail, np.float32),
                                 ("cand_ghat", cand_ghat, np.float32),
                                 ("cand_reward", cand_reward, np.float32)):
                if val is not None:
                    row[key] = np.asarray(val, dtype=dt)
        self.rows.append(row)

    def columns(self) -> list[str]:
        """The datasets this writer emits, training columns first."""
        cols = ["episode_uid", "replan_index", "family_code", "history", "history_null",
                "extero", "cmd_future", "fm_plan"]
        if self.record_candidates:
            cols += ["cand_plans", "cand_scores", "cand_verdict", "sel_index"]
            # Only what every row actually carries, so a control-arm file has no half-written
            # column and `close()`'s stack never sees a missing key.
            cols += [c for c in ("cand_pfail", "cand_ghat", "cand_reward")
                     if self.rows and c in self.rows[0]]
        return cols

    def close(self) -> int:
        """Write the file. Returns the row count."""
        import h5py  # noqa: PLC0415

        n = len(self.rows)
        if not n:
            return 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        cols = self.columns()
        with h5py.File(self.path, "w") as f:
            for key in cols:
                f.create_dataset(key, data=np.stack([r[key] for r in self.rows]))
            f.attrs["num_windows_written"] = n
            f.attrs["is_selected_chunk_recording"] = 1
            f.attrs["has_candidates"] = int(self.record_candidates)
            f.attrs["verdict_codes"] = json.dumps(self.VERDICT_CODES)
            f.attrs["columns_written"] = json.dumps(cols)
            for k, v in self.meta.items():
                f.attrs[k] = v
        return n


def capture_conditioning(policy, state) -> dict | None:
    """The generator's conditioning and its emitted chunk, at the instant of a re-plan.

    Every accessor here is the one ``mjeval.critic.CriticInputs.from_runtime`` uses, for the
    reason its docstring gives: each of these has a shape-identical wrong source sitting
    beside it.  In particular ``policy._fm_extero_row(state)`` is the RAW fused row, while
    ``last_plan()["extero_tokens"]`` is the same shape already through the normalizer -- and
    the fine-tune dataset normalizes what it reads, so recording the latter would normalize
    twice with nothing downstream able to detect it.

    Args:
        policy: The ``GenTrackerPolicy``.
        state: The dict ``build_state`` produced for this control step.

    Returns:
        ``{history, history_null, extero, cmd_future, fm_plan, plan_id}``, or ``None`` when no
        plan is live yet.
    """
    plan = policy.last_plan()
    pred = policy.predicted_window()
    if plan is None or pred is None:
        return None
    hist = np.asarray(plan["history_nodes"], dtype=np.float32)
    cmd = np.asarray(plan["future_cmd"], dtype=np.float32)
    return {
        "history": hist.reshape(-1, hist.shape[-1]),
        "history_null": bool(plan["history_null"]),
        "extero": np.asarray(policy._fm_extero_row(state), dtype=np.float32),
        "cmd_future": cmd.reshape(-1, cmd.shape[-1]),
        "fm_plan": np.asarray(pred, dtype=np.float32),
        "plan_id": int(plan["plan_id"]),
    }


#: Keys the random-selection control's own RNG, kept clear of the candidate and depth ranges
#: for the same reason _LOOKAHEAD_TAG is.
_SELECT_TAG = 5501

#: Appended to every lookahead-continuation draw's seed sequence, so a continuation key can
#: never alias a first-chunk key however K and the depth grow.  An arbitrary literal outside
#: both the candidate and the depth range -- the sequences already differ in LENGTH today, and
#: this keeps that from being the only thing separating them.
_LOOKAHEAD_TAG = 7717


def _sha256(path: str | None) -> str:
    """Hash a file, or return ``""`` when there is none.

    Args:
        path: File to hash, or a falsy value.

    Returns:
        Hex digest, or ``""``. Empty and absent are deliberately the same here -- the caller
        stores it beside the path, which already says whether one was given.
    """
    import hashlib  # noqa: PLC0415

    if not path or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def arrived(course, robot, zex) -> bool:
    """Did the robot arrive, under whichever rule this course carries?

    ISAAC's rule when ``course.isaac_tol`` is set -- a downrange half-plane plus lateral and
    height bands, no traversal gate.  mjeval's own otherwise -- a rectangle around
    ``target_xy`` plus the traversal gate.

    They are not interchangeable and disagree three ways: Isaac counts OVERSHOOT (a
    half-plane, where mjeval's band can be walked out of), Isaac has a HEIGHT band that mjeval
    lacks, and mjeval has a traversal gate that Isaac does not need -- its height band already
    refuses a robot that never went up.

    Args:
        course: The course.
        robot: The MuJoCo robot.
        zex: ``[max_base_z, min_base_z]`` so far, for the mjeval gate.

    Returns:
        Whether the episode should end as a success.
    """
    q = robot.mj_data.qpos
    if course.isaac_tol is not None:
        from goal_anchor import in_isaac_goal  # noqa: PLC0415

        return in_isaac_goal((float(q[0]), float(q[1]), float(q[2])),
                             course.goal_anchor_xyz, course.isaac_tol)
    return bool(course.in_goal(float(q[0]), float(q[1]))
                and course.traversed(zex[0], zex[1]))


def _ep_quality(probe) -> float:
    """Mean measured per-chunk tracking reward over an episode, or NaN without a probe.

    The per-CHUNK mean rather than the per-step one, so it is the same quantity the quality
    head predicts a discounted sum of and the same one ``selection_score`` measures -- an
    episode's quality and a candidate's are then directly comparable numbers.

    Args:
        probe: An episode-level ``TrackingProbe``, or ``None``.

    Returns:
        The mean, or NaN when nothing was scored (no probe, or an episode that ended entirely
        inside the opening command hold).
    """
    if probe is None:
        return float("nan")
    probe.finish()
    return float(np.mean(probe.chunk_rewards)) if probe.chunk_rewards else float("nan")


def selection_score(p_fail: float, g_hat: float | None, chunk_rewards, *,
                    w_fail: float, gamma: float) -> float:
    """BADNESS of one candidate: lower is better, so the caller's ``argmin`` is unchanged.

    The quantity being ordered is ``J = w_track * G_hat_n - w_fail * P_fail``, negated. Both
    terms are in ``[0, 1]`` and the weights sum to 1, so ``w_fail`` is the single knob and its
    ratio against quality is ``w_fail / (1 - w_fail)`` -- 0.75 is 3x, 0.8 is 4x, 0.833 is 5x.
    That bounds the result in ``[-w_track, w_fail] <= [-1, 1]``, which is what lets a
    terminal candidate keep the historical ``+/-1.0`` sentinels as strict bounds.

    ``G_hat_n`` is the n-step estimate of the normalized return, from the identity
    ``G_t = (1 - gamma) * R_t + gamma * G_{t+1}``: the chunks the candidate actually RAN
    contribute their MEASURED reward and only the tail is a model output. At gamma = 0.95 the
    first measured chunk carries 5% of it, so this term is small -- but it is exact, where
    ``g_hat`` is not.

    **A strict violation is NOT a term here.** It is a hard TERMINATION: the caller ends the
    chunk on it and scores the candidate exactly as it scores a fall, because under the strict
    rule a violation IS a failure. Grading it instead -- an additive penalty, which is what
    this function used to take -- would have made the failure definition partly a reward, and
    then two different things would decide what "failed" means. Latching without terminating
    is right for a RECORDING, where it keeps the threshold an offline knob; everything
    downstream of the recording terminates.

    Args:
        p_fail: The survival head's ``P(fail | s')`` -- and its label is the STRICT outcome,
            so the violation rule is already inside this number rather than beside it.
        g_hat: The quality head's ``G_hat(s')``, or ``None`` on a survival-only checkpoint.
        chunk_rewards: Measured per-chunk mean rewards for the chunks this candidate ran,
            earliest first. May be empty when every step fell inside the opening hold.
        w_fail: Weight on failure, in ``[0, 1]``.
        gamma: Per-chunk discount, from the tracking spec.

    Returns:
        The badness, to be minimized.
    """
    if w_fail >= 1.0:
        return float(p_fail)
    w_track = 1.0 - float(w_fail)
    if g_hat is None:
        raise ValueError(
            "w_fail < 1 asks for the quality term, but this checkpoint has no quality head. "
            "Refit with --quality_bins > 0, or run the control arm at --w_fail 1.")
    tail = float(g_hat)
    g_n = tail
    if len(chunk_rewards):
        acc = 0.0
        for i, r in enumerate(chunk_rewards):
            acc += (gamma ** i) * float(r)
        g_n = (1.0 - gamma) * acc + (gamma ** len(chunk_rewards)) * tail
    return -(w_track * g_n - float(w_fail) * float(p_fail))


def _chunk(rig, policy, fm_lag, cmd, n_steps, cfg, zex, scorer=None,
          steer_kp: float = 0.0, steer_wz_max: float = 0.68, capture: dict | None = None,
          hold_until: float = 0.0, probe=None, step0: int = 0):
    """Advance ``n_steps`` control steps, returning early on a terminal condition.

    Mirrors ``runner.run_episode``'s inner loop and its ORDERING, which is load-bearing there
    and here: the fall test runs BEFORE the goal test, so a robot that face-plants across the
    line is a fall and the verdicts stay mutually exclusive.

    Args:
        rig: The course rig.
        policy: The loaded pair.
        fm_lag: The camera-lag ring.
        cmd: ``(3,)`` commanded twist.
        n_steps: Control steps to run.
        cfg: The ``ArmRun``.
        zex: Mutable ``[max_base_z, min_base_z]`` for the traversal gate, carried across
            chunks because it is an EPISODE-level quantity.  ``run_episode`` keeps it in the
            accumulator; this loop has no accumulator, and omitting it makes ``in_goal`` alone
            decide success -- which awards it to a robot that walked BESIDE the course on the
            infinite floor, exactly what the gate exists to refuse.
        scorer: Optional per-step callback ``(robot, policy, state)``.
        probe: Optional ``mjeval.tracking.TrackingProbe``, folded once per control step so the
            candidate's own MEASURED tracking reward and its strict latch are both available
            to the selector. The caller owes ``finish()`` before reading ``chunk_rewards``.
        step0: Control-step index of this chunk's first step, so the latch is stamped on the
            episode's clock rather than on a per-chunk count.
        steer_kp: P gain (1/s) on heading error toward the course's goal; ``0`` (default)
            holds ``cmd`` flat, matching ``runner.run_episode``'s own ``steer_kp`` semantics
            exactly (same function, ``runner.steer_yaw_rate``, called fresh every step).
        steer_wz_max: Symmetric clip on the steering yaw rate, rad/s.

    Returns:
        ``(verdict, failure, last_state)`` -- ``verdict`` is ``""`` if nothing terminal
        happened, and ``last_state`` is the state dict of the final control step, which the
        caller scores on.  Returned rather than rebuilt by the caller because ``build_state``
        MUTATES the depth-lag ring: calling it again to peek pushes an extra frame, and since
        the post-chunk snapshot is taken after that, every candidate was being committed with
        a desynchronised lag. It presented as the selector making outcomes WORSE than no
        selection at all (K=4 fell on three seeds that K=1 survived), which reads as a weak
        critic and is not.
    """
    from gen_tracker_runner import build_state  # noqa: PLC0415
    from runner import steer_yaw_rate  # noqa: PLC0415

    robot, course = rig.robot, rig.course
    last = None
    for _i in range(int(n_steps)):
        if robot.mj_data.time > course.budget_s:
            return "timeout", "", last
        # THE OPENING HOLD, per control step and by absolute sim time -- the same test
        # `runner.run_episode` makes (`time < hold_until` -> zero, then a hard STEP to the
        # command, no ramp). It lives here rather than in the caller because a 0.5 s hold is
        # 25 control steps and the chunk length is 12: it does not divide into chunks, so a
        # per-chunk decision could not express it.
        #
        # This was a NO-OP for the flag's whole life: `--cmd_hold_s` reached `cfg.cmd_hold_s`
        # and the recorded provenance, but only `run_episode` reads that field and this
        # harness never calls it. Two arms run at 0.5 came back BIT-IDENTICAL to the same
        # arms at 0.0, which is what exposed it -- a plausible-looking null result that was
        # really the flag doing nothing.
        step_cmd = cmd
        if hold_until > 0.0 and robot.mj_data.time < hold_until:
            step_cmd = np.zeros(3, dtype=np.float64)
        elif steer_kp > 0.0:
            qw, qx, qy, qz = (float(v) for v in robot.mj_data.qpos[3:7])   # MuJoCo wxyz
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            wz = steer_yaw_rate(course, float(robot.mj_data.qpos[0]), float(robot.mj_data.qpos[1]),
                                yaw, steer_kp, steer_wz_max)
            step_cmd = np.array([cmd[0], cmd[1], wz])
        state = build_state(robot, policy, step_cmd, rig.depth_cams, want_depth_hits=False,
                            fm_depth_cams=rig.fm_depth_cams, fm_lag=fm_lag)
        last = state
        obs = policy.create_obs(state)
        # Step 0 of a chunk IS the re-plan (the caller runs exactly `fm_update_every` steps
        # from a boundary), so this is the one instant the conditioning and the fresh chunk
        # are both live -- the same edge `record_rollouts._emit` reads.
        if capture is not None and not capture:
            got = capture_conditioning(policy, state)
            if got is not None:
                capture.update(got)
        robot.apply_action(policy.get_action(obs, robot.joint_names))
        if robot.failed():
            return "fall", robot.failure_reason(), last
        if scorer is not None:
            scorer(robot, policy, state)
        # The tracking probe folds at the SAME point as the proprio ring -- after the action is
        # applied, so `served_reference_w()` is the node this step was actually driven to. Step
        # 0 of a chunk is the re-plan, which is where the PREVIOUS chunk's reward closes; with
        # a fresh probe that is a no-op, and across a lookahead's continuations it is the
        # boundary. `finish()` closes the last one, which the caller owes.
        if probe is not None:
            probe.fold(robot, policy, step0 + _i, is_replan=(_i == 0))
            # A HARD TERMINATION, in the same slot as the fall test and for the same reason:
            # under the strict rule a violation IS a failure, so it ends the episode rather
            # than being graded. It has to be tested BEFORE the arrival check below, or a
            # robot that strays past the bounds and then reaches the goal would be scored a
            # success -- which is exactly the first-event ordering `schema.derive_success`
            # already encodes, applied live instead of reconstructed afterwards.
            if probe.violated:
                probe.finish()
                return "violation", ",".join(
                    n for n, s in probe.first_check_step.items() if s >= 0), last
        for _ in range(rig.decim):
            robot.step()
        z = float(robot.mj_data.qpos[2])
        zex[0], zex[1] = max(zex[0], z), min(zex[1], z)
        if arrived(course, robot, zex):
            return "success", "", last
    return "", "", last


def run_episode_selected(rig, policy, fm_lag, spawn, cmd, cfg, crit, anchor, *,
                         k: int, seed: int, mjrunner, steer_kp: float = 0.0,
                         steer_wz_max: float = 0.68, same_noise: bool = False,
                         recorder=None, episode_uid: int = -1, family_code: int = 0,
                         lookahead: int = 1, random_select: bool = False,
                         cmd_hold_s: float = 0.0, w_fail: float = 0.75,
                         spec=None, gamma: float = 0.95):
    """One episode driven by a live lookahead selector over ``k`` candidate chunks.

    Args:
        rig: The course rig.
        policy: The loaded pair.
        fm_lag: The camera-lag ring.
        spawn: The initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        crit: A ``critic.FailureCritic``, or ``None`` at ``k == 1``.
        anchor: Goal anchor for the critic's goal features.
        k: Candidates per re-plan.  ``k == 1`` takes the selector's own path with no choice to
            make, so it must reproduce the plain runner exactly -- that is the harness check,
            not a separate code path.
        seed: Episode seed.
        mjrunner: The ``runner`` module.
        steer_kp: P gain (1/s) on heading error toward the goal; ``0`` (default) disables
            steering. Forwarded to every :func:`_chunk` call, so it applies identically to
            the ``k == 1`` path and every ``k > 1`` candidate.
        steer_wz_max: Symmetric clip on the steering yaw rate, rad/s.
        lookahead: Chunks rolled per candidate before ``V`` is read.  ``1`` (default) is the
            myopic selector: score the state one chunk out.  ``n > 1`` continues each
            candidate for ``n - 1`` further chunks with NO branching -- one fresh draw each,
            the generator's own policy -- and bootstraps ``V`` from the end of that, so the
            score is an ``(n-1)``-step return under the unmodified generator plus a terminal
            value rather than a value read one chunk out.

            **Only the FIRST chunk is committed.**  The continuation exists to score that
            chunk and is then discarded, so the decision frequency is identical to
            ``lookahead=1`` and depth is the only thing that changes.  That is what makes
            ``k=4, lookahead=2`` and ``k=8, lookahead=1`` an equal-compute breadth-vs-depth
            pair: both roll 8 chunks per decision.

            **A terminal event in the continuation does NOT end the episode**, since the
            episode never went there -- it scores the branch (fall 1.0, arrival -1.0) and
            the committed verdict stays the first chunk's.  Ignored on the ``k == 1`` path,
            which makes no choice and so has nothing to look ahead for.

        w_fail: Weight on the survival head in the selection score; ``1.0`` is the
            survival-only control arm and reproduces it bit for bit.  See
            :func:`selection_score`.
        spec: The ``TrackingErrorSpec`` the quality head was fitted under, or ``None`` to run
            without a tracking probe (only valid at ``w_fail = 1``).
        gamma: Per-chunk discount for the n-step return, from ``spec``.

    Returns:
        ``(verdict, failure, n_replans, mean_candidates_scored)``.
    """
    import critic as C  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    from mjeval import tracking  # noqa: PLC0415

    # The quality term without a spec is a SILENTLY degraded selector, not a simpler one: no
    # probe means no measured reward and -- far worse -- no strict latch, so a candidate that
    # blew through the bounds mid-chunk would be scored as clean by the one term that cannot
    # see it. Refused rather than run, and refused HERE so the library entry point is as safe
    # as the CLI, whose defaults it now matches.
    if k > 1 and w_fail < 1.0 and spec is None:
        raise ValueError(
            f"w_fail={w_fail} asks for the quality term but no tracking spec was given, so "
            "there is no probe: the measured reward would be absent and the strict violation "
            "latch would never fire. Pass `spec`, or run the control arm at w_fail=1.")

    robot, course = rig.robot, rig.course
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)

    cadence = int(getattr(policy, "fm_update_every", 12))
    # THE PROPRIO RING IS PER CANDIDATE, and the episode's is the WINNER's.
    #
    # It used to be one object shared by every candidate and absent from `Snapshot`, so a
    # candidate began with its predecessor's rows and the episode carried candidate K-1's
    # away. That was harmless by arithmetic coincidence ALONE -- the ring is 10 rows and the
    # chunk 12, so each candidate fully overwrote it before the critic read -- and moving the
    # critic read to the re-plan (below) destroys that coincidence. The two are one fix.
    ep_ring = C.ProprioRing() if (crit is not None and crit.privileged) else None

    def _push_ring(target):
        """A per-step scorer callback that fills ``target``, or a no-op for ``None``."""
        def _cb(robot_, policy_, state):
            if target is not None:
                target.push(state["joint_pos"], state["joint_vel"], state["base_ang_vel"],
                            C.projected_gravity_b(state["root_quat"]))
        return _cb

    _ring = _push_ring(ep_ring)

    z0 = float(robot.mj_data.qpos[2])
    zex = [z0, z0]
    n_replans, scored = 0, []
    # THE K=1 ARM NEEDS THE SAME PROBE AS EVERY CANDIDATE, or the two arms are scored under
    # DIFFERENT FAILURE RULES and the paired delta measures the rule rather than the
    # selection. It did exactly that once: K=1 came back `fall 0.211, violation 0.000` while
    # K=4 came back `violation 0.255, fall 0.000` -- structurally impossible under a shared
    # criterion, and it read as best-of-K COSTING 0.042 with an interval excluding zero.
    #
    # Episode-level rather than fresh per chunk, unlike a candidate's: this is one continuous
    # trajectory, so the latch has to carry across re-plans and the per-chunk rewards
    # accumulate. A candidate gets a fresh probe because a REJECTED branch's latch must not
    # survive into its sibling.
    ep_probe = (tracking.TrackingProbe(spec, hold_until_s=cmd_hold_s)
                if spec is not None else None)
    #: Measured per-chunk tracking reward of the COMMITTED chunks, for the K>1 arm. Success is
    #: no longer the only objective -- the whole point of the dense reward is cleanliness -- so
    #: every arm reports its realized quality beside its verdict.
    ep_rewards: list[float] = []
    while robot.mj_data.time <= course.budget_s:
        if k <= 1:
            v, f, _st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, zex, scorer=_ring,
                              steer_kp=steer_kp, steer_wz_max=steer_wz_max,
                              hold_until=cmd_hold_s, probe=ep_probe,
                              step0=n_replans * cadence)
            if v:
                return v, f, n_replans, 0.0, _ep_quality(ep_probe)
            n_replans += 1
            continue

        snap = Snapshot(robot, policy, fm_lag)
        best, sel_index = None, 0
        # RANDOM-SELECTION CONTROL: every candidate is rolled and scored exactly as usual, and
        # then the winner is drawn UNIFORMLY instead of by argmin. It differs from the real arm
        # in one thing only -- the choice rule -- so it isolates the critic's RANKING, where
        # --same_noise confounds that with the per-re-plan RNG keying (measured worth +0.056
        # on 9 cells, a third of the apparent best-of-K gain, and unexplained).
        #
        # Its own stream, never `policy._rng`: drawing the choice from the policy's generator
        # would consume from the same sequence that produces the flow noise, so the control
        # would silently alter the candidates it is supposed to be choosing among.
        pick_rng = np.random.default_rng([seed, n_replans, _SELECT_TAG])
        all_cands = []
        want_cands = recorder is not None and recorder.record_candidates
        cand_plans, cand_scores, cand_verdict = [], [], []
        cand_pfail, cand_ghat, cand_reward = [], [], []
        #: EVERY candidate's score, collected unconditionally -- `cand_scores` above is only
        #: filled under --record_candidates, and the `spread` diagnostic must not depend on a
        #: recording flag being set.
        cand_scores_all: list[float] = []
        for cand in range(k):
            if cand:
                snap.restore(robot, policy, fm_lag)
            # The ONLY difference between candidates. One flow sample is drawn per re-plan,
            # so re-seeding here varies the chunk and nothing else.
            #
            # THE EPISODE SEED IS PART OF IT. Without it every K>1 episode draws the identical
            # candidate sequence, so all N episodes of an arm are ONE episode repeated -- the
            # measured "success rate" is then 0.000 or 1.000 and nothing else, which reads as
            # a catastrophically bad selector. K=1 is unaffected (it advances policy._rng from
            # the episode seed naturally), which is exactly what made the two arms look so far
            # apart. Seeded as a SEQUENCE rather than an arithmetic combination so no
            # (seed, replan, cand) triple can alias another.
            # SAME-NOISE CONTROL: drop `cand` from the key so every candidate is handed ONE
            # draw. The selector then chooses among K IDENTICAL chunks, so it has zero signal
            # by construction and any rise it shows over the REAL K arm at the same K is
            # manufactured by the harness.
            #
            # ITS COMPARATOR IS REAL K, NOT K=1. The two K>1 paths re-seed per re-plan
            # (`[seed, n_replans, ...]`) while K=1 lets `policy._rng` advance naturally from
            # `default_rng(seed)`, so same-noise K=4 and K=1 run DIFFERENT noise tapes and
            # differ for that reason alone -- a deviation there is the seeding scheme, not the
            # snapshot machinery, and reading it as a restore bug would be a false alarm.
            # What same-noise K=4 does reproduce is a zero-signal draw from the same
            # distribution as K=1, so it is a valid control on the SELECTOR and not a
            # bit-exactness test. The bit-exactness test is the K=1 arm itself, which must
            # match the plain runner.
            #
            # In Isaac this same control is what retired the per-state best-of-K harness: with
            # an imperfect state restore and a non-reproducible sim it still showed a
            # best-of-K rise (+0.005..+0.031, matching the real arm), because independent
            # per-candidate noise manufactures a rise of about its own flip probability. Here
            # the sim is deterministic and the snapshot restores qpos/qvel at 0.000e+00, so
            # any deviation from K=1 is a BUG in the branch machinery rather than a bias to
            # subtract -- which is what makes it a blocking check instead of a correction.
            policy._rng = np.random.default_rng([seed, n_replans] if same_noise
                                                else [seed, n_replans, cand])
            # Each candidate gets its OWN copy of the episode's z extremes: a branch that
            # climbs must not hand its traversal credit to a sibling that did not.
            cz = list(zex)
            cap: dict = {} if recorder is not None else None
            # A FRESH probe per candidate. Reusing one and resetting would work equally, but a
            # rejected candidate's latch must not survive into its sibling and a fresh object
            # makes that unconditional rather than dependent on a reset call being reached.
            cprobe = (tracking.TrackingProbe(spec, hold_until_s=cmd_hold_s)
                      if spec is not None else None)
            # ...and its OWN ring, branched from the episode's. See `_push_ring`.
            cand_ring = copy.deepcopy(ep_ring) if ep_ring is not None else None
            v, f, st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, cz,
                             scorer=_push_ring(cand_ring),
                             steer_kp=steer_kp, steer_wz_max=steer_wz_max,
                             hold_until=cmd_hold_s, capture=cap, probe=cprobe,
                             step0=n_replans * cadence)
            # THE COMMIT POINT IS THE END OF THE FIRST CHUNK, unconditionally now. With a
            # lookahead the continuation moves the sim past where the episode resumes; since
            # the scoring step below does the same at lookahead=1, the snapshot is taken here
            # on every path rather than after the critic read.
            post = Snapshot(robot, policy, fm_lag)
            v_score, st_score = v, st
            if lookahead > 1 and not v and st is not None:
                # The branch's OWN z extremes travel into the continuation (the traversal gate
                # is an episode-level quantity), but on a COPY: what the continuation climbs
                # must not be credited to the chunk being committed.
                cz_look = list(cz)
                for depth in range(1, lookahead):
                    # No branching here -- one draw, the generator's unmodified policy. The
                    # key is longer than any first-chunk key and carries _LOOKAHEAD_TAG, so a
                    # continuation can never be handed a first chunk's stream.
                    policy._rng = np.random.default_rng(
                        [seed, n_replans, depth, _LOOKAHEAD_TAG] if same_noise
                        else [seed, n_replans, cand, depth, _LOOKAHEAD_TAG])
                    v_score, _fl, st_look = _chunk(
                        rig, policy, fm_lag, cmd, cadence, cfg, cz_look,
                        scorer=_push_ring(cand_ring),
                        steer_kp=steer_kp, steer_wz_max=steer_wz_max, hold_until=cmd_hold_s,
                        probe=cprobe, step0=(n_replans + depth) * cadence)
                    if st_look is not None:
                        st_score = st_look
                    if v_score:
                        break
            # SCORE AT THE RE-PLAN, NOT AT THE CHUNK'S LAST STEP.
            #
            # `CriticInputs.from_runtime` takes `history`, `cmd_future` and `history_null`
            # from `policy.last_plan()`, which `_run_fm` refreshes only on the cadence
            # (`_ctrl_step % K == 0`), while `policy._history` appends every control step. So
            # reading the critic at the chunk's LAST step paired a history stashed at the
            # chunk's FIRST step with a pose, depth row, scan and goal twelve steps later --
            # and, fatally, left that history BIT-IDENTICAL across all K candidates, because
            # they all restored the same snapshot. The critic is a head over the FROZEN FM
            # encoder and history+cmd are two of its three token groups, so two thirds of the
            # trunk's input carried no information about which candidate had run.
            #
            # Measured symptom before the fix: the four candidates' `p_fail` differed by less
            # than 0.001 on 29.3% of re-plans and by less than 0.01 on 52.1%, median spread
            # 0.0087 against an across-state sd of 0.1998 -- and best-of-K bought nothing on
            # success over 63 cells. Every OTHER consumer of this critic gates on `is_replan`
            # (`CriticScorer.__call__`, `collect_q_data`, the Q selector in this file); this
            # was the only call site that did not, and `CriticScorer`'s own docstring names
            # the cadence mistake as "the single mistake that would make V_priv quietly
            # worse".
            #
            # ONE extra control step reaches step 0 of the NEXT chunk, which IS the re-plan:
            # `create_obs` there refreshes `last_plan()` with a history whose newest rows are
            # this candidate's own executed tail. That step's physics is discarded -- `post`
            # above is the commit point -- so the episode resumes exactly where it did before.
            # Its verdict is discarded too: the committed verdict stays the first chunk's.
            if not v_score and st_score is not None:
                _sv, _sf, st_replan = _chunk(
                    rig, policy, fm_lag, cmd, 1, cfg, list(cz),
                    scorer=_push_ring(cand_ring), steer_kp=steer_kp,
                    steer_wz_max=steer_wz_max, hold_until=cmd_hold_s,
                    step0=(n_replans + max(lookahead, 1)) * cadence)
                if st_replan is not None:
                    st_score = st_replan

            # The sentinels have to BOUND whatever the survivor branch can return, and that
            # range widens once the quality term is on. At w_fail = 1 they stay at the
            # historical +/-1.0 so the control arm reproduces every recorded score exactly.
            # Set for EVERY candidate, terminal ones included, so the diagnostic arrays stay
            # index-aligned with `cand_plans` / `cand_scores`. Appending only in the survivor
            # branch would shift them by one for every re-plan where a candidate fell, which
            # is a plausible-looking table of the wrong candidates.
            p_f, g_hat, r_meas = float("nan"), float("nan"), float("nan")
            if cprobe is not None:
                cprobe.finish()
            if v_score in ("fall", "violation"):
                # A candidate that already failed is worse than any survivor -- and a strict
                # VIOLATION is a failure, scored identically to a fall rather than discounted
                # for having stayed upright. Scored rather than skipped, so a re-plan where
                # EVERY candidate fails still has an argmin and the episode ends the way it
                # would have anyway. 1.0 bounds the survivor range, which is [-w_track,
                # w_fail] and so never exceeds it.
                score = 1.0
            elif v_score == "success":
                score = -1.0
            elif st_score is None:
                score = 1.0
            else:
                # The chunk's own last state, NOT a rebuilt one. See _chunk's docstring: a
                # rebuild advances the depth-lag ring and corrupts the committed state.
                # THE CANDIDATE'S OWN ring, not a shared one: its rows must be the steps this
                # candidate actually ran, for the same reason its snapshot and its chunk
                # reward are its own.
                x = C.CriticInputs.from_runtime(policy, robot, st_score, anchor=anchor,
                                                ring=cand_ring, want_scan=crit.privileged)
                p_f, g_read = crit.read(x)
                if cprobe is not None and cprobe.chunk_rewards:
                    r_meas = float(np.mean(cprobe.chunk_rewards))
                score = selection_score(
                    p_f, g_read, (cprobe.chunk_rewards if cprobe is not None else ()),
                    w_fail=w_fail, gamma=gamma)
                g_hat = float("nan") if g_read is None else float(g_read)
            # `v`/`f`/`cz` are the FIRST chunk's -- the continuation's verdict scores the
            # branch but is never the episode's, because the episode never ran it.
            cand_scores_all.append(float(score))
            cand_r = list(cprobe.chunk_rewards) if cprobe is not None else []
            if random_select:
                all_cands.append((score, post, v, f, cz, cap, cand_r, cand_ring))
            if best is None or score < best[0]:
                best = (score, post, v, f, cz, cap, cand_r, cand_ring)
                sel_index = cand
            if want_cands:
                # The conditioning is NOT stored per candidate: every candidate restored the
                # same snapshot (sim, policy ring AND depth-lag ring) and re-planned at the
                # same state, so history/extero/cmd are identical across them by construction
                # and are already written once from the winner's capture.
                cand_plans.append(cap["fm_plan"])
                cand_scores.append(score)
                cand_verdict.append(SelectedChunkWriter.VERDICT_CODES.get(v, 0))
                # The score DECOMPOSED, so a sweep can ask whether the two heads even
                # disagreed. The question that matters is agreement ACROSS CANDIDATES at one
                # state -- across states they correlate trivially, because a hard state is
                # both low-quality and high-failure -- and that comparison needs the two
                # stored per candidate rather than reduced to the score.
                cand_pfail.append(p_f)
                cand_ghat.append(g_hat)
                cand_reward.append(r_meas)
        if random_select:
            sel_index = int(pick_rng.integers(len(all_cands)))
            best = all_cands[sel_index]
        # THE REAL SCORE SPREAD, not `k`. This column existed to answer "did the critic
        # separate the candidates at all", and on the V arm it reported `k` -- a constant --
        # so 63 cells x 40 episodes were run with no measurement of whether there was any
        # selection signal. The Q arm writes a genuine max-minus-min under this same key, and
        # its docstring states the thing that was missing: "if that is ~0 the model is not
        # discriminating and any success change is not selection." A near-zero spread here is
        # now visible in `live_select.json` without needing --record_candidates.
        finite = [x for x in cand_scores_all if np.isfinite(x)]
        scored.append(max(finite) - min(finite) if len(finite) > 1 else 0.0)
        # Commit the winner by RESTORING ITS POST-CHUNK STATE -- the branch length is the
        # cadence, so that state is exactly where the episode continues from. Re-executing the
        # winning draw instead would cost a (K+1)th chunk and would diverge silently if the
        # restore were imperfect.
        best[1].restore(robot, policy, fm_lag)
        # The CHOSEN chunk, with the conditioning every candidate shared. Written on commit
        # rather than per candidate: the rejected ones are exactly what filtering removes.
        if recorder is not None and best[5]:
            extra_c = ({"cand_plans": np.stack(cand_plans),
                        "cand_scores": np.asarray(cand_scores, dtype=np.float32),
                        "cand_verdict": np.asarray(cand_verdict, dtype=np.uint8),
                        "cand_pfail": np.asarray(cand_pfail, dtype=np.float32),
                        "cand_ghat": np.asarray(cand_ghat, dtype=np.float32),
                        "cand_reward": np.asarray(cand_reward, dtype=np.float32),
                        "sel_index": sel_index} if want_cands else {})
            recorder.add(episode_uid=episode_uid, replan_index=n_replans,
                         family_code=family_code, history=best[5]["history"],
                         history_null=best[5]["history_null"], extero=best[5]["extero"],
                         cmd_future=best[5]["cmd_future"], fm_plan=best[5]["fm_plan"],
                         **extra_c)
        zex = list(best[4])
        # The winner's ring becomes the episode's, for the same reason its snapshot does: the
        # rows it holds are the only ones the episode actually lived through.
        if ep_ring is not None and best[7] is not None:
            ep_ring = best[7]
            _ring = _push_ring(ep_ring)
        # The COMMITTED chunk's measured tracking reward. Only the winner's, because only the
        # winner was executed -- a rejected candidate's reward describes a trajectory the
        # episode never took.
        ep_rewards.extend(best[6])
        n_replans += 1
        if best[2]:
            return (best[2], best[3], n_replans, float(np.mean(scored)),
                    float(np.mean(ep_rewards)) if ep_rewards else float("nan"))
    return ("timeout", "", n_replans, (float(np.mean(scored)) if scored else 0.0),
            float(np.mean(ep_rewards)) if ep_rewards else float("nan"))
def run(args) -> int:
    """Run the K arms, seed-matched, and report the episode success rate of each."""
    import terrains  # noqa: PLC0415

    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    from fm_noise_attribution import build_arm  # noqa: PLC0415

    if args.catalogue:
        # Matches evaluate.py's own wiring exactly: STAIR_WALLS is a module global read at
        # course-build time, not a per-course field, so every hand-authored stairs course in
        # one run shares this setting.
        terrains.STAIR_WALLS = bool(args.stair_walls)
        print(f"[live] stair walls: {'ON (touching the treads)' if args.stair_walls else 'OFF'}")

    ks = [int(x) for x in args.k.split(",")]
    use_q = bool(args.q_ckpt)
    if args.lookahead < 1:
        raise SystemExit("--lookahead must be >= 1")
    if args.lookahead > 1:
        if use_q:
            raise SystemExit(
                "--lookahead is only wired into run_episode_selected (the V selector). The Q "
                "selector scores (s, a) at the re-plan itself, so a rolled-out continuation "
                "is not the same object -- refusing rather than silently running myopic.")
        print(f"[live] lookahead {args.lookahead}: each candidate rolls {args.lookahead} "
              f"chunks ({args.lookahead - 1} unbranched continuation) before V is read; only "
              "the FIRST is committed. Cost is lookahead x the K=... arm, so "
              f"k=N,lookahead={args.lookahead} matches k={args.lookahead}N,lookahead=1 on "
              "chunks per decision.")
    if args.steer_kp > 0.0:
        if use_q:
            raise SystemExit(
                "--steer_kp is only wired into run_episode_selected (the V selector, "
                "--critic_ckpt/--state_critic_ckpt); run_episode_q_selected (--q_ckpt) does "
                "not read it yet. Refusing rather than silently running unsteered.")
        print(f"[live] steering ON: P={args.steer_kp:.3f}, wz clipped to "
              f"+-{args.steer_wz_max:.3f} rad/s (runner.steer_yaw_rate). OFF-DISTRIBUTION ON "
              "STAIRS -- the training library commands zero turn rate on every stair clip; "
              "this measures a DIFFERENT thing, not a cleaner one.")
    scorer = None
    if args.state_critic_ckpt and ks != [1]:
        scorer = StateCritic(args.state_critic_ckpt, device=args.device,
                             fm_override=args.fm_pt)
    elif use_q and ks != [1]:
        scorer = QScorer(args.q_ckpt, device=args.device, fm_override=args.fm_pt)
    elif ks != [1]:
        if not args.critic_ckpt:
            raise SystemExit("need --critic_ckpt (V selector) or --q_ckpt (Q selector)")
        # `--fm_pt` applies here too, not just to the Q arms: a V_priv checkpoint names the
        # generator it was fitted against by ABSOLUTE path, so a critic trained on one machine
        # cannot be read on another without it.
        scorer = C.FailureCritic(args.critic_ckpt, fm_pt=(args.fm_pt or None),
                                 device=args.device)
    crit = scorer
    # THE TRACKING SPEC, resolved once. The quality head predicts a normalized return under
    # one set of sigmas, bounds and gamma; the selector MEASURES a reward under a spec and adds
    # the two. If they disagree it is summing two different quantities, and both are plausible
    # floats -- so the checkpoint's own stamp wins, `--tracking_spec` is only a fallback for a
    # pre-stamp checkpoint, and a disagreement between the two is refused rather than resolved.
    # THE TRACKING SPEC IS THE FAILURE RULE, so it is resolved for EVERY arm including a
    # K=1-only run. Gating it on `ks != [1]` -- which this did -- leaves a K=1 baseline on the
    # fall-only rule while every K>1 candidate is on the strict one, and the paired delta then
    # measures the rule instead of the selection. That is the same defect as the missing
    # `probe=` in `run_episode_selected`, one level up, and it would have silently invalidated
    # the corrected baseline re-run that fix exists to enable.
    #
    # A K=1 run loads no critic, so its spec can only come from `--tracking_spec`. Absent
    # everywhere, BOTH arms fall back to fall-only, which is at least symmetric -- but it is
    # announced loudly, because a run with no strict number looks identical to one with.
    from deploy.tracking_error import TrackingErrorSpec  # noqa: PLC0415

    track_gamma = 0.95
    stamped = getattr(crit, "tracking_spec", None)
    cli_spec = None
    if args.tracking_spec:
        with open(args.tracking_spec) as fh:
            cli_spec = TrackingErrorSpec.from_dict(json.load(fh))
    if stamped is not None and cli_spec is not None and stamped != cli_spec:
        raise SystemExit(
            f"--tracking_spec {args.tracking_spec!r} disagrees with the spec stamped in "
            f"{os.path.basename(args.critic_ckpt)}. The stamped one is what the quality head "
            "was FITTED under; passing a different one silently changes what the measured "
            "reward means. Drop the flag, or refit.")
    track_spec = stamped if stamped is not None else cli_spec
    if track_spec is None:
        print("[live] WARNING: no tracking spec (none stamped in the critic, none passed). "
              "BOTH arms fall back to the fall-only rule -- no strict number is produced, and "
              "nothing here is comparable to a run that had one.", flush=True)
    else:
        track_gamma = float(track_spec.gamma)
        print(f"[live] failure rule: STRICT -- lateral {track_spec.max_lateral_m:.2f} m / "
              f"height {track_spec.max_height_m:.2f} m / tilt "
              f"{math.degrees(track_spec.max_tilt_rad):.0f} deg vs the served reference, "
              f"TERMINATING, on every arm including K=1. spec "
              f"{'FITTED' if not track_spec.is_bootstrap else 'BOOTSTRAP'}", flush=True)
    if ks != [1]:
        if args.w_fail < 1.0:
            if track_spec is None:
                raise SystemExit(
                    "--w_fail < 1 needs the tracking spec the quality head was fitted under. "
                    f"{os.path.basename(args.critic_ckpt or '?')} carries no `tracking_spec` "
                    "stamp (it predates the stamping fix). Pass --tracking_spec <spec.json> "
                    "-- the same file the recording was made with -- or refit.")
            if getattr(crit, "has_quality", False) is False:
                raise SystemExit(
                    f"--w_fail {args.w_fail} asks for the quality term but "
                    f"{os.path.basename(args.critic_ckpt or '?')} has no quality head. Refit "
                    "with --quality_bins > 0, or run the control arm at --w_fail 1.")
            print(f"[live] selection score: J = {1.0 - args.w_fail:.3f}*G_hat - "
                  f"{args.w_fail:.3f}*P_fail  (failure weighted "
                  f"{args.w_fail / max(1e-9, 1.0 - args.w_fail):.2f}x quality), gamma "
                  f"{track_gamma:.3f}", flush=True)
        else:
            print("[live] selection score: P_fail only (--w_fail 1.0) -- the CONTROL arm, "
                  "bit-identical to every number measured before the quality head existed.",
                  flush=True)
    os.makedirs(args.out, exist_ok=True)
    recorder = None
    if args.random_select and args.same_noise:
        raise SystemExit(
            "--random_select and --same_noise are two DIFFERENT controls and composing them "
            "gives a third thing that answers neither question: with one shared draw every "
            "candidate is identical, so choosing among them at random is not a control on the "
            "ranking, it is just the same-noise arm. Run them separately.")
    if args.random_select and ks == [1]:
        raise SystemExit("--random_select needs a K>1 arm; at K=1 there is nothing to choose.")
    if args.record_candidates and not args.record_selected:
        raise SystemExit("--record_candidates writes into --record_selected's file; pass one.")
    if args.record_selected:
        if ks == [1]:
            raise SystemExit(
                "--record_selected needs a K>1 arm: at K=1 the selector makes no choice, so "
                "the 'selected' chunk is just the generator's own sample and fine-tuning on "
                "it would be a no-op self-distillation.")
        critic_path = args.critic_ckpt or args.state_critic_ckpt or args.q_ckpt
        # A PATH IS NOT A FINGERPRINT. This repo has been bitten three times by a default path
        # that still resolved while naming the wrong artifact (`_DEFAULT_RUN_DIR`,
        # `G1_DEPLOY_POLICY_DIR`, the hand-kept `g1_29j.urdf`), each time silently. A recording
        # that is going to be trained on has to be able to PROVE which critic ranked it, so the
        # hash travels beside the name -- and so does the critic KIND, since V, Q and the state
        # critic rank on different quantities and their checkpoints are not distinguishable by
        # filename.
        recorder = SelectedChunkWriter(args.record_selected, {
            "tracker_onnx": os.path.abspath(args.tracker_onnx),
            "tracker_sha256": _sha256(args.tracker_onnx),
            "fm_onnx": os.path.abspath(args.fm_onnx),
            "fm_sha256": _sha256(args.fm_onnx),
            "critic": os.path.abspath(critic_path) if critic_path else "",
            "critic_sha256": _sha256(critic_path),
            "critic_kind": ("Q" if args.q_ckpt else
                            "state" if args.state_critic_ckpt else "V"),
            "clip_lib": os.path.abspath(args.clip_lib),
            "clip_lib_sha256": "",   # multi-GB; the path + manifest identify it
            "k": args.k, "episodes": int(args.episodes), "speed": float(args.speed),
            "approach": float(args.approach), "steer_kp": float(args.steer_kp),
            "lookahead": int(args.lookahead),
            "tiles": args.tiles, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, record_candidates=bool(args.record_candidates))
    rows, t0 = [], time.time()

    if args.catalogue:
        units = [("course", int(c)) for c in args.catalogue.split(",") if c.strip()]
    elif args.clips:
        units = [("clip", int(c)) for c in args.clips.split(",") if c.strip()]
    else:
        units = [("tile", int(t)) for t in args.tiles.split(",")]
    for kind, uid in units:
        if kind == "clip":
            # ISAAC-MATCHED: the clip's own terrain, start (pose AND velocity), goal, duration
            # and command. No standing approach, and therefore no cmd_hold_s -- the robot
            # begins already moving at the clip's velocity, so there is no standing-start
            # transient to settle. That knob is worth +-0.26 on these courses and this is how
            # it is removed rather than tuned.
            course = terrains.isaac_clip_course(args.clip_lib, uid,
                                                max_offset=args.max_offset,
                                                goal_regions=args.goal_regions)
            course = dataclasses.replace(course, name=f"clip{uid}{args.scene_tag}")
        elif kind == "course":
            # HAND-AUTHORED: terrains.catalogue(), the general mjeval cell/grid harness's own
            # terrain -- no clip ever walked it, so there is no true terminal pose to anchor
            # the critic on (see the module's Course.goal_anchor_xyz docstring). target_xy is
            # what IS available -- the far platform's centre, already used for this course's
            # own mjeval success rule -- and using it for the critic too is a KNOWN, measured
            # approximation (0.44-3.17 m off a real clip anchor by family, see
            # mjeval.goal_anchor), not a substitute for one. isaac_tol stays unset, so the
            # SUCCESS rule (unlike the critic's goal features) is untouched by this.
            course = terrains.catalogue()[uid]
            anchor_z = course.floor_z + max(course.flight_height_m, 0.0) + terrains.SPAWN_BASE_Z
            course = dataclasses.replace(
                course, name=f"cat{uid}{args.scene_tag}",
                goal_anchor_xyz=(course.target_xy[0], course.target_xy[1], anchor_z),
                goal_anchor_source="target_xy_approx")
            print(f"[live] course {uid} ({course.klass}): critic anchor APPROXIMATED from "
                  "target_xy -- expect a 0.44-3.17 m bias against a real clip anchor.",
                  flush=True)
        else:
            # The fm_variance_probe configuration: tile course, 2.5 m standing approach,
            # constant --speed.
            course = terrains.isaac_tile_course(args.clip_lib, uid, speeds=(args.speed,),
                                                approach=args.approach,
                                                goal_anchor_max_offset=args.max_offset,
                                                klass=(args.klass or None))
            course = dataclasses.replace(course, name=f"tile{uid}{args.scene_tag}")
        cfg, policy, fm_lag, spawn, rig = build_arm(
            type("A", (), dict(tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
                               out=args.out, providers=args.providers,
                               fall_height=args.fall_height,
                               fall_tilt_deg=args.fall_tilt_deg,
                               cmd_hold_s=(0.0 if kind == "clip" else args.cmd_hold_s)))(),
            course, history_from=args.history_from,
            allow_privileged_extero=args.allow_privileged_extero)
        if kind == "clip":
            spawn = mjrunner.clip_spawn(course, policy.get_joint_names())
            rig = mjrunner.CourseRig(course, policy, cfg, spawn)
            cmd = np.array(course.params["clip_command"][0], dtype=np.float64)
        else:
            cmd = np.array([args.speed, 0.0, 0.0])
        tile = int(course.params.get("terrain_id", uid))
        rule = "ISAAC (half-plane + lateral + height)" if course.isaac_tol else "mjeval (band + gate)"
        print(f"\n[live] {kind} {uid} tile {tile} {course.klass} cmd {cmd[0]:.2f} m/s "
              f"budget {course.budget_s:.2f}s anchor {np.round(course.goal_anchor_xyz, 3)}"
              f"\n       arrival rule: {rule}", flush=True)

        for k in ks:
            wins = 0
            for ep in range(args.seed_offset, args.seed_offset + args.episodes):
                fn = run_episode_q_selected if use_q else run_episode_selected
                extra = ({} if use_q else
                        {"steer_kp": args.steer_kp, "steer_wz_max": args.steer_wz_max,
                         "same_noise": args.same_noise,
                         "random_select": args.random_select,
                         # Only K>1 selects; a K=1 arm in the same invocation must not write.
                         "recorder": (recorder if k > 1 else None),
                         "episode_uid": (int(tile) * 1000000 + int(k) * 100000
                                         + int(round(args.speed * 10)) * 1000 + int(ep)),
                         "family_code": int(uid) // 20,
                         "lookahead": args.lookahead,
                         "cmd_hold_s": args.cmd_hold_s,
                         "w_fail": args.w_fail,
                         "spec": track_spec, "gamma": track_gamma})
                v, f, nr, sc, q = fn(
                    rig, policy, fm_lag, spawn, cmd, cfg, crit, course.goal_anchor_xyz,
                    k=k, seed=ep, mjrunner=mjrunner, **extra)
                wins += int(v == "success")
                # `track_reward` is reported for EVERY arm, K=1 included. Success stops being
                # the only objective the moment a dense tracking reward exists: a selector
                # that buys survival by tracking worse is invisible in a success rate, and
                # that trade is exactly what the quality head was added to expose. NaN means
                # no probe (no tracking spec) or an episode that never left the opening hold.
                rows.append({"tile": tile, "unit": f"{kind}{uid}", "k": k,
                         "episode": ep, "verdict": v,
                             "failure": f, "n_replans": nr, "spread": float(sc),
                             "track_reward": float(q),
                             "selector": "Q" if use_q else "V"})
                done = ep - args.seed_offset + 1
                if done % max(1, args.episodes // 10) == 0:
                    print(f"  tile {tile} K={k}: {done}/{args.episodes} "
                          f"success {wins / done:.3f}  ({(time.time() - t0) / 60:.1f} min)",
                          flush=True)
            print(f"  tile {tile} K={k}: SUCCESS {wins / args.episodes:.3f} "
                  f"({wins}/{args.episodes})", flush=True)

    with open(os.path.join(args.out, "live_select.json"), "w") as fh:
        json.dump({"rows": rows, "args": vars(args)}, fh, indent=1)
    print(f"\n{'tile':>5} {'K':>3} {'episodes':>9} {'success':>9}")
    for tile in sorted({r["tile"] for r in rows}):
        for k in ks:
            sel = [r for r in rows if r["tile"] == tile and r["k"] == k]
            if sel:
                s = sum(r["verdict"] == "success" for r in sel) / len(sel)
                print(f"{tile:>5} {k:>3} {len(sel):>9} {s:>9.3f}")
    print("\nK=1 is the harness check: with one candidate the selector has no choice, so its "
          "\ntrajectory must match the plain runner exactly. Its success rate is this "
          "benchmark's\nbaseline -- NOT fm_variance_probe's 0.798/0.816, which were measured "
          "with a different\ngenerator (lag48_aug, not the nojumpoff one the critic was "
          "fitted over).")
    if recorder is not None:
        n = recorder.close()
        print(f"[live] recorded {n:,} SELECTED chunks -> {args.record_selected}")
    print(f"\n[live] {len(rows)} episodes -> {args.out}/live_select.json")
    return 0


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--critic_ckpt", default="", help="V(s) critic, for the rollout selector.")
    p.add_argument("--w_fail", type=float, default=0.75,
                   help="Weight on the survival head in the selection score: "
                        "J = (1-w)*G_hat - w*P_fail, so the failure:quality RATIO is "
                        "w/(1-w) -- DEFAULT 0.75 is 3x, 0.8 is 4x, 0.833 is 5x. Needs a "
                        "--quality_bins>0 checkpoint. **--w_fail 1 is the CONTROL arm**: it "
                        "short-circuits to P_fail and reproduces every number measured before "
                        "the quality head existed, bit for bit. Note the authority of each "
                        "input across its [0,1] range is w on P_fail, (1-w)*gamma on G_hat "
                        "and (1-w)*(1-gamma) on the MEASURED chunk -- so the predicted tail "
                        "is gamma/(1-gamma) = 19x the measured chunk at gamma 0.95, which is "
                        "what a 1-step lookahead is rather than a weighting choice.")
    p.add_argument("--tracking_spec", default="",
                   help="tracking_spec.json, ONLY as a fallback for a critic checkpoint that "
                        "predates the spec stamp. When the checkpoint carries one, that wins "
                        "and a disagreement is refused: the head's G_hat is a return under "
                        "the spec it was fitted with, and measuring the reward under another "
                        "sums two different quantities.")
    p.add_argument("--q_ckpt", default="",
                   help="A fit_q checkpoint. Selects with Q(s, a), which scores each candidate "
                        "chunk WITHOUT executing it -- K generator calls and no physics, "
                        "against the V selector needing K rollouts of 0.24 s. This is also the "
                        "only one of the two a robot could run: it cannot execute eight "
                        "futures and undo seven.")
    p.add_argument("--fm_pt", default="", help="Local generator .pt, when the Q checkpoint "
                                               "names a path from another machine.")
    p.add_argument("--state_critic_ckpt", default="",
                   help="An ABLATED fit_q checkpoint, driven as V(s) by the ROLLOUT selector "
                        "(candidates are executed 0.24 s and the resulting state is scored). "
                        "Use this to run the same selector that worked with V_priv, but with "
                        "a V trained on the fit_q split.")
    p.add_argument("--tracker_onnx", required=True)
    p.add_argument("--fm_onnx", required=True)
    p.add_argument("--clip_lib", required=True)
    p.add_argument("--tiles", default="6,20")
    p.add_argument("--allow_privileged_extero", action="store_true",
                   help="Run a generator conditioned on the PRIVILEGED dense height scan "
                        "(121x121 = 14641 rays, exact and noise-free). MuJoCo can honour "
                        "that contract -- it casts the same grid against the same terrain, "
                        "~20 ms/step -- and HARDWARE cannot, which is why the flag is off by "
                        "default and nothing on the robot path passes it. Anything measured "
                        "under it is a statement about the model GIVEN PERFECT TERRAIN, and "
                        "never about a deployment.")
    p.add_argument("--history_from", default="generator",
                   choices=["generator", "measured"],
                   help="What fills the generator's N history rows: its own served nodes "
                        "('generator', the deployed closed loop) or the ROBOT'S OWN achieved "
                        "state ('measured'), bar the base position and linear velocity, "
                        "which no robot can measure and which still come from the served "
                        "node. Run it against a 'generator' arm at the SAME --seed_offset "
                        "and tiles: the arms are then seed-matched per episode and the "
                        "difference is the conditioning. Off-distribution by 0.54-0.69 "
                        "sigma (scripts/flow_matching/measured_history_gap.py), so read a "
                        "result against that rather than as a clean grounding experiment.")
    p.add_argument("--clips", default="",
                   help="comma list of CLIP ids. Builds Isaac-matched per-clip courses -- the "
                        "clip's own terrain, start pose AND velocity, goal, duration and "
                        "command -- instead of tile courses with a standing approach. "
                        "Overrides --tiles, --speed and --cmd_hold_s.")
    p.add_argument("--catalogue", default="",
                   help="comma list of indices into terrains.catalogue() (0-19), the "
                        "hand-authored courses used by the general mjeval cell/grid harness -- "
                        "separate from --tiles/--clips, which read a real clip-library tile. "
                        "Overrides --tiles and --clips. These courses carry NO goal_anchor_xyz "
                        "(no clip ever walked them to record one), so the critic's goal "
                        "features are built from target_xy instead -- a KNOWN, measured "
                        "approximation, biased 0.44-3.17 m depending on family versus the real "
                        "clip-anchor a tile course would give (see mjeval.goal_anchor). Every "
                        "row from this arm is tagged goal_anchor_source=target_xy_approx.")
    p.add_argument("--goal_regions", default="",
                   help="a fitted goal_regions.json; makes arrival ISAAC's rule (downrange "
                        "half-plane + lateral + height bands, no traversal gate) rather than "
                        "mjeval's rectangle plus gate. Clip courses only.")
    p.add_argument("--no_stair_walls", dest="stair_walls", action="store_false", default=True,
                   help="--catalogue only, matches evaluate.py's own flag: build hand-authored "
                        "stairs courses without the visual stairwell walls.")
    p.add_argument("--steer_kp", type=float, default=0.0,
                   help="P gain (1/s) on heading error toward the goal, recomputed every "
                        "control step via runner.steer_yaw_rate; 0 (default) holds the flat "
                        "commanded twist. V selector only (--critic_ckpt/--state_critic_ckpt) "
                        "-- refused with --q_ckpt, which does not read it. "
                        "OFF-DISTRIBUTION ON STAIRS: the training library commands zero turn "
                        "rate on every stair clip (3500 of them), so a stairs run with this "
                        "on is a DIFFERENT measurement, not a cleaner one.")
    p.add_argument("--steer_wz_max", type=float, default=0.68,
                   help="Symmetric clip on the steering yaw rate, rad/s. Matches "
                        "runner.STEER_WZ_MAX. Only used with --steer_kp.")
    p.add_argument("--lookahead", type=int, default=1,
                   help="Chunks rolled per candidate before V is read. 1 (default) is the "
                        "myopic selector. n>1 continues each candidate for n-1 further "
                        "chunks with NO branching -- one fresh draw each, the generator's own "
                        "policy -- and bootstraps V from the end, so the score is an "
                        "(n-1)-step return plus a terminal value. ONLY THE FIRST CHUNK IS "
                        "COMMITTED, so the decision frequency is unchanged and depth is the "
                        "only variable; a terminal event in the continuation scores the "
                        "branch but never ends the episode. Costs lookahead x the arm, which "
                        "makes --k 4 --lookahead 2 and --k 8 --lookahead 1 an equal-compute "
                        "breadth-vs-depth pair (8 chunks per decision each). V selector only; "
                        "ignored on the K=1 path, which makes no choice.")
    p.add_argument("--record_selected", default="",
                   help="Write the CHOSEN chunk at every re-plan to this .h5, with the "
                        "conditioning it was drawn from -- the filtered-BC dataset, consumed "
                        "by finetune_generator.py with no --labels. K>1 arms only: at K=1 "
                        "there is no selection, so a recording would just be the generator's "
                        "own output and training on it is a no-op.")
    p.add_argument("--record_candidates", action="store_true",
                   help="With --record_selected, also store every REJECTED candidate's chunk, "
                        "its critic score and its per-chunk verdict. Diagnostic only, never "
                        "training data: ft_canary.PairGapCanary reads them to separate 'the "
                        "model is learning the selector's ranking' from 'self-distillation "
                        "improved everything together', which no loss on the selected chunks "
                        "alone can tell apart. Nearly free -- candidates share one "
                        "conditioning, so a row grows by K plans, not K exteroception rows.")
    p.add_argument("--random_select", action="store_true",
                   help="CONTROL: roll and score all K candidates exactly as usual, then "
                        "commit a UNIFORMLY RANDOM one instead of the argmin. Differs from the "
                        "real arm in the choice rule ALONE, so the gap between them is the "
                        "critic's ranking and nothing else. Preferred over --same_noise for "
                        "that question: same-noise also changes the RNG keying relative to "
                        "K=1, which is worth a measured +0.056 on its own and confounds the "
                        "two effects.")
    p.add_argument("--same_noise", action="store_true",
                   help="CONTROL: hand every candidate the SAME draw, so the selector has "
                        "zero signal by construction and any rise over the REAL arm AT THE "
                        "SAME K is manufactured by the harness. This is the check the Isaac "
                        "per-state best-of-K harness failed (its control matched the real "
                        "arm). Compare it against real K, NOT against K=1: both K>1 paths "
                        "re-seed per re-plan while K=1 advances its rng naturally from the "
                        "episode seed, so same-noise K=4 and K=1 run different noise tapes "
                        "and differ for that reason alone. Bit-exactness of the "
                        "snapshot/restore is what the K=1 arm tests, not this.")
    p.add_argument("--klass", default="",
                   help="Force the course class (a terrains.CLASS_CODES key) instead of "
                        "inferring it from the tile's manifest family. TILE courses only. "
                        "Needed for families absent from terrains.FAMILY_KLASS -- notably "
                        "jump_off_front, which builds fine with klass=box and recovers a real "
                        "clip anchor on 8 of its 15 tiles (75,76,77,81,82,83,86,87); the "
                        "other 7 fail earlier, on a spawn that lands outside the tile "
                        "footprint. Note the traversal gate and fall datum then read under "
                        "box semantics, which is what makes this explicit rather than "
                        "inferred.")
    import terrains as _terrains  # noqa: PLC0415  the shared CMD_HOLD_S constant

    p.add_argument("--cmd_hold_s", type=float, default=_terrains.CMD_HOLD_S,
                   help="Seconds of zero command before stepping to --speed, on TILE courses. "
                        "Defaults to terrains.CMD_HOLD_S (the one constant every mjeval driver "
                        "reads) rather than a local literal. Clip courses force 0: they start "
                        "already moving.\n"
                        "NOT COSMETIC. Commanding full speed on the first control tick puts the "
                        "robot through a spawn transient that is itself a failure mode: on tile "
                        "20 at 2.4 m/s the base generator scores 0.275 at hold 0 and 0.675 at "
                        "hold 0.5 -- the hold is worth more than the entire fine-tune effect "
                        "measured there. Anything recorded before this became the default was "
                        "run at 0 and is NOT comparable.")
    p.add_argument("--k", default="1,4,8", help="comma list of candidate counts.")
    p.add_argument("--episodes", type=int, default=200,
                   help="per arm per tile, seed-matched across arms (seed = episode index).")
    p.add_argument("--seed_offset", type=int, default=0,
                   help="first episode seed. Shards an arm across processes WITHOUT breaking "
                        "the seed matching, since the seed is the episode index and not a "
                        "position in a per-process sequence: shard [0,100) and [100,200) of "
                        "K=8 cover exactly the seeds K=1 ran, so the arms still pair.")
    p.add_argument("--speed", type=float, default=0.8)
    p.add_argument("--approach", type=float, default=2.5)
    p.add_argument("--max_offset", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--scene_tag", default="_ls")
    return run(p.parse_args())




class QScorer:
    """Scores a candidate CHUNK from the current state, without executing it.

    This is the difference between a ``Q`` selector and the ``V`` one above, and it is not a
    detail.  ``V(s')`` needs a state to score, so every candidate must first be ROLLED 0.24 s
    -- 95.6 ms each, measured.  ``Q(s, a)`` scores the chunk directly, so the candidates cost
    one generator call apiece and no physics at all.  That makes the Q selector roughly an
    order of magnitude cheaper AND deployable: a robot cannot execute eight futures and undo
    seven of them, but it can sample eight plans and pick one.

    Args:
        ckpt: A ``fit_q`` checkpoint.
        device: Torch device.
        fm_override: Path to the generator ``.pt`` the checkpoint borrows, when the recorded
            one does not resolve here -- a checkpoint trained on another machine names ITS
            path. Overriding is safe only because the trunk is frozen and both sides must be
            the same file; the loader below refuses a DIFFERENT generator by content hash
            rather than trusting the caller.
    """

    def __init__(self, ckpt: str, device: str = "cuda", fm_override: str = "") -> None:
        import copy as _copy  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from robot_rl.flow_matching import online, qvalue  # noqa: PLC0415

        self._torch = torch
        self._qv = qvalue
        self.device = device
        k = torch.load(ckpt, map_location="cpu", weights_only=False)
        if bool(k.get("ablate_action", False)):
            raise ValueError(
                f"{ckpt} is the ACTION-ABLATED control arm -- its action tokens were zeroed "
                "during training, so it is a V(s) in a Q's clothing and would rank every "
                "candidate at a state identically. Selecting on it is a no-op that looks "
                "like a working selector.")
        fm = str(k["fm_checkpoint"])
        if fm_override:
            import hashlib as _h  # noqa: PLC0415

            def _sha(q):
                h = _h.sha256()
                with open(q, "rb") as fh:
                    for c in iter(lambda: fh.read(1 << 20), b""):
                        h.update(c)
                return h.hexdigest()
            if os.path.exists(fm) and _sha(fm) != _sha(fm_override):
                raise ValueError(
                    f"fm_override {fm_override!r} is a DIFFERENT generator from the one this Q "
                    f"was fitted over ({fm!r}). The trunk is frozen, so a different one loads "
                    "with the right shapes and three wrong normalizers -- silently.")
            fm = fm_override
        elif not os.path.exists(fm):
            raise FileNotFoundError(
                f"the Q checkpoint names {fm!r}, which does not exist here. It was probably "
                "trained on another machine; pass fm_override with the local copy.")
        self.ctx = online.load_fm_context(fm, device=device)
        self.model = qvalue.FlowQModel(
            _copy.deepcopy(self.ctx.model), action_dim=int(k["action_dim"]),
            action_steps=int(k["action_steps"]), scan_hw=tuple(k["scan_hw"]),
            proprio_dim=int(k["proprio_hw"][-1]), proprio_steps=int(k["proprio_hw"][0]),
            context_dim=int(k["context_dim"])).to(device)
        self.model.load_state_dict(k["model"])
        self.model.eval()
        self.scan_hw = tuple(k["scan_hw"])
        self.steps = torch.as_tensor(self.ctx.cmd_steps, dtype=torch.long, device=device)

    def score(self, inp, window) -> float:
        """``P(fail | s, a)`` for one candidate; lower is better.

        Normalizer order mirrors ``fit_q._batch`` exactly, including the action being
        normalized with ``state_norm`` -- the emitted plan nodes live in the same space as the
        history nodes, which is what that normalizer was fitted on.

        Args:
            inp: A :class:`critic.CriticInputs` for the current state.
            window: ``(F, D)`` the candidate chunk.

        Returns:
            The score.
        """
        import numpy as _np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from robot_rl.flow_matching import transforms, value  # noqa: PLC0415

        from goal_anchor import GOAL_SCALE_M  # noqa: PLC0415

        t = lambda a, d=torch.float32: torch.as_tensor(  # noqa: E731
            _np.asarray(a), dtype=d, device=self.device).unsqueeze(0)
        c = self.ctx
        cmd = (c.cmd_norm.normalize(t(inp.cmd_future)[:, self.steps, :])
               if c.cmd_dim > 0 else None)
        goal = _np.nan_to_num(_np.asarray(inp.goal, dtype=_np.float64) / GOAL_SCALE_M,
                              nan=0.0, posinf=0.0, neginf=0.0)
        st = value.ValueInputs(
            history=c.state_norm.normalize(t(inp.history)),
            extero=c.extero_norm.normalize(t(inp.extero).reshape(1, -1)),
            cmd_tokens=cmd,
            history_null=t(_np.array([inp.history_null]), torch.bool).reshape(1),
            scan=transforms.sanitize_extero(t(inp.scan), "height_scan").reshape(1, *self.scan_hw),
            proprio=t(inp.proprio),
            context=t(_np.concatenate([goal, _np.asarray(inp.twist, dtype=_np.float32)])))
        a = c.state_norm.normalize(t(window)[:, self.steps, :])
        with torch.no_grad():
            return float(self.model.p_fail(self._qv.QInputs(state=st, action=a)).item())


class StateCritic:
    """A ``fit_q`` ABLATED checkpoint used as ``V(s)``, for the rollout selector.

    The ablated arm was trained with its action tokens zeroed, so it is a state critic that
    happens to be a :class:`FlowQModel`.  This exposes it behind the same two members the
    rollout selector reads from ``critic.FailureCritic`` -- ``privileged`` and ``p_fail`` --
    so the same loop can be driven by either.

    **The action is passed as ZEROS, matching training.**  Feeding it a real chunk would put
    the model off the distribution it was fitted on: it never saw a non-zero action, so
    whatever it does with one is unconstrained rather than merely unhelpful.

    Args:
        ckpt: An ablated ``fit_q`` checkpoint.
        device: Torch device.
        fm_override: Local generator ``.pt``, when the checkpoint names another machine's path.

    Raises:
        ValueError: If the checkpoint is NOT ablated.  A live-action Q used here would be
            scored with a zero action at every candidate, making it a state critic by
            accident and silently discarding the thing it was trained for.
    """

    def __init__(self, ckpt: str, device: str = "cuda", fm_override: str = "") -> None:
        import torch  # noqa: PLC0415

        k = torch.load(ckpt, map_location="cpu", weights_only=False)
        if not bool(k.get("ablate_action", False)):
            raise ValueError(
                f"{ckpt} is a LIVE-action Q, not the ablated control. Driving the rollout "
                "selector with it would pass a zero action at every candidate -- turning it "
                "into a state critic by accident and throwing away exactly what it was "
                "trained to add. Use QScorer for a live Q.")
        # Built through a loader that skips QScorer's ablation guard: that guard protects the
        # Q selector, where an ablated model would rank every candidate the same. Here the
        # ablation is the POINT.
        self._q = _load_q_unchecked(ckpt, device, fm_override)
        self.privileged = True
        self._torch = torch

    def p_fail(self, inp) -> float:
        """``P(fail | s)`` with the action held at zero.

        Args:
            inp: A :class:`critic.CriticInputs`.

        Returns:
            The score.
        """
        import numpy as _np  # noqa: PLC0415

        zero = _np.zeros((64, self._q.model.action_in[0].in_features), dtype=_np.float32)
        return self._q.score(inp, zero)


def _load_q_unchecked(ckpt: str, device: str, fm_override: str):
    """A :class:`QScorer` built without the ablation guard.

    Args:
        ckpt: The checkpoint.
        device: Torch device.
        fm_override: Local generator path.

    Returns:
        The scorer.
    """
    import copy as _copy  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from robot_rl.flow_matching import online, qvalue  # noqa: PLC0415

    o = QScorer.__new__(QScorer)
    k = torch.load(ckpt, map_location="cpu", weights_only=False)
    fm = fm_override or str(k["fm_checkpoint"])
    o._torch, o._qv, o.device = torch, qvalue, device
    o.ctx = online.load_fm_context(fm, device=device)
    o.model = qvalue.FlowQModel(
        _copy.deepcopy(o.ctx.model), action_dim=int(k["action_dim"]),
        action_steps=int(k["action_steps"]), scan_hw=tuple(k["scan_hw"]),
        proprio_dim=int(k["proprio_hw"][-1]), proprio_steps=int(k["proprio_hw"][0]),
        context_dim=int(k["context_dim"])).to(device)
    o.model.load_state_dict(k["model"])
    o.model.eval()
    o.scan_hw = tuple(k["scan_hw"])
    o.steps = torch.as_tensor(o.ctx.cmd_steps, dtype=torch.long, device=device)
    return o


def run_episode_q_selected(rig, policy, fm_lag, spawn, cmd, cfg, scorer, anchor, *,
                           k: int, seed: int, mjrunner):
    """One episode where each re-plan picks the best of ``k`` chunks by ``Q(s, a)``.

    Unlike :func:`run_episode_selected`, **no candidate is executed to be scored**.  At a
    re-plan the generator is sampled ``k`` times against the SAME conditioning, each sample is
    scored where it stands, and only the winner is committed.  Cost is ``k`` generator calls,
    not ``k`` chunk rollouts.

    The state half of ``Q``'s input is built ONCE and reused across candidates: it is the
    conditioning at this re-plan, which every candidate shares by construction.  Rebuilding it
    per candidate would additionally re-cast depth, which mutates the lag ring -- the bug that
    made the V selector look worse than no selection at all.

    Args:
        rig: The course rig.
        policy: The loaded pair.
        fm_lag: The camera-lag ring.
        spawn: Initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        scorer: A :class:`QScorer`, or ``None`` for ``k == 1``.
        anchor: Goal anchor for the goal features.
        k: Candidates per re-plan.
        seed: Episode seed.
        mjrunner: The ``runner`` module.

    Returns:
        ``(verdict, failure, n_replans, mean_spread)`` where ``mean_spread`` is the mean
        max-minus-min of ``Q`` across candidates -- if that is ~0 the model is not
        discriminating and any success change is not selection.
    """
    import critic as C  # noqa: PLC0415
    import mujoco  # noqa: PLC0415

    from gen_tracker_runner import build_state  # noqa: PLC0415

    robot, course = rig.robot, rig.course
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)
    ring = C.ProprioRing()
    z0 = float(robot.mj_data.qpos[2])
    zex = [z0, z0]
    n_replans, spreads = 0, []
    last_plan_id = -1

    while robot.mj_data.time <= course.budget_s:
        state = build_state(robot, policy, cmd, rig.depth_cams, want_depth_hits=False,
                            fm_depth_cams=rig.fm_depth_cams, fm_lag=fm_lag)
        ring.push(state["joint_pos"], state["joint_vel"], state["base_ang_vel"],
                  C.projected_gravity_b(state["root_quat"]))
        # Snapshot BEFORE create_obs. `_ctrl_step` is the re-plan cadence counter, so a
        # snapshot taken AFTER it has already advanced restores to MID-CHUNK -- and a
        # create_obs there does not re-plan, it consumes the plan already made. The
        # "candidates" are then different offsets into ONE plan rather than different plans:
        # their windows genuinely differ, so nothing looks wrong, but plan_id never
        # increments and the selector is choosing among readings of the same decision.
        pre = Snapshot(robot, policy, fm_lag)
        obs = policy.create_obs(state)
        plan = policy.last_plan()
        is_replan = plan is not None and int(plan["plan_id"]) != last_plan_id

        if is_replan and k > 1 and scorer is not None:
            last_plan_id = int(plan["plan_id"])
            snap = pre
            # Built once: this is the conditioning at THIS re-plan, shared by every candidate.
            inp = C.CriticInputs.from_runtime(policy, robot, state, anchor=anchor, ring=ring,
                                              want_scan=True)
            w0 = policy.predicted_window()
            cands = [(scorer.score(inp, w0), 0)] if w0 is not None else []
            for c in range(1, k):
                snap.restore(robot, policy, fm_lag)
                policy._rng = np.random.default_rng([seed, n_replans, c])
                policy.create_obs(state)
                w = policy.predicted_window()
                if w is not None:
                    cands.append((scorer.score(inp, w), c))
            if cands:
                sc = [s for s, _ in cands]
                spreads.append(max(sc) - min(sc))
                best_c = min(cands)[1]
                snap.restore(robot, policy, fm_lag)
                # Candidate 0 used whatever rng the episode had reached; reproduce that by
                # restoring rather than re-seeding, so the committed plan is the one scored.
                if best_c:
                    policy._rng = np.random.default_rng([seed, n_replans, best_c])
                obs = policy.create_obs(state)
            n_replans += 1
        elif is_replan:
            last_plan_id = int(plan["plan_id"])
            n_replans += 1

        robot.apply_action(policy.get_action(obs, robot.joint_names))
        if robot.failed():
            return "fall", robot.failure_reason(), n_replans, float(np.mean(spreads or [0.0])), float("nan")
        for _ in range(rig.decim):
            robot.step()
        z = float(robot.mj_data.qpos[2])
        zex[0], zex[1] = max(zex[0], z), min(zex[1], z)
        if arrived(course, robot, zex):
            return "success", "", n_replans, float(np.mean(spreads or [0.0])), float("nan")
    return "timeout", "", n_replans, float(np.mean(spreads or [0.0])), float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
