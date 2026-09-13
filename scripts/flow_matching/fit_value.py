# Copyright 2026 Zachary Olkin. All rights reserved.

"""Fit the trackability critic on closed-loop rollouts, and run the three gates.

Reads what ``scripts/rsl_rl/record_rollouts.py`` wrote -- one row per generator re-plan plus
one per episode -- and fits ``V(s) = -P(fail | s)`` as a Bernoulli failure classifier warm
started from the generator's own conditioning encoder.

**The split is by EPISODE and that is not a preference.**  With a terminal-only reward every
window of an episode carries the SAME label, so a window-level split puts near-duplicate
states either side of it and the held-out metrics measure memorisation.  For the same reason
the effective sample size is close to the EPISODE count, not the window count: 25 windows from
one episode are one label observed from 25 vantage points, and this script prints both numbers
so the second is never mistaken for the first.

ONE variant is fitted (V_priv). A non-privileged `V_onboard` arm and the gap
between them are deleted: the loop never fitted it for a reported result. See
``robot_rl.flow_matching.value``.

Gates, in the order they should be read (``robot_rl.flow_matching.value_gates``):

1. **calibration**, which is testable even where discrimination is impossible;
2. **AUC by PHASE**, never in aggregate -- a critic that has merely learned which tile is hard
   scores a perfect aggregate AUC and a flat, high phase curve;
3. **lead time**, measured against the FALL and not just against the end. If the critic only
   rises once the base is past recovery, the advantage arrives after the chunk that caused the
   failure and nothing is attributable. It also sets ``n`` empirically.

**THE LABEL IS HORIZON-DEPENDENT, and that is the critic's least portable property.**
``failed`` means "this EPISODE failed", and an episode ends where its harness says it does --
in Isaac at ``clip_end`` (``clip_len - 1 - max_offset``), in mjeval at a distance-derived
budget that runs 1.4-3.6x longer. The same state therefore has a genuinely higher P(fail) in
mjeval, so a perfectly transferred critic reads as over-optimistic there with no calibration
error at all. Two consequences worth keeping straight:

* (historical, the survival head this described is deleted) its DISCRIMINATION (AUC) was
  horizon-robust -- inflating everyone's risk barely
  reorders anything -- so ranking-only selection (``w_fail = 1``) is unaffected;
* its CALIBRATION is not, so any cross-harness ECE comparison is confounded, and so is any
  score that MIXES ``p_fail`` with ``G_hat`` and needs their levels commensurable.

``critic_gate.py --isaac_matched`` (since DELETED with the survival head it gated) was an
attempt to match the window and it did not work:
matching the budget without the initial condition and the velocity profile produced 84.7%
timeouts (757/893) against 0.3% for the default budget, because the clip's duration implies
its pace only if you also start at the clip's velocity. The gate now refuses to publish
calibration numbers from a timeout-dominated run.

``TODO(critic-horizon-feature)``: give the critic the REMAINING TIME as a context feature and
fix one horizon everywhere -- e.g. let Isaac episodes run the full 20 s rather than stopping at
``clip_end``, and read the critic at 20 s in every harness. That makes the prediction
conditional on the horizon instead of implicitly tied to whichever harness produced the label,
and it retires ``--isaac_matched`` entirely. **The known cost, which is not small:** the
padded tail is not neutral. The eval harness added ``clip_end`` precisely because without it
"a 300-frame clip becomes 6 s of motion plus 14 s of a frozen pose, and in the FM arm 14 s of
being commanded the last frame's velocity off a finite tile" -- so training the critic to 20 s
would fit most of its label mass on a regime that does not exist at deploy. Worth doing only
with the tail either terminated on its own condition or excluded from the loss.

``TODO(critic-standing-starts)``: collect Isaac rollouts that begin STANDING AT REST, which is
how every mjeval and deploy episode begins and how no training episode does -- the tracker task
spawns on a clip, mid-motion, at the clip's velocity. The ``flat`` family already contains
standing-to-walk transitions (commands starting at 0.0), so the shape exists in the library; it
is the terrain families that never see a standing approach. This is the initial-condition half
of the same mismatch the horizon TODO covers.

Usage::

    python scripts/flow_matching/fit_value.py <rollout stem> \\
        --fm_checkpoint <fm.pt> --out logs/value/<name>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
for _d in (str(_HERE.parent / "rsl_rl"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import rollout_outcomes as ro  # noqa: E402


def _sha256_file(path: str) -> str:
    """SHA-256 of a file, streamed. Stamped beside the generator's path in every checkpoint."""
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

from robot_rl.flow_matching import online, transforms, value, value_gates  # noqa: E402

#: Metres per unit on the goal-offset features. DECLARED, not fitted: a critic's notion of
#: "far from the goal" should be a property of the robot, not of whichever rollout happened to
#: be on disk when the normalizer ran. 5 m spans the approach on every family measured.
GOAL_SCALE_M: float = 5.0

#: Reward-source name of the progress reward (terrain + gated velocity + terminal success),
#: matching ``inspect_terrain_reward.PROGRESS_SOURCE``.  Restated rather than imported: that
#: script pulls in torch and a simulator-free geometry stack, and this one is imported by
#: tests.  A mismatch is caught by ``label_advantage.check_reward_consistency``.
PROGRESS_SOURCE: str = "terrain_progress"

#: Clip for the worst-penalty-so-far context feature, which is then divided by it so the
#: feature lands in ``[0, 1]``.
#:
#: **Clipped, not merely scaled.** The raw terrain penalty is unbounded -- p95 1.72 against a
#: measured max of 47.5 -- and an unbounded input into a linear layer lets one catastrophic
#: chunk dominate the whole feature's scale for every other row in the batch.  2.5 is
#: ``clipped_window_return``'s own ``clip_c``, chosen there at the penalty's p95-p99, so the
#: two places that bound this quantity bound it the same way.
WORST_CLIP: float = 2.5


def _load(stem: str, max_windows: int, tail_chunks: int = 1, *,
          max_episodes: int = 0, seed: int = 0,
          strict_failure: bool = True, reward_npz: str = "", clip_c: float = 0.0,
          window: int = 3, worst_context: bool = False,
          bootstrap_v: str = "", bootstrap_variant: str = "V_priv",
          ) -> tuple[dict, dict, dict]:
    """Read both tables and join them on ``episode_uid``.

    Args:
        stem: The rollout stem.
        max_windows: Cap, so a multi-gigabyte recording can be smoke-tested.
        max_episodes: Keep a random subset of this many episodes (0 = all).
        seed: RNG seed for that subset, so a sizing sweep is reproducible.
        tail_chunks: Trailing chunks the quality target's non-failure tail averages.
        reward_npz: A per-window reward sidecar to fit the quality head on INSTEAD of the
            tracking error -- see :func:`_reward_sidecar`. Empty means the tracking reward,
            which is what every existing artifact was fitted on.
        clip_c: Non-zero switches the target to
            :func:`~robot_rl.flow_matching.advantage.clipped_window_return` over the
            sidecar's RAW penalty, in ``[-1, 0]``.  Zero keeps the ``(1 - gamma)``-normalized
            return in ``[0, 1]``.
        window: Chunks in the clipped window.
        worst_context: Compute the worst-terrain-penalty-so-far context feature. Off by
            default because it widens the critic's input, so a checkpoint that has it must
            say so -- exactly as it must for ``goal_context``.
        strict_failure: Count the STRICT tracking violation as a failure alongside the fall
            crossing.  ``False`` is the LOOSE rule -- the fall alone -- which is what MuJoCo
            scores (``mjeval`` tests ``robot.failed()``: height and tilt, no tracking check).
            It matters enormously on an out-of-distribution course under placeholder
            tolerances: on the box2 grid the strict rule labels 8967 of 9000 episodes failed
            while the loose one labels 8272, because 695 episodes REACHED THE GOAL and tripped
            the 0.2 m / 30 deg bootstrap tolerances on the way -- 33 positives against 728.

    Returns:
        ``(windows, episodes, per_window_labels)``.

    Raises:
        SystemExit: When the join is not exact. A window whose episode has no outcome row
            cannot be labelled, and silently dropping it would quietly change the training
            distribution.
    """
    # Only what the critic reads. `fm_plan` alone is 11.3 kB/window -- 1.6 GB over a 6k
    # episode rollout -- and the value function is a function of the STATE: the chunk enters
    # the advantage through s_{t+n}, never through V. Loading it would cost a third of the
    # file's RAM to hold an array nothing touches.
    W, wa = ro.read_rollout_windows(stem + ".h5", columns=[
        "episode_uid", "replan_index", "family_code", "history", "history_null", "extero",
        "cmd_future", "goal_downrange_m", "goal_lateral_m", "goal_height_err_m",
        # `proprio` and `height_scan` are NOT here: the critic's scan and proprio token
        # streams were deleted in dcea58f0, so requesting them would load two arrays nothing
        # reads -- which is what this column list exists to avoid.
        "base_twist_yaw",
        # The quality head's target. Requested BY NAME like everything else: this loader
        # takes an explicit column list (it never reads `fm_plan`, 11.3 kB/window), so a
        # column that is not named is simply absent and the head silently trains on nothing.
        # That is exactly what happened -- the target code was right and the fit reported no
        # quality metrics at all, twice, because nobody asked for the column.
        #
        # No `try` around it: `read_rollout_windows` RAISES on an absent column, and it
        # already refuses any file whose schema_version is not 2 -- and `track_err_prev` is
        # part of what schema 2 MEANS. So a v2 file without it is malformed, and failing
        # loudly is right.
        "track_err_prev"])
    E, ea = ro.read_rollout_episodes(stem + ".episodes.h5")
    if max_windows and W["episode_uid"].size > max_windows:
        keep = np.arange(max_windows)
        W = {k: v[keep] for k, v in W.items()}
    if max_episodes:
        # RANDOM over episodes, keeping every window of the chosen ones: the label lives on
        # the episode, so a window-level draw would shrink the gradient count without
        # shrinking the information and would answer a different question.
        uids = np.unique(W["episode_uid"])
        if uids.size > max_episodes:
            rng = np.random.default_rng(int(seed))
            chosen = rng.choice(uids, size=int(max_episodes), replace=False)
            keep = np.isin(W["episode_uid"], chosen)
            W = {k: v[keep] for k, v in W.items()}
            print(f"[fit] --max_episodes {max_episodes}: kept "
                  f"{int(keep.sum()):,} of {keep.size:,} windows over "
                  f"{int(np.unique(W['episode_uid']).size):,} episodes")

    live = ~E["censored"].astype(bool)
    reward = ro.terminal_reward(E, strict=strict_failure)
    # `first_bad` is what drop-after-latch below cuts on: the fall crossing or the strict
    # violation, whichever came FIRST. Taken from the same table `terminal_reward` scored, so
    # the label and the cut cannot disagree about when the episode ended.
    first_bad = ro.first_bad_step(E, strict=strict_failure)
    outcome = {int(u): (float(r), bool(l), int(ff), int(fam), int(fb))
               for u, r, l, ff, fam, fb in zip(E["episode_uid"], reward, live,
                                               E["first_fall_step"], E["family_code"],
                                               first_bad)}
    missing = sorted({int(u) for u in W["episode_uid"]} - set(outcome))
    if missing:
        raise SystemExit(
            f"{len(missing)} window episode(s) have no outcome row (e.g. {missing[:5]}). The "
            "two tables must join exactly on episode_uid; dropping the unmatched ones would "
            "change the training distribution without saying so.")
    ep_terrain = {int(u): int(t) for u, t in zip(E["episode_uid"], E["terrain_id"])}
    lab = {
        "failed": np.array([outcome[int(u)][0] == -1.0 for u in W["episode_uid"]]),
        "live": np.array([outcome[int(u)][1] for u in W["episode_uid"]]),
        "first_fall_step": np.array([outcome[int(u)][2] for u in W["episode_uid"]]),
        "family_code": np.array([outcome[int(u)][3] for u in W["episode_uid"]]),
    }
    # DROP EVERY WINDOW AT OR AFTER THE FIRST BAD STEP -- the fall crossing or the strict
    # tracking violation, whichever came first.
    #
    # The recording LATCHES rather than terminating, which is what keeps the threshold an
    # offline knob: lowering a bound just moves the latch earlier and drops more rows, a
    # relabel instead of a re-collection. But every consumer downstream of the recording
    # treats a violation as a TERMINATION, so the training set has to be the one termination
    # would have produced -- windows `0 .. t-1`, labelled by the strict outcome. That is
    # bit-for-bit the same dataset, because under termination the later states never exist.
    #
    # Without this the fit is asked to do two impossible things at once. The post-latch
    # windows LOOK healthy -- the robot recovered, which is why the episode kept going -- yet
    # carry the failure label, so the survival head is fitted on states termination would
    # never have shown it. And the quality target for a PRE-latch window sums chunk rewards
    # from chunks that termination prevents, so `G` is contaminated by post-failure behaviour
    # rather than being absorbing at the failure. Measured on `v4_big`: 33,428 of 446,131
    # windows (7.49%), 20,580 of them after a violation.
    #
    # Placed BEFORE the return computation on purpose: `normalized_returns` recurses backwards
    # over each episode's remaining chunks, so dropping first is what makes the tail absorbing
    # instead of merely zeroed at the end.
    K = int(ea.get("fm_update_every", 12))
    first_bad = np.array([outcome[int(u)][4] for u in W["episode_uid"]])
    step_of = W["replan_index"].astype(np.int64) * K
    alive = ~((first_bad >= 0) & (step_of >= first_bad))
    if not alive.all():
        n_ep_touched = int(np.unique(W["episode_uid"][~alive]).size)
        print(f"[fit] drop-after-latch: {int((~alive).sum())}/{alive.size} windows "
              f"({100 * (~alive).mean():.2f}%) at or after the first bad step, over "
              f"{n_ep_touched} episode(s). These do not exist under termination.")
        # DOES THE CUT REMOVE THE REWARD'S TERMINAL TERM? The progress reward pays its
        # success bonus on the episode's LAST re-plan only, so a cut that lands before that
        # row deletes the bonus rather than truncating a tail -- and value iteration then
        # propagates a reward that has no success in it at all, while every log still reads
        # as a healthy fit.
        #
        # Measured on box2 under `strict`: the tracking tolerances saturate on this
        # out-of-distribution terrain (99.5% of episodes, median step 378 against a median
        # ARRIVAL at 765), so the last re-plan survived on 4.7% of arriving episodes against
        # 100% under `loose`. On the in-distribution buffer the same rule fires on ~42% and
        # the cut is a real tail, which is why this is a warning keyed on the MEASUREMENT
        # rather than a refusal to use the strict rule.
        last_rep = {}
        for u, rp in zip(W["episode_uid"], W["replan_index"]):
            last_rep[int(u)] = max(last_rep.get(int(u), -1), int(rp))
        # `reached_goal`, NOT the strict-scored reward: the question is which episodes carry
        # a success bonus at all, and under strict the reward calls only 0.4% of them
        # arrivals -- precisely the ones whose last re-plan trivially survived. Asking the
        # reward would make the guard inspect the sample that cannot show the defect.
        arrived_eps = {int(u) for u, rg in zip(E["episode_uid"], E["reached_goal"]) if rg}
        if arrived_eps:
            n_ep = {int(u): int(c) for u, c in zip(E["episode_uid"], E["n_replans"])} \
                if "n_replans" in E else {}
            if n_ep:
                kept_last = [last_rep.get(u, -1) >= n_ep.get(u, 0) - 1 for u in arrived_eps]
                frac = float(np.mean(kept_last)) if kept_last else 1.0
                if frac < 0.5:
                    print(f"[fit] !! the cut removes the LAST re-plan of "
                          f"{100 * (1 - frac):.1f}% of the {len(arrived_eps)} ARRIVING "
                          "episodes. The progress reward pays its success bonus there and "
                          "nowhere else, so the critic is about to be fitted on a reward with "
                          "essentially no success term in it. This is what the strict tracking "
                          "rule does on out-of-distribution terrain -- pass "
                          "`--failure_rule loose` unless you mean it.")
        W = {k: v[alive] for k, v in W.items()}
        lab = {k: v[alive] for k, v in lab.items()}

    # The QUALITY head's target. Unconditional: the column is part of schema 2 and the
    # reader refuses anything else, so a guard here would be dead code pretending to be a
    # fallback. `--quality_bins 0` is how the survival-only control arm is selected.
    from robot_rl.flow_matching import advantage as adv_mod  # noqa: PLC0415
    spec = _tracking_spec_from(ea)
    # NOT in `lab`: that dict is row-indexed twice below (the alive filter here and the
    # censored filter in main), so anything in it that is not a per-window array raises on
    # the first boolean index -- or worse, gets silently sliced.
    meta_out: dict = {}
    if reward_npz:
        r, r_ok, rmeta = _reward_sidecar(reward_npz, W["episode_uid"], W["replan_index"],
                                         column=("penalty" if clip_c > 0.0 else "reward"))
        meta_out["reward_source"] = str(rmeta.get("reward_source", "unknown"))
        meta_out["reward_hash"] = str(rmeta.get("reward_hash", ""))
        source_note = f"{meta_out['reward_source']} sidecar"
    else:
        r, r_ok = adv_mod.chunk_rewards_from_track_err(
            W["track_err_prev"], W["episode_uid"], W["replan_index"], spec)
        meta_out["reward_source"], meta_out["reward_hash"] = "track_err", ""
        source_note = "track_err_prev"
    if clip_c > 0.0:
        if not reward_npz:
            raise SystemExit(
                "--clip_c needs --reward_npz: the clipped window return is computed from the "
                "sidecar's RAW penalty, which is not in the recording.")
        g, g_ok = adv_mod.clipped_window_return(
            r, r_ok, W["episode_uid"], W["replan_index"], lab["failed"],
            clip_c=float(clip_c), window=int(window), gamma=float(spec.gamma))
        meta_out["quality_range"] = (-1.0, 0.0)
        print(f"[fit] quality target: CLIPPED WINDOW RETURN from {source_note}, "
              f"c={clip_c:g} N={window} gamma={spec.gamma:g} -> "
              f"{int(g_ok.sum())}/{g_ok.size} windows in [-1, 0], median G "
              f"{np.median(g[g_ok]):.4f}, "
              f"{100 * float((g[g_ok] <= -0.999).mean()):.1f}% saturated at -1")
    elif str(meta_out.get("reward_source", "")) == PROGRESS_SOURCE:
        # The PROGRESS reward: a goodness with a ONE-OFF terminal bonus, so neither existing
        # reducer fits it -- `normalized_returns` assumes a per-step reward in (0, 1] and the
        # terminal chunk carries 2 + C. `bootstrapped_window_return` takes the ongoing and
        # terminal halves apart and bounds the RETURN at 2 + (1 - gamma) * C.
        term = _sidecar_column(reward_npz, W["episode_uid"], W["replan_index"], "success")
        ongoing = r - term
        r_max = 2.0 + (1.0 - float(spec.gamma)) * float(rmeta.get("success_const", 10.0))
        v_prev = (_bootstrap_v(bootstrap_v, bootstrap_variant, W["episode_uid"],
                               W["replan_index"], r_max) if bootstrap_v else None)
        g, g_ok = adv_mod.bootstrapped_window_return(
            ongoing, term, r_ok, W["episode_uid"], W["replan_index"], lab["failed"],
            v=v_prev, window=int(window), gamma=float(spec.gamma), return_max=r_max)
        # **The MONTE CARLO return, always, beside whatever target this round fitted.**
        # The critic's TARGET is bootstrapped by design -- a terminal bonus cannot reach the
        # chunks that chose the mode through a 3-chunk window -- but the ADVANTAGE must stay
        # `G_real - V`. `label_advantage` differences the `g` column against the head, so
        # writing only the bootstrapped target there makes the advantage
        # `(r + gamma^N V_prev) - V`: a TD advantage, which is a legitimate algorithm and is
        # not the one asked for, and nothing downstream could tell the two apart because both
        # are the same shape, the same scale and the same plausible magnitude.
        # Round 0 recomputes an identical array; that costs ~1 s on 200k rows and keeps the
        # consumer unconditional, which is worth more than the second.
        lab["g_mc"], lab["g_mc_valid"] = adv_mod.bootstrapped_window_return(
            ongoing, term, r_ok, W["episode_uid"], W["replan_index"], lab["failed"],
            v=None, window=int(window), gamma=float(spec.gamma), return_max=r_max)
        meta_out["fvi_bootstrap_from"] = str(bootstrap_v)
        meta_out["quality_range"] = (0.0, 1.0)
        meta_out["return_max"] = r_max
        print(f"[fit] quality target: PROGRESS return from {source_note}, "
              f"N={window} gamma={spec.gamma:g} return_max={r_max:.3f} -> "
              f"{int(g_ok.sum())}/{g_ok.size} windows, median G {np.median(g[g_ok]):.4f}, "
              f"terminal bonus on {int((term > 0).sum())} rows, "
              f"{'BOOTSTRAPPED' if bootstrap_v else 'MONTE CARLO (round 0)'}")
    else:
        g, g_ok = adv_mod.normalized_returns(
            r, r_ok, W["episode_uid"], W["replan_index"], lab["failed"],
            gamma=float(spec.gamma), tail_chunks=tail_chunks)
        meta_out["quality_range"] = (0.0, 1.0)
        print(f"[fit] quality target from {source_note}: {int(g_ok.sum())}/{g_ok.size} "
              f"windows, median G {np.median(g[g_ok]):.4f}, "
              f"spec {'FITTED' if not spec.is_bootstrap else 'BOOTSTRAP'}")
    # The WORST-SO-FAR context feature. Computed from the sidecar's raw penalty, and only
    # when asked: it changes the critic's input width, so a checkpoint must record whether it
    # has it (`worst_penalty_context`) exactly as it records `goal_context`.
    if worst_context:
        pen = _sidecar_column(reward_npz, W["episode_uid"], W["replan_index"], "penalty")
        lab["worst_so_far"], _w_ok = adv_mod.worst_penalty_before(
            pen, r_ok, W["episode_uid"], W["replan_index"])
        lab["worst_so_far"] = np.minimum(lab["worst_so_far"], WORST_CLIP) / WORST_CLIP
        print(f"[fit] context: worst terrain penalty BEFORE each chunk, "
              f"clipped at {WORST_CLIP:g} "
              f"-> median {np.median(lab['worst_so_far']):.4f} "
              f"p95 {np.quantile(lab['worst_so_far'], 0.95):.4f}")
    lab["chunk_reward"], lab["g"], lab["g_valid"] = r, g, g_ok
    # `lab` is row-indexed by the caller; anything in it must be a per-window array of the
    # same length, or the index raises (a string) or silently truncates (a short array).
    n_rows = W["episode_uid"].shape[0]
    bad = {k: type(v).__name__ for k, v in lab.items()
           if not isinstance(v, np.ndarray) or v.shape[0] != n_rows}
    if bad:
        raise SystemExit(
            f"[fit] `lab` must hold only per-window arrays of length {n_rows}; got {bad}. "
            "Metadata belongs in the returned meta dict -- `lab` is boolean-indexed twice "
            "downstream, so a scalar there raises and a short array truncates silently.")
    return W, {"cols": E, "attrs": ea, "ep_terrain": ep_terrain, **meta_out}, lab


def _sidecar_column(path: str, episode_uid: np.ndarray, replan_index: np.ndarray,
                    column: str) -> np.ndarray:
    """One column of a reward sidecar, joined on ``(episode_uid, replan_index)``.

    The thin general form of :func:`_reward_sidecar`, for the extra columns the progress
    reward writes -- ``success``, ``penalty``, ``vel_track`` -- which are read for the return's
    terminal half and the worst-so-far feature rather than as the reward itself.

    Joined by KEY for the same reason :func:`_reward_sidecar` is: the sidecar is a row subset
    in its own order, so a positional pairing would silently attach each window to some other
    window's value.

    Args:
        path: The sidecar ``.npz``.
        episode_uid: ``[W]`` this table's episode ids.
        replan_index: ``[W]`` this table's re-plan indices.
        column: The column to read.

    Returns:
        ``[W]`` float64, zero where the sidecar does not cover the row.

    Raises:
        SystemExit: If the sidecar has no such column, which means it was written by a
            grader that did not compute it.
    """
    d = np.load(path, allow_pickle=False)
    if column not in d.files:
        raise SystemExit(
            f"{path} has no {column!r} column (it carries {sorted(d.files)}). A progress "
            "reward needs one written by `inspect_terrain_reward.py --stage reward "
            "--progress`; a terrain-only sidecar cannot be read as one.")
    key = (d["episode_uid"].astype(np.int64) << np.int64(20)) + d["replan_index"].astype(
        np.int64)
    want = (episode_uid.astype(np.int64) << np.int64(20)) + replan_index.astype(np.int64)
    out = np.zeros(want.size, dtype=np.float64)
    if key.size == 0:
        # `clamp(0, size - 1)` gives -1 on an empty key and then indexes an empty array.
        # An empty sidecar is a real state (a grade killed before its first flush), and the
        # message has to say that rather than raise from inside a lookup.
        raise SystemExit(
            f"the sidecar holds no rows, so column {column!r} covers nothing. Re-run the "
            "grade stage.")
    order = np.argsort(key)
    pos = np.clip(np.searchsorted(key[order], want), 0, key.size - 1)
    hit = key[order][pos] == want
    out[hit] = d[column].astype(np.float64)[order][pos[hit]]
    return out


def _reward_sidecar(path: str, episode_uid: np.ndarray, replan_index: np.ndarray,
                    column: str = "reward") -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-window reward read from a sidecar and joined on ``(episode_uid, replan_index)``.

    **Joined by KEY, never by position.** The sidecar is written from a random subset of the
    rollout's rows and this table has already had its post-latch windows dropped, so the two
    orders have no reason to agree -- and a positional join would silently pair each window
    with some other window's reward, producing a critic fitted on a shuffled target that
    trains without complaint.

    Args:
        path: The ``.npz`` written by ``inspect_terrain_reward.py --stage reward``.
        episode_uid: ``[W]`` this table's episode ids.
        replan_index: ``[W]`` this table's re-plan indices.
        column: Which sidecar column to read -- ``"reward"`` for the bounded ``exp(-p/tau)``
            map, or ``"penalty"`` for the RAW penalty the clipped window return needs.

    Returns:
        ``(reward, valid, meta)`` -- ``[W]`` float64, ``[W]`` bool for the rows the sidecar
        covers, and the sidecar's stamped metadata plus its ``reward_hash``.

    Raises:
        SystemExit: If the sidecar covers none of these windows, which means it was written
            for a different rollout.
    """
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    meta["reward_hash"] = str(d["reward_hash"])
    key = (d["episode_uid"].astype(np.int64) << np.int64(20)) + d["replan_index"].astype(
        np.int64)
    want = (episode_uid.astype(np.int64) << np.int64(20)) + replan_index.astype(np.int64)
    order = np.argsort(key)
    pos = np.searchsorted(key[order], want)
    pos_clipped = np.clip(pos, 0, key.size - 1)
    hit = key[order][pos_clipped] == want
    reward = np.zeros(want.size, dtype=np.float64)
    reward[hit] = d[column].astype(np.float64)[order][pos_clipped[hit]]
    if not hit.any():
        raise SystemExit(
            f"{path} covers none of this rollout's windows -- it was written for "
            f"{meta.get('rollout')!r}. A critic fitted on an empty join would train on "
            "nothing but zeros.")
    print(f"[fit] reward sidecar {os.path.basename(path)}: source "
          f"{meta.get('reward_source')!r}, hash {meta['reward_hash']}, column {column!r}, "
          f"covers {int(hit.sum())}/{hit.size} windows ({100 * hit.mean():.1f}%), median "
          f"{np.median(reward[hit]):.4f}")
    return reward, hit, meta


def _tracking_spec_from(episode_attrs: dict):
    """The spec the recording's strict columns were written under.

    Read from the FILE, never re-defaulted: the recorder stamps it precisely so the columns
    are interpretable without knowing which code ran, and silently substituting today's
    default would relabel someone else's artifact.

    Args:
        episode_attrs: Attrs from the episodes table.

    Returns:
        A ``TrackingErrorSpec``.

    Raises:
        SystemExit: If the recording carries no spec. That is a pre-stamp file, and guessing
            one would make its `first_violation_step` mean something it never meant.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import (  # noqa: PLC0415
        TrackingErrorSpec,
    )

    raw = episode_attrs.get("tracking_spec")
    if raw is None:
        raise SystemExit(
            "this rollout has a track_err_prev block but no `tracking_spec` attr, so the "
            "thresholds and kernel widths its columns were written under are unknown. "
            "Re-record with a reward sidecar.")
    return TrackingErrorSpec.from_dict(raw)


def _splits(uids: np.ndarray, tiles: np.ndarray, val_frac: float, seed: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Two validation masks, and the DIFFERENCE between them is the measurement.

    * ``val_ep`` -- held-out EPISODES from tiles the critic trained on. Splitting by episode
      and not by window is already necessary: with a terminal-only reward every window of an
      episode carries the same label, so a window split puts near-duplicate states either
      side of it and the held-out score measures memorisation of the episode.
    * ``val_tile`` -- held-out TILES, which the critic has never seen at all.

    The gap between the two answers the question the phase curve CANNOT. A high AUC on the
    first chunk of an episode has two readings -- the critic has memorised which tile is
    hard, or the depth image genuinely shows the terrain ahead and the outcome really is
    partly predictable before anything has happened -- and those call for opposite responses.
    Held-out tiles separate them: under memorisation the tile-held-out score collapses, under
    real perception it holds.

    Args:
        uids: ``[n]`` per-window episode ids.
        tiles: ``[n]`` per-window terrain tile ids.
        val_frac: Fraction held out, of each.
        seed: RNG seed.

    Returns:
        ``(val_episodes, val_tiles)``, each ``[n]`` bool. Disjoint: the episode split is
        drawn only from TRAINING tiles, so the two measure different things rather than
        overlapping.
    """
    rng = np.random.default_rng(seed)
    tl = np.unique(tiles)
    n_t = max(1, int(round(val_frac * tl.size)))
    held_tiles = set(rng.choice(tl, size=min(n_t, tl.size), replace=False).tolist())
    val_tile = np.array([int(t) in held_tiles for t in tiles])

    ep_pool = np.unique(uids[~val_tile])
    n_e = max(1, int(round(val_frac * ep_pool.size)))
    held_ep = set(rng.choice(ep_pool, size=min(n_e, ep_pool.size), replace=False).tolist())
    val_ep = np.array([int(u) in held_ep for u in uids]) & ~val_tile
    return val_ep, val_tile



def _bootstrap_v(path: str, variant: str, uid: np.ndarray, rep: np.ndarray,
                 return_max: float) -> np.ndarray:
    """Read a PREVIOUS round's value predictions, aligned to this round's window rows.

    Fitted value iteration with a FROZEN target: round 0 fits the Monte Carlo return, and
    round ``k`` fits ``r + gamma^N V_(k-1)``, where ``V_(k-1)`` is a file on disk rather than
    the model being trained.  Freezing it per round is what makes each round an ordinary
    supervised regression -- a target recomputed from the live model as it moves is a moving
    target, and the failure mode is a slow divergence that no metric on the fit reports.

    **The scale conversion is the part that is silently wrong if skipped.** The quality head
    predicts ``G``, which is the ``(1-gamma)``-weighted average ALREADY DIVIDED by
    ``return_max``, while :func:`bootstrapped_window_return` wants ``v`` on the raw weighted
    scale and divides at the end.  Feeding the head's output back unscaled bootstraps off a
    value ``return_max`` times too small, which trains cleanly and shrinks every target
    toward the first ``N`` chunks of real reward.

    **Rows the previous round did not cover are left at the MC tail, not at zero.** A missing
    prediction is an absent opinion; zero is the value of a failed state, and asserting it
    would teach the critic that unseen states are fatal.  They come back as ``nan`` and
    :func:`bootstrapped_window_return` falls back to its own tail on them.

    Args:
        path: A previous round's ``predictions.npz``.
        variant: Which critic's out-of-fold predictions to bootstrap off -- the key
            ``q_<variant>`` in that file.
        uid: ``[W]`` this round's episode uids.
        rep: ``[W]`` this round's replan indices.
        return_max: The declared return bound, to undo the head's normalization.

    Returns:
        ``[W]`` the frozen value per row on the raw weighted scale, ``nan`` where the
        previous round has no prediction.

    Raises:
        SystemExit: If the file carries no predictions for ``variant``, or if it was fitted
            on a different reward -- either would bootstrap onto a different quantity.
    """
    d = np.load(path, allow_pickle=False)
    key = f"q_{variant}"
    if key not in d.files:
        raise SystemExit(
            f"--bootstrap_v {path} has no `{key}`: it was fitted without a quality head, or "
            f"without that variant. Available: {sorted(f for f in d.files if f.startswith('q_'))}")
    prev_max = float(d["return_max"]) if "return_max" in d.files else 0.0
    if prev_max > 0.0 and abs(prev_max - float(return_max)) > 1e-9:
        raise SystemExit(
            f"--bootstrap_v {path} normalised its return by {prev_max:g} but this fit uses "
            f"{return_max:g}. Undoing the wrong division bootstraps onto a value scaled by "
            f"{float(return_max) / prev_max:.3f}, which trains cleanly. Both rounds need the "
            "same reward constants (`success_const`, `gamma`).")
    prev_src = str(d["reward_source"]) if "reward_source" in d.files else ""
    if prev_src != PROGRESS_SOURCE:
        raise SystemExit(
            f"--bootstrap_v {path} was fitted on reward_source={prev_src!r}, not "
            f"{PROGRESS_SOURCE!r}. Bootstrapping across rewards differences two different "
            f"quantities and both are plausible floats.")
    # **WHICH ROWS ACTUALLY CARRY A PREDICTION, and it is not all of them.** `q_all` is
    # assembled OUT OF FOLD: each fold writes only its own validation episodes, so a
    # 2-fold fit covers every row while a 1-FOLD fit leaves every TRAINING row at its
    # initialised 0.0 -- present in the file, numerically indistinguishable from a confident
    # "this state is worthless", and 0 is exactly the value of a FAILED state. Bootstrapping
    # on that would teach the critic that most of its own training set is fatal. Measured on
    # a 1-fold smoke fit: median v came back 0.0000 against a target median G of 0.0901.
    #
    # The file records `num_folds`, and with one fold the covered rows are precisely the two
    # validation scopes, so they can be recovered rather than guessed.
    n_folds = int(d["num_folds"]) if "num_folds" in d.files else 0
    covered = np.ones(d["episode_uid"].size, dtype=bool)
    if n_folds < 2:
        val = d["val"].astype(bool) if "val" in d.files else np.zeros_like(covered)
        vt = d["val_tile"].astype(bool) if "val_tile" in d.files else np.zeros_like(covered)
        covered = val | vt
        print(f"[fit] bootstrap: the source was fitted with num_folds={n_folds}, so only its "
              f"{int(covered.sum())}/{covered.size} VALIDATION rows carry an out-of-fold "
              f"prediction; the rest are dropped to the MC tail rather than read as 0.")
    # The same join key `merge_rollouts` and `label_advantage` use. uid is global after
    # `awr_buffer` rebases, so this is unique across iterations.
    want = (uid.astype(np.int64) << 20) + rep.astype(np.int64)
    have = (d["episode_uid"].astype(np.int64) << 20) + d["replan_index"].astype(np.int64)
    lut = {int(k): i for i, k in enumerate(have) if covered[i]}
    q = np.asarray(d[key], dtype=np.float64)
    out = np.full(want.size, np.nan, dtype=np.float64)
    idx = np.array([lut.get(int(k), -1) for k in want], dtype=np.int64)
    hit = idx >= 0
    out[hit] = q[idx[hit]] * float(return_max)
    if not hit.any():
        raise SystemExit(
            f"--bootstrap_v {path} covers NONE of this fit's rows. Round 0 must be fitted on "
            "the same buffer, with --folds 2 or more so every row gets an out-of-fold "
            "prediction.")
    print(f"[fit] bootstrap: V from {os.path.basename(os.path.dirname(path)) or path} "
          f"[{variant}] covers {int(hit.sum())}/{want.size} rows "
          f"({100.0 * hit.mean():.1f}%), median v {np.nanmedian(out[hit]):.4f} "
          f"(raw scale, return_max {return_max:g})")
    if hit.mean() < 0.5:
        print(f"[fit] bootstrap: WARNING only {100.0 * hit.mean():.1f}% of rows are "
              "bootstrapped; the rest keep the Monte Carlo tail, so this round is mostly "
              "round 0 again. Refit the source with more folds.")
    return out

def _batch(ctx, W: dict, idx: np.ndarray, priv: bool, device,
           *, goal_context: bool = True,
           worst_so_far: np.ndarray | None = None) -> value.ValueInputs:
    """Build one normalized batch from raw recorded rows.

    Normalized with the GENERATOR's own normalizers, so the critic sees exactly what the
    generator saw. Re-fitting them here would put the two models in different input spaces
    while every shape still agreed.

    Args:
        ctx: The loaded FM context, which supplies the encoder and all three normalizers.
        W: The window table.
        idx: Row indices.
        priv: The ``V_priv`` variant -- keeps the LINEAR half
            of the base twist.
        device: Torch device.
        worst_so_far: ``[W]`` the worst terrain penalty over the chunks STRICTLY BEFORE each
            row's own, or ``None`` to leave the block out.  Strictly before, because including
            a row's own penalty would leak part of ``r_t`` into ``V(s_t)`` and cancel exactly
            that much of the advantage the fit exists to produce.
        goal_context: Whether the three goal offsets go into the context token. ``False`` is
            the ablation arm; the width must come from :func:`value.context_width` with the
            same flag or the twist lands in the goal's columns silently.

    Returns:
        The assembled batch.
    """
    to = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)  # noqa: E731
    hist = ctx.state_norm.normalize(to(W["history"][idx]))
    ext = ctx.extero_norm.normalize(to(W["extero"][idx]).reshape(len(idx), -1))
    cmd = None
    if ctx.cmd_dim > 0:
        steps = torch.as_tensor(ctx.cmd_steps, dtype=torch.long, device=device)
        cmd = ctx.cmd_norm.normalize(to(W["cmd_future"][idx])[:, steps, :])
    # No scan and no proprio. The critic used to add a CNN trunk over the privileged height
    # scan plus one token per proprioception step; both token streams are gone, so the
    # recorded `height_scan` / `proprio` columns are simply not read here. They stay in the
    # recording: `height_scan` is the generator's own exteroception and is read all over the
    # repo, so it is not the critic's to retire.
    twist = to(W["base_twist_yaw"][idx])
    # ONBOARD keeps only the ANGULAR half: an IMU measures body rates and the yaw frame is
    # reachable from those plus its own roll/pitch, but nothing on the robot measures the
    # linear half without odometry. Slicing it away here is what makes the two variants'
    # gap a statement about information rather than about architecture.
    # NAMED pieces, assembled by `value.build_context`, which owns the order. Five sites used
    # to concatenate this by hand, so the layout lived five times while its width lived once
    # -- and a disagreement is a linear layer fed the wrong columns, not a shape error.
    pieces: dict[str, torch.Tensor] = {"twist": twist}
    if goal_context:
        # Goal offsets on a DECLARED metre scale rather than a fitted statistic, so a
        # critic's notion of "far" is a property of the robot and not of whichever rollout
        # was on disk -- the same idiom as the generator's `contract` normalization.
        goal = to(np.stack([W["goal_downrange_m"][idx], W["goal_lateral_m"][idx],
                            W["goal_height_err_m"][idx]], axis=-1)) / GOAL_SCALE_M
        pieces["goal"] = torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0)
    if worst_so_far is not None:
        pieces["worst_penalty"] = to(worst_so_far[idx])[:, None]
    ctx = value.build_context(privileged=priv, goal_context=goal_context,
                              worst_penalty_context=worst_so_far is not None, **pieces)
    return value.ValueInputs(history=hist, extero=ext, cmd_tokens=cmd,
                             history_null=to(W["history_null"][idx], torch.bool),
                             context=ctx)


def _predict_quality(model, ctx, W, idx, priv, device, batch: int, *,
                     goal_context: bool = True,
                     worst_so_far: np.ndarray | None = None) -> np.ndarray:
    """``G_hat`` over ``idx``, batched, in eval mode.

    Args:
        model: The critic.
        ctx: The FM context.
        W: The window table.
        idx: Row indices.
        priv: Privileged variant.
        device: Torch device.
        batch: Rows per forward pass.
        goal_context: Whether the goal offsets are in the context token.

    Returns:
        ``[len(idx)]`` predicted normalized returns.
    """
    was = model.training
    model.eval()
    out = np.empty(idx.size, dtype=np.float64)
    # `model.eval()` is NOT `no_grad`: it switches dropout/batchnorm and nothing else, so the
    # head's output still carries `requires_grad` and `.numpy()` raises.
    #
    # This function USED to carry `@torch.no_grad()`, and `0106d088` deleted it as collateral:
    # that commit removed the survival head, whose own `_predict` helper held two of the
    # file's three `no_grad` sites, and the third -- this decorator -- went with them while
    # the function it guarded stayed. Every AWR run up to `awr_vm30_fbc/iter_005` was fine;
    # nothing has fitted a critic since. Restored as a `with` block rather than a decorator so
    # the next deletion pass has to read the body to remove it. The graph is 55k windows wide
    # here, so it is a memory saving as much as a correctness fix.
    with torch.no_grad():
        for i in range(0, idx.size, batch):
            j = idx[i:i + batch]
            out[i:i + j.size] = model.quality(
                _batch(ctx, W, j, priv, device,
                       goal_context=goal_context,
                       worst_so_far=worst_so_far)).cpu().numpy()
    model.train(was)
    return out


def _fit(ctx, W, lab, train, val, *, priv: bool, epochs: int, batch: int, lr: float,
         device, label: str, wb=None, val_tile=None,
         quality_weight: float = 1.0,
         goal_context: bool = True,
         worst_so_far: np.ndarray | None = None) -> tuple[value.FlowValueModel, np.ndarray]:
    """Train one variant and return it plus held-out ``P(fail)``.

    Validation runs EVERY epoch, on both scopes. Without it the only visible signal is the
    training loss, which falls whether the model is learning the task or the tiles -- so
    there is no way to see overfitting happen, and no basis for choosing the epoch count
    other than guessing. The per-epoch unseen-tile AUC is the curve that shows it.
    """
    import copy  # noqa: PLC0415

    # NO `proprio_dim` / `proprio_steps`: those token streams were deleted in dcea58f0 and
    # the model has not taken them since. They survived here as dead keyword arguments, which
    # is not dead code -- it is a `TypeError` at the first critic fit, and nothing before the
    # fit touches this line. `test_value_model_call_sites.py` now parses this call against the
    # model's own signature, because an argument list is exactly the kind of drift no import
    # and no shape check can see.
    model = value.FlowValueModel(
        copy.deepcopy(ctx.model),
        context_dim=value.context_width(privileged=priv, goal_context=goal_context,
                                        worst_penalty_context=(worst_so_far is not None)),
    ).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=1e-4)
    # The QUALITY target. `g_valid` excludes the last window of every episode (its chunk's
    # consequences were never recorded) and any episode with no usable reward at all; those
    # rows train the survival head only, rather than being handed a zero -- which is the worst
    # possible return and would teach the critic that every episode's last decision was bad.
    # The value target is the ONLY target, and `g` is unconditional in schema 2 -- so this is
    # required rather than optional. It used to be `use_q = lab.get("g") is not None`, with
    # `None` arms threaded through the epoch loop that could never be taken; one of them would
    # have surfaced as `TypeError: 'NoneType' is not subscriptable` deep inside training.
    if lab.get("g") is None:
        raise SystemExit(
            "these labels carry no `g` column, so there is no value target to fit. That is a "
            "caller error rather than a mode: re-grade the rollout with a reward sidecar.")
    g_t = torch.as_tensor(lab["g"].astype(np.float32), device=device)
    g_ok = torch.as_tensor(lab["g_valid"].astype(bool), device=device)
    print(f"  [{label}] value head: {int(lab['g_valid'].sum())}/{lab['g_valid'].size} "
              f"windows carry a return target "
              f"(median G {np.median(lab['g'][lab['g_valid']]):.3f})")
    tr = np.flatnonzero(train)
    rng = np.random.default_rng(0)
    # Under quality_only the survival head is never trained, so its AUC is a coin flip and
    # cannot select a model. Select on the fold's own validation quality error instead --
    # lower is better, hence the sign flip so one comparison serves both modes.
    best_state, best_auc, best_epoch, last_auc = None, -1.0, 0, float("nan")
    select_name = "val quality MSE"
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        tot = 0.0
        for i in range(0, tr.size, batch):
            j = tr[i:i + batch]
            opt.zero_grad(set_to_none=True)
            xb = _batch(ctx, W, j, priv, device,
                        goal_context=goal_context, worst_so_far=worst_so_far)
            q_logits = model.quality_logits(xb)
            # ONE term. There used to be a survival BCE here, dropped outright under
            # `--quality_only` rather than down-weighted -- a tiny weight would still shape
            # the shared encoder while producing a p_fail nobody should read. Since every
            # reported result was fitted that way, the head and its loss are gone and this is
            # simply the value regression.
            jt = torch.as_tensor(j, device=device)
            m = g_ok[jt]
            if not bool(m.any()):
                # SKIP, do not backward a constant. `torch.zeros(())` carries no grad, so
                # `loss.backward()` on it raises "element 0 of tensors does not require grad"
                # -- which under the old two-term loss could not happen, because the survival
                # BCE covered every row. Now the value target is the only term and a batch
                # can legitimately hold no row with a measured return.
                continue
            loss = float(quality_weight) * value.quality_mse_loss(q_logits[m], g_t[jt][m])
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[{label}] the loss went non-finite on epoch {ep + 1}. Before this "
                    "check the run printed 'BCE nan' three times and went on to report an "
                    "aggregate AUC of 0.647 from an untrained head -- a number produced by "
                    "a broken fit is worse than no number. The usual cause is a non-finite "
                    "input: the height scan carries -inf for missed rays.")
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * j.size
        bce = tot / max(tr.size, 1)
        # Named for what it IS. With the quality head on this is BCE + w * cross-entropy over
        # 51 bins, and the second term's floor is the target's own entropy rather than zero --
        # so it reads ~3.4 where the survival-only loss reads ~0.31, and calling it "BCE"
        # invites reading a working two-head fit as a broken one.
        name = "quality MSE"
        print(f"  [{label}] epoch {ep + 1}/{epochs}  train {name} {bce:.4f}")
        log = {f"{label}/train_loss": bce, f"{label}/epoch": ep + 1}
        y_np = lab["failed"]
        for scope, m in (("val_seen_tiles", val), ("val_UNSEEN_tiles", val_tile)):
            if m is None or not m.any():
                continue
            j = np.flatnonzero(m)
            # No survival AUC: the head that produced it is gone, and the fit is selected on
            # the value regression alone. Set here so the flow below reads the same.
            a = float("nan")
            # The quality head needs DIFFERENT questions from the classifier, and the
            # first is whether it did anything: r2 is measured against the CONSTANT
            # predictor, so <= 0 means it collapsed to the mean -- a real stationary
            # point here, and one whose calibration curve is a single point on the
            # diagonal (measured ECE 0.00000), so calibration alone would pass it. rho is
            # what the selector needs, since best-of-K takes an argmax and only ordering
            # matters.
            jq = j[lab["g_valid"][j]]
            if jq.size:
                q_hat = _predict_quality(model, ctx, W, jq, priv, device,
                                         batch, goal_context=goal_context,
                                         worst_so_far=worst_so_far)
                qg = value_gates.return_gates(q_hat, lab["g"][jq])
                log.update({f"{label}/{scope}/q_r2": qg["r2"],
                            f"{label}/{scope}/q_rho": qg["rho"],
                            f"{label}/{scope}/q_rmse": qg["rmse"]})
                print(f"      {scope} quality r2 {qg['r2']:+.3f}  rho {qg['rho']:.3f} "
                      f"(shuffled {qg['rho_shuffled']:+.3f})  rmse {qg['rmse']:.4f}  "
                      f"pred_sd {qg['pred_std']:.3f} vs target_sd {qg['target_std']:.3f}")
                # And the DIFFERENCE gates, which are what an advantage actually uses.
                # `return_gates` above scores the LEVEL, and the two are different
                # questions: measured on the v2 critic, pointwise rho 0.809 against
                # 0.590 for the 8-step difference, with the predicted change 1.9x larger
                # than the real one. Printed here because it was only ever found by hand,
                # and `--n` on label_advantage is the knob it argues about.
                dg = value_gates.difference_gates(
                    q_hat, lab["g"][jq], W["episode_uid"][jq], W["replan_index"][jq])
                if dg:
                    cells = "  ".join(
                        f"n{n}: rho {r['rho']:+.3f} ratio {r['ratio']:.2f}"
                        for n, r in sorted(dg.items()))
                    print(f"      {scope} quality DIFFERENCES  {cells}"
                          "     (ratio = sd(predicted change) / sd(real change);"
                          " >1 inflates every advantage)")
                    log.update({f"{label}/{scope}/dq_rho_n{n}": r["rho"]
                                for n, r in dg.items()})
                    log.update({f"{label}/{scope}/dq_ratio_n{n}": r["ratio"]
                                for n, r in dg.items()})
            if scope == "val_seen_tiles":
                # -MSE, so "higher is better" holds for both selection criteria.
                jq = j[g_ok.cpu().numpy()[j]]
                a = (-float(np.mean((_predict_quality(
                    model, ctx, W, jq, priv, device, batch,
                    goal_context=goal_context,
                    worst_so_far=worst_so_far) - lab["g"][jq]) ** 2))
                     if jq.size else float("nan"))
                log[f"{label}/{scope}/quality_mse"] = -a
                print(f"      {scope} quality MSE {-a:.5f}")
            if scope == "val_seen_tiles" and np.isfinite(a):
                last_auc = a
                if a > best_auc:
                    best_auc, best_epoch = a, ep + 1
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
        if wb is not None:
            wb.log(log)
    if best_state is not None:
        # Select on the fold's OWN validation episodes -- seen tiles, unseen episodes. The
        # unseen-TILE score is the generalisation measure and must not also be the selection
        # criterion, or it stops being a holdout. With folds every window gets scored by a
        # model that did not train on it, so this split costs nothing.
        shown = (abs(best_auc), abs(last_auc))
        print(f"  [{label}] restoring epoch {best_epoch} ({select_name} {shown[0]:.5f} "
              f"against {shown[1]:.5f} at the end)")
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> int:
    """Entry point."""
    # Line-buffer stdout. A fit is a ~1 h job that is normally redirected to a file, where
    # python block-buffers stdout -- so every `[fit]` line sits in a 8 kB buffer while wandb's
    # own output (stderr, unbuffered) lands immediately. The log then shows the wandb banner
    # and nothing else, which reads as "still loading" when the job is several epochs in.
    # That misread has now cost twice, here and in `record_rollouts.py`'s progress print.
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stem", help="rollout stem: <stem>.h5 and <stem>.episodes.h5")
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=5,
                   help="Passes over the training folds. 5 because the critic on this task is "
                        "DATA-limited, not optimisation-limited, and that is measured: on a "
                        "2000-episode subset, 20 epochs restored epoch 13/14 by validation "
                        "and reached r2 +0.791 against 4 epochs' +0.786 -- 0.6%% relative for "
                        "3.8x the wall time (27.8 min against 7.2). The effect only gets "
                        "weaker with more data, since 4 epochs over 16k episodes is already "
                        "8x the gradient steps of 4 over 2k. Raise it when the data grows, "
                        "not when the fit looks unfinished; the best-by-val restore means a "
                        "larger number costs time rather than accuracy.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.25,
                   help="Fraction of TILES held out entirely, and (at --folds 1) of episodes.")
    p.add_argument("--quality_weight", type=float, default=1.0,
                   help="Weight on the quality cross-entropy against the survival BCE. Both "
                        "are per-sample means over comparable ranges, so 1.0 is a real "
                        "starting point rather than a placeholder -- but the two heads share "
                        "a frozen trunk and only their own readouts, so they cannot fight for "
                        "capacity the way two heads on one token would.")
    p.add_argument("--folds", type=int, default=1,
                   help="K-fold cross-fit over EPISODES, for labelling. At K>1 every window "
                        "on a training tile gets an OUT-OF-FOLD prediction from a model that "
                        "never saw its episode -- which is what stops the advantage from "
                        "being a lookup of the realized outcome. Costs the same wall-clock "
                        "as K=1 (each fold trains on 1/K less data), at the price of each "
                        "fold model seeing less.")
    p.add_argument("--bootstrap_v", type=str, default="",
                   help="A PREVIOUS round's predictions.npz. Turns the quality target from "
                        "the full-episode Monte Carlo return into the N-step bootstrap "
                        "r + gamma^N V_prev -- one round of fitted value iteration, with the "
                        "target FROZEN because it is a file rather than the live model. Round "
                        "0 leaves this empty: bootstrapping off an untrained head fits noise "
                        "to noise. Progress reward only (--reward_npz whose reward_source is "
                        "terrain_progress); the file's reward_source must match, since "
                        "bootstrapping across rewards differences two unrelated quantities. "
                        "Rows the previous round did not cover fall back to the MC tail.")
    p.add_argument("--bootstrap_variant", type=str, default="V_priv",
                   choices=("V_priv", "V_onboard"),
                   help="Which critic's out-of-fold predictions --bootstrap_v reads. V_priv "
                        "by default because it is the better estimator (it sees the LINEAR "
                        "half of the twist, which nothing on the robot measures without "
                        "odometry), and ONE target is built per fit, so both variants are "
                        "fitted against it. That makes V_onboard's target partly a "
                        "distillation of the privileged critic -- deliberate, and the reason "
                        "this is a flag rather than a constant.")
    p.add_argument("--worst_context", action="store_true",
                   help="Give the critic the worst terrain penalty seen STRICTLY BEFORE this "
                        f"chunk, clipped to {WORST_CLIP:g} and scaled to [0, 1]. The return it "
                        "is asked "
                        "to predict is a windowed sum over chunks that have already happened, "
                        "so a critic told nothing about the episode's history so far has to "
                        "average over how badly it has been going -- and the advantage is the "
                        "difference between the realised return and that average, so the "
                        "unexplained history lands in the advantage as noise. Strictly BEFORE "
                        "is the load-bearing half: including chunk t's own penalty leaks r_t "
                        "into V(s_t) and cancels exactly that much of A_t. Requires "
                        "--reward_npz, and the fitted checkpoint stamps the flag.")
    # Opt-OUT, so the default arm stays the one every existing checkpoint was fitted under.
    p.add_argument("--goal_context", action="store_true", default=True,
                   help="Put the three goal offsets in the context token (default).")
    p.add_argument("--no_goal_context", dest="goal_context", action="store_false",
                   help="ABLATION: fit with NO goal offsets, so the critic is told nothing "
                        "about how far the goal is. `value.py` predicts this fails -- with a "
                        "sparse terminal goal reward, two states identical to the generator "
                        "are worth different amounts depending on whether the terrain is one "
                        "metre ahead or eight -- so a degraded AUC here is the hypothesis "
                        "being confirmed, not the fit going wrong. The context token keeps "
                        "the base twist, so it does not disappear: V_priv goes 9 -> 6 wide "
                        "and V_onboard 6 -> 3.")
    p.add_argument("--clip_c", type=float, default=0.0,
                   help="clip constant c for the reward spec r = -min(p, c). Non-zero "
                        "switches the quality target to the CLIPPED WINDOW RETURN over "
                        "--window chunks, normalized into [-1, 0], with post-termination "
                        "chunks filled with -c. Needs --reward_npz (it reads the raw "
                        "penalty). 2 to 3 is the measured range and the choice inside it is "
                        "INSENSITIVE: the terrain penalty is bimodal (p95 1.72, p97 18.2, "
                        "p99 39.0), so almost nothing lands between them and c=2 vs c=3 "
                        "moves the saturating fraction only 4.67%% -> 4.13%%.")
    p.add_argument("--window", type=int, default=3,
                   help="chunks in the clipped-window return. Start short (2-4) and lengthen "
                        "only if per-mode advantage separation improves.")
    p.add_argument("--reward_npz", default="",
                   help="per-window reward sidecar to fit the QUALITY head on instead of the "
                        "tracking error (from inspect_terrain_reward.py --stage reward). The "
                        "source name and its hash are stamped into predictions.npz so "
                        "label_advantage can refuse to mix two rewards.")
    p.add_argument("--tail_chunks", type=int, default=1,
                   help="Trailing chunks the QUALITY target's non-failure tail averages. The "
                        "tail carries a median 52%% of G on a survived episode, so resting it "
                        "on one measurement is high variance; 3 drops its spread ~12%%. It is "
                        "an ARGUMENT and not a new default because the tail is stamped in no "
                        "existing artifact -- a changed default would silently score every "
                        "critic already fitted against a target it never saw. Stamped into "
                        "the checkpoint, so a refit states what it used.")
    p.add_argument("--failure_rule", choices=("strict", "loose"), default="strict",
                   help="What counts as a failure. 'strict' is the fall crossing OR the "
                        "strict tracking violation; 'loose' is the fall alone, which is what "
                        "MuJoCo scores. Under PLACEHOLDER tracking tolerances the strict rule "
                        "can label almost everything failed -- on the box2 grid it left 33 "
                        "non-failures against the loose rule's 728 -- so check the positive "
                        "count that this prints before trusting an AUC.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_windows", type=int, default=0,
                   help="Cap on WINDOWS, taken as the first N rows -- a smoke-test knob, not "
                        "a sampling one. On a merged multi-speed recording the first N rows "
                        "all come from the first cell, so the cap silently yields a "
                        "single-speed subset whose opening rows are the stand-and-settle "
                        "hold. Use --max_episodes to size a fit.")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="Keep a RANDOM subset of this many episodes (seeded by --seed), with "
                        "all of their windows. This is the knob for asking how much data a "
                        "critic needs: the fit's own effective sample size is the EPISODE "
                        "count, not the window count, because every window of an episode "
                        "carries the same label -- so halving windows within episodes "
                        "changes almost nothing while halving episodes halves the "
                        "information. Drawn over episodes rather than per cell so the speed "
                        "mix is preserved in expectation.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Opt-OUT, matching `scripts/flow_matching/train.py` and the RL agent cfgs.
    p.add_argument("--wandb", action="store_true", default=True,
                   help="Log to Weights & Biases (default; use --no_wandb to disable).")
    p.add_argument("--no_wandb", dest="wandb", action="store_false",
                   help="Disable W&B logging.")
    p.add_argument("--wandb_project", type=str, default="robot_rl_value")
    p.add_argument("--run_name", type=str, default="",
                   help="W&B run name; defaults to the output directory's basename.")
    args = p.parse_args()
    if args.bootstrap_v and float(args.clip_c) > 0.0:
        # `--clip_c` SELECTS THE REDUCER, and the clipped-penalty branch never consults `v`,
        # so this combination would run an ordinary Monte Carlo fit while the operator
        # believed a round of value iteration had happened -- and print nothing either way.
        raise SystemExit(
            f"--bootstrap_v with --clip_c {args.clip_c}: the clipped-penalty reducer takes no "
            "bootstrap, so the flag would be silently ignored. Pass --clip_c 0 for the "
            "progress reward's reducer, which is the one value iteration applies to.")
    if args.bootstrap_v and not args.reward_npz:
        raise SystemExit(
            "--bootstrap_v needs --reward_npz: the bootstrap target is only defined for the "
            "progress reward's ongoing/terminal split, which comes from the sidecar.")
    if args.bootstrap_v and not os.path.exists(args.bootstrap_v):
        raise SystemExit(f"--bootstrap_v: no such file {args.bootstrap_v}")
    if args.worst_context and not args.reward_npz:
        # NOT a warning. The stamp declares the block from the FLAG while the batch builder
        # takes it from the DATA, so a flag with no sidecar writes a checkpoint claiming an
        # input width it was not fitted at -- which loads clean against the wrong layer.
        raise SystemExit(
            "--worst_context needs --reward_npz: the worst-penalty-so-far block is derived "
            "from the sidecar's `penalty` column, and there is nowhere else to get it.")

    stem = args.stem[:-3] if args.stem.endswith(".h5") else args.stem
    stem = stem[:-9] if stem.endswith(".episodes") else stem
    strict_failure = args.failure_rule == "strict"
    W, E, lab = _load(stem, args.max_windows, int(args.tail_chunks),
                      strict_failure=strict_failure,
                      max_episodes=int(args.max_episodes), seed=int(args.seed),
                      reward_npz=args.reward_npz, clip_c=float(args.clip_c),
                      window=int(args.window), worst_context=bool(args.worst_context),
                      bootstrap_v=str(args.bootstrap_v),
                      bootstrap_variant=str(args.bootstrap_variant))
    print(f"[fit] failure rule: {args.failure_rule} "
          + ("(fall crossing OR strict tracking violation)" if strict_failure
             else "(the FALL ALONE -- MuJoCo's rule; the tracking violation is ignored)")
          + ". The QUALITY target is unaffected by this and still answers to the tracking "
            "spec.")
    keep = lab["live"]
    if not keep.all():
        # Censored episodes have NO outcome: they neither fell nor failed to arrive, they were
        # cut when the rollout hit its ceiling. Scoring them would teach the critic that
        # whatever was happening at the ceiling predicts failure, which is a property of the
        # harness.
        print(f"[fit] dropping {int((~keep).sum())} window(s) from censored episodes")
        W = {k: v[keep] for k, v in W.items()}
        lab = {k: v[keep] for k, v in lab.items()}

    # Resolved ONCE and carried, because three places need it -- the checkpoint stamp, the
    # sidecar's gamma, and the printed provenance -- and re-deriving it per fold would let
    # them disagree if the derivation ever grew a default.
    track_spec = _tracking_spec_from(E["attrs"])
    uids = W["episode_uid"]
    n_ep = int(np.unique(uids).size)
    n_fail_ep = int(np.unique(uids[lab["failed"]]).size)
    # PRINTED, and the same idiom as train.py printing the resolved `extero_clip`: the arm
    # is invisible in the output otherwise, and a log that does not say which one ran is a log
    # whose AUC cannot be attributed.
    # The width PRINTED must be the width BUILT, `--worst_context` included: this line is the
    # only place a reader sees the critic's input width, and one that omitted the new block
    # said 9 while the model was 10 and the checkpoint stamped 10.
    print(f"[fit] context token: {'goal(3) + ' if args.goal_context else 'NO GOAL, '}"
          f"twist{' + worst(1)' if args.worst_context else ''}"
          f" -> {value.context_width(privileged=True, goal_context=args.goal_context, worst_penalty_context=args.worst_context)} "
          f"wide (V_priv) / "
          f"{value.context_width(privileged=False, goal_context=args.goal_context, worst_penalty_context=args.worst_context)} "
          f"(V_onboard)"
          + ("" if args.goal_context else
             "   <-- ABLATION: goal_downrange_m / goal_lateral_m / goal_height_err_m are OUT"))
    print(f"[fit] {uids.size} windows over {n_ep} episodes "
          f"({n_fail_ep} failed, {100 * n_fail_ep / max(n_ep, 1):.1f}%)")
    print(f"[fit] EFFECTIVE sample size is the EPISODE count ({n_ep}), not the window count: "
          "every window of an episode carries the same label.")

    tiles = np.array([E["ep_terrain"].get(int(u), -1) for u in uids])
    val_ep, val_tile = _splits(uids, tiles, args.val_frac, args.seed)
    train = ~val_ep & ~val_tile
    print(f"[fit] {int(np.unique(tiles).size)} tiles: "
          f"{int(np.unique(tiles[train]).size)} train / "
          f"{int(np.unique(tiles[val_tile]).size)} HELD OUT")
    print(f"[fit] episodes: {int(np.unique(uids[train]).size)} train / "
          f"{int(np.unique(uids[val_ep]).size)} val (seen tiles) / "
          f"{int(np.unique(uids[val_tile]).size)} val (unseen tiles)")

    if args.folds < 1:
        raise SystemExit(f"--folds must be at least 1, got {args.folds}")
    if args.folds == 1:
        folds = [(train, val_ep)]
    else:
        # Partition the TRAINING-tile episodes into K folds. The tile holdout is excluded
        # from every fold's training set, so it stays a clean generalisation measure.
        pool = np.unique(uids[~val_tile])
        rng_f = np.random.default_rng(args.seed + 1)
        assign = {int(u): int(i % args.folds)
                  for i, u in enumerate(rng_f.permutation(pool))}
        fold_of = np.array([assign.get(int(u), -1) for u in uids])
        folds = []
        for k in range(args.folds):
            va_k = (fold_of == k)
            folds.append(((~val_tile) & ~va_k, va_k))
        sizes = [int(np.unique(uids[v]).size) for _, v in folds]
        print(f"[fit] {args.folds}-fold cross-fit over episodes: fold sizes {sizes}")

    # No height-scan fold. This used to REFUSE a recording that did not state the scan's
    # (height, width), because 176 rays is both 11x16 and 16x11 and a transposed grid is
    # scrambled input to a convolution rather than a reordered vector. The critic no longer
    # has a scan trunk to feed, so the fold -- and that whole failure mode -- is gone.
    # Never let telemetry take down a fit. An uninstalled wandb, a missing login or a
    # network blip would otherwise kill the run before its first epoch; the JSON and the
    # npz are written regardless.
    wb = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415
            wb = wandb.init(project=args.wandb_project,
                            name=args.run_name or os.path.basename(os.path.abspath(args.out)),
                            config={"stem": stem, "fm_checkpoint": args.fm_checkpoint,
                                    "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                                    "val_frac": args.val_frac, "seed": args.seed,
                                    "n_episodes": n_ep, "n_windows": int(uids.size),
                                    "n_failed_episodes": n_fail_ep,
                                    "n_tiles": int(np.unique(tiles).size)})
            # Declare epoch as the x-axis, one per (variant, fold). Without this every
            # series is plotted against wandb's auto-incrementing `_step`, and because the
            # variants and folds train SEQUENTIALLY into one run they land on disjoint parts
            # of that axis -- V_priv on steps 0..N-1 and V_onboard on N..2N-1. Each curve
            # then looks like a stub and overlaying them looks offset, which is a plotting
            # artefact rather than anything about the models.
            for label in ("V_priv", "V_onboard"):
                for k in range(max(int(args.folds), 1)):
                    tag = label if args.folds == 1 else f"{label}/f{k}"
                    wandb.define_metric(f"{tag}/epoch")
                    wandb.define_metric(f"{tag}/*", step_metric=f"{tag}/epoch")
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb logging is OFF: {type(exc).__name__}: {exc}\n"
                  f"       The fit continues; gates.json and predictions.npz still land in "
                  f"{args.out}.\n"
                  "       Pass --no_wandb to silence this, or fix the login (wandb login).")

    ctx = online.load_fm_context(args.fm_checkpoint, device=args.device)
    results = {}
    qpreds = {}
    os.makedirs(args.out, exist_ok=True)
    # ONE variant. There used to be a non-privileged `V_onboard` fitted alongside, and their
    # gap was reported as "how much outcome-relevant information the generator cannot see" --
    # but the fine-tune loop passes `--variants V_priv`, so it was never fitted for any
    # reported result and the gap was never measured on one.
    for label, priv in (("V_priv", True),):
        print(f"[fit] {label}")
        # The value head's out-of-fold prediction: every window scored by a fold that did not
        # train on it, which is what `A = G - V` needs to avoid differencing a row against a
        # baseline fitted on itself.
        q_all = np.zeros(uids.size, dtype=np.float64)
        for k, (tr_k, va_k) in enumerate(folds):
            if len(folds) > 1:
                print(f"  [{label}] fold {k + 1}/{len(folds)}  "
                      f"{int(np.unique(uids[tr_k]).size)} train episodes")
            model = _fit(ctx, W, lab, tr_k, va_k,
                              quality_weight=float(args.quality_weight),
                              goal_context=bool(args.goal_context),
                              worst_so_far=lab.get("worst_so_far"),
                              priv=priv, epochs=args.epochs,
                              batch=args.batch, lr=args.lr, device=args.device,
                              label=(label if len(folds) == 1 else f"{label}/f{k}"),
                              wb=wb, val_tile=val_tile)
            # OUT OF FOLD: each window takes the prediction of the model that did not train
            # on its episode. The held-out TILES are out-of-sample for every fold, so they
            # take the mean -- averaging there is a free variance reduction, and doing it on
            # a training fold would silently reintroduce the in-sample reading.
            if q_all is not None:
                q_k = _predict_quality(model, ctx, W, np.arange(uids.size), priv,
                                       args.device, args.batch,
                                       goal_context=bool(args.goal_context),
                                       worst_so_far=lab.get("worst_so_far"))
                q_all[va_k] = q_k[va_k]
                q_all[val_tile] += q_k[val_tile] / len(folds)
            # Checkpoint it. Without this the fit is write-only: the gates land on disk but
            # the thing that produced them does not, so labelling a rollout or scoring
            # candidate chunks would mean re-fitting from scratch.
            name = f"{label}.pt" if len(folds) == 1 else f"{label}.fold{k}.pt"
            # The QUALITY head's shape and the reward it was fitted against travel WITH the
            # checkpoint. `quality_bins` because a consumer has to build the head before it
            # can load into it, and a strict `load_state_dict` on a mis-sized head fails with
            # a shape error rather than a diagnosis. `tracking_spec` because the head predicts
            # a return under one set of sigmas, bounds and gamma: a selector that measures the
            # reward under a DIFFERENT spec and adds it to this head's prediction is summing
            # two different quantities, and both are plausible floats. This is the fourth
            # instance in this repo of "a checkpoint's saved config is the only record of what
            # it was trained against"; see CLAUDE.md on `extero_clip` for the first.
            torch.save({"model": model.state_dict(), "variant": label,
                        # WHICH FAILURE RULE trained this head. The strict and loose rules
                        # differ by a factor of 22 in positive count on the box2 grid, and
                        # nothing about the weights says which one produced them -- so a
                        # consumer scoring under the other rule would compare two different
                        # quantities, both plausible. Fifth instance of "a checkpoint's saved
                        # config is the only record of what it was trained against".
                        "failure_rule": args.failure_rule,
                        "fm_checkpoint": os.path.abspath(args.fm_checkpoint),
                        # The PATH identifies where it was; the HASH identifies WHAT it was.
                        # Only the second survives being read on another machine, and the
                        # generator is what supplies the frozen encoder and all three
                        # normalizers -- a wrong one loads with correct shapes and is silent.
                        "fm_checkpoint_sha256": _sha256_file(args.fm_checkpoint),
                        "context_dim": value.context_width(
                            privileged=priv, goal_context=bool(args.goal_context),
                            worst_penalty_context=bool(args.worst_context)),
                        "privileged": bool(priv),
                        #: WHETHER THE WORST-PENALTY-SO-FAR BLOCK IS IN THE CONTEXT TOKEN.
                        #: Same reasoning as `goal_context` below -- the arms differ only in
                        #: this token's width, so an undeclared one loads against the wrong
                        #: layer or, worse, at a width that happens to agree.
                        "worst_penalty_context": bool(args.worst_context),
                        # WHETHER THE GOAL WAS IN THE CONTEXT TOKEN. Stamped because the two
                        # arms differ only in this token's width, and four readers build that
                        # token: without the stamp a no-goal checkpoint would be handed a
                        # 9-wide vector against a 6-wide layer, and if the widths ever did
                        # line up the twist would silently occupy the goal's columns. Readers
                        # default an ABSENT stamp to True -- every checkpoint predating this
                        # flag did carry the goal, so that is a fact about them rather than
                        # today's default re-resolved onto someone else's artifact.
                        "goal_context": bool(args.goal_context),
                        "epochs": int(args.epochs), "stem": stem,
                        "fold": int(k), "num_folds": int(len(folds)),
                        "quality_weight": float(args.quality_weight),
                        "tracking_spec": track_spec.to_dict(),
                        "gamma": float(track_spec.gamma),
                        # STAMPED, because the tail is otherwise unrecoverable from
                        # the checkpoint and an advantage built with a different one
                        # prices its baseline against a target this never saw.
                        "tail_chunks": int(args.tail_chunks)},
                       os.path.join(args.out, name))
            del model
        if q_all is not None:
            qpreds[label] = q_all
        # No survival gates. `_gates` scored calibration, AUC, AUC-by-phase and lead-to-fall
        # -- every one of them a property of the P(fail) head, which is gone. The value head's
        # own diagnostics (r2, rho, rmse, and the DIFFERENCE gates that say whether adjacent
        # chunks are ordered correctly) are printed per epoch by `_fit` instead, which is where
        # they can be read against the fit that produced them.
        results[label] = {}

    # The V_priv/V_onboard gap is gone with the second variant: it was reported as "how
    # much outcome-relevant information the generator cannot see", and was never
    # measured for a reported result because the loop fits V_priv alone.
    gap = float("nan")
    with open(os.path.join(args.out, "gates.json"), "w") as f:
        json.dump({"gap_mean_abs": gap, "n_episodes": n_ep, "n_windows": int(uids.size),
                   "results": results}, f, indent=2, default=_jsonable)
    np.savez(os.path.join(args.out, "predictions.npz"), episode_uid=uids,
             replan_index=W["replan_index"], val=val_ep, val_tile=val_tile, tile=tiles,
             failed=lab["failed"],
             first_fall_step=lab["first_fall_step"], family_code=lab["family_code"],
             num_folds=int(args.folds),
             # The quality half of the sidecar: the head's out-of-fold G_hat, the MEASURED
             # per-chunk reward, and the realized return with its validity mask. Saved so
             # `label_advantage` can form the combined V without re-reading a 12 GB rollout
             # -- and so the shift out of `track_err_prev` is undone exactly once, here.
             chunk_reward=lab["chunk_reward"], g=lab["g"], g_valid=lab["g_valid"],
             # The realized MC return, for the ADVANTAGE. Equal to `g` on a Monte Carlo
             # round and different on a bootstrapped one, which is exactly when it matters.
             **({} if "g_mc" not in lab else
                {"g_mc": lab["g_mc"], "g_mc_valid": lab["g_mc_valid"]}),
             gamma=float(track_spec.gamma),
             # The declared RETURN bound this fit's `g` was divided by. Stamped because
             # `_bootstrap_v` has to undo exactly that division, and it otherwise re-derives
             # the bound from the CURRENT sidecar's `success_const` -- which agrees only
             # while the two rounds share a sidecar.
             return_max=float(E.get("return_max", 0.0)),
             # WHICH reward this critic's quality head was fitted on. Stamped so an advantage
             # can never difference a G built from one reward against a baseline fitted on
             # another -- that mismatch is dimensionally plausible and numerically nonsense.
             reward_source=str(E.get("reward_source", "track_err")),
             # WHICH round of value iteration this is, i.e. whether `g` is the realized
             # return or a bootstrapped target. Stamped so `label_advantage` can REFUSE a
             # bootstrapped file that carries no `g_mc` instead of quietly differencing the
             # target against the head and calling it an advantage.
             fvi_bootstrap_from=str(E.get("fvi_bootstrap_from", "")),
             reward_hash=str(E.get("reward_hash", "")),
             # The target SPEC and whether the SURVIVAL head was trained at all. Both are
             # stamped because an advantage must not be formed against a baseline fitted on a
             # differently-shaped return, and because an UNTRAINED p_fail still produces
             # numbers that look exactly like predictions. Range matters as much as source:
             # [0, 1] and [-1, 0] are both plausible-looking.
             quality_range=np.asarray(E.get("quality_range", (0.0, 1.0)), np.float64),
             clip_c=float(args.clip_c), window=int(args.window),
             **{f"q_{k}": v for k, v in qpreds.items()})
    if wb is not None:
        if np.isfinite(gap):
            wb.log({"gap_mean_abs": gap})
            wb.summary["gap_mean_abs"] = gap
        wb.finish()
    print(f"[fit] wrote {args.out}/gates.json and predictions.npz")
    return 0


def _jsonable(o):
    """JSON encoder for the numpy the gates return."""
    if isinstance(o, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v
                for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        v = o.item()
        return None if (isinstance(v, float) and not np.isfinite(v)) else v
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not JSON-serialisable: {type(o)}")




if __name__ == "__main__":
    raise SystemExit(main())
