# What is CHOOSING a chunk worth, at one re-plan?

A split-continuation oracle bound on per-decision selection: the best of K sampled chunks is
picked on one rollout and paid out on an INDEPENDENT one, so no gain can come from the luck of
the draw that made it look good. Plus what three live selectors actually collect.

**Date:** 2026-09-02 · **Tracker:** `2026-08-20_16-36-10_mode11_decor_plan_v1` ·
**Generator:** `2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff` ·
**Simulator:** MuJoCo (`mjeval` / `live_select`)

Tooling: `transfer/sim/branch_probe.py --stage run --second_seed_every 1` (collection),
`transfer/sim/oracle_gain.py` (the ceiling), `transfer/sim/ksweep_report.py` (the live
sweeps). Outputs in `logs/value/oracle_gain_v1/`, `logs/value/ksweep_*/`.

## Headline

| question | answer |
|---|---|
| gain from ONE oracle decision, episode success | **+0.0099** [+0.0036, +0.0161] |
| gain from ONE oracle decision, realized return | **+0.0101** [+0.0077, +0.0124] |
| how much of the naive best-of-K number is luck? | **85%** (survival), **75%** (return) |
| where does it live? | **14.5%** of states; +0.066 there, +0.000 elsewhere |
| same-noise control | **0.000000000** on 117 groups, PASS |
| live selector, critic (v2, opening transient present) | **+0.167** ranking [+0.106, +0.231] |
| live selector, critic (v4, transient fixed) | **−0.133** (partial, 3 cells) |
| live selector, REALIZED tracking quality | **−0.025** [−0.114, +0.047], 9 cells |

**The per-decision prize is about one point of episode success, and 85% of what a naive
best-of-K measurement reports is the maximum of K noisy numbers.** The one live selector that
did pay (+0.167) was measured with a simulator artifact present that has since been fixed;
neither a current learned critic nor a model-free realized-quality rule reproduces it.

## 1. Why a second continuation is the whole measurement

`branch_probe` branches K = 4 sibling chunks from one probed state and rolls each to
completion. With ONE rollout per sibling, "pick the best" is unfalsifiable: the pick is scored
on the very draw that made it look best, so the number returned is the spread of the outcome
noise and it is returned confidently. `--second_seed_every 1` rolls a SECOND,
independently-keyed continuation for every sibling, and then the pick can be paid out on a
draw it never saw.

    honest      select on A, score on B -- and the mirror, averaged
    hindsight   select on A, score on A.  The naive number
    any-of-K    max_k y[g,k,A] - mean_k y[g,k,A].  ALGEBRAICALLY the same as hindsight on
                column A -- the argmax's own value IS the max

The estimator is swap-averaged so neither column is privileged; `baseline` is the mean over
siblings, which is what a random pick gets. **No predictor of a chunk's EXPECTED outcome can
beat knowing a whole independent realization of it**, so the honest gain bounds best-of-K,
filtered BC and advantage conditioning at one decision. It is not a trajectory bound -- §6.

## 2. The data

`--second_seed_every 1`, so every probed group carries 2K = 8 rollouts to completion.

| cell | tile | family | speed | groups |
|---|---|---|---|---|
| t6_s0.8 | 6 | stairs_up | 0.8 | 449 |
| t6_s1.6 | 6 | stairs_up | 1.6 | 358 |
| t6_s2.4 | 6 | stairs_up | 2.4 | 331 |
| t20_s1.6 | 20 | stairs_down | 1.6 | 389 |
| t40_s1.6 | 40 | jump_on (`box`) | 1.6 | 425 |
| **total** | | | | **1952 groups, 234 episodes** |

No branch timed out, so the binary label is a real outcome rather than the clock -- the
verdict histogram travels in `oracle_gain.json` because a timeout-dominated collection would
look identical otherwise, which is the shape of the `--isaac_matched` retraction.

**215 of 2167 probed groups are dropped**, all because a sibling terminated INSIDE its own
chunk and so has no continuation to roll. That bias points DOWN -- a sibling that fell in
0.24 s is the clearest thing a selector could have avoided. `--include_terminal_binary` sizes
it: those siblings have zero continuation variance on the binary outcome, so they can be
restored exactly rather than imputed, and doing so moves the gain **+0.0099 -> +0.0136**
[+0.0074, +0.0197]. So the drop rule costs ~4 points of the estimate and changes neither sign
nor order.

## 3. Controls

| control | expected | measured |
|---|---|---|
| `--same_noise` (siblings handed ONE draw) | exactly 0 | **0.000000000**, spread 0.000000000, n = 117 |
| within-group permutation of column B (1000 reps) | 0 | +0.0000 ± 0.0030, **p = 0.001** |
| swap symmetry, A->B vs B->A | agree | interval contains 0 on all three arms |
| estimator vs closed-form synthetic data | recovers it | `--stage selfcheck`, all pass |

The same-noise arm is the one that retired the Isaac best-of-K harness by failing, and it does
double duty here: bit-identical siblings are what produce a spread of exactly zero, so it also
re-verifies that `Snapshot.restore` is exact.

The permutation null re-links column B to a random sibling within each group, preserving every
marginal -- same K outcomes, same group mean, same maximum, so baseline and hindsight are
numerically untouched -- and destroying only the sibling correspondence, which is the
hypothesis. Not the global label shuffle used elsewhere in this tree, which returns its chance
line analytically and so cannot fail.

**The estimator was validated on synthetic data before any number was quoted**, in the
discipline `seed_headroom.py` paid for the hard way. `oracle_gain.py --stage selfcheck` builds
`y = mu + eps` with a known SNR and checks that the honest gain recovers
`rho * sigma_mu * E[max of K]` at three ratios, that the no-signal case returns 0 while
hindsight returns a quarter of a standard deviation of pure luck, and that the
perfect-predictor extrapolation recovers the TRUE oracle gain exactly under its own model.
`transfer/sim/tests/test_oracle_gain.py` (21 tests) pins those plus the loader's drop and
refusal rules.

## 4. The ceiling

### 4.1 Aggregate

| | realized return | episode success |
|---|---|---|
| base, random pick | 0.5805 | 0.8121 |
| oracle pick, held-out draw | 0.5906 | 0.8220 |
| **honest gain** | **+0.0101** [+0.0077, +0.0124] | **+0.0099** [+0.0036, +0.0161] |
| permutation p | 0.001 | 0.001 |
| hindsight (same draw) | +0.0398 | +0.0645 |
| fraction that is luck | 75% | 85% |
| reliability r | 0.216 | 0.179 |
| perfect-predictor extrapolation | +0.022 | +0.023 |

Intervals are 95% percentile bootstrap **clustered on the episode**: successive probed
re-plans of one episode share a robot that is either coping or already in trouble.

### 4.2 It lives in a minority of states

| survival, by realized spread in the selection draw | n | base | honest gain | 95% CI |
|---|---:|---:|---:|---|
| all siblings agree | 1669 | 0.8489 | +0.0004 (exactly 0.0000 tie-integrated) | [−0.002, +0.003] |
| **siblings disagree** | **283** | 0.5945 | **+0.0663** | **[+0.029, +0.103]** |

| return, by realized spread | n | honest gain | 95% CI | hindsight |
|---|---:|---:|---|---:|
| (0, 0.15] | 1593 | +0.0042 | [+0.003, +0.005] | +0.022 |
| (0.15, 0.30] | 154 | +0.0242 | [+0.012, +0.036] | +0.090 |
| **above 0.30** | **205** | **+0.0454** | [+0.025, +0.064] | +0.144 |

A group whose siblings all came out the same cannot be chosen between, so it contributes
exactly zero by construction -- which is why the honest arm is pinned at zero in the leftmost
bin while both same-draw arms are already positive. That gap IS the bias, drawn in
`logs/value/oracle_gain_v1/oracle_gain.png`.

### 4.3 By family and by position

| survival | n | base | honest gain | 95% CI |
|---|---:|---:|---:|---|
| stairs_up | 1138 | 0.7272 | +0.0141 | [+0.004, +0.024] |
| stairs_down | 389 | 0.8551 | +0.0087 | [−0.007, +0.023] |
| jump_on (`box`) | 425 | **1.0000** | +0.0000 | [0, 0] |

**`jump_on` at tile 40 / 1.6 m/s returns nothing because every branch succeeded.** At a base
of exactly 1.000 there is nothing to select away, so the cell is uninformative by construction
rather than a null about selection -- and it contributes 425 zero-gain groups to the pooled
headline. Read the family rows, not just the total.

By position in the episode the gain declines monotonically (+0.0172 early, +0.0102 middle,
+0.0029 late on survival) while per-group reliability rises, consistent with the outcome being
largely decided by the late third rather than with selection mattering less there.

## 5. Two reliability statistics, and a cross-session check that landed

`critic_branch_metrics.label_ceiling` reported a single-draw reliability of **0.268**;
`oracle_gain.reliability` reports 0.216 here. Different statistics, not a disagreement:
`label_ceiling` takes the mean WITHIN-GROUP Spearman (equal vote per group), this pools the
demeaned Pearson (a group weighted by its own spread). Reliability rises with spread, so the
pooled figure sits above the equal-weight one on the same data -- and `oracle_gain.py` computes
the Spearman version too and **reproduces 0.2676 on exactly the data that produced 0.268**.
The extrapolation uses the Pearson one deliberately: `gain / sqrt(r)` comes from a variance
decomposition of `y = mu + e`, and the mean of a per-group rank statistic is not that ratio.

**The luck-inflated numbers this replaces were quoted elsewhere, and they reproduce here.**
The filtered-BC/AWR effort was stopped on a ceiling of `+0.0411` (realized return) and
`+0.0763` (survival), recorded as luck-inflated because selection and scoring shared a draw,
with the held-out version explicitly not computed. On **tile 6 @ 0.8 m/s, that same cell**,
this report's hindsight arm reads **+0.0381** and **+0.0715** -- so the two measurements agree
on the same quantity. The honest held-out values on that cell are **+0.0152**
[+0.0114, +0.0197] and **+0.0214** [+0.0072, +0.0360]: **2.5-3.3x lower than the ceiling that
decision was taken against**, which strengthens it rather than reopening it.

The extrapolation itself is **a model, not a measurement**: under a bivariate-normal
approximation a perfect predictor of the chunk's expectation would collect `gain / sqrt(r)` =
**+0.022 / +0.023**. The selfcheck shows the formula is exact under that model; nothing says
the outcomes obey it, and on a binary outcome it is cruder still. One significant figure.

## 6. What the live selectors collect

The ceiling above is one decision. A running selector acts at every re-plan, so it can exceed
it -- and one arm did. Three measurements, all `live_select.py`, cell-averaged with
cell-clustered intervals (`ksweep_report.py`):

| selector | ranking value | cells | measured on |
|---|---:|---|---|
| critic `V_priv.fold0` (v2) | **+0.167** [+0.106, +0.231], 9/0/0 | 9 | `sweep_v1` + `randsel_v1` |
| critic `V_priv.fold0` (v4) | **−0.133** (3 cells: −0.200, 0.000, −0.200) | 3 | `ksweep_v5` (stopped) |
| **realized tracking quality** | **−0.025** [−0.114, +0.047], 4/3/2 | 9 | `ksweep_v5_meas` |

"Ranking value" is `critic-pick(K) − random-pick(K)`: `--random_select` rolls and scores all K
candidates exactly as usual and commits a UNIFORMLY RANDOM one, so it differs in the choice
rule alone. The decomposition is additive on the cells all arms reached (residual 0.000000):

    critic ranking (critic - random)   +0.167 [+0.106, +0.231]   9/0/0   p=0.004
    re-keying      (random - K=1)      +0.008 [-0.028, +0.039]   4/2/3   p=0.688
    TOTAL          (critic - K=1)      +0.175 [+0.119, +0.233]   9/0/0   p=0.004

**Random-pick-of-K matches K=1**, so the K>1 machinery contributes nothing on its own and the
whole of the old gain is the critic's ordering. `--same_noise` sits +0.047 above random-pick,
so THAT flag is the confounded control, not `--random_select`.

**But the +0.167 was measured with an opening simulator transient present.** Every arm of
`sweep_v1`/`v3`/`v4_k8` ran with `cmd_hold_s = 0.0`; commit `58d0951c` made a 0.5 s
zero-command hold the default. On the same cells the fixed hold moves K=1 by **+0.118 mean**
and by **+0.450** on `t6_s2.4` -- nearly half of high-speed episodes were being lost to it. The
account that the critic had learned to damp that artifact, and little else, is consistent with
the v4 arm's collapse to −0.133 and with the branch probe's finding that the critic cannot rank
siblings at all on current code (within-state spread 4x BELOW its own two folds' disagreement,
SNR 0.24).

**Two pieces of evidence complicate that account and are recorded rather than dropped.** In
`sweep_v3`, none of the old selection gain is decided during the transient -- 0.000 of the 264
net episodes gained is decided in the first 0.72 s, and the timing tracks ARRIVAL AT THE
OBSTACLE scaled by speed (53% by re-plan 9 at 1.2 m/s, almost none before re-plan 18 at
0.6 m/s). And across the 5 cells with both numbers, the old ranking gain is INVERSELY related
to how much the hold helped (`t6_s2.4`: hold +0.450, ranking +0.075, the weakest of nine;
`t6_s1.6`: hold −0.125, ranking +0.250). Transient-damping predicts the opposite. A transient
that destabilises the gait and only manifests at the terrain would fit both, but that is a
hypothesis, not what was measured.

**The realized-quality arm is the sharper negative result.** It orders candidates by the mean
tracking reward MEASURED over the chunk each one just executed -- no model is read at all
(`--select_on measured`). It finds nothing: −0.025 [−0.114, +0.047], 4 better / 3 worse / 2
tied, sign p = 1.00; excluding `jump_on`, +0.017 [−0.054, +0.075]. One cell dominates the
negative (`t40_s2.4`, jump_on at 2.4 m/s, 0.700 -> 0.375). Its time-to-goal is +0.309 s
[+0.250, +0.369] slower, but the old `random − K=1` was +0.349 s [+0.210, +0.493], so that is
the re-keying rather than the rule; separating them needs a random arm on these cells, which
this run does not have.

That a MODEL-FREE realized quantity also finds no exploitable ordering points at the signal
not being there at this state distribution, rather than at the critic failing to capture it --
and it is consistent with a one-decision ceiling of one point. The caveat is that measured
quality is **myopic**: 0.24 s of tracking, not the outcome. This rules out one specific
realized quantity, not every one; scoring over 2-4 chunks is the same-cost next probe.

## 7. Caveats

* **Three tiles, one library, one pair.** `stairs_up` is 1138 of 1952 groups; `jump_on`
  contributes no ceiling signal because it never fails here. Nothing on `jump_off_front`, and
  no DR axis -- every branch ran the same nominal robot.
* **K = 4 only.** A gain over K is not comparable across K, and both tools refuse to pool
  cells that disagree on it. The K ∈ {1,4,8,16} sweep was stopped before completion
  (`logs/value/ksweep_v5_PARTIAL_stopped.txt` holds the log-derived rates; no per-episode
  data, so no time-to-goal for that arm).
* **The v4 critic number is 3 cells of log-derived rates**, not a completed sweep. Directional
  only.
* **The pair's robot fingerprints disagree on one field.** `tracker.onnx` and the generator
  carry different `mjcf_sha256`; `mjeval.runner.load_arm` allows it deliberately and SIM ONLY,
  on the measured argument that the field reaches nothing in this loop. Nothing on the robot
  path sets it.
* **The live arms span two GPUs.** The measured-quality run went on GPU 2 while the critic
  sweep held GPU 1; cross-GPU is not bit-reproducible here (11/12), so only DELTAS are
  compared across devices, never levels. Each arm carries its own in-process K=1. The five
  cells where both exist agree exactly on K=1, which is a free check that it does not bite.
* **A re-plan is probed one in four, and the commit is a uniform random sibling**, so the
  visited states are not shaped by any critic -- but they are also not the distribution a
  SELECTING policy would visit. That is what makes this a bound on one decision rather than on
  a policy.
