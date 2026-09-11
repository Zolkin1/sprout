# What is the generator's noise draw actually worth?

An upper bound on best-of-K selection for a humanoid terrain-tracking policy, measured by
varying only the flow-matching generator's initial noise and holding every other input fixed.

**Date:** 2026-09-01 · **Tracker:** `2026-08-20_16-36-10_mode11_decor_plan_v1` ·
**Generator:** `2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff` (sha `a596317e4d93`) ·
**Simulator:** Isaac Lab

## Headline

A perfect per-clip chooser of the generator's noise seed is worth about **+0.09 success**
overall, and at least **+0.14** on `stairs_down`. It is *not* the +0.195 the raw statistic
reports, nor the +0.030 a permutation control reports.

| estimator | value | status |
|---|---|---|
| raw | +0.195 | inflated by winner's curse |
| shuffle control | +0.030 | **over-corrected** |
| held-out | **+0.093** | unbiased — use this |
| current success | 0.667 | baseline |

## 1. The question

A flow-matching generator proposes a motion plan every 12 control steps (0.24 s); an RL tracker
follows it. The generator's only stochastic input is the initial noise `x0` of the flow.
Several experiments had asked whether a learned critic can *rank* samples from that
distribution, and all returned null.

A null from a selector experiment has two very different explanations: the critic cannot
identify good samples, or there are no good samples to identify. This measurement separates
them by removing the critic entirely and asking what an **oracle** would get — the ceiling on
any selection method whatsoever.

## 2. Isolating the variable

The premise is "change only the generator's sample." The environment's global `--seed` cannot
do this: it also moves domain-randomisation draws, spawn jitter and observation noise, so any
difference is attributable to nothing in particular.

Instead the flow's noise is served by an addressable source, keyed as:

```
x0 = f(fm_seed, replan_index_within_episode)     # NOT keyed by env
```

> **Load-bearing design choice.** Keying by environment is the obvious implementation and it is
> wrong. The R robots running one clip under one seed must be served the *identical* noise
> sequence, because their spread is the simulator-noise floor. Key by env and that floor
> contains the very effect it exists to measure.

The index is derived from the environment's own episode step counter rather than tracked
separately, so two robots at the same point of their own episodes agree by construction, and a
reset restarts the stream at zero.

## 3. How the data was gathered

One evaluation plan, executed 16 times, changing only `--fm_seed`. Domain randomisation off
(`dr_preset=none`, with the harness asserting every environment is the same robot), clips
pinned, terrain fixed.

| | |
|---|---|
| clips | 150 — 3 families × 10 tiles × 5 clips |
| seeds | 16 (0–15), plus 1 repeat of seed 1 as a control |
| robots / clip / seed | 10 |
| episodes | 24,000 (150 × 16 × 10) |
| parallel envs | 750 — 1000 exhausts a 12 GB card |
| wall clock | ~6 min per seed, ~1.7 h total |
| clip library | `g1_mb_v5_body_mode11_decor.h5` (10,000 clips) |
| clip plan hash | `960959b01596e580` |

**Success criterion.** An episode succeeds if the base **entered the goal region before it ever
crossed the fall predicate**. Arriving and then falling in the run-out counts; falling and later
being dragged over the line does not. Success requires *arrival*, not survival — a purely
temporal criterion is maximised by standing still.

## 4. Distribution of the data

Overall success **0.667**. Per-clip rates are broadly spread rather than saturated, which is the
regime this measurement needs — a clip pinned at 0 or 1 carries no information about seed
choice.

| per-clip success rate | clips |
|---|---|
| 0.00 – 0.10 | 11 |
| 0.10 – 0.25 | 12 |
| 0.25 – 0.50 | 13 |
| 0.50 – 0.75 | 31 |
| 0.75 – 0.90 | 46 |
| 0.90 – 1.00 | 37 |

Two clips sit at exactly 0.0; none at exactly 1.0. Each clip's rate is over its own 160
episodes.

| family | clips | success | clips at 0.0 |
|---|---|---|---|
| jump_on | 50 | 0.575 | 0 |
| stairs_down | 50 | 0.654 | 0 |
| stairs_up | 50 | 0.770 | 2 |
| **all** | **150** | **0.667** | **2** |

## 5. The statistic, and why the obvious ones fail

For each clip, let `s_k` be seed *k*'s success rate over its 10 robots:

```
current    = mean_k s_k          (the clip's overall rate)
attainable = max_k  s_k
gain       = mean over clips of (attainable - current)
```

**Why the raw gain is wrong.** `max_k` is biased upward even when the seed does nothing at all.
Ten robots wobble; a maximum over sixteen wobbles reliably finds a high one. This is the
winner's curse.

**Why the permutation control is also wrong.** The natural fix is a shuffle: within each clip,
pool all 160 episodes, randomly repartition into 16 pseudo-seeds of 10, recompute. That destroys
any seed effect while preserving the noise exactly.

> **Correction — the error most worth propagating.** The shuffle measures how much `max`
> inflates *under the null of no seed effect*. When an effect is real, the maximum is partly
> driven by genuinely better seeds and is less noise-inflated than the null predicts — so
> subtracting the full null bias **over-corrects**. On synthetic data with known truth it
> recovered only **31%** of the real effect.
>
> The shuffle remains a valid **test** (it correctly returns zero under a synthetic null, so
> "the effect is real, p < 0.001" stands). It is not a valid **estimator of magnitude**.

**The estimator that works.** Select the best seed on a subset of robots, then score that seed
on the **held-out rest**. The selection can be lucky; the score cannot. Sweeping the selection
sample size shows whether the estimate is still selection-limited or has converged on the
oracle.

## 6. Validating the estimators

Each estimator was run on synthetic data with a known answer before being trusted on real data:
150 clips, 16 seeds, 10 robots, per-seed probabilities drawn with a known spread.

| scenario | truth | raw | shuffle net | held-out | variance |
|---|---|---|---|---|---|
| null (no effect) | +0.000 | +0.235 | −0.003 | −0.005 | +0.033 |
| effect sd = 0.13 | +0.220 | +0.307 | +0.066 | +0.116 | +0.160 |
| effect sd = 0.25 | +0.351 | — | — | +0.264 | +0.339 |

The shuffle recovers 31% of a real effect. Held-out is unbiased under the null and conservative
under an effect. The variance-based estimator recovers most of a real effect but carries a
+0.033 upward bias under the null and assumes normality that bounded rates violate.

## 7. Result

| family | n | raw | shuffle net | **held-out** | variance |
|---|---|---|---|---|---|
| jump_on | 50 | +0.179 | +0.021 | **+0.069** | +0.193 |
| stairs_down | 50 | +0.248 | +0.049 | **+0.138** | +0.248 |
| stairs_up | 50 | +0.158 | +0.020 | **+0.070** | +0.181 |
| **all** | **150** | +0.195 | +0.030 | **+0.093** | +0.207 |

Held-out value as the selection sample grows — flat means selection is already reliable and the
estimate *is* the oracle; still rising means the true ceiling is higher:

| family | sel=2 | sel=3 | sel=5 | sel=7 | sel=9 |
|---|---|---|---|---|---|
| jump_on | +0.051 | +0.056 | +0.062 | +0.065 | +0.069 |
| stairs_down | +0.054 | +0.074 | +0.107 | +0.129 | +0.138 |
| stairs_up | +0.053 | +0.061 | +0.067 | +0.072 | +0.070 |
| **all** | +0.054 | +0.064 | +0.080 | +0.088 | +0.093 |

`jump_on` and `stairs_up` have flattened, so those are effectively the oracle. `stairs_down` is
**still climbing at sel=9**, so its +0.138 is a floor, not a ceiling.

**Corroborating decomposition.** A repeat of one seed in a second process gives an independent
read; binomial noise cancels in the subtraction:

```
Var(per-clip diff, same seed) = 2·sigma_run^2
Var(per-clip diff, diff seed) = 2·sigma_run^2 + 2·sigma_seed^2

sigma_run = 0.053    sigma_seed = 0.130    ratio 2.44x
```

Switching seed moves a typical clip's success rate ~13 points against ~5 points of run-to-run
noise.

## 8. What this implies

**Prior selector nulls were uninformative.** The selector A/B experiments that motivated this
work had confidence half-widths near 0.055 against a true ceiling of 0.093. A perfect critic
would have been hard to distinguish from no critic. Those nulls do not show the critics are bad;
they show the experiments could not have detected success.

**Per-decision selection is far weaker than whole-episode selection.** A separate oracle that
branched at a single re-plan and rolled candidates to termination found +0.067 raw against a
+0.058 floor. Changing one plan out of ~30 barely matters; changing all 30 is worth ~0.09. The
effect compounds across decisions rather than living in any one.

**`stairs_down` is where the headroom is.** Most available gain (≥0.138), worst generator
performance, and the only family where a steering intervention made things worse.

## 9. Limitations

- **Run-to-run reproducibility fails.** The same seed in two processes reproduced per-clip rates
  exactly in only 58% of clips. Isaac Lab documents determinism for re-running a simulation;
  that did not hold here. The floor therefore has a run-to-run component that must be measured,
  not assumed.
- **The floor is large.** Ten robots given an identical draw disagree on ~70% of clips. Any
  measurement of this kind without an explicit noise control will be dominated by it.
- **Three families only** — `jump_on`, `stairs_down`, `stairs_up`. No flat terrain, no jump-off.
- **The oracle is per clip, not per episode.** It chooses one seed for a clip, so it bounds a
  clip-level selector specifically.
- **Unresolved cross-simulator disagreement.** A MuJoCo probe of the same generator reported
  20–35% of outcomes flipping with the noise draw. That is not obviously compatible with a 0.09
  recoverable headroom.

## 10. Reproducing it elsewhere

1. Make the generator's noise addressable, keyed by `(seed, replan index)` and *not* by
   environment.
2. Verify the seed actually changes outcomes before spending compute. A silently inert flag
   produces a clean, meaningless null.
3. Run the same seed twice in separate processes to measure run-to-run noise.
4. Report raw, shuffled and held-out side by side. Never the raw number alone.
5. Sweep the selection sample size; if the held-out value is still climbing, the ceiling is above
   your estimate.
6. Do not use `p(1−p)/R` as a binomial floor. Robots sharing a draw are strongly correlated —
   measured same-seed variance was ~8× smaller than independence predicts.
