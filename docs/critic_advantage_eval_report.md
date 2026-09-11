# Is the critic good enough for advantage estimation?

An evaluation of the v4 trackability critic as a source of advantages for AWR / filtered-BC
fine-tuning of the flow-matching generator, run before spending cycles on the fine-tune itself.

**Date:** 2026-09-02 · **Tracker:** `2026-08-20_16-36-10_mode11_decor_plan_v1` ·
**Generator:** `2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff` (sha `a596317e4d93`) ·
**Critic:** `logs/value/v4/V_priv.fold{0,1}.pt` + `V_onboard.fold{0,1}.pt` (2-fold cross-fit,
6 epochs, `quality_bins=51`, `gamma=0.95`) · **Data:** `logs/rollouts/v4_big`, 294,252 windows
over 15,037 episodes on 76 tiles (19 held out entirely) · **Simulators:** IsaacLab (the
recording) and MuJoCo (`branch_probe.py`, the counterfactual)

Generated tables and figures: `logs/value/critic_eval_v4/tables.md`.

## Headline

| question | answer | where |
|---|---|---|
| does the critic predict held-out outcomes? | **yes** — AUROC 0.897, ECE 0.014 on tiles it never saw | §3 |
| is it memorised? | **no** — in/out AUC gap 0.016 [0.014, 0.019] against a 0.1 bar | §3 |
| can it tell two actions apart at ONE state? | **barely** — SNR 0.24, ρ = 0.02–0.18 against a ceiling of 0.52 | §4 |
| is the critic-bootstrapped advantage usable? | **no** — ρ = 0.02–0.18, at most a third of `A_MC` | §5 |
| is the Monte-Carlo advantage usable? | **yes** — ρ = 0.268, p = 0.0005, best-pick 0.70 vs 0.50 | §5 |
| what does the critic contribute to it? | **61% of the return's variance**, as a baseline | §6 |
| does the offline arm agree? | **yes, independently** — `A_MC`'s within-episode signal is 1.9× `A_TD`'s | §7 |
| would a filtered-BC set differ from the outcome-only one? | **yes**, 26% of chunks change class | §7 |
| **recommendation (as written 2026-09-02, before the fine-tune)** | **GO, on `A_MC` / high-λ GAE over the QUALITY head** | §8 |
| **outcome (same day, after running it)** | **STOP — the prize is 1.6% of the reward's variance, and half of it was already taken** | §10 |

> **Read §10 before acting on §8.** The GO above was right about the critic and wrong about
> how much room there was to win. The fine-tune was run, and the ceiling was then measured
> directly rather than assumed.

## 1. Two uses of one critic, and why the second does not follow from the first

Best-of-K selection asks the critic to order `K` futures that are one 0.24 s chunk downstream
of the *same* state and differ only by a flow-noise draw. Advantage weighting asks it to order
chunks *across the dataset*. Those are different questions, and the first failing says nothing
about the second — which is the hypothesis this report was written to test.

The distinction has a precise form. Write the two estimators out:

```
A_TD(s_t, a_t) = V(s_{t+1}) - V(s_t)      action-dependence runs ENTIRELY through the critic
A_MC(s_t, a_t) = G_t        - V(s_t)      action-dependence runs through the REALIZED return
```

Under `A_TD` the critic must resolve the different `s_{t+1}` that different chunks produce.
Under `A_MC` it is only a **baseline**: a function of the state alone, whose job is variance
reduction, and which is unbiased for *any* such function. So a critic can be hopeless at the
first and perfectly adequate at the second, and the measurements below separate them.

## 2. What was measured, on what

Two arms, because no offline statistic can answer the action question:

* **Offline** — the recording holds exactly one action per state, so it can establish that the
  critic predicts outcomes, that the advantage varies within an episode, and what the implied
  weights do to the dataset. It cannot say whether a *different* chunk would have done better.
* **Counterfactual** (`transfer/sim/branch_probe.py`) — `K = 4` chunks drawn from one state in
  MuJoCo, each rolled to completion, so each carries a realized outcome the live loop can never
  observe for more than one of them. Random commit (so the visited distribution is not shaped
  by the critic being evaluated), common random numbers on the continuations, and a second
  independently-keyed continuation on **every** sibling.

**The blocking control passes exactly.** `--same_noise` hands every sibling one draw, so no
group can be informative: 29 groups, **0** outcome-disagreeing, **0.000** quality spread. The
harness manufactures no signal — the check the abandoned Isaac best-of-K harness failed.

Everything is read out of fold: each row is scored by the fold that did not train on its
episode, and 19 of 76 tiles were held out of both folds.

## 3. The critic is a good STATE evaluator

| variant | scope | AUROC | ECE | AUROC by episode phase (5 bins) |
|---|---|---|---|---|
| `V_priv` | seen tiles | 0.923 | 0.0068 | 0.871 · 0.889 · 0.921 · 0.955 · 0.982 |
| `V_priv` | **unseen tiles** | **0.897** | 0.0143 | 0.797 · 0.814 · 0.885 · 0.956 · 0.981 |
| `V_onboard` | unseen tiles | 0.874 | 0.0178 | 0.778 · 0.789 · 0.873 · 0.944 · 0.979 |

The quality head predicts the *realized* normalized tracking return at Spearman **0.794** on
held-out tiles (0.811 overall). Discrimination rises smoothly through the episode rather than
being flat-and-high, which is what separates a critic that reads the state from one that has
memorised which tile is hard. Median lead time is 9–10 chunks (2.2–2.4 s) before the fall and
before the end respectively, on held-out tiles.

The privileged-vs-onboard gap is small — 0.897 against 0.874 AUROC — so most of the
outcome-relevant information is already in what the generator itself can see.

**It is not memorised.** Scoring every row with BOTH folds gives, on the training tiles, the
reading from the fold that held the episode out (`V_out`) beside the one from the fold that
trained on it (`V_in`). The gap is small and its tile-clustered interval excludes anything
material:

| head | n | AUC out [95% CI] | AUC in | **gap [95% CI]** | ρ(V,G) out | ρ in |
|---|---|---|---|---|---|---|
| survival | 220,363 | 0.923 [0.912, 0.932] | 0.939 | **0.016 [0.014, 0.019]** | 0.568 | 0.590 |
| quality | 208,722 | 0.865 [0.845, 0.885] | 0.883 | **0.018 [0.015, 0.020]** | 0.816 | 0.839 |

Against the spec's 0.1 threshold that is a comfortable pass, and the held-out-*tile* scope is
stricter still — neither fold ever saw those tiles, and performance holds there too. Fold
assignment was recovered by matching each re-scored reading against the fit's own
`predictions.npz` (median reproduction error 1.7e-6); on 5.4% of rows the two folds agree to
within that noise, where `V_in` and `V_out` are interchangeable and the gap is unaffected.

One flag the spec asks for: **AUROC at the first chunk, on unseen tiles, is already 0.80**
(survival) / 0.76 (quality) over 3,396 episodes. The outcome is substantially predictable
before anything has happened — the depth image genuinely shows the terrain ahead — so early
discrimination is not evidence of leakage here, but it does mean a large part of the critic's
skill is reading the *situation* rather than the *execution*.

## 4. It cannot resolve two actions at one state

`branch_probe` reads the critic at the state each sibling *arrives at*, which is exactly
`V(s_{t+1})`. Within a group `V(s_t)` is a shared constant, so **the spread of that reading
across siblings IS the spread of `A_TD`**. Measured over 241 groups (988 branches) on
stairs-up at 1.6 and 2.4 m/s, stairs-down and jump-on:

| quantity | within-group median \|pairwise\| | fold disagreement median \|pairwise\| | SNR |
|---|---|---|---|
| `p_fail` | 0.0059 | 0.0250 | **0.24** |
| `G_hat` | 0.0095 | 0.0341 | **0.28** |

Both sides are the same statistic (a median absolute pairwise difference), so the ratio does
not depend on `K`. The differences the critic reports *between* siblings are four times smaller
than how much two honest fits of the same critic disagree about *one* sibling.

**And that comparison flatters the critic, by a measured factor.** Two cross-fit folds share a
frozen generator trunk, an architecture, the held-out tiles and the targets, so their errors
are not independent and their disagreement misses whatever they get wrong together. Measured
on the 70,493 held-out-tile rows, the errors of `V_priv` and `V_onboard` — which differ in
their *inputs* as well as their training split, and are therefore a **lower bound** on how
correlated two same-input folds would be — correlate at **+0.87** (`G_hat`) and **+0.91**
(`p_fail`). At that correlation a fold-difference noise floor understates the true error
standard deviation by 2.8× and 3.4× respectively. The real SNR is lower than 0.24, so this is
an **upper bound** on the resolving power, not an estimate of it.

## 5. Which estimator survives that

Both estimators ranked against the **same** independent continuation draw, on the same groups:

| estimator | ρ (within group) | p | best-pick | null |
|---|---|---|---|---|
| `A_TD` — critic at `s'`, `V_priv` f0 / f1 | +0.065 / +0.023 | 0.36 / 0.73 | 0.606 / 0.606 | 0.50 |
| `A_TD` — `V_onboard` | +0.177 | 0.006 | 0.662 | 0.50 |
| **`A_MC` — realized return** | **+0.268** | **0.0005** | **0.704** | 0.50 |

71 groups carrying a second continuation draw. `p` is a within-group permutation null, which
preserves the group structure and destroys only the association under test. (A global label
shuffle is vacuous here — its expectation is 0.5 analytically.)

**`V_onboard` orders siblings better than `V_priv` does**, and by enough to clear a Bonferroni
correction over the three critics. That is the opposite of the state-level ranking, where the
privileged critic wins (AUROC 0.897 vs 0.874) — so the dense scan, proprioception and linear
twist buy prediction of *which state is hard* while adding nothing to *which chunk was better*.
Worth knowing if a bootstrapped advantage is ever revisited; on its own it does not rescue
`A_TD`, which still reaches only a third of `A_MC`.

**The MC advantage is optimal by construction at the chunk level, and that is why it wins.**
Within a group `V(s_t)` is constant, so ranking siblings by `A_MC` *is* ranking them by realized
return. Nothing that is a function of the state can beat it. Its only limitation is how noisy
one realized return is as a measure of the chunk that preceded it, and that is measurable:

| | value |
|---|---|
| single-draw return reliability `r` (71 groups) | 0.268 |
| oracle ceiling `sqrt(r)` — a perfect predictor of `E[outcome]` | **0.517** |

So `A_MC` captures **52%** of what an oracle could (0.268 / 0.517); `A_TD` reaches 34% with the
onboard critic and 4–13% with the privileged one. **`A_MC`'s residual is label noise, not model
error** — and label noise averages out over
294k chunks, where a selector acting on one state's `K` samples cannot average it away. That is
the mechanism by which the same critic is useless for selection and adequate for weighting.

**Stratifying by how much the siblings' outcomes actually differed makes the noise floor
visible.** Groups whose realized returns barely differ cannot be ordered by anything, so the
meaningful stratum is the widest one — and there the critic's own two folds disagree in sign:

| critic | realized ΔG band | groups | ρ | p | best-pick (null 0.50) |
|---|---|---|---|---|---|
| `V_priv` f0 | 0.15+ | 47 | **−0.196** | 0.026 | 0.362 |
| `V_priv` f1 | 0.15+ | 47 | −0.085 | 0.304 | 0.468 |
| `V_onboard` | 0.15+ | 47 | +0.187 | 0.016 | 0.638 |

Two honest fits of the same model, on the same 47 groups, land at −0.20 and −0.09; a third
model lands at +0.19. Across nine strata one crossing p = 0.05 in the *wrong* direction is
unremarkable on its own — the point is the **spread between fits**, which is exactly what §4's
SNR of 0.24 predicts and what makes any single one of these numbers uninterpretable.

## 6. What the critic actually contributes under `A_MC`

As a baseline, judged on variance removed:

| head | sd(`G − mean G`) | sd(`G − V`) | variance removed |
|---|---|---|---|
| quality | 0.240 | 0.150 | **61%** |
| survival | 0.375 | 0.274 | 47% |

Unbiased, since `V` is a function of the state alone. This is real gain at no cost in bias, and
it is the critic's job under `A_MC` — not resolution.

**Credit localises, and `A_MC`'s blame lands where it should.** On the 3,788 failed held-out
episodes, the most negative advantage sits a median of **3 chunks (0.72 s) before the end**
(p25 −6, p75 −1), with only 16% on the final chunk — so the label is not merely "the episode
was about to end". `A_TD` localises tighter (median −2, 32% on the last chunk) but that
tightness is bought with the unreliable variance §7 measures. `A_base` is the control and
behaves exactly as it must: 99.9% on the last chunk, because it is constant within an episode
and carries no localisation at all.

## 7. The estimator sweep, and what a fine-tune set would look like

`V_priv`, quality head, 279,215 rows with a valid return. `varW` is the fraction of the
advantage's variance that lives *within* an episode; `R²out` is how much of it a saturated fit
on the episode outcome alone explains; `novel` is the share of chunks a per-family 35th-
percentile cut classifies differently from "keep every chunk from a successful episode".

| estimator | sd | varW | R² from outcome | **reliability WITHIN ep** | ρ with `A_base` | ESS ep @ β=1 | novel |
|---|---|---|---|---|---|---|---|
| `A_base` | 0.240 | 0.063 | 0.549 | 1.000 | 1.000 | 0.674 | 0.191 |
| `A_TD` (λ=0) | 0.077 | 0.937 | 0.017 | **0.417** | 0.128 | 0.166 | 0.363 |
| `A_GAE` λ=0.5 | 0.078 | 0.856 | 0.049 | 0.531 | 0.233 | 0.181 | 0.341 |
| `A_GAE` λ=0.8 | 0.093 | 0.691 | 0.116 | 0.626 | 0.374 | 0.225 | 0.303 |
| `A_GAE` λ=0.95 | 0.124 | 0.431 | 0.209 | 0.659 | 0.484 | 0.289 | 0.264 |
| **`A_MC` (= λ=1)** | 0.150 | 0.294 | 0.248 | **0.669** | 0.510 | **0.444** | **0.255** |

**The `varW` column is a trap, and the reliability column is the reason.** `A_TD` puts 94% of
its variance inside episodes and is almost unexplained by the outcome, which *looks* like ideal
credit assignment — but on synthetic data a **pure-noise** critic produces the same 94%.
Within-episode variance is only worth having if it is signal, and de-meaning each episode
before correlating the two folds says how much of it is: `A_TD`'s within-episode variation is
**42% signal**, `A_MC`'s is **67%**. In absolute terms the signal variances are 0.0023 and
0.0044 — so **`A_MC` carries ~1.9× more within-episode signal than `A_TD`**, monotonically in λ.

**That is the offline arm reproducing the counterfactual's verdict from entirely different
data.** §5 ranks the two estimators against realized MuJoCo outcomes at 71 groups; this ranks
them by cross-fold agreement over 208,722 recorded windows. Neither can see the other, and both
put `A_MC` ahead of `A_TD` by about a factor of two.

`A_MC` keeps a third of the dataset at β=1 and reclassifies **26%** of chunks relative to the
outcome-only rule — so a filtered-BC run on it is not a re-run of BC on successful episodes. Of
its positives, 5.0% come from episodes that FAILED, and 63% of its negatives come from episodes
that survived: the label is genuinely per-chunk, not the trajectory outcome wearing a
percentile. (`A_TD` reclassifies more — 36%, with 13% of positives from failed episodes — but
§7 shows that extra disagreement is mostly its unreliable half.)

**Quote the ESS in EPISODES, not windows.** ~22 windows of an episode share its terminal label
and are adjacent states, so the window count overstates the independent units an update
actually sees. Aggregating each episode's weight and recomputing:

| β | ESS (windows) | ESS (**episodes**) | top 10% of episodes carry |
|---|---|---|---|
| 0.3 | 4,082 (1.5%) | **850** (5.8%) | — |
| **1.0** | 93,622 (33.5%) | **6,547** (44.4%) | 35.3% of the weight |
| 3.0 | 253,159 (90.7%) | 10,446 (70.9%) | — |

β=1.0 keeps 6,547 of 14,733 episodes and spreads the mass broadly (the top 1% of episodes
carry 6.8%, the top 25% carry 61.8%) — a usable dataset by either measure. β=0.3 is worse than
its window figure suggests: 850 effective episodes is where an update starts being driven by a
handful of trajectories, which is the argument for starting at 1.0 rather than sweeping down to
it.

## 8. Recommendation

**Go — try the fine-tune, on the Monte-Carlo advantage over the quality head.** Specifically:

1. **Estimator: `A_MC`, or GAE at λ ≥ 0.95.** Not `A_TD`, and not a low λ. The bootstrapped
   term is measurably below its own noise floor, and every λ below ~0.8 is mostly that term.
2. **Head: `quality`.** The survival head's return is constant within an episode, so its `A_MC`
   gets within-episode variation only from the baseline and inherits `A_TD`'s problem. The
   quality head has a genuinely dense per-chunk return.
3. **Baseline: `V_priv`.** It removes 61% of the return's variance and its privileged inputs
   cost nothing here — the baseline is never deployed.
4. **Temperature: start at β = 1.0** on the standardized advantage (ESS 0.34). β = 0.3 collapses
   the effective sample to 1.5% and is not worth trying first.
5. **Expect the gain to come from credit assignment across the dataset, not from picking better
   actions at a state.** If the fine-tune is going to work, it works by averaging a 0.27-reliable
   per-chunk label over ~10⁵ chunks — 6.5k effective episodes at β=1.

**What this evaluation does NOT establish, and it bounds the upside.** The fine-tune is
advantage-weighted *self-distillation*: `rollout_dataset.py`'s target is the generator's own
emitted chunk, so the model is taught to reproduce its own actions with some amplified and some
suppressed. It has no access to a better action and cannot invent one. Everything measured here
says the *label* is sound — it orders chunks, it is not a restatement of the episode outcome, and
it survives its noise floor. Whether reweighting a conditional toward its own better draws moves
closed-loop success is a different question, and the only instrument that answers it is the
fine-tune itself. That is why the recommendation is "worth the cycles", not "this will work".

**One implementation note, and it is small.** `flow.flow_matching_loss` reduces with a single
`sq.mean()` and takes no per-sample weight, so continuous AWR weights have nowhere to go today;
the label currently reaches the model through a 3-way embedding. Adding a per-sample weight is a
contained change and is what §7's β grid assumes.

## 9. What would change this answer

- **The critic has only ever scored a perfect camera.** `record_rollouts.py` runs the `-play`
  task, which nulls all six depth DR events and sets `enable_corruption=False`. Nothing here
  says how the baseline behaves on a noisy image.
- **The label is horizon-tied.** `failed` means "failed within this harness's budget", which in
  Isaac is `clip_end`; mjeval runs 1.4–3.6× longer. Discrimination is horizon-robust,
  calibration is not, and any score mixing `p_fail` with `G_hat` needs their levels
  commensurable. `TODO(critic-horizon-feature)` is open.
- **Episodes start mid-clip at the clip's velocity**, and every deploy episode starts standing
  at rest. `TODO(critic-standing-starts)` is open.
- **The counterfactual is three tiles.** stairs-up, stairs-down and jump-on; `jump_off_front` is
  unreachable because `branch_probe` has no `--klass` flag and `klass_for_family` refuses that
  family. The survival half rests on few outcome-disagreeing groups and should be read as
  directional.
- **Two things were deliberately not measured.** *Chunk diversity* (mean pairwise distance
  between the `K` sampled chunks, and its correlation with realized spread) needs
  `branch_probe` to store the emitted chunk, which it does not — that is a schema change plus
  a re-run, and the realized-outcome spread it would explain is already reported directly. And
  the *short proxy fine-tune* was not run: it is the natural next step, not a prerequisite,
  and it is the thing this report exists to decide whether to spend.
- **8,192 of 23,229 episodes are censored** (one per env, cut at the window ceiling) and are
  dropped, leaving 15,037. `spawned_in_goal` (12 live episodes) and `goal_travel_m <= 0`
  (2 live) are *not* dropped and are free successes — too few to matter here, but they are
  labelled successes that required nothing.

## 10. Outcome: the fine-tune was run, and the effort is STOPPED

§8 recommended a filtered-BC attempt on `A_MC`. It was run the same day. **The critic was not
the limiting factor — the generator's own conditional was**, and this section exists because
nothing above measured that.

### What the fine-tunes did

Two arms, both at LR 3e-5 and 1e-4, evaluated in MuJoCo (27 cells / 1080 episodes) and Isaac:

| | tracking reward | success |
|---|---|---|
| `amc` (filtered on `A_MC`, top 30% per family × progress × speed) | **+0.011 to +0.014**, CIs exclude zero | no change |
| `survived` (all chunks of surviving episodes, NO critic) | matched or beat `amc` everywhere | no change |

**The success null is informative, not underpowered**: the same cells and the same episodes
detect the reward effect, and bound success at ±0.036.

### The ceiling, measured causally

`transfer/sim/branch_probe.py` draws K=4 sibling chunks from one **exactly restored** MuJoCo
state — only the flow-noise draw differs — and rolls each to completion. 340 groups, tile 6 @
0.8 m/s. The blocking control passes exactly: under `--same_noise` the within-group spread is
`0.000e+00`, so the restore is bit-exact and the treatment arm measures the action rather than
the simulator. (This is the check the abandoned Isaac best-of-K harness failed.)

    Var(R | same state) / Var(R) = 0.016

    perfect best-of-4 gain     tracking reward  +0.0207 on a base of 0.7373   (+2.8%)
                               realized G       +0.0411                        (+7.1%)
                               survival         +0.0763                       (+12.2%)

**The tracking-reward figure is exact, the other two are luck-inflated.** `r_exec` is measured
over the branch chunk *before* any continuation seed is drawn, so it carries no label noise —
verified directly, its within-group spread under `--same_noise` is `0.000e+00`. `G` and survival
depend on the continuation, so their oracle selects partly on luck; the honest version (select
on seed A, score on seed B) was **not** computed. The old probe data cannot give it — its second
seed ran on sibling 0 only (`{0: 37, 1: 0, 2: 0, 3: 0}`) — and the re-run with the current code
(`--second_seed_every 1`, ~17 s/group) was stopped at 14 groups.

So: the fine-tunes captured **more than half of all the room that exists** in chunk choice on
the metric they moved, and the remaining lever is about +0.008.

### An independent method agrees

`scripts/flow_matching/advantage_aliasing_probe.py`, 60k Isaac windows, kNN in the generator's
own conditioning space (local Var of the score among neighbours, against its global Var):

| score | variance explained by (state, action) | contributed by the ACTION |
|---|---|---|
| `A_MC` | 0.09 | **0.0%** |
| tracking reward, demeaned by family × progress × speed | 0.56 | **1.1%** |

1.1% observational against 1.6% causal, from unrelated instruments.

### Three traps in that probe, each of which produced a plausible wrong reading

1. **`history` is PAST ACTIONS.** `record_rollouts.py:401` stores the generator's own
   previously-served nodes, and they linearly predict **45%** of the next chunk (held out *by
   episode* — adjacent windows share three of four history nodes, so a row split leaks). Drop
   history and the action's apparent share jumps 1.1% → 10.4%. That is the action standing in
   for removed state, not a discovery. Quote the with-history number.
2. **The t-SNE's structure is terrain**, not advantage: family kNN lift 2.63×, advantage sign
   1.00×, held-out-tile control 1.00×.
3. **Raw tracking reward is 16.4% between-stratum**, so colour by it demeaned or you re-find
   terrain and call it signal.

### Two embedding bugs found while doing this, both silent

- `_block_embed` re-standardized every retained PCA component to unit variance. `cmd_future` is
  192 dims of a command that barely moves within a re-plan window and has effective rank 10, so
  components 10–15 (down to `3.07e-08` of the block against a leading `3.35e-01`) were float32
  round-off divided by its own size — one window landed at 236σ. That both hijacked the global
  PCA (its first component became "the direction isolating that row") and gave six of 64
  dimensions to noise, which blurs neighbourhoods and biases the statistic *toward* the
  no-separation answer. A variance floor was the first fix and was **not** sufficient: the block
  is near-degenerate in a graded way, and on the reward row-set a component survived a 1e-4
  floor and still reached 110σ. The fix is a **single scalar per block** — no threshold, so a
  direction carrying no variance contributes no distance on any data.
- `_zscore` pooled its mean over non-finite values. The height scan carries `-inf` for a missed
  ray by construction, which makes a column's mean `-inf`, every centred value NaN, and the
  finite-guard then zeroes the **whole column** — so the `proprio,scan,goal,twist_linear` arm
  had been silently returning numbers identical to proprio-only.

Both are pinned by `source/robot_rl/tests/test_aliasing_probe_embedding.py`, including a test
asserting the unfixed path is visibly wrong so the fix cannot be silently reverted.

### What would reopen this

Not a better critic. **The ceiling is a property of the generator's conditional, not of the
scorer** — it is how much two chunks drawn at one state can differ in consequence. It reopens if
that spread grows (a wider sampling temperature, a less peaked conditional), or if the target
changes to one with more room: survival's ceiling is 4× the reward's, and its held-out value was
never measured. See the sampling-variance result — 20% of draws fall at 0.8 m/s — for why the
*survival* headroom is real even though the reward headroom is not.

### Caveats on the ceiling

- **One cell.** Tile 6 (stairs-up) at 0.8 m/s, chosen because K=1 success sits near 0.5 there,
  which maximises the survival arm's power. It is not a claim about the whole distribution.
- **K=4.** A larger K raises the oracle gain; the *variance* decomposition (1.6%) does not
  depend on K, and it is the more portable number.

## Appendix A: how to reproduce

Offline arm (one GPU pass over the recording's exteroception, ~25 min at `--batch 256`):

```bash
PYTHONPATH=source/robot_rl:source/motion_gen python scripts/flow_matching/critic_advantage_eval.py \
    --stage score --variants V_priv \
    --rollout logs/rollouts/v4_big/rollout_2026-08-20_16-36-10_mode11_decor_plan_v1 \
    --fm_checkpoint logs/flow_matching/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff/best.pt \
    --fit_dir logs/value/v4 --out logs/value/critic_eval_v4
python scripts/flow_matching/critic_advantage_eval.py --stage metrics --out logs/value/critic_eval_v4
python scripts/flow_matching/critic_advantage_eval.py --stage report  --out logs/value/critic_eval_v4
```

`score` re-scores every row with BOTH folds, which is what gives `V_in` beside `V_out` and, on
held-out tiles, two honest estimates whose disagreement is the noise floor. It checks itself
against the fit's own `predictions.npz` and records the verdict as `recovery_ok` in the output.

Counterfactual arm (MuJoCo; ~4 min per episode per cell on an RTX 4070). The critics must carry
`fm_checkpoint_sha256`; if they were fitted elsewhere and do not, adopt the local generator with
`scripts/flow_matching/stamp_critic_generator.py` after hashing it on the box that fitted them:

```bash
cd transfer/sim && PYTHONPATH=../../source/robot_rl:../../source/motion_gen python branch_probe.py \
    --stage run --out ../../logs/value/critic_eval_v4/bprobe_t6s24 \
    --tracker_onnx ../../logs/rsl_rl/g1/<tracker>/exported/tracker.onnx \
    --fm_onnx      ../../logs/rsl_rl/g1/<tracker>/exported/<fm>_fm.onnx \
    --clip_lib     ../../clips/g1_mb_v5_body_mode11_decor.h5 \
    --fm_pt        ../../logs/flow_matching/<fm run>/best.pt \
    --critic f0=.../V_priv.fold0.pt --critic f1=.../V_priv.fold1.pt \
    --tile 6 --speed 2.4 --k 4 --episodes 30 --probe_every 4 --second_seed_every 4 \
    --scene_tag _t6s24            # a DISTINCT tag per concurrent run, or the scene file tears
# ... and the blocking control, which must return 0 informative groups:
#     same command plus --same_noise --out .../bprobe_ctrl --scene_tag _ctrl
python scripts/flow_matching/critic_branch_metrics.py --branches .../bprobe_*/branches.h5 \
    --fold_pair f0 f1 --out logs/value/critic_eval_v4/branch_metrics.json
```

`--fold_pair` names the two tags that are cross-fit folds of ONE variant; without it the
resolution statistic is skipped rather than computed over two different models.

## Appendix B: defects found and fixed while building this

| where | defect | effect |
|---|---|---|
| `value_gates.spearman` | tie-averaging scanned the whole array once per distinct value | O(n²); never returns at 294k rows. Replaced with sorted-run averaging, bit-identical to the old code and to scipy, 503× faster at n=100k |
| `critic_eval.td_residuals` | closed the terminal on the placeholder reward `chunk_rewards_from_track_err` leaves on each episode's last row | biased every λ>0 GAE, confounded with the outcome. λ=0.95 read ρ(`A_base`) = −0.28 before the fix and +0.48 after |
| `critic_branch_metrics` | survival head ranked with no finite mask | `argsort` sorts NaN last, so the 12 branches the critic never scored were credited as its best picks |
| `critic_branch_metrics` | SNR compared a range over `K` against a pairwise difference | inflated ~2× and grew with `K` |
| `branch_probe.py` | second continuation rolled for sibling 0 only | no ceiling; "ρ = 0.1" and "no ranking is stable" were the same observation. Now every sibling |

`branch_probe.py`'s `failed` column is also True whenever a branch terminated inside its own
chunk **including by reaching the goal** (4–20 rows per run here); `critic_branch_metrics`
corrects it on load and reports both rates.
