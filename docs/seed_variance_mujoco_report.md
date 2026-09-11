# Does the generator's noise draw have transferable quality?

A 10,080-episode MuJoCo sweep isolating the flow-matching seed as the only free variable, with
a heading-controller A/B and an artifact-controlled best-of-K measurement.

**Date:** 2026-09-01 · **Tracker:** `2026-08-20_16-36-10_mode11_decor_plan_v1` ·
**Generator:** `2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff` (sha `a596317e4d93`) ·
**Critic:** `V_priv.fold0.pt` · **Simulator:** MuJoCo (`mjeval` / `live_select`)

Companion to `seed_headroom_report.md`, which measures the *oracle ceiling* on seed selection
in Isaac. This measures what a **running selector actually captures** in MuJoCo, and whether
the seed carries quality that transfers across states at all.

## Headline

| question | answer | status |
|---|---|---|
| does a draw's quality transfer across states? | **yes**, σ_seed = 0.107 [0.082, 0.148] | p < 0.0001 at every speed |
| what does the heading controller buy? | **+0.035** [+0.014, +0.060] at K=1 | p = 0.005, 21 tile clusters |
| what does best-of-4 actually capture? | **+0.119** over its same-noise control | 68% of the naive +0.175 |
| simulator noise floor | **σ_run = 0** within a device | not across devices |

The Isaac side's independent σ_seed is **0.130**; this CI contains it. Two harnesses, different
terrain sampling, different episode structure, same effect size.

## 1. The question

Three, sharing one sweep:

1. **Seed transferability.** Isaac's headroom result says choosing the seed per clip is worth
   ~+0.09. That is a ceiling on selection, but it does not say whether a *good draw is good in
   general* or only good against the state it met. The distinction decides whether a selector
   can ever pre-screen a candidate, or must always evaluate it against the current state.
2. **Heading control.** Nothing in this stack steers — the command is a body-frame velocity and
   the generator plans from depth, so heading error accumulates over a course.
3. **Best-of-K validity.** The Isaac per-state best-of-K harness was retired for having no
   resolving power. Does the MuJoCo equivalent survive the same control?

## 2. Isolating the variable

Within one cell the **only** thing differing between the 40 episodes is the flow-matching noise
draw (`policy._rng = default_rng(seed)`). Held byte-identical: spawn pose (standing, 2.5 m
before the feature, laterally centred, pelvis 0.79 m, zero initial velocity), commanded
velocity, terrain, camera lag. No domain randomisation, no depth-sensor noise, no spawn jitter.

> **This is the property everything else rests on.** Because nothing but the draw varies, the
> entire within-cell variance *is* the seed's doing, and no `σ_run` term has to be estimated and
> subtracted. Isaac must do that (its `σ_run` = 0.053, only 58% of clips reproducing exactly);
> here it is measured at zero and the seed component is read undiluted.

## 3. How the data was gathered

Full factorial over 21 real clip-library tiles, 40 episodes per cell, seeds `0…39`
**seed-matched across arms**.

| factor | levels |
|---|---|
| tiles (21) | `stairs_up` 6–11 · `stairs_down` 20–24 · `jump_on` 40–44 · `jump_off_front` 75, 76, 77, 81, 82 |
| speeds (3) | 0.8, 1.6, 2.4 m/s, constant forward, no ramp |
| steering (2) | off; or P on heading error to goal, `kp = 1.5`, clipped ±0.68 rad/s, per control step |
| selector (2) | K = 1 (no choice) and K = 4 (critic argmin over 4 executed candidate chunks) |

252 arm-cells → 126 invocations × 40 episodes = **10,080 episodes**, run in two phases across
talos GPUs 1–2.

**Termination.** Fall = pelvis below the scene's lowest terrain surface + 0.20 m, or roll/pitch
beyond 70° *separately*. Arrival = mjeval's rectangle around `target_xy` plus the traversal
gate. Fall is tested before arrival, so a robot that face-plants across the line scores a fall.
Timeout = the course's own generous `budget_s` (1.4–3.6× the Isaac clip window).

> **Environment defect worth propagating.** talos's shared `env_isaaclab` carries
> onnxruntime-gpu 1.29.0, which links CUDA 13 while the env provides only CUDA 12 libs. The
> CUDA provider fails to load and onnxruntime **silently falls back to CPU** — earlier runs were
> ~5× slower with no error surfaced. Fixed here by an isolated venv carrying onnxruntime 1.22.0.

## 4. Distribution of the data

| outcome | n | share | detail |
|---|---|---|---|
| success | 8,064 | 0.800 | — |
| fall | 1,976 | 0.196 | tilt 1,393 · height 583 |
| timeout | 40 | 0.004 | budget not binding |

Mean 22.9 re-plans per episode. Falls are 70% tilt, 30% height.

| family | 0.8 K=1 / K=4 | 1.6 K=1 / K=4 | 2.4 K=1 / K=4 |
|---|---|---|---|
| ~~jump_off_front~~ | ~~0.980 / 0.965~~ | ~~0.853 / 0.973~~ | ~~0.895 / 0.993~~ |
| jump_on | 0.965 / 1.000 | 0.527 / 0.782 | 0.268 / 0.537 |
| stairs_down | 0.868 / 0.925 | 0.735 / 0.845 | 0.468 / 0.627 |
| stairs_up | 0.917 / 0.973 | 0.850 / 0.912 | 0.619 / 0.692 |

Speed is by far the largest effect: success 0.949 → 0.813 → 0.638 across 0.8 / 1.6 / 2.4.

> **RETRACTED 2026-09-02 — the `jump_off_front` row measured no dismount.** `isaac_tile_course`
> found its far surface among the walkable OVERLAYS only, and a dismount lands on the infinite
> FLOOR, which is not one — so the search returned the block the robot was already standing on.
> On tile 75 that put `target_x` at 2.74 m with `flight_height_m = 0.000`, which switches
> `Course.traversed` off entirely, and `in_goal`'s ±1.0 m band then awarded success at **1.74 m
> — still on the approach platform, before the drop existed.** That is why this family scored
> highest of the four and why its K=1 rate *rose* with speed (0.980 → 0.853 → 0.895): the
> episodes were distance-limited, not difficulty-limited.
>
> Fixed by giving a dismount the rule every other descending course already uses — the goal
> `TARGET_LEAD + DESCENT_RUNOUT` past the far edge of the last block, on the floor. Re-measured
> on tile 75 at 1.6 m/s: episodes go 154–172 → 305–512 steps and 1 of 8 now falls, against 8 of
> 8 succeeding before. **The other three families are unaffected** — each has a real far-platform
> overlay, verified: stairs_up +0.260/+0.850/+1.020, stairs_down −0.777/−0.652, jump_on
> +0.400/+0.750/+0.700 — so every pooled contrast in this report that excludes `jump_off_front`
> still stands, and the pooled ones include 5 void tiles of 21.

`jump_off_front` needed `--klass box` to build at all (absent from `FAMILY_KLASS`); 8 of its 15
tiles build, the rest fail with a spawn outside the tile footprint. Its goal anchors were
recovered from the clip library, not approximated. **That `--klass` override is what walked past
`klass_for_family`'s deliberate refusal of this family, and the refusal was load-bearing** — see
the retraction above.

## 5. Determinism, measured in both directions

| test | config | result | implication |
|---|---|---|---|
| same GPU, 2 processes | tile 6, 1.6 m/s, K=1 | **12/12 identical** | σ_run = 0 |
| GPU 1 vs GPU 2 | same config, same seeds | **11/12 agree** | ~8% episode flip |

> **Cross-device caveat.** TF32 shifts the generator's plan by ~1.37e-2 across GPUs, enough to
> flip a marginal episode. Phase 1 sharded the two steering arms across both GPUs, so its paired
> contrasts carry ~8% flip noise. This **dilutes, never biases** — neither GPU is systematically
> better — so the reported effects are conservative. Phase 2's sharding keeps both arms of a
> `(tile, speed)` on one device.

## 6. The statistic, and why the pairing is stronger here

All estimators are the evaluation harness's own (`scripts/rsl_rl/eval/stats.py`), so these
numbers are method-comparable to the Isaac side rather than a second invention: the **tile is
the cluster**, intervals are a percentile bootstrap resampling tiles (B = 4000), and *p* comes
from a sign-flip permutation over whole tiles.

`paired_delta` documents itself as "deliberately not episode-paired: episode *i* of a clip in
one cell is a different draw from episode *i* in another." That does not hold here — the episode
index **is** the flow-noise seed, the spawn is identical, the sim is deterministic. So episode
*i* of the steer arm and episode *i* of the no-steer arm run the same noise tape from the same
state and diverge only because the command differs, and contrasts are paired on
`(tile, speed, seed)` at the **episode** level.

**The seed-transfer statistic.** Within a cell nothing varies but the draw, so "how big is the
seed effect" is answered by construction. The question that is not: does a draw's quality
transfer to *other states*?

- Build a `seed × cell` matrix using one arm only (no-steer, K=1), so each seed meets each state
  exactly once.
- Statistic: variance of per-seed success rates pooled across cells.
- Null: permute seed labels **within each cell** (B = 4000), holding every cell's success rate
  fixed and destroying only the correspondence between cells.
- **Dispersion index** = observed / null; 1.0 means a draw only interacts with the state it met.
- **σ_seed** = `sqrt(max(0, var_observed − var_null))`, in per-cell success-rate units — the same
  units as the Isaac decomposition's σ_seed. No σ_run term appears, because it is zero.
- Intervals resample **cells**, not seeds: the question is generalisation to other states.

> **Correction — an inflated first version.** The first run pooled all 126 cells and reported a
> dispersion of **4.85**. A seed appears at the same `(tile, speed)` in four cells (steer × K),
> and those four run *one noise tape from one spawn* — near-duplicates, not independent states.
> Permuting labels independently per cell destroys that legitimate correlation in the null while
> keeping it in the observed statistic, manufacturing over-dispersion out of replication alone.
> The reported numbers restrict to one arm per `(tile, speed)`.

## 7. Result

**Seed transfer** (no-steer K=1 cells; index 1.0 = quality is purely state-specific):

| stratum | cells | mean | dispersion [95% CI] | σ_seed [95% CI] | p |
|---|---|---|---|---|---|
| pooled | 63 | 0.730 | 6.72 [4.21, 13.71] | **0.107** [0.082, 0.148] | <0.0001 |
| 0.8 m/s | 21 | 0.923 | 2.49 [1.56, 6.83] | 0.065 [0.038, 0.129] | <0.0001 |
| 1.6 m/s | 21 | 0.730 | 4.44 [2.57, 9.46] | 0.157 [0.104, 0.243] | <0.0001 |
| 2.4 m/s | 21 | 0.538 | 4.10 [2.76, 8.77] | 0.154 [0.119, 0.225] | <0.0001 |

Significant at every speed, every CI excluding 1.0, and monotone in difficulty: the effect is
weakest at nominal speed and ~2.4× larger at 1.6–2.4 m/s.

> **Superseded intermediate result.** At 9 tiles the 0.8 m/s stratum read 1.21 [0.95, 4.20],
> p = 0.12, and was described as a near-null — "at nominal speed a draw's quality is purely
> state-specific." That did **not** survive 21 tiles (2.49, p < 0.0001). The gradient survives;
> the null does not.

**Steering** (paired on `(tile, speed, seed)`, tile-clustered, 21 clusters):

| arm | no-steer | steer | Δ [95% CI] | p |
|---|---|---|---|---|
| K = 1 | 0.730 | 0.765 | **+0.035** [+0.014, +0.060] | 0.005 |
| K = 4 | 0.836 | 0.869 | **+0.033** [+0.012, +0.060] | 0.002 |

At 9 clusters neither reached significance (p = 0.078 / 0.066); at 21 both do with essentially
unchanged point estimates.

> **Retracted — single-tile effects.** Exploratory runs at n = 20 on one tile reported **+0.250**
> on `stairs_up` (tile 6) and **−0.200** on `stairs_down` (tile 20), briefly described as
> family-level effects. They are not: tile 6 is an outlier, and with proper baselines sibling
> tiles 7 and 8 show 0.000 and −0.050 at the same speed. The clustered +0.035 supersedes every
> per-tile figure.

**Best-of-4, against its control.** The control hands every candidate the *same* draw, so the
selector has zero signal by construction. Both arms re-seed per re-plan and both
snapshot/restore; only candidate diversity differs. Mean over 9 cells (tiles 6, 20, 40 × 3
speeds):

| quantity | Δ | reading |
|---|---|---|
| K=4 real − K=1 | +0.175 | naive gain |
| K=4 same-noise − K=1 | +0.056 | machinery + reseeding, no selection signal |
| **K=4 real − K=4 same-noise** | **+0.119** | genuine selection, positive in 8/9 cells |

> **Correction — an invalid first control.** This was first run as same-noise K=4 vs **K=1** and
> appeared to fail badly (25–39 of 40 episodes disagreeing). That comparison is meaningless:
> K=1 seeds the RNG once per episode and lets it advance, while every K>1 path re-seeds per
> re-plan, so the two draw different noise tapes *by construction* and can never match however
> correct the branch machinery is. The valid control is same-noise K=4 vs real K=4.

## 8. Reading this against the Isaac headroom result

| quantity | MuJoCo (this) | Isaac | note |
|---|---|---|---|
| σ_seed | 0.107 [0.082, 0.148] | 0.130 | CI contains it |
| σ_run | 0 (within device) | 0.053 | different regime |
| steering, overall | +0.035 / +0.033 | +0.024 (excl. `flat`) | same magnitude |
| selection worth | +0.119 **captured** | +0.093 **oracle ceiling** (held-out) | see below |

**The two selection numbers are not directly comparable, and the difference is instructive.**
Isaac's +0.093 is an oracle over *whole-episode* seed choice on its clip grid; this +0.119 is
what a running critic actually captured on a different grid — 21 tiles, three commanded speeds
including 2.4 m/s where baseline success falls to 0.538, and a standing-approach spawn rather
than clip-matched starts. A harder grid leaves more room, so the achieved number exceeding the
other harness's ceiling is a statement about different problems, not a contradiction. It should
not be quoted as "the selector beats the oracle."

Isaac's finding that **per-decision selection (+0.067 raw) is far weaker than whole-episode
selection (~+0.09)** is the relevant framing for this selector: it re-plans every 12 control
steps and therefore chooses ~23 times per episode, i.e. the compounding regime rather than the
single-decision one.

> **A stale citation, corrected.** An earlier version of this comparison quoted "87% of the
> Isaac best-of-K gain is artifact." That figure came from the **retired per-state harness**
> (whose same-noise control matched the real arm at ratio 0.80–0.93), not from the headroom
> measurement, and it should not be read as a verdict on the latter. The headroom report's own
> position is that its *shuffle* control over-corrects — recovering 31% of a known synthetic
> effect — and that the unbiased estimate is the held-out +0.093.

## 9. Limitations

- **One initial condition per cell.** Every episode starts from the same standing spawn 2.5 m
  out, so these rates measure sampling reliability from one state, not capability across a
  distribution of starts. This is the main reason absolute levels sit above the Isaac eval's.
- **Phase-1 pairs straddle GPUs** (~8% flip noise), making those contrasts conservative.
- **Per-family splits are underpowered** at 5–6 tiles per family; only pooled contrasts are
  reported as significant.
- **Steering on stairs is off-distribution.** The training library commands `|wz| max = 0.0000`
  on all 3,500 stair clips. `flat` and `box` do turn in training.
- **One generator, one tracker, one critic.** Nothing separates properties of this checkpoint
  pair from properties of the method.
- `jump_off_front` cells score under `box` traversal-gate semantics — an explicit override, not
  a validated mapping. **This limitation turned out to be fatal rather than merely noted: those
  cells measured 1.74 m of platform walking and no dismount at all. See the retraction in §4.**
- **The Isaac headroom statistic cannot be computed on this data**, structurally: it needs
  R > 1 robots per `(clip, seed)` so `s_k` is a rate. Here R = 1 by design, so `s_k ∈ {0,1}`,
  `max_k s_k = 1` whenever any seed succeeds, and the within-clip reshuffle preserves the
  outcome multiset exactly — giving NET ≡ 0 for every cell. Reproducing it would mean adding a
  within-`(cell, seed)` noise source, which gives up the σ_run = 0 property that makes this
  setup's control exact.

## 10. Reproducing it

Raw per-episode records (verdict, failure reason, re-plan count, seed) are in
`live_select.json` under each cell directory.

```
logs/value/sweep_v1/t<tile>_s<speed>_<steer|nosteer>/   # 54 cells, 9 tiles
logs/value/sweep_v2/t<tile>_s<speed>_<steer|nosteer>/   # 72 cells, 12 tiles
logs/value/samenoise_v1/t<tile>_s<speed>/               # 9 control cells
logs/value/sweep_v1/analysis.json                       # serialised estimates
```

```bash
# one cell
python live_select.py --out <dir> --critic_ckpt V_priv.fold0.pt \
  --tracker_onnx tracker.onnx --fm_onnx <gen>_fm.onnx --clip_lib <lib>.h5 \
  --tiles 6 --k 1,4 --episodes 40 --speed 1.6 --device cuda \
  --steer_kp 1.5 --steer_wz_max 0.68 [--klass box] [--same_noise]

# analysis
python analyze_sweep.py --sweep logs/value/sweep_v1 --boot 4000 \
  --json_out logs/value/sweep_v1/analysis.json
```

Analysis code is `transfer/sim/analyze_sweep.py`; the estimators it calls are
`scripts/rsl_rl/eval/stats.py`.
