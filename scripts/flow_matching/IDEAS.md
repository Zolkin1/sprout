<!-- Copyright 2026 Zachary Olkin. All rights reserved. -->

# FM generator: what to try next, and what has already been ruled out

Scratch working doc, not a spec. Kept in the repo so the reasoning and the measured numbers
travel with the code instead of living in a chat log. Dated entries; delete freely once an
idea is settled.

Measure closed-loop survival with `scripts/rsl_rl/fm_survival_sweep.py`, which carries the
named artifacts and the measured tables in its docstring.

---

## The state of the problem (2026-08-03, 200 envs, flat library, K=1)

Survival, XL (27.3M) @ckpt_99. Ceilings below it are what each gap is measured against:

| | survival | gap to the next row |
|---|---|---|
| closed loop (own history) | 135/200 = 67.5% | **18.0 pts** = exposure bias |
| teacher-forced (clip history) | 171/200 = 85.5% | **5.5 pts** = prediction quality |
| clip control (FM off) | 182/200 = 91.0% | 9.0 pts = the teacher's own ceiling |

So the two things worth attacking, in order of size: **autoregressive drift** (18 pts) and
**prediction quality given a correct context** (5.5 pts). Nothing else is in play until
those move.

**Do not compare survival across different `--num_envs`.** A 24-env sweep put the
exposure-bias gap at 3 envs and a 50-env one at 4 pts, where 200 envs says 18. Small samples
drew easier subsets of the 1500-clip library. Every pre-200-env number in git history is
directional only.

---

## Ruled out by measurement

| idea | verdict | evidence |
|---|---|---|
| `noise_scale=2.0` (double the history noise) | **no** | +1-2 envs at 50 envs, and it made the history-free branch *worse* (9/24 -> 4/24). The mismatch it targets is real but this is not the lever. |
| More Euler sampling steps | **no** | 8 -> 32 moved K=1 by ~1 env (42/50 -> 41/50). Integration is already well resolved; the gap is model error, not discretisation. |
| Weight EMA | **no effect on val** | Implemented and on by default. `val_ema/flow_mse` 0.0842 vs `val/flow_mse` 0.0836 on the XL run -- marginally worse, `rmse_base_pos` identical. Closed-loop sweep pending; val has mispredicted closed-loop twice, so this is not yet final. |
| Infer velocity by differencing two predicted poses | **no** | The FD is ~20% worse on linear velocity than simply reading the predicted velocity channel (xl: 0.139 vs 0.112 m/s). Adjacent-node position errors correlate at rho=0.996, but 1-rho ~ 0.004 amplified by 1/dt=50 dominates. The redundancy is exact in the *data* and lossy in the *predictions*. |
| Strided-only FM (predict only the 11 observed nodes) | **dropped** | Zach's call, 2026-08-03. |
| `fm_zero_noise` as a proxy for RTC | **inconclusive, by construction** | 135/200 -> 140/200. Buys determinism but samples off-distribution, so consistency and quality move in opposite directions and confound. Says consistency-bought-crudely is worth 0-5 envs; says nothing about RTC. |

## Tabled

- **Joint-position smoothness loss.** The slim window silently removed *all* joint temporal
  coupling: `joint_vel_consistency` needs `joint_vel` and `fk_pose` needs `ee_pos`, so both
  auto-disable, leaving nothing to constrain `joint_pos` across nodes except the
  transformer's implicit smoothness. A second-difference penalty would need no `joint_vel`.
  Tabled 2026-08-03: joint jerk has not actually shown up as a problem.

---

## Candidates

Two classes, and the distinction drives the plan: an **obs change costs a ~12k-iteration
teacher retrain**, an **FM-only change costs ~1-3 h**. Batch every obs change into ONE
retrain rather than paying for three.

### A. Obs-changing -- batch these together

**ALL THREE LANDED on `worktree-teacher-refine` (2026-08-03), in one obs-width bump.**
`base_pose` is now `[pos(3), rot6d(6)]`, both halves meaning *the reference pose in the
anchor frame*, and the window samples 13 offsets `(0, 1, 2, 5, 10, ... 50)`. D 41 -> 44,
obs 451 -> 572. `BASE_POSE_CONVENTION` bumped, and stale artifacts are refused at load on
BOTH sides -- verified against the real `depth_v5_refslim` tracker export (convention
mismatch) and the 41-node XL checkpoint (`state_dim 44 vs 41`). `F` is unchanged at 51, so
no re-record was needed; the wrap filter and its four tests are deleted. Isaac-vs-numpy obs
parity re-verified at the new width (572, max_err 1.2e-07). Awaiting the ~12k-iteration
teacher retrain.

1. **6D rotation for `base_pose`.** DONE. Eliminates the `quat_box_minus` branch cut at pi
   outright, which would retire `dataset._wrap_keep_mask`, `WRAP_LIMIT_RAD`, and the whole
   class of wrap discontinuity. The helpers already exist and are tested (`matrix_to_rot6d`,
   `rot6d_to_matrix`, `quat_to_rot6d`, `ori6d_orthonormality_penalty`) -- they are the dead
   half of `transforms.py` from the retired relative-state layout. Cost: `base_pose` 6 -> 9,
   D 41 -> 44, obs 451 -> 484.
2. **The `base_pose` sign fix.** DONE. Position is `ref - actual` in the ROBOT frame while
   orientation is `actual - ref` in the REFERENCE frame -- two vectors pointing opposite
   ways in different frames, called one error vector. See `window.BASE_POSE_CONVENTION`.
   Held back only because flipping it is invisible in the obs shape, so a stale policy loads
   clean and steers wrong -- an objection that evaporates when the width is changing anyway.
   Side benefit: afterwards `Exp(ori_err_b) = R_robotT R_ref`, which is what the FD angular
   loss wants.
3. **A short-offset pose sample.** DONE. The window samples at stride 5 (0.1 s), too coarse to
   difference. Adding a node at offset 1 would let the policy see 50 Hz structure. NOTE the
   FD probe above found differencing worse than reading the velocity channel, so do this to
   give the policy *more* information, not to *replace* the velocity blocks.

### B. FM-only

4. **Per-node normalization / loss weighting for `base_pos`.** The best-motivated candidate:
   it has a measured mechanism and a falsifiable prediction. The normalizer is fitted over
   all `(Nw x F)` nodes POOLED, so `base_pos`'s std (0.256 m) is set by the far nodes, while
   node 0's own std is 0.0121 m. Node 0 therefore enters the loss with a normalized std of
   0.047 and carries **0.004%** of the squared-error budget, against 4.6% for node 61 -- a
   factor of ~1000. The models behave exactly as that incentivises:

   | error/signal | n0 | n1 | n5 | n11 | n20 | n40 | n61 |
   |---|---|---|---|---|---|---|---|
   | capacity 9.5M | 0.967 | 0.775 | 0.296 | 0.189 | 0.166 | 0.176 | 0.182 |
   | xl 27.3M | 0.865 | 0.690 | 0.260 | 0.163 | 0.139 | 0.143 | 0.157 |

   Node 0 is barely better than predicting the mean, while every node past 5 is ~6x better
   *relatively*. **Tripling capacity did not change the shape of that curve.** And node 0 is
   the only node that enters the autoregressive history at K=1, and the only one the CLF
   grades. Prediction if fixed: node-0 error/signal falls toward ~0.15.

   Per Zach's cursor observation, the compounding-relevant nodes are `0..K-1` (the served
   ones), not node 0 alone -- at K=12 the history carries nodes 8-11 of the previous solve.
   So weight nodes `0..K-1`, or normalize per node.

5. **RTC (real-time chunking) / correlated sampling noise.** At K=1 a fresh `x0` is drawn
   **50 times a second**, so consecutive plans are independent samples: even a perfect model
   emits a slightly different trajectory each tick and the served reference jitters where a
   clip is smooth. RTC's frozen-prefix inpainting makes successive plans agree by
   construction. Two caveats: the architecture was deliberately moved AWAY from inpainting
   ("no inpainting / known-mask / re-clamp"), so this reintroduces a known-mask; and it is
   the largest change on this list.

   **Cheaper first step: correlate `x0` across re-plans** (reuse or slowly rotate it) instead
   of zeroing it. That keeps every draw on-distribution while removing per-step independence,
   so it isolates consistency from quality -- which `fm_zero_noise` could not. ~5 lines in
   `_run_fm`, no retrain, and plausibly shippable on its own.

6. **Minibatch OT coupling.** Pair `x1` with `x0` by solving a linear assignment on
   `||x1_i - x0_j||^2` within the batch instead of pairing independently. ~30 lines with
   `scipy.optimize.linear_sum_assignment`. The literature's headline benefit is straighter
   paths needing FEWER integration steps -- already moot here, since 8 -> 32 steps changed
   nothing. What remains is gradient-variance reduction: real but second-order. Low cost, so
   worth folding into whatever FM retrain happens next.

7. **Longer training.** XL was still improving at epoch 100 (val 0.0836 below train 0.0818's
   neighbourhood, curve not flat). 300-500 epochs before assuming 1000.

---

## Housekeeping / smaller items

- **Re-measure capacity (9.5M) at 200 envs.** Every capacity-vs-XL comparison on record is
  50-env and not comparable to the 200-env table above.
- **A `deploy_realistic` sweep config.** Every survival number so far comes from the *play*
  task, which disables observation corruption, depth noise, camera jitter and pushes. Those
  are the honest A/B conditions but an optimistic deployment estimate.
- **Re-record**, which unlocks `--skip_families family_jump_off` (the BACKWARDS
  jump-off-a-box family, unobservable to forward-facing cameras -- the tag is implemented but
  the existing 9.8 GB recording predates it). Same recording should regenerate the stale
  world-pinned distractor family.
- **Should the TEACHER also stop seeing the unobservable backwards jump-offs?** The
  unobservability argument does not stop at the generator. Bigger call: 12k iterations.
- **`TODO(deploy-k>1)`.** K=12 is 7.5 pts better than K=1 but the torch-free runtime refuses
  K != 1 (it needs a base-pose estimate to re-anchor). The window-to-world port added on the
  sim side may be most of what this needs.
- **Dead code**: `transforms.py`'s state-layout half, all of `transforms_np.py`.
