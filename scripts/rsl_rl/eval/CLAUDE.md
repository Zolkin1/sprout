# The evaluation harness

`evaluate.py --stage plan|cell|grid` then `report.py <run> [<run> ...]`. Stages share one manifest so
nothing is retyped; `plan` and `report` import no IsaacLab, and `report --stage reduce` imports no
matplotlib. A cell is `(reference_mode, dr_preset, fm_label)`; the generator is a cell DIMENSION, so
several generators share ONE clip arm and every cell is paired on clip id. See README "4b. Evaluate"
for the runbook.

`_paths.py` is the `sys.path` bootstrap that keeps `provenance`, `cli_args`, `play_plots` and
`plot_style` reachable by bare name from this subfolder. `scripts/rsl_rl/plot_style.py` lives in the
PARENT because four plotters outside the eval use it (and `import robot_rl.anything` drags in
IsaacLab); importing it does not touch `rcParams`, only `apply_paper_style()` does.

## Load-bearing, each found by a failure

- **The tracker task has no `clip_end` termination**, so the harness ADDS one. Without it every
  episode runs the padded 20 s: a 300-frame clip becomes 6 s of motion plus 14 s of a frozen pose,
  and in the FM arm 14 s of being commanded the last frame's velocity off a finite tile.
- **`terrain_columns` returns only the DERIVED terrain flags.** The caller owes
  `terrain_start_step` / `terrain_end_step`; trusting it for the whole block zero-filled them, which
  left the derived values correct but made the report's degeneracy gate read every family as
  informative. Pinned by `test_terrain_columns_does_not_supply_the_primitives`.
- **Terrain completion is degenerate on 4 of 9 families** (`flat` has no pinned frame;
  `distractor` / `stairs_*_pure` are pinned to their last frame). The gate is
  `terrain_end_step < last_usable_step`, per clip -- NOT `has_terrain_span`, which is true for 8 of 9.
  `terrain_progress` is the metric that still says something, and its denominator is the REACHABLE
  span (`min(terrain_end, last_usable) - terrain_start`) so a success scores exactly 1.0 on every
  family; the full span would score 0.83-0.91 on the terminal families and smear the success
  point-mass.
- **`*/none` is a NOISE FLOOR, not a determinism assertion.** With all DR and obs noise off and the
  clip pinned, two envs on the same clip and tile still diverge (measured ~0.3 within-clip relative
  spread of `steps_survived`). One env per tile gives the same spread, so it is not tile sharing --
  it is simulator-level. The DR-uniformity check ("every env is the same robot") does hold and is
  asserted. Note `dr_preset=none` also sets `enable_corruption=False`, so a headline gap measured
  there has the tracker's reference-window noise OFF.
- **Only 5 of 8 physical DR terms can re-draw per reset.** `add_joint_default_pos` adds to the live
  `default_joint_pos` buffer (random walk), and `randomize_rigid_body_com` hands the FULL CoM array
  to `set_coms_index` alongside a SUBSET of env ids, which raises on a partial reset.
  `RESET_SAFE_PHYSICAL_TERMS` encodes the split and the manifest records it per term, because the
  three that stay `startup` leave their draw confounded with whichever clips an env runs.
- **`_widen` cannot be used for the OOD preset.** No clamping (it turns `static_friction (0.3, 1.6)`
  into `(-0.025, 1.925)`), and its `len(value) == 2` test silently passes the camera extrinsics
  3-tuples through UNTOUCHED, which would make `ood` identical to `train` on that whole axis. The
  safeguard is not a floor but an assertion that a widened range is never narrower -- a floor of 0.05
  makes `viscous_friction`'s OOD low end higher than training's.
- **Envs may share a tile.** `filter_collisions` is on and the depth raycasts are per-env, so a dozen
  robots in one tile neither collide nor see each other -- confirmed empirically, not just from the
  config default. That is what lets `num_envs` be smaller than the column count.
- **Statistics are clip-averaged with a TILE-clustered bootstrap.** A binomial interval over pooled
  episodes has an actual coverage of 14-70% against a nominal 95% here, and gets worse with more
  draws per clip. Spend budget on tiles, not on `robots_per_clip`: 2x the robots measured as buying
  nothing, while 2x the tiles at half the robots is free and 21% tighter.
- **A flag added to the parser is NOT forwarded to the cell subprocess unless `stage_grid` restates
  it.** It reverts to its argparse default in the child, silently. Measured twice: `--num_envs` ran a
  grid at 256 while the manifest recorded 1024, and `--terrain_ref` produced 3950 episodes with
  `n_plans == 0` and no error. This is the whole reason `fm_tf` is a reference MODE and not a flag --
  a mode rides in `plan.json` and cannot be dropped. Pinned by
  `test_stage_grid_forwards_every_flag_the_cell_stage_reads` (a heuristic: a flag read via `getattr`
  is invisible to it).
- **The derived fall flag once inherited the PREVIOUS episode's orientation.** `_reset_idx` refreshes
  `root_pos_w`/`root_quat_w` but `projected_gravity_b` lagged, so the first fold of a new episode
  read the old tilt while reading the new height -- inflating `fell` by a fixed ~18.5% of the
  PRECEDING fall rate. Per-step folds now use `root_quat_w`; `fold_terminal` takes
  `max(quat, gravity)` because inside the reset hook the LAGGING buffer is the correct one. Success
  rates were never affected -- they come from termination terms, never the accumulator.

## What the harness has established about the depth generator

Tracker `2026-08-20_09-04-12_mode11_decor_v1`, generator `2026-08-22_12-26-34_mode11_decor_lag48_aug`.
Clip-averaged, tile-clustered.

- **The generator, not the tracker, is the limit**: fm/train 0.468 against clip/train 0.930, paired
  delta -0.462 [-0.513, -0.411], p<0.001 over 264 clips.
- **Exposure bias is largely RULED OUT.** Teacher forcing (`fm_tf`) buys +0.078 overall, but removing
  `jump_off` drops that to +0.026 and removing both jump families to +0.006, with 4 of 7 remaining
  families NEGATIVE. `jump_off` jumps BACKWARDS, which no forward-facing camera can see, so teacher
  forcing there smuggles in an unobservable -- that family is dropped from the eval AND from FM
  training.
- **DR is irrelevant**: fm/none 0.495 -> fm/train 0.468, inside the interval. The tracker is already
  near-indifferent to i.i.d. jitter, so adding MORE of it is not the lever.
- **The FM error is structured, and differently per family.** Fraction of window-error energy in
  {const, const+ramp} over the 13 observed offsets (i.i.d. baseline 0.077 / 0.154): `base_pos`
  0.49/0.87 on the transition families vs 0.93/0.99 on the `*_pure` ones; `ang_vel` is 0.04/0.11,
  i.e. SUB-white, nothing to model there. Growth from node 0 to node 50 is 5.1x, so the band
  over-covers the near node and under-covers the far one (the rigid component is under-covered
  3.2-4.5x). Any perturbation built for the tracker should ramp with look-ahead and be drawn per
  re-plan, not per step.
- **`stairs_dn_pure` is the outlier** (0.048 closed loop, 0.073 teacher-forced, 0.087 after the
  lag-ring fix, against a 0.939 clip ceiling) but is NOT a unique defect -- normalised per tread all
  four stair families share an ~8-14%/tread hazard, so 0.048 is 25.4 treads of exposure.
  Deprioritised in favour of `stairs_up`, `stairs_down`, `jump_on`, `jump_off_front`.
- **~48% of FM-attributable falls on the transition families happen on FLAT ground**, before the
  terrain is reached.
- The augmentation was VERIFIED SOUND, not a suspect: the delta-0 self-check's scan residual is
  exactly 0.0, and displaced poses were achieved to 0.35 mm / 0.24 deg median. The alarming
  `frac_gt_1mm = 0.515` is the depth DR being re-drawn.

# The noise draw is not a neutral sample

**`x0 = 0` IS NOT A NEUTRAL SAMPLE.** It sits near the conditional's mode, and both mjeval and the
seed sweeps default to it because that is what the hardware controller runs -- so the trap is on by
default. It overstated one arm gap by ~8x (a -0.889 delta became -0.111 pooled over four Gaussian
seeds, with the SIGN FLIPPING between seeds). **Any single-draw A/B on this stack is measuring the
draw as much as the arms**: pass `--gaussian_noise --fm_seed N` across several `--out` directories
for anything that has to hold for the model.

- **The noise stream must be re-keyed per episode.** `GenTrackerPolicy` builds one RNG at
  construction and draws once per re-plan, so an episode's noise depended on how many re-plans every
  episode before it made -- and an arm that falls early makes fewer, so two arms would have been
  scored on different draws from the first differing cell onward, drifting further apart the longer
  the sweep ran, with clean-looking tables throughout. `GenTrackerPolicy.reseed` plus a per-cell key
  `(fm_seed, terrain_idx, speed_idx)` fixes it; `reseed` is deliberately NOT part of `reset()`, since
  whether an episode resumes or restarts a stream is the harness's experiment design.
- Two related recording bugs, all of the same shape: `runner.py` wrote `ZERO_NOISE_SEED` into every
  row's `seed` column unconditionally (now `ArmRun.recorded_seed`), and the table's and manifest's
  `zero_noise` were the literal `True` (now read off the loaded policy).
- **The generator's OWN sampling variance decides ~1 in 5 outcomes.** From an identical state on a
  real tile with only the flow noise differing: 0.80 success at 0.8 m/s, 0.62 at 1.6. See
  `transfer/sim/CLAUDE.md` for the instrument.

## Measuring what SAMPLING is worth (`seed_headroom.py`)

`evaluate.py --fm_seed K` names the generator's initial-noise draw so it can be varied with
everything else pinned -- `--seed` cannot, because it also moves the DR draws, the spawn jitter and
the observation noise. The draw is keyed `(fm_seed, within-episode re-plan index)` and **deliberately
not by env**: several robots on one clip under one seed must be served the IDENTICAL noise, because
their spread is the simulator floor, and keying by env puts the effect being measured inside its own
control. `docs/seed_headroom_report.md` is the write-up.

- **`max over seeds` is biased upward, and the obvious fix is biased DOWNWARD by more.** A
  permutation control (repartition a clip's episodes into pseudo-seeds) is a valid TEST -- it returns
  zero on a synthetic null -- but not a valid ESTIMATOR: it measures the inflation under the null of
  NO effect, and where an effect is real the max is partly driven by genuinely better seeds and is
  less noise-inflated than the null predicts, so subtracting the full null bias over-corrects
  (measured against synthetic ground truth it recovered 31%). What works is HELD-OUT selection --
  pick the best seed on some robots, score it on the rest -- plus a sweep of the selection sample:
  still climbing means the ceiling is above the estimate. **Validate any estimator here against
  synthetic data with a known answer before quoting it**; all three were, and two failed.
- **Isaac is NOT reproducible run-to-run here.** The same `fm_seed` in two processes reproduced
  per-clip rates exactly in 58% of clips. The docs' determinism claim covers re-running a simulation;
  it did not hold. A same-seed repeat is a REQUIRED arm, not a nicety.
- **Do not use `p(1-p)/R` as the within-clip floor.** Robots sharing a draw are strongly correlated
  -- measured same-seed variance ~8x smaller than independence predicts.
- `--num_envs` above what a plan can populate is refused with `pinned to clip -1`, not run; 1000 envs
  OOMs a 12 GB card where 750 does not.

# The best-of-K selector harness (`headroom*.py`)

`headroom.py --stage select` runs a V(s) best-of-K selector to the END of each episode: at every
re-plan it copies one state into `K` envs, draws a different chunk in each, rolls one chunk, scores
the resulting states with a `fit_value.py` critic and carries the winner forward.
`--stage restore_probe` is the state-copy gate; `headroom_select_report.py` pairs arms on clip id.

- **The arms only mean something against the SAME-NOISE control.** `--same_noise` serves every
  candidate one draw, so every candidate is the same plan and any best-of-K rise is manufactured by
  the simulator. `--no_selection` builds and restores all K envs but always follows candidate 0, so
  it must reproduce the `k=0` baseline; a gap there belongs to the K-env structure rather than to
  selection. Both are cheap and both have caught a defect that produced clean, plausible, entirely
  wrong numbers.
- **The earlier Isaac per-state harness was ABANDONED and its conclusion is WRONG.** Its same-noise
  control produced a best-of-K rise of +0.005 to +0.031 per family, matching or exceeding the real
  arm's at every horizon from 0.24 to 1.20 s (ratio 0.80-0.93): independent per-candidate noise
  MANUFACTURES a best-of-K rise of about its own flip probability, and two estimates of that agreed
  (0.0157 vs 0.0151). So it had no resolving power and its "the gap is bias-limited" reading is not
  supported. A second defect compounded it: the first action after a restore was computed from the
  PRE-restore observation.
- **The state copy is bit-identical only after four fixes.** IsaacLab's `write_*_to_sim` invalidates
  seven `ArticulationData` caches by hand and `_projected_gravity_b` is NOT among them, so a restored
  env served its own orientation from a current-timestamped cache. `joint_acc` is finite-differenced,
  so `_previous_joint_vel` has to travel too. `RayCasterCameraData.output` is a DICT, so a walk that
  only accepted bare tensors copied each camera's pose and intrinsics while leaving the IMAGE behind.
  And the roll loop reused the pre-restore observation for the first action of every window.
  `restore_probe` now checks every buffer AND every observation term for EXACT equality and exits
  non-zero otherwise. `_body_com_acc_w` / `_body_incoming_joint_wrench_b` are excluded by name: they
  rebind `.data` on every read from PhysX solver output, so a copy cannot stick, and nothing in
  `robot_rl` reads either.
- **Isaac reproduces exactly across envs given identical state; the odd one out is an env that was
  never written to.** Two branches restored from one state are bit-identical to each other, while a
  branch is never bit-identical to its un-restored parent -- and self-restoring the parent does not
  close it. So a control comparing RESTORED against UNRESTORED asks a different question, which is
  why the selector carries the group's state in the winning env rather than in a parent.
  `enable_enhanced_determinism` and higher solver iteration counts were both measured and neither
  fixes the residual; the amplification is the robot, not the solver.
- **Success is `rollout_outcomes.terminal_reward`, and it is NOT `report.py`'s rule.** The two
  disagree on episodes that arrive and then fall (success here, failure there) and on ones that fall
  and are later dragged over the line (the reverse). To compare this harness against the episode
  harness, score the EVAL's own table under `terminal_reward`; done that way, the do-nothing control
  (`--no_restore --no_force_replan`) and the episode harness agree to within one standard error.
  `--goal_regions` is REQUIRED for the same reason `record_rollouts.py` requires it.
- **Two ordering traps, both of which read as a model result.** The goal anchor must be resolved
  INSIDE the reset, after the clip commits -- resolved when the allocator hands out the next clip,
  every episode is scored against its predecessor's goal. And it must be seeded for every group's
  FIRST episode, which no advance ever covers; without that those episodes are scored against a goal
  at the world origin with zero tolerance and can never succeed.
- **Per-decision ceiling**: one oracle decision is worth +0.0099 of success, and ~85% of a naive
  best-of-K number is luck. Read that before restarting any selection work.

# The failure critic (`fit_value.py`)

> **SUPERSEDED 2026-09-10 — the `P(fail)` head this describes is DELETED.** The critic now has
> ONE head, a scalar regression of the return (`FlowValueModel.quality`); the survival head and
> its own readout token are gone, because the fine-tune loop dropped its loss and no reported
> result used it. **Every `p_fail` number below predates that cut and does not describe the
> current code** -- and neither do the quality-head numbers, because removing the survival
> readout changed the key/value stream the surviving head attends to, making it a different
> function rather than the same one minus a head. Kept as the record of what was measured.

Trains a `P(fail)` head over the generator's frozen encoder. Three of its context features are
goal-region quantities needing a live `MotionSequenceCommand`, which is why it could only be read in
Isaac until `transfer/sim/mjeval/goal_anchor.py` + `critic.py` closed it.

- **Two arms of the sabotage suite are BLIND, and they are findings about the CRITIC**: a transposed
  scan fold moves `p_fail` 0.0049 against 0.0704 for zeroing the scan, and a history rolled one row
  0.0024 against 0.0577 for zeroing it. So it reads the scan's CONTENT but barely its spatial LAYOUT,
  and the history's content but barely its temporal ORDER. Neither is catchable by a delta test;
  `SCAN_HW` is `(11, 16)` and deliberately not square so the wrong fold raises on shape instead.
- Reference numbers: AUC 0.906 on held-out Isaac tiles; on measured-history data 0.904 unseen /
  0.928 seen, with the quality head at rho 0.752 against a label-shuffle control of 0.003 and
  `pred_sd 0.190` vs `target_sd 0.253` (under-confident, not collapsed to the mean).
- `fit_value` splits BY EPISODE precisely so near-duplicate windows cannot straddle the split, so
  `merge_rollouts.py` renumbers `episode_uid` past the running max when joining per-speed stems (the
  uid is assigned per RUN from 0, and a collision merges two unrelated episodes into one group and
  deals their windows to both sides).

# Terrain misidentification: real, and CLOSED-LOOP ONLY (`terrain_class.py`)

Field observation: the robot sometimes reads a staircase as a box and plans to jump onto it.
`terrain_class.py` turns that into a number -- k-NN over a reference set of real clip windows, in the
generator's own output space, classifying each re-plan flat/jump/stairs. Matched to
`stairs_up`+`stairs_down`, matched classifier:

| | stairs->jump |
|---|---|
| open loop, true clip window (the control) | 0.103 [0.089, 0.121] |
| open loop, FM prediction | 0.104 [0.084, 0.131] |
| **closed loop, FM prediction** | **0.174 [0.146, 0.205]** |

So given in-distribution conditioning the generator reads stairs exactly as well as the ground truth
does; drive it with its own drifted history and depth from a drifted pose and the confusion appears.
**The fix is on the loop/conditioning side, not "the model cannot see stairs".** NEXT: dump the depth
image alongside the flagged plans to separate a perception failure (the image genuinely looks
box-like) from a generation failure -- the one split this number cannot make.

**The classifier is a POPULATION statistic and NOT a per-plan label.** It was only ever validated
against real CLIP windows (control accuracy 0.859) and is then applied to GENERATOR output, which
lives elsewhere in that space. Measured against |served - clip| as a label-free ground truth: within
matched `clip_step` bands a flagged plan is no more divergent from the clip than an unflagged one
(~1.0x; 0.62/1.36/1.05/1.07 over four bands), separating only at the extreme tail (2-4x). So the
rate above is usable -- the population really has shifted -- but "this plan is a misread" is not a
claim the label supports, and selecting plans to eyeball by it gives a sample that says nothing.
`--dump_flagged` therefore selects on |served - clip| and keeps BOTH ends within a step band, since
prediction difficulty rises through an episode (rank correlation 0.29). That metric has the
mirror-image limit: a plan 5 m from the clip is a robot already lost, so its extreme end selects
consequences rather than causes. Neither proxy isolates the cause alone.

Build the reference with `terrain_class.py --windows <pred_tgt.npz> --holdout 0 --out ...`; pass it
to the eval with `--terrain_ref`. **Always read the rate against the control**, never as an absolute:
the classifier's own stairs->jump confusion is ~0.10 on these families and was unstable (0.010 to
0.116) until the reference reached ~12k windows.
