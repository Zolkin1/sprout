<!-- Copyright 2026 Zachary Olkin. All rights reserved. -->

# The 2026 journal paper: where every figure and table came from

*Multiskill Perceptive Locomotion*, submitted 2026-09-23 (journal) and 2026-09-25 (arXiv).
This file is the index for answering reviewers: for each figure and table, it says which data
produced it, where that data is kept, and which artifacts were under test. The detail lives in
the data bundles themselves, each of which carries its own README / PROVENANCE / REPRODUCE, so
this file points at them rather than restating them.

## Where things are

| what | where |
|---|---|
| the code as submitted | git tag **`2026_journal_submission`** (`85b26243`). Branch revisions from it |
| paper PDF, LaTeX source, supplementary video | `~/AmberLab/Project-Isaac-RL/summer_2026/submissions/{TRL_submission_9_23_26,arxiv_9_25_26}/` (the source is `multiskill_perceptive_locomotion.zip`) |
| figure files as placed in the paper | `summer_2026/figures/`. The data figures are **Inkscape-edited** PDFs made from the renders below, so a re-render will not be byte-identical to them |
| **data bundles** | `summer_2026/paper_data/`: dated, self-contained, never updated in place |
| raw renders + a few figure inputs | `sprout/logs/paper/` (untracked, local only) |
| frozen ONNX artifacts for the sweeps | `sprout/logs/vel_tracking/_artifacts_final/`. **Always use these, never `logs/rsl_rl/g1/<run>/exported/`**, which every `play.py` overwrites |

`summer_2026/` and `logs/` are on this desktop only. Nothing in them is in git.

## The artifacts under test

One tracker throughout: `2026-09-11_11-45-14_paper_v6g_v1` at `model_19999.pt`, near-camera
only (sha256 `5a9bb84a418e` as ONNX). It was identified by comparing weights, not by filename
(see the root `CLAUDE.md`).

| generator | role |
|---|---|
| `2026-09-13_10-08-38` (`parent_lag.pt`) | the pre-trained parent: 27M, both cameras, depth-lag trained |
| `ft_{boxes,multi,updown}_lag` iter 5 | AWR fine-tunes of that parent, one per course. `ft_multi_lag_iter005` (sha256 `d1c11c23987e`) is also what is deployed |
| `v6g_lag_nearonly` (sha256 `94c146944174`) | the lower-camera-only generator, for the camera ablation |

## Figures and tables

| paper item | data bundle | inputs, hash-checked 2026-09-25 |
|---|---|---|
| **Fig. `fine_tuning_gains`**: (a) success, (b) contact penalty, (c) mode choice | `paper_data/finetune_terrains/2026-09-15/` | (a, b) `eval_numbers/fig_numbers_lag.json`, identical to `logs/paper/fig_numbers_lag.json`. (c) `mode_probes/*/report.json`, the human blind labels. The bundle's README has the rebuild command (`plot_finetune_bars.py`). ⚠ The render the paper used is `logs/paper/finetune_lag_3panel.*` (2026-09-17, Times font), which is **not in any bundle** |
| **Fig. `forest_no_pert`**: algorithm comparison vs AWR | `paper_data/finetune_terrains/2026-09-22/` | `figures/forest.png` is byte-identical to `logs/paper/forest_np.png`. Its README has the rebuild commands, and the intervals are explained in `2026-09-16/README.md` |
| **Fig. `vel_tracking_all_terrain_v2`**: velocity tracking across terrain | `paper_data/velocity_tracking_sweep/2026-09-17/` | `run_summaries/final_ftmulti_4terrain.summary.json`, identical to `logs/vel_tracking/final_ftmulti_4terrain/summary.json`. Generator `ft_multi_lag_iter005`, 50 trials × 3 speeds (0.8/1.6/2.2 m/s) × 4 terrains. See `REPRODUCE.md`, plus the layout flags in `PROVENANCE.md` |
| **Table `tab:vel_tracking_clips`**: ours vs Motion Bricks velocity RMSE | `paper_data/motion_clips/mb_baseline_v6g_2026-09-14_22-28-02/` | `results.txt`, the "ALL FRAMES" block, OURS and "MB alone" rows. The motion-matching compute-time claim (7x) is in `compute_times.txt`, the jitter claim in the JITTER block. The library is `g1_mb_v6g_decor.h5` (`clip_fingerprint 9f5902752e96`) |
| **Table `tab:camera_ablation`**: both cameras vs lower only | ⚠ **no bundle** | `logs/vel_tracking/camab16_{both,near}/summary.json`, from 2026-09-22 at `31bb0f28` plus the diff in `git/paper-final.diff`. Recomputed 2026-09-25, it matches the table exactly (box 150/150 vs 86/150). "Both" is the pre-trained PARENT `2026-09-13_10-08-38`, not the fine-tune; "lower only" is `v6g_lag_nearonly`. 50 trials × 3 speeds per terrain |
| Figs. `hardware_hero`, `hardware_lower` | video stills, in `summer_2026/figures/raw_stills` | Which generator was on the robot for each clip is **not recorded**. The candidates are the last two rows of the deployment table in the root `CLAUDE.md`. The only notes are the provisional hardware log in `docs/scratchpad.md` |
| Figs. `architecture`, `transformer_architectures` | drawn by hand, sources in `summer_2026/figures/` | none |

## Known gaps

- **The camera-ablation runs live only in `logs/vel_tracking/`**, with no bundle. They are
  meant to be re-created, not archived: each run's `git/paper-final.diff` records the commit
  (`31bb0f28`) and the uncommitted diff it ran from, and `summary.json`'s `meta` records the
  artifacts, speeds and trial count. The generators are the two named above.
- **The final fine-tuning render is not in its bundle.** The data is (the hash matches), so
  a rebuild from the tag reproduces the numbers. It may differ from the submitted PDF in
  cosmetic labels, because the tick and heading text was edited up to submission.
- **The hardware clips have no per-clip record of which pair was running.**
- **Not merged: the PPO arm of the fine-tune ladder.** It is on `worktree-fm-adjuster`
  (`b230b834`), not on main and not in any paper figure. If a reviewer asks for an on-policy
  baseline, start there.

## Statistics, briefly

Success and contact intervals are 95% cluster-t over one paired value per (tile, speed) cell;
the cell is the unit of independence. Never resample cells and then episodes within them:
that inflated every interval by 14-29% (see `finetune_terrains/2026-09-16/README.md`).
Success is scored over each environment's FIRST episode. The mode panel uses Wilson intervals
over blind human labels, since those trials share no tile to cluster on.
