# FINE-TUNING THE GENERATOR ON TERRAIN

Take a trained generator and improve it on ONE hard course by re-weighting its own behaviour, with no
simulator in the training loop: collect rollouts, grade them, fit a value function, weight or filter
the chunks, update the generator, x5. `run_finetune_loop.sh` runs the whole thing and is resumable
per stage; **`FINETUNE.md` is the runbook** and carries the artifact provenance, the library
fingerprints and the traps.

This is NOT the teacher PPO fine-tune `README.md` §4c documents. Nothing here calls
`scripts/rsl_rl/train.py`.

# BUILDING a generator from scratch: `run_fm_pipeline.sh`

The other direction, and the one to reach for after a teacher retrain: `run_fm_pipeline.sh
<tracker_run_name> [<clip_library>]` runs record -> augment -> train in one go. It reproduces how the
SHIPPED generator (`2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff`) was actually built, read off its
own `params/data.yaml` rather than from memory.

- **Training takes BOTH recordings, comma-separated** (`--data <recorded>,<augmented>`), not the
  augmented one alone. That is the step easiest to get wrong and nothing downstream would say so.
- **`NUM_ENVS=1024`, and the number is load-bearing.** Windows snap every 12 control steps
  (`fm_snap_every`), so a full clip yields ~77 and an env only reaches a clip's LATE phase if
  `num_windows / num_envs` comfortably exceeds that. Measured on the shipped recording: 200k windows
  at 256 envs is 781/env, ~10 episodes, covering **6065 distinct clips** with `clip_step` spanning
  12..924. 1024 still gives 195/env and holds coverage; **4096 would give 49/env -- less than ONE
  episode**, so the run ends mid-first-episode for nearly every env and every window comes from the
  early phase of its clip. That bias is invisible in the output.
- **It asserts the recording carries TWO cameras.** The generator reads both; the policy reads the
  near one alone since `efe93b4e`. `record_data` installs the `fm_depth` group unconditionally on a
  depth task for exactly this reason, but a one-camera recording would train a half-blind generator
  that the checkpoint, the `fm.onnx` and every eval then agree about, so the check is worth its line.
- **The exteroception constants are DERIVED, and nothing should declare them.**
  `extero_xyz_offset` resolves from the recording's own camera contract; on the as-built cameras it
  comes out `[[0.51, 0.03, -0.79], [4.25, 0.06, -0.79]]`. Passing the older
  `[[0.55, 0, -0.79], [1.8, 0, -0.79]]` would be the 35-degree far camera, 2.45 m out -- which is
  what `_XYZ_OFFSET_DRIFT_WARN_M` shouts about.

**WHAT IT BUYS, on the two-box course, paired at the ENV level** (both arms draw the same reset
stream, so every env's first episode opens at a bit-identical downrange -- max |diff| 0.00e+00, which
is why the paired interval is the honest one and the level intervals are conservative):

| arm | paired delta vs parent | McNemar |
|---|---|---|
| AWR on the progress reward | +0.3510 [+0.2514, +0.4476] | p=1.8e-226 |
| **+ the velocity-mismatch knob (0.30)** | **+0.4817 [+0.3905, +0.5831]** | p underflows |
| filtered BC, top 30% | +0.3531 [+0.2124, +0.5124] | p=9.2e-242 |

**Filtered BC MATCHES AWR here** -- the intervals overlap almost entirely and fBC's is 55% wider. On
this course the advantage WEIGHTING buys nothing over a top-30% filter, which is the ablation's real
finding. Up-into-down is +0.2069 and the in-distribution control (routes over the original clip
tiles) is +0.1046, so the OOD fine-tune did not cost in-distribution skill.

**Every number above was measured under the PRE-TERMINATION rule** -- `reached_goal` on recordings
that ran past a tracking violation, because that is what existed when they were collected. To
reproduce them from those recordings you must say so: `--metric reached_goal --failure_rule loose`.
Under the current default (`strict`) the same recordings report 0.0052, because the bound is then
applied to episodes it never ended; `box2_ic_report` detects that and says which rule to use, and
`rollout_outcomes.assert_tracking_bounds_agree` refuses a comparison that straddles the change.
Numbers from runs collected AFTER the termination are not comparable to these on the critic's
definition of an episode.

Everything above is regenerable from the repo: `fig_reduce.py` -> `plot_finetune_bars.py` for the
figure, `paired_delta.py` for the paired test, `variance_audit.py` for what is inside the episode
count. They route their episode selection through `awr_buffer.first_per_env` and their success rule
through `rollout_outcomes.terminal_reward`, so the figure, the report and the paired test cannot
describe one recording differently.

## The four algorithms are ONE selectivity ladder, and only the update differs

`ALGO=awr|fbc|arrived|bc`. Collect, grade and buffer churn never branch on it, so a difference
between arms is the objective and not the data -- pinned by parsing the script
(`test_finetune_loop_algos.py`), because that property lives in shell where nothing type-checks it.

| `ALGO` | update | critic | selects |
|---|---|---|---|
| `awr` | `w = clip(exp(A / (beta * sd A)), 0, 20)` | yes | ~10-15% effective |
| `fbc` | filtered BC on the top 30% by advantage | yes | 30% |
| `arrived` | BC on episodes that ARRIVED before failing | **no** | ~8.8% |
| `bc` | plain BC on every row | **no** | 100% |

**AWR's hot limit is unweighted BC on the labelled rows and its sharp limit under `global`
normalization IS filtered BC**, so the ladder only means something if each arm is actually its own
algorithm: `bc` passes no `--labels` at all rather than a large beta, and `fbc`/`arrived` filter
rather than weight.

**`bc` also needs `--unconditioned`, and that is not obvious.** `is_plain_arm` is true for
`filter_label`, `unconditioned` or `awr_beta > 0` -- plain BC passes NONE of those, which is the
whole point of the arm, so without the flag it takes the advantage-CONDITIONED path, grows an
embedding no label ever reaches, and writes a checkpoint with no serve arm that every consumer then
refuses. That is not hypothetical: `eval_bc_s3_i1` is three log files and no data. BC was the FOURTH
plain arm and `is_plain_arm`'s docstring had already recorded the rule going stale once.

## The progress reward, and why the terrain-only one is maximised by standing still

`R_t = R_terrain + gate * exp(-|v_err| / sigma) + gate_worst * C * reached_goal * is_last`, `C = 10`.
The terrain-consistency grade alone rewards not touching the terrain, and it showed: a terrain-only
run drove forward progress 11.50 -> 3.95 m per 1000 steps WHILE survival rose. So the velocity term
and an ARRIVAL bonus are what make it un-gameable by waiting.

**The success bonus is TERMINAL and gated on the episode's WORST chunk.** Two other shapes were
measured and rejected: adding `C` to every chunk made the target near-binary (unsuccessful rows 0.11,
successful 0.95, with terrain and velocity squeezed into the bottom sixth of the range), and
distributing `C/n` diluted it to 2% of the reward. A terminal bonus keeps it where it is earned and
lets the CRITIC propagate it backwards, which is what bootstrapping is for.

**The gate means arriving destructively pays nothing** -- of 352 successful two-box episodes, every
one reached the goal by dismounting a box, because a correct two-jump traverse is rare enough that
the behaviour policy never produced one and AWR can only reweight what it is shown. That is also why
the course declares box-top SPAWN AREAS.

Related, from the terrain-consistency grade itself: **25% of plans put a knee or foot >=3 cm into the
terrain -- use the TAIL, the mean hid it**, and AUC against `fell` is ~0.5 under every control.

## Isaac TERMINATES on the strict tracking bounds, and the bound is 0.12 m

`terminations.tracking_violation` (`mdp.fm_tracking_violation_termination`) ends the episode at the
union of lateral 0.20 / height 0.12 / tilt 30 deg around the SERVED node. It goes through
`tracking_error.violation_checks` rather than restating the tests, so the termination and every
offline reader of `first_violation_step` cannot disagree about what a violation is.

**It replaced an offline LATCH, and the latch was quietly destroying the success signal.** The
recording used to run on past a violation and every consumer cut the episode there afterwards -- so a
run reached the goal, banked the terminal bonus, and the offline cut then deleted the row it was paid
on. Measured: the last re-plan survived on **4.7%** of arriving episodes, so of 728 arrivals ~34 kept
their bonus in the return. The critic was fitted on a reward with essentially no success term in it,
and every log read healthy. Terminating means nothing downstream truncates anything, and
`reached_goal` and `rollout_outcomes.terminal_reward` become the same question.

**0.12 is measured, not fitted.** Watching the course, proper box jump-downs peak at 0.130-0.182 m
against the old 0.20 bound -- nowhere near it -- while stairs-down-off-a-box dismounts approach it,
so the bound separates a correct dismount from the wrong MODE rather than catching a fall. Its cost
was measured on the real opening data set BEFORE the change (`awr_vm30/inc_001`, 2000 parent
episodes, DR on): clean arrivals 21.70% -> 16.85% of all episodes, windows surviving 95.7% -> 93.4%.
Ground-start clean arrivals collapse 4.18% -> 0.46%, so at this bound every clean success in an
opening set is a box-top dismount -- a scarcity that predates the change. The tilt bound never binds
before the height one on that data.

**The bound's discriminative power is a property of the POLICY, not the terrain**, which is the thing
to remember before quoting any of it. Peak |height error|, arrived vs fell: the vm30 arm separates
(0.092 / 0.224 median; at 0.12 it flags 0.219 of box-top arrivals and 0.866 of falls) while the
PARENT is inverted -- the strict union fires on 0.953 of its arrivals against 0.807 of its falls. And
restricted to ground starts the "cost" on vm30 rises from 0.308 to 0.586, because pooling box-top
dismounts with full traverses compares a single dismount against a two-jump course.

**Whether the bound tracks the wrong MODE is UNMEASURED.** It is a hypothesis from watching episodes,
and there is no automated mode detector: `terrain_class.py` is validated as a POPULATION statistic
and its own numbers say "this plan is a misread" is not a claim it supports.
`transfer/sim/mode_probe.py` -- human labels, blind and interleaved, N noise draws on one tile -- is
the only instrument, and `watch_tracking.py --spec` is how to see what a given bound catches (it
pauses the replay ON the latch; press a key with the VIEWER focused). `fit_tracking_thresholds.py`
still does not exist, so the spec is still `bootstrap: True` and publishing consumers refuse it.

## Success is defined ONCE, and three places have to agree

- the REWARD pays on `reached_goal`, gated by the worst chunk;
- the CRITIC keeps every window up to the first failure and drops the rest -- a no-op now that the
  episode ends there;
- the EVAL reports `terminal_reward >= 0`: arrival BEFORE the first failure, so a fall in the run-out
  after the goal is a success and a robot dragged over the line after failing is not.

`FAILURE_RULE` decides what "failure" means for the last two and is `strict` -- what Isaac now
terminates on. **Score a recording made BEFORE the termination with `loose`**, or the strict rule is
applied to episodes it never ended: it then fires on 99.5% of them and reports 0.0049 against the
fall rule's 0.1051. `rollout_outcomes.assert_tracking_bounds_agree` refuses a comparison whose two
arms were latched at different bounds, because nothing in the numbers shows it -- both arms have the
column, both are populated, both are plausible.

## Training draws every spawn area; every EVAL draws the initial GROUND one

Raised areas exist so the second half of a traverse appears in the data at all. Scoring from them
reports a fraction of the course as a traverse, so `run_quick_eval.sh` passes `SPAWN_FROM=ground` on
every terrain and prints it. It costs nothing in comparability, checked rather than argued: every
family authors its ground area at exactly `freeform.FREEFORM_SPAWN_JITTER_M`, which is also what the
reset synthesises for a tile declaring none -- so a ground-only draw is BIT-IDENTICAL to the
pre-spawn-areas libraries every earlier number was measured on.

## Reduce FIRST-PER-ENV, always

A round-based stop ends when the SLOWEST env finishes its k-th episode, and an env finishes more
episodes precisely when its episodes were short -- so the surplus is drawn from the fast-failing tail
and pooling it tilts the sample back the way it came. Measured bias 0.15 on one in-distribution tile.
Worse is `--num_episodes`, a TOTAL-count stop, which cuts every slow episode still in flight: one arm
read 0.0005 against a parent's 0.1960 that way, a policy that looks destroyed and is not. Use
`--episodes_per_env`.

## Five clip-shaped assumptions the recorder had

A manifest-only library refuses every gather by name, so each of these failed loudly:

- `goal_region.goal_anchor_world` recovered the anchor from the placed clip's terminal pose; it now
  dispatches to `freeform_goal_anchor_world`, the exact inverse of the lift `freeform_goal_reached`
  applies to the robot. Both read the SAME route row. **The anchor is a PELVIS height** -- see
  `source/motion_gen/motion_gen/terrain/CLAUDE.md` for why that mattered.
- `goal_travel_m` and `last_usable_step` likewise gathered the clip (the section's own
  start-to-goal run, and the time-out horizon).
- `record_rollouts` required terms named `base_fallen` / `reset_on_clip`; the fine-tune tasks spell
  the same things `base_height` / `reset_on_section`.
- `rollout_outcomes` derived `terrain_id` from a per-CLIP table, so it was **-1 on every row** -- and
  on the box2 terrain the tile IS the initial condition, so the IC an episode ran at was unrecoverable
  from the table. It falls back to the importer's `env_terrain_id`.

## Traps, each of which cost real time

- **IsaacLab type-checks hydra overrides**: `height_tol_m=0` (an int) is refused where `0.0` is
  accepted, and it fails at hydra composition before the GPU is touched.
- **`record_rollouts.py` has its own `--history_from`**, applied AFTER hydra composition, so
  `env.commands.traj_ref.history_from=` is silently discarded. Pass the FLAG.
- **A comment inside a `\`-continued shell command ENDS the command**, making every argument after it
  a separate command. `bash -n` accepts it and these scripts have no `set -e`, so the flag simply
  vanishes and the callee uses its default. `test_shell_continuations.py` scans for it.
- **`robot.data`'s buffers are WARP arrays in a termination.** `eval.accumulate` reads them through a
  shim that converts on every access, so copying its accessor NAMES without the shim hands
  `tracking_error` a `vec3f` and it dies at the first step of collection -- and the suite passed,
  because nothing executed the function.
- **`height_tol_m` means opposite things at zero** -- see the terrain `CLAUDE.md`. It once scored an
  arm 0.0000 over 8400 episodes AND zeroed the reward's success term for five training iterations.

## Related findings worth knowing before restarting any of this

- **AWR on the TRACKING reward hit a CAUSAL ceiling**: the chunk choice is 1.6% of that reward's
  variance and a perfect best-of-4 is worth +0.0207. The terrain/progress reward above is a different
  objective and does not contradict it.
- **The measured-history arm is worth nothing** (four measurements; see the root `CLAUDE.md`).
  Filtered BC onto it buys +0.019, i.e. the ceiling. The parent ran its `generator` arm and the
  fine-tune its `measured` arm -- each on the arm it was TRAINED for, because `GenTrackerPolicy`
  refuses a mismatched pair unconditionally and has no sim override by design -- so conditioning and
  fine-tuning are confounded there; with the conditioning alone at +0.025 there is no room in +0.019
  for the confound to hide a real fine-tune effect. A 2x2 would need that override, which does not
  exist and should not be added casually.
- **FVI trap: `A = g - V` differenced the FITTED target past round 0**, making the advantage 59% too
  narrow, invisible downstream.
- The FM's own losses: the body-pose loss buys base ORIENTATION -5.6%, NOT position, for +2.2%
  flow_mse. The terrain penetration losses were REMOVED (flat `ground_z` is wrong on stairs);
  re-add only via an SDF or a height-scan LOCAL floor.
