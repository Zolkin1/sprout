# Scratchpad — todos, issues, brainstorming

> **NOTHING IN THIS FILE IS FINAL, AND ANY OF IT MAY BE WRONG.** Everything here is
> provisional and subject to change: hunches, half-measurements, single runs, numbers taken
> before a bug was found. Do not cite it, do not build on it, and do not treat a statement
> here as settled because it is written down and dated. If a claim matters, re-measure it.
> The authority is the code, the `CLAUDE.md` files and the reports under `docs/` -- never
> this file.

A dumping ground. Unlike the rest of `docs/`, nothing here has to be true, finished, or
verified: it is for thoughts in the state they arrive in. Move anything that becomes a
measured claim into its own report, or into the relevant `CLAUDE.md` if it becomes a rule --
and until it has been moved, assume it has not earned that status.

Only convention: **date what you write and say whether it is observed or guessed.** A hunch
that reads as a finding six weeks later is how this stack has lost time before.

---

## Hardware log

### 2026-09-13 — stairs, on the real robot, up AND down ✅

**Went up and down a number of different staircases around campus.** Stairs were "much better
than before". This is the first time stairs have been tested on hardware at all.

Why it matters: `transfer/obelisk/CLAUDE.md` still records the 2026-08-25 run as the only
hardware evidence — *"Flat ground: standing, and walking forwards, backwards and turning...
A box jump-up: it got onto the box but stumbled at the top. Stairs: NOT tested. Neither up nor
down."* And `mjeval`'s standing verdict was **STAIRS DOWN BAD**. Both are now out of date.

**TODO: update the hardware sections once the details are written down** — that text is
load-bearing (it is what stops people quoting terrain results the stack never earned), so it
should be corrected rather than left stale. Needs: which pair, which staircases, how many
attempts, what failed.

Pair that should have been running (confirm):
`policies/2026-09-13_v6g_partial_trk9200_hw_ready` — v6g tracker `model_9200` + the finished
two-camera generator `2026-09-12_09-25-46`. Trained with **no depth lag**.

Things worth recording while it is fresh:
- [ ] Which staircases? Rise/run, and how far from the trained bands (0.130–0.180 rise,
      0.190–0.330 tread)?
- [ ] Up vs down — was one visibly worse? Down is the one every sim instrument flagged.
- [ ] Any falls or near-falls? Where on the flight (first riser / mid / last)?
- [ ] Was `--depth-delay` in play, and did the generator cope with real ZED staleness given it
      trained on fresh frames?
- [ ] Commanded speed used. The envelope was `v_x ∈ [-0.4, 0.8]`, `|w_z| ≤ 0.6`.
- [ ] Did the robot ever read a wall/railing as a box? (the known failure mode)

---

### 2026-09-13 (later) — tried to reproduce the hardware failures in MuJoCo; mostly could NOT

Both threads below are **TABLED until the fully-trained tracker can be flown on hardware.**
Observed unless marked otherwise; every sim number is the deployed pair, driven locally.

The pair under test throughout: tracker `2026-09-11_11-45-14_paper_v6g_v1` (its `exported/
tracker.onnx` comes from `model_9200`, the only checkpoint in that dir) + generator
`2026-09-12_09-25-46_fm.onnx` -- i.e. the hardware pair exactly, not a stand-in.

**BOX JUMPS -- did not reproduce.** Built the two measured obstacles and drove them by hand.
The 0.527 m pallet and the 0.418 m bench are both climbed, repeatedly, without falling. A
scripted `--const_cmd 0.8 0 0` pass climbs the pallet (pelvis 1.17 -> 1.32 m, 3.0 s on top)
and the bench (1.03 -> 1.21 m, 2.0 s on top) and survives both. So the half-trained tracker
clears in sim the heights that failed on hardware, which points at sim2real -- tracker or
perception -- rather than at capability. NOT a strong exclusion: MuJoCo cast depth from
nominal geometry with no sensor model (`--depth_noise` was off), so a perception-side gap is
something these runs *could not* have shown.

**RUNNING GAIT -- did not reproduce.** Old pair (`2026-08-20_16-36-10_mode11_decor_plan_v1` +
`..._h64_27M_nojumpoff_fm.onnx`) against the v6g pair on flat: visually similar, torques tame.
No sim explanation found for the hardware running behaviour. Also tabled.

**Torque, since it came up.** `--log` already writes `tau_<joint>` per joint, `plots/
torques.png` and `metrics_summary.json -> torques.peak_utilization`. Two things to know:
`mj_data.actuator_force` is the PRE-CLAMP demand (verified: a saturated ankle logs 131 Nm
while `qfrc_actuator` delivers the 35 Nm limit), and MuJoCo *does* enforce the limit --
`jnt_actfrclimited` is set on 29 of 30 joints, while the actuators are `ctrllimited="false"`
with no `forcerange`, so the joint range is what binds. **Utilisation > 1.0 therefore means
SATURATION, not free torque.** Running on flat is tame, but the pallet CLIMB saturated both
ankle pitches (1.13 left / 1.42 right of limit) -- a different maneuver, and the one hardware
failed, so the tame running trace does not clear the box case. GUESS, not measured: the MJCF
limit is a flat box where a real motor's available torque falls off with speed and heat, so a
climb already at the limit in sim has less headroom on the robot than sim implies.

**The scenes** (`transfer/sim/robots/g1/g1_perceptive_<name>.xml`, run from `transfer/sim`):

| `--scene` | contents |
|---|---|
| `pallet_h527` | 0.527 m tall, 1.016 x 1.219 m (40x48 in GMA pallet), x = 4 m |
| `bench_h418` | 0.418 m tall, 0.45 x 1.80 m (bench seat depth x length), x = 4 m |
| `pallet_bench` | both, in series at y = 0 (pallet x = 4, bench x = 10) |

Regenerate (`--box height:x:y:depth:width`; the depth/width half of that spec was added
2026-09-13, the `Box` dataclass already had the fields):

```
cd transfer/sim && PYTHONPATH=../../source/robot_rl:../../source/motion_gen \
  ../../env_isaaclab/bin/python generate_stairs_scene.py \
  --robot g1_perceptive --scene pallet_bench \
  --box 0.527:4.0:0.0:1.016:1.219 --box 0.418:10.0:0.0:0.45:1.80 \
  --lane 2.44:6:20.0 --wall_lanes ""
```

Drive them (`--fm_onnx` is REQUIRED -- both runs' `exported/` hold more than one graph and
`resolve_fm_onnx` refuses to guess; `--log_dir` because the default is one fixed path per run
dir and repeat runs of a pair clobber each other; one Ctrl-C writes artifacts, a second skips
them):

```
cd transfer/sim
../../env_isaaclab/bin/python gen_tracker_runner.py \
  --load_run 2026-09-11_11-45-14_paper_v6g_v1 \
  --fm_onnx 2026-09-12_09-25-46_fm.onnx \
  --robot_name g1_perceptive --scene pallet_bench \
  --mode joystick --max_vx 2.0 --log --log_dir ../../logs/gait/v6g
```

For the OLD pair swap in `--load_run 2026-08-20_16-36-10_mode11_decor_plan_v1 --fm_onnx
2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx --allow_robot_asset_mismatch`. That
override is safe **for this pair specifically** and the reason is worth keeping: the only
differing fingerprint field is `mjcf_sha256`, and the whole MJCF delta is `3ddab280` -- two
hip-pitch PD numbers (kp 40.18/kv 2.56 -> 99.0984/6.3088). URDF, USD, joint names and body
names are byte-identical, so nothing the FK or camera-pose path reads has moved. The same
flag on a pair whose URDF or body names differ would be hiding exactly the bug the
fingerprint exists to catch. Side effect: MuJoCo uses the CORRECTED gains, so the old pair
gets a stiffer hip pitch than the world it was exported against.

Also note `..._nojumpoff` is the generator to run (root `CLAUDE.md`); `2026-08-22_19-14-39_
plan_v1_lag48_aug` sits in the same directory, is the previously DEPLOYED one, and is the
documented wrong reach. Both carry `depth_latency_probs=(0.0, 0.1, 0.4, 0.5)` and get 40 ms
of lag applied; the v6g generator has none, which is the one axis on which an old-vs-v6g A/B
is not matched.

**Still live, and NOT blocked on the robot:**
- Yaw drift, sim-reproducible: -40 deg over 20 s on a `wz = 0` command, three runs of three.
  See the brainstorming entry below.
- `ft_multi` is a fine-tune of the no-lag parent -- see Deploy below.
- The 0.40 m/s eval censoring is asymmetric against the fine-tuned arm -- see Measurement.

---

## Open issues

### Measurement / instrumentation

- **The 0.40 m/s eval band is censored.** `run_quick_eval` hardcodes
  `env.episode_length_s=40.0`, which was non-binding when 0.83 was the slowest speed. At 0.40
  it binds: 10.5% timeouts in that band, **exactly zero** at 0.83/1.17/1.50. Both arms are cut
  identically so deltas hold, but absolute rates at 0.40 are floors. Fix is safe — raising the
  budget is a no-op for every cell already at 0% timeout — but it changes the eval protocol.
  ~90 s would match the 0.83 band's margin.
  - **UPDATE 2026-09-13, and it is worse than "both arms are cut identically".** The censoring
    is ASYMMETRIC and runs against the arm being tested: the better policy survives longer, so
    it meets the clock where the weaker one has already fallen. Measured at 0.40: parent
    5.1 / 8.5 / 16.4% timeouts against fine-tuned 19.3 / 18.4 / 34.6%. So adding the 0.40 rung
    did not merely dilute the pooled delta, it deleted successes preferentially from the
    fine-tune. `run_quick_eval.sh` now scales the budget with speed (`EPLEN_DIST_M`, 41.5 m,
    never shortening a cell) -- UNCOMMITTED in the tree as of this writing. **Any four-speed
    number measured before that patch needs re-reducing**, which includes the fine-tune-vs-
    parent table at the bottom of this file.
- **`indist` / `multi` report a different reduction** ("pooled over this mix", 384 episodes)
  than `boxes` / `updown` ("pooled over 1536 training episodes"). Within-course deltas are
  like-for-like; the absolute numbers are **not** comparable across configs.
- **The AWR loop names the wrong stage on failure.** It reported "FAILED at grade" when collect
  had died three minutes earlier. The named stage is where the symptom surfaced, not where the
  run broke. Cost a diagnosis cycle twice.

### Deploy

- **obelisk does not derive the depth lag; `transfer/sim` does.** `gen_tracker_runner` reads
  `fm_meta.depth_latency_probs` and applies `FmDepthLag` automatically. obelisk needs
  `run_sim_low.sh --depth-delay <sec>` passed by hand, defaulting to none. So a lag-trained
  generator run in obelisk without the flag silently gets fresh frames — the mirror image of
  the camera-name bug, and just as quiet. **Idea:** have the controller print what the graph
  expects at load ("trained with 40 ms of camera lag; this deployment adds none").
- **All four fine-tunes are of the NO-LAG parent, `ft_multi` included** (2026-09-13,
  observed: `ft_boxes_iter005/last.pt` carries `depth_latency_probs = None`, and the queue ran
  from `2026-09-12_09-25-46`, which was trained without lag). `65522e30` made the latency
  default ON at 08:44 that morning -- after the queue had started. So the artifact currently
  lined up for deployment carries exactly the conditioning that "going fast into terrain
  didn't work" was chalked up to. `fe98fcae` verified that `from_dict` carries a stored tuple
  into a fine-tune's cfg, so **a fine-tune of a lag-trained parent stays lag-trained** -- the
  fix is to re-run `multi` from a lag-trained generator on the finished tracker, ~3.5 h, no
  re-record. Related: the Thor split measured a tolerance cliff between 20 and 40 ms, and the
  real ZED pair sits at ~42 ms median.

- **`docker/.env` and `docker/cyclonedds.xml` are gitignored per box.** A fresh machine (the
  Thor) needs both created from `.example`. A stale cyclone pin once killed every ROS node with
  `rmw_create_node: failed to create domain`.

### Training

- **`multi` needed four fixes to run at all**, each hidden behind the last: a missing sys.path
  entry, CUDA-vs-CPU normalizers, a guard comparing a file shape against an item shape, and one
  wrong fix of mine in between. The ballast path had never been exercised end to end. Worth a
  smoke test that runs one AWR iteration on a tiny buffer before a queue is armed.
- **The scan-removal re-test is still owed.** The simplified critic (one scalar head, no scan
  tokens) reaches r² +0.72–0.79 and an n1 difference ratio of 1.03–1.10 on held-out tiles,
  against the scan-bearing baseline's +0.69 / 1.01 — but on a *different course*, so it is not
  the A/B. A real one fits both critics on one course; the trunk is frozen, so it is ~12 min
  per fit.

---

## Brainstorming

*(dump freely)*

- Robot would lean sideways and tilt over sometimes. Unclear if it was because the tracker policy is only partly trained
  or if its a hardware issue or something else. Not super common but happened enough to be annoying. Sometimes in a
  straight walk, sometimes coming out of stairs, probably mostly while turning. I don't recal seeing this behavior in any sim.

  > **(claude, 2026-09-13 — hypotheses, none measured on hardware yet.)** "Mostly while turning" and
  > "coming out of stairs" together point at one documented thing before they point at hardware:
  >
  > 1. **Commanded yaw is off-distribution, and worst near stairs.** Measured over the training
  >    library, the two stair families (3500 clips) carry `|wz| max = 0.0000` — the generator has
  >    *never once* been commanded a turn on a staircase. So a turn on or just off a flight is a
  >    conditioning the generator has no training support for at all. This is the single best fit
  >    to the description and it is already written down in the root `CLAUDE.md`.
  > 2. **The tracker is a partial checkpoint** — `model_9200` of 20000, your own guess. Now cheaply
  >    testable: `model_19999` finished 2026-09-13 09:35, and a new generator is training on it.
  >    That swap is the clean A/B for this.
  > 3. **Why sim would not show it.** The teacher has *no external-disturbance DR* at all (see the
  >    DR audit), and USD vs MJCF self-collision sets are disjoint — MuJoCo has nothing on the legs
  >    above the ankle, so training's hand-vs-thigh constraint is absent in sim2sim. A lean that
  >    needs a push, or a limb contact, to develop would not appear in either simulator.
  >
  > Cheapest discriminator: the bag already has what is needed. `/obelisk/g1/debug/tick` carries
  > the served reference window and `command_effective`; if the lean coincides with a nonzero
  > commanded `w_z`, that is (1). If it happens at `w_z = 0`, it is not, and the partial tracker or
  > the hardware moves up. **Worth pulling a bag from today before anything else.**

  > **(claude, 2026-09-13 later — hypothesis (1) above is DEAD; replaced by a measured one.)**
  > Zach: the lean never happened on the stairs, always on flat, and at least once on a pure
  > straight command. That rules out the off-distribution-yaw-near-stairs story entirely. Do
  > not repeat it.
  >
  > What replaced it, OBSERVED in sim on the deployed pair: **the pair yaws off a straight
  > command.** In `logs/vel_tracking/v6g_box_parent` (the same `tracker.onnx`/generator that
  > flew), over the 9 m flat approach the heading P controller does continuous work with a
  > CONSISTENT SIGN rather than averaging to zero -- mean `|cmd_wz|` 0.063 / 0.051 / 0.057 at
  > 0.80 / 1.50 / 2.20 m/s, signed +0.034 / +0.015 / +0.023. With steering OFF
  > (`gen_tracker_runner` defaults `--steer_kp 0`) and `--const_cmd 0.8 0 0`, yaw runs
  > +0.6 -> -40 deg over 20 s and lateral y reaches -5 m; three runs of three, on the pallet,
  > bench and pallet+bench scenes. On one run it walked clean past the second obstacle, 4.6 m
  > off it.
  >
  > Why this would not have been seen in sim before: **every `vel_tracking` figure runs a
  > heading P controller at `kp = 1.5` that cancels it by construction.** Hardware has no such
  > controller (CONFIRM against the bag: is `command_effective` exactly the stick?). So the
  > robot has been running the `steer_kp = 0` condition all along and the sim has not.
  >
  > GUESS, not measured: that a heading bias is what becomes the sideways lean. A curve is not
  > a roll, there is no roll channel in `traces.npz`, and the link is unverified. Next step is
  > `vel_tracking_sweep.py --terrain flat --steer_kp 0` over several `--gaussian_noise` seeds --
  > if it drifts and rolls over on flat in MuJoCo, the symptom is on the desk. All of the above
  > is single-draw `zero_noise` (`x0 = 0`), which is what the hardware controller runs but is
  > NOT a neutral sample of the conditional.

- Running didn't look great. Tried it a few times and it looked like it was going to fall or it actually fell. It kinda
  looked like the pelvis was sinking a bit and that it was overcompensating for something or had another issue. The
  pelvis and waist looked like they rolled a lot. Once again, I don't recall seeing this in sim, but I should look again.

  > **(claude, 2026-09-13 — consistent with everything measured; low confidence on cause.)** Speed
  > is where this pair is known to fall apart, and by a lot:
  > - Parent on the boxes holdout, by commanded speed: **0.446 / 0.492 / 0.277 / 0.077** at
  >   0.40 / 0.83 / 1.17 / 1.50 m/s. It is not a gentle decline — it collapses past ~1.2.
  > - MuJoCo variance probe: speed roughly **doubles** the failure rate (1.8x on two tiles
  >   independently, 0.8 → 1.6 m/s).
  >
  > So "it looked like it was going to fall" at running speed is the expected behaviour of this
  > pair, not necessarily a separate fault. Note the deployed envelope caps `v_x` at 0.8 — if
  > running meant raising that, the robot was being asked for something measured to fail.
  >
  > Also worth separating: there IS a distinct "running" task in the repo, but it is on the
  > multiskill base, sim-only, and not part of this pipeline — do not conflate the two.
- Going fast into terrain didn't work super well. I could see this a bit in the obelisk sim when we tested with depth delay
  so it doesn't suprise me and for now I'm going to chalk that up to not having the depth lag in the FM.
- Box jumping worked at a certain height but not at others. For now I am chalking that up to the tracker policy only being
  half trained, but its possible the hardware is just degraded or that there is a seperate issue. 
- So walking and low speed maneuvers and certain box jumping and walking + stairs was all good enough, at least for now.
  Fast stuff and certain box jumps were less good.
- In the real world there are a lot more walls. The walls are generally being picked up as boxes right now and 
sometimes the robot will jump into a wall. This is not great for testing. I think we can fine tune this issue out by
making a walls terrain and then the robot should learn to just ignore them effectively.
- Movement in tight corridors wasn't great - and part of this is the tipping issue above. Part of this is probably the
partly trained.
- Seperately from the hardware we need to make sure we can get the results for the fine tuning and simulations and veloicities,
etc... that we need for the paper. On the previous RL and FM pairing the box jumps were a similar gain, probably a little
bit bigger, but the in distribution and stasirs up down gains were larger than what you reported. Also your success rates
were just very low in general in those reportings, and I know we have the tight terminations, which we want, but I'm
still kinda skeptical of it. So I should probably take a look into that.
- We need to finish the motion matching + MB + our proposed method clip velocity and jitter comparison.
- We should ideally be deploying the fine tuned stuff, which would probably be the multi terrain stuff so I need to test
the current multi fine tune in sim first.
---

## Armed and waiting (2026-09-13)

**`multi` AWR fine-tune on the NEW lag-trained FM** — armed 17:57, holding until the generator
finishes. Log `/mnt/zolkin_new/sprout/ftqueue_lag.log`, root `ftqueue_lag/`.

    CONFIGS=multi  OURS=1  OPPORTUNISTIC=" "   (3 jobs: ft + parent_eval + ft_eval)
    parent  logs/flow_matching/2026-09-13_10-08-38/best.pt
    tracker model_19999.pt

Holding on `parent_ready()`: *"training pid 3794653 still writing 2026-09-13_10-08-38"*. It
cannot start early — the decisive condition is the training PID (read out of the tfevents
filename), not the existence of `best.pt`, which `train.py` rewrites whenever val improves.

**This is a genuinely NEW pair**, not a re-fine-tune: the new generator's recordings are stamped
`model_19999.pt` (the finished tracker), where everything measured today used `model_9200`. So
its numbers are not directly comparable to the four-course table below — two things changed at
once, the tracker AND the depth lag.

`OPPORTUNISTIC=" "` (a space, not empty) because `${VAR:-default}` substitutes on empty too —
an empty string silently re-enabled borrowing GPUs 0 and 6, which belong to other people.

---

## Results worth not losing

**Fine-tune vs parent, Isaac holdout tiles** (2026-09-13). Every arm improves, and the ordering
is the one the method predicts — furthest out of distribution gains most, in-distribution least.

> **CAVEAT, and it has a direction.** Measured over four speeds including 0.40 m/s, BEFORE the
> episode-budget patch, so the 0.40 cells are censored asymmetrically against the fine-tuned
> arm (see Open issues → Measurement). These deltas are therefore LOWER BOUNDS of the same
> quantity measured at 0.83 and above. They are also not comparable to the older pairing's
> +0.351 / +0.207 / +0.105: different courses by design (`varied_courses` drawn geometry, not
> the pinned `box2` / fixed updown), a different pair, and a different termination rule.

| course | parent | fine-tuned | Δ |
|---|---|---|---|
| boxes | 0.2650 | 0.5514 | **+0.286** |
| updown | 0.5742 | 0.6504 | +0.076 |
| indist | 0.682 | 0.734 | +0.052 *(mix)* |
| multi | 0.419 | pending | *(mix)* |

**MuJoCo velocity sweep, single 0.40 m box** (2026-09-13, `logs/vel_tracking/cmp_v6g_box.png`).
150 episodes per arm, 50 × {0.8, 1.5, 2.2} m/s: parent **0.947** → fine-tuned **1.000**,
Fisher exact **p = 0.0071**. All 8 parent failures land *past* the box at 0.4–0.6 m root
height — it climbs and then collapses on the far side. Velocity tracking itself is unchanged
between arms: the fine-tune bought survival, not tracking.

**Depth lag costs nothing measurable in fit.** New run with `(0, .10, .40, .50)` vs the parent
without, at matched epoch 20: train 0.0289 vs 0.0286, `valema_flow_mse` 0.0479 vs 0.0474.
Note this is *not* evidence the lag was free — what it buys is tolerance to stale depth on
hardware, and the training metric cannot see that.
