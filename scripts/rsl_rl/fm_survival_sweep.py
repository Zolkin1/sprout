#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Closed-loop SURVIVAL for the FM tracker, over named generators and named configurations.

The measurement no training metric can make. Both ``val/*`` and ``eval/*`` score the
generator with the CLIP TEACHER's history in its context window -- the teacher-forced
regime -- and that regime has repeatedly sat near its ceiling while the closed loop
collapsed. Only a rollout, with the generator eating its own output, distinguishes them.

Everything is NAMED here rather than passed as a path, which is the point of the file: the
checkpoints, the teacher, the clip libraries and the measurement configurations all carry
what they are and where they came from, so a sweep is reproducible months later without
reconstructing which artifact was which from shell history.

**The metric.** ``envs with >=1 kept step: N/M`` out of ``play.py``'s summary: episodes that
ended via ``time_out`` rather than ``base_fallen``, i.e. the robot stayed up for the full
episode. Two traps this script closes by construction:

* ``--max_steps`` must EXCEED the 1000-step (20 s) episode, or no episode completes and the
  metric reads 0/M whatever the quality. Enforced below.
* ``history_from=clip`` and ``reference_source=clip`` both need ``ego_anchor_fraction=1.0``.
  The FM trained on ego-anchored history, where node 0's base offset is centimetres; at the
  tracker task's default of 0.0 the clip drifts off and the "correct" context is
  off-distribution. Measured cost of getting this wrong: 18/24 instead of 22/24 (at 24 envs). The
  configurations below carry the right anchoring, so it cannot be forgotten.

Measured 2026-08-03, flat library, seed 42, 1100 steps, K=1, **50 envs**. Survival is envs
that stayed up a full 20 s; vel is RMSE of measured base velocity against the COMMANDED
joystick velocity over every logged step (m/s, m/s, rad/s):

==================  ==========  ================  =====  =====  =====
cell                 survival    config            vx     vy     wz
==================  ==========  ================  =====  =====  =====
baseline @ckpt_99        3/50    closed loop       0.694  0.390  0.724
baseline @ckpt_40        3/50    closed loop       0.629  0.374  0.673
capacity @ckpt_99       26/50    closed loop       0.488  0.285  0.536
capacity @ckpt_40       12/50    closed loop       0.565  0.345  0.656
baseline @ckpt_99       40/50    teacher-forced    0.513  0.313  0.513
baseline @ckpt_40       40/50    teacher-forced    0.506  0.288  0.516
capacity @ckpt_99       42/50    teacher-forced    0.507  0.305  0.525
capacity @ckpt_40       44/50    teacher-forced    0.512  0.313  0.523
(any) @(any)            48/50    clip control      0.617  0.338  0.358
==================  ==========  ================  =====  =====  =====

Four readings, and the first is why this file defaults to 50 envs:

1. **The exposure-bias gap is LARGE.** capacity 26 -> 42, baseline 3 -> 40. An earlier
   24-env sweep put capacity's gap at 3 envs; at 50 it is 16. At 24 the metric resolves to
   ~4% per env, which cannot separate 16/24 from 19/24. Do not compare survival across
   different ``--num_envs``: a different env count samples different clips out of the 1500.
2. **Teacher-forced is flat at 40-44/50 for every model and checkpoint.** Given a correct
   context all four are equivalent, so capacity's entire advantage is autoregressive
   STABILITY, not prediction accuracy.
3. **Training length only helps the big model.** baseline is 3/50 at both ckpt_40 and
   ckpt_99 (saturated by epoch 40); capacity goes 12/50 -> 26/50, consistent with it still
   underfitting at epoch 100 (val 0.1025 below train 0.1073).
4. **``timeout_*`` velocity is survivorship-biased and must never be quoted alone.** The
   baseline's linear RMSE over survivors is 0.289 -- the BEST in the table -- because it only
   survives its three easiest envs. Over all steps it is 0.796, the worst. Also note the clip
   control has poor LINEAR error (0.617) while surviving best: tracking a clip faithfully is
   not the same as matching the joystick, and only yaw (0.358) shows its true advantage.

Euler sampling steps are NOT the lever (capacity @ckpt_99, teacher-forced, 50 envs):

    K=1  :  8 steps 42/50   32 steps 41/50
    K=12 :  8 steps 39/50   32 steps 44/50

At K=1 more steps do nothing, so the flow integration is already well resolved at 8 and the
remaining gap to the clip control is MODEL error, not discretisation. The K=12 pair moved 5
envs, which is either "plan quality matters more when a plan is served for 12 ticks" or
run-to-run spread; the K=1 pair bounds the true step effect at about 1 env, so prefer the
second reading until a second seed says otherwise.

The 24-env K / guidance sweep, kept because it is the only re-plan-rate data and the trend
survives even though the absolute numbers do not (ceiling there was 22/24):

===============  ============  ============  ==============  =============
generator        K=1           K=12          teacher-forced  history-free
===============  ============  ============  ==============  =============
baseline           2/24          7/24           22/24            9/24
noise2             3/24          9/24           -                4/24
capacity          16/24         17/24           19/24           13/24
===============  ============  ============  ==============  =============

Raising K helps the closed loop and does nothing for teacher forcing (compounding
frequency, not prediction quality). ``noise_scale=2.0`` bought 1-2 envs and made the
history-free branch worse, so robustness-side fixes were shelved in favour of capacity.

Measured 2026-08-05 on the YAW-FRAME teacher (``--teacher yaw_k12``), flat library, 200 envs,
seed 42, 1100 steps -- the first cell on the reworked observation. Only clip_control exists so
far: no generator matches this teacher's 44/node width AND the yaw convention yet.

==========================  ==========  ================  =====  =====  =====
teacher                      survival    config            vx     vy     wz
==========================  ==========  ================  =====  =====  =====
yaw_k12 @model_4400            180/200   clip control      0.660  0.327  0.410
refslim @model_12000 (old)     182/200   clip control        -      -      -
==========================  ==========  ================  =====  =====  =====

Within 2 envs of the old teacher's ceiling at 4400 of 20000 iterations (22%), which is the only
reading to take from it -- the two are NOT comparable, since the old number came from a
12000-iteration teacher. Velocity is somewhat worse than the old 50-env clip control (0.617 /
0.338 / 0.358), but that was a different env count and the docstring's own warning against
comparing across ``--num_envs`` applies.

VERIFIED VISUALLY GOOD ON FLAT GROUND, 2026-08-05, in BOTH simulators -- the best the pipeline
has looked. teacher ``yaw_k12 @model_4800`` + generator ``yaw_flat3k @best.pt`` at K=12, on the
flat3k library. Judged by eye (Isaac viewer + MuJoCo sim2sim), NOT by a survival number: the
sweep against 135/171/182 had not been run when this was written.

NOTHING BELOW IS IN GIT -- ``clips/`` and ``logs/`` are both gitignored -- so this recipe is the
only way back. Each step feeds the next.

1. CLIP LIBRARY (3.4 GB, 3000 flat clips, 60 tiles). See LIBRARIES["flat3k"] for the exact
   compose command and the --types trap. Needs clips/g1_mb_v5_body.h5.

2. TEACHER (talos, 8192 envs; model_4800 is 4800 of 20000 -- only 24% trained):
     python scripts/rsl_rl/train.py --task G1-teacher-clf-terrain-depth \
         --clip clips/g1_mb_v5_body.h5 --num_envs 8192 --max_iterations 20000 \
         --run_name yaw_k12_v1
   Note it trained on the MIXED library while the FM below trained on flat.

3. RECORDING (9.2 GB, 200000 windows, ~31 min at 256 envs, data sha 5a583069096b):
     python scripts/rsl_rl/record_data.py --task G1-teacher-clf-terrain-depth-play \
         --load_run 2026-08-04_21-55-17_yaw_k12_v1 --checkpoint <...>/model_4800.pt \
         --clip clips/g1_mb_v5_flat3k_body.h5 \
         --num_windows 200000 --num_envs 256 --clip_device cpu --headless
   history_anchor=plan, H=8, F=62 (--ref_future_len defaults to FlowMatchingConfig.future_len).
   KNOWN GAP: the -play task strips resample_depth_noise + the camera extrinsics/intrinsics DR,
   so the recorded depth is CLEAN at the nominal camera pose -- while the teacher trained on
   noisy depth from randomized poses, and at deploy ONE image feeds both. Use the non-play task
   (G1-teacher-clf-terrain-depth) to keep it, and then set noise_depth=0 or it double-counts.
   Barely matters on flat (depth is near-constant); it matters a lot on terrain.

4. GENERATOR (27.3M XL, 40 epochs, ~4 min/epoch, body-pose OFF):
     python scripts/flow_matching/train.py --data <the .h5 from step 3> \
         --d_model 512 --nhead 8 --num_layers 8 --dim_feedforward 2048 \
         --batch_size 256 --lr 3e-4 --num_epochs 40 --cosine --wandb \
         --run_name xl_yawframe_flat3k_bp_off_40ep
   Active losses are only flow_mse + 0.01 * vel_consistency; joint_vel_consistency and fk_pose
   auto-gate to 0 (their blocks are not in the selection) and both body_pose terms were off.

5. LOOK AT IT -- Isaac (K=12 comes from the task default):
     python scripts/rsl_rl/play.py --task G1-tracker-clf-fm-terrain-depth-play \
         --load_run 2026-08-04_21-55-17_yaw_k12_v1 --checkpoint <...>/model_4800.pt \
         --fm_checkpoint <...>/best.pt --clip clips/g1_mb_v5_flat3k_body.h5 \
         --clip_device cpu --num_envs 8

6. LOOK AT IT -- MuJoCo (reads exported/{tracker,fm}.onnx, which step 5 writes):
     cd transfer/sim && python gen_tracker_runner.py \
         --load_run 2026-08-04_21-55-17_yaw_k12_v1 \
         --robot_name g1_29j --scene basic_scene --mode joystick --command smooth_ramp_running
   --robot_name g1_29j is REQUIRED (the default g1_21j is a 21-DoF model under a 29-DoF policy),
   and you must cd first. --fm_update_every is DEAD here: the tracker.onnx metadata wins, which
   is why step 5's K ends up governing this run. Verified both contracts carry a camera spec
   (tracker policy_parameters.depth_camera_spec + fm_meta.extero_spec), which is what
   GenTrackerPolicy.load() cross-checks. CPU-only, ~100 ms/step.

CAVEATS THAT TRAVEL WITH THIS PAIR: the teacher is 24% trained; the FM saw flat ground only, so
neither the depth conditioning nor the body-pose loss was meaningfully exercised; and the depth
was clean. Do not read "looks good on flat" as a terrain result.

Examples:
    # the headline pair for one generator
    python scripts/rsl_rl/fm_survival_sweep.py --models capacity \\
        --configs closed_loop,teacher_forced

    # re-plan-rate sweep, closed loop only
    python scripts/rsl_rl/fm_survival_sweep.py --models capacity --configs closed_loop \\
        --k 1,5,12

    # what the policy can do with a perfect reference (generator irrelevant)
    python scripts/rsl_rl/fm_survival_sweep.py --models capacity --configs clip_control

    # print the resolved commands without launching anything
    python scripts/rsl_rl/fm_survival_sweep.py --models all --configs all --dry_run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field


def _repo_root() -> pathlib.Path:
    """The repository root, verified rather than assumed.

    Anchored on this file's location, then CHECKED for the two top-level directories the
    repo always has -- so moving this script fails loudly here instead of silently
    resolving every asset path against the wrong tree.

    Raises:
        RuntimeError: If the resolved root does not look like the repo.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    if not ((root / "source").is_dir() and (root / "scripts").is_dir()):
        raise RuntimeError(
            f"resolved repo root {root} has no source/ + scripts/; this script moved and "
            "its path arithmetic no longer holds."
        )
    return root


REPO = _repo_root()

#: Python for the spawned rollouts: THIS interpreter, whichever it is. The FM tracker needs
#: to import isaaclab, so the script has to be run with env_isaaclab's python anyway -- and
#: reusing it means a worktree works too, where ``env_isaaclab`` exists only in the main
#: checkout (it is gitignored) and ``REPO / "env_isaaclab"`` would not resolve.
PYTHON = sys.executable

#: The play task. The DEPTH variant specifically: the height-scan tracker has no cameras in
#: its scene, and ``FmTrackingCommand._read_extero`` picks the sensor off the CHECKPOINT, so
#: a depth-trained generator raises there. It also forces ``tracking_mode="joint"``, which
#: keeps the body CLF's 13 ``@root`` links (which no generator predicts) out of the picture.
TASK = "G1-tracker-clf-fm-terrain-depth-play"


@dataclass(frozen=True)
class Teacher:
    """The tracker policy the generator drives. Named for provenance."""

    run: str
    checkpoint: str
    note: str

    @property
    def path(self) -> pathlib.Path:
        """Absolute path of the DEFAULT checkpoint."""
        return self.ckpt_path(self.checkpoint)

    def ckpt_path(self, name: str) -> pathlib.Path:
        """Absolute path of one checkpoint of this run.

        Args:
            name: ``model_19999`` / ``model_19999.pt`` / an absolute path.

        Returns:
            The checkpoint path, which may not exist -- callers check.
        """
        if os.path.isabs(name):
            return pathlib.Path(name)
        stem = name if name.endswith(".pt") else f"{name}.pt"
        return REPO / "logs" / "rsl_rl" / "g1" / self.run / stem


#: Named teachers. The teacher sets EVERY ceiling in this sweep -- clip_control IS its own
#: tracking ability with the generator switched off -- so numbers are only comparable at a
#: FIXED teacher. Registered side by side rather than edited in place, so the tables above
#: keep pointing at the artifact that produced them.
TEACHERS: dict[str, Teacher] = {
    "refslim": Teacher(
        run="2026-08-01_08-49-20_depth_v5_refslim",
        checkpoint="model_12000.pt",
        note="12000 it, body-mode depth teacher, 41/node (451) reference window. Trained on "
             "talos. What every table in this docstring was measured on. PREDATES the "
             "reference-window rework, so play.py on this branch REFUSES it on the "
             "base_pose_convention mismatch -- it is a provenance record, not a runnable cell.",
    ),
    "yaw_k12": Teacher(
        run="2026-08-04_21-55-17_yaw_k12_v1",
        checkpoint="model_4800.pt",
        note="44/node (572) reference window, yaw-aligned anchor, ego_anchor_every=12, torso "
             "CoM + live restitution DR, base_pos noise (0.06, 0.06, 0.03). Trained on talos "
             "against the MIXED library (g1_mb_v5_body.h5). STILL TRAINING -- 4400 of 20000 "
             "iterations, so clip_control here is an under-trained policy and is NOT "
             "comparable to the 182/200 recorded above (a 12000-it teacher). Pass "
             "--teacher_checkpoint to move it as the run progresses.",
    ),
}

#: The selected teacher. Rebound by :func:`main` from ``--teacher``; every helper reads it at
#: call time. Defaults to the yaw-frame teacher because the older one cannot be played on this
#: branch at all -- the convention guard refuses it before the env is built.
TEACHER = TEACHERS["yaw_k12"]


@dataclass(frozen=True)
class Library:
    """A clip library. Carries the TERRAIN as well as the motions, so it picks the ground."""

    path: str
    note: str


#: Named clip libraries. ``flat`` is the one every number in the table above was measured on
#: -- keep using it for anything meant to be comparable.
LIBRARIES: dict[str, Library] = {
    "flat": Library("clips/g1_mb_v5_flat_body.h5",
                    "1500 flat-only clips (terrain_type_code all 0), a subset of "
                    "g1_mb_v5_body.h5. THE comparability baseline."),
    "mixed": Library("clips/g1_mb_v5_body.h5",
                     "the full 10k-clip terrain library: flat + stairs + boxes + distractors."),
    "small": Library("clips/g1_mb_v5_small_body.h5",
                     "36 clips, mixed terrain. Quick viewer runs, too small to measure on."),
    "flat3k": Library("clips/g1_mb_v5_flat3k_body.h5",
                      "ALL 3000 flat clips of g1_mb_v5_body.h5, 60 flat tiles. What the "
                      "yaw_flat3k generator was TRAINED on -- 'flat' above is an evenly "
                      "sampled HALF of the same pool (--flat-clips 1500), so use flat3k for "
                      "training data and flat for numbers comparable to the tables above. "
                      "Rebuild (clips/ is gitignored):\n"
                      "  python scripts/rsl_rl/compose_terrain_library.py \\\n"
                      "      --mix \"clips/g1_mb_v5_body.h5:0\" \\\n"
                      "      --flat-from clips/g1_mb_v5_body.h5 \\\n"
                      "      --flat-clips 3000 --flat-tiles 60 \\\n"
                      "      --out clips/g1_mb_v5_flat3k_body.h5\n"
                      "Do NOT add --types flat: the mix source would then register as a "
                      "second flat pool and every clip would be carried twice."),
}


@dataclass(frozen=True)
class Generator:
    """A trained FM checkpoint, with what it is and what produced it."""

    run: str
    params: str
    trained_on: str
    note: str
    checkpoint: str = "best.pt"

    @property
    def path(self) -> pathlib.Path:
        """Absolute path of this generator's DEFAULT checkpoint."""
        return self.ckpt_path(self.checkpoint)

    def ckpt_path(self, name: str) -> pathlib.Path:
        """Absolute path of one checkpoint of this run.

        Args:
            name: ``best``, ``ckpt_99``, ... (with or without the ``.pt`` suffix).

        Returns:
            The checkpoint path, which may not exist -- callers check.
        """
        stem = name if name.endswith(".pt") else f"{name}.pt"
        return REPO / "logs" / "flow_matching" / self.run / stem


#: Recordings, named once so the generators can point at them by name.
RECORDINGS: dict[str, str] = {
    "depth_v5_200k": "logs/rsl_rl/g1/2026-08-01_08-49-20_depth_v5_refslim/recorded_data/"
                     "recorded_20260802_200006.h5   (200k windows, F=75, 9.8 GB, stamps "
                     "reference_blocks=[base_pose,base_vel,joint_pos])",
    "depth_v1_200k": "logs/rsl_rl/g1/2026-07-30_13-24-10_depth_v1/recorded_data/"
                     "recorded_20260802_161227.h5   (200k windows, F=75, the OLD teacher)",
    "depth_v1_30k": "logs/rsl_rl/g1/2026-07-30_13-24-10_depth_v1/recorded_data/"
                    "recorded_20260730_193528.h5   (30k windows, F=65; the cross-teacher "
                    "--eval_data set)",
}

#: Trained generators, oldest first. All D=41, F=62 (so K<=12 is servable), N=4, depth.
GENERATORS: dict[str, Generator] = {
    "baseline": Generator(
        run="2026-08-02_20-30-38_ar_window_v5_f62_41node",
        params="2.74M (d_model 256, 4 layers, ff 512)",
        trained_on="depth_v5_200k",
        note="the first 41/node generator. 100 epochs, noise_scale 1.0. UNDERFIT: val "
             "0.1376 sat below train 0.1466 and both were still falling.",
    ),
    "noise2": Generator(
        run="2026-08-02_23-04-29_ar_window_v5_f62_41node_noise2",
        params="2.74M (as baseline)",
        trained_on="depth_v5_200k",
        note="baseline + noise_scale=2.0, to close the history train/deploy mismatch. "
             "Bought 1-2 envs and made the history-free branch worse. Shelved.",
    ),
    "capacity": Generator(
        run="2026-08-03_00-07-59_ar_window_v5_f62_41node_big",
        params="9.52M (d_model 384, 6 layers, ff 1024) + cosine",
        trained_on="depth_v5_200k",
        note="the one that worked: closed loop 2/24 -> 16/24. Still underfitting "
             "(val 0.1025 < train 0.1073), so more capacity may keep paying.",
    ),
    "dr27M": Generator(
        run="2026-08-05_17-05-00_xl27M_drdepth_flat3k_100ep",
        params="27.3M (d_model 512, nhead 8, 8 layers, ff 2048) + cosine, body-pose OFF",
        trained_on="flat3k_200k_DRdepth",
        note="First generator trained on depth carrying the sim's SENSOR DR (recorded on the "
             "TRAIN task, so resample_depth_noise + the camera extrinsics/intrinsics DR are "
             "live; extero_corrupted=True, so the dataset zeroed noise_depth). 100 epochs; "
             "best.pt is epoch 95. val flow_mse 0.1170 (null 0.1434), rmse_base_pos 0.0378. "
             "Pairs with dr9M below -- identical data and settings, only width differs.",
    ),
    "dr9M": Generator(
        run="2026-08-05_23-59-27_m9_drdepth_flat3k_100ep",
        params="9.55M (d_model 384, nhead 6, 6 layers, ff 1024) + cosine, body-pose OFF",
        trained_on="flat3k_200k_DRdepth",
        note="The capacity control for dr27M: same 200k DR-depth windows, same schedule, only "
             "d_model/layers/ff reduced. nhead 6 keeps head_dim at 64, matching the XL. val "
             "flow_mse 0.1351 (null 0.1612), rmse_base_pos 0.0460 -- 15%/22% worse than the "
             "27M, so capacity was STILL paying at 100 epochs on this data.",
    ),
    "yaw_flat3k": Generator(
        run="2026-08-05_11-27-19_xl_yawframe_flat3k_bp_off_40ep",
        params="27.3M (d_model 512, nhead 8, 8 layers, ff 2048) + cosine, body-pose OFF",
        trained_on="flat3k_200k (data sha 5a583069096b)",
        note="THE FIRST GENERATOR THAT FITS THE YAW-FRAME TEACHER: 44/node (D=44), F=62, so "
             "K=12 is servable (F >= max_offset 50 + K). Every generator above is 41/node and "
             "F=51, hence K=1-only and refused against this teacher. best.pt is epoch 35 of "
             "40; final val flow_mse 0.1287 (null 0.1510, so the history IS being used), "
             "rmse_base_pos 0.0447, ori_geodesic 0.0725, rmse_joint_pos 0.0977. Still "
             "underfitting at 40 epochs (train 0.1297 and falling), and train.py has no "
             "resume, so a longer run means starting over. Do NOT compare its flow_mse to the "
             "F=51 rows above: the mean now includes offsets 51-61, whose x error runs "
             "0.089-0.110, so extending the horizon raises any horizon-averaged metric "
             "mechanically. At MATCHED offsets it is comparable or better -- but the data is "
             "FLAT, which makes z nearly trivial (offset-50 z 0.019 vs 0.062 on mixed), so "
             "that is a dataset effect and NOT evidence the yaw frame helped.",
    ),
    "xl_ema": Generator(
        run="2026-08-03_11-12-14_ar_window_v5_f62_41node_xl_ema",
        params="27.3M (as xl) + cosine + weight EMA (decay 0.999)",
        trained_on="depth_v5_200k",
        note="xl with --ema_decay 0.999. EMA moved NOTHING on val: val_ema/flow_mse 0.0842 "
             "vs val/flow_mse 0.0836, rmse_base_pos identical at 0.0419. Swept because val "
             "has mispredicted closed-loop twice. load_fm_context prefers the EMA weights; "
             "pass use_ema=False for the raw ones.",
    ),
    "xl": Generator(
        run="2026-08-03_08-23-05_ar_window_v5_f62_41node_xl",
        params="27.3M (d_model 512, 8 layers, ff 2048) + cosine",
        trained_on="depth_v5_200k",
        note="trained on TALOS -- copy best.pt into logs/flow_matching/<run>/ before "
             "sweeping it here, or this entry will fail its path check.",
    ),
}


@dataclass(frozen=True)
class Config:
    """A named measurement configuration: the env overrides plus what it isolates."""

    overrides: dict[str, str]
    isolates: str
    #: The result does not depend on WHICH generator is loaded. A checkpoint is still
    #: required -- ``FmTrackingCommand`` raises on an empty ``fm_checkpoint``, so the env
    #: cannot even be built without one -- but the generator is computed and discarded, so
    #: the sweep runs this ONCE rather than once per model and labels it without a name.
    generator_independent: bool = False
    no_generator: bool = False
    """Pass an EMPTY --fm_checkpoint, so FmTrackingCommand never loads a generator.

    ``generator_independent`` only means the result does not depend on which generator was
    named; the model was still loaded to satisfy the command term. This goes further and is
    what fm_tracking_cmd_cfg documents as "the pure clip control": no FM cost, no FM
    nondeterminism, and no need for a compatible checkpoint to exist on disk at all -- which
    also means this cell survives a teacher whose width no generator matches yet."""
    #: Sweeping K is meaningless for configurations that do not consume the prediction.
    k_matters: bool = True
    extra: list[str] = field(default_factory=list)


#: The four configurations worth distinguishing. Each carries its own anchoring so the
#: ego-anchor trap cannot be reintroduced by forgetting a flag.
CONFIGS: dict[str, Config] = {
    "closed_loop": Config(
        overrides={"history_from": "generator"},
        isolates="THE DEPLOYABLE CASE. The generator conditions on its own last N served "
                 "nodes. Anchoring is irrelevant here (nothing reads the clip reference) -- "
                 "verified byte-identical at ego_anchor_fraction 0.0 and 1.0.",
    ),
    "teacher_forced": Config(
        # BROKEN for a PLAN-FRAME generator, and not by anything this config can fix.
        # FmTrackingCommand._clip_history_node builds its row via reference_window_obs, which
        # anchors on the MEASURED ROBOT AT THAT STEP. Its docstring still claims that is
        # "byte-for-byte the row the DATASET builds a history from" -- true until 3b3af11, which
        # moved the recorder to substitute the PLAN anchor into history rows 0..H-2. So this cell
        # now feeds per-step-anchored history to a generator trained on plan-anchored history,
        # while closed_loop (which pushes _prev_served_node raw, already in its plan's frame) is
        # correct. Measured consequence: teacher_forced 158/200 BELOW closed_loop 176/200 -- not
        # a ceiling at all. Fix belongs in _clip_history_node, not here.
        #
        # The ego_anchor_every="1" pin was also removed (same staleness as clip_control below:
        # match the TEACHER's cadence, do not hardcode). That is correct on principle but was
        # NOT the cause of the inversion -- removing it moved 163 -> 158, i.e. nothing beyond
        # run-to-run spread.
        overrides={"history_from": "clip", "ego_anchor_fraction": "1.0"},
        isolates="The generator still predicts and the tracker still follows, but the "
                 "context is always correct. closed_loop far below this = the LOOP "
                 "diverges; the two together = the MODEL is the limit.",
    ),
    "clip_control": Config(
        # NO ego_anchor_every override: it must match what the TEACHER trained under, and the
        # cfg default now follows REPLAN_PERIOD_K (12). It used to be pinned to "1" here,
        # correctly, back when the teacher also trained at 1 -- hardcoding it again is exactly
        # how that goes stale. ego_anchor_fraction=1.0 still has to be forced: the tracker
        # task defaults it to 0.0 (the deploy value), which would leave the clip on its fixed
        # world placement instead of the ego-anchored one the policy saw.
        overrides={"reference_source": "clip", "ego_anchor_fraction": "1.0"},
        isolates="The policy-side CEILING: the tracker follows the real clip and NO generator "
                 "is loaded at all. No FM contribution, and no FM nondeterminism either.",
        generator_independent=True,
        k_matters=False,
        no_generator=True,
    ),
    "history_free": Config(
        overrides={"history_from": "generator", "fm_guidance_scale": "0.0"},
        isolates="Guidance w=0 returns the learned null branch, so the history is IGNORED. "
                 "A free no-retrain probe of how much the model's own history is worth: "
                 "above closed_loop means it is actively harmful.",
    ),
}


def _check_assets(gen_names: list[str], ckpts: list[str], library: str, teacher_ckpt: str,
                  fatal: bool = True) -> None:
    """Fail before launching anything if a named asset is not on disk.

    Every run costs a simulator start-up, so a missing checkpoint should surface now rather
    than after the first rollout. Under ``--dry_run`` this only warns: printing the command
    for a checkpoint still sitting on talos is a legitimate thing to want.

    Args:
        gen_names: Generators about to be swept.
        ckpts: Checkpoint names each will be swept at.
        library: Clip-library key.
        teacher_ckpt: Teacher checkpoint name being evaluated.
        fatal: Raise on a missing asset (``False`` warns and continues).

    Raises:
        SystemExit: With the offending paths, when ``fatal``.
    """
    tpath = TEACHER.ckpt_path(teacher_ckpt)
    missing = [f"teacher {TEACHER.run}/{teacher_ckpt} -> {tpath}"] if not tpath.is_file() else []
    missing += [f"generator {n} @ {c} -> {GENERATORS[n].ckpt_path(c)}"
                for n in gen_names for c in ckpts
                if not GENERATORS[n].ckpt_path(c).is_file()]
    lib = REPO / LIBRARIES[library].path
    if not lib.is_file():
        missing.append(f"library {library} -> {lib}")
    if missing:
        msg = "[fm-survival] missing assets:\n  " + "\n  ".join(missing)
        if fatal:
            raise SystemExit(msg)
        print(msg + "\n[fm-survival] --dry_run: printing commands anyway.\n")


def build_command(fm_path: pathlib.Path, cfg: Config, library: str, k: int,
                  num_envs: int, max_steps: int, seed: int,
                  sampling_steps: int | None = None,
                  teacher_path: pathlib.Path | None = None) -> list[str]:
    """The full ``play.py`` argv for one measurement.

    Args:
        fm_path: The generator checkpoint to load.
        cfg: The named configuration supplying the env overrides.
        library: Key into :data:`LIBRARIES`.
        k: ``fm_update_every`` (the re-plan period, in control steps).
        num_envs / max_steps / seed: Rollout size and determinism.
        teacher_path: Teacher checkpoint to load; defaults to the registry's.
        sampling_steps: Euler steps for the flow integration, or ``None`` to use whatever
            the checkpoint trained with (8). More steps cut DISCRETISATION error without
            retraining -- the free half of "is the gap the model or the sampler?".

    Returns:
        argv suitable for :func:`subprocess.run`.
    """
    teacher_path = teacher_path or TEACHER.path
    argv = [
        PYTHON, str(REPO / "scripts" / "rsl_rl" / "play.py"),
        "--task", TASK,
        "--load_run", TEACHER.run, "--checkpoint", str(teacher_path),
        "--clip", LIBRARIES[library].path, "--clip_device", "cpu",
        "--num_envs", str(num_envs), "--max_steps", str(max_steps),
        "--seed", str(seed), "--headless",
    ]
    # OMIT the flag entirely for a no-generator cell. The tracker task already defaults
    # fm_checkpoint to "" ("no generator at all" when reference_source=clip), and leaving the
    # flag off also stops play.py trying to export an fm.onnx it has no checkpoint for.
    if not cfg.no_generator:
        argv += ["--fm_checkpoint", str(fm_path)]
    argv += [f"env.commands.traj_ref.fm_update_every={k}"]
    if sampling_steps is not None:
        argv += [f"env.commands.traj_ref.fm_num_sampling_steps={sampling_steps}"]
    argv += [f"env.commands.traj_ref.{key}={val}" for key, val in cfg.overrides.items()]
    argv += cfg.extra
    return argv


#: Where ``play.py`` leaves its summary. OVERWRITTEN by every run against the same task,
#: so the sweep copies both files per cell immediately after each rollout.
def _artifact_dir() -> pathlib.Path:
    """Directory ``play.py`` writes ``stats.txt`` / ``tracking_stats.csv`` into."""
    return REPO / "logs" / "rsl_rl" / "g1" / TEACHER.run / "plots" / TASK


#: Commanded-velocity channels, as ``play_plots`` names them.
VEL_CHANNELS = ("forward_vx", "lateral_vy", "yaw_wz")


def parse_velocity_rmse(csv_path: pathlib.Path) -> dict:
    """Joystick-level velocity-tracking RMSE, from ``tracking_stats.csv``.

    ``play_plots`` reports the MEAN SQUARED error of the robot's measured ``(vx, vy, wz)``
    against the commanded ``target_vel``, per channel, time-averaged per env then averaged
    over envs. This takes the square root so the numbers read in m/s and rad/s.

    Both variants are returned and they answer different questions:

    * ``vel_rmse_*`` -- every logged step. Includes the flailing before a fall, so a model
      that falls often is penalised. The honest cross-model number.
    * ``timeout_vel_rmse_*`` -- steps in episodes that survived. CONDITIONED on success, so
      a model that only survives its two easiest envs can look excellent here. Read it
      alongside the survival count, never alone.

    Args:
        csv_path: A ``tracking_stats.csv`` written by ``play_plots``.

    Returns:
        Per-channel RMSEs under both prefixes, plus ``vel_rmse_lin`` (planar speed error,
        m/s) and ``vel_rmse_yaw`` (rad/s). Empty when the file is missing.
    """
    if not csv_path.is_file():
        return {}
    raw = {}
    for line in csv_path.read_text().splitlines()[1:]:
        key, _, val = line.partition(",")
        try:
            raw[key] = float(val)
        except ValueError:
            continue
    out: dict[str, float] = {}
    for prefix, label in (("", "vel_rmse"), ("timeout_", "timeout_vel_rmse")):
        mses = [raw.get(f"{prefix}vel_track_{c}_mse_mean") for c in VEL_CHANNELS]
        if any(v is None for v in mses):
            continue
        for c, mse in zip(VEL_CHANNELS, mses):
            out[f"{label}_{c}"] = math.sqrt(mse)
        # Dimensionally honest aggregates: the two linear channels combine into a planar
        # speed error; yaw is rad/s and stays separate.
        out[f"{label}_lin"] = math.sqrt(mses[0] + mses[1])
        out[f"{label}_yaw"] = math.sqrt(mses[2])
    return out


_SURVIVAL = re.compile(r"envs with >=1 kept step:\s*(\d+)\s*/\s*(\d+)")
_KEPT = re.compile(r"steps kept:\s*(\d+)\s*/\s*(\d+)\s*\(([\d.]+)%\)")


def parse_survival(log_text: str) -> dict | None:
    """Pull the survival numbers out of a ``play.py`` log.

    Returns:
        ``{"envs_up", "envs_total", "steps_kept", "steps_total", "steps_pct"}``, or ``None``
        when the summary block is absent (which for a >1000-step run means the rollout died
        before finishing rather than that nothing survived).
    """
    m, s = _SURVIVAL.search(log_text), _KEPT.search(log_text)
    if m is None:
        return None
    out = {"envs_up": int(m.group(1)), "envs_total": int(m.group(2))}
    if s is not None:
        out |= {"steps_kept": int(s.group(1)), "steps_total": int(s.group(2)),
                "steps_pct": float(s.group(3))}
    return out


def main() -> None:
    """Parse arguments, run each (generator, configuration, K) cell, report a table."""
    # Rebound from --teacher below, before any helper reads it. Declared up here because
    # Python requires `global` to precede every use of the name in the function.
    global TEACHER
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="capacity",
                   help=f"Comma-separated generators, or 'all'. Known: "
                        f"{','.join(GENERATORS)}")
    p.add_argument("--configs", default="closed_loop,teacher_forced",
                   help=f"Comma-separated configurations, or 'all'. Known: "
                        f"{','.join(CONFIGS)}")
    p.add_argument("--k", default="1",
                   help="Comma-separated fm_update_every values (re-plan period).")
    p.add_argument("--teacher", default="yaw_k12", choices=sorted(TEACHERS),
                   help="Which registered teacher to evaluate. 'refslim' is the artifact the "
                        "docstring tables were measured on and is REFUSED at load on this "
                        "branch (convention guard); it is kept for provenance.")
    p.add_argument("--teacher_checkpoint", default=None,
                   help="Teacher iteration to evaluate, e.g. 'model_19999' (defaults to "
                        "the selected --teacher's own checkpoint). The teacher sets EVERY "
                        "ceiling in this "
                        "sweep -- clip_control is its own tracking ability with the "
                        "generator switched off -- so a newer one moves all of them, and "
                        "generator numbers are only comparable at a FIXED teacher.")
    p.add_argument("--library", default="flat", choices=sorted(LIBRARIES),
                   help="Clip library, which also picks the TERRAIN. Default flat, which is "
                        "what every recorded number was measured on.")
    p.add_argument("--checkpoints", default="best",
                   help="Comma-separated checkpoints per generator: 'best', 'ckpt_99' (most "
                        "trained), 'ckpt_40' (mid-training), ... Every named generator is "
                        "swept at every named checkpoint.")
    p.add_argument("--sampling_steps", default="default",
                   help="Comma-separated Euler step counts for the flow integration, or "
                        "'default' to use the checkpoint's (8). Raising this is a "
                        "no-retrain probe: it reduces integration error only, so a gap that "
                        "closes here was the SAMPLER, not the model. Costs latency "
                        "proportionally -- the FM is already ~99%% of the deploy budget.")
    p.add_argument("--num_envs", type=int, default=200,
                   help="Rollout width, and the metric's resolution: 1/num_envs. 200 by "
                        "default (0.5%% per env). The history here is a warning -- at 24 envs "
                        "a sweep put one model's exposure-bias gap at 3 envs when 50 envs "
                        "showed 16, and differences of 2-3 envs were being read as real when "
                        "they were quantisation. Costs wall time roughly linearly.")
    p.add_argument("--max_steps", type=int, default=1100,
                   help="Must exceed the 1000-step episode; the metric only counts episodes "
                        "that ended via time_out.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Output directory (default logs/fm_survival/<timestamp>).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the resolved commands and exit without launching a simulator.")
    args = p.parse_args()

    if args.max_steps <= 1000:
        raise SystemExit(
            f"--max_steps {args.max_steps} does not exceed the 1000-step (20 s) episode, so "
            "no episode can end via time_out and every cell would read 0. Use >= 1100."
        )
    models = list(GENERATORS) if args.models == "all" else args.models.split(",")
    configs = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    for name, known in (("model", (models, GENERATORS)), ("config", (configs, CONFIGS))):
        bad = [n for n in known[0] if n not in known[1]]
        if bad:
            raise SystemExit(f"unknown {name}(s) {bad}; known: {list(known[1])}")
    ks = [int(v) for v in args.k.split(",")]
    ckpts = args.checkpoints.split(",")
    steps_list = [None if v.strip() in ("", "default") else int(v)
                  for v in args.sampling_steps.split(",")]

    # One teacher per invocation is the intended usage -- cells are only comparable at a
    # fixed teacher, so the helpers all read this single selection.
    TEACHER = TEACHERS[args.teacher]
    teacher_ckpt = args.teacher_checkpoint or TEACHER.checkpoint
    # Cells that load no generator must not require one on disk -- that is the whole point of
    # no_generator, and it is what lets clip_control run against a teacher no generator fits.
    needs_gen = [c for c in configs if not CONFIGS[c].no_generator]
    _check_assets(models if needs_gen else [], ckpts, args.library, teacher_ckpt,
                  fatal=not args.dry_run)

    # One cell per (generator, config, K). A configuration that ignores the generator
    # (clip_control) gets exactly ONE cell however many generators were named -- running it
    # per generator would repeat an identical rollout and invite reading it as a per-model
    # number, which it is not.
    cells: list[tuple[str, str, str, int, int | None]] = []
    for cname in configs:
        cfg = CONFIGS[cname]
        k_values = ks if cfg.k_matters else ks[:1]
        for k in k_values:
            for st in steps_list:
                if cfg.generator_independent:
                    # One cell per (K, steps). models[0] @ ckpts[0] is loaded only to
                    # satisfy the command term; nothing it predicts reaches the robot, so
                    # neither axis can move it -- but keep them for a labelled control.
                    cells.append((models[0], ckpts[0], cname, k, st))
                else:
                    cells += [(m, c, cname, k, st) for m in models for c in ckpts]

    out_dir = pathlib.Path(args.out) if args.out else (
        REPO / "logs" / "fm_survival" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fm-survival] teacher   {TEACHER.run}/{teacher_ckpt}")
    print(f"[fm-survival]           {TEACHER.note}")
    print(f"[fm-survival] library   {args.library}: {LIBRARIES[args.library].note}")
    print(f"[fm-survival] {len(cells)} cell(s), {args.num_envs} envs x {args.max_steps} "
          f"steps, seed {args.seed} -> {out_dir}\n")

    results = []
    for mname, ckpt, cname, k, st in cells:
        gen = GENERATORS[mname]
        independent = CONFIGS[cname].generator_independent
        suffix = f"_k{k}" + (f"_s{st}" if st is not None else "")
        tag = f"{cname}{suffix}" if independent else f"{mname}_{ckpt}_{cname}{suffix}"
        argv = build_command(gen.ckpt_path(ckpt), CONFIGS[cname], args.library, k,
                             args.num_envs, args.max_steps, args.seed, st,
                             TEACHER.ckpt_path(teacher_ckpt))
        if args.dry_run:
            print(f"# {tag}\n{' '.join(argv)}\n")
            continue
        log = out_dir / f"{tag}.log"
        # CLEAR the shared artifacts first. play.py writes stats.txt / tracking_stats.csv to
        # ONE path per task, so a cell that dies leaves the PREVIOUS cell's files behind --
        # and copying those reports the previous cell's survival for a run that never
        # started. That happened: two failed 32-step cells printed their predecessor's
        # 42/50 with only rc=1 to give it away.
        for stale in ("stats.txt", "tracking_stats.csv"):
            (_artifact_dir() / stale).unlink(missing_ok=True)
        print(f"[{time.strftime('%H:%M:%S')}] {tag} ...", flush=True)
        with log.open("w") as fh:
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=REPO, env={**os.environ,
                                               "PYTHONPATH": "source/robot_rl:source/motion_gen"}
                                ).returncode
        # play.py rewrites stats.txt / tracking_stats.csv in ONE place per task, so copy
        # them out before the next cell clobbers them. stats.txt is also the reliable
        # survival source: it is written from an accumulated list, whereas the tail of
        # stdout can be lost when the simulator exits.
        art, kept = _artifact_dir(), {}
        for fname, suffix in (("stats.txt", "_stats.txt"),
                              ("tracking_stats.csv", "_tracking_stats.csv")):
            src = art / fname
            if src.is_file():
                kept[fname] = out_dir / f"{tag}{suffix}"
                shutil.copy2(src, kept[fname])
        # A non-zero exit means nothing about this cell is trustworthy, so do not parse at
        # all -- belt and braces with the artifact clearing above.
        if rc != 0:
            got, vel = None, {}
        else:
            got = (parse_survival(kept["stats.txt"].read_text(errors="replace"))
                   if "stats.txt" in kept else None) or \
                  parse_survival(log.read_text(errors="replace"))
            vel = parse_velocity_rmse(
                kept.get("tracking_stats.csv", pathlib.Path("/nonexistent")))
        row = {"generator": "-" if independent else mname,
               "checkpoint": "-" if independent else ckpt, "config": cname, "k": k,
               "sampling_steps": st if st is not None else "ckpt",
               "loaded_checkpoint": str(gen.ckpt_path(ckpt)), "returncode": rc,
               "log": str(log), **(got or {}), **vel}
        results.append(row)
        shown = (f"{got['envs_up']}/{got['envs_total']} envs up "
                 f"({got.get('steps_pct', float('nan')):.1f}% steps)" if got else
                 f"NO SUMMARY (rc={rc}) -- see {log.name}")
        if vel:
            shown += (f" | vel RMSE lin {vel['vel_rmse_lin']:.3f} m/s "
                      f"yaw {vel['vel_rmse_yaw']:.3f} rad/s")
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {shown}", flush=True)

    if args.dry_run:
        return

    (out_dir / "results.json").write_text(json.dumps(
        {"teacher": TEACHER.__dict__ | {"checkpoint_used": teacher_ckpt},
         "library": args.library,
         "num_envs": args.num_envs, "max_steps": args.max_steps, "seed": args.seed,
         "generators": {m: GENERATORS[m].__dict__ | {"path": str(GENERATORS[m].path)}
                        for m in models},
         "recordings": RECORDINGS, "results": results}, indent=2, default=str))

    labels = [f"{r['generator']}@{r['checkpoint']} {r['config']} K={r['k']} "
              f"s={r['sampling_steps']}" for r in results]
    width = max((len(x) for x in labels), default=20)
    print(f"\n=== {args.library} library, {args.num_envs} envs, seed {args.seed}, "
          f"{args.max_steps} steps ===")
    print(f"  {'cell':<{width}}  {'survival':>9}  {'vx':>6}  {'vy':>6}  {'wz':>6}")
    print(f"  {'':<{width}}  {'':>9}  {'m/s':>6}  {'m/s':>6}  {'rad/s':>6}")
    for label, r in zip(labels, results):
        got = (f"{r['envs_up']}/{r['envs_total']}" if "envs_up" in r
               else f"FAILED rc={r['returncode']}")
        v = [r.get(f"vel_rmse_{c}") for c in VEL_CHANNELS]
        vs = "  ".join(f"{x:6.3f}" if x is not None else "     -" for x in v)
        print(f"  {label:<{width}}  {got:>9}  {vs}")
    n_failed = sum(1 for r in results if "envs_up" not in r)
    if n_failed:
        print(f"\n  !! {n_failed}/{len(results)} cell(s) FAILED -- their logs in {out_dir}")
    print("\n  survival = envs that stayed up for a full 20 s episode")
    print("  vx/vy/wz = RMSE of measured base velocity vs the COMMANDED joystick velocity,")
    print("             over EVERY logged step (so pre-fall flailing counts against a model);")
    print("             the survival-conditioned variant is in results.json as timeout_*")
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
