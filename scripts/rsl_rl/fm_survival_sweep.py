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
        """Absolute checkpoint path."""
        return REPO / "logs" / "rsl_rl" / "g1" / self.run / self.checkpoint


#: The 41/node teacher every generator below was trained against. Its reference_window packs
#: (base_pose, base_vel, joint_pos); a generator predicting a different selection is refused
#: at load by FmTrackingCommand.reference_window_obs, so teacher and generator move together.
TEACHER = Teacher(
    run="2026-08-01_08-49-20_depth_v5_refslim",
    checkpoint="model_12000.pt",
    note="12000 it, body-mode depth teacher, 41/node (451) reference window. Trained on talos.",
)


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
        overrides={"history_from": "clip",
                   "ego_anchor_fraction": "1.0", "ego_anchor_every": "1"},
        isolates="The generator still predicts and the tracker still follows, but the "
                 "context is always correct. closed_loop far below this = the LOOP "
                 "diverges; the two together = the MODEL is the limit.",
    ),
    "clip_control": Config(
        overrides={"reference_source": "clip",
                   "ego_anchor_fraction": "1.0", "ego_anchor_every": "1"},
        isolates="The policy-side CEILING: the tracker follows the real clip and the "
                 "generator is computed but discarded. No FM contribution at all.",
        generator_independent=True,
        k_matters=False,
    ),
    "history_free": Config(
        overrides={"history_from": "generator", "fm_guidance_scale": "0.0"},
        isolates="Guidance w=0 returns the learned null branch, so the history is IGNORED. "
                 "A free no-retrain probe of how much the model's own history is worth: "
                 "above closed_loop means it is actively harmful.",
    ),
}


def _check_assets(gen_names: list[str], ckpts: list[str], library: str,
                  fatal: bool = True) -> None:
    """Fail before launching anything if a named asset is not on disk.

    Every run costs a simulator start-up, so a missing checkpoint should surface now rather
    than after the first rollout. Under ``--dry_run`` this only warns: printing the command
    for a checkpoint still sitting on talos is a legitimate thing to want.

    Args:
        gen_names: Generators about to be swept.
        ckpts: Checkpoint names each will be swept at.
        library: Clip-library key.
        fatal: Raise on a missing asset (``False`` warns and continues).

    Raises:
        SystemExit: With the offending paths, when ``fatal``.
    """
    missing = [f"{TEACHER.run}/{TEACHER.checkpoint} -> {TEACHER.path}"] \
        if not TEACHER.path.is_file() else []
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
                  num_envs: int, max_steps: int, seed: int) -> list[str]:
    """The full ``play.py`` argv for one measurement.

    Args:
        fm_path: The generator checkpoint to load.
        cfg: The named configuration supplying the env overrides.
        library: Key into :data:`LIBRARIES`.
        k: ``fm_update_every`` (the re-plan period, in control steps).
        num_envs / max_steps / seed: Rollout size and determinism.

    Returns:
        argv suitable for :func:`subprocess.run`.
    """
    argv = [
        PYTHON, str(REPO / "scripts" / "rsl_rl" / "play.py"),
        "--task", TASK,
        "--load_run", TEACHER.run, "--checkpoint", str(TEACHER.path),
        "--clip", LIBRARIES[library].path, "--clip_device", "cpu",
        "--num_envs", str(num_envs), "--max_steps", str(max_steps),
        "--seed", str(seed), "--headless",
    ]
    argv += ["--fm_checkpoint", str(fm_path)]
    argv += [f"env.commands.traj_ref.fm_update_every={k}"]
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
    p.add_argument("--library", default="flat", choices=sorted(LIBRARIES),
                   help="Clip library, which also picks the TERRAIN. Default flat, which is "
                        "what every recorded number was measured on.")
    p.add_argument("--checkpoints", default="best",
                   help="Comma-separated checkpoints per generator: 'best', 'ckpt_99' (most "
                        "trained), 'ckpt_40' (mid-training), ... Every named generator is "
                        "swept at every named checkpoint.")
    p.add_argument("--num_envs", type=int, default=50,
                   help="Rollout width. 50 by default: at 24 the metric resolves to ~4%% per "
                        "env, which is too coarse to read a 2-3 env difference.")
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

    _check_assets(models, ckpts, args.library, fatal=not args.dry_run)

    # One cell per (generator, config, K). A configuration that ignores the generator
    # (clip_control) gets exactly ONE cell however many generators were named -- running it
    # per generator would repeat an identical rollout and invite reading it as a per-model
    # number, which it is not.
    cells: list[tuple[str, str, str, int]] = []
    for cname in configs:
        cfg = CONFIGS[cname]
        k_values = ks if cfg.k_matters else ks[:1]
        for k in k_values:
            if cfg.generator_independent:
                # One cell. models[0] @ ckpts[0] is loaded only to satisfy the command term;
                # nothing it predicts reaches the robot.
                cells.append((models[0], ckpts[0], cname, k))
            else:
                cells += [(m, c, cname, k) for m in models for c in ckpts]

    out_dir = pathlib.Path(args.out) if args.out else (
        REPO / "logs" / "fm_survival" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fm-survival] teacher   {TEACHER.run}/{TEACHER.checkpoint}")
    print(f"[fm-survival]           {TEACHER.note}")
    print(f"[fm-survival] library   {args.library}: {LIBRARIES[args.library].note}")
    print(f"[fm-survival] {len(cells)} cell(s), {args.num_envs} envs x {args.max_steps} "
          f"steps, seed {args.seed} -> {out_dir}\n")

    results = []
    for mname, ckpt, cname, k in cells:
        gen = GENERATORS[mname]
        independent = CONFIGS[cname].generator_independent
        tag = f"{cname}_k{k}" if independent else f"{mname}_{ckpt}_{cname}_k{k}"
        argv = build_command(gen.ckpt_path(ckpt), CONFIGS[cname], args.library, k,
                             args.num_envs, args.max_steps, args.seed)
        if args.dry_run:
            print(f"# {tag}\n{' '.join(argv)}\n")
            continue
        log = out_dir / f"{tag}.log"
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
        got = (parse_survival(kept["stats.txt"].read_text(errors="replace"))
               if "stats.txt" in kept else None) or \
              parse_survival(log.read_text(errors="replace"))
        vel = parse_velocity_rmse(kept.get("tracking_stats.csv", pathlib.Path("/nonexistent")))
        row = {"generator": "-" if independent else mname,
               "checkpoint": "-" if independent else ckpt, "config": cname, "k": k,
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
        {"teacher": TEACHER.__dict__, "library": args.library,
         "num_envs": args.num_envs, "max_steps": args.max_steps, "seed": args.seed,
         "generators": {m: GENERATORS[m].__dict__ | {"path": str(GENERATORS[m].path)}
                        for m in models},
         "recordings": RECORDINGS, "results": results}, indent=2, default=str))

    labels = [f"{r['generator']}@{r['checkpoint']} {r['config']} K={r['k']}" for r in results]
    width = max((len(x) for x in labels), default=20)
    print(f"\n=== {args.library} library, {args.num_envs} envs, seed {args.seed}, "
          f"{args.max_steps} steps ===")
    print(f"  {'cell':<{width}}  {'survival':>9}  {'vx':>6}  {'vy':>6}  {'wz':>6}")
    print(f"  {'':<{width}}  {'':>9}  {'m/s':>6}  {'m/s':>6}  {'rad/s':>6}")
    for label, r in zip(labels, results):
        got = (f"{r['envs_up']}/{r['envs_total']}" if "envs_up" in r
               else f"rc={r['returncode']}")
        v = [r.get(f"vel_rmse_{c}") for c in VEL_CHANNELS]
        vs = "  ".join(f"{x:6.3f}" if x is not None else "     -" for x in v)
        print(f"  {label:<{width}}  {got:>9}  {vs}")
    print("\n  survival = envs that stayed up for a full 20 s episode")
    print("  vx/vy/wz = RMSE of measured base velocity vs the COMMANDED joystick velocity,")
    print("             over EVERY logged step (so pre-fall flailing counts against a model);")
    print("             the survival-conditioned variant is in results.json as timeout_*")
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
