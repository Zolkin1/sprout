#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""The MuJoCo terrain eval's CLI: emit the courses, then run each arm through them.

    python evaluate.py --stage scenes                  # write the course XMLs + the manifest
    python evaluate.py --stage run --arm baseline      # one arm
    python evaluate.py --stage all                     # scenes, then every arm in the table
    python report.py <out>/baseline <out>/finetune     # reduce

**The arms are DECLARED in :data:`ARMS`, never defaulted.**  A default path to an artifact is a
thing that rots quietly -- twice in this repo already (``_DEFAULT_RUN_DIR`` in
``dump_metadata_fixtures.py`` and ``G1_DEPLOY_POLICY_DIR`` in ``deploy_cpp/CMakeLists.txt``,
both missing an element and both naming a superseded run, both silent).  So every artifact this
harness loads is named here, hashed into the manifest, and re-hashed on every run.

**Both arms load the SAME generator, by absolute path, and that is the experiment.**  The
baseline tracker's ``exported/`` holds two generators, so ``resolve_fm_onnx`` refuses to pick
one -- by design, since defaulting to the newest would reinstate as a silent default exactly
the ambiguity the export naming removed.  Naming one explicitly settles that, but the reason to
name THIS one is stronger than the resolver: the fine-tune was trained and evaluated against
it, so holding it fixed makes the tracker the only thing that differs between the two tables
and every paired delta a statement about the fine-tune.

Nothing here spawns a subprocess: the arms run in this process, in sequence, sharing nothing
but the courses on disk.  That removes the Isaac harness's sharpest trap -- a flag added to the
parser but not restated by the stage that spawns the child, which reverts to its argparse
default silently and has twice produced a grid that disagreed with its own manifest.  The
manifest still records every resolved value, because "it cannot be dropped" and "it was what
you think" are different claims.

Determinism is the reason there is one trial per cell: ``zero_noise`` integrates the flow from
``x0 = 0`` (what the hardware controller runs), no depth noise is enabled, no domain
randomisation is applied anywhere, and MuJoCo is deterministic -- so a repeat at fixed
``(course, speed)`` is bit-identical and the two arms pair exactly.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_HERE)
for _d in (_HERE, _SIM_DIR, os.path.dirname(_SIM_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import terrains  # noqa: E402

#: ``<repo>/transfer/sim/mjeval`` -> ``<repo>``.  Derived and marker-checked rather than
#: written out: an absolute path pinned to one worktree is the same rot as a counted ``..``.
REPO_ROOT: str = os.path.dirname(os.path.dirname(_SIM_DIR))
if not os.path.isdir(os.path.join(REPO_ROOT, "logs", "rsl_rl", "g1")):
    raise RuntimeError(f"evaluate.py derived REPO_ROOT={REPO_ROOT}, which has no "
                       "logs/rsl_rl/g1; fix the derivation if this module moved.")

#: Where the tracker runs live.
LOG_ROOT: str = os.path.join(REPO_ROOT, "logs", "rsl_rl", "g1")

#: The ONE generator both arms load.  It sits in the FINE-TUNE run's ``exported/`` because that
#: is where it was exported, not because it belongs to that arm -- the baseline loads the same
#: file, by this absolute path, and the manifest records the identical sha256 for both.
GENERATOR_ONNX: str = os.path.join(
    LOG_ROOT, "2026-08-26_23-27-58_ft_lr1e4", "exported",
    "2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff_fm.onnx")


#: Set once from ``--fm_onnx``; empty means use :data:`GENERATOR_ONNX`.
_GENERATOR_OVERRIDE: str = ""


@dataclass(frozen=True)
class Arm:
    """One tracker to sweep, and the generator it is paired with.

    Attributes:
        code: The ``arm`` column's value, and the pairing key the report joins on.
        label: Short name; also the output subdirectory, which ``report.py`` reads as a run.
        tracker_run: Run directory name under :data:`LOG_ROOT`.
        note: One line for the manifest saying what this arm is.
        history_from: What fills the generator's history rows -- ``"generator"`` (its own
            served nodes, the deployed loop) or ``"measured"`` (the robot's achieved state).
            An arm dimension, so the same tracker can appear twice differing only in this;
            the two produce rows of IDENTICAL width, so nothing but the manifest can say
            which ran.
    """

    code: int
    label: str
    tracker_run: str
    note: str
    history_from: str = "generator"

    @property
    def tracker_onnx(self) -> str:
        """Absolute path to this arm's exported tracker."""
        return os.path.join(LOG_ROOT, self.tracker_run, "exported", "tracker.onnx")

    @property
    def fm_onnx(self) -> str:
        """Absolute path to the generator, shared by every arm.

        Reads the module-level override rather than the constant directly, so ``--fm_onnx``
        can name the generator a run actually scored. The arms vary the TRACKER and share one
        generator, which was a constant because there was one; every AWR iteration now
        exports its own, and a run has to be able to say which it used.
        """
        return _GENERATOR_OVERRIDE or GENERATOR_ONNX


#: The two arms.  Same generator, same courses, same commands; the tracker is the variable.
ARMS: tuple[Arm, ...] = (
    Arm(0, "baseline", "2026-08-20_16-36-10_mode11_decor_plan_v1",
        "the deployed plan-anchor tracker -- the pair that drove the robot"),
    Arm(1, "finetune", "2026-08-26_23-27-58_ft_lr1e4",
        "the same tracker fine-tuned on the generator's own window (lr 1e-4)"),
    Arm(2, "ft_v3", "2026-08-27_14-17-38_ft_v3",
        "second fine-tune off the SAME base at the same lr: adds a privileged critic term "
        "(fm_reference_residual, critic group only -- the actor's 9-term observation is "
        "unchanged at 3101) and trains on the narrower nojo_nodist clip library. Exported "
        "with tracking_mode='body', which shapes the CLF reward and not the observation, so "
        "the FM path serves it exactly as it serves the other two"),
    Arm(3, "baseline_meas", "2026-08-20_16-36-10_mode11_decor_plan_v1",
        "the DEPLOYED tracker again, with the generator conditioned on the robot's own "
        "achieved state instead of its own served nodes (base position and linear velocity "
        "excepted -- no robot can measure those, so they still come from the plan). Pairs "
        "with 'baseline' on (course, speed): same tracker, same generator, same courses, "
        "and the history source is the only difference. Read it against "
        "scripts/flow_matching/measured_history_gap.py, which measures the input as 0.54 "
        "sigma off-distribution on the joints and 0.69 on angular velocity -- this arm is "
        "not a clean grounding experiment until a generator is TRAINED on measured rows",
        history_from="measured"),
)


def sha256_prefix(path: str, chars: int = 12) -> str:
    """A short content digest, so the manifest identifies an artifact rather than naming one.

    Args:
        path: File to hash.
        chars: Hex characters to keep.

    Returns:
        The prefix, or ``""`` when the file is absent.
    """
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:chars]


def git_state() -> dict[str, str | bool]:
    """The commit this run was produced at, and whether the tree was dirty.

    Returns:
        ``{"commit", "dirty"}``; the commit is ``""`` outside a repository.
    """
    def _git(*args: str) -> str:
        try:
            return subprocess.run(("git", "-C", REPO_ROOT, *args), capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return ""
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def onnx_metadata(path: str, key: str) -> dict:
    """One artifact's embedded metadata blob, without running or optimising its graph.

    Via onnxruntime with graph optimization DISABLED, exactly as ``stair_probe._read_robot_asset``
    does: ``onnx`` (the protobuf package) is not in ``transfer/requirements.txt``, and a
    metadata read must not pay for a graph rewrite.  Measured at 0.06 s for ``tracker.onnx``
    and 0.21 s for a 27M generator.

    Args:
        path: The ``.onnx`` file.
        key: ``policy_parameters`` for a tracker, ``fm_meta`` for a generator.

    Returns:
        The decoded blob, or ``{}`` when the artifact carries none.
    """
    import onnxruntime as ort  # noqa: PLC0415

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    raw = sess.get_modelmeta().custom_metadata_map.get(key)
    return json.loads(raw) if raw else {}


def resolved_runtime(arm: Arm) -> dict:
    """What the runner WILL resolve, read from the artifacts rather than restated.

    The control period and the re-plan cadence come from the tracker's own metadata (the
    generator's cadence comes from ``tracker.onnx`` and nowhere else), and the generator's
    camera lag from its own trained ``depth_latency_probs`` through the same function the
    runner calls.  Recording them at manifest time makes the manifest checkable against the
    ``episodes.h5`` attributes the runner stamps, which is the only way to catch the two
    coming apart.

    Args:
        arm: The arm to resolve.

    Returns:
        ``{"dt", "fm_update_every", "depth_latency_probs", "fm_depth_lag_steps"}``.
    """
    from deploy.fm_session import default_depth_lag_steps  # noqa: PLC0415

    trk = onnx_metadata(arm.tracker_onnx, "policy_parameters")
    fm = onnx_metadata(arm.fm_onnx, "fm_meta")
    probs = fm.get("depth_latency_probs")
    return {"dt": trk.get("dt"), "fm_update_every": trk.get("fm_update_every"),
            "depth_latency_probs": probs,
            "fm_depth_lag_steps": default_depth_lag_steps(probs)}


def runner_steer_default() -> float:
    """The steering clip's default, read from ``runner`` without importing it eagerly.

    ``--help`` must not pay for mujoco and the deploy runtime, and the constant is one float,
    so it is restated here with the module named as its source rather than imported.

    Returns:
        The default yaw-rate clip (rad/s): the largest ``|wz|`` the training library commands.
    """
    return 0.68


def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    Returns:
        Parser for ``evaluate.py --stage scenes|run|all``.
    """
    p = argparse.ArgumentParser(prog="evaluate.py", description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("scenes", "run", "merge", "all"), default="all")
    p.add_argument("--arm", default="all",
                   help="comma list of labels from the ARMS table, or 'all' "
                        f"({', '.join(a.label for a in ARMS)})")
    p.add_argument("--out", default="",
                   help="run directory; default logs/mjeval/<timestamp>. Pass one explicitly "
                        "to resume -- the timestamp default makes a fresh directory every "
                        "invocation, which nothing can resume into")
    p.add_argument("--redo", action="store_true",
                   help="re-run an arm whose episodes.h5 already exists")
    p.add_argument("--robot_name", default="g1_perceptive")
    p.add_argument("--providers", default="auto",
                   help="onnxruntime providers, comma separated (cpu / cuda / auto)")
    p.add_argument("--fall_height", type=float, default=0.2,
                   help="pelvis clearance (m) above the scene's LOWEST terrain surface")
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--spawn_z_offset", type=float, default=0.03,
                   help="lift the spawn pelvis clear of the floor (the standing config's "
                        "pelvis frame sits ~3 cm below MuJoCo's foot-contact height)")
    p.add_argument("--fm_depth_lag_steps", type=int, default=None,
                   help="override the generator's camera lag; the default DERIVES it from the "
                        "graph's own depth_latency_probs, which is what it trained on")
    p.add_argument("--terrain_ref", default="",
                   help="a terrain_class.py reference .npz; without it the n_plans* columns "
                        "are legitimately zero")
    p.add_argument("--max_courses", type=int, default=0, help="smoke run: first N courses")
    p.add_argument("--courses", default="",
                   help="comma-separated course NAMES to run (e.g. mjeval_box2_h040). "
                        "--max_courses can only take a PREFIX of the catalogue, so it cannot "
                        "select one course by name.")
    p.add_argument("--fm_onnx", default="",
                   help="the GENERATOR to evaluate, overriding the module default. The arms "
                        "vary the TRACKER and share one generator, which was a constant "
                        "because there was one; there are now many (every AWR iteration "
                        "exports its own), and a run must be able to name the one it scored. "
                        "It is hashed into the manifest either way.")
    p.add_argument("--max_speeds", type=int, default=0, help="smoke run: first N speeds")
    p.add_argument("--gaussian_noise", dest="zero_noise", action="store_false", default=True,
                   help="Draw the flow's x0 from a Gaussian instead of integrating from 0. "
                        "OFF by default: x0=0 is what the hardware controller runs and it "
                        "makes a repeat at fixed (course, speed) bit-identical, which is why "
                        "the sweep has one trial per cell. But that draw is NOT a neutral "
                        "sample -- it sits near the conditional's mode and succeeds where a "
                        "third of Gaussian draws fail (fm_variance_probe) -- so a result that "
                        "has to hold for the model rather than for one draw needs this, plus "
                        "several --fm_seed values in separate --out directories.")
    p.add_argument("--fm_seed", type=int, default=0,
                   help="Seed for the generator's noise draws. Only meaningful with "
                        "--gaussian_noise; recorded either way.")
    # OPT-IN, and off by default, because it changes what is being measured rather than how
    # cleanly. See runner.steer_yaw_rate: the stair families were commanded |wz| = 0.0000 over
    # all 3500 clips, so any non-zero yaw there is extrapolation. flat and box do turn.
    # **Two runs with different wall settings MUST NOT overlap in time.**  The scene XMLs are
    # named for the course alone, so a concurrent run with the other setting rewrites the very
    # files this one is loading -- silently, and mid-sweep, producing a table whose rows were
    # collected on two different terrains. `--stage scenes` emits them and the sweep reads them
    # back, so there is no lock to take. Run wall A/Bs SEQUENTIALLY.
    p.add_argument("--no_stair_walls", dest="stair_walls", action="store_false", default=True,
                   help="emit the staircases WITHOUT flanking walls. The walls change what the "
                        "depth cameras see (up to 60%% of the far frame on approach), so a "
                        "grid run with them is not comparable to one run without")
    p.add_argument("--cmd_hold_s", type=float, default=terrains.CMD_HOLD_S,
                   help="seconds of ZERO command before stepping to the commanded speed; 0 "
                        "restores the old instant-full-speed spawn")
    p.add_argument("--steer_kp", type=float, default=0.0,
                   help="P gain (1/s) on heading error toward the goal; 0 disables steering "
                        "and holds the constant (vx, 0, 0) the training clips used")
    p.add_argument("--steer_wz_max", type=float, default=runner_steer_default(),
                   help="clip on the steering yaw rate (rad/s)")
    # Sharding is pure scheduling: the courses are split by global index, `terrain_idx` stays
    # global, and `--stage merge` writes exactly the table a one-process run would have. It is
    # safe because the sweep is deterministic -- 13 runs of one episode at 1-, 4- and 8-way
    # concurrency came back bit-identical (0.000e+00 m on every logged channel). Measured
    # speedup on an RTX 4070: 3.2x at 4 shards, 5.0x at 8 (GPU 62% util, 6.1 of 12.3 GB), so 8
    # is the knee and more would start queueing on the generator rather than the simulator.
    p.add_argument("--shards", type=int, default=1,
                   help="split the catalogue across N processes (run each with --shard i, "
                        "then --stage merge)")
    p.add_argument("--shard", type=int, default=0, help="which shard THIS process runs")
    return p


def stage_scenes(robot_name: str) -> list[str]:
    """Emit every course's scene XML where ``Robot._get_model_data`` looks for it.

    Idempotent and ~milliseconds, so the run stage calls it too: making a run depend on someone
    having remembered a preceding command is the ordering trap this repo keeps paying for.

    Args:
        robot_name: Robot model to ``<include>``.

    Returns:
        The paths written, in catalogue order.
    """
    paths = [terrains.write_scene(c, robot_name) for c in terrains.catalogue()]
    print(f"[INFO] wrote {len(paths)} course scenes to "
          f"{os.path.dirname(paths[0]) if paths else '-'}")
    return paths


def selected_arms(spec: str) -> list[Arm]:
    """Resolve ``--arm`` against the declared table.

    Args:
        spec: ``all`` or a comma list of labels.

    Returns:
        The arms to run, in table order.

    Raises:
        SystemExit: On a label the table does not carry -- a typo must not silently run
            nothing, which reads exactly like a completed sweep.
    """
    if spec.strip().lower() == "all":
        return list(ARMS)
    want = [s.strip() for s in spec.split(",") if s.strip()]
    by_label = {a.label: a for a in ARMS}
    unknown = [w for w in want if w not in by_label]
    if unknown:
        raise SystemExit(f"[ERROR] no such arm(s) {unknown}; the table has "
                         f"{sorted(by_label)}")
    return [by_label[w] for w in want]


def write_manifest(out_dir: str, arms: list[Arm], args: argparse.Namespace) -> dict:
    """Write ``<out>/manifest.json`` and each arm's own, and return the run-level contents.

    Two files, because they answer different questions.  The run-level manifest is the record
    of the whole experiment -- every course, both arms, the git state.  The per-arm one sits
    beside that arm's ``episodes.h5`` because ``report.py`` treats an arm directory as a run
    and reads its ``artifacts`` block to print the inventory; without it the report says the
    artifacts were not recorded and cannot state which pair produced the numbers.

    Args:
        out_dir: The run directory.
        arms: The arms this invocation covers.
        args: Parsed CLI.

    Returns:
        The run-level manifest.
    """
    rows = []
    for arm in arms:
        art = {"tracker": {"name": arm.tracker_onnx, "sha256": sha256_prefix(arm.tracker_onnx)},
               "generator": {"name": arm.fm_onnx, "sha256": sha256_prefix(arm.fm_onnx)}}
        rt = resolved_runtime(arm)
        rows.append({"arm": arm.code, "label": arm.label, "tracker_run": arm.tracker_run,
                     "note": arm.note, "artifacts": art, "runtime": rt,
                     "history_from": arm.history_from,
                     "out_dir": os.path.join(out_dir, arm.label)})
        arm_dir = os.path.join(out_dir, arm.label)
        os.makedirs(arm_dir, exist_ok=True)
        with open(os.path.join(arm_dir, "manifest.json"), "w") as f:
            json.dump({"arm": arm.code, "label": arm.label, "artifacts": art, "runtime": rt,
                       "note": arm.note, "zero_noise": bool(args.zero_noise),
                       "fm_seed": int(args.fm_seed), "domain_randomization": "none",
                       # Two arms can share every artifact hash above and still be different
                       # experiments; this is the only field that says so.
                       "history_from": arm.history_from},
                      f, indent=2, default=str)

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "robot_name": args.robot_name,
        "zero_noise": bool(args.zero_noise),
        "fm_seed": int(args.fm_seed),
        # Stated rather than implied: every physical DR term, the observation corruption and
        # the depth sensor model are OFF, so a number here is what the pair does on a nominal
        # robot with clean cameras and nothing else.
        "domain_randomization": "none -- no physical DR, no observation noise, no depth noise",
        "reference_mode": "fm closed loop (autoregressive); no clip, no teacher forcing",
        "trials_per_cell": 1,
        "determinism": (
            "zero_noise integrates the flow from x0 = 0 and MuJoCo is deterministic, so a "
            "repeat at fixed (course, speed) is bit-identical and the arms pair exactly"
            if args.zero_noise else
            f"GAUSSIAN x0 at fm_seed={args.fm_seed}: MuJoCo is still deterministic and the "
            "arms still pair exactly at fixed (course, speed), because the seed is shared -- "
            "but a repeat at another seed is a DIFFERENT draw, so one seed is one sample of "
            "the sampler and several --out directories are needed to say anything about the "
            "model rather than about the draw"),
        "fall_height_m": args.fall_height, "fall_tilt_deg": args.fall_tilt_deg,
        "spawn_z_offset_m": args.spawn_z_offset,
        "providers": args.providers,
        "terrain_ref": args.terrain_ref,
        "max_courses": args.max_courses, "max_speeds": args.max_speeds,
        "arms": rows,
        "courses": terrains.manifest_rows(),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[INFO] manifest -> {os.path.join(out_dir, 'manifest.json')}")
    return manifest


def stage_run(out_dir: str, arms: list[Arm], args: argparse.Namespace,
              manifest: dict) -> list[str]:
    """Run each selected arm, skipping any whose table is already complete.

    Args:
        out_dir: The run directory.
        arms: The arms to run.
        args: Parsed CLI.
        manifest: The run-level manifest, whose per-arm blocks are stamped onto each table.

    Returns:
        The episode-table paths, one per arm that ran or was already done.
    """
    import runner  # noqa: PLC0415 -- mujoco + the deploy runtime; --help must not pay for it

    by_label = {row["label"]: row for row in manifest["arms"]}
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip())
    out_paths = []
    for arm in arms:
        arm_dir = os.path.join(out_dir, arm.label)
        table = os.path.join(
            arm_dir,
            "episodes.h5" if args.shards == 1
            else f"episodes_shard{args.shard}of{args.shards}.h5")
        if os.path.isfile(table) and not args.redo:
            print(f"[INFO] {arm.label}: {table} exists; skipping (--redo to re-run)")
            out_paths.append(table)
            continue
        cfg = runner.ArmRun(
            arm=arm.code, label=arm.label,
            tracker_onnx=arm.tracker_onnx, fm_onnx=arm.fm_onnx, out_dir=arm_dir,
            robot_name=args.robot_name, providers=providers,
            fall_height=args.fall_height, fall_tilt_deg=args.fall_tilt_deg,
            spawn_z_offset=args.spawn_z_offset,
            fm_depth_lag_steps=args.fm_depth_lag_steps,
            terrain_ref=args.terrain_ref,
            max_courses=args.max_courses, only_courses=args.courses,
            max_speeds=args.max_speeds,
            shards=args.shards, shard=args.shard,
            steer_kp=args.steer_kp, steer_wz_max=args.steer_wz_max,
            cmd_hold_s=args.cmd_hold_s,
            history_from=arm.history_from,
            zero_noise=args.zero_noise, fm_seed=args.fm_seed,
            attrs={"git": manifest["git"], "arm_note": arm.note,
                   "declared_runtime": by_label[arm.label]["runtime"],
                   "artifacts": by_label[arm.label]["artifacts"],
                   "reference_mode": manifest["reference_mode"],
                   "history_from": arm.history_from},
        )
        out_paths.append(runner.run_arm(cfg))
    return out_paths


def main() -> int:
    """Parse the CLI and run the requested stage.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    if args.fm_onnx:
        if not os.path.exists(args.fm_onnx):
            raise SystemExit(f"--fm_onnx: no such file {args.fm_onnx}")
        global _GENERATOR_OVERRIDE                                  # noqa: PLW0603
        _GENERATOR_OVERRIDE = os.path.abspath(args.fm_onnx)
        print(f"[mjeval] generator OVERRIDDEN: {_GENERATOR_OVERRIDE}")
    out_dir = args.out or os.path.join(
        REPO_ROOT, "logs", "mjeval",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    out_dir = os.path.abspath(out_dir)
    arms = selected_arms(args.arm)
    # Set BEFORE stage_scenes, which is what bakes the choice into the emitted XML, and before
    # anything calls terrains.catalogue() -- the builders read this constant at build time.
    terrains.STAIR_WALLS = bool(args.stair_walls)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] stair walls: {'ON (touching the treads)' if args.stair_walls else 'OFF'}")
    print(f"[INFO] run directory {out_dir}")

    stage_scenes(args.robot_name)
    manifest = write_manifest(out_dir, arms, args)
    if args.stage == "scenes":
        return 0

    if args.stage == "merge":
        # Lazily, for the same reason `stage_run` does it: importing `runner` pulls in mujoco
        # and the whole deploy runtime, and `--help` must not pay for that.
        import runner  # noqa: PLC0415

        paths = [runner.merge_shards(os.path.join(out_dir, a.label)) for a in arms]
        print("[INFO] reduce with:  python report.py "
              + " ".join(os.path.dirname(x) for x in paths))
        return 0

    paths = stage_run(out_dir, arms, args, manifest)
    if args.shards > 1:
        print(f"[INFO] shard {args.shard} of {args.shards} done. Run the others, then:\n"
              f"       python evaluate.py --stage merge --out {out_dir}")
        return 0
    print("[INFO] reduce with:  python report.py "
          + " ".join(os.path.dirname(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
