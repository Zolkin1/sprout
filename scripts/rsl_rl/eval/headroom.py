# Copyright 2026 Zachary Olkin. All rights reserved.

"""State-level best-of-K headroom test for the flow-matching generator.

One question: **at a state the deployed policy actually visits, does the generator's
conditional distribution over the next chunk contain a chunk the tracker can follow?**  If it
does, the gap between the ``clip`` ceiling and the ``fm`` arm is a *selection* problem and the
methods that reweight existing probability mass -- advantage-conditioned fine-tuning, filtered
BC, advantage-weighted flow matching -- have something to work with.  If it does not, no
amount of reweighting will produce a chunk that was never sampled, and the fix is upstream
(perception, data balance, output parameterization).

This is a MEASUREMENT.  Nothing here trains or modifies the generator or the tracker.

Runbook -- the plan stage is the EPISODE harness's, unchanged, so both experiments index the
same clips and the same tiles and their numbers are directly comparable::

    python scripts/rsl_rl/eval/evaluate.py --stage plan --out logs/headroom/v1 \\
        --lib clips/g1_mb_v5_body_mode11_decor.h5 \\
        --load_run 2026-08-20_16-36-10_mode11_decor_plan_v1 \\
        --task G1-tracker-clf-fm-terrain-depth-play \\
        --fm_checkpoint deployed=logs/flow_matching/<run>/best.pt \\
        --families stairs_up,stairs_down,jump_on,jump_off_front --reference_modes fm \\
        --dr_presets none

    python scripts/rsl_rl/eval/headroom.py --stage run --out logs/headroom/v1 \\
        --fm_label deployed --k 16 --commit_mode random --num_envs 255
    python scripts/rsl_rl/eval/headroom.py --stage run --out logs/headroom/v1 \\
        --fm_label deployed --k 16 --commit_mode argmax  --num_envs 255 \\
        --episodes_per_clip 1
    python scripts/rsl_rl/eval/headroom.py --stage report --out logs/headroom/v1

``report`` imports no IsaacLab and no simulator.

**The per-cell subprocess trap does not apply here.**  ``evaluate.py --stage grid`` forwards
flags to a child process and a flag it forgets to restate silently reverts to its argparse
default; this harness runs one arm per invocation in one process, so every flag reaches the
code that reads it.  What it does share is the discipline behind that rule: every resolved
argument is written into ``headroom.json`` so a table can be traced to the flags that made it.
"""

from __future__ import annotations

import argparse
import sys

import _paths  # noqa: F401  - puts scripts/rsl_rl and this dir on sys.path

import headroom_runner


def build_parser() -> argparse.ArgumentParser:
    """The full CLI for every stage."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=("run", "report", "restore_probe", "select"),
                   help="run = roll the branch experiment; report = tables and plots; "
                        "restore_probe = measure what the state copy reproduces and how "
                        "fast the rest amplifies.")
    p.add_argument("--out", required=True,
                   help="A run directory already built by `evaluate.py --stage plan`.")

    g = p.add_argument_group("experiment")
    g.add_argument("--fm_label", default="",
                   help="Which generator from the plan to branch. Required for --stage run.")
    g.add_argument("--k", type=int, default=16,
                   help="Candidates per state. Log-subsampled to K=1,2,4,8,16 in the report, "
                        "so there is no reason to sweep it.")
    g.add_argument("--commit_mode", default="random",
                   choices=("random", "argmax", "random_k"),
                   help="random: the parent runs free and is never restored (statistically "
                        "ordinary deployment). argmax: commit the best candidate every "
                        "chunk -- a deployable oracle. random_k: commit a UNIFORMLY RANDOM "
                        "candidate every chunk -- the control for argmax, with the same "
                        "restore cadence, so the difference between them is the oracle and "
                        "the restore cost cancels.")
    g.add_argument("--window_chunks", type=int, default=headroom_runner.DEFAULT_WINDOW_CHUNKS,
                   help="Scoring window in re-plan chunks (5 x 12 steps x 0.02 s = 1.20 s).")
    g.add_argument("--steer_kp", type=float, default=0.0,
                   help="P gain (1/s) on the heading error toward the goal, overriding the "
                        "clip's commanded wz. 0 = off, which is the deployed behaviour. "
                        "NON-ZERO IS OFF-DISTRIBUTION ON STAIRS: the two stair families are "
                        "commanded |wz| = 0.0 across the whole training library, so a steered "
                        "stair number is a different measurement, not a cleaner one.")
    g.add_argument("--steer_wz_max", type=float, default=0.68,
                   help="Clip on the steering yaw rate (rad/s); 0.68 is the largest the "
                        "training library commands anywhere.")
    g.add_argument("--goal_regions", default="",
                   help="goal_regions.json from build_goal_regions.py. REQUIRED for "
                        "--stage select: without it the only outcome available is a clip "
                        "CURSOR condition, which scores a robot that stood still exactly "
                        "like one that climbed the terrain. Use the SAME file the critic "
                        "was fitted with.")
    g.add_argument("--v_checkpoint", default="",
                   help="A fit_value.py V_*.pt. Required for --scorer v.")
    g.add_argument("--scan_sensor", default="height_scanner",
                   help="Privileged scan sensor, for a priv critic.")
    g.add_argument("--branch_once", action="store_true",
                   help="ORACLE/HEADROOM mode: branch ONCE per episode and roll every "
                        "candidate to termination under a shared continuation draw, so each "
                        "candidate's outcome is what its chunk led to rather than an "
                        "estimate. Bounds any critic: no ranking can beat picking the "
                        "candidate that actually worked.")
    g.add_argument("--no_restore", action="store_true",
                   help="Write nothing to the simulator. At k=0 this is a pure no-op, so any "
                        "change it makes is the restore's own cost.")
    g.add_argument("--no_force_replan", action="store_true",
                   help="Do not force a re-plan at the top of each cycle; let the natural "
                        "cadence do it.")
    g.add_argument("--align_cadence", action="store_true",
                   help="Snap episode_length_buf to the re-plan cadence before forcing one, "
                        "as headroom_runner._roll does. Without it the forced and natural "
                        "re-plans are mis-phased and the generator plans twice per cycle.")
    g.add_argument("--isolate", action="store_true",
                   help="Restore only the carrier, so its K-1 siblings are never stacked on "
                        "its pose. Pairs with --no_selection to test whether the K-env "
                        "arms' deficit is the stacking itself.")
    g.add_argument("--no_selection", action="store_true",
                   help="STRUCTURAL CONTROL: build and restore all K envs but always follow "
                        "candidate 0, so no choice is made. Must reproduce the k=0 baseline; "
                        "a gap belongs to the K-env structure rather than to selection.")
    g.add_argument("--scorer", default="oracle", choices=("oracle", "v"),
                   help="oracle = keep whichever candidate survived, which is the BEST any "
                        "selector could do on simulator noise and therefore an upper bound "
                        "on what the control can manufacture. v = the trained value model.")
    g.add_argument("--max_cycles", type=int, default=100000,
                   help="Hard stop on re-plan cycles; a truncated table says so loudly.")
    g.add_argument("--episodes_per_clip", type=int, default=2,
                   help="Parent episodes per clip. Budget goes to TILES, not to repeats -- "
                        "see the episode harness's note on the tile-clustered bootstrap.")
    g.add_argument("--same_noise", action="store_true",
                   help="CONTROL: serve every candidate candidate-0's draw, so the K branches "
                        "differ only in which env slot they landed in. Its best-of-K rise IS "
                        "the state-restore artifact, and is the number the real arm's rise "
                        "has to beat. Written to its own `__samenoise` cell.")
    g.add_argument("--max_rounds", type=int, default=0,
                   help="Stop after this many rounds (0 = run until every clip's episodes "
                        "are done). A ceiling shows up as censored episodes, not an error.")

    s = p.add_argument_group("scene")
    s.add_argument("--num_envs", type=int, default=255,
                   help="Rounded DOWN to a whole number of (1 parent + K) groups.")
    s.add_argument("--num_rows", type=int, default=1, help="Terrain rows to build.")
    # Default MATCHES evaluate.py. The two harnesses index the same clips and tiles so
    # their numbers are directly comparable, and the fall threshold is the one knob that
    # would silently make them measure different events.
    s.add_argument("--fall_min_height", type=float, default=0.40,
                   help="base_fallen threshold, as evaluate.py.")
    s.add_argument("--seed", type=int, default=42,
                   help="Run seed. Also the master seed for every flow-noise draw, so a "
                        "rerun samples the same K chunks at the same states.")
    s.add_argument("--dr_preset", default="none", choices=("none", "train", "ood"),
                   help="none (DEFAULT and recommended): no DR draw, no observation "
                        "corruption, no depth noiser -- so the K branches of a state differ "
                        "only in the flow noise. Anything else adds per-ENV randomness that "
                        "the parent-row noise floor will absorb and report.")
    s.add_argument("--dr_resample", default="reset", choices=("reset", "startup"),
                   help="Passed through to the shared DR preset application.")
    s.add_argument("--ood_scale", type=float, default=1.5, help="Widening for --dr_preset ood.")
    s.add_argument("--ood_depth_noise", action="store_true", default=True,
                   help="Widen the depth sensor model under ood.")
    s.add_argument("--start_step", type=int, default=-1,
                   help="Pin every episode to this clip step. -1 (default) samples, which "
                        "already reaches mid-feature states: the tracker task leaves "
                        "random_start_before_terrain_start off and caps only at terrain end.")
    s.add_argument("--spawn_z_offset", type=float, default=0.005,
                   help="Spawn lift, matching the episode harness's default.")
    s.add_argument("--clip_device", default="cpu", help="Device to hold the clip library on.")
    s.add_argument("--task", default="",
                   help="Only used if the plan carries no cells to read the task from.")

    r = p.add_argument_group("report")
    r.add_argument("--probe_warmup", type=int, default=120,
                   help="restore_probe: steps to walk before copying, so the robot is in a "
                        "real contact state rather than at its spawn pose.")
    r.add_argument("--enhanced_determinism", action="store_true",
                   help="PhysX enable_enhanced_determinism (slower, more reproducible).")
    r.add_argument("--solver_position_iters", type=int, default=0,
                   help="Raise PhysX min_position_iteration_count (0 = leave the default).")
    r.add_argument("--probe_self_restore", action="store_true",
                   help="Also write the parent's own state back onto the parent, so it "
                        "loses its PhysX warm start exactly as the branches do.")
    r.add_argument("--probe_steps", type=int, default=60,
                   help="restore_probe: steps to track the divergence over.")
    r.add_argument("--cells", default="",
                   help="Comma-separated headroom cell keys; default is all of them.")
    r.add_argument("--boot", type=int, default=2000, help="Bootstrap resamples.")
    r.add_argument("--ci", type=float, default=0.95, help="Interval level.")
    r.add_argument("--no_plots", action="store_true", help="Skip the figures.")
    r.add_argument("--ceiling", default="",
                   help="An EPISODE-harness run directory whose clip arm gives the ceiling "
                        "column of Table 1. best-of-K saturating THERE is the good case; it "
                        "is not 1.0 on any family.")
    r.add_argument("--ceiling_cell", default="clip__none__-",
                   help="Which cell of --ceiling to read. Match the headroom arm's "
                        "dr_preset -- the clip arm moves a point or two between none and "
                        "train.")
    return p


def main() -> int:
    """Dispatch to the requested stage."""
    args = build_parser().parse_args()
    if args.stage in ("run", "restore_probe", "select"):
        if not args.fm_label:
            raise SystemExit(f"--stage {args.stage} needs --fm_label")
        if args.stage == "select":
            if not args.goal_regions:
                raise SystemExit(
                    "--stage select needs --goal_regions. Without it the only outcomes "
                    "available are clip-CURSOR conditions, which cannot tell a robot that "
                    "climbed the terrain from one that stood still while the cursor ran "
                    "out -- and a selector scored on those is maximised by standing still. "
                    "record_rollouts.py refuses for the same reason; use the file the "
                    "critic was fitted with.")
            if args.scorer == "v" and not args.v_checkpoint:
                raise SystemExit("--scorer v needs --v_checkpoint")
            import headroom_select  # noqa: PLC0415

            return headroom_select.stage_select(args)
        return (headroom_runner.stage_run(args) if args.stage == "run"
                else headroom_runner.stage_restore_probe(args))
    import headroom_report  # noqa: PLC0415  - keeps matplotlib off the run stage

    return headroom_report.stage_report(args)


if __name__ == "__main__":
    sys.exit(main())
