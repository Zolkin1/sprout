#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain evaluation harness: plan a stratified clip set, then roll a policy over it.

Three stages, sharing one on-disk manifest so nothing is retyped between them and no two
stages can disagree about what was run -- the discipline ``fm_debug.py`` uses:

* ``--stage plan`` -- IsaacLab-free.  Picks tiles and clips, writes a self-contained subset
  clip library, records each clip's terrain extent, validates every generator against the
  tracker, and writes ``plan.json`` + ``manifest.json`` + a git snapshot.
* ``--stage cell`` -- one grid cell (one ``reference_mode`` x ``dr_preset`` x ``fm_label``).
  The only stage that imports IsaacLab.
* ``--stage grid`` -- runs the cells that are not already complete, each as its own process,
  so an Isaac crash cannot poison the next cell and the GPU is fully released between them.

Run from the repo root::

    PYTHONPATH=$PWD/source/robot_rl:$PWD/source/motion_gen <env_isaaclab py> \\
        scripts/rsl_rl/eval/evaluate.py --stage plan \\
            --lib clips/g1_mb_v5_body_mode11_decor.h5 \\
            --checkpoint logs/rsl_rl/g1/<run>/model_XXXX.pt \\
            --fm_checkpoint gen_a=logs/flow_matching/<run>/best.pt \\
            --out logs/terrain_eval/<name>
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import pathlib
import re
import sys

import _paths  # noqa: F401 - puts scripts/rsl_rl on sys.path for the sibling imports below

import numpy as np

import sampler
import schema
from provenance import dump_params_yaml, git_snapshot, sha256_file  # noqa: E402

#: Default per-cell episode counts.
#:
#: The ``none`` cells were originally going to be a cheap determinism ASSERTION -- with
#: randomization and observation noise off and the clip pinned, the clip arm ought to repeat
#: exactly.  **Measured, it does not**: two envs on the same clip and the same tile, in the
#: first wave of a fresh sim, diverge from the start, and the within-clip relative spread of
#: ``steps_survived`` came out at ~0.3 (median).  It is not tile sharing -- one env per tile
#: gives the same spread -- so it is simulator-level, and outside the eval's control.
#:
#: That makes ``*/none`` the **noise-floor** cell rather than a self-check, and arguably the
#: most useful cell in the grid: it is the resolution below which no difference between any
#: two other cells means anything.  So it gets a real sample, not 4 draws.
#:
#: 25 for the DR cells because robots-per-clip resolves only the within-clip term, which is
#: already small against tile-to-tile spread -- doubling it measured as buying nothing, while
#: adding tiles pays.
#: ``fm_tf`` mirrors ``fm`` exactly.  It is a KEY here rather than a fallback because the
#: lookup is ``robots[f"{mode}/{preset}"]`` with a default of 25: a missing key does not
#: fail, it silently gives the teacher-forced arm a different R from the closed-loop arm it
#: exists to be compared against.  Measured in a ``--quick`` smoke run before this line
#: existed: ``fm/none`` R=8 against ``fm_tf/none`` R=25, because ``--quick`` caps the keys it
#: can see and could not see that one.
DEFAULT_ROBOTS = {
    "clip/none": 25,
    "clip/train": 25,
    "clip/ood": 25,
    "fm/none": 25,
    "fm/train": 25,
    "fm/ood": 25,
    "fm_tf/none": 25,
    "fm_tf/train": 25,
    "fm_tf/ood": 25,
    "fm_meas/none": 25,
    "fm_meas/train": 25,
    "fm_meas/ood": 25,
}

#: Tasks the harness defaults to.  The tracker task is used for BOTH arms so that "success"
#: means the same thing in each (survived the clip without falling); the clip arm runs it with
#: ``reference_source=clip``, as ``clip_ab_sweep.py`` does.
DEFAULT_TASK = "G1-tracker-clf-fm-terrain-depth-play"

#: ``max(REFERENCE_OFFSETS)``.  Read off the env config in the cell stage; this is the plan
#: stage's Isaac-free default, checked against the live value later.
DEFAULT_MAX_OFFSET = 50

#: What can drive the tracker's reference.  A cell DIMENSION rather than a CLI flag, and
#: deliberately so: a mode travels in ``plan.json``, lands in the cell key, is covered by
#: ``plan_hash``, and is looked up by the cell stage from the plan.  A flag would have to be
#: forwarded across the grid -> cell subprocess boundary, which is exactly where ``num_envs``
#: is silently dropped today (``stage_grid`` forwards the GRID invocation's value, so a flag
#: not retyped there reverts to its argparse default with no warning).
#:
#: * ``clip``   -- the recorded clip drives the tracker; the generator is not loaded at all.
#: * ``fm``     -- the generator drives it, conditioned on its OWN last N served nodes.
#:                 The real closed loop, and the only deployable one.
#: * ``fm_tf``  -- the generator drives it, but its history comes from the CLIP
#:                 (``history_from='clip'``): teacher forcing.  It still predicts the future
#:                 and the tracker still follows, but the context is always correct.  Pairs
#:                 with ``fm`` on clip id, and the DIFFERENCE between them is the exposure
#:                 bias -- the measurement that says whether to attack the loop or the model.
#:                 Not deployable: there is no clip on the robot.
#: * ``fm_meas`` -- the generator drives it, but its history is the ROBOT'S OWN achieved
#:                 state (``history_from='measured'``), except the base position and linear
#:                 velocity, which no robot can measure and which come from the served node.
#:                 Between the other two in how grounded the context is: real rather than
#:                 self-generated, but off-distribution, because the model was fitted on
#:                 reference rows and an achieved row is a different kind of quantity.
#:                 ``scripts/flow_matching/measured_history_gap.py`` measures by how much
#:                 (0.03 sigma on base orientation, 0.54 on joints, 0.69 on angular
#:                 velocity), so read this arm against that rather than as a clean
#:                 grounding experiment. Deployable in principle, unlike ``fm_tf``.
REFERENCE_MODES = ("clip", "fm", "fm_tf", "fm_meas")

#: Modes that need a generator loaded.
FM_MODES = ("fm", "fm_tf", "fm_meas")


def build_parser() -> argparse.ArgumentParser:
    """The single CLI for every stage.

    ``allow_abbrev=False`` is not optional: with abbreviation on, ``--clip`` binds to
    ``--clip_device`` and ``--cell`` to ``--cells``, both silently.
    """
    p = argparse.ArgumentParser(
        prog="evaluate.py", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=("plan", "cell", "grid", "all"))
    p.add_argument("--out", default=None,
                   help="run directory (default logs/terrain_eval/<UTC timestamp>)")

    g = p.add_argument_group("plan: clip selection")
    g.add_argument("--lib", default=None, help="source clip library .h5 (required for plan)")
    g.add_argument("--tiles_per_family", type=int, default=0,
                   help="0 = every tile the family has (the default; tile count is what "
                        "sets the interval width, not robots per clip)")
    g.add_argument("--clips_per_tile", type=int, default=4,
                   help="clips per tile. The tiled families are a per-tile SWEEP over "
                        "commanded speed, and the pick is endpoint-inclusive, so 2 samples "
                        "ONLY the slowest and fastest clip of every tile -- which is what "
                        "every plan before 2026-09-04 did. 4 gives two interior points.")
    g.add_argument("--flat_tiles", type=int, default=12,
                   help="flat COLUMNS to build; 12 covers all 12 flat_tile_decor variants, "
                        "and the decor is in the depth cameras' target list")
    g.add_argument("--flat_clips", type=int, default=0,
                   help="0 = flat_tiles * clips_per_tile")
    g.add_argument("--families", default="", help="comma list; empty = all")
    g.add_argument("--exclude_families", default="",
                   help="comma list of terrain families to drop (flat is controlled by "
                        "--flat_tiles, not here)")
    g.add_argument("--tile_pick", default="spread", choices=sampler.TILE_PICKS)
    g.add_argument("--clip_pick", default="speed", choices=sampler.CLIP_PICKS,
                   help="'speed' spreads a tile's clips over their COMMANDED SPEED; "
                        "'spread' spreads over the library's write order (what plans before "
                        "2026-09-04 used, kept only to reproduce one)")
    g.add_argument("--allow_degenerate_speeds", action="store_true",
                   help="accept a selection covering fewer than "
                        f"{sampler.MIN_SAMPLED_SPEED_LEVELS} commanded-speed levels in a "
                        "family the library sweeps. Refused by default: the tables it "
                        "produces look normal and their per-family means are averages over "
                        "the extremes rather than rates at any speed.")
    g.add_argument("--min_clip_len", type=int, default=0,
                   help="0 = max(REFERENCE_OFFSETS) + 2, the shortest clip with an "
                        "unsaturated observation window")
    g.add_argument("--plan_seed", type=int, default=20260820)
    g.add_argument("--subset_out", default=None, help="default <out>/eval_clips.h5")
    g.add_argument("--quick", action="store_true",
                   help="smoke preset: 3 tiles/family, 1 clip/tile, 2 flat tiles, R=8")

    g = p.add_argument_group("plan: what is evaluated")
    g.add_argument("--task", default=DEFAULT_TASK)
    g.add_argument("--clip_task", default="", help="task for the clip arm; empty = --task")
    g.add_argument("--checkpoint", default=None, help="tracker policy .pt")
    g.add_argument("--load_run", default=None,
                   help="run directory under logs/rsl_rl/<experiment>; newest model_*.pt")
    g.add_argument("--experiment_name", default="g1")
    g.add_argument("--fm_checkpoint", action="append", default=[],
                   help="LABEL=PATH (repeatable); a bare PATH labels itself from its run dir")
    g.add_argument("--reference_modes", default="clip,fm",
                   help=f"comma list from {','.join(REFERENCE_MODES)}. 'fm_tf' is the "
                        "generator TEACHER-FORCED (history from the clip); see "
                        "REFERENCE_MODES for why it is a mode and not a flag")
    g.add_argument("--dr_presets", default="none,train",
                   help="comma list from none,train,ood. 'ood' is NOT a default: it is "
                        "measured to move this generator less than the interval it is "
                        "quoted with (fm/none 0.495 -> fm/train 0.468), so it costs a "
                        "third of the grid to re-confirm that DR is not the limit. Ask "
                        "for it when robustness is the question.")
    g.add_argument("--start_step", type=int, default=0,
                   help="clip step every episode starts on. 0 (default) plays each clip from "
                        "its beginning, which is what makes terrain_start_step / "
                        "terrain_end_step describe the episode and keeps cells paired at the "
                        "same difficulty. A POSITIVE value is needed to exercise a generator "
                        "history seeded from the clip (it needs N preceding steps to exist). "
                        "-1 draws a random start per episode: that exercises gait-phase "
                        "diversity, but an episode can then begin PAST terrain_start, which "
                        "biases 'terrain done' and 'terrain_progress', and the draws differ "
                        "between cells so arms no longer pair at equal difficulty. Success is "
                        "termination-derived and stays meaningful either way.")
    g.add_argument("--robots_per_clip", default="",
                   help="comma list of KEY=N, e.g. 'clip/train=50,fm/ood=10'; keys not "
                        "given fall back to the built-in defaults")

    g = p.add_argument_group("env")
    g.add_argument("--fm_seed", type=int, default=-1,
                   help="Name the generator's initial-noise draw, so it can be varied while "
                        "EVERYTHING else is held fixed. -1 = off (global RNG). --seed also "
                        "moves DR, spawn and observation noise, so it cannot isolate the "
                        "generator; this can. Keyed by (fm_seed, within-episode re-plan "
                        "index) and NOT by env, so several robots on one clip share a noise "
                        "sequence and their spread is the simulator floor.")
    g.add_argument("--steer_kp", type=float, default=0.0,
                   help="P gain (1/s) on heading error toward the goal, overriding the clip's "
                        "commanded wz. 0 = off (the deployed behaviour). NON-ZERO IS "
                        "OFF-DISTRIBUTION ON STAIRS -- the training library commands |wz| = "
                        "0.0 on both stair families -- so a steered stair number is a "
                        "different measurement, not a cleaner one. Same law and default clip "
                        "as mjeval.runner.steer_yaw_rate and headroom.py --steer_kp.")
    g.add_argument("--steer_wz_max", type=float, default=0.68,
                   help="Clip on the steering yaw rate (rad/s).")
    g.add_argument("--goal_regions", default="",
                   help="goal_regions.json from build_goal_regions.py. Turns on the SPATIAL "
                        "success columns (reached_goal / min_dist_to_goal_m / the three "
                        "terminal offsets). WITHOUT it every outcome signal in the table is a "
                        "clip-CURSOR condition -- `clip_end` and every terrain_* column "
                        "compare step indices -- and none of them can tell a robot that "
                        "climbed the staircase from one that stood still while the cursor ran "
                        "out. Empty = off, and report.py then refuses to derive spatial "
                        "success rather than silently falling back to the temporal rule.")
    g.add_argument("--tracking_spec", default="",
                   help="tracking_spec.json from measure_tracking_error.py --stage report. "
                        "Turns on the STRICT tracking checks -- lateral / height / tilt "
                        "against the SERVED reference node -- which are LATCHED, never "
                        "terminating, so `report.py` derives success_strict and the loose "
                        "success from the same run. Empty = the BOOTSTRAP placeholders, "
                        "which are fine for looking and not for publishing.")
    g.add_argument("--fall_min_height", type=float, default=0.40,
                   help="base_fallen's world-z threshold, m. 0.40 rather than the historical "
                        "0.50: the deepest jump_off_front clip's own pelvis reaches 0.525 m "
                        "and a well-tracking robot legitimately undershoots its reference by "
                        "up to 60 mm, so at 0.50 a CORRECTLY tracking robot is terminated and "
                        "any critic fitted to those episodes learns the threshold rather than "
                        "the fall. Numbers at 0.40 are not comparable with the published "
                        "0.50 ones; re-run both arms.")
    g.add_argument("--terrain_ref", default="",
                   help="terrain_class.py reference .npz; classifies every FM re-plan as "
                        "flat/jump/stairs and counts them per episode. Ignored on the clip "
                        "arm. Empty = off.")
    g.add_argument("--dump_flagged", type=int, default=0, metavar="N",
                   help="Keep N re-plans per (true class, predicted class) bucket, with the "
                        "depth row each was drawn from, to <out>/<cell>/flagged_plans.npz. "
                        "Needs --terrain_ref. The bucket key collects a misread and its own "
                        "control together -- stairs->jump beside stairs->stairs -- because a "
                        "folder of flagged images alone shows whatever the reader expects. "
                        "0 = off.")
    g.add_argument("--num_envs", type=int, default=256)
    g.add_argument("--num_rows", type=int, default=1)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--spawn_z_offset", type=float, default=0.005,
                   help="must match training; survival is only comparable at equal values")
    g.add_argument("--clip_device", default="", help="'' = auto (see provenance.auto_clip_device)")
    g.add_argument("--max_steps", type=int, default=0, help="0 = derived ceiling")

    g = p.add_argument_group("domain randomization")
    g.add_argument("--ood_scale", type=float, default=1.5)
    g.add_argument("--ood_depth_noise", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--dr_resample", default="reset", choices=("reset", "startup"))
    g.add_argument("--allow_dr_drift", action="store_true",
                   help="downgrade the train-vs-trained DR comparison to a warning")

    g = p.add_argument_group("cell / grid")
    g.add_argument("--cell", default=None, help="one cell key (internal to --stage grid)")
    g.add_argument("--cells", default="", help="comma list of cell keys; empty = all")
    g.add_argument("--force", action="store_true", help="re-run cells already complete")
    g.add_argument("--dry_run", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Plan stage.
# ---------------------------------------------------------------------------
def resolve_checkpoint(args) -> str:
    """Locate the tracker checkpoint without importing IsaacLab.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Absolute path to the checkpoint.

    Raises:
        SystemExit: If neither ``--checkpoint`` nor ``--load_run`` resolves to a file.
    """
    if args.checkpoint:
        path = pathlib.Path(args.checkpoint).resolve()
        if not path.is_file():
            raise SystemExit(f"--checkpoint {path} does not exist")
        return str(path)
    if not args.load_run:
        raise SystemExit("plan needs --checkpoint or --load_run")
    root = pathlib.Path("logs") / "rsl_rl" / args.experiment_name / args.load_run
    # The glob is wider than the key can parse: a run directory legitimately holds
    # `model_19999_padded.pt` (pad_critic_for_finetune.py writes one), and keying on
    # `model_(\d+)\.pt$` returned None for it and died with a bare AttributeError pointing
    # at a lambda. FILTER on the same pattern the key reads, so an unparsable name is simply
    # not a candidate -- picking the highest iteration is the whole job, and a name that does
    # not state an iteration cannot be the highest one.
    numbered = []
    for path in glob.glob(str(root / "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt$", path)
        if m is not None:
            numbered.append((int(m.group(1)), path))
    if not numbered:
        extras = sorted(os.path.basename(p) for p in glob.glob(str(root / "model_*.pt")))
        raise SystemExit(
            f"no model_<iteration>.pt under {root}"
            + (f" (found {extras}, none of which states an iteration; pass --checkpoint)"
               if extras else ""))
    return str(pathlib.Path(max(numbered)[1]).resolve())


def fm_contract(path: str) -> dict:
    """Read a generator checkpoint's interface, using torch alone.

    Deliberately not via ``robot_rl.flow_matching.online``: importing anything under
    ``robot_rl`` runs ``from .tasks import *``, which pulls in IsaacLab and would make the
    plan stage need a simulator install.

    Args:
        path: The ``.pt`` checkpoint.

    Returns:
        The fields the eval has to agree on, plus the checkpoint's sha256.

    Raises:
        SystemExit: If the file is unreadable or carries no config.
    """
    import torch  # noqa: PLC0415

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a bad path here should name itself, not trace
        raise SystemExit(f"could not read FM checkpoint {path}: {exc}") from exc
    cfg = blob.get("config")
    if hasattr(cfg, "__dict__"):
        cfg = vars(cfg)
    if not isinstance(cfg, dict):
        raise SystemExit(f"FM checkpoint {path} has no 'config' dict; cannot check it")
    keep = ("extero_source", "extero_encoder", "extero_channels", "extero_sensor_count",
            "extero_sensor_dim", "extero_image_hw", "extero_clip", "extero_norm_mode",
            "future_len", "num_history_states", "cmd_dim", "attention_mode",
            "target_source", "reference_blocks", "num_sampling_steps", "dt")
    out = {k: cfg.get(k) for k in keep}
    out["path"] = str(pathlib.Path(path).resolve())
    out["sha256"] = sha256_file(path)
    out["epoch"] = blob.get("epoch")
    out["num_joints"] = blob.get("num_joints")
    return out


def parse_fm_checkpoints(specs: list[str]) -> dict[str, dict]:
    """Parse ``LABEL=PATH`` (or bare ``PATH``) generator specs into contracts.

    A bare path labels itself from its run directory, so
    ``logs/flow_matching/2026-08-17_.../best.pt`` becomes ``2026-08-17_...``.

    Raises:
        SystemExit: On a duplicate label.
    """
    out: dict[str, dict] = {}
    for spec in specs:
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                label, path = item.split("=", 1)
            else:
                path = item
                label = pathlib.Path(item).parent.name or pathlib.Path(item).stem
            label = label.strip()
            if label in out:
                raise SystemExit(f"duplicate --fm_checkpoint label {label!r}")
            out[label] = fm_contract(path.strip())
    return out


def check_generators(contracts: dict[str, dict], *, max_offset: int,
                     fm_update_every: int) -> list[str]:
    """Refuse a generator that cannot drive this tracker, up front.

    Every one of these is otherwise discovered lazily -- ``_check_horizon`` fires on the
    first look-ahead, the sensor mismatch at the first observation -- i.e. minutes into a
    cell rather than seconds into the plan.

    Args:
        contracts: From :func:`parse_fm_checkpoints`.
        max_offset: ``max(REFERENCE_OFFSETS)``.
        fm_update_every: The re-plan period the cells will run at.

    Returns:
        Human-readable notes worth printing (not failures).

    Raises:
        SystemExit: Listing every problem found, so one pass fixes them all.
    """
    problems: list[str] = []
    notes: list[str] = []
    sources = {label: c.get("extero_source") for label, c in contracts.items()}
    if len(set(sources.values())) > 1:
        problems.append(
            f"generators disagree on exteroception source: {sources}. One physical camera "
            "pair cannot serve a depth generator and a height-scan generator in one run."
        )
    for label, c in contracts.items():
        need = int(max_offset) + int(fm_update_every)
        have = c.get("future_len")
        if have is None or int(have) < need:
            problems.append(
                f"{label}: future_len={have} but the tracker needs "
                f"max(REFERENCE_OFFSETS)+fm_update_every = {max_offset}+{fm_update_every} "
                f"= {need}. Nothing clamps; the reference window's tail would freeze."
            )
        if not c.get("cmd_dim"):
            problems.append(
                f"{label}: cmd_dim={c.get('cmd_dim')}; FmTrackingCommand needs a "
                "command-conditioned generator (cmd_dim > 0)."
            )
        if c.get("extero_clip") is None:
            if str(c.get("extero_source")) == "depth":
                # ``load_fm_context`` refuses this at env build, because the depth band's
                # per-source default moved (+-2.0 -> +-5.575) after some checkpoints were
                # written, so a null is ambiguous rather than merely absent. Refusing here
                # covers every cell at once instead of failing one env build at a time.
                problems.append(
                    f"{label}: extero_clip is null on a DEPTH checkpoint, so its band is "
                    "ambiguous (the per-source default moved from +-2.0 to +-5.575 after "
                    "this was written) and the generator will be refused at load. Use a "
                    "copy with the band stamped -- e.g. 'best_banded.pt' beside it -- or "
                    "re-export from a run that stamps its own."
                )
            else:
                notes.append(
                    f"{label}: extero_clip is null, so the band will be RE-RESOLVED from the "
                    "current per-source default rather than from what it trained under."
                )
    if problems:
        raise SystemExit("generator contract check failed:\n  - " + "\n  - ".join(problems))
    return notes


def stage_plan(args) -> dict:
    """Select the clip set, build the subset library, and write the run manifest.

    Returns:
        The plan dict (also written to ``<out>/plan.json``).
    """
    if not args.lib:
        raise SystemExit("--stage plan needs --lib <clip library .h5>")
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles_per_family = None if args.tiles_per_family <= 0 else args.tiles_per_family
    clips_per_tile = args.clips_per_tile
    flat_tiles = args.flat_tiles
    if args.quick:
        tiles_per_family, clips_per_tile, flat_tiles = 3, 1, 2
    min_clip_len = args.min_clip_len or (DEFAULT_MAX_OFFSET + 2)

    print(f"[plan] reading {args.lib}")
    index = sampler.read_clip_index(args.lib)
    families = tuple(f for f in args.families.split(",") if f.strip()) or None
    clips, columns = sampler.select_clips(
        index, tiles_per_family=tiles_per_family, clips_per_tile=clips_per_tile,
        flat_tiles=flat_tiles,
        flat_clips=args.flat_clips or None, min_clip_len=min_clip_len, families=families,
        exclude_families=tuple(f for f in args.exclude_families.split(",") if f.strip()),
        tile_pick=args.tile_pick, clip_pick=args.clip_pick, seed=args.plan_seed)
    print(f"[plan] selected {len(clips)} clips over {len(columns)} tile columns "
          f"({len(columns) - flat_tiles} terrain + {flat_tiles} flat)")

    # The commanded-speed distribution is PRINTED and then CHECKED. Printed because a
    # selection's speed coverage was invisible in the recorded plan for every run to date,
    # and the one number that was visible (a family median) sat in a gap no clip occupied.
    speed_summary = _speed_summary(index, clips)
    for line in _format_speed_summary(speed_summary):
        print(f"[plan] {line}")
    complaints = sampler.check_speed_coverage(index, clips)
    if complaints and clips_per_tile > 1:
        for c in complaints:
            print(f"[plan] SPEED COVERAGE: {c}")
        if not args.allow_degenerate_speeds:
            raise SystemExit(
                "refusing to write a plan whose per-family commanded-speed coverage is "
                "degenerate (see above). Raise --clips_per_tile or pass "
                "--allow_degenerate_speeds."
            )
        print("[plan] --allow_degenerate_speeds: accepted anyway")

    subset_path = pathlib.Path(args.subset_out or (out_dir / "eval_clips.h5")).resolve()
    _build_subset(args.lib, subset_path, clips, flat_tiles)
    spans = _terrain_spans(str(subset_path), max_offset=DEFAULT_MAX_OFFSET)

    contracts = parse_fm_checkpoints(args.fm_checkpoint)
    notes = check_generators(contracts, max_offset=DEFAULT_MAX_OFFSET, fm_update_every=12) \
        if contracts else []
    for n in notes:
        print(f"[plan] NOTE {n}")

    robots = dict(DEFAULT_ROBOTS)
    for item in (x.strip() for x in args.robots_per_clip.split(",") if x.strip()):
        key, _, val = item.partition("=")
        robots[key.strip()] = int(val)
    if args.quick:
        robots = {k: min(v, 8) for k, v in robots.items()}

    cells = _cell_list(args, contracts, robots)
    plan = {
        "schema_version": schema.SCHEMA_VERSION,
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source_library": {"path": str(pathlib.Path(args.lib).resolve()),
                           "sha256": sha256_file(args.lib),
                           "num_clips": index.num_clips,
                           "counts_flat": index.num_flat_tiles_source},
        "subset_library": {"path": str(subset_path), "sha256": sha256_file(str(subset_path)),
                           "num_clips": len(clips), "flat_tiles": flat_tiles},
        "cmd_speed_by_family": speed_summary,
        "selection": {"tiles_per_family": tiles_per_family, "clips_per_tile": clips_per_tile,
                      "flat_tiles": flat_tiles, "flat_clips": args.flat_clips or None,
                      "min_clip_len": min_clip_len, "families": list(families or []),
                      "exclude_families": [f for f in args.exclude_families.split(",")
                                           if f.strip()],
                      "tile_pick": args.tile_pick, "clip_pick": args.clip_pick,
                      "allow_degenerate_speeds": bool(args.allow_degenerate_speeds),
                      "seed": args.plan_seed},
        "tile_columns": [int(c) for c in columns],
        "dt": index.dt,
        "max_offset": DEFAULT_MAX_OFFSET,
        "clips": [
            {"clip_id": i, "source_clip_id": c.source_clip_id, "terrain_id": c.terrain_id,
             "family": c.family, "column": c.column, "clip_len": c.clip_len,
             "terrain_start_step": int(spans["start"][i]),
             "terrain_end_step": int(spans["end"][i]),
             "last_usable_step": int(spans["last_usable"][i]),
             "has_terrain_span": bool(spans["has_span"][i]),
             "terrain_grace_steps": int(spans["grace"][i]),
             "completion_informative": bool(spans["informative"][i])}
            for i, c in enumerate(clips)
        ],
        "generators": contracts,
        "cells": cells,
    }
    plan["clip_plan_hash"] = _clip_plan_hash(plan)
    plan["plan_hash"] = _plan_hash(plan)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, default=str))
    _write_manifest(args, out_dir, plan)
    _report_plan(plan)
    return plan


def _speed_summary(index, clips: list) -> dict[str, dict]:
    """Per-family commanded-speed coverage of a selection, for the plan record.

    Recorded because it was previously unrecoverable from a finished run: nothing in
    ``plan.json`` said which speeds a family had been sampled at, so a per-family rate could
    not be read against the speeds that produced it.  ``flat`` is reported with a null
    ``levels`` -- its command varies within the clip, so its first-frame speed is not a
    speed (see :attr:`sampler.ClipIndex.cmd_speed`).

    Args:
        index: The source library's :class:`sampler.ClipIndex`.
        clips: The selected clips.

    Returns:
        ``family -> {"levels", "min", "max", "values", "available_levels"}``; ``levels`` is
        ``None`` for flat.
    """
    import numpy as np  # noqa: PLC0415 - matches this module's lazy-numpy convention

    if index.cmd_speed is None:
        return {}
    fam_of_tile = index.family_of_tile
    by_fam: dict[str, list[int]] = {}
    for sc in clips:
        by_fam.setdefault(sc.family, []).append(int(sc.source_clip_id))
    out: dict[str, dict] = {}
    for fam in sorted(by_fam):
        got = index.cmd_speed[np.asarray(by_fam[fam], dtype=np.int64)]
        if fam == "flat":
            out[fam] = {"levels": None, "min": None, "max": None, "values": None,
                        "available_levels": None, "n_clips": len(by_fam[fam])}
            continue
        tiles = [t for t, f in fam_of_tile.items() if f == fam]
        avail = index.cmd_speed[np.isin(index.terrain_id, tiles)]
        vals = sorted(float(v) for v in np.unique(np.round(got, 2)))
        out[fam] = {
            "levels": sampler.speed_levels(got),
            "min": float(got.min()), "max": float(got.max()),
            "values": vals if len(vals) <= 12 else None,
            "available_levels": sampler.speed_levels(avail),
            "n_clips": len(by_fam[fam]),
        }
    return out


def _format_speed_summary(summary: dict[str, dict]) -> list[str]:
    """One printable line per family from :func:`_speed_summary`.

    Args:
        summary: The mapping :func:`_speed_summary` returned.

    Returns:
        Lines to print, empty when the library carried no commanded speeds.
    """
    lines: list[str] = []
    for fam, d in summary.items():
        if d["levels"] is None:
            lines.append(f"speeds {fam:16s} n={d['n_clips']:3d}  (time-varying command; "
                         "not speed-stratified)")
            continue
        vals = (", ".join(f"{v:.2f}" for v in d["values"]) if d["values"]
                else f"{d['min']:.2f}-{d['max']:.2f}")
        lines.append(f"speeds {fam:16s} n={d['n_clips']:3d}  "
                     f"{d['levels']:2d}/{d['available_levels']:2d} levels  [{vals}] m/s")
    return lines


def _build_subset(src: str, dst: pathlib.Path, clips: list, flat_tiles: int) -> None:
    """Write the subset library and repair its flat TILE count.

    ``write_clip_subset`` cannot know the intended tile count, so it records the kept flat
    CLIP count; the importer reads that field as a tile count.  Left alone, 24 flat clips
    would build 24 flat tiles instead of 12 -- and an all-flat library would ask for one
    large tile per clip.
    """
    from motion_gen.clips.clip_io import set_flat_tile_count, write_clip_subset  # noqa: PLC0415

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[plan] writing subset -> {dst}")
    write_clip_subset(src, str(dst), [c.source_clip_id for c in clips],
                      compression="lzf")
    set_flat_tile_count(str(dst), max(1, int(flat_tiles)))
    size_mb = dst.stat().st_size / 1e6
    print(f"[plan] subset: {len(clips)} clips, {flat_tiles} flat tiles, {size_mb:.0f} MB")


def _terrain_spans(subset_path: str, *, max_offset: int) -> dict[str, np.ndarray]:
    """Per-clip terrain extent, read through ``MotionClipManager``.

    Reuses the manager rather than re-deriving the span here, because its two documented
    traps -- the ``255 = unset`` sentinel in the raw per-step mask, and the repeat-last
    padding that makes an unmasked reverse scan run to the end of the array -- are exactly
    what a reimplementation gets wrong.
    """
    from motion_gen.clips.clip_io import load_clip_library  # noqa: PLC0415
    from motion_gen.clips.clip_manager import MotionClipManager  # noqa: PLC0415

    lib = load_clip_library(subset_path, device="cpu")
    mgr = MotionClipManager(lib)
    start = mgr.terrain_start_step.numpy().astype(np.int64)
    end = mgr.terrain_end_step.numpy().astype(np.int64)
    clip_len = mgr.clip_len.numpy().astype(np.int64)
    last_usable = np.array([schema.last_usable_step(int(n), max_offset) for n in clip_len])
    cols = schema.terrain_columns(
        last_usable, terrain_start_step=start, terrain_end_step=end,
        last_usable=last_usable, dt=float(mgr.dt))
    return {
        "start": start, "end": end, "clip_len": clip_len, "last_usable": last_usable,
        "has_span": cols["has_terrain_span"].astype(bool),
        "grace": cols["terrain_grace_steps"],
        "informative": schema.completion_is_informative(end, last_usable),
    }


def _cell_list(args, contracts: dict[str, dict], robots: dict[str, int]) -> list[dict]:
    """Expand the requested modes/presets/generators into cells.

    ``fm_checkpoint`` is a cell DIMENSION rather than a per-run constant, so several
    generators can be compared against one tracker with the clip arm run ONCE and shared.
    Every cell is then paired on clip id by construction, and the grid is ``P + P*G`` cells
    instead of ``P*(1+G)``.
    """
    modes = [m.strip() for m in args.reference_modes.split(",") if m.strip()]
    presets = [d.strip() for d in args.dr_presets.split(",") if d.strip()]
    unknown = [m for m in modes if m not in REFERENCE_MODES]
    if unknown:
        # Refused rather than ignored: an unknown mode reaches _apply_env_overrides, fails
        # every `== "clip"` test, and would run as a plain closed-loop FM cell under a name
        # claiming otherwise -- a wrong number wearing the right label.
        raise SystemExit(
            f"unknown reference_mode(s) {unknown}; choose from {list(REFERENCE_MODES)}")
    if any(m in FM_MODES for m in modes) and not contracts:
        raise SystemExit(
            f"reference_modes includes {[m for m in modes if m in FM_MODES]} but no "
            "--fm_checkpoint was given. Pass at least one generator, or use "
            "--reference_modes clip."
        )
    cells: list[dict] = []
    for preset in presets:
        for mode in modes:
            labels = list(contracts) if mode in FM_MODES else ["-"]
            for label in labels:
                key = schema.cell_key(mode, preset, label)
                r = robots.get(f"{mode}/{preset}", robots.get(mode, 25))
                cells.append({
                    "key": key, "reference_mode": mode, "dr_preset": preset,
                    "fm_label": label, "robots_per_clip": int(r),
                    "task": (args.clip_task or args.task) if mode == "clip" else args.task,
                    "fm_checkpoint": contracts[label]["path"] if label != "-" else "",
                })
    return cells


#: Plan fields that say nothing about WHAT is being evaluated: a timestamp, and the output
#: path of the subset library (which is per run directory by construction).  Hashing them
#: makes two identical plans look different, which is precisely what the cross-run alignment
#: check must not do -- measured: two runs over the same 264 clips, same columns, same
#: selection and same source library still produced different hashes and would have been
#: refused as "different clip plans".
_VOLATILE_PLAN_FIELDS = ("plan_hash", "clip_plan_hash", "created", "subset_library")


def _clip_plan_hash(plan: dict) -> str:
    """Hash of the CLIP PLAN alone: which clips, on which columns, from which library.

    This is the identity two runs must share for their cells to be paired on clip id. It
    deliberately excludes the policy, the generators, the per-cell episode counts and the
    run's own paths, because none of those change what ``clip_id`` MEANS.
    """
    import hashlib  # noqa: PLC0415

    body = {
        "source_sha256": plan["source_library"]["sha256"],
        "selection": plan["selection"],
        "tile_columns": plan["tile_columns"],
        "dt": plan["dt"],
        "max_offset": plan["max_offset"],
        "clips": [[c["clip_id"], c["source_clip_id"], c["terrain_id"], c["column"],
                   c["clip_len"], c["terrain_start_step"], c["terrain_end_step"]]
                  for c in plan["clips"]],
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _plan_hash(plan: dict) -> str:
    """Content hash of the whole plan, for RESUME.

    Covers the per-cell episode counts and the generators, so changing either invalidates a
    completed cell. Excludes :data:`_VOLATILE_PLAN_FIELDS` so re-planning the same thing in
    the same directory does not force a re-run.
    """
    import hashlib  # noqa: PLC0415

    body = {k: v for k, v in plan.items() if k not in _VOLATILE_PLAN_FIELDS}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _write_manifest(args, out_dir: pathlib.Path, plan: dict) -> dict:
    """Record everything needed to reproduce and to compare this run.

    Includes the trained run's own ``params/`` (its env config hash and the conventions it
    was trained under) and BOTH tracking modes: the tracker task pins ``joint`` while the
    current teachers train under ``body``, which is legitimate -- the actor's observation
    groups are identical -- but it changes what the reference-tracking columns measure, so
    the two are recorded separately rather than assumed equal.
    """
    ckpt = resolve_checkpoint(args)
    log_dir = pathlib.Path(ckpt).parent
    trained_env = log_dir / "params" / "env.yaml"
    conventions = log_dir / "params" / "conventions.yaml"
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "schema_minor": schema.SCHEMA_MINOR,
        "created": plan["created"],
        "cli": {"argv": sys.argv, "resolved": {k: v for k, v in sorted(vars(args).items())}},
        "policy": {
            "task": args.task, "clip_task": args.clip_task or args.task,
            "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
            "log_dir": str(log_dir),
            "trained_env_yaml_sha256": sha256_file(str(trained_env))
            if trained_env.is_file() else None,
            "conventions": _read_yaml(conventions),
            "trained_tracking_mode": _trained_tracking_mode(trained_env),
            "rollout_tracking_mode": None,   # filled by the cell stage from the live cfg
        },
        "generators": plan["generators"],
        "clip_library": {"source": plan["source_library"], "subset": plan["subset_library"]},
        "plan_hash": plan["plan_hash"],
        "env": {
            "num_envs": args.num_envs, "num_rows": args.num_rows, "seed": args.seed,
            "spawn_z_offset": args.spawn_z_offset, "max_offset": plan["max_offset"],
            "dt": plan["dt"], "clip_device": args.clip_device or "auto",
        },
        "dr": {"ood_scale": args.ood_scale, "ood_depth_noise": bool(args.ood_depth_noise),
               "dr_resample": args.dr_resample},
        "cells": plan["cells"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    dump_params_yaml(str(out_dir / "params" / "eval_args.yaml"),
                     {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                      for k, v in sorted(vars(args).items())})
    git_snapshot(str(out_dir), repo_root=str(_paths.REPO_ROOT))
    return manifest


def _read_yaml(path: pathlib.Path) -> dict | None:
    """Load a small YAML file, or ``None`` if it is absent/unreadable."""
    if not path.is_file():
        return None
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def _trained_tracking_mode(env_yaml: pathlib.Path) -> str | None:
    """``commands.traj_ref.tracking_mode`` from the trained run's dumped env config.

    Read by regex rather than by parsing: ``env.yaml`` is a full configclass dump with
    python-object tags that ``safe_load`` refuses, and one field is all that is needed.
    """
    if not env_yaml.is_file():
        return None
    try:
        text = env_yaml.read_text()
    except OSError:
        return None
    m = re.search(r"^\s*tracking_mode:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    return m.group(1).strip("'\"") if m else None


def _report_plan(plan: dict) -> None:
    """Print the plan's shape, and say plainly where the terrain metric is a no-op."""
    clips = plan["clips"]
    by_family: dict[str, list[dict]] = {}
    for c in clips:
        by_family.setdefault(c["family"], []).append(c)
    print(f"\n[plan] {len(clips)} clips over {len(plan['tile_columns'])} columns, "
          f"plan_hash {plan['plan_hash'][:12]}")
    print(f"{'family':<18} {'clips':>6} {'tiles':>6} {'len med':>8} {'slack med':>10} "
          f"{'terrain metric':>16}")
    for fam in sorted(by_family):
        rows = by_family[fam]
        tiles = len({r["terrain_id"] for r in rows})
        lens = int(np.median([r["clip_len"] for r in rows]))
        slack = int(np.median([r["last_usable_step"] - r["terrain_end_step"] for r in rows]))
        informative = sum(r["completion_informative"] for r in rows)
        label = ("informative" if informative == len(rows)
                 else "DEGENERATE (= success)" if informative == 0
                 else f"mixed {informative}/{len(rows)}")
        print(f"{fam:<18} {len(rows):>6} {tiles:>6} {lens:>8} {slack:>10} {label:>16}")
    degenerate = [f for f, rows in by_family.items()
                  if not any(r["completion_informative"] for r in rows)]
    if degenerate:
        print(f"\n[plan] terrain completion is identical to success for {sorted(degenerate)}: "
              "their pinned span reaches the end of the episode, so the report will print a "
              "dash rather than a number. 'terrain_progress' is the metric that still says "
              "something there.")
    print(f"\n[plan] cells ({len(plan['cells'])}):")
    for c in plan["cells"]:
        print(f"    {c['key']:<28} R={c['robots_per_clip']:<4} task={c['task']}")
    total = sum(c["robots_per_clip"] * len(clips) for c in plan["cells"])
    print(f"[plan] {total:,} episodes across the grid")


def _explicit_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Which option ``dest``s the caller actually TYPED, as opposed to argparse defaults.

    ``argparse`` cannot tell ``--num_envs 256`` from the 256 default, and that distinction is
    what lets ``stage_grid`` fall back to the value ``--stage plan`` recorded WITHOUT
    overriding a deliberate override on the grid invocation. Without it the fallback would
    have to choose globally between "the plan always wins" (which makes the grid's own flags
    inert) and "the CLI always wins" (which is the bug).

    Reads the raw ``argv`` against the parser's own option table rather than re-parsing with
    ``SUPPRESS`` defaults, because the latter means maintaining every default twice.

    Args:
        parser: The parser whose option strings to recognise.
        argv: Arguments as typed, without the program name.

    Returns:
        The ``dest`` names present in ``argv``.
    """
    by_opt: dict[str, str] = {}
    for action in parser._actions:                      # noqa: SLF001 - the option table
        for opt in action.option_strings:
            by_opt[opt] = action.dest
    seen: set[str] = set()
    for tok in argv:
        if tok in by_opt:
            seen.add(by_opt[tok])
        elif tok.startswith("--") and "=" in tok:
            head = tok.split("=", 1)[0]
            if head in by_opt:
                seen.add(by_opt[head])
    return seen


def main() -> int:
    """Dispatch to the requested stage."""
    parser = build_parser()
    args = parser.parse_args()
    # Carried so `stage_grid` can prefer the plan's recorded value for anything the caller did
    # not type. See `_explicit_dests`, and `runner.stage_grid`'s note on the three times a
    # silently-defaulted flag has invalidated a grid.
    args._explicit = _explicit_dests(parser, sys.argv[1:])   # noqa: SLF001
    if args.out is None:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        args.out = str(pathlib.Path("logs") / "terrain_eval" / stamp)
    if args.stage in ("plan", "all"):
        stage_plan(args)
    if args.stage in ("cell", "grid", "all"):
        from runner import stage_cell, stage_grid  # noqa: PLC0415 - imports IsaacLab

        if args.stage == "cell":
            return stage_cell(args)
        return stage_grid(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
