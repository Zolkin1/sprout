# Copyright 2026 Zachary Olkin. All rights reserved.

"""Walk the pair at ONE staircase, over and over, and label what it decided to do.

The field symptom this exists for: the robot sometimes reads a staircase as a box and plans
to jump onto it.  ``scripts/rsl_rl/eval/terrain_class.py`` already turns that into a number
over a whole eval grid, but it is a k-NN classifier whose own stairs->jump confusion is ~0.10
and it can only ever answer in its own output space.  This is the other end: one staircase,
one constant approach command, a human watching, and a label.

Two knobs sweep by default, because they are the two the training library couples:

* **approach speed.**  Every clip of the seven stair and jump families holds ONE constant
  ``(vx, 0, 0)`` for its whole length -- from frame 0, with no ramp -- so a constant command
  stepped in at the spawn IS the in-distribution way to approach terrain.  (``distractor`` and
  ``flat`` do vary within a clip; no terrain family does.)  The deployed tracker publishes no
  ``max_acc``, so nothing rate-limits the step here either, which matches how the clips open.  On
  ``stairs_up`` the library covers a 50-level grid from 0.600 to 2.600 m/s and stops dead;
  ``jump_on`` runs to 3.498.  Above 2.6 m/s the library's only answer to "fast forward
  approach" is a jump, which is a mechanism a speed sweep can actually test.

  The swept variable is the COMMAND, not the achieved speed.  The robot will not reach 3.0 m/s
  in six metres of run-up and is not meant to: the command is what the generator is
  conditioned on, so "what does it plan when told 3.0" is the question, and the CSV in each
  trial directory carries the achieved ``vx_b`` if the difference ever needs reading.
* **flight width.**  Trained transition flights span 0.81-4.14 m after terrain decoration and
  the pure staircases 5.00-5.29.  A flight barely wider than the robot presents a first riser
  that looks like the face of a free-standing box; a full-tile one cannot.

Each trial regenerates a scene holding exactly one staircase -- no side lanes, no jump boxes,
no walls -- so nothing else in frame can explain a plan.  Geometry, spawn and step shape come
from :mod:`generate_stairs_scene`, so this playground cannot drift from the other one.

Each trial then runs ``gen_tracker_runner.py`` in a SUBPROCESS.  That is not incidental: a
headed run ends in ``os._exit(0)`` to dodge the GL/pygame teardown race, so a loop inside one
interpreter gets exactly one trial.

Watch the viewer, close it (or let ``--total_time`` end it), then answer at the terminal::

    label? [s]tairs [j]ump [b]oth [n]either [x] fell early [r]etry [k]skip [q]uit >

A note may follow the letter (``j plan arced up two treads early``).  Labels append to
``labels.jsonl`` as they are given, so an interrupted session keeps everything, and a rerun
skips cells already labelled unless ``--redo``.

Run it from ``transfer/sim`` with ``env_isaaclab`` active::

    python stair_probe.py --load_run 2026-08-20_16-36-10_mode11_decor_plan_v1 \\
        --fm_onnx 2026-08-22_19-14-39_plan_v1_lag48_aug_fm.onnx

The equivalent single scene by hand, if you just want to drive one::

    python generate_stairs_scene.py --scene one_stair --lane 0.86:6 --no_boxes \\
        --rise 0.15 --run 0.25 --x_start 6.0
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
# `plot_style` for the grid figure; `transfer/` for the deploy artifact resolver. Both by
# path rather than by install, exactly as gen_tracker_runner.py reaches its own siblings.
sys.path.insert(0, os.path.join(_ROOT, "scripts", "rsl_rl"))
sys.path.insert(0, os.path.dirname(_HERE))

import generate_stairs_scene as gss  # noqa: E402

# `gen_tracker_runner` and `deploy.artifacts` are imported INSIDE the functions that need them.
# The former pulls mujoco, onnxruntime and pygame at module scope -- seconds, plus SDL's
# banner -- and neither `--help` nor `--plot_only` touches a simulator. `generate_stairs_scene`
# stays up here because it is stdlib-only and every path needs it.

#: The label keys, in prompt order.  ``stairs`` / ``jump`` are the answer; the rest exist so a
#: trial that answered neither question is not forced into one of them -- a run that fell on
#: the flat approach carries no evidence about the staircase at all, and roughly half of the
#: generator-attributable falls on the transition families happen there.
LABELS: dict[str, str] = {
    "s": "stairs",       # stepped up tread by tread
    "j": "jump",         # committed to a leap at the flight
    "b": "both",         # started one and switched, or a jump onto the first tread then steps
    "n": "neither",      # stopped short, turned, walked into the riser, stood
    "x": "fell_early",   # went down before reaching the first riser -- says nothing about stairs
}
_CONTROL = {"r": "retry", "k": "skip", "q": "quit"}

#: Speeds to sweep by default: the ends and the quartiles of the trained ``stairs_up`` grid,
#: plus 3.0 -- past the 2.600 ceiling, where the library has only ever shown jumps.
DEFAULT_SPEEDS: tuple[float, ...] = (0.6, 1.1, 1.6, 2.1, 2.6, 3.0)

#: Widths to sweep by default (m): the narrow trained end, the decorated median, the widest
#: decorated flight, and a pure-tile staircase.  See :data:`gss.TRAINED_STAIR_WIDTHS`.
DEFAULT_WIDTHS: tuple[float, ...] = (0.86, 1.96, 4.14, 5.11)


#: The one robot-identity field whose disagreement this runner can survive: the generator
#: never reads the MJCF at all (its FK loss used it OFFLINE, at dataset build), and MuJoCo's
#: own actuator gains are overwritten here by `Robot.apply_policy_gains` from
#: `policy_parameters.yaml` unless `--keep_xml_gains` -- so the file's contents reach nothing
#: in this loop.  This is a POLICY of this tool, not a restatement of the runtime's: the
#: runtime forgives nothing, and what counts as identity at all is imported below.
_MJCF_ONLY = {"mjcf_sha256"}


def _read_robot_asset(path: str, key: str) -> dict | None:
    """Read one artifact's ``robot_asset`` fingerprint out of its ONNX metadata.

    Via onnxruntime with graph optimization DISABLED -- `onnx` (the protobuf package) is not
    in ``transfer/requirements.txt``, and a metadata read must not pay for a graph rewrite.
    Measured at 0.06 s for ``tracker.onnx`` and 0.21 s for a 27M generator.

    Args:
        path: The ``.onnx`` file.
        key: The metadata key holding the JSON blob -- ``policy_parameters`` for a tracker,
            ``fm_meta`` for a generator.

    Returns:
        The fingerprint dict, or ``None`` if the artifact carries none. "Absent" and "empty"
        are deliberately different things upstream and stay different here.
    """
    import onnxruntime as ort  # noqa: PLC0415 - a few hundred ms, and only on the preflight

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    raw = sess.get_modelmeta().custom_metadata_map.get(key)
    return json.loads(raw).get("robot_asset") if raw else None


def preflight_gamepad() -> str:
    """Refuse a joystick session with no gamepad attached.

    ``JoystickCommand`` falls back to a CONSTANT ``[0.5, 0, 0]`` and a one-line warning when
    it finds no pad -- so without this check a whole labelling evening would silently be a
    0.5 m/s constant-command sweep wearing the label "my natural approach", and nothing in
    the artifacts would say so.

    Returns:
        The controller's name.

    Raises:
        SystemExit: If no gamepad is present.
    """
    import pygame  # noqa: PLC0415 - the constant-command path must not pay for SDL

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise SystemExit(
            "[ERROR] --joystick: no gamepad detected. The runner would fall back to a "
            "constant [0.5, 0, 0] command and label it as your approach. Plug one in, or "
            "drop --joystick for the constant-command sweep.")
    pad = pygame.joystick.Joystick(0)
    pad.init()
    return str(pad.get_name())


def preflight_pair(run_dir: str, fm_arg: str | None, allow_mismatch: bool) -> bool:
    """Compare the pair's robot fingerprints BEFORE the operator sits through a sweep.

    ``GenTrackerPolicy.load`` already refuses a mismatched pair, but it does so inside the
    first trial's subprocess -- so without this the first run dies after the scene is written
    and the session begins by asking for a label on a trial that never happened.

    Args:
        run_dir: The tracker run directory.
        fm_arg: ``--fm_onnx`` as given, or ``None`` to let the resolver pick.
        allow_mismatch: Carry on past a disagreement this function would otherwise refuse.

    Returns:
        Whether ``--allow_robot_asset_mismatch`` must be forwarded to every trial.

    Raises:
        SystemExit: On a disagreement in a field that matters, or a missing artifact.
    """
    from deploy.artifacts import resolve_fm_onnx  # noqa: PLC0415 - see the import note
    # WHICH fields are robot identity is the runtime's call, not this tool's. Comparing every
    # shared key instead made the preflight stricter than the guard it exists to pre-empt: the
    # fingerprint also carries `version`, `covers`, `mjcf_name`, `urdf_name`, `usd_name` and
    # the derived `hash`/`short`, none of which `_assert_robot_assets_agree` treats as
    # identity. On the deployed pair that read ['hash', 'mjcf_sha256', 'short'] where the
    # runtime reads ['mjcf_sha256'] -- three fields to reason about instead of one, two of
    # them consequences of the third.
    from deploy.gen_tracker import _ROBOT_ASSET_FIELDS  # noqa: PLC0415

    exported = os.path.join(run_dir, "exported")
    tracker = os.path.join(exported, "tracker.onnx")
    if not os.path.exists(tracker):
        raise SystemExit(f"[ERROR] no tracker.onnx at {tracker}")
    try:
        fm = resolve_fm_onnx(exported, fm_arg, required=True)
    except Exception as exc:  # noqa: BLE001 - the resolver's message names the candidates
        raise SystemExit(f"[ERROR] could not resolve the generator: {exc}") from exc
    print(f"[INFO] tracker   {tracker}")
    print(f"[INFO] generator {fm}")

    t_asset = _read_robot_asset(tracker, "policy_parameters")
    f_asset = _read_robot_asset(fm, "fm_meta")
    if not t_asset or not f_asset:
        print("[INFO] one side carries no robot fingerprint; nothing to compare.")
        return False
    diffs = sorted(k for k in _ROBOT_ASSET_FIELDS
                   if k in t_asset and k in f_asset and t_asset[k] != f_asset[k])
    if not diffs:
        print(f"[INFO] robot fingerprints agree ({t_asset.get('short')}).")
        return False

    print(f"[WARN] robot fingerprints DISAGREE: {t_asset.get('short')} vs "
          f"{f_asset.get('short')}, on {diffs}")
    if set(diffs) <= _MJCF_ONLY:
        print("[WARN] the difference is the MJCF hash and nothing else. The generator does "
              "not read the MJCF, and this runner overwrites MuJoCo's actuator gains with "
              "the policy's, so it reaches nothing in the loop. Forwarding "
              "--allow_robot_asset_mismatch and continuing.")
        print("[WARN] the real fix is re-exporting the generator against the current robot; "
              "until then every artifact from this session carries the same caveat.")
        return True
    if allow_mismatch:
        print("[WARN] --allow_asset_mismatch was passed. Anything measured here is a "
              "statement about a pair the fingerprint says is not matched.")
        return True
    raise SystemExit(
        f"[ERROR] the pair disagrees on {diffs}, which is more than the MJCF hash. "
        "Re-export both from one run, or pass --allow_asset_mismatch to say you mean it.")


def scene_name(width: float, rise: float, run: float, steps: int, x_start: float,
               tag: str = "") -> str:
    """A scene name that states its own geometry, e.g. ``stair_probe_w086_r15_t25_n6``.

    Naming by geometry rather than by trial index means the emitted XML is still meaningful
    after the session, and a single trial can be re-driven by hand with ``--scene <name>``.

    Args:
        width: Lateral flight width (m).
        rise: Per-step height (m).
        run: Per-step depth (m).
        steps: Number of ascending steps.
        x_start: Flat approach before the first riser (m). In the name because it is the one
            setup number that changes what a trial MEASURES rather than how long it takes: a
            long approach lets lateral drift accumulate before the flight, a short one denies
            the robot the run-up to reach its command.
        tag: Extra name element. Tests pass one so their scratch scenes cannot collide with a
            live session's -- the name is a pure function of the geometry, so a test that
            happens to use the default staircase writes, and then deletes, exactly the file a
            queued trial is about to load.

    Returns:
        The scene name, without the robot prefix or the ``.xml`` suffix.
    """
    part = f"_{tag}" if tag else ""
    # MILLIMETRES, not centimetres. The sweep, the resume set and the figure all key widths at
    # 1e-4, so `--widths 0.86 0.865` is two cells -- but at centimetre precision they shared one
    # scene file and one trial directory, and the second trial overwrote the first's CSV and
    # verdict while the first row went on naming them. Probing where the behaviour flips is
    # exactly when sub-centimetre steps get asked for.
    return (f"stair_probe{part}_w{int(round(width * 1000)):04d}"
            f"_r{int(round(rise * 1000)):03d}_t{int(round(run * 1000)):03d}_n{steps}"
            f"_x{int(round(x_start * 1000)):04d}")


def write_scene(width: float, rise: float, run: float, steps: int, x_start: float,
                platform_depth: float, descend: bool, robot: str,
                tag: str = "") -> tuple[str, str]:
    """Emit the one-staircase scene for this geometry and return ``(scene_name, path)``.

    One lane at ``y = 0`` and nothing else: no side lanes, no jump boxes, no walls.  A box in
    frame would be the obvious confound for a stairs-vs-jump question, and the default
    playground has four of them.

    Args:
        width: Lateral flight width (m) -- the dimension across the direction of travel.
        rise: Per-step height (m).
        run: Per-step depth along travel (m).
        steps: Number of ascending steps.
        x_start: Flat approach (m) before the first riser; the robot spawns at ``x = 0``.
        platform_depth: Flat platform depth (m) at the top.
        descend: Also emit the mirrored descending flight, as ``terrain_scene`` does.
        robot: Robot model name, e.g. ``g1_perceptive``.
        tag: Passed to :func:`scene_name`; see its note on why tests set one.

    Returns:
        The scene name (for ``--scene``) and the absolute path written.
    """
    lane = gss.Lane(name="stairs", y=0.0, width=width, steps=steps, rise=rise, run=run)
    xml = gss.build_scene_xml(f"{robot}.xml", (lane,), rise, run, x_start, platform_depth,
                              boxes=(), descend=descend, walls=None)
    name = scene_name(width, rise, run, steps, x_start, tag)
    # Beside the other scenes on purpose: the emitted XML `<include>`s the robot model by a
    # path relative to its own directory, so a scene written anywhere else cannot compile.
    path = os.path.join(_HERE, "robots", "g1", f"{robot}_{name}.xml")
    with open(path, "w") as f:
        f.write(xml)
    return name, path


def width_note(width: float) -> str:
    """Whether this flight width is inside a trained band, and which one."""
    bands = gss.TRAINED_STAIR_WIDTHS
    kinds = ("narrow transition", "widened transition", "pure")
    for kind, (lo, hi) in zip(kinds, bands):
        if lo <= width <= hi:
            return f"in the trained {kind} band {lo}-{hi} m"
    return f"OUTSIDE every trained band {bands} m -- deliberate?"


def speed_note(speed: float) -> str:
    """Whether this approach speed is one the STAIR families ever carried.

    The stair families -- ``stairs_up``, ``stairs_down`` and both ``_pure`` variants -- share
    one 50-level grid, 0.600 to 2.600 m/s exactly.  The box families do not: ``jump_on`` runs
    0.504-3.498 and ``jump_off_front`` 0.503-3.499, both sampled rather than gridded.  So the
    ends of the stair grid are not the ends of the library, and the gap above 2.600 is the
    interesting one: there the library holds jumps and no stairs at all.
    """
    if speed < 0.6:
        return ("BELOW the 0.600 m/s floor of the stair grid (the box families do reach "
                "0.50 here)")
    if speed > 2.6:
        return "ABOVE the 2.600 m/s stair ceiling -- only jump_on/jump_off_front go here"
    return "inside the trained stairs grid 0.600-2.600 m/s"


#: The fastest a commanded approach is assumed to be WALKED, for budgeting only (m/s).  The
#: command is the swept variable and the robot does not achieve all of it -- at 3.0 m/s
#: commanded it covers ground at rather less -- so budgeting on the command alone gives the
#: fast cells the least tolerance for exactly the shortfall they are most likely to have.
#: 2.6 is the top of the trained stair grid: past it, budget as if the robot walked 2.6.
BUDGET_SPEED_CAP = 2.6


def trial_seconds(speed: float, x_start: float, run: float, steps: int,
                  platform_depth: float, settle: float, cap: float) -> float:
    """How long to give a trial, from the distance it has to cover.

    A fixed budget either wastes the operator's time at 3.0 m/s or cuts the run off before the
    robot reaches the first riser at 0.6 m/s -- which would look exactly like a failure to
    climb.  Errs long: the operator ends a trial early by closing the viewer, so a generous
    budget costs patience while a short one costs a wrong label.

    Args:
        speed: The COMMANDED forward speed (m/s); the budget assumes the robot walks at
            ``min(speed, BUDGET_SPEED_CAP)``, which it may still not reach.
        x_start: Flat approach before the first riser (m).
        run: Tread run (m).
        steps: Ascending step count.
        platform_depth: Top platform depth (m).
        settle: Seconds added for the spawn transient and the ramp to speed.
        cap: Hard upper bound (s).

    Returns:
        The trial's ``--total_time`` in seconds.
    """
    dist = x_start + steps * run + platform_depth
    walked = min(max(speed, 0.2), BUDGET_SPEED_CAP)
    return float(min(cap, dist / walked + settle))


def run_trial(args, width: float, speed: float | None, lateral: float, yaw: float,
              out_dir: str, allow_mismatch: bool) -> dict:
    """Generate the scene, run one viewer trial, and return what the runner reported.

    Returns:
        A dict with ``scene``, ``scene_path``, ``total_time``, ``argv``, ``returncode``,
        ``ran`` (whether the loop produced a verdict at all) and the runner's own verdict
        fields (``survived``, ``time_to_fall_s``, ``failure_reason``, ``sim_end_time_s``) read
        back from ``sim_meta.yaml``.  The verdict is read from that file and never from the
        exit code: a headed run always exits 0 (it ends in ``os._exit(0)``), and the CSV's
        ``fell`` column structurally cannot see a fall.
    """
    name, scene_path = write_scene(width, args.rise, args.run, args.steps, args.x_start,
                                   args.platform_depth, not args.no_descent, args.robot_name)
    total_time = (float(args.joystick_time) if speed is None else
                  trial_seconds(speed, args.x_start, args.run, args.steps,
                                args.platform_depth, args.settle, args.max_trial_time))
    os.makedirs(out_dir, exist_ok=True)
    # A trial directory is REUSED by [r]etry, and the "did it run" test below is the presence
    # of `survived` in sim_meta.yaml. Without this, a retry that dies before the loop reads the
    # PREVIOUS attempt's verdict and the operator is asked to label a run that never happened
    # -- the exact thing the guard exists to prevent, defeated by a stale file.
    for stale in ("sim_meta.yaml", "sim_log.csv"):
        try:
            os.remove(os.path.join(out_dir, stale))
        except FileNotFoundError:
            pass

    argv = [sys.executable, os.path.join(_HERE, "gen_tracker_runner.py"),
            "--scene", name,
            "--robot_name", args.robot_name,
            "--total_time", f"{total_time:.2f}",
            "--rt_factor", f"{args.rt_factor}",
            # --log for the verdict (sim_meta.yaml), --no_metrics because the six per-trial
            # figures and the summary JSON dumped to stdout cost a few seconds and a
            # screenful each, right where the operator is about to read the verdict.
            "--log", "--no_metrics", "--log_dir", out_dir]
    if speed is None:
        # The operator's own stick, at the runner's own scaling. What they command is then a
        # MEASUREMENT (read back from the log below) rather than a setting.
        argv += ["--mode", "joystick",
                 "--max_vx", f"{args.max_vx}", "--max_vy", f"{args.max_vy}",
                 "--max_wz", f"{args.max_wz}"]
    else:
        argv += ["--const_cmd", f"{speed}", f"{lateral}", f"{yaw}"]
    if allow_mismatch:
        argv += ["--allow_robot_asset_mismatch"]
    if args.experiment:
        argv += ["--experiment", args.experiment]
    if args.load_run:
        argv += ["--load_run", args.load_run]
    if args.fm_onnx:
        argv += ["--fm_onnx", args.fm_onnx]
    argv += list(args.extra)

    print(f"    $ {' '.join(argv)}")
    proc = subprocess.run(argv, cwd=_HERE)

    verdict = {"survived": None, "time_to_fall_s": None, "failure_reason": None,
               "sim_end_time_s": None}
    ran = False
    meta_path = os.path.join(out_dir, "sim_meta.yaml")
    try:
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        for k in verdict:
            verdict[k] = meta.get(k)
        # `survived` is stamped only after the loop returns, so its PRESENCE is the test for
        # "a trial happened". The file itself is written at log open, before the first step.
        ran = "survived" in meta
    except FileNotFoundError:
        pass
    # `survived: True` also means "the operator closed the viewer", which the docstring tells
    # them to do -- so on its own it would read as "the robot got through the staircase" for a
    # trial ended two seconds in. The budget is known, so the short run is recoverable.
    end = verdict["sim_end_time_s"]
    verdict["ended_early"] = (bool(end is not None and end < total_time - 0.5)
                              if verdict["survived"] is not False else False)
    cmd = command_stats(os.path.join(out_dir, "sim_log.csv"), args.x_start)
    return {"scene": name, "scene_path": scene_path, "total_time": total_time,
            "argv": argv, "returncode": proc.returncode, "ran": ran, **verdict, **cmd}


def command_stats(log_path: str, x_start: float) -> dict:
    """Summarise the command the operator gave, from the runner's own per-step log.

    ``cmd_vx`` / ``cmd_vy`` / ``cmd_wz`` are logged every control step, so a joystick trial
    records what was asked for without any new plumbing.  Reported over the APPROACH -- every
    step before the robot reaches the first riser -- as well as over the whole trial, because
    the approach is what conditions the plan that decides stairs-or-box; what the stick did
    afterwards is a reaction to that decision, not an input to it.

    Args:
        log_path: The trial's ``sim_log.csv``.
        x_start: World x of the first riser (m); the approach is ``act_pos_x < x_start``.

    Returns:
        Per-segment medians and extremes of the three command channels, or ``{}`` if the log
        is missing or holds no rows.
    """
    import csv  # noqa: PLC0415 - only the joystick path reads a log back

    try:
        with open(log_path) as f:
            data = list(csv.DictReader(f))
    except FileNotFoundError:
        return {}
    if not data:
        return {}

    def seg(rows: list[dict], tag: str) -> dict:
        if not rows:
            return {}
        out: dict = {}
        for src, key in (("cmd_vx", "vx"), ("cmd_vy", "vy"), ("cmd_wz", "wz")):
            vals = np.array([float(r[src]) for r in rows])
            out[f"{tag}_{key}_med"] = float(np.median(vals))
            # The SIGNED extreme: a stick held left the whole way and one flicked both ways
            # have the same |max| and are different approaches.
            out[f"{tag}_{key}_min"] = float(vals.min())
            out[f"{tag}_{key}_max"] = float(vals.max())
            out[f"{tag}_{key}_absmed"] = float(np.median(np.abs(vals)))
        out[f"{tag}_n"] = len(rows)
        return out

    approach = [r for r in data if float(r["act_pos_x"]) < x_start]
    stats = seg(data, "all") | seg(approach, "app")
    # `app_` prefixed so the row assembly's prefix filter carries it: without the underscore
    # `approach_reached` sorts as a different key and was silently dropped from every row.
    stats["app_reached"] = bool(approach) and len(approach) < len(data)
    return stats


def prompt_label() -> tuple[str, str]:
    """Ask for one label, returning ``(key, note)``.

    ``key`` is a member of :data:`LABELS` or of ``retry`` / ``skip`` / ``quit``.  Anything
    after the letter is kept verbatim as the note.
    """
    def _key(k: str, v: str) -> str:
        """``s`` + ``stairs`` -> ``[s]tairs``; falls back to ``[x] fell_early``."""
        return f"[{k}]{v[1:]}" if v.startswith(k) else f"[{k}] {v}"

    keys = "  ".join(_key(k, v) for k, v in LABELS.items())
    ctrl = "  ".join(_key(k, v) for k, v in _CONTROL.items())
    while True:
        try:
            raw = input(f"  label? {keys}  {ctrl} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit", ""
        if not raw:
            continue
        key, _, note = raw.partition(" ")
        key = key.lower()
        # The full word is accepted as well as the letter: the prompt shows both halves of
        # each ("[s]tairs"), so typing what is on screen must work either way round.
        by_word = {v: v for v in LABELS.values()} | {v: v for v in _CONTROL.values()}
        if key in LABELS:
            return LABELS[key], note.strip()
        if key in _CONTROL:
            return _CONTROL[key], note.strip()
        if key in by_word:
            return by_word[key], note.strip()
        print(f"    not a label: {key!r}")


def _cmd_label(vx: float, lateral: float, yaw: float) -> str:
    """How a constant command reads in the trial header and the dry-run plan."""
    parts = [f"vx {vx:.2f}"]
    if abs(lateral) > 1e-9:
        parts.append(f"vy {lateral:+.2f}")
    if abs(yaw) > 1e-9:
        parts.append(f"wz {yaw:+.2f}")
    return "  ".join(parts)


def _cell_key(width: float, speed: float | None, lateral: float, yaw: float,
              rep: int) -> tuple:
    """The identity of one cell, for the resume set and for the figure.

    ``speed`` is ``None`` on a joystick trial: the command is the operator's, so there is no
    swept speed to key on, and the repeats of a width are DIFFERENT approaches rather than
    repeated draws of one condition.  ``lateral`` and ``yaw`` are 0.0 on every trial run before
    they existed, which is what they were, so old sessions resume unchanged.  Rounded because
    the sweep, the resume set and the figure must agree on floats that have been through JSON.
    """
    return (round(float(width), 4), None if speed is None else round(float(speed), 4),
            round(float(lateral or 0.0), 4), round(float(yaw or 0.0), 4), int(rep))


def _ask_continue() -> bool:
    """After a trial that could not run, ask whether to keep going."""
    try:
        return input("  the trial did not run. continue anyway? [y/N] > ").strip().lower() \
            in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def load_done(path: str) -> set[tuple]:
    """The ``(width, speed, repeat)`` cells already labelled in an existing ``labels.jsonl``."""
    done: set[tuple] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add(_cell_key(r["width"], r.get("speed"), r.get("lateral", 0.0),
                               r.get("yaw", 0.0), r["repeat"]))
    return done


def read_rows(path: str) -> list[dict]:
    """Every labelled row on disk."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _x_axis(rows: list[dict], speeds: list[float]) -> tuple[str, list[float], str]:
    """Which command variable the grid's x axis should be, from what actually varied.

    A run that holds one speed and sweeps ``--lateral`` has a speed axis of length one and a
    real lateral axis; plotting speed there would give a single column and throw the result
    away.  Returns ``(field, values, axis label)``.
    """
    for field, label in (("lateral", r"commanded $v_y$ (m/s)"),
                         ("yaw", r"commanded $\omega_z$ (rad/s)")):
        vals = sorted({round(float(r.get(field) or 0.0), 4) for r in rows})
        if len(vals) > 1:
            return field, vals, label
    vals = sorted({round(float(v), 4) for v in speeds}
                  | {round(float(r["speed"]), 4) for r in rows
                     if r.get("speed") is not None})
    return "speed", vals, r"constant approach $v_x$ (m/s)"


def plot_grid(rows: list[dict], widths: list[float], speeds: list[float], out_stem: str,
              usetex: bool | None = None) -> list[str]:
    """Draw the labelled (width x speed) grid: modal label per cell, with agreement.

    A cell is coloured by its modal label and annotated ``m/n`` -- how many of that cell's
    repeats agreed with the mode.  A cell whose repeats disagree is the interesting one and
    has to be visible as such, not averaged into a number.

    Returns:
        The files written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    import plot_style

    plot_style.apply_paper_style(usetex=usetex)
    order = list(LABELS.values())
    colours = {"stairs": plot_style.COLORS["blue"], "jump": plot_style.COLORS["orange"],
               "both": plot_style.COLORS["magenta"], "neither": plot_style.COLORS["teal"],
               "fell_early": "#BBBBBB"}

    field, speeds, xlabel = _x_axis(rows, speeds)
    cells: dict[tuple, list[str]] = {}
    for r in rows:
        x = r["speed"] if field == "speed" else (r.get(field) or 0.0)
        cells.setdefault((round(float(r["width"]), 4), round(float(x), 4)),
                         []).append(r["label"])

    grid = np.full((len(widths), len(speeds)), np.nan)
    text = np.full((len(widths), len(speeds)), "", dtype=object)
    for i, w in enumerate(widths):
        for j, v in enumerate(speeds):
            labs = cells.get((round(w, 4), round(v, 4)))
            if not labs:
                continue
            mode = max(order, key=lambda L: (labs.count(L), -order.index(L)))
            grid[i, j] = order.index(mode)
            text[i, j] = f"{labs.count(mode)}/{len(labs)}"

    fig, ax = plt.subplots(figsize=plot_style.figsize(2, 1))
    cmap = ListedColormap([colours[L] for L in order])
    cmap.set_bad("#FFFFFF")
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap,
              norm=BoundaryNorm(np.arange(-0.5, len(order)), cmap.N),
              origin="lower", aspect="auto", interpolation="nearest")
    for i in range(len(widths)):
        for j in range(len(speeds)):
            if text[i, j]:
                # Black on the grey `fell_early` fill; white on every saturated one.
                fg = "k" if order[int(grid[i, j])] == "fell_early" else "w"
                ax.text(j, i, text[i, j], ha="center", va="center", fontsize=13, color=fg)
    # Cell separators: two adjacent cells sharing a modal label are otherwise one block, and
    # the count annotation is then the only thing saying how many cells that block is.
    for j in range(len(speeds) - 1):
        ax.axvline(j + 0.5, color="w", lw=1.2)
    for i in range(len(widths) - 1):
        ax.axhline(i + 0.5, color="w", lw=1.2)
    # The 2.600 m/s stair ceiling: to its right the library holds jumps and nothing else.
    if field == "speed":
        # The 2.600 m/s stair ceiling. Meaningless on a lateral or yaw axis, where the
        # library's whole support is the single point 0.
        over = [j for j, v in enumerate(speeds) if v > 2.6]
        if over and over[0] > 0:
            ax.axvline(over[0] - 0.5, color="k", lw=2.0, ls="--")
    elif len(speeds) > 1 and speeds[0] == 0.0:
        # Column 0 IS the training distribution; everything right of it is extrapolation.
        ax.axvline(0.5, color="k", lw=2.0, ls="--")
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([f"{v:.2f}" for v in speeds])
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([f"{w:.2f}" for w in widths])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"flight width (m)")
    ax.legend(handles=[Patch(facecolor=colours[L], label=plot_style.tex_escape(L))
                       for L in order],
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=13)
    caption = (
        "Hand-labelled stair-vs-jump behaviour on a single staircase. Each cell is one "
        "(flight width, constant approach speed) pair; colour is the modal label over that "
        "cell's repeats and the annotation is how many repeats agreed. The dashed rule is "
        "the edge of the training distribution on whichever command the columns sweep: "
        "2.600 m/s on a speed axis (the ceiling of every stair family, past which the "
        "library holds only jumps), or the single column at 0 on a lateral or yaw axis "
        "(every terrain clip is commanded v_y = w_z = 0 exactly, so every other column is "
        "extrapolation). 'fell_early' means the "
        "robot went down before the first riser, so that trial carries no evidence about the "
        "staircase either way. BOTH AXES ARE CATEGORICAL: the rows are the widths that were "
        "run, evenly spaced whatever their values, not a metric axis."
    )
    return plot_style.save_figure(fig, out_stem, caption=caption)


def plot_joystick(rows: list[dict], widths: list[float], out_stem: str,
                  usetex: bool | None = None) -> list[str]:
    """Two panels for a joystick session: outcome per width, and what each approach was.

    A joystick session has no speed axis, so the constant-command grid does not apply.  What
    replaces it is the thing that session is actually for: the operator's approach is a
    MEASUREMENT, so panel (b) plots every trial in the command space the training library
    never covers -- lateral against yaw, both exactly zero on all 7000 terrain clips -- with
    the label as the colour.  If box-selection lives out there, this is the panel that shows
    it; if the jumps sit at the origin with everything else, it does not.

    Returns:
        The files written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import plot_style

    plot_style.apply_paper_style(usetex=usetex)
    order = list(LABELS.values())
    colours = {"stairs": plot_style.COLORS["blue"], "jump": plot_style.COLORS["orange"],
               "both": plot_style.COLORS["magenta"], "neither": plot_style.COLORS["teal"],
               "fell_early": "#BBBBBB"}
    esc = plot_style.tex_escape
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # (a) outcome per width -- stacked counts, since the repeats of a width are the sample.
    ax = axes[0]
    widths = sorted({round(float(w), 4) for w in widths}
                    | {round(float(r["width"]), 4) for r in rows})
    bottom = np.zeros(len(widths))
    for lab in order:
        n = np.array([sum(1 for r in rows
                          if round(float(r["width"]), 4) == w and r["label"] == lab)
                      for w in widths], dtype=float)
        if n.sum() == 0:
            continue
        ax.bar(range(len(widths)), n, bottom=bottom, color=colours[lab], width=0.7,
               label=esc(lab), edgecolor="w", linewidth=1.2)
        bottom += n
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([f"{w:.2f}" for w in widths])
    ax.set_xlabel(r"flight width (m)")
    ax.set_ylabel(r"trials")
    ax.legend(loc="upper right", fontsize=12)
    ax.text(0.0, 1.02, r"(a)", transform=ax.transAxes, va="bottom", fontsize=18)

    # (b) the approach itself, in the two channels training pins to zero.
    ax = axes[1]
    for lab in order:
        pts = [(r.get("app_vy_absmed"), r.get("app_wz_absmed"), r.get("app_vx_med"))
               for r in rows if r["label"] == lab]
        pts = [p for p in pts if p[0] is not None and p[1] is not None]
        if not pts:
            continue
        vy, wz, vx = zip(*pts)
        # Marker area carries the forward command, so a slow careful approach and a fast one
        # are distinguishable without a third axis.
        ax.scatter(vy, wz, s=[30 + 260 * abs(v or 0.0) for v in vx], alpha=0.7,
                   color=colours[lab], edgecolors="k", linewidths=0.8, label=esc(lab))
    ax.plot([0.0], [0.0], "*", ms=18, mfc=plot_style.COLORS["red"], mec="k", mew=1.0,
            zorder=5, label=r"every terrain clip")
    ax.set_xlabel(r"approach $|v_y|$ median (m/s)")
    ax.set_ylabel(r"approach $|\omega_z|$ median (rad/s)")
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01)
    ax.legend(loc="upper right", fontsize=11)
    ax.text(0.0, 1.02, r"(b)", transform=ax.transAxes, va="bottom", fontsize=18)

    caption = (
        "Hand-labelled stair-vs-jump behaviour on a single staircase, driven by GAMEPAD. "
        "(a) outcome per flight width over that width's repeats -- the repeats are different "
        "natural approaches, not draws of one condition. (b) each trial placed by the command "
        "the operator actually gave over the APPROACH (every step before the first riser), "
        "read back from the runner's own cmd_vy / cmd_wz log; marker area is the median "
        "forward command. The star at the origin is where the whole training library sits: "
        "all 7000 terrain clips are commanded v_y = w_z = 0 exactly, so every point away "
        "from it is off-distribution in a channel no terrain clip covers. Read (b) for "
        "whether the jump labels separate from the stairs labels in that space."
    )
    return plot_style.save_figure(fig, out_stem, caption=caption)


def summarise(rows: list[dict]) -> dict:
    """Counts by label, by speed and by width, plus the fall rate."""
    out: dict = {"n": len(rows), "by_label": {}, "by_speed": {}, "by_width": {}}
    for r in rows:
        out["by_label"][r["label"]] = out["by_label"].get(r["label"], 0) + 1
        for key, val in (("by_speed", r.get("speed")), ("by_width", r["width"])):
            # `by_speed` is keyed "joystick" rather than dropped: a session where every trial
            # was hand-driven should still say so in its summary, not show an absent axis.
            name = "joystick" if val is None else f"{float(val):.2f}"
            bucket = out[key].setdefault(name, {})
            bucket[r["label"]] = bucket.get(r["label"], 0) + 1
    out["fell"] = sum(1 for r in rows if r.get("survived") is False)
    # Reported separately because a run the operator cut short is not evidence of survival.
    out["ended_early"] = sum(1 for r in rows if r.get("ended_early"))
    return out


def main() -> None:
    """Sweep the grid, run each trial, take a label, write the artifacts."""
    ap = argparse.ArgumentParser(
        description="Drive the generator+tracker at ONE staircase and hand-label what it did.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--joystick", action="store_true", default=False,
                    help="Drive each trial by GAMEPAD instead of a constant command. Width is "
                         "then the only swept variable and your approach is a measurement: "
                         "the runner logs cmd_vx/cmd_vy/cmd_wz every step, and each row gets "
                         "medians and extremes of all three over the approach. Use --repeats "
                         "to capture several natural approaches per width.")
    ap.add_argument("--joystick_time", type=float, default=30.0,
                    help="Trial length under --joystick (s). There is no commanded speed to "
                         "budget from; close the viewer when you have seen the answer.")
    ap.add_argument("--max_vx", type=float, default=1.0,
                    help="Gamepad scaling, forwarded to the runner. The default 1.0 sits "
                         "inside the trained stair band (0.600-2.600 m/s commanded) and near "
                         "the ~0.9 m/s the robot actually achieves.")
    ap.add_argument("--max_vy", type=float, default=0.5,
                    help="Lateral gamepad scaling. NOTE: every terrain clip in the library is "
                         "commanded v_y = 0 EXACTLY, so any stick deflection here is off "
                         "distribution -- which is the point of measuring it.")
    ap.add_argument("--max_wz", type=float, default=1.0,
                    help="Yaw gamepad scaling. Same note as --max_vy: w_z is exactly 0 on "
                         "every terrain clip.")
    ap.add_argument("--experiment", default="g1",
                    help="Log subtree under logs/rsl_rl/ holding the tracker run.")
    ap.add_argument("--allow_asset_mismatch", action="store_true", default=False,
                    help="Run a pair whose robot fingerprints disagree in a field beyond the "
                         "MJCF hash. An MJCF-only difference is forgiven automatically and "
                         "loudly; anything else is refused without this.")
    ap.add_argument("--load_run", default=None,
                    help="Tracker run dir under logs/rsl_rl/<experiment> (default: latest).")
    ap.add_argument("--fm_onnx", default=None,
                    help="Generator to use. REQUIRED when the run's exported/ holds more than "
                         "one -- the resolver refuses to pick and lists them.")
    ap.add_argument("--robot_name", default="g1_perceptive")
    ap.add_argument("--speeds", type=float, nargs="+", default=None,
                    help=f"Constant approach speeds (m/s). Default {list(DEFAULT_SPEEDS)}. "
                         "Under --plot_only, omitting this takes the axis from the labels "
                         "themselves rather than from that default.")
    ap.add_argument("--widths", type=float, nargs="+", default=None,
                    help=f"Lateral flight widths (m) -- across the direction of travel, NOT "
                         f"the tread run. Default {list(DEFAULT_WIDTHS)}; same --plot_only "
                         "rule as --speeds.")
    ap.add_argument("--lateral", type=float, nargs="+", default=[0.0], metavar="VY",
                    help="Constant LATERAL command(s) held through the approach (m/s), swept "
                         "as a third axis. Every terrain clip in the library is commanded "
                         "v_y = 0 exactly, so any value here is off-distribution by "
                         "construction. Ignored under --joystick. "
                         "CONFOUNDED WITH DRIFT, AND NOT WHAT A JOYSTICK DOES: held for the "
                         "whole approach it also walks the robot sideways -- at the default "
                         "6 m approach and the ~0.9 m/s the robot achieves, vy 0.20 is 1.33 m "
                         "of lateral travel, past the half-width of any real staircase. "
                         "Measured, an operator's lateral command is a BRIEF correction "
                         "(median |vy| 0.036, peak 0.093, peak/median 1.5x), so a sustained "
                         "command tests a different thing. Keep vy*x_start/0.9 under the "
                         "flight's half-width, or the arm measures approach geometry rather "
                         "than command.")
    ap.add_argument("--yaw", type=float, nargs="+", default=[0.0], metavar="WZ",
                    help="Constant YAW command(s) (rad/s), swept the same way. Worth running "
                         "beside --lateral as a NEGATIVE CONTROL: w_z is also exactly 0 on "
                         "every terrain clip, so if only one of the two moves the label that "
                         "is a much stronger result than either alone.")
    ap.add_argument("--rise", type=float, default=0.15, help="Step height (m).")
    ap.add_argument("--run", type=float, default=0.25, help="Tread run (m), along travel.")
    ap.add_argument("--steps", type=int, default=6, help="Number of ascending steps.")
    ap.add_argument("--x_start", type=float, default=6.0,
                    help="Flat approach (m) before the first riser. The trained transition "
                         "tiles give 6.2-10.1 m, and at 2.6 m/s the default playground's "
                         "2.5 m is under a second of run-up -- too short to reach the "
                         "commanded speed, which would confound every fast trial.")
    ap.add_argument("--platform_depth", type=float, default=2.0,
                    help="Flat platform depth (m) at the top of the flight.")
    ap.add_argument("--no_descent", action="store_true", default=False,
                    help="Ascend-only, dead-ending on the platform. Default is the symmetric "
                         "shape terrain_scene's lanes have.")
    ap.add_argument("--repeats", type=int, default=1, help="Trials per (width, speed) cell.")
    ap.add_argument("--rt_factor", type=float, default=1.0,
                    help="Sim speed relative to wall clock; >1 to get through a sweep faster.")
    ap.add_argument("--settle", type=float, default=4.0,
                    help="Seconds added to each trial on top of distance/speed, for the spawn "
                         "transient and the ramp to the commanded speed. Generous on purpose: "
                         "the budget is an upper bound the operator can end early by closing "
                         "the viewer, and a budget that expires before the robot reaches the "
                         "first riser looks exactly like a failure to climb.")
    ap.add_argument("--max_trial_time", type=float, default=45.0,
                    help="Hard cap on a trial (s).")
    ap.add_argument("--out", default=None,
                    help="Session directory. Default is a FRESH logs/stair_probe/<timestamp>, "
                         "so a plain rerun starts a new session rather than continuing the "
                         "last one -- use --resume, or name the directory, to continue.")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Continue the most recent session under logs/stair_probe/ instead of "
                         "opening a new one: cells already labelled there are skipped.")
    ap.add_argument("--redo", action="store_true", default=False,
                    help="Re-run cells that already carry a label in this session directory.")
    ap.add_argument("--shuffle", action="store_true", default=False,
                    help="Present trials in a random order. Worth it for a labelling session "
                         "you intend to quote: in grid order the previous cell's answer is "
                         "visible while judging the next one.")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    ap.add_argument("--dry_run", action="store_true", default=False,
                    help="Resolve and check the policy pair, write the scenes, print the "
                         "plan and the per-trial budgets -- but run no trial.")
    ap.add_argument("--plot_only", action="store_true", default=False,
                    help="Re-draw the grid and summary from an existing session and exit.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Everything after this is passed through to gen_tracker_runner.py "
                         "verbatim (e.g. --extra --depth_noise --fm_cuda_graph). Must be last.")
    args = ap.parse_args()

    # argparse.REMAINDER takes EVERYTHING after --extra, including this parser's own options,
    # and reverts them to their defaults without a word. That is the same silent-default class
    # as the eval harness's unforwarded cell flags: a sweep that says --extra --headless
    # --widths 0.86 runs the DEFAULT four widths and records them as asked-for.
    ours = {o for a in ap._actions for o in a.option_strings}
    swallowed = [x for x in args.extra if x in ours]
    if swallowed:
        sys.exit(f"[ERROR] --extra swallowed this script's own flags {swallowed}: it takes "
                 "every remaining argument, so it must come LAST. Move them before --extra.")

    sessions_root = os.path.join(_ROOT, "logs", "stair_probe")
    if args.out:
        # Relative to the CWD would put a session under transfer/sim on the documented
        # invocation; sessions belong beside each other under logs/.
        out_dir = args.out if os.path.isabs(args.out) else os.path.join(sessions_root,
                                                                        args.out)
    elif args.resume:
        # The latest session THAT HAS LABELS, not the latest directory. A --dry_run or an
        # aborted start leaves an empty session behind, and resuming into one silently orphans
        # the work it was meant to continue -- which is the whole reason --resume exists.
        prior = sorted(d for d in (os.listdir(sessions_root)
                                   if os.path.isdir(sessions_root) else [])
                       if os.path.getsize(os.path.join(sessions_root, d, "labels.jsonl"))
                       if os.path.exists(os.path.join(sessions_root, d, "labels.jsonl")))
        if not prior:
            sys.exit(f"[ERROR] --resume: no session under {sessions_root} carries any labels")
        out_dir = os.path.join(sessions_root, prior[-1])
        n_prior = sum(1 for _ in open(os.path.join(out_dir, "labels.jsonl")))
        print(f"[INFO] resuming {prior[-1]} ({n_prior} labels already in it)")
    else:
        out_dir = os.path.join(sessions_root,
                               datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    labels_path = os.path.join(out_dir, "labels.jsonl")
    if args.plot_only and not os.path.exists(labels_path):
        # Creating the directory first would turn a mistyped --out into an empty session that
        # then reports "no labels", which reads as data loss rather than as a typo.
        sys.exit(f"[ERROR] no labels.jsonl at {labels_path}")
    if not args.dry_run:
        # A dry run writes scenes and prints a plan; it must not leave a session directory,
        # or every rehearsal becomes a candidate for --resume to land in.
        os.makedirs(out_dir, exist_ok=True)

    # Under --plot_only an unset axis is taken from the labelled rows by `_finish`; on a sweep
    # it falls back to the defaults, which are the grid this tool exists to run.
    fallback_w = [] if args.plot_only else list(DEFAULT_WIDTHS)
    fallback_v = [] if args.plot_only else list(DEFAULT_SPEEDS)
    widths = [float(w) for w in (args.widths if args.widths else fallback_w)]
    speeds = [float(v) for v in (args.speeds if args.speeds else fallback_v)]

    if args.plot_only:
        rows = read_rows(labels_path)
        if not rows:
            sys.exit(f"[ERROR] no labels in {labels_path}")
        _finish(rows, widths, speeds, out_dir, labels_path)
        return


    from gen_tracker_runner import find_latest_run  # noqa: PLC0415 - see the import note

    log_root = os.path.join(_ROOT, "logs", "rsl_rl", args.experiment)
    load_run = args.load_run or find_latest_run(log_root)
    if load_run is None:
        sys.exit(f"[ERROR] no runs under {log_root}")
    args.load_run = load_run
    # The preflight runs under --dry_run too: "would this session work" is exactly what a dry
    # run is for, and resolving the pair costs ~0.3 s. It is the cheapest way to find out that
    # exported/ holds two generators, or that the fingerprints disagree.
    allow_mismatch = preflight_pair(os.path.join(log_root, load_run), args.fm_onnx,
                                    args.allow_asset_mismatch)
    if args.joystick and not args.dry_run:
        print(f"[INFO] gamepad: {preflight_gamepad()}  "
              f"(max vx {args.max_vx} / vy {args.max_vy} / wz {args.max_wz})")

    # Under --joystick there is no speed axis: the command is the operator's, so a cell is
    # (width, repeat) and the repeats are different approaches rather than repeated draws.
    if args.joystick:
        cells = [(w, None, 0.0, 0.0, r) for w in widths for r in range(args.repeats)]
    else:
        cells = [(w, v, lat, yw, r)
                 for w in widths for v in speeds
                 for lat in args.lateral for yw in args.yaw
                 for r in range(args.repeats)]
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(cells)
    done = set() if args.redo else load_done(labels_path)
    todo = [c for c in cells if _cell_key(*c) not in done]
    if len(todo) > 60:
        print(f"[WARN] {len(todo)} trials is a long sitting -- roughly "
              f"{len(todo) * 0.75:.0f} minutes at 30 s and a label each. Narrow an axis.")

    print(f"[INFO] session {out_dir}")
    print(f"[INFO] staircase: rise {args.rise} m, run {args.run} m, {args.steps} steps, "
          f"{args.x_start} m approach, "
          f"{'symmetric (up/platform/down)' if not args.no_descent else 'ascend-only'}")
    print(f"[INFO] {len(cells)} cells, {len(done)} already labelled, {len(todo)} to run")
    for w in widths:
        print(f"       width {w:5.2f} m -- {width_note(w)}")
    if args.joystick:
        print(f"       command: GAMEPAD, {args.joystick_time:.0f} s per trial")
    else:
        for v in speeds:
            print(f"       speed {v:5.2f} m/s -- {speed_note(v)}")

    if args.dry_run:
        for w in widths:
            name, path = write_scene(w, args.rise, args.run, args.steps, args.x_start,
                                     args.platform_depth, not args.no_descent,
                                     args.robot_name)
            print(f"[INFO] wrote {path}")
        for w, v, lat, yw, r in todo:
            t = (float(args.joystick_time) if v is None else
                 trial_seconds(v, args.x_start, args.run, args.steps, args.platform_depth,
                               args.settle, args.max_trial_time))
            shown = "GAMEPAD" if v is None else _cmd_label(v, lat, yw)
            print(f"       would run width {w:.3f} {shown} rep {r} for {t:.1f} s")
        return

    try:
        _sweep(args, todo, out_dir, labels_path, allow_mismatch)
    except KeyboardInterrupt:
        # Ctrl-C reaches the whole foreground process group, so the trial subprocess writes its
        # artifacts and THIS process raises here. Every label is already on disk (appended as
        # it was given); what would be lost without this is the figure and the summary.
        print("\n[INFO] interrupted; writing what has been labelled so far.")

    rows = read_rows(labels_path)
    if rows:
        _finish(rows, widths, speeds, out_dir, labels_path)
    else:
        print("[INFO] nothing labelled.")


def _sweep(args, todo: list[tuple], out_dir: str, labels_path: str,
           allow_mismatch: bool) -> None:
    """Run and label every trial in ``todo``, appending each label as it is given."""
    quit_early = False
    for i, (w, v, lat, yw, rep) in enumerate(todo):
        if quit_early:
            break
        while True:
            print(f"\n{'-' * 78}")
            shown = "GAMEPAD" if v is None else _cmd_label(v, lat, yw)
            print(f"trial {i + 1}/{len(todo)}  width {w:.3f} m  {shown}  rep {rep}")
            print(f"    width: {width_note(w)}")
            if v is not None:
                print(f"    speed: {speed_note(v)}")
            # The lateral / yaw suffix appears only when they are non-zero, so a plain sweep
            # keeps the directory names it already had and old sessions stay resumable.
            vpart = "joy" if v is None else f"v{int(round(v * 1000)):04d}"
            for tag, val in (("y", lat), ("z", yw)):
                if abs(val) > 1e-9:
                    vpart += f"_{tag}{int(round(val * 1000)):+05d}"
            trial_dir = os.path.join(
                out_dir, f"w{int(round(w * 1000)):04d}_{vpart}_r{rep}")
            result = run_trial(args, w, v, lat, yw, trial_dir, allow_mismatch)
            if not result["ran"]:
                # No label is asked for here on purpose. A trial that never stepped carries no
                # evidence, and prompting would quietly file the operator's guess about a run
                # that did not happen -- which is worse than losing the cell.
                print(f"    [ERROR] the runner produced no verdict (exit "
                      f"{result['returncode']}). Its output is above; nothing was labelled.")
                if not _ask_continue():
                    quit_early = True
                break
            if result["survived"] is False:
                verdict = (f"FELL at {result['time_to_fall_s']:.2f} s "
                           f"({result['failure_reason']})")
            elif result["survived"]:
                verdict = "SURVIVED" + (
                    f" -- but the run ENDED EARLY at {result['sim_end_time_s']:.1f} s of "
                    f"{result['total_time']:.1f} s (viewer closed?), so this is not "
                    "'got through it'" if result["ended_early"] else "")
            else:
                verdict = "no verdict (interrupted)"
            print(f"\n    ===> runner verdict: {verdict}")
            if result.get("app_n"):
                # Printed BEFORE the prompt on purpose: what the operator commanded on the way
                # in is context for the label they are about to give, and it is the half of a
                # joystick trial that no amount of watching makes visible.
                print(f"    your approach ({result['app_n']} steps before the first riser):"
                      f"  vx med {result['app_vx_med']:+.2f}"
                      f"  |vy| med {result['app_vy_absmed']:.2f} "
                      f"(range {result['app_vy_min']:+.2f}..{result['app_vy_max']:+.2f})"
                      f"  |wz| med {result['app_wz_absmed']:.2f} "
                      f"(range {result['app_wz_min']:+.2f}..{result['app_wz_max']:+.2f})")
                if not result.get("app_reached"):
                    print("    [note] the robot never passed the first riser, so 'approach' "
                          "is the whole trial.")
            label, note = prompt_label()
            if label == "retry":
                continue
            if label == "quit":
                quit_early = True
                break
            if label == "skip":
                break
            row = {"width": w, "speed": v, "joystick": v is None,
                   "lateral": lat, "yaw": yw,
                   "repeat": rep, "label": label, "note": note,
                   "trial_dir": trial_dir, "rise": args.rise, "run": args.run,
                   "steps": args.steps, "x_start": args.x_start,
                   "descend": not args.no_descent,
                   "load_run": args.load_run, "fm_onnx": args.fm_onnx,
                   "stamped": datetime.datetime.now().isoformat(timespec="seconds"),
                   **{k: result[k] for k in ("scene", "total_time", "argv", "returncode",
                                             "survived", "time_to_fall_s", "failure_reason",
                                             "sim_end_time_s", "ended_early")},
                   **{k: val for k, val in result.items() if k.startswith(("app_", "all_"))}}
            with open(labels_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"    -> {label}" + (f" ({note})" if note else ""))
            break


#: Fields that must match across a session's rows for its grid to mean one thing.  Width and
#: speed are the AXES; these are the setup held fixed behind them, and a figure that silently
#: averaged two of any of them would answer no question at all.
_FIXED_SETUP = ("rise", "run", "steps", "x_start", "descend", "load_run", "fm_onnx")


def _warn_on_mixed_setup(rows: list[dict]) -> None:
    """Say so, loudly, when a session's rows do not share one staircase and one policy pair.

    Resuming with a changed ``--x_start`` (or ``--rise``, or a different generator) is an easy
    and reasonable thing to do mid-session, and the grid figure has nowhere to show it: the
    cells would be coloured by labels taken under two different experiments.  ``labels.jsonl``
    keeps every field per row, so the split is always recoverable -- this exists so that
    recovering it is not something you have to think to do.
    """
    joy = {bool(r.get("joystick")) for r in rows}
    if len(joy) > 1:
        print("[WARN] this session mixes GAMEPAD and constant-command trials. They answer "
              "different questions -- one sweeps a speed, the other measures your approach -- "
              "and each gets its own figure. Do not read them as one grid.")
    for field in _FIXED_SETUP:
        seen = {r.get(field) for r in rows if field in r}
        if len(seen) > 1:
            print(f"[WARN] this session mixes {len(seen)} values of {field!r}: "
                  f"{sorted(seen, key=str)}. The grid averages them into one cell colour. "
                  "Split by that field before quoting anything from the figure.")


def _finish(rows: list[dict], widths: list[float], speeds: list[float], out_dir: str,
            labels_path: str) -> None:
    """Write the summary and the grid figure for a session.

    The figure's axes are the UNION of the sweep that was asked for and the values actually
    labelled.  Redrawing a session with ``--plot_only`` and no ``--widths`` / ``--speeds``
    would otherwise use this parser's defaults, which are not that session's axes -- a session
    swept at other values redraws as a blank grid with no complaint.
    """
    widths = sorted({round(float(w), 4) for w in widths}
                    | {round(float(r["width"]), 4) for r in rows})
    # A joystick row carries speed=None -- there was no commanded speed to sweep -- so it
    # contributes no column to a speed axis.
    speeds = sorted({round(float(v), 4) for v in speeds}
                    | {round(float(r["speed"]), 4) for r in rows
                       if r.get("speed") is not None})
    _warn_on_mixed_setup(rows)
    summary = summarise(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] {summary['n']} labels: {summary['by_label']}")
    print(f"[INFO] {labels_path}")
    try:
        joy = [r for r in rows if r.get("joystick")]
        if joy and len(joy) == len(rows):
            drawn = plot_joystick(rows, widths, os.path.join(out_dir, "joystick_labels"))
        elif joy:
            # A mixed session has no single figure: one half has a speed axis and the other
            # replaces it with a measurement. `_warn_on_mixed_setup` has already said so.
            drawn = plot_grid([r for r in rows if not r.get("joystick")], widths, speeds,
                              os.path.join(out_dir, "label_grid"))
            drawn += plot_joystick(joy, widths, os.path.join(out_dir, "joystick_labels"))
        else:
            drawn = plot_grid(rows, widths, speeds, os.path.join(out_dir, "label_grid"))
        for p in drawn:
            print(f"[INFO] wrote {p}")
    except Exception as exc:  # noqa: BLE001 - a plot failure must not lose the labels
        print(f"[WARN] grid figure failed ({exc}); labels.jsonl is intact.")


if __name__ == "__main__":
    main()
