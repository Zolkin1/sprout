#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""One-stop debugger for the flow-matching generator against the pre-made clips.

The question this exists to answer is "did the generator say the wrong thing, or did we
convert it wrong?" -- and to answer it on the SAME three clips every time, so two runs are
comparable. It pins one clip per terrain family (a stair transition, a flat walk, a box
jump-up), rolls the tracker over them twice, plots both, and replays both in viser.

Two rollouts, differing in exactly one config field
(``commands.traj_ref.reference_source``):

``clip``
    The policy tracks the pre-made clip -- the known-good baseline -- while the generator
    runs PASSIVELY every step and its output is recorded. So one rollout gives both "how
    well does the tracker follow a clip" and "what would the FM have said at these exact
    states". Pass ``--no_passive_fm`` for the pure control with no generator loaded at all.

``fm``
    The policy tracks the generator, conditioned on the same clip's ``target_vel`` profile
    and spawned from the same pose. This is the run that has been disappointing.

Stages. Each writes what the next reads, and every argument is persisted in
``<out>/fm_debug.json``, so nothing has to be retyped (and no two stages can disagree)::

    # everything, from a full library (default: prep -> run -> plot -> view)
    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/fm_debug.py \\
        --lib clips/g1_mb_v5_body.h5 \\
        --checkpoint logs/rsl_rl/g1/2026-07-30_13-24-10_depth_v1/model_8200.pt \\
        --fm_checkpoint logs/flow_matching/2026-07-31_11-27-06_depth_fm_v2/best.pt

    # then iterate on the plots / the viewer without touching the GPU
    python scripts/rsl_rl/fm_debug.py --stage plot --out logs/fm_debug/latest
    python scripts/rsl_rl/fm_debug.py --stage view --out logs/fm_debug/latest

``plot`` and ``view`` never import Isaac Lab, so they start in seconds and can run while a
rollout holds the GPU.

Output under ``--out`` (default ``logs/fm_debug/<timestamp>``)::

    fm_debug.json          the manifest: clips chosen, checkpoints, every stage's arguments
    contract_<mode>.txt    the pre-rollout agreement report (see :func:`contract_report`)
    <mode>.npz             the recorded rollout (DataLogger output + world/replay block)
    plots/<mode>/          per-rollout figures, from play_plots.PLOT_REGISTRY
    plots/compare/         clip-driven vs FM-driven, overlaid per case

What to look at, in order:

1. ``contract_<mode>.txt`` -- if the tracker and the generator disagree about the reference
   window, the cameras or the joint order, nothing below means anything.
2. ``plots/<mode>/obs_clip_fm_diff_env*.csv`` -- the reference-observation columns where the
   generator and the clip differ most, ranked. Read within a unit family.
3. ``plots/compare/`` -- where the two rollouts part company.
4. ``--stage view`` with the "FM horizon" and "reference window (observed)" layers both on.
   They must coincide: the window is what the policy sees, reconstructed from the
   observation vector, and the horizon is what the generator predicted. A constant offset
   or a yaw twist between them is a frame bug in ``mdp.reference_window``; agreement moves
   the search to the generator itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

#: The three debug cases. ``family`` matches the terrain manifest's per-tile ``family``
#: field (the per-family discriminator; ``terrain_type_code`` is coarser and would not
#: separate jump-on from jump-off). ``flat`` has no tile at all, hence no family.
DEFAULT_CASES: list[dict[str, Any]] = [
    {"name": "stairs", "type": "transition", "family": "family_stairs_up.h5"},
    {"name": "flat", "type": "flat", "family": None},
    {"name": "box", "type": "box", "family": "family_jump_on.h5"},
]

_MODES = ("clip", "fm")
_DEFAULT_TASK = "G1-tracker-clf-fm-terrain-depth-play"
#: Plots the ``plot`` stage runs by default.
#:
#: Deliberately WITHOUT ``positions`` / ``velocities`` / ``fm_predictions``. Those put one
#: subplot per output CHANNEL -- ~155 of them, most being the ``@root`` body blocks nobody
#: reads here -- so each is a multi-megapixel figure that takes minutes at a 1000-step
#: rollout, and between them they would dominate the stage's runtime. The reference
#: information they carry is in ``trajectory_obs`` (per-block, with the other source
#: overlaid), ``obs_clip_fm_diff`` (the same difference, ranked) and the cross-rollout
#: comparison. Ask for them explicitly with ``--plots`` when you want the full channel dump.
_DEFAULT_PLOTS = [
    "velocity_tracking", "clf", "proprioception", "joint_targets",
    "trajectory_obs", "obs_clip_fm_diff",
]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _manifest_path(out_dir: str) -> str:
    """Path of the run manifest inside ``out_dir``."""
    return os.path.join(out_dir, "fm_debug.json")


def load_manifest(out_dir: str) -> dict:
    """Read the manifest, or fail with the command that would create it.

    Args:
        out_dir: The run directory.

    Returns:
        The manifest dict.

    Raises:
        SystemExit: If no manifest exists there.
    """
    path = _manifest_path(out_dir)
    if not os.path.isfile(path):
        raise SystemExit(
            f"no manifest at {path}. Run the prep stage first:\n"
            f"    python {os.path.relpath(__file__, _REPO)} --stage prep "
            f"--lib <library.h5> --out {out_dir}"
        )
    with open(path) as f:
        return json.load(f)


def save_manifest(out_dir: str, manifest: dict) -> None:
    """Write the manifest, creating ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    with open(_manifest_path(out_dir), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Stage: prep -- freeze three clips into a small library
# ---------------------------------------------------------------------------

def _clip_candidates(lib_path: str) -> dict[str, np.ndarray]:
    """Per-clip selection arrays read straight from the library ``.h5``.

    Returns ``terrain_id``, ``terrain_type_code``, ``clip_len`` and a per-clip ``family``
    resolved through the terrain manifest (``""`` for flat clips, which have no tile).
    """
    import h5py

    with h5py.File(lib_path, "r") as f:
        tid = np.asarray(f["terrain_id"][:]).astype(np.int64)
        code = np.asarray(f["terrain_type_code"][:]).astype(np.int64)
        clip_len = (np.asarray(f["clip_len"][:]).astype(np.int64) if "clip_len" in f
                    else np.full(tid.shape, int(f.attrs["num_steps"]), dtype=np.int64))
        type_names = json.loads(f.attrs["terrain_type_names"])
        manifest = json.loads(f.attrs["terrain_manifest"])
    tile_family = {int(m["terrain_id"]): str(m.get("family", "")) for m in manifest}
    family = np.array([tile_family.get(int(t), "") for t in tid], dtype=object)
    return {"terrain_id": tid, "type_code": code, "clip_len": clip_len,
            "family": family, "type_names": type_names}


def select_case_clips(lib_path: str, cases: list[dict], overrides: dict[str, int]) -> list[dict]:
    """Choose one clip per case: the LONGEST match, so the rollout has the most to compare.

    Deterministic by construction (longest, ties broken by the lowest index), so re-prepping
    the same library reproduces the same three clips without a seed.

    Args:
        lib_path: Source clip library ``.h5``.
        cases: Case specs (``name`` / ``type`` / ``family``); see :data:`DEFAULT_CASES`.
        overrides: ``{case_name: clip_index}`` forcing a specific clip.

    Returns:
        One dict per case with ``name``, ``clip_index``, ``terrain_id``, ``clip_len``,
        ``family``.

    Raises:
        SystemExit: If a case matches no clip in the library.
    """
    c = _clip_candidates(lib_path)
    out: list[dict] = []
    for case in cases:
        name = case["name"]
        if name in overrides:
            i = int(overrides[name])
        else:
            if case["type"] == "flat":
                match = c["terrain_id"] == -1
            else:
                if case["type"] not in c["type_names"]:
                    raise SystemExit(
                        f"case {name!r} wants terrain type {case['type']!r}, but "
                        f"{lib_path} only has {c['type_names']}."
                    )
                match = c["type_code"] == c["type_names"].index(case["type"])
                if case.get("family"):
                    match = match & (c["family"] == case["family"])
            idx = np.nonzero(match)[0]
            if idx.size == 0:
                raise SystemExit(
                    f"case {name!r} (type={case['type']!r}, family={case.get('family')!r}) "
                    f"matches no clip in {lib_path}. Override it with "
                    f"--clip_index {name}=<i>, or drop it with --cases."
                )
            # Longest first, then lowest index -- np.lexsort is stable on the tie.
            i = int(idx[np.argsort(-c["clip_len"][idx], kind="stable")[0]])
        out.append({
            "name": name, "clip_index": i,
            "terrain_id": int(c["terrain_id"][i]),
            "clip_len": int(c["clip_len"][i]),
            "family": str(c["family"][i]) or None,
            "type": c["type_names"][int(c["type_code"][i])],
        })
    return out


def stage_prep(args) -> dict:
    """Pick the case clips, write the subset library, and record both in the manifest."""
    from motion_gen.clips.clip_io import write_clip_subset

    cases = [c for c in DEFAULT_CASES if args.cases is None or c["name"] in args.cases]
    if not cases:
        raise SystemExit(f"--cases {args.cases} matched none of "
                         f"{[c['name'] for c in DEFAULT_CASES]}.")
    overrides = dict(_parse_kv_ints(args.clip_index))
    chosen = select_case_clips(args.lib, cases, overrides)

    clip_lib = os.path.abspath(args.clip_out)
    os.makedirs(os.path.dirname(clip_lib) or ".", exist_ok=True)
    manifest_tiles = write_clip_subset(args.lib, clip_lib, [c["clip_index"] for c in chosen])

    # The subset library re-indexes the clips in the order they were requested, so a case's
    # clip is at its own position -- record it, since that is what the RL side will see.
    for j, case in enumerate(chosen):
        case["subset_index"] = j

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_lib": os.path.abspath(args.lib),
        "clip_lib": clip_lib,
        "cases": chosen,
        "tiles": len(manifest_tiles),
        "steps": max(c["clip_len"] for c in chosen),
    }
    print(f"[prep] {len(chosen)} cases -> {clip_lib}  ({len(manifest_tiles)} terrain tiles "
          f"+ {sum(1 for c in chosen if c['terrain_id'] == -1)} flat)")
    for case in chosen:
        print(f"        {case['name']:8s} clip {case['clip_index']:6d}  "
              f"type={case['type']:<10s} tile={case['terrain_id']:>4d}  "
              f"len={case['clip_len']} steps  family={case['family']}")
    return manifest


def _parse_kv_ints(pairs: list[str] | None) -> list[tuple[str, int]]:
    """Parse ``name=value`` CLI pairs into ``(name, int)`` tuples."""
    out = []
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected name=value, got {item!r}")
        key, _, value = item.partition("=")
        out.append((key.strip(), int(value)))
    return out


# ---------------------------------------------------------------------------
# Stage: run -- dispatch one subprocess per mode
# ---------------------------------------------------------------------------

def stage_run(args, manifest: dict) -> int:
    """Roll out each requested mode in its OWN process.

    One process per mode rather than two envs in one: Isaac Lab does not reliably tear a
    scene down and build another in the same simulation app, and a crash in the second
    rollout would take the first one's bundle with it. Same pattern
    ``compare_sim2sim.py`` uses.
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([
        os.path.join(_REPO, "source", "robot_rl"),
        os.path.join(_REPO, "source", "motion_gen"),
        _HERE,
        os.environ.get("PYTHONPATH", ""),
    ])}
    for mode in args.modes:
        # -u: a rollout is minutes long, and block-buffered progress through a pipe would
        # make it look hung.
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--stage", "run_one",
               "--mode", mode, "--out", args.out, "--task", args.task,
               "--checkpoint", args.checkpoint, "--num_repeats", str(args.num_repeats)]
        if args.fm_checkpoint:
            cmd += ["--fm_checkpoint", args.fm_checkpoint]
        if args.no_passive_fm:
            cmd += ["--no_passive_fm"]
        if args.fm_zero_noise:
            cmd += ["--fm_zero_noise"]
        if not args.log_fm_window:
            cmd += ["--no_log_fm_window"]
        if args.corrupt_obs:
            cmd += ["--corrupt_obs"]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.max_steps is not None:
            cmd += ["--max_steps", str(args.max_steps)]
        if args.device is not None:
            cmd += ["--device", args.device]
        if not args.headless:
            cmd += ["--no_headless"]
        print(f"[run] {mode}: " + " ".join(cmd))
        rc = subprocess.call(cmd, cwd=_REPO, env=env)
        if rc != 0:
            print(f"[run] mode {mode!r} failed (exit {rc}); stopping.")
            return rc
    return 0


# ---------------------------------------------------------------------------
# Stage: run_one -- the actual Isaac Lab rollout (imports Isaac lazily)
# ---------------------------------------------------------------------------

def contract_report(env, cmd, args, manifest: dict) -> str:
    """Everything that has to AGREE before a rollout means anything, as one report.

    Each of these is a silent-failure mode: the pieces load, they predict, and the numbers
    look plausible. The command term raises on some of them at construction; this gathers
    all of them in one place, states the actual values, and is written next to the bundle
    so a stale run can be identified after the fact.

    Args:
        env: The wrapped environment (already built).
        cmd: The ``traj_ref`` command term.
        args: Parsed CLI args.
        manifest: The run manifest.

    Returns:
        The report text (also printed by the caller).
    """
    import torch
    import yaml

    u = env.unwrapped
    lines: list[str] = ["=" * 78, "FM DEBUG CONTRACT REPORT", "=" * 78]

    def _add(section: str, rows: list[tuple[str, Any]]) -> None:
        lines.append(f"\n[{section}]")
        for key, value in rows:
            lines.append(f"  {key:<34s} {value}")

    _add("run", [
        ("task", args.task), ("mode (reference_source)", args.mode),
        ("tracker checkpoint", args.checkpoint),
        ("fm checkpoint", args.fm_checkpoint or "(none -- pure clip control)"),
        ("clip library", manifest["clip_lib"]),
        ("num_envs", u.num_envs), ("steps", manifest["steps"]),
    ])

    om = u.observation_manager
    _add("observation groups (live)", [
        (g, f"{sum(int(np.prod(s)) for s in om.group_obs_term_dim[g])} = "
            f"{list(zip(om.active_terms[g], [int(np.prod(s)) for s in om.group_obs_term_dim[g]]))}")
        for g in om.active_terms
    ])

    # The teacher's own env.yaml, if it sits next to the checkpoint. Catches the real
    # hazard: a policy trained on another branch whose observation layout has since moved.
    trained = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "params", "env.yaml")
    if os.path.isfile(trained):
        try:
            with open(trained) as f:
                cfg = yaml.unsafe_load(f)
            rows = []
            for group in ("policy", "reference"):
                got = list(om.active_terms.get(group, []))
                # A term entry is a dict; the group's own settings are scalars, and a
                # REMOVED term (e.g. the depth teacher dropping height_scan) dumps as None.
                want = [k for k, v in (cfg.get("observations", {}).get(group, {}) or {}).items()
                        if isinstance(v, dict)]
                mark = "OK" if got == want else "MISMATCH"
                rows.append((f"{group} terms", f"{mark}  trained={want}  live={got}"))
            _add("tracker checkpoint vs live env", rows)
        except Exception as exc:  # noqa: BLE001 - a best-effort cross-check
            _add("tracker checkpoint vs live env", [("could not parse", f"{trained}: {exc}")])
    else:
        _add("tracker checkpoint vs live env", [("params/env.yaml", f"not found at {trained}")])

    offsets = cmd.obs_window_offsets()
    max_off = int(offsets.max()) if offsets is not None else 0
    period = int(cmd.cfg.fm_update_every)
    rows = [("reference_window offsets", f"0..{max_off} (n={0 if offsets is None else offsets.numel()})"),
            ("fm_update_every (re-plan)", period)]
    if getattr(cmd, "fm_enabled", False):
        ctx = cmd.ctx
        # The cursor is up to (period - 1) nodes stale when the furthest sample is read.
        needed = max_off + period
        rows += [
            ("horizon F", f"{ctx.F}  (need >= {needed})"),
            ("horizon OK", "OK" if ctx.F >= needed else "TOO SHORT"),
            ("history N (autoregressive)", f"{ctx.N}"),
            ("state_dim / nj / extra frames", f"{ctx.D} / {ctx.nj} / {ctx.num_extra_frames}"),
            ("extero source", ctx.extero_source),
            ("live extero reading shape", tuple(cmd._read_extero().shape)),
            ("recorded fm_snap_every", ctx.fm_snap_every),
            ("num_sampling_steps", ctx.num_steps),
            ("zero_noise", ctx.zero_noise),
            ("frame names", list(ctx.frame_names)),
            ("frames == command ee frames",
             "OK" if list(ctx.frame_names) == list(cmd.resolve_ee_frames(None))
             else f"MISMATCH: command has {cmd.resolve_ee_frames(None)}"),
            # No permutation left to verify: the window's joint block is DEFINED in robot
            # order, so the checkpoint's order has to equal the robot's and _post_init
            # refuses it otherwise. Report the equality itself.
            ("joint order == robot order",
             "OK" if list(ctx.joint_names) == list(cmd.robot.data.joint_names)
             else f"MISMATCH: checkpoint {list(ctx.joint_names)[:3]}... vs robot "
                  f"{list(cmd.robot.data.joint_names)[:3]}..."),
            ("history source", cmd.cfg.history_from),
            ("re-plan period K", cmd.cfg.fm_update_every),
            ("guidance scale", cmd.cfg.fm_guidance_scale),
        ]
    _add("generator", rows)

    if getattr(cmd, "fm_enabled", False) and cmd.ctx.extero_source == "depth":
        try:
            transfer = os.path.join(_REPO, "transfer")
            if transfer not in sys.path:
                sys.path.insert(0, transfer)
            from deploy.depth_obs import assert_depth_specs_agree, parse_depth_spec
            from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
                depth_camera_specs,
                fm_depth_groups,
            )

            # The generator's own cameras when it samples its own grid, else the
            # policy's -- the same scoping FmTrackingCommand._read_extero uses, so
            # this compares the contract the FM is actually being fed.
            live = depth_camera_specs(u, fm_depth_groups(u) or ["policy"])
            fm_spec = parse_depth_spec(cmd.ctx.extero_spec, "fm checkpoint")
            # STRICT here, unlike the deploy: these cameras were BUILT from this checkpoint
            # (play.py reads its resolution and calls add_fm_depth_pair), so they should match
            # exactly. A difference means the pair was built wrong, which is worth surfacing
            # rather than tolerating.
            assert_depth_specs_agree(live, fm_spec, "env", "fm checkpoint",
                                     require_same_resolution=True)
            _add("depth camera contract", [
                ("env vs fm checkpoint", "OK"),
                ("cameras", [c.get("obs_term") or c.get("sensor") for c in live]),
            ])
        except Exception as exc:  # noqa: BLE001 - reported, not fatal: the run still says something
            _add("depth camera contract", [("DISAGREE", str(exc))])

    lib = getattr(cmd, "_clip_lib", None)
    _add("clip library", [
        ("fingerprint", (lib.attrs.get("clip_fingerprint") if lib else None)),
        ("cases", [(c["name"], c["terrain_id"], c["clip_len"]) for c in manifest["cases"]]),
        ("env -> terrain id", u.scene.terrain.env_terrain_id.cpu().tolist()
         if hasattr(u.scene.terrain, "env_terrain_id") else "n/a"),
    ])
    lines.append("=" * 78)
    return "\n".join(lines)


def _apply_env_overrides(env_cfg, args, manifest: dict) -> None:
    """Point the env at the debug library and the requested reference source, in place."""
    clip_lib = manifest["clip_lib"]
    n_cases = len(manifest["cases"])
    env_cfg.scene.num_envs = n_cases * max(1, int(args.num_repeats))
    if hasattr(env_cfg.scene.terrain, "num_rows"):
        env_cfg.scene.terrain.num_rows = 1
    if hasattr(env_cfg.scene.terrain, "clip_library_path"):
        env_cfg.scene.terrain.clip_library_path = clip_lib
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.seed is not None:
        env_cfg.seed = args.seed

    for name, term in vars(env_cfg.commands).items():
        if not hasattr(term, "path"):
            continue
        term.path = clip_lib
        term.clip_device = "cpu"
        # Every episode starts at clip step 0, so env i replays case i whole -- the
        # random start that training uses would make two runs incomparable.
        term.random_start_phase = False
        if hasattr(term, "reference_source"):
            term.reference_source = args.mode
            term.fm_checkpoint = "" if args.no_passive_fm else (args.fm_checkpoint or "")
            term.fm_zero_noise = bool(args.fm_zero_noise)
            if args.mode == "clip":
                # The teacher's anchoring, which is what the clip reference was trained
                # against. The deploy default (0.0) never re-anchors, so leaving it would
                # make the "baseline" a reference the policy has never seen.
                term.ego_anchor_fraction = 1.0
                term.ego_anchor_every = 1
        term.debug_vis = not args.headless
        print(f"[run_one] commands.{name}: source={getattr(term, 'reference_source', 'clip')} "
              f"clip={clip_lib}")

    if args.corrupt_obs:
        for group in ("policy", "reference"):
            if hasattr(env_cfg.observations, group):
                getattr(env_cfg.observations, group).enable_corruption = True
        print("[run_one] observation corruption ON (noise + clipping, as in training)")

    # One episode covers the whole rollout: a mid-run time_out would reset every env onto a
    # fresh (identical) clip and stitch two attempts into one trace.
    steps = int(args.max_steps or manifest["steps"])
    env_cfg.episode_length_s = (steps + 5) * env_cfg.sim.dt * env_cfg.decimation


def stage_run_one(args) -> int:
    """Build the env, roll out one mode, and write ``<out>/<mode>.npz``.

    Imports Isaac Lab only here, so the plot and view stages stay simulator-free.
    """
    manifest = load_manifest(args.out)

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, hydra_args = launcher_parser.parse_known_args(args.app_argv)
    launcher_args.headless = args.headless
    if args.device is not None:
        launcher_args.device = args.device
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(launcher_args).app

    import gymnasium as gym                                                  # noqa: E402
    import torch                                                             # noqa: E402

    import robot_rl.tasks  # noqa: F401, E402  (registers the tasks)
    # Monkey-patches activate_contact_sensors to traverse NESTED bodies; the tasks put
    # contact sensors on paths the non-recursive default never reaches. Must be imported
    # after the SimulationApp exists -- same as play.py.
    import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401, E402
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
    from isaaclab_tasks.utils.hydra import hydra_task_config                 # noqa: E402
    from rsl_rl.runners import OnPolicyRunner                                # noqa: E402

    from data_logger import DataLogger                                       # noqa: E402

    status = {"rc": 1}

    @hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
    def _main(env_cfg, agent_cfg) -> None:
        """Roll out one mode under the hydra-resolved task config."""
        _apply_env_overrides(env_cfg, args, manifest)
        import importlib.metadata as _md
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))

        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg),
                                 clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        cmd = u.command_manager.get_term("traj_ref")

        report = contract_report(env, cmd, args, manifest)
        print(report)
        with open(os.path.join(args.out, f"contract_{args.mode}.txt"), "w") as f:
            f.write(report + "\n")

        _check_case_coverage(u, manifest)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # Same rule as play.py: the tracker's play runner has no privileged observation
        # group to build a matching critic from, and nothing here evaluates the value
        # function. Skipping the critic load is what lets ANY teacher checkpoint drive this
        # task (the actor's groups are identical across tracking modes; only the critic's
        # width moves). The actor is still loaded strictly, so a real policy mismatch fails.
        if getattr(agent_cfg, "load_critic", True):
            runner.load(args.checkpoint)
        else:
            print("[run_one] play-only runner: loading actor weights only (no critic).")
            runner.load(args.checkpoint, load_cfg={"actor": True, "critic": False,
                                                   "optimizer": False, "iteration": True,
                                                   "rnd": True})
        policy = runner.get_inference_policy(device=u.device)

        logger = DataLogger(env, log_fm_window=args.log_fm_window, log_world=True)
        steps = int(args.max_steps or manifest["steps"])
        obs = env.get_observations()
        if isinstance(obs, tuple):          # rsl-rl returns (obs, extras) in some versions
            obs = obs[0]
        print(f"[run_one] rolling out {steps} steps x {u.num_envs} envs ({args.mode})")
        for t in range(steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            # ``obs`` is now computed from the post-step state, which is the same state
            # every other channel in this row describes -- see DataLogger.collect_step.
            logger.collect_step(env, policy_obs=obs)
            if (t + 1) % 100 == 0:
                print(f"[run_one]   {t + 1}/{steps}")

        data, metadata = logger.finalize()
        metadata["cases"] = manifest["cases"]
        metadata["mode"] = args.mode
        metadata["checkpoint"] = os.path.abspath(args.checkpoint)
        metadata["fm_checkpoint"] = (os.path.abspath(args.fm_checkpoint)
                                     if args.fm_checkpoint and not args.no_passive_fm else None)
        out_path = os.path.join(args.out, f"{args.mode}.npz")
        np.savez_compressed(out_path, meta_json=np.asarray(json.dumps(metadata, default=str)),
                            **{k: v for k, v in data.items()})
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[run_one] wrote {out_path} ({size_mb:.1f} MB, {len(data)} arrays)")
        env.close()
        status["rc"] = 0

    try:
        _main()
    finally:
        app.close()
    return status["rc"]


def _check_case_coverage(unwrapped, manifest: dict) -> None:
    """Fail unless every case tile got an env and no env landed on an unknown tile.

    The importer decides the env -> tile mapping, and a silent gap (two cases on one tile,
    or a tile with no env) would produce a bundle that quietly omits a case.
    """
    importer = getattr(unwrapped.scene, "terrain", None)
    if importer is None or not hasattr(importer, "env_terrain_id"):
        return
    got = set(int(t) for t in importer.env_terrain_id.cpu().tolist())
    want = set(int(c["terrain_id"]) for c in manifest["cases"])
    if got != want:
        raise RuntimeError(
            f"env -> tile coverage is wrong: envs are on tiles {sorted(got)} but the cases "
            f"need {sorted(want)}. num_envs must be a multiple of the case count "
            f"({len(manifest['cases'])}) and the subset library must build exactly those "
            "tiles -- check generator_config.counts.flat in the subset library."
        )


# ---------------------------------------------------------------------------
# Bundles (shared by plot + view; no Isaac Lab)
# ---------------------------------------------------------------------------

def load_bundle(out_dir: str, mode: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load one recorded rollout as ``(data, metadata)``.

    Raises:
        SystemExit: If that mode has not been run.
    """
    path = os.path.join(out_dir, f"{mode}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"no rollout at {path}; run `--stage run --modes {mode}` first.")
    with np.load(path, allow_pickle=True) as f:
        meta = json.loads(str(f["meta_json"]))
        data = {k: f[k] for k in f.files if k != "meta_json"}
    return data, meta


def case_env_ids(meta: dict, case_name: str) -> list[int]:
    """Env indices replaying ``case_name``, via the terrain id its clip belongs to."""
    cases = {c["name"]: c for c in meta.get("cases", [])}
    if case_name not in cases:
        return []
    tid = int(cases[case_name]["terrain_id"])
    env_tid = meta.get("env_terrain_id") or []
    return [i for i, t in enumerate(env_tid) if int(t) == tid]


def available_bundles(out_dir: str) -> list[str]:
    """Modes that have a recorded rollout in ``out_dir``."""
    return [m for m in _MODES if os.path.isfile(os.path.join(out_dir, f"{m}.npz"))]


# ---------------------------------------------------------------------------
# Stage: plot
# ---------------------------------------------------------------------------

def stage_plot(args) -> int:
    """Per-rollout figures from ``play_plots``, plus the cross-rollout comparison."""
    from play_plots import run_plots

    modes = [m for m in args.modes if os.path.isfile(os.path.join(args.out, f"{m}.npz"))]
    if not modes:
        raise SystemExit(f"no rollouts in {args.out}; run `--stage run` first.")

    time_range = None
    if args.plot_time_range:
        parts = args.plot_time_range.split(",")
        if len(parts) != 2:
            raise SystemExit(f"--plot_time_range expects 'start,end'; got "
                             f"{args.plot_time_range!r}")
        time_range = (float(parts[0]) if parts[0].strip() else None,
                      float(parts[1]) if parts[1].strip() else None)

    bundles: dict[str, tuple[dict, dict]] = {}
    for mode in modes:
        data, meta = load_bundle(args.out, mode)
        bundles[mode] = (data, meta)
        save_dir = os.path.join(args.out, "plots", mode)
        # One representative env per case (the first replica), so the figure count stays
        # readable when --num_repeats > 1.
        env_ids = [ids[0] for c in meta.get("cases", [])
                   if (ids := case_env_ids(meta, c["name"]))]
        print(f"[plot] {mode}: envs {env_ids} -> {save_dir}"
              + (f"  window {time_range}" if time_range else ""))
        run_plots(data, meta, save_dir, args.plots, env_ids or [0], time_range=time_range)

    if len(bundles) > 1:
        plot_mode_comparison(bundles, os.path.join(args.out, "plots", "compare"))
    return 0


def plot_mode_comparison(bundles: dict[str, tuple[dict, dict]], save_dir: str) -> None:
    """Overlay the clip-driven and FM-driven rollouts, one figure per case.

    The per-rollout plots say how each run went; this says WHERE they part. Both rollouts
    start from the same pose on the same clip and are handed the same velocity profile, so
    up to the point the FM sends the robot somewhere else the two traces should sit on top
    of each other.

    Five panels: commanded vs achieved forward speed, base height, CLF value (log), the
    applied-reference-vs-actual base error, and the generator-vs-clip reference gap.
    Termination (the first ``done``) is marked, so "the FM run fell at 4.2 s" is legible
    without reading the stats file.

    Every reference distance here reads the look-ahead window's sample 0 -- the ANCHORED
    reference, re-datumed onto the robot each step. ``y_des``'s base channels are a
    different quantity on the clip path (the globally-placed clip, which the drift
    termination grades), and using them would report accumulated drift as reference error.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    colors = {"clip": "tab:blue", "fm": "tab:red"}
    any_meta = next(iter(bundles.values()))[1]
    dt = float(any_meta.get("dt") or 0.02)

    for case in any_meta.get("cases", []):
        name = case["name"]
        fig, axs = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
        fig.suptitle(f"clip-driven vs FM-driven -- case {name} "
                     f"(clip {case['clip_index']}, tile {case['terrain_id']}, "
                     f"{case['clip_len']} steps)", fontsize=13)
        drew = False
        for mode, (data, meta) in bundles.items():
            ids = case_env_ids(meta, name)
            if not ids:
                continue
            e = ids[0]
            drew = True
            col = colors.get(mode, "k")
            t = np.arange(data["joint_pos"].shape[0]) * dt

            if "target_vel" in data and "robot_base_vel" in data:
                if mode == list(bundles)[0]:      # the command is identical in both runs
                    axs[0].plot(t, data["target_vel"][:, e, 0], color="0.35", lw=2.0,
                                ls="--", label="commanded vx")
                axs[0].plot(t, data["robot_base_vel"][:, e, 0], color=col, lw=1.4,
                            label=f"{mode}: achieved vx")
            if "root_pos_w" in data:
                axs[1].plot(t, data["root_pos_w"][:, e, 2], color=col, lw=1.4, label=mode)
            if "v" in data:
                axs[2].semilogy(t, np.maximum(data["v"][:, e], 1e-9), color=col, lw=1.4, label=mode)
            err = _base_position_error(data, e)
            if err is not None:
                axs[3].plot(t, err, color=col, lw=1.4, label=mode)
            gap = _source_gap(data, e)
            if gap is not None:
                axs[4].plot(t, gap, color=col, lw=1.4, label=f"{mode}-driven")

            # Where this rollout ended, if it did -- everything after the line is the
            # NEXT episode, which the shared clip start makes look deceptively similar.
            if "dones" in data:
                ended = np.nonzero(data["dones"][:, e].astype(bool))[0]
                if ended.size:
                    for ax in axs:
                        ax.axvline(ended[0] * dt, color=col, ls=":", lw=1.6)
                    axs[0].plot([], [], color=col, ls=":", lw=1.6,
                                label=f"{mode}: episode ended")

        if not drew:
            plt.close(fig)
            continue
        for ax, label in zip(axs, ("forward velocity [m/s]", "base height [m]",
                                   "CLF V (log)", "|applied ref - actual| [m]",
                                   "|FM ref - clip ref| [m]")):
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")
        axs[-1].set_xlabel("time [s]")
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        path = os.path.join(save_dir, f"compare_{name}.png")
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {path}")


def _base_position_error(data: dict[str, np.ndarray], env_id: int) -> np.ndarray | None:
    """``|applied reference base - actual base|`` over time for one env.

    Sample 0 of the APPLIED reference window (the anchored current step), not ``y_des``:
    the window is the same quantity for both sources, whereas ``y_des``'s base is the
    globally-placed clip on the clip path and the generator's own prediction on the FM one.
    """
    for key in ("refwin_obs_base_pos", "refwin_applied_base_pos"):
        if key in data and "root_pos_w" in data:
            ref = data[key][:, env_id, 0, :]
            return np.linalg.norm(ref - data["root_pos_w"][:, env_id, :], axis=-1)
    return None


def _source_gap(data: dict[str, np.ndarray], env_id: int) -> np.ndarray | None:
    """``|generator reference - clip reference|`` at the current step, both anchored.

    The headline FM-vs-clip number. In the clip-driven rollout it is measured at states the
    baseline reached, so it is the generator's error against a reference the robot was
    actually tracking; in the FM-driven one the robot has already followed the generator, so
    it also carries wherever that took it.
    """
    if "refwin_fm_base_pos" not in data or "refwin_clip_base_pos" not in data:
        return None
    fm = data["refwin_fm_base_pos"][:, env_id, 0, :]
    clip = data["refwin_clip_base_pos"][:, env_id, 0, :]
    return np.linalg.norm(fm - clip, axis=-1)


# ---------------------------------------------------------------------------
# Stage: view
# ---------------------------------------------------------------------------

def stage_view(args) -> int:
    """Serve the recorded rollouts in viser (no Isaac Lab)."""
    from fm_debug_scene import serve

    modes = [m for m in args.modes if os.path.isfile(os.path.join(args.out, f"{m}.npz"))]
    if not modes:
        raise SystemExit(f"no rollouts in {args.out}; run `--stage run` first.")
    bundles = {m: load_bundle(args.out, m) for m in modes}
    return serve(bundles, urdf_path=args.urdf, port=args.port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The full argument surface, shared by every stage."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--stage", default="all",
                   choices=["all", "prep", "run", "run_one", "plot", "view"],
                   help="Which stage to run. 'all' = prep (if needed) -> run -> plot -> view. "
                        "'run_one' is the per-mode worker 'run' spawns; you rarely call it.")
    p.add_argument("--out", default=None,
                   help="Run directory (default: logs/fm_debug/<timestamp>, or the newest "
                        "existing one for the plot/view stages).")
    p.add_argument("--modes", default="clip,fm",
                   help="Comma-separated rollouts: clip (baseline + passive FM) and/or fm.")
    p.add_argument("--mode", default=None, choices=list(_MODES),
                   help="run_one only: the single mode to roll out.")

    prep = p.add_argument_group("prep")
    prep.add_argument("--lib", default=None,
                      help="Source clip library .h5 to pick the three debug clips from.")
    prep.add_argument("--clip_out", default="clips/fm_debug_v1.h5",
                      help="Where to write the frozen 3-clip library.")
    prep.add_argument("--cases", default=None,
                      help=f"Comma-separated subset of "
                           f"{[c['name'] for c in DEFAULT_CASES]}.")
    prep.add_argument("--clip_index", action="append", default=None, metavar="NAME=I",
                      help="Force a case onto a specific source-library clip index. Repeatable.")
    prep.add_argument("--reprep", action="store_true",
                      help="Re-run prep even if the run directory already has a manifest.")

    run = p.add_argument_group("run")
    run.add_argument("--task", default=_DEFAULT_TASK, help="Play task id.")
    run.add_argument("--checkpoint", default=None, help="Tracker policy .pt. Required to run.")
    run.add_argument("--fm_checkpoint", default=None, help="Flow-matching .pt. Required to run.")
    run.add_argument("--no_passive_fm", action="store_true",
                     help="Clip mode only: do not load the generator at all (the pure "
                          "control). Makes the clip rollout faster and FM-free, at the cost "
                          "of the FM-vs-clip comparison.")
    run.add_argument("--fm_zero_noise", action="store_true",
                     help="Integrate the flow from x0=0 so the prediction is deterministic "
                          "given the conditioning. Off the training distribution -- use it "
                          "to remove the sampling draw from an A/B, not to judge behaviour.")
    run.add_argument("--corrupt_obs", action="store_true",
                     help="Re-enable observation corruption (noise + clipping) on the actor's "
                          "groups, which the play trims switch off. Turns the viewer's "
                          "'observed' reference window into something that differs from the "
                          "clean one -- i.e. shows how much of the reference the training-time "
                          "noise actually eats. Off by default: a noise-free rollout is the "
                          "right baseline for judging the generator.")
    run.add_argument("--num_repeats", type=int, default=1,
                     help="Envs per case. >1 gives independent generator noise draws on the "
                          "same clip; the plots still show the first replica.")
    run.add_argument("--max_steps", type=int, default=None,
                     help="Override the rollout length (default: the longest case clip).")
    run.add_argument("--log_fm_window", action="store_true", default=True,
                     help="Log the whole predicted horizon in the y_des layout, for the "
                          "fm_predictions plot. On by default; the bundle is a few tens of MB.")
    run.add_argument("--no_log_fm_window", dest="log_fm_window", action="store_false")
    run.add_argument("--headless", action="store_true", default=True,
                     help="Run without a viewport (default). --no_headless opens the Isaac "
                          "GUI with the reference ribbons drawn.")
    run.add_argument("--no_headless", dest="headless", action="store_false")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--device", default=None, help="Sim device, e.g. cuda:0 or cpu.")

    out = p.add_argument_group("plot / view")
    out.add_argument("--plots", default=",".join(_DEFAULT_PLOTS),
                     help="Comma-separated play_plots names, or 'all'. The default set "
                          "omits the per-channel grids (positions / velocities / "
                          "fm_predictions) -- add them by name when you want the full dump; "
                          "each takes minutes at a 1000-step rollout.")
    out.add_argument("--plot_time_range", default=None, metavar="START,END",
                     help="Restrict the plots to a time window in seconds, e.g. '4,9' or "
                          "'6,' -- the usual way to use this tool once the comparison has "
                          "told you WHEN things go wrong. Also the fix for slow plotting: "
                          "the per-channel figures scale with the window, not the rollout.")
    out.add_argument("--port", type=int, default=8080, help="viser port.")
    out.add_argument("--urdf", default=None, help="G1 URDF (default: the repo's g1_21j).")
    out.add_argument("--no_view", action="store_true",
                     help="'all' stage: stop after the plots instead of serving viser.")
    return p


def _resolve_out_dir(args) -> str:
    """Pick the run directory: the given one, the newest existing, or a fresh timestamp."""
    if args.out:
        return args.out
    root = os.path.join(_REPO, "logs", "fm_debug")
    if args.stage in ("plot", "view") and os.path.isdir(root):
        runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if runs:
            return os.path.join(root, runs[-1])
    return os.path.join(root, time.strftime("%Y-%m-%d_%H-%M-%S"))


def main() -> int:
    """Dispatch the requested stage."""
    parser = build_parser()
    args, extra = parser.parse_known_args()
    args.app_argv = extra                       # forwarded to AppLauncher / hydra in run_one
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.plots = [s.strip() for s in args.plots.split(",") if s.strip()]
    if args.cases:
        args.cases = [s.strip() for s in args.cases.split(",") if s.strip()]
    args.out = _resolve_out_dir(args)
    for mode in args.modes:
        if mode not in _MODES:
            raise SystemExit(f"unknown mode {mode!r}; expected one of {_MODES}.")

    if args.stage == "run_one":
        if args.mode is None:
            raise SystemExit("--stage run_one needs --mode.")
        return stage_run_one(args)

    have_manifest = os.path.isfile(_manifest_path(args.out))
    # ``run`` preps too when it was given a library and has no manifest -- passing --lib and
    # being told to go and prep first would be a pure papercut.
    if args.stage in ("all", "prep") or (args.stage == "run" and args.lib and not have_manifest):
        if args.stage == "prep" or args.reprep or not have_manifest:
            if not args.lib:
                raise SystemExit("--lib is required to prep the debug clip library.")
            save_manifest(args.out, stage_prep(args))
        else:
            print(f"[prep] reusing {_manifest_path(args.out)} (pass --reprep to rebuild)")
        if args.stage == "prep":
            return 0

    if args.stage in ("all", "run"):
        manifest = load_manifest(args.out)
        for name, value in (("--checkpoint", args.checkpoint),):
            if not value:
                raise SystemExit(f"{name} is required to roll out.")
        if args.no_passive_fm and "fm" in args.modes:
            raise SystemExit(
                "--no_passive_fm drops the generator entirely, so there is nothing for "
                "--modes fm to track. Use --modes clip for the FM-free control."
            )
        if not args.fm_checkpoint and not args.no_passive_fm:
            raise SystemExit(
                "--fm_checkpoint is required (there is deliberately no default: a stale "
                "generator loads, predicts, and is silently wrong). Pass --no_passive_fm "
                "with --modes clip for the FM-free control."
            )
        manifest.update({"task": args.task, "checkpoint": os.path.abspath(args.checkpoint),
                         "fm_checkpoint": (os.path.abspath(args.fm_checkpoint)
                                           if args.fm_checkpoint else None),
                         "modes": args.modes, "num_repeats": args.num_repeats})
        save_manifest(args.out, manifest)
        rc = stage_run(args, manifest)
        if rc != 0:
            return rc
        if args.stage == "run":
            return 0

    if args.stage in ("all", "plot"):
        rc = stage_plot(args)
        if rc != 0 or args.stage == "plot":
            return rc

    if args.stage == "view" or (args.stage == "all" and not args.no_view):
        return stage_view(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
