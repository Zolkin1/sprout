# Copyright 2026 Zachary Olkin. All rights reserved.

"""Record a flow-matching visualization rollout from Isaac Sim.

Runs the RL policy the flow-matching model was trained on, feeds the live
rollout history into the flow-matching model (batched over ALL envs), and
records everything the offline viser viewer needs into a single ``.npz``:
per-step robot states, velocity commands, height-scan ray hits, the predicted
future trajectories converted to absolute world frame, and the terrain mesh.

View the result with ``scripts/flow_matching/visualize.py`` (no Isaac Sim
needed). The recorder itself is meant to be run ``--headless``.

The history token fed to the model is reconstructed to EXACTLY match the
training dataset (see ``robot_rl.flow_matching.dataset.FutureStateDataset``):
per past step ``[relative_state | velocity_commands (| height_scan)]`` with the
reference frame taken from the most recent (current) step.

Examples:
    # Real run once the model is trained:
    python scripts/flow_matching/record_viz.py \
        --fm_checkpoint logs/flow_matching/<run>/best.pt --num_envs 2 \
        --max_steps 500 --headless

    # Structural smoke test with no trained model (CPU, headless):
    python scripts/flow_matching/record_viz.py \
        --dry_run --task <G1-task> --headless --device cpu --num_envs 2 --max_steps 60
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata as metadata
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401 — register IsaacLab tasks
import robot_rl  # noqa: F401 — register robot_rl gym environments
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

from robot_rl.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingTransformer,
    Normalizer,
    target_state_dim,
)
from robot_rl.flow_matching.dataset import resolve_term_slice
from robot_rl.flow_matching.online import (
    BatchedHistory,
    FmOnlineContext,
    build_history_token,
    load_fm_context,
    sample_world,
)
from robot_rl.flow_matching.viz.recording import (
    VIZ_DIR_DEFAULT,
    VizRecordingMeta,
    save_viz_recording,
    unique_output_path,
)

# Make ``cli_args`` (lives in scripts/rsl_rl) importable, mirroring train.py.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, os.pardir, "rsl_rl"))
import cli_args  # noqa: E402  isort: skip
from provenance import auto_clip_device  # noqa: E402  isort: skip

# NOTE: quaternions are (x, y, z, w) throughout this repo fork (robot.data,
# isaaclab.utils.math, state_to_absolute). This differs from stock IsaacLab
# (wxyz); do not "fix" it. No conversion is needed anywhere below.

_DEFAULT_URDF = os.path.join("robot_assets", "g1", "g1_21j", "g1_21j.urdf")


# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Record a flow-matching visualization rollout from Isaac Sim.")
parser.add_argument(
    "--fm_checkpoint", type=str, default=None,
    help="Path to the trained flow-matching .pt checkpoint. Required unless --dry_run.",
)
parser.add_argument("--task", type=str, default=None, help="Task name (default: the FM checkpoint's task).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to record (default 2).")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--num_sampling_steps", type=int, default=None,
    help="Override the flow-matching Euler sampling steps (default: checkpoint value).",
)
parser.add_argument(
    "--update_every", type=int, default=1,
    help="Re-run flow-matching inference every N control steps (default 1). "
         "Predictions carry forward between updates.",
)
parser.add_argument("--max_steps", type=int, default=500, help="Number of control steps to record (default 500).")
parser.add_argument(
    "--output", type=str, default=VIZ_DIR_DEFAULT,
    help=f"Output directory (timestamped viz_<ts>.npz inside) or an explicit .npz path. Default: {VIZ_DIR_DEFAULT}.",
)
parser.add_argument(
    "--urdf", type=str, default=_DEFAULT_URDF,
    help="Robot URDF path stored in the recording for the viser viewer.",
)
parser.add_argument(
    "--dry_run", action="store_true", default=False,
    help="Skip loading trained weights: use a randomly-initialized model and identity normalizers "
         "(synthesized from the live env when no --fm_checkpoint is given). For structural smoke tests.",
)
parser.add_argument(
    "--print_layout", action="store_true", default=False,
    help="Print the resolved layout (H, F, dims, slices, joint names) and exit before launching sim.",
)
parser.add_argument(
    "--clip_device", type=str, default=None,
    help="Device to hold the motion-clip library on (e.g. 'cpu' to keep the multi-GB "
         "library off a small GPU). Default: env device.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
# Capture whether --load_run was explicit before argv is rewritten, so we can default
# the driver policy to the one the FM checkpoint recorded (meta.checkpoint_path).
_EXPLICIT_LOAD_RUN = any(a == "--load_run" or a.startswith("--load_run=") for a in sys.argv)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


# -- flow-matching checkpoint header (CPU) -----------------------------------
# Loaded before sim launch so the task name can default from it and --print_layout
# can run without starting Kit. Stashed in globals and reused inside main().
_FM_CKPT: dict | None = None
_FM_TASK: str | None = None
if args_cli.fm_checkpoint is not None:
    _FM_CKPT = torch.load(args_cli.fm_checkpoint, map_location="cpu", weights_only=False)
    _FM_TASK = str(_FM_CKPT["meta"]["task_name"])
elif not args_cli.dry_run:
    raise SystemExit("[ERROR] --fm_checkpoint is required unless --dry_run is set.")

_RESOLVED_TASK = args_cli.task or _FM_TASK
if _RESOLVED_TASK is None:
    raise SystemExit("[ERROR] No --task given and no --fm_checkpoint to infer it from.")

if args_cli.print_layout and _FM_CKPT is not None:
    _cfg = FlowMatchingConfig.from_dict(_FM_CKPT["config"])
    _m = _FM_CKPT["meta"]
    print("[layout] task            :", _FM_TASK)
    print("[layout] num_joints      :", int(_FM_CKPT["num_joints"]))
    print("[layout] history_len H   :", _cfg.history_len)
    print("[layout] future_len  F   :", _cfg.future_len)
    print("[layout] state_dim   D   :", _cfg.state_dim)
    print("[layout] history_in_dim  :", _cfg.history_in_dim)
    print("[layout] num_sampling    :", _cfg.num_sampling_steps)
    print("[layout] obs_term_names  :", list(_m["obs_term_names"]))
    print("[layout] obs_term_dims   :", list(_m["obs_term_dims"]))
    print("[layout] state_joint_names:", list(_m["state_joint_names"]))
    raise SystemExit(0)


# NOTE: ``BatchedHistory`` is imported from ``robot_rl.flow_matching.online`` (the
# single source of truth shared with the online RL tracking command). See the
# partial-history TODO above ``episode_windows`` in ``flow_matching/dataset.py``:
# this buffer waits for a full window and must mirror any training-side padding rule.


@hydra_task_config(_RESOLVED_TASK, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Launch sim, run the RL policy, and record states + flow-matching predictions."""
    with launch_simulation(env_cfg, args_cli):
        # Patch activate_contact_sensors for recursive traversal (after SimulationApp).
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        # Post-launch imports (require the running app / sim).
        import trimesh
        import warp as wp
        from isaaclab.sim import SimulationContext
        from isaaclab.terrains.terrain_importer import TerrainImporter

        # -- configure env / agent (mirror play.py) --
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Optionally hold the clip library off-GPU (keeps the ~4 GB library + sim +
        # renderer within a small card's VRAM). Set on any command term with the field.
        # Hold the clip library off-GPU when large (auto: cpu for >1 GB). Override with --clip_device.
        auto_clip_device(env_cfg, args_cli.clip_device)

        # -- resolve the RL driver checkpoint --
        # Default to the driver the FM checkpoint was recorded with (meta.checkpoint_path);
        # --checkpoint wins, and an explicit --load_run takes the play.py resolution path.
        resume_path = None
        if not args_cli.dry_run:
            if args_cli.checkpoint:
                resume_path = retrieve_file_path(args_cli.checkpoint)
            elif not _EXPLICIT_LOAD_RUN and _FM_CKPT is not None:
                meta_ckpt = _FM_CKPT["meta"].get("checkpoint_path")
                if meta_ckpt and os.path.isfile(meta_ckpt):
                    resume_path = meta_ckpt
                    print(f"[auto] driver policy from FM checkpoint meta: {meta_ckpt}")
            if resume_path is None:
                log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO] Loading RL policy checkpoint: {resume_path}")

        # -- create environment, capturing the terrain mesh on the way --
        # TerrainImporter.meshes is deprecated (returns {}), so the trimesh handed
        # to import_mesh() is intercepted here instead.
        captured_meshes: dict[str, trimesh.Trimesh] = {}
        orig_import_mesh = TerrainImporter.import_mesh

        def _capturing_import_mesh(self, name: str, mesh: trimesh.Trimesh, **kwargs):
            """Stash every imported terrain mesh, then defer to the original."""
            captured_meshes[name] = mesh
            return orig_import_mesh(self, name, mesh, **kwargs)

        TerrainImporter.import_mesh = _capturing_import_mesh
        try:
            env = gym.make(_RESOLVED_TASK, cfg=env_cfg, render_mode=None)
        finally:
            TerrainImporter.import_mesh = orig_import_mesh
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        device = env.unwrapped.device
        robot = env.unwrapped.scene.articulations["robot"]
        num_envs = env.unwrapped.scene.num_envs

        terrain_data = _terrain_arrays(captured_meshes, env)

        # -- load RL policy --
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        if resume_path is not None:
            runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)

        # -- build the flow-matching context (model, normalizers, layout) --
        ctx = _build_fm_context(env, robot, device)
        ctx["dt"] = float(env.unwrapped.step_dt)
        print(
            f"[INFO] FM ready: H={ctx['H']} F={ctx['F']} D={ctx['D']} Din={ctx['Din']} "
            f"nj={ctx['nj']} sampling_steps={ctx['num_steps']} "
            f"(cmd_dim={ctx['vel_dim']}, future_cmd={'on' if ctx['future_cmd_dim'] > 0 else 'off'}, "
            f"scan_dim={ctx['scan_dim']}, command={'on' if ctx['cmd_term'] is not None else 'none'})"
        )

        # The viewer wants world-frame ray hit points, which only the sensor has
        # (the height_scan obs term is a processed relative height).
        scanner = env.unwrapped.scene.sensors.get("height_scanner")

        # -- staged per-step CPU arrays --
        rows: dict[str, list[np.ndarray]] = {
            "root_pos": [], "root_quat_xyzw": [], "joint_pos": [], "vel_cmd": [],
            "done": [], "pred_index": [],
        }
        if scanner is not None:
            rows["scan_hits"] = []
        preds: dict[str, list[np.ndarray]] = {"root_pos": [], "root_quat": [], "joint_pos": []}
        num_preds = 0
        latest_pred_idx = np.full(num_envs, -1, dtype=np.int32)

        sim = SimulationContext.instance()
        history = BatchedHistory(num_envs, ctx["H"], device)
        obs = env.get_observations()
        step = 0
        try:
            while step < args_cli.max_steps:
                if sim.is_stopped():
                    break

                # Snapshot the current step into the history BEFORE stepping, so
                # buffer[-1] is the current state (the dataset's reference frame).
                snap = _record_step(history, robot, obs, ctx, wp)
                rows["root_pos"].append(snap["root_pos"].cpu().numpy())
                rows["root_quat_xyzw"].append(snap["root_quat"].cpu().numpy())
                rows["joint_pos"].append(snap["joint_pos"].cpu().numpy())
                rows["vel_cmd"].append(snap["vel_cmd"].cpu().numpy())
                if scanner is not None:
                    rows["scan_hits"].append(scanner.data.ray_hits_w.clone().cpu().numpy())

                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        try:
                            policy.reset(dones)
                        except AttributeError:
                            pass

                # Never let a window cross an environment reset.
                done_mask = dones.reshape(-1).bool()
                history.invalidate(done_mask)
                rows["done"].append(done_mask.cpu().numpy().astype(np.uint8))

                if step % args_cli.update_every == 0:
                    result = _infer_predictions(history, ctx)
                    if result is not None:
                        env_ids, pred = result
                        for v, env_id in enumerate(env_ids.cpu().numpy()):
                            preds["root_pos"].append(pred["root_pos"][v].cpu().numpy())
                            preds["root_quat"].append(pred["root_quat"][v].cpu().numpy())
                            preds["joint_pos"].append(pred["joint_pos"][v].cpu().numpy())
                            latest_pred_idx[env_id] = num_preds
                            num_preds += 1

                # Frame `step` shows the carried prediction even if the env resets on
                # the step taken here (the snapshot above is pre-reset); drop it only
                # for the following frames.
                rows["pred_index"].append(latest_pred_idx.copy())
                latest_pred_idx[done_mask.cpu().numpy()] = -1

                step += 1
        except KeyboardInterrupt:
            print(f"[INFO] Interrupted after {step} steps; saving what was recorded.")
        finally:
            env.close()

        if step == 0:
            raise SystemExit("[ERROR] No steps recorded; nothing to save.")
        _save(rows, preds, terrain_data, ctx, num_envs, step, scanner)


def _terrain_arrays(captured_meshes: dict, env) -> dict[str, np.ndarray | str]:
    """Convert the captured terrain mesh to recording arrays.

    Falls back to a flat quad spanning the env origins (plus margin) when the
    task imported no mesh through ``import_mesh`` (e.g. plane or USD terrain).

    Returns:
        Dict with ``vertices``, ``faces``, optional ``vertex_colors``, ``source``.
    """
    mesh = captured_meshes.get("terrain")
    if mesh is not None:
        out: dict[str, np.ndarray | str] = {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32),
            "faces": np.asarray(mesh.faces, dtype=np.int32),
            "source": "generator",
        }
        colors = getattr(mesh.visual, "vertex_colors", None)
        if colors is not None and len(colors) == len(mesh.vertices):
            out["vertex_colors"] = np.asarray(colors, dtype=np.uint8)
        print(f"[INFO] Captured terrain mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
        return out

    origins = env.unwrapped.scene.env_origins.cpu().numpy()
    lo = origins[:, :2].min(axis=0) - 20.0
    hi = origins[:, :2].max(axis=0) + 20.0
    vertices = np.array(
        [[lo[0], lo[1], 0.0], [hi[0], lo[1], 0.0], [hi[0], hi[1], 0.0], [lo[0], hi[1], 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    print("[WARN] No terrain mesh captured; recording a flat quad fallback.")
    return {"vertices": vertices, "faces": faces, "source": "plane_fallback"}


def _resolve_command_term(env):
    """Find the trajectory command term exposing a commanded velocity (``target_vel``).

    Future-command (``cmd_array``) checkpoints condition on the upstream command,
    which at inference is read from the command term rather than the obs vector.
    """
    cmd_mgr = env.unwrapped.command_manager
    for name in cmd_mgr.active_terms:
        term = cmd_mgr.get_term(name)
        if getattr(term, "command_velocity_available", False):
            return term
    raise RuntimeError(
        "This is a future-command (cmd_array) checkpoint, but no command term on the task "
        "exposes a commanded velocity (command_velocity_available / get_command_velocity)."
    )


def _dry_run_fm_context(live_joint_names: list[str], scan_dim: int, device) -> FmOnlineContext:
    """Synthesize a random-weights, no-command :class:`FmOnlineContext` for smoke tests."""
    nj = len(live_joint_names)
    cfg = FlowMatchingConfig()
    d = target_state_dim(nj)
    din = d + scan_dim
    model = FlowMatchingTransformer(
        history_in_dim=din, state_dim=d, history_len=cfg.history_len, future_len=cfg.future_len,
        d_model=cfg.d_model, nhead=cfg.nhead, num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers, dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout, cmd_dim=0,
    ).to(device).eval()
    return FmOnlineContext(
        model=model, target_norm=_identity_normalizer(d, device),
        history_norm=_identity_normalizer(din, device), H=cfg.history_len, F=cfg.future_len,
        D=d, Din=din, nj=nj, num_extra_frames=0,
        num_steps=args_cli.num_sampling_steps or cfg.num_sampling_steps,
        cmd_dim=0, future_cmd_dim=0, scan_dim=scan_dim,
        joint_names=list(live_joint_names), frame_names=[], device=device,
    )


def _build_fm_context(env, robot, device) -> dict:
    """Assemble the flow-matching inference context (delegates math to ``online.py``).

    The model, normalizers, and layout come from :func:`load_fm_context` (which
    handles command-conditioned and extra-frame checkpoints). When the checkpoint is
    command-conditioned, the command stream (past + future horizon) is read from the
    trajectory command term; otherwise there is no command. ``--dry_run`` synthesizes
    a random model + identity normalizers.

    Returns:
        A dict with the :class:`FmOnlineContext` (``fm``), the live->FM joint
        permutation, the command term (``cmd_term`` or None), the optional
        height-scan slice, and convenience dims used by the recorder.
    """
    live_joint_names = list(robot.data.joint_names)
    obs_mgr = env.unwrapped.observation_manager
    live_obs_names = list(obs_mgr.active_terms["policy"])
    live_obs_dims = [int(td[0]) for td in obs_mgr.group_obs_term_dim["policy"]]

    # Height scan (optional) -- the only perceptive input the history token may carry.
    try:
        scan_slice = resolve_term_slice(live_obs_names, live_obs_dims, "height_scan")
    except KeyError:
        scan_slice = None
    scan_dim = (scan_slice[1] - scan_slice[0]) if scan_slice is not None else 0

    if _FM_CKPT is not None:
        fm = load_fm_context(_FM_CKPT, device, args_cli.num_sampling_steps, scan_dim=scan_dim)
    else:
        # --dry_run: random model, no command, no extra frames -- exercises shapes only.
        fm = _dry_run_fm_context(live_joint_names, scan_dim, device)
        print("[INFO] --dry_run: using a randomly-initialized model and identity normalizers.")

    # Extra body frames (ee): resolve their live body indices in FM frame order so the
    # per-step capture mirrors record_data.py --state_frames (body_pos_w / body_quat_w /
    # body_link_vel_w), the exact layout online.build_history_token consumes.
    frame_body_idx = None
    if fm.num_extra_frames > 0:
        body_names = list(robot.data.body_names)
        missing_f = [f for f in fm.frame_names if f not in body_names]
        if missing_f:
            raise ValueError(f"FM extra frames absent from the live robot bodies: {missing_f}")
        frame_body_idx = torch.tensor(
            [body_names.index(f) for f in fm.frame_names], dtype=torch.long, device=device
        )

    # Joint permutation: reorder live joints into the FM training order.
    missing = [n for n in fm.joint_names if n not in live_joint_names]
    if missing or len(fm.joint_names) != fm.nj:
        raise ValueError(
            f"Joint-name mismatch between checkpoint and live robot.\n"
            f"  missing in live robot: {missing}\n"
            f"  |checkpoint joints|={len(fm.joint_names)} vs nj={fm.nj}"
        )
    joint_perm = torch.tensor([live_joint_names.index(n) for n in fm.joint_names], device=device)

    # Command source: command-conditioned checkpoints read the command (past + the
    # decoder's future horizon) from the trajectory command term. Otherwise no command.
    cmd_term = None
    if fm.cmd_dim > 0:
        cmd_term = _resolve_command_term(env)
        got = int(cmd_term.get_command_velocity().shape[-1])
        if got != fm.cmd_dim:
            raise ValueError(f"Command-term velocity width {got} != checkpoint cmd_dim {fm.cmd_dim}.")

    return {
        "fm": fm, "joint_perm": joint_perm, "scan_slice": scan_slice,
        "cmd_term": cmd_term, "frame_body_idx": frame_body_idx,
        "H": fm.H, "F": fm.F, "D": fm.D, "Din": fm.Din, "nj": fm.nj,
        "num_steps": fm.num_steps, "joint_names": fm.joint_names,
        "vel_dim": fm.cmd_dim, "scan_dim": fm.scan_dim, "future_cmd_dim": fm.future_cmd_dim,
        "num_extra_frames": fm.num_extra_frames, "device": device,
    }


def _identity_normalizer(dim: int, device) -> Normalizer:
    """A pass-through normalizer (mean 0, std 1) for dry-run smoke tests."""
    return Normalizer(
        torch.zeros(dim, device=device),
        torch.ones(dim, device=device),
        torch.zeros(dim, dtype=torch.bool, device=device),
    )


def _record_step(history: BatchedHistory, robot, obs, ctx: dict, wp) -> dict[str, torch.Tensor]:
    """Snapshot every env's current raw state + command terms into the history.

    Quantities and ordering match ``record_data.collect_pairs`` so the
    reconstructed history token is identical to the training data.

    Returns:
        The appended step dict of ``(N, dim)`` tensors (for CPU staging).
    """
    perm = ctx["joint_perm"]
    step = {
        "root_pos": wp.to_torch(robot.data.root_pos_w).clone(),
        "root_quat": wp.to_torch(robot.data.root_quat_w).clone(),       # xyzw
        "root_lin_vel": wp.to_torch(robot.data.root_lin_vel_w).clone(),  # com frame
        "root_ang_vel": wp.to_torch(robot.data.root_ang_vel_w).clone(),  # com frame
        "joint_pos": wp.to_torch(robot.data.joint_pos)[:, perm].clone(),
        "joint_vel": wp.to_torch(robot.data.joint_vel)[:, perm].clone(),
    }
    # Extra body frames (ee): world pose + twist, mirroring record_data --state_frames
    # ([pos(3), quat(4 xyzw), body_link_vel(6) = lin+ang] per frame, FM frame order).
    if ctx["frame_body_idx"] is not None:
        idx = ctx["frame_body_idx"]
        fpos = wp.to_torch(robot.data.body_pos_w)[:, idx]        # (N, K, 3)
        fquat = wp.to_torch(robot.data.body_quat_w)[:, idx]      # (N, K, 4) xyzw
        flink = wp.to_torch(robot.data.body_link_vel_w)[:, idx]  # (N, K, 6) [lin, ang]
        step["frames"] = torch.cat([fpos, fquat, flink], dim=-1).clone()  # (N, K, 13)
    # Command feeding the history token. Future-command checkpoints read the
    # upstream command (target_vel) from the command term and also need the future
    # horizon -- captured here (pre-step) so it spans s+1..s+F relative to THIS step.
    if ctx["cmd_term"] is not None:
        step["vel_cmd"] = ctx["cmd_term"].get_command_velocity().clone()                  # (N, cmd_dim)
        step["future_cmd"] = ctx["cmd_term"].get_command_velocity_horizon(ctx["F"]).clone()  # (N, F, cmd_dim)
    if ctx["scan_slice"] is not None:
        ss = ctx["scan_slice"]
        step["scan"] = obs["policy"][:, ss[0]:ss[1]].clone()
    history.append(step)
    return step


@torch.inference_mode()
def _infer_predictions(
    history: BatchedHistory, ctx: dict
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
    """Run batched flow-matching inference for every env with a valid history.

    Builds the normalized history tokens (reference frame = most recent step),
    samples the future trajectories, denormalizes, and converts back to the
    world frame.

    Returns:
        ``(env_ids, pred)`` where ``env_ids`` is ``(V,)`` and ``pred`` maps
        ``root_pos (V, F, 3)``, ``root_quat (V, F, 4)`` (xyzw), and
        ``joint_pos (V, F, nj)``; or None when no env has a full history.
    """
    fm = ctx["fm"]
    valid = history.valid()
    if not bool(valid.any()):
        return None
    env_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    H = fm.H

    p = history.stack("root_pos")[env_ids]        # (V, H, 3)
    q = history.stack("root_quat")[env_ids]       # (V, H, 4) xyzw
    v = history.stack("root_lin_vel")[env_ids]    # (V, H, 3)
    w = history.stack("root_ang_vel")[env_ids]    # (V, H, 3)
    jp = history.stack("joint_pos")[env_ids]      # (V, H, nj)
    jv = history.stack("joint_vel")[env_ids]      # (V, H, nj)
    scan = history.stack("scan")[env_ids] if fm.scan_dim > 0 else None
    frames = history.stack("frames")[env_ids] if fm.num_extra_frames > 0 else None  # (V, H, K, 13)
    # Command (when the checkpoint is command-conditioned): past in the token, future
    # horizon to the decoder. The horizon for the current step is at the last index.
    if fm.cmd_dim > 0:
        cmd = history.stack("vel_cmd")[env_ids]                       # (V, H, cmd_dim)
        future_cmd = history.stack("future_cmd")[env_ids][:, H - 1]   # (V, F, cmd_dim)
    else:
        cmd = p.new_zeros(p.shape[0], H, 0)
        future_cmd = None

    # online.build_history_token does the (training-matched) relative transform +
    # normalization; sample_world samples, denormalizes, and lifts to world frame.
    token = build_history_token(fm, p, q, v, w, jp, jv, cmd, frames=frames, scan=scan)
    world = sample_world(fm, token, p[:, H - 1], q[:, H - 1], future_cmd=future_cmd)
    return env_ids, {
        "root_pos": world["root_pos_w"],
        "root_quat": world["root_quat_w"],
        "joint_pos": world["joint_pos"],
    }


def _save(
    rows: dict[str, list[np.ndarray]],
    preds: dict[str, list[np.ndarray]],
    terrain_data: dict,
    ctx: dict,
    num_envs: int,
    num_steps: int,
    scanner,
) -> None:
    """Stack the staged rows into recording arrays and write the ``.npz``."""
    F, nj = ctx["F"], ctx["nj"]
    data: dict[str, np.ndarray] = {
        "root_pos": np.stack(rows["root_pos"]),
        "root_quat_xyzw": np.stack(rows["root_quat_xyzw"]),
        "joint_pos": np.stack(rows["joint_pos"]),
        "vel_cmd": np.stack(rows["vel_cmd"]),
        "done": np.stack(rows["done"]),
        "pred_index": np.stack(rows["pred_index"]),
        "terrain_vertices": terrain_data["vertices"],
        "terrain_faces": terrain_data["faces"],
    }
    if "vertex_colors" in terrain_data:
        data["terrain_vertex_colors"] = terrain_data["vertex_colors"]
    num_scan_rays = 0
    if "scan_hits" in rows:
        data["scan_hits"] = np.stack(rows["scan_hits"])
        num_scan_rays = data["scan_hits"].shape[2]
    if preds["root_pos"]:
        data["pred_root_pos"] = np.stack(preds["root_pos"])
        data["pred_root_quat_xyzw"] = np.stack(preds["root_quat"])
        data["pred_joint_pos"] = np.stack(preds["joint_pos"])
    else:
        data["pred_root_pos"] = np.zeros((0, F, 3), dtype=np.float32)
        data["pred_root_quat_xyzw"] = np.zeros((0, F, 4), dtype=np.float32)
        data["pred_joint_pos"] = np.zeros((0, F, nj), dtype=np.float32)

    meta = VizRecordingMeta(
        urdf_path=args_cli.urdf,
        task_name=_RESOLVED_TASK,
        fm_checkpoint=args_cli.fm_checkpoint or "",
        recorded_at=datetime.datetime.now().isoformat(timespec="seconds"),
        quat_format="xyzw",
        dry_run=bool(args_cli.dry_run),
        dt=float(ctx["dt"]),
        update_every=int(args_cli.update_every),
        num_envs=num_envs,
        num_steps=num_steps,
        history_len=ctx["H"],
        future_len=F,
        num_joints=nj,
        num_scan_rays=num_scan_rays,
        joint_names=list(ctx["joint_names"]),
        terrain_source=str(terrain_data["source"]),
    )

    if args_cli.output.endswith(".npz"):
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        path = os.path.abspath(args_cli.output)
    else:
        path = unique_output_path(args_cli.output)
    save_viz_recording(path, data, meta)
    size_mb = os.path.getsize(path) / 1e6
    print(
        f"[INFO] Saved {num_steps} steps x {num_envs} envs "
        f"({len(data['pred_root_pos'])} predictions, {num_scan_rays} scan rays) "
        f"to {path} ({size_mb:.1f} MB)."
    )


if __name__ == "__main__":
    main()
