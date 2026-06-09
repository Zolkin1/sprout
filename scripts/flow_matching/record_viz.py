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
    relative_state,
    sample_trajectory,
    state_to_absolute,
    target_state_dim,
)
from robot_rl.flow_matching.dataset import resolve_term_slice
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
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
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


# TODO(partial-history): if the model is ever trained on partial histories
# (see the TODO above episode_windows in robot_rl/flow_matching/dataset.py),
# this buffer must mirror the training-side padding rule instead of waiting
# for a full window.
class BatchedHistory:
    """Fixed-length rolling history of raw world-frame state for ALL envs.

    Holds the most recent ``maxlen`` steps as ``(num_envs, dim)`` tensors,
    storing the same quantities ``record_data.collect_pairs`` recorded, so the
    reconstructed history token matches the training dataset exactly. Env
    resets are handled with a per-env validity counter instead of physically
    clearing slots: a window is only used once an env has ``maxlen``
    uninterrupted steps since its last reset.
    """

    def __init__(self, num_envs: int, maxlen: int, device: torch.device):
        """Create an empty history for ``num_envs`` envs of length ``maxlen``."""
        self.maxlen = maxlen
        self._steps: deque[dict[str, torch.Tensor]] = deque(maxlen=maxlen)
        self._steps_since_reset = torch.zeros(num_envs, dtype=torch.long, device=device)

    def append(self, step: dict[str, torch.Tensor]) -> None:
        """Append one step. ``step`` maps each tracked key to a ``(N, dim)`` tensor."""
        self._steps.append(step)
        self._steps_since_reset += 1

    def invalidate(self, env_mask: torch.Tensor) -> None:
        """Restart the validity counter for envs that just reset (``(N,)`` bool)."""
        self._steps_since_reset[env_mask] = 0

    def valid(self) -> torch.Tensor:
        """Per-env mask of complete, reset-free history windows (``(N,)`` bool)."""
        if len(self._steps) < self.maxlen:
            return torch.zeros_like(self._steps_since_reset, dtype=torch.bool)
        return self._steps_since_reset >= self.maxlen

    def stack(self, key: str) -> torch.Tensor:
        """Stack a tracked quantity over the window into ``(N, maxlen, dim)``."""
        return torch.stack([s[key] for s in self._steps], dim=1)


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

        # -- resolve RL policy checkpoint (mirror play.py) --
        resume_path = None
        if not args_cli.dry_run:
            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            if args_cli.checkpoint:
                resume_path = retrieve_file_path(args_cli.checkpoint)
            else:
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
            f"(vel_dim={ctx['vel_dim']}, scan_dim={ctx['scan_dim']})"
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


def _build_fm_context(env, robot, device) -> dict:
    """Assemble the flow-matching model, normalizers, and resolved I/O layout.

    Uses the loaded checkpoint when available; otherwise (``--dry_run`` with no
    checkpoint) synthesizes a randomly-initialized model with identity normalizers
    sized from the live environment, so the full shape pipeline can be exercised.

    Returns:
        A dict with the model, normalizers, dimensions (H, F, D, Din, nj),
        sampling-step count, joint permutation, joint names, and obs command slices.
    """
    live_joint_names = list(robot.data.joint_names)
    obs_mgr = env.unwrapped.observation_manager
    live_obs_names = list(obs_mgr.active_terms["policy"])
    live_obs_dims = [int(td[0]) for td in obs_mgr.group_obs_term_dim["policy"]]

    # Command/perception slices come from the LIVE obs layout (the vector we read
    # at runtime). velocity_commands is required; height_scan is optional.
    vel_slice = resolve_term_slice(live_obs_names, live_obs_dims, "velocity_commands")
    try:
        scan_slice = resolve_term_slice(live_obs_names, live_obs_dims, "height_scan")
    except KeyError:
        scan_slice = None
    vel_dim = vel_slice[1] - vel_slice[0]
    scan_dim = (scan_slice[1] - scan_slice[0]) if scan_slice is not None else 0

    if _FM_CKPT is not None:
        cfg = FlowMatchingConfig.from_dict(_FM_CKPT["config"])
        nj = int(_FM_CKPT["num_joints"])
        fm_meta = _FM_CKPT["meta"]
        target_norm = Normalizer.from_state_dict(_FM_CKPT["target_normalizer"]).to(device)
        history_norm = Normalizer.from_state_dict(_FM_CKPT["history_normalizer"]).to(device)
        fm_joint_names = list(fm_meta["state_joint_names"])
        # Validate the history token width matches what the model expects.
        expected = target_state_dim(nj) + vel_dim + scan_dim
        if expected != cfg.history_in_dim:
            raise ValueError(
                "Live obs layout does not match the flow-matching checkpoint.\n"
                f"  target_state_dim({nj}) + vel_dim({vel_dim}) + scan_dim({scan_dim}) = {expected}\n"
                f"  but checkpoint history_in_dim = {cfg.history_in_dim}.\n"
                f"  checkpoint obs terms: {list(fm_meta['obs_term_names'])}\n"
                f"  live obs terms      : {live_obs_names}"
            )
    else:
        # --dry_run, no checkpoint: synthesize from the live env.
        nj = len(live_joint_names)
        cfg = FlowMatchingConfig()
        cfg.state_dim = target_state_dim(nj)
        cfg.history_in_dim = cfg.state_dim + vel_dim + scan_dim
        fm_joint_names = live_joint_names
        target_norm = _identity_normalizer(cfg.state_dim, device)
        history_norm = _identity_normalizer(cfg.history_in_dim, device)
        print("[INFO] --dry_run: using a randomly-initialized model and identity normalizers.")

    # Joint permutation: reorder live joints into the model's training order.
    missing = [n for n in fm_joint_names if n not in live_joint_names]
    if missing or len(fm_joint_names) != nj:
        raise ValueError(
            f"Joint-name mismatch between checkpoint and live robot.\n"
            f"  missing in live robot: {missing}\n"
            f"  |checkpoint joints|={len(fm_joint_names)} vs nj={nj}"
        )
    joint_perm = torch.tensor([live_joint_names.index(n) for n in fm_joint_names], device=device)

    model = FlowMatchingTransformer(
        history_in_dim=cfg.history_in_dim, state_dim=cfg.state_dim,
        history_len=cfg.history_len, future_len=cfg.future_len,
        d_model=cfg.d_model, nhead=cfg.nhead, num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers, dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)
    if _FM_CKPT is not None:
        model.load_state_dict(_FM_CKPT["model"])
    model.eval()

    num_steps = args_cli.num_sampling_steps or cfg.num_sampling_steps

    return {
        "model": model, "target_norm": target_norm, "history_norm": history_norm,
        "H": cfg.history_len, "F": cfg.future_len, "D": cfg.state_dim, "Din": cfg.history_in_dim,
        "nj": nj, "num_steps": num_steps, "joint_perm": joint_perm, "joint_names": fm_joint_names,
        "vel_slice": vel_slice, "scan_slice": scan_slice, "vel_dim": vel_dim, "scan_dim": scan_dim,
        "device": device,
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
    obs_policy = obs["policy"]
    vs = ctx["vel_slice"]
    step["vel_cmd"] = obs_policy[:, vs[0]:vs[1]].clone()
    if ctx["scan_slice"] is not None:
        ss = ctx["scan_slice"]
        step["scan"] = obs_policy[:, ss[0]:ss[1]].clone()
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
    valid = history.valid()
    if not bool(valid.any()):
        return None
    env_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    H, F, D, nj = ctx["H"], ctx["F"], ctx["D"], ctx["nj"]
    V = env_ids.shape[0]

    p = history.stack("root_pos")[env_ids]        # (V, H, 3)
    q = history.stack("root_quat")[env_ids]       # (V, H, 4) xyzw
    v = history.stack("root_lin_vel")[env_ids]    # (V, H, 3)
    w = history.stack("root_ang_vel")[env_ids]    # (V, H, 3)
    jp = history.stack("joint_pos")[env_ids]      # (V, H, nj)
    jv = history.stack("joint_vel")[env_ids]      # (V, H, nj)
    vel_cmd = history.stack("vel_cmd")[env_ids]   # (V, H, vel_dim)

    # Reference frame = the most recent (current) step, broadcast over the window.
    p_ref = p[:, H - 1:H].expand(V, H, 3).reshape(V * H, 3)
    q_ref = q[:, H - 1:H].expand(V, H, 4).reshape(V * H, 4)
    rel = relative_state(
        p.reshape(V * H, 3), q.reshape(V * H, 4), v.reshape(V * H, 3), w.reshape(V * H, 3),
        jp.reshape(V * H, nj), jv.reshape(V * H, nj), p_ref, q_ref,
    ).reshape(V, H, D)

    parts = [rel, vel_cmd]
    if ctx["scan_slice"] is not None:
        parts.append(history.stack("scan")[env_ids])
    hist = torch.cat(parts, dim=-1)                       # (V, H, Din)
    hist_n = ctx["history_norm"].normalize(hist)          # (V, H, Din)

    pred_std = sample_trajectory(ctx["model"], hist_n, F, D, ctx["num_steps"])  # (V, F, D)
    pred = ctx["target_norm"].denormalize(pred_std)                             # (V, F, D)

    p_ref_f = p[:, H - 1:H].expand(V, F, 3).reshape(V * F, 3)
    q_ref_f = q[:, H - 1:H].expand(V, F, 4).reshape(V * F, 4)
    world = state_to_absolute(pred.reshape(V * F, D), p_ref_f, q_ref_f, nj)
    return env_ids, {
        "root_pos": world["root_pos_w"].reshape(V, F, 3),
        "root_quat": world["root_quat_w"].reshape(V, F, 4),
        "joint_pos": world["joint_pos"].reshape(V, F, nj),
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
