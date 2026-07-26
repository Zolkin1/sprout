# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play a trained RSL-RL policy and record (observation, action) pairs to disk.

The recorded dataset is intended for downstream supervised learning. Each row is
one (policy observation, action) pair, where the action is exactly what the
policy produced from that observation. Alongside the pairs we record:
  - ``terminated``: 1 if the pair led to a true termination (failure), excluding
    time-outs -- i.e. the last pair before the robot terminated.
  - ``done`` / ``time_out``: the combined episode-end flag and the time-out flag,
    so the consumer can distinguish failures from episode-length cutoffs.
  - ``episode_id`` / ``step_in_episode``: bookkeeping for episode-aware splits.
Plus provenance metadata (which checkpoint, its SHA-256, task, obs/action layout)
so the data is self-describing. See ``recorded_data_io.py`` for the on-disk
format and the loader used by the learning script.

Specify the total number of pairs with ``--num_pairs``; every run writes a new,
non-overwriting file under ``<log_dir>/recorded_data/`` (override with
``--output_dir``).

Example:
    python scripts/rsl_rl/record_data.py --task <Task> --num_envs 64 \
        --num_pairs 100000 --load_run <run> --headless
"""

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: E402  height_scan obs fn
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from export_parameters import export_policy_parameters  # isort: skip
from recorded_data_io import (  # isort: skip
    RecordingMeta,
    RecordedWindowWriter,
    fingerprint_arrays,
    sha256_file,
    unique_output_path,
    save_recorded_data,
)
from provenance import auto_clip_device, git_snapshot  # isort: skip
from perturbation_scheduler import PerturbationScheduler  # isort: skip
from isaaclab.utils.io import dump_yaml  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Play a trained RSL-RL policy and record (obs, action) pairs for supervised learning."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument(
    "--num_pairs", type=int, default=None,
    help="LEGACY per-step .npz mode: total (observation, action) pairs to record "
         "(buffers all steps in RAM). Pass exactly one of --num_pairs or --num_windows.",
)
parser.add_argument(
    "--num_windows", type=int, default=None,
    help="WINDOWED mode: total FM snap windows to stream to a chunked .h5 (bounded RAM; "
         "only the snaps the FM actually trains on are stored). Pass exactly one of "
         "--num_pairs or --num_windows.",
)
parser.add_argument(
    "--history_len", type=int, default=8,
    help="[--num_windows] History-state rows H stored per window. Must be >= the largest "
         "num_history_states you will train the FM with (this is the one irreversible "
         "record-time choice). Default 8.",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Where to write the dataset. Default: <log_dir>/recorded_data.",
)
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions stochastically. Default is deterministic (mean) actions.",
)
parser.add_argument(
    "--ref_command_name", type=str, default=None,
    help="Name of the trajectory command term to record the reference state from. "
         "Default: auto-detect the command term exposing get_reference_world_state.",
)
parser.add_argument(
    "--ref_base_frame", type=str, default="pelvis_link",
    help="Trajectory output frame used as the robot root when reconstructing the "
         "reference state (must expose pos_x/y/z and ori_x/y/z/w outputs).",
)
parser.add_argument(
    "--state_frames", type=str, nargs="*", default=None,
    help="Extra robot body frames whose world pose/velocity are appended to the "
         "recorded 'state' (and 'ref_state') vectors beyond root+joints -- e.g. the "
         "feet/hands the flow-matching CLF tracks. Default: auto-detect the trajectory "
         "command's end-effector output frames (ankles/wrists). Pass nothing to record "
         "root+joints only.",
)
parser.add_argument(
    "--ref_future_len", type=int, default=40,
    help="Number of FUTURE nodes F to record per step in 'ref_future' (the ego-anchored "
         "clip future = the robot-state-conditioned FM target). Match the FM future_len.",
)
parser.add_argument(
    "--no_ref_future", action="store_true", default=False,
    help="Disable recording the ego-anchored clip future ('ref_future'/'cmd_future'/"
         "'reanchor_active'). By default it is recorded whenever the command exposes the "
         "look-ahead accessors (get_future_world_base) -- the data the new FM trains on.",
)
parser.add_argument(
    "--clip", type=str, default=None,
    help="Override the clip library the teacher tracks (sets commands.traj_ref.path), e.g. "
         "clips/g1_recovery_v1.h5. Default: the task config's clip (clips/g1_flat_v1.h5 for the "
         "teacher tasks). The recorded distribution mirrors THIS clip, so point it at the library "
         "you generated.",
)
parser.add_argument(
    "--clip_device", type=str, default=None,
    help="Device to hold the motion-clip library on (e.g. 'cpu' to keep the multi-GB "
         "library off a small GPU; gathers copy only the per-env slice). Default: env device.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
# -- Perturbation / recovery-data options (see perturbation_scheduler.py) --
parser.add_argument(
    "--perturb", action="store_true", default=False,
    help="Gather recovery data: run a clean fraction of envs (play DR) and a perturbed "
         "fraction with windowed action-noise + push toggled on/off. Logs a per-step "
         "'perturb_active' flag. Record on a PLAY task (e.g. G1-teacher-clf-play) so the "
         "clean group is play-DR and perturbations are layered per-env in the loop.",
)
parser.add_argument("--clean-fraction", type=float, default=0.5,
                    help="[--perturb] Fraction of envs that stay clean (never perturbed).")
parser.add_argument("--perturb-on-range-s", type=float, nargs=2, default=(0.3, 1.0),
                    metavar=("MIN", "MAX"), help="[--perturb] Seconds a perturbation window stays ON.")
parser.add_argument("--perturb-off-range-s", type=float, nargs=2, default=(1.5, 4.0),
                    metavar=("MIN", "MAX"), help="[--perturb] Seconds between windows (recovery).")
parser.add_argument("--action-noise-scale", type=float, default=0.3,
                    help="[--perturb] Std of Gaussian noise added to actions while ON (0 disables).")
parser.add_argument("--push-vel", type=float, default=0.75,
                    help="[--perturb] Magnitude (m/s) of the planar x/y push impulse on each "
                         "OFF->ON edge (0 disables pushes).")
parser.add_argument("--obs-noise-scale", type=float, default=0.0,
                    help="[--perturb] Std of Gaussian noise added to the policy obs while ON "
                         "(0 disables; the recorded state is always ground truth).")
parser.add_argument("--ood-dr", action="store_true", default=False,
                    help="[--perturb] Re-apply the perturbed group's DR terms with widened "
                         "(out-of-distribution) ranges at each reset.")
parser.add_argument("--ood-dr-scale", type=float, default=1.5,
                    help="[--perturb] Range-widening factor for --ood-dr (1.5 = 50%% wider).")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_resume_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the checkpoint path the same way play.py does, without the simulator.

    Args:
        agent_cfg: The RSL-RL agent configuration (already CLI-updated).

    Returns:
        Absolute path to the checkpoint file to load.
    """
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    if getattr(args_cli, "teacher_experiment", None):
        play_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment))
    else:
        play_log_path = log_root_path
    return get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _resolve_ref_command(env: RslRlVecEnvWrapper, explicit_name: str | None):
    """Find the trajectory command term used to reconstruct the reference state.

    Args:
        env: The wrapped Isaac Lab environment.
        explicit_name: Command term name from the CLI, or ``None`` to auto-detect
            the first term exposing ``get_reference_world_state``.

    Returns:
        ``(name, term)`` for the resolved trajectory command, or ``(None, None)``
        if the task has no such command (e.g. velocity-only tasks).

    Raises:
        ValueError: If ``explicit_name`` is given but the term cannot provide a
            reference state.
    """
    cmd_mgr = env.unwrapped.command_manager
    if explicit_name is not None:
        term = cmd_mgr.get_term(explicit_name)
        if not hasattr(term, "get_reference_world_state"):
            raise ValueError(
                f"Command term '{explicit_name}' does not expose get_reference_world_state."
            )
        return explicit_name, term
    for name in cmd_mgr.active_terms:
        term = cmd_mgr.get_term(name)
        if hasattr(term, "get_reference_world_state"):
            return name, term
    return None, None


def _build_recording_meta(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
    log_dir: str,
    num_pairs: int,
    state_frames: list[str],
) -> RecordingMeta:
    """Capture provenance and obs/action layout metadata once at the start.

    Args:
        env: The wrapped Isaac Lab environment.
        agent_cfg: The RSL-RL agent configuration.
        resume_path: Absolute path to the loaded checkpoint.
        log_dir: Directory the checkpoint lives in.
        num_pairs: Final number of pairs that will be saved.
        state_frames: Extra body frames appended to the state vector (in order).

    Returns:
        A populated :class:`RecordingMeta`.
    """
    obs_mgr = env.unwrapped.observation_manager
    obs_term_names = list(obs_mgr.active_terms["policy"])
    obs_term_dims = [int(td[0]) for td in obs_mgr.group_obs_term_dim["policy"]]

    robot = env.unwrapped.scene.articulations["robot"]
    action_joint_names = list(robot.data.joint_names)

    # ``state_frames`` are the extra ee frames; the PARC-style FM conditions on AND
    # predicts the same set (no history/target asymmetry), so ``state_frame_names``
    # below is the single frame set. Training persists it as ``target_frame_names`` in
    # the checkpoint meta; online.load_fm_context falls back to ``state_frame_names``.
    # Raw robot-state layout recorded by collect_pairs (world frame, absolute).
    # Order and dims MUST match the torch.cat in collect_pairs.
    num_joints = len(action_joint_names)
    state_term_names = [
        "root_pos_w", "root_quat_w", "root_link_lin_vel_w", "root_link_ang_vel_w", "joint_pos", "joint_vel",
    ]
    state_term_dims = [3, 4, 3, 3, num_joints, num_joints]
    # Per extra frame: world pose + twist, appended after root + joints.
    for f in state_frames:
        state_term_names += [f"{f}:pos_w", f"{f}:quat_w", f"{f}:lin_vel_w", f"{f}:ang_vel_w"]
        state_term_dims += [3, 4, 3, 3]

    return RecordingMeta(
        checkpoint_path=os.path.abspath(resume_path),
        checkpoint_sha256=sha256_file(resume_path),
        task_name=args_cli.task,
        experiment_name=agent_cfg.experiment_name,
        run_name=os.path.basename(log_dir),
        runner_class=agent_cfg.class_name,
        rsl_rl_version=installed_version,
        recorded_at=datetime.now().isoformat(timespec="seconds"),
        stochastic=bool(args_cli.stochastic),
        dt=float(env.unwrapped.step_dt),
        num_envs=int(env.num_envs),
        num_pairs=int(num_pairs),
        obs_term_names=obs_term_names,
        obs_term_dims=obs_term_dims,
        action_joint_names=action_joint_names,
        state_term_names=state_term_names,
        state_term_dims=state_term_dims,
        state_quat_format="xyzw",
        state_joint_names=action_joint_names,
        state_vel_frame="link",
        state_frame_names=list(state_frames),
    )


def _assemble_state(robot, state_frames, frame_body_idx, n_envs) -> torch.Tensor:
    """Raw world-frame per-step state row ``[N, node_dim]`` (layout == state_term_names).

    Shared by :func:`collect_pairs` and :func:`collect_windows` so the recorded node
    layout stays in one place: root pos(3)/quat(4, xyzw)/lin(3)/ang(3), joints, then
    per extra frame pos(3)/quat(4)/lin(3)/ang(3).
    """
    parts = [
        wp.to_torch(robot.data.root_pos_w),           # [N, 3]
        wp.to_torch(robot.data.root_quat_w),          # [N, 4] (xyzw)
        wp.to_torch(robot.data.root_link_lin_vel_w),  # [N, 3]
        wp.to_torch(robot.data.root_link_ang_vel_w),  # [N, 3]
        wp.to_torch(robot.data.joint_pos),            # [N, nj]
        wp.to_torch(robot.data.joint_vel),            # [N, nj]
    ]
    if state_frames:
        fpos = wp.to_torch(robot.data.body_pos_w)[:, frame_body_idx]        # [N, K, 3]
        fquat = wp.to_torch(robot.data.body_quat_w)[:, frame_body_idx]      # [N, K, 4]
        flink = wp.to_torch(robot.data.body_link_vel_w)[:, frame_body_idx]  # [N, K, 6]
        parts.append(
            torch.cat([fpos, fquat, flink[..., :3], flink[..., 3:]], dim=-1).reshape(n_envs, -1)
        )
    return torch.cat(parts, dim=-1)


def _assemble_ref_future_node(ref_cmd, future_offsets, state_frames, n_envs) -> torch.Tensor:
    """Ego-anchored clip future ``[N, F, node_dim]`` (node layout == state).

    Shared by :func:`collect_pairs` and :func:`collect_windows`.
    """
    pos_w, quat_w, lin_w, ang_w = ref_cmd.get_future_world_base(future_offsets)  # [N,F,·]
    jp, jv = ref_cmd.get_future_joint_state(future_offsets)                       # [N,F,nj]
    node = torch.cat([pos_w, quat_w, lin_w, ang_w, jp, jv], dim=-1)              # [N,F,13+2nj]
    if state_frames:
        ff = ref_cmd.get_future_world_frames(state_frames, future_offsets)       # [N,F,K,13]
        node = torch.cat([node, ff.reshape(n_envs, ff.shape[1], -1)], dim=-1)
    return node


def collect_pairs(
    env: RslRlVecEnvWrapper,
    policy,
    policy_nn,
    initial_obs,
    num_steps: int,
    num_pairs: int,
    stochastic: bool,
    ref_cmd=None,
    ref_base_frame: str = "pelvis_link",
    state_frames: list[str] | None = None,
    scheduler: PerturbationScheduler | None = None,
    ref_future_len: int = 0,
) -> dict[str, np.ndarray]:
    """Roll out the policy and collect flattened (obs, action) pairs.

    The action recorded for an observation is ``policy(obs)`` computed *before*
    stepping the environment, so each pair is correctly aligned. The
    ``terminated``/``done``/``time_out`` flags are recorded *after* the step and
    describe the transition out of that pair: ``terminated`` excludes time-outs,
    so it marks the last pair before a true (failure) termination.

    Args:
        env: The wrapped Isaac Lab environment.
        policy: Inference policy callable returning actions from observations.
        policy_nn: The policy network for ``reset`` on rsl-rl < 4.0.0; ``None``
            when the inference policy itself exposes ``reset`` (rsl-rl >= 4.0.0).
        initial_obs: The starting observation TensorDict (from ``get_observations``).
        num_steps: Number of rollout steps (``ceil(num_pairs / num_envs)``).
        num_pairs: Total pairs to keep; arrays are trimmed to this length.
        stochastic: Whether to sample actions stochastically.
        ref_cmd: Trajectory command term exposing ``get_reference_world_state``,
            or ``None`` to skip recording the reference state.
        ref_base_frame: Trajectory output frame used as the robot root for the
            reference reconstruction.

    Returns:
        A dict of arrays each with leading dimension ``num_pairs``:
        ``obs``, ``action``, ``state``, ``terminated``, ``done``, ``time_out``,
        ``episode_id``, ``step_in_episode`` and, when ``ref_cmd`` is given,
        ``ref_state`` (the desired reference in the same layout as ``state``).
    """
    n_envs = env.num_envs
    device = env.unwrapped.device
    use_policy_reset = version.parse(installed_version) >= version.parse("4.0.0")
    robot = env.unwrapped.scene.articulations["robot"]
    # Height scan (the FM's height-conditioning token), captured when the recording task
    # has a height_scanner sensor active (e.g. G1-teacher-clf-record). Same mdp.height_scan
    # used by the FM deploy + (later) the RL obs term, so record/train/deploy stay in sync.
    has_scan = "height_scanner" in env.unwrapped.scene.sensors
    # Depth cameras (the deployable FM exteroception signal). Recorded ALONGSIDE the scan
    # whenever the task has depth observation terms, so ONE recording can train either FM
    # and the choice is made at FM-train time via cfg.extero_source.
    has_depth = bool(mdp.find_depth_obs_terms(env.unwrapped))

    # Body indices for the extra frames appended to the state vector (in order).
    state_frames = list(state_frames or [])
    if state_frames:
        body_names = list(robot.data.body_names)
        frame_body_idx = torch.tensor(
            [body_names.index(f) for f in state_frames], dtype=torch.long, device=device
        )

    obs = initial_obs
    # Distinct ids per env so step-0 pairs from different envs aren't conflated.
    episode_id = torch.arange(n_envs, dtype=torch.int64, device=device)
    step_in_ep = torch.zeros(n_envs, dtype=torch.int64, device=device)
    next_episode_id = int(n_envs)

    buf: dict[str, list[torch.Tensor]] = {
        "obs": [], "action": [], "state": [],
        "terminated": [], "done": [], "time_out": [],
        "episode_id": [], "step_in_episode": [],
    }
    if ref_cmd is not None:
        buf["ref_state"] = []
    # Per-step commanded velocity (the upstream command that drove the reference),
    # recorded when the clip task exposes it -- enables future-horizon command
    # conditioning in the flow-matching dataset.
    record_cmd_vel = ref_cmd is not None and getattr(ref_cmd, "command_velocity_available", False)
    if record_cmd_vel:
        buf["target_vel"] = []
    # Ego-anchored clip FUTURE target (robot-state-conditioned FM): per step an F-node
    # window of the clip ego-anchored onto the robot, plus the aligned future command and
    # the snap-event flag. Requires the look-ahead accessors + a command velocity.
    record_ref_future = (
        ref_future_len > 0 and ref_cmd is not None
        and hasattr(ref_cmd, "get_future_world_base") and record_cmd_vel
    )
    if record_ref_future:
        future_offsets = torch.arange(int(ref_future_len), device=device)
        buf["ref_future"] = []
        buf["cmd_future"] = []
        buf["reanchor_active"] = []
    if has_scan:
        buf["height_scan"] = []
    if has_depth:
        buf["depth"] = []
    # Per-step perturbation flag (1 while a perturbation window was active for that env).
    if scheduler is not None:
        buf["perturb_active"] = []
        scheduler.reset(env, torch.arange(n_envs, device=device))  # assign groups + initial OOD DR
        print(f"[INFO] {scheduler.summary()}")

    print(f"[INFO] Collecting {num_steps} steps on {n_envs} envs "
          f"({num_steps * n_envs} pairs, trimming to {num_pairs})...")
    for step in range(num_steps):
        # Advance perturbation windows (applies pushes on OFF->ON edges) and corrupt
        # the policy's obs for active envs *before* the forward pass.
        active = None
        if scheduler is not None:
            active = scheduler.step(env)
            if scheduler.obs_noise_scale > 0.0:
                obs["policy"] = scheduler.corrupt_obs(obs["policy"], active)
        with torch.inference_mode():
            actions = policy(obs, stochastic_output=stochastic)
        # The action actually stepped is the (optionally) noise-corrupted action.
        actions_applied = (
            scheduler.corrupt_actions(actions, active) if scheduler is not None else actions
        )

        # Snapshot the (obs, action) pair *before* stepping. Clone because the
        # TensorDict tensors are overwritten by the next env.step. Record the
        # applied (noisy) action so the pair reflects what drove the transition.
        buf["obs"].append(obs["policy"].detach().cpu().clone())
        buf["action"].append(actions_applied.detach().cpu().clone())
        if scheduler is not None:
            buf["perturb_active"].append(active.detach().cpu().clone())
        # Raw robot world-state at the same timestep as obs. Order/dims MUST match
        # state_term_names/state_term_dims in _build_recording_meta.
        state = _assemble_state(robot, state_frames, frame_body_idx, n_envs)
        buf["state"].append(state.detach().cpu().clone())
        # Height scan at the same pre-step moment as state (per-ray ground clearance,
        # [N, K]); the dataset picks the snap-row scan as the FM height token.
        if has_scan:
            buf["height_scan"].append(
                mdp.height_scan(env.unwrapped, SceneEntityCfg("height_scanner")).detach().cpu().clone()
            )
        # Depth at the same pre-step moment, one token row per camera ([N, T, H*W]),
        # normalised by each camera's own obs-term params.
        if has_depth:
            buf["depth"].append(mdp.depth_obs_stack(env.unwrapped).detach().cpu().clone())
        # Desired reference at the same timestep, reconstructed to the world-frame
        # ``state`` layout. ``y_des``/``dy_des`` hold the value backing the current
        # obs (last computed during the previous env.step), so this stays aligned
        # with the pre-step ``state``/``obs`` snapshot above.
        if ref_cmd is not None:
            ref_state = ref_cmd.get_reference_world_state(ref_base_frame)
            # Append the same extra frames (world-lifted clip reference) so ref_state
            # shares the state layout (state_term_names) when a 'reference' target is used.
            if state_frames:
                ref_frames = ref_cmd.reference_world_frames(state_frames)  # [N, K, 13]
                ref_state = torch.cat([ref_state, ref_frames.reshape(n_envs, -1)], dim=-1)
            buf["ref_state"].append(ref_state.detach().cpu().clone())
        if record_cmd_vel:
            buf["target_vel"].append(ref_cmd.get_command_velocity().detach().cpu().clone())
        # Ego-anchored clip future (the FM target), captured at the same pre-step moment
        # as state/ref_state: the clip ego-anchored onto the robot at THIS step, lifted F
        # nodes forward through the current placement (no internal re-snap). Node layout
        # matches 'state': [root_pos(3), root_quat(4 xyzw), root_lin(3), root_ang(3),
        # joint_pos(nj), joint_vel(nj), <per frame pos(3)/quat(4)/lin(3)/ang(3)>].
        if record_ref_future:
            node = _assemble_ref_future_node(ref_cmd, future_offsets, state_frames, n_envs)
            buf["ref_future"].append(node.detach().cpu().clone())
            buf["cmd_future"].append(
                ref_cmd.get_future_command_velocity(future_offsets).detach().cpu().clone()
            )
            buf["reanchor_active"].append(ref_cmd._reanchor_active.detach().cpu().clone())
        buf["episode_id"].append(episode_id.detach().cpu().clone())
        buf["step_in_episode"].append(step_in_ep.detach().cpu().clone())

        with torch.inference_mode():
            obs, _, dones, extras = env.step(actions_applied)
            # Reset recurrent/transformer state for terminated envs.
            if use_policy_reset:
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # Disambiguate termination (failure) from truncation (time-out). The
        # wrapper returns dones = terminated | truncated and exposes truncation
        # via extras["time_outs"] (present when not is_finite_horizon).
        time_out = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        terminated = dones.bool() & ~time_out.bool()
        buf["done"].append(dones.detach().cpu().clone())
        buf["terminated"].append(terminated.detach().cpu().clone())
        buf["time_out"].append(time_out.detach().cpu().clone())

        # Episode bookkeeping for the *next* step's observation: envs that just
        # finished get a fresh id and a reset step counter.
        done_mask = dones.to(torch.bool)
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            new_ids = torch.arange(
                next_episode_id, next_episode_id + n_done, device=device, dtype=torch.int64
            )
            episode_id = episode_id.clone()
            episode_id[done_mask] = new_ids
            next_episode_id += n_done
            step_in_ep = step_in_ep.clone()
            step_in_ep[done_mask] = 0
            step_in_ep[~done_mask] += 1
            # Re-roll group / window / OOD DR for envs that just reset.
            if scheduler is not None:
                scheduler.reset(env, done_mask.nonzero(as_tuple=False).flatten())
        else:
            step_in_ep = step_in_ep + 1

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{num_steps}")

    # Stack to [T, N, ...], flatten time and env into one sample axis, trim.
    out: dict[str, np.ndarray] = {}
    for key, tensors in buf.items():
        stacked = torch.stack(tensors, dim=0).numpy()  # [T, N, ...]
        if stacked.ndim >= 3:
            flat = stacked.reshape(-1, *stacked.shape[2:])
        else:
            flat = stacked.reshape(-1)
        out[key] = flat[:num_pairs]
    return out


def collect_windows(
    env: RslRlVecEnvWrapper,
    policy,
    policy_nn,
    initial_obs,
    num_windows: int,
    history_len: int,
    stochastic: bool,
    ref_cmd,
    ref_base_frame: str,
    state_frames: list[str] | None,
    future_len: int,
    writer,
    max_steps: int | None = None,
) -> int:
    """Roll out and STREAM pre-formed FM snap windows to ``writer`` (bounded RAM).

    Instead of buffering every step (like :func:`collect_pairs`) and discarding all but
    the snaps at train time, this forms one training window per env per step where the
    command's re-anchor flag fires AND the env has ``history_len`` consecutive valid (no
    reset since) history rows, and writes it straight to disk. This is equivalent to
    ``snap_windows(num_history=history_len, stride=1)`` over a full per-step recording
    (proven by ``tests/test_record_windows.py``), but never holds more than a per-env
    ring of the last ``H`` states plus one flush batch in memory.

    Args:
        env, policy, policy_nn, initial_obs, stochastic: as in :func:`collect_pairs`.
        num_windows: Stop after streaming this many windows.
        history_len: ``H``, history-state rows per window (>= trainable N).
        ref_cmd: Trajectory command exposing the look-ahead accessors + ``_reanchor_active``.
        state_frames: Extra body frames appended to each state row.
        future_len: ``F``, future nodes recorded per window.
        writer: A :class:`RecordedWindowWriter` to append batches to.
        max_steps: Safety cap on rollout steps (defaults to a generous multiple).

    Returns:
        The number of windows actually written.
    """
    n_envs = env.num_envs
    device = env.unwrapped.device
    use_policy_reset = version.parse(installed_version) >= version.parse("4.0.0")
    robot = env.unwrapped.scene.articulations["robot"]
    has_scan = "height_scanner" in env.unwrapped.scene.sensors
    has_depth = bool(mdp.find_depth_obs_terms(env.unwrapped))

    state_frames = list(state_frames or [])
    frame_body_idx = None
    if state_frames:
        body_names = list(robot.data.body_names)
        frame_body_idx = torch.tensor(
            [body_names.index(f) for f in state_frames], dtype=torch.long, device=device
        )
    future_offsets = torch.arange(int(future_len), device=device)

    obs = initial_obs
    H = int(history_len)
    # Per-env history ring ([N, H, node]; row H-1 = most recent) + a consecutive-valid
    # counter reset to 0 on episode end so a window can never span a reset.
    hist: torch.Tensor | None = None
    valid = torch.zeros(n_envs, dtype=torch.long, device=device)
    # Distinct episode ids per env so the FM's by-episode train/val split has no leakage.
    episode_id = torch.arange(n_envs, dtype=torch.int64, device=device)
    next_episode_id = int(n_envs)

    if max_steps is None:
        # Generous cap: snaps fire ~ N/ego_anchor_every per step once rings fill; 50x the
        # ideal step count still terminates a run whose snaps are unexpectedly sparse.
        max_steps = 50 * math.ceil(num_windows / max(n_envs, 1)) + 1000

    print(f"[INFO] Streaming up to {num_windows} snap windows (H={H}, F={future_len}) "
          f"on {n_envs} envs; max_steps={max_steps}...")
    written = 0
    step = 0
    while written < num_windows and step < max_steps:
        with torch.inference_mode():
            actions = policy(obs, stochastic_output=stochastic)

        # Pre-step snapshot (aligned with obs), matching collect_pairs' snap timing.
        state_t = _assemble_state(robot, state_frames, frame_body_idx, n_envs)  # [N, node]
        if hist is None:
            hist = torch.zeros(n_envs, H, state_t.shape[1], device=device)
        reanchor_t = ref_cmd._reanchor_active.to(torch.bool)                    # [N]

        # Push this step's state into the ring; mark the row valid.
        hist = torch.roll(hist, shifts=-1, dims=1)
        hist[:, H - 1] = state_t
        valid = torch.clamp(valid + 1, max=H)

        # Emit a window for each env that snaps AND has a full valid ring.
        emit = reanchor_t & (valid >= H)
        if bool(emit.any()):
            idx = emit.nonzero(as_tuple=False).flatten()
            ref_future_t = _assemble_ref_future_node(ref_cmd, future_offsets, state_frames, n_envs)
            cmd_future_t = ref_cmd.get_future_command_velocity(future_offsets)  # [N, F, 3]
            batch = {
                "history_state": hist[idx].detach().cpu().numpy(),
                "ref_future": ref_future_t[idx].detach().cpu().numpy(),
                "cmd_future": cmd_future_t[idx].detach().cpu().numpy(),
                "episode_id": episode_id[idx].detach().cpu().numpy(),
            }
            if has_scan:
                scan = mdp.height_scan(env.unwrapped, SceneEntityCfg("height_scanner"))  # [N, K]
                batch["height_scan"] = scan[idx].detach().cpu().numpy()
            if has_depth:
                dep = mdp.depth_obs_stack(env.unwrapped)                    # [N, T, H*W]
                batch["depth"] = dep[idx].detach().cpu().numpy()
            written += writer.write_batch(batch)
            if written >= num_windows:
                break

        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            if use_policy_reset:
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # Invalidate the ring for envs that just reset so the next window can't span it;
        # assign fresh episode ids (matches collect_pairs' bookkeeping).
        done_mask = dones.to(torch.bool)
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            valid[done_mask] = 0
            new_ids = torch.arange(
                next_episode_id, next_episode_id + n_done, device=device, dtype=torch.int64
            )
            episode_id = episode_id.clone()
            episode_id[done_mask] = new_ids
            next_episode_id += n_done

        step += 1
        if step % 100 == 0:
            print(f"  step {step}: {written}/{num_windows} windows")

    if written < num_windows:
        print(f"[WARN] Reached max_steps={max_steps} with {written}/{num_windows} windows "
              f"(snaps sparser than expected); saving what was collected.")
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load a checkpoint, roll it out, and save recorded (obs, action) pairs.

    Two modes: legacy ``--num_pairs`` buffers every step and writes one ``.npz``;
    ``--num_windows`` streams only the FM snap windows to a chunked ``.h5`` (bounded RAM).
    """
    windowed = args_cli.num_windows is not None
    if windowed == (args_cli.num_pairs is not None):
        raise ValueError(
            "Pass exactly one of --num_pairs (legacy per-step .npz) or "
            "--num_windows (streamed snap-window .h5)."
        )
    count = args_cli.num_windows if windowed else args_cli.num_pairs
    if count <= 0:
        raise ValueError(f"--num_{'windows' if windowed else 'pairs'} must be positive; got {count}.")
    if windowed:
        if args_cli.history_len < 1:
            raise ValueError(f"--history_len must be >= 1; got {args_cli.history_len}.")
        if args_cli.perturb:
            raise ValueError("--perturb (recovery data) is only supported in --num_pairs mode.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    resume_path = _resolve_resume_path(agent_cfg)
    log_dir = os.path.dirname(resume_path)
    out_dir = args_cli.output_dir or os.path.join(log_dir, "recorded_data")

    with launch_simulation(env_cfg, args_cli):
        # Recursive contact sensor patch (matches play.py / analyze_distillation.py).
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = log_dir

        # Override which clip library the teacher tracks (the recorded distribution
        # mirrors this clip). Done before auto_clip_device so its size check sees it.
        if args_cli.clip is not None:
            traj_ref = getattr(getattr(env_cfg, "commands", None), "traj_ref", None)
            if traj_ref is None or not hasattr(traj_ref, "path"):
                raise ValueError("--clip given but the task has no commands.traj_ref.path to override.")
            print(f"[INFO] Overriding clip library: {traj_ref.path} -> {args_cli.clip}")
            traj_ref.path = args_cli.clip

        # Hold the clip library off-GPU when large (auto: cpu for >1 GB libs, which
        # OOM a small card at scene init). Explicit --clip_device overrides.
        auto_clip_device(env_cfg, args_cli.clip_device)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        runner.alg.eval_mode()

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # For rsl-rl < 4.0.0 the inference policy doesn't expose reset(); fetch
        # the underlying network to reset recurrent state instead (see play.py).
        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy_nn = None
        elif version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # Number of records to collect: windows (windowed) or (obs, action) pairs (legacy).
        num_records = count

        # Resolve the trajectory command to record the reference from, plus the extra
        # body frames appended to the recorded state. Default frames = the command's
        # end-effector outputs (ankles/wrists). Resolved before meta so layout matches.
        ref_name, ref_cmd = _resolve_ref_command(env, args_cli.ref_command_name)
        if args_cli.state_frames is not None:
            state_frames = list(args_cli.state_frames)
        elif ref_cmd is not None and hasattr(ref_cmd, "resolve_ee_frames"):
            state_frames = ref_cmd.resolve_ee_frames()
        else:
            state_frames = []
        if ref_cmd is not None:
            print(f"[INFO] Recording reference state from command '{ref_name}' "
                  f"(base frame '{args_cli.ref_base_frame}').")
            if getattr(ref_cmd, "command_velocity_available", False):
                print(f"[INFO] Recording commanded velocity (target_vel) from command '{ref_name}'.")
        else:
            print("[WARN] No trajectory command with get_reference_world_state found; "
                  "skipping ref_state recording.")
        if state_frames:
            print(f"[INFO] Recording {len(state_frames)} extra state frames: {state_frames}")

        meta = _build_recording_meta(env, agent_cfg, resume_path, log_dir, num_records, state_frames)

        # Ego-anchored clip-future target (robot-state-conditioned FM): recorded when the
        # command exposes the look-ahead accessors + a command velocity, unless disabled.
        rf_len = 0 if args_cli.no_ref_future else int(args_cli.ref_future_len)
        will_record_ref_future = (
            rf_len > 0 and ref_cmd is not None and hasattr(ref_cmd, "get_future_world_base")
            and getattr(ref_cmd, "command_velocity_available", False)
        )
        if will_record_ref_future:
            meta.ref_source = "ego_anchored_clip_future"
            meta.ref_future_len = rf_len
            meta.ref_future_term_names = list(meta.state_term_names)  # node layout == state
            meta.ref_future_term_dims = list(meta.state_term_dims)
            print(f"[INFO] Recording ego-anchored clip future (ref_future): F={rf_len} nodes/step "
                  f"+ cmd_future + reanchor_active (snap events).")
        elif ref_cmd is not None:
            meta.ref_source = "reference"

        # Single observation fetch reused for the memory estimate, the provenance
        # export, and the start of collection. compute() defaults to
        # update_history=False, so this does not mutate observation history.
        obs = env.get_observations()

        # Warn (don't fail) on very large in-memory datasets: the legacy npz path buffers
        # everything in RAM before writing (the windowed .h5 path streams, so no warning).
        if not windowed:
            obs_dim = int(obs["policy"].shape[1])
            est_bytes = num_records * (obs_dim + int(env.num_actions)) * 4
            if est_bytes > 8 * (1 << 30):
                print(f"[WARN] Estimated ~{est_bytes / (1 << 30):.1f} GiB of obs+action data will be "
                      f"buffered in RAM before writing. Consider --num_windows (streamed .h5).")

        # Export richer provenance YAML (gains, scales, default joint angles).
        # Optional: skip if the task lacks the terms it needs.
        with torch.inference_mode():
            first_actions = policy(obs, stochastic_output=args_cli.stochastic)
        try:
            export_policy_parameters(env, obs, first_actions, log_dir)
        except (KeyError, ValueError) as exc:
            print(f"[WARN] export_policy_parameters skipped: {exc}")

        # Clear any recurrent/transformer state advanced by the export-time
        # forward pass so the first recorded pair starts from a clean state.
        if policy_nn is None:
            policy.reset()
        else:
            policy_nn.reset()

        # -- Windowed mode: stream only the FM snap windows to a chunked .h5 --------
        if windowed:
            if not will_record_ref_future:
                raise ValueError(
                    "--num_windows requires the ego-anchored clip future (ref_future): use a "
                    "clip-tracking task exposing get_future_world_base + a command velocity, "
                    "and do not pass --no_ref_future."
                )
            if "height_scanner" not in env.unwrapped.scene.sensors:
                raise ValueError(
                    "--num_windows requires a height_scanner sensor (the FM's height token); "
                    "record on a task that has one (e.g. G1-teacher-clf-terrain-play)."
                )
            H = int(args_cli.history_len)
            node_dim = int(sum(meta.state_term_dims))
            K = int(mdp.height_scan(env.unwrapped, SceneEntityCfg("height_scanner")).shape[1])
            cmd_dim = int(
                ref_cmd.get_future_command_velocity(
                    torch.arange(1, device=env.unwrapped.device)
                ).shape[-1]
            )
            # Depth (the deployable FM signal) is recorded alongside the scan whenever the
            # task exposes depth obs terms, so one session can train either FM.
            depth_shape = None
            if mdp.find_depth_obs_terms(env.unwrapped):
                dep = mdp.depth_obs_stack(env.unwrapped)          # [N, T, H*W]
                depth_shape = (int(dep.shape[1]), int(dep.shape[2]))
                print(f"[INFO] recording depth alongside the height scan: {depth_shape} per window")
            meta.format = "windowed_h5"
            meta.history_len = H
            meta.height_scan_dim = K
            # content_sha256 stays empty for the streamed format (the whole dataset is
            # never in RAM to hash); sha256_file(<.h5>) identifies the file if needed.
            out_path = unique_output_path(out_dir, prefix="recorded", ext=".h5")
            with RecordedWindowWriter(
                out_path, num_records, H, node_dim, rf_len, K, cmd_dim, meta,
                depth_shape=depth_shape,
            ) as writer:
                n_written = collect_windows(
                    env, policy, policy_nn, obs, num_records, H, args_cli.stochastic,
                    ref_cmd=ref_cmd, ref_base_frame=args_cli.ref_base_frame,
                    state_frames=state_frames, future_len=rf_len, writer=writer,
                )
            prov_dir = os.path.splitext(out_path)[0] + ".provenance"
            git_snapshot(prov_dir)
            try:
                dump_yaml(os.path.join(prov_dir, "params", "env.yaml"), env_cfg)
                dump_yaml(os.path.join(prov_dir, "params", "agent.yaml"), agent_cfg)
            except Exception as exc:
                print(f"[WARN] recording params dump skipped: {exc}")
            print(f"[INFO] Streamed {n_written} snap windows to: {out_path}")
            print(f"[INFO]   H={H} F={rf_len} node_dim={node_dim} K={K} cmd_dim={cmd_dim}; "
                  f"format=windowed_h5  provenance={prov_dir}")
            env.close()
            return

        # -- Legacy mode: buffer every step, write one .npz --------------------------
        num_steps = math.ceil(num_records / env.num_envs)

        # Optional perturbation scheduler for recovery-data collection.
        scheduler = None
        if args_cli.perturb:
            push_v = float(args_cli.push_vel)
            push_range = {"x": (-push_v, push_v), "y": (-push_v, push_v)} if push_v > 0 else None
            scheduler = PerturbationScheduler(
                num_envs=env.num_envs,
                dt=float(env.unwrapped.step_dt),
                device=env.unwrapped.device,
                clean_fraction=args_cli.clean_fraction,
                on_range_s=tuple(args_cli.perturb_on_range_s),
                off_range_s=tuple(args_cli.perturb_off_range_s),
                action_noise_scale=args_cli.action_noise_scale,
                push_vel_range=push_range,
                ood_dr=args_cli.ood_dr,
                ood_dr_scale=args_cli.ood_dr_scale,
                obs_noise_scale=args_cli.obs_noise_scale,
                seed=agent_cfg.seed,
            )

        data = collect_pairs(
            env, policy, policy_nn, obs, num_steps, num_records, args_cli.stochastic,
            ref_cmd=ref_cmd, ref_base_frame=args_cli.ref_base_frame, state_frames=state_frames,
            scheduler=scheduler, ref_future_len=rf_len,
        )

        if data.get("height_scan") is not None:
            meta.height_scan_dim = int(data["height_scan"].shape[1])
            print(f"[INFO] Recorded height_scan: {data['height_scan'].shape} "
                  f"(K={meta.height_scan_dim} rays) -- the FM height-conditioning token.")

        out_path = unique_output_path(out_dir, prefix="recorded", ext=".npz")
        # Content hash of the recorded arrays -> the dataset self-identifies.
        meta.content_sha256 = fingerprint_arrays(data)
        save_recorded_data(out_path, data, meta)
        # Per-recording provenance sidecar: git snapshot + the env/agent cfg used.
        prov_dir = os.path.splitext(out_path)[0] + ".provenance"
        git_snapshot(prov_dir)
        try:
            dump_yaml(os.path.join(prov_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(prov_dir, "params", "agent.yaml"), agent_cfg)
            if scheduler is not None:
                dump_yaml(os.path.join(prov_dir, "params", "perturb.yaml"), {
                    "clean_fraction": args_cli.clean_fraction,
                    "perturb_on_range_s": list(args_cli.perturb_on_range_s),
                    "perturb_off_range_s": list(args_cli.perturb_off_range_s),
                    "action_noise_scale": args_cli.action_noise_scale,
                    "push_vel": args_cli.push_vel,
                    "obs_noise_scale": args_cli.obs_noise_scale,
                    "ood_dr": args_cli.ood_dr,
                    "ood_dr_scale": args_cli.ood_dr_scale,
                })
        except Exception as exc:
            print(f"[WARN] recording params dump skipped: {exc}")
        print(f"[INFO] Saved {meta.num_pairs} (obs, action) pairs to: {out_path}")
        print(f"[INFO]   content_sha256={meta.content_sha256[:12]}  provenance={prov_dir}")
        ref_shape = data["ref_state"].shape if "ref_state" in data else None
        tvel_shape = data["target_vel"].shape if "target_vel" in data else None
        print(f"[INFO]   obs shape={data['obs'].shape}, action shape={data['action'].shape}, "
              f"state shape={data['state'].shape}, ref_state shape={ref_shape}, "
              f"target_vel shape={tvel_shape}, "
              f"terminations flagged={int(data['terminated'].sum())}")
        if data.get("ref_future") is not None:
            rf = data["ref_future"]; ra = data["reanchor_active"]
            print(f"[INFO]   ref_future shape={rf.shape}, cmd_future shape={data['cmd_future'].shape}, "
                  f"reanchor_active (snap steps)={int(ra.sum()):,}/{ra.size:,} "
                  f"({100.0 * float(ra.mean()):.1f}%)")
        if "perturb_active" in data and data["perturb_active"] is not None:
            pa = data["perturb_active"]
            print(f"[INFO]   perturb_active: {int(pa.sum()):,}/{pa.size:,} steps "
                  f"({100.0 * float(pa.mean()):.1f}%) perturbed")

        env.close()


if __name__ == "__main__":
    main()
