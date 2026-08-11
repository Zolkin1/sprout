# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from provenance import check_run_conventions  # isort: skip
from data_logger import DataLogger  # isort: skip
from episode_stats import EpisodeStatsAccumulator  # isort: skip
from play_plots import run_plots, PLOT_REGISTRY  # isort: skip
from export_parameters import export_policy_parameters  # isort: skip
# The FM exporter lives in scripts/flow_matching; put it on the path like
# the FM script dir on the path so play.py can write both deploy artifacts itself.
_FM_SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              os.pardir, "flow_matching"))
if _FM_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _FM_SCRIPT_DIR)
from export_tracker_onnx import export_tracker_onnx  # isort: skip
from export_fm_onnx import export_fm_onnx  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
# allow_abbrev=False: never accept a prefix of a long option as that option.
# Otherwise ``--clip <h5>`` would silently bind to ``--clip_device`` (the only
# ``--clip*`` flag), feeding a clip PATH as a torch device string. With it off,
# the explicit ``--clip`` flag below is the only thing ``--clip`` can mean, and
# any other ambiguous prefix falls through to hydra as a loud error.
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.", allow_abbrev=False)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_resolution", type=str, default="1920x1080",
    help='Recording resolution as "WxH" (e.g. "1920x1080", "2560x1440"). '
         "Overrides env_cfg.viewer.resolution.  Higher = sharper but slower.",
)
parser.add_argument(
    "--video_follow_robot", action="store_true", default=True,
    help="Lock the recording camera to the robot (origin_type=asset_root). "
         "Default on; pass --no-video_follow_robot to keep the static world camera.",
)
parser.add_argument("--no-video_follow_robot", dest="video_follow_robot", action="store_false")
parser.add_argument(
    "--video_eye", type=float, nargs=3, default=(0.0, 5.0, 0.8), metavar=("X", "Y", "Z"),
    help="Camera eye offset from the robot root when --video_follow_robot is set (m). "
         "Default is a side view: 5 m off the robot's +y axis at ~hip+0.5 height. "
         "Interpreted in world frame by default; pass --video_track_yaw to rotate the "
         "offset into the robot's heading frame (camera stays on the robot's left).",
)
parser.add_argument(
    "--video_lookat", type=float, nargs=3, default=(0.0, 0.0, 0.3), metavar=("X", "Y", "Z"),
    help="Camera lookat offset from the robot root when --video_follow_robot is set (m). "
         "Default aims at root_pos + 0.2 m (the robot's centroid for a G1 / humanoid). "
         "If the robot is missing from frame, this is the first knob to adjust.",
)
parser.add_argument(
    "--video_focal_length", type=float, default=24.0,
    help="USD camera focal length (mm) — controls zoom / FOV.  Smaller = wider angle. "
         "Default 24 mm (~74° horizontal FOV with the standard 35 mm aperture); "
         "drop to ~18 for very wide, raise to 35–50 for telephoto.",
)
parser.add_argument(
    "--video_env_index", type=int, default=0,
    help="Which env to follow when --video_follow_robot is set (default 0).",
)
parser.add_argument(
    "--video_track_yaw", action="store_true", default=False,
    help="Rotate the camera eye/lookat offsets into the robot's heading frame "
         "so the relative view (e.g. side view) stays consistent across turns. "
         "Default off — offsets are world-aligned.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--dr_fingerprint", action="store_true", default=False,
    help="Print a hash of the per-env domain-randomisation state (joint friction, armature, "
         "PD gains, default joint pos) after env creation, then continue. Use it to confirm "
         "that env i is the SAME robot across two runs before comparing them per env -- the "
         "draws come off the seeded RNG, so a changed --num_envs or event set re-rolls them.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--plots", type=str, default="default",
    help='Comma-separated plot names, "default", "all", or "none". '
         f"Available: {', '.join(['default', 'all', 'none', *PLOT_REGISTRY.keys()])}",
)
parser.add_argument("--plot_envs", type=int, default=1, help="Number of envs to generate plots for.")
parser.add_argument(
    "--plot_time_range", type=str, default=None,
    help='Restrict plots to a time window in seconds. Format: "start,end" — '
         'either end may be empty (e.g. "5,", ",10"). Requires logging.',
)
parser.add_argument(
    "--dump_play_data", action="store_true", default=False,
    help="Also save the raw per-step world-frame reference arrays (y_des/y_act + "
         "clip/FM references + the FM prediction window) to play_data.npz in the plots "
         "dir, for offline analysis. "
         "Pair with --plots fm_predictions to include the full FM prediction window.",
)
parser.add_argument(
    "--no_deploy_onnx", action="store_true", default=False,
    help="Skip writing the two deploy ONNX artifacts (exported/tracker.onnx and, with "
         "--fm_checkpoint, exported/fm.onnx). They are written by default because they must "
         "stay in sync with the policy.pt + policy_parameters.yaml this run exports.",
)
parser.add_argument("--max_steps", type=int, default=None, help="Exit after this many steps (None = run forever).")
parser.add_argument(
    "--train_mode",
    action="store_true",
    default=False,
    help="Run the policy as it behaves during training: actor/critic kept in torch .train() "
         "mode (dropout/batchnorm active) AND stochastic action sampling enabled. "
         "Default play is .eval() mode with deterministic actions.",
)
parser.add_argument(
    "--profile", action="store_true", default=False,
    help="Enable the per-frame profiling harness (phase timers for policy / "
         "env.step / logging + a manager and debug-vis breakdown). Prints a "
         "rolling report every --profile_steps frames. Zero cost when off.",
)
parser.add_argument(
    "--profile_steps", type=int, default=100,
    help="Frames per rolling profiling report window (with --profile).",
)
parser.add_argument(
    "--profile_sync", action="store_true", default=False,
    help="Profile in synchronous mode: drain the whole device at each phase "
         "boundary for true, correctly-nested per-phase costs (incl. Warp "
         "physics/render on its own stream). Slower; use to resolve "
         "cross-stream mis-attribution.",
)
parser.add_argument(
    "--no-debug-vis", dest="no_debug_vis", action="store_true", default=False,
    help="Force the trajectory command's debug visualization off "
         "(env_cfg.commands.traj_ref.debug_vis = False) to A/B test its cost.",
)
parser.add_argument(
    "--clip", type=str, default=None,
    help="Path to the clip / trajectory library (.h5) to track, overriding the "
         "cfg default. Applied to any command term exposing a `path` field "
         "(traj_ref). Equivalent to the hydra override "
         "env.commands.traj_ref.path=<h5>, but without the hydra syntax.",
)
parser.add_argument(
    "--clip_device", type=str, default=None,
    help="Device to hold a trajectory/clip library on (e.g. 'cpu' to keep a multi-GB "
         "library off a small GPU). Applied to any command term exposing a clip_device "
         "field. Set here (not via hydra) because the cfg default is None.",
)
parser.add_argument(
    "--fm_checkpoint", type=str, default=None,
    help="Path to the flow-matching generator checkpoint (.pt) to track. Applied to any "
         "command term exposing an fm_checkpoint field (the FmTrackingCommand), "
         "overriding the cfg default. The generator's DEPTH RESOLUTION is read off this "
         "checkpoint and a matching camera pair is built automatically, so nothing else "
         "has to state it.",
)
parser.add_argument(
    "--fm_depth_resolution", type=str, default="",
    help="'WxH' to build the generator's depth cameras at, OVERRIDING what --fm_checkpoint "
         "states. Only useful for deliberately feeding a generator the wrong sampling to "
         "see what it does -- the command term's shape check refuses the mismatch, so this "
         "is for experiments on that check, not a normal knob.",
)
parser.add_argument(
    "--no-logging", dest="no_logging", action="store_true", default=False,
    help="Disable per-step DataLogger collection (overrides --plots) to A/B "
         "test logging cost.",
)
parser.add_argument(
    "--episode_stats", action="store_true", default=False,
    help="Accumulate the env's extras['log'] (Episode_Termination/*, "
         "Episode_Reward/*, Metrics/*, Curriculum/*) over the run and write a "
         "training-comparable summary to <log_dir>/episode_stats.txt. These are "
         "the same step-mean aggregates the training logger reports, so e.g. "
         "frame_drift_post_grace can be compared against the wandb curve. "
         "Independent of --plots / DataLogger.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # Patch activate_contact_sensors for recursive traversal (must be after SimulationApp)
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            # For distillation, allow overriding the experiment via CLI
            if agent_cfg.class_name == "DistillationRunner" and getattr(args_cli, "teacher_experiment", None):
                play_log_path = os.path.abspath(
                    os.path.join("logs", "rsl_rl", args_cli.teacher_experiment)
                )
            else:
                play_log_path = log_root_path
            resume_path = get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # HARD refusal before anything is built: this run's checkpoints are only meaningful
        # under the observation conventions they were trained with, and a mismatch is
        # invisible in every shape. Absence counts as a mismatch -- see check_run_conventions.
        check_run_conventions(log_dir, env_cfg)

        # Trajectory-library version check: the saved env.yaml records the
        # portable content fingerprint of the library this policy trained
        # against.  The cfg built from the gym registry has it unset, so inject
        # the trained-against value here (before env construction); the
        # trajectory command then warns loudly if the locally loaded library
        # has since diverged.  Best-effort — never blocks play.
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "traj_ref"):
            trained_env_yaml = os.path.join(log_dir, "params", "env.yaml")
            if os.path.isfile(trained_env_yaml):
                try:
                    from isaaclab.utils.io import load_yaml

                    saved_traj = (load_yaml(trained_env_yaml).get("commands") or {}).get("traj_ref") or {}
                    trained_fp = saved_traj.get("library_fingerprint")
                    if trained_fp is not None:
                        env_cfg.commands.traj_ref.library_fingerprint = trained_fp
                        env_cfg.commands.traj_ref.library_manifest = saved_traj.get("library_manifest")
                        print(f"[INFO] Trained-against trajectory library fingerprint: {trained_fp}")
                    else:
                        print(
                            "[INFO] Saved env.yaml has no trajectory library fingerprint "
                            "(run predates fingerprinting); skipping library-version check."
                        )
                except Exception as exc:  # noqa: BLE001 - never block play
                    print(f"[WARN] Could not read trained trajectory library fingerprint: {exc}")

        # profiling A/B: optionally force the trajectory debug visualization off
        if args_cli.no_debug_vis and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "traj_ref"):
            env_cfg.commands.traj_ref.debug_vis = False
            print("[profiling] traj_ref.debug_vis forced OFF for this run.")

        # Optionally override the clip/trajectory library path (the file each
        # command term tracks).  Applied to any command term exposing a `path`
        # field; equivalent to env.commands.traj_ref.path=<h5> but without hydra.
        if args_cli.clip is not None and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
            for name, term in vars(env_cfg.commands).items():
                if hasattr(term, "path"):
                    term.path = args_cli.clip
                    print(f"[INFO] clip path='{args_cli.clip}' for commands.{name}.")
            # The ClipTerrainImporter builds its terrain mesh + per-env terrain_id from
            # its OWN clip_library_path, which must stay identical to the command's clip
            # path -- tile terrain ids must line up with the clips' terrain ids. Re-point it
            # too so --clip is a single source of truth. NOTE: nothing asserts the two match;
            # setting only one (e.g. a bare hydra override of commands.traj_ref.path) trains
            # on a terrain the clips do not belong to. Go through --clip.
            terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
            if terrain is not None and hasattr(terrain, "clip_library_path"):
                terrain.clip_library_path = args_cli.clip
                print(f"[INFO] clip path='{args_cli.clip}' for scene.terrain (ClipTerrainImporter).")

        # Optionally hold a clip/trajectory library off-GPU (set in Python, not via
        # hydra: the cfg field default is None so hydra rejects a str override).
        # Needed to play a policy whose multi-GB clip library would otherwise OOM a
        # small GPU -- mirrors the same flag in train.py / record_data.py.
        if args_cli.clip_device is not None and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
            for name, term in vars(env_cfg.commands).items():
                if hasattr(term, "clip_device"):
                    term.clip_device = args_cli.clip_device
                    print(f"[INFO] clip_device='{args_cli.clip_device}' for commands.{name}.")

        # Optionally override the flow-matching generator checkpoint to track
        # (mirrors --clip_device / the same flag in train.py); convenient for
        # playing against a freshly-trained FM without editing the cfg default.
        if args_cli.fm_checkpoint is not None and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
            for name, term in vars(env_cfg.commands).items():
                if hasattr(term, "fm_checkpoint"):
                    term.fm_checkpoint = args_cli.fm_checkpoint
                    print(f"[INFO] fm_checkpoint='{args_cli.fm_checkpoint}' for commands.{name}.")

        # The generator may SAMPLE the depth cameras on its own grid (so its vision input
        # can change without retraining the policy). Read that resolution off the checkpoint
        # and build the matching camera pair here, BEFORE gym.make -- derived rather than
        # declared, so a flag cannot disagree with the artifact. A checkpoint at the policy's
        # own resolution adds nothing and reads the policy's cameras, as it always did.
        if args_cli.fm_checkpoint is not None:
            from robot_rl.flow_matching.online import read_fm_extero_meta
            from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
                add_fm_depth_pair,
            )
            fm_extero = read_fm_extero_meta(args_cli.fm_checkpoint)
            want = (args_cli.fm_depth_resolution or "").strip()
            if want:
                w, h = (int(v) for v in want.replace("x", ",").split(","))
                print(f"[INFO] --fm_depth_resolution {w}x{h} OVERRIDES the checkpoint's "
                      f"{fm_extero['width']}x{fm_extero['height']}; the command term's shape "
                      "check will refuse the pair unless you meant this.")
            elif fm_extero["width"] is not None:
                w, h = fm_extero["width"], fm_extero["height"]
            else:
                w = h = None
            if w is not None and getattr(env_cfg.scene, "depth_camera_near", None) is not None:
                add_fm_depth_pair(env_cfg, w, h)

        # Apply video-recording knobs to the viewer cfg before env construction.
        # ``env_cfg.viewer.resolution`` is the offscreen-render product size the
        # ``rgb_array`` render path reads from (see manager_based_rl_env.py:286).
        # We also create a dedicated USD camera prim and point the render
        # product at it so the recording is decoupled from the GUI viewport
        # ``/OmniverseKit_Persp``.  The viewer cfg's built-in tracking
        # (``origin_type=asset_root``) drives the persp cam via
        # ``sim.set_camera_view`` → ``visualizer._set_viewport_camera``, which
        # writes to the visualizer's pinned camera path, *not* whatever
        # ``cam_prim_path`` we point the render product at.  So we set
        # ``origin_type="world"`` to disable that tracking and instead drive
        # the recording camera ourselves from the main loop using
        # ``isaacsim.core.utils.viewports.set_camera_view`` (which writes the
        # USD transform on the prim path we give it).  No render products are
        # ever created during training, so this whole branch is a no-op there.
        record_cam_state = None  # populated below if --video; consumed in the loop
        if args_cli.video:
            try:
                w_str, h_str = args_cli.video_resolution.lower().split("x")
                env_cfg.viewer.resolution = (int(w_str), int(h_str))
            except (ValueError, AttributeError) as exc:
                raise SystemExit(
                    f"--video_resolution must be 'WxH', got '{args_cli.video_resolution}': {exc}"
                )

            from pxr import UsdGeom
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            recording_cam_path = "/World/RecordingCam"
            cam_prim = stage.GetPrimAtPath(recording_cam_path)
            if not cam_prim.IsValid():
                cam_prim = UsdGeom.Camera.Define(stage, recording_cam_path).GetPrim()
            # Dial the focal length on the USD camera so the framing isn't
            # the default telephoto.  ``horizontalAperture`` stays at the
            # USD default (20.955 mm, 35 mm-equivalent); focal length is
            # the per-shot zoom knob.
            UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(args_cli.video_focal_length)
            env_cfg.viewer.cam_prim_path = recording_cam_path
            # Disable the built-in asset tracker — it would move /OmniverseKit_Persp,
            # not our recording cam.  We drive the recording cam ourselves below.
            env_cfg.viewer.origin_type = "world"

            record_cam_state = {
                "prim_path": recording_cam_path,
                "eye_offset": np.asarray(args_cli.video_eye, dtype=float),
                "lookat_offset": np.asarray(args_cli.video_lookat, dtype=float),
                "env_index": args_cli.video_env_index,
                "follow_robot": args_cli.video_follow_robot,
                "track_yaw": args_cli.video_track_yaw,
            }
            print(
                f"[INFO] Video recording: {env_cfg.viewer.resolution[0]}x{env_cfg.viewer.resolution[1]}"
                f" via dedicated USD camera '{recording_cam_path}'"
                + (f", following robot in env {args_cli.video_env_index}"
                   if args_cli.video_follow_robot else ", static world camera")
            )

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # Per-env DR fingerprint. The survival sweep compares env i across runs, which is only
        # sound if env i is the SAME ROBOT in both -- and the startup DR draws come off the
        # seeded torch RNG, so anything that shifts that stream (a different --num_envs, an
        # added/removed event, or a config change that consumes randomness during scene build)
        # silently re-rolls every env's physics. This prints a hash so that assumption is
        # checkable rather than assumed: same seed + same fingerprint => env i is comparable.
        if args_cli.dr_fingerprint:
            import hashlib
            import warp as wp
            _rb = env.unwrapped.scene.articulations["robot"]
            _parts = []
            for _name in ("joint_friction", "joint_armature", "joint_stiffness",
                          "joint_damping", "default_joint_pos"):
                _v = getattr(_rb.data, _name, None)
                if _v is None:
                    continue
                _a = (wp.to_torch(_v) if not isinstance(_v, torch.Tensor) else _v).detach().cpu().numpy()
                _parts.append((_name, _a))
            print("[dr-fingerprint] seed=%s num_envs=%d" % (args_cli.seed, env.unwrapped.num_envs))
            for _name, _a in _parts:
                _h = hashlib.sha256(np.ascontiguousarray(_a).tobytes()).hexdigest()[:16]
                _pe = hashlib.sha256(np.ascontiguousarray(_a[0]).tobytes()).hexdigest()[:8]
                print(f"[dr-fingerprint]   {_name:20s} shape={tuple(_a.shape)} all={_h} env0={_pe}")

        # --- DEBUG: contact sensor ---
        for name, sensor in env.unwrapped.scene.sensors.items():
            if hasattr(sensor, 'body_names'):
                print(f"[{name}] bodies: {sensor.body_names}, num_bodies: {sensor.num_sensors}")
        # --- END DEBUG ---

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # Playing never evaluates the value function, so a runner cfg can opt out of
        # restoring it (``load_critic = False``). The FM tracker does: it has no privileged
        # observation group to build a matching critic from, and skipping the load is what
        # lets ANY teacher checkpoint drive it -- the actor's observation groups are
        # identical across tracking modes, only the critic's width moves. Actor weights are
        # still loaded strictly, so a genuine policy mismatch still fails loudly.
        if getattr(agent_cfg, "load_critic", True):
            runner.load(resume_path)
        else:
            print("[INFO] Skipping the critic (play-only runner): loading actor weights only.")
            runner.load(resume_path, load_cfg={"actor": True, "critic": False,
                                               "optimizer": False, "iteration": True, "rnd": True})

        # print parameter counts
        if isinstance(runner, DistillationRunner):
            student_params = sum(p.numel() for p in runner.alg.student.parameters())
            teacher_params = sum(p.numel() for p in runner.alg.teacher.parameters())
            print(f"Student Parameters: {student_params:,}")
            print(f"Teacher Parameters: {teacher_params:,}")
        else:
            actor_params = sum(p.numel() for p in runner.alg.actor.parameters())
            critic_params = sum(p.numel() for p in runner.alg.critic.parameters())
            print(f"Actor Parameters: {actor_params:,}")
            print(f"Critic Parameters: {critic_params:,}")
            print(f"Total Parameters: {actor_params + critic_params:,}")

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        if args_cli.train_mode:
            runner.alg.train_mode()
            print("[INFO] Running policy in torch .train() mode with stochastic actions.")

        # Export the trained policy to TorchScript. NOT to ONNX here: rsl-rl's own ONNX
        # exporter wrote `policy.onnx` + a `policy.onnx.data` sidecar (weights as external
        # data, batch axis pinned to 1, no metadata) which nothing in this repo read, and
        # which sat next to the real deploy artifact `tracker.onnx` looking like an
        # alternative to it. `tracker.onnx` is the same network re-exported from this
        # `policy.pt`, but self-contained, batch-dynamic, and carrying the deployment
        # parameters in its metadata -- see export_tracker_onnx.py below.
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT only (see above for why no policy.onnx)
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")

        _policy_model = runner.alg.student if isinstance(runner, DistillationRunner) else runner.alg.actor
        teacher_model = runner.alg.teacher if isinstance(runner, DistillationRunner) else None

        dt = env.unwrapped.step_dt

        # Set up data logging
        plot_names = [s.strip() for s in args_cli.plots.split(",")]
        logging_enabled = plot_names != ["none"] and not args_cli.no_logging
        # The FM peel-off plot needs the full F-step predicted window logged each
        # step (heavier: [T, N, F, P]); enable it only when that plot is requested.
        log_fm_window = any(p in ("fm_predictions", "all") for p in plot_names)
        if logging_enabled:
            logger = DataLogger(env, log_fm_window=log_fm_window)
        env_ids = list(range(min(args_cli.plot_envs, env_cfg.scene.num_envs)))

        # Optional training-comparable episode-metric accumulator. Folds the
        # env's extras["log"] (which play normally discards) into running
        # step-means; written to <log_dir>/episode_stats.txt at the end.
        ep_stats = EpisodeStatsAccumulator() if args_cli.episode_stats else None

        # optional per-frame profiling harness (see scripts/rsl_rl/profiling.py)
        phase_timer = None
        profile_report_path = None
        if args_cli.profile:
            from profiling import PhaseTimer, instrument_env_managers

            profile_report_path = os.path.join(log_dir, "profile_report.txt")
            phase_timer = PhaseTimer(env.unwrapped.device, sync=args_cli.profile_sync)
            # debug-vis + managers run inside env.step → recorded as nested detail.
            instrument_env_managers(env, phase_timer, include_debug_vis=True)
            print(
                f"[profiling] enabled  |  sync mode: "
                f"{'ON (slower, exact nesting)' if phase_timer.sync else 'OFF (async CUDA-event timers)'}"
                f"  |  requested --profile_sync={args_cli.profile_sync}; "
                f"rolling report every {args_cli.profile_steps} frames. "
                f"num_envs={env_cfg.scene.num_envs}, "
                f"logging={logging_enabled}.",
                flush=True,
            )

        # reset environment
        obs = env.get_observations()

        # Export policy parameters (YAML with joint info, gains, obs terms, etc.)
        with torch.inference_mode():
            actions = policy(obs)
        # Pass the actor's group list: the actor concatenates several observation groups
        # (proprioception + reference), and the exported term table has to enumerate them
        # in that order for the deploy to rebuild the flat input vector.
        export_policy_parameters(
            env, obs, actions, log_dir, actor_obs_groups=agent_cfg.obs_groups.get("actor")
        )

        # Deploy artifacts for the MuJoCo / hardware runners: exactly two ONNX files, the RL
        # tracker and (when an FM is loaded) the FM generator. Both carry their metadata
        # embedded, so the runner needs no YAML. Done here rather than in a separate step
        # because tracker.onnx needs the policy.pt + policy_parameters.yaml this run just
        # wrote -- exporting them together is what keeps the three in sync.
        #
        # NOT wrapped in a try/except. It used to be, and a failure printed a [WARN] and
        # left a run directory that looks complete but holds a stale fm.onnx (or none) --
        # which is then what gets benchmarked, sim2sim'd, and uploaded to the robot. A
        # broken export is a broken export; fail here where it is one line to read.
        if not args_cli.no_deploy_onnx:
            export_tracker_onnx(export_model_dir)
            _trk = os.path.join(export_model_dir, "tracker.onnx")
            print(f"[INFO] wrote {_trk} ({os.path.getsize(_trk) / 1e6:.1f} MB)")
            # Truthy, not `is not None`: "" is the documented sentinel for "no generator at
            # all" (see FmTrackingCommandCfg.fm_checkpoint), valid with reference_source=clip.
            # Under `is not None` an explicitly empty value got as far as torch.load("") and
            # died there, so the one configuration that means "no FM" could not be exported
            # past -- which is what the clip-only survival cell needs.
            if args_cli.fm_checkpoint:
                fm_out = os.path.join(export_model_dir, "fm.onnx")
                export_fm_onnx(args_cli.fm_checkpoint, fm_out)
                print(f"[INFO] wrote {fm_out}")
            else:
                print("[INFO] no --fm_checkpoint, so no fm.onnx (tracker only).")

        def _ph(name):
            """Phase context (no-op when profiling is off)."""
            return phase_timer.phase(name) if phase_timer is not None else contextlib.nullcontext()

        timestep = 0
        prof_window = 0  # frames since the last rolling profile report
        # simulate environment
        try:
            while True:
                start_time = time.time()
                with _ph("frame_total"):
                    # Drive the dedicated recording camera before the step so
                    # the next render (triggered inside env.step via
                    # RecordVideo) reads a fresh transform.  Writes directly to
                    # the USD prim via Isaac Sim's viewport util — render
                    # products read the prim transform, no viewport_api needed.
                    if record_cam_state is not None and record_cam_state["follow_robot"]:
                        with _ph("camera"):
                            robot = env.unwrapped.scene.articulations["robot"]
                            import warp as wp
                            env_idx = record_cam_state["env_index"]
                            root_pos_w = wp.to_torch(robot.data.root_pos_w)[
                                env_idx, :3
                            ].detach().cpu().numpy()
                            eye_off = record_cam_state["eye_offset"]
                            lookat_off = record_cam_state["lookat_offset"]
                            if record_cam_state["track_yaw"]:
                                # Rotate the offsets by the robot's yaw so e.g. a
                                # side-view (+y) stays on the robot's left even as
                                # the robot turns.
                                yaw = float(wp.to_torch(robot.data.heading_w)[env_idx].item())
                                c, s = np.cos(yaw), np.sin(yaw)
                                rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                                eye_off = rot @ eye_off
                                lookat_off = rot @ lookat_off
                            eye_w = (root_pos_w + eye_off).tolist()
                            target_w = (root_pos_w + lookat_off).tolist()
                            import isaacsim.core.utils.viewports as _isaacsim_viewports
                            _isaacsim_viewports.set_camera_view(
                                eye=eye_w, target=target_w,
                                camera_prim_path=record_cam_state["prim_path"],
                            )

                    # run everything in inference mode
                    with torch.inference_mode():
                        # agent stepping
                        with _ph("policy"):
                            actions = policy(obs, stochastic_output=args_cli.train_mode)
                            teacher_actions = (
                                teacher_model(obs, stochastic_output=args_cli.train_mode)
                                if teacher_model is not None else None
                            )
                        # env stepping (physics + GUI render + managers)
                        with _ph("env_step"):
                            obs, _, dones, infos = env.step(actions)
                        # reset recurrent states for episodes that have terminated
                        if version.parse(installed_version) >= version.parse("4.0.0"):
                            policy.reset(dones)
                        else:
                            policy_nn.reset(dones)

                    if logging_enabled:
                        with _ph("logger"):
                            logger.collect_step(env)
                            if teacher_model is not None:
                                logger.collect_distillation_actions(
                                    student_actions=actions, teacher_actions=teacher_actions
                                )
                    if ep_stats is not None:
                        tm = env.unwrapped.termination_manager
                        term_fired = {n: tm.get_term(n) for n in tm.active_terms}
                        ep_stats.collect_step(infos, dones, term_fired=term_fired)

                # rolling profiling report
                if phase_timer is not None:
                    phase_timer.tick()
                    prof_window += 1
                    if prof_window >= args_cli.profile_steps:
                        phase_timer.report(
                            total_key="frame_total",
                            top_level=["camera", "policy", "env_step", "logger"],
                            reset=True,
                            title="PLAY per-frame breakdown",
                            out_path=profile_report_path,
                        )
                        prof_window = 0

                timestep += 1
                if args_cli.video and timestep >= args_cli.video_length:
                    break
                if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                    break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            # Flush any partial profiling window (e.g. on Ctrl-C).
            if phase_timer is not None:
                phase_timer.report(
                    total_key="frame_total",
                    top_level=["camera", "policy", "env_step", "logger"],
                    title="PLAY per-frame breakdown (final)",
                    out_path=profile_report_path,
                )
            # Write/print the training-comparable episode-metric summary.
            if ep_stats is not None and ep_stats.num_steps > 0:
                num_envs = env.unwrapped.num_envs
                print("\n" + ep_stats.format_report(num_envs=num_envs))
                ep_stats_path = os.path.join(log_dir, "episode_stats.txt")
                ep_stats.write_report(ep_stats_path, num_envs=num_envs)
                print(f"[INFO] Episode stats written to: {ep_stats_path}")
            # Generate plots from collected data
            if logging_enabled and logger.num_steps > 0:
                data, metadata = logger.finalize()
                # Task-specific subfolder so different tasks loading the SAME run
                # (e.g. G1-teacher-clf-terrain-play vs G1-tracker-clf-fm-terrain-play) keep
                # their plots side by side instead of overwriting each other.
                plots_dir = os.path.join(log_dir, "plots", task_name)
                if args_cli.dump_play_data:
                    os.makedirs(plots_dir, exist_ok=True)
                    # Raw world-frame references + FM prediction window for offline
                    # height-drift analysis. Save the world arrays that exist (the
                    # FM-window ones need --plots fm_predictions) plus the channel-name
                    # metadata the post-processor needs to locate pelvis/feet z.
                    dump_keys = (
                        "y_des", "y_act", "dy_des", "dy_act",
                        "y_des_clip", "dy_des_clip", "y_des_fm", "dy_des_fm",
                        "fm_pred_y", "fm_pred_dy", "fm_active", "fm_pred_idx",
                    )
                    bundle = {k: data[k] for k in dump_keys if k in data}
                    for mk in ("pos_names", "vel_names", "dt", "t_offset",
                               "fm_window_len", "fm_novel_start", "ee_frames"):
                        if mk in metadata:
                            bundle[mk] = np.asarray(metadata[mk], dtype=object)
                    dump_path = os.path.join(plots_dir, "play_data.npz")
                    np.savez(dump_path, **bundle)
                    print(f"[INFO] Dumped raw play data ({len(bundle)} arrays) to {dump_path}")
                time_range = None
                if args_cli.plot_time_range:
                    parts = args_cli.plot_time_range.split(",")
                    if len(parts) != 2:
                        raise ValueError(
                            f"--plot_time_range expects 'start,end'; got {args_cli.plot_time_range!r}"
                        )
                    start = float(parts[0]) if parts[0].strip() else None
                    end = float(parts[1]) if parts[1].strip() else None
                    time_range = (start, end)
                run_plots(data, metadata, plots_dir, plot_names, env_ids, time_range=time_range)

            # close the simulator
            env.close()


if __name__ == "__main__":
    main()