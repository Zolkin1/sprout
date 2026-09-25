#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Record an observation trace from NVIDIA's OWN IsaacLab env, to validate our SONIC adapter.

This is the oracle the whole SONIC baseline rests on. Our adapter's observation layout is verified
against the released ONNX graphs, and the g1 mode's live slices were confirmed by perturbation --
but neither shows that the *semantics* we feed each slice are what SONIC was trained on. This runs
their env on a clip converted from OUR library, dumps per step both the raw inputs our adapter
consumes and the observations their code builds from them, and the replay script then diffs the
two. A block that disagrees names the bug; agreement everywhere means SONIC's behaviour on our
clips is a property of the policy, not of our plumbing.

**Their policy is deliberately not built.** We compare two functions of the same state, so the
actions taken during the trace are irrelevant -- zero actions are fine. That skips instantiating
their ``PolicyAndValueWrapper``, loading ``last.pt`` (a 30 GB download away) and needing
``vector_quantize_pytorch``, which is the fragile half of running their stack.

Runs in the isolated venv: ``env_sonic232`` (Python 3.11, Isaac Sim 5.1, IsaacLab 2.3.2,
gear_sonic). Must be launched from the GR00T checkout root, because their configs use paths
relative to it.

Usage::

    cd /home/zolkin/AmberLab/Project-Isaac-RL/GR00T-WholeBodyControl
    OMNI_KIT_ACCEPT_EULA=YES /home/zolkin/AmberLab/Project-Isaac-RL/env_sonic232/bin/python \\
        <repo>/scripts/rsl_rl/sonic_trace_record.py \\
        --motion_file data/motion_lib_from_our_clips --steps 60 --out /tmp/sonic_their_trace.npz
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
parser.add_argument("--motion_file", required=True,
                    help="Their motion_lib directory (see clip_to_sonic_motion_lib.py).")
parser.add_argument("--train_config", default="sonic_release/config.yaml",
                    help="The released checkpoint's training config, for env parity.")
parser.add_argument("--steps", type=int, default=60, help="Control steps to record.")
parser.add_argument("--out", required=True, help="Output trace .npz.")
parser.add_argument("--drive", default="zero", choices=("zero", "sonic"),
                    help="What to feed their env. 'zero' records a passive trace for the "
                         "observation-parity oracle. 'sonic' runs the RELEASED ONNX pair inside "
                         "THEIR env, driven by THEIR OWN obs groups -- so no code from "
                         "robot_rl.baselines.sonic is involved. That separates 'our integration is "
                         "wrong' from 'the policy cannot track our clips': if it fails to locomote "
                         "here too, that is the baseline's behaviour, not our plumbing.")
parser.add_argument("--sonic_dir", default=None,
                    help="Released ONNX directory; required for --drive sonic.")
parser.add_argument("--terrain", default="plane", choices=("plane", "trimesh"),
                    help="Their terrain. 'plane' keeps the trace about observations, not geometry.")
args_cli = parser.parse_args()

# --- launch Kit before importing anything isaaclab/isaacsim ------------------------------------
from isaaclab.app import AppLauncher                                        # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np                                                          # noqa: E402
import omegaconf                                                            # noqa: E402
import torch                                                                # noqa: E402
from hydra import compose, initialize_config_dir                            # noqa: E402

import gear_sonic.train_agent_trl as train_agent_trl                        # noqa: E402

# Their own backward-compatibility rewrites, copied verbatim from eval_agent_trl.main so the
# released config resolves to the release module paths.
_REWRITES = (
    ("groot.rl.trl.", "gear_sonic.trl."),
    ("groot.rl.envs.", "gear_sonic.envs."),
    ("groot.rl.utils.", "gear_sonic.utils."),
    ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
    ("groot.rl.agents.", "gear_sonic.trl."),
    ("groot/rl/data/", "gear_sonic/data/"),
    ("assets/bm/unitree_description/", "assets/robot_description/"),
    ("1215_bones_seed_filtered", "bones_seed_smpl"),
)


def _load_config():
    """Compose their eval config merged with the released training config, as their script does."""
    cfg_dir = os.path.join(os.path.dirname(train_agent_trl.__file__), "config")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        override_config = compose(config_name="base_eval")

    raw = open(args_cli.train_config).read()
    for a, b in _REWRITES:
        raw = raw.replace(a, b)
    train_config = omegaconf.OmegaConf.load(io.StringIO(raw))
    if train_config.get("eval_overrides") is not None:
        train_config = omegaconf.OmegaConf.merge(train_config, train_config.eval_overrides)
    config = omegaconf.OmegaConf.merge(train_config, override_config)

    with omegaconf.open_dict(config):
        config.num_envs = 1
        config.manager_env.config.num_envs = 1
        config.manager_env.config.terrain_type = args_cli.terrain
        # No observation noise: we are comparing deterministic functions of the state.
        config.manager_env.observations.policy.enable_corruption = False
        config.manager_env.observations.tokenizer.enable_corruption = False
        ml = config.manager_env.commands.motion.motion_lib_cfg
        ml.motion_file = args_cli.motion_file
        # 'dummy' makes their loader synthesise zero SMPL tensors, so no SMPL data is needed and
        # the SMPL encoder is never sampled (motion_has_smpl stays all-False).
        ml.smpl_motion_file = "dummy"
        ml.max_num_motions = 1
        # Start every episode at frame 0 so the trace lines up with our rollout frame-for-frame.
        config.manager_env.commands.motion.start_from_first_frame = True
    return config


def _t(x) -> np.ndarray:
    """Read a tensor as host numpy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main() -> int:
    """Build their env, step it with zero actions, and dump the observation trace."""
    # Progress markers: Kit's own log shows startup succeeding and the app closing a second later,
    # so without these there is no way to tell which of these three steps failed.
    print("[trace] step 1/3: composing their config ...", flush=True)
    config = _load_config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[trace] step 2/3: config OK, building their env on {device} ...", flush=True)
    env = train_agent_trl.create_manager_env(config, device, SimpleNamespace(headless=True))
    print("[trace] step 3/3: env constructed", flush=True)

    # ManagerEnvWrapper is not a gym wrapper: it holds the ManagerBasedRLEnv on ``.env`` and has no
    # ``.scene``/``.unwrapped`` of its own, but it does expose the tracking command as
    # ``.motion_command``.
    base = getattr(env, "env", env)
    base = getattr(base, "unwrapped", base)
    robot = base.scene["robot"]
    cmd = getattr(env, "motion_command", None) or base.command_manager.get_term("motion")
    print(f"[trace] env built. num_envs={base.num_envs} dt={base.step_dt}")
    print(f"[trace] command term: {type(cmd).__name__}, "
          f"num_future_frames={cmd.num_future_frames}, frame_skips={cmd.frame_skips}")
    print(f"[trace] robot joints ({robot.num_joints}): {list(robot.joint_names)[:4]} ...")

    enc_sess = dec_sess = None
    if args_cli.drive == "sonic":
        if not args_cli.sonic_dir:
            raise SystemExit("--drive sonic needs --sonic_dir")
        import onnxruntime as ort
        enc_sess = ort.InferenceSession(os.path.join(args_cli.sonic_dir, "model_encoder.onnx"),
                                        providers=["CPUExecutionProvider"])
        dec_sess = ort.InferenceSession(os.path.join(args_cli.sonic_dir, "model_decoder.onnx"),
                                        providers=["CPUExecutionProvider"])
        print(f"[trace] driving with the released ONNX pair from {args_cli.sonic_dir}")

    obs_dict = env.reset_all()
    print(f"[trace] obs groups: { {k: tuple(v.shape) for k, v in obs_dict.items() if hasattr(v, 'shape')} }")
    tok = obs_dict.get("tokenizer")
    if isinstance(tok, dict):
        print("[trace] tokenizer blocks:")
        for k, v in tok.items():
            print(f"           {k:44s} {tuple(v.shape)}")

    buf: dict[str, list] = {}

    def push(key: str, val) -> None:
        buf.setdefault(key, []).append(_t(val)[0])       # num_envs == 1

    tok_keys: list[str] = []
    for step in range(args_cli.steps):
        # --- what their code BUILT (the thing we are validating against) ---
        # Their ManagerEnvWrapper renames the obs groups: 'actor_obs' / 'critic_obs' / 'tokenizer',
        # and it hands back the tokenizer already FLATTENED to 1761 -- which, with the mode-id
        # scalar prepended, is exactly the released encoder's 1762-wide input.
        policy_obs = obs_dict.get("policy", obs_dict.get("actor_obs"))
        if policy_obs is None:
            raise KeyError(f"no policy/actor_obs group in obs_dict (keys: {list(obs_dict)})")
        push("their_policy_obs", policy_obs)
        tok = obs_dict.get("tokenizer")
        if isinstance(tok, dict):
            tok_keys = list(tok.keys())
            for k, v in tok.items():
                push(f"their_tok__{k}", v)
        elif tok is not None:
            push("their_tok_flat", tok)

        # --- the RAW inputs our adapter consumes, so it can rebuild the above ---
        d = robot.data
        push("root_quat_w_wxyz", d.root_quat_w)          # their IsaacLab 2.3 is wxyz
        push("root_ang_vel_b", d.root_ang_vel_b)
        push("joint_pos", d.joint_pos)
        push("default_joint_pos", d.default_joint_pos)
        push("joint_vel", d.joint_vel)
        push("projected_gravity_b", d.projected_gravity_b)

        # --- their reference window, the other half of the comparison ---
        push("ref_joint_pos_future", cmd.joint_pos_multi_future)
        push("ref_joint_vel_future", cmd.joint_vel_multi_future)
        # NOTE: unlike every other field here, get_root_quat_w returns [num_envs * F, 4] -- the
        # future frames are stacked into the FIRST axis, not a trailing one. Reshape to
        # [num_envs, F, 4] before ``push`` takes env 0, or we would record a single quaternion
        # and silently drop the other F-1 future frames.
        push("ref_root_quat_future",
             cmd.motion_lib.get_root_quat_w(cmd.future_motion_ids, cmd.future_time_steps)
             .reshape(base.num_envs, cmd.num_future_frames, 4))
        push("robot_anchor_quat_w", cmd.robot_anchor_quat_w)
        push("anchor_quat_w", cmd.anchor_quat_w)
        push("ref_ori6d_mf", cmd.root_rot_dif_l_multi_future)

        if enc_sess is None:
            actions = torch.zeros(base.num_envs, robot.num_joints, device=device)
        else:
            # Encoder input is [mode_id] ++ tokenizer, and the tokenizer group arrives already
            # flat (1761) -- their wrapper flattens it. Decoder input is [token] ++ actor_obs.
            # Both graphs are fixed-batch-1, so loop rows.
            tok_flat = obs_dict["tokenizer"].detach().cpu().numpy().astype(np.float32)
            prop = policy_obs.detach().cpu().numpy().astype(np.float32)
            acts = np.empty((tok_flat.shape[0], robot.num_joints), np.float32)
            for i in range(tok_flat.shape[0]):
                enc_in = np.concatenate([np.zeros((1, 1), np.float32), tok_flat[i:i + 1]], axis=1)
                token = enc_sess.run(None, {enc_sess.get_inputs()[0].name: enc_in})[0]
                dec_in = np.concatenate([token, prop[i:i + 1]], axis=1)
                acts[i] = dec_sess.run(None, {dec_sess.get_inputs()[0].name: dec_in})[0][0]
            actions = torch.as_tensor(np.clip(acts, -20.0, 20.0), device=device)
        push("action", actions)
        push("root_pos_w", d.root_pos_w)
        push("joint_pos_now", d.joint_pos)
        results = env.step({"actions": actions, "obs_dict": obs_dict})
        obs_dict = results[0]

    out = {k: np.stack(v) for k, v in buf.items()}
    out["meta_tok_keys"] = np.asarray(tok_keys, dtype=object)
    out["meta_joint_names"] = np.asarray(list(robot.joint_names), dtype=object)
    out["meta_num_future_frames"] = np.asarray(int(cmd.num_future_frames))
    out["meta_frame_skips"] = np.asarray(int(cmd.frame_skips))
    out["meta_dt"] = np.asarray(float(base.step_dt))
    if enc_sess is not None:
        # Headline numbers, so the run answers its question without a second script.
        rp = out["root_pos_w"]
        ref = out["ref_joint_pos_future"].reshape(len(rp), cmd.num_future_frames, -1)[:, 0]
        jerr = float(np.abs(out["joint_pos_now"] - ref).mean())
        trav = float(np.linalg.norm(rp[-1, :2] - rp[0, :2]))
        print(f"\n[trace] ===== SONIC driven inside THEIR env, on OUR clip =====")
        print(f"[trace]   steps {len(rp)} ({len(rp) * float(base.step_dt):.1f} s)")
        print(f"[trace]   mean |joint - reference| : {jerr:.3f} rad")
        print(f"[trace]   pelvis travel            : {trav:.2f} m")
        print(f"[trace]   |action| mean {np.abs(out['action']).mean():.2f} "
              f"max {np.abs(out['action']).max():.1f}")
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    np.savez_compressed(args_cli.out, **out)
    print(f"\n[trace] wrote {args_cli.out}")
    for k, v in sorted(out.items()):
        if getattr(v, "ndim", 0) >= 1 and not k.startswith("meta_"):
            print(f"          {k:40s} {v.shape}")
    return 0


if __name__ == "__main__":
    # Print the traceback BEFORE closing the app, and flush. ``simulation_app.close()`` hard-exits
    # the process, so a bare ``try/finally`` swallows the exception entirely -- Kit reports a clean
    # "Startup Complete" followed a second later by "Closing application" and the real error is
    # never printed anywhere. Learned the hard way.
    import traceback

    try:
        main()
    except BaseException:                                                   # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
