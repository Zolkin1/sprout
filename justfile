# Copyright 2026 Zachary Olkin. All rights reserved.
# Flow-matching workflow recipes. Run `just` (or `just --list`) for the menu.
# Each recipe activates the env_isaaclab venv, so you don't need it active first.

set shell := ["bash", "-c"]

venv      := "source env_isaaclab/bin/activate &&"
task      := "G1-teacher-clf"
source    := "hf/trajectories/retargeted/2026-06-16_16-30-26_merged"
clip_lib  := "clips/g1_flat_v1.h5"
eval_clip := "clips/g1_flat_v1_eval.h5"

# show the menu
default:
    @just --list

# generate the training clip library (FM data is recorded against this)
gen-clips num="10000" seed="0":
    {{venv}} python scripts/rsl_rl/generate_clips.py --source {{source}} --config clips/skill_mix.json \
        --exclude-skills walk_backwards --out {{clip_lib}} --num-clips {{num}} --seed {{seed}}

# generate the held-out eval clip library (unseen stitchings: same source, new seed)
fm-gen-eval-clips num="100" seed="1":
    {{venv}} python scripts/rsl_rl/generate_clips.py --source {{source}} --config clips/skill_mix.json \
        --exclude-skills walk_backwards --out {{eval_clip}} --num-clips {{num}} --seed {{seed}}

# record FM training data: roll out a tracking policy on the train clip library
fm-record run num="2000000" envs="256":
    {{venv}} python scripts/rsl_rl/record_data.py --task {{task}} --load_run {{run}} \
        --num_pairs {{num}} --num_envs {{envs}}

# record held-out eval data on the eval clip library
fm-record-eval run num="150000" envs="256":
    {{venv}} python scripts/rsl_rl/record_data.py --task {{task}} --load_run {{run}} \
        --num_pairs {{num}} --num_envs {{envs}} env.commands.traj_ref.path={{eval_clip}}

# train the FM model (target=auto|actual|reference; auto infers from the data).
# frames: history extra-frames; "" (default) = none, "all" = symmetric, or a space-separated subset.
# noise: history-token obs noise, "on" (default) or "off" (passes --no-obs_noise).
# hist: history length in timesteps (default 25; e.g. 2 = current + one past frame).
fm-train data name="" target="auto" frames="" noise="on" hist="25":
    {{venv}} python scripts/flow_matching/train.py --data {{data}} --target_source {{target}} \
        {{ if name != "" { "--run_name " + name } else { "" } }} \
        {{ if frames != "" { "--history_frames " + frames } else { "" } }} \
        {{ if noise == "off" { "--no-obs_noise" } else { "" } }} --history_len {{hist}} --wandb

# train with a held-out eval dataset (eval/* metrics alongside val/*; target=auto|actual|reference).
# frames: history extra-frames; "" (default) = none, "all" = symmetric, or a space-separated subset.
# noise: history-token obs noise, "on" (default) or "off" (passes --no-obs_noise).
# hist: history length in timesteps (default 25; e.g. 2 = current + one past frame).
fm-train-eval data eval name="" target="auto" frames="" noise="on" hist="25":
    {{venv}} python scripts/flow_matching/train.py --data {{data}} --eval_data {{eval}} --target_source {{target}} \
        {{ if name != "" { "--run_name " + name } else { "" } }} \
        {{ if frames != "" { "--history_frames " + frames } else { "" } }} \
        {{ if noise == "off" { "--no-obs_noise" } else { "" } }} --history_len {{hist}} --wandb

# record a viz rollout (driver auto-resolved from the checkpoint) and open the viewer.
# Pass `task=...` to override a stale/renamed task baked into an old checkpoint.
fm-viz fm task="":
    {{venv}} python scripts/flow_matching/record_viz.py --fm_checkpoint {{fm}} \
        {{ if task != "" { "--task " + task } else { "" } }}
    {{venv}} python scripts/flow_matching/visualize.py

# print a clip library's fingerprint (provenance)
fm-clip-fingerprint lib=clip_lib:
    {{venv}} python scripts/rsl_rl/clip_library_fingerprint.py {{lib}}
