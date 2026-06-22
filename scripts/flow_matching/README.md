<!-- Copyright 2026 Zachary Olkin. All rights reserved. -->

# Flow-Matching Workflow

A conditional flow-matching transformer that predicts a robot's **future references**
(base position, joint pos/vel, base lin/ang velocity, and optionally end-effector
frames — all in the robot's current local base frame) from **state history + a
velocity command**. Built on the open-loop motion-sequence task.

Pipeline: **record data → train → (record viz rollout → view it)**. Steps 1 & 3 need
Isaac Sim; steps 2 & 4 are pure PyTorch (no Isaac). You **record once and train many**
FM variants on the same `.npz` — only re-record when the base policy or the recorded
state layout changes.

---

## Quick start (`just`)

Most steps are one recipe — `just` activates the venv for you, and `just --list` is the menu.
The scripts **auto-configure from the data**: they infer `target_source`, `cmd_array`, and
`clip_device` from the recording's contents, and `record_viz` finds its driver policy from the
checkpoint — so you rarely pass those flags.

```bash
just                                # the menu
just fm-record <tracking-run>       # record training data
just fm-gen-eval-clips              # held-out eval clip library (same source, seed 1)
just fm-record-eval <tracking-run>  # record eval data on that library
just fm-train <data> [name]         # train  (data = a run dir OR a .npz)
just fm-train-eval <data> <eval>    # train with held-out eval/* metrics
just fm-viz <fm_checkpoint>         # record a viz rollout + open the viewer
```

The sections below document the underlying scripts and their (now mostly optional) flags.

---

## 1. Record training data  (needs Isaac Sim)

Rolls out a trained motion-seq tracking policy and records per-step
`obs / action / state / ref_state / target_vel / ee-frames` to one `.npz`.

```bash
python scripts/rsl_rl/record_data.py \
    --task G1-motion-sequence-clf \
    --load_run <tracking-policy run> \
    --num_pairs 2000000 --num_envs 256
```

- **`clip_device` is automatic** — large clip libraries (>1 GB) default to CPU so they don't OOM a small GPU; the per-env gather from host RAM is cheap. Override with `--clip_device cuda`/`cpu`.
- **`target_vel`** (the upstream velocity command) is recorded automatically when the command term exposes it → enables future-command conditioning at train time.
- **`--state_frames`** — extra body frames whose world pose+twist are appended to `state`/`ref_state` (default: the command's EE frames = ankles+wrists; pass nothing for root+joints only). The FM can then predict these.
- **Output:** `logs/rsl_rl/<exp>/<run>/recorded_data/recorded_<timestamp>.npz` (printed at the end, with array shapes).
- **Provenance:** a `recorded_<ts>.provenance/` sidecar (git snapshot + `params/{env,agent}.yaml`) is written next to the `.npz`, and a content hash (`content_sha256`, compression-robust) is stored in the dataset metadata so it self-identifies.

## 2. Train the FM model  (no Isaac Sim)

```bash
python scripts/flow_matching/train.py --data <run-dir-or-.npz> --wandb
```

- **`--data`** accepts a `.npz` **or a run directory** (→ newest `recorded_*.npz` under it).
- **Auto-configured from the recording** (printed as `[auto] target_source=…, cmd_array=…`):
  - `--target_source` → `reference` if the recording has `ref_state`, else `actual`. Override explicitly.
  - `--cmd_array` → `target_vel` if that array was recorded, else no command (future-horizon conditioning: past command in the encoder, future in the decoder). Override with an array name or `none`.
- **`--eval_data <run-or-.npz>`** — a *separate* held-out recording (never trained on); metrics logged under `eval/*`, distinct from the within-recording `val/*`. Must share the train layout. See "Held-out evaluation" below.
- **`--run_name`** — run dir is `logs/flow_matching/<timestamp>_<run_name>` (defaults to just the timestamp).
- Other knobs: `--num_epochs` (200), `--stride` (1; raise to cut RAM on huge datasets), `--history_len` / `--future_len` (25), `--lr`, `--cosine`.
- **Output:** `logs/flow_matching/<ts>_<run_name>/{best.pt, ckpt_<N>.pt}`. wandb logs per epoch (val every `--val_every`=5); per-step loss → TensorBoard.
- **Provenance (parity with RL):** each run writes `git/sprout.diff` + `params/fm_config.yaml` (the resolved config) + `params/data.yaml` (train/eval `.npz` paths + their `sha256` + the recording's metadata), and every `.pt` records `data_path`/`data_sha256` — so a run is fully self-describing.

## 3. Record a viz rollout  (needs Isaac Sim)

Runs a driver policy + the FM model, dumps states + predicted future references to a `.npz`.
Handles `cmd_array` (future-command) and EE-frame checkpoints.

```bash
python scripts/flow_matching/record_viz.py --fm_checkpoint logs/flow_matching/<run>/best.pt \
    --num_envs 8 --max_steps 500
```

- **Driver policy is auto-resolved** from the checkpoint's `meta.checkpoint_path` (the tracker the FM was trained on). Override with `--load_run`/`--checkpoint`. The driver must match the env's *current* policy-obs dim — if the task obs changed since, pass a matching tracker.
- **`clip_device` is automatic** (cpu for large libs). `--task` defaults to the FM checkpoint's recorded task. Runs headless; pass `--viz kit` for the live Isaac viewer.
- **Output:** `logs/flow_matching/viz/viz_<timestamp>.npz`. Confirm the log line `FM ready: … future_cmd=on/off, command=on/none`.

## 4. View it  (no Isaac Sim)

```bash
python scripts/flow_matching/visualize.py            # newest viz_*.npz
# or:  --data logs/flow_matching/viz/viz_<ts>.npz  [--port 8080]
```

Opens a viser server (default `http://localhost:8080`): the robot rollout with FM-predicted
future references as ghost poses. Use `--ghosts` to set how many are shown.

---

## Held-out evaluation

Keep an eval set the model never trained on, at two levels:

**FM eval (different recording).** Record a *second* dataset and pass it to training:
```bash
just fm-train-eval <train> <eval>     # or: train.py --data <train> --eval_data <eval> --wandb
```
`eval/*` metrics (sampled RMSEs, geodesic) are logged separately from `val/*` (the within-recording split), evaluated with the train normalizers — leakage-free and comparable.

**Eval clip library (unseen stitchings) for RL generalization.** Generate a held-out clip library from the *same* source with a different `--seed`:
```bash
just fm-gen-eval-clips     # or: generate_clips.py --source <same> --out clips/g1_flat_v1_eval.h5 --seed 1 --num-clips 100
```
Different seed → different `clip_fingerprint`, so provenance distinguishes it from the train library. Use it by overriding the clip path (record an eval dataset on it, or evaluate the RL tracker directly):
```bash
... env.commands.traj_ref.path=clips/g1_flat_v1_eval.h5
```
The fingerprint check will **warn** that the clip library differs from the trained-against one — that's **expected** for eval (you *want* a different library); it's not an error.

## Data & provenance

Where artifacts land and how they're identified:

| Artifact | Location | Identity | Synced |
|---|---|---|---|
| Recorded datasets (`.npz`) | `logs/rsl_rl/.../recorded_data/` (gitignored) | `content_sha256` (content) + file `sha256` recorded by the FM run | local/talos via mount/rsync |
| FM runs | `logs/flow_matching/<ts>_<name>/` | `git/sprout.diff`, `params/*.yaml`, `.pt` `data_sha256` | — |
| Clip libraries (`.h5`) | `clips/` (gitignored) | `clip_fingerprint` (source+config+seed), verified at load | local |
| Trajectory libraries | `hf/` cache | `library_fingerprint` (content) | HuggingFace Hub |

Recorded datasets deliberately stay **machine-local** (multi-GB, many); the FM run's `params/data.yaml` records their hashes, so correctness is verifiable across manual copies.

A git-tracked **`data/registry.yaml`** records the identity (fingerprint) of named clip/trajectory libraries. Check a local artifact against it with:
```bash
python scripts/rsl_rl/data_registry.py list
python scripts/rsl_rl/data_registry.py verify g1_flat_v1   # local .h5 / lib vs the recorded fingerprint
```
Set `env.commands.traj_ref.strict_fingerprint=true` to make a library/clip fingerprint mismatch **fatal** at env construction (vs the default loud warning) — a guardrail for deploy / cross-machine runs. *(Planned: pulling registry artifacts from HF Hub at a pinned revision — not yet wired.)*

## Gotchas
- **`clip_device`, command, and target-source are auto** — large libs go to CPU, and the command/target are inferred from the recording. Each is overridable, and the resolved choice is printed.
- **`--data` accepts a `.npz` or a run directory** (→ newest recording under it).
- **Record once, train many** — re-record only when the base policy or recorded layout (`--state_frames`, command terms) changes.
- The **driver policy obs must match the env obs** (record_data + record_viz); an env obs-layout change silently strips compatibility with older trackers.
- `train.py` is Isaac-free, so FM training can run wherever (e.g. talos) without the sim stack.
