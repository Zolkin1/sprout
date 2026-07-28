<!-- Copyright 2026 Zachary Olkin. All rights reserved. -->

# Flow-Matching Workflow

A conditional flow-matching transformer that predicts a robot's **future references**
(base position, joint pos/vel, base lin/ang velocity, and optionally end-effector
frames — all in the robot's current local base frame) from **state history + a
velocity command**. Built on the open-loop motion-sequence task.

Pipeline: **record data → train → deploy**. Recording needs Isaac Sim; training is pure
PyTorch (no Isaac). You **record once and train many** FM variants on the same recording —
only re-record when the base policy or the recorded state layout changes.

---

## Quick start

The scripts **auto-configure from the data**: they infer `target_source`, `cmd_array` and
`clip_device` from the recording's contents, so you rarely pass those flags.

```bash
# 1. record teacher rollouts (needs Isaac)
python scripts/rsl_rl/record_data.py --task G1-teacher-clf-terrain-play \
    --load_run <teacher-run> --clip <lib.h5> --num_windows 200000 --num_envs 256

# 2. train the FM (no Isaac)
python scripts/flow_matching/train.py --data <windows.h5> --wandb

# 3. deploy: the teacher tracks the FM
python scripts/rsl_rl/play.py --task=G1-tracker-clf-fm-terrain-play \
    --load_run <teacher-run> --fm_checkpoint <fm.pt> --clip <lib.h5> --clip_device cpu
```

The sections below document the underlying scripts and their (now mostly optional) flags.

---

## 1. Record training data  (needs Isaac Sim)

Rolls out a trained teacher (clip-tracking) policy and records per-step
`obs / action / state / ref_state / target_vel / ee-frames` to one `.npz`.

```bash
python scripts/rsl_rl/record_data.py \
    --task G1-teacher-clf-terrain-play \
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
- **`--history_frames`** — extra body frames the history/encoder token consumes. Default (omit / `none` / bare flag) is an **empty set**: the generator needs **no** measured end-effectors at deployment, while the target still predicts every recorded frame for the tracker's `ee_ref`. Pass `all` for symmetric (history uses every recorded frame), or a space-separated subset.
- Other knobs: `--num_epochs` (100), `--stride` (1; raise to cut RAM on huge datasets), `--history_len` / `--future_len` (25), `--lr`, `--cosine`.
- **Output:** `logs/flow_matching/<ts>_<run_name>/{best.pt, ckpt_<N>.pt}`. wandb logs per epoch (val every `--val_every`=5); per-step loss → TensorBoard.
- **Provenance (parity with RL):** each run writes `git/sprout.diff` + `params/fm_config.yaml` (the resolved config) + `params/data.yaml` (train/eval `.npz` paths + their `sha256` + the recording's metadata), and every `.pt` records `data_path`/`data_sha256` — so a run is fully self-describing.

## 3. Look at a rollout  (no Isaac for the viewer)

`record_viz.py` / `visualize.py` were deleted with the abandoned autoregressive-FM subsystem
(`d9a1d57` removed the producer; the viewer was orphaned). Use the debug-episode pair
instead — it records the FM prediction horizon alongside the height scan and the reference
ribbons, and replays in viser without Isaac:

```bash
python scripts/rsl_rl/debug_episodes.py --task G1-tracker-clf-fm-terrain-play \
    --load_run <teacher-run> --fm_checkpoint <fm.pt> --clip <lib.h5> --out bundle.npz
python scripts/rsl_rl/view_debug_episodes.py bundle.npz
```

For the Isaac-vs-MuJoCo comparison of the same FM + tracker pair, see
`scripts/rsl_rl/compare_sim2sim.py` (documented in the top-level README).

---

## Held-out evaluation

Keep an eval set the model never trained on, at two levels:

**FM eval (different recording).** Record a *second* dataset and pass it to training:
```bash
python scripts/flow_matching/train.py --data <train> --eval_data <eval> --wandb
```
`eval/*` metrics (sampled RMSEs, geodesic) are logged separately from `val/*` (the within-recording split), evaluated with the train normalizers — leakage-free and comparable.

**Eval clip library (unseen stitchings) for RL generalization.** Generate a held-out clip library from the *same* source with a different `--seed`:
```bash
python scripts/rsl_rl/generate_mb_library.py --out clips/g1_mb_v5_eval.h5 --seed 1 ...
```
Different seed → different `clip_fingerprint`, so provenance distinguishes it from the train library. Use it by overriding the clip path (record an eval dataset on it, or evaluate the RL tracker directly):
```bash
... --clip clips/g1_mb_v5_eval.h5
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
- The **driver policy obs must match the env obs** (record_data + debug_episodes); an env obs-layout change silently strips compatibility with older trackers.
- `train.py` is Isaac-free, so FM training can run wherever (e.g. talos) without the sim stack.
