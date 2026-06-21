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

## 1. Record training data  (needs Isaac Sim)

Rolls out a trained motion-seq tracking policy and records per-step
`obs / action / state / ref_state / target_vel / ee-frames` to one `.npz`.

```bash
python scripts/rsl_rl/record_data.py \
    --task G1-motion-sequence-clf \
    --load_run <tracking-policy run> \
    --num_pairs 2000000 --num_envs 256 \
    --clip_device cpu --headless
```

- **`--clip_device cpu`** — keep the ~4 GB clip library off the GPU (otherwise it OOMs a 12 GB card). Per-step gathers copy only the tiny per-env slice.
- **`target_vel`** (the upstream velocity command) is recorded automatically when the command term exposes it → enables future-command conditioning at train time.
- **`--state_frames`** — extra body frames whose world pose+twist are appended to `state`/`ref_state` (default: the command's EE frames = ankles+wrists; pass nothing for root+joints only). The FM can then predict these.
- **Output:** `logs/rsl_rl/<exp>/<run>/recorded_data/recorded_<timestamp>.npz` (printed at the end, with array shapes).

## 2. Train the FM model  (no Isaac Sim)

```bash
python scripts/flow_matching/train.py \
    --data logs/rsl_rl/<exp>/<run>/recorded_data/recorded_<ts>.npz \
    --target_source reference --cmd_array target_vel --wandb
```

- **`--data`** must be the `.npz` **file**, not the directory.
- **`--target_source reference`** predicts the clip reference (what you usually want); `actual` (default) predicts the recorded robot state. History conditioning is always the actual state.
- **`--cmd_array target_vel`** = future-horizon command conditioning (past command in the encoder, future command in the decoder). Supersedes `--vel_cmd_term`. Omit it for past-only conditioning from an obs term (`--vel_cmd_term ref_base_vel`).
- Other knobs: `--num_epochs` (200), `--stride` (1; raise to cut RAM on huge datasets), `--history_len` / `--future_len` (25), `--lr`, `--cosine`, `--wandb`.
- **At startup confirm:** `target_source=…, state_dim=…, history_in_dim=…` and `future-command conditioning: on/off`. (`history_in_dim = state_dim + cmd_dim + scan_dim`.)
- **Output:** `logs/flow_matching/<run>/{best.pt, ckpt_<N>.pt}`. wandb logs per epoch (val every `--val_every`=5); per-step loss → TensorBoard (`tensorboard --logdir logs/flow_matching/<run>`).

## 3. Record a viz rollout  (needs Isaac Sim)

Runs a driver policy + the FM model, dumps states + predicted future references to a `.npz`.
Handles `cmd_array` (future-command) and EE-frame checkpoints.

```bash
python scripts/flow_matching/record_viz.py \
    --fm_checkpoint logs/flow_matching/<run>/best.pt \
    --load_run <driver policy> \
    --num_envs 8 --max_steps 500 --clip_device cpu --headless
```

- **`--load_run` (driver) must match the env's *current* policy-obs dim.** Use the motion-seq tracker the FM was trained on (recorded in the checkpoint's `meta.checkpoint_path`); if the task's policy obs changed since, that driver won't load — retrain a matching one or revert the obs change. Run headless (don't pass `--viz kit`).
- `--task` defaults to the FM checkpoint's recorded task.
- **Output:** `logs/flow_matching/viz/viz_<timestamp>.npz`. Confirm the log line `FM ready: … future_cmd=on/off, source=command-term/obs`.

## 4. View it  (no Isaac Sim)

```bash
python scripts/flow_matching/visualize.py            # newest viz_*.npz
# or:  --data logs/flow_matching/viz/viz_<ts>.npz  [--port 8080]
```

Opens a viser server (default `http://localhost:8080`): the robot rollout with FM-predicted
future references as ghost poses. Use `--ghosts` to set how many are shown.

---

## Gotchas
- **`--clip_device cpu`** on a 12 GB GPU for any step that loads the clip library (record_data, record_viz).
- **`--data` is the `.npz` file**, not its folder.
- **Record once, train many** — re-record only when the base policy or recorded layout (`--state_frames`, command terms) changes.
- The **driver policy obs must match the env obs** in both record_data and record_viz; an env obs-layout change silently strips compatibility with older trackers.
- `train.py` is Isaac-free, so FM training can run wherever (e.g. talos) without the sim stack.
