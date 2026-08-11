<!-- Copyright 2026 Zachary Olkin. All rights reserved. -->

# Flow-Matching Workflow

A conditional flow-matching transformer that predicts the teacher's **`reference_window`
observation vector directly** — up to `[base_pose(9), base_vel(6), joint_pos(nj),
joint_vel(nj), ee_pos(3K)]` per node — from **its own recent output + exteroception + a
velocity command**.

Two things follow from that, and they are the point of the design:

- **The generator's output IS the policy's observation.** There is no world-frame lift on
  the path from the network to the tracker, in simulation or at deploy. The state layout is
  `robot_rl/flow_matching/window.py`, shared by the observation, the dataset and the
  torch-free deploy port so it cannot drift between them.
- **It predicts exactly the blocks that teacher observes**, not always all five. The g1
  teacher packs `(base_pose, base_vel, joint_pos)` today → `D = 44` (`572` over the
  11-sample window), and the FM resolves the same selection from the recording
  (`RecordingMeta.reference_blocks`) rather than assuming the full `D = 82`. A dropped
  block also drops the loss terms that index it, and the agreement is checked where both
  artifacts meet: `FmTrackingCommand.reference_window_obs` in sim,
  `GenTrackerPolicy.load()` at deploy. Override with `--reference_blocks` only to train
  for a teacher other than the recorded one.
- **It is autoregressive over itself.** The `N` context tokens are the nodes it emitted at
  the `N` previous control steps, each already relative to the robot as it was at that step,
  so deploy pushes back what it just served — no base-pose estimate anywhere on the FM path.

`fm_update_every` is pinned to **1**: the window is anchored on the robot at the re-plan, so
serving it a step later would need the egomotion since then, which is exactly the
dependency this representation removes. `F = 51` follows
(`max(REFERENCE_OFFSETS) + 1`).

The caveat to keep in mind when reading results: at training time the history is the CLIP
TEACHER's applied reference, never the model's own, so this carries textbook **exposure
bias**. The defences are `null_cond_prob` and the history-token noise — which is the *same*
`REFERENCE_WINDOW_NOISE` table the policy's reference observation is corrupted with, because
both stand in for the same thing (this model's prediction error).

Pipeline: **record data → train → deploy**. Recording needs Isaac Sim; training is pure
PyTorch (no Isaac). You **record once and train many** FM variants on the same recording —
only re-record when the base policy or the recorded state layout changes.

---

## Quick start

The scripts **auto-configure from the data**: they infer `target_source`, `cmd_array` and
`clip_device` from the recording's contents, so you rarely pass those flags.

```bash
# 1. record teacher rollouts (needs Isaac). Record on the TRAIN task, not -play:
#    -play trims to 24 envs / 1 terrain row AND disables every domain randomisation,
#    including the depth-camera noise/intrinsics/extrinsics jitter. Depth is the FM's only
#    exteroceptive input, so recording on -play trains a generator that has only ever seen
#    a perfect camera. Observation corruption does NOT corrupt the recorded targets -- the
#    recorder reads the command term's world state, not the observation tensor.
python scripts/rsl_rl/record_data.py --task G1-teacher-clf-terrain-depth \
    --load_run <teacher-run> --clip <lib.h5> --clip_device cpu \
    --num_windows 200000 --history_len 8 --ref_future_len 51 --num_envs 256 --headless

# 2. train the FM (no Isaac).  --skip_distractors while the distractor family still has
#    anchor_mask == 0 clips (the recorder warns when it does).
python scripts/flow_matching/train.py --data <windows.h5> \
    --cmd_array target_vel --extero_source depth --skip_distractors --wandb

# 3. deploy: the teacher tracks the FM
python scripts/rsl_rl/play.py --task=G1-tracker-clf-fm-terrain-depth-play \
    --load_run <teacher-run> --fm_checkpoint <fm.pt> --clip <lib.h5> --clip_device cpu
```

The sections below document the underlying scripts and their (now mostly optional) flags.

---

## Exteroception: how a depth image becomes tokens

`--extero_encoder` picks the embedding, and it is the one architectural choice that is
about TERRAIN rather than about capacity.

* **`cnn`** (what `auto` picks for depth) — a 3-layer stride-2 conv trunk **per camera**
  takes each 30x26 frame to a 4x4 grid, and every cell becomes one token carrying a learned
  spatial-slot embedding. 16 tokens per camera, 32 total, sequence length 111.
* **`mlp`** — the original: the whole 780-value frame through one `Linear`, one token per
  camera, sequence length 81. The only option for `--extero_source height_scan`, and the
  baseline the CNN is measured against.

Why it matters: a `Linear` over 780 pixels is **permutation-invariant over pixels**, so
nothing in the flat model can attend to *where* in the frame a step edge is. That is the
shape of the measured failure — displace the robot 0.10 / 0.20 m downrange and the predicted
stair segment slides 0.098 / 0.168 m with it, where the anchor mask requires 0.000
(`scripts/rsl_rl/fm_registration_probe.py`). The convolutional branch is also ~5x *smaller*
than the `Linear` it replaces, so this re-allocates capacity toward structure rather than
adding it.

Two knobs that separate the variables when reading an A/B:

```bash
--extero_cnn_grid 1,1        # a conv trunk at the LEGACY token count: isolates
                             # "CNN beats MLP" from "16 tokens beat 1"
--extero_norm_mode per_sensor  # one (mean, std) per camera instead of per PIXEL
```

`per_sensor` exists because the default per-pixel normalization is a spatially *varying*
transform applied before the convolution sees the image — exactly what weight sharing
assumes away. It is a separate flag so it can be ablated: if a CNN run disappoints, re-run
with `--extero_norm_mode per_element` before blaming the encoder.

Pixel order is a **contract**, not a convention: the flat row is row-major over `(H, W)`
with `v = 0` the top row, and the `(H, W)` to fold it with is derived from the recording's
`depth_camera_spec` — never configured. `extero_sensor_dim` alone cannot tell a 30x26 camera
from a 26x30 one, so the shape travels into `fm.onnx` and the deploy refuses a mismatch
(`transfer/deploy/gen_tracker.py`).

### Changing the generator's resolution without retraining the teacher

The generator can sample the policy's lenses on **its own grid**, so the encoder's input can
change on an FM retrain alone:

```bash
# 1. record: the policy keeps 30x26, the generator's array is 60x52
python scripts/rsl_rl/record_data.py --task G1-teacher-clf-terrain-depth \
    --load_run <teacher> --clip <lib>.h5 --clip_device cpu \
    --num_windows 200000 --history_len 8 --ref_future_len 62 \
    --fm_depth_resolution 60x52 --num_envs 256 --headless

# 2. train: the dataset DERIVES 60x52 from the recording's camera contract
python scripts/flow_matching/train.py --data <windows>.h5 --extero_encoder cnn ...

# 3. play: the resolution is read off the checkpoint and the pair is built to match
python scripts/rsl_rl/play.py --task=G1-tracker-clf-fm-terrain-depth-play \
    --load_run <teacher> --fm_checkpoint <fm.pt> --clip <lib>.h5
```

Only the SAMPLING differs. Both pairs are built from the same `_CameraSpec`, so mount,
apertures, range and normalization are shared by construction, and the three DR events draw
ONCE for both — a mount cannot be in two places, and independent draws would make them two
different physical cameras.

Costs, which is why the resolution is worth choosing before recording rather than after: ray
cost and the `depth` dataset both scale with `W*H`, so 60x52 is **4x** the 30x26 pair. A
200k-window recording is 1.25 GB at 30x26 and 5.0 GB at 60x52 (29 GB at 128x112).

**Raising the resolution raises the TOKEN COUNT unless you say otherwise**, which is the trap
here: the grid is the image ceil-halved once per conv layer, so a bigger image leaves a bigger
grid and the transformer's sequence grows with it. Measured:

| cameras | flags | tokens/cam | extero tokens | sequence |
|---|---|---|---|---|
| 30x26 | — | 16 | 32 | 111 |
| 60x52 | — | **56** | **112** | **191** |
| 60x52 | `--extero_cnn_channels 16,32,64,128` | 16 | 32 | 111 |
| 60x52 | `--extero_cnn_grid 4,4` | 16 | 32 | 111 |

Prefer the 4th conv layer over pooling: it never computes the resolution it would then throw
away, and the extra layer widens the receptive field, which is the point of the finer grid.

MuJoCo casts both grids. **Hardware does not** — `export_sensor_configs.py` raises rather
than emitting obelisk/ZED configs that cover only the tracker, so a split-sampling generator
has to be retrained at the policy's resolution before it can go on the robot.

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
  - `--target_source` → `ego_anchored_clip_future` (the only mode the dataset accepts) when the recording has `ref_future`.
  - `--cmd_array` → `target_vel` if that array was recorded, else no command (future-horizon conditioning: past command in the encoder, future in the decoder). Override with an array name or `none`.
- **`--eval_data <run-or-.npz>`** — a *separate* held-out recording (never trained on); metrics logged under `eval/*`, distinct from the within-recording `val/*`. Must share the train layout. See "Held-out evaluation" below.
- **`--run_name`** — run dir is `logs/flow_matching/<timestamp>_<run_name>` (defaults to just the timestamp).
- **`--history_source`** — what the `N` history tokens are made of. `measured` (default) is the robot's actual state. `reference_partial` is **partially autoregressive**: measured root pose/twist, but joints and end-effector positions taken from the reference applied at those steps — the generator conditioning on its own recent output for posture while the root stays grounded in reality. Both train from the same recording (the applied-reference ring is always stored), so it is a free A/B.
- Other knobs: `--num_epochs` (100), `--stride` (1; raise to cut RAM on huge datasets), `--num_history_states` (2), `--future_len` (65 — sized to the deploy look-ahead; see `FlowMatchingConfig.future_len`), `--lr`, `--cosine`.

**Horizon and re-recording.** `--future_len` must equal the recording's `--ref_future_len`: windows are cut at record time and the dataset hard-errors on a mismatch. Both default to `FlowMatchingConfig.future_len`, so leaving them alone is correct.
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
