# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn a generator's own held-out error into a tokenizable residual dataset.

Two stages, sharing one provenance:

``--stage audit``
    Free of the generator and of any GPU. Decides WHICH held-out rows carry an honest
    residual, and reports the answer instead of assuming it. See :func:`audit` for why this
    is a gate rather than a formality.

``--stage build``
    Runs the generator at ``x0 = 0`` over the chosen rows, forms ``delta = gt - P``, fits the
    residual normalizer and writes a cache the tokenizer eats unchanged.

The generator is sampled the way the DEPLOYED controller samples it -- ``zero_noise``, so the
plan the adjuster learns to correct is the plan the robot will actually be handed. That also
makes this whole build deterministic and removes the per-row seed the general form would need.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))

import recorded_data_io  # noqa: E402
from recorded_data_io import WINDOW_INDEX_KEYS, load_recorded_windows  # noqa: E402


def parse_args() -> argparse.Namespace:
    """CLI. See the module docstring for what the two stages do."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("audit", "build"), required=True)
    p.add_argument("--fm_run", required=True,
                   help="A flow-matching run directory. Its fm_config.yaml and data.yaml are "
                        "read from the run root or from params/ -- both layouts exist.")
    p.add_argument("--checkpoint", default="best.pt",
                   help="Which checkpoint in --fm_run supplies the weights and normalizers.")
    p.add_argument("--data", default="",
                   help="Comma-separated recordings, overriding the run's own data.yaml. The "
                        "ORDER matters: episode ids are offset per file, so a swap renumbers "
                        "every episode and the recorded held-out set stops meaning anything.")
    p.add_argument("--out", default="",
                   help="audit: where to write the JSON report.")
    p.add_argument("--cache", default="",
                   help="build: where to write the residual cache .pt.")
    p.add_argument("--honesty", choices=("clean", "clean+twin"), default="clean",
                   help="build: which held-out rows to keep. 'clean' takes only rows whose "
                        "clip appears in no training window; 'clean+twin' also takes rows "
                        "whose clip reached training through the other file's augmented "
                        "re-render. Run --stage audit first -- it MEASURES whether the two "
                        "differ rather than leaving it to be assumed.")
    p.add_argument("--clip", default="",
                   help="Clip library .h5, enabling the per-node anchor mask that "
                        "train_tokenizer.py --pinned_axis_weight needs. Without it the cache "
                        "carries no mask and a pinned-axis run is refused rather than "
                        "silently unweighted.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--audit_windows", type=int, default=6000,
                   help="audit: cap on the rows sampled PER ARM for the |delta| comparison. "
                        "0 skips the generator entirely and reports only the partition.")
    p.add_argument("--chunk_len", type=int, default=0,
                   help="build: nodes per chunk; 0 takes the generator's own future_len. The "
                        "tokenizer refuses chunk_len != future_len, so this exists only to "
                        "make the coupling explicit at the CLI.")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="build: fraction of the residual set held out for the gate, split by "
                        "CLIP id rather than episode id -- the same reason the audit uses "
                        "clip_id, applied to this stage's own split.")
    p.add_argument("--absolute", action="store_true",
                   help="build: write an ABSOLUTE cache (the clip's own future, in the "
                        "generator's own state normalizer) over the SAME honest rows and the "
                        "SAME split as the residual build. That is the third arm the plan-"
                        "space gate needs: it says whether spending N tokens on the residual "
                        "beats spending them on the trajectory itself, which is the actual "
                        "architectural question and is only answerable on identical windows.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero_chunk_frac", type=float, default=0.05,
                   help="build: fraction of ADDITIONAL all-zero residual chunks appended to "
                        "the train side, so 'the generator was right' is a thing the codebook "
                        "has seen. Counted in the provenance rather than sampled at runtime, "
                        "so the injection is auditable.")
    return p.parse_args()


def run_yaml(run_dir: str, name: str) -> dict[str, Any]:
    """Read one of a run's recorded YAMLs, from either layout.

    ``scripts/flow_matching/train.py`` writes ``params/<name>.yaml``; the hand-assembled
    comparison runs on disk keep the same files at the run root. Both are real, so both are
    accepted -- and a missing file raises rather than defaulting, because every value here is
    a statement about what the checkpoint was trained on.

    Args:
        run_dir: The run directory.
        name: ``"fm_config"`` or ``"data"``.

    Returns:
        The parsed YAML.

    Raises:
        SystemExit: If neither layout holds the file.
    """
    for rel in (os.path.join("params", f"{name}.yaml"), f"{name}.yaml"):
        path = os.path.join(run_dir, rel)
        if os.path.exists(path):
            with open(path) as fh:
                return yaml.safe_load(fh)
    raise SystemExit(
        f"{run_dir} holds no {name}.yaml (looked in params/ and the run root). Without it "
        "the split and the filters this checkpoint trained under are unknown, and guessing "
        "them would silently redefine what 'held out' means."
    )


def index_arrays(paths: list[str]) -> tuple[dict[str, np.ndarray], list[int], Any]:
    """Load only the per-window INDEX arrays, concatenated the way training concatenates them.

    Reproduces ``train.py``'s per-file ``episode_id`` offset exactly -- each subsequent file is
    pushed past the running maximum -- because the recorded ``val_episode_ids`` are ids in
    that offset space and mean nothing without it.

    ``clip_id`` is deliberately NOT offset, and that is the point of this function: it indexes
    the clip LIBRARY, so it is global across files where ``episode_id`` is per-file. It is
    therefore the only key under which "did the generator see this motion" is answerable at
    all once a recording has an augmented twin.

    Args:
        paths: The recordings, in the order training saw them.

    Returns:
        ``(arrays, per_file_window_counts, meta_of_first_file)``.
    """
    parts = [load_recorded_windows(p, keys=list(WINDOW_INDEX_KEYS)) for p in paths]
    counts = [int(d["episode_id"].shape[0]) for d, _ in parts]
    data = {k: np.asarray(v) for k, v in parts[0][0].items()}
    ep_off = int(data["episode_id"].max()) + 1
    for extra, _ in parts[1:]:
        for k in data:
            add = np.asarray(extra[k])
            data[k] = np.concatenate([data[k], add + ep_off if k == "episode_id" else add])
        ep_off = int(data["episode_id"].max()) + 1
    return data, counts, parts[0][1]


def filtered_windows(
    data: dict[str, np.ndarray], meta: Any, cfg: Any
) -> np.ndarray:
    """Row indices surviving the run's own distractor and family filters.

    Uses the dataset's OWN mask functions rather than restating the rule. The recorded
    ``val_episode_ids`` were drawn from the POST-filter episode set, so a build that filtered
    differently would compare against a split that never existed -- and the failure would look
    like a slightly different window count, not like an error.

    Args:
        data: The concatenated index arrays.
        meta: The first file's :class:`RecordingMeta` (``family_names`` is read).
        cfg: The run's :class:`FlowMatchingConfig`.

    Returns:
        ``(W_kept,)`` int64 row indices into the concatenated arrays.
    """
    from robot_rl.flow_matching.dataset import (  # noqa: PLC0415
        _distractor_keep_mask,
        _family_keep_mask,
    )

    w = int(data["episode_id"].shape[0])
    keep = _distractor_keep_mask(data, cfg, w)
    fam = _family_keep_mask(data, meta, cfg, w)
    mask = fam if keep is None else (keep if fam is None else (keep & fam))
    return np.arange(w, dtype=np.int64) if mask is None else np.nonzero(mask)[0]


def partition_honesty(
    data: dict[str, np.ndarray], rows: np.ndarray, val_episode_ids: list[int]
) -> dict[str, np.ndarray]:
    """Split the held-out rows into CLEAN and TWIN-LEAKED, by clip.

    The generator's ``val_episode_ids`` are held out by ``episode_id``, and ``train.py``
    offsets that id per input file -- so when a run trains on a recording PLUS an augmented
    re-render of it, an episode's twin carries a different id and can sit in training while
    the episode itself is held out. The residual on such a row is not the generator's error on
    unseen motion; it is its error on motion it was fitted to, displaced.

    ``clip_id`` is what makes the question answerable: it indexes the clip library, so it is
    the same integer for an episode and for every re-render of it.

    Args:
        data: The concatenated index arrays.
        rows: Post-filter row indices.
        val_episode_ids: The run's recorded held-out episode ids.

    Returns:
        ``{"train": rows, "clean": rows, "twin": rows}`` -- the training rows, the held-out
        rows whose clip never appears in training, and the held-out rows whose clip does.
    """
    ep = data["episode_id"][rows]
    clip = data["clip_id"][rows]
    in_val = np.isin(ep, np.asarray(sorted(val_episode_ids), dtype=ep.dtype))
    train_clips = np.unique(clip[~in_val])
    leaked = np.isin(clip, train_clips)
    return {
        "train": rows[~in_val],
        "clean": rows[in_val & ~leaked],
        "twin": rows[in_val & leaked],
    }


def split_rows_per_file(rows: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    """Map concatenated row indices back to per-file row indices.

    ``load_recorded_windows(..., rows=)`` reads a subset of ONE file, which is what keeps a
    37 GB pair of recordings off this machine's RAM. The honesty partition works in the
    concatenated space, so the two have to be reconciled.

    Args:
        rows: Sorted row indices into the concatenation.
        counts: Windows per file, in order.

    Returns:
        One ``(n_i,)`` int64 array of LOCAL rows per file.
    """
    out, start = [], 0
    for n in counts:
        sel = rows[(rows >= start) & (rows < start + n)] - start
        out.append(np.ascontiguousarray(sel))
        start += n
    return out


def load_rows(paths: list[str], rows: np.ndarray, counts: list[int],
              keys: list[str]) -> tuple[dict[str, np.ndarray], Any]:
    """Read exactly the selected windows, reassembled in concatenated order.

    Args:
        paths: The recordings, in training order.
        rows: Sorted concatenated row indices.
        counts: Windows per file.
        keys: Arrays to read.

    Returns:
        ``(data, meta)`` -- meta is the FIRST file's, matching what training keeps.
    """
    per_file = split_rows_per_file(rows, counts)
    parts = [load_recorded_windows(p, keys=keys, rows=r) if r.size else None
             for p, r in zip(paths, per_file)]
    live = [(d, m) for d, m in (x for x in parts if x is not None)]
    if not live:
        raise SystemExit("the honesty selection kept no windows in any file.")
    data = {k: np.concatenate([d[k] for d, _ in live]) for k in live[0][0]}
    return data, live[0][1]


def generator_config(run_dir: str, checkpoint: str, clip_library: str) -> tuple[Any, Any, str]:
    """The checkpoint's own config, with the three eval-only overrides applied.

    Args:
        run_dir: The run directory.
        checkpoint: Checkpoint filename within it.
        clip_library: A clip ``.h5`` for the anchor-mask join, or ``""``.

    Returns:
        ``(cfg, ckpt, path)``.
    """
    import torch  # noqa: PLC0415

    from motion_gen.features import DEFAULT_MJCF  # noqa: PLC0415
    from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: PLC0415

    path = os.path.join(run_dir, checkpoint)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    # The transition sampler oversamples the TRAIN split; on an eval selection it would
    # silently reweight which windows are measured.
    cfg.transition_fraction = None
    if cfg.fk_model_path and not os.path.exists(cfg.fk_model_path):
        cfg.fk_model_path = str(DEFAULT_MJCF)
    if clip_library:
        cfg.clip_library_path = clip_library
        cfg.pinned_axis_metrics = True
    else:
        # Same overrides train_planner applies for an eval build. The checkpoint trained with
        # pinned_axis_loss_weight=20, which makes the clip-library join FATAL -- correct for
        # training, wrong here, where the mask is an optional extra the audit never reads.
        # Turning them off is what lets the audit run without the library; the BUILD asks for
        # --clip and then the join is fatal again, which is what it should be.
        cfg.pinned_axis_loss_weight = 1.0
        cfg.pinned_axis_metrics = False
    return cfg, ckpt, path


def sample_residuals(paths: list[str], rows: np.ndarray, counts: list[int], run_dir: str,
                     checkpoint: str, clip_library: str, device: str, batch_size: int,
                     future_len: int | None = None) -> dict[str, Any]:
    """Run the generator over ``rows`` and return its plans, the clip targets and the residual.

    The generator is loaded with ``zero_noise=True`` and served through
    ``flow_matching.online.sample_window``, so ``P`` is bit-identical to what Isaac and the
    C++ runtime hand the tracker -- including the SO(3) projection. Anything short of that
    would fit a correction to a plan nobody serves.

    The dataset is constructed with the CHECKPOINT's normalizers rather than refitting: a
    refit on an eval subset is a different space, and the difference is invisible in every
    shape.

    Args:
        paths: The recordings, in training order.
        rows: Sorted concatenated row indices to run.
        counts: Windows per file.
        run_dir: The generator's run directory.
        checkpoint: Checkpoint filename.
        clip_library: Clip ``.h5`` for the anchor mask, or ``""``.
        device: Torch device.
        batch_size: Windows per generator call.
        future_len: Override the checkpoint's horizon (never used in anger; the tokenizer's
            chunk length must equal it).

    Returns:
        ``{"plan", "target", "residual", "family", "anchor_mask", "mask_frame_yaw", "cfg",
        "ckpt", "ctx", "dataset"}``. The three windows are RAW-unit ``(Nw, F, D)`` float32.
    """
    import torch  # noqa: PLC0415

    from robot_rl.flow_matching.dataset import FutureStateDataset  # noqa: PLC0415
    from robot_rl.flow_matching.normalizer import Normalizer  # noqa: PLC0415
    from robot_rl.flow_matching.online import load_fm_context, sample_window  # noqa: PLC0415
    from robot_rl.flow_matching.window import decompose_window_residual  # noqa: PLC0415

    cfg, _, path = generator_config(run_dir, checkpoint, clip_library)
    if future_len is not None:
        cfg.future_len = int(future_len)
    ctx = load_fm_context(path, torch.device(device), zero_noise=True)
    # A residual is only "what the generator got wrong on unseen conditioning" if the
    # conditioning is the arm the checkpoint was fitted for. The mismatch is otherwise
    # width-identical and completely silent.
    print(f"[residual] history_source={cfg.history_source} (the arm this checkpoint states); "
          f"future_len={cfg.future_len}, num_history_states={cfg.num_history_states}")

    keys = [k for k in recorded_data_io.WINDOW_ARRAY_KEYS
            if k not in ("height_scan", "depth") or k == cfg.extero_source]
    data, meta = load_rows(paths, rows, counts, keys)
    # CPU COPIES of the checkpoint's own normalizers. ``load_fm_context`` moves them to the
    # sampling device, but ``FutureStateDataset`` normalizes host tensors at construction, so
    # handing it the GPU ones is a device error. Rebuilding from the state dict rather than
    # refitting is the point: a refit on an eval subset is a different space, and nothing in
    # the shapes would say so.
    cpu_norm = {
        name: Normalizer.from_state_dict(getattr(ctx, attr).state_dict())
        for name, attr in (("state", "state_norm"), ("cmd", "cmd_norm"),
                           ("extero", "extero_norm"))
    }
    ds = FutureStateDataset(
        data, meta, cfg, np.arange(int(rows.shape[0])),
        state_normalizer=cpu_norm["state"], cmd_normalizer=cpu_norm["cmd"],
        extero_normalizer=cpu_norm["extero"], add_noise=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    dev = torch.device(device)
    plans, targets, masks, yaws = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            cmd = batch["cmd_tokens"].to(dev) if "cmd_tokens" in batch else None
            p = sample_window(ctx, batch["history"].to(dev), batch["extero"].to(dev), cmd)
            plans.append(p.float().cpu())
            targets.append(ctx.state_norm.denormalize(batch["state"].to(dev)).float().cpu())
            # The dataset renames this on the way out (``mask_frame_yaw_rel`` there,
            # ``mask_frame_yaw`` on a ChunkData); carrying both halves or neither is enforced
            # by ChunkDataset, so they travel together here too.
            if "anchor_mask" in batch:
                masks.append(batch["anchor_mask"].cpu())
                yaws.append(batch["mask_frame_yaw_rel"].cpu())
    plan = torch.cat(plans)
    target = torch.cat(targets)
    return {
        "plan": plan,
        "target": target,
        "residual": decompose_window_residual(plan, target),
        "anchor_mask": torch.cat(masks) if masks else None,
        "mask_frame_yaw": torch.cat(yaws) if yaws else None,
        "cfg": cfg,
        "ctx": ctx,
        "dataset": ds,
        "checkpoint_path": path,
    }


def group_stats(residual: Any, num_joints: int, num_ee: int, blocks: Any) -> dict[str, dict]:
    """Per-channel-group ``mean / p50 / p99`` of ``|residual|``, in physical units.

    Args:
        residual: ``(Nw, F, D)`` raw-unit residuals.
        num_joints: Robot joint count.
        num_ee: End-effector frames.
        blocks: The block selection.

    Returns:
        ``{group: {"mean", "p50", "p99", "rms"}}`` in metres / rad / m/s / rad/s.
    """
    import torch  # noqa: PLC0415

    from robot_rl.token_planner.metrics import channel_group_labels  # noqa: PLC0415

    names, labels = channel_group_labels(num_joints, num_ee, blocks)
    out: dict[str, dict] = {}
    for g, name in enumerate(names):
        sel = residual[..., labels == g].abs().flatten().double()
        q = torch.quantile(sel[:: max(1, sel.numel() // 2_000_000)], torch.tensor([0.5, 0.99],
                                                                                 dtype=sel.dtype))
        out[name] = {
            "mean": float(sel.mean()),
            "p50": float(q[0]),
            "p99": float(q[1]),
            "rms": float(sel.pow(2).mean().sqrt()),
        }
    return out


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], dict[str, np.ndarray], list[int]]:
    """Decide which held-out rows carry an honest residual, and MEASURE the difference.

    The chain hangs on "the generator did not see this motion", and one thing quietly breaks
    it: a run trained on a recording plus an augmented re-render of it holds out by
    ``episode_id``, which ``train.py`` offsets per file, so an episode's twin can sit in
    training. That is the same failure ``merge_recordings.py --variant_groups`` exists for on
    the val split.

    Rather than assume the twin rows are spoiled, this compares ``|delta|`` on them against
    the clean rows. If the two agree, the augmentation is far enough from its original that it
    does not count as having seen the row and every held-out episode is usable; if the twin
    rows' residuals are visibly smaller, they are the generator recognising its own training
    data and only the clean rows are.

    Args:
        args: Parsed CLI.

    Returns:
        ``(report, paths, partition, per_file_counts)`` -- the caller needs the last three to
        run the ``|delta|`` half without re-reading the index arrays, which is the only
        expensive part of this stage.
    """
    dcfg = run_yaml(args.fm_run, "data")
    paths = [p for p in (args.data or dcfg["data"]).split(",") if p.strip()]
    val_ids = [int(e) for e in (dcfg.get("val_episode_ids") or [])]
    if not val_ids:
        raise SystemExit(
            f"{args.fm_run} records no val_episode_ids, so it cannot say which rows it held "
            "out. Without that the 'no retrain' arm has no honest subset to draw from."
        )
    cfg, _, _ = generator_config(args.fm_run, args.checkpoint, args.clip)
    data, counts, meta = index_arrays(paths)
    rows = filtered_windows(data, meta, cfg)
    part = partition_honesty(data, rows, val_ids)

    fam_names = list(getattr(meta, "family_names", []) or [])
    report: dict[str, Any] = {
        "fm_run": os.path.abspath(args.fm_run),
        "checkpoint": args.checkpoint,
        "history_source": cfg.history_source,
        "data": paths,
        "windows_per_file": counts,
        "windows_after_filters": int(rows.size),
        "val_episodes_recorded": len(val_ids),
        "counts": {k: int(v.size) for k, v in part.items()},
        "clips": {k: int(np.unique(data["clip_id"][v]).size) for k, v in part.items()},
    }
    for k, v in part.items():
        if v.size and fam_names:
            codes, n = np.unique(data["clip_family"][v], return_counts=True)
            report.setdefault("families", {})[k] = {
                fam_names[int(c)] if int(c) < len(fam_names) else str(int(c)): int(x)
                for c, x in zip(codes, n)
            }
    print(f"[audit] {report['windows_after_filters']} windows after the run's own filters; "
          f"held out: {report['counts']['clean']} CLEAN / {report['counts']['twin']} "
          f"TWIN-LEAKED ({report['clips']['clean']} / {report['clips']['twin']} clips)")
    return report, paths, part, counts


def audit_delta(args: argparse.Namespace, report: dict[str, Any], paths: list[str],
                part: dict[str, np.ndarray], counts: list[int]) -> None:
    """Add the clean-vs-twin ``|delta|`` comparison to ``report``, in place.

    This is the measurement the whole stage exists for. A sample is enough: the question is
    whether two DISTRIBUTIONS differ, not what either one's tail is, and running the generator
    over every held-out row twice would cost more than the build it is gating.

    Args:
        args: Parsed CLI.
        report: The audit report, written through.
        paths: The recordings.
        part: The honesty partition.
        counts: Windows per file.
    """
    rng = np.random.default_rng(args.seed)
    for arm in ("clean", "twin"):
        rows = part[arm]
        if rows.size == 0:
            report.setdefault("delta", {})[arm] = None
            continue
        take = np.sort(rng.choice(rows, size=min(args.audit_windows, rows.size), replace=False))
        got = sample_residuals(paths, take, counts, args.fm_run, args.checkpoint, args.clip,
                               args.device, args.batch_size)
        ds = got["dataset"]
        report.setdefault("delta", {})[arm] = {
            "windows": int(take.size),
            "groups": group_stats(got["residual"], ds.num_joints,
                                  len(ds.target_frame_names), got["cfg"].reference_blocks),
        }
        print(f"[audit] |delta| on {arm} ({take.size} windows):")
        for g, s in report["delta"][arm]["groups"].items():
            print(f"          {g:<14} rms {s['rms']:.4f}  p50 {s['p50']:.4f}  "
                  f"p99 {s['p99']:.4f}")
    clean, twin = report.get("delta", {}).get("clean"), report.get("delta", {}).get("twin")
    if clean and twin:
        ratio = {g: twin["groups"][g]["rms"] / max(clean["groups"][g]["rms"], 1e-12)
                 for g in clean["groups"]}
        report["twin_over_clean_rms"] = ratio
        worst = min(ratio.values())
        print(f"[audit] twin/clean RMS by group: "
              + "  ".join(f"{g} {r:.3f}" for g, r in ratio.items()))
        print("[audit] VERDICT: " + (
            "the twin rows' residuals are indistinguishable from the clean ones, so the "
            "augmented re-render did not teach the generator this motion -- --honesty "
            "clean+twin is sound."
            if worst > 0.9 else
            f"the twin rows' residuals are up to {1 / worst:.2f}x SMALLER than the clean "
            "ones, which is the generator recognising motion it was fitted to. Use "
            "--honesty clean."))


def build(args: argparse.Namespace) -> str:
    """Run the generator over the honest rows and write the residual cache.

    Args:
        args: Parsed CLI.

    Returns:
        The cache path.
    """
    import torch  # noqa: PLC0415

    from robot_rl.token_planner.residual import (  # noqa: PLC0415
        ResidualChunkData,
        fit_residual_normalizer,
        save_residual_cache,
    )

    dcfg = run_yaml(args.fm_run, "data")
    paths = [p for p in (args.data or dcfg["data"]).split(",") if p.strip()]
    val_ids = [int(e) for e in (dcfg.get("val_episode_ids") or [])]
    cfg, ckpt, ckpt_path = generator_config(args.fm_run, args.checkpoint, args.clip)
    data, counts, meta = index_arrays(paths)
    rows = filtered_windows(data, meta, cfg)
    part = partition_honesty(data, rows, val_ids)

    keep = (part["clean"] if args.honesty == "clean"
            else np.sort(np.concatenate([part["clean"], part["twin"]])))
    if keep.size == 0:
        raise SystemExit(
            f"--honesty {args.honesty} kept no windows. The audit stage reports the partition; "
            "a build on an empty selection would be a successful-looking no-op.")
    print(f"[build] --honesty {args.honesty}: {keep.size} windows "
          f"({np.unique(data['clip_id'][keep]).size} clips)")

    got = sample_residuals(paths, keep, counts, args.fm_run, args.checkpoint, args.clip,
                           args.device, args.batch_size,
                           future_len=args.chunk_len or None)
    ds, delta, plan = got["dataset"], got["residual"], got["plan"]

    # Split by CLIP, not episode: the same reason the audit does. An episode and its augmented
    # re-render share a clip, so an episode split would put a row and its own twin on opposite
    # sides of this stage's gate too.
    clips = data["clip_id"][keep]
    uniq = np.unique(clips)
    rng = np.random.default_rng(args.seed)
    n_val = int(round(args.val_fraction * uniq.size))
    val_clips = set(rng.choice(uniq, size=n_val, replace=False).tolist()) if n_val else set()
    in_val = torch.from_numpy(np.isin(clips, list(val_clips)) if val_clips
                              else np.zeros(clips.size, dtype=bool))
    fam = torch.from_numpy(data["clip_family"][keep].astype(np.int64))

    # Fitted on TRAIN only, for the same reason every normalizer in this repo is.
    if args.absolute:
        return _write_absolute(args, got, in_val, fam, data, keep, cfg, meta, dcfg, ckpt)

    norm = fit_residual_normalizer(delta[~in_val])
    fitted_mean = delta[~in_val].reshape(-1, delta.shape[-1]).mean(dim=0)
    nodes = norm.normalize(delta)

    amask, ayaw = got["anchor_mask"], got["mask_frame_yaw"]
    if amask is None and args.clip:
        raise SystemExit(
            "--clip was given but the dataset produced no anchor mask, so a later "
            "--pinned_axis_weight run would be refused. Check the clip library joins.")

    zero_n = int(round(args.zero_chunk_frac * int((~in_val).sum())))
    train_nodes = nodes[~in_val]
    train_fam, train_plan = fam[~in_val], plan[~in_val]
    train_mask = None if amask is None else amask[~in_val]
    train_yaw = None if ayaw is None else ayaw[~in_val]
    if zero_n:
        # "The generator was right here" has to be something the codebook has SEEN, or the
        # adjuster has no id to say it with. Appended rather than sampled, so the count is in
        # the provenance and the val split is untouched.
        z = torch.zeros((zero_n, *train_nodes.shape[1:]), dtype=train_nodes.dtype)
        train_nodes = torch.cat([train_nodes, z])
        train_fam = torch.cat([train_fam, torch.zeros(zero_n, dtype=train_fam.dtype)])
        # The tokenizer never sees a plan, so which plan an all-zero residual is paired with
        # is arbitrary; reusing real ones keeps every per-window array the same length, which
        # ChunkDataset checks.
        train_plan = torch.cat([train_plan, train_plan[:zero_n]])
        if train_mask is not None:
            train_mask = torch.cat([train_mask, train_mask[:zero_n]])
            train_yaw = torch.cat([train_yaw, train_yaw[:zero_n]])
    print(f"[build] {int((~in_val).sum())} train / {int(in_val.sum())} val chunks"
          f"{f' (+{zero_n} injected all-zero)' if zero_n else ''}")

    fm_prov = {
        "fm_run": os.path.abspath(args.fm_run),
        "fm_sha256": recorded_data_io.sha256_file(ckpt_path),
        "fm_epoch": int(ckpt.get("epoch", -1)),
        "draw": "zero_noise",
        "history_source": cfg.history_source,
        "honesty": args.honesty,
        "checkpoint": args.checkpoint,
    }
    prov = {
        "data": paths,
        "data_sha256": dcfg.get("data_sha256", ""),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "stride": 1,
        "skip_distractors": bool(cfg.skip_distractors),
        "skip_families": list(cfg.skip_families or []),
        "state_scales": {n: getattr(cfg, f"state_scale_{n}", None)
                         for n in ("base_pos", "base_lin_vel", "base_ang_vel",
                                   "joint_pos", "joint_vel", "ee_pos")},
        "load_rows": 0,
        "zero_chunks_injected": zero_n,
        "honest_windows": int(keep.size),
        "residual_fitted_mean": fitted_mean.tolist(),
        "counts": {k: int(v.size) for k, v in part.items()},
    }
    chunks = ResidualChunkData(
        train_nodes=train_nodes, val_nodes=nodes[in_val],
        train_family=train_fam, val_family=fam[in_val],
        state_normalizer_state=norm.state_dict(),
        family_names=list(getattr(meta, "family_names", []) or []),
        reference_blocks=list(cfg.reference_blocks),
        num_joints=int(ds.num_joints), num_ee_frames=len(ds.target_frame_names),
        chunk_len=int(delta.shape[1]), node_dim=int(delta.shape[2]), dt=float(cfg.dt),
        val_episode_ids=sorted(val_clips), provenance=prov,
        train_anchor_mask=train_mask,
        val_anchor_mask=None if amask is None else amask[in_val],
        train_mask_frame_yaw=train_yaw,
        val_mask_frame_yaw=None if ayaw is None else ayaw[in_val],
        train_plan=train_plan, val_plan=plan[in_val],
        absolute_normalizer_state=got["ctx"].state_norm.state_dict(),
        fm_provenance=fm_prov,
    )
    out = args.cache or os.path.join("cache", "residual.pt")
    save_residual_cache(out, chunks)
    print(f"[build] wrote {out}")
    return out


def main() -> None:
    """Entry point."""
    args = parse_args()
    if args.stage == "audit":
        report, paths, part, counts = audit(args)
        if args.audit_windows > 0:
            audit_delta(args, report, paths, part, counts)
        out = args.out or os.path.join("cache", "residual_audit.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[audit] wrote {out}")
    else:
        build(args)



def _write_absolute(args, got, in_val, fam, data, keep, cfg, meta, dcfg, ckpt) -> str:
    """Write the ABSOLUTE arm of the gate: the clip's own future over the same honest rows.

    Paired by construction -- same generator, same honesty selection, same clip-keyed split,
    same normalizer the generator itself uses -- so the two tokenizers differ in exactly one
    thing: what they are asked to compress. Without that pairing "the residual codes better"
    would be confounded with a different set of windows.

    Args:
        args: Parsed CLI.
        got: The output of :func:`sample_residuals`.
        in_val: ``(Nw,)`` bool held-out mask.
        fam: ``(Nw,)`` family codes.
        data: The concatenated index arrays.
        keep: The kept row indices.
        cfg: The generator config.
        meta: The recording meta.
        dcfg: The run's ``data.yaml``.
        ckpt: The loaded checkpoint.

    Returns:
        The cache path.
    """
    import torch  # noqa: PLC0415

    from robot_rl.flow_matching.normalizer import Normalizer  # noqa: PLC0415
    from robot_rl.token_planner.data import ChunkData, save_chunk_cache  # noqa: PLC0415

    ds = got["dataset"]
    norm = Normalizer.from_state_dict(got["ctx"].state_norm.state_dict())
    nodes = norm.normalize(got["target"])
    amask, ayaw = got["anchor_mask"], got["mask_frame_yaw"]
    prov = {
        "data": [p for p in (args.data or dcfg["data"]).split(",") if p.strip()],
        "data_sha256": dcfg.get("data_sha256", ""),
        "seed": args.seed, "val_fraction": args.val_fraction, "stride": 1,
        "skip_distractors": bool(cfg.skip_distractors),
        "skip_families": list(cfg.skip_families or []),
        "state_scales": {n: getattr(cfg, f"state_scale_{n}", None)
                         for n in ("base_pos", "base_lin_vel", "base_ang_vel",
                                   "joint_pos", "joint_vel", "ee_pos")},
        "load_rows": 0,
        "paired_with": "the residual build over the same rows",
        "honesty": args.honesty,
        "honest_windows": int(keep.size),
    }
    chunks = ChunkData(
        train_nodes=nodes[~in_val], val_nodes=nodes[in_val],
        train_family=fam[~in_val], val_family=fam[in_val],
        state_normalizer_state=norm.state_dict(),
        family_names=list(getattr(meta, "family_names", []) or []),
        reference_blocks=list(cfg.reference_blocks),
        num_joints=int(ds.num_joints), num_ee_frames=len(ds.target_frame_names),
        chunk_len=int(nodes.shape[1]), node_dim=int(nodes.shape[2]), dt=float(cfg.dt),
        val_episode_ids=[], provenance=prov,
        train_anchor_mask=None if amask is None else amask[~in_val],
        val_anchor_mask=None if amask is None else amask[in_val],
        train_mask_frame_yaw=None if ayaw is None else ayaw[~in_val],
        val_mask_frame_yaw=None if ayaw is None else ayaw[in_val],
    )
    out = args.cache or os.path.join("cache", "absolute.pt")
    save_chunk_cache(out, chunks)
    print(f"[build] ABSOLUTE arm: {int((~in_val).sum())} train / {int(in_val.sum())} val "
          f"chunks over the same rows; wrote {out}")
    return out


if __name__ == "__main__":
    main()
