# Copyright 2026 Zachary Olkin. All rights reserved.

"""One training datapoint, end to end: the truth, the tokenizer, the planner, the generator.

Every number this stack reports about the token planner comes out of the training loop that
produced it, so a harness defect and a model defect look the same. This tool takes single
recorded windows -- the exact objects stage A and stage B were fitted on -- and makes each
stage separately inspectable: what the robot was asked to do there, what the tokenizer does to
that motion, what the planner predicts under each way of querying it, what the flow generator
predicts, and how far each lands from the truth. In physical units, drawn in 3-D, and with
every artifact's provenance and mutual consistency checked rather than assumed.

**Two axes, and conflating them is the trap this exists to avoid.**

``history`` -- what fills the model's ``N`` conditioning rows. Training uses the recorded
APPLIED reference, and since the recording was made by the teacher tracking clips, those rows
ARE the clip: the repo calls that arm ``history_from="clip"`` and prints it as TEACHER
FORCING. Isaac instead serves the planner its own previous plan
(``FmTrackingCommandCfg.history_from="generator"``, the default).

``tokens`` -- how the ``N`` ids are drawn. ``tf`` scores each position against the TRUE
prefix (the cross-entropy loss); ``ar_greedy`` conditions each on its own previous draws.

    what                                          history      tokens
    tokenizer gating_report                       --           true ids
    planner loss / ce / top1 / entropy            clip         tf
    planner epoch line rmse/*                     clip         ar_greedy
    play.py / Isaac / MuJoCo / deploy             GENERATOR    ar_greedy

So the token axis is ALREADY autoregressive in the reported trajectory RMSE. **The history is
the only axis on which the training curve and the deployed loop differ** -- plus two things no
recording can show, since at deploy the depth image and the plan anchor come from a robot that
has already drifted. Every arm here is ``history=clip``; the drift arm is a deliberate second
pass, and it is a verbatim slice when it arrives: measured on the shipped recording, window
``r+1``'s four history rows ARE window ``r``'s target nodes 8..11 in the same anchor frame,
both to ``max|d| = 0.000e+00``.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/inspect_datapoints.py \\
        --data logs/rsl_rl/g1/<run>/recorded_data/<recording>.h5 \\
        --planner logs/token_planner/<run>/planner_e50.pt \\
        --fm logs/flow_matching/<run>/best.pt \\
        --tokenizer logs/token_planner/<run>/tokenizer.pt \\
        --num_points 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from motion_gen.features import _REPO_ROOT, DEFAULT_MJCF, FOOT_FRAMES
from motion_gen.fk import FKChain

from robot_rl.flow_matching.config import FlowMatchingConfig
from robot_rl.flow_matching.dataset import FutureStateDataset
from robot_rl.flow_matching.flow import sample_trajectory
from robot_rl.flow_matching.normalizer import Normalizer
from robot_rl.flow_matching.window import (
    project_window_rotations,
    window_block_slices,
    window_to_world,
)
from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig
from robot_rl.token_planner.data import _recorded_data_io
from robot_rl.token_planner.metrics import (
    GROUP_UNITS,
    ReconAccumulator,
    channel_group_labels,
)
from robot_rl.token_planner.planner import TokenPlanner
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer

#: Groups printed in the tables, in this order. Anything else the layout carries is still
#: accumulated; these are the five the tokenizer's gating report and the FM's val metrics
#: both name, so a number here is directly comparable with a number there.
REPORT_GROUPS: tuple[str, ...] = (
    "base_pos", "base_ori", "base_lin_vel", "base_ang_vel", "joint_pos",
)

#: How many of the ``N`` tokens each ``force*`` arm pins to the target. ``num_tokens`` is
#: appended at runtime as the harness CONTROL -- forcing every token must reproduce the
#: tokenizer's own reconstruction, and if it does not, this script is wrong, not the model.
FORCE_PREFIXES: tuple[int, ...] = (1, 4)

#: Draws for the sampled arm. Fixed and small: four is enough to see whether the plan wanders
#: across modes, and every extra one is another robot in the scene.
NUM_SAMPLES: int = 4


def _load_trainer():
    """Import ``train_planner`` as a module, to reuse its config loader verbatim.

    The overrides it applies to a saved ``fm_config`` are not cosmetic -- ``pinned_axis_
    loss_weight`` at its trained 20.0 makes ``FutureStateDataset`` demand a clip-library join
    and RAISE when it fails -- so the dataset here has to be built the way stage B built it.

    Returns:
        The imported module.
    """
    path = _REPO_ROOT / "scripts" / "token_planner" / "train_planner.py"
    spec = importlib.util.spec_from_file_location("train_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_planner"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    """CLI.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="Comma-separated windowed .h5 recordings, the SAME set and the SAME "
                        "ORDER the planner trained on -- episode ids are offset per file, so "
                        "a swap renumbers every episode and the held-out split moves. Only "
                        "the selected rows are read, so file size does not matter.")
    p.add_argument("--verify_sha", action="store_true",
                   help="Hash each recording and check it against what the planner recorded. "
                        "Off by default: it reads every byte, which is ~30 s for a local "
                        "14.5 GB file and minutes over a network mount, and the "
                        "trained-on check below matches by name regardless.")
    p.add_argument("--planner", required=True, action="append", default=None,
                   help="Planner checkpoint (.pt). Repeatable, to inspect two side by side.")
    p.add_argument("--fm", default=None,
                   help="Flow-matching checkpoint, for the baseline arm. Optional.")
    p.add_argument("--extra_tokenizer", default=None, action="append",
                   help="Comma-separated standalone tokenizer.pt files to score as EXTRA "
                        "reconstruction floors on the same windows. Repeatable. A tokenizer "
                        "whose chunk_len is shorter than the plan is scored over its own "
                        "horizon, alongside the planner's own tokenizer restricted to the "
                        "same nodes -- a shorter chunk is an easier problem, so the two "
                        "gating reports are NOT comparable as written.")
    p.add_argument("--tokenizer", default=None,
                   help="A standalone tokenizer.pt. NOT loaded as the tokenizer -- the one "
                        "used is always the planner's own embedded copy. This is COMPARED "
                        "against it, tensor by tensor, and reported.")
    p.add_argument("--num_points", type=int, default=50,
                   help="Datapoints to select, stratified over family x split.")
    p.add_argument("--pinned", action="store_true",
                   help="Split the base-position error into the axes the terrain PINS and "
                        "the ones it frees, per node. Needs the clip library named in the "
                        "recording (a join by clip_id), so it is opt-in.")
    p.add_argument("--no_stratify", action="store_true",
                   help="Sample the split's NATURAL family mix instead of equally per "
                        "family. Use it to compare against a checkpoint's own recorded "
                        "metrics: those were averaged over the natural mix, and a stratified "
                        "sample is a harder set (unknown is 38%% of the held-out windows).")
    p.add_argument("--split", default="val", choices=("val", "train", "both"),
                   help="Which side of the run's OWN recorded episode split to draw from.")
    p.add_argument("--families", default="",
                   help="Comma-separated family names to restrict to (default: all kept).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature for the sampled arms.")
    p.add_argument("--depth_lag_steps", type=int, default=-1,
                   help="Pin the camera-latency draw to exactly this many control steps. "
                        "The lag is otherwise drawn from the run's depth_latency_probs over "
                        "the SELECTED rows, which is not the draw training made for them.")
    p.add_argument("--batch", type=int, default=64,
                   help="Datapoints per forward pass. The teacher-forced logits are "
                        "(B, N, 12800), so this is the knob that decides whether the run "
                        "fits beside whatever else is on the GPU.")
    p.add_argument("--seed", type=int, default=0, help="Row selection and sampling seed.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--raw_weights", action="store_true",
                   help="Use `model` rather than the preferred `model_ema`.")
    p.add_argument("--urdf", default=None,
                   help="Override the robot URDF drawn (default: the perceptive G1).")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--scene_smoke", action="store_true",
                   help="Build the scene, drive every dropdown and toggle once, and exit. "
                        "The interactive callbacks are where a per-datapoint shape mismatch "
                        "would surface, and they never run under --no_serve.")
    p.add_argument("--no_serve", action="store_true",
                   help="Print the tables and exit without starting viser.")
    p.add_argument("--allow_mismatch", action="store_true",
                   help="Downgrade every fatal consistency failure to a warning. The "
                        "numbers are then not comparable across arms; say so if you quote "
                        "them.")
    p.add_argument("--out", default=None, help="Write the full report as JSON here.")
    p.add_argument("--plot_dir", default=None,
                   help="Write one channel-by-channel figure per plotted datapoint here.")
    p.add_argument("--plot_points", default="",
                   help="Comma-separated datapoint indices to plot (default: the 3 on which "
                        "the greedy arm is WORST, since an easy window has every arm "
                        "decoding the same tokens and shows nothing).")
    p.add_argument("--plot_arms", default="truth,tok_floor",
                   help="Comma-separated arm names, or suffixes matched against the "
                        "planner-tagged names (e.g. 'truth,tok_floor,ar_greedy,fm_zero').")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@dataclass
class Planner:
    """One loaded planner checkpoint and everything the report says about it."""

    tag: str
    path: str
    model: TokenPlanner
    cfg: PlannerConfig
    tok_cfg: TokenizerConfig
    ckpt: dict[str, Any]
    weights: str


def load_planner(path: str, tag: str, device: torch.device, raw: bool) -> Planner:
    """Load a planner and its embedded, frozen tokenizer.

    The tokenizer is deliberately NOT read from a separate file: ``TokenPlanner`` holds it as
    a submodule and ``WeightEMA(freeze=("tokenizer.",))`` leaves it unaveraged, so the
    checkpoint's own weights are by construction the ones the planner was fitted against. A
    standalone ``tokenizer.pt`` is compared against this, never substituted for it.

    Args:
        path: Checkpoint path.
        tag: Short name used in tables and arm names.
        device: Where to put the model.
        raw: Take ``model`` rather than ``model_ema``.

    Returns:
        The loaded :class:`Planner`.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PlannerConfig.from_dict(ck["config"])
    tok_cfg = TokenizerConfig.from_dict(cfg.tokenizer)
    model = TokenPlanner(cfg, TrajectoryTokenizer(tok_cfg))
    have_ema = ck.get("model_ema") is not None
    state = ck["model"] if (raw or not have_ema) else ck["model_ema"]
    model.load_state_dict(state)
    model = model.to(device).eval().requires_grad_(False)
    return Planner(tag=tag, path=path, model=model, cfg=cfg, tok_cfg=tok_cfg, ckpt=ck,
                   weights="raw" if (raw or not have_ema) else "ema")


def load_fm(path: str, device: torch.device, raw: bool):
    """Load a flow-matching generator for the baseline arm.

    Args:
        path: Checkpoint path.
        device: Where to put the model.
        raw: Take ``model`` rather than ``model_ema``.

    Returns:
        ``(model, cfg, ckpt, weights)``.
    """
    from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: PLC0415

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ck["config"])
    model = FlowMatchingTransformer.from_config(cfg)
    have_ema = ck.get("model_ema") is not None
    model.load_state_dict(ck["model"] if (raw or not have_ema) else ck["model_ema"])
    model = model.to(device).eval().requires_grad_(False)
    return model, cfg, ck, ("raw" if (raw or not have_ema) else "ema")


# ---------------------------------------------------------------------------
# Selecting datapoints
# ---------------------------------------------------------------------------


@dataclass
class Ref:
    """One selected datapoint, identified by where it lives rather than by a split index."""

    row: int                 # RAW row in the recording. Never a `window_sel` index.
    episode: int
    clip: int
    clip_step: int
    env: int
    family: str
    held_out: bool
    kept: bool               # would the run's own filters have kept this row?
    depth: int = 0           # round-robin depth it was drawn at; see `select_rows`

    @property
    def label(self) -> str:
        """Dropdown / table label."""
        split = "HELDOUT" if self.held_out else "train  "
        return (f"{split}  {self.family:<22} ep={self.episode} clip={self.clip} "
                f"step={self.clip_step} row={self.row}")


def select_rows(index: dict[str, np.ndarray], family_names: list[str], val_eps: set[int],
                skip_families: list[str], skip_distractors: bool, num_points: int,
                split: str, families: list[str], seed: int,
                stratify: bool = True) -> list[Ref]:
    """Pick rows, stratified over family x split, deterministically.

    Two things here are deliberate. The identity is the RAW recording row: ``window_sel``
    indexes the array AFTER the distractor and family filters, so joining any per-window
    label through it is off by the dropped count -- the trap ``token_planner/data.py``
    documents and re-asserts. And split membership is read off the run's own recorded
    ``val_episode_ids`` rather than re-derived from a seed, so this works against whichever
    subset of the training recordings is on the local box.

    Args:
        index: The cheap per-window arrays (``WINDOW_INDEX_KEYS``).
        family_names: ``meta.family_names``.
        val_eps: The run's held-out episode ids.
        skip_families: Families the run dropped.
        skip_distractors: Whether the run dropped distractor windows.
        num_points: How many to select.
        split: ``"val"``, ``"train"`` or ``"both"``.
        families: Restrict to these family names (empty = all).
        seed: Selection seed.
        stratify: Draw equally from every (family, split) bucket. Off draws the split's
            NATURAL mix, which is what a training curve was averaged over -- ``unknown`` is
            38% of the held-out windows here, so a stratified sample is a harder set than
            the one a checkpoint's own metrics describe and the two are not comparable.

    Returns:
        The selected refs, ordered for a stable dropdown.

    Raises:
        SystemExit: If nothing is eligible.
    """
    ep = np.asarray(index["episode_id"]).astype(np.int64)
    fam = np.asarray(index["clip_family"]).astype(np.int64)
    dis = np.asarray(index.get("is_distractor", np.zeros_like(fam))).astype(np.int64)
    n = ep.shape[0]
    dropped = {i for i, nm in enumerate(family_names) if nm in set(skip_families)}
    kept = ~np.isin(fam, list(dropped)) if dropped else np.ones(n, dtype=bool)
    if skip_distractors:
        kept &= dis == 0
    held = np.isin(ep, list(val_eps)) if val_eps else np.zeros(n, dtype=bool)

    want = kept.copy()
    if split == "val":
        want &= held
    elif split == "train":
        want &= ~held
    if families:
        codes = [family_names.index(f) for f in families if f in family_names]
        unknown = [f for f in families if f not in family_names]
        if unknown:
            raise SystemExit(f"unknown families {unknown}; have {family_names}")
        want &= np.isin(fam, codes)
    if not want.any():
        raise SystemExit(
            f"no eligible windows: split={split!r} families={families or 'all'} over "
            f"{n} rows ({int(kept.sum())} survive the run's filters, "
            f"{int((kept & held).sum())} of those held out).")

    buckets: dict[tuple[str, bool], np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for row in np.flatnonzero(want):
        key = ((family_names[int(fam[row])], bool(held[row])) if stratify
               else ("all", True))
        buckets.setdefault(key, []).append(int(row))
    order = {k: rng.permutation(np.asarray(v)) for k, v in sorted(buckets.items())}

    # Round-robin over the buckets, recording the DEPTH each row was taken at. Sorting on it
    # below is what makes an index stable: bucket order and the within-bucket permutation are
    # both keyed on the seed alone, so growing --num_points appends rows rather than
    # renumbering the ones already there, and "#07" means the same window between runs.
    picked: list[tuple[int, int]] = []
    depth = 0
    while len(picked) < num_points and any(len(v) > depth for v in order.values()):
        for key in sorted(order):
            if len(picked) >= num_points:
                break
            if depth < len(order[key]):
                picked.append((int(order[key][depth]), depth))
        depth += 1

    clip_id = np.asarray(index["clip_id"]).astype(np.int64)
    clip_step = np.asarray(index["clip_step"]).astype(np.int64)
    env_id = np.asarray(index["env_id"]).astype(np.int64)
    refs = [Ref(row=r, episode=int(ep[r]), clip=int(clip_id[r]), clip_step=int(clip_step[r]),
                env=int(env_id[r]), family=family_names[int(fam[r])],
                held_out=bool(held[r]), kept=bool(kept[r]), depth=d) for r, d in picked]
    # NOT sorted. The pick order already is (round, then bucket), so the list cycles one
    # window per family per round -- the first few are one of each rather than a run of
    # whichever family sorts first -- and, because it is a prefix, the numbering is STABLE:
    # raising --num_points appends windows instead of renumbering the ones already there, so
    # "#07" keeps meaning the same window. Any re-sort here breaks that, including sorting by
    # family, which would grow the first family's block and shift everything after it.
    return refs


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One row of the preflight table."""

    name: str
    ok: bool
    detail: str
    fatal: bool = True


def _cmp(checks: list[Check], name: str, a: Any, b: Any, a_lbl: str, b_lbl: str,
         fatal: bool = True) -> None:
    """Append an equality check between two config values.

    Dicts and long lists are SUMMARISED rather than dumped: a passing row that prints a
    thirty-field config is a row nobody reads, which is how a failing row next to it gets
    missed.
    """
    ok = a == b
    if ok:
        detail = (f"{len(a)} entries agree" if isinstance(a, (dict, list, tuple))
                  and len(a) > 4 else f"{a_lbl}={a!r}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) + len(b) > 8:
        sa, sb = set(map(str, a)), set(map(str, b))
        detail = (f"{a_lbl} has {len(a)}, {b_lbl} has {len(b)}; "
                  f"{len(sa - sb)} only in {a_lbl}, {len(sb - sa)} only in {b_lbl}")
    elif isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        diff = [k for k in keys if a.get(k) != b.get(k)]
        detail = "; ".join(f"{k}: {a_lbl}={a.get(k)!r} {b_lbl}={b.get(k)!r}"
                           for k in diff[:6]) + (f" (+{len(diff) - 6} more)"
                                                 if len(diff) > 6 else "")
    else:
        detail = f"{a_lbl}={a!r}  vs  {b_lbl}={b!r}"
    checks.append(Check(name, ok, detail, fatal))


def _compare_normalizers(a_state: dict, b_state: dict) -> tuple[bool, str]:
    """Same comparison ``train_planner.assert_normalizers_agree`` makes, returning instead.

    It returns rather than exiting so the whole preflight table prints before anything is
    fatal -- one mismatch usually explains several rows, and seeing them together is the
    point of the table.

    Args:
        a_state: A ``Normalizer.state_dict()``.
        b_state: The other one.

    Returns:
        ``(ok, detail)``; the detail names the worst channel on a mismatch.
    """
    a, b = Normalizer.from_state_dict(a_state), Normalizer.from_state_dict(b_state)
    for field_name in ("mean", "std", "standardize_mask"):
        x, y = getattr(a, field_name).float().cpu(), getattr(b, field_name).float().cpu()
        if x.shape != y.shape:
            return False, f"'{field_name}' is {tuple(x.shape)} vs {tuple(y.shape)}"
        if not torch.allclose(x, y, atol=1e-6):
            i = int((x - y).abs().argmax())
            return False, (f"'{field_name}' differs most at channel {i} "
                           f"({float(x[i]):.6g} vs {float(y[i]):.6g})")
    return True, f"{a.std.numel()} channels agree"


def _state_dicts_equal(a: dict, b: dict) -> tuple[bool, str]:
    """Exact per-tensor comparison of two state dicts."""
    ka, kb = set(a), set(b)
    if ka != kb:
        return False, f"{len(ka - kb)} keys only in the first, {len(kb - ka)} only in the second"
    worst, name = 0.0, ""
    for k in sorted(ka):
        x, y = a[k].detach().float().cpu(), b[k].detach().float().cpu()
        if x.shape != y.shape:
            return False, f"'{k}' is {tuple(x.shape)} vs {tuple(y.shape)}"
        d = float((x - y).abs().max()) if x.numel() else 0.0
        if d > worst:
            worst, name = d, k
    return worst == 0.0, (f"{len(ka)} tensors identical" if worst == 0.0
                          else f"differs, worst {worst:.3e} at '{name}'")


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


@dataclass
class Arm:
    """One trajectory per datapoint, plus the ids that produced it."""

    name: str
    kind: str                              # truth | tokenizer | planner | fm
    nodes: torch.Tensor                    # (M, H, D) NORMALIZED, as the trainer scores them
    ids: torch.Tensor | None = None        # (M, N) token ids, where the arm has any
    hidden: bool = False                   # controls are computed but not drawn by default
    note: str = ""


@torch.no_grad()
def run_arms(planners: list[Planner], fm, fm_cfg, cond: dict[str, torch.Tensor],
             temperature: float, seed: int, device: torch.device, batch: int = 64,
             ) -> tuple[list[Arm], dict[str, dict[str, torch.Tensor]]]:
    """Every arm, over all the datapoints, in chunks.

    Chunked because the teacher-forced logits are ``(B, N, V)`` at ``V = 12800`` -- 41 MB per
    50 windows, before the decode buffers -- and a diagnostic that only runs on an otherwise
    idle GPU is a diagnostic nobody runs. The chunk boundary changes nothing: every arm is a
    function of its own window's conditioning, and the sampled arms are keyed on ``seed + k``
    rather than on a stream, so a draw does not depend on how the batch was cut.

    Args:
        planners: The loaded planners.
        fm: A flow-matching model, or ``None``.
        fm_cfg: That model's config.
        cond: ``state`` / ``history`` / ``extero`` / ``cmd_tokens``, on the CPU.
        temperature: Sampling temperature.
        seed: Base seed; draw ``k`` uses ``seed + k``.
        device: Where to run.
        batch: Windows per chunk.

    Returns:
        ``(arms, token_stats)`` -- ``token_stats[tag]`` carries the teacher-forced per-position
        cross-entropy and predictive entropy for that planner.
    """
    total = cond["state"].shape[0]
    parts: list[list[Arm]] = []
    stat_parts: list[dict[str, dict[str, torch.Tensor]]] = []
    n_arms = 2 + len(planners) * (3 + NUM_SAMPLES + len(FORCE_PREFIXES)) + (fm is not None)
    # Progress, because this loop is the long silent one: every arm is an autoregressive
    # decode over N positions and there is nothing else to print for minutes.
    print(f"[inspect] running {n_arms} arms over {total} windows "
          f"({len(planners)} planner(s){', 1 generator' if fm is not None else ''}, "
          f"batch {batch}) -- this is the slow part.", flush=True)
    t0 = time.perf_counter()
    for lo in range(0, total, max(batch, 1)):
        sl = slice(lo, min(lo + max(batch, 1), total))
        chunk = {k: v[sl].to(device, non_blocking=True) for k, v in cond.items()}
        a, st = _run_arms_chunk(planners, fm, fm_cfg, chunk, temperature, seed, device)
        parts.append([Arm(x.name, x.kind, x.nodes.cpu(),
                          None if x.ids is None else x.ids.cpu(), x.hidden, x.note)
                      for x in a])
        stat_parts.append(st)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        done = sl.stop
        rate = done / max(time.perf_counter() - t0, 1e-9)
        print(f"\r[inspect]   {done}/{total} windows  "
              f"({rate:.1f}/s, ~{max(total - done, 0) / max(rate, 1e-9):.0f} s left)",
              end="", flush=True)
    print(f"\r[inspect]   {total}/{total} windows in "
          f"{time.perf_counter() - t0:.1f} s.".ljust(72), flush=True)
    arms = [Arm(a.name, a.kind, torch.cat([p[i].nodes for p in parts], dim=0),
                None if a.ids is None else torch.cat([p[i].ids for p in parts], dim=0),
                a.hidden, a.note)
            for i, a in enumerate(parts[0])]
    stats = {tag: {k: torch.cat([sp[tag][k] for sp in stat_parts], dim=0)
                   for k in stat_parts[0][tag]} for tag in stat_parts[0]}
    return arms, stats


@torch.no_grad()
def _run_arms_chunk(planners: list[Planner], fm, fm_cfg, cond: dict[str, torch.Tensor],
                    temperature: float, seed: int, device: torch.device,
                    ) -> tuple[list[Arm], dict[str, dict[str, torch.Tensor]]]:
    """One chunk's worth of :func:`run_arms`. See it for the arguments."""
    state, hist = cond["state"], cond["history"]
    extero, cmd = cond["extero"], cond.get("cmd_tokens")

    tok = planners[0].model.tokenizer
    tgt_ids = tok.encode_to_ids(state)
    arms = [
        Arm("truth", "truth", state, tgt_ids,
            note="the recorded ego-anchored clip future"),
        Arm("tok_floor", "tokenizer", tok.decode_ids(tgt_ids), tgt_ids,
            note="decode(encode(truth)) -- no conditioning; the quantization floor"),
    ]

    token_stats: dict[str, dict[str, torch.Tensor]] = {}
    for pl in planners:
        m, tag, n_tok = pl.model, pl.tag, pl.model.num_tokens
        logits = m(tgt_ids, hist, extero, cmd)                    # teacher-forced
        cache = m._last_prefix_cache                              # noqa: SLF001
        logp = torch.log_softmax(logits.float(), dim=-1)
        token_stats[tag] = {
            "ce": -logp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1).cpu(),   # (M, N)
            "entropy": (-(logp.exp() * logp).sum(-1)).cpu(),                   # (M, N)
        }
        tf_ids = logits.argmax(-1)
        arms.append(Arm(f"{tag}/tf", "planner", m.decode(tf_ids), tf_ids,
                        note="argmax given the TRUE prefix -- what the CE measures"))

        greedy, _, _ = m.sample(hist, extero, cmd, greedy=True, prefix_cache=cache)
        arms.append(Arm(f"{tag}/ar_greedy", "planner", m.decode(greedy), greedy,
                        note="autoregressive argmax -- what Isaac serves"))

        for k in range(NUM_SAMPLES):
            gen = torch.Generator(device=device.type).manual_seed(seed + k)
            ids, _, _ = m.sample(hist, extero, cmd, temperature=temperature, greedy=False,
                                 generator=gen, prefix_cache=cache)
            arms.append(Arm(f"{tag}/ar_sample_{k}", "planner", m.decode(ids), ids,
                            hidden=(k > 0), note=f"categorical draw at T={temperature}"))

        # The last entry is the CONTROL: forcing every token must reproduce `tok_floor`.
        for k in (*FORCE_PREFIXES, n_tok):
            force = torch.full_like(tgt_ids, -1)
            force[:, :k] = tgt_ids[:, :k]
            ids, _, _ = m.sample(hist, extero, cmd, greedy=True, prefix_cache=cache,
                                 force_tokens=force)
            arms.append(Arm(f"{tag}/force{k}", "planner", m.decode(ids), ids,
                            hidden=(k == n_tok),
                            note=(f"first {k} tokens pinned to the target"
                                  + (" -- CONTROL, must equal tok_floor" if k == n_tok
                                     else ""))))

    if fm is not None:
        nodes = sample_trajectory(fm, hist, extero, cmd, int(fm_cfg.future_len),
                                  int(fm_cfg.state_dim), int(fm_cfg.num_sampling_steps),
                                  zero_noise=True)
        arms.append(Arm("fm_zero", "fm", nodes, None,
                        note="the generator at x0 = 0 -- the arm the controller runs"))
    return arms, token_stats


# ---------------------------------------------------------------------------
# Physical space: world lift, feet, errors
# ---------------------------------------------------------------------------


def world_of(nodes_norm: torch.Tensor, norm: Normalizer, p_ref: torch.Tensor,
             q_ref: torch.Tensor, nj: int, nee: int, blocks: list[str],
             fk: FKChain) -> dict[str, np.ndarray]:
    """Lift normalized window nodes into the world, and FK the feet.

    ``window_to_world`` is the exact inverse of the packing the dataset applied, and it lifts
    through the anchor's YAW ONLY -- feeding it the full anchor rotation would tilt the whole
    ribbon by the robot's roll and pitch at the re-plan. The rotation projection happens first,
    at the same point ``TokenPlanner.decode`` and ``online.sample_window`` put it: six free
    numbers out of a decoder are only approximately a rotation, and a quaternion has nowhere
    to put the residual.

    Args:
        nodes_norm: ``(M, H, D)`` normalized nodes.
        norm: The state normalizer.
        p_ref: ``(M, 3)`` anchor position (the snap row's measured root).
        q_ref: ``(M, 4)`` anchor orientation, xyzw.
        nj: Joint count.
        nee: End-effector frame count in the layout.
        blocks: The node's block selection.
        fk: A chain over :data:`~motion_gen.features.FOOT_FRAMES`, rooted at the pelvis.

    Returns:
        ``base_pos`` ``(M, H, 3)``, ``base_quat`` ``(M, H, 4)`` xyzw, ``base_lin_vel`` /
        ``base_ang_vel`` ``(M, H, 3)`` (world frame), ``joint_pos`` ``(M, H, nj)`` and
        ``feet`` ``(M, H, 2, 3)``, all numpy.
    """
    from isaaclab.utils.math import quat_apply  # noqa: PLC0415

    phys = project_window_rotations(norm.denormalize(nodes_norm.cpu()), nj, nee, blocks)
    w = window_to_world(phys, p_ref[:, None, :].cpu(), q_ref[:, None, :].cpu(),
                        nj, nee, blocks)
    m, h = phys.shape[0], phys.shape[1]
    q = w["joint_pos"].reshape(m * h, nj).float()
    rel, _, _, _ = fk.pose_and_jacobian(q)                       # (m*h, K, 3) pelvis frame
    k = rel.shape[1]
    qb = w["base_quat_w"].reshape(m * h, 1, 4).expand(m * h, k, 4).reshape(-1, 4)
    feet = (w["base_pos_w"].reshape(m * h, 1, 3)
            + quat_apply(qb, rel.reshape(-1, 3)).reshape(m * h, k, 3)).reshape(m, h, k, 3)
    out = {
        "base_pos": w["base_pos_w"].numpy().astype(np.float32),
        "base_quat": w["base_quat_w"].numpy().astype(np.float32),
        "joint_pos": w["joint_pos"].numpy().astype(np.float32),
        "feet": feet.numpy().astype(np.float32),
    }
    for key, name in (("base_lin_vel_w", "base_lin_vel"), ("base_ang_vel_w", "base_ang_vel")):
        if key in w:
            out[name] = w[key].numpy().astype(np.float32)
    return out


def group_rmse_per_item(target: torch.Tensor, pred: torch.Tensor, labels: torch.Tensor,
                        std: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Physical per-group RMSE, one row per datapoint.

    The same reduction :meth:`ReconAccumulator.group_rmse` performs -- ``err * std``, squared,
    summed over a group's channels and its nodes, square-rooted -- but kept per item so a
    single window can be reported. ``main`` asserts that pooling these reproduces the
    accumulator's own number, so the two cannot drift apart.

    Args:
        target: ``(M, H, D)`` normalized truth.
        pred: ``(M, H, D)`` normalized prediction.
        labels: ``(D,)`` group index per channel.
        std: ``(D,)`` normalizer std.
        n_groups: Number of groups ``G``.

    Returns:
        ``(M, G)`` float64 RMSE in each group's own physical unit.
    """
    err = (pred.cpu() - target.cpu()).double() * std.cpu().double()
    onehot = torch.zeros(n_groups, target.shape[-1], dtype=torch.float64)
    onehot[labels.cpu(), torch.arange(target.shape[-1])] = 1.0
    sq_g = err.pow(2) @ onehot.T                                  # (M, H, G)
    cnt = onehot.sum(dim=1)                                       # (G,)
    return (sq_g.sum(dim=1) / (cnt[None, :] * target.shape[1])).sqrt()


def ori_geodesic_per_item(target: torch.Tensor, pred: torch.Tensor, labels: torch.Tensor,
                          std: torch.Tensor, mean: torch.Tensor, ori_group: int
                          ) -> torch.Tensor:
    """Mean base-orientation angle error per datapoint, in radians.

    Literally the function the tokenizer's report and the flow trainer's ``channel_metrics``
    both call, so the number is comparable with both by construction rather than by comment.

    Args:
        target / pred: ``(M, H, D)`` normalized.
        labels: ``(D,)`` group index per channel.
        std / mean: ``(D,)`` normalizer statistics.
        ori_group: Index of ``base_ori`` in the group list.

    Returns:
        ``(M,)`` mean geodesic angle.
    """
    from robot_rl.flow_matching.transforms import geodesic_angle_rot6d  # noqa: PLC0415

    col = labels.cpu() == ori_group
    sd, mn = std.cpu(), mean.cpu()
    a = (pred.cpu() * sd + mn)[..., col]
    b = (target.cpu() * sd + mn)[..., col]
    return geodesic_angle_rot6d(a, b).mean(dim=1)


def cloud_of(extero_row: torch.Tensor, n_sensors: int, channels: int, sensor_dim: int,
             pose_pos: torch.Tensor, pose_quat: torch.Tensor
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The depth row's xyz channels, lifted into the world. This is what the model SAW.

    ``ds.extero`` holds the row BEFORE the normalizer, so its ``x/y/z`` planes are already
    metres in the root's yaw-aligned LEVEL frame -- no round trip through the contract
    affine is needed or wanted. Lifting is then exact and odometry-free: writing
    ``R_world_root = Rz(psi) @ R_rp``, the yaw frame is what is left after ``psi`` is divided
    out, so ``p_world = root_pos + Rz(psi) @ p_yaw``.

    The pose is the frame's CAPTURE pose, not the snap pose. ``depth_latency_probs`` swaps an
    older frame in for most windows and unprojects it with the state it was captured with;
    using the snap pose instead would displace the cloud by exactly the egomotion those xyz
    channels exist to remove.

    Args:
        extero_row: ``(n_sensors * channels * sensor_dim,)`` pre-normalizer row.
        n_sensors: Cameras.
        channels: Channels per pixel; ``>= 4`` for xyz, ``>= 5`` for validity.
        sensor_dim: Pixels per camera.
        pose_pos: ``(3,)`` capture-time root position, world.
        pose_quat: ``(4,)`` capture-time root orientation, world, xyzw.

    Returns:
        ``(points (P, 3), valid (P,), camera (P,))``, all numpy; empty when the row carries
        depth alone.
    """
    from isaaclab.utils.math import quat_apply, yaw_quat  # noqa: PLC0415

    if channels < 4:
        return (np.zeros((0, 3), np.float32), np.zeros((0,), bool), np.zeros((0,), np.int64))
    row = extero_row.cpu().reshape(n_sensors, channels, sensor_dim)
    pts = row[:, 1:4, :].permute(0, 2, 1).reshape(-1, 3)                  # (T*K, 3) metres
    valid = (row[:, 4, :].reshape(-1) > 0.5 if channels >= 5
             else torch.ones(pts.shape[0], dtype=torch.bool))
    cam = torch.arange(n_sensors).repeat_interleave(sensor_dim)
    yaw = yaw_quat(pose_quat.cpu().reshape(1, 4)).expand(pts.shape[0], 4)
    world = pose_pos.cpu().reshape(1, 3) + quat_apply(yaw, pts)
    return (world.numpy().astype(np.float32), valid.numpy().astype(bool),
            cam.numpy().astype(np.int64))


def pinned_free_rms(target: torch.Tensor, pred: torch.Tensor, norm: Normalizer,
                    anchor_mask: torch.Tensor, frame_yaw_rel: torch.Tensor | None,
                    nj: int, nee: int, blocks: list[str]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Base-position error split into the axes the terrain PINS and the ones it frees, in metres.

    The distinction the whole terrain-registration story turns on. A reference that is off
    LATERALLY is re-anchored onto the robot anyway, so the error costs nothing; one that is
    off along a PINNED axis has slid the stair segment off the terrain, which is what a fall
    looks like. A single ``base_pos`` RMSE averages the two and hides it -- the same reason
    the flow trainer reports ``pinned_axis_split``.

    Which axes are pinned is per NODE, not per window: the mask is the clip's own
    ``anchor_mask_t``, so a window crossing a transition is partly free and partly pinned.
    Flat ground pins height only; stairs and boxes pin downrange, height and yaw.

    Reuses :func:`~robot_rl.flow_matching.flow._planar_terrain_components` for the rotation
    into the terrain's frame -- "downrange is pinned, lateral is free" is a statement about
    the TERRAIN's axes while the window's x/y are the snap robot's, so the projection is not
    optional. It is fed the PHYSICAL error with a unit std, which is what makes the result
    metres rather than the trainer's normalized units.

    Rotation is deliberately excluded even though the mask covers it: a 6-D rotation error
    cannot be pooled with metres, and mixing them would produce a number in no unit at all.

    Args:
        target / pred: ``(M, H, D)`` normalized truth and prediction.
        norm: The state normalizer, to reach physical units.
        anchor_mask: ``(M, H)`` per-node anchor bitmask; bit SET = that DOF is FREE.
        frame_yaw_rel: ``(M,)`` window-frame-to-terrain-frame angle, or ``None`` if aligned.
        nj / nee / blocks: The node layout.

    Returns:
        ``(pinned_rms, free_rms)``, each ``(M,)`` in metres. A window with no member on one
        side reports ``nan`` there.
    """
    err = (norm.denormalize(pred.cpu()) - norm.denormalize(target.cpu())).double()
    base = window_block_slices(nj, nee, blocks)["base_pose"].start
    return pinned_free_from_error(err, base, anchor_mask, frame_yaw_rel)


def pinned_free_from_error(err: torch.Tensor, base: int, anchor_mask: torch.Tensor,
                           frame_yaw_rel: torch.Tensor | None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """The pinned/free split of a PHYSICAL base-position error. See :func:`pinned_free_rms`.

    Split out so a caller already working in physical units -- the tokenizer-floor table,
    where each tokenizer has its own normalizer -- does not have to round-trip through one.

    Args:
        err: ``(M, H, D)`` error in physical units.
        base: Column where the ``base_pose`` block starts.
        anchor_mask: ``(M, H)`` per-node anchor bitmask; bit SET = FREE.
        frame_yaw_rel: ``(M,)`` window-to-terrain angle, or ``None``.

    Returns:
        ``(pinned_rms, free_rms)``, each ``(M,)`` in metres.
    """
    from robot_rl.flow_matching.flow import (  # noqa: PLC0415
        _ANCHOR_BIT,
        _planar_terrain_components,
    )

    err = err.double()
    mask = anchor_mask.cpu()[:, :err.shape[1]]

    def pinned(name: str) -> torch.Tensor:
        """1.0 where the terrain pins this DOF at this node."""
        return ((mask >> _ANCHOR_BIT[name]) & 1 == 0).double()

    ang = (torch.zeros(err.shape[:2], dtype=torch.float64) if frame_yaw_rel is None
           else frame_yaw_rel.cpu().double().unsqueeze(-1).expand(err.shape[:2]))
    dr, lat = _planar_terrain_components(err, base, ang, torch.ones(2, dtype=torch.float64))
    z = err[..., base + 2]
    p_dr, p_lat, p_h = pinned("downrange"), pinned("lateral"), pinned("height")

    def rms(w_dr: torch.Tensor, w_lat: torch.Tensor, w_h: torch.Tensor) -> np.ndarray:
        """Per-window RMS over whichever of the three axes the weights select."""
        sq = (w_dr * dr ** 2 + w_lat * lat ** 2 + w_h * z ** 2).sum(dim=1)
        n = (w_dr + w_lat + w_h).sum(dim=1)
        out = torch.full_like(sq, float("nan"))
        ok = n > 0
        out[ok] = (sq[ok] / n[ok]).sqrt()
        return out.numpy()

    return rms(p_dr, p_lat, p_h), rms(1 - p_dr, 1 - p_lat, 1 - p_h)


@torch.no_grad()
def score_extra_tokenizers(paths: list[str], ref_tok: TrajectoryTokenizer, ref_cfg,
                           state: torch.Tensor, norm: Normalizer, labels: torch.Tensor,
                           groups: list[str], device: torch.device,
                           anchor_mask: torch.Tensor | None = None,
                           frame_yaw_rel: torch.Tensor | None = None,
                           nj: int = 29, nee: int = 4,
                           blocks: list[str] | None = None) -> str:
    """Reconstruction floor of one or more standalone tokenizers, on these windows.

    A tokenizer is judged by ``decode(encode(future))`` and nothing else, so this needs no
    planner -- but two things make a naive comparison of gating reports wrong, and both are
    handled here rather than left to the reader:

    * **Horizon.** A tokenizer with a shorter ``chunk_len`` is solving an easier problem:
      reconstruction error grows through a chunk (the shipped ``base_pos`` interior slope is
      2.3), so half the look-ahead is a materially lower floor at the same token budget. Such
      a tokenizer is scored over ITS OWN horizon, and the reference tokenizer is scored over
      the same nodes beside it. That pair is the comparison; the two full-length numbers are
      not.
    * **Normalizer.** A tokenizer decodes into the space ITS OWN normalizer defined, and a
      different ``chunk_len`` necessarily refits it -- the fit runs over every node of every
      window, and ``joint_pos`` takes a fitted std rather than a declared one. So each
      tokenizer is driven through its own normalizer and the comparison is made in PHYSICAL
      units, which is the only space the two share (and the one both gating reports quote).
      A mismatch is reported, not refused.

    Args:
        paths: Tokenizer checkpoints.
        ref_tok: The planner's own tokenizer, as the baseline row.
        ref_cfg: Its config.
        state: ``(M, H, D)`` normalized truth.
        norm: The state normalizer these windows were built with.
        labels / groups: Channel-group layout for the reduction.
        device: Where to run.
        anchor_mask / frame_yaw_rel: The per-node anchor mask and terrain-frame angle, to add
            the PIN/free split -- the column a base-pose-weighted tokenizer exists to move.
            ``None`` omits it.
        nj / nee / blocks: The node layout, for locating the ``base_pose`` block.

    Returns:
        The rendered table.

    Raises:
        SystemExit: On a layout or normalizer mismatch.
    """
    h = int(state.shape[1])
    rows: list[tuple[str, int, dict[str, float]]] = []
    notes: list[str] = []
    truth_phys = norm.denormalize(state)                       # (M, H, D) physical
    ones = torch.ones(int(state.shape[-1]))

    def floor_of(tok: TrajectoryTokenizer, tnorm: Normalizer, native: int,
                 upto: int) -> dict[str, float]:
        """Physical per-group RMSE of ``decode(encode(.))``, scored over the first ``upto``
        nodes.

        Each tokenizer runs at its OWN ``chunk_len``: the encoder is chunk-local and refuses
        any other length, so a 64-node tokenizer cannot be handed 32 nodes. What is made
        common is the range the ERROR is measured over, which is the comparable thing.
        """
        rec = tnorm.denormalize(tok.decode_ids(tok.encode_to_ids(
            tnorm.normalize(truth_phys[:, :native]).to(device))).cpu())
        err = rec[:, :upto] - truth_phys[:, :upto]
        r = group_rmse_per_item(truth_phys[:, :upto], rec[:, :upto], labels, ones,
                                len(groups))
        out = {g: float(np.sqrt((r[:, k].numpy() ** 2).mean()))
               for k, g in enumerate(groups)}
        if anchor_mask is not None:
            base = window_block_slices(nj, nee, blocks)["base_pose"].start
            pin, free = pinned_free_from_error(err, base, anchor_mask, frame_yaw_rel)
            out["_pin"] = float(np.sqrt(np.nanmean(pin ** 2)))
            out["_free"] = float(np.sqrt(np.nanmean(free ** 2)))
        return out

    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TokenizerConfig.from_dict(ck["config"])
        if int(cfg.node_dim) != int(state.shape[-1]):
            raise SystemExit(
                f"{path}: node_dim {cfg.node_dim} but these windows are "
                f"{state.shape[-1]} wide; it was fitted on a different layout.")
        tnorm = Normalizer.from_state_dict(ck["state_normalizer"])
        ok, detail = _compare_normalizers(norm.state_dict(), ck["state_normalizer"])
        if not ok:
            notes.append(f"{os.path.basename(os.path.dirname(path))}: its normalizer is not "
                         f"these windows' ({detail}), so it is driven through its own -- "
                         "the numbers below are physical either way.")
        n = int(cfg.chunk_len)
        if n != int(ref_cfg.chunk_len):
            # Refused rather than approximated. The encoder is chunk-local and rejects any
            # other length, so the two cannot be run over a common range without one of them
            # working outside the horizon it was fitted for -- and a tokenizer covering fewer
            # nodes at the same token budget is solving a strictly easier problem (the
            # shipped base_pos interior slope is 2.3), so their gating reports are not
            # comparable either. Scoring them side by side would put a number on the page
            # that reads as a comparison and is not one.
            raise SystemExit(
                f"{path}: chunk_len {n} against the planner's {ref_cfg.chunk_len}. A shorter "
                f"chunk covers {n * float(cfg.dt):.2f} s instead of "
                f"{int(ref_cfg.chunk_len) * float(ref_cfg.dt):.2f} s at the same token "
                "budget, which is an easier reconstruction, and the chunk-local encoder "
                "cannot be run over the other's horizon to control for it. Compare it "
                "against a planner built on it, not against this one.")
        tok = TrajectoryTokenizer(cfg)
        tok.load_state_dict(ck.get("model_ema") or ck["model"])
        tok = tok.to(device).eval().requires_grad_(False)
        name = os.path.basename(os.path.dirname(os.path.abspath(path))) or path
        rows.append((f"{name}  (chunk {n}, D={cfg.downsample}, V={cfg.vocab_size}, "
                     f"N={n // cfg.downsample})", n, tok, tnorm))

    def _pin_cells(v: dict[str, float]) -> str:
        """The PIN/free pair, or nothing when the mask was not built."""
        return ("" if "_pin" not in v
                else f"{v['_pin']:>10.4f}{v['_free']:>10.4f}")

    ref_n = int(ref_cfg.chunk_len)
    ref_label = (f"(the planner's own: chunk {ref_n}, D={ref_cfg.downsample}, "
                 f"V={ref_cfg.vocab_size}, N={ref_n // ref_cfg.downsample})")
    horizons = sorted({n for _, n, _, _ in rows} | {ref_n})
    cols = [g for g in REPORT_GROUPS if g in groups]
    head = (f"{'tokenizer':<58}{'scored':>8}"
            + "".join(f"{_SHORT.get(g, g) + '[' + GROUP_UNITS.get(g, '?') + ']':>16}"
                      for g in cols)
            + (f"{'PIN[m]':>10}{'free[m]':>10}" if anchor_mask is not None else ""))
    out = [head, "-" * len(head)]
    for upto in horizons:
        out.append(f"-- error measured over nodes 0..{upto - 1} "
                   f"({upto * 0.02:.2f} s) " + "-" * 20)
        if ref_n >= upto:
            v = floor_of(ref_tok, norm, ref_n, upto)
            out.append(f"{ref_label:<58}{upto:>8}"
                       + "".join(f"{v[g]:>16.5f}" for g in cols) + _pin_cells(v))
        for label, n, tok, tnorm in rows:
            if n < upto:
                continue
            v = floor_of(tok, tnorm, n, upto)
            out.append(f"{label:<58}{upto:>8}"
                       + "".join(f"{v[g]:>16.5f}" for g in cols) + _pin_cells(v))
    out.append("")
    out.append("Physical units; each tokenizer encodes at its OWN chunk length and the error "
               "is scored over the")
    out.append("common range. A tokenizer covering fewer nodes at the same token budget is "
               "solving an easier")
    out.append("problem, so read the block it appears in -- not its gating report against "
               "another's.")
    out += ([""] + [f"note: {n}" for n in notes] if notes else [])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

#: The sagittal leg chain, resolved by NAME against the recording's own joint order. That
#: order is Isaac's interleaved one and differs from the URDF's per-limb order in 27 of 29
#: positions, so an index taken from either list by eye is wrong about the other.
_LEG_JOINTS: tuple[tuple[str, str, str], ...] = (
    ("hip pitch", "left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("knee", "left_knee_joint", "right_knee_joint"),
    ("ankle pitch", "left_ankle_pitch_joint", "right_ankle_pitch_joint"),
)


def _to_anchor(points_w: np.ndarray, p_anchor: np.ndarray, q_anchor: np.ndarray,
               rotate_only: bool = False) -> np.ndarray:
    """World positions (or vectors) into the anchor's yaw frame.

    Everything is plotted in that frame rather than in the world, for two reasons: it is the
    frame the model actually predicts in -- so ``x`` reads as downrange and ``y`` as lateral
    rather than as whichever way the tile happens to face -- and it makes two datapoints on
    opposite corners of a 2.6 km layout directly comparable.

    Args:
        points_w: ``(..., 3)`` world positions, or vectors when ``rotate_only``.
        p_anchor: ``(3,)`` anchor position.
        q_anchor: ``(4,)`` anchor orientation, xyzw.
        rotate_only: Skip the translation (for velocities).

    Returns:
        ``(..., 3)`` in the anchor's yaw frame.
    """
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: PLC0415

    v = torch.from_numpy(np.ascontiguousarray(points_w)).float()
    shape = v.shape
    flat = v.reshape(-1, 3)
    if not rotate_only:
        flat = flat - torch.from_numpy(np.asarray(p_anchor, dtype=np.float32)).reshape(1, 3)
    q = yaw_quat(torch.from_numpy(np.asarray(q_anchor, dtype=np.float32)).reshape(1, 4))
    return quat_apply_inverse(q.expand(flat.shape[0], 4), flat).reshape(shape).numpy()


def _rpy_rel(quat_w: np.ndarray, q_anchor: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw of a world orientation relative to the anchor's yaw frame, in degrees.

    Six rot6d numbers are not a quantity anyone can read off a plot; three angles are. Taken
    relative to the anchor's YAW (not its full rotation), matching the window's own
    convention -- so roll and pitch are the reference's true tilt and yaw is the heading
    change since the re-plan.
    """
    from isaaclab.utils.math import (  # noqa: PLC0415
        euler_xyz_from_quat,
        quat_conjugate,
        quat_mul,
        yaw_quat,
    )

    q = torch.from_numpy(np.ascontiguousarray(quat_w)).float().reshape(-1, 4)
    a = yaw_quat(torch.from_numpy(np.asarray(q_anchor, dtype=np.float32)).reshape(1, 4))
    rel = quat_mul(quat_conjugate(a).expand(q.shape[0], 4), q)
    rpy = torch.stack(euler_xyz_from_quat(rel), dim=-1)
    # euler_xyz_from_quat returns [0, 2pi); wrap to (-pi, pi] or every near-zero angle plots
    # as a 360-degree cliff.
    rpy = (rpy + math.pi) % (2.0 * math.pi) - math.pi
    return np.degrees(rpy.numpy())


def _resolve_plot_arms(wanted: list[str], arms: list[Arm]) -> list[Arm]:
    """Match requested arm names against the tagged ones (``planner/ar_greedy``)."""
    out: list[Arm] = []
    names = [a.name for a in arms]
    for w in wanted:
        hit = [a for a in arms if a.name == w or a.name.rpartition("/")[2] == w]
        if not hit:
            raise SystemExit(f"--plot_arms: no arm matches {w!r}; have {names}")
        out.extend(h for h in hit if h not in out)
    return out


def plot_datapoint(idx: int, ref: Ref, plot_arms: list[Arm], world: dict[str, dict],
                   p_anchor: np.ndarray, q_anchor: np.ndarray, joint_names: list[str],
                   dt: float, out_dir: str) -> list[str]:
    """One figure per datapoint: every channel a reader can judge, truth against each arm.

    Seven rows of three, all against look-ahead time, all in the ANCHOR'S YAW FRAME (see
    :func:`_to_anchor`):

    1-4. the root -- position, orientation as roll/pitch/yaw, linear and angular velocity.
    5.   the FK'd ankles, since a root 5 cm off is recoverable and a swing foot 5 cm off
         catches a riser; left solid, right dashed.
    6.   the sagittal leg chain (hip pitch, knee, ankle pitch), which is what PUTS the foot
         there -- so a foot error can be read back to the joint that caused it.
    7.   each foot in the sagittal plane, where a step is one readable loop rather than three
         time series, and the error against look-ahead, which is where the tokenizer's
         measured 2.3x interior slope shows up.

    Args:
        idx: Datapoint index, for the filename.
        ref: Its identity.
        plot_arms: Arms to draw; the first ``truth`` arm is the reference every other is
            measured against.
        world: ``world[arm.name]`` from :func:`world_of`, indexed by datapoint.
        p_anchor / q_anchor: The window's anchor, world frame, xyzw for the quaternion.
        joint_names: The recording's joint order.
        dt: Control period, for the time axis.
        out_dir: Directory to write into.

    Returns:
        The files written.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # `plot_style` lives in scripts/rsl_rl/ and is reached by bare name, the same bootstrap
    # `eval/_paths.py` and `fm_augment_probe.py` use. Guarded so plotting N datapoints does
    # not push N copies onto sys.path.
    _style_dir = str(_REPO_ROOT / "scripts" / "rsl_rl")
    if _style_dir not in sys.path:
        sys.path.insert(0, _style_dir)
    import plot_style  # noqa: PLC0415

    plot_style.apply_paper_style()
    h = world[plot_arms[0].name]["base_pos"][idx].shape[0]
    t = np.arange(h) * float(dt)
    truth = world["truth"]

    # The paper style is tuned for a single panel: linewidth 4 and 20-pt labels are correct
    # there and illegible on a 21-panel grid. Take its palette, serif and usetex; override
    # the sizes. Same call the repo's other grid plotters would need and the reason
    # play_plots borrows nothing from it.
    grid_rc = {
        "lines.linewidth": 1.8, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
        "axes.titlepad": 4, "axes.labelpad": 3,
    }
    with matplotlib.rc_context(grid_rc):
        # NOT sharex: the bottom row is the sagittal foot plane, whose x is DOWNRANGE IN
        # METRES, and sharing a column with it stretches every time axis above it to the
        # foot's travel. The six time rows are tied by hand instead.
        fig, ax = plt.subplots(7, 3, figsize=(13.5, 19.0))

        def style(arm: Arm) -> dict:
            """Colour and weight: truth is black and heavier, everything else is its own.

            The scene colour DARKENED, not replaced: keeping the hue is what lets "the pink
            one" in the 3-D view be found on the page, and the scene's colours are chosen to
            read against its own background rather than against white paper.
            """
            if arm.kind == "truth":
                return {"color": "black", "lw": 2.4, "zorder": 3}
            return {"color": tuple(0.72 * c for c in arm_color(arm.name, [])),
                    "lw": 1.6, "zorder": 2}

        rows = [
            ("root position [m]", ("downrange $x$", "lateral $y$", "height $z$"),
             lambda w, i: _to_anchor(w["base_pos"][i], p_anchor, q_anchor)),
            ("root orientation [deg]", ("roll", "pitch", "yaw"),
             lambda w, i: _rpy_rel(w["base_quat"][i], q_anchor)),
            ("root linear velocity [m/s]", ("$v_x$", "$v_y$", "$v_z$"),
             lambda w, i: _to_anchor(w["base_lin_vel"][i], p_anchor, q_anchor,
                                     rotate_only=True)),
            ("root angular velocity [rad/s]", ("$\\omega_x$", "$\\omega_y$", "$\\omega_z$"),
             lambda w, i: _to_anchor(w["base_ang_vel"][i], p_anchor, q_anchor,
                                     rotate_only=True)),
        ]
        for r, (ylab, comps, fn) in enumerate(rows):
            for c in range(3):
                ax[r][c].set_title(comps[c])
                ax[r][c].set_ylabel(ylab if c == 0 else "")
            for arm in plot_arms:
                v = fn(world[arm.name], idx)
                for c in range(3):
                    ax[r][c].plot(t, v[:, c], label=arm.name, **style(arm))

        # -- row 5: the ankles, left solid / right dashed -------------------------------
        for c, comp in enumerate(("downrange $x$", "lateral $y$", "height $z$")):
            ax[4][c].set_title(f"ankle {comp}")
        ax[4][0].set_ylabel("foot position [m]")
        for arm in plot_arms:
            f = _to_anchor(world[arm.name]["feet"][idx], p_anchor, q_anchor)   # (H, 2, 3)
            for c in range(3):
                for k, ls in ((0, "-"), (1, "--")):
                    ax[4][c].plot(t, f[:, k, c], ls=ls, **style(arm))

        # -- row 6: the sagittal leg chain ----------------------------------------------
        for c, (label, lname, rname) in enumerate(_LEG_JOINTS):
            ax[5][c].set_title(label)
            li, ri = joint_names.index(lname), joint_names.index(rname)
            for arm in plot_arms:
                q = world[arm.name]["joint_pos"][idx]
                ax[5][c].plot(t, q[:, li], ls="-", **style(arm))
                ax[5][c].plot(t, q[:, ri], ls="--", **style(arm))
        ax[5][0].set_ylabel("joint angle [rad]")

        # -- row 7: sagittal foot loops, and the error against look-ahead ---------------
        for c, (k, side) in enumerate(((0, "left"), (1, "right"))):
            ax[6][c].set_title(f"{side} foot, sagittal")
            ax[6][c].set_xlabel("downrange $x$ [m]")
            for arm in plot_arms:
                f = _to_anchor(world[arm.name]["feet"][idx], p_anchor, q_anchor)
                ax[6][c].plot(f[:, k, 0], f[:, k, 2], **style(arm))
            ax[6][c].set_aspect("equal", adjustable="datalim")
        ax[6][0].set_ylabel("height $z$ [m]")
        ax[6][2].set_title("error against the truth")
        ax[6][2].set_xlabel("look-ahead [s]")
        ax[6][2].set_ylabel("distance [m]")
        for arm in plot_arms:
            if arm.kind == "truth":
                continue
            dp = np.linalg.norm(world[arm.name]["base_pos"][idx] - truth["base_pos"][idx],
                                axis=-1)
            df = np.linalg.norm(world[arm.name]["feet"][idx] - truth["feet"][idx],
                                axis=-1).mean(axis=-1)
            ax[6][2].plot(t, dp, ls="-", **style(arm))
            ax[6][2].plot(t, df, ls="--", **style(arm))

        # The six time rows share one axis, set explicitly; only the last of them carries
        # tick labels, and the sagittal row keeps its own.
        for r in range(6):
            for a in ax[r]:
                a.set_xlim(0.0, float(t[-1]))
                a.tick_params(labelbottom=(r == 5))
        for a in ax[5]:
            a.set_xlabel("look-ahead [s]")
        for row in ax:
            for a in row:
                a.grid(True, alpha=0.25, lw=0.6)
        handles = [plt.Line2D([], [], **{k: v for k, v in style(a).items() if k != "zorder"},
                              label=a.name) for a in plot_arms]
        handles += [plt.Line2D([], [], color="0.35", ls="-", lw=1.4, label="left / base"),
                    plt.Line2D([], [], color="0.35", ls="--", lw=1.4, label="right / feet")]
        fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6),
                   frameon=False, bbox_to_anchor=(0.5, 1.005))
        fig.subplots_adjust(top=0.965)

    caption = (
        f"Datapoint #{idx:02d}: {ref.family}, "
        f"{'held out' if ref.held_out else 'in the training split'}, episode {ref.episode}, "
        f"clip {ref.clip} step {ref.clip_step}, recording row {ref.row}. "
        f"{h} nodes at dt = {dt:g} s ({h * dt:.2f} s of look-ahead). "
        "Every quantity is in the ANCHOR'S YAW FRAME -- the frame the window is packed in -- "
        "so x is downrange, y lateral, and the orientation is relative to the robot's heading "
        "at the re-plan. Solid black is the recorded reference; the other arms are named in "
        "the legend. In the ankle, joint and sagittal rows, solid is the LEFT limb and dashed "
        "the RIGHT; in the error panel, solid is the base position and dashed the mean of the "
        "two feet. Ankles are forward kinematics from each arm's own predicted joint angles "
        "and base pose, through the same chain for every arm, so what remains is the "
        "difference between the motions and not between two models of the robot.")
    stem = os.path.join(out_dir, f"dp{idx:02d}_{ref.family.replace('family_', '')}")
    written = plot_style.save_figure(fig, stem, caption=caption)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_W = 88

#: Short column names. The full block names run the header columns together at any width the
#: numbers still fit in.
_SHORT: dict[str, str] = {
    "base_pos": "pos", "base_ori": "ori", "base_lin_vel": "lin_vel",
    "base_ang_vel": "ang_vel", "joint_pos": "joint",
}


def _rule(title: str = "") -> str:
    """A labelled horizontal rule."""
    return f"\n{('=== ' + title + ' ') if title else ''}".ljust(_W, "=")


def arm_table(arms: list[Arm], stats: dict[str, dict[str, Any]], groups: list[str],
              extra: bool = True) -> str:
    """One table: a row per arm, a column per channel group, in physical units.

    Args:
        arms: The arms, in draw order.
        stats: ``stats[arm.name]`` with ``rmse``, ``rmse_vs_floor``, ``foot_p50``,
            ``foot_max``, ``tok_acc``.
        groups: Group names to show.
        extra: Include the vs-floor, foot and token columns.

    Returns:
        A multi-line string.
    """
    cols = [g for g in REPORT_GROUPS if g in groups]
    head = f"{'arm':<26}" + "".join(
        f"{_SHORT.get(g, g) + '[' + GROUP_UNITS.get(g, '?') + ']':>16}" for g in cols)
    pinned = any(stats[a.name].get("pin") is not None for a in arms)
    if extra:
        if pinned:
            head += f"{'PIN[m]':>10}{'free[m]':>10}"
        head += f"{'geo[rad]':>10}{'d(floor)':>10}{'foot p50':>10}{'foot max':>10}{'tok':>7}"
    lines = [head, "-" * len(head)]
    for a in arms:
        s = stats[a.name]
        row = f"{a.name:<26}" + "".join(
            (f"{s['rmse'][g]:>16.5f}" if g in s["rmse"] else f"{'-':>16}") for g in cols)
        if extra:
            if pinned:
                for k in ("pin", "free"):
                    v = s.get(k)
                    row += (f"{v:>10.4f}" if v is not None and np.isfinite(v)
                            else f"{'-':>10}")
            g = s.get("ori")
            row += (f"{g:>10.5f}" if g is not None else f"{'-':>10}")
            vf = s.get("rmse_vs_floor", {}).get("base_pos")
            row += (f"{vf:>10.4f}" if vf is not None else f"{'-':>10}")
            row += (f"{s['foot_p50']:>10.4f}" if s.get("foot_p50") is not None
                    else f"{'-':>10}")
            row += (f"{s['foot_max']:>10.4f}" if s.get("foot_max") is not None
                    else f"{'-':>10}")
            row += (f"{s['tok_acc'] * 100:>6.0f}%" if s.get("tok_acc") is not None
                    else f"{'-':>7}")
        lines.append(row)
    return "\n".join(lines)


def family_table(arms: list[Arm], per_item: dict[str, dict[str, Any]], groups: list[str],
                 refs: list[Ref], group: str = "base_pos") -> str:
    """One group's RMSE per arm per family, pooled within the family.

    The gating report and the FM's val metrics are both quoted per family, and the families
    are not interchangeable -- the tokenizer's own floor spans 0.0135 on ``stairs_dn_pure``
    to 0.0582 on ``unknown``, a factor of 4.3. A single pooled row hides which of those a
    difference came from.

    Args:
        arms: The arms.
        per_item: Per-arm arrays.
        groups: Group names, to locate ``group``.
        refs: The selected datapoints, for their family labels.
        group: Which channel group to tabulate.

    Returns:
        A multi-line string.
    """
    if group not in groups:
        return ""
    k = groups.index(group)
    fams = sorted({r.family for r in refs})
    idx = {f: np.array([i for i, r in enumerate(refs) if r.family == f]) for f in fams}
    head = f"{'arm':<26}" + "".join(f"{f.replace('family_', '')[:14]:>16}" for f in fams)
    lines = [head, "-" * len(head)]
    for a in arms:
        if a.kind == "truth":
            continue
        r = per_item[a.name]["rmse"][:, k]
        lines.append(f"{a.name:<26}" + "".join(
            f"{float(np.sqrt((r[idx[f]] ** 2).mean())):>16.5f}" for f in fams))
    lines.append("")
    lines.append(f"{'n windows':<26}" + "".join(f"{len(idx[f]):>16d}" for f in fams))
    return "\n".join(lines)


def checks_table(checks: list[Check]) -> str:
    """Render the preflight table."""
    lines = []
    for c in checks:
        mark = " ok " if c.ok else ("FAIL" if c.fatal else "warn")
        lines.append(f"[{mark}] {c.name:<42} {c.detail}")
    return "\n".join(lines)


#: Arm colours by ROLE rather than by draw order, so the family of an arm is readable in the
#: scene without consulting the legend: one grey truth, one yellow floor, cyan for the
#: teacher-forced decode, green for the greedy one, PINKS for the categorical draws, ORANGES
#: for the token-forcing sweep, red for the generator. Declared here rather than in the scene
#: so the terminal legend and the 3-D view name the same colour for the same arm.
_ROLE_COLORS: dict[str, tuple[float, float, float]] = {
    "truth": (0.78, 0.78, 0.80),
    "tok_floor": (0.94, 0.84, 0.24),
    "tf": (0.24, 0.78, 0.92),
    "ar_greedy": (0.20, 0.84, 0.43),
    "ar_sample_0": (0.96, 0.43, 0.71),
    "ar_sample_1": (0.84, 0.35, 0.84),
    "ar_sample_2": (0.67, 0.43, 0.96),
    "ar_sample_3": (0.96, 0.55, 0.47),
    "force1": (0.98, 0.63, 0.20),
    "force4": (0.84, 0.43, 0.12),
    "fm_zero": (0.94, 0.24, 0.24),
}
_FALLBACK_COLOR: tuple[float, float, float] = (0.59, 0.37, 0.16)


def arm_color(name: str, planner_tags: list[str]) -> tuple[float, float, float]:
    """Colour for one arm: the ROLE picks the hue, the planner picks the shade.

    With two planners in one scene the same role appears twice, and giving them unrelated
    colours would hide the comparison the second planner is there to make. So ``p1``'s arms
    are the same hue as ``p0``'s, darkened -- the pair reads as a pair.

    Args:
        name: Arm name, ``"<tag>/<role>"`` for a planner arm and a bare role otherwise.
        planner_tags: Planner tags in order, for the shade.

    Returns:
        RGB in 0-1.
    """
    tag, _, role = name.rpartition("/")
    base = _ROLE_COLORS.get(role or name, _FALLBACK_COLOR)
    idx = planner_tags.index(tag) if tag in planner_tags else 0
    return tuple(c / (1.0 + 0.45 * idx) for c in base)


def main() -> int:
    """Load, check, run every arm, print the tables and serve the scene."""
    args = parse_args()
    tp = _load_trainer()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)

    paths = list(args.planner)
    tags = ["planner"] if len(paths) == 1 else [f"p{i}" for i in range(len(paths))]
    planners = [load_planner(p, t, dev, args.raw_weights) for p, t in zip(paths, tags)]
    p0 = planners[0]
    tok_cfg = p0.tok_cfg

    fm = fm_cfg = fm_ck = None
    fm_weights = "-"
    if args.fm:
        fm, fm_cfg, fm_ck, fm_weights = load_fm(args.fm, dev, args.raw_weights)

    rio = _recorded_data_io()
    from robot_rl.token_planner.data import _EXTERO_BY_COST  # noqa: PLC0415

    # Metadata from the FIRST file, matching `load_recordings` and the FM trainer: the
    # layout must agree across the set anyway, and disagreeing files are refused there.
    first = args.data.split(",")[0].strip()
    meta, _n_first = rio.read_recording_meta(first)
    family_names = [str(s) for s in (getattr(meta, "family_names", None) or ["unknown"])]

    # The dataset is built the way stage B built it, overrides included: at its trained
    # pinned_axis_loss_weight of 20.0 the dataset demands a clip-library join and RAISES when
    # it fails, so skipping the override would make this tool need a 6.8 GB library it has no
    # use for.
    cfg = FlowMatchingConfig.from_dict(p0.ckpt["fm_config"])
    cfg.fk_model_path = tp._relocate_fk_model(cfg.fk_model_path)      # noqa: SLF001
    for field_name, (value, why) in tp._OVERRIDES.items():            # noqa: SLF001
        if getattr(cfg, field_name, None) != value:
            print(f"[inspect] override {field_name}: "
                  f"{getattr(cfg, field_name, None)!r} -> {value!r} ({why})")
            setattr(cfg, field_name, value)
    cfg.future_len = int(tok_cfg.chunk_len)
    cfg.reference_blocks = list(tok_cfg.reference_blocks)
    cfg.seed = int(args.seed)
    if args.pinned:
        # The metric only; the LOSS weight stays at the override's 1.0. Setting that instead
        # would make the dataset RAISE when the library join fails, where the metric degrades
        # to off with a warning -- and no loss is computed here anyway.
        cfg.pinned_axis_metrics = True
        print("[inspect] --pinned: joining the clip library for the per-node anchor mask "
              f"({cfg.clip_library_path or meta.clip_library_path}).")
    if args.depth_lag_steps >= 0:
        cfg.depth_latency_probs = [0.0] * args.depth_lag_steps + [1.0]
        print(f"[inspect] camera latency pinned to {args.depth_lag_steps} control steps.")

    val_eps = {int(e) for e in (p0.ckpt.get("val_episode_ids") or [])}
    paths = [q.strip() for q in args.data.split(",") if q.strip()]
    missing = [q for q in paths if not os.path.exists(q)]
    if missing:
        raise SystemExit(f"recording(s) not found: {missing}")
    # `load_recordings` owns the per-file episode offset -- the thing that decides which
    # windows are held out -- so the index pass goes through it rather than restating it.
    from robot_rl.token_planner.data import load_recordings  # noqa: PLC0415

    index, _ = load_recordings(paths, keys=list(rio.WINDOW_INDEX_KEYS))
    counts = [rio.read_recording_meta(q)[1] for q in paths]
    bounds = np.cumsum(counts)
    refs = select_rows(index, family_names, val_eps, list(cfg.skip_families or []),
                       bool(cfg.skip_distractors), args.num_points, args.split,
                       [f for f in args.families.split(",") if f], args.seed,
                       stratify=not args.no_stratify)
    rows = np.asarray([r.row for r in refs], dtype=np.int64)
    print(f"[inspect] {len(refs)} datapoints selected from {int(bounds[-1])} windows across "
          f"{len(paths)} recording(s) ({len(val_eps)} held-out episodes recorded by the "
          f"planner).")

    keys = [k for k in rio.WINDOW_ARRAY_KEYS
            if k not in set(_EXTERO_BY_COST) - {cfg.extero_source}]
    # Rows are GLOBAL indices into the concatenation; split them back per file, read each
    # file's own rows, and reassemble in the SELECTION's order -- not in file order, or every
    # per-window label would be attached to the wrong window.
    data: dict[str, np.ndarray] = {}
    for fi, path in enumerate(paths):
        lo = int(bounds[fi - 1]) if fi else 0
        take = np.flatnonzero((rows >= lo) & (rows < int(bounds[fi])))
        if take.size == 0:
            continue
        d, _ = rio.load_recorded_windows(path, keys=keys, rows=rows[take] - lo)
        for k, v in d.items():
            if v is None:
                continue
            if k not in data:
                data[k] = np.zeros((len(rows),) + v.shape[1:], dtype=v.dtype)
            data[k][take] = v
    # `episode_id` is the one array whose values were offset by `load_recordings`; take it
    # from the index rather than from the per-file read, which carries the raw ids.
    data["episode_id"] = np.asarray(index["episode_id"])[rows]

    def _norm(key: str) -> Normalizer | None:
        """The planner's own saved normalizer, so nothing here is refitted."""
        st = p0.ckpt.get(key)
        return Normalizer.from_state_dict(st) if st is not None else None

    ds = FutureStateDataset(
        data, meta, cfg, np.arange(len(rows)),
        state_normalizer=_norm("state_normalizer"), cmd_normalizer=_norm("cmd_normalizer"),
        extero_normalizer=_norm("extero_normalizer"), add_noise=False)

    # ---------------- preflight ----------------
    checks: list[Check] = []
    from robot_rl.flow_matching.dataset import (  # noqa: PLC0415
        _distractor_keep_mask,
        _family_keep_mask,
    )

    # The dataset's OWN mask functions, on the loaded subset: every row this tool picked must
    # be one the run would have trained on. Reproducing the filters would only re-create the
    # index-space bug they document; asking them is the check.
    m = len(rows)
    keep = _distractor_keep_mask(data, cfg, m)
    fkeep = _family_keep_mask(data, meta, cfg, m)
    keep = fkeep if keep is None else (keep if fkeep is None else (keep & fkeep))
    n_bad = 0 if keep is None else int((~keep).sum())
    checks.append(Check("selected rows survive the run's filters", n_bad == 0,
                        f"{m} rows, {n_bad} would have been dropped"))

    if args.tokenizer:
        tck = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
        emb = {k[len("tokenizer."):]: v for k, v in p0.model.state_dict().items()
               if k.startswith("tokenizer.")}
        # Both arms, because a planner is trained against ONE tokenizer snapshot and there is
        # no rule saying it was that file's `best`. NOT fatal: the tokenizer actually used for
        # every arm here is the planner's own embedded copy, so a mismatch does not make these
        # numbers incomparable -- it means the standalone file describes a different model,
        # and therefore that ITS gating report is not this planner's floor.
        detail = " | ".join(
            f"vs {arm}: {_state_dicts_equal(emb, tck[arm])[1]}"
            for arm in ("model", "model_ema") if tck.get(arm) is not None)
        ok = any(_state_dicts_equal(emb, tck[arm])[0]
                 for arm in ("model", "model_ema") if tck.get(arm) is not None)
        if not ok:
            detail += (f" -- the planner names {p0.ckpt.get('tokenizer_path')!r}; its "
                       "EMBEDDED copy is what every arm here uses, so this file's "
                       "gating_report is NOT this planner's floor")
        checks.append(Check(f"tokenizer weights ({p0.tag} embedded vs --tokenizer)",
                            ok, detail, fatal=False))
        ok, detail = _compare_normalizers(p0.ckpt["state_normalizer"],
                                          tck["state_normalizer"])
        checks.append(Check("state normalizer (planner vs tokenizer file)", ok, detail))
        _cmp(checks, "tokenizer config", TokenizerConfig.from_dict(tck["config"]).to_dict(),
             tok_cfg.to_dict(), "--tokenizer", p0.tag)

    for other in planners[1:]:
        a = {k: v for k, v in p0.model.state_dict().items() if k.startswith("tokenizer.")}
        b = {k: v for k, v in other.model.state_dict().items() if k.startswith("tokenizer.")}
        ok, detail = _state_dicts_equal(a, b)
        checks.append(Check(f"tokenizer weights ({p0.tag} vs {other.tag})", ok, detail))
        ok, detail = _compare_normalizers(p0.ckpt["state_normalizer"],
                                          other.ckpt["state_normalizer"])
        checks.append(Check(f"state normalizer ({p0.tag} vs {other.tag})", ok, detail))
        _cmp(checks, f"held-out episodes ({p0.tag} vs {other.tag})",
             sorted(val_eps), sorted(int(e) for e in (other.ckpt.get("val_episode_ids") or [])),
             "n=" + str(len(val_eps)), other.tag)

    if fm is not None:
        for key, label in (("state_normalizer", "state"), ("cmd_normalizer", "cmd"),
                           ("extero_normalizer", "extero")):
            a, b = p0.ckpt.get(key), fm_ck.get(key)
            if a is None or b is None:
                checks.append(Check(f"{label} normalizer (planner vs fm)", a == b,
                                    f"planner {'present' if a else 'absent'}, "
                                    f"fm {'present' if b else 'absent'}"))
                continue
            ok, detail = _compare_normalizers(a, b)
            checks.append(Check(f"{label} normalizer (planner vs fm)", ok, detail))
        for field_name in ("reference_blocks", "state_dim", "history_dim", "cmd_dim",
                           "cmd_stride", "num_history_states", "history_source",
                           "extero_source", "extero_channels", "extero_sensor_count",
                           "extero_sensor_dim", "extero_norm_mode", "extero_clip",
                           "attention_mode", "future_len"):
            _cmp(checks, f"cfg.{field_name} (planner vs fm)",
                 getattr(cfg, field_name, None), getattr(fm_cfg, field_name, None),
                 p0.tag, "fm")
        fm_eps = fm_ck.get("val_episode_ids")
        if not fm_eps:
            checks.append(Check("held-out episodes (planner vs fm)", True,
                                "the fm checkpoint records none, so the two splits cannot "
                                "be compared -- an fm arm here may be scored on windows it "
                                "was trained on", fatal=False))
        else:
            _cmp(checks, "held-out episodes (planner vs fm)", sorted(val_eps),
                 sorted(int(e) for e in fm_eps), p0.tag, "fm")

    checks.append(Check("history_source is 'reference' (the clip arm)",
                        str(cfg.history_source) == "reference",
                        f"cfg.history_source={cfg.history_source!r}"))
    _cmp(checks, "node layout (tokenizer vs dataset)",
         [int(tok_cfg.node_dim), int(tok_cfg.chunk_len), int(tok_cfg.num_joints)],
         [int(ds.state_dim), int(cfg.future_len), int(ds.num_joints)],
         "tokenizer", "dataset")

    sha = ([rio.sha256_file(q) for q in paths] if args.verify_sha
           else ["(not computed; pass --verify_sha)"] * len(paths))
    trained_on = [q.strip() for q in str(p0.ckpt.get("data", "")).split(",")]
    have, want = [os.path.basename(q) for q in paths], [os.path.basename(t) for t in trained_on]
    checks.append(Check("recordings are the planner's, in the planner's order",
                        have == want, f"{have} vs {want}" if have != want
                        else f"{len(have)} file(s) match by name and order",
                        fatal=False))
    return _finish(args, tp, dev, planners, fm, fm_cfg, fm_ck, fm_weights, meta, cfg, ds,
                   refs, family_names, checks, sha, data)


def _quat_err(a: np.ndarray, b: np.ndarray) -> float:
    """Max component error between two quaternion fields, sign-insensitive.

    ``window_to_world`` recovers the orientation through ``quat_from_matrix``, which fixes the
    sign, so the returned quaternion may be the negation of the one that went in. That is the
    same rotation; comparing raw components would report a spurious 2.0.
    """
    return float(np.minimum(np.abs(a - b), np.abs(a + b)).max())


def _finish(args, tp, dev, planners, fm, fm_cfg, fm_ck, fm_weights, meta, cfg, ds, refs,
            family_names, checks, sha, data) -> int:
    """Run every arm, reduce, print, and serve.

    Split out of :func:`main` only for length; it is one continuous procedure.

    Returns:
        Process exit code.
    """
    from robot_rl.flow_matching.dataset import resolve_term_slice  # noqa: PLC0415

    p0 = planners[0]
    items = [ds[i] for i in range(len(ds))]

    def stack(key: str) -> torch.Tensor | None:
        """Collate one item key. Kept on the CPU; ``run_arms`` moves each chunk."""
        return torch.stack([it[key] for it in items]) if key in items[0] else None

    cond = {"state": stack("state"), "history": stack("history"), "extero": stack("extero")}
    if "cmd_tokens" in items[0]:
        cond["cmd_tokens"] = stack("cmd_tokens")
    p_ref = torch.stack([it["p_ref"] for it in items])
    q_ref = torch.stack([it["q_ref"] for it in items])

    arms, token_stats = run_arms(planners, fm, fm_cfg, cond, args.temperature, args.seed,
                                 dev, batch=int(args.batch))
    by_name = {a.name: a for a in arms}

    nj, nee = int(ds.num_joints), int(ds.num_extra_frames)
    blocks = list(ds.blocks)
    groups, labels = channel_group_labels(nj, nee, blocks)
    # A CPU copy: `Normalizer.to` moves in place, so borrowing the dataset's would be a
    # side effect on an object the caller still owns.
    norm = Normalizer.from_state_dict(ds.state_normalizer.state_dict())
    std, mean = norm.std, norm.mean

    fk_path = cfg.fk_model_path if os.path.exists(str(cfg.fk_model_path)) else DEFAULT_MJCF
    fk = FKChain(fk_path, [str(j) for j in meta.state_joint_names], list(FOOT_FRAMES),
                 base_link="pelvis_link", dtype=torch.float32)

    # ---------------- physical space ----------------
    print(f"[inspect] lifting {len(arms)} arms to world and FK'ing the ankles "
          f"({len(arms) * len(refs) * int(ds.state.shape[1])} configurations)...",
          flush=True)
    world = {a.name: world_of(a.nodes, norm, p_ref, q_ref, nj, nee, blocks, fk)
             for a in arms}

    # ---------------- controls ----------------
    # EVERY planner, not just the first. The target ids come from p0's embedded tokenizer
    # and are used as p1's teacher-forcing targets, its force_tokens and its token-accuracy
    # reference -- all of which are meaningless if p1 tokenizes differently. The weight
    # comparison above is fatal precisely so that cannot happen, and this is the behavioural
    # half of the same statement: p1 decoding p0's ids must land on p0's floor.
    n_tok = p0.model.num_tokens
    for pl in planners:
        ctrl = by_name.get(f"{pl.tag}/force{n_tok}")
        if ctrl is None:
            continue
        d = float((ctrl.nodes - by_name["tok_floor"].nodes).abs().max())
        checks.append(Check(f"{pl.tag}/force{n_tok} reproduces tok_floor", d == 0.0,
                            f"max|d| = {d:.3e} (a non-zero here means this harness is "
                            "wrong, not the model)"))

    rf = np.asarray(data["ref_future"])[:, :ds.state.shape[1]]
    names, dims = list(meta.ref_future_term_names), list(meta.ref_future_term_dims)

    def term(name: str) -> np.ndarray:
        """One recorded ref_future term, raw world frame."""
        a, b = resolve_term_slice(names, dims, name)
        return rf[..., a:b]

    tw = world["truth"]
    d_pos = float(np.abs(tw["base_pos"] - term("root_pos_w")).max())
    d_quat = _quat_err(tw["base_quat"], term("root_quat_w"))
    d_jnt = float(np.abs(tw["joint_pos"] - term("joint_pos")).max())
    checks.append(Check("window round-trip reproduces the recording",
                        max(d_pos, d_quat, d_jnt) < 1e-4,
                        f"pos {d_pos:.3e} m, quat {d_quat:.3e}, joints {d_jnt:.3e} rad"))

    have_ee = all(f"{f}:pos_w" in names for f in FOOT_FRAMES)
    if have_ee:
        rec_feet = np.stack([term(f"{f}:pos_w") for f in FOOT_FRAMES], axis=2)
        dist = np.linalg.norm(tw["feet"] - rec_feet, axis=-1)
        p50, p99 = (float(v) for v in np.percentile(dist, [50.0, 99.0]))
        mx = float(dist.max())
        # Gated on p99, not the max: a handful of windows whose recorded end-effector columns
        # disagree with their own joint angles is a defect in THOSE windows (an augmented
        # recording whose reference was displaced without its ee block being recomputed looks
        # exactly like this), while a systematic disagreement means the FK chain or the joint
        # order is wrong -- and only the second invalidates the foot metrics. Reported as
        # three numbers so the two are distinguishable at a glance.
        checks.append(Check("FK reproduces the recorded ankle positions", p99 < 2e-3,
                            f"p50 {1000 * p50:.3f} mm, p99 {1000 * p99:.3f} mm, "
                            f"max {1000 * mx:.1f} mm over {dist.size} samples"
                            + ("" if p99 < 2e-3 else
                               "  -- ee_pos is NOT in reference_blocks, so this is a "
                               "statement about the RECORDING, not about the arms")))

    # ---------------- metrics ----------------
    fam_code = torch.as_tensor(np.asarray(data["clip_family"]), dtype=torch.long)
    target = cond["state"]                                    # CPU, like every arm's nodes
    ori_g = groups.index("base_ori") if "base_ori" in groups else -1
    floor = by_name["tok_floor"]
    per_item: dict[str, dict[str, Any]] = {}
    pooled: dict[str, dict[str, float]] = {}
    for a in arms:
        acc = ReconAccumulator(group_names=groups, labels=labels, std=std.clone(),
                               mean=mean.clone(), chunk_len=int(ds.state.shape[1]),
                               vocab_size=int(p0.tok_cfg.vocab_size),
                               num_families=len(family_names), family_names=family_names)
        acc.update(target, a.nodes,
                   a.ids if a.ids is not None else torch.zeros_like(floor.ids), fam_code)
        rep = acc.report()
        pooled[a.name] = {**rep["rmse"], "base_ori_rad": rep["ori_geodesic_rad"],
                          "_by_family": rep["rmse_by_family"]}
        r = group_rmse_per_item(target, a.nodes, labels, std, len(groups))     # (M, G)
        vf = group_rmse_per_item(floor.nodes, a.nodes, labels, std, len(groups))
        ori = (ori_geodesic_per_item(target, a.nodes, labels, std, mean, ori_g)
               if ori_g >= 0 else torch.zeros(r.shape[0]))
        pin_rms = free_rms = None
        if ds.anchor_mask is not None:
            pin_rms, free_rms = pinned_free_rms(
                target, a.nodes, norm, ds.anchor_mask, ds.mask_frame_yaw_rel,
                nj, nee, blocks)
        dist = np.linalg.norm(world[a.name]["feet"] - tw["feet"], axis=-1)      # (M, H, K)
        acc_tok = (None if a.ids is None else
                   (a.ids == floor.ids).float().mean(dim=1).cpu().numpy())
        per_item[a.name] = {
            "rmse": r.numpy(), "rmse_vs_floor": vf.numpy(), "ori": ori.numpy(),
            "foot_p50": np.median(dist.reshape(dist.shape[0], -1), axis=1),
            "foot_max": dist.reshape(dist.shape[0], -1).max(axis=1),
            "tok_acc": acc_tok, "pin": pin_rms, "free": free_rms,
        }
        # Pooling the per-item numbers must reproduce the accumulator's own, or the two
        # tables on screen are computing different quantities.
        back = np.sqrt((per_item[a.name]["rmse"] ** 2).mean(axis=0))
        worst = max(abs(float(back[i]) - rep["rmse"][g]) for i, g in enumerate(groups))
        if worst > 1e-9:
            checks.append(Check(f"per-item reduction matches the accumulator ({a.name})",
                                False, f"worst group differs by {worst:.3e}"))
    checks.append(Check("per-item reduction matches the accumulator", True,
                        "pooled per-item RMSE reproduces ReconAccumulator on every arm"))

    # ---------------- the depth cloud, and the check that it was lifted correctly ----------
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: PLC0415

    n_sensors, nch = int(cfg.extero_sensor_count), int(cfg.extero_channels or 1)
    sensor_dim = int(cfg.extero_sensor_dim) // max(nch, 1)
    lag = getattr(ds, "_lagged_pose", None)
    clouds, worst_lift, world_scale = [], 0.0, 1.0
    for i in range(len(refs)):
        pose = (lag[i] if lag is not None
                else torch.cat([ds.p_ref[i], ds.q_ref[i]]))
        pts, valid, cam = cloud_of(ds.extero[i], n_sensors, nch, sensor_dim,
                                   pose[0:3], pose[3:7])
        clouds.append({"points": pts, "valid": valid, "cam": cam,
                       "pose_pos": pose[0:3].numpy().astype(np.float32)})
        if nch >= 4 and pts.shape[0]:
            # Rotate the lifted cloud back and it must be the raw yaw-frame xyz it came from.
            # The lift is the ONLY step between the dataset's own frame and the drawing, so
            # this is what says the points are where the model saw them rather than merely
            # somewhere plausible.
            raw = ds.extero[i].cpu().reshape(n_sensors, nch, sensor_dim)[:, 1:4, :]
            raw = raw.permute(0, 2, 1).reshape(-1, 3)
            back = quat_apply_inverse(
                yaw_quat(pose[3:7].reshape(1, 4)).expand(raw.shape[0], 4),
                torch.from_numpy(pts) - pose[0:3].reshape(1, 3))
            worst_lift = max(worst_lift, float((back - raw).abs().max()))
            world_scale = max(world_scale, float(np.abs(pts).max()))
    # The tolerance is SCALED, because the tile layout puts world coordinates in the
    # kilometres -- 2589 m on this library -- where one float32 ulp is already 3.1e-4 m. An
    # absolute tolerance would fail a transform that is exact and pass one that is wrong on a
    # tile near the origin.
    ulp = float(np.finfo(np.float32).eps) * world_scale
    checks.append(Check("depth cloud lift round-trips to the yaw frame",
                        worst_lift <= 4.0 * ulp,
                        f"max|d| = {worst_lift:.3e} m = {worst_lift / max(ulp, 1e-30):.2f} "
                        f"float32 ulp at the |p| = {world_scale:.0f} m these tiles sit at "
                        f"({len(refs)} datapoints)"))

    return _report(args, planners, fm, fm_weights, meta, cfg, ds, refs, family_names, checks,
                   sha, data, arms, world, pooled, per_item, token_stats, groups, labels,
                   clouds)


def _stats_for(arms: list[Arm], per_item: dict[str, dict[str, Any]], groups: list[str],
               i: int | None) -> dict[str, dict[str, Any]]:
    """Shape the metrics for :func:`arm_table`, for one datapoint or pooled over all.

    Args:
        arms: The arms.
        per_item: Per-arm arrays from ``_finish``.
        groups: Group names.
        i: Datapoint index, or ``None`` to pool.

    Returns:
        ``{arm name: {"rmse", "rmse_vs_floor", "foot_p50", "foot_max", "tok_acc"}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for a in arms:
        d = per_item[a.name]
        if i is None:
            rmse = {g: float(np.sqrt((d["rmse"][:, k] ** 2).mean()))
                    for k, g in enumerate(groups)}
            vf = {g: float(np.sqrt((d["rmse_vs_floor"][:, k] ** 2).mean()))
                  for k, g in enumerate(groups)}
            foot_p50, foot_max = float(np.median(d["foot_p50"])), float(d["foot_max"].max())
            tok = None if d["tok_acc"] is None else float(d["tok_acc"].mean())
            ori = float(np.mean(d["ori"]))
            # Pooled as a root-mean-square over windows, matching every other pooled cell;
            # nan-aware because a window can have no pinned axis at all.
            pin, free = (None if d[k] is None
                         else float(np.sqrt(np.nanmean(np.asarray(d[k]) ** 2)))
                         for k in ("pin", "free"))
        else:
            rmse = {g: float(d["rmse"][i, k]) for k, g in enumerate(groups)}
            vf = {g: float(d["rmse_vs_floor"][i, k]) for k, g in enumerate(groups)}
            foot_p50, foot_max = float(d["foot_p50"][i]), float(d["foot_max"][i])
            tok = None if d["tok_acc"] is None else float(d["tok_acc"][i])
            ori = float(d["ori"][i])
            pin, free = (None if d[k] is None else float(np.asarray(d[k])[i])
                         for k in ("pin", "free"))
        # `truth` is its own reference; showing 0.000 in every column invites reading the
        # other rows as if they were relative to something else.
        if a.kind == "truth":
            out[a.name] = {"rmse": {}, "rmse_vs_floor": {}, "foot_p50": None,
                           "foot_max": None, "tok_acc": None, "ori": None,
                           "pin": None, "free": None}
            continue
        out[a.name] = {"rmse": rmse, "rmse_vs_floor": vf, "foot_p50": foot_p50,
                       "foot_max": foot_max, "tok_acc": tok, "ori": ori,
                       "pin": pin, "free": free}
    return out


def _report(args, planners, fm, fm_weights, meta, cfg, ds, refs, family_names, checks,
            sha, data, arms, world, pooled, per_item, token_stats, groups, labels,
            clouds) -> int:
    """Print every table, build the scene payload, and serve."""
    p0 = planners[0]
    tok_cfg = p0.tok_cfg

    print(_rule("ARTIFACTS"))
    for q, h in zip([x.strip() for x in args.data.split(",") if x.strip()], sha):
        print(f"recording   {q}")
        print(f"            sha256 {h}")
    print(f"            {ds.state.shape[0]} rows loaded | H={meta.history_len} "
          f"F={meta.ref_future_len} node={sum(meta.state_term_dims)} "
          f"| history_anchor={meta.history_anchor!r} dt={meta.dt}")
    print(f"            clip library {meta.clip_library_path}  "
          f"sha {str(meta.clip_library_sha256)[:12]}")
    print(f"            robot_asset  {str((meta.robot_asset or {}).get('short', '-'))}")
    print(f"            task {meta.task_name!r}  teacher run {meta.run_name!r}")
    print(f"tokenizer   embedded in {p0.tag}  D={tok_cfg.downsample} H={tok_cfg.chunk_len} "
          f"N={p0.model.num_tokens} V={tok_cfg.vocab_size} levels={list(tok_cfg.fsq_levels)}")
    print(f"            it produces tok_floor AND the target ids every planner is scored "
          f"against{'' if len(planners) == 1 else '; the other planners are asserted above to embed the same weights'}.")
    if args.tokenizer:
        print(f"            compared against {args.tokenizer}")
    for pl in planners:
        print(f"{pl.tag:<12}{pl.path}")
        print(f"            epoch {pl.ckpt.get('epoch')} step {pl.ckpt.get('step')} "
              f"weights={pl.weights}  d_model={pl.cfg.d_model} layers={pl.cfg.num_layers} "
              f"prefix_len={pl.model.prefix_len}")
        # What the TRAINING loop recorded for this checkpoint, printed beside what this run
        # measures. Its `rmse/*` is the history=clip, tokens=ar_greedy arm, so it is the
        # `<tag>/ar_greedy` row below -- on the run's whole val split rather than this
        # stratified sample of it, so expect a difference, not a match.
        mt = pl.ckpt.get("metrics") or {}
        if mt:
            keys = [k for k in ("ce", "top1", "entropy_nats", "rmse_mean", "rmse/base_pos",
                                "rmse/joint_pos", "ori_geodesic_rad") if k in mt]
            print("            train-time metrics: "
                  + "  ".join(f"{k}={float(mt[k]):.5g}" for k in keys)
                  + "   (rmse/* is the ar_greedy arm)")
        else:
            print("            train-time metrics: NONE RECORDED in this checkpoint, so "
                  "nothing here can be checked against\n"
                  "                                the numbers this run was selected on.")
    if fm is not None:
        print(f"{'fm':<12}{args.fm}")
        print(f"            weights={fm_weights}  steps={cfg.num_sampling_steps} "
              f"attention_mode={cfg.attention_mode!r}  arm: x0 = 0")
    print(f"{'cond':<12}history_source={cfg.history_source!r} (the CLIP arm) | "
          f"N={cfg.num_history_states} | extero={cfg.extero_source!r} "
          f"{cfg.extero_sensor_count}x{cfg.extero_sensor_dim} "
          f"({cfg.extero_channels} ch, clip +-{cfg.extero_clip}) | "
          f"cmd {cfg.cmd_dim}x{ds.num_cmd_tokens}")
    if cfg.depth_latency_probs:
        print(f"{'':<12}camera latency probs {list(cfg.depth_latency_probs)} "
              "(drawn per row from this run's seed, NOT training's draw for these rows)")

    print(_rule("CONSISTENCY"))
    print(checks_table(checks))
    bad = [c for c in checks if not c.ok and c.fatal]
    if bad and not args.allow_mismatch:
        print(f"\n{len(bad)} fatal consistency failure(s). Nothing below would be "
              "comparable across arms, so the scene is not served. Re-run with "
              "--allow_mismatch to look anyway.")
        return 1
    if bad:
        print(f"\n[WARN] {len(bad)} fatal check(s) downgraded by --allow_mismatch. The arms "
              "are NOT comparable in physical units; do not quote these numbers.")

    print(_rule("DATAPOINTS"))
    print(f"{'#':>3}  {'split':<8}{'family':<24}{'episode':>8}{'clip':>8}{'step':>7}"
          f"{'env':>6}{'row':>9}")
    for i, r in enumerate(refs):
        print(f"{i:>3}  {'HELDOUT' if r.held_out else 'train':<8}{r.family:<24}"
              f"{r.episode:>8}{r.clip:>8}{r.clip_step:>7}{r.env:>6}{r.row:>9}")

    legend = ", ".join(f"{a.name}" for a in arms)
    print(_rule(f"SUMMARY over {len(refs)} datapoints  (pooled, physical units)"))
    print(arm_table(arms, _stats_for(arms, per_item, groups, None), groups))
    print("\n  d(floor)  = base_pos RMSE against tok_floor rather than the truth. On a "
          "planner arm that is what the\n              PLANNER adds to what quantization "
          "already cost; on fm_zero it says whether the\n              generator's error "
          "lands in the same place the tokenizer's does.")
    print("  foot p50/max = FK'd ankle position error against the truth window's own FK [m].")
    if any(_stats_for(arms, per_item, groups, None)[a.name].get("pin") is not None
           for a in arms):
        print("  PIN/free  = base-position error [m] on the axes the terrain PINS vs the ones "
              "it frees, per node.\n"
              "              Free error is re-anchored onto the robot and costs nothing; "
              "PINNED error is the\n"
              "              reference sliding off the terrain, which is what a fall looks "
              "like. Rotation is\n"
              "              excluded -- a 6-D rotation error cannot be pooled with metres.")
    print("  geo       = base-orientation GEODESIC angle [rad], the same quantity both "
          "trainers report as\n              ori_geodesic_rad -- so a row here is directly "
          "comparable with a checkpoint's own metrics.")
    print("  tok       = fraction of the N tokens equal to the target's.")
    # WHERE TO LOOK. The dropdown is stratified, not ordered by difficulty, so its first
    # entry is as likely to be a window every arm agrees on as a revealing one -- and on an
    # easy window every token matches and the arms are bit-identical, which reads as the tool
    # being broken rather than as the model being right.
    ref_arm = next((a.name for a in arms if a.name.endswith("/ar_greedy")), None)
    if ref_arm is not None and "base_pos" in groups:
        k = groups.index("base_pos")
        err = per_item[ref_arm]["rmse"][:, k]
        acc = per_item[ref_arm]["tok_acc"]
        order = np.argsort(-err)
        print(f"\n--- where to look: {ref_arm} base_pos error per datapoint ---")
        for tag, sel in (("worst", order[:5]), ("best", order[::-1][:5])):
            for i in sel:
                print(f"  {tag:<6} #{int(i):02d}  {err[i]:.4f} m   "
                      f"tokens {100 * float(acc[i]):3.0f}% matched   {refs[i].label}")
        n_exact = int((acc >= 1.0).sum())
        print(f"  {n_exact} of {len(refs)} datapoints decode EVERY token correctly; on those "
              "every arm at 100% is\n  bit-identical to the others by construction, and the "
              "only spread you can see is the\n  tokenizer floor and the sampled draws.")

    extra = [q.strip() for grp in (args.extra_tokenizer or []) for q in grp.split(",")
             if q.strip()]
    if extra:
        print(_rule("TOKENIZER FLOORS"))
        print(score_extra_tokenizers(
            extra, planners[0].model.tokenizer, planners[0].tok_cfg,
            next(a for a in arms if a.kind == "truth").nodes,
            Normalizer.from_state_dict(ds.state_normalizer.state_dict()), labels, groups,
            torch.device(args.device), anchor_mask=ds.anchor_mask,
            frame_yaw_rel=ds.mask_frame_yaw_rel, nj=int(ds.num_joints),
            nee=int(ds.num_extra_frames), blocks=list(ds.blocks)))

    ft = family_table(arms, per_item, groups, refs, "base_pos")
    if ft:
        print(f"\n--- base_pos RMSE [m] per family ---")
        print(ft)
    for tag, ts in token_stats.items():
        ce, ent = ts["ce"].mean(dim=0), ts["entropy"].mean(dim=0)
        print(f"\n  {tag} teacher-forced, per token position (nats):")
        print("    pos      " + "".join(f"{i:>7d}" for i in range(ce.numel())))
        print("    CE       " + "".join(f"{float(v):>7.3f}" for v in ce))
        print("    entropy  " + "".join(f"{float(v):>7.3f}" for v in ent))

    print(_rule("PER DATAPOINT"))
    point_tables: list[str] = []
    for i, r in enumerate(refs):
        st = _stats_for(arms, per_item, groups, i)
        body = arm_table(arms, st, groups)
        head = f"#{i:<3} {r.label}"
        point_tables.append(f"{head}\n{body}")
        print(f"\n{head}\n{body}")

    if args.out:
        # Create the directory rather than raising here: everything expensive is already
        # done by this point, so a missing parent would throw the whole run away for a
        # mkdir. `plot_style.save_figure` already does the same for --plot_dir.
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({
                "data": args.data, "data_sha256": sha,
                "planners": {p.tag: {"path": p.path, "epoch": p.ckpt.get("epoch"),
                                     "weights": p.weights} for p in planners},
                "fm": args.fm, "groups": groups,
                "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
                "points": [{"index": i, **{k: getattr(r, k) for k in
                                           ("row", "episode", "clip", "clip_step", "env",
                                            "family", "held_out")},
                            "stats": _stats_for(arms, per_item, groups, i)}
                           for i, r in enumerate(refs)],
                "pooled": _stats_for(arms, per_item, groups, None),
            }, fh, indent=2, default=float)
        print(f"\nwrote {args.out}")

    if args.plot_dir:
        plot_arms = _resolve_plot_arms(
            [a.strip() for a in args.plot_arms.split(",") if a.strip()], arms)
        if args.plot_points:
            which = [int(v) for v in args.plot_points.split(",") if v.strip() != ""]
        else:
            # The WORST three under the greedy arm, not the first three: on an easy window
            # every token matches and the arms decode bit-identically, which makes a perfect
            # figure that says nothing.
            ref_arm = next((a.name for a in arms if a.name.endswith("/ar_greedy")),
                           plot_arms[-1].name)
            k = groups.index("base_pos") if "base_pos" in groups else 0
            which = list(np.argsort(-per_item[ref_arm]["rmse"][:, k])[:3])
        bad = [i for i in which if not 0 <= int(i) < len(refs)]
        if bad:
            raise SystemExit(f"--plot_points out of range for {len(refs)} datapoints: {bad}")
        print(_rule("PLOTS"))
        print(f"arms: {', '.join(a.name for a in plot_arms)}")
        for i in which:
            i = int(i)
            for f in plot_datapoint(i, refs[i], plot_arms, world,
                                    ds.p_ref[i].numpy(), ds.q_ref[i].numpy(),
                                    [str(j) for j in meta.state_joint_names],
                                    float(meta.dt), args.plot_dir):
                print(f"  {f}")

    if args.no_serve:
        return 0

    # ---------------- scene payload ----------------
    # EVERY datapoint is drawn at the origin. The tiles are laid out across kilometres of
    # world -- 2589 m on this library -- so without this, switching the dropdown teleports the
    # camera to somewhere else entirely and leaves the viewer hunting in empty space. The
    # shift is a pure translation applied AFTER every metric is computed, so nothing measured
    # moves; it also buys back the float32 precision that a coordinate in the kilometres costs.
    #
    # x and y come from the anchor, so the robot stands at the origin. z is offset so the
    # NOMINAL GROUND lands on the grid rather than the pelvis -- the robot then stands on the
    # z = 0 plane at its real height, which is the only arrangement in which the grid, the
    # depth cloud and the feet all mean what they look like.
    ground = float(getattr(cfg, "extero_nominal_ground_z", 0.0) or 0.0)
    points = []
    for i, r in enumerate(refs):
        c = clouds[i]
        a_pos = ds.p_ref[i].numpy().astype(np.float64)
        origin = np.array([a_pos[0], a_pos[1], a_pos[2] + ground], dtype=np.float64)
        o32 = origin.astype(np.float32)
        points.append({
            "label": f"#{i:02d}  {r.label}",
            "table": point_tables[i],
            "origin": o32,
            "cloud": (c["points"] - o32).astype(np.float32),
            "cloud_valid": c["valid"], "cloud_cam": c["cam"],
            "anchor_pos": (a_pos - origin).astype(np.float32),
            "anchor_quat": ds.q_ref[i].numpy().astype(np.float32),
            "arms": {a.name: {
                "base_pos": (world[a.name]["base_pos"][i] - o32).astype(np.float32),
                "base_quat": world[a.name]["base_quat"][i],
                "joint_pos": world[a.name]["joint_pos"][i],
                "feet": (world[a.name]["feet"][i] - o32).astype(np.float32),
            } for a in arms},
        })
    c0 = clouds[0]
    if c0["valid"].any():
        z = c0["points"][c0["valid"]][:, 2] - float(ds.p_ref[0][2])
        print(f"\n[inspect] datapoint 0 cloud: {int(c0['valid'].sum())}/"
              f"{len(c0['valid'])} valid points, z relative to the root spans "
              f"[{z.min():.2f}, {z.max():.2f}] m (nominal ground is {ground:.2f}).")
    print(f"[inspect] every datapoint is drawn recentred on its own anchor "
          f"(x, y) with the nominal ground on z = 0; the shift is view-only and is "
          f"applied after every number above.")

    payload = {
        "title": f"{p0.tag} @ epoch {p0.ckpt.get('epoch')}",
        "dt": float(meta.dt), "chunk_len": int(ds.state.shape[1]),
        "joint_names": [str(j) for j in meta.state_joint_names],
        "urdf": args.urdf,
        "arms": [{"name": a.name, "kind": a.kind, "hidden": a.hidden, "note": a.note,
                  "color": arm_color(a.name, [p.tag for p in planners])} for a in arms],
        "artifacts": f"**{p0.tag}** epoch {p0.ckpt.get('epoch')} ({p0.weights}) | "
                     f"tokenizer D={tok_cfg.downsample} N={p0.model.num_tokens} "
                     f"V={tok_cfg.vocab_size} | history=clip",
        "checks": checks_table(checks),
        "points": points,
    }
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inspect_datapoints_scene import serve  # noqa: PLC0415

    return serve(payload, port=args.port, smoke=bool(args.scene_smoke))


if __name__ == "__main__":
    raise SystemExit(main())
