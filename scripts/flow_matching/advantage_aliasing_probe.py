# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does the advantage label depend on state the generator cannot see?

Filtered-BC / AWR conditions the generator on its own advantage label, and the generator's
conditioning is fixed: ``history``, ``extero``, ``cmd_future`` -- exactly the three arguments
to ``FlowMatchingTransformer.encode_prefix``, and nothing else.  ``robot_rl.flow_matching.value``
already draws this line for the critic (``V_onboard`` vs ``V_priv``), but that comparison is
about a STATE-only value and says nothing about the action.  The question this script answers
is sharper and is the one that actually threatens filtered-BC: at one generator-visible
``(s_partial, a)``, does the advantage's SIGN depend on state the generator cannot see (starting
with proprioception)?  If it does, the same conditioning is being asked to carry two different
answers depending on something it cannot read, and the fine-tune would learn an average that is
wrong both ways.

No critic is trained here.  The advantage comes straight from ``label_advantage.py``'s sidecar
-- ``A(s_t, a_t) = P(fail|s_t) - P(fail|s_{t+n})`` from a 2-fold-honest ``V_priv`` -- and this
script only asks whether that number is PREDICTABLE from privileged state once the
generator-visible state and the action are already known.

**The method, one line:** build two feature spaces over the same sampled windows,
``z_partial = [s_partial, a]`` and ``z_full = [s_partial, hidden, a]``, then compare the LOCAL
variance of the advantage and the LOCAL sign-disagreement rate among the k nearest neighbours in
each space, over ``k in {5, 10, 25, 50}``.  If hidden state carries advantage signal,
neighbourhoods in the full space are more homogeneous in outcome than in the partial one --
lower local variance, lower sign flips.

**Every named source is embedded SEPARATELY, to a fixed equal rank, before concatenation --
not a detail.**  ``extero`` alone is ~7800 raw dims against ``history``'s 176, ``cmd_future``'s
192 and a downsampled ``action``'s 352; per-dimension standardization fixes each dimension's
SCALE but not the block's total variance MASS, which after z-scoring equals its raw dimension
count exactly.  PCA over the naively concatenated matrix therefore finds components that are
almost entirely about the sensor with the most pixels, and a block with 4% of the mass gets
roughly 4% of the say in the neighbourhood regardless of how informative it actually is -- so a
global PCA can make two spaces look identical for a reason that has nothing to do with whether
the added block carries signal.  Each of ``history``, ``extero``, ``cmd_future``, ``action`` and
every requested hidden block is instead PCA'd to the SAME ``--block_components`` rank and
re-standardized on its own, so every source enters the concatenation with an equal variance
budget no matter its native width.

**The dimensionality trap, controlled rather than assumed away**: ``z_full`` still has one more
block than ``z_partial``, and kNN in more dimensions can look tighter for purely geometric
reasons even under the null.  The control is a SHUFFLE: permute the hidden block's EMBEDDED rows
independently of everything else (same dimensionality, zero real signal -- and equivalent to
shuffling before its PCA, since a permutation of rows does not change a covariance matrix) and
run the identical pipeline.  The real drop from partial->full must clear the shuffle's drop, or
the result is an artefact of adding dimensions rather than of adding information.

Usage::

    python scripts/flow_matching/advantage_aliasing_probe.py \\
        --rollout logs/rollouts/v2/rollout_2026-08-20_16-36-10_mode11_decor_plan_v1 \\
        --advantage logs/value/v2/advantage.npz \\
        --out logs/value/v2/aliasing_probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "scripts", "rsl_rl"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rollout_outcomes as ro  # noqa: E402

#: label_advantage.py's null marker -- a family/window with no real preference signal.
LABEL_NULL = 2

#: Raw ``.h5`` column(s) each named hidden-state block reads.  ``twist_linear`` reads
#: ``base_twist_yaw`` but keeps only its first 3 (LINEAR) channels -- see
#: :func:`_hidden_block_raw`; the angular half is IMU-observable onboard and does not belong
#: on the "hidden" side, mirroring ``robot_rl.flow_matching.value``'s own V_onboard/V_priv
#: split exactly.
HIDDEN_COLUMNS: dict[str, tuple[str, ...]] = {
    "proprio": ("proprio",),
    "scan": ("height_scan",),
    "goal": ("goal_downrange_m", "goal_lateral_m", "goal_height_err_m"),
    "twist_linear": ("base_twist_yaw",),
}


def _hidden_block_raw(name: str, W: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten one named hidden block from raw window columns, applying its own slice rules.

    Args:
        name: A key of :data:`HIDDEN_COLUMNS`.
        W: Raw window columns, as returned by :func:`_fetch_rows`.

    Returns:
        ``[N, D_name]`` float32.
    """
    if name == "twist_linear":
        return _flatten(W["base_twist_yaw"][:, :3])   # linear half only
    return np.concatenate([_flatten(W[c]) for c in HIDDEN_COLUMNS[name]], axis=1)


#: Raw columns the generator's own conditioning reads -- ``encode_prefix``'s three arguments,
#: literally.  Nothing else is part of ``s_partial`` no matter what the critic also sees.
#: The generator's conditioning, and the DEFAULT definition of "state" here -- exactly the
#: three arguments of ``FlowMatchingTransformer.encode_prefix``.
#:
#: ``history`` is the generator's own previously-served nodes, so it is PAST ACTIONS. It is
#: state in the sense that the model reads it, and not state in the sense that the policy
#: chose it -- which is why ``--state_blocks`` exists rather than this being a constant.
PARTIAL_COLUMNS: tuple[str, ...] = ("history", "extero", "cmd_future")


def _load_advantage(path: str) -> dict[str, np.ndarray]:
    """Read ``label_advantage.py``'s sidecar.

    Args:
        path: Path to ``advantage.npz``.

    Returns:
        Its arrays, keyed by name.
    """
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "meta"}


def _sample_indices(adv: dict[str, np.ndarray], n_samples: int, seed: int,
                    include_null: bool) -> np.ndarray:
    """Pick which rows of the advantage sidecar to analyze.

    Args:
        adv: The loaded sidecar.
        n_samples: How many rows to draw; 0 keeps all of them.
        seed: RNG seed.
        include_null: Keep ``label == LABEL_NULL`` rows too. Off by default: those are windows
            from a family with no measured advantage spread, so they carry no preference for
            this question to find in the first place.

    Returns:
        Row indices into ``adv``'s arrays.
    """
    pool = np.arange(adv["advantage"].shape[0])
    if not include_null:
        pool = pool[adv["label"] != LABEL_NULL]
    if n_samples and pool.size > n_samples:
        rng = np.random.default_rng(seed)
        pool = rng.choice(pool, size=n_samples, replace=False)
    return np.sort(pool)


def _fetch_rows(stem: str, uid: np.ndarray, ridx: np.ndarray,
                columns: list[str]) -> dict[str, np.ndarray]:
    """Pull exactly the sampled window rows from the raw rollout ``.h5``, by join key.

    Positional alignment between the advantage sidecar and the raw file cannot be assumed --
    ``fit_value.py`` drops censored-episode windows before it ever computes an advantage, so
    the sidecar is a strict subset of the file in general. The join is on
    ``(episode_uid, replan_index)``, the same key ``rollout_dataset.join_labels`` uses.

    Args:
        stem: Rollout stem, without ``.h5``.
        uid: ``[n]`` sampled episode ids, in the order the caller wants them back.
        ridx: ``[n]`` sampled re-plan indices, same order.
        columns: Raw window columns to read.

    Returns:
        ``{column: array}``, each ``[n, ...]`` in the SAME order as ``uid``/``ridx``.

    Raises:
        SystemExit: If any sampled ``(uid, replan)`` pair is not in the raw file -- the two
            tables are supposed to be drawn from the same recording, and a miss means they
            are not.
    """
    key_want = (uid.astype(np.int64) << 20) + ridx.astype(np.int64)
    with h5py.File(stem + ".h5", "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        uid_all = f["episode_uid"][:n]
        ridx_all = f["replan_index"][:n]
        key_all = (uid_all.astype(np.int64) << 20) + ridx_all.astype(np.int64)
        order = np.argsort(key_all)
        key_sorted = key_all[order]
        pos = np.searchsorted(key_sorted, key_want)
        pos = np.clip(pos, 0, key_sorted.size - 1)
        hit = key_sorted[pos] == key_want
        if not hit.all():
            miss = int((~hit).sum())
            raise SystemExit(
                f"{miss} sampled window(s) have no matching row in {stem}.h5 -- the advantage "
                "sidecar and this rollout must come from the same recording.")
        row_idx = order[pos]                     # aligned to `uid`/`ridx` order, unsorted
        # h5py wants a sorted, unique index list for fancy indexing; read that way, then
        # scatter back to the caller's order.
        read_order = np.argsort(row_idx)
        sorted_rows = row_idx[read_order]
        inv = np.empty_like(read_order)
        inv[read_order] = np.arange(read_order.size)
        out: dict[str, np.ndarray] = {}
        for col in columns:
            out[col] = np.asarray(f[col][sorted_rows.tolist()])[inv]
    return out


def _flatten(x: np.ndarray) -> np.ndarray:
    """``(N, ...)`` -> ``(N, prod(...))`` float32.

    Args:
        x: Any per-window array, first axis is the window.

    Returns:
        The flattened float32 array.
    """
    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], -1)


def _zscore(x: np.ndarray) -> np.ndarray:
    """Per-dimension standardization, the thing that makes a concatenated feature space usable.

    Without this, whichever block has the largest raw scale (``extero``'s raw depth range
    against a unit-ish command) dominates every distance and the neighbourhood is effectively
    conditioned on that block alone -- the exact failure mode the method note warns about.

    **Statistics are pooled over FINITE values only, and that is load-bearing.** The height
    scan carries ``-inf`` for a missed ray by construction. Taking the mean over it makes the
    column mean ``-inf``, every centred value in that column NaN, and the finite-guard below
    then maps the WHOLE COLUMN to zero -- so a block silently disappears while the run still
    prints its dimension count. Measured: the ``proprio,scan,goal,twist_linear`` arm raised
    this and returned numbers indistinguishable from the proprio-only arm.

    Args:
        x: ``[N, D]`` raw features.

    Returns:
        ``[N, D]``, zero mean and unit variance per column over the finite entries; non-finite
        entries are imputed to the column mean (0.0 after centring) and constant columns are
        left at zero rather than divided by zero.
    """
    finite = np.isfinite(x)
    if not finite.all():
        # Reported rather than merely handled: a column that is mostly imputed is not
        # evidence about the signal it names, and the caller has to be able to see that.
        frac = 1.0 - float(finite.mean())
        worst = float((~finite).mean(axis=0).max())
        print(f"[alias] _zscore: {100 * frac:.3f}% of values non-finite (worst column "
              f"{100 * worst:.1f}%); pooling over finite entries and imputing to the mean")
    xf = np.where(finite, x, np.nan)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(xf, axis=0, keepdims=True)
        sd = np.nanstd(xf, axis=0, keepdims=True)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd >= 1e-8), sd, 1.0)
    z = (np.where(finite, x, mu) - mu) / sd
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


#: Components below this fraction of a block's variance are dropped outright. It is a cheap
#: guard against an EXACTLY degenerate direction (a constant column, a duplicated sensor), not
#: the mechanism that keeps near-degenerate ones in their place -- that is the single-scalar
#: normalization in :func:`_block_embed`, which needs no threshold. A floor was tried as the
#: primary fix and was abandoned: `cmd_future` is near-degenerate in a graded way, so on one
#: row-set it left a component just above 1e-4 that still reached 110 sigma when whitened.
MIN_COMPONENT_EVR: float = 1e-8


def _block_embed(raw: np.ndarray, n_components: int, seed: int,
                 name: str = "") -> np.ndarray:
    """One SOURCE's raw features -> a rank-limited, re-standardized embedding.

    This is what makes concatenating sources across very different native widths fair: after
    z-scoring, a block's total variance mass equals its raw dimension count, so a naive
    concatenation gives ``extero`` (~7800 dims) roughly 20x the say of ``action`` (352 dims)
    regardless of which one actually predicts the advantage. Reducing every source to a
    bounded rank and re-standardizing the result gives each one a comparable variance budget
    before it ever reaches the shared kNN space.

    **The block is scaled by ONE scalar, never whitened per component, and that distinction
    was a bug.** Re-standardizing each retained component to unit variance declares every
    direction equally important, which inverts the block's own spectrum: a near-degenerate
    direction gets the same say in every distance as the leading one, and its content is
    float round-off. Measured on v4_big, ``cmd_future`` is 192 dims of a command that barely
    moves within a re-plan window; its 16th component held 3.07e-08 of the block against a
    leading 3.35e-01, and whitening put one window 236 sigma out. That both wrecked the global
    PCA -- its first component became "the direction isolating that one row" -- and blurred
    every neighbourhood through six noise dimensions, which biases the local-variance
    statistic TOWARD the no-separation answer the probe exists to test.

    A variance FLOOR was the first fix and is not sufficient: the same block is near-degenerate
    in a graded way, so on the reward row-set a component survived a 1e-4 floor and still
    reached 110 sigma. Raising the floor only moves the failure to the next dataset. Scaling
    the whole block by ``sqrt(k / total_variance)`` instead needs no threshold: a direction
    carrying no variance contributes no distance, automatically and on any data.

    Args:
        raw: ``[N, D]`` one source's raw (not yet standardized) features.
        n_components: Rank CEILING; clipped to ``min(N - 1, D)`` and then trimmed to the
            components carrying at least :data:`MIN_COMPONENT_EVR` of the block.
        seed: RNG seed for the randomized PCA solver.
        name: Block name, for the retained-rank line. Purely diagnostic.

    Returns:
        ``[N, k]`` with ``k <= n_components``, zero mean and unit variance per column.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415

    z = _zscore(raw)
    k = int(min(n_components, z.shape[0] - 1, z.shape[1]))
    if k <= 0:
        return np.zeros((z.shape[0], 0), dtype=np.float32)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    emb = pca.fit_transform(z)
    keep = pca.explained_variance_ratio_ >= MIN_COMPONENT_EVR
    if not keep.any():
        return np.zeros((z.shape[0], 0), dtype=np.float32)
    emb = emb[:, keep]
    # ONE scalar, so the block's TOTAL variance is its retained rank -- every block gets the
    # same budget (the point of this function) while keeping its own spectrum, so a direction
    # that carries almost nothing goes on carrying almost nothing.
    total = float(emb.var(axis=0).sum())
    emb = emb * np.sqrt(emb.shape[1] / total) if total > 0 else emb
    var = emb.var(axis=0)
    # The PARTICIPATION RATIO is the honest width of this block: 16 whitened columns always
    # "look" 16-dimensional, whereas this says how many directions actually carry the block.
    pr = float(var.sum() ** 2 / np.square(var).sum()) if total > 0 else 0.0
    print(f"[alias] {name or 'block'}: {emb.shape[1]} components, effective width "
          f"{pr:.1f} (participation ratio), max |z| {float(np.abs(emb).max()):.1f}")
    return emb.astype(np.float32)


def _action_features(fm_plan: np.ndarray, n_nodes: int) -> np.ndarray:
    """Downsample the emitted chunk to a handful of evenly-spaced nodes, then flatten.

    Raw distance on all 64 nodes would let a trajectory's redundant middle dominate a metric
    meant to distinguish DIFFERENT actions; the endpoints and a few nodes between them carry
    most of what makes one chunk a step and another a stumble.

    Args:
        fm_plan: ``[N, F, D]`` the emitted reference window.
        n_nodes: How many nodes to keep, evenly spaced including both endpoints.

    Returns:
        ``[N, n_nodes * D]`` float32, NOT yet standardized.
    """
    f = fm_plan.shape[1]
    idx = np.unique(np.linspace(0, f - 1, num=min(n_nodes, f)).round().astype(int))
    return _flatten(fm_plan[:, idx, :])


def _local_stats(z: np.ndarray, adv: np.ndarray, ks: list[int], seed: int) -> dict[int, dict]:
    """kNN local variance and sign-disagreement rate of ``adv``, over a range of ``k``.

    One neighbour structure is built at ``max(ks)`` and re-used for every smaller ``k`` by
    slicing, rather than rebuilding it once per ``k``.

    Args:
        z: ``[N, d]`` the feature space to search in (already PCA-reduced).
        adv: ``[N]`` the advantage value at each point.
        ks: Neighbourhood sizes to report.
        seed: Unused, kept for signature symmetry with the PCA call it follows.

    Returns:
        ``{k: {"var": float, "sign_disagree": float}}``, each averaged over every query point.
    """
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    kmax = int(max(ks))
    nn = NearestNeighbors(n_neighbors=kmax + 1, n_jobs=-1).fit(z)
    _, neigh = nn.kneighbors(z)            # [N, kmax+1], column 0 is the point itself
    neigh = neigh[:, 1:]                   # drop self
    a_neigh = adv[neigh]                   # [N, kmax]
    sign_self = np.sign(adv)[:, None]
    sign_neigh = np.sign(a_neigh)
    out: dict[int, dict] = {}
    for k in ks:
        sub = a_neigh[:, :k]
        out[int(k)] = {
            "var": float(np.mean(np.var(sub, axis=1))),
            "sign_disagree": float(np.mean((sign_neigh[:, :k] != sign_self).mean(axis=1))),
        }
    return out


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", required=True, help="Rollout stem, without .h5")
    p.add_argument("--advantage", required=True, help="advantage.npz from label_advantage.py")
    p.add_argument("--out", required=True)
    p.add_argument("--state_blocks", default=",".join(PARTIAL_COLUMNS),
                   help="Comma-separated generator-conditioning blocks forming z_partial, "
                        f"from {list(PARTIAL_COLUMNS)}. Drop 'history' to ask whether the "
                        "action looks redundant only because the generator's OWN PAST "
                        "OUTPUT is already in the tuple -- history is the previously-served "
                        "nodes, not a measurement of the robot.")
    p.add_argument("--hidden", default="proprio",
                   help="Comma-separated hidden-state blocks added to build z_full, from "
                        f"{sorted(HIDDEN_COLUMNS)}. Default is the specific worry: does the "
                        "advantage depend on proprioception, at fixed (generator state, "
                        "action). Pass e.g. 'proprio,scan,goal,twist_linear' for the broader "
                        "everything-the-critic-sees-and-the-generator-doesn't check.")
    p.add_argument("--n_samples", type=int, default=60000,
                   help="Rows analyzed. 0 = every non-null window (expensive; PCA and kNN "
                        "both grow with this).")
    p.add_argument("--include_null", action="store_true",
                   help="Keep label==null windows (families with no measured advantage "
                        "spread). Off by default -- there is nothing to alias there.")
    p.add_argument("--ks", default="5,10,25,50", help="Neighbourhood sizes, comma-separated.")
    p.add_argument("--block_components", type=int, default=16,
                   help="Rank each named source (history, extero, cmd_future, action, and "
                        "each requested hidden block) is independently PCA'd to before "
                        "concatenation, so no source's raw sensor width buys it extra say in "
                        "the shared kNN space. z_partial ends up 3*this (+ this again if the "
                        "action is included); z_full adds this once more per hidden block.")
    p.add_argument("--action_nodes", type=int, default=8,
                   help="Evenly-spaced fm_plan nodes kept for the action features.")
    p.add_argument("--no_action", action="store_true",
                   help="Drop the action block from z_partial/z_full entirely, so the "
                        "neighbourhoods and the t-SNE map are built from STATE alone. Answers "
                        "a different question from the default: not 'does hidden state alias "
                        "the same (state, action)' but 'does hidden state explain outcome "
                        "variance at a given generator-visible state, action aside' -- useful "
                        "to check the action itself isn't what is producing the mixed colours "
                        "in the t-SNE panels.")
    p.add_argument("--tsne_n", type=int, default=6000,
                   help="Points plotted in the t-SNE panels (a further subsample of the "
                        "analyzed set; t-SNE does not scale to the full sample).")
    p.add_argument("--no_tsne", action="store_true", help="Skip the t-SNE plots.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    hidden_names = [h.strip() for h in args.hidden.split(",") if h.strip()]
    unknown = [h for h in hidden_names if h not in HIDDEN_COLUMNS]
    if unknown:
        raise SystemExit(f"--hidden: unknown block(s) {unknown}; choose from "
                         f"{sorted(HIDDEN_COLUMNS)}")
    ks = sorted(int(k) for k in args.ks.split(","))
    os.makedirs(args.out, exist_ok=True)

    adv_d = _load_advantage(args.advantage)
    idx = _sample_indices(adv_d, args.n_samples, args.seed, args.include_null)
    uid, ridx = adv_d["episode_uid"][idx], adv_d["replan_index"][idx]
    advantage = adv_d["advantage"][idx].astype(np.float32)
    family = adv_d["family_code"][idx]
    boot = adv_d["bootstrapped"][idx] if "bootstrapped" in adv_d else np.zeros(idx.size, bool)
    print(f"[alias] analyzing {idx.size:,} windows "
          f"({100 * boot.mean():.1f}% bootstrapped on the terminal outcome)")
    print(f"[alias] advantage: mean {advantage.mean():+.4f}  sd {advantage.std():.4f}")

    hidden_cols = sorted({c for h in hidden_names for c in HIDDEN_COLUMNS[h]})
    state_cols = [c.strip() for c in args.state_blocks.split(",") if c.strip()]
    unknown = [c for c in state_cols if c not in PARTIAL_COLUMNS]
    if unknown:
        raise SystemExit(f"--state_blocks has unknown block(s) {unknown}; "
                         f"choose from {list(PARTIAL_COLUMNS)}")
    if not state_cols and args.no_action:
        raise SystemExit("--state_blocks is empty and --no_action is set, so z_partial would "
                         "have no columns at all.")
    if set(state_cols) != set(PARTIAL_COLUMNS):
        print(f"[alias] z_partial state blocks: {state_cols} "
              f"(dropped {sorted(set(PARTIAL_COLUMNS) - set(state_cols))})")
    W = _fetch_rows(args.rollout, uid, ridx, state_cols + hidden_cols + ["fm_plan"])

    raw_dims = {c: int(np.prod(W[c].shape[1:])) for c in state_cols}
    print(f"[alias] raw dims per source: {raw_dims}  -- each embedded to rank "
          f"{args.block_components} separately before concatenation, so none of them buys "
          "extra say in the shared space by sheer sensor width")
    partial_blocks = [_block_embed(_flatten(W[c]), args.block_components, args.seed,
                                   name=c)
                      for c in state_cols]

    action_raw_dim = 0
    if args.no_action:
        action_emb = np.zeros((idx.size, 0), dtype=np.float32)
    else:
        action_raw = _action_features(W["fm_plan"], args.action_nodes)
        action_raw_dim = int(action_raw.shape[1])
        action_emb = _block_embed(action_raw, args.block_components, args.seed,
                                  name="action")

    hidden_raw_dim = 0
    if hidden_names:
        hidden_raw = np.concatenate([_hidden_block_raw(h, W) for h in hidden_names], axis=1)
        hidden_raw_dim = int(hidden_raw.shape[1])
        hidden_emb = _block_embed(hidden_raw, args.block_components, args.seed,
                                  name="+".join(hidden_names))
    else:
        hidden_emb = np.zeros((idx.size, 0), dtype=np.float32)
    # Shuffling the EMBEDDED rows is equivalent to shuffling the raw ones and re-fitting PCA --
    # a row permutation leaves the covariance matrix, and so the fitted components, unchanged.
    rng = np.random.default_rng(args.seed)
    hidden_emb_shuf = hidden_emb[rng.permutation(hidden_emb.shape[0])]

    print(f"[alias] embedded dims: partial={sum(b.shape[1] for b in partial_blocks)} "
          f"({len(state_cols)} x {args.block_components})  action={action_emb.shape[1]}"
          f"{' (excluded, raw was ' + str(action_raw_dim) + ')' if args.no_action else ''}  "
          f"hidden({'+'.join(hidden_names) or 'none'})={hidden_emb.shape[1]} "
          f"(raw {hidden_raw_dim})")

    p_partial = np.concatenate(partial_blocks + [action_emb], axis=1)
    p_full = np.concatenate(partial_blocks + [hidden_emb, action_emb], axis=1)
    p_full_shuf = np.concatenate(partial_blocks + [hidden_emb_shuf, action_emb], axis=1)

    print(f"[alias] kNN local stats over k={ks}")
    stats_partial = _local_stats(p_partial, advantage, ks, args.seed)
    stats_full = _local_stats(p_full, advantage, ks, args.seed)
    stats_shuf = _local_stats(p_full_shuf, advantage, ks, args.seed)

    print(f"\n[alias] hidden = {'+'.join(hidden_names)}   "
          f"(real drop should CLEAR the shuffle-control drop)")
    print(f"{'k':>4}  {'var(partial)':>13}  {'var(full)':>10}  {'var(shuf)':>10}  "
          f"{'ratio p/f':>10}  {'ratio p/shuf':>12}  || "
          f"{'sign(partial)':>13}  {'sign(full)':>10}  {'sign(shuf)':>10}")
    rows = []
    for k in ks:
        vp, vf, vs = stats_partial[k]["var"], stats_full[k]["var"], stats_shuf[k]["var"]
        dp, df, ds = (stats_partial[k]["sign_disagree"], stats_full[k]["sign_disagree"],
                     stats_shuf[k]["sign_disagree"])
        r_pf = vp / vf if vf > 0 else float("nan")
        r_ps = vp / vs if vs > 0 else float("nan")
        print(f"{k:>4}  {vp:>13.5f}  {vf:>10.5f}  {vs:>10.5f}  {r_pf:>10.3f}  {r_ps:>12.3f}  || "
              f"{dp:>13.3f}  {df:>10.3f}  {ds:>10.3f}")
        rows.append({"k": k, "var_partial": vp, "var_full": vf, "var_full_shuffled": vs,
                     "ratio_partial_over_full": r_pf, "ratio_partial_over_shuffled": r_ps,
                     "sign_disagree_partial": dp, "sign_disagree_full": df,
                     "sign_disagree_shuffled": ds})

    real_drop = np.mean([r["var_partial"] - r["var_full"] for r in rows])
    null_drop = np.mean([r["var_partial"] - r["var_full_shuffled"] for r in rows])
    # `null_drop` is signed, not a floor: a shuffle-control that makes local variance WORSE
    # (negative drop, pure noise diluting the neighbourhood) is a valid outcome and must not
    # be clamped toward zero, or any tiny positive real_drop would spuriously "clear" it.
    # Clearing the null means beating it by a margin that is itself a real fraction of the
    # base variance -- not merely being on the correct side of a number near machine epsilon.
    var_partial_mean = float(np.mean([r["var_partial"] for r in rows]))
    margin = real_drop - null_drop
    verdict = ("hidden state carries real advantage signal (real drop clears the shuffle null "
              f"by {margin:+.5f}, {100 * margin / var_partial_mean:.1f}% of the base variance)"
              if margin > 0.02 * var_partial_mean else
              "no clear evidence of hidden-state-dependent advantage past the dimensionality "
              "artifact -- do not add proprioception on this basis alone")
    print(f"\n[alias] mean local-variance drop, partial->full: {real_drop:+.5f}   "
          f"shuffle-control drop: {null_drop:+.5f}")
    print(f"[alias] VERDICT: {verdict}")

    summary = {"hidden": hidden_names, "n_samples": int(idx.size), "ks": ks,
              "block_components": args.block_components, "action_nodes": args.action_nodes,
              "no_action": bool(args.no_action),
              "raw_dims": {**raw_dims, "action": action_raw_dim, "hidden": hidden_raw_dim},
              "embedded_dims": {"partial": int(sum(b.shape[1] for b in partial_blocks)),
                                "action": int(action_emb.shape[1]),
                                "hidden": int(hidden_emb.shape[1])},
              "by_k": rows, "mean_var_drop_partial_to_full": float(real_drop),
              "mean_var_drop_shuffle_control": float(null_drop), "verdict": verdict}
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[alias] wrote {args.out}/summary.json")

    _plot_ratio_curves(rows, hidden_names, args.out, args.no_action)
    if not args.no_tsne:
        _plot_tsne(p_partial, advantage, p_full, args, ks)
    return 0


def _plot_ratio_curves(rows: list[dict], hidden_names: list[str], out: str,
                       no_action: bool) -> None:
    """Variance and sign-disagreement ratios vs ``k``, real vs shuffle-control.

    Args:
        rows: Per-``k`` records from :func:`main`.
        hidden_names: The hidden blocks under test, for the title.
        out: Output directory.
        no_action: Whether the action block was excluded, for the title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(ks, [r["var_partial"] for r in rows], "o-", label="partial", color="#4477AA")
    ax.plot(ks, [r["var_full"] for r in rows], "o-", label="full", color="#CC3311")
    ax.plot(ks, [r["var_full_shuffled"] for r in rows], "o--", label="full, hidden shuffled",
            color="#CC3311", alpha=0.45)
    ax.set_xlabel("k")
    ax.set_ylabel("local variance of advantage")
    ax.set_title("Local advantage variance by neighbourhood space")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(ks, [r["sign_disagree_partial"] for r in rows], "o-", label="partial",
            color="#4477AA")
    ax.plot(ks, [r["sign_disagree_full"] for r in rows], "o-", label="full", color="#CC3311")
    ax.plot(ks, [r["sign_disagree_shuffled"] for r in rows], "o--",
            label="full, hidden shuffled", color="#CC3311", alpha=0.45)
    ax.set_xlabel("k")
    ax.set_ylabel("neighbour sign-disagreement rate")
    ax.set_title("Advantage sign disagreement by neighbourhood space")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"hidden = {'+'.join(hidden_names)}" + ("  (state only, no action)"
                                                        if no_action else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ratio_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[alias] wrote {out}/ratio_curves.png")


def _plot_tsne(p_partial: np.ndarray, advantage: np.ndarray, p_full: np.ndarray,
              args: argparse.Namespace, ks: list[int]) -> None:
    """Embed BOTH spaces, and colour them so the privileged blocks' effect is readable.

    One embedding does double duty: coloured by advantage, it shows where the outcome is good
    or bad; coloured by the FULL-space sign-disagreement rate at a representative ``k``, it
    shows where the generator's own view aliases states with opposite advantage. A region that
    is a solid colour in the first panel and hot in the second is a region where the generator
    cannot tell two different answers apart -- the exemplar worth pulling raw states for.

    Args:
        p_partial: ``[N, d]`` PCA-reduced partial-space features -- what gets embedded.
        advantage: ``[N]`` advantage values -- the first colouring.
        p_full: ``[N, d]`` PCA-reduced full-space features -- where the disagreement rate for
            the second colouring is computed, since aliasing is a property of what the FULL
            state distinguishes and the partial one does not.
        args: Parsed CLI args (``tsne_n``, ``seed``, ``out``).
        ks: Neighbourhood sizes; the median one is used for the disagreement colouring.
    """
    from sklearn.manifold import TSNE  # noqa: PLC0415

    n = p_partial.shape[0]
    take = min(args.tsne_n, n)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
    k_rep = int(np.median(ks))

    disagree = _local_stats(p_full[sel], advantage[sel], [k_rep], args.seed)[k_rep]
    # `_local_stats` returns an aggregate; re-derive the PER-POINT rate for colouring.
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
    nn = NearestNeighbors(n_neighbors=k_rep + 1, n_jobs=-1).fit(p_full[sel])
    _, neigh = nn.kneighbors(p_full[sel])
    neigh = neigh[:, 1:]
    per_point_disagree = (np.sign(advantage[sel])[neigh] != np.sign(advantage[sel])[:, None]
                          ).mean(axis=1)
    print(f"[alias] t-SNE over {take:,} points (mean k={k_rep} full-space disagreement "
          f"{disagree['sign_disagree']:.3f})")

    # BOTH spaces are embedded, because the question is comparative: adding the critic's
    # privileged blocks either pulls the colours apart or it does not, and only the two
    # pictures side by side answer that.
    #
    # PCA is the primary projection. Deterministic, and it reports the variance it explains --
    # so a 2-D picture that represents almost none of the data says so out loud instead of
    # looking like a map, which t-SNE never does.
    from sklearn.decomposition import PCA  # noqa: PLC0415
    perp = min(30, max(5, take // 100))
    spaces = {"z_partial": p_partial, "z_full": p_full}
    pemb, emb, evr = {}, {}, {}
    for zname, z in spaces.items():
        # FIT on the central bulk, PROJECT everything. A single window can carry a whole
        # feature direction on its own -- `cmd_future` has one that is ~0 on every other row
        # -- and PCA maximizes variance, so an unrobust fit points its first component at
        # that window and the remaining 7,999 collapse to a line. Outliers are still drawn,
        # at the edge, where they belong; they just no longer choose the axes.
        zz = z[sel]
        r = np.linalg.norm(zz - np.median(zz, axis=0), axis=1)
        bulk = r <= np.percentile(r, 99.5)
        pca = PCA(n_components=2, random_state=args.seed).fit(zz[bulk])
        pemb[zname] = pca.transform(zz)
        evr[zname] = pca.explained_variance_ratio_
        if not bulk.all():
            print(f"[alias] {zname}: PCA fitted on the central {100 * bulk.mean():.1f}% "
                  f"({int((~bulk).sum())} points excluded from the FIT, all still plotted)")
        print(f"[alias] PCA(2) over {zname} ({z.shape[1]}d) explains "
              f"{100 * evr[zname].sum():.1f}% of its variance "
              f"({100 * evr[zname][0]:.1f} + {100 * evr[zname][1]:.1f})")
        emb[zname] = TSNE(n_components=2, init="pca", random_state=args.seed,
                          perplexity=perp).fit_transform(z[sel])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    tag = " (state only, no action)" if args.no_action else ""
    hid = "+".join(args.hidden.split(",")) if args.hidden else "none"
    # ONE colour scale across every panel. Per-panel autoscaling would make the two spaces
    # look different when only their spreads differ, which is the exact confusion this figure
    # exists to resolve.
    vmax = float(np.percentile(np.abs(advantage[sel]), 95)) or 1.0
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.4))

    for row, (xy, proj) in enumerate(((pemb, "PCA"), (emb, "t-SNE"))):
        for col, zname in enumerate(("z_partial", "z_full")):
            frac = f" [{100 * evr[zname].sum():.1f}% var]" if proj == "PCA" else ""
            sc = axes[row][col].scatter(xy[zname][:, 0], xy[zname][:, 1], c=advantage[sel],
                                        cmap="coolwarm", vmin=-vmax, vmax=vmax, s=4,
                                        linewidths=0)
            what = "generator's own view" if zname == "z_partial" else f"+ privileged {hid}"
            axes[row][col].set_title(f"{proj}({zname}){frac}\n{what}, coloured by ADVANTAGE"
                                     f"{tag}", fontsize=10)
            fig.colorbar(sc, ax=axes[row][col], shrink=0.85)
        sc1 = axes[row][2].scatter(xy["z_full"][:, 0], xy["z_full"][:, 1],
                                   c=per_point_disagree, cmap="viridis", vmin=0, vmax=1, s=4,
                                   linewidths=0)
        axes[row][2].set_title(f"{proj}(z_full), coloured by\nk={k_rep} sign DISAGREEMENT",
                               fontsize=10)
        fig.colorbar(sc1, ax=axes[row][2], shrink=0.85)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "tsne_partial_space.png"), dpi=150)
    plt.close(fig)

    # The embeddings are saved so a DIFFERENT colouring -- family, tile, speed, episode
    # outcome -- costs a scatter call rather than another PCA and t-SNE over 60k windows.
    # Which matters here: the first thing to ask of a cluster is whether it is terrain rather
    # than advantage.
    np.savez_compressed(
        os.path.join(args.out, "embedding.npz"),
        pca_partial=pemb["z_partial"].astype(np.float32),
        pca_full=pemb["z_full"].astype(np.float32),
        tsne_partial=emb["z_partial"].astype(np.float32),
        tsne_full=emb["z_full"].astype(np.float32),
        advantage=advantage[sel].astype(np.float32),
        disagree=per_point_disagree.astype(np.float32),
        rows=sel.astype(np.int64),
        evr_partial=evr["z_partial"].astype(np.float64),
        evr_full=evr["z_full"].astype(np.float64))
    print(f"[alias] wrote {args.out}/tsne_partial_space.png")


if __name__ == "__main__":
    raise SystemExit(main())
