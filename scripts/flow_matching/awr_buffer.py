# Copyright 2026 Zachary Olkin. All rights reserved.

"""The AWR loop's fixed-size episode replay buffer.

The loop collects a fresh increment of episodes with the CURRENT policy each iteration, drops a
random equal-sized slice of what it already holds, and refits the critic on the result -- so the
advantage is re-estimated against a policy that has moved.  ``fit_value.py`` and
``finetune_generator.py`` each take ONE file, so the buffer has to be materialised as a single
merged stem every iteration; this module owns the bookkeeping that makes that safe.

**Increments are frozen after adoption; the buffer is derived.**  Each collection lands in
its own ``inc_<NNN>/``, is rewritten exactly once -- the uid rebase below -- and is then
never touched again, so the buffer is always rebuildable from the increments plus a manifest,
a killed merge can destroy nothing, and the expensive GPU grade runs once per episode ever
rather than once per iteration.  One increment is also exactly one policy, which makes
provenance structural rather than recorded.

**Only UNCENSORED episodes are ever named.**  ``record_rollouts`` ends by flushing every env
still in flight, so an episode table always carries exactly ``num_envs`` extra rows marked
``censored``, and ``fit_value`` drops their windows.  Counting them would make the buffer's
capacity claim false and let a collection that harvested nothing satisfy its episode budget.

**The uid space is GLOBAL, rebased once at adoption, and that is the load-bearing part.**
``merge_rollouts`` normally renumbers each stem past a running maximum -- but that offset is
derived from ``max(uid)+1`` computed AFTER any cut, so under a random drop it depends on WHICH
episodes were removed: an episode's uid would change every iteration and a dropped uid would be
REUSED by a fresh episode.  Every join downstream is ``(episode_uid, replan_index)`` and
``fit_value._reward_sidecar`` refuses only at ZERO coverage, so a 60%-wrong join would print a
coverage percentage and train.  So an increment's uids are shifted once, to
``local + iteration * UID_STRIDE``, in both tables and its reward sidecar, and then frozen;
``merge_rollouts --keep_manifest`` switches renumbering off entirely and its existing collision
check becomes the assertion that the global space really is collision-free.  A useful
consequence: ``uid // UID_STRIDE`` IS the generating iteration, so per-episode policy
provenance survives even losing the manifest.

Subcommands, in the order the loop runs them, each idempotent so a killed run resumes::

    grade   (inspect_terrain_reward.py --stage reward, outside this module)
    adopt        rebase this increment's uids and freeze it
    plan         draw the drop, write iter_<NNN>/buffer.json  <-- the COMMIT POINT
    materialise  merge the named stems into one buffer stem
    verify       the orchestrator's stage guard

``compact`` is a manual extra for long runs, rewriting an old increment down to its survivors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

import h5py
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_RSL = str(_REPO_ROOT / "scripts" / "rsl_rl")
if _RSL not in sys.path:
    sys.path.insert(0, _RSL)

import merge_rollouts as mr  # noqa: E402
import rollout_outcomes as ro  # noqa: E402

#: Uid shift per iteration.  Large enough that no iteration can collide with the next, small
#: enough that ``(uid << 20) + replan_index`` -- the key `fit_value._reward_sidecar` and
#: `rollout_dataset._join_positions` both build -- stays far inside int64: at 10 iterations
#: that key reaches ~1.0e13 against a 9.2e18 ceiling.
UID_STRIDE = 1_000_000

#: Sidecar meta fields that legitimately differ between increments.  EVERYTHING else must
#: agree: `fit_value._reward_sidecar` reads ONE scalar `meta_json`/`reward_hash` for the whole
#: file and `label_advantage.check_reward_consistency` gates on it, so two graders with
#: different settings would silently become one quantity.
_SIDECAR_PER_STEM_FIELDS: frozenset[str] = frozenset({"rollout", "n_windows"})

#: Recording attributes that must not move across the loop's iterations.  `_CONTRACT_ATTRS` in
#: `merge_rollouts` deliberately omits these -- an IC sweep varies the tracker and the library
#: on purpose -- but for AWR a changed `tracking_spec` makes the critic's normalized return a
#: different quantity than `label_advantage`'s return, silently.
_LOOP_CONTRACT_ATTRS: tuple[str, ...] = (
    "tracking_spec", "history_from", "clip_library_path",
)

SIDECAR_NAME = "terrain_reward_sidecar.npz"


def inc_dir(root: str, iteration: int) -> str:
    """Path of one iteration's collection directory.

    Args:
        root: The run root.
        iteration: 1-based iteration number.

    Returns:
        ``<root>/inc_<NNN>``.
    """
    return os.path.join(root, f"inc_{iteration:03d}")


def iter_dir(root: str, iteration: int) -> str:
    """Path of one iteration's buffer directory.

    Args:
        root: The run root.
        iteration: 1-based iteration number.

    Returns:
        ``<root>/iter_<NNN>``.
    """
    return os.path.join(root, f"iter_{iteration:03d}")


def resolve_stem(directory: str) -> str:
    """Find the single rollout stem in a collection directory.

    Args:
        directory: A directory holding exactly one ``*.episodes.h5``.

    Returns:
        The stem, without any suffix.

    Raises:
        SystemExit: On none or several -- picking the newest would reinstate, as a silent
            default, the ambiguity that naming removes.
    """
    found = sorted(pathlib.Path(directory).rglob("*.episodes.h5"))
    if len(found) != 1:
        names = [str(f) for f in found]
        raise SystemExit(
            f"{directory}: expected exactly one '*.episodes.h5', found {len(found)}: {names}. "
            "A collection directory holds one recording; refusing to guess which.")
    return str(found[0])[: -len(".episodes.h5")]


def _read_attrs(path: str) -> dict:
    """Read an HDF5 file's root attributes, JSON-decoding where possible.

    Args:
        path: The file.

    Returns:
        The attribute dict.
    """
    out: dict = {}
    with h5py.File(path, "r") as f:
        for k, raw in f.attrs.items():
            if isinstance(raw, np.ndarray):
                raw = raw.tolist()
            if isinstance(raw, (np.integer, np.floating)):
                raw = raw.item()
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (TypeError, ValueError):
                    pass
            out[k] = raw
    return out


def sidecar_path(stem: str) -> str:
    """Where the terrain-reward sidecar for a stem lives.

    Args:
        stem: The rollout stem.

    Returns:
        ``<stem's directory>/terrain_reward_sidecar.npz``.
    """
    return os.path.join(os.path.dirname(os.path.abspath(stem)), SIDECAR_NAME)


def rebase_uids(stem: str, uid_base: int, rounds: int = 0) -> tuple[int, list[int]]:
    """Shift a frozen increment's episode uids into the global space, IN PLACE.

    Rewrites ``episode_uid`` in BOTH tables and in the reward sidecar, so every
    ``(episode_uid, replan_index)`` join downstream keeps pairing the same rows.  The sidecar
    is rewritten wholesale (an ``.npz`` is a zip, not a mutable array) and its hash recomputed
    exactly as ``inspect_terrain_reward`` computes it -- hygiene, not a gate:
    ``label_advantage.check_reward_consistency`` gates on ``reward_source`` and prints the
    hash without comparing it, so a stale hash would mislead a reader rather than fail.

    Args:
        stem: The rollout stem, already graded.
        uid_base: Value added to every uid.
        rounds: Episodes to keep per env; see :func:`first_per_env`.

    Returns:
        ``(n_rows, usable_uids)`` -- every episode row in the table, and the shifted uids of
        the ones that are NOT censored.  Those differ: see :func:`cmd_adopt`.

    Raises:
        SystemExit: If the sidecar is missing -- the grade must happen BEFORE adoption, or the
            sidecar would keep local uids while the tables moved, and the join would silently
            cover nothing.
    """
    side = sidecar_path(stem)
    if not os.path.isfile(side):
        raise SystemExit(
            f"{side} not found. Grade the increment BEFORE adopting it: adoption shifts the "
            "tables' uids, and a sidecar written afterwards against the shifted tables would "
            "be fine, but one written before and not shifted would join to nothing.")

    # PER-ARTIFACT idempotency, not just the adopt.json marker. This shifts three things and
    # the marker is written last, so a crash in between left a partially shifted increment
    # that a resume shifted AGAIN -- detected downstream (the coverage gate refuses, or the
    # merge dies inside h5py naming nothing) but only recoverable by hand. Each file now
    # carries its own answer to "has this been shifted?", stamped inside the same handle that
    # does the shifting, so a resume finishes the half-done work instead of repeating it.
    for suffix in (".h5", ".episodes.h5"):
        with h5py.File(stem + suffix, "r+") as f:
            stamped = int(f.attrs.get("awr_uid_base", -1))
            if stamped == int(uid_base):
                continue
            if stamped != -1:
                raise SystemExit(
                    f"{stem + suffix} is already rebased to {stamped}, not {uid_base}: two "
                    "iterations have claimed this increment. Delete it and re-collect.")
            local = f["episode_uid"][...].astype(np.int64)
            # THE PRECONDITION THE WHOLE BAND SCHEME RESTS ON. A raw recording numbers from 0
            # (`rollout_outcomes.RolloutEpisodeTracker`), so a local uid at or above the
            # stride means either this file was already shifted by something that did not
            # stamp it, or a recording somehow exceeded a million episodes. Either way the
            # bands would overlap and two different episodes would share a key.
            if local.size and int(local.max()) >= UID_STRIDE:
                raise SystemExit(
                    f"{stem + suffix}: local episode_uid reaches {int(local.max())}, at or "
                    f"above UID_STRIDE ({UID_STRIDE}). Shifting it would overlap the next "
                    "iteration's band and two episodes would share a join key. Refusing.")
            f["episode_uid"][...] = local + np.int64(uid_base)
            f.attrs["awr_uid_base"] = int(uid_base)

    with np.load(side, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    meta = json.loads(str(payload["meta_json"]))
    if int(meta.get("uid_base", -1)) != int(uid_base):
        payload["episode_uid"] = payload["episode_uid"].astype(np.int64) + np.int64(uid_base)
    meta["uid_base"] = int(uid_base)
    payload["meta_json"] = json.dumps(meta, sort_keys=True)
    payload["reward_hash"] = hashlib.sha256(
        payload["episode_uid"].tobytes() + payload["replan_index"].tobytes()
        + payload["reward"].tobytes() + payload["meta_json"].encode()).hexdigest()[:16]
    tmp = side + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, side)

    with h5py.File(stem + ".episodes.h5", "r") as f:
        uids = f["episode_uid"][...].astype(np.int64)
        censored = (f["censored"][...].astype(bool) if "censored" in f
                    else np.zeros(uids.size, dtype=bool))
        env_id = (f["env_id"][...].astype(np.int64) if "env_id" in f
                  else np.zeros(uids.size, dtype=np.int64))
    if uids.size and int(uids.max()) - uid_base >= UID_STRIDE:
        raise SystemExit(
            f"{stem}: uids span {int(uids.min())}..{int(uids.max())}, wider than one "
            f"UID_STRIDE from base {uid_base}. The bands overlap. Refusing.")
    return int(uids.size), first_per_env(uids[~censored], env_id[~censored], rounds)


def first_per_env(uids: np.ndarray, env_id: np.ndarray, rounds: int) -> list[int]:
    """Keep the first ``rounds`` completed episodes of each env, discarding later ones.

    **The extras are not free data -- keeping them would reintroduce the bias the round-based
    stop removes.** A rollout stops when the SLOWEST env finishes its k-th episode, by which
    time the fast envs have finished more; and an env finishes more episodes precisely when
    its episodes were short. So the surplus is drawn from the short tail, and keeping it
    would tilt the sample back the way it came, just less far. Taking exactly the first
    ``rounds`` per env leaves a complete design over ``(env, round)``, whose membership
    cannot depend on how long any episode took.

    A uid is issued when an episode ENDS, so ascending uid within an env is completion order.

    Args:
        uids: ``[n]`` episode uids of the naturally-completed episodes.
        env_id: ``[n]`` the env each one ran in.
        rounds: Episodes to keep per env; ``0`` keeps everything (the total-count stop's
            behaviour, retained for a recording made without rounds).

    Returns:
        The retained uids, sorted.
    """
    if rounds <= 0 or uids.size == 0:
        return sorted(int(u) for u in uids)
    order = np.argsort(uids, kind="stable")
    keep: list[int] = []
    seen: dict[int, int] = {}
    for i in order:
        e = int(env_id[i])
        n = seen.get(e, 0)
        if n < rounds:
            keep.append(int(uids[i]))
            seen[e] = n + 1
    return sorted(keep)


def cmd_adopt(args: argparse.Namespace) -> int:
    """Freeze an increment: check it, rebase its uids, and record what it is.

    Writes ``adopt.json``, whose existence makes a second run a no-op -- rebasing twice would
    double the offset and put the increment in another iteration's uid band.

    Args:
        args: Parsed CLI arguments.

    Returns:
        0 on success.

    Raises:
        SystemExit: If the window cap truncated the collection, too few episodes were
            collected, or the recording contract moved since iteration 1.
    """
    d = inc_dir(args.root, args.iteration)
    marker = os.path.join(d, "adopt.json")
    if os.path.isfile(marker):
        rec = json.loads(pathlib.Path(marker).read_text())
        print(f"[adopt] iter {args.iteration}: already adopted "
              f"({rec['n_episodes']} episodes, uid base {rec['uid_base']}) -- no-op")
        return 0

    stem = resolve_stem(d)
    wa = _read_attrs(stem + ".h5")
    written = int(wa.get("num_windows_written", 0))
    allocated = int(wa.get("num_windows", 0))
    if written == 0:
        raise SystemExit(
            f"{stem}.h5 has num_windows_written == 0. Either the recording was killed before "
            "close() stamped it, or it truly wrote nothing; a reader would see an EMPTY "
            "table rather than an error. Re-collect.")
    # `allocated == 0` means the attribute is absent (hand-built fixtures), not that nothing
    # was allocated -- a real recording always stamps it at construction. Only a positive
    # allocation can tell us whether the cap bound.
    if allocated > 0 and written >= allocated:
        raise SystemExit(
            f"{stem}.h5 wrote {written} of {allocated} allocated windows -- the --num_windows "
            "HARD CAP bound before --num_episodes did, so this increment is TRUNCATED and "
            "holds fewer episodes than asked for. As the policy improves, episodes survive "
            "longer and make more re-plans, so a cap sized for iteration 1 binds later. "
            "Raise --num_windows and re-collect.")

    ea = _read_attrs(stem + ".episodes.h5")
    # **COUNT THE USABLE EPISODES, NOT THE ROWS.** `record_rollouts` ends by flushing every
    # env still in flight (`tracker.flush(u, arange(num_envs))`), so the episode table always
    # carries exactly `num_envs` extra rows marked `censored` -- measured: 5122 rows with
    # 1024 censored at 1024 envs, and 32307 with 2304 at 6 runs x 384. `fit_value` then DROPS
    # every censored episode's windows, so those rows are dead weight the buffer would
    # nevertheless count: the capacity claim would be false, the usable buffer would grow
    # monotonically instead of holding steady, and `--expect_episodes` would be satisfied by
    # a collection that harvested nothing at all (num_envs alone exceeds a 1000-episode
    # target). So only uncensored episodes are ever named.
    with h5py.File(stem + ".episodes.h5", "r") as f:
        n_rows = int(f["episode_uid"].shape[0])
        n_censored = (int(f["censored"][...].astype(bool).sum()) if "censored" in f else 0)
    n_usable = n_rows - n_censored
    if args.expect_episodes and n_usable < int(args.expect_episodes):
        raise SystemExit(
            f"{stem}: {n_usable} USABLE episodes ({n_rows} rows less {n_censored} censored), "
            f"expected {args.expect_episodes}. A short increment silently shrinks the buffer "
            "below its stated capacity. Re-collect with a larger --num_episodes.")

    # Contract check against iteration 1, whose adopt.json is the reference.
    ref_marker = os.path.join(inc_dir(args.root, 1), "adopt.json")
    contract = {k: ea.get(k, wa.get(k)) for k in _LOOP_CONTRACT_ATTRS}
    if args.iteration != 1 and os.path.isfile(ref_marker):
        ref = json.loads(pathlib.Path(ref_marker).read_text()).get("contract", {})
        for k in _LOOP_CONTRACT_ATTRS:
            if mr._attr_key(ref.get(k)) != mr._attr_key(contract.get(k)):
                raise SystemExit(
                    f"{stem}: {k!r} moved since iteration 1 "
                    f"({ref.get(k)!r} -> {contract.get(k)!r}). merge_rollouts deliberately "
                    "does NOT check this -- an IC sweep varies the tracker and library on "
                    "purpose -- but across AWR iterations a changed tracking spec makes the "
                    "critic's normalized return a different quantity than the advantage's, "
                    "silently. Refusing.")

    uid_base = int(args.iteration) * UID_STRIDE
    n_eps, uids = rebase_uids(stem, uid_base, int(args.rounds))
    # Trim to EXACTLY the target, at random rather than by uid order: a uid is assigned when
    # an episode ENDS, so the lowest ones are the shortest episodes, which on this course are
    # disproportionately the failures. Taking a prefix would quietly bias every buffer toward
    # them. Seeded on the iteration so the trim is reproducible from the manifest.
    if args.expect_episodes and len(uids) > int(args.expect_episodes):
        rng = np.random.default_rng([int(args.trim_seed), int(args.iteration)])
        uids = sorted(int(u) for u in rng.choice(np.asarray(uids, dtype=np.int64),
                                                 size=int(args.expect_episodes),
                                                 replace=False))
    side = sidecar_path(stem)
    with np.load(side, allow_pickle=False) as z:
        reward_hash = str(z["reward_hash"])
        side_meta = json.loads(str(z["meta_json"]))

    rec = {
        "stem": os.path.abspath(stem), "iteration": int(args.iteration),
        "uid_base": uid_base, "n_rows": n_eps, "n_censored": n_censored,
        "n_episodes": len(uids), "n_windows": written, "uids": uids,
        "fm_checkpoint": args.policy or "", "contract": contract,
        "reward_sidecar": side, "reward_hash": reward_hash,
        "reward_source": side_meta.get("reward_source", ""),
    }
    pathlib.Path(marker).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(f"[adopt] iter {args.iteration}: {len(uids)} kept of {n_usable} completed "
          f"({n_eps} rows less {n_censored} censored"
          f"{f', first {args.rounds}/env' if args.rounds else ''}), {written} windows, "
          f"uids {min(uids)}..{max(uids)} (base {uid_base}), reward hash {reward_hash}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Draw this iteration's drop and write the manifest -- the loop's COMMIT POINT.

    Once ``buffer.json`` exists it is never redrawn: a re-drawn drop after a half-written merge
    would leave a buffer that does not match the manifest every later stage reads.

    Args:
        args: Parsed CLI arguments.

    Returns:
        0 on success.

    Raises:
        SystemExit: If the previous iteration's manifest is missing, or the pool is too small
            to drop from.
    """
    out_dir = iter_dir(args.root, args.iteration)
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "buffer.json")
    if os.path.isfile(manifest):
        man = json.loads(pathlib.Path(manifest).read_text())
        print(f"[plan] iter {args.iteration}: manifest exists "
              f"({man['n_episodes']} episodes over {len(man['stems'])} stems) -- no-op")
        return 0

    adopted = json.loads(pathlib.Path(os.path.join(
        inc_dir(args.root, args.iteration), "adopt.json")).read_text())
    new_row = {"stem": adopted["stem"], "iteration_collected": adopted["iteration"],
               "uid_base": adopted["uid_base"], "fm_checkpoint": adopted["fm_checkpoint"],
               "reward_sidecar": adopted["reward_sidecar"],
               "reward_hash": adopted["reward_hash"],
               "n_collected": adopted["n_episodes"], "uids": adopted["uids"],
               "n_censored_excluded": adopted.get("n_censored", 0)}

    if args.iteration == 1:
        stems = [new_row]
        dropped: list[int] = []
    else:
        prev_path = os.path.join(iter_dir(args.root, args.iteration - 1), "buffer.json")
        if not os.path.isfile(prev_path):
            raise SystemExit(
                f"{prev_path} not found: iteration {args.iteration} cannot churn a buffer "
                "that was never committed. Run the previous iteration first.")
        prev = json.loads(pathlib.Path(prev_path).read_text())
        pool = np.array([u for row in prev["stems"] for u in row["uids"]], dtype=np.int64)
        if pool.size != int(prev["n_episodes"]):
            raise SystemExit(
                f"{prev_path}: stems hold {pool.size} uids but n_episodes says "
                f"{prev['n_episodes']}. Refusing to churn an inconsistent manifest.")
        if args.refresh > pool.size:
            raise SystemExit(
                f"cannot drop {args.refresh} of {pool.size} episodes.")
        # **OLDEST ITERATION FIRST**, not uniform over the pool. A uniform draw gives data an
        # exponential age -- a few episodes survive many iterations, so the buffer always
        # carries a thin tail collected by policies several generations stale, and how stale
        # is a matter of luck. Retiring in collection order gives a hard, stated window of
        # `capacity / refresh` iterations instead, which is the thing the buffer is supposed
        # to be: a fixed-length view of recent behaviour. Within the partially-drained stem
        # the draw is still random and seeded, so which of an iteration's episodes survives
        # longest is not decided by uid order -- a uid is issued when an episode ENDS, so a
        # prefix there would systematically retire the short ones.
        rng = np.random.default_rng([int(args.seed), int(args.iteration)])
        need = int(args.refresh)
        dropped_l: list[int] = []
        for row in sorted(prev["stems"], key=lambda r: int(r["iteration_collected"])):
            if need <= 0:
                break
            have = np.asarray(row["uids"], dtype=np.int64)
            if have.size <= need:
                dropped_l.extend(int(u) for u in have)
                need -= have.size
            else:
                dropped_l.extend(int(u) for u in rng.choice(have, size=need, replace=False))
                need = 0
        if need > 0:
            raise SystemExit(
                f"cannot drop {args.refresh} episodes: the buffer holds {pool.size} and the "
                f"per-stem walk came up {need} short, which means a stem's uid list "
                "disagrees with n_episodes.")
        dropped = sorted(dropped_l)
        drop_set = set(dropped)
        stems = []
        for row in prev["stems"]:
            kept = [u for u in row["uids"] if u not in drop_set]
            if kept:
                stems.append({**row, "uids": kept})
        stems.append(new_row)

    n_eps = sum(len(r["uids"]) for r in stems)
    if int(args.capacity) and n_eps != int(args.capacity):
        raise SystemExit(
            f"iter {args.iteration}: the churn gives {n_eps} episodes, not the stated "
            f"capacity {args.capacity}. Refusing to commit a manifest whose size claim is "
            "wrong -- every later size check reads it.")
    man = {
        "schema": 1, "iteration": int(args.iteration), "uid_space": "global",
        "uid_stride": UID_STRIDE, "capacity": int(args.capacity), "n_episodes": n_eps,
        "drop": {"policy": "random", "seed": [int(args.seed), int(args.iteration)],
                 "n": len(dropped), "uids": dropped},
        "stems": stems,
    }
    _write_manifest(manifest, man)
    print(f"[plan] iter {args.iteration}: {n_eps} episodes over {len(stems)} stem(s), "
          f"dropped {len(dropped)} -> {manifest}")
    return 0


def _write_manifest(path: str, man: dict) -> None:
    """Write a buffer manifest atomically.

    The manifest is the loop's commit point and is read by every later stage, so a truncated
    write is unrecoverable without deleting it by hand: ``plan`` for the next iteration and
    ``verify`` would both die on a ``JSONDecodeError``.  Write-then-rename makes a reader see
    either the old file or the new one.

    Args:
        path: Destination ``buffer.json``.
        man: The manifest.
    """
    tmp = path + ".tmp"
    pathlib.Path(tmp).write_text(json.dumps(man, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def concat_sidecars(rows: list[dict], out_path: str) -> dict:
    """Concatenate the increments' reward sidecars into one the buffer can be fitted against.

    Args:
        rows: The manifest's ``stems`` entries, in buffer order.
        out_path: Destination ``.npz``.

    Returns:
        The merged metadata.

    Raises:
        SystemExit: If two sidecars disagree on any reward-DEFINING field.  They would
            otherwise become one quantity silently: the consumer reads a single scalar
            ``meta_json`` for the whole file.
    """
    parts: list[dict] = []
    metas: list[dict] = []
    for row in rows:
        with np.load(row["reward_sidecar"], allow_pickle=False) as z:
            parts.append({k: z[k] for k in z.files if k not in ("meta_json", "reward_hash")})
            metas.append(json.loads(str(z["meta_json"])))
    base = metas[0]
    for row, m in zip(rows[1:], metas[1:]):
        for k in set(base) | set(m):
            if k in _SIDECAR_PER_STEM_FIELDS or k == "uid_base":
                continue
            if mr._attr_key(base.get(k)) != mr._attr_key(m.get(k)):
                raise SystemExit(
                    f"{row['reward_sidecar']}: reward field {k!r} is "
                    f"{m.get(k)!r} but the first sidecar says {base.get(k)!r}. "
                    "fit_value reads ONE meta_json for the whole file, so two graders would "
                    "silently become one quantity. Re-grade with matching settings.")
    payload = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    meta = {k: v for k, v in base.items()
            if k not in _SIDECAR_PER_STEM_FIELDS and k != "uid_base"}
    # The sidecars are concatenated WHOLE, not cut to the retained uids -- both consumers are
    # key joins (`fit_value._reward_sidecar`, `rollout_dataset._join_positions`) and never
    # look up the extra rows, so cutting them would be work for nothing. The consequence is
    # that this count is the sidecar's own length and EXCEEDS the buffer's window count
    # (measured ~2x by iteration 5); `verify_buffer` checks coverage of the buffer, which is
    # the quantity that matters. Named so nothing reads it as the buffer's size.
    meta["n_sidecar_rows"] = int(payload["episode_uid"].size)
    meta["sources"] = [{"stem": r["stem"], "reward_hash": r["reward_hash"]} for r in rows]
    payload["meta_json"] = json.dumps(meta, sort_keys=True)
    payload["reward_hash"] = hashlib.sha256(
        payload["episode_uid"].tobytes() + payload["replan_index"].tobytes()
        + payload["reward"].tobytes() + payload["meta_json"].encode()).hexdigest()[:16]
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out_path)
    return meta


def cmd_materialise(args: argparse.Namespace) -> int:
    """Merge the manifest's named episodes into one buffer stem.

    **Always merges, even when one increment holds every named episode.** Using the increment
    in place would be free, but the increment's episode table still carries the ``censored``
    rows that :func:`cmd_adopt` deliberately does not name -- so the buffer would hold
    episodes the manifest does not list, ``fit_value`` would drop their windows anyway, and
    :func:`verify_buffer`'s episode-set equality would have to special-case which file it is
    looking at. One extra copy on the first iteration buys a buffer that is exactly what the
    manifest says, on every iteration, with one code path.

    Merges into a ``.tmp`` directory and renames only once the merge has returned, because a
    killed merge leaves a window file whose ``num_windows_written`` was never stamped -- and
    ``read_rollout_windows`` FALLS BACK to ``num_windows``, so a partial table reads as a
    plausible short one rather than failing.

    Args:
        args: Parsed CLI arguments.

    Returns:
        0 on success.
    """
    out_dir = iter_dir(args.root, args.iteration)
    manifest = os.path.join(out_dir, "buffer.json")
    man = json.loads(pathlib.Path(manifest).read_text())
    rows = man["stems"]

    tmp_dir = os.path.join(out_dir, ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_stem = os.path.join(tmp_dir, "buffer")
    mr.merge([r["stem"] for r in rows], tmp_stem, keep_manifest=manifest)
    concat_sidecars(rows, tmp_stem + "_reward.npz")

    final_stem = os.path.join(out_dir, "buffer")
    for suffix in (".h5", ".episodes.h5"):
        os.replace(tmp_stem + suffix, final_stem + suffix)
    os.replace(tmp_stem + "_reward.npz", final_stem + "_reward.npz")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with h5py.File(final_stem + ".h5", "r") as f:
        man["n_windows"] = int(f.attrs["num_windows_written"])
    man["buffer"] = {"stem": final_stem, "reward_npz": final_stem + "_reward.npz"}
    _write_manifest(manifest, man)
    print(f"[materialise] iter {args.iteration}: {final_stem} ({man['n_windows']} windows)")
    return verify_buffer(args.root, args.iteration)


def verify_buffer(root: str, iteration: int, min_coverage: float = 0.999) -> int:
    """Check that a materialised buffer is the one the manifest describes.

    Cheap by construction -- attributes, the episode table and the sidecar's key columns only,
    never the window payload -- so it is usable as the orchestrator's stage guard rather than
    file existence.

    Args:
        root: The run root.
        iteration: 1-based iteration number.
        min_coverage: Least acceptable fraction of buffer windows the sidecar covers.

    Returns:
        0 if the buffer is sound.

    Raises:
        SystemExit: Naming the first disagreement.
    """
    manifest = os.path.join(iter_dir(root, iteration), "buffer.json")
    if not os.path.isfile(manifest):
        raise SystemExit(f"{manifest} not found -- nothing committed for iteration {iteration}.")
    man = json.loads(pathlib.Path(manifest).read_text())
    if "buffer" not in man:
        raise SystemExit(f"{manifest} has no 'buffer' entry -- materialise has not run.")
    stem = man["buffer"]["stem"]

    with h5py.File(stem + ".h5", "r") as f:
        written = int(f.attrs.get("num_windows_written", 0))
        n_rows = int(f["episode_uid"].shape[0])
        w_uid = f["episode_uid"][:written].astype(np.int64)
        w_ridx = f["replan_index"][:written].astype(np.int64)
    if written <= 0:
        raise SystemExit(
            f"{stem}.h5: num_windows_written is {written}. A merge killed before it stamped "
            "that attribute leaves a file every reader sees as EMPTY, with no error. "
            "Re-materialise.")
    if written != int(man.get("n_windows", -1)):
        raise SystemExit(f"{stem}.h5 holds {written} windows, manifest says "
                         f"{man.get('n_windows')}.")
    if written != n_rows:
        raise SystemExit(
            f"{stem}.h5: {written} written but {n_rows} rows present. The merge appends "
            "exactly, so a gap means it was interrupted.")

    with h5py.File(stem + ".episodes.h5", "r") as f:
        e_uid = f["episode_uid"][...].astype(np.int64)
    want = np.array(sorted(u for row in man["stems"] for u in row["uids"]), dtype=np.int64)
    if e_uid.size != want.size or not np.array_equal(np.sort(e_uid), want):
        extra = np.setdiff1d(e_uid, want)
        missing = np.setdiff1d(want, e_uid)
        raise SystemExit(
            f"{stem}: episode set does not match the manifest -- {e_uid.size} vs "
            f"{want.size} episodes, {extra.size} unexpected (e.g. {extra[:5].tolist()}), "
            f"{missing.size} missing (e.g. {missing[:5].tolist()}).")
    if int(man["capacity"]) and e_uid.size != int(man["capacity"]):
        raise SystemExit(f"{stem}: {e_uid.size} episodes against a stated capacity of "
                         f"{man['capacity']}.")

    npz = man["buffer"]["reward_npz"]
    with np.load(npz, allow_pickle=False) as z:
        key = (z["episode_uid"].astype(np.int64) << np.int64(20)) \
            + z["replan_index"].astype(np.int64)
    want_key = (w_uid << np.int64(20)) + w_ridx
    covered = float(np.isin(want_key, key).mean())
    if covered < min_coverage:
        raise SystemExit(
            f"{npz} covers {covered:.4f} of {stem}'s windows, below {min_coverage}. This is "
            "the end-to-end detector for uid-space drift: a rebase applied twice, or a "
            "sidecar graded before a rebase, lands here rather than in a critic that trains "
            "on zeros. Refusing.")
    print(f"[verify] iter {iteration}: OK -- {e_uid.size} episodes, {written} windows, "
          f"sidecar coverage {covered:.4f}, uid range {e_uid.min()}..{e_uid.max()}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a materialised buffer.

    Args:
        args: Parsed CLI arguments.

    Returns:
        0 if sound.
    """
    return verify_buffer(args.root, args.iteration, args.min_coverage)


def cmd_compact(args: argparse.Namespace) -> int:
    """Rewrite an old increment down to the episodes the current manifest still names.

    Not needed for a short run -- under a uniform drop an increment keeps ``refresh * 0.75^k``
    episodes, so nothing goes fully dead inside five iterations -- but a long loop accumulates
    increments whose retained fraction is tiny.

    Args:
        args: Parsed CLI arguments.

    Returns:
        0 on success.

    Raises:
        SystemExit: If the manifest does not name the increment.
    """
    man = json.loads(pathlib.Path(os.path.join(
        iter_dir(args.root, args.iteration), "buffer.json")).read_text())
    target = os.path.abspath(resolve_stem(inc_dir(args.root, args.increment)))
    row = next((r for r in man["stems"] if os.path.abspath(r["stem"]) == target), None)
    if row is None:
        raise SystemExit(f"{target} is not named by iteration {args.iteration}'s manifest; "
                         "it holds no live episodes and can be deleted outright.")
    sub = {"schema": 1, "uid_space": "global", "stems": [row]}
    tmp_man = target + ".compact.json"
    pathlib.Path(tmp_man).write_text(json.dumps(sub, default=str))
    out = target + ".compact"
    mr.merge([row["stem"]], out, keep_manifest=tmp_man)
    for suffix in (".h5", ".episodes.h5"):
        os.replace(out + suffix, target + suffix)
    os.unlink(tmp_man)
    print(f"[compact] {target}: kept {len(row['uids'])} episodes")
    return 0


def main() -> int:
    """Parse arguments and dispatch a subcommand.

    Returns:
        The subcommand's exit status.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="run root holding inc_*/ and iter_*/")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("adopt", help="rebase an increment's uids and freeze it")
    a.add_argument("--iteration", type=int, required=True)
    a.add_argument("--expect_episodes", type=int, default=0)
    a.add_argument("--policy", type=str, default="", help="checkpoint that collected it")
    a.add_argument("--rounds", type=int, default=0,
                   help="episodes each env was asked to complete; only the first this many "
                        "per env are kept (0 = keep every completed episode)")
    a.add_argument("--trim_seed", type=int, default=0,
                   help="seed for trimming an over-collection down to --expect_episodes")
    a.set_defaults(fn=cmd_adopt)

    b = sub.add_parser("plan", help="draw the drop and commit the manifest")
    b.add_argument("--iteration", type=int, required=True)
    b.add_argument("--capacity", type=int, required=True)
    b.add_argument("--refresh", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_plan)

    c = sub.add_parser("materialise", help="merge the named episodes into one stem")
    c.add_argument("--iteration", type=int, required=True)
    c.set_defaults(fn=cmd_materialise)

    d = sub.add_parser("verify", help="check a materialised buffer")
    d.add_argument("--iteration", type=int, required=True)
    d.add_argument("--min_coverage", type=float, default=0.999)
    d.set_defaults(fn=cmd_verify)

    e = sub.add_parser("compact", help="shrink an old increment to its survivors")
    e.add_argument("--iteration", type=int, required=True)
    e.add_argument("--increment", type=int, required=True)
    e.set_defaults(fn=cmd_compact)

    args = p.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
