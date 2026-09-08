# Copyright 2026 Zachary Olkin. All rights reserved.

"""Log one AWR iteration's headline numbers to the loop-level wandb run.

The critic and the fine-tune each open their OWN wandb run, joined into one group by
``WANDB_RUN_GROUP`` (both scripts pass no ``group=`` to ``init``, so the environment is
honoured -- no change was needed in either).  Those runs carry per-EPOCH detail.  What they
cannot carry is the series across iterations, because each one is a different run: so this
opens a third run, one per loop, keyed on ``iteration``, and appends a row per iteration by
reading the artifacts the stages already wrote.

Nothing here may take down a loop.  Telemetry failure prints a warning and exits 0 -- the
run's real outputs are the checkpoints and the manifests on disk, and this repo already treats
wandb that way in ``fit_value`` and ``train``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib


def _read(path: str) -> dict:
    """Read a JSON file, returning ``{}`` when it is absent or malformed.

    Args:
        path: The file.

    Returns:
        The parsed object, or an empty dict.
    """
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _critic_quality(path: str) -> dict:
    """Recompute the quality head's out-of-fold metrics from the saved predictions.

    Reported on UNSEEN tiles, which is the only split that ever tracked outcome, and
    alongside the dispersion ratio ``pred_sd / target_sd`` -- a ratio above 1 inflates every
    advantage the fine-tune then weights by, so it is the number to watch across iterations.

    Args:
        path: The critic's ``predictions.npz``.

    Returns:
        Metrics keyed for wandb, or ``{}`` if the file is absent or carries no quality head.
    """
    try:
        import numpy as np  # noqa: PLC0415

        with np.load(path, allow_pickle=False) as z:
            if "q_V_priv" not in z.files or "g" not in z.files:
                return {}
            pred = z["q_V_priv"].astype(np.float64)
            targ = z["g"].astype(np.float64)
            ok = z["g_valid"].astype(bool) if "g_valid" in z.files else np.ones(pred.size, bool)
            unseen = z["val_tile"].astype(bool) if "val_tile" in z.files else np.zeros(pred.size, bool)
    except (OSError, ValueError, KeyError):
        return {}

    out: dict = {}
    for tag, mask in (("unseen", ok & unseen), ("seen", ok & ~unseen)):
        if int(mask.sum()) < 8:
            continue
        pm, tm = pred[mask], targ[mask]
        var = float(((tm - tm.mean()) ** 2).mean())
        out[f"critic/{tag}/r2"] = (1.0 - float(((pm - tm) ** 2).mean()) / var) if var > 0 else 0.0
        out[f"critic/{tag}/pred_sd"] = float(pm.std())
        out[f"critic/{tag}/target_sd"] = float(tm.std())
        if tm.std() > 0:
            out[f"critic/{tag}/dispersion_ratio"] = float(pm.std() / tm.std())
        if pm.std() > 0 and tm.std() > 0:
            # Spearman: rank-correlate without pulling in scipy.
            rp = np.argsort(np.argsort(pm)).astype(np.float64)
            rt = np.argsort(np.argsort(tm)).astype(np.float64)
            out[f"critic/{tag}/rho"] = float(np.corrcoef(rp, rt)[0, 1])
        out[f"critic/{tag}/n"] = int(mask.sum())
    return out


def collect_row(root: str, iteration: int, critic_dir: str, ft_dir: str) -> dict:
    """Gather one iteration's metrics from the artifacts on disk.

    Args:
        root: Run root holding ``iter_<NNN>/``.
        iteration: 1-based iteration number.
        critic_dir: The critic's output directory (holding ``gates.json``).
        ft_dir: The fine-tune's output directory (holding ``finetune.json``).

    Returns:
        A flat metric dict, keyed for wandb.
    """
    man = _read(os.path.join(root, f"iter_{iteration:03d}", "buffer.json"))
    gates = _read(os.path.join(critic_dir, "gates.json"))
    ft = _read(os.path.join(ft_dir, "finetune.json"))

    row: dict = {"iteration": iteration}
    if man:
        row["buffer/n_episodes"] = man.get("n_episodes")
        row["buffer/n_windows"] = man.get("n_windows")
        row["buffer/n_stems"] = len(man.get("stems", []))
        row["buffer/n_dropped"] = man.get("drop", {}).get("n", 0)
        # How mixed the buffer is: episodes from the newest increment over the whole buffer.
        newest = max((s.get("iteration_collected", 0) for s in man.get("stems", [])),
                     default=0)
        fresh = sum(len(s["uids"]) for s in man.get("stems", [])
                    if s.get("iteration_collected") == newest)
        if man.get("n_episodes"):
            row["buffer/fresh_fraction"] = fresh / float(man["n_episodes"])

    # **COMPUTED FROM `predictions.npz`, NOT SCRAPED FROM THE LOG.** Under `--quality_only`
    # the critic's `gates.json` `results` block carries only `quality_only: true` -- the AUC
    # and calibration entries are null because there is no failure head -- and the numbers
    # that matter (unseen-tile r2/rho, and the dispersion ratio that says whether the head
    # inflates advantages) are printed to stdout and nowhere else. The out-of-fold
    # predictions ARE saved, so recomputing is both possible and steadier than parsing text.
    if gates:
        row["critic/n_episodes"] = gates.get("n_episodes")
        row["critic/n_windows"] = gates.get("n_windows")
    row.update(_critic_quality(os.path.join(critic_dir, "predictions.npz")))

    if ft:
        awr = ft.get("awr") or {}
        for key, val in awr.items():
            if isinstance(val, (int, float)):
                row[f"awr/{key}"] = val
        hist = ft.get("history") or []
        if hist:
            last = hist[-1]
            for key, val in last.items():
                if isinstance(val, (int, float)):
                    row[f"ft/final/{key}"] = val
            row["ft/epochs"] = len(hist)
    return row


def main() -> int:
    """Parse arguments and append one row to the loop run.

    Returns:
        0 always -- telemetry must never fail a loop.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--critic_dir", required=True)
    p.add_argument("--ft_dir", required=True)
    p.add_argument("--project", default="robot_rl_awr")
    p.add_argument("--run_id", required=True, help="stable id so every iteration appends to "
                                                   "ONE run rather than making a new one")
    p.add_argument("--run_name", default="")
    p.add_argument("--group", default="")
    p.add_argument("--dry_run", action="store_true", help="print the row, log nothing")
    a = p.parse_args()

    try:
        row = collect_row(a.root, a.iteration, a.critic_dir, a.ft_dir)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[awr-log] WARNING could not gather metrics: {type(exc).__name__}: {exc}")
        return 0
    print(f"[awr-log] iteration {a.iteration}: {len(row) - 1} metrics")
    for k in sorted(row):
        print(f"    {k} = {row[k]}")
    if a.dry_run:
        return 0

    try:
        import wandb  # noqa: PLC0415

        run = wandb.init(project=a.project, id=a.run_id, resume="allow",
                        name=a.run_name or a.run_id, group=a.group or None,
                        job_type="loop")
        # Every series is against `iteration`, not wandb's auto-incrementing `_step`: the
        # rows arrive one per process, so an implicit step would number them by call order
        # and a resumed loop would restart at 0.
        wandb.define_metric("iteration")
        wandb.define_metric("buffer/*", step_metric="iteration")
        wandb.define_metric("critic/*", step_metric="iteration")
        wandb.define_metric("awr/*", step_metric="iteration")
        wandb.define_metric("ft/*", step_metric="iteration")
        run.log(row)
        run.finish()
        print(f"[awr-log] appended to {a.project}/{a.run_id}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[awr-log] WARNING wandb logging skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
