# Copyright 2026 Zachary Olkin. All rights reserved.

"""Log an AWR loop into ONE wandb run per arm.

**One run, not one per stage.**  The obvious wiring -- let ``fit_value`` and
``finetune_generator`` each open their own run and join them with ``WANDB_RUN_GROUP`` -- was
tried and is wrong to use.  ``finetune_generator`` takes no ``--run_name`` and falls back to
its output directory's basename, which is ``ft`` for every iteration, so the sidebar fills
with identically-named runs whose only distinguishing feature is their colour; and the count
grows as ``2 * iterations * arms``.  Worse, each fine-tune restarts its step axis at 0, so
the thing one most wants to see -- whether the policy is still improving as the loop turns --
is split across runs that cannot be overlaid.

So the CHILD stages log nothing (the orchestrator passes ``--no_wandb``), and this script
carries the whole loop: it resumes one fixed run id per arm and appends, reading the
artifacts each stage already wrote.  Two step axes, declared so neither disturbs the other:

  * ``ft/global_epoch`` -- ``(iteration - 1) * epochs + epoch``, so the fine-tune's per-epoch
    train/val curve is CONTINUOUS across iterations and reads as one training run, which is
    what it is.
  * ``iteration`` -- the buffer's composition, the critic's out-of-fold quality, and the AWR
    weight statistics, one point per iteration.

The cost, stated because it is a real loss: the critic's PER-EPOCH curves go to its stdout
and nowhere machine-readable, so only its end-of-fit out-of-fold metrics reach wandb. That is
the right granularity for a loop view, but it is not the same as having them.

Nothing here may take down a loop.  Telemetry failure prints a warning and exits 0 -- the
run's real outputs are the checkpoints and the manifests on disk, and this repo already
treats wandb that way in ``fit_value`` and ``train``.
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


def ft_curve(ft_dir: str, iteration: int, epochs_per_iter: int) -> list[dict]:
    """One row per fine-tune EPOCH, on a step axis that continues across iterations.

    Args:
        ft_dir: The fine-tune's output directory.
        iteration: 1-based iteration number.
        epochs_per_iter: Epochs each iteration runs, used to offset the global axis.  Read
            from the artifact when it disagrees, so a changed ``--epochs`` mid-loop does not
            silently overlay two iterations on the same steps.

    Returns:
        Rows ready to log, each carrying ``ft/global_epoch``.
    """
    ft = _read(os.path.join(ft_dir, "finetune.json"))
    hist = ft.get("history") or []
    if not hist:
        return []
    per = int(ft.get("args", {}).get("epochs") or epochs_per_iter or len(hist))
    base = (int(iteration) - 1) * per
    rows: list[dict] = []
    for i, e in enumerate(hist):
        r = {"ft/global_epoch": base + i + 1, "ft/iteration": int(iteration)}
        for key, val in e.items():
            if isinstance(val, (int, float)) and key != "epoch":
                r[f"ft/{key}"] = val
        rows.append(r)
    return rows


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
    p.add_argument("--epochs_per_iter", type=int, default=0,
                   help="fine-tune epochs per iteration, for the global epoch axis; the "
                        "artifact's own value wins when it disagrees")
    p.add_argument("--dry_run", action="store_true", help="print the row, log nothing")
    a = p.parse_args()

    try:
        row = collect_row(a.root, a.iteration, a.critic_dir, a.ft_dir)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[awr-log] WARNING could not gather metrics: {type(exc).__name__}: {exc}")
        return 0
    for k in sorted(row):
        print(f"    {k} = {row[k]}")
    curve = ft_curve(a.ft_dir, a.iteration, a.epochs_per_iter)
    print(f"[awr-log] iteration {a.iteration}: {len(row) - 1} metrics, "
          f"{len(curve)} fine-tune epoch rows")
    if a.dry_run:
        for r in curve:
            print(f"    curve {r}")
        return 0

    try:
        import wandb  # noqa: PLC0415

        # ONE run per arm, resumed. `resume="allow"` creates it on the first iteration and
        # appends on every later one, so the sidebar holds one row per arm however long the
        # loop runs -- against `2 * iterations * arms` under per-stage runs.
        run = wandb.init(project=a.project, id=a.run_id, resume="allow",
                        name=a.run_name or a.run_id, group=a.group or None,
                        job_type="loop")
        # TWO step axes in one run, each declared. Without this every series would be
        # plotted against wandb's auto-incrementing `_step`, which counts LOG CALLS -- so
        # the per-iteration metrics and the per-epoch curve would interleave on a shared
        # axis that means neither, and a resumed loop would restart it at 0.
        wandb.define_metric("iteration")
        wandb.define_metric("ft/global_epoch")
        for prefix in ("buffer/*", "critic/*", "awr/*", "ft/final/*", "ft/epochs"):
            wandb.define_metric(prefix, step_metric="iteration")
        # Listed explicitly rather than as `ft/*`: the `ft/final/*` and `ft/epochs` series
        # also start with `ft/`, and wandb's glob resolution between overlapping patterns is
        # not something to rely on for which axis a series lands on.
        for prefix in ("ft/train_flow_mse", "ft/val_flow_mse", "ft/lr", "ft/secs",
                       "ft/is_best", "ft/iteration"):
            wandb.define_metric(prefix, step_metric="ft/global_epoch")
        for r in curve:
            run.log(r)
        run.log(row)
        run.finish()
        print(f"[awr-log] appended iteration {a.iteration} to {a.project}/{a.run_id}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[awr-log] WARNING wandb logging skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
