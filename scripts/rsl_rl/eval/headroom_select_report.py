# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compare selector arms, PER FAMILY and paired on clip id.

Imports no simulator, so a result can be re-read anywhere the json landed.

**Paired on clip, never pooled.**  The arms run the same 132 clips, and the families differ
enormously in difficulty -- an unpaired difference between two arms is then partly a
difference in which clips each happened to finish.  Pairing also buys the only test with any
power at this sample size: with ~130 clips per arm an unpaired interval is +-5 points, which
is wider than the effect the gate is trying to bound, while the discordant pairs give a
McNemar z on the clips where the arms actually disagreed.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pathlib


def load(path: pathlib.Path) -> dict[int, float]:
    """Per-CLIP success fraction for one arm.

    Clip-averaged, not episode-pooled, which is the convention the eval harness uses and for
    the same reason: episodes of one clip share a tile and a terrain and are not independent
    draws, so pooling them overstates the sample. A clip that ran three episodes contributes
    one number here, its own success fraction.
    """
    rows = json.loads(path.read_text())["rows"]
    by: dict[int, list[float]] = {}
    for r in rows:
        by.setdefault(int(r["clip_id"]), []).append(
            1.0 if r["verdict"] == "success" else 0.0)
    return {c: sum(v) / len(v) for c, v in by.items()}


def load_family(path: pathlib.Path) -> dict[int, int]:
    """Family code per clip id."""
    return {int(r["clip_id"]): int(r["family_code"])
            for r in json.loads(path.read_text())["rows"]}


def paired(base: dict[int, float], arm: dict[int, float], clips, n_boot: int = 4000,
           seed: int = 0) -> dict:
    """Clip-averaged paired difference, with a bootstrap CI over CLIPS.

    Bootstrapping the clips rather than the episodes is what makes the interval honest here:
    the two arms ran the same clips, so the pairing removes the (large) between-clip variance
    that would otherwise swamp the effect.
    """
    b = np.array([base[c] for c in clips])
    a = np.array([arm[c] for c in clips])
    d = a - b
    n = len(clips)
    if n == 0:
        return {"n": 0, "base": float("nan"), "arm": float("nan"), "delta": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "moved": 0}
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {"n": n, "base": float(b.mean()), "arm": float(a.mean()),
            "delta": float(d.mean()),
            "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "moved": int((d != 0).sum())}


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="directory of select__*.json")
    p.add_argument("--plan", required=True, help="plan.json, for family names")
    p.add_argument("--baseline", default="select__k0__oracle__real.json")
    args = p.parse_args()

    d = pathlib.Path(args.dir)
    plan = json.loads(pathlib.Path(args.plan).read_text())
    families = sorted({c["family"] for c in plan["clips"]})
    name_of = dict(enumerate(families))

    base_path = d / args.baseline
    if not base_path.exists():
        raise SystemExit(f"no baseline at {base_path}")
    base = load(base_path)
    fam_of = load_family(base_path)
    others = sorted(x for x in d.glob("select__*.json") if x.name != args.baseline)
    print(f"baseline: {args.baseline}  ({len(base)} clips)\n")

    for path in others:
        arm = load(path)
        shared = sorted(set(base) & set(arm))
        print(f"=== {path.name}  ({len(arm)} clips, {len(shared)} paired) ===")
        print(f"    {'family':<16} {'n':>4} {'base':>7} {'arm':>7} {'delta':>8} "
              f"{'95% CI':>18} {'moved':>6}")
        by: dict[int, list[int]] = {}
        for c in shared:
            by.setdefault(fam_of[c], []).append(c)
        def _row(label: str, clips) -> None:
            """One line of the table."""
            m = paired(base, arm, clips)
            ci = f"[{m['lo']:+.3f}, {m['hi']:+.3f}]"
            print(f"    {label:<16} {m['n']:>4} {m['base']:>7.3f} {m['arm']:>7.3f} "
                  f"{m['delta']:>+8.3f} {ci:>18} {m['moved']:>6}")

        for code, clips in sorted(by.items()):
            _row(str(name_of.get(code, code)), clips)
        _row("ALL", shared)
        noflat = [c for c in shared if name_of.get(fam_of[c]) != "flat"]
        if len(noflat) != len(shared):
            # Reported separately because you flagged the flat rollouts as mislabelled; the
            # metric here is termination-based rather than goal-based, so flat is less
            # exposed, but the split makes that visible instead of assumed.
            _row("ALL minus flat", noflat)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
