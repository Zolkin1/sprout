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
import math
import pathlib


def load(path: pathlib.Path) -> dict[int, dict]:
    """Rows of one arm, keyed by clip id.

    Raises:
        SystemExit: If a clip appears twice.  That is the job-list wrap
            (``GroupAllocator._take`` re-issues the last job once drained), and silently
            keeping one of the two would pair the arms on a clip they ran different numbers
            of times.
    """
    rows = json.loads(path.read_text())["rows"]
    out: dict[int, dict] = {}
    for r in rows:
        cid = int(r["clip_id"])
        if cid in out:
            raise SystemExit(
                f"{path.name}: clip {cid} appears more than once. The arm re-ran a clip, so "
                "it cannot be paired -- re-run with the retire-on-drained fix.")
        out[cid] = r
    return out


def mcnemar(base: dict[int, dict], arm: dict[int, dict], clips) -> dict:
    """Paired survival difference over ``clips``, with the discordant-pair z."""
    b = [base[c]["verdict"] == "success" for c in clips]
    a = [arm[c]["verdict"] == "success" for c in clips]
    win = sum(1 for x, y in zip(b, a) if not x and y)
    lose = sum(1 for x, y in zip(b, a) if x and not y)
    disc = win + lose
    n = len(clips)
    return {"n": n, "base": sum(b) / n if n else float("nan"),
            "arm": sum(a) / n if n else float("nan"),
            "delta": (sum(a) - sum(b)) / n if n else float("nan"),
            "win": win, "lose": lose,
            "z": (win - lose) / math.sqrt(disc) if disc else 0.0}


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
    others = sorted(x for x in d.glob("select__*.json") if x.name != args.baseline)
    print(f"baseline: {args.baseline}  ({len(base)} clips)\n")

    for path in others:
        arm = load(path)
        shared = sorted(set(base) & set(arm))
        print(f"=== {path.name}  ({len(arm)} clips, {len(shared)} paired) ===")
        print(f"    {'family':<16} {'n':>4} {'base':>7} {'arm':>7} {'delta':>8} "
              f"{'+/-':>9} {'z':>6}")
        by: dict[int, list[int]] = {}
        for c in shared:
            by.setdefault(int(base[c]["family_code"]), []).append(c)
        def _row(label: str, clips) -> None:
            """One line of the table."""
            m = mcnemar(base, arm, clips)
            disc = f"+{m['win']}/-{m['lose']}"
            print(f"    {label:<16} {m['n']:>4} {m['base']:>7.3f} {m['arm']:>7.3f} "
                  f"{m['delta']:>+8.3f} {disc:>9} {m['z']:>+6.2f}")

        for code, clips in sorted(by.items()):
            _row(str(name_of.get(code, code)), clips)
        _row("ALL", shared)
        noflat = [c for c in shared if name_of.get(int(base[c]["family_code"])) != "flat"]
        if len(noflat) != len(shared):
            # Reported separately because you flagged the flat rollouts as mislabelled; the
            # metric here is termination-based rather than goal-based, so flat is less
            # exposed, but the split makes that visible instead of assumed.
            _row("ALL minus flat", noflat)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
