#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline checker for CLF (Lyapunov) Q/R weights.

Builds the Lyapunov matrix ``P`` that ``clf.py`` would solve from a set of
``Q``/``R`` weights, overlays the per-output tracking errors measured in a
``plots/stats.txt`` run, and reports how much each output (and group) actually
contributes to ``V`` — split into its position vs velocity parts. This lets you
sweep ``q_p`` factors and see whether position tracking gains weight in ``V``
*before* spending a training run.

Background — the CLF in ``clf.py`` solves the continuous-time ARE for a set of
*decoupled, normalized double integrators* (one per output: state
``[pos_err, vel_err]``, ``A=[[0,1],[0,0]]``, ``B=[[0],[1]]``). So ``P`` is
block-diagonal with one 2x2 block per output, and each block has a closed form
(this script uses it directly, so it needs only numpy):

    p_pos   = sqrt(q_p) * sqrt(2*sqrt(q_p*r) + q_v)
    p_cross = sqrt(q_p*r)
    p_vel   = sqrt(r)   * sqrt(2*sqrt(q_p*r) + q_v)

    V_i = p_pos*e_pos^2 + 2*p_cross*e_pos*e_vel + p_vel*e_vel^2

The position-vs-velocity weight ratio is ``p_pos/p_vel = sqrt(q_p/r)`` and is
independent of ``q_v``. To make position contribute as much as velocity for an
output you need ``sqrt(q_p/r) ~= <e_vel^2>/<e_pos^2>``.

Notes / caveats:
  * Only the *diagonal* (marginal) contributions are summed; the pos-vel cross
    term ``2*p_cross*<e_pos*e_vel>`` is not in stats.txt, so the diagonal-V
    estimate differs from the true mean V (printed for comparison).
  * For orientation outputs, ``V`` uses a tangent box-minus error that is NOT
    what stats.txt reports (stats shows raw quaternion-component errors). Those
    rows use the ori_x/y/z component MSE as a *proxy* and are flagged ``~``.
    Position (``:pos_``) and joint (``joint:``) rows are exact.
  * ``--auto-whiten`` only retunes the exact (pos/joint) channels; it leaves
    orientation channels at baseline.

Examples:
    # Inspect current weights against a run, list worst velocity-dominated dims
    python scripts/rsl_rl/clf_weight_check.py --stats logs/.../plots/stats.txt

    # Preview: make every foot-position channel weight position 2x harder
    python scripts/rsl_rl/clf_weight_check.py --stats logs/.../plots/stats.txt \
        --target-ratio "ankle_roll_link:pos=22,pelvis_link:pos=8"

    # Preview: whiten all exact channels to equal pos/vel contribution
    python scripts/rsl_rl/clf_weight_check.py --stats logs/.../plots/stats.txt --auto-whiten 1.0
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.normpath(
    os.path.join(
        _THIS_DIR, "..", "..", "source", "robot_rl", "robot_rl", "tasks",
        "manager_based", "robot_rl", "g1", "clf_weights.py",
    )
)


def load_weights(path: str, q_name: str, r_name: str) -> tuple[dict, dict]:
    """Load the ``Q``/``R`` weight dicts from a (pure-data) weights module.

    Args:
        path: Path to ``clf_weights.py`` (only executes that file).
        q_name: Name of the Q-weights dict (e.g. ``WALKING_Q_weights``).
        r_name: Name of the R-weights dict (e.g. ``WALKING_R_weights``).

    Returns:
        ``(Q_weights, R_weights)`` — name -> ``[q_pos, q_vel]`` and name -> ``[r]``.
    """
    spec = importlib.util.spec_from_file_location("_clf_weights_mod", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load weights module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, q_name), getattr(mod, r_name)


# ---------------------------------------------------------------------------
# stats.txt parsing
# ---------------------------------------------------------------------------

# A data row in the error tables: "  <name>   <mean>   <std>"
_ROW = re.compile(r"^\s+(\S+:\S+|joint:\S+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")


def parse_stats(path: str, block: str) -> tuple[dict[str, float], dict[str, float], float | None]:
    """Parse per-dimension position/velocity MSE and mean V from a stats.txt.

    Args:
        path: Path to a ``plots/stats.txt`` produced by ``play_plots.py``.
        block: Which block to read — ``"all"`` (All Steps) or ``"timeout"``
            (Successful Tracking, time_out episodes only).

    Returns:
        ``(pos_mse, vel_mse, mean_v)`` keyed by output name; ``mean_v`` is the
        block's reported "Lyapunov Function (V) Mean across envs" (or None).
    """
    want = "Successful Tracking" if block == "timeout" else "All Steps"
    with open(path) as fh:
        lines = fh.readlines()

    pos: dict[str, float] = {}
    vel: dict[str, float] = {}
    mean_v: float | None = None

    in_block = False
    mode: str | None = None  # "pos" | "vel" | None
    capture_v = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("All Steps", "Successful Tracking")):
            in_block = stripped.startswith(want)
            mode = None
            continue
        if not in_block:
            continue
        if stripped.startswith("Lyapunov Function (V)"):
            capture_v = True
            mode = None
            continue
        if capture_v and stripped.startswith("Mean across envs:"):
            mean_v = float(stripped.split(":")[1])
            capture_v = False
            continue
        if stripped.startswith("Position Errors"):
            mode = "pos"
            continue
        if stripped.startswith("Velocity Errors"):
            mode = "vel"
            continue
        if stripped.startswith("Group Summaries") or stripped.startswith("="):
            mode = None
            continue
        if mode is None:
            continue
        m = _ROW.match(line)
        if m:
            name, mean_s, _ = m.groups()
            target = pos if mode == "pos" else vel
            target[name] = float(mean_s)
    return pos, vel, mean_v


# ---------------------------------------------------------------------------
# CARE block solution (closed form for the normalized double integrator)
# ---------------------------------------------------------------------------

def solve_block(q_p: float, q_v: float, r: float) -> tuple[float, float, float]:
    """Exact CARE solution for one normalized-double-integrator output.

    Solves A^T P + P A - P B R^-1 B^T P + Q = 0 with A=[[0,1],[0,0]],
    B=[[0],[1]], Q=diag(q_p, q_v), R=r. Matches scipy.solve_continuous_are.

    Returns:
        ``(p_pos, p_cross, p_vel)`` — entries of the 2x2 block
        ``[[p_pos, p_cross], [p_cross, p_vel]]``.
    """
    p_cross = math.sqrt(q_p * r)
    common = math.sqrt(2.0 * p_cross + q_v)
    p_pos = math.sqrt(q_p) * common
    p_vel = math.sqrt(r) * common
    return p_pos, p_cross, p_vel


def block_eigs(p_pos: float, p_cross: float, p_vel: float) -> tuple[float, float]:
    """Eigenvalues (min, max) of the symmetric 2x2 block."""
    tr = p_pos + p_vel
    disc = math.sqrt(max(0.0, (p_pos - p_vel) ** 2 + 4.0 * p_cross ** 2))
    return 0.5 * (tr - disc), 0.5 * (tr + disc)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def group_of(name: str) -> str:
    """Map an output name to its group (matching play_plots conventions)."""
    if name.startswith("joint:"):
        return "Joints"
    if ":pos_" in name:
        return "Positions"
    if ":ori_" in name:
        return "Orientations"
    return "Other"


def is_exact(name: str) -> bool:
    """True if stats.txt MSE for this output equals the eta error used in V.

    Orientation outputs use a tangent box-minus error in V that stats.txt does
    not report, so those are proxies (False).
    """
    return name.startswith("joint:") or ":pos_" in name


def analyze(
    output_names: list[str],
    q_weights: dict,
    r_weights: dict,
    pos_mse: dict[str, float],
    vel_mse: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Build per-channel records and global P stats for one weight set.

    Returns:
        ``(records, globals)`` where each record has the per-output weights,
        block entries, measured errors, and V contributions; ``globals`` holds
        ``lambda_max``, ``lambda_min``, ``cond``, and the diagonal-V totals.
    """
    records: list[dict[str, Any]] = []
    lam_max = -math.inf
    lam_min = math.inf
    for name in output_names:
        qp, qv = (q_weights.get(name, [1.0, 1.0]) + [1.0, 1.0])[:2]
        r = (r_weights.get(name, [1.0]) + [1.0])[0]
        p_pos, p_cross, p_vel = solve_block(qp, qv, r)
        lo, hi = block_eigs(p_pos, p_cross, p_vel)
        lam_max = max(lam_max, hi)
        lam_min = min(lam_min, lo)
        e_pos2 = pos_mse.get(name)
        e_vel2 = vel_mse.get(name)
        pos_c = p_pos * e_pos2 if e_pos2 is not None else None
        vel_c = p_vel * e_vel2 if e_vel2 is not None else None
        records.append({
            "name": name, "group": group_of(name), "exact": is_exact(name),
            "qp": qp, "qv": qv, "r": r, "ratio": math.sqrt(qp / r),
            "p_pos": p_pos, "p_vel": p_vel,
            "e_pos2": e_pos2, "e_vel2": e_vel2,
            "pos_c": pos_c, "vel_c": vel_c,
        })

    pos_tot = sum(rec["pos_c"] for rec in records if rec["pos_c"] is not None)
    vel_tot = sum(rec["vel_c"] for rec in records if rec["vel_c"] is not None)
    total = pos_tot + vel_tot
    globals_: dict[str, float] = {
        "lambda_max": lam_max, "lambda_min": lam_min,
        "cond": (lam_max / lam_min) if lam_min > 0 else math.inf,
        "pos_tot": pos_tot, "vel_tot": vel_tot, "diag_v": total,
        "pos_frac": (pos_tot / total) if total > 0 else float("nan"),
    }
    return records, globals_


# ---------------------------------------------------------------------------
# Candidate weight overrides
# ---------------------------------------------------------------------------

def _parse_spec(spec: str) -> list[tuple[str, float]]:
    """Parse ``"pat=val,pat2=val2"`` into ``[(pat, val), ...]``."""
    out: list[tuple[str, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        pat, _, val = part.partition("=")
        out.append((pat.strip(), float(val)))
    return out


def apply_overrides(
    q_weights: dict,
    r_weights: dict,
    output_names: list[str],
    args: argparse.Namespace,
    pos_mse: dict[str, float],
    vel_mse: dict[str, float],
) -> tuple[dict, dict]:
    """Return deep-copied weight dicts with the requested candidate changes.

    Application order: auto-whiten (sets q_p from measured errors), then
    target-ratio (sets ``q_p = r*val^2``), then multiplicative bump-qp / bump-r.
    Substring patterns match against the output name.
    """
    qw = {k: list(v) for k, v in q_weights.items()}
    rw = {k: list(v) for k, v in r_weights.items()}

    def qp_for(name: str) -> list[float]:
        return qw.setdefault(name, list(q_weights.get(name, [1.0, 1.0])))

    if args.auto_whiten is not None:
        kappa = args.auto_whiten
        for name in output_names:
            if not is_exact(name):
                continue
            e_pos2, e_vel2 = pos_mse.get(name), vel_mse.get(name)
            if not e_pos2 or not e_vel2:
                continue
            r = (rw.get(name, [1.0]))[0]
            ratio = kappa * (e_vel2 / e_pos2)
            qp_for(name)[0] = r * ratio ** 2

    for pat, val in _parse_spec(args.target_ratio or ""):
        for name in output_names:
            if pat in name:
                r = (rw.get(name, [1.0]))[0]
                qp_for(name)[0] = r * val ** 2

    for pat, val in _parse_spec(args.bump_qp or ""):
        for name in output_names:
            if pat in name:
                qp_for(name)[0] *= val

    for pat, val in _parse_spec(args.bump_r or ""):
        for name in output_names:
            if pat in name:
                rw.setdefault(name, list(r_weights.get(name, [1.0])))[0] *= val

    return qw, rw


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    label: str, globals_: dict[str, float], mean_v: float | None,
    have_stats: bool, max_eta_err: float,
) -> None:
    """Print the one-line P / V summary for a weight set."""
    max_clf = globals_["lambda_max"] * max_eta_err ** 2
    print(f"\n[{label}]")
    print(f"  P: lambda_max={globals_['lambda_max']:.3f}  "
          f"lambda_min={globals_['lambda_min']:.4f}  cond={globals_['cond']:.1f}")
    print(f"  max_clf = lambda_max*max_eta_err^2 (eta={max_eta_err:g}): {max_clf:.2f}  "
          f"(reward = exp(-V/max_clf))")
    if have_stats:
        v_ref = f"  (stats mean V={mean_v:.3f})" if mean_v is not None else ""
        print(f"  diagonal-V estimate: {globals_['diag_v']:.3f}{v_ref}   "
              f"position fraction of V: {100.0 * globals_['pos_frac']:.1f}%")


def print_groups(records: list[dict[str, Any]]) -> None:
    """Print per-group position/velocity V contribution and position fraction."""
    print(f"  {'group':<14}{'pos_contrib':>13}{'vel_contrib':>13}{'pos_frac':>10}")
    print(f"  {'-'*13:<14}{'-'*12:>13}{'-'*12:>13}{'-'*9:>10}")
    for grp in ("Positions", "Orientations", "Joints"):
        recs = [r for r in records if r["group"] == grp and r["pos_c"] is not None and r["vel_c"] is not None]
        if not recs:
            continue
        pc = sum(r["pos_c"] for r in recs)
        vc = sum(r["vel_c"] for r in recs)
        frac = pc / (pc + vc) if (pc + vc) > 0 else float("nan")
        approx = "  ~proxy" if grp == "Orientations" else ""
        print(f"  {grp:<14}{pc:>13.4f}{vc:>13.4f}{100.0*frac:>9.1f}%{approx}")


def print_worst(records: list[dict[str, Any]], top: int) -> None:
    """Print the most velocity-dominated channels (largest vel/pos contribution)."""
    scored = [
        (r["vel_c"] / r["pos_c"], r) for r in records
        if r["pos_c"] not in (None, 0.0) and r["vel_c"] is not None
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    print(f"\nMost velocity-dominated channels (top {top}):")
    print(f"  {'name':<34}{'vel/pos':>9}{'sqrt(qp/r)':>12}{'need':>9}{'flag':>6}")
    print(f"  {'-'*33:<34}{'-'*8:>9}{'-'*11:>12}{'-'*8:>9}{'-'*5:>6}")
    for ratio_vp, r in scored[:top]:
        need = (r["e_vel2"] / r["e_pos2"]) if r["e_pos2"] else float("nan")
        flag = "" if r["exact"] else "~"
        print(f"  {r['name']:<34}{ratio_vp:>9.1f}{r['ratio']:>12.1f}{need:>9.1f}{flag:>6}")


def print_channel_table(records: list[dict[str, Any]]) -> None:
    """Verbose per-channel dump."""
    print(f"\n  {'name':<34}{'qp':>7}{'qv':>6}{'r':>7}{'sqrt(qp/r)':>11}"
          f"{'p_pos':>8}{'p_vel':>8}{'pos_c':>9}{'vel_c':>9}")
    for r in records:
        pc = f"{r['pos_c']:.4f}" if r["pos_c"] is not None else "   n/a"
        vc = f"{r['vel_c']:.4f}" if r["vel_c"] is not None else "   n/a"
        flag = "" if r["exact"] else "~"
        print(f"  {r['name']+flag:<34}{r['qp']:>7.2f}{r['qv']:>6.2f}{r['r']:>7.3f}"
              f"{r['ratio']:>11.2f}{r['p_pos']:>8.3f}{r['p_vel']:>8.3f}{pc:>9}{vc:>9}")


def _fmt_num(v: float) -> str:
    """Format a weight value cleanly for pasting (e.g. 6.05, 0.05, 1.0)."""
    r = round(float(v), 4)
    return f"{int(r)}.0" if r == int(r) else repr(r)


def _changed(a: list | None, b: list | None, tol: float = 1e-9) -> bool:
    """True if two weight lists differ (length or any entry beyond tol)."""
    if a is None or b is None:
        return a is not b
    return len(a) != len(b) or any(abs(x - y) > tol for x, y in zip(a, b))


def emit_weights(
    q_name: str, r_name: str, output_names: list[str],
    base_q: dict, base_r: dict, cand_q: dict, cand_r: dict, mode: str,
) -> None:
    """Print copy-pasteable ``Q``/``R`` assignments for ``clf_weights.py``.

    ``mode="changed"`` emits only entries that differ from baseline (minimal
    diff to paste over); ``mode="full"`` emits every entry in the dicts.
    """
    if mode == "full":
        q_names = list(cand_q.keys())
        r_names = list(cand_r.keys())
        title = "full set"
    else:
        q_names = [n for n in output_names if _changed(base_q.get(n), cand_q.get(n))]
        r_names = [n for n in output_names if _changed(base_r.get(n), cand_r.get(n))]
        title = "changed entries only"

    print("\n" + "=" * 72)
    print(f"  Paste into clf_weights.py  ({title})")
    print("=" * 72)
    if not q_names and not r_names:
        print("  (no changes)")
        return
    for n in q_names:
        qp, qv = (list(cand_q.get(n, [1.0, 1.0])) + [1.0, 1.0])[:2]
        print(f'{q_name}["{n}"] = [{_fmt_num(qp)}, {_fmt_num(qv)}]')
    if r_names:
        print()
    for n in r_names:
        r = (list(cand_r.get(n, [1.0])) + [1.0])[0]
        print(f'{r_name}["{n}"] = [{_fmt_num(r)}]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args, build P, overlay stats, and print baseline (+candidate)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=_DEFAULT_WEIGHTS, help="Path to clf_weights.py.")
    p.add_argument("--q-name", default="WALKING_Q_weights", help="Q-weights dict name.")
    p.add_argument("--r-name", default="WALKING_R_weights", help="R-weights dict name.")
    p.add_argument("--stats", default=None, help="Path to a plots/stats.txt to overlay measured MSE.")
    p.add_argument("--block", choices=["all", "timeout"], default="all",
                   help="Which stats.txt block to read (default: all).")
    p.add_argument("--target-ratio", default=None,
                   help='Set sqrt(q_p/r)=val for matching channels: "pat=val,...".')
    p.add_argument("--bump-qp", default=None, help='Multiply q_p of matching channels: "pat=factor,...".')
    p.add_argument("--bump-r", default=None, help='Multiply r of matching channels: "pat=factor,...".')
    p.add_argument("--auto-whiten", type=float, default=None, metavar="KAPPA",
                   help="Set sqrt(q_p/r)=KAPPA*<e_vel^2>/<e_pos^2> for exact channels (needs --stats).")
    p.add_argument("--max-eta-err", type=float, default=6.0,
                   help="max_eta_err in max_clf=lambda_max*max_eta_err^2 (match the reward cfg; default 6.0).")
    p.add_argument("--emit", choices=["changed", "full"], nargs="?", const="changed", default=None,
                   help="Print copy-pasteable Q/R assignments (default 'changed' = candidate diff only).")
    p.add_argument("--top", type=int, default=12, help="How many worst channels to list.")
    p.add_argument("--verbose", action="store_true", help="Print the full per-channel table.")
    args = p.parse_args()

    q_weights, r_weights = load_weights(args.weights, args.q_name, args.r_name)

    have_stats = args.stats is not None
    pos_mse: dict[str, float] = {}
    vel_mse: dict[str, float] = {}
    mean_v: float | None = None
    if have_stats:
        pos_mse, vel_mse, mean_v = parse_stats(args.stats, args.block)
        # The eta outputs are exactly the rows of the Velocity Errors table.
        output_names = list(vel_mse.keys())
    else:
        # No stats: analyze every output present in the weight dict.
        output_names = [n for n in q_weights if n in r_weights]

    if args.auto_whiten is not None and not have_stats:
        p.error("--auto-whiten requires --stats")

    print("=" * 72)
    print(f"CLF weight check  |  {args.q_name} / {args.r_name}")
    print(f"weights: {args.weights}")
    if have_stats:
        print(f"stats  : {args.stats}  [block: {args.block}]")
    print(f"outputs: {len(output_names)}")
    print("=" * 72)

    base_records, base_globals = analyze(output_names, q_weights, r_weights, pos_mse, vel_mse)
    print_summary("baseline", base_globals, mean_v, have_stats, args.max_eta_err)
    if have_stats:
        print_groups(base_records)
        print_worst(base_records, args.top)
    if args.verbose:
        print_channel_table(base_records)

    has_override = any([args.target_ratio, args.bump_qp, args.bump_r, args.auto_whiten is not None])
    if has_override:
        cand_q, cand_r = apply_overrides(q_weights, r_weights, output_names, args, pos_mse, vel_mse)
        cand_records, cand_globals = analyze(output_names, cand_q, cand_r, pos_mse, vel_mse)
        print("\n" + "=" * 72)
        print_summary("candidate", cand_globals, mean_v, have_stats, args.max_eta_err)
        if have_stats:
            print_groups(cand_records)
        if args.verbose:
            print_channel_table(cand_records)
        # Diff
        print("\n[diff baseline -> candidate]")
        if have_stats:
            print(f"  position fraction of V: {100.0*base_globals['pos_frac']:.1f}%"
                  f"  ->  {100.0*cand_globals['pos_frac']:.1f}%")
        print(f"  lambda_max: {base_globals['lambda_max']:.3f}  ->  {cand_globals['lambda_max']:.3f}")
        print(f"  cond(P)   : {base_globals['cond']:.1f}  ->  {cand_globals['cond']:.1f}")
        base_mc = base_globals["lambda_max"] * args.max_eta_err ** 2
        cand_mc = cand_globals["lambda_max"] * args.max_eta_err ** 2
        eta_hold = args.max_eta_err * math.sqrt(base_globals["lambda_max"] / cand_globals["lambda_max"])
        print(f"  max_clf   : {base_mc:.2f}  ->  {cand_mc:.2f}")
        print(f"  -> to hold max_clf constant (keep reward dynamic range), set "
              f"max_eta_err ~= {eta_hold:.3f}  (was {args.max_eta_err:g})")
        print("\n  NOTE: candidate is preview-only; this script writes nothing.")

    if args.emit is not None:
        if has_override:
            emit_weights(args.q_name, args.r_name, output_names,
                         q_weights, r_weights, cand_q, cand_r, args.emit)
        else:
            emit_weights(args.q_name, args.r_name, output_names,
                         q_weights, r_weights, q_weights, r_weights, "full")


if __name__ == "__main__":
    main()
