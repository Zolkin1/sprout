# Copyright 2026 Zachary Olkin. All rights reserved.

"""Read and verify the data registry (``data/registry.yaml``).

The registry records the content fingerprint (identity) of named data artifacts --
clip libraries (the ``clip_fingerprint`` stored in the ``.h5``) and trajectory
libraries (a portable content hash over the YAMLs). This catches the silent
local/server divergence: verify that the artifact on *this* machine matches the
recorded identity before training or deploying. Read-only; no HuggingFace / network.

Usage:
    python scripts/rsl_rl/data_registry.py list
    python scripts/rsl_rl/data_registry.py verify g1_flat_v1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "data" / "registry.yaml"
_LIB_FP_MODULE = (
    _REPO_ROOT / "source" / "robot_rl" / "robot_rl" / "tasks" / "manager_based"
    / "robot_rl" / "mdp" / "commands" / "traj_tracking" / "library_fingerprint.py"
)
_KINDS = ("clip_libraries", "trajectory_libraries")


def load_registry(path: str | os.PathLike | None = None) -> dict:
    """Load the registry YAML into a dict (empty dict if the file is empty)."""
    import yaml  # noqa: PLC0415

    with open(Path(path) if path else _REGISTRY_PATH) as f:
        return yaml.safe_load(f) or {}


def get(name: str, registry: dict | None = None) -> dict | None:
    """Look up an artifact by name across all kinds; returns its entry + kind/name."""
    reg = registry if registry is not None else load_registry()
    for kind in _KINDS:
        if name in (reg.get(kind) or {}):
            return {"kind": kind, "name": name, **reg[kind][name]}
    return None


def _clip_library_hash(h5_path: str) -> str | None:
    """Read the stored ``clip_fingerprint`` hash from a clip-library ``.h5``."""
    import h5py  # noqa: PLC0415

    with h5py.File(h5_path, "r") as f:
        fp = f.attrs.get("clip_fingerprint")
    if fp is None:
        return None
    if isinstance(fp, (bytes, str)):
        fp = json.loads(fp)
    return fp.get("hash")


def _trajectory_library_hash(folder: str) -> str | None:
    """Compute the portable content fingerprint of a trajectory-library folder."""
    spec = importlib.util.spec_from_file_location("library_fingerprint", _LIB_FP_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fingerprint_library(folder)["hash"]


def verify(name: str) -> tuple[bool, str]:
    """Verify the local artifact ``name`` matches its registry fingerprint.

    Returns ``(ok, message)``. ``ok`` is False for a missing entry, a missing local
    artifact, or a fingerprint mismatch.
    """
    entry = get(name)
    if entry is None:
        return False, f"'{name}' is not in the registry."
    expected = entry.get("fingerprint")
    if entry["kind"] == "clip_libraries":
        path = entry.get("path")
        if not path or not os.path.isfile(path):
            return False, f"clip library '{name}' local path missing: {path}"
        actual = _clip_library_hash(path)
    else:
        path = str(_REPO_ROOT / "hf" / entry.get("hf_path", ""))
        if not os.path.isdir(path):
            return False, f"trajectory library '{name}' not cached locally: {path}"
        actual = _trajectory_library_hash(path)
    if actual == expected:
        return True, f"{name}: fingerprint match ({str(expected)[:12]})"
    return False, (
        f"{name}: FINGERPRINT MISMATCH -- registry {str(expected)[:12]} vs local "
        f"{str(actual)[:12]} ({path})"
    )


def main() -> None:
    """CLI: ``list`` registered artifacts or ``verify <name>`` a local one."""
    p = argparse.ArgumentParser(description="Read/verify the data registry.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List registered artifacts.")
    vp = sub.add_parser("verify", help="Verify a local artifact matches its registry fingerprint.")
    vp.add_argument("name")
    args = p.parse_args()

    if args.cmd == "list":
        reg = load_registry()
        for kind in _KINDS:
            print(f"[{kind}]")
            for n, e in (reg.get(kind) or {}).items():
                loc = e.get("path") or e.get("hf_path", "")
                print(f"  {n:28s} {str(e.get('fingerprint', ''))[:12]}  {loc}")
    elif args.cmd == "verify":
        ok, msg = verify(args.name)
        print(("[OK] " if ok else "[FAIL] ") + msg)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
