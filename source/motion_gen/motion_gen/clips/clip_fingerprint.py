# Copyright 2026 Zachary Olkin. All rights reserved.

"""Portable content fingerprint for generated motion-clip libraries.

A dense clip library is deterministically defined by three things:
  1. the *source* stride library it was stitched from (its portable
     content hash, from :mod:`..library_fingerprint`),
  2. the generator configuration (skill mix, velocity buckets, dt, clip
     length, resampling window, max-acc, transition prob, clip count), and
  3. the RNG seed.

Hashing those (rather than the multi-GB dense arrays) yields a fast,
machine-independent digest that answers "was this clip set generated from
the same inputs?".  It is recorded in the library HDF5 attrs and in
``params/env.yaml`` at train time, and verified at play / resume.

Stdlib-only so the standalone checker
``scripts/rsl_rl/clip_library_fingerprint.py`` can import it under a bare
``python3`` without the IsaacLab environment.
"""

from __future__ import annotations

import hashlib
import json

# Bump if the digest inputs or canonicalisation below change, or if the on-disk
# clip representation for a given config/seed changes (so old and new libraries
# never collide on the fingerprint).
# v2: outputs stored in clip-world frame (per-step stance anchor baked in; the
#     anchor_pos/anchor_quat schedule was dropped).
_CLIP_FINGERPRINT_VERSION = 2


def _canonical(obj) -> str:
    """Deterministic JSON for hashing (sorted keys, compact, stable floats)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_clip_config(
    source_library_fingerprint: dict,
    generator_config: dict,
    seed: int | None,
) -> dict:
    """Compute a portable fingerprint for a generated clip library.

    Args:
        source_library_fingerprint: The source stride library's fingerprint
            dict (from :func:`fingerprint_library`); only its ``hash`` is
            folded in.
        generator_config: JSON-serialisable dict of every input that affects
            the generated clips (skill_probs, velocity_buckets, dt,
            clip_length_s, resampling_time_range, max_acc,
            skill_transition_prob, num_clips, ...).
        seed: RNG seed used for generation (``None`` if unseeded).

    Returns:
        Dict with:
            ``hash``: full sha256 hex digest.
            ``short``: first 12 hex chars.
            ``source_hash``: the folded-in source library hash.
            ``version``: the fingerprint algorithm version.
    """
    source_hash = str(source_library_fingerprint.get("hash", ""))
    h = hashlib.sha256()
    h.update(f"clip-lib-fingerprint;version={_CLIP_FINGERPRINT_VERSION};".encode())
    h.update(f"source={source_hash};".encode())
    h.update(f"seed={seed};".encode())
    h.update(f"config={_canonical(generator_config)};".encode())
    digest = h.hexdigest()
    return {
        "hash": digest,
        "short": digest[:12],
        "source_hash": source_hash,
        "version": _CLIP_FINGERPRINT_VERSION,
    }


#: Bump when the canonicalisation in :func:`fingerprint_terrain_manifest` changes.

_SOLVED_FINGERPRINT_VERSION = 1


def fingerprint_solved_library(source: str, hf_repo: str | None = "zolkin/robot_rl") -> dict:
    """Content fingerprint of the SOLVED traj-opt library a clip library was stitched from.

    Moved here out of ``motion_matching.locomotion_db``, which is the DEPRECATED matcher package.
    Only the deprecated generators ever called it, so the LIVE Motion Bricks chain recorded no
    statement at all of which solved library produced a given clip library -- ``mb_flat_clips``,
    ``mb_assemble_library``, ``mb_pure_stair_clips`` and ``compose_terrain_library`` between them
    write ``generator``, ``terrain_manifest``, ``has_regime`` and a flat-tile count, and stop.

    That is a problem now rather than in principle: the solved library is being replaced, and
    ``hf_cache.download_from_hf`` REUSES whatever is already in the local ``hf/`` cache without
    re-checking the remote -- so two machines can hold different libraries under one name and a
    finished ``.h5`` would have no way to say which one it came from.

    The library is npz + ``manifest.csv`` (not the traj-opt yaml layout ``fingerprint_library``
    expects), so hash the manifest bytes plus the sorted per-clip relative paths and sizes. Sizes
    rather than contents: hashing several hundred npz bodies costs seconds per call for a check
    whose job is to catch "a different library", not "a bit-flip in one clip".

    Args:
        source: library path, repo-relative or absolute (e.g. ``trajectories/skills/library_v2``).
        hf_repo: HuggingFace repo to fall back to; ``None`` for a purely local path.

    Returns:
        ``{version, hash, short, num_clips, source}``.
    """
    from ..hf_cache import resolve_library_path                     # stdlib-only, lazy HF import

    root = resolve_library_path(source, hf_repo)
    h = hashlib.sha256()
    h.update(f"v{_SOLVED_FINGERPRINT_VERSION}".encode())
    manifest = root / "manifest.csv"
    if manifest.is_file():
        h.update(manifest.read_bytes())
    npzs = sorted(root.rglob("*.npz"))
    for f in npzs:
        h.update(str(f.relative_to(root)).encode())
        h.update(str(f.stat().st_size).encode())
    digest = h.hexdigest()
    return {"version": _SOLVED_FINGERPRINT_VERSION, "hash": digest, "short": digest[:12],
            "num_clips": len(npzs), "source": str(source)}


def verify_solved_fingerprint(recorded: dict | None, current: dict | None) -> tuple[bool, str]:
    """``(ok, message)`` for a recorded solved-library fingerprint against a freshly computed one.

    A MISSING recorded fingerprint is not a failure -- libraries built before this was wired carry
    none, and refusing them would be a migration gate rather than a drift check. An empty dict is
    also treated as absent, deliberately: "absent" and "empty" must not be the same thing, because
    an empty fingerprint would otherwise compare equal to anything.
    """
    if not recorded:
        return True, "no solved-library fingerprint recorded (library predates it)"
    if not current:
        return True, "no current solved library to compare against"
    if recorded.get("hash") == current.get("hash"):
        return True, f"solved library matches ({recorded.get('short', '?')})"
    return False, (
        f"solved library MISMATCH: library was built from {recorded.get('source', '?')} "
        f"({recorded.get('short', '?')}, {recorded.get('num_clips', '?')} clips) but the current "
        f"{current.get('source', '?')} hashes {current.get('short', '?')} "
        f"({current.get('num_clips', '?')} clips). The local hf/ cache is never re-checked against "
        f"the remote, so two machines can hold different libraries under one name."
    )


_TERRAIN_FINGERPRINT_VERSION = 1


def fingerprint_terrain_manifest(manifest: list[dict] | None) -> dict:
    """Content hash of the terrain a clip library describes.

    Distinct from :func:`fingerprint_clip_config` in two ways that matter:

    * it hashes the terrain that is actually IN the file, not the generator inputs that
      were meant to produce it -- so it works on every library already on disk, including
      the ones that carry no ``clip_fingerprint`` attribute at all (which is all of them:
      ``fingerprint_clip_config`` is imported in one place and called in none, so the
      existing check has never had anything to compare);
    * the terrain manifest is not one of ``fingerprint_clip_config``'s inputs, so an
      offline stage that rewrites tile geometry while leaving the motions alone -- exactly
      what terrain decoration is -- produces a library that reports an unchanged
      fingerprint while the ground has moved. That is the failure the
      ``--clip``-has-no-default rule exists to prevent, reached by a different road.

    The manifest is canonicalised by sorting tiles by ``terrain_id`` and sorting every
    key, so a re-serialisation that reorders nothing meaningful does not change the digest.

    Args:
        manifest: The library's ``terrain_manifest`` (already parsed), or ``None``.

    Returns:
        Dict with ``hash``, ``short``, ``version``, ``num_tiles`` and ``num_overlays``.
        A ``None`` or empty manifest is a legitimate flat-only library and hashes the
        empty list rather than returning nothing.
    """
    tiles = sorted(manifest or [], key=lambda m: int(m.get("terrain_id", -1)))
    h = hashlib.sha256()
    h.update(f"terrain-manifest;version={_TERRAIN_FINGERPRINT_VERSION};".encode())
    h.update(_canonical(tiles).encode())
    digest = h.hexdigest()
    return {
        "hash": digest,
        "short": digest[:12],
        "version": _TERRAIN_FINGERPRINT_VERSION,
        "num_tiles": len(tiles),
        "num_overlays": sum(len(m.get("overlays", [])) for m in tiles),
    }


def verify_terrain_fingerprint(recorded: dict | None, current: dict | None) -> tuple[bool, str]:
    """Compare a recorded terrain fingerprint against the terrain about to be used.

    Args:
        recorded: Fingerprint stored in the artifact, or ``None`` for one that predates it.
        current: Fingerprint of the loaded library's manifest.

    Returns:
        ``(ok, message)``. ``ok`` is False when the hashes differ or either side is absent;
        the caller decides whether an absent one is fatal.
    """
    rec = str((recorded or {}).get("hash", ""))
    cur = str((current or {}).get("hash", ""))
    if rec and cur and rec == cur:
        return True, f"terrain match ({current.get('short', cur[:12])}, " \
                     f"{current.get('num_tiles')} tiles)"
    if not rec:
        return False, ("no terrain fingerprint recorded in this artifact -- it predates the "
                       "check, so the terrain it was trained on cannot be verified")
    if not cur:
        return False, "no terrain fingerprint for the loaded library to compare against"
    return False, (
        f"TERRAIN MISMATCH: trained on terrain {recorded.get('short', rec[:12])} "
        f"({recorded.get('num_tiles')} tiles, {recorded.get('num_overlays')} overlays) but the "
        f"loaded library describes {current.get('short', cur[:12])} "
        f"({current.get('num_tiles')} tiles, {current.get('num_overlays')} overlays). The clips "
        "may be identical -- the GROUND is not."
    )


def verify_clip_fingerprint(recorded: dict, current: dict) -> tuple[bool, str]:
    """Compare a recorded clip fingerprint against the current one.

    Args:
        recorded: Fingerprint stored at train time (from
            :func:`fingerprint_clip_config`).
        current: Fingerprint of the clip library about to be used.

    Returns:
        ``(ok, message)`` -- ``ok`` is True when the full hashes match;
        ``message`` describes any divergence for a loud warning.
    """
    rec = str(recorded.get("hash", "")) if recorded else ""
    cur = str(current.get("hash", "")) if current else ""
    if rec and cur and rec == cur:
        return True, f"clip library fingerprint match ({current.get('short', cur[:12])})"
    return False, (
        "CLIP LIBRARY FINGERPRINT MISMATCH: policy trained on clip set "
        f"{recorded.get('short', rec[:12]) or '<none>'} (source "
        f"{recorded.get('source_hash', '')[:12]}) but the current clip library is "
        f"{current.get('short', cur[:12]) or '<none>'} (source "
        f"{current.get('source_hash', '')[:12]}). Tracking targets differ."
    )
