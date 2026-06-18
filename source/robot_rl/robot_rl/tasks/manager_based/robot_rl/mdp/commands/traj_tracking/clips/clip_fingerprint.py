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

# Bump if the digest inputs or canonicalisation below change.
_CLIP_FINGERPRINT_VERSION = 1


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
