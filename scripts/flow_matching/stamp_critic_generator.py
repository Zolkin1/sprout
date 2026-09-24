#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stamp a critic checkpoint with the CONTENT hash of the generator it was fitted over.

A critic borrows the generator's encoder AND its three normalizers, so reading one against the
wrong generator loads with every shape agreeing and three wrong input scales -- which nothing
downstream can detect.  ``ValueCritic`` therefore refuses a ``--critic_fm_pt`` that differs
from the path the checkpoint names unless it can verify the pair by content, and a checkpoint
fitted before ``fit_value.py`` began stamping ``fm_checkpoint_sha256`` names an absolute path
on the machine that fitted it and carries no hash.  Moved to another box, that critic is
unusable: the named path does not resolve and there is nothing to compare against.

**This does not weaken the check, it performs it.**  The hash goes in only after the generator
at ``--named_sha256`` -- measured over the checkpoint's OWN named path, on the machine where
that path resolves -- is shown to equal the hash of the local file being adopted.  The
alternative on offer is re-running the fit purely to obtain a string that is already knowable,
and a re-fit would produce different weights.

**It writes a COPY.**  Editing the original in place would leave an artifact whose provenance
says "fitted here" when the stamp was added later and elsewhere, so the copy records both: the
local path it now names, the hash, and how that hash was obtained.

Usage::

    # on the box that fitted it
    ssh talos 'sha256sum /mnt/.../best.pt'
    # here
    python scripts/flow_matching/stamp_critic_generator.py \\
        --critic logs/value/v4/V_priv.fold0.pt \\
        --fm_checkpoint logs/flow_matching/<run>/best.pt \\
        --named_sha256 a596317e... --out logs/value/critic_eval_v4/critics
"""

from __future__ import annotations

import argparse
import hashlib
import os


def sha256_file(path: str) -> str:
    """SHA-256 of a file, streamed.

    Args:
        path: File to hash.

    Returns:
        The lower-case hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit status.

    Raises:
        SystemExit: If the local generator's content does not match the hash measured over the
            checkpoint's own named path. That is the whole check; a mismatch means the two are
            different generators and the critic must not be read against this one.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critic", nargs="+", required=True,
                   help="Critic checkpoint(s) to stamp.")
    p.add_argument("--fm_checkpoint", required=True,
                   help="The LOCAL generator being adopted as the named one.")
    p.add_argument("--named_sha256", required=True,
                   help="SHA-256 of the generator at the path the critic NAMES, measured on "
                        "the machine where that path resolves. This is the evidence; without "
                        "it the stamp would assert a pairing rather than record one.")
    p.add_argument("--out", required=True, help="Directory for the stamped copies.")
    args = p.parse_args()

    import torch  # noqa: PLC0415

    got = sha256_file(args.fm_checkpoint)
    want = args.named_sha256.strip().lower()
    if got != want:
        raise SystemExit(
            f"{args.fm_checkpoint!r} hashes {got[:16]}... but the generator the critic names "
            f"hashes {want[:16]}.... These are different generators: the encoder and all "
            "three normalizers come from it, so adopting this one would give the critic a "
            "silently wrong input space. Nothing is written.")

    os.makedirs(args.out, exist_ok=True)
    for src in args.critic:
        ck = torch.load(src, map_location="cpu", weights_only=False)
        existing = str(ck.get("fm_checkpoint_sha256", "") or "")
        if existing and existing.lower() != want:
            raise SystemExit(
                f"{src!r} already carries sha256 {existing[:16]}..., which disagrees with the "
                f"{want[:16]}... measured over its named path. One of the two is about a "
                "different generator; refusing to overwrite either.")
        ck["fm_checkpoint_sha256"] = want
        ck["fm_checkpoint_named_original"] = str(ck["fm_checkpoint"])
        ck["fm_checkpoint"] = os.path.abspath(args.fm_checkpoint)
        ck["fm_checkpoint_sha256_provenance"] = (
            "measured over the ORIGINAL named path on the machine that fitted this critic, "
            "and verified equal to the adopted local file by "
            "scripts/flow_matching/stamp_critic_generator.py")
        dst = os.path.join(args.out, os.path.basename(src))
        torch.save(ck, dst)
        print(f"[stamp] {src} -> {dst}  (sha {want[:16]}..., was named "
              f"{ck['fm_checkpoint_named_original']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
