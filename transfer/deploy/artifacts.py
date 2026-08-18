# Copyright 2026 Zachary Olkin. All rights reserved.

"""What an exported generator is CALLED, and how a consumer finds it again.

Every generator used to export to ``<run>/exported/fm.onnx``, so exporting a second one
silently replaced the first and the file said nothing about which model it held.  With a
dozen FM runs against one teacher -- depth vs height scan, mlp vs cnn, three pinned-axis
weights -- "the fm.onnx in that directory" stopped identifying anything, and the only way to
tell two apart was the mtime.

So an export is named for the checkpoint it came from (:func:`fm_artifact_name`), several can
sit side by side, and a consumer resolves one by name (:func:`resolve_fm_onnx`), refusing to
guess when a directory holds more than one.

``fm.onnx`` stays meaningful as the DEPLOYMENT name: the robot pulls one pair from a
HuggingFace folder and its controller reads that fixed filename, so ``export_to_hardware.py``
uploads whichever named artifact it was given AS ``fm.onnx``.  The naming exists where models
accumulate (a run's ``exported/``), not where exactly one can be installed.

No torch, no isaaclab -- both the Isaac-side exporters and the off-sim runners import this.
"""

from __future__ import annotations

import glob
import os

#: The deployment filename. One pair is installed on a robot, so this stays a fixed name.
DEPLOY_FM_NAME = "fm.onnx"

#: Suffix marking an exported generator, so a directory listing groups them.
FM_SUFFIX = "_fm.onnx"


def fm_artifact_name(checkpoint: str) -> str:
    """The filename an export of ``checkpoint`` should carry.

    The FM run directory is the identity, because that is what every other artifact of a
    training run is filed under (``params/fm_config.yaml``, ``git/sprout.diff``, wandb).  The
    checkpoint STEM is appended only when it is not the run's ``best``, so the common case
    reads as the run name and an epoch export is still distinguishable from it::

        logs/flow_matching/2026-08-12_17-07-47_scan_pin1/best.pt
            -> 2026-08-12_17-07-47_scan_pin1_fm.onnx
        logs/flow_matching/2026-08-12_17-07-47_scan_pin1/ckpt_60.pt
            -> 2026-08-12_17-07-47_scan_pin1_ckpt_60_fm.onnx

    Args:
        checkpoint: Path to the FM ``.pt``.

    Returns:
        A bare filename (no directory), ending in :data:`FM_SUFFIX`.
    """
    path = os.path.abspath(checkpoint)
    run = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{run}{'' if stem == 'best' else '_' + stem}{FM_SUFFIX}"


def list_fm_onnx(exported_dir: str) -> list[str]:
    """Every exported generator in a directory, newest last.

    Args:
        exported_dir: A run's ``exported`` directory.

    Returns:
        Full paths to each ``*_fm.onnx`` plus a plain ``fm.onnx`` if present, sorted by
        modification time.
    """
    hits = glob.glob(os.path.join(exported_dir, f"*{FM_SUFFIX}"))
    deploy = os.path.join(exported_dir, DEPLOY_FM_NAME)
    if os.path.exists(deploy):
        hits.append(deploy)
    return sorted(set(hits), key=os.path.getmtime)


def resolve_fm_onnx(exported_dir: str, explicit: str | None = None,
                    required: bool = True) -> str | None:
    """The one generator to load out of ``exported_dir``.

    Refuses to choose when the directory holds several, which is the whole point: picking the
    newest would reintroduce, as a silent default, exactly the confusion that naming the
    exports was meant to end.  Name one with ``--fm_onnx``.

    Args:
        exported_dir: A run's ``exported`` directory.
        explicit: A path given on the command line, which always wins. May name a file inside
            ``exported_dir`` or anywhere else.
        required: Raise when nothing is found, rather than returning ``None`` (a tracker-only
            deployment legitimately has no generator).

    Returns:
        The path to load, or ``None`` when nothing is found and ``required`` is False.

    Raises:
        FileNotFoundError: When ``explicit`` does not exist, or when ``required`` and the
            directory holds no generator.
        ValueError: When the directory holds several and none was named.
    """
    if explicit:
        candidate = explicit if os.path.exists(explicit) \
            else os.path.join(exported_dir, explicit)
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"no generator at {explicit!r}"
                + ("" if explicit == candidate else f" or {candidate!r}")
            )
        return candidate
    found = list_fm_onnx(exported_dir)
    if len(found) == 1:
        return found[0]
    if not found:
        if required:
            raise FileNotFoundError(
                f"no generator in {exported_dir}. Export one with play.py --fm_checkpoint, "
                "or scripts/flow_matching/export_fm_onnx.py."
            )
        return None
    raise ValueError(
        f"{exported_dir} holds {len(found)} generators:\n"
        + "\n".join(f"  {os.path.basename(f)}" for f in found)
        + "\nName the one you mean (--fm_onnx). They are kept side by side precisely so "
          "that they do not overwrite each other, so there is no sensible default here."
    )
