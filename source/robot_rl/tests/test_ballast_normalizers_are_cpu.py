# Copyright 2026 Zachary Olkin. All rights reserved.

"""``build_ballast_dataset`` must hand the dataset CPU normalizers, never ``ctx``'s own.

A dataset builds CPU tensors. ``FmOnlineContext``'s normalizers live on the TRAINING device.
Passing the context's straight through therefore dies inside ``Normalizer.normalize`` with::

    RuntimeError: Expected all tensors to be on the same device, but found at least two
    devices, cuda:0 and cpu!

``rollout_dataset._CtxView`` already does this correctly and says why; the ballast path did
not, and the two reasons are worth keeping together:

* the device mismatch above, and
* ``Normalizer.to`` moves **in place**, so cloning is required as well as moving -- without
  the deepcopy the loaded generator is left holding CPU statistics against a CUDA model, which
  fails at the next sample, far from here, looking like a model fault.

**Why a source-level check rather than a run.** The failure is device-CONDITIONAL: it cannot
occur on a CPU-only box, which is exactly how it survived an offline probe of this very
function run under ``CUDA_VISIBLE_DEVICES=""`` -- everything was CPU, so everything matched,
and the check reported the whole ballast path healthy. It then failed on the first real
GPU run, at stage 8 of 8, 64 seconds in. A test that only fails on a CUDA box would have the
same blind spot as the probe did.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_BALLAST = (pathlib.Path(__file__).resolve().parents[3]
            / "scripts" / "flow_matching" / "ballast.py")

#: The keyword arguments that carry a normalizer into the dataset.
_NORMALIZER_KWARGS = ("state_normalizer", "cmd_normalizer", "extero_normalizer")


def _future_state_dataset_call() -> ast.Call:
    """The ``FutureStateDataset(...)`` construction inside ``ballast.py``.

    Returns:
        The call node.
    """
    tree = ast.parse(_BALLAST.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == "FutureStateDataset":
                return node
    pytest.fail("no FutureStateDataset(...) call in ballast.py; has the path moved?")


@pytest.mark.parametrize("kwarg", _NORMALIZER_KWARGS)
def test_no_normalizer_comes_straight_off_the_context(kwarg: str) -> None:
    """``state_normalizer=ctx.state_norm`` is the bug, spelled exactly."""
    call = _future_state_dataset_call()
    passed = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    assert kwarg in passed, f"{kwarg} is no longer passed to FutureStateDataset"
    value = passed[kwarg]
    is_ctx_attr = (isinstance(value, ast.Attribute)
                   and isinstance(value.value, ast.Name)
                   and value.value.id == "ctx")
    assert not is_ctx_attr, (
        f"{kwarg} is passed as ctx.{getattr(value, 'attr', '?')} -- the context's own "
        "normalizer, which lives on the training device. Deep-copy it and move it to the "
        "CPU first, as rollout_dataset._CtxView does.")


def test_the_normalizers_are_deep_copied_and_moved_to_cpu() -> None:
    """The positive half: both halves of the idiom are present.

    ``.to("cpu")`` without ``deepcopy`` would fix the crash and silently break the loaded
    generator instead, so the two are asserted together rather than separately.
    """
    src = _BALLAST.read_text(encoding="utf-8")
    tree = ast.parse(src)
    moved = 0
    for node in ast.walk(tree):
        # match: copy.deepcopy(<anything>).to("cpu")
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "cpu"):
            continue
        inner = node.func.value
        if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "deepcopy"):
            moved += 1
    assert moved >= len(_NORMALIZER_KWARGS), (
        f"found {moved} `copy.deepcopy(...).to(\"cpu\")` expressions in ballast.py, expected "
        f"at least {len(_NORMALIZER_KWARGS)} -- one per normalizer. `Normalizer.to` moves in "
        "place, so the deepcopy is what keeps the loaded generator's own statistics on the "
        "training device.")
