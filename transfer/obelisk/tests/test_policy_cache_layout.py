# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the policy cache layout across the python/C++ boundary.

Every policy lives under a ``policies/`` sub-directory of the HuggingFace repo, and the
local cache mirrors that path. Three places have to agree on it:

* ``deploy.hardware.HF_POLICY_PREFIX`` -- the definition, used by the fetch;
* ``scripts/rsl_rl/export_to_hardware.py`` -- the uploader, which imports it;
* ``g1_control_cpp/src/g1_controller.cc`` -- which CANNOT import it, so it restates it.

The uploader is safe by construction (it imports the constant). The C++ is not, and a
mismatch there is close to undetectable by inspection: the upload succeeds, the python
prefetch succeeds, the python controller runs, and only the C++ controller fails -- at
bring-up, with "no tracker.onnx at <path>", naming a path that looks entirely reasonable.
``hf_repo_id`` already drifted once across exactly this boundary, which is why the repo id
is now a single constant; this is the same lesson applied to the prefix before it bites.

Deliberately reads the C++ as TEXT rather than compiling it. The alternative is exporting
the constant through a generated header, which buys nothing here: there is one literal, it
is a directory name, and a grep that fails loudly is the cheaper guarantee.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

import pytest

OBELISK = pathlib.Path(__file__).resolve().parent.parent
TRANSFER = OBELISK.parent
CONTROLLER_CC = OBELISK / "g1_control_cpp" / "src" / "g1_controller.cc"

sys.path.insert(0, str(TRANSFER))

from deploy.hardware import HF_POLICY_PREFIX, policy_repo_dir  # noqa: E402


def test_policy_repo_dir_prefixes_a_bare_folder() -> None:
    """The helper prepends the prefix and does not double it."""
    assert policy_repo_dir("some_run") == f"{HF_POLICY_PREFIX}/some_run"
    # A caller that already typed the prefix gets it twice rather than silently deduped:
    # the contract is that hf_policy_folder is BARE, and quietly accepting both spellings
    # is how two of them end up in configs meaning the same thing.
    assert policy_repo_dir(f"{HF_POLICY_PREFIX}/x") == f"{HF_POLICY_PREFIX}/{HF_POLICY_PREFIX}/x"


def test_cpp_controller_uses_the_same_prefix() -> None:
    """The C++ cache path restates HF_POLICY_PREFIX, and restates it correctly."""
    src = CONTROLLER_CC.read_text()
    # The cache path is built from a chain of `/`-joined path components ending in
    # hf_policy_folder_. Pull the string literal that immediately precedes it.
    m = re.search(r'"([A-Za-z0-9_\-]+)"\s*/\s*hf_policy_folder_', src)
    assert m is not None, (
        f"{CONTROLLER_CC} no longer builds its cache path as "
        f'... / "<prefix>" / hf_policy_folder_, so this test cannot see the prefix. '
        f"If the path was restructured, update this regex -- do not delete the test."
    )
    assert m.group(1) == HF_POLICY_PREFIX, (
        f"{CONTROLLER_CC} caches policies under '{m.group(1)}/' but "
        f"deploy.hardware.HF_POLICY_PREFIX is '{HF_POLICY_PREFIX}'. The C++ controller "
        f"would look in a directory the fetch never writes to."
    )


def test_fetch_and_cache_hit_agree_on_the_path(tmp_path, monkeypatch) -> None:
    """A cache HIT resolves to the same path a download would have written.

    The hit check and the download used to build their paths separately; if they diverge,
    every run re-downloads on top of a perfectly good cache and nothing says why. Seeding
    the mirrored layout and asserting the fetch is satisfied OFFLINE pins them together.
    """
    from deploy.hardware import ARTIFACT_NAMES, fetch_artifacts

    repo_id, folder = "user/repo", "run_name"
    base = tmp_path / repo_id.replace("/", "_") / HF_POLICY_PREFIX / folder
    base.mkdir(parents=True)
    for name in ARTIFACT_NAMES:
        (base / name).write_bytes(b"")

    # No network: a miss here would raise rather than quietly downloading.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    got = fetch_artifacts(repo_id, folder, str(tmp_path))

    assert [os.path.basename(p) for p in got] == list(ARTIFACT_NAMES)
    for p in got:
        assert os.path.exists(p), f"cache hit resolved to a path that does not exist: {p}"
        assert HF_POLICY_PREFIX in pathlib.Path(p).parts, (
            f"{p} is not under '{HF_POLICY_PREFIX}/', so the cache no longer mirrors the repo"
        )


def test_uploader_imports_the_prefix_rather_than_spelling_it() -> None:
    """The uploader must not carry its own copy of the prefix literal."""
    uploader = TRANSFER.parent / "scripts" / "rsl_rl" / "export_to_hardware.py"
    if not uploader.exists():  # pragma: no cover - the isaaclab-side script may be absent
        pytest.skip(f"{uploader} not present in this checkout")
    src = uploader.read_text()
    assert "policy_repo_dir" in src, (
        "export_to_hardware.py must build its destination with "
        "deploy.hardware.policy_repo_dir, so upload and fetch cannot disagree."
    )
    assert f'"{HF_POLICY_PREFIX}/' not in src and f"'{HF_POLICY_PREFIX}/" not in src, (
        f"export_to_hardware.py spells '{HF_POLICY_PREFIX}/' itself instead of importing "
        f"the constant -- that is the drift this layout exists to prevent."
    )
