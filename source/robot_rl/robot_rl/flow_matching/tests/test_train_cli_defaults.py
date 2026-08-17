# Copyright 2026 Zachary Olkin. All rights reserved.

"""``train.py`` with no flags must reproduce ``FlowMatchingConfig()`` exactly.

An argparse default is a SECOND place a default lives, and the two drift silently. It
happened: ``--extero_cnn_strides`` declared ``default=""``, which parsed to ``None`` and
overrode the ``(1, 2, 2)`` the dataclass had just been given, so a run meant to be 56 tokens
per camera trained at 16 -- sequence 111 rather than 191, three quarters of every depth image
discarded before the first non-linearity. Nothing failed. The config that got saved was
self-consistent, the checkpoint reloaded, and only the token count in the run banner said
otherwise.

Nothing in the suite could have caught it, either, and that is the point of this file: every
other test builds ``FlowMatchingConfig`` directly, so the argparse path -- the one that
actually trains -- was exercised by no test at all.

The check is generic rather than per-flag. It walks ``parse_args`` with only the required
``--data`` and diffs the whole dataclass against a default-constructed one, so a field added
years from now is covered the day someone gives it a CLI flag.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("motion_gen")

from motion_gen.features import _REPO_ROOT  # noqa: E402

from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402

# Off the package root, never a count of ".." hops from this file -- moving a module has
# silently broken such a path before (see CLAUDE.md). Needed so train.py can reach its
# siblings; the module itself is loaded by PATH below, not from here.
_FM_SCRIPTS = os.path.join(str(_REPO_ROOT), "scripts", "flow_matching")
sys.path.insert(0, _FM_SCRIPTS)


def _train_module():
    """``scripts/flow_matching/train.py``, loaded under a name that cannot collide.

    NOT ``import train``. The repo has TWO ``train.py`` -- this one and ``scripts/rsl_rl``'s
    RL entry point -- so a bare import resolves to whichever landed in ``sys.modules`` first.
    Alone this file passed; in the full suite the RL one had already been imported and every
    test here died on ``module 'train' has no attribute 'parse_args'``. Loading by path under
    a unique key is order-independent, and leaves the plain ``train`` name alone for whoever
    else is relying on it.

    Returns:
        The imported module.
    """
    name = "_fm_train_under_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(_FM_SCRIPTS,
                                                                        "train.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


#: Fields whose CLI default is ALLOWED to differ from the dataclass, each with the reason.
#:
#: Both entries are ``"auto"`` sentinels that resolve against the RECORDING rather than
#: silently overriding anything -- 'auto' target_source becomes ego_anchored_clip_future when
#: the recording carries ref_future, and 'auto' cmd_array becomes target_vel if that array is
#: present. A default that depends on the data cannot live in a dataclass default, so the
#: divergence is real but intended. (The tidier form is the one ``extero_encoder`` uses: put
#: the sentinel in the FIELD and resolve it in the dataset, which would empty this dict. That
#: is a behaviour change to the two fields, not a docstring fix, so it is not done here.)
#:
#: Anything else that shows up belongs here with its justification, or gets fixed -- the value
#: of this test is that the list stays short enough to read.
_ALLOWED_DIVERGENCE: dict[str, str] = {
    "target_source": "'auto' resolves to ego_anchored_clip_future from the recording",
    "cmd_array": "'auto' resolves to target_vel if the recording has it, else no command",
}


def _cfg_from_bare_cli() -> FlowMatchingConfig:
    """The config ``train.py`` builds when given nothing but ``--data``.

    Returns:
        The parsed :class:`FlowMatchingConfig`.
    """
    # A path that does not exist: parse_args must not touch the data, and if it ever starts
    # to, this test should be what says so.
    with mock.patch.object(sys, "argv", ["train.py", "--data", "does_not_exist.h5"]):
        cfg, _args = _train_module().parse_args()
    return cfg


def test_the_cli_does_not_shadow_any_config_default():
    """No flag, no change: every field must match a default-constructed config."""
    cli, ref = _cfg_from_bare_cli(), FlowMatchingConfig()
    diffs = {
        f.name: (getattr(ref, f.name), getattr(cli, f.name))
        for f in dataclasses.fields(ref)
        if f.name not in _ALLOWED_DIVERGENCE
        and getattr(cli, f.name) != getattr(ref, f.name)
    }
    assert not diffs, (
        "train.py's argparse defaults disagree with FlowMatchingConfig, so the documented "
        "default is unreachable from the command line:\n"
        + "\n".join(f"  {k}: config={v[0]!r} but CLI gives {v[1]!r}" for k, v in diffs.items())
        + "\nRender the flag's default from the dataclass (see --cmd_scale) rather than "
          "restating it."
    )


def test_the_shipped_depth_architecture_survives_the_command_line():
    """The specific numbers the regression changed, asserted end to end through argparse.

    The generic test above would catch a recurrence, but only as an inequality. These are the
    values a run banner should print, so a failure here says what the architecture became.
    """
    cfg = _cfg_from_bare_cli()
    assert cfg.extero_source == "depth"
    assert cfg.resolve_extero_encoder() == "cnn"
    assert cfg.resolve_extero_channels() == 5
    assert list(cfg.extero_cnn_strides) == [1, 2, 2]
    assert cfg.extero_norm_mode == "contract"
    assert cfg.resolve_extero_clip() == pytest.approx(5.575)


def test_the_declared_xyz_constants_are_settable_from_the_command_line():
    """The contract constants must be reachable, or the remedy for a bad fit is a source edit.

    The dataset PRINTS declared-vs-measured per camera, and the documented response to a large
    residual is to state a better constant. That is only true if one can be stated -- and a
    constant passed here lands in the run's saved config, where a source edit would not.
    """
    argv = ["train.py", "--data", "x.h5", "--extero_xyz_scale", "2.5",
            "--extero_nominal_ground_z", "-0.9",
            "--extero_xyz_offset", "0.55,0,-0.79;1.8,0,-0.79"]
    with mock.patch.object(sys, "argv", argv):
        cfg, _ = _train_module().parse_args()
    assert cfg.extero_xyz_scale == pytest.approx(2.5)
    assert cfg.extero_nominal_ground_z == pytest.approx(-0.9)
    assert cfg.extero_xyz_offset == [[0.55, 0.0, -0.79], [1.8, 0.0, -0.79]]


def test_an_unstated_xyz_offset_stays_None_rather_than_empty():
    """``None`` means DERIVE; ``[]`` would mean "zero cameras stated" and fail a count check."""
    assert _cfg_from_bare_cli().extero_xyz_offset is None
