# Copyright 2026 Zachary Olkin. All rights reserved.

"""Depth lag is ON by default, and an OLD checkpoint must not inherit that.

Every generator that has driven the robot -- the deployed ``..._lag48_aug`` pair and the
canonical fine-tune parent -- was trained with :data:`DEPTH_LATENCY_PROBS`. The field
nonetheless defaulted to ``None``, and ``run_fm_pipeline.sh`` never named it, so a run that
did not pass the flag trained on fresh frames. That regresses exactly the property the
hardware failure was root-caused to: the generator cannot tolerate a stale depth image, with
the cliff between 20 and 40 ms. A v6g generator was trained and uploaded that way.

**Making it default-on creates a FIFTH instance of this file's oldest trap**, which is what
most of these tests are about. ``extero_source``, ``extero_encoder``, ``attention_mode`` and
``skip_families`` each had to be pinned so a checkpoint written before the field existed does
not silently acquire a decision taken afterwards. A checkpoint with no ``depth_latency_probs``
key predates the lag machinery entirely -- its recording carries no ``depth_lag`` array -- so
it is definitionally a no-latency model and must read as one.

This one REACHES INFERENCE, unlike ``skip_families``: ``gen_tracker_runner``'s ``FmDepthLag``
is derived from this field, so a mis-resolved default would make MuJoCo delay frames for a
model that never trained on delayed ones.
"""

from __future__ import annotations

import pytest

cfgmod = pytest.importorskip("robot_rl.flow_matching.config",
                             reason="the FM config imports torch")

FlowMatchingConfig = cfgmod.FlowMatchingConfig
DEPTH_LATENCY_PROBS = cfgmod.DEPTH_LATENCY_PROBS


def test_the_shipped_tuple_is_what_the_deployed_generators_carry() -> None:
    """Pinned by value: both hardware-proven generators stored exactly this."""
    assert DEPTH_LATENCY_PROBS == (0.0, 0.10, 0.40, 0.50)
    assert DEPTH_LATENCY_PROBS[0] == 0.0, (
        "lag 0 must carry zero weight -- a frame cannot arrive in the step it was captured, "
        "and it is the easiest case, which makes the model optimistic about a signal that is "
        "always late")
    assert sum(DEPTH_LATENCY_PROBS) == pytest.approx(1.0)


def test_a_fresh_config_has_lag_on() -> None:
    """The default itself: a run that names nothing now gets the deployed distribution."""
    assert FlowMatchingConfig().depth_latency_probs == DEPTH_LATENCY_PROBS


def test_a_checkpoint_predating_the_field_reads_as_NO_latency() -> None:
    """The fifth-instance guard. An ABSENT key must not inherit the new default."""
    cfg = FlowMatchingConfig.from_dict({"extero_source": "depth"})
    assert cfg.depth_latency_probs is None, (
        "a checkpoint with no depth_latency_probs key predates the lag machinery and its "
        "recording carries no depth_lag array, so it is definitionally no-latency. Inheriting "
        "the new default would make MuJoCo delay frames for a model never trained on them.")


def test_a_stored_null_still_means_off() -> None:
    """The v6g generators stored ``null`` explicitly; that must survive the default flip."""
    cfg = FlowMatchingConfig.from_dict(
        {"extero_source": "depth", "depth_latency_probs": None})
    assert cfg.depth_latency_probs is None


def test_a_stored_tuple_is_carried_through() -> None:
    """The positive case, so the guard is not simply forcing None everywhere."""
    cfg = FlowMatchingConfig.from_dict(
        {"extero_source": "depth", "depth_latency_probs": [0.0, 0.1, 0.4, 0.5]})
    assert tuple(cfg.depth_latency_probs) == (0.0, 0.1, 0.4, 0.5)


def test_train_py_keeps_an_explicit_off_switch() -> None:
    """Omitting the flag now means LAG, so the ablation needs a spelled word.

    Parsed from the source rather than run, because importing ``train.py`` pulls the whole
    training stack in for a two-line branch.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[3]
           / "scripts" / "flow_matching" / "train.py").read_text(encoding="utf-8")
    assert '("none", "off")' in src, (
        "train.py must accept an explicit off value for --depth_latency_probs: with the "
        "config defaulting to lag, omitting the flag no longer disables it and the "
        "no-latency ablation would be unreachable")
