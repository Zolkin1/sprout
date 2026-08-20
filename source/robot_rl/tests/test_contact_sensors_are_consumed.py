# Copyright 2026 Zachary Olkin. All rights reserved.

"""A contact sensor that nothing reads is a silent hole, not a spare part.

``backpack_contact`` was declared with a docstring explaining exactly why it had to exist --
``torso_contact`` matches ``torso_link`` with no trailing ``/.*``, so the pack, a child link,
is invisible to it -- and then never added to ``undesired_contacts.sensor_cfgs``. The sensor
initialised, found its body, and reported to nobody: no reward consumed it, and the
termination terms read no contact sensor at all. 2.64 kg of payload protruding 0.164 m behind
the torso collider was unpenalised on a backwards fall.

That is the same failure as ``torso_contact`` being pointed at a body with no collider, which
went unnoticed for months. Both are "the sensor exists, so it must be working".

This pins the general rule rather than the instance: every contact sensor the scene declares
must be read by something. If a new one is deliberately observation-only, add it to
:data:`_INTENTIONALLY_UNREAD` with the reason, so skipping the reward is a decision on the
record instead of an omission.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab", reason="needs the full Isaac install")

from isaaclab.managers import SceneEntityCfg  # noqa: E402

from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import (  # noqa: E402
    ClfTrackingBaseEnvContactSensorCfg, G1ClfTrackingRewardCfg, G1ClfTrackingSceneCfg)

#: Sensors deliberately not wired to any reward, and why.
#:
#: These predate the perceptive robot. They are the shin and foot sensors, kept for a
#: stairs-specific termination that was never landed and for debugging; leaving them here
#: records that, rather than letting the next reader assume they are live.
_INTENTIONALLY_UNREAD: dict[str, str] = {
    "left_shin_contact": "kept for a shin-contact termination on the stairs that is not wired",
    "right_shin_contact": "kept for a shin-contact termination on the stairs that is not wired",
    "left_foot_contact": "feet are supposed to touch the ground; read by diagnostics only",
    "right_foot_contact": "feet are supposed to touch the ground; read by diagnostics only",
}


def _declared_contact_sensors() -> set[str]:
    """Names of every contact-sensor field on the scene config.

    Read off an INSTANCE, not the class: ``@configclass`` turns each field into a dataclass
    field, so ``vars(cls)`` returns none of them and every check here would vacuously pass.
    Matched by TYPE, not by probing for a ``prim_path`` attribute -- these are ``PresetCfg``
    subclasses that do not expose one, so an attribute probe also finds nothing. Both of
    those mistakes were made writing this file, and both are silent: they make the suite
    pass. Hence :func:`test_every_contact_sensor_is_read_by_something` asserting non-empty
    first.
    """
    scene = G1ClfTrackingSceneCfg(num_envs=1, env_spacing=1.0)
    return {name for name, value in vars(scene).items()
            if isinstance(value, ClfTrackingBaseEnvContactSensorCfg)}


def _consumed_sensors() -> set[str]:
    """Names referenced by any reward term's ``sensor_cfgs`` / ``sensor_cfg`` param."""
    out = set()
    for _term_name, term in vars(G1ClfTrackingRewardCfg()).items():
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        for key in ("sensor_cfgs", "sensor_cfg"):
            val = params.get(key)
            if val is None:
                continue
            for cfg in (val if isinstance(val, (list, tuple)) else [val]):
                if isinstance(cfg, SceneEntityCfg):
                    out.add(cfg.name)
    return out


def test_every_contact_sensor_is_read_by_something():
    declared = _declared_contact_sensors()
    assert declared, "found no contact sensors at all -- this test stopped testing anything"
    unread = declared - _consumed_sensors() - set(_INTENTIONALLY_UNREAD)
    assert unread == set(), (
        f"contact sensor(s) {sorted(unread)} are declared but no reward reads them. Either add "
        "them to undesired_contacts.sensor_cfgs or record the reason in _INTENTIONALLY_UNREAD"
    )


def test_the_backpack_is_penalised():
    """The specific hole, pinned: the pack is the largest thing torso_contact cannot see."""
    assert "backpack_contact" in _consumed_sensors(), (
        "the 2.64 kg pack protrudes 0.164 m behind the torso collider; torso_contact matches "
        "torso_link exactly (no trailing /.*) and cannot cover a child link"
    )


def test_the_exemption_list_has_not_gone_stale():
    """An exemption for a sensor that no longer exists hides a later omission."""
    declared = _declared_contact_sensors()
    gone = set(_INTENTIONALLY_UNREAD) - declared
    assert gone == set(), f"_INTENTIONALLY_UNREAD names sensors that no longer exist: {sorted(gone)}"
    both = set(_INTENTIONALLY_UNREAD) & _consumed_sensors()
    assert both == set(), (
        f"{sorted(both)} are listed as intentionally unread but a reward does read them -- "
        "drop them from _INTENTIONALLY_UNREAD"
    )
