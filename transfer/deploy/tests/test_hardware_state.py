# Copyright 2026 Zachary Olkin. All rights reserved.

"""``build_hardware_state``: joint reordering and the exteroception contract.

Artifact-free -- a stub policy stands in for the loaded ONNX pair, so this pins the
argument handling itself: that the estimator's joint order is converted to the isaac
order, and that a policy which reads terrain cannot be stepped without it.  The
end-to-end version, against real artifacts, is ``test_hardware_offline.py``.

The point of the required-ness checks is WHERE they fail.  A missing reading used to
surface as a ``KeyError`` deep inside the observation assembly -- or, for a term the
tracker lacks but the generator wants, as a silently wrong exteroception token.

Run: ``python transfer/deploy/tests/test_hardware_state.py``  (or via pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy.depth_obs import invalid_depth_value, normalize_depth_np  # noqa: E402
from deploy.hardware import ARTIFACT_NAMES, build_hardware_state, fetch_artifacts  # noqa: E402

JOINTS = ["hip", "knee", "ankle"]
CAM = {
    "obs_term": "depth_near", "sensor": "depth_camera_near", "mount_body": "torso_link",
    "offset_pos": [0.1, 0.0, 0.05], "offset_quat_xyzw": [0.0, 0.6, 0.0, 0.8],
    "width": 4, "height": 3, "fx": 4.0, "fy": 4.0, "cx": 2.0, "cy": 1.5,
    "max_distance": 8.0, "offset": 0.75, "scale": 0.5, "min_dist": 0.1, "max_dist": 7.9,
    "min_clip_val": 0.0, "max_clip_val": 0.0,
}


class _Ctx:
    """The FM-context fields ``build_hardware_state`` reads."""

    def __init__(self, num_extra_frames: int = 0):
        self.num_extra_frames = num_extra_frames
        self.frame_names = [f"frame_{i}" for i in range(num_extra_frames)]


class _StubPolicy:
    """A loaded ``GenTrackerPolicy`` reduced to what the state builder asks of it."""

    def __init__(self, depth_spec=None, height=False, num_extra_frames: int = 0):
        self._spec = list(depth_spec or [])
        self._height = height
        self.ctx = _Ctx(num_extra_frames)

    def get_joint_names(self):
        return list(JOINTS)

    def get_depth_camera_spec(self):
        return list(self._spec)

    def needs_height_scan(self):
        return self._height


def _readings(**kwargs) -> dict:
    """Call the builder with plausible proprioception, overriding what the test cares about."""
    base = dict(
        root_pos=np.array([0.0, 0.0, 0.78]),
        base_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        base_lin_vel_w=np.zeros(3),
        base_ang_vel_b=np.zeros(3),
        joint_pos=np.zeros(len(JOINTS)),
        joint_vel=np.zeros(len(JOINTS)),
        joint_names=list(JOINTS),
        command=np.array([0.3, 0.0, 0.0]),
    )
    policy = base.pop("policy", None) or kwargs.pop("policy")
    base.update(kwargs)
    return build_hardware_state(policy, **base)


##
# Proprioception
##

def test_encoder_order_is_converted_to_isaac_order():
    """The estimator's joint order is arbitrary; the policy's is not."""
    state = _readings(
        policy=_StubPolicy(),
        joint_names=list(reversed(JOINTS)),
        joint_pos=np.array([3.0, 2.0, 1.0]),      # ankle, knee, hip
        joint_vel=np.array([30.0, 20.0, 10.0]),
    )
    np.testing.assert_allclose(state["joint_pos"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state["joint_vel"], [10.0, 20.0, 30.0])


def test_world_angular_velocity_is_derived_when_absent():
    """The FM history wants world-frame angular velocity; body-frame is what the IMU gives."""
    # 180 deg about z: a body-frame +x rate reads as world-frame -x.
    state = _readings(
        policy=_StubPolicy(),
        base_quat_xyzw=np.array([0.0, 0.0, 1.0, 0.0]),
        base_ang_vel_b=np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(state["base_ang_vel"], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(state["root_ang_vel_w"], [-1.0, 0.0, 0.0], atol=1e-12)


##
# Exteroception: required-ness comes from the artifacts
##

def test_depth_policy_refuses_to_run_blind():
    """A camera policy handed no image fails HERE, naming the cameras."""
    with pytest.raises(ValueError, match="depth_camera_near"):
        _readings(policy=_StubPolicy(depth_spec=[CAM]))


def test_height_scan_policy_refuses_to_run_blind():
    """Likewise for a scan policy -- including when only the GENERATOR reads it."""
    with pytest.raises(ValueError, match="height scan"):
        _readings(policy=_StubPolicy(height=True))


def test_raw_depth_is_normalised_with_the_camera_contract():
    """The usual hardware path: metres in, the policy's own normalisation applied."""
    raw = np.full(12, 1.5)
    state = _readings(policy=_StubPolicy(depth_spec=[CAM]), depth_raw=[raw])
    assert state["depth"].shape == (1, 12)
    expected = normalize_depth_np(raw, offset=0.75, scale=0.5, min_dist=0.1, max_dist=7.9)
    np.testing.assert_allclose(state["depth"][0], expected)


def test_dropped_pixels_survive_ingestion():
    """A real camera emits NaN; it must reach the policy as the invalid sentinel."""
    raw = np.full(12, 1.5)
    raw[0] = np.nan
    state = _readings(policy=_StubPolicy(depth_spec=[CAM]), depth_raw=[raw])
    assert state["depth"][0][0] == invalid_depth_value(CAM)


def test_prenormalised_depth_is_passed_through():
    """An upstream that already normalised can hand the image over directly."""
    img = np.linspace(-2.0, 2.0, 12)[None]
    state = _readings(policy=_StubPolicy(depth_spec=[CAM]), depth=img)
    np.testing.assert_allclose(state["depth"], img)


def test_depth_and_depth_raw_together_are_refused():
    """Two sources for one image is a wiring bug, not a precedence question."""
    with pytest.raises(ValueError, match="not both"):
        _readings(policy=_StubPolicy(depth_spec=[CAM]),
                  depth=np.zeros((1, 12)), depth_raw=[np.zeros(12)])


def test_depth_for_a_scan_policy_is_refused():
    """Supplying an image the policy cannot read means the wrong artifact is loaded."""
    with pytest.raises(ValueError, match="no depth cameras"):
        _readings(policy=_StubPolicy(height=True), height_scan=np.zeros(176),
                  depth=np.zeros((1, 12)))


def test_scan_policy_accepts_its_reading():
    """The height-scan path stays available alongside depth."""
    scan = np.linspace(-0.3, 0.3, 176)
    state = _readings(policy=_StubPolicy(height=True), height_scan=scan)
    np.testing.assert_allclose(state["height_scan"], scan)
    assert "depth" not in state


##
# Artifact fetch: offline-first
##

def _seed_cache(root, repo="user/repo", folder="run") -> list[str]:
    """Write stand-in artifacts where fetch_artifacts expects to find them."""
    base = os.path.join(root, repo.replace("/", "_"), folder)
    os.makedirs(base, exist_ok=True)
    paths = []
    for name in ARTIFACT_NAMES:
        path = os.path.join(base, name)
        open(path, "wb").write(b"not really onnx")
        paths.append(path)
    return paths


def test_cache_hit_never_touches_the_network(tmp_path, monkeypatch):
    """A warm cache must not call out AT ALL.

    Recent huggingface_hub fires a best-effort telemetry request on every download call --
    even one that hits the cache. Inside the controller's lifecycle on_configure, on a
    robot LAN with no route out, that is a bring-up hang. So the cache hit must short
    circuit before huggingface_hub is reached, which is what this pins.
    """
    expected = _seed_cache(str(tmp_path))

    def explode(*a, **k):
        raise AssertionError("hf_hub_download called on a cache hit")

    monkeypatch.setattr("deploy.hardware._download", explode)
    assert list(fetch_artifacts("user/repo", "run", str(tmp_path))) == expected


def test_refresh_forces_a_download_over_a_cache_hit(tmp_path, monkeypatch):
    """--refresh is how a policy re-pushed to the SAME folder actually reaches the robot."""
    _seed_cache(str(tmp_path))
    calls = []
    monkeypatch.setattr("deploy.hardware._download",
                        lambda repo, filename, local_dir: calls.append(filename) or "/dev/null")
    fetch_artifacts("user/repo", "run", str(tmp_path), refresh=True)
    assert calls == [f"run/{n}" for n in ARTIFACT_NAMES]


def test_missing_artifact_is_fetched(tmp_path, monkeypatch):
    """A cold cache downloads -- only the file that is actually absent."""
    expected = _seed_cache(str(tmp_path))
    os.remove(expected[1])                       # fm.onnx missing, tracker.onnx present
    calls = []
    monkeypatch.setattr("deploy.hardware._download",
                        lambda repo, filename, local_dir: calls.append(filename) or expected[1])
    fetch_artifacts("user/repo", "run", str(tmp_path))
    assert calls == ["run/fm.onnx"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
