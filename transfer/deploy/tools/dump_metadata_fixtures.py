# Copyright 2026 Zachary Olkin. All rights reserved.

"""Freeze a real ``tracker.onnx``/``fm.onnx``'s metadata, and what python resolves it to.

The C++ port re-implements the metadata parsing, and one specific way of getting that wrong
is invisible: ``observation_terms.policy`` is walked in INSERTION order, and that order **is**
the observation vector's layout. ``nlohmann::json``'s default object type is a ``std::map``,
which sorts keys -- so a C++ port that reaches for the obvious type produces a correctly
sized observation with its terms permuted, and every shape check downstream still passes.

So this writes two files:

* ``tracker_policy_parameters.json`` -- the raw metadata string, verbatim, exactly as
  onnxruntime hands it to a consumer. The C++ parses THIS, not a re-serialised copy, so key
  order on disk is the key order under test.
* ``tracker_expected.json`` -- what the python oracle resolves that string to, in order.

Skips cleanly when the artifacts are absent, matching ``test_hardware_offline.py``'s idiom:
these fixtures are cheap to commit but must not make the suite depend on a policy download.

Run::

    python transfer/deploy/tools/dump_metadata_fixtures.py
    python transfer/deploy/tools/dump_metadata_fixtures.py --run_dir <dir with tracker.onnx>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

#: Where the obelisk stack keeps its prefetched policy pair.
_DEFAULT_RUN_DIR = os.path.join(
    _TRANSFER, "obelisk", "g1_control", "resource", "hf_cache", "zolkin_robot_rl",
    "2026-08-04_21-55-17_yaw_k12_v1")


def resolve_expected(tracker_path: str) -> tuple[str, dict]:
    """Load a tracker artifact and return its raw metadata plus python's resolution.

    Args:
        tracker_path: Path to ``tracker.onnx``.

    Returns:
        ``(raw_policy_parameters_json, expected)``.
    """
    import onnxruntime as ort  # local: the fixture dump is the only thing here needing it

    from deploy.tracker_session import TrackerSession  # noqa: E402

    meta = ort.InferenceSession(
        tracker_path, providers=["CPUExecutionProvider"]).get_modelmeta()
    raw = meta.custom_metadata_map["policy_parameters"]

    trk = TrackerSession(tracker_path)
    expected = {
        "obs_terms": trk.get_obs_terms(),          # ORDER IS THE POINT
        "joint_names_isaac": trk.get_joint_names(),
        "num_obs": trk.get_num_obs(),
        "num_actions": trk.get_num_actions(),
        "dt": trk.get_dt(),
        "kp": list(trk.get_kp()),
        "kd": list(trk.get_kd()),
        "fm_update_every": trk.params.get("fm_update_every"),
        "reference_window_spec": trk.get_reference_window_spec(),
        "command_window_spec": trk.get_command_window_spec(),
        "joint_out_names": trk.get_joint_out_names(),
        "action_scale": [float(v) for v in trk._resolve_action_scale()],
    }
    return raw, expected


def main() -> int:
    """Write the metadata fixtures. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default=_DEFAULT_RUN_DIR,
                    help="directory holding tracker.onnx")
    ap.add_argument("--out", default=os.path.join(_TRANSFER, "deploy", "tests", "fixtures",
                                                  "metadata"))
    args = ap.parse_args()

    tracker = os.path.join(args.run_dir, "tracker.onnx")
    if not os.path.isfile(tracker):
        print(f"[metadata] no tracker.onnx under {args.run_dir}; nothing to dump (this is "
              "not an error -- the C++ metadata test skips when the fixture is absent).")
        return 0

    raw, expected = resolve_expected(tracker)
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "tracker_policy_parameters.json")
    with open(raw_path, "w") as fh:
        fh.write(raw)                       # VERBATIM: do not re-serialise, order is the test
    with open(os.path.join(args.out, "tracker_expected.json"), "w") as fh:
        json.dump(expected, fh, indent=2)

    names = [t["name"] for t in expected["obs_terms"]]
    print(f"[metadata] {len(raw)} bytes of policy_parameters -> {args.out}")
    print(f"[metadata] {len(names)} obs terms, in order: {names}")
    print(f"[metadata] num_obs={expected['num_obs']} num_actions={expected['num_actions']} "
          f"dt={expected['dt']} fm_update_every={expected['fm_update_every']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
