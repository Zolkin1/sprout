# Copyright 2026 Zachary Olkin. All rights reserved.

"""Replay a stream through the PYTHON runtime and check it reproduces its own outputs.

THE HARNESS'S OWN UNIT TEST, and it has to pass before any C++ number from the same stream
means anything. If the recording is lossy -- a field not captured, a normalisation applied on
the wrong side, an ordering assumption -- then a C++ replay diverges for a reason that has
nothing to do with the C++, and it looks exactly like the bug being hunted.

It exercises RUNG 1 of the comparison ladder: observation assembly, teacher-forced. The two
pieces of state that make teacher-forcing non-optional are supplied from the recording rather
than from the runtime:

* the served reference window, which is otherwise the generator's live output;
* ``prev_action``, because the ``actions`` observation term is the runtime's OWN previous
  network output -- the one autoregressive leak in an otherwise teacher-forced rung.

With both fed, every tick is an INDEPENDENT test: nothing the runtime computes feeds the next
tick's input, so a divergence at tick k says nothing about tick k+1 and the whole distribution
is meaningful. That is why this reports counts and percentiles instead of stopping at the
first mismatch.

The generator is never run here. It is stubbed out, which is the point: this isolates our own
arithmetic from anything a network did.

    python transfer/deploy/tools/replay_selfcheck.py --stream /tmp/replay_tile1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy.depth_obs import normalize_camera_images  # noqa: E402
from deploy.diagnostics import obs_checksum  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.replay_stream import ATOL_EXACT, join_checksum, load  # noqa: E402


def main() -> int:
    """Rebuild every tick's observation from the recording and diff it.

    Returns:
        0 when every tick matches to :data:`deploy.replay_stream.ATOL_EXACT`, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", required=True, help="Directory holding manifest.json.")
    ap.add_argument("--providers", default="cpu",
                    help="onnxruntime providers. CPU by default: this rung runs no network, "
                         "so a GPU session would only slow the load down.")
    args = ap.parse_args()

    arrays, manifest = load(args.stream)
    art = manifest["artifacts"]
    ticks = int(manifest["cases"][0]["attrs"]["ticks"])
    print(f"[selfcheck] {ticks} ticks from {args.stream}")
    print(f"[selfcheck] tracker: {art['tracker_onnx']}")

    policy = GenTrackerPolicy(art["tracker_onnx"], art["fm_onnx"] or None,
                              providers=[p.strip() for p in args.providers.split(",")])
    policy.load()
    spec = policy.get_depth_camera_spec()

    # Teacher forcing: the served window comes from the recording, not from a live plan. The
    # generator is never stepped, so nothing here depends on a network.
    served = {"w": None}
    policy._fm_reference_window = lambda _state: served["w"]   # noqa: SLF001

    n_cams = len(spec)
    px = arrays["depth_raw"].shape[1] // n_cams if n_cams else 0
    worst = 0.0
    bad_ticks: list[int] = []
    checksum_mismatch = 0
    for t in range(ticks):
        raw = arrays["depth_raw"][t].reshape(n_cams, px) if n_cams else None
        state = {
            "root_pos": arrays["root_pos"][t],
            "root_quat": arrays["base_quat_xyzw"][t],
            "base_ang_vel": arrays["base_ang_vel_b"][t],
            "joint_pos": arrays["joint_pos"][t],
            "joint_vel": arrays["joint_vel"][t],
            "command": arrays["command"][t],
        }
        if n_cams:
            # Normalised HERE from the raw metres, exactly as the C++ runtime does. If this
            # step were wrong the whole comparison would be measuring the recording.
            state["depth"] = normalize_camera_images(spec, [raw[i] for i in range(n_cams)])
        if arrays["height_scan"].shape[1]:
            state["height_scan"] = arrays["height_scan"][t]

        served["w"] = arrays["window"][t]
        policy._trk.action_isaac = arrays["prev_action"][t].copy()   # noqa: SLF001
        obs = policy._build_obs(state, reference=None).ravel()       # noqa: SLF001

        err = float(np.abs(obs - arrays["obs"][t]).max())
        worst = max(worst, err)
        if err > ATOL_EXACT:
            bad_ticks.append(t)
        recorded_sum = join_checksum(arrays["obs_checksum_hi"][t][0],
                                     arrays["obs_checksum_lo"][t][0])
        if obs_checksum(obs.astype(np.float32)) != recorded_sum:
            checksum_mismatch += 1

    print(f"[selfcheck] worst |d obs| = {worst:.3e}  (tolerance {ATOL_EXACT:.0e})")
    print(f"[selfcheck] ticks over tolerance: {len(bad_ticks)}/{ticks}"
          + (f"  first at {bad_ticks[0]}" if bad_ticks else ""))
    print(f"[selfcheck] checksum mismatches: {checksum_mismatch}/{ticks}")
    if bad_ticks or checksum_mismatch:
        print("[selfcheck] FAIL -- the recording does not reproduce its own outputs, so no "
              "C++ comparison against this stream is meaningful yet.")
        return 1
    print("[selfcheck] PASS -- the stream is a faithful, lossless record of the tick inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
