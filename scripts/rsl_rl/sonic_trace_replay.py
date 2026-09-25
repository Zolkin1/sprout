#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Replay a trace from NVIDIA's env through OUR SONIC adapter and diff, block by block.

The other half of ``sonic_trace_record.py``. Their env produced, per step, both the raw robot
state / reference AND the observations their own code built from them. Here we rebuild those
observations with our adapter from the same inputs and compare. This is the only check that
validates the *semantics* of what we feed SONIC rather than just the layout:

* ``ref joint pos/vel`` -- our future window vs their ``command_multi_future_nonflat``
* ``6D anchor orientation`` -- our :func:`anchor_ori_6d` vs their ``motion_anchor_ori_b_mf_nonflat``
* ``proprioception (930)`` -- our oldest-first history stack vs their ``policy`` group
* ``action`` -- our ONNX fed our vectors vs our ONNX fed vectors assembled from THEIR blocks

The history comparison only becomes meaningful once our ring buffer has seen as many steps as the
window is deep (10), so earlier steps are reported separately rather than silently averaged in.

Runs in the MAIN env (``env_isaaclab``): it needs ``robot_rl.baselines.sonic`` and onnxruntime, not
IsaacLab 2.3.2.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/sonic_trace_replay.py \\
        --trace /tmp/sonic_their_trace.npz \\
        --sonic_dir ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
"""

from __future__ import annotations

import argparse

import numpy as np

from robot_rl.baselines.sonic.obs import (
    HistoryBuffer,
    MotionReference,
    RobotState,
    anchor_ori_6d,
    build_decoder_input,
    build_encoder_input,
    history_depth,
    history_dims,
)
from robot_rl.baselines.sonic.policy import SonicPolicy

NJ = 29


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Their IsaacLab 2.3 is scalar-first; ours is scalar-last."""
    return q[..., [1, 2, 3, 0]]


def _report(name: str, ours: np.ndarray, theirs: np.ndarray, tol: float) -> bool:
    """Print a per-block comparison and return whether it agrees within ``tol``."""
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    if ours.shape != theirs.shape:
        print(f"  {name:34s} SHAPE MISMATCH ours {ours.shape} vs theirs {theirs.shape}")
        return False
    d = np.abs(ours - theirs)
    ok = float(d.max()) <= tol
    worst = int(np.argmax(d.reshape(-1))) if d.size else 0
    print(f"  {name:34s} max|diff| {d.max():.3e}  mean {d.mean():.3e}  "
          f"{'OK' if ok else 'MISMATCH'}"
          + ("" if ok else f"  (worst flat index {worst}: ours {ours.reshape(-1)[worst]:+.5f} "
                          f"theirs {theirs.reshape(-1)[worst]:+.5f})"))
    return ok


def main() -> int:
    """Diff our adapter's observation construction against their env's."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--trace", required=True, help="Trace .npz from sonic_trace_record.py.")
    p.add_argument("--sonic_dir", required=True, help="Released ONNX directory.")
    p.add_argument("--tol", type=float, default=2e-5,
                   help="Tolerance; their env is float32 and ours float64 internally.")
    p.add_argument("--action_tol", type=float, default=1e-3)
    args = p.parse_args()

    d = np.load(args.trace, allow_pickle=True)
    tok_keys = [str(k) for k in d["meta_tok_keys"]]
    n_steps = d["root_quat_w_wxyz"].shape[0]
    frames = int(d["meta_num_future_frames"])
    print(f"[replay] {args.trace}: {n_steps} steps, {frames} future frames, "
          f"dt {float(d['meta_dt'])}")
    print(f"[replay] their tokenizer blocks: {tok_keys}\n")

    pol = SonicPolicy(args.sonic_dir, num_envs=1).load()
    spec = pol.spec
    my_frames, my_step = pol.reference_frames
    if my_frames != frames:
        print(f"[replay] WARNING: our layout wants {my_frames} frames, their trace has {frames}")

    hist = HistoryBuffer(1, history_depth(spec), history_dims(spec))
    hist.reset()
    depth = history_depth(spec)

    results: dict[str, list[bool]] = {}

    def track(name: str, ok: bool) -> None:
        results.setdefault(name, []).append(ok)

    for k in range(n_steps):
        ref = MotionReference(
            joint_pos=d["ref_joint_pos_future"][k].reshape(1, frames, NJ),
            joint_vel=d["ref_joint_vel_future"][k].reshape(1, frames, NJ),
            root_quat=_wxyz_to_xyzw(d["ref_root_quat_future"][k].reshape(1, frames, 4)),
        )
        robot_quat = _wxyz_to_xyzw(d["root_quat_w_wxyz"][k].reshape(1, 4))
        state = RobotState(
            root_quat=robot_quat,
            base_ang_vel=d["root_ang_vel_b"][k].reshape(1, 3),
            joint_pos_rel=(d["joint_pos"][k] - d["default_joint_pos"][k]).reshape(1, NJ),
            joint_vel=d["joint_vel"][k].reshape(1, NJ),
            gravity_dir=d["projected_gravity_b"][k].reshape(1, 3),
            # Their ``last_action`` observation at step k holds the action applied at k-1 (the obs
            # is produced by env.step(k-1)), so the history must be fed the PREVIOUS action. With
            # the zero-action trace this makes no numerical difference, but getting it wrong would
            # silently corrupt any reuse of this script with a non-zero-action trace.
            last_action=(d["action"][k - 1] if k > 0 else np.zeros(NJ)).reshape(1, NJ),
        )
        hist.append(state)

        enc = build_encoder_input(spec, "g1", ref, robot_quat)

        # Their wrapper hands back the tokenizer already flattened (1761). Prepending the mode-id
        # scalar reproduces the released encoder's 1762-wide input exactly, so their vector can be
        # compared to ours slice-for-slice and, better, pushed through the same graph.
        their_enc = None
        if "their_tok_flat" in d:
            flat = d["their_tok_flat"][k].reshape(-1)
            their_enc = np.concatenate(
                [np.asarray([float(spec.modes["g1"].mode_id)], np.float32), flat]
            ).astype(np.float32)[None]
            if their_enc.shape[1] != spec.encoder_dim:
                raise SystemExit(
                    f"[replay] their tokenizer is {flat.size} wide; with the mode id that is "
                    f"{their_enc.shape[1]}, but our layout wants {spec.encoder_dim}."
                )

        if k == 0:
            print("=== step 0: reference blocks (independent of history warm-up) ===")

        # 1+2. The blocks the g1 mode actually reads. Everything else in our encoder vector is
        #      deliberately zero while theirs carries real teleop/SMPL values -- the token check
        #      below is what proves that difference is inert.
        if their_enc is not None:
            for name, label in (("motion_joint_positions_10frame_step5", "ref joint pos"),
                                ("motion_joint_velocities_10frame_step5", "ref joint vel"),
                                ("motion_anchor_orientation_10frame_step5", "anchor_ori_6d")):
                b = spec.encoder_block(name)
                if b is None:
                    continue
                ours, theirs = enc[0, b.offset:b.stop], their_enc[0, b.offset:b.stop]
                ok = _report(f"[{k:02d}] {label}", ours, theirs, args.tol) if k < 3 else \
                    np.abs(ours.astype(np.float64) - theirs.astype(np.float64)).max() <= args.tol
                track(label, bool(ok))

        # 3. The token: our vector vs theirs, through the same encoder. Agreement here means our
        #    reference construction is right AND the zero-filled slices genuinely do not matter.
        tok_ours = pol._run(pol._enc, pol._enc_in, enc, spec.token_dim)      # noqa: SLF001
        if their_enc is not None:
            tok_theirs = pol._run(pol._enc, pol._enc_in, their_enc, spec.token_dim)  # noqa: SLF001
            ok = _report(f"[{k:02d}] TOKEN (64)", tok_ours, tok_theirs, args.tol) if k < 3 else \
                np.abs(tok_ours.astype(np.float64)
                       - tok_theirs.astype(np.float64)).max() <= args.tol
            track("token_64", bool(ok))

        dec = build_decoder_input(spec, tok_ours, hist)

        # 4. proprioception, once our ring buffer is as deep as the window
        if k + 1 >= depth:
            ours = dec[0, spec.decoder_block("token_state").stop:]
            theirs = d["their_policy_obs"][k].reshape(-1)
            ok = _report(f"[{k:02d}] proprio 930", ours, theirs, args.tol) if k < depth + 2 else \
                np.abs(ours - theirs).max() <= args.tol
            track("proprio_930", bool(ok))

            # 5. action: ours end-to-end vs ours with THEIR proprioception spliced in. If the
            #    blocks agree this is redundant; if they do not, it says how much it actually costs.
            a_ours = pol._run(pol._dec, pol._dec_in, dec, 29)                # noqa: SLF001
            dec_theirs = dec.copy()
            dec_theirs[0, spec.decoder_block("token_state").stop:] = theirs
            a_theirs = pol._run(pol._dec, pol._dec_in, dec_theirs, 29)       # noqa: SLF001
            ok = np.abs(a_ours - a_theirs).max() <= args.action_tol
            if k < depth + 2:
                _report(f"[{k:02d}] action (ours vs their-obs)", a_ours, a_theirs,
                        args.action_tol)
            track("action", bool(ok))

    print("\n================ SUMMARY ================")
    all_ok = True
    for name, oks in results.items():
        n_ok = sum(oks)
        verdict = "AGREE" if n_ok == len(oks) else f"DISAGREE on {len(oks) - n_ok}/{len(oks)}"
        print(f"  {name:22s} {n_ok}/{len(oks)} steps within tolerance   -> {verdict}")
        all_ok &= n_ok == len(oks)
    print("\n  " + ("ADAPTER VALIDATED against their env." if all_ok else
                    "ADAPTER DISAGREES with their env -- the failing block names the bug."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
