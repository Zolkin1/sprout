#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Build the small fixed clip library used to debug the tracker WITHOUT the generator.

The question this exists to answer is "is the tracker broken, or is the generator?" -- and
before that, on a new deployment stack, "is anything downstream of the observation wired up
at all?". Driving the tracker from a pre-made clip removes the flow-matching generator from
the loop entirely (it is not even loaded), so a failure is unambiguously the stack.

Four cases, deliberately ordered easiest-first, so a bring-up walks up them:

``stand``
    A settled stand held for the whole clip. The reference never moves, so if the robot
    cannot hold this the fault is in the plumbing -- joint order, gains, frames, the
    estimator -- not in any policy.
``stand_walk_stand``
    The SAME source clip untruncated: the same stand, then a walk, then a stand again.
    Sharing the prefix is the point -- if ``stand`` holds and this one falls, the
    difference is the walk transition and nothing else.
``walk_backward``
    A stand, then a sustained STRAIGHT-BACKWARD walk. Backward locomotion is the mode the
    deployed pair has been seen to stumble in while forward walking looks clean, and it is
    the one direction ``stand_walk_stand`` never exercises (its walk is forward). Because
    the generator is not loaded, a stumble here is the TRACKER; a clean run here that
    stumbles with the generator in the loop is the GENERATOR.
``terrain``
    A stairs-transition clip from the library, which additionally exercises the depth
    cameras and the terrain-pinned anchoring.

.. note::
   ``run_sim_clip.sh`` runs the **C++** controller (``g1_cpp_clip.yml``), the same one the
   deployed generator path uses, so a clip case changes exactly one thing against a normal
   run: where the reference comes from. ``--python`` selects the deprecated controller for a
   bisect; the two are held to 1e-11 by ``deploy_cpp/test/clip_reference_parity_test.cc``,
   so a BEHAVIOURAL difference between them is not in the reference construction.

   The C++ side reads a binary PACK, not this ``.h5``: build it with
   ``transfer/deploy/tools/dump_clip_pack.py`` after running this script.

The cases are SEARCHED for, not hardcoded, so regenerating against a new library keeps
working; the chosen source indices and the source fingerprint are written into the output
attrs and the JSON sidecar so a given library is reproducible and self-describing.

**Where the output lives, and why it is not in ``clips/``.** The debug library is a
DEPLOYMENT artifact, like ``tracker.onnx``: the obelisk container mounts only ``$REPO_ROOT``
and a real robot has no ``clips/`` directory at all, so a library there is unreachable from
the code that needs it. In a git worktree it is worse than unreachable -- ``clips`` is a
SYMLINK out of the tree, so inside the container it dangles and the file simply appears not
to exist. It therefore lands in ``transfer/obelisk/g1_control/resource/debug_clips/``,
alongside ``hf_cache``, which is the established home for "binary the robot needs, not in
git". Both sims read that one file, so they cannot diverge.

Consumers refer to cases by NAME (indices move when the library is rebuilt):

    # MuJoCo sim2sim (--clip resolves against the REPO ROOT, not the cwd)
    cd transfer/sim && python gen_tracker_runner.py --load_run <run> \\
        --clip transfer/obelisk/g1_control/resource/debug_clips/g1_debug_v1.h5 \\
        --clip_case stand --clip_ego_anchor

    # obelisk
    transfer/obelisk/scripts/run_sim_clip.sh --case stand

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/make_debug_clips.py
    ... --src clips/g1_mb_v5_body.h5 --out transfer/obelisk/g1_control/resource/debug_clips/g1_debug_v1.h5
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np

# <repo>/scripts/rsl_rl/make_debug_clips.py -> parents[2] == <repo>. Derived from this
# file's own location rather than a hard-coded hop count relative to a package that could
# move (the CLAUDE.md rule); scripts/ is a stable anchor because it IS the repo layout.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Command magnitudes at or below which a step counts as "standing still" (m/s, rad/s).
#: Not zero: the authored command ramps are smooth, so an exact-zero test would clip the
#: still window a few steps early and leave a small residual command in the stand clip.
STILL_V = 0.05
STILL_W = 0.05

#: Steps to back off from the end of the still window before freezing the stand. The
#: command leads the motion, so the robot is still settling for a few steps after the
#: command starts to ramp; 0.25 s of margin keeps the frozen frame unambiguously at rest.
FREEZE_MARGIN_STEPS = 12

#: Largest base speed (m/s) the frozen frame may carry. Asserted rather than assumed: a
#: freeze onto a moving frame would command a discontinuity, which is exactly the kind of
#: artefact this library exists to rule out.
FROZEN_SPEED_TOL = 0.05

#: ``terrain_type_names`` index for a stairs transition (``clip_io._TERRAIN_TYPE_NAMES``).
TERRAIN_TYPE_TRANSITION = 2

#: Flat sentinel, mirroring ``clip_io._FLAT_TERRAIN_ID``.
FLAT_TERRAIN_ID = -1

#: Commanded ``vx`` (m/s) at or below which a step counts as walking backward. Well clear
#: of :data:`STILL_V` so a clip that merely drifts backward inside a turn cannot qualify.
BACKWARD_VX = -0.35

#: Largest lateral speed / yaw rate (m/s, rad/s) a backward step may carry. The case exists
#: to test ONE direction; a backward-and-turning clip would confound it with a turn.
BACKWARD_LATERAL_TOL = 0.15

#: Settled-stand steps the backward case needs before its walk. The obelisk arm-to-walk
#: handover transient decays within ~0.4 s (measured off the debug bags), so 1 s of stand
#: puts the backward walk unambiguously after it rather than on top of it.
MIN_SETTLE_STEPS = 50

#: Steps the command may take to ramp into the backward walk once the stand ends. Beyond
#: this the clip did something else first, so it is no longer "stand, then straight back".
MAX_RAMP_STEPS = 25

#: Shortest backward run (steps) worth calling a walk. Above this floor the case is scored
#: on SPEED rather than duration: the point is to stress backward locomotion near the
#: command a pilot actually holds, and a long slow reverse is an easier test than a short
#: brisk one. 150 steps is 3 s at dt=0.02 -- several full gait cycles.
MIN_BACKWARD_RUN_STEPS = 150


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def still_mask(target_vel: np.ndarray, clip_len: np.ndarray) -> np.ndarray:
    """Per-step "commanded to stand still" mask, with padding excluded.

    Args:
        target_vel: ``[N, T, 3]`` commanded ``(vx, vy, wz)``.
        clip_len: ``[N]`` real length of each clip (the rest is repeat-last padding).

    Returns:
        ``[N, T]`` boolean, True where the clip is real AND commanded still.
    """
    speed = np.linalg.norm(target_vel[:, :, :2], axis=-1)
    yaw_rate = np.abs(target_vel[:, :, 2])
    real = np.arange(target_vel.shape[1])[None, :] < np.asarray(clip_len)[:, None]
    return (speed <= STILL_V) & (yaw_rate <= STILL_W) & real


def leading_still_steps(still: np.ndarray, clip_len: np.ndarray) -> np.ndarray:
    """Length in steps of each clip's leading still run.

    Args:
        still: ``[N, T]`` from :func:`still_mask`.
        clip_len: ``[N]`` real clip lengths.

    Returns:
        ``[N]`` number of leading steps commanded still.
    """
    # argmin gives the first False; a row that is still throughout has no False, and
    # argmin would wrongly return 0 there, so substitute the real length.
    lead = still.argmin(axis=1)
    return np.where(still.all(axis=1), np.asarray(clip_len), lead)


def select_stand_clip(f: h5py.File, min_walk_speed: float) -> tuple[int, int]:
    """Pick the flat clip with the longest settled stand followed by a real walk.

    Both the ``stand`` and ``stand_walk_stand`` cases come from this one clip, so it has to
    satisfy both: a long leading stand to truncate, and a genuine walk after it.

    Args:
        f: The open source library.
        min_walk_speed: Commanded speed (m/s) the clip must reach after the stand.

    Returns:
        ``(clip_index, still_steps)``.

    Raises:
        SystemExit: If no clip qualifies.
    """
    tv = f["target_vel"][:]
    clip_len = f["clip_len"][:]
    terrain_id = f["terrain_id"][:]
    still = still_mask(tv, clip_len)
    lead = leading_still_steps(still, clip_len)
    speed = np.linalg.norm(tv[:, :, :2], axis=-1)

    # Flat only: the stand must not depend on terrain the debug scene may not have.
    ok = terrain_id == FLAT_TERRAIN_ID
    # A stand long enough to be worth truncating, and a walk after it worth testing.
    ok &= lead > FREEZE_MARGIN_STEPS * 2
    peak_after = np.array([speed[i, int(lead[i]):].max() if lead[i] < speed.shape[1] else 0.0
                           for i in range(speed.shape[0])])
    ok &= peak_after >= min_walk_speed
    if not ok.any():
        sys.exit(
            f"[ERROR] No flat clip has a leading stand plus a walk of >= {min_walk_speed} m/s. "
            "Lower --min_walk_speed, or pass --stand_clip to choose one by hand."
        )
    # Longest stand wins; ties break on the lower index so the choice is deterministic.
    candidates = np.where(ok)[0]
    best = candidates[np.argmax(lead[candidates])]
    return int(best), int(lead[best])


def backward_mask(target_vel: np.ndarray, clip_len: np.ndarray) -> np.ndarray:
    """Per-step "commanded straight backward" mask, with padding excluded.

    Args:
        target_vel: ``[N, T, 3]`` commanded ``(vx, vy, wz)``.
        clip_len: ``[N]`` real length of each clip (the rest is repeat-last padding).

    Returns:
        ``[N, T]`` boolean, True where the clip is real AND commanded straight backward.
    """
    vx, vy, wz = target_vel[..., 0], target_vel[..., 1], target_vel[..., 2]
    real = np.arange(target_vel.shape[1])[None, :] < np.asarray(clip_len)[:, None]
    return ((vx <= BACKWARD_VX) & (np.abs(vy) <= BACKWARD_LATERAL_TOL)
            & (np.abs(wz) <= BACKWARD_LATERAL_TOL) & real)


def base_speed(f: h5py.File) -> np.ndarray:
    """Per-step reference base SPEED for every clip.

    Args:
        f: The open source library.

    Returns:
        ``[N, T]`` speed of the reference root in m/s.
    """
    vel_names = json.loads(f.attrs["vel_output_names"])
    base = [vel_names.index(f"pelvis_link:pos_{a}") for a in ("x", "y", "z")]
    return np.linalg.norm(f["dy_des"][:, :, base], axis=-1)


def settled_step(speed_row: np.ndarray, start: int, stop: int) -> int:
    """First step in ``[start, stop)`` where the reference is actually AT REST.

    A zero COMMAND is not rest. The authored command leads the motion by design, so a clip
    stops being commanded to walk while its root is still coasting -- measured on the shipped
    library, 0.50 m/s six steps after the command reached zero. Freezing there writes an
    instantaneous 0.50 -> 0.00 m/s step into the reference, which is not a motion any robot
    can be asked to follow: it toppled the tracker ~2 s later, and read as a policy failure.

    Args:
        speed_row: ``[T]`` reference base speed for one clip, from :func:`base_speed`.
        start: First step to consider.
        stop: One past the last step to consider.

    The DEEPEST rest in the range wins, not the first frame under tolerance. Both satisfy
    the assertion, but the residual is what gets written into the reference as a step, so
    there is no reason to keep 0.045 m/s of it when 0.000 is available a few frames later.

    Returns:
        The FREEZE step -- one past the settled frame -- or ``-1`` if the reference never
        settles in the range. One past, because ``_frozen_row`` overwrites from the cut
        onward with frame ``cut - 1``, so it is ``cut - 1`` that has to be at rest.
    """
    lo, hi = max(start, 0), max(min(stop, speed_row.shape[0]), 0)
    if hi <= lo:
        return -1
    k = lo + int(np.argmin(speed_row[lo:hi]))
    return k + 1 if speed_row[k] <= FROZEN_SPEED_TOL else -1


def select_backward_clip(f: h5py.File, exclude: tuple[int, ...] = ()) -> tuple[int, int, int]:
    """Pick the flat clip that stands, walks straight backward, then stands again.

    "Stands first" is required, not merely preferred: the case has to start from the same
    settled posture the other cases do, so that a stumble is attributable to the backward
    walk and not to whatever the clip happened to be doing at handover.

    "Stands AFTER" is required for a different reason, and it is a deployment property
    rather than an experimental one. Playback SATURATES on the final real frame rather
    than stopping, and the padding repeats the last real command -- so a case whose last
    real step is still commanding a walk holds that command forever against a reference
    that has stopped moving. The robot is then told to walk while its target stays put,
    and it falls; the fall says nothing about the policy. Freezing the tail onto a
    settled stand (as the ``stand`` case does) makes the case safe to hold indefinitely,
    which matters because obelisk runs until it is stopped while the MuJoCo runner exits
    at the clip's end and never sees the difference.

    Args:
        f: The open source library.
        exclude: Source indices already claimed by another case. ``write_clip_subset``
            rejects duplicates, so a collision would fail the build rather than silently
            reuse a clip.

    Returns:
        ``(clip_index, backward_run_steps, freeze_step)``.

    Raises:
        SystemExit: If no clip qualifies.
    """
    tv = f["target_vel"][:]
    clip_len = f["clip_len"][:]
    terrain_id = f["terrain_id"][:]
    back = backward_mask(tv, clip_len)
    lead = leading_still_steps(still_mask(tv, clip_len), clip_len)

    still = still_mask(tv, clip_len)
    speed = base_speed(f)
    best, best_run, best_cut, best_speed = -1, 0, 0, 0.0
    eligible = (terrain_id == FLAT_TERRAIN_ID) & (lead >= MIN_SETTLE_STEPS)
    for i in np.where(eligible)[0]:
        if int(i) in exclude:
            continue
        start = int(lead[i])
        ramp = back[i, start:start + MAX_RAMP_STEPS + 1]
        if not ramp.any():
            continue          # the first thing after the stand was not a backward walk
        j = start + int(ramp.argmax())
        run = 0
        while j + run < int(clip_len[i]) and back[i, j + run]:
            run += 1
        # A settled stand after the walk, long enough to freeze onto with the usual margin.
        tail = still[i, j + run:int(clip_len[i])]
        if not tail.any():
            continue
        k = j + run + int(tail.argmax())
        hold = 0
        while k + hold < int(clip_len[i]) and still[i, k + hold]:
            hold += 1
        if hold < FREEZE_MARGIN_STEPS * 2:
            continue          # too short to settle before the freeze
        if run < MIN_BACKWARD_RUN_STEPS:
            continue          # a few reverse steps, not a walk
        # Freeze where the REFERENCE IS AT REST, not merely where the command is: see
        # settled_step. Search the whole still window, not a fixed margin into it.
        # Search only the part of the still window that leaves a real hold behind it, so
        # "deepest rest" cannot pick a frame at the very end and freeze onto nothing.
        cut = settled_step(speed[i], k, k + hold - FREEZE_MARGIN_STEPS)
        if cut < 0:
            continue          # never comes to rest early enough in its own stand
        back_speed = -float(tv[i, j:j + run, 0].min())
        if back_speed > best_speed:
            best, best_run, best_cut, best_speed = int(i), run, cut, back_speed
    if best < 0:
        sys.exit(
            "[ERROR] No flat clip stands for >= "
            f"{MIN_SETTLE_STEPS} steps, walks straight backward (vx <= {BACKWARD_VX}) for "
            f">= {MIN_BACKWARD_RUN_STEPS} steps, and then settles into a stand long enough "
            "to freeze onto; pass --backward_clip to choose one by hand."
        )
    return best, best_run, best_cut


def select_terrain_clip(f: h5py.File) -> int:
    """Pick the lowest-indexed stairs-transition clip.

    Args:
        f: The open source library.

    Returns:
        The clip index.

    Raises:
        SystemExit: If the library has no transition clips.
    """
    codes = f["terrain_type_code"][:]
    ids = np.where(codes == TERRAIN_TYPE_TRANSITION)[0]
    if not len(ids):
        sys.exit(
            "[ERROR] The source library has no stairs-transition clips "
            f"(terrain_type_code == {TERRAIN_TYPE_TRANSITION}); pass --terrain_clip."
        )
    return int(ids[0])


# ---------------------------------------------------------------------------
# Building the output
# ---------------------------------------------------------------------------

def _frozen_row(name: str, data: np.ndarray, cut: int) -> np.ndarray:
    """One clip's dataset row with everything from ``cut`` frozen onto frame ``cut - 1``.

    Pose-like channels repeat the last standing frame; rate-like channels go to zero,
    because a frozen pose has, by definition, no velocity and no commanded motion. This has
    to be a DATA edit rather than a ``clip_len`` edit: the deploy-side player
    (``deploy.clip_reference.ClipReference``) never reads ``clip_len`` -- it serves the full
    padded array -- so a length-only truncation would still walk on hardware.

    Per-clip TAGS (``terrain_id``, ``anchor_mask``, ``clip_len``, ...) are 0-dimensional
    per clip rather than time series, so there is nothing to freeze and they pass through.

    Args:
        name: Dataset name.
        data: That dataset's row for one clip -- ``[T, ...]`` for a dense dataset, scalar
            for a per-clip tag.
        cut: First step to overwrite.

    Returns:
        The edited row (a copy).
    """
    row = np.asarray(data).copy()
    if row.ndim == 0:
        return row
    if name in ("dy_des", "target_vel"):
        row[cut:] = 0
    else:
        row[cut:] = row[cut - 1]
    return row


def build_library(src: str, dst: str, stand_clip: int, terrain_clip: int,
                  backward_clip: int, cut: int, backward_cut: int) -> dict:
    """Write the 4-case debug library.

    Built as [stand (truncated copy), stand_walk_stand (the same clip whole), terrain] so
    the first two share a bit-identical prefix and differ only in whether the walk happens.

    ``motion_gen.clips.clip_io.write_clip_subset`` does the real work -- slicing every
    dataset, filtering the terrain manifest and repairing ``generator_config.counts.flat``.
    It rejects duplicate indices (deliberately), so the shared clip is written once and the
    truncated copy is appended here.

    Args:
        src: Source library ``.h5``.
        dst: Output ``.h5`` (overwritten).
        stand_clip: Source index of the stand / stand-walk-stand clip.
        terrain_clip: Source index of the terrain clip.
        backward_clip: Source index of the backward-walk clip.
        cut: Step at which to freeze the stand case.
        backward_cut: Step at which to freeze the backward case's trailing stand, so the
            case is safe to hold once playback saturates on its last frame.

    Returns:
        The case map written to the JSON sidecar.
    """
    from motion_gen.clips.clip_io import write_clip_subset

    tmp = dst + ".subset.tmp"
    # Order here is [stand_walk_stand, terrain, walk_backward]; the truncated stand is
    # prepended below, so the two pre-existing cases keep the indices consumers already
    # recorded and the new one lands last.
    write_clip_subset(src, tmp, [stand_clip, terrain_clip, backward_clip])

    try:
        with h5py.File(tmp, "r") as fi, h5py.File(dst, "w") as fo:
            for key in fi.keys():
                d = fi[key]
                whole = np.asarray(d[0])                       # stand_walk_stand
                rows = [_frozen_row(key, whole, cut), whole, np.asarray(d[1]),
                        _frozen_row(key, np.asarray(d[2]), backward_cut)]
                fo.create_dataset(key, data=np.stack(rows), dtype=d.dtype)
            for key, value in fi.attrs.items():
                fo.attrs[key] = value
            # The added clip is flat, so the manifest is unchanged; only the counts move.
            fo.attrs["num_clips"] = 4
            fo.attrs["num_clips_written"] = 4
            gcfg = json.loads(fo.attrs["generator_config"])
            gcfg.setdefault("counts", {})
            gcfg["counts"]["flat"] = int(gcfg["counts"].get("flat", 0)) + 1
            fo.attrs["generator_config"] = json.dumps(gcfg)
            # A frozen tail IS the intended motion, so none of it is padding.
            clip_len = fo["clip_len"][:]
            clip_len[[0, 3]] = int(json.loads(fo.attrs["num_steps"]))
            fo["clip_len"][...] = clip_len
            fo.attrs["debug_clip_source"] = json.dumps({
                "library": os.path.basename(src),
                "stand_clip": int(stand_clip),
                "terrain_clip": int(terrain_clip),
                "backward_clip": int(backward_clip),
                "backward_freeze_step": int(backward_cut),
                "freeze_step": int(cut),
            })
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return {
        "stand": {"clip_index": 0, "source_clip": int(stand_clip),
                  "note": f"frozen at step {cut}; the reference never moves"},
        "stand_walk_stand": {"clip_index": 1, "source_clip": int(stand_clip),
                             "note": "the same clip untruncated"},
        "terrain": {"clip_index": 2, "source_clip": int(terrain_clip),
                    "note": "stairs transition; exercises depth + terrain anchoring"},
        "walk_backward": {"clip_index": 3, "source_clip": int(backward_clip),
                          "note": f"stand, then a straight-backward walk; frozen at "
                                  f"step {backward_cut} so it is safe to hold"},
    }


def verify(dst: str, cut: int) -> None:
    """Assert the written library says what it is meant to say.

    Cheap, and it catches the two ways this can silently go wrong: a stand case that still
    carries a command (so it is not a stand), and a freeze onto a frame that was still
    moving (so the reference steps discontinuously at the cut).

    Args:
        dst: The written library.
        cut: The freeze step.

    Raises:
        AssertionError: If either invariant fails.
    """
    with h5py.File(dst, "r") as f:
        vel_names = json.loads(f.attrs["vel_output_names"])
        base = [vel_names.index(f"pelvis_link:pos_{a}") for a in ("x", "y", "z")]
        tv, dy, y = f["target_vel"][:], f["dy_des"][:], f["y_des"][:]
        clip_len = f["clip_len"][:]
        f_attrs = dict(f.attrs)

    stand_cmd = np.abs(tv[0]).max()
    assert stand_cmd <= STILL_V, (
        f"the stand case still carries a command of {stand_cmd:.3f} -- the freeze step "
        f"({cut}) is inside the ramp, not before it")
    frozen_speed = float(np.linalg.norm(dy[0, cut - 1, base]))
    assert frozen_speed <= FROZEN_SPEED_TOL, (
        f"froze onto a frame moving at {frozen_speed:.3f} m/s (tol {FROZEN_SPEED_TOL}); "
        "raise FREEZE_MARGIN_STEPS")
    assert np.array_equal(y[0, :cut], y[1, :cut]), \
        "stand and stand_walk_stand must share a bit-identical prefix"
    assert np.abs(dy[0, cut:]).max() == 0.0, "the frozen tail must have zero velocity"

    # The backward case has to actually go backward, and go there straight: a clip that
    # only dips past the threshold for a few steps, or that turns while reversing, would
    # not isolate the direction this case exists to test.
    back_vx = float(tv[3, :, 0].min())
    assert back_vx <= BACKWARD_VX, (
        f"the walk_backward case only reaches vx = {back_vx:.3f} "
        f"(needs <= {BACKWARD_VX}) -- it is not a backward walk")
    back_steps = int(backward_mask(tv[3:4], clip_len[3:4])[0].sum())
    turn = float(np.abs(tv[3, backward_mask(tv[3:4], clip_len[3:4])[0], 2]).max())
    # The tail has to be holdable: zero command and zero reference velocity, or obelisk
    # (which saturates on the last frame and runs until stopped) walks against a static
    # target and falls for reasons that have nothing to do with the policy.
    bcut = int(json.loads(f_attrs["debug_clip_source"])["backward_freeze_step"])
    vel_names2 = json.loads(f_attrs["vel_output_names"])
    base2 = [vel_names2.index(f"pelvis_link:pos_{a}") for a in ("x", "y", "z")]
    back_frozen_speed = float(np.linalg.norm(dy[3, bcut - 1, base2]))
    assert back_frozen_speed <= FROZEN_SPEED_TOL, (
        f"walk_backward froze onto a frame moving at {back_frozen_speed:.3f} m/s "
        f"(tol {FROZEN_SPEED_TOL}) -- the reference would stop instantaneously")
    assert np.abs(tv[3, bcut:]).max() <= STILL_V, (
        f"the walk_backward tail still commands {np.abs(tv[3, bcut:]).max():.3f} after its "
        f"freeze at step {bcut} -- it would be unsafe to hold")
    assert np.abs(dy[3, bcut:]).max() == 0.0, "the walk_backward frozen tail must be at rest"
    print(f"[OK] stand max |command| = {stand_cmd:.4f}, frozen-frame speed = "
          f"{frozen_speed:.4f} m/s, prefix identical through step {cut}")
    print(f"[OK] walk_backward frozen-frame speed = {back_frozen_speed:.4f} m/s")
    print(f"[OK] walk_backward peak vx = {back_vx:.2f} m/s, {back_steps} backward steps "
          f"in total, max |wz| = {turn:.2f} rad/s while reversing")


def main() -> int:
    """Select the cases, write the library and its sidecar, and verify both."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="clips/g1_mb_v5_body.h5",
                   help="Source clip library. Default: the library the current teacher "
                        "trained on, so every debug clip is in-distribution.")
    p.add_argument("--out", default="transfer/obelisk/g1_control/resource/debug_clips/g1_debug_v1.h5",
                   help="Output library .h5. Default is INSIDE the obelisk package's "
                        "resource dir, next to hf_cache, and not in clips/ -- see the "
                        "module docstring for why that matters.")
    p.add_argument("--stand_clip", type=int, default=None,
                   help="Override the searched stand / stand-walk-stand source clip.")
    p.add_argument("--terrain_clip", type=int, default=None,
                   help="Override the searched terrain source clip.")
    p.add_argument("--backward_clip", type=int, default=None,
                   help="Override the searched backward-walk source clip.")
    p.add_argument("--min_walk_speed", type=float, default=0.5,
                   help="Commanded speed (m/s) the stand clip must reach after its stand.")
    args = p.parse_args()

    src = args.src if os.path.isabs(args.src) else str(_REPO_ROOT / args.src)
    dst = args.out if os.path.isabs(args.out) else str(_REPO_ROOT / args.out)
    if not os.path.exists(src):
        sys.exit(f"[ERROR] No source library at {src}")
    # The default output lives in a gitignored directory that therefore does not exist on a
    # fresh clone.
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    with h5py.File(src, "r") as f:
        num_steps = int(json.loads(f.attrs["num_steps"]))
        dt = float(json.loads(f.attrs["dt"]))
        if args.stand_clip is not None:
            stand_clip = args.stand_clip
            still = still_mask(f["target_vel"][stand_clip:stand_clip + 1],
                               f["clip_len"][stand_clip:stand_clip + 1])
            lead = int(leading_still_steps(still, f["clip_len"][stand_clip:stand_clip + 1])[0])
        else:
            stand_clip, lead = select_stand_clip(f, args.min_walk_speed)
        terrain_clip = (args.terrain_clip if args.terrain_clip is not None
                        else select_terrain_clip(f))
        if args.backward_clip is not None:
            backward_clip = args.backward_clip
            back_run = int(backward_mask(f["target_vel"][backward_clip:backward_clip + 1],
                                         f["clip_len"][backward_clip:backward_clip + 1]).sum())
            _s = still_mask(f["target_vel"][backward_clip:backward_clip + 1],
                            f["clip_len"][backward_clip:backward_clip + 1])[0]
            _tail = np.where(_s)[0]
            _tail = _tail[_tail > back_run]
            if not len(_tail):
                sys.exit(f"[ERROR] Clip {backward_clip} never settles after its walk, so "
                         "its tail cannot be frozen and it is unsafe to hold.")
            backward_cut = settled_step(base_speed(f)[backward_clip], int(_tail[0]),
                                        int(f["clip_len"][backward_clip]))
            if backward_cut < 0:
                sys.exit(f"[ERROR] Clip {backward_clip}'s root never comes to rest after its "
                         "walk, so freezing it would write an instantaneous stop.")
        else:
            backward_clip, back_run, backward_cut = select_backward_clip(
                f, exclude=(stand_clip, terrain_clip))

    cut = lead - FREEZE_MARGIN_STEPS
    if cut < 2:
        sys.exit(f"[ERROR] Clip {stand_clip} has only {lead} still steps; too short to freeze.")
    print(f"[INFO] source      {src}")
    print(f"[INFO] stand clip  {stand_clip}: {lead} still steps ({lead * dt:.2f}s), "
          f"freezing at step {cut} ({cut * dt:.2f}s)")
    print(f"[INFO] terrain clip {terrain_clip}")
    print(f"[INFO] backward clip {backward_clip}: {back_run} backward steps "
          f"({back_run * dt:.2f}s), freezing its trailing stand at step {backward_cut} "
          f"({backward_cut * dt:.2f}s)")

    cases = build_library(src, dst, stand_clip, terrain_clip, backward_clip, cut,
                          backward_cut)
    verify(dst, cut)

    sidecar = dst.rsplit(".h5", 1)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump({"source": os.path.basename(src), "num_steps": num_steps, "dt": dt,
                   "freeze_step": cut, "cases": cases}, f, indent=2)
    print(f"[INFO] wrote {dst}\n[INFO] wrote {sidecar}")
    for name, meta in cases.items():
        print(f"         {name:18s} -> clip {meta['clip_index']}  ({meta['note']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
