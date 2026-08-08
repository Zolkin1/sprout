# Copyright 2026 Zachary Olkin. All rights reserved.

"""Emit the cross-language parity fixtures the C++ deploy port is checked against.

The python runtime stays the oracle (see ``transfer/deploy/``); this writes down what it
computes so a C++ build with no python in it can be held to the same numbers.

Format is deliberately dumb: one raw little-endian ``float64`` blob per array plus a JSON
manifest. NOT ``.npz`` -- reading that from C++ needs a zip reader and an npy parser, which
is real code that can itself be wrong, in the one place where being wrong looks like a
passing test. A blob is an ``ifstream`` and a ``resize``.

The manifest records the shape, dtype and sha256 of every blob, plus the numpy version,
platform and seed that produced them, so a stale or half-regenerated fixture set is
detected rather than silently compared against.

Both target architectures are little-endian; that is asserted here rather than assumed.

Run (from the repo root, in ``env_transfer`` or ``env_isaaclab``)::

    python transfer/deploy/tools/dump_parity_fixtures.py
    python transfer/deploy/tools/dump_parity_fixtures.py --out /tmp/fixtures --count 2048

The primitive fixtures this writes are small (~1 MB) and are meant to be committed. The
episode-scale fixtures are generated on demand from a recorded bag and are not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from typing import Any, Callable

import numpy as np

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy import depth_obs as do  # noqa: E402
from deploy import quat_math_np as qn  # noqa: E402
from deploy import window_np as wn  # noqa: E402

#: Manifest format version. Bump when the on-disk layout changes so an old C++ reader
#: refuses a new fixture set instead of misreading it.
MANIFEST_FORMAT = 1

#: Absolute tolerance per function, and how to compare. Lives HERE, travels in the
#: manifest, and is read by the C++ side -- one number per function in one place, so the
#: two languages cannot drift on what "agrees" means.
#:
#: The values are not uniform because the disagreement mechanisms are not:
#:
#: * 1e-14 -- pure algebra, four-term sums at most, identical operation order. The C++ is
#:   transcribed expression for expression from the numpy and built with -ffp-contract=off,
#:   so there is nothing left to differ except the last bit.
#: * 1e-13 -- one sqrt (correctly rounded in both) plus a length-3 dot: Gram-Schmidt.
#: * 1e-12 -- anything through atan2/sin/cos/asin. numpy may route these through SVML on
#:   x86 while calling libm on aarch64, so this is the one row whose slack is really about
#:   the host, not the algorithm.
#: * 1e-9 with `quat_rotation` -- quat_from_matrix picks among four candidates by argmax.
#:   Near a pivot tie the two languages can select different candidates; they are the same
#:   ROTATION to ~1e-8 but differ in sign pattern, so comparing componentwise is simply the
#:   wrong question. test_deploy_parity.py already sets this precedent (atol=1e-8, "asR").
#: * 1e-11 -- composites of the above evaluated at robot scale (metres, m/s).
TOLERANCES: dict[str, tuple[float, str]] = {
    "quat_mul": (1e-14, "elementwise"),
    "quat_conjugate": (1e-14, "elementwise"),
    "quat_inv": (1e-14, "elementwise"),
    "quat_apply": (1e-14, "elementwise"),
    "quat_apply_inverse": (1e-14, "elementwise"),
    "quat_unique": (1e-14, "elementwise"),
    "matrix_from_quat": (1e-14, "elementwise"),
    "matrix_to_rot6d": (1e-14, "elementwise"),
    "quat_to_rot6d": (1e-14, "elementwise"),
    "rot6d_to_matrix": (1e-13, "elementwise"),
    "combine_frame_transforms": (1e-13, "elementwise"),
    "subtract_frame_transforms": (1e-13, "elementwise"),
    "yaw_quat": (1e-12, "elementwise"),
    "axis_angle_from_quat": (1e-12, "elementwise"),
    "quat_from_axis_angle": (1e-12, "elementwise"),
    "quat_box_plus": (1e-12, "elementwise"),
    "quat_box_minus": (1e-12, "elementwise"),
    "quat_from_matrix": (1e-9, "quat_rotation"),
    "rot6d_to_quat": (1e-9, "quat_rotation"),
    "world_to_window": (1e-11, "elementwise"),
    "reanchor_window": (1e-11, "elementwise"),
    # EXACT. No reductions, no transcendentals -- just comparisons and one affine -- so a
    # single ulp of disagreement would be a real behavioural difference, not arithmetic
    # noise. test_depth_obs.py already holds the python to exact equality against its torch
    # original; this inherits that rather than loosening it.
    "normalize_depth": (0.0, "elementwise"),
    "sanitize_extero": (0.0, "elementwise"),
}


class FixtureWriter:
    """Collects named arrays into blobs plus a manifest.

    Args:
        out_dir: Directory to write ``manifest.json`` and the ``.bin`` blobs into.
    """

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.cases: list[dict[str, Any]] = []
        os.makedirs(out_dir, exist_ok=True)

    def _blob(self, case: str, role: str, name: str, value: np.ndarray) -> dict[str, Any]:
        """Write one array and return its manifest entry.

        Args:
            case: Case name, used in the file name.
            role: ``"in"`` or ``"out"``.
            name: Argument or output name.
            value: The array; cast to float64 unless already integral.

        Returns:
            The manifest entry describing the blob.
        """
        arr = np.ascontiguousarray(value, dtype=np.float64)
        rel = f"{case}.{role}.{name}.bin"
        raw = arr.tobytes(order="C")
        with open(os.path.join(self.out_dir, rel), "wb") as fh:
            fh.write(raw)
        return {
            "name": name,
            "file": rel,
            "shape": list(arr.shape),
            "dtype": "float64",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def add(self, case: str, fn: str, inputs: dict[str, np.ndarray],
            outputs: dict[str, np.ndarray], attrs: dict[str, Any] | None = None) -> None:
        """Record one function's inputs and the oracle's outputs.

        Args:
            case: Unique case name.
            fn: The function under test, as the C++ side names it.
            inputs: Argument name -> array, in the order the function takes them.
            outputs: Output name -> array.
            attrs: Scalar/structural parameters the C++ side needs (block lists, joint
                counts). Carried verbatim in the manifest.

        Raises:
            KeyError: If ``fn`` has no entry in :data:`TOLERANCES`. Deliberate -- a new
                function must be given an explicit tolerance rather than inheriting a
                default that nobody chose.
        """
        atol, compare = TOLERANCES[fn]
        self.cases.append({
            "case": case,
            "fn": fn,
            "atol": atol,
            "compare": compare,
            "attrs": attrs or {},
            "inputs": [self._blob(case, "in", k, v) for k, v in inputs.items()],
            "outputs": [self._blob(case, "out", k, v) for k, v in outputs.items()],
        })

    def write_manifest(self, seed: int) -> str:
        """Write ``manifest.json`` and return its path."""
        path = os.path.join(self.out_dir, "manifest.json")
        with open(path, "w") as fh:
            json.dump({
                "format": MANIFEST_FORMAT,
                "generator": "transfer/deploy/tools/dump_parity_fixtures.py",
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": f"{platform.system()}-{platform.machine()}",
                "byteorder": sys.byteorder,
                "seed": seed,
                "cases": self.cases,
            }, fh, indent=2)
        return path


def _rand_quat(rng: np.random.Generator, n: int) -> np.ndarray:
    """``[n, 4]`` uniformly random unit quaternions (xyzw)."""
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _adversarial_quats() -> np.ndarray:
    """Quaternions that break naive implementations, as ``[m, 4]`` (xyzw).

    Covers the cases a uniform random draw essentially never produces: the identity and its
    negation (``quat_unique`` / ``w >= 0`` canonicalisation), rotations by almost exactly pi
    about each axis (where ``quat_from_matrix``'s four-candidate argmax changes pivot and
    ``axis_angle_from_quat``'s small-angle branch is furthest from its expansion), and
    near-identity rotations (where that same branch is closest).
    """
    eps = 1e-8
    out = [
        [0.0, 0.0, 0.0, 1.0],                       # identity
        [0.0, 0.0, 0.0, -1.0],                      # identity, negated
        [1.0, 0.0, 0.0, 0.0],                       # pi about x
        [0.0, 1.0, 0.0, 0.0],                       # pi about y
        [0.0, 0.0, 1.0, 0.0],                       # pi about z
        [eps, 0.0, 0.0, 1.0],                       # near identity
        [0.0, eps, 0.0, 1.0],
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],     # pi about the xy diagonal: pivot tie
        [np.sqrt(1.0 / 3), np.sqrt(1.0 / 3), np.sqrt(1.0 / 3), 0.0],  # three-way tie
        [1.0 - 1e-12, 0.0, 0.0, np.sqrt(2e-12)],    # just short of pi
    ]
    arr = np.asarray(out, dtype=np.float64)
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


def dump_quat_math(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit fixtures for every ``quat_math_np`` entry point the C++ port implements."""
    q_rand = _rand_quat(rng, n)
    q_adv = _adversarial_quats()
    q = np.concatenate([q_rand, q_adv], axis=0)
    p = np.concatenate([_rand_quat(rng, n), q_adv], axis=0)
    # Realistic magnitudes: metres for positions, m/s and rad/s for velocities. Parity at
    # 1e-14 on unit-scale inputs can hide an error that only shows at robot scale.
    v = rng.standard_normal((q.shape[0], 3)) * 2.0
    t = rng.standard_normal((q.shape[0], 3)) * 5.0

    unary: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "quat_conjugate": qn.quat_conjugate,
        "quat_inv": qn.quat_inv,
        "yaw_quat": qn.yaw_quat,
        "quat_unique": qn.quat_unique,
        "matrix_from_quat": qn.matrix_from_quat,
        "axis_angle_from_quat": qn.axis_angle_from_quat,
    }
    for name, fn in unary.items():
        w.add(name, name, {"q": q}, {"result": fn(q)})

    w.add("quat_mul", "quat_mul", {"a": q, "b": p}, {"result": qn.quat_mul(q, p)})
    w.add("quat_apply", "quat_apply", {"q": q, "v": v}, {"result": qn.quat_apply(q, v)})
    w.add("quat_apply_inverse", "quat_apply_inverse", {"q": q, "v": v},
          {"result": qn.quat_apply_inverse(q, v)})
    w.add("quat_box_minus", "quat_box_minus", {"a": q, "b": p},
          {"result": qn.quat_box_minus(q, p)})
    w.add("quat_box_plus", "quat_box_plus", {"q": q, "delta": v},
          {"result": qn.quat_box_plus(q, v)})
    w.add("quat_from_axis_angle", "quat_from_axis_angle", {"delta": v},
          {"result": qn.quat_from_axis_angle(v)})

    # quat_from_matrix is fed matrices built from the SAME quaternions, so the expected
    # answer is a known rotation. Compared as a rotation, not componentwise -- near a pivot
    # tie the two languages can pick different candidates of the four, which are the same
    # rotation to ~1e-8 but differ in sign pattern.
    w.add("quat_from_matrix", "quat_from_matrix", {"matrix": qn.matrix_from_quat(q)},
          {"result": qn.quat_from_matrix(qn.matrix_from_quat(q))})

    p01, q01, p12, q12 = t, q, v, p
    cp, cq = qn.combine_frame_transforms(p01, q01, p12, q12)
    w.add("combine_frame_transforms", "combine_frame_transforms",
          {"p01": p01, "q01": q01, "p12": p12, "q12": q12}, {"pos": cp, "quat": cq})
    sp, sq = qn.subtract_frame_transforms(p01, q01, t + v, p)
    w.add("subtract_frame_transforms", "subtract_frame_transforms",
          {"p01": p01, "q01": q01, "p02": t + v, "q02": p}, {"pos": sp, "quat": sq})


def dump_window(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit fixtures for the rot6d helpers, ``world_to_window`` and ``reanchor_window``."""
    q = np.concatenate([_rand_quat(rng, n), _adversarial_quats()], axis=0)
    m = qn.matrix_from_quat(q)
    w.add("matrix_to_rot6d", "matrix_to_rot6d", {"matrix": m}, {"result": wn.matrix_to_rot6d(m)})
    w.add("quat_to_rot6d", "quat_to_rot6d", {"q": q}, {"result": wn.quat_to_rot6d(q)})

    # rot6d_to_matrix must survive raw generator output, which is only APPROXIMATELY a
    # rotation -- that is the whole reason it Gram-Schmidts. So perturb, and include a
    # zero row to exercise the 1e-8 norm guard that stops it dividing by zero.
    d6 = wn.matrix_to_rot6d(m) + rng.standard_normal((m.shape[0], 6)) * 0.05
    d6 = np.concatenate([d6, np.zeros((1, 6))], axis=0)
    w.add("rot6d_to_matrix", "rot6d_to_matrix", {"d6": d6}, {"result": wn.rot6d_to_matrix(d6)})
    w.add("rot6d_to_quat", "rot6d_to_quat", {"d6": d6}, {"result": wn.rot6d_to_quat(d6)})

    # The live artifact's layout: 29 joints, no ee frames, base_pose + base_vel + joint_pos
    # (D = 44). Also a full five-block case, so a future policy that carries joint_vel and
    # ee_pos is covered before it exists rather than after it breaks.
    for tag, blocks, nj, nee in (
        ("slim", ("base_pose", "base_vel", "joint_pos"), 29, 0),
        ("full", None, 21, 4),
    ):
        k = 13                                   # the artifact serves 13 window offsets
        root_pos = rng.standard_normal((1, 3)) * 3.0
        root_quat = _rand_quat(rng, 1)
        ref_pos = rng.standard_normal((k, 3)) * 3.0
        ref_quat = _rand_quat(rng, k)
        ref_lin = rng.standard_normal((k, 3))
        ref_ang = rng.standard_normal((k, 3))
        joint_pos = rng.standard_normal((k, nj))
        joint_vel = rng.standard_normal((k, nj))
        ee_pos = rng.standard_normal((k, max(nee, 1), 3))
        window = wn.world_to_window(root_pos, root_quat, ref_pos, ref_quat, ref_lin,
                                    ref_ang, joint_pos, joint_vel, ee_pos, blocks)
        w.add(f"world_to_window_{tag}", "world_to_window", {
            "root_pos_w": root_pos, "root_quat_w": root_quat,
            "ref_pos_w": ref_pos, "ref_quat_w": ref_quat,
            "ref_lin_w": ref_lin, "ref_ang_w": ref_ang,
            "joint_pos": joint_pos, "joint_vel": joint_vel, "ee_pos_w": ee_pos,
        }, {"window": window}, attrs={
            "blocks": list(wn.resolve_reference_blocks(blocks)),
            "num_joints": nj, "num_ee_frames": nee, "num_nodes": k,
        })

        # Re-anchor onto a pose a plausible K-1 steps of egomotion away. Also emit the
        # same-anchor case: reanchor onto the anchor it is already in must be the identity
        # to ~2e-15, which is the cheapest possible check that the round trip is not
        # accumulating error.
        new_pos = root_pos + rng.standard_normal((1, 3)) * 0.2
        new_quat = _rand_quat(rng, 1)
        w.add(f"reanchor_window_{tag}", "reanchor_window", {
            "window": window,
            "old_pos_w": root_pos, "old_quat_w": root_quat,
            "new_pos_w": new_pos, "new_quat_w": new_quat,
        }, {"window": wn.reanchor_window(window, root_pos, root_quat, new_pos, new_quat,
                                         nj, nee, blocks)},
            attrs={"blocks": list(wn.resolve_reference_blocks(blocks)),
                   "num_joints": nj, "num_ee_frames": nee})
        w.add(f"reanchor_window_identity_{tag}", "reanchor_window", {
            "window": window,
            "old_pos_w": root_pos, "old_quat_w": root_quat,
            "new_pos_w": root_pos, "new_quat_w": root_quat,
        }, {"window": wn.reanchor_window(window, root_pos, root_quat, root_pos, root_quat,
                                         nj, nee, blocks)},
            attrs={"blocks": list(wn.resolve_reference_blocks(blocks)),
                   "num_joints": nj, "num_ee_frames": nee})


def dump_depth(w: FixtureWriter, rng: np.random.Generator, n: int) -> None:
    """Emit depth-normalization and exteroception-sanitising fixtures.

    Weighted towards the values that distinguish a correct port from a plausible one. The
    ordering inside ``normalize_depth_np`` is the whole game: the too-near test runs first,
    so ``-inf`` becomes ``min_clip_val``, while ``+inf`` and ``NaN`` fall through to the
    non-finite test and become ``max_clip_val``. An implementation that reorders those two
    lines, or that uses non-strict comparisons, passes on random inputs and fails only on
    the readings a real camera actually produces at the edge of its range.
    """
    cam = {"offset": 0.75, "scale": 0.5, "min_dist": 0.2, "max_dist": 9.0,
           "min_clip_val": 0.0, "max_clip_val": 0.0}
    edge = np.array([
        -np.inf, np.inf, np.nan,
        cam["min_dist"], cam["max_dist"],                       # exactly at: still VALID
        np.nextafter(cam["min_dist"], -np.inf),                 # one ulp below: invalid
        np.nextafter(cam["max_dist"], np.inf),                  # one ulp above: invalid
        0.0, -1.0, 1e-30, 1e30,
        cam["offset"],                                          # normalises to exactly 0
    ], dtype=np.float64)
    d = np.concatenate([rng.uniform(-0.5, 10.0, size=n), edge])[:, None]
    w.add("normalize_depth", "normalize_depth", {"d": d},
          {"result": mdp_normalize(d, cam)}, attrs=dict(cam))

    # A second camera contract with non-zero clip substitutions, so a port that hardcodes
    # the 0.0 defaults is caught. Those values are what a real spec carries when "no
    # reading" has to be distinguishable from a legitimate depth of `offset` metres.
    cam2 = {"offset": 1.0, "scale": 0.4, "min_dist": 0.3, "max_dist": 6.0,
            "min_clip_val": 0.25, "max_clip_val": 7.5}
    w.add("normalize_depth_nonzero_clips", "normalize_depth", {"d": d},
          {"result": mdp_normalize(d, cam2)}, attrs=dict(cam2))

    for source in ("height_scan", "depth"):
        clip = do.EXTERO_CLIP_NP[source]
        raw = np.concatenate([
            rng.standard_normal(n) * clip * 1.5,
            np.array([np.nan, np.inf, -np.inf, clip, -clip, 0.0], dtype=np.float64),
        ])[:, None]
        w.add(f"sanitize_extero_{source}", "sanitize_extero", {"values": raw},
              {"result": do.sanitize_extero_np(raw, source)},
              attrs={"source": source, "clip": clip})


def mdp_normalize(d: np.ndarray, cam: dict) -> np.ndarray:
    """``normalize_depth_np`` with a camera contract dict, for brevity above."""
    return do.normalize_depth_np(
        d, offset=cam["offset"], scale=cam["scale"], min_dist=cam["min_dist"],
        max_dist=cam["max_dist"], min_clip_val=cam["min_clip_val"],
        max_clip_val=cam["max_clip_val"])


def main() -> int:
    """Write the fixture set. Returns a process exit code."""
    default_out = os.path.join(_TRANSFER, "deploy", "tests", "fixtures", "parity")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=default_out, help="output directory")
    ap.add_argument("--count", type=int, default=512, help="random cases per function")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")
    args = ap.parse_args()

    if sys.byteorder != "little":
        print(f"ERROR: fixtures are little-endian raw blobs; this host is {sys.byteorder}.")
        return 1

    rng = np.random.default_rng(args.seed)
    w = FixtureWriter(args.out)
    dump_quat_math(w, rng, args.count)
    dump_window(w, rng, args.count)
    dump_depth(w, rng, args.count)
    path = w.write_manifest(args.seed)

    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"[fixtures] {len(w.cases)} cases -> {path} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
