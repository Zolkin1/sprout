# Copyright 2026 Zachary Olkin. All rights reserved.

"""onnxruntime tracker-policy session with metadata-driven parameters.

Loads ``tracker.onnx`` (raw ``obs`` -> ``actions``, normalizer folded in) and reads
the embedded ``policy_parameters`` JSON (the former ``policy_parameters.yaml``),
so deployment needs only the single ``.onnx``. No torch.
"""

from __future__ import annotations

import json
import re

import numpy as np
import onnxruntime as ort


#: Short provider aliases accepted by :func:`resolve_providers` and the benchmark CLI.
PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}

#: Provider request meaning "the GPU if this wheel has one, else the CPU".
#: Not an alias (it resolves against what is installed, not to a fixed name) and not what
#: ``None`` means -- see :func:`_expand_auto`.
AUTO_PROVIDER = "auto"


_GPU_DLLS_PRELOADED = False


def _preload_gpu_dlls() -> None:
    """Make pip-installed CUDA/cuDNN visible to onnxruntime, once per process.

    When the CUDA stack comes from the ``nvidia-*`` wheels rather than a system install,
    onnxruntime cannot find ``libcublasLt.so`` / ``libcudnn.so`` on its own: it logs a
    load failure and **silently falls back to CPU**, so a GPU run reports plausible
    numbers that are really CPU numbers. ``preload_dlls`` loads them from the installed
    wheels. Absent on older onnxruntime builds, where a system CUDA on the loader path is
    the only option -- hence the guard.
    """
    global _GPU_DLLS_PRELOADED                                        # noqa: PLW0603
    if _GPU_DLLS_PRELOADED:
        return
    _GPU_DLLS_PRELOADED = True
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception as exc:                                      # noqa: BLE001
            print(f"[WARN deploy] onnxruntime.preload_dlls() failed ({exc}); a CUDA "
                  "provider may fall back to CPU.")


def _expand_auto(providers: list) -> list:
    """Replace an ``"auto"`` element with the best provider this wheel actually offers.

    ``auto`` exists for the SIMULATION runners, where the GPU is simply the right default:
    the generator is ~99% of the step cost and CPU is not a degraded mode for it but an
    unusable one (~100 ms against a 20 ms budget), so a CPU-by-default sim2sim run is a
    5x-slower run nobody asked for.

    It is deliberately NOT what ``None`` means. On hardware the provider is part of what a
    run claims about itself, so it stays stated rather than discovered -- and a named
    provider that is missing still raises below. ``auto`` is the one request that cannot
    fail: "no CUDA on this box" is an answer to it, not an error.

    Args:
        providers: The requested provider list, possibly containing ``"auto"``.

    Returns:
        The list with every ``"auto"`` replaced by ``"cuda"`` or ``"cpu"``. Returned
        unchanged (and without querying onnxruntime) when there is no ``"auto"`` in it.
    """
    if not any(isinstance(p, str) and p.lower() == AUTO_PROVIDER for p in providers):
        return providers
    best = ("cuda" if PROVIDER_ALIASES["cuda"] in set(ort.get_available_providers())
            else "cpu")
    return [best if isinstance(p, str) and p.lower() == AUTO_PROVIDER else p
            for p in providers]


def resolve_providers(providers=None) -> list:
    """Normalise a provider request into an onnxruntime provider list.

    Args:
        providers: ``None`` for CPU only (the deployment default -- a GPU is opt-in, so
            nothing silently starts using one); a list of names or short aliases
            (``cpu`` / ``cuda`` / ``trt``); ``["auto"]`` to take the GPU when the wheel
            has one and the CPU otherwise (see :func:`_expand_auto`); or a list of
            ``(name, options)`` tuples, which pass through untouched.

    Returns:
        A provider list suitable for :class:`onnxruntime.InferenceSession`. Aliases are
        expanded and CPU is appended as the final fallback, which is what makes a GPU
        request degrade rather than fail on a box without one.

    Raises:
        ValueError: If a requested provider is not one onnxruntime was built with.
    """
    if providers is None:
        return ["CPUExecutionProvider"]
    providers = _expand_auto(providers)
    names = [p[0] if isinstance(p, tuple) else PROVIDER_ALIASES.get(str(p).lower(), str(p))
             for p in providers]
    if any(n != "CPUExecutionProvider" for n in names):
        _preload_gpu_dlls()
    available = set(ort.get_available_providers())
    out = []
    for p in providers:
        if isinstance(p, tuple):
            out.append(p)
            continue
        name = PROVIDER_ALIASES.get(str(p).lower(), str(p))
        if name not in available:
            raise ValueError(
                f"onnxruntime has no '{name}'. Built-in providers here: "
                f"{sorted(available)}. A CUDA/TensorRT provider needs the "
                "'onnxruntime-gpu' wheel (and on Jetson, NVIDIA's JetPack-matched build "
                "-- the PyPI aarch64 wheel targets SBSA, not L4T)."
            )
        out.append(name)
    if "CPUExecutionProvider" not in [p[0] if isinstance(p, tuple) else p for p in out]:
        out.append("CPUExecutionProvider")   # fallback for any op the accelerator lacks
    return out


def make_session(onnx_path: str, num_threads: int = 0, providers=None,
                 cuda_graph: bool = False) -> ort.InferenceSession:
    """Create an ORT session with full graph optimization.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        num_threads: Intra-op threads (0 lets ORT choose). CPU-side only.
        providers: Execution providers; see :func:`resolve_providers`. ``None`` = CPU.
        cuda_graph: Ask the CUDA provider to capture the launch sequence and replay it. The
            session is then usable ONLY through an IoBinding over fixed device buffers --
            ``sess.run()`` returns an empty output list from the second call on -- so the
            caller has to be built for it (``fm_session.enable_cuda_graph``). Session creation
            itself fails if any node landed on the CPU, which is what makes this safe to ask
            for: it cannot half-apply.

    Returns:
        A configured :class:`onnxruntime.InferenceSession`.

    Raises:
        ValueError: If ``cuda_graph`` is set without a CUDA provider in the request.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads > 0:
        so.intra_op_num_threads = num_threads
    resolved = resolve_providers(providers)
    if not cuda_graph:
        return ort.InferenceSession(onnx_path, so, providers=resolved)
    names = [p[0] if isinstance(p, tuple) else p for p in resolved]
    if PROVIDER_ALIASES["cuda"] not in names:
        raise ValueError(
            f"cuda_graph was requested for {onnx_path} but the resolved providers are "
            f"{names}. Capture is a CUDA feature; there is nothing to capture on the CPU."
        )
    # Rebuilt rather than mutated: the option rides on the CUDA entry, and the resolved list
    # may carry it as a bare name.
    with_capture = []
    for p in resolved:
        name = p[0] if isinstance(p, tuple) else p
        if name != PROVIDER_ALIASES["cuda"]:
            with_capture.append(p)
            continue
        opts = dict(p[1]) if isinstance(p, tuple) else {}
        opts["enable_cuda_graph"] = "1"
        with_capture.append((name, opts))
    return ort.InferenceSession(onnx_path, so, providers=with_capture)


class TrackerSession:
    """onnxruntime RL-tracker policy + its embedded deployment parameters."""

    def __init__(self, onnx_path: str, num_threads: int = 0, providers=None):
        """Load the tracker ONNX and parse its ``policy_parameters`` metadata."""
        self.sess = make_session(onnx_path, num_threads, providers)
        meta = self.sess.get_modelmeta().custom_metadata_map
        if "policy_parameters" not in meta:
            raise ValueError(f"{onnx_path} has no 'policy_parameters' metadata; re-export with metadata.")
        self.params = json.loads(meta["policy_parameters"])
        self._in = self.sess.get_inputs()[0].name
        self._out = self.sess.get_outputs()[0].name
        self.joint_names_isaac = list(self.params["joint_names_isaac"])
        self.default_joint_angles = np.asarray(self.params["default_joint_angles"], dtype=np.float64)
        self.action_scale = self._resolve_action_scale()
        self.action_isaac = np.zeros(self.get_num_actions())
        # The term table is immutable after load and read every control step, so resolve it
        # once here rather than re-materialising ~16 dicts per step in the hot path.
        self._obs_terms = self._resolve_obs_terms()

    def _resolve_action_scale(self) -> np.ndarray:
        """Expand the per-joint action-scale patterns into joint order."""
        patterns = self.params.get("action_scale", {})
        out = np.zeros(len(self.joint_names_isaac))
        for i, jn in enumerate(self.joint_names_isaac):
            for pat, val in patterns.items():
                if re.fullmatch(pat, jn):
                    out[i] = val
                    break
            else:
                raise ValueError(f"No action_scale pattern matches joint '{jn}'")
        return out

    # -- parameter accessors --
    def get_num_obs(self) -> int:
        return int(self.params["num_obs"])

    def get_num_actions(self) -> int:
        return int(self.params["num_actions"])

    def get_dt(self) -> float:
        return float(self.params["dt"])

    def get_kp(self) -> list[float]:
        return list(self.params["kp"])

    def get_kd(self) -> list[float]:
        return list(self.params["kd"])

    def get_joint_names(self) -> list[str]:
        return self.joint_names_isaac

    def _resolve_obs_terms(self) -> list[dict]:
        """Parse the embedded policy obs terms into ``{name, shape, scale, history_length, clip}``.

        ``shape`` is the term's FLAT width: multi-dimensional terms (the per-camera image
        groups export ``[1, H, W]``) are flattened, matching how the observation manager
        concatenates them into the policy vector. ``clip`` is the training clip magnitude
        (``None`` when the term is unclipped), so the deploy saturates exactly as training
        did without a hardcoded per-term table.
        """
        out = []
        for name, term in self.params["observation_terms"]["policy"].items():
            clip = term.get("clip")
            out.append({
                "name": name,
                "shape": _flat_dim(term["shape"]),
                "scale": term.get("scale", 1.0),
                "history_length": int(term.get("history_length", 0)),
                "clip": None if clip is None else max(abs(float(clip[0])), abs(float(clip[1]))),
            })
        return out

    def get_obs_terms(self) -> list[dict]:
        """The ordered policy obs terms (see :meth:`_resolve_obs_terms`)."""
        return self._obs_terms

    def get_height_scan_spec(self) -> dict | None:
        """Height-scanner geometry (resolution, size, offset_pos, ray_alignment,
        sensor_offset, clip) embedded at export time, or None if the exported policy
        group has no ``height_scan`` term.

        Lets the MuJoCo / hardware deploy replicate the exact grid the policy + FM were
        trained on. See ``scripts/rsl_rl/export_parameters.py``.
        """
        return self.params.get("height_scan_spec")

    def get_depth_camera_spec(self) -> list[dict] | None:
        """Depth-camera contract, one entry per camera in observation order, or None.

        Mount body + offset, pinhole intrinsics, range and the ``(d - offset) * scale``
        normalisation -- everything needed to reproduce the depth observation off-sim.
        Produced by ``mdp.depth_camera_specs`` at export time; see
        ``transfer/deploy/depth_obs.py`` for the consumer side.
        """
        spec = self.params.get("depth_camera_spec")
        return list(spec) if spec else None

    def get_robot_asset(self) -> dict | None:
        """WHICH ROBOT this policy was trained on, or None for a pre-fingerprint export.

        Content hashes of the URDF / MJCF / USD plus the joint names, body names and total
        mass. Nothing else in this metadata identifies the robot: ``joint_names_isaac`` is
        unchanged by a payload, a hand swap or a gear ratio, so before this field a
        checkpoint trained on a different robot loaded silently. Produced by
        ``motion_gen.robot_asset.fingerprint_robot_asset`` at export time.
        """
        return self.params.get("robot_asset")

    def get_terrain_fingerprint(self) -> dict | None:
        """WHICH TERRAIN this policy trained on, or None for a pre-fingerprint export.

        A content hash of the clip library's ``terrain_manifest``. The library's own
        fingerprint does not cover the terrain, so a library whose tile geometry was
        rewritten offline -- motions untouched -- is otherwise indistinguishable.

        NOTHING ON THIS PATH COMPARES IT, deliberately, and the two obvious comparisons are
        both wrong rather than merely unimplemented:

        * against the GENERATOR, the way ``robot_asset`` is compared. The generator has no
          terrain -- it conditions on depth, which is a measurement of whatever ground is
          actually there -- so there is no second value to disagree with.
        * against the deploy box. It has no terrain either; it has real ground, and a hash of
          a training manifest cannot say anything about it.

        Where it WOULD bite is the sim side, checking a tracker against the scene or clip
        library a run is replaying -- a tile whose geometry was rewritten offline plays a
        reference the policy was not trained for. That belongs in ``gen_tracker_runner`` /
        ``play.py``, which know what they loaded; this runtime does not. Kept here because it
        is cheap to carry and the alternative is re-deriving it from the artifact later.
        """
        return self.params.get("terrain_fingerprint")

    def get_valid_ic_pos(self) -> np.ndarray | None:
        """Valid initial pose [base_pos(3), base_quat(4), joint_pos(N)] or None."""
        ic = self.params.get("valid_ic_pos")
        return np.asarray(ic) if ic is not None else None

    def get_valid_ic_vel(self) -> np.ndarray | None:
        """Valid initial velocity [base_lin(3), base_ang(3), joint_vel(N)] or None."""
        ic = self.params.get("valid_ic_vel")
        return np.asarray(ic) if ic is not None else None

    def convert_joint_order(self, values, names_in, names_out):
        """Instance alias for :func:`convert_joint_order` (drop-in for Robot.set_initial_condition)."""
        return convert_joint_order(values, names_in, names_out)

    def get_ee_frame_names(self) -> list[str] | None:
        """End-effector reference frame names (the ``ee_ref`` output frames), if exported.

        Returns the resolved ee frame order (``traj_ref.resolve_ee_frames()`` at
        export time) used by ``ee_ref`` and, in tracker-only sim2sim, by the clip
        reference.  ``None`` when the export predates the metadata key.
        """
        names = self.params.get("ee_frame_names")
        return list(names) if names is not None else None

    def get_root_frame_bodies(self) -> list[str]:
        """Body links tracked in the ROOT frame, in ``body_*_ref`` stacking order.

        Empty for a tracker trained with ``tracking_mode="joint"`` (the default) and for any
        export predating the metadata key.  Non-empty means the reference source must supply
        root-frame body targets -- see ``transfer/deploy/clip_reference.py``.
        """
        return list(self.params.get("root_frame_bodies") or [])

    def get_reference_window_spec(self) -> dict | None:
        """Contract for the merged ``reference_window`` observation, or None.

        ``{num_steps, stride, offsets, blocks, joint_order, ee_frame_names}`` -- the
        sample offsets in control steps and the per-step block layout
        (``[[name, width], ...]``), which the flat term width alone cannot express.
        The joint blocks are in the ROBOT's joint order, not ``joint_out_names``.
        None when the tracker has no such term, or on an export predating the key.
        """
        return self.params.get("reference_window_spec")

    def get_command_window_spec(self) -> dict | None:
        """Contract for the ``command_window`` observation, or None.

        ``{num_steps, stride, offsets, cmd_dim}``; index 0 is the current command and
        the rest is the look-ahead, at the same offsets the reference window uses.
        """
        return self.params.get("command_window_spec")

    def get_joint_out_names(self) -> list[str] | None:
        """CLF output joint order, if exported (else None -> caller falls back)."""
        if "joint_out_names" in self.params:
            return list(self.params["joint_out_names"])
        names = self.params.get("ordered_pos_output_names")
        if names:
            if isinstance(names, str):
                names = json.loads(names)
            return [n.split(":", 1)[1] for n in names if n.startswith("joint:")]
        return None

    # -- inference --
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run the tracker on ``obs`` ``[N, num_obs]`` -> raw actions ``[N, num_actions]``."""
        return self.sess.run([self._out], {self._in: obs.astype(np.float32)})[0]

    def get_action(self, obs: np.ndarray, joint_names_out: list[str]) -> np.ndarray:
        """Run the tracker and return joint position targets in ``joint_names_out`` order."""
        self.action_isaac = self.infer(obs).squeeze()
        target = self.action_isaac * self.action_scale + self.default_joint_angles
        return convert_joint_order(target, self.joint_names_isaac, joint_names_out)


def _flat_dim(shape) -> int:
    """Flat width of an exported obs-term shape (a scalar, or a list to multiply out)."""
    if isinstance(shape, (list, tuple)):
        return int(np.prod([int(s) for s in shape]))
    return int(shape)


def convert_joint_order(values: np.ndarray, names_in: list[str], names_out: list[str]) -> np.ndarray:
    """Reorder per-joint ``values`` from ``names_in`` order to ``names_out`` order."""
    lookup = {n: v for n, v in zip(names_in, values)}
    out = np.zeros_like(values)
    for i, n in enumerate(names_out):
        if n not in lookup:
            raise ValueError(f"Joint '{n}' not found in input joint names")
        out[i] = lookup[n]
    return out
