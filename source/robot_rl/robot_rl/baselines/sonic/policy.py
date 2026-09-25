# Copyright 2026 Zachary Olkin. All rights reserved.

"""Run NVIDIA's released SONIC tracker from its two ONNX graphs, with onnxruntime only.

The upstream training stack (``gear_sonic``) is never imported: it targets IsaacLab 2.3 and pins
``numpy==1.26.4`` / ``hydra==1.3.2`` / ``trl==0.28.0``, all of which conflict with the IsaacLab
3.0-beta sibling checkout this repo shares. The released encoder/decoder pair is a stateless MLP
pair, so onnxruntime plus the observation port in :mod:`.obs` is the whole runtime.

Both graphs were exported with a fixed batch of 1 (``torch.onnx.export`` without dynamic axes in
``gear_sonic/utils/inference_helpers.py``), so a multi-env step loops rows. At ~24 envs that is
tens of milliseconds per control step -- irrelevant for evaluation, which is all this is for.

Torch-free and isaaclab-free; :mod:`.env_bridge` is the piece that talks to the simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .obs import (
    HistoryBuffer,
    MotionReference,
    RobotState,
    build_decoder_input,
    build_encoder_input,
    history_depth,
    history_dims,
)
from .spec import SonicSpec, load_spec

# Upstream clips the raw network output before it reaches the action term
# (``gear_sonic/envs/wrapper/manager_env_wrapper.py:837``), so the clipped value is what the
# policy was trained to see fed back as ``last_actions``. The C++ deploy path does not clip;
# with the released std clamp (<= 0.5) the bound is never reached in normal operation, so the
# two agree in practice and we follow the training-time behaviour.
ACTION_CLIP = 20.0

ENCODER_FILE = "model_encoder.onnx"
DECODER_FILE = "model_decoder.onnx"
OBS_CONFIG_FILE = "observation_config.yaml"

NUM_ACTIONS = 29


class SonicPolicy:
    """The released SONIC tracker as a batched, per-step callable.

    Args:
        model_dir: Directory holding ``model_encoder.onnx``, ``model_decoder.onnx`` and
            ``observation_config.yaml`` (i.e. ``gear_sonic_deploy/policy/release`` after
            ``download_from_hf.py``).
        num_envs: Batch size of the environment this will drive.
        mode: Encoder mode. Only ``"g1"`` -- the robot-motion-reference mode -- is supported;
            teleop and SMPL would need VR or SMPL inputs this repo has no source for.
        providers: onnxruntime execution providers. Defaults to CPU, which is both deterministic
            and fast enough for two small MLPs.
    """

    def __init__(self, model_dir: str | Path, num_envs: int, mode: str = "g1",
                 providers: list[str] | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.num_envs = int(num_envs)
        self.mode = mode
        self._providers = providers or ["CPUExecutionProvider"]
        self.spec: SonicSpec | None = None
        self._enc = None
        self._dec = None
        self._enc_in = ""
        self._dec_in = ""
        self._hist: HistoryBuffer | None = None
        self._last_action = np.zeros((self.num_envs, NUM_ACTIONS), np.float64)
        # Kept for offline validation against a trace from NVIDIA's own env.
        self.last_encoder_input: np.ndarray | None = None
        self.last_decoder_input: np.ndarray | None = None

    # ------------------------------------------------------------------ lifecycle

    def load(self) -> SonicPolicy:
        """Open both graphs, resolve the observation layout and validate it against them.

        Returns:
            ``self``, so this can be chained onto construction.

        Raises:
            FileNotFoundError: If a graph or the observation config is missing.
            ValueError: If the resolved layout does not match the graphs' declared input widths,
                or the decoder does not emit 29 actions.
        """
        import onnxruntime as ort                                          # deferred: heavy

        for f in (ENCODER_FILE, DECODER_FILE):
            if not (self.model_dir / f).is_file():
                raise FileNotFoundError(
                    f"{self.model_dir / f} not found. Fetch the release with "
                    "'python download_from_hf.py' in the GR00T-WholeBodyControl checkout."
                )
        self.spec = load_spec(self.model_dir / OBS_CONFIG_FILE)
        self._enc = ort.InferenceSession(str(self.model_dir / ENCODER_FILE),
                                         providers=self._providers)
        self._dec = ort.InferenceSession(str(self.model_dir / DECODER_FILE),
                                         providers=self._providers)
        self._enc_in = self._enc.get_inputs()[0].name
        self._dec_in = self._dec.get_inputs()[0].name
        self.spec.validate_dims(
            encoder_dim=int(self._enc.get_inputs()[0].shape[-1]),
            decoder_dim=int(self._dec.get_inputs()[0].shape[-1]),
        )
        n_act = int(self._dec.get_outputs()[0].shape[-1])
        if n_act != NUM_ACTIONS:
            raise ValueError(f"{DECODER_FILE} emits {n_act} actions, expected {NUM_ACTIONS}.")
        token_dim = int(self._enc.get_outputs()[0].shape[-1])
        if token_dim != self.spec.token_dim:
            raise ValueError(f"{ENCODER_FILE} emits a {token_dim}-d token but "
                             f"{OBS_CONFIG_FILE} declares encoder.dimension "
                             f"{self.spec.token_dim}.")
        self._hist = HistoryBuffer(self.num_envs, history_depth(self.spec), history_dims(self.spec))
        return self

    @property
    def reference_frames(self) -> tuple[int, int]:
        """``(frames, step)`` the reference window must be sampled at for this checkpoint."""
        return self._require_loaded().reference_frames(self.mode)

    @property
    def last_action(self) -> np.ndarray:
        """``[N, 29]`` the most recent raw (clipped, pre-scale) action."""
        return self._last_action.copy()

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset the action feedback and history warm-start for ``env_ids`` (all when None)."""
        self._require_loaded()
        if env_ids is None:
            self._last_action[:] = 0.0
        elif len(env_ids):
            self._last_action[np.asarray(env_ids, dtype=np.int64)] = 0.0
        assert self._hist is not None
        self._hist.reset(env_ids)

    # ------------------------------------------------------------------ stepping

    def act(self, state: RobotState, ref: MotionReference) -> np.ndarray:
        """One control step: observations in, raw actions out.

        The returned value is the network's raw output, clipped to +-:data:`ACTION_CLIP`. Turning
        it into joint targets (``default_q + scale * action``) is the environment's action term's
        job, which is where SONIC's own scale lives.

        Args:
            state: This step's proprioception. ``state.last_action`` is filled from the policy's
                own previous output when left None.
            ref: The reference window, sampled at :attr:`reference_frames`.

        Returns:
            ``[N, 29]`` float32 raw actions.

        Raises:
            RuntimeError: If :meth:`load` has not been called.
            ValueError: On a batch-size or reference-window mismatch.
        """
        spec = self._require_loaded()
        assert self._hist is not None
        frames, _ = self.reference_frames
        if ref.joint_pos.shape[:2] != (self.num_envs, frames):
            raise ValueError(
                f"reference window is {ref.joint_pos.shape[:2]}, expected "
                f"{(self.num_envs, frames)} -- sample it at the (frames, step) reported by "
                "SonicPolicy.reference_frames."
            )
        if state.last_action is None:
            state = state.with_last_action(self._last_action)
        self._hist.append(state)

        enc_obs = build_encoder_input(spec, self.mode, ref, state.root_quat)
        token = self._run(self._enc, self._enc_in, enc_obs, spec.token_dim)
        dec_obs = build_decoder_input(spec, token, self._hist)
        action = self._run(self._dec, self._dec_in, dec_obs, NUM_ACTIONS)
        self.last_encoder_input, self.last_decoder_input = enc_obs, dec_obs

        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        self._last_action = action.astype(np.float64)
        return action.astype(np.float32)

    def _run(self, sess, input_name: str, obs: np.ndarray, out_dim: int) -> np.ndarray:
        """Run a fixed-batch-1 graph over every row of ``obs``."""
        out = np.empty((obs.shape[0], out_dim), np.float32)
        for i in range(obs.shape[0]):
            out[i] = sess.run(None, {input_name: obs[i:i + 1]})[0][0]
        return out

    def _require_loaded(self) -> SonicSpec:
        """Return the spec, or explain that :meth:`load` has to run first."""
        if self.spec is None or self._enc is None or self._dec is None:
            raise RuntimeError("SonicPolicy.load() must be called before use.")
        return self.spec
