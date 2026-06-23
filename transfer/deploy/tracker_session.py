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


def make_session(onnx_path: str, num_threads: int = 0) -> ort.InferenceSession:
    """Create an ORT CPU session with full graph optimization.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        num_threads: Intra-op threads (0 lets ORT choose).

    Returns:
        A configured :class:`onnxruntime.InferenceSession`.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads > 0:
        so.intra_op_num_threads = num_threads
    return ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])


class TrackerSession:
    """onnxruntime RL-tracker policy + its embedded deployment parameters."""

    def __init__(self, onnx_path: str, num_threads: int = 0):
        """Load the tracker ONNX and parse its ``policy_parameters`` metadata."""
        self.sess = make_session(onnx_path, num_threads)
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

    def get_obs_terms(self) -> list[dict]:
        """Ordered policy obs terms: list of ``{name, shape, scale, history_length}``."""
        terms = self.params["observation_terms"]["policy"]
        return [
            {"name": n, "shape": int(t["shape"]), "scale": t.get("scale", 1.0),
             "history_length": int(t.get("history_length", 0))}
            for n, t in terms.items()
        ]

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


def convert_joint_order(values: np.ndarray, names_in: list[str], names_out: list[str]) -> np.ndarray:
    """Reorder per-joint ``values`` from ``names_in`` order to ``names_out`` order."""
    lookup = {n: v for n, v in zip(names_in, values)}
    out = np.zeros_like(values)
    for i, n in enumerate(names_out):
        if n not in lookup:
            raise ValueError(f"Joint '{n}' not found in input joint names")
        out[i] = lookup[n]
    return out
