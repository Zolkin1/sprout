# Copyright 2026 Zachary Olkin. All rights reserved.

"""Export a trained token planner to ONNX -- one graph, same signature as the generator's.

The artifact this writes is a drop-in for ``<run>_fm.onnx``: same five inputs, same single
output, same metadata key, same naming rule. That is deliberate, and it is what keeps the
torch-free runtime (``transfer/deploy/fm_session.py``), the C++ runtime
(``transfer/deploy_cpp/src/fm_session.cc``, whose bound input names are hard-coded) and the
MuJoCo runner from each needing their own port.

.. admonition:: The sampling loop, and why ``noise`` survives

    A flow model's loop is data-INDEPENDENT -- ``num_steps`` identical passes of
    ``x <- x + dt*v`` -- so it unrolls into a static graph. An autoregressive planner's is
    data-DEPENDENT: sample token ``i``, embed it, feed it back. ``torch.multinomial`` in the
    middle of that does not trace at all.

    **The Gumbel-max trick removes the obstruction exactly.** Drawing from
    ``softmax(logits / T)`` is identical in distribution to ``argmax(logits / T + g)`` with
    ``g ~ Gumbel(0, 1)`` i.i.d. per class. ``argmax`` traces; ``g`` becomes a graph INPUT. So:

    * the ``noise`` input keeps its slot and its meaning -- "the randomness for this plan,
      supplied by the caller so a probe can hold it fixed across arms" -- and every runtime
      that already binds a tensor called ``noise`` needs no new input name;
    * **zeros give greedy decoding** exactly, since ``argmax(logits/T + 0) = argmax(logits)``.
      That is what a deployment runs, and it needs no separate graph;
    * a Gumbel draw gives an EXACT sample at the baked-in temperature -- not an approximation.

    Its shape differs from the flow model's: ``(V, N_tokens, vocab)`` rather than
    ``(V, F, D)``. That is the one thing a runtime must not assume, and
    ``gen_tracker._sample_noise`` now reads the shape off the graph rather than computing it.

.. admonition:: The token -> trajectory half is a table lookup, on purpose

    Decoding an id normally goes id -> mixed-radix digits -> lattice value, which is integer
    division and modulo. onnxruntime keeps int64 ARITHMETIC on the host, and a single host
    node makes the whole graph ineligible for CUDA Graph capture -- which is where 8.3 of the
    flow model's 14-15 ms Thor budget comes from. So the whole ``id -> latent`` map is
    precomputed here into a ``(vocab, L_DIM)`` table and the graph does one ``Gather``. A
    gather with an int64 INDEX is an ordinary device op; it is the arithmetic that is not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import onnx
import torch
from torch import nn

from motion_gen.features import _REPO_ROOT

sys.path.insert(0, str(_REPO_ROOT / "transfer"))

from deploy.artifacts import fm_artifact_name  # noqa: E402

from robot_rl.flow_matching.window import (  # noqa: E402
    project_window_rotations,
    window_blocks,
)
from robot_rl.token_planner.online import TokenPlannerContext, load_planner_context  # noqa: E402
from robot_rl.token_planner.planner import _run_layer  # noqa: E402

#: Graph input/output names, in ``_PlannerSamplerOnnx.forward`` argument order. IDENTICAL to
#: ``export_fm_onnx._IO_NAMES`` so no runtime has to learn a second set.
_IO_NAMES = ("history", "extero", "cmd_tokens", "noise", "history_null", "trajectory")


class _PlannerSamplerOnnx(nn.Module):
    """ONNX-exportable wrapper: raw conditioning -> denormalized reference window.

    Folds the three input normalizations and the output denormalization into the graph,
    unrolls the ``N`` autoregressive decode steps, gathers each token's lattice value from a
    precomputed table, runs the frozen tokenizer decoder, and projects the rotation block
    back onto SO(3) -- so the torch-free runtime does no maths at all.

    Args:
        ctx: A loaded :class:`~robot_rl.token_planner.online.TokenPlannerContext`.
        temperature: Softmax temperature BAKED IN as a constant. It is a constant at
            inference (a deployment picks one), so it never becomes a graph input; a second
            temperature is a second graph, exactly as the flow exporter treats its advantage
            label.
    """

    def __init__(self, ctx: TokenPlannerContext, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.model = ctx.model
        self.num_tokens = int(ctx.num_steps)
        self.vocab_size = int(ctx.model.vocab_size)
        self.temperature = float(temperature)
        # Plain ints/tuples, so the SO(3) projection traces as fixed slicing.
        self.nj = int(ctx.nj)
        self.num_extra_frames = int(ctx.target_num_extra_frames)
        self.reference_blocks = tuple(ctx.reference_blocks)
        self.register_buffer("s_mean", ctx.state_norm.mean.detach().clone())
        self.register_buffer("s_std", ctx.state_norm.std.detach().clone())
        self.register_buffer("c_mean", ctx.cmd_norm.mean.detach().clone())
        self.register_buffer("c_std", ctx.cmd_norm.std.detach().clone())
        self.register_buffer("h_mean", ctx.extero_norm.mean.detach().clone())
        self.register_buffer("h_std", ctx.extero_norm.std.detach().clone())
        # id -> lattice value, precomputed. See the module docstring for why this is a table
        # and not the mixed-radix arithmetic it replaces.
        table = ctx.model.tokenizer.quantizer.id_to_latent(
            torch.arange(self.vocab_size, dtype=torch.long))
        self.register_buffer("id_to_latent", table.detach().clone())

    def forward(
        self,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor,
        noise: torch.Tensor,
        history_null: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize, decode ``N`` tokens autoregressively, detokenize, denormalize.

        Args:
            history: Raw history nodes ``(V, N_hist, D)``.
            extero: Raw flat exteroception ``(V, T * extero_sensor_dim)``.
            cmd_tokens: Raw strided command tokens ``(V, num_cmd_tokens, cmd_dim)``.
            noise: Gumbel noise ``(V, N_tokens, vocab)``. **Zeros give greedy decoding.**
            history_null: ``(V, 1)`` float flag; ``>0.5`` selects the learned null history.
                Float rather than bool because a bool graph input is awkward across runtimes.

        Returns:
            Denormalized, SO(3)-projected window nodes ``(V, F, D)``.
        """
        hist_n = (history - self.s_mean) / self.s_std
        extero_n = (extero - self.h_mean) / self.h_std
        cmd_n = (cmd_tokens - self.c_mean) / self.c_std
        null_m = history_null.reshape(-1) > 0.5

        m = self.model
        layers = list(m.encoder.layers)
        # The conditioning attends only to itself, so it is encoded ONCE and every decode step
        # reuses it -- the same decomposition the torch path uses, and the reason the prefix
        # does not appear N times in this graph.
        cache = m.encode_prefix(hist_n, extero_n, cmd_n, null_m)
        kv = [m.encoder.layers[i].norm1(cache[i]) for i in range(len(layers))]

        x = m.bos.expand(history.shape[0], 1, m.cfg.d_model) + m.out_pos[:, :1, :] + m.out_type
        latents = []
        for step in range(self.num_tokens):
            h = x
            for i, layer in enumerate(layers):
                # THE layer function, not a second copy of the pre-norm body. Its own
                # docstring flags that there are three dropout modules on the way through and
                # a wrong one is invisible at eval-time dropout -- and a copy here would sit
                # outside `test_run_layer_matches_a_masked_full_pass`, which is the only
                # thing that pins the op order. Measured drop-in: bit-identical output and
                # the same node count.
                kv[i] = torch.cat([kv[i], layer.norm1(h)], dim=1)
                h = _run_layer(layer, h, kv[i], kv_normed=True)
            logits = m.head(m.head_norm(h))[:, -1, :] / self.temperature
            # Gumbel-max: exact sampling at `temperature`, and exactly greedy at noise=0.
            nxt = torch.argmax(logits + noise[:, step, :], dim=-1)
            latents.append(self.id_to_latent[nxt])
            if step + 1 < self.num_tokens:
                x = (m.token_emb(nxt).unsqueeze(1)
                     + m.out_pos[:, step + 1:step + 2, :] + m.out_type)
        latent = torch.stack(latents, dim=1)                       # (V, N_tokens, L_DIM)
        nodes = m.tokenizer.decoder(latent)                        # (V, F, D) normalized
        return project_window_rotations(
            nodes * self.s_std + self.s_mean,
            self.nj, self.num_extra_frames, self.reference_blocks,
        )


def _refuse_incomplete(ctx: TokenPlannerContext) -> None:
    """Refuse to export a graph whose contract is empty, at EXPORT rather than at deploy.

    The flow exporter refuses the same things (a depth graph with no stated band, a
    privileged exteroception), and for the same reason: a deployment can only discover them
    later, and the message it gives then names the wrong tool.

    An EMPTY list is refused, not tolerated. "Absent" is a migration allowance; an empty
    ``joint_names`` is a positive claim about a zero-joint robot, and it silently switches
    off both the robot-fingerprint guard and the tracker/generator camera agreement (which
    then compares the tracker with itself).

    Args:
        ctx: The loaded context.

    Raises:
        ValueError: On any of them, naming the fix.
    """
    if not ctx.joint_names:
        raise ValueError(
            "this checkpoint states no joint names, so the exported graph could not map the "
            "generator's joint order onto a robot's. Re-train with a current "
            "`train_planner.py`, or run `scripts/token_planner/stamp_planner_meta.py` "
            "against the recording it trained on.")
    if len(ctx.target_frame_names) != ctx.target_num_extra_frames:
        raise ValueError(
            f"{len(ctx.target_frame_names)} frame name(s) against K="
            f"{ctx.target_num_extra_frames}; the two are one fact.")
    if ctx.extero_source == "depth" and ctx.extero_clip is None:
        raise ValueError(
            "a depth generator must STATE its extero_clip band; this one says None, which "
            "resolves to 5.575 in training and 2.0 on the deploy side. Train with "
            "--fm_config so the band is inherited.")
    if ctx.extero_source == "height_scan":
        raise ValueError(
            "this generator conditions on a height scan, which no deployment can produce. "
            "The flow exporter refuses the same by default; there is no --allow_privileged "
            "path here because nothing has needed one yet.")


def _tp_meta(ctx: TokenPlannerContext, temperature: float) -> dict:
    """Deployment metadata, stamped under the SAME ``fm_meta`` key the generator uses.

    Same key because every load-time check in ``transfer/deploy/fm_session.py`` is about the
    artifact contract -- node layout, sensor geometry, robot identity, re-plan cadence -- and
    all of it applies here unchanged. A second key would mean a second copy of those checks.

    ``generator_kind`` is what distinguishes the two, and ``noise_shape`` is what a runtime
    must read rather than assume: the flow model's noise is ``(F, D)`` and this one's is
    ``(N_tokens, vocab)``.

    Args:
        ctx: The loaded context.
        temperature: The temperature baked into the graph.

    Returns:
        The metadata dict.
    """
    planner_cfg = dict(ctx.planner_config)
    return {
        "generator_kind": "token_planner",
        "joint_names": list(ctx.joint_names),
        "target_frame_names": list(ctx.target_frame_names),
        "N": int(ctx.N), "F": int(ctx.F), "D": int(ctx.D), "nj": int(ctx.nj),
        "num_steps": int(ctx.num_steps),
        "cmd_dim": int(ctx.cmd_dim),
        "cmd_steps": list(ctx.cmd_steps),
        "num_cmd_tokens": int(ctx.num_cmd_tokens),
        "extero_sensor_dim": int(ctx.extero_sensor_dim),
        "extero_source": str(ctx.extero_source),
        "extero_sensor_count": int(ctx.extero_sensor_count),
        "extero_spec": ctx.extero_spec,
        "robot_asset": ctx.robot_asset,
        "extero_channels": int(ctx.extero_channels),
        # The ENCODER, or the deploy banner reports a convolutional model as folding each
        # camera to one token through an MLP -- and `extero_image_hw` is meaningless for the
        # permutation-invariant flat path, so it is stamped only where it constrains
        # anything, exactly as `export_fm_onnx` stamps it.
        "extero_encoder": str(planner_cfg.get("extero_encoder", "cnn")),
        "extero_image_hw": (ctx.extero_image_hw
                            if planner_cfg.get("extero_encoder") == "cnn" else None),
        "extero_tokens_per_sensor": int(ctx.model.prefix.tokens_per_sensor),
        "history_source": str(ctx.history_source),
        "fm_snap_every": ctx.fm_snap_every,
        # ``[[name, width], ...]``, NOT bare names: `fm_session._check_state_layout` does
        # `for n, _ in declared`, so a list of strings unpacks each STRING and every planner
        # artifact failed to load with "too many values to unpack". The widths are what make
        # it a layout rather than a list of labels.
        "window_blocks": [[n, int(w)] for n, w in
                          window_blocks(ctx.nj, len(ctx.target_frame_names),
                                        ctx.reference_blocks)],
        "has_history_null": True,
        "extero_clip": ctx.extero_clip,
        "depth_latency_probs": (list(ctx.depth_latency_probs)
                                if ctx.depth_latency_probs else None),
        # -- planner-specific --
        "vocab_size": int(ctx.model.vocab_size),
        "num_plan_tokens": int(ctx.num_steps),
        "temperature": float(temperature),
        "noise_shape": [int(ctx.num_steps), int(ctx.model.vocab_size)],
        "noise_distribution": "gumbel",
        "tokenizer": dict(ctx.planner_config.get("tokenizer") or {}),
    }


def export(checkpoint: str, out_path: str | None = None, opset: int = 18,
           temperature: float = 1.0, dynamic_batch: bool = False,
           use_ema: bool = True) -> dict:
    """Export a planner checkpoint to ONNX and stamp its metadata.

    Args:
        checkpoint: A ``planner.pt``.
        out_path: Destination; defaults to the naming rule beside the checkpoint.
        opset: ONNX opset.
        temperature: Baked-in softmax temperature.
        dynamic_batch: Declare the batch axis symbolic. Off by default for the reason
            ``export_fm_onnx`` records: a symbolic batch leaves thousands of host-side shape
            ops that rule out CUDA Graph capture.
        use_ema: Prefer the EMA weights.

    Returns:
        The stamped metadata.
    """
    ctx = load_planner_context(checkpoint, device="cpu", greedy=True, use_ema=use_ema)
    if ctx.cmd_dim <= 0:
        raise ValueError("export expects a command-conditioned checkpoint (cmd_dim > 0).")
    _refuse_incomplete(ctx)
    # Match the flow exporter: tracing a frozen model hits UnsupportedOperatorError on the
    # fused attention path, so the parameters are un-frozen for the trace only.
    for p in ctx.model.parameters():
        p.requires_grad_(True)
    wrapper = _PlannerSamplerOnnx(ctx, temperature).eval()

    v, n_tok = 1, int(ctx.num_steps)
    edim = int(ctx.extero_sensor_count) * int(ctx.extero_sensor_dim)
    dummy = (
        torch.randn(v, ctx.N, ctx.D), torch.randn(v, edim),
        torch.randn(v, ctx.num_cmd_tokens, ctx.cmd_dim),
        torch.zeros(v, n_tok, ctx.model.vocab_size),      # greedy dummy
        torch.zeros(v, 1),
    )
    out_path = out_path or os.path.join(
        os.path.dirname(os.path.abspath(checkpoint)), fm_artifact_name(checkpoint))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    dyn = ({name: {0: "V"} for name in _IO_NAMES} if dynamic_batch else None)
    torch.onnx.export(
        wrapper, dummy, out_path, opset_version=opset, dynamo=False,
        input_names=list(_IO_NAMES[:-1]), output_names=[_IO_NAMES[-1]], dynamic_axes=dyn,
    )
    meta = _tp_meta(ctx, temperature)
    model = onnx.load(out_path)
    model.metadata_props.append(
        onnx.StringStringEntryProto(key="fm_meta", value=json.dumps(meta)))
    onnx.save(model, out_path)
    print(f"[tp-export] wrote {out_path}")
    print(f"[tp-export] noise is {meta['noise_shape']} gumbel "
          f"(zeros = greedy); temperature {temperature} is baked in")
    return meta


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Export a token planner checkpoint to ONNX.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature BAKED IN. Only matters when the caller supplies "
                        "non-zero Gumbel noise; a greedy deployment is unaffected by it.")
    p.add_argument("--dynamic_batch", action="store_true", default=False)
    p.add_argument("--no_ema", dest="use_ema", action="store_false", default=True)
    a = p.parse_args()
    export(a.checkpoint, a.out, a.opset, a.temperature, a.dynamic_batch, a.use_ema)


if __name__ == "__main__":
    main()
