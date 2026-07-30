# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the FM ONNX export: the graph signature and the ``fm_meta`` contract.

``scripts/flow_matching/export_fm_onnx.py`` had no test at all -- ``test_no_syntax_errors``
exists partly because a merge once left it with duplicate keyword arguments. Everything
downstream (``transfer/deploy/fm_session.py``, the MuJoCo runner, the hardware
controller) is driven entirely by this graph's input names and this metadata block, so a
silent change to either is a silent change to deployment.

Runs on CPU with a tiny model; no checkpoint or recording on disk is needed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

from robot_rl.flow_matching import (
    EE_BLOCK_DIM,
    FlowMatchingConfig,
    FlowMatchingTransformer,
    ROOT_BLOCK_DIM,
    build_train_val,
    target_state_dim,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *[".."] * 5))
# The sibling test file is collected as a top-level module (these tests are run by path,
# not as a package), so reach it through the directory rather than a relative import.
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "flow_matching"))
from test_flow_matching import _synthetic_recording  # noqa: E402

onnx = pytest.importorskip("onnx", exc_type=ImportError)
from export_fm_onnx import export_fm_onnx  # noqa: E402


def _tiny_checkpoint(tmp_path) -> tuple[str, int, int, int, int]:
    """Train-free checkpoint in the exact shape ``load_fm_context`` expects.

    Returns:
        ``(checkpoint_path, num_history_states, future_len, num_joints, num_extra_frames)``.
    """
    torch.manual_seed(0)
    data, meta, nj, k, f, state_dim = _synthetic_recording()
    n = 2
    cfg = FlowMatchingConfig(
        num_history_states=n, future_len=f, cmd_array="target_vel", cmd_stride=4,
        target_source="ego_anchored_clip_future", val_fraction=0.2, seed=0,
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
    )
    train, _ = build_train_val(data, meta, cfg)
    model = FlowMatchingTransformer(
        state_dim=cfg.state_dim, future_len=f, num_history_states=n, history_dim=cfg.state_dim,
        height_dim=cfg.height_dim, cmd_dim=cfg.cmd_dim, cmd_stride=cfg.cmd_stride,
        d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
    ).eval()
    path = str(tmp_path / "ckpt.pt")
    torch.save({
        "config": cfg.to_dict(),
        "meta": {"state_joint_names": meta.state_joint_names,
                 "target_frame_names": meta.state_frame_names,
                 "fm_snap_every": 12},
        "num_joints": nj,
        "state_normalizer": train.state_normalizer.state_dict(),
        "cmd_normalizer": train.cmd_normalizer.state_dict(),
        "height_normalizer": train.height_normalizer.state_dict(),
        "model": model.state_dict(),
    }, path)
    assert state_dim == target_state_dim(nj, k)
    return path, n, f, nj, k


def test_fm_onnx_graph_signature(tmp_path):
    """The graph must expose exactly the five inputs the deploy runtime feeds by name."""
    ckpt, n, f, nj, k = _tiny_checkpoint(tmp_path)
    out = str(tmp_path / "fm.onnx")
    export_fm_onnx(ckpt, out, opset=18)
    model = onnx.load(out, load_external_data=False)

    d = target_state_dim(nj, k)
    got_in = {i.name: [dd.dim_value or dd.dim_param for dd in i.type.tensor_type.shape.dim]
              for i in model.graph.input}
    assert set(got_in) == {"history", "extero", "cmd_tokens", "noise", "history_null"}, got_in
    # Batch is dynamic; every other axis is fixed by the checkpoint.
    assert got_in["history"][1:] == [n, d]
    assert got_in["noise"][1:] == [f, d]
    assert got_in["history_null"][1:] == [1]
    assert [o.name for o in model.graph.output] == ["trajectory"]
    out_shape = [dd.dim_value or dd.dim_param
                 for dd in model.graph.output[0].type.tensor_type.shape.dim]
    assert out_shape[1:] == [f, d]
    # All five inputs must share ONE dynamic batch symbol, or a multi-env caller silently
    # binds mismatched batches.
    assert len({got_in[k_][0] for k_ in got_in} | {out_shape[0]}) == 1


def test_fm_meta_contract(tmp_path):
    """``fm_meta`` must carry every key the deploy runtime reads, with the right values."""
    ckpt, n, f, nj, k = _tiny_checkpoint(tmp_path)
    out = str(tmp_path / "fm.onnx")
    returned = export_fm_onnx(ckpt, out, opset=18)

    props = {kv.key: kv.value for kv in onnx.load(out, load_external_data=False).metadata_props}
    assert set(props) == {"fm_meta"}
    meta = json.loads(props["fm_meta"])
    assert meta == returned, "the embedded metadata must equal what the exporter returns"

    # Everything transfer/deploy/fm_session.load_fm_context reads.
    required = {
        "joint_names", "target_frame_names", "N", "F", "D", "nj", "num_steps", "cmd_dim",
        "cmd_steps", "num_cmd_tokens", "ground_z", "height_dim", "extero_source",
        "extero_tokens", "extero_spec", "history_source", "fm_snap_every",
        "root_block_dim", "ee_block_dim", "has_history_null",
    }
    assert required <= set(meta), f"fm_meta is missing {sorted(required - set(meta))}"
    assert (meta["N"], meta["F"], meta["nj"]) == (n, f, nj)
    assert len(meta["target_frame_names"]) == k
    assert len(meta["joint_names"]) == nj
    assert len(meta["cmd_steps"]) == meta["num_cmd_tokens"]
    assert meta["history_source"] == "measured"
    assert meta["fm_snap_every"] == 12
    # The state-layout guard: D must be reconstructible from the advertised block widths,
    # which is the whole point of exporting them.
    assert meta["root_block_dim"] == ROOT_BLOCK_DIM
    assert meta["ee_block_dim"] == EE_BLOCK_DIM
    assert meta["D"] == 2 * meta["nj"] + meta["root_block_dim"] + meta["ee_block_dim"] * k


def test_fm_onnx_matches_torch(tmp_path):
    """The exported graph must reproduce the torch sampler it was traced from."""
    ort = pytest.importorskip("onnxruntime", exc_type=ImportError)
    ckpt, n, f, nj, k = _tiny_checkpoint(tmp_path)
    out = str(tmp_path / "fm.onnx")
    meta = export_fm_onnx(ckpt, out, opset=18)
    d, v = meta["D"], 3

    rng = np.random.default_rng(0)
    feed = {
        "history": rng.standard_normal((v, n, d)).astype(np.float32),
        "extero": rng.standard_normal((v, meta["extero_tokens"] * meta["height_dim"])).astype(np.float32),
        "cmd_tokens": rng.standard_normal((v, meta["num_cmd_tokens"], meta["cmd_dim"])).astype(np.float32),
        "noise": rng.standard_normal((v, f, d)).astype(np.float32),
        "history_null": np.zeros((v, 1), dtype=np.float32),
    }
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(["trajectory"], feed)[0]
    assert got.shape == (v, f, d)
    assert np.isfinite(got).all()

    # Same conditioning through the torch sampler the exporter wrapped.
    from robot_rl.flow_matching.online import load_fm_context  # noqa: PLC0415
    from export_fm_onnx import _FmSamplerOnnx  # noqa: PLC0415
    ctx = load_fm_context(ckpt, device="cpu")
    want = _FmSamplerOnnx(ctx)(*[torch.from_numpy(feed[key]) for key in
                                 ("history", "extero", "cmd_tokens", "noise", "history_null")])
    assert np.abs(got - want.detach().numpy()).max() < 1e-4

    # history_null=1 must select the learned null embedding, i.e. change the output.
    null_feed = dict(feed, history_null=np.ones((v, 1), dtype=np.float32))
    assert not np.allclose(got, sess.run(["trajectory"], null_feed)[0], atol=1e-6)
