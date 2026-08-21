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

import collections
import json
import os
import sys

import numpy as np
import pytest
import torch

from robot_rl.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingTransformer,
    build_train_val,
    window_blocks,
    window_dim,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *[".."] * 5))
# The sibling test file is collected as a top-level module (these tests are run by path,
# not as a package), so reach it through the directory rather than a relative import.
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "flow_matching"))
# The DEPTH fixture, deliberately: depth is the deployable exteroception (and the config
# default), so the export path that ships is the one under test here. It also gives the
# multi-token case -- T=2 cameras rather than the scan's single token -- and a
# depth_camera_spec, so `extero_spec` in fm_meta is exercised rather than left None.
from test_extero_tokens import _synthetic_recording  # noqa: E402

onnx = pytest.importorskip("onnx", exc_type=ImportError)
from export_fm_onnx import export_fm_onnx  # noqa: E402
sys.path.insert(0, os.path.join(_REPO_ROOT, "transfer"))
from deploy.fm_session import _check_state_layout  # noqa: E402


def _tiny_checkpoint(tmp_path, extero_encoder: str = "auto",
                    attention_mode: str | None = None) -> tuple[str, int, int, int, int]:
    """Train-free checkpoint in the exact shape ``load_fm_context`` expects.

    Args:
        tmp_path: pytest temp dir.
        extero_encoder: Which exteroception embedding to build. ``"auto"`` is what a real
            depth run resolves to (the convolutional grid); pass ``"mlp"`` for the flat
            baseline. The model is built from the RESOLVED config rather than the defaults,
            because a config saying one encoder over the other's weights is exactly what
            ``load_fm_context`` refuses.
        attention_mode: ``None`` takes the config default (``"prefix"``); pass ``"full"`` for
            the legacy architecture. Unlike the encoder, a mismatch here is NOT refusable --
            the modes are weight-identical -- so the fixture must build what it saves.

    Returns:
        ``(checkpoint_path, num_history_states, future_len, num_joints, num_extra_frames)``.
    """
    torch.manual_seed(0)
    # The deployable pairing at its real per-camera geometry, so the traced graph is the
    # shipped one -- a placeholder Nx1 "image" would trace a degenerate 1-D encoder.
    data, meta, nj, k, f, _ = _synthetic_recording(depth=(2, 26, 30))
    state_dim = window_dim(nj, k)
    n = 2
    cfg = FlowMatchingConfig(
        num_history_states=n, future_len=f, cmd_array="target_vel", cmd_stride=4,
        target_source="ego_anchored_clip_future", val_fraction=0.34, seed=0,
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        # 1 channel, not the shipped 5: the fixture's robot is four fake joints, and the xyz
        # channels rebuild the camera pose through real FK on the G1's MJCF. What the graph
        # signature and the metadata contract pin here does not depend on the channel count,
        # and ``test_depth_xyz_offline`` covers 5 channels on a real 29-joint recording.
        extero_channels=1, pinned_axis_loss_weight=1.0,
        extero_encoder=extero_encoder,
        **({} if attention_mode is None else {"attention_mode": attention_mode}),
    )
    train, _ = build_train_val(data, meta, cfg)
    # Through from_config, never a hand-written kwarg list. This fixture used to spell the
    # arguments out and drifted exactly as that method's docstring warns: the dataset writes
    # derived fields onto cfg (the image fold, the metric extent, the camera intrinsics, the
    # channel count) and every one omitted here builds a DIFFERENT model than the reload does,
    # which surfaces as missing state-dict keys rather than as anything about the omission.
    # It also settles attention_mode, whose two modes share one parameter set and so load into
    # each other without complaint.
    model = FlowMatchingTransformer.from_config(cfg).eval()
    path = str(tmp_path / "ckpt.pt")
    torch.save({
        "config": cfg.to_dict(),
        # Mirrors train.py::_save_ckpt, including the camera contract it injects -- that is
        # what fm_meta's extero_spec carries to the deploy for the agreement check.
        "meta": {"state_joint_names": meta.state_joint_names,
                 "target_frame_names": meta.state_frame_names,
                 "extero_spec": train.extero_spec,
                 "fm_snap_every": 12},
        "num_joints": nj,
        "state_normalizer": train.state_normalizer.state_dict(),
        "cmd_normalizer": train.cmd_normalizer.state_dict(),
        "extero_normalizer": train.extero_normalizer.state_dict(),
        "model": model.state_dict(),
    }, path)
    assert state_dim == window_dim(nj, k)
    return path, n, f, nj, k


@pytest.mark.parametrize("encoder", ["mlp", "cnn"])
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_fm_onnx_graph_signature(tmp_path, encoder, dynamic_batch):
    """The graph must expose exactly the five inputs the deploy runtime feeds by name.

    Both encoders, because the input SIGNATURE must not depend on which one is inside:
    ``extero`` stays the flat ``T * sensor_dim`` the deploy runtime already feeds, and the
    convolution folds it internally. If that ever stops holding, every off-sim consumer
    (MuJoCo, hardware) needs changing with it.

    Both batch declarations, because the DEFAULT changed: the batch axis is now fixed at 1
    (see ``export_fm_onnx._export_graph`` -- a symbolic batch put 2687 of 6606 kernel nodes
    on the host and ruled out CUDA Graph capture). A fixed axis has to be a concrete 1 in
    the file, not a symbol that happens to be bound to 1 at run time, because it is the
    declaration that constant folding acts on.
    """
    ckpt, n, f, nj, k = _tiny_checkpoint(tmp_path, encoder)
    out = str(tmp_path / "fm.onnx")
    export_fm_onnx(ckpt, out, opset=18, dynamic_batch=dynamic_batch)
    model = onnx.load(out, load_external_data=False)

    d = window_dim(nj, k)
    got_in = {i.name: [dd.dim_value or dd.dim_param for dd in i.type.tensor_type.shape.dim]
              for i in model.graph.input}
    assert set(got_in) == {"history", "extero", "cmd_tokens", "noise", "history_null"}, got_in
    # Every non-batch axis is fixed by the checkpoint, under either declaration.
    assert got_in["history"][1:] == [n, d]
    assert got_in["noise"][1:] == [f, d]
    assert got_in["history_null"][1:] == [1]
    assert [o.name for o in model.graph.output] == ["trajectory"]
    out_shape = [dd.dim_value or dd.dim_param
                 for dd in model.graph.output[0].type.tensor_type.shape.dim]
    assert out_shape[1:] == [f, d]

    batch = {got_in[k_][0] for k_ in got_in} | {out_shape[0]}
    if dynamic_batch:
        # All five inputs must share ONE dynamic batch symbol, or a multi-env caller
        # silently binds mismatched batches.
        assert batch == {"V"}, batch
    else:
        assert batch == {1}, batch


@pytest.mark.parametrize("encoder", ["mlp", "cnn"])
def test_fm_meta_contract(tmp_path, encoder):
    """``fm_meta`` must carry every key the deploy runtime reads, with the right values."""
    ckpt, n, f, nj, k = _tiny_checkpoint(tmp_path, encoder)
    out = str(tmp_path / "fm.onnx")
    returned = export_fm_onnx(ckpt, out, opset=18)

    props = {kv.key: kv.value for kv in onnx.load(out, load_external_data=False).metadata_props}
    assert set(props) == {"fm_meta"}
    meta = json.loads(props["fm_meta"])
    assert meta == returned, "the embedded metadata must equal what the exporter returns"

    # Everything transfer/deploy/fm_session.load_fm_context reads.
    required = {
        "joint_names", "target_frame_names", "N", "F", "D", "nj", "num_steps", "cmd_dim",
        "cmd_steps", "num_cmd_tokens", "extero_sensor_dim", "extero_source",
        "extero_sensor_count", "extero_spec", "fm_snap_every",
        "window_blocks", "has_history_null",
    }
    assert required <= set(meta), f"fm_meta is missing {sorted(required - set(meta))}"
    assert (meta["N"], meta["F"], meta["nj"]) == (n, f, nj)
    assert len(meta["target_frame_names"]) == k
    assert len(meta["joint_names"]) == nj
    assert len(meta["cmd_steps"]) == meta["num_cmd_tokens"]
    assert meta["fm_snap_every"] == 12
    # The state-layout guard. The generator's output goes straight into the tracker's
    # observation, so the block table has to travel: D alone does not identify the layout.
    assert [tuple(b) for b in meta["window_blocks"]] == window_blocks(nj, k)
    assert meta["D"] == window_dim(nj, k) == sum(w for _, w in meta["window_blocks"])
    # And a graph from before the representation change must be refused, not reinterpreted.
    stale = {**meta}
    stale.pop("window_blocks")
    with pytest.raises(ValueError, match="window_blocks"):
        _check_state_layout("stale.onnx", stale)


@pytest.mark.parametrize("encoder,expect_hw", [("cnn", [[26, 30], [26, 30]]), ("mlp", None)])
def test_only_a_convolutional_graph_states_its_image_fold(tmp_path, encoder, expect_hw):
    """``extero_image_hw`` is exported for the CNN and withheld for the MLP.

    The dataset derives the geometry for ANY depth recording, so exporting it unconditionally
    is the easy mistake -- and it makes the deploy enforce a constraint that does not exist:
    a flat embedding is permutation-invariant over pixels, so it would consume a transposed
    camera perfectly well and the fold check would refuse it anyway.

    Only a real export catches this (a hand-built context can carry whatever the test
    wants), which is why it is asserted end to end here rather than in the encoder's units.
    """
    ckpt, *_ = _tiny_checkpoint(tmp_path, encoder)
    meta = export_fm_onnx(ckpt, str(tmp_path / "fm.onnx"), opset=18)
    assert meta["extero_encoder"] == encoder
    assert meta["extero_image_hw"] == expect_hw
    # 56 = 7x8, the grid the DEFAULT strides (1, 2, 2) leave on a 26x30 image: the first stage
    # convolves at native resolution, so only two halvings follow. Stride 2 everywhere gives
    # 4x4 = 16, which is what this asserted before the default changed.
    assert meta["extero_tokens_per_sensor"] == (56 if encoder == "cnn" else 1)


def test_state_layout_guard_accepts_a_subset_but_not_an_incoherent_one():
    """A window may drop blocks; what it may not do is misdescribe itself.

    Which subset is the RIGHT one cannot be known from the generator alone -- that is
    ``GenTrackerPolicy.load``'s job, where the tracker is also in hand. What IS checkable
    here is internal coherence, and each failure below is one a shape check would miss:
    an unknown name, the canonical order broken (the same names, a different vector), a
    width that does not follow from ``(nj, K)``, and a ``D`` that contradicts the table.
    """
    nj, k = 5, 2
    slim = ("base_pose", "base_vel", "joint_pos")
    ok = {"nj": nj, "target_frame_names": ["a", "b"],
          "window_blocks": [[n_, w] for n_, w in window_blocks(nj, k, slim)],
          "D": window_dim(nj, k, slim)}
    assert _check_state_layout("ok.onnx", ok) == slim

    with pytest.raises(ValueError, match="unknown reference-window block"):
        _check_state_layout("x.onnx", {**ok, "window_blocks": [["base_pose", 6], ["elbow", 3]]})
    swapped = [[n_, w] for n_, w in window_blocks(nj, k, slim)][::-1]
    with pytest.raises(ValueError, match="out of packing order"):
        _check_state_layout("x.onnx", {**ok, "window_blocks": swapped})
    bad_width = [[n_, w] for n_, w in window_blocks(nj, k, slim)]
    bad_width[2] = [bad_width[2][0], nj + 1]
    with pytest.raises(ValueError, match="widths"):
        _check_state_layout("x.onnx", {**ok, "window_blocks": bad_width,
                                       "D": sum(w for _, w in bad_width)})
    with pytest.raises(ValueError, match="window_blocks sum"):
        _check_state_layout("x.onnx", {**ok, "D": ok["D"] + 1})


@pytest.mark.parametrize("encoder", ["mlp", "cnn"])
def test_fm_onnx_matches_torch(tmp_path, encoder):
    """The exported graph must reproduce the torch sampler it was traced from.

    Run at ``v=3`` on purpose: it is the check that the convolutional reshapes kept the
    graph's batch axis DYNAMIC. A reshape traced against a concrete batch would pass at
    v=1 and fail here.

    That is why this one exports with ``dynamic_batch=True`` rather than taking the default.
    Deployment fixes the axis at 1 now, but the tracer property this test guards -- that the
    convolutional path does not bake the batch into a reshape -- is only observable in the
    symbolic export, and it is what makes ``dynamic_batch=True`` still work when someone
    reaches for it. The fixed-axis graph gets its own parity check in
    ``test_fm_onnx_static_batch``.
    """
    ort = pytest.importorskip("onnxruntime", exc_type=ImportError)
    ckpt, n, f, nj, k = _tiny_checkpoint(tmp_path, encoder)
    out = str(tmp_path / "fm.onnx")
    meta = export_fm_onnx(ckpt, out, opset=18, dynamic_batch=True)
    d, v = meta["D"], 3

    rng = np.random.default_rng(0)
    feed = {
        "history": rng.standard_normal((v, n, d)).astype(np.float32),
        "extero": rng.standard_normal((v, meta["extero_sensor_count"] * meta["extero_sensor_dim"])).astype(np.float32),
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


@pytest.mark.parametrize("encoder", ["mlp", "cnn"])
def test_fm_onnx_static_batch(tmp_path, encoder):
    """Fixing the batch axis must remove the host-side shape arithmetic, not change the answer.

    The two halves of the claim in ``export_fm_onnx._export_graph``:

    * **Mechanism.** With a symbolic batch, every head reshape in the traced attention
      blocks computes its target shape at run time, so the graph carries ``Shape`` nodes and
      the ``Gather``/``Unsqueeze``/``Concat`` int64 chain hanging off them. onnxruntime keeps
      shape arithmetic on the HOST, so those became 2687 of 6606 kernel nodes on the
      ``CPUExecutionProvider`` for the shipped 27M generator, plus 71 ``MemcpyFromHost``
      nodes -- and any host node makes onnxruntime refuse CUDA Graph capture outright.
      The graph-level statement of that, and it needs no GPU: run onnxruntime's own
      optimizer and count what survives. Constant folding can only delete the shape
      arithmetic if the shapes are known, so ``Shape`` nodes surviving optimization ARE the
      nodes that get placed on the host. Measured on the shipped generator, the counts line
      up exactly -- 0 surviving ``Shape`` against 3052 all-CUDA nodes, 358 against 6606 with
      2687 on the host. Note the assertion is on the OPTIMIZED graph, not the file: the file
      still contains ``Shape`` nodes either way, and folding them is precisely the work being
      checked.
    * **Numerics.** Same answer to ~1e-6, not bit-exact: with the batch known, torch folds a
      few more constants at export. Three orders below the fp32 CPU-vs-CUDA spread the graph
      already has, so it is a rounding difference rather than a behaviour change.

    Both encoders, because the convolutional path is where the batch-dependent reshapes
    actually live -- the flat MLP path has far fewer of them, so ``cnn`` is the case that
    would regress.
    """
    ort = pytest.importorskip("onnxruntime", exc_type=ImportError)
    ckpt, n, f, nj, k = _tiny_checkpoint(tmp_path, encoder)
    static, dynamic = str(tmp_path / "fm_static.onnx"), str(tmp_path / "fm_dyn.onnx")
    meta = export_fm_onnx(ckpt, static, opset=18)
    export_fm_onnx(ckpt, dynamic, opset=18, dynamic_batch=True)
    d = meta["D"]

    def optimized_ops(path: str, tag: str) -> collections.Counter:
        """Op-type histogram of what onnxruntime is left with after it optimizes ``path``."""
        opt = str(tmp_path / f"opt_{tag}.onnx")
        so = ort.SessionOptions()
        # ORT_ENABLE_ALL, matching deploy (``deploy.tracker_session.make_session``): the
        # constant folding under test is what a deployed session actually runs.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = opt
        ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return collections.Counter(nd.op_type for nd in
                                   onnx.load(opt, load_external_data=False).graph.node)

    ops_static = optimized_ops(static, f"static_{encoder}")
    ops_dynamic = optimized_ops(dynamic, f"dyn_{encoder}")
    assert ops_dynamic["Shape"] > 0, "the symbolic export should need run-time shapes"
    assert ops_static["Shape"] == 0, (
        f"the fixed-batch graph still computes {ops_static['Shape']} shapes at run time "
        "after optimization. Those subgraphs are int64 scalar arithmetic, which onnxruntime "
        "places on the host -- the whole cost this export exists to remove, and any host "
        "node also makes it refuse CUDA Graph capture."
    )
    assert sum(ops_static.values()) < sum(ops_dynamic.values())

    rng = np.random.default_rng(0)
    feed = {
        "history": rng.standard_normal((1, n, d)).astype(np.float32),
        "extero": rng.standard_normal(
            (1, meta["extero_sensor_count"] * meta["extero_sensor_dim"])).astype(np.float32),
        "cmd_tokens": rng.standard_normal(
            (1, meta["num_cmd_tokens"], meta["cmd_dim"])).astype(np.float32),
        "noise": rng.standard_normal((1, f, d)).astype(np.float32),
        "history_null": np.zeros((1, 1), dtype=np.float32),
    }
    got = [ort.InferenceSession(pth, providers=["CPUExecutionProvider"]).run(["trajectory"], feed)[0]
           for pth in (static, dynamic)]
    assert got[0].shape == (1, f, d)
    assert np.isfinite(got[0]).all()
    assert np.abs(got[0] - got[1]).max() < 1e-4

    # A fixed axis must REFUSE a batch, rather than quietly reading the first row: the
    # graph's declared 1 is what constant folding acted on, so a v=2 feed has no meaning.
    sess = ort.InferenceSession(static, providers=["CPUExecutionProvider"])
    with pytest.raises(Exception):
        sess.run(["trajectory"], {kk: np.concatenate([vv, vv], 0) for kk, vv in feed.items()})


def test_a_depth_checkpoint_with_no_clip_band_is_refused(tmp_path):
    """`extero_clip=None` on DEPTH is ambiguous, not absent, so it is refused rather than read.

    It means "use the per-source default", and there are two of those: config.EXTERO_CLIP says
    5.575, deploy.depth_obs.EXTERO_CLIP_NP says 2.0. They agreed until the depth band widened,
    and each is now correct for the artifacts of its own era -- so a None resolved one way in
    training and the other at deploy, which is how a generator trained at +-5.575 came to run
    at +-2.0 and saturate range at 4.75 m. Nothing in the artifact says which era it is from,
    so it can only be stated.
    """
    from robot_rl.flow_matching.online import load_fm_context  # noqa: PLC0415

    ckpt, *_ = _tiny_checkpoint(tmp_path)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert blob["config"]["extero_source"] == "depth"
    assert blob["config"]["extero_clip"] is not None, "the dataset should have stamped it"
    blob["config"]["extero_clip"] = None                      # a pre-stamping artifact
    stale = str(tmp_path / "no_band.pt")
    torch.save(blob, stale)

    with pytest.raises(ValueError, match="ambiguous rather than"):
        load_fm_context(stale, device="cpu")


def test_a_height_scan_checkpoint_with_no_clip_band_still_loads(tmp_path):
    """The scan's default never moved, so a None there names one number and stays legal."""
    from robot_rl.flow_matching.online import load_fm_context  # noqa: PLC0415

    ckpt, *_ = _tiny_checkpoint(tmp_path)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    blob["config"]["extero_source"] = "height_scan"
    blob["config"]["extero_clip"] = None
    blob["config"]["extero_channels"] = 1
    scan = str(tmp_path / "scan_no_band.pt")
    torch.save(blob, scan)

    load_fm_context(scan, device="cpu")                       # must not raise


def test_a_depth_graph_with_no_clip_band_is_refused(tmp_path):
    """The same refusal on the ONNX side, for a graph exported before the band was stated.

    Hand-edited metadata, because the exporter goes through ``load_fm_context`` and so can no
    longer produce one -- which is the point, but it means the deploy guard needs a graph built
    the way a legacy one was.
    """
    onnx_mod = pytest.importorskip("onnx")
    from deploy.fm_session import load_fm_context as deploy_load  # noqa: PLC0415

    ckpt, *_ = _tiny_checkpoint(tmp_path)
    good = str(tmp_path / "fm.onnx")
    export_fm_onnx(ckpt, good, opset=18)

    model = onnx_mod.load(good)
    for prop in model.metadata_props:
        if prop.key == "fm_meta":
            meta = json.loads(prop.value)
            assert meta["extero_source"] == "depth"
            meta["extero_clip"] = None                        # as a legacy graph carried it
            prop.value = json.dumps(meta)
    stale = str(tmp_path / "fm_no_band.onnx")
    onnx_mod.save(model, stale)

    with pytest.raises(ValueError, match="ambiguous rather than absent"):
        deploy_load(stale)
