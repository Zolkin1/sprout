# Copyright 2026 Zachary Olkin. All rights reserved.

"""The convolutional exteroception encoder: grid geometry, pixel order, per-camera weights.

The point of the encoder is that the model can tell WHERE in a depth frame something is, so
the tests that matter here are the ones a flat MLP would fail: that the pixel folding is
row-major, that translating a feature changes the output, and that each camera has its own
trunk.
"""

from __future__ import annotations

import pytest
import torch

from robot_rl.flow_matching.extero_encoder import (
    ExteroImageEncoder,
    ExteroImageEncoderSet,
    conv_grid_shape,
    images_from_flat,
)

# The shipped G1 depth geometry. Tests assert against these rather than a toy shape, because
# the numbers they pin (the 4x4 grid, 16 tokens per camera) are the shipped ones.
_H, _W = 26, 30
_D_MODEL = 32


##
# Grid geometry
##

def test_grid_matches_the_specified_stack():
    """Three stride-2 k3p1 layers take the shipped 26x30 image to a 4x4 grid.

    Pinned because the token count -- and therefore the transformer's sequence length --
    follows from it. A kernel/stride/padding edit that moves the grid should fail here
    rather than surface as a changed sequence length somewhere downstream.
    """
    assert conv_grid_shape((_H, _W), 1) == (13, 15)
    assert conv_grid_shape((_H, _W), 2) == (7, 8)
    assert conv_grid_shape((_H, _W), 3) == (4, 4)
    enc = ExteroImageEncoder((_H, _W), _D_MODEL)
    assert enc.native_grid == (4, 4) and enc.tokens_per_sensor == 16


def test_ceil_halving_floors_at_one_cell():
    """A deep stack on a small image degenerates to 1x1 rather than collapsing.

    ``ceil(n / 2)`` never goes below 1, so there is no arithmetic error to catch here --
    which is worth stating, because "too many layers" is silently a global pooling rather
    than a failure. A non-positive image, on the other hand, is a real error.
    """
    assert conv_grid_shape((2, 2), 4) == (1, 1)
    assert conv_grid_shape((1, 1), 10) == (1, 1)
    with pytest.raises(ValueError, match="must be positive"):
        conv_grid_shape((0, 30), 3)


def test_intermediate_shapes_follow_the_stack():
    """Each convolution really does halve, so the documented 13x15 -> 7x8 -> 4x4 holds."""
    enc = ExteroImageEncoder((_H, _W), _D_MODEL, channels=(16, 32, 64))
    x = images_from_flat(torch.zeros(2, _H * _W), (_H, _W))
    seen = []
    for layer in enc.trunk:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            seen.append(tuple(x.shape[1:]))
    assert seen == [(16, 13, 15), (32, 7, 8), (64, 4, 4)]


##
# Pixel order -- the contract a convolution makes load-bearing
##

def test_flat_pixel_layout_is_row_major():
    """Flat index ``v*W + u`` must land at ``[v, u]``.

    This is the one an MLP could never have caught: it is permutation-invariant over pixels,
    so a transposed fold was harmless. A convolution reads neighbourhoods, so the same
    transpose scrambles every kernel's input while leaving the shapes -- and the loss curve
    -- looking fine. The ordering asserted here is the one every producer writes
    (``mdp.depth_image``, ``pinhole_ray_directions``, ``deploy.depth_obs``).
    """
    flat = torch.arange(_H * _W, dtype=torch.float32)[None]
    img = images_from_flat(flat, (_H, _W))
    assert img.shape == (1, 1, _H, _W)
    for v, u in ((0, 0), (0, _W - 1), (1, 0), (_H - 1, _W - 1), (7, 13)):
        assert img[0, 0, v, u].item() == pytest.approx(v * _W + u)


def test_wrong_row_width_is_refused():
    """A row that does not unfold to the stated image is a mismatch, not a reshape."""
    with pytest.raises(ValueError, match="needs"):
        images_from_flat(torch.zeros(2, _H * _W + 1), (_H, _W))


def test_translating_a_feature_changes_the_output():
    """The whole point: WHERE a step edge sits must reach the model.

    A vertical bar of near-depth is moved across the frame with its pixel MULTISET
    unchanged, so a permutation-invariant embedding (a flat Linear over pixels) is free to
    return the same thing. The convolution must not.
    """
    torch.manual_seed(0)
    enc = ExteroImageEncoder((_H, _W), _D_MODEL).eval()

    def bar_at(col: int) -> torch.Tensor:
        img = torch.zeros(1, _H, _W)
        img[:, :, col] = 1.0
        return img.reshape(1, _H * _W)

    left, right = enc(bar_at(4)), enc(bar_at(_W - 5))
    assert left.shape == (1, 16, _D_MODEL)
    assert not torch.allclose(left, right, atol=1e-6), \
        "moving a feature across the frame left the encoding unchanged"


##
# Token count / pooling
##

def test_grid_override_changes_the_token_count():
    """``grid`` decouples the token count from the input resolution."""
    assert ExteroImageEncoder((_H, _W), _D_MODEL, grid=(2, 2)).tokens_per_sensor == 4
    assert ExteroImageEncoder((_H, _W), _D_MODEL, grid=(1, 1)).tokens_per_sensor == 1
    # ...and the native grid is still reported, so a config can say what it pooled FROM.
    assert ExteroImageEncoder((_H, _W), _D_MODEL, grid=(2, 2)).native_grid == (4, 4)


def test_a_fourth_conv_layer_reaches_two_by_two():
    """Depth is set by ``len(channels)``, so a 4th entry halves once more.

    Worth pinning as the cheaper of the two ways to shrink the sequence: pooling discards
    the trunk's resolution after paying for it, a 4th layer never computes it.
    """
    enc = ExteroImageEncoder((_H, _W), _D_MODEL, channels=(16, 32, 64, 128))
    assert enc.native_grid == (2, 2) and enc.tokens_per_sensor == 4


def test_no_channels_is_refused():
    """An empty stack would be a bare projection wearing a conv encoder's name."""
    with pytest.raises(ValueError, match="at least one convolution"):
        ExteroImageEncoder((_H, _W), _D_MODEL, channels=())


##
# The per-camera set
##

def test_each_camera_has_its_own_trunk():
    """Per-camera weights, not shared -- the near/far optics differ by design.

    Feeding the SAME image as both cameras must give different token blocks, because the
    two trunks are independently initialised. A shared-weight encoder would assert the
    opposite, so this is the test that pins the decision rather than describing it.
    """
    torch.manual_seed(0)
    enc_set = ExteroImageEncoderSet([(_H, _W), (_H, _W)], _D_MODEL).eval()
    assert enc_set.num_sensors == 2
    # Two distinct trunks in the parameter set, not one.
    assert {k.split(".")[1] for k in dict(enc_set.named_parameters()) if k.startswith("encoders.")} == {"0", "1"}

    same = torch.randn(3, 1, _H * _W).expand(3, 2, _H * _W).contiguous()
    with torch.no_grad():   # the shared slot embedding is common to both, so zero it out
        enc_set.spatial.zero_()
    out = enc_set(same)
    assert out.shape == (3, 2, 16, _D_MODEL)
    assert not torch.allclose(out[:, 0], out[:, 1], atol=1e-6), \
        "both cameras produced identical tokens from one image -- the trunks are shared"


def test_spatial_slots_are_shared_across_cameras():
    """One slot embedding of shape (1, 1, S, d), and perturbing a slot must bite."""
    torch.manual_seed(0)
    enc_set = ExteroImageEncoderSet([(_H, _W), (_H, _W)], _D_MODEL).eval()
    assert enc_set.spatial.shape == (1, 1, 16, _D_MODEL)
    rows = torch.randn(2, 2, _H * _W)
    before = enc_set(rows)
    with torch.no_grad():
        enc_set.spatial[0, 0, 7] += 1.0
    after = enc_set(rows)
    # Slot 7 moved for BOTH cameras (shared), and no other slot did.
    assert torch.allclose(after[:, :, 7] - before[:, :, 7], torch.ones(2, 2, _D_MODEL), atol=1e-5)
    other = [s for s in range(16) if s != 7]
    torch.testing.assert_close(after[:, :, other], before[:, :, other])


def test_spatial_embedding_starts_at_zero():
    """Zero-init: at init the encoder is exactly its convolutional output."""
    assert torch.count_nonzero(ExteroImageEncoderSet([(_H, _W)], _D_MODEL).spatial) == 0


def test_ragged_token_counts_are_refused():
    """Cameras sharing one slot embedding must agree on how many slots there are."""
    with pytest.raises(ValueError, match="different token counts"):
        ExteroImageEncoderSet([(_H, _W), (64, 64)], _D_MODEL)


def test_pooling_reconciles_different_resolutions():
    """...and an explicit grid is the way to encode cameras of different resolutions."""
    enc_set = ExteroImageEncoderSet([(_H, _W), (52, 60)], _D_MODEL, grid=(4, 4))
    assert enc_set.tokens_per_sensor == 16
    # Each trunk still folds its OWN resolution, so the two row widths differ.
    assert [e.sensor_dim for e in enc_set.encoders] == [_H * _W, 52 * 60]


def test_sensor_count_mismatch_is_refused():
    """A row count that disagrees with the trunk count is a plumbing bug, not a broadcast."""
    enc_set = ExteroImageEncoderSet([(_H, _W), (_H, _W)], _D_MODEL)
    with pytest.raises(ValueError, match="per-camera encoder"):
        enc_set(torch.zeros(2, 3, _H * _W))


def test_the_conv_branch_is_much_smaller_than_the_mlp_it_replaces():
    """Capacity is being RE-ALLOCATED to structure, not added.

    At the shipped width the flat embedding is ``Linear(780, 512) + Linear(512, 512)``;
    two convolutional trunks plus their projections and the slot embedding come to a
    fraction of it. Worth pinning so the A/B is read as "same size, better structured".
    """
    d_model = 512
    mlp = torch.nn.Sequential(
        torch.nn.Linear(_H * _W, d_model), torch.nn.SiLU(), torch.nn.Linear(d_model, d_model)
    )
    mlp_params = sum(p.numel() for p in mlp.parameters())
    cnn_params = sum(p.numel()
                     for p in ExteroImageEncoderSet([(_H, _W), (_H, _W)], d_model).parameters())
    assert cnn_params < 0.25 * mlp_params, (mlp_params, cnn_params)


##
# The model branch
##

def _model(**over):
    """A small :class:`FlowMatchingTransformer` over the shipped depth pairing."""
    from robot_rl.flow_matching.model import FlowMatchingTransformer
    kw = dict(state_dim=12, future_len=8, num_history_states=2, cmd_dim=3, cmd_stride=4,
              d_model=_D_MODEL, nhead=4, num_layers=2, dim_feedforward=64,
              extero_sensor_dim=_H * _W, extero_sensor_count=2)
    kw.update(over)
    return FlowMatchingTransformer(**kw).eval()


def _model_args(model, batch=3):
    """Positional forward arguments for :func:`_model`, minus the exteroception tensor."""
    return (torch.zeros(batch, model.future_len, model.state_dim), torch.rand(batch),
            torch.zeros(batch, model.num_history_states, model.history_dim))


def test_mlp_extero_parameters_are_unchanged_by_the_branch():
    """The flat path must keep its exact parameter set, or the trained baseline stops loading.

    ``load_state_dict`` is strict, so this set IS the compatibility contract for every
    checkpoint trained before the encoder switch (which the ``.pt`` shim renames onto).
    """
    keys = {k for k in dict(_model().named_parameters()) if "extero" in k}
    assert keys == {"extero_in.0.weight", "extero_in.0.bias",
                    "extero_in.2.weight", "extero_in.2.bias",
                    "extero_type", "extero_id"}


def test_mlp_path_is_numerically_identical_to_the_pre_branch_form():
    """Routing both encoders through one set of adds must not perturb the flat path.

    The branch gave the MLP a length-1 slot axis so the identity embeddings could be added
    once for both encoders. That is only refactoring if it is bit-comparable, so compare it
    against the pre-branch expression (``embed + type + id``, no slot axis) computed here.
    """
    torch.manual_seed(0)
    m = _model()
    with torch.no_grad():   # break the zero-init symmetry so the adds actually show up
        m.extero_type.normal_()
        m.extero_id.normal_()
    rows = torch.randn(3, 2, _H * _W)
    with torch.no_grad():
        want = m.extero_in(rows) + m.extero_type + m.extero_id          # the old form
        got = (m.extero_in(rows).unsqueeze(2)
               + m.extero_type.unsqueeze(1)
               + m.extero_id.unsqueeze(2)).reshape(3, 2, _D_MODEL)      # the new one
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_sequence_length_at_the_shipped_shape():
    """81 tokens flat, 111 with the 4x4 grid -- the numbers the token budget was set from."""
    mlp = _model(num_history_states=4, future_len=62, cmd_stride=5)
    cnn = _model(num_history_states=4, future_len=62, cmd_stride=5, extero_encoder="cnn",
                 extero_image_hw=[[_H, _W], [_H, _W]])
    assert (mlp.extero_seq_tokens, mlp.sequence_len) == (2, 81)
    assert (cnn.extero_seq_tokens, cnn.sequence_len) == (32, 111)


@pytest.mark.parametrize("encoder,extra", [("mlp", {}),
                                           ("cnn", {"extero_image_hw": [[_H, _W], [_H, _W]]})])
def test_encoder_reaches_the_transformer_at_the_declared_length(encoder, extra):
    """Whatever the encoder emits must actually arrive as that many sequence tokens."""
    m = _model(extero_encoder=encoder, **extra)
    seen: list[int] = []
    m.encoder.register_forward_pre_hook(lambda _m, inp: seen.append(inp[0].shape[1]))
    out = m(*_model_args(m), torch.randn(3, 2 * _H * _W), torch.zeros(3, len(m.cmd_steps), 3))
    assert seen[0] == m.sequence_len
    assert out.shape == (3, m.future_len, m.state_dim)


@pytest.mark.parametrize("encoder,extra", [("mlp", {}),
                                           ("cnn", {"extero_image_hw": [[_H, _W], [_H, _W]]})])
def test_forward_accepts_flat_and_sensor_shaped_input(encoder, extra):
    """``(B, T*dim)`` and ``(B, T, dim)`` must agree -- the recorder writes the latter."""
    torch.manual_seed(0)
    m = _model(extero_encoder=encoder, **extra)
    rows = torch.randn(3, 2, _H * _W)
    args, cmd = _model_args(m), torch.zeros(3, len(m.cmd_steps), 3)
    torch.testing.assert_close(m(*args, rows.reshape(3, -1), cmd), m(*args, rows, cmd))


def test_cnn_without_image_shapes_is_refused():
    """The image shape is derived, so its absence means the dataset never ran."""
    with pytest.raises(ValueError, match="extero_image_hw"):
        _model(extero_encoder="cnn")


def test_cnn_image_shape_must_unfold_the_sensor_row():
    """A shape that does not multiply out is a mismatch, not something to reshape around."""
    with pytest.raises(ValueError, match="does not unfold"):
        _model(extero_encoder="cnn", extero_image_hw=[[10, 10], [10, 10]])


def test_cnn_image_shape_count_must_match_the_sensor_count():
    with pytest.raises(ValueError, match="extero_sensor_count"):
        _model(extero_encoder="cnn", extero_image_hw=[[_H, _W]])


def test_unknown_encoder_is_refused():
    with pytest.raises(ValueError, match="unknown extero_encoder"):
        _model(extero_encoder="transformer")


def test_only_one_encoder_is_built():
    """Never both, so a state dict cannot half-load into the wrong path."""
    mlp, cnn = _model(), _model(extero_encoder="cnn", extero_image_hw=[[_H, _W], [_H, _W]])
    assert mlp.extero_in is not None and mlp.extero_cnn is None
    assert cnn.extero_cnn is not None and cnn.extero_in is None


def test_cross_encoder_state_dict_is_refused():
    """A CNN checkpoint must not load into an MLP model (or the reverse), and vice versa."""
    mlp, cnn = _model(), _model(extero_encoder="cnn", extero_image_hw=[[_H, _W], [_H, _W]])
    with pytest.raises(RuntimeError):
        mlp.load_state_dict(cnn.state_dict())
    with pytest.raises(RuntimeError):
        cnn.load_state_dict(mlp.state_dict())


def test_swapping_the_two_cameras_changes_the_cnn_prediction():
    """Near and far must not be an interchangeable pair once encoded."""
    torch.manual_seed(0)
    m = _model(extero_encoder="cnn", extero_image_hw=[[_H, _W], [_H, _W]])
    near, far = torch.randn(2, 1, _H * _W), torch.randn(2, 1, _H * _W)
    args, cmd = _model_args(m, batch=2), torch.zeros(2, len(m.cmd_steps), 3)
    a = m(*args, torch.cat([near, far], dim=1), cmd)
    b = m(*args, torch.cat([far, near], dim=1), cmd)
    assert not torch.allclose(a, b, atol=1e-6)


def test_gradients_reach_every_trunk_and_the_slots():
    """Nothing in the encoder is accidentally detached from the loss."""
    enc_set = ExteroImageEncoderSet([(_H, _W), (_H, _W)], _D_MODEL)
    enc_set(torch.randn(2, 2, _H * _W)).square().mean().backward()
    missing = [n for n, p in enc_set.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing, f"no gradient reached {missing}"


##
# The metric scan path: strides, coordinate channels, validity, and ONE builder
##

def test_strided_trunk_lands_on_the_shipped_token_grid():
    """121x121 under strides (1, 2, 2, 2) must give 16x16 = 256 tokens.

    The ray count and the token count are coupled through the trunk, so this is the arithmetic
    that ties the recorded grid to the transformer's sequence length.
    """
    from robot_rl.flow_matching.extero_encoder import conv_grid_shape

    assert conv_grid_shape((121, 121), 4, (1, 2, 2, 2)) == (16, 16)
    # A leading stride-1 layer is what buys the full-resolution first stage: all-stride-2
    # over-shrinks to 8x8, which is a different model.
    assert conv_grid_shape((121, 121), 4) == (8, 8)
    with pytest.raises(ValueError, match="one stride per layer"):
        conv_grid_shape((121, 121), 4, (1, 2, 2))
    with pytest.raises(ValueError, match="stride must be"):
        conv_grid_shape((121, 121), 2, (1, 0))


def test_metric_grid_adds_coordinate_and_validity_channels():
    """A metric grid feeds the convolution (height, x_m, y_m, validity), not just height."""
    from robot_rl.flow_matching.extero_encoder import ExteroImageEncoder

    plain = ExteroImageEncoder((121, 121), 32, channels=(8, 16), strides=(1, 2))
    assert plain.in_channels == 1 and plain.pos_embed is None

    metric = ExteroImageEncoder((121, 121), 32, channels=(8, 16), strides=(1, 2),
                                coord_extent_m=(3.0, 3.0), miss_sentinel=2.0)
    assert metric.in_channels == 4
    assert metric.pos_embed is not None
    assert metric.trunk[0].weight.shape[1] == 4
    # The coordinate channels are in METRES and span the stated extent.
    x, y = metric.coord_channels[0, 0], metric.coord_channels[0, 1]
    assert x.min().item() == pytest.approx(-1.5) and x.max().item() == pytest.approx(1.5)
    assert y.min().item() == pytest.approx(-1.5) and y.max().item() == pytest.approx(1.5)
    # Token coordinates are the cell MEANS, derived by pooling the same grid -- so they sit
    # inside the extent rather than on its edge.
    xy = metric.token_xy_m[0]
    assert xy.shape == (metric.tokens_per_sensor, 2)
    assert xy[:, 0].abs().max() < 1.5 and xy[:, 1].abs().max() < 1.5


def test_coordinate_channels_break_translation_equivariance():
    """The whole point of the metric path, stated as the exact property it changes.

    A shared kernel is translation-EQUIVARIANT: shift the input by a whole number of token
    cells and the token grid is the SAME multiset of vectors, merely relocated. So a bare
    convolution can report that a step edge exists and structurally cannot report WHERE -- the
    failure this feature exists to remove. Coordinate channels break that equivariance, which
    is what makes position representable.

    Measured as the sorted token norms under a stride-aligned shift: bit-identical (0.0)
    without coordinates, and clearly different with them. A weaker statistic like the mean over
    tokens does move, but only by ~6e-5 at random init, which is too close to noise to pin.
    """
    from robot_rl.flow_matching.extero_encoder import ExteroImageEncoder

    torch.manual_seed(0)
    a = torch.zeros(1, 32, 32); a[0, 4:8, 4:8] = 1.0
    b = torch.zeros(1, 32, 32); b[0, 20:24, 20:24] = 1.0    # +16 px = 8 cells at stride 2

    def sorted_norms(enc, img):
        with torch.no_grad():
            return enc(img.reshape(1, -1))[0].norm(dim=-1).sort().values

    bare = ExteroImageEncoder((32, 32), 16, channels=(4, 8), strides=(1, 2)).eval()
    assert torch.equal(sorted_norms(bare, a), sorted_norms(bare, b))

    metric = ExteroImageEncoder((32, 32), 16, channels=(4, 8), strides=(1, 2),
                                coord_extent_m=(1.0, 1.0)).eval()
    assert (sorted_norms(metric, a) - sorted_norms(metric, b)).abs().max() > 1e-3


def test_validity_channel_marks_the_sentinel():
    """A missed ray must be distinguishable from real geometry at the same value."""
    from robot_rl.flow_matching.extero_encoder import ExteroImageEncoder

    enc = ExteroImageEncoder((8, 8), 16, channels=(4,), strides=(1,),
                             coord_extent_m=(1.0, 1.0), miss_sentinel=2.0).eval()
    flat = torch.full((1, 64), 0.25)
    holes = flat.clone(); holes[0, :10] = 2.0        # ten misses at the band edge
    with torch.no_grad():
        assert not torch.allclose(enc(flat), enc(holes), atol=1e-5)


def test_one_model_builder_serves_the_trainer_and_the_loader():
    """``from_config`` must be the only construction path.

    Two hand-written construction sites drifted once already: the trainer built a 64-token,
    single-channel encoder while the run banner reported the 256-token metric one, and nothing
    failed until the checkpoint would not reload. This pins that a config's stated architecture
    is what actually gets built.
    """
    from robot_rl.flow_matching.config import FlowMatchingConfig
    from robot_rl.flow_matching.model import FlowMatchingTransformer

    cfg = FlowMatchingConfig(
        state_dim=44, history_dim=44, future_len=8, num_history_states=4, cmd_dim=3,
        d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
        extero_source="height_scan", extero_encoder="cnn",
        extero_sensor_count=1, extero_sensor_dim=121 * 121,
        extero_image_hw=[[121, 121]], extero_cnn_channels=[8, 16, 16, 16],
        extero_cnn_strides=[1, 2, 2, 2], extero_coord_extent_m=[3.0, 3.0],
        extero_norm_mode="metric", extero_miss_sentinel=2.0,
    )
    model = FlowMatchingTransformer.from_config(cfg)
    enc = model.extero_cnn.encoders[0]
    assert model.tokens_per_sensor == cfg.tokens_per_sensor == 256
    assert model.sequence_len == cfg.sequence_len
    assert enc.in_channels == 4                      # height + x + y + validity
    assert enc.strides == (1, 2, 2, 2)
    assert enc.pos_embed is not None
    assert enc.miss_sentinel == 2.0

##
# Multi-channel input (depth + the pixel's xyz in the root's yaw frame)
##

def test_the_fold_is_channel_major():
    """Plane ``c`` of the row must arrive as image channel ``c``, pixels intact.

    The row layout is the contract between four producers (the Isaac term, the dataset,
    MuJoCo, hardware) and this fold. Interleaving the channels instead would hand every
    kernel a scrambled image with every shape still correct -- the same class of failure a
    transposed ``image_hw`` causes, and just as invisible.
    """
    row = torch.arange(4 * _H * _W, dtype=torch.float32)
    img = images_from_flat(row[None], (_H, _W), 4)
    assert img.shape == (1, 4, _H, _W)
    for c in range(4):
        assert torch.equal(img[0, c].reshape(-1), row[c * _H * _W:(c + 1) * _H * _W])


def test_a_four_channel_trunk_keeps_the_token_grid():
    """Channels widen the first convolution, they do not change the token count.

    The grid comes from ``(H, W)``, so a 4-channel camera still contributes 16 tokens and
    the transformer's sequence length is unchanged -- which is what makes the 1-channel and
    4-channel models comparable on everything except the input.
    """
    enc = ExteroImageEncoderSet([(_H, _W), (_H, _W)], _D_MODEL, row_channels=4)
    assert enc.encoders[0].trunk[0].in_channels == 4
    assert enc.encoders[0].sensor_dim == 4 * _H * _W
    out = enc(torch.randn(3, 2, 4 * _H * _W))
    assert out.shape == (3, 2, enc.tokens_per_sensor, _D_MODEL)
    assert enc.tokens_per_sensor == ExteroImageEncoderSet(
        [(_H, _W), (_H, _W)], _D_MODEL).tokens_per_sensor


def test_the_single_channel_path_is_the_default():
    """An encoder built without ``in_channels`` is exactly the 1-channel one.

    Every existing checkpoint was trained that way, so the default has to stay the old
    architecture -- the same reason ``extero_encoder`` defaults to "mlp" on a bare model.
    """
    enc = ExteroImageEncoderSet([(_H, _W)], _D_MODEL)
    assert enc.encoders[0].trunk[0].in_channels == 1
    assert enc.encoders[0].sensor_dim == _H * _W


def test_a_row_of_the_wrong_channel_count_is_refused():
    """A 1-channel row into a 4-channel trunk is a mismatch, not something to broadcast."""
    enc = ExteroImageEncoderSet([(_H, _W)], _D_MODEL, row_channels=4)
    with pytest.raises(ValueError, match="channel"):
        enc(torch.randn(2, 1, _H * _W))
