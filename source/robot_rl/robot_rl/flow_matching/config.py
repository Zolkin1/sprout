# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching future-state prediction model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from motion_gen.features import DEFAULT_MJCF

from .extero_encoder import conv_grid_shape


#: Recognised :attr:`FlowMatchingConfig.extero_source` values.  These are also the keys
#: ``record_data.py`` writes, so the dataset looks the array up by this name directly.
EXTERO_SOURCES: tuple[str, ...] = ("height_scan", "depth")

#: Recognised :attr:`FlowMatchingConfig.extero_encoder` values (after ``"auto"`` resolves).
EXTERO_ENCODERS: tuple[str, ...] = ("mlp", "cnn")

#: Recognised :attr:`FlowMatchingConfig.extero_norm_mode` values.
#:
#: ``per_element`` (one fitted statistic per pixel) is GONE. A per-pixel affine is a spatially
#: varying transform applied before a shared convolution kernel sees the image, which is
#: exactly what weight sharing assumes away -- the same step edge at row 5 and row 15 arrives
#: as two different patterns. It was the historical default and nothing wants it back.
EXTERO_NORM_MODES: tuple[str, ...] = ("contract", "per_channel", "per_sensor", "metric")

#: Default symmetric clip band per :data:`EXTERO_SOURCES` entry, applied by
#: ``transforms.sanitize_extero`` before normalization.
#:
#: Lives here rather than in ``transforms`` because it is keyed by a config enum and is now
#: overridable per run via :attr:`FlowMatchingConfig.extero_clip`; ``transforms`` re-exports
#: the two scalars under their historical names. Depth's band is in NORMALISED units (the obs
#: term already applied ``(d - 0.75) * 0.5``), the scan's in metres -- which is why they differ.
#:
#: Depth's 5.575 is MEASURED, not a round number: at the old +-2.0 the band saturated at 4.75 m
#: of range while the far camera sees 11.9 m, clipping 1.30% of its pixels and touching 13.8%
#: of windows, against a 99th percentile of 5.22 m. Recordings store depth UNCLIPPED, so
#: widening this recovers signal already on disk; 5.575 measures 0.00% saturated. Changing it
#: re-resolves any checkpoint that stored ``extero_clip=None``, which is why ``train.py``
#: prints the band it resolved.
EXTERO_CLIP: dict[str, float] = {"height_scan": 1.0, "depth": 5.575}

#: Recognised :attr:`FlowMatchingConfig.attention_mode` values.
ATTENTION_MODES: tuple[str, ...] = ("full", "prefix")

#: Config keys renamed when the exteroception contract stopped being height-scan-shaped:
#: old name -> new name. Applied by :meth:`FlowMatchingConfig.from_dict`, so a checkpoint
#: trained before the rename still loads and can still serve as an A/B control.
#:
#: One-time and deliberately narrow. The ONNX side has NO such mapping
#: (``deploy.fm_session`` refuses a stale graph and says to re-export) because the two
#: artifacts cost different things to regenerate.
_LEGACY_KEYS: dict[str, str] = {
    "extero_tokens": "extero_sensor_count",
    "height_dim": "extero_sensor_dim",
    "height_norm_eps": "extero_norm_eps",
    # Renamed because the old name lied: it is a derived bound that is ASSERTED, never
    # applied, and "clip" reads as a transform on the data. Short-lived, but a run trained
    # under the old name exists, and ``from_dict`` drops unknown keys silently -- which would
    # look like the envelope re-deriving itself rather than a value being lost.
    "extero_xyz_clip": "extero_xyz_envelope",
}


#: Largest re-plan period ``fm_update_every`` a shipped horizon must be able to serve.
#:
#: The deployed default is K=1, but a checkpoint sized for K=1 alone CANNOT be used at any
#: larger period -- the horizon check refuses it -- and K=12 measured 7.5 pts of survival
#: better than K=1. Since recovering that costs a full retrain, every generator is trained
#: with the headroom whether or not the run that uses it wants K>1. See
#: :attr:`FlowMatchingConfig.future_len` for the arithmetic and what the slack costs.
MAX_SERVABLE_REPLAN_PERIOD: int = 12



@dataclass
class FlowMatchingConfig:
    """Hyperparameters for the flow-matching model, data pipeline, and training.

    ``state_dim`` and ``cmd_dim`` are resolved from the dataset at build time (they
    depend on the joint/extra-frame count and the command width), so the defaults
    here are placeholders.

    Architecture (single transformer): tokens are assembled as
    ``[history(N) | extero(T) | cmd(Ncmd) | future(F)]``. The history, exteroception and
    strided command tokens are SEPARATE conditioning tokens (never noised, not part of
    the denoised window). The model denoises the ``F`` future tokens -- ALL future, no
    past echo, no inpainting / known-mask.

    **The model is AUTOREGRESSIVE over its own output.** Every token, history and future
    alike, is one node of the reference window in ``flow_matching.window``'s
    representation -- the exact vector ``mdp.reference_window`` feeds the policy. The
    history tokens are the ``N`` nodes this model emitted at the ``N`` previous control
    steps, each anchored on the robot as it was at *that* step. So the generator conditions
    on the plan it is continuing rather than on a state estimate, its output IS the
    observation, and nothing on the path from network to policy touches a world coordinate
    -- in simulation or at deploy.

    The consequence to keep in mind when reading results: at training time that history is
    the CLIP TEACHER's applied reference, never the model's own, so this carries textbook
    exposure bias. The defences are ``null_cond_prob`` (during training the history is
    replaced by a learned null embedding, which also covers the first ``N`` steps after a
    reset, when no history exists yet) and the per-channel history-token noise below.
    """

    # -- data / windowing --
    # Future window length F: the denoised sequence of reference-window nodes. Node 0 is
    # the CURRENT step, nodes 1..F-1 the future, all anchored on the measured robot at the
    # re-plan -- the same anchor and the same representation the policy's
    # ``mdp.reference_window`` observation uses (see ``flow_matching.window``).
    #
    # F is what the POLICY asks of one prediction, plus the re-plan slack:
    #
    #     F = max(REFERENCE_OFFSETS) + MAX_SERVABLE_REPLAN_PERIOD
    #       = 50                     + 12
    #       = 62
    #
    # (``REFERENCE_OFFSETS`` is the teacher's reference-window contract in
    # ``g1_teacher_env_cfg.py``: ``(0, 1, 2, 5, 10, ... 50)``, 13 samples.)  Note the first
    # bound is the FURTHEST offset, not the sample count -- the window is non-uniform, and
    # its three near samples ask for nodes the prediction already contains, so they cost no
    # horizon.
    #
    # THE SLACK TERM IS LOAD-BEARING, which is worth stating because it was briefly dropped.
    # A prediction served ``j`` steps after the re-plan has its cursor ``j`` nodes in, so the
    # furthest sample reads node ``max_offset + j`` -- hence ``F >= max_offset + K`` for a
    # re-plan period K (``FmTrackingCommand._check_horizon``; nothing clamps, a short horizon
    # raises). Sizing F to ``max_offset + 1`` makes a checkpoint K=1-ONLY, and K=12 measured
    # 7.5 pts better than K=1 on the 24-env sweep, so that is headroom worth paying for: it
    # is one retrain to get back and cannot be recovered from an existing checkpoint.
    # ``test_teacher_window_fits_the_shipped_horizon`` asserts this bound so it cannot be
    # removed silently again.
    #
    # The cost of the slack, so it is a choice and not an accident: 11 extra tokens to
    # denoise per re-plan (~15-20% slower per step), and at K=1 nodes 51..61 are predicted
    # and never read. The FM is ~99% of the deploy step budget, so that is not free.
    #
    # cmd_stride (=5) divides F, so the 13 command tokens land on nodes [0, 5, ..., 60].
    # The first 11 cover the STRIDED tail of the teacher's window; they do not cover its two
    # near samples (offsets 1 and 2), and do not need to -- these are the generator's own
    # conditioning tokens, not the policy's ``command_window`` observation, and the commanded
    # velocity is piecewise constant over a 0.04 s gap.
    #
    # RAISING F needs a recording with at least that many ref_future nodes (record_data.py
    # cuts at ``--ref_future_len``, default this value) and a retrain -- but NOT necessarily
    # a re-record: the dataset truncates a longer recording to F, so the 75-node depth_v5
    # recording already serves anything up to 75.
    future_len: int = 62
    # N history (context) tokens: the reference frames the generator itself emitted at the
    # 4 steps before this one. 4 rather than 2 because these are now a *plan* being
    # continued rather than a state being read -- more of the recent trajectory to
    # extrapolate from, and 0.08 s of it instead of 0.04 s.
    num_history_states: int = 4
    stride: int = 1
    val_fraction: float = 0.1
    precompute: bool = True

    # Future-target source. "ego_anchored_clip_future": predict the motion clip
    # ego-anchored onto the robot at the snap (re-anchor) event -- the new robot-state-
    # conditioned paradigm (requires ``ref_future`` + a snap flag in the recording).
    # "actual" predicts the recorded robot state; "reference" predicts the per-step
    # tracked reference. Auto-resolved from the recording's ``ref_source`` when unset.
    target_source: str = "ego_anchored_clip_future"

    # Recovery-aware windowing for perturbed recordings (record_data.py --perturb,
    # which logs a per-step ``perturb_active`` flag). "clean_prediction" (default)
    # keeps only windows whose PREDICTION region (steps N..F) is fully clean, while
    # the HISTORY (steps 0..N) may be perturbed -- so the model learns to predict the
    # clean recovery from a bad-state history. "all" uses every valid window
    # (perturbed steps in history and prediction alike). Recordings without a
    # ``perturb_active`` array fall back to "all" regardless.
    recovery_mode: str = "clean_prediction"

    # Drop the DISTRACTOR-family windows (boxes the robot walks AROUND, not onto). Those clips
    # are ego-anchored like flat walking, so they do produce honest snap windows -- but whether
    # they help the generator or just add obstacle-shaped noise is unresolved. Recordings carry a
    # per-window ``is_distractor`` tag, so this is a train-time switch: no re-recording needed to
    # A/B it. Errors (rather than silently training on them) if the recording has no tag.
    # True: the MVP trained without distractor windows, and the flag exists precisely so
    # A/B-ing it needs no re-record (every window carries `is_distractor`).
    skip_distractors: bool = True

    # Clip FAMILIES to drop, by name (see RecordingMeta.family_names; the ".h5" the composer
    # stamps is already stripped, so write "family_jump_off"). A train-time switch for the
    # same reason skip_distractors is one: the recording tags every window with its family,
    # so A/B-ing a family costs no re-record.
    #
    # The case this exists for: "family_jump_off" is the BACKWARDS jump-off-a-box family, and
    # the depth cameras face forward -- the drop behind the robot is not observable, so those
    # windows ask the generator to predict something it cannot see. Errors (rather than
    # silently training on them) if the recording predates the tag, or if a named family is
    # not in the recording at all -- a typo would otherwise read as "nothing to drop".
    skip_families: list[str] | None = None

    # Oversample windows that CROSS a terrain transition until they are this fraction of each
    # training epoch (e.g. 0.5). ``None`` = natural distribution, which on the v5 library is
    # 17.8% of the trained subset.
    #
    # A sampler weight, not a filter and not a loss weight: epoch size is unchanged and every
    # window keeps its own loss, so only the RATE at which transitions are drawn changes.
    # Weight is w = p*n_flat / ((1-p)*n_trans) on transition windows and 1 elsewhere, which
    # makes the expected per-epoch fraction exactly ``p``. Reaching 0.5 from 17.8% is ~4.6x,
    # so each transition window is seen ~4.6x per epoch -- watch val for overfitting on them.
    #
    # "Crosses a transition" is GROUND TRUTH, not a height heuristic: an ``anchor_mask_t``
    # boundary strictly inside ``[clip_step, clip_step + F)``, joined through the recorded
    # ``clip_id``/``clip_step``. Needs a recording carrying those (and the clip library on
    # disk); errors rather than silently falling back to uniform sampling.
    transition_fraction: float | None = None

    # Clip library to join ``clip_id``/``clip_step`` against for ``transition_fraction``.
    # Empty = use ``RecordingMeta.clip_library_path``. The recorded
    # ``clip_library_sha256`` is verified either way, because clip ids are positions in ONE
    # library and a regenerated library reuses them for different motions.
    clip_library_path: str = ""

    # Dedicated recorded command array (e.g. "target_vel") giving the per-step
    # commanded velocity over the window. It is sampled every ``cmd_stride`` steps
    # (indices 0, cmd_stride, ...) and each sample becomes its own conditioning token
    # (a separate command-embedding MLP). ``None`` means no command conditioning.
    cmd_array: str | None = None
    cmd_stride: int = 5

    # -- what one node of the window contains --
    # The blocks a window node packs, in packing order, out of
    # ``window.REFERENCE_WINDOW_BLOCK_NAMES`` (base_pose, base_vel, joint_pos, joint_vel,
    # ee_pos). This has to be the TEACHER's selection: the generator's output IS the
    # policy's ``reference_window`` observation, so predicting a different set of blocks
    # is not a lossy approximation, it is a scrambled observation of the wrong width.
    #
    # ``None`` means "resolve from the recording" -- ``RecordingMeta.reference_blocks``
    # records what the teacher that produced it observed, and a recording predating that
    # field is necessarily a full five-block one. Leave it None unless you are deliberately
    # training a generator for a DIFFERENT teacher than the one that was recorded; the
    # agreement is then checked where both sides are actually present, at load
    # (``FmTrackingCommand``) and at deploy (``GenTrackerPolicy.load``).
    #
    # Dropping a block also drops the loss terms that index it -- see
    # ``dataset.resolve_block_selection``, which zeroes them and says so.
    reference_blocks: list[str] | None = None
    # -- dimensions (resolved from the dataset at build time) --
    # Per-node width of every token: the reference-window layout for
    # ``reference_blocks`` (see ``flow_matching.window.window_dim``). History and
    # predicted nodes are the SAME representation -- the history is just nodes this model
    # emitted earlier -- so they share one normalizer.
    state_dim: int = 94        # all five blocks at nj=29, K=4; resolved at build time
    # Per-node history-token width (= state_dim; resolved at build time).
    history_dim: int = 94
    # Which recorded signal feeds the exteroception conditioning tokens.
    #   "depth"       -- the two torso depth cameras, ONE token each (T=2), from
    #                    ``G1-teacher-clf-terrain-depth*`` recordings. DEFAULT, because it
    #                    is the deployable signal: the hardware has cameras, not a
    #                    heightmap, and there is nothing on the robot that produces the
    #                    scan. Defaulting to the thing that cannot ship made it too easy
    #                    to train an FM that was never going to leave the simulator.
    #   "height_scan" -- the pelvis ray-cast grid (176 rays) as ONE token. Sim-only; still
    #                    the right choice for a scan-conditioned ablation or a task
    #                    without cameras (pass ``--extero_source height_scan``).
    # ``record_data.py`` stores BOTH arrays when the task has the cameras, so one
    # recording session can train either FM and this is a train-time choice. A recording
    # that lacks the selected array fails loudly at dataset build (see dataset.py), which
    # is what a cameraless task now gets by default.
    extero_source: str = "depth"
    # CAMERA LATENCY. Probability of each depth LAG, index = lag in control steps, so
    # ``(0.0, 0.40, 0.35, 0.25)`` never serves a fresh frame and serves 20 / 40 / 60 ms old
    # ones at those rates. ``None`` disables the model and always uses the frame captured at
    # the re-plan, which is what every recording before 2026-08-21 supports.
    #
    # The depth pipeline is not instantaneous on the robot -- the image is computed and shipped
    # over ROS, measured at 20-40 ms typically and peaking near 60. The camera runs at 50 Hz and
    # so does control, so one lag step is exactly 20 ms and lags 1..3 span that range exactly.
    #
    # **Lag 0 gets zero weight on purpose.** A frame cannot arrive in the step it was captured
    # in, so training on it spends capacity on a case the robot never sees -- and it is the
    # EASIEST case, which makes the model optimistic about a signal it will always get late.
    # It stays reachable only as a no-latency ablation, since the recorder keeps the frame.
    #
    # Drawn once per WINDOW at dataset build (the extero array is materialised there), seeded
    # off ``seed``. With a full recording every lag is thoroughly represented across windows;
    # what it does not do is show the same window at two lags. Recording all the frames and
    # choosing here still means the distribution can change without re-recording, and a
    # no-latency ablation is free. It models pure JITTER -- real latency also has a persistent
    # pipeline component, which a per-episode base would capture and this does not.
    #
    # The lagged frame is unprojected with the pose it was CAPTURED with, and expressed in that
    # capture-time frame. Using the re-plan pose instead would displace every pixel by the
    # egomotion, which is the registration error the xyz channels exist to remove; re-expressing
    # it in the re-plan FRAME would need the root displacement over the lag, and this robot has
    # no base-velocity estimate to integrate. So the terrain arrives correctly registered to
    # where the robot WAS, and the model compensates from its history.
    depth_latency_probs: tuple[float, ...] | None = None

    # How many exteroception SENSOR readings there are, and how wide ONE of them is. Both
    # resolved from the recorded array's shape at dataset build time: 1 x 176 rays for the
    # scan, 2 x (H*W) pixels for the cameras.
    #
    # Sensor-neutral names on purpose. These were ``extero_tokens`` and ``height_dim``, from
    # when exteroception WAS a height scan and one reading was one transformer token. Neither
    # holds any more: the scan is not a camera, and one reading can become a grid of tokens
    # (see the extero encoder), so "tokens" no longer means "sensors". The transformer's
    # sequence count is a separate, derived quantity.
    extero_sensor_count: int = 1
    extero_sensor_dim: int = 176

    # How ONE sensor reading becomes transformer tokens.
    #   "cnn" -- a 3-layer stride-2 conv trunk PER CAMERA, so each camera contributes a grid
    #            of tokens (16 on the 30x26 pair) each carrying a learned spatial-slot
    #            embedding. The deployable choice, and the reason this switch exists: the
    #            flat path is permutation-invariant over pixels, so nothing in the model can
    #            attend to WHERE in the frame a step edge is -- which is the shape of the
    #            measured registration failure (fm_registration_probe.py).
    #   "mlp" -- the original 2-layer MLP, ONE token per sensor. The only option for
    #            ``height_scan`` (an FM recording carries the scan's ray count but not its
    #            grid shape, so there is no image to convolve), and the A/B baseline.
    #   "auto" -- resolved at dataset build from ``extero_source``: depth -> cnn, height
    #            scan -> mlp. Same treatment ``cmd_array`` and ``target_source`` get.
    #
    # NOTE the asymmetry with the dataclass default below: this field defaults to "auto",
    # but ``FlowMatchingTransformer``'s constructor argument defaults to "mlp". That is
    # deliberate -- a bare model built in a test keeps the legacy parameter set, while a
    # trained config gets the current architecture.
    #: Channels per exteroception PIXEL. 1 = depth only, the original signal. 4 adds the
    #: pixel's 3-D position in the root's yaw-aligned frame -- the same frame the generator
    #: predicts its window in -- so a step edge arrives as "0.8 m ahead, 0.15 m up" rather
    #: than as "row 14, column 8, 2.1 m" and the model no longer has to infer the camera
    #: pose it is never shown. See ``mdp.observations.depth.depth_xyz_stack``.
    #:
    #: The channels are DERIVED from the depth image by the pinhole model, never stored: one
    #: helper (``deploy.depth_obs.unproject_to_yaw_frame`` and its torch mirror) serves the
    #: Isaac term, the dataset, MuJoCo and hardware, so the four cannot describe different
    #: geometry. That also means the recording carries only depth plus the camera pose.
    #:
    #: ``extero_sensor_dim`` counts CHANNELS TOO (it is ``channels * H * W``), because it is
    #: the width of one sensor's row as the model receives it. Anything comparing that width
    #: against an image shape has to divide by this first -- see ``check_extero_encoder``.
    #: Meaningless for the MLP encoder and for the height scan.
    #:
    #: ``None`` means RESOLVE FROM THE SOURCE (:meth:`resolve_extero_channels`): 5 for depth,
    #: 1 for a height scan, whose rays already ARE ground points. Same idiom as
    #: ``extero_encoder="auto"`` and ``extero_clip=None`` -- one default that is correct for
    #: either sensor, rather than a number that is right for one and a hard error for the
    #: other. The dataset resolves it and writes the int back, so a checkpoint always states
    #: a number and every consumer downstream reads one.
    extero_channels: int | None = None
    #: Per-camera ``(fx, fy, cx, cy)``, DERIVED from the recording's ``depth_camera_spec``,
    #: enabling the angular token position embedding. Per camera and not shared: the G1's two
    #: are 100x70 deg and 70x42 deg, so one set would mislabel the narrower one's tokens by
    #: 1.43x while every shape stayed correct. ``None`` for the height scan, whose pixels are
    #: ground points and use ``extero_coord_extent_m`` instead.
    extero_camera_intrinsics: list[list[float]] | None = None
    #: Names of those channels, in order, for diagnostics and the ONNX metadata. Purely
    #: descriptive: the ORDER that matters is the one the encoder folds, and that is fixed by
    #: ``extero_channels``.
    extero_channel_names: list[str] = field(default_factory=lambda: ["d"])
    extero_encoder: str = "auto"
    # Per-sensor ``[[H, W], ...]``, DERIVED from the recording's camera contract at dataset
    # build time (``_resolve_extero_spec``) and then verified wherever it is read again --
    # never typed by hand. Per-camera trunks mean two cameras with the same pixel count but
    # transposed shape are legal, so this is stated per sensor rather than once.
    # ``None`` for the height scan, which has no image shape in an FM recording.
    extero_image_hw: list[list[int]] | None = None
    # Output channels per stride-2 k3p1 convolution; ``len`` sets the depth, so it sets how
    # many times the image is halved. Three entries take 26x30 to 4x4; a fourth reaches 2x2
    # natively, which is the CHEAPER way to shrink the sequence (pooling pays for the
    # resolution and then discards it).
    extero_cnn_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    # Optional ``[gh, gw]`` average-pool target after the trunk, i.e. tokens per camera.
    # ``None`` keeps the trunk's native grid. This is the knob that decouples the token
    # count from the input resolution -- and the one that isolates "a CNN beats an MLP" from
    # "16 tokens beat 1", since ``[1, 1]`` gives the legacy token count through a conv trunk.
    extero_cnn_grid: list[int] | None = None
    # Per-layer convolution STRIDE, one entry per ``extero_cnn_channels`` entry. ``None`` means
    # stride 2 everywhere. A LEADING stride-1 layer keeps full resolution through the first
    # stage while the stack still lands on a small token grid: 121 -> 121 -> 61 -> 31 -> 16
    # under (1, 2, 2, 2) for a dense scan, and 26x30 -> 26x30 -> 13x15 -> 7x8 for the cameras.
    # The grid target and this list are coupled -- changing one without the other silently
    # changes the transformer's sequence length.
    #
    # Default (1, 2, 2): 56 tokens per camera rather than the 16 that (2, 2, 2) gives, because
    # the first stage convolves the image at its native 26x30 instead of throwing three
    # quarters of it away before the first non-linearity. Sequence length 191 for the G1's two
    # cameras.
    extero_cnn_strides: list[int] | None = field(default_factory=lambda: [1, 2, 2])
    # ``[size_x, size_y]`` metric extents of a robot-centred exteroception grid, DERIVED from
    # the recording's ``height_scan_spec`` (never typed). When set, the encoder appends each
    # pixel's ``(x, y)`` offset in METRES as two input channels and adds a Fourier embedding of
    # each token's own cell-centre coordinate.
    #
    # This is the targeted fix for the registration failure, not a refinement: a shared
    # convolution kernel is translation-EQUIVARIANT, so it can say a step edge is present but
    # structurally cannot say WHERE it is -- and "where, relative to the robot" is the whole
    # quantity being measured. ``None`` for depth, whose pixels are rays rather than ground
    # points and have no such planar coordinate.
    extero_coord_extent_m: list[float] | None = None
    # Flat-row ordering of that metric grid (``GridPatternCfg.ordering``), so the coordinate
    # channels are laid out the same way the reading is folded.
    extero_grid_ordering: str = "xy"
    # Frequency bands per axis in the token position embedding. A single linear map of (x, y)
    # can only express a plane, which is too smooth to separate neighbouring tokens.
    extero_fourier_bands: int = 4
    # Std floor for the exteroception normalizer. On flat ground the per-ray scan is nearly
    # constant, so the raw std collapses toward the base Normalizer eps (1e-6) and blows up
    # normalization; a larger floor (~1e-2 m) keeps a near-flat scan well-scaled.
    extero_norm_eps: float = 1e-2
    # How the exteroception normalizer is FIT. Every mode is UNIFORM OVER PIXELS, which is the
    # property a shared convolution kernel needs (see :data:`EXTERO_NORM_MODES` for why the old
    # per-pixel mode is gone); they differ in how much else they pool over.
    #   "contract"    -- NOTHING is fitted. The default for depth. Every constant is derived
    #                    from the camera contract and the robot's nominal stance, so it is the
    #                    same number in every recording of the same robot:
    #                      * ``d`` -- IDENTITY. It arrives already declared-normalised,
    #                        ``(depth - 0.75) * 0.5`` out of the contract, and clipped to
    #                        ``extero_clip``. Fitting it again would make a contract constant
    #                        a property of a dataset.
    #                      * ``xyz`` -- ONE declared scale (``extero_xyz_scale``) shared by both
    #                        cameras and all three axes, so a metre reads the same number
    #                        everywhere -- including ACROSS cameras, which matters because both
    #                        feed one transformer and the model has to compare them. The OFFSET
    #                        is per camera and geometric: the boresight's intersection with the
    #                        nominal ground plane (``extero_xyz_offset``, derived at dataset
    #                        build, rounded, recorded). Per camera because the two look 65 and
    #                        35 degrees down from different heights, so one shared offset
    #                        cannot centre both.
    #                      * ``v`` -- identity, as in every mode.
    #                    The cost, stated plainly: the offsets centre the NOMINAL flat-ground
    #                    view, not whatever terrain a recording happened to sample, so on
    #                    stairs and drop-offs the empirical centroid sits some way off. The
    #                    dataset PRINTS declared-vs-measured so that gap is visible rather than
    #                    assumed; widen ``nominal_range_m`` if it matters.
    #   "per_channel" -- one fitted (mean, std) per sensor per CHANNEL GROUP.
    #   "per_sensor"  -- one (mean, std) per sensor, pooled across channels too.
    #   "metric"      -- a FIXED declared affine ``(x - extero_norm_offset) / extero_norm_scale``,
    #                    no fitting at all.
    #
    # ``per_channel`` exists because a 5-channel depth row is not one quantity: channel 0 is
    # normalised depth (band +-5.575), channels 1-3 are metres out to ~11 m, channel 4 is an
    # indicator in {0, 1}. ``per_sensor`` pools all of them into one statistic whose std is
    # dominated by xyz, which compresses depth and squashes validity toward invisibility.
    # The GROUPS are the fix, and they are not simply per channel:
    #   * ``d``       -- its own statistic.
    #   * ``x, y, z`` -- ONE SCALE across all three, so a metre stays a metre in every
    #                    direction; a per-axis scale would rescale the axes against each other
    #                    and destroy the metric comparability the channels exist to provide.
    #                    The OFFSET stays per axis: subtracting one is a rigid translation of
    #                    the cloud, so it preserves every relative distance, and it actually
    #                    centres the channel. Sharing it does not -- measured on the far
    #                    camera, x sits at +6.66 m and z at -3.88 m around a pooled +0.79, so
    #                    one offset parks them at +1.94 and -1.55 sigma before the first conv.
    #   * ``v``       -- NOT standardized at all (``standardize_mask`` False). It is already
    #                    0/1; fitting it would map "valid" and "invalid" to two arbitrary
    #                    values that drift with the invalid RATE of whatever recording was
    #                    used, and a run on cleaner data would hand the network a different
    #                    number for the same fact.
    # For a 1-channel sensor (any height scan) ``per_channel`` is identical to ``per_sensor``.
    #
    # The stored statistics keep their ``(T*K,)`` shape in every mode (the pooled value is
    # broadcast back into it), so the Normalizer, the checkpoint, the folded ONNX
    # normalization and the whole deploy path are shape-identical -- only the numbers change.
    #
    # ``extero_norm_eps`` matters here: a pixel that almost always reads the far-clip sentinel
    # would get a floored std under a per-pixel fit, so its rare real readings became large
    # outliers that one shared kernel then propagated across a whole feature map. Pooling
    # removes that failure mode rather than flooring it.
    #
    # "metric" is the mode for a privileged metric height map. Both fitted modes make the input
    # scale a property of the TRAINING SET rather than of the world: the same terrain maps to
    # different network inputs in two different recordings, and a checkpoint's notion of "1 m"
    # depends on what it happened to be trained on. A declared affine removes that, and brings
    # the scan to parity with depth -- whose first stage, ``(d - 0.75) * 0.5`` inside
    # ``mdp.depth_image``, has always been declared constants carried in the camera contract.
    #
    # That critique applies to ``per_channel`` as well and is the known cost of the default: it
    # is fitted, so the depth and xyz scales are properties of the recording. The trade is
    # deliberate -- one declared scale cannot serve channels in three different unit systems,
    # and the group structure is what keeps the axes mutually comparable. Declaring a scale PER
    # GROUP would get both, and is the obvious next step if recording-to-recording drift ever
    # shows up as a real effect.
    extero_norm_mode: str = "contract"
    #: Declared divisor for the xyz channels under ``extero_norm_mode="contract"``, in metres.
    #: ONE constant for every camera and every axis -- that sharing is the whole point, since a
    #: per-camera or per-axis divisor would make a metre mean different things in different
    #: parts of the same input. 3.0 puts the far camera's full max-range cone inside ~3.2 and
    #: the near camera's inside ~2.4.
    extero_xyz_scale: float = 3.0
    #: Per-camera xyz offset in metres, ``[[x, y, z], ...]`` in sensor order. DERIVED at dataset
    #: build from the camera contract plus the robot's nominal stance (the boresight's
    #: intersection with the nominal ground plane, rounded to 5 cm) and written back, so the
    #: checkpoint carries the constants it trained under and ``fm.onnx`` folds them into its
    #: normalization like any other affine. ``None`` until the dataset resolves it.
    extero_xyz_offset: list[list[float]] | None = None
    #: Nominal ground plane, metres below the pelvis at zero joints. DERIVED from FK on
    #: ``fk_model_path`` (the ankle roll link, plus the sole) unless set. The one constant here
    #: that answers to the ROBOT rather than to the camera, and it is equally declared.
    extero_nominal_ground_z: float | None = None
    #: Symmetric bound on the NORMALISED xyz channels, derived from each camera's own max-range
    #: cone post-offset and rounded up.
    #:
    #: **ASSERTED, never applied**, which is why it is an envelope rather than a clip -- it was
    #: called ``extero_xyz_clip`` and the name claimed a transform the code does not perform.
    #: Nothing clamps against it on either path, and nothing should: because the depth channel
    #: is already clipped to ``extero_clip`` and xyz is derived from that same post-clip array,
    #: no physically reachable point can exceed it, so a value that does means the contract and
    #: the data disagree. The dataset RAISES there rather than squashing, since a clamp sized to
    #: bind on real geometry would repeat the mistake the +-2.0 depth band made.
    #:
    #: Being asserted rather than applied is also what keeps it off the deploy path entirely:
    #: there is no runtime clamp to mirror, so no train/serve asymmetry to get wrong.
    extero_xyz_envelope: float | None = None
    # The declared affine for ``extero_norm_mode="metric"``. The default divides by 1.0, i.e.
    # the network input IS metres -- which is the point, and makes the value directly readable
    # in a diagnostic. Recorded into the checkpoint so the band a run trained under is auditable.
    extero_norm_offset: float = 0.0
    extero_norm_scale: float = 1.0
    # Symmetric clip band applied by ``sanitize_extero`` BEFORE normalization. ``None`` keeps
    # the per-source default (:data:`EXTERO_CLIP`), so every existing checkpoint is unchanged.
    #
    # The scan's 1.0 m default is too tight for a large grid: measured on the real recording,
    # 33% of `stairs_dn_pure` rays already exceed +1 m at the 176-ray grid, and a 3 m window
    # destroys ~11% of the image overall (42% on descents), concentrated downrange where the
    # planning happens. The true range on this terrain is about [-1.6, +5.5] m; +-2.0 leaves
    # 0.5% clipped. Widening it is safe precisely because the generator has its OWN scanner --
    # the teacher's ``clip=(-1, 1)`` obs terms are untouched, so the critic's input distribution
    # does not move.
    extero_clip: float | None = None
    # Normalized value a MISSED ray takes, DERIVED at dataset build under the metric affine.
    # When set, the encoder appends a validity channel so a hole is distinguishable from real
    # geometry. Only computable under a declared affine (otherwise the sentinel depends on the
    # fitted statistics), and only meaningful for a ray-cast source; ``None`` disables it.
    #
    # It matters because the two ends of the band are both physically reachable: a missed ray
    # now maps to +clip, the drop-off end, and real terrain also reaches large positive values
    # where the robot stands high above a distant floor. Without the mask those are one number.
    extero_miss_sentinel: float | None = None
    # DECLARED per-block scales for the state (window) normalizer. ``None`` = fit that block
    # per channel from the data, which is the historical behaviour and the default.
    #
    # A declared scale sets ``mean = 0`` and ``std = scale`` for EVERY channel of the block, so
    # the block is divided by one number instead of standardized per channel. Two reasons that
    # is better for the geometric blocks:
    #
    # * it PRESERVES the metric relationship between axes. Fitted per channel, base position
    #   gets std (0.500, 0.111, 0.145) -- x is 4.5x y -- so a metre forward and a metre
    #   sideways arrive as very different numbers and the model's error is implicitly
    #   reweighted by how far the robot usually travels along each axis. One scale keeps a
    #   metre a metre, the same argument as the exteroception's metric mode.
    # * it makes ZERO mean zero. Standardizing subtracts a mean (0.354 m on base_pos x, since
    #   the robot advances over the horizon), so "no displacement" stops mapping to 0. For a
    #   window whose node 0 is the robot's own pose, that is a worse origin.
    #
    # ``base_ori`` is deliberately absent: the rot6d block is already exempt from
    # standardization (two orthonormal columns, O(1) by construction).
    # Defaults are MEASURED on clips/g1_mb_v5_body.h5 via the depth_v5 recording, chosen so each
    # block's pooled std lands near 1: base_pos 0.4 -> 0.87, base_lin_vel 0.5 -> 1.05,
    # base_ang_vel 0.4 -> 1.01. ``0`` (or ``None``) fits that block per channel instead.
    #
    # ``joint_pos`` / ``joint_vel`` / ``ee_pos`` deliberately stay FITTED: unlike x/y/z there is
    # no cross-channel metric to preserve between joints, and their fitted std spans 0.067 to
    # 0.504, so one shared scale would leave the tightest joints at ~0.17.
    state_scale_base_pos: float | None = 0.4
    state_scale_base_lin_vel: float | None = 0.5
    state_scale_base_ang_vel: float | None = 0.4
    state_scale_joint_pos: float | None = None
    state_scale_joint_vel: float | None = None
    state_scale_ee_pos: float | None = None
    # DECLARED per-channel scale for the command tokens (``[vx, vy, wz]``), or ``None`` to fit.
    #
    # Commands are a DESIGNED range rather than an observed distribution -- the clip generator
    # was told a maximum -- so dividing by that maximum puts them in [-1, 1] by construction
    # and keeps a zero command at zero, which standardizing does not (the fitted vx mean is
    # 1.25 m/s, so "stand still" would arrive as -1.29).
    # Default is the clip generator's own limits, read off clips/g1_mb_v5_body.h5: the library's
    # extremes are vx 3.4989, vy 0.9998, wz 0.7999, so these bound it exactly and 0.000% of
    # commands fall outside. ``None`` fits per channel instead.
    cmd_scale: list[float] | None = field(default_factory=lambda: [3.5, 1.0, 0.8])
    # Relative weight on the flow loss for root DOFs the anchor mask PINS to the terrain --
    # the axes on which the reference stays where the terrain put it rather than following the
    # robot. 1.0 = off (every channel equal, the historical behaviour).
    #
    # Which axes those are is per NODE, from the recorded clip index joined back to the
    # library: a flat node pins height only, a stair/box node pins downrange, height and yaw,
    # a distractor node pins everything. Those are exactly the quantities the generator gets
    # wrong -- it slides its predicted stair segment along with the robot instead of leaving it
    # on the terrain -- so this puts the gradient where the failure is.
    #
    # The weights are RENORMALIZED to mean 1 over each batch, so the loss magnitude (and hence
    # the effective learning rate) matches an unweighted run and a with/without A/B differs in
    # one thing only.
    # 20.0 is what the working generator trained with, NOT a validated optimum: the A/B that
    # would justify it was confounded (the weight=1 arm sat at a real stationary point --
    # predict-the-mean -- while the weight=20 arm escaped, so the comparison says nothing
    # about the weight). Recorded because it is the shipped architecture; re-run the
    # baseline before treating it as tuned.
    pinned_axis_loss_weight: float = 20.0
    # Compute the per-node anchor mask so validation can report ``flow_mse`` SPLIT by whether
    # the terrain pins that coordinate. On by default and independent of the loss weight,
    # because the split is only useful if BOTH arms of a with/without comparison report it --
    # gating it on the weight would leave the control arm blind to the very quantity being
    # compared. Costs one clip-library join at dataset build; degrades to off with a warning
    # when the library is unavailable, unless the loss weight needs it (then it raises).
    pinned_axis_metrics: bool = True
    # Per-step command width (0 disables command conditioning).
    cmd_dim: int = 0

    # -- model (single transformer; history/height/cmd are separate conditioning tokens) --
    # The 27M "XL" transformer the MVP uses (see CLAUDE.md's MVP section). Capacity kept
    # paying on this task -- 2.74M -> 9.5M -> 27M each improved closed-loop survival -- so
    # this is the size the working pair was built at, not a starting guess. Ablate later; for
    # now the code holds it so no run has to remember four numbers.
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.0
    # Probability that a training sample's history (phase) tokens are replaced by the
    # model's learned null embedding -- classifier-free-guidance / phase-agnostic nominal.
    null_cond_prob: float = 0.2
    # Which way the dependency between the conditioning and the denoised window runs.
    #   "prefix" -- the conditioning (history + extero + cmd) attends only to ITSELF, and the
    #               future tokens attend to everything including themselves. Also keeps the
    #               flow-time embedding off the conditioning entirely. DEFAULT.
    #   "full"   -- every token attends to every other and the flow-time embedding lands on
    #               all of them. What every checkpoint before this flag was trained under.
    #
    # Why "prefix" is the right shape: under full attention the conditioning attends to tokens
    # carrying noise at level tau, so its own representation is a function of the noise level
    # -- when it should be a fixed statement about the robot's past, what it can see and what
    # it was asked to do. The dependency should run one way.
    #
    # The consequence that pays for itself: a representation that cannot depend on the future
    # or on tau is computable ONCE per re-plan and reusable by every Euler step, which is how
    # ``flow.sample_trajectory`` and the exported graph now run it (~1.4x on the CNN model,
    # whose conditioning is the larger share of the sequence).
    #
    # NOTE the asymmetry with ``FlowMatchingTransformer``'s constructor, which defaults to
    # "full": a bare model built in a test keeps the legacy path, while a trained config gets
    # the current architecture. Same split as ``extero_encoder``.
    #
    # WARNING, because nothing else can catch it: the two modes have the SAME PARAMETERS, so
    # a mislabelled checkpoint loads without complaint and runs under attention it was never
    # trained with. ``load_state_dict`` caught a wrong ``extero_encoder`` because the weights
    # differed; here it cannot. ``from_dict``'s default below is the only guard.
    attention_mode: str = "prefix"

    # -- observation noise (train-time augmentation on the clean history tokens) --
    # Mirrors the RL teacher/tracker obs corruption (Unoise per term): the inputs the
    # generator conditions on at deployment come from noisy sensors, so the first
    # ``num_history_states`` (conditioning) state tokens are perturbed during training.
    # Noise is applied ONLY to those conditioning tokens, never to the prediction
    # target; it is resampled every access and disabled for val/eval/deployment.
    # ``obs_noise`` is the master flag; ``noise_scale`` scales every amplitude at once.
    # Amplitudes are physical, uniform half-widths (the sample is U(-amp, amp)).
    # History-token noise. The SHAPE comes from
    # ``flow_matching.window.REFERENCE_WINDOW_NOISE`` -- the same table the policy's
    # reference observation is corrupted with -- because both stand in for the same thing:
    # the generator's own prediction error. At deploy the history IS this model's past
    # output, so that is by construction the error corrupting it. The per-block fields that
    # used to live here were calibrated as SENSOR noise (back when the history was measured
    # robot state) and had drifted up to 10x from the observation side; they are gone.
    # ``noise_scale`` is the one knob left, and it scales the whole table.
    obs_noise: bool = True
    noise_scale: float = 1.0
    # Per-element exteroception noise. The token is a real sensor at deploy, so noise it
    # in training too. Applied independently per element, train split only.
    # NO train-time exteroception noise. There used to be ``noise_height_scan`` (0.1 m) and
    # ``noise_depth`` (0.05 normalised = 0.1 m), added per element on top of whatever the
    # recording already carried; both are gone, along with the two gates that tried to keep
    # them from double-counting (``extero_corrupted``, ``extero_noise_free``).
    #
    # The sensor model belongs to the RL env and nowhere else. The cameras already carry
    # structured DR -- per-episode plane tilt and blur sigma, extrinsics/intrinsics jitter,
    # per-step additive noise and 2.5% dropout -- and that corruption is baked into the
    # recording, so a scalar uniform term on top was over-noising the one signal the generator
    # uses to place a reference on terrain. It was also incoherent for the privileged scan,
    # where 0.1 m is four cells of a 0.025 m grid: the vertical channel quantised coarser than
    # the step edge being localised, in a run whose entire point was an exact ground truth.
    #
    # This is deliberately NOT the history/reference-window noise (``noise_scale`` and
    # ``window.REFERENCE_WINDOW_NOISE``), which models the generator's own prediction error
    # feeding back into its history at deploy -- exposure bias, not a sensor.

    # -- flow --
    num_sampling_steps: int = 8

    # -- consistency / forward-kinematics regularization (finite-difference + FK) --
    # ``dt`` is the control step period (seconds), resolved from the recording meta at
    # train time. ``vel_consistency_weight`` ties the predicted ROOT velocity to the
    # finite difference of the predicted root pose; ``joint_vel_consistency_weight`` does
    # the predicted velocity to that of the data velocity. ``fk_pose_weight`` ties each
    # predicted end-effector POSITION (base-relative) to the forward kinematics of the
    # predicted joints, using ``fk_model_path`` (a MuJoCo MJCF) -- this is what stops the
    # joints and the ee positions describing different postures.
    # ``consistency_tau_min`` gates the FK + body-pose terms to samples with flow-time
    # ``tau >= consistency_tau_min`` (where the predicted endpoint is a faithful estimate
    # of the clean target); velocity/joint-velocity stay ungated.
    #
    # The frame terms cover the ROOT only and the FK residual is position-only, because
    # the extra frames are position-only in the state vector (see ``window.py``). The
    # ee-orientation FK residual and the ``J(q)·q̇`` term (the old ``fk_vel_weight``) went
    # with those channels.
    #
    # ``body_pose_*_weight`` are the BODY-LINK POSE terms: forward kinematics of the
    # predicted joints against forward kinematics of the TARGET joints, over the tracked
    # body frames. Accuracy, not self-consistency -- so unlike ``fk_pose_weight`` they need
    # no ``ee_pos`` block, and they add no information: they are a change of METRIC, into
    # the one the CLF actually grades. The teacher runs ``tracking_mode="body"``, where eta
    # is the 13 ``<body>@root:*`` frames plus the pelvis and the world ee frames, with
    # ``joint:*`` DROPPED -- while the flow MSE weights every joint equally in normalized
    # angle space. See ``flow.body_pose_terms``.
    #
    #   * ``body_pose_root_weight`` -- poses in the reference's own pelvis frame. Pure
    #     posture; independent of ``base_pose``, so it cannot be satisfied by moving the
    #     base. Keep it on whenever the world term is, as its anti-cancellation partner.
    #   * ``body_pose_world_weight`` -- the same poses composed with the predicted
    #     ``base_pose``. Numerically the WORLD-frame error (prediction and target share one
    #     anchor, and a rigid transform preserves distances), so it is what cares about
    #     landing a foot on a given stair tread, and it re-weights base position up by the
    #     body multiplicity in physical metres -- outside the pooled normalizer that
    #     currently squashes ``base_pos``.
    #   * ``body_pose_ori_scale`` -- metres per radian, converting the orientation half of
    #     both terms so the two halves of one geometric quantity share a weight. Read it as
    #     a LEVER ARM: the distance over which a link's angular error becomes a positional
    #     one, so a typical link half-length (~0.25 m) makes the term roughly isotropic.
    #
    #     Not a free parameter -- measured. On the depth_v5 recording at init, the halves
    #     are pos 0.0004 / ori 0.0080 (root) and pos 0.0135 / ori 0.1114 (world), i.e. ~2 cm
    #     of position against ~0.09 rad of rotation. At 1.0 m/rad orientation would be ~90%
    #     of both terms, which is not what a loss aimed at foot-placement precision should
    #     be spending itself on. 0.25 puts the two halves within ~2x. The ``*_pos`` /
    #     ``*_ori`` halves are logged separately so this stays tunable on evidence.
    #
    # Both default OFF: this is an ablation switch, one flag each. Nothing else about the
    # generator changes -- same conditioning tokens, same output width, same normalizer, same
    # exported metadata -- so this is FM-only: no teacher retrain, no re-record, and a stale
    # artifact cannot arise from it.
    #
    # PICKING THE WEIGHTS. These are NOT dimensionless: the flow MSE is in NORMALIZED units
    # while these terms are in physical m^2, so the weight carries the unit conversion. This
    # is the OPPOSITE direction from vel_consistency_weight=0.01, whose term logs at ~144 --
    # do not copy that magnitude across.
    #
    # Measured, XL (27.3M) over 100 epochs on the 200k depth_v5 recording, ori_scale=0.25,
    # weights root=400 / world=20:
    #
    #     epoch   flow_mse   body_pose_root   body_pose_world   combined share
    #         0     0.2742           0.0001            0.0009            ~21%
    #        40     0.1011           0.0000            0.0002             ~5%
    #        99     0.0829           0.0000            0.0001            ~2.4%
    #
    # THE SHARE DECAYS, and that is inherent to physical units: these are squared errors that
    # shrink as the model improves, so a weight buys a share at a point in TIME, not a budget.
    # By epoch 99 the root term contributes ~0% and the world term ~2.4%. Read that as the
    # terms being largely SATISFIED rather than ignored -- root position is under ~2 mm RMS by
    # then -- but do not expect either to still be shaping the solution at convergence.
    #
    # So this is closer to an early-training regularizer than a persistent objective. To make
    # it bite late, the weights have to be far larger than the ones above, or the terms have
    # to be normalized by a fixed reference scale -- which would cost the interpretability
    # that let ori_scale be set by measurement rather than guessed.
    #
    # Every term needs the blocks it ties together to BE in ``reference_blocks``:
    # velocity needs base_pose+base_vel, joint-velocity
    # joint_pos+joint_vel, FK joint_pos+ee_pos, body-pose joint_pos (+base_pose for the
    # world half). A weight whose blocks are absent is zeroed at dataset build with a
    # printed line -- the term has nothing to relate, and the default selection
    # (base_pose, base_vel, joint_pos) already switches two of them off.
    dt: float = 0.02
    # 0 by DEFAULT, deliberately a change from the 0.01 every run through 2026-08-10 used:
    # the current experiment is being run without it. Note what switching it off gives up --
    # with the shipped ['base_pose', 'base_vel', 'joint_pos'] selection this is the ONLY term
    # coupling the six base_vel channels to the base_pose they are supposed to differentiate
    # (joint_vel_consistency needs a joint_vel block, which is not packed), and the flow MSE
    # treats all 44 channels independently. At 0 the predicted velocities may drift out of
    # agreement with the predicted pose, and BOTH are served to the policy through
    # reference_window. It is still reported as a val metric so that drift stays visible.
    vel_consistency_weight: float = 0.0
    # Both OFF in the shipped configuration. They are auxiliary consistency terms, and the
    # generator that works was trained without either; leaving them on by default meant every
    # run had to remember to zero them (nothing auto-corrects it, and fk_pose_weight silently
    # does nothing when the dataset has no ee frames, which is its own trap).
    joint_vel_consistency_weight: float = 0.0
    fk_pose_weight: float = 0.0
    body_pose_root_weight: float = 0.0
    body_pose_world_weight: float = 0.0
    body_pose_ori_scale: float = 0.25
    consistency_tau_min: float = 0.75
    # ABSOLUTE, derived from the package root -- it used to be a bare CWD-relative string,
    # which CLAUDE.md's path rule forbids and which silently resolved to nothing whenever a
    # training run was launched from anywhere but the repo root.
    fk_model_path: str = DEFAULT_MJCF
    # Body frames the pose terms grade, as MJCF body names on ``fk_model_path``. ``None``
    # resolves to ``motion_gen.joint_names.G1_BODY_TRACK_FRAMES`` -- the same 13 the clip
    # library carries ``@root`` channels for and the body-mode CLF grades, so the loss and
    # the reward see one frame set. Only read when a body-pose weight is nonzero.
    body_fk_frames: list[str] | None = None

    # NOTE: the terrain-penetration losses (both the teacher-forced ``x1_hat`` term and the
    # sampled-rollout term) were DELETED. Both assumed a single flat ``ground_z`` floor,
    # which is wrong on the stairs/boxes the clip library actually trains on: the floor
    # auto-calibrated near the LOWEST step, leaving the term mostly inert and occasionally
    # penalising correct predictions. Nothing now penalises predicted feet crossing the
    # ground -- re-introduce this only with a real SDF or a height-scan-derived LOCAL floor
    # (the height scan is already in the conditioning), not a scalar.

    # -- training --
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    grad_clip: float = 1.0
    # Optimizer steps over which the LR ramps LINEARLY from lr/warmup_steps to ``lr``
    # (0 = off, the historical behaviour). Recorded in the config rather than left a bare
    # CLI flag because it changes the optimization and two runs are only comparable if both
    # state it.
    #
    # What it buys: every run so far spends its first few hundred steps at the
    # predict-the-marginal-mean plateau (flow MSE ~1.8-2.4 -- the model outputs ~0, which IS
    # a stationary point once the normalizer has centred the targets, since E[target_v] = 0).
    # Escaping it needs the first useful gradient direction to survive; at full LR from step
    # 0, Adam's second moment is still warming from ~nothing, so early updates are close to
    # +-lr per COORDINATE regardless of gradient size, and a few hundred of those can drive
    # the trunk into a configuration whose expected gradient really is zero -- after which
    # ``m/sqrt(v)`` collapses to ~1e-7 and the run is frozen for good, at the plateau value.
    # That is not hypothetical: the 2026-08-12 pinned-axis A/B lost its weight=1.0 arm this
    # way while its weight=20 arm, identical in every other respect (same seed, same data
    # sha, loss differing only by the reweighting), escaped at step ~300.
    # 500: added with the grad-norm/Adam-SNR instrumentation after a run sat at a stationary
    # point for hundreds of steps, and used by the generator that works.
    lr_warmup_steps: int = 500
    seed: int = 0

    @property
    def num_cmd_tokens(self) -> int:
        """Number of strided command tokens (0 when not command-conditioned)."""
        if self.cmd_dim <= 0:
            return 0
        return math.ceil(self.future_len / self.cmd_stride)

    def resolve_extero_channels(self) -> int:
        """Settle :attr:`extero_channels`, expanding ``None`` from the source.

        A depth pixel is a ray, and the channels state where that ray landed; a height scan's
        rays already ARE ground points, so widening one is meaningless and the dataset refuses
        it. One default therefore cannot be a number -- it has to resolve per source, the same
        way :meth:`resolve_extero_encoder` and :meth:`resolve_extero_clip` do.

        Returns:
            5 for depth (``[d, x, y, z, v]``), 1 for a height scan, or the explicit value.

        Raises:
            ValueError: On an unknown ``extero_source``, or a channel count the depth path
                does not emit.
        """
        self.check_extero_source()
        if self.extero_channels is None:
            return 5 if self.extero_source == "depth" else 1
        channels = int(self.extero_channels)
        if channels not in (1, 4, 5):
            raise ValueError(
                f"extero_channels={channels} is not supported; the depth path emits 1 (depth "
                "alone), 4 (depth + xyz) or 5 (plus validity)."
            )
        if channels > 1 and self.extero_source != "depth":
            raise ValueError(
                f"extero_channels={channels} but extero_source={self.extero_source!r}. The "
                "extra planes are a pixel's xyz from the pinhole model; a height scan's rays "
                "already are ground points."
            )
        return channels

    def extero_channel_groups(self) -> tuple[tuple[str, ...], ...]:
        """Channel names grouped as ``extero_norm_mode="per_channel"`` standardizes them.

        The grouping is the whole content of that mode, so it lives here rather than inline in
        the dataset: ``xyz`` share ONE statistic (a metre must stay a metre in every direction,
        and per-axis fits would squeeze downrange hardest), while ``d`` gets its own and ``v``
        is left alone as an indicator.

        Returns:
            One tuple of channel names per group, in row order. ``v`` is returned in its own
            group; the caller is what knows it must not be standardized.
        """
        names = [str(n) for n in self.extero_channel_names]
        groups: list[tuple[str, ...]] = []
        xyz = tuple(n for n in ("x", "y", "z") if n in names)
        for name in names:
            if name in xyz:
                if xyz not in groups:
                    groups.append(xyz)
            else:
                groups.append((name,))
        return tuple(groups)

    def check_extero_source(self) -> None:
        """Raise if :attr:`extero_source` is not a recognised source.

        Raises:
            ValueError: On an unknown source name.
        """
        if self.extero_source not in EXTERO_SOURCES:
            raise ValueError(
                f"extero_source={self.extero_source!r} is not one of {list(EXTERO_SOURCES)}."
            )

    def check_attention_mode(self) -> None:
        """Raise if :attr:`attention_mode` is not a recognised mode.

        Raises:
            ValueError: On an unknown mode name.
        """
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"attention_mode={self.attention_mode!r} is not one of "
                f"{list(ATTENTION_MODES)}."
            )

    def resolve_extero_encoder(self) -> str:
        """Settle :attr:`extero_encoder`, expanding ``"auto"`` from the source.

        ``"auto"`` picks the convolutional grid for depth and the flat MLP for the scan.
        That default is deliberately NOT source-symmetric even though a scan can now be
        convolved: flipping the scan's ``"auto"`` to ``"cnn"`` would silently re-architect
        every existing height-scan run, whose recordings carry no grid geometry at all.
        Convolving a scan is opt-in, via ``--extero_encoder cnn`` on a recording whose
        ``height_scan_spec`` states the fold.

        What replaced the old "cnn implies depth" rule is an invariant that does not mention
        the source: **cnn implies a per-sensor ``(H, W)`` that closes against the recording's
        own spec**, enforced by :meth:`check_extero_encoder` here, by the dataset's
        grid-closes check, and again at load time.

        Returns:
            ``"cnn"`` or ``"mlp"``.

        Raises:
            ValueError: On an unknown encoder name.
        """
        self.check_extero_source()
        encoder = self.extero_encoder
        if encoder == "auto":
            return "cnn" if self.extero_source == "depth" else "mlp"
        if encoder not in EXTERO_ENCODERS:
            raise ValueError(
                f"extero_encoder={encoder!r} is not one of {list(EXTERO_ENCODERS)} (or 'auto')."
            )
        return encoder

    def resolve_extero_clip(self) -> float:
        """The symmetric clip band ``sanitize_extero`` should apply, in the source's units.

        :attr:`extero_clip` when set, else the per-source default. Kept as an explicit
        resolver rather than defaulted in the field so an absent value in an old checkpoint
        reproduces that checkpoint's behaviour exactly.

        Returns:
            The band magnitude.
        """
        self.check_extero_source()
        if self.extero_clip is None:
            return float(EXTERO_CLIP[self.extero_source])
        band = float(self.extero_clip)
        if not band > 0.0:
            raise ValueError(f"extero_clip must be positive, got {band}.")
        return band

    def check_extero_norm_mode(self) -> None:
        """Raise if :attr:`extero_norm_mode` is not one of the three supported modes."""
        if self.extero_norm_mode not in EXTERO_NORM_MODES:
            raise ValueError(
                f"extero_norm_mode={self.extero_norm_mode!r} is not one of "
                f"{list(EXTERO_NORM_MODES)}."
            )
        if self.extero_norm_mode == "metric" and not float(self.extero_norm_scale) > 0.0:
            raise ValueError(
                f"extero_norm_mode='metric' needs a positive extero_norm_scale, got "
                f"{self.extero_norm_scale}."
            )

    def check_extero_encoder(self) -> None:
        """Raise if the exteroception encoder settings are not internally coherent.

        Checks the resolved encoder name, and -- for the CNN -- that the derived per-sensor
        image shapes are present, one per sensor, and each unfold to
        :attr:`extero_sensor_dim`. Cheap, and it turns a silent mis-fold into a message: a
        transposed or absent shape would otherwise reach the convolution as a scrambled
        neighbourhood with every tensor the right size.

        Raises:
            ValueError: On an unknown encoder, or on CNN image shapes that are missing,
                miscounted, or inconsistent with the sensor width.
        """
        if self.resolve_extero_encoder() != "cnn":
            return
        if not self.extero_image_hw:
            raise ValueError(
                "extero_encoder='cnn' needs extero_image_hw, which the dataset derives from the "
                "recording's sensor contract -- depth_camera_spec for depth, height_scan_spec "
                "for the scan. An empty value means the dataset was not built, or the recording "
                "carries no contract for its exteroception (re-record: a bare ray count cannot "
                "say what shape the grid was, and a wrong fold is scrambled input at the right "
                "size)."
            )
        if self.extero_cnn_strides is not None and (
                len(self.extero_cnn_strides) != len(self.extero_cnn_channels)):
            raise ValueError(
                f"extero_cnn_strides has {len(self.extero_cnn_strides)} entries but "
                f"extero_cnn_channels names {len(self.extero_cnn_channels)} convolutions; "
                "one stride per layer."
            )
        if len(self.extero_image_hw) != self.extero_sensor_count:
            raise ValueError(
                f"extero_image_hw lists {len(self.extero_image_hw)} shape(s) but there are "
                f"{self.extero_sensor_count} sensor(s)."
            )
        channels = self.resolve_extero_channels()
        if len(self.extero_channel_names) != channels:
            raise ValueError(
                f"extero_channel_names {list(self.extero_channel_names)} has "
                f"{len(self.extero_channel_names)} entries but extero_channels={channels}."
            )
        if self.extero_sensor_dim % channels:
            raise ValueError(
                f"extero_sensor_dim={self.extero_sensor_dim} is not divisible by "
                f"extero_channels={channels}; the row is channels * H * W."
            )
        pixels = self.extero_sensor_dim // channels
        for hw in self.extero_image_hw:
            # ``pixels``, not ``extero_sensor_dim``: with C > 1 the row holds C planes of the
            # same image, so comparing the whole width against H*W would reject the very
            # thing it is meant to describe.
            if len(hw) != 2 or int(hw[0]) * int(hw[1]) != pixels:
                raise ValueError(
                    f"extero_image_hw entry {list(hw)} does not unfold to {pixels} pixels "
                    f"(extero_sensor_dim={self.extero_sensor_dim} / "
                    f"extero_channels={channels})."
                )

    @property
    def tokens_per_sensor(self) -> int:
        """Transformer tokens ONE sensor reading contributes.

        1 for the flat MLP; the convolutional grid's cell count otherwise. Pure arithmetic
        (no module is built), so ``train.py`` can print the sequence length before the model
        exists.

        Returns:
            Tokens per sensor.
        """
        if self.resolve_extero_encoder() != "cnn":
            return 1
        if self.extero_cnn_grid:
            return int(self.extero_cnn_grid[0]) * int(self.extero_cnn_grid[1])
        self.check_extero_encoder()
        grid = conv_grid_shape(
            (int(self.extero_image_hw[0][0]), int(self.extero_image_hw[0][1])),
            len(self.extero_cnn_channels),
            self.extero_cnn_strides,
        )
        return grid[0] * grid[1]

    @property
    def extero_seq_tokens(self) -> int:
        """Exteroception tokens in the transformer SEQUENCE (sensors x tokens per sensor).

        Distinct from :attr:`extero_sensor_count`, which is how many physical readings
        there are -- the two were one number back when a reading was a token, and the
        deploy-side camera checks still read the sensor count.
        """
        return int(self.extero_sensor_count) * self.tokens_per_sensor

    @property
    def sequence_len(self) -> int:
        """Total transformer sequence length: history + exteroception + command + future."""
        return (int(self.num_history_states) + self.extero_seq_tokens
                + int(self.num_cmd_tokens) + int(self.future_len))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy of the config (for checkpointing)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowMatchingConfig":
        """Build a config from a dict, ignoring unknown keys.

        A checkpoint saved BEFORE :attr:`extero_source` existed was necessarily a
        height-scan model -- the depth path did not exist yet -- so an absent key resolves
        to ``"height_scan"`` rather than to the current default. Taking the default here
        would silently reinterpret every pre-selector checkpoint as a depth model the
        moment that default changed, which is exactly what happened when it flipped to
        ``"depth"``. Checkpoints written since the field exists always carry it, so this
        only ever affects genuinely old ones.

        The exteroception fields were also RENAMED (see :data:`_LEGACY_KEYS`), and a
        checkpoint is the artifact worth shimming: re-exporting an ONNX is one command,
        while re-training a generator is a hundred epochs. Old names are mapped only where
        the new one is absent, so a checkpoint carrying both (there should be none) keeps
        the new one.
        """
        names = {f.name for f in fields(cls)}
        renamed = dict(d)
        for old, new in _LEGACY_KEYS.items():
            if old in renamed and new not in renamed:
                renamed[new] = renamed.pop(old)
        kwargs = {k: v for k, v in renamed.items() if k in names}
        kwargs.setdefault("extero_source", "height_scan")
        # Same reasoning as extero_source above, and the same trap: a checkpoint written
        # before the encoder switch existed is necessarily an MLP one, so it must NOT pick up
        # the current "auto" default. Letting it would reinterpret every pre-CNN checkpoint --
        # including the trained baseline any new encoder has to be measured against -- as a
        # convolutional model, and the weights would then fail to load.
        kwargs.setdefault("extero_encoder", "mlp")
        # THE guard for the attention mode, and the only one there can be: the two modes share
        # one parameter set, so unlike ``extero_encoder`` a wrong value here cannot be caught
        # by ``load_state_dict``. A checkpoint written before the flag existed is definitionally
        # full-attention, so an absent key must resolve to "full" and never to the current
        # default -- which is exactly the mistake this file has now recorded three times.
        kwargs.setdefault("attention_mode", "full")
        return cls(**kwargs)
