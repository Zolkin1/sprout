# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching future-state prediction model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


#: Recognised :attr:`FlowMatchingConfig.extero_source` values.  These are also the keys
#: ``record_data.py`` writes, so the dataset looks the array up by this name directly.
EXTERO_SOURCES: tuple[str, ...] = ("height_scan", "depth")

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
    skip_distractors: bool = False

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
    # Width of ONE exteroception token, and how many there are. Both resolved from the
    # recorded array at dataset build time (176x1 for the scan, 780x2 for the cameras).
    height_dim: int = 176
    extero_tokens: int = 1
    # Std floor for the height-scan normalizer. On flat ground the per-ray scan is nearly
    # constant, so the raw std collapses toward the base Normalizer eps (1e-6) and blows up
    # normalization; a larger floor (~1e-2 m) keeps a near-flat scan well-scaled.
    height_norm_eps: float = 1e-2
    # Per-step command width (0 disables command conditioning).
    cmd_dim: int = 0

    # -- model (single transformer; history/height/cmd are separate conditioning tokens) --
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.0
    # Probability that a training sample's history (phase) tokens are replaced by the
    # model's learned null embedding -- classifier-free-guidance / phase-agnostic nominal.
    null_cond_prob: float = 0.2

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
    # ``noise_height_scan`` is metres of ground clearance and matches the RL policy's
    # height_scan Unoise(±0.1). ``noise_depth`` is in the depth obs's NORMALISED units
    # -- ``mdp.depth_image`` already applies ``(d - 0.75) * 0.5``, so 0.05 here is 0.1 m
    # of range, matching the scan. The depth cameras additionally carry their own sensor
    # DR (plane tilt, blur, dropout, extrinsics/intrinsics jitter) from the RL env, which
    # is baked into the recording -- this term is on top of that.
    noise_height_scan: float = 0.1   # m                (RL match)
    noise_depth: float = 0.05        # normalised units (= 0.1 m)

    # -- flow --
    num_sampling_steps: int = 8

    # -- consistency / forward-kinematics regularization (finite-difference + FK) --
    # ``dt`` is the control step period (seconds), resolved from the recording meta at
    # train time. ``vel_consistency_weight`` ties the predicted ROOT velocity to the
    # finite difference of the predicted root pose; ``joint_vel_consistency_weight`` does
    # the same for joints; ``accel_consistency_weight`` matches the finite difference of
    # the predicted velocity to that of the data velocity. ``fk_pose_weight`` ties each
    # predicted end-effector POSITION (base-relative) to the forward kinematics of the
    # predicted joints, using ``fk_model_path`` (a MuJoCo MJCF) -- this is what stops the
    # joints and the ee positions describing different postures.
    # ``consistency_tau_min`` gates the acceleration + FK terms to samples with flow-time
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
    # velocity needs base_pose+base_vel, acceleration base_vel, joint-velocity
    # joint_pos+joint_vel, FK joint_pos+ee_pos, body-pose joint_pos (+base_pose for the
    # world half). A weight whose blocks are absent is zeroed at dataset build with a
    # printed line -- the term has nothing to relate, and the default selection
    # (base_pose, base_vel, joint_pos) already switches two of them off.
    dt: float = 0.02
    vel_consistency_weight: float = 0.01 #0.0
    accel_consistency_weight: float = 0.0
    joint_vel_consistency_weight: float = 0.01 #0.0
    fk_pose_weight: float = 0.01 #0.0
    body_pose_root_weight: float = 0.0
    body_pose_world_weight: float = 0.0
    body_pose_ori_scale: float = 0.25
    consistency_tau_min: float = 0.75
    fk_model_path: str = "transfer/sim/robots/g1/g1_29j.xml"
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
    seed: int = 0

    @property
    def num_cmd_tokens(self) -> int:
        """Number of strided command tokens (0 when not command-conditioned)."""
        if self.cmd_dim <= 0:
            return 0
        return math.ceil(self.future_len / self.cmd_stride)

    @property
    def noise_extero(self) -> float:
        """Per-element train-time noise for the selected exteroception source.

        Returns:
            ``noise_height_scan`` (metres) or ``noise_depth`` (normalised units).
        """
        self.check_extero_source()
        return self.noise_height_scan if self.extero_source == "height_scan" else self.noise_depth

    def check_extero_source(self) -> None:
        """Raise if :attr:`extero_source` is not a recognised source.

        Raises:
            ValueError: On an unknown source name.
        """
        if self.extero_source not in EXTERO_SOURCES:
            raise ValueError(
                f"extero_source={self.extero_source!r} is not one of {list(EXTERO_SOURCES)}."
            )

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
        """
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        kwargs.setdefault("extero_source", "height_scan")
        return cls(**kwargs)
