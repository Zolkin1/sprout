# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching future-state prediction model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


#: Recognised :attr:`FlowMatchingConfig.extero_source` values.  These are also the keys
#: ``record_data.py`` writes, so the dataset looks the array up by this name directly.
EXTERO_SOURCES: tuple[str, ...] = ("height_scan", "depth")

#: Recognised :attr:`FlowMatchingConfig.history_source` values.
HISTORY_SOURCES: tuple[str, ...] = ("measured", "reference_partial")


@dataclass
class FlowMatchingConfig:
    """Hyperparameters for the flow-matching model, data pipeline, and training.

    ``state_dim`` and ``cmd_dim`` are resolved from the dataset at build time (they
    depend on the joint/extra-frame count and the command width), so the defaults
    here are placeholders.

    Architecture (single transformer, robot-state conditioned): tokens are assembled
    as ``[history(N) | height(1) | cmd(Ncmd) | future(F)]``. The ``num_history_states``
    real-robot-state history tokens (whose job is to supply *phase*), the height-scan
    token (a ``height_dim``-ray ground-clearance vector embedded into ONE token), and
    the strided future-command tokens are SEPARATE conditioning tokens
    (never noised, not part of the denoised window). The model denoises the ``F``
    future state tokens -- ALL future, no past echo, no inpainting / known-mask. During
    training the history is replaced by a learned null embedding with probability
    ``null_cond_prob`` (classifier-free-guidance / phase-agnostic nominal).
    """

    # -- data / windowing --
    # Future window length F: the full denoised state sequence (node 0 = the current /
    # anchor step, ego-anchored onto the robot so its base offset ≈ 0; nodes 1..F-1 are
    # the future). The relative-frame anchor is the robot base at the most recent history
    # step.
    #
    # F is set by what the POLICY asks of one prediction. ``FmTrackingCommand`` re-plans
    # every ``fm_update_every`` steps and walks a cursor one node per step in between, so a
    # prediction must still cover the teacher's whole look-ahead window on the LAST step
    # before the next re-plan:
    #
    #     F_min = (fm_update_every - 1) + (REFERENCE_NUM_STEPS - 1) * REFERENCE_STRIDE + 1
    #           = (12 - 1)             + (11 - 1) * 5                                 + 1
    #           = 62                                              # 1.24 s at dt = 0.02
    #
    # (``fm_update_every`` = the tracker's re-plan period and the recording's
    # ``fm_snap_every``; ``REFERENCE_NUM_STEPS`` / ``REFERENCE_STRIDE`` are the teacher's
    # reference-window knobs in ``g1_teacher_env_cfg.py``.)  65 rounds that up to a multiple
    # of cmd_stride (=5) -> 13 evenly-spaced command tokens, with 3 steps of slack.
    # ``FmTrackingCommand`` re-derives F_min from the offsets it is actually asked for and
    # raises if the checkpoint is short, so this never degrades into a silent clamp.
    #
    # Raising F grows per-window memory and the recorded ``ref_future`` array; bump
    # --stride / --num_workers if training OOMs.  It also requires RE-RECORDING
    # (record_data.py cuts windows at this length) and retraining.
    future_len: int = 65
    # N history (phase) conditioning tokens. Default 2: enough to read gait phase (the FM
    # predicts nominal phase-dependent motion, not recovery).
    num_history_states: int = 2

    # What the history tokens are made of.
    #
    #   "measured"          -- the robot's ACTUAL state at each of the N steps. The default,
    #                          and the paradigm that fixed the autoregressive z-drift: the
    #                          generator is re-grounded in reality every re-plan.
    #   "reference_partial" -- PARTIALLY AUTOREGRESSIVE. The root pose/twist still comes from
    #                          the measured robot, but the joints and the extra-frame
    #                          positions come from the reference that was actually APPLIED at
    #                          those steps -- i.e. the generator's own recent output, closing
    #                          the loop on the posture channels while the root stays honest.
    #
    # The frame is unchanged either way: everything is expressed relative to the measured
    # robot base at the newest history row, which is also the anchor the prediction is lifted
    # through, so node 0 still lands on the robot.
    #
    # Both are train-time choices over the SAME recording -- ``record_data.py`` stores the
    # applied-reference history ring unconditionally -- so this is an A/B with no re-record.
    # The runtime reads it off the CHECKPOINT, not the task config, so a partially
    # autoregressive model cannot silently be fed a measured history.
    #
    # Caveat worth knowing before reading a bad result as a bug: with the root measured and
    # the ee positions taken from the reference, the ee block sits relative to an anchor the
    # reference has drifted from. Over the <= fm_update_every stale steps that drift is small,
    # but it is not zero.
    history_source: str = "measured"
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

    # Dedicated recorded command array (e.g. "target_vel") giving the per-step
    # commanded velocity over the window. It is sampled every ``cmd_stride`` steps
    # (indices 0, cmd_stride, ...) and each sample becomes its own conditioning token
    # (a separate command-embedding MLP). ``None`` means no command conditioning.
    cmd_array: str | None = None
    cmd_stride: int = 5

    # -- dimensions (resolved from the dataset at build time) --
    # Per-step state width of every token: 2*nj + 15*(1 + num_extra_frames). History
    # and predicted states share this single representation (one shared normalizer).
    state_dim: int = 73        # 2*nj + 15 = 2*29 + 15 (nj=29); resolved from the dataset at build time
    # Per-step history-token width (= state_dim; resolved at build time).
    history_dim: int = 73
    # Which recorded signal feeds the exteroception conditioning tokens.
    #   "height_scan" -- the pelvis ray-cast grid (176 rays) as ONE token. Default; the
    #                    validated path.
    #   "depth"       -- the two torso depth cameras, ONE token each (T=2), from
    #                    ``G1-teacher-clf-terrain-depth*`` recordings. This is the
    #                    deployable signal: the hardware has cameras, not a heightmap.
    # ``record_data.py`` stores BOTH arrays when the task has the cameras, so one
    # recording session can train either FM and this is a train-time choice.
    extero_source: str = "height_scan"
    # Width of ONE exteroception token, and how many there are. Both resolved from the
    # recorded array at dataset build time (176x1 for the scan, 780x2 for the cameras).
    height_dim: int = 176
    extero_tokens: int = 1
    # World floor height (flat ground -> 0.0). VESTIGIAL: it fed no conditioning token once
    # the height scan replaced the root-z scalar, and its last real consumer (the
    # terrain-penetration loss) is now deleted. It is still copied into ``FMContext`` and the
    # exported ONNX ``fm_meta``, where nothing reads it back.
    # TODO: drop the field together with the ``ground_z`` key in ``export_fm_onnx.py`` /
    # ``transfer/deploy/fm_session.py`` and the unused ``transforms.height_above_ground``.
    ground_z: float = 0.0
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
    obs_noise: bool = True
    noise_scale: float = 1.0
    noise_joint_pos: float = 0.01   # rad        (RL match)
    noise_joint_vel: float = 0.5    # rad/s      (RL match)
    noise_pos: float = 0.02         # m          base + extra-frame positions (estimator)
    noise_lin_vel: float = 0.1      # m/s        base + extra-frame lin vel (estimator)
    noise_ori: float = 0.05         # ori6d cols (RL: projected_gravity)
    noise_ang_vel: float = 0.2      # rad/s      (RL: gyro)
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
    ori6d_aux_weight: float = 0.0

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
    # the extra frames are position-only in the state vector (see ``transforms.py``). The
    # ee-orientation FK residual and the ``J(q)·q̇`` term (the old ``fk_vel_weight``) went
    # with those channels.
    dt: float = 0.02
    vel_consistency_weight: float = 0.01 #0.0
    accel_consistency_weight: float = 0.0
    joint_vel_consistency_weight: float = 0.01 #0.0
    fk_pose_weight: float = 0.01 #0.0
    consistency_tau_min: float = 0.75
    fk_model_path: str = "transfer/sim/robots/g1/g1_29j.xml"

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

    def check_history_source(self) -> None:
        """Raise if :attr:`history_source` is not a recognised source.

        Raises:
            ValueError: On an unknown source name.
        """
        if self.history_source not in HISTORY_SOURCES:
            raise ValueError(
                f"history_source={self.history_source!r} is not one of {list(HISTORY_SOURCES)}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy of the config (for checkpointing)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowMatchingConfig":
        """Build a config from a dict, ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
