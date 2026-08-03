# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching future-state prediction model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


#: Recognised :attr:`FlowMatchingConfig.extero_source` values.  These are also the keys
#: ``record_data.py`` writes, so the dataset looks the array up by this name directly.
EXTERO_SOURCES: tuple[str, ...] = ("height_scan", "depth")



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
    # F is set by what the POLICY asks of one prediction, and now exactly:
    #
    #     F = (REFERENCE_NUM_STEPS - 1) * REFERENCE_STRIDE + 1
    #       = (11 - 1)                 * 5                 + 1
    #       = 51                                              # 1.0 s at dt = 0.02
    #
    # (``REFERENCE_NUM_STEPS`` / ``REFERENCE_STRIDE`` are the teacher's reference-window
    # knobs in ``g1_teacher_env_cfg.py``.)  There is no ``fm_update_every - 1`` slack term
    # any more because ``fm_update_every`` is PINNED TO 1: the prediction is expressed
    # relative to the robot at the re-plan, so serving it a step later would need the
    # egomotion since then to re-anchor -- odometry the deploy path deliberately no longer
    # carries. ``FmTrackingCommand`` raises rather than let the anchor go stale.
    #
    # cmd_stride (=5) divides 50, so the 11 command tokens land exactly on the window
    # offsets [0, 5, ..., 50].
    #
    # Changing F requires RE-RECORDING (record_data.py cuts windows at this length) and
    # retraining.
    future_len: int = 51
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

    # Dedicated recorded command array (e.g. "target_vel") giving the per-step
    # commanded velocity over the window. It is sampled every ``cmd_stride`` steps
    # (indices 0, cmd_stride, ...) and each sample becomes its own conditioning token
    # (a separate command-embedding MLP). ``None`` means no command conditioning.
    cmd_array: str | None = None
    cmd_stride: int = 5

    # TODO(fm-blocks): make this half block-aware. The teacher now packs only a SUBSET of
    # the window (g1 REFERENCE_WINDOW_BLOCKS = base_pose, base_vel, joint_pos -> 41/node,
    # 451 total), but the dataset still resolves state_dim from window_dim(nj, K) with no
    # selection -> 82. Nothing couples the two, so the suite stays green and a newly trained
    # FM is silently the wrong width, failing only at play time.
    #
    #   * carry the selection here and pass it to world_to_window in dataset.py
    #   * flow.py: fk_pose_weight and joint_vel_consistency must GO -- both index blocks
    #     that no longer exist
    #   * export_fm_onnx / fm_session: make the window_blocks guard selection-aware (that
    #     guard is precisely what would have caught this)
    #   * fm_tracking_cmd._fm_current_outputs writes ee frames the FM no longer predicts
    #   * add the missing load-time check: the FM's selection must equal the teacher's,
    #     same shape of guard as base_pose_convention
    # -- dimensions (resolved from the dataset at build time) --
    # Per-node width of every token: the reference-window layout,
    # 12 + 2*nj + 3*num_extra_frames (see ``flow_matching.window.window_dim``). History
    # and predicted nodes are the SAME representation -- the history is just nodes this
    # model emitted earlier -- so they share one normalizer.
    state_dim: int = 82        # 12 + 2*29 + 3*4 (nj=29, K=4); resolved at build time
    # Per-node history-token width (= state_dim; resolved at build time).
    history_dim: int = 82
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
