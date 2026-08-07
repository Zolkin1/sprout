# Copyright 2026 Zachary Olkin. All rights reserved.

"""Torch-free generator + tracker policy (numpy + onnxruntime).

The single deployment runtime for the MuJoCo sim2sim runner and the hardware
controller (the torch twin it was ported from is gone -- sim2sim is ONNX-only): runs
the flow-matching generator (``fm.onnx``) online and feeds its world-frame prediction
into the RL tracker (``tracker.onnx``), reproducing the ``FmTrackingCommand`` coupling.
No torch / isaaclab / robot_rl imports.

The generator is **robot-state-conditioned**, exactly like the in-loop
``FmTrackingCommand``: the ``N`` history tokens are the LIVE MEASURED robot state
(phase), NOT the FM's own past output -- this fixes the autoregressive z-drift /
exposure bias. Each freshly sampled trajectory is anchored on the robot's current
base pose (xyz+yaw ground frame; node 0 = "now"), with exteroception token(s) -- the
height scan or the depth cameras, per the checkpoint -- describing the terrain. Depth
cameras additionally carry their geometry in the ONNX metadata, checked for agreement
between the two artifacts at load. The FM joint order need not equal the robot's isaac order (the FM
is trained in the clip-library order); measured joints are permuted isaac->FM for
the history and the prediction is permuted FM->CLF-output for the joint reference terms.

Same per-step contract as the torch version: ``create_obs(state)`` builds the
tracker observation from the FM prediction + measured proprioception;
``get_action(obs, joint_names_out)`` returns joint position targets.
"""

from __future__ import annotations

import time

import numpy as np

from .depth_obs import assert_depth_specs_agree
from .fm_session import build_cmd_tokens, build_extero_tokens, build_history_tokens, load_fm_context, sample_window
from .history import BatchedHistory
from .quat_math_np import quat_apply, quat_inv, quat_mul, yaw_quat
from .tracker_session import TrackerSession
from .window_np import (
    reanchor_window,
    rot6d_to_quat,
    window_block_slices,
    window_dim,
    world_to_window,
)

# Fallback clips for artifacts exported before the obs terms carried their own ``clip``
# (``export_parameters.py`` now emits it per term, and the metadata value wins). Anything
# clipped in the env cfg but absent here was silently unclipped on those older exports.
_LEGACY_TERM_CLIP = {"actions": 20.0, "height_scan": 1.0}

# The ``base_pose`` convention this runtime implements, checked against the tracker's
# exported ``reference_window_spec``. See ``robot_rl.flow_matching.window``: the block is
# ``[R_yawT (p_ref - p_act), rot6d(R_yawT R_ref)]``, in the anchor's YAW frame. Worth pinning
# because neither the yaw-vs-full frame nor the SIGN of the orientation half is visible in the
# obs shape -- both change the observation's MEANING at identical width.
_BASE_POSE_CONVENTION = "pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d"


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a ``(w, x, y, z)`` quaternion (MuJoCo) to ``(x, y, z, w)`` (FM/Isaac)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Reorder an ``(x, y, z, w)`` quaternion (FM/Isaac) to ``(w, x, y, z)`` (MuJoCo)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


class _TrackerOnlyCtx:
    """Minimal FM-context stand-in for tracker-only (no-FM) clip playback.

    When the generator is disabled (a pure clip-tracking teacher has no
    ``fm.onnx``), the clip reference drives the tracker directly.  This exposes
    only the fields the shared obs assembly (:meth:`GenTrackerPolicy._build_obs`)
    and the MuJoCo runner read in that mode: the joint order and the ee/target
    frame names (for ``ee_ref`` and the clip reference).  ``num_extra_frames`` is
    0 so the runner skips FM history-frame bookkeeping; ``N``/``F``/``D``/
    ``cmd_dim`` are inert placeholders (the generator is never sampled).  The
    exteroception fields are inert too: without a generator, the tracker's own obs terms
    (and hence its ``depth_camera_spec``) decide what the runner has to sense.
    """

    def __init__(self, joint_names: list[str], target_frame_names: list[str]):
        self.joint_names = list(joint_names)
        self.nj = len(joint_names)
        self.target_frame_names = list(target_frame_names)
        self.target_num_extra_frames = len(target_frame_names)
        self.frame_names: list[str] = []   # FM conditioning frames (none without FM)
        self.num_extra_frames = 0
        self.N = 1                          # history length (unused without FM)
        self.F = 1                          # prediction horizon (unused without FM)
        self.D = 0
        self.cmd_dim = 0
        self.height_dim = 1                 # exteroception conditioning (unused without FM)
        self.extero_source = "height_scan"
        self.extero_tokens = 1
        self.extero_spec: list[dict] = []


class GenTrackerPolicy:
    """Drive the RL tracker with the online flow-matching generator (numpy + ORT)."""

    def __init__(
        self,
        tracker_onnx: str,
        fm_onnx: str | None,
        fm_update_every: int = 1,
        base_frame_name: str = "pelvis_link",
        num_threads: int = 0,
        seed: int | None = 0,
        zero_noise: bool = False,
        null_history_warmup: bool = True,
        providers=None,
    ):
        """Configure the generator + tracker pair from the ONNX artifacts.

        ``fm_onnx`` may be ``None`` for tracker-only (no-FM) clip playback: the
        clip reference then drives the tracker directly (see :class:`_TrackerOnlyCtx`).

        ``null_history_warmup=True`` (default) samples the generator from control step 0 using
        the model's learned ``null_history`` embedding while the N-step history window fills.
        The RL side (``FmTrackingCommand``) does the same unconditionally -- it has no clip
        fallback either -- so the two agree from step 0. It needs an
        ``fm.onnx`` exported with the ``history_null`` input; without one the policy falls back to
        withholding the generator until the window is full (and warns once).

        ``zero_noise=True`` replaces the Gaussian ``x0`` with zeros (see
        :meth:`_sample_noise`). It defaults to False here -- the training-matched
        stochastic sampling -- so hardware keeps that behaviour; the MuJoCo sim2sim
        runner opts in and defaults it on.

        ``providers`` selects the onnxruntime execution providers for BOTH graphs (see
        ``tracker_session.resolve_providers``); ``None`` = CPU, which is the deployment
        default. Both sessions get the same list -- a split would be measurable only as
        an inexplicable latency difference between the two nets.
        """
        self.tracker_onnx = tracker_onnx
        self.fm_onnx = fm_onnx
        self.fm_update_every = max(int(fm_update_every), 1)
        self.base_frame_name = base_frame_name
        self.num_threads = num_threads
        self.zero_noise = bool(zero_noise)
        self.null_history_warmup = bool(null_history_warmup)
        self.providers = providers
        self._rng = np.random.default_rng(seed)
        self._depth_spec: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading + setup
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the tracker + FM ONNX sessions and build the orderings/buffers."""
        self._trk = TrackerSession(self.tracker_onnx, self.num_threads, self.providers)
        self.fm_enabled = self.fm_onnx is not None
        if self.fm_enabled:
            self.ctx = load_fm_context(self.fm_onnx, self.num_threads, self.providers)
            if self.ctx.cmd_dim <= 0:
                raise ValueError("GenTrackerPolicy expects a command-conditioned fm.onnx.")
        # Re-plan cadence: the tracker metadata (the value the tracker task trained with),
        # else the constructor/CLI default. The old middle tier -- falling back to the FM's
        # fm_snap_every -- existed only to satisfy the agreement rule that has since been
        # dropped; with the cadence now a free choice, a teacher export (which carries no
        # fm_update_every, since the field lives on FmTrackingCommandCfg rather than on the
        # clip command) should take the caller's value rather than the recording's.
        meta_every = self._trk.params.get("fm_update_every")
        if meta_every is None and self.fm_enabled:
            raise ValueError(
                f"{self.tracker_onnx} carries no 'fm_update_every', but an fm.onnx was given, "
                "so the re-plan cadence would fall back to the caller's default and the "
                "generator would be sampled at a rate the tracker never trained with. Against a "
                "K=12 pair the MuJoCo runner's default of 2 re-plans 6x too often, and it "
                "presents as a wobbly policy rather than as a configuration error -- which is "
                "exactly how it was found.\n"
                "\n"
                "Cause: export_parameters.py reads the cadence off the traj_ref command term "
                "inside a try/except and OMITS the key when the term has no such field. So any "
                "play.py run without --fm_checkpoint rewrites tracker.onnx without it, while "
                "leaving the older fm.onnx untouched -- a tracker.onnx noticeably newer than its "
                "fm.onnx is the tell.\n"
                "\n"
                "Fix: re-export with play.py on the TRACKER task WITH --fm_checkpoint, so both "
                "artifacts come from one run.\n"
                "\n"
                "Only raised when a generator is actually loaded: a teacher export legitimately "
                "carries no cadence (the field lives on FmTrackingCommandCfg, not on the clip "
                "command), and a tracker-only deploy re-plans nothing."
            )
        if meta_every is not None:
            self.fm_update_every, src = max(int(meta_every), 1), "tracker metadata"
        else:
            src = "constructor default"
        print(f"[INFO GenTrackerPolicy] fm_update_every={self.fm_update_every} (from {src})")
        if self.fm_enabled:
            if self.zero_noise:
                print("[INFO GenTrackerPolicy] zero_noise=True: the flow is integrated from "
                      "x0 = 0 instead of x0 ~ N(0, I) -- one deterministic trajectory per "
                      "conditioning, not the stochastic sampling the FM was trained for.")
        else:  # noqa: PLR5501 -- mirrors the fm_enabled branch above for readability
            # Tracker-only (no FM): a clip reference drives the tracker directly. Build a
            # lightweight ctx from the tracker metadata so the shared obs assembly + the
            # runner's frame/ee bookkeeping work without a generator (clip-only playback).
            self.ctx = self._tracker_only_ctx()
            print("[INFO GenTrackerPolicy] No fm.onnx -> tracker-only mode: the clip reference "
                  "drives the tracker; the generator is disabled.")
        self._resolve_depth_cameras()
        self._build_orderings()
        self._build_window()
        self._check_horizon()
        self._history = BatchedHistory(1, self.ctx.N)
        # ONE buffer: the F predicted reference-window nodes, which ARE the observation.
        self._pred = np.zeros((1, self.ctx.F, self.ctx.D))
        self._pred_valid = False
        self._ctrl_step = -1
        # -- receding-horizon serving state (mirrors FmTrackingCommand) --
        # Node currently being served within the window; 0 on a re-plan, +1 per step after.
        self._pred_idx = 0
        # The robot pose the live prediction is expressed RELATIVE TO, captured at the
        # re-plan. At fm_update_every > 1 the robot moves away from it, so every served node
        # is re-anchored onto the robot now (see _serve_nodes). At K=1 this IS the current
        # pose and the re-anchor is an identity to ~2e-15.
        self._anchor_pos = np.zeros(3)
        self._anchor_quat = np.array([0.0, 0.0, 0.0, 1.0])
        # Node served on the PREVIOUS step, already re-anchored onto the robot as it was
        # then -- the frame the dataset's history rows use. None until the first serve.
        self._prev_served_node: np.ndarray | None = None
        self._future_cmd_override = None  # set by clip→FM handoff to feed the actual future cmd
        # Last generator sample's inputs + identity, for logging (see _run_fm / last_plan).
        self._plan_id = 0
        self._last_plan: dict | None = None
        self._last_reference_window: np.ndarray | None = None
        # Per-term observation-history windows, keyed by term name (see _push_history).
        self._obs_history: dict[str, list[np.ndarray]] = {}
        self._reset_timing()
        self._loaded = True

    def _reset_timing(self) -> None:
        """Zero the FM-generator + tracker inference-time accumulators."""
        self._fm_infer_s = 0.0
        self._fm_infer_n = 0
        self._trk_infer_s = 0.0
        self._trk_infer_n = 0

    def _resolve_depth_cameras(self) -> None:
        """Settle on ONE depth-camera contract for the whole deployment.

        There is a single physical camera pair, so a depth tracker and a depth generator
        running together must have trained on the same one.  Both artifacts carry their
        own contract (``policy_parameters.depth_camera_spec`` /
        ``fm_meta.extero_spec``); when both are present they must agree exactly, and the
        resolved contract is what the runner casts with.

        Raises:
            ValueError: If the two artifacts disagree, or if a depth-conditioned
                generator has no contract to check and none to fall back on.
        """
        tracker_spec = self._trk.get_depth_camera_spec() or []
        fm_spec = list(self.ctx.extero_spec or [])
        if tracker_spec and fm_spec:
            assert_depth_specs_agree(tracker_spec, fm_spec, "tracker.onnx", "fm.onnx")
        self._depth_spec = tracker_spec or fm_spec

        if self.ctx.extero_source == "depth":
            if not self._depth_spec:
                raise ValueError(
                    "fm.onnx conditions on depth but neither artifact carries a "
                    "depth_camera_spec, so the deployment cannot reproduce its cameras. "
                    "Re-export the FM (its recording must have one) or the tracker."
                )
            if len(self._depth_spec) != self.ctx.extero_tokens:
                raise ValueError(
                    f"fm.onnx expects {self.ctx.extero_tokens} depth token(s) but the "
                    f"camera contract lists {len(self._depth_spec)} camera(s)."
                )
            for cam in self._depth_spec:
                pixels = int(cam["width"]) * int(cam["height"])
                if pixels != self.ctx.height_dim:
                    raise ValueError(
                        f"camera '{cam.get('sensor')}' is {cam['width']}x{cam['height']} "
                        f"= {pixels} pixels but fm.onnx's token is "
                        f"{self.ctx.height_dim} wide."
                    )
        if self._depth_spec:
            where = "tracker.onnx" if tracker_spec else "fm.onnx"
            print(f"[INFO GenTrackerPolicy] depth cameras (from {where}): "
                  + ", ".join(
                      f"{c.get('sensor', c['obs_term'])} {c['width']}x{c['height']} "
                      f"on {c['mount_body']}" for c in self._depth_spec
                  ))

    def get_depth_camera_spec(self) -> list[dict]:
        """The resolved depth-camera contract (empty when the policy has no cameras).

        The runner reads this to build its ray caster, so sim and hardware cast exactly
        the cameras the networks were trained on.
        """
        return list(self._depth_spec)

    def get_obs_terms(self) -> list[dict]:
        """The tracker's ordered observation terms, from its embedded metadata.

        Lets a consumer ask what the policy actually reads -- e.g. whether it wants a
        height scan -- without reaching into the tracker session.
        """
        return self._trk.get_obs_terms()

    def active_providers(self) -> dict[str, list[str]]:
        """The execution providers each loaded ONNX session is ACTUALLY running on.

        Not the same question as :func:`onnxruntime.get_available_providers`, which reports
        what the installed wheel was compiled with and is a static list. A session can be
        created with ``providers=["cuda"]`` on a CUDA-capable wheel and still register only
        the CPU provider -- missing runtime libraries, no nvidia container runtime, a driver
        too old for the CUDA minor -- and onnxruntime reports that as a warning, not an
        error. This is the only authority on what a graph will run on.

        Both sessions are reported because the generator is ~99% of the per-step cost, so a
        tracker that made it onto the GPU says almost nothing about the budget.

        Returns:
            Session name (``"tracker"``, and ``"generator"`` when one is loaded) mapped to
            its registered provider list, most-preferred first.
        """
        sessions = {"tracker": self._trk.sess}
        if self.fm_enabled:
            sessions["generator"] = self.ctx.sess
        return {name: list(sess.get_providers()) for name, sess in sessions.items()}

    def needs_height_scan(self) -> bool:
        """Whether this deployment must supply ``state["height_scan"]``.

        True if the tracker has the observation term, or the generator conditions on a
        multi-ray scan.  Either consumer alone is enough to make the reading mandatory.
        """
        if any(term["name"] == "height_scan" for term in self.get_obs_terms()):
            return True
        return self.ctx.extero_source == "height_scan" and self.ctx.height_dim > 1

    def _tracker_only_ctx(self) -> _TrackerOnlyCtx:
        """Build a minimal FM-context stand-in from the tracker metadata (no generator).

        The ee/target frame names come from the tracker's ``ee_frame_names``
        metadata (the ``ee_ref`` output frames); re-export the tracker if absent.
        """
        ee = self._trk.get_ee_frame_names()
        if ee is None:
            raise ValueError(
                "Tracker-only (no-FM) playback needs 'ee_frame_names' in the tracker.onnx "
                "metadata; re-export the tracker with the updated export_parameters.py "
                "(scripts/rsl_rl/export_tracker_onnx.py)."
            )
        return _TrackerOnlyCtx(list(self._trk.get_joint_names()), list(ee))

    def _build_orderings(self) -> None:
        """Resolve joint perms between the robot (isaac), the FM, and the CLF-output orders.

        The FM joint order (``ctx.joint_names``) need NOT equal the robot's isaac
        order -- a clip-conditioned FM is trained in the clip-library joint order.
        Mirrors the in-loop ``FmTrackingCommand`` (which permutes both ways):
        measured joints are permuted isaac->FM for the robot-state history, and the FM
        prediction is permuted FM->isaac for the ``reference_window`` joint blocks (and
        FM->CLF-output for the legacy ``joint_*_ref`` terms a clip reference may serve).
        """
        isaac = list(self._trk.get_joint_names())
        self.joint_names_isaac = isaac
        joint_out = self._trk.get_joint_out_names()
        if joint_out is None:
            print(
                "[WARN GenTrackerPolicy] tracker.onnx metadata has no 'joint_out_names'; falling "
                "back to joint_names_isaac for the CLF output order (WRONG if it differs). "
                "Re-export the tracker with the updated export_parameters.py."
            )
            joint_out = list(isaac)
        self.joint_out_names = joint_out
        if self.fm_enabled:
            fm = list(self.ctx.joint_names)
            # Measured (isaac order) -> FM order, for the robot-state history input.
            self._isaac_to_fm = np.array([isaac.index(n) for n in fm], dtype=np.int64)
            # FM prediction (FM order) -> CLF output joint order.
            self._fm_to_out = np.array([fm.index(n) for n in joint_out], dtype=np.int64)
            # FM prediction (FM order) -> robot/isaac order. What `reference_window` wants:
            # its joint blocks line up elementwise with the joint_pos / joint_vel / actions
            # proprioception terms, which is a DIFFERENT order from joint_*_ref's.
            self._fm_to_isaac = np.array([fm.index(n) for n in isaac], dtype=np.int64)
        else:
            # Tracker-only (no FM): the clip reference (in isaac order) drives directly.
            self._isaac_to_fm = None
            self._fm_to_out = np.array([isaac.index(n) for n in joint_out], dtype=np.int64)
            self._fm_to_isaac = np.arange(len(isaac), dtype=np.int64)

    def _build_window(self) -> None:
        """Resolve the reference/command window offsets from the tracker metadata.

        The window is ``[0, stride, ..., (num_steps-1)*stride]`` control steps, offset 0
        being the CURRENT step -- see ``mdp/observations/observations.py:_window_offsets``.
        Read from the exported ``reference_window_spec`` rather than derived from a
        runner default, so the deploy samples the reference at exactly the offsets the
        policy trained on.

        Raises:
            ValueError: If the tracker observes a window term whose contract the export
                did not record, or if the two windows disagree on their offsets.
        """
        terms = {t["name"] for t in self._trk.get_obs_terms()}
        if "reference_window" not in terms:
            raise ValueError(
                "tracker.onnx has no 'reference_window' observation term. This runtime "
                "builds the merged window layout only; the retired per-term layout "
                "(base_pose_error / joint_*_ref / ee_ref / future_base_*) is gone. "
                f"Its terms are: {sorted(terms)}."
            )
        ref_spec = self._trk.get_reference_window_spec()
        cmd_spec = self._trk.get_command_window_spec()
        missing = [n for n in ("reference_window", "command_window") if n in terms and (
            (ref_spec if n == "reference_window" else cmd_spec) is None)]
        if missing:
            raise ValueError(
                f"tracker.onnx observes {missing} but its metadata carries no "
                f"{'/'.join(n + '_spec' for n in missing)}. The flat term width does not "
                "describe the sample offsets or the per-step block layout, so the "
                "reference cannot be rebuilt. Re-export with the updated "
                "scripts/rsl_rl/export_parameters.py."
            )
        self.reference_window_spec = ref_spec
        self.command_window_spec = cmd_spec
        if cmd_spec is not None and list(ref_spec["offsets"]) != list(cmd_spec["offsets"]):
            raise ValueError(
                f"reference_window offsets {ref_spec['offsets']} != command_window offsets "
                f"{cmd_spec['offsets']}; they index the same reference and must agree."
            )
        self.window_offsets = [int(o) for o in ref_spec["offsets"]]

        # The one thing about this term that a shape check cannot catch: which way round
        # ``base_pose``'s orientation half points. A tracker trained under the other
        # convention loads clean -- same width, same everything -- and simply steers wrong.
        #
        # REQUIRED, not defaulted. It used to fall back to this runtime's own value for
        # artifacts exported before the key existed, which was safe only while there was one
        # convention in history; now that the sign has actually changed, that default would
        # wave through every pre-key artifact as if it agreed.
        convention = ref_spec.get("base_pose_convention")
        if convention is None:
            raise ValueError(
                "tracker.onnx reference_window_spec carries no base_pose_convention. It "
                "predates the key, so it was exported under the legacy mixed convention "
                f"(position ref-minus-actual, orientation actual-minus-ref) and not "
                f"{_BASE_POSE_CONVENTION!r}. Re-export from the training run if the policy "
                "was trained under this runtime's convention; retrain if it was not."
            )
        if convention != _BASE_POSE_CONVENTION:
            raise ValueError(
                f"tracker.onnx reference_window base_pose_convention is {convention!r}, but "
                f"this runtime implements {_BASE_POSE_CONVENTION!r}. Tracker and runtime "
                "disagree about what the observation MEANS, which no shape check can see. "
                "Re-export if the policy was trained under this runtime's convention; "
                "retrain if it was not."
            )

        # Which BLOCKS the term packs. A teacher can drop blocks it does not need, so this
        # is the tracker's own list rather than the full five, and it is what the clip path
        # builds with.
        self.reference_blocks = tuple(str(name) for name, _ in ref_spec["blocks"])
        if not self.fm_enabled:
            return
        # The generator's output goes into this term verbatim, so the two artifacts must
        # name the same blocks. A disagreement is invisible where it matters: drop
        # ``joint_vel`` on one side only and the widths differ (caught late, as a shape
        # error deep in the obs assembly); drop it on both sides but in a different ORDER
        # -- or swap which one is missing between two same-width blocks -- and nothing
        # differs in shape at all, the policy just reads velocities as positions.
        if tuple(self.ctx.reference_blocks) != self.reference_blocks:
            raise ValueError(
                f"fm.onnx predicts window blocks {list(self.ctx.reference_blocks)} but "
                f"tracker.onnx observes {list(self.reference_blocks)}. The generator's "
                "output IS the tracker's reference_window observation -- there is no "
                "translation layer between them -- so the two have to be the same vector. "
                "Retrain the generator against this teacher (train.py resolves the blocks "
                "from the recording, so re-record with this teacher), or pair the tracker "
                "with a generator trained for it."
            )

    def _check_horizon(self) -> None:
        """Refuse a generator whose horizon cannot cover the tracker's window.

        The deploy half of ``FmTrackingCommand._check_horizon``, same rule: the cursor sits
        up to ``K - 1`` nodes past the re-plan when the furthest sample is read, so serving
        offset ``o`` at a re-plan period of ``K`` needs ``F >= o + K``. Nothing clamps -- a
        short horizon would silently freeze the far end of the window, which reads as a
        tracking bug rather than a horizon one.

        Raises:
            ValueError: If the horizon cannot cover the window at this re-plan period.
        """
        if not self.fm_enabled:
            return
        max_off = max(self.window_offsets)
        need = max_off + self.fm_update_every
        if need > self.ctx.F:
            raise ValueError(
                f"The FM predicts F={self.ctx.F} nodes, but serving offset {max_off} at a "
                f"re-plan period of {self.fm_update_every} needs F >= {need} (the cursor is "
                f"up to {self.fm_update_every - 1} nodes past the re-plan when the furthest "
                f"sample is read). Lower fm_update_every, or re-record with "
                f"--ref_future_len {need} and retrain with --future_len {need}."
            )
        if self.fm_update_every > 1:
            print(f"[INFO GenTrackerPolicy] fm_update_every={self.fm_update_every}: served "
                  "nodes are re-anchored onto the measured base each step, so the base pose "
                  f"in `state` must be good over {self.fm_update_every - 1} steps "
                  f"({(self.fm_update_every - 1) * self.get_dt():.2f} s) of egomotion. Only "
                  "the RELATIVE motion since the re-plan matters, not a global pose.")

    # ------------------------------------------------------------------
    # Snapshot + generator inference
    # ------------------------------------------------------------------
    def _make_snapshot(self, state: dict) -> dict[str, np.ndarray]:
        """Pack one autoregressive history row (1 env).

        The row is the reference-window node this runtime SERVED last step -- the
        generator's own output -- not a measured state. It is already expressed relative to
        the robot as it was at that step, which is how the dataset builds the history, so
        there is nothing to transform and no base pose to estimate. Before the first sample
        (and for the first ``N`` steps after a reset) there is nothing real to push; zeros
        go in and the learned ``null_history`` embedding covers them.

        ``command`` rides along for the held-command fallback in :meth:`_run_fm`.
        """
        node = (np.zeros((1, self.ctx.D)) if self._prev_served_node is None
                else self._prev_served_node.reshape(1, self.ctx.D).copy())
        return {"node": node, "cmd": np.array(state["command"], dtype=np.float64, copy=True)[None]}

    def warm_start(self, state: dict) -> None:
        """Make the generator usable at step 0.

        With ``null_history_warmup`` and a graph that accepts ``history_null`` this is a no-op
        beyond resetting the cadence: the first steps are sampled with the learned null-history
        embedding, which is what the model was trained for. Only the legacy path fabricates a
        history by repeating this snapshot into all N slots -- a static window that never occurs
        in training (real history always has motion between its steps).
        """
        if self.null_history_warmup and getattr(self.ctx, "has_history_null", False):
            self._history.invalidate(np.ones(1, dtype=bool))
        else:
            self._history.fill(self._make_snapshot(state))
        self._pred_valid = False

    def observe(self, state: dict) -> None:
        """Append a real snapshot to the history without running the generator.

        Used during the stable-stand warmup: the window fills with measured frames
        while the FM stays dormant (``_pred_valid`` False), so the first live
        ``create_obs`` cleanly triggers the first FM sample anchored on a settled
        stand rather than on a static repeated snapshot.

        No-op in tracker-only (no-FM) mode: there is no generator history to keep.
        """
        if not self.fm_enabled:
            return
        self._ctrl_step += 1
        self._history.append(self._make_snapshot(state))

    def _update_generator(self, state: dict) -> None:
        """Advance the receding horizon by one control step.

        The step order IS the algorithm, and it mirrors
        ``FmTrackingCommand._snapshot_and_infer`` exactly:

        1. push the node served LAST step into the history (already in that step's frame);
        2. decide whether to re-plan -- on cadence, or whenever there is no live plan;
        3. otherwise advance the cursor one node into the existing plan;
        4. re-plan where asked, which resets the cursor and re-captures the anchor;
        5. stash the node served THIS step, re-anchored onto the robot as it is now, for
           the next step's push.

        The cadence is ``_ctrl_step % K``, an ABSOLUTE counter, matching Isaac's
        ``episode_length_buf % K``. A forced off-cadence re-plan (no live plan, e.g. after a
        reset or a stand warmup) therefore leaves the first segment SHORTER than ``K`` and
        exact thereafter -- Isaac behaves the same way, so this is deliberate.
        """
        self._ctrl_step += 1
        self._history.append(self._make_snapshot(state))
        null_hist = bool(self._history.null_mask()[0])
        use_null = (self.null_history_warmup and getattr(self.ctx, "has_history_null", False))

        reinfer = (self._ctrl_step % max(self.fm_update_every, 1)) == 0
        infer = reinfer or not self._pred_valid
        if self._pred_valid and not infer:
            # Walk one node further into the live plan. Assert rather than clamp: freezing
            # the tail silently would look like a tracking failure, not a horizon one.
            self._pred_idx += 1
            if self._pred_idx >= self.ctx.F:
                raise RuntimeError(
                    f"FM prediction cursor ran past the horizon (F={self.ctx.F}) at "
                    f"fm_update_every={self.fm_update_every}: the re-plan cadence is not "
                    "firing. Check _ctrl_step and the reset path."
                )
        # Gate: normally "history complete", but with the null-history warmup the generator runs
        # from step 0 and the incomplete window is replaced by the learned embedding -- the same
        # flow as FmTrackingCommand, so RL and deploy agree on the first N-1 steps.
        if infer and ((not null_hist) or use_null):
            self._run_fm(state, history_null=(null_hist and use_null))
        if self._pred_valid:
            # RAW node at the cursor, in its plan's own frame -- never re-anchored onto the
            # robot now. Keeps odometry out of the FM's observation; see the Isaac-side
            # _push_served_node for why, and note _serve_nodes (which DOES re-anchor) is
            # still what feeds the POLICY.
            self._prev_served_node = self._pred[0, int(self._pred_idx)].copy()

    def _run_fm(self, state: dict, history_null: bool = False) -> None:
        """Sample the F-node reference window for this control step.

        Conditions on the ``N`` nodes this runtime last served, the live exteroception
        reading, and the future velocity command. The result needs no lift: it is already
        the observation, relative to the robot at this step. Mirrors
        ``FmTrackingCommand._run_fm``.
        """
        ctx = self.ctx
        # Padded so the window is N long even on the first control steps; where ``history_null``
        # is set the graph discards its content for the learned null-history embedding.
        nodes = self._history.stack_padded("node")
        cmd = self._history.stack_padded("cmd")
        history = build_history_tokens(ctx, nodes)
        # Exteroception token(s) at "now", from whichever sensor this checkpoint trained
        # on -- the runner puts both readings in the state. Depth is one token per camera
        # (already normalised, matching mdp.depth_obs_stack); the height scan is one token
        # of per-ray clearance. Matches how each is recorded and used in the RL obs.
        if ctx.extero_source == "depth":
            height = build_extero_tokens(ctx, np.asarray(state["depth"], dtype=np.float64)[None])
        else:
            height = build_extero_tokens(ctx, np.asarray(state["height_scan"], dtype=np.float64)[None])
        if self._future_cmd_override is not None:
            # Actual future-command horizon (clip->FM handoff) -- matches IsaacLab's
            # get_command_velocity_horizon. Without it the deploy holds the current
            # command, which is right for joystick (future unknown) but diverges from
            # IsaacLab during command ramps.
            fut_cmd = np.asarray(self._future_cmd_override, dtype=np.float64).reshape(1, ctx.F, ctx.cmd_dim)
        else:
            fut_cmd = np.broadcast_to(cmd[:, -1:], (1, ctx.F, ctx.cmd_dim)).copy()  # held command
        cmd_tokens = build_cmd_tokens(ctx, fut_cmd)
        noise = self._sample_noise()

        t0 = time.perf_counter()
        self._pred = sample_window(ctx, history, height, cmd_tokens, noise,
                                   history_null=np.array([bool(history_null)]))
        infer_s = time.perf_counter() - t0
        self._fm_infer_s += infer_s
        # Stash the conditioning alongside the output so a logger can record what the
        # generator was ASKED, not just what it answered -- the history in particular is the
        # runtime's own past output, so when it goes wrong it does so silently and compounds.
        # Only references are kept (these arrays are freshly built each call and would
        # otherwise be collected), so this costs a few KB and no copy.
        self._plan_id += 1
        self._last_plan = {
            "plan_id": self._plan_id,
            "ctrl_step": self._ctrl_step,
            "history_nodes": nodes,
            "history_cmd": cmd,
            "extero_tokens": height,
            "cmd_tokens": cmd_tokens,
            "history_null": bool(history_null),
            "infer_s": infer_s,
        }
        # Counting the calls, not just their total time: pop_avg_infer_ms divides by this, and
        # the runner suppresses the FM half of its timing line when the count is 0. Without it
        # the generator's inference cost -- ~99% of the deploy step budget -- was invisible in
        # sim2sim while the tracker's was reported every 5 s.
        self._fm_infer_n += 1
        # Node 0 = "now", and the frame it is expressed in is the robot AT THIS INSTANT.
        # Both are needed to serve the window later: the cursor to index it, the anchor to
        # undo the staleness once the robot has moved on.
        self._pred_idx = 0
        self._anchor_pos = np.asarray(state["root_pos"], dtype=np.float64).copy()
        self._anchor_quat = np.asarray(state["root_quat"], dtype=np.float64).copy()
        self._pred_valid = True

    def _sample_noise(self) -> np.ndarray:
        """Draw the sampler's initial point ``x0``, shape ``(1, F, D)``.

        Normally ``x0 ~ N(0, I)``, which is what the rectified flow was trained to
        transport: the ``fm.onnx`` graph Euler-integrates from this point to a sample
        of the future-window distribution.

        With ``zero_noise=True`` it returns exactly zeros -- the flow is integrated
        from the origin (the mode of the source Gaussian) instead of a random draw,
        giving one fixed, fully deterministic trajectory per conditioning. That removes
        sample-to-sample variability, so re-plan discontinuities can be attributed to the
        changing conditioning/anchor rather than to fresh noise, and it empirically tracks
        better in MuJoCo sim2sim (which is why ``gen_tracker_runner.py`` defaults to it).

        Be aware of what it is NOT: the integrated path is off-distribution for the
        sampler and is *not* the mean of the conditional distribution (the flow ODE is
        nonlinear, so the image of the source mode is not the target mean). The library
        default stays Gaussian for hardware and the torch/numpy parity tests.
        """
        shape = (1, self.ctx.F, self.ctx.D)
        return np.zeros(shape) if self.zero_noise else self._rng.standard_normal(shape)

    def _serve_nodes(self, state: dict, offsets) -> np.ndarray:
        """Served nodes at ``offsets``, re-expressed relative to the robot RIGHT NOW.

        The deploy half of ``FmTrackingCommand._reanchor_to_now``. Offsets are relative to
        the CURSOR (node ``_pred_idx + offset``), not to the start of the window, so "offset
        0" means "now" whatever the re-plan period.

        The prediction is anchored on the robot as it was at the re-plan, but the
        observation's contract is "relative to the robot's current actual base". Those
        coincide only at ``fm_update_every=1``; at K>1 they differ by up to ``K-1`` steps of
        egomotion, and serving the stale frame would hand the policy a window wrong by
        exactly the distance travelled since the re-plan.

        Applied unconditionally, including at K=1, because that is what Isaac does -- and it
        is an identity there to ~2e-15, so there is nothing to save by special-casing it.

        Goes through :func:`~window_np.reanchor_window` rather than lifting to the world and
        back, because only ``base_pose`` and ``base_vel`` reference the anchor and the round
        trip cannot be completed for a block subset at all.

        Args:
            state: The step's state dict, supplying the measured base pose.
            offsets: Offsets from the current served node.

        Returns:
            ``[len(offsets), D]`` window nodes in the current robot frame.
        """
        idx = self._pred_idx + np.asarray(offsets, dtype=int)
        return reanchor_window(
            self._pred[0, idx],
            self._anchor_pos[None, :], self._anchor_quat[None, :],
            np.asarray(state["root_pos"], dtype=np.float64)[None, :],
            np.asarray(state["root_quat"], dtype=np.float64)[None, :],
            len(self._fm_to_isaac), self.ctx.target_num_extra_frames, self.reference_blocks,
        )

    def _fm_reference_window(self, state: dict) -> np.ndarray:
        """The predicted nodes at the tracker's window offsets, flattened for the obs.

        A gather plus a re-anchor. The generator emits exactly the vector
        ``mdp.observations.reference_window`` feeds the policy, so nothing between the
        network and the observation reinterprets anything -- the only transform is the frame
        correction :meth:`_serve_nodes` applies, which is an identity at ``K=1``.
        """
        return self._serve_nodes(state, self.window_offsets).reshape(-1)

    def predicted_base_pose_w(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The predicted window's reference base pose, lifted to the world frame.

        FOR DIAGNOSTICS ONLY. The live path never needs this -- the prediction IS the
        observation, which is the whole reason no world pose appears anywhere in the FM
        branch. A viewer that wants to draw the plan has to invert the one block that
        references the anchor. This is the numpy half of
        ``flow_matching.window.window_to_world``'s ``base_pose`` case and must stay
        identical to it:

        * position: forward is ``quat_apply_inverse(q_a, ref_p - p_a)``, so
          ``ref_p = p_a + quat_apply(q_a, block)``;
        * orientation: forward is ``Log(R_a R_refT)`` expressed in the anchor frame, so
          ``ref_q = Exp(-quat_apply(q_a, block)) * q_a`` -- the minus belongs to the legacy
          mixed convention, not to a sign slip.

        Lifts through the STORED re-plan anchor, like ``FmTrackingCommand._pred_world``, and
        takes no pose argument on purpose: the frame the prediction is expressed in is a
        property of the prediction, not of the caller. Handing in the current pose would be
        right only at ``fm_update_every=1`` and would silently shear the whole ribbon by the
        egomotion since the re-plan at any larger period.

        Returns ALL ``F`` nodes, absolute -- not offset from the cursor. The ribbon wants the
        whole plan, and routing it through the serving path would also apply the serving
        horizon check, which offset ``F-1`` cannot satisfy at ``K > 1``.

        Returns:
            ``([F, 3] positions, [F, 4] xyzw orientations)``, or ``None`` when there is no
            live prediction yet or the window has no ``base_pose`` block to invert. The
            quaternions come back sign-normalised (see :func:`quat_from_matrix`).
        """
        if not self._pred_valid:
            return None
        sl = window_block_slices(
            len(self._fm_to_isaac), self.ctx.target_num_extra_frames, self.reference_blocks
        ).get("base_pose")
        if sl is None:
            return None
        block = self._pred[0, :, sl]                                          # [F,9]
        # YAW-only, matching how the block was built: the anchor contributes heading, so
        # lifting through the full anchor rotation would tilt the whole ribbon by the robot's
        # roll/pitch at the re-plan.
        q = np.broadcast_to(yaw_quat(self._anchor_quat), (block.shape[0], 4))
        ref_p = self._anchor_pos + quat_apply(q, block[:, 0:3])
        # R_ref = R_yaw (R_yawT R_ref); the 6D pair has to become a matrix here, so this is
        # where Gram-Schmidt runs (inside rot6d_to_quat).
        ref_q = quat_mul(q, rot6d_to_quat(block[:, 3:9]))
        return ref_p, ref_q

    def predicted_root_path_w(self) -> np.ndarray | None:
        """Just the position half of :meth:`predicted_base_pose_w` (``[F, 3]`` or ``None``)."""
        lifted = self.predicted_base_pose_w()
        return None if lifted is None else lifted[0]

    @property
    def served_node_index(self) -> int:
        """Absolute index of the window node being served this step (the cursor).

        ``0`` on a re-plan and ``+1`` per step until the next one, so it is always 0 at
        ``fm_update_every=1``. Diagnostics that want "the reference at this step" out of the
        absolute-node accessors above have to index with this rather than assume node 0.
        """
        return int(self._pred_idx)

    def predicted_joint_pos(self) -> np.ndarray | None:
        """The predicted window's reference joint angles, ``[F, nj]`` in ISAAC order.

        Frame-invariant, so unlike the base pose there is nothing to lift -- the block is
        served verbatim. ``None`` when there is no live prediction or no ``joint_pos``
        block.
        """
        if not self._pred_valid:
            return None
        sl = window_block_slices(
            len(self._fm_to_isaac), self.ctx.target_num_extra_frames, self.reference_blocks
        ).get("joint_pos")
        return None if sl is None else np.asarray(self._pred[0, :, sl], dtype=np.float64)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------
    def create_obs(self, state: dict) -> np.ndarray:
        """Step the generator and build the tracker policy observation ``[1, num_obs]``."""
        if not self._loaded:
            raise RuntimeError("Call load() before create_obs().")
        if not self.fm_enabled:
            raise RuntimeError(
                "create_obs() needs the FM generator; tracker-only (no-FM) playback must drive "
                "the tracker via create_obs_from_reference(state, <clip reference>)."
            )
        self._update_generator(state)
        return self._build_obs(state, reference=None)

    def last_plan(self) -> dict | None:
        """The generator's most recent sample, inputs and output together.

        FOR LOGGING ONLY -- nothing in the control path reads it. The generator re-plans one
        tick in ``fm_update_every``, so a logger should publish this when ``plan_id``
        changes rather than every tick, and record the per-tick cursor
        (:attr:`served_node_index`) separately.

        The conditioning matters as much as the output: the ``N`` history rows are the
        runtime's OWN previously served nodes, so an error there is invisible from outside
        and compounds autoregressively.

        Returns:
            ``None`` before the first sample (and in tracker-only mode, where there is no
            generator at all); otherwise a dict with ``plan_id``, ``ctrl_step``,
            ``history_nodes`` ``(1, N, D)``, ``history_cmd`` ``(1, N, cmd_dim)``,
            ``extero_tokens``, ``cmd_tokens``, ``history_null`` and ``infer_s``. The output
            window itself is :attr:`_pred`, exposed via :meth:`predicted_window`.
        """
        return self._last_plan

    def last_reference_window(self) -> np.ndarray | None:
        """The reference window handed to the tracker on the last observation build.

        FOR LOGGING ONLY. Flat ``[offset0(D), offset1(D), ...]`` over
        :attr:`window_offsets` and :attr:`reference_blocks`, already expressed relative to
        the robot -- i.e. exactly the ``reference_window`` observation term, before its clip
        and scale are applied.

        Returns:
            The window, or ``None`` before the first observation build.
        """
        return self._last_reference_window

    def predicted_window(self) -> np.ndarray | None:
        """The live plan ``[F, D]`` in its OWN anchor frame, or None if none is live.

        Deliberately not re-anchored onto the robot: the re-anchoring happens per step on
        the way into the observation, so logging the raw plan and the served window
        separately makes a frame bug in that step visible as a disagreement between them
        rather than baked identically into both.
        """
        return self._pred[0].copy() if self._pred_valid else None

    def create_obs_from_reference(self, state: dict, reference: dict) -> np.ndarray:
        """Build the tracker observation from an EXTERNAL reference, bypassing the generator.

        The baked-clip counterpart of :meth:`create_obs`: the caller supplies the world-frame
        reference window that the generator would otherwise have produced, so the tracker can
        be evaluated on a known-good motion with the FM out of the loop entirely. Used by
        ``gen_tracker_runner.py --clip`` and by the obelisk controller's debug-clip mode --
        both go through this one path, so sim2sim and hardware cannot diverge here.

        Safe to call with the generator loaded (it simply is not consulted) and with it
        absent, which is the usual clip-playback configuration.

        Args:
            state: The state dict :meth:`create_obs` takes.
            reference: ``{"window": [...]}``, one world-frame entry per
                :attr:`window_offsets` value, offset 0 first, each optionally carrying its
                own ``"command"``. See :meth:`deploy.clip_reference.ClipReference.reference_at`.

        Returns:
            The tracker observation ``[1, num_obs]``.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before create_obs_from_reference().")
        n_win, n_off = len(reference["window"]), len(self.window_offsets)
        if n_win != n_off:
            raise ValueError(
                f"reference window has {n_win} entries but the tracker observes {n_off} "
                f"offsets {list(self.window_offsets)}. Build it with "
                "ClipReference.reference_at(step, policy.window_offsets)."
            )
        return self._build_obs(state, reference=reference)

    def _reference_window(self, window: list[dict], root_pos, root_quat) -> np.ndarray:
        """Pack the reference window, mirroring ``mdp.observations.reference_window``.

        Per step, in ``reference_window_layout`` order: ``base_pose`` (9) -- the reference
        base pose in the robot's CURRENT YAW frame, position ``ref - actual`` (3) plus
        ``R_yawT R_ref`` as a 6D rotation (6); ``base_vel``
        (6) -- reference base lin+ang velocity rotated into that same current base frame;
        ``joint_pos`` / ``joint_vel`` (nj each) -- **in the robot's own joint order**, so
        they line up elementwise with the proprioception terms; ``ee_pos`` (3 per frame)
        -- reference end-effector positions in *that step's* reference base frame.
        Flattened ``[step0(D), step1(D), ...]``.

        Only the blocks the tracker's ``reference_window_spec`` names are packed -- a
        teacher can drop the ones it does not need, and this has to match it exactly.

        Every block is relative, so no world coordinate reaches the policy.

        Args:
            window: One reference state per window offset, offset 0 first.
            root_pos: The robot's measured base position, world frame.
            root_quat: The robot's measured base orientation, world xyzw.

        Returns:
            The flat window, ``[num_steps * D]`` for the selected blocks.
        """
        f64 = lambda x: np.asarray(x, dtype=np.float64)
        n_ee = self.ctx.target_num_extra_frames
        k = len(window)
        nj = len(self._fm_to_isaac)
        # Stack the window into the [K, ...] arrays the shared forward map takes, so the
        # convention lives in ONE place across sim and deploy rather than being restated
        # here.
        ref_pos = np.stack([f64(r["root_pos"]) for r in window])                 # [K,3]
        ref_quat = np.stack([f64(r["root_quat"]) for r in window])               # [K,4]
        ref_lin = np.stack([f64(r["root_lin"]) for r in window])                 # [K,3]
        ref_ang = np.stack([f64(r["root_ang"]) for r in window])                 # [K,3]
        jp = np.stack([f64(r["jp"])[self._fm_to_isaac] for r in window])         # [K,nj]
        jv = np.stack([f64(r["jv"])[self._fm_to_isaac] for r in window])         # [K,nj]
        ee = (np.stack([f64(r["frame_pos"])[:n_ee] for r in window]) if n_ee > 0
              else np.zeros((k, 0, 3)))                                          # [K,E,3]
        nodes = world_to_window(
            f64(root_pos)[None, :], f64(root_quat)[None, :],
            ref_pos, ref_quat, ref_lin, ref_ang, jp, jv, ee,
            blocks=self.reference_blocks,
        )                                                                        # [K,D]
        assert nodes.shape == (k, window_dim(nj, n_ee, self.reference_blocks)), nodes.shape
        return nodes.reshape(-1)

    def _build_obs(self, state: dict, reference: dict | None) -> np.ndarray:
        """Build the policy obs from proprioception + the world reference (numpy).

        Mirrors ``mdp/observations/observations.py`` exactly. The reference half is the
        merged ``reference_window`` + ``command_window`` pair; the proprioception half is
        the raw current step, which :meth:`_assemble` then stacks into each term's
        history window.
        """
        # The FM branch has no world-frame window to pack: its prediction IS the term.
        window = None if reference is None else reference["window"]
        if window is not None and len(window) != len(self.window_offsets):
            raise ValueError(
                f"Reference window has {len(window)} samples but the tracker was exported "
                f"for {len(self.window_offsets)} (offsets {self.window_offsets})."
            )
        f64 = lambda x: np.asarray(x, dtype=np.float64)
        root_pos = f64(state["root_pos"])
        root_quat = f64(state["root_quat"])
        command = f64(state["command"])
        default = self._trk.default_joint_angles
        inv_root = quat_inv(root_quat)

        values = {
            "base_ang_vel": f64(state["base_ang_vel"]),
            "projected_gravity": quat_apply(inv_root, np.array([0.0, 0.0, -1.0])),
            "joint_pos": f64(state["joint_pos"]) - default,
            "joint_vel": f64(state["joint_vel"]),
            "actions": np.asarray(self._trk.action_isaac, dtype=np.float64),
            "reference_window": (self._fm_reference_window(state) if window is None
                                 else self._reference_window(window, root_pos, root_quat)),
            # The joystick-level command over the same offsets. Deliberately separate from
            # the reference window so it stays clean: at deploy it comes from the joystick
            # or a scripted profile, not the generator. A clip reference supplies each
            # step's actual target_vel; the FM has no per-step command, so hold the
            # current one -- which is what the deployed joystick does anyway.
            "command_window": (np.tile(command, len(self.window_offsets)) if window is None
                               else np.concatenate(
                                   [f64(s["command"]) if "command" in s else command for s in window],
                                   axis=-1)),
        }
        # The served window is the single most diagnostic quantity in the loop -- it is what
        # the tracker was actually told to do -- and on the generator path it exists nowhere
        # else, having been re-anchored out of the raw plan on the way in. Stashed (not
        # copied; _assemble does not mutate it) so a logger need not recompute it and risk
        # a replay bug that looks exactly like a controller bug.
        self._last_reference_window = values["reference_window"]
        # Root-frame body-link targets (tracker trained with tracking_mode="body"). These
        # are pelvis-relative already, so they pass through untransformed. Only a CLIP
        # reference supplies them -- the generator predicts no body links, so on the FM
        # path there is no window to read and ``_assemble`` raises if the tracker wants
        # them. (``window`` is None exactly on that path.)
        if window is not None and "body_pos_root" in window[0]:
            values["body_pos_ref"] = f64(window[0]["body_pos_root"])
            values["body_vel_ref"] = f64(window[0]["body_vel_root"])
        # Height-scan obs term (only if the tracker was trained with it). The runner puts
        # the per-ray clearance in state["height_scan"]; _assemble applies the exported clip.
        if "height_scan" in state:
            values["height_scan"] = f64(state["height_scan"])
        # Depth: one normalised row per camera, in contract order. Keyed here by the term
        # name the export recorded, so their position in the vector stays _assemble's
        # (metadata-driven) job; _assemble then applies the exported obs clip.
        for cam, image in zip(self._depth_spec, state.get("depth", ())):
            values[cam["obs_term"]] = f64(image)
        return self._assemble(values)

    def _push_history(self, name: str, v: np.ndarray, history_length: int) -> np.ndarray:
        """Append ``v`` to term ``name``'s window and return it flattened oldest-first.

        Isaac Lab's :class:`CircularBuffer` semantics, which the policy trained against:
        the FIRST push fills every slot with that sample (so step 0 is a static window,
        not a zero-padded one), later pushes roll, and ``buffer`` reads oldest-at-index-0
        with the most recent last. The manager then flattens it step-major, so the vector
        is ``[v_{t-(H-1)}, ..., v_t]``.

        The caller has already clipped ``v`` and must scale AFTER stacking: Isaac Lab
        clips and scales each sample before it enters the buffer, and since both are
        elementwise and constant that is identical to doing it to the stacked window --
        as long as the scale is tiled across the history when it is a vector.

        Args:
            name: Observation term name (the buffer key).
            v: This step's single-sample value, already clipped.
            history_length: Window length in control steps.

        Returns:
            The flattened window, ``[history_length * v.size]``.
        """
        buf = self._obs_history.get(name)
        if buf is None or len(buf) != history_length or buf[0].shape != v.shape:
            buf = [v.copy() for _ in range(history_length)]      # first push fills the window
            self._obs_history[name] = buf
        else:
            buf.append(v.copy())
            buf.pop(0)
        return np.concatenate(buf, axis=-1)

    def _assemble(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Order, clip, stack and scale the terms per the embedded observation_terms.policy.

        The exported term table is the single authority on the vector: its order is the
        concatenation order, its ``shape`` is the FLAT width (history already multiplied
        in), and its ``clip`` / ``scale`` / ``history_length`` are what training applied.
        Per term, in Isaac Lab's order: clip the raw sample, push it through the term's
        history window (if any), then scale.
        """
        out = np.zeros(self.get_num_obs(), dtype=np.float32)
        idx = 0
        for term in self._trk.get_obs_terms():
            name, shape, scale = term["name"], term["shape"], term["scale"]
            hist = term["history_length"]
            if name not in values:
                if name in ("body_pos_ref", "body_vel_ref"):
                    raise NotImplementedError(
                        f"Obs term '{name}': this tracker was trained with "
                        "tracking_mode='body', so its reference must carry root-frame body "
                        "targets. Clip playback supplies them (ClipReference with "
                        "root_frame_bodies set); the FM path does NOT yet -- the generator "
                        "predicts joint angles, and turning those into root-frame link poses "
                        "needs a torch-free FK, which is not implemented here."
                    )
                raise NotImplementedError(
                    f"Generator-tracker obs term '{name}' not implemented (a depth term "
                    "means the caller supplied no state['depth'] rows)."
                )
            v = np.atleast_1d(values[name].astype(np.float32))
            # `shape` is the flat width INCLUDING the history stack, so the single-sample
            # value is that divided by the window length.
            step_dim = shape // hist if hist > 0 else shape
            if v.shape[0] != step_dim:
                raise ValueError(
                    f"Term '{name}': built dim {v.shape[0]} != metadata dim {step_dim}"
                    + (f" (= {shape} / history_length {hist})" if hist > 0 else "") + "."
                )
            clip = term["clip"] if term["clip"] is not None else _LEGACY_TERM_CLIP.get(name)
            if clip is not None:
                v = np.clip(v, -clip, clip)
            if hist > 0:
                v = self._push_history(name, v, hist)
                # A per-channel scale describes ONE sample, so it repeats across the window.
                if isinstance(scale, (list, tuple, np.ndarray)):
                    scale = np.tile(np.asarray(scale, dtype=np.float32).ravel(), hist)
            elif isinstance(scale, (list, tuple)):
                scale = np.asarray(scale, dtype=np.float32)
            out[idx:idx + shape] = v * scale
            idx += shape
        return out[None]

    # ------------------------------------------------------------------
    # Action + passthroughs
    # ------------------------------------------------------------------
    def get_action(self, obs: np.ndarray, joint_names_out: list[str]) -> np.ndarray:
        """Run the tracker; return joint position targets in ``joint_names_out`` order."""
        t0 = time.perf_counter()
        action = self._trk.get_action(obs, joint_names_out)
        self._trk_infer_s += time.perf_counter() - t0
        self._trk_infer_n += 1
        return action

    def pop_avg_infer_ms(self) -> dict[str, float]:
        """Return the mean FM + tracker inference time (ms) since the last call, then reset.

        Keys: ``fm_ms``/``fm_n`` (FM generator, one per re-plan -- runs every
        ``fm_update_every`` steps) and ``trk_ms``/``trk_n`` (tracker, one per step).
        Counts are 0 when a net has not run in the window (e.g. FM during warmup, or
        tracker-only playback).
        """
        fm_ms = 1e3 * self._fm_infer_s / self._fm_infer_n if self._fm_infer_n else 0.0
        trk_ms = 1e3 * self._trk_infer_s / self._trk_infer_n if self._trk_infer_n else 0.0
        stats = {"fm_ms": fm_ms, "fm_n": self._fm_infer_n, "trk_ms": trk_ms, "trk_n": self._trk_infer_n}
        self._reset_timing()
        return stats

    def set_future_command(self, future_cmd: np.ndarray | None) -> None:
        """Override the decoder's future-command horizon for subsequent FM samples.

        ``future_cmd`` is ``(F, cmd_dim)`` (the actual upcoming command, e.g. from a
        clip); ``None`` restores the held-current-command default (joystick deploy).
        """
        self._future_cmd_override = None if future_cmd is None else np.asarray(future_cmd, dtype=np.float64)

    def reset(self) -> None:
        """Reset the rolling histories + prediction state (the tracker net is stateless).

        Both windows go: the FM's conditioning history AND the per-term observation
        history. Clearing the latter makes the next ``create_obs`` re-fill each window
        with its first sample, which is what Isaac Lab does on an env reset -- keeping a
        stale window would feed the policy motion that happened before the reset.
        """
        if self._loaded:
            self._history = BatchedHistory(1, self.ctx.N)
            self._obs_history = {}
            self._pred_valid = False
            self._ctrl_step = -1
            self._future_cmd_override = None
            # Receding-horizon state. The stash in particular describes a pose from before
            # the reset, so keeping it would push a row from another episode into the
            # history -- ``_pred_valid`` already gates it, but leaving stale data to be
            # gated is the kind of thing that stops being gated when the warmup changes.
            self._pred_idx = 0
            self._anchor_pos = np.zeros(3)
            self._anchor_quat = np.array([0.0, 0.0, 0.0, 1.0])
            self._prev_served_node = None
            # NOT _plan_id: it identifies a sample, and restarting it at 0 would make two
            # plans from different episodes share an id in one recording.
            self._last_plan = None
            self._last_reference_window = None
            self._reset_timing()

    def get_dt(self) -> float:
        return self._trk.get_dt()

    def get_num_obs(self) -> int:
        return self._trk.get_num_obs()

    def get_num_actions(self) -> int:
        return self._trk.get_num_actions()

    def get_kp(self):
        return self._trk.get_kp()

    def get_kd(self):
        return self._trk.get_kd()

    def get_joint_names(self) -> list[str]:
        return self._trk.get_joint_names()
