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
from .fm_session import build_cmd_tokens, build_extero_tokens, build_history_tokens, load_fm_context, sample_world
from .history import BatchedHistory
from .quat_math_np import quat_apply, quat_box_minus, quat_inv
from .tracker_session import TrackerSession

# Fallback clips for artifacts exported before the obs terms carried their own ``clip``
# (``export_parameters.py`` now emits it per term, and the metadata value wins). Anything
# clipped in the env cfg but absent here was silently unclipped on those older exports.
_LEGACY_TERM_CLIP = {"actions": 20.0, "joint_vel_ref": 20.0, "height_scan": 1.0}


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a ``(w, x, y, z)`` quaternion (MuJoCo) to ``(x, y, z, w)`` (FM/Isaac)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


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
        fm_update_every: int = 2,
        future_horizon_s: float = 0.5,
        future_stride: int = 5,
        base_frame_name: str = "pelvis_link",
        num_threads: int = 0,
        seed: int | None = 0,
        zero_noise: bool = False,
        null_history_warmup: bool = True,
    ):
        """Configure the generator + tracker pair from the ONNX artifacts.

        ``fm_onnx`` may be ``None`` for tracker-only (no-FM) clip playback: the
        clip reference then drives the tracker directly (see :class:`_TrackerOnlyCtx`).

        ``null_history_warmup=True`` (default) samples the generator from control step 0 using
        the model's learned ``null_history`` embedding while the N-step history window fills,
        matching ``FmTrackingCommandCfg.fm_null_history_warmup`` on the RL side. It needs an
        ``fm.onnx`` exported with the ``history_null`` input; without one the policy falls back to
        withholding the generator until the window is full (and warns once).

        ``zero_noise=True`` replaces the Gaussian ``x0`` with zeros (see
        :meth:`_sample_noise`). It defaults to False here -- the training-matched
        stochastic sampling -- so hardware keeps that behaviour; the MuJoCo sim2sim
        runner opts in and defaults it on.
        """
        self.tracker_onnx = tracker_onnx
        self.fm_onnx = fm_onnx
        self.fm_update_every = max(int(fm_update_every), 1)
        self.future_horizon_s = future_horizon_s
        self.future_stride = future_stride
        self.base_frame_name = base_frame_name
        self.num_threads = num_threads
        self.zero_noise = bool(zero_noise)
        self.null_history_warmup = bool(null_history_warmup)
        self._rng = np.random.default_rng(seed)
        self._depth_spec: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading + setup
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the tracker + FM ONNX sessions and build the orderings/buffers."""
        self._trk = TrackerSession(self.tracker_onnx, self.num_threads)
        # Prefer the cadence baked into the tracker metadata (the value the policy trained
        # with) over the constructor/CLI default, so sim2sim + hardware can't silently drift
        # from training. Falls back to the constructor value when the export predates it.
        meta_every = self._trk.params.get("fm_update_every")
        if meta_every is not None:
            self.fm_update_every = max(int(meta_every), 1)
        print(f"[INFO GenTrackerPolicy] fm_update_every={self.fm_update_every} "
              f"({'from tracker metadata' if meta_every is not None else 'from constructor default'})")
        self.fm_enabled = self.fm_onnx is not None
        if self.fm_enabled:
            self.ctx = load_fm_context(self.fm_onnx, self.num_threads)
            if self.ctx.cmd_dim <= 0:
                raise ValueError("GenTrackerPolicy expects a command-conditioned fm.onnx.")
            if self.zero_noise:
                print("[INFO GenTrackerPolicy] zero_noise=True: the flow is integrated from "
                      "x0 = 0 instead of x0 ~ N(0, I) -- one deterministic trajectory per "
                      "conditioning, not the stochastic sampling the FM was trained for.")
        else:
            # Tracker-only (no FM): a clip reference drives the tracker directly. Build a
            # lightweight ctx from the tracker metadata so the shared obs assembly + the
            # runner's frame/ee bookkeeping work without a generator (clip-only playback).
            self.ctx = self._tracker_only_ctx()
            print("[INFO GenTrackerPolicy] No fm.onnx -> tracker-only mode: the clip reference "
                  "drives the tracker; the generator is disabled.")
        self._resolve_depth_cameras()
        self._build_orderings()
        self._build_future_offsets()
        F, nj, k = self.ctx.F, self.ctx.nj, self.ctx.target_num_extra_frames
        self._history = BatchedHistory(1, self.ctx.N)
        self._pred = {
            "root_pos": np.zeros((1, F, 3)), "root_quat": np.zeros((1, F, 4)),
            "root_lin": np.zeros((1, F, 3)), "root_ang": np.zeros((1, F, 3)),
            "jp": np.zeros((1, F, nj)), "jv": np.zeros((1, F, nj)),
            "frame_pos": np.zeros((1, F, k, 3)), "frame_quat": np.zeros((1, F, k, 4)),
            "frame_lin": np.zeros((1, F, k, 3)), "frame_ang": np.zeros((1, F, k, 3)),
        }
        self._pred_idx = 0
        self._pred_valid = False
        self._ctrl_step = -1
        self._future_cmd_override = None  # set by clip→FM handoff to feed the actual future cmd
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
        measured joints are permuted isaac->FM for the robot-state history, and the
        FM prediction is permuted FM->CLF-output for ``joint_pos_ref``/``joint_vel_ref``.
        """
        isaac = list(self._trk.get_joint_names())
        self.joint_names_isaac = isaac
        joint_out = self._trk.get_joint_out_names()
        if joint_out is None:
            print(
                "[WARN GenTrackerPolicy] tracker.onnx metadata has no 'joint_out_names'; falling "
                "back to joint_names_isaac for joint_pos_ref/joint_vel_ref (WRONG if the CLF output "
                "joint order differs). Re-export the tracker with the updated export_parameters.py."
            )
            joint_out = list(isaac)
        self.joint_out_names = joint_out
        if self.fm_enabled:
            fm = list(self.ctx.joint_names)
            # Measured (isaac order) -> FM order, for the robot-state history input.
            self._isaac_to_fm = np.array([isaac.index(n) for n in fm], dtype=np.int64)
            # FM prediction (FM order) -> CLF output joint order, for joint_*_ref.
            self._fm_to_out = np.array([fm.index(n) for n in joint_out], dtype=np.int64)
        else:
            # Tracker-only (no FM): the clip reference (in isaac order) drives directly.
            self._isaac_to_fm = None
            self._fm_to_out = np.array([isaac.index(n) for n in joint_out], dtype=np.int64)

    def _build_future_offsets(self) -> None:
        """Strided future step offsets covering the look-ahead window (>=1 offset)."""
        dt = self.get_dt()
        horizon_steps = max(1, int(round(self.future_horizon_s / dt)))
        stride = max(1, int(self.future_stride))
        offs = list(range(stride, horizon_steps + 1, stride))
        self.future_offsets = offs or [horizon_steps]

    # ------------------------------------------------------------------
    # Snapshot + generator inference
    # ------------------------------------------------------------------
    def _make_snapshot(self, state: dict) -> dict[str, np.ndarray]:
        """Pack one MEASURED world state into the FM history layout (1 env).

        Appended to the rolling history every control step (and used to warm-start
        it and to keep it warm during clip playback): the robot-state-conditioned FM's
        history IS the measured robot state. Measured joints are permuted from the
        robot's isaac order into the FM joint order; measured ee frames are already in
        the FM frame order.
        """
        # COPY, never view. ``np.asarray`` on an already-float64 array returns a VIEW, and
        # callers legitimately hand us slices of live simulator/robot state (e.g. the MuJoCo
        # runner's ``np.asarray(qpos[:3])``). Storing such a view in the rolling history makes
        # every slot alias the same buffer, so the whole window reads as the LATEST state and the
        # positional delta between history steps collapses to zero -- the FM then sees a robot
        # that never translated, which never occurs in training. Velocities happened to be safe
        # (they are computed, so allocated), which is what made this hard to see.
        a = lambda x: np.array(x, dtype=np.float64, copy=True)[None]
        jp = np.array(state["joint_pos"], dtype=np.float64, copy=True)
        jv = np.array(state["joint_vel"], dtype=np.float64, copy=True)
        if self._isaac_to_fm is not None:
            jp, jv = jp[self._isaac_to_fm], jv[self._isaac_to_fm]
        snap = {
            "root_pos": a(state["root_pos"]), "root_quat": a(state["root_quat"]),
            "root_lin": a(state["root_lin_vel_w"]), "root_ang": a(state["root_ang_vel_w"]),
            "jp": jp[None], "jv": jv[None], "cmd": a(state["command"]),
        }
        if self.ctx.num_extra_frames > 0:
            frames = np.concatenate(          # concatenate allocates, so this is already a copy
                [state["frame_pos"], state["frame_quat"], state["frame_lin"], state["frame_ang"]], axis=-1
            )
            snap["frames"] = frames[None]
        return snap

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
        self._pred_idx = 0

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
        """Append the LIVE robot-state snapshot and (re-)run or advance the prediction.

        The appended snapshot is always the MEASURED robot state (permuted into FM
        joint/frame order) -- the robot-state-conditioned FM conditions on the real
        robot history (phase), not its own past output, mirroring
        ``FmTrackingCommand._snapshot_and_infer``. ``_run_fm`` anchors the freshly
        sampled trajectory on the robot's current base pose.
        """
        self._ctrl_step += 1
        self._history.append(self._make_snapshot(state))
        null_hist = bool(self._history.null_mask()[0])
        use_null = (self.null_history_warmup and getattr(self.ctx, "has_history_null", False))
        # Gate: normally "history complete", but with the null-history warmup the generator runs
        # from step 0 and the incomplete window is replaced by the learned embedding -- the same
        # flow as FmTrackingCommand, so RL and deploy agree on the first N-1 steps.
        gate = (not null_hist) or use_null
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        due = (self._ctrl_step % self.fm_update_every) == 0
        if gate and (due or not self._pred_valid or exhausted):
            self._run_fm(state, history_null=(null_hist and use_null))
        elif self._pred_valid:
            self._pred_idx = min(self._pred_idx + 1, self.ctx.F - 1)

    def _run_fm(self, state: dict, history_null: bool = False) -> None:
        """Sample an F-step world trajectory anchored on the robot's current base pose.

        Conditions the FM on the N most-recent MEASURED robot states (phase history),
        a root-height token (live root z), and the future velocity command, then lifts
        the F-step ego-anchored prediction back to the world using the robot's current
        base pose as the xyz+yaw anchor (node 0 = "now", base offset ~= 0). Mirrors
        ``FmTrackingCommand._run_fm``.
        """
        ctx = self.ctx
        # Padded so the window is N long even on the first control steps; where ``history_null``
        # is set the graph discards its content for the learned null-history embedding.
        rp = self._history.stack_padded("root_pos")
        rq = self._history.stack_padded("root_quat")
        rl = self._history.stack_padded("root_lin")
        ra = self._history.stack_padded("root_ang")
        jp = self._history.stack_padded("jp")
        jv = self._history.stack_padded("jv")
        cmd = self._history.stack_padded("cmd")
        frames = self._history.stack_padded("frames") if ctx.num_extra_frames > 0 else None

        history = build_history_tokens(ctx, rp, rq, rl, ra, jp, jv, frames=frames)
        # Exteroception token(s) at "now", from whichever sensor this checkpoint trained
        # on -- the runner puts both readings in the state. Depth is one token per camera
        # (already normalised, matching mdp.depth_obs_stack); the height scan is one token
        # of per-ray clearance; legacy scalar-height FMs (height_dim==1) fall back to the
        # root z. Matches how each is recorded and used in the RL obs.
        if ctx.extero_source == "depth":
            height = build_extero_tokens(ctx, np.asarray(state["depth"], dtype=np.float64)[None])
        elif ctx.height_dim > 1:
            height = build_extero_tokens(ctx, np.asarray(state["height_scan"], dtype=np.float64)[None])
        else:
            height = build_extero_tokens(ctx, rp[:, -1, 2])
        if self._future_cmd_override is not None:
            # Actual future-command horizon (clip→FM handoff) -- matches IsaacLab's
            # get_command_velocity_horizon. Without it the deploy holds the current
            # command, which is right for joystick (future unknown) but diverges from
            # IsaacLab during command ramps.
            fut_cmd = np.asarray(self._future_cmd_override, dtype=np.float64).reshape(1, ctx.F, ctx.cmd_dim)
        else:
            fut_cmd = np.broadcast_to(cmd[:, -1:], (1, ctx.F, ctx.cmd_dim)).copy()  # held command
        cmd_tokens = build_cmd_tokens(ctx, fut_cmd)
        noise = self._sample_noise()

        # The robot NOW is the anchor: the most-recent history step IS the live robot
        # state, so its base pose is the xyz+yaw ground frame the FM was trained on.
        p_ref = rp[:, -1].copy()
        q_ref = rq[:, -1].copy()
        t0 = time.perf_counter()
        world = sample_world(ctx, history, height, cmd_tokens, p_ref, q_ref, noise,
                             history_null=np.array([bool(history_null)]))
        self._fm_infer_s += time.perf_counter() - t0
        self._fm_infer_n += 1

        self._pred["root_pos"] = world["root_pos_w"]
        self._pred["root_quat"] = world["root_quat_w"]
        self._pred["root_lin"] = world["root_link_lin_vel_w"]
        self._pred["root_ang"] = world["root_link_ang_vel_w"]
        self._pred["jp"] = world["joint_pos"]
        self._pred["jv"] = world["joint_vel"]
        if ctx.target_num_extra_frames > 0:
            self._pred["frame_pos"] = world["frame_pos_w"]
            self._pred["frame_quat"] = world["frame_quat_w"]
            self._pred["frame_lin"] = world["frame_lin_vel_w"]
            self._pred["frame_ang"] = world["frame_ang_vel_w"]
        # Node 0 = "now" (base offset ~= 0); the applied index advances from here.
        self._pred_idx = 0
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

    def _pred_at(self, idx: int) -> dict[str, np.ndarray]:
        """Gather the generator prediction at horizon index ``idx`` (squeezed to 1 env)."""
        idx = min(max(idx, 0), self.ctx.F - 1)
        return {k: v[0, idx] for k, v in self._pred.items()}

    def _reference_from_fm(self) -> dict:
        """Assemble the current + future world reference from the FM prediction buffers."""
        cur = self._pred_at(self._pred_idx)
        fut = [self._pred_at(self._pred_idx + o) for o in self.future_offsets]
        return {"current": cur, "future": fut}

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
                "the tracker via _build_obs(state, reference=<clip reference>)."
            )
        self._update_generator(state)
        return self._build_obs(state, reference=None)

    def _build_obs(self, state: dict, reference: dict | None) -> np.ndarray:
        """Build the policy obs from proprioception + the world reference (numpy).

        Mirrors ``mdp/observations/observations.py`` exactly (base-frame-rotated
        orientation errors; futures from the FM horizon; held future command).
        """
        ref = reference if reference is not None else self._reference_from_fm()
        f64 = lambda x: np.asarray(x, dtype=np.float64)
        root_pos = f64(state["root_pos"])
        root_quat = f64(state["root_quat"])
        command = f64(state["command"])
        default = self._trk.default_joint_angles
        inv_root = quat_inv(root_quat)

        cur = ref["current"]
        ref_pos = f64(cur["root_pos"]); ref_quat = f64(cur["root_quat"])
        ref_jp = f64(cur["jp"]); ref_jv = f64(cur["jv"])

        pos_err_b = quat_apply(inv_root, ref_pos - root_pos)
        ori_err_b = quat_apply(inv_root, quat_box_minus(root_quat, ref_quat))
        base_pose_error = np.concatenate([pos_err_b, ori_err_b], axis=-1)

        joint_pos_ref = ref_jp[self._fm_to_out]
        joint_vel_ref = ref_jv[self._fm_to_out]

        ee_parts = []
        if self.ctx.target_num_extra_frames > 0:
            inv_ref = quat_inv(ref_quat)
            frame_pos = f64(cur["frame_pos"])
            for kf in range(self.ctx.target_num_extra_frames):
                ee_parts.append(quat_apply(inv_ref, frame_pos[kf] - ref_pos))
        ee_ref = np.concatenate(ee_parts, axis=-1) if ee_parts else np.zeros(0)

        # Reference base velocity (lin+ang, world frame) rotated into the current base
        # frame -- the deployable, heading-invariant base_vel_ref (observations
        # .ref_base_velocity_local). Guarded so an injected reference without the
        # velocity channels (parity tests) simply omits the term.
        base_vel_ref = None
        if "root_lin" in cur:
            ref_lin, ref_ang = f64(cur["root_lin"]), f64(cur["root_ang"])
            base_vel_ref = np.concatenate(
                [quat_apply(inv_root, ref_lin), quat_apply(inv_root, ref_ang)], axis=-1
            )

        fcmd, fpose, fvel = [], [], []
        for step in ref["future"]:
            # Held current command (FM: no per-step command) or the clip's actual
            # future target_vel when the reference source supplies it (parity with
            # IsaacLab's future_command_velocity, which reads the clip lookahead).
            fcmd.append(f64(step["command"]) if "command" in step else command)
            fp = f64(step["root_pos"]); fq = f64(step["root_quat"])
            fl = f64(step["root_lin"]); fa = f64(step["root_ang"])
            fpose.append(np.concatenate(
                [quat_apply(inv_root, fp - root_pos), quat_apply(inv_root, quat_box_minus(root_quat, fq))], axis=-1
            ))
            fvel.append(np.concatenate([quat_apply(inv_root, fl), quat_apply(inv_root, fa)], axis=-1))

        values = {
            "base_ang_vel": f64(state["base_ang_vel"]),
            "projected_gravity": quat_apply(inv_root, np.array([0.0, 0.0, -1.0])),
            "joint_pos": f64(state["joint_pos"]) - default,
            "joint_vel": f64(state["joint_vel"]),
            "actions": np.asarray(self._trk.action_isaac, dtype=np.float64),
            "base_pose_error": base_pose_error,
            "joint_pos_ref": joint_pos_ref,
            "joint_vel_ref": joint_vel_ref,
            "ee_ref": ee_ref,
            "command_vel": command,
            "future_command_vel": np.concatenate(fcmd, axis=-1),
            "future_base_pose": np.concatenate(fpose, axis=-1),
            "future_base_vel": np.concatenate(fvel, axis=-1),
        }
        if base_vel_ref is not None:
            values["base_vel_ref"] = base_vel_ref
        # Root-frame body-link targets (tracker trained with tracking_mode="body").  These
        # are pelvis-relative already, so they pass through untransformed.  Only a clip
        # reference supplies them today -- see the guard in _assemble for the FM case.
        if "body_pos_root" in cur:
            values["body_pos_ref"] = f64(cur["body_pos_root"])
            values["body_vel_ref"] = f64(cur["body_vel_root"])
        # Height-scan obs term (only if the tracker was trained with it). The runner puts
        # the per-ray clearance in state["height_scan"]; _assemble clips it via _TERM_CLIP.
        if "height_scan" in state:
            values["height_scan"] = f64(state["height_scan"])
        # Depth: one normalised row per camera, in contract order. Keyed here by the term
        # name the export recorded, so their position in the vector stays _assemble's
        # (metadata-driven) job; _assemble then applies the exported obs clip.
        for cam, image in zip(self._depth_spec, state.get("depth", ())):
            values[cam["obs_term"]] = f64(image)
        return self._assemble(values)

    def _assemble(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Order, scale, and clip the terms per the embedded observation_terms.policy.

        TODO(deploy): this does NOT implement observation history, so a policy trained
        with ``history_length`` on any term fails the dim check below on the first
        control step (loudly, which is the intent -- the alternative is a silently
        misaligned observation vector). The current teacher does exactly that, so this
        path is broken until ported. What it owes, in order:

        1. per-term ring buffers keyed on ``term["history_length"]`` (already parsed by
           ``TrackerSession._resolve_obs_terms``), flattened OLDEST-FIRST and warm-started
           by replicating the first sample into every slot -- Isaac Lab's
           ``CircularBuffer`` semantics; ``transfer/sim/rl_policy.py:_update_history`` is
           a correct reference port;
        2. clip and scale applied PER TIMESTEP before stacking, not once over the stacked
           slice -- Isaac Lab bakes them into each slot, and ``scale`` may be a vector, so
           it needs tiling across the history (see ``rl_policy.py``'s ``np.tile``);
        3. the new merged ``reference_window`` / ``command_window`` terms built in
           ``_build_obs``, replacing the retired ``joint_pos_ref`` / ``ee_ref`` /
           ``future_base_*`` terms;
        4. ``GenTrackerPolicy.reset()`` clearing the new buffers.

        See also the two blocked FM look-ahead accessors in ``fm_tracking_cmd.py``, which
        must land before the generator can serve the window at all.
        """
        out = np.zeros(self.get_num_obs(), dtype=np.float32)
        idx = 0
        for term in self._trk.get_obs_terms():
            name, shape, scale = term["name"], term["shape"], term["scale"]
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
            if v.shape[0] != shape:
                raise ValueError(f"Term '{name}': built dim {v.shape[0]} != metadata dim {shape}.")
            clip = term["clip"] if term["clip"] is not None else _LEGACY_TERM_CLIP.get(name)
            if clip is not None:
                v = np.clip(v, -clip, clip)
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
        """Reset the rolling history + prediction state (the tracker is stateless)."""
        if self._loaded:
            self._history = BatchedHistory(1, self.ctx.N)
            self._pred_idx = 0
            self._pred_valid = False
            self._ctrl_step = -1
            self._future_cmd_override = None
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
