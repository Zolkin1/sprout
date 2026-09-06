# Copyright 2026 Zachary Olkin. All rights reserved.

"""Interactive plan selection: show the planner's top candidates, let a human pick one.

Wired into ``gen_tracker_runner.py`` by ``--interactive_plans K``. At every re-plan the
runtime's conditioning is handed here instead of to the generator; this samples a pool of
candidate plans from the TORCH token planner, ranks them, draws them in the MuJoCo viewer in
distinct colours, prints a metrics table, and blocks on a keypress. The chosen plan is
returned to the runtime and tracked for the next ``fm_update_every`` control steps, at which
point the next set appears.

The point is to make the model's own uncertainty legible. Aggregate numbers say a plan is
0.06 m from the clip on average; they cannot say "at this stair the model was choosing
between climbing and walking, and it picked walking". Here that choice is on screen.

**Candidates come from the same distribution the deployed sampler draws from** -- one
``sample`` call on a batch-replicated prefix, no top-k or nucleus truncation -- so what is
shown is the conditional the model actually has, not a sharpened view of it. The greedy plan
is always candidate 0 and marked, because that is what ``planner_greedy=True`` serves.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# The token planner lives in `source/robot_rl`, which is NOT what a bare `import robot_rl`
# resolves to here: the editable install points at the MAIN checkout, so in a worktree it
# imports a `robot_rl` with no `token_planner` submodule and fails naming the submodule
# rather than the path. Inserted at the FRONT for that reason -- appending would leave the
# editable install winning. Derived from this file's own location because `transfer/` is
# outside both packages and has no package root to ask.
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_SIM_DIR))
for _p in (os.path.join(_REPO, "source", "robot_rl"),
           os.path.join(_REPO, "source", "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Per-candidate colours (RGBA), greedy first. Chosen to stay distinguishable against both
#: the terrain and the runner's own orange prediction ribbon.
CAND_RGBA = [
    (1.00, 0.55, 0.00, 0.95),   # 0 greedy -- the runner's own ribbon colour
    (0.20, 0.60, 1.00, 0.95),   # 1 blue
    (0.20, 0.85, 0.35, 0.95),   # 2 green
    (0.95, 0.25, 0.75, 0.95),   # 3 magenta
    (0.95, 0.90, 0.20, 0.95),   # 4 yellow
    (0.60, 0.40, 1.00, 0.95),   # 5 violet
]


class PlanChooser:
    """Sample, rank, draw and select among candidate plans at each re-plan.

    Attributes are set by the runner after construction: :attr:`viewer` (for drawing and for
    the sync that makes the candidates visible while this blocks) and :attr:`add_sphere`.

    Args:
        planner_pt: The token planner ``.pt`` -- the TORCH model, which is what can report a
            per-token log-probability. The runtime's own ONNX graph cannot.
        device: Torch device for the planner.
        num_candidates: How many to show (capped at the colour table).
        num_samples: Size of the pool drawn before ranking and de-duplication. Larger finds
            more distinct modes; the cost is one batched forward per re-plan.
        temperature: Softmax temperature for the pool.
        seed: RNG seed, so a session is reproducible.
        auto: Skip the prompt and take this candidate index every time (``None`` = ask).
    """

    def __init__(self, planner_pt: str, device: str = "cuda", num_candidates: int = 4,
                 num_samples: int = 32, temperature: float = 1.0, seed: int = 0,
                 auto: int | None = None, mode: str = "interactive") -> None:
        from robot_rl.token_planner.online import load_planner_context  # noqa: PLC0415
        self.ctx = load_planner_context(planner_pt, device=device)
        self.dev = torch.device(device)
        self.k = int(min(num_candidates, len(CAND_RGBA)))
        self.pool = max(int(num_samples), self.k)
        self.temperature = float(temperature)
        self.auto = auto
        self.gen = torch.Generator(device=self.dev.type).manual_seed(int(seed))
        if mode not in ("interactive", "greedy", "best_of"):
            raise ValueError(f"mode={mode!r} must be interactive, greedy or best_of")
        self.mode = mode
        self.viewer = None
        self.add_sphere = None
        self.step_idx = 0
        self._checked = False
        self._logp_sum = 0.0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _candidates(self, history, extero, cmd_tokens, history_null):
        """Draw the pool, de-duplicate, rank by joint log-probability, keep the best ``k``.

        Returns:
            ``(ids [k,N], logp [k], ent [k], greedy_ids [N])``.
        """
        m = self.ctx.model
        # NORMALIZE. `deploy.fm_session`'s builders return RAW tokens on purpose -- the
        # `fm.onnx` graph folds the affine in -- but this hook drives the TORCH model, which
        # is fitted on the dataset's NORMALIZED inputs. Fed raw, the xyz channels are out by
        # their per-camera offset (~[0.51, 0.03, -0.79] m over a 3.0 m scale), which puts the
        # ground plane about a quarter of a unit below where the model expects it --
        # indistinguishable, to the model, from a drop-off dead ahead. It then planned a
        # DESCENT on a flat slab, in every sample, which is what a systematically shifted
        # observation looks like and not a sampling failure. Mirrors
        # `robot_rl.flow_matching.online.build_{history,extero,cmd}_tokens`.
        self._assert_raw(extero)
        h = self.ctx.state_norm.normalize(
            torch.as_tensor(history, dtype=torch.float32, device=self.dev))
        e = self.ctx.extero_norm.normalize(
            torch.as_tensor(extero, dtype=torch.float32, device=self.dev))
        c = (None if cmd_tokens is None or self.ctx.cmd_norm is None
             else self.ctx.cmd_norm.normalize(
                 torch.as_tensor(cmd_tokens, dtype=torch.float32, device=self.dev)))
        nullm = torch.tensor([bool(history_null)], device=self.dev)

        g_ids, g_lp, g_ent = m.sample(h, e, c, nullm, greedy=True)
        n = self.pool
        hb, eb = h.expand(n, *h.shape[1:]), e.expand(n, *e.shape[1:])
        cb = None if c is None else c.expand(n, *c.shape[1:])
        s_ids, s_lp, s_ent = m.sample(hb, eb, cb, nullm.expand(n),
                                      temperature=self.temperature, greedy=False,
                                      generator=self.gen)
        ids = torch.cat([g_ids, s_ids], 0)
        logp = torch.cat([g_lp, s_lp], 0).sum(-1)
        ent = torch.cat([g_ent, s_ent], 0).mean(-1)
        # De-duplicate on the token sequence: a pool of 32 draws from a peaked conditional is
        # mostly repeats, and four copies of one plan is not a set of options. Greedy is index
        # 0 and is kept unconditionally, so it survives dedup even when the pool re-finds it.
        seen, keep = set(), []
        for i in range(ids.shape[0]):
            key = tuple(ids[i].tolist())
            if key in seen:
                continue
            seen.add(key)
            keep.append(i)
        keep_t = torch.tensor(keep, device=ids.device)
        order = torch.argsort(logp[keep_t], descending=True)
        rest = [int(keep_t[j]) for j in order if int(keep_t[j]) != 0][: self.k - 1]
        sel = torch.tensor([0] + rest, device=ids.device)
        return ids[sel], logp[sel], ent[sel], g_ids[0]

    # ------------------------------------------------------------------
    def _assert_raw(self, extero) -> None:
        """Refuse conditioning that is not RAW, once, on the first re-plan.

        The two runtimes disagree by design about who normalizes: ``deploy.fm_session``'s
        builders return raw tokens because the ONNX graph folds the affine in, while
        ``robot_rl.flow_matching.online``'s normalize before the torch model sees them. This
        hook bridges the two, so it is the one place that can get the convention backwards --
        and it did, silently: fed a raw row, the model saw the ground plane about a quarter
        unit low, read it as a drop-off, and planned a coherent DESCENT on flat ground in
        every sample. Nothing about that is distinguishable from a model that is simply wrong.

        The test is which hypothesis the data is closer to. A RAW row still carries the
        normalizer's own offset, so its per-channel mean sits near ``extero_norm.mean``; an
        already-normalized row sits near zero. Compared only over the channels the normalizer
        actually moves -- under ``contract`` normalization ``d`` and ``v`` are identity, so
        including them would drown the signal in channels that are equal either way.

        Args:
            extero: The exteroception row as the runtime built it.

        Raises:
            ValueError: If the row looks already-normalized.
        """
        if self._checked:
            return
        self._checked = True
        n = self.ctx.extero_norm
        mu = n.mean.detach().flatten().to(self.dev)
        sd = n.std.detach().flatten().to(self.dev)
        active = (mu.abs() > 1e-6) | ((sd - 1.0).abs() > 1e-6)
        if not bool(active.any()):
            return                      # identity normalizer: the two conventions coincide
        x = torch.as_tensor(extero, dtype=torch.float32,
                            device=self.dev).reshape(-1, mu.numel()).mean(0)
        d_raw = float((x[active] - mu[active]).abs().mean())
        d_norm = float(x[active].abs().mean())
        print(f"[plan] conditioning check: |x - norm.mean| {d_raw:.4f} vs |x| {d_norm:.4f} "
              f"over {int(active.sum())} normalizer-active channels -> "
              f"{'RAW (correct)' if d_raw < d_norm else 'ALREADY NORMALIZED'}")
        if d_raw >= d_norm:
            raise ValueError(
                f"the exteroception row handed to PlanChooser looks ALREADY NORMALIZED "
                f"(|x| {d_norm:.4f} <= |x - norm.mean| {d_raw:.4f}), but this hook "
                "normalizes it again. Double-normalizing shifts the ground plane and the "
                "model will plan confident nonsense rather than fail. If "
                "`deploy.fm_session.build_extero_tokens` now returns normalized tokens, drop "
                "the normalization in `_candidates` instead of relaxing this check.")

    # ------------------------------------------------------------------
    def _metrics(self, win: np.ndarray, ids, greedy_ids, logp, ent) -> dict:
        """Per-candidate summary. ``win`` is one ``(F, D)`` window in the plan frame.

        The geometry is read straight out of the window's ``base_pose`` position block, which
        under the ``plan`` anchor is the reference's displacement from the robot at this
        re-plan -- so ``fwd``/``lat``/``dz`` are metres travelled over the plan, not world
        coordinates, and are directly comparable between candidates.
        """
        pos = win[:, 0:3]
        z = pos[:, 2] - pos[0, 2]
        return {
            "logp": float(logp),
            "logp_tok": float(logp) / int(ids.shape[0]),
            "ent": float(ent),
            "ndiff": int((ids != greedy_ids).sum()),
            "fwd": float(pos[-1, 0]),
            "lat": float(pos[-1, 1]),
            # SIGNED, plus both extremes. `rise` and `peak_z` were BOTH max-based, so a
            # descending plan -- the stairs-down mode this tool exists to catch -- reported
            # 0.000 and read as flat. Never summarise a signed quantity one-sidedly.
            "dz": float(z[-1]),
            "zmin": float(z.min()),
            "zmax": float(z.max()),
        }

    # ------------------------------------------------------------------
    def __call__(self, policy, state, history, extero, cmd_tokens,
                 history_null) -> np.ndarray:
        """The ``plan_hook``: choose one window for this re-plan.

        Args:
            policy: The calling :class:`~deploy.gen_tracker.GenTrackerPolicy`.
            state: The world-state dict this plan is being drawn against.
            history / extero / cmd_tokens: Conditioning as the runtime built it.
            history_null: Whether the learned null-history embedding applies this step.

        Returns:
            ``(1, F, D)`` the chosen window, in the runtime's own layout.
        """
        if self.mode != "interactive":
            return self._headless(history, extero, cmd_tokens, history_null)
        ids, logp, ent, g_ids = self._candidates(history, extero, cmd_tokens, history_null)
        wins = self.ctx.model.decode(ids, self.ctx.state_norm).detach().cpu().numpy()
        mets = [self._metrics(wins[i], ids[i], g_ids, logp[i], ent[i])
                for i in range(wins.shape[0])]
        self._draw(policy, state, wins)
        choice = self._prompt(mets, wins)
        self.history.append({"step": self.step_idx, "chosen": choice, "metrics": mets})
        self.step_idx += 1
        return wins[choice][None]

    # ------------------------------------------------------------------
    def _headless(self, history, extero, cmd_tokens, history_null) -> np.ndarray:
        """Serve one plan with no viewer and no prompt -- the two eval arms.

        ``greedy``   per-token argmax, i.e. what ``planner_greedy=True`` deploys.
        ``best_of``  draw ``num_samples`` and serve the one with the highest JOINT
                     log-probability.

        The pair isolates a single question: greedy maximises each token given the prefix,
        which is NOT the same as maximising the product over the sequence, so it can commit
        to a locally-attractive first token and land in a globally unlikely plan. Re-ranking
        the model's own samples by their joint likelihood approximates the sequence argmax
        greedy misses. Nothing here rolls a candidate out or scores it with a critic -- the
        only judge is the generator's own density, so a difference between the arms is
        attributable to the DECODE RULE and to nothing else.

        Args:
            history / extero / cmd_tokens / history_null: Conditioning, as the runtime built it.

        Returns:
            ``(1, F, D)`` the served window.
        """
        self._assert_raw(extero)
        m = self.ctx.model
        h = self.ctx.state_norm.normalize(
            torch.as_tensor(history, dtype=torch.float32, device=self.dev))
        e = self.ctx.extero_norm.normalize(
            torch.as_tensor(extero, dtype=torch.float32, device=self.dev))
        c = (None if cmd_tokens is None or self.ctx.cmd_norm is None
             else self.ctx.cmd_norm.normalize(
                 torch.as_tensor(cmd_tokens, dtype=torch.float32, device=self.dev)))
        nullm = torch.tensor([bool(history_null)], device=self.dev)
        if self.mode == "greedy":
            ids, lp, _ = m.sample(h, e, c, nullm, greedy=True)
            best, logp = ids, float(lp.sum())
        else:
            n = self.pool
            ids, lp, _ = m.sample(h.expand(n, *h.shape[1:]), e.expand(n, *e.shape[1:]),
                                  None if c is None else c.expand(n, *c.shape[1:]),
                                  nullm.expand(n), temperature=self.temperature,
                                  greedy=False, generator=self.gen)
            tot = lp.sum(-1)
            j = int(tot.argmax())
            best, logp = ids[j:j + 1], float(tot[j])
        self.step_idx += 1
        self._logp_sum += logp
        return m.decode(best, self.ctx.state_norm).detach().cpu().numpy()

    # ------------------------------------------------------------------
    def _draw(self, policy, state, wins: np.ndarray) -> None:
        """Draw every candidate's root path, then sync so they are visible while we block.

        The lift to world reuses the runtime's OWN ``predicted_root_path_w`` by swapping the
        candidate into ``_pred`` and restoring afterwards, rather than re-deriving the
        anchor maths here. Re-deriving it is how the two drift: the window is anchor-relative
        and a lift with the wrong anchor is a plausible path in the wrong place.

        The anchor is set to the robot's CURRENT pose first -- which is what ``_run_fm``
        latches for this plan a few lines after the hook returns, so this is that same frame
        and not a guess.
        """
        if self.viewer is None or self.add_sphere is None:
            return
        scn = self.viewer.user_scn
        keep = (policy._pred.copy(), bool(policy._pred_valid),
                np.array(policy._anchor_pos, dtype=np.float64).copy(),
                np.array(policy._anchor_quat, dtype=np.float64).copy())
        try:
            policy._anchor_pos = np.asarray(state["root_pos"], dtype=np.float64).copy()
            policy._anchor_quat = np.asarray(state["root_quat"], dtype=np.float64).copy()
            policy._pred_valid = True
            scn.ngeom = 0
            for i in range(wins.shape[0]):
                policy._pred = wins[i][None]
                path = policy.predicted_root_path_w()
                if path is None:
                    continue
                rgba = CAND_RGBA[i % len(CAND_RGBA)]
                for p in path:
                    self.add_sphere(scn, p, rgba, 0.022 if i == 0 else 0.017)
        finally:
            policy._pred, policy._pred_valid = keep[0], keep[1]
            policy._anchor_pos, policy._anchor_quat = keep[2], keep[3]
        self.viewer.sync()

    # ------------------------------------------------------------------
    def _prompt(self, mets: list[dict], wins: np.ndarray) -> int:
        """Print the table and read the choice. Returns the candidate index."""
        names = ["greedy", "alt 1", "alt 2", "alt 3", "alt 4", "alt 5"]
        colours = ["orange", "blue", "green", "magenta", "yellow", "violet"]
        print(f"\n=== re-plan {self.step_idx} "
              f"({wins.shape[0]} distinct of {self.pool + 1} drawn) ===")
        print(f"{'#':<3}{'colour':<9}{'logP':>9}{'logP/tok':>10}{'H':>7}"
              f"{'!=grdy':>8}{'fwd m':>8}{'lat m':>8}{'dz m':>8}{'zmin':>8}{'zmax':>8}")
        for i, m in enumerate(mets):
            print(f"{i:<3}{colours[i]:<9}{m['logp']:>9.2f}{m['logp_tok']:>10.3f}"
                  f"{m['ent']:>7.2f}{m['ndiff']:>8d}{m['fwd']:>8.3f}{m['lat']:>8.3f}"
                  f"{m['dz']:>8.3f}{m['zmin']:>8.3f}{m['zmax']:>8.3f}")
        # Spread among the shown candidates: how much the choice actually matters here. A
        # near-zero spread means the model is confident and this re-plan is not a decision.
        ends = np.stack([w[-1, 0:3] for w in wins])
        spread = float(np.linalg.norm(ends[:, None] - ends[None], axis=-1).max())
        print(f"max endpoint spread: {spread:.3f} m")
        if self.auto is not None:
            return min(int(self.auto), wins.shape[0] - 1)
        while True:
            raw = input(f"pick [0-{wins.shape[0] - 1}], Enter=0, q=quit: ").strip()
            if raw.lower() in ("q", "quit"):
                raise KeyboardInterrupt("user quit the plan explorer")
            if raw == "":
                return 0
            if raw.isdigit() and int(raw) < wins.shape[0]:
                return int(raw)
            print("  not a listed candidate")
