# Copyright 2026 Zachary Olkin. All rights reserved.

"""Keep the depth row beside the plan, for re-plans the classifier reads as the wrong kind.

``terrain_class`` says HOW OFTEN the generator reads a staircase as a box. It cannot say
WHY, and the two candidate causes want opposite fixes:

* **perception** -- the image genuinely looks box-like (a drifted pose, a stair edge outside
  the frame, the far camera cropping the flight), so the plan is a reasonable answer to a bad
  picture. The lever is what the generator SEES.
* **generation** -- the image is fine and a person looking at it would say "stairs", and the
  plan is wrong anyway. The lever is the model or its history.

Separating them needs the picture that produced the plan, which is why this exists: by the
time a plan is classified the frame is ``depth_latency`` steps stale and no longer
recoverable from the sensor, so it has to be stashed at the re-plan itself
(``FmTrackingCommand.enable_extero_stash``).

**Controls are not optional here.** Every stairs image looks somewhat box-like at a glance,
so a folder of flagged images alone will convince a reader of whatever they expected. The
bucket key is therefore ``(true class, predicted class)``, which collects the failure and its
own control in one pass: ``stairs->jump`` is the flagged set and ``stairs->stairs`` is the
same family, same terrain, classified correctly. Compare those two and the question is
answerable; look at the first alone and it is not.

Sampling is a per-bucket reservoir, so the kept set is uniform over the whole run rather than
the first N re-plans -- which would all come from the opening seconds of the earliest
episodes, where the robot has not reached its terrain yet.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from terrain_class import COARSE_CLASSES


class FlaggedPlanDump:
    """Reservoir-sample re-plans by ``(true class, predicted class)``, with their depth.

    Attributes:
        buckets: ``(true, pred) -> list`` of kept samples.
        n_seen: ``(true, pred) -> int``, how many were offered, for the sampling weight.
    """

    def __init__(self, clip_coarse: np.ndarray, *, per_bucket: int = 48,
                 seed: int = 0) -> None:
        """Allocate the reservoirs.

        Args:
            clip_coarse: ``(num_clips,)`` true coarse-class code per clip id, from the
                library's families through :data:`terrain_class.FAMILY_TO_COARSE`.
            per_bucket: Samples kept per ``(true, pred)`` bucket.
            seed: RNG seed for the reservoir, so a re-run keeps the same set.
        """
        self.clip_coarse = np.asarray(clip_coarse, dtype=np.int64)
        self.per_bucket = int(per_bucket)
        self.rng = np.random.default_rng(seed)
        #: (true class, step band) -> the most / least divergent plans kept for it.
        self.hi: dict[tuple[int, int], list[dict]] = {}
        self.lo: dict[tuple[int, int], list[dict]] = {}
        self.n_seen: dict[tuple[int, int], int] = {}
        #: env -> the plan it is currently serving, awaiting its divergence.
        self.pending: dict[int, dict] = {}

    def offer(self, ids: torch.Tensor, codes: torch.Tensor, win: torch.Tensor, cmd) -> None:
        """Stage this step's fresh re-plans as pending samples.

        A plan cannot be scored when it is made -- its divergence from the clip is only
        known once it has been served -- so it waits here until :meth:`finalize`.

        Args:
            ids: ``(V,)`` env indices that re-planned this step.
            codes: ``(V,)`` predicted coarse class per env.
            win: ``(num_envs, F, D)`` predicted windows.
            cmd: The FM command term, for ``clip_id`` and the stashed exteroception.
        """
        stash = getattr(cmd, "_extero_stash", None)
        if stash is None or ids.numel() == 0:
            return
        ids_np = ids.detach().cpu().numpy()
        pred = codes.detach().cpu().numpy()
        clip = cmd.clip_id.detach().cpu().numpy()[ids_np]
        for k, e in enumerate(ids_np.tolist()):
            c = int(clip[k])
            if not 0 <= c < len(self.clip_coarse):
                continue
            self.pending[int(e)] = dict(
                clip_id=c,
                step=int(cmd._step[e].item()) if hasattr(cmd, "_step") else -1,
                true_code=int(self.clip_coarse[c]), pred_code=int(pred[k]),
                extero=stash[e].detach().cpu().numpy().astype(np.float32),
                plan=win[e].detach().cpu().numpy().astype(np.float32),
            )

    def finalize(self, ids: torch.Tensor, div_sum: torch.Tensor,
                 div_n: torch.Tensor) -> None:
        """Score the pending plans of ``ids`` and offer them to the reservoirs.

        Args:
            ids: Envs whose previous plan has just ended.
            div_sum: ``(num_envs,)`` summed |served - clip| since that plan began.
            div_n: ``(num_envs,)`` steps it was served for.
        """
        if ids.numel() == 0:
            return
        ds = div_sum.detach().cpu().numpy()
        dn = div_n.detach().cpu().numpy()
        for e in ids.detach().cpu().numpy().tolist():
            rec = self.pending.pop(int(e), None)
            if rec is None or dn[e] <= 0:
                continue          # nothing served yet; scoring it would divide by zero
            rec["divergence"] = float(ds[e] / dn[e])
            rec["n_served"] = int(dn[e])
            self._admit(rec)

    #: clip_step edges for the bands a comparison is made WITHIN. Prediction difficulty
    #: rises through an episode (measured: rank correlation 0.29 between clip_step and
    #: |plan - clip|), so a worst-vs-best comparison pooled over the whole clip mostly
    #: compares late plans against early ones.
    STEP_BANDS: tuple[int, ...] = (0, 50, 150, 300, 10 ** 9)

    def _band(self, step: int) -> int:
        """Index of the clip_step band this plan falls in."""
        for i in range(len(self.STEP_BANDS) - 1):
            if self.STEP_BANDS[i] <= step < self.STEP_BANDS[i + 1]:
                return i
        return len(self.STEP_BANDS) - 2

    def _admit(self, rec: dict) -> None:
        """Keep the most and least divergent plans of each (true class, step band).

        NOT a reservoir over the classifier's label. Measured on the deployed generator, a
        plan the classifier flags is no more divergent from the clip than one it does not
        (~1.0x within matched step bands), so selecting on the label produces a sample that
        is uninformative about whether the plan was wrong. Divergence from the clip is the
        ground truth and needs no model.

        Both ends are kept because the question is comparative: the worst plans alone show
        whatever the reader expects, and only "what does the image look like when it goes
        wrong, versus when it goes right, at the same point of the same kind of clip"
        distinguishes a perception failure from a generation one.
        """
        key = (rec["true_code"], self._band(rec["step"]))
        self.n_seen[key] = self.n_seen.get(key, 0) + 1
        hi = self.hi.setdefault(key, [])
        lo = self.lo.setdefault(key, [])
        d = rec["divergence"]
        hi.append(rec)
        hi.sort(key=lambda r: -r["divergence"])
        del hi[self.per_bucket:]
        lo.append(rec)
        lo.sort(key=lambda r: r["divergence"])
        del lo[self.per_bucket:]

    def save(self, path: str, extra: dict | None = None) -> str:
        """Write the kept samples to one ``.npz``.

        Args:
            path: Output ``.npz``.
            extra: Extra scalars recorded alongside (cell key, checkpoint, ...).

        Returns:
            The path written.
        """
        rows, end, band = [], [], []
        for key in sorted(set(self.hi) | set(self.lo)):
            for tag, src in ((1, self.hi.get(key, [])), (0, self.lo.get(key, []))):
                for r in src:
                    rows.append(r)
                    end.append(tag)
                    band.append(key[1])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not rows:
            np.savez_compressed(path, empty=np.array(True),
                                classes=np.array(json.dumps(list(COARSE_CLASSES))))
            return path
        np.savez_compressed(
            path,
            extero=np.stack([r["extero"] for r in rows]),
            plan=np.stack([r["plan"] for r in rows]),
            clip_id=np.array([r["clip_id"] for r in rows], dtype=np.int64),
            step=np.array([r["step"] for r in rows], dtype=np.int64),
            true_code=np.array([r["true_code"] for r in rows], dtype=np.int64),
            pred_code=np.array([r["pred_code"] for r in rows], dtype=np.int64),
            divergence=np.array([r["divergence"] for r in rows], dtype=np.float64),
            n_served=np.array([r["n_served"] for r in rows], dtype=np.int64),
            # 1 = among the MOST divergent of its band, 0 = among the least. This is the
            # contrast to look at; the classifier code rides along as a secondary label.
            worst=np.array(end, dtype=np.int64),
            step_band=np.array(band, dtype=np.int64),
            step_band_edges=np.array(list(self.STEP_BANDS), dtype=np.int64),
            n_seen=np.array(json.dumps({f"{a}_{b}": n for (a, b), n in self.n_seen.items()})),
            classes=np.array(json.dumps(list(COARSE_CLASSES))),
            extra=np.array(json.dumps(extra or {})),
        )
        return path

    def summary(self) -> str:
        """What was kept, and how far apart the two ends of each band are."""
        out = [f"[plan-dump] classes {COARSE_CLASSES}; selection = |plan - clip| extremes",
               f"[plan-dump] step bands {list(self.STEP_BANDS)}"]
        for key in sorted(set(self.hi) | set(self.lo)):
            t, b = key
            hi, lo = self.hi.get(key, []), self.lo.get(key, [])
            if not hi or not lo:
                continue
            mh = float(np.median([r["divergence"] for r in hi]))
            ml = float(np.median([r["divergence"] for r in lo]))
            # How often the classifier calls the WORST plans a misread, against the best.
            fh = float(np.mean([r["true_code"] != r["pred_code"] for r in hi]))
            fl = float(np.mean([r["true_code"] != r["pred_code"] for r in lo]))
            out.append(
                f"  {COARSE_CLASSES[t]:>6s} band {b} (steps {self.STEP_BANDS[b]}-"
                f"{self.STEP_BANDS[b + 1] if self.STEP_BANDS[b + 1] < 10 ** 8 else 'end'}): "
                f"{self.n_seen.get(key, 0):6d} seen | worst {mh:.4f} m (clf flags "
                f"{100 * fh:4.0f}%) vs best {ml:.4f} m (clf flags {100 * fl:4.0f}%)")
        out.append("  (clf flag rates equal at both ends => the classifier label carries no "
                   "information about whether the plan was actually wrong)")
        return "\n".join(out)
