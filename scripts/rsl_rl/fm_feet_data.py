#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Assemble one re-plan's world-frame curves from an ``fm_debug`` bundle, for the x-z plots.

Everything the 2-D figures need -- the draws' feet and base, the clip over the same nodes, the
history the generator conditioned on, the older served tail, the terrain step profile, and the
generator's own depth images -- resolved from a bundle and its checkpoint layout.

Deliberately free of viser and of matplotlib: the viser button and the keyboard browser are two
front-ends over this one assembly, and duplicating it is how their pictures would come to
disagree. The bundle-specific reasoning that lives here rather than in a plot module:

* a **draw** is a window in the plan's frame, so it needs ``window_to_world`` against that
  re-plan's anchor, and FK for the feet (the window predicts ``joint_pos``, never ``ee_pos``);
* the **history** is not a separate array -- under ``history_from="generator"`` it IS the last
  N served nodes, and under ``"clip"`` it is the clip's nodes at those steps, so which array to
  read depends on how the run was configured;
* the **clip** needs no re-anchoring at all: it advances one step per control step, so its
  horizon slice is ``y_des_clip[t : t+F]``, already in world.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import fm_resample as _fm_resample


class FeetPlotSource:
    """Per-bundle assembly of the x-z plot inputs.

    Args:
        data: A loaded ``<mode>.npz`` bundle.
        meta: Its metadata dict.
        mode: The bundle's mode name, for labels.
        depth_image_fn: ``(flat_row, spec) -> RGB`` renderer for the depth panels; omit to
            skip the depth images.
    """

    def __init__(self, data: dict, meta: dict, mode: str = "fm", depth_image_fn=None) -> None:
        self.data, self.meta, self.mode = data, meta, mode
        self._depth_image_fn = depth_image_fn
        self.pos_names = list(meta.get("pos_names", []))
        self.joint_names = list(meta.get("joint_names", []))
        self.frame_names = list(meta.get("fm_frame_names", []))
        self.base_frame = str(meta.get("base_frame_name", "pelvis_link"))
        self.blocks = meta.get("fm_reference_blocks")
        self.history_source = str(meta.get("fm_history_from") or "unknown-recapture-to-label")
        self.history_len = int(meta.get("fm_num_history") or 0)
        self.dt = float(meta.get("dt", 0.02))
        self.extero_spec = list(meta.get("fm_extero_spec") or [])
        self.foot_slots = [i for i, n in enumerate(self.frame_names) if "ankle" in n.lower()]
        self._served: dict[int, dict[str, np.ndarray] | None] = {}
        self.cols: dict[str, Any] = {"base": [], "feet": [], "quat": [], "joints": []}
        try:
            self.cols["base"] = [self.pos_names.index(f"{self.base_frame}:pos_{a}") for a in "xyz"]
            # Read the quaternion order by NAME: this layout is scalar-LAST while the clip
            # library's is scalar-FIRST, and a silent swap rotates every FK offset.
            self.cols["quat"] = [self.pos_names.index(f"{self.base_frame}:ori_{a}")
                                 for a in "xyzw"]
            self.cols["joints"] = [self.pos_names.index(f"joint:{n}") for n in self.joint_names]
            self.cols["feet"] = [[self.pos_names.index(f"{n}:pos_{a}") for a in "xyz"]
                                 for n in self.frame_names if "ankle" in n.lower()]
        except ValueError as exc:
            print(f"[fm_feet_data] missing a position channel ({exc}); some curves will be "
                  "omitted.")
        self.fk = (_fm_resample.make_fk_chain(self.joint_names, self.frame_names,
                                              base_link=self.base_frame)
                   if self.joint_names and self.frame_names else None)

    @property
    def has_draws(self) -> bool:
        """Whether this bundle carries the per-re-plan draw set."""
        return "fm_draws" in self.data

    def replans(self, env: int = 0) -> np.ndarray:
        """Steps at which the generator re-planned."""
        try:
            return _fm_resample.replan_steps(self.data, env)
        except SystemExit:
            return np.arange(np.asarray(self.data["y_des_fm"]).shape[0])

    def num_steps(self) -> int:
        """Rollout length."""
        return int(np.asarray(self.data["y_des_fm"]).shape[0])

    def served_trace(self, env: int) -> dict[str, np.ndarray] | None:
        """Root + feet of the reference actually served, per step (cached).

        Read from ``y_des_fm`` -- the APPLIED desired output, already in world. Its pelvis pose
        and joints are the served node; the ee frames there are NaN (no ``ee_pos`` block, and
        ``strict_fm_only`` poisons what the model did not write), hence FK for the feet.
        """
        if env in self._served:
            return self._served[env]
        arr = self.data.get("y_des_fm")
        if arr is None or not self.cols["base"]:
            self._served[env] = None
            return None
        import torch
        from isaaclab.utils.math import quat_apply

        y = np.asarray(arr[:, env])
        out = {"base": y[:, self.cols["base"]]}
        if self.fk is not None and self.cols["quat"] and self.cols["joints"] and self.foot_slots:
            q = torch.as_tensor(y[:, self.cols["joints"]], dtype=torch.float32)
            valid = torch.isfinite(q).all(dim=1)
            rel, _r, _jv, _jw = self.fk.pose_and_jacobian(torch.nan_to_num(q))
            k = rel.shape[1]
            quat = torch.as_tensor(y[:, self.cols["quat"]],
                                   dtype=torch.float32)[:, None, :].expand(-1, k, 4)
            base = torch.as_tensor(out["base"], dtype=torch.float32)[:, None, :]
            feet = base + quat_apply(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(-1, k, 3)
            out["feet"] = torch.where(valid[:, None, None], feet, torch.nan
                                      ).detach().cpu().numpy()[:, self.foot_slots]
        self._served[env] = out
        return out

    def terrain_profile(self, env: int) -> list[tuple[float, float, float]]:
        """World ``[(x0, x1, top)]`` step profile of this env's tile.

        Overlays are tile-local and the placement is a pure translation onto the tile corner,
        so world x is ``corner_x + local_x``. Every box contributes its own span, which is what
        makes a tread flush on a platform read as a step rather than vanish.
        """
        tids = np.asarray(self.meta.get("env_terrain_id") or []).reshape(-1)
        corners = np.asarray(self.meta.get("env_cell_corner") or [])
        if env >= tids.size or env >= corners.shape[0]:
            return []
        entry = next((m for m in (self.meta.get("terrain_manifest") or [])
                      if int(m.get("terrain_id", -999)) == int(tids[env])), None)
        if entry is None or "overlays" not in entry:
            return []
        cx0, cz0 = float(corners[env][0]), float(corners[env][2])
        segs = []
        for o in entry["overlays"]:
            c, s = float(o["pos_xy"][0]), float(o["size_xyz"][0])
            segs.append((cx0 + c - s / 2.0, cx0 + c + s / 2.0,
                         cz0 + float(o["base_z"]) + float(o["size_xyz"][2])))
        return sorted(segs)

    def clip_label(self, step: int, env: int) -> dict[str, object]:
        """Which clip this env is on and where in it -- ``clip_id``, ``clip_step``, ``label``.

        The clip step is READ from the recorded ``clip_step`` column, not derived from the
        rollout step: the capture starts at clip step 1, and an episode that terminated early
        restarts its clip cursor, so the two indices are neither equal nor even monotonically
        related. That makes the recorded value the only honest answer to "where in the clip am
        I", and a row whose clip step has jumped backwards is exactly how a reset shows up.
        """
        cid = self.data.get("clip_id")
        cst = self.data.get("clip_step")
        clip_id = int(np.asarray(cid[min(step, cid.shape[0] - 1), env])) if cid is not None else -1
        clip_step = int(np.asarray(cst[min(step, cst.shape[0] - 1), env])) if cst is not None else -1
        # The case entry names the SOURCE-library clip index and its true length; `clip_id`
        # above indexes the small debug subset, so they differ and both are worth showing.
        tids = np.asarray(self.meta.get("env_terrain_id") or []).reshape(-1)
        case = None
        if env < tids.size:
            case = next((c for c in (self.meta.get("cases") or [])
                         if int(c.get("terrain_id", -999)) == int(tids[env])), None)
        length = int(case["clip_len"]) if case else -1
        name = case["name"] if case else f"env {env}"
        src_idx = case["clip_index"] if case else clip_id
        label = (f"{name}: clip {src_idx}"
                 + (f" (tile {case['terrain_id']})" if case else "")
                 + f"  ·  clip step {clip_step}" + (f"/{length}" if length > 0 else ""))
        return {"clip_id": src_idx, "clip_step": clip_step, "clip_len": length,
                "case_name": name, "clip_label": label}

    def _lift(self, step: int, env: int) -> dict[str, np.ndarray] | None:
        """Draws at ``step`` lifted to world: ``{"base": [n,F,3], "ee": [n,F,K,3]}``."""
        win = self.data.get("fm_draws")
        if win is None:
            return None
        return _fm_resample.draws_to_world(
            np.asarray(win[step, env]),
            np.asarray(self.data["fm_cond_anchor_pos"][step, env]),
            np.asarray(self.data["fm_cond_anchor_quat"][step, env]),
            len(self.joint_names) or 29, len(self.frame_names) or 4,
            blocks=self.blocks, fk=self.fk)

    def inputs(self, step: int, env: int = 0, tail_steps: int = 50) -> dict | None:
        """Channels + depth + terrain for one ``(step, env)``.

        Returns:
            ``{"channels": [...], "depth_images": [...], "terrain_segs": [...], ...}`` where
            each channel is one x-z panel: ``name``, ``draws`` ``[n, F, 3]``, and the optional
            ``prev`` / ``clip`` / ``hist`` / ``tail`` curves. Feet channels come first, then the
            base. None when this bundle has no draws.
        """
        lifted = self._lift(step, env)
        if lifted is None:
            return None
        slots = self.foot_slots or list(range(lifted.get("ee", np.zeros((1, 1, 1, 3))).shape[2]))
        F = lifted["base"].shape[1]
        prev = None
        marks = self.replans(env)
        before = marks[marks < step]
        if before.size:
            prev = self._lift(int(before[-1]), env)

        # Clip over the same nodes, plus the history/tail traces from whichever source fed the
        # generator on this run.
        arr = self.data.get("y_des_clip")
        clip_feet = clip_base = None
        if arr is not None and self.cols["feet"]:
            end = min(step + F, arr.shape[0])
            blk = np.asarray(arr[step:end, env])
            clip_feet = np.stack([blk[:, c] for c in self.cols["feet"]], axis=1)
            clip_base = blk[:, self.cols["base"]]
        hist_feet = hist_base = tail_feet = tail_base = None
        lo = max(step - self.history_len, 0)
        if self.history_source == "clip" and arr is not None and self.cols["feet"]:
            blk = np.asarray(arr[lo:step, env])
            if blk.shape[0]:
                hist_feet = np.stack([blk[:, c] for c in self.cols["feet"]], axis=1)
                hist_base = blk[:, self.cols["base"]]
        else:
            tr = self.served_trace(env)
            if tr is not None:
                hist_base = tr["base"][lo:step]
                tail_base = tr["base"][max(step - tail_steps, 0):lo]
                if "feet" in tr:
                    hist_feet = tr["feet"][lo:step]
                    tail_feet = tr["feet"][max(step - tail_steps, 0):lo]

        channels = []
        for k, slot in enumerate(slots):
            channels.append({
                "name": self.frame_names[slot] if slot < len(self.frame_names) else f"ee{slot}",
                "draws": lifted["ee"][:, :, slot] if "ee" in lifted else None,
                "prev": prev["ee"][:, :, slot] if (prev and "ee" in prev) else None,
                "clip": clip_feet[:, k] if clip_feet is not None else None,
                "hist": hist_feet[:, k] if hist_feet is not None else None,
                "tail": tail_feet[:, k] if tail_feet is not None else None,
            })
        channels.append({
            "name": f"{self.base_frame} (base)", "draws": lifted["base"],
            "prev": prev["base"] if prev else None, "clip": clip_base,
            "hist": hist_base, "tail": tail_base,
        })
        images, labels = [], []
        extero = self.data.get("fm_cond_extero")
        if self._depth_image_fn is not None and extero is not None and self.extero_spec:
            row = np.asarray(extero[min(step, extero.shape[0] - 1), env])
            for i, spec in enumerate(self.extero_spec):
                if i < row.shape[0]:
                    images.append(self._depth_image_fn(row[i], spec))
                    labels.append(f"{spec.get('obs_term', i)} "
                                  f"{spec.get('width')}x{spec.get('height')}")
        scores = self.data.get("fm_scores")
        chosen = self.data.get("fm_chosen")
        return {
            **self.clip_label(step, env),
            "channels": [c for c in channels if c["draws"] is not None],
            "step": int(step), "dt": self.dt,
            "scores": None if scores is None else np.asarray(scores[step, env]),
            "chosen": None if chosen is None else int(np.asarray(chosen[step, env])),
            "terrain_segs": self.terrain_profile(env),
            "depth_images": images, "depth_labels": labels,
            "history_source": self.history_source, "mode": self.mode,
        }
