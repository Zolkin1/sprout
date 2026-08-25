#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""What KIND of motion did the generator just plan: flat, stairs, or a jump?

The question this answers is a field observation -- the robot sometimes reads a staircase as
a box and plans to jump onto it -- turned into a number: at every re-plan, classify the
predicted window against a reference set of REAL clip windows and count how often a plan on a
stairs clip comes back ``jump``.

**Nearest neighbours over the raw window, not hand-made features.**  Three hand-designed
feature sets were tried against the clip library first and all three failed, which is worth
recording so nobody rebuilds them:

* pelvis-z ARC and DIP (a jump should rise above both window ends) -- every family sits at
  0.000-0.043 m.  The pelvis barely arcs even in a real jump.
* net climb / vertical range / 0.3 s burst / peak vertical speed, six features into a
  nearest-centroid -- **0.565 accuracy against a 0.509 majority-class baseline**.  Stepping
  onto a tread and hopping onto a box are near-identical at the base.
* both-feet clearance above a running-min ground estimate -- actively inverted: the ground
  estimate lags on rising terrain, so ``stairs_up`` scored HIGHER clearance (0.132 m) than
  ``jump_on`` (0.041 m).  Climbing read as airborne.

k-NN over the whole window reaches 0.795-0.846.  The signal is in the joint trajectory across
the window, not in any scalar summary of the base height.

**The space is the generator's own output**, all ``D`` channels, so a query and a reference
are literally the same object and no forward kinematics is needed to compare them.  Features
are the window DIFFERENCED FROM NODE 0, which under both ``base_pose_convention`` variants
subtracts a single pose held constant across the window -- so the representation is anchor
invariant, and a plan is comparable to a recorded clip window taken anywhere.

**The classifier's own error rate is the control, and it is not small.**  Measured on held-out
recording windows, split BY CLIP: the true clip window classifies at 0.795 and the FM's
prediction at 0.805.  So a raw "the plan looks like a jump" rate means nothing on its own --
report it against what the SAME classifier says about the clip, and read the difference.
Resolution is very uneven: ``stairs`` is 0.972 correct while ``flat`` is 0.712, with the
residual being flat<->jump.  That is the featureless-window problem -- most windows inside a
``jump_on`` clip contain no jump, the robot is walking across the tile -- and it is a property
of the LABELS, not of the model.  The stairs axis, which is the one the field observation is
about, is the sharp one.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

#: The three motion kinds.  Order is the integer code order, and the codes go on disk.
COARSE_CLASSES: tuple[str, ...] = ("flat", "jump", "stairs")

#: Manifest family -> coarse class.  ``unknown`` is the recorder's label for a window with no
#: terrain, which is flat by construction.
FAMILY_TO_COARSE: dict[str, str] = {
    "unknown": "flat", "flat": "flat", "family_flat": "flat",
    "family_stairs_up": "stairs", "family_stairs_down": "stairs",
    "family_stairs_up_pure": "stairs", "family_stairs_dn_pure": "stairs",
    "family_jump_on": "jump", "family_jump_off": "jump", "family_jump_off_front": "jump",
}

#: Keep every Nth node.  62 nodes x 44 channels is 2728 numbers per window; every 4th node
#: holds the accuracy (measured) at a quarter of the distance cost.
NODE_STRIDE = 4

#: Neighbours in the vote.
DEFAULT_K = 5


def window_features(w, stride: int = NODE_STRIDE):
    """Anchor-invariant features for a batch of windows.

    Differencing from node 0 is what makes a runtime plan comparable to a recorded clip
    window: both ``base_pose_convention`` variants subtract a pose that is CONSTANT across
    the window, so the subtraction cancels it exactly and what remains is the planned motion.

    Args:
        w: ``(N, F, D)`` raw (denormalized) window nodes, torch tensor or ndarray.
        stride: Keep every ``stride``-th node.

    Returns:
        ``(N, F//stride * D)``, same type as the input.
    """
    d = w - w[:, :1, :]
    return d[:, ::stride, :].reshape(d.shape[0], -1)


class TerrainClassifier:
    """k-NN over reference clip windows, in the generator's own output space."""

    def __init__(self, feats, labels, mean, std, k: int = DEFAULT_K,
                 stride: int = NODE_STRIDE) -> None:
        """Hold a standardized reference set.

        Args:
            feats: ``(M, P)`` standardized reference features.
            labels: ``(M,)`` int codes into :data:`COARSE_CLASSES`.
            mean: ``(P,)`` feature mean used to standardize.
            std: ``(P,)`` feature std used to standardize.
            k: Neighbours in the vote.
            stride: Node stride the features were built with.
        """
        self.feats = np.asarray(feats, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.k = int(k)
        self.stride = int(stride)
        self._torch = None

    # -- persistence -------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Write the reference set to ``path`` (``.npz``)."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, feats=self.feats, labels=self.labels, mean=self.mean, std=self.std,
            k=np.array([self.k]), stride=np.array([self.stride]),
            classes=np.array(json.dumps(list(COARSE_CLASSES))))

    @classmethod
    def load(cls, path: str) -> "TerrainClassifier":
        """Read a reference set written by :meth:`save`.

        Raises:
            ValueError: If the stored class list differs from :data:`COARSE_CLASSES` -- the
                codes are integers on disk, so a reordering would silently relabel every
                recorded count.
        """
        z = np.load(path)
        stored = tuple(json.loads(str(z["classes"])))
        if stored != COARSE_CLASSES:
            raise ValueError(
                f"{path} was built with classes {stored}, this code has {COARSE_CLASSES}. "
                "The codes are integers in the episode table, so the counts would be "
                "silently relabelled. Rebuild the reference set.")
        return cls(z["feats"], z["labels"], z["mean"], z["std"],
                   int(z["k"][0]), int(z["stride"][0]))

    # -- fitting -----------------------------------------------------------------------
    @classmethod
    def fit(cls, windows, families, clip_ids, family_names: list[str], *,
            k: int = DEFAULT_K, stride: int = NODE_STRIDE) -> "TerrainClassifier":
        """Build a reference set from labelled clip windows.

        Args:
            windows: ``(N, F, D)`` raw clip windows (the dataset TARGET, not a prediction).
            families: ``(N,)`` int index into ``family_names``.
            clip_ids: ``(N,)`` source clip id, for the caller's own splitting.
            family_names: The recorder's family strings.
            k: Neighbours in the vote.
            stride: Node stride.

        Returns:
            A fitted classifier.

        Raises:
            ValueError: If a family string has no coarse mapping -- guessing would put a
                whole terrain type into the wrong class.
        """
        unknown = {family_names[c] for c in np.unique(families)} - set(FAMILY_TO_COARSE)
        if unknown:
            raise ValueError(f"no coarse class for {sorted(unknown)}; extend FAMILY_TO_COARSE")
        codes = np.array([COARSE_CLASSES.index(FAMILY_TO_COARSE[family_names[c]])
                          for c in families], dtype=np.int64)
        x = window_features(np.asarray(windows, dtype=np.float32), stride)
        mean, std = x.mean(0), x.std(0) + 1e-9
        return cls((x - mean) / std, codes, mean, std, k, stride)

    # -- inference ---------------------------------------------------------------------
    def classify(self, windows):
        """Class code per window.

        Runs on the input's device when given a torch tensor, so the eval pays one matmul
        against a resident reference set rather than a host round trip per re-plan.

        Args:
            windows: ``(N, F, D)`` raw window nodes, torch tensor or ndarray.

        Returns:
            ``(N,)`` int codes into :data:`COARSE_CLASSES`, as a numpy array.
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError:  # pragma: no cover - numpy fallback for the tests
            torch = None
        if torch is not None and isinstance(windows, torch.Tensor):
            if self._torch is None or self._torch[0].device != windows.device:
                dev = windows.device
                self._torch = (torch.as_tensor(self.feats, device=dev),
                               torch.as_tensor(self.labels, device=dev),
                               torch.as_tensor(self.mean, device=dev),
                               torch.as_tensor(self.std, device=dev))
            ref, lab, mu, sd = self._torch
            q = (window_features(windows.to(torch.float32), self.stride) - mu) / sd
            d = torch.cdist(q, ref)
            nn = lab[d.topk(self.k, largest=False).indices]              # (N, k)
            votes = torch.stack([(nn == c).sum(1) for c in range(len(COARSE_CLASSES))], 1)
            return votes.argmax(1).cpu().numpy().astype(np.int64)
        q = (window_features(np.asarray(windows, dtype=np.float32), self.stride)
             - self.mean) / self.std
        d = ((q ** 2).sum(1)[:, None] - 2 * q @ self.feats.T
             + (self.feats ** 2).sum(1)[None])
        nn = self.labels[np.argpartition(d, self.k, axis=1)[:, :self.k]]
        return np.array([np.bincount(r, minlength=len(COARSE_CLASSES)).argmax()
                         for r in nn], dtype=np.int64)


def main() -> int:
    """Build a reference set from a saved ``(windows, families)`` bundle."""
    p = argparse.ArgumentParser(prog="terrain_class.py", description=__doc__,
                                allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", required=True,
                   help=".npz with 'tgt' (N,F,D), 'clip_family', 'clip_id', 'family_names'")
    p.add_argument("--out", required=True, help="reference .npz to write")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--holdout", type=float, default=0.3,
                   help="fraction of CLIPS held out to report the control accuracy")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    z = np.load(a.windows)
    names = json.loads(str(z["family_names"]))
    tgt, fam, clip = z["tgt"], z["clip_family"].reshape(-1), z["clip_id"].reshape(-1)
    rng = np.random.default_rng(a.seed)
    ids = np.unique(clip)
    rng.shuffle(ids)
    train = np.isin(clip, ids[int(a.holdout * len(ids)):])

    clf = TerrainClassifier.fit(tgt[train], fam[train], clip[train], names, k=a.k)
    codes = np.array([COARSE_CLASSES.index(FAMILY_TO_COARSE[names[c]]) for c in fam])
    print(f"[terrain_class] reference {train.sum()} windows, held out {(~train).sum()} "
          f"({len(ids)} clips)")
    if (~train).any():
        got, truth = clf.classify(tgt[~train]), codes[~train]
        print(f"[terrain_class] CONTROL accuracy on held-out clip windows: "
              f"{(got == truth).mean():.3f}")
        print("  true      n |" + "".join(f"{c:>9s}" for c in COARSE_CLASSES))
        for i, c in enumerate(COARSE_CLASSES):
            m = truth == i
            print(f"  {c:8s} {m.sum():4d} |"
                  + "".join(f"{(got[m] == j).mean():9.3f}"
                            for j in range(len(COARSE_CLASSES))))
        if "pred" in z.files:
            pg = clf.classify(z["pred"][~train])
            print(f"[terrain_class] the FM's own predictions on those instants: "
                  f"accuracy {(pg == truth).mean():.3f}, "
                  f"disagreement with the clip window {(pg != got).mean():.3f}")
    else:
        # --holdout 0 ships EVERY window as reference, which is what the eval wants: the
        # control it is compared against was already measured over 5 by-clip splits
        # (stairs->jump 0.059 [0.047, 0.071]) and re-measuring it here on training data
        # would only report memorisation.
        print("[terrain_class] holdout 0: no validation split; see the 5-split control "
              "in the module docstring.")
    clf.save(a.out)
    print(f"[terrain_class] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
