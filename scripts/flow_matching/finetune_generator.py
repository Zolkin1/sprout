# Copyright 2026 Zachary Olkin. All rights reserved.

"""Fine-tune a trained flow-matching generator on its own rollouts.

**AWR AGAINST THE TERRAIN-CONSISTENCY REWARD IS THE FIRST ARM HERE THAT PRODUCED A REAL
IMPROVEMENT** (measured 2026-09-08, Isaac box2, paired on ``(distance, speed)`` with a
tile-clustered interval).  Earlier attempts targeted the TRACKING reward and were stopped on a
measured causal ceiling -- the chunk choice is only 1.6% of that reward's variance, and
filtered BC had already taken half of it.  Changing the TARGET is what changed the outcome:

    v1 course, 30k episodes    success 0.1077 -> 0.1618   +0.054   4/4 cells
    v1 course, 1.9k episodes   success 0.1077 -> 0.1578   +0.050   6/6 cells
    v2 course, perturbed data  success 0.1421 -> 0.2650   +0.125  30/30 cells

and the reward it optimised moved with it: at matched exposure the fraction of re-plans
putting a body 3 cm or more into the terrain fell 0.113 -> 0.076, and the penetration term
38% (0.808 -> 0.497), improving at every one of the six commanded speeds.

**The largest gain came from the arm trained on PERTURBED data** (`fm_upper_camera_noise_*`
in the FM command cfg: the generator is conditioned on a dead forward camera for a fraction of
re-plans while the recorder stores the true measurement, widening the distribution of states
and plans a run visits).  That arm nearly tripled the gain, and its advantage was
correspondingly wider (sd 0.117 against 0.088).  **It is not an isolated result**: that run
also moved to the v2 course, which adds 3 m of run-out past the second box, so the two changes
are confounded and the perturbation's own contribution is unmeasured.  Separating them needs a
v2 dataset collected without perturbations.  Treat this as a promising start on the method
rather than a finished attribution.

Two things that were load-bearing and are easy to get wrong.  The per-re-plan penalty MUST be
compared at matched exposure -- a better policy survives longer, so its re-plans come from
deeper, harder course, and the unmatched pooled average reads WORSE while every phase-matched
window reads better (it inverted the conclusion three times).  And ``best.pt`` is the wrong
artifact to evaluate: it is selected on UNWEIGHTED ``val_flow_mse``, which AWR moves away from
by design, so it favours the checkpoint that changed least -- score ``ckpt_*`` by the terrain
reward instead.

Four modes, differing only in WHERE the "which chunks are good" judgement enters.  All four
train the model to reproduce chunks it emitted itself, using the SAME supervised
flow-matching objective it was pretrained with, and only AWR touches the loss (as a
per-sample weight; the objective is unchanged) -- none touches the sampler:

* **advantage-conditioned** (``--labels``) -- every chunk is kept and its label
  (``label_advantage.py``) is fed to the model as conditioning.  At inference the label is
  pinned ``positive``, which is the whole intervention.  This is the only mode that changes
  the ARCHITECTURE: it adds a label embedding and produces per-arm serve checkpoints.
* **label-filtered BC** (``--labels --filter_label positive``) -- the label SELECTS rows and
  is then discarded, so the model stays exactly as pretrained.
* **filtered BC on a pre-filtered rollout** (``--unconditioned``) -- the selection already
  happened upstream (e.g. ``merge_selected.py``: the critic chose the chunk AND its episode
  succeeded), so there is no sidecar to point at and nothing to condition on.
* **AWR** (``--labels --awr_beta B``) -- every LABELLED chunk is kept and its loss scaled by
  ``exp(A / beta)``, so the judgement is a continuous weight instead of a hard keep/drop.
  Plain architecture, like the filtered arms.  It is a strict generalisation of both
  neighbours rather than a fourth unrelated idea, which is what makes a ``beta`` sweep worth
  running: as ``beta -> inf`` the weights go uniform and AWR IS unweighted BC over every
  labelled row.  The reported **ESS** is what says where on that line a given run actually
  sat, and it is not inferable from ``beta`` alone.

  **``--awr_normalize global`` is the DEFAULT, and it is the classic AWR estimator.**  The
  weights are scaled once to global mean 1 and the loss takes a plain ``mean(w * mse_i)``, so
  a batch of low-advantage rows genuinely contributes a smaller step and the sharp limit IS
  filtered BC at a shrinking percentile.

  ``batch`` is the self-normalized alternative -- ``sum(w * mse_i) / sum(w)``, dividing by the
  BATCH'S OWN weight sum -- and it is kept as the lower-variance arm.  Know what it costs
  before choosing it: rescaling per batch means every batch contributes a full-size step
  however small its weights are, so a batch of mediocre rows lands the same total gradient as
  one holding a genuinely high-advantage row, and the advantage signal is normalized away
  ACROSS batches while surviving only within them.  Its sharp limit is therefore
  BEST-OF-BATCH, not filtered BC, and the realized effective sample size floors at about
  ``n / (batch / 2)``.  Measured at n = 20k / B = 128 (the old default), the reported ESS falls
  346 -> 17 -> 5 over beta 0.25 / 0.10 / 0.05 while the realized one holds at 580 / 327 / 312
  -- so below beta ~0.1 the printed number keeps collapsing and the update stops changing.
  At the current B = 256 that floor is twice as high again.

  The default changed from ``batch`` on 2026-09-07.  Every AWR run before that date is a
  ``batch`` run; the arm is recorded in the checkpoint's own ``awr`` block, so which one a
  result came from is recoverable rather than inferred from its date.

  **The advantage is centered within the cell the label was cut in** (``--awr_center
  stratum``, the default).  ``binarize`` takes its percentile INSIDE ``family x progress x
  speed`` precisely so the keep-set is free of the between-cell offset; a weight built on
  the raw advantage puts that offset straight back, upweighting whole speed bins instead of
  chunks within them.  Measured on a synthetic two-cell case with a 3-sigma offset and equal
  spreads, the high cell takes 84.1% of all weight uncentered against 49.9% centered.

Three properties make this safe to run against a shipped checkpoint:

* **the conversion is bit-identical.**  The label embedding is zero-initialised, so before
  the first gradient step the converted model reproduces the original exactly, under every
  label.  ``--verify_identity`` asserts it rather than assuming it.
* **the normalizers come from the pretrained checkpoint**, never refitted.  A refit would
  change what every weight means.
* **it always fine-tunes from the PRETRAINED checkpoint**, never from the previous
  iteration -- so iteration k is one well-defined step from a known model rather than the
  end of an unrepeatable chain.

**What decides whether it worked differs by mode, and only one of them has an internal
control.**  Advantage-conditioned runs have the NEGATIVE-conditioning control: sample the
same checkpoint with the label pinned negative, and if it performs the same as positive the
model is ignoring the label and every other number is noise.  Run it before any sweep.  The
two filtered-BC modes have no arm to control against -- the model IS the pretrained
architecture -- so the closed-loop eval against the base generator is the only check, which
is why ``--canary_cache`` exists: four cheap scalars per epoch to say WHERE to spend it.

**None of the training losses selects the checkpoint you want.**  Train loss falls
monotonically, and ``--val_frac``'s held-out split is drawn from the same narrow slice as
train, so it mostly detects memorisation.  The closed-loop number peaks and decays where no
loss here says it does -- hence ``--ckpt_every``, without which the interior epochs are gone
and recovering one means retraining the arm.

Usage::

    # advantage-conditioned
    python scripts/flow_matching/finetune_generator.py \\
        --pretrained logs/flow_matching/<run>/best.pt \\
        --rollout logs/rollouts/v2/rollout_<...>.h5 \\
        --labels logs/value/v2/advantage.npz \\
        --epochs 6 --out logs/flow_matching/<run>_ac_v1

    # filtered BC on an already-filtered rollout (merge_selected.py --successful_only)
    python scripts/flow_matching/finetune_generator.py \\
        --pretrained logs/flow_matching/<run>/best.pt \\
        --rollout logs/value/bc_data_v1/train_success.h5 --unconditioned \\
        --epochs 20 --lr 3e-5 --lr_warmup_steps 100 --weight_decay 0 \\
        --ckpt_every 2 --canary_cache logs/value/canary_v1.pt \\
        --out logs/flow_matching/<run>_bc_v1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "source", "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_rl.flow_matching import online  # noqa: E402
from robot_rl.flow_matching.advantage import (  # noqa: E402
    LABEL_NAMES, LABEL_NULL, label_from_name,
)
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.ema import WeightEMA  # noqa: E402
from robot_rl.flow_matching.flow import flow_matching_loss, sample_trajectory  # noqa: E402
from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: E402
from robot_rl.flow_matching.critic_eval import (  # noqa: E402
    effective_sample_size, stratified_awr_weights,
)
from robot_rl.flow_matching.rollout_dataset import (  # noqa: E402
    RolloutWindowDataset, join_labels, join_values,
)

#: The state-dict keys the conversion adds, per injection mode.  Anything else missing means
#: the pretrained checkpoint is not the architecture this script thinks it is.
_NEW_KEYS = {
    "time_embed": {"advantage_emb.weight"},
    # The token variant also carries its own type embedding, exactly as every other stream in
    # the prefix does.
    "prefix_token": {"advantage_emb.weight", "advantage_type"},
}


def is_plain_arm(args) -> bool:
    """Whether this run trains the PLAIN architecture -- no advantage embedding.

    **One definition, called twice.**  ``main`` uses it to decide what to BUILD and ``_save``
    to decide what to STAMP, and those two must agree or the artifact describes an
    architecture its own weights do not have: ``load_fm_context`` then builds the conditioned
    model and ``load_state_dict`` fails on a missing ``advantage_emb.weight``, at load time,
    in whatever consumer picks the checkpoint up.

    That is not hypothetical -- ``_save`` used to RESTATE the rule "so this stamp can never
    disagree with the architecture main built", which made it a second copy that then went
    out of date the moment AWR was added as a third plain arm.  Every AWR checkpoint written
    before this was stamped ``advantage_cond=True`` over plain weights and could not be
    loaded at all.

    Args:
        args: The parsed command line.

    Returns:
        True for the label-filtered, unconditioned and AWR arms; False for the
        advantage-conditioned one, which is the only arm that grows an embedding.
    """
    return (bool(args.filter_label) or bool(args.unconditioned)
            or float(args.awr_beta) > 0.0)


def _rollout_history_from(path: str) -> str:
    """The history arm a rollout recording was made under, or ``""`` if it does not say.

    ``record_rollouts`` stamps ``history_from`` into the window table's attributes. A file
    written before that stamp existed is definitionally the ``generator`` arm -- the measured
    one did not exist -- and maps to the ``reference`` history source; ``""`` is returned for
    it instead so the caller leaves the parent's value alone rather than asserting one.

    Args:
        path: The rollout window ``.h5``.

    Returns:
        ``"measured"``, ``"reference"``, or ``""`` when unstamped or unreadable.
    """
    import h5py  # noqa: PLC0415 -- only needed here, and the module is heavy

    try:
        with h5py.File(path, "r") as f:
            arm = f.attrs.get("history_from")
    except OSError:
        return ""
    if arm is None:
        return ""
    arm = arm.decode() if isinstance(arm, bytes) else str(arm)
    arm = arm.strip().strip('"')
    # The window ARM maps onto the checkpoint's own SOURCE contract: a served node and a
    # clip node are both applied references, so only "measured" is its own thing.
    return "measured" if arm == "measured" else "reference"


def resolved_config(path: str) -> FlowMatchingConfig:
    """The checkpoint's config, with ``extero_encoder`` resolved the way the loader does.

    ``FmOnlineContext`` does not carry the config, and ``"auto"`` must be settled before
    ``from_config`` builds anything -- so this mirrors ``online.load_fm_context``'s own two
    lines rather than re-deriving the architecture from the weights.

    Args:
        path: A generator ``.pt``.

    Returns:
        The config, encoder resolved.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    cfg.extero_encoder = cfg.resolve_extero_encoder()
    return cfg


def convert(ctx, cfg, device, inject: str | None = None) -> FlowMatchingTransformer:
    """Build the advantage-conditioned twin of a loaded checkpoint's model.

    Args:
        ctx: A loaded :class:`~robot_rl.flow_matching.online.FmOnlineContext`.
        cfg: That checkpoint's resolved config.
        device: Where to put it.
        inject: Override the injection mode ('time_embed' / 'prefix_token').

    Returns:
        A trainable :class:`FlowMatchingTransformer` with ``advantage_cond=True``, carrying
        the pretrained weights and a zero label embedding.

    Raises:
        RuntimeError: If the load is missing anything other than the label embedding, or
            carries an unexpected key.  Either would mean weights silently did not transfer.
    """
    cfg = copy.deepcopy(cfg)
    cfg.advantage_cond = True
    if inject is not None:
        cfg.advantage_inject = inject
    model = FlowMatchingTransformer.from_config(cfg).to(device)
    missing, unexpected = model.load_state_dict(ctx.model.state_dict(), strict=False)
    want = _NEW_KEYS[cfg.advantage_inject]
    if set(missing) != want or unexpected:
        raise RuntimeError(
            f"converting to advantage_cond ({cfg.advantage_inject}) should add exactly "
            f"{sorted(want)}; got missing={sorted(missing)} unexpected={sorted(unexpected)}. "
            "The pretrained checkpoint is not the architecture this config describes.")
    return model


@torch.no_grad()
def verify_identity(ctx, model, ds, device, n: int = 4) -> float:
    """Assert the freshly-converted model reproduces the original under EVERY label.

    Zero-init makes the two models mathematically identical, so any difference is float
    noise rather than a statistical near-miss.  It is not bit-identical on a GPU -- two
    module instances with the same weights can still pick different cuBLAS algorithms --
    which is why the tolerance is 1e-5 against a measured ~4e-6 on the shipped 27M model,
    four orders below the ~1e-2 the label is expected to move things once trained.  Checked
    here against REAL conditioning as well as in the unit tests, which build a toy model.

    Args:
        ctx: The loaded context (its ``model`` is the original).
        model: The converted model.
        ds: A dataset to draw conditioning from.
        device: Device.
        n: Number of windows to check.

    Returns:
        The largest absolute difference seen.

    Raises:
        RuntimeError: If any label changes the output.
    """
    items = [ds[i] for i in range(min(n, len(ds)))]
    hist = torch.stack([it["history"] for it in items]).to(device)
    ext = torch.stack([it["extero"] for it in items]).to(device)
    cmd = (torch.stack([it["cmd_tokens"] for it in items]).to(device)
           if "cmd_tokens" in items[0] else None)
    null = torch.stack([it["history_null"] for it in items]).to(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(len(items), ctx.F, ctx.D, generator=gen).to(device)
    base = sample_trajectory(ctx.model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                             history_null_mask=null, noise=noise)
    worst = 0.0
    for label in range(len(LABEL_NAMES)):
        adv = torch.full((len(items),), label, dtype=torch.long, device=device)
        got = sample_trajectory(model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                                history_null_mask=null, noise=noise, advantage=adv)
        d = float((got - base).abs().max())
        worst = max(worst, d)
        if d > 1e-5:
            raise RuntimeError(
                f"the converted model is NOT identical to its parent under label "
                f"'{LABEL_NAMES[label]}': max |delta| = {d:.3e}. The embedding must be "
                "zero-initialised, or the conversion moved something else.")
    return worst


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of dataset items, keeping the optional keys optional."""
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    return out


def _split_rows(uid: np.ndarray, rows: np.ndarray, val_frac: float, seed: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by EPISODE, so near-duplicate windows do not straddle the split.

    Args:
        uid: ``[W]`` episode id per FILE row.
        rows: Candidate rows.
        val_frac: Fraction of episodes held out.
        seed: RNG seed.

    Returns:
        ``(train_rows, val_rows)``.
    """
    eps = np.unique(uid[rows])
    rng = np.random.default_rng(seed)
    n = max(1, int(round(val_frac * eps.size))) if val_frac > 0 else 0
    held = set(rng.choice(eps, size=min(n, eps.size), replace=False).tolist()) if n else set()
    is_val = np.array([int(u) in held for u in uid[rows]])
    return rows[~is_val], rows[is_val]


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained", required=True, help="The generator .pt to fine-tune.")
    p.add_argument("--rollout", required=True, help="Rollout window .h5.")
    p.add_argument("--labels", default="",
                   help="advantage.npz from label_advantage.py. Omit for an UNCONDITIONED "
                        "self-distillation control -- the same data and the same number of "
                        "steps with the label carrying nothing, which is what separates 'the "
                        "label helped' from 'more training on rollout data helped'.")
    p.add_argument("--out", required=True, help="Output run directory.")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Rows per optimizer step. There is no gradient accumulation -- one "
                        "mini-batch IS one step -- so this is the whole averaging window. "
                        "Raised 128 -> 256 alongside the `global` normalization default: "
                        "under `global` a step's magnitude varies with the advantage mass "
                        "that batch happened to draw, and a wider batch is what pays for "
                        "that variance. Under `batch` it also raises the realized ESS floor "
                        "(~n/(batch/2)), which is the opposite of what a sharp beta wants.")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Deliberately well below the pretraining 3e-4: this is a fine-tune "
                        "from a shipped model, and the failure mode is losing motion quality "
                        "that no success rate will show. Applies to every parameter EXCEPT "
                        "advantage_emb when --emb_lr_multiplier != 1.")
    p.add_argument("--advantage_inject", default="time_embed",
                   choices=("time_embed", "prefix_token"),
                   help="HOW the label reaches the network. `time_embed` adds it to the "
                        "flow-time embedding (zero-init, conversion exactly inert, ONNX "
                        "shapes unchanged) and is MEASURED INADEQUATE here -- across 1x/50x "
                        "embedding lr and a 10x whole-model lr it moved the served window by "
                        "0.3-0.7%% of ordinary sampling noise. `prefix_token` gives the label "
                        "its own token in the conditioning prefix, with its own attention "
                        "weights per layer, so it can modulate WHICH conditioning is attended "
                        "to; it grows prefix_len by 1, so its exported graph is not "
                        "interchangeable with a time_embed one.")
    p.add_argument("--emb_lr_multiplier", type=float, default=1.0,
                   help="advantage_emb.weight gets lr * this, every other parameter gets "
                        "lr unchanged -- a separate AdamW param group, not a global scale. "
                        "MOTIVATION, measured on the v2/n=12 run: the embedding is the only "
                        "parameter starting at exact zero (everything else is pretrained and "
                        "converged), and at multiplier=1 it moved to a row norm of ~0.03-0.04 "
                        "against the ~17-34 scale of the time_mlp output it is summed with -- "
                        "three orders of magnitude too small to matter, while flow_mse barely "
                        "moved. Raising the WHOLE model's lr to compensate would move the "
                        "converged 26.8M-param bulk just as fast, which is the motion-quality "
                        "risk --lr is conservative to avoid. A multiplier here targets only "
                        "the parameter that actually needs a bigger step. Try ~100-300x as a "
                        "starting point (an extrapolation, not a derivation) and re-check the "
                        "embedding norm against the target scale before trusting a full run.")
    p.add_argument("--cosine", action="store_true",
                   help="Anneal the LR from --lr to 0 over the run, cosine-shaped and applied "
                        "per EPOCH. Lets a run start at a peak that would be destructive to "
                        "hold: the model moves early and settles late, instead of taking "
                        "same-size steps on an already-converged checkpoint all the way "
                        "through. Reproduces CosineAnnealingLR(T_max=epochs, eta_min=0) at "
                        "epoch granularity, the same closed form scripts/flow_matching/"
                        "train.py uses.")
    p.add_argument("--lr_warmup_steps", type=int, default=0,
                   help="Linearly ramp to --lr over this many optimizer steps. Worth having "
                        "with a high peak: the first batch against a converged checkpoint is "
                        "the one most able to wreck it, and Adam's second-moment estimate is "
                        "still cold there.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--null_prob", type=float, default=0.3,
                   help="Per-access probability of overriding the label with null, so "
                        "classifier-free guidance over the label is available at inference.")
    p.add_argument("--val_frac", type=float, default=0.05,
                   help="Fraction of ROLLOUT episodes held out. Detects memorisation of the "
                        "training episodes and nothing more: train and val here are both the "
                        "same narrow slice (one generator's selected chunks, a few tiles, a "
                        "few speeds), so a model forgetting everything outside it still "
                        "shows this loss falling. --canary_cache is the metric that sees "
                        "that.")
    p.add_argument("--pair_cache", default="",
                   help="A pair cache from ft_canary.py --build --pairs. Logs the paired "
                        "SELECTED-vs-REJECTED flow-loss gap per epoch, over three groups "
                        "(train / heldout / failed-episode). DIAGNOSTIC ONLY -- it never "
                        "selects a checkpoint. It separates 'the model learned the selector's "
                        "ranking' from 'self-distillation improved every sample together', "
                        "which no loss on the selected chunks alone can distinguish. Read the "
                        "ABSOLUTE levels beside the gap: a gap that opens because rej_loss "
                        "ROSE is collapse onto the selected slice, not learning.")
    p.add_argument("--canary_cache", default="",
                   help="A cache from ft_canary.py --build. Adds two drift metrics per "
                        "epoch, each against a FROZEN batch so an LR sweep's arms are "
                        "comparable: pretrain_val_flow_mse (the pretraining objective on the "
                        "PRETRAINING recordings' own val split -- rising means the model is "
                        "leaving the distribution it was fitted on) and drift_l2 (distance "
                        "from the pretrained model's sampled plan on fixed conditioning, in "
                        "the space the tracker consumes). Both are also logged on the EMA "
                        "weights. Epoch 0 records the pretrained baseline, so later epochs "
                        "read as a move rather than a bare value.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_windows", type=int, default=0,
                   help="Cap the number of windows used, for a smoke test.")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--filter_label", default="", choices=("", "positive", "negative", "null"),
                   help="FILTERED BC: keep only windows carrying this label and train "
                        "UNCONDITIONED on them -- no label embedding, no token, no serve arm. "
                        "The simplest possible version of the idea: if the good chunks are "
                        "distinguishable at all, cloning only those should move the policy on "
                        "its OWN training distribution. It is the control that separates 'the "
                        "label never reaches the model' (which conditioning arms cannot rule "
                        "out) from 'this fine-tuning setup moves nothing', because here there "
                        "is no label pathway left to blame. Requires --labels for the sidecar; "
                        "the model architecture is left exactly as the pretrained one.")
    p.add_argument("--ckpt_every", type=int, default=2,
                   help="Also write ckpt_<epoch>.pt every N epochs. DEFAULT 2, because the "
                        "checkpoint that matters here is chosen by something OTHER than the "
                        "training losses: a closed-loop eval peaks and decays where no loss "
                        "in this script says it does, and last.pt (overwritten every epoch) "
                        "plus best.pt (val_flow_mse, which on an already-filtered set "
                        "declines about monotonically) preserve about ONE point on the "
                        "curve between them. Recovering an interior epoch afterwards means "
                        "retraining the arm, so the default keeps them rather than making "
                        "that the thing you had to remember. 0 turns it off.")
    p.add_argument("--awr_beta", type=float, default=0.0,
                   help="ADVANTAGE-WEIGHTED REGRESSION. Every labelled chunk is kept and its "
                        "loss is scaled by exp(A/beta), instead of the top-percentile subset "
                        "being kept at weight 1 and the rest dropped (--filter_label). "
                        "Smaller beta is sharper; in the hot limit AWR IS unweighted BC on "
                        "every labelled row, so a beta sweep brackets both baselines. Needs "
                        "--labels for the sidecar's continuous `advantage` column, and uses "
                        "the PLAIN architecture like --filter_label -- the advantage steers "
                        "the update, it is not fed to the model. 0 (default) is off.")
    p.add_argument("--awr_clip", type=float, default=20.0,
                   help="Clip each weight at this multiple of the MEAN weight, or 0 for no "
                        "clip. Without it one row can own an update at a sharp beta. Read it "
                        "beside the reported ESS: the clip is what stops the collapse, and "
                        "how much mass it removes is itself a statement about the tail.")
    p.add_argument("--awr_normalize", default="global", choices=("global", "batch"),
                   help="How the weights are normalized, which decides what the SHARP limit "
                        "of AWR is. `global` (DEFAULT, classic AWR) scales the weights to "
                        "global mean 1 and takes a plain mean, so a low-advantage batch "
                        "genuinely contributes less and the sharp limit IS filtered BC. "
                        "`batch` self-normalizes per batch, which rescales every batch to a "
                        "full-size step -- a batch of mediocre rows then lands the same total "
                        "gradient as one holding a high-advantage row, beta -> 0 converges to "
                        "best-of-BATCH, and the ESS floors at ~n/(batch/2). `batch` has lower "
                        "gradient variance and is the reason it is kept.")
    p.add_argument("--awr_balance", default="cell", choices=("cell", "none"),
                   help="Rescale each label CELL's total weight to its share of the rows. "
                        "`cell` (default) is what makes AWR's composition match filtered "
                        "BC's, which is the arm it is compared against: binarize keeps 30%% "
                        "of EVERY cell so its composition equals the data's exactly, while "
                        "centering alone leaves a wider-spread cell capturing more weight "
                        "through its upper tail -- measured max deviation from the row "
                        "share 0.29 points for filtered BC, 2.54 for centered AWR, ~0 with "
                        "this on. `none` reproduces the un-balanced arm. Within-cell "
                        "ranking is untouched either way; the trade is that `w_max` becomes "
                        "a per-cell bound rather than a global one (~6%% overshoot).")
    p.add_argument("--awr_center", default="stratum", choices=("stratum", "none"),
                   help="Center the advantage within the CELL label_advantage cut in "
                        "(family x progress x speed) before exponentiating. `stratum` (the "
                        "default) is what keeps AWR's selection comparable to filtered BC's: "
                        "the per-cell percentile threshold makes the label free of the "
                        "between-cell offset, and a weight built on the RAW advantage puts "
                        "that offset straight back -- upweighting whole speed bins rather "
                        "than chunks within them. `none` is the unstratified control.")
    p.add_argument("--unconditioned", action="store_true",
                   help="FILTERED BC on a rollout that was ALREADY filtered on disk (e.g. "
                        "merge_selected.py's selected + successful chunks). Same plain "
                        "architecture as --filter_label -- no embedding, no token, no serve "
                        "arm, a drop-in generator every consumer loads unchanged -- but it "
                        "needs no label sidecar, because the selection happened upstream and "
                        "left none. WITHOUT this flag a label-free run takes the CONDITIONED "
                        "path instead: every window gets LABEL_NULL, the positive/negative "
                        "embeddings never see a gradient, and best_positive.pt is written "
                        "anyway -- a file naming an arm that was never trained.")
    p.add_argument("--drop_tile_holdout", action="store_true",
                   help="Train on training tiles only, keeping the held-out tiles clean for "
                        "evaluation. Requires --labels (the sidecar carries the split).")
    p.add_argument("--verify_identity", action=argparse.BooleanOptionalAction, default=True,
                   help="Assert the converted model matches its parent before training.")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default="robot_rl_flow_matching")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    ctx = online.load_fm_context(args.pretrained, device=args.device)
    cfg = resolved_config(args.pretrained)
    if cfg.advantage_cond:
        raise SystemExit(
            f"{args.pretrained} is ALREADY advantage-conditioned. An advantage-CONDITIONED "
            "arm fine-tunes from the pretrained checkpoint, never from the previous one: "
            "chaining would confound the conditioning mechanism under test with an "
            "unrepeatable optimisation path.\n"
            "This does NOT apply to a plain AWR arm, which is chained ON PURPOSE by "
            "`run_finetune_loop.sh` -- there the path IS the algorithm (AWR's convergence argument "
            "is about iterating, and re-fitting from the parent every time is what makes a "
            "single step advantage-weighted BC instead). Reproducibility there comes from the "
            "buffer manifest recording each iteration's parent checkpoint, not from refusing "
            "to chain.")

    import h5py
    with h5py.File(args.rollout, "r") as f:
        n_win = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        uid = f["episode_uid"][:n_win]
        ridx = f["replan_index"][:n_win]
    labels = None
    keep = np.ones(n_win, dtype=bool)
    if args.labels:
        side = dict(np.load(args.labels))
        labels = join_labels(side, uid, ridx)
        meta = json.loads(str(side["meta"])) if "meta" in side else {}
        n_hit = int((labels != LABEL_NULL).sum())
        print(f"[ft] labels: {n_hit}/{n_win} windows carry a non-null label "
              f"(n={meta.get('n')}, variant={meta.get('variant')}, "
              f"folds={meta.get('num_folds')})")
        if args.drop_tile_holdout:
            hold = join_labels({"label": side["val_tile"].astype(np.int8),
                                "episode_uid": side["episode_uid"],
                                "replan_index": side["replan_index"]}, uid, ridx)
            keep = hold != 1
            print(f"[ft] dropping {int((~keep).sum())} window(s) on held-out tiles")
    elif args.drop_tile_holdout:
        raise SystemExit("--drop_tile_holdout needs --labels: the tile split lives in the "
                         "sidecar, and guessing it would silently train on the holdout.")
    # PLAIN = leave the architecture exactly as the pretrained one. Two ways to ask for it,
    # because the filtering can happen in either of two places and only one of them leaves a
    # label sidecar behind:
    #   --filter_label  the rows are selected HERE, from a label file;
    #   --unconditioned the rows were already selected UPSTREAM (merge_selected.py's
    #                   selection + episode outcome), so no sidecar exists to point at.
    # Without this second door a label-free filtered-BC run silently took the CONDITIONED
    # path: every window got LABEL_NULL, the positive/negative embeddings received no
    # gradient at all, and `best_positive.pt` was still written -- a file naming a serve arm
    # that was never trained.
    awr = float(args.awr_beta) > 0.0
    plain = is_plain_arm(args)
    if args.unconditioned and args.filter_label:
        raise SystemExit("--unconditioned and --filter_label both ask for the plain "
                         "architecture; --filter_label already implies it. Pass one.")
    if awr and args.filter_label:
        raise SystemExit(
            "--awr_beta and --filter_label are two different answers to the same question. "
            "Filtering keeps the top percentile at weight 1 and drops the rest; AWR keeps "
            "every labelled row at exp(A/beta). Running both would filter first and then "
            "weight the survivors, which is neither baseline and is not what either number "
            "would be compared against. Pass one.")
    if awr and args.unconditioned:
        raise SystemExit("--awr_beta needs --labels for the advantage column, and "
                         "--unconditioned refuses --labels. Pass --awr_beta with --labels.")
    if awr and not args.labels:
        raise SystemExit("--awr_beta needs --labels: the weight is exp(A/beta) and the "
                         "continuous advantage lives in the sidecar.")
    if args.unconditioned and args.labels:
        raise SystemExit(
            "--unconditioned with --labels would load a label sidecar and then ignore it. If "
            "the labels should SELECT rows, that is --filter_label; if they should CONDITION "
            "the model, drop --unconditioned.")
    if args.filter_label:
        if labels is None:
            raise SystemExit("--filter_label needs --labels: there is nothing to filter on.")
        want = label_from_name(args.filter_label)
        keep &= (labels == want)
        print(f"[ft] FILTERED BC: keeping only '{args.filter_label}' windows -> "
              f"{int(keep.sum())} of {n_win}")
        # The label has done its whole job by selecting rows. Handing it to the dataset as
        # well would make this an advantage-CONDITIONED run on a filtered subset, which is a
        # different experiment and not the one this flag is for.
        labels = None

    adv_all = strat_all = awr_stats = None
    if awr:
        # AWR keeps every row the sidecar could LABEL and reweights it. `LABEL_NULL` is not
        # "advantage zero" -- it is a held-out tile (`--exclude_tile_holdout`) or a cell
        # whose spread was below `null_spread`, i.e. rows whose advantage is either
        # forbidden or noise. Dropping them is the same decision `--filter_label` makes by
        # keeping only positives, so the two arms see the same universe of rows.
        adv_all, hit = join_values(side, "advantage", uid, ridx)
        # Counted BEFORE narrowing `keep`, but over the rows `keep` already admits -- so the
        # filtered-BC comparison is against what that arm would keep from the SAME pool.
        # Taken over all `n_win` rows it silently included held-out-tile positives that
        # `--drop_tile_holdout` had already removed from this run.
        would_filter = int((keep & (labels == label_from_name("positive"))).sum())
        keep &= (labels != LABEL_NULL) & hit
        print(f"[ft] AWR: keeping every LABELLED window -> {int(keep.sum())} of {n_win} "
              f"(against {would_filter} the top-percentile filter would keep from the same "
              f"pool)")
        if args.awr_center == "stratum":
            if "stratum" not in side:
                raise SystemExit(
                    "--awr_center stratum needs a `stratum` column and this sidecar has "
                    f"none (it has {sorted(k for k in side)}). It was written before "
                    "label_advantage saved the cell it cut in, and `family_code` is NOT a "
                    "substitute -- the cut was family x progress x speed, so centering on "
                    "the family alone leaves the between-cell offset the threshold removed. "
                    "Re-run label_advantage.py with the same flags to regenerate it, or "
                    "pass --awr_center none for the unstratified control.")
            # PRESENCE IS NOT ENOUGH. `label_advantage` defaults to --progress_bins 1
            # --speed_bins 1, and with no crossing `group_ids(fam, None)` writes a densified
            # FAMILY CODE into `stratum` -- a perfectly valid column that centers on exactly
            # what the message above says is not a substitute. So the sidecar's own record of
            # what it crossed is what decides, and a family-only one is refused by name
            # rather than accepted as "stratified".
            n_prog = int(meta.get("progress_bins", 1) or 1)
            n_speed = int(meta.get("speed_bins", 1) or 1)
            if n_prog <= 1 and n_speed <= 1:
                raise SystemExit(
                    f"this sidecar's `stratum` is the FAMILY alone: it was written with "
                    f"progress_bins={n_prog} and speed_bins={n_speed}, so there is no "
                    "crossing to centre within and --awr_center stratum would report a "
                    "stratified run that removed no between-cell offset at all. Re-run "
                    "label_advantage.py with --progress_bins/--speed_bins (the filtered-BC "
                    "arm you are comparing against was cut that way), or pass "
                    "--awr_center none and read it as the unstratified arm.")
            print(f"[ft] AWR centering within the sidecar's own cells: "
                  f"family x {n_prog} progress x {n_speed} speed")
            strat_all, _ = join_values(side, "stratum", uid, ridx)
        # The label itself is NOT passed on: AWR steers the update through the weight, and
        # also conditioning on it would be a different experiment.
        labels = None

    # UNCONDITIONED for filtered BC: the architecture stays exactly the pretrained one, so
    # there is no embedding, no token, no `prefix_len` change and no serve arm -- the output
    # is a plain generator that every existing consumer loads without knowing this ran.
    model = (convert(ctx, cfg, device, args.advantage_inject) if not plain
             else copy.deepcopy(ctx.model).to(device))
    if plain:
        for prm in model.parameters():
            prm.requires_grad_(True)

    rows = np.flatnonzero(keep)
    if args.max_windows and rows.size > args.max_windows:
        print(f"[ft] --max_windows: using {args.max_windows} of {rows.size} windows")
        rows = rows[:args.max_windows]
    tr_rows, va_rows = _split_rows(uid, rows, args.val_frac, args.seed)

    weight = None
    if awr:
        # Estimated on the TRAIN rows and applied to all of them, so `beta` is not a
        # function of the validation split. Scattered back into a full-length, per-FILE-row
        # array because that is what the dataset indexes with; rows outside `rows` are never
        # read (the datasets only ever index `tr_rows` / `va_rows`) and hold 0.0 rather than
        # NaN so the dataset's own finiteness check stays a real check.
        if float(args.awr_clip) < 0:
            raise SystemExit(f"--awr_clip {args.awr_clip} is negative; 0 means 'no clip' and "
                             "a negative value would silently mean the same thing.")
        fit_mask = np.isin(rows, tr_rows)
        # `strata` is passed in BOTH arms: it sets the pooled within-cell sigma, so the
        # control differs from the treatment in the centering alone rather than also in the
        # temperature.
        strat_sel = strat_all[rows] if strat_all is not None else None
        if args.awr_balance == "cell" and args.awr_center == "stratum":
            # Not a warning about a mistake -- a statement that one of these two flags is
            # doing nothing. A per-cell mean shift is a per-cell CONSTANT factor on exp(),
            # and rescaling each cell's total divides that factor out: measured identical to
            # 4e-18. Left reachable so `--awr_center none --awr_balance none` is still the
            # raw arm, but a run should not report two controls when it applied one.
            print("[ft] note: --awr_balance cell makes --awr_center a no-op (a per-cell "
                  "mean shift is exactly cancelled by the per-cell rescale). The balance "
                  "is the binding control here.")
        w_max = float(args.awr_clip) if float(args.awr_clip) > 0 else None
        kw = dict(beta=float(args.awr_beta), strata=strat_sel, fit=fit_mask,
                  center=(args.awr_center == "stratum"),
                  balance=(args.awr_balance == "cell"))
        w_sel = stratified_awr_weights(adv_all[rows], w_max=w_max, **kw)
        # Counted against the level the clip actually settled at, over the UNCLIPPED weights.
        # Comparing against the post-clip mean over-reported by 1.3x-4.0x as beta sharpened,
        # because clipping lowers the very mean the threshold is stated against.
        if w_max is not None:
            w_raw = stratified_awr_weights(adv_all[rows], w_max=None, **kw)
            n_clip = int((w_raw > w_sel.max() * (1 + 1e-9)).sum())
        else:
            n_clip = 0
        if args.awr_normalize == "global":
            # `flow_matching_loss` takes a plain mean in this arm, so the weights must carry
            # a global mean of 1 or the loss scale -- and with it the effective lr -- moves
            # with beta. Scaled on the TRAIN rows, which are the ones it averages over.
            mu = float(w_sel[fit_mask].mean())
            w_sel = w_sel / mu if mu > 0 else w_sel
        weight = np.zeros(n_win, dtype=np.float64)
        weight[rows] = w_sel
        ess = effective_sample_size(w_sel)
        ess_frac = ess / max(w_sel.size, 1)
        print(f"[ft] AWR beta {args.awr_beta} (center={args.awr_center}, "
              f"balance={args.awr_balance}, "
              f"clip={args.awr_clip or 'none'}, normalize={args.awr_normalize}): "
              f"ESS {ess:,.0f} of {w_sel.size:,} rows ({ess_frac:.1%})")
        print(f"[ft]     weights: min {w_sel.min():.4g}  median {np.median(w_sel):.4g}  "
              f"max {w_sel.max():.4g}  ({n_clip} clipped)")
        awr_stats = {
            "beta": float(args.awr_beta), "clip": float(args.awr_clip),
            "center": args.awr_center, "normalize": args.awr_normalize,
            "balance": args.awr_balance, "realised_max_over_mean":
                float(w_sel.max() / w_sel.mean()),
            "n_rows": int(w_sel.size), "n_clipped": n_clip,
            "would_filter_keep": int(would_filter),
            "ess": float(ess), "ess_frac": float(ess_frac),
            "w_min": float(w_sel.min()), "w_median": float(np.median(w_sel)),
            "w_max": float(w_sel.max()),
            "progress_bins": int(meta.get("progress_bins", 1) or 1),
            "speed_bins": int(meta.get("speed_bins", 1) or 1),
        }
        if args.awr_normalize == "batch":
            # The printed ESS is a GLOBAL statistic, but the loss self-normalizes per batch,
            # so the update's realized ESS floors at roughly n/(batch/2) however small beta
            # goes -- measured 17 reported against 327 realized at beta 0.1, n=20k, B=128.
            floor = w_sel.size / max(args.batch_size / 2.0, 1.0)
            awr_stats["realized_ess_floor"] = float(floor)
            if ess < floor:
                print(f"[ft]     NOTE: under batch normalization the realized ESS cannot "
                      f"fall below ~{floor:,.0f} ({floor / w_sel.size:.1%}), so the "
                      f"{ess:,.0f} above UNDERSTATES this run. Sharpening beta further "
                      f"moves the printed number and not the update; --awr_normalize global "
                      f"is the arm that actually reaches filtered BC.")
        # The number that says whether this is still AWR. As ESS falls the update is carried
        # by fewer and fewer rows, and in the limit AWR IS filtered BC with a softer edge and
        # a smaller sample -- the comparison the beta sweep exists to make, so it is reported
        # rather than guarded.
        if ess / max(w_sel.size, 1) < 0.05:
            print("[ft]     !! under 5% ESS -- at this beta the update is carried by a tiny "
                  "fraction of rows, which is filtered BC with extra variance rather than a "
                  "distinct arm. Raise --awr_beta or read the result against that.")

    common = dict(ctx=ctx, label=labels, seed=args.seed)
    train_ds = RolloutWindowDataset(args.rollout, tr_rows, null_prob=args.null_prob,
                                    weight=weight, **common)
    val_ds = (RolloutWindowDataset(args.rollout, va_rows, null_prob=0.0,
                                   weight=weight, **common)
              if va_rows.size else None)
    print(f"[ft] {len(train_ds)} train / {0 if val_ds is None else len(val_ds)} val windows "
          f"over {int(np.unique(uid[rows]).size)} episodes")
    if labels is not None:
        print(f"[ft] train label counts (before null dropout): {train_ds.label_counts()}")

    if plain:
        arm = "AWR" if awr else "filtered BC"
        print(f"[ft] {arm}: no conversion to verify -- the model IS the pretrained one.")
    elif args.verify_identity and args.advantage_inject == "time_embed":
        worst = verify_identity(ctx, model, train_ds, device)
        print(f"[ft] conversion is identical to the parent under all {len(LABEL_NAMES)} "
              f"labels (max |delta| {worst:.2e})")
    elif args.verify_identity:
        # A TOKEN is a key in the attention softmax, so its presence changes every other
        # token's output at ANY initialisation -- there is no inert conversion to assert, and
        # claiming one would be false. What replaces it is measuring how far the untrained
        # conversion already sits from its parent, so a later change is read against that
        # baseline rather than against an assumed zero.
        with torch.no_grad():
            items = [train_ds[i] for i in range(min(4, len(train_ds)))]
            hist = torch.stack([it["history"] for it in items]).to(device)
            ext = torch.stack([it["extero"] for it in items]).to(device)
            cmd = (torch.stack([it["cmd_tokens"] for it in items]).to(device)
                   if "cmd_tokens" in items[0] else None)
            nullm = torch.stack([it["history_null"] for it in items]).to(device)
            noise = torch.randn(len(items), ctx.F, ctx.D,
                                generator=torch.Generator().manual_seed(0)).to(device)
            base = sample_trajectory(ctx.model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                                     history_null_mask=nullm, noise=noise)
            adv = torch.full((len(items),), 1, dtype=torch.long, device=device)
            got = sample_trajectory(model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                                    history_null_mask=nullm, noise=noise, advantage=adv)
            d = float((got - base).abs().mean())
        print(f"[ft] prefix_token: conversion is NOT inert by construction (a token changes "
              f"the attention softmax at any init). Untrained |delta| from parent: {d:.4f}")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=_collate, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = (torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=_collate) if val_ds is not None else None)

    if args.emb_lr_multiplier != 1.0:
        emb_params = list(model.advantage_emb.parameters())
        emb_ids = {id(p) for p in emb_params}
        rest = [p for p in model.parameters() if id(p) not in emb_ids]
        opt = torch.optim.AdamW(
            [{"params": rest, "lr": args.lr},
             {"params": emb_params, "lr": args.lr * args.emb_lr_multiplier}],
            weight_decay=args.weight_decay)
        print(f"[ft] advantage_emb gets its own param group: lr={args.lr * args.emb_lr_multiplier:.2e} "
              f"({args.emb_lr_multiplier}x the base {args.lr:.2e}); everything else at {args.lr:.2e}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = WeightEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # Epoch 0 = the untouched model, recorded whenever ANY frozen-batch metric is attached.
    # Every one of them is read as a MOVE from this row -- the pair gap most of all, whose
    # whole claim is that it starts at ~0 because the candidates are exchangeable draws from
    # one conditional. Built here rather than inside a single cache's branch, which is what
    # left a pair-only run with no baseline at all.
    history_base: dict = {"epoch": 0}
    pairs = None
    if args.pair_cache:
        from ft_canary import PairGapCanary  # noqa: PLC0415

        pairs = PairGapCanary(args.pair_cache, device=args.device)
        pbase = pairs.score(model)
        history_base.update(pbase)
        print("[ft] pair-gap baseline (pretrained weights) -- expect gap ~0 and win_rate "
              "~0.50; the candidates are exchangeable draws from ONE conditional:")
        for kk in sorted(pbase):
            print(f"       {kk} = {pbase[kk]:+.5f}")
    canary = None
    if args.canary_cache:
        from ft_canary import Canary  # noqa: PLC0415

        canary = Canary(args.canary_cache, device=args.device)
        # Epoch 0 is the BASELINE and is not optional: `pretrain_val_flow_mse` here is the
        # pretrained model's own number, so every later epoch is read as a move from it
        # rather than as a bare value, and `drift_l2` must be ~0 (same weights, same cached
        # sampling noise) -- if it is not, the reference is not a fixed point and the whole
        # metric is measuring something else.
        base = canary.score(model)
        history_base.update(base)
        print(f"[ft] canary baseline (pretrained weights): "
              f"pretrain_val_flow_mse={base['pretrain_val_flow_mse']:.6f}  "
              f"drift_l2={base['drift_l2']:.3e}")
        if base["drift_l2"] > 1e-4:
            raise SystemExit(
                f"the canary's drift reference is not a fixed point: the UNMODIFIED model "
                f"differs from its own cached plans by {base['drift_l2']:.3e}. Every later "
                "drift_l2 would be that noise plus the real signal, indistinguishably. "
                "Rebuild the cache against THIS checkpoint (ft_canary.py --build).")

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project,
                            name=os.path.basename(os.path.abspath(args.out)),
                            config={**vars(args), "n_train": len(train_ds)})
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb logging is OFF: {type(exc).__name__}: {exc}")

    def _lr_at(epoch: int, step: int) -> float:
        """Peak LR shaped by the cosine (per epoch) and the warmup (per step).

        Closed form in ``(epoch, step)`` rather than a stateful scheduler, because the two
        act at different granularities and ``CosineAnnealingLR.get_lr`` is recursive in
        ``param_group["lr"]`` -- mutating the group mid-epoch for the warmup would corrupt
        the cosine's own state rather than scale it. Same reasoning, and same arithmetic, as
        ``train.scheduled_lr``.
        """
        lr = args.lr
        if args.cosine and args.epochs > 1:
            lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            lr *= (step + 1) / args.lr_warmup_steps
        return lr

    def _run(batch, train: bool) -> dict:
        """One forward (and optionally backward) pass over a collated batch."""
        x1 = batch["state"].to(device)
        out = flow_matching_loss(
            model, x1, batch["history"].to(device), batch["extero"].to(device),
            cmd_tokens=(batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None),
            history_null_mask=batch["history_null"].to(device),
            advantage=(batch["advantage"].to(device) if "advantage" in batch else None),
            # TRAIN ONLY. `flow_mse` is unweighted either way, but weighting the val pass
            # would make `loss` there a weighted quantity while the number actually used for
            # checkpoint selection is not -- two different things under one name.
            sample_weight=((batch["weight"].to(device)
                            if "weight" in batch else None) if train else None),
            weight_reduction=args.awr_normalize,
            # No anchor mask and no auxiliary terms: the target is the model's OWN chunk, so
            # there is no clip to be off-axis from and the consistency terms would be
            # regressing the model onto its own already-consistent output.
            blocks=ctx.reference_blocks,
        )
        if train:
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
        return {k: float(v.detach()) for k, v in out.items()}

    history, best_val, gstep = [], float("inf"), 0
    if len(history_base) > 1:
        history.append(history_base)

    for epoch in range(args.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for batch in loader:
            if args.cosine or args.lr_warmup_steps > 0:
                lr_now = _lr_at(epoch, gstep)
                # The embedding's group keeps its multiplier, so a schedule scales BOTH
                # groups rather than silently collapsing them onto one rate.
                for gi, g in enumerate(opt.param_groups):
                    mult = args.emb_lr_multiplier if (gi == 1 and len(opt.param_groups) > 1) else 1.0
                    g["lr"] = lr_now * mult
            gstep += 1
            m = _run(batch, True)
            tot += m["flow_mse"] * batch["state"].shape[0]
            seen += batch["state"].shape[0]
        train_mse = tot / max(seen, 1)
        rec = {"epoch": epoch + 1, "train_flow_mse": train_mse,
               "lr": _lr_at(epoch, gstep), "secs": round(time.time() - t0, 1)}
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vt, vn = 0.0, 0
                for batch in val_loader:
                    vt += _run(batch, False)["flow_mse"] * batch["state"].shape[0]
                    vn += batch["state"].shape[0]
            rec["val_flow_mse"] = vt / max(vn, 1)
        if pairs is not None:
            rec.update(pairs.score(model))
        if canary is not None:
            # Raw AND EMA. At ~120 steps/epoch an EMA decay of 0.999 has a ~1000-step
            # horizon, comparable to the whole run, so the EMA curve lags the raw one and
            # reading only the EMA can hide a real early peak. Logging both makes that lag
            # visible instead of something to be inferred.
            rec.update(canary.score(model))
            if ema is not None:
                with ema.swapped_in(model):
                    rec.update({f"ema_{k}": v for k, v in canary.score(model).items()})
        history.append(rec)
        print(f"[ft] epoch {epoch + 1}/{args.epochs}  train {train_mse:.5f}"
              + (f"  val {rec['val_flow_mse']:.5f}" if "val_flow_mse" in rec else "")
              + (f"  pretrain_val {rec['pretrain_val_flow_mse']:.5f}"
                 f"  drift {rec['drift_l2']:.4f}" if "drift_l2" in rec else "")
              + (f"  gap[tr] {rec['pair/train/gap']:+.4f}"
                 f"  win {rec['pair/train/win_rate']:.2f}"
                 if "pair/train/gap" in rec else "")
              + f"  ({rec['secs']}s)")
        if wb is not None:
            wb.log(rec)
        _save(args, cfg, model, opt, ema, epoch + 1, history,
              os.path.join(args.out, "last.pt"))
        # KEEP THE INTERIOR EPOCHS. `last.pt` is overwritten every epoch and `best.pt` tracks
        # `val_flow_mse`, which on this narrow set declines about monotonically -- so between
        # them they preserve roughly one point on the curve, the end. The quantity that
        # actually decides this fine-tune is a MuJoCo eval, and it peaks and decays where no
        # training loss says it does; without these files, acting on that reading means
        # retraining the arm to recover a checkpoint that already existed.
        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            _save(args, cfg, model, opt, ema, epoch + 1, history,
                  os.path.join(args.out, f"ckpt_{epoch + 1}.pt"))
        # `best.pt` is what every downstream consumer reaches for by convention, so it has to
        # BE the best rather than the last -- a file named for a selection nothing performed
        # is the kind of quiet claim this stack has been bitten by. With no val split there
        # is no selection to make and the last epoch is the honest answer.
        val = rec.get("val_flow_mse")
        if val is None or val < best_val:
            best_val = float("inf") if val is None else val
            _save(args, cfg, model, opt, ema, epoch + 1, history,
                  os.path.join(args.out, "best.pt"))
            if val is not None:
                rec["is_best"] = True
    # SERVABLE copies, one per arm. `best.pt` names no arm, deliberately -- a conditioned
    # checkpoint that does not state one is refused at load, so nothing (an Isaac eval, a
    # probe, a play run) can serve an arm by accident. These two say which they are, in the
    # artifact, exactly as `fm_meta.advantage_label` does on the ONNX side.
    if not plain:
        for name in ("positive", "negative"):
            stamp_serve_label(os.path.join(args.out, "best.pt"),
                              os.path.join(args.out, f"best_{name}.pt"), name)
            print(f"[ft] wrote {args.out}/best_{name}.pt")
    with open(os.path.join(args.out, "finetune.json"), "w") as f:
        # `awr` is recorded because the module docstring's own claim is that ESS "is not
        # inferable from beta alone" -- and `label_counts()` is empty under AWR, since the
        # label selects nothing there. Without this the artifact states the temperature and
        # nothing about what it did.
        json.dump({"args": vars(args), "history": history,
                   "label_counts": train_ds.label_counts(),
                   "awr": awr_stats}, f, indent=2)
    if wb is not None:
        wb.finish()
    print(f"[ft] wrote {args.out}/best.pt")
    if plain:
        why = f"{args.filter_label}-filtered BC" if args.filter_label else "filtered BC"
        print(f"[ft] NEXT: this is a PLAIN generator ({why}), so it "
              "evaluates as ONE cell against the pretrained base -- there is no arm to\n"
              "     control against, and therefore no internal check that the training did "
              "anything. The eval against `base` IS the check.")
        if wb is not None:
            wb.finish()
        return 0
    print("[ft] NEXT, before any sweep: run BOTH arms and compare them.\n"
          f"     Isaac:  --fm_checkpoint pos={args.out}/best_positive.pt "
          f"neg={args.out}/best_negative.pt\n"
          f"     ONNX :  export_fm_onnx.py --checkpoint {args.out}/best.pt "
          "--advantage_label positive|negative\n"
          "     If the two perform the same, the model ignored the label and every other "
          "number here is noise.")
    return 0


def stamp_serve_label(src: str, dst: str, label: str) -> None:
    """Copy a conditioned checkpoint, naming the arm it serves.

    The label is not recoverable from the weights -- one parameter set answers to all three --
    so it has to travel in the artifact, the same way ``attention_mode`` and ``extero_clip``
    already do and for the same reason. A consumer then reads which arm it holds instead of
    being told at the call site, where it could be told wrongly.

    Args:
        src: The unlabelled checkpoint.
        dst: Where to write the labelled copy.
        label: One of :data:`~robot_rl.flow_matching.advantage.LABEL_NAMES`.

    Raises:
        ValueError: On an unrecognised label, or a checkpoint with no label embedding.
    """
    label_from_name(label)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not ck["config"].get("advantage_cond"):
        raise ValueError(f"{src} is not advantage-conditioned; there is no arm to name.")
    ck["config"]["advantage_serve_label"] = label
    torch.save(ck, dst)


def _save(args, cfg, model, opt, ema, epoch: int, history: list, path: str) -> None:
    """Write a checkpoint in the pretrained file's own format.

    Everything except the model weights, the config and the provenance is copied from the
    parent -- the normalizers above all, which were NOT refitted and must travel unchanged
    or nothing downstream (the ONNX export, the deploy contract checks) will agree with the
    generator this came from.
    """
    parent = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(cfg)
    # The plain arms produce a PLAIN generator: claiming advantage_cond would make
    # `load_fm_context` demand a serve arm the weights cannot honour. `is_plain_arm` is the
    # SINGLE definition of that rule, shared with `main` -- restating it here is what let an
    # AWR checkpoint be stamped as conditioned over plain weights.
    plain = is_plain_arm(args)
    cfg.advantage_cond = not plain
    if not plain:
        cfg.advantage_inject = args.advantage_inject
    # THE HISTORY ARM THE ROLLOUTS RAN, not the one the parent was fitted on. A fine-tune
    # onto a new arm is exactly the case where those differ: the rollouts were recorded
    # under `measured` from a `reference` parent, and after training on them the model's own
    # contract is `measured`. Inheriting the parent's value would leave a measured-tuned
    # checkpoint claiming `reference`, which both runtimes would then happily serve the
    # WRONG arm -- identical row widths, no error, just wrong. Read from the recording so
    # the artifact cannot disagree with its own training set.
    _arm = _rollout_history_from(args.rollout)
    if _arm and _arm != cfg.history_source:
        print(f"[ft] history_source: parent {cfg.history_source!r} -> {_arm!r}, the arm "
              f"{os.path.basename(args.rollout)} was recorded under. The fine-tuned "
              "checkpoint's contract is the arm it was TRAINED on.")
        cfg.history_source = _arm
    ckpt = dict(parent)
    ckpt["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt["config"] = cfg.to_dict()
    ckpt["optimizer"] = opt.state_dict()
    ckpt["epoch"] = epoch
    ckpt["finetune"] = {"pretrained": os.path.abspath(args.pretrained),
                        "rollout": os.path.abspath(args.rollout),
                        "labels": os.path.abspath(args.labels) if args.labels else "",
                        "lr": args.lr, "epochs": args.epochs, "history": history}
    if ema is not None:
        ckpt["model_ema"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
        ckpt["ema_state"] = {k: v for k, v in ema.state_dict().items() if k != "shadow"}
    else:
        # An EMA inherited from the PARENT would be the pretrained weights wearing this
        # run's name, and `load_fm_context` prefers `model_ema` -- so the fine-tune would
        # silently evaluate as the model it started from.
        ckpt.pop("model_ema", None)
        ckpt.pop("ema_state", None)
    torch.save(ckpt, path)


if __name__ == "__main__":
    raise SystemExit(main())
