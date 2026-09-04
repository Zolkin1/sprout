# Copyright 2026 Zachary Olkin. All rights reserved.

"""The tokenized autoregressive trajectory planner -- the discrete alternative to the FM.

Two stages, trained in order and shipped together:

1. :mod:`~robot_rl.token_planner.tokenizer` -- a chunk-local autoencoder with an
   :mod:`~robot_rl.token_planner.fsq` bottleneck, mapping one plan of ``H`` reference-window
   nodes to ``N = H / D`` discrete tokens and back. Frozen before stage 2 starts.
2. :mod:`~robot_rl.token_planner.planner` -- a prefix-LM over those tokens. Its
   conditioning is the flow-matching generator's, reused rather than reinvented
   (:mod:`~robot_rl.token_planner.prefix`, held bit-identical to it by test), so a
   difference in closed-loop behaviour is attributable to the output representation.

The point of the discrete output is what it makes possible AFTER supervised training: a
categorical policy has an exact per-token log-probability and an exact entropy, so PPO
fine-tuning needs no reparameterisation trick and no surrogate for the sampling density --
the two things that make the flow-matching generator awkward to fine-tune with RL.

**Nothing here imports IsaacLab.** The models are torch-only so their tests run in
``env_transfer``; the code that builds a dataset out of a recording lives in the training
scripts, which do import the RL side.
"""

from .config import DOWNSAMPLE_D, FSQ_LEVELS, PlannerConfig, TokenizerConfig
from .fsq import FSQ, codes_to_id, id_to_codes
from .planner import TokenPlanner, causal_output_mask
from .prefix import ConditioningPrefix
from .tokenizer import TrajectoryDecoder, TrajectoryEncoder, TrajectoryTokenizer

__all__ = [
    "DOWNSAMPLE_D",
    "FSQ",
    "FSQ_LEVELS",
    "ConditioningPrefix",
    "PlannerConfig",
    "TokenPlanner",
    "TokenizerConfig",
    "TrajectoryDecoder",
    "TrajectoryEncoder",
    "TrajectoryTokenizer",
    "causal_output_mask",
    "codes_to_id",
    "id_to_codes",
]
