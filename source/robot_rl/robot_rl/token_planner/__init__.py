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

**The model modules import no IsaacLab AT MODULE SCOPE** -- ``fsq``, ``tokenizer``,
``prefix`` (whose one import is the flow-matching extero encoder, itself Isaac-free) and the
metrics take their layout lazily, inside the one function that needs it. That is a
maintenance property, not a deployability one: importing this PACKAGE still goes through
``robot_rl/__init__.py``, which registers every gym env and pulls in ~60 isaaclab modules, so
nothing here runs in ``env_transfer`` (which ships no torch at all). The code that builds a
dataset out of a recording lives in the training scripts, which import the RL side outright.
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
